# Writeup — Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA

> Your own explanation of the paper, as if teaching it to a peer who hasn't
> read it. This is **not** a summary of the abstract — it's my synthesis
> after reading and implementing it.

**Paper:** "Macaron-V1: Towards Open Continual Learning with Self-Improvement
and Mixture-of-LoRA" — Mind Lab, 2026, arXiv:2608.09819.

## The one-paragraph version

Macaron-V1 is a system report (not a classic methods paper) that bets on one
big idea: **freeze the base model forever, add every capability as a separate
LoRA adapter, route each user turn to exactly one adapter, and treat the
harness (tools, prompts, runtime config) as a first-class, versioned training
target that the model itself can rewrite in language space.** The trainable
object is not the weights alone but the *model–harness pair*, improved by a
recursive discover → expand → update loop. The flagship "Venti" is a frozen
744B GLM-5.2 plus four ~7.7B-value LoRAs (chat, agent, coding, GenUI), with
the chat adapter doubling as the router via constrained decoding.

## The problem

Post-training today is centralized and static: you tune a model against a
snapshot of tasks, ship a checkpoint, and it never improves from production
experience. Two failure modes follow:

1. **Cross-task interference.** Joint post-training over heterogeneous tasks
   (chat + tool-use + coding + UI generation) makes differently-shaped
   chain-of-thought compete for shared parameters. (The paper is honest that
   the budget-matched single-LoRA comparison is *not* in this release —
   interference is a design motivation, not a measured finding.)
2. **The world moves on.** New tools, knowledge, and user needs arrive after
   release; a frozen checkpoint approximates a static optimum.

The stated goal is "experiential intelligence": **adaptation** (recursive
improvement of versioned model–harness pairs) and **collaboration**
(composition of separately-trained specialists).

## The idea

Three coupled moves:

1. **Mixture-of-LoRA as orchestration, not merging.** Adapters never merge or
   stack; composition happens at request level via routing. Each answer is
   attributable to exactly one specialist, and a new capability is a new
   (A, B) pair on an untouched base.
2. **L0 is the router.** No separate router model, no keyword rules: the chat
   adapter classifies each turn into one of four canonical labels under a
   24-token decode budget with a constrained-decoding grammar (only the 4
   legal labels are even decodable). The request is framed as quoted
   *untrusted text* — a neat prompt-injection defense.
3. **The harness is trainable in language space.** HCP, a versioned TOML
   contract (tools, prompts, skills, hooks, env), carries no gradients — but
   the model can *rewrite it*, propose edits, re-run the affected slice, and
   ship survivors as the next config. The killer evidence: on 122 tasks the
   frozen base fails *all* of, adaptive config search alone (no weight
   changes!) lifts coverage from 2/122 to **122/122**. Failure under a
   baseline config does not mean the capability is absent.

## How it works (the intuition)

The serving loop per user turn is **Route → Answer → Summary**:

- **Route**: L0 decodes one label (constrained grammar, ~0.5 s).
- **Answer**: the chosen specialist answers from its **own-view** — its own
  past turns *verbatim* (full tool traces), every other specialist's turns
  collapsed to a ≤192-token summary.
- **Summary**: the answering specialist emits a ≤192-token summary, stored
  server-side only (never shown to the client).

The own-view is the quietly brilliant part. Because each adapter's context is
*deterministically rebuilt* from an append-only timeline — own turns
byte-identical, foreign turns always summarized — re-entering a LoRA yields a
byte-identical prefix extension of its previous visit. The engine's native
prefix cache just hits. **Per-adapter KV reuse is an emergent property of a
stable prompt format, not an engine feature.**

Around the model sit: a stateful **REPL agent harness** (dependent values
persist as variables — the case study does in 6 turns what takes 48
one-call-per-turn; self-derived tools are promoted only after private
validation), **UI4A** (agent writes ordinary frontend code inside
runtime-enforced boundaries; every user gesture is a structured Action with a
NoAI visibility field), **MindForge** (the RSI control plane: problem banks →
evaluated trajectories → datasets → next model + next HCP, all lineage-linked),
**MinT** (adapter lifecycle: adapter-only handoff 18.3× faster than merged
checkpoints), and **LongStraw** (response-only long-context training: capture
the prompt's forward state once without autograd, replay one response at a
time with autograd — peak memory holds only the longest single response's
graph).

## What I learned by implementing it

(From building the toy re-implementation in `implementation/` — frozen small
base, 4 LoRA slots, real Proxy loop, all on CPU.)

1. **Cross-task interference is real and visible even at toy scale.** The
   paper's own missing ablation — budget-matched single fat LoRA vs MoL — was
   the first thing I ran. MoL mean 0.750 vs fat 0.494, and the failure mode is
   exactly the paper's motivation: the fat adapter collapses the GUI
   specialist to 0.00 while every dedicated slot keeps its skill at 1.00.
   Nothing in the paper *measures* this; at toy scale it's blatant.
2. **KV prefix reuse is a formatting discipline, not an engine feature.** By
   rendering each own turn byte-identically to the request that produced it,
   consecutive own-views nest perfectly (prefix stability 1.000; ~58% of each
   request's context is a reusable prefix in 6-turn mixed conversations). The
   "trick" costs you nothing at training time — it's purely how the Proxy
   rebuilds message lists.
3. **Routing via constrained decoding is trivially reliable.** 1.000 routing
   accuracy (paper: 0.9912) when the label space is 4 and the grammar is
   hard. The hard-constrained decode makes the router a formatting problem,
   not a classification problem.
4. **Sequential training catastrophically forgets — joint mixture training
   fixes it.** Training L0's chat, then its router duty, destroyed the earlier
   skill (generation degenerated to gibberish). Mixing duties into one joint
   SFT distribution recovered both. This is the paper's clustering rule
   ("cluster tasks that share thinking patterns into one LoRA") showing up as
   a hard constraint, not a preference.
5. **The 192-token summary question stays open at toy scale.** Summary-handoff
   vs full-history showed no measurable gap (0.333 vs 0.333) — at this scale
   the extractive summary preserves the load-bearing tokens and the *echo
   format* is the shakier link. The paper's open question is genuinely open.

## What surprised me / was harder than expected

- **The honest accounting surprised me (positively).** The report repeatedly
  scopes its own claims: routing accuracy measured on training data is called
  a diagnostic, not generalization; the multimodal-retention table is labeled
  "not established preservation"; the 122/122 coverage result is explicitly a
  coverage ceiling, not held-out generalization. System reports are usually
  less disciplined.
- **Char-level multi-digit arithmetic was not learnable** by a 4-layer char
  model with rank-8 LoRA in my budget — L1's exact-match stayed at 0.00 (SFT
  loss plateaus ≈0.13: format learned, digits unreliable). I report it as-is;
  the interesting part is the fat-LoRA arm fails the same way (0.05), so the
  comparison stays fair.
- **How little of the paper is weights at all.** Reading it once, you think
  "4 LoRAs on a frozen giant." Reading it twice, you realize the load-bearing
  artifacts are the timeline format, the HCP contract, and the promotion
  gates — things that never receive a gradient.

## References

- Paper: https://arxiv.org/abs/2608.09819
- My implementation: `implementation/` (run `python3 train.py`, ~6.5 min CPU;
  actual output in `implementation/run_output.txt`)
- Breakdown: `breakdown.md`
- Reading notes: `notes.md`
