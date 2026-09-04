# Macaron-V1 — Reading Notes

> **Paper:** "Macaron-V1: Towards Open Continual Learning with Self-Improvement
> and Mixture-of-LoRA"
> **Authors:** Mind Lab (large collective-author report, ~70 listed authors)
> **arXiv:** 2608.09819 (v2, 24 Aug 2026) · cs.LG
> **Models:** https://huggingface.co/collections/mindlab-research/macaron-v1
> **Harness:** https://github.com/MindLab-Research/Mixture-of-LoRA-Harness

---

## Pass 1 — first impressions (skim)

This is a *system report*, not a classic methods paper — closer in genre to a
company technical report (think "LLaMA report" or "Claude model card with a
research agenda"). The contribution is an entire co-designed stack for
"experiential intelligence": agents that keep learning after deployment.

## The problem, in my own words

Post-training today is *centralized and static*: you tune a model against a
snapshot of environments/tasks, ship a checkpoint, and it never improves from
what happens in production. Two failure modes:

1. **Joint post-training over heterogeneous tasks** (chat + tool-use + coding +
   UI generation) causes cross-task interference — the tasks think in
   differently-shaped chain-of-thought, and shared parameters compete.
   (Note: the paper is honest that the budget-matched single-LoRA comparison
   is *not* in this release — interference is a design motivation, not a
   measured finding.)
2. **The world moves on** after release: new knowledge, new tools, new user
   needs arrive; a frozen checkpoint approximates a *static optimum*.

Their target: "experiential intelligence" = learning from experience in a real
environment, continuing to learn after deployment. Operationalized as two
system goals:

- **Adaptation** — recursive improvement of *versioned model–harness pairs*:
  experience from one configuration is evaluated under an external contract
  and used to construct its successor.
- **Collaboration** — composition of separately-trained specialists
  (Mixture-of-LoRA), with "collective intelligence" (composed system beats its
  strongest constituent) as an explicitly open stretch goal.

---

## Pass 2 — deep reading, section by section

### §2 Mixture of LoRA (MoL) — the architectural spine

**§2.1 Design principle.** One clustering rule: *cluster tasks that share
skills/thinking patterns into one LoRA; keep divergent skills in separate
LoRAs.* Two properties: base stays frozen (new capability = new adapter, base
never drifts), adapters are portable (shared base ⇒ specialist from team A can
compose with team B's on the same runtime). MoL is explicitly "a bet on
orchestration over merging" (§2.8): adapters are never merged or stacked
additively; composition happens at *request level* via routing.

**§2.2 Specialists.** Venti = frozen 744B GLM-5.2 + four LoRAs:
- **L0 Chat** — conversation backbone, identity, *and routing entry point*.
- **L1 Agent** — long-horizon tool use; absorbed Preview's OpenClaw adapter.
- **L2 Coding** — code/SWE/terminal.
- **L3 GenUI** — UI4A rendering, TSX (React/SolidJS).

Config details: Venti adapters rank 16, α=32, same target modules for L0–L3,
7,688,042,496 stored values each (≈30.8B total on top of the 744B base ⇒
~774.8B logical; release label "748B"). Tall = Qwen3.6-35B-A3B base, rank 64,
α=128, 3,775,651,840 stored values per adapter (L2 stored in F32) ⇒ ≈50.1B.
Important accounting subtlety: *stored tensor count ≠ active-per-token count ≠
device memory*. The "1B LoRA" release label is logical stored values.

**§2.3 Routing loop — Route → Answer → Summary (per user turn).**
1. **Route**: L0 classifies the request into exactly one canonical label
   L0–L3, 24-token decode budget, *constrained-decoding grammar* restricts
   output to the 4 legal labels. Prompt frames the request as quoted
   *untrusted text* (prompt-injection defense); wrapper families checked in
   priority order: GenUI → code/terminal → personal-agent. **No separate
   router model, no keyword rule library — L0 is the router.**
2. **Answer**: chosen specialist answers from *its own conversation view*
   (§2.4), seeded with cross-adapter summaries.
3. **Summary**: specialist emits ≤192-token summary; Proxy stores it
   server-side only (never returned to client); becomes shared context any
   adapter can inherit.

Short-circuits: **tool-result stickiness** (turn after a tool call is locked to
the same adapter, no routing/summary) and **transactional rollback**
(checkpoint before each turn; restore on engine failure/disconnect so an
undelivered turn never enters history).

Measured costs (48 multi-turn requests, temp 0): Venti route 0.54 s (12%),
answer 3.17 s (68%), summary 0.97 s (20%), total 4.68 s. Tall: 0.20/1.24/0.32,
total 1.76 s. Overhead share stable (~30–32%) across base sizes.

Routing accuracy: 6,448-sample trace drawn from LoRA *training data* (so it's
an implementation diagnostic, not generalization): Venti 6391/6448 = **99.12%**
with 100% canonical-label compliance, zero parse errors; per-class 97.1% (L1,
hardest — semantically closest to L0) to 100% (L2). Tall: 99.04%. Balanced
480-sample integration subset with an earlier prompt: 97.5%. Multi-hop
diagnostic: 8-turn L1/L2 alternating trace ×3 conversations = 24/24 correct
turns, incl. 21/21 domain switches and 18/18 specialist re-entries.

Post-routing quality (Vita delivery, 5 seeds/arm, unpaired): Venti direct L1
0.636±0.026, routed KV-off 0.650±0.030, routed KV-on 0.632±0.019 — no detected
degradation, but "does not establish equivalence."

**§2.4 Per-adapter conversation views (the "own-view") — the key trick.**
The Proxy keeps an **append-only conversation timeline** and, for each hop,
*deterministically rebuilds* the message list the target LoRA sees:
- the specialist's own past turns **verbatim** (full trace, tool calls,
  tool results),
- every *other* specialist's turns collapsed to one assistant message carrying
  that turn's 192-token summary,
- current user turn verbatim.

Consequence: re-entering a LoRA on a later turn yields a **byte-identical
prefix** to its previous visit ⇒ the engine's native LoRA-aware prefix cache
hits and only the new tail is prefilled. *Per-adapter KV reuse is an emergent
property of stable per-adapter prompts — no engine modification needed.* The
summary scaffold is never fed into any own-view, so it can't pollute a
reusable prefix. Two state models behind one interface: Proxy-authoritative
timeline (stateful Responses API) vs. stateless side-context rebuilt from
resent history (Chat Completions). Routing is query-only and identical on both.

**§2.5 KV cache reuse — two layers.**
- **Layer A (production, no patch)**: emergent reuse via stable own-views (above).
- **Layer B (experimental "route-decode overlay")**: within one request, after
  L0 prefills the router prompt, trim router-only tokens and continue decoding
  under the selected LoRA, reusing the shared prefix. vLLM: streaming-input
  session path (two segments + metadata pinning shared prefix length). SGLang:
  runtime monkey-patch of scheduler/serving classes. A cross-turn flag lets
  the decode segment read/write the global prefix cache.
- **Boundary**: only *continuous* prefixes are reused; no arbitrary
  non-contiguous GPU KV splicing across adapters. A specialist switch still
  invalidates the adapter-specific KV portion.
- Quality: Layer A three-arm Vita sweep shows no detected difference. Layer B
  only a 30-turn GLM-5.1 shim functionality check (both arms 100%; reuse-on
  slightly *slower* per-turn due to prefix-trim overhead).

**§2.6 Deployment study (systems question: MoL vs replicated-base).**
MoL stores 1 base + 30.8B adapter values ≈ 774.8B logical vs 4 merged base
copies = 2.976T for replicated layout ⇒ MoL ≈ 26% of replicated, 74%
reduction. Capacity freed goes to KV cache/concurrency: H20 sustains 16×56K,
8×180K, or 4×230K-token requests; B300 DCP2/4/8 + EAGLE ≈ 2.34M/4.67M/9.34M
logical KV tokens. Long-context: CP8 LayerSplit cuts 900K-token cold
needle-test TTFT 107.1 s → 49.2 s; DCP8+EAGLE reaches 8.6 ms TPOT / 110 tok/s
@ concurrency 1, 18.0 ms / 757 tok/s @ 16. Correctness envelope: FlashMLA
sparse attention clean on 48/48 long-context configs for both GLM-5.1 and
GLM-5.2; default DSA decode path clean only 6/48 on GLM-5.1; sharded DCP
showed systematic corruption (replicated DCP passed). Claim is carefully
scoped: MoL removes replicated base-weight residency; **no** comparative
latency/throughput claim vs independently deployed merged specialists.

**§2.7 Continual learning & collective intelligence (affordances, not evidence).**
Three properties: adapter registration (new LoRA without touching deployed
weights), base-weight immutability (gradients only enter adapters), live
harness updates (routing rules/tool exposure/HCP editable without redeploying
weights — new tasks handled by harness change *first*, baked into an adapter
only once trajectory data justifies training). ⇒ three release clocks: base,
specialists, harness, each on its own cadence. Future: multi-team composition
and user-personalized adapters (MinT's million-entry catalog as addressing
mechanism) — *not evaluated in this release*.

**§2.8 Discussion.** Orchestration over merging; logs attribute each answer
to exactly one specialist; adapter-specific updates leave other adapters'
weights untouched (though routing/harness changes can still shift end-to-end
behavior). Open: sharing L0's KV across all specialists (sketched, not
shipped); single-input multi-intent turns (currently routed whole to one
specialist; orchestrator/decomposer is an exploratory branch, not in
production).

### §3 Model–Harness Co-design + Recursive Self-Improvement

Framing: an LLM agent emits text; a **harness** turns text into actions
(tools, UI rendering, feedback formatting). Macaron treats **train–serve
harness divergence as a bug to fix at the source**, and the harness as a
*first-class training target*. Two commitments: (i) harness should get
*lighter* as the model gets stronger; (ii) training and serving share an
explicit runtime contract.

**§3.1.1 UI4A (component-native GenUI).** The two prior poles: HTML-native
(max expressiveness, inherits all of raw web dev's failure modes) and
schema-native (verifiable but capped by the component catalog). UI4A middle
path: agent writes *ordinary frontend code* (imports, components, state,
functions) inside runtime-enforced boundaries. Mental model: **import +
component + state + Action**. Action contract: every user gesture is a
structured object with 4 fields — **Origin** (surface), **State** (data read),
**Execution** (local function or agent event), **Visibility** (incl. a NoAI
boundary for fields the model must not see). Key lesson from prior
schema-native work: *teaching when to render matters more than how*. Measured:
UI4A ≈672 output tokens vs raw HTML ≈1224 on a 48-case gallery (~45%
reduction); with non-reasoning LoRA + streaming + partial rendering, up to
~6× faster time-to-first-render. L3 specializes in *when to render* and
component picking/binding; the substrate carries *how*.

**§3.1.2 REPL agent harness.** The action surface behind L1: a *stateful
Python read–eval–print loop*, "one rung above" discrete function calling/MCP/
shell. Two mechanisms:
- **Executable composition**: dependent values persist as variables; a chain
  of dependent ops resolves in one turn; the model never restates an
  intermediate it already computed. (Case study §6.3: same task = 48 turns
  one-call-per-turn vs **6 turns** REPL; both give exact answer 8208.)
- **Validated reuse**: `save_tool` stages a self-derived helper into a
  candidate pool; `promote_tool` makes it callable later **only after passing
  a private validation run against held-out reference**. Promotion-after-
  validation is load-bearing (a promoted helper is shared across sessions and
  live in training rollouts — promoting unvalidated would turn a one-off
  error into a standing fault). Bad promotions are logged and demoted.
External services enter via **ToolProxy** wrappers with NoAI visibility
boundaries and retry semantics. MindForge imports the *same harness object*
for rollouts — forecloses train–serve tool drift. Honest negative: REPL is
not universally best — stateful observe-before-commit APIs penalize blind
committed chains (BFCL v4: REPL 49.5% vs function-calling 54.0%), and
independent shell tasks give it nothing to compose; harness falls back to
discrete calls/shell there.

**§3.1.3 Harness Context Protocol (HCP).** A **versioned TOML contract** for
recreating a runtime from a portable, auditable artifact. Standardizes:
runtime/model selection (workdir, backend, provider, model id, context window,
gen limits), action surface (tool allowlists, MCP servers, extensions, hooks,
policies), context resources (system prompts, AGENTS.md-style files, skills,
prompt templates), session/workspace state (snapshots, visibility, declared
outputs), environment contract (env names, path resolution, secret
*references* without embedded credentials). Crucial clarification: **HCP
carries no gradients** — no optimizer writes to it. What the model can do is
*rewrite the harness the protocol describes*: propose edits to prompts/skills/
allowlists/hooks, get them evaluated on a re-run of the affected slice, ship
survivors as next config — *self-iteration in language space, not parameter
space*. Neither half suffices alone: config search can't acquire a skill the
base lacks; adapter training can't reach the tool surface or instructions.
Coupled through MinT, the **trainable object is the model–harness pair**.

**§3.1.4 Live harness updates.** New capability lifecycle: first a registered
tool / HCP config / UI4A component / validated REPL helper (no weight update);
then, when trajectory evidence warrants, transfer into a specialist adapter.
Three release clocks (base / specialists / harness); harness changes ship
fastest since they need no weights.

**§3.2 Recursive Self-Improvement (RSI).**
*Math*: policy π_φ(a_t | o_≤t; θ, c) — θ frozen base params, φ trainable LoRA
params, **c a versioned harness configuration**. Actions a_t may be messages,
clarifications, discrete tool calls, or expressions evaluated by the REPL.
*Adapter selection is NOT in this action space* — it's a separate Proxy-
mediated route hop. Episode τ = (o_0, a_0, …, o_T, a_T, y) with y = task
outcome + process judgments. Two distinct update paths: model optimization
(changes φ, θ fixed; GRPO backend) vs configuration search (selects new c;
HCP not a learned parameter). The action substrate is part of c, not an
incidental wrapper.

**MindForge** = control plane of RSI: manages problem banks, evaluation runs,
trajectory→dataset conversion, training jobs, and a version registry linking
each model to parent/data/config/eval. Lineage is the unit of an RSI
generation: (problem bank, model, HCP) → evaluated trajectories → (dataset,
next model, next HCP). "A model checkpoint without its problem-bank version,
selected data, and runtime configuration is an incomplete record."

**Three-stage cycle**:
1. **Discovery** — current model proposes harder task variants (lift
   constraints, hidden preferences, chained sub-goals, embedded
   contradictions); each proposal must carry a verifiable answer or rubric.
   Retention criteria: *quality* (well-defined task+eval) AND *learning value*
   (current model doesn't already solve it reliably).
2. **Expansion** — accepted tasks executed under a fixed model–HCP pair;
   evaluator scores outcome AND process; **audits localize failure to model /
   task / harness config** (e.g. unnecessary round-trips, an action that
   should have stayed discrete, missing tool boundary, premature-routing
   instruction). Candidate HCP changes re-run the affected slice; acceptance
   gate = automatic eval pass, human review only for tool-exposure/safety
   boundaries.
3. **Update** — trajectory selection (filter invalid, dedupe, keep validated +
   informative); train LoRA specialists via GRPO with base frozen; register
   accepted HCP as next generation's runtime config; link everything.

AutoResearch = language-space search inside expansion. "Context learning" =
transfer of validated language-space behavior into the parameter update (not
in-context learning). The report explicitly does *not* attribute gains across
weights/config/interaction — that controlled comparison is missing.

**§3.2.5 Expansion coverage (the headline experiment for me).** 122 simulation
tasks from 29 TerminalBench 2.1 families that the frozen GLM-5.2-FP8 base
fails *all* of. Base stays frozen the whole time — only HCP-carried resources,
skills, tool exposure, hooks change. 69 chronological jobs, 450 attempts
(3.69/task). Phase breakdown:
- Jobs 1–10 retry control: 50 attempts, 12.0% pooled, coverage 2/122.
- Jobs 11–12 full-set sweeps under ONE config each: 244 attempts, 4/122 (3.3%)
  and 11/122 (9.0%); 97 of 109 harness errors live here.
- Jobs 13–48 targeted skill/HCP search: 64.5% pooled, coverage → 60/122.
- Jobs 49–69 + stop-gate hooks: 81.2% pooled, coverage **122/122**, 3 errors.
Final phase ≈ 13× the per-attempt yield of the full-set sweeps. Reading:
failure under baseline config does NOT mean the behavior is absent from the
frozen model — adaptive config search elicited all of it. (Scoped honestly:
adaptive coverage ceiling, not a held-out generalization estimate; the φ
transfer step was NOT executed/measured.)

### §4 Infrastructure

**§4.1 MinT** (MindLab Toolkit): manages LoRA RL over resident bases. Key
distinction: **adapter revision** (immutable LoRA snapshot exported in serving
tensor layout) vs **policy record** (mutable service state: compatible base
version, adapter shape, latest trainer checkpoint + optimizer state, rollout
records, exported revisions). Trainer checkpoints/optimizer state never cross
the serving boundary; rollouts/eval/serving select exported revisions.
Measured: adapter-only handoff vs merged-checkpoint handoff = 18.3× faster
(Qwen3-4B rank-32) / 2.85× (Qwen3-30B rank-16). Catalog: built all 10^6
entries of a packed Qwen3-30B rank-1 adapter catalog with zero build errors,
256-entry audit sample across 100 shards — *addressability*, not "1M adapters
in GPU memory". Kimi K2 (1.04T total/32.6B active) LoRA RL on 64 H800 as
evidence the lifecycle executes at scale.

**§4.2 LongStraw** (response-only long-context execution). Problem: full-
sequence autograd over huge prompts is memory-bound. For shared prompt length
P and GRPO group of G responses with lengths R_i:
  **M_live ≈ M_weights + M_prefix(P) + M_grad + max_i M_graph(R_i) + M_score**
— capture prompt state *without autograd*, replay one response at a time
*with* autograd, release each graph after backward, finalize rank-local
gradients once, step after all G members. Response-only objective: response
tokens condition on the full prompt but no gradient flows through prompt-token
computation. "Exact" = stated token positions + response-only transaction,
NOT parameter-wise equivalence to full backprop. Architecture-specific state:
Qwen3.6-27B hybrid → recurrent boundary state + CP KV pages; GLM-5.2 →
CP-sharded multi-head latent-attention pages + DSA indexer-key pages, sparse
selection across CP ranks, routed response tokens across EP ranks. Receipts
(condensed from companion report): exact-2M (2,097,152 positions) response-
only replay on 8×H20 CP8 for Qwen (G=2,8; G 2→8 adds 0.208 GB peak); 4.25M
(4,456,448) resident-prefix reuse (8 optimizer steps / 64 replays on one
captured prefix — amortization check, prefix NOT recaptured); GLM-5.2 exact-2M
end-to-end online (32×H20, rollout TP8/PP4, train CP32/EP32, G=2, 5 generated
/ 4 scored tokens). Not throughput comparisons, not evidence Macaron was
trained at these lengths.

**§4.3 Rollout–training mismatch on sparse bases.** Separate rollout and
learner engines can assign different token probabilities even with identical
nominal weights; sparse MoE routing + DSA add discrete choices (expert sets,
sparse-attention index sets) ⇒ likelihood ratio compares *different
computations*, breaking the on-policy contract. Three path-dependent controls:
- **R3 (rollout routing replay)**: record selected expert ids per token at
  rollout; learner reuses them where mappable, else *excludes the token* from
  the policy-gradient term. Diagnostic: out-of-route scoring 0.0013% with R3
  (87 steps) vs 0.0097% without (50 steps).
- **DSA implementation alignment**: align rotary-position layout, normalized
  query/key inputs, deterministic top-k, frozen-indexer defaults, CP layout,
  LoRA target loading before rollout.
- **IcePop-style residual filtering**: compute train-vs-rollout probability
  ratio per token; zero importance weight when outside trusted interval.
None is a proof of equivalence; alternatives selected by model path, not
serial stages. No Macaron-V1 ablation attributes benchmark gains to them.

### §5–6 Benchmarks & results (skim of tables)

Three benchmark groups: Personal Intelligence (ChatBench, LivingBench —
internal, dual-judge outcome+process), agent (VitaBench, VitaBench2, τ³-Bench,
PinchBench, ClawGym), coding/terminal (SWE-Verified, TerminalBench 2.1,
DeepSWE, SWE Atlas QnA), GenUI (UI4A-Bench: compile/render correctness,
control wiring, screen fit). Venti highlights vs 6 baselines (point estimates,
mixed provenance — starred = imported): ChatBench 58.3 (best), LivingBench
64.0 (best), UI4A-Bench 87.8 (best), TerminalBench 2.1 87.6 (best listed),
VitaBench 60.0, PinchBench 94.0, ClawGym 77.7, SWE-Verified 85.6 (Opus 4.8
88.6*). Tall vs its own Qwen3.6-35B-A3B base: wins on the seven common rows
(e.g., ChatBench 54.9 vs base; full table in paper). Multimodal retention
under text-only adapters: Tall higher than base on OCRBench/MMBench-EN/MMMU/
MME-cognition, −52.99 on MME perception; no variance/judge analysis — not
established preservation. Case studies: REPL 6-turn vs 48-turn composition;
LivingBench Colombo replanning case (outcome met, process 0.593 — dual-judge
point); UI4A case-level comparison. Evidence summary is admirably scoped: no
cross-benchmark rank; component-level causal attribution (specialization vs
routing vs harness vs compounding) remains open.

### §7 Discussion

Both bets explicitly unsettled: adaptation is a property of *the loop*, and
the release is one snapshot (no cross-generation lift measured); collaboration
is an *architectural affordance* (registry can admit third-party/personal
adapters; routing interface as interoperability contract) — only the four
shipped specialists tested. Limitations: RSI may optimize into a local mode
(self-generated tasks converge on shapes the current policy finds
discoverable); simulator–user mismatch; data-governance documentation absent;
single-LoRA budget-matched comparison missing; character stability degrades
after multiple preference-drift events in long sessions (unquantified).

---

## Resolved terms (were unclear after pass 1)

- **Proxy**: engine-agnostic serving layer in front of vLLM/SGLang; executes
  the 3-stage loop, keeps the timeline, rebuilds own-views, does rollback.
  Single OpenAI-compatible model name "Macaron-V1-Venti"; internal names
  private. Three API surfaces (stateless Chat Completions, stateful Responses,
  Anthropic Messages).
- **Own-view**: deterministic per-adapter message-list reconstruction from the
  append-only timeline (own turns verbatim, others as 192-token summaries).
  Load-bearing for prefix-cache hits and continuity.
- **HCP**: versioned TOML runtime contract (model, tools, prompts, skills,
  hooks, workspace, env/secrets). No gradients; model can *edit* it — that's
  language-space self-iteration.
- **UI4A**: component-native GenUI harness — ordinary frontend code in
  runtime-enforced boundaries; Action objects (Origin/State/Execution/
  Visibility incl. NoAI).
- **LongStraw**: response-only training path for million-token contexts —
  capture prompt state sans autograd, replay responses serially with autograd.
- **DSA**: DeepSeek Sparse Attention — the sparse-attention mechanism in
  GLM-5.x bases; source of discrete indexer-selection mismatch between
  rollout and learner engines.
- **Stored values**: logical count of adapter tensor elements from released
  headers (7.69B/adapter Venti) — not active-per-token, not device memory.

## What I'd implement (toy scale) — from these notes

Frozen small base transformer + 4 LoRA "specialists" trained on synthetic
task families with genuinely different output shapes + L0-as-router via
constrained decoding to canonical labels + Proxy loop (route→answer→summary)
with per-adapter own-views rebuilt from an append-only timeline + tool-result
stickiness + rollback. Measure: routing accuracy/confusion, own-view
byte-stability (prefix-cache hit simulation), summary-based handoff quality vs
full-history, single-LoRA-joint vs MoL interference on a mixed workload.
Optionally: mini expansion search (prompt/skill edits only, frozen weights)
over a failure slice to echo §3.2.5.

— status: `breaking_down` (pass 2 done; all sections read deeply)
