# Writeup — Demystifying Agent Skills: Why They Work—Until They Don't

> **Paper:** Demystifying Agent Skills: Why They Work—Until They Don't
> **Authors:** Zhiyuan Jiang, Fangrui Huang, Hanwen Xing, Xander Wu, Yipeng Gao,
> Rui Cao, Mengdi Wang, Shilong Liu, Yijiang Li
> **Year:** 2026 · **arXiv:** [2608.14036](https://arxiv.org/abs/2608.14036)

## The one-paragraph version

Skills (compact `SKILL.md` procedural packages dropped into an agent's
environment) work — but not for the reason most people assume. Through
controlled, paired experiments the authors show that skills help because they
act as **procedural anchors**: they stabilize *execution* (setup order, tool
sequences, verification habits, pitfall avoidance), not because they inject
missing *facts* (65.7% of skill-win cases are attributed to anchoring vs 4.5%
to knowledge injection). The skill-use pipeline then decomposes into
independent bottlenecks — representation, outcome annotation, transfer,
retrieval — and each breaks differently: distilled skills beat raw workflow
memory by +6 points under matched conditions, distillation is what makes an
artifact portable across agent frameworks, and retrieval precision collapses
(29.6% → 3.3%) as pools grow to 100 while downstream success *stays flat*,
meaning perfect retrieval is neither necessary nor sufficient.

## The problem

Everyone is bolting "skills" onto LLM agents, and everyone evaluates them the
same broken way: run the agent with and without the skill, compare aggregate
success rates. That single number can't tell you:

- whether it was the skill's **content** or merely its **format/placement**
  that helped;
- whether a failure came from **retrieving the wrong skill** or from
  **misapplying the right one**;
- whether the same artifact would survive a **different agent harness**;
- which stage to invest in when building a self-evolving agent (better
  distiller? better retriever? better labels?).

Without mechanism-level answers, skill libraries are cargo cult.

## The idea

Run controlled experiments where **everything except one factor is held
fixed**, then label *why* each trial succeeded or failed using a taxonomy that
is **induced from the data** (open coding) rather than assumed. The key
controls:

1. **Same source experience, different representation.** From one pool of
   trajectories with a fixed success/failure mixture (5s0f … 0s5f), build both
   a Workflow Memory (cleaned traces, inlined in the prompt) and a Skill
   (distilled SKILL.md, placed in the environment). Compare each against raw.
2. **Outcome annotation ablation.** Rebuild skills with success/failure labels
   stripped from the distiller's input. If gains survive, content — not labels
   — is doing the work.
3. **Cross-framework transfer.** Artifacts distilled from Codex runs,
   executed inside Gemini CLI.
4. **Retrieval decomposition.** Skill pools of size 5→100 with distractors
   that are random, semantically similar, or dissimilar — measured at three
   decoupled levels: offline embedding top-1, explicit agent selection, and
   actual use during real execution.

Then a two-round open-coding pass over stratified trial transcripts yields a
**12-mode taxonomy in 3 super-categories** (success / execution failure /
invocation-boundary failure), validated at 95.8% human agreement (κ = 0.952),
and all pairwise contrasts get 1,000-resample bootstrap CIs.

## How it works (the intuition)

The cleanest mental model: an agent's failure mass is *conserved but
redistributable*. A raw agent fails mostly on execution-layer issues —
environment setup, output schemas, forgetting to verify at runtime. Give it
any procedural artifact and those failures roughly halve; the agent now
succeeds "because the artifact told it to" (`skill_guided_success` jumps to
61.6% while `autonomous_clean_success` collapses to ~0 — structural zeros that
make sense once you see them). But the artifact introduces its *own* failure
modes: workflow memory drowns the agent in process noise (timeouts 10.6% vs
1.7% raw), and skills add a new category — `skill_guidance_misapplied_or_
ignored` (10.0%) — the agent opens the right skill and botches or ignores it.

Why does distillation beat memory? Workflow memory preserves the *literal
shape* of its source traces, failed branches and all. The skill encodes the
*pattern*. The paper's worked example (A.3) is perfect: on a React performance
task, raw over-caches and fails an API-call test; WM passes that but keeps the
serialized fetch structure and fails the latency test (915 ms > 800 ms); the
skill says "start promises early, await late, parallelize independent fetches"
— passes all 11 tests at 711 ms. Same source experience; the representation
decides whether you inherit the accident or the principle.

Why does retrieval barely matter? Because a skill that's *approximately*
right still anchors the procedure. The agent over-selects from big pools
(precision craters) yet success stays flat — the anchor effect is robust to
picking a near-miss skill, at least when distractors aren't adversarially
similar.

## What I learned by implementing it

My re-implementation (`implementation/`, pure-stdlib Monte-Carlo simulation of
the whole experimental apparatus — arms, mixtures, distillation, transfer,
retrieval, plus the aggregation math with paired bootstrap CIs) made a few
things concrete that the paper leaves implicit:

- **The paired design is the whole ballgame.** Task difficulty varies wildly;
   the moment you compute Skill−WM deltas *within* task+setting triples and
   bootstrap over triples, the noise floor drops enough that a +6-point effect
   is cleanly significant. In my synthetic world the same paired machinery
   gave +8.50 pts, CI [+5.56, +11.67], from effects I'd have drowned in if I'd
   compared unpaired aggregate rates.
- **Structural zeros fall out of the labeling logic, not the physics.** The
   moment an artifact exists and the trial succeeds, the judge attributes the
   success to the artifact — "autonomous success with a loaded skill" is
   nearly unobservable *by construction*. This made me read Table 11 much more
   carefully: those 0.4/0.2% cells are a measurement artifact as much as a
   behavioral fact.
- **The precision/recall definitions do real work.** Counting each distinct
   touched skill once, scoring P=0 on empty predictions, and computing F1 from
   aggregated P/R (not per-record averages) — getting any of these wrong
   changes the RQ4 story materially. Eq. 1–2 are short; they're load-bearing.
- **The mixture grid is quietly the best ablation in the paper.** Monotone
   degradation from 5s0f to 0s5f, with the skill degrading more gracefully
   than WM, is what convinced me the anchoring story is causal and not just
   "more context helps."

## What surprised me / was harder than expected

- **Workflow memory can be worse than nothing** — carrying failed branches
   inline actively hurts (timeouts ×6 vs raw). I expected "WM < Skill"; I did
   not expect "WM < raw" to be a live possibility.
- **Retrieval precision collapsing without success collapsing** is deeply
   counterintuitive if you come from a RAG background, where retrieval quality
   is treated as the ceiling. Here the ceiling is procedural relevance, and
   it's a soft, fuzzy ceiling.
- **The random-pool curiosity:** on random distractors, Codex success *rises*
   with pool size (24.3 → 44.8) — non-gold skills are sometimes genuinely
   useful. My simulation reproduced this and it's the finding I'd most like a
   follow-up paper on.
- Hardest part to implement faithfully was the **three-level decoupling in
   RQ4** (embedding / selection / execution must not leak into each other —
   no preselection passes through to execution). It's easy to accidentally
   make Arm 3 "fairer" and destroy the paper's point.

## References

- Paper: https://arxiv.org/abs/2608.14036
- Official code: https://github.com/zhiyuanjiang04/demystify-agent-skills
- My implementation: `implementation/` (run `python3 train.py`, ~5 s, stdlib only)
- Breakdown: `breakdown.md`
- First/second-pass notes: `notes.md`
