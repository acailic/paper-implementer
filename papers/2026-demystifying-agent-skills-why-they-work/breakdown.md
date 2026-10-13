# Breakdown — Demystifying Agent Skills: Why They Work—Until They Don't

> **Paper:** Demystifying Agent Skills: Why They Work—Until They Don't
> **Authors:** Zhiyuan Jiang, Fangrui Huang, Hanwen Xing, Xander Wu, Yipeng Gao, Rui Cao, Mengdi Wang, Shilong Liu, Yijiang Li
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2608.14036
> **Code (official):** https://github.com/zhiyuanjiang04/demystify-agent-skills

*Note: this is an empirical/mechanistic study, not a new architecture. Sections 3–5
are therefore mapped onto the paper's actual "method": the controlled experimental
design (Section 3), the measurement definitions (Section 4, the math), and the
experimental/training-style setup (Section 5). Where the template asks for a loss
function, the analogue is the LLM-judge labeling objective; where it asks for an
optimizer, the analogue is the bootstrap aggregation procedure.*

---

## 1. Problem & Motivation

- **Problem.** LLM agents are increasingly augmented with *skills* — compact,
  standardized procedural packages (Anthropic-style `SKILL.md` files: what to do,
  in what order, what to check, what pitfalls to avoid). But the field evaluates
  skills with a single end-to-end number: "did the skill-augmented agent solve
  more tasks?" That treats the skill (and the agent + skill *pipeline*) as a
  monolithic black box. Nobody measures *what changes* in agent behavior when a
  skill is loaded, *which* stage of skill use (representation → transfer →
  retrieval → invocation) breaks when it breaks, and *why* the same artifact
  helps one task and hurts a near-identical one.
- **The paper's three questions.** When do skills help? Why do they work? Where
  do they fail?
- **Why it matters.** If you don't know *why* skills work, you can't build
  self-evolving agents that reliably generate their own skills: you can't
  control for confounds (is it the content or the format?), you can't diagnose
  whether a failure is a retrieval failure or an application failure, and you
  can't decide where to invest (better distiller vs. better retriever).
- **Prior approaches and their limits.**
  - Skill/workflow-memory papers (Voyager, Skill-Library, Workflow-Memory-style
    systems) report aggregate success deltas only.
  - Benchmark papers (SkillsBench etc.) provide task→skill gold annotations but
    evaluate end-to-end success, not mechanism.
  - None of them (a) hold the *source experience* fixed while varying only its
    *representation*, (b) decouple *offline identification* from *execution-time
    use*, or (c) categorize *failures introduced by the skill itself*.

## 2. Key Insight / Contribution

- **Core idea.** Skills help primarily because they act as **procedural
  anchors** — they stabilize *execution* (setup order, tool sequences,
  verification checklists, pitfall avoidance) — not because they inject missing
  *facts*. This is measured directly: in paired same-task contrasts,
  `procedural_anchor` explains **65.7%** of skill mechanism labels while explicit
  `knowledge_injection` explains only **4.5%**. Consequently the whole skill-use
  pipeline decomposes into independent bottlenecks: representation quality,
  outcome annotation, cross-framework portability, and retrieval — and each has a
  *different* failure signature.
- **What is genuinely new.**
  1. A **controlled representation experiment**: same source trajectories,
     packaged two ways (distilled SKILL.md vs. cleaned Workflow Memory) — Skill
     beats Workflow Memory by **+6.06 points** (95% bootstrap CI [+0.76, +11.36])
     in matched comparisons; Workflow Memory can even be *worse than nothing*
     (it carries failed branches and process noise; timeouts 10.6% vs 1.7% raw).
  2. A **12-mode contrastive taxonomy** of skill-work-failure, *induced from
     data* by two-round open coding (not imposed a priori), with strong human
     validation (95.8% exact agreement, Cohen's κ = 0.952).
  3. A **retrieval decomposition** showing actual-use precision collapses
     (29.6% → 3.3%) as pools grow 5 → 100 while downstream success stays flat —
     i.e., exact ground-truth invocation is *neither sufficient nor necessary*.
  4. Identification of a **skill-specific failure mode**: guidance misapplied or
     ignored (10.0% of skill cases vs 0.8% raw), so skills trade execution-layer
     failures for invocation/boundary failures.

## 3. Method

Reconstructable pipeline. There is no trained neural network; the "method" is an
experimental design + a measurement pipeline.

### 3.1 Overview

Four research questions map to four controlled experiments over three arms:

- **Arms** (Table 5): for a given benchmark task,
  - **Raw** — no injected prior experience (baseline).
  - **Workflow Memory (WM)** — cleaned prior workflows from a trajectory pool,
    *appended inline to the task instruction*.
  - **Skill** — the *same* trajectories distilled into a standardized
    `SKILL.md`, placed *in the execution environment* as a reusable resource the
    agent must notice and open (Anthropic skill protocol; not inlined into the
    initial context).
  The composition grid controls the source pool: mixtures **5s0f, 4s1f, 3s2f,
  2s3f, 1s4f, 0s5f** (s = successful, f = failed source traces; budget 5). *From
  the same selected traces, both the WM and the Skill are built* — only the
  representation (and its placement) varies. This is the paper's key control.
- **RQ1 (representation):** does the standardized distillate beat inline
  procedural memory, experience held fixed? → matched comparisons.
- **RQ2 (outcome annotation):** rebuild skills with success/failure labels
  stripped from the skill-creator's input ("no-hint"). If skill gains survive,
  they come from procedural content, not from visible labels.
- **RQ3 (cross-framework transfer):** artifacts built from Codex trajectories,
  executed inside Gemini CLI (different prompt style, tool interface, loop).
- **RQ4 (retrieval):** pools of size k ∈ {5, 10, 20, 50, 100} = gold skill(s)
  + (k−1) distractors sampled as *random*, *semantically similar*
  (embedding near-neighbors), or *dissimilar*; three decoupled measurement arms:
  - **Arm 1 (offline embedding):** Qwen3-Embedding-0.6B cosine similarity
    between task instruction and skill descriptions; strict = top-1.
  - **Arm 2 (explicit selection):** the agent sees the `available_skills` pool
    and picks which it *would* use; no execution.
  - **Arm 3 (real execution):** full pool placed in the environment; **no
    preselection is passed through** from Arms 1/2 (independence); post-hoc
    parse of which skills were actually inspected/invoked + verifier outcome.

### 3.2 Architecture

The contrastive-taxonomy pipeline (Section 4 + A.2) — five stages:

```
 heterogeneous benchmark runs (Codex + GPT-5.3-Codex, Gemini CLI + Gemini-3.1-Pro)
        |
        v
[1] MANIFEST NORMALIZATION
    8,135 trial records -> one table:
    (benchmark, task, setting/mixture, arm, reward, exception, model,
     timestamps, token metrics, artifact pointers); success := verifier reward
        |
        v
[2] STRATIFIED OPEN CODING  (240 trials; benchmark x setting x arm x outcome,
    fixed seed, min per cell)
    LLM judge (Claude Sonnet 4.6, headless, no tools):
      context budgets: instruction <=3k chars, skill <=3k,
      transcript = 6k head + 12k tail  (errors concentrate at the end)
      output includes free-text primary_mode_candidate   <-- vocabulary EMERGES
    -> 238 valid unique labels
        |
        v
[3] TWO-ROUND TAXONOMY INDUCTION
    round 1: batches of ~60 labels -> 8-14 canonical snake_case modes each
    round 2: merge prompt over batch modes -> global taxonomy + deterministic
             remapping (validated: every id assigned exactly once)
    -> 12 modes in 3 super-categories (SC1 success, SC2 execution failure,
       SC3 invocation/budget/boundary failure)
        |
        v
[4] PAIRED CONTRASTIVE LABELING  (528 triples: SB 144, TB2 186, TB-Pro 198;
    88 per mixture; triple = (raw, workflow, skill) on SAME task+setting)
    Judge sees: task instruction (<=3k), taxonomy, per-arm outcome +
    trajectory excerpt (4k head + 8k tail each), injected SKILL.md.
    Outputs: per-arm mode + evidence quote; pairwise deltas
    (workflow_vs_raw, skill_vs_raw, skill_vs_workflow) with
    net_effect in {fixed, regressed, unchanged, mixed, not_comparable};
    mechanism labels (skill_mechanism, workflow_mechanism); confidence.
    -> 1,584 arm-level labels
        |
        v
[5] DETERMINISTIC AGGREGATION  (no LLM)
    per-arm success rates, paired deltas with 1,000-iter bootstrap CIs,
    mode frequencies, fixed/introduced counts, mechanism distributions,
    setting/benchmark trends, token summaries
```

### 3.3 Forward pass / pipeline

Execution side (Harbor evaluation workflow): n = 5 trials per task,
parallelism 20, Docker-isolated, verifier-based success (positive numeric
reward = success; zero/missing/non-positive = failure — benchmark oracle, never
agent self-report). Benchmarks: Terminal-Bench 2.0 (89 tasks), Terminal-Bench-Pro
(200-task public split), SkillsBench (86 tasks, 11 domains; used for RQ4 because
it has native task→skill gold annotations). Pairings: Codex + GPT-5.3-Codex and
Gemini CLI + Gemini-3.1-Pro-Preview for RQ1–RQ3; RQ4's Codex arm uses GPT-5.4
(access drift), so RQ4 is interpreted strictly within-pairing.

A full experiment = pick task → collect raw trajectories until the pool has the
mixture composition (e.g. 3 successful + 2 failed) → build WM and Skill from the
*same* traces → run all three arms 5× → verifier rewards → feed records into the
taxonomy pipeline above.

### 3.4 Loss function

There is no differentiable loss. The two "objectives" in the pipeline:

- **LLM-judge labeling objective** (stages 2–4): the judge is prompted to (a)
  propose/explain a mode grounded in an *evidence quote* from the transcript
  (forces grounding; 714/714 quotes confirmed by human check), and (b) in stage 4
  produce *pairwise* net-effect judgments — i.e., the unit of analysis is the
  contrast `(arm_A trajectory, arm_B trajectory)` on the same task+setting, not
  an absolute score. This pairing is what cancels task difficulty.
- **Aggregation objective** (stage 5): unbiased paired estimates —
  success-rate deltas between arms with bootstrap CIs; precision/recall/F1 under
  Eq. 1–2 (Section 4 below).

## 4. Math

**Notation.** For task *t*: gold skill set G_t (canonical ids from SkillsBench
annotations — never inferred from retrieval or success, to avoid circularity).
For record *i*: predicted set Ĝ_i = distinct skills selected (Arm 2) or parsed
from the trajectory as inspected/invoked (Arm 3). Repeated mentions count once;
useful-but-not-gold skills are false positives.

1. **Precision (Eq. 1):**  P_i = |Ĝ_i ∩ G_t| / |Ĝ_i|, with P_i = 0 if Ĝ_i = ∅.
   *Plain English:* of the skills the agent actually touched, what fraction were
   the right ones.
2. **Recall (Eq. 2):**  R_i = |Ĝ_i ∩ G_t| / |G_t|  (denominator always > 0 since
   every task has ≥ 1 gold skill).
   *Plain English:* of the right skills, what fraction the agent touched.
3. **Aggregation rule:** P̄ = (1/N) Σ_i P_i, R̄ = (1/N) Σ_i R_i over the N valid
   records in a condition; **F1 = 2·P̄·R̄ / (P̄ + R̄)** — recomputed from
   *aggregated* P and R, not averaged per record. Averages across distractor
   regimes computed from unrounded values, rounded only for display.
4. **Success:** s_i = 1[reward_i > 0] (verifier-based). Arm success rate =
   mean over trials; per-task rates averaged within task first, then across
   tasks (avoids over-weighting multi-trial tasks — used in the cost analysis).
5. **Paired delta with bootstrap CI:** Δ = succ(A) − succ(B) over the 528-triple
   paired sample; 95% CI from 1,000 bootstrap resamples of triples.
   E.g. (Table 10): WM−Raw = −0.032 [−0.081, +0.021]; Skill−Raw = +0.028
   [−0.023, +0.080]; Skill−WM = **+0.061 [+0.008, +0.114]** (the headline +6.06).
6. **Mechanism distribution:** for skill arms, fraction of triples whose
   `skill_mechanism` label ∈ {procedural_anchor, knowledge_injection,
   failure_warning, none, counterproductive}. Headline: anchor 65.7%,
   knowledge 4.5%.
7. **Mode frequency:** per arm, count(triples with mode m)/528 — see Table 11
   under Results.

## 5. Training

(Experimental setup in place of training setup.)

- **Datasets / workloads:** Terminal-Bench 2.0 (89 tasks), Terminal-Bench-Pro
  (public 200 of 400), SkillsBench (86 tasks, 11 domains; gold task→skill
  annotations power RQ4). Distractor pools for RQ4: k ∈ {5,10,20,50,100};
  ground truth + (k−1) distractors sampled random / similar / dissimilar
  (embedding-based); fixed tasks, seeds, schedule — only procedure and
  distractor composition vary.
- **Models / agents:** Codex harness + GPT-5.3-Codex-Max (RQ1–RQ3);
  Gemini CLI + Gemini-3.1-Pro-Preview (RQ1, RQ3 transfer target); RQ4 Codex arm
  GPT-5.4 (within-pairing interpretation only). Judge model: Claude Sonnet 4.6,
  headless, no tools, no session persistence, 600 s timeout per call, cached by
  trial id. Embeddings: Qwen3-Embedding-0.6B, cosine similarity.
- **Procedure counts:** 5 trials/task, parallelism 20, Docker isolation per
  trial; 8,135 trial records total (7,837 with transcripts; 3,570 skill-arm
  records with linked artifacts); 240 stratified open-coded trials → 238 valid;
  528 contrastive triples → 1,584 arm-level labels.
- **Skill creation:** an LLM distills the trajectory pool into SKILL.md
  (prompts B.1/B.2 in the appendix); "no-hint" variant removes success/failure
  labels from the distiller's input. WM construction: cleaning and structuring
  the same trajectories while preserving their procedural flow (one-sentence
  description in the paper — exact prompt not published).
- **Compute / cost (A.9, 83-task matched intersection):** Raw 64.1% success at
  555.7K tokens/task; WM 64.8% at 426.2K (+0.7 pp, cheapest); Skill 69.6% at
  521.5K (+5.5 pp, cheaper than Raw but +95.3K over WM).

## 6. Results & Ablations

**Headline numbers.**

- Matched paired comparison (oracle-status success on 528 triples): Skill 61.9%
  vs Raw 59.1% vs WM 55.9%; Skill−WM = +6.06 pts, 95% CI [+0.76, +11.36].
- Mechanism: procedural_anchor **65.7%** vs knowledge_injection **4.5%** →
  skills stabilize *action*, not facts.
- Retrieval (RQ4, averaged): actual-use precision **29.6% → 3.3%** as pool grows
  5 → 100, while downstream success stays flat (~36–39%). Exact ground-truth
  invocation is neither sufficient nor necessary.
- Transfer (RQ3, Gemini CLI, skill vs workflow vs raw by mixture): 5s0f 76/62/56;
  4s1f 76/58; 3s2f 74/60; 2s3f 70/56; 1s4f 70/62; 0s5f 56/54 — skill > workflow
  at *every* mixture; workflow sometimes below raw. Distillation ⇒ portability.
- No-hint (RQ2): ≈ normal at 5s0f (sometimes better: Codex TB2 0.7677 vs
  0.7548) but degrades as failed traces enter (worst: Gemini TB2 3s2f 0.7462 vs
  0.4000) → outcome labels matter exactly when evidence is mixed.

**The 12 modes (Table 11; % of 528 triples per arm):**

| SC | Mode | Raw | WF | Skill |
|---|---|---|---|---|
| SC1 | skill_guided_success | 10.4 | 0.4 | 61.6 |
| SC1 | workflow_guided_success | 0.0 | 54.5 | 0.0 |
| SC1 | autonomous_clean_success | 48.7 | 0.8 | 0.2 |
| SC2 | environment_infrastructure_failure | 5.3 | 1.7 | 0.2 |
| SC2 | output_format_schema_mismatch | 7.4 | 3.8 | 3.2 |
| SC2 | background_service_lifecycle_failure | 2.7 | 2.5 | 0.8 |
| SC2 | shell_code_corruption | 1.1 | 1.9 | 0.2 |
| SC2 | algorithmic_logic_error | 8.3 | 11.0 | 7.4 |
| SC2 | static_verification_without_runtime | 12.5 | 12.5 | 11.7 |
| SC3 | timeout_budget_exhaustion | 1.7 | 10.6 | 4.4 |
| SC3 | skill_guidance_misapplied_or_ignored | 0.8 | 0.4 | 10.0 |
| SC3 | capability_or_safety_limit | 1.1 | 0.0 | 0.4 |

Structural zeros are the point: with an artifact injected, success is almost
always *attributed to the artifact* (autonomous_clean_success collapses from
48.7% to ~0). Super-category counts: SC1 326/528 (skill) vs 294/528 (WF);
SC2 124 (skill) vs 197 (raw) and 176 (WF) — skills cut execution-layer failures
roughly in half; SC3 78 (skill) vs 19 (raw) — but *add* invocation/boundary
failures.

**Retrieval decomposition (Tables 4, 15; k = 5 → 100):**

- Arm 1 top-1 precision: random 97.7→84.1; **similar 70.5→53.4**; dissimilar
  96.6→93.2. Semantic confusability is the dominant offline stressor.
- Arm 2 precision: 70.0→63.7 (mean of pairings); recall at k=100 stays
  70.5–85.2 → agents *include* the gold skill but over-select. Gemini is weak on
  similar pools from the start (54.3 at k=5); Codex degrades hard on similar
  pools (51.9→31.9).
- Arm 3 actual-use precision: Gemini 16.9→0.7; Codex 42.3→5.9 (averaged
  29.6→3.3); recall stays 54.3–73.6 at k=100; success flat (Gemini ~36–39;
  Codex 35.4→42.0 — on random pools it *rises*, 24.3→44.8: extra skills
  sometimes help, an under-explained curiosity).

**Ablations that matter most.**

1. **Composition grid** (mixture of successful/failed source traces): monotone —
   more failed traces in the pool ⇒ worse artifacts; the skill degrades more
   gracefully than WM.
2. **No-hint skill creation** (RQ2): isolates *outcome annotation* as a distinct
   causal ingredient — redundant on success-only pools, crucial on mixed ones.
3. **Lightweight-baseline control** (A.8, 26 TB2 tasks × 5 trials): Raw 50.0%;
   generic instruction-derived short plan 47.7%; workflow-derived test-first
   template 59.2%; WM 62.3%; **Skill 79.2%**. A compact *generic* hint does NOT
   reproduce the effect — vague guidance can even distract (short plan < raw).
   The content must be distilled task-relevant procedure.
4. **Cost frontier** (A.9): WM most token-efficient, Skill most effective —
   an effectiveness–efficiency frontier, not a free lunch.
5. **Worked example** (A.3, react-performance-debugging, 1s4f): Raw fails
   `test_external_api_actually_called` (over-caching bypasses the API);
   WM passes that but fails `test_checkout_fast` (915 ms > 800 ms, serialized
   dependent fetch); Skill ("start promises early, await late; parallel
   independent fetches; trigger dependent fetch as soon as prerequisite
   resolves") passes all 11 tests at 711 ms. The skill encodes the *pattern*;
   WM preserved the literal (slow, serialized) code shape of its source trace —
   the cleanest single illustration of procedural anchoring.

## 7. Limitations

- **Acknowledged by authors:** terminal/tool benchmarks only; few agent–model
  configurations; taxonomy induced from a ~3% stratified sample (rare modes
  underrepresented); paired analysis is LLM-assisted, not exhaustively
  human-coded; representative-trial policy ignores trial repetitions; token
  metadata incomplete for some skill runs.
- **Additional (mine):**
  - The WM-vs-Skill **placement asymmetry** (WM inline in the prompt vs. Skill as
    an environment resource the agent must open) is bundled with the
    representation difference — a cleaner design would cross placement ×
    representation in a 2×2.
  - Skill creation is done by an LLM: "distillation quality" is entangled with
    the creator model's ability.
  - Arm 3 "actual use" is parsed from the agent *opening* skill files — an agent
    that internalizes a skill after skim-reading scores as low-precision.
  - 0s5f skills (built only from failed traces) sometimes still beat raw
    (`failure_warning` mechanism) but usually hurt — interesting asymmetry,
    underexplored.

## 8. Open Questions / Ideas

- Would the +6.06 Skill−WM gap survive the 2×2 placement × representation cross?
  (Inline the SKILL.md; put WM in the environment as a file.)
- Since actual-use precision collapses but success stays flat, *what is the
  agent actually using* from the skills it opens? A token-level attribution
  study (which lines of SKILL.md correlate with stabilized actions) is the
  natural follow-up.
- Can the 12-mode taxonomy be predicted *cheaply* (classifier over trajectory
  features) so practitioners can run the diagnosis without an LLM judge?
- Self-evolving agents: use `skill_guidance_misapplied_or_ignored` (10.0%) as a
  *training signal* — regenerate or split the skill when invocation failures
  concentrate.
- Retrieval fix candidates: hierarchical/faceted skill indices, or letting the
  agent *ask* for skills by describing its current subgoal, rather than
  rank-from-pool.
- The Codex random-pool curiosity (success *rising* with pool size, 24.3→44.8)
  suggests non-gold skills can be genuinely useful — worth a dedicated study of
  "useful false positives".

---

*Sources: paper.txt (full text incl. appendices A.1–A.9, B.1–B.3, Tables 1–16);
cross-checked against notes.md second pass (2026-10-12).*
