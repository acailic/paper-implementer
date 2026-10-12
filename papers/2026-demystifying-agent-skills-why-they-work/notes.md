# Notes — Demystifying Agent Skills: Why They Work—Until They Don't

**Paper**: Demystifying Agent Skills: Why They Work—Until They Don't
**Authors**: Zhiyuan Jiang, Fangrui Huang, Hanwen Xing, Xander Wu, Yipeng Gao, Rui Cao, Mengdi Wang, Shilong Liu, Yijiang Li (Princeton, UCSD, Stanford, USC, JHU)
**arXiv**: 2608.14036 (cs.AI, Aug 2026)
**Code**: https://github.com/zhiyuanjiang04/demystify-agent-skills

---

## First pass — big picture (2026-10-11)

### The problem, in my own words

LLM agents now routinely package prior experience into "skills" (Anthropic-style
SKILL.md files: compact, standardized descriptions of what to do, what to check,
what pitfalls to avoid). Everyone evaluates skills by one number: did the
skill-augmented agent solve more tasks? Nobody asks *what actually changes* in
the agent's behavior when a skill is loaded, *which* parts of execution get
stabilized, and *why the same skill* helps one task and harms another. This
paper is a mechanistic "why" study: when do skills help, why do they work, and
where do they fail — analyzed along the whole skill-use pipeline (representation
→ transfer → retrieval → invocation), not just end-to-end success.

### Core claims (from abstract + intro + findings)

1. **Skills work as procedural anchors, not knowledge injections.** In their
   contrastive taxonomy, `procedural_anchor` accounts for **65.7%** of skill
   mechanism labels vs only **4.5%** for explicit `knowledge_injection`. Skills
   stabilize *action* (setup steps, tool sequences, checks, pitfalls) rather
   than supply missing *facts*.
2. **Representation matters with experience held fixed.** Same source
   trajectories, two representations: distilled SKILL.md vs raw-ish Workflow
   Memory. Skill beats Workflow Memory by **+6.06 points** (95% bootstrap CI
   [+0.76, +11.36]) in matched comparisons. Oracle-status success: 61.9% (skill)
   vs 59.1% (raw) vs 55.9% (workflow memory) — workflow memory can be *worse
   than nothing* because it carries failed branches, verbose exploration, and
   process noise (timeouts: 10.6% of workflow cases vs 1.7% raw).
3. **Retrieval is a separate bottleneck.** As skill pools grow 5 → 100,
   execution-time (Arm 3) *actual-use precision* collapses (29.6% → 3.3%
   averaged) while *downstream success stays flat* (~36–39%). Offline
   identification (embedding ranking 88.3%→76.9%; agent selection 70.0%→63.7%)
   degrades much more gracefully — but *similar* distractors crush it
   (70.5%→53.4% embedding precision; vs 84–98% for random/dissimilar pools).
   So: exact ground-truth skill invocation is **neither sufficient nor
   necessary** for success.
4. **Skills have their own failure surface.** `skill_guidance_misapplied_or_ignored`
   = 10.0% of skill-arm cases vs 0.8% raw / 0.4% workflow. SC3 category
   (invocation/budget/boundary failures) rises to 78/528 with skills vs 19/528
   raw. Skills reduce execution-layer failures (SC2) but don't fix
   algorithmic-logic errors.
5. **Outcome annotations matter when traces are mixed.** No-hint skill creation
   (success/failure labels removed) is fine on success-only pools but degrades
   sharply once failed trajectories enter the pool.

---

## Second pass — deep read (2026-10-12)

### 1. The four research questions, precisely

- **RQ1 (representation)**: Does packaging the *same* prior experience as a
  standardized skill differ from injecting it as direct procedural memory
  (workflow memory)? Control: both artifacts built from the *same trajectory
  pool*; only representation varies.
- **RQ2 (outcome annotation)**: Do skill benefits come from procedural content
  or from visible success/failure labels? Ablation: "no-hint" skill creation —
  same trajectories, annotations removed from the skill-creator's input.
- **RQ3 (cross-framework transfer)**: Are skills coupled to the harness that
  produced them? Artifacts built from Codex trajectories, evaluated inside
  Gemini CLI (different prompting style, tool interface, execution loop).
- **RQ4 (retrieval)**: How do pool size and confusability affect (a) offline
  embedding identification, (b) explicit agent selection, (c) execution-time
  actual use — and downstream success?

### 2. Experimental arms (Table 5) — and the placement asymmetry

| Arm | Injected prior experience | How it reaches the agent |
|---|---|---|
| Raw | none | — (baseline) |
| Workflow Memory | cleaned prior workflows | **appended to the task instruction** (inline procedural memory) |
| Skill | same workflows distilled to SKILL.md | **placed in the execution environment** as a reusable resource (Anthropic skill protocol: not fully inlined into initial context) |

This placement asymmetry is a real confound worth noting: WM = inline text in
the prompt; Skill = a file/resource the agent must *notice and open*. Yet Skill
still wins — and wins *more* in SC3 "misapplied/ignored" failures, consistent
with the invocation burden being real.

Setup details: Harbor evaluation workflow, n=5 trials/task, parallelism 20,
Docker-isolated, verifier-based success (positive numeric reward = success;
zero/missing/non-positive = failure — aligned to benchmark oracle, not agent
self-report). Benchmarks: Terminal-Bench 2.0 (89 tasks), Terminal-Bench-Pro
(200-task public split of 400), SkillsBench (86 tasks, 11 domains — has native
task→skill gold annotations, hence used for RQ4). Agent–model pairings:
Codex + GPT-5.3-Codex and Gemini CLI + Gemini-3.1-Pro-Preview (RQ1–RQ3);
RQ4 Codex arm uses GPT-5.4 (GPT-5.3 access gone) → RQ4 interpreted strictly
within-pairing, no cross-RQ numeric comparison.

**Composition grid**: for each selected task collect successful AND failed raw
trajectories until balanced pool; mixtures 5s0f, 4s1f, 3s2f, 2s3f, 1s4f, 0s5f
(s = successful, f = failed source traces, budget 5). From the *same* selected
traces build both the WM and the Skill. Key control of the whole paper.

### 3. The contrastive taxonomy pipeline (Section 4 + Appendix A.2) — the methodological core

Five stages:

1. **Manifest construction / normalization**: heterogeneous benchmark outputs →
   one trial table: benchmark, task, setting (mixture), arm, reward, exception
   type, model, timestamps, token metrics, artifact pointers. 8,135 records;
   7,837 with transcripts; 3,570 skill-arm records with linked skill artifacts.
   Success derived from verifier reward only.
2. **Stratified open coding**: sample 240 trials stratified by benchmark ×
   setting × arm × outcome (fixed seed, min per cell). Each trajectory →
   Claude Sonnet 4.6 (headless, no tools, no session persistence) with context
   budgets: instruction ≤3k chars, skill ≤3k, transcript 6k head + 12k tail
   (tail larger because final errors/verifier decisions concentrate at the
   end). Output schema includes free-text `primary_mode_candidate` — the
   vocabulary must *emerge from data*, not be imposed. 600s timeout per call,
   cached by trial id. → 238 valid unique labels.
3. **Two-round taxonomy induction**: batch labels (~60/batch) → per-batch 8–14
   canonical snake_case modes with assignments; then a merge prompt over
   batch-mode names/definitions/counts → 9–14 global modes + deterministic
   remapping (validated: every id assigned exactly once). Final: **12 modes**.
4. **Paired contrastive labeling**: 528 triples (SB 144, TB2 186, TB-Pro 198;
   88 per mixture). Triple = (raw, workflow, skill) trajectories of the *same*
   task+setting; raw matched by benchmark+task (no mixture), WM/S by
   benchmark+task+setting. Representative trial = first under stable trial-id
   ordering (deterministic; seeded-random policy available for sensitivity).
   Judge sees task instruction (≤3k), taxonomy, per-arm outcome + trajectory
   excerpt (4k head + 8k tail each), injected SKILL.md. Outputs: per-arm mode +
   evidence quote; pairwise deltas (workflow_vs_raw, skill_vs_raw,
   skill_vs_workflow) with net_effect ∈ {fixed, regressed, unchanged, mixed,
   not_comparable} + fixed_mode/introduced_mode lists; mechanism labels
   (skill_mechanism, workflow_mechanism); confidence. → 1,584 arm-level labels.
5. **Deterministic aggregation** (no LLM): per-arm success rates, paired deltas
   with 1,000-iteration bootstrap CIs, mode frequencies, fixed/introduced
   counts, mechanism distributions, setting/benchmark trends, token summaries.

**Human validation** (Table 3): 238 labels × 3 supporting trajectories = 714
grounding checks (all confirmed); independent mapping of 238 raw labels to the
12 modes: 95.8% exact agreement, Cohen's κ = 0.952.

**The 12 modes** (Table 11, with per-arm frequencies over 528 triples):

| SC | Mode | Raw | WF | Skill |
|---|---|---|---|---|
| SC1 | skill_guided_success | 10.4% | 0.4% | 61.6% |
| SC1 | workflow_guided_success | 0.0% | 54.5% | 0.0% |
| SC1 | autonomous_clean_success | 48.7% | 0.8% | 0.2% |
| SC2 | environment_infrastructure_failure | 5.3% | 1.7% | 0.2% |
| SC2 | output_format_schema_mismatch | 7.4% | 3.8% | 3.2% |
| SC2 | background_service_lifecycle_failure | 2.7% | 2.5% | 0.8% |
| SC2 | shell_code_corruption | 1.1% | 1.9% | 0.2% |
| SC2 | algorithmic_logic_error | 8.3% | 11.0% | 7.4% |
| SC2 | static_verification_without_runtime | 12.5% | 12.5% | 11.7% |
| SC3 | timeout_budget_exhaustion | 1.7% | 10.6% | 4.4% |
| SC3 | skill_guidance_misapplied_or_ignored | 0.8% | 0.4% | 10.0% |
| SC3 | capability_or_safety_limit | 1.1% | 0.0% | 0.4% |

Note the structural zeros: `skill_guided_success` ≈ 0 in raw/WF arms,
`workflow_guided_success` = 0 in raw/skill arms, and
`autonomous_clean_success` collapses from 48.7% (raw) to ~0 in injected arms —
success with injection is almost always attributed to the artifact. Group
counts: SC1 326/528 (skill) vs 294/528 (WF); SC2 124 (skill) vs 197 (raw) and
176 (WF); SC3 78 (skill) vs 19 (raw).

**Mechanism labels** (Table 2): `procedural_anchor` (usable
procedure/ordering/checklist/tool sequence/verification plan),
`knowledge_injection` (supplies concrete domain knowledge the agent lacked),
`failure_warning` (warns about a pitfall the agent avoids), `none` (not used
meaningfully), `counterproductive` (misleads / makes run worse). Skill arms:
procedural_anchor 65.7%, knowledge_injection 4.5% (remaining split among
failure_warning / none / counterproductive — exact numbers not given in paper
text; check repo if needed).

### 4. RQ4 math (Eq. 1–2, Appendix A.7) — in plain English

- Gold set Gt = canonical skill ids attached to task t by SkillsBench
  annotations (never inferred from retrieval or success — avoids circularity).
- Predicted set Ĝi = distinct skills selected (Arm 2) or parsed from the
  trajectory as inspected/invoked (Arm 3). Repeated mentions count once;
  useful-but-not-gold skills are false positives.
- Per record: precision Pi = |Ĝi ∩ Gt| / |Ĝi| (empty prediction → 0), recall
  Ri = |Ĝi ∩ Gt| / |Gt| (always > 0 denominator since every task has ≥1 gold
  skill).
- Aggregation: arithmetic means of per-record P and R over the N valid records
  in a condition; **F1 recomputed from aggregated P and R** (2PR/(P+R)), not
  averaged per-record. Averages across distractor regimes computed from
  unrounded values, rounded only for display.
- Arm 1 (embedding): Qwen3-Embedding-0.6B cosine similarity between task
  instruction and skill descriptions; strict = single nearest skill (top-1);
  top-3/top-5 kept in artifacts (Table 14).
- Arm 2 (explicit selection): agent sees `available_skills` pool, must choose
  skills it *would* use; no execution.
- Arm 3 (real execution): full pool placed in the environment; no
  preselection passed from Arms 1/2 (independence!). Post-hoc parse of which
  skills were actually inspected/invoked + verifier outcome.
- Pools: k ∈ {5,10,20,50,100}; ground truth + (k−1) distractors sampled as
  random / semantically similar (embedding near-neighbors) / dissimilar
  (far-away). Fixed tasks, seeds, schedule; only procedure and distractor
  composition vary.

**Key RQ4 numbers** (Tables 4, 15):
- Arm 1 top-1: random 97.7→84.1; similar 70.5→53.4; dissimilar 96.6→93.2
  (k=5→100). Semantic confusability is the dominant stressor offline.
- Arm 2 P (mean of two pairings): 70.0 (k=5) → 63.7 (k=100); but recall at
  k=100 stays 70.5–85.2% — agents *include* the gold skill but over-select.
  Gemini similar-pool selection is weak from the start (54.3 at k=5); Codex
  degrades hard on similar pools (51.9 → 31.9).
- Arm 3 actual-use P: Gemini 16.9→0.7; Codex 42.3→5.9 (dissimilar/random mix);
  averaged 29.6→3.3. Recall stays 54.3–73.6 at k=100. Success flat: Gemini
  ~36–39, Codex 35.4→42.0 (actually *rises* with pool size on random pools:
  24.3→44.8 — extra skills seem to help Codex, an under-explained curiosity).
- Curiosity worth flagging: Codex Arm-3 P on *similar* pools (51.3 at k=5) is
  far above Gemini's (17.6) — the two pairings differ much more in execution
  behavior than in offline selection.

### 5. RQ2/RQ3 numbers

- RQ2 (Table 16): normal vs no-hint success per mixture. Pattern: no-hint ≈
  normal at 5s0f (sometimes higher, e.g., Codex TB2 0.7677 vs 0.7548), gap
  grows as failed traces enter; biggest gap Gemini TB2 3s2f: 0.7462 vs 0.4000.
  Interpretation: with clean success-only traces the outcome labels are
  redundant (evidence speaks for itself); with mixed evidence the labels are
  what lets the distiller separate signal from noise.
- RQ3 (Figure 4, transfer into Gemini CLI): at 5s0f skill 76 vs workflow 62
  vs raw 56; at 4s1f 76/58; 3s2f 74/60; 2s3f 70/56(?); 1s4f 70/62; 0s5f
  56/54 — skill > workflow at *every* mixture; skill above raw everywhere,
  workflow sometimes below raw. Distillation yields portability; raw-ish
  procedural memory is framework-coupled.

### 6. Cost analysis (A.9, 83-task matched intersection)

| Representation | Success | Total tokens/task | Δsucc vs Raw |
|---|---|---|---|
| Raw trajectories | 64.1% | 555.7K | — |
| Workflow Memory | 64.8% | 426.2K | +0.7 pp |
| Skill | 69.6% | 521.5K | +5.5 pp |

WM = most token-efficient; Skill = most effective, cheaper than Raw but
costlier than WM (+95.3K total vs WM). An effectiveness–efficiency frontier,
not a free lunch. (Per-task averaging first within task, then across tasks, to
avoid over-weighting multi-trial tasks.)

### 7. Lightweight baselines (A.8) — the "is it just compactness?" control

On 26 TB2 tasks, 5 trials each (130 trials): Raw 50.0%, instruction-derived
short plan (3–5 high-level steps) 47.7%, workflow-derived test-first template
59.2%, Workflow Memory 62.3%, Skill 79.2%. So a *generic* compact procedural
hint does NOT reproduce the skill effect — the content must be distilled
task-relevant procedure. (Short plan is actually below raw: vague guidance can
distract.)

### 8. Worked example (A.3) — react-performance-debugging, 1s4f

Same task, three arms: Raw fails `test_external_api_actually_called` (checkout
7 ms — external API bypassed by over-caching); Workflow Memory passes the API
test but fails `test_checkout_fast` (915 ms > 800 ms — serialized dependent
fetch); Skill (whose SKILL.md says "start promises early, await late; fetch
independent data in parallel, trigger dependent fetch as soon as prerequisite
resolves") passes all 11 tests with 711 ms warm latency. Perfect illustration
of "procedural anchoring": the skill encodes the *pattern* (early-start +
partial parallelism), not a fact. WM preserved the literal (serialized) code
shape from its source trace.

### 9. Limitations (paper's own + mine)

- Paper's: terminal/tool benchmarks only; few agent–model configs; taxonomy
  from ~3% stratified sample (rare modes underrepresented); paired analysis is
  LLM-assisted, not exhaustively human-coded; representative-trial policy
  ignores trial repetitions; token metadata incomplete for some skill runs.
- Mine: (a) the WM-vs-Skill *placement* asymmetry (inline vs environment
  resource) is bundled with the representation difference — a cleaner design
  would cross placement × representation; (b) skill creation is done by an LLM
  (prompt in B.1) — "distillation" quality is entangled with the creator
  model; (c) Arm 3 "actual use" parsing depends on the agent *opening* skill
  files — an agent that internalizes a skill after skim-reading scores as
  low-precision; (d) 0s5f skills built only from failed traces still sometimes
  beat raw (failure_warning mechanism) but usually hurt — interesting
  asymmetry for future work.

### 10. Open questions from first pass — now resolved

1. ✅ WM arm = cleaned procedural traces *appended to the task instruction*;
   construction = "cleaning and structuring the trajectories while preserving
   their procedural flow" (no standardization into SKILL.md).
2. ✅ Skill placement = in the execution environment as a reusable resource
   (file the agent opens), per Anthropic skill protocol; NOT inlined in the
   initial context.
3. ✅ Arm 3 precision = post-hoc parse of inspected/invoked skills from the
   trajectory; distinct ids count once; Eq. 1–2 aggregation, F1 from
   aggregated P/R.
4. ✅ Oracle-status success = verifier-reward-derived success on the 528-triple
   paired sample (59.1/55.9/61.9) — "oracle" = benchmark verifier, not agent
   self-assessment.
5. ✅ 12 modes fully enumerated (Table 11, reproduced above).
6. ✅ Mechanism labels assigned by the LLM judge per paired triple (both
   skill_mechanism and workflow_mechanism), with reasons + evidence quotes.
7. ✅ Bootstrap = 1,000 iterations over triples for paired success-rate deltas
   (Table 10: WM−Raw −0.032 [−0.081,+0.021]; Skill−Raw +0.028 [−0.023,+0.080];
   Skill−WM +0.061 [+0.008,+0.114]).
8. ✅ TB-Pro "130 trials per condition" = 26 tasks × 5 trials (the A.8
   lightweight-baseline grid; missing/infra-error trials counted as failures).

### 11. Still unclear / to check in the reference repo when coding

- Exact prompt used to construct Workflow Memory (only the skill-creator
  prompts B.1/B.2 are in the paper; WM cleaning procedure is described in one
  sentence).
- Exact distribution of failure_warning / none / counterproductive mechanism
  labels (only anchor 65.7% and knowledge 4.5% are given).
- How "inspected or invoked" is parsed concretely (regex on file reads?
  tool-call semantics per harness?) — need to look at repo or define our own
  for the reimplementation.
- Whether SkillsBench ground-truth sets ever contain >1 skill per task
  (recall denominators suggest |Gt| can exceed 1).

### 12. Implementation plan (for the coding step)

No neural training — re-implement the *analysis pipeline* on synthetic data:
1. `data.py`: synthetic task/trajectory/skill generator with tunable mixture
   grid (5s0f…0s5f), 12-mode ground-truth generator, skill pool with
   random/similar/dissimilar distractors (embeddings simulated with feature
   vectors or a small hashed representation — no GPU needed).
2. `model.py`: skill-use simulation — agent success probability conditioned on
   (mode, artifact presence, mechanism, pool size); actual-use precision
   sampler that reproduces the "precision collapses, recall stays high,
   success flat" signature.
3. `train.py`/`run.py`: run the contrastive pipeline — paired triples, mode
   aggregation, bootstrap CIs (1,000 iters), Arm1/2/3 precision-recall-F1
   curves vs pool size, and the A.9-style cost table. Output metrics.json +
   printed summary tables mirroring Tables 10, 11, 4.

### Artifacts inventory

- paper.pdf / paper.txt (extracted, 1914 lines, includes appendices A.1–A.9,
  B.1–B.3 prompts, Tables 1–16)
- templates copied: breakdown.md, implementation.md, writeup.md
