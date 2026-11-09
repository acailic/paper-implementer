# Breakdown — Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills

> **Paper:** Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills
> **Authors:** Jianlyu Chen, Yuyang Hu, Hongjin Qian, Jiawei Liu, Wenqing Wei,
> Xiaolong Chen, Defu Lian, Zhicheng Dou, Chaozhuo Li, Qiwei Ye, Zheng Liu
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2609.02749
> **Code (official):** https://github.com/VectorSpaceLab/AREX-Skill

*Note: this is a systems/agent paper, not a trained-model paper. There is no
loss function or optimizer; the "method" is a knowledge-supply-chain pipeline
(distill → verify → route → progressively disclose) evaluated by controlled
agent benchmark runs. Section 4 maps the template's "math" slot onto the
paper's formal setup (Eqs. 1–9); Section 5 maps "training" onto construction
budgets and protocols.*

---

## 1. Problem & Motivation

- **Problem.** Autonomous ML-research agents are built from two components: a
  model backbone Mθ (understanding, reasoning, planning) and a harness H
  (orchestration, memory, verification, refinement). Neither carries
  **domain-specific operational knowledge** — the know-how that separates
  *knowing a method* from *making it work*: which package implements it, how
  that package wants its data laid out, which configurations are sane, which
  pitfalls silently invalidate a run. Without it, an agent burns its task
  budget rediscovering package behavior by trial and error, and whatever it
  does infer is lost when the task ends.
- **Why it matters.** That knowledge is not missing from the field — it sits
  in repositories and papers — but it is written for human readers and far
  too large to load into a task context. The methodological problem is to
  turn this abundant *declarative* material into *operational* knowledge
  automatically and at scale, in a form an agent can command: discoverable,
  loadable, ready to use.
- **Prior approaches and their limits.**
  - Stronger backbones / better harnesses: improve the *how* of research,
    leave the *what it knows* unchanged. Progress is expensive and orthogonal.
  - Experience-derived skills (Voyager, Agent Workflow Memory, ExpeL): derive
    free-form code/workflows/insights from the agent's own interaction
    traces. Quality is hard to verify, failures hard to attribute, and the
    knowledge is limited to what the agent already lived through.
  - Manual skill authoring: real operational knowledge, but cost scales with
    expert labor per domain/stack/release; large-scale SKILL.md analysis
    shows authoring inconsistencies.
  - What's missing: automatic derivation of *version-specific, verified,
    provenance-grounded* operational knowledge from static artifacts
    (repos/papers), reusable across tasks and harnesses.

## 2. Key Insight / Contribution

- **Core idea.** Give the research agent a third component: an
  operational-knowledge layer K, realized as verified Agent Skills
  (Anthropic-style `SKILL.md` + `references/` + `scripts/`), distilled
  automatically from declarative sources and loaded as *operating context*
  via progressive disclosure. The agent becomes A_res = (Mθ, H, K): backbone
  and harness unchanged; the only new variable is what the agent knows when
  research begins. Distillation — not summarization — because **no skill is
  admitted without verification**, and surviving gaps are recorded, not
  hidden.
- **What is genuinely new.**
  1. **Operational knowledge formalized as the missing layer** of research
     agents, cleanly separated from the harness (H = how to research; K =
     what to know before researching).
  2. **DisCo**, one agent in two modes: *creator mode* runs the four-stage
     distillation pipeline; *researcher mode* solves tasks with K drawn from
     the library. Two anchor forms: **task-agnostic** (z = source repo/paper,
     built ahead of time, reusable by any task) and **task-oriented**
     (z = task τ, sources discovered on demand for the task's capability
     gaps).
  3. **The AREX-Skill Library** as existence proof of scale: 5,353 verified
     skills from 1,000 widely used ML repositories, organized by an induced
     two-level taxonomy (20 areas, 178 capability families) behind a
     generated router.
  4. **Controlled evidence that the knowledge layer alone moves benchmarks**:
     same GPT-5.5 + Codex harness + run budget, with vs. without skills →
     MLE-bench +134.3%, PaperBench +34.4%, FrontierCS +9.2%, PassNet +14.0%.

## 3. Method

### 3.1 Overview

Formal setup: a research task τ = (q, D, E, g) — problem, data/materials,
environment (tools + budget), goal. The standard agent A = (Mθ, H) runs the
loop a_t ~ πθ(a | τ, h_t, H), o_t = E(a_t) until g is met or budget spent.
DisCo's agent adds K: **a_t ~ πθ(a | τ, h_t, H, K)**, where K = {S_1..S_m}
is a set of skills. K is *operating context*, not a new control loop — the
harness still plans, acts, and observes.

```
            CREATOR MODE (paid once per source, amortized)           
  anchor z ──scope──> Q ──ground──> X ──construct──> G̃ ──verify──> (G, R)
   repo | task      capabilities   evidence       candidate      accepted graph
   (agnostic|        worth          supporting     skill graph   + construction
    oriented)        covering       them (⊆ of    (3 layers,     record (evidence
                    (Q)            reachable     linked)         used, checks run,
                                   sources X)                    unresolved gaps)
        |                                                                 |
        v                                                                 v
  AREX-Skill Library  <--------------------------------------  verified skill graphs
  (router: 20 areas -> 178 families -> repo graphs)                  (G)
        |
        v
            RESEARCHER MODE (pays only for what a task opens)        
  task τ -> router -> entry skill -> open only needed component skills
            (progressive disclosure; unopened skills cost nothing)
```

### 3.2 Architecture

**The skill artifact (Eq. 6):** S = (`SKILL.md` | `references/` |
`scripts/`) — three layers with distinct jobs:

| Layer | Role | Realizes |
|---|---|---|
| `SKILL.md` | **knowledge interface** — the only layer read up front; what the skill is for, when it applies, the standard operating procedure, routing onward | usage policy (capability → selection criteria) |
| `references/` | **knowledge substrate** — API docs, algorithmic detail, configs; loaded only when needed (progressive disclosure) | capability (evidence) |
| `scripts/` | **execution interface** — executable wrappers with defined I/O the agent invokes rather than reimplements | capability (invocable units) |

**The skill graph (Eq. 7):** one source yields G = (S, L), S = {S_1..S_n},
L ⊆ S×S routing/dependency/composition links. A repository graph has a
router-like **entry skill** (applicability, minimal setup/import check,
routes) and **component skills** for bounded workflows (data prep, training,
evaluation, serving, troubleshooting, maintenance) — split by *likely future
agent tasks*, not by source directory names. Graphs are self-contained:
helpers/examples are bundled as references or skill-owned scripts so the
agent never reopens the original checkout; provenance (source commit,
package version, dirty state, relative evidence paths) is recorded.

**The library:** creator mode writes accepted graphs; a two-level taxonomy
(areas → families) plus a generated router indexes them. At research time
the agent walks area → family → repository graph → skills, so only the
selected branch ever enters K.

### 3.3 Forward pass / pipeline (the four stages, Eq. 8)

z —scope→ Q —ground→ X —construct→ G̃ —verify→ (G, R). The two anchor forms
share the pipeline and differ in what fixes the source set and the
verification signal:

| Stage | Task-agnostic (z = source c) | Task-oriented (z = task τ) |
|---|---|---|
| **scope** | Source Understanding (what is this artifact, how organized) → Capability Identification (which capabilities worth exposing) | Task Decomposition (capabilities the task calls for) → Capability Gap Analysis (which the agent cannot already supply) |
| **ground** | Knowledge Extraction — evidence gathered from the source itself; X is *selected* from fixed material | Source Discovery — search for material covering the gaps; X is *assembled* from the open ecosystem |
| **construct** | Tool Encapsulation (wrap executables behind stable interfaces) + Skill Packaging (assemble 3-layer linked graph) | Skill Generation (distill the assembled material into the task's skills) |
| **verify** | Skill Verification — identical closure for both forms | (same) |

**Repo construction protocol (A.1):**
1. Fix the **evidence boundary** before installing/executing anything:
   include/exclude map over source roots, docs, examples, tests, scripts,
   config; exclude generated files, build outputs, vendored deps, caches,
   unrelated dev internals.
2. Private Python **inspection environment**: minimal install plan; live
   checks of import names, package versions, public signatures, CLI entry
   points, optional backends, small runtime behaviors. Docs establish
   workflow *intent*; source code + live inspection *confirm* API/runtime
   claims before they enter the graph.
3. **Construct** the self-contained graph (entry + components + bundled
   references/scripts + provenance + compact routing metadata).
4. **Verify**: assertion-backed usability cases; safe repository-native
   checks (examples, tests, CLI checks, tiny-fixture checks, smoke scripts)
   classified `pass` / `skill_gap` / `native_fail` / `skip_unsafe` /
   `skip_not_selected`; a `skill_gap` triggers localized repair of the
   responsible skill/reference/script/route, then re-running the relevant
   checks. Static gates: metadata, link integrity, self-containment,
   provenance, routing metadata, no local-path leakage, runtime files
   separated from construction-only artifacts.

**Taxonomy + routing (A.1):** freeze short repo summaries → LLM proposes a
complete two-level tree → 100 stable batches of 10 repos each get a
*locator* (provisional exact/acceptable/poor/no-fit assignment, diagnosis
only) and an independent *judge* (structural review: missing capabilities,
overlapping families, mixed axes, overloaded nodes, catch-alls) → an
aggregator synthesizes reviews + deterministic fit/load statistics → a
reviser writes a complete revised tree → schema + typed-diff validation →
convergence gate (≥2 rounds, bounded no-fit/poor-fit, no blockers, no
overloaded families, explicit stop). Only after the tree is fixed are final
assignments made — classification is separate from skill generation; every
assignment needs rationale + repo evidence + confidence; keyword-only,
dependency-only, optional-integration, and example-only matches rejected;
unclassified is allowed rather than forcing a weak route.

### 3.4 Loss function (analogue)

No differentiable loss. The objective-like constraints are: (a) the
**verification gate** — maximize admitted-skill correctness subject to
checks passing, with repair loops on `skill_gap`; (b) for task-oriented
distillation, an **execution-feedback refinement loop** (Eq. 9):
F_{t+1} = (S_t, Summary(L_t), Analysis(R_t)) — the next refinement step
receives exactly the previous skill, a summary of the execution log, and an
analysis of the observed result; revision continues until budget exhaustion
or diminishing returns; (c) paired-trial retention (FrontierCS/PassNet): a
graph revision is kept only if follow-up trials reproduce the intended
behavioral change.

## 4. Math

The paper's formalism is light; every equation restated:

- **Eq. 1 — task.** τ = (q, D, E, g). q = problem statement; D = data and
  materials provided; E = environment (exposed tools, bounded budget); g =
  target the outcome must meet. Solving τ means producing artifacts y
  (code, models, results, reports) that fulfill g in E.
- **Eq. 2 — standard agent.** A = (Mθ, H). Mθ = LLM backbone (parameters θ);
  H = harness. Together they define a policy running an act/observe loop.
- **Eq. 3 — the loop.** a_t ~ πθ(a | τ, h_t, H); o_t = E(a_t). At step t,
  with history h_t = (a_1, o_1, …, a_{t−1}, o_{t−1}), the agent samples an
  action conditioned on task, history, and harness; the environment responds.
  Artifacts y are what the trajectory leaves behind.
- **Eq. 4 — the paper's agent.** A_res = (Mθ, H, K). K = operational
  knowledge made available as explicit operating context, so actions become
  a_t ~ πθ(a | τ, h_t, H, K). Plain English: the only change is that
  actions are conditioned on curated knowledge the agent can consult.
- **Eq. 5 — knowledge as skills.** K = {S_1, …, S_m}: producing operational
  knowledge ≡ producing skills.
- **Eq. 6 — skill anatomy.** S = (SKILL.md | references/ | scripts/):
  knowledge interface | knowledge substrate | execution interface (see 3.2).
- **Eq. 7 — skill graph.** G = (S, L), S = {S_i}, n ≥ 1, L ⊆ {(S_i, S_j) ∈
  S×S | i ≠ j}. L links encode routing, dependency, or composition; L may be
  empty (single-skill or independent-skill sources).
- **Eq. 8 — distillation.** z —scope→ Q —ground→ X —construct→ G̃ —verify→
  (G, R). Q = capabilities to cover; X ⊆ C = evidence gathered from
  reachable declarative material C; G̃ = candidate graph; (G, R) = accepted
  graph + construction record (evidence used, checks performed, unresolved
  gaps).
- **Eq. 9 — refinement context (MLE-bench).** F_{t+1} = (S_t, Summary(L_t),
  Analysis(R_t)): S_t = previous skill version, L_t = execution log, R_t =
  observed result at iteration t. The refiner sees exactly these three
  things.

## 5. Training

No gradient training anywhere. The analogue of "training details" is the
construction setup:

- **Backbones.** Skill construction: GPT-5.5 and GPT-5.6-sol, xhigh
  reasoning effort. Downstream runs: GPT-5.5 + Codex harness, held fixed
  across with/without-skills conditions.
- **Compute / cost.** ≈ $40 per repository average allocation × 1,000 repos
  for the library. MLE-bench: ≤ 24 GPU-hours construction + ≤ 24 GPU-hours
  running per task (separate budgets). FrontierCS: 5-hour task budget, 2
  CPUs / 4 GiB container. PassNet: A100-SXM4-40GB evaluation; construction
  via screening-and-dispatch over the 4k-instance training split (~1 day
  per pass).
- **Data.** 1,000 ML repositories selected by open-source visibility
  (GitHub stars among curation signals); 153 source papers for PaperBench
  pools; benchmark suites MLE-bench (75), PaperBench (20), FrontierCS Agent
  Track (188), PassNet (200 eval samples).
- **Leakage controls.** MLE-bench: original competition webpage and
  competition-specific content blocked during construction; final skills
  *descriptive only* (no runnable training/inference scripts) so runs can't
  replay a fixed solution. PaperBench: target paper + its artifacts excluded
  as skill sources (skills only from ≤10 related-work papers, 3–5 module
  skills each). FrontierCS: individual problem text, solution fragments,
  task IDs, tuned constants excluded from the shared graph.
- **Verification signals.** Repo graphs: assertion-backed usability cases +
  safe native checks + static gates. Paper modules: isolated assertion
  tests, then a bounded *recovery* experiment that must reconstruct a result
  without reading the original implementation repo. Task graphs: execution
  feedback (Eq. 9), paired trials vs. no-skill baseline and vs. previous
  graph version, revision kept only when the intended behavioral change
  reproduces.

## 6. Results & Ablations

**Headline (all relative gains, same backbone/harness/run budget):**

| Benchmark | no skills | + skills | Δ abs | Δ rel |
|---|---|---|---|---|
| MLE-bench Any-Medal, 75 tasks | 31.11 | 72.89 | +41.78 | **+134.3%** |
| PaperBench replication, 20 papers | 29.45 | 39.59 | +10.14 | +34.4% |
| FrontierCS Agent Track, 188 tasks | 70.63 | 77.14 | +6.51 | +9.2% |
| PassNet AS Score, 200 samples | 1.343 | 1.5313 | +0.1883 | +14.0% |

- **Beats custom harnesses without one.** Codex+skills (72.89) > best public
  MLE-bench entry Famou-Agent 2.0 (64.44), with no custom orchestration.
- **Gains grow with difficulty.** MLE-bench High tier 13.33 → 62.22
  (×4.67); PaperBench gains concentrate on low-baseline papers (ftrl 1.50 →
  17.17 = 11.4×; rice 7.94 → 48.51 = 6.1×); FrontierCS: tasks scoring <50
  rise 19.43 → 45.99 (+26.56). Interpretation: skills pay off most where
  unguided trial-and-error wastes the most budget.
- **Broad, not outlier-driven.** PaperBench improves 18/20 tasks;
  FrontierCS improves 74, unchanged 66; aggregate positive change 3.91×
  the negative; paired bootstrap 95% CI [3.41, 9.83] on FrontierCS.
- **Failure modes reported honestly.** Two PaperBench regressions
  (sample-specific-masks −5.07, stay-on-topic −4.52), both on above-average
  baselines → retrieval-precision failure: retrieved skills distract from a
  task-specific strategy the base agent would find unaided. Authors propose
  better routing or explicit fallback to unguided reasoning.
- **Gains ≠ more compute.** FrontierCS skill runs use more tokens/steps/tool
  calls, but per-task gain is essentially uncorrelated with extra usage
  (Spearman ρ = 0.006–0.015). Codex+skills Pareto-dominates Claude Code
  leaderboard entries (77.14 at 4.47M tokens vs 74.5 at 14.72M).
- **Quality-of-outcome effects.** PassNet: failed samples 14 → 5 (−64.3%),
  correctness 81.35% → 90.76%, and Codex+skills beats TorchInductor's AS
  Score (1.5313 vs 1.419). PassNet construction (Table 6): match failures
  10/50 → 2/50 → 1/50 across graph versions; solve time 40 → 22 min.
- **Construction-stage lesson (PassNet).** An overly absolute skill rule
  ("never reimplement vendor-optimized heavy ops") blocked a legitimately
  winnable conv3d stride==kernel_size case; error analysis of paired
  trajectories led to re-expressing the warning as a *defeasible prior*.
  Skills need escape hatches, not commandments.

## 7. Limitations

- **The headline library is not the measured treatment.** The four benchmark
  results come from *task-oriented / paper-derived* pools built per
  benchmark; the 1,000-repo AREX-Skill Library is an exhibited artifact of
  the same procedure, not something evaluated end-to-end on these suites.
- **Construction budget is uncounted.** Skill construction ($40/repo; ≤24
  GPU-hours per MLE-bench task, incl. web research) is paid before runs and
  excluded from the matched-budget comparison — the reported gains are
  amortized-knowledge effects, and the paper does not measure how much of
  the MLE-bench gain comes from *knowledge acquisition during construction*
  (search, reading) vs. the *skill representation itself*.
- **Retrieval/routing not ablated.** No separate measurement of router
  precision at library scale (the related demystifying-skills literature
  suggests retrieval is a soft bottleneck; consistent with the 2/20
  regressions, but unmeasured here).
- **LLM-authored, LLM-judged.** Verification relies on generated
  assertion cases and model-classified check outcomes; the taxonomy pipeline
  is LLM-proposed with model judges. Human validation of skill content at
  5,353-skill scale is not reported.
- **Freshness/drift.** Skills are provenance-stamped to a source commit, but
  the paper doesn't quantify staleness handling across releases (the meta-
  skills refresh-repo-skill exist in the bundle; no maintenance evaluation).
- **Domain scope.** Everything is ML-research-shaped; transfer to other
  engineering domains is claimed by construction, not shown.

## 8. Open Questions / Ideas

- **Skill or search?** Run the no-skill MLE-bench baseline *with the same
  web-search budget the construction phase gets* — how much of +41.78 points
  survives? That would cleanly separate "knowledge distilled" from
  "knowledge acquired at all".
- **Defeasibility as a first-class field.** The PassNet conv3d lesson
  suggests skills should carry explicit *exception conditions* ("do X, unless
  <structural pattern>, then Y") — could be verified by mutation testing
  rules against evaluator evidence.
- **Cheap distillation frontier.** $40/repo buys GPT-5.5-xhigh distillation;
  what fraction of the downstream gain survives a 100× cheaper deterministic
  extractor (the question my re-implementation probes at toy scale)?
- **Router-free libraries.** If retrieval precision barely matters at pool
  100 (per the demystifying-skills paper), is the elaborate 20/178 taxonomy
  mainly for *humans* browsing the library? A flat embedding index + entry
  skills might match it.
- **Skill graphs vs. one big skill.** Are the routing/dependency links in G
  ever actually followed by downstream agents (progressive-disclosure trace
  analysis), or do agents read the entry skill and improvise?
- **Cross-harness portability at library scale.** The paper argues skills
  move across harnesses; a Claude Code run equipped with the same repo
  graphs would test the claim the demystifying paper made at n=1.
