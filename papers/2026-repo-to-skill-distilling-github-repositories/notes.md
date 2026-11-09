# Notes — Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills

> **Paper:** Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills
> **Authors:** Jianlyu Chen, Yuyang Hu, Hongjin Qian, Jiawei Liu, Wenqing Wei,
> Xiaolong Chen, Defu Lian, Zhicheng Dou, Chaozhuo Li, Qiwei Ye, Zheng Liu
> (BAAI, USTC, Renmin University, HK PolyU)
> **Year:** 2026 · **arXiv:** 2609.02749 · **Code:** https://github.com/VectorSpaceLab/AREX-Skill

---

## First pass (skim, 2026-11-09)

### Big picture

The paper says autonomous ML-research agents are missing a *third* component.
The standard view is agent = (model, harness). The model is broad-but-fixed,
the harness controls procedure but carries no domain content. What's missing
is **operational knowledge** — "the know-how that separates knowing a method
from making it work": which package implements a method, how it wants its
data laid out, which configs are sane, which pitfalls invalidate a run. That
knowledge exists in repos/papers but is written for humans and too big to
load mid-task.

Their answer: distill repos into **Agent Skills** (Anthropic-style SKILL.md +
references/ + scripts/), verify each one, and hand the agent this library as
"operating context" K. Agent = (Mθ, H, K).

### The system in one breath

**DisCo** is an agent with two modes over the same backbone+harness:
- **Creator mode** — runs "skill distillation": anchor z → scope → Q
  (capabilities) → ground → X (evidence) → construct → G̃ (candidate skill
  graph) → verify → (G, R) where R is a construction record (evidence used,
  checks run, unresolved gaps).
- **Researcher mode** — solves tasks τ = (q, D, E, g) with K drawn from the
  library via progressive disclosure (read the router/entry skill, open only
  the branch you need).

Two distillation forms differing only in the **anchor**:
- **Task-agnostic** (z = a source, e.g. a repo): "what does this repo make
  possible?" → long-lived reusable skills. Scaled over 1,000 popular ML repos
  → the **AREX-Skill Library**: 5,353 skills, 20 areas, 178 capability
  families, a two-level taxonomy + generated router.
- **Task-oriented** (z = a task τ): "what does solving this demand?" →
  discover sources for the capability gaps, distill those. Used for MLE-bench
  (one graph per competition), FrontierCS (one shared graph), PassNet (one
  benchmark-level graph).

### Headline numbers (all: GPT-5.5 backbone, Codex harness, matched run
budgets; skill construction is a separate one-time budget, done before runs)

| Benchmark | no skills | + AREX-Skill | relative |
|---|---|---|---|
| MLE-bench Any-Medal (75 tasks) | 31.11 | 72.89 | **+134.3%** |
| PaperBench avg replication (20 papers) | 29.45 | 39.59 | +34.4% |
| FrontierCS Agent Track score (188 tasks) | 70.63 | 77.14 | +9.2% |
| PassNet AS Score (200 samples) | 1.343 | 1.5313 | +14.0% |

On MLE-bench, Codex+skills (72.89) also beats the best public leaderboard
entry (Famou-Agent 2.0, 64.44) — "no new harness needed". Gains grow with
difficulty (High tier 13.33 → 62.22, i.e. ×4.67). On PaperBench, gains
concentrate on low-baseline papers (ftrl 1.50 → 17.17; rice 7.94 → 48.51),
and 2/20 tasks regress (retrieval distraction on already-good tasks).

### Skill structure (Eq. 6)

S = (SKILL.md | references/ | scripts/) — knowledge interface (the only
thing read up front; goals, concepts, SOP, routing), knowledge substrate
(loaded on demand), execution interface (callable wrappers). Per-repo graphs
G = (S, L): an entry skill (router-like: when does this package apply, setup
check, routes to component skills) + component skills (data prep / training /
eval / serving / troubleshooting), links are routing/dependency/composition.

### Verification (their "distillation ≠ summarization" claim)

- Assertion-backed usability cases; safe repo-native checks (examples, tests,
  CLI checks, tiny fixtures, smoke scripts) classified pass / skill_gap /
  native_fail / skip_unsafe / skip_not_selected. skill_gap → localized
  repair → rerun checks.
- Static gates: metadata, link integrity, self-containment, provenance
  (source commit), routing metadata, no local-path leakage, runtime files
  separate from construction artifacts.
- MLE-bench variant: execution feedback loop, refine context =
  F_{t+1} = (S_t, Summary(L_t), Analysis(R_t)) — Eq. 9.
- FrontierCS variant: paired trials vs no-skill baseline AND vs previous
  graph version G^{(t-1)}; revisions kept only if reproduced.
- PassNet story (lovely concrete example): a skill said "don't reimplement
  vendor-optimized heavy ops"; that absolute rule blocked a conv3d with
  stride==kernel_size that reduces to elementwise; revised to a *defeasible
  prior*. Table 6: match failures 10/50 → 1/50 across two graph versions.

### Cost

~$40 of GPT-5.5/GPT-5.6-sol (xhigh) per repository, × 1,000 repos ≈ $40k for
the library. Creator mode paid once per source; researcher mode pays only
for what a task opens. On FrontierCS, Codex+skills Pareto-dominates Claude
Code entries (77.14 score at 4.47M tokens vs 74.5 at 14.72M).

### Terms I don't fully understand yet / to dig into

- "AI4AI" in the title — skills *for* AI (agents), i.e. AI-facing skills?
  Presumably "AI for AI" — knowledge artifacts consumed by AI, authored by
  AI. Need to check if the paper defines it anywhere (subtitle usage).
- Capability families vs areas — the exact 20/178 split; how assignment
  rejects keyword-only matches.
- "construction record R" — what exactly survives into the published graph
  vs stays in construction artifacts?
- MLE-bench "descriptive skills only, no runnable scripts" — why the
  anti-replay design matters (they freeze skills so runs must re-derive
  implementation).
- GPT-5.6-sol — a backbone I don't know (paper-internal naming).
- How the "router" is generated and what the agent actually sees at t=0.
- Whether the 1,000-repo library itself is evaluated end-to-end (the four
  benchmarks use *task-oriented* / *paper-derived* pools, NOT the 1,000-repo
  library — need to confirm this read; if true the headline library is a
  byproduct, not the measured artifact).

### First-impression take

This is the "skills as infrastructure" paper: it doesn't propose a new
control loop, it proposes a *knowledge supply chain* (distill → verify →
route → progressively disclose) and shows the knowledge layer alone moves
benchmarks more than months of harness engineering. The MLE-bench number
(31.11 → 72.89 with the SAME agent) is the loudest single result. Adjacent
to our demystifying-skills paper (2608.14036): that paper asked *why* skills
work (procedural anchoring); this one asks *where skills should come from*
(distillation from repos, at scale, with verification).

---

## Second pass (deep read, 2026-11-09)

### The formalism, unpacked

- Task τ = (q, D, E, g): problem, data/materials, environment (tools +
  budget), goal. Solve = produce artifacts y meeting g in E.
- Standard agent A = (Mθ, H); loop a_t ~ πθ(a | τ, h_t, H), o_t = E(a_t).
- Their agent A_res = (Mθ, H, **K**), actions conditioned a_t ~
  πθ(a | τ, h_t, H, K). K = {S_1..S_m} — the *only* change is operating
  context. This is the paper's core experimental bet: hold (Mθ, H, budget)
  fixed, vary only K.
- Operational knowledge has two constituents: **capability** (methods, code,
  APIs packaged into invocable units — realized by scripts/ + references/)
  and **usage policy** (conditions, reasons, procedures governing use —
  realized by SKILL.md). Capability without policy = tools without selection
  criteria; policy without capability = advice without an interface. Nice
  distinction; it maps exactly onto the skill file layout.
- Declarative (what holds) vs operational (what to do) — the derivation gap
  they want to automate. Today that derivation is manual expert labor.

### Eq. 8 pipeline, stage by stage (the thing to re-implement)

z --scope--> Q --ground--> X --construct--> G̃ --verify--> (G, R)

| Stage | Task-agnostic (z = repo) | Task-oriented (z = τ) |
|---|---|---|
| scope | Source Understanding → Capability Identification | Task Decomposition → Capability Gap Analysis |
| ground | Knowledge Extraction (evidence from the source itself) | Source Discovery (evidence assembled by search) |
| construct | Tool Encapsulation + Skill Packaging | Skill Generation |
| verify | Skill Verification (assertion cases, native checks, static gates) | same closure |

Key asymmetry: task-agnostic *selects* evidence from a fixed source;
task-oriented *assembles* evidence from the open ecosystem. Verification is
common and is what "separates distillation from summarization" — no skill
admitted on the strength of sources alone; surviving gaps recorded in R
rather than hidden.

### Repo graph construction details (A.1) — the "repo-to-skill procedure"

1. Evidence boundary FIRST, before any install/execution: include/exclude
   map over source roots, docs, examples, tests, scripts, config; exclude
   generated files, build outputs, vendored deps, caches.
2. Private Python inspection env: minimal install plan; live checks of
   import names, versions, public signatures, CLI entry points, optional
   backends, small runtime behaviors. Docs establish intent; source + live
   inspection confirm API/runtime claims before they enter the graph.
3. Self-contained graph: entry skill (router-like: applicability, minimal
   setup/import check, routes) + component skills split by *likely future
   agent tasks*, not by directory names. Bundle examples/helpers as
   references or skill-owned scripts so the agent never reopens the repo.
4. Provenance: source commit, package version, dirty state, relative
   evidence paths. Routing metadata = canonical repo identity, skill id,
   taxonomy hash, routing status, exact assignments.
5. Verification pass over entry + components + references + scripts; native
   check taxonomy {pass, skill_gap, native_fail, skip_unsafe,
   skip_not_selected}; skill_gap → localized repair → rerun. Static gates
   incl. link integrity + no local-path leakage.

### Taxonomy induction (A.1) — surprisingly rigorous

Freeze repo summaries → LLM proposes full 2-level tree → 100 stable batches
of 10 repos, each batch gets (locator assigns provisional fits: exact /
acceptable / poor / no-fit) + (independent judge reviews tree structure:
missing capabilities, overlapping families, mixed axes, overloaded nodes,
catch-alls) → aggregator synthesizes reviews + deterministic stats (fit
rates, family loads, unused/overloaded families) → reviser writes full new
tree → schema + typed-diff validation → convergence gate (≥2 rounds, bounded
no-fit, no blockers, explicit stop). Only THEN are final assignments made
(classification separate from skill generation; rationale + confidence +
repo evidence required; keyword-only/dependency-only/optional-integration/
example-only matches rejected; unclassified allowed).

### Evaluation subtleties worth keeping

- MLE-bench: skills per competition are **descriptive only** (no runnable
  training scripts) — prevents "replaying a fixed solution script";
  exploration optimizes *learning progress* not medal (rapid diagnostic
  iterations within ≤24 GPU-hours construction, separate from ≤24 GPU-hours
  running).
- PaperBench: skill pool = 3–5 module skills from each of ≤10 related-work
  papers (636 paper skills from 153 papers); target paper + its artifacts
  excluded (no leakage); module recovery experiment must reconstruct a
  bounded result *without* the original implementation repo.
- FrontierCS: single 9-node/42-link recovery graph; entry skill activates
  only AFTER a focused initial attempt fails (activation timing is itself
  part of the verified design).
- PassNet: candidates first treated as *hypotheses*, validated on 50 paired
  training instances before adoption; then screening-and-disdispatch over 4k
  training set (~1 day) for revision evidence.
- The regressions: PaperBench 2/20 tasks worse (sample-specific-masks
  −5.07, stay-on-topic −4.52) — both had above-average baselines →
  retrieval-precision failure mode; authors suggest explicit fallback to
  unguided reasoning on poor skill match. Honest reporting, good.
- FrontierCS usage analysis: skill runs use more tokens/steps/tool calls,
  but per-task gain uncorrelated with extra usage (Spearman ρ ≈ 0.006–0.015)
  → gains are not "search harder", they're "search better".

### Confirmed reading (was a question in pass 1)

The four benchmark evaluations do NOT run against the 1,000-repo
AREX-Skill library — MLE-bench/FrontierCS/PassNet use task-oriented graphs
built per-benchmark under task constraints, PaperBench uses paper-derived
pools. The public 1,000-repo library is the *demonstration of scale* of the
same procedure, not the measured treatment. So the paper proves "distilled
skills, built under the same 4-stage procedure, transfer to a fixed agent"
and separately exhibits the library. "AI4AI" never seems to get a formal
definition in the text I extracted — it's the title's framing (skills
authored by AI for AI consumption).

### What I'd implement from this (CPU-only, no paid APIs)

The paper's pipeline is LLM-powered at every stage (scoping, grounding,
construction are GPT-5.5 calls; verification runs real checks). A faithful
*capability-level* re-implementation on CPU with zero API spend:

- Stage 1 **scope**: deterministic repo analysis — shallow clone, README +
  structure + AST mining of Python files → capability inventory (what
  workflows does this repo support: CLI entry points, public API surface,
  example/test scripts, setup metadata).
- Stage 2 **ground**: evidence extraction with provenance — every extracted
  claim tagged with (file, line/section) so the skill is
  provenance-grounded like the paper demands.
- Stage 3 **construct**: emit the three-layer artifact — SKILL.md
  (name/description/when-to-use/steps/pitfalls), references/ (extracted API
  notes, README-derived quickstart), scripts/ (wrapped repo commands or a
  generated smoke script). Deterministic rule-based distillation, labeled as
  such.
- Stage 4 **verify**: the paper's cheapest faithful analogue — static gates
  (self-containment, link/reference integrity, provenance present, no
  absolute local paths, required SKILL.md sections present) + a question
  answering check: test questions about the repo answerable *from the skill
  alone* (closed-book vs open-book against the repo as gold evidence). This
  is the "does the skill carry the knowledge" recovery test in miniature.

That's exactly the "small pipeline over one small real repo" the implementable
core calls for. Label the whole thing deterministic extraction — no LLM
distillation claim.

### Remaining open questions after pass 2

- How much of the +134.3% MLE-bench gain is *web-search-time knowledge
  gathering* (the construction phase does web research per competition)
  rather than skill *format*? The no-skill baseline presumably can't browse
  the competition page at all during runs — the paper blocks it for
  construction too, but construction still gets 24 GPU-hours + search. The
  matched-budget argument covers run time, not knowledge acquisition.
- Router quality is never ablated separately from skill content (the
  demystifying-skills paper suggests retrieval barely matters — consistent
  here: 2/20 regressions only).
- $40/repo × 1,000: what's the marginal skill quality vs a 10× cheaper
  single-pass summarizer? No cost ablation.
- The taxonomy pipeline is elaborate — is 20/178 better than a flat
  embedding index? No retrieval ablation either.
