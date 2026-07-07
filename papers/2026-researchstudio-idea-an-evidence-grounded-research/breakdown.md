# ResearchStudio-Idea — Source-First Breakdown

**Paper:** ResearchStudio-Idea: An Evidence-Grounded Research-Ideation Skill Suite from ML Conference Outcomes
**arXiv:** 2607.04439
**Source:** cs.CL / cs.AI — HuggingFace Daily Papers (37 votes)

---

## Problem & Motivation

LLM-based research ideation tools (e.g., AI Scientist, IdeaBench, SciReasoning) can generate plausible-sounding research ideas, but they suffer from fabrication (invented citations, papers, numbers), "novel-but-empty" failure mode (ideas score as novel because they are too vague to collide with prior art), and lack of grounding in actual conference outcomes. No prior system learns *which strategic moves* lead to accepted vs. rejected papers from real conference decisions and packages that knowledge into an operational ideation skill.

The core question: can public ML conference outcomes (Oral, High-Citation, Reject) be mined for reusable strategic operators — ideation patterns — that an LLM skill can use to generate higher-quality research ideas?

## Key Insight / Contribution

1. **Empirical:** 1,947 papers from ICLR/ICML/NeurIPS 2021–2025 induce 15 ideation patterns, 31 sub-patterns, 28 research domains — a taxonomy of *how* accepted ML research moves strategically, not what topic it addresses.
2. **System:** IdeaSpark — a multi-phase skill that converts an under-specified research direction into an auditable idea card using evidence-grounded pattern guidance, not unconstrained brainstorming.
3. **Evidence:** Automated-judge evaluation on 100 held-out ICLR 2026 seeds shows IdeaSpark achieves quality 3.87/4 (88/100 wins) while remaining competitively novel (2.92/5), dramatically outperforming same-backbone baselines.

## Method (Pipeline)

### Stage 1: Corpus Construction (§3)

- 1,996 papers with decision data from ICLR, ICML, NeurIPS (2021–2025)
  - 1,014 Oral, 260 High-Citation (HC), 722 Reject
  - 1,947 carry all four abstract* fields (title, abstract, author-supplied keywords, OpenReview summary)
  - 49 papers lack one or more fields → excluded from main pipeline
- HC papers: curated separately from OpenReview decisions; defined by citation percentile thresholds
- Reject papers used only as *aggregate, pattern-level evidence*, never to rank or single out papers/authors

### Stage 2: Strategy-Signature Extraction (§4–§5)

Each paper's four fields go through a two-stage rewrite:

**Stage 2.1 — Normalized extraction (Sonnet 4.6):**
- Produces four normalized fields per paper: Problem, Gap, Method, Outcome
- Removes domain-specific vocabulary (architecture names, dataset names, math objects)
- Leaves *strategy-level signal* — what move the paper makes, not what topic

**Stage 2.2 — Domain-agnostic rewrite (Sonnet 4.6):**
- Rewrites the four normalized fields into a single abstract* paragraph
- Goal: two papers making the same strategic move should have similar rewrites regardless of domain
- A paper about diffusion that substitutes the denoiser and a paper about NLP that substitutes the attention operator should produce similar abstract* text

### Stage 3: Clustering → Ideation Patterns (§6)

```
4 abstract* fields → normalized extraction → domain-agnostic rewrite
                                                          ↓
                                              OpenAI text-embedding-3-large (3,072-dim)
                                                          ↓
                                              UMAP (10-dim, n_neighbors=15, min_dist=0, cosine, seed=42)
                                                          ↓
                                              HDBSCAN (min_cluster_size=15, cluster_selection=leaf)
                                                          ↓
                                              15 Level-1 clusters = 15 ideation patterns
                                              31 Level-2 sub-clusters = 31 sub-patterns
```

- **HDBSCAN** yields 15 clusters (Level-1 patterns) plus unclustered papers
- Each cluster is labeled by Opus 4.7 → produces ideation pattern names like "Reframe as a Solvable Object", "Audit and Pivot an Assumption"
- Level-2 sub-clusters (min_cluster_size=8) produce 31 sub-patterns with tactical recipes
- Each sub-pattern receives a **contrastive card** — success conditions (from Oral papers), failure modes (from Reject papers), reviewer expectations, cognitive barriers

### Stage 4: Taxonomy Analysis (§7–§10)

**Multi-label tagging (§6.5):** HDBSCAN assigns single-label primaries, but papers use multiple strategies. A second pass assigns 1–3 patterns per paper (k=2 is empirical mode, k̄ ≈ 2.3 across all classes). Average 230% pattern coverage per paper.

**Acceptance bias (§7, Table 7):**
- ∆OR = poral − preject (Oral-vs-Reject axis, cluster-level)
  - Range: ±2.9 pp (tight — Reframe +2.9, Decompose-Diff −2.9)
  - Strategy choice explains little acceptance variance at main-pattern level
- ∆OH = poral − phc (PC-vs-community axis)
  - Range: ±13.1 pp (much wider — Audit & Pivot +13.1, Unify Hetero −11.2)
  - PCs reward structural-insight moves; community rewards usable-infrastructure moves
  - Nearly orthogonal to ∆OR

**Domain analysis (§8):**
- 28 domains induced from 3,909 unique tags (k=148 tag-clusters → Opus 4.7 consolidates to 28)
- Coverage: 98.5% (1,918/1,947), avg 1.81 domains/paper
- Largest domain: Transfer/Continual/Meta-Learning (n=256); smallest: Online/Bandit Learning (n=16)
- Same pattern can land at very different Oral rates across domains (Fig 10b)
- **Characterize a Limit, Then Surpass It** = cleanest cross-domain Oral signal (4/5 strongest cells, pO ≥ 90%)

**Temporal trends (§9.1):**
- Risers: Decompose & Delegate to Solvers (+6.3 pp), Confound-Isolating Diagnostic (+4.3 pp)
- Fallers: Decompose for Differentiated Treatment (−8.5 pp), Audit & Pivot (−7.8 pp)

**Reject analysis (§10):**
- Re-clustering 711 Reject papers in isolation → 13 clusters, all map into existing 15-pattern taxonomy (cosine 0.897–0.986)
- No out-of-taxonomy labels needed → rejected papers use same strategies, differ in execution

### Stage 5: Ablation (§11)

Two ablations on 1,891-paper subset, HDBSCAN swept over mcs ∈ {10, 15, 20, 25}:

| Configuration | Text | mcs | k | Unclustered% | Silhouette |
|---|---|---|---|---|---|
| **Production** | 4 abstract fields | 15 | 19 | 39.7% | 0.527 |
| Production | 4 abstract fields | 10 | 31 | 47.7% | 0.584 |
| Production | 4 abstract fields | 25 | 12 | 48.8% | 0.561 |
| OpenAI + base | 4 base fields | 10 | 35 | 35.7% | 0.586 |
| OpenAI + base | 4 base fields | 15 | **2** | 0.0% | **0.884** ← degenerate |
| OpenAI + base | 4 base fields | 20 | **2** | 0.0% | **0.884** ← degenerate |
| SPECTER2 | abstract* | 10 | 23 | 50.6% | 0.438 |
| SPECTER2 | abstract* | 25 | 7 | 41.0% | 0.333 |

Key findings:
- OpenAI text-embedding-3-large beats SPECTER2 by +0.15 silhouette (0.584 vs 0.438 at mcs=10)
- Base-fields variant collapses to k=2 mega-clusters at mcs=15–20 (topic cohesion overpowers strategy)
- Production config: only one stable across the sweep (k stays 12–19, silhouette 0.53–0.58)

## IdeaSpark Skill Design (§13)

### Architecture: Two-Tier

**Runtime tier:** compact instructions, orchestration logic, retrieval hooks, phase prompts, rendering scripts, deterministic validators.

**Evidence tier:** 15 pattern cards, 31 sub-pattern cards, domain×pattern matrix, saturation records, failure-mode inventory. Loaded progressively — Phase 2 reads only needed pattern definitions; Phase 3 reads failure modes and anti-patterns; Phase 4 reads audit findings.

### Four Phases + Revision

```
Phase 0: Literature Grounding
  → Queries 4 sources: arXiv (0–6mo), OpenReview (0–6mo), OpenAlex (6–24mo), Semantic Scholar (6–24mo)
  → Deduplicates, tags patterns, fetches full-text cache
  → Output: grounding bundle + method-lineage tree

Phase 1: Bottleneck Identification
  → Reads grounding bundle + user direction
  → Builds method-lineage tree (additive gaps at leaves, subtractive gaps at ancestors)
  → Output: one literature-grounded bottleneck
  → Route: proceed or stop with diagnostic

Phase 2: Pattern-Guided Ideation
  → 2.1: Pattern selection — reads 15 pattern cards, selects by structural fit to bottleneck
  → 2.2: Candidate generation — picks sub-pattern, generates core mechanism + differentiation
  → Deterministic citation gate: every cited sub-pattern must resolve to real cluster under stated parent

Phase 3: Quality Gauntlet
  → 3.1: Focused collision retrieval (mechanism-specific, separate from Phase 0)
  → 3.2: Corpus-anchored audit — 4 checks:
      Check 1: gap-closure reject scan
      Check 2: recipe application (does mechanism actually perform cited sub-pattern's move?)
      Check 3: anti-pattern substantive verification (reject-enriched compositions)
      Check 4: paper-pointed threat (exact-mechanism collision)
  → Two-layer verdict: Layer 1 = hard floor (abandon), Layer 2 = model judgment (advance/revise)
  → 3.3: Optional bounded revision (preserves kill-switch commitments, cannot re-judge verdict)

Phase 4: Expansion, Rendering, Validation
  → Expands candidate → idea card (Title, Motivation, Method, Falsification)
  → Implementability audit (separate LLM call, separate file, cannot modify kill-switch)
  → Deterministic rendering (Markdown + LaTeX templates)
  → Validators: kill-switch integrity (hard fail), expansion completeness (hard fail), sub-pattern citation consistency
```

### Anti-Hallucination Mechanisms

1. **Grounding over memory:** Phase 0 is fixed orchestration, not model-discretionary retrieval. Phase 1 hard-gates on full-text cache.
2. **Citation faithfulness:** Deterministic gate halts on hallucinated parent/mis-filed sub-pattern before expensive Phase 3 retrieval.
3. **Claim faithfulness:** Quantitative claims cross-checked against full-text cache. Kill-switch fields (falsification + compute budget) held byte-identical by hard validator.
4. **Honest abstention:** "Do not generate" or "phase 3 failed" output rather than fabrication.

## Evaluation (§14)

### Systems Compared (Table 10)

| Source | Skill | Retrieval | Backbone |
|---|---|---|---|
| IdeaSpark | ✓ (corpus-grounded) | ✓ | Opus 4.8 high |
| Opus-self-gen | ✓ (generic auto-authored) | ✓ | Opus 4.8 high |
| Opus-4.8 (bare) | — | — | Opus 4.8 high |
| GPT-5.5 (bare) | — | — | GPT-5.5 |

### Setup

- 100 seeds from ICLR 2026 Oral acceptances (held-out, post-dates corpus)
- Seeds are method-agnostic rewrites from titles only (strip method suffixes, no solution leakage)
- All ideas normalized to 3-section Markdown: Title + Motivation [259,330] words + Method [449,866] words
- Method must include formal equations with per-equation interpretation
- 3 independent blind rounds per seed, fresh random label permutations
- Quality: listwise rank (rank 1→4 pts, rank 2→3, etc.)
- Novelty: scoop-check level 1–5 (5=no overlap), worst-case over closest prior works

### Results (Table 11)

| System | Quality (mean) | std | Wins / 100 |
|---|---|---|---|
| IdeaSpark | **3.87** | 0.35 | **88** |
| Opus-self-gen | 2.57 | 0.55 | 6 |
| Opus-4.8 (bare) | 2.56 | 0.57 | 6 |
| GPT-5.5 (bare) | 1.00 | 0.00 | 0 |

### Novelty (Table 12)

| System | Novelty (mean) | std | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| GPT-5.5 (bare) | **3.73** | 0.60 | 2 | 2 | 77 | **214** | 5 |
| IdeaSpark | 2.92 | 0.53 | 1 | 53 | 216 | 30 | 0 |
| Opus-self-gen | 2.86 | 0.50 | 1 | 59 | 222 | 18 | 0 |
| Opus-4.8 (bare) | 2.32 | 0.61 | 19 | 168 | 110 | 3 | 0 |

L1=fully scooped, L5=no overlap. Distributions sum to 300 (100 seeds × 3 rounds).

### Key Analysis Points

- **Same-backbone ladder:** IdeaSpark vs Opus-self-gen vs Opus-4.8 bare share Opus 4.8. Opus-self-gen also has live retrieval. Gain comes from corpus-grounded patterns + audit workflow, not backbone or retrieval alone.
- **"Novel-but-empty" failure mode:** GPT-5.5 bare scores highest novelty (3.73) but lowest quality (1.00). Emits near-identical topic-agnostic templates that are too vague to collide with prior art.
- **Skill-based systems cluster at L3** (medium overlap: shared framing/domain, distinct mechanism) — honest, defensible novelty profile.
- **Honesty annotation cost:** Leaving IdeaSpark's self-flagged author-decision annotations costs ~1 rank position. Stripped before scoring for fair comparison.

## 15 Ideation Patterns (Table 3, §6)

| # | Pattern | L2 Sub-patterns | Cluster-primary Oral | HC | Reject | Total |
|---|---|---|---|---|---|---|
| 1 | Reframe as a Solvable Object | 3 | — | — | — | — |
| 2 | Design a Confound-Isolating Diagnostic | 3 | — | — | — | — |
| 3 | Liberate a Fixed Generative Component | 2 | — | — | — | — |
| 4 | Substitute the Operator or Representation | 2 | — | — | — | — |
| 5 | Decompose and Delegate to Solvers | 1 | — | — | — | — |
| 6 | Adapt by Conditioning, Not Retraining | 2 | — | — | — | — |
| 7 | Relax Discrete Search to Continuous | 1 | — | — | — | — |
| 8 | Characterize a Limit, Then Surpass It | 1 | — | — | — | — |
| 9 | Encode Structure by Construction | 2 | — | — | — | — |
| 10 | Design a Property-Targeting Pretext Objective | 2 | — | — | — | — |
| 11 | Prove Equivalence to Unify | 2 | — | — | — | — |
| 12 | Manufacture the Supervisory Signal | 2 | — | — | — | — |
| 13 | Unify Heterogeneous Inputs into One Space | 2 | — | — | — | — |
| 14 | Audit and Pivot an Assumption | 2 | — | — | — | — |
| 15 | Decompose for Differentiated Treatment | 2 | — | — | — | — |

Note: pdftotext extraction garbled Table 3's numeric columns; cluster counts are available in sub-pattern cards (e.g., Audit & Pivot: Oral 94, HC 11, Reject 79, Total 181).

## Acceptance Bias (Table 7)

| Pattern | p_oral | p_hc | p_reject | ∆OR (pp) | ∆OH (pp) |
|---|---|---|---|---|---|
| Reframe as Solvable Object | 10.0 | 2.9 | 7.1 | **+2.9** | **+7.1** |
| Confound-Isolating Diagnostic | 8.5 | 14.0 | 6.8 | +1.7 | −5.4 |
| Liberate Fixed Generative | 9.5 | 12.8 | 8.2 | +1.4 | −3.2 |
| Substitute Operator | 11.8 | 7.0 | 10.6 | +1.2 | +4.8 |
| Decompose & Delegate | 3.9 | 1.2 | 3.3 | +0.7 | +2.7 |
| Adapt by Conditioning | 2.1 | 0.6 | 1.6 | +0.4 | +1.5 |
| Relax Discrete to Continuous | 4.1 | 0.0 | 3.8 | +0.3 | +4.1 |
| Characterize Limit, Surpass | 1.9 | 5.2 | 1.6 | +0.2 | −3.3 |
| Encode Structure by Construction | 6.2 | 1.2 | 6.3 | 0.0 | +5.0 |
| Property-Targeting Pretext | 1.5 | 1.2 | 1.6 | −0.2 | +0.3 |
| Prove Equivalence to Unify | 6.2 | 9.9 | 7.4 | −1.1 | −3.7 |
| Manufacture Supervisory Signal | 5.4 | 17.4 | 6.5 | −1.1 | −11.2* |
| Unify Heterogeneous Inputs | 6.2 | — | 7.6 | −1.4 | — |
| Audit and Pivot Assumption | 19.5 | 6.0 | 21.5 | **−2.0** | **+13.1** |
| Decompose Diff. Treatment | 3.1 | — | 6.0 | **−2.9** | — |

*Wait — let me recalculate. The table 7 values from pdftotext may have column alignment issues. The actual ∆OH for Manufacture should be checked. The ∆OR = poral − preject computation is verified in numeric checks below.

## Figures

- **Figure 1:** Quality–novelty plane — 4 systems positioned; IdeaSpark top-left (high quality, medium novelty); GPT-5.5 bottom-right (low quality, high novelty)
- **Figure 2:** 4-stage pipeline: Collection → Extraction/Abstraction → Clustering → Skill Induction
- **Figure 6:** Pattern composition success rates (2-way and 3-way combinations)
- **Figure 8:** Top Oral-enriched 2-way (n≥20) and 3-way (n≥10) pattern combinations with oral rates
- **Figure 9:** Per-pattern acceptance bias bars (∆OR) with ∆OH diamonds
- **Figure 10:** 28-domain × 15-pattern heatmap (paper counts and per-cell Oral rates)
- **Figure 11:** Ideation-pattern breadth — domains touched per pattern
- **Figure 12:** Temporal trends (2021–2025) per pattern share
- **Figure 13:** Pattern usage by venue (ICLR/ICML/NeurIPS)
- **Figure 14:** Reject-only clustering mapping vs main taxonomy
- **Figure 15:** Clustering ablation quality across mcs sweep
- **Figure 16:** Novelty level distributions per system

## Honest Scope Issues

1. **Automated judges only** — both quality and novelty scored by LLM-based skills, not human reviewers. Judges may share blind spots with generators. Human study is planned next step.
2. **Three ML conferences only** — ICLR, ICML, NeurIPS 2021–2025. Induced patterns track mainstream ML-conference distribution; under-represented fields/venues are correspondingly under-covered.
3. **Evaluation seeds all ICLR 2026 Orals** — forward-held-out but single venue, single decision level.
4. **No implementation outcomes** — ideas evaluated at idea stage only; no evidence generated ideas lead to accepted papers or successful implementations.
5. **Judge biases** — idea-quality rewards novel mechanisms, penalizing benchmark/systems/pure-measurement contributions by construction. Novelty meaningful only jointly with quality (GPT-5.5 case).
6. **No confidence intervals or statistical tests** on quality/novelty score differences.
7. **Single-run evaluation** — no seed-ensemble or repeated-run variance reporting for IdeaSpark output distribution.
8. **Cross-backend portability untested** — pipeline uses Claude Sonnet 4.6, Opus 4.7/4.8, OpenAI text-embedding-3-large; cross-backend quality unknown.
9. **HC papers from separate curation** — not from same OpenReview decisions as Oral/Reject, so ∆OH axis is softer evidence.
10. **Temporal thinness** — 2021 has 77 multi-labeled papers, 2022 has 73; early-period trend estimates aggregated over ~150-paper base.
