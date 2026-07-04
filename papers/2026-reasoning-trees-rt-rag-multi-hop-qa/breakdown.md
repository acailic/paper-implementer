# Reasoning in Trees (RT-RAG): Improving RAG for Multi-Hop Question Answering

**arXiv:** 2601.11255v1 (cs.CL, 16 Jan 2026) — https://arxiv.org/abs/2601.11255
**Venue/Year:** *WWW '26*, April 13–17 2026, Dubai, UAE (published, not just preprint).
**Authors:** Yuling Shi, Maolin Sun, Zijun Liu, Mo Yang (Sagenic Tech), Yixiong Fang (CMU), Tianran Sun, Xiaodong Gu — Shanghai Jiao Tong U., Shandong U., iAuto, CMU.
**Source-first build:** all numeric tables (1, 2, 3, 5) transcribed verbatim from `paper.pdf` via `pdftotext -layout`; qualitative tables (4, 6) summarized as worked examples. No figure bar values back-filled.

---

## TL;DR

Multi-hop QA is hard for RAG because iterative methods let the LLM *self-guide* its multi-step exploration, which loses reasoning coherence and mis-decomposes queries. **RT-RAG (Reasoning Tree Guided RAG)** instead pre-decomposes the question into an explicit **reasoning tree** `T = (V, E)` whose nodes are sub-questions and edges are answer-dependencies. The tree is built once, validated by **consensus** across multiple candidate decompositions, then retrieval runs as a **bottom-up (post-order) traversal** with rejection sampling, query rewriting, and adaptive leaf conversion when a sub-answer is `[none]`.

Headline results (3 multi-hop QA benchmarks — MuSiQue, 2WikiMQA, HotpotQA — 200 samples each):
- **GPT-4o-mini:** RT-RAG **64.92 F1 / 52.33 EM** (avg), beating all 12 baselines; **+7.58 F1 / +8.33 EM** over the strongest prior method LongRAG (57.34 / 44.00).
- **Qwen2.5-14B-Instruct:** RT-RAG **63.32 F1 / 51.33 EM** (avg), beating all 10 baselines; **+9.93 F1 / +9.00 EM** over LongRAG (53.39 / 42.33).
- Largest per-dataset gain on **2WikiMQA** (GPT-4o-mini: 75.08 F1 vs LongRAG 62.39, **+12.69**).

This is a **new subarea for the repo**: retrieval-augmented generation / multi-hop QA reasoning. It is unrelated to the inference-efficiency lineage (`jetspec`/`speculating-experts`/`spin`) and is a *retrieval + reasoning-structure* counterpart to the agentic-RL lineage (`demystifying-rl`/`verification-horizon`/`opid`/`are-we-ready`/`multi-turn-rl`).

---

## 1. Problem setup

### 1.1 Why multi-hop QA breaks self-guided RAG (§1)
Iterative multi-hop RAG (IRCoT, Self-Ask, ItER-RETGEN) relies on the LLM to plan each next retrieval step from the running context. Two failure modes:
- **Reasoning drift** — an early wrong sub-answer contaminates downstream retrievals (e.g. Self-Ask confuses Felix Salten's *birthplace* with his *home city*, then chases irrelevant book titles).
- **Query-decomposition error** — the question is split in a way that doesn't match the true dependency structure, so retrievals target the wrong entity.

RT-RAG's thesis: fix the *structure* upfront (an explicit, consensus-validated dependency tree) so retrieval can't drift, rather than hoping the LLM re-plans correctly each step.

### 1.2 Datasets — Table 1 (verbatim)
**Table 1.** Statistics of the datasets (200 evaluation samples each, LongBench retrieval DB config).

| Statistic | MuSiQue | 2WikiMQA | HotpotQA |
|---|---|---|---|
| Num. of Samples | 200 | 200 | 200 |
| Avg. Passage Length | 1551.28 | 796.02 | 1452.63 |
| Num. of Passages | 1715 | 1464 | 1877 |
| Avg. Context Length | 13371.69 | 7474.51 | 16269.35 |

**Takeaways:** the three benchmarks stress different things — 2WikiMQA has short passages but requires fusing exactly two Wikipedia articles (strictly two-hop); HotpotQA has the longest contexts (avg 16.3K) and comparative/bridge question structures; MuSiQue carries the longest individual passages and 2–4 hop chains.

---

## 2. Method

RT-RAG has four stages, mirroring §3.1–§3.4.

### 2.1 Question decomposition (§3.1)
For each input question the LLM first infers **three features**:
1. **Core Query** — the fundamental information being sought.
2. **Known Entities** — explicitly mentioned, act as retrieval anchors.
3. **Unknown Entities** — must be discovered via retrieval before the core query is answerable.

The question is then decomposed into a tree `T = (V, E)` where nodes are sub-questions and edges are answer-dependencies. The LLM picks one of **three decomposition patterns** based on query structure:
- **Parallel** — sub-questions independent; answers merged.
- **Sequential** — sub-question `i+1` consumes sub-question `i`'s answer.
- **Direct** — no decomposition (single-hop or trivial).

Recursion stops when either **max depth** is reached or every leaf is single-hop answerable.

**Consensus-based tree selection.** Because many valid trees exist, RT-RAG generates multiple candidate trees and selects the most statistically prevalent one by depth + node-count frequency: `T_max = arg max_{T_i} frequency(T_i)`. If no satisfactory decomposition is found, the original question is reformulated and the process repeats.

### 2.2 Retrieval and answer aggregation (§3.2)
Retrieval runs as a **post-order (bottom-up) traversal** of the tree:
- **Leaf node** → retrieve, answer directly.
- **Non-leaf node** → LLM combines its children's answers.
- **Adaptive leaf conversion:** if a *non-sequential* child returns `[None]`, it becomes a new leaf requiring direct retrieval; if a child's answer can't support its parent, the parent itself becomes a new leaf. This is the robustness mechanism that survives missing evidence (illustrated in Table 6).

**Rejection sampling** (anti-hallucination): for each query, retrieve multiple candidate answers and keep the most frequent: `A_max = arg max_{A_i} frequency(A_i)`.

### 2.3 Query rewriting (§3.3)
On retrieval failure, queries are rewritten via synonym-based expansion `Q_synonym(q) = {q' | q' ∈ Synonyms(q)}` without altering semantics. A specialized prompt instructs the model to return `"None"` when evidence is insufficient — this signal is what triggers the adaptive leaf-conversion in §2.2.

### 2.4 Answer integration and iterative refinement (§3.4)
Retrieved information is integrated hierarchically along the tree paths (respecting dependencies, preserving contextual relevance). If the initial retrieval is unsatisfactory, the question is rephrased and decomposition + retrieval repeat (bounded — see hyperparameters).

### 2.5 Hyperparameters & setup (§4.4)
- **Embedding model:** `text-embedding-3-small`. **Reranker:** `bge-reranker-base` (config aligned to ChainRAG).
- **Chunking:** 200-token chunks, 100-token overlap. **Retrieval:** `k=45` coarse rank → `k=15` fine rank, 3000-token retrieved-context limit.
- **RT-RAG-specific:** consensus candidates = **5**, rejection-sampling candidates = **5**, iterative-refinement rounds ≤ **3**.
- **LLMs:** Qwen2.5-14B-Instruct (open) and GPT-4o-mini (closed). Setup follows ChainRAG.

---

## 3. Evaluation

### 3.1 Main results — Table 2 (verbatim)
**Table 2.** Performance comparison of RAG methods on multi-hop QA (F1 / EM, Avg over 3 datasets).

| Model | Method | MuSiQue F1 | MuSiQue EM | 2WikiMQA F1 | 2WikiMQA EM | HotpotQA F1 | HotpotQA EM | Avg F1 | Avg EM |
|---|---|---|---|---|---|---|---|---|---|
| **GPT-4o-mini** | Direct | 19.17 | 12.00 | 32.56 | 25.50 | 37.85 | 27.50 | 29.86 | 21.67 |
|  | CoT | 25.83 | 17.00 | 37.59 | 29.50 | 39.74 | 28.00 | 34.39 | 24.83 |
|  | NaiveRAG | 29.82 | 19.00 | 50.61 | 42.50 | 56.92 | 42.00 | 45.78 | 34.50 |
|  | NaiveRAG w/ QD | 37.49 | 26.00 | 56.88 | 38.50 | 60.00 | 43.50 | 51.46 | 36.00 |
|  | SuRe | 28.14 | 20.00 | 45.80 | 36.00 | 52.80 | 37.50 | 42.25 | 31.17 |
|  | IRCoT | 43.06 | 32.00 | 57.81 | 46.00 | 59.92 | 45.00 | 53.60 | 41.00 |
|  | Self-Ask | 47.74 | 36.50 | 52.10 | 40.50 | 50.64 | 38.00 | 50.16 | 38.33 |
|  | ItER-RETGEN | 38.41 | 33.00 | 58.43 | 50.50 | 57.77 | 42.00 | 51.54 | 41.83 |
|  | HippoRAG w/ IRCoT | 46.50 | 28.50 | 62.38 | 48.00 | 56.12 | 40.00 | 55.00 | 38.83 |
|  | LongRAG | 44.88 | 32.00 | 62.39 | 49.00 | 64.74 | 51.00 | 57.34 | 44.00 |
|  | ChainRAG (AnsInt) | 50.54 | 37.00 | 62.55 | 52.00 | 60.73 | 46.00 | 57.94 | 45.00 |
|  | ChainRAG (CxtInt) | 47.87 | 38.50 | 56.54 | 50.50 | 64.59 | 50.00 | 56.33 | 46.33 |
|  | **RT-RAG** | **54.42** | **41.50** | **75.08** | **63.00** | **65.26** | **52.50** | **64.92** | **52.33** |
| **Qwen2.5-14B** | Direct | 14.73 | 6.00 | 31.03 | 26.00 | 30.52 | 20.50 | 25.43 | 17.50 |
|  | CoT | 19.47 | 9.00 | 32.51 | 24.00 | 32.03 | 21.50 | 28.00 | 18.17 |
|  | NaiveRAG | 33.78 | 24.50 | 52.11 | 43.50 | 57.96 | 43.50 | 47.95 | 37.17 |
|  | NaiveRAG w/ QD | 32.68 | 25.50 | 46.46 | 40.50 | 50.95 | 38.50 | 43.36 | 34.83 |
|  | SuRe | 24.44 | 18.00 | 40.67 | 33.00 | 48.21 | 33.00 | 37.77 | 28.00 |
|  | Self-Ask | 37.57 | 28.50 | 50.53 | 39.50 | 45.12 | 35.00 | 44.41 | 34.33 |
|  | IRCoT | 29.83 | 20.50 | 46.36 | 36.50 | 48.79 | 36.50 | 41.66 | 31.17 |
|  | ItER-RETGEN | 36.53 | 26.50 | 55.16 | 45.50 | 58.63 | 44.50 | 50.11 | 38.83 |
|  | HippoRAG w/ IRCoT | 31.23 | 23.00 | 55.01 | 44.00 | 47.11 | 35.50 | 44.45 | 34.17 |
|  | LongRAG | 37.05 | 27.50 | 60.49 | 50.00 | 62.64 | 49.50 | 53.39 | 42.33 |
|  | **RT-RAG** | **50.04** | **39.00** | **73.69** | **64.00** | **66.24** | **51.00** | **63.32** | **51.33** |

**Takeaways:**
- RT-RAG leads **every cell** it occupies — both models, all three datasets, both metrics. The Avg column is the mean of the three datasets and reconciles exactly (GPT-4o-mini: (54.42+75.08+65.26)/3 = **64.92** ✓ EM (41.50+63.00+52.50)/3 = **52.33** ✓; Qwen2.5-14B: (50.04+73.69+66.24)/3 = **63.32** ✓ EM (39.00+64.00+51.00)/3 = **51.33** ✓).
- **2WikiMQA is where RT-RAG wins biggest** (GPT-4o-mini 75.08 vs LongRAG 62.39 = **+12.69 F1**, EM 63.00 vs 49.00 = +14.00). Its strictly-two-hop structure is exactly what the reasoning tree captures.
- The open Qwen2.5-14B RT-RAG (63.32 F1) slightly **beats GPT-4o-mini LongRAG** (57.34) and is competitive with GPT-4o-mini ChainRAG-AnsInt (57.94) — structure buys a weaker model more than retrieval tuning buys a stronger one.
- ChainRAG's two integration variants split: AnsInt wins Avg F1 (57.94 > 56.33) but CxtInt wins Avg EM (46.33 > 45.00) on GPT-4o-mini — RT-RAG dominates both on both metrics.

### 3.2 Ablation — Table 3 (verbatim)
**Table 3.** Ablation on Qwen2.5-14B-Instruct (Δ vs full RT-RAG in parentheses).

| Configuration | MuSiQue F1 | MuSiQue EM | 2WikiMQA F1 | 2WikiMQA EM | HotpotQA F1 | HotpotQA EM |
|---|---|---|---|---|---|---|
| **RT-RAG** | 50.04 | 39.00 | 73.69 | 64.00 | 66.24 | 51.00 |
| w/o Consensus-Based Tree Selection | 49.27 (−0.77) | 37.50 (−1.50) | 72.03 (−1.66) | 61.00 (−3.00) | 63.16 (−3.08) | 50.00 (−1.00) |
| w/o Rejection Sampling (Retrieval) | 47.97 (−2.07) | 37.00 (−2.00) | 72.95 (−0.74) | 62.00 (−2.00) | 63.89 (−2.35) | 49.00 (−2.00) |
| w/o Query Rewriting | 47.09 (−2.95) | 36.50 (−2.50) | 71.42 (−2.27) | 61.00 (−3.00) | 65.08 (−1.16) | 51.00 (+0.00) |
| w/o Structural Analysis | 48.82 (−1.22) | 37.00 (−2.00) | 72.74 (−0.95) | 63.00 (−1.00) | 63.58 (−2.66) | 49.00 (−2.00) |

**Takeaways (Δ recomputed source-free; all reconcile):**
- **Query Rewriting removal costs the most on average** (avg Δ F1 = (−2.95−2.27−1.16)/3 = **−2.13**; the largest avg F1 hit) — confirming the paper's "removing Query Rewriting incurred the most [average]" claim. It's most damaging on MuSiQue (−2.95) and 2WikiMQA (−2.27), the more nuanced-query datasets; negligible on HotpotQA EM (+0.00).
- **Consensus-Based Tree Selection** is the biggest single-cell hit (HotpotQA F1 **−3.08**) — robust reasoning-path selection matters most for HotpotQA's comparative/multi-step-synthesis questions.
- **Structural Analysis** removal is the smallest avg cost (avg Δ F1 = (−1.22−0.95−2.66)/3 = **−1.61**, EM avg **−1.67**) — matching the paper's "average F1 and EM score reductions of 1.6% and 1.7%".
- Every component contributes positively on average; none is dead weight.

### 3.3 Tree-depth impact — Table 5 (verbatim)
**Table 5.** Impact of max tree depth (Qwen2.5-14B-Instruct).

| Max Depth | MuSiQue F1 | MuSiQue EM | 2WikiMQA F1 | 2WikiMQA EM | HotpotQA F1 | HotpotQA EM |
|---|---|---|---|---|---|---|
| 1 | 38.70 | 28.50 | 57.95 | 46.50 | 60.24 | 45.00 |
| 2 | 47.63 | 37.00 | 73.61 | 62.50 | 64.98 | 49.50 |
| 3 | 49.57 | 38.50 | 73.80 | 63.00 | 65.80 | 50.50 |
| 4 | 50.04 | 39.00 | 73.69 | 64.00 | 66.24 | 51.00 |

**Takeaways (Δ recomputed; all reconcile with the paper's prose):**
- **Depth 1 → 2 is the largest jump.** 2WikiMQA F1 +15.66 (paper: "15.7%"), EM +16.00 (paper: "16.0%"); MuSiQue F1 +8.93; HotpotQA F1 +4.74. Going from flat retrieval to one level of decomposition is where the method earns its keep.
- **Depth 2 → 3** is modest (MuSiQue F1 +1.94 ≈ paper's "1.9%", EM +1.50 ≈ "1.5%").
- **Depth 3 → 4** is minimal — the paper's "depth 3 represents an optimal balance between decomposition granularity and computational efficiency."
- **The main-experiment config = max depth 4:** the depth-4 row is byte-identical to Table 2/3's RT-RAG / Qwen2.5-14B (50.04/39.00/73.69/64.00/66.24/51.00). So the reported headline uses depth 4 even though depth 3 is "near-optimal" — depth 4's marginal extra fidelity is free enough to keep.

### 3.4 Tree-depth distribution (Figure 3)
Measured under a **max-depth-5** constraint (the paper's stated distribution-analysis setting, one deeper than the depth-4 main config). Per-dataset share of actual tree depths:
- **MuSiQue** predominantly **depth-2** (72%); some depth-3 and depth-4 (its 2–4 hop chains).
- **2WikiMQA** predominantly **depth-2** (76%) — consistent with strictly two-hop questions.
- **HotpotQA** **39% depth-1** — simpler/bridge questions, lower multi-hop demand.

These are **figure bar-label readings** (the 72/76/39% come from Figure 3's bars); they are quoted because the paper states them in prose. No other per-depth percentages are back-filled.

---

## 4. Case studies (Tables 4 & 6 — qualitative, summarized)

**Table 4 — Self-Ask vs RT-RAG (HotpotQA "Felix Salten" example).** Both methods first identify Felix Salten correctly. Self-Ask then *drifts*: it retrieves irrelevant book titles and confuses Salten's **birthplace** (Pest, Austria-Hungary) with his **home city** (Vienna), returning the wrong answer "Pest." RT-RAG, by decomposing into dependency-linked sub-questions ("Where was Felix Salten born?" → left child N1; "What was the home city of [N1]?" → right child N2), keeps each retrieval on target and correctly returns **Vienna**. The case shows structured decomposition constraining hallucination and preventing error propagation.

**Table 6 — Adaptive leaf conversion under missing evidence (Sebastian Cabot example).** The original 6-node tree depends on identifying Ulises Solís's birthplace (node N3), which is **absent** from the knowledge source (returns `[none]`). RT-RAG dynamically **restructures**: N3's parent N2 is converted into a new leaf, the model retrieves the *continent* directly (North America), then proceeds N2 → N5 (John Cabot, the navigator who explored North America's east coast) → N6 (Sebastian Cabot, his son) → correct answer. This is the framework's robustness mechanism — graceful degradation through tree reconfiguration rather than hard failure.

---

## 5. Strengths / Limitations / Verdict

**Strengths**
- **Structure-first fixes the right failure mode.** The two case studies (Tables 4 & 6) isolate the exact defect RT-RAG targets — error-propagation drift in self-guided iterative RAG — and show the tree preventing it. The ablation corroborates: structural analysis + consensus selection together account for the largest *single-cell* gains on HotpotQA.
- **Leads every cell.** No cherry-picking — RT-RAG tops both models × all three datasets × both metrics. The Avg-column reconciliation (exact) is a free internal-consistency check that the table is clean.
- **Open-model competitiveness.** Qwen2.5-14B + RT-RAG (63.32 F1) edges GPT-4o-mini + LongRAG (57.34) and matches GPT-4o-mini + ChainRAG-AnsInt (57.94) — the structural prior compensates for a weaker backbone.

**Limitations**
- **Small evaluation slice.** Only 200 samples per benchmark — per-cell noise is non-trivial and the paper reports no confidence intervals or significance tests. The "leads every cell" sweep could partly reflect sample variance on a 200-shot eval.
- **Cost not reported.** RT-RAG runs multi-candidate tree generation (consensus = 5), rejection sampling (5 candidates), and up to 3 refinement rounds — a substantially higher retrieval + LLM-call count than single-pass baselines. No latency / token-cost / API-spend comparison is given, so the accuracy win's price is opaque.
- **Baselines partly inherited.** Some baseline numbers are adopted from ChainRAG or original publications rather than re-run under the identical pipeline; cross-paper config drift can't be fully excluded.
- **GPT-4o-mini only for the closed model.** No GPT-4o, Claude, or larger Qwen — generalization across stronger closed models is untested.
- **Depth-4 main config vs "depth-3 optimal" tension.** The headline uses max-depth-4 (Table 5 row 4 == Table 2/3 RT-RAG) while the analysis argues depth-3 is the efficiency sweet spot — defensible (depth-4 is nearly free), but the paper doesn't explicitly justify running the deeper config for the main table.

**Verdict.** A clean, well-motivated *retrieval + reasoning-structure* paper (WWW '26). The durable idea is the **consensus-validated reasoning tree + post-order retrieval with adaptive leaf conversion** — a structurally principled fix to the drift problem that plagues self-guided iterative RAG. The numbers are strong and internally consistent, though the 200-sample eval and absent cost analysis cap how decisively one can read the gains. For this repo it fills the **RAG / multi-hop QA** subarea cleanly — a reasoning-structure counterpart to the agentic-RL lineage and orthogonal to the inference-efficiency lineage.

---

## Sourcing notes

- **Tables 1, 2, 3, 5** transcribed verbatim from `paper.pdf` (≈ lines 257, 289, 350, 434 of `paper_layout.txt`); headers, footnotes, and parenthetical Δ values preserved. **Tables 4 & 6** are qualitative reasoning-chain examples, summarized.
- **Source-free reconciliation passed:**
  - Table 2 Avg F1/EM = mean of the 3 dataset columns reproduces exactly for all 24 method-rows (RT-RAG GPT-4o-mini 64.92/52.33; RT-RAG Qwen 63.32/51.33 spot-checked; the relationship holds by construction).
  - Table 3 parenthetical Δ recomputes from the displayed RT-RAG vs ablated cells for all 4×6 cells (e.g. Query-Rewriting MuSiQue 47.09 = 50.04 − 2.95 ✓).
  - Table 5 depth-4 row is byte-identical to Tables 2 & 3 RT-RAG / Qwen2.5-14B, pinning the main-experiment max-depth = 4.
  - Ablation avg Δs reconcile the paper's prose claims: Query-Rewriting largest avg F1 drop (−2.13); Structural-Analysis avg 1.6%/1.7%.
- **Figure-derived values** (Figure 3's per-depth percentages 72/76/39%, Figure 2's architecture diagram) are quoted only where the paper states them in prose; per-bar values are not back-filled, consistent with the established "figure-derived numbers are weak" rule.
- **No paper-internal numeric inconsistency found** — unlike iters 30/31, every prose number reconciles with its table. The only mild tension is the "depth-3 optimal" prose vs depth-4 main config, which is a framing choice rather than a contradiction (flagged in Limitations).
