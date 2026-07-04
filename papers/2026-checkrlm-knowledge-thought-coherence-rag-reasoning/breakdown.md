# CheckRLM: Effective Knowledge-Thought Coherence Checking in Retrieval-Augmented Reasoning — Source-First Breakdown

- **arXiv:** 2607.02262v1 (cs.CL, 2 Jul 2026)
- **Authors:** Dingling Xu, Ruobing Wang, Qingfei Zhao, Yukun Yan, Zhichun Wang, Daren Zha, Shi Yu, Zhenghao Liu, Shuo Wang, Xu Han, Maosong Sun (Beijing Normal University + Institute of Information Engineering CAS + Tsinghua University Institute for AI + Northeastern University). Code/data: `github.com/AI9Stars/CheckRLM`.
- **Subarea (new to repo):** **in-reasoning knowledge-coherence checking for Reasoning Language Models via RAG** — i.e. *when* to intervene in a long CoT (mid-chain, paragraph-granular, before errors cascade) rather than *whether* to retrieve (adaptive RAG) or *after* the chain (post-reasoning self-check). **First repo paper on mid-reasoning RAG intervention / error-accumulation mitigation in long-CoT RLMs.** Sibling-in-spirit to the agentic-RL/search-during-reasoning lineage (Search-o1, reasoning-trees-rt-rag, evidence-state-rewards-long-context-reasoning) but uniquely targets **factual-error propagation** with **localized token-level correction** of the chain itself, not retrieval-augmented answer synthesis or reward shaping.
- **Source files:** `paper.pdf` (10pp, 1.1MB), `paper_layout.txt` (`pdftotext -layout`, 1541 lines). All numbers below are prose-/table-/equation-confirmed against `paper_layout.txt`. The paper has **15 explicit tables (T1 main, T2 DPO-short, T3 DPO-data-stats, T4 comprehensive, T5 DPO-full, T6 data-composition, T7 ablation, T8 summary-injection, T9 post-vs-in, T10 retrievers, T11/T12 cost, T13–T15 case studies) + 7 figures**. Figure bar/curve values (Figs 3/4/5/6/7) are NOT back-filled as cells; only the Figure-3/4 values that also appear verbatim in Tables 9/11/12 are quoted (cross-figure-table consistency), per the universal "figure-derived numbers are weak" rule.

---

## 1. The problem (motivation)

Reasoning Language Models (RLMs: OpenAI-o1, DeepSeek-R1, QwQ-32B, Qwen3) extend the reasoning chain, which **propagates factual errors**: an early wrong premise (e.g. wrong director name) becomes the foundation for downstream steps, so the final answer is wrong even if later reasoning is locally correct — **error accumulation** (Ling et al. 2023; Tyen et al. 2024). Figure 1 illustrates: Direct Reasoning hallucinates "Jim Abrahams" as director → wrong birth date → wrong answer "Feb 25, 1940".

Two existing fixes both fail (Figure 1):

- **Direct Reasoning** — relies solely on parametric knowledge; no external check; the early hallucination propagates unchecked.
- **Post-reasoning Check** — revises *after* the full chain is generated; it can fix the director name but **cannot repair the already-corrupted intermediate reasoning** built on top of the error. Late correction addresses only a subset of errors.

The gap CheckRLM fills: **timely, in-process** correction that kills errors *before they cascade*, while keeping intervention **minimal** (token-level, paragraph-granular) so the reasoning structure stays intact.

---

## 2. Method (§3)

### 2.1 Preliminary (Eq 1–2)

A reasoning chain is a sequence `τ = (s_1, …, s_T)` where each step `s_t ∼ P_θ(s_t | q, s_<t)` (Eq 1, product of stepwise conditionals). The final answer `a ∼ P_θ(a | q, R)` (Eq 2). CheckRLM's goal: generate a correct `a` by checking/refining `R`.

### 2.2 Knowledge–Thought Coherence Checking — two stages

**Stage 1 — In-Process Knowledge Claim Recognition (§3.2.1, Eq 3).** An "intermediate intervention" strategy: instead of checking only after the full chain, extract explicit factual claims **during** reasoning. The chain is segmented into **paragraph-level reasoning-chain units** `r_t` (paragraph granularity chosen because token/sentence-level checks "disrupt the logic and semantic meaning … and trigger unnecessary or frequent interruptions"). Input = original question `q` + current unit `r_t` (NOT the full `R_<t`, to avoid noise from earlier steps). A claim-recognition model `M_rec` samples a claim set:

```
y_t^claim ∼ M_rec(· | Instruct_r, r_t, q)        (Eq 3)
```

`y_t^claim = {y_t^1, …, y_t^n}` are factual claims relevant to `q`. `M_rec` feeds a fact-verification step that prioritises question-relevant facts and flags explicit errors for correction before they propagate.

**Stage 2 — Localized Knowledge Coherence Correction via Retrieval (§3.2.2, Eq 4–6).** For each unit `r_t`:

1. Build query set `Q_t = {q} ∪ y_t^claim` (original question ∪ extracted claims).
2. Each `q_i ∈ Q_t` triggers an **independent atomic retrieval** (top-k=3). Union of results → candidate set `D_t^raw`.
3. **Deduplicate** → refined doc set `D_t = ⋃_{q_i ∈ Q_t} Retriever(q_i)` (Eq 4).
4. Feed `D_t + r_t` into the **knowledge correction model** `M_cor`, which samples a corrected unit:

```
r_t' ∼ M_cor(· | Instruct_c, r_t, D_t)            (Eq 5)
```

`Instruct_c` enforces **token-level corrections with minimal cost** while keeping the reasoning structure intact. Two preference-guidance rules: (a) if all facts in `r_t` are correct OR `D_t` is irrelevant → output `r_t` unchanged; (b) if `r_t` has errors → targeted token-level corrections with minimal disruption. Define `r̃_t ∈ {r_t, r_t'}` (the unit's two possible states). The full refined chain is the concatenation:

```
R = r̃_1 ⊕ r̃_2 ⊕ … ⊕ r̃_t                        (Eq 6)
```

After correction, the RLM **continues inference from the corrected unit** — proceeding along a correct path and preventing accumulation. Multiple reason↔correct iterations yield the final chain, then the answer.

### 2.3 Optimization (§3.2.3, Eq 7) — joint DPO of recognition + correction

`M_rec` and `M_cor` suffer in a limited number of cases (low-quality claims, failed corrections, redundant outputs). Both are improved via **DPO** (Rafailov et al. 2023). A **single model `M_θ^RC`** serves both recognition and correction (unified policy θ). Objective:

```
L_DPO(M_θ^RC; M_ref^RC) = − E_{x,y+,y−∼D} [ log σ( β·log(M_θ^RC(y+|x)/M_ref^RC(y+|x)) − β·log(M_θ^RC(y−|x)/M_ref^RC(y−|x)) ) ]    (Eq 7)
```

DPO data `D_DPO` (Appendix A, 2500 samples from the 2WikiMQA train set) has two subsets:
- `D_KCR` (claim recognition): {x_t, y^claim(+), y^claim(−)}, 2351 samples.
- `D_KCC` (coherence correction): {x_t, r̃_t(+), r̃_t(−)}, 2960 samples.
- Total `D_DPO` = **5311 samples** (Table 3).

Positive/negative labels annotated by **GPT-4o-mini** (candidates generated with temp/top_p ∈ {0.1, 0.5, 0.9}; only the first 3 reasoning units sampled since corrections concentrate there). DPO config: batch 8, lr 5e-7, β 0.1, 1 epoch.

---

## 3. Experimental setup (§4)

- **Reasoning RLMs:** primary **QwQ-32B**; also Qwen3-8B, Qwen3-32B, DeepSeek-R1-Distill-Llama-70B.
- **Recognition/correction models `M_rec/M_cor` (same underlying model):** Qwen3-8B, Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct, Llama-3.3-70B-Instruct. **Main results (Table 1) use Llama-3.3-70B-Instruct as the checker.**
- **Datasets:** multi-hop QA — HotpotQA, 2WikiMultiHopQA (2WikiMQA), MuSiQue, IIRC; short-form QA — SimpleQA (500 test samples, KILT corpus).
- **Retriever:** BM25 (sparse) + bge-large-en-v1.5 (dense) on multi-hop; bge-large-en-v1.5 only on SimpleQA. top-k = 3. Max recognition-and-correction steps = 10. Max context = 16384 tokens.
- **Metrics:** F1 (f1), Exact Match (em).
- **Hardware:** vLLM on 8×A800 GPUs.
- **Baselines:** Direct Reasoning; Vanilla RAG; ReAct; FLARE; Self-RAG; RAT (multi-step RAG representative); Search-o1 (coupled retrieval–reasoning).

---

## 4. Results — all tables verbatim

### Table 1 — Overall performance (%) [Llama-3.3-70B-Instruct checker] (L364–385)

| RLM / Method | HotpotQA f1 | em | 2WikiMQA f1 | em | MuSiQue f1 | em | IIRC f1 | em | SimpleQA f1 | em | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **QwQ-32B** | | | | | | | | | | | |
| Direct Reasoning | 38.4 | 29.0 | 34.6 | 28.4 | 18.5 | 7.4 | 24.8 | 21.4 | 10.5 | 4.6 | **21.8** |
| Vanilla RAG | 52.7 | 42.2 | 46.4 | 42.6 | 19.3 | 10.0 | 25.0 | 22.0 | 31.4 | 24.2 | **31.6** |
| ReAct (2022) | 48.2 | 34.6 | 45.9 | 33.0 | 22.3 | 10.0 | 21.9 | 16.2 | 30.1 | 22.2 | **28.4** |
| FLARE (2023) | 43.5 | 32.2 | 46.6 | 38.4 | 18.4 | 9.2 | 12.3 | 9.6 | 24.1 | 17.4 | **25.2** |
| Self-RAG (2024) | 49.7 | 39.9 | 40.0 | 36.2 | 20.0 | 10.6 | 25.1 | 21.3 | 31.1 | 24.8 | **29.9** |
| RAT (2024) | 51.5 | 39.0 | 47.2 | 38.6 | 24.1 | 13.2 | 21.7 | 17.0 | 30.3 | 22.2 | **30.5** |
| Search-o1 (2025) | 62.0 | 49.2 | 71.4 | 60.4 | 33.3 | 20.7 | 29.2 | 25.0 | 35.4 | 27.4 | **41.5** |
| **CheckRLM** | **66.3** | **52.6** | **73.4** | **62.0** | **39.6** | **27.2** | **33.1** | **29.0** | **40.0** | **30.4** | **45.4** |
| **Qwen3-32B** | | | | | | | | | | | |
| Direct Reasoning | 36.4 | 27.6 | 34.0 | 29.4 | 16.6 | 6.6 | 23.0 | 20.2 | 9.5 | 2.4 | **20.6** |
| Vanilla RAG | 51.2 | 41.0 | 44.5 | 41.8 | 19.7 | 10.4 | 22.5 | 19.8 | 29.3 | 22.8 | **30.3** |
| **CheckRLM** | **64.2** | **52.0** | **70.8** | **60.8** | **35.6** | **25.4** | **31.6** | **27.8** | **39.1** | **30.6** | **43.8** |
| **Qwen3-8B** | | | | | | | | | | | |
| Direct Reasoning | 29.3 | 22.2 | 31.1 | 26.6 | 12.9 | 5.2 | 21.0 | 18.2 | 7.7 | 2.8 | **17.7** |
| Vanilla RAG | 44.5 | 35.2 | 40.9 | 38.4 | 13.8 | 6.0 | 19.8 | 16.8 | 29.4 | 23.0 | **26.8** |
| **CheckRLM** | **61.9** | **49.6** | **69.7** | **58.6** | **35.0** | **24.0** | **29.7** | **25.4** | **35.7** | **28.6** | **41.8** |

> **⚠ Avg-column denominator (honest scope).** `Avg` = the mean of **all 10 cells** (5 f1 + 5 em), NOT the f1-only mean. Verified: QwQ-32B Direct Reasoning (38.4+29.0+34.6+28.4+18.5+7.4+24.8+21.4+10.5+4.6)/10 = **21.76 → 21.8** ✓; CheckRLM QwQ-32B = 453.6/10 = **45.36 → 45.4** ✓; all 9 reported Avg cells reproduce within ±0.1 (full-precision display rounding, e.g. Search-o1 414.0/10 = 41.4 → shown 41.5). The f1-only mean would run ~3–4 pp higher (CheckRLM QwQ-32B f1-only = (66.3+73.4+39.6+33.1+40.0)/5 = **50.5**, not 45.4). Cite the column as the 10-cell mean.

### Table 2 — DPO training results (%) [Qwen2.5-14B-Instruct checker, f1 metric] (L415–430)

| RLM / Method | Hot. | 2Wiki. | Simp. |
|---|---|---|---|
| **QwQ-32B** | | | |
| Vanilla RAG | 52.7 | 46.4 | 31.4 |
| Inference Only | 61.6 | 65.5 | 36.0 |
| DPO Training | 63.2 | 71.2 | 36.9 |
| **Qwen3-32B** | | | |
| Vanilla RAG | 51.2 | 44.5 | 29.3 |
| Inference Only | 57.0 | 62.9 | 34.2 |
| DPO Training | 61.3 | 65.3 | 36.1 |

> **⚠ §5.2 "DPO yields a 5.7% improvement" is the in-domain 2WikiMQA delta (honest scope).** 5.7 = DPO−InferenceOnly on **2WikiMQA f1, QwQ-32B** (71.2 − 65.5 = 5.7 ✓). The DPO training data is sampled from the 2WikiMQA train set (Appendix A, 2500 samples), so the gain is **in-domain-concentrated**: on HotpotQA the gain is only +1.6 (61.6→63.2) and on SimpleQA +0.9 (36.0→36.9). The single "5.7%" headline does not generalise uniformly across datasets.

### Table 3 — DPO training-data statistics (L834–836)

| Dataset | # Sample |
|---|---|
| D_KCR (claim recognition) | 2351 |
| D_KCC (coherence correction) | 2960 |
| D_DPO (total) | 5311 |

> Check: 2351 + 2960 = 5311 ✓ (the total is the sum of the two subsets).

### Table 4 — Comprehensive benchmark (%) across RLM × checker pairs (L1212–1237)

| RLM / Method | Hot f1 | em | 2Wiki f1 | em | Mus f1 | em | IIRC f1 | em | Simp f1 | em |
|---|---|---|---|---|---|---|---|---|---|---|
| **QwQ-32B** | | | | | | | | | | |
| Direct Reasoning | 38.4 | 29.0 | 34.6 | 28.4 | 18.5 | 7.4 | 24.8 | 21.4 | 10.5 | 4.6 |
| Vanilla RAG | 52.7 | 42.2 | 46.4 | 42.6 | 19.3 | 10.0 | 25.0 | 22.0 | 31.4 | 24.2 |
| CheckRLM − Check@Qwen-3-8B | 59.9 | 47.4 | 59.9 | 50.6 | 29.0 | 17.8 | 29.0 | 25.4 | 33.4 | 25.2 |
| CheckRLM − Check@Qwen-2.5-14B | 61.6 | 48.6 | 65.5 | 54.8 | 33.3 | 21.2 | 30.7 | 26.0 | 36.0 | 27.3 |
| CheckRLM − Check@Qwen-2.5-32B | 59.4 | 46.6 | 66.3 | 56.6 | 33.1 | 19.8 | 30.9 | 26.2 | 34.0 | 25.2 |
| CheckRLM − Check@Llama-3.3-70B | 66.3 | 52.6 | 73.4 | 62.0 | 39.6 | 27.2 | 33.1 | 29.0 | 40.0 | 30.4 |
| **Qwen3-32B** | | | | | | | | | | |
| Direct Reasoning | 36.4 | 27.6 | 34.0 | 29.4 | 16.6 | 6.6 | 23.0 | 20.2 | 9.5 | 2.4 |
| Vanilla RAG | 51.2 | 41.0 | 44.5 | 41.8 | 19.7 | 10.4 | 22.5 | 19.8 | 29.3 | 22.8 |
| CheckRLM − Check@Qwen-2.5-14B | 57.0 | 46.8 | 62.9 | 50.2 | 29.9 | 20.0 | 26.5 | 23.2 | 34.2 | 26.2 |
| CheckRLM − Check@Llama-3.3-70B | 64.2 | 52.0 | 70.8 | 60.8 | 35.6 | 25.4 | 31.6 | 27.8 | 39.1 | 30.6 |
| **Qwen3-8B** | | | | | | | | | | |
| Direct Reasoning | 29.3 | 22.2 | 31.1 | 26.6 | 12.9 | 5.2 | 21.0 | 18.2 | 7.7 | 2.8 |
| Vanilla RAG | 44.5 | 35.2 | 40.9 | 38.4 | 13.8 | 6.0 | 19.8 | 16.8 | 29.4 | 23.0 |
| CheckRLM − Check@Qwen-2.5-32B | 56.3 | 45.0 | 60.0 | 50.0 | 28.2 | 18.0 | 25.0 | 22.2 | 33.0 | 26.2 |
| CheckRLM − Check@Llama-3.3-70B | 61.9 | 49.6 | 69.7 | 58.6 | 35.0 | 24.0 | 29.7 | 25.4 | 35.7 | 28.6 |

> **⚠ Checker-size scaling is non-monotone (honest scope).** For QwQ-32B reasoning, Avg-implied performance by checker is Llama-3.3-70B (45.4) > Qwen-2.5-14B (40.5) > Qwen-2.5-32B (39.8) > Qwen-3-8B (37.8). The **Qwen-2.5-32B checker (39.8) underperforms the smaller Qwen-2.5-14B checker (40.5)** on the QwQ-32B reasoner — bigger checker is NOT always better. The headline results rely on the **70B checker**, which is ~9× the size of the smallest reasoner evaluated (Qwen3-8B); the cost of running a 70B checker alongside the reasoner is not folded into the per-query token/time figures (which count the reasoner only).

### Table 5 — DPO training, full grid (%) (L1244–1257)

| RLM × Checker (Qwen-2.5-14B) | Hot f1 | em | 2Wiki f1 | em | Mus f1 | em | IIRC f1 | em | Simp f1 | em |
|---|---|---|---|---|---|---|---|---|---|---|
| **QwQ-32B** | | | | | | | | | | |
| Vanilla RAG | 52.7 | 42.2 | 46.4 | 42.6 | 19.3 | 10.0 | 25.0 | 22.0 | 31.4 | 24.2 |
| Inference Only | 61.6 | 48.6 | 65.5 | 54.8 | 33.3 | 21.2 | 30.7 | 26.0 | 36.0 | 27.3 |
| DPO Training | 63.2 | 50.0 | 71.2 | 58.4 | 36.2 | 22.0 | 32.2 | 28.0 | 36.9 | 28.0 |
| **Qwen3-32B** | | | | | | | | | | |
| Vanilla RAG | 51.2 | 41.0 | 44.5 | 41.8 | 19.7 | 10.4 | 22.5 | 19.8 | 29.3 | 22.8 |
| Inference Only | 57.0 | 46.8 | 62.9 | 50.2 | 29.9 | 20.0 | 26.5 | 23.2 | 34.2 | 26.2 |
| DPO Training | 61.3 | 48.6 | 65.3 | 54.8 | 32.7 | 20.2 | 31.0 | 26.6 | 36.1 | 28.8 |
| **DeepSeek-R1-Distill-Llama-70B** | | | | | | | | | | |
| Vanilla RAG | 56.1 | 43.0 | 49.4 | 44.4 | 25.1 | 14.6 | 26.1 | 22.6 | 32.0 | 25.0 |
| Inference Only | 60.9 | 47.4 | 64.1 | 53.4 | 32.3 | 20.8 | 29.0 | 23.0 | 35.9 | 27.9 |
| DPO Training | 62.7 | 49.2 | 65.7 | 54.4 | 31.7 | 20.0 | 31.2 | 26.0 | 36.8 | 27.8 |

> Cross-table consistency: QwQ-32B + Qwen-2.5-14B and Qwen3-32B + Qwen-2.5-14B rows here are byte-identical to the corresponding Table 2 rows (Hot/2Wiki/Simp f1) and Table 4 rows ✓.

### Table 6 — Training-data composition ablation (%) [QwQ-32B × Qwen-2.5-14B] (L1262–1270)

| Composition | Hot f1 | em | 2Wiki f1 | em | Mus f1 | em | IIRC f1 | em | Simp f1 | em |
|---|---|---|---|---|---|---|---|---|---|---|
| D_KCR Only | 59.9 | 48.0 | 67.9 | 56.2 | 33.7 | 21.2 | 31.2 | 27.0 | 36.2 | 27.4 |
| D_KCC Only | 60.3 | 47.6 | 68.9 | 56.0 | 33.8 | 20.6 | 29.9 | 25.6 | 34.1 | 24.0 |
| D_KCR + D_KCC | 63.2 | 50.0 | 71.2 | 58.4 | 36.2 | 22.0 | 32.2 | 28.0 | 36.9 | 28.0 |

> Combined (D_KCR+D_KCC) ≥ each single subset on every f1 cell and most em cells (only IIRC em D_KCR-Only 27.0 ties the combined 28.0 — combined still ≥). Confirms §C.2 "combined data outperforms training on single-type data"; D_KCR (recognition) brings the larger marginal gain, attributed to "higher-quality factual claims enable … more accurate correction."

### Table 7 — Constrained ablation (%) (L1278–1291)

| RLM × Checker | Variant | Hot f1 | em | 2Wiki f1 | em | Mus f1 | em | IIRC f1 | em | Simp f1 | em |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B × Qwen2.5-32B | CheckRLM | 56.3 | 45.0 | 60.0 | 50.0 | 28.2 | 18.0 | 25.0 | 22.2 | 33.0 | 26.2 |
| | − w/o Verification | 45.3 | 36.8 | 54.4 | 47.0 | 20.1 | 10.8 | 24.2 | 20.8 | 24.3 | 17.0 |
| | − w/o Refinement | 50.2 | 39.8 | 56.0 | 48.2 | 26.2 | 15.2 | 24.1 | 20.4 | 32.3 | 24.8 |
| QwQ-32B × Qwen2.5-14B | CheckRLM | 61.6 | 48.6 | 65.5 | 54.8 | 33.3 | 21.2 | 30.7 | 26.0 | 36.0 | 27.3 |
| | − w/o Verification | 52.2 | 40.2 | 62.0 | 54.2 | 30.2 | 18.6 | 29.1 | 25.2 | 30.6 | 21.6 |
| | − w/o Refinement | 58.0 | 45.4 | 64.2 | 53.0 | 29.7 | 18.6 | 28.2 | 23.0 | 33.3 | 25.8 |
| QwQ-32B × Llama-3.3-70B | CheckRLM | 66.3 | 52.6 | 73.4 | 62.0 | 39.6 | 27.2 | 33.1 | 29.0 | 40.0 | 30.4 |
| | − w/o Verification | 62.9 | 50.4 | 72.6 | 60.8 | 39.4 | 26.0 | 33.0 | 28.6 | 37.7 | 28.4 |
| | − w/o Refinement | 64.9 | 52.4 | 68.0 | 56.8 | 37.5 | 23.6 | 32.7 | 28.4 | 36.8 | 29.0 |

> Both modules contribute. "w/o Verification" = remove the claim-recognition check (still retrieve+correct but no claim extraction); "w/o Refinement" = recognise but do not correct. Verification matters more on Hot/MuSiQue/SimpleQA; Refinement matters more on 2WikiMQA (QwQ+Llama: 73.4 → 68.0 without refinement vs → 72.6 without verification). The two modules are interdependent (§C.3) so neither can be fully removed — these are "simplest versions," not pure deletions.

### Table 8 — Retrieved-summary injection (%) [QwQ-32B × Llama-3.3-70B] (L1299–1304)

| Variant | Hot f1 | em | 2Wiki f1 | em | Mus f1 | em | IIRC f1 | em | Simp f1 | em |
|---|---|---|---|---|---|---|---|---|---|---|
| CheckRLM | 66.3 | 52.6 | 73.4 | 62.0 | 39.6 | 27.2 | 33.1 | 29.0 | 40.0 | 30.4 |
| − w/ Summary | 66.2 | 52.0 | 71.5 | 59.6 | 39.8 | 25.2 | 33.0 | 28.0 | 38.5 | 29.8 |

> Injecting summarized retrieved content (instead of minimal token-level correction) slightly **hurts** on 8/10 cells (only MuSiQue f1 +0.2). Justifies the minimal-correction design rationale (§C.5: the RLM "exhibits a clear and coherent problem-solving approach, and … additional information may interfere with its inherent reasoning logic"). Single RLM×checker row only (narrow evidence base).

### Table 9 — Post-reasoning Check vs In-reasoning Check (%) (L1311–1327)

| RLM × Checker | Variant | Hot f1 | em | 2Wiki f1 | em | Mus f1 | em | IIRC f1 | em | Simp f1 | em |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B × Qwen2.5-32B | Post-reasoning | 46.8 | 36.8 | 49.1 | 42.6 | 17.9 | 8.2 | 24.3 | 20.4 | 30.7 | 22.6 |
| | In-reasoning | 56.3 | 45.0 | 60.0 | 50.0 | 28.2 | 18.0 | 25.0 | 22.2 | 33.0 | 26.2 |
| | % improv. | 9.5 | 8.2 | 10.9 | 7.4 | 10.3 | 9.8 | 0.7 | 1.8 | 2.3 | 3.6 |
| QwQ-32B × Qwen2.5-14B | Post-reasoning | 46.7 | 35.8 | 47.6 | 39.6 | 24.3 | 13.0 | 28.3 | 23.6 | 26.3 | 18.6 |
| | In-reasoning | 61.6 | 48.6 | 65.5 | 54.8 | 33.3 | 21.2 | 30.7 | 26.0 | 36.0 | 27.3 |
| | % improv. | 14.9 | 12.8 | 17.9 | 15.2 | 9.0 | 8.2 | 2.4 | 2.4 | 9.7 | 8.7 |
| QwQ-32B × Llama-3.3-70B | Post-reasoning | 58.8 | 47.0 | 60.2 | 52.2 | 31.5 | 19.4 | 32.2 | 27.6 | 38.3 | 29.0 |
| | In-reasoning | 66.3 | 52.6 | 73.4 | 62.0 | 39.6 | 27.2 | 33.1 | 29.0 | 40.0 | 30.4 |
| | % improv. | 7.5 | 5.6 | 13.2 | 9.8 | 8.1 | 7.8 | 0.9 | 1.4 | 1.7 | 1.4 |

> All `% improv.` cells recompute exactly as In−Post (e.g. QwQ+Qwen-14B Hot f1 61.6−46.7 = 14.9 ✓; QwQ+Llama 2Wiki 73.4−60.2 = 13.2 ✓). Cross-figure consistency: the QwQ-32B × Llama-3.3-70B Post/In 2WikiMQA f1 (60.2/73.4) and QwQ-32B × Qwen2.5-14B 2Wiki f1 (47.6/65.5) match the Figure 3a/3b bar heights, and the QwQ+Llama row matches Table 1's CheckRLM row ✓.
>
> **⚠ In-reasoning advantage shrinks as the checker improves (honest scope).** The In−Post f1 gain on HotpotQA collapses from **+14.9 (Qwen-2.5-14B checker)** to **+7.5 (Llama-3.3-70B checker)** — roughly halved. A stronger checker narrows the gap because Post-reasoning Check itself gets better; the "in-reasoning is critical" headline is carried by the weaker-checker regime. IIRC gains are ~0 across all checkers (single-hop, less error propagation to intercept).

### Table 10 — Retriever effect (%) [Llama-3.3-70B checker] (L1333–1355)

| RLM | Retriever | Hot f1 | em | 2Wiki f1 | em | Mus f1 | em | IIRC f1 | em |
|---|---|---|---|---|---|---|---|---|---|
| QwQ-32B | Direct Reasoning | 38.4 | 29.0 | 34.6 | 28.4 | 18.5 | 7.4 | 24.8 | 21.4 |
| QwQ-32B | Vanilla RAG | 52.7 | 42.2 | 46.4 | 42.6 | 19.3 | 10.0 | 25.0 | 22.0 |
| QwQ-32B | CheckRLM − BM25 | 66.3 | 52.6 | 73.4 | 62.0 | 39.6 | 27.2 | 33.1 | 29.0 |
| QwQ-32B | CheckRLM − bge-large-en-v1.5 | 65.2 | 51.8 | 74.3 | 63.8 | 40.1 | 26.0 | 35.7 | 31.0 |
| Qwen3-32B | Direct Reasoning | 36.4 | 27.6 | 34.0 | 29.4 | 16.6 | 6.6 | 23.0 | 20.2 |
| Qwen3-32B | Vanilla RAG | 51.2 | 41.0 | 44.5 | 41.8 | 19.7 | 10.4 | 22.5 | 19.8 |
| Qwen3-32B | CheckRLM − BM25 | 64.2 | 52.0 | 70.8 | 60.8 | 35.6 | 25.4 | 31.6 | 27.8 |
| Qwen3-32B | CheckRLM − bge-large-en-v1.5 | 67.7 | 54.2 | 74.2 | 63.4 | 39.6 | 26.0 | 34.3 | 29.8 |
| Qwen3-8B | Direct Reasoning | 29.3 | 22.2 | 31.1 | 26.6 | 12.9 | 5.2 | 21.0 | 18.2 |
| Qwen3-8B | Vanilla RAG | 44.5 | 35.2 | 40.9 | 38.4 | 13.8 | 6.0 | 19.8 | 16.8 |
| Qwen3-8B | CheckRLM − BM25 | 61.9 | 49.6 | 69.7 | 58.6 | 35.0 | 24.0 | 29.7 | 25.4 |
| Qwen3-8B | CheckRLM − bge-large-en-v1.5 | 64.9 | 51.8 | 74.7 | 64.6 | 36.1 | 24.8 | 35.2 | 30.6 |

> §C.6 claims "bge-large-en-v1.5 yields better performance than BM25" — true on aggregate but **NOT on every cell**: BM25 beats bge on QwQ-32B Hot (66.3 > 65.2) and QwQ-32B MuSiQue em (27.2 > 26.0). The CheckRLM BM25 rows here are byte-identical to Table 1's CheckRLM rows (Table 1's main results use BM25+bge hybrid on multi-hop) ✓ — Table 10 isolates each retriever.

### Table 11 — Cost-efficient reasoning [QwQ-32B × Llama-3.3-70B, 2WikiMQA] (L1361–1369)

| Method | # Tokens (↓) | Time s (↓) | f1 % (↑) |
|---|---|---|---|
| Direct Reasoning | 910.9 | 1.8 | 34.6 |
| Vanilla RAG | 1684.0 | 4.0 | 46.4 |
| Search-o1 | 2008.5 | 3.9 | 71.4 |
| Post-reasoning Check | 1487.2 | 2.9 | 60.2 |
| In-reasoning Check | 1364.1 | 3.3 | 73.4 |

### Table 12 — Cost-efficient reasoning [QwQ-32B × Qwen2.5-14B, 2WikiMQA] (L1374–1382)

| Method | # Tokens (↓) | Time s (↓) | f1 % (↑) |
|---|---|---|---|
| Direct Reasoning | 910.9 | 1.8 | 34.6 |
| Vanilla RAG | 1684.0 | 4.0 | 46.4 |
| Search-o1 | 2008.5 | 3.9 | 71.4 |
| Post-reasoning Check | 1496.7 | 4.2 | 47.6 |
| In-reasoning Check | 1082.8 | 2.0 | 65.5 |

> **⚠ "Lower cost" is token-cost, not always wall-clock (honest scope).** §5.4 claims in-reasoning check "attains a more favorable outcome across both performance and cost dimensions" — true on **tokens + f1** in both tables (T11: 1364.1 < 2008.5 Search-o1, f1 73.4 > 71.4; T12: 1082.8 < all, f1 65.5). But on **wall-clock time**, T11 In-reasoning (3.3s) is **slower** than Post-reasoning (2.9s) — the "both dimensions" claim is carried by tokens+f1, not time, for the Llama-70B checker. (T12 with the Qwen-14B checker does show In faster on both: 2.0s < 4.2s.) The headline abstract "lower costs" should be read as token-consumption + the Qwen-14B-checker time regime.

### Tables 13–15 — Case studies (L1388+)
Qualitative traces on the "Slap Her… She's French" question: T13 Direct Reasoning (wrong director Jim Abrahams → wrong date) + Vanilla RAG (right director Melanie Mayron but no birth-date doc → "Unknown"); T14 Post-reasoning Check (right director but corrupt intermediate → wrong date); T15 In-reasoning Check (corrects director in Step 1, birth date in Step 2 → correct answer). Not transcribed as numeric tables.

---

## 5. Figures (NOT back-filled as cells)

- **Figure 1** (L57–66): error-accumulation illustration (Direct/Post/CheckRLM contrast) — qualitative, prose-confirmed.
- **Figure 2** (L150–205): framework overview. ⚠ Figure artifact: a stray Chinese placeholder string "在此处键⼊公式。" ("type formula here") appears at L181 inside the Knowledge Correction Model box — a leftover editing artifact in the published figure, transcribed verbatim, NOT our error.
- **Figure 3** (L431–477): Post- vs In-reasoning Check bar chart. Bar heights for QwQ-32B × Llama-3.3-70B (2Wiki 60.2/73.4) and × Qwen2.5-14B (2Wiki 47.6/65.5) match Tables 9/11/12 ✓ — quoted via that cross-reference, not axis-read.
- **Figure 4** (L552–568): Time/token/f1 3-D scatter. The 5 method points match Table 11 exactly ✓ — quoted via Table 11.
- **Figure 5** (L554–573): Golden-Reasoning-Step ↔ Checking-Step correspondence (2WikiMQA + MuSiQue). Prose-confirmed claim: corrections concentrate at Checking Steps 1–2 (early interception); deeper steps rare.
- **Figures 6/7** (L1154): correction-step distribution histograms (2WikiMQA/MuSiQue), peaking at Steps 2–5 — prose-confirmed.

---

## 6. Strengths

1. **Falsifiable mechanism + clean ablation.** The in-reasoning vs post-reasoning contrast (Table 9) isolates the *timing* of intervention as the causal lever — In beats Post on 30/30 cells, and the gain scales with how much propagation the early error caused (multi-hop Hot/2Wiki/MuSiQue ≫ single-hop IIRC). The constrained ablation (Table 7) confirms both Recognition and Correction modules contribute.
2. **Minimal-intervention design is empirically justified**, not assumed: Table 8 shows summary injection *hurts* (−1.9 on 2WikiMQA f1), supporting the "correct tokens, don't inject paragraphs" design rationale.
3. **Cost-efficient Pareto improvement over Search-o1**: Table 11 — fewer tokens (1364 vs 2008, −32%) AND higher f1 (73.4 vs 71.4) on 2WikiMQA, while beating Vanilla RAG on both axes. The in-reasoning check avoids the redundant retrieval that error accumulation triggers in baselines.
4. **Joint DPO of recognition + correction** (single model, Eq 7) with data-composition ablation (Table 6) showing the combined `D_KCR+D_KCC` > either subset — the two objectives are complementary, not redundant.
5. **Generalises across 4 reasoners × 4 checkers** (Table 4) and 2 retrievers (Table 10) — not a single-model artifact.

---

## 7. Limitations & honest-scope flags (transcribed verbatim, NOT reconciled)

1. **Avg column = 10-cell mean (5 f1 + 5 em), not f1-only** — see Table 1 note. The reported 45.4 is ~5 pp below the f1-only mean (50.5); cite the column with its denominator.
2. **§5.2 "5.7% DPO improvement" is in-domain 2WikiMQA f1 only** (the DPO train set is sampled from 2WikiMQA); Hot +1.6, SimpleQA +0.9 — see Table 2 note.
3. **Headline results use a 70B checker** (Llama-3.3-70B-Instruct) on top of the reasoner; for the Qwen3-8B reasoner the checker is ~9× its size. Per-query token/time figures (Tables 11/12) count the **reasoner only**, understating deployed cost. Checker-size scaling is also non-monotone (Qwen-2.5-32B < Qwen-2.5-14B on QwQ-32B) — see Table 4 note.
4. **In-reasoning > Post-reasoning gain halves when the checker improves** (+14.9 → +7.5 on Hot f1) — the "in-reasoning is critical" claim is strongest in the weak-checker regime; see Table 9 note.
5. **"Lower cost" = tokens + f1, not always wall-clock** — T11 In-reasoning is slower (3.3s) than Post (2.9s) for the Llama-70B checker; see Table 11/12 note.
6. **MuSiQue "+6.3 f1 over best baseline" is QwQ-32B + Llama-70B-checker specific** (39.6 − 33.3 Search-o1 = 6.3 ✓). The Qwen3-32B/Qwen3-8B rows don't include Search-o1, so the cross-model magnitude of the MuSiQue win isn't shown.
7. **§C.6 "bge > BM25" is an aggregate claim** — BM25 wins on QwQ-32B Hot f1 (66.3 > 65.2) and MuSiQue em; see Table 10 note.
8. **Text-only, single knowledge base** (authors' own Limitations §6): no multimodal verification, no multi-source conflict resolution.
9. **DPO train data is 2WikiMQA-sourced** (Appendix A) yet CheckRLM improves other datasets too — generalisation is empirical, not trained-for; the in-domain 2WikiMQA gain (Table 2) is the largest as expected.

---

## 8. Verdict

CheckRLM is a clean, well-motivated **mid-reasoning RAG intervention** for long-CoT RLMs: it diagnoses error accumulation as the failure mode of both Direct Reasoning (no check) and Post-reasoning Check (too late), and intervenes **paragraph-granularly during reasoning** with token-level corrections grounded in retrieved docs. The in-vs-post ablation (Table 9) is the citable falsifiable evidence that **timing** (not just retrieval) is the lever, and the minimal-correction design is justified by the summary-injection ablation (Table 8). Gains are real and consistent across reasoners/checkers/retrievers, but the headline numbers rest on a **70B checker** whose cost is excluded from the efficiency figures, the **DPO gain is in-domain-concentrated**, and the in-reasoning advantage **shrinks as the checker strengthens** — three honest-scope caveats a reader should weight before citing the top-line 45.4 Avg / +6.3 MuSiQue. Sibling to the search-during-reasoning lineage (Search-o1, RT-RAG) but attacks **factual-error propagation** rather than retrieval coverage; complementary to MAVEN (evidence-state rewards, iter 53) which trains the reasoner with rewards where CheckRLM leaves the reasoner frozen and trains only the checker via DPO.
