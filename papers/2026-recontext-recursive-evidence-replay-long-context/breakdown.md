# ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning — Source-First Breakdown

- **arXiv:** 2607.02509v1 (cs.AI, 2 Jul 2026)
- **Authors:** Yanjun Zhao*, Ruizhong Qiu*, Tianxin Wei*, Yuanchen Bei, Zhining Liu, Lingjie Chen, Ismini Lourentzou, Hanghang Tong, Jingrui He† (University of Illinois Urbana-Champaign). Code: `github.com/Yanjun-Zhao/ReContext`.
- **Subarea (new to repo):** **training-free, inference-time evidence-replay harness for long-context reasoning** — turns the model's *own prompt-internal attention* into an explicit, query-conditioned evidence pool and **recursively replays** it before final generation while **preserving the full original context** (no pruning, no external memory, no training, no attention-logit editing during decoding). **First repo paper to frame long-context utilization as cue-conditioned trace reactivation under an associative-memory model**, with a monotonic-improvement theorem. Sibling-in-spirit to the long-context lineage (`evidence-state-rewards-long-context-reasoning`/MAVEN iter 53 trains the reasoner with evidence-state rewards; `drowning-in-documents` iter 47 diagnoses attention dilution; `checkrlm` iter 64 mid-reasons over a CoT) but ReContext is **training-free + full-context-preserving + model-internal-signal-only**, where MAVEN trains, CheckRLM retrieves externally, and drowning-in-documents changes architecture.
- **Source files:** `paper.pdf` (10pp, 2.99MB), `paper_layout.txt` (`pdftotext -layout`, 1089 lines). All numbers below are prose-/table-/equation-confirmed against `paper_layout.txt`. The paper has **7 explicit tables (T1 main, T2 thinking-enabled, T3 64K-context, T4 token-source ablation, T5 runtime, T6 rounds ablation, T7 top-K ablation) + Eqs 1–22 + Theorem 1 + 6 figures**. Figure bar/curve values (Figs 3/5/6) are NOT back-filled as cells; only the Figure-1/6 motivation stat (top-0.1% tokens ≈ 50–80% relevance) that also appears verbatim in prose is quoted, per the universal "figure-derived numbers are weak" rule.

---

## 1. The problem (motivation)

Long-context LLMs support ≥128K-token windows yet still **fail to use evidence already present in the input** — there is a gap between *context access* and *effective context utilization*. Prior analyses show LLMs are position-sensitive ("lost in the middle", Liu et al. 2023a) and may ignore relevant spans in unfavorable locations. **Figure 1 / Figure 6 motivation stat:** the **top 0.1% of context tokens already account for ≈50–80% of the accumulated question-conditioned relevance score** across all three backbones — only **128 tokens in a 128K-token context** (§1, L151–153; Fig 6 caption L1085–1088). I.e. a tiny, identifiable token subset holds most of the answer-relevant signal, but that signal stays *latent* inside attention and is not surfaced as text.

Existing fix families all fall short (§1, §2.1):

- **Retrieval / external memory** (RAG, A-MEM) add a retriever or agentic memory module — extra system, may lose fine-grained detail, unstable on multi-hop.
- **Compression** (DAC, KV-cache eviction) shortens the effective input — irreversible, can drop the supporting sentence.
- **Attention rescaling** (AttnSharp, DySCO) modify the backbone forward / decoding logic — invasive, engine-coupled, and the selected evidence stays *latent* inside decoding rather than exposed as text.

ReContext's gap: **emphasize without excluding** — keep the full context available, but copy a query-conditioned evidence subset near the question so its utilization becomes explicit. It is a *harness* between full-context reading and final generation, not a retriever, pruner, or trainer.

---

## 2. Method (§3)

### 2.1 Overview (§3.1)

Given a long context `C` and question `q`, a standard long-context LLM generates from `[C; q]`. **ReContext** instead (i) reads the original prompt, (ii) extracts candidate evidence spans using **question-conditioned internal attention**, (iii) replays these spans as an ordered evidence pool near the question, and (iv) generates the final answer from `[C; φ(E); q]` — the **full context, the evidence pool, and the question**. The prompt is never pruned; attention logits are never directly modified during final decoding.

### 2.2 Evidence Selection (§3.2, Eq 1–2)

Let `M` be the backbone, `I_x` the token positions in prompt `x`, `I_C ⊆ I_x` the positions belonging to the *original context* `C`. Let `Q(x) = (t_1, …, t_L)` be the last `L ≤ w` cue positions in the prompt suffix, with **`w = 8`** in the main experiments. These suffix cues are the query-conditioned readout (in later rounds they are conditioned on the replayed scaffold).

For cue token `t_u`, aggregate attention over a **selected set of layer-head pairs** `H`:

```
a_i^(u) = (1/|H|) Σ_{(l,h)∈H} At_(u,i)^(l,h),   i ∈ I_x        (Eq 1)
```

`At_(u,i)^(l,h)` is the attention weight from cue token `t_u` to token `i` at layer-head `(l,h)`. Cue-token evidence is accumulated across `Q(x)` with **exponential decay** (Eq 13, `λ = 0.75`):

```
r^(t) = Normalize( a^(t) + λ · r^(t-1) )                          (Eq 13)
```

and candidates are **restricted to the original context**, top-K:

```
P = TopK_{i ∈ I_C}( r_i ; K )                                    (Eq 2)
```

`K` is the evidence-token budget. These scores are an *inexpensive prompt-internal proposal signal* identifying spans that may help answer generation.

### 2.3 Evidence Materialization and Replay (§3.3, Eq 3–5)

Token-level proposals are too fragmentary (a token may flag an entity/date/predicate but the model needs the surrounding statement). ReContext maps selected tokens back to their containing sentences/spans. Let `S_C = (s_1, …, s_N)` be the ordered span decomposition of `C`, `pos(s_n)` the positions covered by `s_n`. The evidence pool is the ordered subsequence of spans touched by selected tokens:

```
E = ( s_n : pos(s_n) ∩ P ≠ ∅ )_{n=1..N}                          (Eq 3)
```

Each evidence unit is **copied from the original prompt** (grounded, not freely generated). For a single-pass replay:

```
x+ = [ C; φ(E); q ]                                              (Eq 4)
y ~ M(· | x+)                                                    (Eq 5)
```

`φ(E)` is the textual replay format; `⊕` / ordered concatenation is used (not set union — the pool is an ordered list). Replay places selected evidence *near the question* while keeping the full context available: ReContext **selects for emphasis rather than exclusion**.

### 2.4 Recursive Evidence Selection (§3.4, Eq 6–11)

Over `R` rounds ReContext updates the evidence pool recursively. Let `E^(0) = ∅`. At round `j ∈ {1,…,R}` the current prompt is:

```
x^(j-1) = [ C; φ(E^(j-1)); q ]                                   (Eq 6)
```

The model reads `x^(j-1)`, so the **replayed scaffold conditions the hidden states** from which query-side prompt-suffix attention scores are computed. In the main setting, candidate positions are selected **only from the original context** positions `I_C` (the scaffold conditions scoring but is not itself a source of new copied spans). ReContext obtains scores `r^(j)`, selects `P^(j)`, materializes spans from the *original context*, and appends only spans not already in the pool:

```
Sb^(j) = ( s_n : pos(s_n) ∩ P^(j) ≠ ∅ )_{n=1..N}                (Eq 7)
ΔE^(j) = Sb^(j) \ E^(j-1)                                        (Eq 8)
E^(j)  = E^(j-1) ∪ ΔE^(j)          (ordered concatenation)       (Eq 9)
```

The final answer is generated from:

```
x^(R) = [ C; φ(E^(R)); q ]                                       (Eq 10)
y ~ M(· | x^(R))                                                 (Eq 11)
```

"Recursive" is used in this **limited inference-time** sense: each evidence proposal depends on the pool produced by previous rounds, **not** an open-ended reasoning loop. In practice `R` is small and fixed (main scripts use **`R = 2`** with sentence-wrapping disabled), so ReContext is a lightweight inference-time wrapper.

### 2.5 Theoretical Analysis (§3.5, Theorem 1; proof §E)

**Associative-memory interpretation** (§2.2, §E intro): the long context is a *memory store* of traces, the question is a *retrieval cue*, attention is a *prompt-internal proxy for cue–trace association*, and replay is *trace reactivation* near generation. Recursive evidence sifting repeatedly re-queries the same context under a scaffold-conditioned state, improving query–evidence rebinding without external memory / retriever / pruning.

**Theorem 1 (monotonic improvement).** Let `h^(j)` be the hidden embedding after the `j`-th evidence replay step and `y` the answer embedding. Then for every step `j ≥ 1`:

```
cos( h^(j), y ) > cos( h^(j-1), y )                              (Eq 12)
```

**Setup (§E.1, Eq 14–18).** Following Nichani et al. 2025 / Olsson et al. 2022: each context token `i` has a mutually orthogonal unit-norm embedding `c_i ∈ ℝ^d`; the answer has embedding `y = c_{i*}` for some `i*`; the query `q ∈ ℝ^d` is relevant to `y`: `⟨y,q⟩ > ⟨c_i,q⟩` ∀ `i ≠ i*` and `max_{i≠i*} ⟨y−c_i,q⟩ / (a_{i*}^(0) − a_i^(0)) < 1`. Initial attention `a^(0) = softmax([⟨x_i,q⟩])` (Eq 14); `h^(0) = Σ a_i^(0) x_i` (Eq 15). Each step `j` appends the most-relevant evidence `x^(j) = [x^(j-1), x_{argmax(a^(j-1))}]` (Eq 16), updates attention `a^(j) = softmax([⟨x_i,h^(j-1)⟩])` (Eq 17), and the hidden embedding `h^(j) = Σ a_i^(j) x_i` (Eq 18).

**Proof sketch (§E.2, Eq 19–42).** W.l.o.g. `i* = 1`. Express `h^(j) = Σ w_i^(j) c_i` (Eq 19); cos-sim to `y = c_1` reduces (Eq 20–22) to `(w_1^(j)) · ( Σ_{i≠1} (w_i^(j)/w_1^(j))² )^{-1/2}`. Define the weight ratio `R_i^(j) = w_1^(j) / w_i^(j)`. It suffices to show `R_i^(j) > R_i^(j-1)` ∀ `i≠1, j≥1` by induction. **Base case** (Eq 23–26): at `j=0`, `ln R_i^(0) < w_1^(0) − w_i^(0)`; after appending `c_1`, `R_i^(1) = 2e^{w_1^(0)−w_i^(0)} > 2 R_i^(0) > R_i^(0)`. **Inductive step** (Eq 27–42): the recurrence `R_i^(j) = (j+1) e^{Δ_i^(j-1)}` (Eq 28) with `Δ_i^(j) = w_1^(j) − w_i^(j)`; one proves the strictly-stronger `Δ_i^(j) > Δ_i^(j-1)` (Eq 31, 41) by bounding `Σ^(j-1) < ((j+1)/j) Σ^(j-2)` (Eq 34) and showing the numerator `u − Δ_i^(j)` is `< 0` (Eq 32–40). Since `R_i^(j)` strictly increases every step, `w_1`'s relative mass continuously approaches 1, the cos-sim denominator strictly shrinks ⇒ `cos(h^(j), y) > cos(h^(j-1), y)` ∀ `j ≥ 1` (Eq 42 / Eq 12). ∎

**Intuition:** explicitly adding the evidence to the prompt monotonically increases the similarity of the hidden embedding to the answer embedding — a formal cue–trace reactivation result.

---

## 3. Experimental setup (§4, App B–C)

- **Datasets (§4.1, App B.1):** 8 long-context benchmarks at **128K** — NQ, TriviaQA, HotpotQA, PopQA, NarrativeQA, InfBench QA, InfBench MC (first 7 = HELMET 128K-context versions, Yen et al. 2025), and **CLIPPER** (Pham et al. 2025, evidence-grounded claim verification over book-length contexts). Metrics: Acc + token-F1 for NQ/TriviaQA/HotpotQA/PopQA/NarrativeQA/InfBench QA; Acc only for InfBench MC and CLIPPER.
- **Backbones (§4.3):** Qwen3-4B, Qwen3-8B, Llama3.1-8B. Qwen3 models evaluated beyond their native window use **YaRN RoPE scaling**. Main results: **thinking disabled**; 128K context (Table 3 uses 64K; Table 2 uses thinking-enabled).
- **Baselines (§4.2):** **Vanilla** (direct full-context generation); **AttnSharp** (sharpen attention toward question-relevant tokens); **DySCO** (dynamically rescale decoding attention using retrieval-head signals); **A-MEM** (external agentic memory); **DAC** (dynamic attention-aware prompt compression). All methods share the same backbone, context budget, prompting format, decoding config, hardware.
- **ReContext hyperparameters (App B.4):** cue window **`w = 8`** (`context_warmup_steps`); decay **`λ = 0.75`**; **`R = 2`** replay rounds default (sentence-wrapping disabled); chat-template tokens masked; candidate spans restricted to original context (main setting); **<128 additional evidence tokens** inserted (memory ≈ Vanilla).
- **Hardware (App B.2):** NVIDIA **A100** and **H200** GPU servers (H200 on ARM64/aarch64).
- **KV-cache accounting (App B.4):** implementation snapshots the KV cache at the end of the original context; during replay it restores this cache and processes only the inserted evidence + question-side tokens; the generation-length budget is extended by the inserted-replay-token count so evidence does not reduce the answer-token budget.

---

## 4. Results — tables verbatim (with sourcing line-ranges)

### Table 1 — Main benchmark, 8 datasets × 3 backbones, 128K, thinking disabled (L212–240)

Avg Rank = mean rank over **14 metric columns** (Acc+F1 for NQ/TriviaQA/HotpotQA/PopQA/NarrQA/InfQA + Acc for InfMC + Acc for CLIPPER), lower = better.

| Backbone | Method | NQ Acc | NQ F1 | TrivQA Acc | TrivQA F1 | HotQA Acc | HotQA F1 | PopQA Acc | PopQA F1 | NarrQA Acc | NarrQA F1 | InfQA Acc | InfQA F1 | InfMC Acc | Clip Acc | **Avg Rank** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-4B** | Vanilla | 0.02 | 0.21 | 0.04 | 0.24 | 0.00 | 0.10 | 0.00 | 0.11 | 0.02 | 0.17 | 0.09 | 0.21 | 0.51 | 0.38 | 4.39 |
| | AttnSharp | 0.02 | 0.21 | 0.02 | 0.23 | 0.00 | 0.09 | 0.00 | 0.10 | 0.03 | 0.17 | 0.10 | 0.23 | 0.47 | 0.42 | 4.25 |
| | DySCO | 0.02 | 0.21 | 0.10 | 0.30 | 0.03 | 0.13 | 0.00 | 0.10 | 0.01 | 0.17 | 0.11 | 0.23 | 0.50 | 0.44 | 4.00 |
| | A-MEM | 0.02 | 0.20 | 0.19 | 0.37 | 0.06 | 0.15 | 0.06 | 0.16 | 0.04 | 0.19 | 0.07 | 0.18 | 0.43 | 0.48 | 3.57 |
| | DAC | 0.02 | 0.18 | 0.21 | 0.38 | 0.07 | 0.17 | 0.01 | 0.09 | 0.05 | 0.20 | 0.07 | 0.19 | 0.43 | 0.24 | 3.79 |
| | **ReContext** | **0.08** | **0.25** | **0.30** | **0.45** | **0.08** | **0.19** | **0.07** | **0.19** | **0.07** | **0.21** | **0.12** | **0.24** | **0.55** | **0.52** | **1.00** |
| **Qwen3-8B** | Vanilla | 0.06 | 0.26 | 0.53 | 0.66 | 0.18 | 0.31 | 0.18 | 0.32 | 0.19 | 0.33 | 0.22 | 0.36 | 0.64 | 0.30 | 3.96 |
| | AttnSharp | 0.06 | 0.26 | 0.43 | 0.59 | 0.18 | 0.31 | 0.13 | 0.29 | 0.18 | 0.34 | 0.23 | 0.37 | 0.60 | 0.32 | 4.50 |
| | DySCO | 0.09 | 0.30 | 0.51 | 0.64 | 0.19 | 0.31 | 0.22 | 0.35 | 0.18 | 0.33 | 0.23 | 0.36 | 0.63 | 0.34 | 3.25 |
| | A-MEM | 0.08 | 0.28 | 0.58 | 0.67 | 0.21 | 0.32 | 0.19 | 0.30 | 0.17 | 0.31 | 0.18 | 0.28 | 0.58 | 0.18 | 4.21 |
| | DAC | 0.12 | 0.29 | 0.65 | 0.72 | 0.22 | 0.35 | 0.15 | 0.30 | 0.14 | 0.28 | 0.15 | 0.26 | 0.64 | 0.20 | 3.61 |
| | **ReContext** | **0.13** | **0.33** | **0.68** | **0.75** | 0.20 | 0.34 | **0.23** | **0.36** | **0.21** | **0.35** | **0.25** | **0.39** | 0.63 | 0.33 | **1.46** |
| **Llama3-8B** | Vanilla | 0.15 | 0.29 | 0.69 | 0.76 | 0.24 | **0.39** | 0.21 | 0.28 | 0.13 | 0.27 | 0.15 | 0.34 | 0.56 | 0.32 | 3.25 |
| | AttnSharp | 0.15 | 0.28 | 0.69 | 0.76 | 0.23 | **0.39** | 0.20 | 0.28 | 0.13 | 0.27 | 0.17 | 0.34 | 0.57 | 0.22 | 3.29 |
| | DySCO | 0.10 | 0.26 | 0.63 | 0.72 | 0.23 | 0.37 | 0.18 | 0.27 | 0.13 | 0.26 | 0.14 | 0.33 | 0.56 | 0.38 | 4.57 |
| | A-MEM | 0.16 | **0.34** | 0.67 | 0.75 | 0.24 | 0.34 | 0.17 | 0.23 | 0.15 | 0.29 | 0.19 | 0.33 | 0.54 | 0.32 | 3.43 |
| | DAC | 0.06 | 0.23 | 0.56 | 0.68 | 0.16 | 0.30 | 0.15 | 0.22 | 0.16 | **0.30** | 0.15 | 0.28 | 0.49 | 0.28 | 5.18 |
| | **ReContext** | **0.19** | 0.31 | **0.70** | **0.77** | **0.25** | **0.39** | **0.22** | **0.29** | **0.17** | 0.29 | **0.22** | **0.40** | **0.64** | **0.40** | **1.29** |

ReContext column-max on **every one of the 14 Qwen3-4B cells** (⇒ avg-rank 1.00). On Qwen3-8B it leads most QA metrics but is **not** max on HotpotQA Acc/F1 (DAC 0.22/0.35), InfMC Acc (Vanilla/DAC 0.64), CLIPPER Acc (DySCO 0.34). On Llama3-8B it has the best avg-rank and the highest Acc on every task, but NQ F1 (A-MEM 0.34), HotQA F1 (tied 0.39), and NarrativeQA F1 (DAC 0.30) are led by others.

### Table 2 — Robustness, thinking enabled (Qwen3-4B, 128K) (L429, L433–442)

| Method | NQ Acc | NQ F1 | PopQA Acc | PopQA F1 | InfMC Acc |
|---|---|---|---|---|---|
| Vanilla | 0.08 | 0.24 | 0.14 | 0.25 | 0.69 |
| AttnSharp | 0.13 | 0.27 | 0.14 | 0.24 | 0.63 |
| DAC | 0.10 | 0.23 | 0.13 | 0.24 | 0.66 |
| A-MEM | 0.09 | 0.23 | 0.11 | 0.23 | 0.71 |
| DySCO | 0.14 | 0.29 | 0.17 | **0.29** | 0.67 |
| **ReContext** | **0.15** | **0.30** | **0.18** | 0.28 | **0.72** |

ReContext best on NQ Acc/F1, PopQA Acc, InfMC Acc; PopQA F1 is the exception (DySCO 0.29). Macro over 5 scores Vanilla→ReContext 0.280→0.326 (paper: 28.0→32.6).

### Table 3 — Robustness, 64K context budget (thinking disabled) (L429, L433–442)

| Method | NQ Acc | NQ F1 | PopQA Acc | PopQA F1 | InfMC Acc |
|---|---|---|---|---|---|
| Vanilla | 0.07 | 0.24 | 0.04 | 0.20 | 0.48 |
| AttnSharp | 0.07 | 0.24 | 0.03 | 0.20 | 0.44 |
| DAC | 0.09 | 0.23 | 0.17 | **0.32** | 0.53 |
| A-MEM | 0.06 | 0.20 | 0.06 | 0.19 | 0.51 |
| DySCO | 0.11 | 0.26 | 0.07 | 0.23 | 0.46 |
| **ReContext** | **0.11** | **0.26** | **0.18** | 0.30 | **0.54** |

ReContext ties best NQ Acc, best NQ F1 / PopQA Acc / InfMC Acc, second on PopQA F1. Macro Vanilla→ReContext 0.206→0.278 (paper: 0.21→0.28, **35.0% relative**).

### Table 4 — Ablation: evidence-token source (Qwen3-4B, 128K) (L463, L467–473)

| Source | NQ Acc | NQ F1 | PopQA Acc | PopQA F1 | InfMC Acc |
|---|---|---|---|---|---|
| Full prompt | 0.04 | 0.23 | 0.02 | 0.14 | 0.52 |
| **Context (main)** | **0.08** | **0.25** | **0.07** | **0.19** | **0.54** |

Context-only selection beats full-prompt on all 5 metrics; macro 0.190→0.226 (paper: 0.19→0.23). Supports the main setting: scaffold conditions scoring but copied spans come from the original context.

### Table 5 — Runtime on CLIPPER, Llama3-8B, 128K (L765, L768–775)

| Method | Runtime |
|---|---|
| Vanilla | 44 min |
| AttnSharp | 46 min |
| DAC | 34 min |
| A-MEM | 50 min |
| DySCO | 2h 13min |
| **ReContext** | 62 min |

ReContext ≈ Vanilla + ~40% (evidence sifting + replay); **substantially faster than DySCO** (133 min, which changes backbone/decoding logic). GPU memory ≈ Vanilla/DySCO (scaffold adds <128 tokens).

### Table 6 — Ablation: recursive rounds R (Qwen3-4B, 128K) (L825, L831–838)

| R | NQ Acc | NQ F1 | PopQA Acc | PopQA F1 | InfMC Acc |
|---|---|---|---|---|---|
| 1 | 0.04 | 0.21 | 0.01 | 0.10 | 0.48 |
| **2 (main)** | **0.08** | **0.25** | **0.07** | **0.19** | 0.50 |
| 3 | **0.09** | **0.25** | 0.05 | 0.18 | 0.51 |
| 4 | **0.09** | **0.25** | 0.05 | 0.17 | **0.54** |

1→2 rounds jumps macro 0.168→0.218 (paper: 0.17→0.22). Best NQ Acc at R=3/4, best InfMC at R=4, best PopQA at R=2 — **best depth is task-dependent, not uniformly larger**.

### Table 7 — Ablation: top-K candidate budget (Qwen3-4B, 128K, R=2) (L887, L891–899)

| K | NQ Acc | NQ F1 | PopQA Acc | PopQA F1 | InfMC Acc |
|---|---|---|---|---|---|
| 1 | 0.03 | 0.22 | 0.04 | 0.14 | 0.52 |
| **8 (main)** | **0.08** | **0.25** | 0.07 | 0.19 | 0.50 |
| 16 | **0.08** | **0.25** | 0.05 | 0.17 | 0.55 |
| 32 | 0.04 | 0.23 | **0.10** | **0.21** | **0.58** |

Macro rises K=1→32: 0.190→0.232 (paper: 0.19→0.23) but **non-monotonic per task**: larger K helps PopQA/InfMC (recall), hurts NQ (noise) — a **recall–noise trade-off**.

---

## 5. Source-free reconciliation (PASSED)

Independent recomputation from the verbatim cells above (no PDF re-read):

- **Table-1 24-cell macro Acc (3 backbones × 8 Acc cols):** Vanilla `0.2421` → ReContext `0.3017` ⇒ **relative gain 24.61%**, paper states **24.6%** — ✓ EXACT.
- **Table-1 average ranks (recomputed over all 14 metric columns, ties averaged):** Qwen3-4B ReContext **1.00** (column-max on all 14 cells) ✓; Qwen3-8B **1.46** ✓ EXACT (the 4 non-max cells are exactly the paper's stated exceptions — HotQA Acc/F1 DAC 0.22/0.35, InfMC Vanilla/DAC 0.64, CLIPPER DySCO 0.34); Llama3-8B **1.25 computed vs 1.29 paper** (within tie-handling tolerance; non-max cells NQ F1 A-MEM 0.34 / HotQA F1 tied 0.39 / NarrQA F1 DAC 0.30 match the paper's caveat).
- **Table-3 64K macro:** Vanilla `0.206` → ReContext `0.278` ⇒ **34.95%** relative, paper **35.0%** — ✓ EXACT.
- **Table-4 source ablation macro:** Full-prompt `0.190` → Context `0.226`, paper **0.19→0.23** — ✓ EXACT.
- **Table-6 rounds macro:** R=1 `0.168` → R=2 `0.218`, paper **0.17→0.22** — ✓ EXACT.
- **Table-7 top-K macro:** K=1 `0.190` → K=32 `0.232`, paper **0.19→0.23** — ✓ EXACT.
- **Default-config cross-table identity:** T6 R=2 row `(NQ 0.08, F1 0.25, PopQA 0.07, F1 0.19, InfMC 0.50)` is **byte-identical** to T7 K=8 row — confirms the ablation default is `R=2, K=8`.
- **Table-5 runtime internal consistency:** DAC 34 min is the fastest (it compresses first → fewer decoded tokens); DySCO 133 min slowest (per-tenant decoding rescale) — qualitatively consistent with §B.3 prose.

**No numeric prose-vs-table contradiction.** All headline deltas recompute from the displayed cells.

---

## 6. Inline honest-scope notes (⚠ transcribed verbatim, NOT reconciled)

1. **Table-2 thinking-enabled relative-gain rounding (16.4% vs 16.7%):** §4.4 states the macro improves Vanilla→ReContext "from 28.0 to 32.6, a relative gain of **16.7%**", but `(0.326 − 0.280) / 0.280 = 16.43%` (or `(32.6−28.0)/28.0 = 16.43%`). The 16.7% is **0.3pp above** the recomputed value — a minor rounding/upbeat-rounding of the headline relative gain. Diagnostic: when a paper quotes a "relative gain %" alongside the two macro numbers, recompute from the displayed cells; small over-rounding of a relative gain is common (cf. iter-58 MAVEN-LBv2 / iter-60 DemoPSD entropy-range drift) and worth flagging rather than echoing.

2. **Ablation InfMC run-drift vs Table-1 main (0.50 vs 0.55):** the default-config cell InfMC is **0.50** in the Table-6/7 ablations (R=2, K=8) but **0.55** in Table-1 main (Qwen3-4B) and **0.54** in Table-4 Context-source. NQ Acc / PopQA Acc agree exactly across all four tables (0.08 / 0.07); only InfMC drifts. The ablations and the main run are evidently **different seed/exec configurations of the same default**, not a contradiction — but the InfMC drift (0.50–0.58 across settings) means the ablation deltas on InfMC are noisier than on NQ/PopQA. Diagnostic: when an ablation table's default row disagrees with the main table on one cell, check whether NQ/PopQA are stable (here yes) before treating it as transcription error vs run-drift.

3. **Closed-API limitation (authors' own §Limitations, L519–525):** ReContext **requires access to model-internal attention** (Eq 1), so it cannot run on closed-source APIs that do not expose attention/scoring. It also adds a read-and-replay stage ⇒ inference latency > direct full-context decoding (Table 5: 62 min vs Vanilla 44 min), though it remains training-free and uses no persistent external memory.

4. **Avg-rank metric, not accuracy-margin metric:** ReContext's headline "best average rank on all three backbones" is a **ranking** win, not a per-cell dominance. On Qwen3-8B it is **not** column-max on 4/14 cells (HotQA, InfMC, CLIPPER) and on Llama3-8B it trails on 3 F1 cells. The 24.6% Vanilla→ReContext macro-Acc gain is the stronger headline; the avg-rank should be quoted alongside the explicit non-max cells.

5. **Top-K recall–noise trade-off (Table 7) is task-dependent:** larger K helps PopQA/InfMC (more candidate spans ⇒ recall) but **hurts NQ** (K=32 NQ Acc 0.04 < K=8 0.08). The default K=8 is a compromise, not a uniformly optimal budget — the "macro rises with K" claim masks opposite per-task trends. Diagnostic: when an ablation reports a monotone macro but the paper itself flags non-monotone per-task behaviour, quote the per-task direction, not just the macro.

6. **Best recursion depth is task-dependent (Table 6):** R=2 is best for PopQA, R=3/4 best for NQ Acc, R=4 best for InfMC. The main scripts use R=2, so the main Table-1 results are **not at the per-task optimum** for NQ/InfMC — a single fixed R is a practical compromise, and the gains on NQ/InfMC could be larger with per-task R tuning.

7. **Theory assumes orthogonal unit-norm token embeddings (§E.1):** the monotonic-improvement Theorem 1 is proved under the Nichani/Olsson idealized setup (mutually orthogonal `c_i`, `‖c_i‖=1`, single relevant answer token `y = c_{i*}`, exact argmax append). Real Transformer token embeddings are neither orthogonal nor unit-norm, and materialized *spans* (Eq 3) ≠ single tokens (Eq 16). The theorem is an **interpretive** guarantee (replay pushes the hidden state toward the answer in the idealized model), not a bound on the empirical 128K-context results — cite it as motivation, not as proof of the Table-1 gains.

---

## 7. Verdict

ReContext is a clean, well-motivated **training-free long-context harness**: it exploits the empirical fact that ~128 tokens hold ~50–80% of the question-conditioned relevance (Fig 1/6) by surfacing those tokens as a grounded, recursively-refined evidence pool replayed near the question, while **never pruning the context and never editing decoding attention**. The associative-memory framing + monotonic-improvement Theorem 1 (Eq 12, proved by induction on the attention-weight ratio in §E.2) give it a falsifiable theoretical hook rare among training-free inference methods. Empirically it takes the best avg-rank on all 3 backbones (1.00/1.46/1.29) with a 24.6% Vanilla→ReContext macro-Acc gain that recomputes exactly, stays within top-2 under thinking-enabled (T2) and 64K-context (T3) robustness checks, and is ~2× faster than the strongest attention-rescaling baseline (DySCO). Honest scope: attention-access requirement excludes closed APIs; the 16.7% relative-gain quote over-rounds the true 16.4%; ablation InfMC cells drift 0.50–0.58 vs the 0.55 main; avg-rank ≠ per-cell dominance; the theorem is interpretive under orthogonal-embedding idealization. Distinct from MAVEN (trains the reasoner), CheckRLM (external retrieval mid-CoT), and drowning-in-documents (architecture) — ReContext is the repo's first **training-free, full-context-preserving, model-internal-attention-only** long-context utilization method.
