# NeuFS: Neuron-Aware Active Few-Shot Learning for LLMs

- **arXiv:** 2607.02423 (v1, 2 Jul 2026) — https://arxiv.org/abs/2607.02423
- **Authors:** Zhuowei Chen, Liwei Chen, Christian Schunn, Raquel Coelho, Xiang Lorraine Li — University of Pittsburgh, USA
- **Code:** https://github.com/johnnychanv/NeuFS
- **Source:** `paper.pdf` (1.5 MB, 16 pp), extracted `paper_layout.txt` (1186 lines, `pdftotext -layout`). All tables transcribed verbatim with sourcing line-ranges; figures left as figure-bar/axis reads per the universal rule.

> **Sourcing note.** This is a source-first breakdown built directly from `paper_layout.txt`. Table cell values are verbatim. All `Avg.` columns reproduce from the 4 per-shot columns (5/10/20/30) in the appendix per-shot grids (Tables 7–12), and the main-table aggregated values (Tables 1–3) reproduce from those `Avg.` columns — a **4-table consistency triangle** confirming verbatim transcription without re-reading the PDF (see §6 reconciliation).

---

## TL;DR

Active Few-Shot Learning (AFSL) picks which unlabeled samples to **annotate** and use as ICL demonstrations. Prior AFSL methods select on **output-level** signals (predictive entropy, external-embedding semantic similarity). NeuFS shifts selection to the model's **internal FFN neuron activations**: (1) diversify demonstrations by clustering their sparse activated-neuron **sets** (Jaccard K-Medoids) for broad knowledge coverage, and (2) preferentially annotate **low-consensus** samples (more unique activations ⟹ more hallucination-prone ⟹ most informative to label). Across 4 models × 3 datasets (MMLU-Pro reasoning + Edu-Feedback/TREC classification), NeuFS wins **all 4 MMLU-Pro aggregated-accuracy cells** and is competitive (1st/2nd) on classification, validating internal neuron dynamics as a more principled selection signal than external embeddings or output entropy. **Honest scope:** NeuFS is *not* uniformly best on classification (several Entropy-Diverse / FastVoteK cells edge it), and the deployed τ/k hyperparameters fall *outside* the paper's own ablation sweep range on some configurations.

---

## 1. Problem & contribution

- **Setting.** Few-shot ICL adaptation of LLMs to specialized domains (education/medical/law) where (a) large annotated retrieval sets are impractical and (b) fine-tuning is too costly. AFSL = select the most valuable unlabeled samples from a pool, **annotate them** (human-in-the-loop), and use them as the few-shot demonstrations.
- **Gap in prior AFSL.** Two selection axes, both output-level:
  - *Informativeness* via **entropy** — unreliable for LLMs (overconfidence, confidently-wrong hallucination; next-token-prediction objective ≠ knowledge assessment).
  - *Diversity* via **external-embedding** clustering (BERT/KNN) — assumes semantic-knowledge equivalence, which fails (semantically similar samples can invoke different task knowledge).
- **Contribution.** Move selection from output-level proxies to **internal model dynamics** (FFN neuron activation patterns), motivated by mech-interp work linking neuron consensus to hallucination (Chen 2025; Cao 2025). Two internal signals:
  1. **Neuron activation patterns** as the sample representation (replaces external embeddings).
  2. **Neuron consensus** (count of unique activated neurons) as an informativeness/hallucination-risk proxy (replaces output entropy).

---

## 2. Method (§3)

Three sequential stages. **Pipeline = neuron-activation identification → diversification (cluster) → consensus quantification (score) → select.**

### 2.1 Neuron Activation Identification (§3.1)

FFNs are key-value factual memories (Geva 2021), but raw activations are polysemantic/noisy. Filter to *task-contributing* neurons via **Early Unembedding** (Chen 2025): project each neuron's down-projection output vector into vocabulary space at the predicted token ŷ.

- FFN at layer `l`: input `h^l`, up-projection `W_in^l ∈ R^{d×d_ff}`, down-projection `W_out^l ∈ R^{d_ff×d}`. Activation `k^l = σ(h^l W_in^l)`.
- **Contribution score** of neuron `i` at layer `l` to predicted token ŷ:
  `S^l_{ŷ,i} = k^l_i · (w^l_{out,i} · e_ŷ)`  (Eq. in §3.1, unnumbered)
  where `w^l_{out,i}` is the i-th row of `W_out^l`, `e_ŷ` is ŷ's embedding in unembedding `E_u ∈ R^{d×V}`.
- **Activated-neuron set** `N(x) = N(x,y_0) ∪ … ∪ N(x,y_t)`, union over predicted tokens, thresholded by global top-k:
  `N(x,ŷ) = { (i,l) | S^l_{ŷ,i} > η }`, η = k-th-ranked contribution score.
- **Two-stage filtering for efficiency:** (1) retrieve top-`n` neurons per layer by *raw* activation (n=2000 all experiments, footnote 2) → pool; (2) rank pool by contribution score, keep global top-`k`. `N_act(x)` is a sparse set of (layer, neuron) indices = the sample's internal representation.

### 2.2 Neuron-Aware Active Few-Shot Selection (§3.2)

Two signals combined:

**(a) Neuron-Aware Sample Diversification** — cluster on activated-neuron *sets* (not dense embeddings).
- **Jaccard distance** between samples x_i, x_j: `D_J(x_i,x_j) = |N_act(x_i) ∩ N_act(x_j)| / |N_act(x_i) ∪ N_act(x_j)|`  (Eq. 1)
- **K-Medoids** (Kaufman 1990) partitions the pool into `C` clusters, C = target #shots.

**(b) Neuron Consensus Quantification** — informativeness proxy.
- `Q(x) = |N_act(x)|`  (Eq. 2) — count of unique valid activations.
- **Higher Q ⟹ lower consensus ⟹ more hallucination-prone ⟹ more valuable to annotate.** (Inverts the Chen-2025 finding "higher consensus ⟹ fewer hallucinations": for AFSL the *low*-consensus samples are the informative ones to label.)

### 2.3 Sample Scoring and Selection (§3.3)

Dual-criteria score per cluster `C_i` with medoid `µ_i`:
- Min-max normalize within cluster: `D̃_J(x, µ_i)` (Jaccard distance to medoid), `Q̃(x)` (unique-activation count).
- `Score(x) = τ · Q̃(x) + (1−τ) · D̃_J(x, µ_i)`  (Eq. 3), `τ ∈ [0,1]`.
- Select the highest-scoring sample from each cluster. `τ→0` = pure representativeness (cluster centrality); `τ→1` = pure hallucination-risk prioritization.

> ⚠ **Deployed τ/k fall outside the paper's own ablation sweep on several configs (see §7 flags).** The §5.2 sweep varies k∈[2000,10000] and §5.3 sweeps τ∈[0,1] on Qwen3 MMLU-Pro only, but Table 6 deploys k=500/1000 (Edu-Feedback Llama-8B/3.2 + Qwen3-8B) — *below* the swept range — and τ=1.0 (claimed to cause a "sharp drop" in Fig 4) for Llama-3.2-3B MMLU-Pro and Qwen3-8B/4B TREC.

---

## 3. Setup (§4.1–4.3)

- **Models (4, instruction-tuned, non-reasoning mode):** Llama-3.2-3B, Llama-3.1-8B, Qwen3-4B-Instruct-2507, Qwen3-8B.
- **Datasets (3, Table 5 verbatim, layout L766–772):**

| Dataset | Source Domain | #Class | #Candidate | #Test |
|---|---|---|---|---|
| MMLU-Pro | Multi-disciplinary | 10 | 2000 | 10032 |
| Edu-Feedback | Peer-feedback | 2 | 1799 | 14228 |
| TREC | Question | 6 | 5452 | 500 |

  - MMLU-Pro: 10-option reasoning across 14 domains; total dataset 12,034 samples (intro) → 2000 randomly selected as candidate pool, rest 10032 test (⚠ 12034−2000=10034 vs Table-5 10032, a 2-sample filter gap).
  - Edu-Feedback: binary explanatory-vs-non-explanatory peer-feedback classification from a real essay-writing course.
  - TREC: coarse-grained 6-way question classification (ABBR/DESC/ENTY/HUM/LOC/NUM).

- **Shots:** 5, 10, 20, 30; averaged over **3 inference runs** (different seeds); VLLM on **4×A100**, temperature **0.6**.
- **Baselines (6) + InfoType taxonomy (§4.3):**

| Method | InfoType(s) | Signal source |
|---|---|---|
| Random | N/A | fixed-seed random |
| Entropy (Highest / Diverse) | E. | output-logit predictive entropy |
| TypiClust | S. / L. / S.+L. | BERT + K-Means (semantic) |
| FastVoteK | S. / L. / S.+L. | graph + KNN semantic diversity |
| VoteK | S.+E. / S.+E.+L. | FastVoteK + entropy-bin stratification |
| Patron | S.+E. | graph uncertainty propagation + clustering |
| **NeuFS** | **Neuron** | internal FFN activations |

  InfoTypes: **S.**=Semantic (external embeddings), **E.**=Entropy (output logits), **L.**=Linguistic (sparse count + RST + topical features, App. C), **Neuron**=internal activations.

---

## 4. Main results

### 4.1 Reasoning — MMLU-Pro aggregated accuracy (Table 1, layout L338–380)

Average accuracy (± std) over 5/10/20/30-shot.

**(a) 8B models**

| Method | InfoType | Llama-3.1-8B | Qwen3-8B |
|---|---|---|---|
| Random | N/A | 0.325±0.007 | 0.388±0.079 |
| Patron | S.+E. | 0.313±0.013 | 0.416±0.074 |
| Entropy / Highest | E. | 0.317±0.007 | 0.394±0.095 |
| Entropy / Diverse | E. | 0.324±0.003 | 0.392±0.066 |
| TypiClust / S. | S. | 0.323±0.007 | 0.398±0.085 |
| TypiClust / L. | L. | 0.322±0.006 | 0.382±0.072 |
| TypiClust / S.+L. | S.+L. | 0.318±0.006 | 0.383±0.087 |
| FastVoteK / S. | S. | 0.316±0.004 | 0.379±0.103 |
| FastVoteK / L. | L. | 0.323±0.003 | 0.373±0.089 |
| FastVoteK / S.+L. | S.+L. | 0.318±0.009 | 0.402±0.075 |
| VoteK / S.+E. | S.+E. | 0.318±0.004 | 0.391±0.068 |
| VoteK / S.+E.+L. | S.+E.+L. | 0.309±0.007 | 0.402±0.075 |
| **NeuFS** | **Neuron** | **0.327±0.007** | **0.418±0.069** |

**(b) 3B & 4B models**

| Method | InfoType | Llama-3.2-3B | Qwen3-4B |
|---|---|---|---|
| Random | N/A | 0.249±0.005 | 0.412±0.015 |
| Patron | S.+E. | 0.244±0.009 | 0.391±0.045 |
| Entropy / Highest | E. | 0.245±0.005 | 0.430±0.024 |
| Entropy / Diverse | E. | 0.244±0.012 | 0.414±0.019 |
| TypiClust / S. | S. | 0.242±0.005 | 0.437±0.016 |
| TypiClust / L. | L. | 0.248±0.003 | 0.376±0.026 |
| TypiClust / S.+L. | S.+L. | 0.243±0.004 | 0.398±0.028 |
| FastVoteK / S. | S. | 0.245±0.001 | 0.386±0.017 |
| FastVoteK / L. | L. | 0.242±0.002 | 0.401±0.026 |
| FastVoteK / S.+L. | S.+L. | 0.242±0.003 | 0.420±0.014 |
| VoteK / S.+E. | S.+E. | 0.246±0.003 | 0.401±0.012 |
| VoteK / S.+E.+L. | S.+E.+L. | 0.233±0.008 | 0.374±0.012 |
| **NeuFS** | **Neuron** | **0.251±0.005** | **0.452±0.010** |

**Takeaway.** NeuFS is the column-max on **all 4** MMLU-Pro cells (327/418/251/452 ×10⁻³). Largest margin on Qwen3-4B: 0.452 vs Entropy-Highest 0.430 (+2.2pp) and TypiClust-S 0.437 (+1.5pp). Also beats Patron (S.+E.) — internal neuron consensus > combining external semantic density with predictive uncertainty. The MMLU-Pro numbers reproduce exactly from the appendix per-shot `Avg.` column: T7 Llama-8B NeuFS Avg Acc 0.3272→0.327; T7 Llama-3.2 NeuFS 0.2510→0.251; T10 Qwen3-8B NeuFS 0.4178→0.418; T10 Qwen3-4B NeuFS 0.4515→0.452 ✓ (4-table consistency triangle, see §6).

### 4.2 Classification — 8B models (Table 2, layout L398–416)

| Method | InfoType | Llama-8B Edu-F1 | Llama-8B Edu-Acc | Llama-8B TREC-F1 | Llama-8B TREC-Acc | Qwen3-8B Edu-F1 | Qwen3-8B Edu-Acc | Qwen3-8B TREC-F1 | Qwen3-8B TREC-Acc |
|---|---|---|---|---|---|---|---|---|---|
| Random | N/A | 0.645±0.053 | 0.662±0.065 | 0.777±0.052 | 0.764±0.070 | 0.612±0.084 | 0.625±0.097 | 0.844±0.022 | 0.823±0.030 |
| Patron | S.+E. | 0.643±0.027 | 0.690±0.025 | 0.807±0.046 | 0.809±0.045 | 0.659±0.022 | 0.709±0.037 | 0.825±0.023 | 0.824±0.017 |
| Entropy/Highest | E. | 0.619±0.027 | 0.625±0.030 | 0.790±0.029 | 0.785±0.031 | 0.555±0.049 | 0.557±0.051 | 0.837±0.035 | 0.839±0.031 |
| Entropy/Diverse | E. | 0.670±0.028 | 0.696±0.029 | 0.829±0.030 | 0.827±0.033 | 0.632±0.035 | 0.663±0.070 | 0.856±0.012 | 0.841±0.017 |
| TypiClust/S. | S. | 0.642±0.049 | 0.683±0.071 | 0.798±0.026 | 0.791±0.035 | 0.605±0.106 | 0.629±0.122 | 0.831±0.022 | 0.828±0.022 |
| TypiClust/L. | L. | 0.664±0.014 | 0.695±0.021 | 0.806±0.026 | 0.802±0.038 | 0.626±0.025 | 0.637±0.033 | 0.835±0.030 | 0.826±0.040 |
| TypiClust/S.+L. | S.+L. | 0.623±0.055 | 0.637±0.066 | 0.814±0.030 | 0.804±0.039 | 0.589±0.091 | 0.600±0.101 | 0.851±0.008 | 0.835±0.005 |
| FastVoteK/S. | S. | 0.658±0.012 | 0.694±0.004 | 0.828±0.024 | 0.825±0.030 | 0.640±0.027 | 0.662±0.038 | 0.840±0.017 | 0.831±0.021 |
| FastVoteK/L. | L. | 0.654±0.027 | 0.676±0.032 | 0.832±0.046 | 0.825±0.048 | 0.621±0.059 | 0.632±0.066 | 0.854±0.018 | 0.843±0.017 |
| FastVoteK/S.+L. | S.+L. | 0.657±0.042 | 0.688±0.058 | 0.809±0.051 | 0.813±0.052 | 0.614±0.097 | 0.631±0.108 | 0.861±0.022 | 0.854±0.011 |
| VoteK/S.+E. | S.+E. | 0.660±0.028 | 0.692±0.027 | 0.832±0.019 | 0.822±0.020 | 0.668±0.020 | 0.701±0.021 | 0.839±0.006 | 0.839±0.010 |
| VoteK/S.+E.+L. | S.+E.+L. | 0.653±0.043 | 0.676±0.054 | 0.818±0.052 | 0.810±0.051 | 0.631±0.032 | 0.648±0.047 | 0.839±0.006 | 0.826±0.022 |
| **NeuFS** | **Neuron** | 0.660±0.024 | **0.698±0.016** | **0.834±0.030** | 0.823±0.036 | 0.663±0.014 | **0.711±0.019** | **0.862±0.027** | **0.858±0.020** |

### 4.3 Classification — 3B & 4B models (Table 3, layout L417–435)

| Method | InfoType | Llama-3.2 Edu-F1 | Llama-3.2 Edu-Acc | Llama-3.2 TREC-F1 | Llama-3.2 TREC-Acc | Qwen3-4B Edu-F1 | Qwen3-4B Edu-Acc | Qwen3-4B TREC-F1 | Qwen3-4B TREC-Acc |
|---|---|---|---|---|---|---|---|---|---|
| Random | N/A | 0.557±0.029 | 0.558±0.030 | 0.699±0.041 | 0.707±0.049 | 0.653±0.080 | 0.667±0.094 | 0.859±0.037 | 0.860±0.037 |
| Patron | S.+E. | 0.569±0.111 | 0.597±0.134 | 0.727±0.051 | 0.749±0.033 | 0.685±0.027 | 0.742±0.030 | 0.842±0.023 | 0.853±0.024 |
| Entropy/Highest | E. | 0.501±0.080 | 0.506±0.074 | 0.663±0.022 | 0.638±0.030 | 0.555±0.065 | 0.557±0.064 | 0.855±0.012 | 0.857±0.014 |
| Entropy/Diverse | E. | 0.657±0.034 | 0.671±0.042 | 0.736±0.048 | 0.759±0.020 | 0.671±0.056 | 0.684±0.066 | 0.862±0.019 | 0.867±0.013 |
| TypiClust/S. | S. | 0.545±0.107 | 0.557±0.116 | 0.741±0.049 | 0.744±0.039 | 0.626±0.095 | 0.640±0.111 | 0.853±0.014 | 0.866±0.003 |
| TypiClust/L. | L. | 0.606±0.096 | 0.624±0.116 | 0.712±0.030 | 0.740±0.069 | 0.636±0.015 | 0.642±0.017 | 0.862±0.008 | 0.863±0.008 |
| TypiClust/S.+L. | S.+L. | 0.537±0.033 | 0.538±0.033 | 0.753±0.042 | 0.750±0.049 | 0.613±0.086 | 0.623±0.096 | 0.855±0.020 | 0.858±0.013 |
| FastVoteK/S. | S. | 0.507±0.046 | 0.509±0.044 | 0.760±0.026 | 0.761±0.030 | 0.673±0.029 | 0.707±0.047 | **0.873±0.010** | **0.871±0.008** |
| FastVoteK/L. | L. | 0.548±0.104 | 0.555±0.101 | 0.718±0.044 | 0.739±0.042 | 0.662±0.046 | 0.673±0.096 | 0.861±0.027 | 0.867±0.017 |
| FastVoteK/S.+L. | S.+L. | 0.541±0.059 | 0.543±0.061 | 0.726±0.034 | 0.775±0.017 | 0.655±0.083 | 0.679±0.054 | 0.847±0.026 | 0.857±0.025 |
| VoteK/S.+E. | S.+E. | 0.526±0.027 | 0.528±0.029 | 0.755±0.035 | 0.771±0.023 | 0.664±0.044 | 0.681±0.056 | 0.853±0.007 | 0.850±0.012 |
| VoteK/S.+E.+L. | S.+E.+L. | 0.572±0.065 | 0.577±0.076 | 0.757±0.025 | 0.776±0.024 | 0.681±0.055 | 0.702±0.067 | 0.854±0.037 | 0.846±0.023 |
| **NeuFS** | **Neuron** | **0.624±0.007** | **0.636±0.008** | 0.754±0.032 | 0.761±0.044 | **0.692±0.021** | **0.725±0.038** | 0.865±0.019 | **0.878±0.015** |

> ⚠ **NeuFS is NOT uniformly best on classification (honest scope).** On MMLU-Pro it wins all 4 cells, but on the classification grids it is *competitive, not dominant*:
> - Table 2 Llama-8B Edu-F1: Entropy-Diverse 0.670 > NeuFS 0.660.
> - Table 2 Llama-8B TREC-Acc: FastVoteK-S 0.825 > NeuFS 0.823 (NeuFS wins F1 0.834, loses Acc).
> - Table 3 Llama-3.2 Edu: Entropy-Diverse 0.657/0.671 > NeuFS 0.624/0.636.
> - Table 3 Qwen3-4B TREC: FastVoteK-S 0.873/0.871 > NeuFS 0.865/0.878 (NeuFS wins Acc, loses F1).
> The paper frames this honestly ("competitive 1st or 2nd-ranking", "state-of-the-art with Qwen3-8B on TREC"), but a reader should not read "NeuFS outperforms existing AFSL baselines" (abstract) as uniform dominance — it holds decisively on MMLU-Pro reasoning and selectively on classification.

---

## 5. Ablations (§5)

### 5.1 Representation variants (Table 4, layout L438–464)

NeuFS vs dense-representation variants on Qwen3-4B (Avg over 5/10/20/30-shot). Columns: `NeuFS` (neuron activations), `NeuFS w/ Qwen-Embed-0.6B` (decoder-based dense embedding), `NeuFS w/ SimCSE` (encoder-based BERT embedding) — same K-Medoids + consensus pipeline, only the sample-representation changes.

| Dataset | NeuFS (Neuron) | NeuFS w/ Qwen-Embed-0.6B | NeuFS w/ SimCSE |
|---|---|---|---|
| **MMLU-Pro Avg** | **0.4178±0.0687** | 0.4013±0.0887 | 0.4131±0.0577 |
| **Edu-Feedback Avg** | **0.7107±0.0193** | 0.6502±0.0489 | 0.6868±0.0409 |
| **TREC Avg** | **0.8575±0.0199** | 0.8407±0.0263 | 0.7948±0.0334 |

(Full 5/10/20/30-shot rows verbatim in layout L446–462.) NeuFS-neuron ≥ both dense variants on all 3 dataset-averages. Decoder-based Qwen-Embedding generally > encoder-based SimCSE, but both < neuron activations. Confirms sparse neuron sets > dense semantic vectors for few-shot selection.

### 5.2 Neuron-sparsity k (Figure 3, layout L476–496)

Sweep k ∈ {2000, 4000, 6000, 8000, 10000} on Qwen3-4B/8B MMLU-Pro (bar plot, avg over 5/10/20/30-shot). **Overall fluctuation is mild** (NeuFS robust to the activation threshold), but **Qwen3-4B is noticeably more sensitive than Qwen3-8B** — larger models have redundant knowledge circuits buffering against selection noise; smaller models are more sensitive to k. (Per-k bar heights are figure-bar reads, not transcribed.)

> ⚠ **The deployed k (Table 6) on Edu-Feedback (500, 1000) lies *below* this 2000–10000 sweep range.** The ablation does not cover the regime actually used for Edu-Feedback Llama-8B/3.2 + Qwen3-8B, so the "robust to k" claim is only validated in [2000,10000].

### 5.3 Sample-scoring weight τ (Figure 4, layout L498–555)

Sweep τ ∈ [0,1] on MMLU-Pro, Qwen3-4B (Fig 4a) and Qwen3-8B (Fig 4b), per-shot (5/10/20/30) + avg (dashed).
- **Qwen3-4B:** volatile across τ; avg performance *maximized at τ=0* (pure diversification) — neuron-consensus signals are noisier in smaller models, so diversity+representativeness dominate. Performance fluctuates/declines as τ→1.
- **Qwen3-8B:** smooth; avg *peaks at τ=0.5* (consensus complements diversity). Pushing τ→1 causes a sharp drop — consensus is valuable but cannot replace diversity coverage.
- (Per-τ curve points are figure-axis reads, not transcribed.)

> ⚠ **τ=1 is deployed (Table 6) despite Fig 4's "sharp drop at τ=1".** Llama-3.2-3B MMLU-Pro (τ=1), Qwen3-8B TREC (τ=1), Qwen3-4B TREC (τ=1) all use τ=1 as their best. Fig 4's τ-sweep is Qwen3-MMLU-Pro only, so the "τ=1 sharp drop" finding is model×dataset-specific, not universal.

---

## 6. Cross-table reconciliation (source-free)

The main aggregated tables (1–3) are the `Avg.` column of the appendix per-shot grids (Tables 7–12), so every main-table NeuFS cell has a 2-table witness:

| Main-table cell | Appendix witness (per-shot Avg) | Recompute |
|---|---|---|
| T1 Llama-8B MMLU-Pro NeuFS 0.327±0.007 | T7 L793 NeuFS Avg 0.3272±0.0068 | 0.327 ✓ |
| T1 Llama-3.2 MMLU-Pro NeuFS 0.251±0.005 | T7 L809 NeuFS Avg 0.2510±0.0049 | 0.251 ✓ |
| T1 Qwen3-8B MMLU-Pro NeuFS 0.418 | T10 L916 NeuFS Avg 0.4178±0.0687 | 0.418 ✓ |
| T1 Qwen3-4B MMLU-Pro NeuFS 0.452±0.010 | T10 L933 NeuFS Avg 0.4515±0.0099 | 0.452 ✓ |
| T2 Qwen3-8B Edu-FB NeuFS 0.663±0.014 / 0.711±0.019 | T11 L958 Avg 0.6627/0.7107 | ✓ ✓ |
| T2 Qwen3-8B TREC NeuFS 0.862±0.027 / 0.858±0.020 | T12 L999 Avg 0.8616/0.8575 | ✓ ✓ |
| T3 Qwen3-4B Edu-FB NeuFS 0.692 / 0.725 | T11 L975 Avg 0.6918/0.7250 | ✓ ✓ |
| T3 Qwen3-4B TREC NeuFS 0.865 / 0.878 | T12 L1016 Avg 0.8646/0.8783 | ✓ ✓ |

**Appendix per-shot grids (referenced, sourcing line-ranges):** Table 7 Llama×MMLU-Pro (L778–812), Table 8 Llama×Edu-Feedback (L817–853), Table 9 Llama×TREC (L859–895), Table 10 Qwen3×MMLU-Pro (L900–936), Table 11 Qwen3×Edu-Feedback (L942–978), Table 12 Qwen3×TREC (L983–1019). Each is the full 13-method × {5,10,20,30,Avg} × {Macro-F1, Acc} grid; only NeuFS (labelled "NeuronPattern") Avg cells cited above. **0 numeric prose-vs-table contradictions** (unlike the iter-30/31/34/46 inconsistency class) — the paper is internally consistent.

---

## 7. Appendix A — statistical validation of the neuron signal

### A.1 RQ.1 — Does neuron consensus correlate with prediction correctness? (§A.1, Fig 5)

On MMLU-Pro, group correctly vs incorrectly predicted samples into 50 bins each by #Unique Neuron Activations.
- **Two-sample t-test: t = −3.5698, p < 0.001** — incorrectly-predicted samples activate *significantly more* neurons (lower consensus).
- 10 equal-width bins (Fig 5): accuracy (blue) **decreases** with #activations (linear-trend slope **−0.0165**, red dashed); #activations (orange, right axis) monotonically increases across bins (33.0→33.8 ×10³). Validates lower-consensus ⟹ harder/more-error-prone.

### A.2 RQ.2 — Do low-consensus samples *as demonstrations* improve inference? (§A.2, Fig 6, Eq 4)

For each of 30 MMLU-Pro queries, slide a 5-shot window over candidates ranked by #Unique Activations; measure per-sample relative accuracy `∆Acc = Acc_sample − Acc_query` (Eq 4, query-difficulty-bias-removed) across 20 equal-width bins.
- **Linear fit: Pearson r = 0.6664, p < 0.001** (Fig 6) — selecting demonstrations with *higher* #Unique Activations (lower consensus, more model-uncertain) yields measurably better inference. This is the falsifiable causal claim behind NeuFS's Q(x) informativeness signal.

---

## 8. Hyperparameters (Table 6, layout L728–742)

Per model × dataset: K (global top-k activation threshold, Eq for η) and τ (scoring weight, Eq 3). (Per-layer pool n=2000 fixed, footnote 2.)

| Model | MMLU-Pro K / τ | Edu-Feedback K / τ | TREC K / τ |
|---|---|---|---|
| Llama-3.1-8B | 6000 / 0.8 | 500 / 0.2 | 3000 / 0.25 |
| Llama-3.2-3B | 6000 / 1.0 | 1000 / 1.0 | 4000 / 0.5 |
| Qwen3-8B | 8000 / 0.5 | 1000 / 0.5 | 5000 / 1.0 |
| Qwen3-4B-Instruct-2507 | 8000 / 0.5 | 4000 / 0.5 | 5000 / 1.0 |

K spans **500–8000** across configs; τ spans **0.2–1.0**. Note the Llama-3.2-3B MMLU-Pro config uses the *extremes* K=6000, τ=1.0 (pure consensus prioritization) — yet still wins the MMLU-Pro cell, the opposite of the Fig-4 Qwen3-8B "τ=1 sharp drop" finding (see §5.3 flag).

---

## 9. Strengths / Limitations / Verdict

**Strengths.**
- First **model-internal** selection signal for AFSL: replaces both lossy proxies (external embeddings, output entropy) with a single principled source (FFN neuron activations via early unembedding). Mechanistically grounded in the neuron-consensus↔hallucination link (Chen 2025; Cao 2025).
- Dual-criteria score (Eq 3) cleanly separates the two selection goals: diversity (Jaccard K-Medoids on activation sets) vs informativeness (consensus count Q(x)).
- Wins all 4 MMLU-Pro aggregated-accuracy cells; statistically validates the signal (t=−3.5698, r=0.6664, both p<0.001).
- Cross-table consistency triangle (T1↔T7/T10, T2/T3↔T8/T9/T11/T12): all 8 audited NeuFS cells reproduce — paper is internally consistent.

**Limitations (paper-stated §Limitations + this breakdown's flags).**
- Requires **open-weights access** to FFN activations + unembedding matrices (not black-box APIs).
- Higher compute overhead than retrieving static pre-computed embeddings (two-stage filtering still scans large unlabeled pools).
- ⚠ **Not uniformly best on classification** — competitive 1st/2nd, edged by Entropy-Diverse / FastVoteK on several Llama cells (§4.3).
- ⚠ **Deployed hyperparameters fall outside the ablation sweep** on some configs: Edu-Feedback K∈{500,1000} below the §5.2 [2000,10000] sweep; τ=1 deployed despite the §5.3 "τ=1 sharp drop" finding (which is Qwen3-MMLU-Pro-only).
- Evaluation on 3 datasets / 4 models (3B–8B); no >8B test; MMLU-Pro only reasoning benchmark.

**Verdict.** A clean, mech-interp-grounded AFSL method whose central claim (internal neuron dynamics > output-level proxies for selection) holds decisively on reasoning (MMLU-Pro, all 4 cells) and holds *competitively* on classification. The contribution is the **selection-signal paradigm shift** (output→internal) plus the dual diversity/consensus decomposition — not uniform SOTA across every cell. Sibling-in-spirit to **N-OPSD** (iter 54, same author group + same early-unembedding neuron-activation machinery), but where N-OPSD uses neuron activations to select *unlabeled self-distillation data* (annotation-free), NeuFS uses them to select *samples worth human-annotating* as few-shot demos (annotation-efficient) — the active-learning counterpart in the neuron-selection lineage.

---

## Repo subarea placement

NeuFS is the repo's **first active-learning / selective-annotation paper**. Distinct from:
- **N-OPSD** (iter 54): same neuron-activation extraction (early-unembedding, top-k), but annotation-*free* self-distillation data selection (no human labels); NeuFS is annotation-*efficient* (selects what to label).
- **expander-SAE / CoAx / refusal-subspaces**: mech-interp *interpretation* of neuron features; NeuFS *operationalizes* neuron activations for data selection.
- The agentic-RL / reward-design lineage: NeuFS involves no RL, no reward model — pure unsupervised selection + human annotation.
- ICL/few-shot retrieval work: NeuFS selects the *demonstration pool* under an unlabeled-data budget, not test-time retrieval over an annotated set.
