# kNNGuard: Turning LLM Hidden Activations into a Training-Free Configurable Guardrail

**arXiv:** 2607.02072v1 [cs.LG] (2 Jul 2026) — Preprint.
**Authors:** Mahmoud Abdelfattah¹, Hamid Nasiri¹, Peter Garraghan¹·². ¹Lancaster University, ²Mindgard.
**Code:** none stated in paper.
**Subarea (repo lineage):** FIRST training-free, inference-time safety guardrail in the repo. Distinct from `refusal-subspaces` (which *modifies weights* — ablates a learned RFM refusal cone) and `subliminal-clocks` (which *steers* the residual stream of a diffusion LM): kNNGuard touches **no weights** and does **no steering** — it reads a frozen LLM's multi-layer last-token activations and classifies a prompt by cosine kNN against a 50-example labeled bank, fusing that activation signal with a MiniLM embedding-kNN via an adaptive confidence rule. Sibling-in-spirit to the activation-monitoring interpretability line (Arditi et al. 2024 refusal direction; SafeSwitch), but operationalized as a deployable, configurable guardrail rather than a mechanistic probe.

---

## TL;DR

Fine-tuned guardrail classifiers (Llama Nemotron Topic/Safety Guard, Llama Guard 3, Prompt Guard 2) are accurate in-distribution but expensive to adapt to a new domain — each new threat category needs a fresh curated corpus, a LoRA run, and re-validation. Embedding-kNN guardrails are fast but leak nuance (high false-positive / attack-success rates). **kNNGuard** reuses a frozen LLM as a feature extractor: a small bank of 50 safe + 50 unsafe prompts is run once through the model, the last-token hidden activations from 9 evenly-spaced layers are cached, and each layer is Fisher-discriminant-weighted by how well it separates the two classes (`J_ℓ = B_ℓ / W_ℓ`). At inference, a prompt's risk score is the unsafe-fraction among its `k=13` nearest bank neighbours, taken in activation space (**kNNGuard LE**) and fused with the same score in MiniLM embedding space via an adaptive confidence rule (**kNNGuard FE**). Across six domains (Code Instructions / Code Outputs / Medical / Safety / Jailbreak / Prompt Injection), kNNGuard FE averages **F1 87.4%** at **FPR 12.9%** and **45.9 ms/prompt** — competitive with or better than fine-tuned SOTA, **2.7× faster** than Llama Nemotron Topic Guard V1 (126 ms) and **~10× faster** than Nemotron Safety Guard V2 (454.6 ms), with **no gradient updates**. Domain adaptation = swap the 50-example bank, constructible in **7.95 s** (a **3156×** speedup over LoRA fine-tuning on equivalent data).

---

## 1. Problem & positioning

Two existing guardrail families, both with a structural weakness:

1. **Fine-tuned classifiers** (Llama Nemotron Topic Guard V1 = LoRA on Llama-3.1-8B; Nemotron Safety Guard V2 = 8B, 23-category; Llama Guard 3 = 8B, 14-category; WildGuard = Mistral-7B, 13-category; Prompt Guard 2 = 86M DeBERTa). Strong in-distribution accuracy, but operationally expensive to adapt (new corpus + retrain + re-validate) and distributionally fragile — Hackett et al. 2025 show simple character-level perturbations and paraphrasing bypass them.
2. **Embedding kNN** (MiniLM + cosine kNN, as in NeMo Guardrails). Low latency but surface-level: semantically ambiguous inputs (benign coding query vs malicious code-execution) are indistinguishable in embedding space.

kNNGuard's bet: the LLM's **internal activations** encode safety/topic distinctions that sentence embeddings do not (Han et al. 2025, SafeSwitch), so a non-parametric decision surface over those activations should generalize better than embedding-kNN while requiring no fine-tuning. The configurability claim is the operational punchline — adapting to a new domain or threat is a bank swap + optional system-prompt revision, no weight changes (`∇Θ_F L = 0, ∇Θ_E L = 0`, Eq. 23).

---

## 2. Method

Two-phase architecture (**Figure 1**):

### 2.1 Bank building (once)
A bank of `n = 50` safe + `n = 50` unsafe prompts per domain is formatted with an optional domain-specific system prompt `x̃ = P(x; π)` (Eq. 3), run through the frozen LLM, and the **last-token** hidden activations from `M = 9` layers (Llama-3.1-8B layers `{0, 4, 8, …, 31}` — input-proximal through final) are cached. In parallel the same prompts are encoded by a frozen MiniLM sentence embedder. Two cached banks result (Eqs. 13–14): `B_act` (activations) and `B_emb` (embeddings).

### 2.2 Fisher-discriminant layer weighting (kNNGuard LE)
For each layer `ℓ`, class means `µ_c^(ℓ)` (Eq. 5), between-class separation `B_ℓ = ‖µ_0 − µ_1‖² / (2 d_ℓ)` (Eq. 7), within-class dispersion `W_ℓ` (Eq. 8), and Fisher score `J_ℓ = B_ℓ / W_ℓ` (Eq. 9). Softmax over layers gives ensemble weights `α_ℓ = exp(J_ℓ) / Σ exp(J_j)` (Eq. 10). The layer-ensemble representation is the weighted concatenation of `ℓ2`-normalized activations (Eq. 11). Layers that best separate safe from unsafe get the most weight — a data-driven answer to "which layer carries the safety signal?"

### 2.3 kNN risk score (each branch)
Cosine distance `d_r(x, x_i) = 1 − ϕ_r(x̃)⊤ϕ_r(x̃_i)` (Eq. 15). Branch risk = unsafe-fraction among `k_r` nearest bank neighbours (Eq. 16): `s_act(x), s_emb(x) ∈ [0,1]`. `k_act = k_emb = 13` (selected by leave-one-out CV on the bank, sweeping odd `k ∈ {1,3,…,21}`).

### 2.4 Fused-ensemble adaptive decision rule (kNNGuard FE)
Threshold `τ = 0.5`. Per-branch confidence = distance of its score from `τ`: `c_act = |s_act − τ|`, `c_emb = |s_emb − τ|` (Eq. 18). Confidence gap `Δ(x) = |c_act − c_emb|` (Eq. 19) against gap threshold `γ = 0.1`:
- If `Δ > γ`: take the **more-confident** branch's score outright (winner-takes-all).
- Else: confidence-weighted blend `s_FE = (c_act·s_act + c_emb·s_emb)/(c_act + c_emb)` (Eq. 20).

Decision `g_FE(x) = 1[s_FE(x) ≥ τ]` (Eq. 21–22): block if unsafe/off-topic, allow if safe/on-topic. A fixed `α`-blend `s_α = λ s_act + (1−λ) s_emb` (Eq. 17) is the non-adaptive fallback. **Figure 2** diagrams the rule.

**Hyperparameters** (`τ = 0.5, γ = 0.1, k = 13, n = 50/class`, 9 layers) are selected on a validation set by minimizing empirical guardrail loss (Eq. 24) subject to `Θ_F, Θ_E` fixed (Eq. 25).

---

## 3. Experimental setup (paper §4)

**Backbone (main):** Llama-3.1-8B-Instruct (chosen to apples-to-apples compare against Nemotron Topic Guard V1, which LoRA-finetunes the same model). Additional backbones tested (App. A.2, Tables 6–7): Mistral-7B-Instruct-v0.3, Phi-4-mini-instruct, Gemma-4-12B-it (instruct), Gemma-4-12B (base), Llama-3-8B-Instruct-abliterated-v2.

**Datasets — 16 unique, 6 domains (Table 1).** Bank and eval sets are deliberately *distinct sources* to test distributional-shift robustness.

| Domain | Bank Safe | Bank Unsafe | Eval Safe | Eval Unsafe |
|---|---|---|---|---|
| Coding Instructions | MBPP | *(same src, diff label)* | Code (instr. col, Tarun 2023) | Code (instr. col) |
| Coding Outputs | PromptSet | Alpaca | Code (output col) | Dolly-15k |
| Medical | MedMCQA | *(same src)* | ChatDoctor | *(—)* |
| Safety | Aegis Safety 2.0* | | Safety Benchmark (qualifire) | Prompt Safety (SalKhan12) |
| Jailbreak | Jailbreak Classification (Hao)* | | PI Benchmark (rogue-security) | WildJailbreak |
| Prompt Injection | BIPIA-GPT* | | Deepset PI | PI Dataset (NeurAlchemy) |

> `*` = same source dataset used for both safe and unsafe classes (the two banks built from different labels within it). All eval sets = 4000 mixed prompts except Jailbreak & Prompt Injection = 2000. *(Sourcing: paper_layout.txt L439–460, Table 1.)*

**Guardrails compared (7):** kNNGuard FE; kNNGuard FE (No Sys.); Llama Nemotron Topic Guard V1 (LoRA 8B); Llama Nemotron Safety Guard V2 (8B, 23-cat); Llama Guard 3 (8B, 14-cat); Llama Prompt Guard 2 (86M DeBERTa); Embedding-kNN (MiniLM). `*`-tagged guardrails (Prompt Guard 2, Safety Guard V2, Llama Guard 3) evaluated on relevant domains only → not fully comparable.

**Metrics:** Latency (ms/prompt), F1, FPR, FNR (≈ ASR in security domains), Recall. Bank: `k=13`, `n=50/class`, 9 layers. All guardrails at threshold 0.5, deterministic decoding. NVIDIA RTX 6000 Ada GPU.

---

## 4. Results

### 4.1 Headline summary — averaged over all six domains (Table 2)

| Guardrail | F1 (%) | Recall (%) | FPR (%) | FNR (%) | Latency (ms) |
|---|---|---|---|---|---|
| **Llama kNNGuard FE** | **87.4** | 86.6 | **12.9** | 13.4 | 46.8 |
| Llama kNNGuard FE (No Sys.) | 84.2 | 93.5 | 38.5 | 6.5 | 32.7 |
| Llama Nemotron Topic Guard | 82.7 | 90.5 | 34.7 | 9.5 | 126.0 |
| Prompt Guard 2* | 70.4 | 58.6 | 12.5 | 41.4 | 9.7 |
| Llama Nemotron Safety Guard V2* | 79.2 | 77.0 | 12.9 | 23.0 | 454.6 |
| Llama Guard 3* | 74.2 | 62.7 | 4.6 | 37.3 | 104.5 |
| Embedding KNN | 79.6 | 80.9 | 31.6 | 19.1 | 4.0 |

`*` evaluated on relevant domains only — not fully comparable. *(Sourcing: L585–596.)*

> **Source-free reconciliation:** every Table-2 average reproduces from the per-domain cells of Tables 3+4 (kNNGuard FE F1 (99.1+99.3+95.3+73.7+75.0+82.2)/6 = **87.43→87.4** ✓; FPR → 12.95; FNR → 13.38→13.4 ⇒ Recall 86.6 ✓; No-Sys F1→84.2, FPR→38.55, FNR→6.55; Topic Guard F1→82.65; Embedding-kNN F1→79.57, FPR→31.6, FNR→19.1; Prompt Guard 2 over its 2 domains F1→70.45, FPR→12.5, FNR→41.45). **Display-rounding caveat (iter-45 pattern):** re-averaging the *displayed* 1-dp per-domain cells lands ±0.1 off the printed average on boundary cases (FE FPR 12.95 prints 12.9; No-Sys FPR 38.55 prints 38.5; Topic Guard F1 82.65 prints 82.7) because the paper computes averages from full-precision per-prompt values, not the rounded cells — **not** transcription errors.

**Headline deltas reconcile:** `2.7× = 126.0/46.8 = 2.69` ✓; `~10× = 454.6/46.8 = 9.71` ✓; `61× ≈ 25092/413 = 60.8` (prose "approximately 61×") ✓; `3156× = 25092/7.95 = 3156.2` ✓.

### 4.2 Topical domains — Code Instructions, Code Outputs, Medical (Table 3)

| Guardrail | Avg Lat (ms) | Code Instr. F1/FPR/FNR | Code Outputs F1/FPR/FNR | Medical F1/FPR/FNR |
|---|---|---|---|---|
| **kNNGuard FE** | 46.8 | **99.1** / 0.9 / 0.9 | **99.3** / 0.6 / 0.8 | **95.3** / 0.5 / 8.5 |
| kNNGuard FE (No Sys.) | 32.7 | 98.5 / 1.6 / 1.5 | 98.3 / 2.9 / 0.5 | 85.0 / 32.6 / 2.1 |
| Nemotron Topic Guard V1 | 126.1 | 83.4 / 39.1 / 0.5 | 99.6 / 0.2 / 0.5 | 93.9 / 12.0 / 0.9 |
| Embedding-kNN | 4.0 | 87.5 / 4.5 / 18.8 | 88.7 / 2.9 / 17.9 | 87.3 / 0.9 / 21.9 |

*(Sourcing: L639–651.)* kNNGuard FE wins F1 on all three (≥95.3%) at FPR ≤ 0.9%. **Distributional fragility of the fine-tuned SOTA:** Nemotron Topic Guard V1 collapses from 99.6% F1 (Code Outputs) to 83.4% F1 at 39.1% FPR on the closely-related Code Instructions task — sensitivity to the *linguistic style* of its training distribution. kNNGuard FE stays ≥99.1% / ≤0.9% FPR across both coding sub-domains with the *same* system prompt. **Medical is the hardest topical domain** (semantic ambiguity); kNNGuard FE reaches 95.3% F1 (FPR 0.5%, FNR 8.5%) — dropping the system prompt collapses F1 to 85.0% and FPR jumps to 32.6%.

### 4.3 Security & safety domains — Safety, Jailbreak, Prompt Injection (Table 4)

| Guardrail | Avg Lat (ms) | Safety F1/FPR/ASR | Jailbreak F1/FPR/ASR | Prompt Inj. F1/FPR/ASR |
|---|---|---|---|---|
| **kNNGuard FE** | 46.8 | 73.7 / 34.6 / 14.3 | 75.0 / 25.8 / 29.6 | **82.2** / 15.3 / 26.2 |
| kNNGuard FE (No Sys.) | 32.7 | 64.7 / 70.1 / 6.9 | 75.0 / 43.9 / 22.2 | 83.7 / 80.2 / **6.1** |
| Nemotron Topic Guard V1 | 126.1 | 73.5 / 41.1 / 9.5 | 70.5 / **97.4** / 9.7 | 75.0 / 18.4 / 35.9 |
| Nemotron Safety Guard V2 | 454.7 | 79.2 / 12.9 / 22.9 | — | — |
| Llama Guard 3 | 104.4 | 74.2 / 4.7 / 37.3 | — | — |
| Prompt Guard 2 | 9.6 | — | 61.9 / 16.8 / 50.2 | 79.0 / 8.2 / 32.7 |
| Embedding-kNN | 4.0 | 61.9 / 60.3 / 18.5 | 78.4 / 60.9 / 9.0 | 73.6 / 60.2 / 28.5 |

`–` = not evaluated on that domain. *(Sourcing: L727–743.)*

**Key findings:**
- **No single competitor balances FPR and ASR across all three security domains.** Each fine-tuned/lightweight guardrail breaks on at least one axis: Nemotron Topic Guard V1 hits **97.4% FPR on jailbreak** (blocks nearly all benign prompts — operationally non-viable); Embedding-kNN reaches 60.3–60.9% FPR on safety/jailbreak; Prompt Guard 2 lets **50.2% of jailbreaks** through; Llama Guard 3 cuts FPR to 4.7% but at 37.3% ASR (23pp higher ASR than kNNGuard FE — a permissive classifier). Nemotron Safety Guard V2 wins Safety F1 (79.2%, FPR 12.9%) but at ~10× kNNGuard FE's latency and 8.6pp higher ASR.
- **kNNGuard FE is the only method with moderate, non-extreme values on both error components across all three** — the operationally preferable profile when both attack detection and service availability matter.
- **Prompt injection:** kNNGuard FE and Prompt Guard 2 are within 0.5pp combined error but split it differently — kNNGuard FE prioritizes lower ASR (26.2 vs 32.7) at higher FPR (15.3 vs 8.2), the safer trade-off for injection where missing an attack costs more than over-blocking.

### 4.4 System-prompt, representation-space, and cost analysis (paper §5.3, Figs 6–8, Table 5)

**System prompt impact (Figure 6, averaged across domains):** with system prompt → F1 **0.874**, Recall 0.866, FPR **0.129**, FNR 0.134; without → F1 0.842, Recall **0.935**, FPR **0.385**, FNR **0.065**. Adding the prompt **raises F1 0.842→0.874** and **cuts FPR 66%** (0.385→0.129, a *relative* reduction) at the cost of recall — it makes the classifier more conservative (higher precision, lower recall). Cost: +14 ms latency (32.7→46.8 ms).

> ⚠ **Mixed pp-vs-% convention in §5.2–5.3 prose (transcribed verbatim, not reconciled):** the "66%" FPR reduction is genuinely *relative* ((0.385−0.129)/0.385 = 66.5% ✓). But two nearby §5.2 claims use "percent" for *absolute percentage-point* differences: "the No System Prompt variant achieves the lowest ASR on this domain (6.1%, a **20.1% reduction relative** to kNNGuard FE)" — true drop is 26.2→6.1 = **20.1 pp absolute** (= 76.7% relative); and "at an **FPR 64.9% higher**" — true is 80.2 vs 15.3 = **64.9 pp absolute** (= 424% relative). The "relative to" wording is loose; the figures are absolute pp. Reader beware: the same unit symbol "%" means two different things in adjacent sentences.

**Representation geometry (Figure 7):** t-SNE of the bank shows kNNGuard activations cluster safe/unsafe far more sharply than MiniLM embeddings. **Silhouette scores** (figure labels): Embedding-kNN — Medical **0.391**, Coding **0.532**; kNNGuard FE — Medical **0.739**, Coding **0.813**. Relative improvement: Medical **89.0%** ((0.739−0.391)/0.391 ✓), Coding **53.2%** ((0.813−0.532)/0.532 = 52.8% ✓).

> ⚠ **Prose-vs-figure Silhouette inconsistency (transcribed verbatim, flagged not reconciled):** §5.3 prose states the embedding-space Silhouette scores are "**0.527** for the Coding domain and **0.342** for the Medical domain", but the Figure-7 labels read **0.532** (Coding) and **0.391** (Medical). The headline relative improvements (89.0% / 53.2%) reconcile to the **figure** labels, NOT the prose values (prose Medical 0.342 would imply 116%, prose Coding 0.527 → 54.3%). Coding is a near-tie (0.527 vs 0.532); Medical diverges (0.342 vs 0.391). The figure labels are the operative numbers.

**Deployment / adaptation cost (Figure 8, Table 5):**

| Method | Samples | Time (s) | Speedup |
|---|---|---|---|
| LoRA Fine-Tuning | 13,854 | 25,092 | 1× |
| kNNGuard Bank Building | 13,854 | 413 | 60.8× |
| kNNGuard Bank Building (Main) | 100 | 7.95 | 3156× |

*(Sourcing: L931–939.)* Building a full kNNGuard bank from 13,854 prompts (the same count used to train Nemotron Topic Guard V1) takes **6.89 min** vs **6.97 h** for end-to-end LoRA — a **~61×** speedup. The recommended 100-sample bank takes **7.95 s** → **3156×** speedup, enabling real-time / session-based domain adaptation. (`413.35 s/60 = 6.89 min` ✓; `25092/3600 = 6.97 h` ✓; `25092/413 = 60.8` ✓; `25092/7.95 = 3156.2` ✓.)

### 4.5 Backbone robustness — kNNGuard FE across 6 LLMs (Tables 6 & 7)

**Topical domains (Table 6):**

| Backbone | Avg Lat (ms) | Code Instr. F1/FPR/FNR | Code Outputs F1/FPR/FNR | Medical F1/FPR/FNR |
|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | 46.8 | 99.1 / 0.9 / 0.9 | 99.3 / 0.6 / 0.8 | 95.3 / 0.5 / 8.5 |
| Mistral-7B-Instruct-v0.3 | 44.9 | 98.0 / 0.9 / 3.0 | 99.2 / 1.0 / 0.5 | 93.4 / 0.6 / 11.8 |
| Phi-4-mini-instruct | 25.2 | 99.0 / 1.3 / 0.8 | 99.1 / 1.3 / 0.4 | 93.7 / 0.9 / 11.1 |
| Gemma-4-12B-it (instruct) | 54.8 | 98.6 / 1.7 / 1.1 | 98.7 / 1.4 / 1.2 | **97.0** / 0.2 / 5.7 |
| Gemma-4-12B (base) | 61.7 | 97.2 / 5.1 / 0.6 | 97.3 / 4.1 / 1.5 | 86.0 / 22.1 / 7.9 |
| Llama-3-8B-Instruct-abliterated-v2 | 44.3 | 97.1 / 2.2 / 3.6 | 98.9 / 1.6 / 0.6 | 91.0 / 0.4 / 16.2 |

**Adversarial domains (Table 7):**

| Backbone | Avg Lat (ms) | Safety F1/FPR/FNR | Jailbreak F1/FPR/FNR | Prompt Inj. F1/FPR/FNR |
|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | 46.8 | 73.7 / 34.6 / 14.3 | 75.0 / 25.8 / 29.6 | 82.2 / 15.3 / 26.2 |
| Mistral-7B-Instruct-v0.3 | 44.9 | 71.1 / 36.6 / 17.5 | 75.0 / 35.4 / 25.7 | 50.3 / 7.6 / 65.5 |
| Phi-4-mini-instruct | 25.2 | 74.5 / 35.4 / 12.1 | 82.0 / 26.1 / 18.3 | 83.3 / 20.7 / 23.0 |
| Gemma-4-12B-it (instruct) | 54.8 | 70.5 / 52.6 / 6.8 | 83.1 / 40.2 / 9.6 | 85.6 / 11.3 / 21.9 |
| Gemma-4-12B (base) | 61.7 | 61.3 / 67.2 / 15.7 | 72.1 / 42.7 / 27.4 | 83.6 / **96.9** / 1.9 |
| Llama-3-8B-Instruct-abliterated-v2 | 44.3 | 65.5 / 35.6 / 27.8 | 70.2 / 34.5 / 33.2 | 62.4 / 10.5 / 52.9 |

*(Sourcing: Table 6 L1245–1274; Table 7 L1287–1315.)*

> **Cross-table consistency:** the Llama-3.1-8B-Instruct rows of Tables 6 and 7 are byte-identical to the kNNGuard FE rows of Tables 3 and 4 respectively (the main-text results ARE the Llama-3.1-8B-Instruct backbone) — a free source-free check that the appendix tables are self-consistent with the main results.

**Findings:**
- **Topical F1 is backbone-insensitive** — all six models land 86.0–99.3% across the three topical domains; the Llama/Phi/Mistral trio is near-identical on topical and general-safety tasks.
- **Prompt Injection is the most backbone-sensitive domain** (heterogeneous injection strategies interact directly with each model's instruction-following mechanism): Mistral-7B collapses to 50.3% F1 / 65.5% ASR; the abliterated and base models also degrade sharply there.
- **Base (non-instruct) models fail on injection FPR** — Gemma-4-12B (base) hits **96.9% FPR** on Prompt Injection (blocks nearly everything; ASR 1.9% is vacuous). Instruction-tuning materially shapes the injection-detection geometry.
- **Safety-direction ablation (abliterated model):** Llama-3-8B-Instruct-abliterated-v2 (safety direction removed) underperforms Llama-3.1-8B-Instruct on every adversarial column (Safety 65.5 vs 73.7; Jailbreak 70.2 vs 75.0; Prompt Inj. 62.4 vs 82.2) — consistent with kNNGuard partially leveraging the safety direction. ⚠ *Caveat:* the two models differ beyond ablation (Llama-3 vs Llama-3.1, 8B), so this is not a clean single-variable ablation.

---

## 5. Strengths

- **Genuinely training-free & configurable** (`∇Θ_F = ∇Θ_E = 0`): the only per-domain artifact is a 50-example bank + optional system prompt, swappable in 7.95 s. This is the operational contribution — session-based, context-specific guardrails become feasible.
- **Balanced error profile on security domains** where every competitor breaks on at least one axis — the practically important property for deployment.
- **Fisher-weighted multi-layer ensemble** is a principled, data-driven answer to "which layer carries the safety signal," avoiding a single-direction dependence.
- **Adaptive confidence fusion** (winner-takes-all when branches disagree strongly, blend when similarly uncertain) is a clean rule that lets each representation contribute where it is strongest.

## 6. Limitations & honest scope

- **Absolute security-domain F1 is modest** (Safety 73.7, Jailbreak 75.0, Prompt Inj. 82.2) — kNNGuard FE is the *most balanced*, not the *most accurate*, on security. Topic Guard V1 reaches lower jailbreak ASR (9.7%) and Prompt Guard 2 lower prompt-injection FPR (8.2%); kNNGuard trades both off rather than dominating.
- **No-sys variant's low ASR is a false comfort** — the 6.1% Prompt-Injection ASR comes at 80.2% FPR (blocks most benign prompts). The system prompt is doing real geometric work (Silhouette +89% Medical); kNNGuard is not robustly system-prompt-free.
- **Bank-sample variance is acknowledged but only lightly quantified** (Figure 3 shows ±1-std bars over 5 randomized draws; "results varied only marginally"). No per-domain variance table — the headline 87.4% has no reported confidence interval.
- **Static-bank evaluation** — Nasr et al. 2025 (cited) show adaptive white-box adversaries bypass defenses that look robust on static benchmarks; kNNGuard's bank could be reverse-engineered by an adversary with white-box access. The paper flags this motivation but does not evaluate adaptive attacks.
- **Backbone ablation is confounded** (Llama-3 vs Llama-3.1, 8B, plus abliteration) — the safety-direction claim is directional, not isolated.
- **k = 13 and the 9-layer set are global defaults** selected by aggregated LOOCV; per-domain tuning might help the weakest domain (Jailbreak) but is not explored.

## 7. Verdict

A clean, well-motivated training-free guardrail whose real contribution is **operational configurability** (bank-swap domain adaptation in seconds, no weight changes) plus a **balanced FPR/ASR profile on security domains** where every fine-tuned competitor breaks on at least one axis. The headline F1 (87.4% avg) is competitive rather than dominant, and the security-domain absolute numbers (73–82% F1) confirm this is a *practical-deployment* paper, not a *push-the-SOTA* paper. The mechanism — Fisher-weighted multi-layer activation kNN + adaptive embedding fusion — is falsifiable (Silhouette separation, per-layer Fisher scores, the system-prompt geometry effect all reconcile to source) and the cost claim (3156× adaptation speedup) is exactly verifiable. Most citable single result: on the three security domains kNNGuard FE is the only evaluated method that avoids an extreme error component on *any* axis, which is the property a production deployer actually needs.

---

## Source-first verification notes

- **paper_layout.txt** (pdftotext -layout, 1315 lines, 17 pp incl. appendices) is the authoritative extract; all 7 explicit tables (T1–T7) transcribed verbatim with sourcing line-ranges.
- **Source-free reconciliation (all pass):** Table-2 averages reproduce from T3+T4 per-domain cells (display-rounding ±0.1 on boundary cases, computed from full precision — iter-45 pattern, not transcription errors); T5 speedups (2.7×, 9.71≈10×, 60.8≈61×, 3156×) all recompute; Silhouette relative improvements (89.0% Medical, 52.8≈53.2% Coding) reconcile to **figure** labels; T6/T7 Llama-3.1-8B-Instruct rows byte-identical to T3/T4 kNNGuard FE rows (cross-table consistency triangle); cost time conversions (413 s=6.89 min, 25092 s=6.97 h) exact.
- **Inline ⚠ flags (paper-internal, transcribed verbatim not reconciled):** (1) §5.3 Silhouette prose values (Coding 0.527, Medical 0.342) disagree with Figure-7 labels (0.532, 0.391) — the 89.0%/53.2% improvements match the figure, not the prose; (2) §5.2–5.3 "%" convention is mixed — "66% FPR reduction" is relative, but "20.1% ASR reduction relative to" and "64.9% FPR higher" are absolute percentage-point differences; (3) backbone safety-direction ablation is confounded (Llama-3 vs 3.1, 8B + abliteration), so the directional finding is not isolated.
- **Figure-derived values caveated not back-filled:** Figure-3 std-bar values, Figure-4/5 error-decomposition bar heights, Figure-10/11 per-backbone radar points are figure-label reads (axis-tick / scattered-label only) and were not transcribed as precise cells — the verbatim substance lives in the 7 explicit tables + prose-confirmed ranges, consistent with the repo-wide figure-weakness rule.
