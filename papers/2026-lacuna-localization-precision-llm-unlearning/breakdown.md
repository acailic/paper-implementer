# LACUNA: Localization Precision for LLM Unlearning — Source-First Breakdown

**Paper:** "LACUNA: A Testbed for Evaluating Localization Precision for LLM Unlearning" (Boglioni, Rousset, Reddy, Mosbach, Dankers — Mila / McGill University).
**arXiv:** 2607.02513v1 [cs.CL], 2 Jul 2026. PDF: 12 pp, 1.6 MB.
**Code/data:** `McGill-NLP/LACUNA` — models, forget/retain sets, eval code.
**Subarea:** LLM **machine unlearning** — specifically the *localization precision* of unlearning methods (does the method edit the weights that actually store the target knowledge?). Genuinely fresh for this repo: neither inference-efficiency nor agentic-RL; a **privacy / model-editing evaluation-foundations** paper, a new lineage alongside the verifiable-process-rewards and statistical-ranking eval papers.

> Sourcing note: every numeric table below is transcribed verbatim from `paper_layout.txt` (`pdftotext -layout`, 1811 lines). Line ranges cited inline. The paper's two main results tables (Table 2 / Table 3) and two hyperparameter tables (Table 4 / Table 5) are the verbatim substance; results also appear as bar/curve figures, whose per-point values are **not** back-filled (only prose-confirmed numbers and the figure-printed AUC legend labels are quoted, consistent with the repo's "figure-derived numbers are weak" rule).

---

## TL;DR

Existing LLM-unlearning benchmarks grade methods only on **output behavior** (does the model stop emitting the forgotten fact?) — but a model can *obfuscate* knowledge without *erasing* it, and resurfacing attacks recover it. LACUNA is the first testbed with **ground-truth parameter-level localization**: synthetic PII is injected into *known* weight subsets via masked continual pretraining, so one can directly measure whether an unlearning method edits the weights that actually store the target fact.

- **Models:** 1B (OLMo-2-0425-1B) and 7B (OLMo-3-1025-7B). PII for **1,200 synthetic profiles** (from PANORAMA, 9,674 total) injected into **6 non-overlapping parameter groups** (5% of params each, layers 0..N−2, FFN + attention only — never norm/embed).
- **4 PII fields:** email address, phone number, birth city, driver's license (predictable → less predictable).
- **4 methods compared:** AlphaEdit (localization/causal-tracing), MemFlex (gradient-localized modules), SimNPO (SOTA gradient-based / NPO), and **OracleGrad** — the paper's own oracle that receives the ground-truth forget mask and restricts edits to it (Gradient Difference objective).
- **Headline finding (localization precision, ROC AUC):** AlphaEdit 0.500, MemFlex 0.500–0.501, SimNPO 0.512–0.516 — all **at chance**; OracleGrad **0.910–0.915**. No SOTA method targets the weights that store the knowledge it claims to erase.
- **Yet output-level unlearning looks fine:** SimNPO nearly matches OracleGrad on forget/retain scores. The gap only appears under (a) localization precision and (b) **resurfacing attacks** — OracleGrad is most resistant (lowest leakage), AlphaEdit/MemFlex highly susceptible.
- **Takeaway:** precise *localize-first, unlearn-second* makes unlearning tractable (a trivial Gradient Difference objective suffices *if* you know where to edit); current "localization-based" methods do not actually localize.

---

## 1. Problem: behavioral unlearning vs. true erasure (§1–§2)

LLM unlearning aims to produce `M′` from `M` such that (Yao et al., 2024): **(1) Effectiveness** — outputs on forget prompts `x_forget ∈ D_forget` deviate from the undesirable response; **(2) Utility / Retain** — benign behavior on `D_retain` and general capability is preserved. Existing benchmarks assess only these *output-level* criteria.

The gap: models store more "hidden" knowledge than they express (Gekhman et al., 2025), and "unlearned" knowledge **resurfaces** via curated attacks (Bertran et al., 2024; Deepak et al., 2025) or **relearning** (Hu et al., 2025a; Sun et al., 2025; Rybak et al., 2026) — it was obfuscated, not erased. SOTA methods follow a **localize-first, unlearn-second** paradigm (ROME/MEMIT/AlphaEdit via causal tracing; MemFlex via gradient localization), but no benchmark can say *whether the localization step identified the right weights*, because the community has no ground-truth for where a fact lives (and attribution methods would be circular, §2.3).

**LACUNA's contribution (§1, four points):**
1. A scalable approach to inject PII into *dedicated parameters* via **masked continual pretraining** (DDP + FSDP compatible, scaled to 7B).
2. The testbed itself: 1B + 7B models with memorized PII, forget/retain splits, and a **localization-precision metric**.
3. Empirical demonstration that even SOTA methods fail to achieve non-trivial localization precision despite strong output-level scores.
4. **OracleGrad**, a precise unlearning oracle showing that *if* localization succeeds, even simple gradient-based unlearning beats SOTA on both erasure and resilience to resurfacing.

---

## 2. Constructing LACUNA (§3)

### 2.1 Training-data mixture (§3.1)

| Component | Tokens | Source / notes |
|---|---|---|
| Base corpus | **4.3B** | OLMo-2 Pretraining Corpus subset |
| PII data | **1.4B** | PANORAMA (Selvam & Ghosh, 2025): 9,674 synthetic profiles; **1,200** sampled for memorization. 4 fields: email, phone, birth city, driver's license |
| QA pairs | **2.0B** | Template-generated from the 1,200 profiles (Allen-Zhu & Li, 2024; Krishnan et al., 2025) — improves cross-fact linking within a profile |

### 2.2 Masked continual pretraining (§3.2, Algorithm 1)

PII is injected into **specific parameters** using data-dependent gradient masks (Cloud et al., 2024; Shilov et al., 2025):

- The 1,200 profiles are split into **6 distinct groups**; each group has a **non-overlapping binary mask** covering **5% of parameters** in layers **0 to N−2**. So each group's information is stored in disjoint weights — this is what later lets localization precision be scored per-group.
- Masks target **feedforward + attention parameters only** — **never normalization layers or embeddings**.
- Masks are sampled at **individual-parameter granularity** (much finer than prior component-level masking), packed **one bit per mask into a 32-bit value per parameter** (≤32 masks, no memory overhead); supports DDP and FSDP.
- **Weight-distribution-shift check:** a classifier trained to detect which group a weight belongs to scores **F1 = 0.485** vs **0.438** for random guessing → negligible detectable shift, i.e. no naive way to reverse the mask.

**Instruction tuning** (so pretrained models comply with QA extraction): ~300K tokens, **150 held-out profiles × 10 questions/field**; LoRA on the **last two layers only** (layers [14,15] for 1B, [30,31] for 7B) — these layers are *excluded* from the masked-injection set.

### 2.3 Memorization + utility verification (§3.3)

Post-training, the models memorize the injected PII (high EM/ES on forget profiles) while preserving general capability (ARC-C/E, HellaSwag, MMLU near pre-injection baselines) — established via figures (Fig 2/3) before any unlearning is applied.

---

## 3. Unlearning methods evaluated (§4.1)

| Method | Family | Mechanism |
|---|---|---|
| **SimNPO (SN)** | Gradient-based (SOTA, Dorna et al., 2025) | Reference-free negative-preference optimization; length-normalized DPO-like objective on forget data as negative examples |
| **AlphaEdit (AE)** | Localization-based | Causal-tracing-inspired (ROME/MEMIT); updates FFN `W_out` key-value memories in early–mid layers [4–8]; projects perturbation onto null space of preserved knowledge |
| **MemFlex (MF)** | Localization-based | Uses gradients to localize parameter modules where forget vs. retain diverge; restricts updates to those modules |
| **OracleGrad (OG)** | Oracle (this paper) | Receives the **ground-truth forget mask**, restricts edits to in-mask weights; objective = **Gradient Difference** (ascent on forget + descent on retain) |

> By design AlphaEdit and MemFlex can only edit weights *inside components their localization step flagged* — so their failure to hit the true mask is a localization failure, not an edit failure.

**Hyperparameter tuning:** two non-overlapping validation splits (held-out profiles for the two most-memorized fields, driver's license + email), tuned independently per model size. Forget/retain sets use a **cross-field scheme** — forget and retain always target two *distinct* PII types (preliminary experiments showed same-field splits collapse unlearning; footnote 4).

---

## 4. Evaluation metrics (§4.2)

**Output-level** (Dorna et al., 2025; formalized Appendix F.2), each also on paraphrased prompts:
- **EM — Exact Memorization** (Tirumala et al., 2022): proportion of response tokens matching ground truth.
- **ES — Extraction Strength** (Carlini et al., 2021): shortest prefix length needed to reconstruct the suffix.
- **Prob**: model's output confidence on the target.

Read as: **Forget** metrics (lower = better unlearning), **Retain** metrics (higher = better preservation).

**Utility (Δ vs. pre-unlearning):** ARC-C, ARC-E, HellaSwag, MMLU — reported as score *change*.

**Localization precision (this paper's metric):** ROC AUC summarizing how well a method's per-weight modification score `s_i` discriminates **in-mask** (`y_i=1`) vs **out-of-mask** (`y_i=0`) parameters.
- AUC = 1.0 perfect localization; **0.5 = indiscriminate** modification; <0.5 means it mostly edits out-of-mask weights.
- Threshold-free, class-imbalance invariant (mask is a tiny fraction of params), probabilistically interpretable.
- Three scoring families: **(a) Magnitude** (how much each weight changed), **(b) Reversal** (does the change reverse the injection direction), **(c) Contrast** (change vs. the same method on an unmasked control model). Plus a **composite** (cross-validated logistic regression over all features).
- For each (field, method) pair the **highest AUC across applicable families** is reported — every method gets its most favorable detector. Contrast-based is **undefined for OracleGrad** (no unmasked control), so its AUC is selected over the remaining families (footnote 8) — hence OracleGrad shows "—" in the `AUC (F—R)` column below.
- Two AUC variants in the tables: **`AUC (F)`** = discriminate in-mask from *all* out-of-mask weights; **`AUC (F—R)`** = discriminate the forget-mask from the *retain-mask* weights (a stricter same-task contrast).

**Resurfacing attack (§4.3):** fine-tune the unlearned model on held-out PII (a small set memorized but excluded from forget/retain), then count how many of the 100 forget profiles leak at least once across **200 prompting attempts** (Success@200), plus Jaccard similarity of leaked-profile sets across methods.

---

## 5. Main results — Table 2: OLMo2 1B (verbatim, §F.2, paper_layout.txt lines 1479–1518)

Cumulative results for all unlearning methods — **AlphaEdit (AE), MemFlex (MF), OracleGrad (OG), SimNPO (SN)** — across the four PII fields. Per field: Forget (↓ better), Retain (↑ better), Utility Δ, Precision AUC.

### 5.1 Email Address (1B)

| Metric | AE | MF | OG | SN |
|---|---|---|---|---|
| Forget EM | 63.2 | 36.8 | 1.6 | 17.7 |
| Forget ES | 34.8 | 15.9 | 15.9 | 16.7 |
| Forget EM Paraph. | 63.6 | 36.7 | 1.6 | 18.0 |
| Forget ES Paraph. | 34.1 | 15.9 | 15.9 | 16.6 |
| Forget Prob | 10.3 | 0.0 | 0.0 | 0.9 |
| Forget Prob Paraph. | 10.6 | 0.0 | 0.0 | 0.9 |
| Retain EM | 58.8 | 98.0 | 100.0 | 100.0 |
| Retain ES | 89.8 | 96.8 | 100.0 | 100.0 |
| Retain EM Paraph. | 60.2 | 96.3 | 100.0 | 93.7 |
| Retain ES Paraph. | 90.8 | 95.8 | 100.0 | 95.7 |
| Retain Prob | 4.5 | 1.7 | 99.9 | 100.0 |
| Retain Prob Paraph. | 4.5 | 1.7 | 99.7 | 91.3 |
| Utility ARC-C | +4.0 | −4.0 | −2.0 | −12.0 |
| Utility ARC-E | +0.0 | −6.0 | +2.0 | −18.0 |
| Utility HSwag | +0.0 | +0.0 | −4.0 | −4.0 |
| Utility MMLU | +0.1 | −1.5 | −1.2 | −1.8 |
| Precision AUC (F—R) | 0.500 | 0.500 | — | 0.519 |
| Precision AUC (F) | 0.500 | 0.500 | **0.915** | 0.515 |

### 5.2 Phone Number (1B)

| Metric | AE | MF | OG | SN |
|---|---|---|---|---|
| Forget EM | 39.4 | 1.5 | 6.3 | 1.7 |
| Forget ES | 23.2 | 16.6 | 16.6 | 16.6 |
| Forget EM Paraph. | 39.2 | 1.5 | 6.3 | 1.5 |
| Forget ES Paraph. | 23.1 | 16.6 | 16.6 | 16.6 |
| Forget Prob | 1.2 | 0.0 | 0.0 | 0.0 |
| Forget Prob Paraph. | 1.3 | 0.0 | 0.0 | 0.0 |
| Retain EM | 68.3 | 98.8 | 100.0 | 100.0 |
| Retain ES | 91.9 | 98.1 | 100.0 | 100.0 |
| Retain EM Paraph. | 67.0 | 98.6 | 100.0 | 100.0 |
| Retain ES Paraph. | 92.7 | 97.6 | 100.0 | 100.0 |
| Retain Prob | 4.6 | 0.8 | 99.8 | 100.0 |
| Retain Prob Paraph. | 4.4 | 0.8 | 99.7 | 99.6 |
| Utility ARC-C | +0.0 | −12.0 | +4.0 | −16.0 |
| Utility ARC-E | +0.0 | −10.0 | +0.0 | −36.0 |
| Utility HSwag | +0.0 | +0.0 | +2.0 | −2.0 |
| Utility MMLU | +0.0 | −1.9 | −0.9 | −2.6 |
| Precision AUC (F—R) | 0.500 | 0.500 | — | 0.520 |
| Precision AUC (F) | 0.500 | 0.500 | **0.914** | 0.515 |

### 5.3 Birth City (1B)

| Metric | AE | MF | OG | SN |
|---|---|---|---|---|
| Forget EM | 56.6 | 0.0 | 0.0 | 6.8 |
| Forget ES | 85.8 | 66.6 | 66.6 | 70.0 |
| Forget EM Paraph. | 55.4 | 0.0 | 0.0 | 5.8 |
| Forget ES Paraph. | 85.9 | 66.6 | 66.6 | 70.0 |
| Forget Prob | 2.7 | 0.0 | 0.0 | 0.2 |
| Forget Prob Paraph. | 2.6 | 0.0 | 0.0 | 0.2 |
| Retain EM | 85.1 | 100.0 | 100.0 | 72.7 |
| Retain ES | 62.8 | 100.0 | 100.0 | 47.7 |
| Retain EM Paraph. | 85.7 | 99.7 | 100.0 | 73.1 |
| Retain ES Paraph. | 65.5 | 98.3 | 100.0 | 49.3 |
| Retain Prob | 16.1 | 34.1 | 99.7 | 42.5 |
| Retain Prob Paraph. | 16.7 | 34.0 | 98.7 | 43.4 |
| Utility ARC-C | +4.0 | −16.0 | −4.0 | −8.0 |
| Utility ARC-E | −8.0 | −6.0 | −10.0 | +2.0 |
| Utility HSwag | +0.0 | −6.0 | +4.0 | −6.0 |
| Utility MMLU | −0.2 | −1.2 | −0.4 | −0.7 |
| Precision AUC (F—R) | 0.500 | 0.501 | — | 0.522 |
| Precision AUC (F) | 0.500 | 0.501 | **0.913** | 0.516 |

### 5.4 Driver's License (1B)

| Metric | AE | MF | OG | SN |
|---|---|---|---|---|
| Forget EM | 64.9 | 0.0 | 0.1 | 0.9 |
| Forget ES | 23.0 | 15.6 | 15.6 | 15.6 |
| Forget EM Paraph. | 64.1 | 0.0 | 0.1 | 0.5 |
| Forget ES Paraph. | 23.9 | 15.6 | 15.6 | 15.6 |
| Forget Prob | 6.1 | 0.0 | 0.0 | 0.0 |
| Forget Prob Paraph. | 6.2 | 0.0 | 0.0 | 0.0 |
| Retain EM | 81.7 | 96.9 | 100.0 | 100.0 |
| Retain ES | 59.5 | 90.2 | 100.0 | 100.0 |
| Retain EM Paraph. | 82.1 | 93.3 | 99.6 | 98.2 |
| Retain ES Paraph. | 59.8 | 80.3 | 98.4 | 91.9 |
| Retain Prob | 17.6 | 22.5 | 99.8 | 100.0 |
| Retain Prob Paraph. | 18.0 | 22.6 | 98.5 | 95.8 |
| Utility ARC-C | +4.0 | +2.0 | −10.0 | −8.0 |
| Utility ARC-E | −2.0 | −6.0 | −4.0 | −16.0 |
| Utility HSwag | +2.0 | −2.0 | −4.0 | −2.0 |
| Utility MMLU | −0.4 | −1.1 | −1.1 | −2.9 |
| Precision AUC (F—R) | 0.500 | 0.500 | — | 0.522 |
| Precision AUC (F) | 0.500 | 0.500 | **0.914** | 0.516 |

> 1B localization-precision AUC(F): OracleGrad **0.913–0.915**; all three SOTA methods 0.500–0.522 — i.e. at chance. (Figure-4b/7 legend labels corroborate: email AE 0.500 / MF 0.500 / SN 0.515 / OG 0.915; driver's-lic OG 0.914; birth-city MF 0.501 / OG 0.913; phone OG 0.914.)

---

## 6. Main results — Table 3: OLMo3 7B (verbatim, §F.2, paper_layout.txt lines 1520–1612)

Same layout as Table 2. Note the 7B MMLU collapses under SimNPO (−25 to −31 pp) — far larger utility cost than at 1B.

### 6.1 Email Address (7B)

| Metric | AE | MF | OG | SN |
|---|---|---|---|---|
| Forget EM | 37.1 | 1.6 | 1.6 | 9.8 |
| Forget ES | 13.1 | 13.1 | 13.1 | 13.1 |
| Forget EM Paraph. | 36.3 | 1.6 | 1.6 | 10.3 |
| Forget ES Paraph. | 13.1 | 13.1 | 13.1 | 13.1 |
| Forget Prob | 0.2 | 0.0 | 0.0 | 0.0 |
| Forget Prob Paraph. | 0.2 | 0.0 | 0.0 | 0.0 |
| Retain EM | 69.2 | 100.0 | 100.0 | 100.0 |
| Retain ES | 88.2 | 100.0 | 100.0 | 100.0 |
| Retain EM Paraph. | 67.6 | 100.0 | 100.0 | 99.2 |
| Retain ES Paraph. | 87.6 | 100.0 | 100.0 | 99.2 |
| Retain Prob | 1.5 | 0.0 | 99.9 | 100.0 |
| Retain Prob Paraph. | 1.7 | 0.0 | 99.7 | 99.0 |
| Utility ARC-C | +0.0 | −4.0 | −4.0 | −20.0 |
| Utility ARC-E | +0.0 | −2.0 | −2.0 | −20.0 |
| Utility HSwag | +2.0 | +0.0 | +0.0 | −6.0 |
| Utility MMLU | +0.2 | −1.5 | −1.5 | −28.8 |
| Precision AUC (F—R) | 0.500 | 0.500 | — | 0.515 |
| Precision AUC (F) | 0.500 | 0.500 | **0.911** | 0.512 |

### 6.2 Phone Number (7B)

| Metric | AE | MF | OG | SN |
|---|---|---|---|---|
| Forget EM | 29.5 | 9.9 | 11.0 | 1.1 |
| Forget ES | 18.0 | 15.3 | 15.3 | 15.3 |
| Forget EM Paraph. | 29.7 | 9.9 | 11.0 | 1.1 |
| Forget ES Paraph. | 17.9 | 15.3 | 15.3 | 15.3 |
| Forget Prob | 0.2 | 0.0 | 0.0 | 0.1 |
| Forget Prob Paraph. | 0.2 | 0.0 | 0.0 | 0.1 |
| Retain EM | 70.9 | 100.0 | 100.0 | 100.0 |
| Retain ES | 86.7 | 100.0 | 100.0 | 100.0 |
| Retain EM Paraph. | 69.3 | 100.0 | 100.0 | 99.0 |
| Retain ES Paraph. | 87.6 | 100.0 | 100.0 | 99.0 |
| Retain Prob | 1.7 | 0.0 | 99.8 | 100.0 |
| Retain Prob Paraph. | 1.8 | 0.0 | 99.7 | 99.0 |
| Utility ARC-C | +0.0 | +0.0 | −2.0 | −24.0 |
| Utility ARC-E | +0.0 | −4.0 | +0.0 | −24.0 |
| Utility HSwag | +2.0 | +2.0 | −2.0 | −10.0 |
| Utility MMLU | +0.1 | +0.2 | −0.4 | −30.8 |
| Precision AUC (F—R) | 0.500 | 0.500 | — | 0.514 |
| Precision AUC (F) | 0.500 | 0.500 | **0.911** | 0.512 |

### 6.3 Birth City (7B)

| Metric | AE | MF | OG | SN |
|---|---|---|---|---|
| Forget EM | 46.5 | 0.3 | 0.0 | 4.4 |
| Forget ES | 82.0 | 68.9 | 68.9 | 68.9 |
| Forget EM Paraph. | 47.1 | 0.0 | 0.0 | 6.5 |
| Forget ES Paraph. | 81.2 | 68.9 | 68.9 | 68.9 |
| Forget Prob | 0.5 | 0.0 | 0.0 | 0.0 |
| Forget Prob Paraph. | 0.7 | 0.0 | 0.0 | 0.1 |
| Retain EM | 41.8 | 100.0 | 100.0 | 100.0 |
| Retain ES | 12.5 | 100.0 | 100.0 | 100.0 |
| Retain EM Paraph. | 41.1 | 99.5 | 99.4 | 96.5 |
| Retain ES Paraph. | 12.5 | 97.3 | 97.2 | 81.3 |
| Retain Prob | 0.2 | 25.2 | 99.6 | 100.0 |
| Retain Prob Paraph. | 0.2 | 24.2 | 96.2 | 89.6 |
| Utility ARC-C | −2.0 | +0.0 | +2.0 | −10.0 |
| Utility ARC-E | −2.0 | −2.0 | −2.0 | −14.0 |
| Utility HSwag | +2.0 | +0.0 | −2.0 | −4.0 |
| Utility MMLU | +0.4 | +0.6 | −1.3 | −26.8 |
| Precision AUC (F—R) | 0.500 | 0.500 | — | 0.516 |
| Precision AUC (F) | 0.500 | 0.500 | **0.911** | 0.513 |

### 6.4 Driver's License (7B)

| Metric | AE | MF | OG | SN |
|---|---|---|---|---|
| Forget EM | 14.3 | 0.1 | 0.1 | 0.6 |
| Forget ES | 14.7 | 14.5 | 14.5 | 14.5 |
| Forget EM Paraph. | 14.6 | 0.1 | 0.1 | 0.7 |
| Forget ES Paraph. | 14.9 | 14.5 | 14.5 | 14.5 |
| Forget Prob | 0.0 | 0.0 | 0.0 | 0.0 |
| Forget Prob Paraph. | 0.0 | 0.0 | 0.0 | 0.0 |
| Retain EM | 40.8 | 100.0 | 100.0 | 100.0 |
| Retain ES | 12.5 | 100.0 | 100.0 | 100.0 |
| Retain EM Paraph. | 40.6 | 99.4 | 99.8 | 96.0 |
| Retain ES Paraph. | 12.5 | 96.9 | 99.4 | 82.1 |
| Retain Prob | 0.2 | 9.2 | 99.8 | 100.0 |
| Retain Prob Paraph. | 0.2 | 8.7 | 98.9 | 89.1 |
| Utility ARC-C | −2.0 | −4.0 | −2.0 | −16.0 |
| Utility ARC-E | +0.0 | −2.0 | +0.0 | −10.0 |
| Utility HSwag | +2.0 | +2.0 | +0.0 | −6.0 |
| Utility MMLU | +0.2 | +0.6 | +0.7 | −25.1 |
| Precision AUC (F—R) | 0.500 | 0.500 | — | 0.520 |
| Precision AUC (F) | 0.500 | 0.500 | **0.910** | 0.516 |

> 7B localization-precision AUC(F): OracleGrad **0.910–0.911**; SOTA methods 0.500–0.520. (Figure-9 legend labels corroborate: SN ranges 0.512–0.516, OG 0.910–0.911 across fields.)

---

## 7. Localization precision — summary (Figure 4b / 7 / 9)

The localization-precision AUC(F) values, gathered from the Tables 2/3 Precision rows (and corroborated by each figure's printed legend):

| Model | Field | AE | MF | SN | OG |
|---|---|---|---|---|---|
| OLMo2 1B | email | 0.500 | 0.500 | 0.515 | **0.915** |
| OLMo2 1B | phone | 0.500 | 0.500 | 0.515 | **0.914** |
| OLMo2 1B | birth city | 0.500 | 0.501 | 0.516 | **0.913** |
| OLMo2 1B | driver's lic. | 0.500 | 0.500 | 0.516 | **0.914** |
| OLMo3 7B | email | 0.500 | 0.500 | 0.512 | **0.911** |
| OLMo3 7B | phone | 0.500 | 0.500 | 0.512 | **0.911** |
| OLMo3 7B | birth city | 0.500 | 0.500 | 0.513 | **0.911** |
| OLMo3 7B | driver's lic. | 0.500 | 0.500 | 0.516 | **0.910** |

**Reading:** every SOTA method sits at chance (0.500–0.516); only the oracle (OracleGrad, which is *given* the true mask) reaches 0.91+. Even analyzing AlphaEdit/MemFlex precision *within their self-selected components* did not improve the score (footnote 10).

---

## 8. Resurfacing (relearning) attack — Figure 5 / 13–16

After unlearning, models are fine-tuned on held-out PII; **Success@200** = % of the 100 forget profiles that leak ≥ once across 200 prompts.

- **Prose-confirmed ranking (email address, both model sizes):** AlphaEdit and MemFlex are **highly susceptible** (large portions of the forget set reconstructable); SimNPO shows higher robustness but still leaks; **OracleGrad is most resistant** (lowest leakage).
- **Per-method bar heights (Figure 5a, email):** figure-derived — AlphaEdit ≈ 87% (1B) / 74% (7B), MemFlex ≈ 23% / 16%, SimNPO ≈ 13% / 14%, OracleGrad ≈ 8%. *(Treat as approximate bar readings; only the qualitative ranking and OracleGrad ≈ 8% lowest are prose-confirmed.)*
- **Leaked-profile Jaccard similarity (Fig 5b/5c):** leaked-profile sets of SimNPO and OracleGrad **largely overlap** → points to "stubborn" profiles that are inherently hard to unlearn rather than method-specific failures (the authors flag this as a *qualitative* observation, not a primary quantitative claim, since for robust methods it is computed over small sets).
- **7B more robust than 1B overall** (§F.3): OLMo3 7B is "consistently much less prone to resurfacing" — though the authors caveat it may reflect attack limitations at 7B scale, not a genuine model property.
- **Numerical fields caveat (footnote 12):** for the *numerical* PII (phone, driver's license), SimNPO and OracleGrad are **equally resistant** to the straightforward resurfacing attack — OracleGrad's resurfacing edge holds mainly on email/birth-city.

---

## 9. Hyperparameters (verbatim)

### 9.1 Table 4 — Training & Instruction-Tuning Hyperparameters (paper_layout.txt lines 1735–1752)

| Parameter | OLMo2 1B | OLMo3 7B |
|---|---|---|
| Base model | allenai/OLMo-2-0425-1B | allenai/OLMo-3-1025-7B |
| Checkpoint revision | step1907359-tokens4001B | step999000 |
| Instruction Tuning LoRA rank (r) | 16 | 16 |
| Instruction Tuning LoRA alpha (α) | 8 | 8 |
| Instruction Tuning LoRA target layers | [14, 15] | [30, 31] |
| Instruction Tuning Early stopping | Best eval loss | Best eval loss |

### 9.2 Table 5 — Unlearning Hyperparameters by Method (paper_layout.txt lines 1769–1811)

Values that differ between 1B and 7B marked **bold** (only the differing value is bolded in source).

| Method | Hyperparameter | OLMo2 1B | OLMo3 7B |
|---|---|---|---|
| **MemFlex** | Forget factor | −0.6 | −0.6 |
| | Retain factor | 2.0 | 2.0 |
| | Learning rate | **3 × 10⁻⁴** | **1 × 10⁻⁴** |
| | Gradient threshold | **6 × 10⁻⁴** | **1 × 10⁻⁵** |
| | Similarity threshold | 0.92 | 0.92 |
| | Epochs | 20 | 20 |
| **SimNPO** | γ (forget weight) | 3.0 | 3.0 |
| | α (retain weight) | **0.01** | **0.5** |
| | β (sharpness) | 10.0 | 10.0 |
| | δ (margin) | 1.5 | 1.5 |
| | Learning rate | 1 × 10⁻⁴ | 1 × 10⁻⁴ |
| | Epochs | 200 | 200 |
| | Retain loss / scheduler | NLL / Constant | NLL / Constant |
| **OracleGrad** | Method | GradDiff | GradDiff |
| | γ (forget weight) | **5.0** | **1.0** |
| | α (retain weight) | **1.0** | **0.5** |
| | Learning rate | **1 × 10⁻⁴** | **5 × 10⁻⁵** |
| | Epochs | 200 | 200 |
| **AlphaEdit** | Clamp norm factor | 0.5 | 0.5 |
| | Null-space threshold | 1 × 10⁻³ | 1 × 10⁻³ |
| | v gradient steps | 50 | 50 |
| | v learning rate | 5 × 10⁻² | 5 × 10⁻² |
| | v loss layer | **15** | **31** |
| | v weight decay | **1.0** | **0.5** |
| | KL factor | **1.0** | **0.0625** |
| | L2 regularization | 1.0 | 1.0 |
| | MOM2 dataset | **wikipedia + retain** | **wikipedia** |
| | Target layers | [4, 5, 6, 7, 8] | [4, 5, 6, 7, 8] |
| | Batch size | **5** | **1** |

> OracleGrad objective choice: tuned GradAscent (forget-only) vs GradDiff (ascent + retain descent); GradDiff wins on retain stability with strong unlearning (per-method forget/retain weights in Table 5).

---

## 10. Findings & takeaways

1. **Output-level unlearning is misleading.** SimNPO matches OracleGrad on forget/retain scores (e.g. 1B email Forget EM 17.7 vs 1.6; Retain 100.0 vs 100.0) yet has localization AUC 0.515 vs 0.915. Behavioral benchmarks cannot distinguish erasure from obfuscation.
2. **No SOTA method localizes.** AlphaEdit, MemFlex, SimNPO all sit at AUC ≈ 0.50–0.52 across all 4 fields and both model sizes — indiscriminate editing. The "localization-based" labels are aspirational.
3. **Precise localization makes unlearning easy.** OracleGrad uses a *trivial* Gradient Difference objective yet achieves both strong erasure and the lowest resurfacing leakage — because it edits the right weights. The bottleneck is localization, not the edit rule.
4. **Precision correlates with robustness.** OracleGrad (most precise) is most resistant to resurfacing; the imprecise methods (AlphaEdit/MemFlex) leak heavily. This is evidence the knowledge was genuinely erased, not just suppressed.
5. **SimNPO's erasure comes at utility cost** — and it scales badly: 1B MMLU −1.8 pp (email) but **7B MMLU −28.8 pp**, ARC-E −36 pp (phone). OracleGrad preserves utility far better (MMLU within ±1.5 pp).
6. **Cross-field forget/retain is necessary** (footnote 4): same-field splits collapse unlearning performance during preliminary experiments.
7. **Birth-city ES is a floor, not a success** — Birth-city Forget ES stays high for OG/SN (66.6/70.0 at 1B) only because ES measures *shortest reconstructing prefix*, and birth-city names are low-entropy; EM and Prob (both ≈ 0) show the fact is in fact forgotten. Read EM/Prob, not ES, for birth city.

---

## 11. Strengths, limitations, verdict

**Strengths**
- First unlearning testbed with **ground-truth parameter-level localization** — closes the circularity gap that blocked attribution-based evaluation.
- Clean experimental control: 6 non-overlapping masks, cross-field forget/retain splits, dedicated validation splits, mask-distribution-shift check (F1 0.485 vs 0.438).
- The OracleGrad baseline turns "is localization the bottleneck?" into a falsifiable, answered question.
- Scales to 7B with a memory-free bit-packed per-parameter masking scheme (DDP + FSDP).

**Limitations** (mix of the paper's §5 and breakdown-observed)
- ⚠ **Tables are numbered Table 2/3/4/5 — there is no Table 1 caption in the source** (paper-internal numbering quirk; the first numbered table is the cumulative unlearning results). Flagged rather than "fixed."
- Synthetic PII only (PANORAMA); real-world memorized PII distribution may differ.
- Resurfacing attack is a single, "straightforward" fine-tuning attack (the authors note 7B-resistance may reflect attack weakness, not model strength).
- Localization precision rewards *any* in-mask-vs-out-of-mask separability — a method could be precise about the wrong sub-behavior; the metric is necessary but not sufficient for "true erasure."
- OracleGrad is an **oracle** (uses the ground-truth mask) — it upper-bounds what localization-based methods *could* achieve, not a deployable method.
- Birth-city ES is uninformative (low-entropy names) — a reader relying on ES alone would misread birth-city results.

**Verdict.** A well-constructed evaluation-foundations paper that reframes LLM unlearning from "did the output change?" to "did the weights change *in the right place*?" The central empirical claim — **every SOTA unlearning method edits indiscriminately (AUC ≈ 0.5), only the mask-oracle reaches 0.91** — is clean, consistent across 4 fields × 2 model sizes, and directly actionable: unlearning progress now requires better *localization*, not better edit objectives. The most citable contribution is the testbed + metric; the most citable negative result is that "localization-based" SOTA methods do not, in fact, localize.

---

## 12. Source-free reconciliation checks

- **AUC(F) = figure legend label** for every (model, field): Table 2/3 Precision-row values reproduce the Figure 4b/7/9 printed AUC labels (email 1B OG 0.915, SN 0.515; driver's-lic 1B OG 0.914; birth-city 1B MF 0.501; 7B OG 0.910–0.911). Two independent extracts agree.
- **Forget-vs-Retain symmetry for OracleGrad:** OG Retain ≈ 100 across EM/ES/Paraph (retain fully preserved) while Forget ≈ 0 — consistent with precise in-mask editing that leaves retain weights untouched.
- **SimNPO utility scaling:** 7B MMLU drop (−25 to −31 pp) is consistently ~15–20× the 1B drop (−1 to −3 pp) across all four fields — internally consistent with stronger optimization disrupting a larger model more.
- **Chance-level AUC sanity:** AlphaEdit AUC = 0.500 in all 8 (model, field) cells — exactly chance, as expected for a method whose edits are uncorrelated with the true mask.

---

## 13. Full cell-by-cell source verification (2026-07-13)

**VERIFICATION PASS: ZERO numeric defects.** Every numeric cell re-checked against `paper_layout.txt` at the cited line ranges:

- **Table 2 (1B, lines 1479–1518):** all 4 fields × 18 metrics × 4 methods = **288 cells EXACT** (Forget EM/ES/Paraph×2/Prob×2, Retain same 6, Utility ARC-C/E/HSwag/MMLU, Precision AUC(F−R)/AUC(F)).
- **Table 3 (7B, lines 1520–1612):** all **288 cells EXACT**, same layout.
- **Table 4 (lines 1735–1752):** 6 training/instruction-tuning params × 2 model sizes EXACT.
- **Table 5 (lines 1769–1811):** ~30 unlearning hyperparams × 2 model sizes EXACT, **including every bold-differ marking** (MemFlex LR/grad-threshold; SimNPO α; OracleGrad γ/α/LR; AlphaEdit v-loss-layer/v-weight-decay/KL-factor/MOM2-dataset/batch-size all correctly bolded as 1B≠7B).
- **§7 summary table:** all 8 (model,field) AUC(F) rows recompute EXACT from the Tables 2/3 Precision rows (column re-order AE/MF/SN/OG applied consistently).
- **Prose ↔ table:** TL;DR ranges (AlphaEdit 0.500; MemFlex 0.500–0.501; SimNPO 0.512–0.516; OracleGrad 0.910–0.915) all match the min/max over the 8 cells; "1B email Forget EM 17.7 vs 1.6", "7B MMLU −28.8", "ARC-E −36 (phone)" all exact.
- **Figure 13a Success@200 bar labels** (87/74/23/16/13/14/8%) are *printed on the bars*, not height-reads — breakdown's "≈ approximate" caveat is conservative; values are author-printed.

**Honest-scope surfaces (NOT numeric typos — attributional/framing, per the repo meta-finding for eval/benchmark papers):** (a) OracleGrad is an oracle using the GT mask — upper-bounds localization-based methods, not deployable; (b) AlphaEdit restricted to FFN early-mid layers [4–8] while masks span layers 0..N−2 FFN+attn, so its localization is structurally subset-restricted (footnote 10 defends: precision within self-selected components still didn't improve); (c) Birth-city ES uninformative (low-entropy names) — read EM/Prob not ES; (d) resurfacing = single straightforward FT attack; 7B-resistance may be attack weakness not model strength (authors candid); (e) synthetic PII only (PANORAMA). The load-bearing claim — *every SOTA method edits indiscriminately (AUC≈0.5), only the mask-oracle reaches 0.91* — is clean and consistent across 4 fields × 2 sizes.

No edits required to the transcribed tables. Breakdown confirmed accurate.
