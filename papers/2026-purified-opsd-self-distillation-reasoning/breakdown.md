# Purified OPSD: On-Policy Self-Distillation Without Losing How to Think

- **arXiv:** 2607.02234v1 [cs.AI], 2 Jul 2026
- **Authors:** Zhanming Shen, Jintao Tong, Shaotian Yan, Chen Shen‡, Hao Chen, Wentao Ye, Xiaomeng Hu, Rui Miao, Haobo Wang†, Junbo Zhao, Gang Chen, Jieping Ye — Zhejiang University + Tongyi Lab (Alibaba) + HUST + Jilin U
- **Subarea (repo lineage):** long-CoT reasoning **self-distillation** — first paper in the repo on *on-policy self-distillation (OPSD)* failure-mode diagnosis. Sibling-in-spirit to the distillation lineage (danceopd on-policy field distillation, opid on-policy skill distillation) but uniquely targets **long-CoT reasoning models** and diagnoses *why* privileged-teacher OPSD degrades them. Mechanistic cousin to subliminal-clocks (both measure how a supervision signal reshapes reflective/epistemic behavior) and to vprm (both isolate the *transferable* slice of a dense supervision signal).
- **Source:** `paper.pdf` (15pp, 883KB), `paper_layout.txt` (pdftotext -layout, 1071 lines). Only **1 explicit table** (Table 1) + 7 figures (training-curve / marker-distribution / decomposition plots). Built source-first; figure curve points are NOT back-filled (only prose-confirmed marker numbers are quoted).

---

## TL;DR

On-policy self-distillation (OPSD) — a privileged teacher with the reference solution supervises the student's own trajectories — **consistently fails on long-CoT reasoning models** (marginal/short-lived gains, often degrades). Through a clean decomposition of the teacher's update, the authors show the failure is mechanistic: the supervision is **dominated by a reference-induced component** (Δ_ref, rote-memorizing reference-specific shortcuts the student can never use at inference) while the **question-conditioned, inference-transferable component** (Δ_it) is near-orthogonal or actively **anti-aligned** with the update. Fix: construct a **reference-only teacher** π_ref (same model, reference but *no* question) to isolate Δ_ref, take the residual Δ_it = log π_T − log π_ref, and turn it into a well-formed **PMI target distribution** P_PMI ∝ P_0 · exp(Δ_it / β). This target is provably the **closed-form optimum of a KL-regularized distillation objective** (Eq 7–8) with Δ_it as the implicit reward and the clean base P_0 as the reference policy. Across 4 long-CoT models × 2 datasets, OPSD-PMI beats both base and standard OPSD on **8/8** model×dataset Avg cells, while preserving epistemic-marker behavior.

---

## The Diagnosis (§2) — why OPSD breaks long-CoT models

### Decomposition of the teacher's update (Eq 4)

With student π_θ, privileged teacher π_T (= base model conditioned on question **q + reference r**), and a **reference-only teacher** π_ref (= same model conditioned on **r only, no q**):

```
log π_T − log π_θ  =  (log π_ref − log π_θ)  +  (log π_T − log π_ref)
       Δ_total              Δ_ref                    Δ_it
   (total update)     (reference-induced,      (inference-transferable,
                         NON-transferable)        the wanted signal)
```

- **Δ_ref** = supervision that exists even *without* the question — pure reference-memorization signal the student will never have at inference.
- **Δ_it** = how much the teacher's prediction changes when the question is added on top of the reference — the genuinely transferable correction.
- If OPSD worked, Δ_total should align with Δ_it. If it's memorizing, Δ_total aligns with Δ_ref.

### What the decomposition shows (Figure 3; prose-confirmed)

Two complementary metrics, computed over the full vocabulary at each token, averaged over 100 samples/checkpoint:

1. **Direction — cosine similarity.** cos(Δ_total, Δ_ref) stays **high throughout training** on both Qwen3-8B and R1-Distill-7B. cos(Δ_total, Δ_it) on Qwen3-8B rises from **−0.95 → ~0**; on R1, cos(Δ_total, Δ_ref) climbs **0.58 → 0.99** in the first 100 steps then plateaus. → The teacher persistently pulls the student toward the reference, not toward question-conditioned reasoning.
2. **Magnitude — norm fraction.** ‖Δ_ref‖ / ‖Δ_total‖ **consistently exceeds 1.0** — the reference-induced component is *larger than the total update itself*, i.e. Δ_it partially **cancels** Δ_ref rather than reinforcing it.

> ⚠ **Paper-internal note (transcribed verbatim, not "reconciled").** The ‖Δ_ref‖/‖Δ_total‖ > 1.0 claim plus cos(Δ_total, Δ_it) ≈ −0.95 early together imply Δ_it is largely **anti-aligned** with the total update at the start of training (the wanted signal is being *fought*). This is the cleanest single statement of the paper's mechanism. The norm-fraction and cosine values are figure-curve readings (Figure 3 axis ticks); only the qualitative endpoints (−0.95, 0.58→0.99) are prose-confirmed (paper_layout.txt lines 382–388).

### Epistemic-marker destabilization (Figure 2 / Figure 5; prose-confirmed)

Following Kim et al. 2026a, the authors track epistemic markers ("Wait", "Maybe", "Perhaps", "Check"). Under standard OPSD (Math-CoT-20K):

| Model | Pathological pattern (prose-confirmed) |
|---|---|
| Qwen3-8B | total epistemic count **collapses 73K → 40K**; "wait" 27K→10K, "maybe" 12K→3K (uniform suppression) |
| R1-Distill-7B | total count **explodes 71K → 115K**, but increase concentrated on a single token **"Wait" 34K → 83K** (degenerate repetition, not deliberation) |

Under OPSD-PMI both stay **within the ~70K range** (near-baseline) — the central preservation claim.

---

## Method (§3)

### The PMI target (Eq 5–6)

The residual Δ_it is a *per-token log-probability difference*, not a distribution. PMI converts it into one, anchored on the clean base distribution P_0(v) = π_0(v | ŷ_<t, q) (question, **no reference**):

```
P_PMI(v) ∝ P_0(v) · exp( Δ_it(v) / β )              (Eq 5)
log P_PMI(v) = log P_0(v) + Δ_it(v)/β − log Z        (Eq 6)
```

β > 0 is the correction strength (β=1 = full correction; larger β → more conservative, closer to base).

### Why this is optimal, not heuristic (Eq 7–8) — the elegant result

P_PMI is the **closed-form maximizer** of a token-level KL-regularized distillation objective — the exact analogue of the RLHF/DPO policy-improvement step (Rafailov et al. 2024) with P_0 as the reference policy and Δ_it as the implicit reward r(v):

```
P★ = argmax_{P ∈ Δ(V)}  [ E_{v∼P}[ r(v) ]  −  β · D_KL(P ‖ P_0) ]     (Eq 7)
P★(v) = (1/Z) · P_0(v) · exp( r(v) / β )                               (Eq 8)
```

Substituting r(v) = Δ_it(v) recovers Eq 5 exactly. So the PMI target is **not** an ad-hoc renormalization of the residual — it is the optimal KL-regularized target induced by the transferable reward Δ_it. (Derived via a Lagrange multiplier; Eq 9–11.)

### Stabilized implementation (§3.2) — 6 steps

Three **frozen-base forward passes** (same weights, different prompts) + the student forward:

| Pass | Logits | Input |
|---|---|---|
| Teacher ℓ_T | logit_π_0(v \| ŷ_<t, **q, r**) | question + reference |
| Reference probe ℓ_ref | logit_π_0(v \| ŷ_<t, **r**) | reference only |
| Base ℓ_0 | logit_π_0(v \| ŷ_<t, **q**) | question only |
| Student ℓ_θ | logit_π_θ(v \| ŷ_<t, q) | question (gradients tracked) |

1. On-policy generate ŷ ∼ π_θ(·|q); compute the 3 frozen logits + student logits.
2. Raw PMI signal: Δ_it(v) = log π_T(v) − log π_ref(v), with π_T = softmax(ℓ_T), π_ref = softmax(ℓ_ref)  **(Eq 13)**
3. **Centering**: subtract vocabulary-level mean → zero-centered correction, numerically stable, preserves relative preference  **(Eq 14)**
4. **Soft clipping**: Δ̃_it(v) = c · tanh( Δ̄_it(v) / c ), with **c = 10**; identity for small values, saturates at ±c  **(Eq 15)**
5. Construct stabilized target: P_target = softmax( log π_0(·|ŷ_<t,q) + Δ̃_it / β ), **β = 1**  **(Eq 16–17)**
6. Train by minimizing generalized **JSD** between student and P_target  **(Eq 18)**

**Overhead:** +2 frozen forward passes/step (no backprop for those), no new trainable params, **<10% wall-clock** (paper_layout.txt line 561).

### Training loss contrast

```
OPSD-Standard:  L = D_JSD( π_θ ‖ π_T )           (Eq 1/3)  — inherits reference shortcut
OPSD-PMI (Ours): L = D_JSD( π_θ ‖ P_target )      (Eq 12)   — distills only Δ_it, anchored to P_0
```

---

## Setup (§4.1)

- **Models (4 long-CoT):** Qwen3-8B, Qwen3-4B, DeepSeek-R1-Distill-Qwen-7B (R1-Distill-7B), OLMo-7B-Thinking
- **Training data (2 datasets, reference solution = privileged info):** DASD-10K (10K subset of DASD), Math-CoT-20K (20K competition-level math w/ CoT)
- **Baselines (3 configs):** Base / OPSD-Standard (JSD loss, raw teacher) / OPSD-PMI (Ours)
- **Eval:** AIME 2024, AIME 2025, HMMT 2025; checkpoints every 50 steps to 200; **12-run averages**; report best accuracy
- **Implementation:** LoRA rank 64, lr 5e-6, batch 32, gradient checkpointing, vLLM for on-policy generation, β=1, tanh soft-clip c=10, centering, max completion 1024 tokens

---

## Main Results (Table 1 — verbatim, paper_layout.txt lines 599–616)

Bold = best per model-dataset block. All 20 **Avg** cells reproduce from their 3 benchmark cells (source-free reconciliation, 0 mismatches).

| Model | Method | DASD-10K AIME24 | AIME25 | HMMT25 | **Avg** | Math-CoT-20K AIME24 | AIME25 | HMMT25 | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-8B** | Base | 75.8 | 65.6 | 43.9 | 61.8 | 75.8 | 65.6 | 43.9 | 61.8 |
| | OPSD-Standard | 75.4 | 65.2 | 42.2 | 60.9 | 75.8 | 66.7 | 44.4 | 62.3 |
| | **OPSD-PMI (Ours)** | **79.4** | **71.9** | **46.7** | **66.0** | **77.1** | **70.8** | **47.5** | **65.1** |
| **Qwen3-4B** | Base | 74.9 | 66.4 | 42.2 | 61.2 | 74.9 | 66.4 | 42.2 | 61.2 |
| | OPSD-Standard | 74.2 | 65.2 | 42.2 | 60.5 | 73.3 | 64.2 | 40.8 | 59.4 |
| | **OPSD-PMI (Ours)** | **76.3** | **68.3** | **46.4** | **63.7** | **76.1** | **67.5** | **44.4** | **62.7** |
| **R1-Distill-7B** | Base | 52.0 | 39.6 | 24.4 | 38.7 | 52.0 | 39.6 | 24.4 | 38.7 |
| | OPSD-Standard | 51.9 | 39.2 | 24.4 | 38.5 | 52.2 | 38.1 | 24.4 | 38.2 |
| | **OPSD-PMI (Ours)** | **54.0** | **43.1** | **25.3** | **40.8** | **55.3** | **41.1** | **26.1** | **40.8** |
| **OLMo-7B** | Base | 71.9 | 66.7 | 45.2 | 61.3 | 71.9 | 66.7 | 45.2 | 61.3 |
| | OPSD-Standard | 68.1 | 66.4 | 45.0 | 59.8 | 69.7 | 65.6 | 41.7 | 59.0 |
| | **OPSD-PMI (Ours)** | **74.7** | **68.9** | **46.1** | **63.2** | **73.3** | **70.3** | **46.4** | **63.3** |

### Three findings (§4.2) — all verified against the table

1. **OPSD-Standard fails on long-CoT.** On DASD-10K it degrades **3 of 4** models on Avg (Qwen3-8B −0.9, Qwen3-4B −0.7, OLMo −1.5), with only R1-Distill-7B near-zero (−0.2). On Math-CoT-20K it gives a marginal gain **only** on Qwen3-8B (+0.5) and degrades the other three, **OLMo-7B suffering the largest drop (−2.3)**.
2. **OPSD-PMI improves on every model-dataset combination** — beats Base on **8/8** Avg cells (gains +1.5 to +4.2 pp).
3. **The PMI−Standard gap is consistent** — OPSD-PMI beats OPSD-Standard on **8/8** Avg cells (gap +2.3 to +5.1 pp), regardless of architecture or training data → reference-induced noise is a universal OPSD bottleneck for long-CoT.

### OPSD-PMI − Base gains (Avg, pp) — the honest magnitude

| Model | DASD-10K | Math-CoT-20K |
|---|---|---|
| Qwen3-8B | +4.2 | +3.3 |
| Qwen3-4B | +2.5 | +1.5 |
| R1-Distill-7B | +2.1 | +2.1 |
| OLMo-7B | +1.9 | +2.0 |

> ⚠ **Honest-scope flag.** Gains are real and consistent but **modest in absolute terms (+1.5 to +4.2 pp)**, and the smallest gains land on the two models that need help least/most in different ways (R1-Distill-7B is already the weakest baseline at 38.7 Avg; OLMo gains least). The contribution is the *mechanism + stability*, not large headline jumps.

### Training dynamics (Figure 4; prose-confirmed, curves not back-filled)

OPSD-PMI improves and **remains stable** across checkpoints (small variance → no careful early stopping needed); OPSD-Standard peaks briefly then **steadily declines**. This directly contrasts the Figure-1 OPSD collapse. (Figure 4 is AIME-2025-vs-step curves; per-step values are axis-tick readings, not extracted.)

---

## Ablations (§4.4; Figures 6–7; prose-confirmed)

- **Soft-clip threshold c ∈ {5, 10, 20}** (β=1 fixed): all three beat baseline and OPSD-Standard; c=20 ≈ c=10 (most PMI values already in range); c=5 slightly more volatile (more aggressive tanh compression) but does not much lower the ceiling → "robust to c; tanh is a safety net, not performance-critical."
- **Correction strength β ∈ {0.5, 1, 2}** (c=10 fixed): all beat baseline and OPSD-Standard; β=0.5 volatile, β=2 smoother with occasionally higher peaks, β=1 balances → no single β dominates; **β=1 used in all main experiments**.

> Both ablations are training-curve figures (axis-tick readings only); the qualitative robustness claims are prose-confirmed, per-step accuracies are not back-filled (consistent with the universal figure-derived-numbers-are-weak rule).

---

## Strengths

- **Clean mechanistic diagnosis.** The Δ_total = Δ_ref + Δ_it decomposition + the cosine/norm metrics turn "OPSD sometimes hurts" into a *falsifiable* claim: the reference-induced component dominates direction *and* magnitude (‖Δ_ref‖/‖Δ_total‖ > 1.0), while the transferable signal is anti-aligned (cos → −0.95). This is the most citable single result.
- **Principled fix, not a heuristic.** The PMI target is the *closed-form KL-regularized optimum* (Eq 7–8) — connecting OPSD purification directly to the RLHF/DPO policy-improvement lineage with Δ_it as implicit reward and P_0 as reference policy.
- **Faithful to long-CoT epistemics.** Preserves epistemic-marker counts/distributions at near-base levels (Figure 5) — directly addresses the destabilization that makes standard OPSD unsafe for thinking models.
- **Cheap.** +2 frozen forwards, <10% wall-clock, no new params, drops into standard OPSD.
- **Honest scope.** Explicitly flags that standard OPSD *degrades* 3/4 models (DASD) and that gains are consistent but modest — does not over-claim.

## Limitations

- **Math-only evaluation.** All three benchmarks (AIME 2024/2025, HMMT 2025) are competition math. No code, science, or multi-domain long-CoT eval → generalization beyond math is untested.
- **No off-policy / SFT-distillation baselines.** Only Base / OPSD-Standard / OPSD-PMI are compared; the paper does not benchmark against off-policy long-CoT SFT (s1, LIMO, Light-R1, etc.) cited in Related Work, so the practical question "does OPSD-PMI beat a well-tuned off-policy distillation baseline?" is open.
- **Modest absolute gains (+1.5 to +4.2 pp).** The contribution is mechanism + stability, not large topline jumps.
- **Decomposition metrics are figure-derived.** The −0.95 / 0.58→0.99 cosine endpoints and the >1.0 norm-fraction claim come from Figure 3 curves (only the qualitative endpoints are prose-confirmed); per-checkpoint values are not in a table.
- **Single training-step budget (≤200).** No test of whether gains hold or widen at longer training.

## Verdict

A focused, mechanism-first paper. Its real contribution is the **decomposition diagnosis** (Δ_ref dominates, Δ_it is anti-aligned) plus the recognition that the **residual Δ_it is a PMI-style quantity** that the KL-regularized optimum *automatically* delivers as a distillation target. The PMI-as-closed-form-optimum result (Eq 7–8) is the elegant hinge that elevates the fix above heuristic normalization. Practical gains are consistent but modest and math-only; the paper's value is primarily as a **diagnostic + principled correction** for anyone applying OPSD to thinking models, not as a new SOTA on math reasoning.

---

## Sourcing note

All numeric table cells are transcribed **verbatim** from Table 1 (paper_layout.txt lines 599–616); the 20 Avg cells were source-free reconciled (each = mean of its AIME24/AIME25/HMMT25 cells, 0 mismatches), and every distinctive cell grep-confirmed present in paper_layout.txt. All §4.2 prose claims (PMI > Base 8/8, PMI > Std 8/8, DASD-degrades-3-of-4, Math-OLMo-largest-drop) recompute exactly from the table. **No paper-internal numeric prose-vs-table contradiction** was found (unlike the iter-30/31/34/46 class) — the paper is internally consistent. Figure-derived numbers (Figures 1–7 training curves, marker-count bars, decomposition cos/norm curves) are **not** back-filled; only the prose-confirmed marker counts (73K→40K, 71K→115K, "Wait" 34K→83K, "wait" 27K→10K, "maybe" 12K→3K) and the prose-confirmed decomposition endpoints (cos(Δ_total,Δ_it) −0.95→~0; cos(Δ_total,Δ_ref) 0.58→0.99; ‖Δ_ref‖/‖Δ_total‖ > 1.0) are quoted, each with the figure it came from. Inline ⚠ notes flag (a) the figure-curve origin of the decomposition metrics, and (b) the honest-scope modest-magnitude caveat — neither is a numeric defect.
