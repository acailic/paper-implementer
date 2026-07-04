# DemoPSD: Disagreement-Modulated Policy Self-Distillation — Breakdown

**Paper:** "DemoPSD: Disagreement-Modulated Policy Self-Distillation"
**Authors:** Yunhe Li\*, Hao Shi\*, Wenhao Liu, Mengzhe Ruan, Hanxu Hou, Zhongxiang Dai, Shuang Qiu†, Linqi Song†
**Affiliations:** City University of Hong Kong · Tsinghua University · Shenzhen University of Advanced Technology · Chinese University of Hong Kong, Shenzhen
**arXiv:** 2607.02502v1 [cs.LG], 2 Jul 2026 · **Length:** 9pp main (no external appendix in this PDF)
**Source-first build:** every numeric cell below is transcribed verbatim from `paper_layout.txt` (pdftotext -layout, 1078 lines). Sourcing line-ranges cited per table. Source-free reconciliation run; results reported inline.

---

## 1. One-line thesis

On-policy self-distillation (OPSD) with a privileged teacher (conditioned on the verified answer / reasoning trace `y*`) suffers **privileged-information leakage**: the student can never perfectly match a teacher that sees information it never will, and as training progresses the student starts encoding `x → y*` answer-shortcut correlations that are unavailable at test time — empirically SDPO peaks early then degrades. **DemoPSD** fixes this by *selective adoption of teacher guidance*: at each token, measure teacher–student disagreement, and steer the student toward a **reverse-KL barycenter target** that blends the teacher and student distributions — adopting the teacher where they agree, falling back to the student's own reasoning where they diverge (the divergence being the signature of privileged-information distortion).

This is a **third distinct OPSD-leakage fix** in the repo's distillation lineage, sibling-in-mechanism to `purified-opsd` (iter 52: fixes the teacher **UPDATE** via PMI purification of the `Δ_it` residual) and `neuron-aware-data-selection / N-OPSD` (iter 54: fixes the teacher **INPUT** via neuron-aware data+context selection). DemoPSD fixes the teacher **TARGET** — replacing "fit the full privileged teacher" with "fit a disagreement-modulated geometric mixture of teacher and student". All three diagnose the same Yang-et al-2026 leakage phenomenon from a different lever.

---

## 2. Setup / problem framing

- **OPSD setting.** One model, two roles. Teacher = `π_θ(·|x, y*, ŷ_<t)` (current model conditioned on question `x` + privileged info `y*` + student rollout prefix). Student = `π_θ(·|x, ŷ_<t)` (same model, only `x` + prefix). `y*` injected by prepending it to the teacher's context (§4.4).
- **SDPO baseline (Eq 5).** Minimizes per-token `KL(π_θ(·|x,ŷ_<t) ‖ stopgrad(π_θ(·|x,y*,ŷ_<t)))` over a student-generated rollout `ŷ ∼ π_θ(·|x)`. The `stopgrad` keeps the teacher from drifting toward the student and ignoring `y*`.
- **Leakage problem (§3.3, Yang et al. 2026).** Because the teacher conditions on `y*` the student never sees, there is an irreducible mutual-information gap `I(y_t; y* | x, y_<t) > 0` (Eq 6). The student can never perfectly match the teacher's conditional. Early in training the beneficial gradient dominates (fast reward gains); as the student approaches the teacher's *marginal*, the `y*`-specific deviation takes over and the student encodes `x → y*` correlations = leakage. SDPO peaks early then degrades.

---

## 3. Method

### 3.1 Measuring teacher–student disagreement (§4.1)

At each token `t`, privileged teacher prediction `π_T^t(v, y*) := π_θ(v|x,y*,ŷ_<t)`, student prediction `π_S^t(v) := π_θ(v|x,ŷ_<t)`. Disagreement = **Jensen–Shannon divergence** (Eq 7):

```
d_t = JSD(π_S^t ‖ π_T^t) = ½ KL(π_S^t ‖ m_t) + ½ KL(π_T^t ‖ m_t),   m_t = ½(π_S^t + π_T^t)
```

**Leakage attenuation coefficient** `α_t = f(d_t)` (Eq 8) — a rescaled sigmoid:

```
α_t = (σ(β · d_t) − 0.5) · 2 · α_max
```

> ⚠ **Sourcing note (Eq 8 glyph scramble).** pdftotext renders Eq 8 as `α_t = σ(β·d_t) − 0.5 · 2 · α_max`, which is operator-precedence ambiguous. The reconstructed form `(σ(β·d_t) − 0.5)·2·α_max` is pinned by the two constraints the prose states in the same paragraph: `f(0)=0` (σ(0)=0.5 ⇒ α_t=0 when distributions agree ✓) and `lim_{d→∞} f(d) = α_max` (σ→1 ⇒ (1−0.5)·2·α_max = α_max ✓). Properties: `f` monotone increasing in `d_t`; saturates at `α_max` so extreme disagreement never fully discards the teacher. `β` = sensitivity (large β ⇒ sharper gate; small β ⇒ smoother, retains more teacher signal under moderate disagreement).

### 3.2 Reverse-KL barycenter target (§4.2)

The distillation target is a **geometric mixture** of teacher and student in log-prob space (Eq 9):

```
π_target^{α_t}(v | x, y*, ŷ_<t) ∝ π_T^t(v, y*)^{1−α_t} · π_S^t(v)^{α_t}
```

This is the **reverse-KL barycenter** under weight `α_t` (Eq 10): `π_target^{α_t} = argmin_q {(1−α_t) KL(q‖π_T^t) + α_t KL(q‖π_S^t)}`. Equivalently interpolates in log-prob space: `log π_target^{α_t} = (1−α_t) log π_T^t + α_t log π_S^t − log Z_{α_t}` (normalization `Z_{α_t}`, Eq 11).

**Geometric vs arithmetic mixture (§4.2).** Geometric chosen over arithmetic `(1−α_t)π_T + α_t π_S` because: (1) a token gets substantial target mass **only when supported by BOTH** teacher and student — teacher-only-supported tokens are suppressed (arithmetic would still give them mass); (2) arithmetic averages modes into a diffuse high-entropy target, geometric stays sharp/coherent. Consistent with AMiD (Shin et al. 2026) on mixture-geometry controlling mode-covering vs mode-seeking.

### 3.3 Loss and gradient (§4.3)

DemoPSD minimizes reverse-KL toward the barycenter target (Eq 12):

```
L_DemoPSD(θ) = E_{x∼D} E_{ŷ∼π_θ(·|x)} Σ_{t=1}^|ŷ| KL( π_θ(·|x,ŷ_<t) ‖ stopgrad(π_target^{α_t}(·|x,y*,ŷ_<t)) ) )
```

Full target is `stopgrad`-wrapped (teacher `π_T^t`, reference student `π_S^t`, weight `α_t` all fixed on backward pass), so `Z_{α_t}` is constant — no backprop through normalization. Gradient (Eq 13):

```
∇_θ L_DemoPSD = E_{ŷ∼π_θ} Σ_t E_{ŷ_t∼π_θ(·|x,ŷ_<t)} [ (1−α_t) · log( π_θ(ŷ_t|x,ŷ_<t) / π_θ(ŷ_t|x,y*,ŷ_<t) ) · ∇_θ log π_θ(ŷ_t|x,ŷ_<t) ]
```

**Same reverse-KL score-function form as SDPO, but the teacher-induced log-ratio signal is scaled by the disagreement factor `(1−α_t)`.** High-disagreement positions contribute a weaker distillation signal → less privileged-information backprop.

### 3.4 Privileged-info injection + training procedure (§4.4, Algorithm 1)

- **Context construction.** Teacher input: `[Question: x | Privileged Information: y* | Student Response: ŷ_<t]`. Student input: `[Question: x | Student Response: ŷ_<t]`. Same model; only difference = whether `y*` is in context.
- **Reprompting mechanism (Hübotter et al. 2026).** For each prompt group: if ≥1 rollout is correct, randomly select one correct rollout as `y*` (its response contains solution info → serves as privileged context); if no rollout correct, no reliable privileged teacher can be formed → **skip the prompt** for distillation. (Template-syntactic-insensitive per Hübotter.)
- **EMA reference.** A separate EMA copy of the student is used when computing both the disagreement (Eq 7) and the target (Eq 9), for stability.
- **Algorithm 1 loop:** sample batch → generate rollouts → keep only prompts with ≥1 correct rollout → for each: get teacher π_T^t, student π_S^t, compute d_t (Eq 7), α_t (Eq 8), target (Eq 9) → update θ via gradient descent on L_DemoPSD.

---

## 4. Theoretical analysis (§5)

**Leakage rate** (Yang et al. 2026 framing): `R_leak = E_t[‖Δ_t‖²]` where `Δ_t(v) = log π_T^t(v, y*) − log π_S^t(v)` (Eq 14) — the per-position squared log-prob shift induced by `y*`.

**Theorem 1 (Leakage Attenuation).** DemoPSD's effective leakage rate satisfies (Eq 15):

```
R_leak^DemoPSD := E_t[(1−α_t)² ‖Δ_t‖²]  <  E_t[‖Δ_t‖²] = R_leak
```

strict when `Pr(α_t > 0) > 0`. Attenuation is **strongest where leakage risk is greatest**: `α_t` monotone in `d_t`, `d_t` correlates positively with `‖Δ_t‖` (both measure teacher–student divergence) ⇒ large-`‖Δ_t‖` positions get large `α_t` ⇒ strongest suppression. (Full proof Appendix A.1.) DemoPSD doesn't just reduce average leakage — it selectively attenuates the positions that contribute most to it.

**Theorem 2 (Exploration Preservation).** Under the positive-alignment condition `Cov_{q_t^γ}(Δ_t, log π_S^t) ≥ 0` (Eq 16) for every geometric interpolation `q_t^γ ∝ (π_T^t)^γ (π_S^t)^{1−γ}`, `γ ∈ [0,1]`:

```
H(π_S^t)  ≥  H(π_target^{α_t})  ≥  H(π_T^t)                                       (Eq 17)
```

strict when `0 < α_t` and `π_T^t ≠ π_S^t`. The entropy gain `H(π_target^{α_t}) − H(π_T^t) ≥ 0` over the full-teacher target is **non-decreasing in `α_t`**: the more the teacher depends on `y*` (larger `α_t`), the more exploration DemoPSD retains vs SDPO.

*Proof sketch:* entropy along the geometric path `q_t^γ ∝ π_S^t · e^{γΔ_t}` (exponential family, sufficient statistic `Δ_t`):

```
dH/dγ = −γ Var_{q_t^γ}[Δ_t] − Cov_{q_t^γ}(Δ_t, log π_S^t)            (Eq 18)
```

Both terms non-positive under (16) ⇒ `H(q_t^γ)` decreasing in `γ`. DemoPSD halts the interpolation at `γ = 1−α_t < 1` (not the full teacher `γ=1`), so it never pays the final steepest portion of the entropy cost; the saving grows with `α_t`. Condition (16) is the natural self-distillation regime: the teacher (same model + answer info) predominantly *sharpens* existing predictions rather than contradicting them. (Full proof Appendix A.2.)

> The two theorems jointly characterize the barycenter target: disagreement-dependent weighting attenuates privileged-deviation contributions (Thm 1) **and** the target stays strictly more entropic than the full privileged teacher (Thm 2), so dense token-level supervision is retained where teacher/student agree and down-weighted where they diverge.

---

## 5. Experimental setup (§6.1)

- **Base model:** Qwen3-4B-Instruct (all experiments).
- **Train data:** SciKnowEval (Feng et al. 2024) — multi-domain scientific reasoning, 4-choice MCQ; train + eval separately on 4 domains: biology, chemistry, material science, physics.
- **Eval:** SciKnowEval (in-domain, domain-matched test) + **GPQA Extended** (Rein et al. 2023, OOD graduate-level science, bio/chem/physics — **material has no GPQA counterpart**).
- **Metrics (16 rollouts/prompt):** `mean@16` (avg accuracy), `maj@16` (majority-vote accuracy), `best@16` (best of 16).
- **Baselines:** GRPO (Shao et al. 2024, outcome-reward RLVR) and SDPO (Hübotter et al. 2026, OPSD). Same codebase, infra, base model, data; differ only in objective.
- **Shared hyperparams:** lr `1e-6`, batch 64, **8 rollouts/prompt for training**, max prompt 2048, max response 16384, 10 warmup steps, 3 epochs.
- **Distill-method extras (SDPO + DemoPSD):** top-k=100 distillation, EMA rate `η=0.05`, train temp 1.0 / val temp 0.7.
- **DemoPSD-specific:** `α_max = 0.15`; `β` tuned per-domain (§6.5).
- **GRPO-specific:** KL-penalty `β_KL = 0.04`, importance-sampling clip 2.0.
- **Hardware:** 8× NVIDIA H20 GPUs, FSDP, vLLM for rollout gen, flash attention.

---

## 6. Results — verbatim tables

### 6.1 Main results — SciKnowEval (Table 1, `paper_layout.txt` L572–582)

DemoPSD consistently outperforms both GRPO and SDPO across all four domains and all three metrics. Bold = best per metric (reconstructed: DemoPSD wins every Avg cell and every per-domain cell except where noted).

| Domain | mean@16 GRPO / SDPO / DemoPSD | maj@16 GRPO / SDPO / DemoPSD | best@16 GRPO / SDPO / DemoPSD |
|---|---|---|---|
| Biology | 33.51 / 36.88 / **39.25** | 34.84 / 38.07 / **40.64** | 58.36 / 64.04 / **68.51** |
| Chemistry | 65.83 / 71.70 / **72.98** | 66.72 / 72.41 / **73.71** | 80.47 / 85.94 / **90.05** |
| Material | **76.32** / 76.13 / **76.53** | 76.50 / 76.24 / **76.71** | 80.24 / 81.69 / **81.79** |
| Physics | 66.31 / 68.98 / **71.64** | 70.52 / 71.88 / **74.24** | 82.59 / 85.51 / **88.13** |
| **Average** | 60.49 / 63.42 / **65.10** | 62.14 / 64.65 / **66.33** | 75.42 / 79.30 / **82.12** |

**Source-free reconciliation (Avg = mean of 4 domains, all recomputed):**
- mean@16 Avg: GRPO (33.51+65.83+76.32+66.31)/4 = 60.4925 → **60.49 ✓**; SDPO (36.88+71.70+76.13+68.98)/4 = 63.4225 → **63.42 ✓**; DemoPSD (39.25+72.98+76.53+71.64)/4 = 65.10 ✓.
- maj@16 Avg: GRPO 62.145 → **62.14 ✓**; SDPO 64.65 ✓; DemoPSD 66.325 → **66.33 ✓** (half-up at the .xx5 boundary; computed from full-precision per-prompt values, consistent with the iter-45/51 display-rounding pattern).
- best@16 Avg: GRPO 75.415 → **75.42 ✓**; SDPO 79.295 → **79.30 ✓**; DemoPSD 82.12 ✓.

**DemoPSD − SDPO deltas (§6.2 prose, all recompute exact):** mean@16 **+1.68** (65.10−63.42 ✓), maj@16 **+1.68** (66.33−64.65 ✓), best@16 **+2.82** (82.12−79.30 ✓). The best@16 margin is largest ⇒ preserved exploration entropy surfaces higher-quality reasoning paths during sampling.

> ⚠ **Paper-internal prose-vs-table inconsistency (the iter-30/31/34 class — transcribed verbatim, NOT reconciled).** §6.2 (L675) states: *"Compared to GRPO, the total gain from DemoPSD is **5.21** on mean@16."* This does **not** reconcile with Table 1: DemoPSD−GRPO on mean@16 = 65.10−60.49 = **4.61** (delta-of-averages); mean-of-per-domain-deltas = (5.74+7.15+0.21+5.33)/4 = **4.61** — identical, neither gives 5.21. The other metrics don't yield 5.21 either (maj@16 Δ = 4.19; best@16 Δ = 6.70). The "+1.68/+1.68/+2.82 vs SDPO" and "+7.91 vs SDPO on GPQA" claims all reconcile exactly; only the "5.21 vs GRPO" figure is unreconcilable from Table 1 — likely a stale number from an earlier checkpoint/draft. Flag, don't echo.

> ⚠ **Honest-scope note (Material domain).** On Material mean@16, **SDPO (76.13) < GRPO (76.32)** — material is the only domain where dense self-distillation underperforms outcome-only RLVR, and DemoPSD's margin over GRPO there is the smallest of any domain (+0.21 mean@16). The "consistently outperforms both GRPO and SDPO across all four domains" caption holds, but Material is a near-tie, not a decisive win.

### 6.2 OOD generalization — GPQA Extended (Table 2, L584–595)

Values at the final training stage (mean over last 3 evaluations). Material has no GPQA counterpart.

| Method | Biology | Chemistry | Physics | Average |
|---|---|---|---|---|
| SDPO | 57.81 | 28.62 | 52.99 | 46.47 |
| DemoPSD | **61.42** | **41.75** | **59.98** | **54.38** |

**Reconciliation:** SDPO avg (57.81+28.62+52.99)/3 = 46.473 → **46.47 ✓**; DemoPSD avg (61.42+41.75+59.98)/3 = 54.383 → **54.38 ✓**. DemoPSD − SDPO avg = **+7.91** ✓ (matches §6.3 prose "ending 7.91 above SDPO on average").

> ⚠ **SDPO chemistry degradation (figure-anchored).** §6.3 reports SDPO chemistry GPQA peaks early at **40.45** then degrades to the **28.62** final (Table 2) — a **−11.83** drop, the largest OOD decline and the empirical signature of accumulating leakage (Figure 3 curve). 40.45 is a **Figure-3 peak reading** (prose-only at L693); 28.62 is the verbatim T2 cell. DemoPSD chemistry instead *improves* over training (41.75 final > its early value). This peak-then-degrade vs stable-and-improve contrast is the paper's central OOD finding.

### 6.3 Training dynamics at final step (Table 3, L704–720)

| Domain | Method | Entropy | ΔEnt. | mean ᾱ_t | mean d̄_t | Active % |
|---|---|---|---|---|---|---|
| Biology | SDPO | 0.602 | – | – | – | – |
| Biology | DemoPSD | 0.816 | +35.5% | 0.055 | 0.046 | 64.8 |
| Chemistry | SDPO | 0.322 | – | – | – | – |
| Chemistry | DemoPSD | 0.555 | +72.4% | 0.036 | 0.037 | 84.0 |
| Material | SDPO | 0.150 | – | – | – | – |
| Material | DemoPSD | 0.297 | +98.0% | 0.033 | 0.031 | 68.8 |
| Physics | SDPO | 0.385 | – | – | – | – |
| Physics | DemoPSD | 0.511 | +32.7% | 0.040 | 0.026 | 90.6 |

**Reconciliation (ΔEnt = (Demo−SDPO)/SDPO, all recompute):**
- Bio (0.816−0.602)/0.602 = 0.3555 → **+35.5% ✓**; Chem (0.555−0.322)/0.322 = 0.7236 → **+72.4% ✓**; Material (0.297−0.150)/0.150 = 0.980 → **+98.0% ✓**; Physics (0.511−0.385)/0.385 = 0.3273 → **+32.7% ✓**.

> ⚠ **Abstract/conclusion-vs-table entropy-range rounding.** Abstract (L20) + Fig-1a caption (L56) + §5 Thm-2 ref (L559) + §6.4 prose (L737) all say **"33–98% higher entropy"**, but the Conclusion (L819) says **"35–98% higher training entropy"**, and Table 3's true minimum is **+32.7% (Physics)**. So: abstract rounds 32.7 → 33; conclusion's "35" matches Biology's +35.5% as the minimum (i.e. excludes Physics); Table 3 ground truth min = 32.7%. Three different lower bounds for the same range — harmless rounding/editorial drift, not a contradiction; the upper bound 98% (Material) is stable across all four statements.

**Reading the dynamics:**
- **Entropy preservation.** DemoPSD maintains 33–98% higher final entropy than SDPO across all domains. Largest gap = Material (+98.0%), where SDPO's 0.150 is close to entropy collapse (consistent with Thm 2: DemoPSD halts the geometric path before the steepest entropy cost).
- **Disagreement sparsity.** Mean attenuation `ᾱ_t` stays consistently low (0.033–0.055) and mean disagreement `d̄_t` is 0.026–0.046 ⇒ target stays close to the teacher for most tokens; strong attenuation applied only to a small high-disagreement subset. This is the *selective adoption* principle made visible.
- **Active sample fraction.** "Active %" = fraction of training samples with ≥1 correct rollout in the prompt group (i.e. a valid privileged teacher context exists). Correlates with domain difficulty (Material 68.8% / Biology 64.8% harder ⇒ fewer correct rollouts; Physics 90.6% / Chemistry 84.0% easier).

### 6.4 β sensitivity — mean@16 (Table 4, L722–731)

All configurations use `α_max = 0.15`. "–" = configuration not run.

| β | Biology | Chemistry | Material | Physics |
|---|---|---|---|---|
| 15 | – | 71.93 | – | – |
| 25 | – | **72.98** | 76.46 | – |
| 50 | – | 71.90 | **76.53** | 70.55 |
| 70 | **39.25** | – | – | – |
| 100 | 36.88 | – | 76.06 | **71.64** |
| SDPO (ref) | 36.88 | 71.70 | 76.13 | 68.98 |

**Best-β per domain (§6.5):** Biology β=70 (39.25), Chemistry β=25 (72.98), Material β=50 (76.53), Physics β=100 (71.64) — **each best-β cell is byte-identical to the Table 1 DemoPSD column** (39.25 / 72.98 / 76.53 / 71.64), a free cross-table check that Table 1 reports each domain's best-β DemoPSD run.

**Pattern (§6.5):** domains with *smaller* disagreement benefit from *higher* β (Physics, mean `d̄_t`=0.026 ⇒ β=100 amplifies the weak signal); domains with *greater* disagreement benefit from *lower* β (Biology, mean `d̄_t`=0.046 ⇒ β=70 avoids over-aggressive hedging). Across β ∈ [25,100], DemoPSD consistently matches or outperforms SDPO ⇒ moderate robustness. All top configs use the **remapped α-schedule (Eq 8)** constraining `α_t ∈ [0, α_max]`, guaranteeing the teacher retains ≥ `(1−α_max)` of the mixture weight.

> ⚠ **Coincidental identical value (not an error).** β=100 Biology (36.88) and the SDPO Biology reference (36.88) are numerically identical — two different methods/runs happening to land on the same mean@16. Transcribed verbatim; flagged so it isn't read as a duplication or a transcription slip.

### 6.5 Disagreement-attenuation statistics — best run per domain (Table 5, L759–763)

| Statistic | Biology | Chemistry | Material | Physics |
|---|---|---|---|---|
| Mean leakage attenuation coefficient α_t | 0.055 | 0.036 | 0.033 | 0.040 |
| Mean disagreement d_t | 0.046 | 0.037 | 0.031 | 0.026 |
| Active sample fraction | 64.8% | 84.0% | 68.8% | 90.6% |

> ⚠ **Table 5 = Table 3's last three columns, verbatim (paper-internal redundancy, not a contradiction).** Every Table 5 cell (α_t, d_t, Active fraction for all 4 domains) is byte-identical to Table 3's `mean ᾱ_t / mean d̄_t / Active %` columns. Table 5's caption clarifies these are the **best-performing DemoPSD run per domain** — confirming Table 3 *is* the best-β run and re-stating its disagreement statistics in isolation. Useful as a free source-free consistency check (T3 ↔ T5 cross-agreement), not new data.

**Disagreement distribution (§6.6, Figure 4):** strongly right-skewed — the vast majority of tokens have near-zero disagreement (student closely aligned with teacher, dense supervision preserved); only a small subset (≈2–5% of tokens, those exceeding `d_t > 0.25`) show substantial divergence and trigger stronger attenuation. This sparsity is the mechanism by which DemoPSD keeps OPSD's dense token-level supervision for most positions while selectively attenuating leakage only where teacher–student mismatch is pronounced.

---

## 7. Figures (qualitative / figure-derived — NOT back-filled)

Per the universal figure-derived-numbers-are-weak rule, only prose-confirmed markers and printed labels are quoted; per-point curve/axis values are not transcribed.

- **Fig 1a:** policy entropy over training steps — DemoPSD maintains 33–98% higher entropy than SDPO across all domains, avoiding entropy collapse.
- **Fig 1b:** best@16 validation accuracy per SciKnowEval domain over training — higher entropy → better best@16.
- **Fig 2a:** validation mean@16 over training per domain — DemoPSD ≥ SDPO throughout, gap grows in later epochs (leakage reduction helps more as student approaches teacher marginal).
- **Fig 2b:** validation mean@16 vs β per domain (dashed line = SDPO baseline) — DemoPSD competitive/above across β ∈ [25,100], optimum varies by domain.
- **Fig 3:** GPQA accuracy over training per domain — SDPO peaks early then degrades (chemistry 40.45 → 28.62), DemoPSD stable and improving.
- **Fig 4a:** per-token JSD disagreement `d_t` distribution per domain at final step — heavily right-skewed, 2–5% of tokens exceed 0.25.
- **Fig 4b:** mean `α_t` (blue, left axis) and mean `d_t` (pink, right axis) over training — both small and stable.

---

## 8. Strengths / limitations / verdict

**Strengths**
- **Falsifiable mechanism with matching theory.** The leakage fix is not a heuristic: Theorem 1 proves `R_leak^DemoPSD < R_leak` with attenuation strongest where `‖Δ_t‖` is largest; Theorem 2 proves `H(π_target) ≥ H(π_T)` under a stated (and natural-for-self-distillation) covariance condition. The gradient (Eq 13) literally scales the teacher log-ratio by `(1−α_t)`.
- **Cheap, drop-in.** No new parameters, no external teacher. Only added cost = one extra forward (teacher with `y*`) + JSD/barycenter computation per token, all `stopgrad`-wrapped. Sits in the same OPSD training loop as SDPO.
- **Empirically clean sweep.** DemoPSD > SDPO > GRPO on every Avg cell and (except Material near-tie) every per-domain cell of Table 1; +7.91 OOD avg on GPQA; SDPO's peak-then-degrade (chemistry 40.45→28.62) vs DemoPSD's stable-improve is exactly the leakage diagnosis the theory predicts.

**Limitations (honest scope)**
- **Eval is narrow.** One base model (Qwen3-4B-Instruct), one in-domain benchmark family (SciKnowEval, 4-choice scientific MCQ), one OOD benchmark (GPQA). No code/math/generation tasks; no scaling beyond 4B; no ablation of the reprompting template or EMA rate.
- **Material is a near-tie.** On Material, SDPO < GRPO and DemoPSD edges GRPO by only +0.21 mean@16 — the selective-adoption gain is smallest where the teacher and student already agree most (Material has the lowest disagreement, `d̄_t`=0.031).
- **Covariance condition (Eq 16) is an assumption.** Thm 2's entropy ordering holds under `Cov(Δ_t, log π_S) ≥ 0` — natural for self-distillation (teacher sharpens existing predictions) but not proven to always hold; if the teacher ever *contradicts* the student broadly, the entropy-preservation guarantee can fail.
- **"5.21 vs GRPO" prose figure unreconcilable** (see §6.1 ⚠) — the verifiable gain is 4.61 on mean@16.
- **Entropy-range lower bound drifts** across abstract/conclusion/table (33 / 35 / 32.7%) — cosmetic, but a careful reader should cite the Table-3 ground truth (32.7–98%).

**Verdict.** DemoPSD is a clean, theoretically-grounded, drop-in OPSD-leakage fix that targets a **different lever** (the teacher TARGET — a disagreement-modulated reverse-KL barycenter) from `purified-opsd` (teacher UPDATE, PMI purification) and `N-OPSD` (teacher INPUT, neuron-aware data+context). The three together give the repo's distillation lineage a complete picture of where privileged-information leakage enters OPSD and how to block it. The contribution is the mechanism + theorems + the stable-OOD training dynamic; the absolute gains are modest (mean@16 +1.68 vs SDPO, +4.61 vs GRPO) and the eval is single-model/single-domain-family, so it's a recipe/understanding paper, not a new SOTA.

---

## 9. Repo subarea lineage

DemoPSD extends the repo's **distillation / OPSD lineage**:
- `danceopd` — on-policy generative *field* distillation (iter 4) — diffusion-editing teacher.
- `opid` — on-policy *skill* distillation (iter 14) — agentic skill transfer.
- `purified-opsd` (iter 52) — long-CoT OPSD, fixes teacher **UPDATE** via PMI purification of `Δ_it`.
- `neuron-aware-data-selection / N-OPSD` (iter 54) — annotation-free self-distillation, fixes teacher **INPUT** via neuron-aware data+context.
- **DemoPSD (this)** — privileged-info-leakage fix at the teacher **TARGET** via disagreement-modulated reverse-KL barycenter.

It is mechanistically adjacent to `multi-turn-rl / IRC` (iter 30) and `demystifying-rl` (iter 29) in the RLVR family (GRPO is the shared baseline) but its object is the *distillation* signal, not the reward.
