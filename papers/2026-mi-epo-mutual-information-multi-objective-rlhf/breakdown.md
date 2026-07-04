# MI-EPO — Multi-Objective Exploration and Preference Optimization via Mutual Information

**arXiv:** 2607.01392v1 [cs.CL], 1 Jul 2026
**Authors:** Hongyan Xie¹ˢʰᵃʳʳᵒʷ, Yikun Ban¹, Ruiyu Fang², Zixuang Huang¹, Deqing Wang¹, Jianxin Li¹, Shuangyong Song²
**Affil:** ¹School of Computer, Beihang University, Beijing · ²Xingchen AGI Lab, China Telecom AI Technology (Beijing)
(*Work completed during an internship at Xingchen AGI Lab.*)
**Source:** paper.pdf (1.40 MB, **22 pp** — pdfinfo=22pp; `file` misreports **1pp**, the largest file-vs-pdfinfo gap seen in the repo [iters 66/67/68 → now 69]; trust pdfinfo); paper_layout.txt = `pdftotext -layout`, 1231 lines.
**Subarea (NEW for repo):** multi-objective RLHF / multi-objective alignment via **mutual-information maximization** — repo's FIRST paper on MO-RLHF, MOA, preference-conditioned single-policy alignment, or mutual information as an alignment objective. No prior repo paper covers multi-objective preference optimization, Pareto-front alignment within one model, or InfoNCE/CMI for preference control.
**Sibling-in-spirit lineage:**
- *Reward-design / preference-optimization lineage* (`distribution-wise-rewards`, `multi-role-rubric`, `verifiable-process-reward-models`, `evidence-state-rewards`): all shape the training signal; MI-EPO shapes it information-theoretically (max joint CMI between response, feedback, and the conditioning preference vector).
- *Contrastive/InfoNCE-as-objective* is borrowed from representation learning (the paper's own §5 distinguishes it from `Li et al. 2022` MI-for-representation) and re-purposed for *policy* optimization.
- *Online DPO / on-policy self-distillation lineage* (`purified-opsd`, `demopsd`, `neuron-aware-data-selection`): MI-EPO is online (samples y1,y2 from current πθ) and is a DPO relative (Eq 8 = per-objective DPO; Eq 12 = a DPO-style contrastive term over preference vectors).

---

## 1. Problem & central diagnosis

**Setting.** Multi-Objective Alignment (MOA): align ONE policy πθ(y | x, w) with a preference vector **w** ∈ Δ^{K-1} (probability simplex over K objectives) so a single model trades off conflicting preference dimensions (helpfulness/harmlessness/humor). Backbones: Alpaca-7B (safety), Qwen3-8B (helpful assistant).

**Prior best baseline = MO-ODPO** (Gupta 2025): conditions the policy on **w** in the prompt + online DPO. Scalarized MOO (Eq 5): min_θ Σ_k w_k · L_DPO(πθ; (x,w), y_{+,k}, y_{−,k}).

**Central diagnosis (Fig 1).** Online exploration uncertainty **weakens the conditional control of W over Y**: (a) reward distributions of responses under *different* preference vectors substantially **overlap** (Fig 1a); (b) high exploration uncertainty (Fig 1b). The generated responses fail to faithfully align with the conditioning **w**. MI-EPO's fix is information-theoretic: maximize the joint conditional MI between Y and (preference feedback, preference vector) so responses are **distinguishable across preferences** and **aligned with each preference**.

---

## 2. Method (Eqs 1–13, Algorithm 1)

### 2.1 Preliminary — DPO-as-CMI (Eqs 1–4)

DPO reparameterization (Eq 1): r(x,y) = β log(πθ(y|x)/π_ref(y|x)) + β log Z(x). DPO loss (Eq 2): L_DPO = −log σ(β log(πθ(y+|x)/π_ref) − β log(πθ(y−|x)/π_ref)).

Conditional Mutual Information (CMI) I(Y;C|X) bounded below by **InfoNCE** (Eq 3). With critic f_ϕ and preference feedback c∈{0,1}, InfoNCE reduces to a pairwise contrastive loss (Eq 4); and when f_ϕ(x,y) = β log(πθ/π_ref), InfoNCE **reduces to DPO Eq 2** — the key link that lets the authors treat DPO as a variational MI estimator (following Xiao 2025).

### 2.2 Joint CMI objective + chain-rule decomposition (Eqs 5–7)  ⭐ citable contribution

Introduce a **probabilistic objective-routing variable Z ∈ {1,…,K}** with P(Z=k | W) = w_k (the preference vector routes which objective's feedback C_Z is active). The overall alignment objective is the **joint conditional MI**:

> **J(θ) = I(Y ; C_Z, W, Z | X)**

Apply the MI chain rule + recognize Z ⊥ Y | W (so I(Y;Z|X,W)=0):

> **J(θ) = I(Y ; W | X) + Σ_k w_k · I(Y ; C_k | X, W)**     (Eq 6)

This is the paper's central decomposition: the joint MI **exactly splits** into two complementary terms —
- **I(Y ; C_k | X, W)** = *objective-specific preference alignment* (maximized by DPO, §2.3);
- **I(Y ; W | X)** = *preference-conditioned exploration / identifiability* (reduces posterior uncertainty H(W|Y,X); structural lower bound on marginal entropy H(Y|X) ⇒ diversity across preferences, §2.4).

Constraint (Eq 7): max_θ J(θ) s.t. Σw_k=1, w_k≥0.

**Information-theoretic interpretation** (§3.1, prose): I(Y;W|X) = H(W|X) − H(W|Y,X); under fixed prior entropy, maximizing it minimizes posterior uncertainty ⇒ the generation becomes more *informative* about the preference (identifiability). And H(Y|X) = I(Y;W|X) + H(Y|W,X), so I(Y;W|X) is a structural lower bound on marginal entropy ⇒ encourages diversity in Y while staying consistent conditioned on W ⇒ a "partitioned exploration space" of distinguishable-yet-diverse preference-induced modes.

### 2.3 Maximize I(Y ; C_k | X, W) — preference alignment (Eq 8)

Per-objective DPO (Eq 8): **L_YC = −log σ(β_c log(πθ(y_{+,k}|x,w+) / π_ref) − β_c log(πθ(y_{−,k}|x,w+) / π_ref))**, where w+ is the preference vector used for sampling. (DPO-as-CMI ⇒ maximizes I(Y;C_k|X,W).)

### 2.4 Maximize I(Y ; W | X) — reduce exploration uncertainty (Eqs 9–12)

InfoNCE lower bound on I(Y;W|X) (Eq 9). Positive pair (y,w+) ∼ πθ(y|x,w+); negative pair reuses y with independent w− ∼ p(w|x). Pairwise contrastive loss (Eq 10): L_YW = −log(exp(g_ϕ(y,w+)) / (exp(g_ϕ(y,w+)) + exp(g_ϕ(y,w−)))). Implicit-reward critic (Eq 11): g_ϕ(y,w) = β_w log(πθ(y|x,w) / π_ref(y|x,w)). Substituting ⇒ DPO-style InfoNCE loss (Eq 12):

> **L_YW = −log σ(β_w log(πθ(y|x,w+) / π_ref) − β_w log(πθ(y|x,w−) / π_ref))**

### 2.5 Final MI-EPO loss (Eq 13) + stop-gradient

> **L_MI-EPO = Σ_k w_k · L_YC(πθ; y_{+,k}, y_{−,k}) + (1/2) Σ_{y∈{y1,y2}} L_YW(πθ; w+, w−)**     (Eq 13)

with stop-gradient **sg(·)** on the L_YW anchor-logit so the gradient flows only through the counterfactual condition w− (prevents overfitting to self-generated y). Since L_YW is feedback-independent, both y1,y2 (generated under w+) serve as valid anchors.

**Reduction check (βw=0 ⟺ MO-ODPO):** set β_w=0 ⇒ the L_YW term vanishes ⇒ L_MI-EPO = Σ_k w_k L_YC = Eq-5 scalarized MO-ODPO. ✅ **Confirmed structurally** — the L_YW term is the *only* thing MI-EPO adds over MO-ODPO, so ablation βw=0 must recover MO-ODPO exactly. (Paper states this, §4.1.)

### 2.6 Algorithm 1 (verbatim, L486–503)

```
Require: πθ0; reward models {R_k}_{k=1}^K; dataset D; epochs N
1: for n := 1 to N do
2:   for each prompt x ∈ D do
3:      Sample w+, w− ∼ Dirichlet(α)                      # α=0.5 safety, 1.0 helpful (Table 4)
4:      Sample y1, y2 ∼ πθ_{n-1}(· | G(x, w+))            # G(x,w)=Human:{x}⊕RN1{w1}…RNK{wK}⊕Assistant:
5:      for k = 1 to K do
6:         Compute reward scores s_{1,k}, s_{2,k} via R_k
7:         Higher-score response → y_{+,k}; other → y_{−,k}
8:      end for
9:      Compute loss Eq 13, update θ via gradient descent
10:   end for
11: end for
```
Prompt-construction G(x,w) prepends K textual preference tokens RN_k (e.g. "helpfulness", "harmlessness", "humor").

---

## 3. Setup (Tables 3–4, §4, App A/B)

| Item | Safety Alignment | Helpful Assistant |
|---|---|---|
| Dataset | PKU-SafeRLHF-10K (8K train / 2K test) | HH-RLHF (~160K dialogs; 8K train / 2K test) |
| Backbone | Alpaca-7B | Qwen3-8B |
| Objectives K | 2 (helpfulness, harmlessness) | 2 (help, harmless) **and** 3 (help, harmless, humor) |
| Oracle reward models | Helpfulness; Harmlessness (open-source) | 3 independent open-source RMs |

**Common hyperparameters (Table 4, L1128–1149):** Transformer + TRL; NVIDIA Tesla A100 40GB; bf16; **LoRA r=64, α=128, dropout 0.05**; Adam + cosine LR; warmup 0.1; batch 64; max inference tokens 128. **Dirichlet α = 0.5 (safety) / 1.0 (helpful)**. SFT 3 epochs (LR 1e-6 safety / 2e-5 helpful). Online stage 2 epochs (2-obj) / 3 epochs (3-obj); online LR 1e-4 safety / 4e-5 helpful. **β_c = 0.1, β_w = 0.01.** Rewards normalized per-dimension (mean/var on offline data) before online stage. Baselines reproduced from Yang 2024b (RiC) codebase.

**Baselines:** (1) **Rewarded Soups (RS)** — fine-tune K base models, merge in parameter space at inference by w. (2) **RiC** (Rewards-in-Context) — condition policy on multiple contextual rewards + SFT. (3) **MO-ODPO** — multi-objective online DPO, single policy, preference-conditioned prompts (the strongest baseline; MI-EPO reduces to it at βw=0).

---

## 4. Evaluation metrics (App C, Eqs 14–17)

Three multi-objective metrics computed over a discrete grid of preference vectors on the simplex (11 vectors for 2-obj: [0,1]…[1,0]; 13 representative vectors for 3-obj). All rewards **normalized**.

- **HV (Hypervolume, ↑)** — Eq 14 (Lebesgue measure of objective space dominated by solution set A relative to reference z); larger = better convergence + diversity.
- **MIP (Mean Inner Product, ↑)** — Eq 15 (MIP = (1/N) Σ w_i^⊤ s_i); alignment between response reward vector and user preference vector; larger = better conformity.
- **CRD (Conditional Reward Dispersion, ↓)** — Eqs 16–17 (CRD = (1/|W|) Σ_w det(Σ_w), Σ_w = Cov(s|w) = generalized variance / ellipse area); smaller = more stable conditional control, less reward dispersion under a fixed preference.

---

## 5. Results — tables verbatim

### Table 1 — Safety Alignment (2-obj, Alpaca-7B), HV↑ / MIP↑ / CRD↓ (L593–601)

| Method | HV↑ | MIP↑ | CRD↓ |
|---|---|---|---|
| RS | 1.19 | 0.43 | 0.64 |
| RiC | 1.13 | 0.58 | 0.89 |
| MO-ODPO | 1.54 | 0.82 | 0.62 |
| **MI-EPO** | **2.70** | **1.01** | **0.29** |

Reconciliation (vs strongest baseline MO-ODPO):
- **HV: (2.70−1.54)/1.54 = +75.3%** — prose §4.1 (L622) claims **"68.8%"** ⚠️ **DOES NOT RECONCILE** (MIP + CRD below do). The 68.8% would require MO-ODPO HV = 1.60 (shown 1.54) or MI-EPO HV = 2.60 (shown 2.70).
- **MIP: (1.01−0.82)/0.82 = +23.17% ≈ prose "23.2%" ✅ EXACT.**
- **CRD: (0.62−0.29)/0.62 = 53.2% reduction ≈ prose "53.2%" ✅ EXACT.**
- MI-EPO column-max (HV/MIP) + column-min (CRD). ✅

### Table 2a — Helpful Assistant (2-obj, Qwen3-8B), HV↑ / MIP↑ / CRD↓ (L689–723)

| Method | HV↑ | MIP↑ | CRD↓ |
|---|---|---|---|
| RS | 0.68 | 0.21 | 0.58 |
| RiC | 0.46 | 0.07 | 1.05 |
| MO-ODPO | 1.68 | 0.64 | 0.57 |
| **MI-EPO** | **1.98** | **0.68** | **0.52** |

Reconciliation (vs MO-ODPO): HV +17.9%, MIP +6.3%, CRD −8.8% reduction. MI-EPO best on **all 3** metrics (HV max, MIP max, CRD min). ✅ No prose % given for the 2-obj subtable (§4.2 is qualitative), so nothing to contradict.

### Table 2b — Helpful Assistant (3-obj: helpful/harmless/humor), HV↑ / MIP↑ / CRD↓ (L689–723)

⚠️ **Cells LOST in two-column layout scramble** — `pdftotext -layout` collapsed subtables (a)+(b) onto one column and only subtable (a)'s 12 decimals survived; subtable (b)'s numeric cells did not render as distinct tokens (grep of L690–730 returns exactly 12 decimals, all attributable to (a)). The **only** quotable 3-obj result is prose §4.2 (L794–797): "Compared with MO-ODPO, MI-EPO achieves a relative improvement of **87.2% in HV** and a **29.6% gain in MIP**, while maintaining highly competitive CRD." → **UNVERIFIABLE against cells.** (Fig 4a/4b are the 3-obj Pareto-frontier / per-dimension plots, figure-only.)

### Table 3 — dataset & model sources (L1088–1092)
PKU-SafeRLHF-10K (safety); Alpaca-7B base; Helpfulness + Harmlessness oracle RMs. (Helpful-assistant row in original is HH-RLHF / Qwen3-8B / 3 RMs, prose-confirmed §4.2.)

### Table 4 — hyperparameters (L1127–1149)
Verbatim in §3 above (LoRA r=64/α=128/dropout 0.05; Adam+cosine; warmup 0.1; batch 64; 128 tokens; Dirichlet α 0.5/1.0; SFT 3 epochs LR 1e-6/2e-5; Online 2(2-obj)/3(3-obj) epochs LR 1e-4/4e-5; β_c=0.1, β_w=0.01).

---

## 6. Headline (prose-/table-/equation-confirmed; NO figure back-fill)

- **MI-EPO > all baselines on every Table-1 metric** (safety, Alpaca-7B): HV 2.70 (vs MO-ODPO 1.54), MIP 1.01, CRD 0.29 — best HV/MIP, lowest CRD.
- **MI-EPO best on all 3 Table-2a metrics** (helpful, 2-obj, Qwen3-8B): HV 1.98, MIP 0.68, CRD 0.52.
- **3-objective scaling (Table 2b, prose-only):** HV +87.2%, MIP +29.6% vs MO-ODPO — the MI advantage *grows* with #objectives ("as the number of aligned objectives increases, the advantages become more pronounced", §4.2).
- **β_w=0 ⟺ MO-ODPO** (structural, Eq 13→5) — the L_YW exploration term is the sole addition; ablation must recover the baseline.
- **Reward-distribution separation (Fig 2b/3b):** MI-EPO's reward distributions under different w are visibly more separated (less overlap) and lower-variance than MO-ODPO's — the central diagnosis (§1) addressed. (Figure-only; qualitative.)
- **Joint CMI decomposition (Eq 6)** is the citable falsifiable contribution: alignment (I(Y;C_k)) + exploration-uncertainty-reduction (I(Y;W)) unified under one information-theoretic objective via the chain rule, with the probabilistic routing variable Z making the decomposition exact.

---

## 7. ⚠ Honest-scope flags (transcribed verbatim / NOT reconciled where noted)

1. **Table-1 HV prose-vs-table INCONSISTENCY (iter-30/31/34/60 DemoPSD-5.21 class).** §4.1 L622 claims HV improves **68.8%** over MO-ODPO, but the displayed cells give **(2.70−1.54)/1.54 = +75.3%**. MIP (+23.2%) and CRD (−53.2%) reconcile EXACTLY from the same cells — so the gap is specific to the HV attribute. The 68.8% would require a different MO-ODPO HV (1.60) or MI-EPO HV (2.60). **Flag, don't echo 68.8%; the table cells imply 75.3%.**
2. **Table 2b (3-obj) cells are UNVERIFIABLE** — lost in the two-column `pdftotext` scramble; only prose deltas 87.2% HV / 29.6% MIP are quotable, and they cannot be checked against cells. The single most eye-catching scaling claim (advantage grows with #objectives) rests on this un-checkable subtable.
3. **Figure-only central evidence.** The *core* diagnosis (reward-distribution overlap Fig 1a/2b/3b, exploration uncertainty Fig 1b, 3-obj Pareto frontier Fig 4a) is qualitative figure-only — Tables 1/2a are the only fully-numeric evidence. The "distributions more distinctly separated" claim is not reducible to a cell.
4. **No seeds / no CIs / no significance** on any table — single point estimates; the 2-obj Table-2a gaps (HV +17.9%, MIP +6.3%, CRD −8.8%) are small enough to sit within run-to-run noise. No SD/bootstrap reported (parallel to iter-66 SASP no-variance defect).
5. **Two backbones only (Alpaca-7B, Qwen3-8B), two datasets (PKU-SafeRLHF-10K, HH-RLHF), K∈{2,3}.** No scale ablation (e.g. 70B), no K>3, no third domain. "As #objectives increases" claim extrapolates from K=2 vs a single K=3 subtable.
6. **Reward models as ground truth.** "Alignment" is measured against open-source oracle RMs (Helpfulness/Harmlessness/...), not human judgement — so HV/MIP/CRD improvements reflect RM-alignment, not necessarily real human multi-objective preference (standard RLHF caveat, but sharper here since the whole thesis is preference-vector fidelity).
7. **K=2 baselines weak.** RiC especially so (HV 0.46 on helpful 2-obj, MIP 0.07); RS parameter-merging is structurally limited (paper §4.2 L785: "linear combination of parameters constrains the model's ability to capture nonlinear representations"). The real contest is MI-EPO vs MO-ODPO (1 baseline).
8. **CRD = det(Cov) is volume, not per-axis spread.** A determinant can shrink while one reward axis widens (if covariance anti-correlates); "smaller CRD ⇒ more stable conditional control" is true in aggregate but masks axis-specific behaviour. Eq 16/17.
9. **Dirichlet α choice under-specified in effect.** α=0.5 (safety) samples spikier w (more single-objective emphasis) than α=1.0 (uniform). The preference-vector grid used for eval is fixed (App C), but the *training* w-distribution differs across tasks — so cross-task HV/MIP numbers are not directly comparable. Plus β_c=0.1 ≫ β_w=0.01: the L_YC alignment term is weighted 10× the L_YW exploration term in logit-space, so the "unified" objective is in practice alignment-dominated (the exploration term is a regularizer).
10. **CMI decomposition assumes {C_k} conditionally independent given (X,Y)** (§3.1, "standard multi-objective evaluation setting"). Real preference dimensions (helpfulness vs harmlessness) are *negatively correlated* — violating independence. The chain-rule identity Eq 6 still holds, but the interpretation of Σ_k w_k I(Y;C_k) as a clean per-objective sum leans on this assumption.

---

## 8. Verdict

MI-EPO is a **cleanly-posed information-theoretic reframe of multi-objective online alignment**: the joint CMI I(Y; C_Z, W, Z | X) decomposes (Eq 6, chain rule) into a per-objective DPO alignment term (Eq 8) + a preference-vector contrastive exploration term (Eq 12), combined in one loss (Eq 13) with a stop-gradient on the anchor. The β_w=0 ⟺ MO-ODPO reduction is the falsifiable hinge. Empirically MI-EPO beats MO-ODPO on every numeric cell (Tables 1, 2a). **But:** (a) the headline safety-HV gain is mis-stated in prose (68.8% vs the cells' 75.3%); (b) the strongest scaling claim (3-obj +87.2% HV) lives in a subtable whose cells were lost to layout extraction and are unverifiable; (c) no seeds/CIs; (d) the central overlap-reduction evidence is figure-only. Treat Table 1 / Table 2a as the reliable surface; quote the 3-obj and distribution-separation claims as prose-only. Sibling to the reward-design lineage but uniquely **information-theoretic** (max-CMI) rather than reward-shaping or reward-model-architectural.
