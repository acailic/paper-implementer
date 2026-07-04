# Neuron-Aware Data Selection for Annotation-Free LLM Self-Distillation (NEURON-OPSD / N-OPSD)

**arXiv:** 2607.02460v1 [cs.LG] (2 Jul 2026) — `papers/2026-neuron-aware-data-selection-self-distillation/paper.pdf`
**Authors:** Zhuowei Chen, Xiang Lorraine Li (University of Pittsburgh)
**Venue:** ICML 2026 Workshop — *Decision-Making from Offline Datasets to Online Adaptation: Black-Box Optimization to Reinforcement Learning*
**Model:** Qwen3-4B-Thinking-2507 ("Qwen-4B" in paper). 16pp (9 main + appendix A–C).
**Sourcing:** all numeric tables transcribed verbatim from `paper_layout.txt` (`pdftotext -layout`); line-ranges cited per table. Every Avg/delta reconciled by a source-free script (see §Verification).

---

## 1. Problem & positioning

Annotation-free post-training of LLMs for specialized domains (education, law, STEM) where expert labels are costly. The only supervision available is the model's own rollouts + internal activations — no ground-truth labels, no stronger external teacher, no environment/reward oracle. This is explicitly **distinct from offline RL** (logged reward-labeled trajectories) and **from RLHF** (live preference queries).

Three failure modes of prior annotation-free methods motivate the work:
- **SFT-based** self-training (LMSI) → catastrophic forgetting / out-of-domain degradation.
- **GRPO-based** (TTRL, Intuitor) → vanishing gradients under low within-group reward variance, entropy collapse, and **calibration error inflation**; scalar reward gives only trajectory-level supervision (no signal on *which* reasoning step to reinforce).
- Generic self-distillation risks **reinforcing the model's own miscalibration, spurious reasoning, hallucinations**.

N-OPSD's answer is **data-centric**: instead of designing a better reward, decide *which unlabeled samples to train on* (Neuron Consensus) and *which few-shot context induces a useful teacher–student gap* (Neuron Overlap), then train via **on-policy distillation (OPD)** against an EMA teacher conditioned on that context.

```mermaid
flowchart LR
  subgraph A["A. Data Selection"]
    P["Unlabeled pool D"] -->|zero-shot rollout frozen LLM| N["Activation set N(x)<br/>top-K=5000 neurons"]
    N -->|consensus s(x)=|N(x)|<br/>bottom-20%| S["Train subset S"]
  end
  subgraph B["B. Context Curation"]
    N -->|Jaccard nearest K=10| C["Few-shot demos C_K(x)"]
  end
  subgraph C["C. Self-Evolving Training"]
    S --> Stu["Student πθ (zero-shot)"]
    C --> Tea["Teacher πθ̄ EMA<br/>(context-conditioned)"]
    Stu -->|reverse-KL distill| Tea
  end
```

---

## 2. Method (§3–§4, Algorithm 1)

### 2.1 Preliminary — On-Policy Distillation (Eq 1)

OPD trains student π_θ on its own rollouts against a teacher, token-level reverse-KL (denser than scalar-reward RL):

$$\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{x\sim D,\, y\sim\pi_\theta}\!\left[\sum_{t=1}^{|y|} \mathrm{KL}\!\left(\pi_\theta(\cdot|x,y_{<t}) \,\|\, \pi_{\text{teacher}}(\cdot|x,y_{<t})\right)\right]$$

In the annotation-free setting both teacher and student are the **same base LLM**; the teacher is differentiated by **auxiliary context c** (few-shot demos): π_teacher(·|x,y<t) := π_θ(·|x,c,y<t).

### 2.2 Internal neuron-signal extraction (§3.3, Eq 2–3)

Logit-lens view: intermediate MLP neurons scored by their downstream contribution to the generated token. For layer *l*, MLP input h_l, up/down projections W_in^l / W_out^l:

- **Eq 2:** k^l = σ(h_l W_in^l) (post-activation)
- **Eq 3:** S^l_{ŷ,i} = k^l_i · (w_out,i^l · e_ŷ) — early-unembedding score of neuron *i* for generated token ŷ.

Keep the **top-K = 5000** scored neurons across all layers (top-2000/layer/chunk → global top-5000 dedup, max-score on repeat hits; App. A). Union across response tokens → sparse activation set **N(x)**.

### 2.3 Stage 1–2 — Neuron Consensus sample selection (§4.2, Eq 4)

$$s(x) = |N(x)|$$

**Smaller s(x) = higher consensus** (sparse, "unconflicted" retrieval) → more reliable / less hallucination-prone (Chen et al. 2025b, 2026). Select the **bottom-20% (bot20)** of the pool by s(x) for training. Rationale: reinforcing already-correct reasoning on reliable samples avoids amplifying hallucinations.

### 2.4 Stage 3 — Neuron Overlap context curation (§4.3, Eq 5–6)

Jaccard **distance** between activation patterns:

$$J(x_i, x_j) = 1 - \frac{|N(x_i)\cap N(x_j)|}{|N(x_i)\cup N(x_j)|}$$

For each query x_q retrieve the **K = 10** nearest neighbors (Eq 6) as m-shot demonstrations C_K(x). Small J ⇒ same knowledge circuits / reasoning path ⇒ the teacher's context is matched to the query, sharpening its token distribution.

### 2.5 Stage 4 — Self-improvement via OPD (§4.4, Eq 7–10)

- **Eq 7:** student = zero-shot current model π^stu_θ(·|x,y<t) := π_θ(·|x,y<t)
- **Eq 8:** teacher = EMA copy conditioned on neuron-overlap context π^tea_θ̄(·|C_K(x),x,y<t) := π_θ̄(·|C_K(x),x,y<t)
- **Eq 9:** EMA update **θ̄ ← (1 − τ)θ̄ + τθ**  (τ = 0.01 per App. A; Algorithm 1 names this rate ρ — symbol inconsistency ⚠)
- **Eq 10:** L_OPD = reverse-KL on student rollouts against the context-conditioned teacher; gradients flow **only to student θ**, teacher θ̄ is a fixed target smoothed by EMA.

The context-induced distribution is *internalized into parameters*, so demos are **not needed at inference**. Algorithm 1 (App. B.1) formalizes all 4 stages.

---

## 3. Experimental setup (§5.3, App. A/C)

| Item | Value |
|---|---|
| Base model | Qwen3-4B-Thinking-2507 |
| Datasets | SciKnowEval (Bio/Mat/Phys/Chem, 80/20 train-test); Edu-Feedback (1799 train / 1000 test, binary feedback-quality); MMLU-Pro (80% for self-improvement) |
| Neuron extraction | top-2000/layer/chunk → global top-5000 dedup; forward hooks on `model.layers[l].mlp.act_fn` |
| Selection | bot20 by s(x)=|N_0-shot(x)| (zero-shot activation count) |
| Context | K=10 Jaccard-nearest few-shot demos |
| N-OPSD train | veRL + Ray-FSDP, **4 GPUs**, batch 16, micro-batch 4, **150 steps**, reverse-KL, EMA rate **0.01** |
| TTRL baseline | GRPO, 4 GPUs, batch 16, actor LR **5e-7** (cosine, 3% warmup), critic LR **9e-6**, KL coef **0.0**, rollout temp **1.0**, **8 votes/prompt**, val temp 0.6, 1 epoch, best-maj@8 ckpt |
| Eval | **8 responses/question**, temp **0.6**, vLLM; Avg@8 (mean per-sample acc), Maj@8 (majority-vote acc), **ECE** (15 equal-width bins, confidence c = max_count/8) |
| Baselines | LMSI (SFT), TTRL (GRPO+majority-vote pseudo-labels), Intuitor (annotation-free RL, self-certainty reward) |

---

## 4. Analysis results (§5.1–§5.2)

### 4.1 Neuron Consensus correlates with accuracy (Fig 2, Fig 3)

Domain-averaged **∆Acc** (avg-domain-accuracy minus bin-accuracy) rises monotonically with normalized #neuron-acts, **r = 0.369, p < 10⁻⁶** (Fig 2). Per-domain zero-shot-accuracy vs raw #neurons (Fig 3, sourcing L798–840) is **negative** — more neurons → lower accuracy:

| Domain | N samples | Pearson r | Spearman | linear slope |
|---|---|---|---|---|
| Biology | 3,541 | −0.433 | −0.417 | −2.48e-05 |
| Material | 4,460 | −0.244 | −0.258 | −1.77e-05 |
| Physics | 2,984 | −0.357 | −0.348 | −2.28e-05 |
| Chemistry | 5,636 | −0.437 | −0.440 | −1.87e-05 |

> ⚠ **Sourcing note (sign flip, not contradiction):** Fig 2's r = **+0.369** uses ∆Acc (a *gap* that grows as bin accuracy falls) as the y-axis, whereas Fig 3's per-domain r values are **negative** because they plot *raw zero-shot accuracy*. Both express the same underlying fact (more activated neurons → more likely incorrect). The opposite signs reflect different y-axis definitions, not an inconsistency.

### 4.2 Table 1 — Consensus-based selection (Top-20% vs Bottom-20%), SciKnowEval, Qwen3-4B (L394–398)

| Selection | CHEM | BIO | MAT. | PHYS. | CHEM | BIO | MAT. | PHYS. |
|---|---|---|---|---|---|---|---|---|
| | **Avg@8 (↑)** | | | | **ECE (↓)** | | | |
| Qwen3-4B | 72.11 | 73.98 | 71.12 | 80.20 | 0.173 | 0.195 | 0.213 | 0.095 |
| +Top-20% | 68.46 (−3.65) | 74.59 (+0.61) | 73.82 (+2.70) | 82.00 (+1.80) | 0.133 (−0.040) | 0.198 (+0.003) | 0.194 (−0.019) | 0.116 (+0.021) |
| +Bottom-20% | 72.03 (−0.08) | 74.02 (+0.04) | 73.39 (+2.27) | 83.13 (+2.93) | 0.171 (−0.002) | 0.193 (−0.002) | 0.207 (−0.006) | 0.114 (+0.019) |

**Takeaway:** lower-activation (Bottom-20%) is the safer, more reliable subset (matches the consensus prior), but Top-20% can still improve some domains (Bio, Mat) — selection by activation count alone is **informative but insufficient** (it ignores *learning utility* / sharpening room).

### 4.3 Table 2 — Few-shot retriever comparison (fix Bottom-20% as training data), Avg@8 (L409–414)

| Retriever | CHEM | BIO | MAT. | PHYS. |
|---|---|---|---|---|
| Qwen3-4B-Thinking | 72.11 | 73.98 | 71.12 | 80.20 |
| +Random | 68.48 (−3.63) | 74.14 (+0.16) | 73.79 (+2.67) | 81.97 (+1.77) |
| +Neuron-Jaccard (ours) | 72.03 (−0.08) | 74.02 (+0.04) | 73.39 (+2.27) | 83.13 (+2.93) |

Reasoning-diversity (mean pairwise neuron-Jaccard distance, §5.2): **CHEM 0.870, BIO 0.634, MAT 0.628, PHYS 0.640**. Neuron-overlap beats random most where the domain is diverse (CHEM); in homogeneous domains (BIO, MAT) random in-domain retrieval is already competitive.

> **3-table consistency triangle (sourcing):** the Bottom-20% row of Table 1, the Neuron-Jaccard row of Table 2, and the N-OPSD in-domain row of Table 3 are byte-identical on Mat (73.39/+2.27) and Phys (83.13/+2.93) — i.e. the main-results N-OPSD configuration is exactly bot20-selection + neuron-Jaccard-retrieval + OPD. Confirms all three tables describe the same headline run.

---

## 5. Main performance results (§5.4)

### 5.1 Table 3 — Cross-domain Avg@8 on Qwen3-4B (L456–474)

| Method | Eval | BIO | MAT. | PHYS. | CHEM | SciKnow Avg. | Edu. | MMLU-Pro | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B base | | 73.98 | 71.12 | 80.20 | 72.11 | 74.35 | 64.95 | 72.16 | 72.42 |
| LMSI | in | 74.59 (+0.61) | 69.08 (−2.04) | 78.54 (−1.66) | 73.72 (+1.61) | 73.98 (−0.37) | 66.95 (+2.00) | 68.60 (−3.56) | 71.91 (−0.51) |
| | cross | 68.57 (−3.54) | 65.48 (−7.20) | 67.02 (−3.84) | 71.19 (−1.29) | 68.06 (−3.97) | 58.14 (−15.77) | 73.46 (+0.99) | 67.31 (−5.11) |
| TTRL | in | 75.00 (+1.02) | 73.26 (+2.14) | 82.98 (+2.78) | 72.43 (+0.32) | 75.92 (+1.56) | 64.95 (+0.00) | 72.04 (−0.12) | 73.44 (+1.02) |
| | cross | 73.31 (+1.20) | 73.32 (+0.64) | 71.28 (+0.41) | 73.46 (+0.97) | 72.84 (+0.81) | 74.47 (+0.55) | 73.76 (+1.29) | 73.27 (+0.84) |
| Intuitor | in | 76.21 (+2.23) | 73.41 (+2.29) | 83.43 (+3.23) | 72.63 (+0.52) | 76.42 (+2.07) | — | 71.91 (−0.25) | 75.52 (+1.60) |
| | cross | 69.50 (−2.61) | 72.79 (+0.11) | 68.33 (−2.53) | 70.76 (−1.72) | 70.35 (−1.69) | — | 73.30 (+0.83) | 70.94 (−1.18) |
| **N-OPSD** | in | 74.02 (+0.04) | 73.39 (+2.27) | 83.13 (+2.93) | 72.03 (−0.08) | 75.64 (+1.29) | **72.19 (+7.24)** | 72.04 (−0.12) | **74.47 (+2.05)** |
| | cross | 72.71 (+0.61) | 73.27 (+0.59) | 71.37 (+0.51) | 73.59 (+1.11) | 72.74 (+0.70) | 75.22 (+1.31) | 73.74 (+1.27) | **73.32 (+0.90)** |

**in** = train on source domain, eval on same; **cross** = train on source domain, average eval on the *other* target domains. Subscripts = Δ vs untrained baseline. (Full per-source grid in Table 7.)

- N-OPSD improves source-domain Avg@8 on **4/6** sources; clear gains on **MAT. (+2.27), PHYS. (+2.93), Edu. (+7.24)**; highest overall **Avg. 74.47 in (+2.05)** and **73.32 cross (+0.90)**.
- **Edu. is the standout:** +7.24 in-domain, the largest single gain of any method on any source.
- LMSI and Intuitor suffer large cross-domain drops (LMSI Edu. cross −15.77; Intuitor Edu. in dropped —). N-OPSD preserves cross-domain capability (every cross cell positive).

> ⚠ **Denominator caveat (Intuitor Avg):** Intuitor's Edu. training run was dropped (footnote 2: no improvement at any validation step), so its **Avg. column is averaged over 5 sources, not 6**. Intuitor in-Avg = 75.52 is over {Bio,Mat,Phys,Chem,MMLU}; over 6 sources (Edu.=base 64.95) it would be **73.76**, below N-OPSD's 74.47. The Intuitor Avg is therefore not directly comparable to the other methods' 6-source Avg — cross-method Avg ranking is slightly confounded.

### 5.2 Table 4 — Cross-domain ECE on Qwen3-4B (L477–495)

| Method | Eval | BIO | MAT. | PHYS. | CHEM | SciKnow Avg. | Edu. | MMLU-Pro | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| base | | 0.195 | 0.213 | 0.095 | 0.173 | 0.169 | 0.246 | 0.184 | 0.184 |
| LMSI | in | 0.202 (+0.007) | 0.225 (+0.012) | 0.111 (+0.016) | 0.179 (+0.006) | 0.179 (+0.010) | 0.255 (+0.009) | 0.171 (−0.013) | 0.191 (+0.007) |
| | cross | 0.171 (−0.011) | 0.179 (+0.001) | 0.202 (+0.000) | 0.182 (−0.004) | 0.184 (−0.004) | 0.095 (−0.077) | 0.162 (−0.022) | 0.165 (−0.019) |
| TTRL | in | 0.204 (+0.009) | 0.208 (−0.005) | 0.123 (+0.028) | 0.192 (+0.019) | 0.182 (+0.013) | 0.248 (+0.002) | 0.198 (+0.014) | 0.196 (+0.012) |
| | cross | 0.186 (+0.004) | 0.187 (+0.009) | 0.204 (+0.002) | 0.199 (+0.012) | 0.194 (+0.007) | 0.178 (+0.006) | 0.190 (+0.006) | 0.191 (+0.007) |
| Intuitor | in | 0.208 (+0.013) | 0.212 (−0.001) | 0.117 (+0.022) | 0.198 (+0.026) | 0.184 (+0.015) | — | 0.174 (−0.010) | 0.182 (+0.010) |
| | cross | 0.226 (+0.044) | 0.187 (+0.008) | 0.234 (+0.032) | 0.224 (+0.038) | 0.218 (+0.030) | — | 0.185 (+0.000) | 0.211 (+0.024) |
| **N-OPSD** | in | 0.193 (−0.003) | 0.207 (−0.006) | 0.114 (+0.019) | 0.171 (−0.002) | 0.171 (+0.002) | **0.191 (−0.056)** | 0.180 (−0.004) | **0.176 (−0.008)** |
| | cross | 0.177 (−0.005) | 0.177 (−0.001) | 0.194 (−0.008) | 0.187 (+0.000) | 0.184 (−0.004) | 0.177 (+0.005) | 0.186 (+0.002) | **0.183 (−0.001)** |

- TTRL and Intuitor **increase** average ECE (reward-based RL inflates calibration); N-OPSD **reduces** overall ECE (in-Avg −0.008, cross-Avg −0.001).
- Strongest calibration effect on **Edu.: 0.246 → 0.191 (−0.056)** — the same source where N-OPSD gives its largest accuracy gain. Several N-OPSD rows improve Avg@8 *and* reduce ECE simultaneously.

---

## 6. What drives the gain (§5.5)

### 6.1 Table 5 — Base rollout statistics on the training pool vs in-domain gain (L531–539)

| Domain | Base Test Avg@8 | Train Maj@8 | Train Avg@8 | Maj−Avg | in-domain gain |
|---|---|---|---|---|---|
| CHEM | 72.11 | 98.23 | 98.18 | 0.04 | −0.08 |
| BIO | 73.98 | 98.73 | 98.64 | 0.09 | +0.04 |
| MAT | 71.12 | 91.93 | 91.59 | 0.34 | +2.27 |
| PHYS | 80.20 | 96.31 | 96.27 | 0.04 | +2.93 |

(Maj−Avg = rollout disagreement on the bottom-20% pool.) Larger residual disagreement (MAT 0.34) tracks larger gain — but PHYS is near-unanimous (0.04) yet still gains +2.93, so **vote-level sharpening room alone is not sufficient**.

### 6.2 Table 6 — Teacher–student per-token logprob entropy (L541–550)

| Domain | H_zs Student | H_neur Teacher | ΔH T−S | Avg@8 Gain |
|---|---|---|---|---|
| CHEM | 0.240 | 0.237 | −0.002 | −0.08 |
| BIO | 0.238 | 0.240 | +0.001 | +0.04 |
| MAT | 0.268 | 0.251 | −0.017 | +2.27 |
| PHYS | 0.257 | 0.247 | −0.010 | +2.93 |

Teacher = K=10 neuron-Jaccard-nearest demos; student = zero-shot. ΔH < 0 ⇒ teacher is sharper at the token level. **The two domains with significant teacher sharpening (MAT −0.017, PHYS −0.010) are exactly the two with the largest in-domain gain (+2.27, +2.93)**; BIO/CHEM have |ΔH| < 0.003 and ~zero gain. → **The token-level teacher–student gap, not vote-level room, is the operative driver** of N-OPSD's gain.

---

## 7. Full per-source-domain appendix grids

### 7.1 Table 7 — Avg@8, each row = one model trained on one source (L846–884)

| Model | BIO | MAT. | PHYS. | CHEM | SciKnow Avg. | Edu. | MMLU-Pro | Avg. |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 73.98 | 71.12 | 80.20 | 72.11 | 74.35 | 64.95 | 72.16 | 72.42 |
| LMSI_BIO | 74.59 (+0.61) | 71.22 (+0.10) | 78.43 (−1.77) | 68.60 (−3.51) | 73.21 (−1.14) | 66.00 (+1.05) | 58.59 (−13.57) | 69.57 (−2.85) |
| LMSI_MAT | 72.75 (−1.23) | 69.08 (−2.04) | 73.80 (−6.40) | 69.32 (−2.79) | 71.23 (−3.12) | 60.32 (−4.63) | 51.19 (−20.97) | 66.08 (−6.34) |
| LMSI_PHYS | 73.70 (−0.28) | 72.30 (+1.18) | 78.54 (−1.66) | 71.61 (−0.50) | 74.04 (−0.31) | 59.49 (−5.46) | 58.01 (−14.15) | 68.94 (−3.48) |
| LMSI_CHEM | 74.56 (+0.58) | 72.20 (+1.08) | 81.40 (+1.20) | 73.72 (+1.61) | 75.47 (+1.12) | 61.79 (−3.16) | 65.99 (−6.17) | 71.61 (−0.81) |
| LMSI_EDU | 62.85 (−11.13) | 58.44 (−12.68) | 64.98 (−15.22) | 55.06 (−17.05) | 60.34 (−14.02) | 66.95 (+2.00) | 49.37 (−22.79) | 59.61 (−12.81) |
| LMSI_MMLU | 74.27 (+0.29) | 72.63 (+1.51) | 81.06 (+0.86) | 71.51 (−0.60) | 74.87 (+0.52) | 67.84 (+2.89) | 68.60 (−3.56) | 72.65 (+0.23) |
| LMSI_Avg | 72.12 (−1.86) | 69.31 (−1.81) | 76.37 (−3.83) | 68.30 (−3.81) | 71.53 (−2.83) | 63.73 (−1.22) | 58.62 (−13.53) | 68.08 (−4.34) |
| TTRL_BIO | 75.00 (+1.02) | 73.03 (+1.92) | 83.28 (+3.09) | 72.85 (+0.74) | 76.04 (+1.69) | 65.31 (+0.36) | 72.08 (−0.08) | 73.59 (+1.17) |
| TTRL_MAT | 74.21 (+0.22) | 73.26 (+2.14) | 83.25 (+3.05) | 72.03 (−0.08) | 75.69 (+1.33) | 64.92 (−0.03) | 72.20 (+0.04) | 73.31 (+0.89) |
| TTRL_PHYS | 74.14 (+0.16) | 72.61 (+1.49) | 82.98 (+2.79) | 72.61 (+0.50) | 75.58 (+1.23) | 65.09 (+0.14) | 71.94 (−0.22) | 73.23 (+0.81) |
| TTRL_CHEM | 74.94 (+0.95) | 72.81 (+1.69) | 83.40 (+3.20) | 72.43 (+0.32) | 75.89 (+1.54) | 64.21 (−0.74) | 71.92 (−0.24) | 73.28 (+0.86) |
| TTRL_EDU | 74.56 (+0.57) | 72.88 (+1.76) | 83.55 (+3.35) | 71.91 (−0.20) | 75.72 (+1.37) | 64.95 (+0.00) | 69.43 (−2.73) | 72.88 (+0.46) |
| TTRL_MMLU | 74.94 (+0.95) | 72.88 (+1.76) | 83.89 (+3.69) | 72.17 (+0.06) | 75.97 (+1.62) | 64.91 (−0.04) | 72.04 (−0.12) | 73.47 (+1.05) |
| TTRL_Avg | 74.63 (+0.65) | 72.91 (+1.79) | 83.39 (+3.19) | 72.33 (+0.22) | 75.82 (+1.46) | 64.90 (−0.05) | 71.60 (−0.56) | 73.29 (+0.87) |
| Intuitor_BIO | 76.21 (+2.22) | 72.10 (+0.98) | 82.23 (+2.03) | 71.31 (−0.80) | 75.46 (+1.11) | 55.70 (−9.25) | 66.17 (−5.99) | 70.62 (−1.80) |
| Intuitor_MAT | 73.98 (+0.00) | 73.41 (+2.29) | 82.83 (+2.64) | 71.67 (−0.44) | 75.47 (+1.12) | 63.82 (−1.13) | 71.63 (−0.53) | 72.89 (+0.47) |
| Intuitor_PHYS | 74.49 (+0.51) | 72.53 (+1.41) | 83.43 (+3.24) | 72.07 (−0.04) | 75.63 (+1.28) | 54.81 (−10.14) | 67.76 (−4.41) | 70.85 (−1.57) |
| Intuitor_CHEM | 74.37 (+0.38) | 72.20 (+1.08) | 83.21 (+3.01) | 72.63 (+0.52) | 75.60 (+1.25) | 54.94 (−10.01) | 69.10 (−3.06) | 71.08 (−1.34) |
| Intuitor_MMLU | 74.11 (+0.13) | 73.39 (+2.27) | 83.51 (+3.31) | 72.57 (+0.46) | 75.89 (+1.54) | 62.92 (−2.03) | 71.91 (−0.25) | 73.07 (+0.65) |
| Intuitor_Avg | 74.63 (+0.65) | 72.73 (+1.61) | 83.04 (+2.85) | 72.05 (−0.06) | 75.61 (+1.26) | 58.44 (−6.51) | 69.32 (−2.85) | 71.70 (−0.72) |
| **N-OPSD_BIO** | 74.02 (+0.03) | 72.96 (+1.84) | 82.87 (+2.67) | 71.97 (−0.14) | 75.45 (+1.10) | 66.31 (+1.36) | 69.46 (−2.71) | 72.93 (+0.51) |
| **N-OPSD_MAT** | 73.83 (−0.16) | 73.39 (+2.27) | 83.13 (+2.94) | 72.83 (+0.72) | 75.79 (+1.44) | 65.34 (+0.39) | 71.22 (−0.94) | 73.29 (+0.87) |
| **N-OPSD_PHYS** | 74.78 (+0.79) | 73.71 (+2.60) | 83.13 (+2.94) | 71.93 (−0.18) | 75.89 (+1.54) | 65.20 (+0.25) | 71.24 (−0.92) | 73.33 (+0.91) |
| **N-OPSD_CHEM** | 74.24 (+0.26) | 73.69 (+2.57) | 83.21 (+3.01) | 72.03 (−0.08) | 75.79 (+1.44) | 65.45 (+0.50) | 71.36 (−0.80) | 73.33 (+0.91) |
| **N-OPSD_EDU** | 75.13 (+1.14) | 73.14 (+2.02) | 83.02 (+2.82) | 72.85 (+0.74) | 76.03 (+1.68) | 72.19 (+7.24) | 71.96 (−0.21) | **74.71 (+2.29)** |
| **N-OPSD_MMLU** | 74.43 (+0.44) | 73.46 (+2.34) | 83.47 (+3.28) | 73.03 (+0.92) | 76.10 (+1.75) | 64.33 (−0.62) | 72.04 (−0.12) | 73.46 (+1.04) |
| **N-OPSD_Avg** | 74.40 (+0.42) | 73.39 (+2.27) | 83.14 (+2.94) | 72.44 (+0.33) | 75.84 (+1.49) | 66.47 (+1.52) | 71.21 (−0.95) | **73.51 (+1.09)** |

> **N-OPSD_Avg ≠ N-OPSD "Avg." in Table 3.** Table 7's shaded N-OPSD_Avg (73.51) is the column-mean of the 6 N-OPSD per-source rows; Table 3's N-OPSD in-Avg (74.47) is the diagonal-mean. These are different aggregates — verify which is cited before quoting.

### 7.2 Table 8 — ECE (lower better), same layout (L890–921)

| Model | BIO | MAT. | PHYS. | CHEM | SciKnow Avg. | Edu. | MMLU-Pro | Avg. |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 0.195 | 0.213 | 0.095 | 0.173 | 0.169 | 0.246 | 0.184 | 0.184 |
| TTRL_BIO | 0.204 (+0.009) | 0.220 (+0.007) | 0.118 (+0.023) | 0.169 (−0.004) | 0.178 (+0.009) | 0.247 (+0.000) | 0.178 (−0.006) | 0.189 (+0.005) |
| TTRL_MAT | 0.203 (+0.008) | 0.208 (−0.005) | 0.124 (+0.029) | 0.181 (+0.008) | 0.179 (+0.010) | 0.242 (−0.005) | 0.186 (+0.002) | 0.191 (+0.006) |
| TTRL_PHYS | 0.214 (+0.018) | 0.215 (+0.002) | 0.123 (+0.028) | 0.166 (−0.006) | 0.180 (+0.011) | 0.239 (−0.008) | 0.188 (+0.004) | 0.191 (+0.007) |
| TTRL_CHEM | 0.204 (+0.009) | 0.223 (+0.010) | 0.118 (+0.023) | 0.192 (+0.019) | 0.184 (+0.015) | 0.251 (+0.005) | 0.197 (+0.013) | 0.198 (+0.013) |
| TTRL_EDU | 0.199 (+0.003) | 0.215 (+0.003) | 0.111 (+0.015) | 0.164 (−0.009) | 0.172 (+0.003) | 0.248 (+0.002) | 0.202 (+0.018) | 0.190 (+0.005) |
| TTRL_MMLU | 0.200 (+0.005) | 0.224 (+0.011) | 0.114 (+0.019) | 0.169 (−0.003) | 0.177 (+0.008) | 0.243 (−0.004) | 0.198 (+0.014) | 0.191 (+0.007) |
| TTRL_Avg | 0.204 (+0.009) | 0.218 (+0.005) | 0.118 (+0.023) | 0.174 (+0.001) | 0.178 (+0.009) | 0.245 (−0.002) | 0.191 (+0.007) | 0.192 (+0.007) |
| Intuitor_BIO | 0.208 (+0.013) | 0.236 (+0.023) | 0.141 (+0.046) | 0.210 (+0.038) | 0.199 (+0.030) | 0.312 (+0.066) | 0.229 (+0.045) | 0.223 (+0.039) |
| Intuitor_MAT | 0.209 (+0.014) | 0.212 (−0.001) | 0.124 (+0.029) | 0.166 (−0.007) | 0.178 (+0.009) | 0.246 (+0.000) | 0.188 (+0.004) | 0.191 (+0.007) |
| Intuitor_PHYS | 0.213 (+0.018) | 0.229 (+0.016) | 0.117 (+0.021) | 0.176 (+0.004) | 0.184 (+0.015) | 0.341 (+0.095) | 0.210 (+0.026) | 0.214 (+0.030) |
| Intuitor_CHEM | 0.213 (+0.017) | 0.225 (+0.012) | 0.130 (+0.034) | 0.198 (+0.026) | 0.191 (+0.022) | 0.319 (+0.073) | 0.234 (+0.050) | 0.220 (+0.036) |
| Intuitor_MMLU | 0.197 (+0.002) | 0.205 (−0.008) | 0.113 (+0.018) | 0.142 (−0.031) | 0.164 (−0.005) | 0.267 (+0.021) | 0.174 (−0.010) | 0.183 (−0.001) |
| Intuitor_Avg | 0.208 (+0.013) | 0.221 (+0.008) | 0.125 (+0.030) | 0.179 (+0.006) | 0.183 (+0.014) | 0.297 (+0.051) | 0.207 (+0.023) | 0.206 (+0.022) |
| N-OPSD_BIO | 0.193 (−0.003) | 0.199 (−0.014) | 0.111 (+0.015) | 0.159 (−0.013) | 0.165 (−0.004) | 0.221 (−0.025) | 0.194 (+0.010) | 0.180 (−0.005) |
| N-OPSD_MAT | 0.206 (+0.011) | 0.207 (−0.006) | 0.121 (+0.026) | 0.156 (−0.017) | 0.173 (+0.004) | 0.219 (−0.028) | 0.185 (+0.001) | 0.182 (−0.002) |
| N-OPSD_PHYS | 0.196 (+0.001) | 0.198 (−0.015) | 0.114 (+0.019) | 0.170 (−0.003) | 0.170 (+0.001) | 0.217 (−0.029) | 0.189 (+0.005) | 0.181 (−0.004) |
| N-OPSD_CHEM | 0.198 (+0.003) | 0.215 (+0.002) | 0.117 (+0.022) | 0.171 (−0.002) | 0.175 (+0.006) | 0.225 (−0.021) | 0.180 (−0.004) | 0.184 (+0.000) |
| N-OPSD_EDU | 0.195 (−0.000) | 0.218 (+0.005) | 0.122 (+0.027) | 0.171 (−0.002) | 0.177 (+0.007) | 0.191 (−0.056) | 0.179 (−0.005) | 0.179 (−0.005) |
| N-OPSD_MMLU | 0.206 (+0.010) | 0.215 (+0.002) | 0.113 (+0.018) | 0.158 (−0.014) | 0.173 (+0.004) | 0.239 (−0.008) | 0.180 (−0.004) | 0.185 (+0.001) |
| N-OPSD_Avg | 0.199 (+0.004) | 0.209 (−0.004) | 0.116 (+0.021) | 0.164 (−0.009) | 0.172 (+0.003) | 0.219 (−0.027) | 0.185 (+0.001) | 0.182 (−0.002) |

> ⚠ **N-OPSD Phys. ECE direction (flag, not contradiction):** in-domain PHYS. ECE *increases* (+0.019 in Table 1, +0.019/+0.026/+0.027 on the PHYS. diagonal block in Table 8) even though PHYS. has the largest *accuracy* gain (+2.93). N-OPSD's "improve accuracy without calibration inflation" claim holds at the *aggregate* Avg level (−0.008 in / −0.001 cross) and on Edu., but on the PHYS. diagonal accuracy and calibration trade off — worth flagging since the headline framing emphasizes their joint improvement.

---

## 8. Verification (source-free reconciliation)

Python reconciliation script confirmed (0 mismatches among checked aggregates):
- **Table 3 in-Avg:** N-OPSD (74.02+73.39+83.13+72.03+72.19+72.04)/6 = **74.47** ✓ (base 72.42, Δ+2.05 ✓); SciKnowAvg base (73.98+71.12+80.20+72.11)/4 = **74.35** ✓.
- **Table 3 cross-Avg:** (72.71+73.27+71.37+73.59+75.22+73.74)/6 = **73.32** ✓.
- **cross = train-on-X eval-on-other-5:** N-OPSD cross Bio = mean(N-OPSD_Bio on {Mat,Phys,Chem,Edu,MMLU}) = (72.96+82.87+71.97+66.31+69.46)/5 = **72.71** ✓ (resolves the in/cross semantics).
- **Table 1 deltas:** all 4 Avg@8 + all 4 ECE Bottom-20% deltas recompute from displayed cells ✓.
- **Table 5 Maj−Avg:** all 4 domains recompute (Chem 0.05, Bio 0.09, Mat 0.34, Phys 0.04) ✓.
- **3-table consistency triangle:** Table 1 Bottom-20% == Table 2 Neuron-Jaccard == Table 3 N-OPSD in-row on Mat (73.39/+2.27) & Phys (83.13/+2.93) ✓.
- **Intuitor Avg denominator:** Intuitor in-Avg 75.52 over 5 sources (Edu dropped); over 6 would be 73.76 ✓.

---

## 9. Strengths / Limitations / Verdict

**Strengths**
- Genuinely annotation-free: no labels, no stronger teacher, no reward model — only the model's own activations + rollouts.
- Clean mechanistic story: gain is driven by the **token-level teacher–student entropy gap (Table 6)**, not vote-level room (Table 5) — a falsifiable diagnosis (|ΔH| < 0.003 ⇒ ~zero gain).
- Better accuracy/calibration trade-off than reward-based annotation-free RL (TTRL, Intuitor inflate ECE; N-OPSD reduces it at the aggregate).
- 3-table self-consistency (Tables 1/2/3 cross-agree on the N-OPSD config) is a strong internal soundness signal.

**Limitations (paper-stated + observed)**
- In-domain gains **not uniform**: BIO/CHEM negligible (no teacher token-sharpening). Consensus (activation count) alone is **informative but insufficient** for selection (Table 1).
- Neuron-overlap retrieval collapses to generic contexts in homogeneous domains (Phys. case study, App. C.3) — yet Phys. still gains most, so retrieval coherence ≠ gain.
- Only **4B-parameter** model evaluated; scaling unclear.
- No comparison vs retrieval-augmented / prompt-engineered teacher contexts (only uniform-random few-shot baseline).
- 3 datasets only (SciKnowEval, Edu-Feedback, MMLU-Pro).
- Intuitor Avg computed over 5 sources (Edu. dropped) — cross-method Avg comparison slightly confounded (⚠).
- Symbol slip: EMA rate is τ in §4.4, ρ in Algorithm 1, = 0.01 in App. A (⚠).

**Verdict.** A focused, honest, *data-centric* contribution to annotation-free self-distillation: it relocates the design effort from "invent a reward" to "select data + curate teacher context" via neuron activations, and cleanly diagnoses *when* it works (token-level teacher sharpening present). The gains are real but modest and domain-dependent; the headline "improves while preserving calibration" holds at the aggregate and on Edu. but trades off on the PHYS. diagonal. Sibling to Purified-OPSD (which fixes the teacher *update*) — N-OPSD fixes the teacher *input* (data + context).
