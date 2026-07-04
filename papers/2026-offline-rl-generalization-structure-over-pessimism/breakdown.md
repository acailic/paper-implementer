# Generalization in offline RL: The structure is more important than the amount of pessimism

**Paper:** "Generalization in offline RL: The structure is more important than the amount of pessimism"
**Authors:** Max Weltevrede, Matthijs T. J. Spaan, Wendelin Böhmer (Delft University of Technology)
**arXiv:** 2607.02288v1 [cs.LG], 2 Jul 2026 (31pp; accepted at the **Reinforcement Learning Conference 2026 — Finding The Frame Workshop**)
**Contact:** m.r.weltevrede@tudelft.nl
**Source-verified:** `paper.pdf` + `paper_layout.txt` (pdftotext -layout, 1774 lines). All numeric tables (1, 2, 7, 8, 9, 10) transcribed **verbatim** with sourcing line-ranges; hyper-parameter tables (3–6) transcribed verbatim; every cross-table claim recomputed by a source-free reconciliation script (see end). Figure-derived numbers are NOT back-filled (universal rule). Theorem/Counter-example statements quoted from source.

---

## TL;DR

A widespread belief in offline RL is that *over-pessimism hurts generalization*, so several methods reduce the level of conservatism as far as possible (mildly-conservative Q-learning etc.). This paper argues the opposite framing in the **zero-shot policy transfer (ZSPT)** setting: it is not *how much* pessimism an agent has, but **whether the pessimism's structure respects the optimal solution's symmetries**. In the **generalization-through-invariance ZSPT (GTI-ZSPT)** setting of Weltevrede et al. (2025) — where the agent must become invariant to a group *G* after only training on a subgroup *B ≤ G* — the paper proves two theorems and validates them empirically on a rotationally symmetric Reacher CMDP with IQL and CQL.

**Headline (all table-/theorem-confirmed):**
- **Theorem 1:** If the pessimistic value targets *Q̂\*_sym* satisfy the subgroup symmetry *B*, the argmax policy is provably **optimal in the testing contexts** with probability 1−ε, **for arbitrarily large levels of pessimism η_max** — the sufficiency condition C_Θ(ε) is *independent of η_max*.
- **Theorem 2 (+Counter-example 1):** There exist instances where a **mildly** pessimistic but **non-symmetric** target *Q̂\*_asym* generalizes **arbitrarily worse** than an **overly** pessimistic but **symmetric** one — even though both are optimal in training.
- **Empirical (Table 1):** symmetric *Q̂\*_sym* keeps test return at 0.99 even at η_max=10 (an order of magnitude beyond the max return of 1), while non-symmetric *Q̂\*_asym* collapses from 0.76 → 0.23 as η_max grows.
- **Practical lever — DAC-Output (Tables 2, 7):** a **consistency loss on the network *output*** during policy extraction beats the standard "train on the augmented dataset" recipe. DAC-Output is the strict/within-CI winner on IQL Mixed (0.96) & Suboptimal (0.85) and on CQL Expert (0.96) & Mixed (0.78).
- **Mechanism confirmation (Table 8):** applying DA to the *critic only* does **not** improve test return over no-DA — because only the *actor* is queried at test time (the critic never sees augmented obs then), so a symmetric critic is wasted.

---

## 1. Problem & Motivation (L37–84)

- Offline RL learns a policy from a fixed dataset *D = {s, a, s′, r}ⁿ*. To fight overestimation of OOD actions, methods learn a **pessimistic** value function *Q̂^π ≤ Q^π* (CQL, IQL, EDAC, BCQ).
- A line of work claims **over-pessimism hinders generalization** (Wang/Ma/Mediratta/Park et al.) and pushes for **mild** pessimism (Lyu 2022 MacQL; Mao 2024; Shimizu 2024).
- This paper: in **ZSPT to new contexts**, generalization is governed by **symmetry**, not conservatism magnitude. The **structure** of pessimism (which is induced by the **structure of dataset coverage**) matters more than its **amount**.
- If dataset-induced pessimism *contradicts* the optimal symmetry, **data augmentation (DA)** — specifically a **consistency loss** — is the tool to enforce it, because it emphasizes **symmetry over accuracy**.
- Key contrast: prior DA-for-offline-RL work augments the *dataset* then trains normally (Pinneri, Corrado, Sinha, Cho, Jang, Huang, Lee, Yang&Wang). This paper instead applies DA via a **consistency loss during policy extraction**.

---

## 2. Background (L85–217)

**CMDP / ZSPT.** A *contextual MDP* (Hallak 2015) decomposes *S = S′ × C*; a context *c* defines a task, sampled once per episode. In **ZSPT** (Kirk 2023), the agent trains on *C_train ⊂ C* and is evaluated zero-shot on *C_test ⊂ C* with *C_train ∩ C_test = ∅*. This paper uses **in-distribution** generalization (same distribution over *C*).

**Pessimism measure.** Maximal pessimism gap: `η_max = max{Q^π(s,a) − Q̂^π(s,a) | s ∈ D, a ∈ A}`.

**Q-value distillation (Eq. 1).** To *isolate* the generalization effect of a given pessimistic target from the mechanism that produced it, the theory trains a fresh net `q_θ` to predict a chosen target `Q̂^π`:

> `l_Q(θ, D_s, Q̂^π) = (1/n) Σ_{s∈D_s} (q_θ(s) − Q̂^π(s))²`  —— (1)

**Symmetry groups.** A group *G* with operation ◦; orthogonal representation `ψ_X` (`ψ⁻¹ = ψ^⊤`). *Equivariance:* `f(ψ_X(g)x) = ψ_Y(g)⁻¹ f(x)`; *invariance* is the special case `ψ_Y(g) = I`. A finite group *G* has finite size. **Full DA** = train on the augmented dataset generated by applying every *g ∈ G* to each pair.

**§2.1 GTI-ZSPT (Weltevrede et al. 2025).** Generalization = ability to become invariant to *G* having only trained on a subgroup *B ≤ G*. Formally (Definition 1), the optimal-policy on-policy state sets admit:

```
S^{π*}_{M|C}      = { ψ_S(g) s | g ∈ G, s ∈ S̄ }      (testing: full group G)
S^{π*}_{M|C_train} = { ψ_S(b) s | b ∈ B, s ∈ S̄ },  B ≤ G   (training: subgroup B)
Q*(s) = Q*(ψ_S(g) s), ∀ s ∈ S̄, ∀ g ∈ G                  (optimal Q invariant to G)
```

**Rotational Reacher example (Fig 1).** Four training contexts (shoulder at 0°/90°/180°/270°) generate the subgroup *B = C₄* of 90° rotations; the agent must become invariant to the full group *G = SO(2)* (any rotation).

---

## 3. Theoretical analysis: structure of pessimism > amount (L233–343)

### 3.1 Theorem 1 — symmetric pessimism generalizes optimally for free (L271–290)

> **Theorem 1.** Consider Q-value distillation in the GTI-ZSPT setting (Def. 2), with pessimistic targets *Q̂\*_sym* satisfying subgroup symmetry *B ≤ G* in training: `Q̂\*_sym(s) = Q̂\*_sym(ψ_S(b)s), ∀ b ∈ B, s ∈ D_s`. If the minimal gap `δ_Q := max_{a∈A_opt} Q̂\*_sym(s,a) − max_{a∈A^C_opt} Q̂\*_sym(s,a) ≥ C_Θ(ε)`, then the argmax policy `π_{q_θ}` is **optimal in the testing CMDP M|C_test** with probability **1−ε**, **for arbitrarily large levels of pessimism η_max**. The condition `C_Θ(ε)` depends on the NTK Θ (network architecture), the dataset *D_s*, the optimal *Q\**, and confidence ε — **but not on η_max**.

**Proof idea:** uses Gerken & Kessel (2024) — an infinite ensemble of infinitely-wide nets is *perfectly equivariant* to *G* when trained with full DA. The authors instead **bound the deviation from equivariance** for ensembles trained only on subgroup *B ≤ G* (Lemma 1, Appendix 8.4), and bound the single-network vs infinite-ensemble gap via Gaussian concentration (Lemma 2). Combined, the argmax policy is preserved. (Proof: Appendix 8.1.)

**δ_Q** is the minimal margin between the best pessimistic-optimal action and the best pessimistic-suboptimal action; it must be ≥ the network's approximation noise C_Θ(ε).

### 3.2 Theorem 2 — milder-but-non-symmetric can be arbitrarily worse (L292–313)

> **Theorem 2.** Consider Q-value distillation in a GTI-ZSPT setting with two pessimistic targets *Q̂₁, Q̂₂* (pessimism η₁ and η₂ < η₁), both optimal in training *M|C_train*. For certain instances there exist *Q̂₁, Q̂₂* with **η₁ ≫ η₂** where `π_{Q̂₁}` is **optimal in testing** *M|C_test* while `π_{Q̂₂}` is **suboptimal**.

**Proof:** instantiate *Q̂₁ = Q̂\*_sym* (Theorem 1, holds for arbitrarily large η_max) and *Q̂₂ = Q̂\*_asym* (Counter-example 1). Taking η₁ → ∞ satisfies η₁ ≫ η₂.

### 3.2.1 Counter-example 1 (L1051–1075, full proof L1091–1202)

A one-step rotationally-invariant GTI-ZSPT instance (Fig 2): start on a circle, 3 actions (a₁ terminates with reward *r > 0*; a₂, a₃ do nothing). Optimal `Q*(s) = [r, γr, γr]` is rotationally invariant. Choose **incorrectly equivariant** asymmetric targets that rotate the two suboptimal actions by the context angle θ around `[0, γr−η/2, γr−η/2]`:

```
Q̂*_asym(s_0)   = [r,        γr,        γr     ]
Q̂*_asym(s_90)  = [r,        γr − η,    γr     ]
Q̂*_asym(s_180) = [r,        γr − η,    γr − η ]
Q̂*_asym(s_270) = [r,        γr,        γr − η ]
```

**Key insight (L1135–1140):** at the *test* state *s₄₅* (45° rotation), the incorrectly-equivariant value becomes `ψ_Q(45)·Q̂\*_asym(s_0) = [r, 0, γr + 0.21η]` — so for large enough η, **suboptimal action a₃ outweighs optimal a₁**, the greedy policy picks it forever (suboptimal actions don't change state), and return = 0 vs optimal *r*. The sufficient condition is `η > D_{Θ,Z}(ε)`, and the optimality gap `J^Δ(π_{q_θ}) ≥ w_{s₄₅}·r → ∞` as *r → ∞*.

### 3.3 Empirical validation of the theory — Table 1 (L314–342)

**Setup (Appendix 9.1, L1436–1484).** Rotational Reacher; 2-D continuous torque action space **discretized to 9 actions** (torques evenly spaced in [−2,2]²). A DQN (Table 3) trained *only on context 1*, then its value/policy copied symmetrically to the other three contexts (ensures the approximate-optimal policy is symmetric). On-policy optimal states collected; two targets built:

- ***Q̂\*_sym:*** learned DQN Q minus a **constant** pessimism η_max from each *suboptimal* action (suboptimal = all except the greedy action). Symmetric by construction.
- ***Q̂\*_asym:*** `Q̂\*_sym` at η=0.01 **plus a 90° rotation of suboptimal actions 1 and 5** around `[Q*(s,a₁)−η_max/2−0.01, …, Q*(s,a₅)−η_max/2−0.01, …]`, then subtract 0.01 from all suboptimal actions. A baseline η_base=0.01 keeps δ_Q large enough that the finite net stays training-optimal. For η_max=0.01 this equals Q̂\*_sym; for η_max>0.01 it is equivariant-but-incorrect.

**Table 1** (verbatim, L337–342): train/test return for the distilled net `q_θ` on *Q̂\*_sym* vs *Q̂\*_asym* across η_max ∈ {0.01, 0.1, 1, 10}, 50 seeds, bold = best per row incl. overlapping 95% CI.

| target | η_max=0.01 | η_max=0.1 | η_max=1 | η_max=10 |
|---|---|---|---|---|
| **Q̂\*_sym** Train | 0.98 ± 0.07 | 1.0 ± 0.00 | 1.0 ± 0.00 | **1.0 ± 0.00** |
| **Q̂\*_sym** Test | 0.76 ± 0.11 | 0.92 ± 0.08 | 0.99 ± 0.02 | **0.99 ± 0.02** |
| **Q̂\*_asym** Train | 0.98 ± 0.07 | 1.0 ± 0.00 | 1.0 ± 0.00 | 0.51 ± 0.24 |
| **Q̂\*_asym** Test | 0.76 ± 0.11 | 0.73 ± 0.11 | 0.68 ± 0.09 | 0.23 ± 0.09 |

- **Theorem 1 confirmed:** sym test return **stays at 0.99** as η_max grows an *order of magnitude beyond the maximum return (1.0)*.
- **Theorem 2 / Counter-example 1 confirmed:** asym test return is **never higher than sym** and **monotonically collapses** (0.76 → 0.73 → 0.68 → 0.23) as η_max grows — even though asym is "less pessimistic than necessary" (it is bounded above by the symmetric construction at the same nominal η).

---

## 4. Data-augmentation experiments (L344–409)

In real offline RL the agent does *not* distill on chosen targets — it learns them from the data, so pessimism's **shape is dictated by the behavior policy / sampling process**. A suboptimal or non-symmetric behavior policy (or a mixture of policies across states) can force a **non-symmetric** value/policy — the exact failure mode of §3.2. This motivates **DA to enforce symmetry**.

**Four DA approaches** (applied to the actor for IQL; L378–399), under the *C₄* group of 90° rotations:

- **Aug-D:** minibatch `[o_t, o^aug_t]_B` — train on both original and augmented observations (standard "augmented dataset" recipe).
- **Aug-D-Online** (Almuzairee et al. 2024): augment only the actor input; the AWR value weights use original observations `[o_t, o_t]_B`.
- **DAC-Latent** (Yang et al. 2023b): train on original; add a loss minimizing the difference between the **last-hidden-layer latent** of original vs augmented.
- **DAC-Output** (Raileanu et al. 2021): train on original; add a loss minimizing the difference between the **network output** on original vs augmented.

**Why DAC-Output > DAC-Latent (L394–399):** DAC-Latent enforces symmetry only in latent space and *never trains the last linear layer on augmented observations*, leaving room for symmetry-breaking correlations to manifest in the final layer.

**Table 2** (verbatim, L406–409): IQL test return, 5 DA × {Expert, Mixed, Suboptimal} datasets (10 trajectories each from context 1; Expert = greedy DQN, Suboptimal = ε-greedy ε=0.6, Mixed = 5 greedy + 5 ε-greedy), 20 seeds, bold = best per row incl. overlapping 95% CI.

| IQL | No DA | Aug-D | Aug-D-Online | DAC-Latent | **DAC-Output** |
|---|---|---|---|---|---|
| Expert | 0.49 ± 0.10 | 0.94 ± 0.06 | **0.99 ± 0.02** | 0.95 ± 0.11 | 0.98 ± 0.02 |
| Mixed | 0.34 ± 0.10 | 0.68 ± 0.15 | 0.71 ± 0.14 | 0.67 ± 0.16 | **0.96 ± 0.07** |
| Suboptimal | 0.32 ± 0.07 | 0.61 ± 0.11 | 0.59 ± 0.11 | 0.61 ± 0.16 | **0.85 ± 0.22** |

- **DAC-Output is the most effective** — strict max on Mixed (+0.62 over No-DA) and Suboptimal (+0.53), and within overlapping CI of the leader on Expert (0.98 vs Aug-D-Online's 0.99). ⚠ honest-scope note: on Expert the strict point max is Aug-D-Online (0.99); the paper bolds DAC-Output alongside because the 95% CIs overlap.
- Standard "augmented-dataset" training (Aug-D, Aug-D-Online) **improves over no-DA but roughly equals or underperforms** the consistency loss.

---

## 5. Conclusion & limitations (L413–439)

- Structure of pessimism, not its amount, governs ZSPT generalization; **DAC** (consistency-loss DA) enforces symmetry and beats the augmented-dataset recipe for IQL and CQL on Rotational Reacher.
- **Limitations (verbatim scope):** (i) group symmetries assumed **intrinsically consistent** with train/test data — unclear if conclusions extend to *extrinsic/inconsistent* transforms (random convolutions, noise as regularization); (ii) DAC validated in **one environment, two algorithms** — broader scope needed; (iii) theory assumes the **infinite-width limit**; finite-width support is empirical only.

---

## 6. Extended results (Appendix 10, L1693–1774)

### 6.1 CQL mirror — Table 7 (L1703–1711)

Same 5-DA × 3-dataset grid as Table 2 but for CQL. Paper: "qualitatively similar results" to IQL.

| CQL | No DA | Aug-D | Aug-D-Online | DAC-Latent | **DAC-Output** |
|---|---|---|---|---|---|
| Expert | 0.54 ± 0.08 | 0.89 ± 0.11 | 0.88 ± 0.09 | 0.90 ± 0.07 | **0.96 ± 0.04** |
| Mixed | 0.36 ± 0.09 | 0.76 ± 0.18 | 0.76 ± 0.14 | 0.69 ± 0.17 | **0.78 ± 0.16** |
| Suboptimal | 0.30 ± 0.09 | **0.60 ± 0.13** | 0.57 ± 0.15 | 0.45 ± 0.12 | **0.60 ± 0.12** |

- DAC-Output is the strict/within-CI winner on Expert (0.96) and Mixed (0.78), and ties Aug-D on Suboptimal (0.60). Confirms the IQL pattern.

### 6.2 IQL DA-on-critic-only — Table 8 (L1724–1732)

`-C` = DA applied only to the **critic** (value learning), actor untouched.

| IQL | No DA | Aug-D-C | Aug-D-Online-C | DAC-Latent-C | DAC-Output-C |
|---|---|---|---|---|---|
| Expert | 0.49 ± 0.10 | 0.49 ± 0.10 | 0.49 ± 0.12 | 0.55 ± 0.10 | 0.52 ± 0.09 |
| Mixed | 0.34 ± 0.10 | 0.35 ± 0.06 | 0.34 ± 0.09 | 0.36 ± 0.09 | 0.33 ± 0.09 |
| Suboptimal | 0.32 ± 0.07 | 0.30 ± 0.09 | 0.31 ± 0.07 | 0.31 ± 0.09 | 0.32 ± 0.09 |

- **Critic-only DA ≈ no improvement** over No-DA (all deltas within ±0.06). Mechanism (L1718–1720): only the actor is used at test time, and it is extracted by evaluating the critic on **only the original dataset** — so a symmetric critic has no test-time effect. (Critic and actor use independent networks, footnote 3.)

### 6.3 IQL DA-on-both — Table 9 (L1735–1743)

`-AC` = DA on both critic and actor.

| IQL | No DA | Aug-D-AC | Aug-D-Online-AC | DAC-Latent-AC | **DAC-Output-AC** |
|---|---|---|---|---|---|
| Expert | 0.49 ± 0.10 | 0.98 ± 0.03 | **1.0 ± 0.01** | 0.97 ± 0.04 | 0.98 ± 0.02 |
| Mixed | 0.34 ± 0.10 | 0.71 ± 0.12 | 0.67 ± 0.16 | 0.71 ± 0.15 | **0.96 ± 0.09** |
| Suboptimal | 0.33 ± 0.07 | 0.62 ± 0.12 | 0.62 ± 0.11 | 0.56 ± 0.16 | **0.91 ± 0.10** |

- `-AC` ≈ `-A` (actor-only, Table 2) — adding critic-DA on top of actor-DA does **not** help further (L1720–1722), consistent with §6.2: the critic's symmetry is irrelevant to test-time return.

### 6.4 Training-context performance — Table 10 (L1754–1774)

Train-context returns across all DA variants (sanity: DA should not hurt training). 20 seeds.

| CQL train | No DA | Aug-D | Aug-D-Online | DAC-Latent | DAC-Output |
|---|---|---|---|---|---|
| Expert | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 |
| Mixed | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 0.95±0.22 | 0.95±0.22 |
| Suboptimal | 1.0±0.0 | 1.0±0.0 | 0.95±0.22 | 0.85±0.36 | 0.85±0.36 |

| IQL train (-A) | No DA | Aug-D-A | Aug-D-Online-A | DAC-Latent-A | DAC-Output-A |
|---|---|---|---|---|---|
| Expert | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 |
| Mixed | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 |
| Suboptimal | 0.75±0.43 | 0.9±0.30 | 0.95±0.22 | 0.90±0.30 | 0.80±0.40 |

| IQL train (-C) | No DA | Aug-D-C | Aug-D-Online-C | DAC-Latent-C | DAC-Output-C |
|---|---|---|---|---|---|
| Expert | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 |
| Mixed | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 |
| Suboptimal | 0.75±0.43 | 0.85±0.36 | 0.90±0.30 | 0.95±0.22 | 0.95±0.22 |

| IQL train (-AC) | No DA | Aug-D-AC | Aug-D-Online-AC | DAC-Latent-AC | DAC-Output-AC |
|---|---|---|---|---|---|
| Expert | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 | 1.0±0.0 |
| Mixed | 1.0±0.0 | 1.0±0.0 | 0.95±0.22 | 1.0±0.0 | 1.0±0.0 |
| Suboptimal | 0.75±0.43 | 1.0±0.0 | 0.80±0.40 | 0.85±0.36 | 0.95±0.22 |

- Training returns are at or near the max (1.0) everywhere except Suboptimal (where the dataset itself is ε=0.6 noisy, hence the 0.75–0.95 spread). DA does not sacrifice training optimality — the test-time gaps in Tables 2/7/9 are *generalization* gaps, not capacity gaps.

---

## 7. Hyper-parameters (verbatim, Appendix 9.1–9.2)

**Table 3 — DQN** (L1488–1511): timesteps 1,500,000; buffer 500,000; warmup 50,000; batch 512; γ=0.95; grad-clip 1; LR 5e-5; Adam; MLP ReLU [512,256,128].

**Table 4 — Value Distillation** (L1514–1525): 2000 epochs; batch 6; LR 5e-4; Adam; MLP ReLU [512,256,128].

**Table 5 — CQL** (L1618–1644): 2000 epochs; MLP ReLU [512,256,128]; target-update 100. Per dataset (Expert/Mixed/Suboptimal): batch 128/128/128; LR 1e-4/1e-4/5e-4; CQL coef 5/10/5; DAC-Latent coef 100/100/100; DAC-Output coef 100/100/10.

**Table 6 — IQL** (L1648–1689): 2000 epochs; MLP ReLU [512,256,128]; target-update 100. Per dataset: batch 64/64/16; LR 1e-3 throughout; expectile 0.8/0.7/0.8; temperature 7/7/3. Per-variant (`-C`/`-A`/`-AC`) DAC-Latent and DAC-Output coefficients searched over {1,10,100}.

**Tuning protocol (L1553–1616).** Baseline (no-DA) CQL/IQL grid-searched over LR {1e-4,5e-4,1e-3} × batch {16,64,128} × (CQL loss-coef {0.5,5,10} | IQL expectile {0.7,0.8,0.9} × temperature {3,7,10}), 5 seeds each, selected on train+validation (100 held-out validation contexts). Tuned HPs fixed for all DA experiments. DA consistency-coefficient searched over {1,10,100} (5 seeds each). Final results: 20 fresh seeds on 20 fresh datasets.

---

## 8. Strengths / Limitations / Verdict

**Strengths**
- **Clean falsifiable thesis** with two complementary theorems: structure-of-pessimism governs generalization, not amount. Theorem 1 (sym ⇒ optimal for free, condition independent of η_max) + Theorem 2/Counter-example 1 (asym ⇒ arbitrarily worse) together pin the mechanism exactly.
- **Theory → practice bridge is explicit:** the same symmetry that makes symmetric pessimism free (Theorem 1) is *enforced* by DAC-Output, which is empirically the strongest DA — so the practical recommendation (consistency loss during policy extraction) is *derived* from the theory, not a heuristic.
- **Honest ablations:** critic-only-DA no-op (Table 8) and critic+actor ≈ actor-only (Table 9) cleanly isolate that the *actor*'s symmetry is what matters at test time — a non-trivial mechanistic finding.
- **No source-free contradictions:** every cross-table claim (sym-beats-asym monotone; DAC-Output leading; -C no-op; -AC ≈ -A; η_max=0.01 sym==asym by construction) reconciles exactly (see §9).

**Limitations / honest-scope flags (⚠)**
- **⚠ η_max=0.01 sym==asym is by construction** (§9.1, L1478–1480): the asym target is defined to equal sym at η_base=0.01, then diverges for η_max>0.01. The Table-1 identity at η=0.01 (both test=0.76, both train=0.98) is therefore a design check, not an independent measurement.
- **⚠ asym Train collapses at η_max=10 (0.51 ± 0.24).** Theorem 2 assumes *Q̂₂ is optimal in training*; the realized asym target violates this at extreme η (the rotation+large subtraction pushes some rotated suboptimal value above optimal in *training* states for some seeds). The *test*-collapse trend still holds and is the paper's point, but the η=10 asym row is partly a training-failure artifact, not a pure generalization failure.
- **⚠ DAC-Output not the strict point-max on Expert** in Tables 2 (0.98 vs Aug-D-Online 0.99) and 9 (0.98 vs Aug-D-Online-AC 1.0). The paper bolds it via overlapping 95% CI; the headline "DAC-Output most effective" is carried decisively by Mixed/Suboptimal, not Expert.
- **⚠ Single environment + two algorithms.** Rotational Reacher (a 2-DoF toy with a *known, exact* C₄/SO(2) symmetry) is the ideal testbed for the theory but a weak basis for a general practical claim. The conclusion acknowledges this.
- **⚠ Infinite-width theory vs finite-width MLP [512,256,128].** The gap is bridged only empirically; Lemma 1's equivariance-deviation bound D_Θ depends on the NTK of the architecture.

**Verdict.** A focused, theoretically-grounded **negative-result-against-a-prevailing-belief** paper: it refutes "mild pessimism is better for generalization" by showing structure dominates amount, and converts the insight into a concrete, theoretically-motivated recipe (DAC-Output consistency loss during policy extraction) that beats the standard augmented-dataset practice on the one environment tested. The contribution is the **mechanism + reframe**, not a new SOTA — appropriately scoped as a workshop paper. Sibling-in-spirit to the in-context-world-modeling / VRRL lineage (symmetry/invariance as the generalization lever) but at the **value-function pessimism** level unique to offline RL.

---

## 9. Source-free reconciliation (all PASSED)

```python
# Table 1: sym test stays optimal; asym test monotonically collapses; asym never > sym
sym_test  = [0.76, 0.92, 0.99, 0.99]   # eta = 0.01,0.1,1,10
asym_test = [0.76, 0.73, 0.68, 0.23]
assert all(a <= s for a,s in zip(asym_test, sym_test))             # asym never beats sym  -> PASS
assert all(asym_test[i] >= asym_test[i+1] for i in range(3))        # asym monotonically down -> PASS
assert sym_test[-1] == 0.99                                          # optimal at eta=10 (10x max return) -> PASS
assert sym_test[0] == asym_test[0] == 0.76                           # eta=0.01 sym==asym by construction -> PASS

# Table 2 (IQL): DAC-Output strict max on Mixed & Suboptimal; within-CI on Expert
T2 = {'NoDA':[0.49,0.34,0.32],'Aug-D':[0.94,0.68,0.61],'Aug-D-On':[0.99,0.71,0.59],
      'DAC-Lat':[0.95,0.67,0.61],'DAC-Out':[0.98,0.96,0.85]}        # rows: Expert,Mixed,Suboptimal
assert max(v[1] for v in T2.values())==0.96==T2['DAC-Out'][1]       # Mixed -> PASS
assert max(v[2] for v in T2.values())==0.85==T2['DAC-Out'][2]       # Suboptimal -> PASS
assert T2['Aug-D-On'][0]==0.99 > T2['DAC-Out'][0]                   # Expert: strict max is Aug-D-On (within CI) -> flagged

# Table 7 (CQL): DAC-Output strict max Expert & Mixed, ties Suboptimal
T7 = {'NoDA':[0.54,0.36,0.30],'Aug-D':[0.89,0.76,0.60],'Aug-D-On':[0.88,0.76,0.57],
      'DAC-Lat':[0.90,0.69,0.45],'DAC-Out':[0.96,0.78,0.60]}
assert max(v[0] for v in T7.values())==0.96==T7['DAC-Out'][0]       # Expert -> PASS
assert max(v[1] for v in T7.values())==0.78==T7['DAC-Out'][1]       # Mixed -> PASS
assert max(v[2] for v in T7.values())==0.60                          # Suboptimal tied Aug-D & DAC-Out -> PASS

# Table 8 (IQL -C critic-only): no improvement over No-DA (deltas within +-0.06)
T8 = {'NoDA':[0.49,0.34,0.32],'Aug-D-C':[0.49,0.35,0.30],'Aug-D-On-C':[0.49,0.34,0.31],
      'DAC-Lat-C':[0.55,0.36,0.31],'DAC-Out-C':[0.52,0.33,0.32]}
assert all(abs(T8[k][i]-T8['NoDA'][i])<0.07 for k in T8 for i in range(3) if k!='NoDA')  # -> PASS (max delta 0.06)

# Table 9 (IQL -AC) approx equals Table 2 (-A) -> critic-DA adds nothing on top of actor-DA
T9 = {'Aug-D-AC':[0.98,0.71,0.62],'Aug-D-On-AC':[1.0,0.67,0.62],
      'DAC-Lat-AC':[0.97,0.71,0.56],'DAC-Out-AC':[0.98,0.96,0.91]}
for k in ['Aug-D','Aug-D-On','DAC-Lat','DAC-Out']:
    a=T2[k]; b=T9[k+'-AC']
    assert all(abs(a[i]-b[i])<0.08 for i in range(3))               # -AC ~ -A -> PASS
```

**Result:** 0 contradictions. Every numeric claim in the abstract / §3.3 / §4 / Appendix 10 reconciles to the verbatim tables. The 5 flagged items above (⚠) are honest-scope notes about the *paper's own* scope (single env, infinite-width assumption, construction-equality at η=0.01, training-collapse at η=10, Expert point-max tie) — NOT transcription defects.
