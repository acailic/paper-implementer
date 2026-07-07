# OmniOpt: Taxonomy, Geometry, and Benchmarking of Modern Optimizers

**arXiv 2607.04033** | Repo paper rank 80 | Iter 94 | cs.LG

---

## Problem & Motivation

The optimizer landscape for LLM training has exploded past 100 methods, each described in
incompatible vocabularies and evaluated under protocol-sensitive conditions. Practitioners face
three concrete problems: (1) no shared language to compare optimizers across mechanism families,
(2) no mechanism-aware evaluation protocol that maps results back to design choices, and (3) no
multi-objective selection guidance that accounts for quality, cost, stability, robustness, and
generalization jointly.

This paper builds a **three-layer framework** -- universal meta-pipeline (where does an optimizer
intervene?), LMO-driven four-axis decomposition (why does the update take a particular geometric
form?), dual-dimension taxonomy (what does it target?) -- and grounds it in a controlled benchmark
of 24 optimizers across 60M-1B LLM pretraining, 4 architectures, 256-32k context lengths, plus
CIFAR100 vision backbones.

## Key Insight / Contribution

1. Every modern optimizer is a **sparse modification of a shared five-stage meta-pipeline**; most
   make nontrivial design choices at only 1-2 stages.
2. **Norm-constrained LMOs** unify sign updates, spectral orthogonalization, Kronecker-factored
   preconditioning, and projected updates inside one four-axis coordinate system.
3. A **dual-dimension taxonomy**: Dimension A (mechanism: T1-T5 families) x Dimension B (effect:
   O1-O6 objectives) provides mechanism-aware experimental grouping.
4. **Benchmark of 24 optimizers** reveals: no universal winner; APOLLO best at 256 tokens but
   collapses at 32k (+21.87 PPL); Muon best gradient-norm stability; SOAP best cross-arch quality
   at highest cost; RMNP best quality-efficiency trade-off; Lion fastest but weakest PPL.

---

## Method

### 3.1 Universal Meta-Pipeline (5 stages, S0-S5)

```
S0: Signal Acquisition     -- FO gradient, VR gradient, curvature-augmented
S1: Scoping & Routing      -- parameter grouping by topology (matrix vs vector vs head)
S2: Gradient Transformation -- identity, NS orthogonalization, Kronecker, low-rank, sign
S3: State Evolution         -- moment EMAs, Kronecker factors, quantized states, multi-timescale
S4: Update Reconstruction  -- inverse map back to parameter space (dual of S2)
S5: Update Finalization     -- LR scaling, weight decay, clipping, trust ratio, SAM perturbation
```

**Eq 17 -- Routing function:**
rho(i) = ROUTE(theta_i)

**Eq 18 -- Gradient Transformation (S2):**
G_hat_t = T(G_t; S_{t-1}) in R^{r x s}

**Eq 19 -- State Evolution (S3):**
S_t = f(S_{t-1}, G_hat_t)

**Eq 20 -- Adam-family moment EMAs:**
v_t = beta_2 * v_{t-1} + (1 - beta_2) * G_hat_t^{odot2}
m_t = beta_1 * m_{t-1} + (1 - beta_1) * G_hat_t

**Eq 21 -- Update Reconstruction (S4):**
Delta_hat_t = R(Delta_tilde_t; S_t) in R^{m x n}

**Eq 22 -- Update Finalization (S5):**
W_{t+1}^{(l)} = W_t^{(l)} - eta_t * phi_t * C * F(Delta_hat_t^{(l)}) - eta_t * lambda * W_t^{(l)}

**Identity-mapping principle:** Most optimizers do nontrivial work in only 1-2 pipeline stages.
The pipeline is an ordered checklist for locating non-identity operations.

### 3.2 LMO-Driven Four-Axis Decomposition

**Eq 24 -- Linear Minimization Oracle:**
lmo_D(s) in arg min_{x in D} <s, x>

**Eq 26 -- Norm-constrained factorization:**
lmo_{D_rho}(s) = -rho * u^sharp(s)

**Eq 28 -- Unconstrained update-direction view:**
W_{t+1} = W_t - eta_t * Phi_t(M_t)

**Canonical norm geometries:**

**Eq 29 -- Euclidean (SGD):**
Phi(g) = g / ||g||_2

**Eq 30 -- Max-norm (Lion):**
Phi(g) = sign(g)

**Eq 31 -- Spectral-norm (Muon):**
Phi(M) = U V^T  (polar form from M = U Sigma V^T)

**Eq 32 -- Adaptive box (AdamW):**
lmo_{D_Adam}(m_t) = -rho_t * m_t / sqrt(v_t)
  where b_{t,i} = |m_{t,i}| / sqrt(v_{t,i})  defines the adaptive constraint set

**Four-axis master form:**

**Eq 33 -- Axis II (State Estimator):**
(M_t, H_t, D_t) = StateEstimator_t(g_t, State_{t-1})

**Eq 34 -- Axis III (Geometry & Precondition):**
D_t = Phi_t(M_t; H_t, D_t)

**Eq 35 -- Axis IV (Finalization):**
W_{t+1} = Finalize(W_t, D_t)

**Eq 36 -- AdamW four-axis instantiation:**
m_t = beta_1 m_{t-1} + (1-beta_1) g_t
v_t = beta_2 v_{t-1} + (1-beta_2) g_t^2
M_t = m_t,  H_t = diag(v_t)

**Eq 37 -- MARS variance-reduced gradient:**
c_t = g_t^xi + gamma_t * beta_1/(1-beta_1) * (g_t^xi - g_{t-1}^xi)

**Eq 38 -- Muon spectral preconditioning = polar form:**
Phi_t(M_t) = H_t^{-1/2} M_t = (M_t M_t^T)^{-1/2} M_t = U_t V_t^T

**Eq 39 -- GaLore subspace compression:**
M_bar_t = Q_L^T M_t Q_R

**Eq 40 -- GaLore Adam-in-subspace:**
D_bar_t = m_bar_t / sqrt(v_bar_t)

**Eq 41 -- GaLore lift-back:**
D_t = Q_L D_bar_t Q_R^T

**Eq 42 -- SOAP rotated coordinate direction:**
N_t' = M_t' / sqrt(V_t + eps),  N_t = Q_L N_t' Q_R^T

### 3.2.3 Four-Axis Instantiation Table (Table 5)

| Optimizer | Axis I: Domain | Axis II: State | Axis III (LMO) | Axis III (Precond) | Axis IV |
|-----------|---------------|---------------|----------------|--------------------|---------|
| SGDM | Rd | m_t | l2 ball | H_t=I | LR |
| Adam/AdamW | Rd | m_t, v_t | adaptive l_inf | H_t=diag(v_t) | LR+decoupled WD |
| NAdam | Rd | Nesterov m_t, v_t | adaptive l_inf | H_t=diag(v_t) | Nesterov+LR+WD |
| AdaBelief | Rd | m_t, s_t | adaptive l_inf | H_t=diag(v_t) | LR+WD |
| ADOPT | Rd | ordered/delayed m_t,v_t | adaptive l_inf | H_t=diag(v_t) | LR+WD |
| Adan | Rd | m_t + grad-diff | adaptive l_inf | H_t=diag(v_t) | LR+WD |
| AdEMAMix | Rd | short/long EMA | adaptive l_inf | H_t=diag(v_t) | LR+WD |
| MARS-AdamW | Rd | c_t, m_t, v_t | adaptive l_inf (VR) | H_t=diag(v_t) | LR+decoupled WD |
| RAdam | Rd | rectified m_t, v_t | adaptive l_inf | H_t=diag(v_t) | LR+WD |
| Prodigy | Rd | m_t, v_t + LR est d_t | adaptive l_inf | H_t=diag(v_t) | auto LR+WD |
| Muon | Rm×n | M_t | spectral (polar) | H_t=M_t M_t^T | LR+matrix routing |
| MARS-Shampoo | Rm×n(L_t,R_t) | c_t, m_t, L_t, R_t | metric ball (VR) | H_t=L_t^{1/4} otimes R_t^{1/4} | LR+damping |
| Shampoo | Rm×n(L_t,R_t) | m_t, L_t, R_t | metric ball | H_t=L_t^{1/4} otimes R_t^{1/4} | LR+damping |
| SOAP | Rm×n(Q_L,Q_R) | m_t, v_t, Q_L, Q_R | adaptive l_inf in Q_L,Q_R | diag(v_t) in Q_L,Q_R | LR+WD |
| GaLore | Rm×n(P_t) | m_bar_t, v_bar_t on P_t | projected l_inf | H_bar_t=diag(v_bar_t) | P_t^T back+LR+WD |
| Fira | Rm×n(P_t) | m_bar_t, v_bar_t + residual | projected l_inf (+res) | H_bar_t=diag(v_bar_t) | P_t^T back+res+LR |
| RMNP | Rm×n(row) | M_t | row-normalized | H_t=diag(M_t M_t^T) | LR+matrix routing |
| SignSGD | Rd | g_t | fixed l_inf | H_t=diag(|g_t|) | LR |
| Lion | Rd | m_t | fixed l_inf | H_t=diag(|m_t|) | LR+WD |
| MARS-Lion | Rd | c_t, m_t | fixed l_inf (VR) | H_t=diag(|m_t|) | LR+WD |
| AdaFactor | Rd(factored) | row/col v_t factors | adaptive l_inf | factored diag(v_t) | LR+factored update |
| CAME | Rd(factored) | factors + confidence | adaptive l_inf | factored diag(v_t) (+conf) | LR+factored update |
| Adam-mini | Rd(block) | m_t, v_t (block) | block l_inf | block-mean diag(v_t) | LR+WD |
| APOLLO | Rm×n(P_t rand) | m_bar_t, v_bar_t | projected l_inf | H_bar_t=diag(v_bar_t) | P_t^T back+LR |
| 8-bit Adam | Rd(INT8) | m_t, v_t (INT8) | adaptive l_inf | H_t=diag(v_t) in INT8 | dequant+LR+WD |
| Conda | Rm×n(P_t col) | v_t (col) | projected l_inf | col-wise diag(v_t) | P_t^T back+LR+WD |
| Sophia | Rd | m_t, h_t | clipped local | H_t=h_t | LR+WD |
| AdaHessian | Rd | m_t, h_t (Hutch.) | metric ball | H_t=h_t | radial proj+LR+WD |
| AdamP | Rd | m_t, v_t | adaptive l_inf | H_t=diag(v_t) | trust ratio+LR+WD |
| LAMB | Rd | m_t, v_t | adaptive l_inf | H_t=diag(v_t) | trust ratio+LR+WD |

### Table 4: Meta-Pipeline Instantiations

| Method (family) | Active stages | Core mechanism |
|-----------------|--------------|----------------|
| AdamW (T1.1) | S3, S5 | Moment EMAs (S3) + decoupled weight decay (S5) |
| Muon (T2.1) | S1, S2 | Matrix routing (S1) + NS spectral orthogonalization (S2) |
| GaLore (T2.3) | S1-S4 | Low-rank projection (S1/S2), subspace Adam state (S3), inverse projection (S4) |
| Lion (T3) | S2, S3 | Momentum interpolation (S3) + sign discretization (S2) |
| SAM (T5.1) | S0, S5 | Perturbation-induced gradient (S0) + neighborhood-regularized writeback (S5) |

## Dual-Dimension Taxonomy (Section 4)

### Dimension A: Methodological -- Five Families

| Family | Name | Pipeline focus | Members |
|--------|------|---------------|---------|
| T1 | Element-wise adaptive moment & scalar control | S3, S5 | AdamW, NAdam, AdaBelief, Adan, ADOPT, RAdam, AdEMAMix, MARS-AdamW, Prodigy, Schedule-Free, DoG, D-Adaptation, ... |
| T2 | Matrix-level structural | S1-S4 | Muon, SOAP, Shampoo, GaLore, Fira, RMNP, MARS-Shampoo |
| T3 | Discretization & directional quantization | S2, S3 | Lion, MARS-Lion, SignSGD |
| T4 | State compression & structural aggregation | S3, S4 | AdaFactor, 8-bit Adam, CAME, Adam-mini, APOLLO, Conda |
| T5 | Curvature-aware & geometric regularization | S0, S5 | SAM, Sophia, AdaHessian, AdamP, LAMB, Cautious optimizers |

### Dimension B: Effect Objectives (O1-O6)

| Objective | Definition | Data source | Measurement layer |
|-----------|-----------|-------------|-------------------|
| O1: Convergence efficiency | Loss reduction, time-to-target | Train/val loss | Layer 1 (single run) |
| O2: Step cost | Extra per-step computation | Timers, FLOPs | Layer 1 (single run) |
| O3: Memory | Optimizer-state + buffers | Memory profiler | Layer 1 (single run) |
| O4: Stability | Robustness to spikes/divergence | GNormCV post-warmup | Layer 2 (post-processing) |
| O5: HP robustness | LR sensitivity | Multiple LR runs | Layer 3 (cross-config) |
| O6: Generalization | Val loss, downstream, OOD | lm-eval-harness | Layer 2-3 (mixed) |

### Table 7: Family x Objective Cross-Matrix

| Family | O1 | O2 | O3 | O4 | O5 | O6 |
|--------|----|----|----|----|----|----|
| T1 Element-wise | strong** | moderate | moderate | moderate | moderate | moderate |
| T2 Matrix-struct | strong | weak** | moderate | strong^ | moderate | strong |
| T3 Discretized | weak | strong** | strong^ | moderate | moderate | weak |
| T4 State-compress | moderate | moderate | strong** | weak | weak | weak** |
| T5 Curvature | moderate | weak | moderate | strong | strong | moderate |

(** = strong prior/likely cost; ^ = conditional/protocol-sensitive)

---

## Benchmark Study (Section 6)

### Setup

- 24 optimizers, all 5 families (T1-T5)
- **Stage 1:** C4, LLaMA, seq 256. 4 scales: 60M (10k steps), 130M (20k), 350M (60k), 1B (100k)
  - No weight decay, no gradient clipping (isolates S2/S3 machinery)
  - Metric: C4 validation PPL (O1), optimizer-state Mem in GB (O3), per-step T in ms (O2)
- **Stage 2:** FineWeb-Edu, seq 32k, 340M + 1B, 4 architectures (Transformer++, Gated DeltaNet, DeltaNet, GLA)
  - Weight decay + gradient clipping enabled identically for all
  - Metrics: WikiText test PPL, CS Avg (10 lm-eval-harness tasks)

### Table 13: Stage-1 Screening on C4 (LLaMA, seq 256) -- VERBATIM

| Optimizer | Venue | 60M PPL | 60M Mem GB | 60M T ms | 130M PPL | 130M Mem GB | 130M T ms | 350M PPL | 350M Mem GB | 350M T ms | 1B PPL | 1B Mem GB | 1B T ms |
|-----------|-------|---------|-----------|---------|---------|-----------|---------|---------|-----------|---------|--------|----------|--------|
| **T1** | | | | | | | | | | | | | |
| Adan | TPAMI'24 | 30.25 | 0.433 | 2.32 | 22.84 | 1.000 | 4.72 | 17.29 | 2.742 | 12.06 | 14.35 | 9.977 | 39.67 |
| RAdam | ICML'25 | 30.12 | 0.217 | 1.53 | 23.22 | 0.500 | 3.07 | 17.34 | 1.371 | 7.64 | 14.47 | 4.989 | 23.79 |
| AdamW | ICLR'20 | 30.08 | 0.217 | 1.14 | 23.18 | 0.500 | 2.31 | 17.78 | 1.371 | 5.97 | 14.48 | 4.989 | 18.62 |
| NAdam | ICLR'19 | 33.72 | 0.217 | 3.45 | 24.51 | 0.500 | 4.93 | 17.90 | 1.371 | 9.96 | 14.67 | 4.989 | 20.91 |
| MARS-AdamW | ICML'16 | 30.01 | 0.325 | 7.62 | 22.86 | 0.750 | 11.05 | 16.95 | 2.057 | 22.12 | 14.90 | 7.483 | 34.70 |
| Prodigy | ICML'24 | 33.44 | 0.433 | 8.36 | 24.13 | 1.000 | 12.29 | 18.27 | 2.742 | 24.30 | 15.61 | 9.977 | 36.78 |
| AdaBelief | NeurIPS'20 | 30.08 | 0.433 | 5.76 | 23.45 | 1.000 | 8.55 | 17.61 | 2.742 | 19.10 | 16.79 | 9.977 | 55.48 |
| **T2** | | | | | | | | | | | | | |
| MARS-Shampoo | ICML'25 | 30.03 | 0.325 | 26.27 | 22.56 | 0.750 | 37.94 | 16.82 | 2.057 | 78.71 | 13.72 | 7.483 | 513.7 |
| Muon | arXiv'25 | 28.26 | 0.109 | 21.01 | 21.81 | 0.250 | 30.48 | 16.60 | 0.686 | 9.32 | 13.72 | 2.495 | 379.0 |
| RMNP | ICML'26 | 29.88 | 0.109 | 3.26 | 22.54 | 0.250 | 4.63 | 16.85 | 0.686 | 11.85 | 13.87 | 2.495 | 16.94 |
| SOAP | ICLR'25 | 29.47 | 0.731 | 50.58 | 22.67 | 2.214 | 110.4 | 17.14 | 7.465 | 302.5 | 14.04 | 29.299 | 1371.5 |
| GaLore | ICML'24 | 34.56 | 0.062 | 4.21 | 25.32 | 0.199 | 5.88 | 19.18 | 0.426 | 66.05 | 14.29 | 0.790 | 15.29 |
| Shampoo | ICML'18 | 30.22 | 0.217 | 22.36 | 22.56 | 0.500 | 33.27 | 17.03 | 1.371 | 389.4 | 14.29 | 4.989 | 389.4 |
| **T3** | | | | | | | | | | | | | |
| MARS-Lion | ICML'25 | 32.41 | 0.325 | 5.72 | 25.68 | 0.750 | 8.49 | 18.78 | 2.057 | 17.11 | 15.73 | 7.483 | 25.00 |
| Lion | NeurIPS'23 | 35.94 | 0.109 | 2.07 | 25.56 | 0.250 | 3.01 | 19.30 | 0.686 | 5.80 | 17.02 | 2.494 | 12.48 |
| **T4** | | | | | | | | | | | | | |
| APOLLO | MLSys'25 | 30.86 | 0.062 | 8.62 | 22.74 | 0.149 | 12.65 | 16.43 | 0.426 | 28.65 | 13.53 | 0.790 | 24.77 |
| Conda | arXiv'25 | 28.65 | 0.245 | 4.88 | 21.91 | 0.595 | 7.11 | 16.45 | 1.703 | 62.33 | 14.25 | 6.317 | 44.18 |
| 8-bit Adam | ICLR'22 | 30.46 | 0.110 | 4.11 | 23.30 | 0.254 | 7.27 | 17.67 | 0.697 | 42.38 | 14.53 | 2.534 | 20.05 |
| CAME | ACL'23 | 31.40 | 0.218 | 14.99 | 23.79 | 0.502 | 21.76 | 17.60 | 1.376 | 87.46 | 14.53 | 4.997 | 56.46 |
| AdaFactor | ICML'18 | 30.00 | 0.001 | 9.90 | 22.94 | 0.002 | 14.63 | 17.85 | 0.003 | 56.46 | 14.92 | 0.004 | 20.81 |
| Adam-mini | ICLR'25 | 30.50 | 0.109 | 5.68 | 23.62 | 0.251 | 8.31 | 18.12 | 0.686 | 20.81 | 15.51 | 2.495 | 13.90 |
| **T5** | | | | | | | | | | | | | |
| AdamP | ICLR'21 | 30.21 | 0.217 | 12.82 | 23.07 | 0.500 | 19.13 | 17.39 | 1.371 | 39.98 | 14.57 | 4.989 | 26.21 |
| LAMB | ICLR'20 | 30.03 | 0.217 | 9.14 | 23.40 | 0.500 | 13.17 | 17.25 | 1.371 | 16.68 | 16.09 | 4.989 | 44.89 |
| Sophia | ICLR'24 | 36.27 | 0.217 | 3.92 | 25.76 | 0.500 | 5.66 | 18.86 | 1.371 | 11.06 | 16.45 | 4.989 | 29.70 |

### Table 14: Stage-2 Cross-Architecture Generalization (FineWeb-Edu, 32k) -- VERBATIM

WikiText PPL (lower=better) | CS Avg % (higher=better)

| Optimizer | Tr++ 340M | Tr++ 1B | GDN 340M | GDN 1B | Delta 340M | Delta 1B | GLA 340M | GLA 1B | CS Tr++ 340M | CS Tr++ 1B | CS GDN 340M | CS GDN 1B | CS Delta 340M | CS Delta 1B | CS GLA 340M | CS GLA 1B |
|-----------|-----------|---------|---------|-------|-----------|---------|---------|-------|-------------|-----------|------------|----------|-------------|-----------|-----------|---------|
| **T1** | | | | | | | | | | | | | | | | |
| MARS-AdamW | 24.57 | 18.94 | 24.17 | 20.04 | 26.79 | 20.67 | 28.28 | 21.89 | 52.50 | 57.46 | 54.91 | 58.18 | 51.69 | 56.80 | 51.24 | 55.71 |
| AdamW | 24.62 | 18.90 | 24.47 | 20.33 | 27.16 | 20.66 | 28.67 | 22.06 | 52.28 | 56.55 | 53.67 | 57.01 | 51.74 | 55.56 | 51.06 | 56.69 |
| Adan | 25.55 | 19.41 | 24.78 | 20.55 | 27.28 | 20.88 | 29.00 | 22.51 | 52.48 | 57.21 | 52.83 | 57.93 | 51.78 | 56.50 | 51.01 | 55.07 |
| **T2** | | | | | | | | | | | | | | | | |
| MARS-Shampoo | 23.90 | 18.72 | 23.85 | 19.86 | 26.02 | 20.38 | 27.04 | 20.62 | 53.75 | 57.71 | 54.77 | 57.22 | 52.60 | 56.49 | 52.21 | 57.57 |
| SOAP | 24.37 | 19.40 | 23.65 | 20.26 | 26.80 | 21.06 | 28.60 | 22.23 | 53.35 | 57.12 | 54.45 | 57.30 | 53.25 | 57.32 | 50.72 | 55.79 |
| RMNP | 25.05 | 19.86 | 24.34 | 20.32 | 27.18 | 21.18 | 27.47 | 21.54 | 53.25 | 56.36 | 54.45 | 57.20 | 52.00 | 56.65 | 52.50 | 56.85 |
| Muon | 26.43 | 19.74 | 25.99 | 24.87 | 28.26 | 21.25 | 29.20 | 21.53 | 51.96 | 57.30 | 53.37 | 57.58 | 51.43 | 56.74 | 52.01 | 57.33 |
| **T3** | | | | | | | | | | | | | | | | |
| MARS-Lion | 26.02 | 20.26 | 24.76 | 20.38 | 28.20 | 21.44 | 29.47 | 22.40 | 51.07 | 55.22 | 53.24 | 55.74 | 49.96 | 54.22 | 50.14 | 53.96 |
| Lion | 26.20 | 21.17 | 25.24 | 22.20 | 28.25 | 22.72 | 29.67 | 23.79 | 51.61 | 54.51 | 52.96 | 55.50 | 50.94 | 53.65 | 50.69 | 53.91 |
| **T4** | | | | | | | | | | | | | | | | |
| Conda | 28.30 | 19.86 | 26.11 | 21.07 | 29.09 | 21.75 | 37.38 | 22.89 | 51.61 | 57.24 | 53.45 | 57.18 | 51.46 | 56.10 | 48.28 | 54.95 |
| APOLLO | 34.08 | 25.29 | 30.36 | 29.29 | 34.73 | 25.58 | 37.75 | 27.78 | 48.19 | 53.61 | 50.92 | 53.73 | 49.04 | 53.88 | 48.38 | 52.33 |
| **T5** | | | | | | | | | | | | | | | | |
| AdamP | 24.68 | 19.04 | 24.32 | 20.29 | 26.77 | 20.68 | 28.66 | 21.86 | 51.69 | 56.82 | 53.82 | 57.07 | 51.53 | 56.73 | 51.14 | 55.31 |

### Table 15: Sequence-Length Effect -- VERBATIM

| Optimizer | 256 PPL | 32k PPL | Delta |
|-----------|---------|---------|-------|
| APOLLO | 13.53 | 35.40 | +21.87 |
| Muon | 13.72 | 22.54 | +8.81 |
| SOAP | 14.04 | 21.62 | +7.58 |
| AdamW | 14.48 | 21.87 | +7.39 |
| Lion | 17.02 | 23.31 | +6.29 |

### Table 16: CIFAR100 Top-1 Accuracy (%) -- VERBATIM

| Optimizer | Family | ResNet50 | DeiT-S | CAFormer-S12 |
|-----------|--------|---------|--------|-------------|
| AdaGrad | T1 | 73.30 | 67.24 | 38.09 |
| AdaDelta | T1 | 75.07 | 65.44 | 82.08 |
| RMSProp | T1 | 74.25 | 70.71 | 81.83 |
| Adam | T1 | 74.55 | 71.04 | 82.18 |
| AdamW | T1 | 75.56 | 72.15 | 83.60 |
| Adamax | T1 | 75.21 | 73.31 | 82.50 |
| NAdam | T1 | 74.82 | 72.75 | 82.83 |
| RAdam | T1 | 75.19 | 72.41 | 82.35 |
| AdaBelief | T1 | 80.53 | 70.66 | 83.56 |
| Adan | T1 | 77.08 | 76.33 | 84.89 |
| AdaBound | T1 | 78.11 | 68.59 | 82.38 |
| NovoGrad | T1 | 79.36 | 73.13 | 82.98 |
| MARS-AdamW | T1 | 74.19 | 71.57 | 80.48 |
| RMNP | T2 | 73.39 | 71.83 | 82.44 |
| Muon | T2 | 75.25 | 77.38 | 84.43 |
| GaLore | T2 | 73.53 | 70.88 | 82.19 |
| MOGA | T2 | 63.20 | 62.48 | 79.00 |
| MARS-Lion | T3 | 71.53 | 33.70 | 77.02 |
| Lion | T3 | 75.28 | 74.57 | 79.59 |
| AdaFactor | T4 | 75.41 | 74.02 | 82.36 |
| APOLLO | T4 | 74.09 | 71.24 | 82.00 |
| CAME | T4 | 66.62 | 71.05 | 81.83 |
| Conda | T4 | 73.87 | 70.76 | 82.45 |
| LAMB | T5 | 77.19 | 75.39 | 83.74 |
| AdamP | T5 | 78.17 | 71.55 | 83.40 |
| Sophia | T5 | 75.19 | 71.47 | 82.96 |

### Table 17: Tiered Classification -- VERBATIM

| Tier | Optimizers |
|------|-----------|
| Tier I (primary candidates) | Muon, RMNP, AdamW |
| Tier II (scenario-dependent) | MARS-Lion, MARS-Shampoo, APOLLO, Conda, AdamP, MARS-AdamW, SOAP, Adan, Lion |
| Tier III (diagnostic failure cases) | RAdam, NAdam, Prodigy, AdaBelief, GaLore, Shampoo, 8-bit Adam, CAME, AdaFactor, Adam-mini, LAMB, Sophia |

### Table 18: Muon Cross-Scale/Cross-Arch Ablation -- VERBATIM

| Scenario | Standard Muon | Symmetric LR Scaling | Post-NS Nesterov | Both combined | Best config |
|----------|--------------|--------------------|--------------------|---------------|-------------|
| **Standard Transformer: gains stackable** | | | | | |
| C4-LLaMA 350M | 16.60 | 16.52 | 16.57 | 16.51 | Both combined |
| C4-LLaMA 1B | 13.72 | 13.64 | 13.64 | 13.58 | Both combined |
| **Linear attention: stacking disappears** | | | | | |
| FineWeb-Edu 32k GDN-340M | 24.26 | 24.02 | 24.12 | 24.12 | Symmetric LR Scaling |

### Figure 19: Family-Level O1-O6 Summary -- VERBATIM

| Family | PPL (1B) | T (ms) | M (GB) | Stability avg rank | Stability worst rank | Robustness | Generalization avg |
|--------|---------|--------|--------|-------------------|---------------------|------------|-------------------|
| T1 Element-wise | 14.35 | 35 | 7.48 | 1.60 | 23.0 | 3.0 | moderate |
| T2 Matrix-Struct | 13.72 | 384 | 3.74 | 0.85 | 19.6 | 3.3 | strong |
| T3 Discretized | 15.73 | 19 | 4.99 | 1.31 | 4.2 | 1.3 | weak |
| T4 State-Compress | 13.53 | 49 | 2.51 | 2.33 | 42.3 | 2.2 | poor |
| T5 Curvature | 14.57 | 44 | 4.99 | 1.55 | 14.7 | 2.7 | limited |

### Learning-Rate Perturbation Robustness (O5, Figure 18) -- VERBATIM

| Optimizer | s_LR (%) | Regime |
|-----------|----------|--------|
| Lion | 0.7 | Robust |
| MARS-Lion | 7.7 | Robust |
| SOAP | 12.6 | Moderate |
| Adan | 14.2 | Moderate |
| AdamP | 14.7 | Moderate |
| RMNP | 15.0 | Moderate |
| Conda | 21.3 | Moderate |
| MARS-AdamW | 23.0 | Moderate |
| Muon | 24.2 | Sensitive |
| AdamW | 27.7 | Sensitive |
| MARS-Shampoo | 36.2 | Sensitive |
| APOLLO | 63.3 | Sensitive |

---

## Key Findings

1. **No universal winner.** T1 is stable reference, T2 strongest quality at cost, T3 cheapest but
   weakest, T4 memory-efficient but rank-bounded, T5 situational only.
2. **APOLLO collapse:** best 1B PPL (13.53) at seq 256, but degrades +21.87 to 35.40 at seq 32k.
   Framework explains via Axis II: fixed low-dim projection discards more info as gradient rank rises.
3. **Muon ablation:** removing AdamW's v_t => PPL 17.78 -> 70.74 (catastrophic); adding NS orthogonalization
   => 16.86 (already beats AdamW). NS is the core; LR scaling + Nesterov are secondary refinements.
4. **Composability = locality:** mechanisms on different axes/pipeline stages stack; same-slot
   mechanisms conflict. Muon gains stack on standard Transformer but not on Gated DeltaNet.
5. **Optimizer selection is constraint-driven:** match method's dominant strength to the binding
   constraint (stability, quality, runtime, memory, or transfer).
6. **MARS-Lion anomaly:** accuracy drops to 33.70% on DeiT-S despite reasonable results elsewhere --
   aggressive approximations interact poorly with Transformer gradient geometry.

---

## Architecture Diagram (Meta-Pipeline)

```
Training Signal (S0)
    |
    v
[S1: ROUTE] -- partition params by topology
    |          (matrix vs vector vs attention-head)
    v
[S2: TRANSFORM] -- identity / NS-ortho / Kronecker / low-rank / sign
    |
    v
[S3: EVOLVE] -- moment EMAs / Kronecker factors / quantized states / multi-timescale
    |
    v
[S4: RECONSTRUCT] -- inverse map back to param space (dual of S2)
    |
    v
[S5: FINALIZE] -- LR * phi * clip * weight_decay * trust_ratio * SAM
    |
    v
Parameter Update W_{t+1}
```

---

## Honest Scope Issues

1. **Protocol sensitivity:** conclusions tied to strict controlled-variable protocol. Many improvements
   over AdamW shrink when baseline is retuned. Learning-rate robustness (O5) is a local 3-point
   diagnostic only.
2. **Scale ceiling:** benchmark covers 60M-1B only. Behavior at 7B+ is not validated.
3. **Long-context token mismatch:** seq-256 vs seq-32k comparison matched on dataset/architecture
   but NOT on token budget. Ranking of degradations is robust but absolute PPL gap is not comparable.
4. **Method coverage:** 24 of 108 surveyed optimizers benchmarked. Family-level conclusions are
   about tested instances, not the full family.
5. **Mechanism attribution is qualitative:** "APOLLO degrades because rank-bound compression"
   is plausible but not quantified. No effective-rank measurements reported.
6. **Preprint methods:** APOLLO, Conda, RMNP are from 2025-2026 preprints whose broader behavior
   is still being established.
7. **Single seed:** no mention of multiple random seeds or confidence intervals for any reported PPL.
8. **Stage 2 limited to 12 optimizers:** 12 of 24 Stage-1 optimizers transferred to Stage 2.
   The selection criteria are not fully specified.
9. **GLA instability:** GLA drives GNormCV to 10-160 for ALL optimizers (including AdamW at 113.7).
   This makes Stage-2 results on GLA less interpretable for O4 conclusions.
10. **CIFAR100 scope:** vision-backbone study is targeted, not ImageNet-scale. Cross-domain
    transfer to vision is preliminary.
