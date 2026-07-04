# One More Time: Revisiting Neural Quantum States from a Reinforcement Learning Perspective

**arXiv:** 2607.02292v1 [cs.LG] (2 Jul 2026) — Preprint.
**Authors:** Juan Agustín Duque¹·², Sergio García-Heredia², Vinicius Hernandes³, Eliška Greplová³, Thomas Spriggs³, Aaron Courville¹·⁴, Anna Dawid² († equal supervision). ¹Mila / U. de Montréal, ²LIACS & LION Leiden (aQa L), ³QuTech & Kavli Delft, ⁴CIFAR AI Chair.
**Code:** https://github.com/jduquevan/hyperscalenqs
**Subarea (repo lineage):** FIRST physics / quantum-many-body paper in the repo. Sibling-in-spirit to the agentic-RL lineage (imports PPO/trust-region machinery) but the optimized objective is a *variational energy* over the Born distribution, not a return — configurations are "actions", centered local energies are "advantages".

---

## TL;DR

Variational energy minimization for **autoregressive neural quantum states (NQS)** is *exactly* an advantage policy-gradient update over the Born distribution (Proposition 3.1, under a stoquastic Hamiltonian). The authors exploit this to port **PPO** to wavefunction training: **Proximal Wavefunction Optimization (PWO)** clips the probability ratio in the amplitude channel (Eq. 15) and the wrapped phase increment in the phase channel (Eq. 17), enabling on-policy sample reuse across K inner epochs without matrix inversion. PWO carries a first-order-consistency proof (Thm 4.1) + an infidelity energy bound (Thm 4.2) + a clipped improvement certificate (Cor 4.3). Empirically it is the fastest and most stable of {PWO, Adam, minSR, SPRING} on 1-D Ising, 1-D Heisenberg J1–J2, and a 2-D 10×10 frustrated lattice, and it scales to a **1.5B-parameter RWKV-7** NQS — >3 orders of magnitude beyond prior autoregressive NQS.

---

## 1. Background — NQS, VMC, and the Born distribution

A system of N spin-½ particles has configuration `s ∈ {±1}^N` and wavefunction `|ψ⟩ = Σ_s ψ(s)|s⟩` over a 2^N-dim Hilbert space. NQS parameterize the log-amplitude with a neural net, splitting modulus/phase:

> `f_θ(s) = log ψ_θ(s) = log|ψ_θ(s)| + i·arg ψ_θ(s)`   (Eq. 5)

By the **Born rule**, a normalized wavefunction induces `P_θ(s) ∝ |ψ_θ(s)|²`. **Autoregressive NQS** factorize P_θ explicitly, enabling *exact independent* sampling (no MCMC autocorrelation).

**Ground-state search** casts the lowest eigenvalue as a variational minimization:

> `E₀ ≤ E[ψ] = ⟨ψ|Ĥ|ψ⟩ / ⟨ψ|ψ⟩`, equality iff |ψ⟩ is the ground state (Eq. 6–7).

**Variational Monte Carlo (VMC)** estimates the energy and its gradient by sampling configurations from P_θ and averaging **local energies** `E^loc_θ(s) := Σ_{s'} ⟨s|Ĥ|s'⟩ ψ_θ(s')/ψ_θ(s)` (Eq. 8–10). The VMC gradient is:

> `∂_θᵢ L(θ) = E_{s∼P_θ}[ 2·Re{E^loc_θ(s) − E_{s∼P_θ}[E^loc_θ(s)]} · Oᵢ(s)* ]`, `Oᵢ(s) = ∂_θᵢ log ψ_θ` (Eq. 11, score functions).

**Stochastic Reconfiguration (SR)** preconditions this gradient with the Fubini–Study metric `S_ij = Re{Cov[O_i*, O_j]}` (Eq. 12) via `θ ← θ − η·S⁻¹·∇L` (Eq. 13). SR is geometrically principled (≈ natural gradient) but requires solving a large, often ill-conditioned linear system — the cost that motivates a first-order alternative.

---

## 2. The key correspondence — Proposition 3.1 + Table 1

> **Proposition 3.1 (Policy-gradient form of variational energy minimization).** *Assume the Hamiltonian is stoquastic (non-positive off-diagonal in the chosen basis). Then f_θ = log ψ_θ = log|ψ_θ|, and*
> `∇_θ E[ψ_θ] = E_{s∼P_θ}[ (E^loc_θ(s) − E_{s∼P_θ}[E^loc_θ(s)]) · ∇_θ log P_θ(s) ]`   (Eq. 14)

— i.e. variational energy minimization **is** an advantage policy-gradient update with **spin configurations = actions**, **centered local energies = advantages**, and the **Born distribution = policy**. This makes Table 1 the conceptual hinge of the paper.

**Table 1 — RL ↔ VMC correspondence (verbatim, layout lines 259–265):**

| Reinforcement Learning | | Variational Monte Carlo | |
|---|---|---|---|
| Policy | `π_θ(a \| s)` | Born distribution | `P_θ(s) ∝ \|ψ_θ(s)\|²` |
| Advantage | `A^{π_θ}(s_t, a_t)` | Centered local energy | `ΔE(s) = E^loc_θ(s) − E[ψ_θ]` |
| Policy gradient | `A^{π_θ} ∇_θ log π_θ` | VMC force | `2·Re[ΔE(s)·∇_θ log ψ_θ*(s)]` |
| KL trust region | `D_KL(π_{θ_old} ‖ π_θ)` | Infidelity | `I(ψ_θ, ψ_{θ+δθ})` |
| Fisher matrix | `F` | Fubini–Study metric | `S` |

> **Sourcing note.** Eq. 14 requires the **stoquastic** assumption (real amplitudes, fixed phase) so `∇ log P_θ = 2 Re ∇ log ψ_θ`. The complex-valued (frustrated) case needs the extra phase channel below — Prop 3.1 holds exactly only for the Ising model.

---

## 3. Proximal Wavefunction Optimization (PWO)

PPO applied to NQS minimizes a clipped amplitude surrogate over the *reference* Born distribution:

> `L^clip_mod(θ) = E_{s∼P_{θ_old}}[ max( r_θ(s)·A^R_{θ_old}(s), clip(r_θ(s), 1−ε, 1+ε)·A^R_{θ_old}(s) ) ]`   (Eq. 15)

where `r_θ = P_θ/P_{θ_old}` is the importance ratio, ε the amplitude clip, and `A^R_{θ_old}(s) := Re{E^loc_{θ_old}(s) − E_{s∼P_{θ_old}}[Re E^loc_{θ_old}]}` the real advantage (Eq. 16) — PPO's clip, verbatim, on wavefunction probabilities.

**Phase channel (the genuinely novel piece).** The full VMC gradient has an *imaginary* component governing phase evolution (Appendix B.3). PWO adds a second clipped surrogate that clips the **wrapped phase increment** `ϕ_θ(s)` (not a probability ratio):

> `L^clip_arg(θ) = E_{s∼P_{θ_old}}[ sg(r_θ(s)) · max( ϕ_θ(s)·A^I_{θ_old}(s), clip(ϕ_θ(s), −δ, δ)·A^I_{θ_old}(s) ) ]`   (Eq. 17)

with `sg(·)` = stop-gradient, `A^I_{θ_old}(s) := Im{E^loc_{θ_old}(s) − E_{s∼P_{θ_old}}[Im E^loc_{θ_old}]}` (Eq. 18). Two design subtleties worth citing:
- The phase objective carries **detached importance weights** `sg(r_θ)` to correct the reference/current mismatch when the batch is reused across inner epochs (Theorem B.5).
- Using the **phase increment** (not the ratio of phases) makes the combined-loss gradient match the exact VMC gradient (Eq. 11) on-policy (r_θ = 1) — this is what licenses treating PWO as a *controlled approximation* rather than a different objective.

**Algorithm 1 — PWO (verbatim, layout lines 193–214).** Inputs: Ĥ, ψ_θ, batch M, inner epochs K, amplitude clip ε, phase clip δ.
```
Initialize θ
while not converged:
    θ_old ← θ
    sample {s_i}_{i=1}^M ∼ P_{θ_old}
    cache log P_{θ_old}(s_i), arg ψ_{θ_old}(s_i), and normalized real/imag advantages
    for k = 1..K:
        r_i ← exp(log P_θ(s_i) − log P_{θ_old}(s_i))
        ϕ_i ← 2·atan2( sin(Δarg ψ), cos(Δarg ψ) )        # wrapped phase increment
        ℓ^R_i ← max( r_i·A^R, clip(r_i, 1−ε, 1+ε)·A^R )
        ℓ^I_i ← sg(r_i)·max( ϕ_i·A^I, clip(ϕ_i, −δ, δ)·A^I )
        θ ← θ − η ∇_θ ( (1/M) Σ_i [ℓ^R_i + ℓ^I_i] )
```

---

## 4. Theoretical analysis

PWO's justification parallels CPI/TRPO monotonic-improvement theory, with negative energy in place of return and infidelity in place of KL. Figure 1 gives the trust-region picture: inside the region the surrogate gradient ≈ the true VMC gradient; outside they diverge, so sample reuse is safe only while the wavefunction stays close to ψ_{θ_old}.

> **Theorem 4.1 (First-order consistency of PWO).** For differentiable |ψ_θ⟩ at θ_old with common support, for ε, δ > 0: `∇_θ[L^clip_mod(θ) + L^clip_arg(θ)]|_{θ=θ_old} = ∇_θ E[ψ_θ]|_{θ=θ_old}`. (Eq. 19, proof App. B.6)

→ The **first** inner PWO update is *exactly* the VMC gradient; the surrogate only controls how far that direction is followed.

> **Theorem 4.2 (Infidelity energy bound).** For normalized |ψ_{θ_old}⟩, |ψ_θ⟩ with common support, defining `A_{θ_old}(θ) := 2·E_{s∼P_{θ_old}}[ r_θ·cos α_θ·A^R_{θ_old} + sin α_θ·A^I_{θ_old} ]` (Eq. 20):
> `E[ψ_θ] − E[ψ_{θ_old}] ≤ A_{θ_old}(θ) + 2(Ĥ∞ − E[ψ_{θ_old}])·√(1 − √(1 − I(ψ_{θ_old}, ψ_θ)))`. (Eq. 21, proof App. B.4)

This bound is **independent of clipping** and holds for any finite update — the infidelity term is the trust-region penalty.

> **Corollary 4.3 (Clipped PWO improvement certificate).** Under bounded centered local energies and *global* constraints `r_θ(s) ∈ [1−ε, 1+ε]`, `|α_θ(s)| ≤ δ` ∀s: `E[ψ_θ] − E[ψ_{θ_old}] ≤ L^clip_mod(θ) + L^clip_arg(θ) + 2·C_{θ_old}·(1 − √(1 − ε²/2))·cos²(δ/2)`. (Eq. 22, proof App. B.7)

The final penalty vanishes as ε, δ → 0; minimizing the clipped surrogate until the RHS < 0 **guarantees** lower energy. The catch (flagged §4): the global-clip condition is *encouraged* by the method, not *guaranteed* — clipping acts per-batch, so Cor 4.3 is a certificate under a stronger requirement than the algorithm enforces.

---

## 5. Experiments

**Shared setup.** 1-D spin-½ chains, N = 12, periodic boundary, **1024 samples**, single NVIDIA L40S, 10 seeds, IQM (interquartile mean over middle 50% of seeds, per Agarwal et al. 2022 — mean±std is misleading here because unstable runs produce extreme outliers). Metrics: **relative error** `ε_rel = (E[ψ_θ] − E₀)/E₀` (exact diagonalization gives E₀) and **V-score** (scale-invariant energy-variance convergence metric, vanishes for eigenstates).

### 5.1–5.2 Spin-chain benchmarks (Figures 2, 3)

| Hamiltonian | PWO result | Best baseline |
|---|---|---|
| **Transverse-field Ising** (Eq. 23, J=1, h=1, sign-problem-free → Prop 3.1 exact) | reaches rel-err **10⁻⁷ in ~5 min** | minSR needs **~30 min** to 10⁻⁷; Adam/SPRING steady but slower |
| **Frustrated Heisenberg J1–J2 chain** (Eq. 24, J1=1, J2=0.5 → Majumdar–Ghosh point; complex-valued ansatz, phase channel active) | reaches rel-err **10⁻⁷ in ~15 min** with steadily decreasing V-score | Adam plateaus orders of magnitude above PWO (outliers); **minSR: 6/10 seeds → NaN**; minSR+SPRING stuck at **10⁻¹ after 30 min** |

The J1–J2 panel is the paper's strongest empirical evidence: a PPO-style proximal objective retains first-order wall-clock cost while delivering the stability that minSR/SPRING lose on frustrated Hamiltonians. Appendix E.1 confirms the same on the Heisenberg chain (J1=0.25, J2=0).

### 5.3 Two-dimensional frustrated J1–J2 on 10×10 square lattice (Figure 4)

Exact diagonalization intractable at 100 spins, so comparison is by **mean real energy** + V-score. Complex **patch-autoregressive transformer** (2×2 patches → 25 tokens, 2-D axial RoPE, zero-magnetization sector enforced by autoregressive masking). PWO decreases energy faster than Adam and holds a lower V-score over the same wall-clock budget.

> **Sourced energy numbers (App. D.6, layout lines 1850–1853):** PWO-trained 1.5M-param transformer reaches **−185 after 30 min**, **−195.6 after 24 h**, still decreasing. Stated SOTA ≈ **−199.0536** (Rende et al. 2024). The experiment is explicitly an optimizer comparison, not an SOTA push (no lattice symmetries imposed).

### 5.4 Scaling (Figure 5, Table 7)

Three GRU sizes on the J1–J2 chain (Table 7 params). PWO monotonically improves with size and reaches near-final energy within ~30 min; minSR unstable + non-competitive; **Adam wins on tiny but degrades — medium needs ~4× more wall-clock than PWO**. Caveat (App. C.2): PWO converges faster partly because it does *more optimization steps per unit time* (sample reuse amortizes the local-energy cost). Adam eventually reaches lower energies after a full 4 h, but the trend suggests its advantage vanishes at larger scale.

### 5.5 NQS fine-tuning of a 1.5B-parameter RWKV-7 LLM (Figure 6, Table 9)

Fine-tunes **RWKV-7 "goose" (1.5B)** as an autoregressive NQS on 1-D Ising. PWO achieves lower final relative error and V-score than Adam with stable training across the full budget. This is the scale headline: **>3 orders of magnitude beyond prior autoregressive NQS** (Rende et al. 2025). RWKV is chosen (App. C.3) because its per-token inference cost/memory is constant in sequence length (no growing KV cache), matching VMC's repeated sequential sampling.

> **Honest framing preserved (§5.5):** this is *not* a claim that LLMs are the right inductive bias for 1-D Hamiltonians — it is a demonstration that the proximal objective survives a 1000× scale-up of the ansatz.

### Per-iteration cost (Figure 7, App. C.2)

**Figure-7 bar reads** (Heisenberg J1–J2, default arch, single L40S) — sourcing caveat: per-point values are figure bar-label readings, not table cells:

| Panel (a): wall-clock per iteration (s, lower = faster) | | Panel (b): iteration speed normalized to SPRING (×, higher = faster) | |
|---|---|---|---|
| PWO | **0.022** | PWO | **12×** |
| Adam | 0.067 | Adam | 5.081× |
| minSR | 0.30 | minSR | 1.006× |
| SPRING | ≈0.30 (baseline) | SPRING | 1.000 |

PWO is fastest per-iteration despite extra bookkeeping (ratios, clipping, wrapped phase) because, in VMC, the dominant cost is sampling + local-energy evaluation, and PWO **amortizes that across K=4 inner epochs** while Adam re-pays it every update.

---

## 6. Tables (verbatim)

### Table 1 — RL ↔ VMC correspondence
See §2 above (layout lines 259–265).

### Table 2 — Shared NQS architecture (layout lines 1700–1716)
| Architecture hyperparameter | Value |
|---|---|
| Token embedding dimension | 32 |
| Site embeddings | Learned |
| Backbone input dimension | 256 |
| Backbone nonlinearity | tanh |
| Backbone normalization | Layer normalization |
| Recurrent backbone | GRU |
| Number of GRU layers | 3 |
| GRU hidden dimension | 256 |
| Amplitude head | 2-layer MLP |
| Phase head | 2-layer MLP |
| Head hidden dimension | 256 |
| Head activation | GELU |
| Phase output | π·tanh(·) |
| Phase scale | π |

> Shared across Adam / PWO / minSR / SPRING so performance gaps reflect the optimizer, not the ansatz.

### Table 3 — Ising hyperparameters (layout lines 1723–1745)
| Hyperparameter | Adam | PWO | minSR | SPRING |
|---|---|---|---|---|
| Optimizer / method | Adam | Adam + PWO | minSR | SPRING |
| Learning rate | 10⁻⁵ | 10⁻⁵ | 10⁻² | 10⁻² |
| Peak learning rate | 10⁻⁴ | 10⁻⁴ | – | – |
| Transition steps | 5,000 | 20,000 | – | – |
| PPO epochs | – | 4 | – | – |
| PPO clip ε | – | 10⁻³ | – | – |
| Advantage normalization | – | Yes | – | – |
| Phase loss | – | Δϕ clip | – | – |
| Phase coefficient | – | 1.0 | – | – |
| Phase clip | – | 0.3 | – | – |
| Center imaginary advantage | – | Yes | – | – |
| Normalize imaginary advantage | – | Yes | – | – |
| Phase Jacobian baseline | – | Yes | – | – |
| SR diagonal shift | – | – | 10⁻² | 10⁻² |
| NTK / minSR mode | – | – | Yes | Yes |
| On-the-fly SR | – | – | Yes | Yes |
| SPRING momentum | – | – | – | 0.8 |

Shared: N=12, periodic BC, J=1, h=1, complex-valued NQS, 1024 training samples, exact-diagonalization eval, eval every 200 iters; Adam/PWO use cosine one-cycle schedule.

### Table 4 — Heisenberg-chain hyperparameters (layout lines 1750–1771)
| Hyperparameter | Adam | PWO | minSR | SPRING |
|---|---|---|---|---|
| Optimizer / method | Adam | Adam + PWO | minSR | SPRING |
| LR parameter / constant LR | 10⁻⁵ | 10⁻⁵ | 10⁻³ | 10⁻³ |
| Peak learning rate | 3×10⁻⁴ | 10⁻⁴ | – | – |
| Transition steps | 10,000 | 40,000 | – | – |
| PPO epochs | – | 4 | – | – |
| PPO clip ε | – | 10⁻³ | – | – |
| Advantage normalization | – | Yes | – | – |
| Phase loss | – | Δϕ clip | – | – |
| Phase coefficient | – | 1.0 | – | – |
| Phase clip | – | 0.3 | – | – |
| Center imaginary advantage | – | Yes | – | – |
| Normalize imaginary advantage | – | Yes | – | – |
| Phase Jacobian baseline | – | Yes | – | – |
| SR diagonal shift | – | – | 10⁻² | 10⁻² |
| NTK / minSR mode | – | – | Yes | Yes |
| On-the-fly SR | – | – | Yes | Yes |
| SPRING momentum | – | – | – | 0.8 |

Shared: N=12, periodic BC, J=0.25, no sign rule, complex-valued NQS, 1024 samples, exact eval, eval every 200 iters; Adam/PWO cosine one-cycle; minSR/SPRING constant LR.

### Table 5 — Frustrated Heisenberg J1–J2 hyperparameters (layout lines 1779–1801)
| Hyperparameter | Adam | PWO | minSR | SPRING |
|---|---|---|---|---|
| Optimizer / method | Adam | Adam + PWO | minSR | SPRING |
| LR parameter / constant LR | 10⁻⁵ | 10⁻⁵ | 10⁻³ | 10⁻³ |
| Peak learning rate | 3×10⁻⁴ | 10⁻⁴ | – | – |
| Transition steps | 10,000 | 40,000 | – | – |
| PPO epochs | – | 4 | – | – |
| PPO clip ε | – | 10⁻³ | – | – |
| Advantage normalization | – | Yes | – | – |
| Phase loss | – | Δϕ clip | – | – |
| Phase coefficient | – | 1.0 | – | – |
| Phase clip | – | 0.3 | – | – |
| Center imaginary advantage | – | Yes | – | – |
| Normalize imaginary advantage | – | Yes | – | – |
| Phase Jacobian baseline | – | Yes | – | – |
| SR diagonal shift | – | – | 10⁻² | 10⁻² |
| NTK / minSR mode | – | – | Yes | Yes |
| On-the-fly SR | – | – | Yes | Yes |
| SPRING momentum | – | – | – | 0.8 |

Shared: N=12, periodic BC, J1=1, J2=0.5, no sign rule, complex-valued NQS, 1024 samples, exact eval, eval every 200 iters; Adam/PWO cosine one-cycle; minSR/SPRING constant LR.

### Table 6 — 2-D J1–J2 patch-autoregressive transformer (layout lines 1855–1881)
| Hyperparameter | Value |
|---|---|
| Lattice size | 10×10 |
| Number of spins | 100 |
| Boundary conditions | Periodic |
| Hamiltonian couplings | J1=1, J2=0.5 |
| Magnetization sector | S^z_tot = 0 |
| Patch size | 2×2 |
| Patch vocabulary size | 16 |
| Autoregressive tokens | 25 |
| Token embedding dimension | 64 |
| Transformer width | 96 |
| Transformer depth | 8 |
| Attention heads | 6 |
| Transformer MLP hidden dimension | 384 |
| RoPE type | 2-D axial RoPE |
| RoPE base | 100 |
| Amplitude head | 2-layer MLP |
| Phase head | 2-layer MLP |
| Head hidden dimension | 192 |
| Phase parameterization | π·tanh(·) |
| Phase initialization std. | 10⁻³ |
| Prefix-count features | Yes |
| Learned site embeddings | Yes |

### Table 7 — Scaling-experiment model sizes (layout lines 1891–1898)
| Size | GRU layers | RNN hidden | Head hidden | Parameters |
|---|---|---|---|---|
| Tiny | 1 | 64 | 64 | 44,452 |
| Small | 2 | 128 | 128 | 269,156 |
| Medium | 3 | 256 | 256 | 1,456,356 |

All use embedding dimension 32, evaluated on J1–J2 chain (N=12); counts include all weights+biases of the full autoregressive network.

### Table 8 — Scaling-experiment hyperparameters (layout lines 1908–1930)
| Hyperparameter | Adam | PWO | minSR |
|---|---|---|---|
| Optimizer / method | Adam | Adam + PWO | minSR |
| Learning rate | 10⁻⁵ | 10⁻⁵ | 10⁻³ |
| Peak learning rate | 3×10⁻⁴ | 10⁻⁴ | – |
| Transition steps | 40,000 | 40,000 | – |
| PPO epochs | – | 4 | – |
| PPO clip ε | – | 10⁻³ | – |
| Advantage normalization | – | Yes | – |
| Phase loss | – | Δϕ clip | – |
| Phase coefficient | – | 1.0 | – |
| Phase clip | – | 0.3 | – |
| Center imaginary advantage | – | Yes | – |
| Normalize imaginary advantage | – | Yes | – |
| Phase Jacobian baseline | – | Yes | – |
| SR diagonal shift | – | – | 10⁻² |
| NTK / minSR mode | – | – | Yes |
| On-the-fly SR | – | – | Yes |
| Training samples | 2,048 | | |
| Seeds | 10 | | |

Shared schedule/clip params identical to main J1–J2 (Tables 2, 5); only transition steps differ with model size. J1–J2 chain, N=12, J1=1, J2=0.5, periodic BC.

### Table 9 — RWKV-7 fine-tuning hyperparameters (layout lines 1935–1954)
| Hyperparameter | Adam | PWO |
|---|---|---|
| Optimizer / method | Adam | Adam + PWO |
| Model | RWKV-7 | RWKV-7 |
| Model size | 1.5B | 1.5B |
| Learning rate | 10⁻⁵ | 10⁻⁵ |
| Transition steps | 1,200 | 4,800 |
| Decay rate | 0.5 | 0.5 |
| PPO epochs | – | 4 |
| PPO clip ε | – | 10⁻³ |
| Advantage normalization | – | Yes |
| Batch size | 150 | 150 |
| Machine power | 2 | 2 |
| Evaluation samples | 4,096 | 4,096 |
| Evaluation batch size | 128 | 128 |
| Exact diagonalization | Yes | Yes |

N=12, periodic BC, J=1, h=1; samples drawn exactly from the autoregressive Born distribution. PWO's 4,800 transition steps = 4× Adam's 1,200 (one outer transition per PPO epoch — see §7 inline flag).

---

## 7. Source-free reconciliation + inline flags

**Reconciliation (all verified by script, no re-read of PDF needed):**
- J2/J1 = 0.5/1 = 0.5 ✓ (Majumdar–Ghosh point, Eq. 24 caption).
- Patch vocab 2^4 = 16 ✓ (4 spins/patch); T = (L/2)² = (10/2)² = 25 ✓ (Eq. 124).
- Table 9 PWO transition steps 4,800 = 4 × Adam 1,200 ✓ (K=4 PPO epochs, App. D.1 "scale the transition horizon by the number of PPO epochs"). Same ratio in Table 3 (5,000→20,000 = 4×) and Tables 4/5 (10,000→40,000 = 4×).
- Table 7 Medium = 1,456,356 params → "1.5M-parameter" headline (§5.1, abstract) ✓ rounds correctly.
- minSR/SPRING equivalence: SPRING = minSR + momentum 0.8 → both pay the M×M SR solve → both ~0.30 s/iter (Fig 7a) and 1.000/1.006 normalized speed (Fig 7b) ✓.
- RWKV-7 scale: 1.5B vs prior autoregressive NQS ≈ M-scale → ">3 orders of magnitude" ✓.

**⚠ Inline flags (paper-internal, transcribed verbatim not reconciled):**

1. **Shared-arch param-count mismatch.** Appendix D.2 prose states the shared 1-D architecture has **"1.4M parameters"**, while §5.1/§5.2/abstract call it **"1.5M-parameter"**. Table 7's Medium size (1,456,356) rounds to **1.5M**, and the Medium config (3 GRU layers, hidden 256) is byte-identical to Table 2's shared arch — so the shared arch *is* Medium = 1,456,356 ≈ 1.5M. The "1.4M" in D.2 is an internal rounding inconsistency; the true count is 1,456,356.
2. **Prop 3.1 scope vs §5 narrative.** Prop 3.1 (and thus the literal PPO-on-amplitude framing) requires a **stoquastic** Hamiltonian — exactly only the Ising model qualifies. The J1–J2 and 2-D results rely on the **phase channel** (Eq. 17), which is presented as a "useful practical heuristic" (§5.1) rather than something covered by Prop 3.1's exact correspondence. The headline "variational energy minimization is an advantage policy-gradient update" is exact only for Ising; the complex-valued results are the heuristic extension. Worth keeping distinct when citing.
3. **Cor 4.3 certificate vs algorithmic reality.** The improvement guarantee requires the clip to hold **globally** (`r_θ(s) ∈ [1−ε,1+ε]` ∀s), but PWO enforces clipping **per-batch**. §4 explicitly flags this gap ("encouraged by the method but not guaranteed"). So Cor 4.3 is a certificate under a stronger condition than the algorithm provably meets — not a closed-loop guarantee.
4. **Adam wins after 4 h (§5.4).** PWO's wall-clock advantage is a *convergence-speed* win, not a *final-accuracy* win: with enough budget Adam reaches lower energies on at least the tiny/small sizes. The paper states the scaling trend *suggests* PWO's advantage grows with size, but this is an extrapolation, not a measured result at >Medium scale. (This is an honest-scope note, not an inconsistency.)

---

## 8. Strengths / Limitations / Verdict

**Strengths**
- **Genuine cross-field bridge with teeth.** Prop 3.1 + Table 1 make the NQS↔RL correspondence *actionable* (it yields an optimizer with theory), not just analogical. Thm 4.1 (first-order consistency) is the load-bearing result: the first PWO step is *exactly* the VMC gradient, so PWO is a controlled approximation of a principled update, not a heuristic departure.
- **Phase channel is the novel mechanism.** Clipping the wrapped *phase increment* (Eq. 17) — not a probability ratio — is what extends PPO to complex wavefunctions while preserving on-policy gradient matching. This is the piece prior PPO-for-NQS work (Chen et al. 2022 cosine penalty) lacked.
- **Scale demonstration is real.** 1.5B-param RWKV-7 NQS is a 1000×+ jump and a non-trivial feasibility result for "bitter-lesson"-style NQS scaling, even if 1-D Ising is not where LLMs shine.
- **Honest empirics.** IQM over 10 seeds (not mean±std) is the right call given NaN-heavy minSR runs; the SOTA-energy gap on 2-D (−195.6 vs −199.05) and the Adam-wins-at-4h caveat are surfaced rather than buried.

**Limitations (paper's own, App. C.4 + this breakdown)**
- Theory is **local and conservative**: common-support + boundedness assumptions, no global-convergence guarantee, and the Cor 4.3 certificate needs a global clip the algorithm doesn't enforce.
- Benchmarks are narrow: 1-D chains, one 2-D lattice, one RWKV run. **Fermionic / electronic-structure / larger-2-D** validation is open.
- Extra PPO hyperparameters (ε, δ, K) whose optima may depend on Hamiltonian/ansatz/size/budget — a tuning surface Adam lacks.
- minSR's 6/10 NaN rate is strong evidence *for* PWO but also reflects minSR being run at a constant (not scheduled) LR; the comparison is the paper's chosen config, not an exhaustive minSR tuning.

**Verdict.** PWO is a principled, scalable first-order NQS optimizer whose central contribution — turning variational energy minimization into a trust-region policy-optimization problem with an exact first-order-consistency guarantee — is both theoretically clean and empirically the most stable option on frustrated Hamiltonians where curvature methods (SR/minSR/SPRING) numerically collapse. The most citable single result is the J1–J2 panel (PWO → 10⁻⁷ in ~15 min while minSR NaNs 6/10 seeds and SPRING/minSR stall at 10⁻¹). The RWKV-7 fine-tune is a feasibility flag for hyperscale NQS, not a physics result. Caveats: the exact RL correspondence is stoquastic-only; the complex-valued phase channel is a well-motivated heuristic; and PWO's win is convergence speed, with Adam still lower-energy at very long budgets on small models.
