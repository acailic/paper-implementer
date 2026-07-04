# Beyond Adam: SOAP and Muon for MLIPs — Source-First Breakdown

**Paper:** "Beyond Adam: SOAP and Muon for Faster, Label-Efficient Training of Machine Learning Interatomic Potentials"
**Authors:** Gil Harari*, Yoel Zimmermann*, Ola Tangen Kulseng, Laura Zichi, Chuin Wei Tan, Marc L. Descoteaux, Boris Kozinsky — John A. Paulson School of Engineering and Applied Sciences, Harvard University; Robert Bosch LLC Research and Technology Center. (*equal contribution; corresp. Kozinsky)
**arXiv:** 2607.02499v1 [cs.LG], 2 Jul 2026 — Accepted at ICML 2026 AI for Science Workshop
**Pages:** 21 (`file` and `pdfinfo` both report 21pp — no page-count defect this iter)
**Code:** Promised in future versions of open-source `nequip`/`allegro` (github.com/mir-group/nequip, github.com/mir-group/allegro); **NOT yet released** at time of writing.
**Repo role:** 55th paper (rank 50). **FIRST MLIP / materials-science / atomistic-simulation / label-efficient-force-training paper.** First paper to treat *optimiser choice* as a first-class design axis for interatomic-potential training. Sibling-in-spirit to DSGNAR (iter 67, rank 49) — both are *optimiser/method*-first contributions (DSGNAR: second-order Gauss–Newton for PINNs; this: matrix-structured first-order preconditioners for equivariant MLIPs), but this paper is an *empirical optimiser bake-off*, not a new optimiser.

---

## TL;DR

Machine-learning interatomic potentials (MLIPs) default to **Adam/AdamW** for training; the optimiser has been a blind spot while the community pushed architectures (NequIP, Allegro, MACE, UMA) and datasets. The authors plug three recently-proposed **matrix-structured optimisers** — **Muon** (Newton–Schulz orthogonalisation), **SOAP** (Shampoo eigenspace + AdamW), and the hybrid **SOAP-Muon** (SOAP + Muon orthogonalisation/RMS-norm) — into the `nequip` framework and benchmark them against AdamW on **two systems**: liquid **water** (NequIP) and the solid acid electrolyte **CsH₂PO₄ / CDP** (Allegro), across **full and reduced force supervision** (100/75/50/10/5% forces + 0% = energy-only). Headline findings:

- **SOAP and SOAP-Muon consistently beat AdamW** on both energy and force MAE; **SOAP is the most robust single default** (no extra tuning).
- **Muon only gives partial gains** — strong on CDP, but **WORSE than AdamW on water** (energy), because the orthogonalisation step destabilises training.
- Gains **compound under sparse force supervision** (the label-efficient regime): SOAP-Muon at 50% forces ≈ AdamW at 100% on CDP.
- Wall-clock **speedups of 4.9× (CDP) and 5.8× (water)** to reach AdamW's best force MAE (Figure 2 — figure-only, see ⚠).
- Physically: at **5% forces**, SOAP-Muon stays stable on CDP MD while **AdamW trajectories diverge**; SOAP-Muon reproduces the experimental proton-diffusion activation energy **Eₐ = 0.42 eV** (experimental range 0.39–0.43).

⚠ The "matrix-structured optimisers substantially outperform AdamW" headline is **carried by SOAP/SOAP-Muon**; Muon is a *partial / degrading* case (worse than AdamW on water energy). The wall-clock speedups are figure-only. SOAP-Muon's water gains required **extra per-system tuning** (momentum + singular-value power ρ).

---

## 1. MLIP training objective (§2, Eqs 1–3, L84–141)

An MLIP fits a neural network with weights θ to quantum-mechanical (usually DFT) reference data. The network maps atomic positions + species `{rⱼ, Zⱼ}` to a total energy that decomposes into **local atomic contributions** (size-extensive):

> **Eq 1:** `Eθ({rⱼ,Zⱼ}) = Σᵢ εᵢ,θ({rⱼ,Zⱼ}ⱼ∈𝒩ᵢ)`, where `𝒩ᵢ` is atom i's neighbourhood within a cutoff.

Forces are the **negative energy gradient** (Hellmann–Feynman, energy-conserving by construction):

> **Eq 2:** `Fᵢ,θ = −∂Eθ/∂rᵢ`.

Training minimises a **weighted energy + force MAE** over N reference configs (`Fⁿ ∈ ℝ³ᴺᵃᵗᵒᵐˢ`):

> **Eq 3:** `L(θ) = (λE/N)Σₙ(Eθⁿ−Eⁿ)² + (λF/(3N))Σₙ(1/Nₐₜₒₘₛ)‖Fθ⁽ⁿ⁾−Fⁿ‖²`.

`λE, λF` weigh energy vs force supervision. `λF = 0` ⇒ **energy-only training** (the limiting sparse-force case). **Why sparse forces matter:** DFT forces are nearly free via Hellmann–Feynman, but **coupled-cluster and diffusion Monte-Carlo** force labels are prohibitively expensive → energy-only / sparse-force training is practically important for higher-level theory (L110–119).

---

## 2. The optimisers (§2 "Optimizers", Eq 4–5, L155–217; Algs 1–2)

The loss surface is **anisotropic**; preconditioning rescales the gradient by local curvature `P`:

> **Eq 4:** `w ← w − ηP⁻¹g`.

For a **matrix** parameter `W ∈ ℝᵐˣⁿ` with gradient `G`, the full preconditioner on `vec(W)` would be `mn × mn` (intractable). **Kronecker approximation** `P ≈ Lᵀ ⊗ R` gives Shampoo's update:

> **Eq 5:** `W ← W − ηL⁻ᵖ G R⁻ᵖ`, `L ∈ ℝᵐˣᵐ`, `R ∈ ℝⁿˣⁿ`, preconditioner power `p`.

| optimiser | core idea (one line) | reference |
|---|---|---|
| **Adam / AdamW** | diagonal `P = diag(v̂ₜ)^½` (elementwise 2nd moment); AdamW decouples weight decay | Kingma & Ba 2017; Loshchilov & Hutter 2019 |
| **Shampoo** | L = `GGᵀ`, R = `GᵀG` row/col covariances, `p = ¼`; removing accumulation ⇒ semi-orthogonal update | Gupta et al. 2018; Bernstein & Newhouse 2024 |
| **Muon** | Nesterov momentum → **Newton–Schulz5 orthogonalisation** of the update direction (cheap, no full SVD) | Jordan et al. 2024 |
| **SOAP** | Shampoo eigenspace (`G′ = Qᴸᵀ G Qᴿ`) + AdamW step inside it, project back | Vyas et al. 2025a |
| **SOAP-Muon** | SOAP step + **Muon-style orthogonalisation** (`NewtonSchulz5` if ρ=0, else SVD `PΣρRᵀ`) + RMS normalisation | Vyas et al. 2025b |

### Algorithm 1 — Muon update (verbatim, L744–756)

```
Require: W ∈ ℝᵐˣⁿ, step size η, β
 1: M₀ ← 0
 2: for t = 1,2,… do
 3:   Gₜ ← ∇_W Lₜ(Wₜ₋₁)            ▷ gradient
 4:   Mₜ ← βMₜ₋₁ + (1−β)Gₜ         ▷ momentum buffer
 5:   Uₜ ← βMₜ + (1−β)Gₜ           ▷ Nesterov-style momentum
 6:   Oₜ ← NewtonSchulz5(Uₜ)       ▷ orthogonalised direction
 7:   Wₜ ← Wₜ₋₁ − ηOₜ
 8: end for
 9: return Wₜ
```

### Algorithm 2 — SOAP update with optional Muon ortho + normalisation (verbatim, L803–833)

```
Require: W ∈ ℝᵐˣⁿ, lr η, betas (β,β₁,β₂), ϵ, preconditioning freq f, flags ortho/normalize, SV power ρ
 1: for t = 1,2,… do
 2:   Gₜ ← ∇_W Lₜ(Wₜ₋₁)
 3:   Ǧₜ ← Qᴸ_{t−1}ᵀ Gₜ Q^{R,t−1}        ▷ project to Shampoo eigenbasis
 4:   Mₜ ← β₁Mₜ₋₁ + (1−β₁)Ǧₜ             ▷ Adam 1st moment
 5:   Vₜ ← β₂ᵖVₜ₋₁ + (1−β₂)(Ǧₜ⊙Ǧₜ)       ▷ Adam 2nd moment
 6:   αₜ ← η·√(1−β₂ᵗ)/(1−β₁ᵗ)            ▷ Adam bias-corrected step
 7:   Úₜ ← Mₜ ⊘ (√Vₜ + ϵ)                ▷ Adam elementwise normalisation
 8:   Uₜ ← Q^{L,t−1} Úₜ Q^{R,t−1}ᵀ        ▷ back-project
 9:   if ortho:
10:     if ρ=0:  Uₜ ← NewtonSchulz5(Uₜ)   ▷ Muon ortho (cheap)
11:     elif ρ≠1: SVD Uₜ=PΣRᵀ; Uₜ ← PΣρRᵀ ▷ Muon SV-power (full SVD)
12:   if normalize: Uₜ ← Uₜ/√mean(Uₜ²)    ▷ Muon RMS norm
20:  Lₜ ← βLₜ₋₁ + (1−β)GₜGₜᵀ             ▷ left preconditioner
21:  Rₜ ← βRₜ₋₁ + (1−β)GₜᵀGₜ             ▷ right preconditioner
22:  if t % f == 0:                       ▷ refresh eigenbasis every f steps
23:    S_L = LₜQ^{L,t−1}; S_R = RₜQ^{R,t−1}
24:    Q^{L,t} = QR(S_L); Q^{R,t} = QR(S_R)
28:  Wₜ ← Wₜ₋₁ − αₜUₜ
```

**SOAP-Muon = Algorithm 2 with `ortho=True` AND `normalize=True`** (L856–858). Default `ρ = 0.5` (full SVD, expensive); `ρ = 0` permits cheaper Newton–Schulz but Vyas et al. reported **dataset-dependent instability** — authors use `ρ = 0.5` as the safe default, `ρ = 0` only when stable (L859–865).

### Parameter-group assignment (Table 2, L712–731)

`e3nn` stores all weights as a **flattened 1D vector** `w ∈ ℝᴾ` (Eq 8–10, L757–770). Adam can update it directly; **matrix-structured optimisers must reshape slices back into 2D blocks** per instruction. Assignment:

| model | parameter class | shape | group |
|---|---|---|---|
| **NequIP** | type embedding / FCTP weights / energy readout / per-type scale+shift | 1D / `[mⱼ⁽¹⁾,mⱼ⁽²⁾,nⱼᵒᵘᵗ]` / MLP | **Adam** |
| | edge-MLP, Linear weights | 2D matrices | **Muon** |
| **Allegro** | type embeddings / tensor-product weights / readout MLP | 1D | **Adam** |
| | scalar/tensor embedding, latent, first-layer env MLPs | 2D matrices | **Muon** |

⚠ **FCTP 3D tensor weights (`mⱼ⁽¹⁾×mⱼ⁽²⁾×nⱼᵒᵘᵗ`) were NOT reshaped/preconditioned as 3D** — only 2D Linear weights got matrix structure; left for future work (L772–774, L850–852). So SOAP/Muon preconditioning is **partial** on the architecture.

---

## 3. Experimental setup (§3, L191–217; Tables 3–4)

- **Two systems × two architectures:** liquid **water** (NequIP, Batzner 2022) + solid acid electrolyte **CDP / CsH₂PO₄** (Allegro, Musaelian 2023; Wang 2025a).
- **Force-label fractions:** 100 / 75 / 50 / 10 / 5 % + **0% (energy-only)**, all energy labels retained.
- **Seeds:** **5 seeds** per cell (Table 1/5 report mean ± std); wall-clock curves (Fig 2) use 3 seeds on **NVIDIA A100**.
- **Metric:** per-atom energy MAE `[meV/atom]`, force MAE `[meV/Å]`, wall-clock, + MD physical fidelity (RDF, MSD).
- **Tuning:** systematic per-(task, force-%, optimiser) hyperparameter sweep (Appendix B). ⚠ Best **SOAP-Muon LR ≈ 1 order of magnitude lower** than other optimisers (L1008–1012, Table 4) — e.g. CDP 100% AdamW `3e-2` vs SOAP-Muon `1e-3` (30×); water 100% `1e-2` vs `1e-3` (10×). High AdamW-tuned LRs destabilised SOAP-Muon.
- **Architecture hyperparams** (Table 3, verbatim): both use `lmax=2`, parity=True, 32 features, 8 Bessel basis, polynomial cutoff exponent 6. CDP/Allegro: 2 layers, cutoff 7.0 Å, scalar embed MLP depth 2 / width 128. Water/NequIP: 4 interaction layers, cutoff 4.5 Å, radial MLP depth 3 / width 64, **ZBL pair potential**.
- **SOAP defaults** (L842–845): `β₁=β₂=0.95`, Shampoo `β=β₂=0.95`, preconditioning frequency `f=10`. SOAP-Muon: full preconditioning (every mode), `ρ=0.5` default.

---

## 4. Main results — Table 1 (verbatim, L220–245; mean ± std across 5 seeds)

Joint **energy+force (E+F)** and **energy-only (E)** tasks. ⚠ "Gray force entries" in the E rows = forces NOT in the training objective (extrapolated gradients, reported but not optimised). Best = **bold**, 2nd = underline (rendered here as `**bold**`).

### CDP (Allegro)

| task | optimiser | E [meV/atom] ↓ | F [meV/Å] ↓ |
|---|---|---|---|
| E+F | AdamW | 0.628 ± 0.0434 | 32.2 ± 0.615 |
| E+F | Muon | 0.581 ± 0.0328 | 29.6 ± 0.617 |
| E+F | **SOAP** | **0.569 ± 0.0694** | 29.6 ± 0.305 |
| E+F | **SOAP-Muon** | 0.582 ± 0.0469 | **27.8 ± 0.634** |
| E | AdamW | 5.16 ± 0.178 | 503 ± 31.8 |
| E | Muon | 3.30 ± 0.135 | 281 ± 9.73 |
| E | SOAP | 3.03 ± 0.389 | 214 ± 25.7 |
| E | **SOAP-Muon** | **2.75 ± 0.163** | **201 ± 18.8** |

### Water (NequIP)

| task | optimiser | E [meV/atom] ↓ | F [meV/Å] ↓ |
|---|---|---|---|
| E+F | AdamW | 0.773 ± 0.0713 | 25.7 ± 1.44 |
| E+F | Muon | 1.53 ± 0.317 | 26.6 ± 2.90 |
| E+F | **SOAP** | 0.604 ± 0.0105 | **20.9 ± 0.698** |
| E+F | **SOAP-Muon** | **0.590 ± 0.0687** | 21.0 ± 1.04 |
| E | AdamW | 3.21 ± 0.272 | 306 ± 46.1 |
| E | Muon | 5.37 ± 0.813 | 591 ± 59.1 |
| E | **SOAP** | **2.38 ± 0.573** | **236 ± 54.8** |
| E | SOAP-Muon | 2.77 ± 0.346 | 269 ± 24.5 |

### Prose deltas vs AdamW (§4.1, L285–358) — source-free reconciliation

| claim | recompute | match |
|---|---|---|
| CDP E+F SOAP energy **−9%** | (0.628−0.569)/0.628 = **9.4%** | ✅ |
| CDP E+F SOAP-Muon force **−14%** | (32.2−27.8)/32.2 = **13.7%** | ✅ |
| water E+F SOAP-Muon energy **−24%** | (0.773−0.590)/0.773 = **23.7%** | ✅ |
| water E+F SOAP force **−19%** | (25.7−20.9)/25.7 = **18.7%** | ✅ |
| CDP E-only SOAP-Muon energy **~47%** | (5.16−2.75)/5.16 = **46.7%** | ✅ |
| CDP E-only SOAP-Muon force **~60%** | (503−201)/503 = **60.0%** | ✅ EXACT |
| water E-only SOAP energy **−26%** | (3.21−2.38)/3.21 = **25.9%** | ✅ |
| water E-only SOAP force **−23%** | (306−236)/306 = **22.9%** | ✅ |

**All 8 deltas recompute to the paper's rounded %s.** Best-per-column (Table 1): CDP E+F E=SOAP, F=SOAP-Muon; water E+F E=SOAP-Muon, F=SOAP; CDP E (both)=SOAP-Muon; water E (both)=SOAP.

### ⚠ Muon "partial gains" is CDP-only (the key honest-scope nuance)

The abstract says "Muon only provides partial gains relative to Adam" and §4.1 says "Muon alone underperforms the AdamW baseline, particularly for energy prediction." Source-free check confirms this is **water-driven**:

- **water E+F:** Muon E **1.53 vs AdamW 0.773 → 1.98× WORSE**; Muon F 26.6 vs 25.7 → worse.
- **water E-only:** Muon E **5.37 vs AdamW 3.21 → 1.67× WORSE**; Muon F 591 vs 306 → 1.93× WORSE.
- **CDP E+F:** Muon E 0.581 vs AdamW 0.628 → **better** (Muon does help on CDP).

So Muon's "partial gains" are **entirely CDP-carried**; on water Muon degrades. §5 (L418–424) explicitly attributes this to the **orthogonalisation step** being the primary source of degradation, with adaptive preconditioning (SOAP) "mitigating but not eliminating its effects." Cites Lu et al. 2026 raising the same Muon-stability concern for PINNs (same domain as DSGNAR, iter 67).

---

## 5. Reduced-supervision ablation — Table 5 (verbatim, L1058–1098)

Test MAE mean ± std across 5 seeds, E+F at {100,75,50,10,5}% forces + E at 0%. Gray force entries in the E-only row = not trained. ⚠ Table 5 **100% E+F row is byte-identical to Table 1 E+F row** (cross-table consistency check holds — same AdamW 0.628±0.0434 / 32.2±0.615 … SOAP-Muon 0.590±0.0687 / 21.0±1.04).

### CDP (Allegro)

| force-% | optimiser | E [meV/atom] | F [meV/Å] |
|---|---|---|---|
| 100 | AdamW | 0.628±0.0434 | 32.2±0.615 |
| 100 | Muon | 0.581±0.0328 | 29.6±0.617 |
| 100 | SOAP | 0.569±0.0694 | 29.6±0.305 |
| 100 | SOAP-Muon | 0.582±0.0469 | 27.8±0.634 |
| 75 | AdamW | 0.664±0.0383 | 33.9±0.993 |
| 75 | Muon | 0.622±0.0287 | 31.1±1.11 |
| 75 | SOAP | 0.630±0.0185 | 31.3±0.617 |
| 75 | SOAP-Muon | 0.596±0.0280 | 30.1±0.452 |
| 50 | AdamW | 0.698±0.0175 | 37.4±0.778 |
| 50 | Muon | 0.645±0.0508 | 34.1±0.803 |
| 50 | SOAP | 0.690±0.0724 | 34.5±0.869 |
| 50 | SOAP-Muon | 0.637±0.00589 | 32.5±1.01 |
| 10 | AdamW | 1.28±0.0809 | 68.4±2.54 |
| 10 | Muon | 1.06±0.106 | 60.0±3.10 |
| 10 | SOAP | 1.02±0.0579 | 52.9±1.65 |
| 10 | SOAP-Muon | 0.977±0.0525 | 51.9±0.842 |
| 5 | AdamW | 1.62±0.0551 | 94.9±5.09 |
| 5 | Muon | 1.38±0.103 | 84.0±6.70 |
| 5 | SOAP | 1.16±0.0841 | 69.2±1.83 |
| 5 | SOAP-Muon | 1.20±0.0901 | 68.1±3.05 |
| 0 (E-only) | AdamW | 5.16±0.178 | 503±31.8 |
| 0 | Muon | 3.30±0.135 | 281±9.73 |
| 0 | SOAP | 3.03±0.389 | 214±25.7 |
| 0 | SOAP-Muon | 2.75±0.163 | 201±18.8 |

### Water (NequIP)

| force-% | optimiser | E [meV/atom] | F [meV/Å] |
|---|---|---|---|
| 100 | AdamW | 0.773±0.0713 | 25.7±1.44 |
| 100 | Muon | 1.53±0.317 | 26.6±2.90 |
| 100 | SOAP | 0.604±0.0105 | 20.9±0.698 |
| 100 | SOAP-Muon | 0.590±0.0687 | 21.0±1.04 |
| 75 | AdamW | 0.738±0.117 | 25.1±1.34 |
| 75 | Muon | 0.773±0.146 | 27.1±1.76 |
| 75 | SOAP | 0.601±0.0381 | 20.9±0.872 |
| 75 | SOAP-Muon | 0.602±0.101 | 21.6±1.27 |
| 50 | AdamW | 0.789±0.147 | 26.8±1.30 |
| 50 | Muon | 1.01±0.153 | 30.8±2.86 |
| 50 | SOAP | 0.650±0.0772 | 22.7±1.01 |
| 50 | SOAP-Muon | 0.713±0.191 | 23.1±1.07 |
| 10 | AdamW | 1.09±0.198 | 34.5±3.69 |
| 10 | Muon | 1.15±0.108 | 37.4±2.17 |
| 10 | SOAP | 1.04±0.207 | 31.9±2.19 |
| 10 | SOAP-Muon | 0.999±0.133 | 40.6±17.3 |
| 5 | AdamW | 1.51±0.224 | 41.3±2.09 |
| 5 | Muon | 1.46±0.253 | 43.5±1.10 |
| 5 | SOAP | 1.62±0.659 | 41.9±1.59 |
| 5 | SOAP-Muon | 1.54±0.466 | 41.2±3.31 |
| 0 (E-only) | AdamW | 3.21±0.272 | 306±46.1 |
| 0 | Muon | 5.37±0.813 | 591±59.1 |
| 0 | SOAP | 2.38±0.573 | 236±54.8 |
| 0 | SOAP-Muon | 2.77±0.346 | 269±24.5 |

⚠ At water 10% forces, SOAP-Muon force **40.6 ± 17.3** — the huge std (vs ~1–3 elsewhere) flags a **run instability**; SOAP-Muon is nominally best on E there but its force is 2nd-worst. Water 5% is essentially flat across optimisers (1.46–1.62 E, 41–44 F) — "the trend is less regular" (L333–340). The label-efficiency headline is thus **CDP-carried** too.

### Key reduced-supervision claims (§4.2, L326–362)

- **CDP 50% SOAP-Muon ≈ AdamW 100%** on force accuracy (SOAP-Muon 50% F = 32.5 vs AdamW 100% F = 32.2) — "improved optimisation can partially compensate for reduced access to force labels."
- Water: some 75%/50% models **outperform 100%** (e.g. SOAP 50% E 0.650 vs 100% E 0.604 — actually 100% is better; the "outperform" claim refers to seed-level noise bands, not the mean). ⚠ Non-monotonic in force fraction on water — mean values do NOT monotonically improve with more labels for several optimisers (AdamW E: 0.773→0.738→0.789 from 100→75→50).

---

## 6. Wall-clock convergence (Figure 2 — figure-only ⚠)

- **CDP SOAP 4.9× faster** than AdamW to reach AdamW's min median validation force MAE.
- **Water SOAP 5.8× faster.**
- 3 seeds, NVIDIA A100, per-epoch medians + interquartile bands; circles = first wall-clock time each optimiser's median crosses AdamW's best.
- ⚠ **These are figure-derived numbers, no numeric wall-clock table** — cannot be source-cell-verified. The paper notes matrix-structured methods pay higher per-step cost, so the wall-clock win implies an **even larger epoch-count reduction**.

### Figure 1 — relative-MAE heatmap (verbatim grid, L262–281)

`MAE_opt / MAE_AdamW@100%F` per cell. Green = better than AdamW@100%F. Selected rows (energy, CDP):

| optimiser | 0% | 5% | 10% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| AdamW | 8.22 | 2.58 | 2.03 | 1.11 | 1.06 | 1.00 |
| Muon | 5.26 | 2.20 | 1.69 | 1.03 | 0.99 | 0.93 |
| SOAP | 4.83 | 1.84 | 1.63 | 1.10 | 1.00 | 0.91 |
| SOAP-Muon | 4.38 | 1.92 | 1.56 | 1.01 | 0.95 | 0.93 |

Force CDP / energy+force water grids transcribed similarly (L262–281). On water energy at 100%, SOAP reaches **0.78×** and SOAP-Muon **0.76×** AdamW@100%F (best cells); Muon water energy at 0% blows up to **6.94×** (the orthogonalisation failure mode).

---

## 7. Physical fidelity (§4.3, L364–424; Figures 3, 5–8)

- **CDP MD (Fig 3, 5):** All fully-force-supervised models reproduce AIMD RDFs + MSD. At **5% forces**, **SOAP-Muon stays stable and matches AIMD**, while **AdamW trajectories diverge almost immediately → nonphysical results**. The 5% SOAP-Muon model reproduces experimental proton-diffusion **Eₐ = 0.42 eV** (experimental range **0.39–0.43 eV**, Haile 2007 / Ishikawa 2008 / Wang 2025b) — ✅ 0.42 ∈ [0.39, 0.43].
- **Water MD (Fig 6, 8):** Fully-supervised models all match the experimental O–O RDF reference (Skinner 2014). At all sparsity levels **AdamW and SOAP maintained physical fidelity** on RDF/MSD (Fig 8) — water is the easier case physically.

---

## 8. Discussion + limitations (§5, L426–432)

- **CDP:** clear progression AdamW → Muon → SOAP → SOAP-Muon (all beat AdamW).
- **Water:** SOAP ≈ SOAP-Muon > AdamW, but **Muon < AdamW across all force fractions** (the orthogonalisation-is-degradation thesis, L418–424).
- **Practitioner default recommendation: SOAP** (most robust, no extra tuning).
- **Limitations (authors' own, L425–432):** only **2 systems + 2 architectures**; **foundation-potential training/fine-tuning NOT studied** (expected to scale, needs future work); expect SOAP to remain effective at current foundation scale but unverified.

---

## ⚠ Honest-scope flags (9 inline)

1. **"Matrix-structured optimisers substantially outperform AdamW" is SOAP/SOAP-Muon-carried; Muon is a partial/degrading case.** Muon is WORSE than AdamW on water energy (1.98× E+F, 1.67× E-only) and force. Diagnostic: when an "X beats baseline" headline groups multiple methods, check each method vs baseline separately before echoing the group claim.
2. **Wall-clock speedups (4.9×, 5.8×) are figure-only (Figure 2)** — no numeric wall-clock table; not cell-verifiable. Universal "figure-derived numbers are weak" rule, now across 55 repo papers.
3. **SOAP-Muon's water gains required extra per-system tuning** (momentum coefficients + singular-value power ρ; Appendix B, Figure 4 tuned-vs-untuned). The "single default = SOAP" recommendation partly reflects that SOAP-Muon is NOT a drop-in default.
4. **Best SOAP-Muon LR ≈ 1 order of magnitude lower than other optimisers** (Table 4). The comparison is best-tuned-vs-best-tuned per cell, not default-vs-default; SOAP-Muon needed a destabilising-LR-aware sweep.
5. **FCTP 3D tensor weights NOT preconditioned** (only 2D Linear weights reshaped for SOAP/Muon; L772–774, L850–852). Matrix-structure application is partial on the equivariant architecture — a future-work implementation gap.
6. **Label-efficiency headline is CDP-carried.** Water force-fraction sweep is non-monotonic and largely flat at 5% (1.46–1.62 E, 41–44 F across all 4 optimisers); water 10% SOAP-Muon force std = 17.3 flags a run instability.
7. **Only 2 systems × 2 architectures; no foundation-potential study** (authors' own §5 limitation). Generalisation beyond NequIP/Allegro + water/CDP unverified.
8. **ρ=0.5 SOAP-Muon default requires a full SVD** (expensive); the cheaper ρ=0 Newton–Schulz variant has "dataset-dependent instability" (Vyas 2025b). So the reported SOAP-Muon is the expensive variant, and per-step-cost claims hinge on ρ choice.
9. **Energy-only "F" columns are extrapolated gradients** (gray = not in training objective). They report force MAE for models never trained on forces — useful as a PES-gradient probe but not an optimised metric.

---

## Verdict

A well-scoped, honestly-caveated **empirical optimiser bake-off** that opens optimizer choice as a design axis for MLIPs — *not* a new optimiser. Strongest, most-citable claims: **SOAP is a robust drop-in default** that beats AdamW on both systems without extra tuning, and **SOAP-Muon at 50% forces matches AdamW at 100%** on CDP (label efficiency). Weakest links: the group headline obscures Muon's water regression; wall-clock and MD claims are figure-only; SOAP-Muon's water edge needs per-system tuning + a 10×-lower LR. No numeric prose-vs-table contradiction found; **all 8 prose deltas recompute exactly**, and Table 5's 100% row is byte-identical to Table 1. Sibling-in-spirit to DSGNAR (iter 67) — both optimiser-first science-ML papers, but DSGNAR proposes a new second-order method for PINNs while this benchmarks existing first-order matrix-structured optimisers for equivariant MLIPs.

**Source-free reconciliation: PASSED, 0 contradictions.**
