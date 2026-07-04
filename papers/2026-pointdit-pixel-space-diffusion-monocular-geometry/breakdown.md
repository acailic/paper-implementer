# PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation — source-first breakdown

- **arXiv:** 2607.02515v1 [cs.CV], 2 Jul 2026
- **Authors:** Haofei Xu¹²³, Rundi Wu¹, Philipp Henzler¹, Nikolai Kalischek¹, Michael Oechsle¹, Fabian Manhardt¹, Marc Pollefeys²⁴, Andreas Geiger³⁵, Federico Tombari¹⁶, Michael Niemeyer¹
  (¹Google, ²ETH Zürich, ³University of Tübingen / Tübingen AI Center, ⁴Microsoft, ⁵KE:SAI, ⁶TU Munich)
- **Venue:** ICML 2026 (PMLR 306). Project: https://haofeixu.github.io/pointdit
- **PDF:** 16 pp (`pdfinfo`=16pp; `file`=16pp — **NO file-vs-pdfinfo page-count defect this iter**, like iter-68 the intermittent no-defect case). `paper_layout.txt` = 955 lines (pdftotext -layout). **9 explicit tables (T1 main, T2 single-step, T3 ablation a–e, T4 train datasets, T5 eval datasets, T6 train cost, T7 per-dataset point, T8 per-dataset depth, T9 per-dataset BF1) + Eqs 1–8 + Figs 1–7.**
- **Repo rank:** 61st paper, rank 56 unique — **FIRST monocular-geometry / 3D-point-map / depth-estimation / pixel-space-diffusion-for-geometry / DINOv3-conditioning paper.** (Prior "depth correction" / "low-level reconstruction" keyword hits = ViQ iter-7 image-token VQ, unrelated; no folder is geometry/depth/3D-recon.)

## One-line thesis
A **minimalist pixel-space Diffusion Transformer (plain ViT)** predicts dense **3D point maps** (H×W×3, camera-frame XYZ) directly from raw patches — no VAE tokenizer, no hybrid conv head, no intricate geometric-consistency losses — conditioned on frozen **DINOv3** image tokens, trained **from scratch** with **x-prediction flow matching** (predict clean point map, not velocity). Surpasses latent-diffusion (GeometryCrafter) and deterministic regression (MoGe-2) on depth accuracy + boundary sharpness, with 1-step inference (72 ms vs 1,178 ms).

## Method (§3, Eqs 1–8)

**Flow matching (§3.1).** Linear-interpolation probability path between noise ε∼N(0,I) and data point map x:
- `z_t = t·x + (1−t)·ε` (Eq 1); t=0 pure noise, t=1 clean data.
- Constant ground-truth velocity `v_t = dz_t/dt = x − ε` (Eq 2).
- Conditional flow `v_θ(z_t, t | c)` for input image c.

**Point map normalization (Eq 3).** Point maps span varying ranges (indoor vs outdoor); to match the N(0,I) noise scale, standardize: `x̃ = (x − μ)/s`, μ = centroid, s = mean Euclidean distance of points from centroid. ⇒ predictions are **affine-invariant** (recovered up to unknown scale+shift; aligned to GT via MoGe least-squares at eval).

**Sky processing.** Exclude sky from μ,s; project sky onto virtual sphere radius 3 (≈3σ of N(0,1) prior); down-weight (not mask) sky pixels in loss (w_i=0.01 vs 1). At inference discard points with norm > 2.9 (just below sky sphere radius 3).

**Architecture (§3.2).** Plain ViT operating on point-map patches. Patchify noisy z_t into p×p patches (p=16), N=(H/p)(W/p) tokens, linear-project to dim D. **Image conditioning:** frozen DINOv3, **4 uniformly-spaced intermediate layers** (DPT-style layer selection, not last-layer-only like RAE), concatenate along channel → T_c ∈ R^{N×4D}. **Fuse** T_in = Concat(T_c, T_z) ∈ R^{N×5D}, linear → D, stack of Transformer blocks (MHSA + MLP). Linear predict head → unpatchify → x̂ ∈ R^{H×W×3} (clean point map estimate).

**Clean-point-map prediction (x-prediction, the central design choice).** Network outputs x̂ (clean data), NOT velocity — extending JiT (Li & He 2026) from 2D images to 3D point maps. Estimated velocity recovered by rearranging Eq 1: `v̂_t = (x̂ − z_t)/(1−t)` (Eq 4), denominator clipped to δ=0.05 for t→1 stability. **Optimized in velocity space (v-loss),** which empirically beats direct x-loss.

**Losses (§3.3).**
- Flow-matching MSE (Eq 5): `L_fm = E[ (1/M)Σ_i w_i ‖v̂_{t,i} − (x_i − ε_i)‖² ]`, w_i=0.01 sky / 1 otherwise.
- Relative point loss (Eq 6): `L_rel = E[ (1/M)Σ_i w_i · ‖x̂_i − x_i‖₁ / (‖x_i‖₂ + ξ) ]` — normalizes per-pixel error by target magnitude so distant points don't dominate.
- Total (Eq 7): `L = L_fm + λ·L_rel`, λ=0.1. End-to-end; flow matching primary, single lightweight aux term.

**Noise schedule.** Logit-normal (JiT): z∼N(μ=−0.8, σ²=0.8), t=sigmoid(z). **Rectified sampling:** with p_zero=0.1 override t←0 (sigmoid never hits exactly 0 ⇒ train-test discrepancy at inference start). Table 3(b) ablates μ ∈ {−0.8, −1.2, −1.6}.

**Inference (§3.4, Eq 8).** Euler ODE solver from z_0∼N(0,I): `z_{t+Δt} ← z_t + Δt·v̂_t`. **1-step already competitive;** more steps refine BF1. Also works as deterministic estimator with all-zeros input (Table 2).

## Setup (§4)

- **Train (synthetic only):** Stage-1 256×256 pre-train on SceneNet-RGBD (5,359,500 ≈ 5.36M indoor); Stage-2 512×512 fine-tune on 11-dataset mixture (6,217,678 ≈ 6.22M, weighted sampling decoupled from corpus size — TartanGround = 67.07% of samples but weight 0.15; Synscapes 25k weight 0.09). Convert raw depth→point maps via intrinsics.
- **3 variants (JiT configs):** PointDiT-B (223M), PointDiT-L (771M), PointDiT-H (1,807M). Frozen DINOv3 backbone scaled per variant (ViT-L features for PointDiT-L etc.); all Transformer layers + heads trained from scratch. Patch p=16 all variants.
- **Optimization:** AdamW (β1=0.9, β2=0.95, wd=0), bf16 mixed precision, lr=blr·(global batch)/256, no grad clip/accumulation, two EMAs (decay 0.9999 used for eval, 0.9996). Stage-1: blr=5e-5, 5-epoch warmup + constant, 30 epochs. Stage-2: blr=1e-4, constant, init from Stage-1 (pos-embed bicubic 16×16→32×32).
- **Eval (zero-shot real, disjoint from train):** 7 datasets / 3,444 samples — DIODE(771), KITTI(652), NYUv2(654), ETH3D(454), HAMMER(775), iBims-1(100), Booster(38). BF1 only on 3 boundary sets (HAMMER+iBims-1+Booster = 913 samples). 256×256 and 512×512; shorter-side-rescale + center-crop.
- **Metrics:** δ1 (accuracy %, ratio<1.25), Rel (relative abs error, scale-normalized), BF1 (boundary F1, Depth Pro). Reported in point-map domain (Rel_p, δ1_p) and depth domain (Rel_d, δ1_d; depth = z-component of point).
- **Baselines (8):** GeometryCrafter (latent diffusion), PPD (pixel-space diffusion, v-pred — closest related), Depth Pro, UniDepthV2, DA3, MoGe, MoGe-2. PPD predicts depth only ⇒ its point-map metrics computed by recovering intrinsics with MoGe-2 (official PPD repo).

## Tables (verbatim, with sourcing line-ranges)

### Table 1 — Main comparisons (7-dataset avg, 512×512) [L323–358]
Caption: PointDiT-H best depth (Rel_d, δ1_d) + best point δ1_p; PointDiT sharpest BF1; 72 vs 1,178 ms single-step vs GeometryCrafter.

| Method | Rel_p↓ | δ1_p↑ | Rel_d↓ | δ1_d↑ | BF1↑ | Param(M) | Time(ms) |
|---|---|---|---|---|---|---|---|
| GeometryCrafter | 5.45 | 96.75 | 3.52 | 97.84 | 4.64 | 1,937 | 1,178 |
| PPD | 5.54 | 96.59 | 3.88 | 97.78 | **9.28** | 804 | 402 |
| Depth Pro | 5.71 | 96.71 | 3.84 | 97.63 | 9.41 | 952 | 68 |
| UniDepthV2 | 4.45 | 97.35 | 2.86 | 98.52 | 6.94 | 354 | 26 |
| DA3 | 4.77 | 96.63 | 3.22 | 97.81 | 6.33 | 1,356 | 82 |
| MoGe | **4.21** | 97.45 | 3.10 | 98.01 | 5.61 | 314 | 34 |
| MoGe-2 | 4.53 | 97.46 | 2.90 | 98.45 | 7.40 | 326 | 24 |
| PointDiT-B (1 step) | 5.84 | 96.71 | 3.70 | 97.84 | 8.18 | 223 | 31 |
| PointDiT-B (2 steps) | 5.81 | 96.77 | 3.64 | 97.86 | 8.88 | 223 | 47 |
| PointDiT-B (3 steps) | 5.83 | 96.79 | 3.64 | 97.86 | 9.09 | 223 | 63 |
| PointDiT-B (4 steps) | 5.85 | 96.80 | 3.64 | 97.86 | 9.16 | 223 | 79 |
| PointDiT-L (1 step) | 4.90 | 97.42 | 3.15 | 98.22 | 9.56 | 771 | 65 |
| PointDiT-L (2 steps) | 4.84 | 97.52 | 3.09 | 98.24 | 10.11 | 771 | 87 |
| PointDiT-L (3 steps) | 4.85 | 97.54 | 3.09 | 98.25 | 10.36 | 771 | 109 |
| PointDiT-L (4 steps) | 4.85 | 97.55 | 3.09 | 98.25 | **10.50** | 771 | 131 |
| PointDiT-H (1 step) | 4.45 | 97.93 | 2.81 | 98.51 | 9.79 | 1,807 | 72 |
| PointDiT-H (2 steps) | 4.38 | 97.99 | 2.75 | 98.54 | 10.31 | 1,807 | 116 |
| PointDiT-H (3 steps) | 4.39 | 98.01 | 2.75 | 98.54 | 10.44 | 1,807 | 160 |
| PointDiT-H (4 steps) | 4.40 | **98.02** | **2.75** | **98.54** | 10.49 | 1,807 | 204 |

### Table 2 — Single-step feed-forward (PointDiT-H) [L393–403]
Caption: nearly invariant to noise; all-zeros matches/slightly exceeds stochastic.

| Method | Rel_p↓ | δ1_p↑ | Rel_d↓ | δ1_d↑ | BF1↑ |
|---|---|---|---|---|---|
| rand noise (seed 1) | 4.454 | 97.928 | 2.815 | 98.505 | 9.772 |
| rand noise (seed 2) | 4.452 | 97.938 | 2.811 | 98.513 | 9.778 |
| rand noise (seed 3) | 4.454 | 97.921 | 2.812 | 98.513 | 9.772 |
| all zeros (no rand) | 4.446 | 97.934 | 2.806 | 98.508 | 9.792 |

### Table 3 — Ablations (256×256 SceneNet, 7-test avg, single-step PointDiT-L) [L550–580]
Default (x-pred, μ=−0.8 & rand-zero, DINOv3 4-layer, v-loss, p=16) highlighted gray.

| (a) Prediction target | Rel_p↓ | δ1_p↑ | Rel_d↓ | δ1_d↑ | BF1↑ |
|---|---|---|---|---|---|
| v-pred | 35.44 | 30.03 | 24.07 | 58.21 | 0.46 |
| **x-pred** | 9.29 | 91.18 | 5.54 | 95.08 | 13.47 |

| (b) Noise schedule (t-shift) | | | | | |
|---|---|---|---|---|---|
| −0.8 | 12.19 | 84.82 | 7.80 | 91.34 | 8.05 |
| −1.2 | 11.86 | 85.53 | 7.46 | 91.87 | 8.11 |
| −1.6 | 10.73 | 88.06 | 6.74 | 93.09 | 7.31 |
| **−0.8 & rand zero** | 9.68 | 90.54 | 6.00 | 94.64 | 7.24 |

| (c) Image patch embedding | | | | | |
|---|---|---|---|---|---|
| Linear | 13.32 | 82.56 | 9.64 | 88.09 | 9.68 |
| DINOv2 (last layer) | 9.80 | 90.07 | 5.99 | 94.34 | 5.11 |
| DINOv3 (last layer) | 9.68 | 90.54 | 6.00 | 94.64 | 7.24 |
| **DINOv3 (4 layers)** | 9.29 | 91.18 | 5.54 | 95.08 | 13.47 |
| MoGe-2 (4 layers) | 8.29 | 93.35 | 4.93 | 96.19 | 11.75 |
| DA3 (4 layers) | 8.26 | 93.09 | 4.74 | 96.47 | 12.58 |

| (d) Training loss | | | | | |
|---|---|---|---|---|---|
| v-loss | 9.29 | 91.18 | 5.54 | 95.08 | 13.47 |
| **v-loss & point loss** | 9.10 | 91.48 | 5.53 | 94.88 | 13.92 |

| (e) Patch size (512×512) | | | | | |
|---|---|---|---|---|---|
| 32 | 5.35 | 96.88 | 3.48 | 97.78 | 6.17 |
| **16** | 5.01 | 97.34 | 3.06 | 98.17 | 10.37 |

### Table 4 — Training datasets [L796–814]
Stage-1: SceneNet-RGBD indoor 5,359,500 (w=1.00). Stage-2 (6,217,678, Σw=1.00): Hypersim 70,647 (0.12), VKITTI2 42,520 (0.14), UrbanSyn 7,539 (0.05), Synscapes 25,000 (0.09), TartanAir 306,637 (0.10), OmniWorld-Game 1,024,252 (0.19), EDEN 368,663 (0.05), IRS 39,342 (0.02), Dynamic Replica 150,900 (0.03), MVS-Synth 12,000 (0.06), TartanGround 4,170,178 (0.15).

### Table 5 — Zero-shot eval datasets [L819–827]
DIODE 771, KITTI 652, NYUv2 654, ETH3D 454, HAMMER 775 (boundary✓), iBims-1 100 (✓), Booster 38 (✓). Total 3,444.

### Table 6 — Training cost [L835–843]
| Model | Param(M) | Pre-train (256²): ep/GPU/time | Fine-tune (512²): ep/GPU/time |
|---|---|---|---|
| PointDiT-B | 223 | 30 / 16 / 12h | 8 / 64 / 2.5h |
| PointDiT-L | 771 | 30 / 16 / 21h | 5 / 64 / 7h |
| PointDiT-H | 1,807 | 30 / 64 / 22h | 3 / 128 / 5.5h |

### Table 7 — Per-dataset point map (Rel_p↓/δ1_p↑), 4-step, 512×512 [L859–882]
Avg = sample-weighted mean over 3,444. Outdoor (KITTI/DIODE/ETH3D) PointDiT-H loses Rel_p to MoGe/UniDepthV2.

### Table 8 — Per-dataset depth (Rel_d↓/δ1_d↑) [L888–908]
**PointDiT-H BEST Rel_d on DIODE (2.90 < MoGe 3.17 < UniDepthV2 3.62) AND ETH3D (2.46 < MoGe 2.53 < UniDepthV2 2.65)** — only KITTI depth (3.88) loses. ⇒ §B.1 "outdoor inferior" claim is Rel_p-carried, not depth-universal (see honest-scope flag).

### Table 9 — Per-dataset BF1 (HAMMER/iBims-1/Booster, Avg over 913) [L933–943]
GeometryCrafter 4.64, PPD **9.26**, Depth Pro 9.41, UniDepthV2 6.94, DA3 6.33, MoGe 5.61, MoGe-2 7.40, PointDiT-B 9.16, PointDiT-L **10.50**, PointDiT-H 10.49.

## Source-free reconciliation (Python)

**ALL prose deltas recompute EXACT:**
- "PointDiT-H best depth (Rel_d, δ1_d)": Rel_d min baseline UniDepthV2 2.86 vs PointDiT-H min 2.75 ✓; δ1_d max baseline UniDepthV2 98.52 vs PointDiT-H 98.54 ✓; "best point δ1_p": max baseline MoGe-2 97.46 vs PointDiT-H 98.02 ✓.
- "PointDiT sharpest BF1": max baseline Depth Pro 9.41 vs PointDiT max 10.50 ✓.
- "raises BF1 from 9.41 (best baseline) to 10.50": 9.41 ✓, 10.50 (PointDiT-L 4-step) ✓ EXACT.
- "On Rel_p, MoGe slightly ahead (4.21 vs 4.40)": MoGe 4.21, PointDiT-H 4-step 4.40 ✓ EXACT.
- "72 vs 1,178 ms single-step": PointDiT-H 1-step 72, GeometryCrafter 1,178 ✓ (ratio 16.36×, not stated).
- TartanGround 4,170,178/6,217,678 = **67.07% ≈ 67.1%** ✓; Synscapes weight 0.09 ✓; Stage-2 total 6.22M ✓; Stage-1 5.36M ✓.
- Eval 7-dset sum = **3,444** ✓; boundary 913 ✓ (matches T9 caption).
- Table-1 ↔ Table-7/8/9 cross-table consistency: Rel_p/δ1_p (T1↔T7), Rel_d/δ1_d (T1↔T8) **all 20 method×metric Avg cells byte-identical**; BF1 T1↔T9 **9 of 10 methods byte-identical**.

**CAUGHT genuine cross-table inconsistency (iter-30/31/34/60/69 prose-vs-table / cross-table-drift class):**
- **PPD BF1: Table-1 = 9.28 vs Table-9 Avg = 9.26** (0.02 mismatch). Both are the sample-weighted mean over the same 913 boundary samples (T1 caption "3,444 samples… BF1"; T9 caption "913 boundary-annotated samples"). All 9 OTHER methods' BF1 match exactly between T1 and T9 (GeometryCrafter 4.64, Depth Pro 9.41, UniDepthV2 6.94, DA3 6.33, MoGe 5.61, MoGe-2 7.40, PointDiT-B 9.16, PointDiT-L 10.50, PointDiT-H 10.49) — only the PPD cell differs. One of the two tables carries a stale PPD BF1 value. Does NOT affect the "9.41 best baseline → 10.50" headline (PPD 9.26/9.28 < Depth Pro 9.41 either way).

**CAUGHT localized cross-table drift on PointDiT-H 1-step BF1 (lower-confidence, parallel to iter-65/70 run-drift class):**
- Table-1 PointDiT-H 1-step **BF1 = 9.79**, but Table-2 random-noise 3-seed mean BF1 = 9.774 (→9.77) and all-zeros = 9.792 (→9.79). So T1's 9.79 matches the **all-zeros** config, not the random-noise mean (9.77). The OTHER 4 metrics of that same T1 1-step row (Rel_p 4.45, δ1_p 97.93, Rel_d 2.81, δ1_d 98.51) all match the random-noise mean (4.453, 97.929, 2.813, 98.510), NOT all-zeros. ⇒ the BF1 cell of the T1 1-step row is the lone outlier, consistent with a different aggregation/seed for that one cell. Does not affect any headline (10.50/10.49 are 4-step).

## Strengths
1. **VAE-free pixel-space diffusion for geometry** removes the lossy tokenizer that caps latent-diffusion quality (Fig 2a) — citable engineering hinge: x-prediction + DINOv3 multi-layer conditioning makes a plain ViT viable directly on raw H×W×3 point maps.
2. **x-prediction vs v-prediction** is the falsifiable design lever (Table 3a: v-pred collapses to BF1 0.46 / δ1 58.21 vs x-pred 13.47 / 95.08) — clean ablation, parallel to JiT for 2D images.
3. **1-step + deterministic (all-zeros) inference** (Table 2) — the model learns a robust image-feature→geometry mapping, supports variable step count with one network.
4. **Best depth accuracy + sharpest boundaries simultaneously** while 16× faster (72 vs 1,178 ms) than the latent-diffusion SOTA.

## Honest-scope flags (⚠ — transcribed verbatim, NOT reconciled)
1. **Cross-table PPD BF1 mismatch 9.28 (T1) vs 9.26 (T9)** — genuine stale-cell; flag, don't echo either as definitive.
2. **T1 PointDiT-H 1-step BF1=9.79 drift** — matches all-zeros (9.792) not random-noise mean (9.774); other 4 metrics of that row match random-noise mean. Lone-cell outlier, no headline impact.
3. **§B.1 "PointDiT inferior on outdoor (KITTI/DIODE/ETH3D) vs MoGe/UniDepthV2" is Rel_p-carried, NOT depth-universal** — on depth Rel_d PointDiT-H is BEST on DIODE (2.90) and ETH3D (2.46), losing only KITTI depth (3.88); it loses on point-map Rel_p on all 3 outdoor sets. Read the outdoor caveat as point-map-specific.
4. **"More robust on transparent objects (Booster)" is BF1/depth-carried, NOT Rel_p** — Booster Rel_p best is MoGe-2 (2.59) not PointDiT-H (3.01); Booster δ1_p best MoGe-2 (99.90) not PointDiT-H (99.87). PointDiT-H does lead Booster Rel_d/δ1_d/BF1.
5. **"Even a single sampling step surpasses all prior on BF1"** holds for PointDiT-L 1-step (9.56) and H 1-step (9.79) over best-prior Depth Pro 9.41, but **NOT PointDiT-B 1-step (8.18 < 9.41)** — needs the largest 2 variants.
6. **BF1 headline 10.50 is PointDiT-L 4-step, narrowly over PointDiT-H 4-step 10.49** (0.01 margin); caption "PointDiT sharpest" is ambiguous on which variant — and PointDiT-H is the depth winner, so no single variant sweeps all 5 metrics.
7. **Deterministic-vs-generative 10.90 vs 13.92 (Fig 5) — deterministic 10.90 is figure-only** (no table cell); 13.92 = Table 3(d) "v-loss & point loss" row, NOT the default v-loss 13.47. The controlled-comparison generative number includes the relative point loss.
8. **Synthetic-only training** (5.36M + 6.22M); zero-shot real eval — domain gap bridged only via frozen DINOv3 features; authors' own limitation (outdoor weaker).
9. **PPD point-map metrics confounded** — PPD predicts depth only; its point-map Rel_p/δ1_p computed by recovering intrinsics with MoGe-2 (official PPD repo), so the "PointDiT beats PPD across all metrics" claim on point-map metrics is not apples-to-apples.
10. **Single-seed main tables** (T1, T7, T8, T9); only Table 2 reports 3 seeds (PointDiT-H only). No CIs/significance; many gaps are sub-0.1 (Rel_d PointDiT-H 2.75 vs UniDepthV2 2.86 vs MoGe-2 2.90; δ1_d 98.54 vs 98.52).
11. **Fixed resolutions only** (256²/512²); no mixed-resolution training (authors' own limitation).

## Verdict
Clean, well-ablated paper; central x-prediction + VAE-free pixel-space thesis is sound and the main-table win is real (best depth + sharpest boundaries at 16× lower latency). **Two genuine numeric defects caught** (PPD BF1 cross-table 9.28-vs-9.26; PointDiT-H 1-step BF1 lone-cell drift) — neither affects a headline. **Two honest-scope overstatements** to flag when citing: the "outdoor inferior" caveat is point-map-Relp-carried (PointDiT-H wins outdoor depth on DIODE/ETH3D), and the "transparent-robust" claim is BF1/depth-carried (loses Booster Rel_p/δ1_p to MoGe-2). Repo's first monocular-geometry / depth / 3D-point-map / pixel-space-diffusion-for-geometry paper — fresh subarea secured.
