# PixWorld: Unifying 3D Scene Generation and Reconstruction in Pixel Space

**arXiv 2607.05373** | cs.CV | Repo paper rank 3 | Iter 96

---

## Problem & Motivation

3D reconstruction (feed-forward, pixel regression) and 3D generation (latent diffusion) have developed as separate paradigms. Recent attempts to unify them (Gen3R) work in **latent space**, introducing two problems:

1. **Diffusion objective decoupled from 3D**: loss is on latent features, not the rendered 3D output -- the 3D representation is never directly optimized
2. **Information loss**: both branches go through a pretrained VAE or RAE, losing detail and requiring extra training

PixWorld reformulates unification under **pixel-space diffusion**: no VAE/RAE, diffusion loss applied directly on rendered multi-view images via differentiable rendering, aligning the training signal with 3D scene fidelity.

**Authors**: Sensen Gao, Zhaoqing Wang, Qihang Cao, Dongdong Yu, Changhu Wang, Jia-Wang Bian (NTU Singapore / AISphere)

---

## Key Insight

By performing flow matching directly in pixel space over multi-view renderings of a 3D Gaussian Splatting (3DGS) representation, you get:
- No latent bottleneck (no VAE/RAE needed)
- Diffusion signal directly supervises the 3D representation through differentiable rendering
- Unification reduces to: partition views into clean (reconstruction) and noisy (generation) subsets, process both in one forward pass

---

## Method

### 3.1 Pixel-Space Diffusion (Preliminary)

Following JiT (Li & He 2025) -- image prediction in pixel space:

Given clean image x, noise eps, timestep t in [0,1]:
- Noisy input: x_t = t * x + (1-t) * eps
- Denoiser: f_theta(x_t, t, c) -> x_hat (image predictor)
- Velocity: v_hat = (x_hat - x_t) / (1-t)
- Flow matching loss: L_FM = E[||v_hat - v||^2] where v = x - eps

**Eq 2**: L_FM = E_{x,eps,t} [ ||x_hat - x||^2 / (1-t)^2 ]

Key difference from latent diffusion: the diffusion variable stays in RGB space, so loss can be applied on rendered images directly.

### 3.2 PixWorld Framework

**Task formulation**: Given N posed views {I_n}, {T_n}, partition indices into:
- Omega_c (clean) and Omega_n (noisy), disjoint, |Omega_c| >= 1
- When Omega_n = empty -> pure reconstruction
- Otherwise -> conditioned generation
- Optional text prompt y

**Model input per view n**:
- Clean: I_n (unchanged)
- Noisy: I_tilde_n = t * I_n + (1-t) * eps_n (all noisy views share same t)

**Eq 4**: Two-stream DiT: f_theta(I_tilde, T, y) -> (D_hat, G_hat)
- D_hat = predicted multi-view depth maps
- G_hat = pixel-aligned 3D Gaussian scene representation
- Clean/noisy views embedded separately, processed by shared transformer blocks
- Noisy stream conditioned on timestep t; clean stream always uses t=1 (fully denoised)
- Camera params via PRoPE (Li et al. 2025a)
- Text via cross-attention

**3D Gaussian decoding**: Pixel p of view n with depth d_hat_np gets center mu_np = Pi^{-1}(p, d_hat_np, T_n) (inverse projection to world). All pixels aggregated into scene-level G_hat.

**Rendering**: R(G_hat, T_n) -> I_bar_n (rendered images)

**Eq 6 -- Rendering loss**:
L_render = L_recon + L_FM + lambda_lpips * L_lpips

Where:
- L_recon = (1/|Omega_c|) sum_{n in Omega_c} ||I_bar_n - I_n||^2 (clean views: direct MSE)
- L_FM = (1/|Omega_n|) sum_{n in Omega_n} ||v_bar_n - v_n||^2 (noisy views: velocity matching)
- L_lpips = 1[t > t_th] * (1/N) sum_n LPIPS(I_bar_n, I_n) (perceptual, gated)

Also: 4-8 novel views per iteration supervised via MSE + LPIPS.

**Eq 7 -- Depth loss**:
L_depth = (1/N) sum_n rho(log D_hat_n - log D_star_n)
where rho = Huber loss element-wise, D_star from DA3 (Lin et al. 2025)

**Eq 10 -- Full objective**:
L = L_render + lambda_depth * L_depth + lambda_geo * L_geo

### 3.3 Geometry Perception Loss

Motivation: L_render is photometric only -- doesn't fully constrain 3D structure (depth drift along ray, translucent floaters, view-dependent texture compensating geometric misalignment).

Using frozen 3D foundation model Psi (e.g., VGGT or pi^3):

**Eq 8**: H_bar = Psi(I_bar, T), H_star = Psi(I, T) (multi-view geometric features)

**Eq 9 -- Geometry perception loss**:
L_geo = 1[t > t_th] * (1/(N * H' * W')) * sum_{n,p} [1 - cos_sim(h_bar^n_p, h_star^n_p)]

Psi jointly processes all views with cameras -> features encode cross-view 3D structure. Reference branch H_star has stopped gradients; only H_bar backprops. Activated only when t > t_th (geometric matching unstable near pure noise).

---

## Architecture

### Two-Stream MMDiT Denoiser

- 24-layer DiT, hidden dim d=1024, 16 heads (head_dim=64)
- SwiGLU FFN, RMSNorm on Q/K, adaLN-Zero modulation
- Two parallel streams (clean + noisy) per block: independent pre-LN, QKV/output projections, SwiGLU MLP, adaLN-Zero
- Joint full attention over concatenated [Omega_c; Omega_n] tokens with shared q,k-RMSNorm
- Shared cross-attention to text, shared timestep embedder (t=1 for clean, sampled t for noisy)
- Camera via PRoPE
- 16x16 patchify with learnable positional embeddings
- Per-stream output heads: depth (1024 -> 16x16x1) + 3DGS attributes (1024 -> 16x16x35)

### Parameter Budget (Table 6)

| Module | #Params |
|--------|---------|
| Tokenization & conditioning | 8.25M |
| Two-stream MMDiT trunk (24 blocks) | 1007.6M |
| Multi-task output heads (2x) | 27.28M |
| **Total** | **1.044B** |

---

## Training Details

- **Data**: Re10K + DL3DV-10K (~67K scenes) + 10M single images from BLIP-3o
- N in {4,...,8} views per scene, random Omega_c/Omega_n partition (biased toward small |Omega_c|)
- Resolution: 336 x 448
- Optimizer: AdamW, LR 1e-4 -> 1e-5 (linear decay)
- EMA decay: 0.9995, gradient clipping 1.0
- Classifier-free text drop rate: 0.2
- lambda_depth = 1.0, lambda_lpips = lambda_geo = 0.1
- t_th = 0.3 (gate perceptual + geometric losses)
- Geometry critic: pi^3 (frozen)
- Steps: ~200K
- Hardware: 32x NVIDIA A800-SXM4-80G GPUs
- Training: from scratch (no image/video model pretraining)

---

## Results

### Table 1: Novel-View Synthesis (Reconstruction)

| Method | RE10K 4v PSNR | RE10K 4v SSIM | RE10K 4v LPIPS | RE10K 8v PSNR | RE10K 8v SSIM | RE10K 8v LPIPS |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| MVSplat | 22.58 | 0.762 | 0.264 | 21.64 | 0.719 | 0.301 |
| DepthSplat | 25.16 | 0.832 | 0.194 | 27.77 | 0.872 | 0.154 |
| AnySplat | 20.07 | 0.731 | 0.286 | 20.52 | 0.752 | 0.262 |
| YoNoSplat | 25.86 | 0.841 | 0.143 | 28.35 | 0.889 | 0.107 |
| **PixWorld** | **26.21** | **0.844** | **0.138** | **28.58** | **0.892** | **0.101** |

| Method | DL3DV 4v PSNR | DL3DV 4v SSIM | DL3DV 4v LPIPS | DL3DV 8v PSNR | DL3DV 8v SSIM | DL3DV 8v LPIPS |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| MVSplat | 17.11 | 0.501 | 0.410 | 15.75 | 0.432 | 0.491 |
| DepthSplat | 20.38 | 0.719 | 0.320 | 19.26 | 0.692 | 0.360 |
| AnySplat | 20.11 | 0.671 | 0.318 | 20.02 | 0.664 | 0.327 |
| YoNoSplat | 22.89 | 0.710 | 0.228 | 21.92 | 0.678 | 0.262 |
| **PixWorld** | **23.18** | **0.714** | **0.226** | **22.46** | **0.681** | **0.257** |

PixWorld best PSNR and LPIPS everywhere. Best SSIM on RE10K; trails DepthSplat on DL3DV SSIM by 0.011 (8v).

### Table 2: 1-View Generation (averaged First Frame + Bidirectional)

| Method | RE10K PSNR | RE10K SSIM | RE10K LPIPS | RE10K AUC@5 | DL3DV PSNR | DL3DV SSIM | DL3DV LPIPS | DL3DV AUC@5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LVSM | 17.82 | 0.603 | 0.336 | 0.372 | 14.91 | 0.433 | 0.530 | 0.134 |
| GF | 15.63 | 0.553 | 0.454 | 0.290 | 12.69 | 0.356 | 0.591 | 0.113 |
| Gen3C | 17.26 | 0.624 | 0.391 | 0.334 | 15.58 | 0.514 | 0.479 | 0.128 |
| FlashWorld | 16.51 | 0.626 | 0.403 | 0.546 | 15.42 | 0.473 | 0.461 | 0.420 |
| Gen3R | 17.59 | 0.631 | 0.382 | 0.147 | 15.75 | 0.503 | 0.495 | 0.117 |
| **PixWorld** | **18.88** | **0.702** | **0.325** | **0.614** | **16.50** | **0.527** | **0.449** | **0.485** |

PixWorld tops every metric on both datasets. PSNR gain +1.06 dB RE10K, +0.75 dB DL3DV.

### Table 3: 2-View Generation (averaged Interpolation + Extrapolation)

| Method | RE10K PSNR | RE10K SSIM | RE10K LPIPS | RE10K AUC@5 | DL3DV PSNR | DL3DV SSIM | DL3DV LPIPS | DL3DV AUC@5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LVSM | 23.61 | 0.819 | 0.215 | 0.611 | 19.18 | 0.589 | 0.343 | 0.374 |
| GF | 18.27 | 0.647 | 0.353 | 0.223 | 15.38 | 0.459 | 0.470 | 0.147 |
| Gen3C | 20.12 | 0.714 | 0.300 | 0.255 | 17.62 | 0.542 | 0.412 | 0.176 |
| FlashWorld | 21.48 | 0.770 | 0.257 | 0.637 | 18.27 | 0.562 | 0.359 | 0.514 |
| Gen3R | 21.33 | 0.724 | 0.283 | 0.258 | 18.05 | 0.558 | 0.392 | 0.245 |
| **PixWorld** | **23.54** | **0.815** | **0.210** | **0.649** | **19.37** | **0.594** | **0.340** | **0.534** |

PixWorld best on all metrics. LVSM competitive on PSNR/SSIM only (gap < 0.07 dB RE10K).

### Table 4: WorldScore Benchmark

| Method | Camera Ctrl | Object Ctrl | Content Align | 3D Consist | Photo Consist | Style Consist | Subj Quality | Average |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan-2.1 | 23.53 | 40.32 | 45.44 | 78.74 | 78.36 | 77.18 | 59.38 | 57.56 |
| WonderJourney | 84.60 | 37.10 | 35.54 | 80.60 | 79.03 | 62.82 | 66.56 | 63.75 |
| LucidDreamer | 88.93 | 41.18 | 75.00 | 90.37 | 90.20 | 48.10 | 58.99 | 70.40 |
| FlashWorld | 84.43 | 50.28 | 56.54 | 85.87 | 86.72 | 79.36 | 52.75 | 70.85 |
| **PixWorld** | **91.08** | **46.25** | **55.27** | **91.39** | **93.84** | **67.11** | **52.36** | **71.04** |

PixWorld best average (71.04), best camera control (91.08), 3D consistency (91.39), photometric consistency (93.84).

### Table 5: Ablation -- Geometry Perception Loss (1-view, RE10K, 10K subset, 30K steps)

| Variant | PSNR | SSIM | LPIPS | AUC@30 | AUC@15 | AUC@5 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Full model | 19.12 | 0.717 | 0.310 | 0.886 | 0.813 | 0.642 |
| w/o Geometry Perception | 17.99 | 0.612 | 0.332 | 0.847 | 0.763 | 0.562 |
| **Delta** | **-1.13** | **-0.105** | **+0.022** | **-0.039** | **-0.050** | **-0.080** |

Removing L_geo: PSNR -1.13 dB, SSIM -0.105, AUC@5 -0.080 (~12.5% relative). VBench-style scores barely shift -- confirms L_geo targets 3D structure, not 2D appearance.

### Table 9: Inference Speed (A100-SXM4-80G)

| Method | Key Frames | NFE | Time/scene (s) |
|--------|:---:|:---:|:---:|
| Gen3C | 121 | 70 | 791 |
| Gen3R | 49 | 100 | 882 |
| FlashWorld | 24 | 4 | 10 |
| **PixWorld** | **8** | **100** | **15** |

PixWorld 15s/scene (8 key frames, 3DGS rendering). FlashWorld 10s but uses distillation (NFE=4). Note: not strictly apples-to-apples (different resolutions).

---

## Figures

- **Figure 1**: Overview diagram -- PixWorld vs latent-space methods (VAE/RAE bottleneck eliminated)
- **Figure 2**: Detailed architecture -- (a) Model overview, (b) Flow matching loss, (c) Geometry perception loss
- **Figure 3**: Visualization across reconstruction and generation settings
- **Figure 4**: Qualitative comparison with baselines
- **Figure 5**: Ablation visualization (with/without geometry perception loss)

---

## Implementation Complexity Assessment

- **Architecture complexity**: HIGH -- two-stream MMDiT + differentiable 3DGS renderer + frozen pi^3 critic
- **Compute requirements**: VERY HIGH -- 32x A800 GPUs, 200K steps, 1.04B params
- **Dependencies**: pi^3 pretrained model, DA3 for pseudo-depth, differentiable 3DGS renderer
- **Reproducibility**: MODERATE -- detailed architecture specs in Appendix B, but training from scratch requires massive compute; inference is feasible on single A100

---

## Verifiable Claims (for numeric verification)

1. PSNR gain over YoNoSplat: RE10K 4v = 26.21 - 25.86 = +0.35; 8v = 28.58 - 28.35 = +0.23; DL3DV 4v = 23.18 - 22.89 = +0.29; 8v = 22.46 - 21.92 = +0.54
2. 1-view PSNR gain over LVSM: RE10K = 18.88 - 17.82 = +1.06; DL3DV = 16.50 - 14.91 = +0.75 (matches prose)
3. AUC@5 1-view: RE10K 0.614 vs 0.546 (FlashWorld); DL3DV 0.485 vs 0.420 (FlashWorld)
4. Ablation delta: PSNR 19.12 - 17.99 = 1.13; SSIM 0.717 - 0.612 = 0.105; AUC@5 0.642 - 0.562 = 0.080
5. WorldScore average: (91.08+46.25+55.27+91.39+93.84+67.11+52.36)/7 = 497.30/7 = 71.04
6. Parameter count: 1.044B (Table 6 sums to 1.044B)
7. DL3DV SSIM: PixWorld 0.681 (8v) < DepthSplat 0.692 (8v) -- consistent with paper stating "trailing only DepthSplat on SSIM for DL3DV-10K"

---

## Honest-Scope Issues

1. **Massive compute**: 32x A800 GPUs for training -- not reproducible for most researchers
2. **No error bars**: No multiple runs or confidence intervals reported
3. **200 test scenes per dataset**: Relatively small evaluation set (200 RE10K, 200 DL3DV, 2000 WorldScore)
4. **Scene-level datasets only**: No object-centric or outdoor diversity evaluation
5. **Distillation not applied**: PixWorld runs undistilled (NFE=100), while FlashWorld uses NFE=4 distillation -- speed comparison favors FlashWorld
6. **Resolution difference**: PixWorld lower output resolution than video-diffusion baselines -- confounds speed comparison
7. **Ablation scale**: Only one ablation (geometry perception), trained on 10K-subset for 30K steps -- not the full model
8. **Single dataset mixture**: Re10K + DL3DV only (indoor/real estate scenes); generalization to outdoor/object datasets unknown
9. **Frozen pi^3 dependency**: Geometry perception loss requires a separate pretrained 3D foundation model
10. **BLIP-3o prior dependency**: 10M single images from external corpus to strengthen 2D appearance prior
