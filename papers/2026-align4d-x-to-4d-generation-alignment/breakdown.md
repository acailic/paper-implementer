# Align4D — Alignment Is All You Need For X-to-4D Generation

**arXiv 2607.02516v1** (cs.CV, 2 Jul 2026) — Qiaowei Miao, Kehan Li, Yawei Luo*, Yi Yang (Zhejiang University). Repo's 75th paper, rank 70. First **X-to-4D (arbitrary-modality-to-4D) generation** paper + first **object-distance alignment** paper in the library. PDF 12.7 MB, **14 pp by pdfinfo** (file misreports 8 pp — 6-page file-vs-pdfinfo gap defect recurs). `paper_layout.txt` = 1372 lines.

## Verbatim abstract

> Generative diffusion models excel at synthesizing high-quality images, videos, and 3D content under multimodal control. However, arbitrary user-defined modality-to-4D (X-to-4D) generation remains challenging due to the high cost of constructing diverse datasets and the limited scalability of existing methods. This paper presents Align4D, a flexible framework that translates any-modal input into coherent video–3D pairs, using video to guide 4D motion and 3D data to shape 4D geometry. Align4D introduces three key techniques: (1) Object Distance Alignment, which searches Video-Aligned and Multiview-Aligned Object Distances (VAOD/MAOD) respectively, to reconcile 4D renderings to video and the priors of multiview diffusion models; (2) Motion-Geometry Joint Alignment, which constrains known and unknown views through synchronized video and 3D inputs, ensuring consistent 4D generation; and (3) Asynchronous Optimization, which decouples Gaussian attribute and deformation network training to enhance motion and geometry fidelity. We further propose the X4D dataset, integrating prompt, image, video, and 3D data for benchmarking. Experiments on X4D and Consistent4D demonstrate that Align4D achieves state-of-the-art quality and consistency in X-to-4D generation.

## Core idea (source-first)

The hinge is reframing **X-to-4D generation as an ALIGNMENT problem, not a from-scratch training problem**. Instead of training an end-to-end 4D model, Align4D sequences off-the-shelf pretrained diffusion models to first turn any input modality into a (video, 3D) pair, then **aligns** the 4D asset's *temporal motion* with the video prior and its *spatial geometry* with the 3D prior via matched **object distances**. The 4D representation is 3DGS + a deformable motion field, optimized by Score Distillation Sampling (SDS).

Three modules:
- **ODA (Object Distance Alignment).** Searches two scalar camera-to-object distances: VAOD (aligns 4D front-view renderings to the input video, via global-min MSE over a swept distance grid) and MAOD (aligns multiview renderings to the multiview-diffusion prior, via a *local* SDS-loss minimum that lies to the left of VAOD — the global SDS min is a blank-background artifact). Eq 2 (pinhole `w/W = f/d ⇒ d = Wf/w`), Eqs 3–5.
- **MGJA (Motion-Geometry Joint Alignment).** Uses a *single* multiview diffusion model to transfer both motion (from video frames) and geometry (from 3D renderings) to unknown/non-frontal viewpoints. Eqs 6–10. Splits into Known-Spatiotemporal-Viewpoint Alignment (KSVA, front view, MSE + mask, Eq 6) and Unknown-Spatiotemporal-Viewpoint Alignment (USVA = temporal-gradient-weighted `λ·Lmot + (T−t)/T·λ·Lgeo`, Eq 9).
- **AO (Asynchronous Optimization).** Alternately freezes 3DGS vs the deformation network while optimizing the other, vs synchronous Joint Optimization (JO). Mitigates component interference.

## Method / Equations (paper_layout lines)

- **Eq 1** (L265): SDS gradient `∇_θ L_SDS = E_{τ,ε}[w(τ)(ε̂_ϕ(z,v,τ) − ε) ∂x/∂θ]`.
- **Eq 2** (L334): pinhole `w/W = f/d`, `d = Wf/w`.
- **Eq 3** (L367): distance ratio `d_v/d_s = w_s/w_v` (target-size ratio ⇒ matched distance).
- **Eq 4** (L389): VAOD `d_VA = argmin_{d'} ||x_θ^1 − I_1||²` (global MSE min).
- **Eq 5** (L510): modified SDS loss for MAOD search `L_SDS = (1/|C||T|) Σ_{τ∈T} Σ_{c∈C} w(τ)||ε_ϕ(z^{c,1}_θ; I_1, c, d', τ) − ε||²`, `z = α_τ x + σ_τ ε_fix`, timesteps **T = {700,800,900}**.
- **Eq 6** (L596): KSVA `L_KSVA = (1/T)Σ_t ||x^{c'}_{θ_t} − I_t||² + ||m^{c'}_{θ_t} − M_t||²`.
- **Eq 7** (L626): Lmot (motion transfer to N random viewpoints) `L_mot = (1/N)Σ_n w(τ)||ε_ϕ(α_τ x^{c_n}_{θ_t}+σ_τ ε; I_t, c_n, d_VA, τ) − ε||²`.
- **Eq 8** (L645): Lgeo (geometry prior from same-view 3D renderings) `L_geo = (1/N)Σ_n w(τ)||ε_ϕ(α_τ x^{c_n}_{θ_t}+σ_τ ε; x^{c_n}_ψ, 0, d_MA, τ) − ε||²`.
- **Eq 9** (L663): USVA `L_USVA = (1/T)Σ_t [(t/T)λ L_mot + ((T−t)/T)λ L_geo]` (temporal gradient coefficient `(T−t)/T·λ`).
- **Eq 10** (L668): `L_MGJA = L_KSVA + L_USVA`.

Implementation (L636–662): dense percentage 0.1, densification interval 100, densification gradient threshold 0.05; object-distance search `d_min=0.00001`, `d_max=3.00001`, **interval 0.05 ⇒ 61 grid points**; N=4 viewpoints, azimuth [−90°, 0°, 90°, 180°]; Zero123 multiview model; NVIDIA V100 32 GB. DG4D baseline uses fixed object distance 1.5.

## TABLE I — X4D quantitative results (paper_layout L692–705)

Human Evaluation (4 dims, **each column sums to exactly 100% across the 5 methods ⇒ forced-choice "select best" vote**) + VBench (4 dims).

| Method | Appearance%↑ | Structure%↑ | Motion%↑ | Fidelity%↑ | Subj.Consist↑ | Bkg.Consist↑ | Aesthetic↑ | Imaging↑ |
|---|---|---|---|---|---|---|---|---|
| L4GM [8] | 9.2 | 5.5 | 7.9 | 6.6 | 0.80 | 0.88 | 0.46 | 0.40 |
| SC4D [10] | 5.3 | 5.3 | 4.0 | 5.3 | 0.81 | 0.91 | 0.43 | 0.38 |
| STAG4D [48] | 19.7 | 17.1 | 26.3 | 18.4 | 0.83 | 0.91 | 0.50 | 0.41 |
| DG4D [7] (baseline) | 4.0 | 2.6 | 3.9 | 3.9 | 0.71 | 0.85 | 0.40 | 0.31 |
| **Align4D (ours)** | **61.8** | **69.5** | **57.9** | **65.8** | **0.85** | **0.93** | **0.55** | **0.43** |

Source-free check (Python): all 4 human-eval columns sum to **100.0** exactly (Appearance 9.2+5.3+19.7+4.0+61.8=100.0; Structure=100.0; Motion=100.0; Fidelity=100.0). Align4D wins all 8 columns. Margins over 2nd place: Appearance 61.8 vs 19.7 (**3.14×**), Structure 69.5 vs 17.1 (**4.06×**), Motion 57.9 vs 26.3 (2.20×), Fidelity 65.8 vs 18.4 (3.58×). With 30 participants, 61.8% ≈ **18.5 votes**.

## TABLE II — Consistent4D quantitative results (paper_layout L708–733)

6 metrics: PSNR↑ SSIM↑ LPIPS↓ FVD↓ CLIP↑ CLIP-F↑. `*`=video from first+last training frame; `+`=first+last test frame. Best in bold.

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | CLIP↑ | CLIP-F↑ |
|---|---|---|---|---|---|---|
| CRM [69]+Rigging | 13.5 | 0.88 | 0.17 | 1258.5 | 0.79 | – |
| Meshyai [22]+Rigging | 12.7 | 0.87 | 0.19 | 1179.2 | 0.85 | – |
| Kling∗ [59] | 11.0 | 0.81 | 0.28 | – | 0.81 | – |
| Kling+ [59] | 11.4 | 0.80 | 0.26 | – | 0.78 | – |
| 4Diffusion [68] | 12.2 | 0.84 | 0.26 | 1522.9 | 0.83 | 0.912 |
| Free4D [67] | 6.4 | 0.45 | 0.41 | 2513.6 | 0.77 | 0.809 |
| L4GM [8] | 14.2 | 0.84 | 0.20 | 1217.1 | 0.90 | 0.990 |
| SC4D [10] | 16.8 | 0.86 | 0.16 | 1132.2 | 0.91 | 0.988 |
| 4DGen [70] | 12.7 | 0.87 | 0.19 | 1258.5 | 0.80 | 0.981 |
| Efficient4D [9] | 12.8 | 0.85 | 0.21 | 1304.2 | 0.89 | 0.954 |
| STAG4D [48] | 17.0 | 0.87 | 0.14 | 1251.7 | 0.90 | 0.988 |
| DG4D [7] (baseline) | 10.7 | 0.78 | 0.28 | 1262.0 | 0.89 | 0.978 |
| **Align4D (ours)** | **17.8** | **0.90** | **0.11** | **1088.9** | **0.94** | **0.992** |

Source-free check: Align4D is best on **all 6** metrics (PSNR 17.8 > STAG4D 17.0; CLIP-F 0.992 > L4GM 0.990; FVD 1088.9 < SC4D 1132.2 ⇒ 3.82% relative gain). ⚠ Prose (§IV.B, L820) says "Align4D attains the best performance across **five** key metrics" — but the table has **six** columns and Align4D wins all six; minor wording slip (likely CLIP-F or FVD undercounted in prose).

## TABLE III — Ablation on Consistent4D (paper_layout L736–755)

5 metrics (no CLIP-F). Rows build on STAG4D as the base architecture.

| Method (= module set) | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | CLIP↑ |
|---|---|---|---|---|---|
| STAG4D [48] (base) | 17.0 | 0.87 | 0.14 | 1251.7 | 0.90 |
| STAG4D + ODA + MGJA | 17.5 | 0.88 | 0.12 | 1109.4 | 0.93 |
| STAG4D + AO | 17.2 | 0.88 | 0.14 | 1129.2 | 0.91 |
| DG4D [7] (baseline) | 10.7 | 0.78 | 0.28 | 1262.0 | 0.89 |
| Align4D W/O. ODA | 15.9 | 0.86 | 0.16 | 1111.5 | 0.90 |
| Align4D W/O. MGJA | 15.5 | 0.85 | 0.17 | 1113.1 | 0.91 |
| Align4D W/O. AO | 17.6 | 0.89 | 0.13 | 1139.7 | 0.93 |
| **Align4D (ours)** | **17.8** | **0.90** | **0.11** | **1088.9** | **0.94** |

Cross-table byte-identity (Python): Align4D, STAG4D, DG4D rows are **byte-identical** between Table II (first 5 cols) and Table III — strong transcription-consistency witness. ⚠ **Co-dependent ablation pair** (see Honest-scope #1): W/O ODA (15.9) and W/O MGJA (15.5) both fall **BELOW the STAG4D base (17.0)** the method is built on — removing either of the {ODA, MGJA} pair is worse than using neither.

## TABLE IV — AO vs JO loss reduction (paper_layout L784–799)

| Step | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| Align4D + JO (synchronous) | 72.3 | 174.2 | 155.2 | 44.0 | 3.5 |
| Align4D + AO (asynchronous) | 34.4 | 87.7 | 85.7 | 24.8 | 3.2 |

Source-free check: AO loss < JO loss at **every** step (ratios 2.10×, 1.99×, 1.81×, 1.77×, 1.09×). Both curves are non-monotonic (hump at step 200). AO converges to 3.2 vs JO 3.5.

## TABLE V — Runtime (paper_layout L796–810)

| Method | Time (Min.) | VRAM (GB) |
|---|---|---|
| L4GM [8] | 1.5 | 24.6 |
| SC4D [10] | 35 | 8.9 |
| STAG4D [48] | 80 | 12.2 |
| DG4D [7] | 15 | 15.6 |
| Align4D (ours) | 25 | 19.4 |
| ODA-VAOD | 0.0025 | 2.7 |
| ODA-MAOD | 0.20 | 4.5 |

Source-free check: ODA search overhead = 0.2025 min = **0.81%** of Align4D's 25 min (VAOD 0.15 s + MAOD 12 s) — "low overhead" claim holds. Align4D is **3.20× faster than STAG4D** (80→25 min) but **1.67× SLOWER than DG4D** (15 min). VRAM 19.4 GB is the **2nd-heaviest** (only L4GM 24.6 higher).

## TABLE VI — User preference by (3D model × video model) (paper_layout L1128–1145)

| 3D model \ Video model | Kling [59] | VideoCrafter [17] |
|---|---|---|
| Meshai [22] | 27.4% | 22.5% |
| Tripo3d [75] | 26.5% | 23.6% |

Source-free check: sum = **100.0** exactly (forced-choice). Kling total 53.9% > VideoCrafter 46.1%; Meshai 49.9% ≈ Tripo3d 50.1% (near-tie).

## Figures (paper_layout)
Fig 1 teaser (L35); Fig 2 modality pathways (L95); Fig 3 overview (L231); Fig 4 ODA search (L296); Fig 5 VAOD/MAOD search curves — axis ticks at 0.15 spacing (L489, figure-only); Fig 6 SDS-loss vs object distance per timestep bucket τ∈{100-300}/{400-600}/{700-900} (L493); Fig 7 X4D samples (L576); Fig 8 qualitative vs baselines (L662); Fig 9 ablation qualitative (L779); Fig 10 text-to-4D comparison (L886); Fig 11 MAOD selections (L873); Fig 12 object-distance effect (L926); Fig 13 different multiview diffusion models (Zero123/ImageDream) (L1019); Fig 14 mask effect (L1054); Fig 15 seed diversity (L1046); Fig 16 failure case — transparent/neon materials (L1162).

## Strengths
- **Clean falsifiable hinge**: alignment (not from-scratch training) as the X-to-4D paradigm; ODA turns a previously *fixed/heuristic* object distance (DG4D's 1.5) into a *searched* scalar with a clear global-min (VAOD, MSE) vs local-min-before-VAOD (MAOD, SDS) rule. The MAOD "global min is a blank-background artifact, use the local min to the left of VAOD" insight (Fig 4/5) is a genuinely non-obvious, citable design rule.
- **Strong cross-table byte-consistency**: Align4D/STAG4D/DG4D rows byte-identical across Tables II & III; Table I + VI preference columns sum to exactly 100 (forced-choice invariant). Zero numeric cell typos on Python reconciliation.
- **Cheap ODA**: 0.81% of total runtime; the searched distances add negligible cost.
- **Honest failure-case reporting** (Fig 16): transparent/neon materials documented as a limitation.
- **Module migratability**: ODA+MGJA+AO ported into STAG4D (Table III rows), not just demonstrated in the authors' own pipeline.

## Honest-scope issues (12, all attribution/framing — NO numeric cell typo)

1. ⚠ **Co-dependent ablation pair — each-alone-is-below-baseline (NEW subclass).** Removing ODA (PSNR 15.9) OR MGJA (15.5) drops Align4D **below the STAG4D base (17.0)** it builds on — i.e. either module *alone* is worse than having neither. Direct witness: STAG4D+AO (17.2) → add ODA without MGJA (= "W/O MGJA") = 15.5, a **−1.7 pp** drop. The paper's framing ("ODA is a prerequisite for robust MGJA", "every component helps") is asymmetric and omits that **ODA without MGJA is actively harmful**; the truthful claim is "ODA and MGJA are co-dependent — use both or neither." Also note W/O AO (17.6) ≈ STAG4D+ODA+MGJA (17.5), i.e. AO contributes only **+0.2 pp** PSNR despite its central billing.
2. ⚠ **Forced-choice user study inflates the leader.** Table I (and Table VI) columns each sum to exactly 100% across methods ⇒ participants pick *one* best per dimension, not independent ratings. Align4D's 61.8/69.5/57.9/65.8% (= 18.5/30 votes on Appearance) are 2.2–4.06× the runner-up — a margin unusual for 4D user studies, amplified by forced-choice. No blinding / randomization protocol stated; Align4D is the authors' own method.
3. ⚠ **Selective baseline exclusion.** Text-to-4D [2] is dropped "due to substantial differences in motion and geometric control"; the standard 3D-to-4D method [6] is dropped as "impractical due to extreme memory demands" (replaced by CRM/MeshyAI+rigging). Both are strong/standard X-to-4D peers; the comparison set is curated.
4. ⚠ **"Five key metrics" wording slip** (§IV.B): prose claims best on "five" metrics but Table II has six and Align4D wins all six. Minor, but the count is wrong.
5. ⚠ **VRAM cost understated.** "Balanced computational efficiency" — but Align4D's 19.4 GB VRAM is the **2nd-heaviest** of 5 methods (only L4GM 24.6 higher; SC4D 8.9 / STAG4D 12.2 / DG4D 15.6 all lower). On wall-clock, Align4D is **1.67× slower** than DG4D (15 min). Efficiency advantage is only vs STAG4D/L4GM.
6. ⚠ **Synthetic-data circularity.** X4D is built from SDXL/SVD/LGM outputs and the paper admits it "inherits statistical biases" from them (§IV intro). Metrics PSNR/SSIM/LPIPS/FVD are computed against references that are *themselves generated* by the same upstream diffusion family Align4D uses internally — so the metrics measure consistency with synthetic ground truth, not real-world fidelity. Consistent4D (real video) partially mitigates this; X4D results (Table I) do not.
7. ⚠ **AO convergence is loss-based, not quality-based.** Table IV shows AO training-loss < JO at every step, but lower loss ≠ better perceptual quality; Table III confirms AO adds only +0.2 pp PSNR (W/O AO 17.6 → full 17.8). "Finer and more stable convergence" rests on a loss curve, not a quality delta.
8. ⚠ **Module migratability is STAG4D-only.** "Successfully migrate our modules to STAG4D" — generalization of ODA/MGJA/AO is demonstrated on a single target architecture (the base), not on L4GM/SC4D/DG4D/Efficient4D. Single-target migration ≠ architecture-agnostic.
9. ⚠ **No confidence intervals / single-seed quantitative tables.** Table II/III deltas are ≤0.8 pp PSNR (STAG4D 17.0 → Align4D 17.8) with no SE/CIs; seed variation is shown only qualitatively (Fig 15). Several "wins" are within plausible seed noise.
10. ⚠ **ODA search grid is paper-fixed, not adaptive.** VAOD/MAOD use a fixed 61-point grid (`d∈[1e-5, 3.00001]`, step 0.05) + fixed 4 viewpoints + fixed timesteps {700,800,900}. No sensitivity analysis to grid resolution, viewpoint count, or timestep choice; the MAOD "local min to the left of VAOD" rule is hand-specified (Fig 5) and could mis-fire when the local min is absent.
11. ⚠ **User-study participant count is small (n=30)** and unstratified; no inter-rater agreement / Krippendorff α reported. The 5-way forced-choice with n=30 leaves ~6 votes per non-leader method per dimension.
12. ⚠ **MAOD timestep rationale is empirical only.** The choice τ∈{700,800,900} (large-τ) is justified by Fig 6 curve stability, but the link to "noise schedule coefficients w(τ)" is asserted, not derived; the ablation does not isolate timestep choice vs the local-min rule.

## Verdict
Solid, well-engineered X-to-4D framework with a clean citable hinge (alignment-as-generation + searched object distances) and impeccable cross-table numeric consistency (byte-identical rows, sum-to-100 preference invariants, zero cell typos). The headline SOTA claims are **directionally** supported but the two load-bearing caveats are (a) the **co-dependent {ODA, MGJA} ablation pair** where each-alone drops below baseline, contradicting the "every component helps" framing, and (b) the **forced-choice, unblinded, n=30 user study** carrying Table I. Treat the Consistent4D (real-video) numbers as the trustworthy SOTA evidence; treat the X4D (Table I) human-eval margins as suggestive, not decisive.
