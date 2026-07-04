# InvSplat: Inverse Feed-Forward Scene Splatting — Source-First Breakdown

**arXiv:** 2607.02301v1 [cs.CV] (2 Jul 2026) · **Authors:** Polina Karpikova¹ · Wenjing Bian¹ · Haofei Xu¹,² · Hendrik Lensch¹ · Andreas Geiger¹ (¹University of Tübingen, Tübingen AI Center · ²ETH Zurich) · **Project:** https://poliik.github.io/invsplat/ · **Status:** Preprint · **PDF:** 48.7 MB, 19 pp (pdfinfo=19; `file` misreports 6 pp — 13-page gap, page-count defect recurs across recent iters)

**Repo role:** 69th paper, rank 64 unique. **FIRST inverse-rendering / PBR-material (albedo/metallic/roughness) / feed-forward-3DGS-with-intrinsic-materials / relighting paper** in the library. Sibling-in-spirit to `pointdit` (iter 74, monocular geometry) — both feed-forward geometry-from-images — but InvSplat adds **intrinsic material attributes** (albedo/metallic/roughness) + physically-based relighting, where PointDiT predicts only geometry (point maps). Distinct from the feed-forward-3DGS-for-RGB-NVS lineage (pixelSplat/MVSplat/ReSplat) which it extends into *inverse* (intrinsic-material) modelling.

---

## 1. Problem & Contribution

Inverse rendering = recover **3D geometry G AND physically meaningful intrinsic material M** from posed images, enabling relighting + NVS. Existing approaches trade off three axes (Table 1):

| Axis | Optimization (IRIS [2], Intrinsic Image Fusion [18]) | 2D image-space learning (DiffusionRenderer [10], DNF-Intrinsic [11], MVInverse [9]) | Feed-forward 3D RGB (pixelSplat/MVSplat/ReSplat) |
|---|---|---|---|
| Speed | slow per-scene fit | fast | fast (1 forward pass) |
| Multi-view consistency | ✓ (explicit 3D) | ✗/∼ (per-view maps drift) | ✓ (explicit 3D) |
| Intrinsic materials + relighting | ✓ | ✓ (but no 3D) | ✗ (baked RGB / SH) |

**Gap:** no method is simultaneously (a) feed-forward fast, (b) multi-view consistent, (c) intrinsic-material + relightable. InvSplat fills it: the **first feed-forward framework that predicts physically-based 3D Gaussian primitives with intrinsic material parameters** (albedo/metallic/roughness) from posed multi-view images in a single forward pass.

**Citable falsifiable hinge** — predicting normals from a **dedicated normal head** (vs finite-differences-from-depth or Gaussian-head normals) avoids leaking texture into geometry (Supp A.3 / Fig 8); without it, relighting quality degrades.

---

## 2. Method (§3, paper_layout L176–339)

### 2.1 Scene representation with intrinsic properties (§3.1)

3DGS [19] with M Gaussians, but appearance model replaced by **intrinsic materials** (no spherical harmonics):

$$G = \{(\mu_j, q_j, s_j, \sigma_j, n_j)\}_{j=1}^{M}, \quad M = \{(a_j, m_j, r_j)\}_{j=1}^{M} \quad \text{(Eq-system below Eq1)}$$

per Gaussian j: mean µ_j∈ℝ³, rotation quaternion q_j∈ℝ⁴, scale s_j∈ℝ³, opacity σ_j∈[0,1], **surface normal n_j∈ℝ³** (augmented), diffuse **albedo a_j∈[0,1]³**, **metallicity m_j∈[0,1]**, **roughness r_j∈[0,1]**. Predict all jointly in one forward pass:

$$(G, M) = f_\theta\!\left(\{I_i\}_{i=1}^{N}, \{P_i\}_{i=1}^{N}\right) \qquad \text{(Eq 1)}$$

### 2.2 Dual-branch feed-forward architecture (§3.2, Fig 2)

**Geometry branch** (ReSplat-style [16]): ResNet [23] multi-scale pyramid → **Multi-view Geometry Encoder** (transformer cross-view self-attn) → **Multi-view Feature Matching** (warp by poses → depth-candidate cost volume C_i). Shallow pyramid scales bypass matching → decoding heads.

**Intrinsic branch** (MVInverse-style [9]): frozen **DINOv2 [24] ViT-L/14** (register features) per-view patch features → **Multi-view Intrinsic Translator** (**36-block transformer**, alternates intra-/inter-view self-attn) → features {F^m_i}; uniformly-spaced translator layers {3,8,13,17} feed decoders (Supp A.1, L696).

**Decoding (6 heads):** depth d_i (DPT [25] head from {C_i, F^m_i, F^s_i}); rotation/scale/opacity via Point-Transformer [26] lift-to-3D + regress; albedo/metallic/roughness/normal = 4 DPT heads from {F^m_i}, albedo head adds skip-conn to {F^s_i} via 1×1 conv adaptor.

**Unprojection:** predicted depth → Gaussian centres {µ_j}; each Gaussian inherits its source pixel's (a,m,r); normal n_j = predicted camera-space normal rotated to world via extrinsics.

**Rendering:** differentiable Gaussian rasterizer projects Gaussians → albedo/metallic/roughness/normal/depth maps in **one rasterization pass**.

### 2.3 Training (§3.3, Eqs 2–5)

Overall: **L = L_a + L_m + L_r + L_d + L_n** (Eq 2).

Material (per X∈{a,m,r}): **L_X = Σ_i ‖X̂_i − X_i‖₁ + λ_LPIPS · LPIPS(X̂_i, X_i)** (Eq 3).

Affine-invariant depth loss (MoGe-style [27], dataset has depth but no camera params): **L_d = Σ_i (1/|Ω_i|) Σ_{p∈Ω_i} (δ_i(p) − δ̄_i)²** (Eq 4), δ_i(p)=log d̂_i(p)−log d_i(p).

Normal cosine loss: **L_n = Σ_i (1 − ⟨n̂_i, n_i⟩)** (Eq 5).

Loss weights (Supp A.2, L751): L1 albedo w=1.0, L1 metallic w=0.5, L1 roughness w=0.5, L1 depth w=1.0, L1 normal-consistency w=1.0, LPIPS (albedo/metallic/roughness) w=1.0 each. (Metallic/roughness are 1-channel → replicated 3× for LPIPS.)

---

## 3. Experimental Setup (§4, L340–438)

- **Train:** InteriorVerse [8], res 512×384, curated triplets (2 input views, all-3-view GT supervision incl. depth/normal/material). **13k triplets** after filtering.
- **Test:** InteriorVerse standard split, **137 pairs** (largest-overlap pair per scene, no ReSplat filtering → no selection bias).
- **Init:** geometry encoder ← ReSplat [16] (ResNet **frozen** during training, rest fine-tuned); appearance encoder + material heads ← MVInverse [9] (jointly fine-tuned).
- **Train config:** 20k steps, batch 2, single H100, ~12 h. AdamW (wd 0.01), linear warmup + cosine. LR 1e-5 (DINOv2 backbone + Translator, 1000 warmup) / 1e-6 (rest, 1000 warmup); MVInverse-FT LR 1e-5 (500 warmup).
- **Filtering:** InteriorVerse has no poses → COLMAP [33] pose estimation + depth-reprojection overlap>0.4 view selection; triplets filtered by ReSplat RGB PSNR>23 for training only.
- **Baselines (3, all 2D):** DiffusionRenderer [10] (video diffusion), DNF-Intrinsic [11] (single-view flow-matching diffusion), MVInverse [9] (feed-forward multi-view transformer) + MVInverse* (fine-tuned on InvSplat's training data). **Per-scene optimization baselines NOT included** (sparse views insufficient for geometry/material/lighting disentanglement, authors' justification).
- **Datasets:** InteriorVerse (train/test), Structured3D [28] (cross-view consistency), Infinigen [29] (relighting, synthetic GT light), RealEstate10K [30] + DL3DV [31] (real-world qualitative generalization).
- **Metrics:** material+normal quality on **input views** (following [9,11]); albedo scale-aligned per-channel before metric (albedo ambiguous up to scale); **cross-view consistency** = reprojection RMSE (predictions warped between views via GT depth+pose, RMSE on correspondences). Structured3D: depth-reprojection view selection overlap threshold **0.5**, first **251 scenes**.

---

## 4. Results — Tables Verbatim (with sourcing line-ranges)

### Table 1 — High-level method comparison (L93–104, conceptual; ✓/∼/× qualitative)

| Method | Paradigm | Consistent | NVS |
|---|---|---|---|
| IRIS [2] | optimization | ✓ | ✓ |
| Intrinsic Image Fusion [18] | optimization | ✓ | ✓ |
| DiffusionRenderer [10] | diffusion | ∼ | × |
| DNF-Intrinsic [11] | diffusion/flow | ∼ | × |
| MVInverse [9] | feed-forward | ∼ | × |
| **InvSplat (Ours)** | feed-forward | ✓ | ✓ |

*"Consistent" = multi-view consistency; ∼ = partial/limited; × = not supported by design.* InvSplat = only row that is feed-forward AND consistent AND NVS-capable.

### Table 2 — Inverse rendering on InteriorVerse, 2 input views (L357–369) ⚠ see honest-scope flag

| Method | Type | Albedo PSNR↑ | SSIM↑ | LPIPS↓ | Metallic RMSE↓ | Roughness RMSE↓ | Normal RMSE↓ | Normal Cos↑ |
|---|---|---|---|---|---|---|---|---|
| DiffusionRenderer [10] | 2D | 17.32 | 0.800 | 0.253 | 0.1506 | 0.2971 | 0.2825 | 0.9468 |
| DNF-Intrinsic [11] | 2D | 18.64 | 0.850 | 0.211 | 0.1320 | 0.1884 | 0.2124 | 0.9261 |
| MVInverse [9] | 2D | 21.83 | 0.867 | 0.217 | 0.0887 | 0.1039 | 0.1252 | 0.9654 |
| MVInverse* | 2D | 22.92 | 0.886 | 0.182 | 0.0798 | 0.0985 | 0.1221 | 0.9630 |
| **Ours** | 3D | 22.18 | 0.873 | 0.203 | 0.0883 | 0.0993 | 0.1254 | 0.9609 |

### Table 3 — Multi-view consistency + albedo reconstruction on Structured3D, 2 input views (L410–419)

| Method | Type | Reprojection RMSE↓ (Albedo / Metallic / Roughness) | Albedo Recon PSNR↑ / SSIM↑ / LPIPS↓ / RMSE↓ |
|---|---|---|---|
| DiffusionRenderer [10] | 2D | 0.100 / 0.122 / 0.108 | 15.75 / 0.714 / 0.310 / 0.174 |
| DNF-Intrinsic [11] | 2D | 0.122 / 0.183 / 0.147 | 14.37 / 0.703 / 0.303 / 0.209 |
| MVInverse [9] | 2D | 0.044 / 0.056 / 0.038 | 19.83 / 0.771 / 0.268 / 0.108 |
| MVInverse* [9] | 2D | 0.037 / 0.051 / 0.034 | 20.48 / 0.798 / 0.247 / 0.101 |
| **Ours** | 3D | 0.039 / 0.041 / 0.025 | 19.84 / 0.783 / 0.269 / 0.109 |

### Table 4 — Ablation: 3D model design on InteriorVerse, 2 input views (L504–510) ⚠ see honest-scope flag

| Method | Type | Albedo PSNR↑ / SSIM↑ / LPIPS↓ | Metallic RMSE↓ | Roughness RMSE↓ | Normal RMSE↓ / Cos↑ |
|---|---|---|---|---|---|
| MVInverse + ReSplat* | 3D | 20.83 / 0.860 / 0.234 | 0.1011 | 0.1041 | 0.1291 / 0.9582 |
| MVInverse* + ReSplat* | 3D | 21.86 / 0.873 / 0.212 | 0.0901 | 0.1011 | 0.1283 / 0.9607 |
| **Ours** | 3D | 22.18 / 0.873 / 0.203 | 0.0883 | 0.0993 | 0.1254 / 0.9609 |

---

## 5. Source-Free Reconciliation (Python-verified)

**Cross-table byte-identity ✓:** Table-2 Ours row (22.18, 0.873, 0.203, 0.0883, 0.0993, 0.1254, 0.9609) == Table-4 Ours row — byte-identical (same model, same InteriorVerse 2-view eval).

**Table-2 reconstruction ("comparable to 2D baselines") — every cell recomputes, but the framing is the story:**
- Ours Albedo PSNR 22.18 vs MVInverse* 22.92 = **−0.74** ("slightly worse than fine-tuned MVInverse", prose L398–399 ✓).
- Ours Albedo PSNR 22.18 vs MVInverse (orig) 21.83 = **+0.35** (Ours actually HIGHER than the *un*-fine-tuned 2D baseline).
- Ours is **WORSE than MVInverse* on all 7 Table-2 cells**: PSNR −0.74, SSIM −0.013, LPIPS +0.021, metallic +0.0008, roughness +0.0033, normal RMSE +0.0033, normal cos −0.0021. The reconstruction-quality win is consistency/3D-representation-carried, **not** raw per-pixel quality.

**Table-3 consistency ("better multi-view consistency, especially metallic and roughness") ✓ verified:**
- Reprojection Metallic: Ours 0.041 < MVInverse* 0.051 (−0.010) < MVInverse 0.056 → Ours best.
- Reprojection Roughness: Ours 0.025 < MVInverse* 0.034 (−0.009) < MVInverse 0.038 → Ours best.
- Reprojection Albedo: Ours 0.039 **> MVInverse* 0.037** (Ours worse than FT on albedo reprojection; better only vs orig MVInverse 0.044). → "better consistency" is metallic+roughness-carried, not albedo.
- Albedo Recon PSNR: Ours 19.84 ≈ MVInverse 19.83 (+0.01, "same reconstruction quality" vs orig ✓); −0.64 vs MVInverse*.

**Table-4 ablation ("consistently outperforms both baselines across all material factors") — 6/7 cells strictly better, 1 TIE:**
- Ours beats both naive-3D baselines on: Albedo PSNR (+1.35/+0.32), LPIPS, metallic RMSE, roughness RMSE, normal RMSE, normal cos.
- **Albedo SSIM: Ours 0.873 = MVInverse*+ReSplat* 0.873 — TIE, not "outperforms".** The blanket "across all material factors" claim is violated on this one cell (iter-72/74/80 attribution-overstatement class).
- Unified-architecture lift over the best naive-3D (FT) variant: +0.32 PSNR, −0.009 LPIPS, −0.0018 metallic, −0.0018 roughness, −0.0029 normal RMSE, +0.0002 cos.

**No numeric prose-vs-table typo found** — every cited number and ratio recomputes from the displayed cells.

---

## 6. Honest-Scope Flags (⚠)

1. **Table-4 "consistently outperforms both baselines across all material factors" is violated by the Albedo-SSIM TIE** (Ours 0.873 = MVInverse*+ReSplat* 0.873); 6/7 metrics strictly better, but "all" overstates one cell. (Attribution-overstatement class — parallel iter-72 MARVEL / iter-74 PointDiT.)
2. **"Comparable to 2D baselines" understates a uniform reconstruction regression:** on Table 2 InvSplat is WORSE than the fine-tuned MVInverse* on **all 7 cells** (albedo PSNR/SSIM/LPIPS, metallic RMSE, roughness RMSE, normal RMSE/cos). The genuine contribution is **multi-view consistency (Table 3) + an explicit 3D representation enabling NVS/relighting**, NOT raw reconstruction fidelity. Cite the consistency delta, not "comparable quality".
3. **Table-3 "better multi-view consistency" is metallic+roughness-carried:** albedo reprojection RMSE Ours 0.039 > MVInverse* 0.037 (Ours worse). The headline-honest subset is metallic + roughness (where Ours beats both MVInverse variants).
4. **NVS + relighting are QUALITATIVE-ONLY** — no quantitative NVS PSNR/SSIM/LPIPS table on novel views; Figure 4 (RealEstate10K), Figure 6 (DL3DV), Figure 7 (relighting), Figure 10/11 (RGB reconstruction) are figure-only. The "stable novel view rendering" claim rests on figures, not metrics.
5. **No per-scene-optimization baseline** despite Table 1 listing IRIS [2] + Intrinsic Image Fusion [18] as the also-consistent+also-NVS+also-materials alternatives. Authors exclude them (sparse-view-insufficient justification) → no head-to-head vs the strongest conceptual competitors; comparison is only vs weaker 2D feed-forward/diffusion baselines (iter-80 Zeus subset-scoping class).
6. **No seeds / CIs anywhere.** Decisive deltas are tiny: T3 albedo reproj 0.039 vs 0.037 (=0.002); T4 normal cos 0.9609 vs 0.9607 (=0.0002); T4 albedo PSNR +0.32 over FT-naive. Several "wins" sit well within plausible run noise.
7. **"1.5 s reconstruction" (Figure 1) is figure-only timing** — not reported in any table or the text body; unverifiable from source.
8. **Generalization claims on real data are qualitative** (RealEstate10K, DL3DV — Figures 4/6/12/13); no real-world quantitative metric. "Generalizes to more input views at inference" (4-view DL3DV) is also figure-only.
9. **Single training dataset (InteriorVerse, synthetic indoor)** — Structured3D is also synthetic indoor; real-world (RealEstate10K/DL3DV) only qualitative. Outdoor / non-indoor / complex-illumination regimes untested.
10. **Relighting is a "simple point-light renderer based on a standard BRDF shader" (Disney [32])** — not a full path tracer; specular-highlight faithfulness argued qualitatively (Fig 10/11), not quantified.
11. **Affine-invariant albedo scale-alignment** is applied "consistently across all compared methods" (good), but PSNR on a scale-fitted albedo is an optimistic ceiling vs absolute albedo recovery.
12. **Inherited limitations (Supp A.5):** sensitive to input pose errors (Fig 9 failure case); performance degrades with more views / different resolutions; **non-generative** (no generative prior for ambiguous regions, unlike diffusion baselines).

---

## 7. Strengths / Limitations / Verdict

**Strengths:** (1) Genuinely first feed-forward + multi-view-consistent + intrinsic-material + relightable 3DGS — fills the Table-1 gap cleanly. (2) Unified dual-branch architecture removes the redundant backbone of a naive MVInverse+ReSplat stack while matching/beating it (Table 4). (3) Consistency wins on metallic+roughness (Table 3) are real and align with the "explicit 3D rep" motivation. (4) Honest Table-2 caption concedes the reconstruction-quality gap vs 2D baselines rather than hiding it.

**Limitations:** uniform reconstruction regression vs the FT 2D baseline (Table 2); NVS/relighting claims unquantified; no optimization-baseline head-to-head; no seeds/CIs with sub-0.01 deltas; single synthetic-indoor training set; figure-only timing + real-world generalization.

**Verdict — citable falsifiable content:** the **dual-branch unified architecture that predicts intrinsic-material 3D Gaussians in one forward pass** (Eq 1, §3.2) + the **consistency gains on metallic/roughness** (Table 3) + the **dedicated-normal-head design choice** (Supp A.3). The "comparable reconstruction quality" and "stable NVS" framings should be cited as **consistency/representation-carried, not quality-carried** (Table 2 shows a uniform regression vs the FT 2D baseline; NVS is figure-only).
