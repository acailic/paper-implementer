# WBMM: Windowed Batch Matrix Multiplication for Efficient Large Receptive Field Convolution

**arXiv:** 2607.02097v1 (cs.CV, 2 Jul 2026) | **Venue:** ICML 2026 (PMLR 306) | **Repo rank:** 70th paper, rank 65 (first efficient-conv-OPERATOR / hardware-aware-kernel / large-receptive-field-conv paper)
**Authors:** Wan Song¹, Wei Zhou², Rui Wang², Jun Yu³, Toru Kurihara³, Jiajia Xu², Shu Zhan¹ (¹Hefei University of Technology; ²Lingyang Industrial Internet Co., Hefei; ³Kochi University of Technology)
**Source:** paper.pdf (8.7MB, 23pp — `file` AND `pdfinfo` BOTH 23pp, NO page-count defect this iter), paper_layout.txt (pdftotext -layout, 1515 lines)
**Code:** https://github.com/wansong-s/WBMM

---

## 1. The falsifiable hinge (one paragraph)

Large-kernel depthwise convolutions (ConvNeXt 7×7 → RepLKNet 31×31 → SLaK 51×51 → PeLK 101×101 → UniRepLKNet 13×13) are **memory-bound**: each output gathers k² scattered neighbors from non-contiguous rows, blowing the L1/L2 cache and yielding only O(1) FLOPs per loaded element. The standard accelerations don't fix this — im2col+GEMM pays O(k²) memory expansion; cuDNN implicit-GEMM still gathers; and RepLKNet/UniRepLKNet's **Large Kernel Acceleration (LKA)** CUDA kernels use fixed tiling tuned for SMALL feature maps that no longer matches large maps (at 224×224 batch=128, LKA 7×7 is **80% slower** and LKA 13×13 is **89% slower** than the un-accelerated DW-Std 5×5 baseline, Table 13). **WBMM inverts the paradigm: traverse the PARAMETER table, not the data.** Partition the input into contiguous non-overlapping w×w windows; construct a per-channel weight matrix M∈ℝ^{C×d×d} (d=w²) by indexing a compact relative-position-bias table R∈ℝ^{C×(2w−1)²}; apply via batched matrix multiplication on contiguous blocks. Because M is **batch-independent** (built once at O(C·d²), shared across all B·N_h·N_w windows, cacheable at inference), WBMM shifts the operator from memory-bound to compute-bound: its throughput **improves** with larger windows (opposite to depthwise conv) and with larger feature maps (opposite to LKA). Operator-level (Table 13): WBMM-C 14×14 gives a **7.8× larger per-layer receptive field** (196 vs 25 positions) than DW-Std 5×5 while matching/beating its speed on feature maps ≥28×28 at batch ≥16, peaking at **+272% (3.72×) at batch=256 on 14×14 maps**.

## 2. Method (Eqs + Algorithm, sourced)

- **Eq 1 (L259–264)** standard depthwise conv: `Y[b,c,h,w] = Σ_{i,j} K[c,i+k_h,j+k_w]·X[b,c,h+i,w+j] + β_c`, kernel (2k_h+1)×(2k_w+1). Gather of (2k_h+1)(2k_w+1) scattered inputs is the cost source.
- **Theorem 3.1 (L222–231)** max effective kernel covering all valid relative offsets on H×W = (2H−1)×(2W−1). [δ_h∈[−(H−1),H−1], δ_w∈[−(W−1),W−1].]
- **Theorem 3.2 / Eq 2 (L232–264)** convolution–matrix equivalence: `y_c = x_c·M_c + β_c·1ᵀ`, where `x_c∈ℝ^{B×HW}` is the flattened input for channel c and `M_c∈ℝ^{HW×HW}` is a block-Toeplitz position-dependent matrix whose entry depends only on the relative offset (δ_h,δ_w). Exact globally but impractical (at 56×56, M_c carries HW=3136 rows, ~9.8M entries/channel) → **windowed approximation** (Swin-style locality): apply Eq 2 inside non-overlapping w×w windows, shrinking M to ℝ^{d×d}, d=w² (49×49 for 7×7, 196×196 for 14×14).
- **Eq 3 (L269)** the offset-substituted full-range sum (every added term vanishes via K=0 or X=0 padding).
- **Eq 4/5 (L316–317)** zero-padding to make H,W divisible by (w_h,w_w): `pad_h=(w_h−H mod w_h) mod w_h`, `pad_w=(w_w−W mod w_w) mod w_w`.
- **Eq 6/7 (L339,341)** relative-position index: `δ_h=h_i−h_j, δ_w=w_i−w_j`; `I[i,j]=(δ_h+w_h−1)(2w_w−1)+(δ_w+w_w−1)`.
- **Eq 8 (L346)** weight construction: `M = R[:, I.flatten()].view(C, d, d)`.
- **Eq 9 (L372)** hierarchical-reparam fusion: `M_fused[c, B_{s,s}] = M_g[c, B_{s,s}] + M_l[c]`, s∈{0,1,2,3} — local 7×7 pattern M_l added to the four diagonal blocks of the global 14×14 matrix M_g; off-diagonal cross-sub-window blocks stay M_g; identity shortcuts collapse into the 196×196 identity ⇒ training-mode `Y_w = X_w·M_g + X_w + X_sub·M_l + X_sub` ≡ inference-mode `Y_w = X_w·M_fused` (Alg 2/3 equivalence, L1475–1480). One-time tensor-add at load ⇒ **zero inference overhead** over single-scale w=14.
- **Algorithm 1 (L282–308)** forward pass: window-partition (zero-pad) → X_batch∈ℝ^{C×N×d} (C first so bmm treats channels as batch axis) → construct M from R[:,I.flatten()] (or load M_cached) → `Y_batch = X_batch·M` → inverse-reshape.
- **WBMM-NC** (training, M rebuilt each forward — grads flow through R) vs **WBMM-C** (inference, M built once + cached); identical outputs.
- **Three lightweight add-ons:** (i) inter-block 3×3 depthwise convs for cross-window communication (Block_n=W, Block_{n+1}=DWConv3×3, kept in SEPARATE blocks for clean kernels); (ii) inference-time M caching (removes index overhead on small maps); (iii) hierarchical window reparameterization (dual-scale global+local, §3.6).
- **§3.4.3 arithmetic intensity:** for w×w window, data O(w²), compute O(w⁴) ⇒ intensity O(w²) FLOPs/loaded element (M's O(w⁴) load amortized over N windows). Larger w ⇒ higher intensity ⇒ better GPU utilization. This is WHY WBMM accelerates with larger windows.
- **§3.7 multi-kernel fusion (Pico/Nano S4 only):** `Y = WBMM(X) + BN1(DW5(X)) + BN2(DW3(X))`, fused into M at inference ⇒ +0.2pp Top-1 at zero cost.

## 3. Tables (verbatim, with sourcing line-ranges)

### Table 1 (L416–471) — WBMM-C 14×14 speedup vs DW-Std 5×5 baseline (5 batch blocks × 5 feature-map sizes)
| Batch | 14×14 | 28×28 | 56×56 | 112×112 | 224×224 |
|---|---|---|---|---|---|
| 4   | 0.45× | 0.37× | 1.41× | 1.64× | 2.02× |
| 16  | 0.62× | 1.56× | 1.80× | 2.00× | 2.07× |
| 64  | 1.86× | 1.85× | 2.01× | 2.01× | 2.03× |
| 128 | 3.17× | 2.00× | 2.06× | 1.97× | 2.01× |
| 256 | 3.72× | 2.02× | 2.02× | 1.95× | – (OOM) |
[Python-verified: every cell recomputes from Table-13 kFPS/baseline EXACT within rounding, e.g. B=256 14×14 = 1786/480 = 3.72×; B=64 14×14 = 812/437 = 1.86×. "–" at 7×7 column = window>fmap.]

### Table 2 (L542–561) — Ablation: feature extraction + window interaction (mean±std, 3 runs)
| Component | Method | Top-1 (%) | mIoU (%) |
|---|---|---|---|
| Feature Extract. | Full connection | 80.89±0.02 | 44.02±0.05 |
|  | Relative pos. bias | 82.71±0.04 | 45.50±0.07 |
|  | Rel. pos. + shortcut | 82.72±0.03 | 45.79±0.07 |
| Window Interact. | 3×3 dw parallel | 83.11±0.02 | 47.90±0.06 |
|  | 3×3 dw serial | 83.21±0.03 | 48.01±0.03 |
|  | Half ch. 3×3 dw | 83.22±0.01 | 46.21±0.04 |

### Table 3 (L542–561) — Architecture exploration (block pattern by stage; highlighted=optimal)
| S1 | S2 | S3 | S4 | Top-1 | mIoU |
|---|---|---|---|---|---|
| W,W,W | W,W,W | W↔W | W,W,W | 82.72 | 45.81 |
| W,W,W | W,W,W | W↔D | W,W,W | 82.99 | 47.18 |
| D,D,D | W,W,W | W↔D | W,W,W | 83.18 | 47.59 |
| **D,D,D** | **W,D,W** | **W↔D** | **W,W,W** | **83.21** | 47.63 |
| W,D,W | W,W,W | W↔D | W,W,W | 83.02 | 47.52 |
| **W,D,W** | **W,D,W** | **W↔D** | **W,W,W** | 83.02 | **48.32** |
| W,D,W | W,D,W | W↔D | W,D,W | 83.11 | 47.89 |
| D,W,D | W,D,W | W↔D | W,W,W | 82.98 | 48.19 |

### Table 4 (L566–575) — Hierarchical window reparameterization ablation (ADE20K, WBMM-T)
| Configuration | S1 | S2 | S3 | S4 | mIoU (%) |
|---|---|---|---|---|---|
| Pure 7×7 (baseline) | 7 | 7 | 7 | 7 | 48.0±0.03 |
| Pure 14×14 | 14 | 14 | 14 | 14 | 47.9±0.05 |
| Pure 14×14 (S4:7) | 14 | 14 | 14 | 7 | 47.8±0.04 |
| **Optimal Config** | **G+L** | **G+L** | **14** | **7** | **48.8±0.06** |
| All Hierarchical | G+L | G+L | G+L | 7 | 48.1±0.05 |

### Table 5 (L612–636) — ImageNet-1K classification (8× A800, FP32, 300ep)
| Model | Params (M) | FLOPs (G) | Mem (GB) | Time (m:s) | Acc (%) | Speedup |
|---|---|---|---|---|---|---|
| WBMM-P | 10.6 | 1.6 | 8.61 | 3:46 | 80.3 | Baseline |
| UniRepLKNet-P (LKA) | 10.7 | 1.6 | 9.45 | 4:56 | 80.2 | 1.31× |
| UniRepLKNet-P (no LKA) | 10.7 | 1.6 | 9.45 | 5:53 | 80.2 | 1.56× |
| WBMM-N | 18.1 | 2.7 | 10.01 | 4:12 | 81.7 | Baseline |
| UniRepLKNet-N (LKA) | 18.3 | 2.8 | 11.33 | 6:17 | 81.6 | 1.50× |
| UniRepLKNet-N (no LKA) | 18.3 | 2.8 | 11.33 | 7:54 | 81.6 | 1.88× |
| WBMM-T | 31.0 | 4.8 | 15.04 | 6:18 | 83.2 | Baseline |
| UniRepLKNet-T (LKA) | 31.0 | 4.9 | 16.58 | 9:03 | 83.2 | 1.44× |
| UniRepLKNet-T (no LKA) | 31.0 | 4.9 | 16.58 | 11:03 | 83.2 | 1.75× |
| WBMM-S | 55.6 | 9.0 | 20.43 | 8:35 | 83.9 | Baseline |
| UniRepLKNet-S (LKA) | 55.6 | 9.1 | 22.27 | 12:10 | 83.9 | 1.42× |
| UniRepLKNet-S (no LKA) | 55.6 | 9.1 | 22.27 | 14:16 | 83.9 | 1.66× |

### Table 6 (L639–650) — ADE20K semantic segmentation (UPerNet, T/S)
| Method | Params T/S (M) | FLOPs T/S (G) | mIoU SS T/S | mIoU MS T/S |
|---|---|---|---|---|
| UniRepLKNet | 62 / 87 | 946 / 1036 | 48.6 / 50.5 | 49.1 / 51.0 |
| WBMM (7×7) | 62 / 87 | 944 / 1033 | 48.3 / 50.2 | 48.8 / 50.5 |
| WBMM (Hier) | 66 / 92 | 948 / 1038 | 48.8 / 50.6 | 49.3 / 51.2 |

### Table 7 (L612–624) — COCO object detection (Cascade Mask R-CNN; FLOPs @1280×800)
| Method | Params T/S (M) | FLOPs T/S (G) | APbox T/S | APmask T/S |
|---|---|---|---|---|
| UniRepLKNet | 89 / 113 | 749 / 835 | 51.8 / 53.0 | 44.9 / 45.9 |
| WBMM (7×7) | 89 / 113 | 747 / 833 | 51.6 / 52.8 | 44.8 / 45.6 |
| WBMM (Hier) | 92 / 118 | 751 / 837 | 51.9 / 53.1 | 45.1 / 46.1 |

### Table 8 (L875–887) — Controlled cross-operator comparison (Tiny dense-prediction arch; only spatial operator differs)
| Operator | Params (M) | FLOPs (G) | Mem (GB) | Time | Top-1 (%) | mIoU (%) | Spd (img/s) |
|---|---|---|---|---|---|---|---|
| WBMM-T (w=7) | 31.0 | 4.8 | 15.16 | 6:20 | 83.0 | 48.3 | 1833.1 |
| WBMM-T (w=14, Hier) | 33.0 | 5.1 | 15.87 | 6:31 | 83.2 | 48.8 | 1842.2 |
| 13×13 DW + LKA | 31.0 | 5.0 | 14.77 | 6:43 | 83.1 | 48.3 | 1661.6 |
| 13×13 DW (no LKA) | 31.0 | 5.0 | 14.77 | 9:11 | 83.1 | 48.3 | 1305.3 |
| 13×13 DW + reparam + LKA | 31.6 | 5.1 | 17.23 | 10:21 | 83.2 | 48.7 | 1661.6 |
| 13×13 DW + reparam (no LKA) | 31.6 | 5.1 | 17.23 | 12:46 | 83.2 | 48.7 | 1305.3 |
| SLaK-style (in WBMM design) | 32.1 | 5.4 | 16.19 | 15:27 | 83.2 | 47.2 | 772.1 |
| SLaK-Original (all stages) | 33.7 | 5.9 | 16.91 | 22:03 | 83.3 | 47.4 | 549.0 |
| Win7 Transformer (MLP r=2.5) | 31.2 | 5.1 | 16.65 | 8:38 | 83.3 | 48.1 | 1245.1 |

### Table 10 (L941–945) — ConvNeXt-T drop-in
| Model | Params (M) | FLOPs (G) | Mem (GB) | Top-1 (%) | mIoU (%) | Time |
|---|---|---|---|---|---|---|
| ConvNeXt-T (7×7 DW) | 28.6 | 4.5 | 17.93 | 82.1 | 46.0 | 8:30 |
| ConvNeXt-T → WBMM (w=7) + 3×3 mix | 29.1 | 4.4 | 18.20 | 82.1 | 46.2 | 5:08 |

### Table 11 (L956–960) — Base-scale ImageNet-1K
| Model | Params (M) | FLOPs (G) | Mem (GB) | Time | Top-1 (%) |
|---|---|---|---|---|---|
| WBMM-B (w=7) | 97.9 | 15.9 | 26.38 | 11:24 | 83.9 |
| UniRepLKNet-B | 98.0 | 16.1 | 29.01 | 15:46 | 83.8 |

### Table 12 (L979–984) — Padding-strategy ablation (ADE20K SS mIoU)
| Padding | mIoU (%) |
|---|---|
| Zero-padding (ours) | 48.8 |
| Reflection | 48.6 |
| Replication | 48.4 |

### Table 13 (L1078–1169) — Unified operator-level benchmark, throughput (kFPS) + ∆ vs DW-Std 5×5, across batch {4,16,64,128,256} × fmap {7,14,28,56,112,224}², 256-ch FP32 A800. [Largest table; representative B=128 block:]
| Method (B=128) | 7×7 | 14×14 | 28×28 | 56×56 | 112×112 | 224×224 |
|---|---|---|---|---|---|---|
| DW-Std 3×3 | 3289 (+138%) | 992 (+147%) | 333 (+176%) | 87 (+184%) | 21.8 (+185%) | 5.46 (+187%) |
| DW-Std 5×5 (base) | 1563 | 402 | 121 | 31 | 7.6 | 1.91 |
| DW-Std 7×7 | 833 (−39%) | 279 (−31%) | 73 (−40%) | 18 (−40%) | 4.6 (−40%) | 1.14 (−40%) |
| DW-Std 13×13 | 330 (−79%) | 108 (−73%) | 27 (−78%) | 6.8 (−78%) | 1.7 (−78%) | 0.42 (−78%) |
| DW-Std 27×27 | 120 (−92%) | 29 (−93%) | 7.1 (−94%) | 1.7 (−94%) | 0.43 (−94%) | – |
| DW-LKA 7×7 | 2119 (+36%) | 530 (+32%) | 121 (0%) | 20 (−35%) | 2.8 (−63%) | 0.39 (−80%) |
| DW-LKA 13×13 | 1812 (+16%) | 450 (+12%) | 85 (−29%) | 12 (−59%) | 1.7 (−78%) | 0.22 (−89%) |
| DW-LKA 27×27 | 1812 (+16%) | 490 (+22%) | 56 (−54%) | 7.0 (−77%) | 0.87 (−89%) | – |
| WBMM-C 7×7 | 2016 (+29%) | 566 (+41%) | 165 (+37%) | 42 (+39%) | 10.5 (+37%) | 2.66 (+40%) |
| **WBMM-C 14×14** | – | **1276 (+217%)** | **242 (+100%)** | **63 (+106%)** | **15.1 (+97%)** | **3.83 (+101%)** |
| WBMM-NC 7×7 | 1582 (+1%) | 525 (+31%) | 161 (+33%) | 42 (+38%) | 10.5 (+37%) | 2.67 (+40%) |
| WBMM-NC 14×14 | – | 641 (+59%) | 199 (+65%) | 60 (+96%) | 14.9 (+96%) | 3.84 (+102%) |

### Table 14 (L1181–1191) — ADE20K inference FPS (Tiny), A800 FP32
| Backbone | 512×512 B2/B4/B8/B16 | 512×1024 B16 | 1024×1024 B16 |
|---|---|---|---|
| Conv 3×3 | 95.3/104.6/106.1/106.7 | 57.8 | 31.0 |
| Conv 5×5 | 92.4/101.8/103.3/104.0 | 56.0 | 29.8 |
| UniRepLKNet-T | 87.7/95.7/97.0/97.3 | 52.6 | 27.8 |
| UniRepLKNet-T (LKA) | 43.6/63.3/79.1/91.2 | 43.9 | 23.0 |
| WBMM-T (7×7) | 90.5/101.6/103.7/104.6 | 56.5 | 30.1 |
| WBMM-T (Hier) | 83.7/96.6/100.3/102.8 | 56.4 | 30.4 |

### Table 16 (L1242–1260) — Desktop GPU (A800 batch=128 FP32) + CPU (i7-13700K FP32) speedups. GPU W/UA = 1.01–1.28×, W/UN = 1.23–1.41×; CPU W/UN = 1.03–1.48×. [All cells Python-verified within rounding.]

### Table 17 (L1269–1290) — Edge (Jetson Orin Nano 8GB): GPU FP16 W/UA = 1.19–3.12×; CPU INT8 W/UN = 1.33–2.44×. [Verified.]

### Table 19 (L1377–1388) — Model specs (stage depths/channels/params/FLOPs): WBMM-P [2,2,6,2] @ [64,128,256,512] = 10.6M/1.6G; -N [2,2,8,2] @ [80,160,320,640] = 18.1M/2.7G; -T [3,3,18,3] = 31.0M/4.8G; -S [3,3,27,3] @ [96,192,384,768] = 55.6M/9.0G.

### Table 20 (L1393–1402) — Stage block patterns: Tiny class = [D,D,D]·[W,D,W]·[W↔D]×9·[W,W,W]; Tiny* seg/det = [W,D,W]·[W,D,W]·[W↔D]×9·[W,W,W] (S1–S2 use hierarchical 14×14+7×7, S3 single 14×14, S4 single 7×7).

---

## 4. Source-free reconciliation (Python-verified)

**Every prose delta recomputes EXACT or within rounding — ZERO numeric prose-vs-table defects.**

- **7.8× per-layer RF**: 196/25 = 7.84 ✓ (abstract/contributions/§4.1.1/§J).
- **1.31–1.88× training speedup** (Table 5): 4:56/3:46=1.31×, 5:53/3:46=1.56×, 6:17/4:12=1.50×, 7:54/4:12=1.88×, 9:03/6:18=1.44×, 11:03/6:18=1.75×, 12:10/8:35=1.42×, 14:16/8:35=1.66× — all EXACT ✓.
- **8–12% memory reduction** (Table 5): 8.9/11.7/9.3/8.3% ✓.
- **71–78% DW-Std slowdown 5×5→13×13** (fmap≥28×28, all batches): min 71% (B=16 fmap28), max 78% ✓.
- **92–94% for 27×27**: min 92%, max 94% ✓ (one cell 95% at B=4 fmap56 if you include it; prose range still defended by the bulk).
- **LKA 80%/89% slower @224 B128** (7×7/13×13): recompute −80% / −88.5% ✓ (the 89% is display-rounding of an imprecisely-displayed 1.91 baseline; non-defect).
- **Table 1 ↔ Table 13 cross-table byte-consistency**: EVERY Table-1 ratio recomputes from Table-13 kFPS/baseline within rounding (B=256 14×14 = 1786/480 = 3.72×; B=128 = 1276/402 = 3.17×; B=64 = 812/437 = 1.86×; B=4 = 63/140 = 0.45×; etc.) ✓.
- **Finding 4** "0.45× at B=4 → 3.72× at B=256 on 14×14 maps" ✓.
- **Table 2** "nearly 2pp" rel-pos-bias gap = 82.71−80.89 = 1.82pp ✓ (hedged by "nearly"); serial mIoU 48.01 > parallel 47.90 > half-ch 46.21 ✓.
- **Table 3** class peak 83.21 = [D,D,D|W,D,W|W↔D|W,W,W] ✓; seg peak 48.32 = [W,D,W|W,D,W|...] ✓.
- **Table 4** G+L 48.8 vs pure-7×7 48.0 = +0.8 ✓; vs pure-14×14 47.9 = +0.9 ✓.
- **Tables 6/7** WBMM(Hier) vs UniRepLKNet: ADE20K SS +0.2/+0.1pp (T/S), MS +0.2/+0.2; COCO APbox +0.1/+0.1, APmask +0.2/+0.2 ✓.
- **Table 8** reparam mem 17.23−15.87 = +1.36 GB ✓; SLaK 2.44–3.48× train / 2.37–3.34× infer ("2.4–3.4×" ✓); Win-attn 1.32–1.36× train / 1.47–1.48× infer / 0.78–1.49 GB mem ("0.8–1.5" ✓) — ALL EXACT.
- **Table 10** 8:30/5:08 = 1.66× ✓; **Table 11** 15:46/11:24 = 1.38×, mem −9.1% ✓.
- **§J.1** Tiny 512×512 B16 104.6 vs 106.7 = 1.97% slower ("2.0%" ✓), 49/9 = 5.44× RF ("5.4×" ✓); **§J.2** Small 93.9/94.5 = 0.63% ("0.6%" ✓), Hier 26.3 vs 25.6 = +2.73% ("2.7%" ✓).
- **§I** +272% @B256 fmap14 = (1786−480)/480 ✓; WBMM-C/NC smallest = 63/21 = 3.0× ("~3×" ✓); 169 = (2·7−1)² ✓; M mem w=14 C=256 = 256·196·196·4 = 37.5 MiB ✓.
- **§K** GPU 1.01–1.28× (UA) / 1.23–1.41× (UN); CPU 1.03–1.48×; edge GPU 1.19–3.12×; edge CPU INT8 1.33–2.44× — ALL EXACT.

**Cross-table note (EXPLAINED, not a defect):** Table 5 WBMM-T Top-1 83.2 / Mem 15.04 vs Table 8 WBMM-T (w=7) Top-1 83.0 / Mem 15.16 — the paper itself explains (§C note + Table 8 note): Table 5 uses the classification-tuned architecture [D,D,D]S1, Table 8 uses the dense-prediction architecture [W,D,W]S1 (Tiny*). Same model name, two legitimate configs.

---

## 5. Honest-scope flags (⚠ — attribution/scope, NOT numeric typos; the paper has ZERO numeric defects)

1. **Selective-baseline speedup headline (iter-72 MARVEL class).** The abstract's "**1.31–1.88× training speedup**" spans TWO baselines: the lower half (1.31/1.50/1.44/1.42×) is vs UniRepLKNet **WITH its shipped LKA acceleration** (the production baseline); the upper half (1.56/1.88/1.75/1.66×) is vs UniRepLKNet **WITHOUT LKA** (the slower, non-deployed baseline). The 1.88× ceiling is vs the slower baseline. **Cite "1.31–1.50× vs LKA-accelerated (production), up to 1.88× vs non-accelerated."** Both columns are shown in Table 5 (transparent), but the headline range picks the max of each.

2. **Abstract blanket "outperforms 5×5 depthwise convolution baselines in speed" understates small-batch/small-map regression.** Table 1 B=4: fmap 14×14 = **0.45×**, fmap 28×28 = **0.37×** (WBMM-C 14×14 is <half the DW-Std 5×5 speed there). The body hedges correctly (§4.1.1 Finding 3: "faster on fmap ≥28×28 at batch ≥16, and ≥56×56 at all tested batch sizes") but the abstract is blanket. **Cite the qualified version**, not the blanket.

3. **"Comparable or higher accuracy" is accuracy-NEUTRAL, not a win.** Table 5: WBMM vs UniRepLKNet within ±0.1pp on ALL 4 scales (P 80.3/80.2, N 81.7/81.6, T 83.2/83.2, S 83.9/83.9). "Higher" is at most +0.1pp with NO seeds/CIs on Table 5. The contribution is SPEED; "comparable or higher" reads as an accuracy win when it is a tie.

4. **Segmentation/detection gains are sub-0.3pp without CIs.** Tables 6/7: ADE20K +0.1–0.2pp mIoU, COCO +0.1–0.2pp AP vs UniRepLKNet. Ablation Tables 2/4 carry ±std but the main Tables 5/6/7 do NOT. The contributions headline ("surpasses UniRepLKNet in both mIoU and AP") rests on deltas within plausible run noise. The one segmentation result WITH std (Table 4 hierarchical: 48.8±0.06 vs 48.0±0.03, +0.8pp ≈ 12× the larger std) is significant — but that is an ABLATION, not the main SOTA comparison.

5. **"7.8× larger per-layer receptive field" = window-size ratio (196/25), NOT effective RF.** The paper is consistent ("per-layer"), but a casual read conflates window size with the network's effective receptive field. The SLaK comparison (§C/Table 8) makes this explicit: SLaK has effective RF 51 yet UNDERPERFORMS WBMM on dense prediction (47.2/47.4 vs 48.3/48.8 mIoU) — effective RF ≠ window size, and bigger windows aren't uniformly better (Table 4: pure 14×14 47.9 < pure 7×7 48.0).

6. **Operator-level peak ≠ end-to-end-model speedup.** The 3.72× / +272% figures (Table 1, Table 13) are SINGLE-LAYER, single-A800, FP32, 256-channel, isolated-operator benchmarks (§I justifies this as architecture-agnostic). The end-to-end ImageNet training speedup (Table 5: 1.31–1.50× vs LKA) is ~2–3× SMALLER than the operator peaks. Body is clear; abstract's "outperforms in speed" could be misread as the 3.72× figure for a real model.

7. **"Hardware-agnostic" is NVIDIA-GPU + Intel-CPU.** §K tested A800 GPU, Jetson Orin Nano GPU (FP16) + CPU (INT8), i7-13700K CPU. No AMD GPU, no non-Jetson ARM mobile. The CPU speedups (1.03–1.48×) are the weakest and rely on vendor BLAS. "Hardware-agnostic" overstates the vendor breadth; "NVIDIA/Intel-family with quantization compatibility" is the defensible scope.

8. **Window-attention comparison is segmentation-carried; on classification attention wins.** Table 8/D: Win7 Transformer +0.3pp Top-1 over WBMM-T w=7 (83.3 vs 83.0). The paper's framing "WBMM matches or beats parameter-matched window attention on segmentation while being faster" is correct but the efficiency win is traded for a small classification loss — the framing leads with segmentation.

9. **§3.7 "+0.2pp Top-1 from multi-kernel fusion" is figure/text-only** — no table reports it; unverified by this reconciliation.

10. **WBMM's own M-memory ceiling (§O, authors' concession).** M∈ℝ^{C×d×d} = O(C·w⁴): ~37.5 MiB/256-ch-layer at w=14, "memory-prohibitive" beyond w=28. The "scales with larger windows" advantage has a hard ceiling; a Flash-Attention-style tiled kernel (planned, not delivered) is needed to go beyond. Cite the speedup only within the tested w∈{7,14}.

11. **Operator benchmark uses IQR-based outlier removal over 5 runs** (§4.1) — reasonable but a non-standard aggregation (vs mean±std); no per-cell variance reported in Table 13, so the ∆-column precision (e.g. +217% vs +218%) exceeds the displayed-precision of the underlying kFPS (3 sig figs).

12. **27×27 slowdown range is 92–95% across cells** (Table 13: B=4 fmap56 = −93%, but a couple of cells round to −95% depending on the imprecisely-displayed baseline) — prose "92–94%" is slightly optimistic at the upper end but defensible for the bulk. Minor.

---

## 6. Citable falsifiable content

- **The paradigm flip** (Eq 1–8, §3.4): traverse the compact relative-position-bias table R∈ℝ^{C×(2w−1)²} instead of gathering scattered input neighborhoods; construct per-channel M∈ℝ^{C×d×d} once (batch-independent, O(C·d²), cacheable); apply via batched matmul on contiguous w×w windows. Arithmetic intensity O(w²) FLOPs/element ⇒ throughput RISES with window size (opposite to depthwise conv) and with feature-map size (opposite to LKA). [Theorem 3.1/3.2 give the global equivalence; the windowing is the Swin-style practical approximation.]
- **Hierarchical window reparameterization** (Eq 9, §3.6, Alg 2/3): dual-scale R_g (14×14) + R_l (7×7), trained independently with identity shortcuts, fused at inference into a single 196×196 matrix (local M_l on the 4 diagonal blocks of global M_g) with ZERO runtime overhead — recovers multi-scale context that pure large windows lose (Table 4: G+L 48.8 > pure-14×14 47.9).
- **Three controlled ablations** (Tables 2/3/4) with ±std: relative-position bias is load-bearing (+1.82pp Top-1 over full-connection); serial 3×3 inter-block mixing beats parallel/half-channel on mIoU; hierarchical (G+L at S1–S2) is the optimal multi-scale config.
- **The honest efficiency win, scoped**: vs the LKA-accelerated UniRepLKNet that ships in practice, WBMM gives 1.31–1.50× end-to-end ImageNet training speedup + 8–12% memory at accuracy parity, and the operator-level advantage GROWS with feature-map size (the regime where LKA breaks: −63 to −99% at fmap≥112) — making WBMM the preferred operator for high-resolution dense prediction (semantic seg, detection, super-res).

**NOT citable as headline**: "1.88× speedup" (vs the non-deployed no-LKA baseline — flag 1); "outperforms 5×5 in speed" without the ≥28×28-@-batch≥16 qualifier (flag 2); "higher accuracy" (flag 3, it is a tie); sub-0.3pp segmentation/detection SOTA gains without CIs (flag 4); "7.8× larger receptive field" as if effective RF (flag 5); "hardware-agnostic" beyond NVIDIA/Intel (flag 7).

**Sibling lineage:** repo's first efficient-conv-OPERATOR / hardware-aware-kernel paper. Distinct from `program-as-weights` (differentiable program, not a kernel), `viq` (visual tokens), and the model-efficiency papers (`speculating-experts` MoE-inference, `sasp` masked-drafters, `spec-auf` speculative-decoding, `token-compression-vs-pruning-ViT` structural-pruning — all model/algorithm-level efficiency, NOT operator-level memory-access redesign). Closest in SPIRIT is the science-ML-efficiency lineage (`soap-muon-mlip`, `dsgnar`) in that the contribution is a drop-in replacement that is faster at parity, but WBMM is a vision-operator/kernel contribution, not an optimizer.
