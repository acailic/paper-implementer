# OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers

**Source.** arXiv 2607.02461 (2 Jul 2026). Donghyun Lee, Jitesh Chavan, Duy Nguyen, Sam Huang, Liming Jiang, Priyadarshini Panda, Timo Mertens, Saurabh Shukla (Cantina Labs / USC / UIUC). 15 pp (`pdfinfo`=15; `file` misreports **11 pp** — file-vs-pdfinfo defect recurs, iters 66/67/69/70/71/72; trust pdfinfo). `paper_layout.txt` = pdftotext -layout, 960 lines, **7 explicit tables (T1–T7) + Algorithm 1 + Eqs 1–13 + Proposition 1 (Lemmas 1–2) + 6 figures**. Project page: saurabhcantina.github.io/orbitquant.

**Subarea (repo's FIRST).** **Post-training quantization (PTQ) of weight + activation for image & video diffusion transformers (DiTs)** — calibration-free, rotation-based. OrbitQuant replaces per-timestep/per-prompt range calibration with a single offline Lloyd–Max codebook built from the *post-rotation coordinate marginal* `f_d`, applied in a shared rotated+normalized basis so the rotation cancels inside every linear layer. No prior repo paper covers PTQ, low-bit quantization, Hadamard/rotation-based quantization, Lloyd–Max codebooks, or quantization of generative diffusion models (the 3 "quant" hits are unrelated — `viq` learns discrete visual tokens, `program-as-weights` and `bayesian-sparse-lora` are optimizer/adapter design). Sibling-in-spirit to the rotation/math-lineage (`dsgnar` iter 67 second-order PINNs, `marvel` iter 72 vMF hyperspheres) and to the inference-efficiency angle of `speculating-experts` / `exformer` — but OrbitQuant attacks cost at the **tensor-numerics** level (rotate → normalize → fixed-codebook), not the architecture/attention level.

---

## 1. Problem & motivation

Diffusion transformers (FLUX.1, Z-Image-Turbo, Wan 2.1, CogVideoX) are SOTA image/video generators but **expensive**: (a) the transformer trunk runs over many sequential denoising timesteps; (b) unlike LLM decoding (weight-loading-bound), DiT inference is **compute-bound** even at batch 1, so weight-only quant gives no speedup (L70–79). Low-bit weight+activation PTQ is the remedy — **but** DiT activations exhibit channel-wise outliers **and drift across timesteps, prompts, and classifier-free-guidance branches** (L87), so the activation range is a moving target. Prior DiT-PTQ methods (SVDQuant, PTQ4DiT, AdaTSQ, ViDiT-Q) absorb this drift with **calibration**, so each new checkpoint / resolution / modality requires re-collected calibration data. OrbitQuant's thesis: **don't estimate the range — rotate it away**, so one fixed codebook serves all inputs.

## 2. Method

### 2.1 Inherited ingredients (TurboQuant, §3.2, L154–192)

OrbitQuant inherits two pieces from TurboQuant (a KV-cache vector quantizer): (1) a **Haar-random orthogonal rotation** `Φ_d ∈ R^{d×d}`; (2) the fact that each coordinate of `Φ_d x̃` (for unit `x̃`) follows the fixed marginal

```
f_d(t) = sqrt( Γ(d/2) / (π Γ((d−1)/2)) ) · (1 − t²)^{(d−3)/2},  t ∈ [−1,1]      (Eq 2)
```

which for `d≥64` is tightly approximated by `N(0, 1/d)` with near-independent coordinates. Since `f_d` is known offline, an MSE-optimal **Lloyd–Max codebook** `C^{(d,b)} = {c_1,…,c_{2^b}}` is precomputed per `(d, b)` (Eq 3 nearest-centroid map `q̂_b^{(d)}`), with **no scales or zero-points**, shared by all layers/rows of dimension `d`.

### 2.2 Offline weight quantization (§4.2, Eqs 4–6)

Rotate weight into the shared basis `W′ = W Π_d^⊤` (Eq 4); split each row into magnitude `r_i′ = ‖w_i′‖` and unit direction `w̃_i′` (Eq 5); quantize direction with the Lloyd–Max codebook and re-attach magnitude:

```
Ŵ′ = diag(r′) · Q̂_{b_w}^{(d)}(W̃′)                                            (Eq 6)
```

`r′ ∈ R^m` stored in BF16 adds 16m bits/layer (<0.3% of the `b_w·m·d` quantized bits). Because `Π_d` is sampled independently of `w_i`, each unit direction `w̃_i′` has coordinates following `f_d`, so the codebook is MSE-optimal on it.

### 2.3 Online activation quantization (§4.3, Eqs 7–8)

Each incoming activation is rotated, then split into a per-token scalar magnitude `s` and unit direction, quantized with the **same codebook family** and rescaled:

```
x′ = Π_d x,   s = ‖x′‖,   x̃′ = x′/(s+ε)                                      (Eq 7)
x̂′ = s · Q̂_{b_a}^{(d)}(x̃′)                                                   (Eq 8)
```

The **only** input-dependent quantity at inference is the per-token scalar `s`; the codebook is fixed. The weight absorbs `Π_d^⊤` and the activation applies `Π_d`, so `W′x′ = W Π_d^⊤ Π_d x = Wx` — the rotation **cancels** in the product, leaving `Ŵ′x̂′ ≈ Wx` with **no inverse rotation at runtime**.

### 2.4 Randomized permuted block-Hadamard (RPBH), §4.4, Eq 9, Proposition 1

A dense Haar rotation costs `O(d²)` time + storage. OrbitQuant realizes `Π_d` as

```
Π_d = blkdiag(H_h D_1, …, H_h D_{d/h}) · P_π                                   (Eq 9)
```

— block-diagonal `h×h` Walsh–Hadamard `H_h` with Rademacher sign diagonals `D_i`, preceded by a **uniformly-random permutation** `P_π`. Admits an `O(d log h)` transform (permutation gather + per-block Fast Walsh–Hadamard), stores only a sign vector + permutation array, and is **constructible on any `d`** (unlike Full RHT, which needs power-of-two; `h` = largest power of two dividing `d`, giving `h∈{128,512,1024,2048,4096}` across models, including `d=1920` of CogVideoX-2B where no fast size-`d` Hadamard kernel exists).

**Proposition 1 (Universal variance concentration, Eq 10).** For `d=kh`, unit `x̃` with `µ_∞=‖x̃‖_∞`, every `δ∈(0,1)`, with prob ≥ `1−δ` over `Π_d`, every coordinate `z_i` of `Π_d x̃` is mean-zero with

```
Var(z_i | π) ∈ [ (1−ρ)/d , (1+ρ)/d ],   ρ = (d µ_∞)/(2h) · sqrt( (4k/d) log δ )   (Eq 10)
```

Proof (supp A): `z_i` is a mean-zero Rademacher sum (E[z_i]=0); Lemma 1 (sub-Gaussian tail, union bound → Eq 11) + Lemma 2 (Hoeffding without-replacement mass-balancing bound on per-block mass `M_j`, union bound → Eq 12) give `M_j ∈ (1/k)(1±ρ)` ⇒ variance `M_j/h ∈ (1/d)(1±ρ)`. **The permutation enters only through Lemma 2** — it equalizes per-coordinate variance to `1/d` regardless of how outlier channels fall into blocks, which the permutation-free Block-RHT loses at low bit-width (Remark 1, §6.1).

### 2.5 Algorithm 1 (verbatim, L208–235)

Offline (once): for each `d∈D`, draw `Π_d←RPBH(d)`, fit `Q̂_{b_w}^{(d)}, Q̂_{b_a}^{(d)}` via Lloyd-Max; for each weight `W` of dim `d`: `W′←WΠ_d^⊤`, split norm/direction, `Ŵ′←diag(r′)·Q̂_{b_w}(W̃′)`, replace `W` by `Ŵ′`. Online (per token batch `x`): `x′←xΠ_d^⊤`, `s←‖x′‖`, `x̃′←x′/(s+ε)`, `x̂′←s·Q̂_{b_a}(x̃′)`.

## 3. Experimental setup (§5, L306–335)

- **Models.** Image: FLUX.1-schnell (4-step, g=0.0), FLUX.1-dev (50-step, g=3.5), Z-Image-Turbo (10-step, g=0.0) at **W4A4 / W2A4** (+ W3A3/W2A3 supp). Video: Wan 2.1-1.3B (81fr, 480×832, 50-step, CFG 5.0), CogVideoX-2B (49fr, 480×720, 50-step, CFG 6.0) at **W4A6 / W4A4** (+ Wan 14B / HunyuanVideo supp). AdaLN modulation projections kept at INT4 weight RTN (group 64), BF16 activations — identical across all methods.
- **Quantized layers.** Every transformer-block linear projection (Q/K/V/output, FFN, joint-attention text path / cross-attention). AdaLN = one exception (dynamic timestep scale-and-shift can't be folded; single conditioning token so no activation compute to save). Embeddings, timestep MLP, final un-patchify head, text encoder stay BF16 (§B.2, L800–820).
- **Baselines.** Image: SVDQuant, AdaTSQ, ViDiT-Q (calibration-based) + Q-DiT, QuaRot, SmoothQuant. Video: ViDiT-Q, SVDQuant, QuaRot, SmoothQuant (+ DVD-Quant, QAT methods LSQ/Q-DM/EfficientDM/QVGen supp). Baseline numbers taken from AdaTSQ (image) and QVGen (video).
- **Hardware.** NVIDIA H100. Fake-quantization eval for latency/memory (weights+acts dequantized to BF16, matmul in BF16) — measures **quantization overhead, not realized low-bit speedup**.
- **Metrics.** GenEval (6 compositional sub-tasks + Overall, image); VBench (8 dims + Overall Consistency, video).

## 4. Tables (verbatim, all sourcing line-ranges)

### Table 1 — GenEval image results (L335–376), 6 sub-tasks + Overall, per (model, bit-width)

FLUX.1-schnell (FP16 Overall 0.664): W4A4 — Q-DiT 0.373, SmoothQuant 0.281, QuaRot 0.458, ViDiT-Q 0.495, SVDQuant 0.624, AdaTSQ 0.680, **OrbitQuant 0.703** (exceeds FP16 +0.039). W2A4 — QuaRot† 0.001, SmoothQuant† 0.000, ViDiT-Q† 0.001, **OrbitQuant 0.604**.

FLUX.1-dev (FP16 0.667): W4A4 — Q-DiT 0.014, SmoothQuant 0.007, QuaRot 0.243, ViDiT-Q 0.280, SVDQuant 0.573, AdaTSQ 0.618, **OrbitQuant 0.633** (trails FP16 by 0.034). W2A4 — QuaRot† 0.001, SmoothQuant† 0.000, ViDiT-Q† 0.001, **OrbitQuant 0.475**.

Z-Image-Turbo (FP16 0.754): W4A4 — SmoothQuant 0.000, QuaRot 0.519, ViDiT-Q 0.668, SVDQuant 0.718, AdaTSQ 0.762, **OrbitQuant 0.767** (exceeds FP16 +0.013). W2A4 — QuaRot† 0.001, SmoothQuant† 0.001, ViDiT-Q† 0.001, **OrbitQuant 0.319**.

### Table 2 — VBench video PTQ (L411–472), percentages, per (model, bit-width) × 8 dims + Overall Consistency

Wan 2.1-1.3B W4A6 Overall Consistency: SmoothQuant† 22.15, QuaRot† 22.65, ViDiT-Q 19.58, SVDQuant 23.26, **OrbitQuant 24.35** (best). W4A4: SmoothQuant† 15.05, QuaRot† 17.98, ViDiT-Q† 13.11, SVDQuant 21.91, **OrbitQuant 23.86** (best). CogVideoX-2B W4A6: … SVDQuant 21.34, **OrbitQuant 24.55** (best). W4A4: SVDQuant 22.89, **OrbitQuant 23.86** (best).

### Table 3 — Rotation-class ablation on FLUX.1-schnell (L522–535), GenEval Overall (3-seed mean) × 3 bit-widths + per-image activation-rotation latency (s) at 1024² on H100

| Rotation | W4A4 | W3A3 | W2A4 | Latency(s) |
|---|---|---|---|---|
| Haar | 0.696 | 0.669 | 0.591 | 11.65 |
| Full RHT | 0.691 | 0.672 | 0.587 | 0.452 |
| Block-RHT | 0.678 | 0.642 | 0.558 | 0.381 |
| **RPBH (ours)** | 0.690 | **0.674** | **0.595** | 0.451 |

(RPBH best at the low bit-widths W3A3/W2A4; permutation drives the gap over Block-RHT; structured rotations 25.8× faster than dense Haar.)

### Table 4 — GenEval lowest bit-widths W3A3/W2A3 (L796–826)

FLUX.1-schnell W3A3: SVDQuant 0.504, AdaTSQ 0.634, **OrbitQuant 0.678**; W2A3: QuaRot†/SmoothQuant†/ViDiT-Q† ≈0.001, **OrbitQuant 0.517**. FLUX.1-dev W3A3: SVDQuant 0.377, AdaTSQ 0.527, **OrbitQuant 0.584**; W2A3 **OrbitQuant 0.372**. Z-Image-Turbo W3A3: SVDQuant 0.000, AdaTSQ 0.694, **OrbitQuant 0.740**; W2A3 **OrbitQuant 0.105** (degrades sharply — limit of calibration-free codebook at this bit-width).

### Table 5 — VBench on Wan 14B + HunyuanVideo at W4A4 (L826–847), 8 dims + Overall

Wan 14B W4A4 (OrbitQuant best PTQ on 7 of 8 dims, loses only Motion Smoothness to QuaRot 0.9763 vs 0.9754): Imaging 0.6405, Aesthetic 0.6022, Motion 0.9754, Dynamic 0.6250, BG-Cons 0.9559, Subj-Cons 0.9363, Scene 0.3285, Overall 0.2615 — all best-PTQ except Motion. HunyuanVideo W4A4: competitive with the video-specific DVD-Quant (ahead on imaging quality, motion smoothness, scene).

### Table 6 — Seed robustness (L863–877), GenEval mean±SD over 3 seeds

FLUX.1-schnell W4A4 Overall **0.690 ± 0.012**, W2A4 0.595 ± 0.008. FLUX.1-dev W4A4 0.639 ± 0.004, W2A4 0.460 ± 0.014. Z-Image W4A4 0.767 ± 0.001, W2A4 0.276 ± 0.072.

### Table 7 — Video W4A4 vs QAT + PTQ (L878–907), Wan 2.1-1.3B / CogVideoX-2B, P/Q flag

Wan 2.1-1.3B W4A4: QAT (QVGen 23.01 Overall Consistency) vs PTQ — OrbitQuant **23.86** (beats every QAT method on Overall Consistency, Subject Consistency 92.98, Scene 18.81). CogVideoX-2B: OrbitQuant 23.86 Overall Consistency.

## 5. Source-free reconciliation (Python, all cells from layout)

**ALL headline deltas recompute EXACT:**

- **§5.2 "exceeds FP16 on Overall on FLUX.1-schnell and Z-Image-Turbo"**: 0.703−0.664=**+0.039** ✓, 0.767−0.754=**+0.013** ✓ (both genuine exceeds).
- **§5.2 "trailing it by 0.034 on FLUX.1-dev"**: 0.667−0.633=**0.034** ✓ EXACT.
- **W2A4 "only method that produces meaningful scores"**: all baselines ≈0.000–0.001 on all 3 backbones; OrbitQuant 0.604 / 0.475 / 0.319 ✓.
- **Table 3 "RPBH adds 0.070 s over Block-RHT"**: 0.451−0.381=**0.070** ✓ EXACT; "no slower than the Full RHT" 0.451 vs 0.452 ✓; "order of magnitude faster (26×)" 11.65/0.451=**25.8×** ✓.
- **§6.1 "RPBH is the strongest at W3A3 and W2A4, ahead of dense Haar / Block-RHT / Full RHT"**: T3 W3A3 RPBH 0.674 > Haar 0.669 > Full RHT 0.672 > Block-RHT 0.642 ✓; W2A4 0.595 > Haar 0.591 > Full RHT 0.587 > Block-RHT 0.558 ✓.
- **§C.2 "best PTQ on seven of the eight dimensions" (Wan 14B T5)**: OrbitQuant best on Imaging/Aesthetic/Dynamic/BG/Subj/Scene/Overall = **7/8** ✓ (loses Motion Smoothness to QuaRot 0.9763 vs 0.9754).
- **§C.4 "surpasses every QAT method on several, leading on Subject Consistency, Scene, and Overall Consistency on Wan 2.1-1.3B" (T7)**: Subject 92.98 > QVGen 92.57 ✓, Scene 18.81 > QVGen 15.32 ✓, Overall Consistency 23.86 > QVGen 23.01 ✓.
- **§5.5 video peak memory "20.3 vs 19.3 GB (QuaRot/SmoothQuant), still below ViDiT-Q 23.2"**: figure-only (Figure 5), restated verbatim, not table-verifiable.

**TWO genuine prose-vs-table inconsistencies (T6 seed-robustness std claims, iter-30/31/34/60/69 DemoPSD/MI-EPO class):**

1. **§C.3 "At W4A4 the Overall standard deviation is at most 0.005 on every model"** — **FALSE for FLUX.1-schnell**: T6 reports FLUX.1-schnell W4A4 Overall = **0.690 ± 0.012**, which is **2.4× the 0.005 bound**. FLUX.1-dev (0.004) and Z-Image (0.001) satisfy it. The "single seed is representative" conclusion is overstated for FLUX.1-schnell W4A4. ⚠
2. **§C.3 "The FLUX models stay similarly stable at W2A4, within 0.013 on Overall"** — **FALSE for FLUX.1-dev**: T6 reports FLUX.1-dev W2A4 Overall = **0.460 ± 0.014**, exceeding the 0.013 bound (FLUX.1-schnell W2A4 0.008 satisfies it). ⚠

(Both are std-bound overstatements on the stability claim; the central OrbitQuant-is-still-representative conclusion holds for FLUX.1-dev W4A4 and Z-Image, but not uniformly as stated.)

## 6. Strengths

1. **Calibration-free by construction** is the genuine contribution: range estimation is removed *at the source* (Proposition 1 forces every rotated coordinate to the same `f_d` marginal regardless of input), not patched over with per-input scales. The same codebook + recipe transfers image↔video with zero per-modality tuning — verified on 5 backbones spanning 2 modalities and 3 bit-width families.
2. **Proposition 1 (universal variance concentration)** is a clean falsifiable anchor: the random permutation's role is *isolated* (Lemma 2 only; Remark 1 notes Lemma 1 holds with or without it), and §6.1 (T3) empirically reproduces the predicted Block-RHT-without-permutation degradation at low bit-width. The proof is self-contained (sub-Gaussian + Hoeffding-without-replacement + union bounds).
3. **Rotation-cancels-in-the-product** design (weight absorbs `Π_d^⊤`, activation applies `Π_d`) is the engineering hinge that makes the online cost a single forward rotation rather than a quantize-dequantize codec (vs TurboQuant/Polar-Quant which rotate back to reconstruct).
4. **Strong low-bit robustness**: at W2A4 every baseline collapses to ≈0.000–0.001 GenEval on all 3 backbones while OrbitQuant retains 0.604/0.475/0.319 — the only functional method; pushes PTQ to W2A4 with usable quality (and W2A3 on FLUX).
5. **Beats QAT on part of the benchmark without any gradient step** (T7 Wan 2.1: Overall Consistency 23.86 > QVGen 23.01) — strong evidence the rotated-codebook design compensates for the lack of fine-tuning.

## 7. Limitations & honest-scope flags (⚠)

1. **T6 stability-claim overstatement (most important).** §C.3 "Overall std at most 0.005 at W4A4 on every model" is violated by FLUX.1-schnell (0.012); "within 0.013 at W2A4" violated by FLUX.1-dev (0.014). The seed-robustness headline is not uniform across the 3 models as the prose claims. ⚠
2. **Fake-quantization latency/memory (§5.5, authors' own).** All latency/memory numbers are measured under fake quantization (codes dequantized to BF16, matmul in BF16) — they measure **quantization overhead, not realized low-bit speedup**. §D concedes no off-the-shelf kernel computes a non-uniform (Lloyd–Max) codebook GEMM; integer tensor cores need uniform grids, so the current path dequantizes and runs BF16 matmul "as do all baselines." Realized-speedup is future kernel work. ⚠
3. **Online rotation cost is unavoidable** (§D, authors' own). Unlike weight-only/BF16 inference, RPBH is applied to activations at every forward pass; 0.451 s/image on H100 at 1024². Authors call this an implementation limitation, but it is a real runtime cost that weight-only methods avoid.
4. **Proposition 1 is asymptotic/concentration, not exact.** `f_d ≈ N(0,1/d)` is tight only for `d≥64`; the variance bound (Eq 10) degrades when one coordinate carries an outsized norm (`µ_∞` large) — the random permutation mitigates but does not eliminate this. Finite-`d` low-bit benefit is argued empirically (T3, Fig 3 KS-distance), not bounded for the multi-class/large-outlier regime.
5. **Z-Image-Turbo is the weak regime at extreme low-bit.** W2A3 OrbitQuant Overall collapses to **0.105** (vs FLUX-schnell 0.517, FLUX-dev 0.372); W2A4 Z-Image seed-std is 0.072 (vs ≤0.014 elsewhere, T6) — the calibration-free codebook hits its limit here, as authors concede (§C.1).
6. **AdaLN forced to INT4 RTN, not OrbitQuant.** AdaLN modulation (27% of weights) can't use the rotation-cancellation design (dynamic timestep scale-and-shift, single conditioning token), so it's quantized by a different method (INT4 RTN group-64) and kept out of the "4× compression" story only by also being pushed to INT4 (Fig 6 right: BF16 AdaLN → 2.21×, INT4 → 4×). The headline 4× compression is partly AdaLN-RTN-carried.
7. **Baseline numbers sourced from other papers.** Image baselines from AdaTSQ, video from QVGen (§5.1) — not all re-run by the authors (only †-marked SmoothQuant/QuaRot/ViDiT-Q at W2A4/W2A3 are their own implementation). Cross-paper evaluation-protocol mismatch risk.
8. **No FP16/W4A4 latency realized speedup, only overhead.** The "lowest overhead among W+A methods" (Fig 5) is under fake-quant; the actual deployment speedup at low-bit is unmeasured and blocked on the codebook-GEMM kernel (§D).
9. **Single-seed main tables.** Only T6 (3 seeds) reports variance; T1/T2/T4/T5/T7 main results are single-seed, with the stability claim (caveat 1) itself overstated. The "single seed is representative" justification is the very claim that T6 partially contradicts.

## 8. Verdict

OrbitQuant is a **theoretically-grounded (Proposition 1 universal variance concentration), engineering-clean (rotation-cancels-in-product, `O(d log h)` RPBH on any dimension), empirically strong (only functional method at W2A4, beats QAT on part of VBench)** calibration-free weight+activation PTQ for image & video diffusion transformers — the repo's FIRST quantization / DiT-PTQ paper. Every quantitative prose delta source-recomputes EXACT (FLUX-dev −0.034 trail, RPBH-BlockRHT +0.070 latency, 7/8-dim Wan-14B best, QAT 3-dim lead). The caveats a reader must carry: (a) §C.3's seed-robustness std bounds (≤0.005 W4A4, ≤0.013 W2A4) are **violated by FLUX.1-schnell W4A4 (0.012) and FLUX.1-dev W2A4 (0.014)** — the stability headline is overstated; (b) all latency/memory numbers are **fake-quantization overhead, not realized low-bit speedup** (no codebook-GEMM kernel exists yet). Sibling to the repo's rotation/math lineage; repo's FIRST PTQ / quantization / diffusion-transformer-inference paper.
