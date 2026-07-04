# Representation Distribution Matching for One-Step Visual Generation (iRDM) — Source-First Breakdown

**arXiv:** 2607.02375 (cs.CV, 2 Jul 2026) · **PDF:** 23pp (pdfinfo; `file` misreports 12pp → **11-page gap, defect recurs**) · **Authors:** Lan Feng, Wuyang Li (EPFL) · Éloi Zablocki (Valeo.ai) · Matthieu Cord (Valeo.ai / Sorbonne) · Alexandre Alahi (EPFL) · **Project:** https://alan-lanfeng.github.io/rdm/ . Repo entry rank 69 / 74th paper; **first design-space/axis-decomposition of representation-distribution-matching one-step generation / Nyström-MMD-for-generation / multi-encoder-constrained-optimization-generator / Sliced-Wasserstein-14-encoder-eval paper** in the library.

> Sibling-in-spirit to the **distillation lineage** (orbitquant iter-73, danceopd/demopsd/opid/purified-opsd — all *policy/reasoning* distillation; wan-streamer video) but iRDM is **teacher-free feature-distribution matching for image generation**, not trajectory/score distillation; distinct from pointdit/InvSplat (feed-forward geometry). Citable falsifiable hinge = the two-axis decomposition (comparison × representation) + the counterintuitive ablations (Nyström-MMD *beats* exact MMD; single-encoder gaming driven below the real floor while images stay fake) + the SWr14 metric (14-encoder Sliced-Wasserstein, training-loss-independent).

---

## 1. The claim (in one paragraph)

One-step image generators can be trained by **matching generated vs reference feature distributions under frozen pretrained encoders** — no online teacher, adversary, or trajectory to simulate. The authors name this paradigm **Representation Distribution Matching (RDM)** and decompose it into **two design axes**: (i) the **comparison** (which discrepancy `D`, which finite-sample estimator, what reference, which joint law) and (ii) the **representations** (which encoders, how weighted). Three controlled findings overturn current practice:

- **F1** Classical **MMD** (dismissed a decade ago as too weak) becomes strong+scalable **once estimated right** — exact within-batch repulsion + a **Nyström attraction** to a reference frozen once over the full 1.28M-image set (4096 landmarks).
- **F2** The **generation batch `N` is the operative variable**, with a broad optimum **above 2048** (adopted N=5120 ImageNet, N=10240 FLUX) — an order of magnitude past common practice; gradient caching absorbs the memory.
- **F3** **Any single encoder can be gamed** (driven below the real-validation floor while images stay visibly fake), so match against a **diverse 10-encoder battery** balanced by **constrained (proportional-Lagrangian) optimization**, and evaluate with **SWr14** — a Sliced-Wasserstein ratio over 14 encoders (4 held out), independent of the training loss.

Combining the preferred choices gives **iRDM**: one-step ImageNet-256 SOTA at **SWr14 1.30** (prior best pMF-H FD-SIM 2.05; corroborated by PickScore 71.2% win vs that prior best), and post-trains 4-step FLUX.2 [klein] into a one-step model **surpassing the 4-step teacher on GenEval 0.826 vs 0.794 and PickScore 22.76 vs 22.58 in 90 H200 GPU-hours**.

---

## 2. Method (§3, Eqs 1–5) — the two-axis contract

- **One-step generator** `gθ: z∼pz → image in one evaluation`; output law `pθ`. **Eq 1** `L(θ)=D(ϕ∗pθ, ϕ∗pdata)` — pushforward under frozen encoder `ϕ`, distance `D`. Constraining the output law (not a trajectory) makes it one-step by construction.
- **Comparison axis (§3.1–3.2).** **Eq 2** squared MMD with Gaussian kernel `k(x,y)=exp(−‖x−y‖²/2σϕ²)`, bandwidth per encoder by median heuristic. **Eq 3** the operative loss `L_b^ϕ = (1/B)Σ k(g_i,g_j) − (2/B)Σ ψ(g_i)⊤μ̄_ϕ` — **exact within-batch repulsion** (cheap B×B kernel sum) **+ Nyström attraction** to a frozen full-data mean embedding `μ̄_ϕ=(1/n)Σψ(r_t)` over all n=1.28M training images, m=4096 k-means landmarks, cost O(Bm). **Eq 4** conditional joint law `L_joint=D(Φ∗pθ, Φ∗pdata)` with `Φ(x,c)=ϕ(x)⊕τ(c)` (image⊕text features). Data-side reference frozen once; generator side fresh each step.
- **Representation axis (§3.3).** A single encoder gives only a pseudo-metric; the combined kernel of a diverse panel is characteristic and vanishes only at real. **Constrained optimization**: each encoder required to reach its real-validation floor, weight = Lagrange multiplier under **proportional control + satisfaction gate** (PID-Lagrangian, Stooke 2020), fixed budget Σ=10; excess `e_ϕ=s_ϕ−b_ϕ` sets `λ_ϕ ∝ exp(e_ϕ/(τē))`. **Eq 5** the eval metric `SWr_k = (1/k)Σ SW(ϕ_e∗pθ, ϕ_e∗ptrain)/SW(ϕ_e∗pval, ϕ_e∗ptrain)`, k=14, real scores 1 by construction.
- **Putting it together (iRDM):** exact repulsion + Nyström attraction to a once-frozen reference, large fresh batches, joint image-text law on conditional tasks, 10-encoder battery balanced by constrained optimization. **Nothing else enters: no online teacher, adversary, or reward model.**

---

## 3. Tables — verbatim with sourcing

### Table 1 — SWr14 across released ImageNet-256 generators (L410–631) [primary metric]
SW = Sliced-Wasserstein ratio SW(gen,train)/SW(val,train); ≈1 matches a fresh real draw, lower=closer. SWr14 = arithmetic mean over 14 encoders. SWr14† = mean over the 4 held-out encoders (DINOv2, SigLIP-v1, RADIO, FLUX). ⋆ = external representation encoder used in training. (Header columns mangled by pdftotext; the two summary columns SWr14/SWr14† are the verifiable aggregates.)

| Model | SWr14↓ | SWr14†↓ |
|---|---|---|
| Validation baseline | **1.00** | 1.00 |
| Drifting-L⋆ | 5.93 | 4.91 |
| iMF-XL | 5.02 | 4.08 |
| Open-MAGVIT2-L | 4.30 | 4.03 |
| SiT-XL/2 | 4.27 | 3.63 |
| pMF-H (base) | 4.09 | 3.63 |
| DiT-XL/2 | 3.98 | 3.39 |
| VAR-d30 | 3.95 | 3.51 |
| JiT-H | 3.94 | 3.39 |
| MDTv2-XL/2 | 3.90 | 3.14 |
| MAR-H | 3.87 | 3.18 |
| DDT-XL/2⋆ | 3.77 | 3.11 |
| SiT-XL/2+REPA⋆ | 3.61 | 2.99 |
| REG-XL⋆ | 3.21 | 2.62 |
| LightningDiT-XL⋆ | 3.10 | 2.69 |
| RAE-XL⋆ | 2.43 | 2.18 |
| REPA-E SiT-XL/1⋆ | 2.40 | 2.14 |
| pMF-H (FD-SIM)⋆ *(prior best)* | 2.05 | 1.98 |
| **iRDM (ours)⋆** | **1.30** | **1.54** |

*Per-encoder iRDM row (14 cells, mean=1.3007→1.30 EXACT): 1.27, 0.98, 1.35, 0.83, 1.30, 1.02, 1.11, 1.90, 1.22, 1.56, 1.55, 1.44, 1.32, 1.36. iRDM is the best (lowest) entry on 9/14 encoders; cedes 5 (Inception/ConvNeXt/MAE to FD-SIM which games them below the 1.00 floor, e.g. FD-SIM Inception SW=0.67; DreamSim by a hair; held-out FLUX VAE to MAR-H).*

### Table 2 — GenEval + PickScore, one-step FLUX.2 [klein] post-training (L681–692)
GenEval per-category (Single-Obj / Two-Obj / Counting / Colors / Position / Color-Attr) + Overall + PickScore (500 COCO val prompts).

| Method | Single | Two | Count | Color | Pos | Attr | **Overall** | PickScore |
|---|---|---|---|---|---|---|---|---|
| FLUX.2 [klein] (4-step) | 0.994 | 0.904 | 0.791 | 0.880 | 0.575 | 0.623 | **0.794** | 22.58 |
| Untrained (1-step) | 0.894 | 0.323 | 0.603 | 0.673 | 0.225 | 0.128 | 0.474 | 19.95 |
| DMD2 (1-step) | 0.997 | 0.894 | 0.806 | 0.864 | 0.603 | 0.660 | 0.804 | 22.36 |
| iRDM (1-step, marginal) | 0.991 | 0.899 | 0.763 | 0.910 | 0.638 | 0.608 | 0.801 | 22.70 |
| **iRDM (1-step)** | 0.994 | 0.924 | 0.756 | 0.923 | 0.650 | 0.708 | **0.826** | **22.76** |

*Overall = mean of 6 categories, EXACT for all 5 rows (0.7945/0.4743/0.8040/0.8015/0.8258).*

### Table 3 — Per-encoder weighting: gated proportional Lagrangian vs uniform (L726–740)
100 steps from pMF-H on the SWr14 panel (lower better, real floor=1). Better arm bold.

| Aggregate | pMF-H (start) | **Gated** | Uniform |
|---|---|---|---|
| SWr14↓ | 2.09 | **1.88** | 1.90 |
| max↓ | 4.83 | **3.49** | 4.06 |

### Table 4 — Training-distance ablation on DINOv2 (cls) (L743–754)
Six fine-tuning losses warm-start the same pMF-H checkpoint, flip only the per-step distance, 100 steps. Floor-normalized ratio under two neutral distances (lower=closer).

| DINOv2 cls ratio↓ | baseline | mmdx | mmd_rff | mmd_exact | fd | sw | drifting |
|---|---|---|---|---|---|---|---|
| SW | 1.927 | **1.420** | 1.466 | 1.492 | 1.547 | 1.583 | 1.746 |
| RFF-MMD | 10.393 | **4.495** | 4.839 | 5.438 | 5.798 | 6.413 | 8.258 |

*Order `mmdx ≻ mmd_rff ≻ mmd_exact ≻ fd ≻ sw ≻ drifting` IDENTICAL on both neutral metrics. **Exact MMD does not beat Nyström** (mmdx 1.420 < mmd_exact 1.492) — the cheaper estimator wins because the exact full pairwise cross-term injects reference noise. mmdx = biased MMD² with exact within-batch term + Nyström cross-term (4096 landmarks); arms share AdamW lr 1.6e-6, batch 5120, RBF σ=65.*

### Table 5 — The 14-encoder panel (L1176–1196)
10 training / 4 held-out (eval only). Pool: CLS / AVG / ATTN / summary / patch-mean.

| Encoder | Checkpoint | Arch | Input | Pool | D |
|---|---|---|---|---|---|
| **Training (10)** | | | | | |
| Inception-v3 | FID Inception-v3 | CNN | 299 | avg | 2048 |
| ConvNeXt V2-B | convnextv2 base.fcmae ft in22k in1k | CNN | 224 | avg | 1024 |
| MAE | vit large patch16 224.mae | ViT-L/16 | 224 | avg | 1024 |
| CLIP | vit large patch14 clip 224.openai | ViT-L/14 | 256 | cls | 1024 |
| DINOv3-L | vit large patch16 dinov3.lvd1689m | ViT-L/16 | 224 | cls | 1024 |
| PE-Core-L | vit pe core large patch14 336.fb | ViT-L/14 | 224 | attn | 1024 |
| SigLIP2-So400m | vit so400m patch16 siglip 256.v2 webli | ViT-So400m/16 | 224 | attn | 1152 |
| AIMv2-H | aimv2 huge patch14 224.apple pt | ViT-H/14 | 224 | avg | 1536 |
| Web-SSL DINO 1B | webssl-dino1b-full2b-224 | ViT-1B | 224 | cls | 1536 |
| DreamSim | DINO+CLIP+OpenCLIP ensemble | ViT ens. | 224 | cls | 1792 |
| **Held out (4)** | | | | | |
| DINOv2 | vit large patch14 dinov2.lvd142m | ViT-L/14 | 256 | cls | 1024 |
| SigLIP (v1) | vit so400m patch14 siglip 384.webli | ViT-So400m/14 | 384 | attn | 1152 |
| C-RADIOv3-L | NVIDIA C-RADIOv3-L | ViT-L multi-teacher | 256 | summary | 3072 |
| FLUX VAE | FLUX.1 VAE, 4×4 patch-mean | VAE | 256 | patch-mean | 1024 |

### Table 6 — Generation batch-size sweep, matched wall-clock ≈6000s (L1201–1212)
Single-encoder DINOv2 Nyström-MMD arm; lr scaled ∝√N (Malladi 2022). SW ratios (lower=closer).

| Batch N | lr | DINOv2↓ | SWr14↓ |
|---|---|---|---|
| untrained base | n/a | 1.927 | 2.085 |
| 512 | 5.1e-7 | 2.067 | 2.521 |
| 1280 | 8.0e-7 | 1.429 | 2.061 |
| 2560 | 1.1e-6 | 1.363 | 2.053 |
| **5120** | 1.6e-6 | **1.253** | **2.006** |
| 10240 | 2.3e-6 | 1.285 | 2.027 |

*Smallest batch (N=512) **regresses above the untrained base** (2.521 > 2.085) despite the most optimizer steps; optimum is broad (N=2560–10240 all ≈2.006), single best at N=5120 — "optimum above 2048" ✓.*

### Table 7 — MMD-RFF distance ratio (MMDR14) counterpart of Table 1 (L1244–1480)
Same 14 encoders, kernel-MMD ratio (parentheses = raw mmd²(val,train)×10³ normalizer). Validation=1 by definition.

| Model | MMDR14↓ |
|---|---|
| Validation baseline | 1 |
| pMF-H (base) | 43.7 |
| REPA-E SiT-XL/1⋆ | 14.0 |
| RAE-XL⋆ | 13.0 |
| pMF-H (FD-SIM)⋆ *(prior best)* | 10.3 |
| **iRDM (ours)⋆** | **2.69** |

*iRDM per-encoder (14 cells, mean=2.6843→2.69 EXACT): 1.54, 0.98, 3.52, 0.69, 4.76, 1.17, 1.39, 2.69, 1.62, 3.16, 5.31, 3.12, 1.80, 5.83. Confirms the gaming thesis: FD-SIM Inception MMDR=0.22 (and SW=0.67) — driven below the real floor of 1 while samples stay flawed.*

### Table 8 — DMD2 one-step student over distillation steps (L1503–1512)
Best LAION-prompt config vs the 4-step FLUX.2 [klein] teacher. The 500-step peak is the Table-2 baseline.

| Step | GenEval | PickScore |
|---|---|---|
| Teacher (4-step) | 0.794 | 22.58 |
| 250 | 0.778 | 22.17 |
| **500 (reported)** | **0.804** | 22.36 |
| 750 | 0.792 | 22.36 |
| 1000 | 0.793 | 22.27 |

*Cross-table byte-identity: T8 row "500 (reported)" 0.804/22.36 == T2 DMD2 row ✓. Peak-then-erode profile (0.778→0.804→0.792→0.793); reported at the 500-step peak (conservative — favorable to the DMD2 baseline).*

---

## 4. Source-free reconciliation (Python-verified)

EVERY prose delta recomputes EXACT or within rounding — **zero numeric prose-vs-table defects**:
- SWr14 iRDM = 18.21/14 = **1.3007 → 1.30** ✓; FD-SIM = 28.74/14 = **2.0529 → 2.05** (prior SOTA) ✓; improvement **36.6% lower**.
- MMDR14 iRDM = 37.58/14 = **2.6843 → 2.69** ✓; FD-SIM 10.3 ✓.
- **All 5 Table-2 Overall = mean(6 categories) EXACT**: 0.7945→0.794, 0.4743→0.474, 0.8040→0.804, 0.8015→0.801, 0.8258→0.826 ✓.
- GenEval iRDM vs 4-step = **+4.03% rel / +3.2pp** (0.826 vs 0.794) ✓; vs DMD2 +2.74% rel ✓; joint vs marginal **+3.12% rel** ("0.801→0.826") ✓.
- PickScore iRDM vs 4-step = **+0.18 abs / +0.8% rel** (22.76 vs 22.58) ✓; ImageNet iRDM 20.96 vs FD-SIM 20.61 = +1.7% rel (71.2% win, paired z=30.5) ✓.
- T3 gated vs uniform: SWr14 mean 1.88 vs 1.90 = **1.05% rel (within noise)**; max 3.49 vs 4.06 = **14.0% rel cut** vs start 4.83 ✓.
- T4 order `mmdx>mmd_rff>mmd_exact>fd>sw>drifting` **identical on SW and RFF-MMD** ✓; Nyström (mmdx 1.420) **beats exact** (1.492) ✓.
- T6 best N=5120 (SWr14 2.006), "optimum above 2048" ✓; N=512 regresses (2.521 > base 2.085) ✓.
- Cross-table byte-identity: T8 "500 (reported)" 0.804/22.36 == T2 DMD2 row ✓; T1 FD-SIM SWr14 2.05 == §3.3/§4.1 "previous SOTA 2.05"/"strongest reaching about 2.05" ✓; Fig-6 PickScore iRDM 20.96 == body L674 ✓.

---

## 5. ⚠ Honest-scope flags (12; no numeric cell typo — all attribution/framing/scope)

1. **"Surpasses the 4-step teacher on GenEval 0.826 vs 0.794" is overall-only, Counting-blind.** iRDM **LOSES Counting** (0.756 < 0.791, −4.42%) and **TIEs Single-Obj** (0.994=0.994); the overall +3.2pp is carried by Colors (+4.3pp), Position (+7.5pp), Color-Attr (+8.5pp), Two-Obj (+2.0pp). Body L711 admits "trails only on counting"; the abstract cites only the overall. *(category-mixed headline — extends the iter-84 "best-Average-carried" class.)*
2. **PickScore margin vs the 4-step teacher is tiny: +0.18 (22.76 vs 22.58, +0.8% rel).** "Surpassing… on PickScore, 22.76 to 22.58" is a near-tie presented as a win; no CIs/seeds. The real headline is 4-step→1-step (4× NFE cut) at near-equal PickScore, not a quality gain.
3. **"First one-step generator to pass the real-image PickScore" (63.6% vs real photos) is a learned-metric artifact, not "better than real."** PickScore (a trained preference model) is known to favor synthetic sharpness/texture over real photos; beating real on it is a metric-ceiling inversion, and PickScore is itself the paper's *off-objective* corroborator — both the SOTA claim and the "beats real" claim rest on the same proxy.
4. **SWr14 "state of the art" is aggregate-carried; iRDM cedes 5/14 encoders.** Best on 9/14 + the aggregate, but loses Inception/ConvNeXt/MAE (to FD-SIM, which games them below the 1.00 real floor), DreamSim "by a hair", and held-out FLUX VAE (to MAR-H). The aggregate win depends on the paper's own thesis that sub-floor single-encoder scores are gamed and should be discounted — true, but the SOTA is "best on the non-gamed aggregate", not on every encoder.
5. **The 71.2% PickScore headline is iRDM's *lowest* win rate, not its typical one.** iRDM beats RAE-XL 75.7% and REPA-E 73.2% but only FD-SIM 71.2%. Citing 71.2% (vs the strongest opponent, prior SWr14 SOTA) is *conservative/honest*, but a casual reader can read 71.2% as the representative margin when it is the floor.
6. **T3 "constrained optimization" contribution rests on the worst-case (max), not the mean.** Gated vs uniform SWr14 mean 1.88 vs 1.90 = 1.05% (within noise); only the **max** (3.49 vs 4.06, 14% cut) separates them. The run is 100 steps (not converged); "we make no claim about perceptual quality" (L758). The mean improvement is not by itself evidence for constrained weighting.
7. **"No online teacher" is true for the ImageNet arm but NOT teacher-free for the FLUX arm.** §4.2: the FLUX post-training reference is "collected once from the four-step teacher and then frozen" — ~300K **teacher-generated** PickScore-ranked COCO + GenEval-correct samples compressed into the Nyström reference. So the teacher's knowledge is distilled into the reference once; "no online teacher" ≠ "teacher-free". The ImageNet arm (reference = real ImageNet) IS genuinely teacher-free.
8. **SWr14 and MMDr14 are both author-defined panels; no external/community FID/CLIP-score table.** The SOTA is on metrics the authors constructed (14 chosen encoders, 4 chosen hold-outs, Sliced-Wasserstein). Appendix D notes MMDr14 "broadly agrees, with some reordering among the mid-field models" — the top (iRDM/FD-SIM) is stable but mid-field ranking is metric-sensitive.
9. **Single-encoder gaming demonstration (Fig 5, T1/T7 sub-floor) is the strongest claim but rests on a figure + 1 DINOv2 run.** "DINOv2 driven to SW=1.01 (real floor) yet visibly fake" is Fig-5-qualitative (lizard vs typewriter) + one number (N=5120, 1000 steps); no tabulated per-class realism under single-encoder matching.
10. **T2 DMD2 baseline is a self-reimplementation (Appendix E.2), not the released DMD2.** Re-implemented for klein's flow-matching parameterization; released DMD2 targets SD-UNet under ϵ-prediction. The 0.804/22.36 baseline is the authors' own LAION-prompt 500-step peak — a faithful-but-reimplemented comparator, not an off-the-shelf number.
11. **"About 90 H200 GPU-hours" (FLUX post-training) vs DMD2 "~10 H200 GPU-hours" (Appendix E.2) — the cost comparison is not the paper's selling point but the efficiency gap is real and understated.** iRDM is ~9× the DMD2 distillation compute for +0.022 GenEval (0.826 vs 0.804); the value proposition is *teacher-free + no critic/generator alternation + generality*, not compute. (DMD2 needs 3 networks + per-step critic updates.)
12. **No seeds/CIs on any table; decisive margins are small.** GenEval +3.2pp (counting-blind), PickScore +0.18, T3 mean 1.88-vs-1.90, "cedes DreamSim by a hair" — all within plausible run noise; only the broad SWr14 gap (1.30 vs 2.05) and the gaming-below-floor cells are decisively outside noise.

---

## 6. Citable falsifiable content (NOT the framing-level claims)

- **The two-axis decomposition (§3)** locates every prior method's ceiling in a *specific design choice* (Fréchet→two-moment saturation; drifting→batch-rebuilt reference confines to small batches; both→few fixed-weight encoders are gameable). Falsifiable: vary one axis at a time and the ranking is explained — confirmed by T4 (distance), T6 (batch), T3 (weighting).
- **Nyström-MMD beats exact MMD (T4)** — counterintuitive, reproducible (reference-noise argument, §3.1). The cheaper estimator is strictly better on both neutral metrics.
- **Single-encoder gaming driven below the real floor (Fig 5 + T1/T7)** — FD-SIM Inception SW=0.67 / MMDR=0.22 (<1.00 real) while images stay flawed; DINOv2-alone driven to SW=1.01 (the floor) yet visibly fake. This is the empirical engine of the multi-encoder thesis.
- **SWr14 (Eq 5)** — a 14-encoder Sliced-Wasserstein ratio, training-loss-independent (Sliced-Wasserstein shares no estimator with the trained MMD), 4 encoders held out as a generalization check (SWr14† iRDM 1.54 still best).
- **NOT** "surpasses 4-step on GenEval" (counting-blind, +3.2pp), NOR "beats real PickScore" (learned-metric artifact), NOR "constrained optimization helps" (mean within noise; only worst-case), NOR "no teacher" (FLUX arm uses a teacher-generated frozen reference).
