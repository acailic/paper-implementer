# Object-centric LeJEPA — source-first breakdown

**Paper:** "Object-centric LeJEPA" — Jakob Geusen, Ender Konukoglu — Biomedical Image Computing Group, ETH Zurich.
**arXiv:** 2607.02404v1 [cs.CV], 2 Jul 2026. **Length:** 15 pp (pdfinfo=15; `file` misreports 8 pp — file-vs-pdfinfo defect recurs, trust pdfinfo).
**Source files:** `paper.pdf` (16.2 MB), `paper_layout.txt` (pdftotext -layout, 839 lines). 7 explicit tables + Eqs 1–5 + Figs 1–4.
**Repo slot:** 62nd paper, rank 57 unique. FIRST object-centric / scene-partitioning / object-level-SSL / mask-as-given-pretraining paper (prior "object" keyword hits unrelated; no folder covers object-centric representation learning).

---

## 1. Problem & thesis

Image-level SSL (LeJEPA and kin) aligns augmented views of the *whole* image. Under random cropping two views capture different scene regions / different objects, so forcing view-agreement pushes the encoder toward global high-level semantics and works against assigning different objects different representations. Object-centric learning promises greater **data efficiency** via compositional generalization (an object's representation transfers across the many scenes it appears in), but doing partitioning + representation jointly end-to-end is **unstable** on natural images without a frozen pretrained encoder — a cyclic dependency (partitioning needs meaningful features; meaningful features need consistent partitioning).

**Thesis / sidestep:** take object masks as *given* during training (cheap off-the-shelf **SAM 2** proposals), train the encoder from scratch, and **move LeJEPA's alignment + anti-collapse regularization from the image to the object level**. At inference no masks are needed — the backbone yields informative patch features directly.

## 2. Method (§4, Eqs 1–5)

Object masks precomputed once with **SAM 2 automatic mask generator**, prompted with a regular 16×16 point grid (avoids repeated computation across epochs). Train encoder `g` so each patch encodes both instance-level and semantic object info. An object is "present in a view" iff its view-projected mask covers ≥16×16 px (the ViT patch size). `K_{n,v}⊆K_n` = objects in view v of image n; `V_{n,k}` = views where object k appears; projection dim d=64.

**Semantic object representation** `z_{n,v,k} = f(g(x) within mask)` (Fig 2): mask-weighted mean of independently projected patch features → cross-attention masking out keys/values outside the mask → +residual from weighted mean → residual MLP. Principle: "a whole is defined by its parts."

**Instance-level object representation** (Eq 1) — "a part should know which whole it belongs to": MLP `ϕ` on the i-th patch feature then ℓ2-normalize:
  `y_{n,v,i} = ϕ(g(x_{n,v})_i) / ‖ϕ(g(x_{n,v})_i)‖₂`

### 2.1 Object-centric LeJEPA loss (§4.2, Eqs 2–3)
Effective object set `K_n^eff = {k∈K_n | |V_{n,k}|≥2}` (align only objects in ≥2 views).
- **Alignment term** (Eq 2): `ℓ^pred_{n,v,k} = ½‖µ_{n,k} − z̃_{n,v,k}‖₂²`, where `µ_{n,k} = (1/|V_{n,k}|)Σ_{v∈V_{n,k}} z̃_{n,v,k}` (mean of object k across views). `L^pred` = average over all object-level alignment terms.
- **Regularization term `L^SIGReg`** [LeJEPA/Balestriero-Lecun] on projected object reps `z̃_{n,v,k}` to prevent collapse. Computes the empirical characteristic function over ALL object reps from all B images (not per-image). Original LeJEPA uses the **Epps–Pulley test statistic scaled by sample size**; here sample size varies a lot across batches (object count per view is far from constant), so the EP statistic is **re-scaled by a constant B** to stabilize gradient magnitude.
- Combined (Eq 3): `L^ObjectLeJEPA = L^pred + λ_LeJEPA·L^SIGReg`, **λ_LeJEPA = 0.05** (per LeJEPA).

### 2.2 Instance-level loss (§4.3, Eq 4) — supervised-contrastive [Khosla] formulation
For patch i: `A(i)` = all other patches in same view; `P(i)⊆A(i)` = those sharing its dominant mask; temperature **τ=0.1**. Per-anchor (Eq 4):
  `ℓ^instance_{n,v,i} = −(1/|P(i)|) Σ_{p∈P(i)} log[ exp(y_{n,v,i}ᵀy_{n,v,p}/τ) / Σ_{a∈A(i)} exp(y_{n,v,i}ᵀy_{n,v,a}/τ) ]`
Computed **independently within each view** (separating co-occurring objects is intra-image; cross-view consistency already handled by `L^pred`). Patches assigned to a mask if it covers ≥50% of the mask; **background patches serve as negatives but are not anchors**. Averaged over patches-in-view, then over views and images.

### 2.3 Total objective (Eq 5)
  `L = L^ObjectLeJEPA + L^instance`

## 3. Setup (§5.1)

- **Pretrain:** COCO, **150 epochs**, ViT-Base/16 (also ViT-S for scale ablation); λ_LeJEPA=0.05, weight decay 0.05, LeJEPA proj dim 64.
- **Optimizer:** AdamW, batch 256, lr **5e-4**, cosine schedule + 1 linear warm-up epoch.
- **Views:** 2 global views 256×256 + 8 local views 128×128, standard LeJEPA augmentations. Same hyperparameters for image-level LeJEPA and Object LeJEPA.
- **Baselines compared:** image-level LeJEPA, SlotMIM (150 ep, also SlotMIM-800 ep), **DINOv3 ViT-B as upper bound** (official checkpoint, trained at scale). All encoders frozen for downstream probes.
- **Downstream:** linear-probe ImageNet-1k classification; ADE20k dense (linear-probe semantic seg; FG-ARI/mIoU K-Means instance clustering; quadratic-probe same-instance prediction; object classification); DAVIS nearest-neighbour label-propagation tracking; NAVI few-shot instance re-ID.

## 4. Results — all 7 tables verbatim (sourcing line-ranges)

### Table 1 — FG-ARI & mIoU: K-Means clusters of frozen patch features vs GT instance masks (L306–313)
| Encoder | FG-ARI | mIoU |
|---|---|---|
| Image LeJEPA | 0.285 | 0.229 |
| SlotMIM (800) | 0.347 | 0.277 |
| SlotMIM | 0.343 | 0.271 |
| **Object LeJEPA** | **0.431** | **0.355** |
| DINOv3 (upper bound) | 0.357 | 0.300 |

Object LeJEPA leads AND overtakes DINOv3 on both (FG-ARI 0.357→0.431, mIoU 0.300→0.355).

### Table 2 — Quadratic probe: accuracy & AUC predicting whether two patches share an instance (L329–336)
| Encoder | Accuracy | AUC |
|---|---|---|
| Image LeJEPA | 0.877 | 0.914 |
| SlotMIM (800) | 0.894 | 0.938 |
| SlotMIM | 0.890 | 0.931 |
| **Object LeJEPA** | **0.915** | **0.954** |
| DINOv3 | 0.914 | 0.957 |

Object LeJEPA 0.915/0.954 "effectively matching DINOv3" (0.914/0.957) — Acc edges ahead, AUC 0.003 behind.

### Table 3 — Linear-probe semantic segmentation on ADE20k: mIoU & pixel acc (L370–376)
| Encoder | mIoU | Pixel acc. |
|---|---|---|
| Image LeJEPA | 0.339 | 0.664 |
| SlotMIM (800) | 0.409 | 0.735 |
| SlotMIM | 0.368 | 0.696 |
| **Object LeJEPA** | **0.418** | **0.739** |
| DINOv3 | 0.500 | 0.800 |

Object LeJEPA best among COCO-trained (0.418 > SlotMIM-800 0.409 by 0.009); DINOv3 ahead at 0.500.

### Table 4 — DAVIS label propagation (nearest-neighbour on frozen patch features): F_m, J&F mean, J_m (L331–337)
| Encoder | F_m | J&F | J_m |
|---|---|---|---|
| Image LeJEPA | 0.632 | 0.613 | 0.594 |
| SlotMIM (800) | 0.642 | 0.629 | 0.615 |
| SlotMIM | 0.623 | 0.609 | 0.595 |
| **Object LeJEPA** | 0.713 | **0.682** | 0.650 |
| DINOv3 | 0.744 | 0.715 | 0.685 |

Object LeJEPA J&F 0.682, well above Image LeJEPA (0.613) and SlotMIM-800 (0.629); narrows gap to DINOv3 (0.715).

### Table 5 — Object classification on ADE20k: top-1 & top-5 balanced acc (L389–396). `*` = Object LeJEPA's native semantic object rep z.
| Encoder | Top-1 | Top-5 |
|---|---|---|
| Image LeJEPA | 0.168 | 0.307 |
| SlotMIM (800) | 0.201 | 0.328 |
| SlotMIM | 0.181 | 0.311 |
| Object LeJEPA | 0.212 | 0.364 |
| Object LeJEPA* | 0.250 | 0.420 |
| DINOv3 | 0.367 | 0.592 |

Object LeJEPA 0.212 best COCO-trained on avg-pooled; native z lifts to 0.250; all COCO models trail DINOv3 (0.367).

### Table 6 — Linear-probe classification on ImageNet-1k: top-1 & top-5 balanced acc (L403–414). `*` = averaged patch tokens; unstarred = [CLS] token.
| config | Top-1 | Top-5 |
|---|---|---|
| Image LeJEPA* | 0.463 | 0.708 |
| Image LeJEPA | 0.473 | 0.718 |
| SlotMIM (800)* | 0.552 | 0.798 |
| SlotMIM* | 0.467 | 0.713 |
| Object LeJEPA* | 0.539 | 0.784 |
| DINOv3* | 0.776 | 0.946 |
| DINOv3 | 0.790 | 0.951 |

Avg-pooled Object LeJEPA 0.539 improves over LeJEPA-[CLS] 0.473 (+6.6 pp) and trails SlotMIM-800* 0.552 by 1.3 pp.

### Table 7 — Loss & mask ablation, full COCO, ViT-B (L493–503). DAVIS J&F / IN-1k Top-1 / ADE20k mIoU / NAVI 1-shot.
| Loss | DAVIS J&F | IN-1k Top-1 | ADE20k mIoU | NAVI 1-shot |
|---|---|---|---|---|
| Image LeJEPA | 0.613 | 0.463 | 0.339 | 0.337 |
| Object alignment | 0.642 | 0.545 | 0.400 | 0.380 |
| Instance Separation | 0.678 | 0.490 | 0.396 | 0.393 |
| **Object LeJEPA** | **0.682** | 0.539 | **0.418** | **0.476** |
| Object LeJEPA GT (GT masks) | 0.689 | 0.554 | 0.399 | 0.421 |

## 5. Source-free reconciliation (all prose deltas recompute EXACT, 0 contradictions)

- **§5.2 T1 raise** "FG-ARI from 0.357 to 0.431 and mIoU from 0.300 to 0.355" ✓ EXACT (DINOv3→Object).
- **§5.2 T2** "Object LeJEPA reached 0.915 accuracy and 0.954 AUC … effectively matching DINOv3 (0.914/0.957)" ✓ EXACT.
- **§5.2 T3** "best mIoU among COCO-trained models (0.418), edging out SlotMIM (0.409)" ✓; "DINOv3 remained ahead at 0.500" ✓.
- **§5.2 T4** "improved the J&F mean to 0.682, well above image-level LeJEPA (0.613) and SlotMIM (0.629), and narrowed the gap to DINOv3 (0.715)" ✓ EXACT (order DINOv3 0.715 > Obj 0.682 > SlotMIM-800 0.629 > Img 0.613 > SlotMIM-150 0.609).
- **§5.2 T5** "Object LeJEPA again outperformed both baselines using averaged patch features (0.212 top-1), and its native slot representation lifted this further to 0.250, though all COCO-trained models trail DINOv3 (0.367)" ✓ EXACT.
- **§5.2 T6** "improved top-1 balanced accuracy over LeJEPA from 47.3% to 53.9% and trailed SlotMIM trained for 800 epochs (55.2%) by only a small margin" ✓ EXACT — note baseline is Image-LeJEPA **[CLS]** 0.473 (not the `*` avg-pool 0.463); delta +6.6 pp; SlotMIM-800 gap 1.3 pp.
- **§5.3 data efficiency** "trained on only 10% of COCO, Object LeJEPA ViT-B already matched image-level LeJEPA trained on full COCO on every task" — 10% Object (0.648/0.472/0.380/0.343) ≥ full Image (0.613/0.463/0.339/0.337) on all 4 ✓ direction (these are **Figure-4 bar readings**, see honest-scope flag 3).
- **§5.4 loss ablation** "Combining the two losses improved every task over either loss in isolation, with the sole exception of image classification, where the alignment-only setting remained marginally ahead (54.5% vs 53.9%)" ✓ EXACT: combined beats both losses on DAVIS/ADE20k/NAVI; on IN-1k alignment-only 0.545 > combined 0.539.
- **§5.4 mask ablation** "ground-truth masks were marginally ahead on tracking and image classification, whereas SAM masks were better on segmentation and substantially better on NAVI re-identification" ✓ EXACT: GT-vs-SAM DAVIS 0.689>0.682, IN-1k 0.554>0.539, ADE20k 0.399<0.418, NAVI 0.421<0.476.

**NO numeric prose-vs-table contradiction.** Every checked prose number recomputes from a table cell.

## 6. Honest-scope flags (⚠, transcribed verbatim NOT contradicted)

1. **T6 baseline switch (benign):** "improved over LeJEPA from 47.3% to 53.9%" uses Image-LeJEPA **[CLS]**-token (0.473) not the like-for-like `*` avg-pool (0.463) — Object LeJEPA has no [CLS] so avg-pools; the apples-to-apples gain is +7.6 pp (0.539 vs 0.463), not the quoted +6.6 pp. Defensible (Object has no [CLS]) but the baseline choice flatters the delta slightly.
2. **"Upper bound" DINOv3 beaten selectively:** DINOv3 is framed as upper bound yet Object LeJEPA **overtakes** it on T1 (FG-ARI 0.431>0.357, mIoU 0.355>0.300) and edges/matches it on T2 (Acc 0.915>0.914, AUC 0.954<0.957 by 0.003) — i.e. on **instance-awareness** DINOv3 is NOT the bound. DINOv3 dominates the **semantic** tasks (T3 mIoU 0.500 vs 0.418; T5 top-1 0.367 vs 0.250; T6 0.790 vs 0.539) and DAVIS T4 (0.715 vs 0.682). "Effectively matching DINOv3" (T2) carries no data-scale caveat though DINOv3 trained on a far larger corpus.
3. **Figure-only results:** NAVI re-ID full curves (Fig 3) and the dataset-size ablation (Fig 4) are bar readings, not table cells. The §5.3 10%-Object numbers (0.648/0.472/0.380/0.343) and image-full NAVI 0.337 are figure-derived; only T7's NAVI 1-shot cells (Object-full 0.476, Image-full 0.337) are table-verifiable. Fig-3 DINOv3 single-shot 92.2% is figure-only and sits off the shown 0.8-axis range.
4. **No seeds / CIs / significance tests:** all tables are single point estimates; several "wins" are sub-0.01 (T3 mIoU Object 0.418 vs SlotMIM-800 0.409 = 0.009; T2 AUC 0.954 vs DINOv3 0.957 = 0.003; T7 combined IN-1k 0.539 vs alignment-only 0.545 = 0.006). Statistical significance unestablished.
5. **Instance loss hurts image classification:** combined Object LeJEPA (0.539) LOSES to alignment-only (0.545) on IN-1k — the instance-separation term is classification-neutral-to-harmful. The abstract's "outperforms … classification" still holds vs Image-LeJEPA (0.539>0.463) but is alignment-term-carried, not the full combined objective.
6. **Epoch-matching caveat:** Object LeJEPA trained 150 ep beats SlotMIM-800 on T1/T2/T4, but Object LeJEPA at 800 ep is untested — could be unfair to SlotMIM or Object may have saturated (unknown). 150-ep-vs-150-ep (Object vs SlotMIM) is the clean comparison and Object wins it.
7. **"Object-centric > image-level" is Object-LeJEPA-specific, not method-class-general:** SlotMIM (also object-centric) is the WEAKEST on NAVI re-ID (below image-level LeJEPA per §5.2/Fig 3) — its soft slot assignments blend global+local info. So the headline holds for Object LeJEPA's hard-mask design, not for object-centric SSL broadly.
8. **SAM masks BEAT ground-truth on 2/4 tasks (NAVI paradox):** Object LeJEPA GT (0.421) underperforms SAM (0.476) on NAVI by 0.055 and on ADE20k (0.399 vs 0.418). Unsupervised SAM oversegmentation gives cleaner instance boundaries than COCO GT for instance-discriminability — "SAM masks are sufficient" understates that SAM is sometimes *better*, which complicates the "isolate mask quality" framing of the GT row.
9. **COCO-only pretraining (≈118 k images):** "data efficiency" is relative to full-COCO image-level LeJEPA, NOT vs DINOv3's far-larger corpus; the 10×-data claim is within-COCO, not a statement about reaching DINOv3-scale efficiency.
10. **Conclusion restates only the data-efficiency headline** ("same downstream performance as image-level LeJEPA with 10% of the training data") — this is the Fig-4 10%-Object line, figure-derived (flag 3); the cleanest table-level win is the full-COCO T1–T5 dominance over COCO-trained baselines.

## 7. Strengths / limitations / verdict

**Citable falsifiable contribution:** moving LeJEPA's alignment + SIGReg anti-collapse term from the image to the **object** level, enabled by taking SAM masks as given (sidestepping the partition↔representation chicken-and-egg), plus a within-view supervised-contrastive instance term. The object-level SIGReg re-scaling by constant B (to absorb variable per-batch object counts) is the non-obvious engineering hinge that ports the Epps–Pulley statistic from fixed-B images to variable-size object sets.

**Limitations (authors' own + observed):** requires an external mask generator at train time (SAM 2); COCO-only; single ViT-B main scale; no seeds/CIs; DINOv3 upper bound beaten only on instance-awareness; figure-only NAVI + dataset-size claims.

**Verdict:** clean, honestly-reported paper — all checked prose deltas recompute EXACT, zero contradictions. Honest-scope surfaces are framing/attribution (DINOv3 "upper bound" selective; T6 baseline switch; figure-only §5.3) and statistical (no CIs), not numeric. Object-level LeJEPA's instance-awareness win over DINOv3 (T1/T2) at COCO scale is the strongest scoped result; the semantic-task and data-efficiency headlines are softer (DINOv3-dominated / figure-derived).
