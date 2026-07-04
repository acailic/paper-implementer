# FlowCIR: Semantic Transport via Flow Matching for Zero-Shot Composed Image Retrieval — Source-First Breakdown

**arXiv:** 2607.02284v1 [cs.CV] (2 Jul 2026) · **Repo:** 71st paper, rank 66 · **Venue:** ECCV-style preprint (no venue stamp); HKUST + HKU
**Authors:** Zhenqi He, Ziqi Jiang, Yuanpei Liu, Yanghao Wang, Teng Wang, Long Chen†
**Subarea (NEW for repo):** zero-shot composed image retrieval (ZS-CIR) + conditional flow matching (semantic transport in VLM embedding space). Repo's FIRST retrieval / CIR / flow-matching-for-retrieval paper. Sibling-in-spirit to the generation/flow-matching lineage (orbitquant iter-73 diffusion, wan-streamer video, pointdit iter-74 diffusion-for-geometry) but uniquely retrieval-as-transport; distinct from viq (visual tokens) and translation-as-bridging (text transfer).

**Source:** paper.pdf (3.5MB, **18pp pdfinfo — `file` misreports 1pp, 17-page gap, defect recurs**); paper_layout.txt (pdftotext -layout, 976 lines). All numbers below transcribed verbatim from paper_layout.txt with sourcing line-ranges; deltas recomputed in Python.

---

## 1. Problem & paradigm shift (L45–131)

**Composed Image Retrieval (CIR):** query = ⟨reference image I_r, relative text instruction T_r⟩ → retrieve target image I_t. **Zero-shot** setting: no domain-specific annotated triplets at train time.

**Prior paradigm (textual inversion, Fig 1b):** learn a projector I_r → few pseudo-text tokens, concatenate with T_r in TEXT space (Pic2Word, SEARLE, LinCIR, iSEARLE, Context-I2W). Two weaknesses the authors attack: (i) compressing a rich image into a handful of tokens loses fine-grained semantics; (ii) inversion-projector training needs hours of large-batch multi-GPU contrastive training on web-scale data (e.g. Compo-Diff = 128×A100, 231h; Pic2Word = 8×A100, 16h; SEARLE = 8×A100, 4h).

**FlowCIR paradigm (Fig 1c, L117–131):** reframe ZS-CIR as **conditional semantic transport** between reference and target distributions in VLM embedding space, instantiated by **conditional flow matching**. Learns a lightweight transport field mapping the **instruction** embedding toward a **target-aligned text** embedding, **conditioned on the reference image**. Trains only the small transport module — image & text encoders stay frozen — so training fits in **~0.5h on a single RTX 3090** (Fig 2 bubble: lowest cost, top performance).

**Second contribution — Multi-Negative Steering (L131–140, §3.3):** CLIP-style VLMs geometrically conflate affirmative and negated text ("a dog" ≈ "remove/no dog"). Inference-only strategy that steers a negation-containing instruction **away** from its negated semantics (no extra training).

## 2. Method (§3, L209–463)

### 2.1 Preliminaries
**Rectified-flow FM (L211–222).** Linear path `x_t = (1−t)x_0 + t·x_1`, t∈[0,1]; ground-truth velocity constant `v⋆(x_t,t) = x_1 − x_0`. Velocity field regressed by
- **Eq 1** `L_FM(θ) = E[ ‖v_θ(x_t,t) − (x_1−x_0)‖² ]`
- **Eq 2** conditional form `L^(FM)_cond(θ) = E[ ‖v_θ(x_t,t,c) − (x_1−x_0)‖² ]`

**Notation (L233–247).** Frozen image encoder E_I, text encoder E_T; all features ℓ2-normalized: `x_Ir=E_I(I_r)/‖·‖`, `x_Tr=E_T(T_r)/‖·‖`, `x_Tt`, `x_It`.

### 2.2 Conditional Flow Matching for Semantic Transport (L249–290)
Train zero-shot on synthesized tuples ⟨I_r, T_r, T_t, I_t⟩. Learn `(x_Tr, x_Ir) ↦ x_Tt`.
- Sample `t∼U(0,1)`, interpolate `x_t = (1−t)x_Tr + t·x_Tt`.
- Velocity net `v̂_θ = f_θ(x_t, t, x_Ir)` conditioned on reference image.
- **Eq 3** FM regression: `L_FM = E[ ‖v̂_θ(x_t,t,x_Ir) − (x_Tt − x_Tr)‖² ]`.
- One-step transport for predicted target text emb: `x̂_Tt = x_t + (1−t)·v̂_θ`.
- **Eq 4** top-K hard-negative index set `H_K(x̂_Tt) = TopK({ ⟨x̂_Tt/‖·‖, x^j_It⟩ }_{j≠i})`.
- **Eq 5** InfoNCE: `L_RET = −log[ exp(⟨x̂_Tt,x_It⟩/τ) / (exp(⟨x̂_Tt,x_It⟩/τ) + Σ_{j∈H_K} exp(⟨x̂_Tt,x^j_It⟩/τ)) ]`.
- **Total:** `L = L_FM + λ·L_RET`.
- **Inference:** one-step transport `f_θ(x_Tr, x_Ir, 0)` → `x̂_Tt` → nearest-neighbor in gallery.

### 2.3 Multi-Negative Steering (L292–416)
Parse T_r → affirmative intent T_r^a + negated concepts {T_r,k^n}. Model each concept as a **hyper-spherical cap** `R(x) = {z | x⊤z ≥ δ}`, δ∈[−1,1].
- **Eq 6** feasibility `R(T_r) = R(x_{T_r^a}) ∩ ⋂_{k=1..K} R^c(x_{T_r,k^n})` (close to affirmative, outside all negation neighborhoods).
- **Eq 7** closed-form "center" direction per negative: `d̃_k = x_{T_r^a}·sin(α+θ_k/2)/sin(θ_k) − x_{T_r,k^n}·sin(α−θ_k/2)/sin(θ_k)`, then `d_k = d̃_k/‖·‖`; `α=arccos(δ)`, `θ_k=arccos(x_{T_r^a}⊤x_{T_r,k^n})`. (Small θ_k ⇒ `d̃_k = x_{T_r^a} − x_{T_r,k^n}` for stability.)
- **Eq 8** aggregate+normalize: `x̂_T_r = (1/K)Σ_k d_k`, then renormalize; replaces instruction embedding in transport pipeline.

### 2.4 Theoretical justification (§3.4, L418–463)
- **Prop 1 (Mean Collapse, Eq 9):** unconditioned flow transports x_Tr to `E[x_Tt|x_Tr]`; if `p(x_Tt|x_Tr)` is multi-modal, the mean ∉ {x^1_Tt,…,x^K_Tt} ⇒ mean-collapse, can't recover correct target. ⇒ conditioning on reference image is **necessary**.
- **Eq 10** conditional probability path `p_t(x_t|x_1=x^i_Tt, x^i_Ir)`: `δ(x_t−x^i_Tr)` at t=0; `δ(t·x^i_Tt+(1−t)·x^i_Tr)` for t∈(0,1].
- **Prop 2 (Semantic Transport, Eq 11):** under this path, marginal velocity ≡ exact conditional velocity `v⋆(x_t,t,x_Ir) = v⋆(x_t,t|x^i_Ir,x^i_Tt)`.
- **Eq 12** ⇒ exact point-to-point transport `x_Tt = x_Tr + ∫_0^1 v⋆(x_t,t,x_Ir) dt`. Proof in supplementary. ⚠ **Idealized conditional-path assumption** — see honest-scope.

## 3. Experimental setup (§4.1, L466–491)
- **Backbones:** frozen CLIP ViT-B/32 and ViT-L/14.
- **Transport net:** lightweight deep residual MLP with timestep conditioning (design from MAR, ref [32]).
- **Train data:** HQ-Edit-200k [24] synthetic image-editing dataset (ref image + edit instruction + edited result + text description).
- **Neg parsing:** rule-based keywords (no/without/remove/replace) → TinyLlama-1.1B-Chat-v1.0 [59] decouples negative component.
- **Hardware:** single NVIDIA RTX 3090, 24GB; ~0.5h training.
- **Benchmarks:** CIRR [38], CIRCO [4], Fashion-IQ [54].

## 4. Tables (verbatim, with sourcing line-ranges)

### Table 1 — CIRR (R@1/5/10) & CIRCO (mAP@5/10/25/50) + training cost (L501–537)
**ViT-B/32:**
| Method | Cost | CIRR R@1 | R@5 | R@10 | CIRCO mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|---|---|---|---|---|---|---|---|---|
| Image-only | – | 6.7 | 23.0 | 59.2 | 1.5 | 1.9 | 2.3 | 2.6 |
| Text-only | – | 21.8 | 45.2 | 57.4 | 2.5 | 2.6 | 2.9 | 3.1 |
| PALAVRA | – | 16.6 | 43.5 | 58.5 | 4.6 | 5.3 | 6.3 | 6.8 |
| SEARLE | 8×A100,4h | 24.3 | 53.3 | 66.1 | 8.9 | 9.4 | 10.6 | 11.2 |
| iSEARLE | A100,12h | 25.2 | 55.7 | 68.1 | 10.6 | 11.2 | 12.5 | 13.3 |
| SEARLE+CIG | NA | 25.3 | 54.8 | 68.1 | 10.2 | 10.6 | 11.8 | 12.5 |
| MagicLens | 64 TPU,6h | 27.0 | 58.0 | 70.9 | 23.1 | 23.8 | 25.8 | 26.7 |
| **Ours** | RTX3090,0.5h | 25.5 | 56.5 | 69.8 | 13.1 | 13.4 | 14.6 | 15.3 |

**ViT-L/14:**
| Method | Cost | R@1 | R@5 | R@10 | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|---|---|---|---|---|---|---|---|---|
| Image-only | – | 7.3 | 23.0 | 33.3 | 2.5 | 3.1 | 3.9 | 4.4 |
| Text-only | – | 20.9 | 44.0 | 55.4 | 3.3 | 3.7 | 4.1 | 4.4 |
| Pic2Word | 8×A100,16h | 23.9 | 51.7 | 65.3 | 8.7 | 9.5 | 10.6 | 11.3 |
| SEARLE | 8×A100,4h | 24.2 | 52.4 | 66.3 | 11.7 | 12.7 | 14.3 | 15.1 |
| Context-I2W | 8×A100,24h | 25.6 | 55.1 | 68.5 | – | – | – | – |
| LinCIR | 8×A100,0.5h | 25.0 | 53.3 | 66.7 | 12.6 | 13.6 | 15.0 | 15.9 |
| LinCIR+CIG | NA | 25.6 | 54.8 | 67.6 | 13.0 | 13.6 | 15.1 | 16.0 |
| Compo-Diff | 128×A100,231h | 18.2 | 53.1 | 70.8 | 12.6 | 13.4 | 15.8 | 16.4 |
| iSEARLE | 12h | 25.4 | 54.1 | 67.5 | 11.3 | 12.7 | 14.5 | 15.3 |
| MagicLens | 128 TPU,6h | 30.1 | 61.7 | 74.4 | 29.6 | 30.8 | 33.4 | 34.4 |
| MCL (LLaMA2-7B) | NA | 26.2 | 56.8 | 70.0 | 17.7 | 18.9 | 20.8 | 21.7 |
| **Ours** | RTX3090,0.5h | 26.2 | 56.1 | 68.6 | 14.9 | 15.7 | 17.3 | 18.2 |

### Table 2 — Fashion-IQ R@10 / R@50 (Dress/Shirt/Toptee + Avg) (L567–590)
**ViT-B/32** (cols: Dress R10/R50, Shirt R10/R50, Toptee R10/R50, Avg R10/R50):
| Method | Dr10 | Dr50 | Sh10 | Sh50 | To10 | To50 | Av10 | Av50 |
|---|---|---|---|---|---|---|---|---|
| Image-only | 3.9 | 10.8 | 7.5 | 14.0 | 6.2 | 13.4 | 5.9 | 12.7 |
| Text-only | 13.6 | 31.8 | 20.3 | 35.3 | 20.2 | 40.5 | 18.0 | 35.9 |
| PALAVRA | 17.3 | 35.9 | 21.5 | 37.1 | 20.6 | 38.8 | 19.8 | 37.3 |
| SEARLE | 18.2 | 38.6 | 24.8 | 41.1 | 25.6 | 46.2 | 22.9 | 42.0 |
| MagicLens | 21.5 | 41.3 | 27.3 | 48.8 | 30.2 | 52.3 | 26.3 | 47.4 |
| **Ours** | 24.8 | 41.8 | 19.1 | 40.1 | 26.4 | 47.8 | 23.4 | 43.2 |

**ViT-L/14:**
| Method | Dr10 | Dr50 | Sh10 | Sh50 | To10 | To50 | Av10 | Av50 |
|---|---|---|---|---|---|---|---|---|
| Text-only | 18.3 | 30.1 | 13.6 | 30.0 | 17.4 | 33.9 | 16.4 | 31.3 |
| Image-only | 10.7 | 19.9 | 4.5 | 12.2 | 8.4 | 16.5 | 7.8 | 16.2 |
| Pic2Word | 20.0 | 40.2 | 26.2 | 43.6 | 27.9 | 47.4 | 24.7 | 43.7 |
| SEARLE | 20.5 | 43.1 | 26.9 | 45.6 | 29.3 | 50.0 | 25.6 | 46.2 |
| Context-I2W | 23.1 | 45.3 | 29.7 | 48.6 | 30.6 | 52.9 | 27.8 | 48.9 |
| LinCIR | 20.9 | 42.4 | 29.1 | 46.8 | 28.8 | 50.2 | 26.3 | 46.5 |
| MagicLens | 25.5 | 46.1 | 32.7 | 53.8 | 34.0 | 57.7 | 30.7 | 52.5 |
| **Ours** | 31.6 | 48.5 | 24.4 | 44.3 | 33.2 | 53.9 | 29.7 | 48.9 |

### Table 3 — Ablation CFM × Neg-Steering (CIRR/CIRCO, ViT-L/14) (L593–602)
| Config | CFM | Neg | R@1 | R@5 | R@10 | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|---|---|---|---|---|---|---|---|---|---|
| Text-Only | ✗ | ✗ | 20.9 | 44.0 | 55.4 | 3.3 | 3.7 | 4.1 | 4.4 |
| (1) | ✗ | ✓ | 21.1 | 45.7 | 58.0 | 2.5 | 2.8 | 3.2 | 3.4 |
| (2) | ✓ | ✗ | 26.1 | 53.8 | 67.0 | 13.5 | 14.2 | 15.7 | 16.6 |
| FlowCIR | ✓ | ✓ | 26.2 | 56.1 | 68.6 | 14.9 | 15.7 | 17.3 | 18.2 |

### Table 4 — Hyperparameter λ, K, FM-vs-Regressor (CIRR/CIRCO Val+Test) (L627–645)
| Param | Val R@1 | Val R@10 | Test R@1 | Test R@10 | Val mAP@5 | Val mAP@25 | Test mAP@5 | Test mAP@25 |
|---|---|---|---|---|---|---|---|---|
| λ=0.1 | 25.4 | 69.3 | 25.9 | 67.2 | 10.2 | 13.1 | 10.6 | 12.5 |
| λ=0.3 | 26.2 | 69.7 | 26.4 | 68.1 | 13.9 | 15.7 | 14.2 | 16.8 |
| λ=0.5 | 26.6 | 69.6 | 26.2 | 68.6 | 14.8 | 16.9 | 14.9 | 17.3 |
| λ=0.7 | 26.1 | 69.5 | 26.2 | 68.5 | 14.8 | 17.0 | 15.2 | 17.6 |
| λ=0.9 | 26.0 | 69.5 | 25.8 | 68.4 | 14.5 | 16.8 | 14.8 | 17.2 |
| K=1/16·B | 26.1 | 69.2 | 25.8 | 65.8 | 13.9 | 15.7 | 14.2 | 16.7 |
| K=1/8·B | 26.6 | 69.6 | 26.2 | 68.6 | 14.8 | 16.9 | 14.9 | 17.3 |
| K=1/4·B | 25.8 | 58.8 | 25.5 | 67.6 | 13.5 | 15.2 | 13.7 | 16.0 |
| Regressor | 23.7 | 67.6 | 23.8 | 66.1 | 11.9 | 14.1 | 12.3 | 14.3 |
| FM | 26.6 | 69.6 | 26.2 | 68.6 | 14.8 | 16.9 | 14.9 | 17.3 |

### Table 5 — Transport target space: Image vs Text embedding (L683–688)
| Target Space | R@1 | R@5 | R@10 | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|---|---|---|---|---|---|---|---|
| Image Embedding | 23.8 | 50.4 | 62.2 | 12.7 | 13.0 | 14.3 | 15.0 |
| Text Embedding | 26.2 | 56.1 | 68.6 | 14.9 | 15.7 | 17.3 | 18.2 |

## 5. Source-free reconciliation (Python-verified)

**Cross-table byte-identities — ALL EXACT ✓:**
- Ours ViT-L/14 row identical across **Table 1 / Table 3 (FlowCIR row) / Table 4 (λ=0.5 Test = FM Test) / Table 5 (Text Emb)**: 26.2/56.1/68.6 + 14.9/15.7/17.3/18.2.
- Text-only ViT-L/14 row identical across **Table 1 / Table 3**: 20.9/44.0/55.4 + 3.3/3.7/4.1/4.4.

**Headline % claims (§4.2 prose "+X%" with arrows) — all recompute, but `%` is RELATIVE not absolute-pp:**
| Prose claim | Arrow | abs Δ (pp) | relative % | stated |
|---|---|---|---|---|
| B32 CIRR R@5 "+1.4%" | 55.7→56.5 | +0.8 | 1.44% | 1.4% ✓ |
| B32 CIRCO mAP@5 "+23.6%" | 10.6→13.1 | +2.5 | 23.58% | 23.6% ✓ |
| L14 CIRR R@1 "+2.3%" | 25.6→26.2 | +0.6 | 2.34% | 2.3% ✓ |
| L14 CIRCO mAP@5 "+14.6%" | 13.0→14.9 | +1.9 | 14.62% | 14.6% ✓ |
| B32 FIQ Avg R@10 "+2.2%" | 22.9→23.4 | +0.5 | 2.18% | 2.2% ✓ |
| B32 FIQ Avg R@50 "+2.9%" | 42.0→43.2 | +1.2 | 2.86% | 2.9% ✓ |
| **L14 FIQ Avg R@10 "+6.8%"** | **26.3→29.7** | **+3.4** | **vs 26.3 = 12.93%** | **6.8% ✗ (see defect 1)** |

**Ablation deltas (Table 3) — all recompute ✓:** TextOnly→(1)+Neg: CIRR R@1 +0.2 but CIRCO mAP@5 −0.8 (degrade ✓); (1)→(2)+CFM: R@1 +5.0, mAP@5 +11.0 (CFM main driver ✓); (2)→FlowCIR+Neg: R@1 +0.1, mAP@5 +1.4 (Neg marginal ✓).
**FM vs Regressor (Table 4):** Regr Test R@1 23.8 vs FM 26.2 (−2.4), Test mAP@5 12.3 vs FM 14.9 (−2.6) ✓.

## 6. Honest-scope flags (⚠ — NO numeric typos except defect 1 & 2; rest attribution/scope)

1. **⚠ DEFECT — arrow-vs-denominator mismatch (selective-baseline-arrow subclass, extends iter-72 MARVEL):** §4.2 L609 "+6.8% (26.3 → 29.7) gain in average R@10" (Fashion-IQ ViT-L/14). The "+6.8%" recomputes ONLY against **Context-I2W 27.8** (the true best non-MagicLens baseline: 1.9/27.8 = 6.83%), NOT against the arrow-cited **LinCIR 26.3** (vs which it is 12.93%). The arrow cites the WRONG (lower) baseline. Note the same paper's CIRR R@1 ViT-L/14 claim correctly cites Context-I2W 25.6 as baseline — so baseline selection is INCONSISTENT across the two benchmarks. Likely a copy error (meant 27.8, wrote 26.3). Headline impact: vs the true best baseline the R@10 gain is +1.9pp/+6.8%, not the +3.4pp the arrow implies.
2. **⚠ DEFECT (minor) — val/test slip:** §4.3 L697 "mAP@5 decreases from **14.8** to 12.3 on the **test set**". 14.8 is the **Val** mAP@5 (FM row); the **Test** FM mAP@5 is **14.9**. Regressor Test 12.3 ✓. The "14.8" should read 14.9 for a test-set statement.
3. **⚠ pp-vs-relative % framing (iter-79 DNG-Encoder class):** every "+X%" headline is a RELATIVE %, but each is printed with absolute-value arrows (e.g. "+1.4% (55.7→56.5)" is +0.8pp). A reader skimming the arrows may read "+1.4%" as +1.4pp. The relative framing inflates modestly when the baseline is low; the absolute deltas (≤0.8pp on CIRR R@1/R@5) are within plausible run noise (no seeds/CIs reported anywhere).
4. **⚠ "best Average R@10" is Dress+Toptee-carried; Ours is WORST on Shirt R@10 (both backbones):** ViT-B/32 Shirt R@10 — Ours 19.1 < PALAVRA 21.5 < SEARLE 24.8 (Ours worst non-trivial). ViT-L/14 Shirt R@10 — Ours 24.4 < Pic2Word 26.2 < SEARLE 26.9 < LinCIR 29.1 < Context-I2W 29.7 (Ours dead last). The Average-R@10 win is carried by Dress (Ours best both backbones: 24.8 / 31.6) + Toptee (Ours best/2nd: 26.4 / 33.2). Diagnostic: when a retrieval paper reports a per-category Average, check EACH category cell before echoing "best average".
5. **⚠ selective-baseline headline excludes stronger-resource methods (iter-72 MARVEL class):** the "improves over strongest competing baseline" headline excludes **MagicLens** (64–128 TPU, 36.7M private triplets) and **MCL** (external LLaMA2-7B), both of which BEAT FlowCIR. Gap is large: CIRCO mAP@5 — MagicLens 23.1 (B32) / 29.6 (L14) vs Ours 13.1 / 14.9 (Ours ≈ 50–57% of MagicLens); MCL L14 17.7 > Ours 14.9, and MCL ties Ours on CIRR R@1 (26.2). Authors disclose this ("substantially stronger external resources") — honest, but the headline "strongest or highly competitive" should be paired with the absolute gap.
6. **⚠ contribution (ii) Multi-Negative Steering is marginal on aggregate:** Table 3 shows CFM adds +5.0 R@1 / +11.0 mAP@5; Neg-Steering on top adds only +0.1 R@1 / +1.4 mAP@5 (and ALONE degrades CIRCO mAP@5 3.3→2.5). Its benefit is confined to negation-heavy queries, shown **qualitatively only** (Fig 4, no per-query-type table). The 2nd of three contributions is small; CFM is the real driver.
7. **⚠ training-cost comparison is hardware-INEQUIVABLE:** "0.5h on RTX 3090" vs baselines on A100/TPU is not compute-equivalent (different GPUs, different memory). The efficiency headline conflates "fewer GPU-hours" with "fewer GPU-hours on a weaker GPU". MagicLens 6h on 64–128 TPU ≫ FlowCIR 0.5h on 1 RTX3090 in absolute terms, but the per-device comparison is apples-to-oranges.
8. **⚠ no seeds / CIs / significance on any table:** all winning deltas are ≤0.8pp (CIRR R@1/R@5) or ≤1.9pp (CIRCO mAP@5); single-run numbers with no error bars. The "R@50 matches best prior" (48.9 = 48.9 tie) and several "best" cells are within plausible noise.
9. **⚠ Prop 1/2 are idealized-asymptotic (iter-65/72/79 class):** exact point-to-point transport (Eq 12) holds under the idealized conditional path (Eq 10) and exact velocity regression — neither holds for a finite-capacity MLP trained with stochastic FM + InfoNCE on noisy synthetic tuples. The theory motivates reference-conditioning but does not bound the finite trained model's transport error.
10. **⚠ Fashion-IQ "relative gains smaller… domain-specific fine-grained" (L610–613) is post-hoc rationalization that understates the Shirt regression:** the authors attribute smaller Fashion-IQ gains to "fine-grained nature", but the real story is a category-specific COLLAPSE on Shirt (defect 4), not a uniform smaller gain.
11. **⚠ Compo-Diff cost "128×A100, 231h" is an outlier anchor** that makes the efficiency bubble (Fig 2) look more favorable; the closest public-data inversion baseline (LinCIR) also trains in 0.5h (8×A100), same order as FlowCIR — the cost win vs LinCIR is modest, not the 2-orders-of-magnitude the figure implies.
12. **⚠ inference cost of Multi-Negative Steering not quantified:** it adds a rule-based parser + TinyLlama-1.1B forward pass + per-negative closed-form steering (Eq 7) at inference for negation queries; no latency table. "Inference-only" means no training, not zero overhead.

## 7. Citable falsifiable content & verdict

**The citable hinge is the paradigm shift textual-inversion → conditional-flow-matching-as-semantic-transport** (Eqs 3–5), with two clean supporting ablations: (a) Table 5 — transporting to TEXT embedding (26.2/14.9) beats IMAGE embedding (23.8/12.7); (b) Table 4 — FM (26.2/14.9) beats direct regression (23.8/12.3), justifying the time-conditioned velocity field over a residual regressor. Reference-image conditioning is theoretically necessary (Prop 1 mean-collapse) and the conditional path gives exact transport under idealization (Prop 2).

**Verdict:** genuinely novel paradigm (first conditional-FM-for-CIR), training-efficient (0.5h single-GPU), but the retrieval gains over the true best public-data baselines are MODEST (sub-1pp on CIRR R@1, ~1.9–2.5pp on CIRCO mAP@5, relative-% framed), it LOSES to stronger-resource methods (MagicLens/MCL) by a wide margin on CIRCO, is WORST on Fashion-IQ Shirt R@10, and the 2nd contribution (Multi-Negative Steering) is aggregate-marginal + qualitatively-only supported. Cross-table reconciliation is clean (all Ours rows byte-identical across T1/T3/T4/T5); two genuine prose defects (arrow-vs-denominator, val/test slip) neither load-bearing for the paradigm claim.
