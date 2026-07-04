# GLEN — Learning to Evolve Scenes: Reasoning about Human Activities with Scene Graphs

**arXiv:** 2607.02425v1 [cs.CV] (2 Jul 2026)
**Authors:** Francesca Pistilli, Simone Alberto Peirone, Giuseppe Averta — Politecnico di Torino
**Project/code:** https://francescapistilli.github.io/GLEN
**Source files:** `paper.pdf` (4.9MB, 17pp pdfinfo; file misreports `/Pages=5` → **12-page gap defect recurs**), `paper_layout.txt` (pdftotext -layout, 1077 lines).
**Repo rank:** 79th paper, rank 74 unique. **First** egocentric-scene-graph-evolution / activity-conditioned-graph-edit-forecasting / spatio-temporal-scene-graph-as-video-representation paper in the repo.

---

## 1. Thesis (source-first, L21–46, 82–103)

Egocentric video understanding currently entangles objects, interactions, and temporal dynamics into one latent embedding, making it hard to attribute scene changes to specific actions. The paper advocates **explicit, compositional, editable** representations: a video is lifted to a **sequence of spatio-temporal scene graphs**, and scene dynamics are modeled as **activity-conditioned graph edits** — not as shifts in a latent vector.

Three contributions:
1. **SG-Ego** — a large-scale (3.8M) spatio-temporal scene-graph annotation set over Ego4D, built **training-free** (MLLM-captioned + GroundingDINO-grounded + SAM2/DINOv2-consolidated).
2. **GLEN** (Graph-Language Edit Network) — a graph neural network trained for (i) graph–text alignment and (ii) activity-conditioned graph-edit prediction.
3. **A-GEF** (Activity-driven Graph-Edit Forecasting) — a novel task: given initial graph G_t and a textual activity act_t, forecast the final consolidated graph G_{t:t+T} as a sequence of structured edits.

Validated on EgoSchema (long-range reasoning diagnostic), EgoMCQ + EgoCVR (retrieval), A-GEF (scene evolution), EXPLORE-Bench (long-horizon compositional reasoning).

## 2. SG-Ego annotation pipeline (L167–216)

- 3.8M spatio-temporal scene graphs over Ego4D; **N_obj = 1480** object classes, **N_rel = 387** relation classes.
- Splits: **SG-Ego-Align** 3.8M graphs (graph–text alignment); **SG-Ego-Edit** ≈360k train / 7.2k val (G_t, act_t, G_{t:t+T}) tuples for A-GEF.
- SG-Ego-Align drawn from **7297 unique videos**; graphs have **6.58 nodes and 16.06 ± 12.24 edges on average** (App., L747–750).
- **Stage 1** frame-level relation extraction (τ=5 fps; Qwen3.5-9B prompted for `(subject_x, relation, object_y)` triplets; rule-based filtering).
- **Stage 2** frame-level grounding (GroundingDINO concatenates subj–pred–obj, returns node detections + bbox + label; spatial-relation heuristics drop "to the right / inside").
- **Stage 3** cross-frame consolidation (SAM2 mask propagation + DINOv2 feature fallback; Hungarian matching, IoU > 0.5) → single spatio-temporal graph G_{t:t+T}.

## 3. GLEN method (L217–358)

Three components (Fig 2):
- **Graph Encoder F_G : G → R^d** — stack of **L TripletGCN layers** (node+edge joint message passing, Eq 1) interleaved with **cross-attention** to text tokens (Eq 2, contribution weight α). Pool node + edge embeddings → single graph embedding h_g via MLP projection.
- **Text Encoder F_T** — initialized from **EgoVLP text encoder, frozen**; maps act_t to h_act in the same d-space.
- **Graph Edit Model E : (G, T) → G** — augments G_t with **K=128 learnable node queries** (candidate future-object slots, [17]-style), builds a fully-connected noised graph G̃_t, conditions node+edge features on h_act via cross-attention (Eq 4), then a second **Graph Edit Encoder F_Edit** (2 TripletGCN layers) emits node-class scores (N_obj+1, incl. "no object" = deletion) and edge deletion/label heads. Hungarian assignment matches query nodes to inserted nodes; CE for nodes, BCE for edges.

**Training objectives:**
- **GTCA** (Graph-Text Contrastive Alignment, Eq 3) — InfoNCE with positive set P_i (same verb+noun labels) and negatives N_i from the same video; symmetric g2t + t2g. Cross-attention disabled (encoders kept separate). n=3 hard negatives per anchor.
- **GTM** (Graph-Text Matching, [6,44]) — hard-pair mining with CA layers enabled; MLP head predicts pair positive/negative. **GTM†** restricts sampling to the same video.

**Edit operations modeled:** node deletion, node insertion (no explicit replacement — changes in attributes go via delete-and-insert).

**Training:** single A100, ~24h; backbones frozen, only Graph Encoders + a small text projection head trained.

## 4. Experiments — tables verbatim

### Table 1 (L394–403) — EgoSchema long-range reasoning (left) + EgoMCQ graph–text alignment (right)
| Method | EgoSchema (val) Acc | | Method | EgoMCQ Inter | EgoMCQ Intra |
|---|---|---|---|---|---|
| EgoThinker [48] | 71.8 | | EgoVLP [5] | 90.6 | 57.2 |
| Qwen3.5-9B (Blind) | 38.2 | | HierVL [49] | 90.5 | 52.4 |
| Qwen3.5-9B (Frames) | 72.8 | | EgoVLPv2 [6] | 91.0 | 60.9 |
| Qwen3.5-9B (SG-Ego) | 66.0 | | HiERO (EgoVLP) [50] | **91.6** | 59.6 |
| Qwen3.5-9B (Frames + SG-Ego) | 73.2 | | **GLEN (Perc. Enc.)** | 91.2 | 56.2 |

### Table 2 (L407–420) — EgoCVR composed-video-retrieval Recall@k
| Method | Global R@1 | R@5 | R@10 | Local R@1 | R@2 | R@3 |
|---|---|---|---|---|---|---|
| Random | 0.01 | 0.05 | 0.1 | 25.3 | 38.2 | 50.7 |
| CLIP | 7.4 | 33.2 | 55.3 | 26.1 | 43.4 | 57.7 |
| BLIPCoV_R [51]† | 5.4 | 15.2 | 24.3 | 33.1 | 49.5 | 62.9 |
| BLIPCoV_R−ECDE [45]† | 6.0 | 14.8 | 24.3 | 33.4 | 49.3 | 63.0 |
| CIReVL [52] | 2.0 | 6.8 | 10.2 | 21.6 | 35.1 | 46.0 |
| TFR-CVR [14] | 14.1 | 39.5 | 54.4 | 44.2 | 61.0 | 73.2 |
| Thawakar et al. [53]† | 14.6 | **41.3** | 54.9 | 44.8 | 61.7 | 74.0 |
| **GLEN** | **15.3** | 40.3 | **56.9** | **47.7** | **64.8** | **76.3** |

† finetuned on WebVid-CoVR / Dense-WebVid-CoV for CVR.

### Table 3 (L471–477) — A-GEF on SG-Ego, Triplet Recall
| Method | R@20 | R@50 | R@100 |
|---|---|---|---|
| Qwen3.5-9B [38] | 9.14 | 9.14 | 9.14 |
| G_t (static) | 23.17 | 23.17 | 23.17 |
| **GLEN** | **35.06** | **43.92** | **48.49** |

### Table 4 (L480–490) — EXPLORE-Bench long-horizon reasoning (subset; full = Table 7)
| Methods | Short S_obj | S_rel | Medium S_obj | S_rel | Long S_obj | S_rel | Full S_obj | S_rel |
|---|---|---|---|---|---|---|---|---|
| GPT-5.2-Chat | 59.91 | 2.70 | 59.88 | 2.65 | 58.06 | 2.61 | 59.69 | 2.67 |
| Gemini-3-Pro | 61.29 | 2.77 | 60.99 | 2.74 | 59.17 | 2.70 | 60.94 | 2.75 |
| LLaVA-OneVision-1.5-8B | 53.25 | 2.51 | 51.21 | 2.44 | 47.62 | 2.41 | 51.87 | 2.47 |
| Qwen3-VL-8B-Instruct | 61.34 | **2.84** | 60.78 | **2.81** | 56.83 | 2.71 | 60.63 | **2.82** |
| Qwen3-VL-8B-Thinking | 63.77 | 2.85 | 62.61 | 2.78 | 58.02 | 2.63 | 62.70 | 2.80 |
| **GLEN** | **66.12** | 2.67 | **66.71** | 2.73 | **59.37** | 2.67 | **65.59** | 2.69 |

### Table 5 (L980–995) — EgoMCQ ablations
**Left — training objective** (n = #negatives per anchor):
| Configuration | EgoMCQ Inter | Intra |
|---|---|---|
| GTCA (n=1) | 89.3 | 53.2 |
| GTCA (n=3) | 90.6 | 53.9 |
| GTCA (n=1) + GTM | 89.6 | 53.3 |
| GTCA (n=3) + GTM | 91.0 | 54.4 |
| GTCA (n=1) + GTM† | 89.4 | 53.7 |
| GTCA (n=3) + GTM† | 91.0 | 54.9 |

**Right — inference mode:**
| Mode | EgoMCQ Inter | Intra |
|---|---|---|
| GTCA head | 91.0 | 54.9 |
| GTM† head | 80.8 | 54.0 |
| GTCA + GTM† heads | 91.2 | 56.2 |

### Table 6 (L1020–1028) — A-GEF ablation, number of node queries
| Num queries | R@20 | R@50 | R@100 |
|---|---|---|---|
| G_t (static) | 23.17 | 23.17 | 23.17 |
| 64 | 34.98 | 42.19 | 44.73 |
| 72 | 35.12 | 42.84 | 45.89 |
| 100 | 35.05 | 43.71 | 47.84 |
| 128 | 35.06 | 43.92 | 48.49 |

### Table 7 (L1034–1072) — EXPLORE-Bench full (Short/Medium/Long/Full × S_obj/S_att/S_rel/S_uni)
Subsets: Short 11–99, Medium 100–199, Long 200–694 atomic actions. 28 baselines across 4 tiers (proprietary API; open non-thinking; open thinking; embodied/egocentric) + GLEN. **GLEN row reports only S_obj and S_rel** (S_att, S_uni = "—"). Full S_obj GLEN 65.59 (best); Full S_rel GLEN 2.69 (7th, behind Qwen3-VL-8B-Instruct 2.82, Qwen3-VL-8B-Thinking 2.80, Gemini-3-Flash/Pro 2.75, Step3-VL-10B 2.74, GLM-4.6V-Flash 2.73).

## 5. Equations
- **Eq 1** (L258) TripletGCN node update: x_j^{(l+1)} = x_j + φ_2(mean_{i∈Neigh(j)} φ_1(x_i ‖ y_{i→j} ‖ x_j)).
- **Eq 2** (L267) text cross-attention residual: X̃^{(l+1)} = LN(X^{(l+1)} + α·CA(X^{(l+1)}, H_act)).
- **Eq 3** (L286) GTCA graph-to-text loss: L_g2t = (1/|B|) Σ_{i∈B} [Σ_{k∈P_i} exp(h_g,i·h_act,k/τ) / Σ_{k∈N_i} exp(h_g,i·h_act,k/τ)].
- **Eq 4** (L333) action-conditioned node+edge: Z̃^{n,cond}_t = Z̃^n_t + CA(Z̃^n_t, h_act); Z̃^{e,cond}_t = Z̃^e_t + CA(Z̃^e_t, h_act).

## 6. Python source-free reconciliation (`/tmp/reconcile_glen.py`) — VERDICTS

**ZERO numeric prose-vs-table CELL typos.** Every verifiable number recomputes exact; all four cross-table byte-identities hold:
- **T4 ↔ T7**: GLEN + 5 shared rows (GPT-5.2-Chat, Gemini-3-Pro, LLaVA-OV-8B, Qwen3-VL-8B-Instruct, Qwen3-VL-8B-Thinking) byte-identical on S_obj/S_rel × 4 subsets.
- **T3 ↔ T6**: GLEN (35.06/43.92/48.49) == T6[128-queries]; G_t(static) (23.17×3) == T6 G_t(static).
- **T1 ↔ T5**: GLEN(Perc.Enc.) 91.2/56.2 == T5(right) "GTCA + GTM† heads".
- Prose "66.0% v. 72.8%" == Table 1 (SG-Ego 66.0 vs Frames 72.8); Frames+SG-Ego 73.2 = Frames +0.4pp.
- A-GEF: GLEN > static > Qwen at every K (gaps +11.89 / +20.75 / +25.32 over static); Qwen + static both flat across K (degenerate / no-evolution).
- T6 R@100 monotone 44.73→48.49 with #queries; R@20 NOT monotone (100-q 35.05 < 72-q 35.12).
- T5 n=3 > n=1; GTCA+GTM†(n=3) 91.0/54.9 ≥ GTCA(n=3) 90.6/53.9.

## 7. Honest-scope flags (12; load-bearing first — all attribution/framing, NO numeric typo)

1. **【load-bearing】EXPLORE-Bench SPLIT-METRIC overclaim — S_obj win, S_rel loss to 7 models** (extends the iter-89 WorldDirector split-encoder family; here the body is half-honest but the abstract/intro overstate). Full S_obj GLEN **65.59 = SOTA** (beats Qwen3-VL-8B-Thinking 62.70). Full S_rel GLEN **2.69 = 7th**, beaten by Qwen3-VL-8B-Instruct **2.82**, Qwen3-VL-8B-Thinking 2.80, Gemini-3-Flash/Pro 2.75, Step3-VL-10B 2.74, GLM-4.6V-Flash 2.73. Abstract L96 & intro L96 claim "state-of-the-art results on reasoning tasks, including EXPLORE-Bench" (blanket); body L518 qualifies "SOTA on S_obj and **competitive** on S_rel". The relation-edit half of GLEN is the weak half — graph edits predict the object set well but lose to MLLMs on LLM-judged relations.

2. **GLEN unranked on 2/4 EXPLORE-Bench metrics** — Table 7 GLEN reports only S_obj and S_rel; S_att (attribute) and S_uni (unique) are "—" for GLEN while every MLLM baseline reports all four. "SOTA on EXPLORE-Bench" rests on 1–2 of 4 metrics; on the other half GLEN cannot be ranked at all.

3. **EgoCVR "state-of-the-art … outperforming even methods specifically trained for CVR" is FALSE on Global R@5** — GLEN 40.3 < Thawakar† 41.3 (Δ −1.0pp). Thawakar† is finetuned on WebVid-CoVR (CVR-trained), so the "outperform CVR-trained methods" claim is contradicted on the one cell. GLEN wins 5/6 EgoCVR columns (all Local + Global R@1/R@10).

4. **EgoMCQ "comparable to video-language models" is honest framing but the gap is large on Intra** — GLEN rank 2/5 Inter (91.2 < HiERO 91.6), rank **4/5 Intra** (56.2 < EgoVLPv2 60.9, HiERO 59.6, EgoVLP 57.2). The structured graph representation loses 4.7pp Intra to EgoVLPv2.

5. **A-GEF is the authors' own benchmark on the authors' own auto-generated annotations** — both baselines are weak/strawman: Qwen3.5-9B is **degenerate** (flat 9.14 across K — "predicts few triplets"), static is no-evolution (flat 23.17). GLEN wins (+25.32 R@100 over static) but there is **no real graph-forecasting competitor**. SG-Ego is built training-free (Qwen3.5-9B + GroundingDINO + SAM2); GLEN is then evaluated on SG-Ego-derived A-GEF ⇒ **annotation-circularity** (test labels share the annotation pipeline's biases and errors).

6. **EgoSchema "scene graphs encode sufficient details" understates a 6.8pp drop** — Qwen3.5-9B on SG-Ego alone = 66.0 vs on Frames = 72.8 (−6.8pp absolute, −9.3% relative). "Frames + SG-Ego" 73.2 is only **+0.4pp** over Frames alone — adding the scene graph to the frames is near-noise on EgoSchema, undercutting the claim that the structured representation carries information beyond the pixels.

7. **EXPLORE-Bench S_rel is an LLM-judged metric (0–5 scale, standardized prompt)** — the metric itself is a proprietary-LLM artifact; GLEN's S_rel loss could partly reflect prompt/format mismatch rather than relation quality. Same class as iter-89 WorldDirector's closed-model-inference dependency.

8. **EXPLORE-Bench zero-shot transfer claim cuts both ways** — GLEN is NOT trained on EXPLORE-Bench (only on A-GEF), so the S_obj win is a genuine generalization result; but the same transfer produces only "competitive" S_rel, so "viable alternative to MLLMs for compositional reasoning" is S_obj-scoped.

9. **No CIs / seeds / multiple runs anywhere** — every table is single-run; decisive deltas are small (EXPLORE S_rel gaps 0.01–0.13pp; EgoMCQ Inter 0.4pp to HiERO; EgoCVR Global R@5 1.0pp).

10. **SG-Ego stats asymmetric** — 6.58 nodes but 16.06 ± 12.24 edges (std > mean for edges) ⇒ edge count is highly bimodal/skewed across graphs; aggregate A-GEF recall may be dominated by dense-graph samples.

11. **T6 R@20 non-monotone vs R@100 monotone** — "increasing the number of queries allows the model to specialize" (L1011) holds only at high K (R@100); at R@20 the 100-query config (35.05) dips below the 72-query config (35.12), so the ablation's monotone-improvement story is K-dependent.

12. **EXPLORE-Bench GLEN S_rel (2.69) ≈ its own EgoSchema-relative tier** while S_obj (65.59) is best-in-class — the paper's central "graph edits capture scene evolution" thesis is object-set-carried; the relation-edit mechanism (edge deletion/insertion heads) is the under-performing half and is not separately ablated against a relations-only baseline.

## 8. Subarea lineage (repo)

Repo's FIRST egocentric-scene-graph / activity-conditioned-graph-edit / scene-evolution paper. Sibling-in-spirit to **translation-as-bridging** (text transfer across modalities) and **FlowCIR** (iter 84, retrieval-as-transport in VLM space) — all "reframe a video task as structured editing/retrieval rather than latent embedding". Distinct from **invsplat** (scene reconstruction) and **hola** (hippocampal linear attention) which stay in latent space. Citable falsifiable hinge = Eq 4 action-conditioned graph edits + the clean T6 #queries ablation (R@100 monotone 44.73→48.49) + T5 GTCA hard-negative ablation (n=1→n=3 lifts Inter 89.3→90.6).
