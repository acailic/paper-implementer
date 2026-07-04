# When Token Compression Breaks: Structural Pruning vs. Token Reduction for Robust ViT Segmentation under High Compression — Source-First Breakdown

**arXiv:** 2607.02237 (cs.CV, Jul 2026) · **PDF:** 16pp (pdfinfo; `file` misreports 1pp → **15-page gap, defect recurs**) · **Authors:** Tien-Phat Nguyen, Ngai-Man Cheung (Singapore University of Technology and Design, SUTD) · **Code:** linked from abstract ("this https URL"). Repo entry rank 68 / 73rd paper; **first token-compression-vs-structural-pruning head-to-head / robustness-of-compression / prune-then-merge / ViT-segmentation-under-corruption paper** in the library.

> Sibling-in-spirit to **WBMM iter 83** (efficient vision operator) + **token-compression-vs-pruning ViT (this) is the repo's first COMPRESSION×ROBUSTNESS benchmark**; distinct from model-level efficiency (speculating-experts/sasp/spec-auf) and operator-level (WBMM) — here the contribution is a *diagnostic benchmark + a stacking recipe*, not a new operator.

---

## 1. The claim (in one paragraph)

ViT-based semantic segmentation is expensive. Two orthogonal efficiency levers exist: **token compression** (ToMe / ALGM / CTS — reduce the encoder *token sequence*) and **structural pruning** (NViT — reduce encoder *architectural width*). Prior work evaluates each in isolation and only at low-to-moderate compression, never head-to-head, never under corrupted inputs. This paper builds a **unified matched-FLOPs benchmark on ADE20K + Cityscapes + their corruption variants ADE20K-C / Cityscapes-C** (16 corruption types × 5 severities = 80 variants/image) and shows three findings:

- **F1** Mild token compression preserves near-baseline quality at ~1.4–1.9× compute cut.
- **F2** Aggressive token compression (3.4–3.8×) **collapses** (ToMe: 47.22/35.58 → 23.94/16.86 on ADE20K).
- **F3** Structural pruning degrades **smoothly** at the same compute budgets (NViT-S: 45.30/32.86 at 3.8×).
- **Recipe** A **prune-then-merge (PtM)** stack — moderate NViT prune + moderate ToMe on top — gets the best accuracy-robustness trade-off at high compression (ADE20K 45.71/34.33 at 3.7×; Cityscapes 77.52/60.71 at 3.6×).

**Mechanistic diagnosis (§4.3, Fig 4):** aggressive token merging causes **feature dimensional collapse** — normalized effective rank `eRank(X)/min(N,C)` drops sharply for ToMe, gradually for ALGM/CTS, stays flat for NViT (which preserves the full spatial token grid).

---

## 2. Protocol (§3, L266–340) — the controlled-comparison contract

- **Pipeline (fixed across all methods):** Segmenter-style ViT encoder (DeiT-B, ImageNet-pretrained) + unchanged lightweight transformer decoder. Compression applied to the **encoder only**.
- **Methods:** (i) token compression — ToMe (general), ALGM + CTS (segmentation-aware); (ii) structural pruning — NViT (latency target `p`); (iii) **PtM** — ToMe on top of an NViT-pruned backbone.
- **Compression regimes (per-dataset swept thresholds):**
  - ADE20K: ALGM T={0.94,0.75,0.65}, CTS R={0.3,0.6,0.7}, ToMe R={40,95,140}; NViT p∈{0.80,0.70,0.25}.
  - Cityscapes: ALGM T={0.94,0.88,0.70}, CTS R={0.4,0.5,0.65}, ToMe R={145,190,280}.
- **Training:** ADE20K 64ep bs8 lr1e-3 crop512; Cityscapes 216ep bs8 lr1e-2 crop768; SGD mom0.9 poly power0.9 drop-path0.1.
- **Metrics:** `mIoUclean` / `mIoUnoise` (aggregate over 16 corruptions × 5 severities). **Matched-FLOPs** comparison (token-ratio ≠ pruning-ratio, so FLOPs is the common axis). FPS on single RTX A6000, bs32, mixed precision (§4.4).

---

## 3. Tables — verbatim with sourcing

### Table 1 — Full benchmark under matched FLOPs (L438–456)

| Method | ADE GFLOPs↓ | ADE mIoUclean↑ | ADE mIoUnoise↑ | CS GFLOPs↓ | CS mIoUclean↑ | CS mIoUnoise↑ |
|---|---|---|---|---|---|---|
| **Baseline** | 129.70 ×1.0 | 47.22 | 35.58 | 347.77 ×1.0 | 78.25 | 63.26 |
| _Mild_ | | | | | | |
| ALGM | 92.31 ×1.4 | 46.88 | 34.64 | 190.27 ×1.8 | 77.98 | 61.68 |
| CTS | 88.57 ×1.5 | 46.03 | 34.76 | 185.52 ×1.9 | 78.15 | 63.07 |
| ToMe | 93.00 ×1.4 | 46.94 | 35.38 | 183.83 ×1.9 | 77.40 | 62.02 |
| NViT | 96.40 ×1.4 | **47.93** | **36.25** | 215.61 ×1.6 | 78.12 | 61.78 |
| _Moderate_ | | | | | | |
| ALGM | 51.21 ×2.5 | 44.52 | 31.56 | 157.19 ×2.2 | 76.86 | 60.60 |
| CTS | 49.93 ×2.6 | 44.05 | 33.66 | 150.03 ×2.3 | 77.54 | 62.40 |
| ToMe | 49.51 ×2.6 | 41.01 | 30.24 | 142.11 ×2.5 | 75.25 | 60.74 |
| NViT-H | 49.37 ×2.6 | 45.40 | 34.20 | 138.55 ×2.5 | 77.52 | 61.04 |
| NViT+ToMe (PtM) | 49.02 ×2.7 | 46.79 | 34.83 | 141.46 ×2.5 | 77.58 | 60.69 |
| _Aggressive_ | | | | | | |
| ALGM | 34.58 ×3.8 | 42.85 | 30.20 | 98.94 ×3.5 | 73.15 | 57.20 |
| CTS | 37.95 ×3.4 | 42.16 | 32.69 | 100.18 ×3.5 | 73.81 | 60.99 |
| ToMe | 35.08 ×3.7 | 23.94 | 16.86 | 98.60 ×3.5 | 64.50 | 50.56 |
| NViT-S | 33.82 ×3.8 | **45.30** | **32.86** | 95.05 ×3.7 | 76.76 | 58.75 |
| NViT-B+ToMe (PtM) | 34.84 ×3.7 | **45.71** | **34.33** | 95.87 ×3.6 | **77.52** | 60.71 |

_(GFLOPs subscripts = compute-reduction factor vs baseline; bold = best per dataset+level; NViT rows w/o B/H/S suffix = latency target p swept per dataset.)_

### Table 2 — Wall-clock comparison on ADE20K grouped by measured FPS speedup (L589–606)

| Method | rFLOPs↑ | FPS↑ | Speedup↑ | mIoUclean↑ | mIoUnoise↑ |
|---|---|---|---|---|---|
| **Baseline** | 1.00× | 23.69 | 1.00× | 47.22 | 35.58 |
| _Mild (~1.4×)_ | | | | | |
| ALGM | 1.41× | 33.45 | 1.41× | 46.88 | 34.64 |
| CTS | 1.46× | 33.69 | 1.42× | 46.03 | 34.76 |
| ToMe | 1.39× | 32.87 | 1.39× | 46.94 | 35.38 |
| NViT | 1.68× | 32.22 | 1.36× | 46.77 | 35.51 |
| _Moderate (~2.5×)_ | | | | | |
| ALGM | 2.53× | 60.01 | 2.53× | 44.52 | 31.56 |
| CTS | 2.60× | 58.44 | 2.47× | 44.05 | 33.66 |
| ToMe | 2.62× | 61.50 | 2.60× | 41.01 | 30.24 |
| NViT-S | 3.84× | 57.54 | 2.43× | 45.30 | 32.86 |
| _Aggressive (~3.5×)_ | | | | | |
| ALGM | 3.75× | 78.90 | 3.33× | 42.85 | 30.20 |
| CTS | 3.42× | 72.21 | 3.05× | 42.16 | 32.69 |
| ToMe | 3.70× | 84.96 | 3.59× | 23.94 | 16.86 |
| NViT | 7.61× | 87.34 | 3.69× | 41.01 | 26.85 |
| NViT-B+ToMe (PtM) | 4.75× | 88.10 | 3.72× | 45.17 | 33.92 |

_(Bold = best clean/corruption mIoU within each speed group.)_

**Figures (qualitative / figure-only):** Fig 1 (teaser, robustness-compute trade-off), Fig 2 (accuracy-compute trade-off curves ADE20K+Cityscapes, clean+noise), Fig 3 (qualitative seg under aggressive compression), Fig 4 (normalized effective rank vs GFLOPs — **the dimensional-collapse diagnostic, figure-only, no tabulated values**).

---

## 4. Source-free reconciliation (Python-verified) — every prose delta EXACT

All numbers recompute from displayed cells within rounding. **ZERO numeric prose-vs-table cell typos.**

- **Mild range "1.4×–1.9×"** (Finding 1, L432): ADE20K {ALGM 1.41, CTS 1.46, ToMe 1.39, NViT 1.35}; Cityscapes {ALGM 1.83, CTS 1.87, ToMe 1.89, NViT 1.61} → envelope 1.4–1.9× ✓.
- **CTS ADE20K "129.70→88.57 (1.5×)", "47.22/35.58→46.03/34.76"** (L468–469): 129.70/88.57=1.46→1.5× ✓; cells exact ✓.
- **CTS Cityscapes "1.9× (347.77→185.52)", "78.15/63.07 vs 78.25/63.26"** (L470–471): 347.77/185.52=1.87→1.9× ✓.
- **Aggressive range "3.4×–3.8×"** (Finding 2, L477): ADE {ALGM 3.75, CTS 3.42, ToMe 3.70, NViT-S 3.84}; CS {ALGM 3.51, CTS 3.47, ToMe 3.53, NViT-S 3.66} → envelope 3.4–3.8× ✓.
- **ToMe collapse ADE20K "47.22/35.58→23.94/16.86 at 35.08 GFLOPs (3.7×)"** (L482): 129.70/35.08=3.70× ✓; cells exact ✓.
- **ToMe Cityscapes "64.50/50.56"** (L485) ✓; **CTS "73.81/60.99 at 100.18 GFLOPs"** (L489) ✓.
- **NViT ADE20K "47.93/36.25, 45.40/34.20, 45.30/32.86 at 96.40(1.4×)/49.37(2.6×)/33.82(3.8×)"** (Finding 3, L504): 129.70/96.40=1.35, /49.37=2.63, /33.82=3.84 ✓; cells exact ✓.
- **NViT-S Cityscapes "76.76/58.75 at 95.05 (3.7×)"** (L506): 347.77/95.05=3.66→3.7× ✓.
- **PtM (§4.2)** "45.71/34.33 at 34.84 GFLOPs ADE20K (3.7×)", "77.52/60.71 at 95.87 Cityscapes (3.6×)" (L521–522): 129.70/34.84=3.72→3.7×; 347.77/95.87=3.63→3.6× ✓.
- **Table 2 FPS speedups** every row = FPS/23.69 to 2dp: PtM 88.10/23.69=3.72× ✓; NViT agg 87.34/23.69=3.69× ✓; all 13 rows match the Speedup column exactly.
- **NViT-S FLOPs-vs-FPS mismatch (§4.4 thesis):** rFLOPs 3.84× but FPS speedup only 2.43× (ratio 0.63) — the paper's central deployment caveat ✓.

### Cross-table byte-identities (T1 ↔ T2, ADE20K) — 6/6 token-compression rows EXACT; NViT/PtM rows differ (swept configs)

- ALGM mild 46.88/34.64; CTS mild 46.03/34.76; ToMe mild 46.94/35.38 — byte-identical T1↔T2 ✓.
- ALGM mod 44.52/31.56; CTS mod 44.05/33.66; ToMe mod 41.01/30.24 — byte-identical T1↔T2 ✓.
- ALGM agg 42.85/30.20; CTS agg 42.16/32.69; ToMe agg 23.94/16.86 — byte-identical T1↔T2 ✓.
- **NViT-S 45.30/32.86** is byte-identical between **T1-aggressive** (3.8× FLOPs) and **T2-moderate** (2.43× FPS) — *the same model, two regime labels*. This byte-identity is the **smoking gun for the FLOPs≠FPS thesis** (§4.4): NViT-S is "aggressive" by FLOPs but only "moderate" by wall-clock. ✓ (clean reconciliation.)
- **NViT rows are NOT cross-table byte-identical** (T1 mild NViT 47.93/36.25 vs T2 mild NViT 46.77/35.51 — different latency target `p` swept per table). Not a defect: the paper states NViT `p` is swept per regime to hit the matched budget.
- **PtM differs T1 vs T2:** T1 (FLOPs-matched) 34.84 GFLOPs (3.72×) → 45.71/34.33; T2 (FPS-matched) rFLOPs 4.75× → ~27.3 GFLOPs (MORE compressed) → 45.17/33.92. Delta 0.54 clean / 0.41 noise from the extra ToMe merging to hit the FPS target. (See honest-scope flag 4.)

---

## 5. Honest-scope flags (12; NO numeric cell typo — all attribution/scope)

1. **"Consistently better accuracy-robustness trade-off" (abstract + §4.2) slightly overstates Cityscapes-noise.** PtM is the best *clean* mIoU on Cityscapes aggressive (77.52) but **loses *noise* to CTS** (60.71 < 60.99). The headline is exact on ADE20K (PtM best clean+noise) and on Cityscapes-clean, but "consistently" over-reaches by one 0.28-noise cell. *(iter-72 MARVEL / iter-74 PointDiT / iter-82 InvSplat blanket-"all"-outperformance-with-one-tie class.)*
2. **"Mild pruning is free" is buried.** NViT *mild* on ADE20K **beats the baseline on both metrics** (47.93/36.25 vs 47.22/35.58, +0.71/+0.67) **at lower compute** (96.40 vs 129.70 GFLOPs). The "trade-off" framing (implying sacrifice) understates that mild structural pruning strictly dominates baseline on this dataset. Finding 3 cites 47.93/36.25 but does not flag that it exceeds baseline.
3. **"Pruning degrades gracefully" is regime-scoped — pruning ALSO breaks at extreme compression.** Table-2 NViT at **7.61× rFLOPs** drops to 41.01/**26.85** — its noise (26.85) is **below ToMe at only 2.6×** (30.24). The graceful-degradation headline (F3) is verified only to ~3.8×; pushed to 7.6× the central token-compression-vs-pruning dichotomy narrows sharply. The paper uses the 3.8× point to make the contrast and the 7.6× point only in the FPS table.
4. **PtM is reported under two different configs that are not labeled as such.** §4.2 cites 45.71/34.33 (T1, FLOPs-matched, 34.84 GFLOPs); §4.4 cites 45.17/33.92 (T2, FPS-matched, ~27.3 GFLOPs / 4.75× rFLOPs). Both are called "PtM at ~3.7×" but are *different ToMe merging rates*. The stronger 45.71/34.33 headline is the FLOPs-matched config; a reader mapping §4.2→§4.4 gets two different PtM numbers for "the same" 3.7× point.
5. **NViT "mild" / "aggressive" labels are NOT cross-table consistent.** T1 mild NViT (47.93/36.25) ≠ T2 mild NViT (46.77/35.51) — different latency targets `p`. "NViT" without a B/H/S suffix is a swept family, not one model; cross-table NViT rows are not byte-comparable (unlike token-compression rows). The paper discloses the sweep but a reader can still mismatch T1-NViT with T2-NViT.
6. **Coexistence: NViT-S appears in T1-aggressive but T2-moderate** (same 45.30/32.86). This is the paper's FLOPs≠FPS evidence (good), but it also means a casual reader could double-count or mis-rank NViT-S across the two tables. Diagnostic: when one model lands in different regime-buckets across tables, treat regime labels as table-local, not global.
7. **Effective-rank diagnostic (Fig 4, the §4.3 mechanism) is figure-only — no tabulated `eRank` values.** The "dimensional collapse causes the break" causal claim rests on bar/band readings, not a table; unverifiable from text. *(iter-7/16/74 figure-derived class.)*
8. **No seeds / CIs / significance on any table.** Decisive gaps (ToMe collapse −23.28 mIoUclean) are robust, but the PtM-vs-NViT-S edge (+0.41 clean / +1.47 noise) and NViT-mild-beats-baseline (+0.71/+0.67) are within plausible run noise for a single A6000 run with no reported std.
9. **FPS = single hardware point** (RTX A6000, bs32, mixed precision). The "deployment-oriented" framing (§3.2 L325) rests on one GPU; no edge/low-power/mobile hardware, no batch-size sweep — yet the abstract sells a "practical recipe for deployment-oriented ViT segmentation." *(iter-83 WBMM hardware-agnostic-only-NVIDIA class.)*
10. **`mIoUnoise` aggregates 16 corruptions × 5 severities into one number** — hides per-corruption-type behavior. A method could win the aggregate while losing on the corruption type that matters for a given deployment (e.g. blur vs noise vs weather). No per-corruption breakdown reported.
11. **Token-compression methods compared only as encoder-side** (decoder unchanged); NViT prunes encoder capacity with full token grid retained. The "spatial-grid-preserved" advantage of pruning is partly an artifact of *where* each lever is applied, not purely the lever itself — a decoder-aware token-compression baseline is not tested.
12. **"NViT" T2-aggressive clean 41.01 coincidentally equals ToMe-moderate clean 41.01** (noise differs: 26.85 vs 30.24). Not a defect — different models — but a transcription-error trap: a reader copy-pasting "41.01" across rows could conflate the broken-pruned-NViT with ToMe-moderate. Diagnostic: when two unrelated cells share a clean-mIoU value, check the noise column before equating rows.

---

## 6. Citable falsifiable content + subarea placement

**Falsifiable (echo these):**
- The **dichotomy** (F2 vs F3): at matched ~3.7× FLOPs on ADE20K, ToMe collapses (23.94/16.86) while NViT-S degrades gracefully (45.30/32.86) — a clean, large, reproducible gap.
- **PtM beats both levers alone** at high compression on ADE20K (45.71/34.33 > NViT-S 45.30/32.86 and > CTS 42.16/32.69 on both metrics) — the stacking recipe works *where the dichotomy predicts it should* (complementary failure modes).
- **FLOPs≠FPS for pruning** (NViT-S: 3.84× FLOPs cut → only 2.43× FPS) — a deployment-correctness caveat with a byte-identical cross-table witness (same model, two regime labels).

**Do NOT echo uncritically:**
- "Consistently better accuracy-robustness trade-off" (loses Cityscapes-noise to CTS; flag 1).
- "Pruning is more stable at high compression" without the regime bound (breaks at 7.6×; flag 3).
- The 45.71/34.33 PtM headline as the FPS-side number (that's 45.17/33.92; flag 4).
- The effective-rank mechanism as quantitative (figure-only; flag 7).

**Subarea:** compression-vs-pruning efficiency benchmark × robustness (corruption) for ViT segmentation. Sibling to **WBMM iter 83** (vision efficiency, operator-level) and **token-compression family**; distinct because the contribution is a *diagnostic benchmark + stacking recipe*, not a new module/operator. The repo's first **robustness-of-compression** and first **prune-then-merge** paper.

---

## 7. Source

`paper.pdf` (939KB, 16pp pdfinfo; `file` misreports 1pp → 15-page gap, **defect recurs** across iters 66/67/69/70/71/72/73/75/78/79/81/82/84/85; intermittent no-defect iters 68/74/76/77/80/83); `paper_layout.txt` (pdftotext -layout, 837 lines, 2 explicit tables + Figs 1–4). All tables transcribed verbatim with line-ranges; every prose delta Python-verified EXACT.
