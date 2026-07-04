# Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training

**arXiv:** 2607.01232v2 (2026-07-02) | **cs.LG** | **23pp** (pdfinfo)
**Authors:** Zijian Zhang, Rizhen Hu, Athanasios Glentis, Dawei Li, Chung-Yiu Yau, Hongzhou Lin (Amazon), Mingyi Hong (U Minnesota)

## Source

- paper.pdf (758.8KB, 23pp pdfinfo) — pdftotext -layout extraction, 1567 lines
- file-vs-pdfinfo: no page-count defect (23=23)

## Method

**Core question:** How are RL post-training gains distributed across transformer layers?

**Layer contribution metric** (Eq 4):
```
C(k) = (S_k - S_base) / (S_full - S_base)
```
C(k)=1.0 means single-layer k matches full-parameter RL gain; C>1 means surpasses.

**Single-layer training** (Eq 3): Standard GRPO backprop through full network, but only update parameters of isolated layer θ_k (all others frozen, including embedding + head).

**7 models** across 2 families, 3 RL algorithms, 2 task domains:

Table 1 (L264-284): Model configurations — verbatim
| Model | Family | Params | Layers | RL Algorithm | Task / Dataset | Layer Scan |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | Qwen3 | 1.7B | 28 | GRPO | Math / NuminaMath-CoT | Full |
| Qwen3-4B-Base | Qwen3 | 4B | 36 | GRPO | Math / NuminaMath-CoT | Full |
| Qwen3-8B-Base | Qwen3 | 8B | 36 | GRPO | Math / NuminaMath-CoT | Full |
| Qwen2.5-Math-1.5B | Qwen2.5 | 1.5B | 28 | Dr. GRPO | Math / MATH | Full |
| Qwen2.5-1.5B-Instruct | Qwen2.5 | 1.5B | 28 | GiGPO | Agentic / ALFWorld | Partial |
| Qwen2.5-3B-Instruct | Qwen2.5 | 3B | 36 | GiGPO | Agentic / ALFWorld | Partial |
| DeepSeek-Distilled-Qwen-7B | Qwen2.5 | 7B | 28 | GRPO | Math / Skywork | Partial |

## Key Equations

- **Eq 1** (L195): GRPO group-normalized advantage: Â_i = (r(x,y_i) − mean) / std over G samples
- **Eq 2** (L202): GRPO clipped surrogate: L = E[min(ρÂ, clip(ρ,1−ε,1+ε)Â)] − β·KL[π_θ ∥ π_ref]
- **Eq 3** (L222): Single-layer update: θ_k ← θ_k − α·∇θ_k L(θ), all others frozen
- **Eq 4** (L232-234): Layer contribution: C(k) = (S_k − S_base) / (S_full − S_base)

## Main Results

### Qwen3 Per-Layer Results

Table 2 (L338-365): Qwen3 per-layer training — verbatim selected rows
| Model | Setting | MATH500 | GSM8K | Olymp. | AMC | Avg | C_math | Code | Reas. | Lang. | Overall | C_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | Base | 57.4 | 74.4 | 18.7 | 26.1 | 44.1 | 0.00 | 34.9 | 20.7 | 41.7 | 35.4 | 0.00 |
| | Full | 64.0 | 82.0 | 26.9 | 30.2 | 50.8 | 1.00 | 33.5 | 22.6 | 48.2 | 38.8 | 1.00 |
| | Layer 10 | 68.6 | 80.5 | 27.3 | 30.8 | 51.8 | 1.14 | 34.6 | 21.9 | 47.2 | 38.9 | 1.03 |
| | Layer 12 | 65.6 | 81.3 | 27.3 | 32.4 | 51.6 | 1.12 | 36.2 | 21.5 | 47.4 | 39.2 | 1.12 |
| | Layer 1 | 64.4 | 79.4 | 25.9 | 30.2 | 50.0 | 0.87 | 40.0 | 22.7 | 47.1 | 39.9 | 1.32 |
| | Layer 7 | 64.0 | 80.1 | 24.9 | 29.0 | 49.5 | 0.80 | 38.1 | 22.4 | 46.5 | 39.1 | 1.09 |
| | Layer 24 | 60.6 | 74.8 | 21.2 | 27.6 | 46.1 | 0.28 | 30.6 | 21.6 | 44.2 | 35.6 | 0.06 |
| Qwen3-4B | Base | 65.2 | 75.4 | 27.6 | 40.5 | 52.2 | 0.00 | 41.5 | 28.8 | 57.6 | 45.0 | 0.00 |
| | Full | 77.2 | 91.9 | 38.4 | 47.1 | 63.7 | 1.00 | 48.8 | 32.4 | 62.9 | 51.9 | 1.00 |
| | Layer 16 | 79.4 | 92.0 | 40.3 | 45.5 | 64.3 | 1.06 | 51.9 | 33.0 | 64.4 | 53.4 | 1.22 |
| | Layer 14 | 78.4 | 90.3 | 39.9 | 46.5 | 63.8 | 1.02 | 52.9 | 31.6 | 63.3 | 52.9 | 1.14 |
| | Layer 24 | 77.0 | 89.5 | 38.5 | 43.6 | 62.2 | 0.87 | 47.4 | 31.9 | 61.7 | 50.8 | 0.84 |
| | Layer 2 | 73.8 | 87.7 | 35.3 | 42.4 | 59.8 | 0.66 | 49.1 | 31.9 | 62.3 | 50.8 | 0.84 |
| Qwen3-8B | Base | 71.8 | 82.0 | 36.6 | 41.7 | 58.0 | 0.00 | 50.4 | 32.2 | 57.5 | 49.5 | 0.00 |
| | Full | 80.0 | 92.3 | 42.8 | 50.8 | 66.5 | 1.00 | 53.7 | 35.5 | 63.7 | 54.9 | 1.00 |
| | Layer 16 | 80.4 | 91.8 | 44.1 | 52.0 | 67.1 | 1.07 | 54.5 | 35.5 | 68.8 | 56.5 | 1.30 |
| | Layer 15 | 79.8 | 92.8 | 40.6 | 52.7 | 66.5 | 1.00 | 56.8 | 34.0 | 68.9 | 56.5 | 1.30 |
| | Layer 8 | 77.0 | 89.0 | 41.5 | 50.9 | 64.6 | 0.78 | 54.3 | 34.4 | 62.0 | 53.8 | 0.80 |
| | Layer 2 | 72.8 | 84.6 | 38.5 | 43.0 | 59.7 | 0.20 | 51.4 | 33.7 | 59.4 | 51.1 | 0.30 |
| | Layer 0 | 61.8 | 79.6 | 31.0 | 42.4 | 53.7 | −0.51 | 44.0 | 34.0 | 63.0 | 48.7 | −0.15 |

### Cross-Family Results

Table 3 (L524-536): Qwen2.5-Math-1.5B (Dr. GRPO) — verbatim
| Setting | AIME | AIME25 | AMC | MATH500 | Minerva | Olymp. | Avg | C |
|---|---|---|---|---|---|---|---|---|
| Base | 20.0 | 6.7 | 32.5 | 33.0 | 12.5 | 22.8 | 21.2 | 0.00 |
| Full | 16.7 | 10.0 | 51.8 | 74.4 | 25.0 | 38.8 | 36.1 | 1.00 |
| Dr. GRPO† | 20.0 | 6.7 | 53.0 | 74.2 | 25.7 | 37.6 | 36.2 | – |
| Layer 14 | 20.0 | 10.0 | 52.3 | 74.8 | 25.6 | 35.3 | 36.3 | 1.01 |
| Layer 16 | 20.0 | 10.0 | 51.8 | 75.2 | 24.9 | 34.9 | 36.1 | 1.00 |
| Layer 12 | 20.0 | 10.0 | 45.8 | 73.8 | 25.0 | 34.8 | 34.9 | 0.92 |
| Layer 8 | 13.3 | 3.3 | 43.4 | 69.4 | 20.6 | 30.7 | 30.1 | 0.60 |
| Layer 23 | 10.0 | 3.3 | 38.6 | 64.0 | 19.9 | 29.3 | 27.5 | 0.42 |

Table 4 (L595-610): Qwen2.5-1.5B-Instruct (GiGPO, ALFWorld) — verbatim
| Setting | P&P | P2&P | LiL | H&P | C&P | Cl&P | Overall | C |
|---|---|---|---|---|---|---|---|---|
| Base | 5.9 | 0.0 | 5.5 | 9.7 | 4.2 | 3.3 | 4.1 | 0.00 |
| Full | 100 | 81.0 | 91.7 | 83.3 | 81.8 | 88.9 | 87.8 | 1.00 |
| GiGPO† | 94.4 | 76.4 | 67.5 | 94.4 | 79.8 | 94.8 | 86.7 | – |
| Layer 14 | 100 | 85.7 | 100 | 83.3 | 81.8 | 77.8 | 89.1 | 1.02 |
| Layer 16 | 91.9 | 52.4 | 91.7 | 94.4 | 72.7 | 83.3 | 81.2 | 0.92 |
| Layer 0 | 48.6 | 23.8 | 41.7 | 16.7 | 13.6 | 22.2 | 29.7 | 0.31 |
| Layer 24 | 32.4 | 19.0 | 41.7 | 11.1 | 18.2 | 27.8 | 25.0 | 0.25 |

Table 5 (L637-654): Qwen2.5-3B-Instruct (GiGPO, ALFWorld) — verbatim
| Setting | P&P | P2&P | LiL | H&P | C&P | Cl&P | Overall | C |
|---|---|---|---|---|---|---|---|---|
| Base | 57.6 | 9.1 | 37.5 | 0.0 | 12.5 | 8.0 | 24.2 | 0.00 |
| Full | 100 | 81.0 | 75.0 | 83.3 | 86.4 | 100 | 90.2 | 1.00 |
| Layer 18 | 94.6 | 76.2 | 100 | 83.3 | 86.4 | 100 | 90.8 | 1.01 |
| Layer 20 | 97.3 | 52.4 | 100 | 33.3 | 77.3 | 72.2 | 74.2 | 0.76 |
| Layer 0 | 78.4 | 28.6 | 41.7 | 38.9 | 18.2 | 27.8 | 43.8 | 0.30 |
| Layer 4 | 67.6 | 28.6 | 41.7 | 22.2 | 18.2 | 5.6 | 35.2 | 0.17 |

Table 6 (L703-717): DeepSeek-Distilled-Qwen-7B (GRPO, Skywork) — verbatim
| Setting | AIME | AIME25 | AMC | MATH500 | Minerva | Olymp. | Avg | C |
|---|---|---|---|---|---|---|---|---|
| Base | 47.2 | 35.3 | 69.9 | 88.2 | 34.6 | 49.0 | 54.1 | 0.00 |
| Full | 55.0 | 45.0 | 83.1 | 94.0 | 41.2 | 68.7 | 64.5 | 1.00 |
| Layer 16 | 57.5 | 45.0 | 86.7 | 96.6 | 38.6 | 65.6 | 65.0 | 1.05 |
| Layer 14 | 55.0 | 38.3 | 86.7 | 95.6 | 43.4 | 67.7 | 64.5 | 1.00 |
| Layer 0 | 50.4 | 35.4 | 83.1 | 93.2 | 38.2 | 59.3 | 59.9 | 0.56 |
| Layer 24 | 52.5 | 35.4 | 73.5 | 90.8 | 37.6 | 55.1 | 57.5 | 0.33 |

### Cross-Model Summary

Table 7 (L719-730): Layer contribution summary — verbatim
| Model | Family | Algorithm | Task | Best C | Worst C | Layers ≥ 1.0 | Middle conc. |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | Qwen3 | GRPO | Math | 1.14 | 0.28 | 5/28 | ✓ |
| Qwen3-4B-Base | Qwen3 | GRPO | Math | 1.06 | 0.66 | 4/36 | ✓ |
| Qwen3-8B-Base | Qwen3 | GRPO | Math | 1.07 | −0.51 | 4/36 | ✓ |
| Qwen2.5-Math-1.5B | Qwen2.5 | Dr. GRPO | Math | 1.01 | 0.42 | 2/28 | ✓ |
| Qwen2.5-1.5B-Inst | Qwen2.5 | GiGPO | Agentic | 1.02 | 0.25 | 1/8† | ✓ |
| Qwen2.5-3B-Inst | Qwen2.5 | GiGPO | Agentic | 1.01 | 0.17 | 1/11† | ✓ |
| DeepSeek-Dist-Qwen-7B | Qwen2.5 | GRPO | Math | 1.05 | 0.33 | 2/8† | ✓ |

## Guided Training Strategies (§4)

**Layer-adaptive LR** (§4.1): Boost Bk (best-k layers at 1e-5 vs default 5e-6) improves over full:
- 1.7B: Boost B10 53.70±0.40 vs Full 50.82±0.40 (+2.88, 43% of RL gain)
- 4B: Boost B10 64.42±0.13 vs Full 62.97±0.78 (+1.45)
- 8B: Boost B10 67.42±0.40 vs Full 66.43±0.40 (+0.99)

**Layer-selective training** (§4.2): Freeze all but best-k layers:
- 1.7B: Only B5 51.53±0.24 > Full 50.82±0.40
- 4B: Only B5 65.87±0.70 > Full 62.97±0.78 (+2.90, 27% of RL gain)
- 8B: Only B10 69.11±0.10 > Full 66.43±0.40 (+2.68, 32% of RL gain)

**Heuristic middle layers** (§4.3): No profiling needed, just train middle-k:
- 1.7B: Mid-5 51.35±0.28 (+0.53 over Full, 8% of gain)
- 4B: Mid-5 64.80±0.30 (+1.83 over Full, 17% of gain)
- 8B: Mid-5 68.19±0.62 (+1.76 over Full, 21% of gain)

**Majority voting** (§5.1, Fig 8): Top-7 layer models on OlympiadBench:
- Layer×7 Vote: 33.6±0.91 (best single 28.3±0.25; Full RL 26.9±0.40; Full RL×7 Vote 31.3±1.11)

## Cross-Dataset Consistency (§3.3)

- NuminaMath-CoT vs DeepScaleR (both math): Spearman ρ = 0.76 (p < 0.001)
- NuminaMath-CoT vs DeepCoder (math vs code): Spearman ρ = 0.59 (p < 0.001)
- Math C vs Overall C: Pearson r > 0.6 on all three Qwen3 scales (r=0.91, 0.67, 0.65 per Fig 2)

## Honest-Scope Flags

1. **Single-run evaluation for main layer scans**: All per-layer results in Tables 2-6 report single evaluation runs (no std). Only §4 guided strategies report mean±std over 3 runs. The C values that determine the headline "single layer matches full" are single-shot.

2. **Learning rate not independently tuned per layer**: All single-layer runs use the full-parameter-tuned LR (5e-6). Appendix A.7 shows C rankings survive 3× LR boost, but an independently tuned LR per layer could change absolute C values.

3. **"Recover 114% of RL gain" on smallest model only**: C=1.14 is Qwen3-1.7B only. On 4B and 8B, best C is 1.06-1.07. The abstract's "in some cases even surpass" is hedged correctly but the 1.14 headline is the most extreme point.

4. **Task-domain limitation of guided strategies**: §4 strategies validated only on math (NuminaMath-CoT). Authors acknowledge this in §7 (limitations). Agentic tasks show larger RL gains (66-84pp) but guided training not tested there.

5. **Partial layer scans for 4/7 models**: Qwen2.5-1.5B/3B-Instruct and DeepSeek-Distilled-Qwen-7B use representative subsets, not full scans. Best C might be higher with full scans (Table 7 marks with †).

6. **Full RL baselines are themselves single-run in Tables 2-6**: The Full RL values in Tables 2-6 differ from the 3-run averages in §4 (e.g., Qwen3-8B Full 66.5 in Table 2 vs 66.43±0.40 in §4). The per-layer C values computed against the single-run Full may shift under multi-run averaging.

7. **Appendix T16 Layer 0 C=0.75 vs Table 3 text "low-contribution layers near input end"**: In Qwen2.5-Math-1.5B, Layer 0 has C=0.75 (Appendix T16), which is moderate-high, not low. The paper's selected rows in Table 3 don't show this — it shows Layer 8 (C=0.60) and Layer 23 (C=0.42) as the contrast points. The "near input end" pattern holds for Qwen3 (Layer 0 C=-0.51 on 8B, C=0.89 on 1.7B) but NOT uniformly for Qwen2.5-Math.

8. **Jaccard overlap 34.1% claimed as "largely non-overlapping"**: 34.1% pairwise Jaccard among top-7 layers on Qwen3-1.7B-OlympiadBench means ~66% of problems are NOT shared. However, individual pairs range from 31.9% (Layer 10 vs 13) upward — the average masks whether some pairs share much more.

9. **Majority voting compared against self-consistency with same sample count (7)**: The 7× Full RL vote (31.3±1.11) vs 7× Layer vote (33.6±0.91) comparison is fair in sample count but not in compute budget — 7× Layer vote requires 7 independent full training runs.

10. **Call (overall contribution) can exceed Cmath substantially**: Qwen3-1.7B Layer 1 has Cmath=0.87 but Call=1.32 (Table 2). This means single-layer training on Layer 1 improves Code/Reasoning/Language MORE than math, relative to the full RL baseline. The paper notes this but does not investigate why.

11. **Qwen2.5-1.5B-Instruct Layer 14 achieves 100% on 3/6 ALFWorld categories (Pick&Place, LookInLight, Heat&Place)**: This surpasses Full RL on those categories but only matches on Pick2&Place and loses on Clean&Place (77.8 vs 88.9). The per-category breakdown reveals the "C=1.02" average hides category-level heterogeneity.

12. **No significance tests on C values**: No CIs, bootstrap, or statistical tests on whether C>1.0 is significant vs C=1.0, especially given single-run evaluation. The 3-run stds in §4 (e.g., ±0.40 on 1.7B Full at 50.8) suggest that a C of 1.14±? could plausibly overlap 1.0.

13. **ALFWorld Overall ≠ unweighted mean of categories**: The Overall column in Tables 4-5 is the overall success rate across all task instances (weighted by per-category task count), not the unweighted mean of the 6 category scores. Mean of Qwen2.5-1.5B Base categories = 4.77 but reported Overall = 4.1; 3B Base mean = 20.78 but reported = 24.2. The paper says "overall average" but computes it as total_success/total_tasks. This is consistent internally (C values recompute correctly from reported Overall) but the wording "average" is ambiguous.

## Numeric Verification (2026-07-04)

Python source-free reconciliation (`/tmp/reconcile_one_layer.py`): **ZERO genuine numeric cell typos**.

- All 20 Math Avg cells across Tables 2, 3, 6 = mean of 4 or 6 benchmark cells EXACT (within 1dp rounding; 1 artifact: DS7 Base 54.033→54.1)
- All 14 C values recomputed from reported Avg cells match to within 0.02 rounding
- All 6 ALFWorld C values from reported Overall correct (Overall verified independently, not recomputed from categories)
- Cross-table T3↔T16 byte-identical for all 5 shared rows (Qwen2.5-Math layers 8,12,14,16,23)
- Table 12 LR ablation: bottom-5 ΔC ≤ 0.02 (matches paper's "at most 0.02" claim); top-5 ΔC up to 0.03 (paper only claims "retain high contribution", no specific bound)
- All §4 prose deltas verified: +2.88, +1.45, +0.99 (Boost B10); +2.90, +2.68 (Only Bk); +0.53, +1.83, +1.76 (Mid-5)
- file-vs-pdfinfo: no page-count defect (23=23)
