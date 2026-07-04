# Dynamic Neural Graph Encoding of Inference Processes in Deep Weight Space — Source-First Breakdown

**Paper:** "Dynamic Neural Graph Encoding of Inference Processes in Deep Weight Space"
**Authors:** Di Wu (Toronto/NUS), Huan Liu* (McMaster), Zhixiang Chi (Toronto), Yuanhao Yu (McMaster), Konstantinos N. Plataniotis (Toronto), Yang Wang (Concordia)
**Venue:** Transactions on Machine Learning Research (TMLR, 06/2026); arXiv:2607.02166v1 [cs.LG] 2 Jul 2026
**Code:** https://github.com/dddiowww/DNG
**Repo context:** 66th paper / rank 61. FIRST repo paper on **deep weight-space networks / metanetworks / neural-network-as-input / INR (implicit neural representation) classification**. Distinct from `program-as-weights` (differentiable-program, not weight-space-metalearning) and from `viq` (visual token quantization, not weight inputs). Sibling-in-spirit to the **graph-neural-network lineage** but uniquely processes *another network's weights as a dynamic (temporal) graph*.

---

## 1. Core idea (one paragraph)

Treat a neural network's weights not as a static graph (prior work NG-GNN/NG-T, Kofinas 2024) but as a **dynamic graph that evolves over discrete timestamps t₁…t_L = one timestamp per forward-pass layer**. Each timestamp adds the next layer's nodes+edges and *deletes* the layer-before-last, so the graph snapshot at t_l is just a bipartite (v^{l−1}→v^l) block — exactly the l-th forward step. A tailored **RNN-based encoder (DNG-Encoder)**, inspired by the Temporal Graph Network (TGN, Rossi), processes these snapshots with a FiLM-style message function and a GRU node-memory update. The **falsifiable hinge**: a *static* graph forces the L-th MPNN layer to solve an **ill-posed inverse problem** (isolate b^l from the compound b^l + W^l·b^{l−1}); the dynamic graph sidesteps it because each node only ever exists at one timestamp, so no inverse extraction is needed (§3.3 + Appendix A). Downstream app **INR2JLS** maps INR weights → a *joint latent space* with image content via a Latent Generator + transposed-conv Decoder (Fig 4), then classifies INRs from that latent map.

---

## 2. Equations (verbatim, with sourcing)

**Forward pass being simulated (Eq 11, L1011-1017):**
a_i^l = σ( Σ_j W_ij^l · a_j^{l−1} + b_i^l )

**Graph-update event O_{t_l} (Eq 3, L283-292):**
- l=1: { (+V, v¹, t₁), (+E, e¹, t₁) }
- 1 < l ≤ L: { (+V, v^l, t_l), (+E, e^l, t_l), (−V, v^{l−2}, t_l), (−E, e^{l−1}, t_l) }
  → at each step add the new layer's nodes+edges, delete the layer-before-last (keep only 2 consecutive node sets ⇒ bipartite snapshot).

**Message Function — single-edge (MLP case), Eq 4 (L381-385):**
m_i(t_l) = ϕ_m^{t_l}( s_j(t_l⁻), e_ij(t_l) ) = Σ_{j∈N_i} W_m1^{t_l} e_ij(t_l) ⊙ W_m2^{t_l} s_j(t_l⁻)
  → FiLM-style conditional scaling (Perez 2018), no target-node/shift info ⇒ "weights only multiplied by previous-layer activations."

**Multi-head message — N-edge (CNN case), Eq 5 (L395-405):**
m_i(t_l) = Σ_{j∈N_i} ϕ_h^{t_l}( Concat[head¹_ij, …, head^N_ij] ),  head^n_ij = W_m1,n^{t_l} e_ij,n ⊙ W_m2,n^{t_l} s_j(t_l⁻)
  → each weight-scalar is its own edge; N=h·w heads per node-pair.

**Recurrent memory update, Eq 6 (L420-425):**
s_i(t_l) = ϕ_u^{t_l}( m_i(t_l), v_i(t_l) ) = LayerNorm( GRU( m_i(t_l), v_i(t_l) ) )

**Permutation equivariance (Eq 12-13, L1089-1101, Appendix C):** W̃^l = P_{π^l} W^l P_{π^{l−1}}^⊤ , b̃^l = P_{π^l} b^l ⇒ dynamic graph G_T equivariant to neuron permutations.

---

## 3. Tables (verbatim + sourcing line-ranges)

### Table 1 — INR classification accuracy (%) (L578-587)
10-view INR augmentation. #Params = inference-time. NG-GNN/NG-T use 64 probe features, scaled to match ours.

| Method | #Params | MNIST | FashionMNIST | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|---|
| NFN | ∼135M | 92.9±0.38 | 75.6±1.07 | 46.6±0.13 | 20.55±0.93 |
| INR2ARRAY (NFT) | ∼59M | 98.5±0.00 | 79.3±0.00 | 63.4±0.00 | 31.30±0.04 |
| NG-GNN | ∼6M | 97.3±0.02 | 86.53±0.58 | 55.11±1.43 | 26.50±1.32 |
| NG-T | ∼6M | 96.83±0.06 | 85.24±0.13 | 57.7±0.36 | 31.65±0.28 |
| **INR2JLS (Ours)** | ∼6M | **98.6±0.01** | **90.6±0.07** | **73.2±0.28** | **42.4±0.32** |

**Prose deltas (§8.1, L636-640) recompute EXACT:** "surpasses other models by at least **9%** and **10%**" on CIFAR-10/100.
- CIFAR-10: 73.2 − NFT 63.4 = **9.8pp** ✓ (≥9%).
- CIFAR-100: 42.4 − NG-T 31.65 = **10.75pp** ✓ (≈10%).
- Abstract "+10% on CIFAR-100-INR" = same 10.75pp ✓ (absolute pp, not relative — see flag 2).

### Table 2 — Kendall's τ, CNN generalization prediction (L623-632)
| Model | CIFAR-10-GS | SVHN-GS | CNN Wild Park |
|---|---|---|---|
| NFN (HNP) | 0.934±0.001 | 0.931±0.005 | — |
| NFN (NP) | 0.922±0.001 | 0.856±0.001 | — |
| NFT | 0.926±0.001 | 0.858±0.000 | — |
| NG-GNN | 0.930±0.001 | 0.863±0.002 | 0.8040±0.0090 |
| NG-T | 0.935±0.000 | 0.872±0.001 | 0.8170±0.0070 |
| **DNG-Encoder (Ours)** | **0.936±0.000** | 0.867±0.002 | **0.8743±0.0021** |

- CIFAR-10-GS "outperforms all": 0.936 vs NG-T 0.935 = **+0.001** (marginal, within ±0.000/0.001).
- SVHN-GS: authors **concede** NFN(HNP) 0.931 ≫ DNG 0.867 (gap 0.064).
- CNN Wild Park: DNG **+0.0573** over NG-T, +0.0703 over NG-GNN — the cleanest win (heterogeneous architectures; NFN/NFT structurally cannot run here, flag 6).

### Table 3 — Ablation (L685-694), accuracy %
**Top — reconstruction framework:** INR2JLS (Ours) 98.6 / 90.6 / 73.2 / 42.4  vs  INR-INR 98.6 / 88.3 / 56.3 / 30.6
**Bottom — modules:** DNG-Encoder-only 96.6 / 78.4 / 54.0 / 25.7 ; INR2JLS w/o Latent Generator 98.4 / 88.9 / 54.5 / 28.1 ; INR2JLS (Ours) 98.6 / 90.6 / 73.2 / 42.4

### Table 4 — Data-augmentation ablation (L696-702), accuracy %
| Augmentation | MNIST | FashionMNIST | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| No Augmentation | 98.5±0.00 | 89.5±0.07 | 66.4±0.19 | 32.9±0.31 |
| Adding Noise | 98.4±0.01 | 89.5±0.06 | 67.3±0.38 | 33.0±0.24 |
| Rotation & Flip | 98.6±0.01 | 90.6±0.07 | 73.2±0.28 | 42.4±0.32 |

**⚠ DEFECT (caught) — §9.1 L708 prose says "comparing the **third and fifth row** in Table 4". Table 4 has only 3 data rows.** The "fifth row" does not exist — stale row-reference, almost certainly leftover from an earlier 5-row draft of the table (likely one row per individual transform: rot90/180/270/hflip/vflip). No headline impact: the stated deltas themselves recompute — "improve over baseline by around **7%** and **9%**" on CIFAR-10/100 = 73.2−66.4 = **6.8pp ≈7%** ✓ and 42.4−32.9 = **9.5pp ≈9%** ✓ (vs No-Aug).

### Table 5 — Inference efficiency (L750-759), single INR on MNIST-INR
| Method | Run Time (s) | Memory (MB) | Comp. Cost (GFLOPs) |
|---|---|---|---|
| NFN | 0.0082±0.00009 | 273.08 | 2.58 |
| NFT | 0.0527±0.00170 | 241.15 | 10.60 |
| NG-GNN | 0.0124±0.00070 | 27.40 | 2.13 |
| NG-T | 0.0092±0.00041 | 29.77 | 14.82 |
| **INR2JLS (Ours)** | **0.0047±0.00018** | 29.17 | **1.31** |

All prose claims recompute EXACT: fastest runtime (0.0047 vs NG-T 0.0092 = 2.0× faster); "slightly higher memory than NG-GNN" (29.17 vs 27.40, +1.77MB) ✓ but ≪ NFN 273 / NFT 241 ✓; lowest comp cost (1.31 < NG-GNN 2.13) ✓.

### Table 6 — Positional-encoding & non-linearity-embedding ablation (L770-779), accuracy %
| Variant | MNIST | Fashion | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| INR2JLS w/ positional encoding | 98.6±0.02 | 89.9±0.09 | 73.5±0.04 | 42.7±0.16 |
| INR2JLS (Ours) | 98.6±0.01 | 90.6±0.07 | 73.2±0.28 | 42.4±0.32 |
| INR2JLS add non-linearity emb | 98.4±0.16 | 90.4±0.01 | 73.2±0.12 | 42.6±0.04 |
| INR2JLS (Ours) | 98.6±0.01 | 90.6±0.07 | 73.2±0.28 | 42.4±0.32 |

**⚠ honest-scope (flag 3):** "positional encoding … no notable performance improvement" (§9.5) — pos-enc is actually **higher on 2/4** datasets (CIFAR-10 73.5>73.2, CIFAR-100 42.7>42.4) and only lower on FashionMNIST (89.9<90.6). Within overlapping std, so "unnecessary" is defensible, but "no improvement" is loose.

### Table 7 — Pure-graph-encoder comparison, no aug / no joint-latent / no probe features (Appendix B, L1060-1067), accuracy %
| Method | #Params | MNIST | FashionMNIST | CIFAR-10 |
|---|---|---|---|---|
| NG-GNN (Static) | ∼0.3M | 79.60±1.30 | 71.10±0.42 | 43.94±0.06 |
| NG-T (Static) | ∼0.4M | 83.43±0.12 | 72.13±0.51 | 44.69±0.03 |
| **DNG-Encoder (Ours)** | ∼0.4M | **96.60±0.09** | **78.40±0.61** | **54.00±0.07** |

- "upgrading static→dynamic yields +**9.31%** on CIFAR-10" (Appendix B L1054): 54.00 − 44.69 = **9.31pp ✓ EXACT**.
- **Cross-table identity ✓:** DNG-Encoder row here (96.60 / 78.40 / 54.00) == Table-3-bottom DNG-Encoder-only row (96.6 / 78.4 / 54.0, +CIFAR-100 25.7 not in T7). Byte-identical.

### Table 8 — Transformer generalization prediction (Appendix F.2, L1576-1581), ∼0.25M params
| Method | Kendall's τ | GFLOPs | Peak Mem (MB) | Latency (ms) |
|---|---|---|---|---|
| NG-GNN | 0.8844±0.006 | 3.22 | 54.49 | 5.94 |
| NG-T | 0.8917±0.002 | 44.69 | 446.95 | 11.77 |
| **DNG-Encoder (Ours)** | **0.9028±0.005** | **0.12** | **25.36** | **4.38** |

All prose claims recompute EXACT: highest τ (0.9028 > NG-T 0.8917, +0.0111) ✓; "avoiding O(N²) global-attention explosion of NG-T" — GFLOPs 0.12 vs 44.69 = **372× less** ✓; fastest latency (4.38 < NG-GNN 5.94) ✓; lowest mem (25.36 < 54.49) ✓.

---

## 4. Cross-table consistency (Python-verified)

- **INR2JLS (Ours) row byte-identical across T1 / T3-top / T3-bottom / T4-Rot&Flip / T6-Ours** → (98.6, 90.6, 73.2, 42.4) ✓ all five.
- **DNG-Encoder-only ablation row byte-identical across T3-bottom / T7** → (96.6, 78.4, 54.0) ✓ (T7 omits CIFAR-100).
- T1 CIFAR-10/100 deltas (9.8 / 10.75 pp), T7 +9.31%, T5 2.0× speedup, T8 372× GFLOPs — all recompute EXACT.
- **NO numeric prose-vs-table contradiction** beyond the T4 "fifth row" stale-reference defect.

---

## 5. Honest-scope flags (⚠)

1. **Table-4 "fifth row" stale reference** (the caught defect) — §9.1 names a row that doesn't exist in the 3-row table; the augmentation deltas it points to recompute fine.
2. **"+10%" / "9% and 10%" are ABSOLUTE percentage-points, not relative gains.** On CIFAR-100 the *relative* improvement over SOTA (NG-T 31.65) is **+33.97%** (42.4/31.65−1); the abstract's "approximately 10%" reads modestly vs the 34% relative gain. Defensible (pp is a common convention) but the framing understates the headline on the hardest dataset.
3. **Positional-encoding "no improvement" (Table 6)** is loose — pos-enc is marginally higher on CIFAR-10 (73.5 vs 73.2) and CIFAR-100 (42.7 vs 42.4); within std so the design choice stands, but "no notable improvement" is not strictly true on 2/4 datasets.
4. **NFT reports ±0.00 std on all 4 Table-1 cells** (98.5±0.00, 79.3±0.00, 63.4±0.00, 31.30±0.04) — zero variance is implausible for a stochastic method; almost certainly single-run with std suppressed, not measured.
5. **Table-2 SVHN-GS is a clear loss** (DNG 0.867 ≪ NFN(HNP) 0.931) and the CIFAR-10-GS win is +0.001 (within noise); the "outperforms all methods on CIFAR-10-GS" headline rests on a 0.001 margin over NG-T.
6. **NFN/NFT structurally cannot run on CNN Wild Park** (architecture-restricted to homogeneous nets) — DNG's strongest Table-2 win is partly "by default"; the real contest there is DNG vs NG-GNN/NG-T only.
7. **Table-7 +9.31% is vs NG-T with probe features REMOVED.** Authors argue probe features "leak dynamic forward-pass info" into the static graph (a shortcut); removing them is methodologically justified, but the +9.31pp is *partly probe-removal-inflated*, not pure static-vs-dynamic architecture gain. (Same caveat applies faintly to Table 1, where NG-GNN/NG-T use 64 probe features per the L586 footnote.)
8. **Table 7 omits CIFAR-100** (only MNIST/Fashion/CIFAR-10) — the cleanest-architecture comparison doesn't cover the hardest dataset where the headline lives.
9. **Transformer benchmark (Table 8) is self-built, not standardized** — 1,000 SimpleViT models on CIFAR-10, 80/10/10 split; not a public model-zoo, so cross-paper comparability is limited.
10. **10-view INR augmentation inflates inference cost** (F_aug ∈ R^{h_s×w_s×6d}); Table-5 efficiency is per-INR but the 6× channel multiplier from augmentation is real overhead not broken out separately.
11. **Applicability boundary (authors' own, §10):** dynamic graphs "excel in representation learning and discriminative tasks" but the paper does not claim weight *generation* or *optimization* capability — INR2JLS is reconstruction→classification, not weight synthesis.

---

## 6. Source-first reconciliation summary

**Built source-first from paper_layout.txt (1714 lines, pdftotext -layout, 28pp [file misreports 17pp, pdfinfo=28 — file-vs-pdfinfo defect recurs iters 66/67/69/70/71/72/73/75/78; trust pdfinfo]). 8 explicit tables + Eqs 3-6/11-13/30-33 + Figs 1-8. Python-verified: Table-1 CIFAR-10/100 deltas 9.8/10.75pp EXACT, Table-7 +9.31% EXACT, Table-5 2.0× speedup EXACT, Table-8 372× GFLOPs / fastest latency EXACT, INR2JLS(Ours) + DNG-Encoder-only rows byte-identical across all their respective tables. CAUGHT 1 genuine defect (Table-4 "fifth row" stale row-reference in a 3-row table — non-numeric, no headline impact). 11 honest-scope flags. NO numeric prose-vs-table contradiction beyond the row-reference defect.**
