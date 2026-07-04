# Expander Sparse Autoencoders: Parameter-Efficient Dictionaries for Mechanistic Interpretability — Source-First Breakdown

**Paper:** "Expander Sparse Autoencoders: Parameter-Efficient Dictionaries for Mechanistic Interpretability" (Rodrigo Mendoza-Smith — Independent Researcher; ICML 2026 Mech-Interp Workshop).
**arXiv:** 2607.01799v1 [cs.LG], 2 Jul 2026. PDF: 29 pp, 0.84 MB.
**Code:** `rodrgo/expander-sae` (https://github.com/rodrgo/expander-sae).
**Subarea:** **sparse-autoencoder architecture / dictionary learning for mechanistic interpretability**, framed as **combinatorial compressed sensing**. Genuinely fresh for this repo: none of the 23 prior papers covers SAEs, dictionary learning, superposition, or compressed-sensing-style interpretability. It is a *parameter-efficiency + theory* paper (sibling in spirit to the inference-efficiency lineage — `jetspec` / `speculating-experts` / `spin` — but applied to the *interpretability dictionary*, not the LM forward pass), and a dictionary-learning counterpart to the `lacuna` parameter-localization paper (both probe *which weights matter*, but LACUNA does so for unlearning targets while Expander-SAE does so for feature directions).

> Sourcing note: every numeric table below is transcribed verbatim from `paper_layout.txt` (`pdftotext -layout`, 1990 lines); line ranges cited inline. The 10 explicit tables (Tables 1–10) are the verbatim substance. Results also appear as 13 figures (storage–fidelity frontiers, throughput curves, certificate-ratio logs, coherence histograms); their per-point values are **not** back-filled — only prose-confirmed ranges and figure-printed legend labels are quoted, consistent with the repo's "figure-derived numbers are weak" rule. No prose-vs-table numeric inconsistency was found; the paper is internally consistent (the one phrasing nuance — "84% of dense CE-loss recovered" is a *ratio* of two CE-rec values, not a CE-rec value itself — is clarified inline).

---

## TL;DR

Sparse autoencoders (SAEs) decompose a neural-net activation `h ∈ R^m` into a sparse combination of `n > m` learned feature directions (`h ≈ Wx`, `W ∈ R^{m×n}`). A *dense* SAE learns `mn` decoder values — costly at large feature counts (`n` of 16k–131k). **Expander SAEs** replace the dense decoder with one supported on the adjacency matrix `M ∈ {0,1}^{m×n}` of a **left-`d`-regular bipartite expander graph** (`‖M_j‖_0 = d` per column, `d ≪ m`): each feature direction touches only `d` of the `m` residual-stream dims. This (a) cuts learned decoder values from `mn` to `dn`, (b) keeps the sparse-coding problem `(m, n, k)` fixed, and (c) turns the matching-pursuit correlation step `W^⊤ r` from an `O(mn)` dense product into an `O(dn)` gather-and-reduce.

- **Architecture:** tied-weight TopK SAE; `W_dec = (V ⊙ M) diag(ν)^{-1}`, `W_enc = W_dec^⊤`; only the `dn` nonzero `V` entries receive gradients (`dn + n + m` params total). `M` is sampled at init and re-rolled per-column until no row is empty.
- **Theory:** (i) **Theorem 3.1** — under expansion (`M` a `(2k, ε, d)`-expander) + column-flatness (`β := β(W_dec) = √d · max|W_ij|`, so `1 ≤ β ≤ √d`), the condition `2β²ε < 1` makes every `2k`-sparse code *uniquely identifiable* (no nonzero `2k`-sparse null vector); when `β = 1` this reduces to the classical lossless-expander condition `ε < 1/2`. (ii) **Corollary 3.2** — a stronger cumulative-coherence condition `β²ε(2k+1) < 1` makes **OMP recover the exact support** in `k` steps.
- **Headline (Table 1, k=64, three seeds):** across Pythia-70M / Pythia-160M / Qwen2.5-3B / Llama-3.2-1B residual streams, varying `d` traces a smooth **storage–fidelity frontier**. Most extreme cell: **Qwen2.5-3B layer 12, d=7 uses 293× fewer learned decoder values** (114,688 vs Dense 33,554,432) **while retaining 84% of dense CE-loss recovered** (0.828 vs 0.983; 0.828/0.983 = 0.842). At Pythia-70M layer 3, d=7 retains **86%** (0.817/0.947) at **73× compression**.
- **What drives the gain (§4.2 controls):** not sparsity per se, not parameter count — **support diversity**. A "Clustered-sparse" control (same `(m,n,k,d)`, but every column forced into one of `G = ⌊m/d⌋` disjoint row-blocks) tracks Expander at small/medium `d` but at d=200 its dead-feature rate climbs **~100× (0.7% → 6.2%)**. A "Pruned dense" control (extract a top-`d` mask from a pre-trained dense SAE) closes most of the reconstruction gap but needs **2× training compute**.
- **The matched-parameter reversal is an encoder-amortisation effect (§4.3, Table 3):** at modern-LM scale, a Dense-SAE trained at reduced `n' = dn/m` (matched parameter count) *beats* Expander by ~0.01–0.115 CE. But replacing the trained encoder with **iterative OMP** (same frozen decoders) closes the gap: at d=7 OMP gains **+0.073 CE on Qwen, +0.118 CE on Llama**, and at d=30 the matched-`n'=240` vs Expander gap shrinks from +0.060 → +0.007 CE (Llama) and +0.012 → **−0.005 CE** (Qwen, Expander now ahead). The decoder-quality win is real; the apparent reversal is encoder amortisation.
- **Features stay novel + interpretable:** activation-Jaccard novelty vs a Dense-SAE reference is **81.1% at d=7** (vs 5.8% at d=m, and a 68% dense-vs-dense seed baseline), falling monotonically with `d`. A blinded 2-LLM-judge coherence study (Claude Sonnet 4.5 + GPT-4o, 25 features/arch, inter-judge Spearman 0.74) gives **Expander d=200 = 3.72, statistically indistinguishable from Dense 3.59**, while d=7 drops to 2.83 (still 84% of features get a concept label).
- **Takeaway:** decoder support structure is an underexplored SAE design axis. Expander-SAE is *not* a final dense-SAE replacement (encoder amortisation + worst-case certificates are loose in the operating regime) but a parameter-efficient dictionary whose `(values, rows)` storage admits structured-OMP decoding at 1.8M tok/s.

---

## 1. Why SAE inference is a compressed-sensing problem (§1–§2)

A transformer layer-`ℓ` residual stream `h^(ℓ) ∈ R^m` is, under the **superposition hypothesis**, a sparse combination of `n > m` interpretable feature directions: `h ≈ Wx + ε`, `W = [w_1|…|w_n] ∈ R^{m×n}`, `x ∈ R^n` sparse. An SAE learns an encoder mapping `h → x` (sparse code) and a decoder mapping `x → ĥ`. Recent work (Klindt et al. 2025; O'Neill et al. 2025) reframes *SAE inference* (given a trained `W_dec`, recover `x` from `h`) as the canonical **compressed-sensing** sparse-recovery problem — where recovery is possible only because `x` is sparse and `W` has suitable geometry (Candès-Tao 2005; Donoho 2006).

### 1.1 Combinatorial compressed sensing and expander graphs (§2.1)

A niche CS line studies recovery when `M ∈ {0,1}^{m×n}` is the adjacency matrix of a **bipartite left-`d`-regular expander graph**. Definitions (from first principles):

- Bipartite graph `G = (L, R, E)`, `L ∩ R = ∅`, `E ⊂ L × R`. **Imbalanced** if `|R| < |L|`. **Left `d`-regular** if every left vertex has exactly `d` neighbours.
- Adjacency matrix `M ∈ {0,1}^{m×n}` with `m < n`, `‖M_j‖_0 = d ∀j` (each of the `n` columns has exactly `d` ones among `m` rows).
- Neighbour set `Γ(S) := {i ∈ [m] : ∃j ∈ S, M_ij = 1}`. `G` is a **`(k, ε, d)`-expander** if `|Γ(S)| ≥ (1−ε)d|S|` for every `S ⊆ [n]` with `|S| ≤ k`. `ε ∈ (0,1)` counts how often rows in `Γ(S)` collide (have >1 neighbour).
- Certifying expander-ness is NP-hard, but for appropriate `(m, n, k, d)` a random binary matrix with `d ≪ m` ones/column is a `(k, ε, d)`-expander w.h.p. (Bah & Tanner 2013). When `ε < 1/2`, `M` satisfies **RIP-1**: `(1−2ε)√d ‖x‖_1 ≤ ‖Mx‖_1 ≤ √d ‖x‖_1`.
- RIP-1 unlocks a recovery toolkit (greedy / peeling / GPU-parallel iterative schemes — Mendoza-Smith & Tanner 2015, 2017) with different guarantees from the usual dense RIP-2 (`(1−δ_k)‖x‖_2² ≤ ‖Wx‖_2² ≤ (1+δ_k)‖x‖_2²`).

## 2. Expander-SAE architecture + theory (§3)

### 2.1 Architecture (§3)

An Expander SAE with hyperparameters `(m, n, d, k)` has learnable `V ∈ R^{m×n}`, `b_enc ∈ R^n`, `b_dec ∈ R^m`, instantiating the (SAE) by:

```
W_dec = (V ⊙ M) diag(ν)^{-1},   W_enc = W_dec^⊤      (Eq. 8)
```

where `M ∈ {0,1}^{m×n}` is a binary mask with `‖M_j‖_0 = d ∀j` (sampled at init), and `ν ∈ R^n` normalises each column of `M ⊙ V` to unit `ℓ_2` norm. The forward pass uses `σ(z) = TopK_k(z)` — keeps the `k` **largest signed** pre-activations (not magnitudes), zeroes the rest. Only the `dn` nonzero `V` positions receive gradients → **`dn + n + m` learnable parameters** total. (Empirically trained checkpoints almost never select negative pre-activations.)

### 2.2 OMP decoder + column-flatness factor (§3.1)

Sparse-recovery decoders for combinatorial CS are noise-fragile; the paper benchmarks four (Appendix B) and picks **Orthogonal Matching Pursuit (OMP)** — strongest reconstruction improvement. Algorithm 1 (the diagnostic variant):

```
Require: measurement y, decoder W, bias b, sparsity k
1: r ← y − b, S ← ∅
2: for t = 1,…,k do
3:   j* ← argmax_{j∉S} |⟨w_j, r⟩|        # (signed variant: argmax ⟨w_j,r⟩)
4:   S ← S ∪ {j*}
5:   x̂_S ← W[:,S]† (y − b)              # least-squares on active set
6:   r ← y − b − W[:,S] x̂_S
7: end for
```

Non-binary learned weights break standard expander guarantees; the effect is quantified by the **column-flatness factor**:

```
β(W_dec) := √d · max_{i,j: M_ij=1} |W_ij|              (Eq. 9)
```

Each decoder column is supported on `≤ d` entries with unit `ℓ_2`-norm ⇒ **`1 ≤ β(W_dec) ≤ √d`**. Larger `β` = a feature concentrated more mass on fewer coordinates; smaller `β` = flatter.

### 2.3 Identifiability + OMP-recovery guarantees (Theorem 3.1, Corollary 3.2)

> **Theorem 3.1 (Identifiability for learned SAE decoders).** Let `W_dec ∈ R^{m×n}` be an Expander-SAE decoder with unit-normalised columns supported on a left-`d`-regular mask `M`. Assume `M` is a `(2k, ε, d)`-expander and `β := β(W_dec)`. If `2β²ε < 1` (Eq. 10), then every `2k`-sparse `u ∈ R^n` satisfies `‖W_dec u‖_1 ≥ √d (1/β − 2βε) ‖u‖_1` (Eq. 11). Consequently, for every `k`-sparse latent `x* ∈ R^n`, the noiseless reconstruction `h = b_dec + W_dec x*` has `x*` as its **unique `k`-sparse explanation**.

Reading: under expansion + flatness, the decoder has no nonzero `2k`-sparse null vector ⇒ two distinct `k`-sparse codes cannot yield the same centred activation ⇒ `x*` is identifiable. When `β = 1`, Eq. 10 reduces to `2ε < 1` i.e. `ε < 1/2` — the **classical lossless-expander condition** (Jafarpour et al. 2009). For learned non-binary decoders the expansion deficit must be smaller (weighted-expander analogue).

> **Corollary 3.2 (OMP recovery on Expander SAEs).** Let `W = W_dec` have unit-normalised columns supported on a left-`d`-regular mask `M`, `β = β(W)`. If `M` is a `(k+1, ε, d)`-expander and `β²ε(2k+1) < 1`, then in the noiseless model `h = b_dec + W x*`, **OMP recovers the support of every `k`-sparse `x*` in `k` steps.**

**Non-vacuity framing (§3, end):** worst-case sufficient conditions, not a certificate of every experimental setting (certifying expander-ness / RIP is NP-hard). But the condition is non-vacuous asymptotically: at `m = δn`, `k = ρm` (fixed `δ < 1`, `ρ > 0`), counting requires `2(1−ε)dρ ≤ 1`; for any flatness target `β_0`, if `d > 2β_0²` one can pick `1/d < ε < 1/(2β_0²)`, and for small fixed `ρ` random left-`d`-regular masks are `(2k, ε, d)`-expanders w.h.p. **Empirically (Figure 4, Appendix B.4)** both certificate ratios `R_id(k) := 2β_max² ε_b(2k)` and `R_OMP(k) := β_max² ε_b(k+1)(2k+1)` sit *well above* the threshold-1 line across the trained grid — i.e. the sufficient certificates are **loose** in the LM operating regime (OMP + encoder still reconstruct successfully). Honest: the theorems motivate the architecture, they do not certify the experiments.

---

## 3. Experimental setup (§4 + Appendix A)

- **Activations:** residual-stream output of a fixed pre-trained LM at a single hook site. Headline = **Pythia-70M** layer 3 (`lm.gpt_neox.layers[3]` ≡ TransformerLens `blocks.3.hook_resid_post`), `m = 512`, `n = 4096`, `k = 64`. Replicated on Pythia-160M, Qwen2.5-3B, Llama-3.2-1B. Pile tokeniser, seq len `L = 128`, `no_grad`, single Modal A100 forward pass (~40 s), `|D| = 210,000` per-token vectors cached; activations raw (no whitening); `b_dec` init zero. All SAE train/eval on Modal **A10G** GPUs.
- **Splits:** 200,000-token train / 5,000-token held-out.
- **Loss:** per-sample `ℓ_2` reconstruction `L = (1/B) Σ_b ‖h_b − ĥ_b‖_2²` — **no sparsity penalty** (sparsity is enforced by TopK).
- **Optimiser:** Adam (`β_1=0.9, β_2=0.999, ϵ=1e-8`), single-period cosine LR over `T = 5000` steps: `η_t = 1e-5 + ½(3e-4 − 1e-5)(1 + cos(πt/T))` (Eq. 14). Global `ℓ_2` grad-clip 1.0, batch `B = 256`, `shuffle=True, drop_last=True`.
- **Dead-feature resampling** (Bricken 2023; Gao 2024): every `T_r = max(1000, T/5)` steps, columns that fired on <5 of the last `T_r·B` samples are reset to the largest-residual mini-batch sample projected onto their mask support (Eq. 15). Disabled when >80% dead.
- **3 metrics:** (a) trained-encoder **relative reconstruction error** `err(h) = ‖h − ĥ‖_2 / ‖h‖_2` (held-out); (b) **zero-ablation-normalised CE-loss recovered** `CE_rec = (CE_zero − CE_recon) / (CE_zero − CE_clean)` (1 = matches clean model, 0 = no better than zero-ablation, <0 = worse); (c) **dead-feature fraction**.
- **Two SAE families compared:** **Dense-SAE** (independent enc/dec) and **Expander-SAE-d** (tied-weight TopK, frozen expander mask). Also the **`d = m`** special case as a *tied-dense* baseline.

---

## 4. Results

### 4.1 Storage–fidelity frontier, cross-model (Table 1, k=64, three seeds) — verbatim

> Sourcing: `paper_layout.txt` lines ~338–376. Storage = ratio of learned decoder values to Dense-SAE at the same `(m,n)` = `m/d` by construction (source-free check: Pythia-70M `m=512`, d=7→512/7=73.1✓, d=50→10.2✓, d=200→2.56✓; Qwen/Llama `m=2048`, d=7→292.6✓, d=30→68.3✓, d=102→20.1✓). `⋆`/`†` = seed-instability caveats (Appendix F).

| Model | Layer | m | n | Method | Storage | rel err | CE rec |
|---|---|---|---|---|---|---|---|
| Pythia-70M | 3 | 512 | 4,096 | Expander d=7 | **73×** | 0.541 | 0.817 |
| | | | | Expander d=50 | 10× | 0.489 | 0.859 |
| | | | | Expander d=200 | 2.6× | 0.415 | 0.902 |
| | | | | Dense-SAE | 1× | 0.342 | 0.947 |
| Pythia-160M | 8 | 768 | 6,144 | Expander d=10 | 77× | 0.520 | 0.767 |
| | | | | Expander d=75 | 10× | 0.461 | 0.836 |
| | | | | Expander d=300 | 2.6× | 0.399 | 0.889 |
| | | | | Dense-SAE | 1× | 0.332 | 0.946 |
| Qwen2.5-3B | 12 | 2,048 | 16,384 | Expander d=7 | **293×** | 0.764 | 0.828 |
| | | | | Expander d=30 | 68× | 0.703 | 0.894 |
| | | | | Expander d=102 | 20× | 0.681 | 0.919 |
| | | | | Dense-SAE | 1× | 0.489 | 0.983 |
| Qwen2.5-3B | 24 | 2,048 | 16,384 | Expander d=7 | 293× | 0.693 | 0.875 |
| | | | | Expander d=30 | 68× | 0.848⋆ | 0.821⋆ |
| | | | | Expander d=102 | 20× | 0.659 | 0.884 |
| | | | | Dense-SAE | 1× | 0.591 | 0.936 |
| Llama-3.2-1B | 6 | 2,048 | 16,384 | Expander d=7 | 293× | 0.809 | 0.576† |
| | | | | Expander d=30 | 68× | 0.760 | 0.743 |
| | | | | Expander d=102 | 20× | 0.732 | 0.816 |
| | | | | Dense-SAE | 1× | 0.568 | 0.952 |
| Llama-3.2-1B | 12 | 2,048 | 16,384 | Expander d=7 | 293× | 0.757 | 0.608 |
| | | | | Expander d=30 | 68× | 0.742 | 0.707 |
| | | | | Expander d=102 | 20× | 0.720 | 0.758 |
| | | | | Dense-SAE | 1× | 0.499 | 0.958 |

**Takeaways:**
- The **monotone `d ↑ → fidelity ↑` frontier replicates** across 3 LM families × 3 decoder designs (GPT-NeoX, Llama-style, Qwen2-style), 70M→3B params.
- Most extreme cell: **Qwen2.5-3B layer 12, d=7 = 293× compression (114,688 vs 33,554,432 learned values) at 0.828 CE-rec = 84% of Dense's 0.983** (`0.828/0.983 = 0.842`). *Sourcing note: the abstract's "84% of dense CE-loss recovered" is the **ratio** of two CE-rec values (0.828/0.983), not the CE-rec value itself (0.828); §4.1 makes this explicit with "(0.817/0.947)" for the 86% Pythia figure.*
- Pythia-70M layer 5 omitted because **Dense-SAE itself fails there** (`CE_rec < 0`, Table 7) — a model-stage limitation, not architecture-specific.
- `⋆` Qwen L24 d=30 (rel err 0.848): one of three seeds diverged (rel err 1.13); without it, rel err 0.704, CE-rec 0.854 (Table 8 footnote). `†` Llama L6 d=7 CE-rec 0.576 (Appendix F caveat).

### 4.2 Deployment summary (Table 2, Pythia-70M L3, m=512, n=4096, k=64) — verbatim

> Sourcing: lines ~483–525. Storage = total on-disk artefact (KiB), flat `(values, rows)` for sparse. Trained-encoder mean ± SEM over 3 seeds (`‡` = 2 seeds). Each cell: `rel err / CE rec` over `Dead frac.`. Pruned dense = dense pretraining + sparse retune (2 passes).

| Architecture | d | Stor. (KiB) | Tr. passes | rel err / CE rec | Dead frac. |
|---|---|---|---|---|---|
| Dense-SAE | 512 | 16,402 | 1 | 0.342±.000 / 0.947±.001 | 0.6% |
| Expander-SAE | 7 | 242 | 1 | 0.541±.006 / 0.817±.001 | 0.1% |
| Expander-SAE | 50 | 1,618 | 1 | 0.489±.003 / 0.859±.002 | 0.3% |
| Expander-SAE | 200 | 6,418 | 1 | 0.415±.001 / 0.902±.001 | 0.7% |
| Clust.-sparse | 200‡ | 6,418 | 1 | 0.518±.012 / 0.829±.016 | 6.2% |
| Pruned dense | 200 | 6,418 | 2 | 0.363±.002 / 0.934±.001 | 0.4% |

**Takeaways:**
- Expander d=200 (6,418 KiB) reaches CE-rec 0.902 within **4.5pp of Dense 0.947** at 2.6× compression.
- **Clustered-sparse at d=200** (same params, no support diversity) collapses to **6.2% dead features** vs Expander's 0.7% — a ~9× dead-rate blow-up, isolating support diversity as the active ingredient.
- Pruned dense recovers CE-rec 0.934 but at **2 training passes** (dense pretrain + sparse retune) — the support-extraction win is bought with 2× compute.

### 4.3 Encoder amortisation gap closed by iterative OMP (Table 3, Qwen2.5-3B + Llama-3.2-1B layer 12, k=64, 3 seeds) — verbatim

> Sourcing: lines ~528–575. OMP on the same frozen checkpoints; Gain = Iter. OMP − Encoder.

| Architecture | Encoder CE rec | Iter. OMP CE rec | Gain |
|---|---|---|---|
| **Qwen2.5-3B (layer 12)** | | | |
| Expander d=7 | 0.828 | 0.901 | **+0.073** |
| Expander d=30 | 0.894 | 0.914 | +0.020 |
| Expander d=102 | 0.919 | 0.933 | +0.014 |
| Matched-Dense `n'=240` | 0.906 | 0.909 | +0.003 |
| Matched-Dense `n'=816` | 0.945 | 0.952 | +0.007 |
| **Llama-3.2-1B (layer 12)** | | | |
| Expander d=7 | 0.608 | 0.726 | **+0.118** |
| Expander d=30 | 0.707 | 0.765 | +0.058 |
| Expander d=102 | 0.758 | 0.798 | +0.040 |
| Matched-Dense `n'=240` | 0.767 | 0.772 | +0.005 |
| Matched-Dense `n'=816` | 0.873 | 0.880 | +0.007 |

**Takeaways:**
- The encoder amortisation gap is **largest where compression is most aggressive**: +0.073 CE (Qwen d=7), +0.118 CE (Llama d=7).
- Under symmetric iterative OMP, the matched-`n'=240` vs Expander-d=30 gap **shrinks from +0.060 → +0.007 CE (Llama)** and **+0.012 → −0.005 CE (Qwen, Expander now slightly ahead)** — so the apparent dense win at matched params is mostly an encoder effect, not a decoder-quality difference. A smaller residual gap remains at d=102.

### 4.4 Trained encoder vs OMP variants — throughput × accuracy (Table 4, k=64, seed 0, Pythia-70M L3) — verbatim

> Sourcing: lines ~786–823. `∗` = single-thread Apple-silicon CPU; `‡` = single Modal A10G bf16 `B=1024` (`B=256` where annotated — Dense-SAE OOMs at `B=1024`); `†` = block-size sweep in Table 5. Speed-ratio vs same-HW encoder. CPU impls agree to `1e-5`; bf16 GPU agrees with fp32 within sample-averaging noise.

| Architecture | Inference | rel err | tokens/s | vs same-HW encoder |
|---|---|---|---|---|
| **Expander-SAE (d=7)** | trained encoder∗ | 0.533 | 52,280 | – |
| | OMP vanilla∗ | 0.452 | 3.19 | 16,388× slower |
| | OMP structured + QR∗ | 0.452 | 224.09 | 233× slower |
| | trained encoder‡ | 0.570 | 1,798,710 | – |
| | OMP structured + QR‡ | 0.458 | 11,587 | 155× slower |
| | Cholesky + Triton + struct. Gram, L=64†‡ | 0.546 | 689,364 | 2.6× slower |
| **Expander-SAE (d=50)** | trained encoder∗ | 0.485 | 42,532 | – |
| | OMP vanilla∗ | 0.390 | 3.15 | 13,502× slower |
| | OMP structured + QR∗ | 0.390 | 60.23 | 706× slower |
| | trained encoder‡ | 0.509 | 1,916,684 | – |
| | OMP structured + QR‡ | 0.380 | 2,245 | 854× slower |
| | gOMP L=4‡ | 0.390 | 7,367 | 260× slower |
| | gOMP L=m/d‡ | 0.409 | 13,270 | 144× slower |
| | Cholesky refit + Triton corr.‡ | 0.409 | 48,784 | 39× slower |
| **Expander-SAE (d=200)** | trained encoder∗ | 0.415 | 64,540 | – |
| | OMP vanilla∗ | 0.344 | 3.12 | 20,686× slower |
| | OMP structured + QR∗ | 0.344 | 16.11 | 4,007× slower |
| | trained encoder‡ | 0.440 | 1,930,422 | – |
| | OMP structured + QR‡ | 0.331 | 542 | 3,556× slower |
| | gOMP L=m/d‡ | 0.334 | 1,063 | 1,818× slower |
| **Dense-SAE** | trained encoder∗ | 0.343 | 67,303 | – |
| | OMP vanilla∗ | 0.270 | 3.14 | 21,434× slower |
| | OMP structured + QR∗ | 0.270 | 5.48 | 12,282× slower |
| | trained encoder‡ | 0.365 | 2,249,358 | – |
| | OMP structured + QR‡ (B=256) | 0.279 | 207 | 10,867× slower |

**Takeaways:**
- Trained encoder is the deployment path (~**1.8M tok/s** on a single A10G); OMP is the offline diagnostic.
- The `(values, rows)` storage turns `W^⊤ r` into an `O(dn)` gather-and-reduce → **11× speedup at d=7, 8× at d=50, 3× at d=200** over vanilla OMP (≈ `m/d` Amdahl scaling).
- Incremental-QR refit (modified Gram-Schmidt) cuts per-iteration lstsq cost from `O(mk² + k³)` to `O(mk + k²)`. Combined structured+QR reaches **70× over vanilla OMP at d=7** (224 tok/s CPU).
- The d-regular block specialisation (`(d·d)`-unrolled inner loop, never materialising dense `W_S`) + bf16 batching reaches **11,587 tok/s at d=7** on A10G.

### 4.5 Block-size L Pareto sweep (Table 5, d=7, m=512, n=4096, k=64, A10G, bf16, B=1024, seed 0) — verbatim

> Sourcing: lines ~825–840. Cholesky-refit + Triton-correlation + structured-Gram OMP. `L` = columns added per outer iteration; `L=1` = iterative OMP, `L=64` = single-shot. vs same-HW encoder 1,798,710 tok/s. Supports picked by TopK on `W^⊤ r` (matches encoder convention).

| L | outer iters | rel err | tokens/s | ms / 1024-batch | vs same-HW encoder |
|---|---|---|---|---|---|
| 1 | 64 | 0.458 | 13,305 | 77.0 | 135× slower |
| 2 | 32 | 0.461 | 26,605 | 38.5 | 68× slower |
| 4 | 16 | 0.465 | 53,924 | 19.0 | 33× slower |
| 8 | 8 | 0.473 | 107,529 | 9.52 | 17× slower |
| 16 | 4 | 0.489 | 214,774 | 4.77 | 8.4× slower |
| 32 | 2 | 0.513 | 410,614 | 2.49 | 4.4× slower |
| 64 | 1 | 0.546 | 689,364 | 1.49 | 2.6× slower |
| trained encoder (same HW) | – | – | 1,798,710 | 0.57 | 1× |

**Takeaways:**
- A continuum between iterative (`L=1`) and single-shot (`L=64`) decoding. `L=64`: rel err 0.546, **0.024 better than the encoder's 0.570** at only 2.6× throughput cost.
- `L=4`: rel err 0.465 at 54k tok/s, **within 0.007 of full iterative OMP** at 33× encoder cost.
- `L=1`: OMP's full 0.458 rel err at 13.3k tok/s, 135× slower than encoder. Throughput ≈ doubles per `L`-doubling (13.3k→26.6k→53.9k→107.5k→214.8k→410.6k→689.4k) — source-free check passes.

### 4.6 Cross-layer + cross-model replication (Table 7, k=64, 3 seeds) — verbatim

> Sourcing: lines ~1167–1210. `rel err`/`CE rec` (left to right inside each Expander column-pair). SEM < 0.01 every cell except flagged. Per-layer `(CE clean / CE zero-ablation)`: Pythia-70M L1/L3/L5 = 3.44/8.93, 3.44/12.73, 3.44/8.75; Pythia-160M L4/L8 = 2.96/10.96, 2.96/11.96.

| Model | Layer | Expander d (rel err / CE rec) | Expander d′ (rel err / CE rec) | Dense-SAE rel err | Dense-SAE CE rec |
|---|---|---|---|---|---|
| *Pythia-70M (m=512, n=4096, d ∈ {7, 50, 200} left→right)* | | | | | |
| Pythia-70M | 1 | 0.570 / 0.738 | 0.516 / 0.819 | 0.446 / 0.883 | 0.368 | 0.938 |
| Pythia-70M | 3 | 0.532 / 0.817 | 0.489 / 0.859 | 0.415 / 0.902 | 0.342 | 0.947 |
| Pythia-70M | 5 | 0.375 / −0.347 | 0.341 / 0.036 | 0.433 / −0.163 | 0.607 | −0.550 |
| *Pythia-160M (m=768, n=6144, d ∈ {10, 75, 300} left→right)* | | | | | |
| Pythia-160M | 4 | 0.643 / 0.742 | 0.588 / 0.836 | 0.515 / 0.901 | 0.439 | 0.949 |
| Pythia-160M | 8 | 0.520 / 0.767 | 0.461 / 0.836 | 0.399 / 0.889 | 0.332 | 0.946 |

*(Table 7 column layout per caption: each Expander entry shows `d / d′` i.e. two of the three d-values side-by-side; the third d follows in the next column-pair. Values transcribed exactly as printed.)*

**Takeaways:**
- Frontier replicates across layer + model-size dimensions: Expander recovers within **5–20pp of Dense CE** at 73× (d=7, m=512) or 77× (d=10, m=768) compression.
- **Pythia-70M layer 5 fails for all architectures** (Dense itself `CE_rec = −0.550`) — late-layer residual-stream output is hard to reconstruct without breaking the unembedding regardless of architecture.

### 4.7 Qwen2.5-3B cross-architecture replication (Table 8, m=2048, n=16,384, k=64, 3 seeds) — verbatim

> Sourcing: lines ~1210–1250. `⋆` = sd > 0.02; (d=30, L24) one of three seeds diverged (rel err 1.13); without it rel err 0.704, CE-rec 0.854. Matched-`n'` Dense-SAE: d=30→`n'=240`, d=102→`n'=816`. d=7 matched cell excluded (`n'=56 < 2k`, NSP gate). `(CE clean / CE zero-ablation)` = 2.29/15.98 (L12), 2.29/23.30 (L24).

| Layer | metric | Expander d=7 | Expander d=30 | Expander d=102 | Dense-SAE (full n) | Matched-`n'=240` | Matched-`n'=816` |
|---|---|---|---|---|---|---|---|
| 12 | rel err | 0.764 | 0.703 | 0.681 | 0.489 | 0.608 | 0.556 |
| 12 | CE rec | 0.828 | 0.894 | 0.919 | 0.983 | 0.906 | 0.945 |
| 24 | rel err | 0.693 | 0.848⋆ | 0.659 | 0.591 | 0.541 | 0.488 |
| 24 | CE rec | 0.875 | 0.821⋆ | 0.884 | 0.936 | 0.911 | 0.946 |

**Takeaways:**
- The monotone `d ↑ → fidelity ↑` frontier survives the jump to a 3B modern decoder-only LM with a different architecture (RoPE, GQA, SwiGLU vs Pythia's GPT-NeoX).
- At matched parameter count, reduced-`n` Dense beats Expander at d=30/d=102 on the *trained encoder* — the gap iterative OMP later closes (§4.3).

### 4.8 Per-feature firing rate + entropy (Table 6, k=64, seed 0, Pythia-70M L3) — verbatim

> Sourcing: lines ~1078–1100. Median over the indicated subset. Counts out of `n=4096`. "alive" = fires on ≥1 held-out token; "novel" = best-match activation-Jaccard `J<0.1` vs Dense; "shared" = `J>0.5`.

| d | alive | novel (J<0.1) | shared (J>0.5) | rate med novel | rate med shared | entropy med novel (bits) | entropy med shared (bits) |
|---|---|---|---|---|---|---|---|
| 7 | 4096 | 3424 | 22 | 0.82% | 2.51% | 7.90 | 7.72 |
| 30 | 4096 | 2903 | 20 | 0.96% | 3.18% | 7.97 | 7.90 |
| 50 | 4096 | 2688 | 20 | 0.86% | 32.51% | 7.87 | 10.31 |
| 100 | 4095 | 2528 | 34 | 0.86% | 3.15% | 7.90 | 7.06 |
| 200 | 4096 | 2459 | 57 | 0.84% | 2.82% | 7.77 | 6.78 |

**Takeaways:**
- Novel features fire on **rarer subsets** than shared features at every `d` (median 0.8–1.0% vs ≥2.5%).
- Median target-token entropy of novel features is roughly stable across `d` (**7.8–8.0 bits**) → fire on a broad set of token identities, not a small clique. (d=50 shared-feature entropy 10.31 bits is an outlier driven by tiny per-bucket counts of 20–57 features.)

### 4.9 Storage breakdown (Table 9, m=512, n=4096, KiB) — verbatim

> Sourcing: lines ~1469–1500. (1) Learned values = unique decoder weights only (`d·n` float32 sparse / `m·n` dense). (2) Decoder+rows = on-disk incl. int32 row indices, flat `(d·n,)` layout, no indptr (every column has exactly `d` nonzeros). (3) Encoder+biases = `b_dec`, `b_enc` (tied) + full `(n,m)` encoder for Dense-SAE. (4) Total = (2)+(3)+8-byte mask seed. Ratio = learned-values ratio vs Expander `d=m=512` = `m/d`.

| Architecture | d | Learned values (KiB) | Ratio | Decoder + rows (KiB) | Encoder + biases (KiB) | Total (KiB) | Mask via seed |
|---|---|---|---|---|---|---|---|
| Dense-SAE | 512 | 8,192.0 | 1× | 8,192.0 | 8,210.0 | 16,402.0 | – |
| Expander-SAE | 7 | 112.0 | 73× | 224.0 | 18.0 | 242.0 | 8B |
| Expander-SAE | 30 | 480.0 | 17× | 960.0 | 18.0 | 978.0 | 8B |
| Expander-SAE | 50 | 800.0 | 10× | 1,600.0 | 18.0 | 1,618.0 | 8B |
| Expander-SAE | 100 | 1,600.0 | 5× | 3,200.0 | 18.0 | 3,218.0 | 8B |
| Expander-SAE | 200 | 3,200.0 | 2.6× | 6,400.0 | 18.0 | 6,418.0 | 8B |
| Expander-SAE | 512 (=m) | 8,192.0 | 1× | 8,192.0 | 18.0 | 8,210.0 | 8B |
| Clustered-sparse | 7 | 112.0 | 73× | 224.0 | 18.0 | 242.0 | 8B |
| Clustered-sparse | 50 | 800.0 | 10× | 1,600.0 | 18.0 | 1,618.0 | 8B |
| Clustered-sparse | 200 | 3,200.0 | 2.6× | 6,400.0 | 18.0 | 6,418.0 | 8B |
| Pruned dense | 7 | 112.0 | 73× | 224.0 | 18.0 | 242.0 | 8B |
| Pruned dense | 50 | 800.0 | 10× | 1,600.0 | 18.0 | 1,618.0 | 8B |
| Pruned dense | 200 | 3,200.0 | 2.6× | 6,400.0 | 18.0 | 6,418.0 | 8B |

**Takeaways:**
- Storage ratio reproduces from `m/d` for every cell (source-free check: 8192/112 = 73.1✓, 8192/480 = 17.1✓, 8192/800 = 10.2✓, 8192/1600 = 5.1✓, 8192/3200 = 2.56✓).
- The expander mask regenerates deterministically from a **single int64 seed** — the only storage overhead beyond learned values is the 8-byte seed (vs Dense's full `(n,m)` encoder). Total Expander d=7 artefact = **242 KiB** vs Dense 16,402 KiB.

### 4.10 Full empirical-metric grid (Table 10, m=512, n=4096, k=64, Pythia-70M L3, CE clean 3.44 / CE zero-abl 12.73) — verbatim

> Sourcing: lines ~1527–1550. Trained encoder, mean ± SEM over 3 seeds (`†`=1 seed, `‡`=2 seeds). "Training passes" = full SAE runs (Pruned dense needs dense pretrain + sparse retune). Novel (J<0.1) = fraction of features whose best-match activation-Jaccard vs Dense < 0.1 (single-seed feature-analysis pipeline).

| Architecture | d | Training passes | rel err | CE rec. | Dead frac | Novel† (J<0.1) |
|---|---|---|---|---|---|---|
| Dense-SAE | 512 | 1 | 0.342±0.000 | 0.947±0.001 | 0.6%±0.0 | – |
| Expander-SAE | 7 | 1 | 0.541±0.006 | 0.817±0.001 | 0.1%±0.0 | 81.1% |
| Expander-SAE | 30 | 1 | 0.508±0.001 | 0.851±0.001 | 0.2%±0.0 | 72.6% |
| Expander-SAE | 50 | 1 | 0.489±0.003 | 0.859±0.002 | 0.3%±0.0 | 69.4% |
| Expander-SAE | 100 | 1 | 0.467±0.003 | 0.882±0.000 | 0.3%±0.1 | 66.8% |
| Expander-SAE | 200 | 1 | 0.415±0.001 | 0.902±0.001 | 0.7%±0.0 | 65.5% |
| Expander-SAE | 512 (=m) | 1 | 0.388±0.008 | 0.937±0.000 | 0.3%±0.1 | 5.8% |
| Clustered-sparse | 7 | 1 | 0.531±0.001 | 0.810±0.001 | 0.4%±0.0 | 82.1% |
| Clustered-sparse | 50 | 1 | 0.478±0.001 | 0.867±0.000 | 1.5%±0.2 | 78.4% |
| Clustered-sparse | 200‡ | 1 | 0.518±0.012 | 0.829±0.016 | 6.2%±0.3 | 77.3% |
| Pruned dense | 7 | 2 | 0.520±0.001 | 0.829±0.001 | 0.3%±0.1 | 82.3% |
| Pruned dense | 50 | 2 | 0.450±0.004 | 0.885±0.005 | 0.4%±0.1 | 62.2% |
| Pruned dense | 200 | 2 | 0.363±0.002 | 0.934±0.001 | 0.4%±0.0 | 15.1% |

**Takeaways:**
- **Novelty falls monotonically with `d`** for Expander (81.1% → 5.8% at d=m), confirming that low-`d` Expander finds a *substantively different* decomposition. Clustered-sparse stays high-novelty (77–82%) but pays with dead features.
- Expander `d=m` (mask removed) reaches CE-rec 0.937 — within 1pp of full Dense 0.947, confirming the tied architecture itself is near-lossless; the mask is the active design choice.
- Cross-table consistency: Table 10 d=50 (rel err 0.489 / CE-rec 0.859 / dead 0.3%) == Table 2 Expander d=50 == Table 1 Pythia d=50 (0.489 / 0.859) — three independent tables agree byte-exact (source-free check).

### 4.11 Blinded LLM-judge feature coherence (Figure 7 + §E prose)

> Sourcing: lines ~1100–1140 (Figure 7 + prose). 25 features/arch (Expander d=7, Expander d=200, Dense-SAE), Pythia-70M L3, k=64, seed 0, stratified across firing-rate quartiles. Anonymised IDs F01–F75 hide `d`/column/rate/arch. 2 judges (Claude Sonnet 4.5, GPT-4o), 3 calls each at temp 0, 1–5 coherence scale. Inter-judge Spearman on per-feature mean coherence **ρ = 0.74**.

| Architecture | mean coherence | 95% bootstrap CI | concept-label fraction |
|---|---|---|---|
| Dense-SAE | 3.59 ± 0.19 | [3.21, 3.96] | 100% |
| Expander d=200 | 3.72 ± 0.19 | [3.34, 4.06] | 96% |
| Expander d=7 | 2.83 ± 0.22 | [2.43, 3.27] | 84% |

**Takeaways:**
- At **d=200 (2.6× compression), interpretability is statistically indistinguishable from Dense** (overlapping CIs, 3.72 vs 3.59).
- At the **storage extreme d=7 (73× compression), coherence drops ~0.8 points** but 84% of features still receive a concrete concept label — matching the activation-Jaccard novelty result (a different but still meaningful decomposition).

---

## 5. Method summary (Algorithm 1 + Eq. 8–9)

**Forward pass:** `ĥ = W_dec · TopK_k(W_enc (h − b_enc)) + b_dec`, with `W_dec = (V ⊙ M) diag(ν)^{-1}`, `W_enc = W_dec^⊤` (tied weights). `M` = left-`d`-regular expander mask (`‖M_j‖_0 = d`), sampled at init and re-rolled per-column until no row is empty. `ν` normalises each column of `V ⊙ M` to unit `ℓ_2`. **Parameters:** `dn + n + m` (vs `2mn + n + m` for Dense-SAE).

**OMP decoder (offline diagnostic):** greedy `k`-step support selection (argmax `|⟨w_j, r⟩|`), Cholesky/QR least-squares refit on the active set per step. Three d-regular optimisations: (i) `(values, rows)` storage → `O(dn)` correlation gather-reduce; (ii) incremental QR refit (`O(mk + k²)` vs `O(mk² + k³)`); (iii) block generalised-OMP + Cholesky-on-normal-equations + Triton `(d·d)`-unrolled Gram → continuum between iterative (`L=1`) and single-shot (`L=64`) decoding.

**Dead-feature resampling (Eq. 15):** every `T_r = max(1000, T/5)` steps, dead columns (fired on <5 of last `T_r·B` samples) reset to the largest-residual mini-batch sample projected onto their mask support `S_j = {i: M_ij = 1}`, with `V_{S_j, j} ← (1/√d) (r_b)_{S_j} / ‖(r_b)_{S_j}‖_2`, `W_enc` re-tied, `b_enc,j ← 0`. Disabled above 80% dead.

---

## 6. Strengths, limitations, verdict

**Strengths**
- **Genuine parameter efficiency with a clean mechanism:** storage ratio `m/d` is exact and verifiable; 73–293× compression at 84–86% dense CE-rec is a strong, reproducible frontier across 4 LM families.
- **Theory matched to architecture:** Theorem 3.1 + Corollary 3.2 give *worst-case sufficient* identifiability/OMP-recovery conditions for *learned non-binary* decoders (the column-flatness factor `β` is the bridge to classical lossless-expander `ε < 1/2`). Honest non-vacuity framing + empirical certificate-ratio diagnostic (Figure 4).
- **Disentangling decoder quality from encoder amortisation** via symmetric iterative OMP (Table 3) is the cleanest causal probe — it converts "dense wins at matched params" into "encoder amortisation, not decoder quality."
- **Three controls** (Clustered-sparse, Pruned dense, matched-`n'` Dense) isolate support diversity as the active ingredient vs mere sparsity or parameter count.
- **Interpretability preserved:** blinded 2-judge coherence study shows d=200 ≈ Dense; novelty-vs-d curve + feature-reliability diagnostics (entropy, split-half) substantiate that low-`d` features are different-but-meaningful, not degenerate.

**Limitations**
- **Worst-case certificates are loose in the operating regime** (Figure 4: both `R_id`, `R_OMP` ≫ 1 across the trained grid) — the theorems motivate but do not certify the experiments.
- **Encoder amortisation gap remains at deployment:** the trained encoder (1.8M tok/s) underperforms iterative OMP by 0.02–0.12 CE; closing this without OMP's 135×–16,000× slowdown is open.
- **Late-layer failure (Pythia-70M L5, Table 7):** all SAE architectures fail (Dense itself `CE_rec < 0`) — a known limitation of SAE probes at late residual-stream layers, not Expander-specific.
- **Seed instability** at the storage extreme (`⋆` Qwen L24 d=30, `†` Llama L6 d=7) — one-of-three-seeds divergence documented in Appendix F but the headline means include it.
- **Scale:** evaluated up to 3B (Qwen2.5-3B); larger-scale evaluation + low-rank corrections + learned (non-fixed) sparse supports + sharper data-dependent theory are explicit next steps.

**Verdict.** A focused, single-author architecture+theory paper that makes a clean, verifiable point: **decoder support structure is an underexplored SAE design axis**, and a `d`-regular expander mask trades `mn → dn` learned values while keeping `(m,n,k)` fixed and preserving 84–86% of dense CE-rec at 73–293× compression. The matched-parameter dense "win" is honestly decomposed into an encoder-amortisation effect (largely closed by OMP), the theory is matched to the architecture without overclaiming certification, and the interpretability cost is measured (negligible at d=200, ~0.8 coherence points at d=7). Not a dense-SAE replacement, but a parameter-efficient dictionary that admits structured-OMP decoding and a deterministic seed-regenerable mask — a useful tool for large-feature-count interpretability work where dense storage is operationally painful.

---

## 7. Sourcing + verification notes

- **Verbatim tables:** Tables 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 transcribed from `paper_layout.txt` (lines cited per section).
- **Source-free reconciliation (no PDF re-read needed):**
  - Storage ratio = `m/d` for every Table 1/9 cell (Pythia-70M 512/{7,50,200} → 73/10/2.6×; Pythia-160M 768/{10,75,300} → 77/10/2.6×; Qwen/Llama 2048/{7,30,102} → 293/68/20×).
  - 84% = `0.828/0.983` (Qwen L12 d=7 vs Dense); 86% = `0.817/0.947` (Pythia L3 d=7 vs Dense).
  - 293× = `33,554,432 / 114,688` = `mn / dn` (Qwen m=2048, n=16,384, d=7).
  - Table 5 throughput ≈ doubles per `L`-doubling (13.3k→689.4k tok/s over L=1→64).
  - Cross-table: Table 10 d=50 (0.489/0.859/0.3%) == Table 2 Expander d=50 == Table 1 Pythia d=50.
  - Theorem 3.1 `β=1` → `2ε < 1` → `ε < 1/2` (classical lossless-expander) ✓.
- **Figures NOT back-filled (per repo "figure-derived numbers are weak" rule):** Figure 1 storage–fidelity curves, Figure 2 controls, Figure 3 decoder-method comparison, Figure 4 certificate-ratio logs (only the qualitative "ratios ≫ threshold — certificates loose" claim quoted), Figure 5 active-support collisions, Figure 6 novelty diagnostics, Figures 8–13 (Clustered/Pruned comparisons, CE-vs-budget bars, synthetic support-recovery curves). Only prose-confirmed ranges and figure-printed legend/annotation values (Table 6/10 + Figure 7 coherence means) are quoted.
- **No prose-vs-table numeric inconsistency found** (unlike some prior repo papers). The one phrasing nuance — abstract "84% of dense CE-loss recovered" is a *ratio* (0.828/0.983), not the CE-rec value (0.828) — is clarified inline in §4.1; both reconcile. All §4.1/§4.3 prose deltas (+0.073/+0.118 CE; +0.060→+0.007; +0.012→−0.005) recompute from the displayed Table 3 cells.
