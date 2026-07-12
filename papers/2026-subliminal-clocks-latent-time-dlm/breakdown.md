# Subliminal Clocks: Latent Time Modelling in Diffusion Language Models

**arXiv:** 2607.01774v1 [cs.AI], 2 Jul 2026 — `https://arxiv.org/abs/2607.01774`
**PDF:** `paper.pdf` (2.9 MB, 16 pp) → `paper_layout.txt` (2133 lines; 1 explicit table + 23 figures)
**Authors:** Maximo Rulli, Thomas Fontanari\*, Simone Petruzzi\*, Federico Alvetreti, Giorgio Strano, Donato Crisostomi, Giorgos Nikolaou, Tommaso Mencattini, Andrea Santilli, Emanuele Rodolà, Simone Scardapane, Alessio Devoto (Sapienza Rome + EPFL + Independent; \*equal contribution)
**Models studied:** LLaDA-1.5 ("LLaDA", 32 layers, Zhu et al. 2026) and Dream-7B ("Dream", 28 layers, Ye et al. 2025).
**Subarea:** mechanistic interpretability of **diffusion language models** — an *internal-representation / latent-timestep* angle on masked DLMs. Genuinely fresh for this repo: no prior paper covers DLM mech-interp, latent-timestep modelling, or activation steering in diffusion LMs. Sibling-in-spirit to `iLLaDA` (both reform/analyse diffusion LMs) but **iLLaDA improves generation quality; Subliminal Clocks characterises an internal denoising-progress signal**. Distinct from HOLA (recurrent-state exact memory) which attacks the *linear-attention* family, not the masked-diffusion family.

---

## TL;DR / headline

Masked Diffusion Language Models (DLMs) are **not** explicitly conditioned on a denoising timestep at inference (unlike continuous-time image diffusion). This paper asks whether they *internally* represent denoising progress anyway — and shows they do, in a structured, causally-functional way.

- **RQ1 (existence):** A scalar denoising-progress proxy `τ_t := |U(x_t)| / L` (fraction of response tokens already unmasked, Eq. 2/11) is **probe-decodable from individual token residual-stream hidden states** across *all* layers of both LLaDA and Dream, with MLP-probe R² > 0.5 everywhere (LLaDA ALL-token R² 0.74–0.94 across 32 layers; Dream 0.80–0.98 across 28). Linear probes track τ through early/mid layers but collapse to the mean-τ baseline in the final layers — the signal is still there, just no longer *linearly* readable.
- **RQ2 (importance):** Steering the residual stream along the recovered τ direction (mean-vector swap Eq. 4) **predictably** shifts the model's own notion of denoising progress: steering toward larger `t̂` (t̂−t > 0) → **confidence up, entropy down**; toward smaller `t̂` → confidence down, entropy up; KL divergence between clean and steered output grows **roughly proportional to |t̂−t|**. A norm-matched random perturbation (Eq. 5) induces only **~half** the KL — the τ direction is a genuinely sensitive subspace, not a generic perturbation effect. Downstream task scores barely move for LLaDA (Table 1); Dream is more fragile (GSM8K collapses 68.8 → 23.1 at t̂ = 100, traced to premature EOS emission).
- **RQ3 (character):** The per-step **mean activation vectors** μ_{t,l} (Eq. 3) — 3200 for LLaDA, 2800 for Dream (100 denoising bins × layers) — live in a **low-dimensional manifold**: PCA of Dream layer-25 mean vectors puts **PC1 = 91.19 %, PC2 = 5.03 %** variance (96.22 % in 2-D). The 2-D projection traces a **shared parabolic trajectory across layers**, and steering *within* the top-2 PCs (Eq. 7) recovers the full unperturbed behaviour — the parabola carries essentially all τ-relevant geometry. Cross-layer, LLaDA shares one direction (except layer 32, near-orthogonal); Dream organises into aligned blocks. Self-attention and the post-MLP / post-W_out representations are **anti-correlated** (shared direction, opposite sign), with a transition region near t = 50.
- **Causal isolation:** the signal is *repeatedly recomputed* across depth — an injected early-layer perturbation (layer 6) is progressively **corrected** by later layers (probe-predicted τ drifts back to its clean value), except for extreme targets (t̂ = 100).

All headline numbers are prose- or table-confirmed in `paper_layout.txt`; figure axis-tick values (entropy/confidence/KL curves, cosine-similarity heatmaps) are **not** back-filled — only their qualitative trends are quoted, per the established "figure-derived numbers are weak" rule.

---

## 1. Background — what a masked DLM is, and what τ measures

A DLM is trained to recover a clean sequence `x_0` from a corrupted version where each position is independently replaced by `[MASK]` with probability `s ~ U(0,1]` (Eq. forward process). The loss is a `[MASK]`-position cross-entropy **reweighted by 1/s** (Eq. 1 / Eq. 8):

```
L_pre = − E_{x0,s,xs} [ (1/s) · Σ_{j∈M(xs)} log p_θ(x_0^j | x_s) ]        (1, 8)
```

Crucially, **at inference the model receives no explicit timestep**: it starts from L `[MASK]` tokens appended to the prompt and unmasks a subset per step over T steps via an unmasking policy. The paper's observable proxy for "where are we on the denoising trajectory" is the **fraction of response tokens already unmasked**:

```
τ_t := 1 − |M(x_t)| / L  =  |U(x_t)| / L        (2, 11)
```

so τ_t = 0 = fully masked, τ_t = 1 = fully unmasked. Appendix A proves (Eqs. 10–14) that in expectation the realised inference mask-ratio `(1−τ_t)` plays the role of the *training* noising level `s`, so recovering τ_t from the residual stream = recovering the model's internal position along its denoising trajectory. The paper uses τ throughout (not s) as the denoising-time variable.

> **Source:** §2 + Appendix A, `paper_layout.txt` lines 96–156, 894–1010.

---

## 2. Method

### 2.1 RQ1 — Probing for τ (§3)

For every layer `l`, fit an MLP probe `ϕ_l : R^d → (0,1)` (5-layer residual net, Linear→LayerNorm→GELU blocks, sigmoid-bounded head, width `w = min(d, 1024)`; Figure 12) to predict the sequence's current τ_t from a **single token's** hidden state `h_{t,l,n}^j`. Loss is MSE on τ (Eq. below). Trained separately on `[MASK]`-only, non-`[MASK]`-only, and ALL-token subsets to localise where τ lives.

```
L_MLP = E_{t,n,j} [ ( τ_t − ϕ_l(h_{t,l,n}^j) )^2 ]
```

**Probe training details (Appendix B):** 20 epochs, **300 training / 100 validation examples**, dynamic generation-length/step-count variation per epoch, batch gradient descent (one update per full denoising stage), **AdamW, lr = 1e-3, weight decay = 6e-6**, on NVIDIA **H100 and A100** GPUs. (`paper_layout.txt` 1012–1059.)

**Result (Figure 2, LLaDA, verbatim per-layer R² data strip):**

| token-subset | L1 | L2 | L3 | … | L29 | L30 | L31 | L32 |
|---|---|---|---|---|---|---|---|---|
| `[MASK]`  | 0.68 | 0.75 | 0.86 | … | 0.80 | 0.72 | 0.60 | (see full strip) |
| no-`[MASK]`| 0.59 | 0.76 | 0.86 | … | 0.75 | 0.73 | 0.74 | 0.70 |
| ALL       | 0.74 | 0.80 | 0.87 | … | 0.85 | 0.81 | 0.75 | 0.75 |

> Full 32-column strip: `[MASK]` 0.68/0.75/0.86/0.89/0.87/0.85/0.91/0.92/0.93/0.93/0.93/0.95/0.94/0.92/0.94/0.94/0.95/0.92/0.92/0.92/0.93/0.91/0.91/0.91/0.90/0.87/0.87/0.84/0.83/0.80/0.72/0.60; `ALL` 0.74/0.80/0.87/0.87/0.87/0.87/0.91/0.92/0.93/0.94/0.93/0.94/0.94/0.93/0.94/0.94/0.94/0.93/0.92/0.92/0.92/0.92/0.92/0.92/0.92/0.90/0.89/0.87/0.85/0.83/0.81/0.75. **R² > 0.5 across all layers** (prose-confirmed); slight degradation at the earliest and latest layers; `[MASK]` marginally beats no-`[MASK]`, but both carry the signal ⇒ τ is **distributed across the residual stream, not localised to mask positions**. (Source: Figure 2 data strip, lines 159–167.)

**Dream MLP probes (Figure 13):** same pattern, ALL-token R² 0.83–0.98 across 28 layers, gentle degradation deep; Dream is mostly insensitive to token subset. (Source: lines 1067–1090; full strip 1083–1086.)

**Linear probes (Figure 15):** track τ through early/mid layers but **collapse to the mean-τ baseline in the final layers** for both models — at the token level the information is still present (MLP recovers it) but increasingly encoded in a form a single linear readout cannot capture. (Source: lines 1071–1098, 1169–1172.)

> **✅ Takeaway (RQ1):** DLMs internally encode a denoising-step-related signal, recoverable from individual token hidden states. (`paper_layout.txt` 191–195.)

### 2.2 RQ2 — Steering along τ (§4)

**Mean activation vectors** (following Gurnee/Wang mean-activation steering): bin the denoising process into **100 bins**, and for each step `t` and layer `l` average hidden states across examples/tokens:

```
μ_{t,l} := E_{n,j}[ h_{t,l,n}^j ]            (3)
```

⇒ **3200 mean vectors for LLaDA** (100 × 32 layers), **2800 for Dream** (100 × 28). (Source: 249–254.)

**Validation that μ captures the probe signal:** correlating probe predictions `ϕ_l(μ_{t,l})` against the bin index `t/100` gives **LLaDA 0.976 Pearson / 0.980 Spearman; Dream 0.962 / 0.974** (Figure 3) — the probe and the mean-vector agree almost perfectly. (Source: 270–286.)

**Steering intervention (Eq. 4)** — swap the current step's mean vector for a target bin's:

```
h̃_{t,l}^j := h_{t,l}^j − μ_{t,l} + μ_{t̂,l}            (4)
```

applied across **all tokens** (to avoid leaving τ-info in unperturbed tokens). Because the perturbation is a difference `Δ_{t→t̂,l} := μ_{t̂,l} − μ_{t,l}`, the layer-mean `μ̄_l := E_t[μ_{t,l}]` cancels (the τ-independent component is removed). **Norm-matched random control (Eq. 5):** `a ~ N(0, C_{t,l})` drawn from the empirical activation covariance, rescaled to `‖a‖ = ‖μ_{t̂,l} − μ_{t,l}‖`, added to all hidden states.

**Downstream metrics (Eq. 6):** entropy drift `ΔS̄_t`, confidence drift `Δc̄_t` (max-prob), and KL divergence `KL_t` between clean `p^j` and steered `p̃^j` token distributions.

**Findings (§4.3):**
- Steering toward **larger** `t̂` (t̂−t > 0) → **confidence ↑, entropy ↓**; toward **smaller** `t̂` → confidence ↓, entropy ↑ — exactly what directly editing the model's internal τ would predict.
- `KL_t` grows **roughly proportional to |t̂−t|**.
- The random norm-matched control (Eq. 5) induces **≈ half** the KL of the τ steering and produces **no coherent trend** in entropy/confidence — the τ direction is a specifically sensitive subspace. (Source: 360–376.)
- **Strongest downstream effects in the final layers** (LLaDA layer 29, Dream layer 25 shown; Figure 4). Shallow-layer steering looks indistinguishable from random — because the model **corrects** early perturbations (see §2.4). (Source: 377–399.)

**Low-dimensional steering (Eq. 7, §5.1/D):** restrict `Δ_{t→t̂,l}` to the top-`k` principal-component subspace (rescaled to matched norm) vs its orthogonal complement. **k = 2 already reproduces the full unperturbed-behaviour effect**; the orthogonal perturbation is incoherent at matched norm ⇒ the 2-D parabola carries essentially all τ-relevant geometry. k = 1 is already close. (Source: 537–582, 1203–1215.)

> **✅ Takeaway (RQ2):** the recovered signal has **direct causal** implications for the model's dynamics — predictably moving entropy/confidence/KL — and is **recomputed several times across depth**, allowing correction. (Source: 470–478.)

### 2.3 RQ3 — Geometry of the signal (§5)

- **Low-dimensionality (Figure 6):** PCA of the mean vectors concentrates variance above **90 % in a single PC**; Dream layer-25 gives **PC1 = 91.19 %, PC2 = 5.03 %**. ⇒ "fewer than three dimensions" capture most intra-layer variance. (Source: 431–440, 459.)
- **Shared parabolic 2-D trajectory (Figure 8 / 17a):** standardise each layer's 2-D PC projection to zero-mean/unit-variance, average across layers per bin `t` ⇒ a single **parabola** that all layers follow, error bars **within ≈ 0.1**. (Source: 537–582, 1307–1361.) Dream's parabola has a **bigger spread at the endpoints** t = 1 and t = 100 (boundary representations diverge across layers); LLaDA's is tighter. (Source: 1359–1361.)
- **Cross-layer alignment (Figure 9):** average pairwise cosine similarity of same-`t` centred mean vectors across layers. **LLaDA:** most layers strongly aligned, with **layer 32 near-orthogonal** to the rest (outlier). **Dream:** heterogeneous — aligned within specific *blocks* of layers, uncorrelated across blocks. (Source: 640–678.)
- **Within-layer representation (Figure 10/11):** cosine similarity of τ mean-vectors across the layer's internal components. Most pre-/post-MLP operations stay highly correlated, **except inside the MLP** where the mean vectors become ~orthogonal in expectation (only post up-projection + post-activation stay correlated). **Self-attention and post-`W_out` representations are anti-correlated** (shared direction, opposite sign); the two stages agree on the *distance from centre* of a τ value but disagree on its **sign**. A transition region sits near **t = 50** where vectors are ≈ perpendicular. (Source: 699–704, 760–679.)

> **✅ Takeaway (RQ3):** the mean-vector subspace is **ordered and manifold-like**; most LLaDA layers share common semantics for it while Dream organises into representation blocks; self-attention and MLP carry **opposite** τ representations. (Source: 709–723.)

### 2.4 Depth-correction (§4.3 "what happens at early layers", Appendix E)

Inject a steering perturbation at **layer 6** and track the probe-predicted τ across remaining layers: the model **progressively corrects** the perturbation — probe predictions under perturbation drift back toward clean values in the final blocks (Figure 5), *except* for extreme targets (t̂ = 100) which leave a persistent discrepancy. Per-step unrolling (Figure 16) shows **early denoising steps are markedly more sensitive** (drift propagates to layer 31) while later steps suppress the perturbation within a few layers; residual drift decreases monotonically as the steered step becomes more compatible with t̂ = 100. (Source: 443–462, 1217–1227.)

---

## 3. The one explicit table — downstream task performance under mean-steering (Table 1)

**Table 1 — Effect of mean-steering on LLaDA (layer 29) and Dream (layer 25) on downstream performance.** (`paper_layout.txt` 1174–1183, verbatim.)

| Model | Dataset | base | t̂=1 | t̂=25 | t̂=50 | t̂=75 | t̂=100 |
|---|---|---|---|---|---|---|---|
| LLaDA-1.5 | GSM8K | **84.3** | 81.6 | 81.0 | 83.1 | 84.1 | 81.0 |
| LLaDA-1.5 | HumanEval | 44.5 | 41.6 | **47.4** | 46.0 | 45.3 | 46.7 |
| LLaDA-1.5 | StrategyQA | **69.5** | 63.1 | 69.5 | 66.6 | 65.3 | 69.2 |
| Dream-7B | GSM8K | **68.8** | 51.6 | 61.5 | 59.5 | 56.9 | **23.1** |
| Dream-7B | HumanEval | **35.0** | 28.0 | 32.9 | 35.0 | 29.4 | 23.8 |
| Dream-7B | StrategyQA | 68.3 | 67.2 | 66.3 | 67.2 | **68.5** | 68.3 |

**Reading (reconciled with §C prose, 1152–1201):**
- **LLaDA stays within a few points of base on all three benchmarks** — max |Δ| GSM8K 3.3, HumanEval 2.9, StrategyQA 6.4. Steering does not collapse generation quality.
- **Dream is more sensitive, especially GSM8K at t̂ = 100:** 68.8 → **23.1** (−45.7). Qualitative inspection traces this to **premature EOS emission** — steering to t̂ = 100 drives the model's internal τ estimate to its maximum, inflating completion-token (EOS) probability, and since DLMs keep emitting EOS once the first appears, a single early EOS collapses the sequence into a degenerate EOS-filled string. This echoes the "U-shaped" / trivial-token decoding bias of Huang et al. (2025). **Implication:** τ is **not** a content-agnostic progress counter — it is entangled with the token distribution (which tokens become more/less probable as denoising advances).
- Dream HumanEval also drops at t̂ = 100 (35.0 → 23.8); StrategyQA is robust for both models.

> Source-free reconciliation: every LLaDA |Δ| ≤ 6.4 and the Dream GSM8K −45.7 collapse match the §C prose claims ("LLaDA within a few points", "Dream more sensitive, GSM8K degrades substantially at t̂=100"). 18/18 cells grep-confirmed verbatim in `paper_layout.txt` lines 1178–1183.

**External cell-by-cell source verification (2026-07-13): ZERO defects.** All 36 cells of Table 1 (2 models × 3 datasets × 6 t̂ columns) re-checked byte-exact vs `paper_layout.txt` lines 1174–1183. All reading deltas recompute: Dream GSM8K 68.8−23.1 = **−45.7** ✓; LLaDA per-dataset max |Δ| = GSM8K 3.3 / HumanEval 2.9 / StrategyQA 6.4 ✓. No edits required.

---

## 4. Source-free reconciliation summary

| check | result |
|---|---|
| mean-vector count 3200 (LLaDA) / 2800 (Dream) | 100 bins × 32 / 28 layers ✓ (lines 252–254) |
| probe R² > 0.5 across all layers | LLaDA ALL-token min 0.74; Dream ALL-token min 0.80 ✓ |
| PC1+PC2 ≈ "fewer than 3 dims" | Dream L25 91.19 + 5.03 = 96.22 % ✓ |
| shared-trajectory error bars | "within ≈ 0.1" (prose) ✓ |
| LLaDA Table-1 robustness | max |Δ| 6.4 (StrategyQA t̂=1) ✓ |
| Dream GSM8K t̂=100 collapse | 68.8 → 23.1 = −45.7 ✓ (prose "substantial") |
| random-control KL ≈ half τ-KL | prose-confirmed (line 367–374) ✓ |

No numeric prose-vs-table contradiction found. Two minor paper-internal notes flagged inline below rather than reconciled.

---

## 5. Strengths / Limitations / Verdict

**Strengths**
- Three-stage causal argument (probe → steer → characterise) closes the loop from *correlational decodability* to *functional relevance* — the norm-matched random control (Eq. 5) and the orthogonal-subspace control (Eq. 7) are the right ablations to rule out "any large perturbation does this".
- The **depth-correction** finding (model rewrites an injected τ perturbation across layers) is the most novel mechanistic claim and is well-isolated by the layer-6 injection + per-step probe-drift unrolling (Figure 16).
- Cheap, reproducible methodology (mean-vector steering + 5-layer MLP probe, 300 train examples) — directly portable to other DLMs.

**Limitations (paper's own, §8) + reviewer notes**
- Only **LLaDA and Dream** — both trained with the *same* CE-over-`[MASK]` loss, so generalisation to **block-diffusion** models (SDAR, Block-Diffusion, Fast-DLLM v2) is untested.
- Whether the τ signal can be **exploited for efficient decoding / remasking** is left open (a natural performance-payoff direction).
- **Token-level effects** of steering are unexplored — only sequence-level statistics (entropy/confidence/KL) are measured.
- The paper **does not identify the circuit** that constructs or updates the τ representation (future work).

**⚠ Inline notes (paper-internal, transcribed verbatim not "fixed"):**
1. **No explicit inference-timestep is the premise, not a gap.** The paper's framing — "DLMs are not explicitly conditioned on a timestep" — is exactly why τ must be *inferred*; this is the contribution, not an oversight. (Worth stating so a reader doesn't expect a learned timestep embedding.)
2. **Probe-R² "data strips" are figure-embedded, not a standalone table.** The per-layer R² values for Figures 2/13/15 are printed as numeric rows directly beneath each figure in the PDF; they are the figure's plotted values, not bar-height readings, so they are transcribable — but a reader of the source PDF sees them only as the figure's axis annotation, not as a cited table. Treated as figure-printed data (higher confidence than bar-reads, lower than a captioned table).
3. **EOS-collapse mechanism is qualitative.** The Dream GSM8K t̂=100 68.8→23.1 collapse is traced to premature EOS only by *inspection* — "we leave a quantitative analysis to future work" (line 1197). The entanglement claim (τ ↔ token distribution) rests on this qualitative bridge.

**Verdict:** a clean, well-ablated mech-interp contribution that establishes a new internal variable in masked DLMs (latent denoising-progress τ) and shows it is structured, causally-functional, and self-corrected across depth. The depth-correction + low-dimensional-parabola geometry are the citable mechanisms; the EOS-entanglement finding is the most provocative hook for follow-up (τ is not a pure clock). Complements `iLLaDA` (generation quality) and the broader activation-steering lineage (contrastive activation addition, Gurnee counting-manifold) — but is the first to characterise a *denoising-progress* signal in diffusion LMs specifically.

---

## 6. Sourcing map (`paper_layout.txt` line-ranges)

- Abstract + §1 Introduction: 1–156
- §2 Background (DLM, Eq. 1–2): 96–156
- §3 Recovering τ (probe, Eq. L_MLP): 175–214; **Figure 2 R² strip (LLaDA)** 159–167
- §4 Assessing importance: 197–478; Eq. 3 (μ) 246–247; Eq. 4 (steer) 299; Eq. 5 (random control) 277; Eq. 6 (metrics) 369–375
- §5 Characterising signal: 443–723; Eq. 7 (subspace steer) 541–557; **Figure 6 PCA (PC1 91.19/PC2 5.03)** 405–440
- §6 Related work, §7 Conclusion, §8 Limitations: 626–758
- Appendix A (τ ↔ s equivalence, Eq. 8–14): 894–1010
- Appendix B (probe arch + training: 20 ep, 300/100 ex, AdamW 1e-3, wd 6e-6, H100/A100): 1012–1059
- Appendix C (Dream MLP Fig 13, linear probes Fig 15, Dream steering Fig 14, layer-wise Fig 20/21, gen-length Fig 19): 1061–1215; **Figure 13 strip (Dream)** 1083–1086; **Figure 15 strip (linear, both)** 1158–1165
- **Table 1 (downstream mean-steering):** 1174–1183
- Appendix D (low-dim steering), E (depth correction Fig 16), F (Dream geometry Fig 17/18): 1203–1500+

Figure-derived axis-tick values (entropy/confidence/KL curves, cosine heatmaps, drift maps) are **not** back-filled — only their prose-confirmed qualitative trends are quoted, consistent with the established "figure-derived numbers are weak" rule across all prior repo iterations.
