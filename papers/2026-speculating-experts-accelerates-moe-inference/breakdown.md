# Breakdown — Speculating Experts Accelerates Inference for Mixture-of-Experts

> **Paper:** Speculating Experts Accelerates Inference for Mixture-of-Experts
> **Authors:** Vivan Madan\*, Prajwal Singhania\* (equal contribution), Abhinav Bhatele, Tom Goldstein, Ashwinee Panda (UMD + TogetherAI)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2603.19289 (v1, 9 Mar 2026; preprint 23 Mar 2026)
> **Code:** https://github.com/axonn-ai/yalis/tree/offload_prefetch (YALIS inference engine, `offload_prefetch` branch)

> Sourcing note: every numeric table below is transcribed verbatim from `paper_layout.txt` (`pdftotext -layout` of the arXiv PDF, 694 lines, 3 explicit tables + 11 figures). All 12 Table-1 `Average` cells reproduce source-free from their 6 benchmark cells (e.g. Qwen3-30B-A3B baseline (0.939+0.762+0.950+0.800+0.733+0.719)/6 = 0.8172 → **0.817** ✓), confirming verbatim transcription without re-reading the PDF. TPOT bar values from Figures 7–8 are figure-derived and the per-point series assignment is ambiguous in the layout dump — only the prose-confirmed TPOT *ranges* and *percentage shares* are stated; no per-point bar values are back-filled (consistent with the established "figure-derived sections are weak" rule). One paper-internal prose-vs-table inconsistency is flagged inline with ⚠ rather than silently "reconciled": the §6.3 "hybrid recovers ~37% of the accuracy gap on GSM8k" claim does not match the Table-1 GSM8k cells (Hybrid-PF GSM8k 0.946 is already within 0.004 of the 0.950 baseline → ~99% gap recovered, not 37%).

---

## 1. Problem & Motivation

- **Problem.** Mixture-of-Experts (MoE) LLMs scale capacity via sparse activations (a learned router selects Top-K experts per token), making them dominant across Qwen3-MoE, GPT-OSS, GLM 4.7. But in **memory-constrained inference** (single consumer/datacenter GPU), most expert weights must be offloaded to CPU RAM and transferred on demand — and those CPU→GPU transfers dominate decode latency. For **Qwen3-30B-A3B on an A6000, transfers are 84–88% of time-per-output-token (TPOT)**; compute is only 8–13%. Inference is I/O-bound, not compute-bound.
- **Why prior work falls short.** Existing expert-prefetch methods (Prescope, Duoserve-MoE, Eliseev's LRU + next-layer-gate, Pre-gated MoE) treat predicted experts as **cache hints**: on a misprediction they re-fetch the true router-selected expert as a cache miss, which serializes a copy back onto the critical path and caps the achievable compute/transfer overlap. They were also validated mostly on MoEs with small expert pools, leaving unclear whether prediction scales to modern 32–128-expert architectures.
- **This paper's bet.** (1) Internal layer-`l` representations reliably predict layer-`l+1` expert selection even for large expert pools; (2) you can **execute the speculated experts** (not just cache them) and usually preserve task accuracy, eliminating the re-fetch and maximizing overlap; (3) where router-based speculation drifts (a few high-drift layers), a tiny neural estimator fixes the hit rate.

## 2. Contribution Summary

1. **Parameter-free prefetching.** Identify internal representations (the *quasi-hidden state*) that predict next-layer routing across modern large-pool MoEs with no learned predictor and no model fine-tuning — works directly on pretrained weights.
2. **Speculative execution preserves accuracy.** Execute prefetched experts (with their predicted routing weights) instead of falling back to cache-miss re-fetch; downstream task accuracy is largely maintained.
3. **Optimized inference implementation.** Integrated into the open-source **YALIS** engine → **5–14% TPOT reduction** over on-demand CPU expert loading across hardware/model configs (up to **14%**, with **9–14%** on Qwen3-30B-A3B/A6000 and **5–8%** on stronger A100/GH200 GPUs).
4. **Lightweight neural estimators.** For architectures where pure router-based speculation degrades (large early-layer drift), a small FFN estimator (4M–45M params, trained on a few million tokens via KL distillation to the true router logits) substantially lifts expert hit rates on the drifting layers.

## 3. Method — Signals, Speculation, and the Estimator

### 3.1 Default vector and quasi-hidden state

- **Default vector `d_{l,e}`** (from Panda et al. 2025): the average activation associated with expert `e` at layer `l`, computed offline by aggregating observed activations for each selected expert during inference.
- **Layer-level default `d_l`** = Σ_{e∈E_l} g_{l,e} · d_{l,e} — the gating-weight-weighted combination of the selected experts' default vectors. Captures the *expert-conditioned typical contribution* of the MoE block to the residual stream.
- **Quasi-hidden state `q_l`** = `LN_{l+1}(d_l + r_l)`, where `r_l` is the post-attention residual and `LN_{l+1}` is the normalization applied before routing at layer `l+1`. This is an *approximation to the layer-(l+1) router input*, constructible from layer-`l` quantities alone — so it can drive next-layer expert prefetch one layer ahead of execution.
- **Why it helps (Figure 3).** `q_l` has higher average cosine similarity to the true `s_{l+1}` than the raw normalized residual `s_l` does — the default vector adds a useful expert-conditioned bias that approximates inter-layer drift. For **GPT-OSS** this lift is large (drift is prominent in early/late layers); for **Qwen3-30B-A3B** the lift is negligible because drift beyond the first two layers is already small.

### 3.2 Three prefetch strategies

| Strategy | How next-layer experts are chosen | When used |
|---|---|---|
| **Router-PF** (parameter-free) | Apply layer `l+1`'s router to the quasi-hidden state `q_l` | Default; works on all pretrained MoEs, no training |
| **Est-PF** | Replace router prediction at *all* layers with a trained neural estimator | Where router-based hit rates are poor across the network |
| **Hybrid-PF** | Use the estimator *only* on low-hit-rate (high-drift) layers; quasi-hidden-state prediction elsewhere | Recovers accuracy at minimal overhead — the practical default for hard architectures |

Speculative execution = the prefetched expert indices and predicted gating weights are **executed directly**; the true router's ground-truth experts are **not** re-fetched on mismatch (the key difference from cache-hint prefetchers).

### 3.3 Decode-phase scheduling (Algorithm 1, Figure 6)

- Layer 0 is a **cold start**: no prefetched experts exist, so a synchronous blocking CPU→GPU copy of the routed experts happens after Top-K selection.
- For layers `l ≥ 1`: reuse the already-prefetched expert indices/weights (`E_l, G_l`), compute the *next* layer's predictions (`E_{l+1}`), then **wait-and-prefetch** — wait for the current layer's copy to land, issue the async next-layer copy, and yield so the next-layer transfer overlaps with the current layer's MoE compute.
- **Double buffering** alternates two GPU expert buffers across layers, enabling compute–copy overlap with no extra synchronization. Only **expert weights** are offloaded; attention + router params stay GPU-resident (small footprint, see Table 2). Offloaded weights live in **pinned CPU memory** for faster PCIe transfer.

### 3.4 Speedup bound (Equation 1)

- On-demand TPOT: `T_decode^ond = Σ_l [t_attn,l + t_gate+topk,l + t_copy(E_l) + t_expert,l]` — copy is on the critical path.
- Prefetching overlaps next-layer copy with current-layer compute; the **TPOT improvement is**:
  > `ΔT = Σ_l min( t_copy(E_l), t_compute,l )`, where `t_compute,l = t_attn,l + t_gate+topk,l + t_expert,l`
- **Maximum achievable speedup is 2×**, attained only when compute and copy times are equal. In practice GPUs are compute-optimized and PCIe is relatively slow, so **copy dominates (`t_copy ≫ t_compute`)** and the realized gain is well below 2× — but still substantial (5–14%).

### 3.5 Neural expert prefetcher (Section 6, Appendix A / Figure 11)

- **Architecture** (shallow FFN to avoid serial bottleneck): `q_l ∈ R^d` → linear projection `A ∈ R^{d×d/m}` → **+ learned positional embedding** (encodes layer-specific routing) → FFN `B ∈ R^{d/m·n × d/m}` → SiLU → `C ∈ R^{d/m × d/m·n}` with a **residual skip** over the MLP → LayerNorm → linear head `W_head ∈ R^{d/m × E}` producing predicted router logits `ô ∈ R^E` for layer `l+1`. (`m, n > 1` are width-reduction factors.)
- **Training:** distillation minimizing **KL divergence** between estimator logits and the ground-truth router logits. Parameter counts: **4M (GPT-OSS-20B), 45M (GPT-OSS-120B), 17M (Qwen3-30B-A3B)**.
- **Hit-rate behavior (Figures 9–10, prose values):**
  - Qwen3-30B-A3B estimator reaches **~90% average hit rate after 4M training tokens**.
  - GPT-OSS-120B and GPT-OSS-20B estimators reach **83% and 88% respectively after 5M tokens**; further training gives diminishing returns.
  - For Qwen3-30B-A3B the estimator lifts early-layer prediction by **up to ~25% over the fixed router** (Figure 10 bar-read — figure-derived).
  - Beyond the first two layers, Qwen3-30B-A3B router-based **recall@k ≈ 90% on average**; early layers and the final layer have lower hit rates.

---

## 4. Downstream Accuracy — Table 1 (verbatim)

Benchmark accuracy ± standard error across 6 tasks (HumanEval, MBPP+, GSM8k, AIME24, AIME25, StrategyQA) + Average. **Bold = best, underline = second-best within each model family.** 4 prefetch variants per family: baseline (no prefetch), Router-PF, Est-PF, Hybrid-PF.

| Model / variant | HumanEval | MBPP+ | GSM8k | AIME24 | AIME25 | StrategyQA | **Average** |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 0.939±0.019 | 0.762±0.022 | 0.950±0.006 | 0.800±0.073 | 0.733±0.081 | 0.719±0.017 | **0.817** |
| Qwen3-30B-A3B + Router-PF | 0.860±0.027 | 0.659±0.024 | 0.576±0.014 | 0.467±0.091 | 0.600±0.089 | 0.683±0.018 | 0.641 |
| Qwen3-30B-A3B + Est-PF | 0.915±0.022 | 0.741±0.023 | 0.918±0.008 | 0.667±0.086 | 0.567±0.090 | 0.691±0.018 | 0.750 |
| Qwen3-30B-A3B + Hybrid-PF | 0.909±0.023 | 0.762±0.022 | 0.946±0.006 | 0.700±0.084 | 0.600±0.089 | 0.681±0.018 | 0.766 |
| GPT-OSS-20B | 0.970±0.013 | 0.794±0.021 | 0.942±0.006 | 0.667±0.086 | 0.667±0.086 | 0.753±0.016 | **0.799** |
| GPT-OSS-20B + Router-PF | 0.933±0.020 | 0.804±0.020 | 0.929±0.007 | 0.667±0.086 | 0.600±0.089 | 0.738±0.017 | 0.779 |
| GPT-OSS-20B + Est-PF | 0.884±0.025 | 0.751±0.022 | 0.933±0.007 | 0.633±0.088 | 0.533±0.091 | 0.726±0.017 | 0.743 |
| GPT-OSS-20B + Hybrid-PF | 0.896±0.024 | 0.788±0.021 | 0.936±0.007 | 0.667±0.086 | 0.467±0.091 | 0.725±0.017 | 0.747 |
| GPT-OSS-120B | 0.970±0.013 | 0.815±0.020 | 0.955±0.006 | 0.800±0.073 | 0.767±0.077 | 0.789±0.016 | 0.849 |
| GPT-OSS-120B + Router-PF | 0.963±0.015 | 0.812±0.020 | 0.958±0.006 | 0.833±0.068 | 0.800±0.073 | 0.776±0.016 | **0.857** |
| GPT-OSS-120B + Est-PF | 0.951±0.017 | 0.804±0.020 | 0.949±0.006 | 0.700±0.084 | 0.700±0.084 | 0.760±0.016 | 0.811 |
| GPT-OSS-120B + Hybrid-PF | 0.927±0.020 | 0.802±0.021 | 0.945±0.006 | 0.800±0.073 | 0.800±0.073 | 0.769±0.016 | 0.841 |

> *Sourcing:* `paper_layout.txt` lines 207–222, byte-exact. All 12 `Average` cells reconcile source-free (see headnote). Note **Table 1 evaluates {Qwen3-30B-A3B, GPT-OSS-20B, GPT-OSS-120B}** for accuracy, whereas the performance Tables 2–3 + Figures 7–8 evaluate **{Qwen3-30B-A3B, GLM-4.7-Flash, GPT-OSS-120B, Qwen3-235B-A22B}** — different model sets for accuracy vs throughput.

### 4.1 Takeaways from Table 1

- **GPT-OSS architectures tolerate speculative execution almost for free.** GPT-OSS-120B + Router-PF averages **0.857 vs 0.849 baseline** — prefetching is nominally *better* (within standard error) and wins 4 of 7 per-benchmark columns (MBPP−, GSM8k, AIME24, AIME25, Average). GPT-OSS-20B + Router-PF (0.779) loses only 2.0pp average. The quasi-hidden state is a strong routing predictor when inter-layer drift is prominent.
- **Qwen3-30B-A3B is the hard case** (0.817 → 0.641 under Router-PF, −17.6pp average): early-layer representational drift corrupts the speculative experts, and the damage concentrates in **math-heavy tasks** (GSM8k 0.950→0.576, AIME24 0.800→0.467) while common-sense/coding (StrategyQA, HumanEval) degrade less.
- **Hybrid-PF is the rescue.** Applying the estimator only on low-hit-rate layers recovers Qwen3-30B-A3B average to **0.766** (−5.1pp from baseline, vs −17.6pp for pure Router-PF) and GSM8k to **0.946** (within 0.004 of baseline). Est-PF (0.750) sits between. This is the paper's main accuracy-preserving contribution.
- ⚠ **Paper-internal inconsistency (§6.3 prose vs Table 1):** the body claims *"on GSM8k … the hybrid approach recovers approximately **37%** of the accuracy gap to Router-PF."* But Table 1 gives Qwen3-30B-A3B GSM8k = 0.950 (baseline), 0.576 (Router-PF), **0.946 (Hybrid-PF)** → Hybrid-PF recovers (0.946−0.576)/(0.950−0.576) = **~99%** of the gap, not 37%. The 37% figure is not derivable from any Table-1 column (Average-gap recovery is ~71%; Est-PF GSM8k recovery ~91%). Treat the "37%" prose claim as unverified — it likely refers to an intermediate/different measurement not reflected in Table 1.

---

## 5. Performance — TPOT Reduction

### 5.1 Setup (Tables 2 & 3)

**Table 2 — MoE model configuration (bf16)** (`L` = layers, `E` = experts/MoE layer, `H` = hidden, `H_MoE` = expert hidden, `M_experts` / `M_other` = memory footprint). Verbatim, `paper_layout.txt` lines 408–414.

| Model | L | E | H | H_MoE | M_experts | M_other |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 48 | 128 | 2048 | 768 | ~54 GB | ~3 GB |
| GLM-4.7-Flash | 47 | 64 | 2048 | 1536 | ~53 GB | ~3 GB |
| GPT-OSS-120B | 24 | 32 | 2880 | 2880 | ~213 GB | ~4 GB |
| Qwen3-235B-A22B | 94 | 128 | 4096 | 1536 | ~230 GB | ~14 GB |

> Note: GLM-4.7-Flash and GPT-OSS-120B use **random expert weights + simplified RoPE attention** in the throughput experiments (YALIS limitation); GLM-4.7-Flash uses routed experts only (shared expert dropped).

**Table 3 — Hardware configurations (verbatim, lines 359–364).** In all rows, model parameters exceed GPU HBM, forcing CPU offload.

| GPU | HBM | CPU DRAM | CPU↔GPU Link | Models evaluated |
|---|---|---|---|---|
| A6000 | 48 GB | 128 GB | PCIe 4.0 | Qwen3-30B-A3B, GLM-4.7-Flash |
| A100 | 80 GB | 256 GB | PCIe 4.0 | GPT-OSS-120B |
| GH200 | 96 GB | 480 GB | NVLink C2C | Qwen3-235B-A22B |

> Workload/metrics: batch size 1, prompt length ∈ {1024, 4096, 16384, 65536}, generate 32 tokens, report **TPOT** averaged over 3 generations × 3 trials with error bars. PyTorch 2.9, FlashAttention backend, bf16; traces via Nsight Systems + Pipit.

### 5.2 TPOT breakdown (Figure 7) — the I/O bottleneck, quantified

- For **Qwen3-30B-A3B (A6000)**, **copy ≈ 84–88% of TPOT**; **compute ≈ 8–13%** (the upper bound on any prefetch gain per Eq. 1); the remainder is idle/launch/sync overhead. This copy-dominant split holds across all studied models.
- Copy time is ~constant with context length; compute grows with context → **overlap pays more at longer sequences**, so prefetch gains widen at 65536 vs 1024.
- (Figure 7 gives the per-bar Copy/Idle/Compute decomposition; the bar values are figure-derived and the layout series assignment is scrambled, so per-point ms values are not back-filled here. The prose-confirmed 84–88% copy share reconciles the figure totals — e.g. for the long-TPOT bars, copy/total ≈ 267/304 = 87.9% and 265/305 = 86.9%.)

### 5.3 On-demand vs prefetch TPOT (Figure 8) — the headline gains

- **Qwen3-30B-A3B (A6000): 9–14% TPOT reduction**, bigger at longer sequences — closely approaches the max achievable speedup for the observed compute/copy split.
- **GLM-4.7-Flash (A6000):** similar trends (gains in the same regime).
- **GPT-OSS-120B (A100) and Qwen3-235B-A22B (GH200): max 5–8%** — smaller because A100/GH200 have higher compute throughput (and GH200 uses NVLink C2C, faster than PCIe).
- **Overall: 5–14% TPOT reduction, up to 14%, with 12–14% on the A6000** class of weaker-GPU + PCIe setups. Prefetching helps most where the GPU is compute-fast relative to the link (i.e., copy is most dominant).
- (Figure 8 plots per-sequence-length on-demand vs prefetch TPOT bar pairs for all 4 models. Per-point ms bar values are figure-derived and the layout assignment is ambiguous, so only the prose-confirmed *percentage reductions* above are stated; no per-point values back-filled.)

### 5.4 Prefill phase (Section 5.2)

- At typical prompt lengths, **all experts are effectively active at every layer even at B=1**, so any prefetch strategy trivially reduces to "load all experts during prefill" — no non-trivial expert selection occurs. The scheme is implemented for prefill but **not analyzed further**; all gains above are decode-phase.

---

## 6. Strengths, Limitations, Verdict

**Strengths**
- Clean, falsifiable mechanism: the quasi-hidden-state cosine-similarity + recall@k analysis (Figs 3–4) explains *why* GPT-OSS tolerates speculation and Qwen3 doesn't (early-layer drift), and the Hybrid-PF fix is directly motivated by that diagnosis.
- **Parameter-free** Router-PF works on pretrained weights with no fine-tuning — a strong practical default. The estimator is only needed for a few high-drift layers, and is tiny (4–45M params, ≤5M training tokens).
- Speculative-execution framing (run the predicted expert, skip the re-fetch) is the meaningful delta over cache-hint prefetchers and is what unlocks the full compute/copy overlap. Realized 5–14% TPOT gains closely track the Eq.-1 bound given the 84–88% copy share.
- Open-source integration in YALIS; portable to AMD (future work).

**Limitations**
- **Decode-only, batch-1, single-GPU offload.** Prefill is trivial; multi-GPU / tensor-parallel / larger-batch regimes (where active experts approach `E`) are out of scope.
- GLM-4.7-Flash and GPT-OSS-120B throughput runs use **random expert weights + simplified RoPE** (YALIS limitation) — throughput numbers are architecture-realistic but not weight-realistic for those two.
- No KV-cache offload in YALIS caps testable sequence length (HBM-bound); gains at >65536 unmeasured.
- The ⚠ §6.3 "37% GSM8k recovery" prose claim contradicts Table 1 (true Hybrid-PF recovery ≈99%); a reader relying on the prose under-counts the hybrid scheme's effectiveness.
- GPT-OSS-120B Router-PF *beating* baseline (0.857 vs 0.849) is within standard error — should not be read as "speculation improves accuracy," only "preserves it."

**Verdict.** A focused, well-motivated systems paper that isolates the one bottleneck that matters in offloaded MoE inference (CPU→GPU copy = 84–88% of TPOT) and attacks it with a parameter-free prediction (quasi-hidden state) plus a disciplined speculative-execution rule that avoids cache-miss re-fetches. The 5–14% TPOT gain is modest in absolute terms but lands at the compute/copy-bound limit (Eq. 1), and the accuracy story (GPT-OSS nearly free; Qwen3 rescued by Hybrid-PF) is honestly split by architecture. Strongest contribution: the **Hybrid-PF** recipe — estimator only on high-drift layers — which converts Qwen3-30B-A3B's −17.6pp Router-PF average degradation into −5.1pp while preserving the parameter-free default everywhere else.
