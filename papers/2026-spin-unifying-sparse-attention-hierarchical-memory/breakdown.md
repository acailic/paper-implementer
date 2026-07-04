# Spin: Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving

**arXiv:** 2604.26837v1 (cs.LG, 29 Apr 2026) — https://arxiv.org/abs/2604.26837
**Authors:** Zihan Zhao (UVA), Baotong Lu (MSR), Shengjie Lin (GaTech), Yizou Chen (CUHK), Jing Liu (MSR), Yanqi Zhang (MSR), Ziming Miao (MSR), Ming-Chang Yang (CUHK), Haiying Shen (UVA), Qi Chen (MSR), Fan Yang (MSR).
**Venue/Year:** 2026 preprint. *Work done during a Microsoft Research internship.*
**Source-first build:** all numeric tables (1, 2, 3) transcribed verbatim from `paper.pdf` via `pdftotext -layout`. Figure-derived numbers are quoted only as the prose-stated ranges, never back-filled from bar heights.

---

## TL;DR

Long-context LLM serving is bottlenecked by the KV cache (e.g. 32K context for Qwen3-8B ≈ 19 GB; 1 GB+ per decoding step at scale). **Dynamic sparse attention** cuts compute, but its practical system-level benefit has been held back by (a) a lack of a *unified abstraction* (every new sparse algorithm re-implements its own KV management) and (b) *irregular GPU↔CPU KV retrieval* that erodes sparsity gains by underutilizing PCIe bandwidth.

**Spin** is a sparse-attention-aware inference framework built on vLLM that abstracts diverse sparse attention algorithms into a common **five-operation pipeline** (`Index`, `Select`, `Attention` = algorithm-specific; `Offload`, `Retrieve` = shared data movement) keyed on a single logical unit — the **partition**. Around this pipeline Spin builds a KV-cache manager for the GPU–CPU hierarchy with three design decisions: **(1) partition abstraction**, **(2) dynamic locality-aware KV management** (bucketed LRU), and **(3) hierarchical (OS-style multi-level) metadata**.

Headline results (3 models × 2 long-context benchmarks × A100 + B200):
- **1.66–5.66× higher end-to-end throughput** than vLLM; **7–9× lower TTFT**.
- **8.4–58% lower per-token decode latency** vs the *original* sparse-attention implementations (up to **2.39×** throughput), i.e. the system-level optimizations alone — independent of the algorithm — pay off.
- Hierarchical metadata cuts metadata HBM consumption by **49–78×**; locality-aware buffering lifts decode throughput **2.18×**.

This is a **new subarea for the repo**: long-context serving / sparse-attention runtime. It is a sibling to `jetspec` (speculative decoding) and `speculating-experts` (MoE inference) — all three are *inference-efficiency* papers, but Spin attacks the **attention + KV-memory** bottleneck rather than decode-token throughput or expert offloading.

---

## 1. Problem setup

### 1.1 Why long-context serving is hard (§2.1)
Two scaling stresses:
1. **KV cache grows linearly with sequence length** → GPU memory-capacity pressure → fewer resident requests. Common workaround: offload KV to CPU memory, keep only an active subset on GPU.
2. **Decoding is memory-bandwidth-bound** — each step accesses all historical KV. Even with full cache in HBM, decode is bandwidth-limited; once partially offloaded, PCIe transfers enter the critical path and can dominate latency.

### 1.2 Sparse attention's promise and its system gap (§2.2, §3)
Attention is inherently sparse: a small subset of historical tokens dominates each decode step. Sparse attention exploits this to lower KV access volume — and makes hierarchical KV storage (full cache in CPU, working set on GPU) attractive.

But the realized benefit is *not* governed by attention sparsity alone — it depends on how well the runtime orchestrates data movement across the GPU–CPU boundary. Two failure modes:

- **No unified abstraction.** Sparse methods operate at different granularities (blocks, clusters). Existing frameworks offer no common interface, so each algorithm is a bespoke prototype that cannot share kernels / memory mgmt / scheduling.
- **Irregular KV retrieval erodes sparsity.** Unlike dense attention's contiguous, predictable KV access, sparse attention retrieves small scattered subsets that vary per step/layer/head → PCIe underutilization + critical-path latency. (Mitigant: autoregressive decode shows *temporal locality* — critical tokens overlap across consecutive steps.)

### 1.3 The "ideal sparse-serving envelope" (§3.1)
Spin defines an oracle system that (a) serves most selected KV tokens from GPU memory and (b) incurs the theoretically minimum retrieval overhead under an optimal caching policy (Belady). Figure 1 shows existing sparse implementations (ShadowKV) sit far below this envelope and fail to scale to large batch sizes. Spin is engineered to approach the envelope.

The cost model (paraphrased): with cache miss ratio ρ, effective HBM bandwidth β_HBM, PCIe bandwidth β_PCIe, and MLP latency T_MLP, decode latency is dominated by ρ × (transfer cost) — so minimizing ρ via caching is the lever.

---

## 2. Method

### 2.1 The five-operation pipeline (§4.1)

Spin abstracts every sparse attention algorithm into the same execution pattern, split into three **algorithm-specific compute** ops and two **shared data-management** ops:

| Op | Stage | Algorithm-specific? | Role |
|---|---|---|---|
| **Index** | prefill + periodic decode | yes | Organize KV tokens into partitions; produce partition spec, per-partition summaries, algorithm metadata. |
| **Offload** | shared | **no** | Materialize partitioned KV across GPU/CPU tiers; keep summaries GPU-resident; update Spin metadata. |
| **Select** | decode-specific | yes | From current query + partition summaries, identify the most critical partitions for this step. |
| **Retrieve** | shared | **no** | Check residence of critical partitions; pull any missing ones CPU→GPU. |
| **Attention** | compute | yes (usually FlashAttention) | Compute output over selected KV tokens; algorithms may register custom kernels (e.g. RetroInfer). |

The unifying idea: **the partition** — a contiguous group of tokens that is the logical unit mapping algorithm-defined sparsity units onto hardware-efficient page-based KV management. Developers implement only `Index`/`Select` (+ optionally a custom attention op); Spin reuses optimized `Offload`/`Retrieve` for all algorithms.

### 2.2 Operation mappings across algorithms — Table 1 (verbatim)

**Table 1.** Operation mappings across various sparse attention algorithms. SeerAttention-R uses head-wise linear projection for pooled query and key, i.e. `bshi,hio->bsho`, denoted by `exp`. The Attention operator is omitted as it largely follows standard FlashAttention. The copy kernel used in Offload and Retrieve moves data from source region (1st parameter) to destination region (2nd parameter).

| Algorithm | Index | Select | Offload | Retrieve |
|---|---|---|---|---|
| **ShadowKV** | `mean(paged_key, dim=dim_page)` | `score(roped_q, k_sums) → topk` | `copy(gpu_kv, cpu_kv, map)` | `copy(cpu_kv, gpu_kv, indices)` |
| **RetroInfer** | `segment_k_means(main_key)` | `score(roped_q, k_sums) → topk` | (shared) | (shared) |
| **SeerAttention-R** | `einsum(exp, pooled_key, W_kg)` | `einsum(exp, pooled_q, W_qg) → gated_q`; `score(gated_q, k_sums) → topk` | (shared) | (shared) |

**Takeaways from Table 1:**
- The three algorithms differ only in `Index`/`Select` internals; `Offload`/`Retrieve` collapse to the *same* shared `copy(...)` kernel — this is the concrete payoff of the partition abstraction.
- Granularity differs: ShadowKV pools pages by mean, RetroInfer runs k-means segments, SeerAttention-R learns a linear projection — yet all three slot into one pipeline.

### 2.3 Memory management — the three design decisions (§5)

**(1) Partition abstraction.** Partition is elevated to the core abstraction for KV-cache management, decoupling algorithm-defined sparsity units from hardware-efficient page granularity. (This decoupling is what lets the same Offload/Retrieve serve all algorithms.)

**(2) Dynamic, locality-aware KV management.** Beyond append-only paged KV caches, Spin adds a lightweight locality-aware KV manager:
- A **dynamic KV buffer** that supports per-request resizing and in-place updates as requests arrive/advance/complete.
- **Bucketed LRU** replacement: instead of a strict per-page timestamp LRU, timestamps are bucketed into a small fixed range (e.g. 64 values), each bucket = pages of similar recency. A custom LRU kernel exploits GPU parallelism to select victims and perform bulk eviction. Bucketing cuts the GPU metadata + kernel cost of fine-grained recency tracking.
- The scheduler splits a request's GPU allocation into **mandatory pages** (must-resident for the current step) and **buffering pages** (retain for cross-step reuse), balancing per-request throughput against system-wide concurrency.

**(3) Hierarchical metadata (OS-style multi-level indexing).** Per-head sparsity needs per-head page tables; naïvely pre-allocated flat tables blow up GPU memory at long context. Spin uses a **two-level index** (small statically-allocated top-level directory on GPU → on-demand second-level segments) so metadata scales with the *physical working set*, not the worst-case logical address space. Metadata is split GPU/CPU: latency-critical tables on GPU; CPU partition-offset tables and page tables placed in **pinned memory** for direct GPU-kernel access. Four tables total: GPU Meta-partition Table, GPU Page Table, CPU Page Table, CPU partition offset table.

### 2.4 Configs of Spin-powered systems — Table 3 (verbatim)

**Table 3.** Configurations of Spin-powered systems, aligned with the defaults used in their original implementations. ShadowKV and RetroInfer use a *fixed ratio* of context length as retrieval budget, while SeerAttention-R retrieves a *fixed number of tokens* regardless of context length. Partition granularity and physical page size are measured in tokens.

| System | Retrieval Budget | Partition Granularity | Physical Page Size |
|---|---|---|---|
| **ShadowKV** [55] | 1.56% | 8 | 8 |
| **RetroInfer** [13] | 1.8% | variable | 8 |
| **SeerAttention-R** [19] | 2K | 32 | 32 |

**Takeaways:**
- ShadowKV retrieves only ~1.56% of context — aggressive sparsity; its partition == page (granularity 8 == page size 8), the simplest mapping.
- SeerAttention-R retrieves a context-**independent** 2K tokens → at 120K context this is a much smaller fraction than ShadowKV/RetroInfer's ratio-based budgets, which is why Spin-SeerAttention wins decode throughput at the longest contexts (Figure 12c/f) but its *absolute* retrieval volume makes online TPOT higher (§6.2).

---

## 3. Evaluation

### 3.1 Setup (§6.1)

**Testbed (two servers):**
- **A100:** 4× NVIDIA A100 (80 GB HBM each), AMD EPYC 7V12 CPU, 850 GB DRAM, PCIe Gen4 @ 32 GB/s per GPU.
- **B200:** 4× NVIDIA B200 (180 GB HBM each), Intel Xeon Platinum 8570 CPU, 1.5 TB DRAM, PCIe Gen5 @ 64 GB/s per GPU.

**Models:** Qwen3-14B (TP=1), Qwen3-32B (TP=2), Llama-3.1-70B (TP=4) — covering different sizes and architectures.

**Workloads (Table 2):** LongBench-v2 (long input / short output) and LongGenBench (shorter input / long reasoning-driven output). Request arrivals ~ Poisson at varying rates.

**Baselines:** (1) vLLM full attention; (2) vLLM-Offload (KV offload to CPU under HBM pressure); (3) LServe (GPU-only sparsity-exploiting server; offline-only — its scheduler doesn't support online arrivals). Spin integrates ShadowKV, RetroInfer, SeerAttention-R → Spin-ShadowKV / Spin-RetroInfer / Spin-SeerAttention. Spin-SeerAttention is **Qwen3-14B only** (SeerAttention-R is training-based, unavailable for larger models).

### 3.2 Workload characteristics — Table 2 (verbatim)

**Table 2.** Workload characteristics. LongBench-v2 has long inputs and short outputs, whereas LongGenBench exhibits shorter inputs but longer outputs. Lengths are measured in tokens.

| Benchmark | Input Min | Input Max | Input Avg | Output Min | Output Max | Output Avg |
|---|---|---|---|---|---|---|
| **LongBench-v2** [7] | 32K | 120K | 55K | 500 | 15K | 5K |
| **LongGenBench** [36] | 16K | 19K | 18K | 7K | 32K | 12K |

### 3.3 End-to-end online serving (§6.2)

**Throughput (Figures 7, 8).** Spin-powered systems keep scaling with request rate while vLLM / vLLM-Offload plateau and decline under high load (memory thrashing). By sustaining larger batches via reduced per-request HBM demand:

| Benchmark / GPU | Speedup vs vLLM (at the stated rate) |
|---|---|
| LongBench-v2, A100 @ 1.5 req/s | **2.34–3.80×** |
| LongBench-v2, B200 @ 4 req/s | **2.27–4.03×** |
| LongGenBench, A100 @ 1.5 req/s | **2.62–5.66×** |
| LongGenBench, B200 @ 4 req/s | **1.66–3.72×** |

The envelope across all conditions: **1.66–5.66× throughput** (the headline). Figure 9: Spin sustains **2.5–3.3× larger average batch sizes** than vLLM/vLLM-Offload. Among variants, **Spin-SeerAttention delivers relatively lower online throughput** because its larger per-step retrieval budget incurs higher PCIe overhead.

**Latency (Figure 10, Qwen3-14B on A100, Spin-ShadowKV vs vLLM):**
- **TTFT:** comparable at low load (≤ 0.001 req/s); Spin-ShadowKV reaches **9× lower TTFT at 0.5 req/s** (larger batches → less queueing). Stated envelope: **7–9× lower TTFT**.
- **TPOT:** Spin is *higher* than vLLM (larger per-step decode batches + GPU↔CPU retrieval), but the gap is stable across request rates (governed by the minimum per-request buffer). Other variants: **Spin-SeerAttention +107% TPOT** and **Spin-RetroInfer +14.7% TPOT** over Spin-ShadowKV, due to algorithmic retrieval-budget differences.

### 3.4 Offline performance (§6.3)

**Prefill (Figure 11):** Spin-powered systems add at most **0.59 s (16K), 0.92 s (32K), 1.32 s (64K)** over vLLM — indexing/KV-offload run largely asynchronously with prefill compute; some configs even match or undercut vLLM (Spin's GPU-side allocator avoids host-side allocation overhead at long context). LServe achieves **2.5–29% lower prefill latency** via sparse prefill (orthogonal to Spin; integrable).

**Decode (Figure 12):** Spin supports **4–8× larger maximum batch sizes** than baselines → **1.34–3.49× higher decode throughput**. On B200 at small batch sizes, Spin's decode throughput *falls below* vLLM (B200's higher compute + HBM bandwidth alleviates the full-attention bottleneck), but Spin re-takes the lead at larger batches via hierarchical GPU–CPU memory sustaining higher concurrency. Spin-SeerAttention gives the highest decode throughput at 120K context (fixed retrieval budget independent of context length).

**vs original sparse implementations (Figure 13):** Decode-latency reductions at **batch size 1** / **at peak-throughput batch** respectively:

| Variant | vs its original implementation |
|---|---|
| Spin-ShadowKV | **21%** (bs=1) / **35%** (peak) |
| Spin-RetroInfer | **22%** (bs=1) / **8.4%** (peak) |
| Spin-SeerAttention | **47%** (bs=1) / **58%** (peak) |

Envelope: **8.4–58%** decode-latency reduction, translating to **up to 2.39× throughput**. The improvement source differs by algorithm: ShadowKV/RetroInfer gain from the locality-aware buffer cutting PCIe retrieval time; SeerAttention-R (originally GPU-only, no retrieval) gains from Spin's optimized GPU kernels replacing under-optimized original operators.

### 3.5 Ablations (§6.4) — Spin-ShadowKV, Qwen3-14B, 32K context, A100

**GPU buffer management (Figure 14), three incremental configs:**
- **Base** — offload all KV to CPU, no GPU caching: throughput plateaus at low batch (every retrieval traverses PCIe).
- **+ Mandatory** — reuse only pages from the last decode step: reduces transfer volume but captures only single-step reuse.
- **+ Mandatory + Buffering** — full bucketed LRU exploits cross-step temporal locality → **+49% decode throughput**.

**Cache size vs hit ratio (Figure 16):** hit ratio climbs sharply with cache size then **saturates once the Buffering-to-Mandatory ratio reaches 4×** — beyond that, extra cache yields marginal hit-ratio gain while consuming GPU memory that could shrink the max executable batch. (Trend holds across workloads, algorithms, models — it directly guides the scheduler's cache-size setting.)

**Multi-level indexing / HBM savings (Figure 15), 128K context + batch 32:**
- Flat page table + metadata in GPU: **up to 100 GB HBM for Llama-3.1-70B** (prohibitive).
- Flat → **two-level index**: metadata HBM **13.9–17.4×** reduction.
- + offloading selected tables (CPU partition-offset + page tables) to CPU: further **3.5–4.5×** reduction.
- Combined headline: hierarchical metadata cuts metadata HBM by **49–78×** (the abstract + §1 figure).

---

## 4. Implementation notes (§5.5)

- Core execution engine: **7k lines C++/CUDA**; vLLM integration: **3k lines Python** (modifies vLLM's scheduler + memory allocator).
- **Warp-based copy kernel** for non-contiguous head-wise KV page transfer across GPU↔CPU: each warp copies one page with vectorized memory transactions to saturate PCIe; a **persistent-kernel** design distributes pages across all warps regardless of owning head/request for load balance.

---

## 5. Strengths / Limitations / Verdict

**Strengths**
- **Generality via the right abstraction.** The partition + 5-op pipeline turns "N bespoke sparse runtimes" into "N algorithm plugins on one engine" — Table 1 makes the convergence concrete (identical Offload/Retrieve across all three algorithms). This is the kind of systems contribution that ages well as new sparse algorithms appear.
- **The system-vs-algorithm attribution is clean.** The 8.4–58% decode-latency win over the *original* sparse implementations isolates the runtime's value from the algorithms' value — strong evidence the gains are engineering-real, not algorithm-luck.
- **Honest about a regime where it loses.** Spin is upfront that on B200 at small batch sizes its decode throughput falls *below* vLLM (newer GPU's bandwidth alleviates dense attention's bottleneck) — a credible caveat rather than a hidden weakness.

**Limitations**
- **TPOT is higher than dense vLLM** (retrieval on the critical path). This is fundamental to CPU-offload sparse serving and only amortizes at high concurrency — single-user low-latency chat is not Spin's regime.
- **SeerAttention-R coverage is limited** to Qwen3-14B (training-based; unavailable for larger models), so the headline "3 algorithms × 3 models" matrix is partially populated.
- **No quality/accuracy evaluation.** Spin is a serving-runtime paper; it inherits each algorithm's accuracy guarantees and reports no end-task quality numbers — accuracy claims rest entirely on the integrated algorithms' own papers.
- **LServe's sparse-prefill advantage is left on the table** (acknowledged as orthogonal/future), so Spin's prefill latency is 2.5–29% worse than LServe's.

**Verdict.** A well-scoped, well-executed *systems* paper (not a new sparse-attention algorithm). Its durable contribution is the **partition abstraction + 5-op pipeline + hierarchical metadata** that lets sparse attention actually realize its theoretical promise under realistic serving. The 1.66–5.66× throughput / 7–9× TTFT / 49–78× metadata-HBM numbers are the kind that justify a runtime rewrite. For this repo it fills the **long-context-serving / sparse-attention-runtime** subarea cleanly — complementary to the inference-efficiency siblings (`jetspec` = token-level spec-decode, `speculating-experts` = expert-level offload, Spin = attention + KV-memory level).

---

## Sourcing notes

- **Tables 1, 2, 3** transcribed verbatim from `paper.pdf` (lines ~265, ~501, ~571 of `paper_layout.txt`); column headers and footnotes preserved.
- All throughput/latency/ablation numbers are the **prose-stated ranges** from §6.2–§6.4, cross-referenced to their figure. Per-point figure-bar values (Figures 7/8/10/12/13/14/15/16) are **not** back-filled — the layout dump gives axis ticks and scattered series labels but not reliable per-curve point assignment, consistent with the established "figure-derived numbers are weak" rule.
- Figure 1's envelope chart (SPIN-ShadowKV vs ShadowKV vs Ideal at batch sizes 1–64) is qualitative support for the "gap between theoretical and realized performance" thesis; its per-batch bar values are figure-derived and not transcribed.
- **Paper-internal wording note (not a numeric defect):** the abstract/§1 uses the spelling "hierachical" in one occurrence ("The hierachical metadata organization of Spin reduces metadata HBM consumption by 49–78×…"); the body uses "hierarchical" throughout. Transcribed here uniformly as "hierarchical"; the 49–78× figure itself is consistent across abstract, §1, and Figure 15's two-step derivation (13.9–17.4× × 3.5–4.5×).
