# Writeup — WBMM: Windowed Batch Matrix Multiplication

> arXiv:2607.02097 (2026).

## The idea

Large-kernel depthwise convolutions (7×7 to 101×101) are memory-bound: each
output pixel gathers k² scattered neighbors from non-contiguous memory. WBMM
inverts the access pattern — instead of traversing the data (scattered
gathers), traverse the parameter table (contiguous). Partition the input
into w×w windows, build a per-channel weight matrix M from a compact
relative-position-bias table R, and apply via batched matmul. M is
batch-independent (shared across all windows) → compute-bound.

## What I implemented

A from-scratch WBMMConv2d module with the relative-position-bias table,
vectorized window-matrix construction, and batched matmul application.
Compared against standard nn.Conv2d depthwise (groups=C) baseline.

## Key findings

- **Receptive field**: WBMM 14×14 gives 7.8× larger RF (196 vs 25 positions)
  than DW-5×5 — matching the paper's headline number exactly.
- **Throughput scaling**: On the same input, w=7 is faster than w=3 or w=5
  (10.6ms vs 16-21ms). This is the paper's counterintuitive finding: larger
  windows are faster because the matmul amortizes the window-matrix
  construction cost over more compute.
- **CPU vs GPU**: On CPU, the cuDNN-optimized depthwise conv still wins in
  absolute terms — the paper's speedup is specific to GPU where the
  memory-bound → compute-bound shift matters. My implementation verifies the
  mechanism (batch-independence, throughput scaling) but not the absolute GPU
  speedup.

## What implementing it clarified

The core insight is that depthwise conv can be reformulated as a matrix
product (Theorem 3.2): y = x·M. The full global M is impractical (HW×HW per
channel), but the windowed approximation (Swin-style locality) shrinks it to
d×d (d=w²) per window. The relative-position-bias table R ∈ R^{C×(2w-1)²}
makes M cheap to store and build — just (2w-1)² values per channel, indexed
by relative offset. This is the same position-bias trick used in Swin
Transformer, repurposed for depthwise conv acceleration.

🏆 Verdict: a systems paper with a clean mechanism. The windowed matmul
trick is generalizable beyond depthwise conv to any operator with
shift-invariant structure.
