# WBMM — Windowed Batch Matrix Multiplication for Efficient Convolution

From-scratch PyTorch implementation of:

> "WBMM: Windowed Batch Matrix Multiplication for Efficient Convolution"
> arXiv:2607.02097 (2026).

WBMM replaces memory-bound large-kernel depthwise convolution (scattered
gathers) with a compute-bound batched matmul on contiguous w×w windows.
The weight matrix M is built once from a compact relative-position-bias
table and shared across all windows.

## Quick start

```bash
pip install torch numpy
python3 run.py
```

## Findings reproduced

| Finding | Paper | My result |
|---|---|---|
| F1 | WBMM 14×14 has 7.8× larger RF than DW-5×5 | RF 196 vs 25 = 7.8× |
| F2 | Throughput improves with window size | w=7 faster than w=3/5 (10.6 vs 16-21ms) |
| F3 | M is batch-independent | R table: 5408 params; M built once, shared |

Note: absolute speedup over cuDNN depthwise conv is GPU-specific. On CPU,
the optimized cuDNN path wins; on GPU (paper's target), WBMM wins for
large feature maps + batches.

## Files

| File | Purpose |
|------|---------|
| `model.py` | `WBMMConv2d`, `DepthwiseConv2d`, `build_window_matrix` |
| `run.py` | RF comparison + speed benchmarks + throughput scaling |
