# Implementation — Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation

> Mechanism miniature, **not** a model reproduction. Vidu S2 is a proprietary
> tech report (no weights, no architecture sizes, no code for the models).
> What IS implementable from the paper is the set of *control and training
> mechanisms* around the model — that is what this folder demonstrates, on
> synthetic signals, on CPU, in under a minute.

## Paper reference

- **Title:** Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation
- **Authors:** Jintao Zhang, Kai Jiang, Jintao Chen, et al. (Tsinghua University, Shengshu Technology)
- **ArXiv:** https://arxiv.org/abs/2609.11638

## What's implemented

| # | Paper mechanism | Toy analogue | Where |
|---|-----------------|--------------|-------|
| 1 | Block-causal streaming DiT with shared reference r and per-segment conditions c_i (Eq. 1–3) | Tiny attention model over 8-dim chunks (4 motion + 4 reference-appearance dims); reference enters as attention tokens visible to every chunk | `model.py:RefCondStreamModel` |
| 2 | **Self-Replay Forcing**: detached on-policy rollout → per-block re-noise → one gradient-enabled causal replay (Eq. 4) | Four training arms — TF, DF-GT (off-policy Diffusion Forcing), SF (detached rollout, frozen KV), SRF (rollout + re-noise + live-KV replay) — plus a direct gradient-flow probe | `train.py:E1` |
| 3 | **Dynamic reference injection** ("new reference image at any moment") | Swap the reference mid-rollout; measure adoption latency. Control: reference tokens visible only to chunk 0 (I2V-style) | `train.py:E2` |
| 4 | **Frame-aligned attention** for streaming editing (target frame reads only same-timestamp source frame; reference visible to all) | Streaming editor over synthetic source streams; `source_mask="aligned"` vs `"free"` (target attends all past source frames) | `train.py:E3` |
| 5 | **Spatial layer**: depth → disparity → bidirectional warp → hole fill | 1-D image row with depth steps; disocclusion-hole census vs disparity | `train.py:E4` |
| 6 | Diffusion noise convention of Eq. 5 (t=0 clean, t=1 noise) | Used throughout: `x~ = (1-t)·x + t·ε` | `data.py:noised`/`train.py` |

Deliberately NOT implemented (out of reach at toy scale / proprietary): the
audio-visual joint DiT itself, DMD distribution matching (replaced by
regression to the synthetic oracle — see "Differences"), the latent-space SR
Refiner with asymmetric τ_B/τ_R caches, Streaming NFT/DPO, the VLM agentic
prompt layer, the serving stack (SageAttention/W8A8/CUDA graphs/Ulysses), and
real video.

## How to run

```bash
pip install -r requirements.txt   # torch (CPU is fine), numpy
python train.py                   # ~55 s single-threaded CPU
```

Writes `metrics.json` (regenerated per run; gitignored).

## Results (actual run, single-threaded CPU, 55.5 s)

**E1 — forcing schedules.** Trained on 12-chunk windows, evaluated by rolling
48 chunks autoregressively from pure noise (oracle floor 0.0106):

| arm | MSE | motion | look | drift (last/first quarter) |
|------|------|--------|------|------|
| TF (clean GT history) | 1.689 | 0.814 | 2.564 | 2.58 |
| DF-GT (re-noised GT windows) | 0.415 | 0.292 | 0.538 | 29.79 |
| SF (on-policy, frozen KV) | **0.129** | 0.204 | 0.054 | 11.88 |
| SRF (on-policy, re-noised replay) | 0.164 | **0.195** | 0.132 | 16.78 |

- Exposure bias dominates: both on-policy arms beat both off-policy arms by
  2.5–13x. TF (never saw noise, never saw its own outputs as history)
  collapses; DF-GT drifts hardest long-horizon (error 30x by the last
  quarter).
- SRF does not beat SF on raw MSE at this scale; it does get the best motion
  (temporal-structure) error. What SRF uniquely provides — cross-block
  gradient flow — is verified structurally by the probe:
  `d L(last 6 chunks) / d e_early` reaches **6/6** early encodings under SRF
  replay and **0/6** (exactly zero) under SF's frozen KV.
- Building this probe caught a real bug in our own SF baseline: detaching the
  history values but not the history keys silently re-enabled key-side
  gradient flow. Frozen KV means frozen K *and* V.

**E2 — dynamic reference injection** (swap reference at chunk 12 of 24):

| ref visible to | err(old ref) before → after | err(new ref) after | sustained adoption |
|---|---|---|---|
| all chunks (Vidu S2 design) | 0.004 → 1.10 | **0.59** | 3 chunks |
| chunk 0 only (control) | 0.26 → 1.40 | 1.52 (≥ old) | never |

With reference tokens in every chunk's attention, the look flips to the new
reference within ~3 chunks of the swap and motion is unaffected (0.224). With
chunk-0-only injection the model never adopts the new reference — the
conditioning interface, not model capacity, is what makes mid-stream swaps
work.

**E3 — frame-aligned editing** (streaming edit of a synthetic source stream):

| mask | motion MSE (in-dist) | motion MSE (2x speed) |
|---|---|---|
| aligned | **0.227** | **0.226** |
| free | 0.349 | 0.344 |

The hard alignment prior (target frame reads only same-timestamp source frame)
beats learned free cross-attention by ~35% on motion preservation even in
distribution, and neither degrades at 2x source speed at this scale.

**E4 — spatial warp:** disocclusion holes per view grow 0% → 2.5% → 7.5% as
disparity gain rises 3 → 6 → 12 (longest hole runs 0 → 2 → 4 px) — the entire
cost of warp-based stereo is concentrated at depth discontinuities, which is
why the paper's conversion stage is "lightweight hole fill" by necessity.

## Known gaps / TODOs

- DMD is replaced by oracle regression (see below); a faithful toy DMD would
  need a frozen teacher *distribution*, not ground-truth points.
- No Refiner / asymmetric-cache experiment (τ_B > τ_R); a second consumer
  reading the same history at a different noise level is implementable here
  but was cut for scope.
- SRF's replay window equals the training window (12 chunks); the paper's
  "long rollout then replay" (rollout longer than replay) is not modeled.
- Everything is 8-dim synthetic signals; no images, no audio, no VAE.

## Differences from the original

- **Scale and data**: ~10⁴-parameter attention model on synthetic 8-dim
  chunks vs. an audio-visual joint DiT on millions of videos.
- **Objective**: SRF supervises replayed blocks with MSE against the
  synthetic ground truth; the paper uses DMD (KL to a frozen bidirectional
  teacher) + perceptual loss. Gradient-flow structure is preserved;
  distributional benefit is not representable with an oracle.
- **SF baseline**: faithful to the paper's description — history fed in
  clean (t=0), KV frozen (detached K and V), loss at generation time.
- **Noise convention** follows the paper (t=0 clean, t=1 noise), including
  one-step generation from t=1 at inference.
