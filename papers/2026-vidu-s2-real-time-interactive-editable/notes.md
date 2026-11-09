# Vidu S2 — Reading Notes

**Paper:** Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation
**Authors:** Jintao Zhang, Kai Jiang, Jintao Chen, et al. (Tsinghua University, Shengshu Technology)
**ArXiv:** https://arxiv.org/abs/2609.11638 (Sep 2026)
**Project:** https://vidu.com/vidu-stream

---

## First Impressions

This is a product-line tech report from Shengshu (the Vidu team) + Tsinghua — the
successor to Vidu S1 (2607.03118). It is *not* a single-model paper: it announces
two deployable systems, **Vidu S2-Avatar** (real-time interactive digital
character, now at 720p, 25–42 FPS) and **Vidu S2-Editing** (real-time editing of
an incoming video stream: style transfer, virtual try-on, subject replacement,
background replacement), plus a first exploration of **real-time spatial (stereo)
video** for VR headsets. Playable demo online. No weights, no code for the models
themselves.

The interesting science is buried mid-paper:

- **Self-Replay Forcing (SRF)** — their fix to Self-Forcing's training gap. The
  student rolls out autoregressively (detached), then the whole self-generated
  trajectory is re-noised per Diffusion Forcing and replayed in ONE
  gradient-enabled causal pass, so loss on later segments backprops into earlier
  segments without backprop through the rollout itself. They call it "on-policy
  DMD". This is the most citable mechanism in the paper.
- **Frame-aligned attention** for editing — every target frame attends only to
  the source frame at the same timestamp (preserves motion/timing exactly), while
  the reference image stays visible to all frames (carries the new appearance
  through the whole video). During streaming, each consumed source frame is
  dropped from the cache. Simple, clean idea.
- **Asymmetric cache noise levels** between the causal backbone (high-noise
  cache, coarse motion prior) and the one-step super-resolution Refiner
  (low-noise high-res cache, appearance/identity detail). τ_B > τ_R.
- **Dynamic references at any moment** — mid-stream you can hand the model a new
  image (object to hold, clothes to wear, background to move to). A VLM agentic
  layer interprets the reference type, writes structured prompts with *state
  persistence* (once the cup is picked up, later prompts keep "holding the cup"
  unless told otherwise), and visually verifies whether requested actions
  completed.

## Problem (in my own words)

Offline video diffusion generates a whole clip synchronously over many denoise
steps; the user waits minutes and can't steer anything. But a large share of
visual entertainment is *interactive* — video calls, live streams, games. Their
demand argument is cute: if per-user demand for interactive content is α and for
offline content is β (α≈β), interactive demand scales like α·N while offline
*serves* β·N/m where m = average views per generated video (m>100 for shareable
content) — so interactive generation demand is orders of magnitude larger.

Vidu S1 already streamed (540p, 42 FPS, voice control), but: fixed reference
once a stream starts, talking-head-centric, can't follow big body motion
(dancing), can't edit an incoming stream. S2 is "S1 without those four
ceilings", plus stereo.

## What the two models do

- **S2-Avatar**: audio-visual joint DiT, reference-conditioned. r = reference
  image shared across all segments; c_i = per-segment condition (caption/audio).
  Trained bidirectional (I2V + R2V jointly, segment-wise captions), then
  converted to block-causal streaming via hybrid Teacher/Diffusion Forcing, then
  SRF + DMD + perceptual loss, then preference optimization (DPO bidirectional,
  Streaming NFT at the streaming stage), then one-step latent Refiner for 720p.
- **S2-Editing**: DiT conditioned on source video tokens + optional reference
  tokens + instructions; frame-aligned attention to source; same hybrid-forcing
  → SRF pipeline. Streaming: source frame consumed with its target frame,
  nothing retained.
- **Spatial**: monocular chunk → per-frame depth → horizontal disparity → warp
  both directions → lightweight hole fill + temporal depth stabilization →
  synchronized L/R chunks. Or edit-then-convert; or for stereo input, join L/R
  side-by-side into one wide stream, edit jointly, split back.

## Data pipeline details worth remembering

- S2-Avatar data: 5-stage S1 pipeline (clipping, filtering, speech processing,
  captioning, embedding) + new sources (solo dance, 2D/3D animation) + new
  operators: VLM-assisted cut-point detection (<2% false rejection),
  high-clarity selection (weighted score over resolution/bitrate/codec/texture/
  sharpness/artifacts — nominal resolution alone is a bad proxy), background
  stabilization (mask subject, fit global geometric transform to background,
  warp+crop — most dance-camera motion is rotation/focal, small parallax).
- Captions: switched from structured fields (subject/environment/camera) to
  *temporally ordered dense captions* with event boundaries as chunk
  boundaries — chunks become finer and event-aligned, lowering response latency.
- S2-Editing data: 800K curated videos → 4 disjoint 200K subsets (one per
  task). Style-transfer pairs are synthetic: train a conditional V2V model to
  reconstruct raw video from (NormalCrafter surface normals + a sampled frame as
  reference), then at generation time swap in a stylized reference → stylized
  video with identical structure/motion. Generalizes to 50+ styles. Other tasks:
  task-specific editors + open models (Bernini, SCAIL-2, VACE, SAMA-14B,
  CoinVE-Edit) with post-filtering and comparative selection (no single model
  wins everywhere).
- Editing captions are written WITHOUT showing the source video to the VLM —
  prevents source content leaking through text conditioning.

## Headline numbers (from the tables)

- StreamAV-Bench (Table 1): Vidu S2-Avatar best on all 9 metrics — VA .687,
  VQ 3.370, PQ 7.138, AQ 3.286, AVAlign .353, AVSync .617 (↓), AIF 2.985,
  SC .998, BC .993. (Beware: the plain-text PDF extraction scrambles the last
  three columns; the -layout extraction has the right alignment.)
- Sparkle-Bench (Table 2): S2-Editing Overall 3.74 vs best baseline
  Decart-Lucy2.5 3.67; instruction 4.00.
- OpenVE/RefVIE (Table 3): Joint Overall 4.26 vs Bernini-R 14B 3.92 (+0.34).
- ViViD unpaired try-on (Table 4): VFIDI 9.9515 vs CatV2 TON 19.5131 — a 2x gap.
- Internal GSB (Figs 6–7): preferred over Runway GWM-1 (85.7% overall), PixVerse
  Image Avatar and HeyGen (100% overall); editing mean consistency 3.56 vs
  Decart-Lucy2.5 2.59 vs XMax-X2.0 1.67 on 150 paired cases. Duration-stratified
  ratings stay on top from 10 s to 90 s.

## Key things I need to understand better (second-pass targets)

- **SRF vs Self-Forcing vs DMD exactly**: what is detached, what is re-noised,
  where gradients flow, and why "replay in a single causal pass" recovers
  cross-block gradients that Self-Forcing lacks. Eq. (4) is just
  L = L_DMD + L_perc; the structure is all in the procedure.
- **The KV-cache story**: "sink blocks, sliding-window context, noisy KV caches"
  retained from S1 — how do these interact with SRF replay?
- **Frame-aligned attention**: is the source→target alignment a mask on
  cross-attention only, or also inside self-attention? And how RoPE positions
  are assigned to concatenated condition streams.
- **The Refiner's asymmetric caches**: Eq. (5) is a one-step re-noise of the
  upsampled LR latent at fixed t_r; the cache noise levels τ_B, τ_R are fixed
  across segments — why is that safe.
- **No parameter counts, no training compute, no exact latency numbers** in the
  text (only FPS ranges and audio-chunk sizes 160/320/480 ms as eval
  conditions). What's actually measured where.

## What's clear

- The engineering-stack section (SageAttention/SpargeAttention/SLA per layer,
  per-block W8A8 GEMM, kernel fusion + CUDA Graphs, Ulysses context parallelism
  with quantized comms, inter-module GPU scheduling on a shared timeline) is a
  big part of how 720p real time is achieved at all.
- The paper is honest that spatial video is a *feasibility exploration*, with
  resolution/latency called out as the open challenges for headset comfort.

## What's unclear or concerning

- Proprietary tech report: no architecture sizes, no data volumes beyond the
  editing subsets, no code. "Outperforms all baselines" on benchmarks where
  several baselines are their own prior systems.
- Zero ablations separating SRF from plain Self-Forcing, or frame-aligned from
  free cross-attention — the mechanisms are argued, not isolated. (Consistent
  with the Vidu S1 / Wan-Streamer style of systems reports.)
- The α·N vs β·N/m demand argument assumes away that interactive sessions are
  shorter and shared differently. Directionally fine, quantitatively hand-wavy.

---

# Second pass — deep read

## SRF: where the gradients actually flow

The whole trick is worth writing out precisely, because Eq. (4) hides it.

**Self-Forcing's two gaps** (their diagnosis): during training, the
self-generated history that conditions later chunks is (a) fed in *clean* (at
inference it comes from the student's own imperfect one-step/DMD outputs), and
(b) *detached* (KV caches stored without grad), so a loss on chunk k can never
teach the model to produce chunk j<k that would have made chunk k better.

**SRF procedure**:
1. Student runs its real inference procedure — long autoregressive rollout,
   chunk by chunk, each conditioned on its own generated history. All generated
   blocks and KV caches are **detached** (no graph retained; memory stays O(1)
   per chunk). This is what makes the data "on-policy".
2. Take the rollout x̂₀^{1:N}, draw per-block diffusion timesteps
   t = (t₁..t_N), and **re-noise every block independently** à la Diffusion
   Forcing: block i becomes x_t^i at its own noise level.
3. **Replay** the whole re-noised trajectory in ONE forward pass of the causal
   model, gradients ON. The pre-rollout "external history" stays fixed (the KV
   inherited from before the window), but everything inside the window is
   recomputed in a single connected graph.
4. Loss = DMD (align student outputs to the frozen bidirectional teacher
   distribution) + perceptual regularization (anti mode-collapse), applied to
   **all** replayed blocks.

Why this recovers cross-block gradients: during replay, block i is conditioned
on blocks 1..i−1 through attention over their *recomputed* KV representations.
The noisy latents x_t^{<i} are constants, but the keys/values/hidden states the
causal pass computes from them are differentiable functions of θ. So
∂L_block_N/∂θ flows through the representations of blocks 1..N−1 — the later
loss shapes the way earlier blocks are *encoded*, if not their pixel content.
Self-Forcing can't do this because each chunk's cache was frozen at generation
time. And unlike full-sequence backprop through the rollout, the rollout itself
never appears in the graph — you only pay one extra forward+backward over the
window.

The noise re-injection in step 2 also repairs gap (a): blocks are seen at a
*spectrum* of noise levels as history, matching Diffusion Forcing's argument
that the model must be robust to history of varying quality.

## Refiner: asymmetric cache noise (Eq. 5)

z_{t_r}^i = (1−t_r)·U(x̂₀,LR^i) + t_r·ε, so their convention is t=0 clean,
t=1 pure noise. The Refiner upsamples the backbone's LR clean latent estimate,
re-noises it to a FIXED refinement timestep t_r, and one-step predicts the HR
latent. Caches: backbone reads a high-noise history cache x̂_{τ_B}^{<i}
(τ_B large → coarse, low-frequency motion prior, insensitive to artifacts);
Refiner reads a low-noise HR cache x̂_{τ_R,HR}^{<i} (τ_R small → appearance and
identity detail). τ_B > τ_R, both fixed across segments of a replayed
trajectory. The design claim: temporal propagation and spatial detail recovery
want different noise levels, and forcing one cache to serve both compromises
each. ("Inspired by TwinCache stage-aware caches" from S1.)

## Frame-aligned attention (editing), precisely

Bidirectional editing stage: target tokens self-attend bidirectionally; each
target frame's cross-attention to the source stream is masked to the
SAME-timestamp source frame only; the reference image tokens are visible to
every target frame. Effect: motion/timing is structurally pinned to the source
(hard alignment), while appearance flows globally from the reference (soft,
global). Then the same causal conversion + SRF as Avatar. Streaming inference:
source frame i arrives, is paired with target frame i, and is then dropped —
the source cache doesn't grow. (Contrast with Avatar, where generated history
IS the cache.)

RoPE: source, reference, and target token streams get condition-specific RoPE
encodings; everything is concatenated into one sequence.

## The full training stack (both models)

1. Bidirectional base: Avatar = joint I2V+R2V with segment-wise captions
   (r = first frame for I2V, r = reference for R2V) + diffusion DPO. Editing =
   frame-aligned bidirectional editing model.
2. Block-causal conversion: replace bidirectional temporal attention with a
   block-wise causal mask; train with a *hybrid* of Teacher Forcing (clean GT
   history) and Diffusion Forcing (history re-noised at sampled levels),
   sampled with fixed probability.
3. SRF: on-policy DMD + perceptual loss on replayed windows; retain sink
   blocks + sliding-window context + noisy KV caches (S1 machinery).
   Streaming NFT (reward-based) afterwards on the same replay structure.
4. Refiner trained separately in latent space (Avatar only).

## Inference/serving stack (why it's real-time at 720p)

- Layer-wise hybrid attention: per-layer choice among SageAttention (8-bit /
  int4 / fp4), SpargeAttention (training-free sparse), Sparse-Linear Attention
  — aggressive on insensitive layers, exact-ish on sensitive ones.
- Per-block W8A8 GEMM (block-granular scales contain outliers).
- Kernel fusion (e.g. RMSNorm + elementwise) + CUDA Graphs replay.
- Ulysses context parallelism across GPUs, with quantized tensor exchange.
- Inter-module scheduling (Editing): VAE enc / backbone / Refiner / VAE dec
  time-share one GPU pool along a common timeline instead of static placement.

## Evaluation, read carefully

- Public: StreamAV-Bench (Progressive + Interactive tracks; interactive gets a
  runtime update every 30 s, 5 updates, ≤180 s), Sparkle, OpenVE+RefVIE joint,
  ViViD try-on. Identical manifests/preprocessing; published numbers reused
  only when benchmark+split+metric match, else omitted.
- Human: 20 trained evaluators, calibration rounds, randomized GSB pairwise,
  recorded seeds/versions/regions. Internal benchmarks also measure
  interruption/recovery, 5-min stability, end-to-end latency, cost — audio
  chunks of 160/320/480 ms are the test conditions. Actual latency numbers are
  NOT printed in the paper text; only FPS ranges (25–42 FPS @720p) appear.
- Duration-stratified 1–5 ratings from 10 s to 90 s: Vidu on top in all five
  dimensions at all durations.

## Remaining unknowns after second pass

- Parameter counts, training compute, exact chunk length in seconds, number of
  denoise steps at inference (presumably ~1 after distillation, à la S1),
  teacher identity for editing DMD (presumably its own bidirectional editor).
- How dynamic reference injection is trained: "training on data built for
  reference-conditioned generation" + agentic wrapper — is a mid-stream
  reference swap a conditioning change only (r changes for subsequent blocks),
  with no architectural change? The paper implies yes: r is just an input, and
  the VLM agent translates the swap into prompt text. That makes the
  "mechanism" demotratable at toy scale: reference tokens enter every block's
  attention; swapping them mid-stream changes appearance from the next chunk
  onward.

