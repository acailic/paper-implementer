# Breakdown — Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation

> **Paper:** Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation
> **Authors:** Jintao Zhang, Kai Jiang, Jintao Chen, Xu Wang, Deyuan Liu, et al. (Tsinghua University, Shengshu Technology)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2609.11638
> **Code (official):** none for the models (proprietary); playable demo at https://vidu.com/vidu-stream
> **Type:** systems tech report (successor to Vidu S1, arXiv 2607.03118)

---

## 1. Problem & Motivation

**What problem?** Mainstream video generators (Sora, Veo, Wan, Seedance) are
*offline and one-shot*: prompt in, wait minutes, receive a finished clip. The
user cannot steer anything once generation starts. Yet a huge fraction of
visual entertainment is inherently interactive — video calls, live streams,
games, talking characters — where content must *respond immediately*.

**Why it matters (their demand argument):** let α ∈ [0,1] be a user's demand
for real-time interactive visual content and β ∈ [0,1] for offline-generated
content, with α ≈ β. Interactive demand scales as α·N (every session must be
generated, live, per user). Offline content is replayable and shareable, so its
*generation* demand scales like β·N/m, where m = average views per generated
video (easily m > 100). Conclusion: the generation workload for interactive
video is orders of magnitude larger than for offline video. Real-time
interactive generation is where the compute demand will be.

**Prior state:**
- *Offline diffusion* denoises the whole clip synchronously; fundamentally
  incompatible with streaming interaction.
- *Cascaded real-time systems* (ASR → LLM → TTS → avatar renderer) stack
  latencies, can't synchronize modalities natively, and can't be interrupted
  gracefully.
- *Vidu S1* (their own prior system): a real-time interactive streaming model
  (audio-visual joint DiT, block-causal, DMD-distilled, 540p @ up to 42 FPS on
  consumer GPUs, infinite length). But: reference fixed once the stream starts,
  talking-head-centric, cannot follow large body motion (dancing), cannot edit
  an incoming video stream.

**What S2 asks for:** keep S1's real-time interactive streaming, and add
(1) 720p, (2) references that can be swapped *at any moment* mid-stream,
(3) much stronger instruction following (dance-scale motion), (4) real-time
editing of an *incoming* video stream, and (5) a feasibility study of doing all
of this as *spatial* (stereo) video for VR headsets.

## 2. Key Insight / Contribution

Core idea, in three sentences. **(a)** The train–test mismatch of
autoregressive video diffusion can be closed *without* giving up cross-block
gradients: roll out on-policy with everything detached, then re-noise the whole
self-generated trajectory and *replay it in one gradient-enabled causal pass*
(Self-Replay Forcing). **(b)** Real-time editing does not need the model to
"understand" the whole source video: if each target frame may only attend to
the source frame at the *same timestamp* (frame-aligned attention) while a
reference image stays visible to all frames, motion and timing are pinned
structurally and only appearance is free to change. **(c)** Resolution and
temporal consistency want different noise levels: run the backbone fast at low
resolution with a *high-noise* history cache (coarse motion prior), and a
one-step Refiner at high resolution with a *low-noise* cache (appearance
detail) — asymmetric caches instead of one shared history.

Contributions claimed: (1) S2-Avatar (720p real-time, dynamic references,
dance-scale instruction following); (2) S2-Editing (streaming style transfer /
try-on / subject replacement / background replacement); (3) real-time spatial
generation + editing for VR; (4) an inference/serving stack (hybrid attention,
W8A8, fusion + CUDA graphs, context parallelism, inter-module GPU scheduling)
that makes it practical on low-cost GPUs; (5) wins on all benchmarks tried.

## 3. Method

### 3.1 Overview

Two models share one training recipe ("bidirectional base → block-causal
conversion → Self-Replay Forcing distillation") and one serving stack:

- **Vidu S2-Avatar** — audio-visual joint DiT. Streams a talking/dancing
  character conditioned on (i) a reference image r (identity/appearance,
  swappable mid-stream), (ii) per-segment conditions c_i (caption, audio),
  (iii) its own generated history. Output chunk → optional stereo conversion →
  display. A VLM agentic wrapper writes prompts, classifies references
  (object/background/clothing), and visually verifies action completion.
- **Vidu S2-Editing** — video-conditioned DiT. Streams an *edited* version of
  an incoming source video, conditioned on (i) source frames, (ii) optional
  reference image (desired appearance), (iii) edit instruction. Same causal +
  SRF recipe; same serving stack plus inter-module GPU scheduling.
- **Spatial layer** — converts monocular chunks to synchronized left/right
  views (depth → disparity → bidirectional warp → hole fill), or edits stereo
  input directly (join L/R horizontally, edit, split).

### 3.2 Architecture

```
                        Vidu S2-Avatar (training-time view)
 ┌────────────────────────────────────────────────────────────────────┐
 │ r (ref img)   c_1 (cap/audio)   c_2        c_3        ...  c_N     │
 │    │              │               │          │               │     │
 │    │   ┌──────────▼───────────────▼──────────▼───────────────▼──┐  │
 │    │   │        audio-visual joint DiT, BLOCK-CAUSAL mask       │  │
 │    ├──►│  segment i sees: r, c_i, sink blocks, sliding window,  │  │
 │    │   │  noisy KV cache of history x^<i                        │  │
 │    │   └──────┬───────────────┬──────────────┬─────────────┬────┘  │
 │    │          ▼               ▼              ▼             ▼       │
 │    │       x̂_0^1  ──hist──► x̂_0^2 ──hist──► x̂_0^3  ...  x̂_0^N     │
 │    │          │               │              │             │       │
 │    │          └───────────────┴──────┬───────┴─────────────┘       │
 │    │                                 ▼                             │
 │    │                    low-res clean latents (LR)                 │
 │    │                                 │                             │
 │    │              ┌──────────────────▼─────────────────┐           │
 │    │              │ SR Refiner (one step, latent space)│           │
 │    │              │  z = (1-t_r)·U(LR) + t_r·ε         │           │
 │    │              │  reads LOW-noise HR cache τ_R      │           │
 │    │              │  backbone reads HIGH-noise τ_B     │           │
 │    │              └──────────────────┬─────────────────┘           │
 │    │                                 ▼                             │
 │    │                        720p latents → VAE dec → stream        │
 └────────────────────────────────────────────────────────────────────┘

                        Vidu S2-Editing (attention mask)
 target frame:   [t_1] [t_2] [t_3] ...        (bidirectional among selves)
                    │     │     │
                    ▼     ▼     ▼              FRAME-ALIGNED: t_i reads
 source frame:   [s_1] [s_2] [s_3] ...         ONLY s_i (same timestamp)
 reference img:  [R R R R R ...]               visible to ALL target frames

 streaming: s_i arrives ─► paired with t_i ─► s_i dropped from cache
```

Token layout (Editing): source tokens + reference tokens + noisy target tokens
are concatenated; condition-specific RoPE encodings give each stream its own
spatial/temporal positions. Reference tokens are shared across all segments
(Avatar: same role for r — identity anchor visible everywhere).

### 3.3 Forward pass / pipeline

**Avatar, inference:** user text/audio → (VLM agent → structured prompt:
identity/appearance, expression, gaze, pose, actions, held objects) → backbone
denoises the next low-res latent chunk causally (conditioned on r, c_i, noisy
history cache with sink blocks + sliding window) → Refiner one-step upsamples
to 720p latents → VAE decoder → pixels → (optional stereo conversion) → user.
Reference r can be replaced between chunks; the VLM translates the new
reference into the prompt ("now holding the blue mug") with *state persistence*
(once "pick up the cup" completes, later prompts keep "holding the cup" until
told otherwise; verified by visual feedback on generated frames).

**SRF, training (the paper's central procedure):**
1. **On-policy rollout.** The student runs its real inference procedure
   autoregressively for a long window. All generated blocks *and their KV
   caches* are detached — no computation graph is retained.
2. **Re-noise.** Sample per-block timesteps t = (t_1,…,t_N); re-noise each
   block of the self-generated trajectory independently (Diffusion Forcing
   style). External history before the window stays fixed.
3. **Gradient-enabled replay.** One forward pass of the causal model over the
   whole re-noised trajectory, gradients on. Block i attends to the
   *recomputed* representations of blocks 1..i−1 — one connected graph.
4. **Loss on all replayed blocks** (DMD against the frozen bidirectional
   teacher + perceptual regularization). Later blocks' losses backprop into how
   earlier blocks are encoded; the rollout itself is never backpropagated
   through.

**Editing, inference:** source frame s_i arrives → VAE-encodes into condition
tokens → backbone generates edited target t_i (attending to s_i and the
reference) → Refiner → decoder → out. Each source frame is consumed with the
target it produces and dropped; caches stay O(window).

**Spatial conversion:** per-frame monocular depth (temporally stabilized) →
horizontal disparity → treat frame as center view, warp left and right →
lightweight hole fill at disocclusions + artifact mitigation at depth
discontinuities → emit synchronized L/R chunks. For stereo input editing: join
L/R into one wide stream, edit jointly with the same references, split back.

### 3.4 Loss function

- **Bidirectional stage:** standard diffusion denoising loss (predicted clean
  latents vs. ground truth), with *segment-wise* conditions c_i rather than one
  prompt per clip; then diffusion DPO for preference alignment.
- **SRF stage (Eq. 4):** L_SRF = L_DMD(f_causal(x_t^{1:N}, t, r, c^{1:N}))
  + L_perc(f_causal(x_t^{1:N}, t, r, c^{1:N})). L_DMD is distribution matching
  distillation: the student's replayed outputs are pulled toward the
  distribution of the frozen bidirectional teacher; L_perc (perceptual,
  carried over from Vidu S1) fights DMD's mode collapse.
- **Streaming NFT stage:** reward-based (negative-aware) fine-tuning on the
  same detached-rollout-then-replay trajectories.
- **Hybrid forcing stage (before SRF):** probability-mixed Teacher Forcing
  (clean GT history) and Diffusion Forcing (history at sampled noise levels).

## 4. Math

Notation: x^i = latent of video–audio segment i; subscript t = noise level
(their convention: t=0 clean, t=1 pure noise); r = reference image; c_i =
per-segment condition; f_θ = DiT predicting clean latents; U = latent-space
spatial upsampling; N = number of segments in a training window.

**(1) Avatar joint denoising (bidirectional form).**
x̂_0^{1:N} = f_θ^avatar(x_t^{1:N}, t, r, c^{1:N}).
Plain English: given noisy latents for all N segments at one shared noise
level, plus the reference image and every segment's condition, predict all
clean latents at once. r is shared across segments — that sharing is what
makes identity stable.

**(2) Joint I2V + R2V training.**
x̂_0^{1:N} = f_θ^bi(x_t^{1:N}, t, r, c^{1:N}), where r = first frame (I2V) or
user reference (R2V) — the *same* model handles both by reinterpreting r.
Segment-wise supervision with c_i (not one global prompt) is what improves
instruction following.

**(3) Causal segment denoising (the streaming form).**
x̂_0^i = f_θ^causal(x_t^i, t_i, r, c_i, x_{τ_i}^{<i}).
Plain English: segment i is denoised from its own noise level t_i, conditioned
on the reference, its own condition, and its *history* x^{<i} rendered at
per-history noise level τ_i. Block-causal mask: no attention to future
segments. τ_i is what Diffusion Forcing randomizes during training so the
model tolerates imperfect history.

**(4) SRF objective.**
L_SRF = L_DMD(f_θ^causal(x_t^{1:N}, t, r, c^{1:N})) + L_perc(same input),
where x_t^{1:N} is the *re-noised student rollout* (per-block t) and the
forward is the single gradient-enabled replay pass. The gradient structure —
not the equation — is the contribution: ∂L^block_N/∂θ flows through the KV
representations of blocks 1..N−1 recomputed inside the replay.

**(5) Refiner re-noising.**
z_{t_r}^i = (1−t_r)·U(x̂_0,LR^i) + t_r·ε, ε ~ N(0, I).
Plain English: upsample the backbone's clean low-res latent estimate to high
res, add noise back up to a *fixed* refinement level t_r, and let the Refiner
one-step predict the high-res clean latent. Asymmetric caches: backbone reads
history at high noise τ_B, Refiner reads history at low noise τ_R, τ_B > τ_R,
both fixed across segments within a replayed trajectory.

**(6) Editing model.**
x̂_0^{1:N} = f_θ^edit(x_t^{1:N}, t, s^{1:N}, r, c^{1:N}) with s^{1:N} the
source-video condition. Constraints live in the attention mask, not the
equation: target↔source only at equal timestamps; reference visible to all.

## 5. Training

**Data (Avatar).** S1's five-stage pipeline (Clipping, Filtering, Speech
Processing, Captioning, Embedding) plus:
- *Sources:* livestream/talking-head, film/TV (S1) + **solo dance** and
  **2D/3D animation** (for motion range and animated characters).
- *Cut-point detection:* shot detector proposes candidates; a VLM re-checks
  frames around each candidate cut (vlog jump-cuts are subtle); false
  rejection < 2%.
- *High-clarity selection:* hard thresholds on resolution/fps, then a weighted
  quality score over codec, pixel format, bit depth, bitrate, texture detail,
  edge sharpness, compression artifacts (nominal resolution alone is a bad
  proxy — a crushed 1080p is worse than a clean 720p).
- *Background stabilization:* keep smooth camera motion instead of discarding
  it (dance videos orbit/dolly/zoom); mask the subject, fit a global geometric
  transform to background correspondences, warp + crop, re-filter for framing.
  Justified by the observation that dance-camera parallax is mostly
  rotation/focal change, i.e. approximately a homography-like warp of the
  background.
- *Captioning:* **temporally ordered dense captions** — chronological events
  with time boundaries ("between 2.5 s and 3.8 s she raises her right hand…"),
  replacing S1's structured fields; chunk boundaries follow *event*
  boundaries, not fixed durations → finer chunks, lower response latency,
  cleaner action transitions. Multi-agent annotation (recognition /
  description / verification / filtering) on top of detection + ASR grounding.

**Data (Editing).** 800K curated videos → 4 mutually disjoint 200K subsets
(style transfer, subject replacement, background replacement, virtual try-on).
- *Style-transfer pairs:* train a conditional V2V model to *reconstruct* the
  raw video from (NormalCrafter surface normals + one sampled frame as
  reference); at generation time swap the reference for a stylized image →
  stylized video with the original structure/motion. Generalizes to 50+
  styles (cel shading, cyberpunk, Monet, sketch, gongbi).
- *Other tasks:* task-specific editors + open models (Bernini, SCAIL-2,
  Wan2.1 VACE, SAMA-14B, CoinVE-Edit), then post-filtering and *comparative
  selection* across models (no single model wins everywhere).
- *Captions:* written WITHOUT the source video shown to the captioner VLM —
  the instruction must not leak source content into text conditioning.

**Training stages (both models).** (1) bidirectional base with segment-wise
supervision (+ diffusion DPO for Avatar); (2) block-causal conversion with
hybrid Teacher/Diffusion Forcing; (3) SRF (on-policy DMD + perceptual) and
Streaming NFT; (4) latent-space Refiner (Avatar). S1 machinery retained
throughout: sink blocks, sliding-window context, noisy KV caches.

**Compute budget:** not disclosed. No parameter counts, no step counts, no
data volumes for Avatar.

## 6. Results & Ablations

- **StreamAV-Bench (digital character; Table 1):** Vidu S2-Avatar best on all
  nine metrics — VA .687, VQ 3.370, PQ 7.138, AQ 3.286, AVAlign .353,
  AVSync .617 (lower better), AIF 2.985, SC .998, BC .993. Runner-up Live
  Avatar: VA .661, VQ 3.295. SC/BC near ceiling = long-horizon identity and
  background stability. (Note: the plain-text PDF extraction scrambles the
  last three columns of Table 1; the `-layout` extraction has the correct
  alignment.)
- **Sparkle-Bench (editing; Table 2):** Overall 3.74 vs Decart-Lucy2.5 3.67
  (best streaming baseline) and Kiwi-Edit 5B 3.57 (best offline); instruction
  4.00, foreground motion 4.00.
- **OpenVE + RefVIE (Table 3):** Joint Overall 4.26 vs Bernini-R 14B 3.92
  (+0.34); OpenVE Global Style 4.71, Background Change 4.14.
- **ViViD unpaired try-on (Table 4):** VFIDI 9.9515 vs CatV2 TON 19.5131 and
  ViViD 21.8032 — roughly a 2× distributional-gap reduction.
- **Human GSB (internal):** vs Runway Character GWM-1 preferred 85.7%
  (overall), vs PixVerse Image Avatar and HeyGen 100%. Editing: 150 paired
  cases, mean consistency 3.56 vs Decart-Lucy2.5 2.59 vs XMax-X2.0 1.67.
- **Duration-stratified (10→90 s):** highest mean 1–5 ratings at every
  duration on all five dimensions (consistency, video quality, motion,
  emotional expression, overall) — evidence against long-horizon decay.
- **Ablations: none.** No SRF-vs-Self-Forcing, no frame-aligned-vs-free, no
  τ_B/τ_R sensitivity, no Refiner-on/off. The evaluation protocol itself
  (identical manifests, published numbers only when benchmark+split+metric
  match, recorded seeds/versions) is the most rigorous part.

## 7. Limitations

- **Proprietary.** No weights, no code, no architecture sizes, no compute
  numbers; the central mechanisms are described but not isolated by ablation.
- **Spatial video is explicitly a feasibility study.** Authors themselves name
  the gaps: headset FOV demands higher resolution than conventional real-time
  monocular; passthrough latency must not break head-motion sync. Fixed-view
  only — no panoramic/free exploration yet (named future work, with a
  "real-time generated metaverse" vision).
- **Warp-based stereo synthesis** inherits monocular depth failure modes;
  hole-fill and depth stabilization are "lightweight" by latency necessity.
- **Agentic reliability layer admits the model alone is not enough** for
  reference interaction; the VLM wrapper (prompt state persistence, visual
  verification) is doing real work that isn't benchmarked separately.
- Latency is asserted via FPS ranges (25–42 FPS @ 720p) and eval conditions
  (160/320/480 ms audio chunks, interruption/recovery, 5-min stability tests)
  but no end-to-end latency table appears in the paper text.

## 8. Open Questions / Ideas

- SRF's gradient-flow claim is testable at toy scale: does replay-with-grad
  beat detached Self-Forcing on a synthetic streaming task where later chunks
  depend on earlier ones? (This is exactly what our miniature implementation
  probes.)
- Frame-aligned attention is a *hard* alignment prior. What happens to
  temporal re-timing tasks (speed ramps, pauses) where the edited stream
  *should not* keep the source's timing?
- The asymmetric τ_B/τ_R cache split is a clean instance of a broader pattern:
  different consumers of the same history want it at different noise levels.
  Does a three-level split (motion / appearance / audio) help further?
- Dynamic reference injection as pure conditioning implies appearance
  transitions take one chunk latency. Is a cross-fade schedule (reference
  blended over k chunks) learnable, and does it reduce popping?
- The normals→V2V style-pair construction is a general recipe for
  structure-preserving edit data (reconstruct-then-substitute). Portable to
  any conditional backbone, not just styles.
- Panoramic spatial streaming (their future work) collides with the sliding
  KV window: turning your head rewinds time for off-screen content. What is
  the cache coherence story there?
