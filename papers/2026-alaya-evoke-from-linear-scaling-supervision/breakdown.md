# Breakdown — Alaya-EVOKE: From Linear-Scaling Supervision to Endless World

> **Paper:** Alaya-EVOKE: From Linear-Scaling Supervision to Endless World
> **Authors:** Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuanhao Li, Kaipeng Zhang, Feng Zhao
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2608.13546
> **Code (official):** none released; project page https://evoke-world.github.io/Evoke/

---

## 1. Problem & Motivation

**The problem.** An *interactive world model* is a video diffusion model that generates
video continuously and reacts to user actions (camera moves, prompt changes) mid-stream.
Real interactive sessions — gameplay, drone exploration, simulation — are effectively
*endless* (tens of minutes to hours). Two structural obstacles block existing models
from supporting such sessions:

1. **Memory vs. cost.** Long-range visual coherence requires the model to "remember"
   the scene. The two standard mechanisms — concatenating past frames into the
   denoiser's context, or keeping a growing key-value cache — both scale *with session
   length*: attention cost grows quadratically with the concatenated sequence, KV
   caches grow linearly, and positional spans explode. Practitioners are forced to
   trade session length against retained memory; "endless" sessions are impossible.

2. **Few-step students are bounded by their teacher's horizon.** Interactive use needs
   low latency, so the deployed model is a few-step (here 3-step) student distilled
   from a multistep teacher. A student distilled with a *short* supervision window
   cannot learn dynamics beyond that window: any drift mode not expressible in
   window-sized statistics receives zero training gradient. If the teacher itself only
   ever scores short windows, the student inherits short-horizon blindness no matter
   how it is sampled.

**Why it matters.** Interactive world models are the rendering engine of open-ended
agent environments, games, and simulation. Without bounded per-step cost and
long-horizon stability, sessions must be capped at a few minutes and degrade
progressively (exposure shift, texture decay, scene-identity morphing).

**Prior approaches and their limits.**
- *Context concatenation / autoregressive video models* (e.g. long-video DiTs):
  quadratic attention growth; quality degrades as context lengthens.
- *KV-cache compression*: still grows with history; drift accumulates.
- *Distribution-matching distillation* (DMD/SiD family): gives few-step students, but
  only over short horizons — the failure mode is inherited, not solved.
- *Game-model memory* (Genie-style): learned memory, expensive to train, rarely
  revisitable; recall is not guaranteed.

## 2. Key Insight / Contribution

**Core idea (3 sentences).** Split what "memory" must do into two parts with different
owners: *temporal evolution* (how the scene changes moment to moment) belongs to a
teacher redesigned for linear-scaling long-horizon supervision, and *spatial recall*
(what was seen when you look back) belongs to an external, camera-indexed bank of
scene geometry that is read by pure pose-based retrieval. The student is a 3-step,
CFG-free chunk generator whose context is bounded by construction, so session length
only adds recurrent calls — never cost per call. A 30-second distribution-matching
objective under self-forced rollouts transfers the teacher's long-horizon stability
into the student.

**What is genuinely new.**
1. The **recurrent session formulation** (Eq. 1) with *fixed budgets* for both the
   denoiser-side history `h_k` and the external world-state bank `M_k` — per-step
   compute is provably independent of session length.
2. The **causal diagnosis of drift** into *degradation drift* (photometric; cause
   precedes the window) vs. *content drift* (identity; only identifiable when distant
   moments are jointly supervised) — with the observation that a longer window fixes
   the former through rollout-distribution robustness while the latter additionally
   needs revisit *evidence*, which only the bank can supply.
3. **Supervision horizon ≠ gradient horizon**: the teacher+critic score the entire
   31.4 s self-forced rollout jointly, but inter-chunk history is *detached* so each
   chunk's backward graph is chunk-sized. Long-range signal without long-range
   backprop.
4. **Teacher/critic in one backbone** (LoRA toggle) sharing the same mixture-of-experts
   routing, plus **per-chunk text conditioning** that makes mid-session prompt
   changes trainable and exposes "evocation" (timed prompt schedule) at inference.

## 3. Method

### 3.1 Overview

A session is a bounded recurrent process over 1.5-s chunks. Each recurrent call:
render geometry relevant to the current camera from the bank → generate the next
chunk from (render, local history, text) → write the new observation's geometry back
into the bank. Both the local history and the bank have *fixed capacity budgets*, so
context length, positional span, and compute of one recurrent call are independent of
the step index `k`. An "endless" session is just unboundedly many bounded calls.

```
r_k     = Read(M_k, P_k)               # render geometry relevant to current view
x_k    ~ p_theta(. | r_k, h_k, c_k)    # generate chunk (3-step student, no CFG)
M_(k+1) = Write(M_k, x_k, P_k)         # lift new observation into world space
h_(k+1) = bounded_update(h_k, x_k)     # sliding tiered history, fixed size
```

Chunk units: **F = 9 latent frames = 36 pixel frames = 1.5 s @ 24 fps**. `P_k` is the
camera trajectory (extrinsics + intrinsics at 24 fps) for chunk `k`; `c_k` is a text
condition that may change at *every* recurrent step.

### 3.2 Architecture

```
                         SESSION (k = 1 .. K, K unbounded)
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │   CAMERA P_k ──►┌─────────────┐                                      │
 │                 │    Read     │──► r_k (warped view-aligned          │
 │   WORLD BANK ──►│ co-vis rank │      observation + visibility mask)  │
 │   M_k (fixed    │ + z-buffer  │              │                      │
 │   retention     └─────────────┘              ▼                      │
 │   budget)                              ┌──────────────┐   text c_k   │
 │        ▲                               │ 3-STEP       │─────────────►│
 │        │                               │ CFG-FREE     │              │
 │        │        h_k (19 latent fr,     │ DENOISER     │              │
 │        │        tiered 16/2/1) ──────► │ coarse→fine  │              │
 │        │                               │ pyramid      │              │
 │        │                               └──────┬───────┘              │
 │        │                                      │ x_k                  │
 │        │                                      ▼                      │
 │   ┌────┴─────────┐   depth    ┌──────────────┐                       │
 │   │    Write     │◄───────────│ mono depth + │                       │
 │   │ (unproject,  │            │ unproject    │                       │
 │   │  no fusion)  │            └──────────────┘                       │
 │   └──────────────┘                                                   │
 └──────────────────────────────────────────────────────────────────────┘

 TEACHER (training only) — Wan2.2 A14B DiT, high/low-noise experts kept,
 one backbone, LoRA ON = critic, LoRA OFF = teacher.
 Sparse Chunk Attention over the whole rollout (21 chunks × 9 latent fr):

   query chunk ──►┬─ first-frame global sink        (1 latent frame)
                  ├─ local chunks (1-frame overlap)
                  ├─ spatially compressed nearby    (3 latent frames)
                  ├─ importance-selected distant    (M frames)
                  └─ linear-attention global state  (S summary tokens)
   ⇒ attention cost ~ linear in sequence length.
```

**Denoiser-side history (tiny by design):** only the most recent **19 latent frames
(~3.2 s)**, tiered long/mid/short = 16/2/1 frames. Everything older exists *only* in
the bank. Unsupported (visibility < 0.5) history tokens are *removed* from the
sequence at patch resolution, pooled per tier.

**World state bank:** frame-level point geometry. Each `Write` estimates monocular
depth for 12 frames of the generated chunk (with the known camera trajectory),
unprojects with intrinsics/extrinsics, and appends — **no cross-chunk fusion**,
which deliberately avoids scale drift and fusion artifacts over long sessions.
`Read` ranks stored source views by **co-visibility** with the target camera pose,
selects up to **8 sufficiently distinct sources**, and renders them with batched
projection + **z-buffering** into a warped view-aligned observation with a per-pixel
visibility mask. Retrieval is purely geometric — no learned retrieval — because the
content was already observed; this is recall, not inference.

**Visibility gates everything:** unsupported pixels (visibility < 0.5) receive noise
σ = 1 (no visual information); supported regions receive σ ∈ [0, 0.135].

**Retention budget (hour-scale config):** 2160 pixel frames = **90 s of geometry**,
every 3rd frame ingested ⇒ ≤ 720 active source frames. The guarantee is "persistent
recall *within retained coverage*" — not permanent recall of everything ever seen.
Once revisits exceed 90 s away, recall decays to the far-pose floor.

**Student sampler:** 3 CFG-free denoising evaluations over a **coarse-to-fine latent
pyramid** 12×20 → 24×40 → 48×80, one evaluation per level. Geometric conditioning is
injected **only at the coarsest stage** (establishes layout; finer stages refine
appearance), bounding conditioning cost by warp coverage rather than session length.

### 3.3 Forward pass / pipeline (one training rollout)

1. Sample a ground-truth prefix chunk `x0` from data; run the student (3 NFEs) to
   produce 20 self-forced chunks `x1..x20` — the student conditions on its *own*
   history throughout (on-policy rollout). Total: 21 × 9 = **189 latent frames ≈
   31.4 s**; the student executes 60 forward passes.
2. Score ALL 189 frames *jointly* with teacher (LoRA off) → `s_real`, and critic
   (LoRA on) → `s_fake`.
3. Compute the distribution-matching gradient (Eq. 3 below) on the masked region Ω
   (everything except the GT prefix and the first generated chunk `x1`).
4. Backprop per-chunk: history is **detached** between consecutive rollout chunks, so
   each chunk's backward graph is independent (activation memory stays chunk-sized)
   while the scoring window remains the full 31.4 s.
5. Text conditions are assigned **per chunk**: training captions are segmented at 12-s
   intervals and mapped onto latent chunks; each rollout chunk receives the text of
   its temporal position. Prompt transitions live inside one training sequence. At
   inference the same representation is exposed as a **timed prompt schedule**
   ("evocation").

Training stages overall (student is built on Helios): (1) camera control,
(2) few-step distillation, (3) long-horizon distillation (+ a short post-distill
continuation that yields the released checkpoint).

### 3.4 Loss function

Distribution matching (DMD-style) + auxiliary warp-conditioning loss:

- `Δs = s_fake − s_real` — the direction that moves student samples from the
  critic's "fake" score field toward the teacher/data score field.
- `ν = mean_Ω |x̂0 − s_real|` — data-scaled normalizer over the masked region.
- `L_gen = ½ E[(x̂0 − detach(x̂0 − Δs/ν))²]` — gradient wrt `x̂0` is `−Δs/ν`; the
  inner `detach` turns the direction into a fixed L2 regression target.
- **Mask Ω** excludes the GT prefix and the first generated chunk `x1`: at that
  boundary the teacher models the first chunk with its image-model first-frame
  distribution while the student does video continuation; matching there causes
  boundary flicker. Gradient applies to `x2..x20`; `ν` is computed over the same
  mask so excluded frames don't rescale the update.
- The **warp-conditioning loss** (camera controllability, carried over from the
  camera-control training stage) is kept alongside — distribution matching alone does
  not enforce trajectory adherence.

## 4. Math

**Eq. 1 — recurrent session.**
`r_k = Read(M_k, P_k);  x_k ~ p_θ(·|r_k, h_k, c_k);  M_{k+1} = Write(M_k, x_k, P_k)`
- `M_k`: world-state bank at step k (fixed retention budget); `P_k`: camera
  trajectory of chunk k; `r_k`: rendered, view-aligned geometric observation +
  visibility mask; `h_k`: bounded local history (≤ 19 latent frames); `c_k`: chunk
  text condition.
- Plain English: render what the current camera should see from stored geometry,
  generate the next 1.5 s from that render plus recent history plus text, then store
  the new observation's geometry back. Because `h` and `M` have fixed budgets, the
  cost of *one* call never grows with k.

**Eq. 2 — window objective (why teachers bound students).**
`L_W(θ) = E_k [ D( q_θ^{k:k+W−1}, p^{k:k+W−1} ) ]`
- `q_θ`: the student's trajectory distribution over a window of `W` consecutive
  chunks; `p`: the data (teacher-supervised) distribution over the same window; `D`:
  divergence whose *gradient* is estimated from teacher + critic scores.
- Plain English: training only shapes window-sized statistics. Drift modes that need
  longer evidence to identify (content drift) or whose causes precede the window
  (degradation drift) get **zero gradient** — the formal statement of "a student
  can't exceed its teacher's horizon." Evoke uses W ≈ 30 s (21 chunks).

**Eq. 3 — distribution-matching loss.**
`Δs = s_fake − s_real`, `ν = mean_Ω |x̂0 − s_real|`,
`L_gen = ½ E[(x̂0 − detach(x̂0 − Δs/ν))²]`
- `x̂0`: the student's one-step denoised prediction; `s_real`: teacher score (data
  side); `s_fake`: critic score (student side); `ν`: per-batch normalizer making the
  step size data-scaled; Ω: mask excluding the GT prefix and chunk `x1`.
- Plain English: regress the student's prediction toward (itself + the
  critic-to-teacher score difference, rescaled). The detach makes the target fixed,
  so the gradient is exactly `−Δs/ν` — push samples where the critic disagrees with
  the teacher, in the amount the data itself varies.

**Eq. 4 — deployment factorization the teacher must match.**
`p_θ(x_k | x_{<k}, P_k, c_k)` with `c_k` varying per step.
- Plain English: at inference the text condition changes at every recurrent step; a
  teacher trained under one global prompt supervises a fixed-condition distribution
  and never demonstrates how a *newly introduced* instruction should affect an
  ongoing rollout. Hence per-chunk conditioning in training = the "evocation"
  interface at inference.

**Drift taxonomy (the conceptual theorem).**
- *Degradation drift*: photometric decay whose cause precedes any W-window; under
  self-forced rollouts the student is supervised on its own perturbed histories, and
  larger W ⇒ wider distribution of rollout-perturbed histories ⇒ robustness to
  accumulated conditioning shift.
- *Content drift*: scene identity/layout morphs while every short window looks
  locally plausible; identifiable only when sufficiently distant moments are jointly
  inside the window — and correctable only where the bank holds the observation.
- Division of labor: **temporal evolution → long-horizon teacher supervision;
  spatial revisit → world state bank.** A longer window cannot resurrect an
  observation that left context; stored geometry cannot decide how never-observed
  content should evolve.

## 5. Training

- **Base model:** Wan2.2 A14B (14B DiT) teacher keeping its high-noise/low-noise
  experts (routed by sampled timestep); student built on Helios.
- **Teacher/critic sharing:** one backbone; LoRA adapter enabled = critic, disabled =
  teacher ⇒ both scores always come from the same expert (no distribution mismatch).
- **Rollout structure:** 1 GT prefix chunk + 20 self-forced student chunks; 189
  latent frames ≈ 31.4 s; student 3 NFEs/chunk.
- **Distillation schedule:** long-distill 6 × 8 GPUs (48 GPUs), 1981 steps, eight
  scheduler restarts, then a short post-distill continuation that yields the released
  student (released at the onset of the normalized-gradient plateau).
- **Systems tricks:** sequence parallelism, activation recomputation, chunk-level
  backward graphs (detached inter-chunk history).
- **Text conditioning:** captions segmented at 12-s intervals mapped to latent chunks.
- **Auxiliary:** warp-conditioning (camera control) loss retained during
  distillation.
- **Inference cost:** 2.11 s diffusion wall-clock per 1.5-s chunk at 384×640 on one
  H200 (full VAE decode; no KV caching / compilation / quantization tricks). Geometry
  path adds ~1.84 s/chunk, scaling with warp coverage.
- (Dataset specifics for the underlying video model are inherited from Helios/Wan
  training; the paper's tables do not re-specify them.)

## 6. Results & Ablations

**Headline numbers.**
- **WBench navigation split** (n = 158): best quality average among 9 systems
  (incl. Genie 3, LingBot-World v2, HY-GameCraft) while sampling in 3 CFG-free steps.
- **VBench-2.0: 66.77 — rank 1 of 10** (Veo 3 nearest at 66.72).
- **VBench-Long: 85.11 — rank 7 of 10** (leader IPOW 88.26); peers run many-step
  default samplers, so not step-matched — honestly flagged.
- **Hour-scale stability:** 8 sessions × 65.5 min × 2619 chunks × 94,281 frames;
  photometric stats flat after a transient; scene-identity cosine plateaus at 0.523 —
  the same level real footage scores against itself 60 s apart (a *stability* claim,
  not a fidelity claim). Per-step cost stays flat over the hour (§4.2), and the
  active geometric source pool stops growing once the retention budget fills.

**Ablations that matter most.**
1. **Long- vs short-horizon teacher (Fig. 6, controlled):** identical distillation
   recipes; teachers differ only in horizon. Short-horizon student settles at 74% of
   opening brightness; long-horizon at 101% (7/8 clips improve, Wilcoxon p = 0.016).
   Brightness separates most; sharpness not at all; content descriptors n.s. ⇒ the
   transferred benefit is *photometric stability*. This is the direct evidence that
   supervision horizon, not architecture, drives drift resistance.
2. **Geometric recall (pose-addressed):** revisit PSNR between two 12-s windows at
   identical poses; 20/21 comparisons follow the predicted transition — recall
   improves iff the retention window ≥ time away; +2.3–3.2 dB when covered; plateau
   15.4–17.8 dB (recognizable, not pixel-faithful).
3. **Timed text control ("evocation"):** a mid-session clause realizes in 67% of
   cases targeting *unanchored* content (ceiling 83%) but only 4% when it would
   require *overwriting geometry already anchored in the bank* (floor 17%) ⇒ **text
   governs free content; stored geometry resists overwriting.** Anchor-pause
   ablation n.s.; short vs long teacher doesn't separate realization rates.
4. **Appendix C (detectability sweeps):** teacher–critic detectability changes little
   beyond W = 2 chunks across 13 perturbation conditions; single-evaluation noise
   dominates systematic drift (recoverable only by aggregation over 5,749 logged
   steps) ⇒ the W ≈ 30 s benefit is a property of the *whole* trained long-horizon
   teacher + rollout distribution, not of window length alone.

## 7. Limitations

Authors' own:
1. The bank preserves only **coarse scene structure**; fine object identity /
   appearance consistency is limited.
2. **No dynamic state**: object motion, state transitions, and their long-term
   evolution are not represented (geometry is static scaffolding).
3. Real-time interaction still needs faster VAE / fewer-step generators / cheaper
   geometric conditioning (2.11 s/chunk is not yet real-time for 1.5 s of video).

Mine (from the close read):
4. Recall is bounded by the 90-s retention budget — recall beyond it decays to the
   far-pose floor; "endless" applies to *session length*, not *total remembered
   content*.
5. Revisit PSNR plateau (15–18 dB) shows recall is semantic, not photographic.
6. VBench-Long rank 7 shows the 3-step budget costs raw quality vs many-step peers.

## 8. Open Questions / Ideas

- The **size S of the linear-attention global state** and the count M of selected
  distant frames are not fully specified — how sensitive is long-window scoring to
  these? A toy sweep would tell.
- Could the retention budget be made **importance-weighted** (keep poses likely to be
  revisited by the trajectory prior) instead of a FIFO 90-s ring?
- The bank stores *static* geometry. What is the minimal extension that stores
  *dynamic* state (object poses with timestamps) without reintroducing fusion drift?
- Detached-history chunk gradients + full-window scoring is a general recipe — does
  it transfer to other recurrent distillation setups (e.g. policy distillation for
  agents)?
- Toy implementation plan (for `implementation/`): (1) recurrent chunk loop with
  fixed-shape per-step cost — measure cost vs session length, show flat; (2)
  camera-indexed bank with write/unproject, read/co-visibility + z-buffer render +
  visibility mask on a synthetic world — show revisit recall iff retention ≥
  time-away; (3) dense vs sparse (sink + local + selected + linear-state) attention —
  verify linear scaling; (4) miniature DMD-style distillation with self-forced
  rollouts, detached inter-chunk history, short- vs long-horizon teacher ablation —
  show photometric-drift resistance grows with W.

---
*Breakdown written 2026-10-01 from `paper.txt` (arXiv:2608.13546) and two reading
passes recorded in `notes.md`.*
