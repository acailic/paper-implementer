# Alaya-EVOKE: From Linear-Scaling Supervision to Endless World

> **Paper:** Alaya-EVOKE: From Linear-Scaling Supervision to Endless World
> **Authors:** Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuanhao Li, Kaipeng Zhang, Feng Zhao
> **arXiv:** 2608.13546 — https://arxiv.org/abs/2608.13546
> **Project page:** https://evoke-world.github.io/Evoke/

---

## First impressions (pass 1)

Picked from the tracker (rank 9 in queue). PDF fetched and text-extracted to
`paper.txt`. Skimmed the whole thing. Below are my rough first impressions and
the big picture.

### What problem is this solving?

Interactive world models (video diffusion models that generate video
continuously and respond to user actions/prompts mid-generation) face a
fundamental tension:

1. **Memory vs. cost.** To generate a coherent long video, the model needs to
   "remember" the scene history. The standard ways of keeping history —
   concatenating past frames into the denoiser's context or keeping a growing
   KV cache — get more expensive as the session gets longer (quadratic /
   linearly-growing cost). So there is a trade-off between how long a session
   can run and how much memory the model retains. Real interactive sessions
   are "endless" — hours of gameplay, drone exploration etc. — and current
   models can't handle that with in-context history.

2. **Few-step student vs. its teacher.** Interactive applications need low
   latency, so you want a few-step (e.g. 3-step) generator. But a few-step
   student distilled from a multistep teacher typically can't exceed the
   knowledge of its teacher. If the teacher itself can't generate long-horizon
   coherent video, the student will drift no matter how you distill.

### Core idea (in my own words)

EVOKE solves both with one combined move:

- **Externalized world state.** Instead of keeping all history frames in
  context, scene geometry is stored in an external, *camera-indexed* world
  state bank. At each generation step, only the *view-relevant* entries are
  retrieved (like retrieval-augmented generation, but for geometry — the bank
  keys are camera viewpoints). The denoiser context stays bounded no matter
  how long the session runs.

- **Teacher redesigned for long-horizon supervision.** The teacher is not a
  fixed multistep generator. Its attention is sparse and three-way:
  chunk-wise grouping (nearby frames attend to each other), retrieval of
  selected distant frames (retrieval attention over the world state bank),
  and a linear-attention global state (cheap global context). Memory and
  compute grow **linearly** with session length, and it can supervise over
  long horizons. This teacher can "see" content drift that stays locally
  plausible within short windows but breaks global consistency.

- **30-second distribution-matching distillation.** A teacher that can
  generate 30s+ of coherent video distills into a 3-step student using a
  distribution-matching objective (match distributions, not point-wise
  trajectories) under **self-forced rollouts** (the student rolls out its own
  states and the teacher supervises on those states — on-policy distillation,
  like DMD/SiD-style but for interactive world models).

Result: a 3-step student world model with bounded context and recurrent
external memory that supports open-ended interactive generation, SOTA on
WBench, 2.11s per 1.5s chunk on one H200 at 384×640.

---

## Second pass (deep read) — 2026-10-01

Re-read section by section (§3.1–§3.5, §4.1–§4.5, Appendices A–E). Notes
below resolve most of the open questions from pass 1.

### §3.1 Recurrent session formulation (the exact loop)

A session is a bounded recurrent process over chunks. Notation:

- Chunk `x_k`: **F = 9 latent frames** = 36 pixel frames = **1.5 s @ 24 fps**.
- `P_k`: camera trajectory for chunk k (camera-to-world extrinsics +
  intrinsics, resampled to 24 fps).
- `c_k`: text condition, **may change at every recurrent step**.
- `h_k`: bounded local history (denoiser-side).
- `M_k`: world state bank (external).

The loop (Eq. 1):

```
r_k     = Read(M_k, P_k)              # render geometry relevant to current view
x_k     ~ p_theta(. | r_k, h_k, c_k)  # generate chunk from render + history + text
M_(k+1) = Write(M_k, x_k, P_k)        # lift new observation into world space
```

Key property: **both** `h_k` and `M_k` operate under fixed budgets, so the
context length, positional span, and compute of *one recurrent call* are
independent of k. Extending a session only adds more recurrent calls.
(Verified empirically in §4.2: cost per step stays flat over 65.5-min
sessions; the active geometric source pool stops growing once the retention
budget fills.)

### §3.2 Long-horizon supervision (the theory core)

The window objective (Eq. 2):

```
L_W(theta) = E_k [ D( q_theta^(k:k+W-1), p^(k:k+W-1) ) ]
```

i.e. a divergence D between the student's trajectory distribution over a
window of W consecutive chunks and the data distribution over the same
window; D's *gradient* is estimated from teacher + critic scores (DMD-style).
Anything not expressible in W-window statistics gets **zero gradient** — that
is the formal statement of "the student can't exceed its teacher's horizon."

Two distinct failure modes, with different fixes:

1. **Degradation drift** (photometric: exposure shift, saturation creep,
   texture decay). Eventually visible in a local window, but its *cause*
   precedes the window. Under self-forced rollouts the student conditions on
   its own history, so early errors shift the conditioning distribution seen
   by later denoising steps. Larger W ⇒ wider distribution of
   rollout-perturbed histories the student is supervised on ⇒ robustness to
   accumulated conditioning shift.

2. **Content drift** (scene identity / object appearance / layout slowly
   morphing while every short window looks fine). Only identifiable when
   sufficiently distant moments are *jointly* inside the supervision window.

**The load-bearing division of labor:** temporal evolution → long-horizon
teacher supervision; spatial revisit → world state bank. A longer supervision
window cannot resurrect an observation that left context; stored geometry
cannot decide how never-observed content should evolve.

**Why W ≈ 30 s and not W = session length:** the goal is not to train
recovery from arbitrarily corrupted histories but to keep the student's
recurrent dynamics stable over perturbations encountered in *normal*
generation. As the student gets robust, its own rollouts stay in a
better-behaved region of trajectory space ⇒ diminishing returns on W. (The
appendix detectability sweeps complicate this: teacher–critic detectability
changes little beyond W = 2 chunks, so the benefit is attributed to the
*whole trained long-horizon teacher + rollout distribution*, not to window
length alone. Honest ablation.)

### §3.3 The teacher (architecture + distillation details)

**Backbone:** Wan2.2 A14B 14B DiT, keeping its **high-noise / low-noise
experts** selected by the sampled timestep. Crucially, **teacher and critic
share one backbone**: enabling a LoRA adapter ⇒ critic; disabling it ⇒
teacher. So both scores always come from the same expert — no teacher/critic
distribution mismatch.

**Sparse chunk attention** (this is what makes 60s supervision affordable).
Sequence partitioned into 9-latent-frame chunks. Each *query* chunk attends
to a bounded key set:

1. **first-frame global sink** (1 latent frame — an anchor token),
2. **local context** — neighboring chunks with 1-frame overlap,
3. **spatially compressed nearby frames** (3 latent frames),
4. **importance-selected distant frames** (M frames — the "retrieval" part),
5. **linear-attention global state** (S summary tokens — cheap global
   context accumulated across the whole sequence).

Bounded keys per query ⇒ attention cost ~linear in sequence length. The same
chunk partition gives **per-chunk text conditioning**: captions segmented at
12 s intervals and mapped onto their latent chunks, so prompt *transitions*
live inside one training sequence (this is what makes mid-session
instruction changes learnable — the "evocation" interface).

**Distillation (Eq. 3, the DMD loss, in my own symbols):**

- Rollout = 1 ground-truth prefix chunk `x0` + 20 self-forced student chunks
  `x1..x20` ⇒ 21 × 9 = **189 latent frames** = 753 pixel frames ≈ **31.4 s**.
  Student does 3 NFEs/chunk ⇒ 60 student forward passes per rollout.
- Scores: `s_real` = teacher prediction, `s_fake` = critic prediction.
- `Δs = s_fake − s_real`; normalizer `ν = mean_Ω |x̂0 − s_real|`.
- `L_gen = ½ E[(x̂0 − detach(x̂0 − Δs/ν))²]` — gradient wrt x̂0 is `−Δs/ν`,
  i.e. push the student sample in the direction that makes critic scores
  look like teacher/data scores. (Classic DMD trick: the normalizer makes the
  step size data-scaled; the inner detach turns a direction into an L2
  regression target.)
- **Mask Ω** excludes the GT prefix and the *first* generated chunk `x1`:
  at that boundary the teacher models the first chunk with its image-model
  first-frame distribution while the student does video continuation —
  matching there causes boundary flicker. Gradient applies to `x2..x20`;
  `ν` is computed over the same mask so excluded frames don't rescale the
  update.
- **Supervision horizon ≠ gradient horizon.** History is *detached* between
  consecutive rollout chunks: each chunk's backward graph is independent.
  Full 31.4-s window is still *scored* jointly, so each local update is
  derived from a full-rollout distribution discrepancy, but activation
  memory stays chunk-sized. Nice trick to steal.
- Auxiliary **warp-conditioning loss** (camera controllability) is kept from
  the earlier training stage; distribution matching alone doesn't enforce
  trajectory adherence.
- Training stages: (1) camera control, (2) few-step distillation, (3) long
  distillation (+ short post-distill continuation). Systems: sequence
  parallelism, activation recomputation, chunk-level backward graphs.

**Training stages for the student overall:** built on Helios; camera-control
training → few-step distillation → long-horizon distillation → post-distill
continuation.

### §3.4 Geometric world state (the bank, concretely)

- **Denoiser-side history is tiny:** only the most recent **19 latent frames
  (~3.2 s)**, tiered long/mid/short = 16/2/1 frames. Everything older exists
  *only* in the bank.
- **Write:** monocular depth model estimates depth for 12 frames of the
  generated chunk (under the known camera trajectory) → unproject with
  intrinsics/extrinsics → append to bank. **No cross-chunk fusion** —
  inserting geometry from different chunks independently avoids scale drift
  and fusion artifacts over long sequences.
- **Read:** current camera pose determines relevance. Stored source views
  ranked by **co-visibility** with the target view; up to **8 sufficiently
  distinct sources** selected; rendered via batched projection with
  **z-buffering** ⇒ (warped view-aligned observation, per-pixel visibility
  mask). Retrieval is purely geometric — no learned retrieval — which is
  right because the content was already observed; you're recalling, not
  inferring.
- **Visibility gates everything:**
  - unsupported pixels (visibility < 0.5) get noise σ = 1 ⇒ contribute no
    visual information; supported regions get σ ∈ [0, 0.135];
  - visibility pooled at patch resolution per history tier ⇒ unsupported
    *history tokens are removed* from the denoiser sequence.
- **Retention budget (hour-scale config):** 2160 pixel frames = **90 s of
  geometry**, every 3rd frame ingested ⇒ ≤ 720 active source frames. The
  guarantee is "persistent recall *within retained coverage*," NOT permanent
  recall of everything ever seen. Once revisits exceed 90 s away, recall
  decays to the far-pose floor.

### §3.5 Bounded 3-step inference

- 3 **CFG-free** denoising evaluations over a **coarse-to-fine latent
  pyramid** at 12×20 → 24×40 → 48×80 (one evaluation per level).
- Geometric conditioning injected **only at the coarsest stage** (establish
  layout; finer stages refine appearance) — this bounds conditioning cost by
  coverage, not session length.
- 2.11 s diffusion wall-clock per 1.5-s chunk @ 384×640 on one H200 (full
  VAE decode, no KV caching/compile/quantization tricks). Geometry path adds
  ~1.84 s/chunk, scaling with warp coverage.

### §4 Results that matter

- **WBench navigation split** (n = 158): Evoke leads on quality average
  among 9 systems (incl. Genie 3, LingBot-World v2, HY-GameCraft) while
  sampling in 3 CFG-free steps.
- **VBench-2.0: 66.77 — rank 1 of 10** (Veo 3 nearest at 66.72).
  **VBench-Long: 85.11 — rank 7 of 10** (leader IPOW 88.26). Honest note:
  peers run many-step default samplers, so not step-matched.
- **Hour-scale stability:** 8 sessions × 65.5 min × 2619 chunks × 94,281
  frames. Photometric stats stabilize after a transient then flat; scene
  identity cosine plateaus at **0.523 — exactly the level real footage
  scores against itself 60 s apart**. Framed as a *stability* claim
  (evidence against runaway degradation), not a fidelity claim. I like the
  honesty.
- **Long- vs short-horizon teacher (Fig. 6, controlled):** identical
  distillation recipes, teachers differ only in horizon. Short-horizon
  student settles at **74%** of opening brightness; long-horizon at **101%**
  (7/8 clips improve, Wilcoxon p = 0.016). Brightness separates most,
  sharpness not at all, content descriptor n.s. ⇒ the transferred benefit is
  *photometric stability*, not general consistency.
- **Geometric recall (pose-addressed):** revisit PSNR between two 12-s
  windows at identical poses. 20/21 comparisons follow the predicted
  transition (recall improves iff retention window ≥ time away; +2.3–3.2 dB
  when covered). Plateau 15.4–17.8 dB ⇒ recognizable, not pixel-faithful.
- **Timed text control ("evocation"):** a mid-session clause realizes in
  **67%** of cases when it targets *unanchored* content (ceiling 83%), but
  only **4%** when it would require *overwriting geometry already anchored by
  the bank* (floor 17%). So: **text governs free content; stored geometry
  resists overwriting.** Anchor-pause ablation n.s.; short vs long teacher
  doesn't separate realization rates ⇒ per-chunk conditioning is the
  *interface* for timed control, not a separately-causal factor at this
  sample size.
- **Appendix C:** teacher–critic detectability changes little beyond
  W = 2 chunks over 13 perturbation conditions; single-evaluation noise
  dominates systematic drift (recoverable only by aggregation over 5,749
  logged steps). Benefit ⇒ property of the whole trained long-horizon
  supervision process.

### §5 Limitations (authors' own)

1. Bank preserves only **coarse scene structure**; fine object identity /
   appearance consistency limited.
2. No **dynamic state**: object motion, state transitions, their long-term
   evolution aren't represented.
3. Real-time interaction still needs faster VAE / fewer-step generators /
   cheaper geometric conditioning.

### Answers to my pass-1 open questions

- **What's in a bank entry?** Frame-level point geometry: depth-unprojected
  pixels with known camera pose (not fused across chunks; z-buffered render
  at read time). Retrieval score = co-visibility between stored source pose
  and target pose; top-8 distinct sources.
- **Sparse attention exact structure:** sink + local(1-frame overlap) +
  compressed-nearby(3 latent fr) + selected-distant(M) + linear-state(S).
  Query = current 9-frame chunk. Linear cost in sequence length.
- **Distribution-matching loss:** DMD (Yin et al.), Eq. 3 above, with shared
  backbone teacher/critic (LoRA toggle), mask Ω excluding prefix + first
  chunk, per-mask normalizer ν.
- **Self-forced rollouts:** GT prefix chunk + 20 student-generated chunks,
  student's own history as conditioning, teacher+critic score ALL 189 frames
  jointly, gradient applied per-chunk with detached history.
- **Per-chunk conditioning:** 12-s caption segments mapped to chunks; same
  interface exposed at inference as a timed prompt schedule.
- **RAG analogy:** yes — it's "memory as geometric retrieval keyed by camera
  pose," explicitly *not* learned retrieval.
- **Linear-attention global state:** a small set of summary tokens
  accumulated across the sequence (Yang et al. 2023-style linear attention),
  perceiver-ish flavor; details of S not fully specified in the paper.

### What a from-scratch toy implementation should demonstrate

Priority list for `implementation/`:

1. Recurrent chunk loop with **fixed-shape** per-step cost (history + bank
   budgets) — measure cost vs. session length, show it's flat.
2. Camera-indexed world state bank with write (unproject) / read
   (co-visibility rank + z-buffer render + visibility mask) on a synthetic
   2D/3D world — show revisit recall when retention ≥ time-away.
3. Dense vs. sparse (sink+local+selected+linear-state) attention teacher —
   verify linear scaling.
4. DMD-style distribution matching with self-forced rollouts and detached
   inter-chunk history — show photometric-drift resistance improves with
   supervision horizon W (short vs long teacher ablation in miniature).

---
*Notes started 2026-09-29. Deep second pass completed 2026-10-01.*
