# Writeup — Alaya-EVOKE: From Linear-Scaling Supervision to Endless World

> **Paper:** Alaya-EVOKE: From Linear-Scaling Supervision to Endless World
> **Authors:** Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuanhao Li, Kaipeng Zhang, Feng Zhao
> **arXiv:** 2608.13546 — https://arxiv.org/abs/2608.13546
> Written after reading the paper twice and re-implementing its four core
> mechanisms from scratch in `implementation/`.

## The one-paragraph version

An interactive world model should generate video forever and answer to the
user mid-stream. Two things block this: keeping history in the denoiser's
context (or a KV cache) makes cost grow with session length, and a few-step
student distilled for latency can only be as horizon-stable as its teacher.
EVOKE's answer is to split the problem: **memory becomes geometry stored in
an external, camera-indexed world state bank** (retrieve only what the
current view needs; context stays bounded), and **stability becomes a
property of the teacher's supervision horizon** (a sparse-attention teacher
that supervises 30-second windows affordably, distilling into a 3-step
CFG-free student via distribution matching on self-forced rollouts). The
result runs open-ended sessions — 65-minute rollouts stay photometrically
flat — at 2.11 s of wall clock per 1.5-s chunk on a single H200.

## The problem

Interactive world models have three demands that fight each other:

1. **Persistent memory.** If a user turns the camera 180° after ten minutes,
   the scene behind them must still be there. Standard tricks — frame
   concatenation into the denoiser context, growing KV caches — scale with
   session length, so every real system truncates history and quietly
   forgets.
2. **Responsiveness.** Interaction wants few denoising steps. But few-step
   students are distilled, and distillation is bounded by the teacher: if
   the teacher can only judge short windows, drift that is locally plausible
   but globally wrong never produces a training signal.
3. **Long-horizon consistency.** Two flavors: *degradation drift*
   (exposure creep, texture decay — visible in a local window but caused
   earlier) and *content drift* (scene identity morphing while every short
   window looks fine — only identifiable when distant moments are compared
   *jointly*).

The paper's key framing: these two drift types have **different owners**.
Degradation drift is a temporal-evolution problem → fix it with long-horizon
*supervision*. Content drift on revisit is a spatial-recall problem → fix it
with *stored geometry*. A longer supervision window cannot resurrect an
observation that left context; a memory bank cannot decide how never-observed
content should evolve.

## The idea

Three coupled mechanisms:

**1. Recurrent session with externalized state (Eq. 1).** A session is a
loop over 1.5-s chunks (9 latent frames):

```
r_k     = Read(M_k, P_k)              # render geometry relevant to current camera
x_k     ~ p_theta(. | r_k, h_k, c_k)  # generate chunk from render + bounded history + text
M_(k+1) = Write(M_k, x_k, P_k)        # depth-unproject new frames, append to bank
```

Both `h_k` (tiered denoiser history, only ~3.2 s / 19 latent frames) and
`M_k` (the bank, capped at 90 s of ingested geometry ≈ 720 source frames)
operate under **fixed budgets** — so the cost of one recurrent call is
independent of how long the session has run. Endlessness = more recurrent
calls, not bigger ones.

**2. Camera-indexed world state bank.** The Write path runs a monocular
depth model on 12 frames of each generated chunk, unprojects to world space
under the known camera trajectory, and appends — deliberately **without
cross-chunk fusion** (fusion is where scale drift and seams come from over
long sequences). The Read path is purely geometric, no learned retrieval:
rank stored source views by **co-visibility** with the target pose, take up
to 8 sufficiently distinct ones, render with z-buffering into a view-aligned
warp + per-pixel visibility mask. Unsupported pixels get noise σ = 1 (they
carry no visual information); visibility pooled at patch resolution removes
unsupported *history tokens* from the denoiser entirely. This is RAG, but
the keys are camera poses and the documents are geometry — right choice,
because you're recalling what was observed, not inferring what wasn't.

**3. Long-horizon supervision, linearly priced.** The teacher (Wan2.2 A14B
DiT, high/low-noise experts kept) gets **sparse chunk attention**: each
query chunk attends to a first-frame sink, local neighbors, compressed
nearby frames, M importance-selected distant frames, and a linear-attention
global state. Bounded keys per query ⇒ linear cost in sequence length ⇒
supervising ~31-s rollouts (GT prefix + 20 self-forced student chunks, 189
latent frames) becomes affordable. Distillation is DMD-style distribution
matching: teacher and critic **share one backbone** (a LoRA toggle selects
which), the gradient pushes the student sample toward teacher-score
distributions with a data-scaled normalizer ν, a mask Ω excludes the prefix
and first generated chunk (boundary-flicker guard), and — the trick worth
stealing — **history is detached between rollout chunks**: each chunk's
backward graph is independent, so the full 31-s window is *scored* jointly
while activation memory stays chunk-sized. Per-chunk text conditioning
(captions segmented every 12 s and mapped onto chunks) makes mid-session
prompt changes a first-class *interface*, the "evocation."

## How it works (the intuition)

Think of what the student is allowed to learn. A windowed objective
`L_W = E D(q_θ(k:k+W−1), p(k:k+W−1))` gives **zero gradient** for anything
not expressible in W-window statistics. Shrink W and no amount of training
will teach the student that its exposure is creeping, because the cause
sits *before* the window. Widen W and the student is supervised on a wider
distribution of its own rollout-perturbed histories — which is exactly the
distribution it must survive at inference. Under self-forced rollouts this
is on-policy distillation: the teacher doesn't grade rehearsed trajectories,
it grades the student's *own* wandering.

And the bank is the complement: when the camera returns to a pose, text
conditioning cannot and should not conjure back what was seen 60 s ago —
geometry can. The paper's timed-control experiment makes the division crisp:
a mid-session text clause realizes in 67% of cases when it targets
*unanchored* content (ceiling 83%), but only 4% when it would require
*overwriting bank-anchored geometry* (floor 17%). **Text governs free
content; stored geometry resists overwriting.** That's not a bug — it's the
product decision.

## What I learned by implementing it

(From the toy re-implementation in `implementation/`; everything below was
*measured*, numbers in `implementation/results.json`.)

- **Flatness is a saturation property, not an asymptote.** In E1 I ran
  100-chunk recurrent sessions and watched the bank's active source pool
  fill to its budget (~60 sources at the toy's 90-s retention) — after
  which per-step time is flat (19.9 ms vs 19.8 ms in late windows, ratio
  0.98). Before saturation, cost *does* grow. The paper's flat-cost claim
  is really "the budget bounds the work, and the budget fills fast."
- **Recall is literally `retention ≥ time_away`.** E2 swept the retention
  budget against a 60-s revisit: 15 s → 0 dB (nothing), 45 s → 12.2 dB at
  24% coverage (partial), 90 s → 14.6 dB at 98.6% coverage. The paper's
  20/21 pose-addressed comparisons following the predicted transition is
  exactly what the toy shows — it's a *switch*, not a gradient, because
  the source frames are either in the bank or they aren't. And the plateau
  (~15 dB in my toy, 15.4–17.8 dB in the paper) says the recall is
  semantic, not photographic.
- **Why no fusion is the right default.** My Write path just
  unprojects-and-appends. The moment you contemplate fusing two chunks'
  geometry you inherit relative-pose error as accumulated scale drift —
  and z-buffering at Read time already resolves overlaps. Fusion would add
  a failure mode to buy an aesthetic improvement the budget makes
  unnecessary.
- **Sparse attention's win is structural, not implementational.** E3
  measured context per query at 22 tokens *regardless of rollout length*
  (state + sink + local + selected distant), and analytic FLOPs from 8→128
  chunks grew ×22 (sparse) vs ×256 (dense). Once keys-per-query is capped,
  linearity is arithmetic, not engineering.
- **Horizon transfers, and it transfers *photometrically*.** E4 distilled
  two students from teachers differing only in window W: end-of-rollout
  brightness retention 50.9% (W=2) vs 90.7% (W=20), mirroring the paper's
  74% vs 101%. Implementing it forced me to notice *why* the mask Ω
  excludes the first generated chunk (teacher treats chunk 0 as an image
  prior; the student treats it as continuation — matching there teaches
  flicker) and why detached-history backward is load-bearing (you could not
  afford the joint graph anyway).

## What surprised me / was harder than expected

- **The most honest ablation is in the appendix.** Teacher–critic
  detectability barely changes beyond W = 2 chunks across 13 perturbation
  conditions — so the drift benefit is *not* from the raw window length but
  from the whole trained long-horizon teacher plus the rollout
  distribution. The paper says this about itself, which I rarely see.
- **VBench-Long rank 7 of 10, stated plainly.** Evoke wins WBench and
  VBench-2.0 but is mid-pack on VBench-Long against many-step peers — and
  the paper flags that the comparison isn't step-matched rather than hiding
  it. The stability claim (scene-identity cosine plateaus at 0.523 —
  exactly where real footage scores against itself 60 s apart) is framed
  as *evidence against degradation*, not as fidelity. Refreshing.
- **Hardest toy piece: making drift *identifiable*.** To show
  horizon-matters in miniature I had to make drift absolute in rollout
  position (a W=2 student literally never sees positions ≥ 2), otherwise
  short and long teachers learn the same correction. The paper's
  self-forced rollouts are doing this exact work in the real system.
- **The student's 3-step pyramid is the least-documented part** (condition
  geometry only at the coarsest of 3 latent levels, 12×20 → 48×80) and my
  stand-in for it — a conditioning net over a toy pyramid — is the
  weakest link of my re-implementation. The *mechanisms* (budgets, bank,
  sparse attention, windowed distribution matching) are faithful; the
  generative model is a stand-in. Scope honestly noted in
  `implementation/README.md`.

## References

- Paper: https://arxiv.org/abs/2608.13546
- Project page: https://evoke-world.github.io/Evoke/
- My implementation: `implementation/` (run `python3 train.py`, ~15 s CPU,
  writes `results.json`)
- Breakdown: `breakdown.md`
- Reading notes: `notes.md`
