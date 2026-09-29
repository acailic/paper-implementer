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

### Terms / concepts I don't fully understand yet (to dig into on pass 2)

- "camera-indexed world state bank" — what exactly is stored per entry?
  Depth? Point clouds? Multi-view features? How is retrieval scored?
- The three sparse-attention components: "chunk-wise grouping", "retrieval of
  selected distant frames", "linear-attention global state" — how do they
  combine? What exactly attends to what?
- "distribution-matching objective" — exact loss. Is it DMD (distribution
  matching distillation)? GAN? How are "fake" scores computed?
- "self-forced rollouts" — from Self-Forcing (2024). Student rolls its own
  previous chunks as context and the teacher scores them. Need the exact
  training loop.
- "per-chunk conditioning" — how prompts/events are injected per chunk, and
   how that survives distillation into the student.
- Relationship between EVOKE's retrieval and RAG-style retrieval: is this
  basically "memory as retrieval over geometric features keyed by camera"?
- Baselines: WBench / VBench-Long / VBench-2.0 — what do they measure?
- "linear-attention global state" — is this Mamba-like or perceiver-like?

### Structure of the paper (from the skim)

- §1 Intro: the session-length vs. memory tension; few-step student bounded
  by teacher.
- §2 Related work: video diffusion, world models, distillation.
- §3 Method: world state bank; teacher architecture; supervision design;
  distillation objective; student architecture.
- §4 Experiments: WBench SOTA, VBench-Long / VBench-2.0 competitive, latency
  numbers, ablations (bank, sparse attention, self-forcing).
- §5 Conclusion.

### Quick take

This is a systems-flavored paper: the two contributions (external memory +
linear-scaling teacher for supervision) are engineered to work together. The
load-bearing trick is treating *supervision* (the teacher's ability to watch
long horizons) as a first-class design target, rather than fixing the student
architecture alone. A minimal re-implementation should therefore demonstrate:
(1) a toy world state bank with camera-keyed retrieval, (2) a sparse/linear
attention teacher vs. dense attention, (3) a distribution-matching
self-forced distillation loop. On a toy 2D world this is very doable.

---
*Notes started 2026-09-29.*
