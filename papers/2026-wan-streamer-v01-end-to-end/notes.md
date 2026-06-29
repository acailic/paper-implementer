# Notes — Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models

> First-pass reading notes. Raw, thinking-out-loud.
> Paper: Wan Team, Alibaba Group. arXiv:2606.25041 (v2, 25 Jun 2026).
> 6 pages main + references + appendix. Website: https://wan-streamer.com/

## What kind of paper is this?

A **systems / architecture paper** for a real-time multimodal foundation
model. It is closer to a "position + engineering design" paper than a
benchmark paper: there are no big ablation tables or SOTA charts in the
classical sense. The "results" are (a) a latency comparison against other
real-time/omni-modal systems (Tab. 1–2), and (b) qualitative claims about
naturalness, interruption handling, and proactive speaking. The current
release is explicitly a **proof of concept at 192p** — scaling is left to
future work.

So the load-bearing artifact to re-implement is the **architecture + the
streaming contract**, not a set of numbers to reproduce.

## The problem

Real-time human interaction is **streaming and full-duplex**: people watch,
listen, speak, gesture, pause, and interrupt *simultaneously*, at
audio-visual timescales. Today's "interactive" AI systems are almost all
**cascaded**: ASR → LLM → TTS → avatar/video-renderer. That pipeline has
three diseases:

1. **Module-boundary waiting time** — each stage waits for the previous one,
   so latency stacks instead of overlapping.
2. **Error accumulation** — recognition / synchronization errors compound
   down the chain.
3. **Un-learnable behavior** — response timing, turn management, identity
   preservation, and long-horizon consistency can't be learned *as one
   behavior* because they live in separate modules with separate objectives.

The core thesis: **streamability is a modeling constraint, not a serving
optimization.** You cannot bolt low-latency full-duplex behavior onto a
system designed around offline encoders / bidirectional decoders /
round-based dialogue. You have to design for causality from the start.

## The big idea

One Transformer. Everything is one causal stream. Language, audio, and
video on **both the input and output sides**, interleaved as tokens, with
**block-causal attention** coordinating them. No external VAD, ASR, TTS,
animation, or video-generation modules. Perception, reasoning, generation,
response timing, turn-taking, and cross-modal sync are all learned jointly
in one persistent interaction state.

## The five architectural ideas (what we actually re-implement)

| # | Idea | Where |
|---|------|-------|
| 1 | **Interleaved multimodal token stream** — `{text,audio,video}_in` + `{text,audio,video}_out` in one sequence | §2.1, Fig 1 |
| 2 | **Block-causal attention** — input tokens bidirectional within a block; output tokens causal within a block; all tokens see all past blocks | §2.1, Fig 1 |
| 3 | **Streaming unit** = 160 ms chunk at 25 fps (4 frames) — the atomic unit of perception + generation | §2.1 |
| 4 | **Thinker–performer pipeline** — two-GPU split: thinker (encode + KV state + decode), performer (flow-matching solver). KV-cache exchange keeps unified state | §2.4, Fig 2 |
| 5 | **Conditional flow matching** for audio/video latents — discrete text via next-token CE; continuous audio/video via flow-matching velocity prediction | Eq 1–3 |

## The math (key equations)

**Eq 1 — causal streaming factorization.** At step *k*, user observations
`u_k = (u_k^t, u_k^a, u_k^v)` and agent response `y_k = (y_k^t, y_k^a, y_k^v)`:

```
pθ(y_{1:K} | u_{1:K}) = Π_k  pθ( y_k^t, y_k^a, y_k^v |
                                   u_{≤k}^t, u_{≤k}^a, u_{≤k}^v,
                                   y_{<k}^t,  y_{<k}^a,  y_{<k}^v )
```

The agent response is conditioned on *all* user observations up to and
including k, plus all *prior* agent responses. Note the asymmetry: user
side is `≤k` (current frame included), agent side is `<k` (only history) —
then the current `y_k` is generated and committed to history.

**Eq 2 — flow-matching latent construction.** For modality `m ∈ {a,v}`,
clean target latent `z_0^m`, Gaussian noise `ε^m ~ N(0,I)`:

```
z_τ^m = (1 − τ)·z_0^m + τ·ε^m
∂z_τ^m / ∂τ = ε^m − z_0^m
```

τ = 1 → pure noise; τ = 0 → clean. Linear interpolation (rectified-flow
style).

**Eq 3 — flow-matching training loss.** `c_k` = clean streaming context
(all arrived user obs + committed agent responses). Train a *unified*
velocity predictor conditioned on *both* noisy audio and noisy video
latents + context + noise level:

```
L_FM^m = E_ε || fθ( z_τ^a, z_τ^v, c_k, τ ) − ∂z_τ^m/∂τ ||_2^2
```

Key subtlety: **the same `c_k` conditions both audio and video velocity
predictions**, and the velocity net takes *both* noisy latents as input →
speech and motion are coupled before decoding. After denoising, estimated
clean latents are appended to history as clean context for the next unit.

## Training (3 stages, §2.3)

| Stage | Goal | Data |
|-------|------|------|
| **1. Independent-task pretraining** | Init unified Transformer from a language model; train causal audio/video encoders + decoders + flow-matching heads on understanding + generation tasks, mixed | image/audio/video QA, text dialogue, ASR/TTS, T2I/T2A/T2V, joint AV generation |
| **2. End-to-end interaction training** | Adapt to duplex interaction: user + agent t/a/v interleaved on one causal timeline | duplex interaction data (both sides have t/a/v) |
| **3. Distillation for low-latency streaming** | Distill stronger teacher (CFG + more solver steps) into efficient student; **rolling distillation** with self-forcing + distribution matching to cut train-test gap and long-horizon degradation | teacher-generated trajectories |

## Inference: the thinker–performer overlap (§2.4, Fig 2)

This is the clever systems bit. One model, two GPUs:

- **Thinker** (GPU 0): causal A/V encoders + short token-causal Transformer
  path for language + KV-cache construction + causal A/V decoders.
- **Performer** (GPU 1): only the flow-matching solver (latent generation).

Per streaming step *k*:
1. Thinker encodes `u_k`, runs token-causal decode → produces **KV slice k**.
2. At the comm boundary: thinker receives clean A/V latents `y_{k-1}` from
   performer (made last step), sends KV slice *k* to performer.
3. Thinker **decodes `y_{k-1}` → emits audio/video** immediately.
4. Performer appends KV slice *k* to its full-history cache, runs flow-matching
   solver → makes `y_k` latents, keeps them, returns them next step.

Result: current-frame perception/state-update, previous-frame output
decoding, KV/latent comm, and next-frame latent denoising all **overlap**
across adjacent units. **Throughput** ≈ performer wall-time + tiny comm
overhead < 160 ms/unit. **Latency** (full signal-to-signal path) ≈ 200 ms.
With 350 ms bidirectional network → ~550 ms total interaction latency.

## Results (the comparison tables)

**Tab. 1 — speech + omni-modal dialogue latency.** Wan-Streamer: ~200 ms
model-side, ~550 ms total (incl. 350 ms net). Competitors:
- Doubao Realtime Voice ~1 s (speech-only)
- GPT-4o / Realtime API: protocol-dependent, ~500 ms API TTFB
- Hume EVI 3: 0.9–1.4 s web benchmark
- Gemini Live: 0.8–1.2 s
- Moshi / Qwen3-Omni / MiniCPM-o 4.5: report first-packet/RTF, no visual

The authors stress: compare by **measurement boundary**, not smallest raw
number. Many competitors report first-packet or model-internal latency, not
user-visible signal-to-signal, and most don't close the loop with
synchronized visual output.

**Tab. 2 — visual agents / streaming avatars / A-V generators.** Separates
full-loop systems (Body of Her, MIDAS, MAViD, M.I.O, X-Streamer, LPM,
U-Mind) from renderers/generators (VASA-1, TalkingMachines, StreamAvatar,
LiveTalk, Hallo-Live, OmniForcing). Wan-Streamer is the only one in the
comparison that is one end-to-end model producing text I/O + speech + 25 fps
visual response in a single causal stream.

## What surprised me / what's not obvious from the abstract

- The paper has **no quantitative naturalness/interruption metrics** — those
  are described qualitatively ("the agent maintains identity, gaze, posture,
  breathing..."). The evaluation is essentially latency + architectural
  completeness vs. peers. This is honest about v0.1 being a proof of concept.
- The **block-causal attention asymmetry** (input bidirectional, output
  causal within a block) is the crux that lets a 160 ms user chunk be
  "understood" as a whole while the response is still autoregressive.
- The **KV-cache exchange** is what lets you split one model across two GPUs
  *without* breaking the unified causal state — the performer never runs
  decoders, the thinker never runs the expensive flow-matching solver.
- Current output is **192p** — explicitly a scaling starting point.

## Terms / concepts to nail down

| Term | Meaning |
|------|---------|
| **Full-duplex** | Both sides transmit simultaneously; agent perceives while speaking, can be interrupted |
| **Streaming unit** | 160 ms atomic chunk at 25 fps = 4 frames; the unit of both perception and generation |
| **Block-causal attention** | Per-block: input tokens bidirectional, output tokens causal; cross-block: full visibility to all past |
| **Conditional flow matching** | Rectified-flow-style generative model: learn velocity field `dz_τ/dτ`, integrate τ:1→0 |
| **Thinker–performer** | Two-GPU deployment split; KV-cache exchange preserves unified state |
| **Self-forcing / rolling distillation** | Student rolled out over its own history during distillation to match teacher trajectory under realistic rollout (mitigates train-test gap) |
| **VASA-1 / Moshi / OmniFlatten** | Prior systems: audio-driven talking face / full-duplex speech-text / GPT-based duplex voice — all speech-only or single-side |

## What to actually re-implement

A minimal, numpy-or-PyTorch toy that demonstrates the **five ideas**
faithfully (structure, not scale):

1. `StreamingUnit` — the 160 ms interleaved t/a/v in/out token block.
2. `build_block_causal_mask()` — the attention mask (input bidir, output
   causal within block, full past across blocks). **Verifiable** by checking
   the mask properties directly.
3. `StreamingTransformer` — unified Transformer with text-logit head +
   audio/video velocity heads (flow matching), run under the block-causal
   mask.
4. `FlowMatchingSolver` — add-noise (Eq 2), target velocity, training loss
   (Eq 3), Euler denoise τ:1→0.
5. `ThinkerPerformerPipeline` — simulate the two-GPU overlap, exchange a KV
   slice + latents per unit, report per-step thinker/performer/total
   "latency" on a tiny model.

A toy duplex conversation (user/agent turns incl. an interruption + a
proactive agent comment) drives it. Goal: it runs and the mask/flow-matching
properties verify — numbers are not comparable to the 4K-param real model.

## Open questions for the deep-read pass

- Exactly how are the **causal VAEs** structured? The paper says "strictly
  causal audio and video VAEs" but gives no architecture detail — likely
  inherited from the Wan video-gen lineage. For the toy we treat latents as
  abstract continuous vectors.
- How is **text output** interleaved with audio/video latents in time within
  a block? The toy assumes a fixed token budget per modality per unit.
- The **distillation** stage (CFG absorption + rolling self-forcing) is the
  least-specified part — we'll approximate it conceptually in the writeup.
