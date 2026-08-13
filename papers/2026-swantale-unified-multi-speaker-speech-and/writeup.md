# Writeup — SwanTale: Unified Multi-Speaker Speech and Audio Generation

> **Paper:** SwanTale: Unified Multi-Speaker Speech and Audio Generation for Instruct and Zero-Shot Tasks
> **Authors:** Yu Zhang, Ruiqi Li, Changhao Pan, Ke Lei, Xiang Yin, Cheng Yang (ByteDance / Zhejiang University)
> **Year:** 2026 · **ArXiv:** https://arxiv.org/abs/2608.02023

This is my own explanation, written after reading and re-implementing the
paper. It's a synthesis, not a retelling of the abstract.

## The one-paragraph version

SwanTale is an industrial-scale audio generator from ByteDance that does two
things normally done by *different* models — **designing** a voice + scene from a
text caption (the *instruct* task) and **cloning** a voice from a reference clip
(the *zero-shot* task) — inside one non-causal flow-matching Transformer. The
elegant move is that these two tasks differ in only two things: *what* caption
you feed and *which latent frames* you mask as fixed reference context. The rest
of the paper is a tower of engineering that makes this practical at 2B
parameters: a smooth 25 Hz audio VAE (SwanVAE), a hashed n-gram caption memory
(the Engram layer), a dual-router Mixture-of-Experts that spends compute only
where the waveform needs it, a trick that turns "make it high quality" into a
conditional variable instead of a reinforcement-learning problem, and a
flow-matching GRPO stage whose per-element mean log-prob keeps the importance
ratio near 1 across utterance lengths.

## The problem

A media creator — dubbing animation, making an ad, producing an audio drama,
editing a short video — wants a *single* system that can:

1. **Invent a voice from a description** (no reference recording): *"an elderly
   man speaking warmly in a quiet café with distant clinking cups."* That is the
   **instruct** task. The caption specifies the speaker persona, the acoustic
   environment, and the fine-grained content.
2. **Clone and reuse a voice** from a short clip — the **zero-shot** task, the
   classic TTS job — while still supporting multi-speaker dialogue.
3. Place the speech *inside* an acoustic scene with correctly-timed, non-speech
   sound effects (footsteps, wind, music stings), all in one waveform rather
   than stitched together downstream.

Existing systems each cover one of these. Instruct-TTS makes only speech (not
scenes) and can't clone; zero-shot-TTS can't take captions and can't design
voices from scratch; and bolting speech + SFX together in a downstream pipeline
produces timing, reverb, and loudness drift. The authors also name three real
obstacles: **data scarcity** (rich multi-level captions are expensive to
annotate), **task compatibility** (caption→style and audio→style conditioning
paths weaken each other under joint training), and **multi-audio-modality
complexity** (speech, SFX, singing, and music have very different temporal
structures and must coexist).

## The idea

The single most important idea is also the simplest: **task unification via
masking.** Pick a clean latent target `x★`. Sample a flow-matching time `t` and
noise `ε`, form the noised latent `x_t = (1−t)ε + t·x★`, and have the model
predict the velocity `v = x★ − ε`. Now:

- For the **instruct** task, mask `m = 0` everywhere (generate all frames) and
  feed the *full* caption (environment + speakers + content).
- For the **zero-shot** task, mask the *reference* frames `m = m_prompt` (keep
  them at the clean target, so they act as a conditioning prefix) and feed only
  the *content* caption.

One backbone, one velocity objective, two tasks that differ only in caption
content and which frames are masked. That is the whole unification trick, and it
is genuinely clean.

Everything else is engineering to make this practical and to squeeze quality:

- **SwanData-Caption** — a data pipeline (~70M caption records) that cleans
  media audio, runs vocal separation / diarization / ASR / alignment, and uses a
  multimodal LLM to annotate three structured fields (Environment / Speakers /
  Content), refined by quality filters and a clever group-wise best–worst
  expressiveness comparison instead of absolute MOS scoring.
- **SwanVAE** — an asymmetric audio VAE: a pure-CNN encoder downsampled
  aggressively (48 kHz → 25 Hz), a Gaussian bottleneck, and a SAME-style
  Transformer resampling decoder. It produces smooth continuous latents that are
  easy to model by flow matching.
- **Engram layer** — hashed n-gram memory (orders 2 and 3) bolted onto the
  caption embeddings, with a content-dependent sigmoid gate whose bias is
  initialized *negative* so the memory path starts closed and only opens for
  structured markers.
- **Unified MoE** — a dual router replacing every second DiT FFN: a *task-level*
  router picks shared experts, and a *frame-level* audio router uses a dynamic
  Top-P selection with null experts and a time-aware budget `q(t)` so compute is
  spent where the waveform actually needs it. Load balancing is *aux-loss-free*
  (bias adjustment), stabilized by z-loss and a null-collapse penalty.
- **Reward-conditioned quality** — instead of RL, quality (STOI / SI-SDR / PESQ
  / MOS) is fed in as a caption + a flag, and at inference the flag is **forced
  to "high."** A hard problem becomes a conditional-generation problem with no
  rollout.
- **Flow-matching GRPO** — when RL *is* used (pronunciation, attribute
  agreement, speaker similarity), they define a marginal-preserving SDE so each
  Euler step is a Gaussian, and use a **per-element mean log-prob** so the
  importance ratio stays near 1 across utterance lengths, with a closed-form KL
  to a frozen reference policy (same variance → just a squared-mean term).

## How it works (the intuition)

Think of the audio waveform being compressed into a smooth 25-frame-per-second
latent by SwanVAE. The flow-matching generator then learns a *velocity field*
that pushes pure noise (t=0) toward clean data (t=1). The cleverness is in how
the two tasks share that field:

- Instruct = "denoise everything, guided by the full caption."
- Zero-shot = "denoise everything *except* a clean reference prefix, which
  anchors the identity, guided by just the content."

Because the reference frames are kept clean and concatenated into the noised
state, the model sees them as ground-truth context — it does not have to
*reconstruct* identity, just *continue* it. That is why one objective suffices.

The Unified MoE then decides, per frame, how much *extra* compute that frame
deserves. A silence frame routes to a null expert (cheap); a voiced, rich frame
routes to several audio experts (expensive). The time-aware budget `q(t)` shifts
this over the flow time `t` — more capacity early in the denoising (structure)
than late (texture). And the Engram layer gives the caption branch a cheap
lookup table so structured phrases ("a quiet café") retrieve learned embeddings
instead of being inferred from scratch each time.

At inference they run two-stage decomposed classifier-free guidance — null →
text+speaker → full — with a sway sampling grid `t(u) = 1 − cos(πu/2)` that puts
more integration steps near the noisy end.

## What I learned by implementing it

Implementing the four self-contained pieces (Engram, Unified MoE, the flow DiT
with task masking, and sway sampling) on synthetic latents made several things
concrete that the paper states only in passing:

1. **The task-masking unification really does "just work" as one objective.**
   In my toy run, both `inst` and `zero` tasks trained under a single shared
   backbone and the generation-frame MSE dropped well below the initial noise
   level for both. You don't need a separate zero-shot loss — the mask + caption
   split is the entire mechanism.
2. **The negative-init gate bias on the Engram layer matters.** With a
   zero-or-positive bias, the memory path fires immediately and can dominate /
   destabilize early training; the negative bias makes it a *residual that has
   to earn its activation*, which matched the paper's claim but was visceral
   once I watched the loss.
3. **Aux-loss-free load balancing is surprisingly effective on a toy.** Just
   nudging each routed expert's selection bias by `η·(1/R − f_i)` — no auxiliary
   assignment-probability loss — kept my expert usage close to uniform, and the
   null-collapse penalty (`L_null`) genuinely prevented all tokens collapsing
   onto the null expert. The z-loss kept router logits from drifting.
4. **The per-element mean log-prob trick is the transferable gem from the GRPO
   stack.** Because every latent element contributes one log-prob and you *mean*
   over them, the importance ratio `ρ = exp(ℓ_θ − ℓ_old)` stays O(1) regardless
   of sequence length — the sum-based version blows up for long utterances. And
   because the SDE transition shares the same variance for policy and reference,
   the KL collapses to a clean squared-mean-distance term. These two ideas
   generalize far beyond audio.
5. **Reward-as-condition is a quietly powerful idea.** Forcing the quality flag
   to "high" at inference is a one-line trick that sidesteps an entire RL loop.
   I did not implement the full GRPO, but the *conditional* framing is something
   I'd reach for in any conditional generator.

## What surprised me / was harder than expected

- **How little the paper isolates each component.** Engram, dual router,
  quality conditioning, curriculum, GRPO are all stacked together, and the
  ablations don't fully separate them — so when something works, you can't
  always tell *which* knob earned the gain. That's an industrial-paper reality,
  but it made "what do I re-implement?" a judgment call. I picked the four pieces
  that are algorithmically self-contained.
- **The dynamic Top-P + time-aware budget is fiddly.** Getting the budget
  `q(t)`, the Top-P threshold, the null bias, and the capacity factor to move
  *together* as functions of the same scalar — and training the Gumbel
  temperature down without breaking routing — took the most care. The
  "selection uses bias-adjusted logits, weights use base logits" split is easy to
  get backwards.
- **Sway sampling actually changes results.** A non-uniform grid sounds like a
  cosmetic detail, but concentrating steps near `t=0` (where the field is
  steepest) measurably improved the toy sample quality versus uniform steps for
  the same step count.
- **Scale is the real story.** 2B active params, 64×A100, ~70M caption records,
  and a Qwen caption encoder mean the *data pipeline* (SwanData-Caption) is
  arguably the biggest contribution and the least re-implementable. The headline
  ablation — scaling the caption encoder 8B→32B raises instruction accuracy the
  most (3.39→3.70) — confirms that caption-understanding capacity is a genuine
  bottleneck, as much as the acoustic generator.

## References
- Paper: https://arxiv.org/abs/2608.02023
- My implementation: `implementation/` (`model.py`, `train.py`, `data.py`, `README.md`)
- Breakdown (full method, math, architecture): `breakdown.md`
- Deep notes (two-pass reading): `notes.md`
