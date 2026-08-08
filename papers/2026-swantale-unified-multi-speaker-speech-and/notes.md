# Notes — SwanTale: Unified Multi-Speaker Speech and Audio Generation

> **Paper:** SwanTale: Unified Multi-Speaker Speech and Audio Generation for Instruct and Zero-Shot Tasks
> **Authors:** Yu Zhang, Ruiqi Li, Changhao Pan, Ke Lei, Xiang Yin, Cheng Yang (ByteDance / Zhejiang University)
> **Year:** 2026 | **ArXiv:** 2608.02023 (eess.AS)
> **First pass — 2026-08-08**

---

## Big picture (first impression)

SwanTale is a **unified TTS + audio generation system** from ByteDance that does
two jobs in one model:

1. **Instruct task** — generate speech + environment + audio effects from a
   natural-language *caption* (no reference recording needed). You describe the
   speaker persona, the scene, and the fine-grained content, and it produces the
   waveform.
2. **Zero-shot task** — clone a voice from a short reference clip (classic TTS),
   but also keep multi-speaker dialogue capability.

The key claim: most TTS systems do *either* instruct *or* zero-shot. SwanTale does
**both in one model**, and also generates **non-speech audio** (environments,
sound effects, occasional singing, music) in the *same* waveform rather than
farming it out to a separate pipeline. That's the "Unified" in the title.

It's an industrial-scale system (2B active params, 64×A100, 70M caption records),
so the "method" is really a stack of components, each a paper on its own.

---

## Problem statement (in my own words)

Media creators (animation dubbing, ads, games, podcasts, audio drama) need to:
- **Design** a voice from scratch with natural-language descriptions (no reference).
- **Reuse** that designed voice later via zero-shot cloning.
- Embed the speech inside an **acoustic scene** (café ambience, wind, footsteps)
  with correctly-timed local audio effects.
- Support **multi-speaker dialogue** with stable per-speaker identity.

Existing systems fail at the junction of these: instruct-TTS makes only speech
(not scenes), zero-shot-TTS can't take captions, and stitching speech + SFX in a
downstream pipeline causes timing/reverb/loudness drift.

Three stated challenges:
1. **Data scarcity** — rich multi-level captions are expensive to annotate.
2. **Task compatibility** — instruct (caption→style) vs zero-shot (audio→style)
   conditioning paths can weaken each other under joint training.
3. **Multi-audio-modality complexity** — speech, SFX, singing, music have
   different temporal structures and must coexist in one waveform.

---

## Core idea (one paragraph)

Build a **data pipeline** (SwanData-Caption) that turns messy media audio into
clean, structured, fine-grained captions with environment / speakers / content
fields. Then build a **non-causal flow-matching DiT generator** (SwanTale) that
operates on **continuous 25 Hz latents** from a custom VAE (SwanVAE). Condition it
on a separated caption branch (Qwen encoder + Engram memory) + a separate text
branch (CosyVoice tokens), route feed-forward computation through a **Unified MoE**
(task router + dynamic Top-P audio router), apply **reward-conditioned quality
control** (feed quality scores as a condition, set to "high" at inference), train
with a **curriculum** (zero-shot base → dense caption → full MoE → HQ SFT), and
finish with **GRPO RL** to fix pronunciation/stability/attribute-control. At
inference use two-stage decomposed CFG + sway sampling.

---

## Components I can see so far

### Data side — SwanData-Caption (Section 2)
Four stages:
1. **Coverage design** — media-style real data (drama, anime, ads, podcast) +
   3 targeted synthetic subsets (elderly speech, short utterances, hard
   pronunciation) generated with a phoneme-aware TTS teacher, 100k each.
2. **SwanData-Speech preprocessing** — vocal separation (Ultimate Vocal Remover),
   diarization (3D-Speaker / CAM++), ASR (Seed-ASR 2.0 + SenseVoice check),
   alignment (SwanAligner). Keep original audio for scene captioning.
3. **Caption annotation** — Seed2.0 Lite as annotator MLLM; produces 3 fields:
   **Environment**, **Speakers**, **Content**. Uses a **style-persona library**
   (per-media-family style matrices) as a soft prior to avoid impoverished
   speaker descriptions.
4. **Data refinement** — waveform filtering (DNSMOS / SQUIM PESQ/STOI/SI-SDR
   thresholds), caption normalization (SwanVerifier checks gender/age), human
   verification with **group-wise best–worst** expressiveness comparison
   (not absolute MOS, not A/B — more annotation-efficient).

Caption example (very illustrative):
> Environment: {an antique courtyard... faint wind/dripping water}
> Speakers: {Speaker 1: young woman, reborn heroine, calm, low pitch, cold tone}
> Content: {Speaker 1 speaks icy/scrutinizing: <S1>What you owe me...</S1>}

~70M caption records in the final mixture.

### Model side — SwanTale (Section 3)

**SwanVAE (3.1)** — 48 kHz mono → 96-dim continuous latents @ 25 Hz.
- Anti-aliased CNN encoder, downsampling rates [4,4,4,5,6] → 1920-sample hop.
- Gaussian VAE bottleneck (KL weight 0.02), reparam sample z for decoder.
- Decoder = SAME-style Transformer Resampling Block (TRB): project each latent
  + 6 learnable output tokens → local bidirectional Transformer → 6×320-sample
  patches = 40 ms per frame.
- **Asymmetric design**: encoder is local-only (no Transformer, ~0.95s RF),
  decoder holds most capacity (355M of 407M total).
- Reconstruction loss: multi-res complex STFT + multi-res multi-band Mel +
  frame energy + KL + adversarial (MPD + MRD + MBCSD) + feature matching.
- **Latent alignment objectives** (train-only, on posterior mean µ_ϕ): a
  generative flow-matching predictor + a causal latent predictor + semantic
  chroma readouts + multi-band energy readout. These keep the latent space
  *smooth/easy to model by flow*, not just reconstructable. All discarded after.

**Flow-based Transformer / DiT (3.2)**
- Non-causal flow matching (noise-to-data: x_t = (1−t)ε + t·x*).
- **Separated conditioning**: caption branch (Qwen encoder → cross-attention to
  all DiT layers + label embeddings for speech/audio/env/other) vs text branch
  (CosyVoice 2.0 tokenizer → lightweight Transformer → length-normalized onto
  audio-latent timeline). Speaker-turn embeddings ride the text path.
- **Reward-conditioned quality control**: append a "Quality:" caption + a quality
  flag q∈{low,normal,high,unknown} from STOI/PESQ/SI-SDR/MOS. At inference force
  "high". → Makes it a *reward-conditioned policy* with no RL needed for quality.
- **Engram conditioning (3.2)** — a memory layer on the caption branch for
  recurring n-gram patterns (N={2,3}), centered windows, hashed into K tables,
  gated residual update with negative-initialized bias (starts closed). Separates
  fixed-pattern recognition from long-range acoustic planning.
- Unified backbone for both tasks: only caption input + context mask differ.
  Zero-shot uses prompt frames as fixed reference context; instruct generates
  whole sequence.

**Unified MoE (3.3)** — replaces every 2nd DiT FFN.
- **Task router** (sample-level): picks shared experts per task (inst vs zero).
- **Audio router** (frame-level, dynamic Top-P over routed audio + null experts),
  conditioned on time embedding. Null experts = skip path (cheap for silence).
- **Time-aware budget** q(t)=σ(W_b·e_t) controls Top-P threshold, null bias,
  and capacity jointly.
- **Annealed Gumbel mixing** during training (temp ↘), deterministic at inference.
- Auxiliary losses: z-loss (logit magnitude) + null-collapse penalty.
- Expert capacity c(t)·(M/R); overflow dropped by smallest weight.
- Output = shared branch + Σ weighted routed audio experts.

**Curriculum learning (3.4)** — 4 stages:
1. Zero-shot base (SwanVoice-style, single→multi-speaker, ref dropped 50%).
2. Dense caption adaptation on clean speech (MoE replaced by dense FFN), 70/30
   instruct/zero-shot mix.
3. Full caption-mixture + introduce Unified MoE.
4. High-expressiveness/HQ SFT subset.

**GRPO post-training (3.5)** — reward-guided RL for pronunciation accuracy,
generation stability, attribute control.
- 5 shared rewards (phone_core, phone_len, pause_punct, edge_rms, quality) +
  instruct: SwanVerifier attribute agreement / zero-shot: speaker cosine sim.
- Stochastic flow policy: convert deterministic flow ODE into a marginal-
  preserving SDE (adds score + noise schedule η(t)), Euler–Maruyama transitions.
- K=8 candidates per condition, group-relative advantage (no value model).
- **Per-element mean** log-prob (NOT sum) so ρ stays near 1 across utterance lengths.
- KL to frozen reference policy (closed-form, same variance) + supervised anchor
  replay to preserve multi-speaker/audio capability.

**Inference (3.6)** — two-stage decomposed CFG:
  ṽ_t = v_∅ + ω_text(t)(v_text − v_∅) + ω_all(t)(v_full − v_text)
with timestep-dependent guidance annealing γ(t) = a + b(1−t)^p and sway sampling.

---

## Key results (headline)

- **SwanVAE**: best/near-best reconstruction on speech, singing, general audio,
  music vs DAC/EnCodec/WavTokenizer/VoxCPM2/MegaTTS3/Same-L/ACE-Step.
- **Zero-shot (SwanBench-Speech)**: ranks 1st in Timbre Consistency,
  Expressive Richness, Expressive Hierarchy (both mono & dialogue). Content
  accuracy / audio quality still room to improve.
- **Instruct (InstructTTSEval)**: 1st on Chinese APS (86.1), ties 1st English APS
  (84.2), 2nd Chinese DSD. English DSD & Role-Play are weaker.
- **SwanBench-Scene**: highest overall Mean MOS 4.22.
- **Ablation (SwanBench-Caption)**: removing Unified MoE drops all 3 metrics;
  scaling caption encoder 8B→32B raises them (esp. Instruction Accuracy).

---

## Terms / concepts I don't fully understand yet (→ focus for 2nd pass)

- **Engram** layer (Eq. 4–5): the hashing/memory formulation — need to nail down
  the exact retrieval + gated update math. Looks like a retrieval-augmented
  key-value memory but I want to verify the gate/gate-init details.
- **SAME Transformer Resampling Block** — referenced [74], the decoder design
  with 6 output tokens per latent frame. Need to understand the patchify/
  unpatchify + local attention.
- **Dynamic Top-P routing** with annealed Gumbel — how Top-P selection interacts
  with Gumbel-Softmax vs. the mixture-weight computation (Eq. 20 vs 22).
  Subtle: selection uses one softmax, weights use a *different* renormalization.
- **Stochastic flow policy** (Eq. 36–38): the score estimate s_θ and the
  rectified-flow → SDE conversion. Why per-element *mean* log-prob matters.
- **Multi-band complex STFT discriminator (MBCSD)** — band-partitioned complex
  STFT discrimination. New to me.
- **Sway sampling** — the t(u)=1−cos(πu/2) warp; need to confirm it's just a
  non-uniform integration grid.
- The exact **reparameterization convention**: they use x_t = (1−t)ε + t·x*
  (noise-to-data), velocity target = x* − ε. Need to keep this consistent vs.
  the more common data-to-noise convention in my head.

---

## First-pass takeaways

- This is a **systems paper**, not a single neat algorithm. The "method" is a
  *composition* of ~6 known-good ideas (flow DiT, VAE latents, MoE, CFG, GRPO,
  retrieval memory) assembled for a concrete product goal (media TTS+SFX).
- The genuinely interesting/novel pieces to me: **(a)** reward-conditioned
  quality control (clever — turn a reward into a *condition*, avoid RL for it),
  **(b)** Unified MoE's null-expert + time-aware budget (adaptive per-frame
  compute), **(c)** Engram memory for fixed caption patterns, **(d)** the SDE
  bridge for making flow matching amenable to GRPO.
- The data pipeline (SwanData-Caption) is arguably the biggest practical
  contribution — the multi-level caption schema + style-persona library is what
  makes instruct generation controllable.
- For implementation, the most self-contained & implementable-from-scratch pieces
  are likely: the **Unified MoE router** (task + dynamic Top-P audio + null +
  time budget + Gumbel annealing) and the **flow-matching training objective
  with task-specific masking** (Eq. 6–9). SwanVAE and the full GRPO stack are
  too heavy for a toy reimplementation, but the routing + flow loss are tractable
  on synthetic latents.
