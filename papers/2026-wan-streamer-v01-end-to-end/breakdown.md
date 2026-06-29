# Wan-Streamer v0.1 — Full Breakdown

**Paper:** Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models
**Authors:** Wan Team (Alibaba Group), June 2026
**ArXiv:** https://arxiv.org/abs/2606.25041

---

## Problem & Motivation

Building AI systems that interact with humans in real time the way humans interact with each other — continuously, with overlapping speech, gestures, and listening behavior — is fundamentally different from building a good language model or a good video generator.

**Why existing approaches fall short:**

| Approach | Problem |
|----------|---------|
| Cascaded ASR → LLM → TTS → Avatar | Latency accumulates across module boundaries; no native audio-visual sync |
| Speech-only full-duplex (Moshi, GPT-4o voice) | No visual agent output — can't show listening behavior, facial expressions |
| Audio-driven avatar renderers (VASA-1, StreamAvatar) | Fast at rendering but depend on external dialogue/speech modules — true latency is hidden |
| Omni-modal models (Qwen-Omni, MiniCPM-o) | Accept audio/video input but only output speech/text — no synchronized video response |

**Core insight:** Real-time audio-visual interaction isn't just the union of understanding + generation. It's *intrinsically full-duplex*: when the user speaks, the agent should still show visible listening behavior; when the agent responds, it should still perceive the user for interruption and adaptation. Streamability must be a modeling constraint from the start.

---

## Key Insight

> Design every component for causality from the beginning, model all modalities in one Transformer, and the system naturally achieves low-latency full-duplex interaction without pipeline hacks.

The three pillars:
1. **Single Transformer** for language, audio, and video (input and output) — no external modules
2. **Strictly causal stack** — VAEs, encoders, decoders, attention — everything processes left-to-right
3. **Streaming contract** — every observed unit is usable immediately, every generated unit is emitted and committed back to history

---

## Method

### Overview

At each streaming step k, the model sees:
- **User observations:** u_k = (text, audio, video) — what the user just said/did
- **Causal history:** all past user observations + all past agent responses

It produces:
- **Agent response:** y_k = (text, audio, video) — what the agent says/does next

The entire interaction is modeled as one long causal sequence of interleaved multimodal tokens.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Wan-Streamer Pipeline                       │
│                                                                │
│  ┌─────────┐    ┌──────────────────────┐    ┌──────────────┐  │
│  │ User    │───▶│  Causal Encoders      │───▶│              │  │
│  │ Audio   │    │  ┌─────────────────┐  │    │   Single     │  │
│  │ Video   │    │  │ Causal Audio VAE│  │    │   DiT /      │  │
│  │ Text    │    │  │ Causal Video VAE│  │    │   Transformer│  │
│  └─────────┘    │  │ Causal AV Enc   │  │    │              │  │
│                 │  └─────────────────┘  │    │  Block-Causal│  │
│                 └──────────────────────┘    │  Attention   │  │
│                                             │              │  │
│  Agent Output  ◀──┌──────────────────────┐◀──│              │  │
│  ┌─────────┐     │  Causal Decoders      │    └──────────────┘  │
│  │ Text    │     │  ┌─────────────────┐  │                      │
│  │ Audio   │     │  │ Causal Audio Dec│  │    History Context    │
│  │ Video   │     │  │ Causal Video Dec│  │    (full KV cache)   │
│  └─────────┘     │  └─────────────────┘  │                      │
│                  └──────────────────────┘                      │
└──────────────────────────────────────────────────────────────┘
```

**Key architectural components:**

1. **Causal Audio VAE** — compresses audio into a latent space that can be processed strictly left-to-right
2. **Causal Video VAE** — same idea for video frames
3. **Causal Audio-Visual Encoders** — map user observations into tokens for the Transformer
4. **Unified DiT/Transformer** — one model with block-causal attention that processes:
   - Text tokens (discrete, next-token prediction)
   - Audio latent tokens (continuous, flow matching)
   - Video latent tokens (continuous, flow matching)
   - All interleaved in one causal sequence
5. **Causal Audio & Video Decoders** — convert generated latents back to waveforms/pixels

### Block-Causal Attention

The attention mask ensures:
- **Within a streaming unit:** tokens can attend to all past context (full causal history)
- **Across streaming units:** each new unit sees everything before it, nothing after
- **Between modalities:** audio, video, and text tokens can cross-attend within the causal window

This enables streaming units as short as 160ms at 25 FPS.

### Forward Pass (per streaming unit k)

```
Step k: User sends audio+video chunk u_k

1. ENCODE (Thinker GPU):
   - Causal audio VAE encodes user audio → audio latent tokens
   - Causal video VAE encodes user video → video latent tokens
   - Text tokens from user input (if any)

2. TRANSFORMER PASS (Thinker GPU):
   - Run token-causal decoding over language/state slots
   - Produces KV-cache slice for current interaction state
   - Send KV slice to Performer GPU

3. DECODE PREVIOUS (Thinker GPU, overlapped):
   - Receive clean audio+video latents from Performer (generated for step k-1)
   - Decode latents → output audio waveform + video frames
   - Emit immediately to user

4. FLOW MATCHING (Performer GPU):
   - Receive KV slice from Thinker
   - Run flow-matching solver to denoise audio+video latents for step k
   - Keep clean latents, send to Thinker at step k+1
```

### Loss Functions

**Language (cross-entropy):**
Standard next-token prediction over the text output tokens.

**Audio & Video (conditional flow matching):**

For modality m ∈ {audio, video}, construct a noisy latent by interpolating between clean target and Gaussian noise:

- z_τ^m = (1 - τ) · z_0^m + τ · ε^m, where ε ~ N(0, I)

The velocity field is: ∂z_τ^m / ∂τ = ε^m - z_0^m

**Flow matching loss for each modality:**

L_FM^m = E_ε [ || f_θ(z_τ^a, z_τ^v, c_k, τ) - ∂z_τ^m / ∂τ ||² ]

Where:
- f_θ is the unified Transformer (predicts velocity for BOTH audio and video jointly)
- c_k is the full causal context (all past user observations + agent responses)
- τ is the flow time / noise level
- Both modalities share the same context conditioning, so speech and visual motion are coupled

**Total loss:** L = L_CE (language) + L_FM^a (audio) + L_FM^v (video)

---

## Math in Plain English

**Eq. 1 — Autoregressive streaming model:** The probability of the full interaction is the product of per-step conditional probabilities. At each step, the model predicts the agent's text, audio, and video response given ALL past user inputs and ALL past agent outputs (the "causal history"). Once predicted, the response gets appended to the history.

**Eq. 2 — Conditional flow matching:** To generate audio/video, they start from noise and smoothly transform it into the clean signal. The noisy latent is a linear interpolation between clean and noise, controlled by a time parameter τ (0 = clean, 1 = pure noise). The "velocity" tells you which direction to move through this interpolation.

**Eq. 3 — Flow matching loss:** The Transformer predicts the velocity field for both audio and video simultaneously, conditioned on the causal context and noise level. The loss is just the squared error between predicted and true velocity. Key point: both audio and video velocities are predicted from the SAME noisy inputs and SAME context, which couples the two modalities.

---

## Training Details

### Stage 1: Independent-Task Pretraining

- **Init:** Transformer from a pretrained language model (Qwen2.5/Qwen3 family)
- **Tasks mixed:**
  - Understanding: image/audio/video understanding, text dialogue, ASR, TTS
  - Generation: image generation, audio generation, video generation, joint audio-video
- **Goal:** Align perception, language reasoning, and latent generation in one sequence model

### Stage 2: End-to-End Interaction Training

- **Data:** Duplex interaction data where user text/audio/video and agent text/audio/video are interleaved
- **Goal:** Adapt from independent tasks to real-time setting
- **Learned behaviors:** Response timing, active listening, interruption handling, long-context consistency

### Stage 3: Distillation for Low-Latency Streaming

- **Teacher:** Stronger model with CFG and more flow-matching solver steps
- **Student:** Efficient model for deployment (fewer steps, no CFG at inference)
- **Rolling distillation:** Student is rolled out over consecutive streaming units and trained on its OWN generated history (self-forcing) with distribution matching
- **Purpose:** Reduce train-test mismatch for long-form generation

### Missing details:
- No parameter count
- No dataset sizes or composition specifics
- No training compute (GPU-hours, FLOPs)
- No hyperparameters (learning rate, batch size, etc.)

---

## Results

### Latency Comparison

| System | Type | User-visible Latency | Notes |
|--------|------|---------------------|-------|
| Doubao Realtime Voice | speech-to-speech | ~1s overall | Speech-only product |
| GPT-4o Realtime API | speech-to-speech | ~500ms API TTFB; ~800ms voice-to-voice | No visual output |
| Hume EVI 3 | speech-to-speech | 0.9–1.4s web-app benchmark | No visual output |
| Gemini Live API | speech-to-speech | 1.2–3.6s API benchmark | No visual output |
| Moshi | speech-to-speech | ~200ms model latency | Native full-duplex but no visual |
| Qwen3/3.5-Omni | audio-video in, speech out | 234–547ms first-packet | No visual avatar generation |
| **Wan-Streamer** | **text/audio/video in+out** | **~550ms total (incl. 350ms network)** | **Single end-to-end model** |

### Visual Agent Comparison

| System | Scope | Runtime | Difference from Wan-Streamer |
|--------|-------|---------|---------------------------|
| Body of Her | end-to-end humanoid | 42ms/frame at 24 FPS | No deployed signal-to-signal latency |
| X-Streamer | video chat from portrait | 25 FPS on 2×A100 | Absolute response latency undisclosed |
| VASA-1 | audio-driven talking face | 40 FPS, 170ms preceding | Renderer only, no dialogue |
| TalkingMachines | audio-driven video | real-time chunks | External audio LLM needed |
| StreamAvatar | streaming avatar | FFD 0.33–0.39s | No unified dialogue model |
| AvatarForcing (Cui) | one-step streaming avatar | 34ms/frame | Not perceptual dialogue |
| Hallo-Live | text-driven avatar | 20.38 FPS, 0.94s latency | Text-driven, no user perception |
| **Wan-Streamer** | **full perceptual dialogue + video** | **25 FPS, ~550ms total, ~200ms model** | **Single causal Transformer** |

### Naturalness & Full-Duplex Behavior

The paper claims (qualitatively):
- **Idle state:** Agent maintains identity, gaze, posture, breathing — doesn't freeze
- **Listening state:** Responsive non-verbal feedback (gaze shifts, nods, micro-expressions) coupled with user's speech
- **Interruption:** Model keeps consuming user audio-video even while generating response, can stop/redirect when interrupted
- **Proactive speaking:** Can initiate comments based on visual observations without waiting for explicit request
- **Lip sync:** Native synchronization because audio and video are predicted from the same context before decoding

### Ablations

**None.** This is the biggest weakness of the paper. There are zero ablation studies.

---

## Limitations

1. **192p resolution** — very low quality, described as "proof of concept"
2. **No quantitative quality metrics** — no FID, FVD, ASR-WER, user studies, MOS scores
3. **No ablations** — can't tell which design choices matter
4. **No code or model release** — not reproducible
5. **No model size disclosed** — can't assess computational requirements
6. **Latency comparisons are asymmetric** — they compare full end-to-end path against partial metrics from other systems, making direct comparison hard
7. **Training data is vague** — "broad mixture" with no specifics on curation, filtering, or licensing

---

## Open Questions

1. How well does this scale to higher resolutions? They claim it's straightforward but the compute requirements for flow-matching over video latents will increase dramatically.
2. How much training data is needed for the end-to-end interaction stage? This is presumably rare and expensive.
3. What happens with longer conversations — does the KV cache become unwieldy? Is there context window management?
4. How does the joint audio-video flow matching actually work in practice? Does one modality degrade to help the other?
5. Can the thinker-performer split work on a single GPU with pipelining, or does it fundamentally need two?
6. What's the actual model size and can this run on consumer hardware?
7. How does the quality compare to cascaded systems at equal compute budget?
