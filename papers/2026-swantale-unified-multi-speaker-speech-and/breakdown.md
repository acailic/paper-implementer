# Breakdown — SwanTale: Unified Multi-Speaker Speech and Audio Generation

> **Paper:** SwanTale: Unified Multi-Speaker Speech and Audio Generation for Instruct and Zero-Shot Tasks
> **Authors:** Yu Zhang, Ruiqi Li, Changhao Pan, Ke Lei, Xiang Yin, Cheng Yang (ByteDance / Zhejiang University)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2608.02023
> **Code (official):** none published (project page: https://swanaigc.github.io/#swantale)

---

## 1. Problem & Motivation

**The problem.** Media creators (animation dubbing, ads, games, podcasts, audio
drama, short-video) need a *single* system that can:

1. **Design a voice from scratch** using a natural-language description (no
   reference recording) — the **instruct** task. The caption specifies speaker
   persona, acoustic environment, and fine-grained content.
2. **Clone / reuse a voice** from a short reference clip — the **zero-shot**
   task (classic TTS), while keeping multi-speaker dialogue capability.
3. Embed the speech inside an **acoustic scene** (café ambience, wind,
   footsteps) with correctly-timed, non-speech audio effects, all in one
   waveform rather than a stitched downstream pipeline.

**Why it matters.** Existing systems each do *one* of these. Instruct-TTS makes
only speech (not scenes); zero-shot-TTS cannot take captions; and stitching
speech + SFX in a downstream pipeline causes timing, reverb, and loudness drift.
A unified model removes the integration burden and lets one system serve the
full creator workflow.

**Prior approaches and their limitations.**
- **Instruct-TTS / caption-conditioned TTS:** cannot clone a specific voice and
  ignores non-speech audio.
- **Zero-shot voice cloning:** conditions on reference audio, not on captions,
  so it cannot design voices from scratch or reproduce a described scene.
- **Separate speech + audio-effect pipelines:** introduce cross-module drift.
- **Single-task flow-matching / diffusion audio models:** do not unify the two
  conditioning paths, so joint training weakens both.

**Three stated challenges the authors target:**
1. **Data scarcity** — rich multi-level captions are expensive to annotate.
2. **Task compatibility** — instruct (caption→style) vs zero-shot (audio→style)
   conditioning paths can weaken each other under joint training.
3. **Multi-audio-modality complexity** — speech, SFX, singing, music have
   different temporal structures and must coexist in one waveform.

---

## 2. Key Insight / Contribution

**Core idea (in my own words).** One non-causal flow-matching Transformer can
serve *both* the instruct and zero-shot tasks *and* generate non-speech audio,
because the two tasks differ **only in two things**: (a) the content of the
caption fed to the model, and (b) which frames of the latent are masked as
fixed reference context. Beyond that backbone, the paper stacks four
engineering contributions that make this practical at scale:

- **SwanData-Caption** — a data pipeline that turns messy media audio into
  clean, structured, fine-grained captions (Environment / Speakers / Content).
- **SwanVAE** — an asymmetric audio VAE producing smooth 25 Hz continuous
  latents that are easy to model by flow matching.
- **Unified MoE** — a dual-router (task-level shared experts + frame-level
  dynamic-capacity sparse audio experts with null-skip) that spends compute
  where the waveform actually needs it.
- **Reward-conditioned quality control + GRPO post-training** — quality is fed
  in as a *condition* (forced to "high" at inference), and a marginal-preserving
  stochastic flow policy enables turn-free GRPO with a per-element mean log-prob.

**What is genuinely new.** The novelty is mostly in *assembly* plus two clever
tricks: (1) **reward-conditioned quality** turns a hard RL problem into a
conditional-generation problem with no rollout, and (2) the **per-element mean
log-prob + closed-form KL** make flow-based GRPO tractable by keeping the
importance ratio near 1 across utterance lengths.

---

## 3. Method

### 3.1 Overview

The system has two sides that feed one generative backbone:

```
                 SwanData-Caption (data)              SwanTale (model)
        ┌─────────────────────────────┐      ┌──────────────────────────────┐
 media ─► 1. coverage design          │      │ SwanVAE  ── 48kHz → 25Hz     │
 audio  │ 2. preprocessing (vocal     │ text │   (asymmetric, 407M)         │
        │    sep, diarization, ASR,   │ caps │      │ continuous latents    │
        │    alignment)               │──────►      ▼                       │
        │ 3. caption annotation       │      │ Flow DiT (2B active)         │
        │    (MLLM → Env/Spk/Content) │      │   ├ caption branch (Qwen)    │
        │ 4. refinement (quality flt, │      │   │   + Engram memory        │
        │    verifier, human audit)   │      │   ├ text branch (CosyVoice) │
        │   → ~70M caption records    │      │   ├ reward-conditioned qual │
        └─────────────────────────────┘      │   └ Unified MoE FFN         │
                                              │      │                      │
                                              │      ▼                      │
                                              │ curriculum (4 stages)       │
                                              │   → GRPO post-training      │
                                              │   → 2-stage CFG + sway samp │
                                              └──────────────────────────────┘
```

### 3.2 Architecture

```
                      ┌──────────── SwanVAE (3.1) ───────────┐
 48kHz mono ──► Encoder: anti-aliased CNN downsample [4,4,4,5,6]
              (1920-sample hop, 64→1536 ch, no Transformer, RF≈0.95s)
                          │
                  Bottleneck: Gaussian VAE  →  µ_ϕ, log σ²_ϕ  (96-dim)
                          │   (posterior mean µ_ϕ = deterministic target)
                          ▼
              Decoder: SAME-style Transformer Resampling Block
              (latent + 6 learnable tokens → bidir Transformer → 6×320 samples)
                          │
                        48kHz mono  ◄──────── (407M total: 51.7M enc, 355M dec)

                      ┌──────────── Flow DiT (3.2–3.3) ───────┐
  caption text ──► Qwen encoder ──► Engram layer ──► cross-attn ──┐
  (Env/Spk/Content)                                          (all layers)
  quality flag  ──► label emb ─────────────────────────────────►│ global cond
                                                                │
  content text  ──► CosyVoice tokenizer ──► light Tx ───────────►│ length-norm
  (+spk-turn emb)                                              │ onto latent
                                                               ▼
  noise ε ──►  x_t = (1−t)ε + t·x★  ──► [ AdaLN-Zero DiT blocks ]
                                            every 2nd FFN replaced by:
                                          ┌──── Unified MoE (3.3) ────┐
                                          │ task router (sample-level)│
                                          │  → shared experts T_τ     │
                                          │ audio router (frame-level)│
                                          │  → dynamic Top-P + null   │
                                          │  → time-aware budget q(t) │
                                          └───────────────────────────┘
                                               │
                                               ▼
                                       velocity  v̂_θ(x_t, t, c)
```

### 3.3 Forward pass / pipeline

**Training (flow matching with task-specific masking).**
1. Take a clean latent target `x★` (the SwanVAE posterior mean `µ_ϕ`).
2. Sample `t ∈ [0,1]`, noise `ε ~ N(0,I)`.
3. Build the noised latent, masking the reference region so it stays clean:
   `x̃_t = (1−m) ⊙ ((1−t)ε + t·x★) + m ⊙ x★`.
4. Forward through the DiT to get velocity prediction `v̂_θ`.
5. Velocity target is `v = x★ − ε` (noise-to-data convention).
6. Masked MSE over generation frames only.

**The task unification (the elegant part):**
- Instruct task `τ = inst`: full caption `c_full`; context mask `m = 0`
  (generate ALL frames).
- Zero-shot task `τ = zero`: content-only caption `c_content`; context mask
  `m = m_prompt` (1 on reference frames, 0 on generation frames).

→ *One backbone, one velocity objective; two tasks differ only in caption
content and which frames are masked.*

**Inference (3.6).** Two-stage decomposed classifier-free guidance + sway
sampling (see §4).

### 3.4 Loss function

Total training loss:
```
L = L_flow + ω_MoE(n) · L_MoE
```
where
- `L_flow = E[ (1/max(1,T_gen)) · Σ_{i: m_i=0} ‖v̂_θ − (x★ − ε)‖² ]`
- `L_MoE = λ_z · L_z + λ_null · L_null` (annealed weight `ω_MoE(n)`)
- `L_z = E[ (log Σ_j exp(ℓ_{ℓ,m,j}))² ]` — z-loss on base router logits
- `L_null` = average probability mass on null experts (anti-collapse)

Plus SwanVAE's own reconstruction stack (multi-res STFT + multi-res multi-band
Mel + frame energy + KL + adversarial MPD/MRD/MBCSD + feature matching) and
training-only latent-alignment objectives on `µ_ϕ`.

---

## 4. Math

### 4.1 Flow-matching convention (noise-to-data)
```
x_t = (1 − t)·ε + t·x★ ,   t ∈ [0,1], ε ~ N(0,I)
v   = dx_t/dt = x★ − ε
```
- `t = 0` → pure noise `ε`; `t = 1` → clean data `x★`.
- DiT predicts `v̂_θ ≈ (x★ − ε)`; flow ODE is `dx/dt = v_θ(x_t, t)`.

### 4.2 Task-specific masking (Eq. 6–9)
```
caption:   c^(inst) = c_full ;  c^(zero) = c_content
mask:      m^(inst) = 0       ;  m^(zero) = m_prompt
reference: r^(τ) = m^(τ) ⊙ x★
noised:    x̃_t = (1−m)⊙((1−t)ε + t·x★) + m⊙x★
loss:      L_flow = E[ (1/max(1,T_gen)) · Σ_{i:m_i=0} ‖v̂_θ − (x★−ε)‖² ]
```
Plain English: only the *generation* frames get flow-noised; reference frames
stay at the clean target. The `max(1, T_gen)` avoids divide-by-zero when a
batch is all-prompt.

### 4.3 Engram layer (Eq. 4–5) — hashed n-gram memory
```
orders N = {2, 3}
centered window: w_i^(n) = c_{i−⌊(n−1)/2⌋ : i+⌊n/2⌋}
retrieve:        e_i = concat_{n∈N, k=1..K}[ Engram_{n,k}(hash_k(w_i^(n))) ]
update:          ũ_i = u_i + σ( (RMSNorm(u_i)·RMSNorm(W_K e_i))/√d + b ) · (W_V e_i)
```
- `u_i` = projected caption embedding; `e_i` = retrieved memory values.
- Gate `σ(·)` is content-dependent → strong for structured markers, weak for
  free-form text. Bias `b` initialized **negative** → memory starts closed.
- Two gate branches share tables + `W_V` but keep separate `W_K`; outputs
  averaged.

### 4.4 Unified MoE routing (Eq. 10–32)

**Task router (sample-level):**
```
o_shared(h, τ) = Σ_{j∈T_τ} E_j^shared(h)
```

**Audio router (frame-level):**
```
time-injected:   r_{ℓ,m} = h_{ℓ,m} + W_t·e_t
base logits:     ℓ_{ℓ,m} = W_g·r_{ℓ,m} + b_null(t)     (b_null only on null dims)
selection logits:a_{ℓ,m} = ℓ_{ℓ,m} + b                  (b only on routed dims)
```

**Load balancing (aux-loss-free, Eq. 14):** for routed expert `i`, with recent
assignment fraction `f_i`:
```
b_i ← clip( b_i + η·(1/R − f_i), −B, B )
```

**Time-aware budget (Eq. 15–18):** `q(t) = σ(W_b·e_t)` jointly controls:
```
Top-P threshold:  p(t)   = p_min + (p_max − p_min)·q(t)
null bias:        b_null(t) = b_max^null + (b_min^null − b_max^null)·q(t)
capacity factor:  c(t)   = c_min + (c_max − c_min)·q(t)
```

**Dynamic Top-P + annealed Gumbel (training, Eq. 19–22):**
```
g_{ℓ,m,i} = −log(−log u),  u ~ U(0,1)
π^{g,sel} = softmax( (a_{ℓ,m,i} + g_{ℓ,m,i}) / τ_g^(n) )   ← selection
S_{ℓ,m}   = smallest prefix (sorted desc) with cum-prob ≥ p(t)
π̃_{ℓ,m,i}= softmax( (ℓ_{ℓ,m,i} + g_{ℓ,m,i}) / τ_g^(n) ),  i ∈ S   ← weights
```
- **Selection uses bias-adjusted logits `a`; weights use base logits `ℓ`.**
- Gumbel temp `τ_g^(n) ↘ τ_min > 0` over training steps.

**Inference (Eq. 23–24) — deterministic:**
```
π^{sel} = softmax(a_{ℓ,m});  S = Top-P prefix
π̄_{ℓ,m,i} = softmax(ℓ_{ℓ,m,i}),  i ∈ S
```

**Output (Eq. 25–29):**
```
S̃_{ℓ,m} = S_{ℓ,m} ∩ {1..R}
o_audio(h_{ℓ,m}, t) = Σ_{i∈S̃} π̂_{ℓ,m,i}·E_i^audio(h_{ℓ,m})   (0 if S̃=∅)
o_MoE = o_shared + o_audio
capacity: cap = c(t)·(M/R);  overflow → keep largest weights, drop rest
```

**Auxiliary losses (Eq. 30–32):**
```
L_z    = E[ (log Σ_j exp(ℓ_{ℓ,m,j}))² ]
L_null = avg prob mass on null experts
L_MoE  = λ_z·L_z + λ_null·L_null
```

### 4.5 Reward-conditioned quality control (§3.2)
A quality caption ("Quality: speech clarity {STOI}; noise {SI-SDR}; naturalness
{PESQ}; listening quality {MOS}") plus a quality flag `q ∈ {low, normal, high,
unknown}` are prepended to the caption. At inference the flag is **forced to
`high`**. No RL is involved — quality becomes a conditional variable.

### 4.6 GRPO stochastic flow policy (Eq. 33–42)

**Marginal-preserving SDE (noise-to-data):**
```
dx_t = b_θ(x_t,t,c) dt + η(t) dW_t
b_θ = v_θ + (η(t)²/2)·s_θ
s_θ(x_t,t,c) = (t·v_θ − x_t)/(1 − t)          ← score estimate
η(t) = sqrt( t / (1 − t) )                      ← diffusion schedule (endpoints clamped)
```

**Euler–Maruyama transition (Eq. 38):**
```
π_θ(x_{j+1}|x_j,c) = N( µ_{θ,j}, η(t_j)²·Δt_j·I )
µ_{θ,j} = x_j + b_θ(x_j,t_j,c)·Δt_j
```

**GRPO objective (Eq. 39–40):** K=8 trajectories per condition.
```
A_i = (R_i − µ_R(c)) / (σ_R(c) + ε),   clipped to Ã_i ∈ [−A_max, A_max], A_max=2
ρ_{i,j} = exp( ℓ_{θ,i,j} − ℓ_{old,i,j} )
ℓ = MEAN over valid latent elements in G                  ← per-element MEAN (not sum)
L_GRPO = −E[ min( ρ·Ã, clip(ρ,1−ε,1+ε)·Ã ) ]
```

**Reward (Eq. 33–35):** weighted average of pronunciation-gated rewards.
```
R_i = g_i · (Σ_m λ_m r_{i,m}) / (Σ_m λ_m)
```
Shared: phone_core, phone_len, pause_punct, edge_rms, quality. Instruct adds
attribute agreement; zero-shot adds speaker cosine similarity.

**Capability preservation (Eq. 41–42):** closed-form KL (same variance!) +
supervised anchor replay.
```
D_KL(π_θ‖π_ref) = ‖µ_{θ,j} − µ_{ref,j}‖²_G / (2·η(t_j)²·Δt_j)
L_anchor = L_flow(D_anchor)     ← original SFT mixture replayed after each GRPO block
```

### 4.7 Inference — two-stage decomposed CFG + sway (Eq. 43)
```
ṽ_t = v_∅ + ω_text(t)·(v_text − v_∅) + ω_all(t)·(v_full − v_text)
ω_k(t) = γ(t)·ω̄_k ,   γ(t) = a + b(1−t)^p      (reported a,b,p = 0.6, 0.6, 1.0)
sway:   t(u) = 1 − cos(πu/2)                      (non-uniform integration grid)
```
Three model evals: null (unconditional), text+speaker-turn only, full condition.

---

## 5. Training

**Data — SwanData-Caption (~70M caption records):**
1. **Coverage design** — media-style real data (drama, anime, ads, podcast) +
   3 targeted synthetic subsets (elderly speech, short utterances, hard
   pronunciation) of ~100k each, generated with a phoneme-aware TTS teacher.
2. **Preprocessing (SwanData-Speech)** — vocal separation (Ultimate Vocal
   Remover), diarization (3D-Speaker / CAM++), ASR (Seed-ASR 2.0 + SenseVoice
   check), alignment (SwanAligner). Original audio retained for scene captioning.
3. **Caption annotation** — Seed2.0 Lite MLLM produces Environment / Speakers /
   Content fields; a **style-persona library** (per-media-family style matrices)
   acts as a soft prior to avoid impoverished speaker descriptions.
4. **Refinement** — waveform filtering (DNSMOS / SQUIM PESQ/STOI/SI-SDR
   thresholds), caption normalization (SwanVerifier checks gender/age), human
   verification via **group-wise best–worst expressiveness comparison** (not
   absolute MOS, not A/B — more annotation-efficient).

**Curriculum learning (§3.4) — 4 stages:**
1. Zero-shot base (SwanVoice-style; single→multi-speaker; reference dropped 50%).
2. Dense caption adaptation on clean speech (MoE replaced by dense FFN), 70/30
   instruct/zero-shot mix.
3. Full caption mixture + introduce Unified MoE.
4. High-expressiveness / HQ SFT subset.

**GRPO post-training (§3.5):** marginal-preserving SDE, K=8 candidates,
group-relative advantage (no value model), per-element mean log-prob, KL to
frozen reference policy, supervised anchor replay.

**Optimizer / schedule / hyperparameters (model side):**
- SwanVAE: Gaussian VAE, KL weight 0.02; discriminators MPD + MRD + MBCSD.
- Flow DiT: 2B active params; AdaLN-Zero timestep modulation; RMSNorm.
- Caption encoder: Qwen3.0-Instruct-8B (→32B in ablation).
- Text encoder: CosyVoice 2.0 tokenizer + lightweight Transformer.
- MoE: replaces every 2nd DiT FFN; Gumbel temp `τ_g ↘ τ_min`; aux losses
  annealed to a floor; Top-P base weights reported `[1.5, 3.0]`, `(a,b,p)=(0.6,0.6,1.0)`.

**Compute budget:** 64×A100 GPUs. Industrial scale.

**Tricks / regularization:**
- Engram bias initialized negative (memory path starts closed).
- Reward-conditioned quality (quality flag forced "high" at inference).
- On-the-fly source mixing (25% of crops, α~U(0.3,0.7)); effective-bandwidth
  augmentation (1% of crops: downsample to 8–44.1kHz then back to 48kHz).
- Auxiliary-loss-free load balancing for the MoE.
- z-loss + null-collapse penalty for router stability.

---

## 6. Results & Ablations

**Headline numbers.**
- **SwanVAE:** best/near-best reconstruction on speech, singing, general audio,
  and music vs DAC / EnCodec / WavTokenizer / VoxCPM2 / MegaTTS3 / Same-L /
  ACE-Step.
- **Zero-shot (SwanBench-Speech):** ranks 1st in Timbre Consistency,
  Expressive Richness, and Expressive Hierarchy (mono & dialogue). Content
  accuracy / audio quality still have room to improve.
- **Instruct (InstructTTSEval):** 1st on Chinese APS (86.1), ties 1st on
  English APS (84.2), 2nd on Chinese DSD. English DSD & Role-Play are weaker.
- **SwanBench-Scene:** highest overall Mean MOS 4.22.

**Most important ablation (SwanBench-Caption, Table 8):**

| Setting | Instr. Acc | Acoustic Qual | Overall Expr |
|---|---|---|---|
| w/o Unified MoE | 3.02 | 4.09 | 3.56 |
| SwanTale (default) | 3.39 | 4.31 | 3.82 |
| w/ 32B caption encoder | 3.70 | 4.34 | 3.98 |

**What the ablations tell us:**
- The **Unified MoE helps all three metrics** — instruction realization,
  acoustic quality, and expressiveness. Removing it is a consistent regression.
- **Scaling the caption encoder 8B→32B raises all three, with the largest gain
  in Instruction Accuracy** (3.39→3.70). Caption understanding capacity is a
  real bottleneck for instruction-following — the language model doing the
  caption understanding matters as much as the acoustic generator.

---

## 7. Limitations

- **Scale & reproducibility.** 2B active params, 64×A100, ~70M caption records,
  and a Qwen caption encoder make a faithful end-to-end reimplementation
  impractical outside an industrial lab. The data pipeline (SwanData-Caption)
  is the biggest practical contribution but is not re-implementable without the
  70M records + MLLM annotators + human audit.
- **Weaker benchmarks.** English DSD and Role-Play are weaker than Chinese
  tasks; content accuracy and raw audio quality in zero-shot still have room.
- **Complexity.** Many stacked components (Engram, dual router, quality
  conditioning, curriculum, GRPO) — the paper does not fully isolate every
  component's marginal contribution, so some gains may be entangled.
- **No released code/weights** at time of reading (project page demos only).

---

## 8. Open Questions / Ideas

- **Unified MoE is the most algorithmically rich, self-contained piece** and
  the best candidate for a faithful from-scratch reimplementation: dual router
  (task + audio), dynamic Top-P, null experts, time-aware budget, annealed
  Gumbel, auxiliary-loss-free load bias, z-loss + null-collapse penalty
  (Eq. 10–32). I would build this on synthetic latents and verify load
  balancing + dynamic capacity.
- **Task-masking unification** (Eq. 6–9) is the simplest big idea and is
  tractable on toy 2D / synthetic latents: one velocity objective, two tasks
  differing only in caption + mask. Worth demonstrating empirically.
- **Engram layer** (Eq. 4–5) is a standalone module — hashed n-gram memory +
  gated residual with negative-init bias. Could be unit-tested in isolation.
- **Reward-conditioned quality** as a conditional variable (forced "high" at
  inference) is a transferable trick for any conditional generative model —
  worth trying on a small image/latent model.
- **Per-element mean log-prob + closed-form KL** are the transferable insights
  from the GRPO stack; the full SDE/GRPO loop is too heavy for a toy impl but
  the normalization trick generalizes to any flow-based RL.
- What would I try next? Ablate the **time-aware budget `q(t)`** schedule —
  does a learned `q(t)` actually track waveform structure (silence vs voiced),
  and how sensitive is quality to the budget floor?
