# Notes — SwanTale: Unified Multi-Speaker Speech and Audio Generation

> **Paper:** SwanTale: Unified Multi-Speaker Speech and Audio Generation for Instruct and Zero-Shot Tasks
> **Authors:** Yu Zhang, Ruiqi Li, Changhao Pan, Ke Lei, Xiang Yin, Cheng Yang (ByteDance / Zhejiang University)
> **Year:** 2026 | **ArXiv:** 2608.02023 (eess.AS)
> **First pass — 2026-08-08 | Second pass (deep) — 2026-08-09**

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

# ===================== SECOND PASS (DEEP) =====================
> Re-read section by section on 2026-08-09. Below I resolve every open
> question from the first pass, rewrite the math in plain English, and record
> the exact equations I'll need for the breakdown and implementation.

## Flow-matching convention — THE first thing to nail down

SwanTale uses the **noise-to-data** convention (Lipman-style, but with endpoints
swapped vs my usual mental model):

- `x_t = (1 − t)·ε + t·x★`  where `t ∈ [0,1]`, `ε ~ N(0,I)`.
- **t=0 → pure noise ε**; **t=1 → clean data x★**.
- Velocity target: `v = dx_t/dt = x★ − ε`.
- This is *opposite* to the data-to-noise DDPM-style `x_t = √α·x + √(1−α)·ε`.
  I MUST keep this convention consistent in the implementation.

So the DiT predicts `v̂_θ ≈ (x★ − ε)`, and the flow ODE is `dx/dt = v_θ(x_t, t)`.

**Task-specific masking (Eq. 6–9)** — the elegant unification:
- Task type `τ ∈ {inst, zero}`. Target latent `x★ ∈ R^{T×d_z}`.
- Caption: `c^(inst) = c_full` (environment+speakers+content), `c^(zero) = c_content` (content only).
- Context mask: `m^(inst) = 0` (generate ALL frames), `m^(zero) = m_prompt` (1 on reference frames, 0 on generate frames).
- Reference: `r^(τ) = m^(τ) ⊙ x★` (reference frames kept clean).
- Noising only the generation region (1−m): `x̃_t = (1−m) ⊙ ((1−t)ε + t·x★) + m⊙x★`.
  i.e. reference frames stay at x★, generation frames get flow-noised.
- Loss: masked MSE `L_flow = E[ (1/T_gen) · Σ_{i: m_i=0} ‖v̂_θ − (x★−ε)‖² ]`.
  Denominator uses `max(1, T_gen)` to avoid div-by-zero on all-prompt batches.

This is the key insight: **one backbone, one velocity objective, two tasks differ
only in caption content + which frames are masked.** Instruct = full caption, no
prompt context. Zero-shot = content caption + reference frames as fixed context.

## Engram layer — fully resolved (Eq. 4–5)

It's a **retrieval-augmented key-value memory for fixed n-gram patterns** in the
caption, bolted onto the caption-encoder output before cross-attention.

**Setup:**
- Caption token sequence `c = (c_1, ..., c_L)`.
- Order set `N = {2, 3}` (bigrams + trigrams).
- For position `i`, the **centered** n-gram window:
  `w_i^(n) = c_{i−⌊(n−1)/2⌋ : i+⌊n/2⌋}`.
  (Centered because the caption branch is non-causal/bidirectional — the original
  Engram [14] used suffix windows for autoregressive backbones.)
- Each window is hashed by `hash_k` into K **head-specific** lookup tables.
- Retrieved slot values are concatenated across orders n and heads k:
  `e_i = concat_{n∈N, k=1..K} [ Engram_{n,k}( hash_k(w_i^(n)) ) ]`.
- Concatenating *before* read-out lets ONE projection pair weigh orders against
  each other (instead of forcing every order to contribute equally).

**Gated residual update (Eq. 5):**
Given projected caption embedding `u_i`:
```
ũ_i = u_i + σ( (RMSNorm(u_i)·RMSNorm(W_K e_i)) / √d + b ) · (W_V e_i)
```
- The gate `σ(·)` is **content-dependent** (dot product of normalized u_i and
  normalized projected memory key), so the model uses Engram strongly for
  structured markers ("an energetic girl", "a train whistle") and weakly for
  free-form natural language.
- **Two gate branches** share the tables + W_V but keep **separate W_K**; their
  gated outputs are **averaged**.
- **Bias `b` initialized NEGATIVE** → memory path starts nearly closed and
  gradually opens during training. (Crucial — prevents the untrained memory from
  corrupting the caption representation early on.)

→ Engram = "recognize recurring caption phrases via hashed n-gram lookup, inject
as a gated residual, start off, warm up during training." Separates fixed-pattern
recognition from long-range acoustic planning that cross-attention handles.

## Unified MoE routing — fully resolved (Eq. 10–32)

Replaces every **2nd** DiT FFN. Each MoE-FFN has: `R` routed audio experts, `S`
task-shared experts, `U` null experts.

**Task router (sample-level, Eq. 10):** picks a fixed set `T_τ` of shared experts
per task, summed: `o_shared(h,τ) = Σ_{j∈T_τ} E_j^shared(h)`. Same experts reused
across all layers/frames of a sample.

**Audio router (frame-level, the interesting part):**

1. **Time-injected frame rep (Eq. 11):** `r_{ℓ,m} = h_{ℓ,m} + W_t·e_t`
   (add projected time embedding to the post-attention hidden state).

2. **Base router logits (Eq. 12):** `ℓ_{ℓ,m} = W_g·r_{ℓ,m} + b_null(t)`
   where `b_null(t)` is non-zero ONLY on null-expert dims, zero on routed dims.

3. **Selection logits (Eq. 13):** `a_{ℓ,m} = ℓ_{ℓ,m} + b`
   where `b` is a **load-correction bias** (non-zero on routed dims, zero on null).

4. **Load balancing (Eq. 14, auxiliary-loss-free [91]):** for routed expert i,
   `f_i` = fraction of recent non-null assignments. Update:
   `b_i ← clip( b_i + η·(1/R − f_i), −B, B )`.
   Overloaded expert → lower b → less likely selected. Underloaded → higher b.
   If a layer produces NO non-null assignments, bias left unchanged.

5. **Time-aware budget (Eq. 15–18):** `q(t) = σ(W_b·e_t)` ∈ (0,1), predicted from
   time embedding. Jointly controls THREE things:
   - Top-P threshold: `p(t) = p_min + (p_max − p_min)·q(t)`
   - Null bias: `b_null(t) = b_max^null + (b_min^null − b_max^null)·q(t)` (on null dims)
   - Capacity factor: `c(t) = c_min + (c_max − c_min)·q(t)`
   - Larger q → higher threshold (more experts), weaker null preference, more
     capacity. Smaller q → shift compute to shared/null paths at that time.

6. **Dynamic Top-P + annealed Gumbel (Eq. 19–22) — TRAINING:**
   - Draw Gumbel noise `g_{ℓ,m,i} = −log(−log u)`, `u~U(0,1)` for all experts.
   - **Selection distribution** (from selection logits a):
     `π^{g,sel}_{ℓ,m,i} = softmax((a_{ℓ,m,i} + g_{ℓ,m,i}) / τ_g^(n))`.
   - Sort by π^{g,sel} desc, take smallest prefix whose cumulative prob ≥ p(t) → `S_{ℓ,m}`.
   - **Mixture weights** (from BASE logits ℓ, NOT a!) — renormalized over S:
     `π̃_{ℓ,m,i} = softmax((ℓ_{ℓ,m,i} + g_{ℓ,m,i})/τ_g^(n))` for `i ∈ S_{ℓ,m}`.
   - **KEY SUBTLETY (was open question, now resolved):** selection uses
     bias-adjusted logits `a`, but combination weights use base logits `ℓ`.
     This is the auxiliary-loss-free design [91]: load bias affects *who gets
     picked* but not *how much weight they get once picked*.
   - Gumbel temp `τ_g^(n) ↘ τ_min > 0` over training steps — hot start encourages
     exploration over expert combos, then sharpens so experts specialize.

7. **Inference (Eq. 23–24) — deterministic, no Gumbel:**
   - Selection: `π^{sel} = softmax(a_{ℓ,m})`, sort, Top-P prefix → S.
   - Weights: `π̄_{ℓ,m,i} = softmax(ℓ_{ℓ,m,i})` for i ∈ S.
   - Denote both train/infer weights as `π̂_{ℓ,m,i}`.

8. **Output (Eq. 25–28):**
   - Routed audio experts only: `S̃_{ℓ,m} = S_{ℓ,m} ∩ {1..R}`.
   - `o_audio(h_{ℓ,m},t) = Σ_{i∈S̃} π̂_{ℓ,m,i}·E_i^audio(h_{ℓ,m})`.
   - If `S̃ = ∅` → zero output (skip path via null).
   - Null experts participate in normalization (so their prob mass scales down
     the audio branch magnitude) but produce NO feed-forward output.
   - `o_MoE = o_shared + o_audio`, added via the DiT block's residual.

9. **Capacity (Eq. 29):** `cap = c(t)·(M/R)` where M = total non-null assignments.
   Overflow → keep largest-weight assignments, drop rest.

10. **Auxiliary losses (Eq. 30–32):**
    - `L_z = E[ (log Σ_j exp(ℓ_{ℓ,m,j}))² ]` — z-loss on base logits (stability [119]).
    - `L_null` = avg prob mass on null experts (prevents null-collapse).
    - `L_MoE = λ_z·L_z + λ_null·L_null`, annealed `ω_MoE(n)` to a small floor.
    - Total: `L = L_flow + ω_MoE(n)·L_MoE`.

→ Unified MoE = "task-level shared FFN + frame-level dynamic-capacity sparse FFN
whose budget shrinks/grows with diffusion time, with a cheap null-skip for silence,
all stabilized by z-loss + null-collapse penalty + auxiliary-loss-free load bias."

## Reward-conditioned quality control — fully resolved

Not RL. It's a **conditioning trick** (Eq. in 3.2):
- During preprocessing, compute STOI/PESQ/SI-SDR/MOS for each sample.
- Format a **quality caption**: "Quality: speech clarity {STOI level}; noise
  level {SI-SDR level}; signal naturalness {PESQ level}; listening quality {MOS
  level}." Prepend to global caption before the content field.
- Also map scores → **quality flag** `q ∈ {low, normal, high, unknown}`.
- During training, drop the flag to `unknown` with some probability (for CFG).
- At inference: **force `high` + high-quality caption**.

→ The model learns "what does high-quality speech sound like" as a conditional
distribution. No rollout, no reward model in the loop, no extra sampling. This is
the "reward-conditioned policy" framing [58]. Brilliant cost-saving trick.

## GRPO / stochastic flow policy — fully resolved (Eq. 33–42)

**Why not just use the deterministic ODE for RL?** The flow ODE `dx = v_θ dt` is
deterministic → transition distribution is degenerate (a delta) once initial noise
is fixed → no meaningful log-prob for policy gradient.

**Solution: construct a marginal-preserving SDE (Eq. 36):**
```
dx_t = b_θ(x_t,t,c) dt + η(t) dW_t
b_θ = v_θ + (η(t)²/2)·s_θ
```
Under noise-to-data convention `x_t = (1−t)x_0 + t·x_1` (x_0=noise, x_1=data):
- **Score estimate (Eq. 37):** `s_θ(x_t,t,c) = (t·v_θ − x_t)/(1−t)`
- **Diffusion schedule (Eq. 37):** `η(t) = sqrt(t/(1−t))`, endpoints clamped.
- This SDE has the SAME marginals as the deterministic flow (marginal-preserving),
  but now transitions are Gaussian → tractable log-probs for GRPO.

**Euler–Maruyama transition (Eq. 38):** for `t_j < t_{j+1}`:
```
π_θ(x_{j+1}|x_j,c) = N(µ_{θ,j}, η(t_j)²·Δt_j·I)
µ_{θ,j} = x_j + b_θ(x_j,t_j,c)·Δt_j
```
Noise + likelihoods applied ONLY to generated region G; reference frames fixed.

**GRPO objective (Eq. 39–40):**
- Frozen behavior policy samples K=8 trajectories per condition (independent SDE noise).
- Group-relative advantage: `A_i = (R_i − µ_R(c))/(σ_R(c) + ε)`, clipped to `Ã_i ∈ [−A_max, A_max]`, A_max=2.
- `ρ_{i,j} = exp(ℓ_{θ,i,j} − ℓ_{old,i,j})` (importance ratio on stored transition).
- **Per-element MEAN log-prob** (NOT sum): `ℓ = mean over valid latent elements in G`.
  → Resolves the open question: a SUMMED log-likelihood scales with #elements, so
  a per-element discrepancy of 1e-3 already moves ρ by 10 orders of magnitude and
  saturates clipping. The MEAN keeps ρ near 1, making ε comparable across lengths.
- Clipped objective (Eq. 40): `L_GRPO = −E[ min(ρ·Ã, clip(ρ,1−ε,1+ε)·Ã) ]`.

**Reward design (Eq. 33–35):**
- 5 shared speech rewards: phone_core (substitution), phone_len (del/ins),
  pause_punct (pause↔punctuation match), edge_rms (boundary energy), quality (clipping/artifacts).
- Instruct: + attribute agreement (SwanVerifier age/gender vs caption).
- Zero-shot: + speaker similarity `r_sim = (1+cos(f_spk(x̂),f_spk(x_ref)))/2`.
- All calibrated to [0,1]. Total (Eq. 35): weighted avg with pronunciation gate `g_i`:
  `R_i = g_i · (Σ_m λ_m r_{i,m}) / (Σ_m λ_m)`.

**Capability preservation (Eq. 41–42):**
- KL to frozen SFT ref policy, **closed-form** (same variance!):
  `D_KL(π_θ‖π_ref) = ‖µ_{θ,j} − µ_{ref,j}‖²_G / (2·η(t_j)²·Δt_j)`.
  (Averages over valid elements G, matching the log-prob normalization.)
- After each GRPO block: **supervised anchor replay** on original multi-speaker/
  audio SFT mixture with standard flow-matching loss `L_anchor = L_flow(D_anchor)`.
- Behavior policy refreshed only after BOTH update blocks.

→ The per-element mean + closed-form KL (exploiting same-variance Gaussians) +
anchor replay are the three engineering tricks that make flow-GRPO tractable.

## Inference: two-stage decomposed CFG + sway sampling (Eq. 43)

**Two-stage decomposed CFG (Eq. 43):**
```
ṽ_t = v_∅ + ω_text(t)·(v_text − v_∅) + ω_all(t)·(v_full − v_text)
```
- Three model evals: null (unconditional), text+speaker-turn only, full condition.
- `ω_text` guides content/speaker-turn consistency; `ω_all` adds task-specific full condition.
- For instruct: full = full caption + quality flag. For zero-shot: full = content caption + reference context.
- **Timestep-dependent guidance annealing:** `ω_k(t) = γ(t)·ω̄_k`, `γ(t) = a + b(1−t)^p`.
  Stronger guidance early (coarse structure), weaker near data endpoint (refine).
  Reported: (a,b,p) = (0.6, 0.6, 1.0), base weights [1.5, 3.0].

**Sway sampling:** warp uniform grid `u∈[0,1]` to `t(u) = 1 − cos(πu/2)`.
→ Non-uniform integration: more Euler steps early (where structure is built),
fewer late. Just a reparameterization of the integration grid.

## Architecture details confirmed

**SwanVAE** (407M total: 51.7M enc + 0.3M bottleneck + 355M dec):
- Encoder: weight-normed 1D conv → 5 downsample stages [4,4,4,5,6]=1920 hop.
  Each stage: 3 residual units (dilation 1,3,9) + strided projection. Fixed
  low-pass before decimation (anti-aliasing [104]). Channels 64→1536. No Transformer.
  RF ≈ 0.95s. Learnable HF-envelope shortcut alongside.
- Bottleneck: two 1×1 convs → µ_ϕ, log σ²_ϕ. Gaussian VAE, KL weight 0.02.
- Decoder: TRB (SAME [74]). Each latent + 6 learnable output tokens → local
  bidirectional Transformer → 6 patches × 320 samples = 1920 samples = 40ms.
- Posterior mean µ_ϕ (globally normalized) used as SwanTale's deterministic target.
- Training-only alignment objectives (on µ_ϕ only): flow-matching predictor,
  causal latent predictor, multi-scale chroma, multi-band energy. All discarded.
- On-the-fly source mixing (25% of crops, α~U(0.3,0.7)). Effective-bandwidth aug
  (1% of crops: downsample to 8–44.1kHz then back to 48kHz).
- Discriminators: MPD + MRD + MBCSD (multi-band complex STFT). All discarded post-train.

**Flow DiT (2B active params):**
- Qwen3.0-Instruct-8B caption encoder (→32B in ablation).
- CosyVoice 2.0 text tokenizer + lightweight Transformer text encoder.
- AdaLN-Zero timestep modulation [75], RMSNorm [102].
- Quality flag embedding added to global conditioning stream (with timestep emb).

## Ablation takeaways (Section 4.6, Table 8 — SwanBench-Caption)

| Setting | Instr. Acc | Acoustic Qual | Overall Expr |
|---|---|---|---|
| w/o Unified MoE | 3.02 | 4.09 | 3.56 |
| SwanTale (default) | 3.39 | 4.31 | 3.82 |
| w/ 32B caption encoder | 3.70 | 4.34 | 3.98 |

- Unified MoE helps ALL THREE metrics (instr realization + quality + expressiveness).
- Scaling caption encoder 8B→32B raises all three, **largest gain in Instruction Accuracy** (3.39→3.70).
- → Caption understanding capacity is a real bottleneck for instruction-following.

## What's implementable from scratch (for the coding step)

The full system is too heavy (2B params, 64×A100, Qwen encoder, SwanVAE). But the
**conceptually novel + self-contained** pieces are tractable on synthetic latents:

1. **Unified MoE router** (task router + dynamic Top-P audio router + null experts +
   time-aware budget + annealed Gumbel + auxiliary-loss-free load bias + z-loss +
   null-collapse penalty). This is the most algorithmically rich piece (Eq. 10–32).
2. **Flow-matching training with task-specific masking** (Eq. 6–9) — one backbone,
   two tasks via masking. Tractable on toy 2D/synthetic latents.
3. **Engram layer** (Eq. 4–5) — hashed n-gram memory + gated residual. Standalone module.
4. **Two-stage decomposed CFG + sway sampling** (Eq. 43) — inference procedure.

Plan for implementation/: build a tiny flow-matching DiT on synthetic latents
(e.g., 2D toy distribution or small audio-latent surrogate) with the Unified MoE
FFN and task-specific masking, demonstrating that one backbone trains on two
"tasks" (masked vs unmasked generation) via the masking trick, and that the MoE
router produces sensible load-balanced dynamic-capacity routing.

---

## Terms / concepts — RESOLVED (second pass)

| First-pass question | Resolution |
|---|---|
| Engram hashing/gate | Centered n-gram windows → K head-specific hash tables → concat → gated residual with negative-init bias, 2 branches (shared tables+V, separate K) averaged |
| Dynamic Top-P vs Gumbel | Selection uses bias-adjusted logits `a`; mixture weights use base logits `ℓ` (aux-loss-free [91]); Gumbel annealed τ_g↘τ_min |
| Stochastic flow policy | Marginal-preserving SDE: b=v+η²/2·s, s=(tv−x)/(1−t), η=√(t/(1−t)); Euler-Maruyama Gaussian transitions; per-element MEAN log-prob |
| Per-element mean log-prob | Keeps ρ≈1 across utterance lengths; summed log-prob saturates clipping at 1e-3 per-element error |
| MBCSD | Multi-band complex STFT discriminator — partitions freq axis into bands, separate conv stack per band on real+imag STFT components |
| Sway sampling | t(u)=1−cos(πu/2), just a non-uniform ODE integration grid (more steps early) |
| Flow convention | Noise-to-data: x_t=(1−t)ε+tx★, v=x★−ε, t=0 noise, t=1 data |
| KL closed form | Same-variance Gaussians → D_KL = ‖µ_θ−µ_ref‖²/(2η²Δt), averaged over valid elements G |

## Second-pass takeaways

- This is genuinely a **composition paper** — the novelty is in the *assembly* and
  the two clever tricks (**reward-conditioned quality** and **per-element-mean
  flow-GRPO**), not in any single brand-new algorithm.
- The **Unified MoE** is the most algorithmically dense component and the best
  candidate for a faithful from-scratch reimplementation. The dual-router (task +
  audio) + dynamic Top-P + null experts + time-aware budget is a clean, complete
  design with well-specified math (Eq. 10–32).
- The **task-masking unification** (Eq. 6–9) is the simplest big idea: one velocity
  objective, two tasks, differ only in caption + mask. Elegant and implementable.
- The data pipeline (SwanData-Caption) is the biggest *practical* contribution but
  is not re-implementable (needs 70M records + MLLM annotators + human audit).
- The GRPO stack is clever but heavy; the per-element-mean + closed-form KL tricks
  are the transferable insights.
