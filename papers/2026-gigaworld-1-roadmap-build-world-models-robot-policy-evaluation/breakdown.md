# GigaWorld-1 — Source-First Breakdown

**Paper:** GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation
**arXiv:** 2607.02642
**Source:** cs.RO / cs.CV — Tsinghua University (GigaAI team)

---

## Problem & Motivation

Evaluating embodied robot foundation models (VLAs, world-action models) requires slow, costly real-world rollouts limited by hardware and human supervision. Unlike LLM evaluation (cheap via digital benchmarks), robot policy validation demands physical robot hardware, continuous human monitoring, and occupies equipment for lengthy cycles. OpenVLA reports 100 human hours for 2,500 rollouts. Classical simulation suffers from sim-to-real gap and prohibitive digital-twin construction cost.

World models offer a middle ground — learned, visually rich environments that could serve as surrogate policy evaluators. Current literature shows world models *can* be used for evaluation, but leaves the fundamental question unanswered: **what properties make a world model reliable for policy assessment?**

## Key Insight / Contribution

1. **Benchmark:** WMBench — 2,989 paired real-world/world-model trajectories across 8 manipulation task families, with paired success/failure outcomes from real-robot execution and 324,000+ annotated world-model rollouts from 100+ CVPR 2026 challenge teams
2. **Systematic study:** 7 video world models, 4 action representation schemes, 15+ evaluation metrics distilled into 10 empirical findings organized around 3 questions (metric quality, data/pretraining, architecture)
3. **Design roadmap:** Data → Model → Evaluation design map instantiated as GigaWorld-1, trained on 12,980 hours of multi-source data, improving evaluator-alignment by **14.9%** over Wan 2.2 5B and **11.6%** over Cosmos-Predict2.5

## Method (Pipeline)

### Stage 1: WMBench Construction (§4)

**Data source:** 2,989 paired trajectories across 8 task families from two sources:
- Teleoperated real-world dataset (varied manipulations, camera views)
- Policy rollout dataset from GigaBrain checkpoints (successes + failures)
- Ratio ~1:1 teleoperated:rollout data
- After filtering: 82,470s training, 7,200s test

**WMES scoring (4-level ordinal):**
| Score | Outcome | Fidelity |
|---|---|---|
| 3 | Correct | High |
| 2 | Correct | Low |
| 1 | Wrong | High |
| 0 | Wrong | Low |

Each rollout scored by 3 annotators + 1 senior spot-checker. 324,000 rollout segments chained into long-horizon episodes (20–30 segments each).

**Evaluation protocol (4 steps):**
```
1. Real-world policy data → collect rollouts with success labels
2. World model training on train split (test strictly held out)
3. Closed-loop rollout: policy → action → world model → prediction → feedback
4. Metric calculation + outcome assessment (WMES via human or VLM)
```

### Stage 2: Metric System (§4.3)

Three metric families, 15+ metrics total. Six core evaluator-relevant metrics in summary tables:

**Frame & representation fidelity:**
- Image Quality (MUSIQ-style no-reference predictor)
- Aesthetic Quality (LAION/CLIP aesthetic-predictor)
- JEPA Similarity (V-JEPA polynomial-kernel MMD)
- Subject Consistency (DINO features, dynamic-degree penalty)
- Photometric Consistency (optical-flow AEE, normalized inverse)

**Geometry, semantics, interaction:**
- Geometry Accuracy (monocular depth, median scale alignment)
- Perspectivity (Qwen3-VL 3D-plausibility Likert → [0,1])
- Instruction Following (Qwen3-VL task/action/state matching)
- Semantic Alignment (Qwen2.5-VL descriptions → CLIP-text similarity)
- Interaction Quality (Qwen3-VL robot-object contact Likert → [0,1])
- Trajectory Accuracy (SAM segmentation → NDTW)

**Motion & long-horizon:**
- Dynamic Degree (top 5% optical-flow pixels)
- Flow Score (global optical-flow magnitude)
- Motion Smoothness (frame-interpolation comparison)
- PSNR, FID, FVD (standard reconstruction/distribution metrics)

### Stage 3: Three-Question Empirical Study (§5)

#### Question I: How Should Evaluator Quality Be Assessed? (§5.1)

**Pearson correlation formula (Eq 5):**
ρ(m, c) = Σ(mᵢ − m̄)(cᵢ − c̄) / √[Σ(mᵢ − m̄)² · Σ(cᵢ − c̄)²]

With 95% CI via non-parametric bootstrap (10,000 iterations). Group-level = mean of metric-level correlations.

**Finding 1: Visual and geometric fidelity dominate WMES prediction.**
| Metric Group | ρ with WMES |
|---|---|
| Visual Fidelity | 0.78 |
| Geometry | 0.71 |
| Semantics | 0.59 |
| Dynamics | 0.44 |
| Interaction | −0.11 |
| Appearance Stability | −0.44 |

Top individual metrics: Subject Consistency (ρ=0.88), Perspectivity (ρ=0.86), Instruction Following (ρ=0.84).

**Finding 2: Degenerate metrics mislead evaluator ranking.**
Negative predictors: Background Consistency (ρ=−0.45), Photometric Consistency (ρ=−0.42), Interaction Quality (ρ=−0.11). Static videos score high on appearance stability while ignoring all actions.

**Finding 3: Evaluator quality requires long-horizon assessment, not single-step generation.**
Generic backbones (Wan, Cosmos, LTX, SVD) degrade over 40s rollout — viewpoint drift, object-identity collapse, texture accumulation.

**Finding 4: Outcome-centric supervision essential for VLM evaluation.**
LoRA-tuned Qwen3-VL-8B on structured supervision (overall WMES + aspect-level rationales). Score-focused loss: score token weight=8.0, format tokens=1.0, rationale tokens≥0.05.

**Finding 5: VLM evaluators achieve near-perfect human agreement.**

| Metric | Value |
|---|---|
| Exact Agreement | 0.8780 |
| Adjacent Agreement | 0.9916 |
| Large Error (2 levels) | 0.0084 |
| MAE | 0.1304 |
| RMSE | 0.3836 |
| Quadratic Weighted Kappa | 0.7349 |
| Spearman | 0.7574 |
| Kendall τ_b | 0.7507 |
| W-F1 | 0.8744 |
| \|Bias\| | 0.0455 |

#### Question II: Pretraining & Training Data (§5.2)

**Finding 6: Transferable physical priors > raw scale.**

| Model | Type | AVG |
|---|---|---|
| Cosmos-Predict2.5 | Robot/AD pretrained | 0.6123 |
| Wan 2.2 5B | General-purpose | 0.5948 |
| LTX 2.3 22B | General-purpose | 0.5775 |
| CogVideoX | General-purpose | 0.5620 |
| SVD 1.5B | General-purpose | 0.5569 |

Larger model ≠ better evaluator (LTX 22B < Wan 2.2 5B < Cosmos 2B). Trajectory Accuracy: SVD = 0.0926 (worst temporal/action dynamics).

**Finding 7: Broad physical videos best overall trade-off.**
Data composition ablation (Wan 2.1 1.3B backbone):

| Recipe | AVG | Δ from GigaData-only |
|---|---|---|
| GigaData only | 0.5654 | baseline |
| GigaData + AgiBot | 0.5940 | +0.0286 |
| GigaData + PhysData | 0.6144 | +0.0490 |

PhysData: +0.3074 Photometric Consistency, +0.070 Image Quality, +0.0197 Subject Consistency. JEPA −0.0463, Trajectory −0.0069 (small structural drops).

AgiBot: +0.1401 Subject Consistency, +0.3031 Photometric Consistency, but −0.2426 JEPA, −0.1084 Trajectory (narrow robot data over-specializes).

**Finding 8: Robot-specific data improves embodiment fidelity with sharper trade-off.**
AgiBot improves Subject Consistency (+0.1401) and Aesthetic Quality (+0.0367) but degrades JEPA (−0.2426) and Trajectory Accuracy (−0.1084).

#### Question III: Model Design Choices (§5.3)

**Finding 9: Action control must be spatially aligned.**
Four control interfaces compared on Wan 2.1 1.3B:

| Method | Control Type | Traj. Acc. | Dynamic | Smooth | Flow | Subject | Photo. |
|---|---|---|---|---|---|---|---|
| Wan 2.1 1.3B I2V | None | 0.1576 | 0.2429 | 0.4997 | 0.0971 | 0.5568 | 0.2185 |
| Wan 2.1 1.3B Control | Cross-attention | 0.1620 | 0.1049 | 0.4525 | 0.0624 | 0.3573 | 0.1853 |
| Wan 2.1 1.3B Control | ControlNet | 0.2566 | 0.3083 | 0.5197 | 0.1412 | 0.7212 | 0.3187 |
| Wan 2.1 1.3B Control | Channel concat | 0.3528 | 0.3566 | 0.5747 | 0.2179 | 0.8600 | 0.3206 |

Channel-concatenated wins all metrics. Cross-attention barely better than no control (0.1620 vs 0.1576).

**Finding 10: Reliable evaluators require persistent memory.**
Long-horizon rollout quality over 40 seconds (8-second intervals):

| Model | PSNR 0–8s | PSNR 32–40s | FID 0–8s | FID 32–40s | FVD 0–8s | FVD 32–40s |
|---|---|---|---|---|---|---|
| SVD | 14.05 | 6.88 | 142.84 | 419.21 | 173.63 | 443.76 |
| Cosmos2.5 | 13.65 | 12.83 | 235.74 | 260.84 | 203.03 | 289.74 |
| LTX-Video | 13.38 | 12.74 | 257.83 | 295.33 | 217.17 | 260.78 |
| Wan 2.2 | 14.35 | 10.09 | 216.02 | 266.63 | 169.22 | 300.10 |
| Wan 2.1 | 14.46 | 13.37 | 219.67 | 316.77 | 197.46 | 320.52 |
| Wan 2.1+Mem | **19.82** | **17.41** | **40.58** | **121.61** | **35.30** | **98.34** |

Memory improves Wan 2.1 PSNR from 14.46→19.82 (+5.36), FID from 219.67→40.58 (−179), FVD from 197.46→35.30 (−162).

### Stage 4: GigaWorld-1 Design (§6)

**Design map (Table 5):**

| Component | Design Choice | Notes |
|---|---|---|
| Backbone | Wan-[1.3B / 5B] | Open baseline, mature ecosystem |
| Training data | Physical + open-source robot + egocentric + Giga data | ~12,980 hours |
| Data curation | Quality + motion + distribution filtering | Removes noisy/static/misaligned |
| Structured supervision | Semantic masks + depth + fast-slow captions | Improves geometry, task grounding |
| Action interface | Explicit pixel-aligned representation | EE pose maps + ray maps |
| Long-horizon module | Memory-augmented rollout | First-frame anchor + hierarchical history |
| Temporal encoding | Relative RoPE | Reduces position drift in long AR rollout |
| Training recipe | Progressive multi-stage | Foundation → AR learning → optional scene LoRA → distillation |

**Data composition (Table 6):**

| Category | Sources | Views | Robot Type | Hours |
|---|---|---|---|---|
| Physical Data | Internet/Physics videos | Single | N/A | ~1,298 |
| Open-source Robot | Open X, AgiBot | Single & Multi | Single-arm, Dual-arm, Humanoid | ~5,377 |
| Human-centric | EgoDex, SynData | Ego-centric | Human Hands | ~2,411 |
| Giga-collected | Giga Humanoid, Dual-arm | Single & Multi | Humanoid, Dual-arm | ~3,894 |
| **Total** | | | | **~12,980** |

**Architecture (Figure 8):**

```
Input Image/Video → Patchification → VAE (frozen)
                          ↓
                    Control Video → Patchification → VAE (frozen)
                          ↓
Memory: first-frame anchor + hierarchical history (Long/Mid/Short)
                          ↓
                    DiT backbone + LoRA adapters
                    ├─ Self-attention: Q_hist + Q_noisy jointly attend
                    ├─ Cross-attention: Q_noisy × (K_task, V_task) only
                    └─ Relative RoPE (local positions reinit per AR step)
                          ↓
                    Denoised latents → VAE decode → Gen Video
                          ↓
                    Appended to history buffer → next AR step
```

**Unified Control Injection (§6.2.2):**
- Head camera: **EE pose map** — end-effector trajectory projected to image plane (arm position, orientation, gripper state)
- Wrist cameras: **Ray map** — per-pixel ray origin + normalized direction in world coordinates
- Unified: C_t = Concat_W(C_ee, C_ray) → encoded to latent control Z_ctrl
- Temporally aligned per AR window: Z_ctrl^(k) = Z_ctrl[t_k : t_k + T_f]

**Hierarchical History Injection (§6.2.3):**
- H_t = {H^(S) (short-term motion), H^(M) (mid-term action evolution), H^(L) (long-term scene layout)}
- Plus persistent first-frame anchor x_anchor (never evicted)
- H̃_t = {x_anchor, H^(L), H^(M), H^(S)}

**Relative RoPE (§6.2.5):**
- Local temporal positions: P_hist = {0,...,T_h−1}, P_future = {T_h,...,T_h+T_f−1}
- Reinitialized every AR step → same positional distribution at train/inference
- Prevents repetitive motion and temporal instability

**SLERP Prompt Interpolation (§6.2.6):**
- SLERP(e₁, e₂, t) = [sin((1−t)θ)/sinθ]·e₁ + [sin(tθ)/sinθ]·e₂
- Smooth semantic transitions across task phases in long AR generation
- t_i = i/(N−1), e^(i) = SLERP(e₁, e₂, t_i)

**Progressive Training Pipeline (§6.3):**

```
Stage 1: Robot World Foundation Model
  Flow-matching objective: L_FM = E[‖u_t − u_θ(x_t, t)‖²]
  Backbone: pretrained Wan → continue on curated robot corpus
  32 GPUs, batch=32, LR=5e-5, cosine, 13k steps, LoRA r=128

Stage 2: Autoregressive World Modeling
  Diffusion objective: L_AR = E[‖ε − ε_θ(X_τ, H_t, C, τ)‖²]
  Adds: Relative RoPE, Hierarchical History, First-Frame Anchor, Unified Control
  32 GPUs, batch=32, LR=1e-4, cosine, 36k steps, LoRA r=256

Stage 3: Scene Adaptation LoRA (optional)
  W = W₀ + BA (scene-specific appearance, lighting, objects)

Stage 4: Few-step Distillation
  ODE warm start (optional): L_ODE = E[‖x̂_teacher − x̂_student‖²]
  DMD2 (required): L_DMD2 = λ_dm·L_distill + λ_score·L_score + λ_GAN·L_GAN
  32 GPUs, batch=32, LR_G=2e-6, LR_pfake=4e-7, cosine, 2250 steps, LoRA r=256
```

**System Efficiency (§6.4):**
- SageAttention (drop-in backend)
- TinyVAE decoding (TAESD for preview)
- Ulysses sequence parallelism
- Flash normalization (fused LayerNorm/RMSNorm in Triton)
- Flash RoPE (fused Triton kernel)
- Up to 35.93× inference speedup (SageAttention + 6-step DMD2 + Ulysses)

### Stage 5: Final Evaluation (§6.5)

**Model comparison (Table 9):**

| Model | Size | Type | Aesthetic↑ | Image↑ | JEPA↑ | Semantic↑ | Subject↑ | Trajectory↑ | AVG↑ |
|---|---|---|---|---|---|---|---|---|---|
| SVD | 1.5B | General | 0.6454 | 0.8411 | 0.8267 | 0.5568 | 0.0926 | 0.5569 |
| Wan 2.1 1.3B I2V | 1.3B | General | 0.6002 | 0.8705 | 0.5568 | 0.1576 | 0.5355 |
| LTX 2.3 | 22B | General | 0.5380 | 0.8678 | 0.8248 | 0.1479 | 0.5775 |
| CogVideoX | 5B | General | 0.6437 | 0.8633 | 0.6963 | 0.1609 | 0.5620 |
| Wan 2.2 5B TI2V | 5B | General | 0.5853 | 0.8789 | 0.8883 | 0.1643 | 0.5948 |
| Cosmos-Predict2.5 | 2B | Robot/Auto | 0.6781 | 0.8764 | 0.8747 | 0.1770 | 0.6123 |
| GigaWorld-1-Nano | 1.3B | Robot/Auto | 0.8600 | 0.3528 | 0.8920 | 0.9337 | 0.6717 |
| **GigaWorld-1-Plus** | **5B** | **Robot/Auto** | **0.8883** | **0.3561** | **0.8926** | **0.6834** |

GigaWorld-1-Plus: best JEPA (0.9337), Semantic (0.8926), Trajectory Accuracy (0.3561), matching Subject Consistency (0.8883).
- **+14.9%** over Wan 2.2 5B (0.5948 → 0.6834)
- **+11.6%** over Cosmos-Predict2.5 (0.6123 → 0.6834)

Note: Table 9 has garbled columns from pdftotext — ordering reconstructed from prose. SVD column shows only 4 visible values; Aesthetic=0.6454 inferred from Cosmos being 2B Robot/Auto above it in the hierarchy.

**Closed-loop policy consistency (Table 10):**
4 tasks evaluated with subtask-level outcome checks:
- task1: Put banana into basket (Grasp → Place)
- task2: Put green bowl into pink plate (Grasp → Place)
- task3: Fold paper boxes (Grasp flaps → Position → Press lids → Reset)
- task4: Pour fries into box (Move box → Open → Pour → Press)

GigaWorld-1 shows smaller Gen−Real deviations than challenge baselines. Challenge models overestimate policy success; GigaWorld-1 follows real-world diagonal more closely.

---

## Equations

| Eq | Name | Expression |
|---|---|---|
| 1 | Real trajectory | τ_real = {(o_t, s_t, a_t)}[t=1..T] |
| 2 | WM prediction | ô_{t+1:t+H} ~ M_θ(· | o_{≤t}, s_{≤t}, a_{≤t}, l) |
| 3 | WM trajectory | τ_wm = {(ô_t, s_t, a_t)}[t=1..H] |
| 4 | Evaluator alignment | ρ = Corr(S^real(π), S^wm(π)) |
| 5 | Pearson correlation | ρ(m,c) = Σ(mᵢ−m̄)(cᵢ−c̄) / √[Σ(mᵢ−m̄)²·Σ(cᵢ−c̄)²] |
| 6 | Group-level score | ρ(G,c) = (1/|G|)·Σ ρ(m,c) |
| 7 | Image quality vector | q(x) = [s(x), e(x), n(x), c(x), b(x)] |
| 8 | Aggregate IQ | Q_img(v) = (1/K)·w^T·q(xᵢ) |
| 9 | Temporal discontinuity | D_t = λ_h(1−sim(h_t,h_{t+1})) + λ_φ(1−cos(φ_t,φ_{t+1})) |
| 10 | Video gate | keep(v) = 1[Q_img≥τ_img, A(v)≥τ_aes, max D_t≤τ_jump, Var(D_t)≥τ_static] |
| 11 | Frame motion magnitude | M_t = (1/|Ω|)·Σ ‖F_t(u)‖₂ |
| 12 | Jerk penalty | J(v) = 1/(T−2)·Σ|M_{t+1}−2M_t+M_{t−1}| |
| 13 | Semantic masks | S_t = {(m_k, c_k)}[k=1..K] |
| 14 | Fast-slow captions | C(v) = {C_short(v), C_long(v)} |
| 15–16 | Historical/future latents | X_hist, X_future |
| 17–20 | AR factorization | p(X_future | X_hist, C) = ∏p(X_k | H_{k−1}, C) |
| 18 | Diffusion corruption | X_τ = α_τ·X_future + σ_τ·ε |
| 21 | Unified control concat | C_t = Concat_W(C_ee, C_ray) |
| 25–26 | Hierarchical history | H_t = {H^(S), H^(M), H^(L)}, H̃_t = {x_anchor, H^(L), H^(M), H^(S)} |
| 28–31 | Guidance attention | X = X_Self + X_Cross (cross-attention only on noisy window) |
| 32–34 | Relative RoPE | Local positions reinitialized per AR step |
| 35–38 | SLERP interpolation | Geodesic on embedding manifold |
| 39 | Flow matching loss | L_FM = E[‖u_t − u_θ(x_t, t)‖²] |
| 40 | AR diffusion loss | L_AR = E[‖ε − ε_θ(X_τ, H_t, C, τ)‖²] |
| 41 | LoRA | W = W₀ + BA |
| 42 | ODE distillation | L_ODE = E[‖x̂_teacher − x̂_student‖²] |
| 43 | DMD2 | L_DMD2 = λ_dm·L_distill + λ_score·L_score + λ_GAN·L_GAN |

---

## Figures

| Fig | Description |
|---|---|
| 1 | Overview: 4 action reps, 7 video world models, 12K+ hours training data, 324K+ rollouts → GigaWorld-1 |
| 2 | World model as policy evaluator: policy → action → WM → predicted obs → VLM/Human evaluation |
| 3 | WMBench 4-step protocol: collect → train → closed-loop rollout → compute metrics |
| 4 | Metric-group correlation heatmap with WMES (6 groups, color-coded bars) |
| 5 | Full 15×15 Pearson correlation matrix across all metrics |
| 6 | VLM-assisted Rollout Evaluator: 3-view video → Qwen3-VL LoRA → WMES score + rationales |
| 7 | Data pipeline: (a) filtering/balancing, (b) type/robot distribution, (c) Giga DataCrafter |
| 8 | GigaWorld-1 architecture: DiT + LoRA + memory patchification + control injection |
| 9 | SLERP prompt transition: geodesic interpolation on embedding manifold |
| 10 | 4-stage training pipeline: foundation → AR → scene LoRA → distillation |
| 11 | Model comparison: bar chart + radar plot (8 models, 6 metrics) |
| 12 | Long-horizon rollout dynamics: PSNR/FID over 40s for GigaWorld-1 |
| 13 | Long-horizon visual comparison: T=0,5,30,60s for all models |
| 14 | Memory + SLERP ablation: without memory → with memory → with SLERP |
| 15 | OOD generalization: color, content, background, action-outcome shifts |
| 16 | Task-level success-rate alignment scatter plots |
| 17 | Success-rate bias analysis (Gen − Real deviations per subtask per model) |

---

## Honest-Scope Issues

1. **No confidence intervals on main comparison scores** — Table 9 AVG scores reported as point estimates without CIs; Table 4 PSNR/FID/FVD also point-only
2. **8 task families only** — covers tabletop manipulation, not mobile manipulation, dexterous in-hand, or safety-critical autonomy
3. **Video-centric world models only** — structured state-space or hybrid 3D approaches not compared
4. **VLM evaluator still benefits from human verification** — VLM reduces but doesn't eliminate need for human spot-checks on uncertain cases
5. **Single-robot hardware** — all real-world rollouts from one robot platform; cross-platform generalization untested
6. **Cosmos-3 not evaluated** — paper notes multiview Cosmos-3 not yet publicly released; strongest baseline may be missing
7. **32 NVIDIA H20 GPU training cost** — substantial compute barrier; 32 GPUs × 49k+ total training steps
8. **Table 9 partially garbled by pdftotext** — column alignment disrupted; 4 SVD values visible but full row uncertain
9. **Challenge baselines are community submissions** — uneven quality; some may have limited tuning
10. **No ablation on memory hierarchy design** — hierarchical (S/M/L) vs flat memory not compared

---

## ASCII Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    GigaWorld-1 Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Video ──→ Patchify ──→ VAE (frozen) ──→ Noisy Latent│
│                                                  X_τ        │
│  Control Video → Patchify → VAE (frozen) → E(.) ──→ Z_ctrl  │
│           (EE pose maps + Ray maps, Concat_W)               │
│                                                              │
│  History Buffer:                                             │
│    x_anchor ─────────────────────────────────┐              │
│    H^(L) (scene layout, obj identity)       │              │
│    H^(M) (action evolution, interaction)     ├──→ H̃_t      │
│    H^(S) (short-term motion continuity)      │              │
│                                              │              │
│  T5-xxl Prompt ───────────────────────────────│              │
│                                              │              │
│  ┌───────────────────────────────────────────┘              │
│  │                                                          │
│  │  ┌──────────┐     Self-Attn: Q=[Q_noisy, Q_hist]       │
│  │  │ DiT      │         K=[K_noisy, K_hist]               │
│  │  │ Backbone │         V=[V_noisy, V_hist]               │
│  │  │ + LoRA   ├──────────────────────────┐                │
│  │  │ adapters │     Cross-Attn (noisy only):              │
│  │  │ (r=256)  │         Q_noisy × (K_task, V_task)         │
│  │  └──────────┴──────────────────────────┘                │
│  │         │                          │                      │
│  │    Relative RoPE              Denoised X_future          │
│  │  (local per AR step)             │                       │
│  │                                  ↓                       │
│  │                          VAE Decode → Generated Video   │
│  │                                  │                       │
│  └──────────────────────────────────┘                       │
│                                     │                       │
│                              Append to History Buffer       │
│                                     │                       │
│                              Next AR Window ──────────────→│
│                                                              │
│  Training: 4-stage progressive                               │
│    S1: Flow-matching on robot corpus (13k steps)            │
│    S2: AR diffusion + memory + control (36k steps)         │
│    S3: Scene LoRA (optional)                                │
│    S4: DMD2 distillation (2250 steps, 6-step inference)    │
└─────────────────────────────────────────────────────────────┘
```
