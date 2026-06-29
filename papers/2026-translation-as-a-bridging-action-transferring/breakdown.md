# Breakdown — Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots

> **Paper:** Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots
> **Authors:** Sijin Chen, Kaixuan Jiang, Haixin Shi, Yanhui Wang, Weiheng Zhong, Haosheng Li, Bo Jiang, Yuxiao Liu, Xihui Liu
> **Year:** 2026 (arXiv:2606.28133, v1, Jun 2026)
> **ArXiv:** https://arxiv.org/abs/2606.28133
> **Project page:** https://translation-as-a-bridging-action.github.io/
> **Type:** Action representation + training strategy for human-to-robot manipulation transfer.

---

## 1. Problem & Motivation

**Problem.** Human action data is cheap, abundant, and diverse — one of the most promising paths for scaling robot learning. But transferring manipulation skills from humans to robots is hard. The mainstream approach treats humans as just another 6-DOF embodiment: extract wrist pose from hand pose estimators, use it to train robot policies. Two things make this sub-optimal:

1. **Noisy rotation estimation.** Hand pose predictors give unreliable wrist rotation, especially roll and pitch angles.
2. **Contact pattern mismatch.** Human fingers have many more DOFs than parallel grippers. Wrist rotation in a human hand is semantically decoupled from the manipulation behavior in a way it isn't with a gripper. Directly replaying extracted human 6DoF wrist actions on robots often yields distorted, twisted motions.

**Why important.** If we can't reliably learn from human data, we're stuck collecting expensive robot tele-operation demos forever. Human data could be the key data flywheel for embodied AI — but only if the action representation bridges the embodiment gap properly.

**Prior-work limitations:**
1. Existing cross-embodiment methods concatenate action spaces [13, 26, 35], pad missing dimensions [8, 25, 42], or build separate projectors [7, 53] — none of which address the fundamental rotation noise problem.
2. No one has systematically compared translation-only vs full 6DoF human actions for robot transfer.
3. Missing action components across data sources are handled with padding/concatenation rather than structured masking.

## 2. Key Insight / Contribution

**Core idea (one sentence):** Use only the relative wrist translation in the head-camera frame as a shared "bridging action" between humans and robots — it's robust to noisy rotation estimates, physically meaningful, and embodiment-agnostic — then train a π0-like VLA with interleaved action tokens that handles missing components via attention masking.

**What is genuinely new:**
- The **bridging action representation** (`a3D-wrist`): translation-only, camera-frame-relative, shared across embodiments.
- **Interleaved action tokens** ordered `[bridging → 6DoF → gripper]` with attention masking for missing components — enables knowledge transfer within the attention pattern.
- **Random bridging substitution** during co-training: randomly swap a3D-wrist for a6D-eef as prediction target on robot data — the load-binding mechanism (ablation: removing this drops success 38% → 12.5%).
- **Large-scale human-only pre-training** (600h) that only supervises the non-executable bridging signal, yet transfers to full robot actions.
- Upper-bound characterization showing the representation has significant headroom.

## 3. Method

### 3.1 Overview

```
Human video → Hand pose estimator → Wrist poses → Head-camera projection
                                                          ↓
                                                    a3D-wrist (bridging)
                                                          ↓
Robot tele-op → End-effector poses → a6D-eef + agripper + a3D-wrist
                                                          ↓
                                    ┌──────────────────────────────────┐
                                    │  π0-like VLA (Mixture-of-Transformers)  │
                                    │  VLM backbone → VL KV cache         │
                                    │  Action Transformer → flow matching  │
                                    │  Interleaved: [a3D-wrist|a6D-eef|agripper] │
                                    │  Missing components: attention masked  │
                                    └──────────────────────────────────┘
```

### 3.2 Bridging Action — a3D-wrist

The key representation. Given wrist pose `W^w_t ∈ SE(3)` in world frame and head-camera pose `T^{c←w}_t ∈ SE(3)`:

```
W^c_{t+i} = (T^{c←w}_t)^{-1} · W^w_{t+i}

a3D-wrist_t = ΔW^3D = t(W^c_{t+i}) − t(W^c_t),   i = 1,...,k
```

- `t(·)` extracts 3×1 translation from SE(3)
- Bi-manual: `a3D-wrist ∈ R^{k×6}` (3 per arm)
- Properties: (1) physically meaningful under shared observation, (2) robust to noisy rotation, (3) embodiment-agnostic by construction

### 3.3 Robot End-Effector Action — a6D-eef

Standard 6DoF relative wrist pose:

```
a6D-eef_t = ΔW^6D = (W^w_{t+i})^{-1} W^w_{t+i+...}
```

Converted to Cartesian + Euler angles: `a6D-eef ∈ R^{k×12}` (12 per arm).

### 3.4 Gripper Action — agripper

Binary signal per gripper: `agripper ∈ R^{k×2}` (close=1, open=0). For in-lab human data, hand closure is annotated as the gripper signal.

### 3.5 Unified Action Space

`a_t = (a3D-wrist_t, a6D-eef_t, agripper_t)`

Per-source supervision:

| Data source | a3D-wrist | a6D-eef | agripper |
|-------------|:---------:|:-------:|:--------:|
| In-the-wild human (EgoDex + outsourced, ~570h) | ✓ | ✗ | ✗ |
| In-lab human (PICO 4 Ultra, ~45h) | ✓ | ✗ | ✓ |
| Robot tele-op (~72h + task-specific) | ✓ | ✓ | ✓ |

### 3.6 Interleaved Action Tokens

Action tokens organized as `[a3D-wrist → a6D-eef → agripper]` per timestep. Ordering based on two priors:
1. The shared bridging signal should be attended to by 6DoF tokens (enables human→robot transfer within attention)
2. Gripper triggers after end-effector reaches target

Missing components are masked in attention layers, loss is omitted for masked tokens.

### 3.7 Flow Matching Objective

```
L_FM = ||v̂(a^τ_t, o_t, l, τ) − v*||²     where v* = ε − a_t
```

Inference: integrate velocity from τ=0 to 1 with Δτ=0.2 via Euler method. Only generate a6D-eef and agripper for robot control.

### 3.8 VL Co-Training

Standard next-token prediction on vision-language corpora, mixed per batch with flow matching loss.

### 3.9 Three-Stage Training

| Stage | Data | Supervision | Batch size | Iterations |
|-------|------|-------------|:----------:|:----------:|
| **I: Pre-train** | ~600h human (70h EgoDex + 500h outsourced + 45h in-lab) | a3D-wrist only | 1024 | 400k |
| **II: Co-train** | 72h robot pick-place + 3h/task × 15 tasks human in-lab | All + random a3D-wrist↔a6D-eef on robot | 256 | 120k |
| **III: Post-train** | 10 robot traj/task | All three | 256 | 25k |

VLM KV-cache repeated 4× per batch to increase action transformer effective batch size.

## 4. Math

**Bridging action (Eq. 1):**
```
a3D-wrist_t = t(W^c_{t+i}) − t(W^c_t),  i = 1,...,k
where W^c_{t+i} = (T^{c←w}_t)^{-1} · W^w_{t+i}
```

**End-effector action (Eq. 2):**
```
a6D-eef_t = (W^w_{t+i})^{-1} · W^w_{t+i+...}
```

**Flow matching loss (Eq. 3):**
```
L_FM = ||v̂(a^τ_t, o_t, l, τ) − (ε − a_t)||²
```

**NTP loss:**
```
L_NTP = −(1/|s|) Σ log P(s_i | s_{<i}; o_t, l)
```

**Euler inference step:**
```
a^{τ+Δτ}_t = a^τ_t + Δτ · v̂(a^τ_t, o_t, l, τ)   with Δτ = 0.2
```

## 5. Evaluation Setup

### 15 tasks across 4 categories

| Category | Tasks |
|----------|-------|
| **Microwave** | Open door, Close door, Take bowl out, Place bowl in, Wipe L→R, Wipe R→L |
| **Drawer** | Open, Close |
| **Mug/Cup** | Hang left mug, Hang right mug, Stack left cup, Stack right cup, Insert straw |
| **Other** | Toast→plate, Unplug charger |

Per task: 2 scenes × 4 rollouts = 8 trials. Success rate + fine-grained progress score (0–1 scale per task).

### Platform

- **Robot:** ByteMini — bi-manual mobile platform, 2×7-DoF arms, parallel grippers, 3× RGB-D cameras (head + 2 wrists)
- **Human data:** PICO 4 Ultra Enterprise VR headset
- **Model:** ~4B params, Mixture-of-Transformers, Qwen2.5-VL backbone

## 6. Results & Ablations

### Main results (Fig 5-6, Table 2)

| Setting | Overall Progress | Overall Success |
|---------|-----------------:|----------------:|
| Robot pick-place only (no human) | ~18% | ~12% |
| + Human co-training (Stage II) | ~50% | ~31% |
| + Human pre-train (Stage I+II) | ~60% | ~38% |
| + Few-shot post-train (Stage I+II+III) | ~72% | ~55% |

### Bridging vs 6DoF human actions (Table 2)

| Human action type | Overall Prog. | Overall Succ. |
|-------------------|-------------:|--------------:|
| 6DoF wrist actions | 38.02% | 25.00% |
| **Translation-only (bridging)** | **49.06%** | **38.33%** |

Gap is biggest on microwave tasks (64.58% vs 48.13% progress) and drawer tasks (56.88% vs 55.00%). The qualitative difference is stark: 6DoF produces twisted, distorted poses; bridging gives stable, natural manipulation.

### Post-training data efficiency (Table 3)

| Model | Overall Prog. | Overall Succ. |
|-------|-------------:|--------------:|
| Stage III only (no pre-train) | 53.79% | 35.83% |
| **Stage I + III** | **71.21%** | **55.00%** |

Pre-training on non-executable human actions makes few-shot robot fine-tuning substantially more efficient (+17pp progress, +19pp success).

### Bridging objective ablation (Table 4)

| Robot data supervision | Overall Prog. | Overall Succ. |
|------------------------|-------------:|--------------:|
| w/o a3D-wrist on robot data | 39.67% | 12.50% |
| **w/ a3D-wrist on robot data** | **59.75%** | **38.33%** |

🔥 **This is the most important ablation.** Removing the random a3D-wrist↔a6D-eef substitution on robot data during co-training crashes performance. The model needs to be explicitly forced to ground bridging representations into executable actions.

### Upper bound (Table 5)

| Model | Overall Prog. | Overall Succ. |
|-------|-------------:|--------------:|
| Default (real human data) | 59.75% | 38.33% |
| **Upper bound (robot demos as "perfect human")** | **73.54%** | **55.83%** |

Significant headroom (+14pp progress, +18pp success). The bridging representation works well; the gap comes from embodiment mismatch (visual observation gap, action noise).

### Loss alignment (Fig 9)

Pre-training on bridging actions yields **lower** training loss for both a6D-eef and agripper during co-training, despite only supervising a3D-wrist during pre-training. The objective landscapes are aligned.

## 7. Limitations

- **No rotation at all.** Contact-rich tasks needing precise end-effector orientation (insert straw, open drawer) are where it fails. The robot knows *what* to do but can't get the angle right.
- **Thin objects.** Robot struggles to pick up thin objects after co-training — attributed to observation gap + human action noise.
- **Single robot platform.** Only tested on ByteMini (parallel grippers). Unknown how this transfers to dexterous hands.
- **No code released** (as of the paper). The project page exists but code isn't mentioned.
- **600h of human data** is non-trivial to collect — the "cheap and abundant" argument is relative.
- **15 tasks, 8 trials each.** Reasonable but not massive. Statistical robustness could be questioned.

## 8. Open Questions / Ideas

- **Can we add back limited rotation?** The failure cases cry out for it. A conditional rotation signal (only when a reliable pose estimate is available) could fill the gap without re-introducing the noise problem.
- **Scale the bridging to more robot platforms.** The whole argument is embodiment-agnostic by construction — should work on single-arm, mobile, humanoid platforms too.
- **Combine with latent action pre-training.** Current work uses explicit wrist translations. What if you pre-train with latent actions from videos [9, 15, 17, 63] and then fine-tune with bridging?
- **The substitution ablation is the most interesting result.** The idea that randomly swapping prediction targets during training forces the model to ground shared representations is generalizable beyond this paper. Could apply to any multi-embodiment setting.
