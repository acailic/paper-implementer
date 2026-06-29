# Notes — Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's an **action representation + training strategy paper** for human-to-robot skill transfer. The authors propose a specific "bridging action" — relative wrist translation in the head-camera frame — as the shared action space between humans and robots, and build a π0-like VLA model to train on it.

| # | What | Output |
|---|------|--------|
| 1 | Identify why 6DoF human wrist actions fail for transfer | Noisy rotations + contact pattern mismatch |
| 2 | Propose **translation-only bridging action** `a3D-wrist` | Shared across embodiments, robust to noise |
| 3 | Build **interleaved action tokens** with masking | Handle missing action components gracefully |
| 4 | 3-stage training pipeline (pre-train → co-train → post-train) | 15 real-robot tasks, 600h human data |
| 5 | Characterize **upper bound** of bridging representation | How much room is left? |

The re-implementable artifact is the bridging action representation + interleaved action token design, applied to a π0-style VLA.

## The core argument

Most prior work treats human hands as just another 6-DOF embodiment — extract wrist pose (position + rotation) from hand pose estimators, and train the robot policy on that. Two problems:

1. **Rotation estimates from hand pose are noisy.** Pose estimators (MediaPipe, etc.) give unreliable wrist rotation, especially roll/pitch.
2. **Contact patterns differ fundamentally.** Human fingers vs parallel grippers — the extra DOFs in fingers mean wrist rotation is less tightly coupled to manipulation semantics. Rotating your wrist while holding something with fingers ≠ rotating a parallel gripper.

So they throw away rotation entirely and keep only **relative wrist translation** in the head-camera frame. Both humans and robots perceive the world from a head camera, so this is a shared, meaningful, rotation-invariant signal.

## The bridging action math

Given wrist pose `W^w_t ∈ SE(3)` and head-camera pose `T^{c←w}_t ∈ SE(3)`:

```
W^c_{t+i} = (T^{c←w}_t)^{-1} · W^w_{t+i}

a3D-wrist_t = ΔW^3D = t(W^c_{t+i}) − t(W^c_t),   i = 1,...,k
```

where `t(·)` extracts the 3×1 translation from SE(3). For bi-manual: `a3D-wrist ∈ R^{k×6}`.

The robot end-effector action is the standard 6DoF relative pose:
```
a6D-eef_t = ΔW^6D = (W^w_{t+i})^{-1} W^w_{t+i+...}  → R^{k×12}
```

## The interleaved action sequence

This is clever. Different data sources have different available actions:

| Data source | a3D-wrist | a6D-eef | agripper |
|-------------|:---------:|:-------:|:--------:|
| In-the-wild human (EgoDex + outsourced) | ✓ | ✗ | ✗ |
| In-lab human (PICO 4 Ultra) | ✓ | ✗ | ✓ |
| Robot tele-op | ✓ | ✓ | ✓ |

So they interleave action tokens as `[a3D-wrist → a6D-eef → agripper]` and mask the missing ones in attention. The ordering is intentional: the shared bridging signal (a3D-wrist) is attended to by 6DoF tokens, enabling knowledge transfer within the attention pattern itself.

## The 3-stage training

| Stage | Data | Action supervision | Iterations |
|-------|------|-------------------|:----------:|
| **I: Pre-train** | ~600h human actions (EgoDex ~70h, outsourced ~500h, in-lab ~45h) | Only a3D-wrist (bridging) | 400k |
| **II: Co-train** | ~72h robot pick-and-place + ~3h/task human in-lab (15 tasks) | All three + random a3D-wrist↔a6D-eef substitution on robot data | 120k |
| **III: Post-train** | 10 robot trajectories/task | All three | 25k |

**Critical detail in Stage II:** They randomly *add or substitute* a3D-wrist for a6D-eef as the prediction target on robot data. This explicitly binds the bridging representation to executable robot actions. Removing this drops success from 38% → 12.5% (Table 4). This is the load-bearing design choice.

## Key results

**Finding 1 — Bridging transfers beyond pick-and-place.**
Robot-only pick-and-place: ~0% success on all 15 tasks.
Co-train with human bridging actions: up to 49% success, 60% progress.
The robot genuinely learns manipulation it never saw in robot data.

**Finding 2 — Pre-training on humans scales.**
Stage I+II vs Stage II only: consistent improvements across all task groups. Even though pre-training only supervises the non-executable bridging signal, the downstream robot actions benefit.

**Finding 3 — Bridging >> 6DoF human actions.**
Table 2: bridging action gets 49.06% overall progress vs 38.02% for 6DoF baseline.
Qualitatively (Fig 7-8): 6DoF produces distorted, twisted wrist poses; bridging gives stable, natural motions.

**Finding 4 — Post-training efficiency.**
With Stage I pre-training, 10 robot trajectories/task in post-training achieves 71.21% progress and 55% success. Without pre-training (Stage III only): 53.79% progress, 35.83% success. Pre-training on non-executable human actions makes few-shot robot fine-tuning significantly more efficient.

**Finding 5 — Loss alignment.**
Pre-training on bridging actions yields lower training loss for both a6D-eef and agripper during co-training (Fig 9). The objective landscapes are aligned — optimizing wrist translations implicitly helps learn end-effector actions.

**Finding 6 — Upper bound.**
Treating robot demos as "perfect human data" (no visual gap, no noise) and using the same training objective: 73.54% progress, 55.83% success vs default 59.75%/38.33%. Big headroom — the bridging representation itself works, the gap is in the embodiment mismatch.

## 15 evaluation tasks

- Microwave: open/close door, take/place bowl, wipe L→R, wipe R→L
- Drawer: open, close
- Mug: hang left, hang right
- Cup: stack left, stack right, insert straw
- Other: toast→plate, unplug charger

Two scenes per task, 4 rollouts each = 8 trials/task. Both success rate AND fine-grained progress score reported.

## Failure cases

Tasks needing precise end-effector rotation at contact (insert straw, open drawer) are where it breaks. Robot shows clear intent but can't execute the critical rotational step. Consistent with the design choice of discarding rotation from human data.

## Terms / concepts I had to look up

| Term | Meaning |
|------|---------|
| **π0** | Physical Intelligence's VLA model — vision-language-action with flow matching |
| **EgoDex** | Large-scale egocentric manipulation video dataset |
| **Flow matching** | Generative modeling approach — learn velocity field to transform noise to data |
| **ByteMini** | ByteDance's bi-manual mobile manipulation platform (7-DoF arms, parallel grippers) |
| **Mixture-of-Transformers** | Separate transformer params for VL tokens vs action tokens, shared attention |
| **PICO 4 Ultra Enterprise** | VR headset used for egocentric human data collection |
