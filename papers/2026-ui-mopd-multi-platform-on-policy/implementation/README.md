# UI-MOPD — from-scratch implementation (toy scale)

Paper: **UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI
Agent Learning**, Lian et al., arXiv 2607.04425 (2026).

This folder re-implements the paper's *core learning mechanism* on a
toy scale — the Multi-Teacher On-Policy Distillation (MOPD) objective
and its platform-conditioned routing — **not** the full Qwen3-VL-8B/32B
system (which needs 64 H100 GPUs).

## What is reproduced

| Paper element | Equation | File |
|---------------|----------|------|
| K3 KL estimator `D_hat = rho - delta - 1` | Eq. 4–5 | `model.py:k3_kl_estimator` |
| Adaptive group-level KL mask `mu` | Eq. 6 | `model.py:adaptive_kl_mask` |
| Structured outcome reward `+1/-0.5/-1` | Eq. 8 | `model.py:structured_reward` |
| Token-level grouped advantage | Eq. 9 | `model.py:token_advantage` |
| Clipped-PPO + platform-KL combined loss | Eq. 10–12 | `model.py:mopd_loss` |
| Platform-conditioned teacher routing | Eq. 7 | `train.py` (`teacher_lp` where) |
| Two-stage pipeline (SFT teachers → MOPD student) | §3 | `train.py` |

## Run

```bash
pip install torch   # CPU is fine
python3 train.py
```

Expected output: prints per-epoch loss / reward during MOPD, then final
action-correctness on desktop vs. mobile prompts for three settings —
the two platform teachers, a naive mixed-SFT baseline, and the MOPD
student. The MOPD student should match or beat mixed-SFT on **both**
platforms simultaneously, demonstrating the continual-learning benefit
that is the paper's central claim.

## Scope / honesty

- Tiny transformer decoders (`d_model=64`, 2 layers), a ~30-token action
  vocabulary, and synthetic prompt/action templates — **not** real GUI
  screenshots or Qwen3-VL. The point is to exercise the *equations*, not
  to reach the paper's absolute success rates (38.2% OSWorld / 12.0%
  MobileWorld).
- Reward is a coarse match fraction over verb + coordinate tokens rather
  than the full action-dimension `f_a` over bounding boxes / scroll
  directions / key equality.
- Teachers are frozen SFT'd small models standing in for the 32B
  Qwen3-VL-Thinking teachers.
