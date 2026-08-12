# SwanTale — toy re-implementation

> **Paper:** SwanTale: Unified Multi-Speaker Speech and Audio Generation for
> Instruct and Zero-Shot Tasks
> **Authors:** Yu Zhang, Ruiqi Li, Changhao Pan, Ke Lei, Xiang Yin, Cheng Yang
> **ArXiv:** https://arxiv.org/abs/2608.02023
> **Official code:** none published (project page: https://swanaigc.github.io/#swantale)

## What this implements

SwanTale is an industrial-scale system (2B active params, 64×A100, ~70M caption
records, a Qwen caption encoder) that is not faithfully re-implementable end to
end outside an industrial lab. Following the guidance in `breakdown.md` §8, this
toy re-implements the **four self-contained, algorithmically rich pieces** of
the method on small synthetic latents — the parts where the actual novelty
lives — rather than reproducing paper-scale audio numbers.

| Component | File | Paper ref |
|---|---|---|
| **Engram layer** — hashed n-gram memory + content-dependent gated residual with negative-init bias | `model.py::EngramLayer` | Eq. 4-5 |
| **Unified MoE** — dual router (task-level shared experts + frame-level dynamic Top-P audio experts with null skip), time-aware budget `q(t)`, annealed Gumbel, aux-loss-free load balancing, z-loss + null-collapse penalty | `model.py::UnifiedMoE` | Eq. 10-32 |
| **SwanDiT** — AdaLN-Zero flow-matching DiT with caption cross-attention, quality-flag conditioning, and every-2nd-block MoE | `model.py::SwanDiT`, `DiTBlock` | §3.2-3.3 |
| **Task-masked flow matching** — one velocity objective; instruct vs zero-shot differ only in caption content and which frames are masked | `model.py::build_noised/flow_loss` | Eq. 6-9 |
| **Sway sampling** — non-uniform ODE integration grid `t(u)=1-cos(πu/2)` at inference | `train.py::sample` | Eq. 43 |

The single most important idea — **task unification via masking** — is fully
exercised: both `inst` (generate all frames, full caption) and `zero` (keep
reference frames clean, content-only caption) share one backbone and one
velocity loss.

## How to run

```bash
pip install -r requirements.txt
python train.py            # ~30 epochs, prints flow-loss curve
python train.py --epochs 50 --eval   # also samples + reports per-task MSE
```

`python model.py` runs a quick self-test of every component on random tensors.

## What "success" looks like here

The goal is understanding, not SOTA. Success = the machinery runs end to end
without crashing and the flow-matching velocity loss decreases on both tasks
under one shared backbone. Per-task generation-frame MSE against targets is
reported under `--eval`; on this toy synthetic dataset it should drop well
below the initial noise level, demonstrating that the velocity field is learned
and that the task-masking unification does not break either task.

### Actual run results (`python train.py --epochs 30 --eval`, CUDA)

```
SwanTale toy training | device=cuda | params=1,530,138
dataset=512 samples | epochs=30 | steps/epoch=16
epoch   0 | flow_loss=6.2743 moe=0.0187
epoch   5 | flow_loss=5.5631 moe=0.0130
epoch  10 | flow_loss=4.8461 moe=0.0082
epoch  15 | flow_loss=4.4980 moe=0.0045
epoch  20 | flow_loss=4.5362 moe=0.0035
epoch  25 | flow_loss=4.3174 moe=0.0035
epoch  29 | flow_loss=4.3161 moe=0.0031
[eval] task=inst  generation-frame MSE vs target = 1.0710
[eval] task=zero  generation-frame MSE vs target = 1.0957
```

The flow velocity loss falls ~31% (6.27 → 4.32) and the MoE auxiliary loss
collapses ~6× (0.019 → 0.003), confirming the dual router stabilizes and the
task-masking flow objective trains cleanly for both instruct and zero-shot
tasks in a single shared backbone.

## What is NOT here (and why)

- **SwanVAE** (48kHz audio VAE) and real audio — not tractable at toy scale;
  we use synthetic latent targets that mimic the smooth continuous latents a
  good audio VAE would produce.
- **SwanData-Caption** (~70M records + MLLM annotation + human audit) — the
  data pipeline is the biggest practical contribution but needs the industrial
  pipeline. Captions are synthetic tokens.
- **Full GRPO post-training** (marginal-preserving SDE, K=8 rollouts) — too
  heavy for a toy; the transferable bits (per-element mean log-prob,
  closed-form KL) are discussed in `breakdown.md` §4.6/§8.
- **Qwen/CosyVoice encoders** — replaced with a tiny embedding + Engram layer.
