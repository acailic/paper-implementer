# DanceOPD: On-Policy Generative Field Distillation

**Authors:** Wei Zhou, Xiongwei Zhu, Zelin Xu, Bo Dong, Lixue Gong, Yongyuan Liang, Meng Chu, Leigang Qu, Lingdong Kong, Wei Liu, Tat-Seng Chua

**Affiliations:** ByteDance Seed, NUS, UMD, HKUST

**Year:** 2026 | **ArXiv:** [2606.27377](https://arxiv.org/abs/2606.27377) | **Project:** [danceopd.github.io](https://danceopd.github.io)

---

## One-Liner

Compose multiple generative capabilities (T2I, editing, realism) into one flow-matching model by treating each as a velocity field and doing hard-routed on-policy distillation with a single semantic-side query per sample.

## Core Problem

A deployed image generation model needs to do many things: text-to-image (T2I), local editing, global editing, style transfer. These capabilities conflict — editing degrades T2I, local and global editing pull in opposite directions. Naive data mixing, parameter merging, and adapter composition all produce compromise solutions. The question: how do you make one model that genuinely strengthens a target capability while preserving an anchor one?

## Key Insight

Treat each frozen capability source (T2I model, edit model, realism model, even CFG operator) as a velocity field over a shared generative state space. Capability composition becomes a field-query problem with three coupled choices:

| Choice | Question | DanceOPD Answer |
|--------|----------|-----------------|
| **Field selection** | Which field supervises each sample? | Hard-routed: one field per sample (no soft averaging) |
| **Query state** | Where to evaluate the field? | On-policy: student's own rollout states (stop-gradient) |
| **Trajectory supervision** | How many states per rollout? | Single semantic-side low-noise query per sample |

## Method at a Glance

1. **Hard-routed sample-wise field matching** — Each training sample is dispatched to exactly one frozen capability field (T2I or edit or style, etc.). Route probabilities are uniform over active buckets. This avoids averaging unrelated directions into one target.
2. **On-policy field querying** — The routed field is queried on a stop-gradient state drawn from the current student's Euler ODE rollout, not on fixed offline data states. Aligns supervision with the states the student actually visits at inference.
3. **Semantic-side single query** — One query per sample, sampled from a Beta(5,2) distribution biased toward the low-noise (clean image) end of the trajectory. Low-noise states concentrate capability-specific information (style, aesthetics, edit details). Avoids correlated dense trajectory queries.
4. **Plain velocity MSE** — `L = ||v_θ(z̄_t, t, c) - v_m(z̄_t, t, c)||²`. KL-style matching reduces to weighted MSE under a local Gaussian view. Unweighted MSE is empirically most stable.

## Three Identified Challenges

- **Target-field ambiguity** — Soft multi-teacher averaging creates a supervision direction that doesn't correspond to any real capability.
- **State-distribution mismatch** — Off-policy query states leave the student under-supervised on its own visitation distribution.
- **Trajectory-query correlation** — Dense same-rollout queries share noise seed, prompt, dynamics, and path history — not independent supervision.

## Main Results (Z-Image backbone)

| Setting | Key Metric | Gain vs. Best Baseline |
|---------|-----------|----------------------|
| T2I + Edit | GEditBench avg | +8.1% vs. best OPD, +8.5% vs. edit source |
| T2I + Edit | GenEval overall | +2.0% vs. T2I source, +1.6% vs. best composition |
| Local + Global Edit | GEditBench avg | +16.1% vs. best composition, +7.9% vs. local edit source |
| Realism absorption | Realism reward | +9.9% vs. off-policy, closes 85.3% of teacher gap |
| CFG absorption | GEditBench avg | +7.6% vs. train-only absorption |

## Diagnostic Highlights

| Ablation | Key Finding |
|----------|------------|
| Hard routing vs. soft mixing | +15.2% (MSE), +10.6% (KL) — routing matters more than objective choice |
| Low-t vs. median/high-t query | +23.7% vs. median, +19.5% vs. high-t — low-noise is where capability signal lives |
| K=1 vs. K=2,4,8,16 dense | Single query wins by 7.9–16.6% — correlated queries hurt |
| SDE decorrelation for K=2, G=3 | +18.4% recovery but still -8.6% vs. K=1 default |
| Plain MSE vs. alternatives | Beats timestep-weighted (+2.8%), consistency (+4.1%), KL (+4.5%) |
| Local-edit init vs. merged | +37.2% — initialization matters a lot |

## Training Cost

DanceOPD is cheaper per step than DiffusionOPD (dense K=N=16 supervision) and Flow-OPD (PPO + SDE + micro-batch factor 2). Only K=1 gradient state after a 16-step rollout.

## Limitations

- Requires compatible velocity fields over shared state space (same backbone family, latent representation, scheduler)
- Predefined routing buckets — doesn't handle ambiguous task boundaries or prompts needing multiple capabilities simultaneously
- Realism evaluation uses a proprietary reward model

## Code/Data

No open-sourced code at time of paper. Project page has qualitative results.
