# Writeup — DALorRA: Bayesian Sparse Low-Rank Adaptation

> Zhang, Ren, Zhang, Guo.
> "Bayesian Sparse Low-Rank Adaptation for Large Language Model Uncertainty
>  Estimation." arXiv:2607.02182 (2026).

## The idea in one paragraph

Bayesian neural networks give you calibrated uncertainty but are expensive
at billion-parameter scale. Bayesian LoRA methods (BLoB, TFB, C-LoRA) make
it tractable by putting the posterior over the adapter matrices A/B instead
of the full model — but A/B still have thousands of parameters to maintain
a distribution over. DALorRA goes further: **keep A/B deterministic, put
the posterior over a diagonal mask z ∈ {0,1}^r** (r = LoRA rank, typically
8-64). The update becomes ΔW = B·diag(z)·A, and the "Bayesian" part is just
r Bernoulli logits per layer. At inference, sampling different z masks
gives different sub-networks → prediction diversity → epistemic uncertainty.
The entire UQ overhead is +r scalars per layer (+520 params on Llama-3.1-8B).

## What I implemented

| Finding | Paper claim | My result |
|---|---|---|
| **F1** | DALorRA matches accuracy + improves ECE | +9.6% accuracy under label noise (mask regularizes) |
| **F2** | Higher epistemic uncertainty on OOD | 2.85× OOD/ID ratio |
| **F3** | Negligible Bayesian overhead | 0.65% of trainable params |
| **F4** | Posterior reveals rank importance | π diversifies from 0.50 to 0.54-0.62 |

## What implementing it clarified

### 1. The Concrete relaxation is the implementation crux

The Bernoulli posterior q(z) = ∏ Bern(z_i | π_i) is discrete — you can't
backpropagate through {0,1} sampling. The standard fix is the Concrete
(Gumbel-Softmax) relaxation: sample z via the reparameterized sigmoid
`z = σ((log u − log(1−u) + logit) / τ)` where u ~ Uniform(0,1). With the
straight-through estimator (hard {0,1} in forward, soft gradient in
backward), gradients flow through the logits π.

This was the trickiest part to get right. Without the straight-through
trick, the soft relaxation never produces actual {0,1} masks at inference,
so the MC uncertainty estimate is dampened. With it, the forward pass uses
real binary masks (different ranks active per sample) while backward sees
the smooth sigmoid gradient. The temperature τ controls how close the
relaxation is to true Bernoulli — I used τ=0.5 throughout; the paper
anneals it.

### 2. The KL term is critical for learning the posterior

The ELBO is `L = E_q[CE] + β·KL[q||p]`. Without the KL (β=0), the logits π
immediately saturate to +∞ (π→1, all ranks always active) because that
minimizes CE — the mask degenerates to standard LoRA with no UQ. With KL,
the posterior is pulled toward the uniform prior Bern(0.5), forcing the
model to *justify* each rank it activates.

I found β=0.001 works: small enough that CE dominates (accuracy is
preserved), large enough that π doesn't saturate to 1. The paper doesn't
detail the β schedule, but the principle is clear — you're trading
predictive power (CE) against posterior informativeness (KL). Too much KL
→ π≈0.5 everywhere (no rank differentiation); too little → π≈1 everywhere
(no uncertainty signal).

### 3. The mask is a regularizer, not just an uncertainty estimator

Under label noise (15% of training labels flipped), DALorRA's train CE
plateaus at 0.47 while deterministic LoRA drops to 0.01 — deterministic
memorizes the noise, DALorRA doesn't. The result: DALorRA gets 99.8% test
accuracy vs deterministic's 90.2%. The stochastic mask acts like dropout at
the rank level: each forward pass sees a different subset of rank
components, so the model can't memorize label-specific patterns in any
single rank. This is the "bridging BNN-UQ with ensembling" effect the paper
describes — each mask sample is a different sub-network, and averaging
their predictions regularizes.

### 4. OOD detection works because OOD inputs disagree across masks

The epistemic uncertainty (mutual information between predictions and z)
measures how much the prediction *changes* across mask samples. On
in-distribution inputs, all mask configurations agree (they've all seen
this cluster structure). On OOD inputs (unseen clusters), different masks
produce different predictions → high disagreement → high epistemic
uncertainty. My F2 shows 2.85× higher epistemic uncertainty on OOD.

The mechanism is clean: the posterior π learned which ranks are needed for
ID clusters. On OOD inputs, those ranks produce unreliable outputs
(because they've never seen this data), and the rank-stochasticity exposes
that unreliability as prediction variance. This is exactly the Bayesian
epistemic uncertainty story, compressed to r dimensions.

## What was harder than expected

- **Concrete sampler stability.** The `log(u) − log(1−u)` term can blow up
  when u is near 0 or 1. Clamping u to [1e-8, 1-1e-8] fixed it.
- **KL weight tuning.** β=0.01 was too strong (π stuck at 0.52, barely
  moving from the prior); β=0.0001 was too weak (π→1, no UQ signal). The
  sweet spot β=0.001 gave π diversity in the 0.54-0.62 range. The paper
  doesn't report β, so this required manual search.
- **Det-LoRA baseline overfits.** On clean data both models get 100%
  accuracy and the calibration comparison is meaningless. Adding 15% label
  noise revealed the real difference: DALorRA is noise-robust, deterministic
  LoRA isn't.

## Pointers to the code

| File | What |
|------|------|
| `implementation/model.py` | `DALorRALinear` (frozen base + LoRA + Concrete mask), `DALorRAMLP`, MC uncertainty |
| `implementation/data.py` | Gaussian mixture (ID+OOD) + noisy labels |
| `implementation/train.py` | ELBO training + deterministic baseline |
| `implementation/metrics.py` | ECE, NLL, epistemic/aleatoric decomposition |
| `implementation/run.py` | Reproduces F1–F4 |

## Verdict

DALorRA is the most parameter-efficient UQ method I've implemented: +r
scalars per layer buys you calibrated uncertainty, OOD detection, and a
regularization effect. The key insight — shift the posterior from the dense
adapter to a rank-level mask — is elegant and generalizable. The Concrete
relaxation makes it trainable; the KL term keeps it informative.

🏆 Verdict: a clean, practical PEFT-UQ paper. The r-scalar posterior is the
whole trick; everything else is making it train stably.
