# DALorRA — Bayesian Sparse Low-Rank Adaptation

From-scratch PyTorch implementation of:

> Jijie Zhang, Zhe Ren, Quan Zhang, Dandan Guo.
> "Bayesian Sparse Low-Rank Adaptation for Large Language Model Uncertainty
>  Estimation." arXiv:2607.02182 (2026).

DALorRA shifts uncertainty quantification from the dense adapter space
(millions of params) to the **rank level** (r scalars): it keeps LoRA
matrices A/B deterministic and inserts a stochastic diagonal mask
D=diag(z) into the adapter update, learning a factorized Bernoulli
posterior q(z) over which rank components are active.

## Quick start

```bash
pip install torch numpy
python3 run.py
```

## Architecture

```
ΔW = B · diag(z) · A · (α/r)     # z ~ Bernoulli posterior q_ϕ(z)
```

- A ∈ R^{r×m}, B ∈ R^{n×r}: deterministic LoRA factors
- z ∈ {0,1}^r: stochastic mask (Concrete relaxation for gradients)
- Only r logits {π_i} are the Bayesian parameters per layer

## Findings reproduced

| Finding | Paper claim | My result |
|---|---|---|
| **F1** | DALorRA matches/beats deterministic LoRA on accuracy + calibration | +9.6% accuracy under 15% label noise (mask diversity regularizes) |
| **F2** | Higher epistemic uncertainty on OOD inputs | 2.85× OOD/ID epistemic ratio |
| **F3** | Negligible Bayesian overhead (+r scalars) | 48 Bayesian params out of 7344 (0.65%) |
| **F4** | Posterior π reveals rank importance | π diversifies 0.50→0.54-0.62 range |

## Files

| File | Purpose |
|------|---------|
| `model.py` | `DALorRALinear`, `DALorRAMLP`, Concrete relaxation sampler, MC uncertainty |
| `data.py` | Gaussian mixture (ID+OOD) + noisy labels generator |
| `train.py` | ELBO training (CE + KL), deterministic baseline |
| `metrics.py` | ECE, NLL, epistemic vs aleatoric uncertainty |
| `run.py` | Main runner: F1–F4 |

## Known gaps

1. **Synthetic data, not Llama-3.1-8B.** The paper fine-tunes a real LLM on
   NLP benchmarks (ECE/NLL tables across 10 datasets). We use a Gaussian
   mixture to demonstrate the calibration and OOD-detection properties.
2. **No accuracy-matched ECE comparison.** On real LLMs the paper shows
   ECE improvement at matched accuracy. On our toy, the accuracy gap from
   label noise dominates the ECE comparison.
3. **Single temperature.** The Concrete relaxation temperature is fixed at
   0.5; the paper anneals it during training.
