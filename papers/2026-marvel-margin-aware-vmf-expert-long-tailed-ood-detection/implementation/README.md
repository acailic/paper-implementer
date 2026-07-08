# MARVEL: Margin-Aware vMF Expert for Long-Tailed OOD Detection

## What this implements

A **toy** demonstration of the MARVEL paper (Anudeep & Sundaresan 2026,
arXiv:2607.02435). The full method trains a Nonlinear-vMF hyperspherical
classifier head on a ResNet-18 over three medical-imaging datasets. This
implementation compresses the *mathematical core* into a minimal, runnable setup:

- **8-dim unit-sphere embeddings** with 6 classes in a long-tailed distribution
  (counts 1200 -> 15), per-class vMF concentration kappa 30 -> 2 (head tight,
  tail spread)
- **NvMF logit (Eq 7)** as a change in the vMF log-partition function — the
  principled non-linear generalisation of cosine
- **Exact MLE vMF fit** per class (mean direction + concentration)
- **Margin-aware multi-expert ensemble** (Eq 14-15), tau in {0,1,2}
- **Auxiliary (K+1)-th OOD class** + **outlier FC expert** (Eq 16) + combined
  OOD score (Eq 19)

### Key MARVEL ideas demonstrated

| Concept | How it appears here |
|---------|-------------------|
| NvMF logit (Eq 7) | ell = log C_d(kappa) - log C_d(‖kappa·mu + x‖) |
| Theorem 1 (ell -> cosine as kappa->inf) | Convergence verified: ell = rho + O(1/kappa) |
| Per-class concentration (non-linear boundary) | Fitted kappa 30.4->2.4 matches true 30->2 (corr 0.999) |
| Exact vMF MLE | Bisection on A_d(kappa)=r̄, not the lossy closed-form approx |
| Margin asymmetry (Eq 14) | head-y/tail-c => Delta<0; tail-y/head-c => Delta>0 |
| 3-expert ensemble (Eq 18) | tau=0 head-biased, tau=1 balanced, tau=2 tail-biased |
| Outlier expert (Eq 16) | Binary FC head on balanced ID+aux-OOD batches |
| Combined OOD score (Eq 19) | S = 0.5 (s_NvMF_ensemble + s_outlier) |

## Files

- `model.py` — vMF normalising constant, NvMF logit, exact-MLE fit, classifier,
  margin shift, outlier expert, ensemble OOD score, Theorem-1 asymptotics, AUROC
- `data.py` — Synthetic long-tailed hypersphere (vMF sampling, graded OOD spectrum)
- `train.py` — 5 verification checks + summary
- `requirements.txt` — numpy, scipy

## How to run

```bash
pip install -r requirements.txt
python train.py
# or, in a uv-managed env:  uv run --with numpy --with scipy python train.py
```

The script runs 5 checks (~5 s on CPU):
1. Theorem 1 — NvMF logit converges to cosine as kappa -> inf (O(1/kappa))
2. Eq 9 + Eq 10 — log C_d and ‖kappa·mu + x‖ large-kappa expansions
3. NvMF classifier — kappa recovery + non-linear boundary vs cosine
4. Margin asymmetry — decision mass shifts toward tail as tau grows
5. OOD machinery — Eq 17-19 detectors beat chance + fusion non-degrading

## Expected output

```
Summary
  thm1    : PASS     # ell = mu^T x + O(1/kappa);  |ell-rho|@5120 ~ 2e-4
  asymp   : PASS     # log C_d & ||k mu+x|| expansions match (rel err <1e-2)
  nvmf    : PASS     # kappa recovery corr 0.999; boundary differs from cosine on ~4%
  margin  : PASS     # tail_frac 0.08->1.00, tail-most recall 0.15->0.95 as tau 0->2
  ood     : PASS     # all MARVEL detectors AUROC>0.65; Eq-19 fusion >= min(components)
```

## Paper claims verified

- ✓ Theorem 1: NvMF logit -> cosine similarity as kappa -> infinity (error ~ 1/kappa)
- ✓ Eq 9 / Eq 10 Bessel-function asymptotics reproduce the true log-partition
- ✓ Exact vMF MLE recovers the head->tail concentration spread (corr 0.999)
- ✓ Margin Delta_yc = tau·log(pi_c/pi_y) is asymmetric in the documented direction
- ✓ Increasing tau redistributes decision mass toward tail classes (Eq 14-15)
- ✓ The Eq 17-19 OOD machinery yields valid detectors and non-degrading fusion

## Honest scope (what is NOT claimed here)

The paper's headline Pareto gains come from **end-to-end training** of the
margin-aware NvMF loss on a ResNet-18 over real medical images, not from the
score function in isolation. Two findings from this toy mirror the paper's own
limitations (breakdown §7):

- **NvMF does not universally beat cosine on balanced accuracy.** With per-class
  kappa the NvMF boundary is genuinely non-linear (decisions differ on ~4% of
  points), but the balanced-acc margin is modest and seed-dependent — matching
  Table 6 (NvMF>vMF by 1-7pp) and limitation #8 (FC even beats NvMF on OOD).
- **The combined OOD score ties rather than beats cosine-MSP.** Per the paper's
  own Table 7, MSP is the strongest *single* detector, so beating it is not
  asserted; we verify the Eq-19 fusion is valid and never degrades below its
  components.
- **Tail accuracy peaks then declines with tau** (over-correction), reproducing
  the paper's Figure-4 "optimal expert count" phenomenon (a 4th expert hurts).

## Hardware

Works on **CPU**. 8-dim synthetic data, ~5 seconds total runtime. No GPU needed.
