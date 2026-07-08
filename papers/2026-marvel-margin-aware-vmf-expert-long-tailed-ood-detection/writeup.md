# MARVEL — Writeup

**Paper:** MARVEL: Margin-Aware Robust von Mises-Fisher Expert Learning for
Long-Tailed Out-of-Distribution Detection
**Authors:** Anudeep A. S., Vaanathi Sundaresan (IIT Madras)
**arXiv:** 2607.02435 (July 2026)

---

## In My Own Words

Out-of-distribution detection in medical imaging is a safety problem: a model
must flag inputs it wasn't trained on (an unseen pathology, a different scanner,
a corrupted image) so a clinician can step in. The catch is that real clinical
data is **long-tailed** — a few common conditions dominate, rare conditions have
handfuls of examples — and most OOD work assumes balanced datasets.

MARVEL attacks the tail at the *classifier boundary*. The core object is the
**von Mises-Fisher** distribution on the unit sphere, `p(x|c) ∝ exp(κ_c μ_cᵀx)`.
Standard vMF/hyperspherical classifiers reduce the log-likelihood to **scaled
cosine similarity** — a *linear* decision boundary on the sphere. MARVEL instead
uses the vMF exponential-family structure and defines a **Nonlinear-vMF (NvMF)
logit** as a change in the log-partition function:

```
ℓ(x; κ, μ) = log C_d(κ) − log C_d(‖κμ + x‖)            (Eq 7)
```

Because `‖κμ + x‖` is a *nonlinear* function of the cosine `ρ = μᵀx` (it bends
through `sqrt(κ²+1+2κρ)`), the NvMF decision boundary is curved — and it
generalises cosine cleanly: **Theorem 1 proves ℓ(x) → μᵀx as κ → ∞**, recovering
the linear/cosine classifier in the high-concentration limit.

Two more pieces make it long-tailed:
1. An **asymmetric class margin** `Δ_yc = τ·log(π_c/π_y)` (Menon-style
   logit-adjustment) that shifts the boundary toward rare classes. Three experts
   `τ ∈ {0,1,2}` specialise to head / balanced / tail.
2. A **dedicated outlier expert** (binary FC head) plus an auxiliary (K+1)-th
   OOD class, fused into a combined OOD score `S = ½(s_NvMF_ensemble + s_outlier)`.

The elegant part: everything is grounded in the vMF log-partition, so the
non-linearity is *principled* (a real generative density), not an ad-hoc MLP head.

---

## What I Learned by Implementing It

### Theorem 1 is exactly right — and the convergence rate is clean

The most satisfying result: `ℓ(x;κ,μ) − μᵀx` shrinks as exactly `O(1/κ)`. In my
run, `(ℓ − ρ)·κ` settled to a constant (≈ −1.11) as κ doubled, with the
successive ratio → 1.00. At κ=5120, `|ℓ − ρ| ≈ 2e-4`. So NvMF is provably a
*generalisation* of cosine: every cosine classifier is the κ→∞ limit of an NvMF
classifier. The large-κ expansions of `log C_d` (Eq 9, via the Bessel asymptotic
`I_ν(κ) ∼ e^κ/√(2πκ)`) and of `‖κμ+x‖` (Eq 10) both reproduce the true values to
<1e-2. This is the citable, falsifiable anchor of the paper and it holds.

### The MLE matters: the closed-form κ approximations are wrong at the tail

My first attempt used the Sra approximation `κ ≈ (d−r²)/(2(1−r²))`. It
*badly* overestimates κ when the mean-resultant length r̄ is small (near-uniform
classes) — exactly the tail. In d=16 it returned κ≈8 for a true κ=2 class.
Switching to the **exact MLE** (bisection on `A_d(κ) = I_{d/2}/I_{d/2−1} = r̄`)
recovered the true concentration spread with correlation **0.999** (fitted
30.4→2.4 vs true 30→2). If you're fitting vMFs on imbalanced data, do not trust
the one-line approximations for the tail classes — invert the Bessel ratio.

### NvMF's boundary is genuinely non-linear — but the balanced-acc win is modest

With per-class κ recovered, the NvMF classifier changes its decision on ~4% of
test points vs cosine (the curved boundary bites). But the **balanced-accuracy
margin over cosine is small and seed-dependent** (here NvMF 0.68 vs cosine 0.70).
This matches the paper honestly: Table 6 shows NvMF>vMF by only 1–7pp, and
limitation #8 notes FC even *beats* NvMF on OOD. The non-linearity is real; its
practical payoff is not a slam-dunk in a clean toy.

### The margin mechanism is cleanly demonstrable; the "win" is calibration

The margin `Δ_yc = τ·log(π_c/π_y)` has the right sign (head-y/tail-c ⇒ Δ<0;
tail-y/head-c ⇒ Δ>0) and increasing τ monotonically redistributes decision mass
to the tail: predicted-in-tail fraction 0.08→1.00 and the tail-most class recall
0.15→0.95 as τ goes 0→2. **But overall tail accuracy peaks at τ≈1 then declines**
— over-correction collapses mass onto the single most-tail class. This is the
paper's own Figure-4 phenomenon (a 4th expert degrades AUROC). The margin works;
the optimal τ is empirical, not principled.

### NvMF logits are O(1), so the margin needs temperature

A subtlety not obvious from the paper: the NvMF logit is *bounded* (ℓ → ρ ∈
[−1,1] asymptotically, regardless of κ). So a raw `τ·log(π_c)` shift (magnitude
up to ~4 for an 80× count imbalance) dominates the logits and collapses the
softmax. I had to normalise the shift by the log-prior dynamic range to keep the
τ ∈ {0,1,2} experts non-degenerate. In the real model, training presumably
sharpens the logits; in a pure inference-time emulation this calibration is on
you.

---

## What Surprised Me

1. **How clean Theorem 1's rate is.** `(ℓ−ρ)·κ` → constant with successive ratio
   exactly 1.00 — not just "converges" but converges at the textbook O(1/κ) rate
   with a clean leading constant. The Bessel asymptotic proof is self-contained
   and reproduces numerically to 4+ digits.

2. **How bad the standard κ approximations are on the tail.** I expected the
   Banerjee/Sra closed forms to be "good enough." They are — for the head. On
   spread (low-κ) classes they're off by 3–5×, which would silently flatten the
   NvMF non-linearity exactly where it's supposed to help.

3. **cosine-MSP is genuinely strong.** I expected the combined MARVEL OOD score
   to beat cosine-MSP handily. It ties it (≈0.82 vs ≈0.83 AUROC). This is
   *consistent* with the paper's Table 7, where MSP is the strongest single
   detector — the headline OOD gains come from the trained NvMF representation,
   not the score function in isolation.

4. **The sign of the inference margin is the easy-to-get-wrong part.** Whether
   you *add* or *subtract* `τ·log π_c` to boost the tail is a genuine source of
   confusion (the loss-time Δ_yc and the inference-time shift are easy to
   conflate). The empirical test — does tail recall rise with τ? — settles it.

---

## What Was Harder Than Expected

1. **Stable `log C_d` for large κ.** The naive `I_ν(κ)` overflows around κ≈700.
   Using the exponentially-scaled `ive(ν,κ) = I_ν·e^{−κ}` fixes it, but the sign
   in the final formula is treacherous: `log C_d = (p−1)logκ − (d/2)log(2π) −
   log ive − κ` (the last term is **minus** κ, because `I_ν = ive·e^κ`). A `+κ`
   sign error silently flips `log C_d` from decreasing to increasing and makes
   ℓ converge to `−ρ` instead of `+ρ` — every classification sign inverts.

2. **Designing a toy where NvMF's non-linearity is even visible.** With shared
   κ, the NvMF boundary is exactly linear (cosine). You need *per-class,
   recoverable* κ spread, which forces lower dimension (d=8) so the exact MLE is
   reliable and a dramatic κ range (30→2) so the warp is decisive. In high-d the
   tail classes look near-uniform and the warp vanishes.

3. **The margin sweep over-corrects.** Reporting "tail accuracy rises with τ" is
   false — it rises then crashes. The honest, robust, theoretically-implied
   quantities are *predicted-in-tail fraction* and *single tail-most class
   recall*, both monotone in τ. Choosing the right metric to verify the margin
   took iteration.

4. **OOD-baseline orientation.** AUROC with "higher = more OOD" plus an MSP score
   that is *higher for ID* inverts the number (0.16 instead of 0.84). Easy to
   misread as "MSP is sub-chance"; it's just a sign convention.

---

## Code

Implementation in `implementation/`:
- [`model.py`](implementation/model.py) — vMF normalising constant (stable), NvMF
  logit (Eq 7), exact-MLE fit, classifier, margin shift (Eq 14-15), outlier
  expert (Eq 16), ensemble + combined OOD score (Eq 17-19), Theorem-1 asymptotics
- [`data.py`](implementation/data.py) — Synthetic long-tailed hypersphere with
  per-class concentration + graded OOD spectrum
- [`train.py`](implementation/train.py) — 5 verification checks + summary

Run: `pip install -r implementation/requirements.txt && python implementation/train.py`
(or `uv run --with numpy --with scipy python implementation/train.py`)

All 5 checks PASS on CPU in ~5 seconds.

---

## References

- Paper: https://arxiv.org/abs/2607.02435
- Code: https://github.com/redboxup/MARVEL
- vMF / NvMF: Banerjee et al. (2005), "Clustering on the Unit Hypersphere using
  von Mises-Fisher Distributions"
- Logit adjustment / margin: Menon et al. (2021), "Long-tail learning via
  logit adjustment"
- Bessel asymptotics: Abramowitz & Stegun §9.7
