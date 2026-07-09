# Writeup — Fourier Neural Operators for Rayleigh–Bénard Convection

## The one-paragraph version

The paper builds a lean Fourier Neural Operator (314k params, 7 ms) to surrogate
2D Rayleigh–Bénard convection, and its single most important idea is boring to
the point of being easy to miss: **train the FNO to predict the time
*increment* `dU = dt⁻¹(U(t+dt)−U(t))`, not the next solution `U(t+dt)`**. At the
small timestep the solver actually uses (`dt=1e-3`), consecutive states are
nearly identical, so a solution-objective model can score a low relative-L2 loss
just by predicting `U(t)` (the *identity* predictor) — the dynamics is an
`O(dt)`-sized perturbation drowned by the static `‖U‖` baseline. Dividing by
`dt` turns the dynamics into the entire signal. The paper shows the
solution-objective FNO lands **10× worse than doing nothing**, while the
increment-objective FNO beats identity. The second finding is that FNOs are
mesh-*invariant* (they transfer to unseen grids without retraining, because the
spectral weights live on continuous Fourier modes) but **not mesh-*improving***:
a 2× finer inference grid is 4.5× *worse*, because accuracy is capped by the
training-data resolution.

## The problem

Rayleigh–Bénard convection (fluid heated below, cooled above → turbulent
convective rolls; turbulence set by the Rayleigh number `Ra`) is expensive to
simulate. A neural surrogate that could act as a fast one-step integrator (or a
Parareal coarse propagator) would be valuable. Fourier Neural Operators are a
natural fit: they learn a mapping between function spaces in spectral form and
are, in principle, discretization-invariant. The question is how to actually
make one accurate at the tiny timesteps a fluid solver needs.

## The idea

Predict **increments, not solutions.** The FNO output `O(U(t))` is interpreted as
`dU`; the next state is reconstructed as `U(t+dt) = U(t) + dt·O(U(t))` — exactly
the form of an explicit one-step time integrator. Train with relative-L2 on the
increment; evaluate with relative-L2 on the *reconstructed solution*. The
identity baseline (`U(t+dt) ≈ U(t)`, i.e. predict zero increment) guards against
the trivial low-loss regime.

The architecture is a standard FNO: a lifting `P`, a stack of Fourier layers
(each = FFT → truncate to a fixed low-frequency mode set → complex weight → iFFT
`+` a local linear term `Wv` `+` nonlinearity), and a projection `Q`. The
"Improved" model upgrades `P,Q` from a single linear to 4×1D conv and swaps
StepLR for cosine, roughly halving the error again.

## How it works (the intuition)

The whole story rests on a conditioning argument. Write the true next state as
`U(t+dt) = U(t) + dt·∂_tU + O(dt²)`. Under the **solution** objective the
relative-L2 target is `U(t+dt) ≈ U(t)`; its norm is `‖U(t)‖`, and the part that
actually depends on the dynamics — `dt·∂_tU` — is an `O(dt)` sliver of that. The
gradient signal pushing the model to learn `∂_tU` is scaled by `dt`, so at small
`dt` the optimizer is pulled toward the identity map and the dynamics is barely
learned. Under the **increment** objective the target is `∂_tU` itself, an
`O(1)` quantity independent of `dt`: the dynamics is the whole target, well
conditioned, and a fixed budget of optimizer steps actually learns it.

Mesh invariance is the other pillar. The spectral conv keeps the lowest-`m`
Fourier *mode indices* (continuous objects) and applies a complex weight to each.
A mode index means the same thing at every grid resolution, so the *same learned
weights* apply at `N=32` and `N=64` — the operator is continuous, not pinned to
pixels. What it cannot do is invent modes the training data never contained:
modes above the training grid's Nyquist are aliased away at training, so a finer
inference grid that would resolve them gets no benefit.

## What I learned by implementing it

- **The increment-vs-solution hinge reproduces sharply and is a *conditioning*
  fact, not an architecture fact.** With identical FNO, identical Adam budget,
  identical seed, on the *same* small-`dt` data: increment **1.2e-2 < identity
  2.2e-2 < solution 3.0e-1**. The solution model isn't just slightly worse —
  it's **13.6× worse than doing nothing**, because the well-meaning relative-L2
  loss actively rewards the identity map and then the small dynamic residual is
  learned poorly (and wrong is worse than nothing). And identity error is almost
  exactly linear in `dt` (slope 0.99 on a log-log fit), which is the proof that
  the small-`dt` regime is exactly the trap the increment objective escapes.
- **Zeroing a parameter does not disable it.** This cost me a whole debug cycle.
  To ablate the spectral conv I set its weight to zero — but Adam *un-zeroes* it
  one step later, because the gradient of the loss w.r.t. a weight that sits at
  zero is generally *nonzero*. My "spectral-disabled" and "Fourier-disabled"
  baselines were both secretly full FNOs, so they came out identical (1.0×) and
  the ablation looked null. Only `requires_grad_(False)` *plus* zero actually
  freezes a path. I now assert "frozen path stays at 0 after training" inside
  the check. With the bug fixed, the contrast is exactly the theory: on pure
  advection `d_tU=−v·∇U` (a strictly spatial operator) the Fourier layer beats
  identity 1.69× while the per-pixel local map sits dead-on identity (1.00×) — a
  per-pixel map provably cannot compute a spatial derivative.
- **The standard 4-corner spectral conv cannot represent an exact diagonal
  multiplier** because it shares one `(m,m)` weight block across all four
  low-frequency corners; the negative-frequency corner's wavenumber `(−m+i,j)`
  maps to block position `(i,j)`, which is a *different* multiplier value, so a
  preset/analytic multiplier is inconsistent across corners. The FNO is an
  *approximator* of the operator, not its exact embodiment — which is why my
  exact-multiplier "machine-precision" check (my first C2 idea) failed and I
  switched to a trained mechanism contrast.
- **Mesh invariance is real and almost free to show:** train at `N=32`, evaluate
  at `N∈{16,24,32,48,64}` with zero retraining and the error is flat to 3%
  (max/min 1.03). The training-resolution bound is the honest flip side:
  band-limited content (inside training Nyquist) transfers to 2× finer with no
  loss (ratio 0.98); a high-frequency tail *beyond* training Nyquist is not
  recovered (ratio 1.22, finer not better) — that's the paper's Table-4 "(512,128)
  is 4.5× worse than (256,64)" with the mechanism spelled out (aliasing).

## What surprised me / was harder than expected

- Hardest was trusting an ablation that *looked* clean (two numbers, a gate
  passing) but was silently comparing a model to itself. The lesson — sanity-
  assert the *frozen* quantity post-training, not just the pre-training setup —
  is general.
- I expected the solution objective to merely *match* identity; it being
  decisively *worse* (13.6×) is the punchline and it reproduces robustly. The
  honest caveat is that a *perfectly* trained solution FNO on an exactly linear
  operator *could* fit it — the 13.6× is a finite-budget/ill-conditioning
  result, which is exactly the regime the paper is in (it reports single runs,
  no seeds). I reproduce the *ordering and sign*, not the exact factor.

## References
- Paper: https://arxiv.org/abs/2607.02088
- My implementation: `implementation/`
- Breakdown: `breakdown.md`
