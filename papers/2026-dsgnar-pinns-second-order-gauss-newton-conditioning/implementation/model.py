"""DSGNAR — Doubly-Sketched Gauss-Newton with Adaptive Ratio.

Numerical-optimisation core of "An Optimisation Framework for the Well-Conditioned
Training of Physics-Informed Neural Networks" (Webb, Jerad, Cartis 2026,
arXiv:2607.02194).  This module isolates the paper's *optimiser* contribution
(§3) from its PDE zoo (§5): the two coupled ideas are

  1. a doubly-sketched Gauss-Newton / Levenberg-Marquardt model
     (CountSketch on the residual rows + a Subsampled Randomised Cosine Transform
     on the parameter columns -> a small square s x s Jacobian whose SVD is cheap
     and *reused* across regularisation values), and
  2. a conditioning-first step rule that selects each step by a target *decrease
     ratio* rho* (Eqs 11-14, Algorithm 1/3) rather than by fixing lambda or the
     trust-region radius directly.

Everything operates on plain numpy arrays so the same primitives serve the
synthetic least-squares checks (C1-C4) and the PINN check (C5, whose Jacobian is
handed in from torch.func.jacfwd).
"""

import numpy as np
from scipy.fft import dct
from scipy.interpolate import PchipInterpolator

# --------------------------------------------------------------------------- #
# 1. Sketch operators (§3.2)
# --------------------------------------------------------------------------- #


class CountSketch:
    """OSNAP / CountSketch row sketch  C : R^N -> R^s.

    K independent hashes, +/-1 signs (Eq 16).  ``rows`` applies C to a length-N
    vector; ``mat`` applies C to every column of an N x k matrix (i.e. returns
    C @ M, shape s x k).
    """

    def __init__(self, N, s, K=2, rng=None):
        rng = np.random.default_rng(rng)
        self.N, self.s, self.K = N, s, K
        self.hashes = rng.integers(0, s, size=(K, N))       # h_k(j) in [0, s)
        self.signs = rng.integers(0, 2, size=(K, N)) * 2 - 1  # +/-1

    def rows(self, v):
        out = np.zeros(self.s)
        for k in range(self.K):
            np.add.at(out, self.hashes[k], self.signs[k] * v)
        return out / np.sqrt(self.K)

    def mat(self, M):
        """C @ M  for M of shape (N, k) -> (s, k)."""
        assert M.shape[0] == self.N
        out = np.zeros((self.s, M.shape[1]))
        for k in range(self.K):
            # contribution of hash k: scatter rows of M into bucket rows
            np.add.at(out, self.hashes[k], self.signs[k][:, None] * M)
        return out / np.sqrt(self.K)


def srct(d, s, rng=None):
    """Subsampled Randomised Cosine Transform (DCT-II) *lift* L in R^{d x s}.

    The parameter dimension is compressed then lifted back (§3.2.2), so the lift
    L = Omega S must be a faithful near-isometry: L^T L = I_s exactly (its s
    columns are s distinct rows of the orthogonal DCT-II matrix, hence
    orthonormal).  The forward compress is L^T (s x d); the lift is L (d x s).
    """
    rng = np.random.default_rng(rng)
    D = dct(np.eye(d), type=2, norm="ortho")            # (d, d) orthogonal DCT-II
    idx = rng.choice(d, size=min(s, d), replace=False)
    return D[idx, :].T.astype(float)                    # (d, s), orthonormal columns


# --------------------------------------------------------------------------- #
# 2. Gauss-Newton / Levenberg-Marquardt primitives (§2.2, Eqs 6-14)
# --------------------------------------------------------------------------- #


def full_lm_step(J, r, lam):
    """Exact LM step  p = -(J^T J + lam I)^{-1} J^T r  (Eq 11)."""
    d = J.shape[1]
    A = J.T @ J + lam * np.eye(d)
    return -np.linalg.solve(A, J.T @ r)


def svd_step_factory(Jtil, rtil):
    """One SVD of the square sketch Jtil (s x s) -> a reusable step function.

    Returns ``step(lam)`` giving
        p~(lam) = -V diag(sigma_i / (sigma_i^2 + lam)) U^T rtil
    together with the factorisation so callers can read off ||p~(lam)|| and the
    model reduction from the SVD *without re-solving* (the "single SVD yields
    inexpensive candidate steps" claim, §3.2.3).
    """
    U, sig, Vt = np.linalg.svd(Jtil, full_matrices=False)
    Ut_r = U.T @ rtil

    def step(lam):
        coef = sig / (sig * sig + lam)
        return -(Vt.T * coef) @ Ut_r          # -V diag(coef) U^T rtil

    bundle = dict(U=U, sig=sig, Vt=Vt, Ut_r=Ut_r)
    return step, bundle


def step_norm_svd(bundle, lam):
    """||p~(lam)|| from the SVD alone."""
    coef = bundle["sig"] / (bundle["sig"] ** 2 + lam)
    w = coef * bundle["Ut_r"]
    return np.sqrt((w * w).sum())


def model_reduction_svd(bundle, lam):
    """m(0) - m(p) = predicted reduction of the regularised quadratic model,
    from the SVD alone (no re-solve).  Closed form (see derivation below):

        1/2 ||rtil + Jtil p~||^2 + lam/2 ||p~||^2  evaluated at p~(lam) gives
        residual quadratic  = 1/2 sum (lam/(sig^2+lam))^2 (U^T rtil)_i^2
        regulariser         = 1/2 lam sum (sig/(sig^2+lam))^2 (U^T rtil)_i^2

    so m(0)-m(p) = 1/2||rtil||^2 - (residual quadratic + regulariser).  Here we
    return the *model value* m(p) (residual quadratic + regulariser); the
    predicted reduction is ``0.5*||rtil||^2 - model_reduction_svd(...)``.
    """
    sig, Ut_r = bundle["sig"], bundle["Ut_r"]
    a = lam / (sig * sig + lam)                # shrinkage of each component
    pcoef = sig / (sig * sig + lam)
    resid_quad = 0.5 * ((a * a) * (Ut_r * Ut_r)).sum()
    reg = 0.5 * lam * ((pcoef * pcoef) * (Ut_r * Ut_r)).sum()
    return resid_quad + reg


def predicted_reduction(J, r, p, lam):
    """Direct predicted reduction m(0)-m(p) = -p^T J^T r - 0.5 p^T (J^T J + lam I) p.
    Used to cross-check the SVD identity (check C2) and to compute rho."""
    return -(p @ (J.T @ r)) - 0.5 * (p @ ((J.T @ J + lam * np.eye(J.shape[1])) @ p))


def decrease_ratio(theta, p, residual_fn, J, r, lam):
    """rho (Eq 14): actual reduction / predicted reduction."""
    r_new = residual_fn(theta + p)
    actual = 0.5 * (r @ r) - 0.5 * (r_new @ r_new)
    pred = predicted_reduction(J, r, p, lam)
    return actual / pred, actual, pred


# --------------------------------------------------------------------------- #
# 3. Trust-region lambda/delta duality + LambdaSolve (§3.3, Algorithm 3)
# --------------------------------------------------------------------------- #


def solve_lambda_for_radius(bundle, delta, lam_lo=1e-8, lam_hi=1e12):
    """Find lambda such that ||p~(lambda)|| = delta (trust-region duality, Eq 13).

    ||p~(lambda)|| is continuous & strictly decreasing in lambda (verifiable),
    so bisection is exact.  Returns the lambda achieving radius ``delta``.
    """
    def f(lam):
        return step_norm_svd(bundle, lam) - delta
    lo, hi = lam_lo, lam_hi
    if f(lo) <= 0:          # radius already larger than the GN step even at lam~0
        return lo
    for _ in range(200):
        mid = np.sqrt(lo * hi)            # geometric bisection (lambda spans orders)
        if f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi)


def _isotonic_nonincreasing(x, y):
    """Pool-adjacent-violators: project y to be non-increasing in x (ascending).
    Mirrors Algorithm-3 line 13 ('enforce rho_hat(delta) non-increasing')."""
    order = np.argsort(x)
    xo, yo = x[order], y[order]
    w = np.ones_like(yo)
    vals = yo.astype(float).copy()
    # PAV for non-increasing: negate, do non-decreasing PAV, negate back.
    nv, nw = -vals, w.copy()
    stack_v, stack_w, stack_x = [], [], []
    for v, ww, xx in zip(nv, nw, xo):
        cv, cw, cx = v, ww, xx
        while stack_v and cv < stack_v[-1]:
            tot_w = cw + stack_w[-1]
            cv = (cv * cw + stack_v[-1] * stack_w[-1]) / tot_w
            cw = tot_w
            cx = stack_x[-1]              # keep the smaller delta as the bucket key
            stack_v.pop(); stack_w.pop(); stack_x.pop()
        stack_v.append(cv); stack_w.append(cw); stack_x.append(cx)
    return np.array(stack_x), -np.array(stack_v)


def lambda_solve(bundle, theta, residual_fn, J, r, rho_star,
                 lam_base, Delta_k, n_probe=9):
    """Algorithm 3 (LambdaSolve): probe a lambda ladder, measure the actual
    decrease ratio rho at each, build a monotone rho(delta) model and invert it
    to find the step whose ratio matches the *target* rho_star.  Returns
    (lambda*, Delta*), with Delta* clipped to [Delta_k/3, 3 Delta_k].

    This is the conditioning-first rule: we never pick lambda or the radius
    directly -- we pick the *ratio*.
    """
    ladder = lam_base * np.logspace(-2.0, 2.5, n_probe)
    deltas, rhos, lams = [], [], []
    step_fn, _ = svd_step_factory(J, r)          # FULL-system SVD for the optimiser
    full_bundle = _full_bundle(J, r)
    for lam in ladder:
        p = full_lm_step(J, r, lam)
        if not np.all(np.isfinite(p)):
            continue
        rho, _, _ = decrease_ratio(theta, p, residual_fn, J, r, lam)
        deltas.append(np.linalg.norm(p))
        rhos.append(rho)
        lams.append(lam)
    deltas = np.array(deltas); rhos = np.array(rhos); lams = np.array(lams)
    if len(deltas) < 3:
        return lam_base, Delta_k

    # monotone non-increasing rho(delta) (Alg 3 line 13)
    xd, yd = _isotonic_nonincreasing(deltas, rhos)
    order = np.argsort(xd)
    xd, yd = xd[order], yd[order]
    # target radius = the delta whose ratio equals rho_star (PCHIP inverse)
    try:
        chip = PchipInterpolator(yd, xd)
        delta_star = float(chip(rho_star))
    except Exception:
        delta_star = Delta_k
    if not np.isfinite(delta_star):
        delta_star = Delta_k
    delta_star = float(np.clip(delta_star, Delta_k / 3.0, 3.0 * Delta_k))
    lam_star = solve_lambda_for_radius(full_bundle, delta_star)
    return lam_star, delta_star


def _full_bundle(J, r):
    U, sig, Vt = np.linalg.svd(J, full_matrices=False)
    return dict(U=U, sig=sig, Vt=Vt, Ut_r=U.T @ r)


# --------------------------------------------------------------------------- #
# 4. Optimisers: conditioning-first DSGNAR vs fixed-lambda LM baseline
# --------------------------------------------------------------------------- #


def dsgnar(residual_fn, jac_fn, theta0, iters=60,
           rho_stage1=0.10, rho_stage2=0.60, Delta0=1.0, lam0=1.0, seed=0):
    """DSGNAR (Algorithm 1), sketch-free core: full Gauss-Newton Jacobian with
    the conditioning-first target-ratio step rule (Algorithm 3 LambdaSolve) and
    a two-stage rho schedule -- conservative Stage 1 (rho*<=0.2, drives the
    effective regularisation down toward a well-conditioned region), aggressive
    Stage 2 (rho*>=0.5) once minimal regularisation is reached.

    Returns (theta, loss_trace, effective_lambda_trace).
    """
    theta = np.asarray(theta0, dtype=float).copy()
    r = residual_fn(theta)
    L = 0.5 * float(r @ r)
    Delta, lam = Delta0, lam0
    loss_trace = [L]
    lam_trace = [lam]
    stage_switch = iters // 2
    for k in range(iters):
        J = jac_fn(theta)
        rho_star = rho_stage1 if k < stage_switch else rho_stage2
        lam_star, delta_star = lambda_solve(
            None, theta, residual_fn, J, r, rho_star, lam_base=lam, Delta_k=Delta
        )
        p = full_lm_step(J, r, lam_star)
        L_new = 0.5 * float(residual_fn(theta + p) @ residual_fn(theta + p))
        if L_new < L:                       # accept (Alg 1 line 8-9)
            theta = theta + p
            L = L_new
            r = residual_fn(theta)
            Delta = max(delta_star, 1e-6)
            lam = max(lam_star, 1e-12)      # remember effective regularisation
        else:                               # reject, shrink radius (line 10-11)
            Delta = Delta / 3.0
            lam = lam_star * 3.0 if np.isfinite(lam_star) else lam * 3.0
        loss_trace.append(L)
        lam_trace.append(lam)
        if L < 1e-16:
            break
    return theta, np.array(loss_trace), np.array(lam_trace)


def fixed_lambda_lm(residual_fn, jac_fn, theta0, lam, iters=60):
    """Baseline: Levenberg-Marquardt with a *fixed* regularisation lambda (no
    trust region, no ratio rule).  The foil for the conditioning-first claim."""
    theta = np.asarray(theta0, dtype=float).copy()
    r = residual_fn(theta)
    L = 0.5 * float(r @ r)
    loss_trace = [L]
    for _ in range(iters):
        J = jac_fn(theta)
        p = full_lm_step(J, r, lam)
        cand = theta + p
        L_new = 0.5 * float(residual_fn(cand) @ residual_fn(cand))
        if L_new < L:                       # accept only true decreases
            theta = cand
            L = L_new
            r = residual_fn(theta)
        loss_trace.append(L)
        if L < 1e-16:
            break
    return theta, np.array(loss_trace)


# --------------------------------------------------------------------------- #
# 5. Sketched Gauss-Newton step (the doubly-sketched model, §3.2.3)
# --------------------------------------------------------------------------- #


def sketched_lm_step(J, r, Crow, Q, lam):
    """Doubly-sketched LM step.  Jtil = (C J) Q  (s x s), rtil = C r (s).
    p~ = -(Jtil^T Jtil + lam I)^{-1} Jtil^T rtil via one SVD; lifted p = Q p~.
    Returns (p, Jtil, rtil, bundle)."""
    Jtil = Crow.mat(J) @ Q
    rtil = Crow.rows(r)
    step_fn, bundle = svd_step_factory(Jtil, rtil)
    ptil = step_fn(lam)
    p = Q @ ptil
    return p, Jtil, rtil, bundle
