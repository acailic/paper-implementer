"""Problem fixtures for the DSGNAR checks.

Two numpy problems:
  * ``overdetermined_jacobian`` -- a random tall J in R^{N x d} (N >> d) with a
    known right-hand-side, used to verify the doubly-sketched Gauss-Newton model
    is a valid subspace embedding (C1), the SVD-reuse identities (C2) and the
    lambda/delta trust-region duality (C3).
  * ``rosenbrock`` -- the d-dimensional generalised Rosenbrock least-squares,
    the textbook ill-conditioned nonlinear least-squares whose curved valley has
    a huge condition number; analytic residual + Jacobian.  Used for the
    conditioning-first optimiser comparison (C4).

(C5's PINN lives in pinn.py and hands its Jacobian to the same primitives.)
"""

import numpy as np


def overdetermined_jacobian(N=512, d=48, rank=8, lam=3.0, seed=1):
    """Low-rank-effective tall Jacobian  J = U diag(sigma) V^T + small noise
    (N x d), the regime where sketching pays off.

    PINN residual Jacobians are *numerically* low rank: a few dominant singular
    directions carry the Gauss-Newton step, the rest are tiny.  Random full-rank
    J is sketching's worst case (every direction equally important), so we build
    a rank-``rank``-dominant spectrum (``rank`` singular values of order 1,
    decaying thereafter) -- exactly why the paper can sketch d_theta ~ 1e5 down
    to s ~ 4e3.  Returns J (N x d), r (N), lam.
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((N, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    sigma = np.zeros(d)
    sigma[:rank] = np.linspace(1.0, 0.6, rank)            # dominant block
    sigma[rank:] = 10.0 ** (-np.linspace(2.0, 6.0, d - rank))   # negligible tail
    J = (U * sigma) @ V.T
    r = U[:, 0] * 2.0 + 0.1 * rng.standard_normal(N)      # residual in the top modes
    return J, r, lam


def random_fullrank_jacobian(N=512, d=48, lam=3.0, seed=1):
    """A dense random J (sketching worst case) -- used only as a contrast in C1."""
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((N, d)) / np.sqrt(d)
    r = rng.standard_normal(N)
    return J, r, lam


def rosenbrock_res(theta):
    """d-dim generalised Rosenbrock residual (pairs of consecutive coords).

    For pair (a, b): r1 = 1 - a, r2 = 10 (b - a^2).  Minimum r = 0 at a = b = 1.
    theta length must be even.
    """
    theta = np.asarray(theta, dtype=float)
    a = theta[0::2]
    b = theta[1::2]
    r1 = 1.0 - a
    r2 = 10.0 * (b - a * a)
    return np.stack([r1, r2], axis=1).reshape(-1)


def rosenbrock_jac(theta):
    """Analytic Jacobian of rosenbrock_res, shape (n_pairs*2, d)."""
    theta = np.asarray(theta, dtype=float)
    d = theta.shape[0]
    n = d // 2
    J = np.zeros((2 * n, d))
    for i in range(n):
        a = theta[2 * i]
        # r1 = 1 - a  -> d/da = -1
        J[2 * i, 2 * i] = -1.0
        # r2 = 10(b - a^2) -> d/da = -20 a, d/db = 10
        J[2 * i + 1, 2 * i] = -20.0 * a
        J[2 * i + 1, 2 * i + 1] = 10.0
    return J


def rosenbrock_problem(d=20, seed=0):
    """Return (residual_fn, jac_fn, theta0) for a d-dim Rosenbrock."""
    assert d % 2 == 0
    rng = np.random.default_rng(seed)
    # start far from the optimum (1,1,...) in the ill-conditioned valley
    theta0 = -1.2 + 0.1 * rng.standard_normal(d)
    theta0[1::2] = 1.0 + 0.1 * rng.standard_normal(d // 2)
    return rosenbrock_res, rosenbrock_jac, theta0


# default config handed to train.py
CONFIG = dict(
    sketch=dict(N=512, d=48, rank=8, lam=3.0, seed=1,
                sketch_sizes=(8, 16, 24, 32, 40), K=4, n_seed=5),
    rosenbrock=dict(d=20, seed=0, iters=60,
                    lam_grid=np.logspace(-3, 3, 13)),
)
