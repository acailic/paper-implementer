"""OrbitQuant core: rotation-based calibration-free quantization primitives.

Implements the load-bearing math of OrbitQuant (Lee et al. 2026, arXiv:2607.02461):

  * f_d marginal (Eq 2)  -- the fixed post-rotation coordinate density that lets
    ONE Lloyd-Max codebook serve every weight row and activation token of dim d.
  * Haar rotation (§3.2) and RPBH / block-Hadamard (Eq 9) -- the structured
    orthogonal maps that drive any unit vector's coordinates onto f_d.
  * Proposition 1 (Eq 10) -- universal variance concentration: with a uniform
    random permutation every rotated coordinate has Var(z_i) in [1/d(1-+rho)].
  * Lloyd-Max codebook on f_d (Eq 3, 6, 8) -- MSE-optimal, scale/zero-point free.
  * rotation-cancels-in-the-product weight/activation quantization (Eq 4-8).

Everything is dense numpy/scipy so the equations are exercised directly; the
paper's O(d log h) fast Walsh-Hadamard is noted but not needed for correctness.
"""

import numpy as np
from scipy.special import gammaln, betainc


# --------------------------------------------------------------------------- #
#  Eq 2: the fixed post-rotation coordinate marginal f_d
# --------------------------------------------------------------------------- #
def fd_lognorm(d):
    """log of the normalising constant of f_d.

    f_d is the symmetric Beta(1/2, (d-1)/2) density on [-1, 1], i.e. the exact
    distribution of one coordinate of a uniform point on the unit sphere S^{d-1}.
    Its normaliser is 1 / B(1/2, (d-1)/2) = Gamma(d/2) / (sqrt(pi) Gamma((d-1)/2)).

    NOTE: the paper's rendered Eq 2 writes sqrt(Gamma(d/2)/(pi Gamma((d-1)/2))),
    which does NOT integrate to 1 (off by sqrt(Gamma(d/2)/Gamma((d-1)/2))). The
    correct normaliser (verified: integral 1, var 1/d) puts the full gamma ratio
    outside a sqrt-over-(1/pi) only. The codebook is invariant to a constant
    rescale of f_d (we normalise the lattice weights), so this only matters for
    the explicit density/variance self-check, not the quantizer.
    """
    return gammaln(d / 2.0) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2.0)


def fd_density(t, d):
    """f_d(t) = [Gamma(d/2)/(sqrt(pi) Gamma((d-1)/2))] * (1 - t^2)^((d-3)/2).

    Symmetric Beta(1/2, (d-1)/2) on [-1, 1] -- one coordinate of a uniform
    sphere point; mean 0, variance 1/d, tightly approximated by N(0, 1/d).
    """
    t = np.clip(np.asarray(t, dtype=float), -1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.exp(fd_lognorm(d) + ((d - 3) / 2.0) * np.log1p(-t * t))
    val = np.where(np.abs(t) >= 1.0, 0.0, val)
    return val


def fd_cdf(t, d):
    """Exact CDF of f_d via the Beta regularised function.

    T^2 ~ Beta(1/2, (d-1)/2), T symmetric, so
        F(t) = 0.5 * (1 + sign(t) * I_{t^2}(1/2, (d-1)/2)).
    """
    t = np.asarray(t, dtype=float)
    s = np.sign(t)
    return 0.5 * (1.0 + s * betainc(0.5, (d - 1) / 2.0, np.clip(t * t, 0.0, 1.0)))


def fd_mean_var(d):
    """Analytic mean/variance of f_d: mean 0, variance 1/d (sphere coordinate)."""
    return 0.0, 1.0 / d


# --------------------------------------------------------------------------- #
#  Rotations (Eq 9 + Haar baseline)
# --------------------------------------------------------------------------- #
def _hadamard(h):
    """Sylvester-constructed normalized Walsh-Hadamard matrix, h a power of 2."""
    assert h >= 1 and (h & (h - 1)) == 0, "Hadamard order must be a power of 2"
    H = np.array([[1.0]])
    while H.shape[0] < h:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(h)


def _largest_pow2_divisor(d):
    """h = largest power of two dividing d (paper §4.4)."""
    h = 1
    while d % (2 * h) == 0:
        h *= 2
    return h


def random_haar(d, rng):
    """Uniform random orthogonal matrix (Haar). scipy.stats.ortho_group."""
    from scipy.stats import ortho_group

    return ortho_group.rvs(d, random_state=rng)


def rpbh(d, rng, h=None, use_perm=True):
    """Randomized Permuted Block-Hadamard rotation Pi_d (Eq 9).

        Pi_d = blkdiag(H_h D_1, ..., H_h D_k) . P_pi

    with k = d/h blocks, H_h a normalized Walsh-Hadamard matrix, D_i independent
    Rademacher sign diagonals, and P_pi a uniform random permutation matrix.
    `use_perm=False` gives the Block-RHT ablation (no permutation) used to show
    Prop 1 / Remark 1: without the permutation an outlier concentrated in one
    block never spreads, so the marginal drifts off f_d at low bit-width.
    Returns the dense d x d matrix (the O(d log h) fast transform is equivalent).
    """
    if h is None:
        h = _largest_pow2_divisor(d)
    assert d % h == 0, f"d={d} must be a multiple of block size h={h}"
    k = d // h
    H = _hadamard(h)
    blocks = []
    for _ in range(k):
        signs = rng.choice([-1.0, 1.0], size=h)
        blocks.append(H * signs)  # H_h D_i  (signs broadcast over columns)
    Pi = np.zeros((d, d))
    for j, B in enumerate(blocks):
        Pi[j * h:(j + 1) * h, j * h:(j + 1) * h] = B
    if use_perm:
        perm = rng.permutation(d)
        P = np.eye(d)[:, perm]
        Pi = Pi @ P
    return Pi


# --------------------------------------------------------------------------- #
#  Proposition 1 (Eq 10): universal variance concentration
# --------------------------------------------------------------------------- #
def prop1_rho(d, mu_inf, h, delta):
    """Concentration radius rho of Eq 10.

    rho = (d * mu_inf) / (2 h) * sqrt( (4 k / d) * log(1/delta) ),  k = d/h.

    The paper writes `sqrt((4k/d) log delta)`; for delta in (0,1) the log is
    negative, so the intended real-valued bound is log(1/delta) (standard
    Hoeffding-without-replacement form). Bound: |Var(z_i)*d - 1| <= rho.
    """
    k = d // h
    return (d * mu_inf) / (2.0 * h) * np.sqrt((4.0 * k / d) * np.log(1.0 / delta))


# --------------------------------------------------------------------------- #
#  Lloyd-Max codebook on f_d (Eq 3, 6, 8) + uniform-grid baseline
# --------------------------------------------------------------------------- #
def lloyd_max_codebook(d, b, n_grid=200001, iters=400):
    """MSE-optimal scalar codebook with 2**b centroids for the f_d marginal.

    Density-based Lloyd iteration on a fine [-1, 1] lattice weighted by f_d:
    each step recomputes Voronoi boundaries (midpoints of adjacent centroids)
    then sets each centroid to the f_d-mean of its cell. Init = uniform quantiles
    of f_d (already near-optimal), so this converges in a few steps. Returns the
    sorted centroid array and the achieved MSE.
    """
    edges = np.linspace(-1.0, 1.0, n_grid)
    w = fd_density(edges, d)
    w /= w.sum()
    # init: equal-mass quantiles under f_d
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    targets = (np.arange(2 ** b) + 0.5) / 2 ** b
    centroids = np.interp(targets, cdf, edges)
    centroids.sort()
    for _ in range(iters):
        mid = 0.5 * (centroids[:-1] + centroids[1:])
        # assign each grid point to nearest centroid
        idx = np.searchsorted(mid, edges)
        new = np.empty_like(centroids)
        for j in range(2 ** b):
            m = idx == j
            mass = w[m].sum()
            new[j] = (edges[m] * w[m]).sum() / mass if mass > 0 else centroids[j]
        if np.max(np.abs(new - centroids)) < 1e-12:
            centroids = new
            break
        centroids = new
    # MSE on the grid
    mid = 0.5 * (centroids[:-1] + centroids[1:])
    idx = np.searchsorted(mid, edges)
    q = centroids[idx]
    mse = np.sum(w * (edges - q) ** 2)
    return centroids, mse


def uniform_codebook(b):
    """Uniform (RTN-style) grid: 2**b evenly spaced levels on [-1, 1]."""
    return np.linspace(-1.0, 1.0, 2 ** b)


def uniform_mse(d, b, n_grid=200001):
    """MSE of the uniform grid quantizer against the f_d marginal."""
    edges = np.linspace(-1.0, 1.0, n_grid)
    centroids = uniform_codebook(b)
    w = fd_density(edges, d)
    w /= w.sum()
    mid = 0.5 * (centroids[:-1] + centroids[1:])
    idx = np.searchsorted(mid, edges)
    q = centroids[idx]
    return np.sum(w * (edges - q) ** 2)


def quantize_to_codebook(v, centroids):
    """Nearest-centroid quantization (Eq 3), v any shape."""
    v = np.asarray(v, dtype=float)
    mid = 0.5 * (centroids[:-1] + centroids[1:])
    flat = v.ravel()
    idx = np.searchsorted(mid, flat)
    q = centroids[idx].reshape(v.shape)
    return q


def lloyd_centroid_residual(d, centroids, n_grid=200001):
    """Max |centroid - cell f_d-mean| over cells -- 0 at a Lloyd-Max optimum."""
    edges = np.linspace(-1.0, 1.0, n_grid)
    w = fd_density(edges, d)
    w /= w.sum()
    mid = 0.5 * (centroids[:-1] + centroids[1:])
    idx = np.searchsorted(mid, edges)
    resid = 0.0
    for j in range(len(centroids)):
        m = idx == j
        mass = w[m].sum()
        if mass > 0:
            resid = max(resid, abs(centroids[j] - (edges[m] * w[m]).sum() / mass))
    return resid


# --------------------------------------------------------------------------- #
#  Rotation-cancels-in-the-product weight / activation quantization (Eq 4-8)
# --------------------------------------------------------------------------- #
def orbitquant_weight(W, Pi, centroids):
    """Eq 4-6.  W' = W Pi^T; split row norm / unit direction; quantize direction
    with the shared Lloyd-Max codebook; re-attach magnitude -> what-cap W'."""
    Wp = W @ Pi.T
    r = np.linalg.norm(Wp, axis=1, keepdims=True)          # Eq 5: r_i'
    Wdir = Wp / (r + 1e-12)                                # unit direction w~_i'
    Wq_dir = quantize_to_codebook(Wdir, centroids)         # Eq 6: Q_hat(W~')
    return Wq_dir * r                                      # diag(r') . Q_hat(W~')


def orbitquant_activation(x, Pi, centroids):
    """Eq 7-8.  x' = Pi x; s = ||x'||; x~' = x'/s; quantize; rescale -> x-hat'."""
    xp = Pi @ x
    s = np.linalg.norm(xp)
    xdir = xp / (s + 1e-12)
    return s * quantize_to_codebook(xdir, centroids)


def rtn_quantize(V, bits, axis=None):
    """Per-row (axis=1) or per-tensor min-max uniform (RTN) quantization baseline."""
    V = np.asarray(V, dtype=float)
    levels = 2 ** bits
    if axis is None:
        vmin, vmax = V.min(), V.max()
        step = (vmax - vmin) / (levels - 1)
        q = np.clip(np.round((V - vmin) / step), 0, levels - 1)
        return vmin + q * step
    vmin = V.min(axis=axis, keepdims=True)
    vmax = V.max(axis=axis, keepdims=True)
    step = (vmax - vmin) / (levels - 1)
    q = np.clip(np.round((V - vmin) / step), 0, levels - 1)
    return vmin + q * step


def relative_error(y_true, y_approx):
    """||y - ya|| / ||y||."""
    return np.linalg.norm(y_true - y_approx) / (np.linalg.norm(y_true) + 1e-30)
