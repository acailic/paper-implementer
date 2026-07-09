"""ART — Adaptive Reparameterized Time for diffusion sampling (Huang, Tang & Zhou 2026).

Core mechanism, isolated on a 1-D variance-exploding (VE) diffusion with an
*analytically known* score (Gaussian-mixture target).  The paper's contribution is a
control-theoretic framing of the **timestep schedule** (the σ-grid), not a new score
network or sampler.  Everything here is therefore schedule-only: the score model and
Euler integrator are fixed; only the grid changes.

The three load-bearing objects this module exposes:

* the backward probability-flow field ``F(x, sigma) = (x - E[x_0 | x]) / sigma``
  (Eq 5, VE specialization), with ``E[x_0|x]`` the denoising posterior mean;
* the **stiffness indicator** ``Q(sigma)`` — the mean squared trajectory curvature
  ``E_x[ d^2 x / d sigma^2 ]`` (Eq 8), i.e. the leading-order coefficient of the
  one-step Euler local-error surrogate ``~ (1/2) Delta-sigma^2 * sqrt(Q)`` (Eq 7);
* the **optimal ART schedule** ``theta* proportional to 1 / sqrt(Q)`` (Theorem 1
  consequence), derived by minimising ``integral Q * theta^2 dt`` s.t. the
  time-budget ``integral theta dt = T``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


# --------------------------------------------------------------------------- #
# 1. Target distribution + score of the noise-conditional marginal            #
# --------------------------------------------------------------------------- #
class GaussianMixture:
    """A 1-D Gaussian mixture ``sum_i w_i N(mu_i, s_i^2)``.

    The VE forward process ``x = x_0 + sigma * z`` convolves the target with
    ``N(0, sigma^2)``, so the noise-conditional marginal is itself a mixture with
    variances ``v_i + sigma^2``.  Its score (and the denoising posterior mean) are
    available in closed form, which is what makes the 1-D experiment a pure
    discretization-error probe with *no score-estimation error* (paper Table 1).
    """

    def __init__(self, mus, sigmas, weights):
        self.mu = np.asarray(mus, dtype=float)
        self.s = np.asarray(sigmas, dtype=float)
        self.v = self.s ** 2
        self.w = np.asarray(weights, dtype=float)
        self.w = self.w / self.w.sum()

    # ---- sampling ------------------------------------------------------- #
    def sample_target(self, n, rng):
        idx = rng.choice(len(self.w), size=n, p=self.w)
        return rng.normal(self.mu[idx], self.s[idx])

    def sample_noisy(self, n, sigma, rng):
        """Sample x ~ p(.; sigma) = target convolved with N(0, sigma^2)."""
        return self.sample_target(n, rng) + sigma * rng.standard_normal(n)

    # ---- posterior mean + score ---------------------------------------- #
    def _log_resp(self, x, sigma):
        """Log unnormalised component log-densities, shape (n, K)."""
        # log N(x; mu_i, v_i + sigma^2)
        var = self.v + sigma ** 2
        return (np.log(self.w)[None, :]
                - 0.5 * np.log(2.0 * np.pi * var)[None, :]
                - 0.5 * (x[:, None] - self.mu[None, :]) ** 2 / var[None, :])

    def posterior_mean(self, x, sigma):
        """E[x_0 | x] under the noise-conditional marginal (denoiser)."""
        log_r = self._log_resp(x, sigma)
        log_r -= logsumexp(log_r, axis=1, keepdims=True)  # normalise responsibilities
        r = np.exp(log_r)
        # posterior mean of component i: mu_i + v_i/(v_i+sigma^2) (x - mu_i)
        post_mean_i = self.mu[None, :] + (self.v / (self.v + sigma ** 2))[None, :] * (
            x[:, None] - self.mu[None, :]
        )
        return np.sum(r * post_mean_i, axis=1)

    def score(self, x, sigma):
        """Tweedie score  (E[x_0|x] - x) / sigma^2 ."""
        return (self.posterior_mean(x, sigma) - x) / (sigma ** 2)


# --------------------------------------------------------------------------- #
# 2. Backward probability-flow ODE field (Eq 5, VE)                          #
# --------------------------------------------------------------------------- #
def field_F(target, x, sigma):
    """Backward PF-ODE field  F = -sigma * score = (x - E[x_0|x]) / sigma."""
    return (x - target.posterior_mean(x, sigma)) / sigma


def euler_reverse(target, x0, sigmas):
    """Integrate the reverse PF-ODE on a *descending* sigma grid.

    ``sigmas`` must run from sigma_max (noisy) down to sigma_min (clean); each
    Euler step advances by ``Delta-sigma = sigmas[k+1] - sigmas[k] < 0``.  Returns
    the denoised x estimate at sigma_min.  NFE = len(sigmas) - 1.
    """
    x = np.asarray(x0, dtype=float).copy()
    for k in range(len(sigmas) - 1):
        s_k = sigmas[k]
        x = x + (sigmas[k + 1] - sigmas[k]) * field_F(target, x, s_k)
    return x


# --------------------------------------------------------------------------- #
# 3. Stiffness indicator Q(sigma)  (Eq 7-8)                                  #
# --------------------------------------------------------------------------- #
def trajectory_curvature_sq(target, x, sigma, ds):
    """Squared second derivative ``(d^2 x / d sigma^2)`` along one trajectory.

    The local one-step Euler error is ``~ (1/2) Delta-sigma^2 * x-ddot`` (Eq 7);
    ``x-ddot`` is the *total* derivative ``d/d sigma F(x(sigma), sigma)`` along the
    trajectory.  Estimated by a single tiny Euler advance + central difference,
    which is exact for the leading term and works for *any* score field.
    """
    F0 = field_F(target, x, sigma)
    x_plus = x + ds * F0
    s_plus = sigma + ds
    F_plus = field_F(target, x_plus, s_plus)
    xddot = (F_plus - F0) / ds
    return xddot ** 2


def stiffness_Q(target, sigmas, n_samples, rng, ds_rel=1e-3):
    """Monte-Carlo estimate of the stiffness field  Q(sigma) = E_x[ x-ddot^2 ]."""
    Q = np.empty_like(sigmas)
    for i, s in enumerate(sigmas):
        x = target.sample_noisy(n_samples, s, rng)
        ds = max(ds_rel * s, 1e-6)
        Q[i] = np.mean(trajectory_curvature_sq(target, x, s, ds))
    return Q


def Q_func(target, sigma_min, sigma_max, n_samples, rng, n_grid=2000):
    """Return a fine lattice (sigmas, Q(sigmas)) spanning the schedule range."""
    sigmas = np.linspace(sigma_min, sigma_max, n_grid)
    Q = stiffness_Q(target, sigmas, n_samples, rng)
    return sigmas, np.clip(Q, 1e-30, None)


def closed_form_Q_gaussian(sigma, s):
    """Analytic Q for a *single* N(0, s^2) VE target (consistency check only).

    F = x*sigma/(s^2+sigma^2) = x*g, g=sigma/(s^2+sigma^2).  The trajectory
    curvature is the *total* derivative  x-ddot = d/d sigma (x*g) = x*(g^2 + g'),
    with g'=(s^2-sigma^2)/(s^2+sigma^2)^2 and g^2=sigma^2/(s^2+sigma^2)^2, hence
    g^2+g' = s^2/(s^2+sigma^2)^2.  Then E[x^2]=s^2+sigma^2 gives
    Q = E[x-ddot^2] = s^4 / (s^2+sigma^2)^3  (monotone-decreasing, peaked at sigma=0).
    """
    return s ** 4 / (s ** 2 + sigma ** 2) ** 3


# --------------------------------------------------------------------------- #
# 4. Timestep schedules (only the grid differs; score + integrator fixed)    #
# --------------------------------------------------------------------------- #
def schedule_uniform(K, sigma_min, sigma_max):
    """Equally spaced sigma grid (descending)."""
    return np.linspace(sigma_max, sigma_min, K + 1)


def schedule_dpm(K, sigma_min, sigma_max):
    """DPM-Solver VE: geometric grid  sigma_k = sigma_max*(sigma_min/sigma_max)^(k/K)
    (uniform in log-SNR)."""
    ratio = sigma_min / sigma_max
    k = np.arange(K + 1)
    return sigma_max * ratio ** (k / K)


def schedule_edm(K, sigma_min, sigma_max, rho=7.0):
    """EDM (Karras 2022): uniform in sigma^(1/rho)."""
    k = np.arange(K + 1) / K
    return (sigma_max ** (1.0 / rho)
            + k * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho


def schedule_art(K, sigma_min, sigma_max, fine_sigmas, Q):
    """Optimal ART grid: uniform in the reparameterised clock t.

    Equal clock budget per step  <=>  equal  integral sqrt(Q) dsigma  per bin
    (since dt = dsigma/theta and theta* ~ 1/sqrt(Q)).  Equivalent to a grid whose
    local spacing Delta-sigma ~ 1/sqrt(Q): fine where stiff (large Q), coarse where
    the dynamics is smooth.  Built by inverting the cumulative sqrt(Q) measure.
    """
    # ascending lattice
    s = fine_sigmas
    q = np.clip(Q, 1e-30, None)
    integrand = np.sqrt(q)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                            * (s[1:] - s[:-1]))])
    cum /= cum[-1]
    targets = np.linspace(0.0, 1.0, K + 1)
    grid = np.interp(targets, cum, s)          # ascending, sigma_min..sigma_max
    grid[0], grid[-1] = sigma_min, sigma_max   # pin endpoints exactly
    return grid[::-1]                          # descending for the reverse loop


# --------------------------------------------------------------------------- #
# 5. Continuous ART objective J(theta) + the optimal theta* (Theorem 1)      #
# --------------------------------------------------------------------------- #
def schedule_theta(grid, sigma_min, sigma_max):
    """Per-step control rate theta_k = |Delta-sigma_k| / Delta-t  on a grid.

    With the clock length ``T = sigma_max - sigma_min`` split into ``K`` uniform
    intervals ``Delta-t = T/K``, theta_k = |Delta-sigma_k| * K / T.
    """
    K = len(grid) - 1
    T = sigma_max - sigma_min
    return np.abs(np.diff(grid)) * K / T


def art_cost(grid, Q_at_grid):
    """Discretised ART cost  sum_k Q(sigma_k) * Delta-sigma_k^2  (Eq 9 in sigma).

    Proportional to the continuous objective ``integral Q*theta dsigma``; lower is
    better.  Theta-optimal (theta ~ 1/sqrt(Q)) makes  Q * theta^2 = const  across
    steps, so this cost collapses toward its minimum.
    """
    dsig = np.abs(np.diff(grid))
    return np.sum(Q_at_grid[:-1] * dsig ** 2)


def optimal_theta_certificate(grid, Q_at_grid, sigma_min, sigma_max):
    """How flat is  Q(sigma)*theta(sigma)^2  across the grid?

    At the Theorem-1 optimum theta* ~ 1/sqrt(Q) the product Q*theta^2 is *constant*
    (the Euler-Lagrange condition of the budget-constrained problem).  We report the
    coefficient of variation of that product across steps: ~0 iff optimal.
    """
    theta = schedule_theta(grid, sigma_min, sigma_max)
    prod = Q_at_grid[:-1] * theta ** 2
    return float(np.std(prod) / (np.mean(prod) + 1e-30))


# --------------------------------------------------------------------------- #
# 5b. Continuous theta(theta) per schedule  (for the theoretical J comparison) #
# --------------------------------------------------------------------------- #
def continuous_thetas(sigma_lattice, Q, sigma_min, sigma_max, rho=7.0):
    """Each schedule's continuous control rate theta(sigma)=d sigma/d t  on a lattice.

    Used for the *theoretical* continuous-objective comparison  J = integral Q*theta
    d sigma  (Eq 9 in sigma-space) with the budget  integral (1/theta) d sigma = T.
    Returned as a dict of theta(sigma) arrays (all > 0).
    """
    s = np.asarray(sigma_lattice, dtype=float)
    T = sigma_max - sigma_min
    q = np.clip(Q, 1e-30, None)
    sqrtQ = np.sqrt(q)
    c = np.trapezoid(sqrtQ, s) / T                       # budget-normalising constant

    # ART: theta* = c / sqrt(Q)   (Theorem 1)
    theta_art = c / sqrtQ
    # Uniform: constant rate (Delta-sigma = Delta-t  =>  theta = 1)
    theta_unif = np.ones_like(s)
    # DPM (geometric): sigma = sigma_max*(sigma_min/sigma_max)^u, t=u*T
    #   d sigma/d u = sigma*ln(sigma_min/sigma_max)  =>  theta = |d sigma/d u| / T
    theta_dpm = np.abs(s * np.log(sigma_min / sigma_max)) / T
    # EDM: uniform in sigma^(1/rho)  =>  d sigma/d u = rho*sigma^((rho-1)/rho)*(...)
    a_min, a_max = sigma_min ** (1.0 / rho), sigma_max ** (1.0 / rho)
    dsdu = np.abs(rho * s ** ((rho - 1.0) / rho) * (a_min - a_max))
    theta_edm = dsdu / T
    return {"ART": theta_art, "Uniform": theta_unif,
            "DPM": theta_dpm, "EDM": theta_edm}


def continuous_J(theta, Q, sigma_lattice):
    """Discretised continuous ART objective  J = integral Q*theta d sigma."""
    return float(np.trapezoid(np.asarray(Q) * np.asarray(theta), sigma_lattice))


def continuous_budget(theta, sigma_lattice):
    """Clock budget  integral (1/theta) d sigma  (must equal T for a valid schedule).

    Integrated on a log-spaced lattice: DPM/EDM have theta proportional to sigma, so
    1/theta carries a 1/sigma factor that a linear lattice mis-integrates near
    sigma_min.  The budget is exactly T analytically for any monotone schedule
    (sigma:[0,T]->[sigma_min,sigma_max]); this just evaluates it accurately.
    """
    s = np.asarray(sigma_lattice, dtype=float)
    th = np.asarray(theta, dtype=float)
    s_lo, s_hi = s.min(), s.max()
    if s_lo <= 0:
        return float(np.trapezoid(1.0 / th, s))
    s_log = np.logspace(np.log10(s_lo), np.log10(s_hi), 8000)
    th_log = np.interp(s_log, s, th)
    return float(np.trapezoid(1.0 / th_log, s_log))


# --------------------------------------------------------------------------- #
# 6. ART-RL Gaussian policy (Eq 11, Theorem 1)                               #
# --------------------------------------------------------------------------- #
def gaussian_policy_sample(theta_star_at_grid, Q_at_grid, lam, rng):
    """Sample theta ~ N(mu*, lam / Q)  per step  (Eq 11).

    mu* = theta* (the deterministic optimum, by Theorem 1); variance lam/Q is
    *tied to stiffness* — randomness is suppressed in stiff (large-Q) regions.
    """
    var = lam / np.clip(Q_at_grid[:-1], 1e-30, None)
    return theta_star_at_grid + np.sqrt(var) * rng.standard_normal(len(var))


def normalise_theta(theta, sigma_min, sigma_max):
    """Project a sampled rate profile onto the budget manifold  integral theta dt=T.

    Rescales so that the induced sigma-increments  Delta-sigma_k = theta_k*Delta-t
    sum to  sigma_max - sigma_min.  (The Lagrange-multiplier update Eq 30 does this
    incrementally; a one-shot rescale is its fixed point.)
    """
    K = len(theta)
    T = sigma_max - sigma_min
    dt = T / K
    span = np.sum(theta * dt)
    return theta * (T / span) if span > 0 else theta


# --------------------------------------------------------------------------- #
# 7. Exact 1-D squared Wasserstein-2                                         #
# --------------------------------------------------------------------------- #
def wasserstein2_sq(a, b):
    """Exact 1-D squared W2 between equal-size empirical samples (quantile match)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    assert len(a) == len(b), "W2 quantile match needs equal-length samples"
    return float(np.mean((a - b) ** 2))
