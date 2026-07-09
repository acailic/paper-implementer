"""Data for the Rayleigh-Benard-convection FNO verification harness.

The paper studies a Fourier Neural Operator surrogate of a 2D turbulent fluid
PDE. The load-bearing, falsifiable mechanism does NOT need turbulence: it needs a
PDE whose one-step map is (a) known in closed form and (b) Fourier-diagonal, so
an FNO spectral layer can represent it *exactly* and so the SAME continuous
solution can be sampled at arbitrary grid resolution (the mesh-invariance test).

We use 2D periodic linear advection-diffusion

    d a / dt  =  -vel . grad a  +  kappa * laplacian a ,     a(x,y, t) on [0,2pi)^2 .

On the torus the solution is diagonal in Fourier space: each integer-wavenumber
mode (kx,ky) evolves as

    a_hat_{kx,ky}(t+dt) = M_{kx,ky} * a_hat_{kx,ky}(t),
    M_{kx,ky} = exp( (-i*(vx*kx + vy*ky) - kappa*(kx^2+ky^2)) * dt ).

This is the exact one-step integrator (no splitting error). A multi-channel
"state" U = (u, theta) just stacks two such fields with their own (vel, kappa).

The continuous field is held as its integer Fourier-mode table so it can be
sampled on ANY grid N (>= 2*max|index|+1) -- this is what makes the
mesh-invariance test honest: the exact same solution at different resolutions.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def integer_wavenumbers(N):
    """Angular wavenumbers for an N-point grid on [0,2pi): integers
    [0,1,...,N/2-1, -N/2, ..., -1] (numpy.fft.fftfreq convention)."""
    return np.fft.fftfreq(N, d=1.0 / N).astype(float)  # integer-valued floats


def step_multiplier(N, kappa, vel, dt):
    """Exact one-step Fourier-diagonal multiplier M_{kx,ky} on an N-grid.

    Returns a real (kappa>0, vel=0) or complex array shape (N, N) in fft2 layout.
    """
    k = integer_wavenumbers(N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX * KX + KY * KY
    return np.exp((-1j * (vel[0] * KX + vel[1] * KY) - kappa * K2) * dt)


def step_exact(field, N, kappa, vel, dt):
    """Apply the exact one-step advection-diffusion map to a grid-N field."""
    M = step_multiplier(N, kappa, vel, dt)
    return np.fft.ifft2(np.fft.fft2(field) * M).real.astype(np.float32)


class ContinuousField:
    """A 2D periodic scalar field held as its integer Fourier-mode table.

    modes : (M,2) integer wavenumber pairs (kx,ky), may be negative.
    coeffs: (M,) complex amplitudes.

    `.sample(N)` realises the field on an N-point grid by writing the modes into
    the N-grid spectrum (wrapping negative indices) and inverse-FFT-ing. Modes
    with |kx| or |ky| >= N/2 alias -- intentionally, for the resolution-bound
    test.
    """

    def __init__(self, modes, coeffs):
        self.modes = np.asarray(modes, dtype=int).reshape(-1, 2)
        self.coeffs = np.asarray(coeffs, dtype=complex).reshape(-1)

    @classmethod
    def random(cls, max_idx, decay, rng, high_tail=None, n_modes=None):
        """Band-limited random field: integer modes |kx|,|ky|<=max_idx with
        Gaussian amplitudes decaying as exp(-decay*|k|).

        high_tail : optional (idx_lo, idx_hi, rel_amp) adding a Gaussian high-
            frequency tail in the annulus idx_lo<|k|<=idx_hi to probe the
            training-resolution bound.
        n_modes   : cap on number of modes (None = all in the band).
        """
        idx = np.arange(-max_idx, max_idx + 1)
        KX, KY = np.meshgrid(idx, idx, indexing="ij")
        K = np.sqrt(KX * KX + KY * KY)
        sel = K <= max_idx
        kxv, kyv = KX[sel], KY[sel]
        kv = K[sel]
        amp = (rng.standard_normal(sel.sum()) + 1j * rng.standard_normal(sel.sum()))
        amp = amp * np.exp(-decay * kv)
        modes = np.stack([kxv, kyv], axis=1)
        coeffs = amp
        if high_tail is not None:
            lo, hi, rel = high_tail
            Kf = np.sqrt(kxv * kxv + kyv * kyv)
            tail = (Kf > lo) & (Kf <= hi)
            coeffs = coeffs.copy()
            coeffs[tail] *= rel  # amplify the high-frequency tail
        if n_modes is not None and len(modes) > n_modes:
            pick = rng.choice(len(modes), size=n_modes, replace=False)
            modes, coeffs = modes[pick], coeffs[pick]
        return cls(modes, coeffs)

    def sample(self, N):
        spec = np.zeros((N, N), dtype=complex)
        mi = self.modes[:, 0] % N
        mj = self.modes[:, 1] % N
        np.add.at(spec, (mi, mj), self.coeffs)
        return np.fft.ifft2(spec).real.astype(np.float32)


# --------------------------------------------------------------------------
# PDE "state" = a small stack of scalar fields (channels), each its own physics.
# Mirrors the paper's U = [u, w, theta, p]; we use 2 channels (e.g. u, theta)
# which is enough to expose the mechanism without matching RBC numerics.
# --------------------------------------------------------------------------

DEFAULT_CHANNELS = [
    dict(kappa=0.05, vel=(0.6, 0.25)),   # advection-dominated (velocity-like)
    dict(kappa=0.20, vel=(0.10, -0.4)),  # diffusion-dominated (buoyancy-like)
]


def make_state(rng, max_idx, decay, n_channels=None, high_tail=None, n_modes=None):
    """Return a list of ContinuousField, one per channel."""
    chans = DEFAULT_CHANNELS if n_channels is None else DEFAULT_CHANNELS[:n_channels]
    return [
        ContinuousField.random(max_idx, decay, rng, high_tail=high_tail, n_modes=n_modes)
        for _ in chans
    ]


def sample_state(state, N):
    return np.stack([f.sample(N) for f in state], axis=0)  # (C, N, N)


def step_state(state_grid, N, dt):
    """Exact one-step of a (C,N,N) state under DEFAULT_CHANNELS physics."""
    out = np.empty_like(state_grid)
    for c, phys in enumerate(DEFAULT_CHANNELS):
        out[c] = step_exact(state_grid[c], N, phys["kappa"], phys["vel"], dt)
    return out.astype(np.float32)


def gen_pairs(N, dt, n_pairs, max_idx, decay, seed, high_tail=None, n_modes=None):
    """Generate (U_t, U_{t+dt}) pairs at grid N with exact one-step ground truth.

    Returns
    -------
    U0 : (n_pairs, C, N, N) float32   -- state at t
    U1 : (n_pairs, C, N, N) float32   -- exact state at t+dt
    dU : (n_pairs, C, N, N) float32   -- (U1 - U0)/dt  (the increment target)
    """
    rng = np.random.default_rng(seed)
    U0 = np.empty((n_pairs, len(DEFAULT_CHANNELS), N, N), dtype=np.float32)
    U1 = np.empty_like(U0)
    for i in range(n_pairs):
        st = make_state(rng, max_idx, decay, high_tail=high_tail, n_modes=n_modes)
        u0 = sample_state(st, N)
        u1 = step_state(u0, N, dt)
        U0[i], U1[i] = u0, u1
    dU = (U1 - U0) / dt
    return U0, U1, dU


def identity_relative_error(U0, U1):
    """Relative L2 error of the identity predictor (predict U0 for U1),
    per-sample averaged -- the 'do nothing' baseline of the paper."""
    num = np.linalg.norm(U1 - U0)
    den = np.linalg.norm(U1)
    return float(num / den)


def gen_single_channel_pairs(N, dt, n_pairs, max_idx, decay, kappa, vel, seed,
                             high_tail=None):
    """Single-channel advection-diffusion pairs (for the C2 isolation test).

    Returns U0, U1, dU each shape (n_pairs, 1, N, N). With kappa=0 this is pure
    advection d_t U = -vel.grad U -- a strictly spatial operator a per-pixel map
    cannot represent, so it cleanly isolates the spectral conv's role.
    """
    rng = np.random.default_rng(seed)
    U0 = np.empty((n_pairs, 1, N, N), dtype=np.float32)
    U1 = np.empty_like(U0)
    for i in range(n_pairs):
        f = ContinuousField.random(max_idx, decay, rng, high_tail=high_tail)
        u0 = f.sample(N)
        U0[i, 0] = u0
        U1[i, 0] = step_exact(u0, N, kappa, vel, dt)
    dU = (U1 - U0) / dt
    return U0, U1, dU
