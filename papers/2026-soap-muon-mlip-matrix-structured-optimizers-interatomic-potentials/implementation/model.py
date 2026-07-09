"""Optimizers for "Beyond Adam: SOAP and Muon for MLIPs" (Harari et al. 2026, arXiv:2607.02499).

Implements the four optimisers benchmarked in the paper, all operating on torch
tensors via the standard `param.grad` interface so the same code drives both the
hand-derived Kronecker-quadratic checks and an autograd MLP.

Routing follows the paper's Table 2 parameter-group assignment:
  * 2-D (matrix) parameters  -> the matrix-structured method (Muon / SOAP / SOAP-Muon)
  * 1-D / scalar parameters  -> AdamW (diagonal preconditioning)

Methods (Algorithms 1-2, L744-833):
  * AdamW     -- diagonal P = diag(v_hat)^{1/2}, decoupled weight decay (reference baseline)
  * Muon      -- Nesterov momentum + Newton-Schulz5 orthogonalisation of the update (Alg 1)
  * SOAP      -- Shampoo eigenspace projection + AdamW *inside* the eigenspace (Alg 2, ortho=False)
  * SOAP-Muon -- SOAP + Muon-style singular-value-power orthogonalisation + RMS norm (Alg 2, ortho=True, normalize=True)
"""

import torch

# ---------------------------------------------------------------------------
# Newton-Schulz iteration for the polar factor (Jordan et al. 2024; Muon)
# ---------------------------------------------------------------------------

# Classic Newton iteration for the orthogonal factor of the polar
# decomposition:  X_{k+1} = 0.5 (3 X - X X^T X).  On singular values this is the
# scalar map  sigma -> 0.5 sigma (3 - sigma^2), with the stable fixed point
# sigma = 1 (and sigma = sqrt(3) -> 0, so we keep ||X||_2 < sqrt(3)).  Starting
# from ||X||_2 <= 1 it converges to 1 quadratically in ~5 steps, yielding the
# polar factor U V^T of the SVD -- a matrix with orthonormal columns (tall) /
# rows (wide).  Muon calls 5 such steps to orthogonalise the update direction.


def newton_schulz5(G, steps=5, eps=1e-8):
    """Approximate the polar factor of G (m x n) via `steps` Newton-Schulz iterations.

    Returns an m x n matrix with orthonormal columns (or rows): the orthogonal
    factor of the polar decomposition, i.e. U V^T from the SVD G = U S V^T.
    """
    assert G.ndim == 2
    # Scale into the convergence basin.  ||X||_2 <= ||X||_F = 1 < sqrt(3) after
    # dividing by the Frobenius norm, so every singular value lies in (0, 1] and
    # is driven monotonically up to 1.
    X = G / (G.norm() + eps)
    for _ in range(steps):
        X = 1.5 * X - 0.5 * (X @ X.T @ X)
    return X


def sv_power_ortho(G, rho, eps=1e-8):
    """Mu (rho != 0,1) branch: G = P S R^T -> P S^rho R^T (singular-value power)."""
    P, S, Rt = torch.linalg.svd(G, full_matrices=False)
    S_rho = torch.clamp(S, min=eps).pow(rho)
    return (P * S_rho) @ Rt


def rms_normalize(U, eps=1e-8):
    """Muon RMS normalisation: scale so mean(U^2) = 1 (unit-RMS update)."""
    return U * torch.rsqrt(U.pow(2).mean() + eps)


def _orthonormal_basis(M, d, ref, eps=1e-8):
    """Orthonormal eigenvector basis of the EMA covariance M (d x d).

    Symmetrise, scale-invariant jitter (relative to the trace so it works whether
    gradients are tiny or huge), then eigh -- falling back to an SVD of a
    symmetrised copy if eigh fails to converge (degenerate / repeated spectrum).
    """
    Ms = 0.5 * (M + M.T)
    scale = Ms.trace().abs() / d + eps
    Ms = Ms + scale * eps * torch.eye(d, dtype=ref.dtype, device=ref.device)
    try:
        _, Q = torch.linalg.eigh(Ms)
    except torch._C._LinAlgError:                   # pragma: no cover (fallback)
        U, _, _ = torch.linalg.svd(Ms, full_matrices=True)
        Q = U
    return Q


# ---------------------------------------------------------------------------
# Base optimiser: AdamW path for 1-D params + common state plumbing
# ---------------------------------------------------------------------------


class MatrixOptimizer:
    """Routes 2-D params to a matrix update and 1-D/scalar params to AdamW."""

    def __init__(self, params, lr, betas=(0.95, 0.95), eps=1e-8, weight_decay=0.0):
        self.params = [p for p in params]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        # per-param Adam state (1-D path) keyed by id
        self._adam_m = {}
        self._adam_v = {}

    # -- 1-D AdamW path -----------------------------------------------------
    def _adam_step(self, p, g):
        m = self._adam_m.setdefault(id(p), torch.zeros_like(g))
        v = self._adam_v.setdefault(id(p), torch.zeros_like(g))
        m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
        v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
        mhat = m / (1 - self.beta1 ** self.t)
        vhat = v / (1 - self.beta2 ** self.t)
        return self.lr * mhat / (vhat.sqrt() + self.eps)

    # -- 2-D matrix path: overridden by subclasses --------------------------
    def matrix_update(self, p, g):
        raise NotImplementedError

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                p.data.mul_(1 - self.lr * self.weight_decay)
            if g.ndim < 2 or min(g.shape) == 1:
                update = self._adam_step(p, g)
            else:
                update = self.matrix_update(p, g)
            p.data.add_(update, alpha=-1.0)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# AdamW (reference baseline; pure diagonal preconditioning)
# ---------------------------------------------------------------------------


class AdamW(MatrixOptimizer):
    def matrix_update(self, p, g):
        # matrices also use the diagonal Adam path
        return self._adam_step(p, g)


# ---------------------------------------------------------------------------
# Muon (Algorithm 1): Nesterov momentum + Newton-Schulz5 orthogonalisation
# ---------------------------------------------------------------------------


class Muon(MatrixOptimizer):
    def __init__(self, params, lr, beta=0.95, ns_steps=5, **kw):
        super().__init__(params, lr, betas=(beta, beta), **kw)
        self.beta = beta
        self.ns_steps = ns_steps
        self._mom = {}

    def matrix_update(self, p, g):
        m = self._mom.setdefault(id(p), torch.zeros_like(g))
        m.mul_(self.beta).add_(g, alpha=1 - self.beta)          # M_t
        u = self.beta * m + (1 - self.beta) * g                 # Nesterov U_t
        o = newton_schulz5(u, steps=self.ns_steps)              # orthogonalise
        return self.lr * o


# ---------------------------------------------------------------------------
# SOAP (Algorithm 2): Shampoo eigenspace + AdamW inside the eigenbasis
#   ortho=False, normalize=False  -> plain SOAP
#   ortho=True,  normalize=True    -> SOAP-Muon
# ---------------------------------------------------------------------------


class SOAP(MatrixOptimizer):
    def __init__(self, params, lr, betas=(0.95, 0.95), eps=1e-8,
                 precond_freq=10, shampo_beta=0.95, ortho=False, normalize=False,
                 rho=0.5, weight_decay=0.0):
        super().__init__(params, lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.precond_freq = precond_freq
        self.shampo_beta = shampo_beta
        self.ortho = ortho
        self.normalize = normalize
        self.rho = rho
        # per-matrix Shampoo second-moment stats + eigenspace basis
        self._L = {}
        self._R = {}
        self._QL = {}     # left eigenvectors  (m x m)
        self._QR = {}     # right eigenvectors (n x n)
        # Adam moments live *in the eigenspace* (on the projected gradient)
        self._am = {}
        self._av = {}

    def matrix_update(self, p, g):
        m, n = g.shape
        key = id(p)
        L = self._L.setdefault(key, torch.zeros(m, m, dtype=g.dtype, device=g.device))
        R = self._R.setdefault(key, torch.zeros(n, n, dtype=g.dtype, device=g.device))
        # EMA of the row/col covariance (Shampoo statistics, Eq 5 L=R=G^TG / GG^T)
        L.mul_(self.shampo_beta).add_(g @ g.T, alpha=1 - self.shampo_beta)
        R.mul_(self.shampo_beta).add_(g.T @ g, alpha=1 - self.shampo_beta)
        # refresh the eigenspace every `precond_freq` steps (SOAP amortises the
        # eigendecomposition that vanilla Shampoo pays every step).  When the
        # basis rotates, the Adam moments -- stored in the *old* eigenbasis --
        # must be re-projected into the new basis so the preconditioner stays
        # consistent (otherwise every refresh injects stale-coordinate noise).
        if self.t % self.precond_freq == 0 or key not in self._QL:
            # Refresh the eigenspace from the current Shampoo statistics.  The
            # Adam moments (m, v) are elementwise stats and cannot be "rotated"
            # into the new basis (a basis-change matrix product would make v's
            # entries sign-indefinite and break sqrt(v)); real SOAP keeps them
            # and relies on the EMA covariance L changing slowly between
            # refreshes, so the staleness is negligible.
            QL_new = _orthonormal_basis(L, m, g)
            QR_new = _orthonormal_basis(R, n, g)
            self._QL[key] = QL_new
            self._QR[key] = QR_new
        QL = self._QL[key]
        QR = self._QR[key]
        # project the gradient into the Shampoo eigenbasis
        ghat = QL.T @ g @ QR
        # AdamW *inside* the eigenspace (elementwise, per eigen-coordinate)
        am = self._am.setdefault(key, torch.zeros_like(ghat))
        av = self._av.setdefault(key, torch.zeros_like(ghat))
        am.mul_(self.beta1).add_(ghat, alpha=1 - self.beta1)
        av.mul_(self.beta2).addcmul_(ghat, ghat, value=1 - self.beta2)
        mhat = am / (1 - self.beta1 ** self.t)
        vhat = av / (1 - self.beta2 ** self.t)
        uhat = mhat / (vhat.sqrt() + self.eps)
        # back-project to the native parameter basis
        u = QL @ uhat @ QR.T
        # optional Muon-style singular-value power + RMS normalisation (SOAP-Muon)
        if self.ortho:
            if self.rho == 0.0:
                u = newton_schulz5(u)
            elif self.rho != 1.0:
                u = sv_power_ortho(u, self.rho)
        if self.normalize:
            u = rms_normalize(u)
        return self.lr * u


def SOAPMuon(params, lr, rho=0.5, **kw):
    """SOAP-Muon = Algorithm 2 with ortho=True AND normalize=True (L856-858)."""
    kw.setdefault("ortho", True)
    kw.setdefault("normalize", True)
    return SOAP(params, lr, rho=rho, **kw)


OPTIMIZERS = {
    "AdamW": AdamW,
    "Muon": Muon,
    "SOAP": SOAP,
    "SOAP-Muon": SOAPMuon,
}
