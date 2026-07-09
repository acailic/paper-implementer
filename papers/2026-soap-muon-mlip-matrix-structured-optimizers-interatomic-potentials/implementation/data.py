"""Synthetic problems for the SOAP/Muon MLIP optimiser bake-off.

Three problems, each isolating one claim of Harari et al. 2026:

  * kronecker_quadratic()  -- anisotropic L = 0.5 ||A W B - C||_F^2 whose
    Hessian is the Kronecker product A^T A (X) B B^T, i.e. exactly the
    L (X) R structure that Shampoo/SOAP preconditioning is built to undo.
    SOAP should converge far faster than diagonal AdamW here.

  * rank1_gradient_problem() -- a loss whose gradient w.r.t. W is rank-1
    (signal concentrated on a single direction).  Muon's Newton-Schulz
    orthogonalisation promotes that single singular value to a full-rank
    orthonormal update, injecting mass into directions that carry no
    gradient signal -> demonstrable degradation vs AdamW (the paper's
    honest-scope "orthogonalisation destabilises" finding, water-energy).

  * EnergyForceDataset -- a coordinate->scalar-energy regression where the
    "force" label is the analytic gradient dE/dx.  Lets us train with
    energy supervision always on and *force* supervision at a tunable
    fraction p, reproducing the paper's reduced-force-supervision regime
    (the label-efficiency axis: DFT forces cheap, CC/QMC forces costly).
"""

import torch


# ---------------------------------------------------------------------------
# (1) Anisotropic Kronecker-curvature quadratic
# ---------------------------------------------------------------------------

def kronecker_quadratic(m=8, n=6, cond=400.0, seed=0):
    """L(W) = 0.5 ||A W B - C||_F^2 with A, B ill-conditioned.

    Hessian on vec(W) is (A^T A) (X) (B B^T): a genuine Kronecker product, so
    Shampoo's L = G G^T ~ A^T (..) A and R = G^T G ~ B (..) B^T recover the
    true row/column curvature -- the regime where SOAP's preconditioner is
    informative and diagonal AdamW's is not.
    """
    g = torch.Generator().manual_seed(seed)
    # symmetric positive-definite A, B with condition number `cond`:
    # random orthogonal eigenvectors + a linearly-spaced spectrum 1 -> 1/cond.
    def sym_cond(d):
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        s = torch.linspace(1.0, 1.0 / cond, d)
        return Q @ torch.diag(s) @ Q.T
    A = sym_cond(m)
    B = sym_cond(n)
    W_true = torch.randn(m, n, generator=g) * 0.5
    C = A @ W_true @ B
    return A, B, C


def kronecker_loss(W, A, B, C):
    R = A @ W @ B - C
    return 0.5 * (R * R).sum()


def kronecker_grad(W, A, B, C):
    R = A @ W @ B - C
    return A.T @ R @ B.T


# ---------------------------------------------------------------------------
# (2) Rank-1-gradient problem (Muon-degradation probe)
# ---------------------------------------------------------------------------

def rank1_problem(m=8, n=6, seed=1):
    """L(W) = 0.5 (u^T W v - t)^2 for fixed unit vectors u, v and target t.

    Gradient = (u^T W v - t) * u v^T  -- always rank-1 (a single outer product).
    The optimum is the affine subspace {W : u^T W v = t}; the *direction* to
    move is u v^T, and its *magnitude* is the residual.  Orthogonalising this
    rank-1 gradient (Muon) returns the same rank-1 direction (NS5 of a rank-1
    matrix is itself, normalised) -- BUT it discards the residual magnitude
    information, so every step has a fixed-norm update regardless of how close
    we are.  That prevents the natural slow-down near convergence and also
    fights the residual sign when the adaptive LR is mismatched.  Net: on this
    problem the orthogonalised, magnitude-blind update converges to a *worse*
    fixed point than the magnitude-aware AdamW -- the in-vitro analogue of the
    paper's "orthogonalisation step is the primary source of degradation".
    """
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(m, generator=g); u = u / u.norm()
    v = torch.randn(n, generator=g); v = v / v.norm()
    t = torch.tensor(1.0)
    return u, v, t


def rank1_loss(W, u, v, t):
    r = (u @ W @ v - t)
    return 0.5 * r * r


def rank1_grad(W, u, v, t):
    r = (u @ W @ v - t)
    return r * torch.outer(u, v)


# ---------------------------------------------------------------------------
# (3) Energy + sparse-force regression (label-efficiency regime)
# ---------------------------------------------------------------------------

class EnergyForceMLP(torch.nn.Module):
    """E(x) = scalar energy head over a 2-layer MLP; force = -dE/dx via autograd.

    Mirrors the MLIP setup (Eq 1-2): energy decomposes through a network and
    forces are the negative energy gradient w.r.t. coordinates.  The training
    loss (Eq 3) is a weighted energy + force MAE; here we expose the *fraction*
    of samples that carry a force label so we can study reduced supervision.
    """

    def __init__(self, dim=4, hidden=32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )
        # init like a small regression
        for p in self.parameters():
            torch.nn.init.normal_(p, std=0.3, generator=g)
        self.dim = dim

    def energy(self, x):
        return self.net(x).squeeze(-1)

    def forward(self, x):
        # forces = -dE/dx  (Hellmann-Feynman, Eq 2)
        x = x.detach().requires_grad_(True)
        E = self.energy(x)
        g, = torch.autograd.grad(E.sum(), x, create_graph=True)
        return E, -g


def energy_force_step(model, optimizer, x, energy_label, force_label,
                      force_mask, lam_e=1.0, lam_f=1.0):
    """One optimisation step on weighted energy + (masked) force MAE (Eq 3)."""
    E_pred, F_pred = model(x)
    e_loss = (E_pred - energy_label).abs().mean()
    if force_mask.any():
        f_loss = (F_pred[force_mask] - force_label[force_mask]).abs().mean()
    else:
        f_loss = E_pred.new_zeros(())
    loss = lam_e * e_loss + lam_f * f_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), e_loss.item(), f_loss.item() if force_mask.any() else float("nan")


def make_energy_force_data(n=512, dim=4, seed=0):
    """Random coords x with a synthetic target energy = smooth function of x.

    Force label = analytic gradient of that target energy (so the model has a
    consistent, learnable energy/force field).  Returns tensors for a full
    epoch; the force-supervision fraction is applied per-step via a mask.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    # target energy: smooth quadratic + sinusoid (nonlinear, nonconvex surface)
    A = torch.randn(dim, dim, generator=g) * 0.3
    quad = ((x @ A) ** 2).sum(-1)
    sin = torch.sin(2.0 * x).sum(-1)
    E_target = 0.1 * quad + 0.2 * sin
    # force = -dE_target/dx (analytic)
    F_target = -(0.2 * (x @ (A + A.T))) - 0.4 * torch.cos(2.0 * x)
    return x, E_target, F_target
