"""
OmniOpt meta-pipeline framework — toy implementation.

Implements the paper's 5-stage meta-pipeline (S0-S5) and LMO four-axis decomposition
for 4 representative optimizers across 3 families.

Paper: Xu et al. (2026), "OmniOpt: Taxonomy, Geometry, and Benchmarking of Modern Optimizers",
arXiv:2607.04033.

Pipeline stages:
  S0: Signal Acquisition     — gradient computation (handled by loss.backward())
  S1: Scoping & Routing      — parameter grouping (matrix vs vector)
  S2: Gradient Transformation — identity / sign / spectral orthogonalization
  S3: State Evolution         — moment EMAs, state updates
  S4: Update Reconstruction  — inverse of S2 (lift back to param space)
  S5: Update Finalization     — LR scaling, weight decay, clipping

Optimizers:
  SGDM    (T1) — S3(momentum) + S5(LR+WD)
  AdamW   (T1) — S3(m+v moments) + S5(LR+decoupled WD)
  Lion    (T3) — S3(sign+momentum) + S5(LR+WD)
  Muon    (T2) — S1(matrix routing) + S2(spectral orthogonalization) + S5(LR)
"""

import torch
import time


# ── Meta-Pipeline Base ─────────────────────────────────────────────

class PipelineOptimizer:
    """
    Unified 5-stage optimizer following OmniOpt meta-pipeline.
    Subclasses override stages they modify; identity stages are no-ops.
    """

    family = "T?"
    name = "Base"
    active_stages = []
    lmo_geometry = "?"
    axes = {}  # four-axis description

    def __init__(self, params, lr=1e-3, weight_decay=0.0, **kwargs):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.state = {}  # per-parameter state
        self._init_state(**kwargs)

    def _init_state(self, **kwargs):
        """Override to initialize optimizer-specific state."""
        pass

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        t0 = time.perf_counter()

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            # S1: Route — classify parameter (matrix vs vector)
            route = self._route(p, i)

            # S2: Transform — modify gradient geometry
            g_transformed = self._transform(g, p, i, route)

            # S3: Evolve — update internal state
            self._evolve(g_transformed, p, i, route)

            # S4: Reconstruct — lift back to parameter space
            update = self._reconstruct(p, i, route)

            # S5: Finalize — LR, weight decay, clipping
            self._finalize(p, update, i, route)

        self._step_time = time.perf_counter() - t0
        return None

    def state_memory_bytes(self):
        """Return approximate optimizer state memory in bytes."""
        total = 0
        for key, val in self.state.items():
            if isinstance(val, torch.Tensor):
                total += val.nelement() * val.element_size()
        return total

    # ── Stage overrides (identity by default) ──

    def _route(self, p, idx):
        """S1: Return 'matrix' or 'vector'."""
        return "vector"

    def _transform(self, g, p, idx, route):
        """S2: Gradient transformation. Returns transformed gradient."""
        return g

    def _evolve(self, g, p, idx, route):
        """S3: State evolution (moment EMAs, etc.)."""
        pass

    def _reconstruct(self, p, idx, route):
        """S4: Update reconstruction. Returns raw update direction."""
        return self.state.get((idx, "update"), torch.zeros_like(p))

    def _finalize(self, p, update, idx, route):
        """S5: Final update — LR scaling, weight decay, clipping."""
        p.add_(update, alpha=-self.lr)
        if self.weight_decay > 0:
            p.add_(p, alpha=-self.lr * self.weight_decay)


# ── T1 Family: Element-wise Adaptive Moment ────────────────────────

class SGDM(PipelineOptimizer):
    """SGD with Momentum — T1.1. Active: S3(momentum), S5(LR)."""
    family = "T1"
    name = "SGDM"
    active_stages = ["S3", "S5"]
    lmo_geometry = "Euclidean (l2 ball)"

    def _init_state(self, **kwargs):
        self.beta1 = kwargs.get("beta1", 0.9)

    def _evolve(self, g, p, idx, route):
        if (idx, "m") not in self.state:
            self.state[(idx, "m")] = torch.zeros_like(p)
        m = self.state[(idx, "m")]
        m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
        self.state[(idx, "update")] = m

    axes = {
        "I": "R^d (element-wise)",
        "II": "moment m_t",
        "III (LMO)": "l2 ball, Phi(g)=g/||g||",
        "III (Precond)": "H_t = I",
        "IV": "LR",
    }


class AdamW(PipelineOptimizer):
    """AdamW — T1.1. Active: S3(m+v EMAs), S5(LR+decoupled WD)."""
    family = "T1"
    name = "AdamW"
    active_stages = ["S3", "S5"]
    lmo_geometry = "Adaptive l_inf (Adam box)"

    def _init_state(self, **kwargs):
        self.beta1 = kwargs.get("beta1", 0.9)
        self.beta2 = kwargs.get("beta2", 0.999)
        self.eps = kwargs.get("eps", 1e-8)

    def _evolve(self, g, p, idx, route):
        if (idx, "m") not in self.state:
            self.state[(idx, "m")] = torch.zeros_like(p)
            self.state[(idx, "v")] = torch.zeros_like(p)
        m = self.state[(idx, "m")]
        v = self.state[(idx, "v")]
        m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
        v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
        # Adaptive update: m / sqrt(v) — the Adam LMO
        self.state[(idx, "update")] = m / (v.sqrt() + self.eps)

    axes = {
        "I": "R^d (element-wise)",
        "II": "m_t, v_t (EMA pair)",
        "III (LMO)": "adaptive l_inf, lmo = -rho * m/sqrt(v)",
        "III (Precond)": "H_t = diag(v_t)",
        "IV": "LR + decoupled WD",
    }


# ── T3 Family: Discretization & Sign ───────────────────────────────

class Lion(PipelineOptimizer):
    """Lion — T3. Active: S3(sign+momentum interpolation), S5(LR+WD).
    Update = sign(beta1*m + (1-beta1)*g), where m is exponential moving average of updates."""
    family = "T3"
    name = "Lion"
    active_stages = ["S2 (sign)", "S3 (update EMA)"]
    lmo_geometry = "Fixed l_inf (sign)"

    def _init_state(self, **kwargs):
        self.beta1 = kwargs.get("beta1", 0.9)
        self.beta2 = kwargs.get("beta2", 0.99)  # Lion uses beta2 for update EMA

    def _evolve(self, g, p, idx, route):
        if (idx, "c") not in self.state:
            self.state[(idx, "c")] = torch.zeros_like(p)  # update EMA
        c = self.state[(idx, "c")]
        # Lion: mix of gradient and update EMA, then sign
        mixed = self.beta1 * c + (1 - self.beta1) * g
        update = torch.sign(mixed)
        # Update the EMA of the update itself
        c.mul_(self.beta2).add_(update, alpha=1 - self.beta2)
        self.state[(idx, "update")] = update

    axes = {
        "I": "R^d (element-wise)",
        "II": "c_t (update EMA)",
        "III (LMO)": "fixed l_inf, Phi(g)=sign(g)",
        "III (Precond)": "H_t = diag(|m_t|)",
        "IV": "LR + WD",
    }


# ── T2 Family: Matrix Structural ───────────────────────────────────

class Muon(PipelineOptimizer):
    """Muon — T2.1. Active: S1(matrix routing), S2(NS spectral orthogonalization), S5(LR).
    Simplified: for 2D weight matrices, applies polar decomposition (UV^T) to momentum."""
    family = "T2"
    name = "Muon"
    active_stages = ["S1 (matrix)", "S2 (spectral)"]
    lmo_geometry = "Spectral norm (polar)"

    def _init_state(self, **kwargs):
        self.beta1 = kwargs.get("beta1", 0.95)
        self.lr = kwargs.get("lr", 0.02)  # Muon typically uses higher LR
        self.ns_steps = kwargs.get("ns_steps", 5)  # Newton-Schulz iterations

    def _route(self, p, idx):
        """S1: Route square 2D tensors as 'matrix', else 'vector'."""
        return "matrix" if (p.dim() >= 2 and p.shape[0] == p.shape[1] and p.shape[0] > 1) else "vector"

    def _evolve(self, g, p, idx, route):
        if route == "matrix":
            if (idx, "m") not in self.state:
                self.state[(idx, "m")] = torch.zeros_like(p)
            m = self.state[(idx, "m")]
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            # S2: Newton-Schulz spectral orthogonalization
            # Approximates polar decomposition: UV^T where U,S,V = SVD(M)
            update = self._newton_schulz(m, self.ns_steps)
            self.state[(idx, "update")] = update
        else:
            # Vector params: fall back to momentum
            if (idx, "m") not in self.state:
                self.state[(idx, "m")] = torch.zeros_like(p)
            m = self.state[(idx, "m")]
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            self.state[(idx, "update")] = m

    def _newton_schulz(self, M, steps):
        """
        Newton-Schulz iteration for spectral normalization.
        Approximates (MM^T)^{-1/2} M = U V^T (polar form).
        ||M||_2 = ||UV^T||_2 = 1 after convergence.
        """
        # Normalize so largest singular value ≈ 1
        norm = M.norm() / (M.shape[0] ** 0.5) + 1e-7
        X = M / norm
        for _ in range(steps):
            # Newton-Schulz: X_{k+1} = 0.5 * X_k * (3I - X_k^T @ X_k)
            XtX = X.T @ X
            X = 0.5 * (3 * X - X @ XtX)
            # Stability: clamp spectral radius
            sq = X.norm() / (X.shape[0] ** 0.5)
            if sq > 2.0:
                X = X / sq
        return X * norm

    axes = {
        "I": "R^{m×n} (matrix) / R^d (vector)",
        "II": "M_t (matrix momentum)",
        "III (LMO)": "spectral (polar), Phi(M) = UV^T",
        "III (Precond)": "H_t = MM^T, orthogonalization",
        "IV": "LR + matrix routing",
    }


# ── Evaluation helpers ─────────────────────────────────────────────

class MLP(torch.nn.Module):
    """Simple 2-layer MLP for benchmarking.
    Hidden dim = d_in for square weight matrices (Muon compatibility)."""

    def __init__(self, d_in=32, hidden=None):
        super().__init__()
        hidden = hidden or d_in
        self.fc1 = torch.nn.Linear(d_in, hidden, bias=False)
        self.fc2 = torch.nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def train_with_optimizer(model, optim_cls, X, y, steps=500, batch_size=256,
                         lr=1e-3, wd=0.0, **opt_kwargs):
    """
    Train model with given optimizer. Returns metrics dict.
    """
    model_copy = MLP(model.fc1.in_features, model.fc1.out_features)
    # Copy weights
    model_copy.load_state_dict(model.state_dict())

    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = optim_cls(model_copy.parameters(), lr=lr, weight_decay=wd, **opt_kwargs)

    n = X.shape[0]
    losses = []
    step_times = []

    for step in range(steps):
        opt.zero_grad()
        idx = torch.randperm(n)[:batch_size]
        logits = model_copy(X[idx])
        loss = loss_fn(logits, y[idx])
        loss.backward()
        t0 = time.perf_counter()
        opt.step()
        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())

    return {
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "losses": losses,
        "step_time_ms": sum(step_times) / len(step_times) * 1000,
        "state_mem_kb": opt.state_memory_bytes() / 1024,
    }


def taxonomy_table():
    """Print the meta-pipeline instantiation table."""
    optimizers = [SGDM, AdamW, Lion, Muon]
    print(f"\n{'Optimizer':<8} {'Family':<4} {'Active Stages':<25} {'LMO Geometry':<25}")
    print("-" * 70)
    for cls in optimizers:
        print(f"{cls.name:<8} {cls.family:<4} {', '.join(cls.active_stages):<25} {cls.lmo_geometry:<25}")
