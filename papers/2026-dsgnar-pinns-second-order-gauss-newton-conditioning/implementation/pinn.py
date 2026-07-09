"""PINN fixture for DSGNAR check C5.

Solves the 1-D Poisson problem  -u''(x) = f(x)  on [0, 1] with Dirichlet
u(0) = u(1) = 0, exact solution u(x) = sin(pi x), source f(x) = pi^2 sin(pi x).

The solution ansatz is a small SIREN MLP  u_theta(x) = W2 sin(w0 (W1 x + b1)) + b2.
The residual vector is

    r = [ -u''(x_i) - f(x_i)  for collocation x_i ]   (PDE)
        [ w_bc * u(0),  w_bc * u(1) ]                 (BC, soft)

Its Jacobian w.r.t. the flattened parameters is computed *exactly* with
torch.func.jacfwd (no finite differences).  numpy views of r and J are handed to
the same Gauss-Newton primitives in model.py, so the PINN is solved by the
identical DSGNAR optimiser used for the synthetic checks.
"""

import numpy as np
import torch
from torch.func import jacfwd

torch.set_default_dtype(torch.float64)          # "double precision rarely diverges" (§4.2)


def _shapes(H):
    return [(H, 1), (H,), (1, H), (1,)]          # W1, b1, W2, b2


def _unpack(flat, shapes):
    out, i = [], 0
    for shp in shapes:
        n = int(np.prod(shp))
        out.append(flat[i:i + n].reshape(shp))
        i += n
    return out


class PINN:
    """A 1-D SIREN Poisson PINN with exact Jacobian (torch.func.jacfwd)."""

    def __init__(self, H=32, n_coll=64, w_bc=10.0, omega0=8.0, seed=0):
        rng = np.random.default_rng(seed)
        self.shapes = _shapes(H)
        self.omega0 = omega0
        self.w_bc = w_bc
        self.x_coll = torch.linspace(0.0, 1.0, n_coll)
        self.f_src = (np.pi ** 2) * torch.sin(np.pi * self.x_coll)
        W1 = rng.uniform(-1.0, 1.0, self.shapes[0]) * 0.6
        b1 = rng.uniform(0.0, 2.0 * np.pi, self.shapes[1])
        W2 = rng.uniform(-1.0, 1.0, self.shapes[2]) / np.sqrt(H)
        b2 = rng.uniform(-0.1, 0.1, self.shapes[3])
        self.theta0 = torch.cat([torch.tensor(t).reshape(-1) for t in [W1, b1, W2, b2]]).numpy()
        self.n_params = self.theta0.size

    # ---- core torch residual (all derivatives via nested functorch jacfwd,
    #      so the param-Jacobian composes cleanly with the x-derivatives) ----
    def _u(self, params, x):
        W1, b1, W2, b2 = params
        h = torch.sin(self.omega0 * (x.reshape(-1, 1) @ W1.T + b1))
        return (h @ W2.T + b2).reshape(-1)

    def _u_scalar(self, params, xi):
        """u at a single scalar coordinate (kept scalar for clean vmap+jacfwd)."""
        W1, b1, W2, b2 = params
        h = torch.sin(self.omega0 * (xi.reshape(1) @ W1.T + b1))   # (1, H)
        return (h @ W2.T + b2).reshape(())                          # scalar

    def _residual_torch(self, flat):
        params = _unpack(flat, self.shapes)
        x = self.x_coll
        u_of = lambda xi: self._u_scalar(params, xi)
        du = torch.func.jacfwd(u_of)
        ddu = torch.func.jacfwd(du)
        u_xx = torch.func.vmap(ddu)(x)                       # (n,) pointwise 2nd deriv
        pde = -u_xx - self.f_src
        x0 = torch.tensor([0.0]); x1 = torch.tensor([1.0])
        bc = torch.stack([self._u(params, x0)[0], self._u(params, x1)[0]]) * self.w_bc
        return torch.cat([pde, bc])

    # ---- numpy interfaces for model.py primitives ----
    def residual_fn(self, theta_np):
        # NOTE: no torch.no_grad() -- the x-derivative graph must be live.
        return self._residual_torch(torch.as_tensor(theta_np, dtype=torch.float64)).detach().numpy()

    def jac_fn(self, theta_np):
        flat = torch.as_tensor(theta_np, dtype=torch.float64)
        J = jacfwd(lambda p: self._residual_torch(p))(flat)
        return J.detach().numpy()

    def loss_torch(self, flat):
        r = self._residual_torch(flat)
        return 0.5 * (r @ r)

    def predict(self, theta_np, x_np):
        with torch.no_grad():
            params = _unpack(torch.as_tensor(theta_np, dtype=torch.float64), self.shapes)
            return self._u(params, torch.as_tensor(x_np, dtype=torch.float64)).numpy()

    @staticmethod
    def exact(x_np):
        return np.sin(np.pi * x_np)


def adam_baseline(pinn, steps=8000, lr=1e-3):
    """First-order Adam on the same PINN objective -- the 'optimiser bottleneck' foil."""
    theta = torch.tensor(pinn.theta0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = pinn.loss_torch(theta)
        loss.backward()
        opt.step()
    return theta.detach().numpy()


def rel_l2(pinn, theta_np, x_grid=None):
    if x_grid is None:
        x_grid = np.linspace(0.0, 1.0, 501)
    pred = pinn.predict(theta_np, x_grid)
    ex = pinn.exact(x_grid)
    return float(np.linalg.norm(pred - ex) / np.linalg.norm(ex))
