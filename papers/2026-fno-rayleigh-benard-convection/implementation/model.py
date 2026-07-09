"""A minimal Fourier Neural Operator + the two learning objectives compared in
"Fourier Neural Operators for Rayleigh-Benard Convection" (John et al., 2026,
arXiv:2607.02088).

Architecture (paper Sec 3):  a ->[lift P]-> v ->[Fourier layers x L]-> v
->[project Q]-> u.  Each Fourier layer = spectral convolution on a fixed low-
frequency mode set (FFT -> truncate -> complex weight -> iFFT) PLUS a local
linear term W*v and a nonlinearity, for the high-frequency/local content the
truncated spectrum cannot see.

Two objectives (paper Sec 3.2):
  * solution  : predict U(t+dt) directly from U(t);
  * increment : predict dU = (U(t+dt)-U(t))/dt, recover U(t+dt) = U(t)+dt*model.
The increment objective is the paper's load-bearing finding: at small dt,
U(t+dt)~=U(t), so a solution-objective model is tempted toward the identity
predictor and the dynamics is a tiny fraction of its loss; dividing by dt makes
the dynamics the whole, well-conditioned signal.

Everything is grid-agnostic: the spectral weights are indexed by the lowest
|m| Fourier mode *indices* (continuous objects), so a model trained at grid N
applies unchanged at any other grid -- the mesh-invariance property (C3).
"""

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)


class SpectralConv2d(nn.Module):
    """Low-mode spectral convolution. Keeps the 4 low-frequency corners
    (|kx|<m, |ky|<m) of the 2D FFT, applies a complex (out,in,m,m) weight per
    corner via einsum, zero-pads, inverse-FFT. Grid-agnostic: works for any
    spatial size N1,N2 >= 2*m at inference.
    """

    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(out_ch, in_ch, modes, modes, dtype=torch.cfloat)
        )

    def _cmul(self, x_f, m):
        # x_f: (B, C, m, m) complex; apply weight over the channel axis
        return torch.einsum("oimn,bimn->bomn", self.weight, x_f[..., :m, :m])

    def forward(self, x):
        B, C, N1, N2 = x.shape
        m = self.modes
        assert N1 >= 2 * m and N2 >= 2 * m, "grid too small for mode count"
        xf = torch.fft.fft2(x)  # (B,C,N1,N2) complex, default norm
        out_f = torch.zeros(B, self.out_ch, N1, N2, dtype=torch.cfloat, device=x.device)
        # the four low-frequency corners share the same (m,m) weight block
        out_f[:, :, :m, :m] = self._cmul(xf[:, :, :m, :m], m)
        out_f[:, :, -m:, :m] = self._cmul(xf[:, :, -m:, :m], m)
        out_f[:, :, :m, -m:] = self._cmul(xf[:, :, :m, -m:], m)
        out_f[:, :, -m:, -m:] = self._cmul(xf[:, :, -m:, -m:], m)
        return torch.fft.ifft2(out_f).real  # (B,out,N1,N2) real


class FNOBlock(nn.Module):
    """One Fourier layer: spectral conv + local 1x1 (W) + activation."""

    def __init__(self, width, modes, activation=torch.tanh):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes)
        self.W = nn.Conv2d(width, width, 1)
        self.act = activation

    def forward(self, v):
        return self.act(self.spec(v) + self.W(v))


def _identity(x):
    return x


class FNO(nn.Module):
    """Lift -> L Fourier layers -> project. Operates on (B,C_in,N,N) grids."""

    def __init__(self, in_ch, out_ch, width=16, n_layers=2, modes=8,
                 activation=torch.tanh):
        super().__init__()
        self.act = _identity if activation is None else activation
        self.lift = nn.Linear(in_ch, width)
        self.layers = nn.ModuleList(
            [FNOBlock(width, modes, self.act) for _ in range(n_layers)]
        )
        self.proj1 = nn.Linear(width, width)
        self.proj2 = nn.Linear(width, out_ch)

    def forward(self, x):
        # x: (B,C,N,N) -> lift over channel dim -> (B,width,N,N)
        v = self.lift(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for blk in self.layers:
            v = blk(v)
        h = self.proj1(v.permute(0, 2, 3, 1))
        h = self.act(h)
        out = self.proj2(h).permute(0, 3, 1, 2)
        return out  # (B,out,N,N)


class IncrementModel(nn.Module):
    """Predicts dU; reconstructs U(t+dt) = U(t) + dt * dU at eval time."""

    def __init__(self, channels, **kw):
        super().__init__()
        self.net = FNO(channels, channels, **kw)

    def predict_increment(self, U0):
        return self.net(U0)

    def reconstruct(self, U0, dt):
        return U0 + dt * self.predict_increment(U0)


class SolutionModel(nn.Module):
    """Predicts U(t+dt) directly."""

    def __init__(self, channels, **kw):
        super().__init__()
        self.net = FNO(channels, channels, **kw)

    def reconstruct(self, U0, dt=None):
        return self.net(U0)


# --------------------------------------------------------------------------- #
# losses / metrics                                                            #
# --------------------------------------------------------------------------- #

def relative_l2(pred, target, eps=1e-12):
    """Mean over-batch relative L2: ||pred-target|| / ||target|| aggregated
    across channels and space, then averaged per sample."""
    flat_p = pred.reshape(pred.shape[0], -1)
    flat_t = target.reshape(target.shape[0], -1)
    num = torch.linalg.norm(flat_p - flat_t, dim=1)
    den = torch.linalg.norm(flat_t, dim=1) + eps
    return float((num / den).mean())


def _rel_l2_loss(pred, target):
    num = torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1) + 1e-12
    return (num / den).mean()


def fit(model, U0_tr, target_tr, U0_va, U1_va, dt, n_steps=400, lr=2e-3,
        seed=0):
    """Train a model (Increment or Solution) with Adam.

    Training loss (the paper distinguishes the two objectives here):
      * IncrementModel -> relative-L2 of net(U0) vs dU=(U1-U0)/dt  (the dynamics
        is the WHOLE signal, well conditioned even at tiny dt);
      * SolutionModel  -> relative-L2 of net(U0) vs U1  (the dynamics is a tiny
        O(dt) perturbation of the static U0 baseline).

    Evaluation metric is IDENTICAL for both: relative error of the *reconstructed
    solution* U0+dt*net(U0) (increment) or net(U0) (solution) vs the true U1.
    Returns that held-out error.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    U0_tr_t = torch.from_numpy(U0_tr)
    tgt_t = torch.from_numpy(target_tr)
    is_inc = isinstance(model, IncrementModel)
    for _ in range(n_steps):
        opt.zero_grad()
        pred = model.predict_increment(U0_tr_t) if is_inc else model.reconstruct(U0_tr_t)
        loss = _rel_l2_loss(pred, tgt_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        U0_va_t = torch.from_numpy(U0_va)
        pred_va = (U0_va_t + dt * model.predict_increment(U0_va_t)) if is_inc \
            else model.reconstruct(U0_va_t)
        err = relative_l2(pred_va, torch.from_numpy(U1_va))
    return err
