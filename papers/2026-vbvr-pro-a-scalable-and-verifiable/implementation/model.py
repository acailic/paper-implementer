"""MiniVBVR — tiny conditional flow-matching policy + CPS / SDE / ODE samplers.

> Paper: *VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning*
> Junxiang Xu et al., 2026. arXiv:2608.26105 — https://arxiv.org/abs/2608.26105

The paper fine-tunes foundation video/image/interleaved generators; the
re-implementable core is the *denoising-as-decision-process* view with the CPS
exploration sampler (its Eq. 1). We build a miniature conditional flow-matching
model (x0-prediction, linear schedule sigma_t = t) over whole frame sequences,
and implement:

  Eq 1 (CPS):   x_{t-dt} = (1-s)x0_hat + s*[cos(eta*pi/2)*x1_hat + sin(eta*pi/2)*eps]

The bracket rotates the PREDICTED noise x1_hat toward fresh noise eps; because
cos^2+sin^2=1 the injected noise magnitude stays exactly on schedule for every
eta (unit variance in expectation). eta=0 recovers the ODE.

Flow-SDE contrast: adds sqrt(2*eta*sigma*sigma')*eps ON TOP of the drift, so
effective noise EXCEEDS the schedule -> visual artifacts (paper App. B.4).

GRPO: each Euler step's outcome x_{t-dt} is a Gaussian "action" with mean
mu_theta = (1-s)x0_hat + s*cos(eta*pi/2)*x1_hat  and std t'*sin(eta*pi/2);
terminal group-relative advantages A=(r-mean)/std train the policy through the
trajectory log-prob (Flow-GRPO family; clipped ratio, replayed rollouts).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from data import H, W, N_FRAMES

FLAT = N_FRAMES * 3 * H * W          # 2592
SCENE = 3 * H * W                    # 432
INSTR = 15
TEMB = 32


class MiniFlowNet(nn.Module):
    """Conditional denoiser: (noisy frames, scene, instruction, t) -> x0_hat."""

    def __init__(self, hidden=768):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(1, TEMB), nn.GELU(), nn.Linear(TEMB, TEMB))
        self.net = nn.Sequential(
            nn.Linear(FLAT + SCENE + INSTR + TEMB, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, FLAT),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x_t, scene, instr, t):
        # x_t: (B,FLAT); scene: (B,SCENE); instr: (B,INSTR); t: (B,)
        h = torch.cat([x_t, scene, instr, self.temb(t.unsqueeze(-1))], dim=-1)
        return self.net(h)


def _x1_hat(x_t, x0_hat, t):
    """Noise component implied by the schedule x_t = (1-t)x0 + t*x1."""
    if x_t.dim() == 2 and t.dim() == 1:
        t = t.unsqueeze(1)
    return (x_t - (1.0 - t) * x0_hat) / t.clamp_min(1e-4)


def rollout(model, scene, instr, sampler="cps", eta=0.7, steps=10, gen=None,
            frames_only=False):
    """Sample a full denoising trajectory. Returns final frames (B,FLAT) and,
    unless frames_only, per-step replay records for the GRPO log-prob."""
    B = scene.shape[0]
    dev = scene.device
    x = torch.randn(B, FLAT, generator=gen, device=dev)
    records = []
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for k in range(steps):
        t, t_next = ts[k], ts[k + 1]
        tb = t.expand(B)
        x0_hat = model(x, scene, instr, tb)
        x1_hat = _x1_hat(x, x0_hat, tb)
        if sampler == "ode":
            x_next = (1 - t_next) * x0_hat + t_next * x1_hat
        elif sampler == "cps":  # paper Eq 1
            eps = torch.randn(B, FLAT, generator=gen, device=dev)
            c, s = math.cos(eta * math.pi / 2.0), math.sin(eta * math.pi / 2.0)
            x_next = (1 - t_next) * x0_hat + t_next * (c * x1_hat + s * eps)
            if not frames_only:
                # per-sample records: replay later recomputes logp under theta
                logp = _step_logp(x_next, x0_hat, x1_hat, t_next, eta)
                for g in range(B):
                    records.append({"x_in": x[g:g + 1].detach(),
                                    "x_next": x_next[g:g + 1].detach(),
                                    "t": float(t), "t_next": float(t_next),
                                    "eta": eta,
                                    "logp_old": logp[g].detach()})
        elif sampler == "sde":  # Flow-SDE-style: extra noise on top of drift
            eps = torch.randn(B, FLAT, generator=gen, device=dev)
            drift = (1 - t_next) * x0_hat + t_next * x1_hat
            extra = math.sqrt(2.0 * eta * t.item() * (t - t_next).item()) * eps
            x_next = drift + extra
        else:
            raise ValueError(sampler)
        x = x_next
    # final frame = last x0 prediction cleaned of residual noise schedule
    x0_final = model(x, scene, instr, torch.zeros(B, device=dev))
    return x0_final.clamp(0, 1), records


def _step_logp(x_next, x0_hat, x1_hat, t_next, eta):
    """log N(x_next; mu_theta, (t'*sin)^2 I) up to theta-free constants,
    where mu_theta = (1-t')x0_hat + t'cos(theta)x1_hat and by construction
    x_next - mu_theta = t' sin(eta*pi/2) * eps."""
    if not torch.is_tensor(t_next):
        t_next = torch.tensor(t_next, device=x_next.device)
    mu = (1 - t_next) * x0_hat + t_next * math.cos(eta * math.pi / 2.0) * x1_hat
    s = (t_next * math.sin(eta * math.pi / 2.0)).clamp_min(1e-6)
    return -((x_next - mu) ** 2).sum(-1) / (2 * s * s)


def replay_logp(model, records, scene, instr):
    """Recompute trajectory log-prob under current theta (DPPO/Flow-GRPO replay).
    records: one sample's list of per-step dicts; scene/instr: (1,·)."""
    total = 0.0
    for rec in records:
        t = torch.full((scene.shape[0],), rec["t"], device=scene.device)
        x0_hat = model(rec["x_in"], scene.expand(rec["x_in"].shape[0], -1),
                       instr.expand(rec["x_in"].shape[0], -1), t)
        x1_hat = _x1_hat(rec["x_in"], x0_hat, t)
        total = total + _step_logp(rec["x_next"], x0_hat, x1_hat,
                                   rec["t_next"], rec["eta"])
    return total


def grpo_step(model, opt, rollouts, clip=0.2):
    """One GRPO update from a batch of rollouts (batched replay over steps).
    rollouts: list of dicts with keys records, reward, group_id, scene, instr.
    Advantage is group-relative: A = (r - mean_g)/std_g (std=0 -> A=0, no
    gradient — exactly why CPS-style decision diversity is needed)."""
    groups = {}
    for r in rollouts:
        groups.setdefault(r["group_id"], []).append(r["reward"])
    for r in rollouts:
        g = groups[r["group_id"]]
        mu = sum(g) / len(g)
        sd = (sum((x - mu) ** 2 for x in g) / len(g)) ** 0.5
        r["adv"] = 0.0 if sd < 1e-4 else (r["reward"] - mu) / sd
    # batched replay: all rollouts share the step schedule, so step k can be
    # recomputed for the whole batch in one forward
    scene = torch.stack([r["scene"] for r in rollouts])
    instr = torch.stack([r["instr"] for r in rollouts])
    K = len(rollouts[0]["records"])
    logps = []
    for k in range(K):
        rec0 = rollouts[0]["records"][k]
        x_in = torch.cat([r["records"][k]["x_in"] for r in rollouts])
        x_next = torch.cat([r["records"][k]["x_next"] for r in rollouts])
        t = torch.full((len(rollouts),), rec0["t"], device=scene.device)
        x0_hat = model(x_in, scene, instr, t)
        x1_hat = _x1_hat(x_in, x0_hat, t)
        logps.append(_step_logp(x_next, x0_hat, x1_hat, rec0["t_next"], rec0["eta"]))
    logp = torch.stack(logps, dim=1).sum(1)
    old = torch.tensor([rollout_logp_old(r["records"]) for r in rollouts],
                       device=logp.device)
    adv = torch.tensor([r["adv"] for r in rollouts], device=logp.device)
    ratio = torch.exp(logp - old)
    loss = -(torch.min(ratio * adv,
                       torch.clamp(ratio, 1 - clip, 1 + clip) * adv)).mean()
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item()), float(adv.abs().mean().item())


def rollout_logp_old(records):
    return float(sum(r["logp_old"].item() for r in records))


if __name__ == "__main__":  # Eq-1 sampler property selftests
    torch.manual_seed(0)
    # CPS bracket keeps injected-noise variance exactly on schedule:
    # E||cos(a)x1 + sin(a)eps||^2/D = cos^2+sin^2 = 1  (x1, eps ~ N(0,I))
    for eta in (0.3, 0.7, 0.9):
        a = eta * math.pi / 2.0
        c, s = math.cos(a), math.sin(a)
        x1 = torch.randn(64, 4096)
        eps = torch.randn(64, 4096)
        bracket = c * x1 + s * eps
        m2 = bracket.pow(2).mean().item()
        assert abs(m2 - 1.0) < 0.05, (eta, m2)
        # SDE-style additive term sqrt(2*eta*sigma*dsigma)*eps EXCEEDS it by
        # exactly the added variance 2*eta*sigma*dsigma
        sig, dsig = 0.6, 0.1
        extra = math.sqrt(2 * eta * sig * dsig) * torch.randn(64, 4096)
        m2_sde = (bracket + extra).pow(2).mean().item()
        assert m2_sde > m2 + 0.01 and abs(m2_sde - (1 + 2 * eta * sig * dsig)) < 0.05, \
            (eta, m2_sde)
        print(f"eta={eta}: CPS noise magnitude {m2:.3f} (on schedule) | "
              f"SDE style {m2_sde:.3f} (exceeds)")
    # rollout smoke test on a tiny net
    from data import make_batch
    model = MiniFlowNet(hidden=256)
    _, scenes, _, instr = make_batch(4, seed=0)
    scene_flat = scenes.reshape(4, -1)
    for sampler in ("ode", "cps", "sde"):
        fx, recs = rollout(model, scene_flat, instr, sampler, 0.7, 10)
        assert fx.shape == (4, FLAT) and 0.0 <= float(fx.min()) <= float(fx.max()) <= 1.0
        n_rec = 0 if sampler != "cps" else 10
        assert len(recs) == n_rec
    print("model.py selftests OK")
