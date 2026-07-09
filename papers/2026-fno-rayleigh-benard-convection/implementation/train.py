"""Verification harness for "Fourier Neural Operators for Rayleigh-Benard
Convection" (John et al., 2026, arXiv:2607.02088).

Runs five deterministic checks on a closed-form 2D periodic advection-diffusion
PDE (Fourier-diagonal exact one-step map). No RBC turbulence, no Dedalus, no
GPU -- the load-bearing mechanism (predict increments not solutions; spectral
conv = Fourier multiplier; mesh invariance; training-resolution bound;
rollout-error accumulation) is fully exposed by the linear operator.

    uv run --with numpy --with scipy --with torch python train.py
"""

import sys
import numpy as np
import torch

import data
import model

torch.manual_seed(0)
np.random.seed(0)

# ---- shared config (small + fast, converges because the map is simple) ------ #
WIDTH = 10
N_LAYERS = 2
MODES = 8
N_STEPS = 200
LR = 2e-3
SEED = 0
MAX_IDX = 6      # band-limited field content |kx|,|ky|<=6
DECAY = 0.3

PASSED = []


def banner(c):
    print("\n" + "=" * 78)
    print(c)
    print("=" * 78)


def result(name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    PASSED.append(ok)


def fresh_state(max_idx, high_tail=None, n_modes=None, seed_off=0):
    rng = np.random.default_rng(1000 + seed_off)
    return data.make_state(rng, max_idx, DECAY, high_tail=high_tail, n_modes=n_modes)


def eval_grid(inc_model, N, dt, n=40, seed_off=0, high_tail=None, max_idx=MAX_IDX):
    """Held-out reconstructed-solution relative error at grid N for a trained
    increment model (no retraining)."""
    rng = np.random.default_rng(5000 + seed_off + N)
    errs = []
    for _ in range(n):
        st = fresh_state(max_idx, high_tail=high_tail, seed_off=rng.integers(0, 1 << 30))
        u0 = data.sample_state(st, N)
        u1 = data.step_state(u0, N, dt)
        u0t = torch.from_numpy(u0[None])
        with torch.no_grad():
            pred = u0t + dt * inc_model.predict_increment(u0t)
        errs.append(model.relative_l2(pred, torch.from_numpy(u1[None])))
    return float(np.mean(errs))


# --------------------------------------------------------------------------- #
def check_C1():
    banner("C1 -- increment objective is load-bearing (paper Table 1 hinge)")
    # (a) identity baseline scales linearly with dt (the small-dt regime that
    #     tempts the solution objective toward 'predict U0').
    dts = [1e-3, 3e-3, 1e-2, 3e-2]
    id_errs = []
    for dt in dts:
        U0, U1, _ = data.gen_pairs(32, dt, 80, MAX_IDX, DECAY, seed=1)
        id_errs.append(data.identity_relative_error(U0[60:], U1[60:]))
    id_slope = np.polyfit(np.log(dts), np.log(id_errs), 1)[0]
    print(f"  identity rel-err vs dt: {[f'{e:.3e}' for e in id_errs]} (dt {[f'{d:.0e}' for d in dts]})")
    print(f"  log-log slope identity~dt^{id_slope:.2f} (expect ~1)")
    id_linear = id_slope > 0.85

    # (b) equal-budget horse race at small dt: increment beats identity, solution
    #     fails to beat identity (paper: solution is 10x WORSE than identity).
    dt = 1e-2
    U0, U1, dU = data.gen_pairs(32, dt, 80, MAX_IDX, DECAY, seed=1)
    inc = model.IncrementModel(2, width=WIDTH, n_layers=N_LAYERS, modes=MODES)
    ei = model.fit(inc, U0[:60], dU[:60], U0[60:], U1[60:], dt=dt,
                   n_steps=N_STEPS, lr=LR, seed=SEED)
    sol = model.SolutionModel(2, width=WIDTH, n_layers=N_LAYERS, modes=MODES)
    es = model.fit(sol, U0[:60], U1[:60], U0[60:], U1[60:], dt=dt,
                   n_steps=N_STEPS, lr=LR, seed=SEED)
    id_e = data.identity_relative_error(U0[60:], U1[60:])
    print(f"  increment={ei:.3e}  identity={id_e:.3e}  solution={es:.3e}")
    print(f"  solution/identity={es/id_e:.1f}x (paper Table1 ~10x), identity/increment={id_e/ei:.1f}x")
    hinge = ei < id_e < es
    result("C1a identity~dt (small-dt regime)", id_linear,
           f"slope {id_slope:.2f} ~ 1")
    result("C1b increment < identity < solution (Table-1 hinge)", hinge,
           f"inc {ei:.2e} < id {id_e:.2e} < sol {es:.2e}")
    return inc  # reuse a trained increment model downstream


def _disable_W(m):
    """Freeze+zero the local 1x1 shortcut so only the Fourier layer is active.
    (Zeroing alone does NOT disable it -- Adam un-zeroes a parameter via its
    nonzero gradient. requires_grad_(False) is what actually freezes it.)"""
    with torch.no_grad():
        for blk in m.net.layers:
            blk.W.weight.zero_(); blk.W.weight.requires_grad_(False)
            blk.W.bias.zero_(); blk.W.bias.requires_grad_(False)


def _disable_spec(m):
    """Freeze+zero the spectral conv so only the per-pixel local path is active."""
    with torch.no_grad():
        for blk in m.net.layers:
            blk.spec.weight.zero_(); blk.spec.weight.requires_grad_(False)


def check_C2():
    banner("C2 -- spectral conv is the load-bearing Fourier-multiplier piece")
    # Single-channel PURE ADVECTION d_t U = -vel.grad U: a strictly spatial
    # operator a per-pixel (1x1) map cannot compute -- so it cleanly isolates
    # whether the Fourier layer (cross-mode spectral mixing) is doing the work.
    dt = 0.05
    N = 32
    vel = (2.0, 1.2)   # ~0.6 rad max phase shift -- well past magnitude-only fit
    adv_max = 6
    U0, U1, dU = data.gen_single_channel_pairs(
        N, dt, 80, adv_max, 0.15, kappa=0.0, vel=vel, seed=2)
    id_e = data.identity_relative_error(U0[60:], U1[60:])

    spec_only = model.IncrementModel(1, width=8, n_layers=2, modes=10)
    _disable_W(spec_only)
    e_spec = model.fit(spec_only, U0[:60], dU[:60], U0[60:], U1[60:], dt=dt,
                       n_steps=N_STEPS, lr=3e-3, seed=SEED)

    w_only = model.IncrementModel(1, width=8, n_layers=2, modes=10)
    _disable_spec(w_only)
    e_w = model.fit(w_only, U0[:60], dU[:60], U0[60:], U1[60:], dt=dt,
                    n_steps=N_STEPS, lr=3e-3, seed=SEED)
    # sanity: confirm the frozen path stayed at zero
    spec_frozen = all(float(b.spec.weight.abs().max()) < 1e-8
                      for b in w_only.net.layers)
    print(f"  identity={id_e:.3e}  spec-only(Fourier)={e_spec:.3e}  "
          f"W-only(local)={e_w:.3e}  (spec frozen at 0: {spec_frozen})")
    print(f"  spec-only beats identity by {id_e/e_spec:.2f}x; "
          f"W-only/identity={e_w/id_e:.2f} (per-pixel map cannot do spatial)")
    ok = e_spec < 0.75 * id_e and e_spec < e_w and spec_frozen
    result("C2 spectral conv (not the local path) captures the spatial operator",
           ok, f"spec {e_spec:.2e} < 0.75*id {0.75*id_e:.2e}; W-only {e_w:.2e}")


def check_C3(inc_model):
    banner("C3 -- zero-shot mesh invariance (train N=32, no retrain, no blowup)")
    dt = 1e-2
    Ns = [16, 24, 32, 48, 64]
    errs = {N: eval_grid(inc_model, N, dt, n=30, seed_off=N) for N in Ns}
    for N in Ns:
        print(f"  N_eval={N:3d}: rel-err={errs[N]:.3e}")
    ratios = [e / min(errs.values()) for e in errs.values()]
    print(f"  max/min error ratio across grids = {max(ratios):.2f} (mesh-invariant in order)")
    # no blowup at finer grids: error at 2x finer (64) not orders worse than train (32)
    no_blow = errs[64] < 5 * errs[32] and max(ratios) < 6
    result("C3 error same order across grids (no retrain, no blowup)", no_blow,
           f"max/min {max(ratios):.2f}, err(64)/err(32)={errs[64]/errs[32]:.2f}")


def check_C4():
    banner("C4 -- training-resolution bound (finer inference != better; Table 4)")
    dt = 1e-2
    N_tr = 24   # training grid: Nyquist index 12

    def train_at(high_tail):
        U0, U1, dU = data.gen_pairs(N_tr, dt, 70, MAX_IDX, DECAY, seed=3,
                                    high_tail=high_tail)
        m = model.IncrementModel(2, width=WIDTH, n_layers=N_LAYERS, modes=MODES)
        model.fit(m, U0[:54], dU[:54], U0[54:], U1[54:], dt=dt,
                  n_steps=N_STEPS, lr=LR, seed=SEED)
        return m

    # (a) band-limited content |k|<=6, well inside N_tr Nyquist (12): clean.
    m_band = train_at(high_tail=None)
    e_band_24 = eval_grid(m_band, 24, dt, n=30, seed_off=1, max_idx=MAX_IDX)
    e_band_48 = eval_grid(m_band, 48, dt, n=30, seed_off=2, max_idx=MAX_IDX)
    print(f"  band-limited: err@train(24)={e_band_24:.3e}  err@fine(48)={e_band_48:.3e}  "
          f"ratio {e_band_48/e_band_24:.2f} (mesh-invariant: ~1)")

    # (b) high-freq tail up to idx 20, far beyond N_tr Nyquist 12 -> aliases at
    #     training, unrecoverable; finer inference exposes it -> finer is WORSE.
    m_tail = train_at(high_tail=(12, 20, 6.0))
    e_tail_24 = eval_grid(m_tail, 24, dt, n=30, seed_off=3, high_tail=(12, 20, 6.0), max_idx=20)
    e_tail_48 = eval_grid(m_tail, 48, dt, n=30, seed_off=4, high_tail=(12, 20, 6.0), max_idx=20)
    print(f"  high-tail:    err@train(24)={e_tail_24:.3e}  err@fine(48)={e_tail_48:.3e}  "
          f"ratio {e_tail_48/e_tail_24:.2f} (finer NOT better)")

    ok_a = e_band_48 / e_band_24 < 2.5          # band-limited transfers cleanly
    ok_b = e_tail_48 / e_tail_24 > 1.0          # high-freq tail: finer does not help
    result("C4a band-limited field is mesh-invariant (fine ~ train)", ok_a,
           f"ratio {e_band_48/e_band_24:.2f}")
    result("C4b high-freq tail is training-resolution-bound (fine >= train)", ok_b,
           f"ratio {e_tail_48/e_tail_24:.2f} (Table-4: finer 4.5x worse)")


def check_C5(inc_model):
    banner("C5 -- autoregressive rollout error accumulates (Straat/Table-5 finding)")
    dt = 1e-2
    N = 32
    n = 30
    Ks = [1, 2, 5, 10, 20]
    errs = {K: [] for K in Ks}
    rng = np.random.default_rng(7000)
    for _ in range(n):
        st = data.make_state(rng, MAX_IDX, DECAY)
        # exact trajectory
        true = [data.sample_state(st, N)]
        for _ in range(max(Ks)):
            true.append(data.step_state(true[-1], N, dt))
        # model rollout
        u = true[0].copy()
        hat = [u]
        for k in range(1, max(Ks) + 1):
            ut = torch.from_numpy(hat[-1][None])
            with torch.no_grad():
                u = (ut + dt * inc_model.predict_increment(ut)).numpy()[0]
            hat.append(u)
        for K in Ks:
            errs[K].append(model.relative_l2(
                torch.from_numpy(hat[K][None]), torch.from_numpy(true[K][None])))
    mean = {K: float(np.mean(errs[K])) for K in Ks}
    for K in Ks:
        print(f"  rollout horizon K={K:2d} steps: rel-err={mean[K]:.3e}")
    monotone = all(mean[Ks[i]] <= mean[Ks[i + 1]] + 1e-9 for i in range(len(Ks) - 1))
    grows = mean[Ks[-1]] > 3 * mean[Ks[0]]
    result("C5 rollout error accumulates with horizon", monotone and grows,
           f"err K=1 {mean[Ks[0]]:.2e} -> K={Ks[-1]} {mean[Ks[-1]]:.2f} (monotone, grows)")


def main():
    print("FNO Rayleigh-Benard verification -- "
          f"width={WIDTH} layers={N_LAYERS} modes={MODES} steps={N_STEPS}")
    inc = check_C1()
    check_C2()
    check_C3(inc)
    check_C4()
    check_C5(inc)

    n_pass = sum(PASSED)
    n_all = len(PASSED)
    print("\n" + "#" * 78)
    print(f"RESULT: {n_pass}/{n_all} checks PASS")
    print("#" * 78)
    sys.exit(0 if n_pass == n_all else 1)


if __name__ == "__main__":
    main()
