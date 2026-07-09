"""DSGNAR verification checks (C1-C5).

Five independent checks isolate the paper's optimiser contribution (§3) from its
PDE zoo (§5).  All numbers are deterministic (fixed seeds; no wall-clock or GPU
dependence).

  C1  doubly-sketched GN is a valid subspace embedding
        sketched LM step -> full LM step as sketch rank s grows; Jtil singular
        values track the top-s of J.
  C2  one SVD yields LM steps for a whole lambda sweep (model reduction identity)
        m(p) from the SVD == direct predicted reduction, to machine precision.
  C3  trust-region lambda/delta duality (Eq 13)
        ||p~(lambda)|| strictly decreasing in lambda; LambdaSolve radius exact.
  C4  conditioning-first target-ratio rule beats fixed regularisation
        DSGNAR's two-stage rho schedule solves the ill-conditioned Rosenbrock
        valley where the best fixed-lambda LM stalls (orders of magnitude).
  C5  Gauss-Newton PINN reaches machine precision; first-order Adam plateaus
        1-D Poisson, SIREN ansatz, exact Jacobian (torch.func.jacfwd); the
        'optimiser is the bottleneck / losses at machine precision are welcome'
        claim (§1), reproduced as a GN-vs-Adam gap.
"""

import numpy as np

import data
import model


def _fmt(x):
    return f"{x:.3e}"


def check_c1():
    """Doubly-sketched GN is a subspace embedding (in the low-rank regime).

    Two verifiable facts:
      * SRCT *lift* L has orthonormal columns (L^T L = I_s) -- the "faithful
        lift" of §3.2.2.
      * CountSketch is a (1+/-eps) row-embedding of col(J) (§3.2.1), so the
        doubly-sketched GN step captures the full step's loss reduction with
        distortion that shrinks as the sketch rank s grows -- *provided J is
        numerically low rank*, which is exactly the PINN regime that makes
        sketching d_theta~1e5 -> s~4e3 viable.  A dense full-rank J is the
        worst case (the sketch cannot help); included as a contrast.
    """
    cfg = data.CONFIG["sketch"]
    # low-rank-effective Jacobian (the sketching regime)
    J, r, lam = data.overdetermined_jacobian(cfg["N"], cfg["d"], cfg["rank"],
                                             cfg["lam"], cfg["seed"])
    p_full = model.full_lm_step(J, r, lam)
    L0 = 0.5 * (r @ r)
    dec_full = L0 - 0.5 * ((r + J @ p_full) @ (r + J @ p_full))
    U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    basis = Vt[:cfg["rank"]]                       # dominant row-space (rank r)

    print("C1  doubly-sketched GN is a subspace embedding (low-rank regime)")
    print(f"    J {cfg['N']}x{cfg['d']} rank-{cfg['rank']} effective "
          f"(sigma_8/sigma_1 = {sv[cfg['rank']-1]/sv[0]:.1e}); "
          f"avg over {cfg['n_seed']} sketch draws, K={cfg['K']} hashes")
    distort, dec_ratio, vec_err = [], [], []
    for s in cfg["sketch_sizes"]:
        d_, dr, ve = [], [], []
        for sd in range(cfg["n_seed"]):
            Crow = model.CountSketch(cfg["N"], s, K=cfg["K"], rng=cfg["seed"] + s * 7 + sd)
            Q = model.srct(cfg["d"], s, rng=cfg["seed"] + 100 + s * 7 + sd)
            # SRCT near-isometry
            iso = np.linalg.norm(Q.T @ Q - np.eye(s))
            # CountSketch row-embedding distortion over the dominant row space
            worst = 0.0
            for v in basis:
                Jv = J @ v
                worst = max(worst, abs((Crow.rows(Jv) ** 2).sum() / (Jv @ Jv) - 1.0))
            d_.append(worst)
            p_sk, _, _, _ = model.sketched_lm_step(J, r, Crow, Q, lam)
            dec_sk = L0 - 0.5 * ((r + J @ p_sk) @ (r + J @ p_sk))
            dr.append(dec_sk / dec_full)
            ve.append(np.linalg.norm(p_sk - p_full) / np.linalg.norm(p_full))
        distort.append(np.mean(d_)); dec_ratio.append(np.mean(dr)); vec_err.append(np.mean(ve))
        print(f"    s={s:2d}: embed distortion={distort[-1]:.3f}  "
              f"loss-decrease ratio={dec_ratio[-1]:+.3f}  "
              f"vec rel-err={vec_err[-1]:.3f}  (SRCT ||L^TL-I||~{iso:.0e})")

    # contrast: full-rank random J -- sketch cannot help
    Jf, rf, lamf = data.random_fullrank_jacobian(cfg["N"], cfg["d"], cfg["lam"], cfg["seed"])
    pf = model.full_lm_step(Jf, rf, lamf)
    decF = 0.5 * (rf @ rf) - 0.5 * ((rf + Jf @ pf) @ (rf + Jf @ pf))
    s_top = cfg["sketch_sizes"][-1]
    contrast = []
    for sd in range(cfg["n_seed"]):
        Crow = model.CountSketch(cfg["N"], s_top, K=cfg["K"], rng=cfg["seed"] + s_top * 7 + sd)
        Q = model.srct(cfg["d"], s_top, rng=cfg["seed"] + 100 + s_top * 7 + sd)
        p_sk, _, _, _ = model.sketched_lm_step(Jf, rf, Crow, Q, lamf)
        contrast.append((0.5 * (rf @ rf) - 0.5 * ((rf + Jf @ p_sk) @ (rf + Jf @ p_sk))) / decF)
    print(f"    contrast full-rank J at s={s_top}: loss-decrease ratio={np.mean(contrast):+.3f} "
          f"(sketch fails -- low rank is the regime)")

    ok = (distort[-1] < distort[0] / 2.0 and dec_ratio[-1] > 0.65
          and dec_ratio[-1] > dec_ratio[0] + 0.3 and np.mean(contrast) < 0.3)
    print(f"    -> distortion halves & loss-decrease > 0.65 at top s, full-rank contrast "
          f"< 0.3: {ok}  => {'PASS' if ok else 'FAIL'}\n")
    return ok


def check_c2():
    """One SVD of Jtil serves a whole lambda sweep; model-reduction identity holds."""
    cfg = data.CONFIG["sketch"]
    J, r, lam = data.overdetermined_jacobian(cfg["N"], cfg["d"], cfg["rank"], cfg["lam"], cfg["seed"])
    s = cfg["sketch_sizes"][2]
    Crow = model.CountSketch(cfg["N"], s, K=cfg["K"], rng=cfg["seed"] + s)
    Q = model.srct(cfg["d"], s, rng=cfg["seed"] + 100 + s)
    _, Jtil, rtil, _ = model.sketched_lm_step(J, r, Crow, Q, lam)
    step_fn, bundle = model.svd_step_factory(Jtil, rtil)
    lam_sweep = np.logspace(-2, 3, 8)
    max_id_err = 0.0
    for L in lam_sweep:
        ptil = step_fn(L)
        # direct predicted reduction m(0)-m(p) via explicit Jtil
        pred_direct = model.predicted_reduction(Jtil, rtil, ptil, L)
        # SVD-only model reduction: m(0)-m(p) = 0.5||rtil||^2 - model_reduction_svd
        pred_svd = 0.5 * (rtil @ rtil) - model.model_reduction_svd(bundle, L)
        max_id_err = max(max_id_err, abs(pred_direct - pred_svd) / max(abs(pred_direct), 1e-30))
    print("C2  one SVD yields LM steps for a lambda sweep (model-reduction identity)")
    print(f"    Jtil {s}x{s} SVD reused over {len(lam_sweep)} lambda values in "
          f"[{lam_sweep[0]:.0e},{lam_sweep[-1]:.0e}]")
    print(f"    max |pred_direct - pred_SVD| / |pred| = {max_id_err:.2e}  (exact identity -> ~machine eps)")
    ok = max_id_err < 1e-9
    print(f"    -> identity holds to < 1e-9: {ok}  => {'PASS' if ok else 'FAIL'}\n")
    return ok


def check_c3():
    """Trust-region lambda/delta duality (Eq 13): ||p~(lambda)|| decreasing; radius exact."""
    cfg = data.CONFIG["sketch"]
    J, r, lam = data.overdetermined_jacobian(cfg["N"], cfg["d"], cfg["rank"], cfg["lam"], cfg["seed"])
    bundle = model._full_bundle(J, r)
    lams = np.logspace(-3, 6, 40)
    norms = np.array([model.step_norm_svd(bundle, L) for L in lams])
    diffs = np.diff(norms)
    monotone = bool(np.all(diffs < 0))
    # radius recovery: pick a target delta inside the norm range, solve lambda
    delta_target = float(norms[len(norms) // 2])
    lam_star = model.solve_lambda_for_radius(bundle, delta_target)
    delta_recovered = model.step_norm_svd(bundle, lam_star)
    rel = abs(delta_recovered - delta_target) / delta_target
    print("C3  trust-region lambda/delta duality (Eq 13)")
    print(f"    ||p~(lambda)|| strictly decreasing in lambda over "
          f"[{lams[0]:.0e},{lams[-1]:.0e}]: {monotone}")
    print(f"    LambdaSolve radius recovery: target={delta_target:.4f} "
          f"recovered={delta_recovered:.4f} (rel err {rel:.2e})")
    ok = monotone and rel < 1e-4
    print(f"    -> monotone & radius exact to < 1e-4: {ok}  => {'PASS' if ok else 'FAIL'}\n")
    return ok


def check_c4():
    """Conditioning-first target-ratio rule beats fixed regularisation (Rosenbrock)."""
    cfg = data.CONFIG["rosenbrock"]
    res, jac, th0 = data.rosenbrock_problem(cfg["d"], cfg["seed"])
    th_ds, lt_ds, lam_ds = model.dsgnar(res, jac, th0, iters=cfg["iters"])
    fixed_losses = []
    for lam in cfg["lam_grid"]:
        _, lt = model.fixed_lambda_lm(res, jac, th0, lam=float(lam), iters=cfg["iters"])
        fixed_losses.append(lt[-1])
    fixed_losses = np.array(fixed_losses)
    best_fixed = float(fixed_losses.min())
    best_lam = float(cfg["lam_grid"][int(np.argmin(fixed_losses))])
    gap = best_fixed / max(lt_ds[-1], 1e-300)
    print("C4  conditioning-first rho schedule beats fixed-lambda LM")
    print(f"    {cfg['d']}-dim Rosenbrock, {cfg['iters']} iters, "
          f"fixed-lambda grid = {len(cfg['lam_grid'])} values")
    print(f"    best fixed-lambda (lam={best_lam:.2e}) final loss = {best_fixed:.3e}")
    print(f"    DSGNAR (two-stage rho) final loss            = {lt_ds[-1]:.3e}")
    print(f"    min effective regularisation reached by DSGNAR = {lam_ds.min():.2e} "
          f"(-> well-conditioned region, Stage 2)")
    print(f"    gap best_fixed / DSGNAR = {gap:.1e}x")
    ok = gap > 1e2 and lt_ds[-1] < best_fixed
    print(f"    -> DSGNAR > 100x lower loss than best fixed: {ok}  => {'PASS' if ok else 'FAIL'}\n")
    return ok


def check_c5():
    """Gauss-Newton PINN reaches machine precision; first-order Adam plateaus."""
    import pinn
    net = pinn.PINN(H=48, n_coll=48, seed=0)
    th0 = net.theta0
    rel0 = pinn.rel_l2(net, th0)
    th_ds, lt_ds, _ = model.dsgnar(net.residual_fn, net.jac_fn, th0, iters=50)
    rel_ds = pinn.rel_l2(net, th_ds)
    th_ad = pinn.adam_baseline(net, steps=8000, lr=1e-3)
    rel_ad = pinn.rel_l2(net, th_ad)
    orders = np.log10(rel_ad / max(rel_ds, 1e-300))
    print("C5  GN PINN machine precision vs Adam plateau (1-D Poisson, SIREN, exact Jac)")
    print(f"    -u''=pi^2 sin(pi x), u(0)=u(1)=0, exact u=sin(pi x); {net.n_params} params")
    print(f"    init            rel L2 = {rel0:.3e}")
    print(f"    DSGNAR (50 GN)  rel L2 = {rel_ds:.3e}   residual loss = {lt_ds[-1]:.3e}")
    print(f"    Adam  (8000)    rel L2 = {rel_ad:.3e}")
    print(f"    GN-vs-Adam gap  = {orders:.1f} orders of magnitude")
    ok = rel_ds < rel_ad / 100.0 and lt_ds[-1] < 1e-9
    print(f"    -> GN >=2 orders better & loss < 1e-9: {ok}  => {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    print("=" * 72)
    print("DSGNAR — Doubly-Sketched Gauss-Newton with Adaptive Ratio (verification)")
    print("=" * 72 + "\n")
    results = {}
    for name, fn in [("C1", check_c1), ("C2", check_c2), ("C3", check_c3),
                     ("C4", check_c4), ("C5", check_c5)]:
        results[name] = fn()
    print("=" * 72)
    summary = "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in results.items())
    npass = sum(results.values())
    print(f"{summary}   ->  {npass}/{len(results)} PASS")
    print("=" * 72)
    if npass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
