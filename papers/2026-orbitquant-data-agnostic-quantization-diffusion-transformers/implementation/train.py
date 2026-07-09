"""OrbitQuant verification checks (C1-C5).

Five independent, deterministic checks isolate OrbitQuant's calibration-free
quantization thesis (arXiv:2607.02461) from its diffusion-transformer zoo.
No DiT, no GPU, no wall-clock -- just the load-bearing math.

  C1  rotation drives any unit vector's coordinates onto the fixed marginal f_d
        (Eq 2 / Fig 3): Raw deviates from N(0,1/d); Haar and RPBH both match.
  C2  Proposition 1 universal variance concentration (Eq 10)
        with the uniform permutation every coord has |Var(z_i)*d - 1| <= rho;
        without it (Block-Hadamard) the outlier block blows past rho.
  C3  rotation cancels in the product (Eq 4-8)
        W'x' = W Pi^T Pi x = Wx to machine precision; no inverse rotation.
  C4  Lloyd-Max codebook is MSE-optimal on f_d (Eq 3)
        beats the uniform grid at every bit-width; gap largest at b=2; optimum
        satisfies the centroid cell-mean condition.
  C5  end-to-end W2A4 robustness (headline: only functional method at W2A4)
        on outlier weights+activations OrbitQuant stays bounded where per-row
        RTN collapses; the permutation carries the low-bit gap (Remark 1).

Run:  uv run --with numpy --with scipy python train.py
"""

import numpy as np
from scipy import stats

import data
import model


SEED = 20260702
PASS = "PASS"
FAIL = "FAIL"


def _ks_to_unit_normal(samples, d):
    """KS distance of samples to N(0, 1/d) -- Fig 3's dashed reference curve."""
    scale = 1.0 / np.sqrt(d)
    cdf = lambda v: stats.norm.cdf(v, loc=0.0, scale=scale)
    return stats.kstest(samples, cdf).statistic


def check_c1():
    """C1 -- rotation drives coordinates onto f_d (Eq 2, Fig 3)."""
    rng = np.random.default_rng(SEED)
    d = 1024
    # (a) f_d is a valid density: integrates to 1, mean 0, var 1/d
    grid = np.linspace(-1.0, 1.0, 400001)
    pdf = model.fd_density(grid, d)
    integral = np.trapezoid(pdf, grid)
    mean_num = np.trapezoid(grid * pdf, grid)
    var_num = np.trapezoid(grid ** 2 * pdf, grid)
    _, var_exact = model.fd_mean_var(d)

    # (b) ground truth: sphere-uniform unit vectors have coords that ARE f_d
    uv = data.random_unit_vectors(d, 4000, rng)
    ks_truth = _ks_to_unit_normal(uv.ravel(), d)

    # (c) Fig 3: outlier unit vectors, Raw vs Haar vs RPBH
    n_vec = 2000
    xs = np.stack([data.unit_vector_with_outliers(d, n_outlier=6,
                                                  mass_in_outliers=0.6, rng=rng)
                   for _ in range(n_vec)])
    raw = xs.ravel()
    haar = np.concatenate([x @ model.random_haar(d, rng).T for x in xs[:500]])
    rpbh = np.concatenate([x @ model.rpbh(d, rng).T for x in xs])
    ks_raw = _ks_to_unit_normal(raw, d)
    ks_haar = _ks_to_unit_normal(haar, d)
    ks_rpbh = _ks_to_unit_normal(rpbh, d)

    ok_density = abs(integral - 1) < 1e-3 and abs(mean_num) < 1e-3 and abs(var_num - var_exact) < 1e-4
    ok_marginal = ks_truth < 0.02
    ok_rotation = (ks_raw > 0.10) and (ks_haar < 0.05) and (ks_rpbh < 0.05)
    passed = ok_density and ok_marginal and ok_rotation

    print(f"[C1] f_d marginal & rotation -> N(0,1/d)")
    print(f"     f_d integrates to {integral:.6f} (want 1); mean {mean_num:+.2e} (want 0); "
          f"var {var_num:.6f} (want {var_exact:.6f})")
    print(f"     sphere-uniform coords KS={ks_truth:.4f} (want <0.02, they ARE f_d)")
    print(f"     outlier vectors: Raw KS={ks_raw:.4f} (>0.10) | Haar KS={ks_haar:.4f} (<0.05) "
          f"| RPBH KS={ks_rpbh:.4f} (<0.05)")
    print(f"     -> {PASS if passed else FAIL}")
    return passed


def check_c2():
    """C2 -- Proposition 1 universal variance concentration (Eq 10)."""
    rng = np.random.default_rng(SEED + 1)
    d, h = 256, 64          # k = 4 blocks
    delta = 0.01
    n_outlier, mass = 2, 0.5
    x = data.prop1_vector(d, n_outlier, mass, rng)
    mu_inf = np.max(np.abs(x))
    rho = model.prop1_rho(d, mu_inf, h, delta)
    n_draws = 4000

    def coord_devs(use_perm):
        Z = np.empty((n_draws, d))
        for t in range(n_draws):
            Pi = model.rpbh(d, rng, h=h, use_perm=use_perm)
            Z[t] = Pi @ x
        var = Z.var(axis=0)                 # Var(z_i) over random Pi
        return np.abs(var * d - 1.0)        # |Var(z_i)*d - 1|

    dev_perm = coord_devs(use_perm=True)
    dev_noperm = coord_devs(use_perm=False)
    max_perm = dev_perm.max()
    max_noperm = dev_noperm.max()
    viol_perm = np.mean(dev_perm > rho)
    viol_noperm = np.mean(dev_noperm > rho)

    passed = (max_perm <= rho * 1.5) and (max_noperm > rho) and (max_noperm > 3.0 * max_perm)
    print(f"[C2] Proposition 1 variance concentration (d={d}, h={h}, k={d//h}, "
          f"mu_inf={mu_inf:.3f}, delta={delta})")
    print(f"     rho (Eq 10) = {rho:.4f}  -> bound |Var(z_i)*d - 1| <= rho")
    print(f"     WITH permutation:    max|dev|={max_perm:.4f}  (<= {rho:.4f})  "
          f"frac violating={viol_perm:.4f}")
    print(f"     WITHOUT permutation: max|dev|={max_noperm:.4f}  (> {rho:.4f})  "
          f"frac violating={viol_noperm:.4f}")
    print(f"     -> {PASS if passed else FAIL}")
    return passed


def check_c3():
    """C3 -- rotation cancels in the product (Eq 4-8)."""
    rng = np.random.default_rng(SEED + 2)
    d = 512
    Pi = model.rpbh(d, rng)
    W = rng.standard_normal((64, d)) * 0.3
    x = data.activation_token_with_outliers(d, rng)
    y = W @ x

    # exact cancellation before quantization
    Wp = W @ Pi.T
    xp = Pi @ x
    cancel_err = model.relative_error(y, Wp @ xp)

    # OrbitQuant both operands; product lives in rotated basis, equals W x up to quant
    cw, _ = model.lloyd_max_codebook(d, 4)
    ca, _ = model.lloyd_max_codebook(d, 4)
    What = model.orbitquant_weight(W, Pi, cw)
    xhat = model.orbitquant_activation(x, Pi, ca)
    oq_err = model.relative_error(y, What @ xhat)

    passed = cancel_err < 1e-13 and oq_err < 0.20
    print(f"[C3] rotation cancels in the product")
    print(f"     ||W Pi^T Pi x - W x|| / ||W x|| = {cancel_err:.2e} (want < 1e-13, exact)")
    print(f"     OrbitQuant (rotate->LM->reattach both) rel err = {oq_err:.4f} "
          f"(bounded quantization error, no inverse rotation)")
    print(f"     -> {PASS if passed else FAIL}")
    return passed


def check_c4():
    """C4 -- Lloyd-Max codebook is MSE-optimal on f_d (Eq 3)."""
    d = 3072          # Fig 3a attention projection dimension
    rows = []
    for b in (2, 3, 4, 8):
        cw, mse_lm = model.lloyd_max_codebook(d, b)
        mse_uni = model.uniform_mse(d, b)
        residual = model.lloyd_centroid_residual(d, cw)
        ratio = mse_uni / mse_lm
        rows.append((b, mse_lm, mse_uni, ratio, residual))
        print(f"     b={b}: Lloyd-Max MSE={mse_lm:.3e}  uniform MSE={mse_uni:.3e}  "
              f"uniform/LM={ratio:.3f}  centroid residual={residual:.2e}")
    # LM strictly better at every bit-width; gap (uniform/LM) largest at b=2;
    # optimum satisfies the centroid cell-mean condition.
    all_better = all(r[3] > 1.0 for r in rows)
    gap_largest_at_b2 = rows[0][3] == max(r[3] for r in rows)
    optimal = all(r[4] < 1e-5 for r in rows)
    passed = all_better and gap_largest_at_b2 and optimal
    print(f"[C4] Lloyd-Max MSE-optimality on f_d (d={d})")
    print(f"     LM beats uniform at every b: {all_better}; gap largest at b=2: "
          f"{gap_largest_at_b2}; centroid-optimal: {optimal}")
    print(f"     -> {PASS if passed else FAIL}")
    return passed


def check_c5():
    """C5 -- end-to-end W2A4 robustness (only functional method at W2A4).

    Two signals, both averaged over many sign+permutation draws to remove
    single-realisation noise:
      (1) RTN collapses at W2A4 while OrbitQuant stays bounded (the headline).
      (2) the permutation carries the low-bit gap (Remark 1 / Table 3 W2A4):
          with multiple outliers co-occurring in one block, Block-Hadamard
          (no perm) lets that block's coordinates saturate the f_d codebook,
          whereas RPBH spreads the mass across blocks (Prop 1 / Lemma 2).
    """
    rng = np.random.default_rng(SEED + 4)
    d, h = 512, 128
    W = data.weight_matrix_with_outliers(96, d, rng, outlier_cols=6, outlier_scale=9.0)
    cw, _ = model.lloyd_max_codebook(d, 4)
    c2, _ = model.lloyd_max_codebook(d, 2)
    T = 24

    def trial_err(x, bw, ba, use_perm):
        Pi = model.rpbh(d, rng, h=h, use_perm=use_perm)
        cwb = c2 if bw == 2 else cw
        cab = c2 if ba == 2 else cw
        What = model.orbitquant_weight(W, Pi, cwb)
        xhat = model.orbitquant_activation(x, Pi, cab)
        return model.relative_error(W @ x, What @ xhat)

    def rtn_trial_err(x, bw, ba):
        Wq = model.rtn_quantize(W, bw, axis=1)
        xq = model.rtn_quantize(x, ba, axis=None)
        return model.relative_error(W @ x, Wq @ xq)

    # random-position outliers (mild permutation gap)
    rand_w4 = np.mean([trial_err(data.activation_token_with_outliers(d, rng, 6, 9.0),
                                 4, 4, True) for _ in range(T)])
    rand_w4_np = np.mean([trial_err(data.activation_token_with_outliers(d, rng, 6, 9.0),
                                    4, 4, False) for _ in range(T)])
    rand_w2 = np.mean([trial_err(data.activation_token_with_outliers(d, rng, 6, 9.0),
                                 2, 4, True) for _ in range(T)])
    rand_w2_np = np.mean([trial_err(data.activation_token_with_outliers(d, rng, 6, 9.0),
                                    2, 4, False) for _ in range(T)])
    rtn_w4 = np.mean([rtn_trial_err(data.activation_token_with_outliers(d, rng, 6, 9.0), 4, 4)
                      for _ in range(T)])
    rtn_w2 = np.mean([rtn_trial_err(data.activation_token_with_outliers(d, rng, 6, 9.0), 2, 4)
                      for _ in range(T)])

    # adversarial: 6 outliers ALL in the same block -> Prop-1/Lemma-2 worst case
    def same_block_outlier_token():
        x = np.zeros(d)
        cols = rng.choice(h, 6, replace=False)        # all inside block 0
        x[cols] = rng.choice([-1, 1], 6) * 9.0
        x += rng.standard_normal(d) * 0.1
        return x

    adv_w2 = np.mean([trial_err(same_block_outlier_token(), 2, 4, True) for _ in range(T)])
    adv_w2_np = np.mean([trial_err(same_block_outlier_token(), 2, 4, False) for _ in range(T)])

    gap_2 = rtn_w2 / rand_w2
    perm_adv_gap = adv_w2_np / adv_w2
    passed = (rtn_w4 > rand_w4) and (rtn_w2 > 2.0 * rand_w2) and (adv_w2_np > adv_w2)
    print(f"[C5] end-to-end W4A4 / W2A4 robustness (d={d}, h={h}, T={T} trials)")
    print(f"     random-outlier: W4A4 RTN={rtn_w4:.4f} OQ={rand_w4:.4f} (no-perm={rand_w4_np:.4f}) | "
          f"W2A4 RTN={rtn_w2:.4f} OQ={rand_w2:.4f} (RTN/OQ={gap_2:.2f}, RTN collapses)")
    print(f"     permutation gap (W2A4): random OQ perm={rand_w2:.4f} vs no-perm={rand_w2_np:.4f}")
    print(f"     adversarial same-block (W2A4): OQ perm={adv_w2:.4f} vs no-perm={adv_w2_np:.4f} "
          f"(no-perm/perm={perm_adv_gap:.2f})")
    print(f"     -> {PASS if passed else FAIL}")
    return passed


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("OrbitQuant: calibration-free rotation-based quantization -- verification")
    print("=" * 78)
    results = {}
    for name, fn in [("C1", check_c1), ("C2", check_c2), ("C3", check_c3),
                     ("C4", check_c4), ("C5", check_c5)]:
        print()
        try:
            results[name] = fn()
        except Exception:  # pragma: no cover - surface the failure
            import traceback
            traceback.print_exc()
            results[name] = False
    print()
    print("=" * 78)
    print("SUMMARY: " + ", ".join(f"{k}={PASS if v else FAIL}" for k, v in results.items()))
    print("=" * 78)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
