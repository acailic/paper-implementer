"""
Synthetic data for the MARVEL toy demo.

The key design choice: classes live on S^{d-1} and have *different* per-class
concentrations kappa_c. With shared kappa the NvMF decision boundary is linear
(cosine); with per-class kappa it is non-linear on the sphere (Theorem 1's
finite-kappa regime). We also make the distribution long-tailed: the tail
classes are the low-kappa (spread) ones, which is exactly where a margin-aware
expert and the NvMF warp help most.
"""

import numpy as np


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def sample_vmf(mu, kappa, n, rng):
    """Sample n points from vMF(mu, kappa) on S^{d-1} (Wood 1994)."""
    d = len(mu)
    if kappa < 1e-6:
        X = rng.standard_normal((n, d))
        return normalize(X)
    # sample offset angle
    b = (-2.0 * kappa + np.sqrt(4.0 * kappa * kappa + (d - 1) ** 2)) / (d - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1) * np.log(1.0 - x0 * x0)
    out = np.empty((n, d))
    i = 0
    while i < n:
        beta = rng.beta((d - 1) / 2.0, (d - 1) / 2.0, size=(n - i,))
        w = (1.0 - (1.0 + b) * beta) / (1.0 - (1.0 - b) * beta)
        u = rng.uniform(size=(n - i,))
        keep = kappa * w + (d - 1) * np.log1p(-x0 * w) - c >= np.log(u)
        k = keep.sum()
        out[i:i + k, 0] = w[keep]
        i += k
    # fill remaining d-1 coords with N(0,1), normalize, then embed along mu
    rest = rng.standard_normal((n, d - 1))
    rest = normalize(rest)
    # build orthonormal basis with mu as first axis
    Q = np.zeros((d, d))
    e0 = mu.copy()
    Q[:, 0] = e0
    # Gram-Schmidt the rest of identity
    basis = np.eye(d)
    cols = [e0]
    for j in range(1, d):
        v = basis[:, j]
        for c in cols:
            v = v - (v @ c) * c
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            v = v / nv
            cols.append(v)
            Q[:, len(cols) - 1] = v
        if len(cols) == d:
            break
    pts = np.einsum("ij,nj->ni", Q[:, 1:], rest)         # tangential part
    pts *= np.sqrt(np.maximum(0.0, 1.0 - out[:, 0] ** 2))[:, None]
    pts += out[:, 0][:, None] * mu[None, :]
    return normalize(pts)


def make_longtailed_sphere(d=16, K=6, seed=0, n_train_factor=1.0,
                           aux_n=2000, ood_n=1500):
    """Long-tailed hyperspherical dataset.

    Returns dict with:
      Xtr/ytr, Xval/yval, Xtest/ytest (ID, per-class counts long-tailed),
      aux_X (ImageNet-100 proxy: uniform-ish OOD for the (K+1)th class),
      ood_near, ood_far (test OOD sets), and the true mus/kappas/priors.

    Head classes get large kappa (concentrated), tail classes small kappa
    (spread), so the NvMF finite-kappa warp is decisive on the tail.
    """
    rng = np.random.default_rng(seed)
    # class directions: well-separated random points on the sphere
    M = rng.standard_normal((K, d))
    mus = normalize(M)
    # concentration schedule: head concentrated, tail spread (kappa span large
    # enough that the finite-kappa NvMF warp is decisive on the tail)
    kappas = np.geomspace(30.0, 2.0, K)        # 30 -> 2
    # long-tailed counts: geometric decay
    counts = np.ceil(np.geomspace(1200, 15, K) * n_train_factor).astype(int)
    priors = counts / counts.sum()

    def gen(split_factor):
        Xs, ys = [], []
        for c in range(K):
            n = max(int(counts[c] * split_factor), 20)
            Xs.append(sample_vmf(mus[c], kappas[c], n, rng))
            ys.append(np.full(n, c))
        return np.concatenate(Xs), np.concatenate(ys)

    Xtr, ytr = gen(1.0)
    Xval, yval = gen(0.25)
    Xtest, ytest = gen(0.25)

    # auxiliary OOD data for the (K+1)th class + outlier expert: uniform sphere
    aux_X = normalize(rng.standard_normal((aux_n, d)))

    # nearOOD: directions *between* classes (midpoints of pairs), moderate shift
    pairs = [(i, (i + 1) % K) for i in range(K)]
    near_dirs = normalize(np.array([mus[i] + mus[j] for i, j in pairs]))
    near_idx = rng.integers(0, len(near_dirs), size=ood_n)
    ood_near = np.stack([sample_vmf(near_dirs[j], 0.8, 1, rng)[0] for j in near_idx])

    # farOOD: uniform on the sphere
    ood_far = normalize(rng.standard_normal((ood_n, d)))

    return {
        "d": d, "K": K,
        "mus": mus, "kappas": kappas, "priors": priors, "counts": counts,
        "Xtr": Xtr, "ytr": ytr,
        "Xval": Xval, "yval": yval,
        "Xtest": Xtest, "ytest": ytest,
        "aux_X": aux_X,
        "ood_near": ood_near, "ood_far": ood_far,
    }
