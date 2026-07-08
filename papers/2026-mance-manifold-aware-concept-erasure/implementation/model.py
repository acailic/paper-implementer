"""
MANCE: Manifold Aware Concept Erasure — core algorithm.

Implements LEACE, CovMatch, and MANCE (with MANCE+/MANCE++ variants).
Paper: Avitan, Goldberg, Elazar (2026), arXiv:2607.03973.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors as KNNRegressor


# ── Probe ──────────────────────────────────────────────────────────

class Probe(nn.Module):
    """2-layer MLP probe for binary concept classification."""

    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_probe(X, y, d, hidden=128, lr=0.01, steps=200, batch_size=256):
    """Train probe on (X, y). Returns trained probe and final accuracy."""
    probe = Probe(d, hidden)
    opt = torch.optim.SGD(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n = X.shape[0]
    for step in range(steps):
        idx = torch.randperm(n)[:batch_size]
        xb, yb = X[idx], y[idx]
        logits = probe(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Accuracy
    with torch.no_grad():
        logits = probe(X)
        preds = (logits > 0).float()
        acc = (preds == y).float().mean().item()
    return probe, acc


# ── LEACE ──────────────────────────────────────────────────────────

def leace(X, y):
    """
    LEACE: Least-squares Antialiasing Concept Erasure.
    Rank-1 cross-covariance projection.
    Returns: X_erased, projector P such that X_erased = X @ P.
    """
    X_np = X.numpy()
    y_np = y.numpy()

    y_centered = y_np - y_np.mean()
    X_centered = X_np - X_np.mean(axis=0)

    # Cross-covariance: C = X^T y  (d × 1)
    cross_cov = X_centered.T @ y_centered / len(y_np)

    # P = I - (C @ C^T) / (C^T @ C)
    C = cross_cov[:, None]  # d × 1
    denom = (C.T @ C).clip(min=1e-12)
    P = np.eye(X_np.shape[1]) - (C @ C.T) / denom

    X_erased = X_np @ P
    return torch.tensor(X_erased, dtype=torch.float32), P


# ── CovMatch ────────────────────────────────────────────────────────

def covmatch(X, y):
    """
    CovMatch: rank-2 covariance asymmetry removal.
    Removes leading directions where class-conditional covariances differ.
    """
    X_np = X.numpy()
    y_np = y.numpy()

    idx0 = y_np == 0
    idx1 = y_np == 1

    X0 = X_np[idx0] - X_np[idx0].mean(axis=0)
    X1 = X_np[idx1] - X_np[idx1].mean(axis=0)

    S0 = X0.T @ X0 / max(len(X0) - 1, 1)
    S1 = X1.T @ X1 / max(len(X1) - 1, 1)

    # Covariance asymmetry
    Delta_S = S1 - S0
    U, S, Vt = np.linalg.svd(Delta_S, full_matrices=False)

    # Mean direction
    mean_diff = X_np[idx1].mean(axis=0) - X_np[idx0].mean(axis=0)
    mean_dir = mean_diff / max(np.linalg.norm(mean_diff), 1e-12)

    # Build D = [mean_dir, top-2 eigenvectors of Delta_S]
    D = np.column_stack([mean_dir, U[:, 0], U[:, 1]])  # d × 3
    D, _ = np.linalg.qr(D)  # orthonormalize

    P = np.eye(X_np.shape[1]) - D @ D.T
    X_erased = X_np @ P
    return torch.tensor(X_erased, dtype=torch.float32), P


# ── MANCE ──────────────────────────────────────────────────────────

def mance(
    X_nat,      # Natural (original) representations, N × d — FIXED reference
    X,           # Current representations to edit, N × d
    y,           # Target concept labels, N
    *,
    H=20,        # Erasure rounds
    k=20,        # kNN neighborhood size
    r=8,         # Tangent rank (local PCA dimension)
    eps=0.5,     # Neighborhood scale
    lambda_max=1.0,
    alpha=1.0,   # Spectral exponent
    tau=8,       # Probe refit period (rounds)
    d_model=None,
    probe_hidden=128,
    verbose=True,
):
    """
    MANCE: iterative gradient-based erasure projected onto local tangent space.

    At each round:
      1. Fit/refresh nonlinear probe f_t on (X, y)
      2. For each sample:
         a. kNN from NATURAL X_nat → local PCA → tangent basis B_i
         b. Gradient direction ∇f_t → project onto tangent space via B_i
         c. Spectral weighting by singular values
         d. Closed-form step size with local-neighborhood cap
      3. Update X

    Returns: X_edited (N×d), per-round metrics dict.
    """
    N, d = X.shape
    if d_model is None:
        d_model = d

    X = X.clone()
    X_nat_np = X_nat.numpy()

    # Precompute kNN indices from natural representations
    knn = KNNRegressor(n_neighbors=k, algorithm='auto')
    knn.fit(X_nat_np)

    metrics = {"leakage": [], "probe_acc": []}

    # Initial probe
    probe, acc = train_probe(X, y, d_model, hidden=probe_hidden)
    metrics["probe_acc"].append(acc)
    if verbose:
        print(f"  MANCE init probe acc: {acc:.4f}")

    for t in range(1, H + 1):
        # Refresh probe every tau rounds
        if t == 1 or t % tau == 0:
            probe, acc = train_probe(X, y, d_model, hidden=probe_hidden)
            if verbose and t > 1:
                print(f"  Round {t}: refreshed probe acc={acc:.4f}")

        # kNN from NATURAL representations (no grad needed)
        with torch.no_grad():
            X_np = X.numpy()
            distances, indices = knn.kneighbors(X_np)  # N × k

        # Batch gradient computation (needs grad)
        X_grad = X.detach().requires_grad_(True)
        logits = probe(X_grad)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        grad_all = torch.autograd.grad(loss, X_grad)[0].detach().numpy()  # N × d

        # Per-sample manifold-constrained updates
        X_new = X_np.copy()

        for i in range(N):
            # ─ Step 1: Local manifold estimation ─
            neighbor_idx = indices[i]  # k indices into X_nat
            neighbors = X_nat_np[neighbor_idx]  # k × d
            x_bar = neighbors.mean(axis=0)
            S_i = neighbors - x_bar  # k × d centered

            # SVD for local PCA
            try:
                _, sigma_i, Vti = np.linalg.svd(S_i, full_matrices=False)
            except np.linalg.LinAlgError:
                continue

            # Tangent basis: top-r right singular vectors
            r_actual = min(r, len(sigma_i))
            B_i = Vti[:r_actual].T  # d × r

            # ─ Step 2: Tangent erasure direction ─
            u_i = grad_all[i]
            u_norm = np.linalg.norm(u_i)
            if u_norm < 1e-10:
                continue
            u_i = u_i / u_norm

            # Project onto tangent basis
            c_i = B_i.T @ u_i  # r

            # Spectral weighting
            sigma_weights = sigma_i[:r_actual] ** alpha
            d_i = B_i @ (sigma_weights * c_i)  # d

            d_norm = np.linalg.norm(d_i)
            if d_norm < 1e-10:
                continue
            d_hat = d_i / d_norm

            # ─ Step 3: Local-neighborhood cap ─
            r_i = distances[i].mean()  # avg distance to natural neighbors
            projection = X_np[i] @ d_hat
            if abs(projection) < 1e-10:
                continue

            lambda_i = min(lambda_max, eps * r_i / abs(projection))

            X_new[i] = X_np[i] - lambda_i * projection * d_hat

        X = torch.tensor(X_new, dtype=torch.float32)

        # Metrics (probe needs grad for training)
        probe, acc = train_probe(X, y, d_model, hidden=probe_hidden, steps=50)
        metrics["probe_acc"].append(acc)

        if verbose:
            print(f"  Round {t}/{H}: probe_acc={acc:.4f}")

    return X, metrics


def mance_pp(X, y, **mance_kwargs):
    """
    MANCE++: LEACE → CovMatch → MANCE loop.
    """
    print("=== LEACE preprocessing ===")
    X_leace, _ = leace(X, y)
    # Quick probe after LEACE
    _, acc_leace = train_probe(X_leace, y, X.shape[1], steps=50)
    print(f"  After LEACE: probe_acc={acc_leace:.4f}")

    print("\n=== CovMatch preprocessing ===")
    X_cov, _ = covmatch(X_leace, y)
    _, acc_cov = train_probe(X_cov, y, X.shape[1], steps=50)
    print(f"  After CovMatch: probe_acc={acc_cov:.4f}")

    print("\n=== MANCE loop ===")
    X_final, metrics = mance(X, X_cov, y, **mance_kwargs)

    return X_final, metrics


# ── Evaluation ─────────────────────────────────────────────────────

def evaluate_erasure(X_clean, X_erased, y_target, y_control):
    """
    Evaluate concept erasure quality.

    Returns dict with:
      - target_leakage: probe accuracy on erased (lower = better)
      - control_r2: R² of control regression on erased (higher = better)
      - surgicality_delta: change in control R² (should be small)
    """
    d = X_clean.shape[1]

    # Target probe accuracy
    _, acc_clean = train_probe(X_clean, y_target, d, steps=100)
    _, acc_erased = train_probe(X_erased, y_target, d, steps=100)

    # Control R² (linear regression)
    def r2_score(X, y):
        X_np = X.numpy()
        y_np = y.numpy()
        # Simple linear regression
        X_c = np.column_stack([X_np, np.ones(len(X_np))])
        beta = np.linalg.lstsq(X_c, y_np, rcond=None)[0]
        pred = X_c @ beta
        ss_res = np.sum((y_np - pred) ** 2)
        ss_tot = np.sum((y_np - y_np.mean()) ** 2)
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    r2_clean = r2_score(X_clean, y_control)
    r2_erased = r2_score(X_erased, y_control)

    return {
        "target_acc_clean": acc_clean,
        "target_acc_erased": acc_erased,
        "target_leakage_pp": (acc_erased - 0.5) * 100,  # above chance, in pp
        "control_r2_clean": r2_clean,
        "control_r2_erased": r2_erased,
        "surgicality_delta_r2": r2_clean - r2_erased,
    }
