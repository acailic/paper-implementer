"""
MARVEL: Margin-Aware Robust von Mises-Fisher Expert Learning for Long-Tailed
Out-of-Distribution Detection — core algorithm.

Implements the Nonlinear vMF (NvMF) classifier (Eq 7, Theorem 1), the
margin-aware multi-expert ensemble (Eqs 14-15), the dedicated outlier expert
(Eq 16) and the combined OOD score (Eqs 17-19).

Paper: Anudeep & Sundaresan (2026), arXiv:2607.02435.

Convention: every embedding x and every class direction mu lives on the unit
sphere S^{d-1}. The vMF normalizing constant is
    C_d(kappa) = kappa^{d/2-1} / ((2 pi)^{d/2} I_{d/2-1}(kappa))
which is *decreasing* in kappa (its log is ~ -kappa + (d-1)/2 log kappa).
"""

import numpy as np
from scipy.special import ive, iv  # Bessel I_v (ive stable, iv for ratio)


# ── vMF normalizing constant ────────────────────────────────────────

def vmf_logCd(d, kappa):
    """log C_d(kappa), numerically stable for large kappa via ive.

    C_d(kappa) = kappa^{p-1} / ((2 pi)^{d/2} I_p(kappa)), p = d/2 - 1.
    ive(p, kappa) = I_p(kappa) e^{-kappa}, so I_p = ive * e^{kappa} and
        log I_p = log ive + kappa  =>  -log I_p = -log ive - kappa.
    Hence log C_d = (p-1) log kappa - (d/2) log(2 pi) - log(ive(p,kappa)) - kappa.
    (Decreasing in kappa, as required: ~ -kappa + (d-1)/2 log kappa.)
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    p = d / 2.0 - 1.0
    kk = np.maximum(kappa, 1e-12)
    out = ((p - 1.0) * np.log(kk)
           - (d / 2.0) * np.log(2.0 * np.pi)
           - np.log(ive(p, kk) + 1e-300)
           - kk)
    return out if out.ndim > 0 else float(out)


def vmf_Cd(d, kappa):
    return np.exp(vmf_logCd(d, kappa))


# ── NvMF logit (Eq 7) ───────────────────────────────────────────────

def nvmf_logit(X, mu, kappa, d):
    """NvMF logit (Eq 7):  ell(x) = -log( C_d(||kappa*mu + x||) / C_d(kappa) ).

    X: (N, d) unit vectors. mu: (d,) unit vector. kappa >= 0 scalar.
    Returns (N,) logits. Because C_d is decreasing and ||kappa mu + x|| is
    increasing in rho = mu^T x, the logit is *increasing* in rho -> argmax
    classifies by the best-aligned direction, exactly like cosine, but with a
    non-linear (log-partition) warp.
    """
    X = np.atleast_2d(X)
    r = np.linalg.norm(kappa * mu[None, :] + X, axis=1)   # (N,)
    return -vmf_logCd(d, r) + vmf_logCd(d, kappa)


# ── MLE fit for a vMF class ─────────────────────────────────────────

def _mean_resultant(d, kappa):
    """A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa) = E[mu^T x] under vMF."""
    nu = d / 2.0
    return float(iv(nu, kappa) / iv(nu - 1.0, kappa))


def fit_vmf(Xc, d):
    """Exact MLE vMF fit. Returns (mu_hat unit vector, kappa_hat).

    mu_hat = normalized mean direction; kappa_hat solves A_d(kappa) = r_bar
    (the mean resultant length) by bisection — this is the vMF MLE (Sra 2012),
    far more accurate at small kappa than the closed-form approximations.
    """
    mu_hat = Xc.mean(axis=0)
    r = float(np.clip(np.linalg.norm(mu_hat), 1e-6, 1.0 - 1e-6))
    mu_hat = mu_hat / max(np.linalg.norm(mu_hat), 1e-12)
    lo, hi = 1e-4, 1e5
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if _mean_resultant(d, mid) < r:
            lo = mid
        else:
            hi = mid
    return mu_hat, float(np.clip(np.sqrt(lo * hi), 1e-3, 1e4))


# ── NvMF classifier ─────────────────────────────────────────────────

class NvMFClassifier:
    """Per-class (mu_c, kappa_c) NvMF classifier over K classes (+ optional
    auxiliary (K+1)-th OOD class).

    forward(X) returns the K (+1) NvMF logits (Eq 7) for every class.
    """

    def __init__(self, mus, kappas, d, aux_mu=None, aux_kappa=None):
        self.mus = np.stack(mus)                  # (K, d)
        self.kappas = np.asarray(kappas, float)   # (K,)
        self.d = d
        self.aux_mu = aux_mu
        self.aux_kappa = aux_kappa

    @classmethod
    def fit(cls, X, y, d, K, aux_X=None):
        mus, kappas = [], []
        for c in range(K):
            mu, ka = fit_vmf(X[y == c], d)
            mus.append(mu)
            kappas.append(ka)
        aux_mu = aux_kappa = None
        if aux_X is not None and len(aux_X):
            aux_mu, aux_kappa = fit_vmf(aux_X, d)
        return cls(mus, kappas, d, aux_mu, aux_kappa)

    def logits(self, X):
        """(N, K) NvMF logits; (N, K+1) if auxiliary OOD class is set."""
        X = np.atleast_2d(X)
        L = np.stack([nvmf_logit(X, self.mus[c], self.kappas[c], self.d)
                      for c in range(len(self.kappas))], axis=1)
        if self.aux_mu is not None:
            Laux = nvmf_logit(X, self.aux_mu, self.aux_kappa, self.d)[:, None]
            L = np.concatenate([L, Laux], axis=1)
        return L

    def predict(self, X):
        L = self.logits(X)
        return L[:, :len(self.kappas)].argmax(axis=1)


# ── Margin-aware shift (Eqs 14-15) ──────────────────────────────────

def margin_shifted_logits(L, priors, tau, temp=None):
    """Apply the margin-aware inference shift that emulates an expert trained
    with Delta_yc = tau * log(pi_c / pi_y) (Eq 14-15).

    To *boost* the tail (small pi_c) we SUBTRACT tau*log(pi_c): head classes
    (log pi_c near 0) barely move, tail classes (very negative log pi_c) are
    pushed up. tau=0 => head-biased argmax ell_c; tau=2 => strongly tail-biased.

    `priors` may be shorter than L by one column (the auxiliary OOD class);
    it is padded with its mean so the aux logit is shifted by a neutral amount.

    NvMF logits are O(1) (bounded in ~[-1, 1], Theorem 1), so a raw log-prior
    shift of magnitude |log pi_min| can dominate. `temp` normalises the shift to
    the logit scale (default = the log-prior dynamic range), keeping softmax
    non-degenerate as tau ranges over {0,1,2} as in the paper.
    """
    priors = np.asarray(priors, float)
    if L.shape[1] == len(priors) + 1:               # pad aux OOD class (neutral)
        priors = np.concatenate([priors, [priors.mean()]])
    if temp is None:
        spread = float(np.log(priors.max()) - np.log(priors.min()))
        temp = max(spread, 1.0)
    return L - (tau / temp) * np.log(priors)


def margin_aware_loss(L_tau, y):
    """Margin-aware NvMF loss (Eq 15):
        L = mean_n log( 1 + sum_{c != y} exp(L_c - L_y) ).  (lower = better)
    """
    return softplus_margin_loss(L_tau, y)


def softplus_margin_loss(L_tau, y):
    """Clean Eq 15: mean over samples of log(1 + sum_{c!=y} exp(L_c - L_y)).
    Stable form: m = max_c d_c, S = sum_{c!=y} exp(d_c - m) (the argmax
    competitor contributes 1), loss = logaddexp(0, m + log S) = log(1 + e^m S).
    """
    N, C = L_tau.shape
    Ly = L_tau[np.arange(N), y][:, None]
    mask = np.ones_like(L_tau, dtype=bool)
    mask[np.arange(N), y] = False
    diffs = np.where(mask, L_tau - Ly, -1e9)
    m = diffs.max(axis=1, keepdims=True)
    S = np.where(mask, np.exp(diffs - m), 0.0).sum(axis=1, keepdims=True)
    loss = np.logaddexp(0.0, m + np.log(S + 1e-12))
    return float(np.mean(loss))


# ── Outlier expert (Eq 16) ──────────────────────────────────────────

class OutlierExpert:
    """Binary FC expert g_out: R^d -> R^2 (ID vs OOD), trained by logistic
    regression on balanced ID+aux-OOD batches. Proxy for the paper's FC head."""

    def __init__(self, d, l2=1e-2, lr=0.1, steps=400, seed=0):
        self.d = d
        self.l2 = l2
        self.lr = lr
        self.steps = steps
        self.rng = np.random.default_rng(seed)

    def fit(self, X_id, X_ood):
        # balance by subsampling to the smaller side
        n = min(len(X_id), len(X_ood))
        idx_id = self.rng.choice(len(X_id), n, replace=False)
        idx_ood = self.rng.choice(len(X_ood), n, replace=False)
        Xb = np.concatenate([X_id[idx_id], X_ood[idx_ood]])
        yb = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
        W = np.zeros((self.d, 2))
        b = np.zeros(2)
        N = len(Xb)
        for _ in range(self.steps):
            Z = Xb @ W + b
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
            G = (P - np.eye(2)[yb]) / N
            W -= self.lr * (Xb.T @ G + self.l2 * W)
            b -= self.lr * G.sum(axis=0)
        self.W, self.b = W, b
        return self

    def ood_prob(self, X):
        Z = X @ self.W + self.b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        return P[:, 1]   # P(OOD)


# ── MARVEL ensemble + combined OOD score (Eqs 17-19) ────────────────

def softmax_np(Z, axis=-1):
    Z = Z - Z.max(axis=axis, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=axis, keepdims=True)


def marvel_ood_scores(clf, priors, X, taus=(0.0, 1.0, 2.0)):
    """Aggregate per-expert NvMF OOD score (Eq 18) over margin experts.

    Requires clf.aux (the (K+1)-th OOD class). s_NvMF = mean_t softmax(L^t)_{K+1}.
    Returns dict with per-expert scores, the aggregated NvMF score, and the
    argmax-ID prediction under the balanced (tau=1) expert."""
    K = len(clf.kappas)
    aux_col = K   # index of auxiliary OOD class
    per_expert = {}
    preds = None
    for t in taus:
        L = clf.logits(X)
        Lt = margin_shifted_logits(L, priors, t)
        P = softmax_np(Lt, axis=1)
        per_expert[t] = P[:, aux_col]
        if abs(t - 1.0) < 1e-9:
            preds = Lt[:, :K].argmax(axis=1)
    s_nvmf = np.mean(np.stack(list(per_expert.values()), axis=0), axis=0)  # Eq 18
    return {"per_expert": per_expert, "s_nvmf": s_nvmf, "id_pred": preds}


def combined_ood_score(s_nvmf, s_out):
    """Eq 19: S_OOD = 0.5 (s_NvMF + s_ood)."""
    return 0.5 * (s_nvmf + s_out)


# ── Theorem 1 / asymptotic helpers (for verification) ──────────────

def bessel_logCd_asymptotic(d, kappa, order=2):
    """Large-kappa expansion of log C_d(kappa) (Eq 9):
        log C_d(kappa) = -kappa + (d-1)/2 log kappa + a0 + a1/kappa + O(kappa^-2).
    Coefficients from I_nu(kappa) ~ e^kappa/sqrt(2 pi kappa)(1 - (4 nu^2-1)/(8 kappa)+...),
    nu = d/2 - 1, so 4 nu^2 - 1 = (d-3)(d-1).
    """
    kappa = np.asarray(kappa, float)
    nu = d / 2.0 - 1.0
    a0 = -(d / 2.0) * np.log(2 * np.pi) + 0.5 * np.log(np.pi / 2.0)
    term = -kappa + (d - 1.0) / 2.0 * np.log(kappa) + a0
    if order >= 1:
        term = term - (4 * nu * nu - 1) / (8.0 * kappa)
    return term


def norm_kmu_plus_x_expansion(kappa, rho, order=2):
    """||kappa mu + x|| expansion (Eq 10):
        r = kappa + rho + (1 - rho^2)/(2 kappa) + O(kappa^-2).
    """
    r = kappa + rho + (1 - rho * rho) / (2.0 * kappa)
    return r


# ── metrics ─────────────────────────────────────────────────────────

def auroc(id_scores, ood_scores):
    """Higher score = more OOD. AUROC via the rank-statistic formula."""
    s = np.concatenate([id_scores, ood_scores])
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    order = np.argsort(s)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(s) + 1)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    if n0 == 0 or n1 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))
