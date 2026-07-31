"""Small policy models and the MOPD objective for UI-MOPD.

Paper: "UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI
Agent Learning", arXiv 2607.04425 (Lian et al., 2026).

This module implements the *core* learning mechanism of the paper on a
toy scale:

  * `PolicyLM`           - a tiny transformer decoder used for both the
                            shared student and the two platform teachers.
  * `k3_kl_estimator`    - Eq. 4-5: the non-negative, low-variance K3 KL
                            estimator  D_hat = rho - delta - 1.
  * `adaptive_kl_mask`   - Eq. 6: per-prompt-group mask mu that drops the
                            teacher penalty once a group's mean reward
                            exceeds tau_KL.
  * `structured_reward`  - Eq. 8:  +1.0 / -0.5 / -1.0 outcome reward based
                            on action-dimension match fraction f_a.
  * `mopd_loss`          - Eq. 9-12: clipped PPO policy loss + beta * K3 KL,
                            aggregated token-mean with the adaptive mask.

It is intentionally model-agnostic: any autoregressive model exposing
`log_probs(seq, mask)` can be plugged in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Tokenizer / vocabulary (shared across both platforms)
# ---------------------------------------------------------------------------
# A minimal action vocabulary. Desktop and mobile share some tokens
# (numbers, coords, type-text) but have platform-specific action verbs,
# mirroring the paper's Table 5 action spaces.

DESKTOP_VERBS = ["key", "type", "mouse_move", "left_click", "right_click",
                 "scroll", "wait", "terminate"]
MOBILE_VERBS = ["click", "long_press", "swipe", "type", "answer",
                "system_button", "wait", "terminate"]

PAD, BOS, EOS, SEP = "<pad>", "<bos>", "<eos>", "<sep>"
_SPECIAL = [PAD, BOS, EOS, SEP]
COORDS = [f"c{i}" for i in range(16)]   # discretised coordinates / keys
_DIGITS = [f"d{i}" for i in range(10)]
_TEXT = [f"w{i}" for i in range(8)]       # type-text slots
# task keywords that uniquely identify each action template so the toy
# task is learnable from the prompt (see data.py).
_TASKS = ["click_task", "rightclick_task", "move_task", "scroll_task",
          "keypress_task", "typing_task", "wait_task", "end_task",
          "tap_task", "hold_task", "swipe_task", "input_task",
          "reply_task", "sysbtn_task"]

VOCAB = (_SPECIAL + sorted(set(DESKTOP_VERBS) | set(MOBILE_VERBS))
         + COORDS + _DIGITS + _TEXT + _TASKS)
VOCAB_SIZE = len(VOCAB)
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for i, t in enumerate(VOCAB)}

DESKTOP_VERB_IDS = {STOI[v] for v in DESKTOP_VERBS}
MOBILE_VERB_IDS = {STOI[v] for v in MOBILE_VERBS}


def encode(tokens: list[str]) -> torch.Tensor:
    return torch.tensor([STOI[t] for t in tokens], dtype=torch.long)


def decode(ids: torch.Tensor) -> list[str]:
    return [ITOS[int(i)] for i in ids.tolist()]


def valid_action_tokens(ids: list[int], platform: str) -> bool:
    """Return True iff the first non-special token is a legal verb for the
    platform (a coarse stand-in for the paper's f_a match fraction)."""
    verbs = DESKTOP_VERB_IDS if platform == "desktop" else MOBILE_VERB_IDS
    for i in ids:
        if i in (STOI[PAD], STOI[BOS], STOI[EOS], STOI[SEP]):
            continue
        return i in verbs
    return False


# ---------------------------------------------------------------------------
# PolicyLM - tiny transformer decoder
# ---------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_len: int = 24
    dropout: float = 0.0


class PolicyLM(nn.Module):
    """A minimal causal transformer language model.

    Exposes `log_probs(seq, mask)` returning the log-probability the model
    assigns to each *target* token (the token at the next position), which
    is what the KL / PPO objectives need.
    """

    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        c = self.cfg
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model, nhead=c.n_heads,
            dim_feedforward=c.d_model * 4,
            dropout=c.dropout, batch_first=True, activation="gelu",
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=c.n_layers)
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size)

    def _logits(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        # causal mask so position t cannot attend to t+1..
        mask = torch.triu(
            torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1
        )
        x = self.tr(x, mask=mask)
        x = self.ln_f(x)
        return self.head(x)

    def log_probs(self, idx: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Log-prob the model assigns to each token.

        A token at position t is predicted from the context at positions
        0..t-1, i.e. by logits at position t-1 (standard next-token LM).

        idx:           (B, T) full sequence  [bos, prompt..., response...]
        response_mask: (B, T) 1 for response-token positions to score, else 0
        Returns:       (B, T) token log-probs (0 where mask==0 or at t=0).
        """
        logits = self._logits(idx)                       # (B, T, V)
        lp = F.log_softmax(logits, dim=-1)               # (B, T, V)
        # token at t (t>=1) is predicted by logits[t-1]
        token_lp = lp[:, :-1, :].gather(
            2, idx[:, 1:].unsqueeze(-1)).squeeze(-1)     # (B, T-1)
        token_lp = F.pad(token_lp, (1, 0))               # (B, T), col 0 = 0
        return token_lp * response_mask

    @torch.no_grad()
    def sample(self, prefix: torch.Tensor, new_tokens: int,
               temperature: float = 1.0, device: str = "cpu") -> torch.Tensor:
        """Greedy-ish sampling for rollouts. Returns full sequence."""
        out = prefix.clone()
        for _ in range(new_tokens):
            logits = self._logits(out)[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, nxt], dim=1)
            if out.size(1) >= self.cfg.max_len:
                break
        return out


# ---------------------------------------------------------------------------
# Eq. 4-5: K3 KL estimator
# ---------------------------------------------------------------------------
def k3_kl_estimator(logp_theta: torch.Tensor,
                    logp_ref: torch.Tensor,
                    clamp_delta: float = 10.0) -> torch.Tensor:
    """D_hat_KL = rho - delta - 1  (Eq. 5), with delta clamped (Eq. 4).

    Non-negative and unbiased for D_KL(pi_theta || pi_ref) under samples
    drawn from pi_theta.  Returns per-token estimate, same shape as inputs.
    """
    delta = logp_ref - logp_theta                       # Eq. 4
    delta = delta.clamp(-clamp_delta, clamp_delta)
    rho = torch.exp(delta)                              # Eq. 5 line 1
    return rho - delta - 1.0                            # Eq. 5 line 2


# ---------------------------------------------------------------------------
# Eq. 6: adaptive (group-level) KL mask
# ---------------------------------------------------------------------------
def adaptive_kl_mask(group_rewards: torch.Tensor,
                     tau_kl: float) -> torch.Tensor:
    """mu = 0 if mean reward of the prompt-group > tau_KL else 1 (Eq. 6).

    group_rewards: (G,) mean reward of each prompt group (G groups).
    Returns mu: (G,) in {0, 1}.
    """
    return (group_rewards <= tau_kl).float()


# ---------------------------------------------------------------------------
# Eq. 8: structured outcome reward
# ---------------------------------------------------------------------------
def structured_reward(seq_ids: list[int], platform: str,
                      target: list[int]) -> float:
    """R(x, y) = +1 if action fully matches, -0.5 if partial, -1 if invalid.

    f_a = fraction of matched action dimensions (here: verb correctness +
    coordinate/key match against the target template). Mirrors Eq. 8.
    """
    ids = [i for i in seq_ids
           if i not in (STOI[PAD], STOI[BOS], STOI[EOS], STOI[SEP])]
    if not ids or not valid_action_tokens(ids, platform):
        return -1.0                                     # unparsable / invalid
    # dimension 1: correct verb
    verb_ok = 1.0 if ids[0] == target[0] else 0.0
    # dimension 2: correct coordinate/key token (if target specifies one)
    coord_ok = 1.0
    if len(target) > 1:
        coord_ok = 1.0 if (len(ids) > 1 and ids[1] == target[1]) else 0.0
    f_a = 0.5 * verb_ok + 0.5 * coord_ok
    if f_a >= 1.0:
        return 1.0
    if f_a > 0.0:
        return -0.5
    return -1.0


# ---------------------------------------------------------------------------
# Eq. 9-12: the full MOPD objective
# ---------------------------------------------------------------------------
def token_advantage(rewards: torch.Tensor,
                    group_ids: torch.Tensor) -> torch.Tensor:
    """A_t = R(x,y) - mean(R over the prompt group)  (Eq. 9).

    rewards: (N,) per-rollout reward.
    group_ids: (N,) id of the prompt each rollout belongs to.
    Returns A: (N,).
    """
    A = torch.zeros_like(rewards)
    for g in group_ids.unique():
        sel = (group_ids == g)
        baseline = rewards[sel].mean()
        A[sel] = rewards[sel] - baseline
    return A


def mopd_loss(student_lp_new: torch.Tensor,   # log p_theta(y) after update
              student_lp_old: torch.Tensor,   # log p_theta_old(y) at rollout
              teacher_lp: torch.Tensor,       # log p_ref(y), platform-routed
              response_mask: torch.Tensor,    # (N, T)
              advantages: torch.Tensor,       # (N,)
              mu: torch.Tensor,               # (N,) adaptive KL mask
              beta: float = 0.01,
              eps_low: float = 0.2,
              eps_high: float = 0.28) -> dict:
    """Combined clipped-PPO + platform-conditioned K3-KL loss (Eq. 10-12).

    Returns a dict with the scalar loss to minimise plus diagnostic terms.
    """
    # PPO ratio r_t = pi_new / pi_old
    log_ratio = student_lp_new - student_lp_old
    ratio = torch.exp(log_ratio)

    # broadcast token-level advantage (constant across tokens of a rollout)
    A = advantages.unsqueeze(1).expand_as(ratio)
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * A
    pg_per_token = -torch.min(surr1, surr2)            # Eq. 11 (to minimise)
    m = response_mask
    pg_loss = (pg_per_token * m).sum() / m.sum().clamp(min=1.0)

    # K3 KL, masked by adaptive mu
    kl_per_token = k3_kl_estimator(student_lp_new, teacher_lp)   # Eq. 5
    mu_b = mu.unsqueeze(1).expand_as(kl_per_token)
    weight = (mu_b * m).sum().clamp(min=1.0)
    kl_loss = (kl_per_token * mu_b * m).sum() / weight

    loss = pg_loss + beta * kl_loss                      # Eq. 12
    return {
        "loss": loss,
        "pg_loss": pg_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "mean_ratio": ratio.mean().detach(),
        "mean_kl_raw": kl_per_token[m.bool()].mean().detach()
        if m.bool().any() else torch.tensor(0.0),
    }
