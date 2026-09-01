"""AgentOPSD — from-scratch re-implementation on a synthetic multi-turn task.

Paper: "AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement
Learning" (Wang et al., 2026, arXiv:2608.05987). Written from our own
breakdown.md; no reference code copied.

What is implemented (following the paper's Section 3):
  1. GRPO group advantage  A = (R - mean)/std                (Eq. 1-2)
  2. Teacher/student per-token log-prob gap, detached        (Eq. 4-5)
  3. Turn evidence  e_k = sum_t delta_{k,t}                  (Eq. 6)
  4. Recursive log-odds belief  B_k = sigmoid(logit(B0) + gamma*c_{k-1} + e_k)
     with B0 = group success rate                            (Eq. 8)
  5. Marginal credit  dB_k = B_k - B_{k-1}                   (Eq. 9)
  6. Outcome-aligned credit  q_k = sign(A) * dB_k            (Eq. 10)
  7. Bounded sign-preserving reshaper
     w_k = clip(1 + b*z_k, 1-b, 1+b);  A~_k = A*((1-lam) + lam*w_k)   (Eq. 11)
  8. Clipped PPO/GRPO update                                 (Eq. 12)

The policy is a tiny GRU "LLM" over a discrete vocabulary (obs ids, action
ids, skill entries). Tiny = the algorithm's structure is the point, not the
architecture. The SAME weights serve as student and (privileged-context)
teacher, exactly as in the paper: the teacher is the same network run a
second time with the skill c+ prepended to the context.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Policy: a tiny autoregressive "LLM" (GRU over a discrete vocab).
# ----------------------------------------------------------------------------


class TinyPolicy(nn.Module):
    """Token vocab: [0, H) observation ids, then A action ids offset by H.

    Input stream per turn:  [obs_k] -> sample one action token.
    The skill (teacher context) is prepended as [SKILL] c1 c2 ... [/SKILL].
    One action token per turn keeps the math identical to the paper while
    staying runnable on CPU in seconds.
    """

    def __init__(self, n_obs: int, n_actions: int, d_model: int = 64):
        super().__init__()
        self.n_obs, self.n_actions = n_obs, n_actions
        self.vocab = n_obs + n_actions + 2  # + [SKILL], [/SKILL]
        self.SKILL, self.SKILL_END = n_obs + n_actions, n_obs + n_actions + 1
        self.embed = nn.Embedding(self.vocab, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, n_actions)  # predicts action id in [0, A)

    def initial_state(self, batch: int, device):
        return torch.zeros(1, batch, self.gru.hidden_size, device=device)

    def forward(self, tokens, hidden):
        h, hidden = self.gru(self.embed(tokens), hidden)
        logits = self.head(h)  # (B, L, A)
        return logits, hidden

    def render_context(self, obs: int, skill=None):
        """Context tokens for one turn: optional skill block, then obs.
        Skill entries equal to -1 (obs types with no pivotal turn) are
        skipped — they carry no information."""
        if skill is not None:
            toks = [self.SKILL] + [int(s) for s in skill if int(s) >= 0] + [self.SKILL_END]
        else:
            toks = []
        return toks + [obs]


# ----------------------------------------------------------------------------
# Rollout: run the policy on a task; return per-turn log-probs (student ctx
# and teacher ctx), actions, reward.
# ----------------------------------------------------------------------------


def rollout(policy, task, device, temperature: float = 1.0, greedy: bool = False):
    """Sample one trajectory. Returns dict with:
    actions      : list[int]     length K (one action per turn)
    logp_student : list[float]   log pi(a_k | plain ctx)   per turn
    logp_teacher : list[float]   log pi(a_k | skill ctx)   per turn (same weights)
    reward       : int           binary terminal reward
    """
    policy.eval()
    T = task.T
    skill = task.skill_vector()
    actions, logp_s, logp_t = [], [], []
    with torch.no_grad():
        # ---- pass 1: STUDENT context (no skill) — samples the actions ----
        hidden = policy.initial_state(1, device)
        for k in range(T):
            ctx = policy.render_context(task.obs(k), skill=None)
            tokens = torch.tensor([ctx], dtype=torch.long, device=device)
            logits, hidden = policy(tokens, hidden)
            logp_all = F.log_softmax(logits[0, -1] / temperature, dim=-1)
            if greedy:
                a = int(torch.argmax(logp_all).item())
            else:
                a = int(torch.multinomial(F.softmax(logp_all, dim=-1), 1).item())
            actions.append(a)
            logp_s.append(float(logp_all[a].item()))
        # ---- pass 2: TEACHER context (skill prepended) — re-scores the SAME
        #      actions under the privileged context (OPSD trick, Eq. 4-5) ----
        hidden = policy.initial_state(1, device)
        for k in range(T):
            ctx = policy.render_context(task.obs(k), skill=skill)
            tokens = torch.tensor([ctx], dtype=torch.long, device=device)
            logits, hidden = policy(tokens, hidden)
            logp_all = F.log_softmax(logits[0, -1], dim=-1)
            logp_t.append(float(logp_all[actions[k]].item()))
    reward = task.terminal_reward(actions)
    return dict(actions=actions, logp_student=logp_s, logp_teacher=logp_t, reward=reward)


def rollout_group(policy, task, device, group_size: int, temperature: float = 1.0):
    """Sample G trajectories for one task (GRPO group)."""
    return [rollout(policy, task, device, temperature) for _ in range(group_size)]


# ----------------------------------------------------------------------------
# AgentOPSD credit construction (Eqs. 1-11).
# ----------------------------------------------------------------------------


def grpo_advantage(rewards):
    """Eq. 1-2: group-normalized outcome advantage (one scalar per trajectory)."""
    r = np.array(rewards, dtype=np.float64)
    adv = (r - r.mean()) / (r.std() + 1e-4)
    return adv


def group_prior(rewards, eps0: float = 1e-4) -> float:
    """B0 = clip(group success rate, eps0, 1-eps0) — the belief prior (Eq. 8)."""
    r = np.array(rewards, dtype=np.float64)
    return float(np.clip(r.mean(), eps0, 1.0 - eps0))


def belief_and_credit(turn_evidence, B0: float, gamma: float):
    """Eqs. 8-9: recursive log-odds belief and marginal credit dB_k.

    turn_evidence : list/array of e_k (sum of per-token gaps in turn k)
    Returns B_list (B_1..B_K), dB_list, c_list.
    """
    c = 0.0
    logit0 = math.log(B0 / (1.0 - B0))
    B_prev = B0
    Bs, dBs, cs = [], [], []
    for e in turn_evidence:
        c = gamma * c + e
        B = 1.0 / (1.0 + math.exp(-logit0 - c))
        Bs.append(B)
        dBs.append(B - B_prev)
        B_prev = B
        cs.append(c)
    return Bs, dBs, cs


def agentopsd_advantages(traj, A_seq: float, gamma: float, lam: float, b: float,
                         eps0: float = 1e-4):
    """Full AgentOPSD reshaping for one trajectory (Eqs. 6-11).

    traj : dict from rollout() — uses logp_student, logp_teacher, B0 (set by
           the caller to the group prior).
    Returns per-turn advantages A~_k and diagnostics (B, dB, q, w).
    """
    # Eq. 6: turn evidence = sum over action tokens of teacher-student gap.
    # One action token per turn => e_k = logp_teacher - logp_student.
    e = np.array(traj["logp_teacher"], dtype=np.float64) \
        - np.array(traj["logp_student"], dtype=np.float64)

    B0 = traj["B0"]
    Bs, dBs, _ = belief_and_credit(e, B0, gamma)

    # Eq. 10: outcome-aligned credit.
    q = np.sign(A_seq) * np.array(dBs, dtype=np.float64)

    # Eq. 11: standardize within trajectory, bound, blend. If q is constant
    # (degenerate group / zero evidence), z=0 => w=1 => plain GRPO advantage.
    sigma = q.std()
    z = (q - q.mean()) / (sigma + eps0)
    w = np.clip(1.0 + b * z, 1.0 - b, 1.0 + b)
    A_tilde = A_seq * ((1.0 - lam) + lam * w)

    return A_tilde, dict(e=e, B=Bs, dB=dBs, q=q, w=w)


# ----------------------------------------------------------------------------
# Policy update (Eq. 12): clipped surrogate on re-scored action tokens.
# ----------------------------------------------------------------------------


def surrogate_loss(policy, task, traj, adv_turns, device, temperature: float = 1.0,
                   clip_low: float = 0.2, clip_high: float = 0.24,
                   dual_clip: float = 3.0, entropy_coef: float = 0.0):
    """Eq. 12 for one trajectory. Returns (loss, entropy, kl_approx)."""
    policy.train()
    actions = traj["actions"]
    old_logp = torch.tensor(traj["logp_student"], dtype=torch.float64, device=device)
    A = torch.tensor(np.asarray(adv_turns, dtype=np.float64), dtype=torch.float64,
                     device=device)

    hidden = policy.initial_state(1, device)
    new_logp_list, ent_list = [], []
    for k in range(task.T):
        ctx = policy.render_context(task.obs(k), skill=None)
        tokens = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, hidden = policy(tokens, hidden)
        logp_all = F.log_softmax(logits[0, -1].double() / temperature, dim=-1)
        new_logp_list.append(logp_all[actions[k]])
        ent_list.append(-(logp_all.exp() * logp_all).sum())

    new_logp = torch.stack(new_logp_list)            # (K,)
    entropy = torch.stack(ent_list).mean()
    ratio = torch.exp(new_logp - old_logp)           # (K,)

    sur1 = ratio * A
    sur2 = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * A
    sur = torch.minimum(sur1, sur2)
    if dual_clip is not None:
        # DAPO-style dual clip: for negative advantages, floor the surrogate.
        neg = A < 0
        sur = torch.where(neg, torch.maximum(sur, dual_clip * A), sur)

    loss = -(sur.mean()) - entropy_coef * entropy
    with torch.no_grad():
        kl = (ratio - 1.0 - torch.log(ratio + 1e-12)).mean()
    return loss, entropy, kl


# ----------------------------------------------------------------------------
# Credit-localization metric (possible only on the synthetic oracle).
# ----------------------------------------------------------------------------


def spearman(a, b):
    """Rank correlation without scipy."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])
