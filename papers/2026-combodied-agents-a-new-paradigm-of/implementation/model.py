"""Model: belief net q(Z|D) + Personal World Model (PWM) + policies.

From-scratch implementation of the core Combodied-Agents components
(Eqs. 2, 7, 8 of the paper) at toy scale:

  * BeliefNet    -- Eq. 2: posterior over latent human state (which of
                    {forgot, confused, refused} caused the missed dose)
                    from noisy event features.
  * PWM          -- Eq. 7: given (belief, memory stats, person state,
                    action) predict the distribution over outcomes:
                    P(took dose) plus utility deltas (relationship,
                    autonomy/capability). Small MLP ensemble heads for a
                    crude uncertainty estimate.
  * Policies     -- Eq. 8: hard admissibility filter -> Pareto/lexicographic
                    selection. We compare:
                      - NaivePolicy   : argmax_a predicted P(take) alone
                                        (the "task-metric optimizer").
                      - CombodiedPolicy: admissibility filter + lexicographic
                                        vector utility (safety > consent >
                                        autonomy > benefit), using the
                                        intervention-response memory.

Paper: "Combodied Agents: a New Paradigm of Human-Centric Agentic AI"
       Ding et al., 2026. arXiv:2608.10915
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import N_ACTIONS, ACTION_NAMES


# ---------------------------------------------------------------------------
# Eq. 2: latent human-state posterior
# ---------------------------------------------------------------------------

class BeliefNet(nn.Module):
    """q(Z_t | D_<=t): event features -> posterior over missed-dose cause."""

    def __init__(self, n_features=3, n_states=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_states),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)  # a distribution, never a point


# ---------------------------------------------------------------------------
# Eq. 7: Personal World Model  p(outcomes | belief, memory, state, action)
# ---------------------------------------------------------------------------

class PWM(nn.Module):
    """Predicts [P(took), d_rel, d_auto] from (belief | memory | state | action).

    Input: 3 belief probs + 5 memory success-rate stats + rel + cap = 10,
    plus a one-hot action -> 15 dims.
    Heads: take-logit (BCE), utility deltas (Huber). An ensemble of K
    models gives a crude epistemic-uncertainty signal (mean pairwise
    disagreement on P(take)) used by the admissibility filter.
    """

    N_IN = 3 + N_ACTIONS + 2

    def __init__(self, hidden=48):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(self.N_IN + N_ACTIONS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.take_head = nn.Linear(hidden, 1)
        self.util_head = nn.Linear(hidden, 2)  # d_rel, d_auto

    def forward(self, x, a_onehot):
        h = self.trunk(torch.cat([x, a_onehot], dim=-1))
        take_logit = self.take_head(h).squeeze(-1)
        d_util = self.util_head(h)
        return take_logit, d_util

    @torch.no_grad()
    def predict_action(self, x1, action_idx):
        """x1: single input vector (10 dims). Returns (p_take, d_rel, d_auto)."""
        self.eval()
        x = x1.unsqueeze(0)
        a = torch.zeros(1, N_ACTIONS)
        a[0, action_idx] = 1.0
        take_logit, d = self.forward(x, a)
        p = torch.sigmoid(take_logit)[0].item()
        d_rel, d_auto = d[0].tolist()
        return p, d_rel, d_auto


class PWMEnsemble:
    """K PWMs; the std over members' P(take) is the uncertainty signal."""

    def __init__(self, k=3, hidden=48, seed=0):
        self.members = [PWM(hidden) for _ in range(k)]
        for i, m in enumerate(self.members):
            for p in m.parameters():  # decorrelate inits
                nn.init.normal_(p, mean=0.0, std=0.3 / (1 + i))

    def parameters(self):
        for m in self.members:
            yield from m.parameters()

    def train_mode(self):
        for m in self.members:
            m.train()

    def losses(self, x, a_onehot, y):
        total = None
        for m in self.members:
            take_logit, d = m(x, a_onehot)
            l = (F.binary_cross_entropy_with_logits(take_logit, y[:, 0])
                 + F.smooth_l1_loss(d, y[:, 1:]))
            total = l if total is None else total + l
        return total

    @torch.no_grad()
    def predict_action(self, x1, action_idx):
        """Returns (mean p_take, mean d_rel, mean d_auto, std p_take)."""
        ps, rels, autos = [], [], []
        a = torch.zeros(1, N_ACTIONS)
        a[0, action_idx] = 1.0
        for m in self.members:
            take_logit, d = m(x1.unsqueeze(0), a)
            ps.append(torch.sigmoid(take_logit)[0].item())
            rels.append(d[0, 0].item())
            autos.append(d[0, 1].item())
        mean_p = sum(ps) / len(ps)
        var = sum((p - mean_p) ** 2 for p in ps) / len(ps)
        return mean_p, sum(rels) / len(rels), sum(autos) / len(autos), var ** 0.5


# ---------------------------------------------------------------------------
# Eq. 6: memory (longitudinal, with intervention-response component)
# ---------------------------------------------------------------------------

class InterventionMemory:
    """The paper's intervention-response memory, as a per-action outcome table.

    Stores, for each action, an exponentially-decayed estimate of P(took |
    action) from *this person's* realized outcomes, plus usage counts for
    habituation-aware admissibility. This is the component that lets the
    policy adapt to the individual instead of the population average.
    """

    def __init__(self, decay=0.85):
        self.decay = decay
        self.p_take = [0.5] * N_ACTIONS          # running estimate per action
        self.uses = [0] * N_ACTIONS
        self.last_day_used = [-1] * N_ACTIONS
        self.recent_reminds = 0                   # for anti-nagging rule

    def update(self, action_idx, took, day):
        self.uses[action_idx] += 1
        self.last_day_used[action_idx] = day
        y = 1.0 if took else 0.0
        # interpolate toward realized outcome, faster early (uncertainty shrink)
        n = self.uses[action_idx]
        w = max(1.0 / n, self.decay)
        self.p_take[action_idx] = (1 - w) * self.p_take[action_idx] + w * y
        if action_idx == 1:
            self.recent_reminds += 1
        elif self.recent_reminds > 0:
            self.recent_reminds = max(0, self.recent_reminds - 1)

    def stats(self):
        return list(self.p_take)


# ---------------------------------------------------------------------------
# Eq. 8: policies -- admissible set first, then selection
# ---------------------------------------------------------------------------

class NaivePolicy:
    """Baseline: argmax predicted P(take). Ignores utility, consent, memory."""

    def __init__(self, pwm):
        self.pwm = pwm

    def choose(self, x1, mem, day, consecutive_untaken):
        best_a, best_p = 0, -1.0
        for a in range(N_ACTIONS):
            p, _, _, _ = self.pwm.predict_action(x1, a)
            if p > best_p:
                best_p, best_a = p, a
        return best_a, {}


class CombodiedPolicy:
    """Hard admissibility filter -> weighted vector-utility selection.

    Admissibility (all must hold; non-intervention/clarify always in):
      A1 safety    : if >=3 consecutive untaken doses, only escalate
                     (+none) is admissible -- a hard medical-safety bound.
      A2 uncertainty: if ensemble disagreement on P(take) > 0.25, drop
                     actions whose effect we cannot predict confidently.
      A3 anti-nag  : if >=3 reminders were used recently, drop remind.
      A4 scope     : escalate (call caregiver) is out of scope for routine
                     optimization -- it is admissible only under A1, or when
                     no in-scope action predicts P(take) >= 0.45. Escalation
                     is an emergency action, not a default.

    Selection (the paper leaves the rule open "by design"; we instantiate
    it as a user-inspectable weighted utility -- the implicit exchange rate
    "how much adherence is a unit of autonomy worth" is exactly what a user
    should be able to read and edit):

        U(a) = P_take(a) + 5 * d_auto(a) + 4 * d_rel(a)

    With the ground-truth effect sizes this makes a -0.03 autonomy cost
    (escalate) worth ~0.15 adherence -- consent/autonomy are expensive to
    trade away, which is the paper's central policy claim. 'none' gets a
    small autonomy bonus (silence respects autonomy).
    """

    UNC_THRESH = 0.25
    W_AUTO, W_REL = 5.0, 4.0

    def __init__(self, pwm, belief_net):
        self.pwm = pwm
        self.belief = belief_net

    def choose(self, x1, mem, day, consecutive_untaken):
        adm = {0, 2}  # none, clarify always admissible
        preds = {}
        for a in range(N_ACTIONS):
            p, d_rel, d_auto, unc = self.pwm.predict_action(x1, a)
            preds[a] = (p, d_rel, d_auto, unc)
            if a in adm or a == 4:
                continue
            if unc <= self.UNC_THRESH:          # A2: uncertainty gate
                adm.add(a)
        # A4: escalation scope -- only if no in-scope action works well
        in_scope_best = max(preds[a][0] for a in adm)
        if in_scope_best < 0.45 or consecutive_untaken >= 3:
            adm.add(4)
        # A1: safety override -- emergency trumps everything
        if consecutive_untaken >= 3:
            adm &= {0, 4}
        # A3: anti-nagging
        if mem.recent_reminds >= 3 and 1 in adm:
            adm.discard(1)

        def score(a):
            p, d_rel, d_auto, _ = preds[a]
            if a == 0:
                d_auto += 0.01  # silence respects autonomy
            return p + self.W_AUTO * d_auto + self.W_REL * d_rel

        best = max(adm, key=score)
        info = {"admissible": sorted(adm), "scores": {a: round(score(a), 3) for a in adm}}
        return best, info


ACTION_LABEL = {i: n for i, n in enumerate(ACTION_NAMES)}
