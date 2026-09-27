"""Data: a toy longitudinal "medication companion" simulation.

Implements the environment side of the Combodied-Agents loop
(Eqs. 4-6 of the paper): a simulated person with a *latent* human state,
an uncertain response to agent interventions, and outcomes that feed back.

Paper: "Combodied Agents: a New Paradigm of Human-Centric Agentic AI"
       Ding et al., 2026. arXiv:2608.10915

The ground-truth response function `true_response` is the *hidden* mechanics
of the simulated person. The agent never calls it; the agent's Personal
World Model must *learn* an approximation of it from synthetic training
data, and the policy must act under the belief net's uncertain state
estimate. The whole point (result (a) in the README) is that the same
action helps or harms depending on the latent cause of a missed dose.
"""

import math
import random

import torch

CAUSE_NAMES = ["forgot", "confused", "refused"]
ACTION_NAMES = ["none", "remind", "clarify", "coach", "escalate"]
N_CAUSES = 3
N_ACTIONS = 5

# feature names observed by the agent on a missed-dose day (noisy indicators)
FEATURE_NAMES = ["busy_day", "disorientation", "side_effect_complaint"]

# mean of each feature under each cause (rows: causes, cols: features)
FEATURE_MEANS = [
    [0.80, 0.20, 0.10],  # forgot    -> busy calendar
    [0.30, 0.85, 0.20],  # confused  -> disorientation signs
    [0.20, 0.15, 0.80],  # refused   -> complained about side effects
]
FEATURE_NOISE = 0.12


def true_response(cause_idx, action_idx, habit=1.0):
    """Hidden ground truth: P(take) and utility deltas for (cause, action).

    habit: nudge habituation in [0.5, 1] -- repeated reminders wear out.
    Returns (p_take, d_rel, d_auto, d_cap, d_miss_prob) where d_* are
    per-episode changes to relationship quality, autonomy (self-capability),
    capability growth, and the person's future miss probability multiplier.
    """
    c, a = cause_idx, action_idx
    if a == 0:  # none / non-intervention
        return (0.05, 0.0, 0.0, 0.0, 1.0)
    if a == 1:  # remind (a nudge)
        p = {0: 0.85 * habit, 1: 0.25, 2: 0.10}[c]
        d_rel = -0.08 if c == 2 else -0.02
        return (p, d_rel, -0.01, 0.0, 1.0)
    if a == 2:  # clarify (call and explain -- the "clarification" action)
        p = {0: 0.50, 1: 0.80, 2: 0.20}[c]
        return (p, 0.01, -0.005, 0.0, 1.0)
    if a == 3:  # coach (reflective support, builds capability)
        p = {0: 0.35, 1: 0.30, 2: 0.55}[c]
        return (p, 0.03, 0.015, 0.004, 0.995)
    # a == 4: escalate (call caregiver -- safe but substitutive)
    return (0.90, -0.05, -0.03, 0.0, 1.0)


def observe_features(cause_idx, gen):
    """Noisy event features the agent sees on a missed-dose day."""
    return [
        min(1.0, max(0.0, gen.gauss(FEATURE_MEANS[cause_idx][j], FEATURE_NOISE)))
        for j in range(3)
    ]


class PersonSim:
    """One simulated older adult over T days (hidden state, Eq. 4/5)."""

    def __init__(self, pid, gen):
        self.pid = pid
        self.gen = gen
        # latent person-level parameters (never shown to the agent directly)
        self.miss_prob = min(0.6, random.Random(gen.random()).betavariate(2.0, 8.0) * 3)
        self.miss_prob = min(0.6, max(0.08, self.miss_prob))
        w = [gen.gammavariate(6.0, 1.0), gen.gammavariate(2.0, 1.0), gen.gammavariate(2.5, 1.0)]
        s = sum(w)
        self.cause_probs = [x / s for x in w]
        self.habit = 1.0            # nudge habituation (decays with reminders)
        self.rel = 1.0              # relationship quality      (utility axis)
        self.self_cap = 1.0         # autonomy / self-capability (utility axis)
        self.consecutive_untaken = 0  # safety signal
        self.n_miss_days = 0

    def sample_cause(self):
        r = self.gen.random()
        acc = 0.0
        for i, p in enumerate(self.cause_probs):
            acc += p
            if r <= acc:
                return i
        return len(self.cause_probs) - 1

    def step(self, cause_idx, action_idx):
        """Apply an intervention on a missed-dose day; return outcome dict."""
        p_take, d_rel, d_auto, d_cap, d_miss = true_response(cause_idx, action_idx, self.habit)
        took = self.gen.random() < p_take
        self.rel = min(1.0, max(0.0, self.rel + d_rel))
        self.self_cap = min(1.0, max(0.0, self.self_cap + d_auto + d_cap))
        self.miss_prob = min(0.9, max(0.05, self.miss_prob * d_miss))
        if action_idx == 1:  # reminders cause habituation
            self.habit = max(0.5, self.habit * 0.95)
        else:                # and it slowly recovers otherwise
            self.habit = min(1.0, self.habit + 0.01)
        if took:
            self.consecutive_untaken = 0
        else:
            self.consecutive_untaken += 1
        return {
            "took": took,
            "p_take": p_take,
            "d_rel": d_rel,
            "d_auto": d_auto,
        }

    def daily_event(self):
        """True iff today is a (would-be) missed dose -> agent decision point."""
        return self.gen.random() < self.miss_prob


# ---------------------------------------------------------------------------
# Synthetic training datasets
# ---------------------------------------------------------------------------

def make_belief_dataset(n=4000, seed=0):
    """(features -> latent cause) pairs for training the belief net q(Z|D)."""
    gen = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        c = gen.randrange(N_CAUSES)
        X.append(observe_features(c, gen))
        y.append(c)
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))


def make_pwm_dataset(n=6000, seed=1):
    """(belief posterior, memory stats, person state, action -> outcome).

    Belief posteriors are sampled as peaked-but-uncertain Dirichlet-style
    vectors around the true cause (mimicking what the trained belief net
    outputs on noisy features). Memory stats carry an imperfect proxy of
    the person's hidden habituation. Labels come from `true_response`.
    """
    gen = random.Random(seed)
    IN, A, Y = [], [], []
    for _ in range(n):
        c = gen.randrange(N_CAUSES)
        # peaked noisy posterior: exp(log-target + noise), normalized
        logz = [math.log(0.08)] * N_CAUSES
        logz[c] += gen.gauss(2.2, 0.6)
        z = [math.exp(v) for v in logz]
        zs = sum(z)
        z = [v / zs for v in z]
        habit = gen.uniform(0.6, 1.0)
        # memory proxy stats per action (success-rate estimates, noisy)
        mem = []
        for a in range(N_ACTIONS):
            if a == 1:
                base = 0.85 * habit
            else:
                base = true_response(c, a)[0] if a != 0 else 0.05
            mem.append(min(1.0, max(0.0, gen.gauss(base, 0.08))))
        rel = gen.uniform(0.5, 1.0)
        cap = gen.uniform(0.6, 1.0)
        a = gen.randrange(N_ACTIONS)
        p_take, d_rel, d_auto, _, _ = true_response(c, a, habit)
        took = 1 if gen.random() < p_take else 0
        IN.append(z + mem + [rel, cap])
        A.append(a)
        Y.append([took, d_rel, d_auto])
    return (torch.tensor(IN, dtype=torch.float32),
            torch.tensor(A, dtype=torch.long),
            torch.tensor(Y, dtype=torch.float32))
