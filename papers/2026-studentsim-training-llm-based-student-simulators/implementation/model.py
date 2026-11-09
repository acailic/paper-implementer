"""Simulator "brains" for the StudentSim miniature.

Re-implementation companion to: Ke Yang et al., "StudentSim: Training
LLM-based Student Simulators", arXiv:2609.01591 (2026).
https://arxiv.org/abs/2609.01591

A Brain is a deterministic probabilistic student simulator with the three
parts the paper's simulators must have:

  1. KNOWLEDGE-STATE TRACKING — an online per-skill mastery tracker updated
     from the student's response stream (the KT/state-tracking heritage).
  2. ANSWER GENERATION WITH ERROR PATTERNS — a four-way answer distribution
     combining the mastery estimate with a misconception profile (the
     individual behavioral signature F measures).
  3. GUIDANCE-RESPONSIVENESS PATHWAY — a per-mode revision model
     P(emit canonical m* | mode, mastery gap, coachability) (what R measures;
     the state-tracking family simply lacks this input pathway).

The three families of the paper are realized by switching inputs on/off:

  state_tracking : tracker + population error profile, NO guidance pathway
                   (Maia2 / knowledge-tracing analogue: F yes, R floor)
  role_play      : static profile descriptor + population profile, guidance
                   pathway with population coachability, no individual state
                   (prompted-LLM analogue: fluent in guidance, blind to the
                   individual -> low F)
  studentsim     : pooled Stage-1 parameters + per-student Stage-2
                   specialization (shrunk misconception profile + shrunk
                   coachability intercept) -- pooled-then-specialized

"Training" here is statistical estimation, not gradient descent on a neural
net: Stage 1 = maximum-likelihood estimation of population parameters from
the pooled corpus; Stage 2 = empirical-Bayes shrinkage of per-student
parameters toward the Stage-1 estimates. That is the honest CPU miniature of
"pooled LoRA pretraining, then per-student continuation".

Stdlib only.
"""

import math
import random

from data import sigmoid, logit, MODES, SKILLS, MISCONCEPTIONS

_RNG = random.Random(1234)   # seeded: the simulation loop is reproducible


# ---------------------------------------------------------------------------
# Knowledge-state tracking
# ---------------------------------------------------------------------------

class KnowledgeTracker:
    """Online per-skill mastery tracker.

    Sees, per response: the problem's skill, the (estimated) skill difficulty,
    and whether the answer was correct. Update is a clipped online logistic
    step:  m_hat[s] += lr * (y - sigmoid(m_hat[s] - b_hat[s])).

    The state-tracking family and StudentSim both use it (correctness-only
    input, like knowledge tracing). The role-play family does not track at
    all — its only conditioning is a static profile descriptor, which is the
    paper's point about prompted role-play.
    """

    def __init__(self, priors, lr=0.6):
        self.m = dict(priors)   # skill -> mastery logit
        self.lr = lr

    def mastery(self, skill):
        return self.m[skill]

    def observe(self, skill, b_hat, correct):
        p = sigmoid(self.m[skill] - b_hat)
        self.m[skill] += self.lr * ((1.0 if correct else 0.0) - p)


# ---------------------------------------------------------------------------
# A simulator brain
# ---------------------------------------------------------------------------

class Brain:
    def __init__(self, sid, kind, tracker, profile, b_hat_skill,
                 rev_intercepts=None, rev_slopes=None,
                 coach_shift=0.0, band_scalar=None, guidance=True,
                 b_item=None):
        self.sid = sid
        self.kind = kind                    # family label
        self.tracker = tracker              # KnowledgeTracker or None
        self.profile = profile              # misconception -> prob
        self.b_hat = b_hat_skill            # skill -> estimated difficulty
        self.b_item = b_item or {}          # pid -> estimated item difficulty
        self.rev_a = rev_intercepts or {}   # mode -> intercept a_g
        self.rev_d = rev_slopes or {}       # mode -> slope d_g on (m - b)
        self.coach_shift = coach_shift      # per-student intercept shift u_i
        self.band_scalar = band_scalar      # role-play static mastery scalar
        self.guidance = guidance            # False -> tau-blind (floor)

    # -- mastery estimate ------------------------------------------------
    def mastery_for(self, problem):
        if self.tracker is not None:
            return self.tracker.mastery(problem.skill)
        return self.band_scalar

    def difficulty_for(self, problem):
        # calibrated item difficulty when the arm's corpus covers the item,
        # else the skill-mean difficulty
        return self.b_item.get(problem.pid, self.b_hat[problem.skill])

    # -- part 2: answer generation with error patterns ---------------------
    def answer_dist(self, problem):
        """P over the 4 options: correct-mass from mastery, wrong-mass split
        over distractors by the misconception profile."""
        p_correct = sigmoid(self.mastery_for(problem)
                            - self.difficulty_for(problem))
        distractors = [i for i in range(4) if i != problem.correct_idx]
        w = [self.profile[problem.misc_by_option[i]] for i in distractors]
        s = sum(w) or 1.0
        dist = [0.0] * 4
        dist[problem.correct_idx] = p_correct
        for i, wi in zip(distractors, w):
            dist[i] = (1.0 - p_correct) * wi / s
        return dist

    def predict_answer(self, problem):
        dist = self.answer_dist(problem)
        return max(range(4), key=lambda i: dist[i])

    # -- part 3: guidance-update pathway ----------------------------------
    def p_revise(self, problem, mode):
        """P(simulator emits the canonical correction m* after guidance).

        With no guidance pathway (state-tracking family, or a guidance mode
        never seen in training) the simulator re-answers from state alone:
        it lands on m* only when m* is already its own top-1 answer — the
        paper's 'zero-instruction floor'."""
        if not self.guidance or mode not in self.rev_a:
            dist = self.answer_dist(problem)
            top = max(range(4), key=lambda i: dist[i])
            return 1.0 if top == problem.correct_idx else 0.0
        z = self.mastery_for(problem) - self.difficulty_for(problem)
        return sigmoid(self.rev_a[mode] + self.rev_d[mode] * z
                       + self.coach_shift)

    # -- the simulation loop step ------------------------------------------
    def respond_then_revise(self, problem, mode):
        """One simulated student-tutor exchange. Returns (answer, revised)."""
        answer = self.predict_answer(problem)
        if answer == problem.correct_idx:
            return answer, True   # already correct; nothing to remediate
        revised = _RNG.random() < self.p_revise(problem, mode)
        return answer, revised


# ---------------------------------------------------------------------------
# Stage 1: pooled estimation (the "pooled training" of the miniature)
# ---------------------------------------------------------------------------

def estimate_skill_difficulty(single_records):
    """b_hat per skill from observed correctness: choose b so that a
    zero-mastery student matches the observed correct rate. (Population item
    calibration — the shared 'what is hard in this domain' knowledge.)"""
    by_skill = {s: [0, 0] for s in SKILLS}
    for rec in single_records:
        problem, chosen = rec
        by_skill[problem.skill][1] += 1
        if chosen == problem.correct_idx:
            by_skill[problem.skill][0] += 1
    out = {}
    for s, (c, n) in by_skill.items():
        rate = c / n if n else 0.5
        out[s] = -logit(rate)   # m=0 prior: sigmoid(0 - b) = rate
    return out


def estimate_item_difficulty(single_records, skill_b_hat, shrink=6.0):
    """Per-item difficulty calibration from whatever response records the
    arm can see (the pooled Stage-1 corpus sees ~20 responses per item; a
    self-pooled arm sees ~1). Empirical item logit shrunk toward the skill
    mean — the population 'what is hard' knowledge that Stage 1 pools."""
    by_item = {}
    seen_skill = {}
    for problem, chosen in single_records:
        c, n = by_item.get(problem.pid, (0, 0))
        by_item[problem.pid] = (c + (1 if chosen == problem.correct_idx else 0),
                                n + 1)
        seen_skill[problem.pid] = problem.skill
    out = {}
    for pid, (c, n) in by_item.items():
        emp = -logit((c + 0.5) / (n + 1.0))     # Laplace-smoothed item logit
        prior = skill_b_hat[seen_skill[pid]]
        out[pid] = (n * emp + shrink * prior) / (n + shrink)
    return out


def estimate_pool_profile(multi_and_single_records):
    """Population misconception profile = pooled wrong-answer misconception
    counts (what is shared across students: the common mistakes)."""
    counts = {m: 0.0 for m in MISCONCEPTIONS}
    for problem, chosen in multi_and_single_records:
        if chosen != problem.correct_idx:
            counts[problem.misc_by_option[chosen]] += 1.0
    total = sum(counts.values()) or 1.0
    return {m: c / total for m, c in counts.items()}


def _fit_revision_model(multi_records, trackers, b_hat, b_item=None,
                        modes=MODES, iters=400, lr=0.35):
    """Logistic fit of  P(revised | mode, z) = sigmoid(a_g + d_g * z)  where
    z is the tracker's mastery gap for that student at the time of the record.
    This is the pooled guidance->revision pathway: 'given guidance in mode g,
    how does a student in this state update?' Pooling across students is what
    makes the per-mode parameters estimable from sparse per-student data."""
    b_item = b_item or {}
    a = {m: 0.0 for m in modes}
    d = {m: 0.0 for m in modes}
    feats = []
    for sid, recs in multi_records.items():
        tr = trackers[sid]
        for rec in recs:
            p = rec["problem"]
            b = b_item.get(p.pid, b_hat[p.skill])
            z = tr.mastery(p.skill) - b
            feats.append((rec["mode"], z, 1.0 if rec["revised"] else 0.0))
    if not feats:
        return a, d
    for _ in range(iters):
        ga = {m: 0.0 for m in modes}
        gd = {m: 0.0 for m in modes}
        for mode, z, y in feats:
            p = sigmoid(a[mode] + d[mode] * z)
            err = y - p
            ga[mode] += err
            gd[mode] += err * z
        for m in modes:
            a[m] += lr * ga[m] / len(feats)
            d[m] += lr * gd[m] / len(feats)
    return a, d


def _fallback_mode_params(a, d, missing_mode, trained_modes):
    """A mode absent from pooled training (rho=0 for it, or left out): the
    pathway has never seen it. We use the mean of the trained modes' params —
    the best an LLM-style generalist could do without mode-specific evidence."""
    if not trained_modes:
        return -1.0, 0.0
    return (sum(a[m] for m in trained_modes) / len(trained_modes),
            sum(d[m] for m in trained_modes) / len(trained_modes))


# ---------------------------------------------------------------------------
# Stage 2: per-student specialization (shrunk per-student estimates)
# ---------------------------------------------------------------------------

PROFILE_PRIOR_STRENGTH = 8.0   # tau_w: pseudo-counts of the pooled profile
COACH_PRIOR_STRENGTH = 15.0    # tau_u: pseudo-records of zero shift


def specialize_profile(train_single, pool_profile, use_pool_prior=True):
    """Per-student misconception profile: empirical counts shrunk toward the
    pooled profile (Stage 2 initializing from Stage 1). use_pool_prior=False
    shrinks toward uniform instead — the 'no pooled Stage 1' condition."""
    counts = {m: 0.0 for m in MISCONCEPTIONS}
    n_wrong = 0
    for problem, chosen in train_single:
        if chosen != problem.correct_idx:
            counts[problem.misc_by_option[chosen]] += 1.0
            n_wrong += 1
    prior = pool_profile if use_pool_prior else {
        m: 1.0 / len(MISCONCEPTIONS) for m in MISCONCEPTIONS}
    tau = PROFILE_PRIOR_STRENGTH
    return {m: (counts[m] + tau * prior[m]) / (n_wrong + tau)
            for m in MISCONCEPTIONS}, n_wrong


def solve_coach_shift(train_multi, tracker, b_hat, b_item, a, d):
    """Per-student coachability intercept u_i: moment-match the observed
    revision rate, then shrink toward 0 (the pooled student) by record count.
    Returns (u_shrunk, n_multi)."""
    if not train_multi:
        return 0.0, 0
    b_item = b_item or {}
    zs = [(rec["mode"],
           tracker.mastery(rec["problem"].skill)
           - b_item.get(rec["problem"].pid, b_hat[rec["problem"].skill]),
           1.0 if rec["revised"] else 0.0)
          for rec in train_multi]
    obs = sum(z[2] for z in zs) / len(zs)

    def mean_at(u):
        return sum(sigmoid(a[m] + d[m] * z + u) for m, z, _ in zs) / len(zs)

    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < obs:
            lo = mid
        else:
            hi = mid
    u_emp = 0.5 * (lo + hi)
    n = len(zs)
    u = (n / (n + COACH_PRIOR_STRENGTH)) * u_emp
    return u, n


# ---------------------------------------------------------------------------
# Building the families
# ---------------------------------------------------------------------------

def _pool_records_for_stage1(dataset, rho=1.0, drop_mode=None, rng=None):
    """Assemble the Stage-1 pooled corpus. rho = fraction of multi-turn
    (guidance) records sampled in (the paper's multi-turn ratio); drop_mode
    removes one guidance mode entirely (the paper's leave-one-out sweep)."""
    rng = rng or random.Random(0)
    singles, multis = [], {}
    for entry in dataset["pool"]:
        singles.extend(entry["single"])
        if rho > 0.0:
            keep = []
            for rec in entry["multi"]:
                if drop_mode and rec["mode"] == drop_mode:
                    continue
                keep.append(rec)
            if rho < 1.0:
                keep = rng.sample(keep, max(0, int(round(rho * len(keep)))))
            if keep:
                multis[entry["student"].sid] = keep
    return singles, multis


def build_stage1(dataset, rho=1.0, drop_mode=None, self_pool_sid=None,
                 rng=None):
    """Estimate all Stage-1 (pooled) parameters. self_pool_sid replays the
    paper's D.1 ablation: the 'pooled' corpus is ONE student's records (the
    specialization target) instead of many students' records."""
    if self_pool_sid is None:
        singles, multis = _pool_records_for_stage1(dataset, rho, drop_mode,
                                                   rng)
        # difficulty calibration first (needed by the tracker replay):
        # skill-level means, then per-item estimates shrunk toward them
        b_hat = estimate_skill_difficulty(singles)
        b_item = estimate_item_difficulty(singles, b_hat)
        # one tracker per pooled student, replayed over its full single-turn
        # stream so z matches the state a simulator would hold at eval time
        trackers = {}
        for entry in dataset["pool"]:
            if entry["student"].sid in multis:
                tr = KnowledgeTracker({s: 0.0 for s in SKILLS})
                for problem, chosen in entry["single"]:
                    tr.observe(problem.skill,
                               b_item.get(problem.pid,
                                          b_hat[problem.skill]),
                               chosen == problem.correct_idx)
                trackers[entry["student"].sid] = tr
        pool_profile = estimate_pool_profile(singles)
        a, d = _fit_revision_model(multis, trackers, b_hat, b_item)
    else:
        entry = dataset["roster"][self_pool_sid]
        singles = list(entry["train_single"])
        multis = {self_pool_sid: list(entry["train_multi"])}
        b_hat = estimate_skill_difficulty(singles)
        b_item = estimate_item_difficulty(singles, b_hat)
        tr = KnowledgeTracker({s: 0.0 for s in SKILLS})
        for problem, chosen in singles:
            tr.observe(problem.skill,
                       b_item.get(problem.pid, b_hat[problem.skill]),
                       chosen == problem.correct_idx)
        trackers = {self_pool_sid: tr}
        # no population to pool over: the 'shared' profile is just this
        # student's own empirical misconception mix
        counts = {m: 0.0 for m in MISCONCEPTIONS}
        for problem, chosen in singles:
            if chosen != problem.correct_idx:
                counts[problem.misc_by_option[chosen]] += 1.0
        tot = sum(counts.values()) or 1.0
        pool_profile = {m: c / tot for m, c in counts.items()}
        a, d = _fit_revision_model(multis, trackers, b_hat, b_item)
    # modes with no training data get the generalist fallback
    trained = [m for m in MODES
               if any(rec["mode"] == m
                      for recs in multis.values() for rec in recs)]
    for m in MODES:
        if m not in trained:
            a[m], d[m] = _fallback_mode_params(a, d, m, trained)
    return {"b_hat": b_hat, "b_item": b_item, "profile": pool_profile,
            "a": a, "d": d, "has_guidance": len(trained) > 0}


def _tracker_after_training(train_single, priors, b_hat, b_item=None,
                            use_tracker=True):
    """Replay the student's chronological training stream through the
    knowledge tracker (this is the inference-time state the simulator brings
    to held-out problems)."""
    b_item = b_item or {}
    tr = KnowledgeTracker(dict(priors)) if use_tracker else None
    if not use_tracker:
        return tr
    for problem, chosen in train_single:
        tr.observe(problem.skill,
                   b_item.get(problem.pid, b_hat[problem.skill]),
                   chosen == problem.correct_idx)
    return tr


def _band_scalar(train_single, b_hat, b_item=None):
    """Role-play conditioning: a static 'competence descriptor' — the single
    mastery level implied by the student's overall accuracy (the thin textual
    profile a prompted LLM receives)."""
    b_item = b_item or {}
    c = sum(1 for p, ch in train_single if ch == p.correct_idx)
    rate = c / len(train_single) if train_single else 0.5
    mean_b = sum(b_item.get(p.pid, b_hat[p.skill])
                 for p, _ in train_single) / max(len(train_single), 1)
    return logit(rate) + mean_b


def build_brain(dataset, sid, stage1, family):
    """Build one student's simulator of the given family from Stage-1 params.
    families:
      studentsim    — Stage-2 specialization on top of Stage-1 (pooled)
      pooled_only   — Stage 1 only, no specialization (lambda = 0)
      self_pool     — D.1 ablation: Stage 1 estimated from THIS student only
      role_play     — prompted-LLM analogue (static descriptor, no state)
      state_tracking— KT/Maia2 analogue (state yes, guidance pathway no)
    """
    entry = dataset["roster"][sid]
    train_single = entry["train_single"]
    train_multi = entry["train_multi"]

    if family == "self_pool":
        stage1 = build_stage1(dataset, rho=1.0, self_pool_sid=sid)

    b_hat = stage1["b_hat"]
    b_item = stage1["b_item"]
    pool_profile = stage1["profile"]

    if family == "role_play":
        tracker = None
        band = _band_scalar(train_single, b_hat, b_item)
        profile = pool_profile
        coach_shift = 0.0
    elif family == "state_tracking":
        tracker = _tracker_after_training(train_single, {s: 0.0 for s in SKILLS},
                                          b_hat, b_item, use_tracker=True)
        profile = pool_profile
        coach_shift = 0.0
    else:  # studentsim / pooled_only / self_pool
        tracker = _tracker_after_training(train_single, {s: 0.0 for s in SKILLS},
                                          b_hat, b_item, use_tracker=True)
        if family == "pooled_only":
            # Stage 1 only: population profile, no per-student signature
            profile = dict(pool_profile)
        elif family == "self_pool":
            profile, _ = specialize_profile(train_single, pool_profile,
                                            use_pool_prior=False)
        else:  # studentsim: Stage 2 on top of Stage 1
            profile, _ = specialize_profile(train_single, pool_profile,
                                            use_pool_prior=True)
        if family == "studentsim" or family == "self_pool":
            coach_shift, _ = solve_coach_shift(
                train_multi, _tracker_after_training(
                    train_single, {s: 0.0 for s in SKILLS}, b_hat, b_item),
                b_hat, b_item, stage1["a"], stage1["d"])
        else:
            coach_shift = 0.0

    rev_d = dict(stage1["d"])
    if family == "role_play":
        # the prompted role-player follows the guidance mode's directness,
        # but its only student channel is a static competence descriptor:
        # it cannot gate uptake on this student's per-skill state here and
        # now (the paper's 'thin conditioning channel')
        rev_d = {m: 0.0 for m in rev_d}

    return Brain(
        sid=sid, kind=family, tracker=tracker, profile=profile,
        b_hat_skill=b_hat, b_item=b_item,
        rev_intercepts=stage1["a"], rev_slopes=rev_d,
        coach_shift=coach_shift,
        band_scalar=(band if family == "role_play" else None),
        guidance=(family != "state_tracking" and stage1["has_guidance"]),
    )
