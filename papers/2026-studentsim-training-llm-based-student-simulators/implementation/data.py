"""Synthetic learner world for the StudentSim miniature.

Re-implementation companion to: Ke Yang et al., "StudentSim: Training
LLM-based Student Simulators", arXiv:2609.01591 (2026).
https://arxiv.org/abs/2609.01591

This module is the GROUND TRUTH side of the experiment: a population of
synthetic learners with latent knowledge states, misconception affinities and
coachability, plus a problem bank. Every observable record is generated from
these latents. The simulator "brains" in model.py never see the latents;
they only see the records, exactly like the paper's simulators only see
learner records.

The domain is a miniature of the paper's math benchmark: four-way
multiple-choice problems, each distractor tagged with a misconception type,
three tutor guidance modes (error remediation / conceptual / Socratic).

Stdlib only. Deterministic under a fixed seed.
"""

import math
import random

# ---------------------------------------------------------------------------
# World constants
# ---------------------------------------------------------------------------

SKILLS = [
    "fractions", "negatives", "rates", "exponents",
    "equations", "percents", "geometry", "probability",
]

MODES = ["error_remediation", "conceptual", "socratic"]

# Ground-truth guidance-mode parameters (unknown to the brains):
#   alpha = directness: how strongly guidance carries the correction
#   delta = mastery leverage: how much uptake requires the student to
#           derive the answer themselves (the Socratic/ZPD gate)
MODE_ALPHA = {"error_remediation": 2.0, "conceptual": 1.4, "socratic": 0.9}
MODE_DELTA = {"error_remediation": 0.0, "conceptual": 0.8, "socratic": 1.6}
REV_INTERCEPT = 1.5           # kappa: global uptake calibration (the paper's
                              # domains all sit in the regime where tutor
                              # guidance moves learners far more than
                              # unassisted re-answering does)
DIFF_PENALTY = 1.0            # weight on the mastery gap z = m - b

# Population ability distribution
ABILITY_MEAN = 0.30
ABILITY_SD = 0.90
SKILL_NOISE_SD = 0.50         # per-skill deviation around personal ability
SKILL_OFFSET = {              # population-level skill ease (logits)
    "fractions": 0.1, "negatives": -0.2, "rates": -0.3, "exponents": -0.1,
    "equations": 0.2, "percents": 0.4, "geometry": 0.0, "probability": -0.4,
}

MISCONCEPTIONS = [
    "sign_flip", "operation_swap", "off_by_one",
    "unit_confusion", "reciprocal", "arithmetic_slip",
]
# Population misconception profile: what students at large get wrong
POP_PROFILE = {
    "sign_flip": 0.28, "operation_swap": 0.22, "off_by_one": 0.18,
    "unit_confusion": 0.12, "reciprocal": 0.10, "arithmetic_slip": 0.08,
}
PROFILE_CONCENTRATION = 6.0   # Dirichlet concentration around POP_PROFILE
COACH_MEAN = 0.75             # population coachability
COACH_SD = 0.35

DIFFICULTY_SD = 1.10          # per-problem difficulty noise around 0

# Guidance surface text (cosmetic: the brains consume (mode), the transcript
# consumes the text; the paper's chess/math guidance is LLM-authored under
# fixed style templates with m* fixed in advance — same structure here)
GUIDANCE_TEMPLATES = {
    "error_remediation": (
        "TUTOR: The option you marked is not correct. The right answer is "
        "{target}. Work through the {skill} step again and you will see why "
        "{target} is the consistent choice."
    ),
    "conceptual": (
        "TUTOR: Before re-answering, recall the {skill} principle: apply the "
        "operation to both sides, then simplify. Which of the four options "
        "is consistent with that principle?"
    ),
    "socratic": (
        "TUTOR: Three questions. What does the {skill} rule say should happen "
        "first? Which option survives if you check it against a simple case? "
        "Which one fails?"
    ),
}


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0
    e = math.exp(x) if x < 700 else float("inf")
    return e / (1.0 + e)


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Ground-truth objects
# ---------------------------------------------------------------------------

class Problem:
    """A four-way MC problem. Option 0..3; one correct; distractors tagged."""

    __slots__ = ("pid", "skill", "difficulty", "correct_idx", "misc_by_option")

    def __init__(self, pid, skill, difficulty, correct_idx, misc_by_option):
        self.pid = pid
        self.skill = skill
        self.difficulty = difficulty
        self.correct_idx = correct_idx
        self.misc_by_option = misc_by_option  # idx -> misconception (distractors only)


class Student:
    """Latent learner: per-skill mastery logits, misconception affinity,
    coachability. Unknown to the simulator brains."""

    __slots__ = ("sid", "ability", "mastery", "profile", "coachability")

    def __init__(self, sid, ability, mastery, profile, coachability):
        self.sid = sid
        self.ability = ability
        self.mastery = mastery          # skill -> logit
        self.profile = profile          # misconception -> prob
        self.coachability = coachability

    # -- ground-truth response model ------------------------------------
    def p_correct(self, problem):
        return sigmoid(self.mastery[problem.skill] - problem.difficulty)

    def answer(self, problem, rng):
        """Sample the student's answer; returns option index."""
        if rng.random() < self.p_correct(problem):
            return problem.correct_idx
        distractors = [i for i in range(4) if i != problem.correct_idx]
        weights = [self.profile[problem.misc_by_option[i]] for i in distractors]
        return rng.choices(distractors, weights=weights, k=1)[0]

    # -- ground-truth guidance-uptake model ------------------------------
    def mastery_gap(self, problem):
        return self.mastery[problem.skill] - problem.difficulty

    def p_revise(self, problem, mode):
        a = MODE_ALPHA[mode]
        d = MODE_DELTA[mode]
        z = self.mastery_gap(problem)
        return sigmoid(REV_INTERCEPT + a * self.coachability
                       + d * DIFF_PENALTY * z)

    def revises(self, problem, mode, rng):
        return rng.random() < self.p_revise(problem, mode)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _dirichlet(mean_probs, concentration, rng):
    alphas = [concentration * p for p in mean_probs]
    draws = [rng.gammavariate(max(a, 1e-9), 1.0) for a in alphas]
    s = sum(draws)
    return [d / s for d in draws]


def make_problem_bank(n_problems, rng):
    bank = []
    for pid in range(n_problems):
        skill = SKILLS[pid % len(SKILLS)]
        difficulty = rng.gauss(0.0, DIFFICULTY_SD)
        correct_idx = rng.randrange(4)
        misc = rng.sample(MISCONCEPTIONS, 3)
        misc_by_option = {}
        k = 0
        for i in range(4):
            if i != correct_idx:
                misc_by_option[i] = misc[k]
                k += 1
        bank.append(Problem(pid, skill, difficulty, correct_idx, misc_by_option))
    return bank


def make_students(n_students, rng, sid_prefix="s"):
    students = []
    for j in range(n_students):
        ability = rng.gauss(ABILITY_MEAN, ABILITY_SD)
        mastery = {
            s: ability + SKILL_OFFSET[s] + rng.gauss(0.0, SKILL_NOISE_SD)
            for s in SKILLS
        }
        prof = _dirichlet([POP_PROFILE[m] for m in MISCONCEPTIONS],
                          PROFILE_CONCENTRATION, rng)
        profile = dict(zip(MISCONCEPTIONS, prof))
        coach = rng.gauss(COACH_MEAN, COACH_SD)
        students.append(Student(f"{sid_prefix}{j:03d}", ability, mastery,
                                profile, coach))
    return students


def gen_single_records(student, problems, rng):
    """[(problem, chosen_idx)] — observable single-turn records S_i."""
    out = []
    for p in problems:
        out.append((p, student.answer(p, rng)))
    return out


def gen_multi_records(student, wrongs, rng):
    """For (problem, chosen) wrong answers: pick a mode uniformly and sample
    the student's actual revision outcome. Observable multi-turn records."""
    out = []
    for p, chosen in wrongs:
        mode = MODES[rng.randrange(len(MODES))]
        revised = student.revises(p, mode, rng)
        out.append({"problem": p, "chosen": chosen, "mode": mode,
                    "revised": revised})
    return out


# ---------------------------------------------------------------------------
# Dataset assembly (mirrors the paper's protocol shape)
# ---------------------------------------------------------------------------

def make_dataset(seed=11,
                 n_pool_students=200,
                 pool_records_per_student=120,
                 n_roster=15,
                 roster_train=90,
                 roster_heldout_single=66,
                 roster_heldout_multi_per_mode=15,
                 n_problems=1200):
    """Build the full dataset dict.

    Structure (math-domain miniature of Table 1 in the paper):
      - pool: many students, moderate records each -> Stage-1 pooled training
      - roster: the Stage-2 students (per-student specialization + eval)
      - per roster student: train_single / train_multi (specialization data),
        heldout_single (S_i, fidelity eval), heldout_multi (T_i, evenly
        across modes, responsiveness eval), plus heldout_single_answers for
        the ground-truth comparison.
    """
    rng = random.Random(seed)

    bank = make_problem_bank(n_problems, rng)
    pool_students = make_students(n_pool_students, rng, sid_prefix="pool")
    roster_students = make_students(n_roster, rng, sid_prefix="stu")

    # ---- Stage-1 pooled corpus ----
    pool = []
    for st in pool_students:
        probs = rng.sample(bank, pool_records_per_student)
        singles = gen_single_records(st, probs, rng)
        wrongs = [(p, c) for (p, c) in singles if c != p.correct_idx]
        # each wrong answer yields one multi-turn record (mode sampled)
        multis = gen_multi_records(st, wrongs, rng)
        pool.append({"student": st, "single": singles, "multi": multis})

    # ---- Stage-2 roster ----
    roster = {}
    for st in roster_students:
        probs = rng.sample(bank, roster_train + 8 * roster_heldout_single)
        train_probs = probs[:roster_train]
        eval_probs = probs[roster_train:]
        train_single = gen_single_records(st, train_probs, rng)
        train_wrongs = [(p, c) for (p, c) in train_single
                        if c != p.correct_idx]
        train_multi = gen_multi_records(st, train_wrongs, rng)

        # held-out single-turn: fixed count per student (equal-weight F)
        held_single = gen_single_records(
            st, eval_probs[:roster_heldout_single], rng)
        # held-out multi-turn: from the remaining wrong answers, cap per mode
        rest = eval_probs[roster_heldout_single:]
        rest_answers = gen_single_records(st, rest, rng)
        rest_wrongs = [(p, c) for (p, c) in rest_answers
                       if c != p.correct_idx]
        rng.shuffle(rest_wrongs)
        per_mode = {m: [] for m in MODES}
        for p, c in rest_wrongs:
            for m in MODES:
                if len(per_mode[m]) < roster_heldout_multi_per_mode:
                    per_mode[m].append((p, c))
                    break
        held_multi = []
        for m in MODES:
            for p, c in per_mode[m]:
                held_multi.append({
                    "problem": p, "chosen": c, "mode": m, "revised": None,
                })

        roster[st.sid] = {
            "student": st,
            "train_single": train_single,
            "train_multi": train_multi,
            "held_single": held_single,
            "held_multi": held_multi,
        }

    return {"bank": bank, "pool": pool, "roster": roster,
            "seed": seed, "modes": MODES, "skills": SKILLS,
            "misconceptions": MISCONCEPTIONS}


if __name__ == "__main__":
    ds = make_dataset(seed=11, n_pool_students=30, n_roster=5,
                      pool_records_per_student=40, roster_train=60,
                      roster_heldout_single=20,
                      roster_heldout_multi_per_mode=5,
                      n_problems=400)
    n_wrongs = sum(1 for sid in ds["roster"]
                   for (p, c) in ds["roster"][sid]["train_single"]
                   if c != p.correct_idx)
    print(f"smoke: pool={len(ds['pool'])} roster={len(ds['roster'])} "
          f"train wrongs (roster, total)={n_wrongs}")
    sid = next(iter(ds["roster"]))
    r = ds["roster"][sid]
    print(f"{sid}: |S_train|={len(r['train_single'])} "
          f"|T_train|={len(r['train_multi'])} "
          f"|S_test|={len(r['held_single'])} "
          f"|T_test|={len(r['held_multi'])}")
