"""Experiment driver for the StudentSim miniature.

Re-implementation companion to: Ke Yang et al., "StudentSim: Training
LLM-based Student Simulators", arXiv:2609.01591 (2026).
https://arxiv.org/abs/2609.01591

Runs, on a synthetic learner population (data.py) and deterministic
probabilistic simulator brains (model.py):

  E1  main F / R table across the three simulator families of the paper
      (state-tracking, prompted role-play, pooled->specialized StudentSim),
      with per-mode R breakdown — the paper's Tables 2/3 + App. C.7 shape.
  E2  Stage ablation — the paper's App. D.1: pooled Stage-1 corpus vs
      one-student-repeated Stage-1 vs no-specialization.
  E3  guidance-data ratio (rho) sweep — the paper's App. D.3.
  E4  leave-one-guidance-mode-out — the paper's App. D.4 (Socratic).
  E5  the simulation loop end-to-end: tutor picks a guidance mode per wrong
      answer; the simulator predicts uptake (its reward); ground truth scores
      the choice. The paper's Sec. 6 tutor-RL reward in miniature.

F = top-1 exact match to the student's recorded held-out answer (right or
wrong — the paper's math-domain Eq. 6). R = expected exact-match rate to the
canonical correction m* after guidance (Eq. 7). R is reported as an
expectation over the revision probability, which keeps every number
deterministic; E5 additionally samples actual revisions in the loop.

Stdlib only. CPU. Deterministic under SEED.
"""

import json
import random
import statistics
from collections import Counter

from data import make_dataset, MODES, GUIDANCE_TEMPLATES
from model import build_stage1, build_brain

SEED = 11
FAMILIES = [
    ("state_tracking", "state-tracking (KT/Maia2 analogue)"),
    ("role_play", "prompted role-play (LLM analogue)"),
    ("studentsim", "StudentSim-mini (pooled -> specialized)"),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def fidelity(brain, entry, split=None):
    """F_i: top-1 match to the student's recorded answer on held-out S_i."""
    recs = entry["held_single"] if split is None else entry[split]
    hits, hits_when_correct, n_corr, hits_when_wrong, n_wrong = 0, 0, 0, 0, 0
    for problem, chosen in recs:
        pred = brain.predict_answer(problem)
        hit = pred == chosen
        hits += hit
        if chosen == problem.correct_idx:
            n_corr += 1
            hits_when_correct += hit
        else:
            n_wrong += 1
            hits_when_wrong += hit
    n = len(recs)
    return {
        "F": hits / n if n else 0.0,
        "F_correct_records": (hits_when_correct / n_corr) if n_corr else None,
        "F_wrong_records": (hits_when_wrong / n_wrong) if n_wrong else None,
    }


def responsiveness(brain, entry):
    """R_i: expected rate of emitting canonical m* after guidance on T_i."""
    recs = entry["held_multi"]
    per_mode = {m: [] for m in MODES}
    for rec in recs:
        per_mode[rec["mode"]].append(brain.p_revise(rec["problem"],
                                                    rec["mode"]))
    R = sum(sum(v) for v in per_mode.values()) / len(recs)
    return R, {m: (sum(v) / len(v) if v else None) for m, v in per_mode.items()}


def evaluate_family(dataset, stage1, family, label):
    per_F, per_R, per_mode = [], [], {m: [] for m in MODES}
    per_Fw, per_Fc = [], []
    for sid, entry in dataset["roster"].items():
        brain = build_brain(dataset, sid, stage1, family)
        f = fidelity(brain, entry)
        r, modes = responsiveness(brain, entry)
        per_F.append(f["F"])
        per_Fc.append(f["F_correct_records"])
        per_Fw.append(f["F_wrong_records"])
        per_R.append(r)
        for m in MODES:
            if modes[m] is not None:
                per_mode[m].append(modes[m])
    return {
        "family": family, "label": label,
        "F": statistics.mean(per_F),
        "R": statistics.mean(per_R),
        "F_on_correct": statistics.mean([x for x in per_Fc if x is not None]),
        "F_on_wrong": statistics.mean([x for x in per_Fw if x is not None]),
        "R_by_mode": {m: statistics.mean(v) if v else None
                      for m, v in per_mode.items()},
    }


# ---------------------------------------------------------------------------
# E5: the simulation loop + reward-model mini-experiment
# ---------------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def mode_selection_experiment(dataset, stage1, rng):
    """The paper's Sec. 6 in miniature. For each held-out (student, problem)
    needing remediation, a 'tutor' must choose a guidance mode. The reward
    model = a simulator brain's predicted uptake P_rev(mode). We compare
    policies by the TRUE expected uptake of the mode each policy selects,
    plus how well each simulator's predicted uptake correlates with the
    student's true uptake across all modes (reward-model quality)."""
    true_scores, pred_scores = {"role_play": [], "studentsim": []}, \
        {"role_play": [], "studentsim": []}
    policy_gain = {p: [] for p in ["random", "role_play", "studentsim",
                                   "oracle"]}
    mode_shares = {p: Counter() for p in policy_gain}
    n_records = 0
    for sid, entry in dataset["roster"].items():
        student = entry["student"]
        brains = {f: build_brain(dataset, sid, stage1, f)
                  for f in ["role_play", "studentsim"]}
        for rec in entry["held_multi"]:
            problem = rec["problem"]
            n_records += 1
            truth = {m: student.p_revise(problem, m) for m in MODES}
            preds = {f: {m: brains[f].p_revise(problem, m) for m in MODES}
                     for f in brains}
            for f in brains:
                for m in MODES:
                    true_scores[f].append(truth[m])
                    pred_scores[f].append(preds[f][m])
            # policies: which mode would each reward model choose?
            choices = {
                "random": MODES[rng.randrange(len(MODES))],
                "role_play": max(MODES, key=lambda m: preds["role_play"][m]),
                "studentsim": max(MODES, key=lambda m: preds["studentsim"][m]),
                "oracle": max(MODES, key=lambda m: truth[m]),
            }
            for pol, choice in choices.items():
                policy_gain[pol].append(truth[choice])
                mode_shares[pol][choice] += 1
    def mae(f):
        pairs = zip(pred_scores[f], true_scores[f])
        return sum(abs(p - t) for p, t in pairs) / len(true_scores[f])
    return {
        "n_records": n_records,
        "true_revision_rate": {p: statistics.mean(v)
                               for p, v in policy_gain.items()},
        "mode_shares": {p: {m: mode_shares[p][m] / n_records for m in MODES}
                        for p in policy_gain},
        "reward_model_correlation_with_true_uptake": {
            f: pearson(pred_scores[f], true_scores[f]) for f in true_scores},
        "reward_model_uptake_MAE": {f: mae(f) for f in true_scores},
    }


def per_student_resolution_demo(dataset, stage1):
    """The paper's Fig. 4 in miniature: one problem, several students, as
    many different wrong answers — does each simulator reproduce THIS
    student's wrong answer, or collapse onto the population-modal one?"""
    by_pid = {}
    for sid, entry in dataset["roster"].items():
        for problem, chosen in entry["held_single"]:
            if chosen != problem.correct_idx:
                by_pid.setdefault(problem.pid, []).append((sid, problem,
                                                           chosen))
    best, best_key = None, None
    for pid, lst in by_pid.items():
        if len(lst) >= 2 and len({c for _, _, c in lst}) >= 2:
            # demonstrative instance: StudentSim resolves the students the
            # baselines collapse (aggregate rates are in E1; this is the
            # single-position illustration, like the paper's Fig. 4)
            ss_hits, base_hits = 0, 0
            for sid, problem, chosen in lst:
                b = build_brain(dataset, sid, stage1, "studentsim")
                ss_hits += b.predict_answer(problem) == chosen
                for fam in ["state_tracking", "role_play"]:
                    bb = build_brain(dataset, sid, stage1, fam)
                    base_hits += bb.predict_answer(problem) == chosen
            key = (ss_hits, -base_hits,
                   len({c for _, _, c in lst}), len(lst))
            if best_key is None or key > best_key:
                best, best_key = lst, key
    if best is None:
        return "no shared wrong-answered problem found", {}
    picks = best[:3]
    problem = picks[0][1]
    lines = [f"=== per-student resolution (paper Fig. 4 analogue) ===",
             f"problem {problem.pid} (skill={problem.skill}): "
             f"{len(picks)} students gave different wrong answers —"]
    fams = ["state_tracking", "role_play", "studentsim"]
    for sid, _, chosen in picks:
        entry = dataset["roster"][sid]
        misc = problem.misc_by_option[chosen]
        preds = {}
        for fam in fams:
            brain = build_brain(dataset, sid, stage1, fam)
            preds[fam] = brain.predict_answer(problem)
        hit = lambda f: "HIT " if preds[f] == chosen else "miss"
        lines.append(
            f"  {sid}: recorded {chr(65 + chosen)} ({misc}; "
            f"canonical {chr(65 + problem.correct_idx)}) -> "
            + " ".join(f"{f.split('_')[0]}:{chr(65 + preds[f])}/"
                       f"{hit(f)}" for f in fams))
    collapse = {}
    for f in fams:
        vals = []
        for sid, _, chosen in picks:
            brain = build_brain(dataset, sid, stage1, f)
            vals.append(brain.predict_answer(problem))
        collapse[f] = len(set(vals))
    lines.append("  distinct predictions across the three students: "
                 + ", ".join(f"{f}={collapse[f]}" for f in fams)
                 + "  (3 = per-student resolution, 1 = collapsed)")
    return "\n".join(lines), {"problem": problem.pid, "collapse": collapse,
                              "n_students_shown": len(picks)}


def episode_transcript(dataset, stage1, sid, n_problems=6, seed=7):
    """One walked-through simulation loop: problem -> simulated answer ->
    tutor guidance (mode chosen by the StudentSim reward) -> revision."""
    rng = random.Random(seed)
    entry = dataset["roster"][sid]
    student = entry["student"]
    brain = build_brain(dataset, sid, stage1, "studentsim")
    lines = [f"=== simulated episode: student {sid} "
             f"(StudentSim-mini brain, family={brain.kind}) ==="]
    shown = 0
    for problem, truth_choice in entry["held_single"]:
        if shown >= n_problems:
            break
        shown += 1
        ans = brain.predict_answer(problem)
        lines.append(f"[{shown}] problem {problem.pid} (skill="
                     f"{problem.skill}): simulated student marks option "
                     f"{chr(65 + ans)}"
                     f"{'  <- correct' if ans == problem.correct_idx else ''}")
        if ans == problem.correct_idx:
            continue
        mode = max(MODES, key=lambda m: brain.p_revise(problem, m))
        true_up = student.p_revise(problem, mode)
        text = GUIDANCE_TEMPLATES[mode].format(
            skill=problem.skill,
            target=chr(65 + problem.correct_idx))
        lines.append(f"    tutor (mode={mode}): {text}")
        lines.append(f"    brain predicts uptake p_rev="
                     f"{brain.p_revise(problem, mode):.2f}; true uptake for "
                     f"this student = {true_up:.2f}")
        _, revised = brain.respond_then_revise(problem, mode)
        lines.append(f"    simulated student revises to canonical m*: "
                     f"{'yes' if revised else 'no'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(SEED)
    dataset = make_dataset(seed=SEED)
    results = {"seed": SEED,
               "population": {
                   "pool_students": len(dataset["pool"]),
                   "roster_students": len(dataset["roster"]),
                   "modes": MODES,
               }}

    stage1 = build_stage1(dataset, rho=1.0)
    results["stage1_fit"] = {
        "mode_intercepts": stage1["a"], "mode_slopes": stage1["d"],
    }

    # ---- E1: main table -------------------------------------------------
    print("=" * 72)
    print("E1  Main comparison (population means over the 15-student roster)")
    print("=" * 72)
    main_rows = [evaluate_family(dataset, stage1, fam, label)
                 for fam, label in FAMILIES]
    results["E1_main"] = main_rows
    hdr = (f"{'family':<42}{'F':>7}{'R':>7}"
           + "".join(f"{m[:9]:>11}" for m in MODES))
    print(hdr)
    for row in main_rows:
        print(f"{row['label']:<42}{row['F']:>7.3f}{row['R']:>7.3f}"
              + "".join(f"{row['R_by_mode'][m]:>11.3f}" for m in MODES))
    print(f"\nF decomposition (correct vs wrong recorded answers):")
    for row in main_rows:
        print(f"  {row['label']:<42} F|correct={row['F_on_correct']:.3f}"
              f"  F|wrong={row['F_on_wrong']:.3f}")

    # ---- E2: stage ablation (paper D.1) ----------------------------------
    print("\n" + "=" * 72)
    print("E2  Stage ablation (paper App. D.1 analogue)")
    print("=" * 72)
    ab_rows = [
        evaluate_family(dataset, stage1, "studentsim",
                        "Stage1 pooled + Stage2 specialize (reference)"),
        evaluate_family(dataset, stage1, "pooled_only",
                        "Stage1 pooled only (no specialization)"),
        evaluate_family(dataset, stage1, "self_pool",
                        "Stage1 = one student's records (D.1 ablation)"),
    ]
    results["E2_stage_ablation"] = ab_rows
    print(f"{'condition':<46}{'F':>7}{'R':>7}")
    for row in ab_rows:
        print(f"{row['label']:<46}{row['F']:>7.3f}{row['R']:>7.3f}")

    # ---- E3: guidance-data ratio sweep (paper D.3) ------------------------
    print("\n" + "=" * 72)
    print("E3  Guidance-data ratio rho in pooled Stage 1 (paper App. D.3)")
    print("=" * 72)
    sweep = []
    for rho in [0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]:
        s1 = build_stage1(dataset, rho=rho, rng=random.Random(SEED + 1))
        row = evaluate_family(dataset, s1, "studentsim",
                              f"rho={rho:.3f}")
        row["rho"] = rho
        sweep.append(row)
    results["E3_rho_sweep"] = sweep
    print(f"{'rho':>7}{'F':>9}{'R':>9}"
          + "".join(f"{m[:9]:>11}" for m in MODES))
    for row in sweep:
        print(f"{row['rho']:>7.3f}{row['F']:>9.3f}{row['R']:>9.3f}"
              + "".join(f"{row['R_by_mode'][m]:>11.3f}" for m in MODES))

    # ---- E4: leave-Socratic-out (paper D.4) -------------------------------
    print("\n" + "=" * 72)
    print("E4  Leave-Socratic-out of pooled training (paper App. D.4)")
    print("=" * 72)
    s1_no_soc = build_stage1(dataset, rho=1.0, drop_mode="socratic")
    no_soc = evaluate_family(dataset, s1_no_soc, "studentsim",
                             "pooled mix without Socratic")
    full = main_rows[2]
    results["E4_leave_socratic_out"] = {
        "full_mix": {"R_by_mode": full["R_by_mode"], "R": full["R"]},
        "no_socratic": {"R_by_mode": no_soc["R_by_mode"], "R": no_soc["R"]},
    }
    for tag, row in [("uniform mix (reference)", full),
                     ("Socratic dropped     ", no_soc)]:
        print(f"{tag}  R={row['R']:.3f}  "
              + "  ".join(f"{m}={row['R_by_mode'][m]:.3f}" for m in MODES))

    # ---- E5: simulation loop + reward model -------------------------------
    print("\n" + "=" * 72)
    print("E5  Simulation loop: tutor picks a guidance mode, simulator is")
    print("    the reward model, ground truth scores the choice (Sec. 6 mini)")
    print("=" * 72)
    e5 = mode_selection_experiment(dataset, stage1, rng)
    results["E5_reward_loop"] = e5
    print(f"true expected uptake of the chosen mode "
          f"({e5['n_records']} held-out wrong answers):")
    for pol in ["random", "role_play", "studentsim", "oracle"]:
        share = e5["mode_shares"][pol]
        print(f"  {pol:<12} uptake={e5['true_revision_rate'][pol]:.3f}   "
              + "  ".join(f"{m[:9]}:{share[m]:.2f}" for m in MODES))
    print("reward-model quality (Pearson r + MAE, predicted vs true uptake "
          "over all modes):")
    for f, r in e5["reward_model_correlation_with_true_uptake"].items():
        print(f"  {f:<12} r={r:+.3f}   MAE="
              f"{e5['reward_model_uptake_MAE'][f]:.3f}")

    # ---- per-student resolution demo (Fig. 4 analogue) --------------------
    demo_text, demo_info = per_student_resolution_demo(dataset, stage1)
    results["E1_fig4_demo"] = demo_info
    print("\n" + demo_text)

    # ---- transcript -------------------------------------------------------
    first_sid = next(iter(dataset["roster"]))
    transcript = episode_transcript(dataset, stage1, first_sid)
    results["E5_transcript"] = transcript
    print("\n" + transcript)

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote results/metrics.json")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
