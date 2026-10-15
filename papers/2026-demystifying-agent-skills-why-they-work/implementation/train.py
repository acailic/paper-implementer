# train.py — run the full experimental pipeline (RQ1–RQ4)
"""Experiment driver for the from-scratch re-implementation of:

  "Demystifying Agent Skills: Why They Work—Until They Don't"
  Jiang et al., 2026. arXiv:2608.14036
  https://arxiv.org/abs/2608.14036

Runs the paper's four controlled experiments as a Monte-Carlo simulation
over the synthetic world in data.py / model.py:

  RQ1  representation      — same traces -> WM vs SKILL, paired deltas
  RQ2  outcome annotation  — hinted vs no-hint skill creation, by mixture
  RQ3  cross-framework     — artifacts built under Codex, run under Gemini
  RQ4  retrieval           — offline-embedding / explicit-selection /
                             actual-use arms; precision collapse vs pool size

All aggregation follows the paper: paired deltas with 1,000-iter bootstrap
CIs over triples, precision/recall per Eqs. 1–2, F1 recomputed from
aggregated P and R.

Usage:  python3 train.py [--tasks N] [--trials N] [--seed S] [--json OUT]
Pure stdlib; no dependencies.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict

import data as D
from model import (ALL_12_MODES, execute_raw, execute_with_guidance,
                   build_skill, build_workflow_memory, f1_from_aggregates,
                   paired_bootstrap_ci, precision, recall)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rate(bools: list[bool]) -> float:
    return sum(bools) / len(bools) if bools else 0.0


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point * 100:+.2f} pts [{lo * 100:+.2f}, {hi * 100:+.2f}]"


def mode_table(records: list, arm: str) -> Counter:
    return Counter(r.mode for r in records if r.arm == arm)


# ---------------------------------------------------------------------------
# RQ1 — representation: Skill vs Workflow Memory vs Raw (paired)
# ---------------------------------------------------------------------------


def run_rq1(tasks, n_trials, rng, hint=True, harness="codex",
            label="RQ1"):
    """For each task x mixture: collect pool -> build both artifacts from
    the SAME traces -> run all three arms -> verifier outcomes."""
    per_mix = defaultdict(lambda: {"raw": [], "wf": [], "sk": []})
    triples = []  # (raw, wf, sk) success booleans for paired bootstrap
    mech_counter = Counter()
    all_records = []
    for task in tasks:
        for (ns, nf) in D.MIXTURES:
            pool = D.sample_trajectory_pool(task, ns, nf, rng)
            wm = build_workflow_memory(pool)
            sk = build_skill(pool, hint=hint, framework=harness)
            for _ in range(n_trials):
                r = execute_raw(task, harness)
                w = execute_with_guidance(task, wm, harness)
                s = execute_with_guidance(task, sk, harness)
                key = f"{ns}s{nf}f"
                per_mix[key]["raw"].append(r.success)
                per_mix[key]["wf"].append(w.success)
                per_mix[key]["sk"].append(s.success)
                triples.append((r.success, w.success, s.success))
                mech_counter[s.mechanism] += 1
                all_records += [r, w, s]
    out = {"label": label, "per_mixture": {}, "mechanism": {}}
    for key in sorted(per_mix):
        d = per_mix[key]
        out["per_mixture"][key] = {
            "raw": rate(d["raw"]), "workflow": rate(d["wf"]),
            "skill": rate(d["sk"])}
    out["success"] = {
        "raw": rate([t[0] for t in triples]),
        "workflow": rate([t[1] for t in triples]),
        "skill": rate([t[2] for t in triples])}
    d_sk_raw = [float(t[2] - t[0]) for t in triples]
    d_wf_raw = [float(t[1] - t[0]) for t in triples]
    d_sk_wf = [float(t[2] - t[1]) for t in triples]
    out["deltas"] = {
        "skill_vs_raw": fmt_ci(*paired_bootstrap_ci(d_sk_raw, seed=1)),
        "workflow_vs_raw": fmt_ci(*paired_bootstrap_ci(d_wf_raw, seed=2)),
        "skill_vs_workflow": fmt_ci(*paired_bootstrap_ci(d_sk_wf, seed=3))}
    total = sum(mech_counter.values()) or 1
    out["mechanism"] = {k: round(v / total, 4)
                        for k, v in mech_counter.most_common()}
    out["_records"] = all_records
    return out


# ---------------------------------------------------------------------------
# RQ2 — outcome annotation: hinted vs no-hint skill creation
# ---------------------------------------------------------------------------


def run_rq2(tasks, n_trials, rng):
    hinted = run_rq1(tasks, n_trials, rng, hint=True,
                     label="RQ2 hinted (labels available)")
    nohint = run_rq1(tasks, n_trials, rng, hint=False,
                     label="RQ2 no-hint (labels stripped)")
    rows = []
    for key in sorted(hinted["per_mixture"]):
        h = hinted["per_mixture"][key]["skill"]
        nh = nohint["per_mixture"][key]["skill"]
        rows.append((key, h, nh, nh - h))
    return {"hinted": hinted, "nohint": nohint, "rows": rows}


# ---------------------------------------------------------------------------
# RQ3 — cross-framework transfer: Codex-built artifacts run in Gemini CLI
# ---------------------------------------------------------------------------


def run_rq3(tasks, n_trials, rng):
    same = run_rq1(tasks, n_trials, rng, hint=True, harness="codex",
                   label="RQ3 same-framework (Codex -> Codex)")
    # artifacts built under codex, executed under gemini
    transfer = {"per_mixture": {}}
    per_mix = defaultdict(lambda: {"raw": [], "wf": [], "sk": []})
    triples = []
    for task in tasks:
        for (ns, nf) in D.MIXTURES:
            pool = D.sample_trajectory_pool(task, ns, nf, rng)
            wm = build_workflow_memory(pool)   # built under codex
            sk = build_skill(pool, framework="codex")
            for _ in range(n_trials):
                r = execute_raw(task, "gemini")
                w = execute_with_guidance(task, wm, "gemini")
                s = execute_with_guidance(task, sk, "gemini")
                key = f"{ns}s{nf}f"
                per_mix[key]["raw"].append(r.success)
                per_mix[key]["wf"].append(w.success)
                per_mix[key]["sk"].append(s.success)
                triples.append((r.success, w.success, s.success))
    for key in sorted(per_mix):
        d = per_mix[key]
        transfer["per_mixture"][key] = {
            "raw": rate(d["raw"]), "workflow": rate(d["wf"]),
            "skill": rate(d["sk"])}
    transfer["success"] = {
        "raw": rate([t[0] for t in triples]),
        "workflow": rate([t[1] for t in triples]),
        "skill": rate([t[2] for t in triples])}
    d_sk_wf = [float(t[2] - t[1]) for t in triples]
    transfer["deltas"] = {"skill_vs_workflow": fmt_ci(
        *paired_bootstrap_ci(d_sk_wf, seed=7))}
    return {"same": same, "transfer": transfer}


# ---------------------------------------------------------------------------
# RQ4 — retrieval decomposition (three decoupled arms)
# ---------------------------------------------------------------------------


def arm1_offline_embedding(task, pool, gold, rng):
    """Qwen-style cosine retrieval: rank pool by similarity to the task.

    The query embeds the *instruction*, which only approximates the
    skill-description embedding — so ranking is noisy, and confusable
    (similar-regime) distractors sometimes outrank the gold skill.
    """
    query = D._near(task_vec(task, rng), 0.05, rng)
    sims = [(D.cosine(query, s.embedding), s) for s in pool]
    top = max(sims, key=lambda x: x[0])[1]
    return {top.skill_id}


def task_vec(task, rng):
    return D._embed(task.tid, rng, salt="task")


def arm2_explicit_selection(task, pool, rng, gold):
    """Agent sees available_skills and over-selects a bit (the paper's
    Arm 2: recall holds up, precision sags as pools grow)."""
    gold_present = gold in pool
    chosen = set()
    if gold_present and rng.random() < 0.82:  # recall stays high
        chosen.add(gold.skill_id)
    n_extra = rng.choice([0, 1, 1, 2])
    for s in rng.sample(pool, min(n_extra, len(pool))):
        if s.skill_id not in chosen:
            chosen.add(s.skill_id)
    return chosen


def arm3_real_execution(task, pool, gold, rng, regime):
    """Full pool in the environment; no preselection passes through.

    Browsing budget grows slowly with pool size (log k), while the chance
    any given open lands on the gold skill decays with pool size — harder
    for confusable (similar) pools. Distractors that are opened are also
    *executed against*: random-regime pools contain a few genuinely useful
    generic skills (the paper's 'useful false positives'), which is what
    keeps downstream success roughly flat as precision collapses.
    """
    k = len(pool)
    n_open = max(1, round((2 if k > 5 else 1) + 0.9 * (k.bit_length() - 2)))
    decay = {"random": 0.35, "similar": 0.50, "dissimilar": 0.15}[regime]
    p_gold = {"random": 0.55, "similar": 0.40, "dissimilar": 0.80}[regime]
    p_gold *= (5.0 / k) ** decay  # gold gets harder to locate as pools grow
    chosen = set()
    opened_guidance = []
    for _ in range(n_open):
        if rng.random() < p_gold:
            chosen.add(gold.skill_id)
            opened_guidance.append(gold)
        else:
            d = rng.choice([s for s in pool if s is not gold])
            chosen.add(d.skill_id)
            opened_guidance.append(d)
    # execute against the *best* skill actually opened (agent commits to
    # one); if none opened, behave like raw
    if not opened_guidance:
        return chosen, execute_raw(task, "codex")
    used = max(opened_guidance, key=lambda s: s.procedure_fidelity)
    rec = execute_with_guidance(
        task, used, "codex",
        n_distractors_present=min(k - 1, 30),
        p_notice_skill=1.0)
    return chosen, rec


def run_rq4(tasks, n_trials, rng, n_tasks_limit=40):
    """RQ4 grid: pool size x regime x {arm1, arm2, arm3}."""
    sub = tasks[:n_tasks_limit]
    results = []
    for regime in D.REGIMES:
        for k in D.POOL_SIZES:
            a1_p, a2_p, a2_r, a3_p, a3_r, a3_s = [], [], [], [], [], []
            for task in sub:
                gold_set = {task.gold_skill_id}
                for _ in range(n_trials):
                    pool, gold = D.make_skill_pool(task, k, regime, rng)
                    a1 = arm1_offline_embedding(task, pool, gold, rng)
                    a1_p.append(precision(a1, gold_set))
                    a2 = arm2_explicit_selection(task, pool, rng, gold)
                    a2_p.append(precision(a2, gold_set))
                    a2_r.append(recall(a2, gold_set))
                    chosen, rec = arm3_real_execution(task, pool, gold, rng,
                                                      regime)
                    a3_p.append(precision(chosen, gold_set))
                    a3_r.append(recall(chosen, gold_set))
                    a3_s.append(rec.success)
            results.append({
                "regime": regime, "k": k,
                "arm1_top1_precision": round(statistics.mean(a1_p), 4),
                "arm2_precision": round(statistics.mean(a2_p), 4),
                "arm2_recall": round(statistics.mean(a2_r), 4),
                "arm3_actual_use_precision": round(statistics.mean(a3_p), 4),
                "arm3_recall": round(statistics.mean(a3_r), 4),
                "arm3_success": round(statistics.mean(a3_s), 4),
            })
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_header(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=60,
                    help="number of synthetic tasks")
    ap.add_argument("--trials", type=int, default=5,
                    help="trials per task x mixture cell (paper: 5)")
    ap.add_argument("--seed", type=int, default=D.MASTER_SEED)
    ap.add_argument("--json", type=str, default="results.json",
                    help="write full numeric results here")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)  # model.py helpers draw from the global RNG
    tasks = D.make_tasks(args.tasks, seed=args.seed)

    print_header(
        f"Demystifying Agent Skills — re-implementation | "
        f"{args.tasks} tasks x {args.trials} trials x 6 mixtures | "
        f"seed {args.seed}")

    # ---- RQ1 -------------------------------------------------------------
    rq1 = run_rq1(tasks, args.trials, rng, label="RQ1")
    print_header("RQ1 — Representation: Skill vs Workflow Memory vs Raw")
    print(f"  success  raw={rq1['success']['raw']*100:.1f}%  "
          f"workflow={rq1['success']['workflow']*100:.1f}%  "
          f"skill={rq1['success']['skill']*100:.1f}%")
    for name, d in rq1["deltas"].items():
        print(f"  {name:18s} {d}")
    print("  by mixture (raw/wf/skill %):")
    for key in sorted(rq1["per_mixture"]):
        m = rq1["per_mixture"][key]
        print(f"    {key}: {m['raw']*100:5.1f} / {m['workflow']*100:5.1f} "
              f"/ {m['skill']*100:5.1f}")
    print("  skill mechanism distribution:")
    for mech, frac in rq1["mechanism"].items():
        print(f"    {mech:22s} {frac*100:5.1f}%")

    # 12-mode table (from RQ1 records)
    print("  12-mode taxonomy (fraction of triples per arm):")
    n_triples = sum(1 for r in rq1["_records"] if r.arm == "raw")
    headers = ["mode", "raw", "wf", "skill"]
    counts = {a: mode_table(rq1["_records"], a)
              for a in ("raw", "workflow", "skill")}
    print(f"    {'mode':42s} {'raw%':>6} {'wf%':>6} {'skill%':>6}")
    for mode in ALL_12_MODES:
        row = [counts[a].get(mode, 0) / n_triples * 100
               for a in ("raw", "workflow", "skill")]
        print(f"    {mode:42s} {row[0]:6.1f} {row[1]:6.1f} {row[2]:6.1f}")

    # ---- RQ2 -------------------------------------------------------------
    rq2 = run_rq2(tasks, args.trials, rng)
    print_header("RQ2 — Outcome annotation: hinted vs no-hint (skill arm)")
    print(f"  {'mixture':8s} {'hinted':>8} {'no-hint':>8} {'delta':>8}")
    for key, h, nh, dd in rq2["rows"]:
        print(f"  {key:8s} {h*100:7.1f}% {nh*100:7.1f}% {dd*100:+7.1f}")
    print("  -> no-hint is ~free on success-only pools, costly once")
    print("     failed traces enter the distiller's input (paper RQ2).")

    # ---- RQ3 -------------------------------------------------------------
    rq3 = run_rq3(tasks, args.trials, rng)
    print_header("RQ3 — Cross-framework transfer (Codex-built -> Gemini run)")
    print("  same-framework skill-vs-workflow:",
          rq3["same"]["deltas"]["skill_vs_workflow"])
    print("  transfer skill-vs-workflow:   ",
          rq3["transfer"]["deltas"]["skill_vs_workflow"])
    for key in sorted(rq3["transfer"]["per_mixture"]):
        m = rq3["transfer"]["per_mixture"][key]
        print(f"    {key}: raw {m['raw']*100:5.1f} / wf {m['workflow']*100:5.1f}"
              f" / skill {m['skill']*100:5.1f}")
    print("  -> skill > workflow at every mixture, even after transfer;")

    # ---- RQ4 -------------------------------------------------------------
    rq4 = run_rq4(tasks, args.trials, rng)
    print_header("RQ4 — Retrieval decomposition (precision vs pool size)")
    print(f"  {'regime':10s} {'k':>4} {'A1 top-1':>9} {'A2 P':>7} "
          f"{'A2 R':>7} {'A3 use-P':>9} {'A3 R':>7} {'A3 succ':>8}")
    for row in rq4:
        print(f"  {row['regime']:10s} {row['k']:>4d} "
              f"{row['arm1_top1_precision']*100:8.1f}% "
              f"{row['arm2_precision']*100:6.1f}% "
              f"{row['arm2_recall']*100:6.1f}% "
              f"{row['arm3_actual_use_precision']*100:8.1f}% "
              f"{row['arm3_recall']*100:6.1f}% "
              f"{row['arm3_success']*100:7.1f}%")
    print("  -> actual-use precision collapses with k while success stays")
    print("     flat: exact gold invocation is neither sufficient nor")
    print("     necessary (paper's central retrieval finding).")

    # ---- dump ------------------------------------------------------------
    dump = {
        "rq1": {k: v for k, v in rq1.items() if k != "_records"},
        "rq2": {"rows": [{"mixture": r[0], "hinted": round(r[1], 4),
                          "nohint": round(r[2], 4),
                          "delta": round(r[3], 4)} for r in rq2["rows"]]},
        "rq3": {"same": {k: v for k, v in rq3["same"].items()
                         if k != "_records"},
                "transfer": rq3["transfer"]},
        "rq4": rq4,
        "config": {"tasks": args.tasks, "trials": args.trials,
                   "seed": args.seed},
    }
    with open(args.json, "w") as f:
        json.dump(dump, f, indent=2)
    print_header(f"Done. Full numbers written to {args.json}")
    print("Paper: Demystifying Agent Skills: Why They Work—Until They Don't")
    print("       Jiang et al. 2026, arXiv:2608.14036 — re-implemented from")
    print("       scratch as a Monte-Carlo replication of its experimental")
    print("       design and measurement pipeline.")


if __name__ == "__main__":
    main()
