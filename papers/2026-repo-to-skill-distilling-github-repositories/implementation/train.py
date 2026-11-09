"""train.py — end-to-end Repo-To-Skill pipeline runner (arXiv:2609.02749).

Paper: "Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills"
        (Chen et al., 2026)

Runs the paper's Eq. 8 pipeline end to end for a task-agnostic anchor
(z = one GitHub repository), deterministically:

    anchor   : shallow-clone the repo (or reuse a checkout)      [data.py]
    scope    : capability identification Q                       [model.py]
    ground   : evidence mapping X with file:line provenance      [model.py]
    construct: three-layer skill artifact G~ + graph.json        [model.py]
    verify   : static gates + QA carry test -> record R          [verify.py]

Outputs (under --out):
    skills/<repo>/SKILL.md, references/, scripts/, graph.json
    record.json        (construction record R: gates + QA + unresolved gaps)
    results.json       (run summary)

Example:
    python3 train.py --repo-url https://github.com/karpathy/minGPT

Stdlib only; CPU only; no repository code is ever executed except the
generated smoke_check.py, which itself only reads files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import data  # noqa: E402
import model  # noqa: E402
import verify  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="deterministic repo-to-skill distiller")
    ap.add_argument("--repo-url", default="https://github.com/karpathy/minGPT")
    ap.add_argument("--repo-path", default="", help="reuse an existing checkout instead of cloning")
    ap.add_argument("--work-dir", default=os.path.join(HERE, "work"))
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--questions", default=os.path.join(HERE, "questions.json"))
    args = ap.parse_args()

    t0 = time.time()
    report: dict = {"pipeline": ["anchor", "scope", "ground", "construct", "verify"]}

    # ---- stage 0: anchor z ------------------------------------------------
    if args.repo_path:
        repo_root = os.path.abspath(args.repo_path)
        commit = subprocess.run(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "unknown"
    else:
        repo_root, commit = data.clone(args.repo_url, args.work_dir)
    print(f"[anchor]   {args.repo_url or args.repo_path} @ {commit} ({repo_root})")
    report["anchor"] = {"url": args.repo_url, "commit": commit}

    # ---- stage 1: scope (evidence boundary + source profile) --------------
    profile = data.build_profile(args.repo_url or repo_root, repo_root, commit)
    n_by_kind: dict[str, int] = {}
    for e in profile.evidence:
        n_by_kind[e.kind] = n_by_kind.get(e.kind, 0) + 1
    print(f"[scope]    {len(profile.files)} evidence files admitted; "
          f"{len(profile.evidence)} evidence units {n_by_kind}; "
          f"deps={sorted(profile.dependencies)}")
    report["evidence_boundary"] = {"files": len(profile.files),
                                    "units": len(profile.evidence), "by_kind": n_by_kind}

    # ---- stage 2: scope -> Q, ground -> X ----------------------------------
    caps = model.scope(profile)
    X = model.ground(profile, caps)
    print(f"[scope]    capabilities Q: {[c.key for c in caps]}")
    print(f"[ground]   evidence X: " + ", ".join(f"{k}:{len(v)}" for k, v in X.items()))
    report["capabilities"] = {c.key: len(X[c.key]) for c in caps}

    # ---- stage 3: construct -> G~ ------------------------------------------
    skill_dir = model.write_skill(args.out, profile, caps, X)
    print(f"[construct] skill graph written: {os.path.relpath(skill_dir, args.out)}/ "
          "(SKILL.md + references/ + scripts/ + graph.json)")

    # ---- stage 4: verify -> (G, R) -----------------------------------------
    gates = verify.static_gates(skill_dir, profile, X, caps)
    for g in gates:
        print(f"[verify]   {'PASS' if g['pass'] else 'FAIL'}  {g['gate']:<28} {g['detail']}")
    qa = verify.qa_carry_test(skill_dir, repo_root, profile, args.questions)
    print(f"[verify]   QA carry (retrieved): closed-book {qa['closed_book_pass']}/{qa['n_questions']}"
          f" vs open-book {qa['open_book_pass']}/{qa['n_questions']}"
          f" -> carry ratio {qa['carry_ratio']}")
    print(f"[verify]   QA carry (present):   closed-book {qa['closed_present']}/{qa['n_questions']}"
          f" vs open-book {qa['open_present']}/{qa['n_questions']}"
          f" -> present-carry ratio {qa['present_carry_ratio']}")
    for r in qa["per_question"]:
        cb, ob = r["closed_book"], r["open_book"]
        print(f"           {r['id']:<22} closed={'hit ' if cb['pass'] else 'miss'}"
              f"(present={'Y' if cb['present'] else 'N'}, r={cb['gold_recall']:.2f}) "
              f"open={'hit' if ob['pass'] else 'miss'}"
              f"(present={'Y' if ob['present'] else 'N'}, r={ob['gold_recall']:.2f})")

    # run the generated smoke script against the checkout (reads files only)
    smoke = os.path.join(skill_dir, "scripts", "smoke_check.py")
    sm = subprocess.run([sys.executable, smoke, repo_root], capture_output=True, text=True)
    smoke_lines = sm.stdout.strip().splitlines()
    print(f"[verify]   smoke_check.py: {len(smoke_lines)} probes, exit={sm.returncode}")
    qa["smoke_check"] = {"exit": sm.returncode, "probes": len(smoke_lines),
                         "tail": smoke_lines[-3:]}

    record = verify.write_record(skill_dir, gates, qa)
    print(f"[verify]   construction record R: {os.path.relpath(record, args.out)}")

    ok = all(g["pass"] for g in gates)
    report.update({
        "skill_dir": os.path.relpath(skill_dir, args.out),
        "gates": gates,
        "gates_all_pass": ok,
        "qa": {k: v for k, v in qa.items() if k != "per_question"},
        "qa_per_question": qa["per_question"],
        "unresolved_gaps": json.load(open(record))["unresolved_gaps"],
        "seconds": round(time.time() - t0, 2),
        "distiller": "deterministic rule-based (no LLM) — arXiv:2609.02749 re-implementation",
    })
    results_path = os.path.join(args.out, "results.json")
    json.dump(report, open(results_path, "w"), indent=2)
    print(f"\n[done] gates_all_pass={ok}  carry_ratio={qa['carry_ratio']}  "
          f"({report['seconds']}s)  -> {results_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
