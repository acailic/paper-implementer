"""StateM demo — a deterministic scripted agent walks the demo runbook.

Shows every control property the paper claims, without an LLM:
  1. blocked transition: failing command check keeps the run IN Execute,
     reason recorded (check name + evidence class + detail) — step 2 of the
     goto protocol bites before any commit.
  2. guard routing: Verify with tests_pass=false cannot reach Handoff;
     the edge's on_guard_fail routes to Repair — repair is graph, not prose.
  3. checklist self-attestation acknowledged explicitly.
  4. resume: reload the run from the journal; the authoritative phase is
     whatever the last committed transition says — no reconstruction.
  5. history is the audit trail; metrics.json counts transitions, blocks,
     and evidence classes used.
"""

from __future__ import annotations

import json
from pathlib import Path

from statem import Run, StateM, Runbook

HERE = Path(__file__).parent
WORK = HERE / "work"
OUT = HERE / "metrics.json"


def show(entry: dict) -> None:
    print(f"\n[phase] {entry['phase']}  →  outgoing: {', '.join(entry['outgoing']) or '(terminal)'}")
    print(f"        {entry['prompt'][:88]}")


def main() -> None:
    WORK.mkdir(exist_ok=True)
    answer = WORK / "answer.txt"
    if answer.exists():
        answer.unlink()

    book = Runbook.load(HERE / "runbooks/demo.json")
    run = Run(runbook=book.name, state=book.initial, workspace=str(WORK))
    rt = StateM(book, run)
    metrics = {"transitions": [], "blocked": [], "guard_routes": [], "evidence_used": set()}

    print("== StateM demo runbook:", book.name, f"(v{book.version}) ==")

    # --- Plan ----------------------------------------------------------
    show(rt.enter())
    metrics["evidence_used"].add("predicate (host-reproducible)")
    print("  body: plan written (receipt plan_done=true recorded by the body, not the runtime)")
    run.receipts["plan_done"] = True  # the BODY's work — receipts precede goto
    r = rt.goto("Execute")
    assert r["ok"]; metrics["transitions"].append(("Plan", "Execute"))

    # --- Execute: FIRST goto fails — artifact missing -------------------
    print("\n  body: implementation 'done'... but artifact missing")
    run.receipts["implementation_done"] = True  # predicate passes...
    r = rt.goto("Verify")                        # ...command check does NOT
    assert not r["ok"] and r["stage"] == "before_transfer"
    f = r["failed"][0]
    print(f"  ✗ BLOCKED in Execute by '{f.name}' [{f.evidence}]: {f.detail}")
    metrics["blocked"].append({"state": "Execute", "check": f.name, "evidence": f.evidence})
    assert run.state == "Execute", "failed check must keep the run in the source state"

    # repair the artifact, retry
    answer.write_text("42\n")
    print("  repair: artifact written; retry goto")
    r = rt.goto("Verify")
    assert r["ok"]; metrics["transitions"].append(("Execute", "Verify"))
    metrics["evidence_used"].add("command (host-reproducible)")
    show(r["entry"])

    # --- Verify: tests fail once → guard routes to Repair ---------------
    print("  body: verification suite runs; tests_pass = false (recorded honestly)")
    run.receipts["tests_pass"] = False
    run.acknowledged["verify-checklist"] = ["consumer command run", "evidence recorded"]
    metrics["evidence_used"].add("checklist (self-attestation)")
    print("  acknowledged: verify-checklist (self-attestation, labelled as such)")
    # guard routing: goto Handoff is RE-ROUTED to Repair (ok=True, different target)
    r = rt.goto("Handoff")
    assert run.state == "Repair", f"expected guard routing to Repair, got {r}"
    print("  ✗ GUARD: Handoff requires tests_pass=true → routed to Repair (explicit edge)")
    metrics["guard_routes"].append("Verify → Repair")
    show({"phase": run.state, "prompt": book.states["Repair"]["prompt"], "outgoing": ["Verify"]})
    run.receipts["tests_pass"] = True  # repair body: fix + honest re-record
    r = rt.goto("Verify")
    assert r["ok"]; metrics["transitions"].append(("Repair", "Verify"))
    show(r["entry"])

    # --- Handoff ---------------------------------------------------------
    r = rt.goto("Handoff")
    assert r["ok"], r
    metrics["transitions"].append(("Verify", "Handoff"))
    show(r["entry"])
    print("\n  ✓ TERMINAL reached — success recorded only here, per the runbook.")

    # --- resume from journal ----------------------------------------------
    run.save(WORK / "run.json")
    resumed = Run.load(WORK / "run.json")
    print(f"\n== resume: journal reload → authoritative phase = {resumed.state} "
          f"({len(resumed.history)} events; no reconstruction from a trace)")

    metrics["evidence_used"] = sorted(metrics["evidence_used"])
    metrics["final_state"] = resumed.state
    metrics["events"] = len(resumed.history)
    OUT.write_text(json.dumps(metrics, indent=1))
    print(f"[metrics] {OUT}  ({len(metrics['transitions'])} transitions, "
          f"{len(metrics['blocked'])} blocked, {len(metrics['guard_routes'])} guard routes)")


if __name__ == "__main__":
    main()
