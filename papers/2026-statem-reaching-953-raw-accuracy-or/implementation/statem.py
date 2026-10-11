"""StateM-style runtime, re-implemented from scratch (stdlib only).

Paper: Qin et al., "StateM: Reaching 95.3% Raw Accuracy, or a $15 Frontier
Run, on Terminal-Bench 2.1 via Harness Scaling" (arXiv:2608.15089).

This module implements the paper's *control surface* faithfully:

  Runbook ℬ = (S, s0, ST, E, Φ)   — states, initial, terminals, edges, specs
  State anatomy — in_hook → body(discretion) → out_hook → before_transfer
  goto protocol — SIX ordered steps; failure keeps the run in the source
                  state with the reason recorded and visible
  Check ladder — command/predicate > manual > checklist/message >
                 llm_review, each labelled by evidentiary strength
  Guards       — edge-level routing over check results/receipts/failures
  Run          — append-only history journal, receipts, resume

Differences from the paper, declared: runbooks are JSON (the paper uses
YAML; stdlib-only constraint), and `llm_review` checks are recorded as
opinions with no verifier (the paper's point: structure is never proof).
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# checks — the evidentiary-strength ladder (paper Sec 3.2)
# ---------------------------------------------------------------------------

EVIDENCE = {
    "command": "host-reproducible",
    "predicate": "host-reproducible",
    "manual": "operator-decision",
    "checklist": "self-attestation",
    "message": "self-attestation",
    "llm_review": "semantic-judgment",
}


@dataclass
class CheckResult:
    name: str
    kind: str
    passed: bool
    detail: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        self.evidence = EVIDENCE.get(self.kind, "unspecified")


def run_check(spec: dict, run: "Run") -> CheckResult:
    """Evaluate one check spec against the run. Never raises on user content."""
    kind, name = spec.get("kind"), spec.get("name", "?")
    try:
        if kind == "command":
            # Host-evaluated: independently reproducible evidence.
            proc = subprocess.run(spec["cmd"], shell=True, capture_output=True, text=True, timeout=60)
            return CheckResult(name, kind, proc.returncode == 0,
                               (proc.stdout + proc.stderr).strip()[:200])
        if kind == "predicate":
            # Host-evaluated over receipts: spec = {"receipt": k, "equals": v}
            key, want = spec["receipt"], spec.get("equals")
            got = run.receipts.get(key)
            return CheckResult(name, kind, got == want, f"receipt {key}={got!r} (want {want!r})")
        if kind == "manual":
            # Operator decision — demo mode reads an override so the scripted
            # run proceeds; a real deployment blocks here for a human.
            approved = bool(spec.get("_demo_approved", False))
            return CheckResult(name, kind, approved, "operator decision" + ("" if approved else " (pending)"))
        if kind == "checklist":
            items = spec.get("items", [])
            acked = set(run.acknowledged.get(name, []))
            ok = acked >= set(items)
            return CheckResult(name, kind, ok, f"{len(acked & set(items))}/{len(items)} acknowledged")
        if kind == "message":
            msg = run.messages.get(name, "")
            return CheckResult(name, kind, bool(msg and msg == spec.get("expect", msg)), f"assertion: {msg[:80]!r}")
        if kind == "llm_review":
            # Semantic judgment: recorded, never treated as deterministic.
            verdict = run.reviews.get(name, "")
            return CheckResult(name, kind, verdict == "pass", f"review verdict: {verdict!r}")
    except Exception as exc:  # a failed check must not crash the runtime
        return CheckResult(name, kind or "?", False, f"error: {exc}")
    return CheckResult(name, kind, False, f"unknown check kind {kind!r}")


# ---------------------------------------------------------------------------
# runbook + runtime
# ---------------------------------------------------------------------------


@dataclass
class Runbook:
    """ℬ = (S, s0, ST, E, Φ). JSON-loaded, versioned artifact."""

    name: str
    version: str
    states: dict[str, dict]                     # Φ(s): prompt, hooks, checks, artifacts
    initial: str
    terminals: list[str]
    edges: list[dict]                           # {from, to, guard?}

    @classmethod
    def load(cls, path: str | Path) -> "Runbook":
        raw = json.loads(Path(path).read_text())
        return cls(
            name=raw["name"], version=str(raw.get("version", "0")),
            states=raw["states"], initial=raw["initial"],
            terminals=raw.get("terminals", []), edges=raw["edges"],
        )

    def edge(self, src: str, dst: str) -> dict | None:
        return next((e for e in self.edges if e["from"] == src and e["to"] == dst), None)

    def outgoing(self, src: str) -> list[dict]:
        return [e for e in self.edges if e["from"] == src]


# guard evaluation context — routing sees checks, receipts, failure counts
def guard_passes(guard: dict, run: "Run", last_results: list[CheckResult]) -> bool:
    if not guard:
        return True
    for cond in guard.get("all", []):
        if cond.get("receipt") is not None:
            if run.receipts.get(cond["receipt"]) != cond.get("equals"):
                return False
        if cond.get("failures_gte") is not None:
            if run.failure_count < cond["failures_gte"]:
                return False
    return True


@dataclass
class Run:
    """Mutable per-run state: authoritative phase, receipts, history journal."""

    runbook: str
    state: str
    history: list[dict] = field(default_factory=list)
    receipts: dict = field(default_factory=dict)
    acknowledged: dict = field(default_factory=dict)
    messages: dict = field(default_factory=dict)
    reviews: dict = field(default_factory=dict)
    failure_count: int = 0

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Run":
        d = json.loads(Path(path).read_text())
        return cls(**d)

    def log(self, event: str, **kw) -> None:
        self.history.append({"t": round(time.time(), 3), "event": event, **kw})


class StateM:
    """The runtime: transition validation, hook execution, history, recovery."""

    def __init__(self, book: Runbook, run: Run):
        if run.state not in book.states:
            raise ValueError(f"run references unknown state {run.state!r}")
        self.book = book
        self.run = run
        self._entered = False

    # -- state entry (step 6 / resume) ------------------------------------
    def enter(self) -> dict:
        """Create a state-entry record and run the in_hook: refresh phase,
        valid outgoing transitions, state-local instructions, durable progress."""
        self._entered = True
        spec = self.book.states[self.run.state]
        in_hook = spec.get("in_hook", {})
        for k, v in (in_hook.get("load_receipts") or {}).items():
            self.run.receipts.setdefault(k, v)
        self.run.log("state_entry", state=self.run.state)
        return {
            "phase": self.run.state,
            "prompt": spec.get("prompt", ""),
            "outgoing": [e["to"] for e in self.book.outgoing(self.run.state)],
            "durable_progress": dict(self.run.receipts),
        }

    # -- the goto protocol (paper Sec 3.3): six ordered steps --------------
    def goto(self, target: str) -> dict:
        run, book = self.run, self.book
        if not self._entered:
            self.enter()
        # 1. the requested edge must exist
        edge = book.edge(run.state, target)
        if edge is None:
            run.log("goto_rejected", target=target, reason="no such edge",
                    valid=[e["to"] for e in book.outgoing(run.state)])
            return {"ok": False, "stage": "edge", "reason": "no such edge"}
        # 2. evaluate the CURRENT state's before_transfer checks
        results = [run_check(c, run) for c in book.states[run.state].get("before_transfer", [])]
        failed = [r for r in results if not r.passed]
        if failed:
            run.failure_count += 1
            run.log("goto_blocked", target=target, stage="before_transfer",
                    failed=[{"check": f.name, "evidence": f.evidence, "detail": f.detail} for f in failed])
            return {"ok": False, "stage": "before_transfer", "failed": failed}
        # 3. run the out_hook persistence
        for k, v in (book.states[run.state].get("out_hook", {}).get("receipts") or {}).items():
            run.receipts[k] = v(run) if callable(v) else v
        # 4. edge guards + transfer hooks (routing sees checks/receipts/failures)
        if not guard_passes(edge.get("guard"), run, results):
            run.failure_count += 1
            alt = edge.get("on_guard_fail")
            run.log("goto_guard_failed", target=target, route=alt)
            if alt and book.edge(run.state, alt):
                return self.goto(alt) if alt != target else {"ok": False, "stage": "guard", "route": alt}
            return {"ok": False, "stage": "guard", "reason": "guard rejected"}
        # 5. commit + append history only now
        prev = run.state
        run.state = target
        run.log("transition", frm=prev, to=target)
        # 6. target entry + in_hook
        entry = self.enter()
        return {"ok": True, "frm": prev, "to": target, "entry": entry}
