# StateM runtime — re-implemented from scratch

> Qin, Lu, Wang, Wang — "StateM: Reaching 95.3% Raw Accuracy, or a $15
> Frontier Run, on Terminal-Bench 2.1 via Harness Scaling".
> arXiv:2608.15089, 2026. [Paper](https://arxiv.org/abs/2608.15089)

The paper's contribution is **harness scaling**: improve the execution system
around a frozen model. This is a stdlib-only re-implementation of the StateM
control surface — the state-machine runtime the agent drives through `goto` —
demoed end-to-end by a deterministic scripted agent (no LLM needed to show
the control properties).

## Quick start

```bash
python3 run.py        # no dependencies
```

Watch: a blocked transition (failing command check keeps the run IN the
state), guard-routed repair, explicit terminal success, and resume from the
journal. `metrics.json` records transitions / blocks / evidence classes.

## What is re-implemented

| Paper concept | Where |
|---|---|
| Runbook ℬ = (S, s₀, S_T, E, Φ), versioned artifact | `Runbook.load` (`runbooks/demo.json`) |
| State anatomy: `in_hook` → body → `out_hook` → `before_transfer` | `statem.StateM.enter`, runbook spec |
| **goto protocol, six ordered steps** — edge → checks → persistence → guards → commit+log → entry | `statem.StateM.goto` |
| **Evidentiary-strength ladder**: command/predicate > manual > checklist/message > llm_review | `statem.EVIDENCE`, `run_check` (every result carries its class) |
| Guards + `on_guard_fail` repair routing (repair is graph, not prose) | `statem.guard_passes`, edge spec |
| Append-only history, receipts, **resume** (authoritative phase without trace reconstruction) | `statem.Run` |

## Declared deltas from the paper

- Runbooks are JSON, not YAML (stdlib-only constraint; structure identical).
- `llm_review` and `manual` checks record opinions/decisions with no
  verifier attached — which is precisely the paper's point: structured
  self-attestation and semantic judgments are labelled weaker evidence than
  host-reproducible checks, and structure is never mistaken for proof.
- The demo agent is scripted; the paper's headline numbers (92.1/95.3/88.1,
  $15 vs $574.68) require the real Terminal-Bench suite and model APIs.

## Files

| File | Purpose |
|---|---|
| `statem.py` | The runtime: runbook, checks ladder, guards, goto protocol, run journal |
| `runbooks/demo.json` | Plan → Execute → Verify → Repair → Handoff coding runbook |
| `run.py` | Deterministic scripted agent exercising every control property |
| `work/`, `metrics.json` | Demo artifacts + counters |
