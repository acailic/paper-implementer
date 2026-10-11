# Notes — StateM (first pass)

> Qin, Lu, Wang, Wang — "StateM: Reaching 95.3% Raw Accuracy, or a $15
> Frontier Run, on Terminal-Bench 2.1 via Harness Scaling". arXiv:2608.15089.

## One-line

Harness scaling: don't change model weights — change the execution system
around the model. StateM = YAML-configured state-machine runtime the agent
drives via CLI; states are context+contract boundaries; transitions are
checked, logged, recoverable.

## Core claims (Sec 4)

- GPT-5.5 xhigh + StateM: **92.1%** on Terminal-Bench 2.1 vs 83.1% reference;
  runbook transfers unmodified to GPT-5.6 (>Sol Ultra 91.9%).
- GPT-5.6 Sol xhigh: 95.3% raw across 445 trials, ≥1 success on all 89 tasks.
- Frozen profile lifts GPT-5.6 Luna 76.7 → 85.4 (above Sol xhigh ref 84.9).
- **$15 frontier run**: DeepSeek-V4 Flash 82.7 → 88.1 with <$38 adaptation,
  total recorded spend $52.22 vs $574.68 for the GPT reference.
- BusinessBench: heterogeneous tasks gain little (0.55/1.34 pts) but
  mechanism-matched families gain 10.04 — rules generalize when execution
  structure matches, not by topic.

## The runtime (Sec 3 — what to re-implement)

- Runbook ℬ = (S, s₀, S_T, E, Φ): states, initial, terminals, permitted
  edges, per-state spec (prompt, hooks, checks, artifacts). YAML, versioned.
- State anatomy: `in_hook` (load durable context/progress/state files) →
  **state body** (agent's open-ended work — model discretion preserved) →
  `out_hook` (persist progress, write receipts) → `before_transfer` (blocking
  checks; failure keeps the run IN the state).
- `goto TARGET` protocol (6 ordered steps): (1) edge exists; (2) source
  state's before_transfer checks; (3) out_hook persistence; (4) edge guards +
  transfer hooks; (5) commit + append history only if all pass; (6) target
  entry + in_hook. "Checked, logged, recoverable — not transactional."
- Check types by evidentiary strength: command/predicate (host-reproducible)
  > manual > checklist/message (structured self-attestation) > llm_review
  (semantic, not deterministic). A receipt is never proof by structure alone.
- Repair is explicit graph routing (Verify --fail--> Repair), not an
  instruction the agent must remember. Guards see check results, receipts,
  failure counts, external readiness.
- Runtime vs control profile: runtime = generic mechanisms (persistence,
  validation, hooks, history, recovery); profile = workflow-specific runbook
  content. Benchmark results evaluate RUNTIME + evolved profile jointly —
  the abstraction alone is not isolated.

## Why it works (mechanism)

- Control signal decay: plan authority weakens as trace grows → each state
  entry re-anchors phase + obligations + durable progress (outside model
  context, not hidden from agent/user — shared workspace).
- Ambiguous current state from append-only history → authoritative phase is
  explicit runtime state, not reconstruction.
- Harness learning = selective procedural memory: postmortem findings become
  persistent executable preconditions (checks), not prose the model may skip.
- 22-hour development run endurance: durable states survive context resets.

## Transfer hierarchy (Sec 4.3)

Same model + better harness ≈ a model-generation gain; profile transfers
cleanly within model family; cross-family needs cheap re-adaptation ($38);
heterogeneous task domains gain only where execution structure matches.

## Open questions / skeptic notes

- Raw accuracy = pre-adjudication; "≥1 success in 445 trials" is a max, not a
  mean — the paper is explicit about this, good.
- Benchmark-adapted profile on a public leaderboard = overfitting surface;
  the BusinessBench held-out result is the honest counterweight.
- Evidentiary strength ladder is the most reusable idea independent of the
  benchmark numbers.

## What the re-implementation should be

A from-scratch StateM-style runtime in stdlib Python: runbook YAML loader
(keep stdlib-only: use a tiny embedded YAML subset or JSON runbooks), the
goto protocol exactly as specced (ordered, blocking, logged), all four check
types, guard-routed edges, receipts + history journal, recovery on resume —
demoed by a deterministic scripted "agent" walking Plan → Execute → Verify →
(fail → Repair) → Handoff, with one transition blocked by a failing command
check and then recovered. No LLM calls needed to demonstrate the control
surface.
