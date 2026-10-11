# Breakdown — StateM

> Qin, Lu, Wang, Wang — "StateM: Reaching 95.3% Raw Accuracy, or a $15
> Frontier Run, on Terminal-Bench 2.1 via Harness Scaling". arXiv:2608.15089.
> [PDF](https://arxiv.org/abs/2608.15089) · Core code open-sourced by authors.

---

## 1. Problem & Motivation

**Problem.** Long-horizon agents fail while their models can solve every
constituent step: mutable state gets lost, earlier lessons don't reactivate,
known procedures get skipped, runs stop before the deliverable is verifiably
complete.

**Two operational pressures.** (1) A plan's control signal decays as the
trace grows. (2) Reconstructing "where are we" from an append-only history is
ambiguous. **Design tension:** runtime enforceability vs agent autonomy —
micro-managing every action destroys the model's discretion.

**Why existing answers fall short.** Planning frameworks add pre-execution
structure but nothing durable enforces it mid-run; orchestration tools hide
state in tooling the agent can't audit; agent-memory systems store knowledge
but rarely *execute* obligations; harness-adaptation work tunes prompts, not
the execution substrate.

## 2. Key Insight / Contribution

**Harness scaling:** improve the execution system around a frozen model.
StateM = an **agent-native runtime** organized as a versioned state-machine
runbook the agent drives through ordinary CLI commands:

1. **States as context+contract boundaries** — coarse phases (not micro-actions);
   `in_hook` re-anchors phase-local instructions + durable progress; `out_hook`
   persists; `before_transfer` checks block exit.
2. **Checked, logged, recoverable transitions** — the ordered `goto` protocol;
   failure keeps the run in the source state with the reason visible.
3. **Evidentiary-strength ladder for checks** — command/predicate >
   manual > checklist/message self-attestation > llm_review; structure is
   never mistaken for proof.
4. **Repair as explicit graph routing** + harness learning as selective
   procedural memory: postmortems become executable preconditions
   ("golden rules"), not prose.

Shared workspace: agent operates, user audits, both see the same artifacts.

## 3. Method

- Runbook ℬ = (S, s₀, S_T, E, Φ) in YAML; runtime keeps mutable per-run state
  separate; run-local additions recorded but distinct from the versioned base.
- `goto TARGET` protocol (order is the contract):
  1. edge (current → TARGET) must exist;
  2. evaluate current state's `before_transfer` checks;
  3. run `out_hook` persistence;
  4. evaluate edge guards + transfer hooks (routing sees check results,
     receipts, failure counts);
  5. commit + append history **only if all pre-commit steps succeed**;
  6. create target entry, run its `in_hook`.
- Check types with independent evidentiary claims (see ladder above).
- Recovery: resume re-enters the authoritative state with its obligations —
  endurance demonstrated by a 22-hour development run.

## 4. Math

State machine formalism: ℬ = (S, s₀, S_T ⊆ S, E ⊆ S×S, Φ). Transition
commit predicate: commit(e) ⟺ edge(e) ∧ ∀c ∈ before_transfer(src): c ✓ ∧
out_hook(src) ✓ ∧ guard(e) ✓. Nothing exotic — the contribution is the
protocol discipline, not equations. Results are counts/proportions with
cost accounting (88.09% = 392/445; $15 final-score evidence; $52.22 total
recorded DeepSeek spend vs $574.68 GPT reference).

## 5. "Training" (evaluation setup)

- Terminal-Bench 2.1 (89 tasks, 445 public-submission trials), models:
  GPT-5.5/5.6 (xhigh, Sol, Luna), DeepSeek-V4 Flash; base runbook + evolved
  benchmark profile; frozen profiles for transfer arms.
- BusinessBench: family-specific runbooks developed on dev sets, first frozen
  held-out eval — the honesty check for rule generalization.
- 22-hour continuous development run as endurance evidence.

## 6. Results

- 83.1 ref → **92.1** (GPT-5.5 xhigh + StateM) → runbook transfers to 5.6
  unchanged → **95.3 raw** (Sol xhigh, ≥1/89 all tasks) — a model-generation-
  sized gain without touching weights.
- Luna +StateM 85.4 > Sol-xhigh reference 84.9 (frozen profile).
- DeepSeek-V4 Flash: 82.7 → 88.1 (<$38 adaptation); transfer follows model
  distance (Sec 4.3 hierarchy).
- BusinessBench aggregate +0.55/+1.34 but +10.04 on mechanism-matched
  families — **rules generalize by execution structure, not topic**.
- Failure-boundary analysis (4.6) identifies where state checks intervene.

## 7. Limitations (stated & derived)

- Stated: runtime ≠ profile — benchmark results evaluate the joint system;
  raw accuracy is pre-adjudication; "≥1 success" is a max-statistic.
- Derived: benchmark-adapted profiles on public leaderboards are an
  overfitting surface (BusinessBench is the counterweight); llm_review checks
  are semantic opinions with structure around them; no general rollback
  ("recoverable, not transactional").

## 8. Re-implementation plan

From-scratch, stdlib-only Python (`implementation/`):

| Paper concept | File/function |
|---|---|
| Runbook ℬ + Φ | `statem.py: Runbook.load` (JSON runbooks — stdlib-only) |
| goto protocol, 6 ordered steps | `statem.py: Runtime.goto` (blocking, ordered, logged) |
| Check types + strength ladder | `statem.py: CHECKS` (command/predicate/manual/checklist/llm_review) with `evidence` label |
| Edge guards + repair routing | `statem.py: Runtime._guards` |
| History journal + receipts | `statem.py: Run` (append-only events, resume) |
| Demo agent | `run.py` — deterministic scripted agent: Plan → Execute → Verify (blocked by failing command check) → Repair → Verify ✓ → Handoff |

Demo assertions: a failing `before_transfer` keeps the run in state with the
reason visible; resume re-enters the authoritative state; history is the
audit trail. Output `metrics.json` (transition counts, blocked attempts,
check evidence types used).
