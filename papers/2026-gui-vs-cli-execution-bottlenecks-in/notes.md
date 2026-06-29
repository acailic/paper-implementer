# Notes — GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's a **benchmark + diagnostic analysis paper**, NOT a new-model paper. The authors do four things:

| # | What | Output |
|---|------|--------|
| 1 | Build a **matched execution-layer benchmark** | 440 tasks, 18 apps, 12 workflow categories |
| 2 | Compare **GUI vs CLI** agents under identical conditions | Same goals, states, verifiers, different action spaces |
| 3 | Run **skill-coverage diagnostics** (patched skills) | Estimate recoverable CLI gap |
| 4 | Produce a **complementary failure taxonomy** | 3 GUI + 3 CLI failure modes |

No new algorithm. The re-implementable artifact is the **benchmark protocol itself**: the construction pipeline, verifier-based evaluation, and the diagnostic skill-coverage methodology.

## The big picture

Can a computer-use agent do the same task better through a GUI or through CLI skills? The answer: it depends on the workflow structure, but the CLI gap is mostly a *skill coverage* problem, not a model capability problem. When you fix the skills, CLI can exceed GUI.

## The matched protocol (why this paper matters)

Prior work confounds four things simultaneously:
1. **Different tasks** per modality
2. **Different initial states**
3. **Different verifiers**
4. **Different action spaces**

This paper holds #1-#3 constant and only varies #4. That's the whole contribution — a clean experimental design that nobody else has done.

### Three-stage construction pipeline

```
Stage I: App + Task Selection
  Start from OpenComputer tasks → select apps with CLI-Anything skill support
    ↓
Stage II: Task Rewriting
  Rewrite GUI-oriented step-by-step instructions → modality-agnostic descriptions
  Remove modality-biased tasks, balance distribution
    ↓
Stage III: Manual Validation
  Check each task is solvable in both modalities
  Ensure instruction avoids modality-specific procedural cues
```

## Key numbers to remember

| Setting | Model | Full Pass Rate | Avg Time |
|---------|-------|---------------:|----------|
| **GUI (best)** | GPT-5.4 | **59.1%** | 455.8s |
| GUI | Claude Opus 4.7 | 55.9% | 346.4s |
| GUI | Claude Sonnet 4.6 | 49.1% | 245.4s |
| **CLI (best, original skills)** | Codex GPT-5.5 | **48.2%** | 188.1s |
| CLI | Codex GPT-5.4 | 24.3% | 254.4s |
| CLI | Claude Code Opus 4.7 | 24.3% | 248.8s |
| **CLI (patched skills)** | Codex GPT-5.5 | **69.3%** | 162.6s |

> GUI best (59.1) > CLI best original skills (48.2). But CLI patched (69.3) > GUI best (59.1).
> The CLI gap is a skill-coverage problem.

### Skill coverage fact: only **37.6%** of verifier checkpoints satisfiable with original skills.

## Per-workflow winners

| Workflow | GUI best | CLI best (original) | CLI best (patched) |
|----------|---------|---------------------|---------------------|
| Visual Design | 47.1 (GPT-5.4) | 54.9 (Codex GPT-5.5) | 66.7 |
| Audio | 73.5 (Claude Opus) | 42.9 (Codex GPT-5.5) | 81.6 |
| Knowledge | 42.9 (GPT-5.4) | 22.4 (Codex GPT-5.5) | 81.6 |
| CAD & 3D | 63.3 (Claude Opus) | 67.3 (Codex GPT-5.5) | 73.5 |
| Graphics Debugging | 51.2 (Claude Opus) | 9.8 (Codex GPT-5.5) | 48.8 |
| Documents | 60.5 (GPT-5.4) | 60.5 (Codex GPT-5.5) | 86.8 |
| Video & Streaming | 61.1 (Claude Opus) | 47.2 (Codex GPT-5.5) | 41.7 |
| Spreadsheets | 78.1 (GPT-5.4) | 46.9 (Codex GPT-5.5) | 65.6 |
| Presentations | 70.0 (GPT-5.4) | 50.0 (Codex GPT-5.5) | 95.0 |
| Communication | 84.2 (Claude Opus) | 35.3 (Codex GPT-5.5) | 35.3 |
| Game | 88.2 (Claude Opus) | 89.5 (Codex GPT-5.5) | 100.0 |
| Web | 84.2 (Claude Opus) | 100.0 (Codex GPT-5.5) | 100.0 |

> Key insight: GUI dominates Audio, Presentations, Communication (interface-exposed workflows). CLI competitive in Visual Design, CAD, Documents, Video, Game (structured-artifact workflows).

## The failure taxonomy

### CLI failures (93.8% in two categories)
| Error Type | Description |
|-----------|-------------|
| **Skill Coverage & Contract Gap** | Required operations not in skill library, or documented behavior ≠ actual behavior |
| **Implicit Default Reconstruction** | Agent must guess defaults that GUI users get automatically (object names, identifier rules) |
| **Unobservable Application Semantics** | Critical state not exposed through skills; agent hallucinates plausible defaults |

### GUI failures (61.3% workflow + 38.7% navigation)
| Error Type | Description |
|-----------|-------------|
| **UI Navigation & Control Discovery** | Can't locate correct menus/tabs/dialogs/hidden settings |
| **Workflow Execution** | Wrong order, missing confirmations, premature termination of long sequences |
| **Self-Checking & Verification Gap** | Declares success without verifying exported files / saved state |

## The procedural grounding experiment

| Setting | Full Pass | Avg Reward | Time |
|---------|----------:|------------:|-----:|
| Before grounding | 59.7% | 0.7401 | 397.0s |
| After grounding | 60.2% | 0.7576 | 314.8s |
| Change | ↑0.8% | ↑2.4% | ↓20.7% |

> Giving GUI agents explicit procedure steps barely helps completion (+0.8%) but cuts wasted exploration by 20%.
> GUI's real bottleneck is visual grounding + long execution chains, not just "not knowing the steps."

## The four-phase verifier-coverage pipeline

1. **Verifier-to-skill mapping** — inspect verifier code, harness code, skill docs; label each checkpoint Pass/Partial/Fail
2. **Skill implementation & tests** — repair Partial/Fail checkpoints, add comprehensive tests
3. **Coverage report** — app-level README with pass/fail totals
4. **Skill documentation update** — update SKILL.md to match actual capabilities

## Interaction modality can compensate for model limitations

GPT-5.4 through GUI (59.1%) > Codex GPT-5.5 through CLI original skills (48.2%), even though GPT-5.5 is a stronger model. The interface matters more than raw model strength when the skill layer is incomplete.

## Terms I had to look up

| Term | Meaning |
|------|---------|
| **CLI-Anything** | Framework that packages desktop apps as reusable CLI harnesses with skill layers |
| **OpenComputer** | Verifiable desktop task benchmark (Wei et al. 2026) |
| **Full pass rate** | All verifier checkpoints must pass for a task to count as success |
| **Skill contract** | The set of operations exposed by a CLI skill, including documented vs actual behavior |
| **EvoCUA** | Open computer-use agent (Xue et al. 2026) |
