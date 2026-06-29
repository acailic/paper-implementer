# Breakdown — GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents

> **Paper:** GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents
> **Authors:** Xiao Zhou, Siyue Zhang, Yilun Zhao, Jinbiao Wei, Tingyu Song, Arman Cohan, Chen Zhao (NYU Shanghai, Yale NLP Lab, NTU, Data & Models)
> **Year:** 2026 (arXiv:2606.24551v1, Jun 2026)
> **ArXiv:** https://arxiv.org/abs/2606.24551
> **Code (official):** https://github.com/rebeccaz4/gui-vs-cli
> **Type:** Benchmark + diagnostic analysis (no new model).

---

## 1. Problem & Motivation

**Problem.** Computer-use agents can execute desktop tasks through two modalities — GUI (screenshots + clicks) or CLI (programmatic skills). Existing evaluations confound modality with differences in tasks, initial states, verifiers, and permitted actions, so nobody can isolate *why* one modality outperforms the other.

**Why important.** Every agent framework must choose (or combine) an interaction strategy. Without controlled evidence, these choices are based on intuition and marketing claims rather than empirical grounding.

**Prior-work limitations** (the paper's diagnosis):
1. GUI benchmarks and CLI benchmarks use *different tasks* — cross-modality comparison is impossible.
2. They use *different initial states and verifiers* — success criteria aren't aligned.
3. They allow *different action spaces* — you can't tell if the gap is the model or the interface.
4. Nobody has measured how much of the CLI gap is explained by incomplete skill coverage.

## 2. Key Insight / Contribution

**Core idea (one sentence):** Hold task goals, initial states, and final-state verifiers fixed; vary only the native action interface — and you discover that the CLI gap is primarily a skill-coverage bottleneck, not a model capability gap.

**What is genuinely new:**
- The **matched execution-layer protocol** — identical tasks, states, verifiers across modalities.
- **440-task benchmark** across 18 applications and 12 workflow categories.
- **Verifier-guided skill-coverage diagnostic** — original skills satisfy only 37.6% of verifier checkpoints; patched skills raise CLI from 48.2% → 69.3%.
- **Complementary failure taxonomy** — 3 GUI failure modes + 3 CLI failure modes.
- **Procedure-guided grounding experiment** — explicit workflow steps barely improve GUI completion but cut wasted exploration by 20%.

## 3. Method

### 3.1 Benchmark composition

```
440 tasks
├── 18 applications (GIMP, Krita, draw.io, Audacity, MuseScore, Obsidian, Zotero,
│                    FreeCAD, CloudCompare, RenderDoc, LibreOffice Writer/Calc/Impress,
│                    Shotcut, OBS, Zoom, Godot 4, Chrome)
└── 12 workflow categories (Visual Design, Audio, Knowledge, CAD & 3D,
                            Graphics Debugging, Documents, Video & Streaming,
                            Spreadsheets, Presentations, Communication, Game, Web)
```

### 3.2 Three-stage construction pipeline

```
Stage I: Application & Task Selection
  Source: OpenComputer (Wei et al. 2026a)
  Filter: apps with CLI-Anything skill support
    ↓
Stage II: Task Rewriting & Curation
  Rewrite: step-by-step GUI instructions → modality-agnostic outcome descriptions
  Balance: remove modality-biased tasks, add tasks to offset imbalances
    ↓
Stage III: Manual Validation
  Verify: each task solvable in both modalities
  Check: instruction avoids modality-specific procedural cues
```

### 3.3 Action space enforcement

| Modality | Allowed | Disallowed |
|----------|---------|------------|
| **GUI** | Screenshot observation + screen-level actions (click, drag, type, scroll, keyboard shortcuts) | Code execution, shell commands, filesystem/database edits |
| **CLI** | CLI-Anything skill discovery/invocation, read-only inspection, verification | Direct file edits, Python/sed/awk for mutation, manual artifact editing |

### 3.4 Evaluation protocol

- **Final-state verification** — not trajectory matching. Executable verifier checkpoints over application state and artifacts.
- **Full pass rate** — all verifier checks for a task must pass.
- **Average task time** — wall-clock runtime per task as execution cost measure.

## 4. Math

No significant mathematical content. The key metric is:

```
Full Pass Rate = (# tasks where ALL verifier checkpoints pass) / (# total tasks)

Skill Coverage = (# Pass checkpoints) / (# all verifier checkpoints)

Original CLI-Anything coverage = 37.6%
Patched coverage (after repair) = 100%
```

The patched-skill setting uses verifier information during repair and is explicitly framed as a *diagnostic upper bound*, not a deployable baseline.

## 5. Evaluation Setup

### Models tested

| Modality | Models |
|----------|--------|
| **GUI** | GPT-5.4, Claude Sonnet 4.6, Claude Opus 4.7, EvoCUA-32B, Qwen3.5-27B, Kimi-K2.6 |
| **CLI** | Codex GPT-5.4, Codex GPT-5.5, Claude Code Sonnet 4.6, Claude Code Opus 4.7 |

### Three experimental settings

| Setting | What changes |
|---------|-------------|
| **Original skills** | CLI agents use unmodified CLI-Anything skills (37.6% coverage) |
| **Patched skills** | CLI skills repaired via verifier-coverage pipeline (100% coverage, diagnostic only) |
| **Procedure-guided GUI** | GUI agents receive explicit step-by-step workflow cues (176-task subset) |

## 6. Results & Analyses

### Main results (Table 1)

| Setting | Model | Full Pass | Avg Time |
|---------|-------|----------:|----------|
| GUI | GPT-5.4 | **59.1%** | 455.8s |
| GUI | Claude Opus 4.7 | 55.9% | 346.4s |
| GUI | Claude Sonnet 4.6 | 49.1% | 245.4s |
| GUI | EvoCUA-32B | 23.9% | 254.4s |
| GUI | Qwen3.5-27B | 19.3% | 1306.8s |
| GUI | Kimi-K2.6 | 38.6% | 1421.2s |
| CLI (original) | Codex GPT-5.5 | **48.2%** | 188.1s |
| CLI (original) | Codex GPT-5.4 | 24.3% | 254.4s |
| CLI (original) | Claude Code Sonnet 4.6 | 25.2% | 208.6s |
| CLI (original) | Claude Code Opus 4.7 | 24.3% | 248.8s |

### Skill-coverage diagnostic (Table 2)

| Setting | Full Pass | Avg Time |
|---------|----------:|----------|
| GUI GPT-5.4 (reference) | 59.1% | 455.8s |
| CLI Codex GPT-5.5 (original skills) | 48.2% | 188.1s |
| CLI Codex GPT-5.5 (patched skills) | **69.3%** | 162.6s |

> +21.2 percentage points from skill repair alone. CLI now beats GUI overall.
> Biggest gains: Knowledge (+264.3%), Graphics Debugging (+398%), Communication (+90%).

### Per-workflow modality advantages

- **GUI dominates:** Audio, Presentations, Communication, Web — interface-exposed workflows.
- **CLI competitive/stronger:** Visual Design, CAD & 3D, Documents, Video, Game — structured-artifact workflows.

### Procedural grounding (Table 4)

| | Full Pass | Avg Reward | Time |
|--|----------:|-----------:|-----:|
| Before | 59.7% | 0.7401 | 397.0s |
| After | 60.2% | 0.7576 | 314.8s |
| Δ | +0.8% | +2.4% | −20.7% |

### Failure taxonomy distributions

| CLI failure types | Share |
|-------------------|------:|
| Skill Coverage & Contract Gap | 93.8% (combined with unobservable semantics) |
| Implicit Default Reconstruction | part of 93.8% |
| Unobservable Application Semantics | part of 93.8% |

| GUI failure types | Share |
|-------------------|------:|
| Workflow Execution | 61.3% |
| UI Navigation & Control Discovery | 38.7% |

## 7. Limitations

- **No unrestricted agents.** Real deployed systems combine GUI + CLI + scripts + direct file edits. This paper restricts to modality-native actions for clean comparison, but results shouldn't be read as claims about unconstrained agents.
- **Patched skills are diagnostic, not deployable.** Repair uses verifier information → doesn't guarantee generalization to unseen tasks.
- **Failure taxonomy is coarse.** GUI failures often mix navigation with workflow execution; assigning a single primary label is a simplification.
- **Snapshot of models and skills.** Results are tied to specific 2026 models and CLI-Anything skill versions.
- **No new method proposed.** This is purely diagnostic/benchmark work.

## 8. Open Questions / Ideas

- **Build the benchmark construction pipeline.** The three-stage rewriting + verification protocol is the most re-implementable artifact. Automating Stage II (task rewriting) with an LLM is a natural extension.
- **Automatic skill construction.** The paper shows 37.6% coverage is the bottleneck. Can LLMs build CLI-Anything skills from application documentation without verifier peeking?
- **Hybrid agents.** The paper deliberately avoids hybrid settings. How much does combining GUI + CLI improve over either alone?
- **Per-workflow modality routing.** Since modality advantage is workflow-dependent, an adaptive router could pick GUI vs CLI per task.
- **Scale the failure taxonomy.** 80 failed trajectories per modality is a start — scaling to thousands would give finer-grained proportions.
