# Implementation — Apodex 1.1: Scaling Agentic Intelligence for Complex Work

> Companion notes for the code re-implementation in `implementation/`.
> Paper: arXiv:2608.23283 — https://arxiv.org/abs/2608.23283

## What's implemented

**ChainWorld** — a toy executable task instantiating the paper's task contract
(Eq. 1–5), AgentOS runtime state (Eq. 7), acquisition pressure (Eq. 6),
publication lease + manifest replay, hindsight **PIVOT-RL** (pivot detection,
state-restored hinted localized continuations mixed with full tasks), and a
scripted Agent Team (decompose → delegate → asymmetric verify → synthesize).
Policy: small pointer-transformer trained with batched REINFORCE.

Full details, honest results, and documented deviations: see
`implementation/README.md`.

## How to run

```bash
cd implementation
pip install -r requirements.txt
python data.py     # mechanism selftests
python train.py    # 3 arms + AT script -> results.json (~2 min CPU)
```

## Results

All three arms (full-task PG, PIVOT-RL, random-cut continuations) reach 100%
eval SR with the optimal 9-step provenance-following policy (2L+1). PIVOT-RL
did **not** accelerate learning at this 14-step horizon — episodes to ≥90%:
full 1920 < pivot 3249 < randcut 4160, roughly tied per wall-clock round; the
hint-discipline claim does replicate (100% with hints absent at eval). The
scripted Agent Team also hits 100% but spends 18 tool calls vs the solo
policy's 14 — coordination lift, not compute-matched, mirroring the paper's
own missing control.

## Known gaps / TODOs

Async human interventions (12-category), evidence-grounded synthesis, tiered
compaction, SFT mixture, asynchronous optimization over heterogeneous
rollouts — out of scope at toy scale.

## Differences from the original

1. Verifier-derived reward shaping added (pure terminal reward gives zero
   gradient at toy scale; shaping cannot be gamed into success — exact-set
   delivery stays fatal).
2. Pivot detection is oracle-based (we own the generator), not model-based
   retrospective analysis.
3. Scale: 12 files / 4-step chain / 25 actions vs 397B-parameter model on
   multi-benchmark agent workloads.
