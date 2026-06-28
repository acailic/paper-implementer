# Implementation — Are We Ready For An Agent-Native Memory System?

> Companion notes for the from-scratch re-implementation of the 4-module
> agent-memory system (`M_sys = ⟨R, S, Q, U⟩`) whose design choices are the
> *winning* ones validated by the paper's ablations.
> See `../../AGENTS.md` Step 5 and `../breakdown.md` for the full rationale.

## Paper reference
- **Title:** Are We Ready For An Agent-Native Memory System?
- **ArXiv:** https://arxiv.org/abs/2606.24775
- **Official code:** https://github.com/OpenDataBox/MemoryData (not consulted — implemented from the paper)

## What's implemented

A minimal but faithful 4-module agent-memory system, **plus** a toy
LoCoMo-style multi-session benchmark that exercises it:

| Module | Paper concept | This implementation |
|--------|---------------|---------------------|
| **R** Representation & Storage | Heterogeneous composite objects + multi-engine backend | `MemoryObject` (text + metadata{role, session, timestamp, entity_ids} + embedding + links) backed by a dense vector index (brute-force cosine) + a BM25 inverted index + an entity adjacency dict |
| **S** Extraction | Late-filtering, schema-free, preserve raw context (Finding 7 / O9) | Each dialogue turn stored **verbatim**; lightweight capitalized-word entity extraction. No summarization (raw text wins per M1 / O8) |
| **Q** Retrieval & Routing | Balanced hybrid fusion (dense + BM25 via RRF) + explicit planning; **no** reflection (Finding 8 / O10) | `retrieve()` fuses dense-KNN + BM25 via Reciprocal Rank Fusion (k=60); optional query-expansion planning step splits multi-constraint queries |
| **U** Maintenance | Timestamp multi-versioning (logical invalidation, append-only) + conservative consolidation + score-based eviction (Findings 3, 9 / O7, O11) | On conflicting fact: logically invalidate old (`valid=False`, `valid_until=now`, `superseded_by=new`). Conservative merge only when entity-overlap high AND text cosine > 0.85. Heat-score eviction (`α·access − β·exp(−λ·age)`) of stale-then-cold entries when over capacity |

The four modules map 1:1 to the paper's tuple and are each independently
switchable, so the ablation dimensions from §5 are reproducible in principle.

## How to run

```bash
pip install -r requirements.txt     # numpy only
python3 run.py
```

No API keys, no model downloads, no GPU. Runs in ~1 second on CPU.

## What it does (run.py)

1. Loads a **toy 4-session dialogue** between a user (Alice) and an assistant.
   Facts are deliberately **revised across sessions**: "I live in Paris" →
   later "I moved to London"; "I work as a nurse" → "I now work as a doctor".
   This exercises the temporal-update module (RQ3 in the paper).
2. Ingests all 32 turns into the memory system (Module S), building the dense +
   BM25 + entity indices (Module R).
3. Runs **10 QA queries**: 3 temporal-revision (must return the *new* value),
   multi-hop entity (Rex → breed; Bob → city), and single-hop factual.
4. For each query prints: the retrieved evidence with RRF scores, whether each
   retrieved item is gold (`<GOLD>`), the predicted answer, and substring EM.
5. Prints **Recall@1/3/5** summary, an ASCII cumulative-EM curve, a per-query EM
   bar chart, and a Module-U invalidation demo (Paris assertion marked
   `valid=False`, superseded by the London assertion).
6. Writes `metrics.json`.

## Real results (from an actual run)

```
Ingested 32 turns across 4 sessions -> 32 memory objects.
Indices: dense(rows=32) bm25(terms=129) entity(nodes=28)

metrics:
    Recall@1     : 0.80      # 8/10 queries had gold evidence at rank 1
    Recall@3     : 1.00      # 10/10 within top-3
    Recall@5     : 1.00      # 10/10 within top-5
    SubstringEM  : 0.90      # 9/10 answers exactly correct

invalidation_ok : True       # Paris (old) invalidated, London (new) current
```

The one EM miss is Q8 ("Where did Alice live *before* London?") — a query that
must *deliberately* surface an invalidated fact, which the default
`valid_only=True` retrieval filter excludes. This is a known limitation (see
below), not a bug.

### Temporal-revision correctness (the RQ3 demo)

```
OLD (invalidated):  #0  "Hi, I'm Alice. I live in Paris."        valid=False
                    #14 "I still live in Paris, near the Louvre." valid=False
NEW (current):      #16 "Big news: I moved to London last month." valid=True

Q1 "Where does Alice live now?" -> "London"  EM=1   (NOT "Paris")
Q2 "What is Alice's current job?" -> "doctor" EM=1  (NOT "nurse")
```

The multi-versioning (Module U) correctly prevents "hallucinations of the past"
— the exact failure mode the paper calls out in Finding 3 / O4.

## Known gaps / limitations

1. **Pseudo-embeddings, not a real encoder.** Because this must run with no API
   keys and no model downloads, `FakeEmbedder` builds a deterministic
   char-n-gram TF vector hashed to a fixed dimension. Cosine is meaningful on a
   tiny toy dataset but is far weaker than a real sentence-embedding model. The
   paper's backbone-robustness finding (O5) predicts the *ordering* of good vs
   bad designs stays stable across backbones, but absolute Recall@1 would rise
   substantially with a real encoder. **This is the biggest gap vs the paper's
   systems.**

2. **No LLM in the loop.** The paper's systems use an LLM for extraction (S),
   consolidation (U), and answer generation. This implementation uses
   deterministic heuristics (regex entities, strict-threshold merge,
   substring-answer extraction). The *architecture* is identical; the
   *components* are stubs. Wiring a real LLM into `extract()`,
   `consolidate()`, and `answer()` is a drop-in change.

3. **Toy dataset, not LoCoMo.** 4 sessions / 32 turns vs the paper's
   full LoCoMo + LongMemEval + DB-Bench suite. The toy is enough to
   demonstrate all four modules and the temporal-revision behavior, but
   the absolute numbers are not comparable to the paper's benchmarks.

4. **No reflection ablation wired through run.py.** The paper's M3 result
   (Planning+Reflect < Planning-Only) is encoded as the *absence* of a
   reflection step in `retrieve()`; there's no toggle to turn it on for a
   head-to-head. Adding one would be a small change to `AgentMemory.retrieve`.

5. **Invalidation is default-on.** Retrieval filters `valid=True` by default.
   A query that legitimately asks about a *past* state (Q8) must opt into
   `include_invalid=True`. A more complete system would infer this from query
   intent.

## Differences from the original

This is **not** a port of `OpenDataBox/MemoryData`. It is a from-scratch
implementation of the *framework* (the 4-module tuple and the winning design
choices the ablations point to), built to be readable and dependency-free rather
than to reproduce the paper's benchmark numbers. The paper itself never builds
"the implied winner" as a single system — it leaves that as an exercise. This
is that exercise.

## Files
- `agent_memory.py` — `FakeEmbedder`, `MemoryObject`, `AgentMemory` (R/S/Q/U)
- `data.py` — toy multi-session dataset + dataloader
- `run.py` — benchmark runner + ASCII charts
- `metrics.json` — last run's metrics (regenerated each run)
- `requirements.txt` — `numpy`
