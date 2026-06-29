# Agent Memory System — "Are We Ready For An Agent-Native Memory System?"

From-scratch Python implementation of the 4-module memory architecture whose
design choices are the **winning ones** validated by the ablations in:

> Wei Zhou et al., "Are We Ready For An Agent-Native Memory System?"
> arXiv:2606.24775, 2026. [PDF](https://arxiv.org/abs/2606.24775)

The paper decomposes agent memory into `M_sys = ⟨R, S, Q, U⟩` and benchmarks
each module independently across 5 workloads. It never builds the implied
winner as one system — **this implementation does.**

## Quick start

```bash
pip install numpy          # only external dep
python3 run.py
```

Output: per-query retrieval + answer, Recall@1/@3/@5 table, substring-EM,
ASCII quality curve, multi-versioning invalidation demo, and `metrics.json`.

## Architecture — winning design choices from the paper

| Module | Paper finding | Implementation |
|--------|--------------|----------------|
| **R** (Representation) | Heterogeneous composite objects | Each memory object = `{id, text, metadata{role,session_id,timestamp,entity_ids}, embedding, links[]}`. Multi-engine backend: dense vector index (brute-force cosine) + BM25 inverted index + entity adjacency graph. |
| **S** (Extraction) | **F7/O9 — Late Filtering**: preserve context at write time; raw > summary | Each dialogue turn stored **verbatim**. Lightweight entity extraction (capitalized-word heuristic). No summarization. |
| **Q** (Retrieval) | **F8/O10 — Balanced hybrid**: dense KNN + BM25 fused via RRF, with explicit planning. No reflection. | Dense cosine KNN + BM25 → Reciprocal Rank Fusion (k=60). Optional query-expansion planning (split multi-constraint queries). One-hop graph expansion over entity adjacency. Recency re-rank for "current/now" queries. No reflection. |
| **U** (Maintenance) | **F3** — timestamp multi-versioning; **F9/O11** — conservative consolidation; **O7** — localized maintenance | Logical invalidation (valid=False) on conflicting updates, append-only. Conservative merge only when entity overlap + cosine > 0.85. Heat-score eviction (α·access − β·exp(−λ·age)). |

## Files

| File | Purpose |
|------|---------|
| `agent_memory.py` | Core 4-module system: `AgentMemorySystem`, `MemoryObject`, `FakeEmbedder`, all index backends |
| `data.py` | Toy LoCoMo-style multi-session dataset (4 sessions, revised facts) + 10 QA queries with gold answers |
| `run.py` | Main runner: ingest → retrieve → evaluate → print metrics + ASCII charts |
| `requirements.txt` | `numpy` (only dep) |
| `README.md` | This file |

## Results (on the toy dataset)

```
Metric                  Value
------------------------------
Recall@1                0.800
Recall@3                1.000
Recall@5                1.000
Substring EM            1.000
```

Multi-versioning invalidation demo:
- "I live in Paris" → logically invalidated (valid=False, superseded_by=moved-to-London)
- "I moved to London" → current valid version (version_of=Paris-id)
- Invalidated objects are **kept** in the store (append-only) — retrievable for historical queries

## Known gaps / limitations

1. **FakeEmbedder** — Deterministic char-n-gram TF hashing, NOT a semantic encoder.
   Cosine similarity is a meaningful proxy for surface/lexical overlap but has
   zero semantic generalization. Swap in `sentence-transformers` or an API
   encoder for production fidelity. The paper itself (O5/O8) shows retrieval
   quality is dominated by *pipeline design*, not backbone.

2. **Answer extraction is heuristic** — The `answer_from_evidence` function uses
   regex-based slot filling instead of an LLM. This works for the toy dataset
   but would not generalize. In a real system the retrieved evidence would be
   passed to a generator.

3. **Conflict detection is regex-based** — The invalidation heuristic matches
   on predicate patterns ("lives in", "works as") and extracts values via
   regex. A real system would use an LLM for conflict detection and value
   comparison.

4. **Tiny dataset** — 4 sessions, 32 turns, 10 queries. The paper evaluates on
   LoCoMo, LongMemEval, DB-Bench (100s of sessions). This is a proof-of-concept
   that exercises all four modules, not a benchmark replication.

5. **No persistence** — Everything is in-memory. The paper's physical storage
   taxonomy includes databases, vector stores, etc. This toy uses numpy arrays +
   Python dicts.

6. **Single-user assumption** — The invalidation logic assumes all "I" statements
   refer to the same subject. Multi-user scenarios need per-subject tracking.

7. **BM25 is approximate** — Term frequency is approximated (presence-only). A
   real BM25 implementation would count actual term frequencies per document.
