# Writeup — Are We Ready For An Agent-Native Memory System?

> How I'd explain this to a friend over a beer, if they asked "what have you been reading?"

> **Languages:** English (this file) · [Srpski](writeup-sr.md)

---

The simple story goes something like this.

Agent memory has grown from "search the transcript and see what pops up" into a
proper little database system — somewhere things get packed, somewhere they get
looked up, somewhere they get changed, somewhere they get deleted. This isn't
just some RAG sitting on top of chat history anymore. It's infrastructure.

The problem is that all the evaluation is stuck in the last decade. People
measure end-to-end F1 and say "Mem0 is 21, Zep is 84" — and stop there. Nobody
can tell you *why*. Is it because of how they store memory? How they pull it
out? How they maintain it? Dead end.

This paper says: stop. Break every memory system into four parts, then measure
each one separately.

## The four modules

Every agent-memory system in the world is really the same pattern, just filled
in differently:

- **R** — how you remember and where it sits (flat text? graph? composite?) and
  physically where (in context? in a vector DB? in multiple engines at once?)
- **S** — how raw dialogue becomes memory (just glue the lines together? extract
  facts? parse into triplets?)
- **Q** — how you find what you need when a query comes in (attention? vector
  search? walking a graph? LLM as planner? hybrid?)
- **U** — how it's maintained over time (append with timestamps? FIFO eviction?
  LLM merges similar things? offline fine-tuning?)

The authors write this as `M_sys = ⟨R, S, Q, U⟩` and then — and this is the
actual contribution — benchmark each slot independently, with proper
evidence-level metrics and with cost. Not just "is the answer good" but "did you
even surface the evidence you needed" and "how many seconds did that cost you
per query."

And they test 12 serious systems across 5 different workloads. Suddenly you can
see where each one cracks.

## Two things that genuinely surprised me

First — and this was my biggest "aha" moment — **memory retrieval isn't a
ranking problem, it's an evidence-assembly problem.**

The RQ2 experiment is the key. SimpleMem gets Recall@1 (first hit) of 39 —
phenomenal in that single glance, it immediately pulls out the one obviously
important line. But the moment you widen the budget to Recall@5 and Recall@10,
or the moment the evidence sits further back in history, A-MEM and MemTree blow
it out of the water (69.5/85.9 and 59.7/80.5 vs flat Embedding RAG which drops
from 37.1 to 7.4 F1).

What does that mean? The hard case isn't finding the *first* relevant thing.
The hard part is gathering the *complete* scattered set of memories needed to
answer — and that's where explicit structure, links, hierarchy pays for itself.
Flat similarity is a short-range weapon. That's something you carry with you
even after you close the paper.

Second — **structure doesn't drive cost, maintenance scope does.** This is O7
and it actually flips the whole "graph vs vector" debate. People assume
structured memory (graphs, hybrids) is expensive because it's structured. The
data says the opposite: it's expensive when every write **propagates globally**.
LightMem and MemTree stay cheap because their writes are local — a new fact
touches only its own subtree. Cognee and Zep get expensive (116s, 155s) because
every write triggers a whole-graph consolidation. Structure is fine. **Global
recomputation is the enemy.** That's a concrete engineering principle, not just
a number on a table.

## What I learned when I coded it myself

When you build those four modules from scratch (see `implementation/`), three
things surface that the paper implies but doesn't say out loud:

**The modules are more coupled than the taxonomy looks.** You can't just freely
pick R, S, Q, U independently. For example, timestamp multi-versioning (a U
choice) only works if R exposes per-entity version chains — otherwise you can't
bind a revised fact to the entity it updates. And balanced hybrid retrieval (Q)
only pays off if S preserved enough raw text for BM25 to match against. The
tuple is a *lens*, not an orthogonality guarantee.

**"Late filtering" (Finding 7) is the counterintuitive winner.** I naturally
assumed that aggressive fact extraction — the Mem0-style "distill into one clean
fact" — had to be the right design. The ablation says the opposite.
`Fast-Memorize` destroys `Fine-Memorize` on LoCoMo (25.5 vs 2.5 EM). Raw lines
beat summaries on *all four* metrics. Why? Because aggressive extraction throws
away the connective tissue that later makes multi-hop reasoning possible. At
write-time you don't know which detail will matter in combination with something
later. So — preserve now, filter late. That's why my implementation stores raw
lines verbatim.

**Reflection is overrated for retrieval.** The M3 result — SimpleMem
`Planning+Reflect` does *worse* than `Planning-Only` — is the one I'd most like
to see replicated hard. The current agent-building spirit puts
reflection/rethink everywhere. For memory at least, the data says it adds cost
with no gain. Once the route is planned, extra deliberation weakens the
decision. I implemented planning (query expansion) and deliberately *did not*
add reflection.

## What was harder than I expected

Embeddings without a model. All the paper's systems use real embedding models. I
needed this to run on a machine with no API keys and no model downloads, so I
baked a deterministic char-n-gram hash embedding. Meaningful enough for cosine
to work on a small dataset, but far weaker than a real encoder — and that's the
biggest gap between my implementation and the paper's systems. The paper
predicts this too (O5, backbone robustness): the ordering of good vs bad designs
is fairly stable across backbones, because grounding happens before generation.

Conservative consolidation is fiddly. "Merge two memories when they're about the
same thing" sounds simple. In practice the threshold is delicate: too loose and
you collapse distinct facts; too strict and you never merge so memory grows
unbounded. Finding 9 ("conservative merge wins, delayed flush loses") is easy to
say, but the actual threshold (I went with 0.85 cosine + entity overlap) is a
real engineering knob.

Multi-versioning takes discipline. Logical invalidation (never delete, just mark
stale) is elegant, but it means every query must by default filter `valid=True`
— and *optionally* include invalid ones when the query is explicitly temporal
("where did I *used to* live?"). That dual mode is easy to get wrong.

## The cost vs quality graph

This is one of the more useful diagrams in the whole paper, worth remembering —
normalized utility vs average latency per query:

```
utility
100 │                           · Cognee (84 @ 116s)
    │                    · Zep (84 @ 155s)
 80 │             · MemoryOS (82 @ 29s)
    │
 60 │       · A-MEM (58 @ 18s) · MemTree (64 @ 16s)
    │
 40 │ · LightMem (48 @ 4s)
    │
 20 │       · MemoChat (28 @ 15s) · Mem0 (21 @ 36s)
    └─────────────────────────────────────────────► latency (log)
       1s      10s        100s       1000s
```

The Pareto front runs: **LightMem → MemTree → A-MEM → MemoryOS**, then a sharp
jump to Cognee/Zep for the last few utility points at 4-10× the cost. The
message for a builder: pick your point on that curve based on how
latency-sensitive your workload is. The highest utility isn't always worth it.

## References
- Paper: https://arxiv.org/abs/2606.24775
- Official code/benchmark: https://github.com/OpenDataBox/MemoryData
- My implementation: `implementation/`
- Breakdown: `breakdown.md`
