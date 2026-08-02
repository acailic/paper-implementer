# Notes — AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis

> **Paper:** AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis
> **Authors:** Bing Yan, Gregory Wolfe, Stefano Martiniani, Kyunghyun Cho
> **Year:** 2026 · **ArXiv:** 2607.28618v1 [cs.CL] (30 Jul 2026)
> **Code:** https://github.com/bingyan4science/askchem · **Live:** https://askchem.org

---

## First impressions (pass 1)

This is a **systems / information-infrastructure** paper, not a learning-algorithm
paper. It is much closer to a RAG architecture / data-management paper than to a
neural-net method paper. The "model" being built is an **information architecture**
plus a **retrieval pipeline**, deployed at corpus scale (2.4M claims / 147K papers).

### The problem (in my own words)

Chemists routinely ask questions whose answers are *distributed across many papers*,
e.g. *"what electrocatalysts reduce CO₂ to CO, and at what Faradaic efficiency?"*
The answer is a set of specific findings (catalyst + conditions + measured FE),
each living in a different paper. But every existing search tool returns a **ranked
list of documents**, forcing the scientist (or an LLM agent) to open each paper,
find the relevant evidence, verify the numbers, and stitch the answer together by
hand. Worse, if you let an LLM answer from parametric memory, it **fabricates
plausible-looking citations** (Agrawal et al. 2024). So neither document retrieval
nor naked LLMs solve cross-paper synthesis.

### The core idea (one sentence)

**Make the atomic, provenance-carrying *claim* — not the paper — the unit of
retrieval**, then layer three complementary structures (faceted taxonomy, evidence
graph, principle-centered living taxonomy) over the same shared claim store.

The analogy the authors lean on: just as SAM (Segment Anything) decomposed images
into reusable masks, AskChem uses LLMs to **segment papers into claims**.

### What a "Claim" is

An atomic, typed scientific assertion extracted from a paper, grounded by:
- a **source DOI**,
- a **verbatim quote** (or an explicit evidence locator for structured full-paper
  claims that lack a contiguous quote),
- a **claim_type** (reaction / method / comparison / mechanism / computation /
  limitation / experiments / surprises / scope — 9 types),
- structured chemistry fields (reactants, products, conditions, measurements,
  materials),
- an extraction **confidence score**.

100% of the 2.4M claims are source-grounded (carry type + DOI + quote/locator).

### The three structures over the claim store

1. **Stabilized faceted taxonomy** — "what is it about?" Organizes claims along
   corpus-induced facets: reaction type, substance class, application, technique,
   mechanism topic, claim type, data, time. Category paths are *induced* while
   digesting papers, then **stabilized** via canonical L1 routing, synonym
   normalization, and fuzzy clustering of near-duplicates into persistent L1/L2/L3
   paths (e.g. `coupling/cross_coupling/suzuki`). 307K populated nodes. This is an
   *operational* index, not a scientific ontology.

2. **Evidence graph** — "how are findings related?" Typed directed edges between
   claims: `supports`, `contradicts`, `extends`, `derives_from`,
   `cites_as_evidence`. 171,342 edges. Expert audit: 143/146 decidable edges had
   correct relation type → **97.9% edge-type precision**. Used as a relational
   layer *over* retrieval, not a replacement for search.

3. **Exploratory living taxonomy** — "what principle governs it?" Principle-centered
   hierarchy (principles, theories, models, mechanisms, phenomena) that situates
   papers under scientific ideas. 4,931 nodes, 1.1M claims, 361K placements. Has an
   **abstention mechanism** (proposes new branches when nothing fits). Treated as
   exploratory, not validated ontology.

### Retrieval pipeline (the part I care about most for re-implementation)

- **Storage:** SQLite + FTS5 (full-text search) + a dense vector index, served via
  FastAPI.
- **Hybrid search `/search`** fuses FOUR signals via **reciprocal rank fusion**
  (RRF, Cormack et al. 2009):
  1. FTS5 claim-text retrieval,
  2. paper-level recall,
  3. taxonomy-node recall,
  4. dense-vector recall.
- Each query is rewritten into 3–4 keyword subqueries, fanned out, then merged
  + diversified to ≤40 claims before grounded synthesis.

### Extraction (two pipelines)

- **Abstract extractor (high-throughput):** GPT-5-mini over title+abstract. Covers
  the 102K-paper abstract-only slice.
- **Deep full-text extractor:** Gemini 3.1 Pro with native-PDF input via Vertex AI
  batch. Catches claim types absent from abstracts (hypotheses, limitations,
  surprises). Covers 44K full papers.
- All calls use **JSON-object-constrained decoding**
  (`response_format={"type":"json_object"}`), validated against the claim schema,
  with retry on invalid/invalid-schema output.

### Evaluation — AskChem-Bench

30 cross-paper chemistry questions in 3 task types:
- **CA** condition aggregation,
- **TC** temporal tracking / evolution,
- **CS** conflict / contradiction surfacing.

Five settings compared (GPT-5.5 reader throughout):

| Setting | DOI existence % | Citation density | Paper relevance 0–3 | On-topic ≥2 % |
|---|---|---|---|---|
| LLM only | 88.3 | 9.6 | 1.66 | 65.8 |
| +AskChem | **100** | **18.1** | 2.15 | 86.6 |
| +Paperclip | 100 | 7.5 | 1.72 | 57.8 |
| Edison Scientific | 99.1 | 10.7 | 2.07 | **89.7** |
| NotebookLM | 93.7 | 7.9 | 1.84 | 78.9 |

Headline: grounding in AskChem → **100% resolvable DOIs** (vs 88.3% bare LLM) and
highest **citation density** (18.1 verified DOIs/answer). DOIs verified via
CrossRef. Relevance judged by Gemini 3.1 Pro calibrated on 100 expert labels
(93% agreement, κ = 0.914).

### Interfaces

Web UI, REST (`/api/search`, `/api/claims/{id}`,
`/api/claims/{id}/neighborhood`, `/api/sources/{doi}`), SDK, and **MCP server**
(Anthropic 2024) — all return the *same* claim identities.

### Limitations (authors' own)

- Corpus covers only a fraction of chemistry.
- Abstract extraction is shallower than full-text.
- LLM-generated claims/relations/taxonomy placements *can be wrong*; provenance
  enables verification but doesn't guarantee semantic correctness.
- Bench measures groundedness on 30 questions, not full factual accuracy.
- String-based taxonomy normalization can merge distinct categories or retain
  near-duplicates; retrieval gain from taxonomy not isolated.

---

## Terms / concepts to nail down in pass 2

- **Reciprocal Rank Fusion (RRF)** — the actual formula and why it beats learned
  rank-fusion here. This is the single most re-implementable algorithmic piece.
- **Faceted taxonomy induction vs. stabilization** — how exactly are paths induced
  then stabilized (canonical L1 routing, synonym normalization, fuzzy clustering)?
  The paper is light on detail; may need the repo.
- **Evidence-graph extraction prompt** — how does the second relation-extraction
  pass decide supports/contradicts/extends? confidence scoring.
- **Diversification** of merged evidence to ≤40 claims — which algorithm?
- **Living taxonomy abstention** mechanism details (Appendix B).
- **Claim schema** in full — fields, required vs optional, validation gates.

## Implementation plan (early thought)

This is highly re-implementable as a **toy claim-centered retrieval system**,
without needing any frontier LLM at run time:
- A `Claim` dataclass + JSON schema with provenance fields.
- A tiny hand-authored corpus of ~20–40 chemistry claims (I can fabricate
  plausible grounded claims with DOIs/quotes — clearly marked as synthetic).
- SQLite + FTS5 claim store; a small dense-vector recall over Sentence-Transformers
  embeddings (or even TF-IDF cosine to stay dependency-light).
- Faceted taxonomy paths per claim.
- A few typed evidence edges.
- **Hybrid search with reciprocal rank fusion** across (FTS5, taxonomy, vector,
  paper) signals — this is the load-bearing algorithm and very writable from
  scratch.
- A mini AskChem-Bench with ~5 questions + a DOI-existence / citation-density
  metric computed against the synthetic corpus.

`python train.py` semantics: there's no "training" in the ML sense here; the
runnable entry point will be a `run.py` / `search.py` that builds the store and
answers queries, printing RRF-fused ranked claims + provenance. I'll keep a
`train.py` name for workflow compatibility but it will *index* the corpus.
