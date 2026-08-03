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

*(worked through below in the second pass — kept for traceability)*

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

---

## Second pass (deep) — method, math, schema

The second read confirms the structure and lets me pin down the parts that
were vague after pass 1. I'll organise this as: (A) data model, (B) the two
extraction pipelines, (C) the three structures, (D) the retrieval algorithm
with the actual math, (E) the benchmark and its exact metrics.

### A. Data model (the four objects, all sharing `claim_id`)

Everything is anchored to one shared `claim_id`. Four object types sit on it:

1. **Claim** — atomic typed assertion. Required provenance triple:
   `(claim_type, source_doi, verbatim_quote)` for abstract/quote claims, or
   `(claim_type, source_doi, evidence_locator)` for structured full-paper
   claims that have no contiguous quote. So the invariant is: **every claim
   carries a quote OR a locator, plus a DOI, plus a type.** Optional structured
   chemistry fields: `reaction_type`, `reactants[]` (name + role), `products[]`,
   `outcomes{selectivity, other}`, `measurements`, `materials`. Plus an
   extraction `confidence`. From the paper's worked example (claim_id
   `7c92fcacd8cb64d4`):
   ```
   claim_type: "reaction"
   source_doi: "10.1002/anie.201914977"
   reaction_type: "electrocatalytic CO2 reduction (to CO)"
   reactants: [{name:"CO2", role:"substrate"},
               {name:"Ni SA-N2-C", role:"catalyst"}]
   products: [{name:"CO"}]
   outcomes: {selectivity:"CO Faradaic efficiency 98%",
              other:"turnover frequency 1622 h-1"}
   verbatim_quote: "the Ni SA-N2-C catalyst, with the lowest N coordination
                    number, achieves very high CO Faradaic efficiency (98%)
                    and turnover frequency (1622 h-1)"
   view_paths: {by_reaction_type:["electrocatalysis","co2_reduction"], ...}
   ```
   Claim types (9 + the bulk `property`): method (331K), comparison (233K),
   mechanism (222K), computation (222K), reaction (122K), limitation (56K),
   experiments (51K), surprises (51K), scope (49K). Plus 964K `property`
   claims — so really ~10 categories dominated by property/measurement claims.

2. **Source** — paper-level metadata: DOI, venue, year, citation count,
   authors disambiguated via OpenAlex. One source → many claims.

3. **TreeNode** — places a claim under one or more faceted taxonomy paths
   (2–5 segments each, e.g. `by_reaction_type/coupling/cross_coupling/suzuki`).
   A claim can sit in multiple views simultaneously.

4. **Edge** — typed directed relation between two claims:
   `supports | contradicts | extends | derives_from | cites_as_evidence`,
   each with `confidence` and `evidence` text. Example from the paper:
   ```
   edge_type: "supports"
   to_doi: "10.1021/acs.joc.9b01692"
   confidence: "high"
   evidence: "The observed selective coupling at the chloride position using
              SIPr directly provides evidence for the overarching claim of
              high ligand-controlled selectivity."
   ```

Key design property: because search, browse, and graph all return objects
keyed by the same `claim_id`, a user moving from a search hit → its taxonomy
path → its evidence neighbourhood never loses provenance.

### B. Extraction (two pipelines, both LLM + JSON-schema-constrained)

- **Abstract extractor (high throughput):** GPT-5-mini over `(title, abstract)`.
  Covers the 102K-paper abstract-only slice. Cheaper, shallower.
- **Deep full-text extractor:** Gemini 3.1 Pro with native-PDF input via Vertex
  AI **batch**. Covers 44K full papers. Catches claim types absent from
  abstracts: hypotheses, limitations, surprises, mechanism claims.
- A small legacy slice predates this (GPT-4o / GPT-4o-mini).
- **All calls** use JSON-object-constrained decoding
  (`response_format={"type":"json_object"}`), at provider-default temperature,
  with automatic retry on invalid JSON or schema-invalid output.
- **Validation gates (Appendix B):** (1) response must parse against the claim
  schema; (2) required provenance fields present; (3) numeric/chemical fields
  checked when present; (4) faceted paths routed to canonical L1 before
  lower-level normalisation; (5) evidence edges keep extractor confidence +
  provenance; (6) Living-Taxonomy placements may *abstain*.

These checks guarantee **traceability** (every claim links back to a DOI +
quote/locator), NOT semantic correctness. 100% of the 2.4M claims are
source-grounded; 99.9% are DOI-verified via CrossRef. That is the paper's
honest framing and it's important.

### C. The three structures (lenses over the same claim store)

1. **Stabilized faceted taxonomy** (the operational one, used in search).
   - Paths *induced* while digesting papers (LLM emits category paths per claim),
     then **stabilised** via:
     (i) **canonical L1 routing** — each raw path is routed to one of the fixed
         top-level views (`by_reaction_type`, `by_substance_class`,
         `by_application`, `by_technique`, `by_mechanism_topic`, `claim_type`,
         `data`, `time`);
     (ii) **synonym normalisation** — string-level merging of synonyms;
     (iii) **fuzzy clustering** of near-duplicate subcategories into persistent
          L1/L2/L3 nodes.
   - Result: 307K populated nodes. Each claim gets a 2–5 segment path per
     populated view. This is corpus-derived but persistent — stable enough for
     production retrieval and browsing.
   - The paper is deliberately vague on the exact clustering algorithm; I'll
     use normalised-string hashing + a simple synonym map in the toy impl.

2. **Evidence graph** (relational layer over retrieval).
   - Second extraction pass emits typed directed edges with confidence.
   - 171,342 edges. Expert audit: 148 sampled, 2 undecidable excluded, 143/146
     correct type → **97.9% edge-type precision**.
   - Surfaced via `/api/claims/{id}/neighborhood` (inbound + outbound) and as a
     graph induced over top search hits in the UI. Used to *navigate*, not to
     replace search.

3. **Exploratory Living Taxonomy** (principle-centered, exploratory).
   - Asks "which scientific idea *governs* this paper's contribution?" rather
     than "what is it about?". Organises paper-grounded leaves under
     principles / theories / models / mechanisms / phenomena.
   - 4,931 nodes, 1.1M claims, 361K paper placements (663 of those nodes are
     *open proposed branches* = the abstention mechanism: if no host fits, the
     LLM proposes a new branch rather than force-fitting).
   - Constructed by an LLM reading each paper's claims and naming the best host.
   - Explicitly treated as exploratory, NOT a validated ontology.

### D. Retrieval algorithm — the load-bearing math (RRF)

This is the single most re-implementable piece, so I'll write it out fully.

**Step 1 — query rewrite.** Each user question is rewritten into 3–4 keyword
subqueries (LLM, but a deterministic keyword splitter works for a toy).

**Step 2 — four parallel recall channels**, each producing a *ranked list* of
claims (by that channel's own score):
  1. **FTS5** claim-text retrieval (BM25-ish over claim text + quote).
  2. **Paper-level recall** — rank claims by the score of their source paper
     (so all claims from a top paper get a paper-rank boost).
  3. **Taxonomy-node recall** — match the query against taxonomy-node labels,
     then recall claims placed under matched nodes.
  4. **Dense-vector recall** — cosine similarity between query embedding and
     claim embeddings (Sentence-Transformers in production; TF-IDF cosine is
     fine for a toy).

**Step 3 — Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).**
Merge the four ranked lists into one. For each claim `d`, let `r_i(d)` be its
rank in channel `i` (1-indexed; absent = ∞). The fused score is:

```
              1
RRF(d) = Σ  ──────────
          i   k + r_i(d)
```

with the standard constant **k = 60**. RRF is parameter-free and rank-only —
it does NOT need the raw scores to be comparable across channels (which is
exactly why it works across heterogeneous recall signals: FTS BM25, vector
cosine, taxonomy match-count, paper citation). This is the key reason the
authors chose it over a learned rank-fusion model: the four channels have
incommensurable score scales, so a rank-based combiner is the robust choice
and needs no training data. Claims absent from a channel contribute 0 from
that channel (rank ∞ → 1/(k+∞) = 0).

**Step 4 — diversify** the merged evidence to ≤ 40 claims (the paper says
"diversifies" without naming the algorithm; I'll use a simple greedy
MMR-ish de-dup by source DOI / taxonomy path in the toy).

**Step 5 — grounded synthesis.** The ≤40 fused claims (with quotes + DOIs)
are handed to the reader LLM, which may only cite DOIs that appear in the
retrieved evidence → this is what drives DOI existence to 100%.

### E. Benchmark (AskChem-Bench) and exact metrics

- **30 questions**, 10 each in three cross-paper task types:
  - **CA** condition aggregation (e.g. "electrocatalysts for CO2→CO + FE"),
  - **TC** temporal tracking / evolution,
  - **CS** conflict / contradiction surfacing.
  - Topics span C–N coupling, Suzuki–Miyaura, CO2 reduction, water splitting,
    perovskite degradation, MOF stability, etc.
- **Five settings**, all using a GPT-5.5 reader: LLM-only, +AskChem,
  +Paperclip, Edison Scientific (PaperQA-family agent), NotebookLM Deep
  Research. AskChem and Paperclip share the *same* rewriter + synthesizer;
  they differ only in retrieval backend (claims vs papers).
- **Six metrics** (exact definitions from Appendix A):

  | Metric | Definition |
  |---|---|
  | DOI existence (%) | Fraction of cited DOIs that resolve in CrossRef |
  | Citation density | Distinct verified DOIs per answer |
  | Grounded specificity | Quantitative tokens sharing a sentence with a citation marker |
  | Recent high-impact (%) | Cited papers from last 5 yrs with ≥ 50 citations |
  | Paper relevance (0–3) | Judge score: 3 direct, 2 on-topic, 1 loose, 0 irrelevant |
  | On-topic ≥ 2 (%) | Fraction of cited papers scored ≥ 2 |

- **Relevance judge:** Gemini 3.1 Pro, calibrated on 100 expert labels →
  93% agreement, κ = 0.914.
- **DOI verification:** every extracted DOI checked through CrossRef.

- **Headline numbers (Table 1):** +AskChem → 100% DOI existence (vs 88.3%
  LLM-only), 18.1 citation density (best), 2.15 relevance (best mean),
  86.6% on-topic. Edison Scientific wins on-topic (89.7%) and grounded
  specificity (29.2) because it's a deeper agentic system, but AskChem wins
  on citation density and ties DOI existence. The honest takeaway: AskChem's
  profile is *claim-level, open-data, interactive, agent-usable, zero DOI
  hallucination* — not raw quantitative depth.

### What changed in my understanding after pass 2

- The **invariant** is clearer: it's not "claims have quotes", it's "every
  claim has a DOI + a type + (quote XOR locator)". The locator fallback for
  structured full-paper claims matters and I'd missed it in pass 1.
- **RRF with k=60** is the concrete algorithm; I can implement it in ~10 lines.
  The four-channel fusion is the whole "method" from a re-implementation view.
- The taxonomy is **induced-then-stabilised**, not predefined — canonical L1
  routing + synonym normalisation + fuzzy clustering. Stabilisation is what
  makes it usable as a recall signal, not just a browse tree.
- The benchmark is **about groundedness, not factual accuracy** — the authors
  are explicit about this limitation. 30 questions, judge-calibrated.
- Extraction is two-tier (abstract GPT-5-mini / full-text Gemini 3.1 Pro) with
  JSON-constrained decoding + schema validation + retry. The *validation gates*
  are the engineering substance; the LLM choice is secondary.
- **Living Taxonomy abstention** = 663 proposed (open) branches out of 4,931
  nodes — the system would rather propose a new host than force-fit.

### Still unclear (will check repo during coding if needed)

- Exact fuzzy-clustering metric for taxonomy stabilisation (paper omits it).
- Diversification algorithm to ≤40 claims (unnamed — I'll use greedy DOI/path
  de-dup, which is the obvious baseline).
- Query-rewriter prompt details (released with source; for the toy I'll use a
  deterministic keyword splitter).
