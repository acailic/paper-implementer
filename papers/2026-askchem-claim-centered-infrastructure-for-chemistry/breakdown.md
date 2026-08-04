# Breakdown — AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis

> **Paper:** AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis
> **Authors:** Bing Yan, Gregory Wolfe, Stefano Martiniani, Kyunghyun Cho
> **Year:** 2026 · **ArXiv:** https://arxiv.org/abs/2607.28618
> **Code (official):** https://github.com/bingyan4science/askchem · **Live:** https://askchem.org

---

## 1. Problem & Motivation

**The problem.** Chemists routinely ask questions whose answers are *distributed
across many papers*. A question like *"what electrocatalysts reduce CO₂ to CO,
and at what Faradaic efficiency?"* has an answer that is a *set* of specific
findings (catalyst + conditions + measured value), each one living in a
different paper. Today, every search tool returns a **ranked list of documents**.
The scientist — or an LLM agent — then has to open each paper, find the relevant
evidence, verify the numbers, and stitch the cross-paper answer together by
hand. This is slow, error-prone, and does not scale.

**Two failure modes of the status quo.**
- *Document retrieval* returns papers, not answers, so cross-paper synthesis is a
  manual join.
- *Bare LLMs* answer from parametric memory and **fabricate plausible-looking
  citations** (e.g. invented "Agrawal et al. 2024"), because the model does not
  actually know which DOI backs which claim.

So neither paradigm solves grounded cross-paper synthesis.

**Why it matters.** Chemistry (and science generally) is a literature-synthesis
field: progress means assembling specific, numeric, verifiable findings scattered
across thousands of papers. The unit of scientific record is the *finding*, but
the unit of retrieval is still the *paper*. This mismatch is the core pain.

**Prior approaches and their limits.**
- Keyword/paper search engines (Google Scholar, Semantic Scholar) → return docs,
  not findings; no provenance at sub-paper granularity.
- Document-level RAG over PDFs → still paper-sized chunks; citations are coarse.
- Agentic deep-research systems (Paperclip / PaperQA-family, Edison Scientific)
  → retrieve papers then read them at inference time; expensive, and the LLM can
  still misattribute which paper said what.
- LLMs without retrieval → hallucinate citations.
- None of them treat the **atomic claim with source DOI + verbatim quote** as the
  addressable unit.

---

## 2. Key Insight / Contribution

**Core idea (in my own words).** Make the **atomic, provenance-carrying claim**
— not the paper — the unit of retrieval. Each paper is decomposed into typed
claims, each carrying its source DOI, a verbatim quote (or explicit evidence
locator), and structured chemistry fields. Then three complementary structures
(a faceted taxonomy, an evidence graph, and a principle-centered living
taxonomy) are layered over the *same shared claim store*, so that search,
browse, and graph navigation all return objects keyed by one `claim_id` and
never lose provenance.

**What is genuinely new.**
1. The **claim as the retrieval primitive** with a mandatory provenance triple
   `(claim_type, source_doi, verbatim_quote | evidence_locator)` — 100% of 2.4M
   claims are source-grounded, 99.9% DOI-verified via CrossRef.
2. The **decomposition analogy**: just as SAM segments images into reusable
   masks, AskChem segments papers into reusable, composable claims.
3. A **four-channel hybrid retrieval fused by Reciprocal Rank Fusion (RRF)**,
   which works across heterogeneous score scales (FTS5, paper-rank, taxonomy,
   dense-vector) precisely because it is rank-only and parameter-free.
4. A **groundedness-first benchmark (AskChem-Bench)** that measures *whether
   citations are real and relevant* rather than raw factual accuracy, evaluated
   against CrossRef and a judge calibrated on expert labels (κ = 0.914).

---

## 3. Method

### 3.1 Overview

AskChem is an information-architecture / data-management system with five stages:

```
 Papers ──► 1. EXTRACT ──► Claims ──► 2. STRUCTURE ──► 3. STORE ──► 4. RETRIEVE ──► 5. SYNTHESIZE
 (PDFs)     (LLM + JSON       (taxonomy +     (SQLite + FTS5 +    (hybrid 4-channel     (reader LLM,
            schema)            graph)           dense vectors)       RRF fusion)           grounded)
```

A user query enters at stage 4 (Retrieve) and exits at stage 5 (Synthesize) as a
grounded, citation-dense answer. The first three stages are offline indexing.

### 3.2 Architecture (data model + structures)

**Four object types, all sharing one `claim_id`:**

```
                    ┌─────────────────────────────────────────────┐
                    │                 CLAIM (atomic)                │
                    │  claim_id  (shared key)                       │
                    │  claim_type  (reaction/method/comparison/...) │
                    │  source_doi  ──────────────┐                 │
                    │  verbatim_quote | evidence_locator            │
                    │  confidence, structured chemistry fields      │
                    └──────┬──────────────┬───────────────┬────────┘
                           │              │               │
              ┌────────────▼──┐   ┌───────▼────────┐  ┌──▼──────────┐
              │   SOURCE      │   │   TREE NODE     │  │   EDGE       │
              │ (paper meta)  │   │ (faceted path)  │  │ (typed rel)  │
              │ DOI, year,    │   │ by_reaction_type│  │ supports /   │
              │ venue, cites, │   │ by_substance_.. │  │ contradicts /│
              │ authors       │   │ by_application..│  │ extends /    │
              └───────────────┘   │ ... 2–5 levels  │  │ derives_from │
                                  └─────────────────┘  └─────────────┘
```

**The Claim invariant.** *Every* claim carries a DOI + a type + (a verbatim
quote XOR an evidence locator). The locator fallback exists for structured
full-paper claims (e.g. whole-paper property claims) that have no contiguous
quote — instead they point to an explicit location (table/figure/section).

**Three structures over the same claim store:**

1. **Stabilized faceted taxonomy** — *"what is it about?"* Corpus-induced facet
   paths under fixed top-level views: `by_reaction_type`, `by_substance_class`,
   `by_application`, `by_technique`, `by_mechanism_topic`, `claim_type`, `data`,
   `time`. Induced while digesting papers, then **stabilized** via
   (i) canonical L1 routing, (ii) synonym normalization, (iii) fuzzy clustering
   of near-duplicate subcategories into persistent L1/L2/L3 nodes
   (e.g. `coupling/cross_coupling/suzuki`). 307K populated nodes. The
   *operational* index used in search.

2. **Evidence graph** — *"how are findings related?"* Typed directed edges:
   `supports | contradicts | extends | derives_from | cites_as_evidence`, each
   with `confidence` and `evidence` text. 171,342 edges. Expert audit:
   143/146 decidable edges had correct relation type → **97.9% edge-type
   precision**. A relational layer *over* retrieval, surfaced via the
   `/neighborhood` endpoint.

3. **Exploratory living taxonomy** — *"what principle governs it?"*
   Principle-centered hierarchy (principles / theories / models / mechanisms /
   phenomena). 4,931 nodes, 1.1M claims, 361K placements. Has an **abstention
   mechanism**: 663 nodes are *open proposed branches* — if nothing fits, the
   system proposes a new host rather than force-fitting. Exploratory, not a
   validated ontology.

### 3.3 Forward pass / pipeline

**Offline indexing (stages 1–3):**

1. **Extract.** Two LLM pipelines, both JSON-object-constrained
   (`response_format={"type":"json_object"}`), both validated against the claim
   schema with retry on invalid output:
   - *Abstract extractor* (high-throughput): GPT-5-mini over `(title, abstract)`
     → 102K papers. Shallower.
   - *Deep full-text extractor*: Gemini 3.1 Pro with native-PDF input via Vertex
     AI *batch* → 44K papers. Catches claim types absent from abstracts
     (hypotheses, limitations, surprises, mechanism).
   - Output: typed claims with provenance triple + structured chemistry fields.
2. **Structure.** Induce+stabilize faceted paths per claim; run a second
   extraction pass to emit typed evidence edges; assign living-taxonomy hosts
   (with abstention).
3. **Store.** Persist in **SQLite + FTS5** (full-text over claim text + quote) +
   a **dense vector index** (Sentence-Transformers embeddings), served via
   FastAPI.

**Online query (stages 4–5):**

4. **Retrieve** — the load-bearing algorithm (see §3.4 / §4):
   - (a) Rewrite the query into 3–4 keyword subqueries.
   - (b) Run four parallel recall channels → four ranked claim lists:
     1. FTS5 (BM25-style over claim text + quote),
     2. paper-level recall (rank claims by source-paper score),
     3. taxonomy-node recall (match query to node labels → recall claims),
     4. dense-vector recall (query embedding vs claim embeddings).
   - (c) **Fuse** the four lists via Reciprocal Rank Fusion (k = 60).
   - (d) **Diversify** to ≤ 40 claims (greedy de-dup by source DOI / taxonomy
     path in the toy).
5. **Synthesize.** The ≤40 fused claims (with quotes + DOIs) go to the reader
   LLM, which may only cite DOIs present in the retrieved evidence. This
   constraint is what drives DOI existence to 100%.

**Interfaces.** Web UI, REST (`/api/search`, `/api/claims/{id}`,
`/api/claims/{id}/neighborhood`, `/api/sources/{doi}`), SDK, and an **MCP
server** (Anthropic 2024). All return the *same* claim identities.

### 3.4 Loss function

There is **no learned loss function** in the classical sense. AskChem's
"learning" is LLM-driven *extraction* (zero-loss, schema-constrained JSON
decoding) and the ranking is the **parameter-free RRF combiner**. The only
trained components are the off-the-shelf embedding model (for dense recall) and
the FTS5 BM25 statistics. So the "objective" being optimized at query time is
the RRF fused rank, not a gradient.

---

## 4. Math

### 4.1 Reciprocal Rank Fusion (RRF) — Cormack, Clarke & Buettcher, SIGIR 2009

**The equation (my notation):**

```
              N        1
RRF(d)  =    Σ   ────────────
             i=1  k + r_i(d)
```

**Symbols.**
- `d` — a candidate claim (document).
- `N` — number of recall channels (here `N = 4`: FTS5, paper, taxonomy, vector).
- `i` — channel index.
- `r_i(d)` — the 1-indexed **rank** of claim `d` in channel `i`'s ranked list.
  If claim `d` is absent from channel `i`, then `r_i(d) = ∞`.
- `k` — a smoothing constant; AskChem uses the standard **`k = 60`**.

**In plain English.** For each claim, sum up one term per channel: `1/(k+rank)`.
A claim ranked #1 in a channel contributes `1/61`; ranked #2 → `1/62`; absent →
`1/(60+∞) = 0`. The claim with the highest total summed reciprocal rank across
all four channels wins.

**Why RRF here (key design rationale).** The four channels have
**incommensurable score scales**: FTS5 returns a BM25 score, paper recall returns
a citation/authority score, taxonomy recall returns a match-count, vector recall
returns a cosine in [0,1]. A learned rank-fusion model would need those scales
to be comparable (or learned), requiring labeled data. **RRF uses rank only**,
so it is parameter-free, training-free, and robust to heterogeneous scales —
exactly the property a production multi-signal retrieval system wants.

### 4.2 Dense-vector recall (channel 4)

```
score_vec(d) = cos( e(q), e(d) )
```
- `e(·)` — embedding function (Sentence-Transformers in production; TF-IDF
  cosine is fine for a toy re-implementation).
- `q` — query text, `d` — claim text.
- Claims ranked by descending `score_vec`.

### 4.3 Diversification (to ≤ 40 claims)

The paper does not name the algorithm. The natural greedy formulation:

```
D ← []
for d in claims_sorted_by_RRF_desc:
    if source_doi(d) not in {source_doi(x) : x ∈ D}
       or taxonomy_path(d) not seen too many times:
        D.append(d)
    if |D| == 40: break
```
i.e. a greedy MMR-ish de-dup that caps repetition of the same source paper /
same taxonomy path, trading redundancy for coverage. (This is the obvious
baseline; AskChem's exact variant is unspecified.)

### 4.4 Metrics (AskChem-Bench)

| Metric | Formula / definition |
|---|---|
| **DOI existence (%)** | (# cited DOIs that resolve in CrossRef) / (# cited DOIs) |
| **Citation density** | distinct *verified* DOIs per answer |
| **Grounded specificity** | quantitative tokens sharing a sentence with a citation marker |
| **Recent high-impact (%)** | cited papers from last 5 yrs with ≥ 50 citations |
| **Paper relevance (0–3)** | judge: 3 = direct, 2 = on-topic, 1 = loose, 0 = irrelevant |
| **On-topic ≥ 2 (%)** | (# cited papers scored ≥ 2) / (# cited papers) |

DOI existence and citation density are the headline groundedness metrics; they
are computed *objectively* via CrossRef (no judge needed). Relevance is judge
scored (Gemini 3.1 Pro, calibrated on 100 expert labels → 93% agreement,
κ = 0.914).

---

## 5. Training

AskChem has **no model-training loop** in the paper. "Training" maps to:

- **Corpus.** 147K chemistry papers (102K abstract-only + 44K full-text),
  producing **2.4M claims**.
- **Extraction models (frozen, off-the-shelf):**
  - GPT-5-mini (abstract slice),
  - Gemini 3.1 Pro (full-text slice, Vertex AI batch),
  - legacy GPT-4o / GPT-4o-mini slice.
  - All with JSON-object-constrained decoding at provider-default temperature.
- **Embedding model:** Sentence-Transformers (for dense recall) — frozen.
- **Index build:** SQLite + FTS5 tokenize + vector index populate; taxonomy
  induce+stabilize; evidence-edge second extraction pass.
- **"Hyperparameters":** RRF constant `k = 60`; top-`K` per channel and the
  diversification cap (≤ 40 fused claims) are the main tunables.
- **Compute budget.** Not reported in detail; the dominant cost is the offline
  LLM extraction over 147K papers (full-text Gemini 3.1 Pro batch is the
  expensive part). Online retrieval is cheap (SQLite + vector lookups).
- **Validation gates (engineering substance):** (1) response parses against
  schema; (2) required provenance present; (3) numeric/chemical fields checked;
  (4) facet paths routed to canonical L1 before normalization; (5) edges keep
  confidence + provenance; (6) living-taxonomy placements may abstain. These
  guarantee **traceability, not semantic correctness**.

> **For the re-implementation:** there is no `train.py` in the ML sense. The
> runnable entry point will be a `train.py`/`run.py` that *builds the claim
> store* (parses a small synthetic corpus into `Claim` objects, builds the FTS5 +
> TF-IDF indexes + faceted paths), then answers queries via RRF-fused retrieval,
> printing ranked claims with provenance. A mini AskChem-Bench will compute
> DOI-existence and citation-density on the toy corpus.

---

## 6. Results & Ablations

**Headline (Table 1, AskChem-Bench, 30 questions, GPT-5.5 reader):**

| Setting | DOI existence % | Citation density | Paper relevance (0–3) | On-topic ≥2 % |
|---|---|---|---|---|
| LLM only | 88.3 | 9.6 | 1.66 | 65.8 |
| **+AskChem** | **100** | **18.1** | **2.15** | 86.6 |
| +Paperclip | 100 | 7.5 | 1.72 | 57.8 |
| Edison Scientific | 99.1 | 10.7 | 2.07 | **89.7** |
| NotebookLM | 93.7 | 7.9 | 1.84 | 78.9 |

**What the numbers say:**
- **+AskChem** drives DOI existence from 88.3% → **100%** and nearly *doubles*
  citation density (9.6 → 18.1) vs the bare LLM. This is the core win: every
  citation is real and there are ~2× more of them.
- **+AskChem vs +Paperclip** is the cleanest controlled comparison (same
  rewriter + synthesizer, only the retrieval backend differs: claims vs papers).
  Claims win on relevance (2.15 vs 1.72) and on-topic rate (86.6 vs 57.8) —
  strong evidence that the **claim granularity** (not the LLM) is what helps.
- **Edison Scientific** (a deeper agentic system) edges AskChem on on-topic
  (89.7) and grounded specificity (29.2), so AskChem is not the deepest reader
  — its profile is *claim-level, open-data, interactive, agent-usable, zero DOI
  hallucination*.

**Quality of the structures (ablations / audits):**
- **Evidence graph precision:** expert audit of 148 sampled edges, 2 undecidable
  excluded → 143/146 correct relation type = **97.9% edge-type precision**.
- **Relevance judge calibration:** 93% agreement with expert labels,
  κ = 0.914 — the subjective metric is reliable.
- **Living taxonomy abstention:** 663 of 4,931 nodes are *open proposed
  branches* — the system prefers to abstain than force-fit, which is itself a
  quality signal.

**Why it works (mechanism).** Two things, both visible in the controlled
comparison: (1) claim granularity aligns retrieval with the *finding* a user
actually wants, so the reader LLM gets precise evidence instead of whole papers;
(2) RRF robustly combines four heterogeneous signals without needing
score-scale calibration, so no single weak channel can drag the result down.

---

## 7. Limitations

- **Corpus coverage** — only a fraction of chemistry is indexed; recall is
  bounded by what was ingested.
- **Abstract vs full-text depth** — the 102K abstract-only slice is shallower
  than the 44K full-text slice; claim types like limitations/surprises are
  under-represented there.
- **LLM-extracted content can be semantically wrong** — provenance enables
  *verification* (you can click through to the quote) but does *not* guarantee
  the claim is semantically correct. The 97.9% edge precision and 100%
  grounding are about *traceability*, not truth.
- **Bench measures groundedness, not factual accuracy** — 30 questions, judged
  on citation reality/relevance. A system could be perfectly grounded yet still
  miss the right answer if the corpus lacks it.
- **Taxonomy stabilization is heuristic** — string-based normalization can merge
  distinct categories or leave near-duplicates; the retrieval gain attributable
  to the taxonomy channel is *not* isolated (no channel-level ablation reported).
- **Diversification algorithm unspecified** — the paper says "diversifies" to
  ≤40 claims without naming the method.
- **Cost** — full-text Gemini 3.1 Pro batch extraction over 147K papers is the
  expensive offline step; not quantified.

---

## 8. Open Questions / Ideas

- **Channel-level ablation.** Which of the four RRF channels actually carries
  the gain? A clean leave-one-out (FTS5 / paper / taxonomy / vector) on
  AskChem-Bench would quantify each channel's marginal value — the paper does
  not report this.
- **Taxonomy as a recall signal in isolation.** The faceted taxonomy is the most
  novel structure; how much does *taxonomy-only* recall add vs FTS5+vector
  alone?
- **Better diversification.** Replace the greedy DOI/path de-dup with an explicit
  MMR (maximal marginal relevance) objective over claim embeddings — does
  coverage/precision trade off better?
- **Claim-level faithfulness metric.** Beyond DOI existence (is the citation
  real?) add a *quote-entailment* check (does the quote actually support the
  claim as the extractor framed it?).
- **Living taxonomy as an active-learning loop.** The 663 open proposed branches
  are natural "ask an expert" candidates — could curating them back into the
  taxonomy measurably improve recall?
- **Generalization beyond chemistry.** The claim schema is largely
  domain-agnostic (type + DOI + quote + structured fields). How well does it
  transfer to materials science, biomedicine, or social-science literature where
  the unit of finding is also sub-paper?
- **For the toy implementation:** the most faithful minimal re-implementation is
  the **4-channel hybrid retrieval + RRF fusion** over a hand-authored
  synthetic claim corpus — it isolates the load-bearing algorithm and needs no
  frontier LLM at run time.
