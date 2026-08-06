# Writeup — AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis

> **Paper:** AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis
> **Authors:** Bing Yan, Gregory Wolfe, Stefano Martiniani, Kyunghyun Cho
> **Year:** 2026 · **ArXiv:** https://arxiv.org/abs/2607.28618

My own explanation, written after reading and re-implementing the paper. This is
synthesis, not an abstract restatement.

## The one-paragraph version

AskChem argues that the *unit of retrieval in scientific search is wrong*: we
keep returning whole papers when what a researcher actually needs is the atomic
**finding** — a single typed assertion with its source DOI and a verbatim quote
attached. So they decompose 147K chemistry papers into 2.4M such provenance-carrying
claims, build three complementary structures (a faceted taxonomy, an evidence
graph, a principle-centered "living" taxonomy) over the *same* shared claim store,
and serve them through a hybrid retrieval pipeline that fuses four recall signals
with rank-only Reciprocal Rank Fusion. The payoff, measured on a groundedness
benchmark, is 100% resolvable citations and roughly double the citation density of
a bare LLM — every cited DOI is real because the synthesizer is *constrained* to
only cite DOIs that appear in the retrieved evidence.

## The problem

Chemistry is a literature-synthesis field. A typical research question — *"which
electrocatalysts reduce CO₂ to CO, and at what Faradaic efficiency?"* — is not
answered by one paper; it's answered by a *set* of specific findings scattered
across dozens of papers, each contributing one (catalyst + conditions + measured
value) tuple. But every search tool you have returns a **ranked list of documents**.
So the scientist — or an LLM agent — has to open each paper, hunt for the relevant
sentence, verify the number, and stitch the cross-paper answer together by hand.
That's slow and error-prone.

The alternative isn't better. If you just let an LLM answer from its parametric
memory, it **fabricates plausible-looking citations** ("Agrawal et al., 2024")
because it has no real notion of which DOI backs which claim. So neither document
retrieval nor a naked LLM solves grounded cross-paper synthesis.

The core mismatch AskChem identifies: **the unit of scientific record is the
finding, but the unit of retrieval is still the paper.**

## The idea

Make the **atomic, provenance-carrying claim** the retrieval primitive. Every
claim carries a mandatory triple: `(claim_type, source_doi, verbatim_quote)` —
or, for structured full-paper claims that have no contiguous quote, an explicit
evidence locator pointing at a table/figure/section. The authors lean on a nice
analogy: just as SAM (Segment Anything) decomposes images into reusable masks,
AskChem uses LLMs to **segment papers into reusable, composable claims**.

Then three structures are layered over the *same* claim store, all keyed by one
`claim_id`:

1. A **stabilized faceted taxonomy** — "what is it about?" Induced from the corpus
   along fixed views (by_reaction_type, by_substance_class, by_application, ...),
   then stabilized via canonical L1 routing + synonym normalization + fuzzy
   clustering into persistent paths. The *operational* index used in search.
2. An **evidence graph** — "how are findings related?" Typed directed edges:
   supports / contradicts / extends / derives_from. A relational layer *over*
   retrieval, surfaced as a neighborhood around any claim.
3. An **exploratory living taxonomy** — "what principle governs it?" A
   principle-centered hierarchy (principles / theories / models / mechanisms) with
   an **abstention mechanism**: if nothing fits, the system *proposes a new branch*
   rather than force-fitting. Explicitly exploratory, not a validated ontology.

Because search, browse, and graph navigation all return objects keyed by the same
`claim_id`, you never lose provenance as you move between them.

## How it works (the intuition)

**Indexing (offline).** Two LLM extraction pipelines, both with JSON-object-
constrained decoding and schema validation + retry:
- an *abstract extractor* (GPT-5-mini, 102K papers) — high-throughput, shallow;
- a *deep full-text extractor* (Gemini 3.1 Pro via Vertex AI batch, 44K papers) —
  catches claim types absent from abstracts (limitations, surprises, mechanism).

Claims then get facet paths, evidence edges, and living-taxonomy hosts assigned.

**Retrieval (online) — the load-bearing algorithm.** This is the single most
re-implementable piece, and it's where I focused:

1. Rewrite the query into 3–4 keyword subqueries.
2. Run **four parallel recall channels**, each producing a ranked claim list:
   FTS5 (BM25-ish over claim text + quote), paper-level recall (rank by source-paper
   authority), taxonomy-node recall (match query to facet labels → recall claims),
   and dense-vector recall (query embedding vs claim embeddings).
3. **Fuse** the four lists with Reciprocal Rank Fusion (RRF, `k=60`).
4. **Diversify** to ≤ 40 claims.
5. **Synthesize** with a reader LLM that may *only* cite DOIs present in the
   retrieved evidence — this constraint is exactly what drives DOI existence to 100%.

The key intuition for *why RRF*: the four channels return incommensurable raw
scores (a BM25 score, a citation count, a match-count, a cosine in [0,1]). A
learned rank-fusion model would need those scales to be comparable, which needs
labels. RRF uses **rank only**, so it's parameter-free, training-free, and robust
to heterogeneous scales — precisely the property a production multi-signal system
wants. The math is just:

```
RRF(d) = Σ_i  1 / (k + r_i(d))      (absent channel → 1/(k+∞) = 0)
```

## What I learned by implementing it

(The things that only became clear once I wrote the code.)

- **RRF is almost embarrassingly simple.** It's ~10 lines of Python. The real work
  is building the four recall channels so each produces a *ranked list*; the fusion
  is trivial. That reframes the paper: the intellectual content is the *information
  architecture* (claim + three structures), not the combiner.
- **The claim invariant is "DOI + type + (quote XOR locator)", not "claims have
  quotes".** I'd glossed over the locator fallback in pass 1. It matters: structured
  full-paper property claims genuinely have no contiguous quote — they point at a
  table/figure/section instead. Forcing a quote would produce garbage.
- **Validation gates are the engineering substance; the LLM choice is secondary.**
  The paper's honest framing is that the JSON-schema validation + provenance checks +
  retry guarantee **traceability**, not semantic correctness. 100% grounding means
  you can *click through to the quote* — it does **not** mean the claim is true.
  That distinction is the whole epistemological position of the system.
- **The benchmark measures groundedness, not factual accuracy.** DOI existence is
  computed objectively via CrossRef (no judge needed); relevance is judge-scored
  with κ = 0.914. A perfectly-grounded system can still miss the right answer if the
  corpus lacks it.
- **Channel ablation is illuminating.** In my toy, lexical channels (FTS, vector)
  surface the gold claim fastest (mean 1st-gold rank ≈ 1.0–1.4), while paper-only
  and taxonomy-only lag (2.6). So authority and taxonomy *broaden recall*; the
  lexical channels drive *top-rank precision*. That's exactly why fusing all four
  with RRF is robust — the weak channels can't drag the result down, but they
  contribute when the lexical channels miss.

## What surprised me / was harder than expected

- **There's no model-training loop at all.** This is an information-architecture /
  data-management paper, not a learning-algorithm paper. The only "trained"
  components are a frozen off-the-shelf embedding model and FTS5 BM25 statistics.
  So `train.py` semantics had to become "build the index and answer queries" —
  there's nothing to gradient-descend.
- **The controlled comparison is the cleanest evidence.** AskChem vs Paperclip share
  the *same* rewriter and synthesizer and differ *only* in retrieval backend (claims
  vs papers). Claims win on relevance (2.15 vs 1.72) and on-topic rate (86.6 vs 57.8).
  That isolates the gain to **claim granularity**, not to the LLM — which is a much
  stronger claim than "our system is good."
- **The living taxonomy's abstention is itself a quality signal.** 663 of 4,931
  nodes are *open proposed branches*. Preferring "I'd propose a new host" over
  force-fitting is a small but meaningful epistemic discipline; I hadn't expected
  that to be highlighted as a feature rather than a bug.
- **Diversification is unspecified.** The paper says "diversifies to ≤40 claims"
  without naming the method. I used a greedy DOI/path cap (the obvious baseline); a
  proper MMR over claim embeddings would be the natural next experiment.
- **No per-channel ablation is reported in the paper.** My toy's leave-one-out is
  genuinely informative and the paper omits it — a clear open question.

## References
- Paper: https://arxiv.org/abs/2607.28618
- Official code: https://github.com/bingyan4science/askchem · Live: https://askchem.org
- My implementation: `implementation/` (stdlib-only toy: `data.py` Claim schema +
  23-claim synthetic corpus + 5-question mini-bench; `model.py` ClaimStore with all
  four recall channels + RRF `k=60` + greedy diversification + evidence-graph
  neighborhood; `train.py` build → retrieve → grounded synthesis → mini AskChem-Bench
  → channel ablation)
- Breakdown: `breakdown.md`
- Notes: `notes.md`
