# AskChem — toy re-implementation (claim-centered chemistry retrieval)

> **Paper:** AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis
> **Authors:** Bing Yan, Gregory Wolfe, Stefano Martiniani, Kyunghyun Cho (2026)
> **ArXiv:** https://arxiv.org/abs/2607.28618
> **Official code:** https://github.com/bingyan4science/askchem · **Live:** https://askchem.org

This is a **from-scratch, dependency-free** re-implementation of the
load-bearing algorithm of AskChem: **claim-centered 4-channel hybrid retrieval
fused by Reciprocal Rank Fusion (RRF)**, plus a mini AskChem-Bench. No LLM, no
network, no GPU — pure Python 3.10+ standard library.

## What's implemented

| Paper component | Status | How it's realized in the toy |
|---|---|---|
| **Atomic claim primitive** with provenance triple `(type, doi, quote)` | ✅ | `Claim` dataclass in `data.py` |
| **Claim store** (SQLite + FTS5 + vectors) | ✅ | `ClaimStore` in `model.py` (in-memory) |
| **Channel 1: FTS5 (BM25)** | ✅ | pure-Python BM25 (`channel_fts`) |
| **Channel 2: paper/authority recall** | ✅ | rank by source-paper citations (`channel_paper`) |
| **Channel 3: taxonomy-node recall** | ✅ | stabilized faceted-path label match (`channel_taxonomy`) |
| **Channel 4: dense-vector recall** | ✅ | TF-IDF cosine (`channel_vector`) |
| **RRF fusion (k=60)** | ✅ | `rrf_fuse` — the paper's exact combiner |
| **Diversification (≤40)** | ✅ | greedy DOI-cap MMR-style (`diversify`) |
| **Evidence graph / neighborhood** | ✅ | typed edges (`supports/contradicts/...`) |
| **Reader LLM (grounded synthesis)** | ◐ | deterministic toy synthesizer (`synthesize_answer`) |
| **AskChem-Bench metrics** | ✅ | DOI existence, citation density, recall@k |
| **Channel ablation** | ✅ | leave-one-out + per-channel first-gold rank |

## What's simplified / skipped

- **Extraction** — the paper runs GPT-5-mini / Gemini 3.1 Pro over 147K real
  PDFs to emit 2.4M claims. The toy uses a **hand-authored synthetic corpus**
  of 23 claims across 8 plausible chemistry papers. Extraction-by-LLM is
  orthogonal to the retrieval algorithm that is the paper's actual contribution.
- **Embedding model** — production uses Sentence-Transformers; the toy uses
  **TF-IDF cosine**, which is the standard minimal substitute and exercises the
  same dense-recall code path.
- **Taxonomy stabilization** — the paper induces+stabilizes 307K nodes; the toy
  uses pre-stabilized facet paths attached to each claim.
- **CrossRef** — real DOI existence checks against CrossRef; the toy treats the
  local paper registry as the resolvable set.

## How to run

```bash
# no dependencies — Python 3.10+ only
python train.py
```

## Results (actual run output)

The toy reproduces the **qualitative direction of the paper's Table 1**:

| Setting | DOI existence % | Citation density | Recall@8 |
|---|---|---|---|
| **+AskChem** (claim retrieval) | **100.0** | **3.4** | **1.00** |
| LLM only (hallucinating baseline) | 12.0 | 0.6 | 0.07 |

The claim-grounded path achieves 100% DOI existence because every cited DOI is,
by construction, one of the retrieved claims' provenance anchors — exactly the
guarantee AskChem's reader-LLM constraint provides in production.

**Channel ablation (leave-one-out)** shows the lexical channels (FTS, vector)
surface the relevant claim fastest (mean 1st-gold rank ≈ 1.0–1.4), while
paper-only and taxonomy-only lag (2.6) — consistent with the intuition that
semantic/lexical match drives precision and the authority/taxonomy channels
broaden recall.

## Files

```
implementation/
├── README.md          # this file
├── data.py            # Claim schema + synthetic chemistry corpus + bench
├── model.py           # ClaimStore: 4 channels, RRF, diversify, neighborhood
├── train.py           # build store → retrieve → bench → ablation
└── requirements.txt   # (none needed — stdlib only)
```

## Differences from the original

See "What's simplified / skipped" above. The core algorithm (RRF fusion of 4
heterogeneous rank lists + claim-grounded synthesis → 100% DOI existence) is
faithfully reproduced; only scale, LLM extraction, and learned embeddings are
toyed down.
