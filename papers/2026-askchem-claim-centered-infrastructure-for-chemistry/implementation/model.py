"""
model.py — The AskChem retrieval engine, from scratch.

Paper: AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis
       Yan, Wolfe, Martiniani, Cho (2026) — https://arxiv.org/abs/2607.28618

This module implements the load-bearing algorithm of the paper:

  1. A CLAIM STORE keyed by claim_id, with provenance (DOI + verbatim quote).
  2. Four parallel recall channels over the same claim store:
       (1) FTS5-style BM25 full-text search  (pure-python BM25)
       (2) paper-level / authority recall     (rank by source-paper citations)
       (3) taxonomy-node recall               (match query to facet labels)
       (4) dense-vector recall                (TF-IDF cosine similarity)
  3. Reciprocal Rank Fusion (RRF, k=60) to fuse the four ranked lists.
  4. Greedy MMR-style diversification to a budget (<= 40 claims).
  5. An evidence-graph neighborhood surface (typed edges: supports /
     contradicts / extends / derives_from / cites_as_evidence).

Everything is implemented in pure Python (stdlib only) so the toy runs with
`python train.py` and no network / GPU / external model. BM25 and TF-IDF
faithfully stand in for SQLite-FTS5 and Sentence-Transformers respectively.
"""

from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

from data import Claim, PAPERS, EDGES


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "for", "is", "are", "was",
    "were", "with", "at", "by", "on", "as", "its", "it", "from", "that",
    "this", "these", "those", "be", "or", "after", "over", "both", "into",
    "more", "than", "but", "we", "our", "their", "his", "her", "no", "not",
    "under", "toward", "towards", "during",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokens, stop-words removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

@dataclass
class PaperMeta:
    doi: str
    title: str
    citations: int
    year: int
    authors: list[str]


class ClaimStore:
    """In-memory analogue of AskChem's SQLite + FTS5 + vector index.

    Holds the claims, the source-paper metadata (authority scores), the
    faceted taxonomy, and the evidence graph edges. Provides the four recall
    channels and the RRF-fused retrieval entry point.
    """

    def __init__(self, claims: list[Claim]):
        self.claims = claims
        self.by_id: dict[str, Claim] = {c.claim_id: c for c in claims}
        self.papers: dict[str, PaperMeta] = {
            doi: PaperMeta(doi,
                           title=str(m["title"]),
                           citations=int(m["citations"]),
                           year=int(m["year"]),
                           authors=list(m["authors"]))
            for doi, m in PAPERS.items()
        }
        # ---- FTS / BM25 index over claim text + verbatim quote ----
        self._docs = []
        for c in claims:
            toks = tokenize(c.text) + tokenize(c.verbatim_quote)
            self._docs.append((c, toks))
        self._build_bm25()
        self._build_tfidf()
        self._build_taxonomy_index()
        self._edges = EDGES  # (src, dst, relation, conf, evidence)

    # -- BM25 (channel 1: FTS5-style full-text search) -----------------------
    def _build_bm25(self):
        N = len(self._docs)
        df = Counter()
        for _, toks in self._docs:
            for t in set(toks):
                df[t] += 1
        self._idf = {t: math.log(1 + (N - df_t + 0.5) / (df_t + 0.5))
                     for t, df_t in df.items()}
        self._doc_len = [len(toks) for _, toks in self._docs]
        self._avgdl = (sum(self._doc_len) / N) if N else 1.0
        self._tf = [Counter(toks) for _, toks in self._docs]

    def _bm25_score(self, q_terms: list[str], i: int, k1=1.5, b=0.75) -> float:
        score = 0.0
        tf = self._tf[i]
        dl = self._doc_len[i]
        denom_norm = k1 * (1 - b + b * dl / self._avgdl)
        for t in q_terms:
            if t not in tf:
                continue
            idf = self._idf.get(t, 0.0)
            f = tf[t]
            score += idf * (f * (k1 + 1)) / (f + denom_norm)
        return score

    def channel_fts(self, q_terms: list[str], k: int = 50) -> list[Claim]:
        """Channel 1: FTS5 (BM25) ranked list."""
        scored = [(self._bm25_score(q_terms, i), self._docs[i][0])
                  for i in range(len(self._docs))]
        scored.sort(key=lambda x: (-x[0], x[1].claim_id))
        return [c for s, c in scored[:k] if s > 0]

    # -- TF-IDF cosine (channel 4: dense-vector recall) ----------------------
    def _build_tfidf(self):
        # vector per doc: tf-idf, then L2-normalized (the "embedding")
        self._vecs: list[dict[str, float]] = []
        for _, toks in self._docs:
            tf = Counter(toks)
            v = {t: (tf[t] / len(toks)) * self._idf.get(t, 0.0) for t in tf}
            norm = math.sqrt(sum(w * w for w in v.values())) or 1.0
            v = {t: w / norm for t, w in v.items()}
            self._vecs.append(v)
        # precompute query vector lazily

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b.get(t, 0.0) for t, w in a.items())

    def channel_vector(self, q_terms: list[str], k: int = 50) -> list[Claim]:
        """Channel 4: dense-vector (TF-IDF cosine) ranked list."""
        tf = Counter(q_terms)
        if not q_terms:
            return []
        qvec = {t: (tf[t] / len(q_terms)) * self._idf.get(t, 0.0)
                for t in tf}
        norm = math.sqrt(sum(w * w for w in qvec.values())) or 1.0
        qvec = {t: w / norm for t, w in qvec.items()}
        scored = [(self._cosine(qvec, self._vecs[i]), self._docs[i][0])
                  for i in range(len(self._vecs))]
        scored.sort(key=lambda x: (-x[0], x[1].claim_id))
        return [c for s, c in scored[:k] if s > 0]

    # -- Paper / authority recall (channel 2) --------------------------------
    def channel_paper(self, q_terms: list[str], k: int = 50) -> list[Claim]:
        """Channel 2: rank claims by the authority score of their source paper.

        AskChem ranks by a paper-level impact/recall score; here we use the
        source paper's citation count (the natural toy analogue). We only
        consider papers whose *title* shares a query term, mirroring a
        paper-recall stage that first narrows by topical match.
        """
        qset = set(q_terms)
        candidate_dois = set()
        for doi, p in self.papers.items():
            if qset & set(tokenize(p.title)):
                candidate_dois.add(doi)
        if not candidate_dois:
            candidate_dois = set(self.papers)
        scored = [
            (self.papers[c.source_doi].citations, c)
            for c in self.claims if c.source_doi in candidate_dois
        ]
        scored.sort(key=lambda x: (-x[0], x[1].claim_id))
        return [c for s, c in scored[:k]]

    # -- Taxonomy-node recall (channel 3) ------------------------------------
    def _build_taxonomy_index(self):
        # map facet-path -> [claim_ids]; also map a flat label word -> paths
        self._facet_paths: dict[str, list[str]] = defaultdict(list)
        for c in self.claims:
            for view, path in c.facets.items():
                self._facet_paths[path].append(c.claim_id)
        # label index: each leaf word -> set of paths
        self._facet_labels: dict[str, set[str]] = defaultdict(set)
        for path in self._facet_paths:
            for tok in path.replace("/", " ").split():
                self._facet_labels[tok].add(path)

    def channel_taxonomy(self, q_terms: list[str], k: int = 50) -> list[Claim]:
        """Channel 3: match query terms to facet-path labels, recall claims.

        This is AskChem's stabilized-faceted-taxonomy recall: a query term
        matches taxonomy node labels; the matched node recalls all claims
        filed under that path.
        """
        matched_paths: set[str] = set()
        for t in q_terms:
            # direct + synonym-ish normalization (suzuki/coupling/co2...)
            for path in self._facet_labels.get(t, set()):
                matched_paths.add(path)
            # also match by substring of leaf word (e.g. "c2" in co2_to_c2)
            for label, paths in self._facet_labels.items():
                if t in label:
                    matched_paths.update(paths)
        recalled_ids: list[str] = []
        for path in matched_paths:
            recalled_ids.extend(self._facet_paths[path])
        # rank recalled claims by how many matched paths they sit under
        cnt = Counter(recalled_ids)
        scored = [(n, self.by_id[cid]) for cid, n in cnt.items()]
        scored.sort(key=lambda x: (-x[0], x[1].claim_id))
        return [c for n, c in scored[:k]]

    # -- Reciprocal Rank Fusion ---------------------------------------------
    @staticmethod
    def rrf_fuse(channels: list[list[Claim]], k_const: int = 60) -> list[Claim]:
        """Fuse N ranked lists by Reciprocal Rank Fusion.

            RRF(d) = sum_i  1 / (k + r_i(d))

        where r_i(d) is the 1-indexed rank of claim d in channel i (absent =>
        +inf => contributes 0). k_const = 60 is the AskChem default.
        """
        scores: dict[str, float] = defaultdict(float)
        order: dict[str, Claim] = {}
        for ranked in channels:
            for rank_0, c in enumerate(ranked):
                r = rank_0 + 1            # 1-indexed rank
                scores[c.claim_id] += 1.0 / (k_const + r)
                order[c.claim_id] = c
        fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [order[cid] for cid, _ in fused]

    # -- Diversification (greedy MMR-style, to a budget) ---------------------
    def diversify(self, fused: list[Claim], budget: int = 40,
                  max_per_doi: int = 2) -> list[Claim]:
        """Greedy de-dup: cap how many claims share the same source paper
        (analogous to AskChem diversifying to <= 40 claims so a single paper
        cannot dominate the evidence set)."""
        out: list[Claim] = []
        doi_count: Counter = Counter()
        for c in fused:
            if doi_count[c.source_doi] >= max_per_doi:
                continue
            out.append(c)
            doi_count[c.source_doi] += 1
            if len(out) >= budget:
                break
        return out

    # -- The full forward pass: query -> diversified fused claims -----------
    def retrieve(self, query: str, k_per_channel: int = 50,
                 budget: int = 40, k_const: int = 60,
                 return_channel_lists: bool = False):
        """End-to-end retrieval: query text -> RRF-fused, diversified claims."""
        q_terms = tokenize(query)
        ch_fts = self.channel_fts(q_terms, k_per_channel)
        ch_paper = self.channel_paper(q_terms, k_per_channel)
        ch_tax = self.channel_taxonomy(q_terms, k_per_channel)
        ch_vec = self.channel_vector(q_terms, k_per_channel)
        fused = self.rrf_fuse([ch_fts, ch_paper, ch_tax, ch_vec], k_const)
        diversified = self.diversify(fused, budget)
        if return_channel_lists:
            return diversified, {
                "fts": ch_fts, "paper": ch_paper,
                "taxonomy": ch_tax, "vector": ch_vec,
            }
        return diversified

    # -- Evidence-graph neighborhood -----------------------------------------
    def neighborhood(self, claim_id: str) -> list[dict]:
        """Return typed edges touching a claim (the /neighborhood surface)."""
        out = []
        for src, dst, rel, conf, ev in self._edges:
            if src == claim_id or dst == claim_id:
                other = dst if src == claim_id else src
                out.append({"relation": rel, "to_claim_id": other,
                            "to_text": self.by_id[other].text,
                            "confidence": conf, "evidence": ev})
        return out


# ---------------------------------------------------------------------------
# A toy "reader LLM" that synthesizes a grounded answer from retrieved claims.
# ---------------------------------------------------------------------------

def synthesize_answer(retrieved: list[Claim], query: str) -> dict:
    """Stands in for AskChem's reader LLM (GPT-5.5 in the paper).

    The real reader is an LLM constrained to cite only DOIs present in the
    retrieved evidence set. Our toy synthesizer approximates this by:
      - extracting the numeric findings from the most relevant claims,
      - emitting one grounded sentence per claim with an inline [DOI] marker,
      - only ever citing DOIs that actually appear in `retrieved`.
    This is what makes every cited DOI "real" (100% DOI existence).
    """
    if not retrieved:
        return {"text": "No relevant claims were found in the corpus.",
                "cited_dois": []}

    lines = [f"Query: {query}", ""]
    cited = []
    for c in retrieved[:6]:
        value = ""
        if c.numeric_value is not None:
            value = f" ({c.numeric_value:g} {c.numeric_unit})".rstrip()
        lines.append(f"- {c.text}{value} [{c.source_doi}]")
        if c.source_doi not in cited:
            cited.append(c.source_doi)
    lines.append("")
    lines.append("Each claim above is grounded in a verbatim quote from the "
                 "source paper (see claim store).")
    return {"text": "\n".join(lines), "cited_dois": cited}
