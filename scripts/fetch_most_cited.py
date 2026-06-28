#!/usr/bin/env python3
"""
fetch_most_cited.py — pull the most-cited / most-discussed papers and build
the reading queue for the paper-implementer agent.

BACKGROUND
  paperswithcode.com now redirects to huggingface.co (the PWC catalog and the
  /api/v1 endpoints were absorbed by Hugging Face in 2025). The live successor
  of the old "trending / most discussed" list is the Hugging Face Daily Papers
  feed (https://huggingface.co/papers), which exposes a public JSON API.

  True citation counts live on Semantic Scholar, but its public API rate-limits
  aggressively without a key. So:
    - PRIMARY source  = Hugging Face Daily Papers (no key, always works)
                        ranked by `upvotes` (the community engagement signal —
                        the closest available proxy to "most cited / trending").
    - ENRICHMENT      = Semantic Scholar citation counts, OPTIONAL, used only
                        when S2_API_KEY is set in the environment. With a key
                        the tracker records real citation_count; without one it
                        records the HF upvote count as `popularity`.

Usage:
    python fetch_most_cited.py --top 10                    # print current top 10
    python fetch_most_cited.py --top 10 --update-tracker   # rewrite tracker.json
    python fetch_most_cited.py --next                      # print next pending paper

Optional env:
    S2_API_KEY   Semantic Scholar API key — enables real citation-count enrichment
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "ERROR: requests is required. Install with:\n"
        "  pip install -r scripts/requirements.txt\n"
    )
    sys.exit(1)

HF_API = "https://huggingface.co/api/daily_papers"
S2_API = "https://api.semanticscholar.org/graph/v1/paper"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKER_PATH = REPO_ROOT / "papers" / "tracker.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def slugify(title: str) -> str:
    """Filesystem-safe short slug from a paper title."""
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9\s-]", "", t)
    t = re.sub(r"[\s_-]+", "-", t).strip("-")
    return "-".join(t.split("-")[:6])[:60]


def _get(url: str, params: dict | None = None, headers: dict | None = None,
         retries: int = 4, backoff: float = 2.0) -> requests.Response:
    """GET with exponential backoff on 429/5xx."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = backoff ** attempt
                sys.stderr.write(
                    f"  rate-limited/transient ({r.status_code}); retrying in {wait:.0f}s\n"
                )
                time.sleep(wait)
                continue
            return r
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff ** attempt)
    if last_exc:
        raise last_exc
    return r  # type: ignore[name-defined]


# --------------------------------------------------------------------------- #
# primary source: Hugging Face Daily Papers
# --------------------------------------------------------------------------- #
def fetch_hf_papers(target: int = 50) -> list[dict]:
    """
    Page through the HF daily-papers feed until we have ~`target` entries.

    The feed returns recent papers day-by-day. We walk back through days and
    keep the highest-upvoted ones to build a "most discussed" ranking.
    """
    collected: list[dict] = []
    seen_ids: set[str] = set()
    # The API supports a `date` query (YYYY-MM-DD). Walk backwards from today.
    from datetime import date as _date, timedelta
    day = _date.today()
    days_tried = 0
    while len(collected) < target and days_tried < 90:
        params = {"date": day.isoformat()}
        try:
            r = _get(HF_API, params=params)
            if r.status_code == 200:
                batch = r.json()
                if isinstance(batch, list):
                    for entry in batch:
                        pid = (entry.get("paper") or {}).get("id") or entry.get("title")
                        if pid and pid not in seen_ids:
                            seen_ids.add(pid)
                            collected.append(entry)
            else:
                sys.stderr.write(f"  HF {day}: status {r.status_code}\n")
        except Exception as e:  # pragma: no cover
            sys.stderr.write(f"  HF {day}: {e}\n")
        day -= timedelta(days=1)
        days_tried += 1
        time.sleep(0.2)
    sys.stderr.write(f"  HF: collected {len(collected)} papers over {days_tried} days\n")
    return collected


def normalize_hf(entry: dict) -> dict:
    paper = entry.get("paper") or {}
    title = (entry.get("title") or paper.get("title") or "untitled").strip()
    arxiv_id = paper.get("id") or entry.get("paper_id") or ""
    # HF ids are already arxiv ids like "2606.23835"
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id and "/" not in arxiv_id else ""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id and "/" not in arxiv_id else ""
    code_url = paper.get("githubRepo") or ""
    if code_url and not code_url.startswith("http"):
        code_url = f"https://github.com/{code_url}"
    pub = entry.get("publishedAt") or paper.get("publishedAt") or ""
    year = pub[:4] if pub and len(pub) >= 4 else ""
    authors = [a.get("name", "") for a in (paper.get("authors") or []) if isinstance(a, dict)]
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "arxiv_id": arxiv_id,
        "arxiv_url": arxiv_url,
        "pdf_url": pdf_url,
        "code_url": code_url,
        "project_page": paper.get("projectPage") or "",
        "abstract": (entry.get("summary") or paper.get("summary") or "").strip(),
        "ai_keywords": paper.get("ai_keywords") or [],
        "popularity": int(paper.get("upvotes") or 0),  # HF upvotes = engagement proxy
        "citation_count": None,  # filled by S2 enrichment if available
        "pwc_url": f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else "",
    }


# --------------------------------------------------------------------------- #
# optional enrichment: Semantic Scholar citation counts
# --------------------------------------------------------------------------- #
def enrich_with_s2(papers: list[dict]) -> None:
    """Mutate `papers` in place, adding real citation_count where S2 knows it."""
    key = os.environ.get("S2_API_KEY")
    if not key:
        sys.stderr.write("  S2 enrichment: skipped (set S2_API_KEY to enable)\n")
        return
    headers = {"x-api-key": key}
    sys.stderr.write(f"  S2 enrichment: looking up {len(papers)} papers ...\n")
    for p in papers:
        aid = p.get("arxiv_id")
        if not aid:
            continue
        try:
            r = _get(
                f"{S2_API}/{aid}",
                params={"fields": "citationCount"},
                headers=headers,
                retries=3,
            )
            if r.status_code == 200:
                p["citation_count"] = int(r.json().get("citationCount") or 0)
            else:
                sys.stderr.write(f"    S2 {aid}: status {r.status_code}\n")
        except Exception as e:
            sys.stderr.write(f"    S2 {aid}: {e}\n")
        time.sleep(0.4)  # respect S2 public limit (~100 req/5min w/o key, more with)


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #
def rank_key(p: dict) -> tuple:
    """Prefer real citation_count when present, else fall back to popularity."""
    cc = p.get("citation_count")
    return (cc if cc is not None else 0, p.get("popularity", 0))


def top_n(papers: list[dict], n: int) -> list[dict]:
    return sorted(papers, key=rank_key, reverse=True)[:n]


# --------------------------------------------------------------------------- #
# tracker
# --------------------------------------------------------------------------- #
def load_tracker() -> dict:
    if TRACKER_PATH.exists():
        try:
            return json.loads(TRACKER_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"created_at": datetime.now(timezone.utc).isoformat(), "papers": []}


def save_tracker(tracker: dict) -> None:
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(json.dumps(tracker, indent=2, ensure_ascii=False) + "\n")


def update_tracker(top: list[dict]) -> tuple[int, int]:
    tracker = load_tracker()
    existing = {p["title"].lower(): p for p in tracker.get("papers", [])}
    added = 0
    for rank, paper in enumerate(top, 1):
        key = paper["title"].lower()
        if key in existing:
            keep = {"status", "folder", "started_at", "finished_at", "skip_reason", "notes_path"}
            for k, v in paper.items():
                if k not in keep:
                    existing[key][k] = v
            existing[key]["rank"] = rank
        else:
            paper["rank"] = rank
            paper["status"] = "pending"
            paper["folder"] = f"papers/{paper['year'] or 'XXXX'}-{slugify(paper['title'])}"
            tracker["papers"].append(paper)
            added += 1
    tracker["papers"].sort(key=lambda p: p.get("rank", 9999))
    tracker["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_tracker(tracker)
    return added, len(tracker["papers"])


# --------------------------------------------------------------------------- #
# CLI commands
# --------------------------------------------------------------------------- #
def cmd_next() -> int:
    tracker = load_tracker()
    pending = [p for p in tracker.get("papers", []) if p.get("status") == "pending"]
    if not pending:
        print("No pending papers. Run with --update-tracker to refresh the queue.")
        return 1
    p = pending[0]
    print(f"NEXT PAPER (rank {p.get('rank')}):")
    print(f"  title:    {p['title']}")
    print(f"  year:     {p.get('year', '?')}")
    cc = p.get("citation_count")
    print(f"  cites:    {cc if cc is not None else 'n/a'}  (popularity/upvotes: {p.get('popularity', 0)})")
    print(f"  arxiv:    {p.get('arxiv_url', '-')}")
    print(f"  pdf:      {p.get('pdf_url', '-')}")
    print(f"  code:     {p.get('code_url', '-')}")
    print(f"  folder:   {p.get('folder', '-')}")
    return 0


def cmd_top(n: int, do_update: bool) -> int:
    sys.stderr.write("Fetching papers from Hugging Face (paperswithcode successor) ...\n")
    raw = fetch_hf_papers(target=max(n * 3, 30))
    papers = [normalize_hf(e) for e in raw]
    enrich_with_s2(papers)
    top = top_n(papers, n)
    metric = "citation_count" if top and top[0].get("citation_count") is not None else "popularity"
    print(f"\n{'='*72}\nTop {n} papers by {metric} (as of {date.today()}):\n{'='*72}\n")
    for i, p in enumerate(top, 1):
        cc = p.get("citation_count")
        score = f"{cc:>7} cites" if cc is not None else f"{p.get('popularity',0):>4} upvotes"
        print(f"{i:>2}. [{score}] {p['title']}")
        print(f"    {p.get('year', '?')}  arxiv: {p.get('arxiv_url', '-')}\n")
    if do_update:
        added, total = update_tracker(top)
        sys.stderr.write(
            f"\nTracker updated: +{added} new, {total} total in "
            f"{TRACKER_PATH.relative_to(REPO_ROOT)}\n"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=10, help="how many top papers to fetch (default 10)")
    ap.add_argument("--update-tracker", action="store_true", help="write results into papers/tracker.json")
    ap.add_argument("--next", action="store_true", help="print the next pending paper from the tracker")
    args = ap.parse_args()
    if args.next:
        return cmd_next()
    return cmd_top(args.top, args.update_tracker)


if __name__ == "__main__":
    raise SystemExit(main())
