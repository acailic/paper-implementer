"""
run.py — Main runner for the toy agent-memory benchmark.

Pipeline:
  1. Load the toy multi-session dataset (4 sessions, with revised facts).
  2. Build the memory system M_sys = <R,S,Q,U> by ingesting each turn.
  3. Run ~10 QA queries (temporal reasoning + multi-hop entity).
  4. Print:
       (a) which memory objects were retrieved for each query,
       (b) whether the answer was found (substring match vs gold),
       (c) a Recall@1/@3/@5 summary table,
       (d) a demonstration that a REVISED fact correctly invalidates the old,
       (e) an ASCII quality curve.
  5. Save a metrics JSON for reproducibility.

Run:  python3 run.py
Deps: numpy + stdlib only.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from agent_memory import AgentMemorySystem, MemoryObject
from data import QAItem, Session, dataloader

ART = True  # ASCII-art on

# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #


def recall_at_k(
    retrieved_sessions: List[int], gold_sessions: List[int], k: int
) -> float:
    """Evidence-level Recall@K (paper's RQ2 definition): a hit requires the
    top-K retrieved source-id groups to contain the annotated gold evidence."""
    topk = retrieved_sessions[:k]
    return 1.0 if any(g in topk for g in gold_sessions) else 0.0


def substring_em(pred: str, gold: str) -> float:
    """Substring exact-match (LongMemEval style). Case-insensitive."""
    p, g = pred.lower().strip(), gold.lower().strip()
    if not g:
        return 1.0 if not p else 0.0
    return 1.0 if g in p else 0.0


# --------------------------------------------------------------------------- #
# ASCII curve                                                                 #
# --------------------------------------------------------------------------- #


def ascii_curve(series: List[float], title: str, width: int = 50, height: int = 12, ymax: float = 0) -> str:
    """Render a tiny ASCII line chart for a 1-D series."""
    if not series:
        return f"{title}: (no data)"
    lo = min(series)
    hi = max(series)
    if ymax > 0 and hi < ymax:
        hi = ymax
    if hi - lo < 1e-9:
        # When all values are identical (e.g. all EM=1.0), extend range so
        # the chart renders a flat line rather than exploding to lo+1.0.
        margin = 0.05 * max(abs(lo), 0.1)
        lo -= margin
        hi = lo + 2 * margin
    cols = max(width, len(series))
    # map each point to a column
    grid = [[" "] * cols for _ in range(height)]
    prev_row = None
    for i, v in enumerate(series):
        col = int(i / max(1, len(series) - 1) * (cols - 1))
        row = int((1 - (v - lo) / (hi - lo)) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"
        if prev_row is not None:
            # interpolate a vertical line between prev and current
            step = 1 if row >= prev_row else -1
            for r in range(prev_row, row, step):
                if grid[r][col] == " ":
                    grid[r][col] = ":"
        prev_row = row
    lines = [f"{title}  (min={lo:.3f} max={hi:.3f})"]
    # y-axis labels
    for r in range(height):
        val = hi - (r / (height - 1)) * (hi - lo)
        label = f"{val:5.2f} |"
        lines.append(label + "".join(grid[r]))
    lines.append("      +" + "-" * cols)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Demo: invalidation explicitly                                               #
# --------------------------------------------------------------------------- #


def show_invalidation(sys: AgentMemorySystem) -> Dict:
    """Demonstrate Module U multi-versioning: find the 'Paris' memory and show
    it is logically invalidated, superseded by the 'London' memory."""
    paris_objs = [
        mo for mo in sys.memories.values() if "Paris" in mo.text and "live" in mo.text.lower()
    ]
    london_objs = [
        mo for mo in sys.memories.values()
        if "London" in mo.text and "moved" in mo.text.lower()
    ]
    report = {"paris": [], "london": []}
    for mo in paris_objs:
        report["paris"].append(
            {
                "id": mo.id,
                "text": mo.text,
                "valid": mo.valid,
                "valid_until": mo.valid_until,
                "superseded_by": mo.superseded_by,
            }
        )
    for mo in london_objs:
        report["london"].append(
            {"id": mo.id, "text": mo.text, "valid": mo.valid, "version_of": mo.version_of}
        )
    return report


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def banner(title: str, ch: str = "=") -> str:
    line = ch * 72
    return f"\n{line}\n{title}\n{line}"


def main() -> int:
    sessions, queries = dataloader()
    print(banner("M_sys = <R, S, Q, U>  —  Agent Memory Benchmark (toy)"))
    print(
        "Paper: 'Are We Ready For An Agent-Native Memory System?' "
        "(arXiv:2606.24775, 2026)"
    )
    print(
        "Winning design: composite R + raw/late-filter S + balanced-hybrid Q "
        "(RRF) + conservative multi-version U.\n"
    )

    # ---- build the system ------------------------------------------------ #
    sys_ = AgentMemorySystem(
        embedder=None,          # default FakeEmbedder
        rrf_k=60,               # F8 RRF constant
        consolidation_sim=0.85, # F9 conservative threshold
        capacity=1000,
    )
    n_turns = 0
    for s in sessions:
        for turn in s.turns:
            sys_.ingest_turn(turn.text, turn.role, turn.session_id, turn.timestamp)
            n_turns += 1
    print(f"Ingested {n_turns} turns across {len(sessions)} sessions -> "
          f"{len(sys_.memories)} memory objects.")
    print(
        "Indices: dense(rows={d}) bm25(terms={t}) entity(nodes={e})".format(
            d=len(sys_._matrix),
            t=len(sys_._inverted),
            e=len(sys_._entity_index),
        )
    )

    # ---- per-query evaluation ------------------------------------------- #
    print(banner("Per-query retrieval + answer", "-"))
    per_query: List[Dict] = []
    recall1: List[float] = []
    recall3: List[float] = []
    recall5: List[float] = []
    em_scores: List[float] = []
    # we also track a running "quality" series for the ASCII curve:
    # cumulative average EM over the query stream.
    cum_em: List[float] = []

    for i, q in enumerate(queries, 1):
        results = sys_.retrieve(q.query, k=5, plan=True, only_valid=True)
        retrieved_sessions = [mo.session_id for mo, _ in results]
        r1 = recall_at_k(retrieved_sessions, q.gold_session_ids, 1)
        r3 = recall_at_k(retrieved_sessions, q.gold_session_ids, 3)
        r5 = recall_at_k(retrieved_sessions, q.gold_session_ids, 5)
        recall1.append(r1)
        recall3.append(r3)
        recall5.append(r5)
        # answer extraction scans a WIDER evidence pool so the slot-filler can
        # find the right sentence even when it ranks just below top-k.
        wide = sys_.retrieve(q.query, k=12, plan=True, only_valid=True)
        pred = AgentMemorySystem.answer_from_evidence(q.query, [mo for mo, _ in wide])
        em = substring_em(pred, q.gold_answer)
        em_scores.append(em)
        cum_em.append(float(np.mean(em_scores)))
        per_query.append(
            {
                "query": q.query,
                "gold": q.gold_answer,
                "pred": pred,
                "em": em,
                "recall@1": r1,
                "recall@3": r3,
                "recall@5": r5,
                "retrieved": [(mo.id, mo.session_id, round(s, 4)) for mo, s in results],
            }
        )
        print(f"\n[Q{i}] {q.query}")
        print(f"      note: {q.note}")
        print(f"      gold: {q.gold_answer!r}  ->  pred: {pred!r}  EM={em:.0f}")
        print(f"      Recall@1/3/5 = {r1:.0f}/{r3:.0f}/{r5:.0f}")
        print("      retrieved evidence:")
        for mo, score in results:
            star = " <GOLD>" if mo.session_id in q.gold_session_ids else ""
            print(f"        - {mo.short(70)}  rrf={score:.4f}{star}")

    # ---- summary table --------------------------------------------------- #
    print(banner("Summary metrics", "-"))
    mean_r1 = float(np.mean(recall1))
    mean_r3 = float(np.mean(recall3))
    mean_r5 = float(np.mean(recall5))
    mean_em = float(np.mean(em_scores))
    print(f"{'Metric':<20} {'Value':>8}")
    print("-" * 30)
    print(f"{'Recall@1':<20} {mean_r1:>8.3f}")
    print(f"{'Recall@3':<20} {mean_r3:>8.3f}")
    print(f"{'Recall@5':<20} {mean_r5:>8.3f}")
    print(f"{'Substring EM':<20} {mean_em:>8.3f}")
    print(f"{'#queries':<20} {len(queries):>8}")
    print(f"{'#memory objects':<20} {len(sys_.memories):>8}")

    # ---- ASCII quality curve -------------------------------------------- #
    print(banner("Cumulative answer-quality (substring EM) curve over the query stream", "-"))
    print(ascii_curve(cum_em, "cumulative mean EM", ymax=1.0))
    print("\nEach '*' is one query (left->right = query order). Up = better.")

    # also: per-query EM as a bar-ish chart
    print(banner("Per-query substring EM (1.0 = hit, 0.0 = miss)", "-"))
    print(ascii_curve(em_scores, "per-query EM", ymax=1.0))

    # ---- invalidation demo (Module U) ----------------------------------- #
    print(banner("Module U — multi-versioning / logical invalidation demo", "-"))
    inv = show_invalidation(sys_)
    print("OLD assertion (should be INVALIDATED):")
    for p in inv["paris"]:
        print(
            f"  #{p['id']}: valid={p['valid']}  valid_until={p['valid_until']}  "
            f"superseded_by={p['superseded_by']}"
        )
        print(f"       text: {p['text']!r}")
    print("\nNEW assertion (the current valid version):")
    for l in inv["london"]:
        print(
            f"  #{l['id']}: valid={l['valid']}  version_of={l['version_of']}"
        )
        print(f"       text: {l['text']!r}")
    ok = bool(inv["paris"]) and all(not p["valid"] for p in inv["paris"]) and bool(inv["london"])
    print(f"\nInvalidation correctly applied: {ok}")

    # ---- save metrics ---------------------------------------------------- #
    out = {
        "n_turns": n_turns,
        "n_memory_objects": len(sys_.memories),
        "n_queries": len(queries),
        "metrics": {
            "Recall@1": mean_r1,
            "Recall@3": mean_r3,
            "Recall@5": mean_r5,
            "SubstringEM": mean_em,
        },
        "per_query": per_query,
        "invalidation": inv,
        "invalidation_ok": ok,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMetrics written to: {out_path}")
    print(banner("DONE", "="))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
