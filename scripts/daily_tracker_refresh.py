#!/usr/bin/env python3
"""
daily_tracker_refresh.py — smart tracker diff for the paper-implementer cron.

Runs fetch_most_cited.py --update-tracker, then diffs before/after.
Prints a summary ONLY if something changed (new papers added, status shifts).
Empty stdout = silent (watchdog pattern).
"""
import json, subprocess, sys
from pathlib import Path

REPO = Path("/home/nistrator/Documents/github/paper-implementer")
TRACKER = REPO / "papers" / "tracker.json"

def load_tracker():
    if not TRACKER.exists():
        return {}
    try:
        return json.loads(TRACKER.read_text())
    except Exception:
        return {}

# snapshot before
before = load_tracker()
before_titles = {p["title"].lower(): p.get("status") for p in before.get("papers", [])}

# run fetch + update (stderr goes to console, we don't want it in stdout)
r = subprocess.run(
    [sys.executable, str(REPO / "scripts" / "fetch_most_cited.py"),
     "--top", "10", "--update-tracker"],
    capture_output=True, text=True, cwd=str(REPO), timeout=120,
)
if r.returncode != 0:
    sys.stderr.write(r.stderr)
    print(f"⚠️ Tracker refresh FAILED: {r.stderr[:200]}")
    sys.exit(0)  # don't alert on transient API errors

# snapshot after
after = load_tracker()
after_papers = after.get("papers", [])

new_papers = []
status_changes = []
for p in after_papers:
    key = p["title"].lower()
    if key not in before_titles:
        new_papers.append(p)
    elif before_titles[key] != p.get("status"):
        status_changes.append((p, before_titles[key], p.get("status")))

# silent if nothing changed
if not new_papers and not status_changes:
    sys.exit(0)

lines = ["📋 **Paper Tracker — Daily Refresh**\n"]
if new_papers:
    lines.append(f"🆕 {len(new_papers)} new paper(s) added to the queue:\n")
    for p in new_papers:
        score = p.get("citation_count") or f"{p.get('popularity',0)} upvotes"
        lines.append(f"  • **{p['title']}** ({p.get('year','?')}) — {score}")
        lines.append(f"    {p.get('arxiv_url','')}\n")

if status_changes:
    lines.append("📊 Status changes:\n")
    for p, old, new in status_changes:
        lines.append(f"  • {p['title'][:50]}: {old} → **{new}**")

# queue summary
pending = [p for p in after_papers if p.get("status") == "pending"]
in_progress = [p for p in after_papers if p.get("status") not in ("done", "pending", "skipped")]
done = [p for p in after_papers if p.get("status") == "done"]
lines.append(f"\n📈 Queue: {len(pending)} pending · {len(in_progress)} in progress · {len(done)} done")
if pending:
    nxt = pending[0]
    lines.append(f"⏭️ Next up: **{nxt['title'][:60]}**")

print("\n".join(lines))
