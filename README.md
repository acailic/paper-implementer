# Paper Implementer

An autonomous agent that reads, breaks down, and re-implements the most-cited
machine-learning papers from [paperswithcode.com](https://paperswithcode.com).

## How it works

1. **Fetch** the current "most cited" papers list via the
   paperswithcode API (`scripts/fetch_most_cited.py`).
2. **Pick** the next paper from the queue (`papers/tracker.json`) — the agent
   works through the top 10, one or two papers per week.
3. **Read → Re-read → Break down → Code → Write up**, producing a folder per
   paper under `papers/<year>-<short-title>/` containing:
   - `breakdown.md` — section-by-section analysis of the paper
   - `notes.md` — personal reading notes, key insights, questions
   - `implementation/` — a working (or partial) from-scratch code re-implementation
   - `writeup.md` — the agent's own write-up / explanation of the paper
4. **Track** progress in `papers/tracker.json` so the next run picks up where
   the last one left off.

## Folder layout

```
paper-implementer/
├── AGENTS.md                    # Agent operating instructions (the playbook)
├── README.md
├── scripts/
│   ├── fetch_most_cited.py      # Pulls top papers from paperswithcode API
│   └── requirements.txt
├── templates/
│   ├── breakdown.md             # Template for paper breakdown
│   ├── writeup.md               # Template for the write-up
│   └── implementation.md        # Template for the code companion doc
└── papers/
    ├── tracker.json             # Paper queue + progress state
    └── <year>-<short-title>/    # One folder per paper (created per run)
        ├── breakdown.md
        ├── notes.md
        ├── writeup.md
        └── implementation/
```

## Usage

```bash
# See the current most-cited papers
python scripts/fetch_most_cited.py --top 10

# Update the tracker with the latest top-10
python scripts/fetch_most_cited.py --top 10 --update-tracker

# Show what's next in the queue
python scripts/fetch_most_cited.py --next
```

Then point an AI coding agent (Hermes, Claude Code, etc.) at this repo and
tell it to follow `AGENTS.md`.

## License

MIT
