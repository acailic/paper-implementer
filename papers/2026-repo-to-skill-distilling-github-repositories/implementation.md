# Implementation — Repo-To-Skill (deterministic re-implementation)

> Companion notes for the code re-implementation in this folder.
> See `AGENTS.md` Step 5 for the full requirements.

## Paper reference
- **Title:** Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills
  (Chen et al., 2026)
- **ArXiv:** https://arxiv.org/abs/2609.02749

## What's implemented

The paper's Eq. 8 task-agnostic distillation pipeline
`z → scope → Q → ground → X → construct → G̃ → verify → (G, R)` for **one
anchor: a GitHub repository**, with a **deterministic, rule-based
distiller** (no LLM, no paid APIs, CPU-only — the paper uses GPT-5.5 with
xhigh reasoning at ~$40/repo; that entire budget is replaced here by
static analysis + heuristics, and the output is labeled as such in every
generated artifact's frontmatter).

| Paper stage | Here | File |
|---|---|---|
| anchor z | shallow clone (or `--repo-path`) of karpathy/minGPT | `data.py:clone` |
| scope (Source Understanding) | include/exclude evidence boundary + README/packaging/AST source profile | `data.py:evidence_boundary, build_profile` |
| scope (Capability Identification) | trigger-rule capability catalog → Q (8 capabilities) | `model.py:scope` |
| ground (Knowledge Extraction) | capability → evidence mapping X, every unit carrying file:line provenance | `model.py:ground` |
| construct (Tool Encapsulation + Skill Packaging) | three-layer skill: `SKILL.md` + `references/` + `scripts/` + `graph.json` (Eq. 6/7) | `model.py:write_skill` |
| verify (Skill Verification) | 7 static gates + QA carry test (miniature recovery experiment) + construction record R | `verify.py` |
| library router | out of scope (single-repo toy) — see `graph.json` for the graph-level routing that would hang off it | — |

**Not implemented** (paper-only): the LLM creator agent itself, the
1,000-repo library + 20/178 taxonomy induction, paper-derived and
task-oriented distillation, benchmark execution. The QA carry test is a
deliberate miniature of the paper's A.2.2 recovery experiment: *can the
skill answer test questions about the repo, closed-book?*

## How to run

```bash
pip install -r requirements.txt   # nothing to install — stdlib only
python3 train.py                  # clones karpathy/minGPT (default anchor)
# or against an existing checkout:
python3 train.py --repo-path /path/to/minGPT
```

Outputs under `output/`:
- `skills/mingpt/SKILL.md` — the distilled skill (knowledge interface)
- `skills/mingpt/references/{api,quickstart}.md` — knowledge substrate
- `skills/mingpt/scripts/smoke_check.py` — execution interface (static
  probe; reads a checkout, never executes repo code)
- `skills/mingpt/graph.json` — Eq. 7 skill graph (nodes + routing links)
- `record.json` — construction record R (gates, QA, unresolved gaps)
- `results.json` — full run summary

The committed `output/` is a real run against minGPT @ `37baab7`.

## Results

Honest report of the committed run (`python3 train.py --repo-path work/repo`):

- Evidence boundary: 14 files admitted, 88 evidence units
  (37 methods, 11 classes, 13 README sections, ...); 8 capabilities
  identified, all grounded.
- Static gates: **7/7 PASS** (required sections, link integrity,
  provenance, no local-path leakage, capabilities-grounded,
  runtime-separation, scripts-parseable).
- Generated `smoke_check.py` run against the checkout: 42 probes, exit 0.
- QA carry test (10 examiner-written questions about minGPT):
  - **present-carry 10/10 closed-book vs 10/10 open-book (ratio 1.0)** —
    every tested fact from the repo is *contained* in the skill;
  - **retrieved 6/10 closed-book vs 5/10 open-book (ratio 1.2)** — the
    skill is slightly *more* searchable than the repo itself; the four
    misses are retrieval failures, not missing content (present=Y).
- Wall time ≈ 0.3 s on CPU, $0.

## Known gaps / TODOs

- The QA metric is token-overlap retrieval, not comprehension; "retrieved"
  misses (q5, q6, q8, q10) are facts a reader would find by scanning the
  skill but a naive keyword retriever ranks below threshold.
- Capabilities come from a fixed catalog with hand-written triggers;
  repos outside the small-ML-library shape (web services, monorepos) would
  need new triggers.
- No dynamic verification (the paper runs native tests/CLI checks);
  our smoke probe is static only, so runtime claims in the README are
  carried as evidence, not re-executed.
- Single skill graph, no library router, no taxonomy.

## Differences from the original

The paper's distiller is an LLM agent (GPT-5.5/GPT-5.6-sol, xhigh,
~$40/repo) that reads the repo, decides capabilities, writes agent-facing
prose, and verifies with generated assertion cases. Ours is deterministic:
AST/README/packaging extraction with provenance, template assembly, static
gates, and a retrieval-QA recovery test. The four-layer output shape
(SKILL.md / references / scripts / graph), the provenance stamping, the
no-skill-without-verification rule, and the record-R-with-gaps convention
are the parts faithfully carried over. What the deterministic version
cannot do — synthesize procedures the source never states, mine failure
modes from issues, judge evidence admissibility — is exactly the delta a
real LLM distiller would add; `../writeup.md` shows the minGPT skill and
discusses that delta concretely.
