# implementation/ — deterministic repo-to-skill distiller

Re-implementation of the task-agnostic distillation pipeline from
**"Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills"**
(Chen et al., 2026, arXiv:2609.02749) — see `../breakdown.md` for the
method and `../implementation.md` for scope/limitations.

```
data.py       anchor + evidence boundary + README/AST/packaging parsing (scope/ground inputs)
model.py      the Distiller: scope->Q, ground->X, construct->G (deterministic, no LLM)
verify.py     static gates + QA carry test (mini recovery experiment) + construction record R
train.py      end-to-end runner (clone -> distill -> verify -> report)
questions.json  examiner-written test questions about the anchor repo (minGPT)
output/       committed artifacts of a real run: skills/mingpt/** + record.json + results.json
work/         scratch (repo checkout; gitignored, untrusted data, never executed)
```

## Run

```bash
python3 train.py                      # default anchor: https://github.com/karpathy/minGPT
python3 train.py --repo-path /path/to/repo --out /tmp/out
```

Stdlib only, CPU only, ~0.3 s. Exit code 0 iff all static gates pass.
The QA carry result and any unresolved gaps are recorded in
`output/record.json` — gaps are recorded, not hidden (paper Sec. 3.2).
