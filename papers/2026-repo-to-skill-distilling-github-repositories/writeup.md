# Writeup — Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills

> **Paper:** Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills
> **Authors:** Jianlyu Chen, Yuyang Hu, Hongjin Qian, Jiawei Liu, Wenqing Wei,
> Xiaolong Chen, Defu Lian, Zhicheng Dou, Chaozhuo Li, Qiwei Ye, Zheng Liu
> **Year:** 2026 · **arXiv:** [2609.02749](https://arxiv.org/abs/2609.02749)

## The one-paragraph version

Autonomous ML-research agents are missing a third organ. Everyone builds
them as (model, harness): the model is broad-but-fixed, the harness
controls procedure but carries no domain content. The missing layer is
**operational knowledge** — which package implements your method, how it
wants its data laid out, which configs are sane, which pitfalls silently
invalidate a run. The paper's system, **DisCo**, manufactures that layer
automatically: it distills declarative sources (1,000 popular ML repos,
153 papers) into **verified Agent Skills** — `SKILL.md` +
`references/` + `scripts/` graphs with provenance stamps — behind a
router the agent walks via progressive disclosure. With the GPT-5.5
backbone, Codex harness, and run budget all held fixed, adding only this
knowledge layer takes MLE-bench from 31.11 → 72.89 Any-Medal (+134.3%),
PaperBench +34.4%, FrontierCS +9.2%, PassNet +14.0%. The loudest single
fact: vanilla Codex + distilled skills *beats the best custom-harness
agent on the public leaderboard* (72.89 vs 64.44) without touching the
agent at all.

## The problem

A research agent that doesn't know a library rediscovers it by trial and
error — inside the same budget it needs for the actual experiments. A
misconfigured run can eat half of E before producing one meaningful
measurement, and whatever the agent does learn evaporates when the task
ends. The knowledge to prevent this exists — in repos, docs, papers — but
it's written for humans and far too big to stuff into context. And the
manual alternative (experts hand-writing skills) doesn't scale past one
domain/stack/release. So: produce operational knowledge automatically, at
scale, from declarative sources, in a form the agent can *command*
(discoverable, loadable, verified).

## The idea

Make the knowledge a **supply chain**, not a prompt. Four stages, Eq. 8:
anchor → **scope** (which capabilities matter) → **ground** (what evidence
supports them) → **construct** (package as a three-layer skill graph) →
**verify** (checks pass or the skill doesn't ship; gaps get *recorded*,
not hidden). Run it in two forms: **task-agnostic** (anchor = a repo,
built ahead of time, reusable forever — scaled to 1,000 repos → the
AREX-Skill Library, 5,353 skills behind a 20-area/178-family router) and
**task-oriented** (anchor = the task at hand; discover sources for the
capability gaps). Then hand the agent `A_res = (Mθ, H, K)` where K is just
... more context, conditioned on. No new control loop. That's the whole
trick, and the matched-budget benchmarks are built to prove nothing else
changed.

The paper's best engineering detail is that **verification is what
separates distillation from summarization**. Every graph gets
assertion-backed usability cases, safe native checks (tests, CLI probes,
tiny fixtures) classified into pass / skill_gap / native_fail / skip_*,
localized repair loops, and static gates (link integrity, self-containment,
provenance, no local-path leakage). My favorite anecdote in the paper:
a PassNet skill said "never reimplement vendor-optimized heavy ops"; the
agent met a conv3d with stride==kernel_size (reducible to elementwise),
recognized the win, and *abandoned it because the warning was phrased too
absolutely*. The fix wasn't more knowledge — it was re-expressing the rule
as a defeasible prior. Skills need escape hatches, not commandments.

## How it works (the intuition)

Think of the agent's failure budget as conserved but *redirectable*.
Without K, budget goes to rediscovering APIs and misconfigured runs. With
K, the trial-and-error tax is prepaid — during creation mode, once per
source, amortized over every future task. The asymmetry is the economics:
creator mode costs ~$40/repo once; researcher mode pays only for the
skills a task actually opens (progressive disclosure means holding 5,000+
skills costs nothing until you read one). That's why gains *grow* with
task difficulty (MLE-bench High tier: 13.33 → 62.22): harder tasks have
more places to waste budget, so prepaid knowledge compounds. And it's why
the biggest PaperBench gains are on the *worst* baselines (ftrl 1.50 →
17.17, an 11× — when you're lost, a map is worth more than when you
already know the city).

The two skill constituents map exactly onto the file layout:
`scripts/`+`references/` are **capability** (invocable units, evidence),
`SKILL.md` is **usage policy** (when, why, in what order). Capability
without policy is tools without selection criteria; policy without
capability is advice without an interface.

## What I learned by implementing it

My re-implementation (`implementation/`, stdlib-only, CPU, $0, ~0.3 s)
replaces the paper's $40-per-repo GPT-5.5 creator with deterministic
extraction: evidence boundary → README/AST/packaging parsing with
file:line provenance → trigger-rule capability scoping → templated
three-layer skill graph → 7 static gates + a QA carry test. The anchor:
karpathy/minGPT. Things that only became real once I wrote the code:

- **The evidence boundary is doing more work than it looks like.** Before
  anything is parsed, deciding what is *admissible* (source, docs,
  examples, tests — not caches, build junk, vendored deps) is what makes
  the output "tied to public, reusable package behavior rather than
  arbitrary checkout state". Skip it and the skill inherits the repo's
  noise. It's the least glamorous stage and quietly the most important.
- **Verification found real bugs in my distiller — exactly as designed.**
  My first run's QA test failed 0/10 *even open-book*, which exposed a
  broken retrieval metric. Fixing it exposed a genuine distillation gap:
  my AST miner skipped `__init__` as "private", so the single most
  operationally important signature — `Trainer(config, model,
  train_dataset)` — was missing from the skill. The paper's
  skill_gap → repair → rerun loop isn't ceremony; it's load-bearing. Two
  more QA misses drove two more repairs (README prose sections missing
  from `references/quickstart.md`; the "**note (Jan 2023)**: semi-archived,
  see nanoGPT" caveat not reaching the pitfalls section).
- **Separating "present" from "retrieved" is the honest way to score a
  skill.** I report both: gold-keyword recall over the whole skill corpus
  (is the fact carried at all?) vs. recall in the top-3 retrieved passages
  (can a simple retriever find it?). Final run: **present-carry 10/10
  closed-book** (the deterministic skill loses zero tested facts vs the
  whole repo) but **retrieved 6/10** — and amusingly, closed-book
  retrieval *beats* open-book (5/10): the distilled skill is more
  searchable than the repo it came from, because distillation is
  reorganization, not compression. All four retrieval misses are facts a
  human would find by scanning `references/api.md`.
- **Provenance stamps are nearly free and change behavior.** Stamping
  every claim with `file:line` and the commit hash cost ~20 lines. It's
  what turns the artifact from "plausible text" into "auditable claim
  set" — and it's what my link-integrity and leakage gates check.

## What surprised me / was harder than expected

- **The headline library is not the measured artifact.** The four
  benchmark results come from *task-oriented* and *paper-derived* pools
  built per benchmark under task constraints; the 1,000-repo AREX-Skill
  Library is an exhibited byproduct of the same procedure, not the
  treatment. The paper is honest about this if you read Appendix A.2
  carefully, but the abstract's framing invites conflation.
- **The construction budget is the quiet confound.** MLE-bench skill
  construction gets ≤24 GPU-hours *plus web search* per task, before the
  matched-budget comparison even starts. The no-skill baseline never gets
  that knowledge-acquisition budget at run time. The gains are real, but
  "distilled skills help" and "an agent that researched the task for 24
  GPU-hours helps" are entangled in a way the paper doesn't fully split.
- **How little intelligence the skeleton needs.** With the four-stage
  shape, provenance, verification gates, and the record-with-gaps
  convention all in place, a *zero-LLM* distiller already carries 10/10
  tested facts for a small, well-documented repo. The expensive part of
  the paper's pipeline isn't the scaffolding — it's the judgment layers
  (what's admissible evidence, what procedure the docs *mean*, what a
  check should assert) on messy repos.
- Two PaperBench tasks *regressed* under skills (−5.07, −4.52), both with
  above-average baselines — retrieval distraction beats the agent's own
  good instinct. Consistent with the demystifying-skills paper
  (2608.14036, this repo): retrieval is a soft bottleneck, procedural
  anchoring is the mechanism. The two papers snap together cleanly.

## The distilled skill, honestly assessed

The generated artifact (`implementation/output/skills/mingpt/SKILL.md`,
12 nodes / 18 links) is genuinely useful *and* visibly mechanical.
Useful: install steps quote the real README commands; the API surface
with `GPT.from_pretrained`, `Trainer.__init__(config, model,
train_dataset)`, `GPT.generate(idx, max_new_tokens, temperature=1.0,
do_sample=False, top_k=None)` each carry file:line provenance; the
pitfalls section carries the repo's own semi-archived/nanoGPT caveat.
Mechanical: Step bullets are extracted docstring fragments, not written
procedures; "Use the command-line entry points" lists `runnable_module`
markers instead of saying "run `python projects/chargpt/chargpt.py
--input-datafile shakespeare.txt`"; nothing synthesizes *why* you'd
choose minGPT's `configure_optimizers` weight-decay split.

**What a real LLM distiller (the paper's GPT-5.5 creator) would add:**
(1) *synthesis* — turning extracted fragments into instructions ("to
train on your own text, subclass CharDataset, then Trainer(TrainerConfig,
model, dataset).train()"), including procedures no single file states;
(2) *selection judgment* — which of 88 evidence units matter (my Steps
show 6 per capability regardless of importance); (3) *failure-mode
mining* — the paper pulls pitfalls from issues and test behavior, mine
greps for "note/warning"; (4) *defeasible phrasing* — rules with
exceptions, the conv3d lesson; (5) *dynamic verification* — actually
running the repo's tests/CLI as admission checks instead of static
probes. What it would NOT need to add: the skeleton, the provenance, the
gates, the record. The deterministic floor is higher than I expected,
and the ceiling is exactly the judgment gap.

## Video angle

Three bullets for OpenMontage — the natural sequel to "Why Your AI Agent
Relearns the Same Lesson":

- **The cure for relearning is a supply chain, not a bigger brain.**
  Last video: your agent re-learns the same lesson because knowledge
  dies with the session. This paper ships the fix at industrial scale —
  1,000 repos distilled into 5,000+ verified skills, and the *same*
  unchanged agent jumps 31 → 73 on MLE-bench, beating every
  custom-harness rival. The hook: "they didn't upgrade the agent; they
  upgraded what it knows before it starts."
- **Distillation ≠ summarization — verification is the product.** The
  demo beat: run my $0 deterministic distiller live on a small repo
  (0.3 s), show the SKILL.md appearing with provenance stamps, then show
  the verification gate catching a missing constructor and the repair
  loop fixing it — the same loop the paper runs 5,353 times with
  GPT-5.5. Punchline: a skill without checks is a blog post; the paper's
  moat is that nothing ships unverified.
- **Rules need escape hatches (the conv3d story).** The most teachable
  moment in the paper: the agent *saw* a winning optimization, walked
  away because the skill said "never" instead of "usually not, unless
  stride==kernel_size". Perfect bridge to our skills-ecosystem content:
  write skills as defeasible priors, and pair every "always/never" with
  the evidence condition that licenses an exception.

## References

- Paper: https://arxiv.org/abs/2609.02749
- Official library/code: https://github.com/VectorSpaceLab/AREX-Skill
- My implementation: `implementation/` (run `python3 train.py`, ~0.3 s, stdlib only)
- Generated skill artifact: `implementation/output/skills/mingpt/`
- Breakdown: `breakdown.md`
- First/second-pass notes: `notes.md`
- Companion paper in this repo: `papers/2026-demystifying-agent-skills-why-they-work/` (why skills work — procedural anchoring)
