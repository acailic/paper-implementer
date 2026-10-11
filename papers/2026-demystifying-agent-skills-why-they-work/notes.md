# Notes — Demystifying Agent Skills: Why They Work—Until They Don't

**Paper**: Demystifying Agent Skills: Why They Work—Until They Don't
**Authors**: Zhiyuan Jiang, Fangrui Huang, Hanwen Xing, Xander Wu, Yipeng Gao, Rui Cao, Mengdi Wang, Shilong Liu, Yijiang Li (Princeton, UCSD, Stanford, USC, JHU)
**arXiv**: 2608.14036 (cs.AI, Aug 2026)
**Code**: https://github.com/zhiyuanjiang04/demystify-agent-skills

---

## First pass — big picture (2026-10-11)

### The problem, in my own words

LLM agents now routinely package prior experience into "skills" (Anthropic-style
SKILL.md files: compact, standardized descriptions of what to do, what to check,
what pitfalls to avoid). Everyone evaluates skills by one number: did the
skill-augmented agent solve more tasks? Nobody asks *what actually changes* in
the agent's behavior when a skill is loaded, *which* parts of execution get
stabilized, and *why the same skill* helps one task and harms another. This
paper is a mechanistic "why" study: when do skills help, why do they work, and
where do they fail — analyzed along the whole skill-use pipeline (representation
→ transfer → retrieval → invocation), not just end-to-end success.

### Core claims (from abstract + intro + findings)

1. **Skills work as procedural anchors, not knowledge injections.** In their
   contrastive taxonomy, `procedural_anchor` accounts for **65.7%** of skill
   mechanism labels vs only **4.5%** for explicit `knowledge_injection`. Skills
   stabilize *action* (setup steps, tool sequences, checks, pitfalls) rather
   than supply missing *facts*.
2. **Representation matters with experience held fixed.** Same source
   trajectories, two representations: distilled SKILL.md vs raw-ish Workflow
   Memory. Skill beats Workflow Memory by **+6.06 points** (95% bootstrap CI
   [+0.76, +11.36]) in matched comparisons. Oracle-status success: 61.9% (skill)
   vs 59.1% (raw) vs 55.9% (workflow memory) — workflow memory can be *worse
   than nothing* because it carries failed branches, verbose exploration, and
   process noise (timeouts: 10.6% of workflow cases vs 1.7% raw).
3. **Retrieval is a separate bottleneck.** As skill pools grow 5 → 100,
   execution-time (Arm 3) *actual-use precision* collapses (29.6% → 3.3%
   averaged) while *downstream success stays flat* (~36–39%). Offline
   identification (embedding ranking 88.3%→76.9%; agent selection 70.0%→63.7%)
   degrades much more gracefully — but *similar* distractors crush it
   (70.5%→53.4% embedding precision; vs 84–98% for random/dissimilar pools).
   So: exact ground-truth skill invocation is **neither sufficient nor
   necessary** for success.
4. **Skills have their own failure surface.** `skill_guidance_misapplied_or_ignored`
   = 10.0% of skill-arm cases vs 0.8% raw / 0.4% workflow. New SC3 category
   (invocation/budget/boundary failures) rises to 78/528 with skills vs 19/528
   raw. Skills reduce execution-layer failures (SC2: env-infra failure 5.3% →
   0.2%; output-format mismatch 7.4% → 3.2%) but don't fix algorithmic-logic
   errors (8.3% raw, 11.0% workflow, 7.4% skill — roughly flat).
5. **Outcome annotations matter when traces are mixed.** No-hint skill creation
   (success/failure labels removed) is fine on success-only pools but degrades
   sharply once failed trajectories enter the pool (e.g., Gemini TB2 3s2f:
   0.7462 normal vs 0.4000 no-hint).

### Method skeleton (what they actually did)

- **RQ1/RQ2 (representation + outcome labels)**: for each task, collect
  successful + failed trajectories, build a fixed-budget *composition grid*
  (5s0f, 4s1f, ..., 0s5f = mixture of source trajectories). From the *same*
  selected traces build either a Workflow Memory or a distilled SKILL.md;
  evaluate on matched target tasks under identical protocol. Three arms: Raw /
  Workflow Memory / Skill. Two agent-model pairings: Codex + GPT-5.3-Codex,
  Gemini CLI + Gemini-3.1-Pro-Preview. Benchmarks: Terminal-Bench-2,
  SkillsBench, Terminal-Bench-Pro. n=5 trials per task, Docker-isolated,
  test-based verification.
- **RQ3 (cross-framework transfer)**: artifacts built from Codex trajectories,
  evaluated inside Gemini CLI. Skill transfers better than workflow memory
  (skill > workflow at every mixture; e.g., 76 vs 62 at 5s0f vs raw 56).
- **RQ4 (retrieval)**: 3 *independent* experiments on SkillsBench: Arm 1 =
  embedding ranking (Qwen3-Embedding-0.6B, no execution); Arm 2 = agent
  explicitly selects skills (no execution); Arm 3 = full-pool real execution
  with post-hoc parsing of which skills were actually opened/used. Pools of
  size k ∈ {5,10,20,50,100} contain ground-truth skill + (k−1) distractors
  (random / semantically similar / dissimilar). Outputs are NOT passed between
  arms.
- **Contrastive taxonomy**: normalize 8,135 trial records (7,837 with
  transcripts) into a shared manifest; open-code 240 sampled trajectories →
  238 valid unique labels → 12 canonical modes grouped in 3 categories:
  - **SC1** guided/autonomous success
  - **SC2** execution & verification failures (env infra, format/schema,
    service lifecycle, shell corruption, algorithmic logic, static verify w/o
    runtime)
  - **SC3** invocation & budget failures (timeout/budget, guidance misapplied,
    capability/safety limit)
  Human validation: 95.8% exact agreement, Cohen's κ = 0.952. 528 paired
  triples (SkillsBench 144, TB2 186, TB-Pro 198) → 1,584 arm-level labels.
- **Mechanism labels** (how the injected artifact acts): `procedural_anchor`,
  `knowledge_injection`, `failure_warning`, `none`, `counterproductive`.
- **Controls**: lightweight-baseline check (instruction-derived short plan
  47.7%, workflow-derived test-first template 59.2% — both below Workflow
  Memory 62.3% and Skill 79.2% on 26 TB2 tasks), so it's not "any compact
  procedural hint" that explains the effect.

### My first impressions

- This is an *empirical science* paper, not a new-method paper: no new
  architecture, no training. The contribution is a controlled protocol + a
  validated taxonomy + causal-ish attributions (composition grid is a nice
  idea: hold evidence fixed, vary representation). Implementation for us will
  mean re-implementing the *analysis pipeline* (paired arms, composition grid,
  precision/success-vs-pool-size curves, a 12-mode classifier over synthetic
  trajectories), not a neural net.
- The headline asymmetry is memorable: **offline retrieval looks fine, online
  use-precision collapses, task success doesn't care**. That's the "exact
  invocation is neither necessary nor sufficient" punchline.
- Skills trade SC2 failures for SC3 failures — a lifecycle view, not a free
  lunch.

### Terms / things I don't fully understand yet (→ second pass)

- Exact definition of "Workflow Memory" arm: how are traces *cleaned* before
  injection? (Appendix A needed.)
- How is a SKILL.md actually *placed* in the environment ("reusable procedural
  resources rather than fully incorporated into the initial context") — file in
  workspace? Which prompt tells the agent skills exist?
- Arm 3 "actual-use precision": how exactly do they parse which skills were
  "accessed" (file reads? cat commands?) and how overlap with ground truth is
  computed.
- Oracle-status success rate vs verifier success rate — precise definition.
- The 12 modes in full (Table 11 in appendix) — I've only seen ~11 named so far.
- Mechanism-label assignment procedure: who labels which arm (LLM judge on
  triples?) and how "pairwise changes between arms" are used.
- Statistics: how the +6.06 CI bootstrap is set up (resampling over tasks?
  triples?), and the "effectiveness–efficiency trade-off" (A.9).
- Terminal-Bench-Pro "130 trials per condition" — why 130.

### Artifacts inventory

- paper.pdf / paper.txt (extracted, 1914 lines, includes appendices A.1–A.9,
  Tables 1–16 references visible)
- templates copied: breakdown.md, implementation.md, writeup.md
