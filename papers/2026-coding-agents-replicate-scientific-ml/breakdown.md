# Coding-agents can replicate scientific machine learning papers — source-first breakdown

- **arXiv:** 2607.02134v1 [cs.AI], 2 Jul 2026
- **Authors:** Atharva Hans, Ilias Bilionis — School of Mechanical Engineering, Purdue University
- **Code/data:** https://github.com/PredictiveScienceLab/paper-replication-paper (Codex + Claude Code skills, 12 agent-generated case-study workspaces, analysis scripts)
- **Source files:** `paper.pdf` (379 KB, **16 pp — `pdfinfo` 16 pp BUT `file` misreports 5 pp [defect recurs iters 66/67/69/70/71/72/73/75/78/79; intermittent no-defect iters 68/74/76/77/80; 11-page gap this iter; trust pdfinfo]**), `paper_layout.txt` (`pdftotext -layout`, 1035 lines, **2 explicit tables + Eqs 1–7 + Figs 1–6**)

## Thesis (one sentence)
A coding-agent **skill** called **Paper-replication** turns paper replication into a **target-level evidence contract** in a **persistent workspace** with **external validation checks**, so that *completion is a workspace state* (every recorded target matched + report PDF exists + checks pass) rather than the agent's final-message claim — evaluated on 12 independent runs (3 each × 4 scientific-ML papers: PIFT, PINN-I, PINN-II, SINDy), all 12 reaching the completion gate with all 158 recorded targets matched.

## Why this paper is meta to the repo
This is the repo's **first coding-agent / harness-engineering / paper-replication / scientific-reproducibility paper** (sibling-in-spirit to `the-verification-horizon`, `are-we-ready-for-agent`, `scorio-ranking-reasoning-llms` evaluation-fidelity lineage, and methodologically adjacent to `verification-horizon`'s "no silver bullet" theme). It is literally a system that does what this repo does — read a paper, reconstruct the method, regenerate results, write a report — but formalized as an agent skill with an evidence gate. The repo's own source-first breakdown + Python reconciliation workflow is a hand-rolled analogue of Paper-replication's reproduction-matrix + comparison-check.

## Method

### 2.1 Targets and evidence (L125–176), Eqs 1–2
- Paper materials `P` (LaTeX source tree, figures/tables, bibliography, appendices, datasets). Agent inspects `P`, records each computational claim as an element of a finite **target set** `T = {t_j}_{j=1}^J`. A target = a result/claim (figure, table, reported scalar, learned field, distribution, trajectory, discovered equation, structural statement).
- **Eq 1 (L152):** `b_y_j = F_j(D_j; θ_j, ω_j)` — candidate result from the agent's reconstructed implementation `F_j`, input data `D_j`, recorded config `θ_j`, seed/stochastic state `ω_j`. `y*_j` = corresponding paper-reported quantity (text value, table entry, distributional property, or figure feature).
- **Eq 2 (L164):** evidence bundle `E_j = (b_y_j, R_j, P_j, C_j, G_j)` — `R_j` execution record, `P_j` provenance (code/config/seed/paper passages), `C_j` comparison of `b_y_j` vs `y*_j` under the target's acceptance rule, `G_j` report coverage. A target is matched **only when every required part exists AND external checks accept it** — an output artifact alone never counts as evidence.

### 2.2 Persistent workspace (L233–294)
- **Manifest** (paper source, hash, run rules, compute env, rerun instructions) → resumable after interruption.
- **Source inventory** (`spec/paper_inventory.json`): included TeX files, bibliography, appendices, figure assets, data refs; hashes of paper source + rendered pages → later checks detect copied paper material.
- **Reproduction matrix** + **task ledger** (one active target) keep the agent on one result at a time; unfinished work visible.
- **Specification files**: target defs, agent's restatement of equations/algorithms, implementation plan, assumptions, missing-detail hypotheses (turns guesses into recorded hypotheses).
- **Run recorder** (`scripts/paper_replication.py`, `artifacts/runs/`): command, working dir, start/finish, success flag, messages/errors, expected outputs, output hashes. Failed/superseded runs retained (trial-and-correction path preserved).
- **Provenance** (`artifacts/provenance/`): links output → implementation file, config, seed, method components, paper passages; stores hashes of output/impl/config so later edits can't silently change the claim.
- Separation: code/config separate from generated outputs; **paper-provided assets separate from both** (`artifacts/paper_figures/` vs `artifacts/figures/`, `artifacts/tables/`).

### 2.3 Validation and completion (L295–368), Eq 3
- **Acceptance rule** recorded BEFORE judging (prevents post-hoc rule-changing). Four target types:
  - **Numeric** — discrepancy metric + units + tolerance; accept iff `|b_y_j − y*_j| ≤ tolerance`.
  - **Distributional** — compare distribution properties (mean/variance/quantiles/coverage/#modes/distance), not identical samples (posterior, empirical dist, uncertainty bands).
  - **Structural** — pattern/relationship (sparsity, ordering, stability, regime transition, phase-portrait, attractor, selective identifiability).
  - **Visual** — appearance is part of the claim; record reference + candidate + visual comparison; used only when the claim requires it.
- External checks reject: visual-only evidence for numeric/distributional targets; missing visual comparison for visual targets; structural comparisons that don't explain agreement/disagreement.
- **Hash checks** compare generated outputs vs indexed paper assets + rendered pages → detect direct reuse. (Don't catch every transformed copy; in these runs none observed.)
- **Method-replication requirement**: provenance must show how the run implements the paper's method (implemented components + justifying paper passages + code/config/seed) — prevents a substitute method passing on output similarity alone.
- **Eq 3 (L343–352), the completion gate:**
  `V_complete = V_spec ∧ V_progress ∧ V_report ∧ (∧_{j=1}^J [s_j = MATCHED]) ∧ [a = ∅] ∧ [report PDF exists]`
  where `V_spec/V_progress/V_report` = specification/progress/report-coverage checks, `s_j` target status, `a` active-target field (`∅` = none active). Completion is a **workspace state**, not a final-message claim.

### 2.4 Skill implementation (L311–412)
Two layers: (a) **persistent agent instructions** (`SKILL.md` + per-agent Codex/Claude Code prompt files); (b) **workspace utilities** (`scripts/paper_replication.py`) that create + check the files making the rules enforceable. Instruction files define rules; utilities create/check files. `references/` defines workspace contract, author-code rule, compute env, acceptance-rule types — no paper-specific science.

### 2.5 Case-study analysis (L413–535), Eqs 4–7
- 4 papers × 3 independent runs (`N=4, R=3`); workspaces share no code/outputs/state. Agent = **Codex with GPT-5.4 at Extra High reasoning** (MacBook Pro M4 Max 128 GB + Purdue Gautschi cluster: CPU 192-core/384 GB nodes, GPU 8×H100/112-core/~1 TB nodes).
- Initial prompt + fixed queue of **10 follow-up prompts** (resume-until-complete; completion stays workspace-defined).
- **Target coverage** `Q_pr` (targets in final matrix) — Gamma-Poisson model **Eq 4 (L430–436)**: `Q_pr | λ_p, ϕ ~ GammaPoisson(ϕ, ϕ/λ_p)`, `log λ_p = µ_Q + a_p`, hierarchical priors. **Decomposition ratio** = max/min final target count across a paper's 3 runs.
- **Paper-anchored numeric fidelity** — for scalar claims, fixed threshold `τ_pc` from the source paper's accuracy scale (Table 1); headroom `h_pcr = log10(τ_pc / d'_pc)` where `d'_pc = max(d_pcr, ε_pc)` (**Eq 5, L487**); positive = inside threshold. Headroom model **Eq 6 (L456–461)**: `h_pcr ~ Normal(µ + b_p + g_pc, σ²_{ε,p})` with paper-level `b_p`, claim-level `g_pc`, run-residual `σ_{ε,p}`.
- **Effort** = elapsed replication time `H_pr` (initial prompt → first completion-after-gate); log-normal **Eq 7 (L499–505)**: `log10 H_pr ~ Normal(µ_H + u_p, σ²_H)`.
- **Superseded tracked execution** = recorded command later replaced by correction work before final evidence accepted.
- **Judgment variation** = fraction of aligned claims where every run records the same acceptance-rule type.
- Bayesian fit: NumPyro NUTS, 4 chains × 2000 retained draws.

---

## Results tables (verbatim, with sourcing line-ranges)

### Table 1 — Paper-anchored analysis rules (L427–449)
| Paper | Claim family | Paper-anchored analysis rule | Analysis type |
|---|---|---|---|
| PINN-I | solution relative L2 error | `d_pcr ≤ 10⁻²` | scalar |
| PINN-II | coefficient percentage error | `d_pcr ≤ 10%` | scalar |
| SINDy | Lorenz coefficient relative error | `d_pcr ≤ 10⁻³` | scalar |
| SINDy | sparse-support recovery and trajectory geometry | exact support OR structural agreement (depending on target) | structural |
| PIFT | posterior collapse, bimodality, selective identifiability | distributional OR structural evidence (depending on target) | non-scalar |

### Table 2 — Twelve case-study runs (L496–505)
| Paper | Runs | Targets/run | Matched | Elapsed time (h) | Superseded exec |
|---|---|---|---|---|---|
| PIFT | 3 | 8, 8, 25 | all | 2.2 [1.1, 4.4] | 3 |
| PINN-I | 3 | 8, 8, 8 | all | 5.0 [2.5, 9.9] | 11 |
| PINN-II | 3 | 9, 9, 15 | all | 6.9 [3.0, 13.4] | 10 |
| SINDy | 3 | 20, 20, 20 | all | 1.9 [1.0, 4.3] | 1 |

(Elapsed = posterior median [95% CI] from Eq 7 effort model.)

### Key prose numbers (verbatim)
- **158 targets recorded, all MATCHED, all with report coverage; 12/12 workspaces pass completion gate** (abstract L21–23, §3 L514–518).
- Decomposition ratios (§3.1, L537): **PIFT 3.1, PINN-II 1.7, PINN-I 1.0, SINDy 1.0**. PIFT expected target count posterior median **13.4 [8.8, 20.5]** (L540).
- **13 scalar anchors** (4 PINN-I + 8 PINN-II + 1 SINDy) → **39 anchor-run observations**; **37 inside** the fixed paper-anchored threshold, **2 outside** (§3.2, L597–610): **Schrödinger run 3 relative L2 error 4.8×10⁻² vs 1% threshold**; **Navier–Stokes λ2 run 1 coefficient error 16.4% vs 10% threshold**. Both still MATCHED in-workspace.
- Headroom posterior medians (§3.2, L629): **PINN-I 0.51, PINN-II 1.75, SINDy 0.42** → reproduced discrepancies sit **~3.2×, 57×, 2.6× inside** thresholds on average. Posterior predictive P(inside) **0.79, 0.90, 0.73** (L633).
- Run-residual scale `σ_{ε,p}` (§3.2, L687–693): **PINN-II 1.16 [0.86, 1.66] ≈ factor of 14**; PINN-I 0.41 [0.26, 0.76]; SINDy 0.31 [0.11, 1.60] (wide — one anchor).
- Burgers λ2 coefficient error across 3 runs: **7.3%, 0.14%, 0.014%** (all < 10%) (L620–625).
- Elapsed replication time **ranges 1.2 to 13.0 hours** (§3.3, L702); posterior medians PINN-II 6.9 h, PINN-I 5.0 h, PIFT 2.2 h, SINDy 1.9 h. PINN-paper-longer posterior prob **0.947 to 0.972** across 4 comparisons; repeated runs differ by **~factor of 2** (L748).
- **25 superseded tracked executions** corpus-wide; **21 in the two PINN papers** (11 PINN-I + 10 PINN-II), PIFT 3, SINDy 1 (L755–767). PINN-I run 1 = **71 tracked executions (11 superseded)**, runs 2&3 = **12 each (0 superseded)** (L768–770).
- Judgment variation same-rule fractions (§3.4, L785–787): **SINDy 19/20 = 0.95, PIFT 8/11 = 0.73, PINN-I 4/8 = 0.50, PINN-II 5/11 = 0.46**.

---

## Source-free reconciliation (Python-verified)

- **158-target total:** PIFT 8+8+25=41, PINN-I 8+8+8=24, PINN-II 9+9+15=33, SINDy 20+20+20=60 → **41+24+33+60 = 158 ✓ EXACT.**
- **Decomposition ratios (max/min):** PIFT 25/8=3.125→**3.1 ✓**, PINN-II 15/9=1.667→**1.7 ✓**, PINN-I 8/8=**1.0 ✓**, SINDy 20/20=**1.0 ✓**.
- **13 anchors = 4+8+1 ✓; 39 obs = 13×3 ✓; 37/39 inside ⇒ 2 outside ✓.** Per-paper empirical in-rates: PINN-I 11/12=0.917, PINN-II 23/24=0.958, SINDy 3/3=1.0.
- **Headroom → multiplier (10^median):** PINN-I 10^0.51=**3.24→"3.2×" ✓**; PINN-II 10^1.75=**56.2→"57×" ✓** (round); SINDy 10^0.42=**2.63→"2.6×" ✓**.
- **Run-residual PINN-II 10^1.16=14.5→"factor of 14" ✓**; PINN-I 10^0.41=2.57; SINDy 10^0.31=2.04.
- **Judgment fractions:** 19/20=**0.95 ✓**, 8/11=**0.727→0.73 ✓**, 4/8=**0.50 ✓**, 5/11=**0.455→0.46 ✓**.
- **Superseded:** 11+10+3+1=**25 ✓**; 21 of 25 in PINN = 11+10=**21 ✓**.
- **Burgers λ2 span:** 7.3/0.014=**521× = 2.7 orders**, all < 10% ✓.
- Posterior-median elapsed times (6.9/5.0/2.2/1.9) **byte-match Table 2** ✓.

**Every quoted integer/ratio/multiplier/fraction recomputes EXACT or within rounding. No numeric prose-vs-table contradiction.**

---

## ⚠ Honest-scope flags (inline)

1. **NO ablation without the skill (the load-bearing caveat; authors' own limitation, L828–830).** The paper "does not include an ablation without the skill, so it characterizes what Paper-replication **produces** rather than estimating the workflow's **effect** on an unstructured prompt." The headline 12/12-completion + 158/158-matched is therefore **uncontrolled**: with no prompt-only baseline, you cannot attribute completion to the Paper-replication skill vs raw Codex/GPT-5.4 capability. The paper shows the skill *can* reach completion, not that it is *necessary* (or even *better*) for it. Cite the completion numbers as "under the skill", never as "the skill causes completion".

2. **"All 158 matched" conflates two different standards (parallel iter-72 MARVEL selective-baseline / iter-80 Zeus TSFM-subset-scoping).** MATCHED = the **agent's own recorded acceptance rule** passed the external checks. The **paper-anchored scalar fidelity** analysis is explicitly *separate* (§3.2 L626–629), and there **2/39 scalar anchor-runs fall OUTSIDE the paper's reported accuracy class** — Schrödinger run 3 (4.8×10⁻² vs 1%) and Navier–Stokes λ2 run 1 (16.4% vs 10%) — yet both still count as MATCHED. So "158/158 matched" overstates absolute fidelity: it means "agent-judged sufficient under a self-recorded rule", and ≥2 reproductions are outside the source paper's accuracy class. The two standards are honestly separated in the paper, but the abstract/completion headline uses only the weaker one.

3. **Completion is relative to each run's self-recorded target set (decomposition-confound).** PIFT decomposition ratio = **3.1** (one run 25 targets, two runs 8 each). A run that records fewer/easier targets faces an easier completion gate (Eq 3 only requires *its recorded* targets be matched). So 12/12 completion is partly **tautological** — each run defines its own pass standard — and cross-run completion-rate comparisons are confounded by decomposition granularity (PIFT run with 8 targets ≠ PIFT run with 25). The honest framing "completion relative to the recorded target set" (L840–851) is in the paper but not in the abstract.

4. **Single agent / model / reasoning setting.** Eval is **Codex + GPT-5.4 + Extra High only** (L543). The skill ships for **both Codex and Claude Code**, but **no Claude Code run** and no second model is evaluated. Generalization across agents/models/reasoning levels is untested (authors' own limitation, L828).

5. **4 papers × 3 runs = 12 runs; corpus too small for between-paper claims.** Authors concede the corpus "remains too small to draw strong conclusions about between-paper variation" (L833–835) and that the effort results "show repeated optimization and correction work in these workspaces, **not that the PINN papers are intrinsically harder to replicate**" (L824–825) — good honest framing, but it means the per-paper effort/fidelity medians are descriptive, not difficulty-ranked.

6. **Paper-anchored fidelity rests on 13 scalar anchors; PIFT has NO scalar anchor.** 1 of 4 papers (PIFT) is evaluated **distributionally/structurally only** — no scalar fidelity number contributes (L634, L683). The headroom model is built on 13 anchors / 39 obs, with SINDy contributing **a single anchor** (3 obs) → its wide credible interval [0.11, 1.60] and low posterior-predictive 0.73 reflect paucity, not poor fidelity.

7. **Posterior-predictive P(inside) (0.79/0.90/0.73) reads more pessimistic than the empirical in-rates (0.92/0.96/1.0).** The model's predictive is conservative (rightly accounts for single-anchor uncertainty), but citing "0.73–0.90 probability another run is inside-threshold" understates the observed **37/39 = 0.949** in-rate. Not a contradiction — a prior-vs-likelihood shrinkage artifact — but the two numbers diverge enough to choose carefully which to quote.

8. **Reproduced values move orders of magnitude yet stay MATCHED (lenient-standard surface).** Burgers λ2 coefficient error = **7.3% / 0.14% / 0.014%** across 3 runs = **521× spread (2.7 orders)**, all MATCHED. Exact numerical equality is *deliberately* abandoned (L610–625) — defensible for stochastic sci-ML, but it means MATCHED is a **weak/lenient** standard: a 500× discrepancy in a coefficient can still pass. Anyone citing "replicated" should note the per-run reproducibility band, not just the MATCHED label.

9. **Effort variance masked by the completion rate.** PINN-I run 1 = **71 tracked executions (11 superseded)**; runs 2 & 3 = **12 each (0 superseded)** — a **~6× execution-count variance** within one paper under the same prompt/skill. Elapsed time spans **1.2–13.0 h** with **~2× run-to-run** movement. 12/12 completion hides that two sibling runs took vastly different correction paths; the *path*, not just the endpoint, is the scientifically variable quantity (honestly reported in §3.3, absent from the abstract).

10. **Self-judged acceptance rules → MATCHED is not a consistent cross-run standard.** The acceptance-rule **type and tolerance** is *inferred and recorded by the agent itself*. Judgment variation shows the agent classifies the **same aligned claim differently across runs** — PINN-II same-rule agreement only **5/11 = 0.46**, PINN-I **4/8 = 0.50** (§3.4). So two MATCHED targets for the same claim can rest on different rule types (numeric vs structural), making "both matched" a weaker agreement than it sounds.

11. **No shared-benchmark head-to-head.** The 4 case-study papers are **not** a standard benchmark; results are not comparable to PaperBench / CORE-Bench / ReplicatorBench / ReplicationBench / SciReplicate-Bench numbers cited in §1. No system-vs-system comparison on a common task — contribution is the workflow + a 4-paper case study, not a leaderboard result.

12. **Agent-generated, agent-judged workspaces (self-reference surface).** The 12 case-study workspaces were "produced by a Codex coding agent equipped with the paper-replication skill" (Declaration L859–860) — the **same class of agent** whose capability is being evaluated — and judged by external checks that the same authors designed. Authors are human and reviewed outputs, but the reproductions are agent-made and the evidence standard is author-designed; no third-party independent replication of the 12 workspaces.

**No numeric prose-vs-table contradiction. Every quoted integer/ratio/multiplier/fraction recomputes EXACT or within rounding. The honest-scope weight sits on flags 1 (no-skill ablation absent), 2 (matched-vs-fidelity conflation), 3 (decomposition-confounded completion), and 8 (lenient MATCHED standard).**

---

## Strengths
- **Genuinely novel framing**: paper replication as a *target-level evidence contract* in a persistent workspace, with completion as a *workspace state* (Eq 3) rather than an agent's final message — directly attacks the documented prompt-only failure modes (agent stops early, treats copied figures / substitute-method outputs / its own progress description as success).
- **Two-mechanism design is clean and falsifiable**: (a) persistent workspace records (manifest, reproduction matrix, task ledger, spec files, run/provenance records) make state resumable and inspectable; (b) external validation checks (paper-asset hash checks, method-provenance requirement, report-coverage) make evidence enforceable outside the agent.
- **Honest separation of "matched" vs "paper-anchored fidelity"** (§3.2) is methodologically sound — the paper does NOT claim MATCHED = exact reproduction; it explicitly reports the 2/39 scalar outliers and the orders-of-magnitude run-to-run spread.
- **Repeated-run design surfaces real variation** (decomposition ratio, headroom, effort, judgment) instead of hiding it behind a single completion label — the variation analysis is the scientific contribution, not the 12/12 headline.

## Limitations
- **No without-skill ablation** (flag 1) — cannot attribute completion to the skill; the central causal claim is untested.
- **MATCHED is a lenient, self-judged, decomposition-confounded standard** (flags 2, 3, 8, 10) — "158/158 matched" overstates absolute fidelity and cross-run consistency.
- **Single agent/model/setting** (flag 4); **4-paper/12-run corpus too small** for between-paper claims (flag 5); **13 scalar anchors, PIFT scalar-less** (flag 6).
- **No shared-benchmark comparison** to prior replication systems (flag 11); **agent-generated/agent-judged workspaces** (flag 12).

## Verdict
A well-engineered, honestly-reported harness-engineering paper whose real contribution is the **target-level evidence-contract + persistent-workspace + external-check workflow** and the **repeated-run variation analysis** — NOT the 12/12-completion headline (which is uncontrolled, lenient-standard, and decomposition-confounded). Cite the **workflow design (Eq 1–3)** and the **variation findings** (decomposition ratios 3.1/1.7/1.0/1.0; headroom multipliers 3.2×/57×/2.6×; judgment agreement 0.95/0.73/0.50/0.46; effort 1.2–13 h with ~2× run-to-run) as the citable falsifiable content; quote "158/158 matched" only with its explicit "under the agent's recorded acceptance rule" qualifier and the 2/39 scalar-outlier caveat. Repo's first **coding-agent / paper-replication / reproducibility-benchmark** paper; sibling-in-spirit to the `verification-horizon` / `are-we-ready-for-agent` evaluation-fidelity lineage, and methodologically a formalized mirror of this repo's own source-first-breakdown + Python-reconciliation workflow.
