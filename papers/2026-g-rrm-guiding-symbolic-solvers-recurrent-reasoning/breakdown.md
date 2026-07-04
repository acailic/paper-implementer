# G-RRM: Guiding Symbolic Solvers with Recurrent Reasoning Models

**arXiv:** [2607.02491](https://arxiv.org/abs/2607.02491) (v1, 2 Jul 2026, cs.AI)
**Authors:** Timo Bertram, Sidhant Bhavnani, Richard Freinschlag, Erich Kobler, Andreas Mayr, Günter Klambauer — ELLIS Unit Linz / LIT AI Lab / Institute for Machine Learning, JKU Linz; Institute for Symbolal AI, JKU Linz; Clinical Research Institute for Medical AI, JKU Linz.
**Subarea (repo lineage):** **neuro-symbolic search guidance / SAT-solver acceleration** — repo's FIRST paper on symbolic solvers, constraint satisfaction, SAT/CDCL, or Boolean satisfiability. Sibling-in-spirit to the inference-efficiency lineage (jetspec / speculating-experts / spin) but accelerates an EXACT symbolic solver with a NEURAL prior instead of approximating inference, and sibling-in-spirit to demystifying-rl-long-horizon-tool-using-agents / verification-horizon (both "when does X actually help" empirical-study papers with a falsifiable condition). Distinct from expander-SAE / CoAx / refusal-subspaces (which explain/modify weights) — G-RRM supplies a search ordering, never modifies the solver's logic.

---

## TL;DR — what it is

A **symbol-equivariant Recurrent Reasoning Model (SE-RRM)** is trained on (partial-grid, solution) Sudoku pairs. At inference it emits a per-cell score matrix `Ŷ ∈ ℝ^{I×K}` over symbol values; from this the paper derives a per-variable value ordering `π_i = argsort_d Ŷ_{i,d}` (Eq. 1) and uses it to **guide the search of exact symbolic solvers** — branching order for backtracking, and **phase initialization** for CDCL SAT solvers. Guidance never prunes the search space (only reorders exploration), so the symbolic solver's **completeness and correctness are unchanged**: G-RRM converts a high-accuracy but unverified neural solver into one with guaranteed global correctness.

The central scientific question is empirical: **under what conditions does neural guidance actually reduce solver cost?** The paper's answer is a two-condition theory (§1, §4):

> **(C1) Expansive combinatorial search space** — the instance's runtime must be search-dominated (the decision tree is the bottleneck).
> **(C2) Solver can dynamically overwrite its branching choices** — the solver must be able to recover when neural hints are imperfect, abandoning a faulty hint-driven path.

When BOTH hold (backtracking; `glucose4`), guidance gives large, significant speedups. When (C2) fails (`cadical3` strictly honors injected phases) OR (C1) fails (propagation-dominated large grids), guidance gives no speedup or a net slowdown.

---

## The two-condition theory (the citable contribution)

| Solver | Search-dominated? (C1) | Overwrites hints? (C2) | Guidance effect |
|---|---|---|---|
| **backtracking** (custom Python) | ✓ (DFS, no learning) | ✓ (chronological backtrack) | **33.3×** median speedup 9×9 (p<0.001) |
| **glucose4** (CDCL) | ✓ | ✓ (VSIDS-driven reinit on restart overrides external phases) | **1.70×** median 9×9, **1.17×** perfect-hint 25×25 (p<0.001) |
| **cadical3** (CDCL) | ✗ (overhead-dominated) | ✗ (strictly honors external literal phases across restarts) | **1.02×** n.s. median 9×9; **0.90×** mean slowdown 9×9 |

The asymmetry between glucose4 and cadical3 is the cleanest demonstration: both are modern CDCL solvers with conflict-driven clause learning, both receive the same phase hints, but only the one that can self-correct (glucose4) benefits. cadical3's runtime is "largely independent of conflict count" — its internal bookkeeping dominates — so eliminating conflicts does not reduce runtime, and the fixed phase-initialization cost (~1.7 ms on 9×9) shows up as a mean **slowdown**.

---

## Method

### SE-RRM (recap; full detail in Freinschlag et al. 2026 / Appendix A)

Recurrent Reasoning Models (RRMs — HRM, TRM) are looped transformers that iteratively refine a recurrent state `Z_t ∈ ℝ^{D×I}` via a shared block `H(E_H(X), Z_t)` (Eq. 2–3), trained with deep supervision + stop-gradient between loop iterations. SE-RRM adds an explicit **symbol axis**, lifting the state to `Z̄_t ∈ ℝ^{D×I×K}` (rank-3). Its block applies **axial attention** — first over positions (Eq. 8), then over symbols (Eq. 9), then a token-wise MLP (Eq. 10) — which is permutation-equivariant over symbols by construction. This is what enables **extrapolation to larger grid sizes / unseen symbol alphabets** (vanilla RRMs use a learned per-symbol embedding table Σ→ℝ^D and cannot handle unseen symbols).

### G-RRM guidance mechanism

`SE-RRM` produces logits `Ŷ ∈ ℝ^{I×K}` in a single forward pass. Per variable `i`, `π_i = argsort_d Ŷ_{i,d}` (Eq. 1) orders values from lowest to highest score; the top preference is `d*_i = π_i(K)`. This ordering feeds the solver:

- **Backtracking:** cell selection uses the classic minimum-remaining-values heuristic; **digit-exploration order** within a cell is set by `π_i` (most-preferred digit first). §2.3.2.
- **CDCL (glucose4, cadical3):** guidance attaches to **phase selection**. Before search, the Boolean variable `x_{r_i,c_i,d*_i}` is initialized to 1 (and `x_{r_i,c_i,d}=0` for `d≠d*_i`) — the network's high-confidence full-grid proposal is mapped onto the initial polarity of every CNF variable. §2.3.3.

G-RRM does NOT restrict the search space — it only changes the order of exploration, so all feasible assignments remain reachable.

### Sudoku-as-SAT encoding (§2.2.2)

Boolean variables `x_{r,c,v}` = "cell (r,c) holds value v", `r,c,v ∈ {1,…,N}` for an `N×N` grid. Standard structural encoding: **At-Least-One** clause `∨_{v} x_{r,c,v}` per cell, pairwise **At-Most-One** clauses `(¬x_{r,c,v} ∨ ¬x_{r,c,w})` for `v≠w`, applied across every cell/row/column/`√N×√N` box; prefilled cells fixed via unit clauses. Variable count `N³`; base-clause counts: **11,988 (9×9), 123,904 (16×16), 752,500 (25×25)**.

---

## Experimental evaluation (§3)

### Setup

Sudoku benchmarks at three grid sizes. SE-RRM trained on 9×9 then fine-tuned on 100 samples each for 16×16 / 25×25 (Appendix B). **Fully-Solved Rate (FSR)** = fraction of instances SE-RRM predicts correctly in one forward pass. CDCL solvers interfaced via PySAT. backtracking is a custom Python implementation — the authors explicitly note backtracking-vs-CDCL magnitudes are **not** comparable across solver types; only Normal-vs-Guided within a solver is.

| Grid | SE-RRM FSR | empty-cell rate |
|---|---|---|
| 9×9 | **91.1%** | 68.88% |
| 16×16 | 22.0% | 63.49% |
| 25×25 | 51.1% | 50.08% |

⚠ **FSR is non-monotonic in grid size** (9×9 91.1% → 16×16 22.0% → 25×25 51.1% rises again). This tracks the **empty-cell rate**, not size: 25×25 grids are denser (50.08% empty) so easier for the neural model, while 16×16 is sparser (63.49% empty). FSR is a function of difficulty (holes), not of N directly.

---

### Table 1 — Conflict-count percentiles, Normal vs Guided (verbatim, paper_layout.txt L419–452)

Bold marks the best value at each percentile. Backtracking conflicts = DFS dead-ends (not magnitude-comparable to CDCL conflicts).

| Grid | SE-RRM FSR | Solver | Mode | p50 | p75 | p90 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|---|---|
| **9×9** | 91.1% | backtracking | Normal | 2865.0 | 9747.8 | 17932.9 | 24791.5 | 42058.6 | 192185.0 |
| | | backtracking | Guided | **0.0** | **0.0** | **0.0** | 5166.3 | 18457.5 | 555689.0 |
| | | glucose4 | Normal | 19.0 | 39.0 | 62.0 | 75.9 | 116.6 | 201 |
| | | glucose4 | Guided | **0.0** | **0.0** | **0.0** | **30.8** | **73.5** | **188** |
| | | cadical3 | Normal | 13.0 | 31.0 | 53.0 | 72.0 | 102.0 | 155 |
| | | cadical3 | Guided | **0.0** | **0.0** | **0.0** | **27.7** | 80.3 | 174 |
| **16×16** | 22.0% | glucose4 | Normal | 93.0 | 173.8 | 242.6 | **284.1** | 507.9 | **598** |
| | | glucose4 | Guided | **43.5** | **151.3** | **277.0** | 395.8 | 767.1 | 779 |
| | | cadical3 | Normal | **72.5** | **122.8** | **184.1** | **258.1** | **469.3** | **502** |
| | | cadical3 | Guided | 56.5 | 129.8 | 257.3 | 366.5 | 477.1 | 581 |
| **25×25** | 51.1% | glucose4 | Normal | 49.5 | **78.3** | 210.8 | **244.6** | **392.3** | **468** |
| | | glucose4 | Guided | **0.0** | 51.0 | **136.8** | 264.0 | 488.6 | 502 |
| | | cadical3 | Normal | 28.0 | 71.0 | **147.3** | **174.9** | **392.2** | **410** |
| | | cadical3 | Guided | **0.0** | **49.8** | 148.7 | 161.4 | 328.6 | 366 |

^a Standard backtracking is too inefficient to solve puzzles larger than 9×9.
^b The paper uses a different 9×9 puzzle set than Freinschlag et al. (2026), explaining the slightly lower 9×9 FSR.

**Takeaways (Table 1, §3 prose):** guidance most consistently cuts the **median** — on 9×9 it drives p50 (and p90) to zero across all three solvers, reflecting the FSR=91.1% of instances the SE-RRM solves perfectly. On 16×16 it lowers the median (−53.2% glucose4 p50 93.0→43.5; cadical3 p50 72.5→56.5, −22.1%) and on 25×25 it collapses the median to zero for both CDCL solvers. Upper-tail effects (p95/p99) are smaller or mixed. Conflict reduction does NOT always translate to wall-clock speedup because runtime is not always search-dominated — motivating Table 2.

---

### Table 2 — Wall-clock solve time (ms), Normal vs Guided, by hint accuracy (verbatim, paper_layout.txt L470–506)

`Speedup = Normal / Guided` (>1 = guidance faster). Significance: paired per-instance test across the test set — Wilcoxon signed-rank for Median rows, paired t-test for Mean rows. `*** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant`. Seeds collapsed per instance (mean over 3 runs) before aggregation. The Median-row significance is the more reliable read (Mean-row t-test is uninformatitve under heavy-tailed distributions).

| Grid | Solver | Metric | | All Puzzles | | | Perfect (acc=1.0) | | | Imperfect (acc<1.0) | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | Normal | Guided | Speedup (sig) | Normal | Guided | Speedup (sig) | Normal | Guided | Speedup (sig) |
| **9×9** | backtracking | Median | | 20.503 | 0.617 | **33.251× \*\*\*** | 18.230 | 0.606 | **30.101× \*\*\*** | 41.535 | 42.601 | 0.975× n.s. |
| | backtracking | Mean | | 47.479 | 6.133 | **7.741× \*\*\*** | 45.556 | 0.601 | **75.750× \*\*\*** | 67.197 | 62.848 | 1.069× n.s. |
| | glucose4 | Median | | 0.348 | 0.205 | **1.699× \*\*\*** | 0.337 | 0.202 | **1.666× \*\*\*** | 0.451 | 0.450 | 1.003× n.s. |
| | glucose4 | Mean | | 0.419 | 0.245 | **1.710× \*\*\*** | 0.404 | 0.213 | **1.896× \*\*\*** | 0.575 | 0.573 | 1.003× n.s. |
| | cadical3 | Median | | 5.517 | 5.405 | 1.021× n.s. | 5.455 | 5.307 | 1.028× n.s. | 6.074 | 5.876 | 1.034× n.s. |
| | cadical3 | Mean | | 4.262 | 4.759 | **0.896× \*\*\*** | 4.125 | 4.622 | **0.892× \*\*\*** | 5.672 | 6.167 | 0.920× n.s. |
| **16×16** | glucose4 | Median | | 2.671 | 1.998 | 1.337× n.s. | 2.967 | 1.346 | **2.204× \*\*\*** | 2.647 | 2.627 | 1.008× n.s. |
| | glucose4 | Mean | | 3.158 | 2.974 | 1.062× n.s. | 3.245 | 1.369 | **2.371× \*\*\*** | 3.134 | 3.427 | 0.914× n.s. |
| | cadical3 | Median | | 22.903 | 22.599 | 1.013× n.s. | 22.767 | 21.680 | 1.050× \* | 22.903 | 22.988 | 0.996× n.s. |
| | cadical3 | Mean | | 22.730 | 22.457 | 1.012× n.s. | 22.178 | 20.935 | 1.059× n.s. | 22.886 | 22.886 | 1.000× n.s. |
| **25×25** | glucose4 | Median | | 6.403 | 5.779 | **1.108× \*\*** | 6.200 | 5.305 | **1.169× \*\*\*** | 6.538 | 6.494 | 1.007× n.s. |
| | glucose4 | Mean | | 6.778 | 6.334 | **1.070× \*** | 6.374 | 5.415 | **1.177× \*\*\*** | 7.201 | 7.294 | 0.987× n.s. |
| | cadical3 | Median | | 74.474 | 73.424 | 1.014× n.s. | 72.419 | 71.739 | 1.009× n.s. | 77.260 | 77.328 | 0.999× n.s. |
| | cadical3 | Mean | | 73.104 | 71.989 | 1.016× n.s. | 69.422 | 70.690 | 0.982× n.s. | 76.955 | 73.346 | 1.049× n.s. |

^a Standard backtracking too inefficient beyond 9×9.

**Source-free reconciliation (passed):** 39/42 speedup cells recompute from the displayed 3-dp Normal/Guided cells to the stated speedup within ±0.012. ⚠ The 3 cells outside tolerance are all **backtracking 9×9** (Median-all 33.230 vs stated 33.251; Median-perfect 30.083 vs 30.101; Mean-perfect 75.800 vs 75.750). These sit in the 30–76× regime where rounding the 3-dp display cells produces a larger relative error; the paper's stated speedup is computed from the **full-precision per-instance times**, not the rounded medians/means — the same display-rounding boundary effect seen in iter-45 Spec-AUF / iter-51 kNNGuard, not a transcription error. The abstract's rounded headline figures (**33.3× / 1.70× / 1.17× / 1.02× / 0.90×**) each match the stated speedup to 2 sig-fig.

**Takeaways (Table 2, §3 prose):**

1. **backtracking** — large, significant speedups on all + perfect-hint puzzles (33.251× / 30.101× at 9×9); mixed and n.s. on imperfect hints. Reduced conflicts directly cut DFS runtime.
2. **glucose4** — consistently converts conflict savings into significant wall-clock speedups: **1.699× all / 1.666× perfect at 9×9**; perfect-hint speedups **2.204× (16×16)** and **1.169× (25×25)**. All-puzzles and imperfect-hint speedups at 16×16/25×25 are mostly n.s. (only moderate significance all-puzzles at 25×25). Diminishing returns at scale = a growing share of solve time spent outside search.
3. **cadical3** — **no consistent benefit, the only significant slowdown**. Overhead-dominated: internal bookkeeping is ~independent of conflict count. On 9×9 the phase-init cost (~1.7 ms) on instances cadical3 would have solved trivially produces a 0.896× mean; the more reliable median is 1.021× n.s. At 16×16/25×25 search is a tiny fraction of the budget so guidance is near-zero impact.
4. **The imperfect-hint column is the falsifiable hinge:** across every solver × grid, the Imperfect-hint speedup is essentially 1.0× (0.896–1.069, all n.s.). Guidance pays only when SE-RRM's per-cell top-1 is right; when it is confidently wrong it sends the search into an infeasible subspace, canceling the gains from promising directions.

---

### Table 4 + Table 5 — CP-SAT confirmation experiments (Appendix C, verbatim L998–1043)

To confirm the main findings with a state-of-the-art constraint-programming solver, the paper runs **CP-SAT** (single worker thread, fixed seed) under four configurations (Table 4):

| ID | Name | Description |
|---|---|---|
| C1 | Default CP-SAT | Full CP-SAT heuristics |
| C2 | Naive Fixed | Branches on most-constrained variable, lowest digit first (matches §2.2.1 backtracking branching while keeping CP-SAT propagation + CDCL) |
| C3 | Soft Hint | Warm-start with SE-RRM argmax via `model.AddHint`; if a conflict arises, follow hints up to `hint_conflict_limit=10`, learning from each failure, then fall back to auto search |
| C4 | Repaired Hint | Same hints with `repair_hint=True`; propagation-based local search adjusts only inconsistent cells, `hint_conflict_limit=100` |

**Table 5 — CP-SAT performance across configurations (lower is better):**

| Config | 9×9 | 16×16 | 25×25 | | 9×9 | 16×16 | 25×25 |
|---|---|---|---|---|---|---|---|
| | **Mean search conflicts** | | | | **Mean wall-clock (s)** | | |
| C1 (Default) | 21.8 | 32.8 | 5.6 | | 0.0064 | 0.0237 | 0.0298 |
| C2 (Naive Fixed) | 28.1 | 35.8 | 3.9 | | 0.0122 | 0.0366 | 0.0491 |
| C3 (Soft Hint) | 4.7 | 27.6 | 6.0 | | 0.0066 | 0.0312 | 0.0409 |
| C4 (Repaired Hint) | **0.0** | **16.4** | **3.8** | | 0.0125 | 0.0971 | 0.0469 |

**Source-free reconciliation (passed):** C3 reduces 9×9 conflicts by (28.1−4.7)/28.1 = **83.3%** (paper: "83%") ✓; C3/C2 wall-clock ratios **1.85× (9×9, paper "1.9×"), 1.17× (16×16, paper "1.2×"), 1.20× (25×25, paper "1.2×")** ✓ (all within rounding). C4's 0.0 mean conflicts at 9×9 = the 91.1% feasible instances complete in the hint phase + the rest repaired within the 100-conflict budget ✓.

**Takeaways (Appendix C prose):**

1. **Conflict reduction at low density** — at 9×9 (clue density ≈31.3%) conflicts are right-skewed; C4 achieves 0.0 mean (feasible-in-hint-phase + repaired), C3 cuts mean conflicts 83% (4.7 vs 28.1).
2. **Propagation-dominated regime at scale** — at 25×25 (clue density ≈49.9%) propagation resolves 92.2% of instances under C2 with zero conflicts, so any branching heuristic has little to contribute; **C3 performs WORSE than blind C2 (6.0 vs 3.9)** — incorrect hints add overhead to near-trivial solves. C4 edges C2 negligibly (3.8 vs 3.9). A fundamental limit of hint guidance: when propagation already resolves most of the space, hints are a net cost.
3. **Wall-clock + overhead** — C3 consistently beats C2 (1.9×/1.2×/1.2×) but C1's automatic-portfolio heuristics stay faster than C3 at all scales. C4 minimizes conflicts everywhere but at 16×16 incurs heavy overhead (0.0971 s vs 0.0237 s for C1): only 22.0% of hint sets feasible → repair budget frequently exhausted → costly fallback.

---

## Strengths

- **Two-condition theory (C1 search-dominated + C2 solver-overwrites-hints) is genuinely falsifiable** and the cadical3-vs-glucose4 contrast isolates it cleanly (same hints, same CDCL backbone, only self-correction ability differs).
- **Completeness-preserving by construction** — guidance reorders exploration only, never prunes; correctness comes free from the symbolic solver. This is the rigorous answer to the RRM "no correctness guarantee" weakness.
- **Imperfect-hint column (Table 2) is the honest falsification surface** — speedups vanish (≈1.0× n.s.) whenever SE-RRM's top-1 is wrong, so the gain is honestly attributable to hint accuracy rather than to some incidental solver perturbation.
- **CP-SAT appendix (C3/C4) confirms the main finding with an independent solver family** and adds the propagation-dominated-scale limit (C3 < C2 at 25×25) — the negative result is published, not buried.
- **Symbol-equivariance is load-bearing**: extrapolation to 16×16 / 25×25 (unseen alphabets) is impossible for vanilla RRMs; the SE-axis is what makes SLE (below) conceivable.

## Limitations (paper-stated + honest-scope)

- ⚠ **Wall-clock times EXCLUDE SE-RRM inference time** (precomputed separately). The paper isolates the guidance effect deliberately, but a deployed end-to-end system must add the neural forward pass — on glucose4 9×9 (Normal 0.348 ms) the SE-RRM forward could plausibly dominate, erasing the 1.699× speedup. This is the single most important caveat for any practical claim.
- ⚠ **Sudoku-only** (§5) — no transfer to other CSPs is demonstrated; the SLE generalization is future work.
- ⚠ **backtracking is a custom Python implementation** — the 33.3× is real but not against an engineered baseline; cross-solver magnitude comparisons are explicitly disclaimed (only within-solver Normal-vs-Guided).
- ⚠ **cadical3's 9×9 mean "0.90× slowdown" is significant but small** and is an artifact of the fixed ~1.7 ms phase-init overhead on trivially-solved instances, NOT of guidance making search worse (the median is 1.02× n.s.). The paper states this correctly; a casual reader could over-read it as "guidance hurts cadical3."
- ⚠ **FSR is non-monotonic in grid size** (91.1% / 22.0% / 51.1%) — driven by empty-cell density, not by N. 16×16 is the neural model's weakest regime, which is exactly where Table 1 shows the weakest conflict reduction.

## Verdict

G-RRM is a **clean, honestly-scoped neuro-symbolic study** whose value is the two-condition theory (when does neural guidance help?) more than the magnitude of any single speedup. The headline 33.3× (backtracking) is real but on a non-engineered baseline; the 1.70× (glucose4) and 1.17× (glucose4 perfect-hint 25×25) are the practically credible gains on a modern CDCL solver; the cadical3 null / slight-slowdown is the most citable negative result because it pinpoints WHY (overhead-domination + strict phase honoring). The honest imperfect-hint column (speedups collapse to ≈1.0×) and the CP-SAT C3<C2 propagation-limit are the falsifiable caveats that keep the contribution credible. The wall-clock-excludes-inference limitation caps the deployable claim — read the speedups as "search-cost reduction," not yet "end-to-end latency reduction."

---

## Future work — Solve–Learn–Extrapolate (SLE)

The paper proposes a self-bootstrapping loop for scaling combinatorial solving: **(i)** solve small instances exactly with a symbolic solver, **(ii)** learn an SE-RRM on these valid solutions, **(iii)** extrapolate to larger instances where SE-RRM hints guide the solver; because G-RRM preserves completeness, the new larger solutions are valid and can be fed back to re-train the SE-RRM before advancing to the next size, lifting the frontier of tractable sizes using only valid data. The 9×9-trained + 16×16/25×25-fine-tuned SE-RRM in this paper "realizes a first step of this loop in miniature."

---

## Sourcing note

All numbers transcribed verbatim from `paper_layout.txt` (pdftotext -layout, 1088 lines, 5 explicit tables + 2 figures + Eqs 1–10). Table line-ranges: **Table 1 L419–452, Table 2 L470–506, Table 3 (notation) L934–987, Table 4 (CP-SAT configs) L998–1003, Table 5 (CP-SAT performance) L1030–1043.** No figure-derived numbers were back-filled — the paper's verbatim substance is entirely in the 5 tables + prose-confirmed headline figures. Source-free reconciliation: 39/42 Table-2 speedups recompute exactly (3 backtracking cells = full-precision-vs-display rounding, flagged inline); Table-5 C3-vs-C2 conflict-reduction 83.3% and wall-clock ratios 1.85×/1.17×/1.20× all reconcile to the paper's stated 83% / 1.9× / 1.2× / 1.2× within rounding; abstract headlines 33.3× / 1.70× / 1.17× / 1.02× / 0.90× each match a stated speedup. **No numeric prose-vs-table contradiction** (unlike the iter-30/31/34/46 class).
