# DecompRL: Solving Harder Problems by Learning Modular Code Generation

**arXiv:** 2607.02390v1 [cs.LG] (2 Jul 2026) — Preprint.
**Authors:** Juliette Decugis¹·²\*, Fabian Gloeckle¹·³\*, Francis Bach², Taco Cohen¹, Gabriel Synnaeve¹ (\* equal contribution). ¹FAIR at Meta, ²Inria / École Normale Supérieure, ³CERMICS / École des Ponts ParisTech.
**Code:** none stated in paper.
**Subarea (repo lineage):** FIRST modular/hierarchical code-generation RL paper in the repo. The repo's agentic-RL lineage (`demystifying-rl`, `verification-horizon`, `opid`, `are-we-ready`, `multi-turn-rl`) optimizes tool-use *trajectories*; DecompRL optimizes the *structure of the solution* — it trains an LLM to decompose a problem into independently-implementable functions and recombine `k` implementations of `n` modules into `kⁿ` candidate programs. Sibling-in-spirit to `distribution-wise-rewards` (both reshape what RL optimizes — that paper the reward granularity, this paper the credit-assignment over a recombination lattice) and to `speculating-experts` (both shift compute off the GPU — experts to CPU, here candidate *evaluation* to CPU).

---

## TL;DR

Repeated sampling (pass@k) and standard RL both fail when the base policy has near-zero probability of producing a correct solution: the search space is simply too large. **DecompRL** makes the task easier instead of sampling harder. It trains two cooperative policies — a **decomposition policy** `π(D)` (break the problem into `n` function signatures + docstrings) and an **implementation policy** `π(I|D)` (write `k=8` implementations of each function in parallel) — so that `n·k` model forwards yield `kⁿ` recombined candidate programs whose correctness is checked by cheap CPU unit tests. The RL objective is a **logmeanexp₆ multi-sample advantage** (Eq. 1–2) with a leave-one-out baseline that interpolates between mean (standard RL) and max (pass@k RL), avoiding the gradient saturation of pure pass@k training. On LiveCodeBench + CodeContests (Qwen 2.5 7B, Code World Model 32B), DecompRL **cuts GPU token cost ~50×** (198k→4k tokens/problem at 512 evaluations) and **solves problems beyond 10⁵ tokens/problem** that standard/diversity-optimized RL cannot reach (up to **35% of the LiveCodeBench hard subset**, up to 3M tokens/problem, pass@1k).

---

## 1. Background — test-time compute, pass@k, and why both scaling strategies stall

Competitive-programming SOTA relies on **repeated sampling**: generate many candidate solutions, keep those that pass a verifier (the **pass@k** regime, `pass@k = 1−(1−p)ᵏ` for per-attempt success rate `p`). Two failure modes:

1. **GPU cost grows linearly** with attempts `k`; performance grows only **logarithmically** (Appendix B). Diminishing returns.
2. **RL post-training improves pass@1 but destroys pass@k diversity** for `k ≫ 100` — the policy sharpens on one solution mode, which undermines test-time scaling.

Both bottom out when the base policy's `p ≈ 0`: no amount of sampling or gradient signal overcomes a search space too large to hit by luck. DecompRL's bet: **don't sample harder, decompose** — make each independent module easy enough that `p_module ≫ 0`, then let recombination do the combinatorial search.

---

## 2. Method — hierarchical inference, recombination, and the logmeanexp objective

### 2.1 Hierarchical inference (after Parsel / Zelikman 2023)

Standard whole-code generation models the joint autoregressively (an optional plan `D` is sampled first):

> `π(I₁,…,Iₙ) = π(I₁|D) · ∏ᵢ π(Iᵢ | Iᵢ₋₁,…,I₁, D)`

DecompRL instead assumes a **hierarchical decomposition** with *conditional independence* of the implementations given the plan:

> `π(D, I₁,…,Iₙ) = π(D) · ∏ᵢ π(Iᵢ | D)`

All inter-module interaction is captured by the decomposition `D`. For Python code, `D` = a set of function signatures + docstrings; the parts `Iᵢ` are the function bodies. This *loses single-sample precision* (conditional independence ≠ true joint) but buys cheap recombination.

### 2.2 Recombination — polynomial candidates, linear cost

Given a decomposition of size `n` and `k` implementations per function `(Iᵢⱼ)`, form **all** complete trajectories `(D, I₁ʲ¹,…,Iₙʲⁿ)` for `1 ≤ jᵢ ≤ k`:

> `n·k` model forwards  →  `kⁿ` candidate programs

Candidate count scales **polynomially** in `k` while inference cost scales **linearly**. With `k=8, n_max=6` the theoretical ceiling is `8⁶ = 262,144` recombinations per decomposition (the source prints this as "kⁿᵐᵃˣ = 86" — the `8⁶` superscript is stripped by pdftotext; **sourcing note**: 8⁶ = 262,144, not 86). Verification (running hidden unit tests) is a cheap CPU operation; generation is the expensive GPU operation — so recombination shifts the bottleneck exactly where the paper wants it.

### 2.3 Policy-gradient estimators — why the recombination lattice is a better gradient estimate

Writing a trajectory `τ = (D, I₁,…,Iₙ)`, the policy-gradient theorem gives `g = E_{τ∼πθ}[ Σ_{a∈τ} (r(τ)−b) ∇θ log πθ(a) ]` for any action-independent baseline `b`.

- **Standard estimator** `ĝ_standard` sums only the *diagonal* rollouts `(Iᵢⁱ)` — optimal for *linear* (autoregressive) rollouts where `Iₖ` depends on `Iⱼ` for `k>j`.
- **Hierarchical estimator** `ĝ_hierarchical` sums over **all** `kⁿ` recombinations `(I₁ʲ¹,…,Iₙʲⁿ)`.

Both are **unbiased**, but `Var(ĝ_hierarchical) ≤ Var(ĝ_standard)` (**Theorem A.2**) — the recombination lattice is a provably lower-variance Monte-Carlo estimate of the hierarchical policy gradient.

### 2.4 The logmeanexp₆ multi-sample objective (Eq. 1–2)

Pure pass@k training takes `f = max` over rollout rewards — but with thousands of correlated recombinations, max **saturates**: a non-zero advantage appears only if `τᵢ` is the *unique* solver among `k` attempts (extremely sparse → vanishing gradients; Appendix Fig 10). Standard RL takes `f = mean` — no diversity pressure.

DecompRL uses the smooth interpolation (**Eq. 1**):

> `logmeanexp₆(r₁,…,rₙ) = β · log( (1/n) Σᵢ e^{rᵢ/β} )`

- `β → ∞`: recovers the **mean** (standard RL).
- `β → 0`: recovers the **max** (pass@k RL).
- well-chosen `β`: approximates a **log-uniform mixture of pass@k objectives** (Fig 3), balancing exploration (high-k) and exploitation (pass@1).

**DecompRL is a cooperative 2-policy MARL framework.** For `d` decompositions `Dᵢ` (each size `nᵢ`), `k` implementations/function, and `m ≤ k^{max(n₁..n_d)}` evaluated combinations per decomposition, with rewards `rᵢⱼ` and `A(rᵢⱼ)` the set of actions in recombination `rᵢⱼ`:

- **Decomposition-policy gradient (Eq. 2):** advantage = `logmeanexp₆(r) − logmeanexp₆({r_{i'j} | i' ≠ i})` — the leave-one-out drop in multi-sample objective when decomposition `i` is removed.
- **Implementation-policy gradient:** advantage = `logmeanexp₆(r) − logmeanexp₆({rⱼ | Iₗⱼₗ ∉ A(rⱼ)})` — the leave-one-out drop over rewards the action did *not* participate in.

Because the baseline is action-independent, the gradient stays **unbiased**; leave-one-out cuts variance (Appendix E). Empirically the reward tensor over recombinations is well-approximated by **rank-1** (Appendix J) — credit concentrates along individual function axes, so the leave-one-out baseline is reliable from a sparse subset of `kⁿ` combinations. The two-stage EM-style training addresses the non-stationarity of cooperative MARL (counterfactual action-value estimation, Foerster 2024).

---

## 3. Experimental setup

- **Models:** Qwen 2.5 7B, Llama 3.1 8B Instruct, Code World Model (CWM) 32B.
- **Training data:** 15,000 competitive-programming problems (CodeContests + TACO training sets). Online RL.
- **Evaluation:** CodeContests validation (117 problems), LiveCodeBench 2024/08/01–2025/02/01 (279 problems).
- **Hardware:** 80× H100 split into workers (sample generation) and trainers (policy updates).
- **Baselines (Qwen 2.5 7B, all 16 samples/prompt except where noted):** GRPO · instruct (no RL) · **pass@8-training** (k=8, Chen 2025) · **SPO** (Soft Policy Optimization, Cohen 2025) · **logmeanexp-16** (β=0.3, this paper's objective at 16 samples) · **logmeanexp-48** (β=0.1, 48 samples — to mimic DecompRL's per-decomposition forward count of ≤6 functions × 8 implementations).
- **DecompRL training (EM-style, 2 stages):** (1) train **decomposition** policy 30k steps with implementation fixed; (2) train **implementation** policy 30k steps with decomposition fixed. 8 decomposition samples/problem, `n_max = 6` functions, `k = 8` implementations/function → `8ⁿ` combinations, **512 randomly sampled** for evaluation (breadth-vs-depth balance). β = 0.3 (decomposition stage) / 0.1 (joint, cf. Fig 5).

---

## 4. Results

### 4.1 DecompRL wins only at high token budgets (Table 1 — verbatim)

**Table 1.** pass@tokens on **LiveCodeBench, Qwen 2.5 7B** — solve rate at fixed token budgets across online-RL methods.

| Token budget | GRPO | instruct | lme16 | lme48 | pass@8 | **DecompRL (ours)** |
|---|---|---|---|---|---|---|
| 1,000    | 0.18 | 0.06 | 0.19 | 0.12 | 0.15 | **0.18** |
| 5,000    | 0.29 | 0.16 | 0.28 | 0.26 | 0.27 | 0.18 |
| 10,000   | 0.32 | 0.21 | 0.31 | 0.30 | 0.31 | 0.25 |
| 50,000   | 0.38 | 0.30 | 0.38 | 0.36 | 0.39 | **0.40** |
| 100,000  | 0.40 | 0.33 | 0.40 | 0.39 | 0.41 | **0.44** |
| 500,000  | 0.44 | 0.39 | 0.46 | 0.44 | 0.46 | **0.48** |

*Sourcing: paper_layout.txt L321–333.*

**Takeaways (honest-scope — the central nuance):** DecompRL **loses or ties at low budgets** (5k: 0.18 vs GRPO 0.29; 10k: 0.25 vs 0.32) and **wins only at ≥50k tokens** (50k +0.01, 100k +0.03, 500k +0.02 over the best baseline). This is the **format tax** — hierarchical inference wastes single-sample precision, which only pays off once the recombination budget is large enough to exploit diversity. The abstract's "outperforms … beyond 10⁵ tokens/problem" is *exactly* scoped to this crossover; DecompRL is a **high-budget search policy**, not a pass@1 improver. This is the paper's honest framing and should not be over-read as a blanket win.

### 4.2 ~50× GPU token cut, bottleneck → CPU (Table 2 — verbatim)

**Table 2.** Wall-clock cost breakdown **per training step, Qwen 2.5 7B, 512 evaluations/problem** (numbers per prompt, full sample size).

| Method | Samples | Gen tokens | Gen time (s) | Exec calls | Exec time (s) | Train time (s) | Total (s) | GPU / CPU |
|---|---|---|---|---|---|---|---|---|
| Standard | 16  | 10k  | 108  | 16  | 371    | 1.1  | 479    | 23% / 77% |
| Standard | 512 | 198k | 2100 | 512 | 11878  | 21.8 | 14000  | 15% / 85% |
| **DecompRL** | 512 | **4k** | **42** | 512 | 11878 | 0.4 | **11900** | **0.4% / 99.6%** |

*Sourcing: paper_layout.txt L539–549.*

**Takeaways:** At matched 512 evaluations, DecompRL generates **~4k tokens** (1 decomposition + `n×8` short implementations) vs **~198k** for standard sampling → **~50× GPU-token reduction** (198000/4000 = 49.5×). Execution cost is *identical* (same 512 candidate programs evaluated). Wall-clock savings scale with how GPU-bound the cluster is: **3.6×** at 8 GPUs/128 threads, **2.3×** at 8 GPUs/64 threads, shrinking to **1.3×** at 72 GPUs/128 threads (CPU already dominates). The paper's argument: scaling CPUs (cheaper than GPUs) unlocks larger gains since marginal recombination evaluation is ~free on the GPU side.

### 4.3 RL training inflates tokens-per-attempt (Table 3 — verbatim)

**Table 3.** Token count per code attempt, **Qwen 2.5 7B**.

| Method | Tokens/attempt |
|---|---|
| no training | 387 |
| GRPO | 545 |
| pass@8 | 570 |
| logmeanexp, 16 samples | 500 |
| logmeanexp, 48 samples | 538 |
| **DecompRL** | **4000** |

*Sourcing: paper_layout.txt L1728–1732.*

**Takeaway:** A DecompRL "attempt" is 1 decomposition + `n×8` implementations, so it intrinsically carries ~10× the tokens of a standard attempt (4000 vs 387–570). This is why the pass@tokens comparison (Table 1) is the fair one — it normalizes by *total token budget*, not attempt count. RL training broadly increases tokens/attempt (all RL methods > no-training 387), consistent with policies learning longer/more elaborate solutions.

### 4.4 The gain is RL, not hierarchy alone (Table 4 — verbatim)

**Table 4.** Pass@1 at ~4k tokens, **LiveCodeBench** — hierarchical inference *without* RL underperforms standard inference.

| Model | Method | Pass@1 (%) |
|---|---|---|
| Qwen 2.5 7B | Standard | 14.0 |
| Qwen 2.5 7B | Hierarchical (no RL) | 11.0 |
| Qwen 2.5 7B | **DecompRL** | **18.0** |
| CWM 32B | Hierarchical (no RL) | 11.9 |
| CWM 32B | **DecompRL** | **27.8** |

*Sourcing: paper_layout.txt L1824–1833.*

**Takeaway:** Prompted hierarchical inference *without* RL **underperforms** standard (Qwen 7B 11.0 < 14.0; CWM 32B similarly below standard) — the modular format is a tax, not a free win. RL training is what flips it into a gain (Qwen 7B 11.0→18.0; CWM 32B 11.9→27.8, +15.9pp). This isolates the contribution: **the logmeanexp objective + cooperative 2-policy training**, not the hierarchical scaffold itself.

### 4.5 Solving new problems (§4.3, Figures 1/4)

- DecompRL solves up to **35% of the LiveCodeBench hard subset** (2024/08/01–2025/02/01) at high token budgets (Fig 1) — problems unreachable by standard generation.
- Evaluation scales to **m = 4096 combinations/decomposition** (~80× more than the `n·k+1 = 49` standard forwards, since 4096/49 = 83.6) and up to **3M tokens/problem**, **pass@1k**, **10⁷ generated tokens**.
- Diversity keeps scaling where baselines saturate: success rate from <50 forwards shows no plateau even at m=1000 evaluations (Fig 5a); solve rate keeps climbing at fixed m=512 throughout RL training (Fig 4b).

---

## 5. Limitations & honest scope

- **Format tax.** Hierarchical inference without RL underperforms standard (Table 4); DecompRL **degrades Llama 3.1 8B Instruct** from 26.6%→14.5% on CodeContests validation (same token budget; 10 DecompRL attempts ≈ pass@316 standard). Worse than the starting policy on the LiveCodeBench *easy* split and at low token budgets (Fig 1, Table 1). DecompRL is a high-budget explorer, not a universal fine-tune.
- **Two-model copies.** Although only one policy trains at a time (cost ≈ a traditional RL reference policy), the EM-style 30k+30k sequential schedule is heavier than single-policy GRPO.
- **Decomposition-size tension (⚠, paper-internal, figure-derived).** Fig 4c caption says "the decomposition policy learns to use **less** functions during training," yet §4.2 + Fig 5c report that "models that create **larger** decompositions improve faster" and "best-performing models for large attempt numbers produce **bigger** decompositions on average." Reconcilable as different axes (within-run temporal trend vs cross-model/budget correlation) but worth flagging — both are figure-derived and not pinned by a table.
- **Narrow domain.** Expressed for Python competitive programming; the structural assumption (hierarchically decomposable + cheaply verifiable) must hold. The paper generalizes speculatively to proofs, hardware modules, scientific codebases.

---

## 6. Strengths / Limitations / Verdict

**Strengths**
- **Genuinely new RL lever:** instead of more compute or a bigger model, DecompRL changes *what is being searched* — a recombination lattice over independently-solvable modules. The `kⁿ`-from-`n·k`-forwards asymmetry is the crisp, citable mechanism.
- **Theoretically grounded gradient.** Theorem A.2 (lower-variance hierarchical estimator) + the rank-1 reward-tensor empirical finding (App J) justify the leave-one-out advantage over a sparse subset of `kⁿ` combinations.
- **logmeanexp₆ objective** elegantly resolves the mean-vs-max tension that makes pass@k training saturate — a smooth, single-knob (β) interpolation that approximates a log-uniform pass@k mixture.
- **Honest scoping.** The format tax (Table 4), the Llama degradation, and the low-budget losses (Table 1) are reported plainly rather than buried.

**Limitations**
- The win is **budget-conditional** (≥50k tokens) and **model-conditional** (Llama 3.1 8B regresses). A reader wanting a pass@1 improver will be disappointed.
- Heavy figure-dependence for the diversity/scaling claims (Figs 1/4/5); only 4 explicit tables, and Tables 2–4 are small.
- No released code; reproducibility depends on the prompts in Appendix I and the (long) 30k+30k schedule.

**Verdict.** DecompRL is the strongest example in the repo of **RL that expands the reachable solution space** rather than sharpening pass@1 — a search-policy contribution whose value is precisely at the high-budget regime where standard RL has already saturated. The most citable single result: **~50× GPU token cut** (Table 2) at matched evaluation count, with the honest caveat that this buys *new* solves only beyond ~10⁵ tokens/problem. Conceptually it reframes post-training exploration: train an *explorer* policy whose recombination outputs feed a downstream pass@1 *distiller* (the explorer/distiller pipeline the conclusion proposes) — a clean division of labor between discovery and deployment.

---

## Sourcing notes

- **Source:** `paper.pdf` (993KB, 29pp) → `paper_layout.txt` (pdftotext -layout, 1845 lines). 4 explicit tables (caption format `Table N <text>`, no colon — `^Table N:` regex misses them; bare `Table N ` grep finds all 4) + 10 figures. No Algorithm block (the method is prose + Eq. 1–2).
- **Table locations:** Table 1 L321, Table 2 L539, Table 3 L1728, Table 4 L1824. All 4 transcribed verbatim above.
- **Superscript-strip artifact (⚠):** §4.3's "theoretical maximum `kⁿᵐᵃˣ = 86`" is `8⁶ = 262,144` with the exponent glyph lost in extraction — not a paper typo. Transcribed as 8⁶ = 262,144.
- **Source-free reconciliation (all passed):** 50× = 198000/4000 = 49.5 ✓; 80× = 4096/(6·8+1) = 4096/49 = 83.6 ✓; `kⁿ` = 8⁶ = 262,144 ✓; Table-1 DecompRL is column-max at budgets {50k, 100k, 500k} and column-min at {5k, 10k} (honest format tax); Table-2 GPU fractions {23/77, 15/85, 0.4/99.6} sum to 100%; Table-4 DecompRL > Hierarchical-no-RL > Standard for both models where Standard is reported. No numeric prose-vs-table contradiction (unlike `multi-turn-rl`/`speculating-experts`/`vprm`); the one surface tension (Fig 4c "less" vs §4.2 "larger" decomposition size) is figure-derived and flagged inline.
- Figure curves (Figs 1/4/5 pass@tokens-vs-budget, decomposition-size evolution, scaling) NOT back-filled — only prose-confirmed ranges (35% hard subset, ≥10⁵ tokens crossover, m=4096, 3M tokens/problem, pass@1k, 10⁷ tokens) and the explicit tables carry the verbatim substance, consistent with the repo-wide "figure-derived numbers are weak" rule.
