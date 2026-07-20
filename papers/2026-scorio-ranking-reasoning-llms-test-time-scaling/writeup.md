# Writeup — Ranking Reasoning LLMs under Test-Time Scaling (Scorio)

> Hariri, Hinczewski, Ma, Chaudhary.
> "Ranking Reasoning LLMs under Test-Time Scaling." arXiv:2603.10960 (2026).
> Code: https://github.com/mohsenhariri/scorio

## What the paper is actually about

Most LLM leaderboards report a single number per model: "Qwen3-Thinking
scores 0.875 on AIME'24." But under test-time scaling (TTS), each model is
evaluated by sampling N independent responses per question, and the score
is the average over those samples. That turns benchmarking into a
**repeated-sampling problem**: there is no single "the score" — only an
estimate, and the estimate's stability depends on N.

The paper asks a question most leaderboard builders never ask: **which
ranking rule** do you apply to the resulting response tensor
`R ∈ {0,1}^{L×M×N}` (L models × M questions × N trials)? There are dozens of
options — mean accuracy, Bradley-Terry MLE, Borda count, PageRank, Elo,
Bayesian posterior-mean with various priors — and the choice is usually
made by habit. This paper shows the choice is **not** cosmetic at low N,
and proves a minimal case where two standard rules disagree.

## What I implemented

A from-scratch Python re-implementation (`numpy` only) covering a
representative 8 of the paper's 72 ranking methods, a synthetic
response-tensor generator, and the Appendix-C theoretical result. Four
headline findings are reproduced:

| Finding | Paper claim | My result |
|---|---|---|
| **F1** (Table 1) | At N=80, mean Kendall τ_b vs gold ≈ 0.93–0.96; 19–34 methods recover exact ordering | mean τ_b = 0.964; 7/8 exact |
| **F2** (Table 2) | At N=1, best methods reach τ_b ≈ 0.86; greedy prior helps on easy benchmarks | bayes_greedy best at 0.982 (synthetic is easier than real math) |
| **F3** (Table 4) | Greedy prior always cuts variance (16–52%) but biases when τ_G-S low | aligned: Δτ=+0.065, std↓58%; misaligned: Δτ=−0.096, std↓52% |
| **F4** (App. C) | BT ≠ average at M_min=8; no disagreement for M≤7 | counterexample verified; 0 disagreements across 3552 strict datasets for M≤7 |

## What implementing it clarified (that the paper didn't make obvious)

### 1. The M_min=8 counterexample is tiny — you can hold it in your head

The paper's Appendix C derivation spans several pages of equations. But the
actual counterexample is just 8 questions with 3 outcome patterns:

```
Type-A (×2):  model outcomes (0,1,1)   — only model 0 fails
Type-B (×3):  model outcomes (1,0,0)   — only model 0 solves
Type-C (×3):  model outcomes (1,1,0)   — model 2 always fails
```

Count up the column marginals and model 0 solves 6/8, model 1 solves 5/8,
model 2 solves 2/8. So **average ranks 0 > 1 > 2**. But count the
*pairwise decisive wins* (where one model solves and the other doesn't):

```
W = [[0, 3, 6],     # model 0 beats 1 three times, beats 2 six times
     [2, 0, 3],     # model 1 beats 0 twice, beats 2 three times
     [2, 0, 0]]     # model 2 beats 0 twice, never beats 1
```

Model 1 beats model 0 head-to-head only 2 times vs 3 — but those 2 wins
are "cheaper" in the BT likelihood because model 1's overall strength is
lower. Solving the BT fixed-point conditions flips the ranking to
**1 > 0 > 2**. The disagreement is not about total accuracy; it's about
*where* the wins land. Average throws away the opponent identity; BT
doesn't.

Running `python3 bt_vs_avg.py` builds this exact tensor and prints both
rankings side by side. It's the most satisfying 20 lines of the whole
implementation.

### 2. The "no disagreement for M≤7" claim is expensive to verify naively

The paper says they enumerated 1506 datasets with M≤7 and found zero
disagreements. My first implementation found 576 disagreements and I
thought I had a bug. The issue: **ties**. When two models have identical
marginal accuracy (e.g. both solve 3/8), the "average ranking" is
ambiguous — `argsort` breaks the tie by model index, but BT breaks it by
win-structure. Those aren't real disagreements; they're artifacts of
tie-breaking on an ill-defined ordering.

The fix: only count a dataset as a valid test case when all marginal
accuracies are distinct (a *strict* average ordering). After that filter,
3552 strict datasets for M≤7, **zero** disagreements. The paper's claim
holds. This is a subtle point the paper glosses over — "1506 instances" is
the count *after* whatever dedup/tie filtering they applied.

### 3. The greedy prior is shrinkage, not magic

`Bayes_R0@N` (greedy prior) is the paper's most actionable practitioner
result: it cuts single-trial variance by 16–52%. The paper frames this as
"the greedy decode is an informative prior." Implementing it made clear
it's really just **Beta-Binomial shrinkage toward the greedy ordering** —
you're adding pseudo-counts from one greedy trial, which pulls the
posterior mean toward wherever greedy landed.

The bias-variance flip (Table 4) follows directly: when greedy and sampling
agree (high τ_G-S), shrinkage toward greedy is helpful. When they disagree
(low τ_G-S, e.g. HMMT'25 where greedy under-explores hard problems),
shrinkage toward a wrong target **biases** the ranking — Δτ goes negative.
My F3 reproduces both regimes by perturbing the skill vector used only for
the greedy draw (`greedy_skill_noise`), which cleanly separates the two
effects without needing real LLMs.

### 4. PageRank direction matters more than you'd think

My first PageRank implementation returned τ_b = −1.0 (perfectly
anti-correlated with the gold standard). The bug: I'd built the transition
matrix so the random walk followed *wins* (winner → loser), which puts high
PageRank on models that *lose to strong models* — the opposite of what you
want. The fix: the walk should follow *losses* (loser → winner), so high
PageRank lands on models that beat many others. This is the same direction
convention as Rank Centrality (walk toward whoever beat you). A 4-character
fix (`.T` → identity) but it took reading the negative τ_b to notice.

## What was harder than expected

- **Bradley-Terry MLE convergence.** The iterative-scaling update (Hunter
  2004) is simple but sensitive: near-separation (one model beats everyone
  on every question) makes the MLE diverge. I cap iterations and clamp
  updates; the paper notes the same instability (§2.3, "unstable under
  near-separation"). On the M=8 counterexample it converges cleanly in
  ~50 iterations.
- **Elo is path-dependent.** Unlike every other method here, Elo's final
  rating depends on the *order* in which comparisons are processed. The
  paper doesn't discuss this; I process (question, trial, model-pair) in
  index order for determinism. This is likely why Elo underperforms in my
  F1 (τ_b = 0.709 vs 1.0 for the others) — it's noisier than the
  simultaneous-update methods.
- **Synthetic data calibration.** Matching the paper's absolute τ_b values
  (0.86 at N=1) requires the synthetic latents to be as noisy as real
  LLM-on-math data. My defaults give τ_b ≈ 0.92 at N=1, which is higher
  (easier) than the paper's real-data figure. I note this as a limitation
  rather than tuning the generator to hit a target number.

## Pointers to the code

| File | What |
|------|------|
| `implementation/rankings.py` | 8 methods: `avg`, `bayes_uniform`, `bayes_greedy`, `bradley_terry_mle`, `borda`, `copeland`, `pagerank`, `rank_centrality`, `elo` + `kendall_tau_b` |
| `implementation/bt_vs_avg.py` | M_min=8 counterexample (`verify_counterexample`) + exhaustive no-disagreement search for M≤7 |
| `implementation/data.py` | Synthetic response-tensor generator with tunable τ_G-S |
| `implementation/run.py` | Reproduces F1–F4 end to end |

## Verdict

A foundational evaluation-methodology paper that's more useful than its
"ranking rules" framing suggests. The practical takeaway (`Bayes_U@N` is
the safe default; pilot-check before `Bayes_R0@N`) is a 2-line recipe every
leaderboard builder should know. The theoretical anchor (M_min=8) prevents
the empirical agreement at N=80 from being over-read as "all rules are
equivalent." Reproducing it takes ~500 lines of numpy and zero GPU time —
the cleanest paper-to-code mapping in this repo so far.

🏆 Verdict: worth implementing. The counterexample alone is worth the entry.
