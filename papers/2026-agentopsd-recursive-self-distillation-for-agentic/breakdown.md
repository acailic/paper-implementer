# Breakdown — AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning

> **Paper:** "AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning"
> **Authors:** Zi-Han Wang, Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Jie Wu, Zhengzhou Cai, Yueqing Sun, Ziang Ye, Linji Hao, Qi Gu, Xunliang Cai, Yongliang Shen, Yujiu Yang (Tsinghua University, Zhejiang University, Meituan)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2608.05987
> **Code (official):** https://github.com/ZethWang/AgentOPSD

---

## 1. Problem & Motivation

**The problem: credit assignment in multi-turn agentic RL.**

When an LLM agent is trained with RL (GRPO-style) on environments like ALFWorld,
WebShop, or Search-QA, the reward is usually a **single binary success/failure
signal delivered once at the end of the episode**. GRPO's answer is purely
statistical: sample `G` trajectories per task, compute each trajectory's
advantage as

```
A_seq = (R − mean(R_1..G)) / (std(R_1..G) + ε)
```

and then **broadcast that one scalar to every token in the trajectory**. In a
30-turn episode where 2–3 turns were actually pivotal (the right search query,
the right object picked up) and the rest were routine, every turn receives the
same credit. The pivotal signal is drowned in noise, and the damage grows with
horizon length: the paper measures GRPO losing ~2.9 success points per extra
ALFWorld turn, vs ~0.5 for their method.

**Why it matters.** Long-horizon agent training is exactly where RL is being
applied today (tool use, web navigation, embodied tasks), and the
one-advantage-per-trajectory assumption is the standard stack's weakest link.
Classical fixes are expensive: a learned critic (PPO), extra rollouts from
intermediate states (VinePPO-style), or a trained process reward model — each
adds parameters, samples, or supervision the agentic setting can't easily afford.

**What prior approaches did.**
- *Outcome-relative baselines*: GiGPO builds turn-level advantages from
  environment rewards at repeated "anchor states" across trajectories — needs
  the environment to emit per-step rewards / revisitable states.
- *Privileged self-distillation (OPSD family)*: use the same policy weights as
  its own teacher by adding a privileged context (here: a retrieved "skill"
  text) and contrast teacher vs student log-probs of the sampled action. Dense
  and cheap, but the signal is **token-local and turn-myopic**: it scores each
  token/action in isolation, ignoring (a) that tokens form whole actions the
  environment only responds to at turn boundaries, and (b) what evidence
  earlier turns already accumulated — a strong local gap at turn 20 is
  redundant if the trajectory was already clearly winning.

**Limitation AgentOPSD attacks:** none of the cheap methods turn a *sequence*
of local signals into *sequential* credit. That conversion is the paper's whole
contribution.

## 2. Key Insight / Contribution

**Core idea (own words):** A local teacher–student log-prob gap is NOT credit.
Credit is **how much a turn's evidence moves a running Bayesian belief that the
trajectory will succeed**. So: aggregate the per-token gaps into per-turn
evidence, fold that evidence recursively into a belief state in log-odds space,
and give each turn the **marginal belief revision** ΔB it caused, signed by the
outcome. Turns that changed the trajectory's fate get more of the GRPO
advantage; turns that were redundant get less.

**What is genuinely new:**
1. Recasting self-distillation gaps as **Bayesian evidence** in a recursively
   updated log-odds belief (γ=1 literally recovers Wald's 1945 SPRT
   accumulator), with the prior anchored at the GRPO group success rate.
2. Defining turn credit as the **marginal belief revision** ΔB_k (with the
   sigmoid derivative B(1−B) acting as an automatic uncertainty gate), not the
   raw gap.
3. A **bounded, sign-preserving advantage reshaper** — the reshaped advantage
   can only modulate GRPO's magnitude within (1±λb), never flip its direction —
   so the method is a safe drop-in on top of standard PPO/GRPO. No critic, no
   extra rollouts, no new parameters; overhead = one extra forward pass per
   trajectory.

## 3. Method

Reconstruct-from-scratch walkthrough. Symbols: task `x`, turn index `k`
(1..K), token index `t`, policy `π_θ`, episode `τ`, binary terminal reward
`R(τ) ∈ {0,1}`.

### 3.1 Overview

```
per task x:
  retrieve skill c⁺ (keyword match, training-only)
  sample G trajectories with π_θ
  verifier → binary rewards R⁽ⁱ⁾
  GRPO → one A_seq per trajectory                       (Eq. 1–2)
  per trajectory:
      B₀ = clip(group success rate, ε₀, 1−ε₀)           (prior, Eq. 8)
      per turn k = 1..K:
          e_k = Σ_t [log π(y_t | skill ctx) − log π(y_t | plain ctx)]   (Eq. 4–6)
          c_k = γ·c_{k−1} + e_k                          (accumulator)
          B_k = σ( logit(B₀) + c_k )                     (belief, Eq. 8)
          ΔB_k = B_k − B_{k−1}                           (credit, Eq. 9)
          q_k = sign(A_seq) · ΔB_k                       (outcome-aligned, Eq. 10)
      standardize q within trajectory → z_k
      w_k = clip(1 + b·z_k, 1−b, 1+b)                    (Eq. 11)
      Ã_k = A_seq · ((1−λ) + λ·w_k)                      (bounded reshape)
  every token of turn k inherits Ã_k
  clipped PPO/GRPO loss on r = π_θ/π_old                 (Eq. 12)
```

### 3.2 Architecture

There is **no new architecture** — the policy is the same LLM (Qwen2.5-3B/7B
-Instruct) throughout. The "teacher" is the *same network, same weights* run a
second time with an augmented context:

```
                 ┌────────────────────────────────────────────────┐
                 │                policy π_θ (frozen copy of      │
                 │                current weights, sg-detached)   │
                 │                                                │
 state s_k ──┬──▶│ context h⁺ = (s_k, c⁺, y_<t)  → log π(y_t|h⁺) │──┐
             │   │                                                │  │ per-token gap
             └──▶│ context h  = (s_k,      y_<t)  → log π(y_t|h ) │──┤ δ_t (detached)
                 └────────────────────────────────────────────────┘  │
                                                                    ▼
              turn evidence        recursive belief             bounded reshaper
             ┌──────────────┐    ┌───────────────────────┐    ┌────────────────────┐
   δ_{k,·} ─▶│ e_k = Σ_t δ_t│──▶ │ c_k = γc_{k−1} + e_k  │──▶ │ z_k = stdev(q_k)   │
             └──────────────┘    │ B_k = σ(logit(B₀)+c_k)│    │ w_k = clip(1+b·z_k) │
                                 │ ΔB_k = B_k − B_{k−1}  │    │ Ã_k = A((1−λ)+λw_k) │
                                 │ q_k = sign(A)·ΔB_k    │    └────────────────────┘
                                 └───────────────────────┘            │ token t of
              B₀ = clip(S/G, ε₀, 1−ε₀)  (prior = group success rate)  ▼ turn k
                                                              clipped GRPO update
```

Where the baselines sit on the same skeleton: OPSD distills at token level
(aux loss); StepOPSD aggregates per step but does NOT recurse; RLSD uses the
raw gap as a magnitude coefficient; SDAR adds a gated aux loss and keeps GRPO's
advantage untouched. AgentOPSD = **turn granularity + recursion + bounded
signed reshaping**.

### 3.3 Forward pass / pipeline

1. **Rollout.** For task `x` with initial observation `o₀` and state
   `s₁=(x,o₀)`: at each turn the policy samples a multi-token action
   `a_k = y_{k,1..L_k} ~ π_θ(·|s_k)`; environment returns `o_k`;
   `s_{k+1} = (s_k, a_k, o_k)`. Repeat until done or K_max. Sample G such
   trajectories per task.
2. **Reward & GRPO advantage.** Verifier gives binary `R⁽ⁱ⁾`.
   `A_seq⁽ⁱ⁾ = (R⁽ⁱ⁾ − R̄)/(σ̂_R + ε₀)` with group mean `R̄` and std `σ̂_R`.
   All-success / all-fail groups have A_seq ≈ 0 → no update (correctly so).
3. **Skill retrieval.** One skill text `c⁺` per task, keyword-matched from the
   SkillBank of SkillRL. **Training-only** — inference uses no skills.
4. **Teacher pass (the only overhead).** Re-run the same weights over each
   trajectory with `c⁺` prepended to each turn's context. Compute per-token
   detached contrast
   `δ_{k,t} = log π_θ(y_{k,t} | h⁺_{k,t}) − log π_θ(y_{k,t} | h_{k,t})`
   (student log-probs already available from rollout; only the teacher branch
   is an extra forward). `sg[·]` — no gradient flows through the teacher.
5. **Turn aggregation.** `e_k = Σ_t δ_{k,t}` — exactly the log-likelihood
   ratio `log [π(a_k|s_k,c⁺)/π(a_k|s_k)]` of the whole sampled action.
6. **Recursive belief update** (turn boundaries only):
   - `B₀ = clip(R̄, ε₀, 1−ε₀)` where `R̄ = S/G` = group success fraction
     (binomial MLE, Prop. 7; clip keeps log-odds finite in degenerate groups).
   - `c₀ = 0`; `c_k = γ·c_{k−1} + e_k`;
     `ℓ_k = logit(B₀) + c_k`; `B_k = σ(ℓ_k)`.
   - Note the prior `logit(B₀)` is **retained at every step**; only the
     evidence accumulator decays with γ.
7. **Credit & alignment.** `ΔB_k = B_k − B_{k−1}` (exact difference, per
   Algorithm 1); `q_k = sign(A_seq)·ΔB_k`. On a success, upward revisions are
   consistent; on a failure the same upward revision is inconsistent — the sign
   does that bookkeeping.
8. **Bounded reshaping.** Within trajectory: `μ_q, σ_q` over turns;
   `z_k = (q_k − μ_q)/(σ_q + ε₀)`; `w_k = clip(1 + b·z_k, 1−b, 1+b)`;
   `Ã_k = A_seq·((1−λ) + λ·w_k)`. Edge case: K=1 ⇒ z=0 ⇒ w=1 ⇒ Ã = A_seq
   (graceful pure-GRPO fallback).
9. **Update.** Every token `t` of turn `k` inherits `Ã_{κ(t)}`; standard
   clipped surrogate with importance ratio `r_{t} = π_θ/π_old`, asymmetric
   clipping, KL penalty toward reference.

### 3.4 Loss function

**No separate distillation loss.** The teacher signal enters *only* through
the advantage:

```
L = −(1/G) Σ_i (1/Σ_t M_{i,t}) Σ_t M_{i,t} · min( r_{i,t}·Ã_{κ(t)},
        clip(r_{i,t}, 1−ε_low, 1+ε_high)·Ã_{κ(t)} )  + β·L_KL(π_θ ‖ π_ref)
```

with `M_{i,t}` the response-token mask. Asymmetric clip (ε_low=0.2,
ε_high=0.24 — "clip-higher"), dual-clip c=3.0, one PPO epoch.

## 4. Math

All equations in my notation; plain-English reading for each.

**(1–2) GRPO advantage.**
`A⁽ⁱ⁾ = (R⁽ⁱ⁾ − R̄)/(σ̂_R + ε₀)`, `R̄ = (1/G)Σ_j R⁽⁲⁾`.
*Reads:* "compared to its siblings, how good was this trajectory" — normalized
so advantages are zero-mean, unit-ish scale per group. Every token of
trajectory i gets this same number.

**(3) Bayes factor identity (hindsight evidential view).** Let C = "episode
eventually succeeds":
`logit p(C|s_k,a_k) − logit p(C|s_k) = log [ p(a_k|s_k,C) / p(a_k|s_k,¬C) ]`.
*Reads:* the change in success log-odds caused by taking action a_k equals the
log-ratio of the action's likelihood under success vs failure futures. The
ideal credit signal — but those conditionals are intractable, so *estimate*.

**(4–5) Self-distillation contrast.** Teacher context `h⁺ = (s_k, c⁺, y_{<t})`
vs student context `h = (s_k, y_{<t})`; per-token
`δ_t = log π(y_t|h⁺) − log π(y_t|h)`, detached. *Reads:* "how much more does
the policy believe in this exact sampled token when a useful skill is in
context?" — same weights, two contexts, so the gap isolates the skill's effect.

**(6) Turn evidence.** `e_k = Σ_t δ_t = log [π(a_k|s_k,c⁺) / π(a_k|s_k)]`.
*Reads:* whole-action log-likelihood ratio; positive iff the skill endorses
the action the policy actually took.

**(7) Approximation theorem (Appendix A.1).** Under assumption A1
(skill-conditioned ≈ success-conditioned, `π(a|s,c⁺) ≈ p(a|s,C)`), `e_k`
equals the pointwise mutual information `log[p(C|s,a)/p(C|s)]`; with the extra
assumption A2 (success rate ρ small ⇒ marginal ≈ failure-conditional),
`e_k ≈ B_k → B_k` correction that is **monotone**, so sign and within-set
ranking are preserved. *Reads:* the proxy may be biased, but the method only
consumes sign + ranking, which the bias cannot flip. A1 is the honest,
load-bearing hand-wave.

**(8) Recursive belief.** `B₀ = clip(S/G, ε₀, 1−ε₀)`; `c_k = γ·c_{k−1} + e_k`;
`ℓ_k = logit(B₀) + c_k`; `B_k = σ(ℓ_k)`. *Reads:* start from "how often does
the current policy solve this task" (the group success rate — a better prior
than 0.5), then add up decaying evidence in log-odds space where independent
evidence simply sums. γ=1 ⇒ Wald's SPRT; γ<1 ⇒ recency-weighted, so stale
evidence can't permanently pin the belief. B_k is "relative support", not a
calibrated probability.

**(9) Marginal credit.** `ΔB_k = B_k − B_{k−1}`; first-order expansion
`ΔB_k ≈ B_{k−1}(1−B_{k−1})·e_k − (1−γ)c_{k−1}`. *Reads:* credit = new
evidence net of decayed carry-over, **gated by uncertainty** — the sigmoid
derivative B(1−B) is maximal at B=½ and vanishes near 0/1, so identical
evidence counts when the outcome is still open and counts ~nothing once the
belief is saturated. This gating is exactly what "raw gap" credit lacks.
(Implementation uses the exact difference.)

**(10) Outcome alignment.** `q_k = sign(A_seq)·ΔB_k`. *Reads:* keep the
magnitude of the revision, but decide its polarity from the verifier's actual
verdict — an upward revision is good news on a success and bad news on a
failure.

**(11) Bounded reshaping.** `z_k = (q_k−μ_q)/(σ_q+ε₀)`;
`w_k = clip(1+b·z_k, 1−b, 1+b)`; `Ã_k = A_seq·((1−λ)+λ·w_k)`. *Reads:*
standardize credits *within* the trajectory (so only relative turn importance
matters), squash into a multiplicative band, then blend with the uniform GRPO
advantage. Proven properties: **bounded** (|Ã−A| ≤ λb|A|), **sign-preserving**
((1−λ)+λw ≥ 1−λb > 0 ⇒ the update direction is never reversed), **recovery**
(λ=0 ⇒ exact GRPO), and **telescoping** (Σ_k ΔB_k = B_K − B₀: the credit
budget equals total belief movement).

**(12) Clipped policy loss.** Standard PPO/GRPO surrogate (Section 3.4 above)
+ β·KL. *Reads:* nothing exotic — all novelty lives in Ã.

**Prop. 6 (why per-turn signals are needed at all):** two trajectories with
identical returns can have completely different per-turn contribution patterns
(concentrated vs spread). *Reads:* per-turn credit is not identifiable from
the return alone — you must inject an extra per-turn signal; AgentOPSD's
choice is to synthesize it from the policy itself.

## 5. Training

- **Models:** Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct; FSDP across 8×H800.
- **Environments:** ALFWorld (6 household task types, ≤50 turns), WebShop
  (≤15 turns, 128 fixed validation tasks of Feng et al. 2025), Search-QA
  (Search-R1 setup, 7 datasets, ~4 turns).
- **Privileged info:** skills from SkillRL's SkillBank, retrieved by keyword
  matching, used only during training (inference is skill-free).
- **Hyperparameters (one shared config, no per-task tuning; Table 3–4):**
  lr 1e−6, group size G=8, λ=0.5, b=0.2, γ=0.95, KL coef 0.01,
  clip ε_low/ε_high = 0.2/0.24, dual-clip c=3.0, grad clip 1.0, entropy coef
  0.001, 1 PPO epoch, ε₀=1e−4. Environment side: 150 steps, batch 16/16/128
  (ALFWorld/WebShop/Search-QA), max turns 50/15/4, max response 512 tokens,
  temperature 1.0 train / 0.4 eval.
- **Tricks that matter:** teacher gaps sg-detached (no gradient through the
  skill branch); B₀ anchored per group (not a constant); exact ΔB (not the
  approximation); standardization of q is *within-trajectory*.
- **Compute:** paper-scale is 8×H800; the method's *overhead* is one extra
  teacher forward per trajectory plus O(K) elementwise ops.

## 6. Results & Ablations

**Headline (Qwen2.5-7B):**
- ALFWorld: **89.1** vs GRPO 81.2, SDAR 85.9, StepOPSD 88.4, RLSD 82.0,
  Skill-SD 85.1.
- Search-QA avg: **49.2** vs GRPO 42.0.
- WebShop: 90.2 score / 79.7 acc vs GRPO 80.9 / 72.6.
- At 3B: ALFWorld 84.4, Search-QA 46.7 (vs GRPO 36.4), WebShop 90.4/69.5.

**Horizon robustness (the money plot, Fig 1b):** success lost per extra turn
on ALFWorld-7B: AgentOPSD **−0.54** vs GRPO −2.91, RLSD −3.59. The flattest
curve by far; consistent with the mechanism (credit concentrated where it's
earned instead of amortized uniformly). Counter-check: on short-horizon
Search-QA (4 turns) hyperparameter spreads collapse — the method does
something precisely where long-horizon credit assignment is needed.

**Controlled information:** AgentOPSD and the privileged baselines see the
SAME retrieved skills and differ only in *how the gap enters learning*.
Beats GRPO+OPSD / Skill-SD / RLSD on all 8 aggregate comparisons, SDAR on
6/8. The gain is from credit construction, not privileged access.

**Ablations (Table 2, ALFWorld-7B, full method 89.1):**

| Component changed | Score | Δ |
|---|---|---|
| per-token accumulation (vs per-turn) | 85.9 | −3.2 |
| raw local gap e_k instead of ΔB_k | 82.8 | −6.3 |
| magnitude \|ΔB_k\| only (drop outcome sign) | 80.5 | −8.6 |
| drop B₀ empirical anchor | 78.9 | −10.2 |

*Reading:* the two biggest hits come from removing the outcome sign and the
group-rate prior — i.e. **directional correctness and state anchoring matter
more than recursion itself**. Recursion (e_k → ΔB_k) is the central *conceptual*
step (a local gap is not sequential credit) and still worth −6.3; turn
granularity is the smallest but consistent (the environment only responds to
whole actions). Also from Fig 1c: AgentOPSD maintains higher policy entropy
during training than GRPO (less premature collapse).

**Sensitivity (§3.4):** only λ is systematic — λ=0.5 best (89.1 vs
84.4/85.9/83.6 for 0.25/0.1/0.01); smaller λ throws away the turn-level
credit. γ ∈ {1.0, 0.95, 0.9, 0.8} within a few points, no trend;
ε_high ∈ {0.2, 0.24, 0.28} nearly flat.

## 7. Limitations

- **A1 is a proxy, not a theorem.** The whole evidence chain assumes
  skill-conditioned ≈ success-conditioned behavior. With a weak or mismatched
  skill, e_k measures "skill endorsement", not success evidence. The paper
  shows sign/ranking robustness *under* A1 but never measures sensitivity to
  skill quality — no skill-ablation experiment. This is the genuine open gap.
- **B_k is not calibrated.** It's relative support from a self-teacher; the
  paper says so, but anything downstream that interprets it as a probability
  would be wrong.
- **Needs a privileged context at training time.** SkillBank/SkillRL infra
  (or some c⁺) is a real dependency; domains without retrievable skills must
  invent one.
- **Binary terminal rewards only.** All math (sign(A_seq), group success rate
  prior) leans on the {0,1} reward structure; graded/continuous rewards would
  need rework.
- **Small-scale, single-family evidence:** Qwen2.5 3B/7B on three benchmarks,
  shared hyperparameter config; no 30B+/other-family replication, and
  WebShop gains are modest. Search-QA (4 turns) mostly shows it "does no harm"
  at short horizons rather than helping.
- **Overhead is small but not zero:** one extra (long-context) forward pass
  per trajectory per iteration.

## 8. Open Questions / Ideas

- **Skill-quality sensitivity:** rerun ALFWorld with (a) shuffled/wrong
  skills, (b) empty skills, (c) oracle-perfect skills. How much of the +7.9
  over GRPO survives each? My guess: sign-preserving bounded reshaping degrades
  gracefully to ~GRPO (w_k becomes noise, λ·w averages out) — testable in the
  toy implementation.
- **Is the belief prior load-bearing at 3B too, or mostly a 7B effect?**
  The −10.2 ablation is 7B-only in the paper.
- **Replace SPRT decay with learned decay:** γ fixed at 0.95 everywhere; would
  a per-task γ (or normalizing evidence by turn length — long turns accumulate
  more δ by construction) help? e_k currently scales with L_k, which feels like
  a confound worth controlling.
- **Combine with GiGPO:** self-distillation evidence (policy-internal) and
  anchor-state group advantages (environment-external) are orthogonal signal
  sources — what happens when both enter the belief?
- **Calibration drift:** does ΔB's relative scale stay comparable across
  training as the policy absorbs skills (Appendix G shows mean δ̄ shrinking)?
  A running normalization might be needed for very long runs.
- **Toy-implementation angle (my plan):** a synthetic N-turn task with known
  pivotal turns lets me measure *credit localization error* directly — the
  metric the real paper can't compute because ALFWorld ground-truth credit
  doesn't exist.
