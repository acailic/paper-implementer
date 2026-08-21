# AgentOPSD: Recursive Self-Distillation for Agentic RL — Notes

> **Paper**: "AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning"
> **Authors**: Zi-Han Wang, Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Jie Wu, Zhengzhou Cai,
> Yueqing Sun, Ziang Ye, Linji Hao, Qi Gu, Xunliang Cai, Yongliang Shen, Yujiu Yang
> (Tsinghua University, Zhejiang University, Meituan)
> **arXiv**: 2608.05987 (Aug 2026) · **Code**: https://github.com/ZethWang/AgentOPSD
>
> Pass 1: first impressions (kept below). Pass 2: deep read, section by section,
> including Appendices A–G. Open questions from pass 1 are all resolved (see
> "Open questions — RESOLVED (pass 2)").

---

## First impressions (pass 1)

A clean, tightly-scoped methods paper about **credit assignment** in multi-turn agentic RL.
The writing is unusually precise for an RL-for-LLMs paper — everything reduces to a handful
of small equations. The core idea is elegant and, on reflection, almost obvious in hindsight
(which is usually a sign of a good paper): **a local teacher–student log-prob gap is NOT
credit; credit is how much that gap *changes* a running belief that the trajectory will
succeed.**

## The problem, in my own words

When you train an LLM agent with RL (GRPO-style) on environments like ALFWorld or WebShop,
the reward is usually a single binary success/failure signal given **once per trajectory**.
GRPO computes one advantage per trajectory (reward minus group mean, divided by group std)
and **broadcasts that same advantage to every token in the trajectory**.

That's wasteful and noisy: in a 30-turn episode, maybe 2–3 turns were pivotal (the right
search query, the right object pick-up), while the rest were routine. Uniform credit drowns
the pivotal turns in noise, and the problem gets *worse* the longer the horizon.

Recent "on-policy self-distillation" (OPSD) methods provide a denser local signal: compare
the policy's own log-prob for its sampled action, with vs. without some privileged
conditioning (here: a retrieved "skill" text). But that signal is token-local and
turn-myopic — it doesn't know (a) that tokens form whole actions that the environment
responds to only at turn boundaries, nor (b) what evidence earlier turns already accumulated.

## The core idea (one paragraph)

Treat each turn's aggregated teacher–student gap as **Bayesian evidence** about eventual
success. Maintain a belief `B_k` = probability the trajectory will succeed, expressed in
**log-odds space** where evidence just adds. Initialize the belief from the GRPO group
success rate (`B_0 = clip(R̄, ε, 1−ε)`). At each turn, add the turn's evidence `e_k`
(with geometric decay γ on the accumulator). The **credit of turn k is the marginal belief
revision ΔB_k = B_k − B_{k−1}**, signed by the trajectory's outcome advantage. Turns whose
evidence *moved* the belief get more of the trajectory's advantage; turns whose evidence was
redundant (belief already saturated) get less. Then the reshaped per-turn advantage is a
**bounded** blend: `Ã_k = A_seq · ((1−λ) + λ·w_k)`, where `w_k ∈ [1−b, 1+b]` is a clipped
standardized version of the signed credit — so it can only *modulate magnitude*, never flip
the direction GRPO chose. No critic, no extra rollouts, one extra teacher forward pass.

---

# PASS 2 — DEEP READ

## Method walkthrough (Section 2, equation by equation)

### 2.1 Problem setup

- Task `x`, initial observation `o_0`, first state `s_1 = (x, o_0)`.
- At turn `k` the policy samples a multi-token action `a_k = (y_{k,1..L_k}) ~ π_θ(·|s_k)`;
  environment returns `o_k`; history grows: `s_{k+1} = (s_k, a_k, o_k)`.
- Episode `τ` with `K` turns gets a **binary** terminal reward `R(τ) ∈ {0,1}`.
- GRPO: sample `G` trajectories per task; `A_seq^(i) = (R^(i) − R̄)/(σ̂_R + ε_0)` with
  `R̄ = (1/G)ΣR^(j)`; every token in trajectory `i` gets the same `A_seq^(i)` (Eq. 1–2).
  `ε_0` reused throughout as a numerical stabilizer.

### 2.2 From outcome contribution to Bayesian turn evidence (Eqs. 3–7)

The counterfactual contribution of turn k would require marginalizing the outcome over all
continuations after `a_k` — intractable. Instead: **hindsight evidential view**. Let `C` =
"trajectory eventually succeeds". Bayes' rule on the action side (Eq. 3):

```
logit p(C|s_k,a_k) − logit p(C|s_k) = log [ p(a_k|s_k,C) / p(a_k|s_k,¬C) ]
```

The RHS is the ideal **Bayes factor** between success-conditional and failure-conditional
action likelihoods. Its *sign* says whether the action raises or lowers support for success.

Those outcome-conditional distributions aren't available, so estimate retrospectively with a
**self-distillation contrast** — same weights θ, same sampled action, two contexts (Eq. 4):

```
h_{k,t}   = (s_k, y_{k,<t})          # student context
h⁺_{k,t} = (s_k, c⁺, y_{k,<t})      # teacher context (c⁺ = retrieved skill, training-only)
```

`c⁺` is a skill from SkillBank (SkillRL, Xia et al. 2026) describing useful subgoals and
action patterns — retrieved by **keyword matching**. The skill-conditioned branch *approximates*
success-associated behavior; the unconditioned branch is the background.

Per-token detached contrast (Eq. 5): `δ_{k,t} = log π_θ(y_{k,t}|h⁺_{k,t}) − log π_θ(y_{k,t}|h_{k,t})`.
Sum over the turn (Eq. 6):

```
e_k = Σ_t δ_{k,t} = log [ π_θ(a_k|s_k,c⁺) / π_θ(a_k|s_k) ]
```

— a log-likelihood ratio. Eq. 7 (pure Bayes identity): this equals
`log [p(C|s_k,a_k)/p(C|s_k)]` **if** the teacher branch really were success-conditional.
So `e_k` is a *Bayesian-inspired evidence proxy*: sign-consistent, ranking-preserving
(see Appendix A.1 below). Important: `δ` is **detached** (`sg[...]` in Algorithm 1) — the
teacher signal enters only through the advantage, never through gradients.

### 2.3 Recursive belief update (Eq. 8–9)

A local `e_k` doesn't say whether the evidence is *pivotal or redundant* given earlier turns.
So keep a decaying evidence accumulator (Eq. 8):

```
B_0 = clip(R̄, ε_0, 1−ε_0),  c_0 = 0
c_k = γ·c_{k−1} + e_k
ℓ_k = logit(B_0) + c_k = logit(B_0) + Σ_j γ^{k−j} e_j
B_k = σ(ℓ_k)
```

Key details:
- `R̄ = S/G` = fraction of successes in the GRPO group — the MLE of task success probability
  under Binomial(G, θ) (Prop. 7). Clip ε_0 = 1e−4 keeps log-odds finite for all-success /
  all-fail groups — which have zero group-relative advantage anyway, so they don't update.
- **Only the evidence accumulator c_k decays with γ; the prior logit(B_0) is retained at
  every step.** γ=1 recovers Wald's (1945) SPRT log-likelihood-ratio accumulator; γ<1 makes
  the state recency-weighted so ancient evidence doesn't pin the support level.
- Since `e_k` is a self-teacher proxy, `B_k` is "relative support", NOT a calibrated
  success probability.

Credit of turn k = marginal support revision (Eq. 9):

```
ΔB_k = B_k − B_{k−1} = σ(ℓ_k) − σ(ℓ_{k−1})
     ≈ B_{k−1}(1−B_{k−1})·e_k − (1−γ)·c_{k−1}          (first-order, for intuition)
```

Increment in log-odds: `ℓ_k − ℓ_{k−1} = e_k − (1−γ)c_{k−1}` = new evidence net of decayed
carry-over, gated by sigmoid-derivative `B_{k−1}(1−B_{k−1})`: **evidence matters most under
uncertainty, is suppressed when support saturates**. Prop. 4 formalizes this (ΔB =
B(1−B)Δℓ + O(Δℓ²), gate maximal at B=½, vanishes at 0/1). **In practice the EXACT
difference is computed** (Algorithm 1 line 14: `ΔB_k ← B_k − B_{k−1}`); the approximation
is only for interpretation. Updates happen at **turn boundaries**; a token-level variant
exists only as the granularity ablation.

Outcome-aligned credit (Eq. 10): `q_k = sign(A_seq)·ΔB_k`.
Magnitude = how much support the turn revises; sign = whether the revision agrees with the
verifier outcome. On a successful trajectory an upward revision is consistent; on a failed
one the same upward revision is inconsistent — this distinction is exactly what the
magnitude-only ablation loses.

### 2.4 Bounded advantage reshaping (Eq. 11–12)

Within trajectory i (K_i turns): standardize `q`, clip to a multiplier band, blend (Eq. 11):

```
μ_q = mean(q_1..K),  σ_q = std(q_1..K)
z_k = (q_k − μ_q)/(σ_q + ε_0)
w_k = clip(1 + b·z_k, 1−b, 1+b),      b ∈ (0,1)
Ã_k = A_seq · ((1−λ) + λ·w_k),        λ ∈ [0,1]
```

Every token t of turn k inherits `Ã_{κ_i(t)}`. Loss = standard PPO/GRPO clipped objective
(Eq. 12): `L = −(1/G)Σ_i (1/ΣM)Σ_t M_{i,t}·min(r_{i,t}·Ã, clip(r_{i,t},1−ε,1+ε)·Ã) + β·L_KL`
with importance ratio `r_{i,t} = π_θ/π_θold`. **No separate distillation loss** — the teacher
signal acts only through Ã. (Note from Table 3: asymmetric clip ε_low=0.2, ε_high=0.24 —
i.e. clip-higher (Yu et al. 2025); plus dual-clip c=3.0, entropy coef 0.001, 1 PPO epoch.)

Properties proven in Appendix A.2 (all trivial but nice to have on record):
- **P1 Boundedness**: |Ã_k − A| = |A|·λ|w_k−1| ≤ λb|A| → magnitude can only move within
  (1±λb) of GRPO's.
- **P2 Sign preservation**: (1−λ)+λw_k ≥ 1−λb > 0 ⇒ sign(Ã)=sign(A) always. Never reverses
  GRPO's update direction.
- **P3 Recovery**: λ=0 ⇒ Ã ≡ A (exact GRPO gradient).
- **P5 Telescoping**: Σ_k ΔB_k = B_K − B_0 (the idealized credit budget equals total belief
  movement).
- **P6 Non-identifiability**: two trajectories with equal returns can have completely
  different per-turn contributions (concentrated vs. spread) — per-turn credit is NOT
  identifiable from the return alone, justifying an extra per-turn signal.

## Algorithm 1 (Appendix B) — the whole method in 22 lines

For each iteration: batch of tasks → per task, retrieve skill c⁺ → sample G trajectories →
verifier rewards → A_seq per trajectory → per trajectory: B_0 from group success rate,
c_0=0 → per turn: e_k from ONE extra teacher forward (sg-detached), recursive c/ℓ/B/ΔB →
q_k = sign(A_seq)·ΔB_k → within-trajectory standardize → clip to w_k → blend Ã_k → tokens
inherit → clipped GRPO update. **Overhead over GRPO = one teacher forward pass per
trajectory + elementwise ops. No new parameters, no extra rollouts, no critic.**

## Experiments (Section 3)

Setup: Qwen2.5-3B/7B-Instruct, 8×H800. Environments:
- **ALFWorld** (6 household task types; up to 50 turns)
- **Search-QA** (Search-R1 setup; 7 datasets; only 4 turns — short horizon)
- **WebShop** (interactive shopping; ≤15 turns; 128 fixed validation tasks)

Headline numbers (Qwen2.5-7B): ALFWorld **89.1** vs GRPO 81.2, SDAR 85.9, StepOPSD 88.4,
RLSD 82.0, Skill-SD 85.1. Search-QA avg **49.2** vs GRPO 42.0. WebShop 90.2/79.7 vs
80.9/72.6. At 3B: ALFWorld 84.4, Search-QA 46.7 (vs GRPO 36.4 — huge), WebShop 90.4/69.5.

**"The gain comes from credit construction, not privileged access"** — AgentOPSD and the
privileged baselines see the SAME retrieved skills; they differ only in how the gap enters
learning. AgentOPSD beats GRPO+OPSD/Skill-SD/RLSD on all 8 aggregate comparisons and SDAR
on 6/8. This is the paper's controlled-information argument.

**Horizon robustness** (Fig 1b): success points lost per extra turn on ALFWorld-7B:
AgentOPSD **−0.54** vs GRPO −2.91, RLSD −3.59. Flattest by far — the money plot.
Consistent story: on short-horizon Search-QA (4 turns) the hyperparameter spreads collapse —
the method acts where long-horizon credit assignment is actually needed.

## Ablations (Table 2, ALFWorld-7B; full method 89.1)

| Component removed | Score | Δ |
|---|---|---|
| per-token accumulation instead of per-turn | 85.9 | −3.2 |
| raw local gap e_k instead of ΔB_k | 82.8 | −6.3 |
| magnitude \|ΔB_k\| only (drop outcome sign) | 80.5 | −8.6 |
| drop B_0 empirical anchor | 78.9 | −10.2 |

Reading: **signed direction and the state prior matter most**; recursion and granularity are
smaller but consistent. Interpretations from the paper:
- Token-level fragments a single decision; environment feedback is per-action.
- Raw e_k scores turns in isolation; the same gap is decisive while the outcome is open but
  redundant once the accumulated state already points to an outcome. (This is THE central
  principle: *a local gap is not sequential credit*.)
- Magnitude-only can't tell whether a revision agrees with the verifier outcome.
- B_0 anchors both the initial log-odds AND the operating region of the B(1−B) gate;
  without it early revisions are mis-scaled and which early turns look pivotal gets distorted.

## Hyperparameters (Appendix F, Tables 3–4) — single shared config, no per-task tuning

η=1e−6, G=8, clip ε_low/ε_high=0.2/0.24, α_KL=0.01, **λ=0.5, b=0.2, γ=0.95**, skill
retrieval = keyword matching. Dual-clip c=3.0, grad clip 1.0, entropy 0.001, 1 PPO epoch,
FSDP. Environment configs: 150 steps everywhere; batch 16/16/128; max turns 50 (ALFWorld) /
15 (WebShop) / 4 (Search-QA); max response 512; train/val temperature 1.0/0.4.

Sensitivity (§3.4): only **λ** has a systematic effect (0.5 best; smaller = worse).
γ ∈ {1.0,0.95,0.9,0.8} within a few points, no trend. ε_high ∈ {0.2,0.24,0.28} nearly flat.

## Baseline taxonomy (Appendix D) — what each injects and where

- **OPSD**: teacher branch w/ privileged context → dense token-level distillation targets
  (distribution matching), detached. RL-free.
- **GRPO+OPSD**: just add the two losses.
- **Skill-SD**: skill only in teacher branch; student absorbs via importance-weighted
  distillation loss.
- **RLSD**: gap → bounded coefficient scaling each token's GRPO magnitude; sign still from
  outcome advantage. (Closest in spirit to AgentOPSD, but token-local, no recursion.)
- **SDAR**: auxiliary gated self-distillation loss on top of unchanged GRPO advantage.
- **StepOPSD**: distillation signal at step (turn) level — but each step's local log-ratio
  in isolation. (The "aggregate but don't recurse" point on the design axis.)
- **GiGPO** (related work): turn-level credit from repeated anchor states across
  trajectories (environment-reward-based) — complementary signal source to AgentOPSD's
  self-distillation evidence.

Design axis that emerges: {token, step, recursive-belief} × {aux loss, magnitude gate,
advantage reshape}. AgentOPSD = turn granularity + recursion + bounded signed reshaping.

---

## Open questions — RESOLVED (pass 2)

- [x] **A.1 approximation quality**: two assumptions. (A1) skill-conditioned ≈
      success-conditional `π(a|s,c⁺) ≈ p(a|s,C)`; (A2) when success is rare (ρ_k small),
      the marginal is failure-dominated: `π(a|s) = ρ_k p(a|s,C) + (1−ρ_k)p(a|s,¬C) ≈ p(a|s,¬C)`.
      Substituting: `e_k ≈ B_k − log(1−ρ_k+ρ_k e^{B_k}) → B_k` as ρ_k→0. Under (A1) ALONE,
      `e_k` = pointwise mutual information `log[p(a|s,C)/p(a|s)] = log[p(C|s,a)/p(C|s)]` —
      positive iff a_k raises the posterior. The correction is monotone in B_k, so
      **sign(e_k)=sign(B_k) and ranking preserved** — and the method only uses e_k through
      sign and (within-trajectory) ranking, which is why the proxy suffices. Verdict:
      assumption A1 is the load-bearing, honestly-hand-wavy bit ("skill ≈ success
      conditioning"), but the downstream use is robust to the approximation.
- [x] **Prop. 7**: B_0 = S/G is the binomial MLE; clip only for numerical safety.
- [x] **Appendix B**: full pseudocode read (summarized above); baseline injection points
      cataloged (Appendix D).
- [x] **Skill retrieval**: keyword matching against SkillBank of SkillRL; training-only;
      inference uses no skills. Sensitivity to skill quality is NOT ablated in the paper —
      a genuine open gap (see limitations).
- [x] **Exact vs approximate ΔB**: exact difference is computed in practice
      (Algorithm 1); the first-order form is intuition/analysis only.
- [x] **Few-turn edge case (K_i = 1 or 2)**: with K=1, z_k = (q−μ)/(σ+ε) = 0/(0+ε_0) = 0 →
      w=1 → Ã = A_seq (pure GRPO fallback — safe). With K=2 the standardization just
      contrasts the two turns. Graceful.
- [x] **Table 3**: captured above (λ=0.5, b=0.2, γ=0.95, G=8, η=1e−6, α_KL=0.01,
      ε 0.2/0.24).
- [x] **Entropy**: not analyzed directly; Fig 1c shows AgentOPSD maintains higher policy
      entropy than GRPO during training; Appendix G tracks teacher–student gap (mean δ̄ is
      slightly negative and shrinks in magnitude over training — interesting: the skill
      initially *decreases* likelihood of sampled actions on average).
- [x] **Detachment**: confirmed — `sg[...]` in Algorithm 1 line 13; teacher pass is pure
      signal.

## Terms / background (from pass 1, now clearer)

- **OPSD** (Zhao et al. 2026): privileged self-distillation, teacher = same weights +
  privileged context. **SDAR** (Lu et al. 2026b): GRPO + gated auxiliary SD loss (strongest
  baseline). **RLSD** (Yang et al. 2026a): gap as bounded magnitude coefficient.
  **StepOPSD** (Zhang et al. 2026): step-span local log-ratio. **GiGPO** (Feng et al. 2025):
  anchor-state group advantages from environment rewards.
- **SPRT** (Wald 1945): γ=1 case is literally the sequential probability-ratio accumulator.
- **Bayes factors** (Kass & Raftery 1995); belief updates in log-odds (Åström 1965;
  Kaelbling et al. 1998).
- **SkillRL/SkillBank** (Xia et al. 2026): the skill library.
- Lineage the paper itself draws (§4.3): belief state ↔ GAE value baseline; ΔB ↔ TD signal;
  but critic-free. RUDDER/VinePPO/PRMs = the "pay with rollouts/scorers" alternative.

## Why I think this is implementable as a toy

The whole method is: (1) a policy that emits multi-token actions per turn, (2) a privileged
"skill" context, (3) two forward passes (with/without skill) → per-token log-prob gaps,
(4) a tiny recursive belief accumulator in log-odds space, (5) a bounded advantage
reshaper, (6) a PPO-style clipped loss. All of this can be built with a small GRU/transformer
policy on a synthetic multi-turn environment with binary terminal reward, where I *know*
which turns are pivotal — so I can actually measure whether AgentOPSD's credit localizes
better than uniform GRPO credit. Concretely for `implementation/`:
- Environment: synthetic N-turn task where one (or two) specific turn(s) decide success
  (e.g. a keyed multi-armed bandit with a hidden sequence), binary terminal reward only.
- Policy: tiny transformer/GRU emitting multi-token actions per turn.
- Teacher branch: same policy, input augmented with a "skill" hint (the privileged info —
  e.g. the correct key at that turn, as a learned embedding prepended to context).
- Compare: GRPO vs GRPO+raw-gap (RLSD-ish) vs AgentOPSD, plus the Table-2 ablations
  (per-token, magnitude-only, no B_0 anchor). Metrics: task success AND credit-localization
  error vs. ground-truth pivotal turns. This is the plan for the coding step.
