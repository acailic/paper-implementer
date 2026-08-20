# AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning — First-Pass Notes

> **Paper**: "AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning"
> **Authors**: Zi-Han Wang, Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Jie Wu, Zhengzhou Cai,
> Yueqing Sun, Ziang Ye, Linji Hao, Qi Gu, Xunliang Cai, Yongliang Shen, Yujiu Yang
> (Tsinghua University, Zhejiang University, Meituan)
> **arXiv**: 2608.05987 (Aug 2026) · **Code**: https://github.com/ZethWang/AgentOPSD

---

## First impressions (after one skim)

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
success rate (`B_0 = clip(R̄, ε, 1-ε)`). At each turn, add the turn's evidence `e_k`
(with geometric decay γ on the accumulator). The **credit of turn k is the marginal belief
revision ΔB_k = B_k − B_{k−1}**, signed by the trajectory's outcome advantage. Turns whose
evidence *moved* the belief get more of the trajectory's advantage; turns whose evidence was
redundant (belief already saturated) get less. Then the reshaped per-turn advantage is a
**bounded** blend: `Ã_k = A_seq · ((1−λ) + λ·w_k)`, where `w_k ∈ [1−b, 1+b]` is a clipped
standardized version of the signed credit — so it can only *modulate magnitude*, never flip
the direction GRPO chose. No critic, no extra rollouts, one extra teacher forward pass.

## Key numbers from the skim

- **ALFWorld** (Qwen2.5-7B-Instruct): AgentOPSD **89.1%** vs GRPO 81.2, SDAR 85.9,
  StepOPSD 88.4, RLSD 82.0.
- **Search-QA avg** (3B): 46.7 vs GRPO 36.4. (7B: 49.2 vs 42.0.)
- **WebShop Score/Acc** (3B): 90.4 / 69.5 vs GRPO 79.8 / 63.3.
- **Horizon robustness** (Figure 1b): success points lost per extra turn — AgentOPSD
  **−0.54** vs GRPO −2.91, RLSD −3.59. This is the money plot for the "long-horizon" claim.
- Ablations (ALFWorld 7B): full 89.1 → per-token accumulation 85.9 → raw local gap 82.8 →
  magnitude-only (no outcome sign) 80.5 → no B₀ anchor 78.9. Signed direction and the
  empirical prior matter most.
- Hyperparams: λ = 0.5 (reshaping weight, clearest knob), γ = 0.95 (evidence decay,
  insensitive), ε_high = 0.24 (clip), b bounds the multiplier.

## Mechanics I want to remember (equations as I read them)

1. **Turn evidence** (Eq. 5–6): for token `y_{k,t}`, gap
   `δ_{k,t} = log π_θ(y | s_k, c⁺, y_{<t}) − log π_θ(y | s_k, y_{<t})`; sum over the turn:
   `e_k = Σ_t δ_{k,t} = log [π(a_k | s_k, c⁺) / π(a_k | s_k)]` — a log-likelihood ratio,
   i.e. a **Bayes factor** approximating the idealized `logit P(C|s,a) − logit P(C|s)` (Eq. 3).
   Same weights θ for both branches (self-distillation); teacher branch just also sees the
   retrieved skill `c⁺` (training-only).
2. **Recursive belief** (Eq. 8): `c_k = γ·c_{k−1} + e_k`, `ℓ_k = logit(B₀) + c_k`,
   `B_k = σ(ℓ_k)`. The *accumulator* decays; the prior logit does not.
3. **Credit** (Eq. 9–10): `ΔB_k ≈ B_{k−1}(1−B_{k−1})·e_k − (1−γ)·c_{k−1}` (first-order);
   `q_k = sign(A_seq)·ΔB_k`. The `B(1−B)` factor is a sigmoid-derivative gate: evidence
   matters most under uncertainty.
4. **Bounded reshaping** (Eq. 11): standardize `q_k` within the trajectory → `z_k`; clip
   `w_k = clip(1 + b·z_k, 1−b, 1+b)`; final per-turn advantage
   `Ã_k = A_seq·((1−λ) + λ·w_k)`. Always positive multiplier → never reverses GRPO direction.
5. **Loss** (Eq. 12): standard PPO/GRPO clipped objective with token→turn mapping
   `κ_i(t)`, plus KL term. No separate distillation loss — the teacher signal enters *only*
   through the advantage.

## Things I don't yet understand / to dig into on the second pass

- [ ] **Appendix A.1** — the formal justification for why the skill-conditioned
      self-teacher approximates the success-conditional likelihood `p(a_k | s_k, C)`.
      This feels like the weakest link; how hand-wavy is it?
- [ ] **Prop. 7** mentioned re: `B₀ = R̄` being "the standard GRPO group mean" — read the
      propositions in the appendix.
- [ ] **Appendix B** — full algorithm pseudocode for all baselines (SDAR, RLSD, StepOPSD,
      GRPO+OPSD, Skill-SD) so I can pin down exactly what each injects and where.
- [ ] How exactly the **skill `c⁺` is retrieved** (SkillBank of SkillRL, keyword matching) —
      and how sensitive results are to skill quality. Could ANY privileged conditioning work?
- [ ] Eq. 9's second form: is the first-order approximation `ΔB ≈ B(1−B)e − (1−γ)c` used in
      practice, or the exact `σ(ℓ_k) − σ(ℓ_{k−1})`? (Looks like exact is computed; the
      approximation is for intuition.)
- [ ] What is `σ̂_q` standardization doing when a trajectory has very few turns (K_i = 1 or 2)?
      Edge-case behavior.
- [ ] **Appendix F Table 3** — full hyperparameters (LR, batch, group size G, KL coef β, b).
- [ ] How the entropy curve (Fig 1c) differs from GRPO — does belief reshaping implicitly
      regularize entropy?
- [ ] Whether `e_k` uses **detached** log-probs everywhere (yes per Eq. 5 "detached") — so
      the teacher pass is pure signal, no gradient flows through it.

## Terms / background to look up

- **OPSD** (on-policy self-distillation), **SDAR** (Lu et al. 2026b, "Self-distilled agentic
  RL" — the main training-recipe baseline), **RLSD**, **StepOPSD**, **GiGPO**
  (group-in-group PO — complementary, uses repeated anchor states instead of distillation).
- **Sequential probability ratio test** (Wald 1945) — the γ=1 case is literally an SPRT
  accumulator; nice lineage.
- **Bayes factors** (Kass & Raftery 1995).
- SkillRL / SkillBank (Xia et al. 2026) — the skill library used for privileged conditioning.

## Why I think this is implementable as a toy

The whole method is: (1) a policy that emits multi-token actions per turn, (2) a privileged
"skill" context, (3) two forward passes (with/without skill) → per-token log-prob gaps,
(4) a tiny recursive belief accumulator in log-odds space, (5) a bounded advantage
reshaper, (6) a PPO-style clipped loss. All of this can be built with a small GRU/transformer
policy on a synthetic multi-turn bandit-ish environment with binary terminal reward, where
I *know* which turns are pivotal — so I can actually measure whether AgentOPSD's credit
localizes better than uniform GRPO credit. That's the plan for the implementation step.
