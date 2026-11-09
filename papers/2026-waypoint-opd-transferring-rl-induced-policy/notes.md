# Waypoint-OPD — Notes

> **Paper**: "Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation"
> **Authors**: Bingxi Hou, Guochao Jiang, Guofeng Quan, Weiqing Li, Wenfeng Feng, Guohua Liu, Yuewei Zhang (Alibaba Cloud Computing)
> **arXiv**: 2610.21601 (Oct 2026) · **Code**: https://github.com/hhhbxloveai/WaypointOPD
>
> Pass 1: intro + method (Sections 1–3). Pass 2: deep read of the equations,
> Algorithm 1, and Appendices A–H (ablation definitions, hyperparameters, gate
> diagnostics, timing).

---

## First impressions (pass 1)

A methods paper in the now-crowded "on-policy distillation of RL gains" genre.
The question it asks is crisp: when you distill a teacher's RL improvement into
a student by using the teacher's pre-RL → post-RL log-prob shift as an implicit
reward on the *student's own* trajectories, you only see the **endpoint** of the
teacher's training. The same endpoint difference can hide two very different
histories: early and late shifts that **reinforced** each other (both pushed the
same direction — trustworthy movement) or that **cancelled** (overshoot then
correction, or vice versa — the residual endpoint is then a small, noisy leftover
of two large opposing forces). Endpoint-only transfer cannot tell these apart.

The fix is almost embarrassingly simple: keep **one intermediate checkpoint**
(the "Waypoint") from the teacher's RL run. At every prefix the student visits,
compute the early shift (Initial→Waypoint) and the late shift (Waypoint→Final)
over the student's top-k next-token candidates, center both, take their
agreement (cosine), and **gate the endpoint shift by max(0, cosine)**.
Reinforcing prefixes pass the signal through; reversing prefixes are zeroed.
Cost: one extra frozen-teacher scoring pass. That's the whole method.

My immediate reaction: this is a *diagnostic* dressed as a training objective —
and a good one. The motivating observation (endpoint ambiguity) is real and the
measurement (directional consistency between consecutive segments of the same RL
trajectory) is information genuinely absent from the endpoint pair. Whether
agreement actually predicts "beneficial supervision" is an empirical question
the paper is honest about ("Directional agreement alone does not guarantee
beneficial supervision").

## The problem, in my own words

RL on LLMs (GRPO on verifiable math/code rewards) is expensive: thousands of
rollouts, verifier calls, GPU-hours. If a small teacher (Qwen3-4B) has already
paid that cost, you want to transfer the *improvement* — not the absolute
policy — into other models. On-policy distillation (OPD) imitates the teacher's
final distribution on student-generated prefixes, but that ties the student to
the teacher's ceiling and its quirks. Direct-OPD instead transfers the
**change**: δ_t(a) = log π_final(a|s_t) − log π_init(a|s_t) as a dense implicit
reward on the student's own rollouts. But δ is a difference of two logs taken
1200 RL steps apart — it compresses the whole trajectory of the teacher's
learning into one vector and throws away the dynamics.

## The core idea (one paragraph)

Split the teacher's RL trajectory into two segments with a Waypoint checkpoint
π_m. On the student's top-k candidate support at each visited prefix, form the
centered early shift u_t and centered late shift v_t (centered by the student's
own normalized probabilities q_t, so action-independent offsets cancel and the
comparison focuses on tokens the student actually considers). Their q_t-weighted
cosine c_t measures whether the teacher's *relative* action preferences moved
consistently. The transferred signal stays the centered endpoint d_t = u_t + v_t
— the Waypoint only decides its **weight**: g_t = max(0, c_t), with a batch-level
normalizer κ_B holding the mean rescaled gate at τ = 0.8. Final per-candidate
advantage A_t(a) = sg[κ_B q_t(a) g_t d_t(a)] feeds a candidate-wise dual-clipped
PPO update plus a KL leash to the student's initialization.

---

# PASS 2 — DEEP READ

## Method, equation by equation (Section 3)

Setup: prompt x, response y = (y_1..y_T), prefix s_t = (x, y_<t). Student π_θ
rolls out on-policy; π_S (student init) stays frozen as the KL anchor. Three
frozen teacher checkpoints from ONE RL run: π_0 (Initial, before RL), π_m
(Waypoint, mid-run), π_1 (Final). Same tokenizer for everyone.

- **Eq 5 (student support):** K_t = TopK(π_θ(·|s_t), k), k = 16 in the paper.
  q_t(a) = π_θ(a|s_t) / Σ_{b∈K_t} π_θ(b|s_t), detached. The comparison happens
  only on tokens the student is actually choosing among, weighted by how much
  the student cares about each.
- **Eq 6–7 (segment shifts):** δ^(1)_t(a) = log π_m(a|s_t) − log π_0(a|s_t)
  (early), δ^(2)_t(a) = log π_1(a|s_t) − log π_m(a|s_t) (late). Telescope:
  δ^(1) + δ^(2) = δ = log π_1 − log π_0 (the Direct-OPD endpoint signal).
- **Eq 8 (centering):** C_qt[z](a) = z(a) − E_{b~qt}[z(b)]. Kills any
  action-independent offset (a uniform shift across candidates changes no
  relative preference but would contaminate an uncentered cosine). u_t = C[δ^(1)],
  v_t = C[δ^(2)], d_t = C[δ] = u_t + v_t by linearity (Eq 14).
- **Eq 9 (agreement):** c_t = ⟨u_t, v_t⟩_qt / (‖u_t‖_qt ‖v_t‖_qt) — a
  q_t-weightized cosine over the candidate support. Inner product
  ⟨·,·⟩_qt = Σ_a q_t(a) u(a) v(a).
- **Eq 10 (gate):** g_t = max(0, c_t) when both centered norms > ε = 1e−8;
  **fallback g_t = 1** when either norm ≤ ε (too little weighted variation to
  define a direction — keep the endpoint signal rather than destroy it). One
  gate per prefix, shared by all its candidates. So the gate is *graded*: fully
  aligned (c=1) passes everything, orthogonal (c=0) or opposed (c<0) passes
  nothing.
- **Eq 11 (batch-scale normalization):** κ_B = clip(τ / max(mean_{t∈B_pref} g_t,
  ε_num), κ_min, κ_max) with τ = 0.8, [κ_min, κ_max] = [0.75, 1.5]. A single
  shared positive multiplier per batch: preserves relative gate weights, keeps
  zero gates zero, but can push κ_B·g_t above 1. It targets the *mean* rescaled
  gate, not the reward norm.
- **Eq 12 (advantage):** A_t(a) = sg[κ_B q_t(a) g_t d_t(a)] — fully detached.
- **Eq 13 (objective):** maximize Σ_t Σ_{a∈K_t} A_t(a) log π_θ(a|s_t) −
  β·D̂_KL(π_θ ‖ π_S), β = 2.5, KL estimated on sampled response tokens.
- **Eq 15 (zero-sum):** Σ_a A_t(a) = κ_B g_t Σ_a q_t(a) d_t(a) = 0 by
  construction (d_t is q_t-centered). So the distillation term only
  *redistributes* probability among the k candidates — no absolute push.
- **Eq 16 (local gradient):** ∂/∂z_b [−Σ_a A_t(a) log π_θ(a|s_t)] = −A_t(b)
  for b ∈ K_t, 0 otherwise — the zero-sum coefficients cancel the softmax's
  common term. (Local statement only; shared parameters let mass leak elsewhere.)
- **Eq 17 (implementation):** candidate-wise **dual-clipped PPO** with ratio
  r_t(a) = π_θ(a|s_t)/π_old(a|s_t): J_t(a) = min{rA, clip(r,1−ε,1+ε)A} for
  A ≥ 0; max{min{...}, cA} for A < 0 (dual-clip constant c = 3, from Ye et al.
  2020). At the rollout policy r = 1 and clipping is inactive, so this matches
  the surrogate's gradient exactly.

## Algorithm 1 (Appendix B)

For each update: sample prompts, roll out student responses → for each valid
prefix: top-k + q_t → score candidates with all three frozen teachers →
u_t, v_t → c_t, g_t (with fallback) → d_t → after all prefixes: κ_B →
A_t(a) = sg[κ_B q_t g_t d_t] → dual-clipped PPO update + KL to π_S.
No verifier calls, no new parameters, one extra scoring pass vs Direct-OPD.

## Experiments (Section 4–5, Appendices C–H)

- Teachers: Qwen3-4B, GRPO, 1200 steps. Math: ~57K DeepMath examples at
  difficulty ≥ 6; code: ~25K Eurus-RL-Code. Waypoint = step 600 (= 50%).
  GRPO: batch 128, rollout n=8, lr 1e−6, temp 1.0, KL coef 0.0.
- Students: Qwen3-8B / 14B / 30B-A3B (bigger than the teacher, same family /
  tokenizer). Transfer: DAPO-Math-17K / Codeforces-derived prompts, 100 steps,
  batch 256, rollout n=1, lr 1e−6, β=2.5, k=16, τ=0.8, clip [0.75, 1.5].
- Eval: math mean@32 over 7 benchmarks (AIME 24/25/26, HMMT ×3, AMC23); code
  mean@8 over 5 (MBPP, MBPP+, LCB v4/v5/v6). Checkpoint selection on AIME24.
- **Math (Table 1):** Direct-OPD → Waypoint-OPD: 53.96 → 56.14 (8B),
  57.61 → 59.20 (14B), 55.96 → 57.85 (30B-A3B). Final teacher = 56.13, so the
  14B/30B students *exceed the RL teacher*; 8B matches it. OPD (51.04) and
  OPD2 (51.46) lag at 8B.
- **Code (Tables 2/7):** 50.59 → 53.25 (8B), 56.30 → 57.82 (14B),
  56.35 → 57.54 (30B); teacher Final 55.37. Gains in all 15 code comparisons.
- **Ablations (Table 3, 8B math; full = 56.14):** No gate (g≡1, keep centering
  + κ rule → κ_B ≡ 0.8) = 53.50; Shuffled gate = 51.20; No normalization
  (κ≡1) = 55.22; Direct-OPD = 53.96. Two striking facts:
  1. **Shuffled is worse than removed** (51.20 < 53.50): a wrong gate is worse
     than no gate — the gate–prefix *association* is the payload, not the gate
     distribution. Great control design.
  2. **No-gate ≈ Direct-OPD** (53.50 vs 53.96): the candidate-support +
     centering + κ machinery alone buys nothing; the *temporal gate* is the
     entire gain. (The paper is careful: "This comparison does not isolate the
     effect of centering.")
- **Waypoint position (Appendix F):** steps 500/600/700 → 56.18/56.14/56.13.
  Range 0.05 points. The method does not need a carefully chosen waypoint —
  any mid-run checkpoint works.
- **Gate diagnostics (Appendix G):** mean gate decays 0.810 → 0.683 over the
  100 transfer steps; zero-gate fraction grows 7.26% → 16.59%; κ_B stays in
  [0.955, 1.223] and pins the mean rescaled gate at exactly 0.8. So as the
  student's prefixes drift away from the teacher's training manifold, early/late
  agreement degrades and the method self-throttles.
- **Timing (Appendix H):** Waypoint-OPD is *faster* than Direct-OPD (8h02m vs
  8h21m on 8×H20, −3.73%) despite one extra scoring pass, because gated
  students write shorter responses (truncation 11.97% vs 14.83%). The gate
  suppresses signal at prefixes that presumably encourage rambling.

## Baseline taxonomy (from Related Work, Section 6)

The 2026 "selective on-policy distillation" landscape this sits in:
- **OPD** (Agarwal et al. 2024): reverse-KL to the Final teacher on student
  rollouts — imitation of the endpoint policy.
- **OPD2** (Heo et al. 2026): delta distillation; keeps token-level change only
  when its sign agrees with the centered standard distillation signal.
- **Direct-OPD** (Feng et al. 2026): endpoint log-ratio log π_1 − log π_0 as
  implicit reward on student rollouts (the base method here).
- **W2S-OPD** (Yu et al. 2026): proxy teacher = student base logits + weak-model
  logit contrast. **OPRD** (Park et al. 2026): centered teacher–reference shift
  amplifying verifier gradients. **Prune-OPD**, **EOPD** (entropy-gated),
  **PW-OPSD** (position-weighted), **RG-OPD** (reward-gated trajectories).
- Waypoint-OPD's distinguishing evidence: **consecutive shifts within the
  teacher's own RL trajectory**, scored on the student's supported candidates,
  with no verifier in the loop.

## Open questions / things I want to poke at

- **Why should agreement mean "trustworthy"?** The honest answer: hypothesis +
  empirics. Mechanism story: a cancelling history leaves a small endpoint
  residual that is dominated by whatever segment overshot — arguably noise (or
  overshoot artifact) rather than stable preference. Aligned histories mean both
  segments independently confirmed the direction. The paper never tests this
  story directly (e.g., by correlating gate value with downstream student gain
  per prefix); the ablations are the indirect evidence.
- **The gate is magnitude-graded, not binary**: g = c when c > 0. Partial
  alignment (c = 0.3) transfers 30% of the endpoint signal. Is the linear
  scaling load-bearing vs. binary keep/drop? Not ablated.
- **Zero-sum advantages** mean the distillation term can only redistribute mass
  within the top-16 of a 150k vocab. Combined with the KL leash, that's a very
  local update — consistent with why 100 steps suffice.
- **Fallback subtlety**: g = 1 (full transfer) when a segment barely moved. In
  the late RL phase, many prefixes will have tiny v_t (teacher converged) —
  those all take the fallback path? The diagnostics say zero-gate fraction is
  7–17%, so most prefixes have informative norms.
- **Waypoint at 50% of steps worked; would any intermediate checkpoint do?**
  Appendix F says yes for 500/600/700 — but all are near mid-run. Early (step
  50) or late (step 1150) waypoints untested. In the limit waypoint→initial or
  waypoint→final, one segment degenerates and c_t is undefined or trivially 1 —
  worth understanding in the toy.

## Why the toy implementation is straightforward

The entire method is: per-prefix log-probs from three frozen nets, an
elementwise centering, a weighted cosine, a batch scalar, and a PPO-style
update on a small candidate set. Nothing needs a 4B model. What the toy MUST
reproduce to be meaningful is the *phenomenon the gate exploits*: a teacher RL
run whose early and late shifts genuinely oppose at some prefixes. Plan: build
a shortcut-learning setup — SFT the Initial teacher on a biased annotator
(correct on 75% of instances via a salient spurious feature), then GRPO with
the true verifier reward. The teacher should first exploit the shortcut
(early shift: boost shortcut token = suppress correct token on conflict
instances) and later slowly learn the subtle true feature (late shift: reverse
that). Conflict instances then have c_t ≈ −1 with a small cancelled endpoint,
agreement instances c_t ≈ +1 with a large endpoint — exactly Figure 2's
"Aligned vs Reversing" picture, and measurable per instance type.
