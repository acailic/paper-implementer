# Breakdown — Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation

> **Paper:** "Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation"
> **Authors:** Bingxi Hou, Guochao Jiang, Guofeng Quan, Weiqing Li, Wenfeng Feng, Guohua Liu, Yuewei Zhang (Alibaba Cloud Computing)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2610.21601
> **Code (official):** https://github.com/hhhbxloveai/WaypointOPD

---

## 1. Problem & Motivation

**Problem.** RL with verifiable rewards (GRPO on math/code) makes an LLM better,
but at large cost: massive rollouts, verifier calls, GPU time. Given a teacher
that already paid that cost, transfer the *RL improvement* into other models —
without them running RL, without verifiers, and without capping them at the
teacher's absolute policy.

**The endpoint ambiguity.** The prior art (Direct-OPD, Feng et al. 2026)
transfers the *change* in the teacher's log-probabilities between a pre-RL
checkpoint π₀ and a post-RL checkpoint π₁:

```
δ_t(a) = log π₁(a | s_t) − log π₀(a | s_t)      (per-token implicit reward)
```

But this endpoint difference compresses the teacher's entire RL trajectory into
one vector. The same endpoint can arise from:

- **Reinforcing history:** early and late shifts point the same way
  (π₀ → π_m and π_m → π₁ both pushed the same relative preferences). The
  endpoint is the *sum* of two consistent movements — strong evidence of a
  stable preference change.
- **Cancelling history:** early and late shifts oppose (overshoot then
  correction, or backtrack). The endpoint is the small *residual* of two large
  opposing forces — dominated by whatever happened to overshoot, i.e. plausibly
  noise rather than stable preference.

Endpoint-only transfer treats these identically. The paper's hypothesis:
**directional agreement between consecutive teacher shifts is evidence for how
much weight the endpoint signal deserves at each student-visited prefix.**

**Why important / limitations of priors:**

- *Standard OPD* (Agarwal et al. 2024): reverse-KL to the Final teacher —
  imitation; tied to teacher's ceiling and quirks.
- *OPD2* (Heo et al. 2026): transfers the reasoning-tuning delta but gates by
  sign-agreement with the standard distillation signal — still endpoint-only,
  and requires that auxiliary signal.
- *Direct-OPD*: transfers the RL delta directly — endpoint-only, blind to how
  the delta was produced.
- Checkpoint-selection / progressive-distillation work uses intermediate
  checkpoints as *targets* or *curricula*; Waypoint-OPD uses one as *evidence*
  for weighting, never as a target.

## 2. Key Insight / Contribution

**Core idea (my words).** Keep one intermediate checkpoint (Waypoint π_m) from
the teacher's RL run. At every prefix the student visits, split the teacher's
policy shift into its early segment (π₀→π_m) and late segment (π_m→π₁),
measured over the student's own top-k candidate tokens and centered by the
student's candidate weights. If the two segments' *relative preference changes*
agree in direction (weighted cosine > 0), pass the centered endpoint shift to
the student scaled by that cosine; if they oppose, suppress it (gate = 0).
Batch-normalize so the mean rescaled gate stays at a target τ. The transferred
signal is always the endpoint shift — the Waypoint only decides its weight.

**What is genuinely new:**
1. Temporal agreement between *consecutive segments of the teacher's own RL
   trajectory* as the gating evidence — information absent from any endpoint
   pair and requiring no verifiers, rollouts, or extra training.
2. The full recipe (student-support restriction, q_t-centering, positive-cosine
   gate, low-norm fallback, batch-scale normalization, candidate-wise
   dual-clipped PPO with KL leash) as a drop-in on Direct-OPD for exactly one
   extra frozen-teacher scoring pass.
3. Empirically: best domain-average in all 6 settings (math/code × 8B/14B/30B
   students), with ablations showing gate–prefix *association* (not just gate
   presence) is load-bearing.

## 3. Method

### 3.1 Overview

Four-stage loop per update (Figure 2 of the paper):

```
 1 On-policy rollout        2 Teacher shifts           3 Gate & scale            4 Student update
 ─────────────────          ─────────────────          ─────────────────         ─────────────────
 prompt x ─→ student π_θ    score the k candidates     agreement gate g_t        dual-clipped PPO on
 rolls out y = y_1..y_T     with FROZEN teachers       on centered shifts        candidate advantages
                            π₀, π_m, π₁  →  early u_t,  g_t = max(0, c_t)         A_t(a) = sg[κ_B q_t g_t d_t a]
                            late v_t, endpoint d_t     batch scale κ_B           + sampled-token KL to π_S
```

### 3.2 Architecture

No new modules. The cast:

```
                 one teacher RL run (GRPO), frozen afterwards
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │ Initial π₀ │ ──▶ │ Waypoint π_m│ ──▶ │  Final π₁  │     (same tokenizer as student)
   └────────────┘     └────────────┘     └────────────┘
     before RL          mid-run            end of RL
                  (e.g. step 600 of 1200)

   ┌────────────┐     ┌────────────┐
   │ Student π_θ│ ──▶ │ Ref π_S    │  π_S = student init, frozen; KL anchor only
   └────────────┘     └────────────┘
```

### 3.3 Forward pass / pipeline (Algorithm 1)

For each update:

1. **Rollout.** Sample prompts x ∼ D; sample responses y ∼ π_θ(·|x).
2. **Per valid prefix** s_t = (x, y_<t):
   a. **Support:** K_t = TopK(π_θ(·|s_t), k) — the k (paper: 16) tokens the
      student considers most likely. Detached normalized weights
      q_t(a) = π_θ(a|s_t) / Σ_{b∈K_t} π_θ(b|s_t).
   b. **Score:** evaluate log π₀(a|s_t), log π_m(a|s_t), log π₁(a|s_t) for all
      a ∈ K_t (three frozen forward scores on the k candidates only).
   c. **Segment shifts:** δ⁽¹⁾_t(a) = log π_m − log π₀ (early);
      δ⁽²⁾_t(a) = log π₁ − log π_m (late). Endpoint δ_t = δ⁽¹⁾ + δ⁽²⁾ telescopes.
   d. **Center:** u_t = C_qt[δ⁽¹⁾], v_t = C_qt[δ⁽²⁾], d_t = C_qt[δ_t] = u_t + v_t.
   e. **Agreement:** c_t = ⟨u_t, v_t⟩_qt / (‖u_t‖_qt ‖v_t‖_qt);
      g_t = max(0, c_t), with fallback g_t = 1 if either centered norm ≤ ε.
3. **Batch scale:** κ_B = clip(τ / max(mean_{t∈B_pref} g_t, ε_num), κ_min, κ_max).
4. **Advantages:** A_t(a) = sg[κ_B q_t(a) g_t d_t(a)] (fully detached).
5. **Update:** candidate-wise dual-clipped PPO (Eq 17) on A, plus sampled-token
   KL penalty to π_S.

### 3.4 Loss function

The idealized surrogate (Eq 13):

```
J_WP(θ) = E_x E_{y~π_θ} [ Σ_t Σ_{a∈K_t} A_t(a) · log π_θ(a|s_t) ]  −  β · D̂_KL(π_θ ‖ π_S)
```

with rollout samples and candidate supports held fixed, A detached, β = 2.5,
KL estimated on the sampled response tokens.

Implementation: per-candidate dual-clipped PPO (Eq 17) with importance ratio
r_t(a) = π_θ(a|s_t)/π_old(a|s_t), clip ε, and dual-clip constant c = 3:

```
J_t(a) = min{ r_t(a)·A_t(a), clip(r_t(a), 1−ε, 1+ε)·A_t(a) }          if A_t(a) ≥ 0
J_t(a) = max{ min{...same...}, c·A_t(a) }                              if A_t(a) < 0
loss    = − Σ_t Σ_a J_t(a)  (+ β·KL)      token-mean aggregation over valid response tokens
```

At the rollout policy r = 1, clipping inactive → identical gradient to the
surrogate (proved via the zero-sum property, Eq 15–16).

## 4. Math

**Notation.** x prompt, y = (y₁..y_T) response, s_t = (x, y_<t) prefix,
π_θ student, π_S student init (frozen), π₀/π_m/π₁ Initial/Waypoint/Final
teachers (frozen, one RL run), K_t top-k support, q_t normalized student
weights on K_t.

**Eq 5 — student-weighted support.**
`q_t(a) = π_θ(a|s_t) / Σ_{b∈K_t} π_θ(b|s_t)`, a ∈ K_t.
*Plain English:* restrict attention to the k tokens the student actually
contemplates, and weight each by the student's own (detached) preference. The
comparison focuses on the support that receives the update.

**Eq 6/7 — segment shifts.**
δ⁽¹⁾_t(a) = log π_m(a|s_t) − log π₀(a|s_t); δ⁽²⁾_t(a) = log π₁(a|s_t) − log π_m(a|s_t).
*Plain English:* how the teacher's log-prob for candidate a moved during the
first half vs the second half of RL. Their sum is the Direct-OPD endpoint.

**Eq 8 — centering.**
C_qt[z](a) = z(a) − E_{b~q_t}[z(b)] = z(a) − Σ_b q_t(b) z(b).
*Plain English:* subtract the q_t-weighted average, so any component that lifts
or lowers *all* candidates equally (action-independent offset) vanishes; only
*relative* preference changes survive. Needed because a uniform shift changes
no relative preference yet would corrupt a raw-vector cosine.

**Eq 9 — weighted agreement.**
c_t = ⟨u_t, v_t⟩_qt / (‖u_t‖_qt ‖v_t‖_qt) with ⟨u,v⟩_qt = Σ_a q_t(a) u(a) v(a).
*Plain English:* cosine similarity between early and late centered shifts,
computed in the student's own weighting geometry. c_t = +1: identical relative
movement (reinforcing). c_t ≈ 0: unrelated. c_t = −1: exactly opposed
(cancelling).

**Eq 10 — positive-cosine gate with fallback.**
g_t = max(0, c_t) if both ‖u_t‖_qt, ‖v_t‖_qt > ε = 1e−8; else g_t = 1.
*Plain English:* the gate passes aligned signal at full-to-partial strength
(graded, not binary), zeroes opposed signal, and if a segment barely moved
there is no meaningful direction to compare — keep the endpoint (fallback to
no gating for that prefix).

**Eq 11 — batch-scale normalization.**
κ_B = clip( τ / max( (1/|B_pref|) Σ_t g_t , ε_num ), κ_min, κ_max ), τ = 0.8,
[κ_min, κ_max] = [0.75, 1.5], ε_num ≈ 1.18e−38 (smallest normal float32).
*Plain English:* one shared positive multiplier per rollout batch that pins
the *mean* rescaled gate at τ. Preserves relative gate weights; zero gates
stay zero; κ_B·g_t may exceed 1. It stabilizes gate scale, not reward norm.

**Eq 12 — detached candidate advantage.**
A_t(a) = sg[κ_B q_t(a) g_t d_t(a)].
*Plain English:* the endpoint signal, centered, restricted to the support,
reweighted by student preference, gated by temporal agreement, rescaled by the
batch constant — then fully detached (pure reward, no gradient flows to it).

**Eq 15 — zero-sum.**
Σ_{a∈K_t} A_t(a) = κ_B g_t · Σ_a q_t(a) d_t(a) = 0 (because d_t is q_t-centered).
*Plain English:* advantages within a prefix sum to zero — the distillation term
can only *redistribute* probability among the k candidates, never add net mass.

**Eq 16 — local gradient.**
∂/∂z_b [−Σ_a A_t(a) log π_θ(a|s_t)] = −A_t(b) if b ∈ K_t else 0.
*Plain English:* because A is zero-sum, the softmax partition term cancels;
each candidate's logit is pushed by exactly (minus) its advantage. (Local fact;
shared parameters still couple prefixes.)

**Eq 14 — endpoint preservation (Appendix A.1).**
At fixed s_t, K_t, q_t: u_t + v_t = C_qt[δ⁽¹⁾+δ⁽²⁾] = d_t — the Waypoint
affects only g_t and κ_B; the transferred object d_t is *identical* to
Direct-OPD's centered endpoint. Also: C_qt[z + b] = C_qt[z] for constants b
(offset invariance); independent positive rescaling of u, v leaves the gate
unchanged (cosine); κ_B g_t ≥ 0 means A can never reverse the sign of d.

**Dual-clipped PPO (Eq 17, Ye et al. 2020).** Standard PPO clip for positive
advantages; for negative ones an outer max{·, c·A} prevents the unclipped
objective from collapsing the ratio toward 0. Same gradient as Eq 13 at the
rollout policy.

## 5. Training

**Teacher (per domain).** Qwen3-4B, GRPO 1200 steps, batch 128, rollout n = 8,
lr 1e−6, temp 1.0, KL coef 0.0, max response 16,384. Math: ~57K DeepMath
examples (difficulty ≥ 6). Code: ~25K Eurus-RL-Code. Checkpoints saved at
step 0 (Initial), 600 (Waypoint = 50%), 1200 (Final). All frozen for transfer.

**Student transfer (per domain × student).** Students Qwen3-8B/14B/30B-A3B
(same family/tokenizer as teacher — required for token-level comparison).
Built on verl + FSDP + vLLM, 8×H20 GPUs. 100 optimization steps, train batch
256, mini-batch 64, rollout n = 1, lr 1e−6, temp 1.0, max response 16,384.
Method hyperparameters: k = 16, β = 2.5, τ = 0.8, clip [0.75, 1.5],
ε = 1e−8. Math prompts DAPO-Math-17K; code prompts Codeforces-derived.
Checkpoint selection by AIME24 validation accuracy.

**Evaluation.** Math mean@32 over AIME 24/25/26, HMMT Feb25/Nov25/Feb26, AMC23.
Code mean@8 over MBPP, MBPP+, LCB v4/v5/v6. Temp 0.7, top-p 0.95.

**Baselines/controls.** Base student; OPD (reverse-KL to Final); OPD2
(sampled-token delta distillation); Direct-OPD (endpoint reward). Controls on
8B math: *No gate* (g≡1, keeps centering + κ rule → κ_B ≡ 0.8), *Shuffled
gate* (permute g_t across prefixes in the global batch — preserves gate
distribution and κ_B, breaks gate–prefix pairing), *No normalization*
(κ_B ≡ 1, keeps gate + fallback).

## 6. Results & Ablations

**Math (Table 1, mean@32 avg over 7 benchmarks):**

| | Teacher Initial | Teacher Waypoint | Teacher Final | Base | OPD | OPD2 | Direct-OPD | **Waypoint-OPD** |
|---|---|---|---|---|---|---|---|---|
| 8B student | 24.84 | 51.82 | 56.13 | 26.04 | 51.04 | 51.46 | 53.96 | **56.14** |
| 14B student | 24.84 | 51.82 | 56.13 | 27.32 | 56.17 | 52.83 | 57.61 | **59.20** |
| 30B-A3B student | 24.84 | 51.82 | 56.13 | 28.03 | 51.22 | 52.68 | 55.96 | **57.85** |

Gains over Direct-OPD: +2.18 / +1.59 / +1.89. 19/21 benchmark comparisons
improved; 14B and 30B students *exceed the Final teacher* (59.20 and 57.85 vs
56.13); 8B matches it.

**Code (Tables 2/7, mean@8 avg over 5 evaluations):** Direct-OPD → Waypoint-OPD:
50.59 → **53.25** (8B), 56.30 → **57.82** (14B), 56.35 → **57.54** (30B);
teacher Final 55.37; gains in all 15 comparisons.

**Ablations (Table 3, 8B math):** Full 56.14 > No-norm 55.22 > No-gate 53.50 ≈
Direct-OPD 53.96 > Shuffled 51.20.

Reading:
- **The temporal gate is the whole gain.** No-gate (still centering + support +
  normalization) lands at Direct-OPD's level — the surrounding machinery buys
  ~nothing without the agreement weighting.
- **Wrong gates are worse than no gates.** Shuffling (same gate distribution,
  broken prefix association) costs 2.30 points vs no-gate. The information is
  *which* prefixes get gated, so it is not acting as generic regularization or
  noise injection.
- **Normalization is a small additive bonus** (+0.92), not the driver.

**Waypoint position (Appendix F):** steps 500/600/700 → 56.18/56.14/56.13 —
insensitive within mid-run; all beat Direct-OPD on every benchmark.

**Gate diagnostics (Appendix G):** mean gate decays 0.810 → 0.683 over
training; zero-gate fraction rises 7.26% → 16.59%; κ_B ∈ [0.955, 1.223] keeps
mean rescaled gate at 0.8 exactly. As the student drifts from the teacher's
training distribution, agreement degrades and transfer self-throttles.

**Timing (Appendix H):** Waypoint-OPD *faster* than Direct-OPD (8h02m vs
8h21m; −3.73%) despite the extra scoring pass — gated students produce shorter
responses (truncation 11.97% vs 14.83%).

## 7. Limitations

- **Agreement ≠ correctness.** The paper states plainly: "Directional
  agreement alone does not guarantee beneficial supervision." The link between
  temporal consistency and transfer utility is an empirical regularity, not a
  theorem; no per-prefix analysis directly validates it (only end-task
  ablations).
- **Needs three checkpoints from ONE RL run** with a shared tokenizer across
  teacher/student — fine within the Qwen3 family, awkward across families.
- **Endpoint comparisons at scale:** gains are ~1.2–2.7 points; benchmark
  variance and checkpoint selection on a single validation set (AIME24) are not
  accompanied by seed replicates.
- **Gate is graded linearly (g = c)** and shared across all candidates of a
  prefix; binary gates, per-candidate gates, or other agreement→weight maps are
  untested. Centering's isolated effect is explicitly not ablated (No-gate
  keeps centering).
- **Waypoint choice tested only in a narrow window** (500–700 of 1200).
  Behavior with near-endpoint waypoints (degenerate segments) is unexamined.
- One teacher size (4B), two domains, one model family; students larger than
  the teacher only.

## 8. Open Questions / Ideas

- Does gate value correlate with *per-prefix* student improvement? (Directly
  testable in a toy: measure Δ student log-prob of the correct token at
  prefixes with c ≈ +1 vs c ≈ −1.)
- Would a binary gate (keep/drop at c > 0 threshold) match the graded one?
  The graded form silently scales by c — is magnitude information load-bearing?
- Near-degenerate waypoints: as π_m → π₀ or π_m → π₁, one segment norm → 0 and
  everything falls into the g = 1 fallback — does the method smoothly degrade
  to Direct-OPD? The toy can measure this exactly.
- Interaction with student capacity: the paper's students are *bigger* than the
  teacher. Does gating still help a smaller student (where the teacher's
  cancellations might carry usable signal)?
- What is being suppressed? The Appendix H length result hints the gate kills
  prefixes associated with rambling; a toy with measurable "reasoning" tokens
  could make that concrete.
