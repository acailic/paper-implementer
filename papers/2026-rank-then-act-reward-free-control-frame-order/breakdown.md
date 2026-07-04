# Rank-Then-Act: Reward-Free Control from Frame-Order Progress (RTA)

**Paper:** "Rank-Then-Act: Reward-Free Control from Frame-Order Progress"
**Authors:** Yuriy Maksyuta, George Bredis\*, Ruslan Rakhimov, Daniil Gavrilov  (T-Tech)
**arXiv:** 2607.01897v1 [cs.LG], 2 Jul 2026 (7pp main + appendices; accepted at **ICML 2026 Workshop on RLxF: Reinforcement Learning from World Feedback**)
**Project page:** https://corl-team.github.io/rank-then-act
**Source-verified:** `paper.pdf` + `paper_layout.txt` (pdftotext -layout, 1228 lines). All numeric tables transcribed **verbatim** with sourcing line-ranges; every delta recomputed by a source-free reconciliation script (see end). Figure-derived numbers are NOT back-filled (universal rule).

---

## TL;DR

Learning control from expert video usually means either (a) regressing a scalar progress/success head (VLM-RM, VLAC) — brittle, uncalibrated, exploit-able — or (b) adversarial imitation (GAIL/GAIfO) — needs careful training and per-task tuning. **RTA replaces the scalar reward head with an *ordinal* signal**: a VLM is fine-tuned to output per-frame *progress ranks* on shuffled clips, and the RL reward is the **Spearman rank correlation** between those ranks and the true frame timestamps over a sliding window. The reward is **bounded in [−1, 1]**, **scale-invariant** (no calibration drift across tasks), and **reward-free** (no extrinsic environment reward at any stage).

**Headline (all prose-/table-confirmed):**
- **Discrete control (Table 1):** RTA achieves the **strongest success rate on Catrap L2/L4/L6** and is the **only reward-free method with non-zero success on Kirby** (0.07); it *beats the Oracle (binary environment reward) on 3 of 4 levels* despite using no env reward.
- **Cross-domain (Table 2):** a single Stage-1 scorer trained on YouTube / full-Catrap / MetaWorld / COIN video sources still solves Catrap L2 (1.00) and stays competitive on L4/L6 — strong cross-source transfer of the ordinal prior.
- **Reward–success coherence (Table 8):** RTA's correlation reward is the *only* signal whose mean-cumulative-reward ↔ pass@5 Pearson correlation is **positive on all 4 levels** (0.76/0.87/0.42/0.13); every baseline is near-zero or negative.

---

## 1. Problem & Motivation (L33–69)

Training VLM-based agents from video hits three obstacles:
1. **"Later-is-better" shortcut:** chronological inputs let the scorer output a vacuous monotonically-increasing progress curve.
2. **Scale ambiguity:** absolute progress magnitudes are not comparable across tasks/episodes.
3. **Online cost:** the reward must be cheap, informative, and robust to distribution shift.

RTA's answer is **supervision over ordinal structure, not scalar reward prediction**: the only quantity optimized or used as reward is a **rank correlation** — invariant to scale, calibration drift, and cross-task shifts by construction.

---

## 2. Method

### 2.1 Correlation primitive (L156–170)

The sole scalar signal in **both** stages is the **Spearman rank correlation**:

$$\mathrm{spr}(x, y) := \mathrm{Pearson}(\mathrm{rank}(x), \mathrm{rank}(y)) \in [-1, 1]$$

Used as the Stage-1 training objective *and* the Stage-2 reward. The idea of using rank correlation as an estimator of trajectory correctness follows GVL [16].

### 2.2 Stage 1 — Listwise Progress Scorer via GRPO (L179–233)

**Anchor + shuffle (defeats the "later-is-better" shortcut).** Given an expert clip τ = (s₁,…,s_T), segment into K frames, **anchor** the first frame s_anc := s₁, and **randomly permute** the rest: (s_anc, s_π̃(2), …, s_π̃(K)). Only the shuffled non-anchor frames are scored; the anchor fixes a starting point so the model cannot track local appearance drift and must judge *task* progress.

**VLM input/output.** A VLM f_ϕ receives the K-frame sequence and is prompted to emit, per shuffled frame, a reasoning trace + a predicted progress rank p_i (higher = later):

```
Frame i:
Frame Description: ...
Rank: p_i.
```

Collecting the K−1 scalars gives **p** = (p₂,…,p_K). Parsing is strict-regex; on parse failure (≠ K−1 scores) the reward is the minimum R_min = −1. Ties are broken by random permutation.

**Listwise reward (Eq 1).** With **p** = predicted ranks and **q** = ground-truth temporal indices:

$$R = \mathrm{spr}(\mathbf{p}, \mathbf{q}) \tag{1}$$

Spearman on continuous ranks makes the objective insensitive to the scale of p_i while rewarding correct monotone order.

**GRPO optimization.** VLM text generation is treated as a sequential policy over tokens; optimized with a **GRPO** objective (importance ratio r_i = π_θ/π_θ_old, clip ε, group-relative advantage Â) maximizing E[segment,shuffle] E_{output∼f_ϕ}[R]. After convergence, **f_ϕ is frozen** for Stage 2.

### 2.3 Stage 2 — Online Control from Progress–Time Consistency (L235–258)

**Windowed reward.** At step t, build window W_t = (s_{t−m+1},…,s_t) with m = min(N, t); the oldest frame is the reference anchor s_anc. The frozen scorer is queried **only on query steps** (t mod N = 0 or t = T), otherwise r_t = 0 (saves VLM cost). On each query step, draw **L independent permutations** of the m−1 non-anchor frames, score each, compute the progress–time Spearman ρ, and **average** → scalar reward r_t. Defaults **N = 15, L = 2**.

**Policy optimization.** Train π_θ with policy-gradient using r_t as the *sole* reward (no environment reward). Reward is bounded [−1,1] and sparse (every N steps), so the VLM-agent uses **GAE** (following VL-DAC [2]). Backbones:
- **Discrete games:** Qwen2.5-VL-7B (handles text-described action combos, needed for Kirby's simultaneous button presses, VideoGameBench protocol). Also an **MLP + PPO** backbone (Appendix A.5) to show gains are from reward design, not VLM bias.
- **Continuous control:** **DrQv2** [27] as policy backbone (PointMaze-UMaze, MetaWorld).
- Stage-2 algorithms: **VL-DAC** [2] and a multi-step GRPO variant **LOOP** [6] (Appendix A.1), the latter with **starting-point refreshing** when mean episode reward exceeds threshold τ to enable long-horizon completion.

---

## 3. Experimental Setup (L260–269, 845–899)

- **Reward-free protocol:** agents receive *only expert video demonstrations*; no extrinsic rewards, env APIs, or task annotations at any stage.
- **Stage-1 tasks:** Catrap levels (1–6) + full-game playthroughs; a 70-gameboy-playthrough YouTube pool for cross-domain pretraining.
- **Stage-2 discrete benchmarks:** Catrap L2/L4/L6 (L4/L6 need backtracking out of dead ends) + **Kirby** (long-horizon).
- **Stage-2 continuous benchmarks:** PointMaze-UMaze; MetaWorld (door-open, door-close, drawer-open, button-press-topdown, hammer, reach). 1M env steps; R2R trained 1M too.
- **Baselines (discrete, Table 1):** GVL [16] (= untrained Stage-1 = the formulation upper bound for an *untrained* scorer), GVL-Gemini (Gemini-3.1-Thinking backbone), VLM-RM [19], VLM-RM_reg (α=0.5), **Rank2Reward (R2R)** [26] (ranking baseline, first VLM application), and **Oracle reward** (binary end-of-level success — *has* env reward, included as a reference, not strictly comparable).
- **KL control** on Stage 1: target KL = 0.1.

---

## 4. Results

### 4.1 Discrete Control — Table 1 (L338–350, verbatim)

Success rate, mean ± std over **5 seeds**. Oracle = binary env-reward reference.

| Method | Level 2 | Level 4 | Level 6 | Kirby (lvl 0) |
|---|---|---|---|---|
| GVL | 0.47 ± 0.25 | 0.00 ± 0.00 | 0.04 ± 0.08 | 0.00 ± 0.00 |
| GVL-Gemini | 0.27 ± 0.09 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| VLM-RM | 0.40 ± 0.28 | 0.16 ± 0.20 | 0.08 ± 0.16 | 0.00 ± 0.00 |
| VLM-RM_reg (α = 0.5) | 0.44 ± 0.32 | 0.00 ± 0.00 | 0.08 ± 0.16 | 0.00 ± 0.00 |
| Rank2Reward | 0.60 ± 0.28 | 0.20 ± 0.00 | 0.13 ± 0.09 | — |
| Oracle reward | 0.50 ± 0.22 | 0.07 ± 0.09 | 0.20 ± 0.00 | 0.40 ± 0.28 |
| **RTA (w/ stage-1 training)** | **1.00 ± 0.00** | **0.72 ± 0.35** | **0.32 ± 0.27** | **0.07 ± 0.09** |

**Takeaways (every delta source-free-recomputed, see §7):**
- **RTA is the strongest method on L2/L4/L6** and the **only reward-free method with non-zero success on Kirby** (0.07 ± 0.09); the caption's "strongest performance across all levels except Kirby" refers to Oracle's env-reward edge on Kirby (0.40 > 0.07).
- **RTA beats Rank2Reward (the ranking baseline) on every directly-comparable level:** L2 +0.40 (1.00 vs 0.60), L4 +0.52 (0.72 vs 0.20), L6 +0.19 (0.32 vs 0.13) — the abstract's "outperforms rank-based baselines."
- ⚠ **Surprising: RTA (reward-free) beats Oracle (binary env reward) on 3/4 levels** — L2 +0.50 (1.00 vs 0.50), L4 +0.65 (0.72 vs 0.07), L6 +0.12 (0.32 vs 0.20). The binary end-of-level Oracle reward is *sparse*; RTA's dense shaped ρ ∈ [−1,1] reward gives more learning signal per step. Honest but counterintuitive — surface rather than bury.

### 4.2 Cross-Domain Video Sources — Table 2 (L352–362, verbatim)

Stage-1 scorer trained on a *different* visual domain, then deployed on Catrap. Mean ± std over **3 seeds**.

| Source | Level 2 | Level 4 | Level 6 | Kirby (lvl 0) |
|---|---|---|---|---|
| Youtube | 1.00 ± 0.00 | 0.47 ± 0.25 | 0.60 ± 0.28 | 0.20 ± 0.16 |
| Full Catrap | 1.00 ± 0.00 | 0.47 ± 0.38 | 0.60 ± 0.28 | 0.07 ± 0.09 |
| Full Kirby | – | – | – | 0.07 ± 0.09 |
| MetaWorld | 1.00 ± 0.00 | 0.87 ± 0.19 | 0.53 ± 0.19 | 0.13 ± 0.09 |
| Coin (AssembleSofa) | 1.00 ± 0.00 | 0.07 ± 0.09 | 0.20 ± 0.28 | 0.00 ± 0.00 |

**Takeaways:**
- **L2 is solved (1.00) regardless of source domain** — even a scorer trained on *MetaWorld* or *COIN AssembleSofa* video transfers to Catrap L2. Strong evidence the ordinal prior is domain-general.
- **L6 actually *improves* with cross-domain sources** (Youtube/Full-Catrap/MetaWorld all give 0.53–0.60 vs the in-domain Table-1 RTA 0.32) — a counterintuitive transfer gain, plausibly because diverse source video regularizes the scorer away from L6-specific overfitting. Worth flagging as a non-obvious result.
- **Kirby best transfer is from Youtube (0.20)** — beats in-domain Full-Kirby (0.07) and in-domain RTA (0.07). The diverse 70-playthrough pool generalizes best to the long-horizon game.
- **COIN (AssembleSofa) is the weakest source** (L4 0.07, L6 0.20) — the most visually dissimilar domain transfers worst, as expected.

### 4.3 Reward–Success Coherence — Table 8 (L1068–1079, verbatim)

Pearson correlation of mean cumulative reward and pass@5 during training, over **5 seeds**. 0 = method cannot solve the level; N/A = experiment not conducted.

| Method | Level 2 | Level 4 | Level 6 | Kirby (lvl 0) |
|---|---|---|---|---|
| GVL | −0.01 | 0.00 | −0.01 | 0.00 |
| GVL-Gemini | 0.33 | 0.00 | 0.00 | 0.00 |
| Rank2Reward | −0.04 | −0.02 | −0.04 | N/A |
| VLM-RM | 0.53 | −0.19 | −0.33 | 0.00 |
| VLM-RM_reg (α = 0.5) | 0.25 | 0.00 | −0.16 | 0.00 |
| **RTA (w/ stage-1 training)** | **0.76** | **0.87** | **0.42** | **0.13** |

**Takeaways:**
- **RTA is the only method whose reward positively correlates with success on all 4 levels.** Every baseline is near-zero or negative on L4/L6/Kirby (their reward signal is uninformative or anti-correlated where they fail).
- RTA's reward is **most coherent on L4** (0.87) — its hardest *solved* level — and weakest on Kirby (0.13), consistent with Kirby being the long-horizon edge case.
- This is the falsifiable coherence claim: the correlation-only reward is *informative*, not just bounded.

### 4.4 Continuous Control — Figures 3 & 4 (L423–489, figure-reads — NOT back-filled)

- **PointMaze-UMaze (Fig 3, 5 seeds):** "RTA consistently outperforms R2R on UMaze, both in the ranking-only and mixed-reward settings" (success-rate curves; no numeric table). Oracle (binary) included but "not directly comparable."
- **MetaWorld (Fig 4, 4 seeds):** RTA surpasses R2R under ranking-only reward on all 6 tasks shown; **RTA+GAIL is comparable to R2R across all tasks and outperforms on door-open and door-close.** RTA underperforms GAIL-augmented methods on exploration-heavy tasks (assigns high VOC-scores to plausible-but-unsuccessful trajectories). Crucially, **R2R pretrains a separate model per task; RTA reuses one scorer across all tasks** — the scalability win.
- ⚠ Honest scope: continuous-control success rates are **figure-only** (curve endpoints, not a verbatim table); only the directional claims above are prose-confirmed.
- **Matched-compute efficiency (Figs 14/15, Appendix A.7):** RTA stronger on door-open/door-close/reach, near-parity on drawer-open, R2R stronger on button-press-topdown/hammer — "competitive across all tasks, better on a majority," advantage not from extra compute.

### 4.5 Policy-Backbone Ablation — Table 6 (L938–946, verbatim)

RTA reward with VLM vs MLP backbone, reward-every-15 vs only-end. Mean ± std over **3 seeds (VLM) / 5 seeds (MLP)**.

| Task | Level 2 | Level 4 | Level 6 |
|---|---|---|---|
| VLM + RTA (reward every 15 steps) | 1.00 ± 0.00 | 0.60 ± 0.28 | 0.33 ± 0.34 |
| VLM + RTA (only-end reward) | 0.33 ± 0.19 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| MLP + RTA (reward every 15 steps) | 0.79 ± 0.14 | 0.27 ± 0.21 | — |
| MLP + RTA (only-end reward) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |

**Takeaways:**
- **The VLM and MLP prefer *opposite* reward schedules** — VLM does best with dense every-15-step shaping (1.00/0.60/0.33) and collapses on only-end (0.33/0.00/0.00); an initialized **MLP does best with only-end trajectory reward (1.00/1.00/1.00)** and worse with dense shaping. The paper attributes this to different exploration dynamics (a randomly-initialized MLP benefits from full-trajectory feedback; a VLM benefits from shorter-window shaping).
- **Gains are from reward design, not VLM bias:** the MLP backbone also reaches 1.00 on L2/L4 — RTA's correlation reward transfers across policy architectures.
- ⚠ **T1 vs T6 config note:** Table-1 RTA (VL-DAC default, 5 seeds) reports L2/L4/L6 = 1.00/0.72/0.32; Table-6 VLM+RTA(every-15, 3 seeds) reports 1.00/0.60/0.33. The L4 gap (0.72 vs 0.60) is within the large std (±0.35 / ±0.28) and reflects different seed counts — not a contradiction. (MLP+RTA L6 = "—" = not run.)

### 4.6 Hyperparameter Ablation — Table 7 (L992–1008, verbatim, Catrap L2)

| Ablation | Success rate | Update steps |
|---|---|---|
| **Window length M:** 5 | 0.73 ± 0.09 | 65.33 ± 12.76 |
| **Window length M:** 15 | 1.00 ± 0.00 | 37.00 ± 10.98 |
| **Window length M:** 25 | 1.00 ± 0.00 | 43.33 ± 5.79 |
| **Reward frequency:** 5 | 1.00 ± 0.00 | 175.60 ± 76.53 |
| **Reward frequency:** 15 | 1.00 ± 0.00 | 37.00 ± 10.98 |
| **Reward frequency:** 25 | 1.00 ± 0.00 | 128.67 ± 84.32 |
| **Shuffles L:** 1 | 1.00 ± 0.00 | 143.66 ± 63.43 |
| **Shuffles L:** 2 | 1.00 ± 0.00 | 37.00 ± 10.98 |
| **Shuffles L:** 4 | 0.93 ± 0.09 | 123.33 ± 83.83 |

**Takeaways:**
- **Success rate is near-1.00 across almost all settings** — RTA is robust to M / frequency / L on a solvable level; only the *very short* window (M=5, 0.73) and *too many* shuffles (L=4, 0.93) degrade slightly. This is honest-scope robustness, not a tuned-to-the-edge win.
- **Update-steps (sample efficiency) is where the defaults matter:** the default **(M=15, freq=15, L=2) converges fastest at 37 steps** — roughly **2–5× faster** than off-default configs (L=1 needs 143.66, freq=5 needs 175.60, L=4 needs 123.33). The defaults are efficiency-optimal, not success-rate-optimal.
- ⚠ Note the **update-steps column is the more sensitive metric** — the paper's "performance degrades slightly at extreme configurations" framing (caption) actually applies to *sample efficiency*, while success rate stays flat. Worth distinguishing.

### 4.7 Stage-1 Convergence (Fig 7, figure-reads — qualitative)

Per-level Catrap training shows "fast rises in progress–time ρ with slower convergence on intricate levels"; full-playthrough training across games "also converges, with slightly lower plateaus due to non-informative segments (e.g., menus)." Prose claims "almost all videos converge to Spearman ρ > 0.9 in at most 300 steps" (L279) — figure-derived endpoint, not transcribed as a table.

---

## 5. Hyperparameters (verbatim)

### Table 5 — Stage 1 scorer (L913–927)

| Hyperparameter | Value |
|---|---|
| Algo steps | 200–400 |
| Learning Rate | 1e-5 |
| Scheduler | constant with warmup |
| Num. warmup steps | 10 |
| Grad Accum. Steps | 16 |
| Mini-batch Size | 1 |
| GRPO Epochs | 1 |
| Obs. Image Length | 15 |
| K (frames) | 4 |
| Temperature | 1 |

### Table 3 — Stage 2 VL-DAC (L853–872)

| Hyperparameter | Value |
|---|---|
| Env. steps | 57600 |
| Learning Rate (init → final) | 1e-5 → 5e-7 |
| Scheduler | cosine |
| GAE λ_g | 0.95 |
| γ_g | 0.99 |
| Value Loss Coeff. | 0.15 |
| KL β | 0.05 |
| Policy Freeze (steps) | 2 |
| Grad Accum. Steps | 32 |
| Mini-batch Size | 1 |
| PPO Epochs | 2 |
| Obs. Image Length | 5 |
| Rollout Size | 256 |
| Max Episode Steps | 64 |
| Temperature | 0.2 |

### Table 4 — Stage 2 LOOP (L874–892)

| Hyperparameter | Value |
|---|---|
| Algorithm steps | 225 |
| Learning Rate (init → final) | 1e-5 → 1e-6 |
| Scheduler | linear with warmup |
| Num. warmup steps | 10 |
| KL β | 0.05 |
| Grad Accum. Steps | 32 |
| Mini-batch Size | 1 |
| PPO Epochs | 1 |
| Obs. Image Length | 5 |
| Rollout Size | 120 |
| τ threshold | 0.5 |
| τ threshold steps | 3 |
| K | 4 |
| Temperature | 1 |

> **LOOP starting-point refresh (Fig 6, L729–731):** without refreshing, rewards improve but may not reach success when window N is shorter than the task; refreshing from the best terminal state once mean reward exceeds τ=0.5 (for τ-threshold-steps=3 consecutive queries) enables long-horizon completion. Brief post-refresh dips arise if N exceeds remaining steps.

---

## 6. Figures (qualitative / figure-reads — NOT back-filled)

- **Fig 1:** two-stage RTA framework diagram.
- **Fig 2:** cross-level transfer heatmap (validation progress–time ρ, source→target); diagonals = in-level upper bounds, off-diagonals = transfer. Pooled training transfers best; Level 1 generalizes from most sources. (Heatmap cells = figure-bar reads, not transcribed.)
- **Figs 3/4:** PointMaze-UMaze / MetaWorld success-rate & return curves — directional claims only (§4.4).
- **Fig 5:** cyclic-trajectory window-size behavior (small windows → spurious monotone reward; intermediate → correctly flat; large → dilution). Motivates the default intermediate window.
- **Fig 6:** LOOP with/without starting-point refresh.
- **Fig 7:** Stage-1 per-level / per-game convergence curves.
- **Figs 8–11:** good/bad level-completion examples with reward annotations (0.77 / −0.04 / 0.66 / 0.15) — qualitative, illustrates reward-scorer behavior on in/out-of-distribution and trained/undertrained scorers.
- **Fig 12:** Catrap L2 mean VOC-reward vs success-rate during training (highly correlated) — the coherence evidence behind Table 8.
- **Fig 13:** Catrap L3 training sensitivity (ρ alternates over 400 GRPO steps; L3 did not converge in 200 steps).
- **Figs 14/15:** matched-compute efficiency RTA vs R2R (MetaWorld / UMaze) — directional claims only (§4.4).

---

## 7. Source-Free Reconciliation (verification, no PDF re-read)

Python recomputation of every cited delta from displayed cells:

- **Abstract "matches or outperforms prior video-based reward learning methods and rank-based baselines":** RTA > Rank2Reward on L2/L4/L6 by **+0.40 / +0.52 / +0.19** (1.00 vs 0.60, 0.72 vs 0.20, 0.32 vs 0.13); RTA > every reward-free VLM baseline (GVL/GVL-Gemini/VLM-RM/VLM-RM_reg) on L2/L4/L6; RTA is the only reward-free method non-zero on Kirby. ✓
- **Caption "strongest across all levels except Kirby":** RTA column-max on L2/L4/L6 (1.00/0.72/0.32 > all baselines incl. Oracle 0.50/0.07/0.20); on Kirby, Oracle 0.40 > RTA 0.07 — "except Kirby" ✓.
- **Caption "only method that attains non-zero success on Kirby":** all reward-free baselines = 0.00 on Kirby; RTA = 0.07; Oracle (env reward) = 0.40. ✓ (among reward-free methods)
- **RTA beats Oracle on 3/4 levels:** L2 +0.50, L4 +0.65, L6 +0.12; Kirby −0.33. ✓ (the counterintuitive reward-free > sparse-env-reward result)
- **Table 8 "RTA positive on all 4 levels":** 0.76/0.87/0.42/0.13 — only method with all-positive; max on L2/L4/L6; Kirby 0.13 = max (others 0.00). ✓
- **Table 2 "L2 = 1.00 across all sources":** Youtube/Full-Catrap/MetaWorld/Coin all 1.00 on L2. ✓ Cross-source ordinal-prior transfer.
- **Table 7 default (M=15, freq=15, L=2) = 1.00 / 37 steps is the update-step minimum:** L=2's 37.00 < L=1's 143.66 and L=4's 123.33; freq=15's 37.00 < freq=5's 175.60 and freq=25's 128.67; M=15's 37.00 ≤ M=25's 43.33 ≪ M=5's 65.33. ✓ Default is efficiency-optimal.
- **Table 6 VLM/MLP schedule inversion:** VLM every-15 (1.00/0.60/0.33) ≫ VLM only-end (0.33/0.00/0.00); MLP only-end (1.00/1.00/1.00) ≫ MLP every-15 (0.79/0.27/—). ✓ Opposite-optimal-schedule confirms backbone-dependent exploration dynamics.

**No numeric prose-vs-table contradiction.** All 8 tables transcribed verbatim with sourcing line-ranges; continuous-control success rates (Figs 3/4) and Stage-1 convergence (Fig 7) are figure-only and quoted as directional/prose-confirmed claims, not back-filled.

---

## 8. Strengths, Limitations, Verdict

**Strengths**
- **Genuinely reward-free:** no extrinsic reward, env API, or task annotation at any stage — only expert video. The Spearman ρ ∈ [−1,1] reward is bounded and scale-invariant by construction (no calibration drift).
- **One scorer, many tasks:** a single Stage-1 scorer transfers across Catrap levels, games, MetaWorld, COIN, and YouTube sources (Table 2) — R2R, by contrast, pretrains per task.
- **Falsifiable coherence claim (Table 8):** RTA is the only reward whose cumulative-reward ↔ pass@5 Pearson correlation is positive on all 4 levels — the signal is informative, not merely bounded.
- **Reward-design, not VLM-bias, win:** MLP backbone also reaches 1.00 (Table 6), isolating the gain to the correlation reward.
- **Anchor+shuffle is a clean, well-motivated anti-shortcut device** — removes the "later-is-better" vacuous-monotone solution at the data-construction level.

**Limitations / honest scope**
- **Kirby is the failure edge:** RTA (0.07) ≪ Oracle env-reward (0.40) on the long-horizon game; "strongest except Kirby" is a real gap, not a rounding caveat.
- **Continuous control is figure-only:** MetaWorld/PointMaze success rates are curve endpoints (Figs 3/4), not a verbatim table; RTA also *underperforms GAIL-augmented* methods on exploration-heavy tasks (assigns high scores to plausible-but-unsuccessful trajectories).
- **Seed-count / config drift between T1 and T6:** L4 RTA 0.72 (5 seeds, VL-DAC) vs 0.60 (3 seeds, every-15) — within std, but the breakdown should not silently equate them.
- **T7 ablation "performance degrades at extremes"** actually describes *sample efficiency* (update-steps), not success rate, which stays ~1.00 — the caption conflates the two.
- **Workshop paper scope (7pp + appendices):** no human evaluation, no real-robot transfer, limited benchmark breadth; ordinal reward can still be fooled by visually-plausible non-progress (Fig 10 example).
- **KL-control and per-episode standardization** details are under-specified in the main text (target KL=0.1 only in appendix prose L897).

**Verdict.** RTA is a clean, well-motivated reward-free contribution: it replaces scalar reward regression with an *ordinal* Spearman-correlation signal, the anchor+shuffle device elegantly kills the "later-is-better" shortcut, and the cross-domain transfer (Table 2) + reward-coherence (Table 8) results are genuinely supportive of the "ordinal video supervision suffices" thesis. The reward-free > Oracle-env-reward result on Catrap L2/L4/L6 is striking. Honest weaknesses: Kirby and continuous-control are real edges where the ordinal signal weakens (long-horizon + exploration), and the continuous-control evidence is figure-only. A solid workshop paper whose core idea (correlation-as-reward) is reusable beyond video IL.
