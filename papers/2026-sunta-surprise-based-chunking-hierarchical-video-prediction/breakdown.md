# SUNTA — Surprise-based Nested Temporal Abstraction for Hierarchical Video Prediction

**arXiv:** 2607.02087v1 [cs.AI], 2 Jul 2026
**Authors:** Tomoshi Iiyama, Masahiro Suzuki, Yutaka Matsuo (The University of Tokyo)
**Source:** paper.pdf (5.7 MB, **19 pp** — pdfinfo=19pp; `file` misreports **17pp**; trust pdfinfo [iters 66/67/68/69 → now 70]); paper_layout.txt = `pdftotext -layout`, 1157 lines.
**Subarea (NEW for repo):** hierarchical state-space models (HSSMs) for **video prediction / world modeling** with **surprise-based (prediction-error / Bayesian-surprise) temporal chunking** — repo's FIRST paper on HSSMs, video prediction, recurrent world models (RSSM), temporal abstraction, or chunk-boundary discovery. No prior repo paper covers world models for visual prediction, multi-timescale latent dynamics, variational temporal abstraction, or predictive-coding-driven segmentation.
**Sibling-in-spirit lineage:**
- *World-model / predictive-model lineage* (`physisforcing` physics-RL world simulator): both learn internal predictive models of environments. physisforcing is an RL/action world simulator; SUNTA is a **generative video-prediction world model with no action/RL loop** (action generation left to future work).
- *Structure-discovery / intrinsic-alignment lineage* (`jetspec`, `speculating-experts`, DSGNAR sketching): all replace a hand-set discretisation with a learned/intrinsic one. SUNTA replaces fixed-length chunking with surprise-driven boundaries that align to the environment's intrinsic temporal structure.
- *Gap-diagnosis / theory-first lineage* (offline-RL-generalization iter 61, DSGNAR iter 67): SUNTA names two falsifiable failure modes (Gap 1 hierarchical collapse, Gap 2 missing open-loop surprise) and gives each a targeted fix — a reusable "diagnose-then-fix" experimental template.

---

## 1. Problem & central diagnosis

**Setting.** Long-horizon open-loop video prediction: condition on 50 context frames, generate the next 250, evaluate SSIM + MSE vs ground truth. Three datasets with natural temporal hierarchies:
- **Bouncing Ball** (2D toy): colored ball bounces off walls, changes color at each bounce; post-bounce color = 6th-previous color (long-range dependency).
- **3D Maze** (egocentric, from VTA): action-conditioned dynamics, wall colors vary across episodes.
- **Serial Nine Rooms** (Miniworld): agent traverses 9 serially-connected rooms, top-down egocentric; each room's texture = room visited 3 rooms earlier (~120 timesteps delayed dependency).

**Prior HSSM chunking strategies (§2):**
1. *Fixed interval* — `mt=1` iff `t mod H=0` (CW-VAE, fixed timescales).
2. *Learning under constraints* — `mt` as latent var via variational posterior `q(m|X)` (VTA, LOVE).
3. *Similarity-based* — `mt=1` when successive latent reps differ past a threshold (VPR).

**Central diagnosis (Fig 1).** All three misalign with intrinsic temporal structure: fixed intervals ignore it; similarity-based over-segments superficial visual changes and misses non-salient semantic transitions. SUNTA's claim: **chunk boundaries should be driven by prediction error (surprise) in the internal world model**, because surprise reflects shifts in the underlying dynamics rather than superficial appearance.

**Two critical gaps that block naive surprise-based HSSMs (§1):**
- **Gap 1 — Hierarchical collapse under surprise minimization.** Low-level prediction errors mark boundaries; but as the high-level model improves, effective top-down conditioning makes low-level dynamics easier to predict → low level becomes less surprised → the surprise signal that *defined* the chunks erases → hierarchy cycles between emergence and collapse.
- **Gap 2 — Missing surprise during open-loop generation.** Surprise needs external observations to compare against; in open-loop rollout observations are unavailable, so bottom-up surprise can't trigger chunk switches.

**SUNTA's fixes:** (Gap 1) **decoupled level-wise training** — train each level independently & sequentially so high-level improvements don't suppress low-level surprise; (Gap 2) **top-down surprise** — replace observation-based bottom-up surprise with the mismatch between the evolving low-level rollout and the current high-level context (Eq 9).

---

## 2. Method (Eqs 1–12, Algorithm 1)

**Low-level model (level 1)** = RSSM backbone (DreamerV2-style CNN encoder/decoder + GRU), parameterized θ1 (§3, Eqs 1–3):

```
Posterior:  s_t^(1) ~ q_θ1(s_t^(1) | s_<t, x_t)        (1)
Prior:      ŝ_t^(1) ~ p_θ1(s_t^(1) | s_<t)             (2)
Decoder:    x̂_t ~ p_θ1(x_t | s_t^(1))                  (3)
```
Stochastic latent = diagonal Gaussian (Sigmoid-parameterized std). GRU posterior over latent history; CNN encoder over `x_t`.

**High-level model (level 2)** — **decoupled**: treats already-learned low-level latents as its observations (Eqs 4–7). `τ(t)` = chunk-start of `t` (Eq 4: `max{t' | t'≤t ∧ m_t'=1}`):

```
Posterior:        s_t^(2) ~ q_θ2(s_t^(2) | s_<t^(2), s_{τ(t):t}^(1))    (5)
Prior:            ŝ_t^(2) ~ p_θ2(s_t^(2) | s_<t^(2))                    (6)
Top-down Decoder: s̃_{τ(t):t}^(1) ~ p_θ2(s_{τ(t):t}^(1) | s_t^(2))      (7)
```
`q_θ2` = GRU encoder aggregating the chunk's low-level latents; `p_θ2` = GRU predicting next high-level state from past high-level states only; top-down decoder reconstructs the chunk's low-level sequence from `s_t^(2)`.

**Surprise-based chunking.** Surprise = Bayesian surprise = reverse-KL `D_KL(q‖p)` between posterior and prior (§3, §D). 
- *Inference (recognition) criterion* (Eq 8): chunk boundary when low-level surprise spikes:
  `D_KL( q_θ1(s_t^(1)|s_<t,x_t) ‖ p_θ1(s_t^(1)|s_<t) ) > τ_inf`  (8)
- *Open-loop generation criterion* (Eq 9, the Gap-2 fix) — top-down surprise between low-level rollout and high-level context:
  `D_KL( q_θ2(s_t^(2)|s_<t^(2), s̃_{τ(t):t}^(1)) ‖ p_θ2(s_t^(2)|s_<t^(2)) ) > τ_gen`  (9)
- Both thresholds realized as **SciPy `find_peaks` prominence** (single hyperparameter = 0.1, §D). Reverse-KL `D_KL(q‖p)` chosen because it "more reliably detects shifts in the posterior toward unexpected outcomes" (§D). Signals normalized by running max; sentinel = running mean appended at sequence end.

**Temporal Pattern Completion (TPC)** — bridges the train/test mismatch (training sees full chunk, online inference sees partial chunk). Eq 10 minimizes KL between the **complete-chunk posterior** `q^c` and a **partial-prefix posterior** `q^p`, plus a margin-`γ` contrastive term against a **negative-chunk posterior** `q^n`:
```
L_TPC(θ2) = D_KL(q^c ‖ q^p) + [γ − D_KL(q^c ‖ q^n)]_+      (10)
```
`[z]_+ = max(0,z)`, `γ=50` (Table 3). Encourages robustness to missing future context while staying discriminative across chunks.

**Training objectives.**
- Level-1 ELBO (Eq 11): `Σ log p_θ1(x_t|s_t^(1)) − Σ D_KL(q_θ1‖p_θ1)` — standard RSSM ELBO.
- Level-2 objective (Eq 12): reconstruction of low-level latents `Σ log p_θ2(s_{τ(t):t}^(1)|s_t^(2))` − high-level KL `Σ D_KL(q_θ2‖p_θ2)` − `L_TPC`.

**Algorithm 1 (L818–847) — two-stage decoupled training + open-loop generation:**
- *Stage 1:* init θ1; for 250k steps sample observation batches, compute level-1 ELBO, update θ1; **freeze θ1**.
- *Stage 2:* init θ2; for 250k steps infer `s_{1:T}^(1)` via frozen `q_θ1`, compute low-level surprise + detect boundaries `m_{1:T}` via `find_peaks`, sample partial-prefix + negative-chunk for TPC, compute level-2 ELBO + `L_TPC`, update θ2.
- *Open-loop generation:* infer `s_t^(2)` for most-recent chunk; for each rollout step decode top-down `ŝ_t'^(1)` + roll low-level prior; compute top-down surprise + detect boundaries `m̃_t'`; on boundary, transition high-level state via level-2 prior.

---

## 3. Experimental setup

**Datasets (§5.1):** Bouncing Ball (2D toy), 3D Maze (VTA-derived), Serial Nine Rooms (Miniworld). All images 64×64.
**Baselines (§5.2, 5 total):** RSSM (flat, capacity-boosted: det-dim 2048, stoch-dim 64), VTA (2-level, learned binary boundaries, Nmax=20/hmax=15), VPR (similarity-based; **boundary-only — cannot open-loop generate**), LOVE (VTA + compression, codebook 30, coding-length coef 0.1), CW-SUNTA (SUNTA's fixed-timescale ablation, chunk length 10).
**Training (§5.3, Table 3, App F):** ADOPT optimizer, lr 3e-4, batch 64, 250k steps/level, grad-clip 10, det-state 512, stoch-state 32, TPC margin γ=50, all loss weights=1.0, single NVIDIA H100, ~48h/level ⇒ ~96h/seed for 2-level.
**Metrics:** F1 (boundary detection vs ground-truth points), SSIM ↑, MSE ↓; mean±std over **3 random seeds** (Tables 1–2). Open-loop: condition 50 ctx frames, generate 250.

---

## 4. Results — tables verbatim

### Table 1 (L402–410) — Open-loop video prediction over 250 timesteps (SSIM ↑ / MSE ↓, mean±std, 3 seeds)

| Model | Bouncing Ball SSIM ↑ | MSE ↓ | 3D Maze SSIM ↑ | MSE ↓ | Serial Nine Rooms SSIM ↑ | MSE ↓ |
|---|---|---|---|---|---|---|
| RSSM | 0.911 ± 0.014 | 1395 ± 25.6 | 0.635 ± 0.019 | 7750 ± 25.1 | 0.686 ± 0.004 | 2325 ± 72.1 |
| VTA | 0.883 ± 0.004 | 1308 ± 16.2 | 0.471 ± 0.048 | 7355 ± 102 | 0.259 ± 0.029 | 5648 ± 371 |
| LOVE | 0.887 ± 0.010 | 1285 ± 19.8 | 0.517 ± 0.027 | 7194 ± 104 | 0.206 ± 0.009 | 5895 ± 373 |
| **SUNTA** | **0.981 ± 0.004** | **210 ± 27.8** | **0.933 ± 0.011** | **580 ± 48.0** | **0.858 ± 0.025** | **507 ± 104** |

*(4 models in T1: VPR is boundary-only; CW-SUNTA appears only in ablations T6/T7.)*

### Table 2 (L494–508) — Chunk boundary detection F1 ↑ (mean±std, 3 seeds)

| Model | Bouncing Ball F1 ↑ | 3D Maze F1 ↑ |
|---|---|---|
| VTA | 0.639 ± 0.082 | 0.577 ± 0.025 |
| VPR | 0.467 ± 0.034 | 0.174 ± 0.039 |
| LOVE | 0.276 ± 0.083 | 0.159 ± 0.008 |
| **SUNTA** | **0.984 ± 0.002** | **0.992 ± 0.003** |

*(Serial Nine Rooms has no F1 column — ground-truth boundaries unavailable there, §D.)*

### Table 3 (L770–788) — Hyperparameters
ADOPT optimizer; lr 3×10⁻⁴; batch 64; 250,000 training steps/level; grad-clip 10.0; image 64×64; recurrent(det) state dim 512; stochastic state dim (s^(1),s^(2)) 32; peak-detection prominence 0.1; TPC margin γ 50; all 5 loss weights (NLL recon, level-1 state KL, level-2→level-1 recon KL, TPC alignment KL, TPC contrastive margin) = 1.0.

### Table 4 (L881–894) — Sensitivity to peak-detection prominence

| prominence | F1 BB ↑ | F1 3DM ↑ | SSIM BB ↑ | SSIM 3DM ↑ |
|---|---|---|---|---|
| 0.001 | 0.712 | 0.834 | 0.917 | 0.706 |
| 0.01 | 0.989 | 0.989 | 0.955 | 0.818 |
| 0.05 | 0.994 | 0.996 | 0.986 | 0.940 |
| **0.1 (chosen)** | 0.995 | 0.996 | 0.986 | 0.942 |
| 0.3 | 0.995 | 0.996 | 0.984 | 0.927 |
| 0.5 | 0.978 | 0.982 | 0.935 | 0.693 |
| 1.0 | 0.706 | 0.853 | 0.903 | 0.491 |

### Table 5 (L937–943) — Parameter count (fixed hidden dim)

| Model | # Parameters ↓ |
|---|---|
| RSSM (flat) | 30.74M |
| VTA (2-level) | 44.26M |
| LOVE (2-level) | 44.21M |
| **SUNTA (2-level)** | **15.40M** |

### Table 6 (L1031–1039) — Independent ablation, Bouncing Ball SSIM ↑

| Variant | SSIM ↑ |
|---|---|
| SUNTA (full) | 0.986 |
| w/o surprise-based chunking (CW-SUNTA) | 0.918 |
| w/o temporal pattern completion (NoTPC) | 0.936 |
| w/o decoupled level-wise training | 0.880 |

### Table 7 (L1052–1061) — Incremental ablation, Bouncing Ball SSIM ↑

| Configuration | SSIM ↑ |
|---|---|
| RSSM (flat) | 0.904 |
| + End-to-end two-level RSSM (fixed chunk length) | 0.889 |
| + Surprise-based chunking | 0.883 |
| + Decoupled level-wise training | 0.936 |
| + Temporal pattern completion (= SUNTA) | 0.986 |

### Table 8 (L1092–1097) — Backbone generality, Bouncing Ball

| Variant | SSIM ↑ | MSE ↓ |
|---|---|---|
| SUNTA w/ GRU | 0.986 | 185 |
| SUNTA w/ xLSTM | 0.992 | 103 |

---

## 5. Source-free reconciliation (PASSED, 0 contradictions; 1 cross-table run-drift flagged)

**Headline deltas — all recompute EXACT from the displayed cells:**
- **MSE reduction vs RSSM (T1):** BB 1395→210 = **6.64×**; 3D Maze 7750→580 = **13.36×**; Serial Nine Rooms 2325→507 = **4.59×**.
- **SSIM absolute gain vs RSSM (T1):** BB +0.070, 3DM +0.298, SNR +0.172.
- **Boundary F1 (T2):** SUNTA BB 0.984 vs best baseline VTA 0.639 = **+0.345**; 3DM 0.992 vs VTA 0.577 = **+0.415**. Abstract's "near-perfect boundary detection" reconciles.
- **Parameter efficiency (T5):** SUNTA 15.40M is **2.0× fewer than RSSM** (30.74M), **2.87× fewer than VTA** (44.26M) and LOVE (44.21M); LOVE<VTA (44.21<44.26) and SUNTA fewest overall — matches §G "fewer trainable parameters than all baselines".
- **T7 incremental deltas:** flat RSSM 0.904 → +end-to-end two-level 0.889 (−0.015: hierarchy HURTS under end-to-end) → +surprise 0.883 (−0.006: even worse) → +decoupled 0.936 (+0.053: reverses collapse) → +TPC 0.986 (+0.050). Reproduces the Gap-1 hierarchical-collapse-then-recovery story quantitatively.
- **T6 independent ablation deltas (full 0.986):** w/o surprise −0.068, w/o TPC −0.050, w/o decoupled −0.106 — **decoupled is the most damaging single removal**, matching §I prose.
- **T4 prominence stability:** chosen 0.1 sits in stable range [0.05,0.3] (F1≥0.994, SSIM≥0.927); extremes degrade — 0.001 F1-BB 0.712 (over-segmentation), 1.0 F1-BB 0.706 (under-segmentation). Matches §D failure-mode descriptions.
- **T8 backbone transfer:** GRU 0.986/185 → xLSTM 0.992/103 (SSIM +0.006, MSE −44%); matches "preserved and slightly improved".
- **Two-level sufficiency (§K):** justified by Fig 10 — level-2 surprise goes flat once 6-bounce color history is observed; matches "additional levels unwarranted on current benchmarks".

**Cross-table run-drift (flagged, NOT a transcription error — iter-65 ReContext / iter-59 RTA defect class):**
- SUNTA Bouncing-Ball SSIM = **0.981 in T1** (3-seed mean over 250-step horizon) but **0.986 in T6, T7, T8(GRU), T4(prom 0.1)** — four independently-stated 0.986 values vs T1's 0.981. Same direction for MSE (T1 210 vs T8-GRU 185). And RSSM BB SSIM = 0.911 (T1) vs 0.904 (T7-flat). The ablations evidently use a single-seed / different-aggregation eval protocol than the 3-seed-250-step mean of T1; only the noisier SUNTA-vs-RSSM gap drifts ~0.005–0.007. Diagnostic: when a paper's main table reports a 3-seed-std aggregate while ablations report a single SSIM, expect a small systematic offset on the headline cell — do not treat the 0.981/0.986 mismatch as a contradiction.

---

## 6. Honest-scope notes (caveats that bound the claims)

1. **Cross-table SSIM/MSE drift (above):** SUNTA BB SSIM 0.981 (T1) vs 0.986 (T6/T7/T8/T4); RSSM 0.911 (T1) vs 0.904 (T7); MSE 210 (T1) vs 185 (T8). Ablations use a different eval protocol/seed than the 3-seed-250-step main table — the gap is ~0.005–0.007, not a contradiction but worth not echoing as identical.
2. **"250 timesteps / degrade within first 10" is figure-derived:** T1 is an aggregate over the 250-step horizon; the "all baselines degrade within the first 10 timesteps" claim is supported only qualitatively by Figs 4–5 (no per-timestep numeric table). Standard repo rule: figure-derived numbers are weak.
3. **Synthetic/simulated data only:** all three datasets are toy/simulated (Bouncing Ball, VTA-3D-Maze, Miniworld-Nine-Rooms) at 64×64; no real-world or high-resolution video (authors' own Limitation).
4. **Two levels only; deeper hierarchy untested:** §K restricts to 2 levels (level-2 surprise flat after 6 bounces); deeper hierarchies in multi-scale-dependency environments left to future work (authors' own).
5. **GRU-primary; xLSTM tested only on Bouncing Ball (T8):** backbone generality is single-dataset; Transformer/non-recurrent backbones untested (authors' own future work).
6. **"5 baselines" headline splits across tables:** §5.2 lists 5 (RSSM, VTA, VPR, LOVE, CW-SUNTA), but T1 video-prediction has only 4 — VPR is boundary-only (can't open-loop generate) and CW-SUNTA appears only in ablations T6/T7. The main video-prediction comparison is effectively SUNTA vs 3 (RSSM/VTA/LOVE).
7. **No wall-clock comparison vs baselines:** only parameter count (T5) and prediction quality are compared; no inference/training wall-clock-per-method table (only SUNTA's own ~96h/seed in App F).
8. **Large std on the headline Serial-Nine-Rooms cell:** SUNTA SNR MSE 507±104 (~20% of mean) — the single noisiest headline cell; SNR SSIM 0.858±0.025 also the widest SSIM std. Sub-noise caution on per-dataset magnitudes.
9. **Surprise signal needs discriminative peaks:** method fails if dynamics are uniformly unpredictable (no peaks to detect) — authors' own Limitation; also implies environments with continuous (non-event) dynamics are out of scope.
10. **Chunk-length-10 for CW-SUNTA is the sole fixed-length choice:** the "fixed chunking delivers no gain" conclusion (§5.4) is demonstrated at H=10 only; other fixed lengths untested, so the claim is H=10-specific.
11. **Predictive-only (no action/decision-making):** no policy learning or planning; action generation is future work (authors' own). Limits direct comparison to RL world-model lineage (physisforcing).
12. **CW-SUNTA "matches flat RSSM" (§5.4) rests on T6/T7 single-eval SSIM:** CW-SUNTA 0.918 (T6) ≈ RSSM-flat 0.904 (T7) under the ablation protocol; under T1's 3-seed protocol RSSM=0.911 — same drift regime as note 1.

---

## 7. Verdict

SUNTA makes a clean, falsifiable contribution: **surprise-driven (Bayesian-surprise / reverse-KL) chunk boundaries** for HSSM video prediction, unlocked by two targeted engineering fixes — **decoupled level-wise training** (kills Gap-1 hierarchical collapse; the single most damaging component to remove, T6) and **top-down surprise** (solves Gap-2 missing-observation chunking during open-loop rollout). The headline gains are large and internally consistent (MSE 4.6–13.4× lower than RSSM, F1 0.984/0.992 vs ≤0.639, with **2.0–2.9× fewer parameters**), and the incremental ablation (T7) quantitatively reproduces the collapse-then-recovery narrative.

Scope is honestly bounded by the authors (synthetic 64×64 data, 2 levels, GRU-primary, predictive-only) and by this breakdown's reconciliation (a small T1-vs-ablation SSIM/MSE run-drift; figure-derived "10-step degrade" claim; large std on the SNR headline cell). The strongest citable artifact for re-implementation is **Algorithm 1 + Eqs 8–10** (the two surprise criteria + the TPC regularizer) — the decoupled-training + top-down-surprise pair is the reusable mechanism; the RSSM/GRU backbone is off-the-shelf.
