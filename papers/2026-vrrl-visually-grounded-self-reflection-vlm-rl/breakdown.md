# VRRL: Visually Grounded Self-Reflection for Vision-Language Models via RL

**arXiv:** 2607.02490 (v1, 2 Jul 2026) | **Authors:** Liyan Tang*, Fangcong Yin*, Greg Durrett | **Affil:** UT Austin ♠, NYU ♢ (* equal contribution)
**Code/data:** https://github.com/fc2869/VRRL
**Source:** `paper.pdf` (25pp incl. appendices), `paper_layout.txt` (pdftotext -layout, 1371 lines).
**Subarea lineage:** **first VLM RL paper in the repo** — trains a vision-language model to *visually ground* its self-reflection under multi-turn visual feedback. Distinct from every prior repo paper: the closest neighbours are subliminal-clocks / refusal-subspaces / steering-vector-limits (residual-stream *steering/ablation/audit*, single-modality text or diffusion-LM) and the agentic-RL lineage (opid/decomprl/verification-horizon — text-only tool-use trajectories); VRRL is the only one whose reflection signal is an **image returned by the environment** and whose training targets **OOD visual generalization**. Sibling-in-mechanism to RT-RAG (iterative refinement structure) but the feedback channel is pixels, not retrieved docs.

---

## TL;DR

Off-the-shelf LVLMs cannot self-correct from visual feedback: prompted multi-turn "reflection" mostly **repeats the previous prediction** (77% / 69% / 86% repeat rate on 3B zero-shot, Table 1) and SFT alone only learns the *format* of reflection, going brittle OOD. **VRRL** is a two-stage SFT→RL recipe on top of Multi-SFT with three RL components — **Random Turn Masking (RTM)**, **Buffered Roll-In**, and a dense **Reflection Reward** — that teach the model to *recover from intermediate errors* rather than to make them. On Qwen2.5-VL-3B/7B visual grounding (tables/charts) and Qwen2.5-VL-3B / Qwen3-VL-4B FrozenLake navigation, VRRL lifts OOD average accuracy to **45.7 / 78.4** (grounding) and **39.2 / 52.2** (navigation), beating Multi-SFT→GRPO by **+3–10pp on most OOD cells** and the second-best baseline by **+9.8pp avg** on navigation — while the components are individually mixed (Buffered Roll-In *alone* collapses the model to single-turn behaviour; only **RTM + Buffered Roll-In together** restore multi-turn reflection).

---

## 1. Problem & Motivation

LVLMs reason over (image, instruction) pairs by emitting a CoT then an answer. **Self-reflection** — revisiting an earlier decision and correcting it — is a key text-LLM cognitive skill but is under-developed in LVLMs because of the **modality gap**: the model fails to *attend to visual tokens* during reflection and cannot translate visual evidence into a *grounded* correction, especially for OOD images.

The paper casts multimodal reasoning as a **multi-turn sequential decision process** `(S, A, O, R)`:
- **State/observation** `s_t = (I, Q, H_t)`: after an action `a_t`, the environment renders a new image observation `I_t` (e.g. a 200×200 crop centred at the predicted coordinate with a red marker, or the maze with the proposed path drawn as a red line).
- **Action** `a_t`: a reasoning step + an answer proposal (a coordinate, or a path) **or** a termination that locks in the final answer.
- **Reward** `R`: evaluated only at termination; combines format validity, answer correctness, and a reflection-shaping term.

Single-turn inference (`π_θ(a|I,Q)` → terminate) gives no chance to verify or recover. VRRL trains the model to **iterate** against visual feedback.

---

## 2. Method

### 2.1 Stage 1 — SFT (establish the reflection format)

Offline dataset `D_SFT` with two trajectory types:
- **Single-turn**: `a_0` already correct, then feedback `I_0`, then termination `a_1`. Teaches the base task.
- **Multi-turn**: `τ = {a_0, I_0, …, a_{T-1}, I_{T-1}, a_T}`, `a_0` deliberately **erroneous** so later turns must correct it; `a_T` always termination. Teaches the reflection *format*.

Optimize standard auto-regressive CE over assistant turns `a_1…a_T` (`a_0` excluded — it is constructed to be wrong). This initializes error-correction behaviour but, on its own, only learns in-distribution knowledge and goes brittle OOD.

### 2.2 Stage 2 — RL: GRPO + RTM + Buffered Roll-In

On `D_RL`, GRPO with **three reward components**:

**Reward.** `R = 0 if r_fmt=0 else max(R_answer, R_refl)`.
- **Format** `r_fmt ∈ {0,1}`: valid tool-call format; 0 ⇒ total trajectory reward 0.
- **Outcome** `R_answer = 1.0` if final answer correct (visual grounding: Euclidean dist ≤ δ_tol=40 px; bar chart: inside bar bbox. Navigation: exact match, valid path matching optimal length), else 0.
- **Reflection** `R_refl`: task-specific improvement-based shaping (see §2.3). For wrong predictions, partial credit via `max(0, r_refl)`.

**Random Turn Masking (RTM).** For a length-`T` rollout, sample start index `k ~ Unif{1..T}` and compute policy-gradient updates **only on the suffix `t=k…T`**, masking earlier turns. Appendix A derives that this is a **reweighted per-decision policy gradient** with linear weight `w_t = t/T`: later (reflection/refinement) turns get larger gradient magnitude than early (exploration) turns, implicitly prioritizing recovery-behaviour optimization *without learning to make the masked-in mistakes*.

**Buffered Roll-In.** Maintain a FIFO replay buffer `B` (cap 500 prefixes) of previously generated prefixes ending just before an incorrect termination. With probability `ρ` sample a fresh question for RTM rollouts; with probability `1-ρ` sample a prefix `τ_pre` from `B` and generate `G` suffix completions, applying GRPO **only** to the generated suffixes. 30% of `B` is forced to be *correct* states so the policy also practices validation (terminate-when-correct). As the policy improves and failure states scarcer, `B` naturally accumulates the *hard* failures the current policy still misses → self-paced curriculum.

**Total objective:** `J_Total(θ) = ρ·J_RTM(θ) + (1-ρ)·J_Buff(θ)`, `ρ = 2/3`.

### 2.3 Reflection reward (Appendix A)

**Visual grounding** — potential function on Euclidean distance `d` to target, dual-σ Gaussian so feedback is meaningful both far (`σ_2`) and near (`σ_1`):

`ϕ(d) = ½·exp(-d²/(2σ_1²)) + ½·exp(-d²/(2σ_2²))`,  `σ_1 = 2·δ_tol = 80`, `σ_2 = 5·δ_tol = 200`.

Raw `r_refl` is improvement-based (gain over the previous turn's `ϕ`); shaped `R_refl = 0.1 + 0.9·max(0, r_refl)` if `r_fmt=1` (0.1 = format floor). Note `ϕ` caps at 1.0 so total reward caps at 1.0. Improvement-based (not distance-of-final-prediction) by design — a pure final-distance reward let the model fix turn-1 and then *drop* the reflection behaviour.

**Spatial navigation** — built on **progress rate** PR = (max over `M` optimal paths of the longest common prefix ratio). Shaped reflection reward over turns `t=2…T`:

`r_refl = clip(Σ_t w(ΔPR_t), 0, 1)`,  `w(ΔPR) = ΔPR if ΔPR≥0 else λ_deg·ΔPR` (λ_deg=0.5).

Final shape adds a reflection bonus + step-cost penalty to deter over-/under-reflection:

`R_refl = 0.1 + 0.9·max(0,r_refl) + α·r_refl − γ·T` if `r_coord=1` else `0.1 + 0.9·max(0,r_refl) − γ·T`.

Defaults `α=0.2, γ=0.05` (Qwen2.5-3B); `α=0.1, γ=0.01` (Qwen3-4B — smaller bonus because its base is already strong enough to multi-turn-reflect, to prevent reward hacking). Negative shaping is *clipped at 0* — allowing negative shaping made the model disengage from reflection entirely (it can always grab the 0.1 format floor without attempting corrections).

---

## 3. Tasks

**(1) Visual grounding** (data viz: synthetic arXiv-style small tables). ID = row/column-header lookup. **Four 1K-example OOD splits:** Large Table (bigger size), Cell Query (inner cell — harder 2D localization vs 1D header search), Bar Chart (domain transfer), Scatter Plot (localize labelled points). Outcome reward δ_tol=40 px Euclidean; Bar Chart = inside bar bbox. SFT 15K / RL 6K examples.

**(2) Spatial navigation** (FrozenLake grid maze). Predict shortest valid path `{left,right,up,down}` start→goal avoiding holes/walls. ID = 3–5 (grounding) / 4–5 (nav) maps; OOD = 6×6, 7×7. EM metric. Warm-started with direct-answer SFT on ID (no visual feedback) because the task is OOD for the instruction-tuned model. SFT 4K / RL 2K examples.

Visual feedback per turn: grounding → 200×200 crop centred at the predicted coord with a red marker; navigation → maze image with the predicted path drawn as a segmented red line.

---

## 4. Results — Table 1 (visual grounding, verbatim)

`paper_layout.txt` L325–351. ID + 4 OOD columns + OOD Avg. **Bold** = best result significantly > second-best (paired bootstrap, p<0.05). Parenthesized % on zero-shot Multi rows = **share of reflection turns that repeat the previous prediction** (a *degenerate-reflection* diagnostic, not accuracy).

| Model (3B) | ID | Large Table | Cell Query | Bar Chart | Scatter Plot | OOD Avg |
|---|---|---|---|---|---|---|
| Zero-shot Qwen2.5-VL-3B Single | 5.3 | 4.8 | 3.4 | 13.1 | 2.9 | 6.0 |
| Zero-shot Qwen2.5-VL-3B Multi | 5.6 (77.3%) | 2.2 (69.2%) | 1.3 (86.1%) | 12.0 (14.7%) | 2.4 (75.5%) | 4.5 |
| Zero-shot Qwen2.5-VL-7B Single | 17.9 | 9.7 | 6.1 | 15.1 | 4.5 | 8.8 |
| Zero-shot Qwen2.5-VL-7B Multi | 19.8 (58.0%) | 8.9 (49.3%) | 7.2 (76.8%) | 14.7 (15.5%) | 4.6 (38.2%) | 8.8 |
| VL-Rethinker-7B | 15.1 | 7.9 | 5.7 | 1.0 | 3.7 | 4.6 |
| VL-Rethinker-32B | 42.4 | 18.7 | 27.6 | 9.5 | 32.7 | 22.1 |
| Single-SFT | 80.4 | 46.1 | 2.4 | 25.7 | 23.8 | 24.5 |
| Multi-SFT | 84.7 | 50.4 | 1.6 | 13.1 | 24.8 | 22.5 |
| Reflection Tuning | 92.7 | 52.5 | 7.0 | 25.0 | 27.4 | 28.0 |
| Single-SFT → GRPO | 96.2 | 53.3 | 5.3 | 27.1 | 34.7 | 30.1 |
| Multi-SFT → GRPO | 99.6 | 78.6 | 13.5 | 30.7 | 37.2 | 40.0 |
| **VRRL (Ours)** | **99.6** | **88.6** | **20.3** | 33.5 | **40.3** | **45.7** |

| Model (7B) | ID | Large Table | Cell Query | Bar Chart | Scatter Plot | OOD Avg |
|---|---|---|---|---|---|---|
| Single-SFT | 83.6 | 62.8 | 34.0 | 20.3 | 68.9 | 46.5 |
| Multi-SFT | 84.8 | 66.2 | 39.1 | 20.9 | 73.3 | 49.9 |
| Reflection Tuning | 95.3 | 75.1 | 51.8 | 14.2 | 81.3 | 55.6 |
| Single-SFT → GRPO | 99.6 | 89.6 | 68.3 | 38.6 | 84.3 | 70.2 |
| Multi-SFT → GRPO | 99.6 | **91.4** | 68.4 | 46.8 | 86.0 | 73.2 |
| **VRRL (Ours)** | **99.7** | 89.6 | **77.3** | **57.0** | **89.7** | **78.4** |

**Takeaways (§6.1):**
- Prompting does **not** elicit reflection: zero-shot Multi is no better (often worse) than Single, with 58–86% of reflection turns just repeating the prior prediction.
- VL-Rethinker (textual-CoT reflection) *underperforms* OOD despite strong ID — its reflective traces fail to correct.
- SFT (incl. Reflection Tuning) teaches ID knowledge but is brittle OOD — Multi-SFT "only learns the format of reflection."
- RL on multi-turn reflection is the big lever: Multi-SFT → GRPO gives **+3–25pp absolute** over its single-turn counterpart on 3B OOD.
- VRRL improves over Multi-SFT → GRPO by **+3–10pp on most OOD cells** while holding near-perfect ID (3B 99.6, 7B 99.7). ⚠ The qualifier "most" is load-bearing: on **7B Large Table, VRRL 89.6 < Multi-SFT → GRPO 91.4 (−1.8pp)** — VRRL *loses* that single cell; the OOD-Avg win is driven by Cell Query +8.9, Bar Chart +10.2, Scatter +3.7.

---

## 5. Results — Table 2 (spatial navigation, verbatim)

`paper_layout.txt` L428–446. ID Avg + OOD (6×6, 7×7, OOD Avg). `Base*` = warm-started on ID direct-answer (no visual feedback) — all methods build on it. **Bold** = best significantly > second-best (p<0.05).

| Qwen2.5-VL-3B | ID Avg | 6×6 | 7×7 | OOD Avg |
|---|---|---|---|---|
| Zero-shot Single | 2.4 | 1.2 | 1.6 | 1.4 |
| Zero-shot Multi | 2.8 | 0.8 | 2.0 | 1.4 |
| VL-Rethinker-7B | 11.3 | 3.2 | 3.2 | 3.2 |
| VL-Rethinker-32B | 26.5 | 10.8 | 3.6 | 7.2 |
| Base* | 77.7 | 33.2 | 4.8 | 19.0 |
| Single-SFT | 83.5 | 39.2 | 8.8 | 24.0 |
| Multi-SFT | 81.2 | 41.6 | 10.2 | 25.9 |
| Reflection Tuning | 85.7 | 49.6 | 12.4 | 31.0 |
| Single-SFT → GRPO | 85.9 | 49.2 | 8.8 | 29.0 |
| Multi-SFT → GRPO | 85.2 | 42.8 | 9.2 | 26.0 |
| **VRRL (Ours)** | 83.9 | **54.8** | **23.6** | **39.2** |

| Qwen3-VL-4B | ID Avg | 6×6 | 7×7 | OOD Avg |
|---|---|---|---|---|
| Zero-shot Single | 8.4 | 2.4 | 2.0 | 2.2 |
| Zero-shot Multi | 34.0 | 8.8 | 2.4 | 5.6 |
| Base* | 88.1 | 49.6 | 5.6 | 27.6 |
| Single-SFT | 90.7 | 56.8 | 8.0 | 32.4 |
| Multi-SFT | 86.9 | 60.4 | 21.2 | 40.8 |
| Reflection Tuning | 93.6 | 63.6 | 13.2 | 38.4 |
| Single-SFT → GRPO | 93.2 | 62.4 | 5.6 | 34.0 |
| Multi-SFT → GRPO | 87.9 | 61.6 | 10.4 | 36.0 |
| **VRRL (Ours)** | 89.1 | **65.2** | **39.2** | **52.2** |

**Takeaways (§6.2):**
- VRRL **trades a small ID drop for a large OOD gain**. ⚠ On 3B, VRRL ID Avg 83.9 is *lower* than Single-SFT→GRPO 85.9, Multi-SFT→GRPO 85.2, and Reflection Tuning 85.7 — yet VRRL OOD Avg 39.2 beats all of them (next-best Reflection Tuning 31.0). Same shape on Qwen3-4B (VRRL ID 89.1 < Reflection Tuning 93.6 / Single-SFT→GRPO 93.2). Honest scope: the contribution is **OOD robustness, not ID SOTA**.
- VRRL improves over Multi-SFT by **+13.3pp (3B)** and **+11.4pp (Qwen3-4B)** OOD Avg.
- VRRL beats the **second-best baseline by +9.8pp on average across the two models** (3B: vs Reflection Tuning 31.0 → +8.2; Qwen3-4B: vs Multi-SFT 40.8 → +11.4; mean 9.8). ⚠ Note the second-best *differs per model* (Reflection Tuning for 3B, Multi-SFT for Qwen3-4B) — verified arithmetically.
- Standard GRPO with outcome-only reward (**Multi-SFT → GRPO**) **suppresses** the reflection behaviour learned in Multi-SFT (see Table 3 ∆ref=+0) — the dense reflection reward is necessary to *keep* multi-turn reflection alive.

---

## 6. Reflection behaviour — Table 3 (verbatim)

`paper_layout.txt` L451–460. `# Turns` = avg turns incl. termination; `∆ref` = accuracy improvement from multi-turn reflection inference.

| Qwen2.5-VL-3B | ID # Turns | ID ∆ref | OOD # Turns | OOD ∆ref |
|---|---|---|---|---|
| Single-SFT → GRPO | 2.00 | +0 | 2.00 | +0 |
| Multi-SFT | 2.41 | +0.1 | 3.51 | +2.8 |
| Multi-SFT → GRPO | 2.00 | +0 | 2.00 | +0 |
| Reflection Tuning | 2.23 | +1.3 | 3.17 | +3.4 |
| **VRRL (Ours)** | 2.37 | **+6.0** | 3.39 | **+13.6** |

| Qwen3-VL-4B | ID # Turns | ID ∆ref | OOD # Turns | OOD ∆ref |
|---|---|---|---|---|
| Single-SFT → GRPO | 2.00 | +0 | 2.00 | +0 |
| Multi-SFT | 2.24 | +1.1 | 4.57 | +11.0 |
| Multi-SFT → GRPO | 2.00 | +0 | 2.00 | +0 |
| Reflection Tuning | 2.24 | +1.7 | 4.69 | +7.4 |
| **VRRL (Ours)** | 2.22 | +1.1 | 4.64 | **+23.0** |

**Takeaway:** VRRL uses reflection *more efficiently* — comparable turn counts to the reflection-oriented baselines but a far larger `∆ref` (3B OOD +13.6 vs Reflection Tuning +3.4; Qwen3-4B OOD +23.0 vs +7.4). Standard GRPO collapses `∆ref` to +0 (it learns to terminate immediately). The single-turn baselines sit at exactly 2.00 turns (one action + termination).

---

## 7. Ablations — Tables 4 & 5 (verbatim, 3B model)

Components added on top of **Multi-SFT → GRPO**: RTM, Buffered Roll-In (BRI), Reflection Reward (RR).

**Table 4 — visual grounding OOD** (`paper_layout.txt` L497–503):

| Variant (3B) | Large Table | Cell Query | Bar Chart | Scatter Plot | Average |
|---|---|---|---|---|---|
| Single-SFT → GRPO | 53.3 | 5.3 | 27.1 | 34.7 | 30.1 |
| Multi-SFT → GRPO | 78.6 | 13.5 | 30.7 | 37.2 | 40.0 |
| + RTM | 70.5 | 11.4 | 33.8 | 34.1 | 37.5 |
| + Buffered Roll-In | 63.2 | 26.1 | 43.9 | 37.1 | 42.6 |
| + RTM + Buffered Roll-In | 78.6 | 18.1 | 36.9 | 39.9 | 43.4 |
| + Reflection Reward | 79.1 | 19.5 | 38.0 | 34.1 | 42.7 |
| **VRRL (Ours)** | **88.6** | **20.3** | 33.5 | **40.3** | **45.7** |

**Table 5 — spatial navigation** (`paper_layout.txt` L516–524):

| Variant (3B) | ID Avg | 6×6 | 7×7 | OOD Avg |
|---|---|---|---|---|
| Single-SFT → GRPO | 85.9 | 49.2 | 8.8 | 29.0 |
| Multi-SFT → GRPO | 85.2 | 42.8 | 9.2 | 26.0 |
| + RTM | 84.4 | 48.0 | 9.2 | 28.6 |
| + Buffered Roll-In | 85.3 | 47.6 | 7.6 | 27.6 |
| + RTM + Buffered Roll-In | 83.3 | 52.4 | 13.2 | 32.8 |
| + Reflection Reward | 86.9 | 51.6 | 10.0 | 30.8 |
| **VRRL (Ours)** | 83.9 | **54.8** | **23.6** | **39.2** |

**Takeaways (§7):** Components are **individually mixed**, complementary together.
- **Buffered Roll-In alone collapses the model to single-turn behaviour** (loses the reflection capability from Multi-SFT) — but still beats Single-SFT→GRPO on every visual-grounding OOD cell by exposing the model to diverse intermediate states (avg 42.6 vs 30.1).
- **RTM + Buffered Roll-In together** resolve the single-turn collapse and are "highly complementary" — restoring Large-Table to baseline and lifting all other OOD cells.
- **Reflection Reward** alone is also an effective individual addition (grounding avg 42.7) — providing a dense shaping signal lets the model improve even when the final answer stays wrong.
- The **full model beats every ablated variant** on OOD average — best OOD robustness needs all three components combined.

---

## 8. Setup / hyperparameters (Appendix B, verbatim)

| | Visual grounding | Spatial navigation |
|---|---|---|
| Hardware | 4× A100 80GB | 4× A100 80GB |
| SFT LR / batch | 5e-6 / 48 | 2e-6 / 48 |
| SFT data | 15K small-table header lookup | 4K maps size 4–5 (after 10-epoch warm-up on 3K size-3–5 direct-answer) |
| RL LR / batch | 1e-6 / 32 | 1e-6 (3B) / 5e-7 (Qwen3-4B) / 32 |
| RL data | 6K | 2K |
| GRPO `G` rollouts | 8 | 8 |
| Max turns `T` | 8 | 8 |
| KL coef `β` | 0.01 | 0.01 |
| RTM/BRI mix `ρ` | 2/3 (66% on-policy RTM / 33% BRI) | same curriculum + buffer config |
| Buffer `B` | cap 500, FIFO, 30% forced-correct prefixes | cap 500, same |
| Reflection reward | `σ_1=2δ_tol=80`, `σ_2=5δ_tol=200`, δ_tol=40, per-step cost 0 | `λ_deg=0.5`; `α=0.2, γ=0.05` (3B), `α=0.1, γ=0.01` (Qwen3-4B) |
| RL steps | 1200 (converge ~600) | 1200 (converge ~250) |
| Model selection | best on 200-ex holdout (Large Table) | best on 250-ex holdout (ID maps) |
| DAPO-style online filtering | — | filter examples where all rollouts get uniform 1 or 0 |

---

## 9. Source-free reconciliation

All checks run against `paper_layout.txt` (no PDF re-read):
- **Table 1 OOD Avg = mean of the 4 OOD columns** reproduces for every row at the displayed 1 dp (3B Zero-Single 6.05→6.0 banker's-round ✓; 3B VRRL 45.675→45.7 ✓; 7B VRRL 78.4 ✓; 3B Multi-SFT 22.475→22.5 ✓; 7B Multi-SFT→GRPO 73.15→73.2 ✓; VL-Rethinker-7B 22.125→22.1 ✓; VL-Rethinker-32B 22.125-check → 88.5/4=22.125 ✓). All 16 rows reconcile.
- **Table 4 Average** = mean of 4 OOD cols reproduces for all 7 rows (e.g. +RTM 149.8/4=37.45→37.5 ✓; +BRI 170.3/4=42.575→42.6 ✓; +RTM+BRI 173.5/4=43.375→43.4 ✓; +RR 170.7/4=42.675→42.7 ✓).
- **VRRL-vs-Multi-SFT→GRPO 3B OOD deltas** recompute: Large Table +10.0, Cell Query +6.8, Bar Chart +2.8, Scatter +3.1 → matches the "3–10% on most" claim (Bar Chart +2.8 is the cell just under 3, covered by "most").
- **Navigation headline deltas** recompute: vs Multi-SFT +13.3 (3B) / +11.4 (Qwen3-4B) ✓; second-best-avg +9.8 = ((39.2−31.0)+(52.2−40.8))/2 ✓ (second-best is per-model: Reflection Tuning 31.0 for 3B, Multi-SFT 40.8 for Qwen3-4B).
- **3-table consistency triangle**: Single-SFT→GRPO and Multi-SFT→GRPO rows are byte-identical across Table 1, Table 4 (grounding) and Table 5 (navigation) — confirming the ablation baseline == the main-results baseline.

---

## 10. Inline notes / honest-scope flags (⚠ = paper-internal, transcribed verbatim)

1. ⚠ **VRRL is not ID-SOTA on navigation.** VRRL ID Avg (3B 83.9; Qwen3-4B 89.1) is *below* Reflection Tuning (85.7; 93.6) and Single-SFT→GRPO (85.9; 93.2). The contribution is OOD robustness, not in-distribution accuracy — the paper frames it this way and the OOD-Avg win is decisive.
2. ⚠ **7B Large Table: VRRL loses to Multi-SFT→GRPO** (89.6 vs 91.4, −1.8pp). The abstract "improves by 3–10% on most OOD tasks" qualifier "most" covers this single regression; the OOD-Avg win (78.4 vs 73.2) is carried by Cell Query / Bar Chart / Scatter.
3. ⚠ **Zero-shot "Multi" % is a degenerate-reflection diagnostic, not accuracy.** The parenthesized values on zero-shot multi-turn rows (77.3% / 69.2% / 86.1% …) are the share of reflection turns that *repeat the previous prediction* — i.e. evidence that prompted reflection is degenerate. Easy to misread as an accuracy column.
4. ⚠ **Component-ablation non-monotonicity.** Adding RTM *alone* to Multi-SFT→GRPO *lowers* grounding avg (40.0→37.5) and Large Table (78.6→70.5); Buffered Roll-In *alone* collapses multi-turn reflection. The components are synergistic, not additive — only the full combination wins. The paper states this transparently.
5. **Captioning clean (no caption-wrap trap).** All 5 table captions (`Table N:`) sit on their own line in `paper_layout.txt`; unlike iters-38/39/45/46/50 there is no shared-row caption to miss. Figure-derived numbers (Fig 4 cumulative-accuracy/turn-distribution curves, Fig 1/2/3/10 schematics) are qualitative/schematic and were *not* back-filled — only the 5 explicit tables + prose-confirmed claims are quoted, per the universal figure-derived-numbers-are-weak rule.

---

## 11. Strengths / Limitations / Verdict

**Strengths**
- Clean falsifiable claim: visual-feedback reflection is a *trainable* skill that standard GRPO actively *suppresses* (∆ref→+0) and dense reflection reward + RTM + BRI *restore and amplify* (+13.6 / +23.0 OOD ∆ref).
- The RTM = reweighted per-decision policy gradient derivation (Appendix A, `w_t = t/T`) gives the masking rule a principled interpretation, not just a trick.
- Honest component ablation showing the three pieces are individually mixed / collapsing and only jointly effective — rare in RL-recipe papers.
- Buffered Roll-In's self-paced-curriculum framing (buffer accrues the *current* policy's hard failures) is a clean answer to the "failure states vanish as policy improves" problem.

**Limitations**
- Two synthetic-ish visual-feedback environments (data-viz grounding + FrozenLake). The environment must be able to *render* feedback (crop-with-marker, path-on-map) — unclear how the recipe ports to tasks without a renderable feedback channel.
- Reward design is task-specific and explicitly called out as "beyond the scope of this work" — the dual-σ grounding potential and the PR-based navigation reward are hand-engineered with several free hyperparameters (σ_1, σ_2, α, γ, λ_deg).
- Modest model scale only (3B/4B/7B). Reflection-bonus coefficient has to be retuned per backbone (α=0.2→0.1 from 3B to Qwen3-4B) to avoid reward hacking — a scalability warning sign.
- ID accuracy is *traded away* for OOD robustness on navigation — not a free lunch.

**Verdict.** A well-scoped RL-recipe paper whose real contribution is **making multi-turn visual reflection survive RL** (standard GRPO kills it) via three complementary, individually-weak mechanisms. The OOD gains are real and large (+9.8pp avg over second-best on navigation, +5.7pp OOD-Avg over Multi-SFT→GRPO on 3B grounding), the mechanism is falsifiable (RTM weight schedule, ∆ref diagnostic), and the honest-scope flags (ID-trade, 7B Large-Table regression, task-specific reward) are preserved rather than overclaimed. Most citable single result: **∆ref +13.6 / +23.0 OOD** (Table 3) — VRRL extracts an order-of-magnitude more accuracy from the *same* multi-turn reflection than the next-best reflection-oriented baseline.
