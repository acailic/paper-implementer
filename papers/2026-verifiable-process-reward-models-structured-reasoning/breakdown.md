# Verifiable Process Reward Models (VPRMs) for Structured Reasoning — Risk-of-Bias Assessment

**arXiv:** 2601.17223v1 (cs.CL, 23 Jan 2026) — https://arxiv.org/abs/2601.17223
**Full title:** *Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning*
**Authors:** Massimiliano Pronesti¹², Anya Belz², Yufang Hou¹³ — ¹IBM Research Europe (Ireland), ²Dublin City University, ³IT:U Austria.
**Source-first build:** all 7 numeric tables (1–7) transcribed verbatim from `paper.pdf` via `pdftotext -layout` (1021 layout lines); Table 8 is a qualitative worked example, summarized. No figure bar values back-filled (Fig 3 reward-dynamics is a curve, described qualitatively; Figs 1/2 are schematics).

---

## TL;DR

RL with verifiable rewards (RLVR) works for tasks with an *outcome* check (unit tests, exact-match). But outcome-only rewards give no signal about whether the model reasoned correctly *along the way*. Existing *process* reward models (PRMs) score intermediate CoT steps with **neural judges** — reintroducing opacity, bias, and reward hacking. This paper closes the gap with **Verifiable Process Reward Models (VPRMs)**: each reasoning step is checked by a **deterministic, rule-based verifier** grounded in domain guidelines, and the verifier is itself composed into the RL reward.

Applied to **risk-of-bias (RoB) assessment in medical systematic reviews** (Cochrane) — a task with a rigid guideline-defined decision tree per bias domain, so every intermediate step is programmatically checkable. Headline results (Qwen2.5-7B trained with GRPO + VPRM):
- **87.9 Acc / 76.7 F1** on COCHRANE FOREST — beats every pretrained model incl. Llama-3.1-405B (68.4 / 45.5) and GPT-OSS-120B (67.1 / 49.8).
- **+6.4 Acc / +6.5 F1** over outcome-only GRPO (81.5 / 70.2) on COCHRANE FOREST.
- **+9.7 Acc / +20.6 F1** over the best neural-PRM baseline (GPT-OSS-PRM 78.2 / 56.1).
- **Coherence 89.5** (vs 50.7 for the best pretrained Llama-3.1-405B) — VPRM-trained models follow their own decision logic and *then* get the answer right.

This is a **new subarea for the repo**: verifiable process rewards / step-level rule-based supervision. It is the process-supervision counterpart to the agentic-RL lineage (`demystifying-rl`/`verification-horizon`/`opid`/`are-we-ready`/`multi-turn-rl`), but uniquely replaces the neural step-judge with a deterministic one and proves a reward-separation guarantee.

---

## 1. Problem setup

### 1.1 Why RoB admits verifiable process supervision (§1, §2.4)
Risk-of-bias assessment scores a primary study per bias domain (randomization, allocation concealment, blinding, …) into low / high / moderate risk. The Cochrane RoB2 tool prescribes, **per domain**, a fixed sequence of assessment questions whose answers *deterministically* map to a risk label (Figure 1: the rule-based decision tree). That rigidity is what makes the task verifiable: each intermediate step (e.g. `Identify_randomization_report → reported`) has a gold answer obtainable by applying the rules, so a step verifier is a deterministic program — no learned judge needed.

### 1.2 The gap in prior work (§1, §5)
- **RLVR / outcome-only** (Guo 2025, Wang 2025c, Zeng 2025): verifiable, but only the terminal label is checked → no guarantee the model used a valid intermediate process.
- **Process supervision / neural PRMs** (Lightman 2024, Zhang 2025, Zou 2025): dense step feedback, but scored by neural judges → opaque, biased, reward-hackable.
- **No prior method** makes the *process* reward itself verifiable on tasks admitting deterministic symbolic checking. VPRM is the first, and (per §E) the first RL-based method for RoB assessment at all.

---

## 2. Preliminaries

### 2.1 Policy optimisation algorithms used (§2.1)
- **GRPO** (Shao 2024): sample G completions per input, normalise rewards into advantages `Aᵢ = (Rᵢ − E[Rⱼ]) / √V[Rⱼ]`, optimise a clipped + KL-regularised objective with strength β.
- **DAPO** (Yu 2025): builds on GRPO — drops the KL penalty, adds clip-higher (1−εH, 1+εL), token-level policy-gradient loss, overlong reward shaping, and **dynamic sampling** (filter prompts whose group accuracy is exactly 0 or 1, keeping only prompts with effective gradient). Empirically GRPO > DAPO in this paper's setting (Table 2).

### 2.2 Rule-based reward modelling (§2.2)
Reward is a binary verifier: `R(y) = 1` if output passes hand-crafted correctness rules else `0`. Deterministic, no learned preference model — the substrate VPRMs extend from outcome to process.

### 2.3–2.4 Systematic reviews & RoB (§2.3, §2.4)
Systematic reviews aggregate evidence via predefined search + inclusion criteria; Cochrane is the gold-standard repository. RoB assessment scores each included study per bias domain into low/medium/high risk, weighting studies in synthesis by credibility.

---

## 3. VPRM method (§3)

### 3.1 Reasoning trajectories and steps (§3.1)
For input x, the model emits trajectory `Y = (o₁,…,o_T)` under `πθ(Y|x) = Π πθ(oₜ|o_<t, x)`. Each step `t` carries **two discrete outputs**: a step identifier `sₜ ∈ S` and a step label `ℓ̂ₜ ∈ Lₜ` (the model's answer for that step). Domain guidelines specify the gold step identifier `s★ₜ` and gold label `ℓ★ₜ` per prefix.

### 3.2 Verifiers and process rewards (§3.2)
Two bounded scoring functions map (model output, gold) → [0,1]:
- `snₜ(sₜ, s★ₜ)` — step-identifier correctness
- `slₜ(ℓ̂ₜ, ℓ★ₜ)` — step-label correctness

Instantaneous step reward: `rₜ(Y;x) = wtn·snₜ + wtl·slₜ` (weights `wtn, wtl ≥ 0`).

Terminal outcome reward `r_label`: 1 iff the final risk value predicted from the full trace matches gold. Full verifiable process reward:
> **R(Y;x) = Σₜ rₜ(Y;x) + r_label**

Every component is a deterministic rule-based check (Figure 2 contrasts the outcome-only left panel with the step-verified right panel).

### 3.3 Reward-separation guarantee — Theorem 1 (§3.3 + Appendix A)
Let C = event Y is correct; `µc := E[R(Y)|C]`, `µᵢ := E[R(Y)|Cᶜ]`. Under mild conditions (i) finite variance, (ii) **reward separation µc > µᵢ**, (iii) sufficiently large group G:
> **Theorem 1.** GRPO and DAPO advantages satisfy `E[Â(Y)|C] > 0` and `E[Â(Y)|Cᶜ] < 0`.

i.e. correct trajectories get positive expected weight, incorrect ones negative — sound reasoning is rewarded *in expectation*. (Proof in Appendix A.1–A.3; rests on the Wen et al. 2025 theoretical framework.)

---

## 4. Experiments (§4)

### 4.1 Datasets — Table 1 (verbatim)
**Table 1.** Dataset statistics (train/test split applies only to COCHRANE FOREST EXT and RoBBR Cochrane).

| Dataset | Train | Test | Total | Avg tokens |
|---|---|---|---|---|
| COCHRANE FOREST EXT | 2651 | 295 | 2946 | 13,596.9 |
| COCHRANE FOREST | – | 1846 | 1846 | 12,722.8 |
| RoBBR Cochrane | 774 | 906 | 1680 | 9,084.6 |
| RoBBR Non-Cochrane | – | 2489 | 2489 | 7,940.7 |

**Takeaways:** training pool = 2,651 (COCHRANE FOREST EXT) + 774 (RoBBR Cochrane) instances; three disjoint test sets span in-distribution (COCHRANE FOREST, 1,846 instances from 48 Cochrane reviews / 202 forest plots) and two OOD generalisation probes (RoBBR Cochrane 906 from 204 papers/58 reviews; RoBBR Non-Cochrane 2,489 from 496 non-Cochrane reviews/496 papers). Full corpus ≈4M tokens. Per-risk-type breakdown in Table 6 (Appendix).

### 4.2 Synthetic step-label annotation (§4.2)
Silver step-level labels generated with **Llama-3.1-405B**, temperature 0.7, 2048-token limit, using the Figure-4 system prompt. Manual verification of 20 random traces (Table 7): 100% coherent, 100% correct steps, 96.7% correct labels — high-quality silver labels for VPRM training.

### 4.3 Experimental setup (§4.3)
- **Backbone:** Qwen2.5-7B (instruct). Two regimes: SFT with reasoning-trace augmentation, and RL with verifiable rewards.
- **SFT:** 5 epochs, per-device batch 1, lr 5×10⁻⁵, AdamW.
- **RL:** 3 epochs, lr 1×10⁻⁶, per-device batch 1, **16 sampled generations per batch**, gradient accumulation 8 steps. Two algorithms (GRPO, DAPO) × two reward types (verifiable outcome, VPRM process).
- **Hardware/framework:** Open-R1 (HuggingFace 2025), 8× NVIDIA A100 80GB, vLLM inference serving.
- **Metrics:** Accuracy and macro-F1 over discrete risk labels; plus **Coherence** (proportion of datapoints whose predicted risk matches the decision `D(ℓ̂₁,…,ℓ̂_T)` implied by their own step labels) and **Coherent Accuracy (CA)** (accuracy restricted to coherent instances) for §4.6 analyses.
- **Baselines:** 3 model families (Qwen 2.5, Llama 3.1, Granite 3.1) + DeepSeek-R1-distilled Qwen/Llama + OpenAI models, zero-shot; plus neural-PRM baselines (LLM-as-step-judge per Song 2025, and MedPRM Yun 2025).

### 4.4 Main results — Table 2 (verbatim)
**Table 2.** Evaluation across models on three datasets (Acc / macro-F1). "–" = unparsable/inconclusive output. Best bolded; second-best underlined.

| Model | Think | CF Acc | CF F1 | RoBBR-C Acc | RoBBR-C F1 | RoBBR-NC Acc | RoBBR-NC F1 |
|---|---|---|---|---|---|---|---|
| **Pretrained LLMs** | | | | | | | |
| GPT-4-0125 | ✗ | 52.4 | 41.6 | 56.0 | 47.9 | 47.8 | 42.3 |
| GPT-OSS-20B | ✓ | 61.4 | 43.9 | 56.4 | 50.3 | 46.3 | 42.8 |
| GPT-OSS-120B | ✓ | 67.1 | 49.8 | 59.5 | 51.0 | 48.8 | 44.2 |
| Qwen2.5-7B | ✗ | 32.9 | 31.6 | 35.8 | 34.1 | 36.4 | 34.5 |
| Qwen2.5-14B | ✗ | 39.0 | 35.1 | 37.0 | 35.5 | 35.4 | 32.5 |
| Qwen2.5-72B | ✗ | 51.3 | 42.1 | 56.1 | 51.0 | 47.5 | 43.6 |
| Llama-3.1-8B | ✗ | 36.4 | 30.6 | 34.5 | 32.1 | 36.4 | 32.5 |
| Llama-3.1-70B | ✗ | 38.8 | 30.2 | 49.5 | 40.0 | 42.5 | 38.9 |
| Llama-3.1-405B | ✗ | 68.4 | 45.5 | 59.4 | 44.0 | 52.5 | 39.8 |
| DeepSeek-Qwen-7B | ✓ | – | – | – | – | – | – |
| DeepSeek-Qwen-14B | ✓ | 33.3 | 19.2 | 35.8 | 23.5 | 35.4 | 23.5 |
| DeepSeek-Qwen-32B | ✓ | 40.8 | 35.9 | 44.9 | 40.4 | 46.4 | 41.3 |
| DeepSeek-Llama-8B | ✓ | – | – | – | – | – | – |
| DeepSeek-Llama-70B | ✓ | 44.2 | 33.5 | 57.3 | 41.2 | 48.3 | 42.7 |
| Granite-3.1-3B | ✗ | 24.4 | 23.6 | 22.2 | 21.8 | 13.7 | 14.9 |
| Granite-3.1-8B | ✗ | 24.7 | 22.0 | 35.8 | 31.6 | 33.2 | 28.2 |
| Granite-4.0-h-small (32B) | ✗ | 48.2 | 33.5 | 45.4 | 41.2 | 40.9 | 33.1 |
| **Our Models** | | | | | | | |
| Qwen2.5-7B-SFT | ✓ | 45.1 | 36.9 | 38.6 | 32.4 | 38.3 | 31.9 |
| Qwen2.5-7B-GRPO | ✓ | 81.5 | 70.2 | 63.1 | 58.0 | 56.8 | 45.1 |
| Qwen2.5-7B-DAPO | ✓ | 76.8 | 57.3 | 60.2 | 45.4 | 55.8 | 43.6 |
| **Qwen2.5-7B-GRPO-VPRM** | ✓ | **87.9** | **76.7** | **65.2** | **58.5** | **60.7** | **47.2** |
| Qwen2.5-7B-DAPO-VPRM | ✓ | 79.2 | 60.6 | 60.7 | 48.9 | 57.1 | 45.3 |

(CF = COCHRANE FOREST; RoBBR-C = RoBBR Cochrane; RoBBR-NC = RoBBR Non-Cochrane.)

**Takeaways:**
- **GRPO-VPRM wins all 6 columns** (3 datasets × Acc/F1). The win generalises to both OOD test sets (RoBBR Cochrane + Non-Cochrane never seen in training), so the gain is not dataset-specific overfitting.
- **vs strongest pretrained:** on CF, +19.5 Acc / +31.2 F1 vs Llama-3.1-405B (68.4/45.5); +20.8 Acc / +26.9 F1 vs GPT-OSS-120B (67.1/49.8). A 7B model beats a 405B and a 120B.
- **vs outcome-only RL:** GRPO-VPRM (87.9/76.7) vs GRPO (81.5/70.2) = **+6.4 Acc / +6.5 F1** on CF — the marginal value of *process* supervision on top of outcome verification.
- **GRPO > DAPO here:** both with and without VPRM, GRPO beats DAPO (87.9 vs 79.2; 81.5 vs 76.8). DAPO's dynamic-sampling/no-KL machinery underperforms vanilla GRPO on this compact structured task (small data, deterministic labels) — an empirical reversal of the usual "DAPO improves on GRPO" narrative.
- **SFT alone underwhelms:** Qwen2.5-7B-SFT (45.1/36.9) barely beats the 7B pretrained base (32.9/31.6) and far below RL — the reasoning-trace augmentation helps but RL with verifiable rewards is where the gain lives.

### 4.5 Neural-PRM comparison — Table 3 (verbatim, CF only)
**Table 3.** Neural judges vs rule-based rewarding vs verifiable process rewarding on COCHRANE FOREST.

| Method | Acc | F1 |
|---|---|---|
| *Neural PRMs* | | |
| Qwen2.5-7B-GRPO-PRM-GPT-OSS | 78.2 | 56.1 |
| Qwen2.5-7B-GRPO-MedPRM | 76.8 | 53.4 |
| *Verifiable Rewards* | | |
| Qwen2.5-7B (pretrained) | 32.9 | 31.6 |
| Qwen2.5-7B-GRPO (outcome) | 81.5 | 70.2 |
| Qwen2.5-7B-GRPO-VPRM | **87.9** | **76.7** |

**Takeaways:** neural PRMs beat outcome-only training but are themselves beaten by VPRM — best neural PRM (GPT-OSS judge) 78.2/56.1 vs VPRM 87.9/76.7 = **+9.7 Acc / +20.6 F1**. Learned step-judges inject noise/misalignment that deterministic guideline verification avoids; this is the paper's central empirical claim.

### 4.6 Outcome + process ablation — Table 4 (verbatim, CF only)
**Table 4.** Ablation on outcome and process reward components.

| Setting | Acc | F1 |
|---|---|---|
| *w/o Outcome Reward* | | |
| Steps-only process reward | 34.4 | 32.3 |
| Full VPRM | 40.2 | 35.3 |
| *w/ Outcome Reward* | | |
| Steps-only process reward | 83.1 | 71.8 |
| Full VPRM | **87.9** | **76.7** |

**Takeaways:**
- **Outcome reward is essential:** without it, even full VPRM collapses to 40.2 Acc (near the 32.9 pretrained base) — step-structure verification alone cannot optimise the task.
- **Process reward adds on top of outcome:** with outcome reward, steps-only = 83.1 Acc, full VPRM = 87.9 → **+4.8 Acc / +4.9 F1** from verifying step *correctness* (not just presence). The two signals are complementary, and combining them is best.

### 4.7 Coherence analysis — Table 5 (verbatim, CF only)
**Table 5.** Coherence and Coherent Accuracy (CA) for VPRM-trained Qwen models vs pretrained LLMs.

| Model | Coherence | CA |
|---|---|---|
| GPT-OSS-120B | 36.2 | 28.5 |
| Qwen2.5-72B | 44.3 | 24.9 |
| Llama-3.1-405B | 50.7 | 27.1 |
| **Qwen2.5-7B-GRPO-VPRM** | **89.5** | **75.0** |
| Qwen2.5-7B-DAPO-VPRM | 80.1 | 69.4 |

**Takeaways:**
- Pretrained models have **low coherence (36–51) and very low CA (25–29)**: even when their step-level reasoning looks self-consistent, it rarely produces a correct final judgement.
- VPRM-trained models hit **89.5 coherence / 75.0 CA** — they follow the decision logic faithfully *and* reach accurate conclusions when they do. GRPO-VPRM > DAPO-VPRM on both, mirroring Table 2.
- The gap between Coherence (89.5) and CA (75.0) quantifies residual reasoning-vs-label misalignment even after VPRM training.

---

## 5. Reward dynamics (§4.6, Figure 3)
Figure 3 plots three training-reward curves over steps (process / accuracy / thought-format):
- **Process and accuracy rewards track each other closely** — both rise sharply early, stabilise in the same oscillatory range, peak together. Improved step-level reasoning directly improves final-label correctness (the central VPRM claim, now shown dynamically).
- **Thought-format reward saturates fast and stays flat** — formatting is learned quickly and contributes little thereafter.

(Per-point values not back-filled: Figure 3 is a curve plot with shared y-axis 0–1 over ~2,000 steps; only the qualitative co-movement is sourced.)

---

## 6. Appendix tables

### Table 6 — Per-risk-type dataset statistics (verbatim)
9 RoB domains A–I (RoB2 tool): A random sequence generation, B allocation concealment, C blinding participants/personnel, D blinding outcome assessment, E incomplete outcome data, F selective reporting, G baseline outcomes similar, H baseline characteristics similar, I contamination.

| Dataset | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| COCHRANE FOREST EXT | 498 | 498 | 498 | 498 | 498 | 273 | 61 | 61 | 61 |
| COCHRANE FOREST | 330 | 330 | 330 | 330 | 330 | 112 | 28 | 28 | 28 |
| RoBBR Cochrane | 125 | 125 | 198 | 133 | 206 | 119 | 0 | 0 | 0 |
| RoBBR Non-Cochrane | 412 | 474 | 467 | 472 | 478 | 186 | 0 | 0 | 0 |

**Note:** domains G/H/I exist only in COCHRANE FOREST (EXT) — RoBBR covers A–F. Training data is concentrated in A–E.

### Table 7 — Manual verification of 20 silver-labelled traces (verbatim)
| Metric | Fraction |
|---|---|
| Coherent instances | 100.0% |
| Correct steps | 100.0% |
| Correct labels | 96.7% |

Two master's-in-NLP annotators checked (i) whether the step sequence was a valid decision path per Cochrane guidelines and (ii) whether each step+label was valid given the paper. All traces coherent, all steps correct, 96.7% labels correct — silver labels are training-grade.

### Table 8 — Worked example (qualitative, summarized)
The Cochrane forest-plot entry for *Hawkey 2015* (Figure 7) with its risk-of-bias map; rationales are not used for training (step labels only). Demonstrates the paper → forest-plot → risk-map alignment that defines one training instance.

---

## 7. Strengths, limitations, verdict

**Strengths**
- Replaces the opaque neural step-judge with a **deterministic, auditable verifier** — directly removes the reward-hacking surface that neural PRMs inherit. The verifier is just the domain guideline encoded as rules.
- **Theorem 1** gives a non-trivial guarantee (positive expected advantage for correct trajectories) under a checkable reward-separation assumption — rare for an empirical RL-paper, anchors why VPRM training converges to sound reasoning.
- Strong, consistent gains: GRPO-VPRM wins all 6 result columns, generalises OOD (RoBBR Non-Cochrane), and lifts Coherence from ~50 to ~90. The Table-4 ablation cleanly decomposes the contribution of outcome vs process reward.

**Limitations** (paper-stated + observed)
- **Requires deterministic, domain-specific rules** — tasks without well-defined intermediate reasoning steps can't directly benefit. VPRM's scope is exactly the class of structured-reasoning tasks admitting symbolic checking.
- **Single-domain empirical evaluation** (RoB assessment). Generalisation to other structured domains (legal, regulatory, checklist-based clinical) is claimed but not demonstrated.
- **DAPO < GRPO** here is under-explained — the paper adopts DAPO as a second algorithm but does not analyse why its dynamic-sampling/no-KL design underperforms on this task.

> **⚠ Paper-internal prose-vs-table inconsistency (flagged, not reconciled):** the abstract claims VPRMs achieve "up to **20% higher F1** than state-of-the-art models." The +20 magnitude matches the **Accuracy** column on COCHRANE FOREST (87.9 vs GPT-OSS-120B 67.1 = +20.8, or vs Llama-3.1-405B 68.4 = +19.5), *not* F1 (87.9-F1 76.7 vs best-pretrained-F1 GPT-OSS-120B 49.8 = +26.9). The paired abstract claim — "6.5% higher than verifiable outcome rewards" — *is* exact F1 (76.7 vs GRPO 70.2 = +6.5). So the abstract's "20% higher F1" appears to mislabel an Accuracy gain as F1; the true F1 gap to SOTA is larger (+26.9). Recorded verbatim per the source; readers should compare against Table 2 directly.

**Verdict:** a clean, well-motivated contribution — the first *verifiable* process reward (vs neural PRMs) with a theoretical guarantee, demonstrated on a real clinical-NLP task where the determinism assumption holds. The headline numbers are strong and the ablation is honest (steps-only without outcome reward collapses to 40.2). The main caveats are scope (one domain, rule-availability required) and the abstract's F1-vs-Accuracy label slip. Most citable contributions: (1) deterministic step-verifier replacing the neural judge, (2) Theorem 1 reward-separation guarantee, (3) the Coherence metric showing VPRM training aligns reasoning with conclusions (89.5 vs ~50 baseline).
