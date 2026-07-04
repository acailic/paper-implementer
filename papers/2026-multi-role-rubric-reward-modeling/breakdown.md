# Multi-Role Rubric Generation for LLM Judging and Reward Modeling (MRRG)

**Paper:** "Many Voices, One Reward: Multi-Role Rubric Generation for LLM Judging and Reward Modeling"
**Authors:** Dazhi Fu¹², Jiuding Yang², Yiwen Guo³*, Jicong Fan¹*  (¹CUHK-Shenzhen School of Data Science, ²LIGHTSPEED, ³Independent)
**arXiv:** 2607.01830v1 [cs.LG], 2 Jul 2026 (9pp main + appendices, 27pp total)
**Source-verified:** `paper.pdf` + `paper_layout.txt` (pdftotext -layout, 1523 lines). All numeric tables transcribed **verbatim** with sourcing line-ranges; every delta recomputed by a source-free reconciliation script (see end). Figure-derived numbers are NOT back-filled (universal rule).

---

## TL;DR

Rubric-based LLM judges decompose an opaque quality score into verifiable yes/no criteria, but **existing annotation-free rubric generators use a single generic evaluator** and so miss preference dimensions outside that one viewpoint — a failure mode the authors name **"dimensional blind spots."** This causes (a) domain-dependent judging accuracy and (b) **rubric hacking** (a response satisfies the listed criteria while remaining low-quality on uncovered dimensions).

**MRRG (Multi-Role Rubric Generation)** is a **training-free, reference-free** fix: instruct one LLM to adopt **5 complementary evaluative roles** — {USER, DOMAIN EXPERT, EDUCATOR, AI RESEARCHER, LINGUIST} — each emits a compact rubric of atomic, verifiable criteria with self-assessed importance (3/2/1 = core/significant/polish); the per-role rubrics are concatenated and **exact-match deduplicated** (deliberately *not* semantically merged) into one auditable scorer `S(x,y;R̄) ∈ [0,1]` usable unchanged as either an LLM-as-a-judge *or* a GRPO reward.

**Headline (all prose-/table-confirmed):**
- **Preference validation:** MRRG beats the strongest listed baseline by **3.1–16.4 pp** across 5 backbones × 3 benchmarks (RewardBench-2 / JudgeBench / PPE); >20 pp over the no-sample-response single-voice baseline on RewardBench-2 for every Qwen2.5 size.
- **RLVR reward:** as the GRPO reward for Qwen2.5-3B-Instruct, MRRG lifts BiGGen Bench **62.0→63.7 (+1.7)** and HealthBench-Hard **28.7→32.1 (+3.4)** over the strongest single-voice reward.
- **Exact-match dedup beats LLM consolidation** (MRRGC) on every displayed model/benchmark.

---

## 1. Problem: Dimensional Blind Spots

Human judgment of open-ended responses is **distributed across complementary perspectives** (users want usefulness, domain experts want correctness, educators want clarity/evidence/safety). A single generic evaluator cannot recover the full structure of the latent quality function `q⋆: X×Y → ℝ`, so single-voiced rubrics omit preference-relevant dimensions. Two concrete failures:

1. **Domain-dependent judging accuracy** (Figure 1a): single-voiced rubric judgment performs well on technically-structured domains but degrades where domain-specific criteria are under-covered.
2. **Rubric hacking** (Figure 1b): a response can score well by optimizing covered criteria while remaining low quality.

**Table 1 (L148–182, case study, qualitative)** illustrates this for *"list and analyse the investment strategies of every NASDAQ-listed company"*: the **single-voiced rubric** rewards surface-level task completion (lists all companies / describes strategies / historical analysis) — its "passing answer" is a brief firm-by-firm overview of Apple and Amazon. The **multi-voiced rubric** instead requires acknowledging impracticality / setting scope boundaries, avoiding unsupported generalizations, flagging outdated data, providing a methodology rather than exhaustive listing, and acknowledging infeasibility — its passing answer reframes the task as a scoping+methodology problem. The single-voiced rubric is gameable; the multi-voiced one is not. (Table 9 / L893+ is a second qualitative case study on the ambiguous term *"huli"*.)

---

## 2. Method

### 2.1 Preliminaries: Rubric-Based Scoring

Classical reward models parametrize `q⋆` as a scalar network `rθ(x,y)` trained on preference pairs; LLM-as-a-judge instantiates it as a prompted model `q⋆(x,y) ≈ G(prompt_judge)(x,y)`. **Both collapse the multi-dimensional structure into an opaque scalar.** A **rubric** `R(x) = {c_k, w_k}_{k=1}^K` is a set of criteria, each `c_k` an atomic verifiable proposition; each is evaluated by the same model to `s_k = G(prompt_judge)(x,y,c_k) ∈ {0,1}` and aggregated:

$$S(x,y;R) = \frac{\sum_{k=1}^K w_k \cdot G_{\text{judge}}(x,y,c_k)}{\sum_{k=1}^K w_k} \in [0,1] \tag{1}$$

`G(·)` denotes a single call to a pretrained LM (model-agnostic); `prompt_judge` / `prompt_m` / `prompt_r` distinguish the per-criterion-judge / role-conditioned / generic rubric-generation templates.

### 2.2 Multi-Role Rubric Generation (the core idea)

Existing generators call `R(x) = G(prompt_r)(x, ŷ)` once, with `ŷ = G(x)` a sample response and a "universal evaluator" enumerating every criterion. Repeating this process only enlarges apparent diversity while staying inside one viewpoint's bias. **MRRG recasts rubric generation as multi-role elicitation.** Let `P = {p_1,…,p_M}` be a small fixed role pool; default `P = {USER, DOMAIN EXPERT, EDUCATOR, AI RESEARCHER, LINGUIST}` (M=5). Each role gets a template `prompt_m` foregrounding its concerns (USER → problem-solving utility / intent satisfaction; EDUCATOR → evidential support / communication quality / safety):

$$R_m(x) = G_{\text{prompt}_m}(x, ŷ) = \{(c_{m,k}, w_{m,k})\}_{k=1}^{K_m} \tag{2}$$

Each `R_m` is compact (`K_m ∈ [3,7]`) and internally coherent. The importance weight is produced **jointly** with the criterion inside the same role-conditioned call on a 3-point scale:

$$w_{m,k} = \begin{cases} 3 & \text{core need} \\ 2 & \text{significant issue} \\ 1 & \text{non-critical polish} \end{cases} \tag{3}$$

**Why a fixed (not question-selected) role pool — four reasons:** (i) selecting "most relevant" roles per question is itself ill-posed and would need an LLM role-selector that could inherit/amplify the very biases MRRG mitigates; (ii) DOMAIN EXPERT is already question-conditioned, giving question-aware specialization without exposing role-selection to model bias; (iii) the 5 default roles cover the principal human-evaluation axes (utility / correctness / clarity / AI-specific reasoning / linguistic quality); (iv) a small pool bounds rubric-generation cost to **O(M) LLM calls per prompt, independent of dataset size**.

### 2.3 Rubric Deduplication (exact-match, deliberately not semantic)

Concatenate all role rubrics:

$$R_{1:M}(x) = \{(c_{m,k}, w_{m,k})\}_{m=1,k=1}^{M,K_m} \tag{4}$$

then scan in order and remove criteria whose text **exactly matches** an earlier one (first occurrence kept with its original weight):

$$\bar{R}(x) = \{(\tilde{c}_j, \tilde{w}_j)\}_{j=1}^{N} \tag{5}$$

Only **exact duplicates** are removed; semantically similar or tense-in-criteria items are all kept. The authors deliberately reject LLM-based semantic consolidation for two reasons: (1) it adds a biased judgment step (redundancy/conflict decisions can propagate bias); (2) collapsing semantically similar criteria from independent roles **weakens important dimensions** — when several roles independently emphasize a criterion, that agreement is itself a signal of importance that merging would erase. The empirical cost of this choice is measured in the consolidation ablation (Table 6, §6.5).

### 2.4 From Rubric Score to Alignment Signal

`S(x,y;R̄)` plugs into both downstream settings **unchanged**:
- **Preference validation (LLM-as-a-judge):** given `(y_A, y_B)`, predict `argmax_{i∈{A,B}} S(x,y_i;R̄)`.
- **Reward modeling for RLVR:** use `S(x,y;R̄)` directly as the GRPO reward. `R̄(x)` is generated once per prompt and cached, so the per-rollout reward reduces to **N yes/no judge calls** that parallelize trivially. The group-relative advantage:

$$A_i = \frac{S(x,y_i;\bar{R}(x)) - \mu_S(x)}{\sigma_S(x) + \varepsilon} \tag{6}$$

(`µ_S, σ_S` = mean/std of the rollout-group scores; ε for numerical stability.)

---

## 3. Experimental Setup (L322–373)

- **Generator/judge backbones (preference validation):** Qwen2.5-3B/7B/32B-Instruct, GPT-OSS-20B, GPT-OSS-120B.
- **Benchmarks (preference validation):** **RewardBench-2** (Malik 2025), **JudgeBench** (Tan 2024), **PPE** (Frick 2024).
- **RLVR:** base policy **Qwen2.5-3B-Instruct**, GRPO; rubric generation by **GPT-4o**, per-criterion judging by **gpt-oss-120b**; **2000 WildChat** (Zhao 2024) training prompts; evaluated on **BiGGen Bench** (Kim 2024, 9 core capabilities across 77 tasks / 765 instances) and **HealthBench-Hard** (Arora 2025, 1000 hard examples, many frontier models score 0).
- **Baselines (3 single-voiced rubric generators):**
  - **SVRG w/o SR (S0):** generates rubrics for the question *without* sample responses.
  - **SVRG w/ SR (S1):** generates rubrics *with* sample responses.
  - **Chasing the Tail (CtT)** (Zhang 2025): generates/refines rubrics by comparing great and diverse responses.
  - **RSVRG** (ablation only): repeats single-voiced generation 5× then applies MRRG's post-processing — tests whether MRRG's gain is just "more criteria."
- **Hardware:** ~500 GB GPU memory; GRPO ~7 hours. (Table 5 hyperparams below.)

---

## 4. Results

### 4.1 Preference Validation — Table 2 (L334–346, verbatim)

Accuracy (%) on RewardBench-2 / JudgeBench / PPE. S0=SVRG w/o SR, S1=SVRG w/ SR, CtT=Chasing the Tail. **Best per model–benchmark in bold.**

| Model | RB2: S0 | S1 | CtT | **MRRG** | JB: S0 | S1 | CtT | **MRRG** | PPE: S0 | S1 | CtT | **MRRG** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 29.4 | 40.3 | 21.3 | **51.2** | 35.1 | 40.9 | 31.1 | **55.1** | 31.2 | 36.3 | 31.3 | **46.2** |
| Qwen2.5-7B-Instruct | 37.6 | 50.7 | 36.5 | **59.2** | 32.3 | 35.1 | 40.5 | **56.9** | 34.0 | 42.3 | 37.5 | **48.5** |
| Qwen2.5-32B-Instruct | 42.3 | 54.9 | 59.4 | **65.3** | 37.1 | 47.4 | 52.0 | **65.1** | 44.0 | 48.2 | 49.5 | **53.9** |
| GPT-OSS-20B | 51.2 | 62.4 | 60.3 | **73.7** | 54.3 | 67.7 | 63.1 | **74.0** | 40.1 | 52.2 | 53.2 | **56.3** |
| GPT-OSS-120B | 52.3 | 64.4 | 61.3 | **74.5** | 56.5 | 67.8 | 64.0 | **74.8** | 43.3 | 52.8 | 54.4 | **57.8** |

**Takeaways (every delta source-free-recomputed, see §7):**
- **MRRG is column-max on all 15 (model, benchmark) cells.** Gain over the strongest listed baseline (max of S0/S1/CtT) ranges **3.1–16.4 pp** — exactly the abstract's headline range (min = GPT-OSS-20B PPE +3.1; max = Qwen2.5-7B JudgeBench +16.4).
- On **RewardBench-2**, MRRG beats the SVRG-w/o-SR baseline by **>20 pp for all three Qwen2.5 sizes** (3B +21.8, 7B +21.6, 32B +23.0) and by +22.2/+22.5 on the GPT-OSS models.
- **GPT-OSS-120B** jumps **52.3→74.5** (RewardBench-2) and **56.5→74.8** (JudgeBench).
- The advantage is **backbone-agnostic** (consistent across Qwen2.5 and GPT-OSS scales/families), so it is not tied to a particular generator/judge.

### 4.2 Component Ablation — Table 3 (L410–422, verbatim, Qwen2.5-3B-Instruct)

| Variant | RewardBench-2 | JudgeBench |
|---|---|---|
| USER (role m) | 34.4 | 36.2 |
| DOMAIN EXPERT | 33.0 | 44.5 |
| EDUCATOR | 37.8 | 44.8 |
| RESEARCHER | 33.9 | 42.0 |
| LINGUIST | 35.1 | 37.4 |
| RSVRG w/o SR (5× single-voice) | 39.1 | 38.5 |
| RSVRG w/ SR (5× single-voice) | 43.9 | 45.1 |
| **MRRG (all 5 roles)** | **51.2** | **55.1** |

**Takeaways:**
- **No single role suffices:** every single-role variant (34.4–37.8 RB2 / 36.2–44.8 JB) trails full MRRG by a large margin. The best single role is **EDUCATOR** (37.8 / 44.8), still **−13.4 / −10.3** behind MRRG.
- **The gain is role-diversity, not criterion-count:** RSVRG repeats single-voiced generation 5× (matching MRRG's call budget) then applies the same post-processing. RSVRG w/ SR reaches 43.9 / 45.1 — better than any single role, but still **−7.3 / −10.0** below MRRG. Simply producing more single-voiced rubrics cannot substitute for explicit multi-role elicitation.
- **Leave-one-role-out robustness** (Figure 4, figure-bar reads — qualitative only): the most effective single role varies by backbone (no single perspective is uniformly optimal), but removing one role from MRRG causes only moderate drops; excluding a stronger single role gives a larger drop but the overall degradation stays limited. ⇒ MRRG does not depend critically on any one role.

### 4.3 Reward Modeling for RLVR — Table 4 (L449–459, verbatim)

Post-RL policy accuracy (%) under different reward sources (base policy Qwen2.5-3B-Instruct). Best in bold.

| Reward source | BiGGen Bench | HealthBench-Hard |
|---|---|---|
| Base policy (Qwen2.5-3B-Instruct) | 57.8 | 25.6 |
| SVRG w/o SR | 60.0 | 26.0 |
| SVRG w/ SR | 62.0 | 28.7 |
| Chasing the Tail | 61.1 | 27.0 |
| **MRRG** | **63.7** | **32.1** |

**Takeaways:**
- MRRG is the best reward on both benchmarks: **+1.7** over the strongest baseline (SVRG w/ SR) on BiGGen Bench, **+3.4** on HealthBench-Hard (the larger, domain-transfer gain).
- Every rubric-based reward improves over the base policy; MRRG gives the largest gain on both. The HealthBench-Hard margin is ~2× the BiGGen margin, supporting the claim that multi-role rewards generalize better under distribution shift.

### 4.4 Detailed Post-RL Results — Tables 7 & 8 (verbatim)

**Table 7 (L830–838): BiGGen Bench by capability** (best annotation-free reward per column in bold).

| Reward source | grounding | instr. follow | multilingual | planning | reasoning | refinement | safety | theory of mind | tool usage | average |
|---|---|---|---|---|---|---|---|---|---|---|
| Base policy | 72.5 | 67.5 | 32.9 | 52.1 | 68.2 | 54.6 | 61.7 | 62.2 | 30.7 | 57.8 |
| SVRG w/o SR | 73.0 | 69.8 | 35.0 | 58.6 | 70.0 | 57.9 | 56.7 | 64.8 | 38.6 | 60.0 |
| SVRG w/ SR | 73.3 | 72.0 | 32.1 | 60.0 | 72.3 | 63.2 | 61.1 | 67.8 | 40.4 | 62.0 |
| Chasing the Tail | 73.3 | 71.3 | 34.3 | 59.3 | 72.5 | 60.2 | 55.7 | 66.8 | 40.7 | 61.1 |
| **MRRG** | **74.0** | 70.2 | **38.9** | 57.9 | **74.3** | **66.1** | **66.7** | **68.2** | **42.1** | **63.7** |

> Note: MRRG is the annotation-free column-max on 7 of 9 capabilities + average; it is **not** max on instruction-following (SVRG-w/SR 72.0) or planning (SVRG-w/SR 60.0) — an honest-scope caveat that the average win is not a per-category sweep.
>
> ⚠ **Denominator distinction (BiGGen "average" is instance-weighted, not the 9-capability mean):** the unweighted mean of the 9 capability columns systematically **under-states** the printed average by **+1.66 to +1.98 pp** (Base 55.82 vs 57.8; MRRG 62.04 vs 63.7; all 5 rows). BiGGen Bench spans 77 tasks / 765 instances with **unequal per-capability instance counts**, so the "average" column is the official **instance-weighted** aggregate, not the simple 9-capability mean. (Table 8 / HealthBench-Hard's 5-dimension average, by contrast, **does** reproduce as the unweighted mean — gap ≤0.04 — so only the BiGGen column carries this weighting.) Parallels the LongBench-v2 "Overall"-vs-split-mean denominator distinction (MAVEN); transcribed verbatim, flagged not "reconciled".

**Table 8 (L841–849): HealthBench-Hard by dimension** (best annotation-free reward per column in bold).

| Reward source | accuracy | communication quality | completeness | context awareness | instruction following | average |
|---|---|---|---|---|---|---|
| Base policy | 21.5 | 40.6 | 14.9 | 16.9 | 34.0 | 25.6 |
| SVRG w/o SR | 22.9 | 40.0 | 17.1 | 18.9 | 30.9 | 26.0 |
| SVRG w/ SR | 25.8 | 43.0 | 19.5 | 19.5 | 35.5 | 28.7 |
| Chasing the Tail | 21.9 | 48.7 | 15.3 | 13.1 | 35.9 | 27.0 |
| **MRRG** | 24.3 | **55.1** | 18.7 | **20.5** | **42.0** | **32.1** |

> Note: MRRG is annotation-free column-max on communication quality (+12.1 over CtT's 48.7 — the headline driver), context awareness, instruction following, and average; it is **not** max on accuracy (SVRG-w/SR 25.8) or completeness (SVRG-w/SR 19.5). The +3.4 average win is concentrated in the communication/awareness/following dimensions.

### 4.5 LLM-Consolidation Ablation — Table 6 (L796–807, verbatim) ⚠

`MRRGC` = MRRG + an extra LLM step that semantically merges overlapping criteria and removes conflicting ones. Better per model–benchmark in **bold**.

| Model | RB2: MRRGC | MRRG | JB: MRRGC | MRRG |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 46.2 | **51.2** | 47.1 | **55.1** |
| Qwen2.5-7B-Instruct | 52.7 | **59.2** | 43.7 | **56.9** |
| Qwen2.5-32B-Instruct | 59.2 | **65.3** | 57.1 | **65.1** |

**Takeaway:** LLM consolidation does **not** help — MRRG beats MRRGC on every displayed model/benchmark (displayed RB2 gap +5.0/+6.5/+6.1; displayed JB gap +8.0/+13.2/+8.0). This empirically justifies the exact-match-dedup design (§2.3): semantic merging removes useful role-specific criteria and suppresses legitimate differences between evaluative perspectives, and adds a biased judgment step.

> ⚠ **Paper-internal prose-vs-displayed-table gap (flagged, not reconciled):** the §C.3 prose (L811–815) cites the consolidation-degradation range as **"5.0 to 15.0 points" on RewardBench-2** and **"8.0 to 14.6 points" on JudgeBench**, and states the degradation is *"particularly pronounced for the GPT-OSS models, where consolidation leads to more than 12-point drops on both benchmarks."* But the **displayed Table 6 contains only the 3 Qwen2.5 rows** — its max displayed gap is 6.5 (RB2, Qwen-7B) and 13.2 (JB, Qwen-7B). The values **15.0 (RB2) and 14.6 (JB) and the GPT-OSS >12-pt drops appear only in prose, not in any table** (grep of `paper_layout.txt` confirms 15.0/14.6 occur only at L811/L813). So the GPT-OSS-20B/120B consolidation rows are *referenced* by the prose but **absent from the displayed Table 6** — either truncated in extraction or never tabulated. The displayed Qwen rows are verbatim and self-consistent with the "5.0" / "8.0" lower bounds of the cited ranges.

### 4.6 GRPO Hyperparameters — Table 5 (L752–769, verbatim)

| Hyperparameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Use PEFT | False |
| Number of rollouts | 8 |
| Temperature | 1.0 |
| Top-p | 1.0 |
| Top-k | −1 |
| β (KL coef) | 0.001 |
| Learning rate | 1×10⁻⁶ |
| Batch size per device | 8 |
| Gradient accumulation steps | 4 |
| Number of epochs | 3 |
| Warmup ratio | 0.1 |
| LR scheduler | constant with warmup |
| Optimizer | adamw |

---

## 5. Figures (qualitative / figure-reads — NOT back-filled)

- **Figure 1:** (a) domain-dependent single-voiced judging accuracy; (b) rubric-hacking illustration. Qualitative.
- **Figure 2:** MRRG framework diagram.
- **Figure 3:** T-SNE of rubrics by role/method — role-specific rubrics cover the single-voiced regions and extend into underexplored regions (qualitative evidence of broader coverage).
- **Figure 4:** single-role vs leave-one-role-out MRRG bars on JudgeBench / RewardBench-2 (per-role bar reads — **not** reliably extractable; the verbatim single-role numbers live in Table 3, §4.2).
- **Figure 5:** per-domain RewardBench-2 / JudgeBench method comparison (bar reads).
- **Rubric-hacking experiment (Appendix C.2):** protocol = sample 400 query prompts, generate the rubric, prompt GPT-4o for an adversarial answer that satisfies the rubric while remaining low-quality on uncovered dimensions, then GPT-4o evaluates on a 0–10 scale. Motivation: a more comprehensive rubric should be harder to hack. **No numeric table is provided for this experiment in the layout extract** — only the protocol and prompt templates (Figures 7/8), so results (if any) are figure-only and not transcribed.

---

## 6. Strengths, Limitations, Verdict

**Strengths**
- **Training-free + reference-free:** no preference data, no gold answers, no learned reward model — any sufficiently capable LLM serves as `G`. Cost is O(M) calls per prompt for generation + N yes/no judge calls per rollout (cached rubric).
- **One scorer, two uses:** `S(x,y;R̄)` plugs into LLM-as-a-judge *and* GRPO reward unchanged.
- **Auditable:** every contribution `w_k·s_k` is attributable to a named criterion; the rubric is human-readable.
- **Falsifiable mechanism:** the RSVRG ablation (Table 3) cleanly separates "more criteria" from "role diversity" — the gain survives matching the call budget, so it is genuinely the multi-role elicitation, not quantity.
- **Backbone-agnostic** 3.1–16.4 pp gain across two model families and three scales.

**Limitations / honest scope**
- **Reliance on a strong generator/judge:** preference-validation gains are largest when the backbone is weaker (Qwen-3B +10.9–14.2) and shrink as the backbone strengthens (GPT-OSS-120B +3.1–10.1) — the multi-role prior matters most where the single evaluator is least comprehensive.
- **Open-ended-task evaluation only:** BiGGen Bench + HealthBench-Hard; no math/code/verifiable-answer benchmarks (where a deterministic checker already dominates).
- **Per-category RLVR win is not a sweep:** MRRG is not column-max on BiGGen instruction-following/planning nor on HealthBench accuracy/completeness (§4.4 notes). The average win is driven by communication-quality/awareness/following dimensions.
- **Role pool is hand-designed (M=5):** the leave-one-role-out robustness (Fig 4) is figure-only; the claim that no role is critical rests on bar-read evidence, not a verbatim table.
- **Table 6 prose-vs-displayed gap** (§4.5 ⚠): the GPT-OSS consolidation rows cited in prose are not in the displayed table.
- **Rubric-hacking experiment** has no reported numeric result in the layout extract (protocol only).

**Verdict.** MRRG is a clean, well-motivated, training-free contribution: it identifies a real failure mode (dimensional blind spots), gives a simple mechanism (fixed 5-role pool + exact-match dedup), and validates it with a cleanly-isolating ablation (RSVRG shows the gain is diversity not quantity; MRRGC shows exact-match dedup beats semantic consolidation). The 3.1–16.4 pp preference-validation gain and the +1.7/+3.4 RLVR gain are modest-but-consistent and honestly scoped (per-category wins are not sweeps; gains shrink on stronger backbones). The one weakness is reporting hygiene: Table 6 omits the GPT-OSS rows its own prose cites, and the rubric-hacking experiment reports no numbers — both worth tightening.

---

## 7. Source-Free Reconciliation (verification, no PDF re-read)

Python recomputation of every cited delta from displayed cells:

- **Abstract "3.1–16.4 pp":** MRRG − max(S0,S1,CtT) across all 15 (model,benchmark) cells → **min 3.1 (GPT-OSS-20B PPE), max 16.4 (Qwen2.5-7B JudgeBench).** ✓ exact.
- **">20 pp over SVRG-w/o-SR on RewardBench-2 for 3B/7B/32B":** +21.8 / +21.6 / +23.0. ✓
- **GPT-OSS-120B 52.3→74.5 (RB2), 56.5→74.8 (JB):** ✓
- **Table 3 EDUCATOR lag 13.4 / 10.3:** 51.2−37.8 / 55.1−44.8. ✓
- **Table 3 RSVRG-w/SR gap 7.3 / 10.0:** 51.2−43.9 / 55.1−45.1. ✓
- **Table 4 RLVR +1.7 / +3.4:** 63.7−62.0 / 32.1−28.7. ✓
- **Table 6 displayed MRRG−MRRGC:** RB2 {5.0, 6.5, 6.1}, JB {8.0, 13.2, 8.0} — matches the prose "5.0" / "8.0" lower bounds; the "15.0" / "14.6" upper bounds and GPT-OSS >12-pt drops are prose-only (see §4.5 ⚠).
- **Table 4 ↔ Table 7/8 cross-agreement:** BiGGen Bench 57.8/60.0/62.0/61.1/**63.7** and HealthBench-Hard 25.6/26.0/28.7/27.0/**32.1** match Table 4's headline reward-source cells exactly — a 2-table cross-agreement pinning the RLVR result. **HealthBench-Hard averages reproduce as the unweighted 5-dimension mean** (gap ≤0.04). **BiGGen Bench averages do NOT** — they run +1.66 to +1.98 pp above the unweighted 9-capability mean, confirming "average" is instance-weighted (see §4.4 ⚠).

**No numeric prose-vs-table contradiction** beyond the §4.5 Table-6 display gap. All 9 tables (1 qualitative, 8 numeric) transcribed verbatim with sourcing line-ranges.
