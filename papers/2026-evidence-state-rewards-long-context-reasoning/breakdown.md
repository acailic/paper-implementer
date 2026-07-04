# MAVEN: Evidence-State Rewards for Long-Context Reasoning

**arXiv:** 2607.02073 (v1, 2 Jul 2026) | **Authors:** Ya Gao, Pekka Marttinen | **Affil:** Aalto University
**Source:** `paper.pdf` (16pp incl. appendices), `paper_layout.txt` (pdftotext -layout, 1114 lines).
**Subarea lineage:** long-context RL with **action-local process rewards over an editable evidence memory** — sibling to the agentic-RL / reward-design lineage (distribution-wise-rewards, vprm, opid, decomprl) but uniquely supervises *evidence-state transitions* (add/link/drop) rather than tool-use trajectories, final answers, or static evidence overlap. Counterpart to drowning-in-documents (which diagnoses attention-dilution at the architecture level); MAVEN attacks the *training-signal* level.

---

## TL;DR

Long-context RL usually rewards only the final answer or a *static* evidence-overlap score. MAVEN (Marginal-Value Evidence Navigation) gives the policy an **editable evidence memory** and rewards each `add` / `link` / `drop` / `answer` action by how much it changes an **answer-conditioned evidence-state value** V_ψ — the relative reduction in gold-answer NLL caused by the current memory, scored by a **frozen verifier** (no learned PRM). Adds are credited by online marginal gain (CAIG) **plus** hindsight leave-one-out Shapley credit (HCC); links by a 2nd-order synergy score; drops by whether removing the evidence improves the state. Across Llama-3.1-8B, Qwen2.5-14B, Qwen3-30B-A3B on LongBench v2 / LongReason / RULER, MAVEN beats outcome-only and evidence-identification RL by **+3.5 / +3.2 / +4.0** LongBench-v2-overall points, lifts evidence sufficiency **>85%**, and *lowers* distractor retention — i.e. it teaches the model to **revise** evidence, not just collect it.

---

## 1. Problem & Motivation

Long-context models fail by (a) premature commitment to a plausible distractor, (b) missing information distributed across distant spans, (c) failure to synthesize multi-hop evidence. Existing long-context RL rewards fall into two camps, both **extraction-centric**:

- **Outcome-only** rewards (Wan 2025; Wang 2025): final-answer correctness — sparse, no process signal.
- **Context/evidence-aware** rewards (Chen 2026; Guan 2026; Ping 2026): score *isolated* chunks, quoted spans, or a *final* extracted set. They supervise **what** evidence is found, not **how the evidence state evolves**.

MAVEN's claim: long-context reasoning is **stateful evidence navigation**, not one-shot extraction. A segment may only become useful after another is found; a plausible doc must be discarded once a later one reveals it as a distractor; two insufficient pieces may jointly bridge a hop. Extraction-centric rewards cannot encode this — they anchor on the first plausible chunk and stop (Fig 1).

---

## 2. Method

### 2.1 Problem formulation

Instance `x = (C, Q, a⋆)`: long context, question, gold answer. Policy `π_θ` parses its trajectory into ordered high-level actions `m_k`, each a contiguous token span `I_k`. **Four action types:**

| Action | Tag | Effect on memory `E_k` |
|---|---|---|
| add | `<add id=i> ... </add>` | append evidence span with source id `i` |
| link | `<link ids=i,j> ... </link>` | explain how `i`,`j` jointly support answer (memory unchanged) |
| drop | `<drop id=i> ... </drop>` | remove evidence `i` (irrelevant / redundant / misleading) |
| answer | `<answer> ... </answer>` | produce final answer |

Action order is **not predetermined**; the model may add, link, drop repeatedly before answering. `E_k = {e_1,…,e_{n_k}}` is the memory after action `k` (`E_0 = ∅`); add/drop mutate it, link does not; `E_F` = final kept set. **Structural constraints:** unique ids, drop/link must reference existing ids, `|E_F| ≤ K_max`, span-length cap — control cost + prevent degenerate add–drop loops.

### 2.2 Answer-conditioned evidence-state value (the core)

A **frozen verifier** `p_ψ` (Qwen3-4B-Instruct-2507) — *not* a trained PRM — scores whether a memory helps predict the gold answer. Teacher-forced NLL and normalized value:

$$\ell_ψ(Q,E) = -\tfrac{1}{|a^\star|}\sum_{j=1}^{|a^\star|}\log p_ψ(a^\star_j \mid Q, E, a^\star_{<j}) \tag{1}$$

$$V_ψ(Q,E) = \frac{\ell_ψ(Q,\emptyset) - \ell_ψ(Q,E)}{\ell_ψ(Q,\emptyset) + \epsilon} \tag{2}$$

`V_ψ > 0` ⇔ the evidence makes the correct answer more predictable; it is the **relative reduction in answer NLL** caused by the memory. ε avoids division by zero.

### 2.3 Action rewards

**Online Conditional Answer Information Gain (add):**
$$\mathrm{CAIG}_k(e_k) = V_ψ(Q,E_k) - V_ψ(Q,E_{k-1}) = \frac{\ell_ψ(Q,E_{k-1}) - \ell_ψ(Q,E_k)}{\ell_ψ(Q,\emptyset)+\epsilon} \tag{3}$$
"Does the newly added evidence improve the *current* state, conditioned on what is already collected?"

**Problem:** online gain alone **under-credits early multi-hop evidence**. If `e_i, e_j` are only useful together (`V(Q,{e_i})≈0`, `V(Q,{e_j})≈0`, `V(Q,{e_i,e_j})>0`), adding `e_i` first gets ~0 reward though it is necessary. **Fix — hindsight credit (HCC)**, a leave-one-out Shapley approximation over the final set:
$$\mathrm{HCC}(e_i; E_F) = [\,V_ψ(Q,E_F) - V_ψ(Q,E_F\setminus\{e_i\})\,]_+, \quad [z]_+=\max(z,0) \tag{4}$$
Items later dropped get **no** hindsight credit.

**Add reward** (α balances online vs hindsight, default α=0.6):
$$\rho^{add}_k = \alpha\,\mathrm{clip}(\mathrm{CAIG}_k(e_k), -c, c) + (1-\alpha)\,\mathbf{1}[e_k \in E_F]\,\mathrm{HCC}(e_k; E_F) \tag{5}$$

**Drop reward** (`E_k = E_{k-1}\setminus\{e\}`):
$$\rho^{drop}_k = \mathrm{clip}(V_ψ(Q,E_k) - V_ψ(Q,E_{k-1}), -c, c) \tag{6}$$
Positive when removal improves the state (recover from a distractor), negative when it harms (dropping useful evidence is penalized).

**Pairwise synergy (link)** — 2nd-order Shapley interaction; positive only if `e_i, e_j` are more valuable together than separately:
$$\mathrm{Syn}(e_i,e_j;E_F) = V_ψ(Q,E_F) - V_ψ(Q,E_F\setminus\{e_i\}) - V_ψ(Q,E_F\setminus\{e_j\}) + V_ψ(Q,E_F\setminus\{e_i,e_j\}) \tag{7}$$
$$\rho^{link}_k = [\mathrm{Syn}(e_i,e_j;E_F)]_+ \tag{8}$$
If either piece is not retained, or the pair is redundant, synergy → 0 and the link gets no positive reward. Multi-hop chains form an evidence graph over `E_F`.

**Answer reward:** binary substring match `ρ^{ans} ∈ {0,1}` (gold covered by final answer) — keeps training anchored to task success.

### 2.4 GRPO with action-local advantages

Per-type group-relative advantage (rewards grouped by action type `b(i,k)`):
$$\hat{A}_{i,k} = \lambda_{b(i,k)}\cdot\frac{\rho_{i,k} - \mathrm{mean}(R_{x,b(i,k)})}{\mathrm{std}(R_{x,b(i,k)})} \tag{9}$$
Token importance ratio and clipped GRPO objective (KL-regularized):
$$r_{i,t}(\theta) = \frac{\pi_θ(y_{i,t}\mid C,Q,y_{i,<t})}{\pi_{θ_{old}}(y_{i,t}\mid C,Q,y_{i,<t})}, \quad \mathcal{J}(\theta)=\mathbb{E}\Big[\tfrac{1}{G}\sum_i\tfrac{1}{|I_i|}\sum_{k,t\in I_{i,k}}\min\big(r\hat{A},\,\mathrm{clip}(r,1-\tfrac{\epsilon}{c},1+\tfrac{\epsilon}{c})\hat{A}\big)-\beta D_{KL}\Big] \tag{10,11}$$
**All tokens in action `k` share one advantage** — preserves process-level credit (a bad add span can go negative while a useful drop span goes positive in the same trajectory). **Verifier-call budget** `O(N_max + K_max + L_max)` (Eq 12) — cheap because the verifier conditions on `Q` + snippets, not the full context, and only teacher-forces the gold answer.

**Two-stage training:** (1) small **cold-start SFT** to teach the editable-memory grammar (no fixed action order, only reduces invalid rollouts); (2) GRPO with action-local rewards from the cold-start policy, with a **curriculum** (first 20% updates: context ≤ 16K, ≤ 3 evidence chunks) and action caps to stabilize early exploration.

**Inference:** trained policy gets only `(C, Q)` and emits an editable trajectory + answer (Eq 13); the verifier is **not used at inference**.

---

## 3. Experimental setup

| Item | Value |
|---|---|
| Policy models | Llama-3.1-8B-Instruct (8.01B), Qwen2.5-14B-Instruct (14.7B), Qwen3-30B-A3B-Instruct-2507 (30.5B total / 3.3B activated) |
| Frozen verifier | Qwen3-4B-Instruct-2507 |
| Cold-start SFT | lr 2e-5, global batch 16, 30 warmup; 2 epochs (Llama-8B, Qwen2.5-14B), 1 epoch (Qwen3-30B) |
| RL (GRPO) | rollout group G=8, global prompt batch 16, 1 epoch, lr 1e-6 cosine, 10 warmup, max response 4096, temp 0.8, top-p 0.95, KL coef β=0.001, GRPO clip ε=0.2, **local reward clip c=0.3** |
| Reward weights | λ_add=0.5, λ_drop=0.2, λ_link=0.3, λ_ans=1.0; α=0.6 |
| Action caps | K_max=6 (final memory), N_add=7, N_drop=3, N_link=4 |
| RL data | 9K examples (3K LongRLVR + 6K multi-hop: 1K HotpotQA + 1K 2WikiMultiHopQA + 4K MuSiQue), contexts 8K–64K |
| SFT data | 2K trajectories, contexts 4K–16K |
| Distractors | random filtered-out docs + teacher hard distractors (Qwen3-235B-A22B-Thinking-2507); partial-chain distractors sharing entities/relations/answer-types; single-evidence-shortcut filter `max_i V_ψ(Q,{g_i}) > 0.8·V_ψ(Q,G)` (Eq 14) |
| Hardware | 8× NVIDIA A100 + 4× AMD MI250X; **~1,250 GPU-hours** total (SFT+RL, 3 models, averaged over 2 envs) |
| Benchmarks | LongBench v2 (Short/Medium/Long/Overall), LongReason (32K/64K/128K + AVG), RULER (64K/128K + Avg; NIAH, variable tracking, SQuAD QA subsets); YaRN extends Qwen to 128K |
| Diagnostic set | 150 held-out multi-hop examples w/ gold-evidence + distractor labels → **evidence sufficiency** (frac of gold chunks covered in `E_F`) and **distractor retention** (frac of `E_F` that are distractors) |
| Reference models | Llama-3.1-70B, Qwen3-32B (Thinking), QwenLong-L1-32B |
| Framework | verl; vLLM for eval; 3-run average accuracy |

**Controlled baselines** (same data + optimization): base model; cold-start SFT; outcome-only RL; outcome + evidence-identification RL (F1 between final `E_F` and gold set); outcome + evidence-ID with prompted exploration.

---

## 4. Results

### 4.1 Main results — Table 1 (L354–381, verbatim)

> Bold = best method per trained model. Green = best across all listed models. **LongBench v2 Overall is the official (category-weighted) LongBench-v2 metric, NOT the mean of Short/Medium/Long** — e.g. Llama-8B MAVEN (39.8/36.2/32.1) → Overall 36.6, whereas the 3-split mean is 36.0. LongReason AVG = mean(32K,64K,128K); RULER Avg = mean(64K,128K) (both reconcile exactly, source-free).

| Model | LBv2 Short | Medium | Long | Overall | LR 32K | 64K | 128K | AVG | RULER 64K | 128K | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LLaMA-3.1-70B** (ref) | 42.8 | 38.0 | 31.2 | 38.3 | 61.2 | 63.3 | 48.3 | 57.6 | 93.2 | 69.8 | 81.5 |
| **Qwen3-32B (Thinking)** (ref) | 56.7 | 44.0 | 45.1 | 48.7 | 86.6 | 84.4 | 79.3 | 83.5 | 92.1 | 84.4 | 88.2 |
| **QwenLong-L1-32B** (ref) | 52.8 | 36.2 | 32.7 | 41.4 | 84.1 | 83.6 | 75.1 | 80.9 | 81.7 | 74.3 | 78.0 |
| LLaMA-3.1-8B | 33.3 | 30.7 | 22.5 | 29.9 | 51.4 | 49.9 | 46.5 | 49.3 | 85.1 | 77.2 | 81.1 |
| &nbsp;&nbsp;+ SFT | 34.5 | 30.4 | 24.1 | 30.5 | 51.0 | 49.2 | 47.1 | 49.1 | 85.6 | 77.4 | 81.5 |
| &nbsp;&nbsp;+ Outcome | 36.7 | 32.0 | 24.4 | 32.0 | 51.8 | 50.5 | 46.2 | 49.5 | 85.8 | 77.9 | 81.9 |
| &nbsp;&nbsp;+ Outcome+Evidence ID | 38.1 | 32.6 | 25.3 | 33.0 | 52.1 | 50.9 | 48.7 | 50.6 | 86.9 | 78.3 | 82.6 |
| &nbsp;&nbsp;+ Outcome+Evidence ID (Prompted exploration) | 37.6 | 32.9 | 25.9 | 33.1 | 52.4 | 51.3 | 48.9 | 50.8 | 86.6 | 78.5 | 82.5 |
| &nbsp;&nbsp;**+ MAVEN** | **39.8** | **36.2** | **32.1** | **36.6** | **55.9** | **55.2** | **55.8** | **55.6** | **88.4** | **80.1** | **84.3** |
| Qwen2.5-14B | 47.6 | 33.9 | 30.2 | 38.0 | 68.1 | 66.2 | 62.3 | 65.5 | 83.7 | 75.5 | 79.6 |
| &nbsp;&nbsp;+ SFT | 46.8 | 34.7 | 31.2 | 38.3 | 67.6 | 66.8 | 61.5 | 65.3 | 83.9 | 75.2 | 79.6 |
| &nbsp;&nbsp;+ Outcome | 48.1 | 34.9 | 31.2 | 38.8 | 69.5 | 67.4 | 62.0 | 66.3 | 84.1 | 75.7 | 79.9 |
| &nbsp;&nbsp;+ Outcome+Evidence ID | 49.3 | 36.1 | 33.0 | 40.1 | 70.3 | 67.8 | 64.7 | 67.6 | 86.7 | 76.9 | 81.8 |
| &nbsp;&nbsp;+ Outcome+Evidence ID (Prompted exploration) | 49.3 | 36.5 | 33.0 | 40.3 | 70.2 | 67.6 | 64.5 | 67.4 | 86.5 | 77.1 | 81.8 |
| &nbsp;&nbsp;**+ MAVEN** | **51.5** | **40.2** | **37.0** | **43.5** | **73.0** | **71.7** | **70.2** | **71.6** | **90.0** | **81.6** | **85.8** |
| Qwen3-30B-A3B | 50.7 | 39.4 | 40.7 | 43.7 | 84.8 | 82.9 | 77.1 | 81.6 | 88.2 | 82.6 | 85.4 |
| &nbsp;&nbsp;+ SFT | 49.3 | 38.1 | 38.9 | 42.2 | 83.8 | 82.6 | 76.3 | 80.9 | 88.5 | 82.7 | 85.6 |
| &nbsp;&nbsp;+ Outcome | 49.8 | 39.7 | 40.7 | 43.5 | 84.8 | 81.7 | 77.2 | 80.9 | 87.2 | 82.3 | 84.8 |
| &nbsp;&nbsp;+ Outcome+Evidence ID | 48.9 | 41.1 | 44.4 | 44.6 | 85.3 | 82.6 | 79.3 | 82.4 | 89.6 | 84.4 | 87.0 |
| &nbsp;&nbsp;+ Outcome+Evidence ID (Prompted exploration) | 49.3 | 41.6 | 43.8 | 44.8 | 85.6 | 82.5 | 79.6 | 82.6 | 89.8 | 84.5 | 87.2 |
| &nbsp;&nbsp;**+ MAVEN** | **53.9** | **45.9** | **46.3** | **48.8** | **86.6** | **85.1** | **81.7** | **84.5** | **93.4** | **88.8** | **91.1** |

**Verified prose claims (§3.2):**
- LBv2 Overall MAVEN − strongest baseline (Prompted-exploration) = **+3.5 / +3.2 / +4.0** for Llama-8B / Qwen2.5-14B / Qwen3-30B. ✓ exact.
- LBv2 **Long** split: Llama-8B 25.9→32.1 (+6.2), Qwen2.5-14B 33.0→37.0 (+4.0), Qwen3-30B 43.8→46.3 (+2.5) — largest gains on the hardest (Long) split. ✓
- **Qwen3-30B-A3B + MAVEN is best/tied-best on most columns incl. the Overall/Avg of all 3 benchmarks**: LBv2 Overall 48.8 > Qwen3-32B-Thinking 48.7; LongReason AVG 84.5 > 83.5; RULER Avg 91.1 > 88.2. (Caveat ⚠: not literally every column — Qwen3-32B-Thinking leads LBv2 **Short** 56.7 vs MAVEN 53.9; "most" is accurate.)
- **Llama-3.1-8B + MAVEN RULER Avg 84.3 > Llama-3.1-70B 81.5** — an 8B policy trained with MAVEN surpasses the 70B reference on RULER. ✓ (Even the base 8B is ~tied at 81.1; MAVEN pushes it clearly past.)
- **Qwen3-30B-A3B RULER Avg 85.4 → 91.1** with MAVEN — gains persist even on a strong long-context base. ✓
- Outcome+Evidence-ID beats outcome-only (dense supervision helps), but stays below MAVEN → **final-overlap reward alone is insufficient**; the model needs feedback on *how evidence is added, revised, synthesized*. Prompted exploration adds only marginal gain over Outcome+Evidence-ID → instructing exploration < explicitly rewarding useful transitions.

### 4.2 Contrastive answer scoring — Table 2 (L514–518, verbatim, Qwen2.5-14B, LBv2)

| Scoring | Short | Medium | Long | Overall |
|---|---|---|---|---|
| whole vocabulary (main) | 51.5 | 40.2 | 37.0 | 43.5 |
| contrastive answer set | 51.6 | 41.1 | 37.6 | 44.1 |

Concern: the verifier's prior knowledge may inject noise into full-vocab answer-NLL. Alternative: score the gold answer's normalized probability within a contrastive set (gold + similar-but-wrong + abstain). Result: **+0.6 Overall** — a small gain, "limited relative to its additional construction cost," so the main method uses full-vocabulary answer-token NLL as an efficient approximation. ⚠ Honest-scope: contrastive helps but is not worth the cost. (Cross-table consistency: the "whole vocabulary" row is byte-identical to the Table-1 MAVEN row and the Table-3 `w/ Qwen3-4B` row for Qwen2.5-14B — 51.5/40.2/37.0/43.5.)

### 4.3 Verifier choice — Table 3 (L876–888, verbatim, LBv2)

| Model / verifier | Short | Medium | Long | Overall |
|---|---|---|---|---|
| LLaMA-3.1-8B w/ Qwen3-4B | 39.8 | 36.2 | 32.1 | 36.6 |
| LLaMA-3.1-8B w/ Qwen2.5-7B | 40.2 | 35.8 | 32.3 | 36.6 |
| Qwen2.5-14B w/ Qwen3-4B | 51.5 | 40.2 | 37.0 | 43.5 |
| Qwen2.5-14B w/ Qwen2.5-7B | 51.8 | 40.4 | 36.9 | 43.7 |

Swapping the frozen verifier Qwen3-4B → Qwen2.5-7B changes Overall by **+0.0 / +0.2** — "only minor differences." MAVEN is not sensitive to the specific frozen verifier, so the smaller Qwen3-4B is used for efficiency. ✓

### 4.4 General short-context reasoning — Table 4 (L963–974, verbatim, MMLU-Pro, CoT)

| Model | Base | MAVEN |
|---|---|---|
| Llama-3.1-8B-Instruct | 44.3 | 45.7 |
| Qwen2.5-14B-Instruct | 64.0 | 64.2 |
| Qwen3-30B-A3B-Instruct-2507 | 77.5 | 76.9 |

Long-context RL training does **not** clearly degrade general short-context reasoning: +1.4 / +0.2 / **−0.6**. ⚠ The Qwen3-30B −0.6 is a small dip; the paper frames the trio as "no clear degradation" — defensible (within run-to-run noise of a 3-run average) but worth noting the largest model does dip slightly rather than improve.

### 4.5 Training dynamics & action-reward impact (Figures 3–5)

- **Fig 3** (training curves, Qwen2.5-14B): outcome-only RL improves slowly and saturates early; adding evidence-ID reward → stronger improvement; MAVEN improves more steadily with a higher ceiling. Diagnostic curves: Outcome+Evidence-ID raises **evidence sufficiency** but keeps **distractor retention** high; MAVEN pushes sufficiency **>85%** and substantially lowers distractor retention → it teaches *revision* (discard misleading chunks), not just retrieval.
- **Fig 4 / Fig 5** (action-reward ablation, Qwen2.5-14B): removing the **add** reward → largest drop in LBv2 Overall (and increases distractor retention); removing **link** → lowers evidence sufficiency (explicit synthesis matters); removing **drop** → highest distractor retention (confirms drop supervision drives evidence revision). Effects are **not isolated** to their own behavior — removing drop also reduces sufficiency, removing add raises distractor retention — supporting the central claim that long-context reasoning is a dynamic process with interacting actions. A **final-evidence-only** variant (add/link/drop scored only from `E_F`) is substantially below full MAVEN → **action-local process rewards** (not just final-set credit) are what teach memory construction/revision.

### 4.6 Hyperparameter sensitivity (Figs 6–9, Qwen2.5-14B subset)

- **Answer/process balance** (Fig 6): `λ_ans / (λ_add+λ_drop+λ_link)` peaks at **1.0** — balanced answer correctness vs process supervision. Under-weighting answer weakens final accuracy; over-weighting weakens evidence construction.
- **Process weights** (Figs 7–8): best near default `λ_add=0.5, λ_drop=0.2, λ_link=0.3`. Larger add weight → higher sufficiency (but ceiling on accuracy); larger drop weight → lower distractor retention (but too much removes useful evidence); link precision peaks at moderate link weight.
- **Online-vs-hindsight α** (Fig 9): best at **α=0.6** — online marginal progress slightly outweighs hindsight credit, both needed for stable evidence construction.

(Per-axis sweep endpoints are figure-axis-tick readings; only the prose-confirmed optima — ratio 1.0, α=0.6, default process weights — are quoted, consistent with the figure-derived-numbers-are-weak rule.)

---

## 5. Related-work positioning

- **Long-context grounding/reasoning**: architecture/positional extension (Chen 2023; Su 2024; Ding 2024), retrieval (Lewis 2020; Jiang 2024; Zhao 2024), agentic decomposition (Zhang 2024) — improve *access* to information but don't teach *reasoning over* long inputs. Closer line: long-context SFT/RL (Bai 2024; Wan 2025; Wang 2025; Chen 2026). Existing long-context RL optimizes **final-answer** or **static chunk** rewards; MAVEN rewards **evidence-state transitions**. Complementary to retrieval/agentic/architectural extension — targets a different bottleneck.
- **Reward design / process credit**: verifiable rewards (Jaech 2024; Guo 2025) give trajectory-level feedback; process supervision (Lightman 2024; Zhang 2025b; Khalifa 2025) assigns feedback to intermediate steps — but long-context reasoning needs process feedback over **evidence editing**, not free-form reasoning traces. MAVEN's action-local advantages over add/link/drop/answer distinguish it from static evidence rewards and learned black-box PRMs.

---

## 6. Strengths

1. **Mechanistically grounded reward, no learned PRM.** V_ψ is a *closed-form* answer-NLL reduction from a frozen verifier — no reward-model training, no reward hacking surface from a learned scorer; the verifier runs only at training time and only on `(Q, snippets)` (cheap; budget `O(N_max+K_max+L_max)`).
2. **Stateful, not extraction-centric.** Drop + link + hindsight credit encode exactly what extraction-centric rewards cannot: evidence revision, synergy, and late-revealed distractors. The drop reward's explicit *positive* signal for removing a misleading doc is the cleanest embodiment.
3. **Honest baseline ladder.** Outcome → +Evidence-ID → +Prompted-exploration → MAVEN under identical data/optimization isolates *what* drives the gain (state-transition feedback, not exploration prompting or mere dense supervision). The final-evidence-only ablation further isolates action-local vs final-set credit.
4. **Consistent across families/scales/benchmarks**, with the largest gains on the hardest (Long) split, and an 8B+MAVEN policy surpassing a 70B reference on RULER.
5. **Cheap-feasible training**: ~1,250 GPU-hours for 3 models; verifier is 4B; inference adds no verifier overhead.

## 7. Limitations

1. ⚠ **Requires verifiable gold answers at training time** to compute V_ψ — directly applicable only to tasks with checkable answers; open-ended generation needs new designs.
2. ⚠ **Edit interface assumes evidence = bounded text spans**; multimodal / unbounded evidence needs additional action designs + evaluation protocols.
3. ⚠ **Diagnostic evidence annotations are constructed for controlled analysis** (150 examples); broader human evaluation of evidence quality/faithfulness is future work — and §B explicitly warns generated evidence trajectories are **not guaranteed faithful explanations**.
4. ⚠ Gains are **concentrated where there is headroom**: Qwen3-30B-A3B is already strong (LBv2 Overall 43.7 base); MAVEN's marginal gains shrink on the strongest base, and the largest model dips −0.6 on MMLU-Pro. The Long-split gains (+2.5 to +6.2) are the headline, not the Short split.
5. Fig 4/5 ablation deltas and §4.6 sweep values are figure-bar/axis reads (only the prose-confirmed optima are quoted here).

## 8. Verdict

MAVEN reframes long-context RL from "find the right evidence once" to "build, revise, and synthesize an evolving evidence state" — and shows that an **answer-conditioned, frozen-verifier evidence-state value** yields clean, action-local process rewards (CAIG + HCC + synergy + drop-improvement) without a learned PRM. The reward design is the contribution: it is the first long-context-RL scheme to score *state transitions* (drop rewarded for *improving* the state; link rewarded only for genuine 2nd-order synergy) rather than static overlap, and the ablations cleanly attribute gains to this. The honest scope (verifiable-answer-only, text-span-only, modest gains on the strongest base) keeps the claim proportional. The most citable single result is the **drop reward** — direct positive credit for discarding a misleading document — which operationally distinguishes "evidence-state navigation" from "evidence extraction."

---

## Sourcing notes

- All 4 explicit tables (T1 L354–381, T2 L514–518, T3 L876–888, T4 L963–974) transcribed verbatim from `paper_layout.txt`.
- Source-free reconciliation: LongReason AVG = mean(32K,64K,128K) reconciles for **all 9 model rows** (3 reference + 6 policy); RULER Avg = mean(64K,128K) reconciles for **all 9**; headline LBv2-Overall deltas **+3.5/+3.2/+4.0** exact; LBv2-Long deltas +6.2/+4.0/+2.5 exact; cross-table Qwen2.5-14B MAVEN row byte-identical across T1/T2/T3.
- ⚠ LBv2 **Overall ≠ mean(Short,Medium,Long)** — it is the official category-weighted LongBench-v2 metric (e.g. 39.8/36.2/32.1 → 36.6 not 36.0). Do not recompute as a 3-split mean.
- Equations 1–14 cited by number; LaTeX glyphs in `paper_layout.txt` scramble some inline math but all numeric content + prose extract cleanly.
- 0 numeric prose-vs-table contradictions found (paper is internally consistent). Inline ⚠ notes are scope/rounding caveats, not defects.
