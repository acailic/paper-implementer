# WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory

**arXiv:** 2607.02517v1 (cs.CV, 2 Jul 2026) · **pdf:** `paper.pdf` (14.3 MB, **20 pp `pdfinfo`**; `file` misreports 13 pp — 7-page gap defect recurs) · **layout:** `paper_layout.txt` (1252 lines)
**Authors:** Hanlin Wang¹·², Hao Ouyang², Qiuyu Wang², Wen Wang³, Qingyan Bai¹·², Ka Leong Cheng², Yue Yu¹·², Yixuan Li⁴·², Yihao Meng¹·², Zichen Liu¹·², Yanhong Zeng², Yujun Shen², Qifeng Chen¹*  (¹HKUST ²Ant Group ³ZJU ⁴CUHK)
**Foundation model:** LingBot-World-Base [38] (post-trained). **Repo rank:** 71 (76th paper).

**Abstract (L51–64):** Highly controllable video world-model framework for *persistent dynamic object memory* + *unrestricted viewpoint exploration*. Unlike world models that entangle physical dynamics with pixel rendering and rely on continuous visual observation to sustain motion, WorldDirector **explicitly decouples semantic motion orchestration from visual generation** — an LLM coordinates 3D trajectories with camera movements, and those orchestrated trajectories become control signals for video generation. Claim: preserves exact visual identities of dynamic entities even when they re-enter the scene after prolonged out-of-view periods.

---

## 1. Problem & paradigm (L72–119)

Two foundational pillars for a world simulator with robust dynamic memory:
1. **Independent motion** — entity trajectories follow continuous physical logic *unconstrained by camera visibility*; unobserved dynamics progress naturally.
2. **Strict appearance consistency** — when a hidden entity re-enters, its visual identity / fine details remain intact.

Prior approaches criticized:
- **Monitor-based [16] (LiveWorld):** explicit "monitors" track + fast-forward unobserved active entities — scales poorly, prohibitive compute with many entities.
- **Implicit feature tracking [12] (HyDRA):** delegates trajectory extrapolation to internal generative priors — fails for prolonged occlusion / intricate interactions (trajectory collapse, frozen states, identity errors on re-entry).

**WorldDirector insight:** explicitly **decouple motion planning of dynamic objects from video synthesis**. Use controllable generation: transmit semantic-level planning as conditions to the generative model. An **LLM acts as central orchestrator** → translates user instructions into **3D bounding-box + camera trajectories** → projected to 2D bbox sequences → location conditions for video synthesis. **Appearance Binding** injects RGB dynamic-object features from context as visual anchors. **Spatial-aware cross-attention [41]** routes entity-specific text prompts to their regions. Integrated in a **causal autoregressive architecture**.

## 2. Method (L227–412)

### 2.1 Data curation pipeline (§3.1, L234–270)
- **Game-based platform** generates 15-s videos with precise camera params, scripted to induce target disappearances/reappearances (real-world data with clean exit/re-entry is scarce).
- **SAM3 [10]** extracts 2D bbox trajectories (robust re-id tracks objects through FOV exits).
- For training, sample a contiguous **5-s window** maximizing newly-visible objects (absent in first frame, appearing later) — captures re-entry. Remaining **10 s = candidate pool** for historical appearance + spatiotemporal context.
- Superimpose **unique color-coded bboxes** on source frames → **Qwen2.5-VL-72B [3]** generates fine-grained per-entity action captions.
- **Dual-Conditioning Preparation** (two conditioning videos per sequence):
  - *Spatial location condition video* — fill each dynamic object's 2D bbox with a unique color identifier against zero background.
  - *Appearance conditioning video* — for object `a` at frame `t` with bbox `box_{a,t}`, retrieve reference `box_{a,t'}` of the same object from the 10-s pool **minimizing aspect-ratio divergence** vs `box_{a,t}`; crop `box_{a,t'}` region, spatially resample, map onto `box_{a,t}` coordinates → exact visual features at designated spatiotemporal indices.
- **Static + Dynamic Context Retrieval** (dual-perspective, Alg 1 L860–894):
  - *Static* — top-K frames maximizing **FoV overlap** (following [50]).
  - *Dynamic* — greedy, prioritize frames with active object identities in the current chunk; **minimum temporal stride of 4 frames** between selected frames.
  - Interleave + deduplicate → final N memory-frame indices.

### 2.2 Model: conditional tuple (§3.2, L272–336)
Generative process conditioned on composite tuple **T = {B, A, P, M}** (L278–290):
- **Location Condition B ∈ ℝ^{T×3×H×W}:** precise spatiotemporal trajectories of 2D bboxes for all entities, rendered as identity-preserving color-coded masks.
- **Appearance Condition A ∈ ℝ^{T×3×H×W}:** sparse RGB features from contextual frames for dynamic-object appearance consistency.
- **Multi-Granularity Prompts P = {p_global, p_1,…,p_k}:** global prompt + fine-grained per-entity descriptions.
- **Contextual Memory Frames M:** retrieved via dual-stream selection, paired with their location+appearance conditioning (align feature dims with current window).

**Eq 1 (L314):** input latent fused by channel concat + Conv3D
> `z_in = Conv3D( z_t ⊕ E(B) ⊕ E(D_τ(A)) )`
> `z_t` noisy latent, `E(·)` pretrained 3D VAE encoder, `⊕` channel concat. `D_τ(·)` = Temporal Drop. Channel weights for `E(D_τ(A))` initialized from the first-frame processor (RGB identity transfer); weights for `E(B)` **zero-initialized** (trajectory-guided generation learned as a residual process).

**Eq 2 (L318–326): Temporal Drop Mechanism** `D_τ` — prevents over-reliance on A (sliding artifacts where entities translate without articulated motion). For entity `i` at global frame `t` with instance-relative frame index `k^(i)≥0` (frames elapsed since `i` newly entered view):
> `D_τ(A^(i)_t) = A^(i)_t` if `k<16`  (dense first 16 frames)
> `D_τ(A^(i)_t) = A^(i)_t` if `k≥16 and (k−16) mod 6 = 0`  (1 ref frame per 6-frame interval)
> `D_τ(A^(i)_t) = 0` otherwise  (masked → null embedding)
Full `D_τ(A)` aggregates entity-level reps; information bottleneck forces synthesis driven by trajectories + captions, A as identity anchor only.

### 2.3 Contextual integration (§3.2.2, L337–351)
- Context **M sequence-concatenated** (prepended to noisy latent along temporal axis).
- Location+appearance conditioning per context frame channel-concat'd, mirroring training input formulation.
- **RoPE offset:** context-frame timesteps shifted by an offset *substantially exceeding max training sequence length* → definitive frequency boundary.
- **Asymmetric attention mask:** context tokens exclusively self-attend (remain stable, noise-free references) — prevents noisy training latent from polluting M.
- **30% context-drop:** randomly discard M during training (so the model can generate the first chunk from scratch).

### 2.4 Camera injection + spatial-aware text (§3.3, L353–372)
- Convert camera poses of context frames + current chunk to **relative poses** w.r.t. first frame of current chunk. Encode via **Plücker coordinates [39]**.
- Spatial downsampling → conv modules → multi-level camera-motion embeddings → injected per DiT block via **adaptive normalization**.
- **Spatial-Aware Weighted Cross-Attention [41]:** identify visual tokens within each entity's 2D bbox trajectory, apply targeted spatial weight bias to pre-softmax attention logits between those tokens and the entity's text tokens → mitigates semantic leakage, fine-grained multi-object control.

### 2.5 Training objective + inference (§3.4, L374–412)
Flow-matching [31,17] post-training, **MSE loss**, applied **only to the current target segment** (context remains non-noisy reference).
- `x_1` GT latent of target chunk, `x_0 ~ N(0,I)`. At timestep `t∈[0,1]`: `x_{tgt,t} = t·x_1 + (1−t)·x_0`; GT velocity `v_t = x_1 − x_0`.
- Model receives `[x_ctx, x_{tgt,t}]` (clean context + noisy target).

**Eq 3 (L384–388):**
> `L = E_{x0,x1,t,Ω}[ Σ_{i∈I_tgt} ‖ u(x_t, t, Ω; θ)_i − v_{t,i} ‖² ]`
> `Ω = {B,A,P,M}`; `I_tgt` = token indices of current segment. Restricting loss to `I_tgt` ⇒ model learns to synthesize new content anchored by clean memory without reconstructing already-determined context.

**Inference — two stages:**
- *World Planning via LLM:* estimate 3D bboxes of target dynamic objects in initial image (SAM [26] + DepthAnything v2 [47]); LLM forecasts continuous 3D box trajectories (coords + orientations) + designed camera path. Objects absent from initial frame synthesized from captions when first entering view; subsequent generations conditioned on these for appearance consistency. 3D trajectories **projected to 2D image plane** → spatial condition B (matches training format). **Inference orchestrator = Gemini [37].**
- *Causal Chunk-Based Generation (Alg 2, L1020–1042):* partition B + camera poses into N chunks of size K. Global video buffer V init'd with reference frame I_0. Chunk 1: A from I_0 (guided by B), M=∅. Chunk n>1: A + M retrieved from updated buffer V. Last frame of buffer `V_last` = conditional initial frame `I_start` for next chunk (temporal smoothness at boundaries). Append generated chunk sans first frame → arbitrary-length exploration.

---

## 3. Experiments (§4, L414–599)

### Implementation (L415–438)
- Pre-trained **LingBot-World-Base [38]**. Resolution **832×480 @ 16 fps**. Context length **N=10** frames (each independently encoded by 3D VAE).
- Trained **3,000 steps**, global batch **64**, constant LR **1×10⁻⁵**, AdamW, BF16.
- **Compute (App. A, L836–854):** 8 nodes × 8 A100-80GB = **64 GPUs**, FSDP + activation checkpointing, **~72 h (3 days)** to converge.
- Inference: **Gemini** orchestrator; video partitioned into **5-s segments** generated chunk-by-chunk autoregressively.

### Baselines (L439–444)
Yume 1.5 [32] (uniform temporal downsampling memory); HY-World 1.5 [24] (FOV attention on mixed data); Infinite World [44] (hierarchical context compression); LingBot-World-Fast [38] (causal attention, infinite generation); HyDRA [12] (spatiotemporal retrieval for off-screen motion). ⚠ *HyDRA uses the first 10 s of WorldDirector's own output as its reference video (L466–468) — comparison is not fully independent.*

### Evaluation protocol (L445–452)
Test set: **100 video samples**, novel scenes/subjects unseen during training, **constructed via the authors' own data pipeline** (⚠ same synthetic game-platform distribution as training). Metrics:
- **PSNR / SSIM / LPIPS** — overall reconstruction fidelity (pixel-wise), following HyDRA [12].
- **VBench [23] Subject Consistency + Background Consistency** — frame-level coherence.
- **Dynamic Subject Consistency (DSC)** — crop YOLO-detected bboxes of dynamic objects, average **DINO** and **CLIP** similarity with contextual counterparts; captures dynamic-object consistency, especially off-screen reappearance (the paper's central claim).

### Table 1 — Quantitative results (L420–429). Best bold, runner-up underlined.

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | SubjCons↑ | BgCons↑ | DSC_DINO↑ | DSC_CLIP↑ |
|---|---|---|---|---|---|---|---|
| Yume1.5 | 14.391 | 0.455 | 0.425 | 0.898 | 0.919 | 0.765 | 0.898 |
| HY-World | 14.782 | 0.418 | 0.398 | 0.923 | 0.931 | 0.758 | 0.911 |
| Infinite-World | 14.574 | 0.431 | 0.406 | **0.934** | 0.908 | **0.773** | 0.913 |
| LingBot-World | 14.116 | 0.409 | 0.412 | 0.887 | 0.911 | 0.736 | 0.891 |
| HyDRA | 13.421 | 0.352 | 0.439 | 0.855 | 0.902 | 0.632 | 0.877 |
| **Ours** | **18.127** | **0.502** | **0.359** | 0.891 | 0.909 | 0.769 | **0.917** |

*Per-column best (higher better, except LPIPS lower):* PSNR/SSIM/LPIPS/DSC_CLIP → **Ours**; SubjCons → Infinite-World (0.934); BgCons → HY-World (0.931); **DSC_DINO → Infinite-World (0.773), Ours runner-up (0.769)**. ✓ bold/underline assignment consistent with "best/runner-up" rule.

**§4.1 prose claims (L457–465) — verified:**
- *"WorldDirector achieves state-of-the-art performance across all three reconstruction metrics"* (PSNR/SSIM/LPIPS) → **TRUE** (Ours best on all 3). ✓
- *"For the VBench results, Yume, HY-World, and Infinite-World attain the best performance"* → TRUE (SubjCons Infinite-World 0.934 best; BgCons HY-World 0.931 best; Yume is runner-up BgCons 0.919 — grouped as the top set that beats Ours on VBench).
- *"…they generate less subject or camera motion, giving them an inherent advantage"* — qualitative rationale for VBench loss.
- ⚠ **"Even though these methods also have an inherent advantage on the DSC metric due to their limited motion, our method still attains superior results."** → **OVERCLAIM / HALF-TRUE.** DSC has two columns: **DSC_CLIP Ours 0.917 wins** (vs Infinite-World 0.913), but **DSC_DINO Ours 0.769 LOSES to Infinite-World 0.773** (−0.004). DINO similarity is the *more identity-/structure-diagnostic* of the two encoders, and DSC is the paper's load-bearing metric for its central "persistent dynamic object memory / object permanence" thesis — so "superior results" on DSC is true on only 1 of 2 columns, and false on the more diagnostic one. *"This proves our method's strong capability in preserving dynamic consistency"* is therefore not supported by DSC_DINO. (Both decisive DSC deltas are 0.004 — within plausible run noise; **no CIs/seeds reported**.)

### Table 2 — Ablation on Appearance Condition (§4.2, L522–528)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | SubjCons↑ | BgCons↑ | DSC_DINO↑ | DSC_CLIP↑ |
|---|---|---|---|---|---|---|---|
| No A | 16.764 | 0.469 | 0.385 | 0.878 | 0.898 | 0.693 | 0.882 |
| No A + routing | 17.461 | 0.486 | 0.372 | 0.881 | 0.901 | 0.686 | 0.886 |
| **Ours** | **18.127** | **0.502** | **0.359** | **0.891** | **0.909** | **0.769** | **0.917** |

- **Cross-table byte-identity:** Ours row (18.127/0.502/0.359/0.891/0.909/0.769/0.917) is **byte-identical** in T1 and T2. ✓
- *"all metrics drop without the Appearance Condition"* (No A → Ours): **TRUE on all 7** (Ours strictly better than No A on every column). ✓
- ⚠ **Self-attention routing (No A + routing) is dismissed qualitatively but quantitatively helps 6/7 metrics.** Prose (L541–545): routing *"fundamentally disrupts the pre-trained latent distribution, inducing severe artifacts, blurring, and the loss of fine-grained textures"* (Fig 4, figure-only). But T2 shows routing *improves* No A on PSNR/SSIM/LPIPS/SubjCons/BgCons/DSC_CLIP and only **regresses DSC_DINO** (0.693→0.686, −0.007). The "routing fails" claim is figure-only/artifact-based and is in tension with the table where routing recovers most of the No-A regression — honest reading is "metrics improve but perceptual artifacts persist", which the dismissal understates.

### Further ablations (Appendix D, L1050–1064) — **figure-only (Fig S1)**
- Ablate dynamic context stream + Temporal Drop Mechanism → **qualitative only**, no numeric table. Claim: dropping dynamic context → semantically-similar-but-non-identical re-entering entities; dropping Temporal Drop → motion rigidity ("sliding" not walking).

## 4. Limitation (L608–610)
Relying on **synthetic game data** introduces a domain gap (unnatural locomotion, blurry faces). Future work: incorporate real-world datasets.

---

## 5. Source-free reconciliation summary (Python-verified)

- **Cross-table byte-identity T1-Ours == T2-Ours:** EXACT ✓ (all 7 cells).
- **Bold=best / runner-up:** all 8 columns consistent ✓.
- **"All three reconstruction metrics SOTA":** TRUE — Ours best on PSNR (18.127), SSIM (0.502), LPIPS (0.359) ✓.
- **PSNR delta:** +3.345 abs / **+22.6% rel** vs runner-up HY-World (14.782); **+28.4% rel** vs LingBot-World base (14.116) ✓.
- **⚠ DSC "superior results" OVERCLAIM:** DSC_DINO Ours 0.769 < Infinite-World 0.773 (−0.004); only DSC_CLIP Ours wins (0.917 > 0.913, +0.004). "Superior" true on 1/2 DSC columns, false on the more identity-diagnostic (DINO). No CIs.
- **"All metrics drop without A":** TRUE all 7 (No A < Ours) ✓.
- **Routing helps 6/7 metrics** (regresses only DSC_DINO 0.693→0.686) — tension with figure-only "routing disrupts distribution / severe artifacts" dismissal.
- **ZERO numeric prose-vs-table CELL typos.** The single load-bearing issue is the attributional DSC "superior" overclaim + the figure-only routing dismissal; every cited number / ratio recomputes EXACT.

## 6. Honest-scope flags (12, NO numeric cell typo — all attribution/framing/scope)

1. **DSC "superior results" overclaim (load-bearing, iter-72/88 blanket-"all"-with-one-loss class)** — §4.1 "our method still attains superior results" on DSC is half-true: DSC_CLIP Ours wins (0.917), but **DSC_DINO Ours 0.769 LOSES to Infinite-World 0.773**. DINO is the more identity-diagnostic encoder and DSC is the paper's central metric; "This proves strong capability in preserving dynamic consistency" not supported by DSC_DINO. Pair the headline with the DSC_DINO loss.
2. **DSC deltas are sub-0.5pp with NO CIs/seeds** — both decisive DSC margins are ±0.004 (DINO Ours loses by 0.004, CLIP Ours wins by 0.004); within plausible run noise on n=100 samples; no significance test.
3. **VBench "inherent advantage" rationale is unfalsifiable for the VBench loss** — Ours loses SubjCons (0.891 vs 0.934) and BgCons (0.909 vs 0.931); attributed post-hoc to competitors "generating less motion". No motion-quantity covariate reported to confirm the explanation.
4. **Self-attention routing prose-vs-table tension** — routing dismissed as distribution-disrupting (Fig 4 figure-only) yet improves 6/7 T2 metrics vs No A; the "routing fails" claim rests on perceptual artifacts not reflected in the table.
5. **Appendix-D ablations (dynamic context, Temporal Drop) figure-only** — no numeric table (Fig S1); the necessity of two of the four design elements is asserted qualitatively only.
6. **Synthetic-game-data circularity (iter-88 Align4D class)** — training data from a custom game platform; the **100-sample test set is "constructed via our data pipeline"** ⇒ same synthetic distribution; the "novel scenes and subjects unseen during training" novelty is *within-distribution*, not real-world generalization. Authors concede the domain gap in Limitations.
7. **Inference-time closed-model dependency** — the headline "highly controllable" + LLM-orchestrated 3D trajectories depend on **Gemini [37]** at inference (and Qwen2.5-VL-72B + SAM3/SAM/DepthAnything-v2 in the data loop); reproducibility of the control signal hinges on a proprietary model not released.
8. **HyDRA comparison not independent** — HyDRA uses the first 10 s of WorldDirector's output as its reference video (L466–468); its (weaker) numbers are partly conditioned on Ours' own generation.
9. **Heavy compute asymmetric vs "3000-step post-training" framing** — 64 A100 × 72 h for 3 k steps (App. A); the light "3,000 steps" headline understates a 64-GPU / 3-day cost, and the comparison to baselines does not control for their compute.
10. **"Five vs two" ablation completeness** — only the Appearance Condition has a numeric ablation (T2); Location Condition B, Contextual Memory M, and Camera/Plücker injection have no numeric ablation (only App-D figure-only for M/Drop) — the contribution of 3 of the 5 condition channels is unquantified.
11. **Persistent-memory claim scoped to ≤15 s / re-entry within one video** — no evaluation of truly long-horizon (multi-minute, many re-entries) object permanence; "arbitrary-length" (Alg 2) is architecturally claimed, not benchmarked beyond the chunk scale.
12. **DSC via YOLO-detected bboxes** — metric depends on YOLO detection quality on synthetic frames; detection failures/merges on fast or occluded entities could bias DSC in either direction, unreported.

## 7. Citable falsifiable hinge (what survives scrutiny)

The **decoupling paradigm** — an LLM orchestrator producing 3D box + camera trajectories projected to 2D location condition B, plus an explicit Appearance Condition A with Temporal Drop, injected into a causal-chunk autoregressive video model with asymmetric context self-attention — yields a **large, reproducible reconstruction-fidelity gain** (PSNR +22.6% rel over the best baseline, SOTA on all 3 of PSNR/SSIM/LPIPS) and a **clean Appearance-Condition ablation** (all 7 metrics drop without A). The contested part is the *dynamic-memory* claim: the central DSC metric is **split** (CLIP win, DINO loss, both ±0.004, no CIs), so "persistent dynamic object memory / object permanence" is demonstrated qualitatively + on DSC_CLIP, not on the more diagnostic DSC_DINO. Sibling-in-spirit to the controllable-video / world-model lineage (Boximator, GLIGEN, motion-prompting, DragAnything) and to memory-augmented world models (LiveWorld, HyDRA, Infinite-World, WorldMem) — but uniquely **LLM-orchestrated 3D-trajectory decoupling from latent synthesis**. Distinct from passive long-video generators (StreamingT2V, FreeNoise) and from physics-engine-style world models (DIAMOND, Oasis, Genie) that entangle dynamics in generative weights.
