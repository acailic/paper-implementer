# Notes — ShutterMuse: Capture-Time Photography Guidance with MLLMs

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's a **benchmark + dataset + model** paper. The authors do four things:

| # | What | Output |
|---|------|--------|
| 1 | Propose **CaptureGuide-Bench** | Benchmark for capture-time photography guidance (photographer + subject side) |
| 2 | Build **CaptureGuide-Dataset** | ~130K samples with rationales, boxes, keypoints |
| 3 | Develop **ShutterMuse** | Unified MLLM (Qwen3-VL-8B backbone) for both tasks |
| 4 | Evaluate against 20+ baselines | Shows best photographer-side, competitive subject-side at ~20× lower cost |

## The big picture

Existing aesthetic cropping benchmarks treat photography as a post-hoc crop prediction problem — "here's a photo, find a better crop." But real photography is messier: sometimes the shot is already good (keep), sometimes it's unsalvageable (reject), and sometimes the *subject* needs guidance on how to pose. Nobody had benchmarked this capture-time, two-sided problem properly.

## The three-way decision (photographer side)

The key conceptual shift: **not every image should be cropped.**

| Decision | Meaning | Annotation |
|----------|---------|------------|
| **Refine** | Cropping/recomposition will improve the shot | Bounding box + rationale |
| **Keep** | Current framing is already strong | Strengths explanation |
| **Reject** | Image is unsalvageable (blur, no subject, tilt, etc.) | Defects explanation |

This is fundamentally different from every prior cropping benchmark that assumes all images have a preferable crop.

## The dataset construction pipeline

### Photographer side — EMDP (Expert-seeded, MLLM-verified Self-Distillation Pipeline)

1. 10 trained annotators create 12K seed set (refine/keep/reject with boxes + comments)
2. MLLM (Gemini-3.0-Pro) normalizes raw comments into structured rationales
3. Train initial model → generate pseudo-annotations on 500K unlabeled pool
4. MLLM verifier checks rationale correctness + rationale-box consistency
5. Verified samples → iterative retraining (3 rounds)
6. Fixed expert validation set monitors quality; expert test set for reliability
7. Final: 100K photographer-side samples

### Subject side — SGGP (Subject-side Guidance Generation Pipeline)

1. Start from portrait images
2. Remove person with Nano-Banana-Pro → person-free scene
3. Extract COCO-17 keypoints with YOLO26x-Pose
4. Gemini-3.0-Pro writes pose-recommendation rationale
5. Human experts review + revise rationales + correct keypoints
6. Final: 30K person-free scene + keypoints + visibility + rationales

## The model — ShutterMuse

Built on Qwen3-VL-8B. Two training stages:

**Stage 1 — SFT:** Learns to output JSON with:
- Photographer side: `task_type=composition`, `reason`, `composition_xy` (empty=reject, [0,0,1,1]=keep, [x1,y1,x2,y2]=refine)
- Subject side: `task_type=pose`, `reason`, `keypoints_xyn` (17 COCO coords), `visibility` (1/0/-1)

**Stage 2 — RFT (GRPO):** Reinforcement with task-specific rewards:
- `Rdec` — correct three-way decision (binary)
- `Rmask` — refined box preserves main subject (BiRefNet salient-object mask, coverage ≥ 0.9)
- `Rsub` — visibility vector exact match (binary)

Subject-side reward is simple: exact visibility match = 1, else 0. Photographer-side gets both decision + mask preservation.

## The benchmark — CaptureGuide-Bench

- Photographer: 421 held-out samples (3–5 ground-truth boxes per refine)
- Subject: 552 samples, balanced pose + scene types
- All held out from training

**Metrics — photographer side:** IoU, BDE, R (refinement success, IoU > 0.7), RSR, KSR, MLLM-Score
**Metrics — subject side:** Three MLLM-judged dimensions (physical plausibility, scene interaction, pose aesthetics), each 0/0.5/1.0

## Key headline results

### Photographer side (Table 1)

| Method | IoU% | R% | RSR% | KSR% | MLLM-Score |
|--------|-----:|---:|-----:|-----:|-----------:|
| **ShutterMuse** | **74.30** | **70.03** | **82.76** | **74.55** | **0.64** |
| Venus | 69.43 | 57.27 | 0.00 | 3.64 | 0.57 |
| InstructCrop | 69.53 | 56.97 | 0.00 | 0.00 | 0.43 |
| Gemini-3.1-Pro | 65.63 | 51.34 | 13.79 | 20.69 | — |
| GPT-5.5 | 65.44 | 41.84 | 48.28 | 10.34 | — |

Clear pattern: specialized cropping models (Venus, InstructCrop) have good IoU but **zero** keep/reject ability. General MLLMs make OK decisions but bad crops. ShutterMuse is the only one good at both.

### Subject side (Table 2)

| Method | Plausibility | Interaction | Aesthetics | Mean | Time (s) |
|--------|-------------:|------------:|-----------:|-----:|---------:|
| Nano-Banana-Pro | **0.63** | **0.59** | **0.58** | **0.39** | 55.16 |
| GPT-Image-2 | 0.53 | 0.54 | 0.53 | 0.35 | 102.61 |
| **ShutterMuse** | 0.52 | 0.54 | 0.53 | 0.34 | **4.96** |

Foundation models slightly ahead on quality but ShutterMuse is **10-20× faster** and uses 3× fewer tokens.

## Ablation insights

- **GRPO helps a lot** — IoU 72.39→74.30, RSR 68.97→82.76, KSR 63.64→74.55
- **Rdec is critical** — removing it tanks RSR (82.76→62.07) and KSR (74.55→65.45)
- **Rmask matters for localization** — removing it drops IoU and MLLM-Score
- **Rsub improves plausibility consistency** — without it, visibility match degrades
- EMDP reliability: verifier F1 > 87% across all categories and rounds, acceptance > 52%

## User study

- SRCC = 0.90 between MLLM-Score and human rankings (photographer side)
- Subject side: MLLM ranking **identical** to human ranking
- 6 participants, 100 samples per subset, blind evaluation

## Terms / concepts I had to look up

| Term | Meaning |
|------|---------|
| **GRPO** | Group Relative Policy Optimization — RL method that uses group-relative advantage without a value model |
| **COCO-17** | Standard 17-keypoint body format: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles |
| **BDE** | Minimum Boundary Displacement Error — average distance between predicted and GT box boundaries |
| **BiRefNet** | Salient object detection model used to extract subject masks for the mask-preservation reward |
| **vLLM** | Inference acceleration engine for LLM serving (PagedAttention) |

## What's re-implementable

The core model pipeline is clear: Qwen3-VL-8B → SFT on structured JSON → GRPO with the three reward components. The dataset construction pipelines (EMDP + SGGP) are the heavy lift. The evaluation protocol (MLLM-judged three-way scoring) is fully specified with prompts in the appendix.
