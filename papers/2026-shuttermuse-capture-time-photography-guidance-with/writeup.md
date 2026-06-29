# Writeup — ShutterMuse: Capture-Time Photography Guidance with MLLMs

> How I'd explain this to a friend over a beer, if they asked "what have you been reading?"

> **Languages:** English (this file) · [Srpski](writeup-sr.md)

---

Imagine you're taking a photo of a castle. You frame the shot, but something's off. The horizon is tilted, the castle is tiny in the corner, and there's a trash can in the foreground. What do you do? Crop? Keep it? Throw it away entirely?

Now imagine you're the *subject* — your friend is photographing you on some stairs and has no idea how to pose you. Where do you put your hands? How do you angle your body?

Every existing AI photography tool assumes option one: "here's your photo, let me find a better crop." But that's wrong a lot of the time. Sometimes the frame is already great. Sometimes the photo is unsalvageable. And nobody was building tools that tell the *person in the photo* how to stand.

This paper fixes all three problems at once.

## The three-way decision

The central insight sounds almost trivial in hindsight: **not every photo needs cropping.** The authors define three decisions:

- **Refine** — the shot has potential, but the framing needs adjustment (crop/recompose)
- **Keep** — the framing is already solid, leave it alone
- **Reject** — the photo is beyond help (blur, no subject, extreme tilt)

Every prior benchmark assumed every image has a preferable crop. Specialized cropping models like Venus and InstructCrop are literally incapable of saying "don't crop this" — they always output a bounding box, even when it makes things worse. On the benchmark, these models score **zero** on both reject and keep success rates. That's not a minor flaw — it's a fundamental mismatch between what the tools are built for and what photographers actually need.

## What they built

Three artifacts, layered on top of each other:

**CaptureGuide-Bench** — a benchmark with 421 photographer-side samples (three-way decisions with 3–5 expert bounding boxes per refine) and 552 subject-side samples (scene-conditioned pose recommendations). Evaluated with both geometric metrics (IoU, BDE) and an MLLM judge that scores compositional quality.

**CaptureGuide-Dataset** — ~130K training samples. The photographer side is built through an expert-seeded, MLLM-verified self-distillation pipeline (EMDP): 10 annotators create 12K seed samples, an MLLM normalizes them, a model pseudo-labels 500K unlabeled images, and a verifier (Gemini-3.0-Pro) checks quality before accepting. Three rounds expand the dataset to 100K. The subject side uses a clever trick: start with portrait photos, remove the person with an image editing model, extract the pose keypoints with YOLO, and generate scene-conditioned rationales with another MLLM. 30K samples.

**ShutterMuse** — a unified MLLM on Qwen3-VL-8B that handles both tasks. Stage 1 is supervised fine-tuning on structured JSON outputs. Stage 2 is GRPO (reinforcement learning) with three reward components: correct decision, subject preservation in the crop box, and visibility consistency for poses.

## Two things that genuinely surprised me

First — **the keep/reject gap is dramatic, and nobody noticed.** Specialized cropping models that state-of-the-art papers celebrate (Venus, InstructCrop) literally score 0% on reject success rate and near-0% on keep success rate. That means if you feed them a perfectly composed photo, they'll crop it anyway. If you feed them an unusably blurry mess, they'll still try to "improve" it. General MLLMs like GPT-5.5 and Gemini are better at the decision (48-51% refinement rate) but their crop boxes are way off (IoU ~65% vs ShutterMuse's 74%). Nobody was good at both sides before this paper.

Second — **a 8B model can rival GPT-Image-2 for pose guidance at 20× the speed.** On subject-side pose recommendation, Nano-Banana-Pro leads with mean score 0.39, GPT-Image-2 gets 0.35, and ShutterMuse gets 0.34. The quality gap is small. But ShutterMuse does it in 5 seconds with 412 tokens, versus 55-103 seconds and 1300-1400 tokens for the foundation models. For a real-time "stand like this" assistant embedded in a camera app, that's the difference between usable and not.

## What the ablation tells us

The GRPO stage is the real difference-maker. SFT alone gets you to IoU 72.39% and RSR 68.97%. Add GRPO with the full reward and you jump to IoU 74.30%, RSR 82.76%, KSR 74.55%. The decision reward (`Rdec`) is the most important single component — removing it drops RSR from 83% to 62% and KSR from 75% to 65%. That makes sense: the three-way decision is the novel part, and the RL reward directly trains it.

The mask preservation reward (`Rmask`) is clever. It uses BiRefNet to detect the main subject, then checks whether the predicted crop box covers ≥90% of the subject mask. This prevents the model from learning to "cheat" by cropping to empty space. Without it, the MLLM-Score drops, confirming it genuinely helps compositional quality.

## What was harder than expected

The dataset pipeline is the real engineering feat. The self-distillation loop — train model → pseudo-label → MLLM verify → retrain — could easily spiral into garbage-in-garbage-out. They control this with a fixed expert validation set that monitors quality at each round, an independent expert test set that's never touched by the pipeline, and a verifier that maintains >87% F1 across all categories. The data scales from 12K to 100K without quality degradation, which is impressive for a pipeline with this many moving parts.

The subject-side construction is particularly creative. Starting from existing portrait photos, removing the person, and using the extracted pose as the "answer" for the now-empty scene — that's a smart way to bootstrap a dataset for a problem that would otherwise require staging thousands of photoshoots with pose annotations.

## The user study is worth noting

MLLM-Score and human preference rankings align with SRCC = 0.90 on the photographer side and are literally identical on the subject side. That's strong evidence the evaluation protocol is meaningful and not just an artifact of the judge model.

## What's missing

The subject-side quality gap (0.34 vs 0.39) is real. The 17-keypoint COCO format can't represent foot contact — Appendix D shows floating feet in skeleton visualizations, acknowledged as a limitation. The model only handles single frames, not video. And the training data likely skews toward conventional composition norms — it would be interesting to see how this generalizes across different photographic traditions.

## References

- Paper: https://arxiv.org/abs/2606.25763
- Official code: https://github.com/lijayuTnT/ShutterMuse
- Project page: https://lijayutnt.github.io/ShutterMuse/
- Breakdown: `breakdown.md`
