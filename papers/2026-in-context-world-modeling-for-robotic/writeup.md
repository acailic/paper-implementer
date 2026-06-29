# In-Context World Modeling for Robotic Control — Writeup

> 🇧🇦 Bosnian/Serbian version: [writeup-sr.md](writeup-sr.md)

**ArXiv:** 2606.26025v2 · June 2026
**Authors:** Siyin Wang, Junhao Shi, Senyu Fei, Zhaoyang Fu, Li Ji, Jingjing Gong, Xipeng Qiu

---

## One-Paragraph Version

VLA models (like π0, OpenVLA, RT-2) break when you change the camera angle because they've baked the system configuration into their weights during training. This paper's fix is dead simple: before doing the task, wiggle the robot around randomly for a few seconds, record what happens, and feed those clips as context to the model. During training, you prepend similar random interaction clips to every training sample, teaching the model to use them as a calibration signal. At test time, the model implicitly figures out the camera angle and action mapping from these context clips — no fine-tuning, no demonstrations, no extra parameters. It works: +13% on unseen viewpoints in simulation and massive gains on a real UR5e robot.

---

## The Problem

Imagine you're controlling a robot with a joystick, but you've never used this particular setup before. Pushing forward — does the robot go left, right, or forward? You don't attempt the task immediately. You wiggle the joystick. You watch. Within seconds, you've built an internal model of how inputs map to outputs. Now you can execute the task confidently.

VLA models can't do this. They learn π(a | o, l) — action given observation and language. The camera angle, robot morphology, mounting offsets — all of that is implicitly fixed in the training data and absorbed into the weights. Deploy on a different camera? The model has no mechanism to figure out the new observation-action correspondence. It just outputs garbage actions.

The standard fix is fine-tuning per new setup. That's expensive, requires human intervention, and doesn't scale to the "generalist robot" vision everyone's chasing.

## The Idea

The paper reframes this as a system identification problem. The policy needs to know ψ (the system configuration) at test time. Instead of fine-tuning, recover ψ from a short history of interactions.

The trick: repurpose the transformer's context window. Most ICL work in robotics uses the context for behavior specification — "here's a demo, copy it." ICWM uses it for system identification — "here are some random movements, figure out how the system works."

At training time, each sample gets N=5 random interaction clips prepended as context. These clips come from diverse configurations in the training data. The model is forced to learn to extract configuration info from context to predict actions correctly.

At test time, the robot does ~20 random probing movements (takes about 5-6 seconds), you grab 5 clips from those, and prepend them. Single forward pass. No gradient updates. No task demos needed.

## How It Works (Intuition)

Think of it like this: a single photo of a robot workspace doesn't tell you the camera angle — the same scene looks similar from many angles. But a *video* of the robot moving, showing how objects shift in the image as the end-effector moves, reveals the camera geometry. The relationship between "robot moved left" and "objects shifted right in the image" encodes the viewpoint.

The model sees 5 such interaction clips before the task. Each clip is a triplet: start image → action → end image. From these, it implicitly reconstructs the camera-to-robot mapping. Then when it sees the task image, it can correctly interpret spatial relationships and output accurate actions.

The formal argument (Proposition 1) is elegant: under mild assumptions, a sequence of (observation, action) pairs carries strictly more information about ψ than any single observation. This holds for *any* action distribution — so random movements work fine. You don't need task-relevant exploration.

The whole thing adds **zero extra parameters**. The interaction context is processed by the same Qwen2.5-VL backbone. The only change is the training data format and a 5-6 second calibration wiggle at deployment.

## What Surprised Me

**False context is worse than no context.** When they feed the model context clips from a 180°-offset viewpoint, performance drops below the no-context baseline. The model isn't just benefitting from longer sequences or extra tokens — it's genuinely extracting configuration info, and wrong configuration info actively misleads it. The symmetric magnitude (gains from correct context ≈ losses from false context) is a clean demonstration.

**The probing strategy barely matters.** Random, XY-only, Z-only, rotation-only — they all improve over baseline by similar margins. You don't need a clever exploration strategy. Any spatially diverse movements expose enough of the dynamics manifold. This is great news for real deployments: just wiggle and go.

**In-context training is essential.** Take a standard behavior-cloning model (trained without context) and try to prepend interaction clips at test time. Performance collapses to <1%. The capability doesn't emerge from sequence modeling alone — you have to explicitly train the model to use context for configuration inference.

**The biggest gains are on long-horizon tasks.** On LIBERO-Long, ICWM beats the multi-view baseline by 26.3% on unseen viewpoints. The paper explains: long tasks amplify small spatial errors from viewpoint shifts, causing cascading failures. ICWM's calibration prevents the initial error accumulation. Makes intuitive sense — a small depth error on a single pick is recoverable; on a 10-step manipulation sequence, it snowballs.

**No code released.** For a paper with such a clean idea, that's a bummer. Would love to see this reproduced.

---

## References

- Paper: [arXiv 2606.26025](https://arxiv.org/abs/2606.26025)
- LIBERO Benchmark: [Liu et al., NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/8c3c668620ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html)
- Qwen2.5-VL: [Bai et al., 2025](https://arxiv.org/abs/2502.13923)
- FAST Action Tokenizer: [Pertsch et al., 2025](https://arxiv.org/abs/2501.09747)
- π0: [Black et al., 2024](https://arxiv.org/abs/2410.24164)
- OpenVLA: [Kim et al., 2024](https://arxiv.org/abs/2406.09246)
- RT-2: [Zitkovich et al., CoRL 2023](https://proceedings.mlr.press/v229/zitkovich23a.html)
- ICRT (in-context imitation learning): [Fu et al., ICRA 2025](https://arxiv.org/abs/2406.09246)
