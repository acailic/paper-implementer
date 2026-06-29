# Writeup — Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots

> How I'd explain this to a friend over a beer, if they asked "what have you been reading?"

> **Languages:** English (this file) · [Srpski](writeup-sr.md)

---

Here's the setup. You have a robot with two arms and grippers. You want to teach it to open microwaves, wipe counters, hang mugs on hooks, unplug chargers — the kind of stuff humans do all day without thinking. The obvious move: record humans doing those things, extract their hand movements, and train the robot on them.

Except it doesn't work well. And this paper explains why, then shows a fix that's annoyingly simple in hindsight.

## The problem with human wrist rotations

The mainstream approach — used by EgoMimic, GR-3, EMMA, and basically everyone in this space — treats the human hand as just another robotic arm. You run a hand pose estimator on ego-centric video, get the wrist position and rotation (6 degrees of freedom), and feed that to the robot policy as if the human was a 7-DoF arm with a gripper.

Two problems. First, wrist rotation estimates from pose predictors are noisy. Roll and pitch especially — the estimator just isn't confident enough. Second, and this is the one that matters: human fingers and parallel grippers don't move the same way. When you hold a door handle with your fingers, your wrist can rotate all over the place while your fingers maintain contact. A parallel gripper doesn't have that luxury — rotation of the gripper directly changes the contact. So a wrist rotation that's perfectly normal for a human hand translates to a wildly distorted, useless pose on a robot gripper.

The paper shows this qualitatively and it's ugly. The robot with 6DoF human actions twists into knots. The paper shows it quantitatively too — 38% progress vs 49% for their approach.

## The fix: just throw away rotation

Their key idea is embarrassingly simple. Forget rotation entirely. Keep only the wrist translation — the direction and distance the hand moves — in the head-camera frame. Both the human and the robot see the world from roughly the same head camera, so relative wrist translation is a shared language. It's physically meaningful, it's robust to pose estimator noise (translations are much more reliable than rotations), and it works the same way regardless of whether you have fingers or a parallel gripper.

They call this the "bridging action": `a3D-wrist`. For a bi-manual setup it's 6 numbers per timestep (3 per arm) — just how far and in what direction each wrist moves.

## The interleaved action tokens

This is the architecture contribution and it's more interesting than it sounds at first.

The model is a π0-style VLA — a vision-language-action model with a pre-trained VLM backbone and a separate action transformer that generates actions via flow matching. The trick is how they handle the fact that different data sources have different available actions:

- In-the-wild human video: you only get wrist translation (no gripper, no end-effector)
- In-lab human data: you get wrist translation + annotated gripper (hand open/close)
- Robot tele-operation: you get everything — translation, 6DoF end-effector, and gripper

They arrange the action tokens in a specific order: `[bridging → 6DoF end-effector → gripper]`. The bridging signal comes first. This isn't random — it means the 6DoF end-effector tokens can attend to the bridging tokens, so the knowledge about *where to move* (learned from human data) flows directly into *how to move the robot arm* (needed for execution). Missing components are masked in attention and excluded from the loss.

## The training pipeline

Three stages, and each one matters:

**Stage I: Pre-train on humans only.** ~600 hours of human manipulation data — a mix of EgoDex clips, outsourced free-form household tasks, and in-lab recordings. Only the bridging action is supervised. The model never sees a robot action. This is pure human motion understanding.

**Stage II: Co-train humans and robots.** They add ~72 hours of robot pick-and-place data (generic, across 100 objects) plus ~3 hours per task of task-specific human demonstrations (someone opening a microwave, wiping, etc.). On robot data, they randomly swap between a3D-wrist and a6D-eef as the prediction target. This is critical — it forces the model to ground the bridging representation into executable robot actions. Without this, the bridging signal just floats in latent space and never connects to actual robot control.

**Stage III: Few-shot robot post-training.** Just 10 robot demonstrations per task. This is where you see the pre-training payoff.

## The results that matter

The headline number: training only on robot pick-and-place data gets you roughly 0% success on all 15 evaluation tasks. Add human bridging actions via co-training: up to 31% success. Add human pre-training on top: 38% success. Add 10 robot demos per task: 55% success. The robot genuinely learns manipulation it never saw in robot data.

The most striking comparison is bridging vs 6DoF human actions (Table 2). Same setup, same data, same model — just different action representation for the human data. Bridging wins by 11 percentage points on progress and 13 on success. And the qualitative difference is dramatic: one produces stable, natural manipulation; the other produces twisted, distorted wrist poses.

Then there's the ablation that really matters (Table 4). Removing the random bridging substitution on robot data during co-training crashes success from 38% to 12.5%. The model absolutely needs to be explicitly forced to connect the shared bridging representation to executable robot actions. It won't do it on its own.

And the upper bound experiment (Table 5) is quietly one of the most interesting results. They take real robot demonstrations, strip them of rotation and observation advantages, and train with the same objective as human data. Performance jumps to 73.5% progress and 55.8% success — well above the default 59.8%/38.3%. The bridging representation has real headroom. The bottleneck isn't the representation, it's the gap between human and robot embodiments.

## What I found most interesting

The loss alignment result (Figure 9) is a quiet gem. Pre-training only on the bridging signal — a non-executable, translation-only representation — yields *lower* training loss for both the 6DoF end-effector action and the gripper action during co-training. The model that learned "where hands move" in human videos converges faster on "how to move robot arms." The objective landscapes are aligned, even though the pre-training objective is completely non-executable. This is strong evidence that the bridging representation captures something fundamental about manipulation that transcends embodiment.

The failure cases are honest and informative. Tasks needing precise end-effector rotation at contact — inserting a straw into a cup, opening a drawer — are where the approach breaks. The robot shows clear task intent, reaches the right area, but can't execute the critical rotational step. This is exactly the trade-off you'd expect from deliberately discarding rotation. The authors are upfront about it and point to adding limited, reliable rotation cues as future work.

## What would I build from this?

The most immediately useful idea is the random prediction target substitution during co-training. The idea that you can force a model to ground a shared representation into a specific action space by randomly swapping prediction targets is generalizable well beyond this paper. If you have any multi-embodiment setting — different robots, different grippers, different action spaces — you could apply the same trick: train on the shared representation, but randomly swap in the embodiment-specific action as the target to force the binding.

The bridging representation itself is platform-agnostic. It should work on single-arm setups, mobile manipulators, even humanoids — anything with a head camera and wrists. The interleaved token design with attention masking is a clean solution to the heterogeneous action problem that other papers handle with uglier padding and concatenation.

## References
- Paper: https://arxiv.org/abs/2606.28133
- Project page: https://translation-as-a-bridging-action.github.io/
- Breakdown: `breakdown.md`
