# Writeup — PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation

> How I'd explain this to a friend over a beer, if they asked "what have you been reading?"

> **Languages:** English (this file) · [Srpski](writeup-sr.md)

---

The simple story goes something like this.

You've seen those impressive video generation models — Sora, Wan, Cosmos — pump out stunning footage. Now imagine you're a robotics researcher and you want to use one of these as a world simulator. You feed it an image of a robot arm and a prompt like "pick up the red cup and place it on the shelf." What you get back looks gorgeous — until you notice the gripper phase-shifts through the cup, or the cup floats mid-air for a frame, or the robot pushes the cup but the cup doesn't budge.

Pretty pictures ≠ physical plausibility. And for robotics, that's a dealbreaker.

This paper's observation is almost frustratingly obvious in hindsight: physics errors in robot manipulation videos come in two flavors, and they live in specific places. **Locally**, individual points jump around (trajectory discontinuity). **Globally**, the relationships between objects go wrong (pushed object stays still, grasped object slips away). And all of this happens around the **contact zones** — the gripper tips, the object surfaces, the moving parts. Background pixels don't matter. The ceiling doesn't need physics supervision.

So the fix is: find where the action is, then apply two losses — one for local motion, one for global relations — and only on those regions. That's PhysisForcing. Two training-time losses, zero inference cost.

## How they find where physics matters

First they run a point tracker (CoTracker3) on the video to get dense trajectories — where every pixel moves over time. Then they use a depth estimator (Depth-Anything-V2) on the first frame to figure out what's in the foreground. Combine motion magnitude with foreground proximity, threshold adaptively, and you get a spatiotemporal mask that highlights manipulators, objects, and contact areas. Background is out.

## The two losses

**Pixel-level trajectory loss.** Take an intermediate layer of the DiT (not the first, not the last — the middle, empirically best). Project it through a small MLP. Use the first-frame feature as a query, other frames as keys. Compute similarity maps, softmax, and extract predicted point locations as weighted averages of spatial coordinates. Compare those predictions against the tracker's ground-truth trajectories, masked to physics regions only.

What this does: it forces the DiT's internal features to encode smooth, continuous motion at contact points. No more grippers teleporting or objects popping.

**Semantic-level relational loss.** Run a frozen video understanding encoder (V-JEPA 2) on the same clip. It produces token representations that naturally capture object relationships — the encoder knows a gripper and grasped cup should be tightly coupled because it was trained self-supervised on videos. Now take the DiT features from the same middle layer, project them into the encoder's space, and compare the pairwise cosine similarity matrices. Same mask, same selected tokens. Force the DiT to replicate the encoder's relational structure.

What this does: it ensures that regions that should move together actually do, at a semantic level. Even if individual pixel trajectories are okay, this catches cases where the overall interaction is wrong — like a pushed object staying static while the robot clearly made contact.

## Two things that genuinely surprised me

First — **middle layers are king.** They sweep which DiT block to align (layer 10, 15, 25 out of ~40). Layer 15 wins decisively (85.2 vs 83.9 vs 83.2 on PAI-Bench). The reasoning is clean: early layers carry shallow appearance features, late layers are already specialized for noise prediction and resist being steered. Middle layers have both the semantic structure and the plasticity you need. This feels like a general principle for any alignment-on-intermediate-features approach, not just this paper.

Second — **background dilution is real, not just theoretically expected.** The ablation is: apply the exact same two losses uniformly over all tokens (no mask) vs only on physics-informative tokens. Uniform helps (44.8 → 46.0), but masked is better (44.8 → 47.5). The 1.5 point gap comes entirely from the task-oriented metrics (35.4 → 38.9). Background pixels actively hurt physics learning by diluting the gradient signal. I wouldn't have expected it to be this pronounced — the mask isn't just an efficiency trick, it's a quality knob.

## The numbers that matter

On R-Bench (650 robot manipulation + locomotion prompts), PF-Cosmos (Cosmos3-Nano trained with PhysisForcing) gets 63.8, beating everyone including commercial Wan2.6 (60.7). PF-Wan hits 62.0 on the 14B backbone. Over vanilla fine-tuning, the gains are +4.1 and +7.1 respectively — and over the raw base models, +5.4 and +22.3.

On PAI-Bench's robot domain (174 real-world prompts, judged by Qwen3-VL-235B), PF-Cosmos scores 85.2 overall, again beating commercial Wan2.5 (81.0) and the best robotics-specific model Abot-PhysWorld (84.9).

The zero-shot EZS-Bench result is the one I find most convincing. It's training-independent — 196 unseen robot-task-scene combinations, no overlap with training data. PhysisForcing still improves both backbones (79.0→80.5, 80.3→81.1), suggesting the physics priors actually generalize rather than memorizing training distributions.

## Beyond generation: this helps robots

Here's where the paper goes beyond "we made nicer videos." They plug the PhysisForcing-trained world model into two robotics pipelines:

**WorldArena action planner:** The world model predicts future frames, an inverse dynamics model decodes those into actions, and a robot executes them in simulation. PhysisForcing lifts closed-loop success from 16.0% to 24.0%, beating the best specialized world-model planner WoW (20.5%).

**Fast-WAM downstream policy:** They use the PhysisForcing-trained Wan2.2-TI2V-5B as the video backbone inside a world-action model. Average policy success goes from 68.2% to 72.8% on RoboTwin 2.0 tasks. The biggest gains are on the most contact-rich tasks: placing an empty cup (+21.5%) and pressing a stapler (+11.0%). These are exactly the scenarios where physical plausibility matters most.

The argument is clear: physically aligned video models learn better internal representations for robotics, not just better-looking outputs.

## What I'd question

The training data is 500K clips filtered from RoVid-X's 4M. That's a lot of filtering. How much of the improvement comes from having clean in-domain data vs the physics losses specifically? They partially address this with the vanilla fine-tuning baseline, but a data-controlled experiment (same data, different loss combinations) would be cleaner.

Also, the generation experiments are all image-to-video (text + image conditioning), not action-conditioned. The WorldArena and Fast-WAM experiments use action-conditioned models, but those are downstream evaluations with a different backbone (Wan2.2-TI2V-5B). I'd like to see PhysisForcing applied directly to an action-conditioned generation model and evaluated on generation quality.

Finally, V-JEPA 2 as the semantic teacher is a strong but somewhat arbitrary choice. Is it the relational structure that matters, or is V-JEPA 2 specifically good? A teacher-ablation (swap in DINOv2, InternVideo, etc.) would clarify this.

## References
- Paper: https://arxiv.org/abs/2606.28128
- Project page: https://dagroup-pku.github.io/PhysisForcing.github.io/
- Breakdown: `breakdown.md`
