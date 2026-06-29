# In-Context World Modeling for Robotic Control — Reading Notes

**Authors:** Siyin Wang, Junhao Shi, Senyu Fei, Zhaoyang Fu, Li Ji, Jingjing Gong, Xipeng Qiu (Fudan / Shanghai Innovation Institute / Tongji)
**ArXiv:** 2606.26025v2, June 2026

---

## First Impressions

The framing is clean: a human handed a joystick they don't understand will wiggle it around before doing anything useful. VLA models don't have that wiggle reflex. The paper asks: what if we gave them one?

I like that this is conceptually simple. It's not a new architecture or a fancy objective — it's just "prepend random exploration clips to the context window during training, and at test time, do a quick calibration wiggle." The fact that this works without any extra parameters or task demonstrations is surprising.

The strongest evidence for me is the "false context" ablation: giving the model context from a 180°-offset viewpoint is *worse* than no context at all. That tells me the model isn't just pattern-matching on token patterns — it's genuinely extracting system configuration from the interaction clips.

## Problem in My Own Words

Current VLA models (π0, OpenVLA, RT-2, etc.) learn a policy π(a|o, l) that maps observations and language to actions. The problem: this formulation implicitly assumes the system configuration ψ (camera angle, robot morphology, mounting offsets) is fixed and absorbed into the weights. When you deploy on a different camera angle, the policy has no way to recover the correct observation-action correspondence. It just outputs the wrong actions and fails.

The standard fix is fine-tuning on the new setup, which is expensive and doesn't scale. Multi-view training helps but doesn't solve geometric extrapolation to unseen angles.

## Unclear Concepts / Questions

1. **Why does the probing strategy not matter much?** Table 2 shows all four strategies (random, XY-only, Z-only, R-only) are within a few points of each other. That's both reassuring (you don't need a clever strategy) and puzzling (what exactly is the model learning from these diverse but individually impoverished signals?). The paper says "no single strategy dominates across all viewpoints" but doesn't dig deeper into why.

2. **How does KV caching work with the context?** They mention the interaction context can be pre-computed and cached since ψ is fixed. Makes sense in theory, but what if the robot moves to a new location and the scene changes? Is it truly static enough?

3. **The 135° viewpoint is hard for everyone.** The paper attributes this to occlusion and reduced visible workspace. But is there something structural about that angle relative to the training distribution? 135° is between 120° (trained) and 225° (not trained), so it's in a "gap" — but so are other OOD angles that perform better.

4. **What counts as "self-exploration" in simulation?** They say "in simulation, this step is unnecessary entirely" because you can just replay transitions. That's a real advantage in sim but means the real-robot protocol (actual random movements in ~6 seconds) is the true test. More real-world diversity would strengthen the claims.

5. **Scalability concerns.** 5 context clips means 10 extra images per forward pass. They report 0.185s vs 0.112s latency on RTX 4090. For a control loop that's probably fine, but it's not free. And for very long-horizon tasks, does the context "wash out" as the transformer processes more tokens?

6. **Proposition 1 proof is nice but the assumptions are strong.** A2 (information-preserving transitions) assumes ψ is time-invariant, which is fine for camera angles but might not hold for things like lighting changes or dynamic obstacles. The paper doesn't discuss when ψ might change mid-task.

## Things I Want to Remember

- No extra parameters. The context processing reuses the VLA backbone. Zero overhead in terms of model size.
- The interaction context is task-agnostic. Random movements. No demonstrations needed.
- False context hurts more than no context → genuine system identification, not just sequence modeling benefit.
- In-context training is necessary: a BC model trained without context supervision gets <1% when you try to prepend context at test time.
- t-SNE of Ψ(T) shows tight within-viewpoint clusters and clear between-viewpoint separation.
- Real robot: baseline goes 68% → 17% on novel viewpoints; ICWM maintains much higher performance.
