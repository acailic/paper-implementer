# DanceOPD: On-Policy Generative Field Distillation — Writeup

You've trained a great text-to-image model. Now you want it to edit images too. You fine-tune, but T2I quality drops. You try merging two separately-trained models, and you get a mediocre jack-of-all-trades. You mix data from both tasks during training, and the gradients fight each other. This is the capability interference problem, and it's one of the central headaches in deploying multi-capability image generation models.

DanceOPD from ByteDance Seed and NUS offers a clean answer: stop trying to force everything into one objective, and start thinking in terms of velocity fields.

## The Field View

Flow-matching models generate images by learning a velocity field — a function that tells you which direction to push a noisy latent at each point along the denoising trajectory. The authors observe that different capabilities (T2I, editing, style transfer) are really just different velocity fields defined over the same latent space. A T2I model has its own velocity field, an editing model has another, and so on.

This reframing turns the multi-capability composition problem into something concrete: which field should a student model imitate, where should it be queried, and how many supervision points do you need per training sample?

## Three Problems, Three Solutions

**Problem 1: Target-field ambiguity.** If you average the T2I velocity and the edit velocity to create a single supervision target, you get a direction that corresponds to neither capability. It's a Frankenstein target. DanceOPD fixes this with hard routing — each sample goes to exactly one teacher field. A T2I sample queries the T2I field, an edit sample queries the edit field. The composition happens statistically across updates, not by muddying individual targets.

**Problem 2: State-distribution mismatch.** If you evaluate the teacher field on fixed data states or the teacher's own trajectory, you're supervising the student on states it may never visit during inference. DanceOPD rolls out the current student model and queries the teacher on the student's own states. This is the "on-policy" part — the teacher supervises where the student actually is, not where we wish it were.

**Problem 3: Trajectory-query correlation.** A natural instinct is to query many states along the same rollout for denser supervision. But those states are highly correlated — same noise seed, same prompt, same trajectory history. Adding more doesn't give you independent information. DanceOPD uses just one query per sample, placed on the low-noise (semantic) side of the trajectory where capability-specific signals are concentrated.

The loss is dead simple: `||v_student - v_teacher||²` — plain velocity MSE on the routed, on-policy query.

## What Actually Happens in Practice

The experiments cover four settings, all on the Z-Image backbone (SD3.5-M for realism absorption):

1. **T2I + Edit composition** — The student improves editing (GEditBench +8.1% vs. best OPD baseline) while actually nudging T2I quality slightly above the original T2I source. Background change and style change see the biggest gains (21.9% and 21.3% over DiffusionOPD).

2. **Local + Global Edit composition** — Local editing preserves, global editing transforms. These pull in opposite directions. DanceOPD improves GEditBench by 16.1% over the best competing composition, with particularly strong gains on background change (33.5%).

3. **Realism-field absorption** — A photorealism-oriented teacher field gets absorbed into the student, closing 85.3% of the student-to-teacher gap in realism reward while keeping T2I quality within 0.1% of off-policy distillation.

4. **CFG absorption** — Classifier-free guidance, normally an inference-time operation, gets baked into the model as an operator-defined velocity field. Training-time and inference-time CFG compose multiplicatively (effective guidance ≈ αβ), so you need to be careful not to over-guide.

## Why the Ablations Matter

The diagnostic studies are unusually thorough and each design choice is well-justified:

- **Hard routing vs. soft mixing:** +15.2% under MSE. The issue is target construction, not the loss function. This holds even when you switch to KL weighting (+10.6%).
- **Low-t vs. other timesteps:** +23.7% over median-t, +19.5% over high-t. Capability-specific information really is concentrated in low-noise states.
- **Single vs. dense queries:** K=1 beats K=2,4,8,16 by 7.9–16.6%. Correlated trajectory states are not free supervision.
- **SDE decorrelation test:** Adding stochastic noise to the rollout recovers 18.4% of the dense-query degradation but stays 8.6% below the single-query default. Confirms correlation is the failure mode without being a practical alternative.
- **Plain MSE vs. alternatives:** Timestep-weighted, KL-weighted, DMD-EMA, consistency, feature distillation — plain MSE beats them all. When your target is a deterministic velocity, direct regression is the right tool.
- **Initialization:** Local-edit init beats merged init by 37.2%. The initial rollout quality matters because that's where teacher fields get queried early in training.

## Cost Efficiency

DanceOPD needs a 16-step rollout per training step (same as any on-policy method) but only evaluates one gradient-bearing state (K=1). DiffusionOPD evaluates all 16 states (K=N=16), and Flow-OPD adds PPO overhead and a 2× micro-batch factor. So DanceOPD is faster than both while delivering better results.

## Limitations to Keep in Mind

The method needs all teacher fields to operate on the same latent space with compatible velocity parameterization — you can't trivially compose models from different architectures. The routing is predefined by data identity, so it struggles if a prompt genuinely needs multiple capabilities at once (e.g., "make it photorealistic AND change the background"). The realism evaluation uses a proprietary reward model, which limits reproducibility of that particular experiment.

## Verdict

DanceOPD is a well-motivated, carefully validated, and practically useful contribution. The field-based framing is a natural fit for flow-matching models, and the three design choices (hard routing, on-policy querying, single semantic-side query) each address a real problem with clear ablation evidence. The simplicity of the final objective (plain velocity MSE) is a strength, not a weakness — it suggests the authors identified the right problem rather than over-engineering the solution. For anyone building multi-capability image generation systems, this is worth paying attention to.
