# Writeup — DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation

> How I'd explain this to a friend over a beer, if they asked "what have you been reading?"

> **Languages:** English (this file) · [Srpski](writeup-sr.md)

---

The simple story goes something like this.

Say you have a photo of your dog and you want an AI to generate a video of that dog
running through a field. That's in-domain — keep the dog looking exactly the same,
just animate it. Existing subject-driven video methods are pretty good at this.

Now say you want a video of your dog as a watercolor painting. Or as a 3D claymation
character. Or your dog printed on the side of a yellow school bus. That's
cross-domain — you need to keep the dog recognizable (ears, coloring, shape) while
completely changing the style, lighting, and medium around it. Almost nobody has
been working on this, and the methods that exist either just copy-paste the
reference photo into the video or lose the subject entirely when they try to
transform it.

DomainShuttle says: let's do both, and let's do them with the same model.

## The architecture in plain language

The whole trick is **decoupling**. In a standard video diffusion transformer,
reference image tokens and video tokens are mixed together in the same attention
layers, processed with the same projections, and positioned in the same RoPE
(Rotary Positional Encoding) space. DomainShuttle tears this apart into three
separate moves.

**Move 1: Domain-MoT (independent attention branches).** Give video tokens and
reference tokens their own Q/K/V projection matrices instead of sharing them. The
video branch keeps doing what the base model already knows how to do — generate
good video. The reference branch specializes in extracting subject features. On top
of that, the reference branch gets an extra conditioning signal: a *domain
attribute* (real-world human, real-world object, fantasy character, background)
that tells it what kind of subject this is. The video branch only sees time; the
reference branch sees time *plus domain*. This means at inference you can swap the
domain attribute to change styles without touching anything else.

**Move 2: VR-DualRoPE (separate positional spaces).** Normally reference images are
treated as extra video frames in the RoPE space — they get a temporal index like
any other frame. That's wrong for two reasons: different reference subjects have
no temporal relationship with each other, and multiple photos of the *same* subject
should be associated, not spread across time. VR-DualRoPE puts all reference
tokens in a completely separate RoPE space with temporal index fixed at zero, and
uses spatial offsets to separate different subjects while keeping same-subject
images close together.

**Move 3: CCL (cross-pair consistency).** For each training video, they have
multiple sets of reference images (different angles, lighting, crops). During
training they feed two different reference sets through the model at the *same
noise level* and force the predictions to match. One branch is frozen, one is
trainable. This teaches the model: "these two different photos show the same
subject — learn what's shared, ignore what's different." The result is that the
model learns intrinsic subject features (identity, shape, color) rather than
overfitting to the specific artifacts of a single reference image.

## The numbers that matter

The headline result: **CD-Score of 0.861**, which is 18.7% better than Kling 1.6
(0.725) and 54.5% better than the next-best open method (FFGO at 0.558). CD-Score
measures how well intrinsic subject features survive a domain transformation — it's
exactly the metric this paper targets.

The in-domain metrics (DINO-I, CLIP-I) are competitive but not always the absolute
best. That's fine — it's a deliberate trade-off. The model gives up maybe 1-2% of
in-domain fidelity for a massive gain in cross-domain capability. For anyone who
actually wants to do creative work with video generation, that's a great deal.

The ablation tells a clean story:

```
CD-Score:  0.697 → 0.715 → 0.783 → 0.813 → 0.861
           naive   +dual   +MoT    +RoPE   +CCL
```

Domain-MoT is the single biggest contributor. Adding the domain-aware AdaLN to the
reference branch is what unlocks cross-domain transformation at all — without it,
the naive method just fails to transfer subjects into the target domain. The other
two modules add further refinements.

## Two things I found interesting

**CCL is about controllability, not fidelity.** This was surprising. I expected
CCL to improve both cross-domain and in-domain metrics equally. Instead it bumps
CD-Score by 5.9 percentage points but barely touches CLIP-I (+0.3%) and DINO-I
(+1.5%). That means CCL specifically teaches the model *what to transform and
what to preserve* — it's not just making the model better at recognizing subjects,
it's making it better at separating intrinsic features from domain-specific ones.
That's a more nuanced capability than I expected from what's essentially a
consistency regularization loss.

**VR-DualRoPE slightly hurts CLIP-I.** This seems counterintuitive — better spatial
modeling should help similarity, right? But the subject-decoupled offset strategy
pulls same-subject reference images closer in RoPE space, which means the model
treats them as a cluster rather than individual high-fidelity copies. CLIP-I measures
frame-level similarity to a single reference image, so the clustering behavior can
actually reduce the score. The subject-level metrics (CD-Score, DINO-I) improve,
though. It's a reminder that optimizing for the right metric matters — frame-level
similarity is not the same as subject-level consistency.

## The training recipe

Two-stage. First, 2,000 steps on 200K image personalization data to give the base
model basic subject-awareness. Then 12,000 steps on 750K video personalization data
for the real training. Cross-attention is frozen throughout stage 2 to preserve
text-following ability. Total: 30,000 GPU-hours on a 14B model. Expensive, but the
official code is Apache 2.0 and includes training scripts, which makes reproduction
far more feasible than most video generation papers.

The data pipeline is the hidden workhorse. Building cross-pair reference sets
requires Grounding-DINO for detection, SAM2 for segmentation, and an MLLM for
quality filtering. The Ditto-1M dataset also provides editing pairs (reference →
edited video) as augmentation. Without Ditto-1M, the model still works (CD-Score
0.823 vs 0.861) and still beats all baselines — so the core method is robust,
the editing data is just a bonus.

## What was harder than I expected to understand

The domain attribute annotation is subtle. The paper says domain attributes refer
to "the subject attributes in the *generated* video, not in the reference images."
So if you have a photo of a real person and you want to generate them as a fantasy
character, the domain attribute is "fantasy subject" — not "real-world human."
The attribute describes where you're going, not where you started. That took me a
re-read to catch.

Also, the CCL mechanism uses a *frozen* branch, not a momentum encoder or EMA.
The frozen branch `G*_θ` never updates — it's a snapshot. Only the trainable
branch `G_θ` learns. This is simpler than the typical consistency regularization
setups (BYOL, SimSiam) and avoids the representational collapse problem by
construction — the frozen target provides a stable learning signal. Clean design.

## References
- Paper: https://arxiv.org/abs/2606.26058
- Official code: https://github.com/HKUST-C4G/DomainShuttle
- Project page: https://cn-makers.github.io/DomainShuttle/
- Breakdown: `breakdown.md`
