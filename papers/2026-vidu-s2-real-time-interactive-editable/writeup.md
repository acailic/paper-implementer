# Writeup — Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation

> Your own explanation of the paper, as if teaching it to a peer who hasn't
> read it. This is **not** a summary of the abstract — it's your synthesis
> after reading and implementing.

## The one-paragraph version

Vidu S2 is Shengshu/Tsinghua's second-generation real-time interactive video
system: one model (S2-Avatar) streams a 720p digital character at 25–42 FPS
that you can re-costume, re-locate, and hand new objects *mid-stream*, and a
sibling model (S2-Editing) live-edits an incoming video stream — restyle it,
re-dress the person, swap the character or the background — while preserving
the source's motion exactly. The science worth remembering is not "bigger
DiT" but three interface-level mechanisms: **Self-Replay Forcing** (roll out
on-policy with everything detached, re-noise the whole self-generated
trajectory, replay it in one gradient-enabled causal pass so late chunks'
losses train how early chunks are encoded), **frame-aligned attention**
(each edited frame may attend only to the source frame at the same timestamp,
while the reference image stays visible to all frames — motion pinned by
mask, appearance free by design), and **asymmetric cache noise levels** (the
backbone reads history at high noise for coarse motion, the one-step 720p
Refiner reads it at low noise for detail). On top sits a warp-based stereo
layer that turns either model's output into synchronized left/right views
for VR. It's a proprietary tech report: no weights, no sizes, no ablations —
so I built a CPU miniature of the mechanisms on synthetic signals instead of
the model (details below).

## The problem

Offline video diffusion is a batch job: prompt in, minutes later a finished
clip out. The paper opens with a neat demand argument: if per-user appetite
for interactive visual content (α) is even comparable to appetite for offline
content (β), interactive *generation* demand scales like α·N while offline
generation demand scales like β·N/m (m = views per shared video, easily 100+)
— so the interactive workload is orders of magnitude bigger. S1 (their prior
system) already streamed interactively at 540p, but the reference image froze
at stream start, motion was talking-head-scale, and you couldn't edit an
incoming stream. S2 removes those four ceilings and adds stereo for
headsets.

## The idea

Three mechanisms carry the paper:

1. **Self-Replay Forcing (SRF).** Self-Forcing fixed the train-test gap by
   conditioning training on the model's own outputs, but (their diagnosis)
   the self-history is fed in *clean* and *detached* — no gradient ever flows
   from a later chunk's loss to how an earlier chunk was encoded. SRF: do
   the long autoregressive rollout with everything detached (on-policy data,
   O(1) graph), then re-noise every block of the self-generated trajectory
   to its own noise level (Diffusion Forcing style) and run ONE
   gradient-enabled causal pass over the whole window. Inside that replay,
   chunk i's keys/values are recomputed from earlier chunks in the same
   graph, so ∂L_i/∂θ flows through them. Rollout stays cheap; replay
   recovers cross-block credit assignment.
2. **Frame-aligned attention (editing).** The edited stream must keep the
   source's motion and timing *exactly*, while the new look comes from a
   reference image. They get both from a mask: target frame t may attend to
   source frame s only at the same timestamp; the reference tokens are
   visible to everyone. At streaming time each source frame is consumed with
   the target it produces and dropped — the source cache never grows.
3. **Dynamic references as pure conditioning.** "New reference at any
   moment" is not an architectural event: the reference is just tokens that
   every chunk already attends to, so swapping them mid-stream changes the
   appearance from the next chunk on. A VLM agent layer translates the new
   reference into prompt text with state persistence (once you've picked up
   the mug, later prompts keep "holding the mug" until told otherwise) and
   visually verifies actions completed.

## How it works (the intuition)

For the forcing story, think of the history KV cache as the model's short
term memory of what it just did. Teacher Forcing trains with a perfect
memory it will never have; Diffusion Forcing trains with a blurry memory of
the *ground truth* it also won't have (its own outputs have errors no re-noising
of GT reproduces); Self-Forcing trains with its actual memory but freezes
each memory trace at write time, so it can never learn "encode this chunk in
a way that makes the *next* chunk easier." SRF re-reads the whole remembered
trajectory with fresh eyes and lets the ending correct the remembering.

For editing: alignment-by-attention is the streaming-compatible version of
correspondence estimation. Optical-flow or point-tracking methods must be
computed and cached; a same-timestamp mask is free, exact, and can't drift —
the trade is it can't re-time the video either.

For stereo: depth → disparity → warp both directions is the classic 2D-to-3D
recipe; the streaming twist is that holes only appear at depth
discontinuities, so the fill budget is small and constant per frame — that's
what makes "lightweight processing" compatible with real time.

## What I learned by implementing it

(`implementation/`, CPU-only miniature: 8-dim synthetic chunks = 4 motion
dims with real cross-chunk dynamics + 4 reference-appearance dims; ~55 s to
run all experiments.)

- **The exposure-bias effect is enormous and the on-policy fix is most of
  the story.** Rolling 48 chunks from noise: Teacher-Forcing-trained model
  MSE 1.69, off-policy Diffusion-Forcing 0.42, on-policy arms 0.13–0.16.
  The gap between "noised GT history" and "own noisy history" (3x) is much
  bigger than anything else I measured — bigger than any SRF-vs-SF
  difference.
- **SRF's specific claim is a gradient-flow property, and it is exactly
  checkable.** Loss on the last 6 chunks reaches 6/6 early-chunk encodings
  under replay; under frozen-KV Self-Forcing it reaches 0/6 — exactly zero,
  as the paper's "detached from the computation graph" implies.
- **At toy scale SRF did not beat faithful Self-Forcing on raw MSE** (0.16
  vs 0.13; SRF did win motion error, 0.195 vs 0.204). The paper's SRF
  advantage comes through DMD's distribution matching, which my oracle
  regression cannot represent — the honest conclusion is that the miniature
  confirms the premise (on-policy > off-policy, massively) and verifies the
  mechanism (gradient channel), not the leaderboard claim.
- **The reference-conditioning interface is the feature.** With reference
  tokens visible to every chunk, a mid-stream reference swap is adopted with
  sustained fidelity within 3 chunks while motion error stays flat. With
  chunk-0-only injection (I2V-style), the model *never* adopts the new
  reference — err-to-new ≥ err-to-old forever. "Dynamic references" cost
  nothing architecturally; they're impossible without the all-chunks
  visibility.
- **The frame-aligned mask beats learned attention even in distribution**:
  0.227 vs 0.349 motion MSE, and both hold up at 2x source speed. The mask
  is an inductive bias you don't have to pay capacity for.

## What surprised me / was harder than expected

- **My own Self-Forcing baseline was silently wrong until the probe caught
  it.** I detached the history *values* in attention but forgot the history
  *keys* — gradient still flowed through the score matmul, making "SF"
  secretly stronger than the real thing. The probe (0/6 vs 6/6) is what
  exposed it. Frozen KV means frozen K and V; autograd will find any wire
  you leave connected.
- **How much of "real-time" is the serving stack, not the model.** A third
  of the paper is layer-wise attention choice (Sage/Sparge/SLA per layer),
  per-block W8A8, kernel fusion + CUDA graphs, Ulysses context parallelism,
  and time-sharing GPUs between VAE encoder / backbone / Refiner / decoder
  on one timeline. None of it is learnable from a paper; all of it is why
  720p at 25–42 FPS exists.
- **The paper prints no end-to-end latency number** despite "real-time"
  being the title — only FPS ranges and evaluation conditions (160/320/480
  ms audio chunks, interruption/recovery, 5-minute stability). The latency
  evidence lives in the demo, not the text.
- **Zero ablations** for SRF vs Self-Forcing or aligned vs free attention —
  the two claims my miniature could most directly have wished for numbers
  on. The evaluation *protocol* (identical manifests, published numbers only
  when the exact benchmark/split/metric matches) is meanwhile unusually
  careful.

## Video angle (OpenMontage channel ideas)

- **"The train-test gap that breaks streaming video — and the forcing
  ladder that fixes it"**: animate the four-arm toy (TF collapse → DF-GT
  drift → SF → SRF) with our gradient-flow probe visualized as light
  reaching back through the KV cache only under replay; ties directly into
  our light-forcing real-time video line — same exposure-bias problem, same
  forcing-schedule family, and our audience already knows self-forcing.
- **"Editing live video without breaking the motion"**: frame-aligned
  attention is a one-sentence idea with a killer demo format — restyle a
  webcam stream live, then show the failure gallery from the paper
  (spurious people, ghosting, clothes that ignore hands) next to the
  aligned-vs-free toy chart (0.227 vs 0.349); ends with "the mask IS the
  product feature."
- **"Mid-stream reference swaps: the interface that makes AI video feel
  interactive"**: screen-record our toy swapping references (adoption in 3
  chunks vs never) intercut with Vidu S2's demo (hand the character a mug,
  change clothes, move the background), arguing the conditioning interface —
  not model scale — is what unlocks live interactivity; segue to stereo
  passthrough as the next frontier (holes at depth edges, 0% → 7.5% as
  objects approach).

## References

- Paper: https://arxiv.org/abs/2609.11638
- Project/demo: https://vidu.com/vidu-stream
- Prior system: Vidu S1 (arXiv:2607.03118); Self-Forcing (NeurIPS 2026),
  Diffusion Forcing (NeurIPS 2024), DMD (CVPR 2024)
- My implementation: `implementation/`
- Breakdown: `breakdown.md`
