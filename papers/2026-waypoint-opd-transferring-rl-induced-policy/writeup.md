# Writeup — Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation

> Paper: "Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated
> On-Policy Distillation" — Hou et al. (Alibaba Cloud), arXiv:2610.21601.
> This is my synthesis after reading and implementing it, not a summary of
> the abstract.

## The one-paragraph version

You want to reuse a teacher's RL gains without re-running RL: distill the
*change* the RL caused (log π_after − log π_before, per token) into a student
on the student's own rollouts. But that endpoint difference is a lossy
compression of the whole training run: the same endpoint can be the sum of
two shifts that agreed (a stable, twice-confirmed preference change) or two
shifts that fought (overshoot, then correction — leaving a small residual
dominated by whatever happened to win). Waypoint-OPD keeps one intermediate
checkpoint from the teacher's RL run, splits the shift into its early and
late segments on the student's own top-k candidate tokens, centers both by
the student's candidate weights, and gates the endpoint by
max(0, weighted cosine of the two segments). Aligned histories pass; opposed
histories are zeroed. One extra frozen-teacher scoring pass, no verifiers, no
new parameters — and in the paper, the highest domain-average in all six
domain×student settings plus *faster* wall-clock training than endpoint-only
transfer (gated students ramble less).

## The problem

RL on verifiable rewards (GRPO on math/code) is the expensive part of modern
LLM training. Distilling the *result* (standard on-policy distillation,
reverse-KL to the final teacher) caps the student at the teacher and inherits
its quirks. Direct-OPD (Feng et al. 2026) had the better idea — transfer the
*delta* the RL produced, as a dense implicit reward at student-visited
prefixes — but treats the delta as if it were a single coherent instruction.
It is not: it is the integral of a noisy trajectory. The paper's observation
is that the delta's *provenance* is invisible in the delta itself, and
provenance plausibly predicts reliability. That is a diagnostic insight
first, a training method second.

## The idea

Ask the teacher's own training history whether the endpoint at this prefix,
for these candidate tokens, means what it says. If the teacher's relative
preferences moved the same way in the first half and the second half of RL,
the endpoint is their sum — two independent confirmations. If they moved
opposite ways, the endpoint is their small residue, and rescaling it (as
implicit reward magnitude) amplifies what is essentially the winner of a
fight between two large forces. So: keep the Waypoint checkpoint, measure the
two segment shifts on the student's candidate support, center them (so a
uniform shift across candidates — which changes no relative preference —
cannot fake agreement), take their student-weighted cosine, and let
max(0, cosine) scale the endpoint. Batch-normalize so the mean rescaled gate
stays at τ = 0.8 across the rollout batch. Everything stays detached; the
update is a candidate-wise dual-clipped PPO step plus a KL leash to the
student's init.

Two details I initially underweighted: (1) the gate is *graded*, not binary —
partial alignment passes partial signal; (2) the zero-sum advantage
(Σ_a A_t(a) = 0 over the support, a direct consequence of centering) means
the distillation term can only redistribute probability among the k
candidates the student already considers — a very local, very cheap update.

## How it works (the intuition)

Picture the teacher's RL as a walk through policy space. Endpoint transfer
gives you the displacement vector and says "walk this way". Waypoint-OPD asks
whether the walk was straight. Straight walk → the displacement is a reliable
direction. There-and-back → the net displacement is small and mostly tells
you which leg was longer, which is not a direction you want to walk. The
student only ever receives the displacement (the centered endpoint); the
Waypoint only decides the volume. And because agreement is measured on the
*student's* candidate support with the *student's* weights, it is evaluated
exactly where the update will land — no teacher-forced prefixes.

## What I learned by implementing it

(From `implementation/` — a CPU miniature: a shortcut-vs-truth task where a
teacher's GRPO run genuinely learns the salient 65%-shortcut first and the
subtle 100%-truth second, creating real early/late opposition on conflict
instances; then all seven transfer objectives, 3 teacher seeds × 3 student
seeds, ~3 minutes total.)

- **Building the phenomenon was harder than building the method.** The
  method is ~150 lines. Getting a teacher RL run whose early and late shifts
  genuinely oppose took most of the debugging: the first design (SFT on a
  biased annotator) collapsed into the GRPO exploration blind spot — a
  confident shortcut policy makes every conflict-instance rollout group
  all-wrong, the group advantage is exactly zero, and learning freezes with
  no gradient and no error. The fix was salience-ordered feature learning
  from a chance init: RL discovers features in order of visibility, so the
  shortcut consolidates first (early shift raises the wrong token on conflict
  instances) and the truth corrects second (late shift reverses it). The
  two-phase structure is *natural*, you just have to give it room.
- **The gate works exactly as advertised at the prefix level.** On conflict
  instances the early/late weighted cosine is −0.59/−0.74 (79–87% of prefixes
  negative); on agreement instances +0.78 to +0.88 (6–11% negative) — with
  comparable endpoint magnitudes (~1–4 nats) on both. The paper's motivating
  figure (same endpoint, opposite histories) is not an illustrative cartoon;
  it is easy to produce and measure.
- **−inf is a landmine in log-prob-difference methods.** A saturated final
  teacher underflows some candidate probabilities to exactly zero in float32;
  log 0 = −inf enters the centered shifts and the cosine; one poisoned
  prefix NaNs the whole batch mid-training. Every teacher log-prob that gets
  differenced is now clamped at −60 nats. The paper's ε = 1e−8 low-norm
  fallback does not catch this (the norms are infinite, not tiny).
- **The zero-sum gradient is real and pleasing.** Because Σ_a A_t(a) = 0,
  the softmax partition term cancels and each candidate's logit is pushed by
  exactly its own advantage (Eq 16 of the paper). The code confirms it: the
  distillation term literally cannot inflate the candidate mass.

## What surprised me / was harder than expected

- **The paper's headline ordering does not survive miniature scale — and the
  reason is interesting.** In my toy, ungated transfer (0.933) beats gated
  transfer (0.863), the reverse of the paper. The gate suppresses
  conflict-prefix endpoints as designed — but in a 2-answer vocabulary those
  "cancelled" endpoints still point at the true answer, because the late
  correction overshoots the early consolidation. The student (a
  generalizing MLP) recovers the true feature from them anyway. The per-
  subset split is unambiguous: waypoint *matches* ungated transfer on
  agreement instances (0.993 vs 0.994) and loses only where the gate is
  doing its job (conflict subset 0.628 vs 0.810). So the method's value
  hinges on an empirical property of the regime — cancelled endpoints being
  noise rather than weak signal — that a 16-token vocabulary cannot express
  and a 150k-vocab reasoning trace plausibly can. That is a sharper
  statement of *when* this method should work than anything in the paper.
- **Gating bought stability even where it cost accuracy.** The gated runs
  have the lowest seed variance of all transfer methods (std 0.063 vs
  0.098–0.111), and Direct-OPD produced one collapsed seed (0.808 on a
  teacher where its siblings hit 1.00). Suppressing opposed-history
  endpoints is a variance-reduction move; the paper's benchmark means cannot
  show this, but its Appendix-G diagnostics (zero-gate fraction rising as
  the student drifts) are the same phenomenon.
- **κ_B saturates its clip in the toy** (mean raw gate ~0.5 → τ/g ≈ 1.6 →
  clipped to 1.5). The paper's κ stays in [0.96, 1.22]. Not a bug — a
  reminder that the normalizer's clip bounds are load-bearing hyper-
  parameters, not furniture.
- **The OPD baseline flips rank at toy scale.** Imitating a near-perfect
  final teacher through a 16-token vocab is nearly unbeatable (0.904);
  in the paper it is the weakest method. The relative difficulty of
  "match the absolute distribution" vs "follow the change" is itself
  scale-dependent.
- **Wall-clock honesty check**: the paper reports Waypoint-OPD training
  *faster* than Direct-OPD despite one more scoring pass, because gated
  students write shorter responses. The toy has no length channel
  (fixed 4 tokens), so I could not test the paper's most surprising
  empirical claim at all.

## Honest findings (miniature scale)

Replicates: the two-phase teacher dynamics; prefix-level reversal detection
(the gate fires on opposing histories and passes aligned ones); the
zero-sum/centering algebra (Eq 14–16); κ_B normalization mechanics; weak-to-
strong transfer (students match/exceed an imperfect final teacher);
waypoint > no-norm (normalization ablation direction). Does not replicate:
the aggregate waypoint-beats-everything ordering (reverses at toy scale, for
the recoverable-endpoint reason above); OPD as weakest baseline. Not testable
here: response-length effects, 100-step→8-GPU-H20-scale compute, MoE
students, waypoint-position sensitivity beyond the rule I used.

## Video angle

(For the OpenMontage channel — this repo feeds a YouTube channel about ML
papers.)

- **52-second Short — "The same ending can hide opposite stories."** Split
  screen: two hikers (early/late policy shifts) both end at the same point
  (the endpoint delta). Left: both walked the same direction — trustworthy.
  Right: one walked out, the other walked back — the endpoint is the small
  gap between them, i.e. noise. Punchline graphic: "RL distillation reads
  only the endpoint. Waypoint-OPD asks how you got there." End card: gate =
  cosine of the two legs, zero when they cancel; +2 points average over
  endpoint-only across 6 settings in the paper.
- **2–3 min explainer — "Distill the journey, not the destination."** Act 1:
  RL is expensive, distillation is cheap, transfer the *delta* (Direct-OPD).
  Act 2: the ambiguity — animate two probability trajectories whose
  difference is identical but whose histories reinforce vs cancel (use my
  toy's real numbers: cosine −0.74 on conflict instances, +0.88 on
  agreement, same endpoint norms). Act 3: the gate — one frozen mid-training
  checkpoint, one extra scoring pass, agree → pass, oppose → suppress; show
  the paper's table (Waypoint-OPD best in all 6 settings) and the surprise
  (students beat their teacher, and train *faster* because they ramble
  less). Honest closer: my miniature shows the detection mechanism works
  perfectly but its benefit depends on cancelled endpoints being noise —
  scale decides.
- **52-second Short — "When the student beats the teacher."** Hook: a 14B
  student outscoring its 4B RL teacher (59.2 vs 56.1) by *not imitating* it.
  Beat 1: endpoint transfer = "push where RL pushed". Beat 2: the gate
  photo — one mid-run checkpoint that vetoes contradictory pushes. Beat 3:
  the counterintuitive billing: the extra teacher pass made training 3.7%
  faster, not slower. Tagline: "The best teacher is a checkpoint you were
  going to delete."

## References

- Paper: https://arxiv.org/abs/2610.21601
- Official code: https://github.com/hhhbxloveai/WaypointOPD
- My implementation: `implementation/` (data.py, model.py, train.py — run
  `python3 train.py`, ~3 min CPU; honest results in `run_output.txt` and
  `results.json`)
- Breakdown: `breakdown.md`
