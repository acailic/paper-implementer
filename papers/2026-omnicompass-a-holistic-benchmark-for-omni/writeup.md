# Writeup — OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding

> Your own explanation of the paper, as if teaching it to a peer who hasn't
> read it. This is not a summary of the abstract — it's a synthesis after
> reading and implementing.

**Paper:** Liu et al., arXiv:2610.18829 (2026) · PolyU VCLab / OPPO / CSM /
NTU / SJTU.

## The one-paragraph version

Every video-understanding benchmark lives in one corner of a four-axis box:
how you point at the target (text or image), what evidence exists (visual or
visual+audio), what the model must output (when / where / what), and how long
the videos are. OmniCompass is the first to cross all four axes at once — and
its real trick is that **every single question is asked twice**: once as
text ("when does USA's #7 score?") and once as text+image ("when does *this*
player score?"), same video, same target, same ground truth. That pairing
enables a new metric family, **Cross-Specification Consistency (CS)**: your
score on a question pair is the *minimum* of (right-as-text,
right-as-image, said-the-same-thing-twice). The headline result across 50
MLLMs: models with nearly identical accuracy in the two settings are
frequently right on *different instances*, and the best CS^QA (69.1) towers
over the best temporal-grounding CS (27.1) — modern models can talk about
video far better than they can point into it.

## The problem

Video benchmarks have been measuring fragments. QA benchmarks can't tell you
*whether* the model failed to find the moment, find the object, or
understand the event — the answer is just wrong. Audio-visual benchmarks
rarely test grounding. Grounding benchmarks never test the same target via
image reference. And nobody tests whether a model binds "the player in red
#7" and a photo of her to the *same* thing. So the field's leaderboards
reward aggregate accuracy while hiding target-binding failures.

## The idea

Ask every question two ways and score the pair with `min(correctness under
text, correctness under image, agreement between your own two answers)`.
Agreement alone scores zero (a model that confidently outputs the same wrong
interval twice gets nothing); being right twice but differently is also
penalized. Wrap that in a serious benchmark: 820 videos (240 with native
audio), 7 domains, 30 s–88.4 min, ~1,800 annotation hours, TG/SG/QA per
video, paired specs per instance.

## How it works (the intuition)

- **TG ("when"):** model returns a JSON interval, scored by tIoU.
- **SG ("where"):** the harness *gives* the model the ground-truth interval
  and asks for one bounding box at each of its two boundary timestamps —
  score is the mean of the two box IoUs. This is the V-STaR protocol: SG
  here tests localization-at-a-known-time, not temporal search.
- **QA ("what"):** five options, one letter, exact match, tied to a clue
  interval (CG-Bench style).
- The paired settings share everything except how the target is specified —
  and the Text+Image wording is deliberately under-specified so the image is
  *required* ("the player in the reference image").
- Single-setting metrics per Table 2/3: TG gets R@0.5 + mtIoU, SG gets
  AP@0.5 + msIoU, QA gets Accuracy, and **Average =
  mean(mtIoU, msIoU, Accuracy)** — I verified this identity numerically
  against Gemini-3.8-Flash's row: (32.6+37.7+73.3)/3 = 47.87 ≈ 47.9. ✓

The results worth remembering: proprietary > open by 3.2 CSAvg on VL but by
**32.8** on AV (audio-visual binding is where open models collapse);
Text+Image helps only 11/36 models; frame-level timestamps buy +6.5 CSAvg on
VL (and +18 CS^TG for the best model — temporal alignment is that
fragile); audio adds ~+23 CS^QA but ≈0 CS^SG; TG degrades on long videos
for 10/10 models; and TG targets span only 4.6%/1.7% of video duration —
literal needle-in-haystack localization.

## Honest findings: what replicates locally and what doesn't

The implementable miniature (see `implementation/`) is the **evaluation
harness** — schema, prompts, parser, metrics, scoring pipeline — run on a
hand-built 39 VL + 10 AV video mini-benchmark with deterministic mock
models, CPU-only. The boundary is explicit: **no real MLLM and no real video
was evaluated**; mock scores demonstrate the machinery, not the paper's
leaderboard.

Which metric mechanics replicate (verified by 24 unit tests + two oracles):

| Mechanism | Check | Result |
|---|---|---|
| tIoU / bbox IoU formulas | hand-computed values | exact |
| SG = mean of boundary IoUs | half-right boundary → 0.5 | exact |
| Average = mean(mtIoU, msIoU, Acc) | paper's Gemini row reproduces 47.9/50.6 | exact |
| CS = min(g, g', a) semantics | oracle-confidently-wrong (identical wrong answers, a=1) → CS ≈ 0 | PASS |
| Perfect model → all metrics 1.0 incl. CS | oracle-perfect | PASS |
| CS ≤ lower single-setting score | 200-case random property test | holds |
| Parse failure → 0, stays in denominator | MockFragile: 14 zero-scored QA pairs of 39; Accuracy ≈ 0.78×skill | matches official convention |
| QA both/one/neither decomposition | flagship: 39 pairs = 28 both + 2 one + 9 neither (paper Kimi-K3: 808 = 546+87+175) | works, similar shape |
| Audio helps QA not SG (Fig. 7) | +20.0 CS^QA flagship, CS^SG unchanged (paper +18.6..+28.1, SG −0.4..+1.8) | reproduces (by construction of the mock) |
| Timestamps help (Fig. 6) | removing them: CSAvg −2.2 (VL) / −5.4 (AV) flagship; paper +6.5/+4.9 when added | direction reproduces, magnitude smaller |

Sample miniature scores (Text/Text+Image, mocks calibrated loosely to the
paper's archetypes): flagship VL CSAvg **41.7** (paper Gemini-3.8-Flash
42.8), CS^TG 28.4 (27.1), CS^SG 24.9 (32.3), CS^QA 71.8 (69.1), SG msIoU
37.4/39.9 (37.7/41.8); AV CSAvg 51.3 (57.4). Treat the closeness as
calibration, not replication — the mock's skills were *set* to the paper's
ratios.

What does NOT replicate locally, and why:

1. **Any real model ranking** — no GPU, no API budget; 50 real MLLMs on
   80.6 h of video is simply out of scope for this repo.
2. **The duration/granularity analyses** (long videos hurt TG; bigger
   targets are easier) — those are properties of real perception. A mock
   whose skill is duration-independent would show no effect, so testing it
   would be theater; the harness supports the slice, the mock doesn't earn it.
3. **AP@0.5's exact value** — the paper never defines it, and the official
   repo implements threshold metrics as plain `score >= 0.5` hit-rates (no
   ranking). I implemented that interpretation and flagged it; if the
   authors meant true detection AP, everyone's numbers move.

## What I learned by implementing it

- **The CS metric is three constraints wearing a trench coat.** Requiring
  correctness twice is a soft AND (costs ~4-6 points, the paper's main
  gap); requiring prediction-agreement is nearly free *for good models*
  (0.3–1.8 points) but is a hard floor for shaky ones. When I built the
  confidently-wrong oracle, min() correctly refused to pay out despite
  perfect self-agreement — the metric's whole point in one assert.
- **"SG" is quietly a different task than it sounds.** The model is *given*
  the interval; it only boxes two frames. So OmniCompass's SG can't fail by
  choosing the wrong moment — comparing its SG numbers to free-form
  referring-grounding benchmarks would be a category error. I'd missed that
  on both read-throughs; it only landed when I wrote the prompt builder.
- **Aggregate equality masks instance-level chaos.** My flagship mock hit
  76.9/71.8 QA accuracy but only 71.8 CS^QA — and the decomposition showed
  2 pairs right in exactly one setting. Building the decomposition (the
  paper only reports it for Kimi-K3) is 20 lines and is the most diagnostic
  output in the whole harness.
- **Parse-failure policy is load-bearing.** "Unparseable = 0, stays in the
  denominator" (the official convention) means format fragility is
  indistinguishable from incompetence in the final numbers. My MockFragile
  loses ~22% off every metric purely to formatting. Benchmarks that instead
  *dropped* unparseable rows would inflate exactly the models most likely to
  hallucinate structure.

## What surprised me / was harder than expected

- The two-column PDF with styled typography extracts terribly; I had to
  reconstruct Table 2's column semantics and then *prove* my reading via the
  Average identity before trusting any number I quoted.
- The paper's own example (Figure E3) has options scrambled — "A 0, B 2, C
  3, D 4, E 1" — a small detail that matters once you realize text-only
  models score 18.8–30.1 CS^QA with no media at all: letter priors are a
  real floor, and my mocks reproduce it (~10–20% at n=10–39).
- How little of the "AP@0.5" story is written down anywhere. Paper says
  AP@0.5; repo computes a thresholded fraction; nobody ranks anything.
  Benchmark papers need their metrics to be reproducible from the paper
  text, and this one gap forced a documented interpretation choice.

## Video angle

Angles for OpenMontage (benchmark papers are strong "the test they don't
show you" material):

- **"The leaderboard test they don't show you":** run the same clip twice —
  "when does the player in red score?" vs. "when does *this* player score?"
  with a photo — and show on-screen that models with identical accuracy
  scores answer *different instances* correctly; dramatize CS as a
  three-way min() gate with a simple animated Venn/overlap graphic.
- **"Point, don't paraphrase":** the 69-vs-27 gap — models talk about video
  far better than they point into it (needle-in-haystack: targets are ~5% of
  a video's duration, ~1.7% when audio matters). A split-screen demo of a
  model nailing the multiple-choice answer while missing the timestamp by 40
  seconds makes the finding visceral, and maps directly onto
  edit/search/clip-retrieval UX (you can't cut a highlight reel with a
  27% hit rate).
- **"The 20% you get for free":** strip the media entirely and top models
  still 'pass' ~1 in 5 five-choice video questions on language priors alone
  — a segment on benchmark floors (letter bias, distractor quality,
  clue leakage) using OmniCompass's text-only ablation plus our own
  mock-harness demo of a confidently-wrong model scoring 0 under CS.

## References

- Paper: https://arxiv.org/abs/2610.18829
- Official code + data: https://github.com/PolyU-VCLab/OmniCompass (data on
  HF: `VCLab-PolyU/OmniCompass`)
- My implementation: `implementation/` (harness + mini benchmark + mocks +
  24 tests; `python3 run.py` reproduces `implementation/results.txt`)
- Breakdown: `breakdown.md`
