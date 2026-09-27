# Combodied Agents — from-scratch toy implementation

> Paper: **"Combodied Agents: a New Paradigm of Human-Centric Agentic AI"**
> Ding et al., 2026 · arXiv:2608.10915
> Breakdown: `../breakdown.md`

The paper is a *position/framework* paper: it defines a class of agents whose
actions target **human states** (not digital or physical states), and
specifies a closed loop — event-based perception → belief over latent human
state → Personal World Model (PWM) → admissibility-gated policy → feedback
into longitudinal memory. It contains **no experiments of its own**, so this
implementation operationalizes the framework's core equations (Eqs. 2, 6, 7,
8) as a runnable toy: a medication-companion agent over a simulated older
adult, where the *same action helps or harms depending on the latent cause of
a missed dose*.

## Scenario

A simulated person (hidden per-person parameters) occasionally misses a
medication dose. The miss has one of three latent causes — `forgot`,
`confused`, `refused` — each with noisy event features (busy calendar,
disorientation signs, side-effect complaints). The agent perceives features,
infers a posterior over causes, consults its intervention-response memory,
predicts outcomes under each of 5 actions, and chooses.

Actions: `none` (silence), `remind` (nudge), `clarify` (call & explain),
`coach` (reflective support), `escalate` (caregiver). Ground truth is
cause-dependent: `remind` works for `forgot` but fails/backfires for
`refused`; `coach` is slower but builds capability; `escalate` is safe but
substitutive (costs autonomy); repeated `remind`s habituate.

## Files

| file | role |
|---|---|
| `data.py`   | `PersonSim` (Eqs. 4–5: latent transition + noisy observation), synthetic datasets for belief net + PWM, hidden `true_response` |
| `model.py`  | `BeliefNet` (Eq. 2), `PWMEnsemble` (Eq. 7), `InterventionMemory` (Eq. 6), `NaivePolicy` vs `CombodiedPolicy` (Eq. 8: hard admissibility filter A1–A4 → weighted vector utility `U = P_take + 5·d_auto + 4·d_rel`) |
| `train.py`  | trains both nets, then runs 120-day × 30-person longitudinal comparison of `no_agent` / `naive` / `combodied` policies |
| `results.json` | full output of the last run |

## Run

```bash
python3 train.py            # ~7 s on CPU (torch only, no numpy needed)
```

## Actual results (last run, 120 days × 30 people)

**(a) Same action, opposite effect by latent state** — the learned PWM's
P(take), nobody hard-codes this:

| cause | none | remind | clarify | coach | escalate |
|---|---|---|---|---|---|
| forgot  | 0.09 | **0.67** | 0.52 | 0.39 | 0.92 |
| confused | 0.04 | 0.23 | **0.70** | 0.39 | 0.92 |
| refused | 0.06 | 0.11 | 0.20 | **0.51** | 0.90 |

**(b) Task success vs agency diverges** (the paper's central claim):

| policy | adherence | relationship | self-cap | interv/dec | dominant action |
|---|---|---|---|---|---|
| no_agent  | 0.051 | 1.000 | 1.000 | 0.000 | — |
| naive     | **0.907** | 0.033 | 0.066 | 1.000 | escalate 100% |
| combodied | 0.491 | **0.939** | **0.941** | 0.767 | mixed (coach 19, remind 12, none 13, clarify 9, escalate 3 per person) |

The naive argmax-P(take) policy *completes the task* and simultaneously
destroys the relationship and the person's self-capability — it always
escalates. The combodied policy buys a 10× adherence gain over no agent
while keeping agency axes near baseline, and it sometimes chooses silence.

**(c) Intervention-response memory** — one person's take-rate after the
policy's chosen actions: 0.33 (first half) → 1.00 (second half); the action
mix shifts as per-person outcomes accumulate.

**(d) PWM calibration** — predicted vs realized P(take) per action:
remind 0.62/0.62, coach 0.42/0.50, clarify 0.70/0.89, escalate 0.91/1.00.

## Honest status

Everything runs (see `results.json` and the printed tables). The numbers are
toy-scale and seed-dependent; the point is the *shape* of the results, not
SOTA claims. The paper itself provides no baselines to compare against — its
contribution is the contract this code instantiates.
