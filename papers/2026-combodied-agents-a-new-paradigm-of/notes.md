# Combodied Agents — Reading Notes

> **Paper:** "Combodied Agents: a New Paradigm of Human-Centric Agentic AI"
> **Authors:** Qianggang Ding, Xingyao Wang, Rui Feng, et al. (UdeM/Mila, A*STAR, and 16 more institutions; corresponding: Ding, Liu)
> **arXiv:** 2608.10915 (v2, 12 Aug 2026) · cs.AI — preprint, ~50 pages
> **Code/Project:** none linked (position/framework paper)

---

## Pass 1 — first impressions (skim)

This is a **position / framework paper**, not a methods paper with experiments.
No new model, no benchmark numbers of its own — the contribution is a
*definition*, a *closed-loop technical framework*, a *taxonomy*, and an
*evaluation/governance agenda* for a proposed class of agents called
**Combodied Agents** (portmanteau of **Companion + Body**).

The genre is like "Agent Constitution meets systems blueprint": long, dense,
well-cited, organized as a research program others are invited to execute.

## The problem, in my own words

Current agentic AI comes in two flavors:

- **Digital Agents** — act on digital states (GUIs, code, APIs). Success =
  task correctness/completion.
- **Embodied Agents** — act on physical states (robots, vehicles). Success =
  successful, safe physical action.

Both optimize *external task states*. Neither makes **the person's evolving
state** (health, cognition, emotion, habits, capability, relationships) the
primary object of modeling, intervention, and evaluation. Opening example
(medication reminder): a digital agent can re-send a reminder; an embodied
agent can bring the pill box; **neither models *why* the dose was missed**
(forgot? confused? side effects? deliberate refusal?) or what support would
actually be appropriate.

Worry driving the paper: task success and human development can **diverge** —
AI can finish the document while leaving the author less capable. Metrics
reward substitution by default; dependence, deskilling, manipulation are
invisible to F1.

Their answer: agents organized around **beneficial trajectories of human
state and agency**, with tools/robots/sensors/human services as mere *action
channels*. Not maximal automation as success — sustained human benefit.

## Key constructs spotted on pass 1

- **Definition 1 + five properties**: human-centric state modeling,
  longitudinality, intervention (remind/coach/execute/escalate), co-agency
  (adaptive division of labor, replacement not the default), agency
  preservation (autonomy, control, dignity, long-term capability).
- **Action substrate** (§2.2): the *class of target states* that organizes an
  agent's modeling/decision/evaluation. Digital / Embodied / Combodied are
  three overlapping centers of gravity (Venn with cross-substrate tasks like
  robot-assisted medication in the middle). Generic loop:
  `o_t → b_t → (g_t, p_t) → a_t → x_{t+1} → o_{t+1}`.
- **Deliberate distance from Human Digital Twins**: no exhaustive replica of
  the person; instead *purpose-bounded, uncertainty-aware, user-correctable*
  representations tied to an agreed support context.
- **Closed loop** (four modules + feedback):
  1. event-based multimodal **perception** (recover *events that matter* from
     noisy, intermittent, unevenly sampled streams — not hoard everything);
  2. longitudinal, correctable **memory** (provenance + uncertainty
     preserved; *intervention-response memory* as first-class evidence);
  3. **Personal World Model (PWM)** — calibrated distributions over future
     personal states/outcomes *under alternative decisions and interventions*;
  4. **admissible intervention policy** — proportionate support under
     consent, uncertainty, safety, reversibility, user control.
- **Edge-native personal models** (§5): cloud→edge migration, user-owned
  memory/PWMs, local policies with selective cloud use.

---

## Pass 2 — deep read (section by section)

### §2 Foundations — the actual formalization

**Definition 1** (paraphrased tightly): a Combodied Agent *perceives, models,
and influences the evolving state of a person* through continuous multimodal
sensing + longitudinal interaction. The name = Companion + Body, but it is
NOT a chatbot companion; the human body/behavior/cognition/emotion/context is
the primary *perceptual and action domain*. Defining characteristic:
perception of humans, understanding of humans, **action on human states**.

Five jointly-necessary properties: (1) human-centric state modeling,
(2) longitudinality, (3) intervention (remind/recommend/explain/coach/
coordinate/protect/execute/escalate — not just respond), (4) co-agency
(adaptive division of labor, replacement not the default), (5) agency
preservation (autonomy, control, dignity, relationships, long-term
capability). Important nuance: these are a "center of gravity", not a rigid
interface checklist — a chatbot that remembers a name is not automatically
Combodied; a Combodied agent may act purely through conversation.

**Action substrate** = the class of *target states* that organizes modeling,
decision, intervention, evaluation. Key claim: classify agents by what their
actions are ultimately *for* (which user-state trajectory + success
criteria), not by interface, model, or actuation mode. Overlaps are
cross-substrate task configurations (e.g., robot-assisted medication =
digital + physical + human-state).

**The math of §2.4 (closed loop)** — equations 1–6, in my notation:

- **Eq. 1 (generic agent loop):** `o_t → b_t → (g_t, p_t) → a_t → x_{t+1} → o_{t+1}`.
  Observation → belief → (goal, plan) → action → environment state → next
  observation. Feedback may update belief/plan/action *without* implying
  weight updates.
- **Eq. 2 (latent state posterior):** `Z_t ~ q_φ(· | D_≤t, M_t, C_t)`.
  The *unobservable human state* `H_t` is estimated from `D_≤t` = the
  **event-evidence records** (not raw data!) reconstructed up to t, plus
  longitudinal memory `M_t` and current context `C_t`. Uncertainty-bearing
  posterior, not a point estimate.
- **Eq. 3 (memory read):** `R_t = Read(M_t; Z_t, C_t, G_t)` — retrieve
  decision-relevant evidence given state, context, and user goals `G_t`.
- **Eq. 4 (human transition):** `H_{t+1} ~ T_H(· | H_t, a_t^agent, a_t^user, Ξ_t)`.
  The next human state depends on the agent's action AND the user's own
  action AND exogenous influences Ξ. This is why intervention effects are
  inherently uncertain — the agent controls only one input.
- **Eq. 5 (observation):** `O_{t+1} ~ Ω(· | H_{t+1}, C_{t+1})` — a noisy,
  context-dependent view of the latent state.
- **Eq. 6 (memory update):** `M_{t+1} = Update(M_t, O_{t+1}, a_t^agent, Y_{t+1}, F_{t+1})`
  — memory absorbs new observations, outcomes Y, and explicit/implicit
  feedback F. This closes the loop *without* assuming deterministic or
  immediately observable intervention effects.

**Longitudinal memory = temporal evidence, structured into 7 components**
(Table 2): episodic, semantic person memory, trajectory memory (health/
behavior/emotion/cognition/routines over time), goal & commitment memory,
relationship memory, **intervention-response memory** (links past
interventions → acceptance/rejection/benefit/harm), and user-control memory
(do-not-remember rules, corrections, deletion requests). Each memory item
carries: content, timestamp, source, modality, confidence, sensitivity,
relevant goals, intervention/outcome links, retention policy. Provenance and
uncertainty must survive writing/consolidation/retrieval/correction.

**Action space (Table 3)** — 10 action types, each with a distinct *primary
target*: Inform (cognition), Remind (attention/execution), Recommend
(decision-making), Coach (capability/habit), Nudge (behavioral tendency),
Reflect (self-understanding), Coordinate (social/institutional), Protect
(safety/autonomy), Escalate (high-risk states), Execute (delegated external
action). Crucially: **non-intervention is always in the action space** —
silence, clarification, confirmation, intensity reduction, referral.

### §3 Perception — event-evidence, not data streams

Perception defined as **event-based personal data perception**: acquisition,
filtering, alignment, interpretation of *fragmentary* personal data. The
output is NOT a raw stream — it's **governed event-evidence records**
(what was observed, when/how acquired, which interpretation it supports,
remaining uncertainty, whether relevant enough to influence memory/action).

Modality-by-modality (§3.1–3.8): language (most direct for goals/consent but
subject to social desirability), speech/audio (prosody + non-speech events),
vision (episodes not frames; process locally, keep event descriptions),
physiology (personal baselines > population thresholds; same HR is fine
during exercise, concerning at rest), motion/behavior (changes that matter:
falls, missed routines, gait change), social/relational (data rarely belong
to one person — participant-scoped visibility needed!), environment
(disambiguation context), institutional records (three relevant times:
event time, entry time, receipt time).

The pipeline logic (§3.9–3.10): quality gates at acquisition → temporal
alignment respecting uncertain boundaries → cross-modal comparison
(corroborate/contradict) → reconstruction against personal baselines.
**One high-quality user correction can outweigh multiple agreeing sensors.**
The chain to keep straight (this is THE key architectural invariant):

> observation → event → inferred state → predicted trajectory → authorized intervention

Example: elevated HR = observation; elevated HR + slow recovery after
exercise = reconstructed event; possible fatigue = inferred state; delayed
recovery if exercise continues = predicted trajectory; recommending rest =
policy decision. Fusion must be person-calibrated and provenance-aware, and
*purpose-limited* (once evidence suffices, more sensing = privacy cost with
no decision benefit). If alternatives remain unresolved, **clarification is
a legitimate perceptual outcome** — stronger inference is not always better.

### §4 Personal World Model — the core technical abstraction

**Definition:** a PWM is a *purpose-bounded, individual-specific
event-dynamics model*. Input: governed event history `D_≤t`, spatiotemporal
context `C_t`, and a **candidate scenario** `s_{t:t+Δ} = (a^user, a^agent, Ξ)_{t:t+Δ}`.
Output: calibrated distribution over future states, events, outcomes.

- **Eq. 7 (PWM predictive distribution):**
  `p_θ(Z_{t+1:t+Δ}, E_{t+1:t+Δ}, Y_{t+1:t+Δ} | D_≤t, Z_t, C_t, G_t, s_{t:t+Δ})`
  where E = future observable personal events, Y = scenario-relevant outcomes
  (adherence, goal progress, wellbeing, capability, safety, relationship
  quality, agency). Unresolved scenario components are *sampled/marginalized*
  in rollouts. The defining comparison: how the trajectory distribution
  **changes across explicit alternatives** (non-intervention vs clarification
  vs timings/intensities vs acceptance/refusal).
- **Eq. 8 (policy as constrained Pareto selection):**
  `a_t^{agent,*} ∈ ParetoArgmax_{a∈A_t^adm} E_{p_θ}[U(Y_{t+1:t+Δ}, G_t)]`.
  The admissible set `A_t^adm` enforces consent, scope, safety, uncertainty,
  reversibility, escalation, and *includes non-intervention, clarification,
  referral*. U is a vector — benefit, capability, autonomy, relationships
  stay explicit trade-offs, NOT collapsed into one scalar. A user-approved
  selection rule picks among non-dominated actions. **Safety cannot be
  traded for engagement/average utility.**
- **Eq. 9 (causal extension):** `p_θ(Y_{t+1:t+Δ} | do(a_t^agent = a), Z_t, R_t, C_t)`
  — an *interventional* distribution, explicitly NOT an individual
  counterfactual (that needs potential outcomes or abduction–action–prediction
  in an SCM). Writing `do(·)` doesn't remove confounders (motivation, hidden
  context, selective engagement). Identification needs a defensible causal
  graph + consistency/positivity/time-varying-confounding control, ideally
  micro-randomized trials or N-of-1 studies. **High-risk systems must not
  explore on humans just to improve the model.**

PWM ≠ profile / memory / personalized agent / generative agent (Table 5):
the *functional contract* is action-conditioned, person-specific, calibrated
state–event–outcome dynamics, validated on trajectory validity,
intervention-response accuracy, calibration, scenario discrimination.

Learning paradigms (§4.3): latent dynamics (MBRL tradition), generative
scenario simulation, causal intervention modeling, mechanistic/hybrid,
mental-state modeling, memory-augmented, hybrid foundation+personal
(population priors + individual adaptation layers). Practical constraints:
sparse personal data → start from population priors, adapt limited personal
components, maintain parameter/state posteriors, detect drift, support
correction/reset. **Different horizons = different models** (fatigue minutes
ahead ≠ habit weeks ahead ≠ capability months ahead). Fidelity requirements
are domain-scaled: medication companion needs calibrated risk + clinical
boundaries; emotional companion needs relationship-safety modeling because
prediction errors can intensify dependency.

### §5 Edge — three stages of where authority lives

Stage I (cloud-centric): cloud does reasoning/memory; device is interface +
sensors. OK for prototyping/cold-start/low-risk. Failure mode: treating
retrieved context as if it were a persistent personal model.
Stage II (hybrid): edge = privacy/interpretation/authority *mediator* —
authoritative sensitive memory stays local, disclosure permissions enforced,
cloud outputs reviewed before touching protected memory or triggering
action. Not "a small model on device" — authority is the criterion.
Stage III (edge-native): authoritative memory + PWM + policy + safety
boundaries on user devices; cloud invoked selectively, results return for
local contextualization/authorization. Defining property is not "everything
local" but **interpretation and intervention authority under user control**
(inspect, correct, delete, pause, export, reset, migrate).
Local evolution targets 4 things: memory evolution, perception calibration
(user baselines), dynamics adaptation (PWM), policy adaptation (which
timing/modality/intensity works). Every update inspectable, reversible,
safety-bounded.

### §6 Evaluation — agency preservation as first-class metric

Layer × horizon matrix (Table 7): perception/memory, PWM/decision,
intervention, human-outcome — each at interaction, episode, and longitudinal
horizons, with **severe failures non-compensatory** (drift, persistent false
memory, unauthorized retention, unsafe exploration, manipulation,
irreversible unauthorized action, dependence, skill erosion).

**Eight agency-preservation metrics** (§6.3): autonomy preservation,
contestability & correction, informed decision-making, capability
preservation & growth, over-reliance/dependence risk, reversibility &
accountability, boundary & consent respect, relationship & social-world
preservation. Explicitly NOT task success / satisfaction / engagement — an
agent can complete tasks and still fail agency. Requires with/without-agent
baselines and policy comparisons (direct execution vs confirmation-based vs
reflective coaching vs no intervention).

**CombodiedBench** (§6.4): modular suite — Human State Perception, Memory
Continuity, Goal Negotiation, Intervention Appropriateness, Agency
Preservation, Relationship Boundaries, Escalation, Longitudinal Outcomes.
Unit = longitudinal scenario episode (prior trajectory, context, evidence,
permissible actions, delayed outcome window, authority spec, scoring rule,
unacceptable failure modes).

## Answers to my Pass-1 open questions

1. **Is "admissible" formalized?** Yes — Eq. 8: `A_t^adm` is a hard
   constraint set (consent/scope/safety/uncertainty/reversibility/
   escalation) that *filters before* Pareto optimization over a vector
   utility. Non-compensatory by construction.
2. **Is the PWM mathematically defined?** Yes — Eq. 7 gives the predictive
   contract; Eq. 9 the causal extension; Table 5 the functional-contract
   boundaries. It's a *contract*, not an architecture: any estimator
   satisfying it counts.
3. **What makes an event concrete?** A governed event-evidence record with
   acquisition metadata, processing provenance, observed-vs-inferred fields,
   alternative explanations, the personal baseline used for deviation, and
   governance fields (sensitivity, consent basis, retention).
4. **Intervention-response memory?** Table 2 component: links interventions
   to acceptance/rejection/benefit/harm; feeds Eq. 6 updates and PWM
   validation; explicitly the evidence base for adapting future support.
5. **Non-compensatory scoring?** Yes — severe failures (manipulation,
   unauthorized irreversible action, privacy violation, harmful dependency)
   reported as critical, not averaged into a score.
6. **Purpose-boundedness in practice?** = scope tied to an agreed support
   context; PWM is a *family* of purpose- and horizon-specific models, not a
   monolithic person-simulator; sensing is purpose-limited (stop when
   evidence suffices).
7. **Edge architecture?** Stage I/II/III above; criterion is *where
   authoritative representation + final action authority live*.

## Still unclear / to nail down at breakdown step

- The paper never instantiates `Read`/`Update` (Eq. 3/6) concretely —
  any concrete retriever/consolidator satisfying the provenance requirements
  is admissible. For my implementation I'll pick a simple, inspectable
  design.
- ParetoArgmax over a vector U with a "user-approved selection rule" — the
  paper leaves the selection rule open (by design). In the toy
  implementation I'll make it explicit (e.g., lexicographic safety first,
  then user-confirmed weighting).
- How to demo "calibration" convincingly in a toy sim — probably report
  predicted-vs-realized outcome frequencies (reliability-style) per
  intervention arm.
- Table 1 (intro) and §7 taxonomy details I only skimmed — enough for
  breakdown; revisit §7 examples when writing the writeup.

## Implementability (updated after pass 2)

Even more concrete than pass 1 suggested. The paper's own formalism maps
directly onto a toy longitudinal simulation:

- a simulated user with latent state `H_t` (e.g., adherence risk + skill +
  annoyance), transition Eq. 4 depending on `a^agent` AND `a^user` (accept/
  ignore/refuse) AND exogenous noise Ξ (busy day, side effects);
- observation process Eq. 5 (noisy sensor glimpses);
- event-evidence records built by a perception layer (observation → event);
- a belief posterior `q_φ` (Eq. 2) over the latent state (could be a small
  Bayes filter or learned encoder);
- a PWM (Eq. 7) = action-conditioned trajectory model, trained by
  simulation rollouts, evaluated on calibration + scenario discrimination;
- a policy (Eq. 8) = filter by `A_t^adm` (consent flags, reversibility,
  uncertainty threshold, escalation trigger), then Pareto/lexicographic
  selection over outcome vector (adherence benefit, autonomy/annoyance,
  dependence risk);
- metrics: task-side adherence AND agency-side (independent capability over
  time, refusal acceptance, correction propagation).

That demonstrates the paper's central claims: (a) same action helps or harms
depending on latent state; (b) admissibility gates beat naive
max-benefit policies on agency metrics; (c) intervention-response memory
improves policy adaptation over time. Decide exact toy world at breakdown.
