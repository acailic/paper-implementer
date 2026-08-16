# Breakdown — MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations

> **Paper:** MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations
> **Authors:** Qiming Shi, Yulong Tao, Linbo Jin, Zhaolu Kang, Yibo Dou, Jiawen Zhu, Tianjun Pan, Shaokang Fu, Chengyu Wang, Siyue Li, Yaping Cheng, Di Weng, Chengfu Huo (Alibaba Group, ZJU, PKU, Fudan)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2607.28956
> **Code (official):** https://github.com/KhanCold/merchantbench

---

## 1. Problem & Motivation

- **Problem.** Existing LLM-agent benchmarks evaluate *bounded* tasks with
  immediate success criteria (one question, one tool call, one episode of a
  few steps). Real deployments instead demand **Long-Term Coherence (LTC)**:
  maintaining purposeful behavior over hundreds of decision points while
  adapting decisions to *accumulated evidence that arrives with
  heterogeneous delays*. No benchmark measures this.
- **Why it matters.** Agents that look competent on short benchmarks may
  drift, abandon goals, or stop acting entirely when horizons stretch to
  months. The paper shows this empirically: every tested frontier LLM
  degrades over a 365-day simulation, and the best reaches only 27.3% of
  human performance.
- **Prior approaches and limits.** Agent benchmarks fall into (a) static
  QA/knowledge, (b) bounded tool-use/web tasks (WebShop, WebArena, GAIA…),
  (c) game/embodied environments (ALFWorld, ScienceWorld), and (d) a few
  long-horizon simulators (e.g. 30-day store ops) — but none combine
  (i) a horizon of hundreds of days, (ii) actions that *constrain future
  choices* (committed cash, occupied listing slots), (iii) feedback whose
  delay varies from hours (supplier events) to days (order outcomes), and
  (iv) *measurable cumulative effects* of incoherence (net assets).

## 2. Key Insight / Contribution

- **Core idea.** Seller-side e-commerce is a natural LTC testbed: run an
  order-level, hour-by-hour simulation of a 365-day wholesale store
  ("you are the merchant") grounded in 98,843 real 1688 product records,
  where actions commit cash immediately while their consequences surface
  days later. Coherence failures then show up directly in the asset curve.
- **What is new.** (1) A POMDP environment with *mixed-latency* feedback —
  promptly observable Upstream Supplier Events vs delayed Downstream Order
  Outcomes — forcing agents to track individual order lifecycles. (2) A
  conceptual analysis framework splitting LTC into *Operational Coherence*
  (staying active) and *Strategic Coherence* (goal consistency +
  evidence-calibrated adaptation), with metrics like the Sustained Working
  Rate. (3) A humans-vs-LLM gap quantification: best LLM = 27.3% of human
  mean final net assets; most LLM configs barely beat a rule-based script.

## 3. Method

The paper's method = the *environment + evaluation protocol*. No new model
is trained; 8 LLMs are plugged into 2 agent frameworks and compared against
humans and a rule baseline.

### 3.1 Overview

MerchantBench = a finite-horizon POMDP simulating one wholesale store for
365 days at hourly resolution. The agent (an LLM) gets a decision window
every 12 hours (730 windows total) in which it may invoke a sequence of
26 tools (search catalog, list/delist/reprice products, query finances,
monitor orders/suppliers, read/write a memory doc, end the step).
Four interdependent decision components:

1. **Product Sourcing** — pick products from the 98,843-product catalog.
2. **Listing & Pricing Control** — up to 50 active listings; changes take
   effect on demand only from the *next* step.
3. **Cash-Flow Management** — placing an order *commits* cash immediately
   (procurement debit); revenue arrives only after delivery + settlement.
   Fines draw from the security deposit; deposit exhausted ⇒ store closes.
4. **Mixed-Latency Feedback Adaptation** — supplier events are visible at
   once; order outcomes (refund, bad review, late shipment) surface days
   later and must be traced back to the sourcing decision that caused them.

### 3.2 Architecture

```
                 ┌──────────────────────── 365 days, hourly ticks (t = 0..8759) ────────────────────────┐
                 │                                                                                     │
  REAL DATA      │   SIMULATOR (latent state s_t)                                                      │
  98,843 prods ──┼─▶ demand: λ_{m,i,t} = D_{i,d(t)}·w_{c(i),h(t)}·r_{m,t}·ℓ_{m,i,t}·(p/p_ref)^{-ε_i}    │
  36,576 suppliers├─▶ orders N ~ Poisson(λ) ──▶ latent outcome presampled ──▶ lifecycle state machine    │
  daily reports  │        Placed→Procured→Shipped→Delivered→Settled  (or abnormal terminal)             │
                 │   supplier events (price↑, delisting, delay) + recovery times                        │
                 │   store rating R_{m,d} (daily EW update) ──▶ demand multiplier r_{m,t}                │
                 │   cash B, deposit D, in-transit I, receivables Q                                      │
                 └───────────────┬───────────────────────────────────────────────────┬─────────────────┘
                                 │ observation Z (merchant-visible only)             │ reward (only at T)
                                 ▼                                                   ▼
                 ┌───────────────────────────────────┐                    J(π) = E[R(s_T)],
                 │  AGENT (LLM in ReAct or Hermes)   │                       R = B_T + D_T + I_T + Q_T
                 │  every 12h: tool-call sequence    │                          (net assets, RMB)
                 │  26 tools + memory doc            │
                 └───────────────────────────────────┘
```

### 3.3 Forward pass / pipeline

**Per hour t, the simulator:**
1. Fires any due order-lifecycle transitions (settle, refund, fine, review)
   and supplier-event recoveries.
2. Samples 3 Upstream Supplier Events (price change ×[0.9,1.5], product
   delisting, shipment delay +12–96h) with product-level probabilities.
3. Replenishes supplier inventories (hourly, 1–19 units).
4. Computes λ_{m,i,t} for each active listing; samples N ~ Poisson(λ);
   each arrival becomes an order candidate with **one presampled latent
   outcome** (Normal / Cancellation / Returnless Refund / Return+Refund /
   Bad Review) drawn from the product's risk profile.
5. Each new order immediately procures at the current supplier price
   (debit cash, decrement supplier stock) — drop-shipping, no merchant
   inventory. Supplier stockout ⇒ fine RMB 5. Missed 48h dispatch
   deadline ⇒ fine RMB 3 and the order continues.

**Every 12 hours** the agent observes the merchant-visible state and may
act; listing/price edits affect demand from the next step.

**At t = 8,760** demand stops; active orders run to terminal settlement
(T ≥ Hc); final reward = net assets. Zero intermediate reward — the agent
must infer progress from its own bookkeeping.

**Store rating (daily):** each eligible order contributes experience score
u ∈ {4.5, 3.0, 2.0, 1.5, 1.0} with evidence weight v ∈ {1,1,1,2,2,3}
(normal / late / return+refund / returnless / bad review / stockout) into
an exponentially-decayed weighted mean; the continuous rating maps to star
bands with demand multipliers 0.10–1.20 — so quality failures propagate to
the *entire* portfolio's demand.

### 3.4 Loss function

No training loss (benchmark paper). The optimization objective for the
agent is the return above; for the benchmark, the reported metrics are:

- **Business**: final net assets (primary), GMV, margin, order count, fines.
- **Store Reliability**: rating, anomaly rate.
- **Long-Horizon Activity**: avg active listings, tool calls, and
  **SWR (Sustained Working Rate)** = the minimum, over all rolling 30-day
  windows, of the share of decision windows with ≥1 environment tool call.
  SWR directly measures Operational Coherence (activity decay ⇒ low SWR).
- **Time-aware Sourcing Gain** G_r = Σ_m H_{r,m}(S_{r,m} − B_{r,m}) / Σ_m H_{r,m}:
  monthly demand alignment of the listing portfolio (S) vs a fixed-mix
  counterfactual (B), weighted by listing hours H — measures seasonal
  reallocation (Strategic Coherence).

## 4. Math

**1) POMDP.** M = ⟨S, A, P, O, Z, R, µ0, Hc⟩, Hc = 8,760 hourly steps.
   - s_t ∈ S: clock, demand profiles, supplier conditions, listings,
     finances, active orders, pending events. a_t = a *sequence* of tool
     invocations (or null). P mixes tool-induced changes with autonomous
     demand/supplier/order evolution. Z reveals only merchant-visible info.
   - Plain English: the agent never sees true demand, risk parameters, or
     future events — it must act under uncertainty and delay.

**2) Demand intensity.**
   λ_{m,i,t} = D_{i,d(t)} · w_{c(i),h(t)} · r_{m,t} · ℓ_{m,i,t} · (p_{m,i,t}/p_i^ref)^{−ε_i}
   - D_{i,d}: real daily demand of product i on data day d (from 1688 data).
   - w_{c,h}: hour-of-day profile splitting category demand across 24 hours.
   - r_{m,t}: store-rating multiplier (star bands above).
   - ℓ_{m,i,t}: listing exposure — cold-start ramp then decay:
     g(a) = ℓ0 + (1−ℓ0)·a/T_r for listing age a ≤ T_r (linear ramp),
     g(a) = ℓ_min + (1−ℓ_min)·e^{−κ(a−T_r)} for a > T_r (exponential decay).
     Config: ℓ0=0.2, T_r=14 days, κ=0.0092, ℓ_min=0.10.
   - (p/p_ref)^{−ε_i}: price elasticity — pricing above the reference price
     cuts demand at product-specific rate ε_i.
   - Orders: N_{m,i,t} ~ Poisson(λ_{m,i,t}).
   - Plain English: daily real demand is spread over hours, then scaled by
     how well-rated your store is, how fresh your listing is, and how your
     price compares to the catalog reference price.

**3) Store rating.**
   R_{m,d} = (α·R0 + Σ_{j∈C} γ^{d−1−d_j} v_j u_j) / (α + Σ_{j∈C} γ^{d−1−d_j} v_j)
   - R0 = 4.0 prior, α = 20 prior weight, γ = 2^{−1/30} (30-day half-life).
   - Plain English: a decayed weighted average of per-order experience
     scores; old evidence fades in 30 days; bad evidence weighs 2–3× more.

**4) Return.** J(π) = E[R(s_T)], R(s_T) = B_T + D_T + I_T + Q_T
   (cash + deposit + in-transit procurement funds + receivables).
   - Plain English: final wealth, nothing before — a pure delayed-reward
     credit-assignment stress test.

**5) SWR.** SWR = min over 30-day windows W of (#windows in W with ≥1 env
   tool call) / (total decision windows in W).
   - Plain English: the agent's *worst month* of engagement; abandoning the
     task for any month caps this at that month's activity level.

## 5. Training

Not a training paper — evaluation protocol instead:

- **Subjects**: 8 LLMs (GPT-5.6 Sol, Claude Opus 4.8, Qwen3.7-Max/Plus,
  GLM-5.2, DeepSeek-V4-Pro/Flash, Kimi K2.6) × 2 agent frameworks
  (ReAct; "Hermes" harness with skills/memory) × 3 runs = 48 runs.
  Plus 3 human participants and a rule-based baseline script.
- **Initial conditions**: cash RMB 2,000, deposit RMB 1,000, ≤50 listings.
- **Context management**: ReAct history truncated to 30k tokens past 160k
  (with a reminder to summarize into the memory doc); Hermes uses its
  default summarization; each model summarizes for itself.
- **Lifecycle constants**: promised shipment 48h; delivery→settlement and
  outcome realization ≤168h each. Fines: stockout 5, late 3, return+refund
  8, bad review 5 RMB. Supplier: init inventory 20–399, capacity 50–499,
  hourly replenishment 1–19, base dispatch 1–47h, logistics 12–72h.

## 6. Results & Ablations

- **Headline.** Human mean final net assets 217.61k RMB. Best LLM:
  Hermes/Qwen3.7-Max 59.46k (27.3% of human). Best ReAct: GPT-5.6 Sol
  40.89k. Rule-based baseline 24.48k — *most LLM configs barely beat a
  simple script*. Humans: 9,442 orders, 49.1 avg active listings, SWR 100%;
  LLM SWR range 10.6–99.4%.
- **Framework ablation.** Hermes > ReAct on 7/8 models (avg +53.3% net
  assets) — harness memory/skills matter as much as the model.
- **LTC ablations (the conceptual core):**
  - *Operational decay*: e.g. ReAct Qwen3.7-Max effective-window rate
    68%→23% across quarters; Hermes Kimi K2.6 "gave up" on day 104 and
    acted in only 168 of 523 remaining windows (premature abandonment).
  - *Goal drift*: sourcing loop collapses into reactive supplier-event
    firefighting (supply checks 14%→34% of calls); a Qwen run misremembered
    the end date and stopped refilling slots for 83 days; a Claude run
    falsely believed delisting weak products concentrates traffic and
    shriveled from 47 to 3 listings.
  - *Evidence calibration*: humans raise procurement prices over the year
    (43–53 → 59–91 RMB) as liquidity grows; weak agents stay flat.
  - Sourcing gain G correlates with final net assets; 17/24 Hermes runs
    auto-created a reusable "RealShop" operating skill (quality varied).
- **Takeaway**: the bottleneck is coherence, not competence — failures are
  activity decay, memory errors, and uncalibrated adaptation, not lack of
  tool skill.

## 7. Limitations

- Simulated economy: demand model is a multiplicative factorization fitted
  to real data — real platform dynamics (competition, ad auctions, other
  sellers) are absent; the merchant is a monopolist on its listings.
- 3 human participants is a small reference group; 217.61k mean may be
  noisy, and humans likely differ in effort/familiarity.
- Single domain (e-commerce ops); LTC conclusions may not transfer to
  other long-horizon settings (research, personal assistance).
- Rule-based baseline is simple; "most LLMs barely beat it" partly reflects
  LLM incoherence, partly that the task rewards patient bookkeeping.
- LLM-written daily market reports inject an LLM prior into the
  environment; 48 runs is modest for variance at 365-day horizons.
- Agent-side summarization is model-dependent — conflates harness quality
  with summarization ability.

## 8. Open Questions / Ideas

- Would an explicit *calendar/planner* tool (milestones, seasonal plan)
  fix operational decay better than free-form memory docs?
- SWR punishes abandonment but not busywork — is there a metric for
  *useful* sustained activity?
- The exposure decay ℓ(a) means listings rot: is the optimal policy a
  steady-state relisting rotation? Derivable analytically in a simplified
  model — a good toy-theory exercise for our implementation.
- Credit assignment: can a turn-level credit method (cf. AgentOPSD, next
  in our queue) recover evidence-calibrated adaptation here?
- For our re-implementation: reproduce the environment core (Poisson
  demand × exposure × rating × elasticity, order lifecycle with presampled
  outcomes, cash/deposit/fines, upstream events + recovery) with a
  synthetic seasonal catalog and a rule-based merchant as the baseline —
  then measure whether the rule agent's own SWR stays pinned at 100%.
