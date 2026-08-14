# MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations — First Impressions

**Paper:** Shi et al., arXiv:2607.28956v2 [cs.AI], Aug 2026
**Code:** https://github.com/KhanCold/merchantbench

## First-pass summary (what is this?)

A benchmark paper, not a new model. The authors argue that existing LLM-agent
benchmarks test *bounded* tasks with immediate success criteria, while real
deployments need **Long-Term Coherence**: the ability to keep purposeful
behavior over long horizons while adapting decisions to *accumulated evidence*
that arrives with *heterogeneous delays*.

Their testbed: a simulated 365-day seller-side e-commerce store operation
("you are the merchant"). The agent must:

1. **Product Sourcing** — pick products from a catalog of 98,843 real product
   records, deal with suppliers (upstream events are visible promptly).
2. **Listing and Pricing Control** — list products, adjust prices; price/listing
   changes affect demand from the next step.
3. **Cash-Flow Management** — new orders *commit* cash before settlement;
   unpaid fines draw from a security deposit; if the deposit is exhausted the
   store dies.
4. **Mixed-Latency Feedback Adaptation** — order outcomes (abnormal deliveries,
   refunds, ratings damage) surface *later*; the agent must trace individual
   order lifecycles and revisit earlier decisions.

Key headline result: 8 LLMs × 2 agent frameworks × 48 runs of 365 days; the
best LLM config reaches only **27.3% of the mean final net assets** achieved
by human participants. Big gap.

## Formal setup (as extracted so far)

- Formulated as a finite-horizon **POMDP**: M = ⟨S, A, P, O, Z, R, µ0, Hc⟩.
- Simulator ticks **hourly** over 365 days → Hc = 8,760 steps.
- Agent gets a **decision window every 12 steps** (i.e., roughly twice a day);
  actions = sequences of merchant tool invocations (26 tools), null action
  otherwise.
- Latent state: clock, demand profiles, supplier conditions, listings &
  finances, active orders, pending events. Observation kernel exposes only
  merchant-visible info — demand profiles, risk params, and *future event
  times* stay hidden until effects materialize.
- At t = Hc new demand/activations stop but active orders run to terminal
  settlement at T ≥ Hc.
- **Reward:** zero intermediate reward; objective = expected terminal net
  assets J(π) = E[R(sT)], with R(sT) = B_T + D_T + I_T + Q_T
  (presumably cash Balance + Deposit + Inventory + something — need to check
  exact definitions; Q might be quality/rating-linked value).

## Why this is interesting to implement

- It's a benchmark/systems paper, so the "method" to re-implement is the
  **environment**: order-level simulation with coupled cash constraints,
  delayed outcomes, nonstationary demand, and a tool interface.
- Nice toy version: a mini e-commerce POMDP with ~50–200 synthetic products,
  a handful of tools (search_products, place_purchase_order, list_item,
  set_price, get_orders, get_cash, ...), delayed order lifecycle
  (placed → shipped → delivered → settled/abnormal), and a simple
  heuristic/scripted "merchant policy" to verify the environment is solvable
  and coherent.

## Terms / things I don't fully understand yet (second-pass focus)

- Exact decomposition of R(sT) = B_T + D_T + I_T + Q_T — what are Q and I
  precisely (inventory value at what valuation? quality score?).
- How demand is generated from the real product records (demand profiles —
  intensity, seasonality, price elasticity model?).
- The 26 tools — full list and signatures; how pricing affects demand
  quantitatively.
- Upstream supplier events: what kinds (stockouts, price changes, delivery
  delays?), their stochastic model.
- Store rating/reputation dynamics: how abnormal orders penalize rating and
  how rating feeds back into demand.
- The two agent frameworks compared (ReAct-style vs. something else?).
- Metrics beyond final net assets: what "decision trace" analyses they run
  (their Long-Term Coherence diagnostics).
- Human participant protocol: how humans played the same simulation.

## Notes on scope for our re-implementation

Per AGENTS.md, we re-implement the *core method*. Here the core artifact is
the simulation + evaluation loop. Goal: a minimal, runnable MerchantBench-like
environment (synthetic catalog, delayed order lifecycle, cash/deposit
mechanics, a few tools) with a scripted baseline merchant that survives 365
days and outputs a final net-assets number. No LLM calls needed for the toy
version (a rule-based agent stands in), but the tool interface should be
LLM-callable in principle.
