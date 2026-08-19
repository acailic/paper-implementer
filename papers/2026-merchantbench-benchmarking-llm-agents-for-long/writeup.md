# Writeup — MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations

> **Paper:** MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations
> **Authors:** Qiming Shi, Yulong Tao, Linbo Jin, et al. (Alibaba Group, ZJU, PKU, Fudan)
> **Year:** 2026 · **ArXiv:** https://arxiv.org/abs/2607.28956

This is my own explanation, written after reading and re-implementing the
paper. It's a synthesis, not a retelling of the abstract.

## The one-paragraph version

MerchantBench is not a new model — it's a ruler. The authors argue that we
have no way to measure **Long-Term Coherence (LTC)**: an agent's ability to
stay purposeful across hundreds of decision points while adapting to
evidence that arrives with *heterogeneous delays*. Their answer is a
365-day, hour-by-hour simulation of a wholesale store on 1688 (Alibaba's B2B
marketplace), grounded in 98,843 real product records, where an LLM "is the
merchant": every 12 hours it gets 26 tools to source products, manage 50
listing slots, and steward cash — and where *placing* an order commits cash
today while its refund/bad-review outcome surfaces days later. When you run
8 frontier LLMs in this world for a simulated year each, the best one ends
with 27.3% of the net assets three human players accumulate. The
fascinating part is *how* they fail: not by choosing bad products, but by
slowly going quiet — agents progressively stop acting, misremember the
endpoint date, or collapse into reactive supplier-event handling while
their portfolio rots.

## The problem

Agent benchmarks overwhelmingly test *bounded* tasks: answer a question,
complete a web purchase, navigate a room. Success is binary and immediate.
Real deployments are nothing like this. A deployed agent faces:

1. **Long horizons** — hundreds to thousands of decision points, with the
   reward only materializing at the end.
2. **Actions that constrain the future** — cash committed to procurement
   today is cash you can't spend tomorrow; a listing slot occupied by a
   dud product blocks a good one.
3. **Feedback with mixed latency** — supplier price changes are visible
   within the hour; whether a given order refunds, arrives late, or earns
   a 1-star review is only revealed days after the sourcing decision that
   caused it.
4. **Cumulative, measurable consequences** — incoherence compounds into
   the final balance sheet.

No existing benchmark combines all four. So a model that aces WebArena can
still drift into incoherence over a year of operation, and nobody would
notice until deployment.

## The idea

Use seller-side e-commerce as the LTC testbed, and make **net assets after
365 simulated days** the single, un-fakeable score. Formally it's a finite-
horizon POMDP (hourly ticks, 730 decision windows, zero intermediate
reward, J(π) = E[final net assets] = cash + deposit + in-transit +
receivables). The design choices that make it work:

- **Order-level simulation with drop-shipping.** Demand arrives as Poisson
  events per listing: λ = D·w·r·ℓ·(p/p_ref)^{−ε} — real daily demand ×
  hour-of-day profile × store-rating multiplier × listing-exposure ramp ×
  price elasticity. Each order immediately procures from the supplier
  (debiting cash), then walks a lifecycle: Placed → Procured → Shipped →
  Delivered → Settled, possibly branching into Cancellation, Refund, or
  Bad Review — with each order's latent outcome **presampled at birth**
  but only revealed at delivery.
- **Asymmetric feedback.** Upstream supplier events (price hikes,
  delistings, shipment delays) hit instantly; downstream order outcomes
  arrive days later and must be traced back to the responsible sourcing
  decision. The agent has to keep its own books.
- **One shared rating kills laziness.** Every abnormal order drags the
  store rating, and the rating multiplies *all* demand (0.10×–1.20× in
  star bands). One bad product family can starve the whole portfolio.
- **A death condition.** Fines draw from a security deposit; if the
  deposit hits zero the store closes permanently. Incoherence isn't just
  suboptimal — it can be terminal.

## How it works (the intuition)

Think of it as a treadmill with delayed pain. The agent earns margin
(≈21 RMB/order at paper scale) only if it keeps 50 slots filled with
seasonally-relevant products at prices the market accepts. The catalog's
demand is *nonstationary* (618 and 11.11 promotion peaks, Spring Festival
trough), so last month's winners decay and the agent must continually
re-source. Meanwhile every window it doesn't act is compounding: empty
slots earn nothing, expired deadlines cost fines, stale listings lose the
exposure ramp.

The paper's analytical payoff is a decomposition of LTC into:

- **Operational Coherence** — do you keep *acting*? Measured by the
  **Sustained Working Rate (SWR)**: the minimum, over all rolling 30-day
  windows, of the share of decision windows with ≥1 environment tool
  call. Humans: 100%. LLMs: 10.6–99.4%.
- **Strategic Coherence** — do you keep acting *toward the goal, updated
  by evidence*? Failure modes named and evidenced: *Control-Loop
  Narrowing* (the sourcing loop collapses into reactive supplier
  firefighting), *Premature Abandonment* (a Kimi K2.6 run concluded on
  Day 104 the store was unrecoverable and went silent for 355 of 523
  remaining windows), memory errors (a run misremembered Day 285 as the
  endpoint on Day 282 and stopped refilling slots), and flat procurement
  trajectories where humans escalate prices (43→91 RMB) as liquidity
  grows.

A neat diagnostic they add: the **Time-aware Sourcing Gain** — compare the
demand alignment of the agent's actual monthly listing mix against the
counterfactual of holding its January mix fixed all year. Positive gain =
genuine seasonal reallocation, and it correlates with final net assets.

## What I learned by implementing it

(Things that only became clear once I wrote the code — `implementation/`,
pure Python, no dependencies.)

1. **The exposure ramp ℓ is the hidden curriculum.** New listings start at
   20% exposure and ramp to full over 14 days, then decay (κ=0.0092,
   floor 0.10). This one constant forces long-horizon behavior: a
   myopic agent sees "new listing = weak sales" and delists, destroying
   the very asset that would have paid off. Half of the gap between my
   random merchant (65k) and rule merchant (205k) is just *leaving good
   listings alone*.
2. **Presampled outcomes make the economy honest.** Because each order's
   fate is fixed at birth, the environment can't accidentally reward
   luck; the only lever the agent has is *which risk profiles it lets in
   the door*. In my synthetic catalog, picking the wrong tail of the
   risk distribution costs ~40% of net assets via the rating channel —
   the coupling (bad orders → rating → all-listings demand) is much
   tighter than the paper's prose suggests.
3. **Liquidity is the real constraint early, rating later.** With 2,000
   RMB starting cash and instant procurement debits, the first month is a
   cash-flow puzzle (my rule merchant budgets listing adds against
   settleable receivables). By month three, cash is abundant and the
   binding constraint is rating repair. The problem *changes shape* over
   the horizon — that's something static benchmarks can't teach.
4. **SWR is trivially gameable and still diagnostic.** A clock that calls
   one tool per window scores SWR=100%. The metric only means something
   bundled with net assets — "were you *alive* AND solvent". The paper
   understands this; reading only the SWR table would mislead you.
5. **Calibrating to the paper's scale is finicky.** Humans averaged 9,442
   orders/year (~26/day) and 217.6k net assets. Getting a 200-product
   synthetic catalog to reproduce those magnitudes (≈21 RMB margin per
   order) took as much fiddling as the lifecycle FSM itself. The paper's
   per-order economics are doing a lot of quiet work.

## What surprised me / was harder than expected

- **How badly frontier models do relative to a dumb script.** The paper's
  own rule baseline hit 24.48k — and *most of the 48 LLM runs* barely
  beat it, with the best at 59.46k vs humans' 217.61k. My oracle-ish rule
  merchant on my synthetic economy hit 205k, i.e. ~94% of the human mean
  — so the gap isn't that the task is *hard* per se; it's that LLMs fail
  at *sustained, self-directed operation* even when each individual
  decision is easy.
- **The failure stories read like organizational pathologies**, not model
  limitations: premature abandonment of a recoverable store, cargo-cult
  delisting ("fewer listings concentrate traffic" — one Claude Opus 4.8
  run shrank from 47 to 3 listings on that theory), misremembered
  deadlines. These are *belief-maintenance* failures, which reframes LTC
  as partly a memory-systems problem.
- **Hermes-style frameworks beat ReAct on 7/8 models** (+53.3% avg net
  assets) — scaffolding that manages context and skills mattered more
  than the choice of model. For a benchmark paper, that's a strong
  systems lesson hiding in the appendix.
- The trickiest code was mundane: **settlement vs. outcome clocks** (money
  settles ≤168h after delivery, but the rating/fine consequence can fire
  on a *different* clock). Getting the two decoupled without
  double-counting fines took three attempts.

## References

- Paper: https://arxiv.org/abs/2607.28956
- Official code: https://github.com/KhanCold/merchantbench
- My implementation: `implementation/` (run: `python3 train.py`, ~2s,
  pure stdlib; results in `implementation/results.txt`)
- Breakdown: `breakdown.md` · Reading notes: `notes.md`
