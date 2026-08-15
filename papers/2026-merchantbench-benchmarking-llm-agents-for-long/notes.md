# MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations

**Paper:** Shi et al. (Alibaba Group, ZJU, PKU, Fudan), arXiv:2607.28956v2 [cs.AI], Aug 2026
**Code:** https://github.com/KhanCold/merchantbench

---

## Pass 1 — First impressions (what is this?)

A benchmark paper, not a new model. The authors argue that existing LLM-agent
benchmarks test *bounded* tasks with immediate success criteria, while real
deployments need **Long-Term Coherence (LTC)**: the ability to keep purposeful
behavior over long horizons while adapting decisions to *accumulated evidence*
that arrives with *heterogeneous delays*.

Their testbed: a simulated 365-day seller-side e-commerce store operation
("you are the merchant"). Four interdependent decision components:

1. **Product Sourcing** — pick products from a catalog of 98,843 real product
   records (from 1688, Alibaba's wholesale marketplace), deal with suppliers.
2. **Listing and Pricing Control** — list/delist/reprice; changes affect
   demand only from the next step.
3. **Cash-Flow Management** — new orders *commit* cash before settlement;
   unpaid fines draw from a security deposit; deposit exhausted ⇒ store dies.
4. **Mixed-Latency Feedback Adaptation** — order outcomes (abnormal
   deliveries, refunds, bad reviews) surface *later*; agent must trace
   individual order lifecycles and revisit earlier decisions.

Headline: 8 LLMs × 2 frameworks (ReAct, Hermes) × 3 runs = 48 runs of 365
days; best LLM config reaches only **27.3% of mean final net assets** of 3
human participants (Qwen3.7-Max/Hermes 59.46k RMB vs Human 217.61k RMB).

---

## Pass 2 — Deep read (section by section)

### Task formulation (Sec. "MerchantBench / Task Formulation")

Finite-horizon **POMDP** M = ⟨S, A, P, O, Z, R, µ0, Hc⟩:

- Simulator ticks **hourly** over 365 days → Hc = 8,760 steps, t ∈ {0..8759}.
- Agent gets a **decision window every 12 steps** (2×/day) → 730 windows/run.
  Action at = a *sequence of tool invocations*; null action otherwise.
- Latent state s_t: clock, product demand profiles, supplier conditions,
  listings & finances, active orders, pending events.
- Transition P: tool-induced store changes + autonomous demand/supplier/order
  evolution. Listing/pricing changes affect demand from the **next** step.
- Observation Z exposes only merchant-visible info; demand profiles, risk
  params, presampled outcomes, future event times stay hidden.
- At t = Hc new demand & activations stop; active orders run to terminal
  settlement at T ≥ Hc. Zero intermediate reward;
  **J(π) = E[R(s_T)], R(s_T) = B_T + D_T + I_T + Q_T**
  where B = cash balance, D = security deposit, I = funds in transit
  (in-transit procurement), Q = receivables. I.e. R = **net assets**.

Key asymmetry (Fig. 1): an order *commits* cash at placement (procurement
debit) but its outcome (normal/refund/bad review) is revealed only at/after
delivery — liquidity pressure is immediate, evidence is delayed.

### Real-data grounding

- Source: 1688 (largest Chinese wholesale marketplace). 10 first-level
  categories (apparel…womenswear). Data: product+supplier attributes,
  365-day product-level daily demand histories (Jun 1 2025–May 31 2026),
  platform quality/fulfillment signals, plus 365 date-aligned daily market
  reports. After filtering: **98,843 products / 36,576 suppliers**.
- Demand seasonality: peaks at 618 and 11.11 promotions, trough at Spring
  Festival → the opportunity set is *nonstationary*; promising products
  emerge and fade, forcing continual portfolio revision.

### Downstream (order-level) simulation — the core math

Hourly arrival intensity for product i listed by merchant m at step t:

  **λ_{m,i,t} = D_{i,d(t)} · w_{c(i),h(t)} · r_{m,t} · ℓ_{m,i,t} · (p_{m,i,t}/p_i^ref)^{−ε_i}**

- d(t) = ⌊t/24⌋ (data day), h(t) = t mod 24 (hour of day).
- D_{i,d}: real daily demand of product i (from data).
- w_{c,h}: intra-category hour-of-day profile distributing daily demand over hours.
- r_{m,t}: store-rating demand multiplier (see rating dynamics below).
- ℓ_{m,i,t}: **listing exposure** — cold-start ramp then decay:
  g(a) = ℓ0 + (1−ℓ0)·a/Tr for a ≤ Tr (linear ramp over Tr days),
  g(a) = ℓmin + (1−ℓmin)·e^{−κ(a−Tr)} for a > Tr (exponential decay).
  Config: ℓ0=0.2, Tr=14d, κ=0.0092, ℓmin=0.10.
- (p/p_ref)^{−ε_i}: **price elasticity** — selling above the reference price
  cuts demand at rate ε_i (product-specific).

Orders: N_{m,i,t} ~ Poisson(λ_{m,i,t}); each arrival = an order candidate.
At creation each candidate gets ONE **latent customer outcome** sampled from
its product-specific risk profile: Normal / Cancellation / Returnless Refund
/ Return+Refund / Bad Review. Stockout and Late Shipment instead arise
endogenously from procurement/fulfillment dynamics. Outcome + realization
time hidden until the lifecycle transition fires.

**Order lifecycle (single-item drop-shipping, no merchant inventory):**
Placed → (immediate procurement at current supplier price; success debits
cash, decrements supplier stock) → Procured → Shipped → Delivered (sale
price becomes a *receivable*) → Settled (after sampled delay, receivable
credited to cash). Abnormal terminals:
- **Cancellation**: restores procurement cost, no fine.
- **Stockout** (supplier delisted/qty=0 at procurement): prevents
  procurement, fine RMB 5.
- **Late Shipment** (missed dispatch deadline): continues to fulfillment,
  fine RMB 3.
- **Return+Refund**: removes receivable but restores procurement cost; fine RMB 8.
- **Returnless Refund**: removes receivable, procurement cost lost; no fine.
- **Bad Review**: keeps sales revenue; fine RMB 5.
All abnormal outcomes except Cancellation also feed the store rating.

### Upstream supplier simulation

Supply pool has time-varying procurement price, available inventory,
availability, shipment times; inventory replenishes hourly. Three
**Upstream Supplier Events** sampled per step with product-level
probabilities calibrated from real fulfillment signals:
- **Price Change**: procurement price × factor ∈ [0.9, 1.5].
- **Product Delisting**: procurement suspended.
- **Shipment Delay**: dispatch time +12–96 h.
Stockouts arise endogenously when orders deplete stock faster than
replenishment. Every abnormality gets a sampled **recovery time**
(168–672 h) after which attributes return to base. Agent sees realized
changes (price/availability/qty/ship time) via catalog/supplier queries but
never the abnormality flag, trigger probability, or recovery schedule.

Supplier config: initial inventory 20–399 units, capacity 50–499, hourly
replenishment 1–19 units; base dispatch 1–47 h, logistics 12–72 h.

### Store rating dynamics (daily update)

Each eligible order j (not Cancellation / insufficient-balance) contributes
experience score u_j and evidence weight v_j, ordered [normal settled, late
shipped, return+refund, returnless refund, bad review, stockout]:
u = [4.5, 3.0, 2.0, 1.5, 1.0, 1.0], v = [1, 1, 1, 2, 2, 3].

  **R_{m,d} = (α·R0 + Σ_{j∈C} γ^{d−1−d_j} v_j u_j) / (α + Σ_{j∈C} γ^{d−1−d_j} v_j)**

with prior R0 = 4.0, prior weight α = 20, decay γ = 2^{−1/30} (30-day
evidence half-life). Continuous rating → star bands: thresholds 2.50, 3.30,
3.80, 4.20 → demand multipliers 0.10 / 0.35 / 0.80 / 1.00 / 1.20. So one
bad-review cluster can cut *all* demand by 20–90% — product failures
propagate to the whole portfolio through r_{m,t}.

### Agent interface

- Shared observation protocol + **26 tools** (5 sourcing reads, 2+3 listing
  writes/reads, 4 finance reads, 5 supplier/order monitoring, 4 agent
  support: memory doc read/write, get_observation, list_tools, end_of_step).
  Full inventory in appendix Table 2.
- Store config: cash RMB 2,000, deposit RMB 1,000, ≤ 50 active listings.
  Fines deduct balance first, then deposit; balance 0 doesn't kill the
  store, **deposit 0 closes it permanently**. Cash credits first restore
  the deposit to 1,000 before entering balance.
- Order lifecycle config: promised shipment 48 h; Delivered→Settled and
  outcome realization each ≤ 168 h.
- Context management: ReAct history truncated to 30k tokens past 160k with
  a reminder to summarize into memory; Hermes uses its default
  summarization; each model is its own summarizer.

### Experiments & results

- 8 LLMs (GPT-5.6 Sol, Claude Opus 4.8, Qwen3.7-Max/Plus, GLM-5.2,
  DeepSeek-V4-Pro/Flash, Kimi K2.6) × 2 frameworks × 3 runs.
- Best per framework: ReAct GPT-5.6 Sol (40.89k net assets); Hermes
  Qwen3.7-Max (59.46k). Human mean 217.61k → best LLM = 27.3% of human.
  Rule-based baseline: 24.48k — most LLM configs barely beat a simple
  rule script; Hermes beats ReAct on 7/8 models (avg +53.3% net assets).
- Metrics: Business (Net Assets, GMV, margin, orders), Store Reliability
  (fines, rating, anomaly rate), Long-Horizon Activity (avg active
  listings, **SWR** = min share of decision windows with ≥1 env tool call
  over all rolling 30-day periods, tool calls).
- Humans: SWR 100%, 9,442 orders, avg 49.1 active listings; LLMs SWR
  10.6–99.4%.

### Long-Term Coherence analysis (the paper's key conceptual contribution)

Two failure modes:
1. **Operational Coherence loss** — activity decay: agents progressively
   stop intervening (SWR collapses over the year; e.g. ReAct Qwen3.7-Max
   quarterly effective-window rate 68%→23%). Extreme: *Premature
   Abandonment* — Hermes Kimi K2.6 decided on Day 104 the store couldn't
   recover and took no action in 355 of the remaining 523 windows.
2. **Strategic Coherence loss** — two sub-dimensions:
   - **Goal Consistency**: Control-Loop Narrowing (sourcing loop collapses
     into reactive supplier-event handling; supply checks rose 14%→34% of
     remaining calls) or Premature Abandonment; also memory errors (a
     Qwen3.7-Max run misremembered Day 285 as the endpoint on Day 282 and
     stopped refilling slots for 83 remaining days).
   - **Evidence-Calibrated Adaptation**: humans raise listing procurement
     prices over the year (43–53 → 59–91 RMB) as liquidity grows; weak
     agents keep flat trajectories. Humans do full *product attribution →
     risk removal → demand-preserving replacement* chains; some agents do
     none. One Claude Opus 4.8 run falsely inferred that removing weak
     listings concentrates traffic and shriveled from 47 to 3 listings.
- **Time-aware Sourcing Gain** G_r = Σ_m H_{r,m}(S_{r,m} − B_{r,m}) / Σ_m H_{r,m}
  compares monthly demand alignment S (listing-hour-weighted catalog demand
  percentile) against a no-reallocation counterfactual B (same annual mix
  held fixed). Positive G ⇒ active seasonal reallocation; correlates with
  final net assets.
- Hermes case study: code use amplifies quantitative models (GPT-5.6 Sol
  scripts a launch portfolio with p = max(1.70c, c+6) rounded up to .9
  endings); 17/24 runs auto-create a "RealShop" operating skill; skill
  quality varies from evidence-cited rule revision to unstructured dumps.

### What remains fuzzy / to verify from the repo when coding

- Exact functional form of w_{c,h} hour-of-day profiles (we'll approximate).
- How p_i^ref is set (platform reference price — presumably the catalog
  list price; we'll treat catalog price as ref).
- Daily market-report generation (LLM-written from real 1688 reports; our
  toy will generate synthetic keyword-trend reports).
- Fine details of "in_transit" accounting between procurement and delivery
  (per task prompt: debit at procure → in_transit; at delivery cost leaves
  in_transit, sale_price enters receivable — clear enough).

## Scope of our re-implementation (decided)

Core artifact = the **environment**: mini MerchantBench-like POMDP with
synthetic catalog (~100–200 products, seasonal demand), drop-shipping order
lifecycle with presampled latent outcomes, Poisson demand with exposure ×
rating × elasticity factors, cash/deposit/fine mechanics, upstream events
with recovery, ~8–12 of the 26 tools, and a rule-based merchant (the
paper's own baseline logic: daily checks, delist 7-day-no-sale or
supplier-affected products, refill slots from the daily report). Success
criterion: runs 365 days, prints net-assets curve and final R(s_T) > 0.
