"""MerchantBench simulator core — a from-scratch re-implementation of the
365-day hourly POMDP environment from:

  "MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in
   E-Commerce Operations", Shi et al., arXiv:2607.28956 (2026).

Environment pieces (see breakdown.md, Sections 3-4):

  1. Demand intensity
       lambda_{m,i,t} = D_i(d) * w_{c,h} * r_m * l_{m,i,t} * (p/p_ref)^(-eps)
       orders ~ Poisson(lambda)
  2. Listing exposure l(a): cold-start ramp then exponential decay
       (l0=0.2, T_r=14d, kappa=0.0092, l_min=0.10)
  3. Store rating: exponentially-decayed weighted mean of per-order
     experience scores (R0=4.0, alpha=20, gamma=2^(-1/30))
  4. Order lifecycle: Placed -> Procured -> Shipped -> Delivered -> Settled
     (or abnormal terminal), outcome presampled at placement; drop-shipping
     procurement debits cash immediately; revenue arrives at settlement.
  5. Cash / deposit / fines: stockout 5, late 3, return+refund 8,
     bad-review 5 RMB; deposit exhausted + insolvency => store closes.
  6. Upstream supplier events: price change x[0.9,1.5] band, product
     delisting, shipment delay +12-96h; sampled per hour; delayed recovery.

The agent acts every 12 hours (730 decision windows). Reward is paid ONLY
at the end: R = B_T + D_T + I_T + Q_T (net assets). Pure delayed reward.

Pure standard library. Deterministic given (env_seed, agent_seed).
"""

import math
import random

from data import (make_catalog, daily_demand, rating_multiplier,
                  SCORE_U, SCORE_V)

HOURS = 365 * 24                 # 8760
DECISION_INTERVAL = 12           # agent acts every 12h
N_WINDOWS = HOURS // DECISION_INTERVAL   # 730
MAX_LISTINGS = 50

INIT_CASH = 2000.0               # RMB
INIT_DEPOSIT = 1000.0            # RMB

# Listing exposure constants
EXPOSURE_L0 = 0.2
RAMP_DAYS = 14
EXPOSURE_KAPPA = 0.0092
EXPOSURE_LMIN = 0.10

# Rating model
RATING_R0 = 4.0
RATING_ALPHA = 20.0
RATING_GAMMA = 2.0 ** (-1.0 / 30.0)

# Fines (RMB)
FINE_STOCKOUT = 5.0
FINE_LATE = 3.0
FINE_RETURN_REFUND = 8.0
FINE_BAD_REVIEW = 5.0

# Lifecycle constants
PROMISED_DISPATCH_H = 48
MAX_SETTLE_LAG_H = 168           # delivery -> settlement <= 168h
MAX_OUTCOME_LAG_H = 168          # outcome realization lag
DELIVERY_H = 48                  # shipped -> delivered (logistics window)

NORMAL, CANCEL, RETURNLESS, RETURN_REFUND, BAD_REVIEW = (
    "normal", "cancel", "returnless", "return_refund", "bad_review")


class SupplierState:
    """Per-product supplier condition (merchant observes partially)."""
    __slots__ = ("price_mult", "delisted_until", "delay_extra_h", "stock",
                 "capacity", "replen")

    def __init__(self, prod):
        self.price_mult = 1.0
        self.delisted_until = -1
        self.delay_extra_h = 0
        self.stock = prod["sup_init_stock"]
        self.capacity = prod["sup_capacity"]
        self.replen = prod["sup_replen"]

    def visible(self, hour):
        return {
            "price_mult": round(self.price_mult, 2),
            "delisted": self.delisted_until > hour,
            "delay_extra_h": self.delay_extra_h,
        }


class Listing:
    __slots__ = ("pid", "price", "listed_hour")

    def __init__(self, pid, price, listed_hour):
        self.pid = pid
        self.price = price
        self.listed_hour = listed_hour


class Order:
    __slots__ = ("oid", "pid", "placed_h", "outcome", "procurement_cost",
                 "revenue", "dispatch_h", "settle_h", "outcome_h", "state",
                 "settled")

    def __init__(self, oid, pid, placed_h, outcome, procurement_cost,
                 revenue, dispatch_h, settle_h, outcome_h):
        self.oid = oid
        self.pid = pid
        self.placed_h = placed_h
        self.outcome = outcome
        self.procurement_cost = procurement_cost
        self.revenue = revenue
        self.dispatch_h = dispatch_h
        self.settle_h = settle_h
        self.outcome_h = outcome_h
        self.state = "in_flight"
        self.settled = False


def exposure(age_h):
    """Listing exposure l(a): ramp then decay (age in hours)."""
    a = age_h / 24.0
    if a <= RAMP_DAYS:
        return EXPOSURE_L0 + (1.0 - EXPOSURE_L0) * (a / RAMP_DAYS)
    return EXPOSURE_LMIN + (1.0 - EXPOSURE_LMIN) * math.exp(
        -EXPOSURE_KAPPA * (a - RAMP_DAYS))


def _poisson(rng, lam):
    """Knuth's method for small lambda; normal approximation for large."""
    if lam <= 30:
        L = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            p *= rng.random()
            if p <= L:
                return k
            k += 1
    return max(0, int(rng.gauss(lam, math.sqrt(lam)) + 0.5))


class MerchantBenchEnv:
    """The environment. The agent interface is the act_* tool methods."""

    def __init__(self, catalog=None, seed=42):
        self.rng = random.Random(seed)
        self.catalog = catalog or make_catalog(seed=7)
        self.by_id = {p["id"]: p for p in self.catalog}
        self.hour = 0
        self.day = 0
        self.cash = INIT_CASH
        self.deposit = INIT_DEPOSIT
        self.closed = False
        self.close_reason = None
        self.listings = {}                      # pid -> Listing
        self.supply = {p["id"]: SupplierState(p) for p in self.catalog}
        # rating bookkeeping (incremental decayed sums)
        self._rating_day = 0
        self._rating_num = RATING_R0 * RATING_ALPHA
        self._rating_den = RATING_ALPHA
        self.rating = RATING_R0
        self.active_orders = []
        self.finished_orders = []
        self._oid = 0
        # metrics
        self.orders_placed = 0
        self.fines_total = 0.0
        self.gmv = 0.0
        self.revenue_settled = 0.0
        self.procurement_spent = 0.0
        self.stockouts = 0
        self.late_dispatches = 0
        # observation buffers
        self.env_events = []
        self.recent_outcomes = []
        self.tool_calls_this_window = 0

    # ------------------------------------------------------------------ #
    # rating                                                              #
    # ------------------------------------------------------------------ #
    def _rating_decay_to(self, day):
        while self._rating_day < day:
            self._rating_num *= RATING_GAMMA
            self._rating_den *= RATING_GAMMA
            self._rating_day += 1

    def _apply_rating_evidence(self, kind, hour):
        day = hour // 24
        self._rating_decay_to(day)
        u, v = SCORE_U[kind], SCORE_V[kind]
        self._rating_num += v * u
        self._rating_den += v
        self.rating = self._rating_num / self._rating_den

    # ------------------------------------------------------------------ #
    # agent tools                                                         #
    # ------------------------------------------------------------------ #
    def act_list(self, pid, price):
        if self.closed:
            return "fail: store closed"
        if pid not in self.by_id:
            return "fail: unknown product"
        if len(self.listings) >= MAX_LISTINGS:
            return f"fail: {MAX_LISTINGS} listing slots full"
        if pid in self.listings:
            return "fail: already listed"
        if self.supply[pid].delisted_until > self.hour:
            return "fail: supplier currently delisted"
        price = max(1.0, float(price))
        self.listings[pid] = Listing(pid, price, self.hour)
        self.tool_calls_this_window += 1
        return f"ok: listed {self.by_id[pid]['name']} @ {price:.2f}"

    def act_delist(self, pid):
        if pid in self.listings:
            del self.listings[pid]
            self.tool_calls_this_window += 1
            return "ok: delisted"
        return "fail: not listed"

    def act_reprice(self, pid, price):
        if pid not in self.listings:
            return "fail: not listed"
        self.listings[pid].price = max(1.0, float(price))
        self.tool_calls_this_window += 1
        return f"ok: repriced to {price:.2f}"

    def in_transit(self):
        """I_t: procurement funds tied up in unfinished orders."""
        return sum(o.procurement_cost for o in self.active_orders)

    def receivables(self):
        """Q_t: expected revenue from unfinished non-cancelled orders."""
        return sum(o.revenue for o in self.active_orders
                   if o.outcome != CANCEL)

    def act_query_finance(self):
        self.tool_calls_this_window += 1
        return (f"cash={self.cash:.2f} deposit={self.deposit:.2f} "
                f"in_transit={self.in_transit():.2f} "
                f"receivables={self.receivables():.2f} "
                f"net_assets={self.net_assets():.2f}")

    def act_query_orders(self):
        self.tool_calls_this_window += 1
        lines = [f"#{o.oid} {self.by_id[o.pid]['name']} "
                 f"placed_h={o.placed_h} state={o.state}"
                 for o in self.active_orders[-30:]]
        return "\n".join(lines) if lines else "no active orders"

    def act_query_supply(self, pid):
        self.tool_calls_this_window += 1
        if pid not in self.supply:
            return "fail: unknown product"
        s = self.supply[pid]
        return (f"supplier({pid}): price_mult={s.price_mult:.2f} "
                f"delisted={s.delisted_until > self.hour} "
                f"delay_extra_h={s.delay_extra_h}")

    def act_memory_write(self, text):
        """Stub for the LLM memory doc; rule agents use it too."""
        self.tool_calls_this_window += 1
        self.memory_doc = getattr(self, "memory_doc", "") + text + "\n"
        return "ok: memory updated"

    TOOLS = ("list", "delist", "reprice", "query_finance", "query_orders",
             "query_supply", "memory_write")

    # ------------------------------------------------------------------ #
    # simulation                                                          #
    # ------------------------------------------------------------------ #
    def _supplier_price(self, pid):
        return self.by_id[pid]["sup_price0"] * self.supply[pid].price_mult

    def _sample_outcome(self, prod):
        r = self.rng.random()
        acc = 0.0
        for k in (NORMAL, CANCEL, RETURNLESS, RETURN_REFUND, BAD_REVIEW):
            acc += prod["risk"][k]
            if r < acc:
                return k
        return NORMAL

    def _fine(self, amount, hour):
        """Fines draw from the security deposit; overflow hits cash."""
        self.fines_total += amount
        if self.deposit >= amount:
            self.deposit -= amount
        else:
            remain = amount - self.deposit
            self.deposit = 0.0
            self.cash -= remain
            if self.cash < 0:
                self.cash = 0.0
                self.closed = True
                self.close_reason = "insolvent after fines"

    def _tick_orders(self, h):
        """Fire due settlements (money) and outcome realizations (rating).

        The two clocks are separate, as in the paper: settlement moves money
        at settle_h; the outcome (review/refund/late verdict) realizes at
        outcome_h <= 168h later and only then feeds the store rating. The
        order stays alive until both have fired.
        """
        still = []
        for o in self.active_orders:
            if h >= o.settle_h and not o.settled:
                self._settle_money(o)
                o.settled = True
                o.state = "settled_awaiting_outcome"
            if h >= o.outcome_h:
                self._realize_outcome(o, h)
                self.finished_orders.append(o)
                continue
            still.append(o)
        self.active_orders = still

    def _settle_money(self, o):
        """Money flows at settlement (outcome was presampled at placement)."""
        if o.outcome == CANCEL:
            pass                                # no money moves
        elif o.outcome in (NORMAL, RETURNLESS, BAD_REVIEW):
            self.cash += o.revenue
            self.revenue_settled += o.revenue
        elif o.outcome == RETURN_REFUND:
            pass  # refunded to customer at outcome realization

    def _realize_outcome(self, o, h):
        """Delayed outcome realization: rating evidence, refunds, fines."""
        outcome = o.outcome
        late = (o.dispatch_h - o.placed_h) > PROMISED_DISPATCH_H
        if outcome == CANCEL:
            # customer cancelled: no money moved, no service experience
            self.recent_outcomes.append((h, o.oid, CANCEL, 0.0))
            return
        kind = "late" if (outcome in (NORMAL, RETURNLESS) and late) else outcome
        self._apply_rating_evidence(kind, h)
        if outcome == BAD_REVIEW:
            self._fine(FINE_BAD_REVIEW, h)
        elif outcome == RETURN_REFUND:
            self._fine(FINE_RETURN_REFUND, h)
        elif outcome == NORMAL and late:
            self.late_dispatches += 1
            self._fine(FINE_LATE, h)
        # delayed feedback the agent must trace back to its sourcing choice
        self.recent_outcomes.append(
            (h, o.oid, kind, round(o.revenue, 2)))
        if len(self.recent_outcomes) > 64:
            self.recent_outcomes = self.recent_outcomes[-64:]

    def _hourly_supplier_events(self, h):
        """3 upstream events per hour: price change / delisting / delay."""
        # recover old events
        for pid, s in self.supply.items():
            if s.delisted_until != -1 and s.delisted_until <= h:
                s.delisted_until = -1
            if s.delay_extra_h and self.rng.random() < 0.01:
                s.delay_extra_h = 0
        for _ in range(3):
            pid = self.catalog[self.rng.randrange(len(self.catalog))]["id"]
            prod = self.by_id[pid]
            r = self.rng.random()
            if r < prod["p_price_event"]:
                old = self.supply[pid].price_mult
                new = min(1.5, max(0.9, old * self.rng.uniform(0.90, 1.15)))
                self.supply[pid].price_mult = new
                self.env_events.append(
                    {"h": h, "pid": pid, "type": "price_change",
                     "detail": f"{old:.2f}->{new:.2f}"})
            elif r < prod["p_price_event"] + prod["p_delist_event"]:
                until = h + self.rng.randint(24 * 3, 24 * 14)
                self.supply[pid].delisted_until = max(
                    until, self.supply[pid].delisted_until)
                self.env_events.append(
                    {"h": h, "pid": pid, "type": "delisting",
                     "detail": f"until_h={until}"})
            elif r < (prod["p_price_event"] + prod["p_delist_event"]
                      + prod["p_delay_event"]):
                extra = self.rng.randint(12, 96)
                self.supply[pid].delay_extra_h = extra
                self.env_events.append(
                    {"h": h, "pid": pid, "type": "delay",
                     "detail": f"+{extra}h"})
        if len(self.env_events) > 64:
            self.env_events = self.env_events[-64:]

    def _replenish(self):
        for s in self.supply.values():
            if s.stock < s.capacity:
                s.stock = min(s.capacity, s.stock + s.replen)

    def _demand_and_orders(self, h):
        day_of_year = (h // 24) % 365
        hour_of_day = h % 24
        r_mult = rating_multiplier(self.rating)
        for pid, lst in list(self.listings.items()):
            prod = self.by_id[pid]
            D = daily_demand(prod, day_of_year)
            w = prod["hour_profile"][hour_of_day]
            l = exposure(h - lst.listed_hour)
            ratio = lst.price / prod["p_ref"]
            lam = D * w * r_mult * l * (ratio ** -prod["eps"])
            if lam <= 0:
                continue
            n = _poisson(self.rng, lam)
            for _ in range(n):
                self._place_order(pid, h)

    def _place_order(self, pid, h):
        prod = self.by_id[pid]
        sup = self.supply[pid]
        # immediate drop-ship procurement
        if sup.delisted_until > h or sup.stock <= 0:
            self.stockouts += 1
            self._apply_rating_evidence("stockout", h)
            self._fine(FINE_STOCKOUT, h)
            return
        cost = self._supplier_price(pid)
        if self.cash < cost:
            return                          # cannot procure; order lost
        self.cash -= cost
        self.procurement_spent += cost
        sup.stock -= 1
        outcome = self._sample_outcome(prod)
        revenue = self.listings[pid].price
        dispatch_lag = prod["sup_dispatch_h"] + sup.delay_extra_h
        dispatch_h = h + dispatch_lag
        settle_h = h + dispatch_lag + DELIVERY_H + self.rng.randint(
            1, MAX_SETTLE_LAG_H)
        outcome_h = settle_h + self.rng.randint(1, MAX_OUTCOME_LAG_H)
        self._oid += 1
        o = Order(self._oid, pid, h, outcome, cost, revenue, dispatch_h,
                  settle_h, outcome_h)
        self.active_orders.append(o)
        self.orders_placed += 1
        self.gmv += revenue

    def step_hours(self, n_hours):
        """Advance n_hours (the agent is silent between decision windows)."""
        for _ in range(n_hours):
            if self.closed or self.hour >= HOURS:
                return
            h = self.hour
            self._tick_orders(h)
            if self.closed:
                return
            self._hourly_supplier_events(h)
            self._replenish()
            self._demand_and_orders(h)
            self.hour += 1
            if self.hour % 24 == 0:
                self.day += 1

    def run_to_settlement(self, max_extra_hours=2 * MAX_OUTCOME_LAG_H):
        """After HOURS, demand stops; active orders run to terminal state."""
        limit = HOURS + max_extra_hours
        while self.active_orders and self.hour < limit and not self.closed:
            h = self.hour
            self._tick_orders(h)
            self.hour += 1
            if self.hour % 24 == 0:
                self.day += 1

    def net_assets(self):
        return self.cash + self.deposit + self.in_transit() + self.receivables()

    def observation(self):
        """Merchant-visible state at a decision window."""
        return {
            "hour": self.hour,
            "day": self.day,
            "cash": round(self.cash, 2),
            "deposit": round(self.deposit, 2),
            "net_assets": round(self.net_assets(), 2),
            "rating": round(self.rating, 3),
            "n_listings": len(self.listings),
            "n_active_orders": len(self.active_orders),
            "supplier_events": [
                {"h": e["h"], "pid": e["pid"], "type": e["type"],
                 "detail": e["detail"]} for e in self.env_events[-8:]],
            "recent_outcomes": self.recent_outcomes[-8:],
        }
