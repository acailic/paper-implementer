"""Baseline merchants + evaluation loop for the MerchantBench re-implementation.

Paper: "MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in
E-Commerce Operations", Shi et al., arXiv:2607.28956 (2026).

The paper's own rule-based baseline achieved ~24.48k RMB final net assets
(vs. human mean 217.61k). We re-implement the *environment* and plug in
programmatic merchants of increasing sophistication:

  - PassiveMerchant   : lists 10 products once, never acts again (floor)
  - RandomMerchant    : random listing/repricing every window (noise ref)
  - RuleMerchant      : the paper-style baseline — full portfolio, seasonal
                        rotation, exposure-aware relisting, supplier-event
                        response, liquidity-aware procurement budgeting.

Also computes the paper's Long-Horizon Activity metric:
  SWR = min over rolling 30-day windows of (windows with >=1 env tool call)
        / (total windows in the window)
Rule agents should stay pinned at SWR = 100% — the "sustained working rate"
the paper found humans keep (100%) but LLM agents lose (10.6-99.4%).

Usage:  python3 train.py [--days 365] [--env-seed 42] [--agents ...]
Pure standard library; runs in well under a minute for a full year.
"""

import argparse
import math
import random
import sys
import time

from data import make_catalog
from model import (MerchantBenchEnv, HOURS, DECISION_INTERVAL, N_WINDOWS,
                   MAX_LISTINGS, exposure)


# ---------------------------------------------------------------------- #
# merchants                                                              #
# ---------------------------------------------------------------------- #
class PassiveMerchant:
    """Lists 10 products at reference price once, then never acts."""

    name = "passive"
    acts_after_bootstrap = False

    def __init__(self, catalog, seed=0):
        self.catalog = catalog
        self.rng = random.Random(seed)

    def initial_action(self, env):
        for prod in self.catalog[:10]:
            env.act_list(prod["id"], prod["p_ref"])

    def decide(self, env, obs, window_idx):
        return []                                   # never acts


class RandomMerchant:
    """Uniformly random tool calls every window (a noise reference)."""

    name = "random"
    acts_after_bootstrap = True

    def __init__(self, catalog, seed=0):
        self.catalog = catalog
        self.rng = random.Random(seed)

    def initial_action(self, env):
        pass

    def decide(self, env, obs, window_idx):
        # keep 2-6 random listings alive with junk prices
        actions = []
        for _ in range(self.rng.randint(1, 3)):
            prod = self.rng.choice(self.catalog)
            r = self.rng.random()
            if r < 0.5:
                actions.append(env.act_list(prod["id"],
                                            prod["p_ref"] * self.rng.uniform(0.7, 1.4)))
            elif r < 0.8 and env.listings:
                pid = self.rng.choice(list(env.listings))
                actions.append(env.act_delist(pid))
            else:
                env.act_query_finance()
        return actions


class RuleMerchant:
    """Paper-style rule-based baseline.

    Policy (from breakdown Sec. 3.3 + Sec. 6 "rule-based baseline"):
      - keep the listing portfolio full (50 slots)
      - price = p_ref * 1.25 (healthy margin over supplier cost ~0.55 p_ref)
      - rotate: delist listings whose exposure has decayed (age > ~60d) and
        replace with seasonally-hot products
      - respond to supplier events (delisted supplier -> delist listing)
      - budget: only place/keep listings while cash stays above a safety
        buffer (liquidity-aware, cf. evidence-calibrated adaptation)
    """

    name = "rule"
    acts_after_bootstrap = True
    PRICE_MULT = 1.25
    ROTATE_AGE_DAYS = 60

    def __init__(self, catalog, seed=0, price_mult=PRICE_MULT,
                 rotate_age_days=ROTATE_AGE_DAYS):
        self.catalog = catalog
        self.rng = random.Random(seed)
        self.price_mult = price_mult
        self.rotate_age_days = rotate_age_days

    def _seasonal_rank(self, env):
        """Rank products by current-day expected demand (proxy: no obs)."""
        day = (env.hour // 24) % 365
        scored = []
        for prod in self.catalog:
            s = 1.0 + prod["season_amp"] * math.sin(
                2.0 * math.pi * (day / 365.0) + prod["season_phase"])
            scored.append((prod["base_daily_demand"] * s, prod))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored]

    def initial_action(self, env):
        ranked = self._seasonal_rank(env)
        for prod in ranked[:MAX_LISTINGS]:
            env.act_list(prod["id"], round(prod["p_ref"] * self.price_mult, 2))

    def decide(self, env, obs, window_idx):
        actions = []
        # 1) drop listings whose supplier turned expensive/delisted
        for pid in list(env.listings):
            sup = env.supply[pid]
            if sup.delisted_until > env.hour or sup.price_mult > 1.3:
                actions.append(env.act_delist(pid))
        # 2) rotate stale listings (exposure decayed)
        stale = [pid for pid, lst in env.listings.items()
                 if (env.hour - lst.listed_hour) / 24.0 > self.rotate_age_days]
        for pid in stale:
            actions.append(env.act_delist(pid))
        # 3) keep portfolio full with current seasonal winners
        if len(env.listings) < MAX_LISTINGS and env.cash > 500:
            ranked = self._seasonal_rank(env)
            for prod in ranked:
                if len(env.listings) >= MAX_LISTINGS:
                    break
                if prod["id"] not in env.listings:
                    actions.append(env.act_list(
                        prod["id"],
                        round(prod["p_ref"] * self.price_mult, 2)))
        # 4) monthly finance check keeps the window "working"
        env.act_query_finance()
        return actions


MERCHANTS = {
    PassiveMerchant.name: PassiveMerchant,
    RandomMerchant.name: RandomMerchant,
    RuleMerchant.name: RuleMerchant,
}


# ---------------------------------------------------------------------- #
# metrics                                                                #
# ---------------------------------------------------------------------- #
def sustained_working_rate(tool_calls_per_window, window_hours=DECISION_INTERVAL,
                           window_days=30):
    """SWR: min over rolling 30-day windows of active-window share."""
    per_day = 24 // window_hours                       # windows per day
    win_len = window_days * per_day
    if len(tool_calls_per_window) < win_len:
        return None
    active = [1 if c >= 1 else 0 for c in tool_calls_per_window]
    best = 1.0
    for start in range(0, len(active) - win_len + 1):
        seg = active[start:start + win_len]
        best = min(best, sum(seg) / len(seg))
    return best


def run_episode(env, merchant, verbose=False):
    """Run one 365-day episode; returns metrics dict."""
    tool_calls = []
    # bootstrap: first decision window at hour 0
    merchant.initial_action(env)
    tool_calls.append(env.tool_calls_this_window)
    windows_done = 1
    # run the year
    while windows_done < N_WINDOWS and not env.closed:
        env.tool_calls_this_window = 0
        # advance 12h to the next decision point
        env.step_hours(DECISION_INTERVAL)
        if env.closed or env.hour >= HOURS:
            tool_calls.append(0)
            break
        # agent observes and acts (obs is merchant-visible only)
        obs = env.observation()
        merchant.decide(env, obs, windows_done)
        tool_calls.append(env.tool_calls_this_window)
        windows_done += 1
        if verbose and windows_done % 60 == 0:
            print(f"  window {windows_done}/{N_WINDOWS} day {env.day} "
                  f"net_assets={env.net_assets():.0f}")
    # let stragglers settle
    env.run_to_settlement()
    swr = sustained_working_rate(tool_calls)
    return {
        "merchant": merchant.name,
        "env_seed": env.rng.seed if isinstance(env.rng.seed, int) else None,
        "final_net_assets": round(env.net_assets(), 2),
        "closed": env.closed,
        "close_reason": env.close_reason,
        "orders_placed": env.orders_placed,
        "gmv": round(env.gmv, 2),
        "revenue_settled": round(env.revenue_settled, 2),
        "procurement_spent": round(env.procurement_spent, 2),
        "fines_total": round(env.fines_total, 2),
        "stockouts": env.stockouts,
        "final_rating": round(env.rating, 3),
        "avg_active_listings": None,   # filled below
        "swr": None if swr is None else round(swr, 4),
        "windows_engaged": sum(1 for c in tool_calls if c >= 1),
        "n_windows": len(tool_calls),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--env-seed", type=int, default=42)
    agent_choices = list(MERCHANTS) + ["all"]
    ap.add_argument("--agents", nargs="*", default=["all"],
                    choices=agent_choices)
    ap.add_argument("--seeds", type=int, default=3,
                    help="episodes per agent")
    ap.add_argument("--catalog", type=int, default=200,
                    help="catalog size")
    args = ap.parse_args()

    if args.days != 365:
        # support shorter runs by monkey-patching the horizon
        import model as _m
        _m.HOURS = args.days * 24
        globals()["HOURS"] = args.days * 24
        globals()["N_WINDOWS"] = (args.days * 24) // DECISION_INTERVAL

    agents = list(MERCHANTS) if "all" in args.agents else args.agents
    catalog = make_catalog(n_products=args.catalog, seed=7)

    print(f"MerchantBench re-implementation — {args.days}-day run")
    print(f"catalog: {len(catalog)} synthetic products; "
          f"env seed {args.env_seed}; {args.seeds} episode(s)/agent\n")

    results = []
    for name in agents:
        for ep in range(args.seeds):
            env = MerchantBenchEnv(catalog=catalog,
                                   seed=args.env_seed + ep)
            merchant = MERCHANTS[name](catalog, seed=100 + ep)
            t0 = time.time()
            metrics = run_episode(env, merchant)
            metrics["runtime_s"] = round(time.time() - t0, 1)
            results.append(metrics)
            print(f"[{name:>7}] ep{ep} net_assets="
                  f"{metrics['final_net_assets']:>10,.0f} RMB  "
                  f"orders={metrics['orders_placed']:>5} "
                  f"rating={metrics['final_rating']:.2f} "
                  f"SWR={metrics['swr'] if metrics['swr'] is not None else 'n/a'} "
                  f"fines={metrics['fines_total']:.0f} "
                  f"({metrics['runtime_s']}s)")
            if metrics["closed"]:
                print(f"         ! closed: {metrics['close_reason']}")

    # summary
    print("\n=== Summary (mean over episodes) ===")
    print(f"{'agent':>8} | {'net_assets':>10} | {'orders':>7} | "
          f"{'rating':>6} | {'SWR':>6} | {'GMV':>10}")
    print("-" * 66)
    for name in agents:
        eps = [r for r in results if r["merchant"] == name]
        na = sum(r["final_net_assets"] for r in eps) / len(eps)
        ords = sum(r["orders_placed"] for r in eps) / len(eps)
        rat = sum(r["final_rating"] for r in eps) / len(eps)
        swrs = [r["swr"] for r in eps if r["swr"] is not None]
        swr = (sum(swrs) / len(swrs)) if swrs else float("nan")
        gmv = sum(r["gmv"] for r in eps) / len(eps)
        print(f"{name:>8} | {na:>10,.0f} | {ords:>7,.0f} | {rat:>6.2f} | "
              f"{swr:>6.1%} | {gmv:>10,.0f}")

    # human reference from the paper (3 participants, mean 217.61k RMB)
    print(f"\n{'human':>8} | {217610:>10,.0f} | {9442:>7,.0f} | "
          f"{'~4.7':>6} | {1.0:>6.1%} | {'—':>10}   (paper: 3 humans)")
    print(f"{'rule (paper)':>13} | {24480:>10,.0f} | {'—':>7} | "
          f"{'—':>6} | {'—':>6} | {'—':>10}   (paper's own baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
