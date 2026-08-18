"""Synthetic catalog for the MerchantBench re-implementation.

The real benchmark grounds demand in 98,843 product records from the 1688
wholesale platform (not redistributable); per Section 8 of our breakdown we
instead generate a *synthetic seasonal catalog* that exposes exactly the same
statistical structure the simulator needs:

  - per-product daily demand D_i(d) with category-level seasonality
  - per-category hour-of-day demand profile w_{c,h}
  - reference price p_ref and price elasticity eps_i
  - per-product outcome risk profile (Normal / Cancel / Returnless /
    Return+Refund / BadReview)
  - per-product supplier parameters (inventory, capacity, replenishment,
    dispatch base time, logistics time)

Paper: "MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in
E-Commerce Operations", Shi et al., arXiv:2607.28956 (2026).
Pure standard library — no external dependencies.
"""

import math
import random

CATEGORIES = [
    # (name, seasonal phase, seasonal amplitude, peak hour)
    ("apparel", 0.0, 0.5, 20),        # peaks in evening
    ("electronics", 2.0, 0.3, 14),    # afternoon peak, mild season
    ("home_garden", 3.0, 0.6, 10),    # spring-ish seasonality
    ("sports_outdoor", 1.5, 0.7, 8),  # summer peak, morning buyers
    ("beauty", 0.5, 0.25, 21),
    ("auto_parts", 4.0, 0.35, 12),
]


def _hour_profile(peak_hour, sharpness=4.0):
    """Hour-of-day profile w_{c,h}: a smooth bump over 24 hours, normalized."""
    raw = []
    for h in range(24):
        dh = min(abs(h - peak_hour), 24 - abs(h - peak_hour))
        raw.append(math.exp(-sharpness * (dh / 12.0) ** 2) + 0.05)
    s = sum(raw)
    return [r / s for r in raw]


def _seasonal(day_of_year, phase, amp):
    """Multiplicative seasonal factor in [1-amp, 1+amp]."""
    return 1.0 + amp * math.sin(2.0 * math.pi * (day_of_year / 365.0) + phase)


def make_catalog(n_products=200, seed=7):
    """Generate the synthetic product/supplier catalog."""
    rng = random.Random(seed)
    products = []
    for i in range(n_products):
        cat_name, phase, amp, peak = CATEGORIES[i % len(CATEGORIES)]
        p_ref = round(rng.uniform(15.0, 60.0), 2)
        products.append({
            "id": i,
            "name": f"{cat_name}_item_{i:03d}",
            "category": cat_name,
            # base daily demand (orders/day at reference conditions).
            # Calibrated so ~50 healthy listings yield ~26 orders/day,
            # matching the paper's economy (humans: 9,442 orders/year).
            "base_daily_demand": round(rng.uniform(0.05, 1.2), 3),
            "season_phase": phase,
            "season_amp": amp,
            "hour_profile": _hour_profile(peak),
            "p_ref": p_ref,
            "eps": round(rng.uniform(0.8, 2.5), 3),      # price elasticity
            # outcome risks (must sum to 1)
            "risk": _sample_risk(rng),
            # supplier parameters (ranges from the paper)
            "sup_price0": round(p_ref * rng.uniform(0.45, 0.65), 2),
            "sup_init_stock": rng.randint(20, 399),
            "sup_capacity": rng.randint(50, 499),
            "sup_replen": rng.randint(1, 19),            # units/hour
            "sup_dispatch_h": rng.randint(1, 47),        # base dispatch hours
            "sup_logistics_h": rng.randint(12, 72),
            # per-product event probabilities (upstream supplier events)
            "p_price_event": rng.uniform(0.005, 0.02),
            "p_delist_event": rng.uniform(0.002, 0.008),
            "p_delay_event": rng.uniform(0.005, 0.02),
        })
    return products


def _sample_risk(rng):
    """Outcome risk profile. ~80-92% Normal; rest abnormal outcomes."""
    p_bad_review = rng.uniform(0.01, 0.05)
    p_ret_refund = rng.uniform(0.02, 0.07)
    p_returnless = rng.uniform(0.01, 0.04)
    p_cancel = rng.uniform(0.02, 0.06)
    p_normal = 1.0 - (p_bad_review + p_ret_refund + p_returnless + p_cancel)
    return {
        "normal": p_normal,
        "cancel": p_cancel,
        "returnless": p_returnless,
        "return_refund": p_ret_refund,
        "bad_review": p_bad_review,
    }


def daily_demand(prod, day_of_year):
    """D_i(d): expected orders on data-day d under reference conditions."""
    s = _seasonal(day_of_year, prod["season_phase"], prod["season_amp"])
    return prod["base_daily_demand"] * s


# Experience scores u / evidence weights v (paper Sec. rating model).
# (normal / late / return+refund / returnless / bad review / stockout)
SCORE_U = {"normal": 4.5, "late": 3.0, "return_refund": 2.0,
           "returnless": 1.5, "bad_review": 1.0, "stockout": 1.0}
SCORE_V = {"normal": 1, "late": 1, "return_refund": 2,
           "returnless": 2, "bad_review": 3, "stockout": 3}

# Star bands: rating -> demand multiplier r_m (paper range 0.10-1.20).
STAR_BANDS = [
    (4.75, 1.20), (4.50, 1.10), (4.25, 1.05), (4.00, 1.00),
    (3.75, 0.85), (3.50, 0.70), (3.25, 0.50), (3.00, 0.30), (0.00, 0.10),
]


def rating_multiplier(rating):
    for lo, mult in STAR_BANDS:
        if rating >= lo:
            return mult
    return 0.10


if __name__ == "__main__":
    cat = make_catalog()
    print(f"catalog: {len(cat)} products across {len(CATEGORIES)} categories")
    p = cat[0]
    print("example:", {k: v for k, v in p.items() if k != "hour_profile"})
    print("sum hour profile:", round(sum(p["hour_profile"]), 4))
