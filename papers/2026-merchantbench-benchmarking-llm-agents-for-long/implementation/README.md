# MerchantBench — from-scratch re-implementation (environment core)

> **Paper:** "MerchantBench: Benchmarking LLM Agents for Long-Term Coherence
> in E-Commerce Operations" — Qiming Shi et al., Alibaba Group / ZJU / PKU /
> Fudan, arXiv:2607.28956 (2026).

MerchantBench is a *benchmark/environment* paper, not a training paper, so
the from-scratch deliverable is the **environment core + baseline merchants +
LTC metrics**, all written from the breakdown (no reference code used).

## What is implemented

| Piece | Paper mechanism | File |
|---|---|---|
| Demand | `λ = D·w·r·ℓ·(p/p_ref)^{-ε}`, Poisson arrivals | `model.py` |
| Listing exposure | ramp ℓ0=0.2 → 14d, decay κ=0.0092, ℓmin=0.10 | `model.py` |
| Store rating | decayed weighted mean, R0=4.0, α=20, γ=2^(-1/30), star bands 0.10–1.20 | `model.py`, `data.py` |
| Order lifecycle | presampled outcome, settle (money) vs outcome (rating/fines) clocks ≤168h apart | `model.py` |
| Cash flows | drop-ship procurement debits instantly; revenue at settlement; fines 5/3/8/5 RMB draw from deposit | `model.py` |
| Upstream events | 3/hour: price ×[0.9,1.5], delisting 3–14d, delay +12–96h | `model.py` |
| Synthetic catalog | 200 products, 6 categories, seasonal demand, hour profiles, risk profiles, supplier params (1688 data is not redistributable) | `data.py` |
| Merchants | passive / random / rule-based (portfolio rotation, event response, liquidity budgeting) | `train.py` |
| Metrics | final net assets `B+D+I+Q`, GMV, rating, fines, **SWR** (min 30-day rolling activity) | `train.py` |

## Run

```bash
python3 train.py                 # all 3 merchants, 3 episodes, 365 days (~2s)
python3 train.py --agents rule --seeds 1 --days 60
```

Pure Python 3 standard library — no dependencies (`requirements.txt` is a
stub). Deterministic per env seed.

## Actual results (this machine, env seeds 42–44)

```
   agent | net_assets | orders | rating |  SWR   |    GMV
 passive |      8,188 |    410 |   3.69 |   0.0% |  14,805
  random |     65,360 |  5,309 |   3.75 |  82.2% | 188,967
    rule |    205,221 |  9,851 |   3.86 | 100.0% | 439,834
   human (paper) | 217,610 | 9,442 | ~4.7 | 100% | —
```

- The **rule merchant lands at ~94% of the paper's human mean** (205k vs
  217.6k RMB) with almost identical order volume (9.9k vs 9.4k/year) — the
  synthetic economy is calibrated to the paper's scale (≈26 orders/day for a
  full 50-slot portfolio, ≈21 RMB margin/order).
- **SWR behaves as designed**: the rule agent stays pinned at 100% (it acts
  every window), the random agent drifts (~82%), the passive agent collapses
  to 0% — reproducing, in miniature, the paper's operational-coherence
  findings (humans 100%, LLMs 10.6–99.4%).

## Honest deviations from the paper

1. **Synthetic catalog** — the 98,843 real 1688 product records are not
   redistributable; we generate a catalog with the same statistical
   structure (seasonality, hour profiles, elasticity, risk, supplier
   params).
2. **Rule baseline uses catalog priors** — `_seasonal_rank` reads
   `base_daily_demand`/season parameters directly (oracle knowledge), so our
   rule merchant upper-bounds an information-limited merchant; the paper's
   own (dumber) rule baseline reached only 24.48k.
3. **No LLM agent plugged in** — the 26-tool LLM interface is exposed via
   `act_*` methods, but no LLM calls are made here (API cost/keys); the
   programmatic merchants validate the environment instead.
4. Simplified settlement FSM: dispatch/delivery lateness folded into the
   settle/outcome clocks rather than explicit per-state transitions.
