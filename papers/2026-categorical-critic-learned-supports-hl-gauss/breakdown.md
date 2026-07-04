# Learning the Supports for Categorical Critic in Reinforcement Learning — Source-First Breakdown

- **arXiv:** 2607.01880 (RL Journal, 2026)
- **Authors:** Jen-Yen Chang, Takayuki Osa, Tatsuya Harada (The University of Tokyo)
- **Venue:** Reinforcement Learning Journal, 2026
- **Subarea (new to repo):** distributional RL / classification-based value learning — dynamic support-endpoint learning for the Gaussian Histogram Loss (HL-Gauss) categorical critic. **First repo paper on distributional RL, categorical critics, or HL-Gauss.** Sibling-in-spirit to the inference-efficiency lineage (both replace a hand-set discretisation with a learned one) but applied to the value-function support, not inference.
- **Source files:** `paper.pdf` (19pp, 1.9MB), `paper_layout.txt` (`pdftotext -layout`, 1207 lines). All numbers below are prose-/table-/equation-confirmed against `paper_layout.txt`; **no figure bar-height back-fill** (the IQM results live in Figure 2 bar curves — per-task IQM endpoints are NOT reliably extractable and are reported only qualitatively, per the universal "figure-derived numbers are weak" rule).

---

## 1. The problem (motivation)

Classification-based value learning via **HL-Gauss** (Imani & White 2018; Farebrother et al. 2024) reframes scalar Q-regression as classification: each Bellman target is encoded as a Gaussian-smoothed categorical distribution over `k` fixed bins spanning a **pre-defined support interval `[νmin, νmax]`**, trained with cross-entropy. This gives smoother gradients and scales better than MSE regression.

The fundamental limitation: **the support interval must be specified a priori**, but the true return range is rarely known and is **non-stationary** (early-exploration vs converged-policy returns differ substantially). This forces a lose-lose trade-off:

- **Too narrow → truncation bias.** Target mass leaks outside `[νmin, νmax]`; the decoded categorical mean `E_qy[z]` diverges from the true mean `µ` (Figure 1: leaked mass `1−Z` is the cause).
- **Too broad → quantisation bias.** With fixed `k`, a wider support widens each bin `wi = (νmax−νmin)/k`, diluting per-bin resolution and weakening the learning signal.

No single fixed interval suits every policy encountered during training.

---

## 2. Core theoretical contribution: HL-Gauss upper-bounds the MSE Bellman error

The central result (§3.1, Equations 6–11). Starting from the MSE Bellman error and applying (a) the `(M+N)² ≤ 2M²+2N²` split, (b) the substitution `Q(s,a) := E_hx[z]`, and (c) Imani et al. 2024 Eq. 7, the paper derives:

```
MSE_Bellman  ≤  8 · max(|νmin|,|νmax|)² · min( ½·KL(qy‖hx), 1−exp(−KL(qy‖hx)) )        (Eq 11)
                    |       {z       }    |                  {z                  }|
                  "absolute width"          "distribution matching (≈ cross-entropy)"
                 + 2·(E_qy[z] − (TQ)(s,a))²
                    |         {z          }|
                      "truncation bias"
```

**Three takeaways from Eq 11** (these are the paper's citable falsifiable claims):

1. The MSE Bellman error is **upper-bounded by the HL-Gauss loss** — so minimising HL-Gauss provably controls MSE.
2. The bound's tightness is **directly correlated with `max(|νmin|,|νmax|)`** — the absolute width of the support. A narrower support ⇒ tighter bound. This motivates **seeking the narrowest feasible interval**.
3. Prior HL-Gauss works discard the absolute-width term because, with a *fixed* broad support, it is constant w.r.t. trainable parameters and the truncation bias is negligible — a valid simplification **only under the fixed-broad-support assumption**. DySEL breaks that assumption, so the width term re-enters the gradient.

> ⚠ **Honest-scope note on Eq 11's constants (transcribed verbatim, not "fixed").** The pdftotext rendering of the AM-GM application (Eq 12) partially garbles the leading constant (`4·max·CE ≤ α·max² + (1/α)·CE²`). The standard AM-GM `αX² + Y²/α ≥ 2XY` bounds `2·max·CE`, so the paper's `4` (and the `8` in Eq 11) absorb the `min(½KL, 1−exp(−KL)) → CE` substitution factor. Cite Eq 11/12 by their stated form and describe from the prose rather than re-deriving the constant — do not treat the rendered coefficient as exact.

### 3. Preliminaries (verbatim definitions)

- Truncated Gaussian target (Eq 1): `q(y) = (1/Z)·(1/√(2π))·exp(−(y−µ)²/(2σ²))`.
- **Covering mass** `Z` (Eq 2) = integral of the truncated-Gaussian PDF over `[νmin, νmax]` (defined via `erf`). `Z ≈ 1` iff the support is sufficiently broad. **Unrelated to the random-return variable `Z^π`.**
- Per-bin weight (Eq 3): `ci = ½·[erf((li+wi−µ)/√(2σ)) − erf((li−µ)/√(2σ))]`, with bin edges `li = νmin + (i−1)·wi` (Eq 4).
- HL-Gauss loss (Eq 5): `L = −Σ_i ci·log hi(x)`, where `hi(x)` is the predicted probability of bin `i` and `zi = li + wi/2` is its centre; the categorical decodes to `Σ_i hi(x)·zi`.
- Two biases when decoding: **truncation bias** `= E_qy[z] − µ` (narrow-support pathology) and **quantisation bias** `∝ wi = (νmax−νmin)/k` (broad-support pathology).

---

## 4. The DySEL method (§4, Eq 12–17)

**DySEL = Dynamic Support Endpoint Learning.** Recasts the Eq 11 minimisation as a **constrained optimisation / min-max game** with two opposing forces:

- **Width penalty** (compress): minimise `max(|νmin|,|νmax|)` → tighter MSE upper bound.
- **Mass constraint** (cover): keep leaked mass `(1−Z) ≤ ε` → no truncation bias.

**Step 1 — address the non-convex product (Eq 12).** The product `max(|νmin|,|νmax|)·CE` is non-convex; apply AM-GM with a positive scaling constant `α > 0`:

```
4·max(|νmin|,|νmax|)·CE  ≤  α·max(|νmin|,|νmax|)²  +  (1/α)·CE²           (Eq 12)
```

**Step 2 — reduce truncation bias to leaked mass (Eq 13–14).** Let `µout` = expected value of the leaked mass. By the law of total expectation:

```
(TQ)(s,a) = Z·E_qy[z] + (1−Z)·µout            (Eq 13)
E_qy[z] − (TQ)(s,a) = (1−Z)·(E_qy[z] − µout)   (Eq 14)
```

For a thin-tailed truncated Gaussian the displacement `(E_qy[z] − µout)` is bounded, so **driving `(1−Z) → 0` drives the truncation bias to zero** — it suffices to control `(1−Z)` directly. Since `max(Z) = 1`, `(1−Z)` is usable as a constraint.

**Step 3 — constrained problem (Eq 15):**

```
J(θ,ϕ) = α·max(|νmin|,|νmax|)  +  (1/α)·(−Σ_i ci log hi(x))
         |       {z         }       |        {z         }
            width penalty              cross-entropy
subject to  (1−Z) ≤ ε                                                   (Eq 15)
```

**Step 4 — final min-max Lagrangian objective (Eq 16):** introduce a trainable Lagrangian multiplier `λ ≥ 0`:

```
L(θ,ϕ,λ) = α·max(|νmin|,|νmax|)  +  (1/α)·(−Σ_i ci log hi(x))  +  λ·((1−Z) − ε)
           |       {z         }       |        {z         }          |   {z      }|
              width penalty              cross-entropy              mass constraint    (Eq 16)

θ*,ϕ* = argmin_{θ,ϕ} L(θ,ϕ,λ);   λ* = argmax_{λ≥0} L(θ,ϕ,λ)            (Eq 17)
```

**`α` is the single remaining hyperparameter** (governs width-penalty vs cross-entropy trade-off). `λ` is learned automatically via the adversary. The support endpoints `[νmin, νmax]` are output by a small MLP (see §5).

> **Citable mechanism (the min-max tension).** The width penalty wants the support *narrow*; the mass constraint (enforced by the adversary `λ`) wants it *broad enough to cover the target mass*. DySEL converges to the narrowest support that still satisfies `(1−Z) ≤ ε` — i.e. the tightest valid MSE bound. This is the cleanest single-sentence statement of the contribution.

### Implementation details (§A, all grep-confirmed vs `paper_layout.txt`)

| Item | Value | Source |
|---|---|---|
| Base algorithm | TD3 (Fujimoto et al. 2018) | §5.1 |
| Support-interval network | 3-layer MLP, hidden `(256,256)`, final layer outputs 2 values, sorted so output[0]=νmin, output[1]=νmax | §A |
| Init support interval (DySEL) | `[−10, 10]` for all tasks | §A |
| Baseline HL-Gauss support | `[−100, 100]` (DM-Control rewards ∈ [0,1]/step ⇒ discounted return ≤ `1/(1−γ)=100`, so `[−100,100]` is truncation-free; the cost is *resolution*) | §A |
| Bins `k` | 128 (both HLG and DySEL; raised from C51's 51 because 51 over `[−100,100]` ⇒ bin width ≈ 3.92 ≈ "nearly 4") | §A |
| σ-to-bin-width ratio | 0.75 (truncated-Gaussian std) | §A |
| **σ clip minimum** | **0.3** (without it the adversary cheats by shrinking σ in the `Z` formula early in training) | §A, after Table 1 |
| `ε` (mass-constraint slack) | 0.005 for all tasks, untuned | §A |
| Adversary optimiser | Adam, lr `1e−03` | §A |
| Shared: replay buffer | 10⁶ | §A |
| Shared: discount `γ` | 0.99 | §A |
| Shared: target-net rate `τ` | 0.005 | §A |
| Shared: init random steps | 10000 | §A |
| Shared: batch size | 256 | §A |
| Shared: networks | 2 hidden layers × 256 units, ReLU | §A |
| Shared: optimiser | Adam, lr `0.0003` (3e-4) | §A |
| Framework | JAX 0.6.2, MuJoCo 3.3.7, DM Control Suite 1.0.34, gym 0.23.1 | §A |

---

## 5. Experiments (§5)

### 5.1 Setup (verbatim, §5.1)

- **Benchmark:** DeepMind Control Suite, **11 tasks** (max achievable return 1000).
- **Baselines:** TD3; **TD3+HLG** (HL-Gauss on TD3, support `[−100,100]`, `k=128`); **TD3+DySEL** (this paper, `k=128`).
- **Protocol:** **10 seeds** `{0..9}`, **3 million timesteps**, evaluate **every 10000 steps**, **20 evaluation episodes** per eval, report **IQM** (Inter-Quantile Mean) with 95% bootstrapped-CI shaded bounds — best practice per Agarwal et al. 2021.

### 5.2 Results — 5 Q&As (§5.2, prose-confirmed; per-task IQM numbers are Figure-2 curve reads)

> ⚠ **Figure-derived-results caveat.** The IQM returns for each task are depicted as **Figure 2 learning curves** (TD3 / TD3+HLG / TD3+DySEL over 0–3M env-steps). Per-task final IQM values are **axis/curve readings, not table cells** — they are NOT reliably transcribable and are reported here only as the paper's qualitative ranking, consistent with the universal "figure-derived numbers are weak" rule.

**Q1 — Performance vs HL-Gauss?**
**A1:** TD3+DySEL is **generally competitive with vanilla HL-Gauss and yields clear gains on several tasks, most notably the humanoid tasks** (humanoid-run / humanoid-stand / humanoid-walk). Hypothesised mechanism: the dynamically learnt (narrower) support gives finer granularity, letting the critic discern subtle return variations obscured by the fixed broad support. **On `finger-turn_hard` and `hopper-stand`, both HL-Gauss-based approaches underperform or match plain TD3** — Figure 3 shows these are tasks where the support converges almost immediately, suggesting the underperformance is tied to the cross-entropy objective itself, not to dynamic-support learning.

**Q2 — Important hyperparameters?**
**A2:** **`α` is the dominant hyperparameter** (width-penalty vs cross-entropy balance, Eq 16). Larger `α` ⇒ narrower initial support (width penalty dominates). Three choices remain: `α`, the initialising support interval, and `k` (shared with all categorical critics). Starting from `[-10,10]` or `[-5,5]` is "good enough"; init-interval sensitivity is in Appendix B.1.

**Q3 — What supports are actually learnt?**
**A3:** Highly task-dependent, correlated with `α`. Two behavioural regimes (Figure 3): (i) **immediately stable** — `finger-turn_hard`, `hopper` tasks converge to a near-constant interval; (ii) **gradually expanding** — `cheetah-run`, `fish-swim`, humanoid, `quadruped` tasks broaden over training. The learnt interval is **almost symmetric**. Tasks needing finer granularity require higher `α`; tasks needing a stable interval require lower `α`.

**Q4 — Use the learnt interval as a fixed reference for vanilla HL-Gauss?**
**A4:** Mixed. Taking DySEL's *final* (3M-step) support as a fixed `[νmin,νmax]` for TD3+HLG **helps on stable-support tasks** (`finger-turn_hard`, `humanoid-run`) but **hurts on expanding-support tasks** (`fish-swim`, `humanoid-stand`) — where the final interval is too narrow for early training. DySEL adapts to both regimes; a single fixed interval cannot. **Practical advantage: removes the need to tune `[νmin,νmax]` per task.**

**Q5 — Component ablation (Figure 5)?**
**A5:** **Both components of Eq 16 are critical for stability.**
- **Remove the absolute-width penalty** ⇒ the algorithm **diverges** — cross-entropy artificially minimises its objective by *expanding the support toward infinity* (exploiting the supports instead of matching the distribution; `|νmin|,|νmax|` grow without bound; plotted on sym-log scale).
- **Remove the mass constraint (no adversary)** ⇒ for expanding-support tasks like `cheetah-run`, the width penalty overpowers and the interval becomes **too narrow to cover the target distribution** ⇒ insufficient mass coverage mid-training.

Mechanism (verbatim): "value targets effectively lie in a bounded interval with probability one; increasing the parametrisation support with a fixed bin size eventually places all targets in a single bin, which trivially yields zero cross-entropy. **The width penalty is what prevents this degenerate widening.**"

---

## 6. Table 1 — per-task `α` (verbatim, `paper_layout.txt` L1008–1019)

> Selected via grid search over `[0.1, 0.2, 0.3, 0.4, 0.45, 0.5]`. `α` is the **only** method-specific tuned quantity — DySEL replaces HL-Gauss's per-task `[νmin,νmax]` tuning with a single scalar.

| Task | α | Task | α | Task | α |
|---|---|---|---|---|---|
| cheetah-run | 0.45 | hopper-stand | 0.1 | quadruped-run | 0.5 |
| finger-turn_hard | 0.1 | humanoid-run | 0.2 | quadruped-walk | 0.3 |
| fish-swim | 0.5 | humanoid-stand | 0.5 | walker-run | 0.5 |
| hopper-hop | 0.2 | humanoid-walk | 0.5 | | |

*Source-free reconciliation: 11 tasks ✓ (matches "11 tasks" in §5.1). All 11 α values ∈ grid `{0.1,0.2,0.3,0.4,0.45,0.5}` ✓. Histogram: 0.5×5, 0.2×2, 0.1×2, 0.45×1, 0.3×1. `α=0.4` is in the grid but selected by **no** task — the best per-task α never landed on 0.4.*

---

## 7. Related-work positioning (§6)

- **Distributional RL / categorical value learning:** C51 (Bellemare et al. 2017), HL-Gauss (Imani & White 2018; Farebrother et al. 2024), two-hot encoding, categorical-distribution analysis (Rowland et al. 2018/2019). Modern SOTA distributional-RL components: BRO (Nauman et al. 2024), SimbaV2 (Lee et al. 2025). All rely on a pre-defined support.
- **Dynamic supports (closest prior work):** Chen et al. 2025 (*Adaptive HL-Gaussian*) also targets the MSE upper bound but via **pure minimisation**; DySEL **diverges by reformulating as constrained min-max** (expand to cover mass, contract to tighten the bound).
- **Non-stationarity / value scaling:** PopArt (van Hasselt et al. 2016; Hessel et al. 2019) normalises scalar targets via mean-side statistics — does not preserve distributional shape for categorical critics.
- **Quantile regression (the alternative branch):** IQN (Dabney et al. 2018a), FQF (Yang et al. 2019) avoid categorical supports by learning quantile locations — no support to set, but suffer **quantile crossing** (invalid distributions; Zhou et al. 2020) and lack the categorical inductive bias. DySEL stays on the categorical branch and makes its support adaptive.

---

## 8. Honest-scope limitations (§7, surfaced not buried)

1. ⚠ **`α` selected on the same seeds used to report results.** Verbatim from §A: *"α was selected on the same seeds subsequently used to report results in Figure 2; the per-task performance of DySEL should therefore be read as an **optimistic (upper-bound) estimate rather than an unbiased one**."* This is the most important caveat — the headline per-task gains are an upper bound on expected performance; Figure 7 (α-sensitivity) is supplied to let the reader assess variation across the searched range directly.
2. ⚠ **Saddle-point stability is sensitive to the dual learning rate.** Aggressive dual (`λ`) updates induce oscillatory bound-network behaviour early in training. The authors propose (but do not implement) a **PID-controlled Lagrangian update** (Stooke et al. 2020) to damp this.
3. ⚠ **Learnt support is almost symmetric** — penalising each bound separately (decoupling νmin/νmax) is left to future work; an asymmetric shifted support might better match skewed true return ranges.
4. ⚠ **Evaluation is DM-Control-focused** (11 continuous-control tasks, max return 1000). Broader benchmarks (Atari, other suites) are left to future work.
5. ⚠ **Adaptive `σ` unexplored** — fixed σ-to-width ratio (0.75); adaptive σ could give finer control over compression/expansion and accommodate increasing-`γ` schedules (François-Lavet et al. 2015).
6. ⚠ **Per-task IQM numbers are Figure-2 curve reads** (not table cells); only the qualitative ranking (DySEL competitive-to-better, humanoid gains, finger-turn_hard/hopper-stand no-improvement) is prose-confirmed — see §5.2 caveat.
7. ⚠ **Eq 11/12 leading constants** are partially garbled in pdftotext (`4`/`8` absorb the `min(½KL,1−exp(−KL))→CE` substitution factor) — cite by stated form, describe from prose; do not treat the rendered coefficient as exact (see §2 note).

---

## 9. Strengths / Limitations / Verdict

**Strengths.**
- **Genuine theoretical contribution:** HL-Gauss provably upper-bounds the MSE Bellman error (Eq 11), and the bound tightens with `max(|νmin|,|νmax|)`. This converts "pick a wide-enough interval" from folklore into "minimise the interval subject to a mass constraint" — a principled constrained-optimisation formulation (Eq 15–16).
- **Eliminates the most fragile HL-Gauss hyperparameter** (`[νmin,νmax]` per task) in exchange for a single scalar `α`, which itself admits a sensible default range `[0.1,0.5]`.
- **Clean component ablation (Figure 5):** width-prevents-degenerate-widening, mass-prevents-narrowing — both forces are necessary and their failure modes are mechanistically explained.
- **Honest self-assessment:** the same-seed-α-selection caveat (§A) is stated openly, and the α-sensitivity sweep (Figure 7) is provided so the reader can bound the optimism.

**Limitations (beyond §8).**
- Results are **figure-only** (no IQM table) — makes independent numerical verification hard; a single aggregated IQM/median table across the 11 tasks (with CIs) would have substantially strengthened the paper.
- **Single base algorithm (TD3)** — no SAC/PPO ablation; generality across actor-critic families is assumed, not shown.
- The theoretical bound (Eq 11) is an **upper bound**, not an exact equality; the link between "tighter bound" and "better policy" is empirical, not proven.

**Verdict.** A clean, well-motivated theory-first paper that solves a real, long-standing practical pain point of HL-Gauss (per-task support tuning under non-stationary returns) by recasting it as a principled min-max constrained problem. The contribution is the **mechanism + the Eq 11 upper-bound decomposition**, not a new DM-Control SOTA — the per-task gains are real but modest, figure-only, and (per the authors' own caveat) an optimistic upper bound. The most citable single result is the **two-force min-max formulation** (compress for tightness, cover for correctness), which generalises beyond HL-Gauss to any fixed-support categorical value-learning method. **Recommended as a reference for anyone deploying HL-Gauss/categorical critics who cannot pre-specify the return range.**
