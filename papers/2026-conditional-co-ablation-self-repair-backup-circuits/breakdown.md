# Conditional Co-Ablation (CoAx): Recovering Self-Repair Backups in Transformer Circuits

- **arXiv:** 2607.01940v1 [cs.LG], 2 Jul 2026 (12pp + appendices)
- **Authors:** Zhiren Gong, Zihao Zeng, Chau Yuen, Wei Yang Bryan Lim — Nanyang Technological University (NTU), Singapore
- **Links:** Code / Project Page / Tutorial (see paper); arXiv abs https://arxiv.org/abs/2607.01940
- **Subarea:** mechanistic interpretability — **component-importance scoring under self-repair / redundancy**. A *second-order, conditional* importance score that recovers dormant backup components no first-order (node-additive) score can see. Sibling-in-spirit to `2026-expander-sparse-autoencoders` (both mech-interp: expander studies *which feature directions a dictionary encodes*, CoAx studies *which heads a circuit falls back on*), but a different mechanism (redundancy/conditional importance, not dictionary architecture).

---

## TL;DR

First-order interpretability scores a unit by ablating it **alone** and measuring output change. When a transformer **self-repairs** (the Hydra effect: a primary component removed → a dormant backup takes over), that score misreads *both* sides of the redundancy — the primary looks unimportant (damage repaired) and the backup looks irrelevant (silent on the intact model). **Conditional Co-Ablation (CoAx)** flips the question: not "how much does unit `u` matter on the intact model?" but "how much does it matter **once the primary set `S` is already ablated?**" A dormant backup has near-zero solo effect but a large *conditional growth*; an inert unit has neither.

- **Headline (GPT-2-small IOI):** CoAx lifts documented-backup-head recovery from **0.33 → 0.91 ROC-AUC**, beating every node-ranking baseline incl. self-repair-aware **AtP\* GradDrop 0.82** and the fair seeded **GIM 0.63** (Table 1).
- **Label-free + forward-only:** no backward pass, no task gradient. `2|U|+1` forward passes per seed — `O(|U|)`, ≈36× cheaper than the explicit `O(|U|²)` pairwise synergy that carries the same second-order signal (Tables 6, 7).
- **Closes the downstream loop** with the one recovered set: attribution **0.22 → 1.76** logit-diff drop (Table 3), capability knockout **1.00 → 0.70 IOI acc** matching the documented **0.72** oracle while a first-order top-up overshoots to **0.24** (Table 4), and repair-aware structured head pruning that beats weight/magnitude/gradient baselines at every scale **124M → 7B** (Tables 26, 27).
- **Generalizes** label-free across the GPT-2 family (scale) and to a second redundant circuit (induction) across **8 models / 6 architecture families**, attribution factor **2.1×–12×** (Tables 21, 22).

---

## 1. The problem: self-repair breaks additivity

Standard mech-interp primitives — automated circuit discovery (ACDC), attribution/edge patching, EAP-IG, AtP\*, sparse feature circuits, Wanda saliency — all score a component by its **single-node** ablation effect and combine these additively across the circuit. This is fine when components contribute independently; it fails when a transformer self-repairs.

**The IOI blind spot (concrete).** In GPT-2-small Indirect-Object-Identification, ablating the name-mover heads (that write the answer) drops the task logit-difference by only **0.22** from a clean **2.53** — not because the name-movers are weak, but because backup name-movers immediately take over. The IOI name-mover module is **1.9× super-additive**: the effect of removing a set is *not* the sum of the effects of removing its members. The resulting bias falls squarely on the redundant components that make a circuit robust, and propagates into attribution, knockout, and pruning.

> A component's importance is therefore not merely an isolated-unit property: in robust circuits, the components that matter can become visible **only under the interventions that make them necessary.**

---

## 2. Method: the CoAx score

Setup: frozen decoder-only transformer; a structured unit `u` is any component with an additive residual-stream contribution that ablation can zero (experiments use **attention heads**, the granularity with head-level circuit ground truth). Ablating set `M` → logits `z_M`; `z_∅ ≡ z_0` clean; `p_0 = softmax(z_0)`. Single-unit ablation effect `δz_u = z_0 − z_{u}`.

### 2.1 Fisher geometry over ablations

The natural metric is the one the output distribution induces. Fisher information `F = diag(p_0) − p_0 p_0ᵀ` is exactly the Hessian of `KL(p_0 ∥ softmax(z))` at `z_0`, so a logit perturbation `δz` has second-order KL cost `½ δzᵀ F δz + O(‖δz‖³)`. Realized via the centered, Fisher-weighted feature:

```
δz_e_u = √p_0 ⊙ (δz_u − E_{p_0}[δz_u]·1)        # on the top-r logits
```

Centering subtracts the shared logit-shift an ablation imparts to every output coordinate at once, leaving only the perturbation that changes distribution *shape*; the `√p_0` weighting makes the plain inner product coincide with the Fisher form — `⟨δz_e_u, δz_e_v⟩ = δz_uᵀ F δz_v` (Proposition 1) — so `E(δz_u) := E_{x,t}‖δz_e_u‖²` is the **mean KL energy** of ablating `u`. The design matrix `D̄` (one column/unit, rows over top-r logits × P calibration positions) gives the Gram kernel `H = (1/P)D̄ᵀD̄` and the **Fisher-cosine affinity** `A_uv = H_uv / √(H_uu H_vv) ∈ [−1,1]` — the first-order baseline. `H` is stable but first-order; the genuinely second-order content lives in two objects it omits.

### 2.2 Pairwise synergy (the cooperation lens)

For units `u, v`, joint-ablation perturbation `δz_uv = z_0 − z_{u,v}`. The **synergy** is the non-additive part:

```
I_uv = δz_uv − δz_u − δz_v ,    S_uv = E_{x,t}‖I_e_uv‖²          # (Eq. 1)
```

vanishes when units act independently, large when they compensate. Captures symmetric cooperation that first-order affinity cannot model. (Used for same-circuit clustering, Table 28; **the headline discovery + all applications use the CoAx score below**, not synergy directly.)

### 2.3 The CoAx score (the substitution lens)

A backup is dormant until the primaries are gone, so score units **after conditioning on a seed set `S`** (typically the high-saliency primaries).

> **Definition 1 (Conditional co-ablation score).** The conditional ablation effect of `u` given `S` is its marginal effect once `S` is already ablated: `δz_{u|S} = z_S − z_{S∪{u}}` (so `δz_u ≡ δz_{u|∅}`), with energy `E(δz_{u|S}) = E_{x,t}‖δz_e_{u|S}‖²`. The **CoAx score** is the growth of this energy under conditioning:
>
> ```
> comp_u(S) = E(δz_u | S) − E(δz_u | ∅)              # (Eq. 2)
> ```

Large for backups (effect appears once `S` is ablated); near zero for non-redundant units. **Cost:** `O(|U|)` forward passes per seed — same order as one single-ablation scan, far below `O(|U|²)` for all pairs — which is what lets the second-order signal scale. Algorithm, cost, calibration efficiency: Appendix A.

**Division of labor.** Pairwise synergy `I_uv` = lens for *cooperation* (symmetric same-circuit structure); CoAx `comp_u(S)` = lens for *substitution* (dormant backups as a node-level compensating set).

### 2.4 Why first-order misses backups and CoAx does not (Propositions 1–3)

Three short results (proofs in Appendix A.1.3):

- **Proposition 1 (Fisher identity).** `⟨δz_e_u, δz_e_v⟩ = δz_uᵀ F δz_v` with `F = diag(p_0) − p_0 p_0ᵀ`. So `H` is PSD and `E(δz_u) = E_{x,t}[δz_uᵀ F δz_u]` is the mean KL energy of ablating `u`. The centering term is exactly the `−p_0 p_0ᵀ` of the Fisher form.
- **Proposition 2 (Additivity blind spot).** Call `u` a *pure backup* for `S` if dormant on the clean model (`δz_b = 0`) yet carries the effect once `S` is ablated (`E(δz_b|S) = Δ > 0`); *inert* if silent throughout. Any score `g(θ_u)` over a per-unit statistic `θ_u` from the clean pass **that is invariant between a pure backup and an inert unit** (as any function of clean marginal effect or clean activation must be — both vanish dormant) assigns them the same value and cannot rank one above the other. By contrast `comp_b(S) = Δ > 0` while `comp_inert(S) = 0`: **CoAx separates them.**
- **Proposition 3 (Conditional growth is seed synergy).** Under a pairwise-interaction truncation of the joint ablation effect, `δz_{u|S} = δz_u + Σ_{s∈S} I_su`, hence `comp_u(S) = E_{x,t}⟨δz_e_u, Σ_s I_e_su⟩ + ½‖Σ_s I_e_su‖²`. For a dormant unit (`δz_u ≈ 0`), `comp_u(S) ≈ E_{x,t}‖Σ_{s∈S} I_e_su‖²` — the energy of its **synergistic coupling to the seed**. So CoAx is genuinely second-order, not an output-grounded re-ablation.

**Picture:** a backup is an output-space *substitute* for the primaries — writes a correlated logit direction, so its synergy `I_sb` with the seed is large even as its solo effect `δz_b` stays small. Prop 2 makes such a head invisible to any score meeting the invariance condition; Prop 3 shows CoAx reads exactly the off-diagonal interaction those scores discard. On IOI the name-movers and their backups bind into a dense high-synergy module (Figure 2): off-diagonal mass in `I_uv` that single-ablation saliency — reading only the diagonal — leaves dark.

---

## 3. Establishing CoAx on GPT-2-small IOI

Protocol: GPT-2-small IOI circuit (Wang et al. 2022) — the **only** circuit with head-level backup ground truth (8 documented backups). Two ROC-AUCs, never compared: **backup-AUC** (node-level, ranking documented backups given the primaries — Table 1's headline metric) and **cluster-AUC** (pair-level, whether two heads share a circuit — Table 28). Backup numbers = mean over 4 prompt seeds (std ≤ 0.04).

### 3.1 Backup discovery — Table 1 (verbatim, paper_layout.txt lines 343–358)

Central result. Backup name-movers are hard for *every* node-ranking baseline, including those explicitly designed for self-repair. CoAx is the only score that clears 0.9.

| signal | backup AUC | primary AUC |
|---|---|---|
| single ablation (1st) | 0.33 ± 0.00 | 0.43 ± 0.03 |
| AtP (1st) | 0.60 ± 0.03 | 0.81 ± 0.01 |
| GIM-style † (1st) | 0.63 ± 0.05 | — |
| EAP-IG (1st) | 0.70 ± 0.02 | 0.75 ± 0.03 |
| AtP\* GradDrop (1st) | 0.82 ± 0.03 | — |
| **CoAx † (2nd, ours)** | **0.91 ± 0.00** | — |

> † CoAx uses a primary-ablated seed. The *fair* same-seed comparison is the seeded GIM (**0.63**), not the seed-free AtP\* (0.82). The gradient baselines locate the primary name-movers well (≈0.78; single-ablation is itself partly masked at 0.43). The gap is not conditioning or a smarter gradient — it is the node-additive *form* itself: a backup's contribution is a non-additive substitution that **no additive score, however corrected, can model.**

- CoAx places **6 of 8** documented backups in its top 20 of 141 candidates (top-10 recall **4/8**), far above chance.
- **Statistics (do not lean on the small positive set):** label-permutation test puts 0.91 entirely outside the null (**p < 1e-4**); hypergeometric top-k significant at every cutoff (**6/8 in top 20, p = 9e-5**); paired DeLong over the 8 backups vs seeded GIM 0.63 gap 0.26 (**p = 2e-3**). The smaller 0.09 gap over the seed-free AtP\* (0.82) holds in every seed but is underpowered on only 8 positives — an artifact of comparing against a baseline denied the seed.
- **CoAx completes; it does not discover from scratch.** As a standalone finder that must detect its own seed, CoAx peaks at **0.60** (Appendix C.5.1), below seed-free AtP\* (0.82). The 0.91 headline is a **completion** result with documented primaries as the seed.

**Control worth flagging.** An input-side **co-activation** score also ranks the IOI backups highly (**0.92 AUC**) since they co-fire with the primaries — so CoAx is not the *only* signal that finds them. But co-activation is correlational: no causal/patching validation, it **collapses on movement circuits** (duplicate-token **0.32 vs CoAx 0.97**), and as a completion signal it over-ablates (top-k pulls in co-firing core heads, flipping the IOI logit-difference sign). Treated separately (§5); causal node-ranking scores kept.

### 3.2 Are the discovered heads real backups?

A discovery method must surface heads that *behave* like backups, not merely match labels. For the top-10 CoAx heads, two label-free signatures (activation ratio = output norm primaries-ablated/clean; conditional causal effect = logit-diff drop from ablating the head once primaries gone):

- Documented backups among the top-10: **1.21 / +0.21** (wake up and load-bearing) vs other surfaced heads **1.03 / −0.12**. Keeping only heads that both wake up and are load-bearing raises documented-backup precision in the top-10 from **0.40 → 1.00**.
- **Independent structural check (anti-circularity):** a name-mover attends to and copies the IO token — a bespoke read sharing *none* of CoAx's machinery — ranks the backups at **0.96 ROC-AUC** with Spearman **ρ = 0.09** to CoAx, corroborating rather than restating the discovery.

**How the backups take over (Figure 3).** Ablating more primary name-movers (`k=0,1,2,3`, strongest first): discovered backups wake up monotonically (output-norm ratio **1.00 → 1.15**, conditional causal drop **+0.05 → +0.11**) while a matched random control stays flat. Direct-logit-attribution (DLA) decomposition: primaries carry **+0.76** clean DLA to the IO−S direction; backups' DLA **more than doubles once primaries ablated (+0.07 → +0.21)**. **Counterfactual patching closes the causal loop:** ablating primaries but freezing backups to their dormant value removes **55%** of the self-repair, while freezing a matched random set removes none. The wake-up *causally* drives the repair.

### 3.3 Faithfulness, completeness, minimality (community-standard criteria)

- **Completeness** (the criterion that most sharply separates CoAx — self-repair is exactly a completeness failure). Mean-ablate every head outside a circuit, measure IOI logit-diff response to ablating name-movers vs the full model. First-order circuit (documented IOI circuit *without* backups) is badly incomplete (gap **0.72**); completing with CoAx backups closes it to **0.15**, matching the complete documented circuit (**0.16**), whereas matched-random completion does not (**0.61**).
- **Minimality.** Once name-movers ablated, documented backups among surfaced heads have positive mean conditional logit-diff drop (**+0.21**) while non-backup surfaced heads average negative (**−0.12**). Individual-head drops are noisy — read as *evidence* for minimality, not proof every selected head is necessary.
- **Faithfulness.** Counterfactual patching (Figure 3d) gives the causal form: the completed circuit's backups are the components the model actually uses under intervention — freezing them to dormant removes >half the self-repair, freezing random removes none.

### 3.4 CoAx as a completion module — Table 2 (verbatim, lines 344–352)

Realistic test: the primary seed comes from a first-order method. Seed CoAx with the top-3 heads of AtP / EAP-IG / AtP\*, add its top-4 backups — **roughly doubles** the joint-ablation IOI logit-diff drop, far above matched-random:

| finder | prim. | +CoAx | +own | +rand | FO pct. |
|---|---|---|---|---|---|
| AtP | 1.28 | 3.11 | 3.09 | 1.69 | 0.80 |
| EAP-IG | 0.68 | 1.71 | 1.75 | 0.95 | 0.81 |
| AtP\* | 1.85 | 3.73 | 3.77 | 2.25 | 0.45 |

> `+own` = finder's own next-`m` heads; FO pct. = mean first-order percentile of CoAx-added heads (1 = top). On raw faithfulness CoAx merely *ties* `+own`, but recruits far lower-ranked heads (**0.69 vs 0.97** percentile) — what matters is the *identity* of the heads, not their number (settled by the knockout of §4.2: `+own` overshoots into the core circuit, CoAx matches the documented set). Completion quality tracks seed quality (Appendix C.4.2).

**Same-circuit structure (built-in ablation).** Pairwise synergy clusters same-circuit heads (cluster-AUC, Table 28): wins decisively on information-movement circuits over input-side controls (duplicate-token **0.97 vs 0.32/0.59**; induction **0.94 vs 0.75/0.89**), because these heads share aligned output effects but neither correlated activations nor aligned value subspaces; co-activation wins the co-located *writing* heads — the two lenses are complementary. Doubles as a CoAx ablation: dropping the second-order term collapses name-mover clustering **0.76 → 0.34** — the gain is the interaction term, not merely output-grounded ablation.

### 3.5 Generalization: scale + architecture

- **Scale (IOI across the GPT-2 family).** Identical label-free pipeline on GPT-2 medium + large (no backup labels). Recovered set wakes up when primaries ablated on every size — output-norm ratio **1.15 / 1.05 / 1.13** (small / medium / large) vs ≈1.00 for the rest of the model (≤0.01 std over two seeds) — and is load-bearing only then. Not a GPT-2-small artifact.
- **Architecture (induction across 8 models / 6 families).** Apply CoAx to induction (a second attention-mediated redundant circuit) with a fully label-free pipeline that also detects the primaries. On GPT-2-small: conditional causal drop **0.89 vs 0.05** random; primaries-only drop 0.27, adding compensators drops induction log-prob by **8.5 (~10× matched-random)**. Holds across 8 models, attribution factor **2.1×–12×** (Tables 21, 22).

**Scope/regime (honest).** CoAx targets the harder *dormant-substitution* regime (silent until primary removed; first-order provably fails — Prop 2). Where redundancy is shared among already-co-firing units (induction is closer), components are not hidden, so even co-activation finds them and CoAx is *complementary*, not a replacement — indeed on induction a `+own` control (model's own next-strongest induction heads) is comparable to CoAx (recovery, not unique identification). The head-level signal does **not** transfer to the MLP-dominated *greater-than* circuit (a preliminary FFN-group probe recovers only 1.5× over random, within one std) — suggesting greater-than carries much weaker recoverable self-repair at this granularity (a property of the circuit, not the unit).

---

## 4. Closing the loop: attribution, knockout, pruning

A blind spot in component importance does not stay confined to discovery. Attribution, knockout, and pruning all rank components by the same node-additive score, so each inherits the same error wherever a circuit is redundant. The one CoAx-recovered set serves all three.

### 4.1 Attribution — Table 3 (verbatim, lines 646–650)

IOI logit-difference drop from ablating the name-mover primaries plus a top-up set (4 seeds; clean **2.53**):

| top-up set | prim. only | +rand | +doc. | +CoAx |
|---|---|---|---|---|
| logit-diff drop | 0.22 | 1.0 ± 0.7 | 1.15 | **1.76** |

> Ablating primaries alone drops the logit-diff by only **0.22** (single-seed 0.11, reconciled in App. D) because backups absorb the damage. Re-attributing together with the CoAx backups recovers a **1.76** drop — the effect the redundancy had hidden — exceeding matched-random (1.0±0.7) and even the curated documented backups (1.15) at every seed. The small but consistent margin over the documented set hints the hand-annotated list does not exhaust the functional compensators.

### 4.2 Capability knockout — Table 4 (verbatim, lines 688–693)

GPT-2-small IOI accuracy under ablation (mean over 4 seeds):

| ablated set | clean | −prim. | +CoAx | +own | +rand | +doc. |
|---|---|---|---|---|---|---|
| IOI accuracy | 1.00 | 0.97 | **0.70** | 0.24 | 0.81 | 0.72 |

> Ablating the documented name-mover primaries barely dents IOI accuracy (**1.00 → 0.97**); behavior survives, fully masked. Adding the label-free CoAx backups brings it to **0.70, matching the documented-backup oracle (0.72)**. The first-order top-up (`+own`) overshoots to **0.24**, cutting past the backups into the core name-movers. CoAx recovers the *specific* compensators a complete knockout needs, not merely more of them. (Accuracy need not fall to chance — IOI retains redundancy beyond the name-mover family — so the informative quantity is the ordering across sets, not the absolute floor.)

### 4.3 Pruning: repair-aware removal order (Tables 26, 27)

A one-shot pruner scores every head independently, so self-repair hides a redundant group's *joint* importance (primary + backup each rate low → pruned together → behavior collapses). Fix: prune sequentially, re-measuring CoAx after each removal so a backup's importance rises the moment its primary leaves. Benefit in two layers:

1. **Co-ablation energy is already a strong standalone head score** — across 4 models (124M→7B) it beats every weight-, magnitude-, and gradient-based baseline, incl. gradient-Taylor wherever Taylor is defined (50%-sparsity PPL **80.6 vs 201.4** on GPT-2-small, margin holding on GPT-2-large + Pythia-1.4B).
2. **Re-measuring sequentially adds a further self-repair-aware gain** that widens with sparsity (**80.6 vs 112.6** for the static order on GPT-2-small) — and because the two orders share the identical signal and differ only in conditioning, this increment **isolates the value of conditioning itself**.

**Table 26 — WikiText-2 PPL on GPT-2-small (dense 47.9), full 10–70% sweep, all 6 orders (verbatim, lines 2545–2555):**

| order (PPL ↓) | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|
| random | 78 | 109 | 89 | 214 | 373 | 417 | 538 |
| magnitude | 1428 | 1016 | 4884 | 164 | 247 | 239 | 443 |
| Wanda | 51 | 609 | 2810 | 192 | 337 | 548 | 664 |
| Taylor | 50 | 59 | 70 | 97 | 201 | 568 | 23155 |
| co-abl. static | 50 | 56 | 65 | 85 | 113 | 149 | 353 |
| **CoAx** | **50** | **54** | **60** | **70** | **81** | **105** | **172** |

> Magnitude/Wanda rows are non-monotonic in sparsity (Wanda 51, 609, 2810, 192, 337 over 10–50%) — dips reproduce across runs; not interpreted mechanistically, but show magnitude/Wanda adapted to head pruning give *unstable* orders (why full curves, not one budget, are reported). Co-ablation orders are smoothly monotone. Taylor is monotone and far stronger than magnitude/Wanda yet still trails co-ablation and collapses at 70%.

**Table 27 — full 10–70% sweep on three further scales (verbatim, lines 2582–2609; best per column bolded in source):**

| order (PPL ↓) | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|
| **GPT-2-large (dense 29.6)** |||||||
| random | 31 | 34 | 40 | 45 | 117 | 277 | 488 |
| magnitude | 30 | 32 | 38 | 51 | 77 | 124 | 184 |
| Wanda | 31 | 37 | 63 | 88 | 210 | 316 | 653 |
| Taylor | 30 | 33 | 36 | 41 | 45 | 55 | 78 |
| co-abl. static | 30 | 31 | 32 | 36 | 44 | 55 | 91 |
| **CoAx** | 30 | 31 | 32 | 34 | 37 | 42 | 52 |
| **Pythia-1.4B (dense 21.9)** |||||||
| random | 25 | 38 | 108 | 158 | 491 | 1155 | 1302 |
| magnitude | 23 | 24 | 28 | 78 | 178 | 217 | 459 |
| Wanda | 22 | 25 | 149 | 171 | 356 | 450 | 558 |
| Taylor | 23 | 24 | 26 | 31 | 41 | 55 | 99 |
| co-abl. static | 23 | 24 | 25 | 30 | 38 | 46 | 66 |
| **CoAx** | 23 | 23 | 25 | 29 | 35 | 47 | 73 |
| **Qwen-2.5-7B (dense 16.2)** |||||||
| random | 46 | 173 | 601 | 1195 | 7544 | 18298 | 119377 |
| magnitude | 49 | 66 | 123 | 29613 | 65157 | 80280 | 79277 |
| Wanda | 46 | 66 | 29709 | 66182 | 79750 | 80023 | 78816 |
| Taylor | – | – | – | – | – | – | – |
| co-abl. static | 20 | 27 | 31 | 50 | 91 | 426 | 1341 |
| **CoAx** | 21 | 24 | 29 | 35 | 54 | 92 | 240 |

> Self-repair-aware gain over the static order grows with scale + sparsity: negligible on Pythia-1.4B (static already near-optimal, the two tie within noise at 60–70%), moderate on GPT-2-large (37 vs 44 at 50%, 52 vs 91 at 70%), largest on Qwen-7B (54 vs 91 at 50%, 240 vs 1341 at 70%) — larger models carry more backup redundancy to preserve. Taylor needs a backward pass and exceeds memory on Qwen-7B.

---

## 5. Efficiency, robustness, and what makes CoAx work

### 5.1 Compute cost — Tables 6, 7 (verbatim)

CoAx is forward-only and label-free; `2|U|+1` forwards per seed, parallelizable, never differentiates the model. Gradient baselines each need a backward pass + task metric yet top out at 0.82. Conditional form avoids the `O(|U|²)` explicit pairwise wall — ≈**36× saving** at GPT-2-small scale, growing with model size.

**Table 6 — wall-clock + memory, one GPU, 48 calibration prompts (lines 1148–1153):**

| model | units | single (s) | cond. (s) | GB |
|---|---|---|---|---|
| GPT-2-small | 144 | 2.9 | 5.7 | 0.8 |
| GPT-2-medium | 384 | 19 | 39 | 1.7 |
| Pythia-410m | 384 | 18 | 37 | 1.9 |
| Gemma-2-2b | 208 | 18 | 35 | 6.0 |
| GPT-Neo-1.3b | 384 | 64 | 128 | 5.6 |

**Table 7 — per-method cost (GPT-2-small, |U|=144, L=12; FPE = forwards + 2× backwards; lines 1227–1233):**

| method | forwards | backwards | FPE | label-free | backup AUC |
|---|---|---|---|---|---|
| single ablation (1st) | \|U\| | 0 | 144 | ✓ | 0.33 |
| AtP (1st) | 1 | 1 | 3 | — | 0.60 |
| EAP-IG (1st) | 5 | 5 | 15 | — | 0.70 |
| AtP\* GradDrop (1st) | 1 | L | 25 | — | 0.82 |
| explicit pairwise synergy (2nd) | \|U\|²/2 | 0 | ~10⁴ | ✓ | — † |
| **CoAx conditional (2nd, ours)** | **2\|U\|+1** | **0** | **289** | **✓** | **0.91** |

> † The explicit pairwise route carries the same second-order signal as conditional CoAx (Prop 3) but enumerates all pairs; CoAx recovers it at `O(|U|)` by conditioning on the seed once.

### 5.2 Data-cheap + the load-bearing design choice (Figure 7)

- **Data efficiency:** calibration-only, so already strong at **32 unlabeled prompts (0.90 AUC)**, saturates by ~64 (even 16 reach 0.86).
- **Feature-design ablation:** the one choice that matters is **centering** the features against the output distribution — worth **+0.11 AUC (0.91 → 0.80)** by removing the shared logit-shift direction that otherwise inflates every affinity; Fisher `√p_0` weighting over plain `ℓ_2` is a small consistent **+0.01**.
- **Alignment-invariance (Figure 7c):** on the synthetic benchmark, sweeping backup answer-alignment `β` shows CoAx's full-distribution energy is **flat in `β`** while an answer-direction GIM gradient degrades from above-CoAx (aligned backups) toward chance as backups move off the task metric. Measured real-IOI backups (`β ≈ 0`) sit at the off-answer end, where an answer-gradient is near-blind while CoAx holds.

### 5.3 Robustness to ablation value — Table 9 (verbatim, lines 1293–1295)

| | zero | mean | resample | Gaussian |
|---|---|---|---|---|
| single ablation (1st) | 0.33 | 0.33 | 0.39 | 0.35 |
| **CoAx (ours)** | **0.91** | **0.92** | **0.88** | **0.87** |

> Compensation AUC stays 0.87–0.92 across all four ablation values while single-ablation saliency stays 0.33–0.39: the CoAx ranking is a property of the conditional geometry, not of one ablation choice.

### 5.4 Controlled-redundancy synthetic benchmark — Table 12 (verbatim, lines 1605–1609)

100 planted conditionally-active backups, 40 independent trials, realistic observation noise (`β=0.45`). `p` = median DeLong `p` over the 40 trials.

| score (synthetic, n=100) | ROC-AUC ↑ | DeLong vs CoAx |
|---|---|---|
| **CoAx (conditional)** | **0.90 ± .02** | — |
| first-order energy (clean) | 0.42 ± .05 | p < 1e-15 |
| AtP\*-style gradient (clean) | 0.51 ± .04 | p < 1e-15 |
| GIM-style (conditional) | 0.85 ± .04 | n.s. (β=0.45) |

> Clean-state scores sit at or below chance exactly as Prop 2 predicts (first-order energy **anti-ranks** dormant backups at 0.42); both conditional scores clear it. The conditional GIM proxy is close here *only* because these synthetic backups are answer-aligned — CoAx is alignment-invariant and beats it where the real IOI backups sit (off-answer, `β < 0.01`).

---

## 6. Generalization tables

### Table 21 — induction generalization on three scales (label-free; lines 2317–2325)

| | GPT-2-sm | GPT-2-md | GPT-2-lg |
|---|---|---|---|
| conditional drop, discovered | 0.89 | 0.17 | 0.07 |
| conditional drop, random | 0.05 | 0.04 | ≈0 |
| primaries-only drop | 0.27 | 0.20 | 0.25 |
| +discovered drop | 8.5 | 8.1 | 1.62 |
| +random drop | 0.81 | 0.43 | 0.29 |
| attribution factor | 32× | 40× | 6.5× |
| over matched-random | 10× | 19× | 5.7× |
| activation ratio | 1.04 | 1.06 | 1.10 |

> Output-norm wake-up is weaker than IOI's 1.21; induction backups take over *functionally*, so the conditional causal drop is the signature that transfers. (Robust to using detected rather than documented primaries on GPT-2-small: conditional drop 1.86 vs 0.26 random, attribution 6.6× and 4× over matched-random.)

### Table 22 — cross-architecture induction completion, fully label-free (8 models; lines 2367–2375)

Induction log-prob drop when each selector's heads are added to the primary ablation (over-primary factor in parentheses):

| model | prim.-only | +CoAx | +own | +random |
|---|---|---|---|---|
| Pythia-160M | 4.61 | 9.75 (2.1×) | 11.96 (2.6×) | 6.37 (1.4×) |
| Pythia-410M | 0.87 | 10.46 (12.1×) | 7.56 (8.7×) | 0.94 (1.1×) |
| Pythia-1.4B | 1.67 | 8.37 (5.0×) | 6.19 (3.7×) | 2.09 (1.2×) |
| GPT-Neo-1.3B | 1.59 | 3.84 (2.4×) | 12.34 (7.8×) | 2.32 (1.5×) |
| Gemma-2-2B | 2.85 | 7.43 (2.6×) | 10.05 (3.5×) | 3.52 (1.2×) |
| Qwen2.5-7B | 0.51 | 2.88 (5.6×) | 2.13 (4.2×) | 0.55 (1.1×) |
| OLMo-2-7B | 0.78 | 3.29 (4.2×) | 4.48 (5.8×) | 0.99 (1.3×) |
| Llama-3.1-8B | 1.02 | 5.85 (5.7×) | 1.98 (1.9×) | 1.37 (1.3×) |

> Attribution factor ranges **2.1× (Pythia-160M) to 12× (Pythia-410M)**; `+CoAx > +random` on all eight. Both `+CoAx` and `+own` far exceed random on every model (induction is genuinely redundant across homogeneous heads), and the two are comparable — CoAx strictly larger on 4/8 (decisively Llama-3.1-8B 5.7× vs 1.9×, and Qwen2.5-7B), smaller on the other 4. The discriminating "right heads, not more heads" result is IOI's (Table 4: `+own` overshoots to 0.24 vs CoAx 0.70); on induction the paper claims only label-free recovery of a load-bearing set.

---

## 7. Hyperparameters — Table 8 (verbatim, lines 1266–1278)

| knob | value |
|---|---|
| top-r logits | 192 (stable for r ∈ {96, 192, 384}) |
| discovery prompts | 96 IOI-exercised |
| calibration windows (pruning) | 48–64 |
| ablation value | zero (default; variants in Tab. 9) |
| interaction strength λ | 1.0 (0.5 for GQA models) |
| primary seed \|S\| | documented count (3 name-movers, 4 induction) |
| FFN-group size | 96 groups (GPT-2-small) |
| EAP-IG integration steps | 5 |
| prompt seeds | {42, 1, 8, 22} |
| random-control draws | 20 (attribution / knockout) |
| permutation shuffles | 10,000 |
| synthetic trials | 40 |

---

## 8. Strengths / Limitations / Verdict

**Strengths**
- **Principled, not procedural.** CoAx is grounded in a Fisher-geometry view of ablations with three propositions: Prop 1 (Fisher identity), Prop 2 (the additivity blind spot — any clean-state-invariant score provably conflates a pure backup with an inert unit), Prop 3 (conditional growth *is* seed synergy). The score is genuinely second-order, not a re-ablation.
- **Honest about what it is.** The paper is explicit that CoAx *completes* circuits, does not discover them from scratch (standalone peaks at 0.60 < seed-free AtP\* 0.82); that the fair comparison is the seeded GIM (0.63), not the headline AtP\* (0.82); and that co-activation (0.92) also finds the IOI backups but is correlational and collapses on movement circuits.
- **One set, three downstream wins, all label-free + forward-only.** Attribution 0.22→1.76, knockout 0.97→0.70 (= oracle 0.72, while +own overshoots to 0.24), pruning 50% PPL 80.6 vs Taylor 201.4 — and the sequential-vs-static increment (80.6 vs 112.6) cleanly isolates the value of *conditioning itself* since the signal is otherwise identical.
- **Scales.** `O(|U|)` forward-only, ≈36× cheaper than explicit pairwise synergy; transfers label-free across 8 models / 6 families (2.1×–12×); pruning wins 124M→7B.

**Limitations / scope**
- **Needs a primary seed.** CoAx is a completion method, conditioned on a known/discovered primary circuit `S`; it is not a standalone circuit finder.
- **Regime-specific.** Targets dormant-substitution redundancy (IOI). Where redundancy is shared among already-co-firing homogeneous heads (induction), CoAx is complementary, not uniquely identifying — `+own` is comparable there.
- **Head granularity only (so far).** Does not transfer to the MLP-dominated *greater-than* circuit (FFN-group probe only 1.5× over random, within 1 std) — a property of the circuit, not the unit.
- **Small ground truth.** IOI has only 8 documented backups; the 0.09 gap over seed-free AtP\* (0.82) is underpowered at that size (the paper leans on the controlled-redundancy synthetic benchmark with 100 planted backups, where CoAx 0.90 vs clean-state 0.42/0.51, p<1e-15, to get a powered comparison).

**Verdict.** A clean, well-motivated contribution: it formalizes self-repair as an *additivity failure* of component importance, gives a label-free `O(|U|)` score that provably separates dormant backups from inert units (Prop 2), and shows the one recovered set repairs attribution/knockout/pruning alike. The most citable single result is the **0.33 → 0.91 IOI backup ROC-AUC** at a *lower* compute cost than the gradient baselines it beats — and the framing that faithful interpretability of a robust model must be **conditional**: redundancy is structure to condition on, not noise to average out.
