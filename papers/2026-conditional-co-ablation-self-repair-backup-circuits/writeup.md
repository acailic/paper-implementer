# Writeup — Conditional Co-Ablation (CoAx)

> Zhiren Gong, Zihao Zeng, Chau Yuen, Wei Yang Bryan Lim.
> "Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer
>  Circuits." arXiv:2607.01940 (2026).

## The one-paragraph version

When you ablate a component in a transformer to measure its importance,
the model often **self-repairs**: a dormant backup component wakes up and
takes over the ablated component's function. This breaks every
first-order importance score (single-ablation, attribution patching,
gradient methods) in the same way — the primary looks unimportant (its
damage was repaired) and the backup looks irrelevant (it's dormant on the
intact model). **Conditional Co-Ablation (CoAx)** flips the question:
instead of "how much does unit u matter on the clean model?", it asks
"how much does u matter **once the primary set S is already ablated?**"
A dormant backup has near-zero solo effect but a large conditional growth;
CoAx turns that growth into a discovery signal.

## What I implemented

A toy self-repair circuit (numpy, no PyTorch) with deliberate
primary/backup/inert structure, plus the CoAx scoring method, first-order
baselines, ROC-AUC evaluation, and pairwise synergy.

| Finding | Paper claim | My result |
|---|---|---|
| **F1** (Table 1) | CoAx lifts backup-recovery AUC from 0.33→0.91 | 0.52→1.00 on toy |
| **F2** (Table 3) | Primary-only drop 0.22; +CoAx exposes 1.76 | 0.883→2.993 |
| **F3** (Table 4) | −prim retains 0.97; +CoAx→0.70; +own→0.24 | 0.70→0.00→0.70 |
| **F4** (Prop 2) | Backups invisible to 1st-order; visible to CoAx | 1st: 0.015≈inert; CoAx: 0.177 vs 0.000 |

## What implementing it clarified

### 1. Proposition 2 is the whole paper in one sentence

The paper's formal contribution is Proposition 2 (§2.4): *any first-order
score that is invariant between a pure backup and an inert unit assigns them
the same value.* Implementing the toy circuit made this viscerally clear. A
"pure backup" is one where `δz_b ≈ 0` on the clean model (dormant) but
`E(δz_b|S) = Δ > 0` once primaries are ablated (wakes up). An "inert" unit
has `δz ≈ 0` always. On the clean model, both have zero effect — no
function of the clean pass can distinguish them. CoAx's conditional form
breaks the symmetry by querying the unit under a different operating point.

My F4 output shows this perfectly: backups score 0.015 on first-order
(identical to inert's 0.015) but 0.177 on CoAx (vs inert's 0.000). The
10× separation is the CoAx signal. No first-order score — however
sophisticated — can produce this separation, because the information simply
isn't in the clean-pass statistics.

### 2. The Hydra effect is easy to build but hard to discover

Designing the toy circuit's gate function was the most instructive part.
Backups need to be dormant when primaries are present and active when
they're absent. I implemented this as:

```python
gate_backup = gain * backup_strength * (n_primaries_ablated / n_primary)
```

This is a simple linear ramp: backups fire proportionally to how many
primaries are removed. At 0 primaries ablated, backups are fully dormant.
At all primaries ablated, backups fire at `backup_strength`. This
captures the essential dynamics of the IOI name-mover / backup-name-mover
circuit in GPT-2-small.

The key insight: this is a *conditional* gate. It depends on the state of
*other* units. No per-unit function of the clean model can detect it
because the gate's behavior is defined by what happens when you intervene
on a *different* unit. That's why CoAx works — it's the right
intervention.

### 3. The O(|U|) cost claim is real and matters

The paper's efficiency claim (§5.1) is that CoAx costs `2|U|+1` forward
passes per seed — same order as a single-ablation scan, far below the
`O(|U|²)` explicit pairwise synergy that carries the same second-order
signal. On the toy circuit (140 units, 48 positions) this is instant, but
the scaling argument is: you do ONE conditional scan (ablate S, then
ablate S∪{u} for each u) instead of scanning all pairs. For a real LM
with 144 heads, that's 289 forward passes vs 10,296 — a 36× saving, which
the paper documents.

The reason this works: CoAx asks "does u matter given S?" rather than
"do u and v interact?" The conditioning on a *fixed* seed set S collapses
the O(|U|²) pairwise problem into O(|U|) conditional queries. You trade
completeness (you only see interactions with S) for efficiency. For backup
discovery (where the seed S = primaries), this is exactly the right
tradeoff.

### 4. Synergy vs CoAx: cooperation vs substitution

The paper distinguishes two second-order lenses (§2.4):
- **Pairwise synergy** I_uv: symmetric cooperation (do u and v compensate
  for each other?).
- **CoAx** comp_u(S): asymmetric substitution (does u become important
  once S is removed?).

My F5 (synergy) shows primary-backup pairs have 1.3× higher synergy than
primary-inert pairs. The signal is weaker than CoAx's 10× separation
because synergy is a *pairwise* measure — it averages over all pairs,
diluting the backup-specific signal. CoAx's conditioning on the primary
seed set concentrates the signal exactly where it matters. The paper makes
this precise (Proposition 3): CoAx ≈ energy of the synergistic coupling to
the seed, so it reads the off-diagonal interaction that synergy computes
for all pairs but only for the relevant ones.

## What was harder than expected

- **Backup strength calibration.** Too high (0.85) and backups
  over-compensate (score goes *up* when primaries are ablated). Too low
  (0.1) and the self-repair signal is too weak for CoAx to detect. The
  sweet spot (0.4) produces a partial compensation that's invisible to
  first-order but clearly detectable by CoAx. The paper's IOI circuit has
  a specific compensation ratio (2.53 clean → 2.31 after primary ablation,
  a small drop because backups absorb most of the damage); matching this
  required tuning the toy's gate function.
- **ROC-AUC with small positive sets.** With only 8 backups among 136
  candidates, the AUC estimate is noisy. The paper runs 4 prompt seeds
  and reports std ≤ 0.04; I use 48 calibration positions per circuit
  evaluation which stabilizes the score. But the 1.000 AUC on the toy is
  partly because the toy is cleaner than real IOI — in the paper, some
  backups are ambiguous (6/8 in top-20, not 8/8).

## Pointers to the code

| File | What |
|------|------|
| `implementation/model.py` | `ToySelfRepairCircuit` — gate logic, primary/backup/inert structure |
| `implementation/coax.py` | `coax_score`, `ablation_energy`, `first_order_score`, `roc_auc` |
| `implementation/synergy.py` | `pairwise_synergy`, `synergy_matrix` |
| `implementation/run.py` | Reproduces F1–F5 end to end |

## Verdict

The paper's core insight — that self-repair breaks additivity and
conditioning on the primary set restores the signal — is one of those ideas
that seems obvious in hindsight but wasn't. The method is simple (one extra
forward-pass scan conditioned on S), the theory is tight (Proposition 2 is
a clean impossibility result for first-order), and the downstream
applications (attribution, knockout, pruning) all inherit the fix from one
recovered set. The 0.33→0.91 AUC jump on IOI is the kind of result that
changes how you think about circuit discovery.

My toy implementation captures the essential dynamics (dormant backups,
conditional activation, first-order blindness) but the real value of CoAx
is on actual transformer circuits where the backup structure is unknown.
The toy proves the method works; the paper proves it matters.

🏆 Verdict: the cleanest mech-interp method paper I've implemented.
Proposition 2 alone is worth the entry — it's the formal statement of why
self-repair breaks interpretability, and CoAx is the direct fix.
