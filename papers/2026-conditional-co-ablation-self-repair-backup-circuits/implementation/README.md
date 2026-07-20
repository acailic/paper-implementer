# Conditional Co-Ablation (CoAx) — Recovering Self-Repair Backups

From-scratch Python implementation of:

> Zhiren Gong, Zihao Zeng, Chau Yuen, Wei Yang Bryan Lim.
> "Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer
>  Circuits." arXiv:2607.01940 (2026).

The paper addresses a fundamental blind spot in mechanistic interpretability:
when a transformer **self-repairs** (the Hydra effect — a primary component
removed → a dormant backup takes over), first-order importance scores
misread *both* sides of the redundancy. CoAx fixes this by scoring units
**conditionally** — not "how much does unit u matter on the intact model?"
but "how much does it matter once the primary set S is already ablated?"

## Quick start

```bash
pip install numpy
python3 run.py
```

Output: five reproduced findings on a toy self-repair circuit.

## The CoAx score (Eq. 2)

```
comp_u(S) = E(δz_u | S) − E(δz_u | ∅)
```

where `δz_u|S = z_S − z_{S∪{u}}` is the conditional ablation effect. A
dormant backup has near-zero solo effect but a large conditional growth;
an inert unit has neither.

## What is implemented

| File | Purpose |
|------|---------|
| `model.py` | `ToySelfRepairCircuit`: a synthetic circuit with deliberate primary/backup/inert structure exhibiting the Hydra effect (self-repair). No real LM needed. |
| `coax.py` | CoAx score, first-order baseline, ROC-AUC evaluation, backup-recovery harness |
| `synergy.py` | Pairwise synergy (the cooperation lens, §2.2 Eq. 1) |
| `run.py` | Main runner: reproduces findings F1–F5 |

## Findings reproduced

**F1 (Table 1).** Backup-head recovery ROC-AUC: CoAx lifts from 0.52→1.00
(paper: 0.33→0.91 on GPT-2-small IOI). First-order single-ablation is near
chance because backups are dormant on the clean model.

**F2 (Table 3).** Attribution: ablating primaries alone drops the task score
by only 0.883 (masked by self-repair); adding CoAx backups exposes the true
combined effect of 2.993.

**F3 (Table 4).** Capability knockout: −primaries retains 70% of score;
−primaries−CoAx-backups brings it to ~0%; the first-order top-up fails to
knock out the circuit.

**F4 (Proposition 2).** Backups have ~0.015 first-order energy (identical
to inert units) but 0.177 CoAx energy — a 10× separation. First-order scores
provably cannot distinguish dormant backups from inert units; CoAx can.

**F5 (bonus).** Pairwise synergy: primary-backup pairs show higher synergy
than primary-inert pairs (same-circuit compensation structure).

## Toy circuit design

The paper works on GPT-2-small's IOI circuit (the only circuit with head-level
backup ground truth). We build a toy circuit with the same structure:

```
h = Σ_u gate_u(ablated) · w_u + noise

gate_primary   = 1 if not ablated
gate_backup    = 0 when primaries intact; ∝ backup_strength × fraction-primaries-removed
gate_inert     ≈ 0 always

w_primary/backup = task_direction + noise  (correlated — they write the answer)
w_inert          = random                  (uncorrelated with task)
```

This captures the essential phenomenon: backups are dormant until primaries
are removed, then they take over. The ground-truth backup set is known by
construction, enabling clean ROC-AUC evaluation.

## Known gaps / limitations

1. **Toy circuit, not a real transformer.** No attention, no tokenization,
   no real IOI prompts. The toy demonstrates the *method* and the *phenomenon*
   but not on real language-model circuits.
2. **Fisher geometry simplified.** The paper uses Fisher-weighted ablation
   energy (Proposition 1: KL-energy via the output distribution). We use
   direct ℓ_2 norm of logit perturbation (equivalent for uniform output).
3. **No gradient baselines.** The paper compares against AtP, EAP-IG, AtP\*
   (gradient-based methods). Our toy circuit has no gradients by design
   (it's a numpy circuit, not a PyTorch model), so we only compare against
   single-ablation first-order.
4. **No pruning experiment.** The paper's pruning results (Tables 26–27) need
   a real LM with perplexity evaluation.
