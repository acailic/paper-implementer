"""
OmniOpt demo pipeline.

1. Instantiate 4 optimizers from 3 families via unified meta-pipeline
2. Train identical MLP on synthetic binary classification
3. Compare: convergence (O1), step cost (O2), memory (O3)
4. Print taxonomy mapping + benchmark table

Paper: Xu et al. (2026), "OmniOpt: Taxonomy, Geometry, and Benchmarking of Modern Optimizers",
arXiv:2607.04033.
"""

import torch
import numpy as np

from data import generate_data
from model import MLP, SGDM, AdamW, Lion, Muon, train_with_optimizer, taxonomy_table


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("OmniOpt: Meta-Pipeline Optimizer Demo")
    print("=" * 60)

    # ── 1. Generate data ──
    print("\n[1] Generating synthetic data (d=32, N=2000)")
    X, y = generate_data(n_samples=2000, d_in=32, seed=42)
    print(f"    X: {X.shape}, y: {y.shape}, balance: {y.mean():.3f}")

    # ── 2. Taxonomy table ──
    print("\n[2] Meta-Pipeline Taxonomy (Table 4/5)")
    taxonomy_table()

    # ── 3. Benchmark ──
    print("\n[3] Benchmark: 500 steps, batch=256, MLP(32→32→1)")
    print("-" * 60)

    model = MLP(d_in=32)
    steps = 500

    configs = [
        ("SGDM",  SGDM,  {"lr": 0.01,  "beta1": 0.9}),
        ("AdamW", AdamW, {"lr": 1e-3,  "beta1": 0.9, "beta2": 0.999}),
        ("Lion",  Lion,  {"lr": 5e-4,  "beta1": 0.9, "beta2": 0.99}),
        ("Muon",  Muon,  {"lr": 0.02,  "beta1": 0.95, "ns_steps": 5}),
    ]

    results = {}
    for name, cls, kwargs in configs:
        print(f"\n  Training with {name} (family={cls.family})...")
        res = train_with_optimizer(
            model, cls, X, y,
            steps=steps, batch_size=256,
            lr=kwargs["lr"], wd=0.0,
            **{k: v for k, v in kwargs.items() if k != "lr"},
        )
        results[name] = res
        print(f"    Final loss: {res['final_loss']:.4f}  "
              f"Min loss: {res['min_loss']:.4f}  "
              f"Step: {res['step_time_ms']:.3f}ms  "
              f"Mem: {res['state_mem_kb']:.1f}KB")

    # ── 4. Comparison table ──
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON (O1: convergence, O2: cost, O3: memory)")
    print("=" * 60)
    print(f"{'Optimizer':<8} {'Family':<4} {'Final Loss':>10} {'Min Loss':>9} "
          f"{'Step (ms)':>9} {'State (KB)':>10}")
    print("-" * 60)
    for name, cls, _ in configs:
        r = results[name]
        print(f"{name:<8} {cls.family:<4} {r['final_loss']:>10.4f} {r['min_loss']:>9.4f} "
              f"{r['step_time_ms']:>9.3f} {r['state_mem_kb']:>10.1f}")

    # ── 5. Family-level summary ──
    print("\n" + "=" * 60)
    print("FAMILY-LEVEL SUMMARY (mirrors paper's Figure 19)")
    print("=" * 60)

    families = {}
    for name, cls, _ in configs:
        fam = cls.family
        if fam not in families:
            families[fam] = []
        families[fam].append(results[name])

    print(f"{'Family':<25} {'Avg Loss':>9} {'Avg Step(ms)':>12} {'Avg Mem(KB)':>12}")
    print("-" * 60)
    for fam in sorted(families.keys()):
        rs = families[fam]
        avg_loss = sum(r["final_loss"] for r in rs) / len(rs)
        avg_time = sum(r["step_time_ms"] for r in rs) / len(rs)
        avg_mem = sum(r["state_mem_kb"] for r in rs) / len(rs)
        print(f"{fam:<25} {avg_loss:>9.4f} {avg_time:>12.3f} {avg_mem:>12.1f}")

    print("\n" + "=" * 60)
    print("KEY PAPER FINDINGS REFLECTED IN DEMO")
    print("=" * 60)
    print("• T1 (AdamW): Strong convergence, moderate cost — the stable reference")
    print("• T3 (Lion): Cheapest step cost (sign ops only), slightly weaker convergence")
    print("• T2 (Muon): Spectral orthogonalization via Newton-Schulz, needs higher LR")
    print("• No universal winner — pick optimizer by binding constraint (paper §7)")
    print("\n✓ Demo complete.")


if __name__ == "__main__":
    main()
