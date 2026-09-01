"""AgentOPSD training loop — from-scratch re-implementation.

Paper: "AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement
Learning" (Wang et al., 2026, arXiv:2608.05987).

Runs two arms on the synthetic pivotal-turn task (data.py):
  * GRPO   — plain group-relative advantage, broadcast to every turn;
  * AgentOPSD — the full recursive belief reshaper (Eqs. 6-11).

Reports per-iteration: success rate, entropy, approx-KL, and (the metric the
real benchmarks can't give) credit localization: Spearman correlation
between assigned per-turn credit and the ORACLE credit (1 at pivotal turns).
Also runs a small ablation over gamma at the end.

Usage:  python3 train.py [--iters 30] [--seed 0]
"""

import argparse
import copy
import time

import numpy as np
import torch

from data import sample_batch
from model import (TinyPolicy, rollout, rollout_group, grpo_advantage,
                   group_prior, agentopsd_advantages, surrogate_loss, spearman)

N_OBS, N_ACT, D_MODEL = 6, 4, 64
FAMILY_SEED = 7  # fixes the family-level obs->action mapping (the "domain skill")
TASK_KW = dict(n_turns=8, n_pivotal=2)


def make_envs(n_tasks: int, seed: int):
    # family_seed fixed => all instances (train + eval) share the obs->action
    # mapping; the student must internalize it from outcomes alone.
    return sample_batch(n_tasks, family_seed=FAMILY_SEED, seed=seed, **TASK_KW)


def evaluate(policy, envs, device, n_rep: int = 3):
    """Greedy success rate over envs (n_rep passes to average noise)."""
    policy.eval()
    hits: int = 0
    with torch.no_grad():
        for env in envs:
            for _ in range(n_rep):
                traj = rollout(policy, env, device, greedy=True)
                hits += traj["reward"]
    return hits / (len(envs) * n_rep)


def train_arm(name: str, use_agentopsd: bool, iters: int, seed: int,
              gamma=1.0, lam=0.7, b=0.5, group_size=8, n_tasks_per_iter=4,
              lr=1e-3, entropy_coef=0.01, log_every=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    policy = TinyPolicy(N_OBS, N_ACT, D_MODEL)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # fixed eval env set (unseen seeds) to track generalization
    eval_envs = make_envs(20, seed=seed + 999)

    print(f"\n=== arm: {name} (gamma={gamma}, lam={lam}, b={b}) ===")
    print(f"{'iter':>4} {'succ':>6} {'loss':>8} {'ent':>6} {'kl':>7} {'loc':>6} {'sec':>5}")
    t0 = time.time()
    for it in range(1, iters + 1):
        envs = make_envs(n_tasks_per_iter, seed=seed * 1000 + it)
        losses, ents, kls = [], [], []
        credits_all, oracle_all = [], []
        for env in envs:
            group = rollout_group(policy, env, device, group_size)
            rewards = [t["reward"] for t in group]
            A = grpo_advantage(rewards)          # (G,) outcome advantage
            B0 = group_prior(rewards)            # belief prior
            for i, traj in enumerate(group):
                diag = None
                if use_agentopsd:
                    traj["B0"] = B0
                    adv, diag = agentopsd_advantages(
                        traj, float(A[i]), gamma=gamma, lam=lam, b=b)
                else:
                    adv = np.full(len(traj["actions"]), A[i])
                loss, ent, kl = surrogate_loss(policy, env, traj, adv, device,
                                               entropy_coef=entropy_coef)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                ents.append(float(ent.item()))
                kls.append(float(kl.item()))
                # credit localization (oracle available only in synthetic env)
                credit = diag["q"] if diag is not None else np.full(env.T, A[i])
                credits_all.append(credit)
                oracle_all.append(env.oracle_credit())
        if it % log_every == 0 or it == 1 or it == iters:
            succ = evaluate(policy, eval_envs, device)
            loc = np.mean([spearman(c, o) for c, o in zip(credits_all, oracle_all)])
            print(f"{it:>4} {succ:6.3f} {np.mean(losses):8.4f} {np.mean(ents):6.3f} "
                  f"{np.mean(kls):7.4f} {loc:6.3f} {time.time()-t0:5.1f}")

    final = evaluate(policy, eval_envs, device, n_rep=5)
    print(f"[{name}] final greedy success: {final:.3f}")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("AgentOPSD re-implementation — synthetic pivotal-turn task "
          f"(T=8 turns, {N_ACT} actions, 2 pivotal turns, {N_OBS} obs types)")
    print("Paper: Wang et al. 2026, arXiv:2608.05987 — this is a toy task, not ALFWorld.")

    grpo = train_arm("GRPO", use_agentopsd=False, iters=args.iters, seed=args.seed)
    opsd = train_arm("AgentOPSD", use_agentopsd=True, iters=args.iters, seed=args.seed)

    print("\n=== summary ===")
    print(f"GRPO      final success: {grpo:.3f}")
    print(f"AgentOPSD final success: {opsd:.3f}")

    # small gamma ablation (paper reports gamma matters; ours: gamma=1 is SPRT)
    if args.iters >= 20:
        print("\n=== gamma ablation (AgentOPSD) ===")
        for g in (0.5, 0.8, 1.0):
            train_arm(f"OPSD-gamma={g}", True, iters=args.iters, seed=args.seed, gamma=g)


if __name__ == "__main__":
    main()
