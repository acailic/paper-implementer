"""PIVOT-RL vs full-task policy gradient on ChainWorld.

> Paper: "Apodex 1.1: Scaling Agentic Intelligence for Complex Work"
> (arXiv:2608.23283). From-scratch miniature; see README.md for scope.

Three training arms (same policy architecture, same number of gradient
episodes, same REINFORCE-with-baseline + entropy):

  full     — PG on full tasks only (the vanilla agentic-RL baseline)
  pivot    — PIVOT-RL: 50% full tasks + 50% hinted localized continuation
             tasks cut at hindsight-detected pivots of recent failures
  randcut  — ablation: 50% full tasks + 50% hinted continuations cut at a
             RANDOM point of the SAME failed trajectories (isolates "cut at
             the failure-relevant state" from "train on continuations at all")

Continuation tasks replay the environment to the cut point from its manifest
(state restore), append a short directional hint (counts only, never file
ids, never the answer), and optimize the continuation return. Evaluation is
always on UNHINTED full tasks (hints are absent at inference, as the paper
requires).

Also runs the scripted Agent Team baseline (decompose -> delegate ->
asymmetric verify -> synthesize) with the budget its organized compute needs,
to mirror the paper's ReAct-vs-AT comparison — explicitly NOT compute-matched.
"""
from __future__ import annotations

import copy
import json
import random
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import (ChainWorld, TaskContract, agent_team_script, detect_pivot,
                  make_contract)
from model import PointerPolicy

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
N_FILES = 12
N_SIGNALS = 4
BUDGET = 14            # 2L+1 = 9 optimal; exhaustive N+L+1 = 17 > B
ROUNDS = 250           # training rounds per arm
EPS_FULL = 16          # full-task episodes per round
EPS_CONT = 16          # continuation episodes per round (pivot/randcut arms)
EVAL_EVERY = 10
EVAL_EPS = 100
SEED = 20261119
LR = 1e-3
ENTROPY_BETA = 0.01


def rollout(env: ChainWorld, policy: PointerPolicy, greedy: bool = False,
            hint: Optional[Dict] = None, rng: Optional[random.Random] = None):
    """Run one episode; store per-step features + actions (log-probs are
    recomputed under the CURRENT weights in pg_update — avoids stale graphs
    across Adam steps)."""
    rng = rng or random
    feats_ff, feats_gf, acts = [], [], []
    rews = []
    snaps: List[Dict] = []
    done = False
    info = {"event": "na"}
    while not done:
        ff, gf = env.features(hint)
        with torch.no_grad():
            logits = policy(torch.tensor(ff).unsqueeze(0),
                            torch.tensor(gf).unsqueeze(0))[0]
        if greedy:
            a = int(torch.argmax(logits))
        else:
            a = int(torch.multinomial(
                torch.softmax(logits, dim=-1), 1))
        feats_ff.append(ff)
        feats_gf.append(gf)
        acts.append(a)
        _, r, done, info = env.step(a)
        rews.append(r)
        snaps.append(env.snapshot())
    return {"ff": feats_ff, "gf": feats_gf, "acts": acts, "rews": rews,
            "actions": acts, "snaps": snaps,
            "success": bool(getattr(env, "success", False)),
            "event": info["event"], "env": env}


def batched_pg_update(trajs, opt: torch.optim.Optimizer,
                      policy: PointerPolicy) -> float:
    """One REINFORCE step over a batch of trajectories: returns computed per
    trajectory, advantages normalized across the batch, entropy bonus."""
    all_lp, all_ent, all_ret = [], [], []
    for traj in trajs:
        R, returns = 0.0, []
        for r in reversed(traj["rews"]):
            R = r + R  # undiscounted return from each step
            returns.append(R)
        returns.reverse()
        logits = policy(torch.tensor(traj["ff"], dtype=torch.float32),
                        torch.tensor(traj["gf"], dtype=torch.float32))
        logp = torch.log_softmax(logits, dim=-1)
        lp = logp.gather(1, torch.tensor(traj["acts"]).unsqueeze(1)).squeeze(1)
        ent = -(logp.exp() * logp).sum(-1)
        all_lp.append(lp)
        all_ent.append(ent)
        all_ret.append(torch.tensor(returns))
    ret = torch.cat(all_ret)
    adv = (ret - ret.mean()) / (ret.std() + 1e-6)
    lp = torch.cat(all_lp)
    ent = torch.cat(all_ent)
    loss = -(adv.detach() * lp).sum() - ENTROPY_BETA * ent.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(ret.mean())


def run_arm(arm: str, rounds: int = ROUNDS, seed: int = SEED) -> Dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    policy = PointerPolicy(N_FILES)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    fail_buf: List[Dict] = []        # recent failed rollouts (full tasks)
    history = []
    eps_trained = 0

    for rnd in range(rounds):
        batch: List[Dict] = []
        # ---- full-task episodes (all arms) --------------------------------
        for _ in range(EPS_FULL):
            env = ChainWorld(make_contract(rng.randrange(1 << 30),
                                           N_FILES, N_SIGNALS, BUDGET))
            env.reset()
            traj = rollout(env, policy, greedy=False, rng=rng)
            batch.append(traj)
            eps_trained += 1
            if not traj["success"]:
                fail_buf.append(traj)
                if len(fail_buf) > 64:
                    fail_buf.pop(0)

        # ---- continuation episodes (pivot / randcut arms) ------------------
        if arm in ("pivot", "randcut") and fail_buf:
            n_cont = 0
            attempts = 0
            while n_cont < EPS_CONT and attempts < EPS_CONT * 3 and fail_buf:
                attempts += 1
                traj = fail_buf[rng.randrange(len(fail_buf))]
                if arm == "pivot":
                    piv = detect_pivot(traj["env"], traj["actions"],
                                       traj["snaps"])
                    if piv is None:
                        continue
                    cut, category, hint = piv
                else:  # random cut on the same failed trajectories
                    cut = rng.randrange(len(traj["actions"]))
                    snap_env = ChainWorld(traj["env"].c)
                    snap_env.restore(traj["snaps"][cut])
                    rem = traj["env"].c.budget - snap_env.steps
                    if rem < 0 or snap_env.poisoned or snap_env.delivered:
                        continue
                    hint = {"flag": 1.0,
                            "rem_sig": sum(1 for s in traj["env"].c.signal_ids
                                           if not snap_env.collected[s]),
                            "rem_b": float(rem)}
                    category = "random-cut"
                env = ChainWorld(traj["env"].c)
                env.reset()
                for a in traj["actions"][: cut + 1]:
                    env.step(a)
                if env.poisoned or env.delivered:
                    continue
                # directional hint; identical numeric format across arms
                ctraj = rollout(env, policy, greedy=False, hint=hint, rng=rng)
                ctraj["pivot_category"] = category
                batch.append(ctraj)
                eps_trained += 1
                n_cont += 1

        # ---- one batched policy-gradient step over the round ----------------
        if batch:
            batched_pg_update(batch, opt, policy)

        # ---- eval (unhinted full tasks, greedy) ----------------------------
        if (rnd + 1) % EVAL_EVERY == 0 or rnd == rounds - 1:
            ev = evaluate(policy, seed + 777)
            sr = ev["sr"]
            history.append({"round": rnd + 1, "eval_sr": sr,
                            "eps": eps_trained})
            print(f"  [{arm:8s}] round {rnd + 1:3d}  eval SR {sr:5.1f}%  "
                  f"(eps trained {eps_trained})", flush=True)

    final = evaluate(policy, seed + 424242)
    eps_to_90 = next((h["eps"] for h in history if h["eval_sr"] >= 90.0),
                     None)
    return {"arm": arm, "final_sr": final["sr"],
            "best_sr": max(h["eval_sr"] for h in history),
            "eps_to_90": eps_to_90, "final": final,
            "history": history, "eps_trained": eps_trained}


def evaluate(policy: PointerPolicy, seed: int, eps: int = EVAL_EPS) -> Dict:
    rng = random.Random(seed)
    succ, steps_ok, poisoned, stalls = 0, [], 0, 0
    for _ in range(eps):
        env = ChainWorld(make_contract(rng.randrange(1 << 30), N_FILES,
                                       N_SIGNALS, BUDGET))
        env.reset()
        traj = rollout(env, policy, greedy=True, rng=random.Random(0))
        if traj["success"]:
            succ += 1
            steps_ok.append(len(traj["actions"]))
        if getattr(env, "poisoned", False):
            poisoned += 1
        if traj["event"] == "budget-exhausted":
            stalls += 1
    return {"sr": 100.0 * succ / eps,
            "mean_steps_success": (sum(steps_ok) / len(steps_ok)
                                   if steps_ok else None),
            "poisoned_pct": 100.0 * poisoned / eps,
            "stall_pct": 100.0 * stalls / eps}


def main():
    t0 = time.time()
    print("=" * 72)
    print("Apodex 1.1 miniature — ChainWorld task contract + AgentOS runtime")
    print(f"N={N_FILES} files, L={N_SIGNALS} load-bearing, B={BUDGET} tool "
          f"calls | rho_acq={(N_FILES + N_SIGNALS) / BUDGET:.3f} "
          "(exhaustive N+L+1=17 > B; provenance-follow 2L+1=9 <= B)")
    print("=" * 72)

    import data
    data.selftest()

    results = {}
    for arm in ("full", "pivot", "randcut"):
        print(f"\n--- arm: {arm} ---", flush=True)
        results[arm] = run_arm(arm)

    # Agent Team script (organized compute; NOT compute-matched)
    rng = random.Random(SEED)
    at_succ, at_steps = 0, []
    for _ in range(200):
        c = make_contract(rng.randrange(1 << 30), N_FILES, N_SIGNALS,
                          budget=32)
        out = agent_team_script(c)
        at_succ += out["success"]
        at_steps.append(out["steps"])
    results["agent_team_script"] = {
        "sr": 100.0 * at_succ / 200,
        "mean_steps": sum(at_steps) / len(at_steps),
        "budget_used": 32,
        "note": "reads ALL N files (organized parallel search) — needs "
                "N+L(+1 verify)+1 tool calls; ReAct arms get B=14",
    }

    print("\n" + "=" * 72)
    print(f"{'arm':12s} {'final SR':>9s} {'best SR':>8s} "
          f"{'steps@succ':>11s} {'poison%':>8s} {'stall%':>7s}")
    print("-" * 72)
    for arm in ("full", "pivot", "randcut"):
        r, f = results[arm], results[arm]["final"]
        ms = f"{f['mean_steps_success']:.1f}" if \
            f["mean_steps_success"] is not None else "n/a"
        print(f"{arm:12s} {r['final_sr']:8.1f}% {r['best_sr']:7.1f}% "
              f"{ms:>11s} {f['poisoned_pct']:7.1f}% {f['stall_pct']:6.1f}%")
    at = results["agent_team_script"]
    print(f"{'AT-script':12s} {at['sr']:8.1f}% {'':>8s} "
          f"{at['mean_steps']:.1f}".ljust(30)
          + f"   (budget 32, not compute-matched)")
    print("=" * 72)

    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nresults.json written | wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
