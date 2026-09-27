"""Train + evaluate: the Combodied loop vs a naive task-maximizing loop.

Trains (1) the belief net q(Z|D) on event features and (2) the PWM
ensemble p(outcomes | belief, memory, state, action) on synthetic data,
then runs a longitudinal simulation per policy per person seed:

  - NaivePolicy      : argmax predicted P(take) -- pure task optimization
  - CombodiedPolicy  : admissibility filter + lexicographic vector utility
                       + intervention-response memory
  - NoAgent baseline : never intervene (ground for agency metrics)

Reports the paper's promised toy results (see README):
  (a) same action helps or harms depending on latent state
  (b) admissibility-gated policy matches task metrics while scoring far
      better on agency metrics
  (c) intervention-response memory improves adaptation over time
  (d) PWM calibration: predicted vs realized take-rates per action

Paper: "Combodied Agents: a New Paradigm of Human-Centric Agentic AI"
       Ding et al., 2026. arXiv:2608.10915

Run:  python3 train.py [--epochs 60] [--days 120] [--people 40]
"""

import argparse
import json
import random

import torch
import torch.nn.functional as F

from data import (PersonSim, make_belief_dataset, make_pwm_dataset,
                  N_ACTIONS, ACTION_NAMES, CAUSE_NAMES, observe_features,
                  true_response)
from model import (BeliefNet, PWMEnsemble, NaivePolicy, CombodiedPolicy,
                   InterventionMemory)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_belief(epochs=60, lr=1e-2, seed=0):
    """Train q(Z|D): event features -> cause posterior."""
    X, y = make_belief_dataset(n=4000, seed=seed)
    net = BeliefNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n, bs = X.shape[0], 256
    hist = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logp = torch.log(net(X[idx]).clamp_min(1e-9))
            loss = F.nll_loss(logp, y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot / n)
    with torch.no_grad():
        acc = (net(X).argmax(-1) == y).float().mean().item()
    return net, hist, acc


def train_pwm(epochs=60, lr=1e-2, seed=1):
    """Train the K-member PWM ensemble on synthetic (s, a) -> outcome data."""
    X, A, Y = make_pwm_dataset(n=6000, seed=seed)
    A_onehot = F.one_hot(A, N_ACTIONS).float()
    ens = PWMEnsemble(k=3)
    opt = torch.optim.Adam(ens.parameters(), lr=lr)
    n, bs = X.shape[0], 512
    hist = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = ens.losses(X[idx], A_onehot[idx], Y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot / n)
    return ens, hist


# ---------------------------------------------------------------------------
# Longitudinal evaluation
# ---------------------------------------------------------------------------

def run_policy(policy, belief_net, days, seed, collect_traj=False):
    """One longitudinal run: person sim + belief + PWM + policy loop."""
    gen = random.Random(seed)
    person = PersonSim(seed, gen)
    mem = InterventionMemory()
    taken = missed = acted = 0
    action_counts = [0] * N_ACTIONS
    correct_cause_top1 = 0
    n_decision_days = 0
    calib_pred = {a: [] for a in range(N_ACTIONS)}
    calib_real = {a: [] for a in range(N_ACTIONS)}
    traj = []
    for day in range(days):
        if not person.daily_event():
            continue
        n_decision_days += 1
        cause = person.sample_cause()
        feats = observe_features(cause, gen)
        x1 = torch.tensor(feats, dtype=torch.float32)
        z = belief_net(x1.unsqueeze(0))[0]            # Eq. 2 posterior
        correct_cause_top1 += int(z.argmax().item() == cause)
        # PWM input: belief + memory stats + person state (10 dims)
        pwm_in = torch.tensor(z.tolist() + mem.stats() +
                              [person.rel, person.self_cap],
                              dtype=torch.float32)
        a, info = policy.choose(pwm_in, mem, day, person.consecutive_untaken)
        action_counts[a] += 1
        if a != 0:
            acted += 1
        out = person.step(cause, a)
        mem.update(a, out["took"], day)               # Eq. 6 memory update
        calib_pred[a].append(pred_p_take(policy, pwm_in, a))
        calib_real[a].append(1.0 if out["took"] else 0.0)
        if out["took"]:
            taken += 1
        else:
            missed += 1
        if collect_traj and n_decision_days % 10 == 0:
            traj.append({
                "day": day,
                "action": ACTION_NAMES[a],
                "pwm_p_take": calib_pred[a][-1],
                "took": out["took"],
                "rel": round(person.rel, 3),
                "self_cap": round(person.self_cap, 3),
                "habit": round(person.habit, 3),
            })
    adh = taken / max(1, taken + missed)
    return {
        "adherence": round(adh, 3),
        "final_rel": round(person.rel, 3),
        "final_self_cap": round(person.self_cap, 3),
        "interventions_per_decision": round(acted / max(1, n_decision_days), 3),
        "n_decisions": n_decision_days,
        "action_mix": {ACTION_NAMES[a]: action_counts[a] for a in range(N_ACTIONS)},
        "belief_top1_acc": round(correct_cause_top1 / max(1, n_decision_days), 3),
        "calib": {ACTION_NAMES[a]: {
            "pred": round(sum(v) / len(v), 3) if v else None,
            "real": round(sum(r) / len(r), 3) if r else None,
        } for a, (v, r) in enumerate(zip(calib_pred.values(), calib_real.values()))},
        "traj": traj,
    }


def pred_p_take(policy, pwm_in, a):
    """Predicted P(take) for the chosen action (calibration tracking).

    All policies hold the PWMEnsemble, whose predict_action returns
    (p, d_rel, d_auto, unc) -- we only need p here.
    """
    p, _, _, _ = policy.pwm.predict_action(pwm_in, a)
    return round(p, 4)


def avg_over_people(policy_factory, belief_net, days, people, base_seed):
    agg = {}
    runs = [run_policy(policy_factory(), belief_net, days, base_seed + i)
            for i in range(people)]
    keys = ["adherence", "final_rel", "final_self_cap",
            "interventions_per_decision", "belief_top1_acc"]
    for k in keys:
        agg[k] = round(sum(r[k] for r in runs) / len(runs), 3)
    mix = {a: round(sum(r["action_mix"][a] for r in runs) / len(runs), 1)
           for a in ACTION_NAMES}
    agg["action_mix"] = mix
    return agg


class NoAgentPolicy:
    """Never intervene -- the with/without-agent baseline the paper demands."""

    def __init__(self, pwm):
        self.pwm = pwm

    def choose(self, x1, mem, day, consecutive_untaken):
        return 0, {}


# ---------------------------------------------------------------------------
# Result (a): same action, different latent state -> opposite effect
# ---------------------------------------------------------------------------

def action_state_dependence(pwm):
    """Show P(take | action) flips across the three latent causes."""
    rows = []
    for c, cname in enumerate(CAUSE_NAMES):
        logz = [-2.5] * 3
        logz[c] = 1.6
        z = torch.tensor(logz, dtype=torch.float32).softmax(-1)
        mem = [0.5] * N_ACTIONS
        row = {"cause": cname}
        for a, aname in enumerate(ACTION_NAMES):
            xin = torch.tensor(z.tolist() + mem + [1.0, 1.0])
            p, _, _, _ = pwm.predict_action(xin, a)
            row[aname] = round(p, 2)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--people", type=int, default=40)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    print("== Training belief net q(Z|D) ==")
    belief_net, bhist, bacc = train_belief(epochs=args.epochs)
    print(f"  final NLL {bhist[-1]:.4f}   held-out-free train acc {bacc:.3f}")

    print("== Training PWM ensemble (K=3) ==")
    pwm, phist = train_pwm(epochs=args.epochs)
    print(f"  final loss {phist[-1]:.4f}")

    print("\n== (a) Action effect depends on latent state (PWM, P(take)) ==")
    rows = action_state_dependence(pwm)
    hdr = f"  {'cause':<10}" + "".join(f"{a:<10}" for a in ACTION_NAMES)
    print(hdr)
    for r in rows:
        print(f"  {r['cause']:<10}" +
              "".join(f"{r[a]:<10}" for a in ACTION_NAMES))

    print(f"\n== Longitudinal sim: {args.days} days x {args.people} people ==")
    res = {}
    res["no_agent"] = avg_over_people(lambda: NoAgentPolicy(pwm), belief_net,
                                      args.days, args.people, 10_000)
    res["naive"] = avg_over_people(lambda: NaivePolicy(pwm), belief_net,
                                   args.days, args.people, 10_000)
    res["combodied"] = avg_over_people(lambda: CombodiedPolicy(pwm, belief_net),
                                       belief_net, args.days, args.people, 10_000)
    for name in ["no_agent", "naive", "combodied"]:
        r = res[name]
        print(f"  [{name:<10}] adherence {r['adherence']:.3f}  "
              f"rel {r['final_rel']:.3f}  self_cap {r['final_self_cap']:.3f}  "
              f"interv/dec {r['interventions_per_decision']:.3f}  "
              f"mix {r['action_mix']}")

    # (c) memory adaptation: first vs second half take-rate after 'remind'
    print("\n== (c) Intervention-response memory adaptation (combodied, 1 person) ==")
    run = run_policy(CombodiedPolicy(pwm, belief_net), belief_net,
                     args.days, 4242, collect_traj=True)
    half = len(run["traj"]) // 2
    if half:
        f = run["traj"][:half]
        s = run["traj"][half:]
        ftr = sum(t["took"] for t in f) / max(1, len(f))
        strr = sum(t["took"] for t in s) / max(1, len(s))
        print(f"  take-rate 1st half {ftr:.2f} -> 2nd half {strr:.2f} "
              f"(habit {'up' if s[-1]['habit'] >= f[-1]['habit'] else 'down'}: "
              f"{f[-1]['habit']} -> {s[-1]['habit']})")
    for t in run["traj"][:6]:
        print(f"   day {t['day']:>3} {t['action']:<9} p_take={t['pwm_p_take']:.2f} "
              f"took={t['took']} rel={t['rel']} cap={t['self_cap']} habit={t['habit']}")

    # (d) calibration per action on one combodied run
    print("\n== (d) PWM calibration: predicted vs realized P(take) ==")
    for aname, c in run["calib"].items():
        if c["pred"] is not None:
            print(f"  {aname:<9} pred {c['pred']:.3f}   realized {c['real']:.3f}")

    out = {
        "belief_acc": round(bacc, 3),
        "belief_loss": round(bhist[-1], 4),
        "pwm_loss": round(phist[-1], 4),
        "state_dependence": rows,
        "policies": res,
        "memory_adaptation": run["traj"],
        "calibration": run["calib"],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
