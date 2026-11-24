"""MiniVBVR — training driver: SFT -> GRPO RL (RLVR vs RLVLM) -> sampler study.

> Paper: *VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning*
> Junxiang Xu et al., 2026. arXiv:2608.26105 — https://arxiv.org/abs/2608.26105

Reproduces the paper's three-arm comparison in miniature (its Tab. 8 + App. B.5):

  Baseline (no training)  vs  SFT  vs  RLVLM (noisy judge, bg-blind)
  vs  RLVR (deterministic scorer incl. background gate)

all under matched CPS exploration (eta=0.7), plus the sampler ablation
(ODE vs CPS vs Flow-SDE, App. B.4) and a judge-reproducibility audit.
Run:  python train.py          (~2 min, CPU only)
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import torch

from data import make_batch, N_FRAMES, H, W
from model import MiniFlowNet, rollout, grpo_step
from scorers import score_full, judge_reward

torch.manual_seed(0)


def frames_as_video(flat: torch.Tensor, B: int):
    """(B,FLAT) in [0,1] -> (B,N,3,H,W)."""
    from data import H, W
    return flat.view(B, N_FRAMES, 3, H, W)


def evaluate(model, insts, scenes, instr, gen):
    """Deterministic ODE rollout scored with the FULL scorer (incl. g_bg)."""
    model.eval()
    with torch.no_grad():
        flat, _ = rollout(model, scenes, instr, "ode", 0.0, 10, gen=gen,
                          frames_only=True)
    vids = frames_as_video(flat, len(insts))
    details = [score_full(i, vids[b]) for b, i in enumerate(insts)]
    score = sum(d["score"] for d in details) / len(details)
    soft = sum(d["soft"] for d in details) / len(details)
    bgf = sum(d["bg_fidelity"] for d in details) / len(details)
    return {"score": score, "soft": soft, "bg_fidelity": bgf}


def rl_loop(model, insts, scenes, instr, reward_fn, iters, group, eta, lr, gen,
            eval_fn, eval_every, label):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    n_prompts = scenes.shape[0]
    for it in range(iters):
        model.train()
        rollouts = []
        idx = torch.randperm(n_prompts)[:8]
        # group rollouts: same prompt, independent CPS trajectories
        for p in idx.tolist():
            sc = scenes[p:p + 1].expand(group, -1)
            ins = instr[p:p + 1].expand(group, -1)
            with torch.no_grad():  # rewards/replay need no graph; grpo replays
                flat, recs = rollout(model, sc, ins, "cps", eta, 10, gen=gen)
            vids = frames_as_video(flat, group)
            for g in range(group):
                # records were appended [step0: g0..g3, step1: g0..g3, ...]
                rollouts.append({"records": recs[g::group],
                                 "reward": reward_fn(insts[p], vids[g]),
                                 "group_id": p, "scene": scenes[p],
                                 "instr": instr[p]})
        loss, mab = grpo_step(model, opt, rollouts)
        if (it + 1) % eval_every == 0 or it == 0:
            ev = eval_fn(model)
            hist.append({"iter": it + 1, **ev})
            print(f"  [{label}] iter {it+1:3d} | loss {loss:8.2f} | |A| {mab:.2f} | "
                  f"score {ev['score']:.3f} soft {ev['soft']:.3f} bg_fid {ev['bg_fidelity']:.3f}")
    return hist


def sampler_study(model, insts, scenes, instr, gen, group=8, eta=0.7):
    """Paper B.4 in miniature: ODE vs CPS vs Flow-SDE on a fixed policy.
    Diversity is measured in REWARD space (std of scorer outputs within a
    group) — the GRPO-relevant quantity: if all rollouts earn the same reward,
    group-relative advantages are zero and there is no gradient. Pixel-space
    spread is reported too, but on this near-unimodal toy it mostly tracks
    residual noise passthrough, not decisions."""
    out = {}
    for name in ("ode", "cps", "sde"):
        with torch.no_grad():
            flat, _ = rollout(model, scenes, instr, name, eta, 10, gen=gen,
                              frames_only=True)
        vids = frames_as_video(flat, len(insts))
        sc = [score_full(i, vids[b]) for b, i in enumerate(insts)]
        out[name] = {"score": sum(d["score"] for d in sc) / len(sc),
                     "bg_fidelity": sum(d["bg_fidelity"] for d in sc) / len(sc)}
    for name in ("ode", "cps", "sde"):
        r_stds, px_divs, degenerate = [], [], 0
        for p in range(min(8, len(insts))):
            sc = scenes[p:p + 1].expand(group, -1)
            ins = instr[p:p + 1].expand(group, -1)
            with torch.no_grad():
                flat, _ = rollout(model, sc, ins, name, eta, 10, gen=gen,
                                  frames_only=True)
            vids = frames_as_video(flat, group)
            rs = [score_full(insts[p], vids[g])["score"] for g in range(group)]
            mu = sum(rs) / group
            sd = (sum((x - mu) ** 2 for x in rs) / group) ** 0.5
            r_stds.append(sd)
            degenerate += int(sd < 1e-3)   # GRPO zero-advantage group
            d = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0)
            px_divs.append(d[torch.triu(torch.ones_like(d), 1) > 0].mean().item())
        out[name]["reward_group_std"] = sum(r_stds) / len(r_stds)
        out[name]["degenerate_group_frac"] = degenerate / len(r_stds)
        out[name]["pixel_diversity"] = sum(px_divs) / len(px_divs)
    return out


def judge_audit(insts, vids, n=48):
    """Re-score identical outputs twice with the judge -> % changed (paper:
    judges flip 54.6-92.8% of identical re-evaluations)."""
    gen = torch.Generator().manual_seed(7)
    flips = 0
    for b, i in enumerate(insts[:n]):
        r1 = judge_reward(i, vids[b], gen)
        r2 = judge_reward(i, vids[b], gen)
        flips += int(abs(r1 - r2) > 1e-6)
    return flips / min(n, len(insts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-steps", type=int, default=3000)
    ap.add_argument("--rl-iters", type=int, default=40)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--eta", type=float, default=0.7)
    ap.add_argument("--lr-rl", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()

    # ---- data: train pool + fixed eval set (train/test split safety) ----
    train_insts, tr_scenes, tr_frames, tr_instr = make_batch(1024, seed=100)
    eval_insts, ev_scenes, _, ev_instr = make_batch(24, seed=999)
    ev_scenes_f, ev_instr_f = ev_scenes.reshape(24, -1), ev_instr
    print(f"data: {len(train_insts)} train / {len(eval_insts)} eval instances "
          f"({sum(1 for i in train_insts if i.task=='maze')} maze in train)")

    model = MiniFlowNet(hidden=768)
    nparam = sum(p.numel() for p in model.parameters())

    # ---- baseline (no training) ----
    base = evaluate(model, eval_insts, ev_scenes_f, ev_instr_f, gen)
    print(f"baseline (untrained): score {base['score']:.3f} soft {base['soft']:.3f} "
          f"bg_fid {base['bg_fidelity']:.3f} | params {nparam/1e6:.2f}M")

    # ---- SFT: x0-prediction flow matching on solver ground truth ----
    # Attribute-weighted MSE: agent/target pixels are ~6% of each frame, so a
    # plain pixel loss lets the net ignore them (the paper's scorers likewise
    # care about key elements, not average pixels). w: agent x5, walls x2.
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.sft_steps,
                                                       eta_min=1e-4)
    gt = tr_frames.reshape(1024, -1)
    scene_f = tr_scenes.reshape(1024, -1)
    from data import AGENT as AGENT_C, WALL as WALL_C
    agent_t = torch.tensor(AGENT_C)
    wall_t = torch.tensor(WALL_C)
    for step in range(args.sft_steps):
        idx = torch.randint(0, 1024, (32,))
        t = torch.rand(32) * 0.99 + 0.01
        x0 = gt[idx]
        eps = torch.randn(32, x0.shape[1], generator=gen)
        x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * eps
        px = x0.view(32, N_FRAMES, H, W, 3)          # (B,N,H,W,3) per-pixel colors
        w = 1.0 + 4.0 * ((px - agent_t).abs().sum(-1) < 0.6).float() \
            + 1.0 * ((px - wall_t).abs().sum(-1) < 0.6).float()
        w = w.unsqueeze(2).expand(32, N_FRAMES, 3, H, W).reshape(32, -1)  # -> (B,FLAT)
        pred = model(x_t, scene_f[idx], tr_instr[idx], t)
        loss = (w * (pred - x0) ** 2).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            print(f"  [sft] step {step+1}: weighted x0-MSE {loss.item():.5f}")
    sft = evaluate(model, eval_insts, ev_scenes_f, ev_instr_f, gen)
    print(f"SFT: score {sft['score']:.3f} soft {sft['soft']:.3f} bg_fid {sft['bg_fidelity']:.3f}")

    # ---- RL arms from the same SFT init (paper: matched everything but reward) ----
    eval_fn = lambda m: evaluate(m, eval_insts, ev_scenes_f, ev_instr_f, gen)
    rlvr = copy.deepcopy(model)
    rlvlm = copy.deepcopy(model)
    jgen = torch.Generator().manual_seed(3)
    print("RL arm 1/2: RLVR (deterministic scorer, includes background gate)")
    h_rlvr = rl_loop(rlvr, train_insts, scene_f, tr_instr,
                     lambda i, v: score_full(i, v)["score"],
                     args.rl_iters, args.group, args.eta, args.lr_rl, gen,
                     eval_fn, eval_every=5, label="RLVR")
    print("RL arm 2/2: RLVLM (judge reward: bg-blind + noise — the control)")
    h_rlvlm = rl_loop(rlvlm, train_insts, scene_f, tr_instr,
                      lambda i, v: judge_reward(i, v, jgen),
                      args.rl_iters, args.group, args.eta, args.lr_rl, gen,
                      eval_fn, eval_every=5, label="RLVLM")

    final = {"RLVR": eval_fn(rlvr), "RLVLM": eval_fn(rlvlm)}

    # ---- sampler study on the SFT policy (ODE vs CPS vs SDE, eta=0.7) ----
    sam = sampler_study(model, eval_insts, ev_scenes_f, ev_instr_f, gen)

    # ---- judge reproducibility audit on SFT outputs ----
    with torch.no_grad():
        flat, _ = rollout(model, ev_scenes_f, ev_instr_f, "ode", 0.0, 10, gen=gen,
                          frames_only=True)
    vids = frames_as_video(flat, 24)
    flip_rate = judge_audit(eval_insts, vids)

    # ---- report ----
    print("\n================ MiniVBVR results ================")
    print(f"arm      | score | soft  | bg_fid   (matched CPS eta={args.eta})")
    for name, ev in [("baseline", base), ("SFT", sft), ("RLVR", final["RLVR"]),
                     ("RLVLM", final["RLVLM"])]:
        print(f"{name:8s} | {ev['score']:.3f} | {ev['soft']:.3f} | {ev['bg_fidelity']:.3f}")
    print("\nsampler (SFT policy):  score / bg_fid / reward-group-std / degenerate-frac")
    for name, d in sam.items():
        print(f"  {name:4s}: {d['score']:.3f} / {d['bg_fidelity']:.3f} / "
              f"{d['reward_group_std']:.4f} / {d['degenerate_group_frac']:.2f}")
    print(f"\njudge audit: identical outputs re-scored differently {flip_rate*100:.1f}% "
          f"of the time (paper: 54.6-92.8%)")
    print(f"total wall time: {time.time()-t0:.1f}s")

    json.dump({"baseline": base, "sft": sft, "final": final, "sampler": sam,
               "judge_flip_rate": flip_rate, "rlvr_hist": h_rlvr,
               "rlvlm_hist": h_rlvlm, "wall_s": time.time() - t0,
               "config": vars(args)}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
