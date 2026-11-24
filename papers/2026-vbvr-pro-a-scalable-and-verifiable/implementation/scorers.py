"""MiniVBVR — deterministic attribute scorers + biased judge (VBVR-Pro Sec 3).

> Paper: *VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning*
> Junxiang Xu et al., 2026. arXiv:2608.26105 — https://arxiv.org/abs/2608.26105

Scorers grade EXTRACTED ATTRIBUTES (agent centroids, colors, wall hits),
never raw pixels, and aggregate exactly as the paper does:

    score = ( sum_i w_i c_i / sum_i w_i ) * prod_j g_j        (soft x hard gates)

The RLVLM "judge" mimics the paper's VLM-as-judge control: fluent on the task
checks but BLIND to background-color drift, and noisy on re-evaluation
(non-reproducible). The RLVR scorer is its deterministic counterpart that also
gates on background color — this difference is the paper's cleanest causal
claim (App. B.5: judge blindness -> policy drifts into graying backgrounds).
"""

from __future__ import annotations

import math

import torch

from data import (AGENT, BG, H, W, N_FRAMES, Instance, PALETTE, TARGET_COLORS)

_PAL = torch.tensor(PALETTE).permute(1, 0)          # (3,P)
AGENT_IDX = PALETTE.index(AGENT)
_VIS_TOL = 1.2   # legacy tolerance kept for reference


def _palette_assign(frame: torch.Tensor) -> torch.Tensor:
    """Nearest-palette assignment per pixel. frame: (3,H,W) -> (H,W) indices.
    Nearest-color (not radius) matching avoids yellow-target/red-agent
    collisions that would corrupt centroid extraction."""
    px = frame.permute(1, 2, 0)                      # (H,W,3)
    d = (px.unsqueeze(-1) - _PAL).abs().sum(2)       # (H,W,P) L1 to each color
    return d.argmin(-1)


def _agent_mask(frame: torch.Tensor) -> torch.Tensor:
    return _palette_assign(frame) == AGENT_IDX


def extract_centroids(frames: torch.Tensor):
    """Per-frame agent centroid (row,col) from agent-colored pixels.
    frames: (B,N,3,H,W) -> list of B lists of N (float row,col) or None."""
    B, N = frames.shape[:2]
    out = []
    for b in range(B):
        cents = []
        for k in range(N):
            m = _agent_mask(frames[b, k])
            if m.sum() == 0:
                cents.append(None)
            else:
                idx = m.nonzero(as_tuple=False).float().mean(0)
                cents.append((idx[0].item(), idx[1].item()))
        out.append(cents)
    return out


def _cell(c):
    return None if c is None else (int(round(c[0])), int(round(c[1])))


def score_colorseq(inst: Instance, frames: torch.Tensor) -> dict:
    """Soft-check family (weighted average, no hard gates)."""
    cents = extract_centroids(frames.unsqueeze(0))[0]
    # visit time per instructed target: frame index of closest centroid approach
    visits = []
    for name in inst.order:
        tpos = inst.targets[name]
        best, bk = 1e9, -1
        for k, c in enumerate(cents):
            if c is not None:
                d = abs(c[0] - tpos[0]) + abs(c[1] - tpos[1])
                if d < best:
                    best, bk = d, k
        visits.append((bk, best))
    ok_pairs = sum(1 for (k1, d1), (k2, d2) in zip(visits, visits[1:])
                   if k1 >= 0 and k2 >= 0 and d1 <= 2.0 and d2 <= 2.0 and k1 <= k2)
    c_order = ok_pairs / max(1, len(visits) - 1)
    d_final = 9.0 if cents[-1] is None else abs(cents[-1][0] - inst.goal[0]) + abs(cents[-1][1] - inst.goal[1])
    c_final = 1.0 if d_final <= 2.0 else math.exp(-(d_final - 2.0) / 3.0)
    # color purity: agent pixels within radius 2.5 of that frame's centroid
    purities = []
    for k, c in enumerate(cents):
        m = _agent_mask(frames[k])
        if c is None or m.sum() == 0:
            purities.append(0.0)
            continue
        idx = m.nonzero(as_tuple=False).float()
        r = idx.sub(torch.tensor(c)).norm(dim=-1)
        purities.append((r <= 2.5).float().mean().item())
    c_purity = sum(purities) / len(purities)
    # temporal continuity between consecutive centroids (tolerance scales with
    # the instance's waypoint spacing; teleporting still tanks it)
    conts = []
    for c1, c2 in zip(cents, cents[1:]):
        d = 9.0 if c1 is None or c2 is None else abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
        conts.append(max(0.0, 1.0 - max(0.0, d - inst.spacing) / 4.0))
    c_cont = sum(conts) / len(conts)
    checks = {"c_order": c_order, "c_final": c_final, "c_purity": c_purity, "c_cont": c_cont}
    weights = {"c_order": .35, "c_final": .25, "c_purity": .20, "c_cont": .20}
    soft = sum(checks[k] * weights[k] for k in checks) / sum(weights.values())
    return {"soft": soft, **checks}


def score_maze(inst: Instance, frames: torch.Tensor) -> dict:
    """Hard-gate family: wall gate x goal gate multiply the soft part."""
    cents = extract_centroids(frames.unsqueeze(0))[0]
    # progress: solver-path coverage by centroids
    hit = 0
    for cell in inst.path:
        if any(c is not None and abs(c[0] - cell[0]) + abs(c[1] - cell[1]) <= 1.2
               for c in cents):
            hit += 1
    s_progress = hit / len(inst.path)
    conts = []
    for c1, c2 in zip(cents, cents[1:]):
        d = 9.0 if c1 is None or c2 is None else abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
        conts.append(max(0.0, 1.0 - max(0.0, d - inst.spacing) / 4.0))
    s_cont = sum(conts) / len(conts)
    # gates (paper: 3 wall violations -> wall gate 0.73 => slope 0.09)
    viol = sum(1 for c in cents if c is not None and inst.wall_mask[_cell(c)])
    missing = sum(1 for c in cents if c is None)
    g_wall = max(0.0, 1.0 - 0.09 * (viol + missing))
    d_goal = 9.0 if cents[-1] is None else abs(cents[-1][0] - inst.goal[0]) + abs(cents[-1][1] - inst.goal[1])
    g_goal = 1.0 if d_goal <= 1.5 else math.exp(-(d_goal - 1.5) / 3.0)
    checks = {"s_progress": s_progress, "s_cont": s_cont, "g_wall": g_wall, "g_goal": g_goal}
    soft = (0.7 * s_progress + 0.3 * s_cont)
    return {"soft": soft, **checks}


def _static_bg_mask(inst: Instance) -> torch.Tensor:
    """(H,W) bool: cells that must show the background color (not wall, not
    target/goal). Cached on the instance — the agent blob is excluded per
    frame at scoring time via the centroid radius."""
    mask = ~inst.wall_mask.clone()
    if inst.task == "colorseq":
        for cell in inst.targets.values():
            mask[cell] = False
    else:
        mask[inst.goal] = False
    return mask


def background_gate(inst: Instance, frames: torch.Tensor):
    """g_bg: expected-background pixels must stay near the exact bg color.
    This is the check the VLM judge does NOT perform. Also returns raw
    bg_fidelity in [0,1] as a drift metric. frames: (N,3,H,W)."""
    cents = extract_centroids(frames.unsqueeze(0))[0]
    base = _static_bg_mask(inst)
    errs, npix = 0.0, 0
    for k in range(frames.shape[0]):
        m = base.clone()
        c = cents[k]
        if c is not None:  # exclude the moving agent's blob neighborhood
            rr = torch.arange(H).unsqueeze(1)
            cc = torch.arange(W).unsqueeze(0)
            m &= (abs(rr - c[0]) + abs(cc - c[1])) > 2.5
        if m.any():
            errs += float(frames[k].permute(1, 2, 0)[m].abs().sum())
            npix += int(m.sum())
    fidelity = 1.0 - min(1.0, errs / max(1, npix))
    # toy calibration: slope 1.5 keeps the gate responsive across the SFT
    # model's operating range (the paper hand-calibrates weights per task too)
    g_bg = max(0.0, 1.0 - 1.5 * (1.0 - fidelity))
    return g_bg, fidelity


def score_full(inst: Instance, frames: torch.Tensor) -> dict:
    """Deterministic RLVR reward + interpretable error decomposition."""
    parts = score_colorseq(inst, frames) if inst.task == "colorseq" else score_maze(inst, frames)
    g_bg, bg_fid = background_gate(inst, frames)
    gates = [g_bg]
    for key in ("g_wall", "g_goal"):
        if key in parts:
            gates.append(parts[key])
    score = parts["soft"]
    for g in gates:
        score *= g
    return {"score": score, "soft": parts["soft"], "g_bg": g_bg,
            "bg_fidelity": bg_fid, **{k: v for k, v in parts.items() if k != "soft"}}


def judge_reward(inst: Instance, frames: torch.Tensor, rng: torch.Generator) -> float:
    """RLVLM control: same task checks but NO background gate, plus judge noise
    quantized to a 1-decimal score sheet (paper: judges re-score identical
    outputs differently 54.6-92.8% of the time — ours flips when noise crosses
    a rounding boundary)."""
    parts = score_colorseq(inst, frames) if inst.task == "colorseq" else score_maze(inst, frames)
    r = parts["soft"]
    for key in ("g_wall", "g_goal"):
        if key in parts:
            r *= parts[key]
    noisy = r + torch.normal(0.0, 0.05, (1,), generator=rng).item()
    return round(noisy, 1)


def score_batch(insts, frames) -> tuple[torch.Tensor, list[dict]]:
    scores, details = [], []
    for i, inst in enumerate(insts):
        d = score_full(inst, frames[i])
        scores.append(d["score"])
        details.append(d)
    return torch.tensor(scores), details


def judge_batch(insts, frames, gen) -> torch.Tensor:
    return torch.tensor([judge_reward(i, frames[b], gen) for b, i in enumerate(insts)])


if __name__ == "__main__":  # scorer selftests on ground truth
    from data import make_batch
    insts, _, frames, _ = make_batch(6, seed=1)
    for i, inst in enumerate(insts):
        d = score_full(inst, frames[i])
        assert d["bg_fidelity"] > 0.999, d
        if inst.task == "colorseq":
            assert d["c_order"] == 1.0 and d["c_final"] == 1.0 and d["c_purity"] >= 0.8, d
        else:
            assert d["g_wall"] == 1.0 and d["g_goal"] >= 0.99 and d["s_progress"] >= 0.5, d
    # judge is blind to bg: corrupt background, full score tanks, judge doesn't
    bad = frames.clone()
    bad[..., :, :, 0] = 0.35                       # "graying" drift
    full_drop = [score_full(i, bad[b])["score"] / max(1e-9, score_full(i, frames[b])["score"])
                 for b, i in enumerate(insts)]
    gen = torch.Generator().manual_seed(0)
    judge_stable = [abs(judge_reward(i, bad[b], gen) - judge_reward(i, frames[b], gen))
                    for b, i in enumerate(insts)]
    print(f"data.py scorers OK. GT scores: "
          f"{[round(score_full(i, frames[b])['score'], 3) for b, i in enumerate(insts)]}")
    print(f"bg corruption: full-scorer keeps {min(full_drop):.3f} of score; "
          f"judge |delta| from bg change: mean {sum(judge_stable)/len(judge_stable):.4f}")
