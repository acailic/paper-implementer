"""Waypoint-OPD miniature — full pipeline on a synthetic shortcut task.

Paper: "Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated
On-Policy Distillation" — Hou et al., arXiv:2610.21601.

Pipeline (all CPU, minutes):
  1. TEACHER RL RUN.  SFT a teacher on a biased (shortcut) annotator, then
     GRPO with the true binary verifier reward. Save Initial (step 0),
     Waypoint (50%), Final (end). The shortcut is salient, the true feature
     subtle, so RL entrenches the shortcut first (early shift) and learns the
     truth later (late shift) — the two OPPOSE on conflict instances.
  2. STUDENT TRANSFER. Initialize a differently-seeded student (format SFT
     only) and train it on its own rollouts with:
       base        — no training (reference)
       opd         — reverse-KL to the Final teacher (Eq 1)
       direct_opd  — endpoint shift log pi1 - log pi0 as token reward (Eq 2-4)
       waypoint    — full Algorithm 1 (top-k support, q_t-centering,
                     positive-cosine gate, kappa_B batch scale, dual-clipped
                     candidate PPO + KL leash)
       no_gate     — g_t = 1, keep centering + kappa rule (=> kappa_B = 0.8)
       shuffled    — permute gates across prefixes in the batch
       no_norm     — kappa_B = 1
  3. EVALUATION. mean@8 accuracy on held-out instances, overall and split by
     agreement vs conflict instances; validation-selected checkpoints.
  4. DIAGNOSTICS. Teacher two-phase curve; gate statistics over training;
     early/late cosine by instance type at answer prefixes (the mechanism
     check); endpoint-vs-history pairing.

Honesty note: this is a deliberate miniature. The claim tested is the
MECHANISM (gating endpoint shifts by early/late agreement), not the paper's
benchmark numbers.
"""

import copy
import json
import time

import torch
import torch.nn.functional as F

from data import ShortcutTask
from model import TokenPolicy

VOCAB = 16
SEQ_LEN = 4
K_CAND = 8               # paper: k=16 of ~150k vocab; here 8 of 16
TAU = 0.8                # target mean rescaled gate
KAPPA_MIN, KAPPA_MAX = 0.75, 1.5
EPS_LOW_NORM = 1e-8      # low-norm fallback threshold
PPO_EPS = 0.2
DUAL_CLIP_C = 3.0
BETA = 0.25              # student KL leash to pi_S (full-distribution, at prefixes)
TEACHER_SEEDS = [0, 1, 2]
STUDENT_SEEDS = [10, 11, 12]
METHODS = ["base", "opd", "direct_opd", "no_gate", "shuffled", "no_norm", "waypoint"]

torch.set_num_threads(4)


# ---------------------------------------------------------------- teacher ---


def make_teacher(task, seed, grpo_steps=500, sft_steps=150, lr_sft=1e-2, lr_rl=1e-3,
                 group_n=8, prompts_per_step=64, gen=None, log=None):
    """Format-only SFT init -> GRPO on the true verifier. Returns
    (initial, waypoint, final) state_dicts and the eval curve.

    Expected dynamics (verified empirically): GRPO learns the SALIENT
    shortcut feature first (val acc -> ~agree_frac, ~0.65-0.70), then the
    subtle true feature (val acc -> 1.0). Checkpoint selection mirrors the
    paper's regime of FINITE-STEP RL with the Waypoint between the phases:
      - Initial  = before RL (format-only, ~chance);
      - Waypoint = end of the consolidation phase (last snapshot at the
        shortcut plateau) — on conflict instances its policy actively
        prefers the wrong token;
      - Final    = MID-correction (first snapshot past the plateau by a
        margin) — an imperfect teacher whose conflict-instance endpoints are
        the small residue of two large opposing shifts, exactly the
        cancelled-endpoint regime the paper's gate is for.

    Toy-scale recipe (deviations from the paper's 4B/GRPO config, needed to
    make the dynamics work at MLP scale — documented in README.md):
      - the policy-gradient loss uses only the answer position: the toy's
        reward depends only on the final token, and scratch-token gradients
        (pure noise here) dilute learning at this scale;
      - unnormalized group advantage (R - group mean) and a small entropy
        bonus keep exploration alive on conflict instances.
    """
    torch.manual_seed(seed)
    model = TokenPolicy(hidden=64, seed=seed, feat_dim=task.feat_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr_sft)
    for _ in range(sft_steps):
        x, _, _ = task.sample_prompts(256, gen)
        seqs = task.format_only_seq(256, gen)
        loss = F.cross_entropy(
            model.logits_all(x, seqs).reshape(-1, VOCAB), seqs.reshape(-1)
        )
        opt.zero_grad(); loss.backward(); opt.step()

    initial = copy.deepcopy(model.state_dict())
    curve = [evaluate(model, task, task.val_x, task.val_y, task.val_agree, n=4)]

    opt = torch.optim.Adam(model.parameters(), lr=lr_rl)
    ent_coef = 2e-2
    snapshots = []  # (val_acc, state_dict) every 25 steps
    for step in range(grpo_steps):
        x, y, _ = task.sample_prompts(prompts_per_step, gen)
        xs = x.repeat_interleave(group_n, 0)
        ys = y.repeat_interleave(group_n, 0)
        with torch.no_grad():
            seqs = model.sample(xs, temperature=1.0, gen=gen)
            rew = task.reward(seqs, ys).view(prompts_per_step, group_n)
            adv = (rew - rew.mean(1, keepdim=True)).reshape(-1)  # unnormalized
            lp_old = model.log_probs(xs, seqs)
        logits = model.logits_all(xs, seqs)
        lp = F.log_softmax(logits, dim=-1).gather(-1, seqs.unsqueeze(-1)).squeeze(-1)
        # answer position only (see docstring)
        lp_a, lpo_a = lp[:, -1], lp_old[:, -1]
        ratio = torch.exp(lp_a - lpo_a)
        clipped = torch.clamp(ratio, 1 - PPO_EPS, 1 + PPO_EPS) * adv
        J = torch.min(ratio * adv, clipped)
        J = torch.where(
            adv < 0, torch.maximum(J, DUAL_CLIP_C * adv * torch.ones_like(J)), J
        )
        probs_a = F.softmax(logits[:, -1], dim=-1)
        ent = -(probs_a.clamp_min(1e-9).log() * probs_a).sum(-1).mean()
        loss = -J.mean() - ent_coef * ent
        opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % 25 == 0:
            acc = evaluate(model, task, task.val_x, task.val_y, task.val_agree, n=4)
            curve.append(acc)
            snapshots.append((acc["acc"], copy.deepcopy(model.state_dict())))
    final = copy.deepcopy(model.state_dict())

    # Waypoint: end of the shortcut-consolidation plateau (last snapshot not
    # yet past it). Final: mid-correction (first snapshot clearly past the
    # plateau) — an imperfect teacher, like finite-step RL.
    plateau_hi = 0.70
    corr_reach = 0.78
    below = [(i, sd) for i, (acc, sd) in enumerate(snapshots) if acc <= plateau_hi]
    waypoint = below[-1][1] if below else snapshots[0][1]
    past = [(i, sd) for i, (acc, sd) in enumerate(snapshots) if acc >= corr_reach]
    if past:
        final = past[0][1]
    else:
        final = snapshots[-1][1]
    return initial, waypoint, final, curve


# ------------------------------------------------------------------- eval ---

@torch.no_grad()
def evaluate(model, task, x, y, agree, n=8, temp=0.7, gen=None):
    model.eval()
    correct = torch.zeros(x.shape[0])
    for _ in range(n):
        seqs = model.sample(x, temperature=temp, gen=gen)
        correct += task.reward(seqs, y)
    acc = (correct / n).mean().item()
    a = agree.bool()
    return {
        "acc": acc,
        "acc_agree": (correct / n)[a].mean().item(),
        "acc_conflict": (correct / n)[~a].mean().item(),
    }


# ------------------------------------------------- waypoint signal helpers ---

LP_FLOOR = -60.0  # numerical guard: saturated teachers underflow to log-prob -inf


@torch.no_grad()
def candidate_signals(student, teachers, x, seqs, k=K_CAND):
    """Per-prefix candidate supports, student weights, teacher segment shifts.

    The support is the student's top-k tokens; positions with fewer than k
    valid tokens (the answer position has 2) pad with masked tokens, which
    get zero weight and are excluded from every statistic (their teacher
    log-probs are -inf).

    Returns dict with (B, T, k) tensors: cand ids, qt, u (early, centered),
    v (late, centered), d (endpoint, centered), plus the weighted cosine c
    and gate g (B, T) with the low-norm fallback applied.
    """
    lp_old = student.prefix_logprobs(x, seqs)              # (B,T,V)
    probs_old = lp_old.exp()
    cand = probs_old.topk(k, dim=-1).indices               # (B,T,k)
    qt_raw = probs_old.gather(-1, cand)                    # (B,T,k)
    valid_cand = qt_raw > 0                                # masked ids have p=0
    qt_raw = torch.where(valid_cand, qt_raw, torch.zeros_like(qt_raw))
    qt = qt_raw / qt_raw.sum(-1, keepdim=True)

    lps = {}
    for name, net in teachers.items():
        lp = net.prefix_logprobs(x, seqs).gather(-1, cand).clamp_min(LP_FLOOR)
        lps[name] = torch.where(valid_cand, lp, torch.zeros_like(lp))
    d1 = lps["m"] - lps["0"]                               # early shift
    d2 = lps["1"] - lps["m"]                               # late shift

    def center(z):
        return z - (qt * z).sum(-1, keepdim=True)

    u, v = center(d1), center(d2)
    d = u + v                                              # == center(d1+d2)
    num = (qt * u * v).sum(-1)
    nu = (qt * u * u).sum(-1).clamp_min(0).sqrt()
    nv = (qt * v * v).sum(-1).clamp_min(0).sqrt()
    denom = (nu * nv).clamp_min(1e-30)
    c = num / denom
    fallback = (nu <= EPS_LOW_NORM) | (nv <= EPS_LOW_NORM) | ~torch.isfinite(c)
    g = torch.where(fallback, torch.ones_like(c), c.clamp_min(0))
    return {
        "cand": cand, "qt": qt, "u": u, "v": v, "d": d,
        "c": c, "g": g, "fallback": fallback, "valid_cand": valid_cand,
        "lp_old_cand": torch.where(
            valid_cand, lp_old.gather(-1, cand).clamp_min(LP_FLOOR),
            torch.zeros_like(qt_raw),
        ),
    }


def dual_clip_ppo(A, ratio):
    """Candidate/token-wise dual-clipped PPO objective J (Eq 17), maximized."""
    clipped = torch.clamp(ratio, 1 - PPO_EPS, 1 + PPO_EPS) * A
    J = torch.min(ratio * A, clipped)
    J = torch.where(A < 0, torch.maximum(J, DUAL_CLIP_C * A * torch.ones_like(J)), J)
    return J


# --------------------------------------------------------------- transfer ---

def transfer_student(method, task, teachers, seed, steps=100, prompts=128,
                     lr=1e-2, minibatches=4, gen=None, log=None):
    """Train one student under one transfer objective. Returns best-val state
    (by validation accuracy, mirroring the paper's checkpoint selection) and
    diagnostics."""
    torch.manual_seed(seed)
    student = TokenPolicy(hidden=96, seed=seed, feat_dim=task.feat_dim)
    opt_sft = torch.optim.Adam(student.parameters(), lr=1e-2)
    for _ in range(200):  # format-only SFT: knows the protocol, not the task
        seqs = task.format_only_seq(256, gen)
        x, _, _ = task.sample_prompts(256, gen)
        loss = F.cross_entropy(
            student.logits_all(x, seqs).reshape(-1, VOCAB), seqs.reshape(-1)
        )
        opt_sft.zero_grad(); loss.backward(); opt_sft.step()
    pi_S = copy.deepcopy(student)  # frozen reference
    pi_S.eval()
    for p in pi_S.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    best_val, best_state = -1.0, copy.deepcopy(student.state_dict())
    diag = {"gate_mean": [], "zero_gate_frac": [], "kappa": []}

    for step in range(steps):
        x, y, agree = task.sample_prompts(prompts, gen)
        with torch.no_grad():
            seqs = student.sample(x, temperature=1.0, gen=gen)

        if method == "base":
            continue  # reference: no training

        # ---------------- signals (all detached) ----------------
        if method in ("waypoint", "no_gate", "shuffled", "no_norm"):
            sig = candidate_signals(student, teachers, x, seqs)
            g = sig["g"]
            if method == "no_gate":
                g = torch.ones_like(g)
            elif method == "shuffled":
                perm = torch.randperm(g.numel(), generator=gen)
                g = g.reshape(-1)[perm].reshape(g.shape)
            valid = ~sig["fallback"].reshape(-1)
            # kappa uses the mean of the gates actually used (paper App. E:
            # no-gate => mean 1 => kappa_B = 0.8; shuffle preserves the mean)
            mean_g = g.reshape(-1)[valid].mean().item() if valid.any() else 1.0
            if method == "no_norm":
                kappa = 1.0
            else:
                kappa = min(max(TAU / max(mean_g, 1.18e-38), KAPPA_MIN), KAPPA_MAX)
            A_cand = kappa * sig["qt"] * g.unsqueeze(-1) * sig["d"]   # (B,T,k)
            if method in ("waypoint", "no_gate", "shuffled"):
                diag["gate_mean"].append(sig["g"].reshape(-1)[valid].mean().item())
                diag["zero_gate_frac"].append(
                    (sig["g"].reshape(-1)[valid] == 0).float().mean().item()
                )
                diag["kappa"].append(kappa)
        elif method == "direct_opd":
            with torch.no_grad():
                lp0 = teachers["0"].prefix_logprobs(x, seqs).gather(-1, seqs.unsqueeze(-1)).squeeze(-1).clamp_min(LP_FLOOR)
                lp1 = teachers["1"].prefix_logprobs(x, seqs).gather(-1, seqs.unsqueeze(-1)).squeeze(-1).clamp_min(LP_FLOOR)
                lp_old_store = student.prefix_logprobs(x, seqs).gather(
                    -1, seqs.unsqueeze(-1)
                ).squeeze(-1).clamp_min(LP_FLOOR)
            A_tok = lp1 - lp0                                    # (B,T)
        elif method == "opd":
            pass

        # ---------------- update ----------------
        B, T = seqs.shape
        idx_all = torch.randperm(B, generator=gen)
        mb_size = B // minibatches
        for mb in range(minibatches):
            idx = idx_all[mb * mb_size:(mb + 1) * mb_size]
            xm, ym = x[idx], seqs[idx]
            if method == "opd":
                lp_s = F.log_softmax(student.logits_all(xm, ym), dim=-1)
                with torch.no_grad():
                    lp_t = F.log_softmax(teachers["1"].logits_all(xm, ym), dim=-1)
                p = lp_s.exp()
                # clamp: masked entries have p=0 and lp=-inf; 0*(-inf) is nan
                kl = (p * (lp_s.clamp_min(-1e9) - lp_t.clamp_min(-1e9))).sum(-1)
                loss = kl.mean()
            else:
                logits = student.logits_all(xm, ym)
                lp_full = F.log_softmax(logits, dim=-1)
                # full-distribution KL leash to pi_S at the rollout prefixes
                # (nonnegative; the naive sampled-token estimator can be
                # driven negative and destabilizes training at this scale)
                with torch.no_grad():
                    lp_ref_full = F.log_softmax(pi_S.logits_all(xm, ym), dim=-1)
                p_full = lp_full.exp()
                kl_pen = (
                    p_full * (lp_full.clamp_min(LP_FLOOR) - lp_ref_full.clamp_min(LP_FLOOR))
                ).sum(-1).mean()
                if method == "direct_opd":
                    lp_tok = lp_full.gather(-1, ym.unsqueeze(-1)).squeeze(-1)
                    with torch.no_grad():
                        lp_old_tok = lp_old_store[idx]
                    ratio = torch.exp(lp_tok - lp_old_tok)
                    J = dual_clip_ppo(A_tok[idx].detach(), ratio).mean()
                    loss = -J + BETA * kl_pen
                else:
                    cand = sig["cand"][idx]
                    valid = sig["valid_cand"][idx]
                    lp_cand = lp_full.gather(-1, cand).clamp_min(LP_FLOOR)
                    lp_cand = torch.where(valid, lp_cand, torch.zeros_like(lp_cand))
                    ratio = torch.exp(lp_cand - sig["lp_old_cand"][idx])
                    J = dual_clip_ppo(A_cand[idx].detach(), ratio)
                    J = torch.where(valid, J, torch.zeros_like(J))
                    J = J.sum(-1).mean()
                    loss = -J + BETA * kl_pen
            opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % 10 == 0 or step == steps - 1:
            v = evaluate(student, task, task.val_x, task.val_y, task.val_agree,
                         n=8, gen=gen)
            if v["acc"] > best_val:
                best_val = v["acc"]
                best_state = copy.deepcopy(student.state_dict())
    if method == "base":
        best_state = copy.deepcopy(student.state_dict())
    student.load_state_dict(best_state)
    return student, diag, best_val


# ---------------------------------------------------------- mechanism probe ---

@torch.no_grad()
def mechanism_report(task, teachers, student, gen):
    """The paper's Figure-2 story at prefix level: at answer prefixes, early
    and late teacher shifts on conflict instances should OPPOSE (c < 0) while
    agreement instances ALIGN (c ~ +1), with comparable endpoint magnitudes."""
    x, y, agree = task.sample_prompts(512, gen)
    seqs = student.sample(x, temperature=1.0, gen=gen)
    sig = candidate_signals(student, teachers, x, seqs)
    a = agree.bool()
    ok = ~sig["fallback"]  # fallback prefixes have undefined cosine
    c_ans = sig["c"][:, -1]
    d_norm = sig["d"].norm(dim=-1)

    def stat(mask):
        sel = c_ans[mask & ok[:, -1]]
        return round(sel.mean().item(), 3) if sel.numel() else None

    def negfrac(mask):
        sel = c_ans[mask & ok[:, -1]]
        return round((sel < 0).float().mean().item(), 3) if sel.numel() else None

    scratch_ok = sig["c"][:, :-1][ok[:, :-1]]
    return {
        "n_agree_ok": int((a & ok[:, -1]).sum()),
        "n_conflict_ok": int((~a & ok[:, -1]).sum()),
        "answer_c_agree_mean": stat(a),
        "answer_c_conflict_mean": stat(~a),
        "answer_frac_c_neg_agree": negfrac(a),
        "answer_frac_c_neg_conflict": negfrac(~a),
        "answer_endpoint_norm_agree": round(d_norm[a, -1].mean().item(), 3),
        "answer_endpoint_norm_conflict": round(d_norm[~a, -1].mean().item(), 3),
        "scratch_c_mean": round(scratch_ok.mean().item(), 3) if scratch_ok.numel() else None,
        "scratch_frac_c_neg": round((scratch_ok < 0).float().mean().item(), 3)
        if scratch_ok.numel() else None,
        "answer_fallback_frac": round(sig["fallback"][:, -1].float().mean().item(), 3),
        "scratch_fallback_frac": round(sig["fallback"][:, :-1].float().mean().item(), 3),
    }


# ------------------------------------------------------------------- main ---

def main():
    t0 = time.time()
    out = {"teachers": [], "results": []}

    for t_seed in TEACHER_SEEDS:
        gen = torch.Generator().manual_seed(1000 + t_seed)
        task = ShortcutTask(seed=t_seed)
        st = time.time()
        initial, waypoint, final, curve = make_teacher(task, t_seed, gen=gen)
        tinfo = {
            "seed": t_seed,
            "curve": [round(c["acc"], 4) for c in curve],
            "initial": evaluate(_load(TokenPolicy, initial), task, task.val_x,
                                task.val_y, task.val_agree, n=8, gen=gen),
            "waypoint": evaluate(_load(TokenPolicy, waypoint), task, task.val_x,
                                 task.val_y, task.val_agree, n=8, gen=gen),
            "final": evaluate(_load(TokenPolicy, final), task, task.val_x,
                              task.val_y, task.val_agree, n=8, gen=gen),
        }
        print(f"[teacher seed {t_seed}] {time.time()-st:.1f}s "
              f"curve={tinfo['curve']}")
        print(f"  initial acc={tinfo['initial']['acc']:.3f} "
              f"waypoint={tinfo['waypoint']['acc']:.3f} "
              f"final={tinfo['final']['acc']:.3f}")
        out["teachers"].append(tinfo)

        teachers = {name: _load(TokenPolicy, sd) for name, sd in
                    (("0", initial), ("m", waypoint), ("1", final))}

        mech = mechanism_report(task, teachers, _load(TokenPolicy, final), gen)
        print(f"  mechanism @final-teacher prefixes: c_agree="
              f"{mech['answer_c_agree_mean']} c_conflict="
              f"{mech['answer_c_conflict_mean']} "
              f"frac_neg(conflict)={mech['answer_frac_c_neg_conflict']} "
              f"ans_fallback={mech['answer_fallback_frac']}")
        out["teachers"][-1]["mechanism"] = mech

        for method in METHODS:
            for s_seed in STUDENT_SEEDS:
                gen_t = torch.Generator().manual_seed(2000 + s_seed + 100 * t_seed)
                st = time.time()
                student, diag, best_val = transfer_student(
                    method, task, teachers, s_seed, gen=gen_t
                )
                te = evaluate(student, task, task.test_x, task.test_y,
                              task.test_agree, n=8, gen=gen_t)
                rec = {
                    "teacher_seed": t_seed, "method": method, "student_seed": s_seed,
                    **{f"test_{k}": round(v, 4) for k, v in te.items()},
                    "best_val": round(best_val, 4),
                    "diag": {
                        "gate_mean_first20": _mean_prefix(diag["gate_mean"], 0, 20),
                        "gate_mean_last20": _mean_prefix(diag["gate_mean"], -20, None),
                        "zero_gate_first20": _mean_prefix(diag["zero_gate_frac"], 0, 20),
                        "zero_gate_last20": _mean_prefix(diag["zero_gate_frac"], -20, None),
                        "kappa_min": min(diag["kappa"]) if diag["kappa"] else None,
                        "kappa_max": max(diag["kappa"]) if diag["kappa"] else None,
                    },
                }
                out["results"].append(rec)
                print(f"  {method:11s} s{s_seed}: test={te['acc']:.3f} "
                      f"agree={te['acc_agree']:.3f} conflict={te['acc_conflict']:.3f} "
                      f"({time.time()-st:.1f}s)")

    # ---------------- aggregate table ----------------
    print("\n==== aggregate (mean over 3 teacher seeds x 3 student seeds) ====")
    print(f"{'method':12s} {'test':>7s} {'agree':>7s} {'conflict':>9s}")
    agg = {}
    for m in METHODS:
        rows = [r for r in out["results"] if r["method"] == m]
        agg[m] = {
            "test": _mean([r["test_acc"] for r in rows]),
            "agree": _mean([r["test_acc_agree"] for r in rows]),
            "conflict": _mean([r["test_acc_conflict"] for r in rows]),
            "std": _std([r["test_acc"] for r in rows]),
            "gate_first": _mean([r["diag"]["gate_mean_first20"] for r in rows
                                 if r["diag"]["gate_mean_first20"] is not None]),
            "gate_last": _mean([r["diag"]["gate_mean_last20"] for r in rows
                                if r["diag"]["gate_mean_last20"] is not None]),
            "zero_last": _mean([r["diag"]["zero_gate_last20"] for r in rows
                                if r["diag"]["zero_gate_last20"] is not None]),
        }
        print(f"{m:12s} {agg[m]['test']:7.3f} {agg[m]['agree']:7.3f} "
              f"{agg[m]['conflict']:9.3f}  (+/- {agg[m]['std']:.3f})")
    out["aggregate"] = agg

    with open("results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\ntotal {time.time()-t0:.1f}s -> results.json")


def _load(cls, sd, feat_dim=8, **kw):
    m = cls(feat_dim=feat_dim, **kw)
    m.load_state_dict(sd)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _mean_prefix(xs, a, b):
    import math
    xs = [x for x in xs[a:b] if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else None


if __name__ == "__main__":
    main()
