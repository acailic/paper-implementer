"""Training loop for UI-MOPD (arXiv 2607.04425).

Pipeline (toy-scale mirror of the paper's two stages):

  Stage 1 - SFT: train two small platform-specific teachers on their own
            platform's templates.  (Paper: Qwen3-VL-32B teachers on Uni-GUI.)
  Stage 2 - MOPD: a single shared student is trained with the combined
            clipped-PPO + platform-conditioned K3-KL objective, using the
            two frozen teachers as behavioural anchors, with the adaptive
            KL mask (Eq. 6).

Run:  python3 train.py
"""

from __future__ import annotations

import random
import sys

import torch
import torch.nn.functional as F

from model import (PolicyLM, PolicyConfig, BOS, EOS, PAD, STOI,
                   adaptive_kl_mask, structured_reward, token_advantage,
                   mopd_loss)
from data import make_dataset, split_by_platform


DEVICE = "cpu"
SEED = 7
RESP_LEN = 3      # action tokens per rollout: verb [+ coord] + eos
GROUP_SIZE = 4    # rollouts per prompt (paper uses 8)


def set_seed(s: int):
    random.seed(s); torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Stage 1: SFT a teacher on its platform templates
# ---------------------------------------------------------------------------
def sft_teacher(prompts, cfg: PolicyConfig, epochs: int = 40,
                lr: float = 3e-3, device: str = DEVICE) -> PolicyLM:
    model = PolicyLM(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # build training sequences:  prompt + target action + eos
    seqs, masks = [], []
    for p in prompts:
        full = p.prompt_tokens + p.target + [STOI[EOS]]
        seqs.append(torch.tensor(full, dtype=torch.long))
        m = [0] * len(p.prompt_tokens) + [1] * (len(p.target) + 1)
        masks.append(torch.tensor(m, dtype=torch.float))
    S = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                        padding_value=STOI[PAD])
    M = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
    last_loss = None
    for ep in range(epochs):
        lp = model.log_probs(S, M)
        loss = -(lp * M).sum() / M.sum().clamp(min=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = float(loss.detach())
    return model


# ---------------------------------------------------------------------------
# Build a rollout batch from the mixed-platform prompt pool
# ---------------------------------------------------------------------------
def rollout(student: PolicyLM, prompts, device: str = DEVICE):
    """Sample RESP_LEN action tokens per prompt; score with Eq. 8 reward.

    Also freezes the student's log-probs over the sampled sequences (lp_old)
    so the PPO ratio is meaningful across the subsequent gradient steps.
    """
    student.eval()
    seqs, masks, rewards, platforms, targets, groups = [], [], [], [], [], []
    group_id = 0
    for p in prompts:
        for _ in range(GROUP_SIZE):
            prefix = p.prompt_tensor.unsqueeze(0).to(device)
            full = student.sample(prefix, RESP_LEN, temperature=1.0,
                                  device=device)
            seq_len = full.size(1)
            ids = full.squeeze(0).tolist()[len(p.prompt_tokens):]
            r = structured_reward(ids, p.platform, p.target)
            seqs.append(full.squeeze(0))
            m = ([0] * len(p.prompt_tokens)
                 + [1] * (seq_len - len(p.prompt_tokens)))
            masks.append(torch.tensor(m, dtype=torch.float))
            rewards.append(r)
            platforms.append(p.platform)
            targets.append(p.target)
            groups.append(group_id)
        group_id += 1
    S = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                        padding_value=STOI[PAD])
    M = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
    S = S.to(device); M = M.to(device)
    # freeze the log-probs of the policy that generated these samples
    with torch.no_grad():
        lp_old = student.log_probs(S, M)
    student.train()
    return (S, M, lp_old,
            torch.tensor(rewards, device=device),
            torch.tensor(groups, device=device, dtype=torch.long))


# ---------------------------------------------------------------------------
# Stage 2: MOPD
# ---------------------------------------------------------------------------
def train_mopd(student: PolicyLM, t_desk: PolicyLM, t_mob: PolicyLM,
               prompts, epochs: int = 40, beta: float = 0.01,
               tau_kl: float = 0.5, lr: float = 3e-3,
               inner_epochs: int = 4,
               device: str = DEVICE):
    """Multi-Teacher On-Policy Distillation (Stage 2).

    Each outer epoch: sample a rollout batch once, then run `inner_epochs`
    PPO/SGD passes over that *same* frozen batch. This is the standard PPO
    pattern and is what makes the policy ratio r_t = pi_new/pi_old move
    away from 1 so the clipped objective has something to optimise.
    """
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for ep in range(epochs):
        # ---- collect one rollout batch + freeze old policy log-probs ----
        S, M, lp_old, R, G = rollout(student, prompts, device)

        # teacher log-probs, platform-routed (Eq. 7)
        with torch.no_grad():
            t_desk.eval(); t_mob.eval()
            lp_desk = t_desk.log_probs(S, M)
            lp_mob = t_mob.log_probs(S, M)
        platforms = rollout_platforms(prompts)
        route_mob = torch.tensor(platforms, device=device) == 1  # mobile=1
        teacher_lp = torch.where(route_mob.unsqueeze(1), lp_mob, lp_desk)

        # adaptive KL mask per group (Eq. 6)
        group_means = torch.stack(
            [R[G == g].mean() if (G == g).any() else torch.tensor(0.0)
             for g in G.unique()]
        )
        mu = adaptive_kl_mask(group_means, tau_kl)
        mu_per_rollout = mu[G]

        # advantages (Eq. 9) - constant across inner steps
        A = token_advantage(R, G)

        # ---- inner optimisation over the frozen batch ----------------
        student.train()
        last_out = None
        for _ in range(inner_epochs):
            lp_new = student.log_probs(S, M)
            out = mopd_loss(lp_new, lp_old, teacher_lp, M, A, mu_per_rollout,
                            beta=beta)
            opt.zero_grad(); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            last_out = out

        if ep % 5 == 0 or ep == epochs - 1:
            mean_r = float(R.mean())
            loss_v = float(last_out['loss'].detach())
            pg_v = float(last_out['pg_loss'])
            kl_v = float(last_out['kl_loss'])
            ratio_v = float(last_out['mean_ratio'])
            print(f"  [mopd ep {ep:02d}] loss={loss_v:+.4f} "
                  f"pg={pg_v:+.4f} "
                  f"kl={kl_v:.4f} "
                  f"mean_reward={mean_r:+.3f} "
                  f"ratio={ratio_v:.3f}")


def rollout_platforms(prompts):
    """Map each rollout back to its platform (1=mobile, 0=desktop)."""
    out = []
    for p in prompts:
        out.extend([1 if p.platform == "mobile" else 0] * GROUP_SIZE)
    return out


# ---------------------------------------------------------------------------
# Evaluation: fraction of rollouts that produce a valid, correct action
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(student: PolicyLM, prompts, device: str = DEVICE):
    student.eval()
    n_ok, n = 0, 0
    for p in prompts:
        prefix = p.prompt_tensor.unsqueeze(0).to(device)
        full = student.sample(prefix, RESP_LEN, temperature=0.0, device=device)
        ids = full.squeeze(0).tolist()[len(p.prompt_tokens):]
        r = structured_reward(ids, p.platform, p.target)
        if r == 1.0:
            n_ok += 1
        n += 1
    return n_ok / max(n, 1)


# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    print("=" * 64)
    print("UI-MOPD: Multi-Platform On-Policy Distillation (toy impl)")
    print("Paper: arXiv 2607.04425  |  Eq. 1-12 reproduced")
    print("=" * 64)

    cfg = PolicyConfig()
    prompts = make_dataset(n_per_platform=8, seed=SEED)
    desk_p, mob_p = split_by_platform(prompts)

    print(f"\nDataset: {len(desk_p)} desktop + {len(mob_p)} mobile prompts")

    # ---- Stage 1: SFT the two teachers ---------------------------------
    print("\nStage 1: SFT platform-specific teachers ...")
    t_desk = sft_teacher(desk_p, cfg, epochs=30)
    t_mob = sft_teacher(mob_p, cfg, epochs=30)
    print(f"  desktop teacher acc = {evaluate(t_desk, desk_p):.2%}")
    print(f"  mobile  teacher acc = {evaluate(t_mob, mob_p):.2%}")

    # ---- baseline: a single SFT-only student (mixed) -------------------
    print("\nBaseline: single mixed-SFT student ...")
    base = sft_teacher(prompts, cfg, epochs=30)
    print(f"  mixed-SFT desktop acc = {evaluate(base, desk_p):.2%}")
    print(f"  mixed-SFT mobile  acc = {evaluate(base, mob_p):.2%}")

    # ---- Stage 2: MOPD -------------------------------------------------
    # The paper's student is the pre-trained Qwen3-VL-8B-Thinking; MOPD
    # *refines* an already-competent policy. The toy analogue: initialise
    # the student from the mixed-SFT baseline, then refine with MOPD.
    print("\nStage 2: Multi-Teacher On-Policy Distillation ...")
    print("  (student initialised from mixed-SFT baseline)")
    student = PolicyLM(cfg).to(DEVICE)
    student.load_state_dict(base.state_dict())
    train_mopd(student, t_desk, t_mob, prompts, epochs=40,
               beta=0.01, tau_kl=0.5)
    print(f"\nUI-MOPD desktop acc = {evaluate(student, desk_p):.2%}")
    print(f"UI-MOPD mobile  acc = {evaluate(student, mob_p):.2%}")

    print("\nDone. (Toy scale: small transformers, synthetic GUI actions.)")


if __name__ == "__main__":
    main()
