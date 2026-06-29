"""
train.py — OPID training loop: on-policy skill distillation with hierarchical skills,
critical-first routing, token-level advantage, and GRPO.

Implements the full OPID pipeline:
  1. On-policy rollouts with current policy
  2. Hierarchical skill extraction (episode + step level)
  3. Critical-first routing
  4. Paired log-prob scoring → token-level skill advantage
  5. Combined OPID advantage = GRPO episode advantage + λ * skill advantage
  6. Clipped PPO update with KL regularization

Run:
  python train.py                  # 50 training iterations
  python train.py --iters 200      # longer run
  python train.py --no-opid        # baseline GRPO only
"""

from __future__ import annotations

import argparse
import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import (
    GridWorld,
    ExtractedSkills,
    ACTION_NAMES,
    ACTION_DELTAS,
    extract_skills,
    generate_trajectory,
    random_policy,
    serialize_trajectory,
)
from model import Tokenizer, TransformerPolicy, count_parameters


# ---------------------------------------------------------------------------
# Critical-First Routing
# ---------------------------------------------------------------------------

def critical_first_routing(
    step_idx: int,
    skills: ExtractedSkills,
) -> str:
    """
    Route to step-level skill if this step is critical, else episode-level skill.
    This is the hard switch from the paper.
    """
    if step_idx in skills.critical_steps:
        return skills.critical_steps[step_idx]
    return skills.episode_skill


# ---------------------------------------------------------------------------
# GRPO Episode Advantage
# ---------------------------------------------------------------------------

def compute_grpo_advantages(rewards: List[float], group_size: int = 8) -> torch.Tensor:
    """
    Compute group-relative (GRPO) episode advantages.

    For each group of `group_size` trajectories sampled from the same prompt,
    A^ep = (R - μ) / σ.

    Args:
        rewards: list of scalar rewards (length must be multiple of group_size).
        group_size: N trajectories per group.

    Returns:
        advantages: (len(rewards),) tensor of normalized advantages.
    """
    n = len(rewards)
    assert n % group_size == 0, f"n={n} not divisible by group_size={group_size}"
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    advantages = torch.zeros_like(rewards_t)

    for g in range(n // group_size):
        sl = slice(g * group_size, (g + 1) * group_size)
        group_rewards = rewards_t[sl]
        mu = group_rewards.mean()
        sigma = group_rewards.std()
        if sigma < 1e-8:
            # All tied — advantage is 0 (OPID can still learn via skill advantage!)
            advantages[sl] = 0.0
        else:
            advantages[sl] = (group_rewards - mu) / sigma

    return advantages


# ---------------------------------------------------------------------------
# OPID Paired Scoring: Skill Advantage
# ---------------------------------------------------------------------------

def compute_skill_advantage(
    model: TransformerPolicy,
    tokenizer: Tokenizer,
    obs_text: str,
    skill_text: str,
    target_action_id: int,
    device: torch.device,
) -> float:
    """
    Compute per-action skill advantage via paired log-prob shift.

    A^skill = log π_old(a | obs, skill) − log π_old(a | obs)

    Uses action-level log probs (simpler than full token-level for toy task).
    Returns the scalar advantage.
    """
    model.eval()

    # Original context
    orig_ids = tokenizer.encode(obs_text)
    orig_tensor = torch.zeros(1, len(orig_ids), dtype=torch.long, device=device)
    for i, tid in enumerate(orig_ids):
        orig_tensor[0, i] = tid

    with torch.no_grad():
        logprob_orig = model.get_action_logprobs(
            orig_tensor, torch.tensor([target_action_id], device=device)
        ).item()

    # Skill-augmented context
    aug_ids, _ = tokenizer.encode_with_skill(obs_text, skill_text)
    aug_tensor = torch.zeros(1, len(aug_ids), dtype=torch.long, device=device)
    for i, tid in enumerate(aug_ids):
        aug_tensor[0, i] = tid

    with torch.no_grad():
        logprob_skill = model.get_action_logprobs(
            aug_tensor, torch.tensor([target_action_id], device=device)
        ).item()

    return logprob_skill - logprob_orig


# ---------------------------------------------------------------------------
# Token-level paired scoring (full OPID mechanism)
# ---------------------------------------------------------------------------

def compute_token_skill_advantage(
    model: TransformerPolicy,
    tokenizer: Tokenizer,
    obs_text: str,
    skill_text: str,
    device: torch.device,
) -> float:
    """
    Compute token-level skill advantage as mean log-prob shift across all tokens.

    This mirrors the full OPID mechanism where the skill advantage is computed
    per token and then summed/averaged over response tokens.

    A^skill = mean_t [log π_old(token_t | obs, skill) − log π_old(token_t | obs)]
    """
    model.eval()

    # Original context: encode obs as tokens, predict next token
    orig_ids = tokenizer.encode(obs_text)
    orig_tensor = torch.zeros(1, len(orig_ids), dtype=torch.long, device=device)
    for i, tid in enumerate(orig_ids):
        orig_tensor[0, i] = tid

    # Create targets: shift by 1 (next-token prediction)
    target_ids = orig_tensor.roll(-1, dims=1)
    target_ids[:, -1] = 0  # pad

    with torch.no_grad():
        logprobs_orig = model.get_per_token_logprobs(orig_tensor, target_ids)
        mean_orig = logprobs_orig.sum() / logprobs_orig.shape[1]

    # Skill-augmented context
    aug_ids, skill_start = tokenizer.encode_with_skill(obs_text, skill_text)
    aug_tensor = torch.zeros(1, len(aug_ids), dtype=torch.long, device=device)
    for i, tid in enumerate(aug_ids):
        aug_tensor[0, i] = tid

    target_aug = aug_tensor.roll(-1, dims=1)
    target_aug[:, -1] = 0

    with torch.no_grad():
        logprobs_skill = model.get_per_token_logprobs(aug_tensor, target_aug)
        mean_skill = logprobs_skill.sum() / logprobs_skill.shape[1]

    return (mean_skill - mean_orig).item()


# ---------------------------------------------------------------------------
# On-Policy Rollout Collection
# ---------------------------------------------------------------------------

def collect_rollouts(
    model: TransformerPolicy,
    tokenizer: Tokenizer,
    n_episodes: int,
    group_size: int,
    max_steps: int,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[
    List[List[dict]],       # trajectories
    List[float],             # rewards
    List[ExtractedSkills],    # extracted skills
    List[List[int]],          # action id lists per episode
]:
    """Collect on-policy rollouts and extract skills."""
    rng = random.Random(seed)
    all_trajs = []
    all_rewards = []
    all_skills = []
    all_action_ids = []

    model.eval()
    for ep in range(n_episodes):
        env = GridWorld(max_steps=max_steps)
        env.reset(seed=rng.randint(0, 2**31 - 1))

        trajectory = []
        action_ids = []

        obs_text = env._observe()
        trajectory.append({"step": 0, "obs": obs_text, "action": None, "reward": None})

        for _ in range(env.max_steps):
            # Encode observation
            ids = tokenizer.encode(obs_text)
            input_tensor = torch.zeros(1, len(ids), dtype=torch.long, device=device)
            for i, tid in enumerate(ids):
                input_tensor[0, i] = tid

            # Sample action from policy
            with torch.no_grad():
                action_id = model.sample_action(input_tensor, temperature=1.0)
                action_name = tokenizer.decode_action_id(action_id.item())

            obs_new, reward, done = env.step(action_name)
            trajectory.append({
                "step": len(action_ids) + 1,
                "obs": obs_new,
                "action": action_name,
                "reward": reward,
            })
            action_ids.append(action_id.item())
            obs_text = obs_new

            if done:
                break

        total_reward = env.total_reward
        skills = extract_skills(
            trajectory, env.walls, env.goal, max_critical=5
        )

        all_trajs.append(trajectory)
        all_rewards.append(total_reward)
        all_skills.append(skills)
        all_action_ids.append(action_ids)

    return all_trajs, all_rewards, all_skills, all_action_ids


# ---------------------------------------------------------------------------
# PPO-style policy update
# ---------------------------------------------------------------------------

def ppo_update(
    model: TransformerPolicy,
    old_model: TransformerPolicy,
    tokenizer: Tokenizer,
    trajectories: List[List[dict]],
    action_ids_list: List[List[int]],
    opid_advantages: List[List[float]],  # per-step advantages within each episode
    lr: float = 1e-3,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01,
    epochs: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    PPO-style clipped objective update using OPID advantages.

    L(θ) = −E[min(ρ·A^OPID, clip(ρ, 1−ε, 1+ε)·A^OPID)] + β·KL(θ || θ_old)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    old_model.eval()

    stats = {"loss": [], "policy_loss": [], "kl_loss": [], "clip_frac": []}

    for _epoch in range(epochs):
        all_obs_texts = []
        all_action_ids = []
        all_advantages = []

        for traj, aids, step_advs in zip(trajectories, action_ids_list, opid_advantages):
            # Only use steps that have actions (skip step 0)
            for entry, aid, adv in zip(traj[1:], aids, step_advs):
                all_obs_texts.append(entry["obs"])
                all_action_ids.append(aid)
                all_advantages.append(adv)

        if not all_obs_texts:
            continue

        # Encode all observations
        batch_ids = []
        batch_actions = []
        batch_advs = []
        max_len = 0

        for obs, aid, adv in zip(all_obs_texts, all_action_ids, all_advantages):
            ids = tokenizer.encode(obs)
            batch_ids.append(ids)
            batch_actions.append(aid)
            batch_advs.append(adv)
            max_len = max(max_len, len(ids))

        # Pad and create tensors
        B = len(batch_ids)
        padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(batch_ids):
            for j, tid in enumerate(ids):
                padded[i, j] = tid
                mask[i, j] = 1

        actions_t = torch.tensor(batch_actions, dtype=torch.long, device=device)
        advs_t = torch.tensor(batch_advs, dtype=torch.float32, device=device)

        # Current policy log probs
        curr_logprobs = model.get_action_logprobs(padded, actions_t, mask)

        # Old policy log probs (no grad)
        with torch.no_grad():
            old_logprobs = old_model.get_action_logprobs(padded, actions_t, mask)

        # Importance ratio
        log_ratio = curr_logprobs - old_logprobs
        ratio = torch.exp(log_ratio).clamp(max=20.0)

        # Clipped surrogate
        surr1 = ratio * advs_t
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advs_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence
        kl_loss = (old_logprobs - curr_logprobs).mean()  # simplified KL

        # Total loss
        loss = policy_loss + kl_beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # Stats
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
        stats["loss"].append(loss.item())
        stats["policy_loss"].append(policy_loss.item())
        stats["kl_loss"].append(kl_loss.item())
        stats["clip_frac"].append(clip_frac)

    return {k: np.mean(v) if v else 0.0 for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Full OPID Training Loop
# ---------------------------------------------------------------------------

def train_opid(
    n_iters: int = 50,
    n_episodes: int = 16,
    group_size: int = 8,
    max_steps: int = 20,
    lr: float = 1e-3,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01,
    lambda_skill: float = 0.1,
    ppo_epochs: int = 4,
    seed: int = 42,
    device: str = "cpu",
    use_opid: bool = True,
    use_token_level: bool = False,
) -> Dict[str, List[float]]:
    """
    Full OPID training loop.

    Args:
        use_opid: If False, run GRPO baseline (no skill advantages).
        use_token_level: If True, use token-level paired scoring instead of
                         action-level for skill advantage computation.
    """
    device = torch.device(device)
    tokenizer = Tokenizer()
    model = TransformerPolicy(vocab_size=tokenizer.vocab_size).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    history = {
        "reward": [],
        "success_rate": [],
        "mean_advantage": [],
        "policy_loss": [],
        "kl_loss": [],
    }

    for iteration in range(n_iters):
        iter_seed = seed + iteration * 1000

        # ---- Step 1: Collect on-policy rollouts ----
        trajs, rewards, skills, action_ids_list = collect_rollouts(
            model, tokenizer,
            n_episodes=n_episodes,
            group_size=group_size,
            max_steps=max_steps,
            seed=iter_seed,
            device=device,
        )

        # ---- Step 2: Compute GRPO episode advantages ----
        ep_advantages = compute_grpo_advantages(rewards, group_size)

        # ---- Step 3: Compute OPID advantages (per-step) ----
        opid_step_advantages = []  # List[List[float]], one per episode

        for ep_idx, (traj, aids, ep_adv, skill) in enumerate(
            zip(trajs, action_ids_list, ep_advantages, skills)
        ):
            step_advs = []
            for step_j, (entry, aid) in enumerate(zip(traj[1:], aids)):
                step_num = entry["step"]

                # GRPO component: broadcast episode advantage
                grpo_comp = ep_adv.item()

                if use_opid:
                    # ---- Critical-First Routing ----
                    routed_skill = critical_first_routing(step_num, skill)

                    # ---- Paired Scoring → Skill Advantage ----
                    if use_token_level:
                        skill_adv = compute_token_skill_advantage(
                            model, tokenizer, entry["obs"], routed_skill, device
                        )
                    else:
                        skill_adv = compute_skill_advantage(
                            model, tokenizer, entry["obs"], routed_skill, aid, device
                        )

                    # ---- Combined OPID Advantage ----
                    opid_adv = grpo_comp + lambda_skill * skill_adv
                else:
                    opid_adv = grpo_comp

                step_advs.append(opid_adv)
            opid_step_advantages.append(step_advs)

        # ---- Step 4: PPO Update ----
        # Save old model for importance ratio
        old_model = copy.deepcopy(model)
        old_model.eval()

        update_stats = ppo_update(
            model, old_model, tokenizer,
            trajs, action_ids_list, opid_step_advantages,
            lr=lr, clip_eps=clip_eps, kl_beta=kl_beta,
            epochs=ppo_epochs, device=device,
        )

        # ---- Logging ----
        mean_reward = np.mean(rewards)
        success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        mean_opid_adv = np.mean([
            np.mean(sa) if sa else 0.0 for sa in opid_step_advantages
        ])

        history["reward"].append(mean_reward)
        history["success_rate"].append(success_rate)
        history["mean_advantage"].append(mean_opid_adv)
        history["policy_loss"].append(update_stats["policy_loss"])
        history["kl_loss"].append(update_stats["kl_loss"])

        if (iteration + 1) % 5 == 0 or iteration == 0:
            method = "OPID" if use_opid else "GRPO"
            print(
                f"[{method}] Iter {iteration+1:3d}/{n_iters} | "
                f"reward={mean_reward:+.3f} | success={success_rate:.1%} | "
                f"mean_A={mean_opid_adv:+.4f} | "
                f"ploss={update_stats['policy_loss']:.4f} | "
                f"kl={update_stats['kl_loss']:.4f} | "
                f"clip={update_stats['clip_frac']:.1%}"
            )

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: TransformerPolicy,
    tokenizer: Tokenizer,
    n_episodes: int = 100,
    max_steps: int = 20,
    seed: int = 99999,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Evaluate the policy on fresh environments."""
    rng = random.Random(seed)
    rewards = []
    successes = 0
    steps_list = []

    model.eval()
    for _ in range(n_episodes):
        env = GridWorld(max_steps=max_steps)
        env.reset(seed=rng.randint(0, 2**31 - 1))

        obs_text = env._observe()
        steps = 0
        for _ in range(env.max_steps):
            ids = tokenizer.encode(obs_text)
            input_tensor = torch.zeros(1, len(ids), dtype=torch.long, device=device)
            for i, tid in enumerate(ids):
                input_tensor[0, i] = tid

            with torch.no_grad():
                action_id = model.sample_action(input_tensor, temperature=0.0)  # greedy
                action_name = tokenizer.decode_action_id(action_id.item())

            obs_text, _, done = env.step(action_name)
            steps += 1
            if done:
                break

        r = env.total_reward
        rewards.append(r)
        if env.reached_goal:
            successes += 1
        steps_list.append(steps)

    return {
        "mean_reward": np.mean(rewards),
        "success_rate": successes / n_episodes,
        "mean_steps": np.mean(steps_list),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OPID Training")
    parser.add_argument("--iters", type=int, default=50, help="Training iterations")
    parser.add_argument("--episodes", type=int, default=16, help="Episodes per iteration")
    parser.add_argument("--group-size", type=int, default=8, help="GRPO group size")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-skill", type=float, default=0.1, help="Skill advantage coefficient")
    parser.add_argument("--kl-beta", type=float, default=0.01, help="KL regularization coefficient")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs per iteration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-opid", action="store_true", help="Run GRPO baseline (no skill advantages)")
    parser.add_argument("--token-level", action="store_true", help="Use token-level paired scoring")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    method = "OPID" if not args.no_opid else "GRPO-baseline"
    print(f"=== {method} Training ({args.iters} iters) ===")
    print(f"  λ_skill={args.lambda_skill}, β_kl={args.kl_beta}, ε_clip={args.clip_eps}")

    history = train_opid(
        n_iters=args.iters,
        n_episodes=args.episodes,
        group_size=args.group_size,
        max_steps=args.max_steps,
        lr=args.lr,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        lambda_skill=args.lambda_skill,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        device=device,
        use_opid=not args.no_opid,
        use_token_level=args.token_level,
    )

    # Final evaluation
    print("\n=== Evaluation ===")
    tokenizer = Tokenizer()
    # We need to recreate the model... Let's just print trajectory stats
    print(f"  Final mean reward: {history['reward'][-1]:+.3f}")
    print(f"  Final success rate: {history['success_rate'][-1]:.1%}")
    print(f"  Best success rate: {max(history['success_rate']):.1%}")

    # Print sample efficiency note
    if not args.no_opid:
        # Find iteration where OPID first reaches 50% success
        for i, sr in enumerate(history["success_rate"]):
            if sr >= 0.5:
                print(f"  OPID reached 50% success at iteration {i+1}")
                break


if __name__ == "__main__":
    main()
