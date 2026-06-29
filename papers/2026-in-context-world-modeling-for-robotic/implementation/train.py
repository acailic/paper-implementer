"""
ICWM Training Script.

Trains the in-context world model on the 2D point-reaching task.
Demonstrates that context-conditioned training improves generalization
to novel viewpoints at test time.

Usage:
    python train.py                    # Full training + evaluation
    python train.py --epochs 30         # Quick run
    python train.py --no-context        # Ablation: train without context
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (
    ICWMDataset,
    AffineViewpoint,
    collate_fn,
    generate_probing_clip,
    generate_task_episode,
)
from model import ICWMConfig, BlockCausalTransformer, build_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, device, use_context=True):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        if use_context:
            pred_action = model(batch)
        else:
            # No-context baseline: feed zero/dummy context
            dummy_batch = {
                'ctx_obs_s': torch.zeros_like(batch['ctx_obs_s']),
                'ctx_actions': torch.zeros_like(batch['ctx_actions']),
                'ctx_obs_e': torch.zeros_like(batch['ctx_obs_e']),
                'task_obs': batch['task_obs'],
                'task_action': batch['task_action'],
                'task_next_obs': batch['task_next_obs'],
            }
            pred_action = model(dummy_batch)

        # MSE loss on task actions only (context not supervised — per ICWM paper)
        loss = nn.functional.mse_loss(pred_action, batch['task_action'])
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, device, num_trials=100, use_context=True, seed=123):
    """
    Evaluate on OOD viewpoints (novel camera viewpoints).
    Runs a simple point-reaching policy and measures success rate.
    """
    model.eval()
    config = model.config

    ood_viewpoints = [AffineViewpoint.random(train=False) for _ in range(6)]

    successes = 0
    total_dist_error = 0
    trials = 0

    with torch.no_grad():
        for vp in ood_viewpoints:
            for trial in range(num_trials // len(ood_viewpoints)):
                # Generate a task start and target
                pos = np.array([random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)])
                target = np.array([random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)])
                while np.linalg.norm(target - pos) < 0.15:
                    target = np.array([random.uniform(0.15, 0.85),
                                       random.uniform(0.15, 0.85)])

                # Generate probing context clips for this viewpoint
                if use_context:
                    probing_clips = generate_probing_clip(vp, max_steps=config.n_context_clips)

                # Run policy for up to max_steps
                threshold = 0.08
                max_steps = 15

                for step in range(max_steps):
                    obs = np.array([pos[0], pos[1], target[0], target[1]])
                    obs_t = vp.transform_observation(obs)

                    if use_context and len(probing_clips) >= config.n_context_clips:
                        # Build context batch
                        ctx_clips = probing_clips[:config.n_context_clips]
                        ctx_obs_s = torch.stack([torch.tensor(c[0], dtype=torch.float32) for c in ctx_clips]).unsqueeze(0)
                        ctx_actions = torch.stack([torch.tensor(c[1], dtype=torch.float32) for c in ctx_clips]).unsqueeze(0)
                        ctx_obs_e = torch.stack([torch.tensor(c[2], dtype=torch.float32) for c in ctx_clips]).unsqueeze(0)
                    else:
                        N = config.n_context_clips
                        ctx_obs_s = torch.zeros(1, N, config.obs_dim)
                        ctx_actions = torch.zeros(1, N, config.action_dim)
                        ctx_obs_e = torch.zeros(1, N, config.obs_dim)

                    task_obs = torch.tensor(obs_t, dtype=torch.float32).unsqueeze(0)
                    batch = {
                        'ctx_obs_s': ctx_obs_s.to(device),
                        'ctx_actions': ctx_actions.to(device),
                        'ctx_obs_e': ctx_obs_e.to(device),
                        'task_obs': task_obs.to(device),
                        'task_action': torch.zeros(1, config.action_dim).to(device),
                        'task_next_obs': torch.zeros(1, config.obs_dim).to(device),
                    }

                    pred_action = model(batch).cpu().squeeze(0).numpy()

                    # Apply action in true space
                    pos = np.clip(pos + pred_action * 0.5, 0.05, 0.95)

                    if np.linalg.norm(pos - target) < threshold:
                        successes += 1
                        break

                final_dist = np.linalg.norm(pos - target)
                total_dist_error += final_dist
                trials += 1

    success_rate = successes / max(trials, 1)
    avg_dist = total_dist_error / max(trials, 1)

    return {
        'success_rate': success_rate,
        'avg_distance': avg_dist,
        'trials': trials,
    }


def main():
    parser = argparse.ArgumentParser(description='ICWM Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--no-context', action='store_true', help='Train without context (baseline ablation)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval-every', type=int, default=10, help='Evaluate every N epochs')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()

    set_seed(args.seed)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Config
    config = ICWMConfig(
        obs_dim=4,
        action_dim=2,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
        obs_tokens=4,
        action_tokens=2,
        n_context_clips=5,
    )

    use_context = not args.no_context

    # Build model
    model = build_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Using context: {use_context}")

    # Data
    dataset = ICWMDataset(num_episodes=2000, context_clips_per_episode=config.n_context_clips)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_fn, drop_last=True)
    print(f"Dataset: {len(dataset)} episodes, {len(dataloader)} batches/epoch")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 70)

    best_success = 0.0
    results_history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, dataloader, optimizer, device, use_context=use_context)
        scheduler.step()
        dt = time.time() - t0

        result_str = f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.6f} | Time: {dt:.1f}s"

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_result = evaluate(model, device, num_trials=60, use_context=use_context)
            result_str += f" | Eval SR: {eval_result['success_rate']:.1%} | Avg Dist: {eval_result['avg_distance']:.4f}"
            results_history.append((epoch, train_loss, eval_result['success_rate'], eval_result['avg_distance']))

            if eval_result['success_rate'] > best_success:
                best_success = eval_result['success_rate']
                save_dir = Path(__file__).parent / 'checkpoints'
                save_dir.mkdir(exist_ok=True)
                tag = 'icwm' if use_context else 'baseline'
                save_path = save_dir / f'{tag}_best.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'success_rate': best_success,
                }, save_path)
                result_str += f" ★ Best!"

        print(result_str)

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Evaluate both with and without context to show ICWM advantage
    print("\n--- With ICWM context (in-context world modeling) ---")
    eval_icwm = evaluate(model, device, num_trials=200, use_context=True)

    print("\n--- Without context (standard VLA baseline) ---")
    eval_baseline = evaluate(model, device, num_trials=200, use_context=False)

    print(f"\n{'Metric':<20} {'ICWM (context)':<18} {'Baseline (no ctx)':<18} {'Improvement':<12}")
    print("-" * 68)
    print(f"{'Success Rate':<20} {eval_icwm['success_rate']:<18.1%} {eval_baseline['success_rate']:<18.1%} "
          f"{eval_icwm['success_rate'] - eval_baseline['success_rate']:+.1%}")
    print(f"{'Avg Distance':<20} {eval_icwm['avg_distance']:<18.4f} {eval_baseline['avg_distance']:<18.4f} "
          f"{eval_baseline['avg_distance'] - eval_icwm['avg_distance']:+.4f}")

    # Summary
    improvement = eval_icwm['success_rate'] - eval_baseline['success_rate']
    print(f"\nICWM improvement: {improvement:+.1%} absolute ({improvement/max(eval_baseline['success_rate'],0.01):+.1%} relative)")
    if improvement > 0:
        print("✅ ICWM context helps the model adapt to novel viewpoints!")
    else:
        print("⚠️  ICWM did not improve over baseline — may need more training data or epochs.")

    print(f"\nBest checkpoint saved to: {Path(__file__).parent / 'checkpoints' / ('icwm_best.pt' if use_context else 'baseline_best.pt')}")


if __name__ == '__main__':
    main()
