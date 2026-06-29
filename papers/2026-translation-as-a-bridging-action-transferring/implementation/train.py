"""
Training script for 'Translation as a Bridging Action' VLA.

Implements the three-stage training strategy:
  Stage I:   Human-only pre-training (bridging action loss only)
  Stage II:  Human-robot co-training (all losses + random bridging substitution)
  Stage III: Few-shot robot post-training (all losses, robot data only)

Uses flow matching loss for action prediction.
"""

import os
import sys
import math
import time
import argparse
from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import BridgingActionVLA, flow_matching_loss
from data import (
    create_dataloaders,
    BridgingActionDataset,
    TableTop2D,
    extract_bridging_action,
    extract_eef_action,
    apply_bridging_substitution,
)


def train_one_epoch(
    model: BridgingActionVLA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    stage: int,
    device: torch.device,
    substitution_prob: float = 0.0,
    max_batches: int = None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: VLA model
        dataloader: training data
        optimizer: optimizer
        stage: 1, 2, or 3
        device: torch device
        substitution_prob: probability of bridging substitution (Stage II only)
        max_batches: limit batches per epoch (for quick testing)

    Returns:
        dict of average losses
    """
    model.train()
    total_losses = defaultdict(float)
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        # Move to device
        images = batch['image'].to(device)
        lang_tokens = batch['lang_tokens'].to(device)
        bridging_gt = batch['bridging'].to(device)      # (B, k*3)
        eef_gt = batch['eef'].to(device)                 # (B, k*6)
        gripper_gt = batch['gripper'].to(device)          # (B, k*1)
        data_sources = batch['data_source']               # list of strings

        B = images.shape[0]

        # Stage II: apply random bridging substitution on robot data
        if stage == 2 and substitution_prob > 0:
            apply_bridging_substitution(
                {'bridging': bridging_gt, 'eef': eef_gt,
                 'gripper': gripper_gt, 'data_source': data_sources},
                substitution_prob=substitution_prob,
            )

        # Sample noise level τ ~ U(0, 1) and noise ε ~ N(0, I)
        tau = torch.rand(B, 1, device=device)
        eps_bridging = torch.randn_like(bridging_gt)
        eps_eef = torch.randn_like(eef_gt)
        eps_gripper = torch.randn_like(gripper_gt)

        # Construct noisy actions: a^τ = τ·ε + (1-τ)·a
        noisy_bridging = tau * eps_bridging + (1 - tau) * bridging_gt
        noisy_eef = tau * eps_eef + (1 - tau) * eef_gt
        noisy_gripper = tau * eps_gripper + (1 - tau) * gripper_gt

        # Ground truth velocity: v* = ε - a
        gt_vel_bridging = eps_bridging - bridging_gt
        gt_vel_eef = eps_eef - eef_gt
        gt_vel_gripper = eps_gripper - gripper_gt

        # Forward pass (per data source)
        total_loss = torch.tensor(0.0, device=device)
        n_loss_terms = 0

        # Process samples grouped by data source (batch may be mixed)
        # For simplicity, process the whole batch assuming same source
        # In practice, you'd group by source or use per-sample masks
        # Here we process individually and accumulate
        for i in range(B):
            ds = data_sources[i]

            pred_vel, comp_present = model(
                images[i:i+1],
                lang_tokens[i:i+1],
                noisy_bridging[i:i+1],
                noisy_eef[i:i+1],
                noisy_gripper[i:i+1],
                tau[i:i+1],
                ds,
            )

            # Stage I: only bridging loss
            if stage == 1:
                loss = flow_matching_loss(
                    pred_vel['bridging'],
                    gt_vel_bridging[i:i+1]
                )
                total_losses['bridging'] += loss.item()
                total_loss += loss
                n_loss_terms += 1

            # Stage II / III: all available losses
            elif stage in (2, 3):
                sample_loss = torch.tensor(0.0, device=device)
                sample_terms = 0

                if comp_present['bridging']:
                    l_b = flow_matching_loss(
                        pred_vel['bridging'],
                        gt_vel_bridging[i:i+1]
                    )
                    total_losses['bridging'] += l_b.item()
                    sample_loss += l_b
                    sample_terms += 1

                if comp_present['eef']:
                    l_e = flow_matching_loss(
                        pred_vel['eef'],
                        gt_vel_eef[i:i+1]
                    )
                    total_losses['eef'] += l_e.item()
                    sample_loss += l_e
                    sample_terms += 1

                if comp_present['gripper']:
                    l_g = flow_matching_loss(
                        pred_vel['gripper'],
                        gt_vel_gripper[i:i+1]
                    )
                    total_losses['gripper'] += l_g.item()
                    sample_loss += l_g
                    sample_terms += 1

                if sample_terms > 0:
                    total_loss += sample_loss / sample_terms
                    n_loss_terms += 1

        if n_loss_terms > 0:
            total_loss = total_loss / n_loss_terms

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        num_batches += 1
        total_losses['total'] += total_loss.item()

    # Average
    metrics = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
    metrics['num_batches'] = num_batches
    return metrics


@torch.no_grad()
def validate(
    model: BridgingActionVLA,
    dataloader: DataLoader,
    stage: int,
    device: torch.device,
    max_batches: int = None,
) -> Dict[str, float]:
    """Validation loop."""
    model.eval()
    total_losses = defaultdict(float)
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        images = batch['image'].to(device)
        lang_tokens = batch['lang_tokens'].to(device)
        bridging_gt = batch['bridging'].to(device)
        eef_gt = batch['eef'].to(device)
        gripper_gt = batch['gripper'].to(device)
        data_sources = batch['data_source']
        B = images.shape[0]

        tau = torch.rand(B, 1, device=device)
        eps_bridging = torch.randn_like(bridging_gt)
        eps_eef = torch.randn_like(eef_gt)
        eps_gripper = torch.randn_like(gripper_gt)

        noisy_bridging = tau * eps_bridging + (1 - tau) * bridging_gt
        noisy_eef = tau * eps_eef + (1 - tau) * eef_gt
        noisy_gripper = tau * eps_gripper + (1 - tau) * gripper_gt

        gt_vel_bridging = eps_bridging - bridging_gt
        gt_vel_eef = eps_eef - eef_gt
        gt_vel_gripper = eps_gripper - gripper_gt

        total_loss = torch.tensor(0.0, device=device)
        n_terms = 0

        for i in range(B):
            ds = data_sources[i]
            pred_vel, comp_present = model(
                images[i:i+1], lang_tokens[i:i+1],
                noisy_bridging[i:i+1], noisy_eef[i:i+1], noisy_gripper[i:i+1],
                tau[i:i+1], ds,
            )

            if stage == 1 and 'bridging' in pred_vel:
                l = flow_matching_loss(pred_vel['bridging'], gt_vel_bridging[i:i+1])
                total_losses['bridging'] += l.item()
                total_loss += l
                n_terms += 1
            else:
                if 'bridging' in pred_vel:
                    l = flow_matching_loss(pred_vel['bridging'], gt_vel_bridging[i:i+1])
                    total_losses['bridging'] += l.item()
                    total_loss += l; n_terms += 1
                if 'eef' in pred_vel:
                    l = flow_matching_loss(pred_vel['eef'], gt_vel_eef[i:i+1])
                    total_losses['eef'] += l.item()
                    total_loss += l; n_terms += 1
                if 'gripper' in pred_vel:
                    l = flow_matching_loss(pred_vel['gripper'], gt_vel_gripper[i:i+1])
                    total_losses['gripper'] += l.item()
                    total_loss += l; n_terms += 1

        if n_terms > 0:
            total_loss = total_loss / n_terms
        num_batches += 1
        total_losses['total'] += total_loss.item()

    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}


def run_stage(
    model: BridgingActionVLA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stage: int,
    epochs: int,
    lr: float,
    device: torch.device,
    substitution_prob: float = 0.0,
    max_batches: int = None,
    stage_name: str = "",
) -> BridgingActionVLA:
    """Run a training stage."""
    print(f"\n{'='*60}")
    print(f"  Stage {stage}: {stage_name}")
    print(f"  Epochs: {epochs}, LR: {lr}, Substitution prob: {substitution_prob}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    best_val_loss = float('inf')
    for epoch in range(epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, stage, device,
            substitution_prob, max_batches,
        )
        val_metrics = validate(
            model, val_loader, stage, device, max_batches,
        )
        scheduler.step()
        dt = time.time() - t0

        # Log
        parts = [f"Epoch {epoch+1}/{epochs} ({dt:.1f}s)"]
        parts.append(f"train_loss={train_metrics['total']:.4f}")
        parts.append(f"val_loss={val_metrics['total']:.4f}")
        if 'bridging' in train_metrics:
            parts.append(f"L_bridge={train_metrics['bridging']:.4f}")
        if 'eef' in train_metrics:
            parts.append(f"L_eef={train_metrics['eef']:.4f}")
        if 'gripper' in train_metrics:
            parts.append(f"L_grip={train_metrics['gripper']:.4f}")
        print("  | ".join(parts))

        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']

    print(f"  Best val loss: {best_val_loss:.4f}")
    return model


def test_generation(model, device, num_samples=3):
    """Test action generation via flow matching inference."""
    print(f"\n--- Testing Action Generation ({num_samples} samples) ---")
    model.eval()

    env = TableTop2D()
    dataset = BridgingActionDataset(num_samples=num_samples)

    for i in range(num_samples):
        sample = dataset[i]
        image = sample['image'].unsqueeze(0).to(device)
        lang = sample['lang_tokens'].unsqueeze(0).to(device)
        ds = sample['data_source']

        with torch.no_grad():
            actions = model.generate_actions(image, lang, num_steps=5, data_source=ds)

        parts = [f"Sample {i} ({ds}):"]
        for key, val in actions.items():
            parts.append(f"  {key}: shape={val.shape}, "
                        f"mean={val.mean().item():.4f}, "
                        f"std={val.std().item():.4f}")
        print("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description="Train Translation as Bridging Action VLA"
    )
    parser.add_argument('--epochs-per-stage', type=int, default=10,
                       help='Epochs per training stage')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-train', type=int, default=2000)
    parser.add_argument('--num-val', type=int, default=200)
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Max batches per epoch (for quick testing)')
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Run specific stage (0=all)')
    parser.add_argument('--substitution-prob', type=float, default=0.5)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("Initializing Translation as Bridging Action VLA...")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs per stage: {args.epochs_per_stage}")

    # Create model
    model = BridgingActionVLA(
        img_size=64, patch_size=8, in_channels=3,
        vision_dim=128, vision_depth=4, vision_heads=4,
        lang_vocab=256, lang_max_len=32, lang_dim=128,
        lang_depth=2, lang_heads=4,
        action_chunk_size=4,
        bridging_dim=3, eef_dim=6, gripper_dim=1,
        action_dim=128, action_depth=4, action_heads=4,
        fm_hidden_dim=256, fm_layers=2,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    if args.test_only:
        test_generation(model, device)
        return

    # Stage I: Human-only pre-training
    if args.stage in (0, 1):
        print("\nGenerating Stage I data (human-only)...")
        train_dataset = BridgingActionDataset(
            num_samples=args.num_train,
            action_chunk_size=4,
            human_ratio=0.8,   # mostly human
            lab_human_ratio=0.2,
        )
        val_dataset = BridgingActionDataset(
            num_samples=args.num_val,
            action_chunk_size=4,
            human_ratio=0.8,
            lab_human_ratio=0.2,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                              shuffle=False)

        model = run_stage(
            model, train_loader, val_loader,
            stage=1, epochs=args.epochs_per_stage, lr=args.lr,
            device=device, substitution_prob=0.0,
            max_batches=args.max_batches,
            stage_name="Human-only Pre-training (bridging loss only)",
        )

    # Stage II: Human-robot co-training
    if args.stage in (0, 2):
        print("\nGenerating Stage II data (mixed human + robot)...")
        train_dataset = BridgingActionDataset(
            num_samples=args.num_train,
            action_chunk_size=4,
            human_ratio=0.5,
            lab_human_ratio=0.1,
        )
        val_dataset = BridgingActionDataset(
            num_samples=args.num_val,
            action_chunk_size=4,
            human_ratio=0.5,
            lab_human_ratio=0.1,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                              shuffle=False)

        model = run_stage(
            model, train_loader, val_loader,
            stage=2, epochs=args.epochs_per_stage, lr=args.lr * 0.5,
            device=device, substitution_prob=args.substitution_prob,
            max_batches=args.max_batches,
            stage_name="Human-Robot Co-training (all losses + substitution)",
        )

    # Stage III: Few-shot robot post-training
    if args.stage in (0, 3):
        print("\nGenerating Stage III data (robot-only)...")
        train_dataset = BridgingActionDataset(
            num_samples=args.num_train,
            action_chunk_size=4,
            human_ratio=0.0,     # robot only
            lab_human_ratio=0.0,
        )
        val_dataset = BridgingActionDataset(
            num_samples=args.num_val,
            action_chunk_size=4,
            human_ratio=0.0,
            lab_human_ratio=0.0,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                              shuffle=False)

        model = run_stage(
            model, train_loader, val_loader,
            stage=3, epochs=args.epochs_per_stage, lr=args.lr * 0.2,
            device=device, substitution_prob=0.0,
            max_batches=args.max_batches,
            stage_name="Few-shot Robot Post-training",
        )

    # Test generation
    test_generation(model, device)

    # Save model
    save_path = os.path.join(os.path.dirname(__file__), 'model_checkpoint.pt')
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == '__main__':
    # Need numpy import for seed
    import numpy as np
    main()
