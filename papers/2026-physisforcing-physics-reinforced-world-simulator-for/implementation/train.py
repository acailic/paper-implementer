"""
Training script for PhysisForcing physics-reinforced world simulator.

Trains a small DiT video generation model with:
- Flow matching loss (L_FM)
- Pixel-level trajectory alignment (L_pix)
- Semantic-level relational alignment (L_sem)

Uses synthetic 2D physics scenes (bouncing balls) for demonstration.
"""

import os
import sys
import argparse
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import PhysisForcingModel
from data import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='PhysisForcing Training')

    # Model
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--dim', type=int, default=192, help='DiT feature dimension')
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--n_blocks', type=int, default=8)
    parser.add_argument('--encoder_dim', type=int, default=128)
    parser.add_argument('--mlp_hidden', type=int, default=256)

    # Data
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('--n_train', type=int, default=800)
    parser.add_argument('--n_val', type=int, default=200)
    parser.add_argument('--traj_grid', type=int, default=8,
                        help='Grid size for trajectory points')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # PhysisForcing losses
    parser.add_argument('--lambda_pix', type=float, default=1.0,
                        help='Weight for L_pix (0.0 = disabled)')
    parser.add_argument('--lambda_sem', type=float, default=0.5,
                        help='Weight for L_sem (0.0 = disabled)')
    parser.add_argument('--n_query_points', type=int, default=64)
    parser.add_argument('--max_sem_tokens', type=int, default=512)

    # Misc
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str,
                        default='checkpoints')
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--generate_every', type=int, default=10,
                        help='Generate samples every N epochs')
    parser.add_argument('--save_samples', type=bool, default=True,
                        help='Save generated video samples as images')

    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device, epoch, args, log_file=None):
    """Train for one epoch."""
    model.dit.train()
    model.trajectory_mlp.train()
    model.semantic_mlp.train()

    epoch_losses = defaultdict(list)
    pbar = tqdm(loader, desc=f'Epoch {epoch}', file=log_file or sys.stdout)

    for batch in pbar:
        video = batch['video'].to(device)           # (B, T, C, H, W)
        trajectories = batch['trajectories'].to(device)  # (B, N_pts, T, 2)
        depth_map = batch['depth_map'].to(device)     # (B, H, W)

        optimizer.zero_grad()

        total_loss, loss_dict, _ = model.compute_total_loss(
            video, trajectories, depth_map
        )

        total_loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.trainables(), args.grad_clip)

        optimizer.step()

        # Log losses
        for k, v in loss_dict.items():
            epoch_losses[k].append(v)

        pbar.set_postfix({
            'fm': f"{loss_dict['fm']:.4f}",
            'pix': f"{loss_dict['pix']:.4f}",
            'sem': f"{loss_dict['sem']:.4f}",
            'total': f"{loss_dict['total']:.4f}",
        })

    # Compute epoch averages
    avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(model, loader, device, epoch, args):
    """Validate and compute metrics."""
    model.dit.eval()
    model.trajectory_mlp.eval()
    model.semantic_mlp.eval()

    epoch_losses = defaultdict(list)
    trajectory_errors = []

    for batch in loader:
        video = batch['video'].to(device)
        trajectories = batch['trajectories'].to(device)
        depth_map = batch['depth_map'].to(device)

        total_loss, loss_dict, middle_features = model.compute_total_loss(
            video, trajectories, depth_map
        )

        for k, v in loss_dict.items():
            epoch_losses[k].append(v)

        # Compute trajectory prediction error
        from model import compute_pixel_trajectory_loss
        B, T, C, H, W = video.shape

        # Extract physics mask
        from model import extract_physics_mask
        physics_mask = extract_physics_mask(trajectories, depth_map, H=H, W=W)

        _, pred_trajs = compute_pixel_trajectory_loss(
            dit_features=middle_features,
            trajectory_mlp=model.trajectory_mlp,
            trajectories_gt=trajectories,
            physics_mask=physics_mask,
            T=T, H=H, W=W,
            dit_patch_size=model.dit_patch_size,
            n_query_points=model.n_query_points,
        )

        # Average endpoint error (only for masked points)
        gt_trajs = trajectories[:, :model.n_query_points]
        err = (pred_trajs - gt_trajs).norm(dim=-1).mean().item()
        trajectory_errors.append(err)

    avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
    avg_traj_err = sum(trajectory_errors) / len(trajectory_errors) if trajectory_errors else 0

    return avg_losses, avg_traj_err


@torch.no_grad()
def generate_samples(model, device, epoch, args, save_dir):
    """Generate and optionally save video samples."""
    model.dit.eval()
    samples = model.generate(n_samples=4, n_steps=20)

    if args.save_samples and save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save each frame as an image
        import numpy as np

        for sample_idx in range(min(4, samples.shape[0])):
            for frame_idx in range(samples.shape[1]):
                frame = samples[sample_idx, frame_idx].cpu().numpy()
                frame = (frame + 1) / 2  # [-1, 1] -> [0, 1]
                frame = (frame * 255).astype(np.uint8)

                try:
                    from PIL import Image
                    img = Image.fromarray(frame.transpose(1, 2, 0))
                    img.save(os.path.join(
                        save_dir,
                        f'epoch{epoch:03d}_sample{sample_idx}_frame{frame_idx:02d}.png'
                    ))
                except ImportError:
                    pass  # Skip if PIL not available

    return samples


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = PhysisForcingModel(
        img_size=args.img_size,
        in_channels=3,
        dit_dim=args.dim,
        dit_n_heads=args.n_heads,
        dit_n_blocks=args.n_blocks,
        encoder_dim=args.encoder_dim,
        mlp_hidden=args.mlp_hidden,
        n_frames=args.n_frames,
        n_query_points=args.n_query_points,
        max_sem_tokens=args.max_sem_tokens,
        lambda_pix=args.lambda_pix,
        lambda_sem=args.lambda_sem,
    ).to(device)

    total_params, trainable_params = model.n_params()
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"PhysisForcing config: λ_pix={args.lambda_pix}, λ_sem={args.lambda_sem}")

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        n_train=args.n_train,
        n_val=args.n_val,
        H=args.img_size,
        W=args.img_size,
        n_frames=args.n_frames,
        trajectory_grid=args.traj_grid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")

    # Optimizer and scheduler
    optimizer = AdamW(model.trainables(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training loop
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    samples_dir = os.path.join(save_dir, 'samples')
    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    history = {'train': [], 'val': [], 'traj_err': []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args
        )

        # Validate
        val_losses, traj_err = validate(model, val_loader, device, epoch, args)

        # Step scheduler
        scheduler.step()

        dt = time.time() - t0

        # Log
        history['train'].append(train_losses)
        history['val'].append(val_losses)
        history['traj_err'].append(traj_err)

        lr = optimizer.param_groups[0]['lr']

        print(
            f"[Epoch {epoch:3d}/{args.epochs}] "
            f"Train: fm={train_losses['fm']:.4f} "
            f"pix={train_losses['pix']:.4f} "
            f"sem={train_losses['sem']:.4f} "
            f"total={train_losses['total']:.4f} | "
            f"Val: total={val_losses['total']:.4f} "
            f"traj_err={traj_err:.2f}px | "
            f"lr={lr:.2e} | "
            f"{dt:.1f}s"
        )

        # Save best checkpoint
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'args': vars(args),
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_losses['total'],
            'args': vars(args),
        }, os.path.join(save_dir, 'latest_model.pt'))

        # Generate samples periodically
        if epoch % args.generate_every == 0 or epoch == 1:
            generate_samples(model, device, epoch, args, samples_dir)
            print(f"  → Generated sample videos")

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final trajectory error: {history['traj_err'][-1]:.2f}px")
    print(f"Checkpoints saved to: {save_dir}")

    # Ablation summary
    print(f"\n--- Ablation Summary ---")
    print(f"λ_pix = {args.lambda_pix}, λ_sem = {args.lambda_sem}")
    if args.lambda_pix > 0 and args.lambda_sem > 0:
        print("Mode: Full PhysisForcing (both losses)")
    elif args.lambda_pix > 0:
        print("Mode: L_pix only")
    elif args.lambda_sem > 0:
        print("Mode: L_sem only")
    else:
        print("Mode: Baseline (flow matching only)")


if __name__ == '__main__':
    main()
