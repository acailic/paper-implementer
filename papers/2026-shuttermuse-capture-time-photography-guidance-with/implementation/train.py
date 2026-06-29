"""
ShutterMuse Training Pipeline — SFT + GRPO
==========================================

Two-stage training matching the paper:
  Stage 1 (SFT): Supervised fine-tuning on structured JSON outputs (Eq. 1)
  Stage 2 (GRPO): Group Relative Policy Optimization with task-specific rewards (Eqs. 2-9)

SFT Hyperparameters (from paper):
  - Optimizer: AdamW, lr=1e-4
  - Batch size: 64 (effective)
  - Epochs: 5

GRPO Hyperparameters (from paper):
  - Batch size: 64
  - G (rollouts per input): 32
  - Epochs: 1
  - lr: 1e-6
  - weight decay: 0.1
  - β (KL coefficient): 0.01
  - clip ε: 0.2
  - mask threshold τ_m: 0.9

Usage:
  python train.py --mode sft --epochs 3 --batch_size 4
  python train.py --mode grpo --epochs 1 --batch_size 2 --num_rollouts 4
  python train.py --mode full  # SFT then GRPO
"""

import os
import sys
import math
import argparse
import copy
import time
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import (
    ShutterMuseModel,
    ShutterMuseConfig,
    compute_decision_reward,
    compute_mask_coverage_reward,
    compute_visibility_reward,
    compute_group_advantages,
    compute_grpo_loss,
    get_log_probs,
    format_crop_decision,
    format_visibility_output,
)
from data import (
    ShutterMuseDataConfig,
    PhotographerSideDataset,
    SubjectSideDataset,
    photographer_collate_fn,
    subject_collate_fn,
)


# ─── Training Configuration ────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ShutterMuse Training Pipeline")
    parser.add_argument("--mode", type=str, choices=["sft", "grpo", "full", "eval"],
                        default="sft", help="Training mode")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of synthetic samples")
    parser.add_argument("--num_rollouts", type=int, default=4, help="GRPO: G rollouts per input")
    parser.add_argument("--kl_beta", type=float, default=0.01, help="GRPO: KL coefficient β")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="GRPO: clip ε")
    parser.add_argument("--mask_threshold", type=float, default=0.9, help="Mask coverage threshold τ_m")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def get_device(args):
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


# ─── Stage 1: SFT Training ───────────────────────────────────────────────────

def train_sft_epoch(
    model: ShutterMuseModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
) -> Dict[str, float]:
    """Run one epoch of SFT training."""
    model.train()
    total_losses = defaultdict(float)
    num_batches = 0

    for step, batch in enumerate(dataloader):
        # Move to device
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        decision = batch['decision'].to(device)
        crop_box = batch['crop_box'].to(device)

        # Create dummy action tokens (interleaved)
        B = images.shape[0]
        num_action = model.config.num_decision_tokens + model.config.num_crop_tokens
        action_tokens = torch.zeros(B, num_action, dtype=torch.long, device=device)

        # Forward pass
        outputs = model(
            image=images,
            input_ids=input_ids,
            task_type="composition",
            action_tokens=action_tokens,
        )

        # Compute SFT loss
        losses = model.compute_sft_loss(
            outputs=outputs,
            target_ids=input_ids,
            decision_label=decision,
            crop_label=crop_box,
        )

        # Backward
        optimizer.zero_grad()
        losses['total_loss'].backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Accumulate metrics
        for k, v in losses.items():
            total_losses[k] += v.item()
        num_batches += 1

        if (step + 1) % args.log_interval == 0:
            avg_loss = total_losses['total_loss'] / num_batches
            avg_dec = total_losses.get('decision_loss', 0) / num_batches
            avg_crop = total_losses.get('crop_loss', 0) / num_batches
            avg_lm = total_losses.get('lm_loss', 0) / num_batches
            print(f"  [SFT] Epoch {epoch}, Step {step+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} (LM: {avg_lm:.4f}, Dec: {avg_dec:.4f}, Crop: {avg_crop:.4f})")

    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}


def train_sft_subject_epoch(
    model: ShutterMuseModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
) -> Dict[str, float]:
    """Run one epoch of SFT training on subject-side data."""
    model.train()
    total_losses = defaultdict(float)
    num_batches = 0

    for step, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        visibility = batch['visibility'].to(device)

        B = images.shape[0]
        num_action = model.config.num_visibility_tokens
        action_tokens = torch.zeros(B, num_action, dtype=torch.long, device=device)

        outputs = model(
            image=images,
            input_ids=input_ids,
            task_type="pose",
            action_tokens=action_tokens,
        )

        losses = model.compute_sft_loss(
            outputs=outputs,
            target_ids=input_ids,
            visibility_label=visibility,
        )

        optimizer.zero_grad()
        losses['total_loss'].backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        for k, v in losses.items():
            total_losses[k] += v.item()
        num_batches += 1

        if (step + 1) % args.log_interval == 0:
            avg_loss = total_losses['total_loss'] / num_batches
            avg_vis = total_losses.get('visibility_loss', 0) / num_batches
            avg_lm = total_losses.get('lm_loss', 0) / num_batches
            print(f"  [SFT-Subject] Epoch {epoch}, Step {step+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} (LM: {avg_lm:.4f}, Vis: {avg_vis:.4f})")

    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}


# ─── Stage 2: GRPO Training ──────────────────────────────────────────────────

@torch.no_grad()
def generate_rollouts(
    model: ShutterMuseModel,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    num_rollouts: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Generate G rollouts per input for GRPO.

    For each input, we:
      1. Sample G different action token sequences (via temperature / top-k)
      2. Get model outputs for each rollout
      3. Collect log probabilities for the GRPO loss

    Returns dictionary with rollout outputs.
    """
    B = images.shape[0]  # batch size = number of unique inputs
    L = input_ids.shape[1]

    all_log_probs = []
    all_decision_logits = []
    all_crop_boxes = []

    for g in range(num_rollouts):
        # Add slight noise to action tokens to create diverse rollouts
        num_action = model.config.num_decision_tokens + model.config.num_crop_tokens
        action_tokens = torch.randint(0, 16, (B, num_action), device=device)

        outputs = model(
            image=images,
            input_ids=input_ids,
            task_type="composition",
            action_tokens=action_tokens,
        )

        # Get log probs for text tokens
        log_probs = get_log_probs(outputs['logits'], input_ids)
        all_log_probs.append(log_probs)
        all_decision_logits.append(outputs['decision_logits'])
        all_crop_boxes.append(outputs['crop_box'])

    # Stack rollouts: (B*G, ...) flattened
    log_probs = torch.cat(all_log_probs, dim=0)  # (B*G, L)
    decision_logits = torch.cat(all_decision_logits, dim=0)  # (B*G, 3)
    crop_boxes = torch.cat(all_crop_boxes, dim=0)  # (B*G, 4)

    return {
        'log_probs': log_probs,
        'decision_logits': decision_logits,
        'crop_boxes': crop_boxes,
    }


def train_grpo_epoch(
    model: ShutterMuseModel,
    ref_model: ShutterMuseModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
) -> Dict[str, float]:
    """
    Run one epoch of GRPO training.

    For each batch:
      1. Generate G rollouts per input using current (old) policy
      2. Compute task-specific rewards for each rollout
      3. Compute group-relative advantages
      4. Compute GRPO loss with clipping + KL regularization
      5. Update model parameters
    """
    model.train()
    ref_model.eval()
    total_losses = defaultdict(float)
    total_rewards = defaultdict(float)
    num_batches = 0

    for step, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        gt_decision = batch['decision'].to(device)
        gt_crop_box = batch['crop_box'].to(device)
        subject_mask = batch['subject_mask'].to(device)

        B = images.shape[0]

        # ─── Step 1: Generate rollouts with old policy ───
        # Save old policy state
        old_params = {k: v.clone() for k, v in model.state_dict().items()}

        rollout_outputs = generate_rollouts(
            model, images, input_ids, args.num_rollouts, device
        )

        old_log_probs = rollout_outputs['log_probs']  # (B*G, L)
        old_decision_logits = rollout_outputs['decision_logits']  # (B*G, 3)
        old_crop_boxes = rollout_outputs['crop_boxes']  # (B*G, 4)

        # ─── Step 2: Compute rewards ───
        pred_decisions = old_decision_logits.argmax(dim=-1)  # (B*G,)

        # Repeat GT labels for each rollout
        gt_dec_expanded = gt_decision.unsqueeze(1).expand(-1, args.num_rollouts).reshape(-1)
        gt_crop_expanded = gt_crop_box.unsqueeze(1).expand(-1, args.num_rollouts, -1).reshape(-1, 4)
        mask_expanded = subject_mask.unsqueeze(1).expand(-1, args.num_rollouts, -1, -1).reshape(-1, *subject_mask.shape[1:])

        # R_dec (Eq. 2)
        r_dec = compute_decision_reward(pred_decisions, gt_dec_expanded)

        # R_mask (Eqs. 3-4): only for refine samples
        r_mask = compute_mask_coverage_reward(
            old_crop_boxes.detach(),
            mask_expanded,
            threshold=args.mask_threshold,
        )
        # Zero out mask reward for non-refine samples
        is_refine = (gt_dec_expanded == 0)
        r_mask = r_mask * is_refine.float()

        # Composite photographer-side reward (Eq. 5): R_photo = R_dec + R_mask
        rewards = r_dec + r_mask

        # ─── Step 3: Group-relative advantages (Eq. 7) ───
        advantages = compute_group_advantages(rewards, args.num_rollouts)

        # ─── Step 4: Reference model log probs ───
        with torch.no_grad():
            num_action = model.config.num_decision_tokens + model.config.num_crop_tokens
            action_tokens = torch.zeros(B * args.num_rollouts, num_action, dtype=torch.long, device=device)
            images_expanded = images.unsqueeze(1).expand(-1, args.num_rollouts, -1, -1, -1).reshape(
                B * args.num_rollouts, *images.shape[1:]
            )
            input_ids_expanded = input_ids.unsqueeze(1).expand(-1, args.num_rollouts, -1).reshape(
                B * args.num_rollouts, -1
            )

            ref_outputs = ref_model(
                image=images_expanded,
                input_ids=input_ids_expanded,
                task_type="composition",
                action_tokens=action_tokens,
            )
            ref_log_probs = get_log_probs(ref_outputs['logits'], input_ids_expanded)

        # ─── Step 5: New policy forward pass ───
        new_outputs = model(
            image=images_expanded,
            input_ids=input_ids_expanded,
            task_type="composition",
            action_tokens=action_tokens,
        )
        new_log_probs = get_log_probs(new_outputs['logits'], input_ids_expanded)

        # ─── Step 6: GRPO loss (Eq. 9) ───
        grpo_loss = compute_grpo_loss(
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            ref_log_probs=ref_log_probs,
            rewards=rewards,
            advantages=advantages,
            clip_epsilon=args.clip_epsilon,
            kl_beta=args.kl_beta,
        )

        # ─── Step 7: Update ───
        optimizer.zero_grad()
        grpo_loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Metrics
        total_losses['grpo_loss'] += grpo_loss.item()
        total_rewards['r_dec'] += r_dec.mean().item()
        total_rewards['r_mask'] += r_mask.mean().item()
        total_rewards['r_total'] += rewards.mean().item()
        total_rewards['decision_acc'] += (pred_decisions == gt_dec_expanded).float().mean().item()
        num_batches += 1

        if (step + 1) % args.log_interval == 0:
            avg_loss = total_losses['grpo_loss'] / num_batches
            avg_r_dec = total_rewards['r_dec'] / num_batches
            avg_r_mask = total_rewards['r_mask'] / num_batches
            avg_r = total_rewards['r_total'] / num_batches
            avg_acc = total_rewards['decision_acc'] / num_batches
            print(f"  [GRPO] Epoch {epoch}, Step {step+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"R_dec: {avg_r_dec:.3f}, R_mask: {avg_r_mask:.3f}, R_total: {avg_r:.3f} | "
                  f"Decision Acc: {avg_acc:.3f}")

    metrics = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
    metrics.update({k: v / max(num_batches, 1) for k, v in total_rewards.items()})
    return metrics


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: ShutterMuseModel,
    dataloader: DataLoader,
    device: torch.device,
    task: str = "photographer",
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    metrics = defaultdict(float)
    num_batches = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)

        if task == "photographer":
            gt_decision = batch['decision'].to(device)
            gt_crop_box = batch['crop_box'].to(device)

            B = images.shape[0]
            num_action = model.config.num_decision_tokens + model.config.num_crop_tokens
            action_tokens = torch.zeros(B, num_action, dtype=torch.long, device=device)

            outputs = model(
                image=images,
                input_ids=input_ids,
                task_type="composition",
                action_tokens=action_tokens,
            )

            pred_decisions = outputs['decision_logits'].argmax(dim=-1)

            # Decision accuracy
            metrics['decision_acc'] += (pred_decisions == gt_decision).float().mean().item()

            # Per-class accuracy
            for cls_name, cls_idx in [("refine", 0), ("keep", 1), ("reject", 2)]:
                mask = gt_decision == cls_idx
                if mask.sum() > 0:
                    cls_acc = (pred_decisions[mask] == cls_idx).float().mean().item()
                    metrics[f'{cls_name}_acc'] += cls_acc

            # Crop IoU (for refine samples only)
            pred_boxes = outputs['crop_box']
            refine_mask = gt_decision == 0
            if refine_mask.sum() > 0:
                iou = compute_mean_iou(pred_boxes[refine_mask], gt_crop_box[refine_mask])
                metrics['crop_iou'] += iou
                metrics['refinement_success'] += float(iou > 0.5)

        elif task == "subject":
            gt_visibility = batch['visibility'].to(device)

            B = images.shape[0]
            num_action = model.config.num_visibility_tokens
            action_tokens = torch.zeros(B, num_action, dtype=torch.long, device=device)

            outputs = model(
                image=images,
                input_ids=input_ids,
                task_type="pose",
                action_tokens=action_tokens,
            )

            pred_vis = outputs['visibility_logits'].argmax(dim=-1) - 1  # (B, 17) in {-1, 0, 1}
            vis_acc = (pred_vis == gt_visibility).float().mean(dim=1)  # per-keypoint accuracy
            metrics['visibility_acc'] += vis_acc.mean().item()
            metrics['visibility_per_kp'] += vis_acc.mean()

        num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in metrics.items()}


def compute_mean_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> float:
    """Compute mean IoU between two sets of boxes."""
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    union = area1 + area2 - inter

    iou = inter / union.clamp(min=1e-6)
    return iou.mean().item()


# ─── Main Training Loop ───────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args)
    print(f"Device: {device}")

    # Create model
    config = ShutterMuseConfig()
    model = ShutterMuseModel(config).to(device)

    param_counts = model.count_parameters()
    print(f"\nModel parameters:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")

    # Create datasets
    data_config = ShutterMuseDataConfig(num_samples=args.num_samples)
    photo_ds = PhotographerSideDataset(data_config)
    subject_ds = SubjectSideDataset(data_config)

    # Split
    n_train = int(len(photo_ds) * 0.8)
    photo_train_ds, photo_val_ds = torch.utils.data.random_split(photo_ds, [n_train, len(photo_ds) - n_train])
    subj_train_ds, subj_val_ds = torch.utils.data.random_split(subject_ds, [n_train, len(subject_ds) - n_train])

    photo_train_loader = DataLoader(photo_train_ds, batch_size=args.batch_size, shuffle=True,
                                     collate_fn=photographer_collate_fn)
    photo_val_loader = DataLoader(photo_val_ds, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=photographer_collate_fn)
    subj_train_loader = DataLoader(subj_train_ds, batch_size=args.batch_size, shuffle=True,
                                    collate_fn=subject_collate_fn)
    subj_val_loader = DataLoader(subj_val_ds, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=subject_collate_fn)

    print(f"\nDatasets:")
    print(f"  Photographer: {len(photo_train_ds)} train, {len(photo_val_ds)} val")
    print(f"  Subject: {len(subj_train_ds)} train, {len(subj_val_ds)} val")

    os.makedirs(args.save_dir, exist_ok=True)

    # ─── SFT Stage ─────────────────────────────────────────────────────────
    if args.mode in ["sft", "full"]:
        print("\n" + "=" * 60)
        print("Stage 1: Supervised Fine-Tuning (SFT)")
        print("=" * 60)

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(args.epochs):
            print(f"\n--- SFT Epoch {epoch + 1}/{args.epochs} ---")

            # Photographer-side SFT
            print("Photographer-side:")
            sft_metrics = train_sft_epoch(model, photo_train_loader, optimizer, device, epoch, args)

            # Subject-side SFT
            print("Subject-side:")
            sft_subj_metrics = train_sft_subject_epoch(model, subj_train_loader, optimizer, device, epoch, args)

            # Evaluate
            photo_eval = evaluate(model, photo_val_loader, device, "photographer")
            subj_eval = evaluate(model, subj_val_loader, device, "subject")

            print(f"\n  [Val-Photo] Decision Acc: {photo_eval['decision_acc']:.3f} | "
                  f"Refine Acc: {photo_eval.get('refine_acc', 0):.3f} | "
                  f"Keep Acc: {photo_eval.get('keep_acc', 0):.3f} | "
                  f"Reject Acc: {photo_eval.get('reject_acc', 0):.3f} | "
                  f"Crop IoU: {photo_eval.get('crop_iou', 0):.3f}")
            print(f"  [Val-Subject] Visibility Acc: {subj_eval['visibility_acc']:.3f}")

            scheduler.step()

        # Save SFT model
        sft_path = os.path.join(args.save_dir, "shuttermuse_sft.pt")
        torch.save(model.state_dict(), sft_path)
        print(f"\n✅ SFT model saved to {sft_path}")

    # ─── GRPO Stage ────────────────────────────────────────────────────────
    if args.mode in ["grpo", "full"]:
        print("\n" + "=" * 60)
        print("Stage 2: Group Relative Policy Optimization (GRPO)")
        print("=" * 60)

        # Create reference model (SFT checkpoint)
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # GRPO uses much smaller learning rate (paper: 1e-6)
        grpo_lr = 1e-6 if args.mode == "full" else args.lr
        optimizer = AdamW(model.parameters(), lr=grpo_lr, weight_decay=0.1)

        for epoch in range(args.epochs):
            print(f"\n--- GRPO Epoch {epoch + 1}/{args.epochs} ---")

            grpo_metrics = train_grpo_epoch(
                model, ref_model, photo_train_loader,
                optimizer, device, epoch, args
            )

            # Evaluate
            photo_eval = evaluate(model, photo_val_loader, device, "photographer")

            print(f"\n  [Val-Photo GRPO] Decision Acc: {photo_eval['decision_acc']:.3f} | "
                  f"Refine: {photo_eval.get('refine_acc', 0):.3f} | "
                  f"Keep: {photo_eval.get('keep_acc', 0):.3f} | "
                  f"Reject: {photo_eval.get('reject_acc', 0):.3f} | "
                  f"Crop IoU: {photo_eval.get('crop_iou', 0):.3f}")

        # Save GRPO model
        grpo_path = os.path.join(args.save_dir, "shuttermuse_grpo.pt")
        torch.save(model.state_dict(), grpo_path)
        print(f"\n✅ GRPO model saved to {grpo_path}")

    # ─── Eval-only mode ────────────────────────────────────────────────────
    if args.mode == "eval":
        print("\n" + "=" * 60)
        print("Evaluation Mode")
        print("=" * 60)

        # Try to load checkpoint
        ckpt_path = os.path.join(args.save_dir, "shuttermuse_grpo.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            print(f"Loaded GRPO checkpoint from {ckpt_path}")
        else:
            sft_path = os.path.join(args.save_dir, "shuttermuse_sft.pt")
            if os.path.exists(sft_path):
                model.load_state_dict(torch.load(sft_path, map_location=device, weights_only=True))
                print(f"Loaded SFT checkpoint from {sft_path}")

        photo_eval = evaluate(model, photo_val_loader, device, "photographer")
        subj_eval = evaluate(model, subj_val_loader, device, "subject")

        print(f"\n📊 Photographer-Side Results:")
        print(f"  Decision Accuracy: {photo_eval['decision_acc']:.3f}")
        for cls in ["refine", "keep", "reject"]:
            print(f"  {cls.capitalize()} Success Rate: {photo_eval.get(f'{cls}_acc', 0):.3f}")
        print(f"  Crop IoU: {photo_eval.get('crop_iou', 0):.3f}")
        print(f"  Refinement Success (IoU > 0.5): {photo_eval.get('refinement_success', 0):.3f}")

        print(f"\n📊 Subject-Side Results:")
        print(f"  Visibility Accuracy: {subj_eval['visibility_acc']:.3f}")

        # Demo: show a sample prediction
        print(f"\n📝 Sample Prediction Demo:")
        demo_batch = next(iter(photo_val_loader))
        demo_images = demo_batch['image'][:1].to(device)
        demo_ids = demo_batch['input_ids'][:1].to(device)
        B = 1
        num_action = model.config.num_decision_tokens + model.config.num_crop_tokens
        demo_action = torch.zeros(B, num_action, dtype=torch.long, device=device)

        with torch.no_grad():
            demo_out = model(demo_images, demo_ids, "composition", demo_action)
            result = format_crop_decision(demo_out['crop_box'], demo_out['decision_logits'])
        print(f"  {result}")

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
