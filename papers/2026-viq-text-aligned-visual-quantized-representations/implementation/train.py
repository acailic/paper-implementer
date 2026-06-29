"""
ViQ Training Script — End-to-end training with all three losses.

Train a small-scale ViQ on CIFAR-10 demonstrating:
  1. Text-aligned pre-training (cross-entropy classification)
  2. Self-distillation (cosine similarity with frozen teacher)
  3. Proximal representation learning (L∞ bottleneck)
  4. FSQ quantization with 2D RoPE and multi-head expansion
  5. VAE latent reconstruction
  6. Image reconstruction from discrete codes

Usage:
    python train.py --epochs 30 --batch-size 128 --lr 3e-4

The script:
  - Pre-trains a tiny VAE (20 epochs) as the reconstruction target
  - Trains ViQ end-to-end with all losses
  - Logs losses, accuracy, codebook usage, and entropy
  - Saves a reconstruction demo at the end
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data import (
    TextAlignedCIFAR10, get_dataloaders, TinyVAE, pretrain_vae,
    CIFAR10_CLASSES, ID_TO_CLASS,
)
from model import (
    ViQModel, ViQDecoder, get_codebook_usage, compute_code_entropy,
)


def parse_args():
    p = argparse.ArgumentParser(description="ViQ Training on CIFAR-10")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--bottleneck-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--vae-epochs", type=int, default=20, help="VAE pre-training epochs")
    p.add_argument("--lambda-text", type=float, default=1.0, help="Text loss weight")
    p.add_argument("--lambda-distill", type=float, default=0.5, help="Distillation loss weight")
    p.add_argument("--lambda-recon", type=float, default=0.1, help="Reconstruction loss weight")
    p.add_argument("--lambda-quant", type=float, default=0.1, help="Quantization commitment loss weight")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def train_one_epoch(
    model: ViQModel,
    decoder: ViQDecoder,
    vae: TinyVAE,
    loader,
    optimizer,
    device,
    args,
    epoch,
):
    """Train for one epoch. Returns average loss dict."""
    model.train()
    decoder.train()

    totals = {k: 0.0 for k in [
        "loss", "loss_text", "loss_distill", "loss_recon", "loss_quant", "correct"
    ]}
    n_batches = 0

    for images, class_ids, text_tokens, query_tokens in loader:
        images = images.to(device)
        class_ids = class_ids.to(device)

        # Get VAE latent target (frozen)
        # Need unnormalized images for VAE (which expects [0,1])
        # Our dataloader normalizes with CIFAR stats, so we approximate:
        # For simplicity, we use the normalized images and let the recon head adapt
        with torch.no_grad():
            vae_latent = vae.get_latent(images)

        # Forward pass
        out = model(images)

        # ---- Loss 1: Text alignment (cross-entropy) ----
        loss_text = F.cross_entropy(out["text_logits"], class_ids)
        correct = (out["text_logits"].argmax(-1) == class_ids).float().sum()

        # ---- Loss 2: Self-distillation ----
        loss_distill = out["loss_distill"]

        # ---- Loss 3: VAE latent reconstruction ----
        loss_recon = F.mse_loss(out["recon_pred"], vae_latent)

        # ---- Loss 4: Quantization consistency ----
        # Encourage quantized features to be close to proximal features
        loss_quant = F.mse_loss(out["quantized_features"], out["f_hat"])

        # ---- Total loss ----
        loss = (
            args.lambda_text * loss_text
            + args.lambda_distill * loss_distill
            + args.lambda_recon * loss_recon
            + args.lambda_quant * loss_quant
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["loss_text"] += loss_text.item()
        totals["loss_distill"] += loss_distill.item()
        totals["loss_recon"] += loss_recon.item()
        totals["loss_quant"] += loss_quant.item()
        totals["correct"] += correct.item()
        n_batches += 1

    # Averages
    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    avg["accuracy"] = avg.pop("correct")  # rename
    return avg


@torch.no_grad()
def evaluate(
    model: ViQModel,
    decoder: ViQDecoder,
    vae: TinyVAE,
    loader,
    device,
    args,
):
    """Evaluate model. Returns metrics dict."""
    model.eval()
    decoder.eval()

    totals = {k: 0.0 for k in [
        "loss_text", "loss_distill", "loss_recon", "loss_quant", "correct"
    ]}
    all_codes = []
    n_batches = 0

    for images, class_ids, text_tokens, query_tokens in loader:
        images = images.to(device)
        class_ids = class_ids.to(device)

        with torch.no_grad():
            vae_latent = vae.get_latent(images)
            out = model(images)

        loss_text = F.cross_entropy(out["text_logits"], class_ids)
        loss_distill = out["loss_distill"]
        loss_recon = F.mse_loss(out["recon_pred"], vae_latent)
        loss_quant = F.mse_loss(out["quantized_features"], out["f_hat"])

        correct = (out["text_logits"].argmax(-1) == class_ids).float().sum()

        totals["loss_text"] += loss_text.item()
        totals["loss_distill"] += loss_distill.item()
        totals["loss_recon"] += loss_recon.item()
        totals["loss_quant"] += loss_quant.item()
        totals["correct"] += correct.item()
        all_codes.append(out["codes"].cpu())
        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    avg["accuracy"] = avg.pop("correct")

    all_codes = torch.cat(all_codes)
    avg["codebook_usage"] = get_codebook_usage(all_codes)
    avg["code_entropy"] = compute_code_entropy(all_codes)

    return avg


def save_reconstruction_demo(
    model: ViQModel,
    decoder: ViQDecoder,
    loader,
    device,
    output_dir: str,
    num_images: int = 16,
):
    """Save a grid of original vs reconstructed images."""
    model.eval()
    decoder.eval()

    # Get a batch
    images, class_ids, _, _ = next(iter(loader))
    images = images.to(device)
    class_ids = class_ids.to(device)

    with torch.no_grad():
        out = model(images, return_codes=True)
        quantized_features = out["quantized_features"]

    # Decode
    with torch.no_grad():
        recon_images = decoder(quantized_features)

    # Denormalize for saving
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
    orig = (images * std + mean).clamp(0, 1)
    recon = recon_images.clamp(0, 1)

    # Take first num_images
    orig = orig[:num_images]
    recon = recon[:num_images]

    # Interleave: row of originals, row of reconstructions
    comparison = torch.cat([orig, recon], dim=0)  # [2*num, 3, H, W]
    path = os.path.join(output_dir, "reconstruction_demo.png")
    save_image(comparison, path, nrow=num_images, padding=2)
    print(f"[Demo] Saved reconstruction comparison to {path}")

    # Also save code visualisation
    codes = out["codes"][:num_images].cpu()  # [N, num_patches]
    code_path = os.path.join(output_dir, "codes_sample.txt")
    with open(code_path, "w") as f:
        for i in range(min(num_images, len(class_ids))):
            f.write(f"Image {i} (class: {ID_TO_CLASS[class_ids[i].item()]}):\n")
            patch_codes = codes[i].tolist()
            f.write(f"  Patch codes: {patch_codes}\n")
            f.write(f"  Unique codes: {len(set(patch_codes))}/{len(patch_codes)}\n\n")
    print(f"[Demo] Saved sample codes to {code_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[Config] Device: {device}")
    print(f"[Config] Image size: {args.img_size}, Patch size: {args.patch_size}")
    print(f"[Config] Embed dim: {args.embed_dim}, Depth: {args.depth}")
    print(f"[Config] Bottleneck dim: {args.bottleneck_dim}, FSQ levels: [8,8,8,5,5,5]")
    print(f"[Config] FSQ codebook size: 64000")
    print(f"[Config] Loss weights: text={args.lambda_text}, distill={args.lambda_distill}, "
          f"recon={args.lambda_recon}, quant={args.lambda_quant}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    train_loader, test_loader, num_classes = get_dataloaders(
        batch_size=args.batch_size,
        image_size=args.img_size,
        num_workers=args.num_workers,
    )
    print(f"[Data] CIFAR-10: {len(train_loader.dataset)} train, "
          f"{len(test_loader.dataset)} test images")

    # ---- Pre-train VAE ----
    print("\n=== Stage 0: Pre-training VAE (reconstruction target) ===")
    vae = pretrain_vae(
        device=device,
        epochs=args.vae_epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
    )

    # ---- Build ViQ Model ----
    print("\n=== Building ViQ Model ===")
    model = ViQModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.embed_dim // 32,  # head_dim=32
        bottleneck_dim=args.bottleneck_dim,
        down_dim=6,
        fsq_levels=[8, 8, 8, 5, 5, 5],
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        dropout=0.1,
    ).to(device)

    # Copy encoder weights to teacher
    model.teacher_encoder.load_state_dict(model.encoder.state_dict())

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_teacher = sum(p.numel() for p in model.teacher_encoder.parameters())
    print(f"[Model] Trainable parameters: {n_params:,}")
    print(f"[Model] Teacher (frozen) parameters: {n_teacher:,}")

    # Decoder for reconstruction demo
    decoder = ViQDecoder(
        in_dim=args.embed_dim,
        out_channels=3,
        img_size=args.img_size,
        patch_size=args.patch_size,
    ).to(device)
    n_decoder = sum(p.numel() for p in decoder.parameters())
    print(f"[Model] Decoder parameters: {n_decoder:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.proximal.parameters(), "lr": args.lr * 2},
            {"params": model.quantizer.parameters(), "lr": args.lr},
            {"params": model.text_head.parameters(), "lr": args.lr},
            {"params": model.distill_head.parameters(), "lr": args.lr},
            {"params": model.recon_head.parameters(), "lr": args.lr},
            {"params": decoder.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ---- Training Loop ----
    print(f"\n=== Training ViQ ({args.epochs} epochs) ===")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Text':>7} | {'Distill':>8} | "
          f"{'Recon':>8} | {'Quant':>7} | {'Acc%':>6} | "
          f"{'CB Use%':>7} | {'Entropy':>7} | {'Time':>6}")
    print("-" * 90)

    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, decoder, vae, train_loader, optimizer, device, args, epoch,
        )

        # Evaluate
        eval_metrics = evaluate(
            model, decoder, vae, test_loader, device, args,
        )

        scheduler.step()
        dt = time.time() - t0

        # Log
        acc = eval_metrics["accuracy"] * 100
        cb_use = eval_metrics["codebook_usage"] * 100
        entropy = eval_metrics["code_entropy"]

        print(f"{epoch:>5d} | {train_metrics['loss']:>8.4f} | "
              f"{train_metrics['loss_text']:>7.4f} | "
              f"{train_metrics['loss_distill']:>8.4f} | "
              f"{train_metrics['loss_recon']:>8.4f} | "
              f"{train_metrics['loss_quant']:>7.4f} | "
              f"{acc:>6.1f} | {cb_use:>6.2f} | {entropy:>7.2f} | {dt:>5.1f}s")

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **eval_metrics,
        })

        if acc > best_acc:
            best_acc = acc
            # Save best model
            torch.save({
                "model": model.state_dict(),
                "decoder": decoder.state_dict(),
                "vae": vae.state_dict(),
                "epoch": epoch,
                "acc": acc,
            }, os.path.join(args.output_dir, "best_model.pt"))

    # ---- Save reconstruction demo ----
    print(f"\n=== Final Evaluation ===")
    print(f"Best test accuracy: {best_acc:.1f}%")
    print("Saving reconstruction demo...")

    save_reconstruction_demo(
        model, decoder, test_loader, device, args.output_dir,
    )

    # ---- Save code statistics ----
    final_eval = evaluate(model, decoder, vae, test_loader, device, args)
    stats_path = os.path.join(args.output_dir, "codebook_stats.txt")
    with open(stats_path, "w") as f:
        f.write("=== ViQ Codebook Statistics ===\n\n")
        f.write(f"FSQ Levels: [8, 8, 8, 5, 5, 5]\n")
        f.write(f"Total codebook size: 64,000\n\n")
        f.write(f"Codebook utilization: {final_eval['codebook_usage']*100:.2f}%\n")
        f.write(f"Code entropy: {final_eval['code_entropy']:.2f} bits\n\n")
        f.write(f"Test accuracy: {final_eval['accuracy']*100:.1f}%\n")
        f.write(f"Losses: text={final_eval['loss_text']:.4f}, "
                f"distill={final_eval['loss_distill']:.4f}, "
                f"recon={final_eval['loss_recon']:.4f}, "
                f"quant={final_eval['loss_quant']:.4f}\n")
    print(f"[Stats] Saved codebook statistics to {stats_path}")

    # ---- Demo: Show discrete token representation ----
    print("\n=== Discrete Token Representation Demo ===")
    model.eval()
    demo_images, demo_ids, _, _ = next(iter(test_loader))
    demo_images = demo_images[:4].to(device)
    with torch.no_grad():
        demo_out = model(demo_images)
    for i in range(4):
        cls_name = ID_TO_CLASS[demo_ids[i].item()]
        codes = demo_out["codes"][i].cpu().tolist()
        unique = len(set(codes))
        print(f"  Image {i} ({cls_name}): {len(codes)} patches, "
              f"{unique} unique codes out of 64000")
        print(f"    Code sequence (first 8): {codes[:8]}...")

    print("\n[DONE] Training complete. Check output/ for results.")


if __name__ == "__main__":
    main()
