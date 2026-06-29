"""
JetSpec Training Script.

Trains the draft head via knowledge distillation (forward KL) from the target model,
then benchmarks speculative decoding speedup.

Usage:
    python train.py [--epochs 5] [--lr 3e-4] [--batch_size 32] [--device cpu]
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import (
    TargetModel, DraftHead, TargetConfig, DraftConfig,
    FeatureFusion, TreeNode,
    full_tree_construction, verify_tree_greedy,
    forward_kl_loss, speculative_decode_step,
    build_tree_causal_mask,
)
from data import (
    generate_synthetic_text,
    build_char_vocab,
    encode_text,
    decode_tokens,
    create_dataloaders,
)


def train_draft_head(
    target_model: TargetModel,
    draft_head: DraftHead,
    train_loader,
    config: DraftConfig,
    device: torch.device,
    n_epochs: int = 5,
    lr: float = 3e-4,
):
    """
    Train the draft head using forward KL distillation loss.

    For each training sequence:
    1. Run target model (frozen) to get teacher logits + hidden states
    2. Run draft head with tree-causal mask to get student logits
    3. Compute forward KL loss
    4. Backprop through draft head only

    Returns:
        losses: list of average losses per epoch
    """
    optimizer = optim.AdamW(draft_head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    draft_head.train()
    target_model.eval()

    all_losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            B, T = input_ids.shape

            # --- Target model forward (frozen) ---
            with torch.no_grad():
                target_logits, all_hidden = target_model(input_ids)
                # Select layers for fusion
                selected_hidden = [all_hidden[i] for i in config.fuse_layers]

            # --- Draft head forward with tree-causal mask ---
            # For training: treat the sequence as a single branch (sequential)
            # The tree-causal mask reduces to standard causal for a single branch
            T = input_ids.size(1)
            n_prefix = T  # prefix and draft input are the same tokens for training
            # Draft head concatenates fused features (T_prefix) + draft embeddings (T_draft)
            # so total self-attention length = T_prefix + T_draft = T + T = 2T
            T_total = 2 * T

            # Build combined causal mask: prefix positions attend bidirectionally,
            # draft positions attend to all prefix + causal among themselves
            combined_mask = torch.full((T_total, T_total), float("-inf"), device=device)
            # Draft positions (T_prefix:) can see all prefix (0..T-1) and earlier draft positions
            combined_mask[T:, :T] = 0.0  # draft sees all prefix
            # Causal among draft positions
            draft_causal = torch.tril(torch.ones(T, T, device=device))
            combined_mask[T:, T:] = torch.where(draft_causal == 1, 0.0, float("-inf"))
            # Prefix positions attend bidirectionally among themselves
            combined_mask[:T, :T] = 0.0
            # Prefix positions do NOT attend to draft positions
            combined_mask[:T, T:] = float("-inf")
            draft_logits, draft_probs = draft_head(
                input_ids, combined_mask, selected_hidden
            )

            # Both models receive input_ids of shape (B, T)
            # Both produce logits of shape (B, T, V)
            # logits[t] predicts the next token given input_ids[:t+1]
            # We compare draft logits directly with target logits (same positions)
            teacher_logits = target_logits   # (B, T, V)
            student_logits = draft_logits    # (B, T, V)

            loss = forward_kl_loss(
                student_logits, teacher_logits,
                temperature=config.temp_kd,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft_head.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        all_losses.append(avg_loss)
        print(f"  Epoch {epoch+1} average loss: {avg_loss:.4f}")

    return all_losses


def benchmark_speculative_decoding(
    target_model: TargetModel,
    draft_head: DraftHead,
    bench_loader,
    char2idx: dict,
    idx2char: dict,
    config: DraftConfig,
    device: torch.device,
    n_steps: int = 20,
):
    """
    Benchmark speculative decoding speedup.

    Uses linear speculative decoding: draft K tokens, verify against target,
    accept longest matching prefix. This is the standard speculative decoding
    algorithm that JetSpec generalizes to trees.

    Measures:
    - Tokens generated per step (with and without speculation)
    - Acceptance rate
    - Effective speedup

    Returns:
        results: dict with benchmark metrics
    """
    target_model.eval()
    draft_head.eval()

    total_accepted = 0
    total_proposed = 0
    total_steps = 0
    acceptance_rates = []

    # Time measurements
    total_time_greedy = 0.0
    total_time_draft = 0.0

    bench_iter = iter(bench_loader)
    samples_checked = 0
    K = 5  # number of draft tokens per step

    for step_idx in range(n_steps):
        try:
            prefix_ids = next(bench_iter).to(device)
        except StopIteration:
            bench_iter = iter(bench_loader)
            prefix_ids = next(bench_iter).to(device)

        current_prefix = prefix_ids

        # --- Greedy baseline: generate K tokens one-by-one ---
        t0 = time.time()
        with torch.no_grad():
            greedy_generated = []
            for _ in range(K):
                logits, all_hidden = target_model(current_prefix)
                next_tok = logits[0, -1].argmax().item()
                greedy_generated.append(next_tok)
                current_prefix = torch.cat([
                    current_prefix,
                    torch.tensor([[next_tok]], device=device)
                ], dim=1)
        t_greedy = time.time() - t0
        total_time_greedy += t_greedy

        # --- Speculative decoding: draft K tokens, verify all at once ---
        current_prefix = prefix_ids
        t0 = time.time()
        with torch.no_grad():
            # Step 1: Draft K tokens using draft head
            # Get target hidden states for feature fusion
            target_logits_prefix, all_hidden = target_model(current_prefix)
            selected_hidden = [all_hidden[i] for i in config.fuse_layers]

            # Draft K tokens autoregressively using draft head
            draft_tokens = []
            draft_log_probs = []
            temp_prefix = current_prefix
            n_prefix = current_prefix.size(1)

            for k in range(K):
                # Build combined mask for single-step draft
                T_in = temp_prefix.size(1) - n_prefix  # draft tokens so far
                T_total = n_prefix + max(T_in, 1)

                combined_mask = torch.full((T_total, T_total), float("-inf"), device=device)
                combined_mask[:n_prefix, :n_prefix] = 0.0
                if T_in > 0:
                    combined_mask[n_prefix:, :n_prefix] = 0.0
                    causal_part = torch.tril(torch.ones(T_in, T_in, device=device))
                    combined_mask[n_prefix:, n_prefix:] = torch.where(
                        causal_part == 1, 0.0, float("-inf")
                    )
                else:
                    # First draft token: just sees prefix
                    combined_mask[n_prefix:, :n_prefix] = 0.0

                # Use only the draft portion as input
                if T_in > 0:
                    draft_input = temp_prefix[:, n_prefix:]
                else:
                    # For first token, use last prefix token as context
                    draft_input = temp_prefix[:, -1:]

                logits_d, probs_d = draft_head(
                    draft_input, combined_mask, selected_hidden
                )

                # Sample from draft distribution (greedy for deterministic benchmark)
                next_tok = logits_d[0, -1].argmax().item()
                log_prob = torch.log(probs_d[0, -1, next_tok] + 1e-10).item()

                draft_tokens.append(next_tok)
                draft_log_probs.append(log_prob)
                temp_prefix = torch.cat([
                    temp_prefix,
                    torch.tensor([[next_tok]], device=device)
                ], dim=1)

            # Step 2: Verify all K draft tokens against target model in parallel
            if draft_tokens:
                draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=device)
                full_seq = torch.cat([current_prefix, draft_tensor], dim=1)
                target_logits_full, _ = target_model(full_seq)
            else:
                target_logits_full, _ = target_model(current_prefix)

            # Step 3: Accept longest matching prefix (greedy verification)
            n_accepted = 0
            for k in range(len(draft_tokens)):
                # Target prediction at position n_prefix + k - 1 (after seeing tokens 0..k-1)
                if k == 0:
                    pred = target_logits_prefix[0, -1].argmax().item()
                else:
                    pred = target_logits_full[0, n_prefix + k - 1].argmax().item()

                if pred == draft_tokens[k]:
                    n_accepted += 1
                else:
                    break

            # Get correction token from target
            if n_accepted == len(draft_tokens):
                # All accepted, sample one more from target
                correction_tok = target_logits_full[0, -1].argmax().item()
            elif n_accepted > 0:
                # Some accepted, correction from rejection point
                correction_tok = target_logits_full[0, n_prefix + n_accepted - 1].argmax().item()
            else:
                # None accepted, correction from prefix
                correction_tok = target_logits_prefix[0, -1].argmax().item()

            n_proposed = len(draft_tokens)

        t_draft = time.time() - t0
        total_time_draft += t_draft

        total_accepted += n_accepted
        total_proposed += n_proposed
        total_steps += 1
        acceptance_rates.append(n_accepted / max(n_proposed, 1))
        samples_checked += 1
    # Compute metrics
    avg_acceptance = sum(acceptance_rates) / max(len(acceptance_rates), 1)
    total_spec_tokens = total_accepted + total_steps  # accepted + correction tokens

    # Theoretical speedup: E[tokens] / (N*c + 1)
    # where c = draft_cost / verify_cost
    # We measure actual time-based speedup
    time_speedup = total_time_greedy / max(total_time_draft, 1e-6)

    # Expected tokens formula: (1 - alpha^(N+1)) / (1 - alpha)
    alpha = avg_acceptance
    N = config.node_budget - 1  # number of draft tokens

    results = {
        "avg_acceptance_rate": avg_acceptance,
        "total_accepted": total_accepted,
        "total_proposed": total_proposed,
        "total_steps": total_steps,
        "time_speedup": time_speedup,
        "greedy_time": total_time_greedy,
        "speculative_time": total_time_draft,
        "n_samples": samples_checked,
        "theoretical_expected_tokens": (1 - alpha ** (N + 1)) / (1 - alpha) if alpha < 1 else N + 1,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="JetSpec Training & Benchmark")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seq_len", type=int, default=64, help="Training sequence length")
    parser.add_argument("--bench_steps", type=int, default=20, help="Benchmark steps")
    parser.add_argument("--n_text_chars", type=int, default=100_000, help="Synthetic text length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Generate synthetic data ---
    print("=" * 60)
    print("Generating synthetic text data...")
    text = generate_synthetic_text(n_chars=args.n_text_chars, seed=args.seed)
    char2idx, idx2char = build_char_vocab(text)
    vocab_size = len(char2idx)
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Text length: {len(text)} chars")
    print(f"  Sample text: {text[:100]}...")

    # --- Create dataloaders ---
    print("\nCreating dataloaders...")
    train_loader, bench_loader, _, _, _ = create_dataloaders(
        text=text,
        char2idx=char2idx,
        train_seq_len=args.seq_len,
        train_stride=args.seq_len // 2,
        bench_prefix_len=32,
        bench_stride=32,
        batch_size=args.batch_size,
    )
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Benchmark samples: {len(bench_loader)}")

    # --- Build models ---
    target_config = TargetConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
    )
    draft_config = DraftConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_depth=6,
        branching_width=3,
        node_budget=15,
        fuse_layers=tuple(range(target_config.n_layers)),
        temp_kd=1.5,
    )

    print(f"\nTarget model config: {target_config}")
    print(f"Draft head config: max_depth={draft_config.max_depth}, "
          f"width={draft_config.branching_width}, "
          f"budget={draft_config.node_budget}")

    target_model = TargetModel(target_config).to(device)
    draft_head = DraftHead(draft_config).to(device)

    n_target_params = sum(p.numel() for p in target_model.parameters())
    n_draft_params = sum(p.numel() for p in draft_head.parameters())
    print(f"\nTarget model parameters: {n_target_params:,}")
    print(f"Draft head parameters: {n_draft_params:,}")
    print(f"Draft/Target ratio: {n_draft_params/n_target_params:.2%}")

    # --- Train draft head ---
    print("\n" + "=" * 60)
    print("Training draft head with Forward KL distillation...")
    print("=" * 60)

    # Pre-train target model briefly on the synthetic data
    print("\nPre-training target model (5 epochs)...")
    target_optimizer = optim.AdamW(target_model.parameters(), lr=1e-3)
    target_criterion = nn.CrossEntropyLoss()
    target_model.train()

    for epoch in range(5):
        epoch_loss = 0.0
        n_batches = 0
        for input_ids, target_ids in tqdm(train_loader, desc=f"Target pretrain {epoch+1}/5"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            target_optimizer.zero_grad()
            logits, _ = target_model(input_ids)
            # input_ids: (B, T), target_ids: (B, T), logits: (B, T, V)
            # Standard next-token prediction: logits[t] predicts target_ids[t]
            loss = target_criterion(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
            )
            loss.backward()
            target_optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"  Target epoch {epoch+1} loss: {epoch_loss/max(n_batches,1):.4f}")

    target_model.eval()  # Freeze target model

    # Train draft head
    losses = train_draft_head(
        target_model, draft_head, train_loader, draft_config,
        device=device, n_epochs=args.epochs, lr=args.lr,
    )

    # --- Benchmark ---
    print("\n" + "=" * 60)
    print("Benchmarking Speculative Decoding...")
    print("=" * 60)

    results = benchmark_speculative_decoding(
        target_model, draft_head, bench_loader,
        char2idx, idx2char, draft_config,
        device=device, n_steps=args.bench_steps,
    )

    # --- Print results ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Average acceptance rate:   {results['avg_acceptance_rate']:.2%}")
    print(f"  Total accepted tokens:    {results['total_accepted']}")
    print(f"  Total proposed tokens:    {results['total_proposed']}")
    print(f"  Total verification steps: {results['total_steps']}")
    print(f"  Greedy time:              {results['greedy_time']:.3f}s")
    print(f"  Speculative time:         {results['speculative_time']:.3f}s")
    print(f"  Time-based speedup:       {results['time_speedup']:.2f}x")
    print(f"  Theoretical expected tokens per step: {results['theoretical_expected_tokens']:.2f}")
    print(f"  Benchmark samples:        {results['n_samples']}")
    print("=" * 60)

    # Speedup analysis
    alpha = results['avg_acceptance_rate']
    N = draft_config.node_budget - 1
    if alpha > 0 and alpha < 1:
        expected_tokens = (1 - alpha ** (N + 1)) / (1 - alpha)
        # Assuming draft cost c ≈ 0.05 (single forward pass / large verification)
        c = 0.05
        theoretical_speedup = expected_tokens / (N * c + 1)
        print(f"\n  Scaling Analysis (theoretical):")
        print(f"    Expected tokens per iteration (α={alpha:.3f}, N={N}): {expected_tokens:.2f}")
        print(f"    Theoretical speedup (c={c}): {theoretical_speedup:.2f}x")
        print(f"    At α=0: speedup = 1.0x (no benefit)")
        print(f"    At α=1: speedup = {N/(N*c+1):.2f}x (perfect acceptance)")
    print()

    return results


if __name__ == "__main__":
    main()
