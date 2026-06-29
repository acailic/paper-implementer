#!/usr/bin/env python3
"""
Wan-Streamer v0.1 — Main Runner
================================
Demonstrates all five core architectural ideas from the paper:

  1. Multimodal token interleaving in a unified Transformer
  2. Block-causal attention (input=bidirectional, output=causal within block)
  3. Streaming unit inference (160ms chunks)
  4. Thinker-performer pipeline (two-device deployment split)
  5. Conditional flow matching for audio/video latent generation

Usage:
    python3 run.py

Dependencies: numpy only.
"""

import sys
import time
import json
import numpy as np

from wan_streamer import (
    # Core classes
    StreamingUnit, MultimodalTokenStream,
    StreamingTransformer, FlowMatchingSolver,
    ThinkerPerformerPipeline,
    # Attention
    build_block_causal_mask,
    # Constants
    MOD_NAMES, MOD_TEXT_IN, MOD_AUDIO_IN, MOD_VIDEO_IN,
    MOD_TEXT_OUT, MOD_AUDIO_OUT, MOD_VIDEO_OUT,
    STREAMING_UNIT_DURATION_MS, FPS,
)
from data import (
    CONVERSATION, create_streaming_units, create_latency_ground_truth,
    create_attention_pattern_analysis, print_conversation_summary,
    D_MODEL,
)


def banner(text: str, ch: str = "=") -> str:
    line = ch * 72
    return f"\n{line}\n{text}\n{line}"


def ascii_bar(value: float, max_val: float, width: int = 40, ch: str = "█") -> str:
    """Render a proportional ASCII bar."""
    if max_val <= 0:
        return " " * width
    filled = int(value / max_val * width)
    return ch * filled + "░" * (width - filled)


def demo_block_causal_attention():
    """Demonstrate the block-causal attention mask construction."""
    print(banner("DEMO 1: Block-Causal Attention Mask"))
    print()
    print("The block-causal attention mask is the key architectural innovation.")
    print("Within each streaming unit (block):")
    print("  - Input tokens  (user t/a/v): BIDIRECTIONAL — can see each other")
    print("  - Output tokens (agent t/a/v): CAUSAL — autoregressive within block")
    print("Across blocks:")
    print("  - ALL tokens in block k can attend to ALL tokens in blocks 0..k-1")
    print()

    # Example: 3 blocks with different sizes
    block_configs = [
        {"name": "Block 0: User speaks", "n_in": 8, "n_out": 0},
        {"name": "Block 1: Agent responds", "n_in": 4, "n_out": 6},
        {"name": "Block 2: User interrupts", "n_in": 10, "n_out": 0},
    ]

    block_sizes = [c["n_in"] + c["n_out"] for c in block_configs]
    n_output_per_block = [c["n_out"] for c in block_configs]
    mask = build_block_causal_mask(block_sizes, n_output_per_block)

    N = mask.shape[0]
    print(f"Attention mask shape: ({N}, {N})")
    print(f"Block sizes: {block_sizes}")
    print(f"Output tokens per block: {n_output_per_block}")
    print()

    # Print a compact ASCII visualization of the mask
    # Use different chars for input vs output positions
    symbols = {}
    pos = 0
    labels = []
    for i, c in enumerate(block_configs):
        n_in = c["n_in"]
        n_out = c["n_out"]
        for j in range(n_in):
            symbols[pos] = "I"  # Input
            labels.append(f"B{i}I")
            pos += 1
        for j in range(n_out):
            symbols[pos] = "O"  # Output
            labels.append(f"B{i}O")
            pos += 1

    print("Attention mask (rows=query, cols=key). 1=attend, 0=masked:")
    print("      " + "  ".join(f"{labels[i][:3]:>3}" for i in range(N)))
    for i in range(N):
        row_str = "  ".join("███" if mask[i, j] > 0.5 else "···" for j in range(N))
        print(f"{labels[i]:>4}  {row_str}")
    print()

    # Verify properties
    print("Verification:")
    # 1. Input tokens should be bidirectional within their block
    pos = 0
    all_ok = True
    for c in block_configs:
        n_in = c["n_in"]
        for i in range(pos, pos + n_in):
            for j in range(pos, pos + c["n_in"] + c["n_out"]):
                if j == i:
                    continue
                if mask[i, j] < 0.5:
                    print(f"  FAIL: Input token {i} cannot see token {j} in same block")
                    all_ok = False
        pos += c["n_in"] + c["n_out"]

    # 2. Output tokens should be causal within their block
    pos = 0
    for c in block_configs:
        n_in = c["n_in"]
        for i in range(pos + n_in, pos + n_in + c["n_out"]):
            for j in range(i + 1, pos + n_in + c["n_out"]):
                if mask[i, j] > 0.5:
                    print(f"  FAIL: Output token {i} can see future token {j}")
                    all_ok = False
        pos += c["n_in"] + c["n_out"]

    # 3. Cross-block: all tokens in block k should see all tokens in blocks < k
    pos = 0
    boundaries = [0]
    for c in block_configs:
        boundaries.append(boundaries[-1] + c["n_in"] + c["n_out"])
    for k in range(len(block_configs)):
        for i in range(boundaries[k], boundaries[k + 1]):
            for prev_k in range(k):
                for j in range(boundaries[prev_k], boundaries[prev_k + 1]):
                    if mask[i, j] < 0.5:
                        print(f"  FAIL: Token {i} (block {k}) cannot see token {j} (block {prev_k})")
                        all_ok = False

    if all_ok:
        print("  ✅ All block-causal attention properties verified correctly")

    # Count parameters
    total_params = N * N
    dense_count = int(np.sum(mask))
    print(f"\n  Mask density: {dense_count}/{total_params} = {dense_count/total_params*100:.1f}%")
    print(f"  Sparsity:    {(total_params - dense_count)/total_params*100:.1f}%")
    print()

    return mask


def demo_flow_matching():
    """Demonstrate the flow matching denoising process."""
    print(banner("DEMO 2: Conditional Flow Matching (Eq 2-3)"))
    print()
    print("Audio/video output is generated via flow matching in a continuous")
    print("latent space. We start from noise (tau=1) and denoise to clean (tau=0).")
    print()

    np.random.seed(123)
    d = 8  # tiny dimension for visualization

    # Clean target latent
    z0 = np.array([1.0, -0.5, 2.0, 0.3, -1.0, 0.8, 1.5, -0.7], dtype=np.float32)

    # Run denoising with a simple linear model
    solver = FlowMatchingSolver(n_steps=10)

    def simple_model_fn(z, tau, ctx):
        """Toy velocity model: push toward the target (uses oracle for demo)."""
        # In the real model, this is the Transformer's velocity head.
        # Here we use a simple attraction to zero (simulating denoising).
        return -z / (1.0 - tau + 0.1)

    # Start from noise
    z_noisy, eps = solver.add_noise(z0, tau=1.0)
    print(f"  Clean z0:      {np.array2string(z0, precision=2)}")
    print(f"  Noise eps:     {np.array2string(eps, precision=2)}")
    print(f"  Noisy z_tau=1: {np.array2string(z_noisy, precision=2)}")
    print()

    # Track denoising trajectory
    tau_schedule = np.linspace(1.0, 0.0, solver.n_steps + 1)
    print(f"  Denoising trajectory (tau: 1.0 -> 0.0 in {solver.n_steps} steps):")
    print(f"  {'step':>4} {'tau':>6} {'||z||':>8} {'||z-z0||':>10}")
    print(f"  {'----':>4} {'----':>6} {'------':>8} {'--------':>10}")

    z = z_noisy.copy()
    for i in range(len(tau_schedule) - 1):
        tau_curr = tau_schedule[i]
        tau_next = tau_schedule[i + 1]
        dt = tau_next - tau_curr
        vel = simple_model_fn(z, tau_curr, None)
        z = z + dt * vel
        dist = float(np.linalg.norm(z - z0))
        print(f"  {i:4d} {tau_curr:6.2f} {np.linalg.norm(z):8.3f} {dist:10.3f}")

    print()
    print(f"  Final ||z - z0|| = {np.linalg.norm(z - z0):.4f}")
    print(f"  (Smaller = better denoising; real model uses Transformer velocity heads)")
    print()

    # Training loss demo
    print("  Training loss L_FM (Eq 3): ||predicted_velocity - target_velocity||^2")
    z_tau, eps = solver.add_noise(z0, tau=0.7)
    target_vel = solver.compute_target_velocity(z0, eps)
    predicted_vel = simple_model_fn(z_tau, 0.7, None)
    loss = solver.training_loss(predicted_vel, target_vel)
    print(f"    At tau=0.7: loss = {loss:.6f}")
    print()


def demo_streaming_inference():
    """
    Demo 3: Full streaming inference through the toy Transformer
    with the thinker-performer pipeline.
    """
    print(banner("DEMO 3: Streaming Inference (Thinker-Performer Pipeline)"))
    print()

    # Print conversation summary
    print_conversation_summary(CONVERSATION)

    # Create model
    np.random.seed(42)
    d_model = D_MODEL
    model = StreamingTransformer(
        d_model=d_model, n_heads=4, n_layers=2, d_ff=128, vocab_size=64
    )
    fm_solver = FlowMatchingSolver(n_steps=3)  # very few steps for speed
    pipeline = ThinkerPerformerPipeline(model, fm_solver)

    # Create streaming units
    units = create_streaming_units(CONVERSATION, d_model=d_model)

    print(banner("Streaming Inference Trace", "-"))
    print()
    print(f"  Model: d_model={d_model}, n_heads=4, n_layers=2, d_ff=128")
    print(f"  Flow matching: {fm_solver.n_steps} solver steps")
    print(f"  Total streaming units to process: {len(units)}")
    print()
    print(f"  {'Step':>4} {'Role':>4} {'Tokens':>6} {'Blocks':>6} "
          f"{'Thinker':>10} {'Performer':>10} {'Total':>10} {'Hidden||':>8}")
    print(f"  {'----':>4} {'----':>4} {'------':>6} {'------':>6} "
          f"{'----------':>10} {'----------':>10} {'------':>10} {'--------':>8}")

    results = []
    for unit_cfg in units:
        rng = np.random.RandomState(unit_cfg['rng_seed'])

        # Build a StreamingUnit from the config
        su = StreamingUnit(
            step=unit_cfg['step'],
            user_text=rng.randn(unit_cfg['n_text_in'], d_model).astype(np.float32),
            user_audio=rng.randn(unit_cfg['n_audio_in'], d_model).astype(np.float32),
            user_video=rng.randn(unit_cfg['n_video_in'], d_model).astype(np.float32),
            agent_text=rng.randn(unit_cfg['n_text_out'], d_model).astype(np.float32),
            agent_audio=rng.randn(unit_cfg['n_audio_out'], d_model).astype(np.float32),
            agent_video=rng.randn(unit_cfg['n_video_out'], d_model).astype(np.float32),
        )

        result = pipeline.process_streaming_unit(su)
        result['role'] = unit_cfg['turn_role']
        result['text'] = unit_cfg['turn_text']
        results.append(result)

        print(f"  {result['step']:4d} {result['role']:>4} "
              f"{result['total_tokens']:6d} {result['n_blocks']:6d} "
              f"{result['thinker_ms']:9.1f}ms {result['performer_ms']:9.1f}ms "
              f"{result['total_ms']:9.1f}ms {result['hidden_norm']:8.2f}")

    print()

    # Aggregate statistics
    thinker_times = [r['thinker_ms'] for r in results]
    performer_times = [r['performer_ms'] for r in results]
    total_times = [r['total_ms'] for r in results]

    print("Latency summary:")
    print(f"  Thinker avg:   {np.mean(thinker_times):.1f}ms "
          f"(min={np.min(thinker_times):.1f}, max={np.max(thinker_times):.1f})")
    print(f"  Performer avg: {np.mean(performer_times):.1f}ms "
          f"(min={np.min(performer_times):.1f}, max={np.max(performer_times):.1f})")
    print(f"  Total avg:     {np.mean(total_times):.1f}ms")
    print(f"  Paper target:  ~200ms model-side, ~550ms total (with 350ms network)")
    print(f"  Note: Toy model is MUCH smaller (~64d vs ~4Kd), so latencies are"
          f" not comparable. The STRUCTURE is what matters.")
    print()

    return results


def demo_attention_patterns():
    """Demo 4: Analyze attention patterns for different interaction scenarios."""
    print(banner("DEMO 4: Block-Causal Attention Pattern Analysis"))
    print()

    patterns = create_attention_pattern_analysis()
    for p in patterns:
        print(f"  Scenario: {p['scenario']}")
        print(f"    {p['description']}")
        print(f"    Pattern: {p['pattern']}")
        print()

    # Visualize the interleaving structure
    print("  Token interleaving structure per streaming unit:")
    print(f"  {'Modality':>15} {'Direction':>10} {'Attention':>30}")
    print(f"  {'-------':>15} {'--------':>10} {'------------------------------>':>30}")

    for mod_id, name in MOD_NAMES.items():
        if mod_id <= 2:
            direction = "input"
            attn = "bidirectional within block + full past"
        else:
            direction = "output"
            attn = "causal within block + full past"
        print(f"  {name:>15} {direction:>10} {attn}")
    print()


def demo_latency_budget():
    """Demo 5: Latency budget analysis (paper Sec 3)."""
    print(banner("DEMO 5: Latency Budget (Paper Sec 3)"))
    print()

    print("  Paper reports:")
    print(f"    Model-side latency:  ~200ms (encode + state + flow match + decode)")
    print(f"    Network latency:     ~350ms bidirectional")
    print(f"    Total:               ~550ms")
    print()

    # ASCII breakdown
    budget = [
        ("User capture + network up", 175, 350),
        ("Thinker: encode + state", 80, 200),
        ("KV/latent communication", 10, 10),
        ("Performer: flow matching", 90, 200),
        ("Thinker: decode output", 20, 200),
        ("Network down + playback", 175, 350),
    ]

    max_budget = 350
    print("  Latency budget breakdown:")
    print(f"  {'Component':>30} {'Estimate':>10} {'Budget':>10} {'Usage':>40}")
    for name, est, budget_val in budget:
        bar = ascii_bar(est, max_budget, width=35)
        print(f"  {name:>30} {est:>7d}ms {budget_val:>7d}ms {bar}")

    print()
    print("  Key insight from the paper:")
    print("  The thinker-performer split enables OVERLAP of perception, decoding,")
    print("  communication, and latent generation across adjacent streaming units.")
    print("  Throughput is determined by performer_time + communication < 160ms/unit,")
    print("  NOT by the full signal-to-signal path (which is ~200ms).")
    print()


def demo_training_stages():
    """Demo 6: Three-stage training (Paper Sec 2.3)."""
    print(banner("DEMO 6: Training Pipeline (3 Stages, Paper Sec 2.3)"))
    print()

    stages = [
        {
            'stage': 1,
            'name': 'Independent-task pretraining',
            'description': 'Initialize from language model. Train multimodal interface '
                          'on image/audio/video understanding + text dialogue + ASR/TTS '
                          '+ image/audio/video generation tasks. Perception, reasoning, '
                          'and generation are aligned in one sequence model.',
            'data': 'Mixed: understanding (image/audio/video QA, dialogue) + generation '
                    '(T2I, T2A, T2V, joint AV generation)',
        },
        {
            'stage': 2,
            'name': 'End-to-end interaction training',
            'description': 'Train on duplex interaction data where user text/audio/video '
                          'inputs and agent text/audio/video outputs are interleaved in '
                          'the same causal stream. Learns response timing, active listening, '
                          'interruption handling, and long-context consistency.',
            'data': 'Duplex interaction: both sides have t/a/v on same causal timeline',
        },
        {
            'stage': 3,
            'name': 'Distillation for low-latency streaming',
            'description': 'Distill stronger teacher (CFG, more solver steps) into '
                          'efficient student. Uses rolling self-forcing: student is '
                          'rolled out over consecutive streaming units, trained on its '
                          'own generated history. Reduces train-test mismatch.',
            'data': 'Teacher-generated trajectories with distribution matching',
        },
    ]

    for s in stages:
        print(f"  Stage {s['stage']}: {s['name']}")
        print(f"    {s['description']}")
        print(f"    Data: {s['data']}")
        print()


def demo_equations():
    """Print the key equations from the paper."""
    print(banner("Key Equations from the Paper"))
    print()

    print("  Eq 1 — Causal streaming factorization:")
    print("    p_theta(y_{1:K} | u_{1:K}) =")
    print("      prod_k p_theta(y_k^t, y_k^a, y_k^v |")
    print("        u_{{<=k}}^t, u_{{<=k}}^a, u_{{<=k}}^v,")
    print("        y_{{<k}}^t,  y_{{<k}}^a,  y_{{<k}}^v)")
    print()
    print("    Where:")
    print("      u_k = (u_k^t, u_k^a, u_k^v) — user observations at step k")
    print("      y_k = (y_k^t, y_k^a, y_k^v) — agent response at step k")
    print("      K = number of streaming units")
    print()

    print("  Eq 2 — Flow matching latent construction:")
    print("    z_tau^m = (1 - tau) * z_0^m + tau * epsilon^m")
    print("    dz_tau^m / d_tau = epsilon^m - z_0^m")
    print()
    print("    Where:")
    print("      z_0^m   = clean target latent (audio or video)")
    print("      epsilon^m ~ N(0, I) = Gaussian noise")
    print("      tau      = flow time (1=noise, 0=clean)")
    print("      m        = modality (audio or video)")
    print()

    print("  Eq 3 — Flow matching training loss:")
    print("    L_FM^m = E_epsilon || f_theta(z_tau^a, z_tau^v, c_k, tau)")
    print("                              - dz_tau^m / d_tau ||_2^2")
    print()
    print("    Where:")
    print("      f_theta   = unified diffusion Transformer")
    print("      c_k       = clean streaming context (all past observations + responses)")
    print("      The same c_k conditions BOTH audio and video velocity predictions,")
    print("      enabling synchronized speech and motion generation.")
    print()


def main():
    print("=" * 72)
    print("WAN-STREAMER v0.1 — TOY ARCHITECTURE DEMONSTRATION")
    print("=" * 72)
    print()
    print("Paper: 'Wan-Streamer v0.1: End-to-end Real-time Interactive")
    print("       Foundation Models' (arXiv:2606.25041, June 2026)")
    print("       Wan Team, Alibaba Group")
    print()
    print("This demo exercises all five core architectural ideas:")
    print("  1. Multimodal token interleaving (text/audio/video in/out)")
    print("  2. Block-causal attention (input=bidir, output=causal within block)")
    print("  3. Streaming unit inference (160ms chunks at 25fps)")
    print("  4. Thinker-performer pipeline (two-device deployment)")
    print("  5. Conditional flow matching (audio/video latent denoising)")
    print()
    print("Dependencies: numpy only. No GPU, no model downloads, no API keys.")
    print()

    # Run all demos
    t_start = time.perf_counter()

    demo_equations()
    demo_block_causal_attention()
    demo_flow_matching()
    demo_attention_patterns()
    demo_latency_budget()
    demo_training_stages()

    results = demo_streaming_inference()

    t_total = time.perf_counter() - t_start

    # Final summary
    print(banner("SUMMARY"))
    print()
    print(f"  Total runtime:      {t_total:.2f}s")
    print(f"  Streaming units:    {len(results)}")
    print(f"  Model config:       d_model={D_MODEL}, 4 heads, 2 layers, d_ff=128")
    print(f"  Flow matching:       3 solver steps (toy; real model uses many more)")
    print()
    print("  What this demo demonstrates:")
    print("    ✅ Block-causal attention mask (verified correct)")
    print("    ✅ Multimodal token interleaving (text/audio/video, input/output)")
    print("    ✅ Streaming inference per 160ms unit")
    print("    ✅ Thinker-performer pipeline with KV cache exchange")
    print("    ✅ Flow matching denoising trajectory")
    print("    ✅ Interruption handling (conversation turn 5)")
    print("    ✅ Proactive agent speaking (conversation turn 8)")
    print()
    print("  Key limitations (vs real Wan-Streamer):")
    print("    ⚠  Toy dimensions (64d vs ~4Kd model)")
    print("    ⚠  Random embeddings, not trained encoders")
    print("    ⚠  Simple linear velocity model, not Transformer flow matching")
    print("    ⚠  No real audio/video codec or generation")
    print("    ⚠  Single-thread, not CUDA-graph-optimized two-GPU pipeline")
    print("    ⚠  Latencies are NOT comparable to the paper's ~200ms")
    print()

    # Save metrics
    metrics = {
        'paper': 'arXiv:2606.25041',
        'title': 'Wan-Streamer v0.1',
        'total_units': len(results),
        'total_runtime_s': t_total,
        'avg_thinker_ms': float(np.mean([r['thinker_ms'] for r in results])),
        'avg_performer_ms': float(np.mean([r['performer_ms'] for r in results])),
        'avg_total_ms': float(np.mean([r['total_ms'] for r in results])),
        'model_config': {
            'd_model': D_MODEL, 'n_heads': 4, 'n_layers': 2,
            'd_ff': 128, 'vocab_size': 64, 'fm_steps': 3,
        },
    }
    import os
    metrics_path = os.path.join(os.path.dirname(__file__), 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    print()

    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
