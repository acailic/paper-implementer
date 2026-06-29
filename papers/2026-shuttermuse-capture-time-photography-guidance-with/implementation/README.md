"""
ShutterMuse — Capture-Time Photography Guidance with MLLMs
==========================================================

A simplified but runnable implementation of the ShutterMuse model (Li et al., 2026).
Based on Qwen3-VL-8B, uses SFT + GRPO for capture-time photography guidance.

Two tasks in one model:
  - Photographer-side: 3-way decision (refine/keep/reject) + crop box prediction
  - Subject-side: COCO-17 keypoint visibility scoring + pose rationale

This implementation replaces Qwen3-VL-8B with a lightweight Vision-Language Transformer
(ViT + GPT-2 decoder) so it can run on any machine without downloading a 8B model.
All architectural concepts (attention masking, interleaved tokens, GRPO) are preserved.

Files:
  model.py     — Vision-Language model with decision heads + attention masking
  data.py      — Synthetic data generation (random crops, aesthetic scores)
  train.py     — SFT + GRPO training pipeline
  README.md    — Documentation

Usage:
  python train.py --mode sft --epochs 3 --batch_size 4
  python train.py --mode grpo --epochs 1 --batch_size 2 --num_rollouts 4
"""

__version__ = "0.1.0"
