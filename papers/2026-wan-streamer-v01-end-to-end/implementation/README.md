# Wan-Streamer v0.1 — Simplified Implementation

This is a **minimal runnable implementation** of the Wan-Streamer block-causal multimodal streaming model described in:

> Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models (Alibaba, June 2026)

## Architecture Overview

The model is a single causal Transformer that jointly processes text, audio, and video tokens using **block-causal attention**. Each streaming unit (160ms) contains interleaved text, audio, and video tokens that can cross-attend within their block but cannot attend to future blocks.

### Key Components

| Component | Description |
|-----------|-------------|
| **Causal Audio Encoder/Decoder** | 1D causal convolutions + linear projection to compress/reconstruct audio waveforms into/from latent tokens |
| **Causal Video Encoder/Decoder** | 2D causal convolutions + linear projection to compress/reconstruct video frames into/from latent tokens |
| **Block-Causal Transformer** | Unified DiT-style transformer with block-causal attention mask; processes interleaved text/audio/video tokens |
| **Flow Matching** | Conditional flow matching for joint audio-video generation (velocity prediction with Euler ODE solver) |
| **Streaming Inference** | Thinker-Performer pipeline with KV-cache and chunked processing (simulated single-GPU) |

### File Structure

```
model.py    — All model components (encoders, decoders, transformer, block-causal attention)
data.py      — Synthetic toy dataset generating text, audio waveforms, and video frames
train.py     — Training loop with CE loss (text) + FM loss (audio/video)
```

## Running

```bash
pip install -r requirements.txt
python train.py
```

This runs a short training loop on synthetic data and validates that all components (block-causal attention, causal encoders/decoders, flow matching, streaming inference) work end-to-end.

## Simplifications

- Small model dimensions (d_model=128, 4 layers, 4 heads) for fast iteration
- Synthetic toy dataset instead of real multimodal data
- Single-GPU simulation of Thinker-Performer pipeline (no actual GPU-to-GPU transfer)
- Audio: 1D waveform chunks (8000 samples = 500ms at 16kHz)
- Video: 4 grayscale frames per chunk at 32×32 (simplified from paper's 192p)
- No classifier-free guidance or distillation (Stage 3)
