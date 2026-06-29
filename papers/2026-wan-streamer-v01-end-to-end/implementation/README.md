# Wan-Streamer v0.1 — Toy Implementation

From-scratch Python implementation of the core architecture from:

> Wan Team, Alibaba Group. "Wan-Streamer v0.1: End-to-end Real-time Interactive
> Foundation Models." arXiv:2606.25041, June 2026.
> [Paper](https://arxiv.org/abs/2606.25041) · [Project page](https://wan-streamer.com/)

## Quick start

```bash
pip install numpy
python3 run.py
```

Output: 6 interactive demos covering all core architectural ideas, streaming inference trace, latency analysis, and metrics.

## What this demonstrates

The paper's five key architectural innovations, implemented from scratch:

| # | Idea | Paper section | What the demo shows |
|---|------|--------------|-------------------|
| 1 | **Multimodal token interleaving** | Sec 2.1 | Unified sequence of text/audio/video input + output tokens |
| 2 | **Block-causal attention** | Sec 2.1, Fig 1 | Attention mask: input=bidirectional within block, output=causal, full cross-block |
| 3 | **Streaming unit inference** | Sec 2.1, Eq 1 | Process interaction in 160ms chunks at 25fps |
| 4 | **Thinker-performer pipeline** | Sec 2.4, Fig 2 | Two-device split: thinker (encode+state+decode), performer (flow matching) |
| 5 | **Conditional flow matching** | Sec 2.1, Eq 2-3 | Audio/video latent denoising via Euler-step flow matching |

## Demos

1. **Key Equations** — Print the paper's core equations (streaming factorization, flow matching)
2. **Block-Causal Attention** — Build and verify the attention mask with ASCII visualization
3. **Flow Matching** — Track a denoising trajectory from noise to clean latent
4. **Attention Pattern Analysis** — How the mask handles different interaction scenarios
5. **Latency Budget** — ASCII breakdown of the ~200ms model-side / ~550ms total latency
6. **Training Pipeline** — The 3-stage training process (pretrain → interaction → distillation)
7. **Streaming Inference** — Full pipeline trace through a 10-turn duplex conversation

## Architecture (paper's design)

```
User Observation (u_k)
  ├── text tokens ──────┐
  ├── audio tokens ──────┼──→ Interleaved Token Sequence ──→ Block-Causal Attention
  └── video tokens ──────┘           │                              │
                                    │                         Unified Transformer
                                    │                              │
Agent Response (y_k)                │                         ┌────┴────┐
  ├── text tokens (AR) ────────────┘                         │ Thinker │ (encode + state + decode)
  ├── audio latent (FM) ──────────────────────────────→      └────┬────┘
  └── video latent (FM) ──────────────────────────────→           │
                                                                    │ KV slice + latents
                                                              ┌────┴────┐
                                                              │Performer│ (flow matching solver)
                                                              └─────────┘
```

## Conversation scenarios

The toy dataset includes a 10-turn duplex conversation that exercises:
- Normal turn-taking (user → agent → user → ...)
- **User interruption** (turn 5: user cuts off agent mid-speech)
- **Agent proactive speaking** (turn 8: agent reacts to user's visual nod)
- Multi-turn context carry-over (weather question → weekend question)

## Results (toy)

```
Avg thinker time:   ~Xms (tiny 64d model)
Avg performer time: ~Xms
Paper target:       ~200ms model-side, ~550ms total (with 350ms network)
```

Latencies are NOT comparable to the paper — the toy model is orders of magnitude smaller. The value is in exercising the correct architectural structure.

## Known gaps / limitations

1. **Toy dimensions** — 64d model vs ~4Kd in real Wan-Streamer. Everything is tiny.
2. **Random embeddings** — No real multimodal encoders/decoders. Tokens are random vectors.
3. **Untrained model** — All weights are random. No actual training loop. The architecture is correct but produces no meaningful outputs.
4. **Simple velocity model** — Flow matching uses a linear model, not the Transformer's velocity heads with proper conditioning.
5. **No real audio/video** — No codec, no visual tokens, no actual generation.
6. **Single-thread** — No CUDA graph capture, no actual two-GPU pipeline.
7. **No turn management** — The model doesn't actually learn when to speak; the pipeline processes units sequentially.
8. **No rolling distillation** — Stage 3 training (self-forcing) is described but not implemented.

## Files

| File | Purpose |
|------|---------|
| `wan_streamer.py` | Core architecture: Transformer, block-causal attention, flow matching, thinker-performer pipeline |
| `data.py` | Toy duplex conversation dataset with streaming unit configs |
| `run.py` | Main runner: 6 interactive demos + full streaming inference |
| `requirements.txt` | `numpy` (only dep) |
| `README.md` | This file |
