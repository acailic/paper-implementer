# Wan-Streamer v0.1 — Reading Notes

**Paper:** Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models
**Authors:** Wan Team (Alibaba Group), June 2026
**ArXiv:** https://arxiv.org/abs/2606.25041
**Project:** https://wan-streamer.com/

---

## First Impressions

This is an ambitious paper from Alibaba's Wan team. They've built a single Transformer that handles text, audio, and video as both inputs AND outputs — no external ASR, TTS, avatar animation, or video generation modules. Everything in one model, running as a causal stream. The demos on their site show a 192p avatar doing face-to-face interaction in what appears to be real time.

The scope is massive. Most "real-time" dialogue systems are either speech-only (Moshi, GPT-4o voice) or they cascade a language model → TTS → avatar renderer. Wan-Streamer argues that this cascaded approach fundamentally can't achieve natural full-duplex behavior because the modules weren't trained together. The avatar doesn't know what the language model is planning, the TTS doesn't know what the face is doing, etc.

The resolution is 192p though — that's very low. They explicitly say this is a proof of concept and scaling is straightforward. I buy that argument for the architecture, but the demos do look a bit rough.

## Problem (in my own words)

Current real-time AI assistants are built by bolting together separate modules: a speech recognizer, a language model, a text-to-speech engine, and maybe an avatar renderer. Each module has its own latency, and they communicate through intermediate representations (usually text). This creates problems:

1. **Latency adds up** — each module boundary introduces delay
2. **Error accumulation** — mistakes in ASR cascade into downstream modules
3. **No native synchronization** — lip sync and facial expressions are patched on after the fact
4. **Can't do full-duplex** — when the user interrupts, the system has to tear down and rebuild its pipeline
5. **No visual listening behavior** — most systems freeze or loop an animation while the user speaks

The core claim: streamability needs to be a *modeling constraint* from the start, not a serving optimization bolted on at the end.

## Key things I need to understand better

- **Block-causal attention:** How exactly does this work for interleaving modalities with different token rates? The paper mentions it but doesn't give tons of detail on the mask structure.
- **Flow matching for joint audio-video generation:** They condition both audio and video velocity predictions on the same causal context. How do they prevent one modality from dominating the other?
- **Rolling distillation:** The self-forcing + distribution matching approach for reducing train-test mismatch sounds interesting but is described briefly.
- **KV-cache exchange:** The thinker-performer split preserves unified state through KV exchange. What's the communication overhead? How big are these caches?
- **Model size:** The paper doesn't explicitly state the parameter count. It's initialized from a Qwen LM and seems large-ish but the exact number isn't given.
- **Training data:** They mention a broad mixture but give no specific dataset sizes, compositions, or curation details. This is a big gap.
- **Quantitative quality metrics:** No FID, FVD, ASR metrics, or user studies for quality assessment. Everything is qualitative or latency-focused.

## What's clear

- The architecture is coherent and well-motivated. Building everything causal from day one is the right call for streaming.
- The thinker-performer split is clever — it lets them keep one model while parallelizing the expensive flow-matching on a separate GPU.
- The three-stage training (pretrain → interaction → distillation) is sensible.
- The latency numbers (~200ms model, ~550ms total) are competitive given that this includes synchronized video output.

## What's unclear or concerning

- No code release, no model weights, no public API to test.
- 192p is very low — "scaling is straightforward" is a claim, not a result.
- No ablations. At all. This is a big red flag for understanding what actually matters.
- No quantitative quality comparison to other systems. The only metric tables are about latency/scope.
- The paper reads more like a system description / tech report than a research paper with rigorous experiments.
- How much of the "natural" behavior (nodding, listening, interruption) is learned vs. emergent vs. baked into the training data?

## Verdict

Ambitious architecture, clean design philosophy, but light on empirical validation. It's a v0.1 in every sense — the proof of concept is there, but the evidence that each design choice matters is missing. Would love to see ablations, quality metrics, and a higher-resolution version.
