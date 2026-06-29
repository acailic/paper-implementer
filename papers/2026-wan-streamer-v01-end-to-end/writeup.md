# Wan-Streamer v0.1 — Explain It To a Friend

> **Bosnian/Serbian version:** [writeup-sr.md](writeup-sr.md)

---

## One-Paragraph Version

Wan-Streamer is a single Transformer from Alibaba that does real-time face-to-face AI interaction — it listens to your speech and watches your video, then responds with synchronized speech and video of its own animated avatar, all without chaining together separate ASR, language model, TTS, and avatar rendering modules. Every component is designed to be strictly causal (processes left-to-right, like reading a book), so the model can stream 160ms chunks at 25 FPS while maintaining full conversational context. At inference it splits across two GPUs (a "thinker" that handles perception and decoding, and a "performer" that does the expensive video/audio generation), achieving ~200ms model latency and ~550ms total including network round-trip. The current v0.1 runs at 192p resolution — a deliberate proof of concept — and they claim higher resolutions scale straightforwardly.

---

## The Problem

Imagine you're on a video call with an AI. You want it to feel like talking to a real person: you speak, it listens and nods, you interrupt, it stops, it notices you picked up a coffee and comments on it. Now think about how you'd build that.

The obvious approach — and what most systems actually do — is to wire together a bunch of specialized modules: a speech recognizer hears you, a language model decides what to say, a text-to-speech engine generates audio, and an avatar renderer animates a face to match. This works okay for turn-based chat, but it falls apart for natural conversation:

- **Latency stacks up.** Each module takes time, and they run sequentially. The user waits.
- **No sync.** The avatar's lip movements are generated after the speech, so they're always slightly off. You can patch this with post-processing, but it never feels quite right.
- **No listening behavior.** While you're talking, the system is just recording — the avatar either freezes or loops a generic animation. A real person would be nodding, making eye contact, reacting.
- **Interruptions are awkward.** To handle "wait, let me finish that thought," the system has to tear down the TTS pipeline mid-word and start over.

The fundamental issue: these cascaded systems were never designed for streaming. They were designed for turn-based interaction where you finish talking, then the AI thinks, then the AI talks. That's not how humans work.

---

## The Idea

Wan-Streamer's answer: throw away all the modules. Build one single Transformer that handles everything — text, audio, and video on both the input and output sides. Design every single component to be causal (can only look at the past, never the future). Then the whole thing can run as one continuous stream, just like human conversation.

The key design principle they call the "streaming contract": every component must operate causally, every newly observed chunk must be usable immediately, and every generated chunk must be emitted and committed back into the interaction history. No waiting for a complete utterance, no batching, no post-hoc sync.

---

## How It Works (Intuition)

Think of the model as processing a single long sequence that interleaves everything:

```
[user text] [user audio frames] [user video frames] [agent text] [agent audio latent] [agent video latent] [user text] ...
```

At each 160ms step (4 frames at 25 FPS):

1. **See the user.** Audio and video VAEs compress the user's latest chunk into latent tokens. These are strictly causal — they only look at past frames, not future ones.

2. **Think.** The Transformer runs a causal pass over the new tokens plus the full conversation history (stored as a KV cache). It produces text output (word by word, like a language model) and predicts audio+video latents (using flow matching, like a diffusion model but faster).

3. **Speak and show.** Audio and video decoders turn the latents into actual sound and pixels. These get sent to the user immediately.

4. **Remember.** The generated latents are added back to the conversation history, so the next step has full context.

The tricky part is the flow matching for audio and video. Instead of running a slow diffusion process, they use conditional flow matching — starting from noise and iteratively refining it toward the target. The audio and video are denoised *jointly* (conditioned on the same context), which means lip movements and speech are naturally synchronized because they were generated together, not stitched together afterward.

For inference, they split the model across two GPUs:
- **Thinker GPU:** encodes user input, runs the Transformer for text/state, decodes previous step's output
- **Performer GPU:** runs the flow-matching solver for the next step's audio+video

They communicate via KV-cache exchange. The key insight: at step k, the thinker decodes step k-1's output while the performer generates step k's latents. This pipelining hides most of the latency.

---

## What Surprised Me

1. **No ablations at all.** For a paper making strong architectural claims (causal VAEs, block-causal attention, joint flow matching, thinker-performer split), having zero ablations is striking. It's a v0.1 tech report, but even basic comparisons would help.

2. **The latency is genuinely competitive.** ~200ms model-side for full audio+video output is impressive, even at 192p. Speech-only systems like Moshi report similar numbers without having to generate video.

3. **The "listening behavior" is learned, not hardcoded.** The model wasn't explicitly trained with "nod when the user speaks" labels. It learned this from the interleaved interaction data where human conversational partners naturally show these behaviors. That's pretty cool.

4. **The paper is unusually honest about measurement boundaries.** They explicitly point out that comparing their full end-to-end latency to other systems' partial metrics (model-only, first-packet, renderer-only) is misleading, and they tabulate what each system actually measures. That's good scientific practice.

5. **Rolling distillation for reducing train-test mismatch.** During distillation, the student model is trained on its own generated history rather than teacher outputs, using distribution matching to align trajectories. This is a nice technique borrowed from the self-forcing literature.

6. **192p and they're proud of it.** Most papers would try to hide a 192p output. They put it front and center as a design choice — proof of concept for the architecture, not a quality claim. Refreshing.

---

## References

- Paper: https://arxiv.org/abs/2606.25041
- Project page: https://wan-streamer.com/
- Related: Wan2.1 video generation (the base architecture family)
- Related: Self-forcing and distribution matching (distillation techniques used in stage 3)
- Related: Moshi (full-duplex speech, no video), VASA-1 (audio-driven avatar, no dialogue), TalkingMachines (audio-driven video with external LLM)
