"""
Toy duplex conversation dataset for Wan-Streamer.
===================================================
Simulates a multi-turn audio-visual conversation between a user and
an AI agent. Each turn is decomposed into 160ms streaming units with
multimodal tokens (text, audio, video for both user and agent).

In the real system:
  - User audio comes from a microphone (16kHz codec tokens)
  - User video comes from a camera (compressed visual tokens at 25fps)
  - Agent text is generated autoregressively (discrete tokens)
  - Agent audio is generated via flow matching (codec latents)
  - Agent video is generated via flow matching (video latents)

In this toy:
  - All tokens are random embeddings of fixed dimension
  - We simulate the STRUCTURE (number of tokens per modality per unit)
  - The conversation has meaningful annotations (who's speaking, what about)
  - We track user vs agent turns and simulate interruption events
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass


# Configuration
D_MODEL = 64        # embedding dimension (real model: ~4096)
FPS = 25             # frames per second
UNIT_DURATION_MS = 160  # 160ms per streaming unit
FRAMES_PER_UNIT = FPS * UNIT_DURATION_MS // 1000  # = 4 frames

# Tokens per modality per streaming unit (toy sizes)
# In real Wan-Streamer: text ~10-100, audio ~hundreds of codec tokens,
# video ~hundreds of compressed visual tokens per unit
TOKENS_TEXT_IN_PER_UNIT = 4
TOKENS_AUDIO_IN_PER_UNIT = 8
TOKENS_VIDEO_IN_PER_UNIT = 8
TOKENS_TEXT_OUT_PER_UNIT = 6
TOKENS_AUDIO_OUT_PER_UNIT = 8
TOKENS_VIDEO_OUT_PER_UNIT = 8


@dataclass
class ConversationTurn:
    """One side of the conversation (user or agent)."""
    role: str  # "user" or "agent"
    text: str  # what is being said
    n_units: int  # how many streaming units this turn spans
    has_interruption: bool = False  # was this turn interrupted?
    has_proactive: bool = False  # agent proactive comment (no user prompt)


# A sample duplex conversation that exercises key scenarios:
#  1. Normal turn-taking (user speaks, agent responds)
#  2. User interrupts agent mid-speech
#  3. Agent proactive speaking (reacts to visual event)
#  4. Multi-turn with context carry-over
CONVERSATION: List[ConversationTurn] = [
    # Turn 1: User greets
    ConversationTurn("user", "Hello! Can you see me?", n_units=3),

    # Turn 2: Agent responds (perceives user, generates synchronized response)
    ConversationTurn("agent",
                     "Yes, I can see you! You look like you're in a bright room. "
                     "How can I help you today?",
                     n_units=4),

    # Turn 3: User asks a question
    ConversationTurn("user", "I was wondering about the weather forecast for tomorrow.", n_units=4),

    # Turn 4: Agent starts responding...
    ConversationTurn("agent",
                     "Tomorrow's forecast shows partly cloudy skies with a high "
                     "of 24 degrees. You might want to bring",
                     n_units=5),

    # Turn 5: User INTERRUPTS agent mid-speech (tests full-duplex behavior)
    ConversationTurn("user",
                     "Wait, actually I changed my mind. I want to know about the weekend.",
                     n_units=3, has_interruption=True),

    # Turn 6: Agent adapts to interruption (paper: model learns this from data)
    ConversationTurn("agent",
                     "No problem! For the weekend, Saturday looks sunny and warm, "
                     "around 26 degrees. Sunday might see some rain in the afternoon.",
                     n_units=5),

    # Turn 7: User nods (video-only, minimal audio) — tests active listening
    ConversationTurn("user", "[nods silently]", n_units=2),

    # Turn 8: Agent proactive — reacts to user's visual cue (nod)
    ConversationTurn("agent",
                     "I see you nodding. Shall I look up indoor activities "
                     "for Sunday in case of rain?",
                     n_units=4, has_proactive=True),

    # Turn 9: User confirms
    ConversationTurn("user", "Yes please, that would be great.", n_units=2),

    # Turn 10: Agent responds with detailed info
    ConversationTurn("agent",
                     "There are several great options: the city museum has a new "
                     "exhibition, the cinema downtown is showing a new release, "
                     "and the indoor market is open all day Sunday.",
                     n_units=6),
]


def create_streaming_units(
    conversation: List[ConversationTurn],
    d_model: int = D_MODEL,
    seed: int = 42,
) -> List[Dict]:
    """
    Convert the conversation into a list of streaming unit configs.
    
    Each unit dict contains:
      - step: global streaming unit index
      - turn_role: "user" or "agent"
      - turn_text: what's being said in this turn
      - is_idle: True if this is a listening/idle unit (agent waiting)
      - is_interruption: True if user interrupted
      - is_proactive: True if agent initiated without prompt
      - token counts per modality
    """
    units = []
    global_step = 0

    for turn in conversation:
        for local_step in range(turn.n_units):
            is_first = (local_step == 0)

            unit = {
                'step': global_step,
                'turn_role': turn.role,
                'turn_text': turn.text,
                'is_first_unit_of_turn': is_first,
                'has_interruption': turn.has_interruption and is_first,
                'has_proactive': turn.has_proactive,
                # Token counts per modality
                'n_text_in': TOKENS_TEXT_IN_PER_UNIT if turn.role == "user" else 2,
                'n_audio_in': TOKENS_AUDIO_IN_PER_UNIT if turn.role == "user" else 2,
                'n_video_in': TOKENS_VIDEO_IN_PER_UNIT if turn.role == "user" else 2,
                'n_text_out': TOKENS_TEXT_OUT_PER_UNIT if turn.role == "agent" else 0,
                'n_audio_out': TOKENS_AUDIO_OUT_PER_UNIT if turn.role == "agent" else 0,
                'n_video_out': TOKENS_VIDEO_OUT_PER_UNIT if turn.role == "agent" else 0,
                'd_model': d_model,
                'rng_seed': seed + global_step,
            }
            units.append(unit)
            global_step += 1

    return units


def create_latency_ground_truth(
    conversation: List[ConversationTurn],
) -> List[Dict]:
    """
    Ground truth for latency measurements.
    
    In the real Wan-Streamer:
      - Model-side: ~200ms (encode + state update + flow matching + decode)
      - Network: ~350ms bidirectional
      - Total: ~550ms
    """
    gt = []
    for i, turn in enumerate(conversation):
        gt.append({
            'turn_idx': i,
            'role': turn.role,
            'text': turn.text,
            'expected_model_ms': 200.0,
            'expected_total_ms': 550.0,
            'is_interruption': turn.has_interruption,
            'is_proactive': turn.has_proactive,
        })
    return gt


def create_attention_pattern_analysis() -> List[Dict]:
    """
    Analysis of the block-causal attention pattern for different scenarios.
    """
    return [
        {
            'scenario': 'Normal turn (user speaks)',
            'description': 'User input tokens within the block are bidirectional '
                          '(can see each other). They can also see ALL past context.',
            'pattern': 'input_tokens: bidirectional within block + full past',
        },
        {
            'scenario': 'Normal turn (agent responds)',
            'description': 'Agent output tokens are causal (autoregressive within '
                          'block). They see all input tokens in current block + full past.',
            'pattern': 'output_tokens: causal within block + bidirectional input + full past',
        },
        {
            'scenario': 'User interrupts agent',
            'description': 'New user tokens arrive in a new block. They can see '
                          'ALL previous context including the agent speech they '
                          'interrupted. The model learns to stop generating.',
            'pattern': 'new_block: full access to interrupted agent output',
        },
        {
            'scenario': 'Agent proactive speaking',
            'description': 'Agent can generate output based on visual events '
                          'in the user video input without waiting for speech.',
            'pattern': 'output conditioned on video_in within same block',
        },
        {
            'scenario': 'Cross-block streaming',
            'description': 'Block k can attend to blocks 0..k-1 fully. Within '
                          'block k, the pattern depends on input vs output.',
            'pattern': 'cross_block: full visibility; intra_block: mixed bidir/causal',
        },
    ]


def print_conversation_summary(conversation: List[ConversationTurn]):
    """Print a readable summary of the conversation."""
    print("=" * 72)
    print("TOY DUPLEX CONVERSATION")
    print("=" * 72)

    total_units = sum(t.n_units for t in conversation)
    user_units = sum(t.n_units for t in conversation if t.role == "user")
    agent_units = sum(t.n_units for t in conversation if t.role == "agent")
    total_duration_ms = total_units * UNIT_DURATION_MS

    n_user_turns = sum(1 for t in conversation if t.role == "user")
    n_agent_turns = sum(1 for t in conversation if t.role == "agent")

    print(f"  Turns:           {len(conversation)} ({n_user_turns} user, {n_agent_turns} agent)")
    print(f"  Streaming units: {total_units} ({user_units} user, {agent_units} agent)")
    print(f"  Duration:        {total_duration_ms / 1000:.1f}s @ {UNIT_DURATION_MS}ms/unit, {FPS}fps")
    print(f"  Interruptions:   {sum(1 for t in conversation if t.has_interruption)}")
    print(f"  Proactive:       {sum(1 for t in conversation if t.has_proactive)}")
    print()

    for i, turn in enumerate(conversation):
        prefix = f"[{'USER' if turn.role == 'user' else 'AGNT'}]"
        duration = turn.n_units * UNIT_DURATION_MS
        flags = []
        if turn.has_interruption:
            flags.append("INTERRUPT")
        if turn.has_proactive:
            flags.append("PROACTIVE")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        text = turn.text[:60] + "..." if len(turn.text) > 60 else turn.text
        print(f"  T{i+1:2d} {prefix} [{duration:3d}ms, {turn.n_units} units]{flag_str}")
        print(f"       \"{text}\"")
        print()
