"""
data.py — Character-level toy dataset for iLLaDA demonstration.

Uses a small corpus (nursery rhymes + sentences) with a character-level
vocabulary.  Sequences are packed up to `max_len` characters.  Each training
example is a tensor of character ids plus the length of the actual content
(without padding).

Key paper concept demonstrated:
  • Diffusion LLMs can train on repeated data without degradation (the
    "super data learner" property).  We deliberately use a tiny corpus so
    that every epoch iterates over the same data many times.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Tiny corpus — intentionally small so epochs repeat the same data many times
# ---------------------------------------------------------------------------
CORPUS = (
    "the cat sat on the mat. "
    "the dog ran in the park. "
    "a quick brown fox jumps over the lazy dog. "
    "she sells sea shells by the sea shore. "
    "peter piper picked a peck of pickled peppers. "
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood. "
    "mary had a little lamb whose fleece was white as snow. "
    "jack and jill went up the hill to fetch a pail of water. "
    "humpty dumpty sat on a wall humpty dumpty had a great fall. "
    "twinkle twinkle little star how i wonder what you are. "
    "ring a ring a roses a pocket full of posies. "
    "baa baa black sheep have you any wool. "
    "the sun sets in the west and rises in the east. "
    "all that glitters is not gold. "
    "to be or not to be that is the question. "
) * 3  # repeat 3× to make overfitting/learning-on-repeated-data visible

# Build character-level vocabulary (printable ascii subset)
CHARS = sorted(set(CORPUS))
CHAR_TO_IDX = {ch: i for i, ch in enumerate(CHARS)}
IDX_TO_CHAR = {i: ch for ch, i in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(CHARS)

# Special tokens
PAD_IDX = VOCAB_SIZE
MASK_IDX = VOCAB_SIZE + 1
EOS_IDX = VOCAB_SIZE + 2

TOTAL_VOCAB = VOCAB_SIZE + 3  # real chars + PAD + MASK + EOS


class CharDataset(Dataset):
    """Character-level dataset packed into fixed-length sequences."""

    def __init__(self, corpus: str = CORPUS, max_len: int = 64):
        self.max_len = max_len
        # Encode the whole corpus
        ids = [CHAR_TO_IDX[ch] for ch in corpus if ch in CHAR_TO_IDX]
        # Pad / chunk into sequences of length max_len
        self.sequences: list[torch.Tensor] = []
        for i in range(0, len(ids), max_len):
            chunk = ids[i : i + max_len]
            # pad to max_len if needed
            chunk = chunk + [PAD_IDX] * (max_len - len(chunk))
            self.sequences.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def encode(text: str) -> list[int]:
    """Encode a string to character ids (no padding)."""
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]


def decode(ids: list[int]) -> str:
    """Decode character ids back to string (stop at MASK/EOS only)."""
    chars = []
    for i in ids:
        if i == MASK_IDX or i == EOS_IDX:
            break
        if i == PAD_IDX:
            chars.append(" ")  # PAD renders as space for readability
        elif i in IDX_TO_CHAR:
            chars.append(IDX_TO_CHAR[i])
        else:
            chars.append("?")
    return "".join(chars)
