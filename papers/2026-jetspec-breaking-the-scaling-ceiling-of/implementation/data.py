"""
Data module for JetSpec: synthetic text dataset for training and benchmarking.

Provides character-level vocabulary from synthetic text, with utilities for:
- Training data generation (sequences of tokens with anchor positions)
- Benchmark data (longer sequences for generation speedup measurement)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Character-level vocabulary from a text corpus
# ---------------------------------------------------------------------------

def build_char_vocab(text: str) -> Tuple[dict, dict]:
    """
    Build character-level vocabulary from text.

    Returns:
        char2idx: mapping char -> int
        idx2char: mapping int -> char
    """
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    return char2idx, idx2char


def encode_text(text: str, char2idx: dict) -> List[int]:
    """Encode text to token ids."""
    return [char2idx[c] for c in text]


def decode_tokens(tokens: List[int], idx2char: dict) -> str:
    """Decode token ids to text."""
    return "".join(idx2char[i] for i in tokens if i in idx2char)


# ---------------------------------------------------------------------------
# Synthetic text generation
# ---------------------------------------------------------------------------

def generate_synthetic_text(
    n_chars: int = 500_000,
    seed: int = 42,
    patterns: Optional[List[str]] = None,
) -> str:
    """
    Generate synthetic text with repetitive patterns and some randomness.

    This mimics a mix of code-like and natural-language-like text with
    enough structure for the model to learn patterns.

    Args:
        n_chars: total characters to generate
        seed: random seed for reproducibility
        patterns: list of pattern strings (if None, uses defaults)

    Returns:
        text: synthetic text string
    """
    rng = np.random.RandomState(seed)

    if patterns is None:
        patterns = [
            # Code-like patterns
            "def function_{0}():\n    return {1}\n",
            "for i in range({0}):\n    x += {1}\n",
            "if x > {0}:\n    y = {1}\nelse:\n    y = {0}\n",
            "class {0}:\n    def __init__(self, x={1}):\n        self.x = x\n",
            # Math-like patterns
            "the result is {0} plus {1} equals {2}\n",
            "given that x = {0}, then y = {1} * {0}\n",
            "we know that {0} + {1} = {2}, so {0} = {2} - {1}\n",
            # Natural language patterns
            "the quick brown fox jumps over the lazy dog\n",
            "in the beginning there was nothing but then something appeared\n",
            "to be or not to be that is the question\n",
            "the answer to life the universe and everything is {0}\n",
        ]

    text_parts = []
    current_len = 0

    while current_len < n_chars:
        pattern = rng.choice(patterns)

        # Fill in placeholders with numbers or words
        filled = pattern
        for placeholder in ["{0}", "{1}", "{2}"]:
            if placeholder in filled:
                val = rng.randint(0, 100)
                filled = filled.replace(placeholder, str(val), 1)

        text_parts.append(filled)
        current_len += len(filled)

    text = "".join(text_parts)
    return text[:n_chars]


def generate_simple_repetitive_text(
    n_chars: int = 100_000,
    vocab_size: int = 26,
    seed: int = 42,
) -> str:
    """
    Generate simple repetitive text using a small alphabet.
    Creates structured sequences with some variability.

    Args:
        n_chars: total characters
        vocab_size: number of unique characters to use
        seed: random seed

    Returns:
        text: string of characters
    """
    rng = np.random.RandomState(seed)
    # Use lowercase letters as vocabulary
    chars = [chr(ord('a') + i) for i in range(min(vocab_size, 26))]

    text_parts = []
    current_len = 0

    while current_len < n_chars:
        # Generate a "word" of 3-8 chars
        word_len = rng.randint(3, 9)
        word = "".join(rng.choice(chars) for _ in range(word_len))
        # Repeat the word 1-4 times with slight variations
        n_repeats = rng.randint(1, 5)
        for _ in range(n_repeats):
            text_parts.append(word + " ")
            current_len += word_len + 1

    return "".join(text_parts)[:n_chars]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticTextDataset(Dataset):
    """
    Dataset of token sequences for training the draft head.

    Each example is a sequence of tokens. For draft head training,
    the first token is treated as the "anchor" and the rest are
    predicted under the tree-causal attention mask.
    """

    def __init__(
        self,
        tokens: List[int],
        seq_len: int = 64,
        stride: int = 32,
    ):
        """
        Args:
            tokens: flat list of token ids
            seq_len: length of each training sequence
            stride: stride for sliding window
        """
        self.seq_len = seq_len
        self.tokens = tokens
        self.stride = stride

        # Create sequences
        self.sequences = []
        for i in range(0, len(tokens) - seq_len, stride):
            self.sequences.append(tokens[i:i + seq_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)  # (seq_len - 1,)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)   # (seq_len - 1,)
        return input_ids, target_ids


class BenchmarkDataset(Dataset):
    """
    Dataset for benchmarking speculative decoding speedup.
    Provides longer prefix sequences for generation.
    """

    def __init__(
        self,
        tokens: List[int],
        prefix_len: int = 128,
        stride: int = 64,
    ):
        self.prefix_len = prefix_len
        self.tokens = tokens
        self.stride = stride

        self.sequences = []
        for i in range(0, len(tokens) - prefix_len, stride):
            self.sequences.append(tokens[i:i + prefix_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def create_dataloaders(
    text: str,
    char2idx: dict,
    train_seq_len: int = 64,
    train_stride: int = 32,
    bench_prefix_len: int = 64,
    bench_stride: int = 32,
    batch_size: int = 32,
    train_ratio: float = 0.9,
):
    """
    Create training and benchmark dataloaders.

    Args:
        text: raw text string
        char2idx: character to index mapping
        train_seq_len: sequence length for training
        train_stride: stride for training sequences
        bench_prefix_len: prefix length for benchmarking
        bench_stride: stride for benchmark sequences
        batch_size: batch size
        train_ratio: fraction of data for training

    Returns:
        train_loader, bench_loader, char2idx, idx2char, vocab_size
    """
    tokens = encode_text(text, char2idx)
    idx2char = {v: k for k, v in char2idx.items()}
    vocab_size = len(char2idx)

    # Split data
    split = int(len(tokens) * train_ratio)
    train_tokens = tokens[:split]
    bench_tokens = tokens[split:]

    train_dataset = SyntheticTextDataset(
        train_tokens,
        seq_len=train_seq_len,
        stride=train_stride,
    )
    bench_dataset = BenchmarkDataset(
        bench_tokens,
        prefix_len=bench_prefix_len,
        stride=bench_stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    bench_loader = DataLoader(
        bench_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, bench_loader, char2idx, idx2char, vocab_size
