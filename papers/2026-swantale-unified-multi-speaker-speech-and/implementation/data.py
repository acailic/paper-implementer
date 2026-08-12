"""Synthetic data for the SwanTale toy re-implementation.

Paper: "SwanTale: Unified Multi-Speaker Speech and Audio Generation..."
ArXiv:  https://arxiv.org/abs/2608.02023

There is no public SwanData-Caption (~70M caption records) and no 48kHz audio
available, so we generate synthetic latent targets that mimic the structure the
real SwanVAE would produce: short continuous sequences (the "latent frames")
paired with a coarse tokenized caption and a quality flag.

Two task variants mirror SwanTale's task unification (breakdown Eq. 6-9):
  * instruct (τ=inst): full caption, generate ALL frames (mask = 0).
  * zero-shot (τ=zero): content-only caption, first `prompt_len` frames kept
    clean as a reference prompt (mask = 1 on prompt, 0 on generation).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from model import TASK_INST, TASK_ZERO


def make_speaker_bank(n_speakers: int = 8, latent_channels: int = 4, seed: int = 0):
    """Each 'speaker' is a fixed latent template (a mean trajectory)."""
    g = torch.Generator().manual_seed(seed)
    bank = []
    for _ in range(n_speakers):
        # a smooth random trajectory: low-frequency sinusoid per channel
        phases = torch.rand(latent_channels, generator=g) * 6.28
        amps = 0.5 + torch.rand(latent_channels, generator=g)
        bank.append((phases, amps))
    return bank


def sample_latent(bank, T: int, latent_channels: int, rng: torch.Generator):
    """Sample a latent target for a random speaker, length T."""
    phases, amps = bank[torch.randint(len(bank), (1,), generator=rng).item()]
    t = torch.arange(T).float() / max(T - 1, 1)
    x = torch.zeros(T, latent_channels)
    for c in range(latent_channels):
        x[:, c] = amps[c] * torch.sin(2 * math.pi * (1 + c) * t + phases[c])
    # add small noise so targets aren't perfectly periodic
    x = x + 0.05 * torch.randn(T, latent_channels, generator=rng)
    return x


# need math imported here for the sin above
import math  # noqa: E402


class SyntheticLatentDataset(Dataset):
    """Synthetic latent + caption pairs for instruct and zero-shot tasks.

    Each item returns a dict with:
      x_star      (T, C)  clean latent target
      caption     (L,)    tokenized caption (toy: random tokens, but speaker
                          id is embedded so the model can learn the mapping)
      quality     int     quality flag (we force HIGH at inference)
      task        str     'inst' or 'zero'
      mask        (T, 1)  reference mask (1 on prompt frames for zero-shot)
    """

    def __init__(
        self,
        n_samples: int = 256,
        seq_len: int = 32,
        latent_channels: int = 4,
        caption_len: int = 8,
        n_speakers: int = 8,
        prompt_len: int = 8,
        tasks=(TASK_INST, TASK_ZERO),
        seed: int = 0,
    ):
        self.seq_len = seq_len
        self.latent_channels = latent_channels
        self.caption_len = caption_len
        self.prompt_len = prompt_len
        self.tasks = tasks
        self.bank = make_speaker_bank(n_speakers, latent_channels, seed)
        self.rng = torch.Generator().manual_seed(seed)
        # Pre-generate per-speaker caption tokens (deterministic) so the model
        # can in principle learn a caption->speaker mapping.
        self.speaker_captions = []
        for _ in range(n_speakers):
            self.speaker_captions.append(torch.randint(0, 200, (caption_len,), generator=self.rng))
        self.items = []
        for i in range(n_samples):
            spk = i % n_speakers
            task = tasks[i % len(tasks)]
            x = sample_latent(self.bank, seq_len, latent_channels, self.rng)
            cap = self.speaker_captions[spk].clone()
            # quality: mostly 'high'(2) for clean, occasionally 'normal'(1)
            qual = 2 if torch.rand(1, generator=self.rng).item() > 0.3 else 1
            mask = torch.zeros(seq_len, 1)
            if task == TASK_ZERO:
                mask[: self.prompt_len] = 1.0
            self.items.append(
                {"x_star": x, "caption": cap, "quality": qual, "task": task, "mask": mask, "speaker": spk}
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    """Stack a list of dataset items into batched tensors."""
    x_star = torch.stack([b["x_star"] for b in batch])  # (B,T,C)
    caption = torch.stack([b["caption"] for b in batch])  # (B,L)
    quality = torch.tensor([b["quality"] for b in batch], dtype=torch.long)  # (B,)
    mask = torch.stack([b["mask"] for b in batch])  # (B,T,1)
    tasks = [b["task"] for b in batch]
    speakers = torch.tensor([b["speaker"] for b in batch], dtype=torch.long)
    return {
        "x_star": x_star,
        "caption": caption,
        "quality": quality,
        "mask": mask,
        "task": tasks,
        "speaker": speakers,
    }
