"""Pointer policy for ChainWorld.

> Paper: "Apodex 1.1: Scaling Agentic Intelligence for Complex Work"
> (arXiv:2608.23283). Our from-scratch miniature — a small transformer over
> file slots whose heads score READ_i / COLLECT_i / SUBMIT (pointer-style
> action head over workspace entries, as the workspace is the action space).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data import FILE_FEATS, GLOBAL_FEATS


class PointerPolicy(nn.Module):
    def __init__(self, n_files: int, d: int = 48, n_layers: int = 2,
                 heads: int = 2):
        super().__init__()
        self.n = n_files
        self.d = d
        self.file_in = nn.Linear(FILE_FEATS, d)
        self.glob_in = nn.Linear(GLOBAL_FEATS, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=2 * d, dropout=0.0,
            batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.read_head = nn.Linear(d, 1)
        self.collect_head = nn.Linear(d, 1)
        self.submit_head = nn.Linear(d, 1)

    def forward(self, file_feats: torch.Tensor,
                glob_feats: torch.Tensor) -> torch.Tensor:
        """file_feats (B, N, F), glob_feats (B, G) -> logits (B, 2N+1)."""
        tok_files = self.file_in(file_feats)
        tok_glob = self.glob_in(glob_feats).unsqueeze(1)
        h = self.enc(torch.cat([tok_files, tok_glob], dim=1))
        hf = h[:, : self.n, :]
        hg = h[:, self.n, :]
        logits = torch.cat([
            self.read_head(hf).squeeze(-1),
            self.collect_head(hf).squeeze(-1),
            self.submit_head(hg),
        ], dim=1)
        return logits
