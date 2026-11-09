"""Mechanism miniature for: Vidu S2: Real-Time Interactive, Editable, and
Spatial Video Generation (Zhang et al., arXiv:2609.11638).

This is NOT a reproduction of Vidu S2 (proprietary weights/architecture).
It re-implements, at toy scale on synthetic signals, the *control-interface
mechanisms* the paper describes:

  1. block-causal streaming with per-chunk conditions c_i and a reference r
     injected as attention tokens visible to every chunk (temporal reference
     conditioning)                [model.RefCondStreamModel]
  2. dynamic reference injection: r can be swapped mid-stream; appearance
     should follow from the next chunk on                       [rollout()]
  3. frame-aligned attention for streaming editing: target frame i may attend
     ONLY to source frame i (+ the global reference), vs free cross-attention
                                                                [source_mask]
  4. the SRF training schedule vs Teacher Forcing / Diffusion Forcing /
     detached Self-Forcing                                        [train.py]

Everything runs on CPU in seconds: chunks are 8-dim synthetic vectors
(4 motion dims + 4 reference-appearance dims), not video latents.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

MAX_POS = 96  # max stream length (learned chunk-position embeddings)


class RefCondStreamModel(nn.Module):
    """Block-causal chunk model with reference-token conditioning.

    Per chunk i the model receives a noisy chunk x~_i, its noise level t_i and
    its condition c_i, encodes them to e_i, and attends (block-causally) over
    reference tokens + encodings of chunks <= i to predict the clean chunk.

    ref_injection:
      "all"   - reference tokens are attention keys for every chunk
                (Vidu S2 design: r shared across all segments)
      "first" - reference tokens are keys only for chunk 0 (control: an
                I2V-style init cannot adopt a mid-stream reference swap)
    source mode (editing):
      use_source=False         - pure generation (Avatar-style)
      source_mask="aligned"    - target i attends to source encoding i ONLY
                                 (Vidu S2 frame-aligned attention)
      source_mask="free"       - target i attends to source encodings <= i
                                 (unrestricted cross-attention control)
    """

    def __init__(
        self,
        d_chunk=8,
        d_cond=4,
        d_ref=4,
        d_hidden=48,
        n_ref_tokens=2,
        ref_injection="all",
        use_source=False,
        source_mask="aligned",
    ):
        super().__init__()
        self.d_chunk = d_chunk
        self.ref_injection = ref_injection
        self.use_source = use_source
        self.source_mask = source_mask
        self.n_ref_tokens = n_ref_tokens
        self.d_hidden = d_hidden

        self.chunk_enc = nn.Sequential(
            nn.Linear(d_chunk + 1 + d_cond, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_hidden), nn.Tanh(),
        )
        self.pos_emb = nn.Embedding(MAX_POS, d_hidden)
        if use_source:
            self.source_enc = nn.Sequential(
                nn.Linear(d_chunk, d_hidden), nn.Tanh(),
                nn.Linear(d_hidden, d_hidden), nn.Tanh(),
            )
        self.ref_enc = nn.Sequential(
            nn.Linear(d_ref, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_hidden * n_ref_tokens),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_chunk),
        )

    def forward(self, x_noisy, t, c, r, src=None, detach_history_kv=False):
        """One pass over a whole window — also the SRF 'replay' call site.

        x_noisy: (B,N,d_chunk) noisy chunks; t: (B,N) noise levels;
        c: (B,N,d_cond) per-chunk conditions; r: (B,d_ref) reference;
        src: (B,N,d_chunk) source stream for the editing model, else None;
        detach_history_kv: if True, chunk encodings enter attention keys and
            values as CONSTANTS (a Self-Forcing-style frozen KV cache: no
            gradient from any chunk's loss reaches how earlier chunks are
            encoded). If False (SRF replay), the K/V encodings stay in the
            graph, so the loss at chunk i backpropagates into the encodings
            of chunks j < i — the cross-block gradient flow SRF reclaims.

        Returns x_hat: (B,N,d_chunk), e: (B,N,d_hidden) chunk encodings.
        """
        B, N, _ = x_noisy.shape
        dev = x_noisy.device
        pos = torch.arange(N, device=dev)
        e = self.chunk_enc(torch.cat([x_noisy, t.unsqueeze(-1), c], dim=-1))
        e = e + self.pos_emb(pos)[None]                       # (B,N,H)
        rho = self.ref_enc(r).view(B, self.n_ref_tokens, self.d_hidden)

        keys, vals, extra_allowed = [rho], [rho], []
        if self.use_source:
            assert src is not None, "editing model needs src"
            s = self.source_enc(src) + self.pos_emb(pos)[None]  # (B,N,H)
            keys.append(s)
            vals.append(s)
            if self.source_mask == "aligned":
                extra_allowed = torch.eye(N, dtype=torch.bool, device=dev)
            elif self.source_mask == "free":
                extra_allowed = torch.tril(torch.ones(N, N, dtype=torch.bool,
                                                      device=dev))
            else:
                raise ValueError(self.source_mask)
        keys.append(e.detach() if detach_history_kv else e)
        vals.append(e.detach() if detach_history_kv else e)
        K = torch.cat(keys, dim=1)                            # (B,N+R(+N),H)
        V = torch.cat(vals, dim=1)
        scores = torch.bmm(e, K.transpose(1, 2)) / math.sqrt(self.d_hidden)

        R = self.n_ref_tokens
        n_src = N if self.use_source else 0
        total = R + n_src + N
        qi = torch.arange(N, device=dev)[:, None]
        kj = torch.arange(total, device=dev)[None]
        allowed = torch.ones(N, total, dtype=torch.bool, device=dev)
        off = R + n_src
        allowed &= (kj - off) <= qi        # block-causal over chunk encodings
        if self.ref_injection == "first":
            allowed[:, :R] = qi == 0       # reference seen by chunk 0 only
        if self.use_source:
            allowed[:, R:R + N] = extra_allowed

        scores = scores.masked_fill(~allowed[None], float("-inf"))
        ctx = torch.softmax(scores, dim=-1) @ V
        x_hat = self.head(torch.cat([e, ctx], dim=-1))
        return x_hat, e


@torch.no_grad()
def rollout(model, c, r, n_chunks, t_gen=1.0, swap_at=None, r_new=None,
            first_chunk=None):
    """Autoregressive streaming rollout — the inference procedure.

    Each chunk is generated in ONE step from pure noise (t_gen=1.0),
    conditioned on the reference tokens, its condition c_i, and the model's
    own generated history (the exposure-bias regime of a distilled streaming
    model at test time).

    swap_at/r_new implement dynamic reference injection: from chunk swap_at
    on, the reference tokens come from r_new. Vidu S2's "upload a new
    reference image at any moment".

    first_chunk: optional ground-truth first chunk (I2V-style start).

    Note: history is re-encoded each step rather than kept as a literal KV
    cache; encodings are deterministic functions of (chunk, t, c, position),
    so this is numerically equivalent to caching them, just O(N^2) compute —
    irrelevant at toy scale.
    """
    model.eval()
    B, N, d_cond = c.shape
    assert N >= n_chunks
    dev = c.device
    hist_x, hist_t, hist_c = [], [], []
    outs = []
    for i in range(n_chunks):
        if i == 0 and first_chunk is not None:
            x_prev, t_prev = first_chunk, torch.zeros(B, device=dev)
        else:
            x_prev = torch.randn(B, model.d_chunk, device=dev)
            t_prev = torch.full((B,), t_gen, device=dev)
        hist_x.append(x_prev)
        hist_t.append(t_prev)
        hist_c.append(c[:, i])
        r_i = r_new if (swap_at is not None and i >= swap_at) else r
        xin = torch.stack(hist_x, dim=1)
        tin = torch.stack(hist_t, dim=1)
        cin = torch.stack(hist_c, dim=1)
        x_hat, _ = model(xin, tin, cin, r_i)
        x_i = x_hat[:, -1]
        outs.append(x_i)
        # the model's own output becomes the history for the next chunk
        hist_x[-1], hist_t[-1] = x_i, torch.full((B,), t_gen, device=dev)
    return torch.stack(outs, dim=1)
