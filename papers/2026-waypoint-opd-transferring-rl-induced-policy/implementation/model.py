"""Tiny autoregressive token policy for the Waypoint-OPD toy.

Paper: "Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated
On-Policy Distillation" — Hou et al., arXiv:2610.21601.

An MLP emits next-token logits at every prefix of a length-T response,
conditioned on the task instance features and the tokens generated so far.
Positions 0..T-2 are "scratch" positions (scratch vocab only); position T-1 is
the answer position (two answer tokens only) — enforced with a hard
position-dependent logit mask, so the support at the answer prefix is exactly
the two answer tokens, like a verified final-answer slot in a reasoning trace.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPolicy(nn.Module):
    def __init__(self, feat_dim=6, vocab=16, seq_len=4, hidden=64, seed=0):
        super().__init__()
        self.vocab = vocab
        self.seq_len = seq_len
        torch.manual_seed(seed)  # reproducible init per construction
        self.inst = nn.Linear(feat_dim, hidden)
        self.tok = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(seq_len, hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.head = nn.Linear(hidden, vocab)
        # position-dependent mask: scratch positions -> scratch vocab, last -> answers
        mask = torch.full((seq_len, vocab), float("-inf"))
        mask[: seq_len - 1, : vocab - 2] = 0.0
        mask[seq_len - 1, vocab - 2 :] = 0.0
        self.register_buffer("pmask", mask)

    def logits_all(self, x, seqs):
        """Logits at every prefix of `seqs` (B, T) given instances x (B, F).

        Position t conditions on tokens y_{<t} (teacher forcing on the
        student/teacher's own rollout), matching s_t = (x, y_{<t}). Returns
        (B, T, V) raw logits with the position mask applied.
        """
        B, T = seqs.shape
        pos = torch.arange(T)
        # context: instance features + embedding of all tokens (masked by position
        # via a lower-triangular gather is overkill for an MLP toy; we embed the
        # full prefix by summing embeddings of tokens before t)
        tok_emb = self.tok(seqs)  # (B, T, H)
        ctx = []
        for t in range(T):
            prefix_emb = tok_emb[:, :t].sum(dim=1) if t > 0 else torch.zeros(
                B, tok_emb.shape[-1], device=seqs.device
            )
            ctx.append(self.inst(x) + prefix_emb + self.pos(pos[t]))
        ctx = torch.stack(ctx, dim=1)  # (B, T, H)
        h = self.net(ctx)
        logits = self.head(h) + self.pmask[pos].unsqueeze(0)
        return logits

    def sample(self, x, temperature=1.0, gen=None):
        """Autoregressive rollout (the loop matters: token t conditions on the
        SAMPLED prefix, not a teacher-forced one)."""
        B = x.shape[0]
        T, device = self.seq_len, x.device
        seqs = torch.zeros(B, T, dtype=torch.long, device=device)
        for t in range(T):
            logits = self.logits_all(x, seqs)[:, t]  # conditions on seqs[:, :t]
            probs = F.softmax(logits / temperature, dim=-1)
            if gen is not None:
                seqs[:, t] = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
            else:
                seqs[:, t] = torch.multinomial(probs, 1).squeeze(-1)
        return seqs

    def log_probs(self, x, seqs):
        """log pi(y_t | x, y_<t) for every t — (B, T)."""
        logits = self.logits_all(x, seqs)
        lp = F.log_softmax(logits, dim=-1)
        return lp.gather(-1, seqs.unsqueeze(-1)).squeeze(-1)

    def prefix_logprobs(self, x, seqs):
        """Full next-token log-softmax at every prefix — (B, T, V), detached
        callers decide; keep grads to the caller."""
        return F.log_softmax(self.logits_all(x, seqs), dim=-1)
