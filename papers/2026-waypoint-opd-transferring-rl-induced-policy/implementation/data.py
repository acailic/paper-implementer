"""Synthetic shortcut-learning task for Waypoint-OPD.

Paper: "Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated
On-Policy Distillation" — Hou et al., arXiv:2610.21601.

Task design goal: a teacher RL run whose early and late policy shifts
genuinely OPPOSE at some prefixes, so the paper's temporal-agreement gate has
real structure to detect.

Each instance carries:
  - a subtle "true" feature      (dim 0, scale ~0.7: always encodes the answer),
  - a salient "shortcut" feature (dim 1, scale ~3.0: encodes the answer on
    `agree_frac` of instances — a spurious correlate),
  - gaussian noise dims.

A policy that reads only the shortcut scores `agree_frac`; a policy that reads
the true feature scores 1.0. Both teacher and student start from a
format-only SFT init (they know the response protocol, answer near chance).
GRPO with the true verifier reward then learns features in SALIENCE order:
the shortcut first (fast — it is 4x more visible), the truth later (slow —
it is subtle). On the "conflict" instances where the shortcut disagrees with
the truth, the early shift RAISES the wrong token and the late shift LOWER it
— a genuine directional reversal during a single RL run, which is exactly
the endpoint-ambiguity the paper motivates.

Responses are length-T token sequences over a small vocab: T-1 unconstrained
"scratch" tokens (a stand-in for unverified reasoning tokens) followed by one
answer token. Reward is binary and depends only on the answer token, like a
verifiable math/code reward that checks only the final answer.
"""

import torch


class ShortcutTask:
    def __init__(
        self,
        n_train=2048,
        n_val=256,
        n_test=512,
        feat_dim=8,
        agree_frac=0.65,
        true_scale=0.45,
        shortcut_scale=3.0,
        seq_len=4,
        vocab=16,
        seed=0,
    ):
        assert vocab >= 4 and seq_len >= 1
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.vocab = vocab
        # answer token ids (last position); ids below are scratch tokens
        self.ans_ids = (vocab - 2, vocab - 1)

        g = torch.Generator().manual_seed(seed)

        def make(n):
            x = torch.randn(n, feat_dim, generator=g)
            label = (torch.rand(n, generator=g) > 0.5).long()  # 0/1 uniform
            agree = (torch.rand(n, generator=g) < agree_frac).long()
            shortcut_label = torch.where(agree == 1, label, 1 - label)
            x[:, 0] = (2.0 * label - 1).float() * true_scale + 0.05 * x[:, 0]
            x[:, 1] = (2.0 * shortcut_label - 1).float() * shortcut_scale + 0.05 * x[:, 1]
            return x, label, agree

        self.train_x, self.train_y, self.train_agree = make(n_train)
        self.val_x, self.val_y, self.val_agree = make(n_val)
        self.test_x, self.test_y, self.test_agree = make(n_test)

    def sample_prompts(self, batch, gen):
        idx = torch.randint(0, self.train_x.shape[0], (batch,), generator=gen)
        return self.train_x[idx], self.train_y[idx], self.train_agree[idx]

    def reward(self, seqs, labels):
        """Binary verifier: 1 iff the final token is the correct answer token."""
        ans = seqs[:, -1]
        return torch.where(
            labels == 1, (ans == self.ans_ids[1]).float(), (ans == self.ans_ids[0]).float()
        )

    def format_only_seq(self, n, gen):
        """Format-only demonstrations for SFT inits (teacher and student):
        correct response shape, random scratch, random answer."""
        seq = torch.randint(0, self.vocab - 2, (n, self.seq_len), generator=gen)
        seq[:, -1] = torch.randint(0, 2, (n,), generator=gen) + self.ans_ids[0]
        return seq
