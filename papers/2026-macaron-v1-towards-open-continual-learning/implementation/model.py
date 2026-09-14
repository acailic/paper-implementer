"""
Macaron-V1 toy re-implementation — model side.

Paper: "Macaron-V1: Towards Open Continual Learning with Self-Improvement
and Mixture-of-LoRA" — Mind Lab, 2026, arXiv:2608.09819.

Written from scratch from our own ../breakdown.md (no reference code copied).

What this file implements:

1. A small decoder-only transformer ("the frozen base").
2. Mixture-of-LoRA over that frozen base: every Linear in the transformer
   blocks is wrapped with N LoRA slots (dW = (alpha/r) * B @ A); exactly one
   adapter is active per forward pass — one per specialist (L0 chat+router,
   L1 agent, L2 coding, L3 gui), mirroring the paper's MoL architecture.
   A "fat" single adapter with the summed parameter budget is also
   supported, for the paper's own missing ablation (MoL vs single LoRA).
3. The Proxy serving loop: Route -> Answer -> Summary per user turn, with
   per-adapter "own-views" rebuilt from an append-only conversation
   timeline (own turns verbatim, other specialists' turns collapsed to a
   short summary), plus tool-result stickiness.
4. Constrained decoding for routing: only the 4 canonical labels are legal
   continuations (L0 is the router; no separate router model, no keyword
   rules).
5. A simple extractive summarizer standing in for the <=192-token summary
   (toy scale, no big LM available offline).

Everything is CPU-friendly and tiny on purpose.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------- config

SPECIALISTS = ["L0", "L1", "L2", "L3"]
SPECIALIST_NAMES = {"L0": "chat", "L1": "agent", "L2": "coding", "L3": "gui"}

# Per-specialist context header — the toy analog of the paper's HCP-carried
# system prompts (a harness *config*, not weights).
SPECIALIST_HEADERS = {
    "L0": "<chat>",
    "L1": "<agent>",
    "L2": "<coding>",
    "L3": "<gui>",
}


@dataclass
class MoLConfig:
    vocab_size: int = 96          # toy vocab (see data.py)
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    max_len: int = 256            # covers prompt + answer + summary
    dropout: float = 0.0          # paper: LoRA dropout 0
    lora_rank: int = 8
    lora_alpha: float = 16.0      # alpha/r = 2, same ratio as the paper
    n_specialists: int = 4        # L0 chat/router, L1 agent, L2 coding, L3 gui


# ----------------------------------------------------------------- base model


class SelfAttention(nn.Module):
    """Plain multi-head self-attention built from 4 Linears (q, k, v, out)
    so that LoRA slots can be attached to every projection, as in the paper
    (q_a, q_b, kv_a, kv_b, o)."""

    def __init__(self, cfg: MoLConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, attn_mask):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att + attn_mask            # additive float mask: -inf = blocked
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y)


class Block(nn.Module):
    def __init__(self, cfg: MoLConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyBase(nn.Module):
    """The frozen base model (stands in for GLM-5.2 744B / Qwen3.6)."""

    def __init__(self, cfg: MoLConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # active adapter slot, switched by the Proxy at request time
        self.active_adapter = 0

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)[None]
        mask = torch.full((T, T), float("-inf"), device=idx.device)
        mask = torch.triu(mask, diagonal=1)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        x = self.ln_f(x)
        return self.head(x)


# ------------------------------------------------------------- LoRA machinery


class AdapterLinear(nn.Module):
    """Frozen base Linear + N LoRA slots; applies ONLY the active slot.

    dW = (alpha/r) * B @ A with A ~ kaiming init and B = 0, so every adapter
    starts as the identity (plain frozen base) — standard LoRA init.
    """

    def __init__(self, base: nn.Linear, cfg: MoLConfig, n_adapters: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = cfg.lora_rank
        self.scale = cfg.lora_alpha / cfg.lora_rank
        d_out, d_in = base.weight.shape
        # one (A, B) pair per adapter slot; ParameterList indexing returns
        # leaf Parameters, so per-slot optimizers work directly.
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.empty(self.rank, d_in)) for _ in range(n_adapters)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_out, self.rank)) for _ in range(n_adapters)]
        )
        for i in range(n_adapters):
            nn.init.kaiming_uniform_(self.lora_A[i].data, a=math.sqrt(5))
        self.active = 0  # mutated by the owning model's active_adapter

    def forward(self, x):
        a = self.lora_A[self.active]
        b = self.lora_B[self.active]
        return self.base(x) + self.scale * ((x @ a.T) @ b.T)


def attach_adapters(model: TinyBase, cfg: MoLConfig) -> None:
    """Replace every Linear inside the transformer blocks (attention in/out
    projections and MLP) with AdapterLinear slots. Embeddings, final LayerNorm
    and the LM head stay shared and frozen — as in the paper."""

    def rec(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, AdapterLinear(child, cfg, cfg.n_specialists))
            elif isinstance(child, (nn.Sequential, nn.ModuleList, SelfAttention)):
                rec(child)

    for blk in model.blocks:
        rec(blk)


def set_active_adapter(model: TinyBase, idx: int) -> None:
    """Route: select exactly one LoRA for the whole forward pass."""
    model.active_adapter = idx
    for m in model.modules():
        if isinstance(m, AdapterLinear):
            m.active = idx


def lora_parameters(model: TinyBase, adapter_idx: int):
    """Yield ONLY adapter `adapter_idx`'s A/B tensors (base stays frozen)."""
    for m in model.modules():
        if isinstance(m, AdapterLinear):
            yield m.lora_A[adapter_idx]
            yield m.lora_B[adapter_idx]


def count_lora_params(model: TinyBase) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, AdapterLinear):
            n += m.lora_A[0].numel() + m.lora_B[0].numel()
    return n


# -------------------------------------------------------------- proxy serving


@dataclass
class Turn:
    """One entry of the append-only conversation timeline."""
    user: str
    specialist: str                 # which LoRA answered ("L0".."L3")
    answer: str                     # verbatim specialist answer
    summary: str                    # <= summary_budget token stand-in
    was_tool_result: bool = False   # marks turns produced by tool stickiness


class Proxy:
    """Route -> Answer -> Summary, per user turn (the paper's serving loop).

    - Route: L0 scores the request under a constrained-decoding grammar:
      only the 4 canonical label tokens are legal next tokens. We pick the
      argmax over those 4 logits ("24-token decode budget" collapses to 1
      constrained step at toy scale).
    - Answer: the selected specialist answers from its OWN-VIEW — its own
      past turns verbatim, every other specialist's turns collapsed to one
      summary message, current user turn verbatim.
    - Summary: the answering specialist emits a short summary; the Proxy
      stores it server-side only (never shown to the client).
    - Stickiness: if the previous turn was a tool result, routing is locked
      to the same specialist for this turn (no route, no summary).
    """

    def __init__(self, model: TinyBase, tokenizer, summary_tokens: int = 12,
                 route_from_own_view: bool = False):
        self.model = model
        self.tok = tokenizer
        self.summary_tokens = summary_tokens
        self.timeline: List[Turn] = []
        self.route_from_own_view = route_from_own_view

    # -- own-view reconstruction (the load-bearing serving detail) --------
    def own_view(self, specialist: str, current_user: str) -> str:
        """The specialist's private context: its own turns verbatim, every
        other specialist's turns collapsed to one summary message. The
        per-specialist header is the toy analog of the HCP-carried system
        prompt (harness config, not weights) — it MUST match training or we
        get exactly the train–serve divergence the paper warns about.

        Format detail that makes KV prefix reuse work (paper's Layer A): an
        own turn is rendered EXACTLY as the request that produced it —
        'user: u\\nassistant: answer' — so the k-th visit's context is a
        byte-identical extension of the (k-1)-th visit's full context."""
        parts: List[str] = [SPECIALIST_HEADERS[specialist]]
        for t in self.timeline:
            parts.append(f"user: {t.user}")
            if t.specialist == specialist:
                parts.append(f"assistant: {t.answer}")
            else:
                parts.append(f"assistant: {t.summary}")  # collapsed
        parts.append(f"user: {current_user}")
        parts.append("assistant: ")
        return "\n".join(parts)

    def full_history(self, current_user: str, specialist: str = "L0") -> str:
        parts = [SPECIALIST_HEADERS[specialist]]
        for t in self.timeline:
            parts.append(f"user: {t.user}")
            parts.append(f"assistant: {t.answer}")
        parts.append(f"user: {current_user}")
        parts.append("assistant: ")
        return "\n".join(parts)

    # -- routing -----------------------------------------------------------
    def route(self, current_user: str) -> str:
        """Constrained decoding with L0 as the router: the prompt ends with
        'route: L' and the ONLY legal next tokens are the digits 0-3 — a
        grammar with exactly 4 legal labels, like the paper's 24-token
        constrained decode budget. No separate router model, no keywords."""
        set_active_adapter(self.model, 0)
        ctx = self.own_view("L0", current_user) if self.route_from_own_view \
            else f"user: {current_user}"
        prompt = ctx + "\nroute: L"
        ids = torch.tensor([self.tok.encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(ids)[0, -1]
        digit_ids = [self.tok.stoi[d] for d in "0123"]
        legal = logits[torch.tensor(digit_ids)]
        return SPECIALISTS[int(legal.argmax())]

    # -- one full turn ------------------------------------------------------
    def step(self, user_text: str, true_label: Optional[str] = None,
             is_tool_result: bool = False, gen_limit: int = 40,
             force_label: Optional[str] = None) -> Turn:
        """One Route -> Answer -> Summary turn. `force_label` bypasses
        routing (oracle-routing diagnostics, e.g. isolating the summary
        handoff from router errors)."""
        # 1) Route (with tool-result stickiness short-circuit)
        if force_label is not None:
            chosen = force_label
        elif self.timeline and self.timeline[-1].was_tool_result:
            chosen = self.timeline[-1].specialist  # locked
        else:
            chosen = self.route(user_text)

        # 2) Answer from the specialist's own-view
        view = self.own_view(chosen, user_text)
        answer = self.tok.generate(self.model, view, adapter=chosen,
                                   max_new=gen_limit)

        # 3) Summary (server-side only)
        summary = summarize(answer, self.summary_tokens)

        turn = Turn(user=user_text, specialist=chosen, answer=answer,
                    summary=summary, was_tool_result=is_tool_result)
        self.timeline.append(turn)
        return turn


# ------------------------------------------------------------- summarization


def summarize(text: str, budget_tokens: int) -> str:
    """Extractive stand-in for the paper's <=192-token model-written summary.
    A real model summary preserves the salient fact; our extractive version
    keeps the head of the answer PLUS the 'answer: N' tail if present (the
    outcome of an agent turn is its most summary-worthy content)."""
    toks = text.split()
    head = toks[:budget_tokens]
    out = " ".join(head)
    m = None
    import re
    m = re.search(r"answer: (\d+)", text)
    if m and f"answer: {m.group(1)}" not in out:
        out = out + f" ... answer: {m.group(1)}"
    return out


# ------------------------------------------------- prefix-cache-hit checking


def own_view_prefix_stability(proxy: Proxy, specialist: str,
                              new_user: str) -> Tuple[bool, int, int]:
    """Simulate the paper's KV-prefix-reuse property (Layer A): re-entering
    a LoRA must yield a byte-identical prefix to its previous visit.

    We rebuild, for each of the specialist's past turns, the FULL context of
    that request (view + generated answer). Consecutive full contexts must
    nest: the previous request's context is a strict byte-prefix of the next
    one's — exactly the property the engine's prefix cache exploits."""
    views = []
    timeline = proxy.timeline
    for i, t in enumerate(timeline):
        if t.specialist != specialist:
            continue
        sub = Proxy.__new__(Proxy)
        sub.timeline = timeline[:i]
        sub.tok = proxy.tok
        views.append(sub.own_view(specialist, t.user) + t.answer)
    if len(views) < 2:
        return True, 0, 0
    prev, cur = views[-2], views[-1]
    stable = cur.startswith(prev)
    return stable, len(prev), len(cur)
