"""Synthetic GUI-action trajectory data for UI-MOPD (arXiv 2607.04425).

Generates tiny, deterministic *action templates* for two platforms
(desktop / mobile) so the MOPD loop has something concrete to optimise.
Each "prompt" is a task description token sequence; the *target* action
template (verb + coordinate) is the ground-truth the reward compares
against (Eq. 8). Two teachers are pre-trained on their own platform's
templates, mirroring Stage-1 SFT.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from model import (BOS, EOS, SEP, STOI, encode,
                   DESKTOP_VERBS, MOBILE_VERBS)


# ---------------------------------------------------------------------------
# Prompt + target templates per platform
# ---------------------------------------------------------------------------
# Each template pairs an action (verb, coord) with a *unique task keyword*
# that the prompt exposes, so the action is actually predictable from the
# prompt (a learnable toy task). Without this the LM has no signal.
DESKTOP_TEMPLATES = [
    ("left_click",  "c0", "click_task"),
    ("right_click", "c1", "rightclick_task"),
    ("mouse_move",  "c2", "move_task"),
    ("scroll",      "c3", "scroll_task"),
    ("key",         "c4", "keypress_task"),
    ("type",        "w0", "typing_task"),
    ("wait",        None, "wait_task"),
    ("terminate",   None, "end_task"),
]
MOBILE_TEMPLATES = [
    ("click",         "c5", "tap_task"),
    ("long_press",    "c6", "hold_task"),
    ("swipe",         "c7", "swipe_task"),
    ("type",          "w1", "input_task"),
    ("answer",        "w2", "reply_task"),
    ("system_button", "c8", "sysbtn_task"),
    ("wait",          None, "wait_task"),
    ("terminate",     None, "end_task"),
]


@dataclass
class Prompt:
    platform: str            # "desktop" | "mobile"
    prompt_tokens: list[int] # [bos, sep, task...]
    target: list[int]        # ground-truth action ids [verb, coord?]
    prompt_tensor: torch.Tensor


def _make_prompt(platform: str, verb: str, coord: str | None,
                 task_keyword: str) -> Prompt:
    """Build a prompt token list: [bos, sep, task_keyword, sep].

    The task_keyword uniquely identifies the intended action, so the
    sequence [bos, sep, <keyword>, sep] -> [verb, coord] is learnable.
    """
    toks = [STOI[BOS], STOI[SEP]]
    if task_keyword in STOI:
        toks.append(STOI[task_keyword])
    else:
        # fall back: hash the keyword into the vocab deterministically
        toks.append(STOI.get(task_keyword,
                             list(STOI.values())[hash(task_keyword)
                                                  % len(STOI)]))
    toks.append(STOI[SEP])
    target = [STOI[verb]] + ([STOI[coord]] if coord else [])
    return Prompt(
        platform=platform,
        prompt_tokens=toks,
        target=target,
        prompt_tensor=torch.tensor(toks, dtype=torch.long),
    )


def make_dataset(n_per_platform: int = 16, seed: int = 0) -> list[Prompt]:
    """Deterministic synthetic dataset.

    Each platform's templates are cycled, optionally repeated to reach
    `n_per_platform`. Because every (task_keyword -> action) pair is
    unique, the mapping is learnable.
    """
    rng = random.Random(seed)
    prompts: list[Prompt] = []
    for i in range(n_per_platform):
        v, c, kw = DESKTOP_TEMPLATES[i % len(DESKTOP_TEMPLATES)]
        prompts.append(_make_prompt("desktop", v, c, kw))
    for i in range(n_per_platform):
        v, c, kw = MOBILE_TEMPLATES[i % len(MOBILE_TEMPLATES)]
        prompts.append(_make_prompt("mobile", v, c, kw))
    rng.shuffle(prompts)
    return prompts


def split_by_platform(prompts: list[Prompt]):
    d = [p for p in prompts if p.platform == "desktop"]
    m = [p for p in prompts if p.platform == "mobile"]
    return d, m
