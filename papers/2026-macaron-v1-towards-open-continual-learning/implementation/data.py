"""
Macaron-V1 toy re-implementation — data side.

Paper: "Macaron-V1: Towards Open Continual Learning with Self-Improvement
and Mixture-of-LoRA" — Mind Lab, 2026, arXiv:2608.09819.

Four synthetic task families, one per specialist, each with a DIFFERENT
output shape (the paper's motivation: differently-shaped CoT competes for
shared parameters -> keep divergent skills in separate LoRAs):

  L0 chat    : persona-stable smalltalk           -> short warm reply
  L1 agent   : tool-call style arithmetic chains  -> call->result->answer
  L2 coding  : code generation                    -> python function body
  L3 gui     : UI4A-style structured action spec  -> Action(field=...) lines

A character-level tokenizer keeps everything tiny and deterministic.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

# ------------------------------------------------------------------ tokenizer


class CharTokenizer:
    """Character-level tokenizer over a fixed toy alphabet."""

    def __init__(self):
        chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,:;!?()[]{}=+-*/_<>\"'#\n@%\n"
        )
        # dedupe, keep order
        seen = []
        for c in chars:
            if c not in seen:
                seen.append(c)
        self.chars = seen
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def generate(self, model, prompt: str, adapter: "str | None" = None,
                 max_new: int = 40, temperature: float = 0.0) -> str:
        """Greedy (temperature=0) generation, adapter-routed."""
        from model import set_active_adapter, SPECIALISTS
        if adapter is not None:
            set_active_adapter(model, SPECIALISTS.index(adapter))
        ids = self.encode(prompt)
        model.eval()
        stop_id = self.stoi.get("\n", None)
        with torch.no_grad():
            for _ in range(max_new):
                window = ids[-(model.cfg.max_len - 8):]
                x = torch.tensor([window], dtype=torch.long)
                logits = model(x)[0, -1]
                nxt = int(logits.argmax())
                ids.append(nxt)
                # stop on a blank line (end of answer) like real serving
                if len(ids) >= 2 and ids[-1] == stop_id and ids[-2] == stop_id:
                    break
        # cut the prompt: return only the generated continuation
        return self.decode(ids[len(self.encode(prompt)):])


import torch  # noqa: E402  (used inside CharTokenizer.generate)

# ------------------------------------------------------------ task generators

NAMES = ["ava", "ben", "cal", "dex", "eli", "fen", "gus", "hal"]
TOPICS = ["weather", "music", "books", "food", "travel", "sleep", "plants"]
# deterministic topic -> adjective map: the reply must be PREDICTABLE from
# the request (a persona the chat LoRA can actually learn)
TOPIC_ADJ = {
    "weather": "calm",
    "music": "bright",
    "books": "quiet",
    "food": "warm",
    "travel": "fresh",
    "sleep": "cozy",
    "plants": "kind",
}


def gen_chat(rng: random.Random) -> Tuple[str, str]:
    n = rng.choice(NAMES); t = rng.choice(TOPICS); a = TOPIC_ADJ[t]
    user = f"hi im {n} lets talk about {t}"
    reply = f"hello {n}! {t} is {a}. what do you like about {t}?"
    return user, reply


def gen_agent(rng: random.Random) -> Tuple[str, str]:
    a = rng.randint(2, 30); b = rng.randint(2, 30)
    c = rng.randint(2, 5)
    user = f"compute {a} plus {b} then multiply by {c}"
    answer = (a + b) * c
    reply = f"call add({a},{b}) -> {a+b}; call mul({a+b},{c}) -> {answer}; answer: {answer}"
    return user, reply


def gen_coding(rng: random.Random) -> Tuple[str, str]:
    fn = rng.choice(["square", "double", "negate"])
    user = f"write a python function {fn}(x)"
    body = {
        "square": "def square(x):\n    return x * x",
        "double": "def double(x):\n    return 2 * x",
        "negate": "def negate(x):\n    return -x",
    }[fn]
    return user, body


def gen_gui(rng: random.Random) -> Tuple[str, str]:
    origin = rng.choice(["card", "panel", "dialog"])
    fn = rng.choice(["refresh", "submit", "close"])
    user = f"make a {origin} with a button that runs {fn}"
    reply = (
        f"Action(origin={origin}, state={{'id': 1}}, "
        f"execution={fn}(), visibility={{'id': 'NoAI'}})"
    )
    return user, reply


GENERATORS = {
    "L0": gen_chat,
    "L1": gen_agent,
    "L2": gen_coding,
    "L3": gen_gui,
}


def make_dataset(n_per_family: int = 200, seed: int = 7) -> List[Dict]:
    """Mixed workload: list of {family, user, target}."""
    rng = random.Random(seed)
    data = []
    for label, gen in GENERATORS.items():
        for _ in range(n_per_family):
            user, target = gen(rng)
            data.append({"family": label, "user": user, "target": target})
    rng.shuffle(data)
    return data


def framing_for(family: str, history: str, user: str) -> str:
    """Format the training/serving context for a specialist."""
    return f"{history}user: {user}\nassistant: "


# ------------------------------------------------- pretraining & harness cfg

GENERIC_HEADER = "<assistant>"   # baseline harness config (no specialist)


def make_pretrain_corpus(n: int = 1500, seed: int = 11) -> List[Dict]:
    """Generic mixed corpus for base pretraining (all families under the
    generic '<assistant>' header, no specialist alignment). The base is then
    FROZEN — all specialist capability arrives via LoRA, as in the paper."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        label = rng.choice(list(GENERATORS.keys()))
        user, target = GENERATORS[label](rng)
        data.append({"family": label, "user": user, "target": target,
                     "header": GENERIC_HEADER})
    return data


def routing_example(user: str, true_family: str) -> Tuple[str, str]:
    """Router training/serving format: constrained next-char = digit 0-3."""
    idx = ["L0", "L1", "L2", "L3"].index(true_family)
    return f"user: {user}\nroute: L", str(idx)


def make_dependency_conversations(n: int = 30, seed: int = 23):
    """Two-turn conversations where turn 2 depends on the CONTENT of turn 1's
    answer produced by a DIFFERENT specialist — probes whether the <=N-token
    summary handoff is a lossy bottleneck (paper's own open question)."""
    rng = random.Random(seed)
    convs = []
    for _ in range(n):
        a = rng.randint(10, 60); b = rng.randint(2, 20)
        total = a + b
        t1_user = f"compute {a} plus {b} then multiply by 1"
        t1_target = (f"call add({a},{b}) -> {total}; call mul({total},1) -> "
                     f"{total}; answer: {total}")
        # L0 chat turn must echo the number the agent computed
        t2_user = "say the answer you computed and say bye"
        t2_target = f"the answer is {total}. bye!"
        convs.append({
            "turns": [
                {"user": t1_user, "family": "L1", "target": t1_target},
                {"user": t2_user, "family": "L0", "target": t2_target},
            ]
        })
    return convs
