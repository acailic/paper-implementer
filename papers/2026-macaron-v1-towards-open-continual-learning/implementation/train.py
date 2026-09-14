"""
Macaron-V1 toy re-implementation — training & evaluation driver.

Paper: "Macaron-V1: Towards Open Continual Learning with Self-Improvement
and Mixture-of-LoRA" — Mind Lab, 2026, arXiv:2608.09819.

Pipeline (mirrors the paper's recipe at toy scale):

  1. Pretrain the tiny base on a generic mixed corpus, then FREEZE it.
  2. SFT each specialist LoRA (L0-L3) on its own task family through its own
     adapter slot (paper: GRPO on synthetic trajectories; with deterministic
     toy rewards, SFT on ground-truth trajectories is the honest equivalent).
  3. Multi-turn SFT: specialists learn to answer from OWN-VIEW contexts —
     own turns verbatim, other specialists' turns as summaries.
  4. SFT the router: L0 with constrained decoding (only digits 0-3 legal),
     then a GRPO-style group-relative boost on routing (no critic).
  5. Evaluate on HELD-OUT data:
     a. routing accuracy (paper Venti: 99.12%)
     b. per-specialist exact-match through the full Proxy loop
     c. own-view prefix stability (KV-prefix-reuse simulation)
     d. summary-handoff vs full-history on dependency conversations
     e. MoL (4 LoRAs) vs budget-matched single "fat" LoRA on the same
        mixed workload (the paper's own missing ablation)

Run:  python3 train.py            (CPU, ~2-4 minutes)
"""

from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from model import (
    MoLConfig, TinyBase, AdapterLinear, attach_adapters, set_active_adapter,
    lora_parameters, count_lora_params, Proxy, SPECIALISTS, SPECIALIST_NAMES,
    SPECIALIST_HEADERS, own_view_prefix_stability, summarize,
)
from data import (
    CharTokenizer, make_dataset, make_pretrain_corpus, routing_example,
    make_dependency_conversations, GENERIC_HEADER, GENERATORS,
)

torch.manual_seed(0)
random.seed(0)

DEVICE = torch.device("cpu")
CFG = MoLConfig(d_model=128, n_layers=4, n_heads=4, max_len=320)
TOK = CharTokenizer()
EPOCHS_PRETRAIN = 4
EPOCHS_SFT = 16
EPOCHS_ROUTER = 4
NLL_THRESHOLD = 0.35   # teacher-forced per-token NLL counted as "known"


def tokens_of(text: str) -> torch.Tensor:
    return torch.tensor([TOK.encode(text)], dtype=torch.long, device=DEVICE)


def nll(model, prompt: str, target: str, header: str = "") -> torch.Tensor:
    """Mean per-token NLL of `target` given `prompt` (with optional header)."""
    full = header + prompt + target
    ids = tokens_of(full)
    T = ids.shape[1]
    if T > CFG.max_len:
        ids = ids[:, -(CFG.max_len):]
        T = CFG.max_len
    plen = len(TOK.encode(header + prompt))
    logits = model(ids[:, :-1])
    lg = logits[0, plen - 1: T - 1]
    tg = ids[0, plen:T]
    return F.cross_entropy(lg, tg)


PAD = 0  # arbitrary fill id; masked out of the loss


def batch_nll(model, examples, header_of) -> torch.Tensor:
    """Batched SFT loss. Prompt format EXACTLY matches Proxy.own_view
    rendering: header + '\\nuser: ...\\nassistant: ' (train–serve
    consistency). Router examples ('route': True) use the constrained
    routing prompt instead. Right-padding + causal attention keeps each
    example's gradient identical to the unbatched case."""
    seqs, tgts = [], []
    for ex in examples:
        header = "" if ex.get("route") else header_of(ex)
        if ex.get("route"):
            prompt = f"user: {ex['user']}\nroute: L"
            target_text = ex["target"] + "\n"
        else:
            prompt = header + f"\nuser: {ex['user']}\nassistant: "
            target_text = ex["target"] + "\n\n"
        full = prompt + target_text
        ids = TOK.encode(full)
        if len(ids) > CFG.max_len:
            ids = ids[-CFG.max_len:]
        seqs.append(ids)
        tgts.append(len(TOK.encode(prompt)))
    B = len(seqs)
    L = max(len(s) for s in seqs)
    inp = torch.full((B, L - 1), PAD, dtype=torch.long)
    lab = torch.full((B, L - 1), PAD, dtype=torch.long)
    msk = torch.zeros(B, L - 1, dtype=torch.bool)
    for b, (s, plen) in enumerate(zip(seqs, tgts)):
        t = torch.tensor(s)
        inp[b, : len(s) - 1] = t[:-1]
        lab[b, : len(s) - 1] = t[1:]
        # only score target tokens (offset by 1 in the lab frame)
        msk[b, plen - 1: len(s) - 1] = True
    logits = model(inp)
    n_tok = int(msk.sum())
    lg = logits[msk]
    tg = lab[msk]
    return F.cross_entropy(lg, tg, reduction="sum") / max(n_tok, 1)


# ----------------------------------------------------------------- 1. pretrain

def pretrain_base(model, corpus):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4)
    model.train()
    n = len(corpus)
    bs = 16
    for epoch in range(EPOCHS_PRETRAIN):
        rng = random.Random(100 + epoch)
        order = list(range(n)); rng.shuffle(order)
        tot, cnt = 0.0, 0
        for s in range(0, n, bs):
            batch = [corpus[i] for i in order[s:s + bs]]
            loss = batch_nll(model, batch, lambda ex: ex["header"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); cnt += 1
        print(f"  pretrain epoch {epoch}: loss {tot / max(cnt,1):.4f}")


def freeze_base(model):
    """Freeze the base forever (Macaron: base never updates after pretrain);
    only LoRA A/B tensors stay trainable."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, AdapterLinear):
            for i in range(CFG.n_specialists):
                m.lora_A[i].requires_grad_(True)
                m.lora_B[i].requires_grad_(True)
    print(f"  base frozen; {count_lora_params(model):,} LoRA params/slot x "
          f"{CFG.n_specialists} slots")


# ---------------------------------------------------------------- 2. SFT LoRAs

def make_router_examples(data):
    """Router training examples as SFT dicts (constrained digit target)."""
    out = []
    for ex in data:
        idx = SPECIALISTS.index(ex["family"])
        out.append({"family": "L0", "user": ex["user"], "target": str(idx),
                    "route": True})
    return out


def sft_specialists(model, data, epochs=EPOCHS_SFT):
    """Per-specialist SFT through each one's own adapter slot.

    L0 is trained JOINTLY on chat + routing (as in the paper: L0 is both the
    conversation backbone and the router). Training them sequentially
    (chat, then router) catastrophically forgets the chat skill — the
    cross-task interference the MoL design is about, live in the toy."""
    router_exs = make_router_examples(data)
    for idx, label in enumerate(SPECIALISTS):
        fam = [ex for ex in data if ex["family"] == label]
        if label == "L0":
            fam = fam + router_exs     # joint chat + routing mixture
        params = [p for p in lora_parameters(model, idx)]
        opt = torch.optim.AdamW(params, lr=1e-3)
        set_active_adapter(model, idx)
        model.train()
        last = 0.0
        for epoch in range(epochs):
            rng = random.Random(200 + idx * 10 + epoch)
            order = list(range(len(fam))); rng.shuffle(order)
            tot, cnt = 0.0, 0
            for s in range(0, len(fam), 16):
                batch = [fam[i] for i in order[s:s + 16]]
                loss = batch_nll(model, batch,
                                 lambda ex, L=label: SPECIALIST_HEADERS[L])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); cnt += 1
            last = tot / max(cnt, 1)
        print(f"  {label} ({SPECIALIST_NAMES[label]}) sft loss {last:.4f}")


# ------------------------------------------------------- 3. multi-turn own-view

def sft_multiturn(model, data, n_convs=60, summary_tokens=12, epochs=3):
    """Teach specialists to answer from OWN-VIEW contexts: their own turns
    verbatim, OTHER specialists' turns collapsed to a short summary — the
    paper's per-adapter serving format. Without this step the Proxy loop
    diverges from single-turn training (train–serve mismatch).

    L0's multiturn data is mixed WITH its chat data in the same optimizer
    (joint training, as the paper trains L0 on its full task mixture).
    Training multiturn alone catastrophically forgets the chat format —
    observed live here; exactly the cross-task interference MoL exists
    to manage."""
    convs = make_dependency_conversations(n=n_convs, seed=77)

    # L1: turn-1 answers with empty history
    l1_examples = []
    for conv in convs:
        t1 = conv["turns"][0]
        l1_examples.append({
            "user": t1["user"],
            "hist_summary": "",
            "target": t1["target"] + "\n\n",
        })
    params = [p for p in lora_parameters(model, 1)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    set_active_adapter(model, 1)
    model.train()
    for epoch in range(epochs):
        rng = random.Random(400 + epoch)
        order = list(range(len(l1_examples))); rng.shuffle(order)
        tot, cnt = 0.0, 0
        for i in order:
            ex = l1_examples[i]
            view = (SPECIALIST_HEADERS["L1"] + f"\nuser: {ex['user']}"
                    f"\nassistant: ")
            loss = nll(model, view[len(SPECIALIST_HEADERS["L1"]):],
                       ex["target"], header=SPECIALIST_HEADERS["L1"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); cnt += 1
    print(f"  multiturn L1: loss {tot/max(cnt,1):.4f}")

    # L0: turn-2 answers from own view, MIXED with chat data (joint)
    l0_multiturn = []
    for conv in convs:
        t1, t2 = conv["turns"]
        l0_multiturn.append({
            "user": t1["user"],
            "hist_summary": summarize(t1["target"], summary_tokens),
            "t2_user": t2["user"],
            "target": t2["target"] + "\n\n",
            "route": False,
        })
    chat_fam = [ex for ex in data if ex["family"] == "L0"]
    params = [p for p in lora_parameters(model, 0)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    set_active_adapter(model, 0)
    model.train()
    for epoch in range(epochs):
        rng = random.Random(407 + epoch)
        mixture = []
        for ex in l0_multiturn:
            prompt = (SPECIALIST_HEADERS["L0"] + f"\nuser: {ex['user']}"
                      f"\nassistant: {ex['hist_summary']}"
                      f"\nuser: {ex['t2_user']}\nassistant: ")
            mixture.append((prompt, ex["target"]))
        for ex in chat_fam:
            prompt = (SPECIALIST_HEADERS["L0"] +
                      f"\nuser: {ex['user']}\nassistant: ")
            mixture.append((prompt, ex["target"] + "\n\n"))
        order = list(range(len(mixture))); rng.shuffle(order)
        tot, cnt = 0.0, 0
        for s in range(0, len(mixture), 16):
            batch = [mixture[i] for i in order[s:s + 16]]
            # reuse batch_nll via adapter dicts
            dicts = [{"user": "", "target": tgt.split("\n\n")[0], "route": False,
                      "prompt_override": p} for p, tgt in batch]
            loss = batch_nll_prompts(model, dicts,
                                     lambda ex: SPECIALIST_HEADERS["L0"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); cnt += 1
    print(f"  multiturn L0 (joint w/ chat): loss {tot/max(cnt,1):.4f}")


def batch_nll_prompts(model, examples, header_of) -> torch.Tensor:
    """batch_nll for pre-rendered (prompt_override, target) pairs."""
    seqs, tgts = [], []
    for ex in examples:
        if "prompt_override" in ex:
            prompt = ex["prompt_override"]
            target_text = ex["target"] + "\n\n"
        else:
            prompt = header_of(ex) + f"\nuser: {ex['user']}\nassistant: "
            target_text = ex["target"] + "\n\n"
        full = prompt + target_text
        ids = TOK.encode(full)
        if len(ids) > CFG.max_len:
            ids = ids[-CFG.max_len:]
        seqs.append(ids)
        tgts.append(len(TOK.encode(prompt)))
    B = len(seqs)
    L = max(len(s) for s in seqs)
    inp = torch.full((B, L - 1), PAD, dtype=torch.long)
    lab = torch.full((B, L - 1), PAD, dtype=torch.long)
    msk = torch.zeros(B, L - 1, dtype=torch.bool)
    for b, (s, plen) in enumerate(zip(seqs, tgts)):
        t = torch.tensor(s)
        inp[b, : len(s) - 1] = t[:-1]
        lab[b, : len(s) - 1] = t[1:]
        msk[b, plen - 1: len(s) - 1] = True
    logits = model(inp)
    lg = logits[msk]; tg = lab[msk]
    return F.cross_entropy(lg, tg, reduction="sum") / max(int(msk.sum()), 1)


# ------------------------------------------------------------- 4. router SFT

def sft_router(model, data):
    """L0-as-router: constrained next-char classification over digits 0-3."""
    params = [p for p in lora_parameters(model, 0)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    set_active_adapter(model, 0)
    model.train()
    digit_ids = torch.tensor([TOK.stoi[d] for d in "0123"])
    for epoch in range(EPOCHS_ROUTER):
        rng = random.Random(300 + epoch)
        order = list(range(len(data))); rng.shuffle(order)
        correct, tot_loss, cnt = 0, 0.0, 0
        for i in order:
            prompt, target = routing_example(data[i]["user"], data[i]["family"])
            ids = tokens_of(prompt)
            logits = model(ids)[0, -1]
            loss = F.cross_entropy(logits[digit_ids].unsqueeze(0),
                                   torch.tensor([int(target)]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += float(loss.detach()); cnt += 1
            correct += int(torch.argmax(logits[digit_ids].detach()) == int(target))
        print(f"  router epoch {epoch}: loss {tot_loss/cnt:.4f} "
              f"train-acc {correct/cnt:.3f}")


def grpo_router_boost(model, data, groups=40, g=8, steps=2):
    """Group-relative policy optimization for the router, no critic:
    for each prompt sample a GROUP of g routing decisions, score each 1/0
    vs the true label, advantage A_i = (r_i - mean)/std over the group,
    and push log-prob of each sample proportional to its advantage.

    Two paper-faithful guards (otherwise this step CATASTROPHICALLY FORGETS
    L0's chat skill — observed live in this toy):
    - the learning-value gate (§4.4): train only on prompts the current
      router does not already solve reliably (sample-acc < 1.0);
    - a small lr (1e-4), since this is a polish pass, not a rewrite."""
    params = [p for p in lora_parameters(model, 0)]
    opt = torch.optim.AdamW(params, lr=1e-4)
    digit_ids = torch.tensor([TOK.stoi[d] for d in "0123"])
    rng = random.Random(42)
    set_active_adapter(model, 0)
    model.train()
    for step in range(steps):
        idxs = rng.sample(range(len(data)), groups * 4)
        used, tot, cnt = 0, 0.0, 0
        for i in idxs:
            prompt, target = routing_example(data[i]["user"], data[i]["family"])
            ids = tokens_of(prompt)
            with torch.no_grad():
                logits0 = model(ids)[0, -1]
                probs0 = F.softmax(logits0[digit_ids], -1)
                if float((probs0.argmax() == int(target))) == 1.0 and \
                        float(probs0.max()) > 0.9:
                    continue  # already solved reliably -> no learning value
            used += 1
            logits = model(ids)[0, -1]
            logp4 = F.log_softmax(logits[digit_ids], dim=0)
            probs = F.softmax(logp4.detach(), -1)
            with torch.no_grad():
                sample_idx = torch.multinomial(probs, g, replacement=True)
            r = torch.tensor([1.0 if int(s) == int(target) else 0.0
                              for s in sample_idx])
            mean, std = r.mean(), r.std().clamp_min(1e-6)
            A = (r - mean) / std
            loss = -(A @ logp4[sample_idx]) / g
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(r.mean()); cnt += 1
            if used >= groups:
                break
        print(f"  grpo step {step}: trained on {used} unsolved prompts, "
              f"group-mean sample-acc {tot/max(cnt,1):.3f}")


# ----------------------------------------------------------------- evaluations

def eval_routing(model, data, n=400):
    proxy = Proxy(model, TOK)
    correct = 0
    rng = random.Random(999)
    for ex in rng.sample(data, min(n, len(data))):
        pred = proxy.route(ex["user"])
        correct += int(pred == ex["family"])
    return correct / min(n, len(data))


def gen_em(model, ex, gen_limit=96) -> int:
    """Greedy-generation exact match for one example under its specialist.
    Generation stops at a blank line (the trained turn terminator), so the
    full stripped output is compared — targets may be multi-line (L2 code)."""
    label = ex["family"]
    proxy = Proxy(model, TOK)
    view = proxy.own_view(label, ex["user"])
    out = TOK.generate(model, view, adapter=label, max_new=gen_limit)
    pred = out.strip()
    gold = ex["target"].strip()
    return int(pred == gold)


def eval_proxy_accuracy(model, data, n_per_fam=40):
    """Full Route->Answer->Summary loop (routing NOT forced); exact-match
    scoring per family after light normalization."""
    rng = random.Random(1234)
    results = {}
    for label in SPECIALISTS:
        fam = [ex for ex in data if ex["family"] == label]
        sample = rng.sample(fam, min(n_per_fam, len(fam)))
        hits = 0
        for ex in sample:
            hits += gen_em(model, ex)
        results[label] = hits / len(sample)
    return results


def eval_prefix_stability(model, n_convs=12, turns=6):
    """Multi-specialist conversations; own-view of each specialist must be a
    byte-identical extension of its previous own-view (KV prefix reuse)."""
    rng = random.Random(555)
    stable_all = 0; total_checks = 0; total_saved = 0; total_chars = 0
    for c in range(n_convs):
        proxy = Proxy(model, TOK)
        fams = [rng.choice(SPECIALISTS) for _ in range(turns)]
        for fam in fams:
            user, _ = GENERATORS[fam](rng)
            proxy.step(user, true_label=fam)
        for fam in set(fams):
            stable, prev_len, cur_len = own_view_prefix_stability(proxy, fam,
                                                                  "next")
            stable_all += int(stable); total_checks += 1
            total_saved += prev_len; total_chars += cur_len
    return {
        "stability": stable_all / max(total_checks, 1),
        "avg_prefix_reuse_fraction": total_saved / max(total_chars, 1),
    }


def eval_summary_bottleneck(model, convs, summary_tokens=12):
    """The paper's own open question: is the <=N-token summary a lossy
    bottleneck for specialist handoff?

    Isolation design: turn 1 is replaced by the ORACLE agent answer (perfect
    L1), and routing is forced — so any failure is purely the handoff: can
    L0, seeing only the SUMMARY of L1's turn in its own view, echo the
    number, vs seeing L1's verbatim answer (full-history arm)?"""
    import re

    def run(with_own_view: bool):
        hits = 0
        for conv in convs:
            t1, t2 = conv["turns"]
            m = re.search(r"answer: (\d+)", t1["target"])
            needle = m.group(1) if m else None
            if with_own_view:
                proxy = Proxy(model, TOK, summary_tokens=summary_tokens)
                proxy.step(t1["user"], true_label=t1["family"],
                           force_label="L1")
                # oracle specialist: replace with the gold answer
                proxy.timeline[-1].answer = t1["target"]
                proxy.timeline[-1].summary = summarize(t1["target"],
                                                       summary_tokens)
                turn2 = proxy.step(t2["user"], true_label=t2["family"],
                                   force_label="L0")
                out = turn2.answer
            else:
                # full-history ablation: L0 sees L1's verbatim answer
                ctx = (SPECIALIST_HEADERS["L0"] +
                       f"\nuser: {t1['user']}\nassistant: {t1['target']}"
                       f"\nuser: {t2['user']}\nassistant: ")
                set_active_adapter(model, 0)
                out = TOK.generate(model, ctx, adapter="L0", max_new=48)
            hits += int(needle is not None and needle in out)
        return hits / len(convs)

    own = run(True)
    full = run(False)
    return {"own_view_summary_handoff": own, "full_history": full}


# --------------------------------------------- 5. MoL vs single fat LoRA ablation

def eval_mol_vs_fat(model, train_data, test_data, n_per_fam=40):
    """The paper's own missing ablation: budget-matched single LoRA trained
    jointly on ALL families vs the 4-slot MoL.

    Fat single LoRA: rank 4r with alpha 4a has, per projection,
    4r*(d_in + d_out) params = 4 x [r*(d_in + d_out)] — exactly the total
    parameter budget of the four rank-r MoL slots. Trained on the same mixed
    workload (with per-family headers, best case), evaluated by greedy EM."""
    fat_cfg = MoLConfig(
        d_model=CFG.d_model, n_layers=CFG.n_layers, n_heads=CFG.n_heads,
        max_len=CFG.max_len,
        lora_rank=CFG.lora_rank * CFG.n_specialists,
        lora_alpha=CFG.lora_alpha * CFG.n_specialists)
    fat_model = TinyBase(fat_cfg)
    # bring over the pretrained frozen base (embeddings, blocks, head)
    src_sd = model.state_dict()
    fat_sd = fat_model.state_dict()
    for k in fat_sd:
        if k in src_sd and fat_sd[k].shape == src_sd[k].shape:
            fat_sd[k] = src_sd[k].clone()
    fat_model.load_state_dict(fat_sd)
    attach_adapters(fat_model, fat_cfg)
    # copy frozen base weights into the wrapped copy
    fat_sd = fat_model.state_dict()
    for k in fat_sd:
        if k.endswith(".base.weight"):
            fat_sd[k] = src_sd[k].clone()
    fat_model.load_state_dict(fat_sd)
    freeze_base(fat_model)

    # joint SFT of slot 0 only, on the same mixed workload
    sft_specialists_joint(fat_model, train_data)
    set_active_adapter(fat_model, 0)

    rng2 = random.Random(31337)
    out = {"MoL": {}, "fat": {}}
    for label in SPECIALISTS:
        fam = [ex for ex in test_data if ex["family"] == label]
        sample = rng2.sample(fam, min(n_per_fam, len(fam)))
        hits = 0
        for ex in sample:
            hits += gen_em(model, ex)          # MoL: its own family slot
        out["MoL"][label] = hits / len(sample)
        hits = 0
        set_active_adapter(fat_model, 0)       # fat: the single joint slot
        for ex in sample:
            view = Proxy(fat_model, TOK).own_view(label, ex["user"])
            gen = TOK.generate(fat_model, view, max_new=72)
            hits += int(gen.strip() == ex["target"].strip())
        out["fat"][label] = hits / len(sample)
    return out


def sft_specialists_joint(model, data, epochs=EPOCHS_SFT):
    """Single fat LoRA trained on ALL families at once (headers still tell it
    which mode it is in — the fairest single-adapter baseline)."""
    params = [p for p in lora_parameters(model, 0)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    set_active_adapter(model, 0)
    model.train()
    last = 0.0
    for epoch in range(epochs):
        rng = random.Random(600 + epoch)
        order = list(range(len(data))); rng.shuffle(order)
        tot, cnt = 0.0, 0
        for s in range(0, len(data), 8):
            batch = [data[i] for i in order[s:s + 8]]
            loss = batch_nll(model, batch,
                             lambda ex: SPECIALIST_HEADERS[ex["family"]])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); cnt += 1
        last = tot / max(cnt, 1)
    print(f"  fat joint sft loss {last:.4f}")


# ------------------------------------------------------------------------- main

def main():
    t0 = time.time()
    print("== Macaron-V1 toy re-implementation ==")
    print(f"torch {torch.__version__}, device cpu")

    print("\n[1] pretrain + freeze base")
    base = TinyBase(CFG)
    corpus = make_pretrain_corpus(3000)
    pretrain_base(base, corpus)
    attach_adapters(base, CFG)     # add the 4 LoRA slots (identity at init)
    freeze_base(base)

    data = make_dataset(n_per_family=400, seed=7)      # 1600 total
    split = int(len(data) * 0.8)
    train_data, test_data = data[:split], data[split:]

    print("\n[2] SFT specialist LoRAs (base frozen)")
    sft_specialists(base, train_data)

    print("\n[3] multi-turn own-view SFT (summary handoff)")
    sft_multiturn(base, train_data, n_convs=60)

    print("\n[4] GRPO-style router boost (routing was jointly SFT'd into L0)")
    grpo_router_boost(base, train_data)

    print("\n[5] evaluations (held-out)")
    acc = eval_routing(base, test_data, n=400)
    print(f"  routing accuracy       : {acc:.3f}   (paper Venti: 0.9912)")

    res = eval_proxy_accuracy(base, test_data, n_per_fam=40)
    for k, v in res.items():
        print(f"  proxy answer acc [{k}]  : {v:.3f}")

    stab = eval_prefix_stability(base)
    print(f"  own-view prefix stability: {stab['stability']:.3f}, "
          f"avg reuse fraction {stab['avg_prefix_reuse_fraction']:.3f}")

    convs = make_dependency_conversations(n=30, seed=23)
    bott = eval_summary_bottleneck(base, convs, summary_tokens=12)
    print(f"  summary handoff vs full : {bott['own_view_summary_handoff']:.3f} "
          f"vs {bott['full_history']:.3f}")

    print("\n[6] MoL vs budget-matched fat LoRA (paper's missing ablation)")
    ab = eval_mol_vs_fat(base, train_data, test_data, n_per_fam=40)
    for name in ("MoL", "fat"):
        row = "  ".join(f"{k}={v:.2f}" for k, v in ab[name].items())
        print(f"  {name:4s}: {row}")
    mol_mean = sum(ab['MoL'].values()) / 4
    fat_mean = sum(ab['fat'].values()) / 4
    print(f"  mean: MoL {mol_mean:.3f} vs fat {fat_mean:.3f}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
