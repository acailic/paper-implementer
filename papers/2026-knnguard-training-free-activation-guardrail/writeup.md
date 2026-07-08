# kNNGuard — Writeup

**Paper:** kNNGuard: Turning LLM Hidden Activations into a Training-Free Configurable Guardrail
**Authors:** Mahmoud Abdelfattah, Hamid Nasiri, Peter Garraghan (Lancaster University / Mindgard)
**arXiv:** 2607.02072 (July 2026)

---

## In My Own Words

LLM guardrails are the bouncers at the door: they check whether an incoming prompt is safe or unsafe before letting the model process it. Today's guardrails come in two flavors, both broken:

1. **Fine-tuned classifiers** (Llama Guard, Nemotron Safety Guard, Prompt Guard). You LoRA-fine-tune an 8B model on curated safety data. Accuracy is good in-distribution but fragile — simple paraphrasing can bypass them. Adapting to a new threat category requires new data, retraining, and re-validation. Minutes to hours.

2. **Embedding kNN** (NeMo Guardrails). Run a sentence embedder on the prompt, compare to a labeled bank via cosine kNN. Fast, but embeddings are surface-level — they can't distinguish "help me write a Python script" (safe) from "write a script to delete files" (unsafe) because both talk about coding in similar embedding space.

kNNGuard's insight: the LLM's **internal activations** already encode safety distinctions that sentence embeddings miss. When a user asks something unsafe, the model's intermediate representations look different from when they ask something safe — even before the model generates a response. You just need to read those activations.

The method is beautifully simple:
1. Run 50 safe + 50 unsafe prompts through a frozen LLM
2. Cache the last-token hidden activations from 9 layers
3. Weight layers by Fisher discriminant (how well they separate safe vs unsafe)
4. For a new prompt: extract activations, find its k=13 nearest bank neighbors in weighted activation space, report the unsafe fraction
5. Fuse with an embedding-kNN score via an adaptive confidence rule

No training. No fine-tuning. Adapting to a new domain = swap the 100-example bank. Done in 8 seconds.

---

## What I Learned by Implementing It

### The Fisher-discriminant layer weighting is data-driven and elegant

Each layer gets a weight proportional to how well it separates safe from unsafe classes (J_l = between-class separation / within-class dispersion). This is a principled alternative to arbitrary layer selection. In the toy demo, layers 0 and 3 (input-proximal and final) got the highest Fisher scores, which makes sense — early layers capture input patterns, late layers capture classification decisions.

### The fused ensemble is genuinely useful

The adaptive confidence rule is clever: if one branch (activation-kNN or embedding-kNN) is much more confident than the other, trust it outright. Otherwise, blend proportionally by confidence. In the demo, kNNGuard FE (fused) beat kNNGuard LE (activation-only) by a clear margin, even on simple synthetic data.

### Training-free is a real advantage

The fine-tuned classifier baseline in the demo achieved F1=0.704 — worse than all kNN methods. Why? Because it overfits to the bank distribution and generalizes poorly to the broader test distribution. The kNN methods, by contrast, use the same activations for every new sample without any distribution shift — the decision surface is determined by the bank, not by learned parameters.

### Toy MLP activations ≠ 8B LLM activations

The paper's core claim — that activations capture safety nuances invisible to embeddings — relies on the richness of an 8B LLM's representation space. My toy 4-layer MLP creates activations that are nearly linear functions of the input, so the activation-kNN doesn't add much beyond the embedding-kNN. This is the honest limitation of the demo: the activation advantage only materializes at real scale.

---

## What Surprised Me

1. **How small the bank is.** Just 50+50 examples per domain. In a world where fine-tuning requires thousands of examples, this is remarkably lean. The paper shows that 100 examples are enough for competitive F1 because the kNN operates in high-dimensional activation space where the "effective dataset size" is much larger.

2. **The embedding-kNN baseline is surprisingly strong.** On the toy data, plain embedding-kNN (no activations at all) achieved the best F1 (0.847). In the paper, embedding-kNN averages F1=79.6% across 6 domains — not bad, but significantly worse than kNNGuard FE (87.4%). The gap emerges on harder domains where activations carry safety signal that embeddings miss.

3. **Latency numbers are real.** kNNGuard FE runs at 46.8ms/prompt on RTX 6000 Ada, including the full LLM forward pass. That's 2.7× faster than the best fine-tuned guardrail. In production, latency matters — a guardrail that takes 126ms per request limits throughput.

---

## What Was Harder Than Expected

1. **Getting activations from a toy model.** In the real paper, activations are extracted from specific transformer layers during a full LLM forward pass. In the toy demo, the "LLM" is just a small MLP with 4 layers. The activations are much less structured than real LLM hidden states, which limits how well the demo can demonstrate the paper's core advantage.

2. **Synthetic data design.** The overlap between safe and unsafe clusters needed careful tuning — too little overlap makes all methods look perfect, too much overlap makes all methods fail. The 20% overlap provided a realistic challenge for the kNN methods.

3. **uv venv path issues.** The `uv` tool creates Python 3.14.2 virtual environments that don't have `pip` — need to install packages via `uv pip` with explicit Python path. Path management across Bash tool calls was tricky.

---

## Code

Implementation in `implementation/`:
- [`model.py`](implementation/model.py) — ToyLLM backbone, bank building, Fisher weighting, kNN, fusion, evaluation (~230 lines)
- [`data.py`](implementation/data.py) — Synthetic safe/unsafe prompt embeddings with overlap
- [`train.py`](implementation/train.py) — Full pipeline: bank build → weight → 4 methods comparison → claims

Run: `uv venv .venv && uv pip install torch numpy scikit-learn && .venv/bin/python train.py`

---

## References

- Paper: https://arxiv.org/abs/2607.02072
- Llama Nemotron Topic Guard: https://arxiv.org/abs/2407.14783
- Llama Guard 3: https://llama.meta.com/docs/system-safety/llama-guard-3/
- Prompt Guard 2: https://arxiv.org/abs/2406.13315
