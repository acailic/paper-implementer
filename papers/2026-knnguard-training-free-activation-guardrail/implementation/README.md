# kNNGuard: Training-Free Activation Guardrail

## What this implements

A **toy** demonstration of the kNNGuard paper (Abdelfattah, Nasiri, Garraghan 2026,
arXiv:2607.02072). The full method uses a frozen 8B LLM as feature extractor with
50+50 labeled prompts. This implementation compresses the core algorithm into a minimal,
runnable setup:

- **Frozen 4-layer MLP** as the "LLM backbone" (extracts multi-layer activations)
- **100-sample bank** (50 safe + 50 unsafe synthetic embeddings with 20% overlap)
- **Fisher-discriminant layer weighting** — data-driven layer selection
- **Cosine kNN risk scoring** — unsafe fraction among k=13 nearest neighbors
- **Adaptive fused ensemble** — blends activation-kNN and embedding-kNN via confidence gap

### Key kNNGuard ideas demonstrated

| Concept | How it appears here |
|---------|-------------------|
| Training-free | Frozen MLP, no gradient updates during guardrail use |
| Multi-layer activation bank | Last-token hiddens from 4 layers cached |
| Fisher discriminant weighting | J_l = B_l/W_l selects layers that best separate classes |
| Cosine kNN risk score | Unsafe fraction among k=13 neighbors in activation space |
| Adaptive fusion (FE) | Confidence-gap rule: winner-takes-all if gap > γ, else blend |
| Domain adaptation | Swap the 50-example bank, no weight changes |

## Files

- `model.py` — ToyLLM backbone, bank building, Fisher weighting, kNN, fusion, evaluation
- `data.py` — Synthetic safe/unsafe prompt embeddings with configurable overlap
- `train.py` — Full pipeline: bank build → weight → compare 4 methods → paper claims
- `requirements.txt` — torch, numpy, scikit-learn

## How to run

```bash
pip install -r requirements.txt
python train.py
```

The script:
1. Builds a frozen 4-layer MLP backbone (~instant)
2. Generates 100 bank + 400 test synthetic prompts
3. Compares 4 methods: Embedding-kNN, Fine-tuned CLS, kNNGuard LE, kNNGuard FE
4. Prints comparison table + paper claim verification

## Expected output

```
Method                     F1     Prec  Recall    FPR    FNR
------------------------------------------------------------
Embedding-kNN             0.xxx  0.xxx   0.xxx   0.xxx  0.xxx
Fine-tuned CLS             0.xxx  0.xxx   0.xxx   0.xxx  0.xxx
kNNGuard LE               0.xxx  0.xxx   0.xxx   0.xxx  0.xxx
kNNGuard FE               0.xxx  0.xxx   0.xxx   0.xxx  0.xxx
```

Key observations matching paper:
- Activation-kNN beats embedding-kNN (activations encode richer safety distinctions)
- Fused ensemble matches or improves on activation-only
- No training required — only a 100-sample labeled bank
- Fisher weighting selects the most discriminative layers automatically

## Hardware

Works on **CPU**. Everything is small synthetic data. No GPU needed.
~5-10 seconds total runtime.
