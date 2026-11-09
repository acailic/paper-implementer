---
name: mingpt-operations
description: A PyTorch re-implementation of [GPT](https://github.com/openai/gpt-2), both training and inference. minGPT tries to be small, clean, interpretable and educational, as most of the currently available GPT model implementations can a bit sprawling. GPT is not a complicated model and this implementation
source: https://github.com/karpathy/minGPT
commit: 37baab7
distiller: deterministic-rule-based (no LLM) — see implementation/README.md
---

# mingpt-operations — operational skill

> What this is: distilled operational knowledge for **minGPT** @ `37baab7`.
> Distilled automatically from the repository evidence; every claim below
> is grounded in the files listed in `references/`. Layout follows the
> Agent Skills convention (SKILL.md + references/ + scripts/).

## When to use this skill

- **Install / environment setup** — You need to install the package or set up its environment before anything else.
- **Instantiate and configure the core model** — You need to create/configure the repository's main model component.
- **Run / customize training** — You need to train or fine-tune, or adapt the training loop.
- **Prepare / preprocess data** — You need to build or preprocess the dataset the repo expects.
- **Run inference / sampling / evaluation** — You need to generate outputs, sample, or evaluate a trained model.
- **Load pretrained / provided checkpoints** — You need to initialize from published weights or a checkpoint.
- **Use the command-line entry points** — You need to drive the repo from the shell rather than Python.
- **Understand the repository layout** — You need to know which file owns which responsibility.

## Requirements

```bash
# install observed third-party dependencies:
pip install numpy regex requests setuptools torch transformers
```

## Steps

### Step 1. Install / environment setup

You need to install the package or set up its environment before anything else.

- `git clone https://github.com/karpathy/minGPT.git`  *(evidence: README.md:22)*
- `pip install -e .`  *(evidence: README.md:24)*

### Step 2. Understand the repository layout

You need to know which file owns which responsibility.

- `AdditionDataset` — Creates n-digit addition problems. For example, if n=2, then an example  *(evidence: projects/adder/adder.py:43)*
- `CausalSelfAttention` — A vanilla multi-head masked self-attention layer with a projection at the end.  *(evidence: mingpt/model.py:29)*
- `NewGELU` — Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).  *(evidence: mingpt/model.py:21)*
- `CfgNode` — a lightweight configuration class inspired by yacs  *(evidence: mingpt/utils.py:31)*
- `BPETokenizer` — PyTorch-aware class that wraps the Encoder above  *(evidence: mingpt/bpe.py:257)*
- `Block` — an unassuming Transformer block  *(evidence: mingpt/model.py:73)*

### Step 3. Prepare / preprocess data

You need to build or preprocess the dataset the repo expects.

- `AdditionDataset` — Creates n-digit addition problems. For example, if n=2, then an example  *(evidence: projects/adder/adder.py:43)*
- `BPETokenizer` — PyTorch-aware class that wraps the Encoder above  *(evidence: mingpt/bpe.py:257)*
- `CharDataset` — Emits batches of characters  *(evidence: projects/chargpt/chargpt.py:42)*
- `Encoder` — (no docstring)  *(evidence: mingpt/bpe.py:62)*
- `Encoder.bpe(self, token)` — this function uses self.bpe_ranks to iteratively merge all the possible bpe tokens  *(evidence: mingpt/bpe.py:95)*
- `Encoder.encode_and_show_work(self, text)` — debugging function, same as encode but returns all intermediate work  *(evidence: mingpt/bpe.py:180)*

### Step 4. Instantiate and configure the core model

You need to create/configure the repository's main model component.

- `CfgNode` — a lightweight configuration class inspired by yacs  *(evidence: mingpt/utils.py:31)*
- `GPT.configure_optimizers(self, train_config)` — This long function is unfortunately doing something very simple and is being very defensive:  *(evidence: mingpt/model.py:215)*
- `CfgNode.merge_from_args(self, args)` — update the configuration from a list of strings that is expected  *(evidence: mingpt/utils.py:62)*
- `CfgNode.to_dict(self)` — return a dict representation of the config  *(evidence: mingpt/utils.py:55)*
- `CausalSelfAttention.__init__(self, config)` — (no docstring)  *(evidence: mingpt/model.py:36)*
- `Block.__init__(self, config)` — (no docstring)  *(evidence: mingpt/model.py:76)*

### Step 5. Load pretrained / provided checkpoints

You need to initialize from published weights or a checkpoint.

- `CfgNode` — a lightweight configuration class inspired by yacs  *(evidence: mingpt/utils.py:31)*
- `GPT.configure_optimizers(self, train_config)` — This long function is unfortunately doing something very simple and is being very defensive:  *(evidence: mingpt/model.py:215)*
- `GPT.from_pretrained(cls, model_type)` — Initialize a pretrained GPT model by copying over the weights  *(evidence: mingpt/model.py:175)*

### Step 6. Run / customize training

You need to train or fine-tune, or adapt the training loop.

- `Trainer` — (no docstring)  *(evidence: mingpt/trainer.py:13)*
- `GPT.configure_optimizers(self, train_config)` — This long function is unfortunately doing something very simple and is being very defensive:  *(evidence: mingpt/model.py:215)*
- `CfgNode.merge_from_args(self, args)` — update the configuration from a list of strings that is expected  *(evidence: mingpt/utils.py:62)*
- `GPT.from_pretrained(cls, model_type)` — Initialize a pretrained GPT model by copying over the weights  *(evidence: mingpt/model.py:175)*
- `Trainer.get_default_config()` — (no docstring)  *(evidence: mingpt/trainer.py:16)*
- `Trainer.__init__(self, config, model, train_dataset)` — (no docstring)  *(evidence: mingpt/trainer.py:31)*

### Step 7. Run inference / sampling / evaluation

You need to generate outputs, sample, or evaluate a trained model.

- `GPT.generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None)` — Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete  *(evidence: mingpt/model.py:283)*
- `purpose`  *(evidence: README.md:1)*

### Step 8. Use the command-line entry points

You need to drive the repo from the shell rather than Python.

- `runnable_module`  *(evidence: mingpt/bpe.py:—)*
- `runnable_module`  *(evidence: projects/adder/adder.py:—)*
- `runnable_module`  *(evidence: projects/chargpt/chargpt.py:—)*
- `runnable_module`  *(evidence: tests/test_huggingface_import.py:—)*


## Pitfalls

- **note (Jan 2023)**: though I may continue to accept and change some details, minGPT is in a semi-archived state. For more recent developments see my rewrite [nanoGPT](https://github.com/karpathy/nano  *(from README.md)*
- Dependency surface: the code imports numpy, regex, requests, setuptools, torch, transformers — version mismatches are the first thing to check when imports fail.

## Evidence & provenance

- Source: https://github.com/karpathy/minGPT @ commit `37baab7`
- Evidence files: 14 admitted by the include/exclude boundary
- Full API surface with file:line provenance: `references/api.md`
- README-derived quickstart snippets: `references/quickstart.md`
- Static sanity probe for a local checkout: `scripts/smoke_check.py`
- Construction record (verification results): `record.json` next to this graph
