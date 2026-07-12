# Program-as-Weights: A Programming Paradigm for Fuzzy Functions

**arXiv:** [2607.02512v1](https://arxiv.org/abs/2607.02512) · **Venue:** Preprint (cs.LG) · **Date:** 2 Jul 2026
**Authors:** Wentao Zhang, Liliana Hotsko, Woojeong Kim, Pengyu Nie, Stuart Shieber, Yuntian Deng
**Affiliations:** University of Waterloo · Cornell University · Harvard University (* first three equal contribution)
**Code:** https://github.com/programasweights · **Demo:** https://programasweights.com
**Sourcing:** all numeric tables transcribed verbatim from `paper_layout.txt` (`pdftotext -layout`); line ranges cited per table. Figure-derived numbers are flagged; no per-point figure back-fill.

---

## TL;DR

Many everyday "fuzzy" programming tasks (alert on important log lines, repair malformed JSON, rank search results by intent) resist clean rule-based implementation and are today outsourced to LLM APIs at the cost of **locality, reproducibility, and price**. This paper proposes **fuzzy-function programming**: compile a fuzzy function *once* from a natural-language specification into a small, locally-executable neural artifact, then run a frozen lightweight interpreter on every subsequent call — offline, cheap, reproducible.

Instantiated as **Program-as-Weights (PAW)**: a 4B compiler (trained on **FuzzyBench-10M**, released) emits a PEFT adapter; a frozen **0.6B Qwen3 interpreter** executes it.

**Headline:** PAW (Qwen3-0.6B) hits **73.78% exact-match on FuzzyBench**, beating direct prompting of **Qwen3-32B (68.70%)** at **~50× less inference memory** (~1.2 GB bf16 vs ~60 GB), running at **~30 tok/s on a MacBook M3** from a ~430 MB GGUF base + 23 MB per-program LoRA.

> Source: abstract + §1 (line 88: "73.78% vs. 68.70%") + §6 main result. All three reconcile to Table 2.

---

## 1. Core Paradigm

A fuzzy function `f: X → Y` is one whose behavior is more naturally specified by natural language / examples / constraints than by symbolic code. PAW replaces repeated per-input LLM calls with a **compile-once / run-locally** split mirroring classical programming:

```
p = Compiler(s),    ŷ = Interpreter(p, x) ≈ f(x)        (Eq. 1)
```

The "executable" `p` is a learned parameter blob; the "runtime" is a frozen neural network. Introducing a new fuzzy function only requires compiling a new `p` — the interpreter is never retrained.

**Hybrid program.** `p` is a tuple of a **discrete** and a **continuous** component (Eq. 2):

```
p = ( p_discrete , p_continuous )
```

- `p_discrete` — a variable-length token sequence ("pseudo-program"): a clean paraphrase of the spec + a few I/O examples, fed to the interpreter as part of its input.
- `p_continuous` — a PEFT module (LoRA in the current system; prefix-tuning KV cache in the precursor) that re-tunes the frozen interpreter for this one task.

The discrete half **shields the interpreter from typos/ambiguity** in the original spec; the continuous half supplies fine-grained behavioral control text alone cannot.

A compiled PAW program is a **single file** (~23 MB at Q4_0 for a 0.6B interpreter + one-time shared base) that can be saved, version-controlled, distributed via package managers, and called from Python/JavaScript with a two-line API — "objects of the same kind as Python modules."

---

## 2. The Compiler–Interpreter System

Three components, none tied to a specific PEFT:

| Component | Role | Instantiation |
|---|---|---|
| **Pseudo compiler** `Cp` | Read spec `s` → discrete pseudo-program `p_discrete` | Off-the-shelf **Qwen3-4B-Instruct-2507** (never trained); small task-rewriting prompt (App. C) |
| **PEFT compiler** `CPEFT` | Read `[s \| p_discrete]` → PEFT module `p_continuous` from hidden states | Trained 4B Qwen3 (LoRA compiler `CL` or prefix compiler `CP`) |
| **Interpreter** | Frozen LM; ingest `p_continuous` + prepend `p_discrete` to input `x`; generate `ŷ` | Frozen **Qwen3-0.6B** (default) |

### 2.1 Text-to-LoRA (current best, §3.2)

`CL` (4B Qwen3, trained) runs one forward pass on `[s | p_discrete | EOS | τ₁…τT]` where `τ₁:T` is a fixed sequence of **T = 64 learned prefix tokens**. It extracts prefix-position hidden states from **L compiler layers spaced uniformly by depth ratio** (one per interpreter layer), stacked into `H ∈ R^{L×T×d_teacher}`.

**LoRA mapper.** H is mean-pooled over both L depth-aligned layers and T prefix positions:

```
h̄ = (1/LT) Σ_{l,t} H_{l,t}
```

passed through a shallow MLP trunk `ϕ`, projected into mixing coefficients `α^{A,B}_{l,m,n}` per layer `l`, module type `m`, basis index `n`. The per-(layer,module) LoRA is a shared-basis mixture (Eq. 3):

```
A^ex_{l,m} = Σ_n α^A_{l,m,n} · A^(m)_n ,   B^ex_{l,m} = Σ_n α^B_{l,m,n} · B^(m)_n
```

- Shared learnable bases `A^{(m)}_{1:N} ∈ R^{N×r×d_in}`, `B^{(m)}_{1:N} ∈ R^{N×d_out×r}` per module type `m` (attention q/k/v/o + MLP gate/up/down = 7 types).
- **rank r = 64, N = 64 shared bases per module type**, applied to all layers + module types.
- Per fuzzy function, this injects **~38.5M LoRA parameters** into the interpreter (line 207).

Interpreter execution: (i) attach the LoRA to target modules, (ii) prepend `p_discrete` to `x`, (iii) generate autoregressively. Because the interpreter is frozen and the LoRA hot-swappable, **one device-resident interpreter serves unboundedly many PAW programs** ("one runtime, many programs," Figure 19).

### 2.2 Prefix-tuning precursor (§3.3)

The precursor swaps the LoRA mapper for a prefix-tuning mapper: a small linear mapper `ψ` projects the same `H` position-wise into KV pairs `(K^ex_{l,t}, V^ex_{l,t}) ∈ R^{2×d_int}` prepended to the interpreter's attention KV cache at every layer (standard prefix-tuning; architecture in Figure 18).

### 2.3 PEFT-instantiation comparison (Table 1, lines 240–245)

At controlled comparison scale (same training compute):

| Method | Accuracy |
|---|---|
| Prompting | 0.098 |
| Prefix Tuning | 0.504 |
| Text-to-LoRA, r=18 | 0.565 |
| **Text-to-LoRA, r=64 (default)** | **0.657** |

> r=18 matches the prefix-tuning program size. Both PEFTs beat the no-compiler prompting baseline (9.8%). **LoRA is the stronger PEFT** and is the instantiation scaled to the full training data. (§3.3, lines 234–238.)

---

## 3. Training (§4)

Only the PEFT compiler is trained; pseudo compiler `Cp` and interpreter are both frozen. With both endpoints frozen, training reduces to a single supervised objective — the negative mean-token log-likelihood of the target `y` under the frozen interpreter (Eq. 4):

```
L(θ) = E_{(s,x,y)} [ −log P_ϕ( y | p_discrete, p_LoRA(θ; s, p_discrete), x ) ]
```

Gradient flows back through the frozen interpreter into the LoRA mapper and from there into `CL`'s hidden states. **No policy-gradient term, no group baseline** (line 1183).

**Configuration (Appendix G, lines 1168–1187):**

- **Pseudo compiler `Cp`** (untrained): Qwen/Qwen3-4B-Instruct-2507, examples template (App. C). Pseudo-programs for the entire 10M training set pre-generated once with vLLM, indexed by spec, stored in JSONL; read from disk during training (never sampled live).
- **LoRA compiler `CL`** (trained): Qwen3-4B-Instruct-2507, fully unfrozen, lr **2×10⁻⁵**, bf16, gradient checkpointing on. Input = minimal spec wrapper + pseudo from `Cp` + fixed T=64 learned prefix tokens.
- **LoRA mapper**: fp32 for numerical stability, mean-pool + single residual MLP trunk + shared bases. r=64, N=64 bases/module-type, target modules `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (7 modules).
- **Interpreter**: frozen, default Qwen/Qwen3-0.6B.
- **Training loop**: **3 epochs** over 10M examples; **batch 16, grad-accum 3 (effective batch 48)**; max CL seq len **1280**, max interpreter seq len **1024**.
- **Optimizer**: AdamW default PyTorch settings; **no warmup, no LR schedule**.
- **Hardware**: single B300 (early) or 8×H200 (later). The 0.6B run completed 3 epochs in **~72 hours on 3 GPUs**.

---

## 4. FuzzyBench-10M (§5)

A 10M-example dataset where every example is a triple `(s, x, y)` = (specification, input, target output), generated with **gpt-5.2**.

**Construction (two-stage):**
1. Prompt gpt-5.2 to generate NL specifications of fuzzy functions (**8 specs per call**, repeated under different category constraints for breadth).
2. For each spec, prompt gpt-5.2 again to generate **8 I/O pairs**.

**Split:** 80/10/10 by spec into train/val/test → **test specifications are entirely unseen at training time**. Verified test set constructed where an independent strong model (**gpt-5-mini**) and gpt-5.2 agree on the output, removing ambiguous targets.

**Thematic coverage:** built incrementally over **29 versions**, each adding 100K–500K examples covering a new family. Figure 3 groups the 10M into 7 high-level families:

| Family | Examples | Share |
|---|---|---|
| Core text processing & NLP | 2.95M | 30% |
| Search, matching & web intelligence | 1.80M | 18% |
| Custom classification & filtering | 1.50M | 15% |
| Code & natural-language commands | 1.25M | 12% |
| Safety, verification & domain knowledge | 1.25M | 12% |
| Agentic & tool use | 0.75M | 8% |
| Format repair & validation | 0.50M | 5% |

> "Core text processing" is largest because the v1 base layer (2.5M examples; 277 base categories) covers parsing, classification, NER, coref, sentiment; the remaining 7.5M spread across the other six families. Final dataset covers **800+ sub-categories**. (§5, lines 298–305.)

**Noise variants** (robustness eval): test set perturbed along **8 axes** — typos, grammar errors, ambiguity, formatting drift, "all noise" (combined), terse phrasing, casual phrasing, paraphrase — each at **3 intensities** (light/medium/heavy). (§5, lines 307–310; §8.)

**Empirical ceiling** (line 312): the data-generating model itself bounds reachable accuracy — **gpt-5.2: 96.09%**, **gpt-5-mini: 91.87%**. (Reconciles to Table 2 FuzzyBench column, rows 1–2.)

### FuzzyBench-10M version timeline (Table 10, lines 1198–1241)

| v | Size | Specs added | Categories added (theme) |
|---|---|---|---|
| 1 | 2.50M | 81,920 | Core text processing (parsing, classification, NER, coref, sentiment, etc.; 277 base categories) |
| 2 | 2.70M | +24,576 | New categories + freeform (data filtering, diff parsing, cron, resume parsing, …) |
| 3 | 3.00M | +32,768 | Fuzzy/soft matching (fuzzy search, approximate string match, phonetic, entity resolution) |
| 4 | 3.25M | +32,768 | Format repair (JSON/XML/CSV/YAML/SQL repair, schema conformance, encoding repair) |
| 5 | 3.50M | +32,768 | Natural-language commands (NL→shell/jq/awk/git/curl/find, command flag inference) |
| 6 | 3.75M | +32,768 | Agentic/tool use (tool call generation, tool selection, schema generation) |
| 7 | 4.00M | +32,768 | Custom classification/filtering (criteria-based, log filtering, salience, anomaly) |
| 8 | 4.25M | +32,768 | File/code semantic classification (filepath classification, code purpose, commit message) |
| 9 | 4.50M | +32,768 | DSL/code cleanup (NL-to-DSL, comment stripping, dialogue extraction, build error interp.) |
| 10 | 4.75M | +32,768 | Log monitoring (importance detection, streaming anomaly, alert evaluation) |
| 11 | 5.00M | +32,768 | NL constraint validation (constraint formalization, password policy, schema synthesis) |
| 12 | 5.25M | +32,768 | Constraint satisfaction (boolean SAT, type checking, dependency order, API contracts) |
| 13 | 5.50M | +32,768 | Operation safety/secrets (command risk classification, secret detection, redaction) |
| 14 | 5.75M | +32,768 | Reversibility/output masking/token reduction (traceback distillation, output summarization) |
| 15 | 6.00M | +32,768 | Auto-completion (context-aware, history-based, cwd-aware, domain-specific) |
| 16 | 6.25M | +32,768 | Pseudo-program execution (execution trace, result prediction, NL-to-Python) |
| 17 | 6.50M | +32,768 | Chemistry properties + domain commonsense (SMILES, reaction extraction, service-time est.) |
| 18 | 7.00M | +65,536 | Domain-knowledge plugins + counterfactual reasoning (30 categories spanning STEM/health/social) |
| 19 | 7.25M | +32,768 | Browser semantic matching + transcript cleanup + narrow translation |
| 20 | 7.50M | +32,768 | Agent watchdog / wait interruption (process state classification, completion detection) |
| 21 | 7.75M | +32,768 | AI text detection / authorship / style analysis |
| 22 | 8.00M | +32,768 | Semantic search + explicit AI detection (relevance, query reformulation, snippet extraction) |
| 23 | 8.25M | +32,768 | spaCy superset / custom NLP pipeline (custom NER, span labeling, dependency parsing) |
| 24 | 8.50M | +32,768 | HTML understanding / browser intelligence (ad detection, boilerplate removal) |
| 25 | 8.75M | +32,768 | Intent-based HTML + semantic equivalence / deduplication |
| 26 | 9.00M | +32,768 | Agent tool store / streaming intelligence |
| 27 | 9.25M | +32,768 | Intent-to-navigation / settings discovery |
| 28 | 9.75M | +32,768 | Document-grounded QA / spec-based classification |
| 29 | 10.0M | +32,768 | Smart search pipeline completion (keyword extraction, term weighting, reranking) |

> Each version adds 2,000 new validation + 2,000 new test specifications (caption, line 1199). v18 adds 65,536 specs (double the usual 32,768).

---

## 5. Main Results (Table 2, lines 344–367)

Three baseline families, all on the same test sets as PAW: (i) direct prompting of open-weight LMs + two API models bounding the ceiling; (ii) symbolic code generation (ALCHEmist LM→Code); (iii) same-0.6B-base adaptation (full FT, fixed LoRAs). **FuzzyBench = exact-match Acc on the verified test set; SMS = F1 (WRENCH setup); YouTube/Yelp/IMDB = Acc.** `Contained` = self-contained / offline-executable. `PS` = per-program shipping size (prompt/spec size for prompting baselines; deployed Q4_0 PEFT adapter for PAW on Qwen3-0.6B + 0.8B, fp32 for GPT-2).

| Method | Contained | Interp. Size | FuzzyBench Acc (PS) | YouTube Acc (PS) | SMS F1 (PS) | Yelp Acc (PS) | IMDB Acc (PS) |
|---|---|---|---|---|---|---|---|
| gpt-5.2 (API) | ✗ | – | 96.09% (0.73 KB) | 95.20% (0.95 KB) | 97.06% (1.03 KB) | 98.55% (1.69 KB) | 95.60% (2.25 KB) |
| gpt-5-mini (API) | ✗ | – | 91.87% (0.73 KB) | 93.60% (0.95 KB) | 91.03% (1.03 KB) | 98.13% (1.69 KB) | 94.96% (2.25 KB) |
| Local LM (Qwen3 0.6B) | ✓ | 0.6B | 9.84% (0.73 KB) | 52.80% (0.95 KB) | 0.00%\* (1.03 KB) | 89.55% (1.69 KB) | 66.88% (2.25 KB) |
| Local LM (Qwen3 4B) | ✓ | 4B | 49.63% (0.73 KB) | 90.80% (0.95 KB) | 92.54% (1.03 KB) | 97.53% (1.69 KB) | 93.76% (2.25 KB) |
| Local LM (Qwen3 8B) | ✓ | 8B | 52.15% (0.73 KB) | 94.40% (0.95 KB) | 91.55% (1.03 KB) | 97.95% (1.69 KB) | 94.52% (2.25 KB) |
| Local LM (Qwen3 14B) | ✓ | 14B | 63.96% (0.73 KB) | 93.20% (0.95 KB) | 92.75% (1.03 KB) | 97.74% (1.69 KB) | 92.64% (2.25 KB) |
| Local LM (Qwen3 32B) | ✓ | 32B | 68.70% (0.73 KB) | 93.60% (0.95 KB) | 89.04% (1.03 KB) | 98.11% (1.69 KB) | 94.64% (2.25 KB) |
| Local LM (OLMo3 7B) | ✓ | 7B | 39.84% (0.73 KB) | 90.00% (0.95 KB) | 90.14% (1.03 KB) | 97.66% (1.69 KB) | 93.28% (2.25 KB) |
| Local LM (gpt-oss-20B) | ✓ | 20B | 85.45% (0.73 KB) | 91.60% (0.95 KB) | 89.05% (1.03 KB) | 97.42% (1.69 KB) | 92.08% (2.25 KB) |
| LM→Code [Huang et al., 2024b] | ✓ | 29 MB | – | 89.10%† (–) | 90.00%† (–) | 57.50%† (–) | 66.20%† (–) |
| LM→Code (Reimplementation) | ✓ | 29 MB | 35.81% (0.08 KB) | 70.46% (0.09 KB) | 86.41% (0.06 KB) | 50.35% (0.05 KB) | 73.92% (0.08 KB) |
| **PAW (Qwen3 0.6B)** | ✓ | 0.6B | **73.78%** (23 MB) | 90.40% (23 MB) | 80.77% (23 MB) | 95.82% (23 MB) | 90.64% (23 MB) |
| PAW (Qwen3.5 0.8B) | ✓ | 0.8B | 67.29% (16 MB) | 88.40% (16 MB) | 84.55% (16 MB) | 94.05% (16 MB) | 82.68% (16 MB) |
| PAW (GPT-2 124M) | ✓ | 124M | 54.39% (38 MB) | 93.60% (38 MB) | 77.50% (38 MB) | 93.16% (38 MB) | 82.12% (38 MB) |

> **†** numbers from Huang et al. [2024b] using 10-sample majority voting; the Reimplementation row uses single-sample for fairness. **\*** zero F1 due to zero recall (Qwen3 0.6B scores 0.00 on SMS).

**Readings (all prose-confirmed, §6 lines 326–340):**
- **PAW (0.6B) beats Qwen3-32B prompting on FuzzyBench: 73.78% vs 68.70%** at ~50× less memory (~1.2 GB bf16 vs ~60 GB).
- **gpt-oss-20B is the strongest local baseline (85.45%)** — it is an open-weight frontier model, not a pure-base comparison; PAW-0.6B (73.78%) trails it but at ~33× fewer parameters and fully offline.
- **Cross-interpreter:** among GPT-2 124M / Qwen3-0.6B / Qwen3-0.8B, **0.6B is strongest** (73.78 > 67.29 > 54.39). GPT-2 124M (1/5 the params, no instruction tuning) still reaches **54%** — the compiler LoRA encodes usable task adaptations even into a very small, weakly-capable base.
- PAW wins FuzzyBench and Yelp; trails API models everywhere (expected — they bound the ceiling).

---

## 6. Image-Conditioned Fuzzy Functions (Table 3, lines 370–379)

The compiler–interpreter abstraction extends to **image-conditioned** tasks **without changing the interpreter**: swap the text Qwen3-4B-Instruct compiler for **Qwen3-VL-4B**, keep the same Qwen3-0.6B interpreter, reuse the same LoRA mapper. Image conditioning is fully encoded in the PEFT module — the small text interpreter never sees pixels.

| Method | Interp. Size | Circuit | Chemical | Music | Im2SMILES | Im2LaTeX | TextVQA |
|---|---|---|---|---|---|---|---|
| AndesVL 0.6B | 0.6B | 0.183 | 0.214 | 0.448 | 0.000 | 0.435 | 0.718 |
| Qwen3-VL 2B-Instruct | 2B | 0.186 | 0.258 | 0.470 | 0.016 | 0.408 | 0.836 |
| Qwen3-VL 4B-Instruct | 4B | 0.196 | 0.221 | 0.450 | 0.044 | 0.399 | 0.822 |
| PAW prefix-tuning (Qwen3 0.6B) | 0.6B | 0.241 | 0.365 | 0.525 | 0.175 | 0.391 | 0.612 |
| PAW LoRA (Qwen3 0.6B) | 0.6B | **0.274** | **0.414** | **0.552** | 0.203 | 0.181 | 0.721 |
| PAW LoRA (Qwen3.5 0.8B) | 0.8B | **0.284** | **0.438** | **0.573** | **0.285** | 0.204 | 0.755 |

> Six tasks: three CoSyn-400K diagram-understanding (Chemical, Circuit, Music), structured-output Im2LaTeX-100K + Im2SMILES-20K, open-ended TextVQA.

**Readings (§6 lines 387–392):** PAW-LoRA beats all VLM baselines (≤4B) on the three CoSyn diagram tasks — Circuit **0.274 vs 0.196** best baseline (Qwen3-VL-4B); Chemical **0.414 vs 0.258** (Qwen3-VL-2B); Music **0.552 vs 0.470** (Qwen3-VL-2B) — at ~0.6B interpreter size. On long-form **Im2LaTeX**, PAW-LoRA (0.181) is *weaker* than its prefix-tuning precursor (0.391): the long I/O examples in the pseudo-program crowd the small interpreter's context budget on long-form tasks (decomposition in Table 9).

### Image-task component decomposition (Table 9, lines 1091–1098)

| Variant | Circuit | Chemical | Music | Im2SMILES | Im2LaTeX | TextVQA |
|---|---|---|---|---|---|---|
| Discrete pseudo only (REINFORCE) | 0.009 | 0.004 | 0.007 | 0.041 | 0.267 | 0.025 |
| Continuous KV-cache only (no pseudo) | 0.181 | 0.364 | 0.493 | 0.234 | 0.471 | 0.439 |
| PAW prefix-tuning (both) | 0.241 | 0.365 | 0.525 | 0.175 | 0.391 | 0.612 |

> Cross-task pattern: short-phrase outputs (Circuit/Chemical/Music, TextVQA) → discrete pseudo-program is a strong inductive bias, adds **5–40 EM points** over continuous-only. Long structured sequences (Im2SMILES, Im2LaTeX) → pseudo-program's I/O examples crowd the context budget; removing it returns **6–8 EM points**. Suggests future image-to-markup PEFT instantiations should drop or lighten the pseudo-program. (§D.1, lines 1102–1109.)

---

## 7. Ablations (§7 + Appendix H)

### 7.1 Architectural variants of the LoRA mapper (Table 4, lines 410–418)

| Mapper variant | Accuracy |
|---|---|
| **Default (r=64, N=64, shared bases)** | **0.6223** |
| Per-position aggregation | 0.5598 |
| Per-position + per-layer bases | 0.5559 |
| Per-layer bases (only) | 0.6028 |
| LoRA + prefix-tuning (both pathways) | 0.6033 |

> Every "more expressive" variant underperformed the simple default (mean-pool + single residual MLP + shared bases). The authors report no clean theoretical explanation. (§7, lines 394–400.)

### 7.2 Compiler vs. no-compiler (Table 5, lines 410–418)

| Method (0.6B base) | Accuracy |
|---|---|
| Fixed LoRA r=18 | 0.4236 |
| Fixed LoRA r=64 | 0.5210 |
| Fixed LoRA r=128 | 0.5159 |
| Full fine-tuning | 0.5840 |
| **PAW (Qwen3 0.6B)** | **0.7378** |

> Same data, same base, same training budget; only the compiler is removed. PAW exceeds **full fine-tuning by 15.4pp** (0.7378−0.5840 = 0.1538) and the **strongest fixed LoRA by 21.7pp** (0.7378−0.5210 = 0.2168). Gain comes specifically from *compiler-generated* LoRA. (§7, lines 402–406.)

### 7.3 Additional ablations (Table 11, lines 1262–1280)

EM on test_clean (Qwen3 0.6B interpreter unless stated). Default in bold.

| Ablation axis | Variant | EM (test_clean) |
|---|---|---|
| Continuous component | KV-prefix (epoch 2) | 0.5044 |
| | LoRA r=18 (epoch 2) | 0.5652 |
| | LoRA r=64, 7 modules (epoch 2) | 0.6572 |
| Compiler input for LoRA | Spec only | 0.6411 |
| | Pseudo only | 0.6165 |
| | **Spec + pseudo (default)** | **0.6443** |
| Discrete-and-LoRA coupling | Shared (one head) | 0.6350 |
| | **Separate (default)** | **0.6443** |
| LoRA-mapper input norm | With LayerNorm | 0.6377 |
| | **Without (default)** | **0.6443** |
| Interpreter initialization | Start from finetuned interpreter | 0.6038 |
| | **Start from base (default)** | **0.6223** |

> ⚠ **"Default" varies across ablation tables — see §13 consistency note.** Table 4 default = 0.6223; Table 11's "default" differs by axis (0.6572 continuous-component epoch-2; 0.6443 compiler-input/coupling/norm; 0.6223 interpreter-init). These were run at different checkpoints / on test_clean (not the verified set), so the "default" baseline value is not a single fixed number — cite per-table, do not average.

### 7.4 Compiler scaling — INCONCLUSIVE (Table 12, lines 1283–1292)

EM on test_clean (Qwen3.5 0.8B interpreter, 0.6M training examples, epoch 1). Reported as exploratory data only.

| Compiler | Frozen? | test_clean |
|---|---|---|
| Qwen3 4B | No | 0.6455 |
| Qwen3 4B | Yes | 0.6128 |
| Qwen3 4B + input norm | Yes | 0.6228 |
| Qwen3 14B | No | 0.6257 |
| Qwen3 32B | Yes | 0.6174 |
| gpt-oss-20B | Yes | 0.5823 |
| Qwen3.5 4B (hybrid) | Yes | 0.5046 |

> Labelled **inconclusive**: pattern is non-monotonic — unfreezing 4B beats frozen 32B; frozen gpt-oss-20B underperforms frozen Qwen3-4B-Instruct-2507. No controlled large-scale study run (each combination is expensive). (App. I, lines 1245–1252.)

---

## 8. Robustness to Noisy Specifications (§8)

### 8.1 Robustness to noise (Table 6, lines 421–425)

PAW Qwen3 0.6B (epoch 2), test_clean EM. 8-axis variants modify the spec but leave the input unchanged.

| | clean | typos | grammar | ambiguity | formatting | all-noise | terse | paraphrase |
|---|---|---|---|---|---|---|---|---|
| EM | 0.6692 | 0.6621 | 0.6731 | 0.6511 | 0.6526 | 0.6326 | 0.6499 | 0.6614 |
| drop from clean | – | −0.7% | +0.4% | −1.8% | −1.7% | −3.7% | −1.9% | −0.8% |

> PAW degrades only slightly even under combined heavy noise (worst: all-noise −3.7%).

### 8.2 Per-noise-type × intensity (Table 13, lines 1294–1305)

Qwen3 0.6B interpreter, epoch 2. EM on test_clean.

| Noise axis | Light | Medium | Heavy |
|---|---|---|---|
| Typos | 0.6709 | 0.6685 | 0.6621 |
| Grammar | 0.6687 | 0.6672 | 0.6731 |
| Ambiguity | 0.6731 | 0.6628 | 0.6511 |
| Formatting | 0.6702 | 0.6575 | 0.6526 |
| All noise (combined) | 0.6670 | 0.6650 | 0.6326 |
| Terse (heavy only) | – | – | 0.6499 |
| Casual (heavy only) | – | – | 0.6675 |
| Paraphrase (heavy only) | – | – | 0.6614 |

### 8.3 Pseudo-program denoises the spec (Table 7, lines 428–432)

Hypothesis: the 4B compiler denoises by converting the noisy spec into a clean pseudo-program before the small interpreter sees it. Tested by bypassing the pseudo-program (raw spec → interpreter).

| Interpreter input | Accuracy (clean) | Accuracy (heavy typos) |
|---|---|---|
| Pseudo-program (default) | 0.6443 | 0.6108 |
| Raw spec | 0.6285 | 0.5662 |

> Clean: pseudo is only **1.6pp** better (0.6443−0.6285 = 0.0158). Heavy-typo: gap widens to **4.5pp** (0.6108−0.5662 = 0.0446). **Confirms the denoising hypothesis** — the pseudo-compiler shields the interpreter from corrupted specs. (§8, lines 441–449.)

---

## 9. Local Execution & Quantization (§9 + Appendix K)

A PAW program is a single file: `paw.compile(prompt)` → serializable program; `paw.function(id)` → Python callable. After first download, all execution is local, no API calls (Listings 1–2, Figure 4).

### 9.1 Quantization on Qwen3 0.6B (Table 8, lines 482–491)

4096-example test_clean subset.

| Configuration | Base size | Adapter size | Accuracy |
|---|---|---|---|
| PyTorch bf16 (no quantization) | 1515 MB | – | 0.6580 |
| fp16 base + fp32 LoRA | 1509 MB | 162 MB | 0.6594 |
| Q8_0 base + Q4_0 LoRA | 805 MB | 23 MB | 0.6567 |
| Q6_K base + Q4_0 LoRA | 623 MB | 23 MB | 0.6575 |
| Q5_K_M base + Q4_0 LoRA | 551 MB | 23 MB | 0.6477 |
| Q4_K_M base + Q4_0 LoRA | 484 MB | 23 MB | 0.6453 |
| IQ4_XS base + Q4_0 LoRA | 430 MB | 23 MB | 0.6462 |

> **Q6_K base + Q4_0 LoRA is indistinguishable from bf16** (0.6575 vs 0.6580). **Q4_K_M + Q4_0 loses only 1.3pp** (0.6580−0.6453 = 0.0127) and cuts total disk to ~507 MB. (§9, lines 494–496.)

**Latency (MacBook M3, Metal):** Q5_K_M base + Q4_0 adapter runs at **31.6 tok/s** with a **0.48 s cold load** (line 498–499).

### 9.2 Full Qwen3 0.6B quantization sweep (Table 14, lines 1323–1340)

4096-example test_clean. fp32 LoRA adapter unless stated.

| Base format | bpw | Base size | Adapter size | EM (test_clean) |
|---|---|---|---|---|
| PyTorch bf16 | 16 | 1515 MB | – | 0.6580 |
| fp16 | 16 | 1509 MB | 162 MB | 0.6594 |
| Q8_0 | 8.5 | 805 MB | 162 MB | 0.6550 |
| Q6_K | 6.56 | 623 MB | 162 MB | 0.6558 |
| Q5_K_M | 5.5 | 551 MB | 162 MB | 0.6499 |
| Q4_K_M | 4.8 | 484 MB | 162 MB | 0.6460 |
| IQ4_XS | 4.25 | 430 MB | 162 MB | 0.6484 |
| Q4_K_S | 4.5 | 449 MB | 162 MB | 0.6348 |
| Q3_K_L | 3.9 | 416 MB | 162 MB | 0.6055 |
| Q3_K_M | 3.5 | 395 MB | 162 MB | 0.5874 |
| *Q4_0 adapter (23 MB) instead of fp32 (162 MB):* | | | | |
| Q6_K + Q4_0 | 6.56 | 623 MB | 23 MB | 0.6575 |
| Q5_K_M + Q4_0 | 5.5 | 551 MB | 23 MB | 0.6477 |
| Q4_K_M + Q4_0 | 4.8 | 484 MB | 23 MB | 0.6453 |
| IQ4_XS + Q4_0 | 4.25 | 430 MB | 23 MB | 0.6462 |

### 9.3 GPT-2 124M quantization sweep (Table 15, lines 1345–1357)

36-example handcrafted set, fp32 38 MB LoRA adapter. (Smaller benchmark because GPT-2's accuracy ceiling makes 4096-example differences harder to isolate.)

| Base format | bpw | Base size | tok/s | EM (36) |
|---|---|---|---|---|
| fp16 | 16 | 252 MB | 100.7 | 21/36 |
| Q8_0 | 8.5 | 137 MB | 115.7 | 23/36 |
| Q6_K | 6.5 | 107 MB | 111.0 | 22/36 |
| Q5_K_M | 5.5 | 99 MB | 108.0 | 24/36 |
| Q4_K_M | 4.8 | 91 MB | 110.5 | 24/36 |
| IQ4_NL | 4.5 | 85 MB | 107.1 | 24/36 |
| IQ4_XS | 4.25 | 82 MB | 115.4 | 24/36 |
| Q3_K_L | 3.9 | 88 MB | 116.6 | 23/36 |
| IQ2_M | 2.7 | 63 MB | 192.5 | 16/36 |

### 9.4 Qwen3.5 0.8B quantization sweep (Table 16, lines 1362–1372)

36-example handcrafted set, Q4_0 16 MB LoRA adapter. **Q4_K_S and below crash** with `llama_decode failed (code -3)` — the Mamba-hybrid architecture is incompatible with aggressive quantization.

| Base format | bpw | Base size | tok/s | EM (36) |
|---|---|---|---|---|
| fp16 | 16 | 1517 MB | 6.1 | 30/36 |
| Q8_0 | 8.5 | 774 MB | 6.4 | 30/36 |
| Q6_K | 6.5 | 601 MB | 6.7 | 30/36 |
| Q5_K_M | 5.5 | 551 MB | 6.5 | 30/36 |
| Q4_K_M | 4.8 | 505 MB | 6.5 | 31/36 |
| Q4_K_S | 4.5 | 505 MB | – | crash |
| Q3_K_L and below | – | – | – | crash |

> ⚠ **Paper-internal tension worth flagging:** the abstract + §11 conclusion quote **"~30 tok/s" / "30 tok/s on a MacBook M3"**, while §9 (line 499) gives the precise **31.6 tok/s** for the Q5_K_M+Q4_0 Qwen3-0.6B config. Consistent as rounding (31.6 → ~30), but the precise figure is the Qwen3-0.6B number; the abstract's "~30 tok/s" is the same config rounded. The GPT-2 (100–192 tok/s) and Qwen3.5-0.8B (6.1–6.7 tok/s) sweeps show the rate is highly interpreter-dependent — the headline "~30 tok/s" applies specifically to the 0.6B path.

---

## 10. Case Studies (§9, App. M)

Five applications, each a fuzzy task resisting symbolic implementation but not needing a 30B-param API call per input:

1. **Event-driven log monitoring** — local classifier fires only on log lines that matter (replaces naive `wait`-based terminal watching in Cursor). Spec: `Classify log lines. Return ONLY one word: ALERT or QUIET.` (App. M.1)
2. **Intent-based site navigation** — NL quick-find for a website, no per-request LLM API call.
3. **Semantic search reranking** — intent-aware fuzzy search layered on a keyword index, no LLM in the request path.
4. **Agent preprocessing / tool calling** — a **10-PAW-function pipeline scores 93% on TOOLCALL-15**, capturing tool-routing behavior usually reserved for much larger models.
5. **Creative generation (Alien-Taboo)** — multilingual word-guessing game; each player turn served by a 0.6B PAW interpreter, one PAW program per language; LLM invoked only at compile time → economical to host.

> Figure 19 sketches the multi-program library: each spec compiled once into its own program; all served by a single device-resident interpreter with the appropriate LoRA hot-attached per call.

---

## 11. Related Work — positioning (§10)

PAW sits at the intersection of four lines, with three concrete differentiators:

- **Hypernetworks** (Ha 2017 → Hypter, HINT, HyperTuning, Text-to-LoRA, Generative Adapter, HyperSteer, Gist, MEND; recent: SHINE, HypeLoRA, Doc-to-LoRA, Latent Context Compilation). **PAW differs** by (a) emitting a *hybrid* (discrete pseudo-program + continuous PEFT) program vs continuous-only adapters; (b) training on programmer-style fuzzy-function specs (FuzzyBench-10M's 800+ families) vs QA contexts / distilled per-task adapters; (c) targeting a developer-facing API where the compiled program is a versioned, distributable software artifact.
- **PEFT** (Adapters, prefix-tuning, prompt tuning, LoRA, AdaLoRA, DoRA, QLoRA). PAW generates the PEFT per-example from a textual spec via a separate compiler, rather than learning it per-task by GD on the target task. (T-Few learns PEFT per-task; PAW generates from a description.)
- **Synthetic instruction data** (Self-Instruct, Unnatural Instructions, Textbooks-All-You-Need, Magpie). FuzzyBench-10M differs: task-class-specific 29 thematic versions + a verified test split where two strong LLMs must agree.
- **Model distillation / neural programs** (ALCHEmist → Python code; Binder → SQL/Python w/ embedded API calls; Distilling Step-by-Step). PAW compiles into *weights* not textual code, implementing fuzzy functions that resist symbolic encoding.

---

## 12. Strengths, Limitations, Verdict

**Strengths**
- **Genuinely novel framing** — reframes the foundation model from per-input problem solver into a per-function tool builder; the compiled artifact is a versioned, distributable software object (Python-module analog). The compile-once/run-locally split is clean and the analogy to classical compilers (source → executable → runtime) is well-motivated.
- **Real efficiency win** — 0.6B interpreter beats 32B prompting on FuzzyBench at ~50× less memory; quantized to ~507 MB total and ~30 tok/s on a laptop. The 54% reached by GPT-2-124M (no instruction tuning) shows the compiler LoRA injects usable task adaptations into very weak bases.
- **Modality generality without interpreter change** — swapping only the compiler (text→VL) reuses the same frozen 0.6B text interpreter for image-conditioned tasks, beating ≤4B VLMs on CoSyn diagram understanding.
- **Released dataset + code + demo** — FuzzyBench-10M (10M triples, 800+ categories, 29 versions, 8-axis noise perturbations) is a concrete contribution enabling future PAW-style work.

**Limitations**
- **Empirical ceiling is a gpt-5.2-generated ceiling.** FuzzyBench labels come from gpt-5.2 (96.09%) with a gpt-5-mini agreement filter (91.87%) — the "verified test set" verifies against another LLM, not ground truth. Tasks where the two LLMs systematically agree on a wrong answer are not caught.
- **Long-form structured generation regresses** (Im2LaTeX 0.181 < prefix-tuning 0.391) — the pseudo-program's I/O examples crowd the small interpreter's context budget. The hybrid discrete+continuous design, the paper's headline contribution, is actively harmful in this regime.
- **Compiler scaling is inconclusive** (Table 12) — non-monotonic, under-explored at scale; the choice of 4B compiler is not justified by a clean scaling law.
- **"Default" is not a single number** across ablation tables (see §13 ⚠) — citing a default baseline without its table/checkpoint context is misleading.
- **gpt-oss-20B (85.45%) dominates PAW-0.6B (73.78%) on FuzzyBench** — PAW's headline "beats Qwen3-32B" is true but the strongest local baseline is a different (open-weight frontier) model; the clean win is specifically *at matched small scale + offline + reproducible*.
- **Training cost non-trivial** — ~72 GPU-hours on 3 GPUs for the 0.6B run (plus pseudo-program pre-generation over 10M with vLLM).

**Verdict.** A conceptually clean, well-engineered instantiation of "small models as runtimes, large models as compilers." The strongest citable contributions are (i) the paradigm itself (compile fuzzy functions into distributable weight-artifacts), (ii) FuzzyBench-10M, and (iii) the demonstration that a compiler-generated LoRA + discrete pseudo-program lets a 0.6B frozen interpreter match a 32B prompt. The weakest points are the LLM-as-ceiling evaluation, the long-form regression, and the inconclusive compiler scaling. A credible, deployable step toward the small-model future it advocates.

---

## 13. Internal-Consistency Notes (⚠ flags)

These are paper-internal tensions a breakdown must surface (not silently reconcile), per the established pattern:

1. **⚠ "Default" baseline varies across ablation tables.** Table 4 default = **0.6223**; Table 11's "default" = **0.6572** (continuous-component, epoch 2) / **0.6443** (compiler-input, coupling, input-norm axes) / **0.6223** (interpreter-init axis). These differ because (a) ablations are run on **test_clean**, not the verified test set (Table 2's 73.78% is verified-test-set EM, ~10pp higher), and (b) different ablation axes were measured at different checkpoints. *Diagnostic:* never cite a "default" PAW number without its table + test-set context. The Table 2 / Table 5 PAW value (0.7378) is the verified-set headline; ablation-table defaults (0.62–0.66) are test_clean sub-scale.

2. **⚠ "~30 tok/s" (abstract + §11) vs "31.6 tok/s" (§9).** Same config (Q5_K_M+Q4_0, Qwen3-0.6B, M3); 31.6 rounded → ~30. Consistent as rounding, but the precise figure is interpreter-specific — the GPT-2 path runs 100–192 tok/s and the Qwen3.5-0.8B Mamba-hybrid only 6.1–6.7 tok/s.

3. **Verified-set vs test_clean gap.** Table 2 reports PAW-0.6B FuzzyBench = 73.78% (verified test set, exact match); Tables 4/6/7/8/11/13/14 report test_clean EM in the 0.62–0.66 range. These are **different test sets** (verified = gpt-5-mini∩gpt-5.2 agreement-filtered; test_clean = unfiltered). Not an error — a reader comparing 73.78% to a Table-11 0.6443 default must know they are different denominators.

4. **Table 1 ↔ Table 11 rounding.** Table 1 "Text-to-LoRA r=64 (default) 0.657" vs Table 11 "LoRA r=64, 7 modules (epoch 2) 0.6572" — same value, 4-decimal vs 3-decimal. (Likewise Prefix 0.504↔0.5044, r=18 0.565↔0.5652.) Consistent.

5. **No numeric prose-vs-table contradiction** was found (unlike some prior papers' abstract-claims-wrong-metric). Every checked prose delta reconciles: PAW−FT 15.4pp (0.1538✓), PAW−FixedLoRA 21.7pp (0.2168✓), pseudo-vs-raw clean 1.6pp (0.0158✓), pseudo-vs-raw typo 4.5pp (0.0446✓), Q4_K_M+Q4_0 loss 1.3pp (0.0127✓), all Table-6 drop-from-clean percentages match recomputation, image-task best-baseline attributions (Circuit 0.196=Qwen3-VL-4B, Chemical 0.258=Qwen3-VL-2B, Music 0.470=Qwen3-VL-2B) all correct.

### Full cell-by-cell source verification (2026-07-13)

**1 numeric defect FOUND + FIXED; all other checked cells exact.**

- **Table 2 (main results, lines 344–367):** all 14 method rows × 5 datasets verified — every accuracy cell exact, every per-program-shipping-size (PS) sub-cell exact **except one**: the gpt-5-mini FuzzyBench PS had been transcribed as `0.95 KB`, but source reads `0.73 KB`. PS is a per-dataset constant across prompting baselines (FuzzyBench always 0.73 KB, YouTube 0.95 KB, …); the YouTube value had leaked into the FuzzyBench cell. Accuracy 91.87% was correct; only the PS sub-cell was wrong. **Fixed** → 0.73 KB. No accuracy claim or takeaway depended on it.
- **Table 4 (LoRA-mapper variants, lines 410–418):** all 5 rows exact (Default 0.6223, Per-position 0.5598, Per-position+per-layer 0.5559, Per-layer 0.6028, LoRA+prefix 0.6033).
- **Table 5 (no-compiler baselines, lines 410–418):** all 5 rows exact (Fixed-LoRA r=18/64/128 = 0.4236/0.5210/0.5159, Full-FT 0.5840, PAW 0.7378). Takeaway deltas recompute (PAW−FT 0.1538, PAW−FixedLoRA-r64 0.2168).
- **Tables 4 & 5 share source lines 410–418** because they are printed side-by-side (two-column), not a citation error.

**Honest-scope surfaces (NOT numeric typos):** (a) the "default" PAW accuracy varies 0.6223–0.6572 across ablation tables because ablations run on `test_clean` while Table 2's 0.7378 is the verified (gpt-5-mini∩gpt-5.2 agreement-filtered) test set — ~10pp gap is a denominator difference, flagged in item 1/3; (b) PAW-0.6B (73.78%) trails gpt-oss-20B (85.45%) on FuzzyBench — the headline "beats Qwen3-32B" (68.70%) is the fair same-paradigm comparison, gpt-oss-20B is a stronger open frontier model; (c) image-task Im2LaTeX regression (PAW-LoRA 0.181 < prefix-tuning 0.391) candidly reported.
