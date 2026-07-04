# On the Limits of Steering Vectors for Preference-Aligned Generation

**arXiv:** 2607.01802v1 [cs.CL], 2 Jul 2026 (8pp)
**Authors:** Melanie Subbiah\*, Zara Hall\*, Kathleen McKeown — Department of Computer Science, Columbia University
**Code/data:** https://github.com/melaniesubbiah/steering-vectors
**Source-first breakdown** built from `paper.pdf` / `paper_layout.txt` (pdftotext -layout, 772 lines).
Sourcing line-ranges cite `paper_layout.txt`.

---

## 1. Thesis

Steering vectors (residual-stream directions extracted from paired trait-positive /
trait-negative examples) are a popular training-free, interpretable alignment knob, but
prior work tests only a handful of traits (helpful, harmless) and applies one vector at a
time. This paper stress-tests them along **three axes** — **(1) trait expressibility**,
**(2) task transfer**, **(3) multi-trait composition** — and finds meaningful limits on all
three, concluding steering vectors are **not** a general-purpose preference-alignment tool.
Headline experimental surface: **36 stylistic traits × 2 models × 2 transfer tasks,
combinations up to 4 vectors simultaneously** (Fig 1).

**Subarea placement (repo context):** this is the repo's first **empirical-limits / scaling
study of activation steering**. It is a *negative-result* sibling to the mech-interp steering
lineage — distinct from `refusal-subspaces` (which *learns* a multi-dim refusal cone to
ablate) and `subliminal-clocks` (which *steers* a diffusion-LM residual stream toward a
denoising-timestep). Where those papers *build* a steering mechanism, this paper *audits* the
ParadigmVectors (Chen et al. 2025) extraction-and-apply pipeline and shows it breaks under
scale along trait-type, task, and count.

---

## 2. Method — PersonaVectors pipeline (§3, L142–226)

For a language model `fθ` with `L` transformer layers and hidden dim `d`, hidden state of
token `t` at layer `l` is `h_t^(l) ∈ R^d`. Per trait `k`, a layer-`l` steering vector is the
**difference of mean hidden activations** between trait-present and trait-absent corpora
(L203–205):

```
v_k^(l) = h̄_k^(l) − h̄_\k^(l)
```

where `h̄_k^(l)` is the mean hidden activation at layer `l` over responses exhibiting trait
`k` (and across all tokens in the response), `h̄_\k^(l)` the mean over responses/tokens
without it. At inference, applied to the residual stream at a chosen layer (L213–218):

```
h̃_t^(l) = h_t^(l) + α · v_k^(l)
```

`l` and `α` are tuned parameters (§4.1). Four-step pipeline:

1. **Paired data generation** — generate positive prompts (elicit trait) and negative prompts
   (suppress trait), paired with content-agnostic questions ("How important is exercise for
   health?") so vectors are trait-only, not topic-only. 125 positive + 125 negative questions
   per trait. Prompt-generation instructions from Chen et al. (2025) with this paper's own
   trait definitions (Appendix A). **Claude-3.7-Sonnet** generates the prompts/questions
   (powerful LLM for nuance); **responses must come from the steered model itself** (steering
   needs direct activation access), so 100 responses/question → **5000 responses/preference
   before filtering**.

2. **Extract activation-based steering vectors** — layer-level activation differences (above).

3. **Steering output** — add `α·v` to residual stream at layer `l`.

4. **Combine multiple traits** (Appendix B) — 4 methods:
   - **Orthogonalized** — orthogonalize the vectors, multiply by individual α's, sum.
   - **Different layers** — apply each vector at a different layer × its α.
   - **Tuned mean** — multiply each vector by its α, take the mean.
   - **Unit norm** — mean of the unit-norm vectors × an α.

---

## 3. Experimental setup (§4, L179–286)

| Item | Value |
|---|---|
| **Models** (response gen) | Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct (similarly sized, two architectures) |
| Prompt/question gen | Claude-3.7-Sonnet |
| Paired data | 125 pos + 125 neg questions/trait; 100 responses/question → 5000/preference pre-filter |
| Filter | extraction trait-expression judge: expression **>50** (pos), **<50** (neg) |
| Hardware | **2× 40GB A100** GPUs |
| #Traits | **36** stylistic preferences |
| **Tasks** | (a) **Extraction tasks** — Claude-generated trait-elicitation prompts ("How can someone improve public speaking?"); used to tune α/layer. (b) **PLUME tasks** — summarization + email-writing preference-aligned tasks; used for transfer + multi-trait. |
| PLUME sizes | 25 examples/source-dataset → **125 summarization tasks, 100 email-writing tasks** |
| Candidate layers | **16 and 20** (prior work) |
| α tuning | coarse integer search **[1,5]** at each layer (5 questions/combo) → pick max expression w/ coherence **≥75**; fine sweep ±0.5 around anchor at **0.1** increments; early stop if expression **≥90** w/ coherence ≥75 |

**Three LLM-as-judge metrics** (GPT-4o / GPT-4.1-mini):

| Metric | Judge | Scale | Use |
|---|---|---|---|
| **PLUME trait expression (PPCM)** | GPT-4o | 5-pt Likert (−2..2) averaged over ground-truth prefs `P`: `PPCM = Σ_{p∈P} J(y,p) / |P|` | PLUME tasks |
| **Extraction trait expression** | GPT-4.1-mini | 0–100 | extraction tasks + paired-data filtering |
| **Coherence score** | GPT-4.1-mini | 0–100; prior threshold 50, **this paper uses stricter 75** | both |

---

## 4. Results

### 4.1 Trait expressibility — *which traits can be steered?* (§5.1, Fig 3, L261–368)

**Figure 3** = trait-expression vs coherence scatter (top) + ranked top-10/bottom-10 bars
(bottom), per model. Findings:

- Both models have vectors giving **minimal trait expression and/or incoherent output**.
  **Qwen has more incoherent outputs; Llama has more unexpressed traits.**
- **Top-10 shared across both models** (reliably steerable): *bullets (bullet parallel),
  step-by-step, formal tone, rhetorical questions, all-caps emphasis, informal tone*. These
  follow common structural/stylistic tendencies the models already have.
- **Bottom-10 shared across both models** (least effective): *tweet style, third person
  perspective, Q&A style, rhyming structure, conditional expressions, screenplay*. These use
  unconventional rhetorical devices that don't affect all output equally (e.g. rhyming only
  matters at line-ends).
- Model-specific: **emoji usage** works for Qwen; **onomatopoeia** for Llama.
- Takeaway: traits following common LLM output styles / affecting **whole-output structure**
  are most steerable; **local/unconventional** styles least.

> **Sourcing caveat (Fig 3):** per-trait expression/coherence points are scatter/bar
> readings — the qualitative top-10/bottom-10 lists and the Qwen-vs-Llama direction are
> prose-confirmed (§5.1, L330–368); individual per-trait 0–100 scores are figure-bar reads
> and were not back-filled. The Top/Bottom-10 trait-name lists above are taken verbatim from
> the figure's printed label strips (L328–349).

### 4.2 Task transfer — *does steering transfer across tasks?* (§5.2, Fig 4, L370–470)

**Figure 4** = (left) per-trait extraction-score vs downstream-PLUME-score scatter (diagonal
= perfect transfer); (right) transfer-gap bars (extraction − downstream). Findings:

- **Significant change** in steering effectiveness between extraction and PLUME tasks —
  steering **depends on context** and vectors may capture **task-dependent** aspects of a
  trait.
- **Summarization** has more traits that degrade significantly (more impersonal task → less
  natural to inject style).
- Both models: **second-person narrative** and **header structured** transfer **poorly**;
  **tweet style** **increases** in expression downstream.
- Performance is highly dependent on task + judge used for extraction/tuning, and **can
  degrade substantially** when the resulting vectors are applied downstream.

> **Sourcing caveat (Fig 4):** per-trait points are scatter/bar readings; the directional
> claims (poor transfer, summarization worse, second-person/header/tweet) are prose-confirmed
> (§5.2, L360–379).

### 4.3 Multi-trait combination — *best method for ≥2 vectors?* (§5.3, Table 2, L473–486)

**Table 2 (verbatim, mean ± SEM; trait + coherence on 0–100; Δ vs baseline = signed diff
from single-trait score).** Two-trait simultaneous application, same setting/judge as
extraction (isolates combination from transfer):

| Model | Method | Trait 1 | Trait 2 | Average | Δ vs baseline | Coherence |
|---|---|---|---|---|---|---|
| **Qwen** | Orthogonalized | 45.2 ±4.2 | 48.3 ±4.4 | 46.8 ±3.7 | −28.5 ±2.7 | 36.8 ±3.3 |
| Qwen | Different layers | 51.3 ±3.9 | 48.5 ±3.9 | 49.9 ±3.2 | −26.6 ±2.5 | 40.5 ±3.2 |
| Qwen | Tuned mean | 43.2 ±4.4 | 47.6 ±4.6 | 45.4 ±3.6 | −29.9 ±2.8 | 47.9 ±4.2 |
| Qwen | Unit norm (layer=16, α=30) | 39.4 ±5.4 | 33.5 ±5.2 | 36.5 ±3.7 | −40.1 ±3.5 | 97.0 ±1.2 |
| Qwen | **Unit norm (layer=20, α=58)** | **63.0 ±5.5** | 49.7 ±5.7 | **56.4 ±4.1** | −20.1 ±3.3 | 75.9 ±3.5 |
| **Llama** | Orthogonalized | 63.9 ±4.4 | 49.0 ±5.1 | 56.5 ±3.7 | −15.7 ±2.9 | 56.0 ±4.0 |
| Llama | Different layers | 55.2 ±4.0 | 50.7 ±4.1 | 53.0 ±3.1 | −18.9 ±2.4 | 47.7 ±3.4 |
| Llama | Tuned mean | 61.9 ±4.6 | 42.8 ±4.9 | 52.3 ±3.6 | −19.9 ±2.9 | 65.0 ±4.1 |
| Llama | Unit norm (layer=16, α=4) | 51.0 ±5.9 | 41.4 ±5.8 | 46.2 ±4.2 | −25.7 ±3.5 | 94.5 ±1.0 |
| Llama | Unit norm (layer=20, α=8) | 55.8 ±6.0 | 42.6 ±5.7 | 49.2 ±4.7 | −22.7 ±3.3 | 87.6 ±2.5 |

*(Table 2 source: paper_layout.txt L504–525.)*

Findings (§5.3, L476–486):

- **All methods: trait expression drops ≥15%** when combining two vectors vs single-trait.
- Many outputs incoherent (coherence <75%).
- One trait often dominates; **more prominent in unit-norm methods** (traits balanced equally,
  no re-balancing from different coefficients).
- The only method reducing trait expression *only moderately* while maintaining coherence on
  **both** models is **unit norm at layer 20** — but trait expression is **imbalanced** there
  too.

> **Δ-vs-baseline reconciliation (source-free check):** Δ = Average − single-trait baseline
> ⇒ baseline = Average − Δ. Per model this clusters tightly: Qwen 75.3–76.6 (Orthog 75.3,
> Diff-layers 76.5, Tuned-mean 75.3, Unit-L16 76.6, Unit-L20 76.5); Llama 71.9–72.2 (Orthog
> 72.2, Diff-layers 71.9, Tuned-mean 72.2, Unit-L16 71.9, Unit-L20 71.9). The small
> within-model spread reflects each method's own single-trait reference (tuned at its own
> layer/α), not an inconsistency. **Qwen Unit-norm-L16 (layer 16, α=30) and Llama Unit-norm-L16
> (layer 16, α=4) buy coherence (97.0 / 94.5) at the price of the worst expression (36.5 /
> 46.2) and largest Δ-drop (−40.1 / −25.7)** — the coherence-vs-expression tradeoff is
> sharpest for layer-16 unit-norm.

### 4.4 Scaling to more vectors (§5.4, Fig 5, L488–545)

**Figure 5** = PPCM trait-expression (left) + coherence (right) vs number of combined traits
(1→4) per method.

- **Unit-norm methods stay substantially more coherent** across more traits; other methods
  cross into incoherence.
- Placing vectors on **different layers amplifies** each trait → incoherence; **same layer**
  lets the model balance competing effects.
- **Trait expression decreases across all methods as traits added**; **most methods have
  negative or no expression by 4 traits.**
- **Qwen: unit norm at layer 20** balances expression/coherence best; **Llama: tuned mean**
  best. For unit-norm, coherence *increases* as traits added → **α likely needs to increase
  with #traits** to balance — exposing yet another parameter requiring tuning.

> **Sourcing caveat (Fig 5):** per-(method, #trait) PPCM/coherence points are curve readings;
> the qualitative rankings (unit-norm most coherent, expression monotone-decreasing, best
> method per model) are prose-confirmed (§5.4, L537–545).

---

## 5. Conclusion (§6, L547–558)

- Some traits are **not effectively expressed** through steering; transfer depends on how
  natural steering is in the target task.
- **Mean of unit-normalized vectors × α is the most effective combining method**, but
  expression + coherence still decrease as traits combine; α may need **re-tuning per
  #traits**.
- Steering vectors face **meaningful limits on generality** for both controllable generation
  and for alignment research that uses them to monitor/indicate trait expression in
  data/inputs (unreliable off the extraction distribution).

**Limitations:** (a) many tuning params (layer hooks, LLM-judge eval) — cannot exhaustively
tune; selecting perfect coefficients for 4 simultaneous vectors is a complex optimization; (b)
must modify/extract residual stream → **open-source only**; (c) only 2 models → **cannot
conclude limits apply to much larger models** (but patterns give pointers); (d) relies on
**LLM-as-judge** (human-in-the-loop infeasible at this scale); metrics are prior-work-validated
and serve coarse tradeoff demonstration.

---

## 6. Strengths / Limitations / Verdict

**Strengths**
- First **systematic empirical-limit study** of steering-vector scale (36 traits × 2 models ×
  2 transfer tasks × up to 4 simultaneous vectors) — fills a gap in a literature that mostly
  tests 1–3 traits with 1 vector.
- Clean 3-axis decomposition (expressibility / transfer / composition) with a single coherent
  pipeline (PersonaVectors) — makes the failure modes comparable across axes.
- Honest negative framing: surfaces a concrete coherence-vs-expression tradeoff and a
  per-#traits α-re-tuning burden rather than overclaiming steering as general-purpose.
- Table 2's full mean±SEM grid lets a reader see that **no method dominates**: the only
  coherence-preserving option (unit-norm L20) is also expression-imbalanced.

**Limitations**
- **Scale ceiling unclear**: only 7B–8B models; the authors flag they cannot conclude for
  larger models — the headline "limits on generality" is therefore a 7B–8B finding.
- **All evaluation is LLM-as-judge** (GPT-4o PPCM, GPT-4.1-mini extraction/coherence) with
  no human cross-validation in this paper (only prior-work validation cited); trait-expression
  absolute scores inherit judge bias.
- Trait set is **stylistic-only** (PLUME writing personalization) — does not test safety,
  factuality, or reasoning traits where steering is also deployed, so "general-purpose" limits
  are scoped to stylistic preference alignment.
- Tuning budget is bounded — the negative result partly reflects under-tuned α/layer at
  composition, which the authors acknowledge.

**Verdict:** a well-scoped negative-result paper that should temper claims of steering vectors
as a universal alignment/interpretability tool. The most citable single result is the
**multi-trait composition failure**: by 4 simultaneous vectors, most methods reach **negative
or zero** trait expression (Fig 5), and even at 2 vectors every method loses **≥15%**
expression (Table 2). The honest actionable finding is the **unit-norm-at-layer-20**
compromise (best coherence preservation) and the warning that **α must scale with the number
of traits** — i.e. composition is not free even with the best method.
