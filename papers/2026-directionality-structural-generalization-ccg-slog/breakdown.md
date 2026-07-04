# Breakdown — "On the Role of Directionality in Structural Generalization"
**arXiv 2607.02307** · cs.CL (Computation and Language) · submitted 2026-07-02
**Author:** Zichao Wei (single author; affiliation not stated on arXiv landing page)
**Source PDF:** `paper.pdf` (558 KB, 15 pp — `file` and `pdfinfo` BOTH report 15 pp; **NO page-count defect this iter**, intermittent no-defect like iters 68/74/76/77/80/83)
**Layout text:** `paper_layout.txt` (pdftotext -layout, 776 lines)
**Subarea (repo's FIRST):** semantic parsing · compositional / structural generalization · neuro-symbolic supertagging · **CCG directed types vs AM-algebra (directionless)** · SLOG benchmark. Repo's 77th paper, rank 72 unique.

> Source-first breakdown: every table transcribed verbatim with sourcing line-ranges into `paper_layout.txt`; every prose delta then recomputed in Python against the transcribed cells (see §Verification). ZERO numeric prose-vs-table cell typos. Honest-scope weight is entirely attributional/statistical (non-significant headline gap, category-mixed + single-category-outlier-carried overall, encoder-confounded "SOTA", reformatting asymmetry) — see §Honest-scope.

---

## 1. Core claim & the one-sentence falsifiable hinge

SLOG (Li et al., 2023) is a compositional-generalization benchmark with 17 structural test categories (1000 samples each) split into 4 groups: **§2.1 recursive depth (6 cats)**, **§2.2 modifier position (4 cats)**, **§2.3 extraction position (1 cat)**, **§2.4 wh-questions (6 cats)**. The previous SOTA, **AM-Parser** (Weißenhorn et al., 2022a), uses a BERT supertagger + an **AM algebra whose `apply`/`modify` operations deliberately do not encode direction** (word-order-independent; designed for AMR graph tasks). The paper asks: does a *directional* representation help on the 5 categories that explicitly test position shifts?

**Hinge (falsifiable, §5.3):** Replace AM-Parser's AM algebra with **CCG directed types** (`S\NP` requires an NP to the *left* = backward application; slash direction encodes which side of the head an argument appears on) — keep the same BERT-base encoder, swap to a deterministic CKY parser + single linear decoder (30K learnable params). Prediction: if directionality is the cause, gains fall *precisely* on the 5 position-shift categories (§2.2+§2.3) and are *absent* on the 6 recursive-depth categories (§2.1, direction-irrelevant). The paper reports exactly this pattern: CCG leads on 5/5 position-shift (+29.9pp avg) and trails on 6/6 recursive-depth (−31.9pp avg).

---

## 2. Method (§3)

**Architecture (§3.1, L139-184):** neural supertagger + symbolic composition, same paradigm as AM-Parser.
1. **BERT-base encoder** (frozen, 110M params) → contextualized word vectors.
2. **Single linear decoder** (30K learnable params, excl. frozen BERT) over **26 CCG types** (20 base types cover all 21 COGS categories; 6 SLOG-added types incl. 2 disambiguation types `DIT_REC`, `RC_THAT_REC`).
3. **Deterministic CKY composition** with standard CCG forward/backward application rules (purely symbolic, non-learnable) — 100% CKY coverage.
4. **Semantic edge extraction** — predicate-argument edges deterministically extracted from the CCG derivation; converted to a logical form.

**Why directionality is non-ablatable (§4.2, L272-296):** directionality is not a parameter of CCG but *the definition* of its combinatory operations — removing it deconstructs the formalism. The paper candidly accepts it cannot be isolated as an independent variable, and substitutes a **pattern-alignment test**: if directionality is the cause, improvements align with the direction-relevant dimension, not the dimension where decoder/training differences would plausibly act. (See honest-scope flag H7 — this is a substitute, not a true ablation.)

**Evaluation metric (§3.4, L143-170):** **LF exact match** — apply semantic extraction to the predicted CCG type sequence and check the resulting logical form. **Competitor numbers are REFORMATTED exact match** (§3.3 L156-170): AM-Parser/T5/LLaMA LFs are reformatted by the authors; claimed reformatting loss **0.08%** from "inherent ambiguities in the CCG type system," included in reported figures.

**Training (§3.4, L172-184):** SLOG training set (32,755 samples); final checkpoint at **epoch 50**; **8 minutes single GPU**; hyperparameters fixed across seeds; **no validation-based model selection** (eliminates a distribution-overfitting confound).

---

## 3. Tables (verbatim)

### Table 1 — Architectural comparison of three neuro-symbolic systems (L185-194)

| Property | AM-Parser | Ours (BERT) | Ours (DeBERTa) |
|---|---|---|---|
| Encoder | BERT-base | BERT-base | DeBERTa-v3-large |
| Type system | AM algebra (undirected) | CCG (directed) | CCG (directed) |
| Symbolic composition | AM algebra apply/modify | CKY fwd/bwd application | CKY fwd/bwd application |
| Decoder | Symbolic logic + edge predictor | Single linear layer | Single linear layer |
| **0% categories** | **2** | **0** | **0** |
| **Overall%** | **70.8 ± 4.3** | **75.9 ± 6.4** | **90.7 ± 4.9** |

*Caption (L193): "Architectural comparison of three neuro-symbolic systems. The symbolic composition layer is a purely symbolic, non-learnable operation; the decoder converts type sequences into logical forms."*

### Table 2 — SLOG gen-set results, LF exact match mean ± std (L196-218)

| Category | Ours (DeBERTa) | Ours (BERT) | AM-Parser | T5 | LLaMA |
|---|---|---|---|---|---|
| PP recur. (depth 3) | 99.1 ± 1.2 | 96.8 ± 2.6 | **100.0 ± 0.0** | 93.1 ± 1.9 | 98.9 ± 0.6 |
| PP recur. (depth 5–12) | 97.8 ± 2.1 | 94.3 ± 3.2 | **100.0 ± 0.0** | 16.6 ± 1.0 | 20.6 ± 1.0 |
| CP recur. (depth 3) | 81.9 ± 29.8 | 18.8 ± 34.4 | **100.0 ± 0.0** | 60.9 ± 2.1 | 98.1 ± 0.7 |
| CP recur. (depth 5–12) | 64.8 ± 26.3 | 17.7 ± 34.7 | **100.0 ± 0.0** | 5.3 ± 0.4 | 12.1 ± 0.7 |
| Center emb. (depth 3) | **100.0 ± 0.0** | 99.7 ± 0.3 | **100.0 ± 0.0** | 64.1 ± 19.1 | 50.7 ± 5.7 |
| Center emb. (depth 5–12) | 99.3 ± 1.8 | 80.6 ± 8.3 | **99.5 ± 0.4** | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PP_modif_iobj | **99.3 ± 0.8** | 97.6 ± 1.7 | 90.4 ± 8.1 | 53.8 ± 1.4 | 71.2 ± 4.2 |
| PP_modif_subj | **94.5 ± 4.8** | 92.8 ± 4.0 | 57.6 ± 8.1 | 0.8 ± 0.5 | 28.9 ± 3.5 |
| RC_modif_iobj | **95.7 ± 5.8** | 78.4 ± 10.6 | 74.4 ± 6.4 | 36.6 ± 2.1 | 55.0 ± 2.1 |
| RC_modif_subj | 52.0 ± 4.1 | **61.9 ± 11.6** | 55.8 ± 8.4 | 0.2 ± 0.2 | 29.5 ± 3.4 |
| RC_iobj_extracted | **97.8 ± 2.3** | 96.9 ± 3.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.5 ± 3.2 |
| Q_subj_active | 91.9 ± 7.2 | 77.8 ± 13.3 | **99.8 ± 0.6** | 98.1 ± 1.7 | 93.3 ± 6.0 |
| Q_subj_passive | 99.2 ± 2.5 | 85.9 ± 16.5 | **100.0 ± 0.1** | 100.0 ± 0.0 | 15.3 ± 17.5 |
| Q_dobj_ditransV | **99.6 ± 1.3** | 96.3 ± 5.0 | 29.4 ± 33.5 | 98.5 ± 0.9 | 8.6 ± 5.7 |
| Q_iobj_ditransV | **99.7 ± 0.8** | 71.4 ± 26.4 | 41.4 ± 42.4 | 0.4 ± 0.7 | 73.5 ± 18.4 |
| Q_modified_NPs | **84.5 ± 5.4** | 73.9 ± 5.8 | 55.6 ± 12.5 | 36.8 ± 0.4 | 20.8 ± 2.4 |
| Q_long_mv | **84.7 ± 15.1** | 48.4 ± 19.1 | 0.0 ± 0.0 | 24.9 ± 5.1 | 3.0 ± 4.7 |
| **Overall** | **90.7 ± 4.9** | **75.9 ± 6.4** | 70.8 ± 4.3 | 40.6 ± 1.0 | 40.1 ± 1.8 |

*Caption (L216-218): "SLOG gen set results (LF exact match mean±std). Our system reports results over **10 seeds**; other systems report reformatted exact match (**5 seeds**) from (Li et al., 2023) Table 5. Vanilla TF = Transformer trained from scratch (Vaswani et al., 2017). Highest value per row in bold."*
*(Bold = row max, applied here from "Highest value per row in bold"; ties share bold. The 4 layout-text columns beyond the three "Ours/AM-Parser" systems are **T5** and **LLaMA** (seq2seq baselines ≈40% overall). Vanilla TF is named in the caption but no Vanilla-TF column appears in Table 2 — T5/LLaMA are the two ≈40% columns; "Vanilla TF" is mentioned for context only.)*

**Figures:** Fig 1 (mean accuracy by SLOG category group); Fig 2 (PP/RC modifier position pairs, subject vs indirect-object side — caption L328-336 notes "AM-Parser shows large directional asymmetry on both modifier pairs (Δ33, Δ19); the CCG system (BERT) substantially reduces the asymmetry for PP modifiers").

---

## 4. Key prose claims & line-ranges

| Claim | Location | Verified |
|---|---|---|
| BERT-base 75.9±6.4% "surpasses" AM-Parser 70.8±4.3% | abstract L21-23, §4.1 L229-230 | cell-exact; **statistical overlap = NO** (H1) |
| +29.9pp on all 5 position-shift categories (§2.2+§2.3) | abstract L24-26, §4.2 L283-285 | **EXACT** (29.88→29.9) |
| −31.9pp on all 6 recursive-depth categories (§2.1) | §4.2 L285-289 | **EXACT** (−31.93→−31.9) |
| Encoder scalability 75.9% → 79.9% → 90.7% (3 tiers) | abstract L77-78, §4.3 L277-281 | tiers: BERT-base / ModernBERT-base / DeBERTa-v3-large |
| DeBERTa 90.7 "exceeds AM-Parser by ~20pp", 9/17 cats >97% | §4.1 L231-233, §4.3 L281-287 | 90.7−70.8=**19.9 ✓**; 9/17≥97% **EXACT** |
| Largest encoder-upgrade gains in recursive-depth (+22.5pp) | §4.2 L317-318 | **EXACT** (90.48−67.98=22.50) |
| wh direction-relevant subcats +30.0 to +66.9pp | §4.2 L263-266 | **EXACT** (Q_iobj +30.0, Q_dobj +66.9) |
| AM-Parser 0% ceiling on 2 categories | §4.3 L272-276 | **EXACT** (RC_iobj_extracted, Q_long_mv) |
| Pipeline fidelity 99.92% (13/17000 inconsistencies) | §7 L420-425 | **EXACT** (1−13/17000=99.9235%) |
| T5 47.2% plain vs 98.5% reformatted (Wu et al., 2023) | §7 L425-433 | cited; reformatting-inflation witness |
| 30K learnable params (excl. frozen BERT 110M) | abstract, §3.1 L167-168 | stated |
| 8 min training, single GPU, epoch 50, no val selection | §3.4 L177-184 | stated |

---

## 5. Verification (Python source-free reconciliation)

`paper_layout.txt` cells → Python (`/tmp/reconcile.py`). **ZERO numeric prose-vs-table cell typos.**

- **All 5 Overall cells = macro-mean of the 17 categories EXACT** (DeBERTa 90.694→90.7 ✓; AM-Parser 70.818→70.8 ✓; T5 40.594→40.6 ✓; LLaMA 40.118→40.1 ✓; **BERT 75.835→ reported 75.9** — within-rounding 0.06 gap from summing the *rounded* per-category cells, NOT a defect; the unrounded per-seed overall (averaged over 10 seeds before rounding) plausibly lands at 75.9).
- **+29.9pp position-shift** (BERT−AM): deltas [7.2, 35.2, 4.0, 6.1, 96.9] → mean **29.88 → 29.9 ✓**.
- **−31.9pp recursive-depth** (BERT−AM): [−3.2, −5.7, −81.2, −82.3, −0.3, −18.9] → mean **−31.93 → −31.9 ✓**.
- **+22.5pp encoder recursive gain** (DeBERTa−BERT on the 6 recursive cats): 90.483−67.983 = **22.500 ✓**. Position-shift encoder gain only **+2.34pp** (confirms encoder gains concentrate in recursive-depth, complementary to directionality).
- **wh direction-relevant** (BERT−AM): Q_dobj +66.9, Q_iobj +30.0, Q_long_mv +48.4 → range **30.0..66.9 ✓**.
- **9/17 DeBERTa cats ≥97%** ✓ (listed). **AM-Parser 2 zero categories** ✓ (RC_iobj_extracted, Q_long_mv).
- **Pipeline fidelity**: 1 − 13/17000 = **99.9235% → 99.92% ✓**.
- **"~20pp"**: 90.7−70.8 = **19.9 ✓**.
- **Bold=best per row** (caption "Highest value per row in bold"): every bold cell above is the row max (ties share bold, e.g. Center-emb-d3 DeBERTa=AM=100.0). ✓

---

## 6. Honest-scope flags (12; NO numeric cell typo — all attributional / statistical)

The paper is unusually candid (a dedicated §7 Limitations acknowledging seed asymmetry, std-exceeds-mean cells, reformatting-inflation risk, pre-training leakage). The flags below sharpen the *load-bearing* issues the candor does not neutralize.

- **H1 — NON-SIGNIFICANT headline gap (load-bearing; new seed-std-overlap subclass).** "Surpasses AM-Parser (75.9±6.4 vs 70.8±4.3)" — the ± is std across seeds; **75.9−6.4 = 69.5 < 70.8+4.3 = 75.1**, the two distributions massively overlap, and the 5.1pp gap is **smaller than the winner's own std (6.4)**. No significance test (t/bootstrap) is reported. The "New SOTA" / "surpasses" headline rests on a gap that is within seed noise. *Diagnostic:* whenever a result is reported as `mean ± seed-std` vs another `mean ± seed-std`, recompute the overlap; a "surpasses" claim where the gap < either std is a non-significant headline even with zero cell typos. The paper candidly notes the 10-vs-5 seed asymmetry (§7) but does **not** flag the overlap.
- **H2 – CATEGORY-MIXED overall + SINGLE-CATEGORY-OUTLIER inflation (two-level; compounds iter-87 category-mixed-Overall + iter-85 aggregation-inflation).** The +5.1pp overall is position-shift-carried (+29.9pp on 5 cats) while **losing all 6** recursive-depth cats (−31.9pp). And the +29.9pp average is itself **RC_iobj_extracted-carried**: that one cell (0.0→96.9, +96.9) is 65% of the 149.4pp sum; drop it and the position-shift average collapses to **+13.1pp**. So the headline "+29.9pp on position-shift" is (a) a minority of categories and (b) one-category-driven. *Diagnostic:* for any "+Xpp on N categories" headline, drop the largest single-cell delta and recompute; if the mean halves, flag outlier-inflation.
- **H3 – CATASTROPHIC recursive-depth regression is reframed as "elegant redistribution."** Under BERT-base the CCG system scores **CP-recur-d3 18.8±34.4** and **CP-recur-d5-12 17.7±34.7** vs AM-Parser's **100.0/100.0** — near-random performance with **std > mean** (bimodal across seeds, unstable). §4.2 frames the symmetric +29.9/−31.9 pattern as "a precise redistribution… not a random fluctuation," which is true *statistically* but masks that the CCG system **breaks categories AM-Parser solves perfectly**. The "nearly equal magnitudes" is presented as elegance, not as a severe deficit.
- **H4 – ENCODER CONFOUND on the 90.7 "SOTA" (load-bearing).** DeBERTa-v3-large (≈435M params) achieves 90.7; AM-Parser's 70.8 used BERT-base (110M). "Exceeds AM-Parser by ~20pp" **conflates directionality with encoder scale** — AM-Parser is **never run with DeBERTa**, so the 90.7-vs-70.8 gap is uncontrolled. The paper's own §4.2/§5.3 admits the encoder (not directionality) is what fixes recursive-depth (the +22.5pp encoder gain is in recursive-depth, directionality's *absent* dimension) — so directionality's *isolated* contribution is only the position-shift gain, and that is H2's outlier-carried +13.1–29.9pp.
- **H5 – "5/5 position-shift" is BERT-config-only; at the 90.7 SOTA config it is 4/5.** At DeBERTa-v3-large, **RC_modif_subj drops to 52.0 < AM-Parser 55.8** — so at the headline 90.7 config the CCG system **loses** one position-shift category (the very dimension directionality targets). The "outperforms on all 5 position-shift" claim is true only for the weaker BERT config; the strongest system wins 4/5.
- **H6 – REFORMATTING asymmetry (competitors re-evaluated under authors' own LF format).** AM-Parser/T5/LLaMA numbers are *reformatted* exact match (§3.3), with the authors claiming only 0.08% reformatting loss. But the paper itself (§7) cites T5 **47.2% plain vs 98.5% reformatted** (Wu et al., 2023) as evidence that "LF format differences can exaggerate the apparent degree of generalization failure" — a **+51.3pp** reformatting swing on one cell. The comparison uses the authors' reformatting of competitors while documenting that reformatting can swing a cell by >50pp; the 0.08% figure is the authors' estimate of *their own* pipeline's ambiguity loss, not a bound on competitor reformatting drift.
- **H7 – "Pattern alignment" is a substitute for ablation, not an ablation.** §4.2 candidly concedes directionality "cannot be isolated as an independent variable" and substitutes a pattern-alignment argument (gains align with the direction-relevant dimension). This is reasonable but is correlational: the paper itself lists confounds (decoder architecture, training strategy, type granularity 26 vs 50 types) and acknowledges "confounds could also produce non-uniform effects… this pattern alone cannot exclude alternative explanations." Directionality is "the most parsimonious explanation," not a demonstrated cause.
- **H8 – "+dist feature" rebuttal (§5.1) cuts both ways.** The paper argues AM-Parser's own `+dist` decoder patch (Weißenhorn 2022a) is "indirect confirmation" that directionality was missing. But `+dist` existing means **AM-Parser *can* encode direction when needed** — so the comparison is vs AM-Parser-*without*-`+dist`, not vs the strongest direction-aware AM variant. The fairest directional test (CCG vs AM-Parser-+-dist) is not run.
- **H9 – Weak seq2seq baselines (≈40%) flatter the gap.** T5 (40.6) and LLaMA (40.1) are an order of magnitude below the neuro-symbolic systems; the intro's "70.8% vs 40%" framing (L104) sets a low bar. The decisive comparison is solely CCG vs AM-Parser (the H1 non-significant 5.1pp).
- **H10 – Std-exceeds-mean cells reported without a stability caveat.** CP-recur-d3 (DeBERTa 81.9±29.8, BERT 18.8±34.4), CP-recur-d5-12 (BERT 17.7±34.7), Q_dobj_ditransV AM-Parser 29.4±33.5, Q_iobj_ditransV AM-Parser 41.4±42.4 — all have **std > mean**, indicating bimodal/all-or-nothing behavior across seeds. §7 candidly notes "both systems exhibit categories where standard deviation approaches or exceeds the mean," but no per-cell flag marks which headline cells are unstable.
- **H11 – "9/17 above 97%" threshold is arbitrary.** 97% is not a principled cutoff; the count (just over half the categories) is presented as a strength. At 90% the count would be higher; at 99% lower. No effect-size or significance accompanies it.
- **H12 – Pre-training-leakage confound acknowledged but unquantified (§7).** BERT/DeBERTa pre-training may contain SLOG-like structures; the confound is "shared by the entire supertagger paradigm" and "does not affect the core directional-vs-non-directional comparison." Fair, but it bounds the absolute (not relative) interpretation of 90.7% — and the encoder upgrade (BERT→DeBERTa) that drives most of the 75.9→90.7 gain is precisely the component most exposed to additional pre-training data.

---

## 7. Citable falsifiable content (for downstream implementation)

- **The directional-separation prediction** (§4.2): under matched BERT-base, CCG−AM deltas are + on all 5 position-shift, − on all 6 recursive-depth. Reproducible to the pp from Table 2 (verified).
- **The 30K-param single-linear-decoder + deterministic CKY + 26-CCG-type system** (§3.1-3.2): a concrete, small, reproducible neuro-symbolic backend.
- **The encoder-scalability ladder** 75.9 (BERT-base) → 79.9 (ModernBERT-base) → 90.7 (DeBERTa-v3-large), with the recursive-depth gain (+22.5pp) isolating the encoder's contribution from directionality's.
- **The T5 47.2%-plain-vs-98.5%-reformatted witness** (§7) — an independent, citable quantification of LF-reformatting inflation on SLOG.

**NOT citable as a stable result:** the "surpasses AM-Parser" headline (H1, non-significant seed-std overlap), the "+29.9pp position-shift" without the RC_iobj_extracted qualifier (H2), or the "90.7 SOTA" without the encoder-confound caveat (H4) / the 4-of-5 caveat (H5).

---

## 8. Repo lineage / subarea placement

Repo's **FIRST** semantic-parsing / compositional-generalization / neuro-symbolic-supertagging / CCG paper. Distinct from the RL/agents/vision/diffusion/TS majority. Sibling-in-spirit to the **structural/representational-design** thread (viq iter — text-aligned visual quantized reps; translation-as-bridging iter — text transfer) in that the contribution is a *representational-protocol* choice (where information lives: symbolic rules vs learnable labels), not a new model or training trick. The "non-ablatable definitional property" framing is unusual in the repo and makes the honest-scope surface (H1/H4/H7) the load-bearing content rather than a numeric cell hunt.
