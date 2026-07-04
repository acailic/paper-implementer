# SpeechCombine — Instruction-Following Speech Language Models without Instruction Tuning

**arXiv:** 2607.02214v1 (cs.CL, 2 Jul 2026) — ICML 2026, PMLR 306
**Authors:** Congrui Du*, Yang Zhang*, Kaizhi Qian*, Shiyu Chang (UC Santa Barbara; MIT-IBM Watson AI Lab, IBM Research)
**Code:** github.com/CongruiDu/SpeechCombine · **Demo:** auspicious3000.github.io/SpeechCombine-Demo
**Source PDF:** `paper.pdf` (924 KB, **18 pp — `file` AND `pdfinfo` both 18 pp, NO page-count defect this iter**), `paper_layout.txt` (pdftotext -layout, 1110 lines, **7 explicit tables + Eqs 1–2 + Figs 1–4**).

---

## 1. Core idea (one sentence)

Build an instruction-following speech LM (SLM) **without any speech instruction-tuning** — only a single 30k-hour speech continuous-pre-training round — by linearly **combining two weight-difference (task) vectors** in parameter space: the speech-modality direction `∆θ_speech` and the text-LLM instruction-following direction `∆θ_inst`.

This is **model arithmetic / Chat-Vector-style weight composition** (Ilharco Task Arithmetic 2022; Huang Chat Vector 2024) applied to SLM training, escaping the speech "data inflation" bottleneck (a 5-token sentence → 60–200 speech tokens, ~12–40×).

---

## 2. Method (Eqs 1–2, §3, L145–162)

Let `θ_base` = text LLM base, `θ_inst` = its instruction-tuned twin. Continuous-pre-train `θ_base` on speech → `θ_speech`.

**Eq 1** — two directions in parameter space:
```
∆θ_inst   = θ_inst   − θ_base      # encodes instruction-following, no speech knowledge
∆θ_speech = θ_speech − θ_base      # encodes speech modality, no instruction-following
```

**Eq 2** — SpeechCombine (`λ ∈ (0,1]`, soft weight, **λ = 0.85** main):
```
θ_SC = θ_base + λ·∆θ_speech + ∆θ_inst
```
i.e. transplant the instruction direction onto the speech-adapted base. A smaller λ keeps `θ_base + λ∆θ_speech` closer to base → easier ∆θ_inst transfer, but weaker speech knowledge (sweet-spot study Tables 4–5).

**Critical design choice (§3.1, L176–178):** continuous pre-training must start from `θ_base`, **not** `θ_inst` — "major distinction from existing SLM training recipes that also involve model merging (Fun-Audio-Chat)" — to retain the compositional structure of knowledge/skill. (∆θ_inst is full-coefficient 1; only ∆θ_speech is scaled by λ.)

### 2.1 Pre-training data structure (§3.3, L246–273)
Standard next-token prediction on interleaved sentences:
```
[S1 cap][S1 text][S1 speech][S1 cap] [S2 cap][S2 text][S2 speech][S2 cap] ...
```
- `[speech]` = prosody-token sequence (only prosody, not waveform) → familiarises model with speech-token structure (**goal 1**).
- `[cap]` = natural-language speech caption (content + prosody) → teaches speech information (**goal 2**).
- `[text]` = exact transcription, **always paired with `[speech]`** — serves as the **anchor** for transferring ∆θ_inst.
- Randomised `[cap]` placement: before `[speech]` w.p. 0.3, after w.p. 0.3, omitted w.p. 0.4 → 3 transition patterns teach generation / understanding / multi-sentence generation.

### 2.2 Speech caption (§3.4, L843–883)
Per-utterance attributes extracted then a random subset given to **GPT-OSS-120B** to render a natural-language caption:
- avg pitch level (Low `F0<0.75·m_s`/`<0.85·m_g`; High `>1.30·m_s`/`>1.20·m_g`; else Medium), OR-fused across speaker-level `m_s` and gender-level `m_g` medians (RMVPE F0).
- avg speaking rate (Low `S_r>1.3·m`; High `S_r<0.85·m`).
- emphasised words (Whistress extractor), perceived emotion, emotional arousal (`<0.282` Low / `>0.54` High; whisper-large-v3-msp-podcast-emotion-dim, sliding window).
- text transcription.
Sampling: 1 attribute chosen uniformly; if text, dropped w.p. 0.9; emotion datasets always sample emotion attr.

### 2.3 Speech tokenisation — prosody tokens (§3.5 + App A.3, L885–906)
Follows **ProsodyLM** (Qian 2025, modified StyleTTS-2 decoder). Per **word**, 5 numerical prosody values quantised to **512 discrete levels** each: `<f0 med>` (median Log-F0), `<f0 range>` (95th−5th pct Log-F0), `<f0 slope>` (best-fit-line slope), `<Dur>` (duration in frames), `<energy>` (log-norm mel-spectrogram). Plus `<spk f0 med>` (speaker median Log-F0) at section start, `<SIL>` silences. Enclosed by `<speech>`/`</speech>` delimiters. **Encodes prosody only → much shorter than full-waveform codecs; no timbre/voice-quality/accent (authors' own limitation).**

### 2.4 Inference + format forcing + long thinking (§3.6–3.7, App A.4–A.5)
- External **ASR** (whisper-large-v3) transcribes speech query → `[text]` anchor; prosody then extracted. ASR internalisation left as future work.
- Answer template mirrors the text instruct LLM's, with `[speech]` sections: `<|im_start|>Assistant\n[text][speech][text][speech]...`.
- **Format forcing** (4 logits processors): Speaker-F0-Median (force `<spk f0 med>` after `<speech>`; first gen token non-speech); Thinking (ban speech tokens during `<think>`); Deep Format-Forcing (force speech tokens after text); Temperature (different T for text vs speech segments).
- **Long thinking "for free":** insert `<think>[text]</think>` after Assistant; append "Okay" to disambiguate thinking-vs-continuation; ban prosody tokens during think; ban `</think>` until **160-token** minimum length. No training/weight change. Enabled by default.
- Special-token anti-forgetting: `<|im_end|>` inserted w.p. 0.20 and `</think>` w.p. 0.02 at end of each `[text]` section in training data.

---

## 3. Experimental setup (§4.1, L384–403)

- **Backbone:** QWEN3-8B-base (`θ_base`) & -instruct (`θ_inst`); LoRA continuous pre-training **rank 64, α 16**.
- **Speech data ~30k hours** after dropping extraction failures: LibriLight, BEAT, CREMA-D, ESD, JL Corpus, EmoV-DB, Expresso, MEAD, TESS. (~100h emotion captions, >15k h emphasis info.)
- **λ = 0.85.** whisper-large-v3 transcription; ProsodyLM prosody tokens.
- **Baselines:**
  - **Group A** (controlled, same base/data): ASR+TEXT-LLM (topline, no forgetting); CONT. PRE-TRAIN (pre-train on `θ_inst` instead of base); CONT. PRE-TRAIN+SFT (+ ~10k h speech instruction data from SIFT-50M/InstructS2S-200K/VoiceAssistant-400K).
  - **Group B** (SOTA SLMs): GPT-4o-AUDIO (excluded from bolding, much larger); GLM-4-Voice 9B; Audio-Flamingo-3; Step-Audio-2-Mini(7B)/-Think; Osum-eChat 3B; Qwen-2.5-Omni; Kimi-Audio 7B; Fun-Audio-Chat 8B (trained on **millions** of hours, "over 100× the size" of SpeechCombine's data).
- **Tasks:** (a) text-oriented QA (OpenbookQA, MMSU) + reasoning (GSM8k-Eval, TruthfulEval, MLC, MLCpro-en) from VoiceBench/URO-Bench; (b) speech understanding (UnderEmotion-en, EmphAssess-det understanding, ex04 speaker only); (c) speech generation (GenEmotion-en = Emotion2Vec-prob × (1−WER); EmphAssess generation via CosyVoice synthesis, P/R/F1 from emphasis detector).

---

## 4. Results tables (verbatim, source-first)

### Table 1 — Text-oriented tasks, accuracy %↑ (L358–376). Group B bold/underline excl. GPT-4o.
| Method | OpenbookQA | MMSU | GSM8k | Truthful | MLC | MLCpro |
|---|---|---|---|---|---|---|
| GPT-4o-AUDIO¹² | 89.23 | 80.25 | 80.00 | 82.67 | 80.00 | 46.67 |
| ASR+TEXT-LLM | 83.29 | 73.22 | 94.61 | 71.12 | 93.26 | 94.13 |
| CONT. PRE-TRAIN | 78.46 | 68.21 | 87.05 | 42.11 | 85.31 | 88.27 |
| CONT. PRE-TRAIN+SFT | 80.21 | 60.8 | 87.34 | 42.58 | 83.23 | 88.27 |
| GLM-4-Voice¹² | 53.41 | 39.75 | 30.93 | 59.28 | 57.82 | 65.20 |
| Audio-Flamingo-3³ | 58.68 | 42.19 | 37.11 | 32.02 | 68.36 | 61.9 |
| Step-Audio-2-Mini | 72.74 | 54.42 | 42.89 | 53.87 | 87.38 | 80.21 |
| Step-Audio-2-Mini-Think | 65.70 | 53.87 | 39.46 | 52.89 | 85.12 | 76.55 |
| Osum-eChat³ | 78.46 | 60.11 | 32.41 | 36.82 | 46.70 | 51.28 |
| Qwen-2.5-Omni | 81.1 | 61.32 | 38.09 | 46.25 | 73.33 | 64.83 |
| Kimi-Audio¹ | 83.52 | 62.17 | 95.53 | 57.61 | 93.59 | 87.91 |
| Fun-Audio-Chat⁴ | 83.52 | 71.08 | 88.31 | 61.27 | 93.97 | 93.40 |
| **SpeechCombine** | **86.59** | **73.38** | 90.03 | 60.09 | **93.97** | 89.01 |

*(Footnotes: ¹QA from Chen 2024. ²Reasoning from Yan 2025. ³QA from Geng 2025. ⁴QA from Chen 2025.)*

### Table 2 — Speech understanding + generation (L423–446). Group B bold/underline excl. GPT-4o.
| Method | UnderEmo Acc↑ | EmphDet P↑ | EmphDet R↑ | EmphDet F1↑ | GenEmo Score↑ | EmphGen P↑ | EmphGen R↑ | EmphGen F1↑ |
|---|---|---|---|---|---|---|---|---|
| GPT-4o-AUDIO¹ | 48.53 | 33.00 | 61.68 | 42.99 | 33.46 | 66.95 | 63.19 | 65.02 |
| ASR+TEXT-LLM² | 55.42 | 15.79 | 26.94 | 19.91 | 5.06 | 11.35 | 29.71 | 16.42 |
| CONT. PRE-TRAIN | 47.15 | 1.53 | 6.51 | 2.48 | 24.23 | 6.9 | 17.5 | 9.89 |
| CONT. PRE-TRAIN+SFT | 48.90 | 2.12 | 3.42 | 2.61 | 24.23 | 16.76 | 26.95 | 20.67 |
| GLM-4-Voice | 52.41 | 11.92 | 29.52 | 16.98 | 48.13 | 24.67 | 28.75 | 26.55 |
| Audio-Flamingo-3³ | 24.28 | 17.09 | 33.32 | 22.59 | – | – | – | – |
| Step-Audio-2-Mini | 44.57 | 19.59 | 25.76 | 22.25 | 31.35 | 22.48 | 23.23 | 22.85 |
| Step-Audio-2-Mini-Think | 45.83 | 20.37 | 39.45 | 26.87 | 23.73 | 22.75 | 23.11 | 22.93 |
| Qwen-2.5-Omni | 35.86 | 3.75 | 26.26 | 6.56 | 13.54 | 18.82 | 34.03 | 24.24 |
| Osum-eChat | 48.71 | 22.51 | 9.02 | 12.88 | 36.01 | 13.87 | 19.61 | 16.25 |
| Kimi-Audio | 63.69 | 19.71 | 20.04 | 19.88 | 28.01 | 18.21 | 26.81 | 21.69 |
| Fun-Audio-Chat | 74.74 | 23.46 | 37.15 | 28.76 | 39.30 | 23.01 | 22.82 | 22.91 |
| **SpeechCombine** | 52.70 | **55.11** | **67.89** | **60.84** | 45.42 | **25.91** | **39.90** | **31.42** |

*(Footnotes: ¹UnderEmo/GenEmo from Yan 2025. ²Speech generation via appended Kokoro TTS. ³Generation unavail. — audio decoder not released.)*

### Table 3 — Ablation (L575–585), 3 representative tasks (OpenbookQA Acc, EmphDet F1, GenEmo Score)
| Method | OpenbookQA Acc↑ | EmphDet F1↑ | GenEmo Score↑ |
|---|---|---|---|
| Original (λ=0.85) | 86.59 | 60.84 | 45.42 |
| No Thinking | 64.83 | 53.14 | 36.37 |
| No `[cap]` | 84.83 | 0.39 | 27.18 |
| No `[Text]` | 83.95 | 0.40 | 35.69 |
| No ∆θ_inst (replaced by in-context examples) | 71.68 | 40.38 | 15.34 |

### Table 4 — Text-oriented tasks across λ (L1043–1050; OpenBookQA, MMSU, MLCpro)
| λ | OpenBookQA | MMSU | MLCpro |
|---|---|---|---|
| 0.60 | 89.89 | 77.78 | 90.10 |
| 0.65 | 88.35 | 76.41 | 91.57 |
| 0.70 | 88.13 | 76.64 | 89.37 |
| 0.75 | 87.03 | 76.35 | 93.04 |
| 0.80 | 86.81 | 74.26 | 90.84 |

### Table 5 — Speech understanding + generation across λ (L1052–1061)
| λ | UnderEmo Acc↑ | EmphDet P↑ | EmphDet R↑ | EmphDet F1↑ | GenEmo Score↑ | EmphGen P↑ | EmphGen R↑ | EmphGen F1↑ |
|---|---|---|---|---|---|---|---|---|
| 0.80 | 62.04 | 49.51 | 63.90 | 55.79 | 37.91 | 22.15 | 35.13 | 27.17 |
| 0.85 | 50.75 | 55.11 | 67.89 | 60.84 | 45.42 | 26.16 | 40.17 | 31.68 |
| 0.90 | 50.99 | 59.25 | 70.93 | 64.57 | 41.95 | 27.23 | 39.36 | 32.19 |
| 0.95 | 37.76 | 52.69 | 70.93 | 60.47 | 43.50 | 28.07 | 39.20 | 32.72 |
| 1.00 | 29.87 | 52.69 | 70.93 | 60.47 | 43.22 | 26.81 | 34.64 | 30.23 |

### Table 6 — Removing format forcing (L1072–1077), thinking disabled
| Model | OpenBook Acc↑ | GenEmo Score↑ | EmphDet P↑ | EmphDet R↑ | EmphDet F1↑ |
|---|---|---|---|---|---|
| SpeechCombine (no think) | 64.83 | 36.37 | 42.90 | 69.78 | 53.14 |
| SpeechCombine (no format forcing) | 65.71 | 36.36 | 20.38 | 43.34 | 27.73 |

### Table 7 — Generalisation across text-model families (L1099–1105)
| Model | OpenBook Acc↑ | GenEmo Score↑ | EmphDet P↑ | EmphDet R↑ | EmphDet F1↑ |
|---|---|---|---|---|---|
| SpeechCombine-Qwen | 86.59 | 45.42 | 55.11 | 67.89 | 60.84 |
| SpeechCombine-OLMO (OLMO-3-7B-Think) | 72.52 | 21.35 | 28.18 | 37.88 | 32.32 |
| SpeechCombine-LLAMA (LLaMA-3.1-8B) | 62.85 | 32.34 | 22.79 | 59.92 | 33.02 |

---

## 5. Source-free reconciliation (Python-verified)

- **Table 1 "best or second-best across all datasets" (Group B, excl. GPT-4o):** VERIFIED — rank 1 on OpenbookQA (86.59) & MMSU (73.38); rank 2 on GSM8k (behind Kimi 95.53), Truthful (behind Fun 61.27), MLC (tied-Fun 93.97), MLCpro (behind Fun 93.40). ✓
- **Emphasis-detection SOTA margin (T2):** SpeechCombine F1 60.84 = **2.12×** the next-best (Fun 28.76); also column-max on EmphDet P (55.11) & R (67.89). ✓
- **Emotion-gen 2nd (T2b):** GenEmo 45.42 < GLM 48.13, > Fun 39.30. ✓ **Emphasis-gen best (T2b):** EmphGen F1 31.42 column-max Group B (next Qwen 24.24). ✓
- **Ablation prose (§4.6) vs Table 3:** "No ∆θ_inst (in-context) beats No-Thinking on text (71.68>64.83) but worse on speech" — EmphDet F1 40.38<53.14, GenEmo 15.34<36.37 ✓. "No `[cap]`/`[text]` devastate speech generation+understanding" — EmphDet F1 0.39/0.40 ✓.
- **F1 harmonic-mean recomputes from P/R in BOTH Table 2 and Table 5** (e.g. EmphDet `2·55.11·67.89/(55.11+67.89)=60.83≈60.84`; EmphGen Table-2 `2·25.91·39.90/(25.91+39.90)=31.41≈31.42`; Table-5 `2·26.16·40.17/66.33=31.69≈31.68`) → all F1 cells internally consistent.
- **Cross-table identity checks:** Table 3 "Original" row == Table 1/2 SpeechCombine row (86.59/60.84/45.42) ✓; Table 6 "no think" == Table 3 "No Thinking" (64.83/36.37/53.14) ✓; Table 7 Qwen row == Table 1/2 SpeechCombine ✓.
- **"~20×" token-inflation** (intro L78): 5 text tokens → 60–200 speech ⇒ 12× (low) / 26× (mid) / 40× (high); geometric mean ≈ 22×. ✓ order-of-magnitude, range is wide.

---

## 6. Honest-scope notes (⚠ inline caveats)

1. **GENUINE cross-table run-drift — Table 2 vs Table 5 λ=0.85 row** (iter-65 ReContext / iter-70 SUNTA / iter-74 PointDiT lone-cell-drift class): the λ=0.85 row of the λ-sweep (Table 5) is **the default config** (λ=0.85, §4.1), so it should byte-match the SpeechCombine row of Table 2 — but **4 cells differ** while 4 match exactly:
   - **UnderEmo Acc: 52.70 (T2) vs 50.75 (T5)** — Δ **1.95** (largest; ~3.7% relative).
   - **EmphGen P 25.91 vs 26.16 (Δ0.25); R 39.90 vs 40.17 (Δ0.27); F1 31.42 vs 31.68 (Δ0.26)**.
   - Cells that **do** match exactly: EmphDet P/R/F1 (55.11/67.89/60.84) and GenEmo Score (45.42).
   Both tables' F1 cells recompute correctly from their own P/R (harmonic mean), so each is internally consistent — the λ-sweep evidently **re-ran** the stochastic UnderEmo (LLM-as-judge) and EmphGen (emphasis-detector) tasks, giving slightly different numbers, while the more stable EmphDet and GenEmo reproduced. No headline impact (SpeechCombine's SOTA/2nd claims hold under either set), but the UnderEmo value one cites depends on which table — flag before echoing either.

2. **"<1% of training data" headline is borderline** (conclusion L616 + §4.1 L395): 30k / (100× = 3M) = **exactly 1.00%**. "<1%" holds only because the paper says Fun-Audio-Chat is "**over** 100× the size" (strictly >3M hours ⇒ strictly <1.0%); at the literal 100× point it is 1.0%, not <1%. Denominator-dependent — cite as "~1%" or quote the "100×" framing rather than the "<1%".

3. **SDQA text-vs-table coverage gap:** §4.2 (L406) states "For QA, we evaluate on OpenbookQA, **SDQA**, and MMSU in VoiceBench" — but Table 1(a) QA lists only **OpenbookQA, MMSU**; **SDQA is absent** from the table. Either SDQA was dropped or the prose over-lists; the QA-column count is 2, not 3.

4. **Format-forcing is load-bearing for emphasis detection (T6):** removing format forcing drops EmphDet F1 53.14→27.73 (Prec 42.90→20.38) — i.e. **half** the emphasis-detection SOTA headline (T2 F1 60.84) rests on the 4 inference-time logits processors, not on the trained weights. The "no format forcing" row is also run **without thinking**, so the format-forcing contribution is entangled with the thinking-mode contribution.

5. **External ASR dependency** (authors' own limitation): SpeechCombine relies on whisper-large-v3 to transcribe each speech query into the `[text]` anchor before prosody extraction — added latency + transcription errors flow into downstream metrics; ASR internalisation is future work. The emphasis-detection SOTA therefore partly inherits Whisper's transcription quality.

6. **Prosody-only tokenisation, no timbre/voice-quality/accent** (authors' own limitation, §5): the 5-value-per-word ProsodyLM scheme encodes only `f0_med/f0_range/f0_slope/Dur/energy`; speaker identity, voice quality, and accent are not represented — limits applicability to speaker-tracking / voice-conversion / accent tasks. Also the encoder+decoder are frozen third-party (ProsodyLM/StyleTTS-2-derived), so only the LLM is trained.

7. **"Long thinking for free" is template + 4 inference interventions, not free:** it requires forcing first token = `<think>`, appending "Okay", banning prosody tokens during think, and banning `</think>` until a 160-token minimum (App A.5). This is inference-time engineering, not a zero-cost emergent capability; the no-thinking ablation (T3) loses 21.8 OpenBookQA points (86.59→64.83), so "for free" means "no extra training", not "no extra mechanism".

8. **Single backbone family main (Qwen3-8B); OLMO/LLaMA generalisation (T7) bounded by weaker text models** (§C.3): OLMO GenEmo 21.35 flagged as a prosody-token-loss convergence issue; LLaMA EmphDet P 22.79 weak. Generalisation "successful but bounded" — only 2 non-Qwen families, both 7–8B, both weaker base LLMs, so the compositional transfer is not stress-tested against a stronger non-Qwen instruct model.

9. **λ-collapse claims off-table:** §C.1 states "λ below 0.6 → model doesn't recognise `[speech]` sections; λ above 0.85 → forgets text LLM knowledge" — but Table 4 only spans 0.60–0.80 (no 0.85 default row, no <0.6 collapse row) and Table 5 only 0.80–1.00 (no >0.85 text-forgetting shown). The bracketing collapse phenomena are asserted, not tabulated.

10. **No seeds/CIs/significance** on any of the 7 tables; several "best/2nd-best" gaps are small (MMSU 73.38 vs ASR-LLM 73.22 = 0.16; Truthful 60.09 vs Fun 61.27 — SpeechCombine actually 2nd by 1.18; MLC tied 93.97). UnderEmo run-drift (note 1) of 1.95 between tables shows the noise floor is at least that large on the LLM-as-judge task ⇒ sub-2-point "wins" are within run noise.

11. **Group-A "catastrophic forgetting milder than anticipated" (§4.2 L451)** is confounded by the **small** 10k-hour SFT addition — the paper itself attributes the mildness to dataset size, so the Group-A contrast under-sells the forgetting that a full-scale SFT would cause; SpeechCombine's advantage over Group A is thus partly a function of Group A's deliberately-limited extra training.

---

## 7. Strengths / limitations / verdict

**Strengths**
- Strikingly simple recipe (one pre-training round, 30k h, two weight differences added) that nonetheless yields SOTA on emphasis detection (2.12× next-best F1), 2nd on emphasis + emotion generation, and best/2nd on all 6 text-oriented tasks — empirically validating **weight-space compositionality across modalities**.
- Clean falsifiable hinges: removing ∆θ_inst (T3) collapses speech tasks; removing ∆θ_speech ⇒ ASR+TEXT-LLM; λ-sweep (T4/T5) shows the text↔speech trade-off the combination-weight balances.

**Limitations** (honest-scope notes above): prosody-only tokens (no timbre/accent), external ASR, format-forcing-dependent, Qwen-main with weaker-family generalisation, run-drift between T2 and T5 λ=0.85 rows, no significance tests.

**Verdict:** A clean, well-motivated demonstration that **instruction-following capability is a transferable weight-direction** (∆θ_inst) that composes with a modality-direction (∆θ_speech) — extending Task Arithmetic / Chat Vector into the speech-modality setting and sidestepping the speech data-inflation bottleneck. The compositional-powers framing is the citable contribution; the emphasis-detection SOTA is the strongest empirical cell. Cite λ=0.85 UnderEmo/EmphGen numbers from Table 2 (main) and note the Table-5 λ-sweep re-run drift.
