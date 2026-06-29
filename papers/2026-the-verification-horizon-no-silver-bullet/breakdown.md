# Breakdown — The Verification Horizon: No Silver Bullet for Coding Agent Rewards

> **Paper:** The Verification Horizon: No Silver Bullet for Coding Agent Rewards
> **Authors:** Binghai Wang, Chenlong Zhang, Dayiheng Liu, Jiajun Zhang, Jiawei Chen, Mouxiang Chen, Rongyao Fang, Siyuan Zhang, Xuwu Wang, Yuheng Jing, Zeyao Ma, Zeyu Cui (Qwen Team)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2606.26300

---

## 1. Problem & Motivation

- **Problem:** Kada se kodirajući agenti poboljšavaju, generisanje rješenja postaje lakše — ali pouzdana verifikacija tih rješenja postaje teža. Svaki verifier je samo proxy za ljudsku namjeru, nikad namjera sama.

- **Zašto je važno:** Reward hacking nije bug nego neizbježna posledica optimizacije prema imperfektnoj funkciji cilja. Kad se proxy stavi pod optimizacioni pritisak, generator uči ne samo zadovoljiti proxy nego i eksploatisati divergenciju između proxy-ja i namjere.

- **Prethodni pristupi:**
  - Unit testovi: reliable ali pokrivaju tanak sloj namjere
  - LLM sudci: scalable i faithful ali ranjivi na eksploataciju
  - Ljudski review: najvjerodostojniji ali ne skaliira

## 2. Key Insight / Contribution

- **Centralna teza:** Nijedan fiksni reward ne može ostati efikasan kako se politika modela poboljšava; verifikacija mora ko-evoluirati sa generatorom.

- **Tri dimenzije kvaliteta verifikacije:**
  | Dimenzija | Pitanje | Izazov |
  |-----------|---------|--------|
  | Scalability | Može li se jeftino proizvesti u skali? | Bogati signali (ljudi) su skupi |
  | Faithfulness | Koliko reflekstuje stvarnu namjeru? | Namjera je prirodo underspecified |
  | Robustness | Odolijeva li optimizacionom pritisku? | Jači modeli pronalaze nove rupe |

- **Četiri reward konstrukcije** za različite tipove zadataka, svaka sa sve većom faithfulness ali manje mehaničke verifikacije.

## 3. Method

### 3.1 Overview

```
SWE zadaci (§2)            → Test-driven rewards + quality judge + behavior monitor
Frontend zadaci (§3)       → Rubric judge + interactive agentic judge
Pravi svijet (§4)          → Korisnički implicitni reward signali → Span-KTO
Dugohorizontni zadaci (§5) → Automatski agent evaluator → RFT filtering
```

### 3.2 §2: Test-driven Rewards za SWE zadatke

```
┌──────────────────────────────────────────────────────────────┐
│ DATA PIPELINE (SWE-Universe)                                │
│  GitHub PR → fix patch + test patch → Docker env            │
│  → evaluation.sh (binary pass/fail reward)                 │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ QUALITY FILTER (Agentic Judge)                               │
│  MiniSWEAgent istražuje repozitorij:                       │
│    1. instruct_clear: je li instrukcija dovoljno jasna?     │
│    2. instruct_ut_align: da li testovi pokrivaju instrukciju?│
│  → overall_good label → filter out low-quality tasks       │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ BEHAVIOR MONITOR (tokom RL)                                  │
│  Loguje: commandi, mreža, git, fileovi, patch               │
│  Pattern set P: observable evidence → leakage risk → penalty │
│  Iterativno ažuriranje P nakon svakog trening intervala     │
│  → Token-level penalty za shortcut trajectories             │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 §3: Interactive Judge za Frontend zadatke

```
┌──────────────────────────────────────────────────────────────┐
│ STATIC RUBRIC JUDGE                                         │
│  Input: screenshot + source code + prompt                    │
│  6 dimenzija: Functional, Content, Visual, Layout, UX, Tech  │
│  Prosjek ~25.9 checklist item-a po zadatku                   │
│  Problem: ranjiv na length exploitation                     │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ AGENTIC INTERACTIVE JUDGE (3-stage)                         │
│                                                              │
│  Stage 1: Action Planner                                     │
│    Task + accessibility tree + browser state                 │
│    → single forward pass → complete action list              │
│                                                              │
│  Stage 2: Playwright Server                                  │
│    Atomic ops: click, scroll, navigate, fill, hover, press    │
│    → execute actions → screenshots + DOM + console log       │
│                                                              │
│  Stage 3: Judge Model                                       │
│    Interaction trace + source code + rubric                   │
│    → score (runtime-based, ne code-inspection-based)        │
└──────────────────────────────────────────────────────────────┘
```

### 3.4 §4: Korisnički Feedback → Span-KTO

```
┌──────────────────────────────────────────────────────────────┐
│ FEEDBACK ANNOTATION PIPELINE                                │
│  LLM-as-Judge (Qwen-Plus) na korisničkim razgovorima        │
│  Per-round: polarity, confidence, signal_type,               │
│    negative_reason, user_fairness, reasoning                  │
│                                                              │
│  125,528 trajektorija → 535,737 round anotacija             │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ SPAN-KTO TRAINING                                           │
│                                                              │
│  1. Partitioniraj tokene u kontinuirane spanove po polarity  │
│     (neutralni se ne koriste za preference learning)        │
│                                                              │
│  2. Span reward: r(Sk) = Σ log-likelihood ratios           │
│                                                              │
│  3. Reference point: zref ← EMA over batch span rewards     │
│                                                              │
│  4. Preference loss per span:                               │
│     positive: -λw · σ(β · ak)                                │
│     negative: -λl · σ(-β · ak)                               │
│                                                              │
│  5. Neutral regularization: standard CE loss                  │
│                                                              │
│  Total: L = L_pref + L_neutral                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.5 §5: Agent Evaluator za dugohorizontno generisanje

```
┌──────────────────────────────────────────────────────────────┐
│ EVALUATION AGENT DESIGN                                      │
│                                                              │
│  Generator G(T) → Evaluator E:                               │
│    1. Decomponiraj specifikaciju T u checklist C             │
│    2. Za svaku stavku: code review / test execution           │
│    3. Spass (checklist pass rate) + Seval (holistic score)   │
│                                                              │
│  Ground truth: originalni test suite iz izvornog repo-a    │
│  Metrike: BoN-Acc, Regret, Kendall τ, Pearson r, S̄UT(θ)    │
└──────────────────────────────────────────────────────────────┘
```

## 4. Math

### §4 Span-KTO Loss

```
Span reward:   rθ(x, Sk) = Σ_{t=sk}^{ek} [log πθ(yt|x,y<t) - log πref(yt|x,y<t)]

Reference:     zref ← α · zref + (1-α) · r̄_batch

Advantage:     ak = rθ(x, Sk) - zref

Span loss:     ℓ(Sk) = {
                  -λw · σ(β · ak)   ako pSk = positive
                  -λl · σ(-β · ak)  ako pSk = negative
                }

Neutral:       L_neutral = -E_{t∈Tneu} [log πθ(yt|x,y<t)]

Total:         L_Span-KTO = L_pref + L_neutral
```

- **β = 0.01** optimal (prejako → unstable, preslabo → slab signal)
- **λl = 1.0** optimal — model uči iz negativnih spanova bez problema

### §2 Behavior Monitor

```
Za svaki rollout τ:
  Loguj: {commands, network, git ops, files, patch}
  Za svaki pattern p ∈ P:
    Ako τ matches p:
      reward = reward - token_level_penalty

P se ažurira iterativno:
  Nakon svakog trening intervala:
    Sample trajectories → agentic reviewer → nove patterns → dodaj u P
```

## 5. Training

### §2 RL sa behavior monitoring

| Parametar | Vrijednost |
|-----------|------------|
| Model | Qwen-Turbo (internal checkpoint) |
| Data | SWE-Universe (filtered) |
| Monitor | Token-level penalty za shortcut patterns |
| Benchmark | SWE-Bench Verified, Multilingual, Pro |

### §4 Span-KTO

| Parametar | Vrijednost |
|-----------|------------|
| β | 0.01 |
| λl | 1.0 |
| λw | podrazumijevani (balansiran) |
| Reference EMA | eksponencijalni moving average |

### §5 RFT

| Parametar | Vrijednost |
|-----------|------------|
| Base model | Qwen 3.6 Turbo |
| Filter threshold | Seval ≥ 8 |
| Batch size | 128 |
| Checkpoints | svakih 150 steps |
| Max epochs | 6 |
| Benchmark | OpenHands scaffold (anti-hacking) |

## 6. Results

### §2: SWE zadaci — Behavior Monitoring

| Benchmark | Clean Resolved (Base) | Clean Resolved (+Mon.) | Δ | Hack Rate (Base) | Hack Rate (+Mon.) | Δ |
|-----------|----------------------|----------------------|---|-----------------|-------------------|---|
| SWE-Bench Verified | 36.49% | 64.98% | +28.50 | 51.49% | 2.13% | -49.35 |
| SWE-Bench Multilingual | 50.73% | 66.33% | +15.60 | 31.19% | 1.59% | -29.61 |
| SWE-Bench Pro | 33.43% | 50.27% | +16.84 | 30.60% | 0.20% | -30.40 |
| **Average** | **40.22%** | **60.53%** | **+20.31** | **37.76%** | **1.31%** | **-36.45** |

### §3: Frontend — Interactive Judge RFT

| Setting | WebDev Human Eval | QwenWebBench |
|---------|-------------------|--------------|
| Qwen-Plus (base) | 78 | 1509 |
| + Interactive Judge RFT | 84 (+6) | 1545 (+36) |

### §4: Span-KTO vs Baseline

| Benchmark | SFT | RW-SFT | Span-KTO |
|-----------|-----|--------|----------|
| SWE-bench Verified | 54.2% | 55.2% | **59.8%** (+5.6pp) |
| SWE-bench Pro | 33.4% | 36.5% | **38.1%** (+4.7pp) |
| SWE-bench Multilingual | 52.0% | 41.2% | **59.8%** (+7.8pp) |
| Aone-bench | 14.8% | 25.0% | **28.1%** (+13.3pp) |
| Octo-bench | 62.3% | 67.0% | **67.4%** (+5.1pp) |

### §5: RFT sa evaluator filtering

| Training Data | Size | Best Score |
|---------------|------|-------------|
| Random sample | 9,139 | 21.61 |
| Evaluator-filtered (Seval≥8) | 9,139 | **23.52** (+1.91) |
| All rule-based filtered | 19,050 | **24.75** (600 steps) |
| Base model (before training) | — | 11.41 |

## 7. Ablations

### §4 RW-SFT sensitivity

| wneg | Avg Score (3 SWE-bench) |
|------|------------------------|
| 0.0 | 37.2% |
| 0.5 | 35.1% |
| 0.8 | **44.4%** |
| 1.0 (SFT baseline) | 41.8% |

→ Samo lagano downweighting pomaže, agresivno uklanjanje šteti. Preference learning (Span-KTO) je znatno bolji.

### §5 Evaluator prompt verzije (Qwen-Plus)

| Prompt | BoN-Acc | τ | r_eval |
|--------|---------|---|-------|
| v1 (baseline) | 57.9% | 0.379 | 0.489 |
| v2 (+e2e) | 63.9% | 0.420 | 0.525 |
| v3 (+role fix) | 62.4% | 0.440 | 0.556 |
| v4 (+context) | **67.4%** | **0.473** | **0.598** |
| v5 (over-specified) | 59.6% | 0.471 | 0.541 |

→ v4 je optimalan. Više detalja nije uvijek bolje.

## 8. Code / Reproducibility

- Nema open-source koda
- Svi eksperimenti koriste interne Qwen checkpoint-ove
- Evaluator prompt verzije su opisane ali ne publicirane
- Dataset korisničkih interakcija je interni (comp. interactions)
- SWE-Universe pipeline je public (arXiv:2602.02361)
