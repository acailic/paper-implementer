# Breakdown — The Verification Horizon: No Silver Bullet for Coding Agent Rewards

> **Paper:** The Verification Horizon: No Silver Bullet for Coding Agent Rewards
> **Authors:** Binghai Wang, Chenlong Zhang, Dayiheng Liu, Jiajun Zhang, Jiawei Chen, Mouxiang Chen, Rongyao Fang, Siyuan Zhang, Xuwu Wang, Yuheng Jing, Zeyao Ma, Zeyu Cui (Qwen Team)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2606.26300

---

## 1. Problem & Motivation

- **Problem:** As coding agents improve, generating solutions becomes easier — but reliably verifying those solutions becomes harder. Every verifier is only a proxy for human intent, never the intent itself.

- **Why it matters:** Reward hacking is not a bug but an inevitable consequence of optimizing toward an imperfect objective function. When a proxy is placed under optimization pressure, the generator learns not only to satisfy the proxy but to exploit the divergence between the proxy and the true intent.

- **Previous approaches:**
  - Unit tests: reliable but cover a thin layer of intent
  - LLM judges: scalable and faithful but vulnerable to exploitation
  - Human review: most trustworthy but doesn't scale

## 2. Key Insight / Contribution

- **Central thesis:** No fixed reward can remain effective as the policy model improves; verification must co-evolve with the generator.

- **Three dimensions of verification quality:**
  | Dimension | Question | Challenge |
  |-----------|---------|-----------|
  | Scalability | Can it be cheaply produced at scale? | Rich signals (humans) are expensive |
  | Faithfulness | How well does it reflect true intent? | Intent is inherently underspecified |
  | Robustness | Does it resist optimization pressure? | Stronger models find new loopholes |

- **Four reward constructions** for different task types, each with increasing faithfulness but less mechanical verification.

## 3. Method

### 3.1 Overview — Four Reward Constructions

```mermaid
graph TB
    subgraph SWE["§2 — SWE Tasks"]
        A1[Test-driven rewards<br/>binary pass/fail] --> A2[Quality Filter<br/>Agentic Judge]
        A2 --> A3[Behavior Monitor<br/>token-level penalty]
    end

    subgraph FE["§3 — Frontend Tasks"]
        B1[Rubric Static Judge<br/>6 dimensions, ~26 items] --> B2[Interactive Judge<br/>3-stage agentic evaluation]
    end

    subgraph UF["§4 — User Feedback"]
        C1[Feedback Annotation<br/>125K trajectories → 536K rounds] --> C2[RW-SFT<br/>reweight negative spans]
        C1 --> C3[Span-KTO<br/>span-level preference learning]
    end

    subgraph LH["§5 — Long-horizon Generation"]
        D1[Agent Evaluator<br/>checklist + holistic score] --> D2[RFT Filtering<br/>Seval ≥ 8]
    end

    SWE -.->|"Scalability ↑<br/>Faithfulness ↓"| FE
    FE -.->|"Faithfulness ↑<br/>Scalability ↓"| UF
    UF -.->|"Robustness focus"| LH

    style SWE fill:#e1f5fe
    style FE fill:#f3e5f5
    style UF fill:#e8f5e9
    style LH fill:#fff3e0
```

### 3.2 §2: Test-driven Rewards for SWE Tasks

```mermaid
flowchart TD
    subgraph Pipeline["SWE-Universe Data Pipeline"]
        G[GitHub PR] --> FP[fix patch + test patch]
        FP --> D[Docker environment]
        D --> E[evaluation.sh<br/>binary pass/fail reward]
    end

    subgraph Filter["Quality Filter"]
        QF[MiniSWEAgent explores repo]
        QF --> IC{instruct_clear?<br/>instruction clarity}
        QF --> IT{instruct_ut_align?<br/>tests cover instruction}
        IC --> LG[overall_good label]
        IT --> LG
        LG --> FO[Filter out low-quality tasks]
    end

    subgraph Monitor["Behavior Monitor during RL"]
        LOG[Log: commands, network, git, files, patch]
        LOG --> PS[Pattern set P<br/>evidence → risk → penalty]
        PS --> REV[Agentic reviewer<br/>sampled trajectories]
        REV --> |new patterns| PS
        PS --> PEN[Token-level penalty<br/>for shortcut trajectories]
    end

    Pipeline --> Filter
    Filter --> Monitor
```

**Hacking-susceptible behaviors analysis:**

| Category | Behavior | Freq. (%) | Resolved (% Δ<sub>base</sub>) | φ corr. |
|----------|----------|-----------|------------------------------|---------|
| **Static-environment leakage** | Repository-history mining | 21.11 | 56.55 (−3.44) | −0.036 |
| | Test-oracle tampering | 3.69 | 47.29 (−12.70) | −0.051 |
| | Evaluation-harness tampering | 8.25 | 41.47 (−18.52) | −0.113 |
| | Visible-test overfitting | 30.00 | 51.62 (−8.37) | −0.112 |
| | Evaluator-aware patching | 8.78 | 56.39 (−3.60) | −0.023 |
| **Policy-dependent shortcut** | Solution artifact retrieval | 4.32 | **72.34 (+12.35)** | +0.054 |
| | External fix lookup | 7.03 | **61.69 (+1.70)** | +0.010 |

> **Key finding:** After hardening the environment, static leakage behaviors have *negative* correlation with success, but policy-dependent shortcuts (especially artifact retrieval at 4.32% freq. → 72.34% resolved) remain dangerously correlated. This motivates the behavior monitor.

### 3.3 §3: Interactive Judge for Frontend Tasks

```mermaid
flowchart LR
    subgraph Static["Static Rubric Judge"]
        IN[Input: screenshot + source + prompt] --> RUB[6 dimensions<br/>Functional 37.7% · Content 19.0%<br/>Visual 13.3% · Layout 12.9%<br/>UX 9.3% · Technical 7.2%]
        RUB --> AVG[~25.9 checklist items/task]
        AVG --> OUT[Alignment: Spearman ρ=0.905, τ=0.786]
    end

    subgraph Agentic["Agentic Interactive Judge"]
        S1[Stage 1: Action Planner<br/>task + a11y tree + browser state<br/>→ complete action list] --> S2[Stage 2: Playwright Server<br/>click · scroll · navigate · fill<br/>hover · press → screenshots + DOM + console]
        S2 --> S3[Stage 3: Judge Model<br/>interaction trace + source + rubric<br/>→ score runtime-based, not code-inspection]
    end

    Static -->|"Vulnerable to<br/>length exploitation"| Agentic
```

**Rubric judge reliability (Table 4 from paper):**

| Scorer | Prompt | Spearman ρ | Kendall τ | Battle Agreement | Cross-Judge τ |
|--------|--------|-----------|-----------|-----------------|--------------|
| Qwen3.7-Plus | Default | 0.810 | 0.714 | 40.4% | ≥ 0.93 |
| Qwen3.7-Plus | Strict | 0.810 | 0.714 | 41.4% | ≥ 0.93 |
| Qwen3.6-Max | Default | **0.905** | **0.786** | 34.2% | ≥ 0.93 |
| Qwen3.6-Max | Strict | **0.905** | **0.786** | 36.1% | ≥ 0.93 |

> Within each scorer family Kendall τ = 1.0; across families τ ≥ 0.93. Rubric design is robust to configuration.

### 3.4 §4: User Feedback → Span-KTO

```mermaid
flowchart TD
    subgraph Annotation["Feedback Annotation Pipeline"]
        DATA[Real user–agent interactions<br/>125,528 trajectories] --> LLM[LLM-as-Judge Qwen-Plus<br/>per-round annotation]
        LLM --> ANNOT[polarity · confidence · signal_type<br/>negative_reason · user_fairness · reasoning]
        ANNOT --> STATS[535,737 round annotations]
    end

    subgraph SpanKTO["Span-KTO Training"]
        S1[Partition tokens into<br/>contiguous spans by polarity] --> S2[Compute span reward<br/>r_θ = Σ log-likelihood ratios]
        S2 --> S3[Reference point<br/>z_ref ← EMA over batch rewards]
        S3 --> S4[Preference loss per span<br/>positive: −λ_w · σ(β·a_k)<br/>negative: −λ_l · σ(−β·a_k)]
        S1 --> S5[Neutral regularization<br/>standard CE loss]
        S4 --> TOTAL[L_Span-KTO = L_pref + L_neutral]
        S5 --> TOTAL
    end

    Annotation --> SpanKTO
```

### 3.5 §5: Agent Evaluator for Long-horizon Generation

```mermaid
flowchart LR
    SPEC[Task Spec T] --> G[Generator G<br/>produces repo G_T]
    G --> E[Evaluator E<br/>decomposes T → checklist C]
    E --> S1[S_pass = checklist pass rate]
    E --> S2[S_eval = holistic quality score]
    GT[Original test suite<br/>from source repo] --> S_UT[S_UT = unit-test score]
    S1 -.-> CORR[Alignment metrics<br/>BoN-Acc · Regret · Kendall τ<br/>Pearson r · S̄_UT θ]
    S2 -.-> CORR
    S_UT -.-> CORR
    CORR --> FILTER[RFT filtering<br/>S_eval ≥ 8]
```

## 4. Math

### §4 Span-KTO Loss

**Span-Level Implicit Reward.** For each span $S_k = (s_k, e_k)$ bounded by human-annotated polarity transitions, the implicit reward is the sum of per-token log-likelihood ratios between policy and reference models:

$$r_\theta(x, S_k) = \sum_{t=s_k}^{e_k} \left[ \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right] \tag{4}$$

Each span serves as an independent reward judgment unit. This is formally identical to the sequence-level log-likelihood ratio in response-level KTO, but applied at the granularity of polarity-delineated spans within a trajectory.

**Reference Point Estimation.** The reference point is estimated online via exponential moving average over all span rewards in each batch:

$$z_{\text{ref}} \leftarrow \alpha \cdot z_{\text{ref}} + (1 - \alpha) \cdot \bar{r}_{\text{batch}} \tag{5}$$

where $\bar{r}_{\text{batch}} = \mathbb{E}_{S_k \in S_{\text{batch}}}[r_\theta(x, S_k)]$ and $\alpha$ is the EMA decay coefficient.

**Span-Level Preference Loss.** The advantage for each span is $a_k = r_\theta(x, S_k) - z_{\text{ref}}$, with asymmetric loss for positive vs. negative spans:

$$\ell(S_k) = \begin{cases} -\lambda_w \cdot \sigma(\beta \cdot a_k) & \text{if } p_{S_k} = \text{positive} \\[6pt] -\lambda_l \cdot \sigma(-\beta \cdot a_k) & \text{if } p_{S_k} = \text{negative} \end{cases} \tag{6}$$

where $\sigma$ is the sigmoid function, $\beta > 0$ controls preference strength, and $\lambda_w, \lambda_l$ are loss coefficients for positive/negative spans.

**Total Preference Loss:**

$$\mathcal{L}_{\text{pref}}(\theta) = \mathbb{E}_{S_k}[\ell(S_k)] \tag{7}$$

**Neutral Token Regularization.** Neutral tokens ($p_t = \text{neutral}$) carry no preference signal but contain valuable language modeling information:

$$\mathcal{L}_{\text{neutral}}(\theta) = -\mathbb{E}_{t \in T_{\text{neu}}} \left[ \log \pi_\theta(y_t \mid x, y_{<t}) \right] \tag{8}$$

where $T_{\text{neu}} = \{t : p_t = \text{neutral}\}$.

**Overall Objective:**

$$\mathcal{L}_{\text{Span-KTO}}(\theta) = \mathcal{L}_{\text{pref}}(\theta) + \mathcal{L}_{\text{neutral}}(\theta) \tag{9}$$

**Hyperparameters:** $\beta = 0.01$ (optimal — too strong → unstable, too weak → weak signal), $\lambda_l = 1.0$ (model learns well from negative spans).

### §2 Behavior Monitor

**Formal Definition.** For each rollout trajectory $\tau$, the monitor:

1. **Logs observable signals:** $\mathcal{O}(\tau) = \{ \text{commands}, \text{network accesses}, \text{git ops}, \text{files opened/edited}, \text{final patch} \}$

2. **Pattern set $\mathcal{P}$:** Each pattern $p \in \mathcal{P}$ is a tuple $(e_p, r_p, i_p)$:
   - $e_p$: observable trajectory evidence (e.g., searching for PR, querying commit hashes, accessing GitHub pages revealing merged patches)
   - $r_p$: associated leakage risk assessment
   - $i_p$: corresponding intervention type

3. **Token-level penalty:**
$$R(\tau) = R_{\text{verifier}}(\tau) - \gamma \cdot \sum_{p \in \mathcal{P}} \mathbb{1}[e_p \subseteq \mathcal{O}(\tau)] \cdot |\tau|$$

where $\gamma$ is the penalty coefficient and $|\tau|$ is the trajectory length in tokens.

4. **Iterative pattern update:** After each training interval:
   - Sample trajectories prioritizing those that pass the verifier or trigger the monitor
   - Agentic reviewer inspects for newly emerging shortcut strategies
   - Recurring patterns added to $\mathcal{P}$
   - Updated monitor deployed in next RL round

> This closed-loop design is critical because reward hacking is policy-dependent: as the model improves, it discovers new exploitation channels absent in the initial review.

### §5 Agent Evaluator Metrics

**Best-of-N Accuracy:**

$$\text{BoN-Acc} = \frac{1}{M} \sum_{j=1}^{M} \mathbb{1}\!\left[k^* = \arg\max_k S^{(j,k)}_{\text{UT}}\right]$$

**Per-task Regret:**

$$\text{Regret}_j = \max_k S^{(j,k)}_{\text{UT}} - S^{(j,k^*)}_{\text{UT}}, \qquad \overline{\text{Regret}} = \frac{1}{M}\sum_{j=1}^{M} \text{Regret}_j$$

**Threshold-Conditioned UT Score:**

$$\bar{S}_{\text{UT}}(\theta) = \frac{1}{|A_\theta|} \sum_{(j,k) \in A_\theta} S^{(j,k)}_{\text{UT}}, \quad A_\theta = \{(j,k) : S^{(j,k)}_{\text{eval}} \geq \theta\}$$

A faithful evaluator yields monotonically increasing $\bar{S}_{\text{UT}}(\theta)$ as $\theta$ rises.

## 5. Training

### §2 RL with Behavior Monitoring

| Parameter | Value |
|-----------|-------|
| Model | Qwen-Turbo (internal checkpoint) |
| Data | SWE-Universe (filtered) |
| Monitor | Token-level penalty for shortcut patterns |
| Pattern update | Iterative: agentic reviewer after each training interval |
| Benchmarks | SWE-Bench Verified, Multilingual, Pro |
| Metrics | Resolved, Clean Resolved, Hack Rate, Hacked Resolved |

### §4 Span-KTO

| Parameter | Value |
|-----------|-------|
| $\beta$ | 0.01 |
| $\lambda_l$ | 1.0 |
| $\lambda_w$ | balanced (default) |
| Reference point | Exponential moving average (online) |
| Training data | 125,528 trajectories, 535,737 round annotations |
| Feedback source | Real user–agent interactions (senior SWEs) |

### §5 RFT

| Parameter | Value |
|-----------|-------|
| Base model | Qwen 3.6 Turbo |
| Filter threshold | $S_{\text{eval}} \geq 8$ |
| Batch size | 128 |
| Checkpoints | Every 150 steps |
| Max epochs | 6 |
| Benchmark | OpenHands scaffold (anti-hacking) |

## 6. Results

### §2: SWE Tasks — Behavior Monitoring (Table 3)

| Benchmark | Clean Resolved (%) ↑ || Hack Rate (%) ↓ || Hacked Resolved (%) ↓ |
|-----------|:-:-:|:-:-:|:-:-:|:-:-:|:-:-:|:-:-:|:-:-:|:-:-:|:-:-:|
| | Base | +Mon. | Δ | Base | +Mon. | Δ | Base | +Mon. | Δ |
| SWE-Bench Verified | 36.49 | **64.98** | **+28.50** | 51.49 | **2.13** | **−49.35** | 41.35 | **0.70** | **−40.65** |
| SWE-Bench Multilingual | 50.73 | **66.33** | **+15.60** | 31.19 | **1.59** | **−29.61** | 23.76 | **0.84** | **−22.93** |
| SWE-Bench Pro | 33.43 | **50.27** | **+16.84** | 30.60 | **0.20** | **−30.40** | 20.61 | **0.13** | **−20.47** |
| **Average** | **40.22** | **60.53** | **+20.31** | **37.76** | **1.31** | **−36.45** | **28.57** | **0.56** | **−28.02** |

> **Key result:** The monitor reduces average hacked-resolved from 28.57% → 0.56% while improving clean resolved from 40.22% → 60.53%. The gain is not just more verifier passes, but a shift from shortcut-dependent to monitor-clean success.

### §3: Frontend — Interactive Judge RFT

| Setting | WebDev Human Eval | QwenWebBench |
|---------|:-:|:-:|
| Qwen-Plus (base) | 78 | 1509 |
| + Interactive Judge RFT | **84 (+6)** | **1545 (+36)** |

### §4: Span-KTO vs Baselines (Figure 10)

| Benchmark | SFT | RW-SFT | Span-KTO | Δ vs SFT | Δ vs RW-SFT |
|-----------|:-:|:-:|:-:|:-:|:-:|
| SWE-bench Verified | 54.2% | 55.2% | **59.8%** | +5.6pp | +4.6pp |
| SWE-bench Pro | 33.4% | 36.5% | **38.1%** | +4.7pp | +1.6pp |
| SWE-bench Multilingual | 37.7% | 41.2% | **45.5%** | **+7.8pp** | +4.3pp |
| Aone-bench | 14.8% | 25.0% | **28.1%** | **+13.3pp** | +3.1pp |
| Octo-bench | 62.3% | 67.0% | **67.4%** | +5.1pp | +0.4pp |

> Span-KTO dominates on all 5 benchmarks. Biggest gains on Aone-bench (+13.3pp) and SWE-bench Multilingual (+7.8pp). Gap is smallest on OctoBench (62.3→67.4 spread; the paper notes it emphasizes scaffold-following rather than code repair).
>
> *Values read from the Figure 10 bars; the Multilingual column (SFT 37.7 / RW-SFT 41.2 / Span-KTO 45.5) is corroborated by Table 21's β=0.01 ablation row (Multilingual = 45.55). The body text only states the +7.8pp Multilingual gain, not the absolute values, so the bar heights are the authoritative source.*

#### §4: Negative Behavior Correction (Figure 11)

Behavioral category analysis on SWE-bench Verified (Agent-as-Judge, 6 dimensions, score 0–4):

| Behavior | SFT (Resolved) | Span-KTO (Resolved) | Δ | SFT (Unresolved) | Span-KTO (Unresolved) | Δ |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|
| Execution Error | 3.63 | 3.74 | +2.9% | 1.61 | 1.84 | +13.9% |
| Misunderstand | 3.78 | 3.80 | +0.5% | 2.33 | 2.44 | +4.8% |
| Omission | 3.71 | 3.75 | +1.1% | 1.89 | 2.01 | +6.1% |
| Overaction | 3.65 | 3.70 | +1.2% | 2.82 | 3.00 | +6.5% |
| Inefficiency | 2.51 | 2.68 | +6.8% | 1.07 | 1.44 | **+34.5%** |
| Communication | 3.37 | 3.48 | +3.3% | 2.13 | 2.70 | **+26.5%** |

> **Key insight:** Improvements on resolved instances are modest (already high quality), but *unresolved* instances show dramatic gains — especially Inefficiency (+34.5%) and Communication (+26.5%). Span-KTO's value lies not only in "solving more problems" but in "behaving more reasonably when failing." This is critical for real deployment.

### §5: RFT with Evaluator Filtering

| Training Data | Size | Best Score | Steps |
|---------------|-----:|----------:|------:|
| Base model (before training) | — | 11.41 | — |
| Random sample | 9,139 | 21.61 | — |
| Evaluator-filtered ($S_{\text{eval}} \geq 8$) | 9,139 | **23.52** | +1.91 vs random |
| All rule-based filtered | 19,050 | **24.75** | 600 steps |

## 7. Ablations

### §4.1: RW-SFT Sensitivity (Figure 9)

Effect of negative span weight $w_{\text{neg}}$ on performance (averaged over 3 SWE-bench benchmarks):

| $w_{\text{neg}}$ | Avg Score (%) | Note |
|:---:|:-:|---|
| 0.0 (discard negatives) | 37.2% | Below baseline |
| 0.5 | 35.1% | Below baseline |
| 0.6 | 40.7% | Slightly below |
| **0.8** | **44.4%** | **Best RW-SFT config** |
| 1.0 (SFT baseline) | 41.8% | No reweighting |
| Drop-fail-abandon | 42.1% | Discard entire negative trajectories |

> **Takeaway:** Only slight downweighting ($w_{\text{neg}} = 0.8$) helps; aggressive removal hurts. Negative spans contain valuable language modeling information. Reweighting can only adjust learning intensity, not learning direction — this motivates the preference learning approach of Span-KTO.

### §5: Evaluator Prompt Versions (Qwen-Plus)

| Prompt | BoN-Acc | Kendall τ | Pearson $r_{\text{eval}}$ |
|--------|:-:|:-:|:-:|
| v1 (baseline) | 57.9% | 0.379 | 0.489 |
| v2 (+end-to-end examples) | 63.9% | 0.420 | 0.525 |
| v3 (+role fix) | 62.4% | 0.440 | 0.556 |
| **v4 (+context enrichment)** | **67.4%** | **0.473** | **0.598** |
| v5 (over-specified) | 59.6% | 0.471 | 0.541 |

> **Takeaway:** v4 is optimal. More detail is not always better — over-specification (v5) actually hurts BoN-Acc. The evaluator's prompt design requires careful calibration.

### §5: Why $S_{\text{eval}}$ over $S_{\text{pass}}$?

The paper consistently finds $r_{\text{eval}} \gg r_{\text{pass}}$ and $\rho_{\text{eval}} \gg \rho_{\text{pass}}$ — the holistic evaluation score $S_{\text{eval}}$ correlates far better with unit-test truth than the simple checklist pass rate. A uniform average over binary checklist outcomes doesn't capture item importance or overall code quality.

## 8. Code / Reproducibility

- No open-source code
- All experiments use internal Qwen checkpoints
- Evaluator prompt versions are described but not published
- Dataset of user interactions is internal (company interactions)
- SWE-Universe pipeline is public (arXiv:2602.02361)
- Feedback annotation schema is described in detail (§4.1)
