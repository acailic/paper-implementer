# Breakdown — OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning

> **Paper:** OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning
> **Authors:** Shuo Yang, Jinyang Wu, Zhengxi Lu, Yuhao Shen, Fan Zhang, Lang Feng, Shuai Zhang, Haoran Luo, Zheng Lian, Zhengqi Wen, Jianhua Tao
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2606.26790
> **Code (official):** https://github.com/jinyangwu/OPID

---

## 1. Problem & Motivation

- **Problem:** Outcome-based RL (GRPO) for agentic LLMs provides only sparse
  trajectory-level rewards. A terminal reward tells you whether a trajectory
  succeeded but not *which intermediate decisions* caused success or failure.
  This is especially bad in long-horizon tasks where a single early mistake
  derails the entire episode.

- **Why important:** Agentic LLMs are increasingly used for embodied tasks,
  web navigation, and search-based reasoning — all multi-step settings with
  delayed feedback. Without fine-grained credit assignment, RL optimization
  wastes samples on high-variance gradient estimates.

- **Prior approaches and limitations:**
  - Skill-GRPO: uses external skills during training but drops them at test
    → huge train-test mismatch, performance collapses without skills
  - OPSD / Skill-SD / RLSD / SDAR: introduce self-distillation or
    skill-conditioned distillation, but rely on external skill libraries,
    retrieved skill files, or maintained skill memories
  - These external skills are costly to maintain and can be mismatched
    with the current policy's state distribution

## 2. Key Insight / Contribution

- Completed on-policy trajectories already contain the decision knowledge
  needed — they just need to be extracted as hindsight skills.
- Since skills come from the current policy's own rollouts, they're
  automatically distribution-matched. No external skill library needed.
- Two-level skill hierarchy with critical-first routing captures both
  global workflows and local critical decisions.
- The method is purely a training augmentation — at inference time the
  policy acts normally, no analyzer calls or skill retrieval.

## 3. Method

### 3.1 Overview

OPID augments GRPO-style RL with dense token-level supervision in three stages:

$$
\underbrace{\text{Skill Extraction}}_{\text{Stage 1}} \;\longrightarrow\; \underbrace{\text{Critical-First Routing + Self-Distillation}}_{\text{Stage 2}} \;\longrightarrow\; \underbrace{\text{Policy Optimization}}_{\text{Stage 3}}
$$

**Stage 1:** Completed trajectories $\tau \sim \pi_{\theta}$ are serialized and fed to an external LLM analyzer $A(\tau)$, which produces:
- An **episode-level skill** $s^{\mathrm{ep}}_{\tau}$: a global workflow summary (for successful trajectories) or an avoidance rule (for failed ones).
- **Step-level skills** $\{s^{\mathrm{step}}_{\tau,t}\}_{t \in \mathcal{C}_{\tau}}$ for critical timesteps identified by the analyzer, where $\mathcal{C}_{\tau} \subseteq \{1, \ldots, T\}$ is the set of critical step indices.

**Stage 2:** A critical-first routing mechanism selects the most appropriate skill per timestep and injects it into the context. The old policy $\pi_{\theta_{\mathrm{old}}}$ then performs paired scoring (with and without the skill) to produce a per-token skill advantage.

**Stage 3:** The skill advantage is combined with the GRPO episode advantage into a unified OPID advantage, and the policy is updated via a clipped PPO objective with KL regularization.

### 3.2 Architecture

$$
\boxed{
\begin{array}{c}
\textbf{ON-POLICY ROLLOUTS} \\
q \;\xrightarrow{\;\pi_{\theta}\;}\; G_q = \{\tau^{(1)}, \ldots, \tau^{(N)}\} \;\xrightarrow{\;}\; \{R(\tau)\}
\end{array}
}
$$

$$
\downarrow
$$

$$
\boxed{
\begin{array}{c}
\textbf{STAGE 1: SKILL EXTRACTION} \\
A(\tau) \;\longrightarrow\;
\begin{cases}
s^{\mathrm{ep}}_{\tau} & \text{(global workflow / avoidance)} \\
\{s^{\mathrm{step}}_{\tau,t}\}_{t \in \mathcal{C}_{\tau}} & \text{(critical-step guidance)}
\end{cases}
\end{array}
}
$$

$$
\downarrow
$$

$$
\boxed{
\begin{array}{c}
\textbf{STAGE 2: CRITICAL-FIRST ROUTING + PAIRED SCORING} \\
\forall\, t \in \tau:\quad
s_{\tau,t} =
\begin{cases}
s^{\mathrm{step}}_{\tau,t} & \text{if } t \in \mathcal{C}_{\tau} \;\text{(critical)} \\
s^{\mathrm{ep}}_{\tau} & \text{otherwise}
\end{cases} \\[12pt]
\tilde{h}_{\tau,t} = \mathcal{H}(h_{\tau,t},\; s_{\tau,t}) \quad \text{(skill-augmented context)} \\[12pt]
\ell^{\mathrm{old}}_{\tau,t,\ell} = \log \pi_{\theta_{\mathrm{old}}}(y_{\tau,t,\ell} \mid h_{\tau,t},\, y_{<\ell}) \\
\ell^{\mathrm{skill}}_{\tau,t,\ell} = \log \pi_{\theta_{\mathrm{old}}}(y_{\tau,t,\ell} \mid \tilde{h}_{\tau,t},\, y_{<\ell}) \\[12pt]
A^{\mathrm{skill}}_{\tau,t,\ell} = \bigl(\ell^{\mathrm{skill}}_{\tau,t,\ell} - \ell^{\mathrm{old}}_{\tau,t,\ell}\bigr) \cdot m_{\tau,t,\ell}
\end{array}
}
$$

$$
\downarrow
$$

$$
\boxed{
\begin{array}{c}
\textbf{STAGE 3: POLICY OPTIMIZATION} \\
A^{\mathrm{ep}}_{\tau} = \frac{R(\tau) - \mu_q}{\sigma_q} \\[8pt]
A^{\mathrm{OPID}}_{\tau,t,\ell} = A^{\mathrm{ep}}_{\tau} \cdot m_{\tau,t,\ell} \;+\; \lambda_{\mathrm{skill}} \cdot A^{\mathrm{skill}}_{\tau,t,\ell} \\[8pt]
\mathcal{L}(\theta) = -\mathbb{E}_{\tau,t,\ell}\!\left[\min\!\bigl(\rho \cdot A^{\mathrm{OPID}},\; \mathrm{clip}(\rho, 1{-}\epsilon, 1{+}\epsilon) \cdot A^{\mathrm{OPID}}\bigr)\right] + \beta \cdot \mathcal{L}_{\mathrm{KL}}(\theta)
\end{array}
}
$$

### 3.3 On-Policy Hierarchical Skill Extraction (Formal)

Given a completed trajectory $\tau = (q, o_1, a_1, \ldots, o_T, a_T, R(\tau))$ sampled from the current policy $\pi_{\theta}$, OPID serializes the full interaction record and queries an external analyzer model $A$:

$$
\left(s^{\mathrm{ep}}_{\tau},\; \{(t_i, s^{\mathrm{step}}_{\tau,t_i})\}_{i=1}^{K}\right) = A\!\left(\mathrm{serialize}(\tau)\right)
$$

where $K = |\mathcal{C}_{\tau}|$ is the number of critical steps and $t_i \in \mathcal{C}_{\tau}$ are the critical timestep indices. The analyzer is a separate LLM (GLM-5.2 in experiments, $\mathrm{temp}=0.4$) — not the training backbone.

**Episode-level skill.** For a successful trajectory ($R(\tau) > 0$), the skill captures the effective workflow:

$$
s^{\mathrm{ep}}_{\tau} = \text{"strategy that led to success: workflow steps, key decisions, ordering"}
$$

For a failed trajectory ($R(\tau) \leq 0$), it captures the avoidance rule:

$$
s^{\mathrm{ep}}_{\tau} = \text{"why it failed: what to avoid, missing prerequisites, wrong ordering"}
$$

**Step-level skills.** Localized guidance at critical decision points, where "critical" is defined as timesteps where the agent's action had outsized impact on the trajectory outcome:

$$
\{s^{\mathrm{step}}_{\tau,t_i}\}_{i=1}^{K}, \qquad K \leq K_{\max} = \begin{cases} 5 & \text{ALFWorld / WebShop} \\ 2 & \text{Search QA} \end{cases}
$$

The average number of critical steps is $\approx 3.7$ per trajectory on ALFWorld.

### 3.4 Critical-First Routing

Rather than combining both skill levels additively at every timestep, OPID uses a hard routing scheme:

$$
s_{\tau,t} = \begin{cases}
s^{\mathrm{step}}_{\tau,t} & \text{if } t \in \mathcal{C}_{\tau} \quad \text{(precise, local guidance)} \\
s^{\mathrm{ep}}_{\tau} & \text{if } t \notin \mathcal{C}_{\tau} \quad \text{(broad, global guidance)}
\end{cases}
$$

This is a **hard switch, not superposition**. The motivation: at critical states the step-level skill provides sharper supervision; at non-critical states the episode-level skill provides a stable background signal. Blindly combining both at every step is suboptimal (ablation: $-6.8$ on ALFWorld).

**Theoretical guarantee (Proposition 3).** With perfect critical-step detection, critical-first routing recovers the oracle choice between step-level and episode-level teachers. With imperfect detection, the degradation is bounded by $\Gamma \cdot \Pr(\text{detection error})$, where $\Gamma$ is the maximum advantage gap between the two skill levels.

### 3.5 Skill-Aware Self-Distillation via Log-Prob Shift

The routed skill $s_{\tau,t}$ is injected into the interaction history via a context augmentation function $\mathcal{H}$:

$$
\tilde{h}_{\tau,t} = \mathcal{H}(h_{\tau,t},\; s_{\tau,t})
$$

where $h_{\tau,t}$ is the original history prefix at timestep $t$ and $\tilde{h}_{\tau,t}$ is the skill-augmented version.

The **old policy** $\pi_{\theta_{\mathrm{old}}}$ (frozen parameters from before the update) scores each response token $y_{\tau,t,\ell}$ under both contexts:

$$
\ell^{\mathrm{old}}_{\tau,t,\ell} = \log \pi_{\theta_{\mathrm{old}}}(y_{\tau,t,\ell} \mid h_{\tau,t},\, y_{<\ell})
$$

$$
\ell^{\mathrm{skill}}_{\tau,t,\ell} = \log \pi_{\theta_{\mathrm{old}}}(y_{\tau,t,\ell} \mid \tilde{h}_{\tau,t},\, y_{<\ell})
$$

The **per-token skill advantage** is the log-probability shift, masked to valid response tokens:

$$
\boxed{A^{\mathrm{skill}}_{\tau,t,\ell} = \bigl(\ell^{\mathrm{skill}}_{\tau,t,\ell} - \ell^{\mathrm{old}}_{\tau,t,\ell}\bigr) \cdot m_{\tau,t,\ell}}
$$

where $m_{\tau,t,\ell} \in \{0, 1\}$ is the response-token mask (1 for generated response tokens, 0 for prompt/observation tokens).

**Interpretation:**
- $A^{\mathrm{skill}}_{\tau,t,\ell} > 0$: The hindsight skill makes this token *more* likely → the token is consistent with the extracted knowledge → **reinforce**.
- $A^{\mathrm{skill}}_{\tau,t,\ell} < 0$: The skill makes this token *less* likely → the token contradicts what hindsight says → **suppress**.

This requires only a paired forward pass through the same frozen old policy — no separate teacher model is needed.

### 3.6 Forward pass / pipeline

**Training iteration:**

1. Sample batch of task prompts $\{q_i\}_{i=1}^{B}$
2. For each prompt $q_i$, sample $N=8$ on-policy trajectories from current policy: $\{\tau^{(j)}_{q_i}\}_{j=1}^{N}$
3. Compute GRPO group-relative episode advantages:
   $$
   \mu_{q_i} = \frac{1}{N}\sum_{j=1}^{N} R(\tau^{(j)}_{q_i}), \qquad
   \sigma_{q_i} = \sqrt{\frac{1}{N}\sum_{j=1}^{N} \bigl(R(\tau^{(j)}_{q_i}) - \mu_{q_i}\bigr)^2}, \qquad
   A^{\mathrm{ep}}_{\tau} = \frac{R(\tau) - \mu_q}{\sigma_q}
   $$
4. For each trajectory $\tau$:
   a. Serialize $\tau$ → call analyzer $A(\tau)$ → $(s^{\mathrm{ep}}_{\tau}, \{s^{\mathrm{step}}_{\tau,t}\}_{t \in \mathcal{C}_{\tau}})$
   b. For each timestep $t$:
      - Route: $s_{\tau,t} = s^{\mathrm{step}}_{\tau,t}$ if $t \in \mathcal{C}_{\tau}$, else $s_{\tau,t} = s^{\mathrm{ep}}_{\tau}$
      - Construct $\tilde{h}_{\tau,t} = \mathcal{H}(h_{\tau,t}, s_{\tau,t})$
      - Score: $\ell^{\mathrm{old}}$ and $\ell^{\mathrm{skill}}$ via frozen $\pi_{\theta_{\mathrm{old}}}$
      - Compute $A^{\mathrm{skill}}_{\tau,t,\ell} = (\ell^{\mathrm{skill}} - \ell^{\mathrm{old}}) \cdot m$
      - Combine: $A^{\mathrm{OPID}}_{\tau,t,\ell} = A^{\mathrm{ep}}_{\tau} \cdot m + \lambda_{\mathrm{skill}} \cdot A^{\mathrm{skill}}_{\tau,t,\ell}$
5. Update $\pi_{\theta}$ via clipped PPO on $A^{\mathrm{OPID}}$ with KL regularization

**Inference:** Standard policy forward pass $\pi_{\theta}(y \mid h)$, no skills, no analyzer, no extra context.

### 3.7 Loss Function

$$
\boxed{\mathcal{L}(\theta) = -\mathbb{E}_{\tau,t,\ell}\!\left[ \min\!\Bigl( \rho_{\tau,t,\ell}(\theta) \cdot A^{\mathrm{OPID}}_{\tau,t,\ell},\; \mathrm{clip}\bigl(\rho_{\tau,t,\ell}(\theta),\, 1{-}\epsilon,\, 1{+}\epsilon\bigr) \cdot A^{\mathrm{OPID}}_{\tau,t,\ell} \Bigr) \right] + \beta \cdot \mathcal{L}_{\mathrm{KL}}(\theta)}
$$

where the **importance ratio** is:

$$
\rho_{\tau,t,\ell}(\theta) = \exp\!\bigl( \log \pi_{\theta}(y_{\tau,t,\ell} \mid h_{\tau,t},\, y_{<\ell}) - \log \pi_{\theta_{\mathrm{old}}}(y_{\tau,t,\ell} \mid h_{\tau,t},\, y_{<\ell}) \bigr) = \frac{\pi_{\theta}(y \mid h, y_{<\ell})}{\pi_{\theta_{\mathrm{old}}}(y \mid h, y_{<\ell})}
$$

**Hyperparameters:**

| Parameter | Symbol | Value |
|-----------|--------|-------|
| PPO clip range | $\epsilon$ | 0.2 |
| KL regularization coeff. | $\beta$ | 0.01 |
| Skill advantage coeff. | $\lambda_{\mathrm{skill}}$ | 0.001 |
| Group size | $N$ | 8 |
| Learning rate | $\eta$ | $1 \times 10^{-6}$ |

## 4. Mathematical Analysis

### 4.1 GRPO Episode Advantage

For a task prompt $q$, let $G_q = \{\tau^{(1)}, \ldots, \tau^{(N)}\}$ be the group of $N$ sampled trajectories:

$$
\mu_q = \frac{1}{N}\sum_{j=1}^{N} R\!\left(\tau^{(j)}\right), \qquad
\sigma_q = \sqrt{\frac{1}{N}\sum_{j=1}^{N} \left(R\!\left(\tau^{(j)}\right) - \mu_q\right)^2}
$$

$$
\boxed{A^{\mathrm{ep}}_{\tau} = \frac{R(\tau) - \mu_q}{\sigma_q}}
$$

This is broadcast to all tokens via the mask: $A^{\mathrm{ep}}_{\tau,t,\ell} = A^{\mathrm{ep}}_{\tau} \cdot m_{\tau,t,\ell}$.

### 4.2 Skill Advantage (Log-Prob Shift)

The core distillation signal. For each response token at position $(t, \ell)$ in trajectory $\tau$:

$$
\Delta\ell_{\tau,t,\ell} = \underbrace{\log \pi_{\theta_{\mathrm{old}}}(y_{\tau,t,\ell} \mid \tilde{h}_{\tau,t},\, y_{<\ell})}_{\ell^{\mathrm{skill}}} - \underbrace{\log \pi_{\theta_{\mathrm{old}}}(y_{\tau,t,\ell} \mid h_{\tau,t},\, y_{<\ell})}_{\ell^{\mathrm{old}}}
$$

$$
\boxed{A^{\mathrm{skill}}_{\tau,t,\ell} = \Delta\ell_{\tau,t,\ell} \cdot m_{\tau,t,\ell}}
$$

Key properties:
- The skill advantage is **zero-mean** on average when the skill provides no information (since both log-probs come from the same policy).
- It requires **no separate teacher model** — just a second forward pass through $\pi_{\theta_{\mathrm{old}}}$ with a modified context.
- The signal is **automatically on-policy** because the skills are extracted from the same rollout distribution.

### 4.3 Combined OPID Advantage

$$
\boxed{A^{\mathrm{OPID}}_{\tau,t,\ell} = \underbrace{A^{\mathrm{ep}}_{\tau} \cdot m_{\tau,t,\ell}}_{\text{trajectory-level (GRPO)}} + \underbrace{\lambda_{\mathrm{skill}}}_{\times\, 0.001} \cdot \underbrace{A^{\mathrm{skill}}_{\tau,t,\ell}}_{\text{token-level (distillation)}}}
$$

The episode advantage provides coarse "was this trajectory good?" signal; the skill advantage provides fine-grained "should this specific token be reinforced?" signal.

### 4.4 Relative-KL Decomposition (Proposition 1)

The unclipped skill loss decomposes as:

$$
\boxed{\mathcal{L}^{\mathrm{unclip}}_{\mathrm{skill}}(\theta) = \lambda_{\mathrm{skill}} \left[ \underbrace{\mathcal{L}_{\mathrm{RKL}}(\theta)}_{\text{reverse-KL to skill distillation target}} - \underbrace{D_{\mathrm{KL}}\!\left(\pi_{\theta} \,\|\, \pi_{\theta_{\mathrm{old}}}\right)}_{\text{behavior KL}} \right]}
$$

where the reverse-KL term encourages the policy to match the skill-conditioned distribution and the behavior-KL term prevents excessive deviation from the old policy.

**At the behavior policy** ($\theta = \theta_{\mathrm{old}}$): $D_{\mathrm{KL}}(\pi_{\theta} \|\pi_{\theta_{\mathrm{old}}}) = 0$, so the skill loss reduces to pure scaled reverse-KL:

$$
\mathcal{L}^{\mathrm{unclip}}_{\mathrm{skill}}(\theta_{\mathrm{old}}) = \lambda_{\mathrm{skill}} \cdot \mathcal{L}_{\mathrm{RKL}}(\theta_{\mathrm{old}})
$$

**Away from the behavior policy**, the two KL terms diverge, making this a **relative-KL** objective — not pure reverse-KL. This is significant: it means the skill signal is automatically regularized by the policy's own movement, preventing runaway distillation.

### 4.5 Exact Recovery of Reverse-KL (Corollary 2)

When the KL regularization coefficient exactly matches the skill coefficient ($\beta = \lambda_{\mathrm{skill}}$), the two objectives cancel precisely and the skill loss recovers standard reverse-KL distillation everywhere:

$$
\mathcal{L}^{\mathrm{unclip}}_{\mathrm{skill}}(\theta) + \beta \cdot D_{\mathrm{KL}}(\pi_{\theta} \| \pi_{\theta_{\mathrm{old}}}) \;\stackrel{\beta = \lambda_{\mathrm{skill}}}{=}\; \lambda_{\mathrm{skill}} \cdot \mathcal{L}_{\mathrm{RKL}}(\theta)
$$

In practice, $\beta = 0.01 \gg \lambda_{\mathrm{skill}} = 0.001$, so the behavior-KL penalty dominates and the skill signal acts as a gentle bias rather than a full distillation target.

### 4.6 On-Policy Occupancy Matching (Proposition 2)

Since skills are extracted from the current policy's own rollouts, the context distribution under training matches the behavior policy's context distribution:

$$
d_{\mu}(h) = d_b(h) \implies D_{\mathrm{TV}}(d_{\mu},\, d_b) = 0
$$

This eliminates the context-distribution mismatch that plagues off-policy skill distillation methods. External skill libraries, by contrast, can induce $\mathrm{D}_{\mathrm{TV}}(d_{\mu}, d_b) > 0$, causing systematic bias in the distillation signal.

### 4.7 Learning Signal Under Reward Ties (Corollary 3)

When all trajectories in a group receive tied rewards, the GRPO advantage vanishes:

$$
R(\tau^{(1)}) = R(\tau^{(2)}) = \cdots = R(\tau^{(N)}) \implies A^{\mathrm{ep}}_{\tau} = 0 \;\forall\, \tau
$$

Under this degenerate case, standard GRPO provides **zero learning signal**. OPID, however, still produces non-trivial gradients provided the skill distribution differs from the behavior distribution:

$$
q(s \mid \tau) \neq b(s \mid \tau) \implies A^{\mathrm{skill}}_{\tau,t,\ell} \neq 0 \;\text{in general}
$$

This means OPID can learn even from trajectories that are equally rewarding but internally different — extracting signal from *how* the trajectory succeeded, not just *that* it succeeded.

## 5. Training

### Datasets

| Benchmark | Domain | Train | Test |
|-----------|--------|-------|------|
| ALFWorld | Embodied household | 2,400 | 140 seen + 134 unseen |
| WebShop | E-commerce navigation | 2,400 | 128 |
| Search-based QA | 7 QA subsets | 19,200 | 51,713 |

### Optimizer / schedule / hyperparameters

| Parameter | Value |
|-----------|-------|
| Training steps | 150 |
| Batch size | 16 (ALFWorld/WebShop), 128 (Search) |
| Group size $N$ | 8 |
| Learning rate $\eta$ | $1 \times 10^{-6}$ |
| PPO clip $\epsilon$ | 0.2 |
| Skill coefficient $\lambda_{\mathrm{skill}}$ | 0.001 |
| KL coefficient $\beta$ | 0.01 |
| Max prompt len | 2048 (ALFWorld), 4096 (others) |
| Response len | 512 |
| Max steps $T$ | 30 (ALFWorld), 15 (WebShop), 4 (Search) |

### Implementation tricks

- LLM analyzer (GLM-5.2) with $\mathrm{temperature}=0.4$ for diverse skill extraction
- Max critical steps capped: $K_{\max} = 5$ for ALFWorld/WebShop, $K_{\max} = 2$ for Search QA
- Response-token masking: only compute advantage on response tokens ($m_{\tau,t,\ell} = 1$ for generated tokens, 0 for prompt/observation tokens)
- Group-relative normalization (GRPO-style) for variance reduction

### Compute budget

- $8 \times$ NVIDIA A800 80 GB GPUs
- Analyzer is a separate model (GLM-5.2), not the training backbone
- Training overhead per step: 1 analyzer call + 2 forward passes per trajectory step (original context + skill-augmented context)

## 6. Results & Ablations

### 6.1 Headline numbers (Qwen2.5-3B backbone)

| Benchmark | GRPO | OPID | $\Delta$ |
|-----------|------|------|----------|
| ALFWorld Avg | 75.0 | **84.3** | +9.3 |
| Search QA Avg | 36.4 | **45.0** | +8.6 |
| WebShop Score | 79.8 | **85.0** | +5.2 |
| WebShop Succ. | 63.3 | **74.2** | +10.9 |

OPID provides consistent improvements across all three benchmarks, with the largest absolute gain on WebShop success rate (+10.9 percentage points).

### 6.2 Comparison with Skill-GRPO

A critical finding: OPID avoids the train-test mismatch that cripples Skill-GRPO:

| Method | ALFWorld Avg | Notes |
|--------|-------------|-------|
| GRPO | 75.0 | No skills, baseline |
| Skill-GRPO (inference skills removed) | 60.2 | **Collapses** — train-test mismatch |
| Skill-GRPO* (inference skills kept) | 80.5 | External skills at test time |
| **OPID** | **84.3** | Skills distilled into weights |

Skill-GRPO without inference skills is 14.8 points *worse* than plain GRPO, demonstrating that relying on external skills at training then removing them is actively harmful. OPID's distillation approach entirely avoids this by absorbing skill knowledge into model parameters.

### 6.3 Key ablation findings

| Ablation | ALFWorld Avg | WebShop Succ. | Interpretation |
|----------|-------------|---------------|----------------|
| Full OPID | **84.3** | **74.2** | — |
| w/o episode skill | 74.1 ($-10.2$) | 67.2 ($-7.0$) | Global workflows are the backbone of the method |
| w/o step skill | 79.1 ($-5.2$) | 65.6 ($-8.6$) | Local precision at critical states is essential |
| w/o routing (blind combine) | 77.5 ($-6.8$) | — | Smart routing > naive combination |

Both skill levels contribute, but episode-level skills carry more weight on ALFWorld while step-level skills are relatively more important on WebShop. The routing ablation shows that simply using both skills everywhere is worse than critical-first routing — precision at the right moments matters.

### 6.4 Sample Efficiency (Table 7)

| Data fraction | GRPO | OPID | $\Delta$ |
|---------------|------|------|----------|
| 20%  | 27.3 | 36.7 | +9.4 |
| 40%  | 42.2 | 54.7 | +12.5 |
| 60%  | 56.3 | **71.9** | +15.6 |
| 80%  | 58.6 | **78.9** | +20.3 |
| 100% | 75.0 | **84.3** | +9.3 |

OPID's advantage is largest in the low- and mid-data regimes: the Δ grows from +9.4 at 20% to a peak of **+20.3 at 80%**, then narrows back to +9.3 at 100% as GRPO finally gets enough rollouts to catch up. Notably OPID trained on **60%** of the data (71.9) already approaches full-data GRPO (75.0), and OPID at 80% (78.9) *exceeds* full-data GRPO — confirming that dense token-level skill supervision extracts substantially more learning signal per rollout than sparse outcome rewards alone.

> **Source:** Appendix C.1, Table 7 (verbatim 5-fraction grid; Qwen2.5-3B-Instruct, ALFWorld success rate). The paper's §C.1 prose explicitly cites the +15.6 (60%) and +20.3 (80%) figures.

### 6.5 Cross-Domain Generalization (ALFWorld Unseen, Table 8)

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | Avg. |
|--------|------|------|-------|------|------|-------|------|
| ReAct  | 17.4 | 6.7  | 8.8   | 7.4  | 9.1  | 0.0   | 8.2  |
| GRPO   | 73.9 | 60.0 | 82.4  | 59.3 | 72.7 | **76.9** | 70.9 |
| **OPID** | **78.3** | **86.7** | **82.4** | **77.8** | **77.3** | 69.2 | **78.6** |
| $\Delta$ (OPID−GRPO) | +4.4 | +26.7 | +0.0 | +18.5 | +4.6 | −7.7 | +7.7 |

OPID's distilled skills transfer to unseen task types (+7.7 Avg over GRPO), with the largest gains exactly where the no-skill ReAct baseline collapses hardest — **Look (+26.7)** and **Heat (+18.5)** (ReAct manages only 6.7 / 7.4 there). Clean is a tie (82.4 = 82.4). The one regression is **Pick2 (−7.7)**, where GRPO's already-strong 76.9 leaves little headroom. The pattern indicates the distilled episode-level workflows and step-level decision rules capture generalizable decision principles rather than memorized training trajectories.

> **Source:** Appendix C.2, Table 8 (verbatim 6-task grid; Qwen2.5-3B-Instruct, ALFWorld Unseen split). The paper's §4.2 multi-backbone prose corroborates the +7.7 Avg gain and the "particularly clear gains on Look and Heat" ordering.

### 6.6 Training Dynamics

- OPID diverges from GRPO around mid-training and maintains a growing lead through step 150.
- **Episode length reduction**: OPID reduces average episode length from 17–18 steps to 15–16 steps on ALFWorld.
- Both **success rate ↑ AND episode length ↓** — the agent learns more direct, efficient workflows rather than succeeding via longer detours.

### 6.7 Multi-Backbone Results

| Backbone | ALFWorld Avg (GRPO → OPID) |
|----------|---------------------------|
| Qwen2.5-3B-Instruct | 75.0 → **84.3** (+9.3) |
| Qwen2.5-7B-Instruct | 81.2 → **90.0** (+8.8) |
| Qwen3-1.7B-Instruct | 46.1 → **58.9** (+12.8) |

Gains are consistent across all tested backbone sizes, from 1.7B to 7B parameters, and are most pronounced on the smallest Qwen3-1.7B backbone (+12.8).

## 7. Limitations

- Only three benchmarks tested — no web arena, software engineering, or open-ended tasks.
- Analyzer quality is critical but not studied as a variable (always GLM-5.2, a different model from the policy backbone).
- Only Qwen backbones tested — unclear if results transfer to LLaMA, Mistral, or other families.
- No comparison with value-based methods (PPO with learned critic, GAE).
- Skill extraction prompt is hand-designed and domain-specific.
- Training overhead: analyzer LLM call + paired forward pass per trajectory step.
- Theoretical analysis relies on common-support and bounded-range assumptions.
- On Search-based QA with Qwen3-1.7B, OPID is marginally below GRPO (40.4 vs 40.8 average accuracy).
- $\lambda_{\mathrm{skill}} = 0.001$ is the only value reported — sensitivity analysis is absent.

## 8. Open Questions / Ideas

- What happens with a weaker analyzer (e.g., same backbone model)? How much does analyzer quality matter?
- Can the skill extraction be learned end-to-end instead of prompted?
- How does OPID interact with longer horizons ($T > 100$)? Does the critical-step cap become a bottleneck?
- What about iterative skill refinement across multiple training iterations?
- Could critical-step detection be learned rather than prompted?
- How sensitive is the method to $\lambda_{\mathrm{skill}}$? The paper only reports 0.001.
- Extension to value-based RL (critic + skill shaping)?
- Combination with exploration methods (e.g., SPARK)?
- Does the relative-KL structure suggest a natural curriculum (start with high $\beta/\lambda$ ratio, decrease over training)?
