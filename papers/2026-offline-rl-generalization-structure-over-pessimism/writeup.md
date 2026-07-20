# Writeup — Generalization in Offline RL: Structure > Pessimism

> arXiv:2607.02288 (2026).

## The idea in one sentence

In zero-shot policy transfer, it doesn't matter *how much* pessimism your
offline RL agent has — what matters is whether the pessimism's *structure*
respects the optimal solution's symmetries.

## What I verified

Two theorems on a one-step rotational Reacher (Counter-example 1):

**Theorem 1.** If the pessimistic target Q̂* is symmetric under the training
subgroup B, the greedy policy is optimal at test contexts for *arbitrarily
large* pessimism η. My result: test return = 1.00 for η from 0.01 to 10.0.

**Theorem 2.** There exist asymmetric targets that are optimal in training
but arbitrarily worse at test. My result: Q̂_asym drops from 1.00 to 0.00
at η ≥ 0.5 — the exact theoretical threshold (r−γr)/0.21 ≈ 0.48.

## What implementing it clarified

### 1. The phase transition is sharp and predictable

The asymmetric target fails exactly when γr + 0.21η > r, i.e., when the
incorrectly-equivariant boost on the suboptimal action a₃ exceeds the gap
between r and γr. At η = 0.48, Q(s₄₅, a₃) = 0.9 + 0.21·0.48 = 1.0 = r.
Below that, the greedy policy still picks a₁. Above it, the policy switches
to a₃ → return drops to 0. The toy makes this visible as a clean step
function: return 1.0 → 0.0 at η ≈ 0.5.

### 2. "Equivariant but incorrect" is the failure mode

The asymmetric target isn't random noise — it's *equivariant* (it
transforms correctly under the training subgroup). The problem is it's
equivariant in the wrong action subspace: it rotates the penalties on a₂/a₃
by the context angle, which is a valid equivariance but doesn't match the
true Q*'s equivariance. At the unseen 45° context, the rotation
interpolates to boost a₃ above a₁. The failure is structural: you can't
detect it from training performance (both targets give return 1.0 on all
training contexts).

### 3. Theorem 1 is the surprising result

Most offline RL intuition says "less pessimism is better" — over-pessimism
suppresses good actions. Theorem 1 says: if the pessimism is symmetric, you
can crank η to infinity and the policy is still optimal. The reason: the
margin δ_Q between the best and second-best action grows proportionally to
η (both suboptimal actions get penalized by η), so the greedy policy is
preserved. This is why C_Θ(ε) is independent of η_max — the margin grows as
fast as the pessimism.

## Pointers to the code

| File | What |
|------|------|
| `implementation/model.py` | `RotationalReacher` + symmetric/asymmetric target constructors |
| `implementation/run.py` | Theorem 1 + 2 verification, Table 1 reproduction |

## Verdict

A clean theory paper with a clean counter-example. The toy environment
makes both theorems viscerally clear: symmetric pessimism is a free lunch
(arbitrary η, same optimal policy); asymmetric pessimism has a sharp
failure threshold determined by a simple algebraic condition. The broader
lesson — structure of regularization matters more than amount — extends
beyond offline RL to any setting with group symmetries.

🏆 Verdict: theorems you can see. Two lines of numpy verify what 30 pages
of proof establish.
