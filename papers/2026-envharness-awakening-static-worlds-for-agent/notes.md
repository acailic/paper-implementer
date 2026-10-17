# Notes — EnvHarness: Awakening Static Worlds for Agent Learning

**Paper**: EnvHarness: Awakening Static Worlds for Agent Learning
**Authors**: Chengsong Huang (WUSTL), Zifeng Wang, Rujun Han, Jun Yan, Yanfei Chen, Zoey CuiZhu, Ke Jiang, Peng Xia, Han Yu, Yufan Zhuang, Yifei Ming, Jiaqi Pan, Bhavana Dalvi Mishra, Jiaxin Huang, Burak Gokturk, Tomas Pfister, Chen-Yu Lee (Google Cloud AI Research + WUSTL + UNC)
**arXiv**: 2608.19880 (cs.AI, Aug 2026)
**Code**: https://github.com/google-research/envharness
**Project page**: https://envharness.com/

---

## First pass — big picture (2026-10-17)

### The problem, in my own words

LLM agents learn by interacting with environments (web sites, codebases, embodied
simulators, office tools). But the environments themselves are hand-built and
**static**: they behave identically no matter which agent interacts with them or
how good that agent already is. This causes two failure modes for learning:
(1) no *targeted* signal — the environment can't isolate a specific weakness of
a specific agent; (2) nothing left to teach — once the agent solves the existing
instances, the environment is exhausted.

The obvious fix — auto-generating new environments — has two problems of its
own: generation pipelines are domain-specific (a web-nav generator can't make
SWE tasks), and generated/LLM-made verifiers are unreliable, forcing expensive
over-generation + filtering.

The paper's move: **don't create new environments — wrap the existing one.**
Just like an *agent harness* (tools, memory, skills) upgrades a frozen LLM into
an agent without touching its weights, an **EnvHarness** upgrades a frozen
static environment into a customized one without touching its code or verifier.

### Core idea (one sentence)

A programmable wrapper layer that reshapes an environment strictly through its
standard `reset()/step()/obs()` interface — changing the initial state
(**Stage**), the interaction rules (**Contract**), or joining environments
(**Chain**) — so the original, trusted, human-built verifier is inherited
untouched, while an automated loop (**EnvRigger**) diagnoses a target policy's
failures from trajectories and writes/validates components that target exactly
those failures.

### The three plug-in components (Sec. 2.2)

Formally an environment is E = (S, A, O, T, R, s0) and a component is an
interface-level transformation w(E) = E':

1. **Stage** `w_stage,δ` — specified by a short action script δ = (a1..ak)
   executed on s0 after reset: s0' = T(...T(T(s0,a1),a2)...,ak). Only the
   *initial state* changes. Use: hide objects (make task harder / force
   search), or pre-complete subgoals (shorten horizon / isolate a skill).
2. **Contract** `w_contract,r` — a triplet r = (f_A, f_T, f_O) of maps
   (default identity) rewriting the action space, transition dynamics, and
   observations. Use: block shortcut actions (f_A), enforce preconditions like
   "must hold mug before cleaning" (f_T), truncate/mask observations to force
   multi-step spatial reasoning (f_O), attach structured feedback.
3. **Chain** `w_chain,ℓ` — pair ℓ = (E_ext, g) joining two environments with
   composition logic g; spaces union (A' = A ∪ A_ext), reward R' = conjunction
   of both verifiers. Use: extend horizon, force goal persistence past the
   point the agent would normally stop.

Components **compose** (w_chain ∘ w_contract ∘ w_stage)(E) but are
**non-commutative** — nesting order matters.

### EnvRigger (Sec. 3.2) — the automation loop

Task-policy-conditioned map H(E, t; π) = (wk ∘ ... ∘ w1)(E). Treats π as a
black box. Four stages:

1. **Observe** — run π on task t, collect success AND failure rollouts
   (successes bound the flaw; failures expose it).
2. **Diagnose** — find root causes (repetitive action loops, long-obs parsing
   failures, misread tool constraints). Direction depends on regime: struggling
   policy → scaffold/simplify; perfect success rate → make it harder.
3. **Write** — synthesize candidate components (often Stage + Contract
   together) targeting the diagnosis, e.g. a Contract that blocks a fragile
   shortcut action.
4. **Validate** — wrap env with candidates, fresh rollouts; accept /
   reject (unsolvable or non-challenging) / refine (signal poorly scaled) in a
   write-and-validate loop until accepted or budget exhausted.

Assumes deterministic resets for reproducible Stage mutations.

### Headline results (first pass)

- **5 benchmarks, 4 domains**: ALFWorld (embodied), WebArena (web), SWE-bench
  Verified (SWE), OfficeQA + SpreadsheetBench (office automation).
- Skill-based learning: up to **+9.0 pts** (ALFWorld OOD: 61.4 → 70.4) over
  skills from original envs; beats domain-specific generators (GenEnv, VeriEnv,
  SWE-smith) *in their own domains*. Office automation has no generation
  baseline at all — EnvHarness is the only generic option.
- Efficiency: SWE-bench avg steps **53.6 → 49.6** (−9.8% ish) while original-env
  skills *increase* steps to 55.0.
- RL (GRPO, Qwen3-8B-base, ALFWorld + WebShop): EnvHarness-trained policy wins
  3/4 metrics (e.g. ALFWorld in-dist 87.9 vs 81.4).
- **Scaling**: with identical env budget on SWE-bench, EnvHarness climbs
  47.67 → 54.79 @300 envs and still rising, vs 52.13 original / 50.37
  generated — because each batch targets the *current* skill-equipped policy
  (co-evolution).
- Chain-only skills: big step reduction (AS 53.6 → 42.0) but slightly lower SR
  alone; combined Stage/Contract + Chain skill sets = best of both (SR 54.3,
  AS 43.1).
- Cross-model: gains (+2.7–3.7 abs) hold across Gemini 3.1 Flash-Lite, Qwen3.6
  27B, Gemini 3.5 Flash, Claude Sonnet 4.6 — same pipeline, only diagnoses
  differ. Weakest models benefit most from any skills.
- EnvRigger also accepts **explicit user-specified weaknesses** ("submits
  patches without running tests" → Contract rejects submission until tests ran
  → distills a general verification-driven-development skill).

### What I like immediately

- The agent-harness ↔ env-harness symmetry (Fig. 2, Table 1) is a genuinely
  clean framing. "Environment construction as wrapping, not authoring."
- Keeping the human-built verifier is the key safety/economics argument vs.
  generated environments: no verifier reliability problem at all.
- EnvRigger uses the *same* model backbone as the policy (no distillation from
  a stronger teacher) — gains are honest.

### Terms / concepts to dig into on second pass

- ReasoningBank skill-extraction protocol (Ouyang et al. 2025) — how exactly
  are skills distilled from EnvHarness trajectories? 2–3 skills per 50-env
  bank; need the retrieval protocol.
- The exact class architecture / interface protocol (Appendix C) —
  `modify_transition`, `env_state.extras`, how Contract hooks are implemented.
- Validation acceptance criteria details (Appendix A) — what thresholds count
  as "solvable but challenging"? The success-rate window.
- Why Chain was excluded from the automated EnvRigger loop ("difficult to
  observe internal states of joined environments") — and how they used it
  separately (random pairing).
- GRPO training details (Appendix F.1) and how reward comes from the inherited
  verifier during RL.
- Appendix D Chain examples (concatenate / interleave / branch).
- Baseline details (GenEnv, VeriEnv, SWE-smith) and the split configs (E.1).
- Unsupervised environment design (PAIRED, Dennis et al. 2020) — the RL
  curriculum line this paper cites as kindred.

### Limitations (from Appendix H, noted for later)

- Cost of the iterative design loop (weaker designer → more revisions → more
  rollouts); paid once per environment, not per episode.
- Requires a **resettable, gym-style interface** — excludes live services,
  real user accounts, physical robots.
- Chain is purely sequential concatenation; no branching, no shared state, no
  notion of semantic compatibility between subtasks.
- Text-only actions/observations for now.

---

**Next**: second pass — deep read of Sec. 2–3 math + appendices A/C/D/F, deepen
these notes, then breakdown.
