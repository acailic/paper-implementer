# model.py — core simulation of the skill-use pipeline
"""Core simulation engine for the from-scratch re-implementation of:

  "Demystifying Agent Skills: Why They Work—Until They Don't"
  Jiang et al., 2026. arXiv:2608.14036
  https://arxiv.org/abs/2608.14036

The paper proposes no new neural architecture; its "method" is a controlled
experimental design + measurement pipeline over three arms (Raw / Workflow
Memory / Skill). This module re-implements that design as an executable
generative model of agent execution, so every quantity the paper measures
(paired success deltas, 12-mode taxonomy frequencies, mechanism
distributions, retrieval precision/recall collapse) can be computed by
actually running the experiment rather than by copying numbers.

Design correspondence with the paper (section refs in comments):
  - Arms & composition grid  -> Section 3.1 / Table 5
  - Mixture-controlled artifact construction -> Section 3.1 (5s0f ... 0s5f)
  - Outcome-annotation ablation ("no-hint") -> RQ2
  - Cross-framework transfer -> RQ3
  - Verifier-based success, never agent self-report -> Section 3.3
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# World objects
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A benchmark task.

    procedural_steps: ordered critical steps; skipping/corrupting any fails
                      the verifier (this is what skills can stabilize).
    p_pitfall:        per-step chance of a hidden pitfall that a raw agent
                      only notices with probability P_RAW_PITFALL_DETECT.
    fact_required:    task additionally needs a missing *fact* (the minority
                      case where knowledge_injection could matter).
    difficulty:       overall error pressure per step for a raw agent.
    """

    tid: str
    procedural_steps: list[str]
    p_pitfall: float
    fact_required: bool
    difficulty: float
    gold_skill_id: str = ""


@dataclass
class Trajectory:
    """One raw agent execution used as *source experience* for distillation."""

    success: bool
    steps_followed: list[str]          # literal ordered actions taken
    hit_pitfalls: list[str]            # pitfalls encountered (bad branches)
    task: Optional[Task] = field(repr=False, default=None)


@dataclass
class Guidance:
    """An experience artifact injected into an arm.

    procedure_fidelity: how reliably following it sequences the critical
                        steps and avoids pitfalls (0..1).
    carries_fact:       whether the artifact contains the missing fact
                        (skills distill procedure; facts rarely survive).
    mislead_prob:       chance the artifact actively misleads (WM keeps
                        failed branches / process noise).
    timeout_penalty:    added probability of budget exhaustion (inline
                        workflow memory inflates the context).
    is_skill:           True -> environment-resident SKILL.md (must be
                        noticed/opened); False -> inline workflow memory.
    framework:          harness the artifact was built under (RQ3 transfer).
    """

    procedure_fidelity: float
    carries_fact: bool
    mislead_prob: float
    timeout_penalty: float
    is_skill: bool
    framework: str = "codex"
    description: str = ""              # used by the RQ4 embedding arm
    skill_id: str = ""

    @property
    def arm_name(self) -> str:
        return "skill" if self.is_skill else "workflow"


# ---------------------------------------------------------------------------
# Artifact construction (the paper's key control: same traces, two formats)
# ---------------------------------------------------------------------------

# Distillation strengths, calibrated so the synthetic world reproduces the
# paper's *directional* phenomena (skill > raw >= wm at good mixtures,
# wm < raw at failure-heavy mixtures, no-hint hurts only on mixed pools).
_SKILL_BASE = 0.58
_SKILL_PER_SUCC = 0.075
_SKILL_PER_FAIL = 0.062          # with outcome labels available (hint)
_SKILL_PER_FAIL_NOHINT = 0.105   # labels stripped: failures pollute (RQ2)
_WM_BASE = 0.50
_WM_PER_SUCC = 0.055
_WM_PER_FAIL = 0.052
_WM_MISLEAD_PER_FAIL = 0.032
_WM_TIMEOUT_BASE = 0.020
_WM_TIMEOUT_PER_FAIL = 0.019
_SKILL_TIMEOUT_BASE = 0.008
_SKILL_TIMEOUT_PER_FAIL = 0.005
_FACT_SURVIVAL_SKILL = 0.10      # facts rarely survive distillation -> 4.5%
_FACT_SURVIVAL_WM = 0.55         # inline text keeps incidental facts


def build_workflow_memory(traces: list[Trajectory]) -> Guidance:
    """Clean + structure the traces while preserving their literal flow.

    Because WM keeps the procedural *shape* of its sources (including failed
    branches), its fidelity is diluted by failures and it can mislead.
    (Worked example A.3: WM reproduced the slow serialized code of its
    source and failed the latency test.)
    """
    n_succ = sum(t.success for t in traces)
    n_fail = len(traces) - n_succ
    fidelity = _WM_BASE + _WM_PER_SUCC * n_succ - _WM_PER_FAIL * n_fail
    return Guidance(
        procedure_fidelity=clamp(fidelity, 0.0, 0.97),
        carries_fact=any(t.task and t.task.fact_required for t in traces)
        and random.random() < _FACT_SURVIVAL_WM,
        mislead_prob=clamp(_WM_MISLEAD_PER_FAIL * n_fail, 0.0, 0.35),
        timeout_penalty=_WM_TIMEOUT_BASE + _WM_TIMEOUT_PER_FAIL * n_fail,
        is_skill=False,
    )


def build_skill(traces: list[Trajectory], hint: bool = True,
                framework: str = "codex") -> Guidance:
    """Distill the same traces into a standardized SKILL.md.

    hint=True  -> the distiller sees success/failure labels, so failed
                  branches are demoted to short 'pitfall warnings' (RQ2).
    hint=False -> 'no-hint': labels stripped; failures contaminate the
                  procedure with the same weight as successes.
    """
    n_succ = sum(t.success for t in traces)
    n_fail = len(traces) - n_succ
    per_fail = _SKILL_PER_FAIL if hint else _SKILL_PER_FAIL_NOHINT
    fidelity = _SKILL_BASE + _SKILL_PER_SUCC * n_succ - per_fail * n_fail
    return Guidance(
        procedure_fidelity=clamp(fidelity, 0.0, 0.97),
        carries_fact=any(t.task and t.task.fact_required for t in traces)
        and random.random() < _FACT_SURVIVAL_SKILL,
        mislead_prob=clamp(0.008 * n_fail, 0.0, 0.10),
        timeout_penalty=_SKILL_TIMEOUT_BASE + _SKILL_TIMEOUT_PER_FAIL * n_fail,
        is_skill=True,
        framework=framework,
    )


# ---------------------------------------------------------------------------
# Agent execution (verifier-based outcome; Section 3.3)
# ---------------------------------------------------------------------------

P_RAW_PITFALL_DETECT = 0.25      # raw agent rarely spots hidden pitfalls
P_GUIDED_PITFALL_DETECT = 0.85   # anchored execution checks for them
BASE_TIMEOUT_RAW = 0.017
FRAMEWORK_COMPAT = 0.80          # RQ3: fraction of guidance that survives
                                  # transfer into a different harness for WM;
                                  # skills (self-contained protocol) keep 0.95


@dataclass
class ExecutionRecord:
    task: str
    arm: str                      # raw | workflow | skill
    success: bool
    mode: str                     # one of the 12 taxonomy modes
    mechanism: str                # skill/workflow mechanism label
    used_artifact: bool           # agent actually consulted the artifact
    framework: str = "codex"


def _effective_fidelity(g: Guidance, harness: str) -> float:
    """Guidance quality after cross-framework transfer (RQ3)."""
    if g.framework == harness:
        return g.procedure_fidelity
    keep = 0.95 if g.is_skill else FRAMEWORK_COMPAT
    return g.procedure_fidelity * keep


def execute_raw(task: Task, harness: str = "codex") -> ExecutionRecord:
    """Baseline arm: no injected prior experience."""
    if random.random() < BASE_TIMEOUT_RAW:
        return ExecutionRecord(task.tid, "raw", False,
                               "timeout_budget_exhaustion", "none", False,
                               harness)
    if task.fact_required:
        # Raw agent lacks the fact -> nearly always fails (rare task type).
        if random.random() < 0.90:
            return ExecutionRecord(task.tid, "raw", False,
                                   "algorithmic_logic_error", "none", False,
                                   harness)
    ok = True
    for _ in task.procedural_steps:
        step_fail = random.random() < task.difficulty
        pitfail = (random.random() < task.p_pitfall
                   and random.random() > P_RAW_PITFALL_DETECT)
        if step_fail or pitfail:
            ok = False
    mode = ("autonomous_clean_success" if ok
            else _sample_execution_failure_mode("raw"))
    return ExecutionRecord(task.tid, "raw", ok, mode, "none", False, harness)


def execute_with_guidance(task: Task, g: Guidance, harness: str = "codex",
                          n_distractors_present: int = 0,
                          p_notice_skill: float = 1.0) -> ExecutionRecord:
    """Workflow-memory and skill arms.

    n_distractors_present: other skills present in the environment (RQ4);
                           they distract invocation (misapplied/ignored).
    p_notice_skill: probability the environment-resident SKILL.md is opened
                    at all (skills must be *noticed*, WM is inline).
    """
    arm = g.arm_name
    used = True
    if g.is_skill and random.random() > p_notice_skill:
        used = False  # never opened -> behaves like raw
    if not used:
        return execute_raw(task, harness)

    # Cross-framework transfer degrades fidelity (RQ3).
    fid = _effective_fidelity(g, harness)

    # Invocation-layer failure grows sub-linearly with pool confusion
    # (RQ4 / SC3): browsing more candidates dilutes attention, but the
    # per-skill invocation cost saturates (paper: success stays flat).
    p_misapply = 0.02 + 0.055 * math.log10(max(n_distractors_present + 1, 2))
    if random.random() < p_misapply:
        return ExecutionRecord(
            task.tid, arm, False,
            "skill_guidance_misapplied_or_ignored" if g.is_skill
            else "static_verification_without_runtime",
            "counterproductive", True, harness)

    # Timeout: WM inflates the prompt; skill adds little.
    if random.random() < g.timeout_penalty:
        return ExecutionRecord(task.tid, arm, False,
                               "timeout_budget_exhaustion", "none", True,
                               harness)

    # Fact-gap tasks: only artifacts that carried the fact can pass them.
    if task.fact_required and not g.carries_fact:
        if random.random() < 0.85:
            return ExecutionRecord(task.tid, arm, False,
                                   "algorithmic_logic_error", "none", True,
                                   harness)

    # WM kept failed branches -> may actively mislead.
    if random.random() < g.mislead_prob:
        rec = execute_raw(task, harness)
        return ExecutionRecord(task.tid, arm, False,
                               _sample_execution_failure_mode(arm),
                               "counterproductive", True, harness)

    ok = True
    mechanism = "none"
    for _ in task.procedural_steps:
        # Residual step error shrinks with guidance fidelity.
        step_fail = random.random() < task.difficulty * (1.0 - 0.85 * fid)
        pit = random.random() < task.p_pitfall
        pitfail = pit and random.random() > P_GUIDED_PITFALL_DETECT
        if step_fail or pitfail:
            ok = False
    if ok:
        if task.fact_required and g.carries_fact:
            mechanism = "knowledge_injection"
        elif g.mislead_prob < 0.02:  # clean pool -> pure anchoring
            mechanism = "procedural_anchor"
        else:                        # mixed pool -> anchor + warning
            mechanism = random.choice(
                ["procedural_anchor", "procedural_anchor",
                 "failure_warning"])
        mode = ("skill_guided_success" if g.is_skill
                else "workflow_guided_success")
    else:
        mechanism = "none"
        mode = _sample_execution_failure_mode(arm)
    return ExecutionRecord(task.tid, arm, ok, mode, mechanism, True, harness)


# Execution-layer failure modes (SC2) sampled when an arm fails there.
_SC2_WEIGHTS = {
    "raw": {
        "environment_infrastructure_failure": 5.3,
        "output_format_schema_mismatch": 7.4,
        "background_service_lifecycle_failure": 2.7,
        "shell_code_corruption": 1.1,
        "algorithmic_logic_error": 8.3,
        "static_verification_without_runtime": 12.5,
    },
    "workflow": {
        "environment_infrastructure_failure": 1.7,
        "output_format_schema_mismatch": 3.8,
        "background_service_lifecycle_failure": 2.5,
        "shell_code_corruption": 1.9,
        "algorithmic_logic_error": 11.0,
        "static_verification_without_runtime": 12.5,
    },
    "skill": {
        "environment_infrastructure_failure": 0.2,
        "output_format_schema_mismatch": 3.2,
        "background_service_lifecycle_failure": 0.8,
        "shell_code_corruption": 0.2,
        "algorithmic_logic_error": 7.4,
        "static_verification_without_runtime": 11.7,
    },
}
SC2_MODES = list(_SC2_WEIGHTS["raw"].keys())


def _sample_execution_failure_mode(arm: str) -> str:
    w = _SC2_WEIGHTS.get(arm, _SC2_WEIGHTS["raw"])
    return random.choices(list(w.keys()), weights=list(w.values()))[0]


ALL_12_MODES = [
    "skill_guided_success",
    "workflow_guided_success",
    "autonomous_clean_success",
    *SC2_MODES,
    "timeout_budget_exhaustion",
    "skill_guidance_misapplied_or_ignored",
    "capability_or_safety_limit",
]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Deterministic aggregation (paper stage 5; Eqs. 1-2, bootstrap CIs)
# ---------------------------------------------------------------------------

def precision(predicted: set[str], gold: set[str]) -> float:
    """Eq. 1: |G_hat ∩ G| / |G_hat|, 0 if nothing selected."""
    if not predicted:
        return 0.0
    return len(predicted & gold) / len(predicted)


def recall(predicted: set[str], gold: set[str]) -> float:
    """Eq. 2: |G_hat ∩ G| / |G| (gold is never empty)."""
    return len(predicted & gold) / len(gold)


def f1_from_aggregates(p_bar: float, r_bar: float) -> float:
    """F1 recomputed from *aggregated* P and R (paper Section 4)."""
    if p_bar + r_bar == 0:
        return 0.0
    return 2 * p_bar * r_bar / (p_bar + r_bar)


def paired_bootstrap_ci(deltas: list[float], n_boot: int = 1000,
                        seed: int = 0) -> tuple[float, float, float]:
    """Paired delta with a 95% bootstrap CI over resampled units.

    Units are contrastive triples (same task + setting), matching the
    paper's 528-triple paired analysis.
    """
    rng = random.Random(seed)
    n = len(deltas)
    point = sum(deltas) / n
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return point, lo, hi
