"""Deterministic mock MLLMs for the OmniCompass miniature harness.

For the from-scratch re-implementation of:

    OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and
    Understanding — Liu et al., arXiv:2610.18829 (2026).

The paper evaluates 50 real MLLMs (Gemini/GPT/Seed/Qwen closed models, Kimi,
GLM, DeepSeek, TimeLens, LLaVA, ...). None of those can run on a CPU-only
box, so this module provides SEEDED, ARCHETYPAL stand-ins whose only job is
to exercise the harness end-to-end and reproduce the paper's QUALITATIVE
phenomena (grounding lags QA, CS < min single-setting, audio helps QA/TG but
not SG, text+image helps SG, parse failures score zero, text-only blind
priors score ~20%).

Two "oracle" models pin down metric semantics:
  - oracle-perfect           -> every metric should equal 1.0 (incl. all CS)
  - oracle-confidently-wrong -> IDENTICAL predictions in both settings, all
    wrong: agreement a_i = 1 but correctness g = 0, so CS ~ 0 ("agreement
    alone does not score").

Deterministic: every draw comes from a Random seeded by
(model, track, video, task, setting) — re-running never changes results.
"""

from __future__ import annotations

import json
import random

from schema import TEXT_IMAGE


class MockModel:
    """Archetypal MLLM with per-track skills and paired-setting behavior."""

    def __init__(self, name, frames, *,
                 tg_skill_vl=0.3, sg_skill_vl=0.3, qa_skill_vl=0.6,
                 tg_skill_av=0.1, sg_skill_av=0.3, qa_skill_av=0.55,
                 audio_gain_qa=0.0, audio_gain_tg=0.0,   # AV, Vision+Audio
                 tg_jitter=0.25, sg_jitter=0.16,
                 image_gain_sg=0.0, image_gain_qa=0.0, image_gain_tg=0.0,
                 indep_sigma=0.12,       # setting-specific noise on the shared
                                          # per-instance latent
                 parse_fail_rate=0.0,
                 tg_fail_shift=(0.6, 4.0),
                 consistent_when_wrong=False,
                 qa_letter_prior=None):
        self.name = name
        self.frames = frames
        self.s = dict(
            VL=dict(TG=tg_skill_vl, SG=sg_skill_vl, QA=qa_skill_vl),
            AV=dict(TG=tg_skill_av, SG=sg_skill_av, QA=qa_skill_av),
        )
        self.audio_gain = dict(QA=audio_gain_qa, TG=audio_gain_tg)
        self.tg_jitter = tg_jitter
        self.sg_jitter = sg_jitter
        self.image_gain = dict(SG=image_gain_sg, QA=image_gain_qa,
                               TG=image_gain_tg)
        self.indep_sigma = indep_sigma
        self.parse_fail_rate = parse_fail_rate
        self.tg_fail_shift = tg_fail_shift
        self.consistent_when_wrong = consistent_when_wrong
        self.qa_letter_prior = qa_letter_prior or {}

    # -- helpers ------------------------------------------------------------

    def _rng(self, inst, setting, tag=""):
        return random.Random(f"{self.name}|{inst.track}|{inst.video_id}|"
                             f"{inst.task}|{setting}|{tag}")

    def _latent(self, inst):
        """Per-instance latent difficulty, SHARED across the two settings.

        This shared latent is what makes paired outcomes correlated
        (both-correct / one-correct / neither), as in the paper's
        Appendix B.3 (Kimi-K3) decomposition.
        """
        return random.Random(f"{self.name}|{inst.track}|{inst.video_id}|"
                             f"{inst.task}|latent").random()

    def _skill(self, inst, setting, condition):
        skill = self.s[inst.track][inst.task]
        if inst.track == "AV" and condition == "full":
            skill += self.audio_gain.get(inst.task, 0.0)   # Fig. 7: audio
        if setting == TEXT_IMAGE:
            skill += self.image_gain.get(inst.task, 0.0)   # paired spec
        return min(1.0, skill)

    # -- per-task generation --------------------------------------------------

    def _tg_output(self, inst, rng, success, timestamps):
        g0, g1 = inst.tg_interval
        span = max(g1 - g0, 0.2)
        jitter = self.tg_jitter * (1.9 if not timestamps else 1.0)
        if success:
            p0 = g0 + rng.gauss(0, jitter * span)
            p1 = g1 + rng.gauss(0, jitter * span)
            p0, p1 = min(p0, p1), max(p0, p1)
            return {"start_time": round(max(0.0, p0), 2),
                    "end_time": round(max(0.1, p1), 2)}
        m = rng.uniform(*self.tg_fail_shift) * span * rng.choice((-1, 1))
        w = span * rng.uniform(0.5, 1.5)
        c = (g0 + g1) / 2 + m
        return {"start_time": round(max(0.0, c - w / 2), 2),
                "end_time": round(max(0.1, c + w / 2), 2)}

    def _sg_output(self, inst, rng, success):
        out = {}
        for k, gt in inst.sg_boxes.items():
            w, h = gt[2] - gt[0], gt[3] - gt[1]
            if success:
                j = self.sg_jitter
                box = [max(0.0, g + rng.gauss(0, j * (w if i % 2 == 0 else h)))
                       for i, g in enumerate(gt)]
            elif self.consistent_when_wrong:
                # confidently wrong: mirror the box away from the target —
                # identical in both settings (a_i = 1) but 0 IoU
                box = [1000.0 - gt[2], gt[1], 1000.0 - gt[0], gt[3]]
                if (min(box[2], gt[2]) - max(box[0], gt[0])) > 0:
                    box = [box[0], 1000.0 - gt[3], box[2], 1000.0 - gt[1]]
                if (min(box[2], gt[2]) - max(box[0], gt[0])) > 0 or \
                   (min(box[3], gt[3]) - max(box[1], gt[1])) > 0:
                    box = [0.0, 0.0, 50.0, 50.0]   # far corner, tiny box
            else:  # localize the wrong region / wrong object
                cx, cy = rng.uniform(80, 920), rng.uniform(80, 920)
                w2, h2 = w * rng.uniform(0.6, 1.6), h * rng.uniform(0.6, 1.6)
                box = [cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2]
            out[k] = [round(min(1000.0, max(0.0, v)), 2) for v in box]
        s, e = inst.tg_interval
        return {"start_time": s, "start_bbox": out["start_bbox"],
                "end_time": e, "end_bbox": out["end_bbox"]}

    def _qa_output(self, inst, rng, success, blind=False):
        letters = list(inst.qa_options)
        gt = inst.qa_answer
        if blind:                                     # text-only prior floor
            r = rng.random()
            acc = 0.0
            for letter, p in self.qa_letter_prior.items():
                acc += p
                if r < acc:
                    return letter
            return rng.choice(letters)
        if success:
            return gt
        wrong = [l for l in letters if l != gt]
        return rng.choice(wrong)

    # -- public API -----------------------------------------------------------

    def predict(self, inst, setting, condition="full", timestamps=True):
        """Raw output string for one (instance, setting, condition).

        condition: "full" (VL frames / AV frames+audio), "vision" (AV frames
        only — the Figure 7 modality ablation), "none" (text-only: both
        question wordings, no media at all).
        """
        rng = self._rng(inst, setting)
        shared = self._rng(inst, "shared")   # setting-independent draws

        if condition == "none":
            # Text-only: language priors only (paper Fig. 7: CS^TG 0.1-0.8%,
            # CS^QA 18.8-30.1%).
            if inst.task == "QA":
                return self._qa_output(inst, rng, 0.0, blind=True)
            if rng.random() < 0.02:          # lucky guess, ~0 tIoU anyway
                g0, g1 = inst.tg_interval
                c = rng.uniform(0, inst.duration_s)
                return json.dumps({"start_time": round(c, 2),
                                   "end_time": round(c + (g1 - g0), 2)})
            return json.dumps({"start_time": 0.0, "end_time": 0.0})

        if rng.random() < self.parse_fail_rate:
            # malformed output: prose without a JSON object, or empty string
            return rng.choice([
                "The event happens roughly in the middle of the clip.",
                "Based on the frames, around 40 seconds.",
                "",
                "I cannot determine this from the given input.",
            ])

        u = self._latent(inst)
        w = u + rng.gauss(0, self.indep_sigma)  # setting-specific view
        skill = self._skill(inst, setting, condition)
        success = w < skill                     # setting-level success gate
        # Confidently-wrong oracle: identical (wrong) output in both settings
        # -> high agreement a_i, zero correctness g. (Its skills are 0 and
        # the shared RNG makes both settings emit the same wrong answer.)
        r = shared if self.consistent_when_wrong else rng

        if inst.task == "TG":
            obj = self._tg_output(inst, r, success, timestamps)
        elif inst.task == "SG":
            obj = self._sg_output(inst, r, success)
        else:
            return self._qa_output(inst, r, success)
        return json.dumps(obj)

# ---------------------------------------------------------------------------
# The mock roster (names deliberately "Mock*" — no real model is evaluated;
# archetypes echo the paper's Table 2/3 model groups)
# ---------------------------------------------------------------------------

def roster():
    return [
        # Proprietary archetype (paper: Gemini-3.8-Flash): strong everywhere,
        # big audio gains, small reference-image gains on SG/QA.
        # Calibrated (loosely) to Table 2/3: VL CSAvg ~42.8, AV ~57.4.
        MockModel("MockProprietary-Flash", 64,
                  tg_skill_vl=0.45, sg_skill_vl=0.48, qa_skill_vl=0.73,
                  tg_skill_av=0.45, sg_skill_av=0.78, qa_skill_av=0.53,
                  audio_gain_qa=0.28, audio_gain_tg=0.17,
                  tg_jitter=0.14, sg_jitter=0.17,
                  image_gain_sg=0.06, image_gain_qa=0.035,
                  indep_sigma=0.05, parse_fail_rate=0.01,
                  qa_letter_prior={"B": 0.4, "C": 0.3, "A": 0.2}),
        # Open MoE archetype (paper: Kimi-K3 on VL, MiMo-V2.5 on AV): strong
        # QA, mediocre grounding, small audio gains (AV binding gap).
        MockModel("MockOpen-MoE", 64,
                  tg_skill_vl=0.38, sg_skill_vl=0.44, qa_skill_vl=0.73,
                  tg_skill_av=0.13, sg_skill_av=0.50, qa_skill_av=0.52,
                  audio_gain_qa=0.07, audio_gain_tg=0.02,
                  tg_jitter=0.16, sg_jitter=0.15,
                  image_gain_sg=0.02, image_gain_qa=0.0,
                  indep_sigma=0.09, parse_fail_rate=0.02,
                  qa_letter_prior={"C": 0.45, "B": 0.25, "D": 0.2}),
        # Small open-source archetype (paper: LLaVA / Qwen-VL 7-8B): weak.
        MockModel("MockOpen-7B", 32,
                  tg_skill_vl=0.09, sg_skill_vl=0.10, qa_skill_vl=0.56,
                  tg_skill_av=0.02, sg_skill_av=0.04, qa_skill_av=0.46,
                  audio_gain_qa=0.01, audio_gain_tg=0.0,
                  tg_jitter=0.30, sg_jitter=0.26,
                  indep_sigma=0.16, parse_fail_rate=0.04,
                  qa_letter_prior={"A": 0.5, "B": 0.3}),
        # Audio-capable open model that effectively ignores audio.
        MockModel("MockAudioBlind-8B", 32,
                  tg_skill_vl=0.24, sg_skill_vl=0.30, qa_skill_vl=0.62,
                  tg_skill_av=0.05, sg_skill_av=0.28, qa_skill_av=0.52,
                  audio_gain_qa=0.0, audio_gain_tg=0.0,
                  tg_jitter=0.27, sg_jitter=0.18,
                  indep_sigma=0.14, parse_fail_rate=0.03,
                  qa_letter_prior={"B": 0.35, "D": 0.35}),
        # Format-fragile model: high parse-failure rate -> zeros in every
        # denominator (official scoring convention stress test).
        MockModel("MockFragile-8B", 32,
                  tg_skill_vl=0.30, sg_skill_vl=0.40, qa_skill_vl=0.62,
                  tg_skill_av=0.08, sg_skill_av=0.35, qa_skill_av=0.50,
                  tg_jitter=0.22, sg_jitter=0.15,
                  indep_sigma=0.12, parse_fail_rate=0.22),
        # Harness sanity anchors -------------------------------------------
        MockModel("oracle-perfect", 64,
                  tg_skill_vl=1, sg_skill_vl=1, qa_skill_vl=1,
                  tg_skill_av=1, sg_skill_av=1, qa_skill_av=1,
                  tg_jitter=0.0, sg_jitter=0.0, indep_sigma=0.0),
        MockModel("oracle-confidently-wrong", 64,
                  tg_skill_vl=0, sg_skill_vl=0, qa_skill_vl=0,
                  tg_skill_av=0, sg_skill_av=0, qa_skill_av=0,
                  tg_jitter=0.0, sg_jitter=0.0, indep_sigma=0.0,
                  consistent_when_wrong=True),
    ]
