"""Task schema, input serialization, prompts, and response parsing.

From-scratch re-implementation of the *evaluation-harness structure* of:

    OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and
    Understanding — Liu et al., arXiv:2610.18829 (2026).

Task contracts, prompt wording, input serialization order, and response
formats follow Section 3 and Appendix E of the paper (Figures E1-E6).
Written from the paper + breakdown; no reference code copied.

Deterministic, CPU-only, Python standard library only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Instance schema
# ---------------------------------------------------------------------------

TEXT = "text"            # target specified by text only
TEXT_IMAGE = "text_image"  # target specified by text + reference image

SETTINGS = (TEXT, TEXT_IMAGE)

VL_QA_TYPES = (  # 5 perception + 3 reasoning (paper: Fig. 4c, Table C3)
    "Fact Recognition", "Action Recognition", "Attribute Perception",
    "Visual Counting", "Spatial Perception",          # perception
    "Temporal Reasoning", "Causal Reasoning", "Comparative Judgment",  # reasoning
)
AV_QA_TYPES = (  # 3 perception + 2 reasoning (paper: Table C4)
    "Speech Understanding", "Event Understanding", "Emotion Recognition",
    "Temporal Reasoning", "Causal Reasoning",
)
REF_CUE_TYPES = ("entity", "action", "attribute", "scene")


@dataclass
class TargetSpec:
    """Paired target specification: same video, same target, same GT."""
    text_query: str                 # e.g. "USA's No. 33 player"
    image_query: str                # e.g. "the player in the reference image"
    cue_type: str = "entity"        # entity | action | attribute | scene


@dataclass
class Instance:
    """One (video, task) instance, evaluated under BOTH paired settings."""
    track: str                      # "VL" | "AV"
    video_id: str
    domain: str                     # one of 7 OmniCompass domains
    duration_s: float
    duration_bin: str               # short (<90s) | medium (<300s) | long (>=300s)
    task: str                       # "TG" | "SG" | "QA"
    spec: TargetSpec
    # --- ground truth, shared across the two settings ---------------------
    tg_interval: tuple = None       # (start_s, end_s)  -- also SG/QA clue interval
    sg_boxes: dict = None           # {"start_bbox": [...], "end_bbox": [...]} 0-1000
    qa_question_text: str = None    # full question, Text setting
    qa_question_image: str = None   # full question, Text+Image setting
    qa_options: dict = None         # {"A": "...", ..., "E": "..."} (scrambled order)
    qa_answer: str = None           # correct letter
    qa_type: str = None             # QA category (VL 8 types / AV 5 types)

    def question_for(self, setting: str, task: str) -> str:
        """The question text under a given target-specification setting."""
        if task == "QA":
            return (self.qa_question_text if setting == TEXT
                    else self.qa_question_image)
        return (self.spec.text_query if setting == TEXT
                else self.spec.image_query)


def duration_bin(duration_s: float) -> str:
    """Paper bins: short 30-90s, medium 90-300s, long >=300s."""
    if duration_s < 90:
        return "short"
    if duration_s < 300:
        return "medium"
    return "long"


# ---------------------------------------------------------------------------
# Input serialization (Appendix E.1) and prompts (Figures E1-E6)
# ---------------------------------------------------------------------------

def sample_frame_timestamps(duration_s: float, k: int) -> list:
    """K uniformly sampled timestamps, i * duration / K, paper-style text."""
    return [round(i * duration_s / k, 3) for i in range(k)]


def serialize_input(duration_s: float, k: int, track: str, setting: str) -> str:
    """Timestamp/frame pairs -> (audio on AV) -> (ref image in Text+Image) -> prompt.

    Angle-bracketed items are media placeholders (no literal tokens), exactly
    as in Appendix E.1.  Arrows show input order.
    """
    ts = sample_frame_timestamps(duration_s, k)
    parts = [f"Timestamp: {t} seconds. <frame {i}>" for i, t in enumerate(ts)]
    if track == "AV":
        parts.append("<audio>")
    if setting == TEXT_IMAGE:
        parts.append("<ref image>")
    parts.append("[Prompt]")
    return " → ".join(parts)


def _preamble(duration_s: float, track: str, setting: str) -> str:
    dur = f"{duration_s:g}"
    if track == "AV":
        base = (f"You are given a sequence of frames uniformly sampled across a "
                f"{dur}-second video and the corresponding audio")
        task_from = "using the video frames and audio"
    else:
        base = (f"You are given a sequence of frames uniformly sampled across a "
                f"{dur}-second video")
        task_from = "based on the visual content of these video frames"
    if setting == TEXT_IMAGE:
        base += ", followed by a reference image"
        if track == "AV":
            task_from = "using the video frames, audio, and reference image"
        else:
            task_from = "based on the visual content of the video frames and the reference image"
    return (f"{base}. Each frame is preceded by its timestamp in seconds. "
            f"Complete the task {task_from}.")


def build_prompt(inst: Instance, setting: str, k: int = 64) -> str:
    """Task prompt exactly in the shape of Appendix E, Figures E1-E6."""
    head = _preamble(inst.duration_s, inst.track, setting)
    q = inst.question_for(setting, inst.task)

    if inst.task == "TG":
        return (
            f"{head}\n"
            f"Question: When in the video did {q}?\n"
            "Output format: Answer with only this JSON object, where "
            "start_time and end_time are measured in seconds from the "
            "beginning of the video:\n"
            '{"start_time": <number>,\n"end_time": <number>}'
        )

    if inst.task == "SG":
        s, e = inst.tg_interval
        return (
            f"{head}\n"
            f"Question: Where in the frame is {q} during this event?\n"
            f"The relevant interval starts at {s:g} seconds and ends at {e:g} "
            "seconds. Locate the target described in the question at each of "
            "these two timestamps and report one bounding box for each frame.\n"
            "Write each box in [x1, y1, x2, y2] order, corresponding to "
            "[left, top, right, bottom]. Normalize all coordinates from 0 to "
            "1000, with the top-left of the frame as the origin.\n"
            "Output format: Return only this JSON object:\n"
            '{"start_time": %g,\n"start_bbox": [<x1>, <y1>, <x2>, <y2>],\n'
            '"end_time": %g,\n"end_bbox": [<x1>, <y1>, <x2>, <y2>]}' % (s, e)
        )

    if inst.task == "QA":
        opts = "\n".join(f"{letter}. {text}"
                         for letter, text in inst.qa_options.items())
        return (
            f"{head}\n"
            f"Question: {q}\n"
            f"{opts}\n"
            "You must choose the most appropriate answer from the five "
            "options (A, B, C, D, or E). If uncertain, make your best choice "
            "based on the provided inputs. Output only the option letter."
        )

    raise ValueError(f"unknown task {inst.task}")


# ---------------------------------------------------------------------------
# Response parsing (official convention: unparseable -> None, scores 0,
# stays in the denominator)
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*?\}|\{.*\}", re.S)
_NUM = r"-?\d+(?:\.\d+)?"


def _extract_json(raw: str):
    """Best-effort JSON object extraction from a raw model output."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    for m in _JSON_RE.finditer(text):          # tolerate surrounding prose
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def parse_prediction(raw: str, task: str):
    """Parse a raw response into a normalized prediction.

    Returns (prediction, parse_error). prediction is None on failure.
    TG -> [start, end]; SG -> {"start_bbox", "end_bbox"}; QA -> "B".
    Mirrors omnicompass_scoring.py: accepts start_time or start_sec keys;
    empty output is an error; everything else degrades to None.
    """
    if raw is None or not str(raw).strip():
        return None, "empty_output"

    if task == "QA":
        # Official scoring compares str(parsed).upper() to the GT letter.
        # The instructed output is a bare option letter (no JSON object), so
        # search the raw text; prose without a letter is a parse failure.
        text = str(raw).strip().upper()
        m = re.search(r"\b([A-E])\b", text)
        if m:
            return m.group(1), None
        return None, "no_option_letter"

    obj = _extract_json(str(raw))
    if obj is None:
        return None, "no_json_object"

    if task == "TG":
        s = obj.get("start_time", obj.get("start_sec"))
        e = obj.get("end_time", obj.get("end_sec"))
        if not (isinstance(s, (int, float)) and isinstance(e, (int, float))):
            return None, "missing_start_end"
        return [float(s), float(e)], None

    if task == "SG":
        sb, eb = obj.get("start_bbox"), obj.get("end_bbox")
        ok = all(isinstance(b, (list, tuple)) and len(b) == 4 and
                 all(isinstance(v, (int, float)) for v in b) for b in (sb, eb))
        if not ok:
            return None, "missing_or_malformed_boxes"
        return {"start_bbox": [float(v) for v in sb],
                "end_bbox": [float(v) for v in eb]}, None

    raise ValueError(f"unknown task {task}")
