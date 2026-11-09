"""Metric definitions and scoring pipeline.

From-scratch re-implementation of the metrics of:

    OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and
    Understanding — Liu et al., arXiv:2610.18829 (2026).

Section 3.1 of the paper defines the per-task scores and the
Cross-Specification Consistency (CS) family; the official repo pins down the
scoring conventions (tIoU/bbox-IoU formulas, SG = mean of the two boundary
IoUs, unparseable outputs score 0 and stay in the denominator).

Deterministic, CPU-only, Python standard library only.
"""

from __future__ import annotations

from schema import TEXT, TEXT_IMAGE, parse_prediction

THRESHOLD = 0.5          # R@0.5 / AP@0.5 threshold (score >= threshold counts)
EPS = 1e-3               # SG timestamp tolerance (diagnostic only)


# ---------------------------------------------------------------------------
# Primitive overlaps (identical semantics to the official implementation)
# ---------------------------------------------------------------------------

def temporal_iou(pred, gt) -> float:
    """tIoU = overlap length / enclosing span. 0 when disjoint or empty."""
    intersection = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return intersection / union if union > 0 else 0.0


def bbox_iou(a, b) -> float:
    """Standard IoU on [x1, y1, x2, y2] boxes; degenerate boxes -> 0."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-instance task scores (paper Section 3.1 / official scoring)
# ---------------------------------------------------------------------------

def score_tg(pred, gt_interval) -> float:
    return temporal_iou(pred, gt_interval) if pred else 0.0


def score_sg(pred, gt) -> tuple:
    """SG score = mean(bbox IoU at start, bbox IoU at end).

    Returns (score, flag): flag reports a >1e-3 deviation of the echoed
    timestamps from the given interval — diagnostic only, score unaffected.
    """
    if not pred:
        return 0.0, None
    s = (bbox_iou(pred["start_bbox"], gt["start_bbox"]) +
         bbox_iou(pred["end_bbox"], gt["end_bbox"])) / 2.0
    return s, None


def score_qa(pred, gt_letter) -> float:
    """Case-insensitive exact option-letter match."""
    return 1.0 if pred is not None and str(pred).upper() == str(gt_letter).upper() else 0.0


def score_instance(task, pred, gt_interval, sg_boxes, qa_answer) -> float:
    if task == "TG":
        return score_tg(pred, gt_interval)
    if task == "SG":
        return score_sg(pred, sg_boxes)[0]
    if task == "QA":
        return score_qa(pred, qa_answer)
    raise ValueError(task)


# ---------------------------------------------------------------------------
# Single-setting aggregates (paper Tables 2-3 column semantics)
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def hit_rate(scores) -> float:
    """Fraction of scores >= THRESHOLD (R@0.5 for TG, AP@0.5 for SG).

    The paper's released code computes threshold metrics as plain
    `value >= threshold` fractions — no ranking-based average precision —
    so AP@0.5 here is a hit-rate of the mean-boundary-IoU score.
    """
    return sum(1 for s in scores if s >= THRESHOLD) / len(scores) if scores else None


def setting_metrics(task, scores) -> dict:
    """Metrics for one task under one target-specification setting."""
    if not scores:
        return {}
    m = {"mean": _mean(scores), "hit@0.5": hit_rate(scores)}
    if task == "TG":
        return {"R@0.5": m["hit@0.5"], "mtIoU": m["mean"]}
    if task == "SG":
        return {"AP@0.5": m["hit@0.5"], "msIoU": m["mean"]}
    if task == "QA":
        return {"Accuracy": m["mean"]}
    raise ValueError(task)


def overall_average(tg_mtIoU, sg_msIoU, qa_acc):
    """Paper Table 2: Average = mean(TG mtIoU, SG msIoU, QA Accuracy)."""
    vals = [v for v in (tg_mtIoU, sg_msIoU, qa_acc) if v is not None]
    return _mean(vals) if len(vals) == 3 else None


# ---------------------------------------------------------------------------
# Cross-Specification Consistency (the paper's new metric family)
# ---------------------------------------------------------------------------

def cs_tg_pair(pred_t, pred_ti, gt_interval):
    """min(gT, gTI, a): correctness under both settings and self-agreement."""
    if pred_t is None or pred_ti is None:
        return 0.0, "parse_failure"
    g_t = temporal_iou(pred_t, gt_interval)
    g_ti = temporal_iou(pred_ti, gt_interval)
    a = temporal_iou(pred_t, pred_ti)
    return min(g_t, g_ti, a), None


def cs_sg_pair(pred_t, pred_ti, gt_boxes):
    """Mean over boundaries k in {start, end} of min(gT_k, gTI_k, a_k)."""
    if pred_t is None or pred_ti is None:
        return 0.0, "parse_failure"
    vals = []
    for k in ("start_bbox", "end_bbox"):
        g_t = bbox_iou(pred_t[k], gt_boxes[k])
        g_ti = bbox_iou(pred_ti[k], gt_boxes[k])
        a = bbox_iou(pred_t[k], pred_ti[k])
        vals.append(min(g_t, g_ti, a))
    return _mean(vals), None


def cs_qa_pair(pred_t, pred_ti, gt_letter):
    """1 iff BOTH predictions equal the ground-truth letter."""
    if pred_t is None or pred_ti is None:
        return 0.0, "parse_failure"
    ok_t = str(pred_t).upper() == str(gt_letter).upper()
    ok_ti = str(pred_ti).upper() == str(gt_letter).upper()
    return (1.0 if (ok_t and ok_ti) else 0.0), None


def cs_pair(task, pred_t, pred_ti, inst):
    if task == "TG":
        return cs_tg_pair(pred_t, pred_ti, inst.tg_interval)
    if task == "SG":
        return cs_sg_pair(pred_t, pred_ti, inst.sg_boxes)
    if task == "QA":
        return cs_qa_pair(pred_t, pred_ti, inst.qa_answer)
    raise ValueError(task)


def cs_task_mean(task, pairs) -> dict:
    """CS for one task over all pairs + the diagnostic decomposition.

    pairs: list of (pred_text, pred_text_image, inst). Parse failures score
    0 and stay in the denominator (official convention). The decomposition
    generalizes the paper's Appendix B.3 (Kimi-K3) breakdown to any model.
    """
    values, n_parse_fail = [], 0
    both = one = neither = 0                    # correctness decomposition
    for pred_t, pred_ti, inst in pairs:
        v, err = cs_pair(task, pred_t, pred_ti, inst)
        values.append(v)
        if err == "parse_failure":
            n_parse_fail += 1
        if task == "QA":
            ok_t = score_qa(pred_t, inst.qa_answer)
            ok_ti = score_qa(pred_ti, inst.qa_answer)
            both += bool(ok_t and ok_ti)
            one += bool((ok_t or ok_ti) and not (ok_t and ok_ti))
            neither += bool(not ok_t and not ok_ti)
    out = {
        "CS": _mean(values),
        "scored_pairs": len(values),
        "parse_failure_pairs_scored_zero": n_parse_fail,
    }
    if task == "QA":
        out.update({"both_correct": both, "exactly_one_correct": one,
                    "neither_correct": neither})
    return out


def cs_avg(cs_tg, cs_sg, cs_qa):
    """CSAvg = mean(CS^TG, CS^SG, CS^QA)."""
    vals = [v for v in (cs_tg, cs_sg, cs_qa) if v is not None]
    return _mean(vals) if len(vals) == 3 else None


# ---------------------------------------------------------------------------
# Full scoring pipeline: raw outputs -> per-task report for one model
# ---------------------------------------------------------------------------

def evaluate_track(runs) -> dict:
    """Score one model's raw outputs on one track.

    runs: list of (inst, {setting: raw_output_string}) covering all three
    tasks for every video of the track. Returns a Table-2-style report.
    """
    report = {}
    for task in ("TG", "SG", "QA"):
        per_setting_scores = {s: [] for s in (TEXT, TEXT_IMAGE)}
        pairs = []
        for inst, raws in sorted(runs, key=lambda r: (r[0].video_id, r[0].task)):
            if inst.task != task:
                continue
            preds = {}
            for setting in (TEXT, TEXT_IMAGE):
                pred, _err = parse_prediction(raws.get(setting), task)
                preds[setting] = pred
                per_setting_scores[setting].append(
                    score_instance(task, pred, inst.tg_interval,
                                   inst.sg_boxes, inst.qa_answer))
            pairs.append((preds[TEXT], preds[TEXT_IMAGE], inst))
        m_t = setting_metrics(task, per_setting_scores[TEXT])
        m_ti = setting_metrics(task, per_setting_scores[TEXT_IMAGE])
        cs = cs_task_mean(task, pairs)
        report[task] = {"text": m_t, "text_image": m_ti, "cs": cs}

    avg_t = overall_average(report["TG"]["text"].get("mtIoU"),
                            report["SG"]["text"].get("msIoU"),
                            report["QA"]["text"].get("Accuracy"))
    avg_ti = overall_average(report["TG"]["text_image"].get("mtIoU"),
                             report["SG"]["text_image"].get("msIoU"),
                             report["QA"]["text_image"].get("Accuracy"))
    report["Average"] = {"text": avg_t, "text_image": avg_ti}
    report["CSAvg"] = cs_avg(report["TG"]["cs"]["CS"],
                             report["SG"]["cs"]["CS"],
                             report["QA"]["cs"]["CS"])
    return report
