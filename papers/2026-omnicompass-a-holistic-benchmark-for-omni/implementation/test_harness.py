"""Unit tests for the OmniCompass miniature harness.

For the from-scratch re-implementation of:

    OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and
    Understanding — Liu et al., arXiv:2610.18829 (2026).

Run:  python3 test_harness.py      (stdlib unittest, no dependencies)
"""

from __future__ import annotations

import random
import unittest

import metrics
from metrics import (bbox_iou, cs_avg, cs_task_mean, cs_tg_pair,
                     cs_qa_pair, evaluate_track, hit_rate, overall_average,
                     score_instance, score_qa, score_sg, score_tg,
                     setting_metrics, temporal_iou)
from schema import Instance, TargetSpec, parse_prediction, build_prompt, TEXT
from data import build_benchmark


def _inst(**kw):
    """Minimal QA/TG/SG instance for hand-computed tests."""
    base = dict(track="VL", video_id="v1", domain="Sports", duration_s=100.0,
                duration_bin="medium", task="QA",
                spec=TargetSpec("the player dunk", "the referenced player dunk"),
                tg_interval=(40.0, 50.0),
                sg_boxes={"start_bbox": [100, 100, 300, 300],
                          "end_bbox": [150, 150, 350, 350]},
                qa_question_text="q?", qa_question_image="q'?",
                qa_options={"A": "x", "B": "y", "C": "z", "D": "w", "E": "v"},
                qa_answer="B", qa_type="Fact Recognition")
    base.update(kw)
    return Instance(**base)


class TestOverlaps(unittest.TestCase):
    def test_temporal_iou_hand_values(self):
        self.assertEqual(temporal_iou([0, 10], [0, 10]), 1.0)      # identical
        self.assertEqual(temporal_iou([0, 10], [20, 30]), 0.0)    # disjoint
        self.assertAlmostEqual(temporal_iou([0, 10], [5, 15]), 5 / 15)
        self.assertAlmostEqual(temporal_iou([0, 10], [0, 20]), 0.5)  # contain
        self.assertEqual(temporal_iou([5, 5], [5, 5]), 0.0)       # empty span

    def test_bbox_iou_hand_values(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        # inter 50, union 100+50-50 = 100 -> 1/2
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 5]), 0.5)
        self.assertEqual(bbox_iou([0, 0, 5, 5], [0, 0, 0, 0]), 0.0)  # degenerate


class TestScoring(unittest.TestCase):
    def test_tg_score(self):
        self.assertEqual(score_tg([40, 50], (40, 50)), 1.0)
        self.assertEqual(score_tg(None, (40, 50)), 0.0)      # parse failure

    def test_sg_score_is_boundary_mean(self):
        gt = {"start_bbox": [0, 0, 100, 100], "end_bbox": [0, 0, 100, 100]}
        pred = {"start_bbox": [0, 0, 100, 100],     # IoU 1
                "end_bbox": [200, 200, 300, 300]}   # IoU 0
        self.assertAlmostEqual(score_sg(pred, gt)[0], 0.5)
        self.assertEqual(score_sg(None, gt)[0], 0.0)

    def test_qa_case_insensitive(self):
        self.assertEqual(score_qa("b", "B"), 1.0)
        self.assertEqual(score_qa("A", "B"), 0.0)
        self.assertEqual(score_qa(None, "B"), 0.0)


class TestParser(unittest.TestCase):
    def test_tg_json(self):
        pred, err = parse_prediction('{"start_time": 48.0, "end_time": 50.0}',
                                     "TG")
        self.assertEqual((pred, err), ([48.0, 50.0], None))

    def test_tg_sec_alias_and_prose(self):
        pred, err = parse_prediction('{"start_sec": 1, "end_sec": 2}', "TG")
        self.assertEqual(pred, [1.0, 2.0])
        pred, err = parse_prediction("It happens near the end.", "TG")
        self.assertIsNone(pred)
        self.assertEqual(err, "no_json_object")

    def test_tg_fenced_json_with_prose(self):
        raw = '```json\n{"start_time": 3, "end_time": 4}\n```'
        self.assertEqual(parse_prediction(raw, "TG")[0], [3.0, 4.0])
        raw = 'Sure! {"start_time": 3, "end_time": 4} hope this helps'
        self.assertEqual(parse_prediction(raw, "TG")[0], [3.0, 4.0])

    def test_sg_malformed_boxes(self):
        pred, err = parse_prediction(
            '{"start_bbox": [1, 2], "end_bbox": [0, 0, 1, 1]}', "SG")
        self.assertIsNone(pred)

    def test_qa_letter_extraction(self):
        self.assertEqual(parse_prediction("B", "QA")[0], "B")
        self.assertEqual(parse_prediction("The answer is C.", "QA")[0], "C")
        self.assertIsNone(parse_prediction("no letter here", "QA")[0])

    def test_empty_output(self):
        self.assertEqual(parse_prediction("  ", "TG"),
                         (None, "empty_output"))


class TestAggregates(unittest.TestCase):
    def test_setting_metrics_keys(self):
        self.assertEqual(setting_metrics("TG", [0.6, 0.2]),
                         {"R@0.5": 0.5, "mtIoU": 0.4})
        self.assertEqual(setting_metrics("SG", [0.6, 0.4]),
                         {"AP@0.5": 0.5, "msIoU": 0.5})
        self.assertEqual(setting_metrics("QA", [1.0, 0.0]),
                         {"Accuracy": 0.5})

    def test_hit_rate_threshold_is_inclusive(self):
        self.assertEqual(hit_rate([0.5, 0.4999]), 0.5)   # >= threshold

    def test_paper_average_identity(self):
        # Table 2, Gemini-3.8-Flash: (32.6 + 37.7 + 73.3)/3 = 47.87 ~ 47.9
        avg = overall_average(0.326, 0.377, 0.733)
        self.assertAlmostEqual(avg, 0.47866666, places=5)


class TestCS(unittest.TestCase):
    def test_cs_tg_min_semantics(self):
        gt = (40.0, 50.0)
        v, _ = cs_tg_pair([40, 50], [40, 50], gt)   # both perfect, agree
        self.assertEqual(v, 1.0)
        # gT = 8/10 = .8, gTI = 6/10 = .6, a = 6/8 = .75 -> min = gTI
        v, _ = cs_tg_pair([42, 50], [44, 50], gt)
        self.assertAlmostEqual(v, 0.6)
        self.assertAlmostEqual(temporal_iou([42, 50], gt), 0.8)
        self.assertAlmostEqual(temporal_iou([44, 50], gt), 0.6)
        self.assertAlmostEqual(temporal_iou([42, 50], [44, 50]), 0.75)

    def test_cs_agreement_alone_scores_zero(self):
        gt = (40.0, 50.0)
        # identical WRONG predictions: a = 1, g = 0 -> CS = 0
        v, _ = cs_tg_pair([70, 80], [70, 80], gt)
        self.assertEqual(v, 0.0)
        v, _ = cs_qa_pair("C", "C", "B")
        self.assertEqual(v, 0.0)

    def test_cs_qa_both_correct_required(self):
        self.assertEqual(cs_qa_pair("B", "B", "B")[0], 1.0)
        self.assertEqual(cs_qa_pair("B", "A", "B")[0], 0.0)

    def test_cs_parse_failure_zero_in_denominator(self):
        inst = _inst()
        out = cs_task_mean("QA", [(None, "B", inst), ("B", "B", inst)])
        self.assertEqual(out["CS"], 0.5)                 # 1 of 2 pairs
        self.assertEqual(out["scored_pairs"], 2)         # none excluded
        self.assertEqual(out["parse_failure_pairs_scored_zero"], 1)

    def test_qa_decomposition_sums_to_n(self):
        inst = _inst()
        pairs = [(None, "B", inst), ("B", "B", inst), ("B", "A", inst),
                 ("A", "A", inst)]
        out = cs_task_mean("QA", pairs)
        self.assertEqual(out["both_correct"] + out["exactly_one_correct"]
                         + out["neither_correct"], out["scored_pairs"])

    def test_cs_avg(self):
        self.assertAlmostEqual(cs_avg(0.2, 0.4, 0.6), 0.4)

    def test_cs_never_exceeds_lower_single_setting_mean(self):
        """Property test: CS(task) <= min(setting means) on random data."""
        rng = random.Random(0)
        for _ in range(200):
            inst = _inst(task="TG", tg_interval=(10.0, 20.0))
            runs = []
            t_scores = {"text": [], "text_image": []}
            for i in range(12):
                preds = {}
                for s in ("text", "text_image"):
                    if rng.random() < 0.15:
                        pred = None
                    elif rng.random() < 0.6:
                        c = rng.uniform(8, 22)
                        w = rng.uniform(0.5, 8)
                        pred = [max(0, c - w / 2), c + w / 2]
                    else:
                        pred = [10, 20]
                    preds[s] = pred
                    t_scores[s].append(
                        score_tg(pred, inst.tg_interval))
                runs.append((inst, preds))
            rep = evaluate_track(runs)["TG"]
            means = [metrics._mean(t_scores[s])
                     for s in ("text", "text_image")]
            self.assertLessEqual(rep["cs"]["CS"], min(means) + 1e-9)


class TestEndToEnd(unittest.TestCase):
    def test_benchmark_shape_and_stats(self):
        vl, av = build_benchmark()
        self.assertEqual(len(vl), 117)     # 39 videos x 3 tasks
        self.assertEqual(len(av), 30)      # 10 videos x 3 tasks
        # every video has all three tasks with a shared GT interval
        by_video = {}
        for i in vl + av:
            by_video.setdefault((i.track, i.video_id), set()).add(i.task)
        self.assertTrue(all(v == {"TG", "SG", "QA"} for v in by_video.values()))
        # duration bins cover all three classes
        bins = {i.duration_bin for i in vl}
        self.assertEqual(bins, {"short", "medium", "long"})

    def test_prompt_contains_paper_format(self):
        vl, _ = build_benchmark()
        tg_prompt = build_prompt(vl[0], TEXT)
        self.assertIn('"start_time"', tg_prompt)
        self.assertIn("Question: When in the video did", tg_prompt)
        qa = next(i for i in vl if i.task == "QA")
        qa_prompt = build_prompt(qa, TEXT)
        self.assertIn("Output only the option letter", qa_prompt)
        self.assertEqual(len(qa.qa_options), 5)

    def test_score_instance_dispatch(self):
        inst = _inst()
        self.assertEqual(
            score_instance("QA", "B", None, None, "B"), 1.0)
        self.assertEqual(
            score_instance("TG", [40, 50], (40, 50), None, None), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
