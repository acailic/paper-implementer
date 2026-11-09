"""OmniCompass miniature evaluation pipeline (entry point).

From-scratch re-implementation of the evaluation-harness STRUCTURE of:

    OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and
    Understanding — Liu et al., arXiv:2610.18829 (2026).

Pipeline: build mini benchmark (data.py) -> generate raw model outputs for
every (instance, setting) (models.py) -> parse + score + aggregate
(metrics.py) -> Table-2/3-style report + the paper's analyses in miniature
(CS-vs-single-setting gap, QA both/one/neither decomposition, Figure-6
timestamps ablation, Figure-7 modality ablation) + oracle sanity checks.

CPU-only, deterministic, stdlib only.  Run:  python3 run.py
Writes results.json + results.txt next to this file.
"""

from __future__ import annotations

import io
import json

from data import build_benchmark
from metrics import evaluate_track
from models import roster
from schema import TEXT, TEXT_IMAGE, build_prompt, serialize_input


def collect_runs(model, instances, condition="full", timestamps=True):
    """(inst, {setting: raw_output}) for every instance of a track."""
    return [(inst, {s: model.predict(inst, s, condition=condition,
                                     timestamps=timestamps)
                    for s in (TEXT, TEXT_IMAGE)})
            for inst in instances]


def pair(a, b, fmt="{:.1f}"):
    if a is None or b is None:
        return "  N/A  "
    return f"{fmt.format(100*a)}/{fmt.format(100*b)}"


def one(v, fmt="{:.1f}"):
    return "N/A" if v is None else fmt.format(100 * v)


def render_table(reports, frames):
    """Table 2/3-style rendering: Text/Text+Image columns + CS columns."""
    head = (f"{'Model':<26}{'Fr':>3} {'TG R@0.5':>15} {'TG mtIoU':>15} "
            f"{'CS^TG':>6} {'SG AP@0.5':>15} {'SG msIoU':>15} {'CS^SG':>6} "
            f"{'QA Acc':>15} {'CS^QA':>6} {'Average':>15} {'CSAvg':>6}")
    lines = [head, "-" * len(head)]
    for model_name, rep in reports:
        tg, sg, qa = rep["TG"], rep["SG"], rep["QA"]
        lines.append(
            f"{model_name:<26}{frames[model_name]:>3} "
            f"{pair(tg['text'].get('R@0.5'), tg['text_image'].get('R@0.5')):>15} "
            f"{pair(tg['text'].get('mtIoU'), tg['text_image'].get('mtIoU')):>15} "
            f"{one(tg['cs']['CS']):>6} "
            f"{pair(sg['text'].get('AP@0.5'), sg['text_image'].get('AP@0.5')):>15} "
            f"{pair(sg['text'].get('msIoU'), sg['text_image'].get('msIoU')):>15} "
            f"{one(sg['cs']['CS']):>6} "
            f"{pair(qa['text'].get('Accuracy'), qa['text_image'].get('Accuracy')):>15} "
            f"{one(qa['cs']['CS']):>6} "
            f"{pair(rep['Average']['text'], rep['Average']['text_image']):>15} "
            f"{one(rep['CSAvg']):>6}")
    return "\n".join(lines)


def cs_gap_analysis(reports, w):
    w("\n== CS vs single-setting (paper finding: CS sits 4.3/4.6/6.4 "
      "pts below the LOWER single-setting score) ==\n")
    w(f"{'Model':<26}{'TG: CS / lower':>16}{'gap':>7}"
      f"{'SG: CS / lower':>16}{'gap':>7}"
      f"{'QA: CS / lower':>16}{'gap':>7}\n")
    for name, rep in reports:
        row = f"{name:<26}"
        for task, mean_key in (("TG", "mtIoU"), ("SG", "msIoU"),
                               ("QA", "Accuracy")):
            cs = rep[task]["cs"]["CS"]
            lows = [v for v in (rep[task]["text"].get(mean_key),
                                rep[task]["text_image"].get(mean_key))
                    if v is not None]
            if cs is None or not lows:
                row += f"{'N/A':>16}{'':>7}"
                continue
            lo = min(lows)
            row += (f"{100*cs:7.1f} / {100*lo:6.1f}{100*(cs-lo):+7.1f}")
        w(row + "\n")


def qa_decomposition(reports, w):
    w("\n== QA pair decomposition (paper App. B.3, Kimi-K3: "
      "808 pairs = 546 both + 44+43 exactly-one + 175 neither) ==\n")
    w(f"{'Model':<26}{'N':>5}{'both':>7}{'one':>6}{'neither':>9}"
      f"{'parse0':>8}\n")
    for name, rep in reports:
        d = rep["QA"]["cs"]
        w(f"{name:<26}{d['scored_pairs']:>5}{d['both_correct']:>7}"
          f"{d['exactly_one_correct']:>6}{d['neither_correct']:>9}"
          f"{d['parse_failure_pairs_scored_zero']:>8}\n")


def oracle_checks(reports, w):
    w("\n== Harness sanity checks (oracles) ==\n")
    perf = dict(reports)["oracle-perfect"]
    wrong = dict(reports)["oracle-confidently-wrong"]
    ok = []
    for task, key in (("TG", "mtIoU"), ("SG", "msIoU"), ("QA", "Accuracy")):
        for s in ("text", "text_image"):
            ok.append(abs(perf[task][s][key] - 1.0) < 1e-9)
        ok.append(abs(perf[task]["cs"]["CS"] - 1.0) < 1e-9)
    ok.append(abs(perf["CSAvg"] - 1.0) < 1e-9)
    for task, bound in (("TG", 0.05), ("SG", 0.02), ("QA", 1e-9)):
        ok.append(wrong[task]["cs"]["CS"] < bound)  # agreement alone ~ 0
    w(f"oracle-perfect: all metrics == 1.0 (incl. CS) ......... "
      f"{'PASS' if all(ok[:10]) else 'FAIL'}\n")
    w(f"oracle-confidently-wrong: identical-but-wrong -> CS~0 .. "
      f"{'PASS' if all(ok[10:]) else 'FAIL'}\n")
    return all(ok)


def main():
    out = io.StringIO()
    w = out.write
    vl, av = build_benchmark()

    w("OmniCompass miniature — from-scratch evaluation harness\n")
    w(f"(paper: Liu et al., arXiv:2610.18829; mocks only, CPU, "
      f"deterministic)\n\n")
    w("Sample serialized input (52 s basketball clip, 64 frames, VL, "
      "Text+Image):\n  "
      + serialize_input(52.0, 64, "VL", TEXT_IMAGE)[:150] + " ...\n")
    w("Sample TG prompt (Text):\n  "
      + build_prompt(vl[0], TEXT).splitlines()[1] + "\n")

    models = roster()
    frames = {m.name: m.frames for m in models}

    results = {"VL": {}, "AV": {}}
    for track_name, instances in (("VL", vl), ("AV", av)):
        reports = []
        for m in models:
            runs = collect_runs(m, instances)
            rep = evaluate_track(runs)
            results[track_name][m.name] = rep
            reports.append((m.name, rep))
        w(f"\n=== Mini OmniCompass-{track_name} "
          f"({len(instances) // 3} videos, {len(instances)} instances, "
          f"Text/Text+Image) ===\n")
        w(render_table(reports, frames) + "\n")
        cs_gap_analysis(reports, w)
        qa_decomposition(reports, w)
        w("\n")

    ok = oracle_checks([(n, results["VL"][n]) for n in results["VL"]], w)

    # --- Figure 6 miniature: frame-level timestamps on/off -----------------
    w("\n== Timestamps ablation (paper Fig. 6: CSAvg +6.5 VL / +4.9 AV, "
      "largest gains in TG) ==\n")
    w(f"{'Model':<26}{'track':>6}{'CSAvg w/ ts':>13}{'w/o ts':>9}{'delta':>8}\n")
    ts_effects = {}
    for m in roster():
        if m.name.startswith("oracle"):
            continue
        for track_name, instances in (("VL", vl), ("AV", av)):
            if track_name == "AV" and m.name in ("MockOpen-7B",):
                continue
            base = evaluate_track(collect_runs(m, instances, timestamps=True))
            off = evaluate_track(collect_runs(m, instances, timestamps=False))
            d = off["CSAvg"] - base["CSAvg"]
            ts_effects[f"{m.name}|{track_name}"] = {
                "with": base["CSAvg"], "without": off["CSAvg"], "delta": d}
            w(f"{m.name:<26}{track_name:>6}{100*base['CSAvg']:>12.1f}"
              f"{100*off['CSAvg']:>9.1f}{100*d:>+8.1f}\n")
    results["timestamps_ablation"] = ts_effects

    # --- Figure 7 miniature: input modalities on AV -------------------------
    w("\n== Modality ablation on AV (paper Fig. 7: audio lifts CS^QA by "
      "+18.6..28.1; CS^SG -0.4..+1.8; text-only CS^QA 18.8..30.1) ==\n")
    w(f"{'Model':<26}{'cond':>8}{'CS^TG':>8}{'CS^SG':>8}{'CS^QA':>8}\n")
    mod_effects = {}
    for m in roster():
        if m.name.startswith("oracle") or m.name in ("MockOpen-7B",):
            continue
        for cond in ("none", "vision", "full"):
            rep = evaluate_track(collect_runs(m, av, condition=cond))
            mod_effects[f"{m.name}|{cond}"] = {
                t: rep[t]["cs"]["CS"] for t in ("TG", "SG", "QA")}
            w(f"{m.name:<26}{cond:>8}"
              + "".join(f"{100*rep[t]['cs']['CS']:>8.1f}"
                        for t in ("TG", "SG", "QA")) + "\n")
    results["modality_ablation_av"] = mod_effects

    text = out.getvalue()
    print(text)
    with open("results.txt", "w") as f:
        f.write(text)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results.txt / results.json")
    if not ok:
        raise SystemExit("oracle sanity checks FAILED")


if __name__ == "__main__":
    main()
