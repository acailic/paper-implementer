"""Hand-built miniature OmniCompass benchmark.

For the from-scratch re-implementation of:

    OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and
    Understanding — Liu et al., arXiv:2610.18829 (2026).

The real benchmark (820 VL videos / 240 AV videos, 3,180 instances, 80.6 h
of video) cannot ship inside a code repo. This module builds a structurally
faithful miniature: 39 VL videos + 10 AV videos, one TG/SG/QA instance each,
paired Text/Text+Image specifications of the SAME target, with the paper's
statistics respected in miniature:

  - 7 domains, 3 duration bins (short <90s, medium <300s, long >=300s)
  - TG spans ~4.6% (VL) / ~1.7% (AV) of video duration
  - SG boxes ~13.2% (VL) / ~30.7% (AV) of the 1000x1000 normalized frame
  - reference cue mix ~80% entity (paper: 79.7/7.0/8.1/5.2 %)
  - 5-choice QA, scrambled option order (paper Fig. E3: A 0, B 2, C 3, D 4, E 1)
  - VL QA: 8 categories (5 perception / 3 reasoning); AV QA: 5 (3/2)

VL videos 1-6 and AV videos 1-3 are hand-written, mirroring the paper's own
Figures E1-E6 examples (the 52 s basketball clip, the 270 s news clip);
the rest are hand-phrased procedural filler. Fully deterministic.
"""

from __future__ import annotations

import random

from schema import Instance, TargetSpec, duration_bin

RNG = random.Random(2610188)   # seed from the arXiv id

DOMAINS = ("Sports", "Traffic", "Safety", "Life", "Film", "Variety", "News")

# (video_id, domain, dur_s, text query, text+image query, cue, TG span %, QA type)
_VL_FILLER = [
    ("soccer_counter_04", "Sports", 213.0,
     "the striker in the black kit volley home the rebound",
     "the striker shown in the reference image volley home the rebound",
     "entity", 3.8, "Action Recognition"),
    ("marathon_finish_11", "Sports", 468.0,
     "the lead runner in the orange singlet cross the finish line",
     "the runner shown in the reference image cross the finish line",
     "entity", 1.6, "Temporal Reasoning"),
    ("intersection_violation_02", "Traffic", 61.0,
     "the silver sedan run the red light at the north approach",
     "the sedan shown in the reference image run the red light",
     "attribute", 6.9, "Causal Reasoning"),
    ("highway_merge_17", "Traffic", 342.0,
     "the blue box truck force its way into the left lane",
     "the truck shown in the reference image force its way into the left lane",
     "attribute", 2.1, "Comparative Judgment"),
    ("cctv_breakin_03", "Safety", 118.0,
     "the hooded figure pry open the side door",
     "the person shown in the reference image pry open the side door",
     "entity", 5.5, "Causal Reasoning"),
    ("platform_theft_09", "Safety", 512.0,
     "the man in the grey hoodie grab the unattended backpack",
     "the person shown in the reference image grab the unattended backpack",
     "entity", 0.9, "Temporal Reasoning"),
    ("kitchen_souschef_05", "Life", 97.0,
     "the chef flambé the mushrooms in the copper pan",
     "the cook shown in the reference image flambé the mushrooms",
     "action", 7.4, "Attribute Perception"),
    ("market_haggling_12", "Life", 388.0,
     "the vendor in the green apron wrap the watermelon",
     "the vendor shown in the reference image wrap the watermelon",
     "entity", 1.2, "Fact Recognition"),
    ("noir_interrogation_06", "Film", 517.0,
     "the detective in the trench coat slam the table",
     "the detective shown in the reference image slam the table",
     "entity", 1.1, "Causal Reasoning"),
    ("romance_rooftop_14", "Film", 156.0,
     "the rooftop tablecloth blow off in the gust",
     "the scene shown in the reference image recur during the storm",
     "scene", 4.2, "Comparative Judgment"),
    ("courtroom_verdict_15", "Film", 641.0,
     "the foreman read out the guilty verdict",
     "the man shown in the reference image read out the verdict",
     "entity", 0.8, "Fact Recognition"),
    ("broadcast_storm_16", "News", 254.0,
     "the rain-lashed reporter signal to the camera crew",
     "the reporter shown in the reference image signal to the crew",
     "entity", 2.7, "Action Recognition"),
    ("exit_poll_18", "News", 512.0,
     "the anchor announce the first exit polls",
     "the anchor shown in the reference image announce the exit polls",
     "entity", 1.0, "Fact Recognition"),
    ("dragon_boat_20", "Sports", 302.0,
     "the drummer in the red headband lose the rhythm",
     "the drummer shown in the reference image lose the rhythm",
     "entity", 2.4, "Temporal Reasoning"),
    ("tram_descent_22", "Traffic", 143.0,
     "the vintage green tram descend the steep hill",
     "the tram shown in the reference image descend the hill",
     "attribute", 3.3, "Spatial Perception"),
    ("night_market_23", "Life", 512.0,
     "the lantern seller light the paper lamps",
     "the seller shown in the reference image light the lamps",
     "action", 0.9, "Action Recognition"),
    ("gallery_heist_25", "Safety", 277.0,
     "the guard leave the surveillance desk unattended",
     "the guard shown in the reference image leave the desk",
     "entity", 2.2, "Causal Reasoning"),
    ("debate_rebuttal_26", "Variety", 371.0,
     "the moderator cut off the shouting guest",
     "the guest shown in the reference image get cut off",
     "entity", 1.5, "Temporal Reasoning"),
    ("arena_penalty_28", "Sports", 95.0,
     "the goalkeeper in the neon gloves save the penalty",
     "the goalkeeper shown in the reference image save the penalty",
     "attribute", 8.1, "Action Recognition"),
    ("subway_busker_29", "Life", 88.0,
     "the busker pack up the guitar mid-song",
     "the busker shown in the reference image pack up",
     "entity", 7.7, "Causal Reasoning"),
    ("velodrome_sprint_31", "Sports", 189.0,
     "the rider in the rainbow helmet begin the final sprint",
     "the rider shown in the reference image begin the sprint",
     "entity", 3.1, "Temporal Reasoning"),
    ("ferry_docking_32", "Traffic", 421.0,
     "the white ferry nudge against the pier",
     "the ferry shown in the reference image dock",
     "attribute", 1.1, "Action Recognition"),
    ("warehouse_forklift_36", "Safety", 233.0,
     "the forklift driver drop the pallet",
     "the driver shown in the reference image drop the pallet",
     "entity", 2.6, "Causal Reasoning"),
    ("cooking_final_37", "Life", 302.0,
     "the chef plate the finished dish",
     "the chef shown in the reference image plate the dish",
     "action", 1.7, "Action Recognition"),
    ("silent_film_chase_38", "Film", 174.0,
     "the bumbling constable trip over the crate",
     "the constable shown in the reference image trip",
     "entity", 4.4, "Attribute Perception"),
    ("awards_envelope_39", "Variety", 264.0,
     "the presenter open the wrong envelope",
     "the presenter shown in the reference image open the envelope",
     "entity", 2.9, "Fact Recognition"),
    ("weather_front_40", "News", 118.0,
     "the meteorologist point at the incoming front",
     "the meteorologist shown in the reference image point at the front",
     "entity", 5.2, "Spatial Perception"),
    ("penalty_shootout_41", "Sports", 511.0,
     "the substitute step up for the deciding kick",
     "the player shown in the reference image step up",
     "entity", 0.8, "Comparative Judgment"),
    ("tollgate_u_turn_42", "Traffic", 79.0,
     "the camper van reverse out of the toll lane",
     "the van shown in the reference image reverse",
     "attribute", 7.9, "Causal Reasoning"),
    ("street_performer_43", "Life", 63.0,
     "the mime pretend to lean on an invisible wall",
     "the performer shown in the reference image lean",
     "action", 9.3, "Attribute Perception"),
    ("archive_footage_44", "News", 620.0,
     "the black-and-white clip cut to the crowd",
     "the scene shown in the reference image cut to the crowd",
     "scene", 0.7, "Comparative Judgment"),
    ("stunt_double_45", "Film", 288.0,
     "the stunt double leap between the rooftops",
     "the person shown in the reference image leap",
     "entity", 2.0, "Action Recognition"),
    ("quiz_buzzer_46", "Variety", 137.0,
     "the challenger hit the buzzer first",
     "the challenger shown in the reference image hit the buzzer",
     "entity", 4.8, "Temporal Reasoning"),
]

# (video_id, domain, dur_s, text query, text+image query, TG span %, QA type)
_AV_FILLER = [
    ("variety_guesstune_21", "Variety", 288.0,
     "the contestant in the yellow jacket attempt the high note",
     "the contestant shown in the reference image attempt the high note",
     1.9, "Speech Understanding"),
    ("wildfire_report_24", "News", 361.0,
     "the field reporter mention the evacuated hamlet",
     "the reporter shown in the reference image mention the evacuated hamlet",
     1.3, "Event Understanding"),
    ("courtroom_outburst_27", "Film", 422.0,
     "the witness raise her voice at the prosecutor",
     "the witness shown in the reference image raise her voice",
     1.6, "Emotion Recognition"),
    ("wildlife_dawn_30", "Life", 333.0,
     "the offscreen howl of the alpha wolf startle the herd",
     "the animal making the sound heard in the reference clip startle the herd",
     1.1, "Event Understanding"),
    ("talkshow_anecdote_33", "Variety", 244.0,
     "the host finish the story about the lost passport",
     "the host shown in the reference image finish the story",
     2.0, "Speech Understanding"),
    ("orchestra_tuning_34", "Film", 517.0,
     "the first violinist tap the stand and begin the tuning",
     "the musician shown in the reference image begin the tuning",
     1.2, "Event Understanding"),
    ("rescue_call_35", "Safety", 196.0,
     "the dispatcher confirm the address with the caller",
     "the dispatcher heard in the reference clip confirm the address",
     2.3, "Speech Understanding"),
]


def _box(cx, cy, w, h):
    """Normalized [x1,y1,x2,y2] box in 0-1000 space, clipped to frame."""
    return [max(0.0, cx - w / 2), max(0.0, cy - h / 2),
            min(1000.0, cx + w / 2), min(1000.0, cy + h / 2)]


def _qa(rng, q_text, q_image, correct, distractors, qa_type):
    """5-choice QA, scrambled option order (paper: A 0, B 2, C 3, D 4, E 1)."""
    texts = [correct] + list(distractors)
    rng.shuffle(texts)
    options = dict(zip(["A", "B", "C", "D", "E"], texts))
    answer = next(l for l, t in options.items() if t == correct)
    return q_text, q_image, options, answer, qa_type


def _video_instances(track, vid, domain, dur, text_q, image_q, cue,
                     span_pct, qa_type, sg_area, rng):
    """One video -> three paired instances (TG, SG, QA), same target + GT."""
    span = dur * span_pct / 100.0
    start = rng.uniform(0.1 * dur, max(0.11 * dur, 0.9 * dur - span))
    interval = (round(start, 2), round(start + span, 2))

    w = 1000.0 * (sg_area ** 0.5) * rng.uniform(0.75, 1.3)
    h = w * rng.uniform(0.75, 1.45)
    cx, cy = rng.uniform(180, 820), rng.uniform(180, 820)
    sb = [round(v, 2) for v in _box(cx, cy, w, h)]
    eb = [round(v, 2) for v in _box(cx + rng.uniform(-45, 45),
                                    cy + rng.uniform(-35, 35),
                                    w * rng.uniform(0.85, 1.2),
                                    h * rng.uniform(0.85, 1.2))]

    n = rng.randint(2, 6)
    q_t = f"How many times does {text_q} happen in total in the video?"
    q_i = f"How many times does {image_q} happen in total in the video?"
    q_t, q_i, options, answer, qa_type = _qa(
        rng, q_t, q_i, f"{n} times",
        [f"{n + d} times" for d in (-2, -1, 1, 2)], qa_type)

    spec = TargetSpec(text_query=text_q, image_query=image_q, cue_type=cue)
    common = dict(track=track, video_id=vid, domain=domain, duration_s=dur,
                  duration_bin=duration_bin(dur), spec=spec,
                  tg_interval=interval,
                  sg_boxes={"start_bbox": sb, "end_bbox": eb},
                  qa_question_text=q_t, qa_question_image=q_i,
                  qa_options=options, qa_answer=answer, qa_type=qa_type)
    return [Instance(task=t, **common) for t in ("TG", "SG", "QA")]


# ---------------------------------------------------------------------------
# Hand-written instances (mirroring the paper's Figure E1-E6 examples)
# ---------------------------------------------------------------------------

def _handwritten_vl():
    insts = []

    # 1. The paper's running example (52 s basketball clip, Figs. E1-E3);
    #    GT interval and boxes copied from the paper's own annotations.
    spec = TargetSpec("Team USA player No. 7 score her first basket",
                      "the player in the reference image score her first basket")
    base = dict(track="VL", video_id="basketball_firstbasket_01", domain="Sports",
                duration_s=52.0, duration_bin="short", spec=spec,
                tg_interval=(48.0, 50.0),
                sg_boxes={"start_bbox": [587.61, 473.21, 628.53, 666.34],
                          "end_bbox": [644.16, 536.71, 694.75, 833.00]})
    insts += [
        Instance(task="TG", **base),
        Instance(task="SG", **base),
        Instance(task="QA", **{**base, **dict(
            qa_question_text="How many points does Team USA player No. 7 "
                             "score in total in the video?",
            qa_question_image="How many points does the player in the "
                              "reference image score in total in the video?",
            qa_options={"A": "0 points", "B": "2 points", "C": "3 points",
                        "D": "4 points", "E": "1 point"},
            qa_answer="B", qa_type="Visual Counting")}),
    ]

    # 2. Crosswalk near-miss (Traffic, short).
    spec = TargetSpec("the delivery rider in the red helmet swerve to "
                      "avoid the pedestrian",
                      "the delivery rider shown in the reference image "
                      "swerve to avoid the pedestrian")
    base = dict(track="VL", video_id="crosswalk_nearmiss_07", domain="Traffic",
                duration_s=71.0, duration_bin="short", spec=spec,
                tg_interval=(31.40, 35.20),
                sg_boxes={"start_bbox": [402.10, 512.44, 588.02, 790.15],
                          "end_bbox": [455.87, 540.10, 641.55, 802.71]})
    insts += [
        Instance(task="TG", **base),
        Instance(task="SG", **base),
        Instance(task="QA", **{**base, **dict(
            qa_question_text="What does the delivery rider in the red helmet "
                             "do to avoid the pedestrian?",
            qa_question_image="What does the rider shown in the reference "
                              "image do to avoid the pedestrian?",
            qa_options={"A": "Swerves left", "B": "Stops completely",
                        "C": "Dismounts", "D": "Speeds up", "E": "Swerves right"},
            qa_answer="A", qa_type="Causal Reasoning")}),
    ]

    # 3. Kitchen flare (Safety, medium) — attribute-conditioned reference.
    spec = TargetSpec("the pan with the blackened handle catch fire",
                      "the pan shown in the reference image catch fire")
    base = dict(track="VL", video_id="kitchen_flare_02", domain="Safety",
                duration_s=141.0, duration_bin="medium", spec=spec,
                tg_interval=(88.10, 92.60),
                sg_boxes={"start_bbox": [310.52, 480.20, 505.33, 660.88],
                          "end_bbox": [330.10, 455.66, 528.91, 688.02]})
    insts += [
        Instance(task="TG", **base),
        Instance(task="SG", **base),
        Instance(task="QA", **{**base, **dict(
            qa_question_text="Why does the pan with the blackened handle "
                             "catch fire?",
            qa_question_image="Why does the pan shown in the reference image "
                              "catch fire?",
            qa_options={"A": "Gas leak", "B": "Overheated oil",
                        "C": "Faulty wiring", "D": "Paper towel nearby",
                        "E": "It does not catch fire"},
            qa_answer="B", qa_type="Causal Reasoning")}),
    ]

    # 4. News anchor (News, medium) — the paper's 270 s clip, VL variant.
    spec = TargetSpec("the news anchor with long brown curly hair, a "
                      "black-and-white houndstooth jacket, and a black "
                      "turtleneck begin the report on France and Africa",
                      "the news anchor in the reference image begin the "
                      "report on France and Africa")
    base = dict(track="VL", video_id="news_africa_04", domain="News",
                duration_s=270.0, duration_bin="medium", spec=spec,
                tg_interval=(89.0, 92.0),
                sg_boxes={"start_bbox": [356.20, 210.44, 641.80, 812.30],
                          "end_bbox": [349.71, 205.12, 655.40, 820.55]})
    insts += [
        Instance(task="TG", **base),
        Instance(task="SG", **base),
        Instance(task="QA", **{**base, **dict(
            qa_question_text="Which region does the anchor with long brown "
                             "curly hair report on in this segment?",
            qa_question_image="Which region does the anchor in the reference "
                              "image report on in this segment?",
            qa_options={"A": "Southeast Asia", "B": "North Africa",
                        "C": "Scandinavia", "D": "West Africa", "E": "Andes"},
            qa_answer="D", qa_type="Fact Recognition")}),
    ]

    # 5. Film rooftop (Film, long).
    spec = TargetSpec("the bodyguard draw his pistol",
                      "the man shown in the reference image draw his pistol")
    base = dict(track="VL", video_id="heist_roof_11", domain="Film",
                duration_s=512.0, duration_bin="long", spec=spec,
                tg_interval=(318.44, 323.90),
                sg_boxes={"start_bbox": [588.30, 402.11, 701.42, 720.65],
                          "end_bbox": [601.87, 398.40, 718.03, 733.20]})
    insts += [
        Instance(task="TG", **base),
        Instance(task="SG", **base),
        Instance(task="QA", **{**base, **dict(
            qa_question_text="How many guards does the bodyguard confront "
                             "on the roof?",
            qa_question_image="How many guards does the man shown in the "
                              "reference image confront on the roof?",
            qa_options={"A": "1", "B": "4", "C": "3", "D": "2", "E": "0"},
            qa_answer="C", qa_type="Visual Counting")}),
    ]

    # 6. Variety balance beam (Variety, medium) — scene-conditioned reference.
    spec = TargetSpec("the contestant tumble off the balance beam",
                      "the contestant shown in the reference image tumble "
                      "off the balance beam")
    base = dict(track="VL", video_id="variety_beam_03", domain="Variety",
                duration_s=168.0, duration_bin="medium", spec=spec,
                tg_interval=(97.20, 101.80),
                sg_boxes={"start_bbox": [180.44, 300.10, 512.77, 811.42],
                          "end_bbox": [210.05, 355.80, 560.12, 866.90]})
    insts += [
        Instance(task="TG", **base),
        Instance(task="SG", **base),
        Instance(task="QA", **{**base, **dict(
            qa_question_text="Where is the contestant standing right before "
                             "tumbling off the beam?",
            qa_question_image="Where is the contestant shown in the "
                              "reference image standing right before "
                              "tumbling off?",
            qa_options={"A": "Left edge", "B": "Center", "C": "Right edge",
                        "D": "On the mat", "E": "On the stairs"},
            qa_answer="B", qa_type="Spatial Perception")}),
    ]
    return insts


def _handwritten_av():
    insts = []
    # 1. The paper's AV example: the 270 s news clip with native audio (Fig. E4).
    spec = TargetSpec("the female news anchor with long brown curly hair, a "
                      "black-and-white houndstooth jacket, and a black "
                      "turtleneck report that France is seeking to redefine "
                      "its role in Africa",
                      "the news anchor in the reference image report that "
                      "France is seeking to redefine its role in Africa")
    base = dict(track="AV", video_id="news_africa_04", domain="News",
                duration_s=270.0, duration_bin="medium", spec=spec,
                tg_interval=(89.0, 92.0),
                sg_boxes={"start_bbox": [356.20, 210.44, 641.80, 812.30],
                          "end_bbox": [349.71, 205.12, 655.40, 820.55]})
    insts += [
        Instance(task="TG", **base),
        Instance(task="SG", **base),
        Instance(task="QA", **{**base, **dict(
            qa_question_text="According to the anchor's spoken report, what "
                             "does France seek to do in Africa?",
            qa_question_image="According to the spoken report, what does the "
                              "anchor in the reference image say France "
                              "seeks to do in Africa?",
            qa_options={"A": "Exit the Sahel coalition",
                        "B": "Redefine its role", "C": "Double aid",
                        "D": "Open new bases", "E": "Host a summit"},
            qa_answer="B", qa_type="Speech Understanding")}),
    ]

    # 2. Film: audio-visual event binding.
    spec = TargetSpec("the telephone ring interrupt the interrogation",
                      "the telephone shown in the reference image ring "
                      "during the interrogation")
    base = dict(track="AV", video_id="noir_phonecall_08", domain="Film",
                duration_s=384.0, duration_bin="long", spec=spec,
                tg_interval=(201.60, 204.90),
                sg_boxes={"start_bbox": [70.15, 601.33, 220.87, 855.40],
                          "end_bbox": [82.90, 588.02, 235.41, 870.12]})
    insts += [
        Instance(task="TG", **base),
        Instance(task="SG", **base),
        Instance(task="QA", **{**base, **dict(
            qa_question_text="What does the detective say when the telephone "
                             "rings?",
            qa_question_image="What does the detective say when the "
                              "telephone shown in the reference image rings?",
            qa_options={"A": "\"Leave the room\"", "B": "\"Take a message\"",
                        "C": "Nothing", "D": "\"It's for you\"",
                        "E": "\"Turn it off\""},
            qa_answer="D", qa_type="Event Understanding")}),
    ]

    # 3. Variety: emotion from voice + face.
    spec = TargetSpec("the judge react to the off-key performance",
                      "the judge shown in the reference image react to the "
                      "off-key performance")
    base = dict(track="AV", video_id="variety_judge_19", domain="Variety",
                duration_s=203.0, duration_bin="medium", spec=spec,
                tg_interval=(141.02, 145.40),
                sg_boxes={"start_bbox": [60.33, 155.20, 350.80, 690.55],
                          "end_bbox": [48.90, 142.11, 335.60, 702.88]})
    insts += [
        Instance(task="TG", **base),
        Instance(task="SG", **base),
        Instance(task="QA", **{**base, **dict(
            qa_question_text="How does the judge feel about the performance, "
                             "judging by voice and face?",
            qa_question_image="How does the judge shown in the reference "
                              "image feel about the performance?",
            qa_options={"A": "Delighted", "B": "Bored", "C": "Amused",
                        "D": "Angry", "E": "Moved"},
            qa_answer="C", qa_type="Emotion Recognition")}),
    ]
    return insts


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_vl() -> list:
    """Mini OmniCompass-VL: 6 hand-written + 33 filler videos, 117 instances."""
    insts = _handwritten_vl()
    for vid, dom, dur, tq, iq, cue, pct, qtype in _VL_FILLER:
        insts += _video_instances("VL", vid, dom, dur, tq, iq, cue, pct,
                                  qtype, sg_area=0.132, rng=RNG)
    return insts


def build_av() -> list:
    """Mini OmniCompass-AV: 3 hand-written + 7 filler videos, 30 instances."""
    insts = _handwritten_av()
    for vid, dom, dur, tq, iq, pct, qtype in _AV_FILLER:
        insts += _video_instances("AV", vid, dom, dur, tq, iq, "entity", pct,
                                  qtype, sg_area=0.307, rng=RNG)
    return insts


def build_benchmark():
    """(vl_instances, av_instances)."""
    return build_vl(), build_av()


if __name__ == "__main__":
    vl, av = build_benchmark()
    print(f"VL: {len(vl)} instances ({len(vl) // 3} videos x TG/SG/QA)")
    print(f"AV: {len(av)} instances ({len(av) // 3} videos x TG/SG/QA)")
    for name, xs, ref_span, ref_area in (("VL", vl, 0.046, 0.132),
                                         ("AV", av, 0.017, 0.307)):
        spans = [(i.tg_interval[1] - i.tg_interval[0]) / i.duration_s
                 for i in xs]
        areas = [((b[2] - b[0]) * (b[3] - b[1]) / 1e6)
                 for i in xs for b in (i.sg_boxes["start_bbox"],)]
        print(f"{name}: TG span ratio mean {sum(spans)/len(spans):.3f} "
              f"(paper {ref_span}), SG area mean {sum(areas)/len(areas):.3f} "
              f"(paper {ref_area})")
