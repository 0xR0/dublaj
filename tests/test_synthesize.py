# tests/test_synthesize.py
from dubber.synthesize import (pick_reference_segment, timing_factor,
                               edge_voice_for, build_preset_map, plan_voices)
from dubber.models import Segment
from dubber import config

def test_pick_reference_longest_segment():
    segs = [Segment(0, 1, "A"), Segment(5, 12, "A"), Segment(2, 3, "A")]
    ref = pick_reference_segment(segs, "A")
    assert (ref.start, ref.end) == (5, 12)

def test_pick_reference_none_when_speaker_absent():
    segs = [Segment(0, 1, "A")]
    assert pick_reference_segment(segs, "Z") is None

def test_timing_factor_speeds_up_when_too_long():
    # üretilen 4s, hedef 2s -> 2.0 faktör
    assert timing_factor(generated=4.0, target=2.0) == 2.0

def test_timing_factor_one_when_short_enough():
    assert timing_factor(generated=1.5, target=2.0) == 1.0

def test_edge_voice_for_gender():
    assert edge_voice_for("male", 0) == ("tr-TR-AhmetNeural", 0)
    assert edge_voice_for("female", 1)[0] == "tr-TR-EmelNeural"


def test_build_preset_map_distinct_by_gender():
    available = config.XTTS_VOICES["female"] + config.XTTS_VOICES["male"]
    genders = {"A": "female", "B": "female", "C": "male"}
    m = build_preset_map({"A", "B", "C"}, genders, available)
    assert m["A"] in config.XTTS_VOICES["female"]
    assert m["B"] in config.XTTS_VOICES["female"]
    assert m["C"] in config.XTTS_VOICES["male"]
    assert m["A"] != m["B"]          # aynı cinsiyet farklı ses
    assert len(set(m.values())) == 3


def test_build_preset_map_falls_back_to_available():
    # Tercih listesindeki isimler modelde yoksa mevcut seslerden seçer
    available = ["Voice X", "Voice Y"]
    m = build_preset_map({"A", "B"}, {"A": "female", "B": "male"}, available)
    assert set(m.values()) <= set(available)
    assert m["A"] != m["B"]


def test_build_preset_map_empty_available():
    assert build_preset_map({"A"}, {"A": "male"}, []) == {}


def test_edge_voice_for_child_high_pitch():
    voice, pitch = edge_voice_for("child", 0)
    assert voice == config.EDGE_VOICES["child"]
    assert pitch == config.EDGE_CHILD_PITCH


def test_plan_voices_child_always_cloned():
    # preset modunda bile çocuk klonlanır, yetişkinler preset alır
    ids = {"A", "B", "C"}
    genders = {"A": "male", "B": "female", "C": "child"}
    available = config.XTTS_VOICES["female"] + config.XTTS_VOICES["male"]
    preset_ids, clone_ids = plan_voices(ids, genders, available,
                                        mode="preset", child_mode="xtts")
    assert "C" in clone_ids and "C" not in preset_ids
    assert {"A", "B"} <= preset_ids


def test_plan_voices_clone_mode_everyone_cloned():
    ids = {"A", "B"}
    genders = {"A": "male", "B": "female"}
    preset_ids, clone_ids = plan_voices(ids, genders, ["X"],
                                        mode="clone", child_mode="xtts")
    assert preset_ids == set()
    assert clone_ids == ids


def test_plan_voices_child_off_treated_as_adult():
    ids = {"C"}
    available = config.XTTS_VOICES["female"]
    preset_ids, clone_ids = plan_voices(ids, {"C": "child"}, available,
                                        mode="preset", child_mode="off")
    assert "C" in preset_ids
