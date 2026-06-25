# tests/test_diarize.py
from dubber.diarize import choose_engine

def test_choose_engine_pyannote_when_token():
    assert choose_engine(forced=None, hf_token="hf_abc") == "pyannote"

def test_choose_engine_speechbrain_when_no_token():
    assert choose_engine(forced=None, hf_token=None) == "speechbrain"

def test_choose_engine_respects_force():
    assert choose_engine(forced="speechbrain", hf_token="hf_abc") == "speechbrain"
    assert choose_engine(forced="pyannote", hf_token=None) == "pyannote"

from dubber.diarize import _merge_short
from dubber.models import Segment

def test_merge_short_merges_adjacent_same_speaker():
    segs = [Segment(0, 1, "A"), Segment(1.2, 2, "A"), Segment(2.1, 3, "B")]
    out = _merge_short(segs)
    assert len(out) == 2
    assert out[0].start == 0 and out[0].end == 2
    assert out[1].speaker_id == "B"
