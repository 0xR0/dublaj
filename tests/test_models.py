# tests/test_models.py
from dubber.models import Segment

def test_segment_defaults_and_duration():
    s = Segment(start=1.0, end=3.5, speaker_id="SPEAKER_00")
    assert s.duration == 2.5
    assert s.gender == "unknown"
    assert s.text == ""
    assert s.text_tr == ""

def test_segment_to_dict_roundtrip():
    s = Segment(start=0.0, end=2.0, speaker_id="SPEAKER_01",
                gender="female", text="hello", text_tr="merhaba")
    d = s.to_dict()
    assert d["speaker_id"] == "SPEAKER_01"
    assert Segment.from_dict(d) == s
