# tests/test_reconstruct.py
from dubber.reconstruct import placement_ms
from dubber.models import Segment

def test_placement_ms_converts_start_seconds():
    results = [(Segment(1.5, 2.0, "A"), "a.wav"),
               (Segment(3.0, 3.4, "B"), "b.wav")]
    assert placement_ms(results) == [(1500, "a.wav"), (3000, "b.wav")]
