# tests/test_transcribe.py
from dubber.transcribe import assign_words_to_segments
from dubber.models import Segment

def test_assign_words_fills_text_by_overlap():
    segments = [Segment(0, 2, "A"), Segment(2, 4, "B")]
    words = [
        {"start": 0.1, "end": 0.5, "word": "merhaba"},
        {"start": 1.0, "end": 1.4, "word": "dünya"},
        {"start": 2.5, "end": 2.9, "word": "ikinci"},
    ]
    out = assign_words_to_segments(words, segments)
    assert out[0].text == "merhaba dünya"
    assert out[1].text == "ikinci"

def test_assign_words_drops_empty_segments():
    segments = [Segment(0, 2, "A"), Segment(2, 4, "B")]
    words = [{"start": 0.1, "end": 0.5, "word": "tek"}]
    out = assign_words_to_segments(words, segments)
    assert len(out) == 1
    assert out[0].speaker_id == "A"
