# tests/test_transcribe.py
from dubber.transcribe import assign_words_to_segments, merge_into_sentences
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


def test_merge_combines_fragments_until_sentence_end():
    segs = [Segment(0, 1, "A", text="Hello"),
            Segment(1, 2, "A", text="world."),
            Segment(2, 3, "A", text="Next one.")]
    out = merge_into_sentences(segs)
    assert len(out) == 2
    assert out[0].text == "Hello world."
    assert (out[0].start, out[0].end) == (0, 2)
    assert out[1].text == "Next one."


def test_merge_does_not_cross_speaker():
    segs = [Segment(0, 1, "A", text="Hi"),
            Segment(1, 2, "B", text="there.")]
    out = merge_into_sentences(segs)
    assert len(out) == 2
    assert out[0].speaker_id == "A"
    assert out[1].speaker_id == "B"


def test_merge_respects_max_seconds():
    segs = [Segment(0, 10, "A", text="long talk no punct"),
            Segment(10, 20, "A", text="continues")]
    out = merge_into_sentences(segs, max_seconds=12.0)
    assert len(out) == 2


def test_merge_preserves_gender():
    segs = [Segment(0, 1, "A", gender="male", text="Bir"),
            Segment(1, 2, "A", gender="male", text="iki.")]
    out = merge_into_sentences(segs)
    assert len(out) == 1
    assert out[0].gender == "male"
    assert out[0].text == "Bir iki."
