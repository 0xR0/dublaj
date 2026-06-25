# tests/test_pipeline.py
from unittest.mock import patch
from dubber import pipeline
from dubber.models import Segment


def test_pipeline_calls_stages_in_order(tmp_path):
    segs = [Segment(0, 2, "A", text="hi", text_tr="selam")]
    with patch.object(pipeline, "preprocess", return_value="16k.wav") as p_pre, \
         patch.object(pipeline, "separate", return_value=("voc.wav", "bg.wav")) as p_sep, \
         patch.object(pipeline, "diarize", return_value=segs) as p_dia, \
         patch.object(pipeline, "assign_genders", return_value={"A": "male"}) as p_gen, \
         patch.object(pipeline, "transcribe", return_value=(segs, "en")) as p_tr, \
         patch.object(pipeline, "translate_segments", return_value=segs) as p_tl, \
         patch.object(pipeline, "synthesize", return_value=[(segs[0], "c.wav")]) as p_sy, \
         patch.object(pipeline, "reconstruct", return_value="out.mp3") as p_rc, \
         patch.object(pipeline, "_audio_duration", return_value=2.0):
        out = pipeline.run("in.mp3", "out.mp3")
    assert out == "out.mp3"
    p_pre.assert_called_once()
    p_sep.assert_called_once()
    p_dia.assert_called_once()
    p_gen.assert_called_once()
    p_tr.assert_called_once()
    p_tl.assert_called_once()
    p_sy.assert_called_once()
    p_rc.assert_called_once()
