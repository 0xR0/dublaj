# tests/test_preprocess.py
from dubber.preprocess import build_ffmpeg_args

def test_build_ffmpeg_args_16k_mono():
    args = build_ffmpeg_args("in.mp3", "out.wav")
    assert "in.mp3" in args
    assert "out.wav" in args
    assert "16000" in args          # ar 16000
    assert "1" in args              # ac 1 (mono)
    assert "loudnorm" in " ".join(args)
