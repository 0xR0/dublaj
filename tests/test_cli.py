# tests/test_cli.py
from dub import parse_args

def test_parse_args_defaults():
    a = parse_args(["input.mp3"])
    assert a.input == "input.mp3"
    assert a.background is True
    assert a.tts == "xtts"
    assert a.diarizer is None

def test_parse_args_flags():
    a = parse_args(["in.mp3", "--no-background", "--tts", "edge",
                    "--diarizer", "speechbrain", "--output", "o.mp3"])
    assert a.background is False
    assert a.tts == "edge"
    assert a.diarizer == "speechbrain"
    assert a.output == "o.mp3"
