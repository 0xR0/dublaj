# tests/test_separate.py
from pathlib import Path
from dubber.separate import separate

def test_separate_skipped_returns_vocals_as_input(tmp_path):
    src = tmp_path / "audio_16k.wav"
    src.write_bytes(b"RIFF")  # sahte wav
    vocals, background = separate(str(src), enabled=False)
    assert Path(vocals) == src
    assert background is None
