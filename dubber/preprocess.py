# dubber/preprocess.py
from pathlib import Path

from dubber import config
from dubber.utils import get_logger, run_ffmpeg

logger = get_logger()


def build_ffmpeg_args(input_path: str, output_path: str) -> list[str]:
    return [
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(config.SAMPLE_RATE),
        "-af", "loudnorm",
        str(output_path),
    ]


def preprocess(input_path: str) -> Path:
    """MP3/WAV -> 16kHz mono normalize WAV. temp/audio_16k.wav döner."""
    out = config.TEMP_DIR / "audio_16k.wav"
    logger.info("Önişleme: %s -> %s", input_path, out)
    run_ffmpeg(build_ffmpeg_args(input_path, str(out)))
    return out
