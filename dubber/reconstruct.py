# dubber/reconstruct.py
from pathlib import Path

from dubber import config
from dubber.models import Segment
from dubber.utils import get_logger

logger = get_logger()


def placement_ms(results: list[tuple]) -> list[tuple]:
    """[(Segment, clip_path)] -> [(başlangıç_ms, clip_path)]"""
    return [(int(seg.start * 1000), str(clip)) for seg, clip in results]


def reconstruct(results: list[tuple], background_path: str | None,
                total_duration: float, output_path: str) -> Path:
    from pydub import AudioSegment

    total_ms = int(total_duration * 1000)
    base = AudioSegment.silent(duration=total_ms)

    for start_ms, clip in placement_ms(results):
        speech = AudioSegment.from_file(clip)
        base = base.overlay(speech, position=start_ms)

    if background_path:
        bg = AudioSegment.from_file(background_path)
        bg = bg - 6  # arka planı biraz kıs
        base = bg.overlay(base) if len(bg) >= len(base) else base.overlay(bg)

    base = base.normalize()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    base.export(str(out), format="mp3", bitrate="192k")
    logger.info("Çıktı yazıldı: %s", out)
    return out
