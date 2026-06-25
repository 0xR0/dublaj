# dubber/transcribe.py
from dubber import config
from dubber.models import Segment
from dubber.utils import get_logger

logger = get_logger()


def assign_words_to_segments(words: list[dict],
                             segments: list[Segment]) -> list[Segment]:
    """Kelimeleri zaman ortası en çok örtüşen segmente atar.
    Metni boş kalan segmentler düşürülür."""
    buckets: dict[int, list[str]] = {i: [] for i in range(len(segments))}
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        idx = _best_segment(mid, segments)
        if idx is not None:
            buckets[idx].append(w["word"].strip())
    out = []
    for i, seg in enumerate(segments):
        text = " ".join(buckets[i]).strip()
        if text:
            seg.text = text
            out.append(seg)
    return out


def _best_segment(t: float, segments: list[Segment]) -> int | None:
    for i, s in enumerate(segments):
        if s.start <= t <= s.end:
            return i
    # örtüşme yoksa en yakın segment
    if not segments:
        return None
    return min(range(len(segments)),
               key=lambda i: min(abs(t - segments[i].start),
                                 abs(t - segments[i].end)))


def transcribe(vocals_path: str, segments: list[Segment]) -> list[Segment]:
    from faster_whisper import WhisperModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(config.WHISPER_MODEL, device=device,
                         compute_type=compute, download_root="models")
    logger.info("Konuşma tanıma (faster-whisper)...")
    seg_iter, info = model.transcribe(vocals_path, word_timestamps=True)
    words = []
    for s in seg_iter:
        for w in (s.words or []):
            words.append({"start": w.start, "end": w.end, "word": w.word})
    logger.info("Algılanan dil: %s", info.language)
    result = assign_words_to_segments(words, segments)
    return result, info.language


def detected_language(vocals_path: str) -> str:
    """Sadece dil tespiti gerektiğinde kullanılır (nadiren)."""
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    _, info = model.transcribe(vocals_path)
    return info.language
