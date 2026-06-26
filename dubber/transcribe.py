# dubber/transcribe.py
from dataclasses import replace

from dubber import config
from dubber.models import Segment
from dubber.utils import get_logger

logger = get_logger()

_SENTENCE_END = (".", "!", "?", "…")


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(_SENTENCE_END)


def merge_into_sentences(segments: list[Segment],
                         max_seconds: float | None = None) -> list[Segment]:
    """Aynı konuşmacının ardışık parçalarını, cümle sonu noktalamasına kadar
    tek bir segmentte birleştirir. Böylece çeviri yarım cümle yerine tam cümle
    görür (daha anlamlı) ve daha az/uzun TTS klibi oluşur (daha az hızlandırma).
    Konuşmacı değişiminde, cümle bittiğinde veya süre sınırında keser."""
    cap = config.MAX_SENTENCE_SECONDS if max_seconds is None else max_seconds
    merged: list[Segment] = []
    cur: Segment | None = None
    for s in segments:
        joinable = (cur is not None and s.speaker_id == cur.speaker_id
                    and not _ends_sentence(cur.text)
                    and (s.end - cur.start) <= cap)
        if joinable:
            cur = replace(cur, end=s.end,
                          text=(cur.text + " " + s.text).strip())
        else:
            if cur is not None:
                merged.append(cur)
            cur = replace(s)
    if cur is not None:
        merged.append(cur)
    return merged


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


def transcribe(vocals_path: str, segments: list[Segment]) -> tuple[list[Segment], str]:
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
