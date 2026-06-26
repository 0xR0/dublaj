# dubber/gender.py
import math

from dubber import config
from dubber.models import Segment
from dubber.utils import get_logger

logger = get_logger()


def classify_from_f0(f0_hz: float) -> str:
    if f0_hz is None or math.isnan(f0_hz):
        return "unknown"
    if f0_hz < config.GENDER_F0_THRESHOLD_HZ:
        return "male"
    if f0_hz < config.CHILD_F0_THRESHOLD_HZ:
        return "female"
    return "child"


def _mean_f0(y, sr) -> float:
    import librosa
    import numpy as np
    f0, voiced, _ = librosa.pyin(
        y, fmin=70, fmax=500, sr=sr)
    vals = f0[~np.isnan(f0)]
    return float(np.median(vals)) if len(vals) else math.nan


def assign_genders(vocals_path: str, segments: list[Segment]) -> dict[str, str]:
    """Her speaker_id için medyan F0'dan cinsiyet. Segmentlere de yazar."""
    import librosa
    import numpy as np

    y, sr = librosa.load(vocals_path, sr=16000)
    by_speaker: dict[str, list] = {}
    for s in segments:
        i0, i1 = int(s.start * sr), int(s.end * sr)
        if i1 > i0:
            by_speaker.setdefault(s.speaker_id, []).append(y[i0:i1])

    genders: dict[str, str] = {}
    for spk, chunks in by_speaker.items():
        audio = np.concatenate(chunks)[: sr * 30]  # ilk 30 sn yeter
        genders[spk] = classify_from_f0(_mean_f0(audio, sr))
        logger.info("Konuşmacı %s -> %s", spk, genders[spk])

    for s in segments:
        s.gender = genders.get(s.speaker_id, "unknown")
    return genders
