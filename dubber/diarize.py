# dubber/diarize.py
import os

from dubber.models import Segment
from dubber.utils import get_logger

logger = get_logger()


def choose_engine(forced: str | None, hf_token: str | None) -> str:
    if forced in ("pyannote", "speechbrain"):
        return forced
    return "pyannote" if hf_token else "speechbrain"


def diarize(vocals_path: str, forced: str | None = None) -> list[Segment]:
    hf_token = os.environ.get("HF_TOKEN")
    engine = choose_engine(forced, hf_token)
    logger.info("Diarization motoru: %s", engine)
    try:
        if engine == "pyannote":
            return _diarize_pyannote(vocals_path, hf_token)
        return _diarize_speechbrain(vocals_path)
    except Exception as e:  # pyannote başarısız -> speechbrain yedeği
        logger.warning("pyannote başarısız (%s), speechbrain'e geçiliyor", e)
        return _diarize_speechbrain(vocals_path)


def _diarize_pyannote(vocals_path: str, hf_token: str) -> list[Segment]:
    from pyannote.audio import Pipeline
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=hf_token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    import torch
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    annotation = pipeline(vocals_path)
    segs = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segs.append(Segment(start=turn.start, end=turn.end, speaker_id=speaker))
    return _merge_short(segs)


def _diarize_speechbrain(vocals_path: str) -> list[Segment]:
    """ECAPA gömme + agglomerative clustering ile tokensiz diarization."""
    import numpy as np
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/ecapa")
    y, sr = librosa.load(vocals_path, sr=16000)
    win, hop = int(1.5 * sr), int(0.75 * sr)
    embs, times = [], []
    import torch
    for i in range(0, max(1, len(y) - win), hop):
        chunk = torch.tensor(y[i:i + win]).unsqueeze(0)
        emb = encoder.encode_batch(chunk).squeeze().detach().cpu().numpy()
        embs.append(emb)
        times.append((i / sr, (i + win) / sr))
    embs = np.vstack(embs)
    n = _estimate_speakers(embs)
    labels = AgglomerativeClustering(n_clusters=n).fit_predict(embs)
    segs = [Segment(start=t0, end=t1, speaker_id=f"SPEAKER_{lab:02d}")
            for (t0, t1), lab in zip(times, labels)]
    return _merge_short(segs)


def _estimate_speakers(embs, max_k: int = 6) -> int:
    """Basit silhouette tabanlı konuşmacı sayısı tahmini."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    if len(embs) < 4:
        return 1
    best_k, best_score = 1, -1.0
    for k in range(2, min(max_k, len(embs)) + 1):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(embs)
        score = silhouette_score(embs, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _merge_short(segs: list[Segment]) -> list[Segment]:
    """Bitişik aynı konuşmacı segmentlerini birleştir, zamana göre sırala."""
    segs = sorted(segs, key=lambda s: s.start)
    merged: list[Segment] = []
    for s in segs:
        if merged and merged[-1].speaker_id == s.speaker_id and \
                s.start - merged[-1].end < 0.5:
            merged[-1] = Segment(start=merged[-1].start, end=s.end,
                                 speaker_id=s.speaker_id)
        else:
            merged.append(s)
    return merged
