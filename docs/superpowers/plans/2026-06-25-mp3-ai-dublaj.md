# MP3 AI Türkçe Dublaj Sistemi — Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MP3/WAV sesini girdi alıp konuşmacıları ayıran, cinsiyet tespiti yapan, her karaktere XTTS ile orijinalinden klonlanmış ses atayan, Türkçeye çeviren ve `output_dubbed.mp3` üreten tam otomatik dublaj sistemi kurmak.

**Architecture:** Modüler `dubber/` paketi + sıralı pipeline. Her aşama tek sorumluluklu bir modül, ortak `Segment` veri modeli üzerinden konuşur. Ağır ML çağrıları (Demucs, pyannote/speechbrain, faster-whisper, NLLB, XTTS) ince sarmalayıcıların arkasında izole; saf mantık (zamanlama, ses atama, hizalama, eşik) birim testlerle TDD edilir. Ağır modeller Colab GPU'da duman testiyle doğrulanır.

**Tech Stack:** Python 3.10+, pytest, ffmpeg, faster-whisper, pyannote.audio, speechbrain, demucs, Coqui TTS (XTTS-v2), edge-tts, transformers (NLLB-200), librosa, numpy, pydub, soundfile.

**Test ortamı notu:** Geliştirme makinesinde GPU/model yoktur. Testler yalnızca model gerektirmeyen saf mantığı kapsar (komut üretimi, hizalama, eşik, zamanlama, atama, yedek seçimi). Model çağrıları mock'lanır. Uçtan uca duman testi Colab'da elle yapılır (Task 14).

---

## Dosya Yapısı

```
dublaj/
├── dub.py                      # CLI giriş (argparse)
├── dubber/
│   ├── __init__.py
│   ├── config.py               # yollar, ses haritası, eşik sabitleri
│   ├── models.py               # Segment dataclass + yardımcı tipler
│   ├── utils.py                # logging, zaman biçimleme, ffmpeg sarmalayıcı
│   ├── preprocess.py           # FFmpeg 16k mono normalize
│   ├── separate.py             # Demucs vokal/arka plan ayırma
│   ├── diarize.py              # pyannote|speechbrain seçimi + segmentler
│   ├── gender.py               # F0 pitch → male/female
│   ├── transcribe.py           # faster-whisper + diarize hizalama
│   ├── translate.py            # NLLB-200 → Türkçe
│   ├── synthesize.py           # XTTS klonlama + Edge yedek + atempo timing
│   ├── reconstruct.py          # timeline mix + normalize
│   └── pipeline.py             # aşamaları bağlar
├── notebook/
│   └── dublaj_colab.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_utils.py
│   ├── test_preprocess.py
│   ├── test_separate.py
│   ├── test_diarize.py
│   ├── test_gender.py
│   ├── test_transcribe.py
│   ├── test_translate.py
│   ├── test_synthesize.py
│   ├── test_reconstruct.py
│   ├── test_pipeline.py
│   └── test_cli.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── input/  temp/  output/  models/  logs/   (.gitkeep ile)
```

---

## Task 0: Proje iskeleti

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `.gitignore`
- Create: `dubber/__init__.py`, `tests/__init__.py`
- Create: `input/.gitkeep`, `temp/.gitkeep`, `output/.gitkeep`, `models/.gitkeep`, `logs/.gitkeep`

- [ ] **Step 1: Git deposu başlat**

Run: `git init && git config user.email maynax@gmail.com && git config user.name dublaj`
Expected: "Initialized empty Git repository"

- [ ] **Step 2: Klasörleri ve .gitkeep dosyalarını oluştur**

```bash
mkdir -p dubber tests input temp output models logs notebook docs/superpowers
touch input/.gitkeep temp/.gitkeep output/.gitkeep models/.gitkeep logs/.gitkeep
touch dubber/__init__.py tests/__init__.py
```

- [ ] **Step 3: requirements.txt yaz**

```
faster-whisper
pyannote.audio
speechbrain
demucs
TTS
edge-tts
transformers
sentencepiece
torch
torchaudio
librosa
numpy
pydub
soundfile
```

- [ ] **Step 4: requirements-dev.txt yaz**

```
pytest
```

- [ ] **Step 5: .gitignore yaz**

```
__pycache__/
*.pyc
temp/*
output/*
models/*
logs/*
input/*
!**/.gitkeep
.venv/
*.wav
*.mp3
```

- [ ] **Step 6: Dev bağımlılıklarını kur**

Run: `pip install pytest`
Expected: pytest kurulur (model bağımlılıkları sadece Colab'da kurulur).

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: proje iskeleti ve dizin yapısı"
```

---

## Task 1: Veri modeli (Segment)

**Files:**
- Create: `dubber/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Failing test yaz**

```python
# tests/test_models.py
from dubber.models import Segment

def test_segment_defaults_and_duration():
    s = Segment(start=1.0, end=3.5, speaker_id="SPEAKER_00")
    assert s.duration == 2.5
    assert s.gender == "unknown"
    assert s.text == ""
    assert s.text_tr == ""

def test_segment_to_dict_roundtrip():
    s = Segment(start=0.0, end=2.0, speaker_id="SPEAKER_01",
                gender="female", text="hello", text_tr="merhaba")
    d = s.to_dict()
    assert d["speaker_id"] == "SPEAKER_01"
    assert Segment.from_dict(d) == s
```

- [ ] **Step 2: Test fail olduğunu doğrula**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — "No module named 'dubber.models'"

- [ ] **Step 3: models.py yaz**

```python
# dubber/models.py
from dataclasses import dataclass, asdict


@dataclass
class Segment:
    start: float
    end: float
    speaker_id: str
    gender: str = "unknown"
    text: str = ""
    text_tr: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(**d)
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_models.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dubber/models.py tests/test_models.py && git commit -m "feat: Segment veri modeli"
```

---

## Task 2: Config ve utils

**Files:**
- Create: `dubber/config.py`, `dubber/utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: config.py yaz (testsiz sabitler)**

```python
# dubber/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
TEMP_DIR = ROOT / "temp"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

SAMPLE_RATE = 16000
GENDER_F0_THRESHOLD_HZ = 165.0   # altı male, üstü female
MIN_REF_SECONDS = 3.0            # XTTS referansı için alt sınır
PREFERRED_REF_SECONDS = 6.0     # tercih edilen referans süresi
MAX_ATEMPO = 1.3                # zamanlama sığdırmada üst hız sınırı

# Edge TTS yedeği için cinsiyet→ses
EDGE_VOICES = {"male": "tr-TR-AhmetNeural", "female": "tr-TR-EmelNeural"}
# Aynı cinsiyette çoklu konuşmacı için Edge pitch ofsetleri (Hz)
EDGE_PITCH_OFFSETS = [0, -15, 10, -8, 18]

WHISPER_MODEL = "large-v3"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

for _d in (INPUT_DIR, TEMP_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: utils için failing test yaz**

```python
# tests/test_utils.py
from dubber.utils import format_timestamp, atempo_chain

def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3725) == "62:05"

def test_atempo_chain_single():
    assert atempo_chain(1.2) == "atempo=1.2"

def test_atempo_chain_splits_above_2():
    # ffmpeg atempo tek filtrede max 2.0; 3.0 -> iki filtreye bölünür
    assert atempo_chain(3.0) == "atempo=1.732,atempo=1.732"

def test_atempo_chain_clamps_to_one_when_not_speeding():
    assert atempo_chain(0.8) == "atempo=1.0"
```

- [ ] **Step 3: Test fail doğrula**

Run: `pytest tests/test_utils.py -v`
Expected: FAIL — "No module named 'dubber.utils'"

- [ ] **Step 4: utils.py yaz**

```python
# dubber/utils.py
import logging
import math
import subprocess
from pathlib import Path

from dubber import config


def get_logger(name: str = "dublaj") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(config.LOGS_DIR / "run.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def format_timestamp(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def atempo_chain(factor: float) -> str:
    """ffmpeg atempo filtre zinciri. <1 ise hız değiştirmez (1.0)."""
    if factor <= 1.0:
        return "atempo=1.0"
    parts = []
    remaining = factor
    while remaining > 2.0:
        parts.append(2.0)
        remaining /= 2.0
    # Kalan >2 değil; eşit böl ki her parça <=2 olsun
    if parts:
        n = len(parts) + 1
        each = round(factor ** (1.0 / n), 3)
        parts = [each] * n
    else:
        parts = [round(remaining, 3)]
    return ",".join(f"atempo={p}" for p in parts)


def run_ffmpeg(args: list[str]) -> None:
    """ffmpeg'i verilen argümanlarla çalıştırır, hatada exception fırlatır."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True)
```

- [ ] **Step 5: Test pass doğrula**

Run: `pytest tests/test_utils.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add dubber/config.py dubber/utils.py tests/test_utils.py && git commit -m "feat: config sabitleri ve utils (zaman/atempo/ffmpeg)"
```

---

## Task 3: Önişleme (preprocess)

**Files:**
- Create: `dubber/preprocess.py`
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Failing test yaz (komut üretimi)**

```python
# tests/test_preprocess.py
from dubber.preprocess import build_ffmpeg_args

def test_build_ffmpeg_args_16k_mono():
    args = build_ffmpeg_args("in.mp3", "out.wav")
    assert "in.mp3" in args
    assert "out.wav" in args
    assert "16000" in args          # ar 16000
    assert "1" in args              # ac 1 (mono)
    assert "loudnorm" in " ".join(args)
```

- [ ] **Step 2: Test fail doğrula**

Run: `pytest tests/test_preprocess.py -v`
Expected: FAIL — "No module named 'dubber.preprocess'"

- [ ] **Step 3: preprocess.py yaz**

```python
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
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_preprocess.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add dubber/preprocess.py tests/test_preprocess.py && git commit -m "feat: ffmpeg önişleme (16k mono normalize)"
```

---

## Task 4: Kaynak ayırma (separate / Demucs)

**Files:**
- Create: `dubber/separate.py`
- Test: `tests/test_separate.py`

Demucs ağır model; gerçek çağrı Colab'da. Burada `--no-background` atlama mantığını test ederiz.

- [ ] **Step 1: Failing test yaz**

```python
# tests/test_separate.py
from pathlib import Path
from dubber.separate import separate

def test_separate_skipped_returns_vocals_as_input(tmp_path):
    src = tmp_path / "audio_16k.wav"
    src.write_bytes(b"RIFF")  # sahte wav
    vocals, background = separate(str(src), enabled=False)
    assert Path(vocals) == src
    assert background is None
```

- [ ] **Step 2: Test fail doğrula**

Run: `pytest tests/test_separate.py -v`
Expected: FAIL — "No module named 'dubber.separate'"

- [ ] **Step 3: separate.py yaz**

```python
# dubber/separate.py
from pathlib import Path

from dubber import config
from dubber.utils import get_logger

logger = get_logger()


def separate(audio_path: str, enabled: bool = True):
    """Demucs ile vokal/arka plan ayırır.
    enabled=False ise ayırmaz: (girdi, None) döner.
    Döner: (vocals_path, background_path | None)
    """
    if not enabled:
        logger.info("Arka plan ayırma atlandı (--no-background)")
        return audio_path, None

    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torchaudio

    logger.info("Demucs ile kaynak ayırma...")
    model = get_model("htdemucs")
    model.eval()
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # demucs stereo bekler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    sources = apply_model(model.to(device), wav[None].to(device),
                          split=True, overlap=0.25)[0]
    sources = sources * ref.std() + ref.mean()
    stems = dict(zip(model.sources, sources))
    vocals = stems["vocals"].mean(0, keepdim=True).cpu()
    background = sum(stems[s] for s in model.sources if s != "vocals")
    background = background.mean(0, keepdim=True).cpu()

    vocals_path = config.TEMP_DIR / "vocals.wav"
    background_path = config.TEMP_DIR / "background.wav"
    torchaudio.save(str(vocals_path), vocals, sr)
    torchaudio.save(str(background_path), background, sr)
    return str(vocals_path), str(background_path)
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_separate.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add dubber/separate.py tests/test_separate.py && git commit -m "feat: Demucs kaynak ayırma + atlama mantığı"
```

---

## Task 5: Diarization (çift motor)

**Files:**
- Create: `dubber/diarize.py`
- Test: `tests/test_diarize.py`

Motor seçimini test ederiz (token varsa pyannote, yoksa speechbrain). Gerçek model çağrısı Colab'da.

- [ ] **Step 1: Failing test yaz**

```python
# tests/test_diarize.py
from dubber.diarize import choose_engine

def test_choose_engine_pyannote_when_token():
    assert choose_engine(forced=None, hf_token="hf_abc") == "pyannote"

def test_choose_engine_speechbrain_when_no_token():
    assert choose_engine(forced=None, hf_token=None) == "speechbrain"

def test_choose_engine_respects_force():
    assert choose_engine(forced="speechbrain", hf_token="hf_abc") == "speechbrain"
    assert choose_engine(forced="pyannote", hf_token=None) == "pyannote"
```

- [ ] **Step 2: Test fail doğrula**

Run: `pytest tests/test_diarize.py -v`
Expected: FAIL — "No module named 'dubber.diarize'"

- [ ] **Step 3: diarize.py yaz**

```python
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
        emb = encoder.encode_batch(chunk).squeeze().detach().numpy()
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
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_diarize.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: `_merge_short` için ek test yaz**

```python
# tests/test_diarize.py (ekle)
from dubber.diarize import _merge_short
from dubber.models import Segment

def test_merge_short_merges_adjacent_same_speaker():
    segs = [Segment(0, 1, "A"), Segment(1.2, 2, "A"), Segment(2.1, 3, "B")]
    out = _merge_short(segs)
    assert len(out) == 2
    assert out[0].start == 0 and out[0].end == 2
    assert out[1].speaker_id == "B"
```

- [ ] **Step 6: Test pass doğrula**

Run: `pytest tests/test_diarize.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add dubber/diarize.py tests/test_diarize.py && git commit -m "feat: çift motor diarization (pyannote/speechbrain) + segment birleştirme"
```

---

## Task 6: Cinsiyet tespiti (gender)

**Files:**
- Create: `dubber/gender.py`
- Test: `tests/test_gender.py`

- [ ] **Step 1: Failing test yaz**

```python
# tests/test_gender.py
from dubber.gender import classify_from_f0

def test_classify_low_pitch_is_male():
    assert classify_from_f0(120.0) == "male"

def test_classify_high_pitch_is_female():
    assert classify_from_f0(210.0) == "female"

def test_classify_nan_is_unknown():
    import math
    assert classify_from_f0(math.nan) == "unknown"
```

- [ ] **Step 2: Test fail doğrula**

Run: `pytest tests/test_gender.py -v`
Expected: FAIL — "No module named 'dubber.gender'"

- [ ] **Step 3: gender.py yaz**

```python
# dubber/gender.py
import math

from dubber import config
from dubber.models import Segment
from dubber.utils import get_logger

logger = get_logger()


def classify_from_f0(f0_hz: float) -> str:
    if f0_hz is None or math.isnan(f0_hz):
        return "unknown"
    return "male" if f0_hz < config.GENDER_F0_THRESHOLD_HZ else "female"


def _mean_f0(y, sr) -> float:
    import librosa
    import numpy as np
    f0, voiced, _ = librosa.pyin(
        y, fmin=70, fmax=400, sr=sr)
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
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_gender.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dubber/gender.py tests/test_gender.py && git commit -m "feat: F0 pitch tabanlı cinsiyet tespiti"
```

---

## Task 7: Konuşma tanıma + hizalama (transcribe)

**Files:**
- Create: `dubber/transcribe.py`
- Test: `tests/test_transcribe.py`

faster-whisper kelime zaman damgaları verir; bunları diarization segmentlerine atayan hizalama mantığını test ederiz.

- [ ] **Step 1: Failing test yaz**

```python
# tests/test_transcribe.py
from dubber.transcribe import assign_words_to_segments
from dubber.models import Segment

def test_assign_words_fills_text_by_overlap():
    segments = [Segment(0, 2, "A"), Segment(2, 4, "B")]
    words = [
        {"start": 0.1, "end": 0.5, "word": "merhaba"},
        {"start": 1.0, "end": 1.4, "word": "dünya"},
        {"start": 2.5, "end": 2.9, "word": "ikinci"},
    ]
    out = assign_words_to_segments(words, segments)
    assert out[0].text == "merhaba dünya"
    assert out[1].text == "ikinci"

def test_assign_words_drops_empty_segments():
    segments = [Segment(0, 2, "A"), Segment(2, 4, "B")]
    words = [{"start": 0.1, "end": 0.5, "word": "tek"}]
    out = assign_words_to_segments(words, segments)
    assert len(out) == 1
    assert out[0].speaker_id == "A"
```

- [ ] **Step 2: Test fail doğrula**

Run: `pytest tests/test_transcribe.py -v`
Expected: FAIL — "No module named 'dubber.transcribe'"

- [ ] **Step 3: transcribe.py yaz**

```python
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
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_transcribe.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dubber/transcribe.py tests/test_transcribe.py && git commit -m "feat: faster-whisper transkript + diarization hizalama"
```

---

## Task 8: Çeviri (translate / NLLB)

**Files:**
- Create: `dubber/translate.py`
- Test: `tests/test_translate.py`

Dil kodu eşleme ve metin parçalama mantığını test ederiz.

- [ ] **Step 1: Failing test yaz**

```python
# tests/test_translate.py
from dubber.translate import to_nllb_lang

def test_to_nllb_lang_known():
    assert to_nllb_lang("en") == "eng_Latn"
    assert to_nllb_lang("de") == "deu_Latn"
    assert to_nllb_lang("ar") == "arb_Arab"

def test_to_nllb_lang_unknown_defaults_english():
    assert to_nllb_lang("zz") == "eng_Latn"
```

- [ ] **Step 2: Test fail doğrula**

Run: `pytest tests/test_translate.py -v`
Expected: FAIL — "No module named 'dubber.translate'"

- [ ] **Step 3: translate.py yaz**

```python
# dubber/translate.py
from dubber import config
from dubber.models import Segment
from dubber.utils import get_logger

logger = get_logger()

_NLLB_MAP = {
    "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn",
    "es": "spa_Latn", "it": "ita_Latn", "ru": "rus_Cyrl",
    "ar": "arb_Arab", "pt": "por_Latn", "nl": "nld_Latn",
    "tr": "tur_Latn",
}
TARGET_TR = "tur_Latn"


def to_nllb_lang(whisper_lang: str) -> str:
    return _NLLB_MAP.get(whisper_lang, "eng_Latn")


def translate_segments(segments: list[Segment], src_lang: str) -> list[Segment]:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    src = to_nllb_lang(src_lang)
    if src == TARGET_TR:
        logger.info("Kaynak zaten Türkçe; çeviri atlanıyor")
        for s in segments:
            s.text_tr = s.text
        return segments

    tok = AutoTokenizer.from_pretrained(config.NLLB_MODEL, src_lang=src,
                                        cache_dir="models")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.NLLB_MODEL,
                                                  cache_dir="models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    bos = tok.convert_tokens_to_ids(TARGET_TR)

    logger.info("Çeviri (NLLB) %s -> tur_Latn, %d segment", src, len(segments))
    for s in segments:
        if not s.text.strip():
            s.text_tr = ""
            continue
        enc = tok(s.text, return_tensors="pt", truncation=True,
                  max_length=512).to(device)
        out = model.generate(**enc, forced_bos_token_id=bos, max_length=512)
        s.text_tr = tok.batch_decode(out, skip_special_tokens=True)[0]
    return segments
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_translate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dubber/translate.py tests/test_translate.py && git commit -m "feat: NLLB-200 Türkçe çeviri + dil kodu eşleme"
```

---

## Task 9: Seslendirme (synthesize / XTTS + Edge yedek)

**Files:**
- Create: `dubber/synthesize.py`
- Test: `tests/test_synthesize.py`

Referans seçimi, atempo timing ve yedek seçimi mantığını test ederiz.

- [ ] **Step 1: Failing test yaz**

```python
# tests/test_synthesize.py
from dubber.synthesize import (pick_reference_segment, timing_factor,
                               edge_voice_for)
from dubber.models import Segment

def test_pick_reference_longest_segment():
    segs = [Segment(0, 1, "A"), Segment(5, 12, "A"), Segment(2, 3, "A")]
    ref = pick_reference_segment(segs, "A")
    assert (ref.start, ref.end) == (5, 12)

def test_pick_reference_none_when_speaker_absent():
    segs = [Segment(0, 1, "A")]
    assert pick_reference_segment(segs, "Z") is None

def test_timing_factor_speeds_up_when_too_long():
    # üretilen 4s, hedef 2s -> 2.0 faktör
    assert timing_factor(generated=4.0, target=2.0) == 2.0

def test_timing_factor_one_when_short_enough():
    assert timing_factor(generated=1.5, target=2.0) == 1.0

def test_edge_voice_for_gender():
    assert edge_voice_for("male", 0) == ("tr-TR-AhmetNeural", 0)
    assert edge_voice_for("female", 1)[0] == "tr-TR-EmelNeural"
```

- [ ] **Step 2: Test fail doğrula**

Run: `pytest tests/test_synthesize.py -v`
Expected: FAIL — "No module named 'dubber.synthesize'"

- [ ] **Step 3: synthesize.py yaz**

```python
# dubber/synthesize.py
from pathlib import Path

from dubber import config
from dubber.models import Segment
from dubber.utils import get_logger, atempo_chain, run_ffmpeg

logger = get_logger()


def pick_reference_segment(segments: list[Segment], speaker_id: str):
    spk = [s for s in segments if s.speaker_id == speaker_id]
    if not spk:
        return None
    return max(spk, key=lambda s: s.duration)


def timing_factor(generated: float, target: float) -> float:
    if target <= 0 or generated <= target:
        return 1.0
    return round(generated / target, 3)


def edge_voice_for(gender: str, speaker_index: int):
    voice = config.EDGE_VOICES.get(gender, config.EDGE_VOICES["male"])
    offsets = config.EDGE_PITCH_OFFSETS
    pitch = offsets[speaker_index % len(offsets)]
    return voice, pitch


def _extract_reference(vocals_path: str, seg: Segment, speaker_id: str) -> Path:
    out = config.TEMP_DIR / "refs" / f"{speaker_id}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = min(seg.duration, 15.0)
    run_ffmpeg(["-i", vocals_path, "-ss", str(seg.start), "-t", str(dur),
                "-ar", "22050", "-ac", "1", str(out)])
    return out


def _fit_duration(clip_path: Path, target: float) -> None:
    import soundfile as sf
    info = sf.info(str(clip_path))
    factor = timing_factor(info.duration, target)
    if factor > 1.0:
        tmp = clip_path.with_suffix(".fit.wav")
        run_ffmpeg(["-i", str(clip_path), "-filter:a",
                    atempo_chain(factor), str(tmp)])
        tmp.replace(clip_path)


def synthesize(segments: list[Segment], genders: dict[str, str],
               vocals_path: str, engine: str = "xtts") -> list[tuple]:
    """Her segment için TR ses üretir. Döner: [(Segment, clip_path), ...]"""
    out_dir = config.TEMP_DIR / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    references: dict[str, Path] = {}
    use_xtts = engine == "xtts"
    tts = None
    if use_xtts:
        try:
            import os
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            from TTS.api import TTS as CoquiTTS
            import torch
            tts = CoquiTTS(config.XTTS_MODEL).to(
                "cuda" if torch.cuda.is_available() else "cpu")
            for spk in set(s.speaker_id for s in segments):
                ref_seg = pick_reference_segment(segments, spk)
                if ref_seg and ref_seg.duration >= config.MIN_REF_SECONDS:
                    references[spk] = _extract_reference(vocals_path, ref_seg, spk)
        except Exception as e:
            logger.warning("XTTS yüklenemedi (%s), Edge TTS'e geçiliyor", e)
            use_xtts = False

    speaker_order = {spk: i for i, spk in
                     enumerate(sorted(set(s.speaker_id for s in segments)))}
    results = []
    for idx, seg in enumerate(segments):
        if not seg.text_tr.strip():
            continue
        clip = out_dir / f"seg_{idx:04d}.wav"
        spk = seg.speaker_id
        ok = False
        if use_xtts and spk in references:
            try:
                tts.tts_to_file(text=seg.text_tr, language="tr",
                                speaker_wav=str(references[spk]),
                                file_path=str(clip))
                ok = True
            except Exception as e:
                logger.warning("XTTS segment %d başarısız (%s), Edge yedeği", idx, e)
        if not ok:
            _edge_tts(seg, genders.get(spk, "unknown"),
                      speaker_order[spk], clip)
        _fit_duration(clip, seg.duration)
        results.append((seg, clip))
    return results


def _edge_tts(seg: Segment, gender: str, speaker_index: int, clip: Path) -> None:
    import asyncio
    import edge_tts

    voice, pitch = edge_voice_for(gender, speaker_index)
    pitch_str = f"{pitch:+d}Hz"

    async def _run():
        communicate = edge_tts.Communicate(seg.text_tr, voice, pitch=pitch_str)
        await communicate.save(str(clip))

    asyncio.run(_run())
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_synthesize.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dubber/synthesize.py tests/test_synthesize.py && git commit -m "feat: XTTS klonlama + Edge yedek seslendirme + atempo timing"
```

---

## Task 10: Yeniden kurma (reconstruct)

**Files:**
- Create: `dubber/reconstruct.py`
- Test: `tests/test_reconstruct.py`

Klip zaman çizelgesine yerleştirme mantığını (pydub overlay konumları) test ederiz.

- [ ] **Step 1: Failing test yaz**

```python
# tests/test_reconstruct.py
from dubber.reconstruct import placement_ms
from dubber.models import Segment

def test_placement_ms_converts_start_seconds():
    results = [(Segment(1.5, 2.0, "A"), "a.wav"),
               (Segment(3.0, 3.4, "B"), "b.wav")]
    assert placement_ms(results) == [(1500, "a.wav"), (3000, "b.wav")]
```

- [ ] **Step 2: Test fail doğrula**

Run: `pytest tests/test_reconstruct.py -v`
Expected: FAIL — "No module named 'dubber.reconstruct'"

- [ ] **Step 3: reconstruct.py yaz**

```python
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
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_reconstruct.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add dubber/reconstruct.py tests/test_reconstruct.py && git commit -m "feat: timeline yerleştirme + arka plan mix + normalize"
```

---

## Task 11: Pipeline orkestrasyonu

**Files:**
- Create: `dubber/pipeline.py`
- Test: `tests/test_pipeline.py`

Tüm aşamalar mock'lanarak çağrı sırası ve veri akışı test edilir.

- [ ] **Step 1: Failing test yaz**

```python
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
```

- [ ] **Step 2: Test fail doğrula**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — "No module named 'dubber.pipeline'"

- [ ] **Step 3: pipeline.py yaz**

```python
# dubber/pipeline.py
from dubber import config
from dubber.preprocess import preprocess
from dubber.separate import separate
from dubber.diarize import diarize
from dubber.gender import assign_genders
from dubber.transcribe import transcribe
from dubber.translate import translate_segments
from dubber.synthesize import synthesize
from dubber.reconstruct import reconstruct
from dubber.utils import get_logger, format_timestamp

logger = get_logger()


def _audio_duration(path: str) -> float:
    import soundfile as sf
    return sf.info(path).duration


def run(input_path: str, output_path: str, *, background: bool = True,
        diarizer: str | None = None, tts_engine: str = "xtts") -> str:
    logger.info("=== Dublaj başlıyor: %s ===", input_path)

    audio16k = preprocess(input_path)
    vocals, bg = separate(str(audio16k), enabled=background)
    segments = diarize(str(vocals), forced=diarizer)
    genders = assign_genders(str(vocals), segments)

    for s in segments:
        logger.info("%s (%s): [%s–%s]", s.speaker_id, s.gender,
                    format_timestamp(s.start), format_timestamp(s.end))

    segments, src_lang = transcribe(str(vocals), segments)
    segments = translate_segments(segments, src_lang)
    results = synthesize(segments, genders, str(vocals), engine=tts_engine)

    total = _audio_duration(str(audio16k))
    out = reconstruct(results, bg, total, output_path)
    logger.info("=== Tamamlandı: %s ===", out)
    return str(out)
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add dubber/pipeline.py tests/test_pipeline.py && git commit -m "feat: pipeline orkestrasyonu"
```

---

## Task 12: CLI (dub.py)

**Files:**
- Create: `dub.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Failing test yaz**

```python
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
```

- [ ] **Step 2: Test fail doğrula**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — "No module named 'dub'"

- [ ] **Step 3: dub.py yaz**

```python
# dub.py
import argparse
import sys

from dubber import config
from dubber.pipeline import run


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MP3 AI Türkçe Dublaj")
    p.add_argument("input", help="Girdi MP3/WAV dosyası")
    p.add_argument("--output", default=None, help="Çıktı MP3 yolu")
    p.add_argument("--no-background", dest="background", action="store_false",
                   help="Arka plan ayırmayı atla")
    p.add_argument("--tts", choices=["xtts", "edge"], default="xtts",
                   help="Seslendirme motoru")
    p.add_argument("--diarizer", choices=["pyannote", "speechbrain"],
                   default=None, help="Diarization motorunu zorla")
    p.set_defaults(background=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or str(config.OUTPUT_DIR / "output_dubbed.mp3")
    run(args.input, output, background=args.background,
        diarizer=args.diarizer, tts_engine=args.tts)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Test pass doğrula**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Tüm test paketini çalıştır**

Run: `pytest -v`
Expected: PASS (tüm testler ~26 passed)

- [ ] **Step 6: Commit**

```bash
git add dub.py tests/test_cli.py && git commit -m "feat: CLI giriş noktası (dub.py)"
```

---

## Task 13: Colab notebook

**Files:**
- Create: `notebook/dublaj_colab.ipynb`

Notebook elle hazırlanır (JSON). Aşağıdaki hücreleri sırasıyla içerir. Her hücre `nbformat` ile yazılabilir veya Colab'da elle oluşturulup commit edilir.

- [ ] **Step 1: nbformat ile notebook üret**

```python
# scripts/make_notebook.py (geçici, sonra silinebilir)
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# MP3 AI Türkçe Dublaj — Colab\n"
    "GPU çalışma zamanı seçin: Runtime > Change runtime type > GPU."))

cells.append(nbf.v4.new_code_cell(
    "# 1. Repo + sistem bağımlılıkları\n"
    "!apt-get -qq install -y ffmpeg\n"
    "!git clone https://github.com/KULLANICI/dublaj.git || true\n"
    "%cd dublaj"))

cells.append(nbf.v4.new_code_cell(
    "# 2. Python bağımlılıkları\n"
    "!pip install -q -r requirements.txt"))

cells.append(nbf.v4.new_code_cell(
    "# 3. (Opsiyonel) pyannote için ücretsiz HF token\n"
    "import os\n"
    "os.environ['HF_TOKEN'] = ''  # boş bırakırsan speechbrain kullanılır\n"
    "os.environ['COQUI_TOS_AGREED'] = '1'  # XTTS lisans kabulü"))

cells.append(nbf.v4.new_code_cell(
    "# 4. Ses dosyasını yükle\n"
    "from google.colab import files\n"
    "up = files.upload()\n"
    "import shutil, os\n"
    "name = list(up.keys())[0]\n"
    "shutil.move(name, f'input/{name}')\n"
    "INPUT = f'input/{name}'"))

cells.append(nbf.v4.new_code_cell(
    "# 5. Dublajı çalıştır\n"
    "!python dub.py \"$INPUT\""))

cells.append(nbf.v4.new_code_cell(
    "# 6. Sonucu indir\n"
    "from google.colab import files\n"
    "files.download('output/output_dubbed.mp3')"))

nb.cells = cells
with open("notebook/dublaj_colab.ipynb", "w") as f:
    nbf.write(nb, f)
print("yazıldı")
```

Run: `pip install nbformat && python scripts/make_notebook.py`
Expected: "yazıldı" ve `notebook/dublaj_colab.ipynb` oluşur.

- [ ] **Step 2: Geçici scripti sil**

Run: `rm scripts/make_notebook.py && rmdir scripts 2>/dev/null || true`

- [ ] **Step 3: Commit**

```bash
git add notebook/dublaj_colab.ipynb && git commit -m "feat: Colab notebook"
```

---

## Task 14: README + Colab uçtan uca duman testi

**Files:**
- Create: `README.md`

- [ ] **Step 1: README.md yaz**

````markdown
# MP3 AI Türkçe Dublaj

MP3/WAV sesini konuşmacılara ayırıp, cinsiyet tespiti yapıp, her karaktere
orijinalinden klonlanmış (XTTS) sesle Türkçe dublaj üreten otomatik sistem.

## Yerel kullanım
```bash
pip install -r requirements.txt
python dub.py input.mp3
# çıktı: output/output_dubbed.mp3
```

## Bayraklar
- `--output PATH` çıktı yolu
- `--no-background` arka plan ayırmayı atla (daha hızlı)
- `--tts edge` XTTS yerine Edge TTS yedeği
- `--diarizer pyannote|speechbrain` diarization motorunu zorla

## Ortam değişkenleri
- `HF_TOKEN` (opsiyonel): pyannote için ücretsiz HF token. Yoksa speechbrain kullanılır.
- `COQUI_TOS_AGREED=1`: XTTS lisans kabulü (ticari olmayan kullanım).

## Colab
`notebook/dublaj_colab.ipynb` dosyasını Colab'da açın, GPU seçin, hücreleri çalıştırın.

## Testler
```bash
pip install pytest && pytest -v
```
````

- [ ] **Step 2: README commit**

```bash
git add README.md && git commit -m "docs: README kullanım kılavuzu"
```

- [ ] **Step 3: Colab duman testi (elle, GPU gerekir)**

Colab'da notebook'u baştan sona çalıştır:
1. GPU runtime seç.
2. Kısa (~30 sn, 2 konuşmacı, 1 erkek 1 kadın) test MP3'ü yükle.
3. `python dub.py input.mp3` çalıştır.
Expected:
- `logs/run.log`'da her konuşmacı için `SPEAKER_xx (male/female): [mm:ss–mm:ss]` satırları.
- `output/output_dubbed.mp3` üretilir, Türkçe konuşma duyulur.
- İki konuşmacı farklı/tutarlı seslerle gelir.
- HF_TOKEN boşken speechbrain ile de uçtan uca çalışır.

- [ ] **Step 4: Sonuçları doğrula ve gerekirse düzelt**

Hata çıkarsa: superpowers:systematic-debugging skill ile kök neden bul, ilgili task'a dön.

---

## Self-Review Sonucu (plan yazarı)

**Spec kapsamı:** Spec'in tüm bölümleri karşılandı:
- §3 önişleme → Task 3; §4/separate(arka plan) → Task 4 + reconstruct Task 10;
- diarization çift motor → Task 5; cinsiyet → Task 6; ASR+hizalama → Task 7;
- çeviri NLLB → Task 8; XTTS klonlama + Edge yedek + atempo → Task 9;
- reconstruct/mix → Task 10; pipeline → Task 11; CLI → Task 12;
- Colab notebook → Task 13; proje yapısı → Task 0; README/duman testi → Task 14.

**Tip tutarlılığı:** `Segment` alanları (start, end, speaker_id, gender, text, text_tr)
tüm modüllerde tutarlı. `synthesize` döner tipi `[(Segment, clip_path)]`, `reconstruct`
ve `pipeline` bununla uyumlu. `transcribe` ve `pipeline` `(segments, lang)` tuple döner.

**Placeholder taraması:** Saf-mantık adımlarının hepsinde gerçek kod var. Model çağrıları
gerçek API kullanımıyla yazıldı; GPU gerektirenler Colab duman testinde (Task 14) doğrulanır.
````
