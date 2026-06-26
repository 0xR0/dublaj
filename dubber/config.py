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
BACKGROUND_GAIN_DB = -2         # arka plan/müzik miksaj seviyesi (yüksek = daha duyulur)

# XTTS ses modu: "preset" (hazır seslerden konuşmacıya ata) | "clone" (kişinin kendi sesi)
XTTS_VOICE_MODE = "preset"
# Cinsiyete göre tercih edilen XTTS-v2 hazır sesleri (model destekliyorsa kullanılır)
XTTS_VOICES = {
    "female": ["Claribel Dervla", "Daisy Studious", "Sofia Hellen",
               "Alison Dietlinde", "Tammie Ema"],
    "male": ["Andrew Chipper", "Damien Black", "Viktor Eka",
             "Luis Moray", "Aaron Dreschner"],
}

# Edge TTS yedeği için cinsiyet→ses
EDGE_VOICES = {"male": "tr-TR-AhmetNeural", "female": "tr-TR-EmelNeural"}
# Aynı cinsiyette çoklu konuşmacı için Edge pitch ofsetleri (Hz)
EDGE_PITCH_OFFSETS = [0, -15, 10, -8, 18]

WHISPER_MODEL = "large-v3"
NLLB_MODEL = "facebook/nllb-200-distilled-1.3B"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

for _d in (INPUT_DIR, TEMP_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
