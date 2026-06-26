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
CHILD_F0_THRESHOLD_HZ = 255.0    # üstü çocuk (female ile child arası sınır)
MIN_REF_SECONDS = 3.0            # XTTS referansı için alt sınır
PREFERRED_REF_SECONDS = 6.0     # tercih edilen referans süresi
MAX_ATEMPO = 1.3                # zamanlama sığdırmada üst hız sınırı
BACKGROUND_GAIN_DB = -2         # arka plan/müzik miksaj seviyesi (yüksek = daha duyulur)

# Yetişkin XTTS ses modu: "clone" (kişinin kendi/orijinal sesi) | "preset" (hazır seslerden ata)
XTTS_VOICE_MODE = "clone"
# Çocuk konuşmacı için: "xtts" (XTTS ile çocuğun kendi sesini klonla) | "off" (yetişkin gibi davran)
CHILD_VOICE_MODE = "xtts"
# Cinsiyete göre tercih edilen XTTS-v2 hazır sesleri (preset modunda, model destekliyorsa)
XTTS_VOICES = {
    "female": ["Claribel Dervla", "Daisy Studious", "Gracie Wise",
               "Tammie Ema", "Alison Dietlinde", "Ana Florence",
               "Annmarie Nele", "Asya Anara", "Brenda Stern",
               "Gitta Nikolina"],
    "male": ["Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler",
             "Royston Min", "Viktor Eka", "Abrahan Mavros",
             "Damien Black", "Gilberto Mathias", "Ilkin Urancan",
             "Kumar Dahl"],
    "child": [],   # XTTS'te etiketli çocuk sesi yok; çocuk klonlanır
}

# Edge TTS yedeği için cinsiyet→ses
EDGE_VOICES = {"male": "tr-TR-AhmetNeural", "female": "tr-TR-EmelNeural",
               "child": "tr-TR-EmelNeural"}
# Aynı cinsiyette çoklu konuşmacı için Edge pitch ofsetleri (Hz)
EDGE_PITCH_OFFSETS = [0, -15, 10, -8, 18]
EDGE_CHILD_PITCH = 35           # çocuk sesi için Edge yedeğinde pitch yükseltme (Hz)

WHISPER_MODEL = "large-v3"
NLLB_MODEL = "facebook/nllb-200-distilled-1.3B"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

for _d in (INPUT_DIR, TEMP_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
