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
