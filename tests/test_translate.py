# tests/test_translate.py
from dubber.translate import to_nllb_lang

def test_to_nllb_lang_known():
    assert to_nllb_lang("en") == "eng_Latn"
    assert to_nllb_lang("de") == "deu_Latn"
    assert to_nllb_lang("ar") == "arb_Arab"

def test_to_nllb_lang_unknown_defaults_english():
    assert to_nllb_lang("zz") == "eng_Latn"
