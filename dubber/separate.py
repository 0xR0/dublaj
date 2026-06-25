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
    # htdemucs 44.1kHz'de eğitildi; modelin beklediği örnekleme hızına getir.
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate
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
