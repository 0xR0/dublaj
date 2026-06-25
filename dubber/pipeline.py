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
