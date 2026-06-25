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
    # Aşırı hızlandırma sesi bozar; MAX_ATEMPO ile sınırla (spec 5.4).
    factor = min(timing_factor(info.duration, target), config.MAX_ATEMPO)
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
