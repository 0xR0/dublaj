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
