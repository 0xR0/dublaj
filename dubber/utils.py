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
