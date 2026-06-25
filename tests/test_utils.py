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
