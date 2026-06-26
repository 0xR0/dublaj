# tests/test_gender.py
from dubber.gender import classify_from_f0

def test_classify_low_pitch_is_male():
    assert classify_from_f0(120.0) == "male"

def test_classify_high_pitch_is_female():
    assert classify_from_f0(210.0) == "female"

def test_classify_very_high_pitch_is_child():
    assert classify_from_f0(300.0) == "child"

def test_classify_nan_is_unknown():
    import math
    assert classify_from_f0(math.nan) == "unknown"
