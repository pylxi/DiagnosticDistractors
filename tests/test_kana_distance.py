"""
Tests for pipeline/kana_distance.py -- the mora-aware katakana distance used to
rank gairaigo near-neighbours. Needs pykakasi (a core dependency); pure
computation, no data files.
"""
from pipeline import kana_distance as kd


def test_identical_readings_are_zero():
    assert kd.mora_distance("ライト", "ライト") == 0.0


def test_long_vowel_difference_is_nearly_free():
    # ラン vs ラーン differ only in vowel length -- the most common confusion.
    d = kd.mora_distance("ラン", "ラーン")
    assert 0 < d <= 0.3


def test_same_row_vowel_swap_beats_consonant_swap():
    # ラ->リ (same ラ行 row, different vowel) must cost less than ラ->ダ
    # (different consonant), so vowel look-alikes rank first.
    assert kd.mora_distance("ラ", "リ") < kd.mora_distance("ラ", "ダ")


def test_run_ranks_vowel_lookalikes_above_distant_readings():
    # ring/ラン~リン and turn/ラン~ターン are closer than fall/ラン~ダウン.
    run = "ラン"
    assert kd.mora_distance(run, "リン") < kd.mora_distance(run, "ダウン")
    assert kd.mora_distance(run, "ターン") < kd.mora_distance(run, "ダウン")


def test_mora_segmentation_fuses_small_kana_and_long_marks():
    assert kd.to_morae("ウィン") == ["ウィ", "ン"]
    assert kd.to_morae("ラーメン") == ["ラー", "メ", "ン"]
