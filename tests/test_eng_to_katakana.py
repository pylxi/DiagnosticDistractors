"""
Tests for pipeline/eng_to_katakana.py -- the rule-based English->katakana
fallback used when JMdict has no loanword reading for a word. Pure computation,
no data files.
"""
from pipeline import eng_to_katakana as e2k


def test_common_short_words_transliterate_as_expected():
    cases = {
        "ask": "アスク", "run": "ラン", "bus": "バス", "task": "タスク",
        "mask": "マスク", "desk": "デスク", "glass": "グラス", "light": "ライト",
        "sun": "サン", "help": "ヘルプ",
    }
    for word, kata in cases.items():
        assert e2k.to_katakana(word) == kata, word


def test_final_n_is_moraic():
    assert e2k.to_katakana("run").endswith("ン")


def test_consonant_cluster_gets_epenthetic_vowels():
    # "ask" = a + s + k -> ア + ス + ク (no bare consonants)
    assert e2k.to_katakana("ask") == "アスク"


def test_empty_or_nonalpha_is_empty():
    assert e2k.to_katakana("") == ""
    assert e2k.to_katakana("123") == ""


def test_transliteration_is_deterministic():
    assert e2k.to_katakana("ask") == e2k.to_katakana("ask")
