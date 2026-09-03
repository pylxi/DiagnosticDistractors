"""
Tests for pipeline/gairaigo.py -- the JMdict loanword lookup, built on the real
pipeline/cache/loanwords.json (built from JMdict by build_loanword_index.py).
Run build_loanword_index.py first if that cache file doesn't exist yet.
"""
from pipeline import gairaigo


def test_katakana_forms_for_known_loanword():
    romajis = {f["romaji"] for f in gairaigo.katakana_forms_for("bus", include_non_eng_source=True)}
    assert "basu" in romajis


def test_katakana_forms_for_unknown_word_is_empty():
    assert gairaigo.katakana_forms_for("zzznotarealword", include_non_eng_source=True) == []


def test_primary_head_is_the_first_gloss():
    # "talk" is the primary gloss of トーク (which also glosses chat/banter);
    # career's own reading is キャリア.
    forms = {f["katakana"]: gairaigo._primary_head(f)
             for f in gairaigo.katakana_forms_for("talk", include_non_eng_source=True)}
    assert forms.get("トーク") == "talk"


def test_loanword_index_excludes_native_words_with_katakana_readings():
    # Regression: 刷毛 (hake, "brush") is a NATIVE word read はけ/ハケ, not a
    # gairaigo. It used to leak into the index; brush must now resolve only to
    # its real katakana loanword ブラシ (burashi), never the native "hake".
    romajis = {f["romaji"] for f in gairaigo.katakana_forms_for("brush",
                                                                include_non_eng_source=True)}
    assert "hake" not in romajis
    assert romajis == {"burashi"}
