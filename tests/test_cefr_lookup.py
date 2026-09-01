"""
Tests for pipeline/cefr_lookup.py -- the CEFR-J word/POS/level gate every
candidate from both branches has to clear. Run against the real CEFR-J
wordlist (data/CEFR-J/), not a fixture -- these are the real level/POS
assignments the pipeline actually uses.
"""
import cefr_lookup as cefr


def test_entries_known_word_returns_pos_level_pairs():
    assert ("verb", "A1") in cefr.entries("ask")


def test_entries_unknown_word_returns_empty_list():
    assert cefr.entries("zzznotarealword") == []


def test_entries_is_case_and_whitespace_insensitive():
    assert cefr.entries("ASK") == cefr.entries(" ask ")


def test_levels_for_and_pos_for_wrap_entries():
    # "light" has entries at more than one POS -- levels_for/pos_for should
    # be the deduped, sorted projections of entries().
    entries = cefr.entries("light")
    assert entries, "fixture word 'light' must exist in CEFR-J for this test to mean anything"
    assert set(cefr.levels_for("light")) == {lvl for _, lvl in entries}
    assert set(cefr.pos_for("light")) == {pos for pos, _ in entries}
    # levels_for must come back in CEFR order, not alphabetical/insertion order
    assert cefr.levels_for("light") == sorted(cefr.levels_for("light"), key=cefr.LEVEL_INDEX.get)


def test_is_in_cefr_j():
    assert cefr.is_in_cefr_j("ask") is True
    assert cefr.is_in_cefr_j("zzznotarealword") is False


def test_all_words_contains_known_fixtures():
    words = set(cefr.all_words())
    for w in ("ask", "light", "right", "glass"):
        assert w in words


def test_adjacent_levels_radius_1_no_wraparound_at_bottom():
    # A1 has no level below it -- radius-1 must not error or wrap
    assert cefr.adjacent_levels("A1", radius=1) == ["A1", "A2"]


def test_adjacent_levels_radius_1_no_wraparound_at_top():
    assert cefr.adjacent_levels("C2", radius=1) == ["C1", "C2"]


def test_adjacent_levels_radius_1_middle():
    assert cefr.adjacent_levels("B1", radius=1) == ["A2", "B1", "B2"]


def test_matches_exact_level_and_pos():
    ok, level, pos = cefr.matches("ask", target_pos="verb", target_level="A1")
    assert ok is True
    assert level == "A1"
    assert pos == "verb"


def test_matches_falls_back_to_adjacent_level():
    # "bank" as a verb is A2, not A1 -- this is the real ±1 fallback the
    # semantic branch's "control" tier picks for "ask" rely on.
    ok, level, pos = cefr.matches("bank", target_pos="verb", target_level="A1")
    assert ok is True
    assert level == "A2"


def test_matches_rejects_adjacent_when_disallowed():
    ok, level, pos = cefr.matches(
        "bank", target_pos="verb", target_level="A1", allow_adjacent=False
    )
    assert ok is False
    assert level is None
    assert pos is None


def test_matches_rejects_wrong_pos_even_if_word_exists():
    # "light" exists as a noun/adjective at A1 but as a verb only at B1 --
    # asking for verb+A1 must fail, not silently match a different POS.
    ok, level, pos = cefr.matches("light", target_pos="verb", target_level="A1", allow_adjacent=False)
    assert ok is False


def test_matches_unknown_word_returns_false_none_none():
    assert cefr.matches("zzznotarealword", target_pos="verb", target_level="A1") == (False, None, None)


def test_matches_with_no_target_level_returns_first_entry():
    ok, level, pos = cefr.matches("ask", target_pos="verb")
    assert ok is True
    assert (pos, level) in cefr.entries("ask")
