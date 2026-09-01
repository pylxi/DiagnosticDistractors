"""
Tests for pipeline/phonetic_swaps.py -- the L/R, B/V, TH->S/Z, F/H
substitution-rule signal in the spelling branch.
"""
from pipeline import phonetic_swaps as ps


def test_candidates_light_matches_known_rule_hits():
    # Confirmed real output: "light" only has an l->r substring hit (light
    # -> right) and an h->f hit (light -> ligft, not a real word but the
    # rule doesn't know that -- filtering to real CEFR-J words is
    # find_valid_swaps()'s job, not candidates()'s).
    got = dict(ps.candidates("light"))
    assert got == {"right": "l->r", "ligft": "h->f"}


def test_candidates_never_returns_the_original_word():
    for cand, rule in ps.candidates("light"):
        assert cand != "light"


def test_candidates_is_deduplicated():
    results = list(ps.candidates("light"))
    words = [c for c, _ in results]
    assert len(words) == len(set(words))


def test_candidates_empty_when_no_rule_letters_present():
    # "cat" contains none of l, r, b, v, f, h, or the th/si/shi digraphs
    assert list(ps.candidates("cat")) == []


def test_find_valid_swaps_keeps_real_cefr_j_match():
    valid = ps.find_valid_swaps("light", target_pos="adjective", target_level="A1")
    words = {v["word"] for v in valid}
    assert "right" in words


def test_find_valid_swaps_drops_candidates_failing_the_cefr_gate():
    # think -> "sink"/"zink" via th->s/th->z, but "sink" is only a CEFR-J
    # verb at B1, and A1's ±1 window is [A1, A2] -- B1 is out of range, so
    # nothing should survive even though the spelling substitution is valid.
    assert ps.find_valid_swaps("think", target_pos="verb", target_level="A1") == []


def test_find_valid_swaps_respects_pos_filter():
    # "right" also exists as a noun/adverb at other levels -- asking for a
    # POS "right" doesn't have at the target level should still work for
    # the POS it does have, and not silently match the wrong POS's level.
    valid = ps.find_valid_swaps("light", target_pos="adjective", target_level="A1")
    for v in valid:
        assert v["pos"] == "adjective"
