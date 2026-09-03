"""
Tests for pipeline/phonetic_swaps.py -- the L/R, B/V, TH->S/Z, F/H
substitution-rule signal in the spelling branch.
"""
from pipeline import phonetic_swaps as ps


def test_candidates_light_matches_known_rule_hits():
    # Confirmed real output: "light" has an l->r substring hit (light -> right)
    # and an h->f hit (light -> ligft, not a real word -- filtering to real
    # words is the caller's job, not candidates()'s).
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
