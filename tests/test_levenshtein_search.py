"""
Tests for pipeline/levenshtein_search.py -- the plain English-spelling
edit-distance fallback signal (bottom of the spelling branch's cascade).
"""
from pipeline import levenshtein_search as lev


def test_neighbors_ask_finds_known_close_words():
    results = lev.neighbors("ask", target_pos="verb", target_level="A1", max_dist=2, top_n=10)
    words = {r["word"] for r in results}
    # confirmed real output at the time these tests were written
    assert {"add", "bank", "mark"} <= words


def test_neighbors_excludes_the_target_word_itself():
    results = lev.neighbors("ask", target_pos="verb", target_level="A1", max_dist=3)
    assert all(r["word"] != "ask" for r in results)


def test_neighbors_respects_max_dist():
    results = lev.neighbors("ask", target_pos="verb", target_level="A1", max_dist=1)
    assert all(r["distance"] <= 1 for r in results)


def test_neighbors_respects_top_n():
    results = lev.neighbors("ask", target_pos="verb", target_level="A1", max_dist=3, top_n=2)
    assert len(results) <= 2


def test_neighbors_exclude_param_removes_specific_words():
    baseline = lev.neighbors("ask", target_pos="verb", target_level="A1", max_dist=2, top_n=10)
    assert baseline, "expected at least one neighbor for this fixture"
    to_drop = baseline[0]["word"]
    filtered = lev.neighbors("ask", target_pos="verb", target_level="A1", max_dist=2,
                              top_n=10, exclude={to_drop})
    assert all(r["word"] != to_drop for r in filtered)


def test_neighbors_sorted_by_distance_then_word():
    results = lev.neighbors("ask", target_pos="verb", target_level="A1", max_dist=3, top_n=30)
    keys = [(r["distance"], r["word"]) for r in results]
    assert keys == sorted(keys)


def test_neighbors_deduplicates_multi_pos_words():
    results = lev.neighbors("ask", target_pos="verb", target_level="A1", max_dist=3, top_n=30)
    words = [r["word"] for r in results]
    assert len(words) == len(set(words))
