"""
Tests for pipeline/gairaigo.py -- katakana-collision and near-neighbor
loanword lookups, built on the real pipeline/cache/loanwords.json (built
from JMdict by build_loanword_index.py). Run build_loanword_index.py first
if that cache file doesn't exist yet.
"""
import gairaigo


def test_exact_katakana_collisions_bus_finds_bath_and_bass():
    # bus/bath/bass/double bass all share the basu katakana reading -- the
    # canonical example of the strongest spelling-branch signal.
    collisions = gairaigo.exact_katakana_collisions("bus")
    assert "バス" in collisions
    assert {"bath", "bass", "double bass"} <= collisions["バス"]


def test_exact_katakana_collisions_excludes_the_word_itself():
    collisions = gairaigo.exact_katakana_collisions("bus")
    for words in collisions.values():
        assert "bus" not in words


def test_exact_katakana_collisions_respects_exclude_words():
    baseline = gairaigo.exact_katakana_collisions("bus")
    to_drop = next(iter(baseline["バス"]))
    filtered = gairaigo.exact_katakana_collisions("bus", exclude_words={to_drop})
    assert to_drop not in filtered.get("バス", set())


def test_exact_katakana_collisions_unknown_word_is_empty():
    assert dict(gairaigo.exact_katakana_collisions("zzznotarealword")) == {}


def test_near_katakana_neighbors_glass_finds_class_at_distance_one():
    neighbors = gairaigo.near_katakana_neighbors("glass", max_dist=1, top_n=10)
    by_word = {n["word"]: n for n in neighbors}
    assert "class" in by_word
    assert by_word["class"]["distance"] == 1


def test_near_katakana_neighbors_never_returns_zero_distance():
    # the function's own guard is `if 0 < dist <= max_dist`, i.e. an exact
    # reading match to yourself should never show up as a "near neighbor"
    for n in gairaigo.near_katakana_neighbors("glass", max_dist=2, top_n=30):
        assert n["distance"] > 0
        assert n["word"] != "glass"


def test_near_katakana_neighbors_respects_top_n():
    neighbors = gairaigo.near_katakana_neighbors("glass", max_dist=2, top_n=3)
    assert len(neighbors) <= 3


def test_near_katakana_neighbors_word_with_no_loanword_form_is_empty():
    assert gairaigo.near_katakana_neighbors("zzznotarealword", max_dist=2) == []


def test_near_katakana_neighbors_sorted_by_distance_ascending():
    neighbors = gairaigo.near_katakana_neighbors("glass", max_dist=2, top_n=30)
    distances = [n["distance"] for n in neighbors]
    assert distances == sorted(distances)
