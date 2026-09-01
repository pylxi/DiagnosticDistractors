"""
Tests for pipeline/spelling_branch.py -- the four-signal priority cascade
(gairaigo_exact -> phonetic_swap -> gairaigo_near -> levenshtein), run
against real CEFR-J/JMdict data. Pure Python, no model/network needed.
"""
import pytest

import spelling_branch as spb


def test_result_shape():
    out = spb.spelling_distractors("ask", n=3)
    assert set(out.keys()) == {
        "target", "target_pos", "target_level", "distractors",
        "all_found", "sufficient", "funnel",
    }
    assert out["target"] == "ask"
    assert out["target_pos"] == "verb"
    assert out["target_level"] == "A1"


def test_distractors_is_capped_at_n_but_all_found_is_not():
    out = spb.spelling_distractors("ask", n=2)
    assert len(out["distractors"]) <= 2
    assert len(out["all_found"]) >= len(out["distractors"])


def test_funnel_has_one_entry_per_signal_in_priority_order():
    out = spb.spelling_distractors("ask", n=3)
    sources = [f["source"] for f in out["funnel"]]
    assert sources == ["gairaigo_exact", "phonetic_swap", "gairaigo_near", "levenshtein"]


def test_cascade_stops_once_quota_is_met():
    # confirmed real behavior: "brush" gets its quota of 3 from
    # gairaigo_near alone, so levenshtein must be recorded as skipped, not
    # silently run-but-empty.
    out = spb.spelling_distractors("brush", n=3)
    funnel_by_source = {f["source"]: f for f in out["funnel"]}
    assert funnel_by_source["gairaigo_near"]["ran"] is True
    assert funnel_by_source["gairaigo_near"]["passed_cefr_gate"] >= 3
    assert funnel_by_source["levenshtein"]["ran"] is False
    assert "quota" in funnel_by_source["levenshtein"]["reason"]


def test_sufficient_flag_matches_all_found_length():
    out = spb.spelling_distractors("ask", n=3)
    assert out["sufficient"] == (len(out["all_found"]) >= 3)


def test_every_candidate_passes_the_cefr_gate_it_claims():
    import cefr_lookup as cefr
    out = spb.spelling_distractors("ask", n=8)
    for cand in out["all_found"]:
        ok, level, pos = cefr.matches(
            cand["word"], target_pos=out["target_pos"], target_level=out["target_level"]
        )
        assert ok is True
        assert level == cand["level"]
        assert pos == cand["pos"]


def test_no_candidate_is_the_target_word_itself():
    out = spb.spelling_distractors("ask", n=8)
    words = [c["word"] for c in out["all_found"]]
    assert "ask" not in words


def test_unknown_word_without_explicit_pos_level_raises():
    with pytest.raises(ValueError):
        spb.spelling_distractors("zzznotarealword")


def test_unknown_word_with_explicit_pos_level_does_not_raise():
    # a word need not be IN the CEFR-J list itself to be scored, as long as
    # its level/POS are supplied -- only auto-detection depends on cefr.entries()
    out = spb.spelling_distractors("zzznotarealword", target_pos="noun", target_level="A1", n=2)
    assert out["target_pos"] == "noun"
    assert out["target_level"] == "A1"
    assert isinstance(out["distractors"], list)
