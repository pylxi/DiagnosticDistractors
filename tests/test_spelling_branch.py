"""
Tests for pipeline/spelling_branch.py -- the four-signal priority cascade
(gairaigo_exact -> phonetic_swap -> gairaigo_near -> levenshtein), run
against real CEFR-J/JMdict data. Pure Python, no model/network needed.
"""
import pytest

from pipeline import spelling_branch as spb


def test_result_shape():
    out = spb.spelling_distractors("ask", n=3)
    assert set(out.keys()) == {
        "target", "target_pos", "target_level", "distractors",
        "all_found", "sufficient", "funnel", "n_corroborated",
    }
    assert out["target"] == "ask"
    assert out["target_pos"] == "verb"
    assert out["target_level"] == "A1"
    for c in out["all_found"]:
        assert set(c.keys()) == {"word", "level", "pos", "source", "sources", "score"}


def test_distractors_is_capped_at_n_but_all_found_is_not():
    out = spb.spelling_distractors("ask", n=2)
    assert len(out["distractors"]) <= 2
    assert len(out["all_found"]) >= len(out["distractors"])


def test_funnel_has_one_entry_per_signal_in_priority_order():
    out = spb.spelling_distractors("ask", n=3)
    sources = [f["source"] for f in out["funnel"]]
    assert sources == ["gairaigo_exact", "phonetic_swap", "gairaigo_near", "levenshtein"]


def test_all_signals_run_and_report_gate_counts():
    # No cascade short-circuit any more: every signal runs and reports how many
    # raw candidates it proposed and how many passed the CEFR gate.
    out = spb.spelling_distractors("glass", n=3)
    for f in out["funnel"]:
        assert f["passed_cefr_gate"] <= f["raw_candidates"]
        assert "ran" not in f and "running_total" not in f


def test_corroborated_candidates_outrank_single_signal_ones():
    # run/verb look-alikes found by BOTH spelling (levenshtein) and sound
    # (gairaigo_near) -- rain/ring/turn/win -- rank at the top; ruin (a strong
    # levenshtein edit-1 the old stop-at-quota cascade skipped) now makes the
    # cut; and phone (katakana-near only, weak) does not.
    out = spb.spelling_distractors("run", target_pos="verb", target_level="A1", n=8)
    top = [c["word"] for c in out["distractors"]]
    assert {"rain", "ring", "turn", "win"} <= set(top)
    assert "ruin" in top          # levenshtein edit-1, formerly crowded out
    assert "phone" not in top     # katakana-near only, weak
    assert out["n_corroborated"] >= 3
    # a corroborated candidate scores at least as high as a single-signal one
    by_word = {c["word"]: c for c in out["all_found"]}
    assert by_word["rain"]["score"] > by_word["phone"]["score"]
    assert len(by_word["rain"]["sources"]) >= 2


def test_no_multi_word_candidates():
    # Multi-word CEFR-J entries (e.g. "all right") are not plausible look-alike
    # distractors for a single spelled target and must be filtered out.
    for w in ("light", "run", "glass"):
        out = spb.spelling_distractors(w, n=8)
        assert not [c for c in out["all_found"] if " " in c["word"]]


def test_sufficient_flag_matches_all_found_length():
    out = spb.spelling_distractors("ask", n=3)
    assert out["sufficient"] == (len(out["all_found"]) >= 3)


def test_every_candidate_passes_the_cefr_gate_it_claims():
    from pipeline import cefr_lookup as cefr
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
