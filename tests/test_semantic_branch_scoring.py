"""
Tests for the masked-LM-scoring half of pipeline/semantic_branch.py:
score_cefr_candidates() and semantic_distractors(). Unlike
test_semantic_branch_stem.py, these DO need torch/transformers and the
cached roberta-large weights, so they can only run in YOUR native venv (with
network/model access) -- not in the sandboxed device_bash environment and not
in Claude's cloud container, both of which block huggingface.co/pytorch.org.
Every test skips itself cleanly if the model can't be loaded, so
`pytest tests/` is always safe to run anywhere; only here, on your machine, do
these actually execute.

Two of the tests below are regression checks pinned to the real output of
`python3 pipeline/run_batch.py` (pipeline/cache/batch_result.json, captured
2026-09-01) rather than hand-picked expectations -- if a future change to
the model, the scoring code, or the pruned FastText vectors shifts which
words get picked, these are the tests that will tell you. lm_score/cosine
values are compared with a tolerance (pytest.approx) rather than exact
equality, since float32 model math can differ in its last few bits across
hardware/library versions without the ranking actually changing; the
*words* chosen and the funnel counts are checked exactly, since those
should be perfectly reproducible for the same model/data.
"""
import pytest

import cefr_lookup as cefr
import semantic_branch as sb

MODEL_NAME = "roberta-large"


@pytest.fixture(scope="module")
def model_ready():
    """Load the real model once for this whole file, or skip every test in
    it if that's not possible here (no torch, no transformers, no network,
    no cached weights -- any of which can be true outside your own venv)."""
    try:
        sb._get_model_and_tokenizer(MODEL_NAME)
    except Exception as e:
        pytest.skip(f"roberta-large not available in this environment ({e!r}); "
                     f"these tests only run in your native venv with the model cached")


# ---- score_cefr_candidates() -----------------------------------------

def test_score_cefr_candidates_shape(model_ready):
    stem = "Did you ___ the price?"
    candidates, pool_size = sb.score_cefr_candidates(stem, "ask", "verb", "A1", top_n=50)
    # pool_size counts the CEFR-J words at this pos/level that get scored. It is
    # tokenizer-dependent (which words exist, how they split into subwords), so
    # it is not pinned to one exact number here -- the ask/brush regression
    # tests below pin it precisely for their specific queries.
    assert pool_size >= 50
    assert len(candidates) == 50
    for c in candidates:
        assert set(c.keys()) == {"word", "level", "pos", "lm_score"}
        assert isinstance(c["lm_score"], float)


def test_score_cefr_candidates_scores_multi_subword_words(model_ready):
    # Words that split into >1 subword (e.g. "frighten") used to be dropped
    # from the pool entirely; they are now scored via a k-mask pass and appear
    # like any other candidate, with a score on the same [0, 1] scale.
    tokenizer, _ = sb._get_model_and_tokenizer(MODEL_NAME)
    assert len(tokenizer.encode(" frighten", add_special_tokens=False)) > 1
    candidates, _ = sb.score_cefr_candidates("Did you ___ the price?", "ask",
                                             "verb", "A1", top_n=1000)
    scored = {c["word"]: c for c in candidates}
    assert "frighten" in scored
    assert 0.0 <= scored["frighten"]["lm_score"] <= 1.0


def test_score_cefr_candidates_never_includes_the_target_word(model_ready):
    stem = "Did you ___ the price?"
    candidates, _ = sb.score_cefr_candidates(stem, "ask", "verb", "A1", top_n=50)
    assert all(c["word"] != "ask" for c in candidates)


def test_score_cefr_candidates_every_candidate_passes_the_cefr_gate_it_claims(model_ready):
    stem = "Did you ___ the price?"
    candidates, _ = sb.score_cefr_candidates(stem, "ask", "verb", "A1", top_n=50)
    for c in candidates:
        ok, level, pos = cefr.matches(c["word"], target_pos="verb", target_level="A1")
        assert ok is True
        assert level == c["level"]
        assert pos == c["pos"]


def test_score_cefr_candidates_lm_scores_are_probabilities_sorted_descending(model_ready):
    stem = "Did you ___ the price?"
    candidates, _ = sb.score_cefr_candidates(stem, "ask", "verb", "A1", top_n=50)
    scores = [c["lm_score"] for c in candidates]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_score_cefr_candidates_respects_top_n(model_ready):
    stem = "Did you ___ the price?"
    candidates, pool_size = sb.score_cefr_candidates(stem, "ask", "verb", "A1", top_n=5)
    assert len(candidates) == 5
    assert pool_size > 5  # pool size is the full eligible pool, unaffected by top_n (exact count is tokenizer-dependent)


def test_score_cefr_candidates_disallowing_adjacent_levels_never_grows_the_pool(model_ready):
    stem = "Did you ___ the price?"
    _, pool_with_adjacent = sb.score_cefr_candidates(stem, "ask", "verb", "A1", top_n=50, allow_adjacent=True)
    _, pool_exact_only = sb.score_cefr_candidates(stem, "ask", "verb", "A1", top_n=50, allow_adjacent=False)
    assert pool_exact_only <= pool_with_adjacent


# ---- semantic_distractors() -- regression against real batch output --

def test_semantic_distractors_matches_known_good_run_for_ask(model_ready):
    # pinned against a real roberta-large run (pipeline/cache/batch_result.json,
    # captured 2026-09-01 after switching off the broken deberta-v3-large head)
    result = sb.semantic_distractors("Did you ___ the price?", "ask", target_pos="verb", target_level="A1")

    # 344 = single-token pool (340) + 4 multi-subword verbs now scored rather
    # than skipped (criticise/frighten/pollute/terrify); all 4 rank well below
    # the top 50, so the picks below are unchanged by that addition.
    assert result["n_cefr_candidate_pool"] == 344
    assert result["n_scored_by_model"] == 50
    assert result["n_with_vectors"] == 49

    tiered_stage = next(stage for stage in result["funnel"] if stage["stage"] == "tiered")
    assert tiered_stage["counts"] == {"discard": 4, "near_miss": 10, "thematic": 20, "unclassified": 12, "control": 3}

    picks = result["distractors"]
    assert picks["near_miss"]["word"] == "know"
    assert picks["near_miss"]["cosine"] == pytest.approx(0.6408244873930076, abs=1e-3)
    assert picks["thematic"]["word"] == "receive"
    assert picks["thematic"]["cosine"] == pytest.approx(0.5270602680589567, abs=1e-3)
    assert picks["control"]["word"] == "record"
    assert picks["control"]["cosine"] == pytest.approx(0.32510058770355293, abs=1e-3)


def test_semantic_distractors_matches_known_good_run_for_brush(model_ready):
    # pinned against a real roberta-large run (pipeline/cache/batch_result.json,
    # captured 2026-09-01 after switching off the broken deberta-v3-large head)
    result = sb.semantic_distractors("The leaves ___ her cheek.", "brush", target_pos="verb", target_level="A1")

    assert result["n_cefr_candidate_pool"] == 344  # incl. 4 multi-subword verbs (see ask test)
    assert result["n_scored_by_model"] == 50
    assert result["n_with_vectors"] == 48

    picks = result["distractors"]
    assert picks["near_miss"]["word"] == "hide"
    assert picks["near_miss"]["cosine"] == pytest.approx(0.4280337515013492, abs=1e-3)
    assert picks["thematic"]["word"] == "hit"
    assert picks["thematic"]["cosine"] == pytest.approx(0.3351288842668256, abs=1e-3)
    assert picks["control"]["word"] == "read"
    assert picks["control"]["cosine"] == pytest.approx(0.24283236230459979, abs=1e-3)


# ---- semantic_distractors() -- structural invariants (any word) -------

def test_semantic_distractors_never_picks_the_target_word(model_ready):
    result = sb.semantic_distractors("Did you ___ the price?", "ask", target_pos="verb", target_level="A1")
    picked_words = {p["word"] for p in result["distractors"].values()}
    assert "ask" not in picked_words


def test_semantic_distractors_picks_are_drawn_from_the_tier_they_are_named_for(model_ready):
    result = sb.semantic_distractors("Did you ___ the price?", "ask", target_pos="verb", target_level="A1")
    for tier_name, pick in result["distractors"].items():
        tier_words = {c["word"] for c in result["tiers_debug"][tier_name]}
        assert pick["word"] in tier_words


def test_semantic_distractors_tiers_partition_the_scored_pool_exactly(model_ready):
    result = sb.semantic_distractors("Did you ___ the price?", "ask", target_pos="verb", target_level="A1")
    all_tiered_words = [c["word"] for tier in result["tiers_debug"].values() for c in tier]
    assert len(all_tiered_words) == result["n_with_vectors"]
    assert len(all_tiered_words) == len(set(all_tiered_words))  # no word double-counted across tiers


def test_semantic_distractors_funnel_counts_are_monotonically_non_increasing(model_ready):
    result = sb.semantic_distractors("Did you ___ the price?", "ask", target_pos="verb", target_level="A1")
    plain_counts = [stage["count"] for stage in result["funnel"] if "count" in stage]
    assert plain_counts == sorted(plain_counts, reverse=True)


def test_semantic_distractors_raises_for_word_not_in_cefr_j_without_explicit_pos_level(model_ready):
    with pytest.raises(ValueError):
        sb.semantic_distractors("Did you ___ it?", "zzznotarealword")
