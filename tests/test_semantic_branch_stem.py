"""
Tests for the non-ML parts of pipeline/semantic_branch.py: inflections(),
make_stem(), cosine(), and load_pruned_vectors(). None of these touch
torch/transformers (that import only happens lazily inside
score_cefr_candidates()/_get_model_and_tokenizer()), so this file runs
anywhere the real data files (data/fasttext/pruned_cefr_j.vec) are present,
with no venv, network, or model download required.
"""
import math

from pipeline import semantic_branch as sb


# ---- inflections() ----------------------------------------------------

def test_inflections_regular_verb():
    # confirmed real output -- note "askes" and "askest" are heuristic
    # over-generation (not real English), which is expected and documented:
    # inflections() is explicitly "good enough for A1, not a real
    # morphological analyzer". These tests pin its actual behavior so a
    # future change to the heuristic is a deliberate, visible diff.
    assert sb.inflections("ask") == {
        "ask", "asks", "askes", "asked", "asking", "asker", "askest",
    }


def test_inflections_covers_hardcoded_irregular_form():
    assert "bought" in sb.inflections("buy")


def test_inflections_covers_expanded_irregular_verbs():
    # Spot-check the broadened IRREGULAR_FORMS table beyond the original
    # pilot handful -- past and past-participle forms the suffix rules can't
    # derive should all be present.
    expected = {
        "go": {"went", "gone"}, "eat": {"ate", "eaten"}, "make": {"made"},
        "think": {"thought"}, "understand": {"understood"}, "pay": {"paid"},
        "give": {"gave", "given"}, "see": {"saw", "seen"},
    }
    for base, forms in expected.items():
        assert forms <= sb.inflections(base), base


def test_inflections_y_ending_after_consonant():
    forms = sb.inflections("try")
    assert {"tries", "tried", "trying"} <= forms


def test_inflections_always_includes_the_base_word():
    for word in ("ask", "buy", "try", "brush"):
        assert word in sb.inflections(word)


# ---- make_stem() --------------------------------------------------------

def test_make_stem_matches_base_form():
    stem, matched = sb.make_stem("Did you ask the price?", "ask")
    assert stem == "Did you ___ the price?"
    assert matched == "ask"


def test_make_stem_matches_inflected_form():
    stem, matched = sb.make_stem("The leaves brushed her cheek.", "brush")
    assert stem == "The leaves ___ her cheek."
    assert matched == "brushed"


def test_make_stem_matches_irregular_past_tense():
    stem, matched = sb.make_stem("I bought my car second-hand.", "buy")
    assert stem == "I ___ my car second-hand."
    assert matched == "bought"


def test_make_stem_preserves_original_casing_of_the_matched_token():
    stem, matched = sb.make_stem("Ask him now.", "ask")
    assert matched == "Ask"
    assert stem == "___ him now."


def test_make_stem_returns_none_none_when_word_absent():
    assert sb.make_stem("Nothing here at all.", "zzznotarealword") == (None, None)


def test_make_stem_does_not_match_a_substring_of_another_word():
    # "ask" must not spuriously match inside "asked" being searched for a
    # DIFFERENT target, or inside an unrelated longer word like "gasket"
    stem, matched = sb.make_stem("Check the gasket before you go.", "ask")
    assert (stem, matched) == (None, None)


# ---- cosine() -------------------------------------------------------------

def test_cosine_identical_vectors_is_one():
    assert sb.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_orthogonal_vectors_is_zero():
    assert sb.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_opposite_vectors_is_negative_one():
    assert math.isclose(sb.cosine([1.0, 0.0], [-1.0, 0.0]), -1.0)


def test_cosine_zero_vector_returns_zero_not_nan():
    # a naive dot/(|a||b|) would divide by zero here -- must be guarded
    assert sb.cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---- load_pruned_vectors() ------------------------------------------------

def test_load_pruned_vectors_returns_expected_shape():
    vecs = sb.load_pruned_vectors()
    assert len(vecs) > 0
    assert "ask" in vecs
    dim = len(vecs["ask"])
    assert dim > 0
    # every vector should share the same dimensionality
    sample = list(vecs.values())[:20]
    assert all(len(v) == dim for v in sample)
