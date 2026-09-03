"""
Tests for pipeline/spelling_challenge.py -- the spelling-focused distractor
generator (real-word look-alikes + a gairaigo-preferred transliteration
non-word). Needs the real data: pipeline/cache/english_words.txt (build with
`python3 -m pipeline.build_english_wordlist`) and the loanword index.
"""
from pipeline import spelling_challenge as sc


def _words(out):
    return [d["word"] for d in out["distractors"]]


def test_real_word_lookalikes_ignore_pos_and_level():
    # "bush" (noun) and "blush" (above A1) are prime look-alikes for "brush"
    # that the POS/level-gated branch drops; here they must appear.
    out = sc.spelling_challenge_distractors("brush", n=12)
    words = set(_words(out))
    assert "blush" in words and "bush" in words


def test_phonetic_swap_lookalike_ranks_top():
    # brush -> blush is both an L/R sound-swap and an edit-1 neighbour, so it
    # should outrank plain edit-1 look-alikes.
    out = sc.spelling_challenge_distractors("brush", n=12)
    assert out["distractors"][0]["word"] == "blush"
    assert "phonetic" in out["distractors"][0]["sources"]


def test_transliteration_prefers_gairaigo_over_rule_based():
    # career has a JMdict loanword (キャリア -> "kyaria"); the transliteration
    # distractor must use that, not the clumsy rule-based literal ("kariiru").
    out = sc.spelling_challenge_distractors("career", n=12)
    tr = [d for d in out["distractors"] if d["source"] == "transliteration"]
    assert tr and tr[0]["word"] == "kyaria"
    assert tr[0]["katakana"] == "キャリア"
    assert tr[0]["is_real_word"] is False


def test_transliteration_falls_back_to_rule_for_non_loanwords():
    # "ask" has no JMdict katakana, so the rule-based reading (アスク) is used.
    out = sc.spelling_challenge_distractors("ask", n=12)
    tr = [d for d in out["all_found"] if d["source"] == "transliteration"]
    assert tr and tr[0]["word"] == "asuku"


def test_target_inflections_are_excluded():
    out = sc.spelling_challenge_distractors("talk", n=20)
    words = set(_words(out))
    assert "talks" not in words and "talked" not in words


def test_vulgar_words_are_blocked():
    out = sc.spelling_challenge_distractors("glass", n=20)
    assert "ass" not in set(_words(out))


def test_real_word_lookalikes_share_the_first_letter():
    # An orthographic distractor must share the target's opening letter, so
    # brush keeps blush/brash/bush but drops crush/rush. (The transliteration
    # distractor is exempt -- career -> "kyaria".)
    out = sc.spelling_challenge_distractors("brush", n=20)
    for d in out["distractors"]:
        if d["source"] != "transliteration":
            assert d["word"][0] == "b", d["word"]
    assert "crush" not in set(_words(out)) and "rush" not in set(_words(out))
