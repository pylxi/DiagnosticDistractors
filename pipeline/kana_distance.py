"""
Mora-aware distance between two katakana readings, for the gairaigo/loanword
branch's near-neighbour search.

Plain romaji Levenshtein (what the branch used before) treats ラン ("ran") and
ダウン ("daun") as only two edits apart and ranks them alongside genuine
look-alikes like ターン ("taan"), even though no learner confuses "run" with
"fall". This metric instead works at the level a Japanese learner actually
hears -- the mora -- and weights edits by how confusable they really are:

- a mora that differs only in vowel *length* (ラ vs ラー) is nearly free -- long
  vs short vowel is the single most common katakana confusion;
- a mora with the SAME consonant row but a different vowel (ラ vs リ vs レ) is
  cheap -- these sit in one kana row and are routinely mixed up (and L/R is
  already collapsed here, since both are ラ行);
- a different consonant with the same vowel is moderate;
- an unrelated mora, or inserting/deleting a whole mora, is a full edit.

The result is a small non-negative float; callers threshold it (see
gairaigo.near_katakana_neighbors_among).
"""
from functools import lru_cache

import pykakasi

_kks = pykakasi.kakasi()

_SMALL = set("ァィゥェォャュョヮ")   # small kana that fuse onto the preceding mora
_VOWELS = "aeiou"

# edit costs, tuned so long/short-vowel and same-row-vowel swaps rank above
# genuine consonant differences (see kana_distance calibration).
_COST_LENGTH = 0.2      # differ only by a long-vowel mark
_COST_SAME_CONSONANT = 0.5
_COST_SAME_VOWEL = 0.8
_COST_UNRELATED = 1.0
_COST_INDEL = 1.0       # insert/delete a whole mora


def to_morae(kana):
    """Split a katakana string into mora units: a base kana plus any fused
    small kana, with a trailing long mark (ー) kept on its mora."""
    morae = []
    for ch in kana:
        if ch == "ー" and morae:
            morae[-1] += ch
        elif ch in _SMALL and morae:
            morae[-1] += ch
        else:
            morae.append(ch)
    return morae


@lru_cache(maxsize=4096)
def _mora_cv(mora):
    """(consonant, vowel, is_long) for one mora, via its Hepburn romaji.
    ラ -> ('r','a',False), ラー -> ('r','a',True), ン -> ('n','',False)."""
    is_long = "ー" in mora
    base = mora.replace("ー", "")
    romaji = "".join(x["hepburn"] for x in _kks.convert(base))
    v_idx = max((i for i, c in enumerate(romaji) if c in _VOWELS), default=-1)
    if v_idx == -1:
        return (romaji, "", is_long)          # e.g. ン -> ('n','')
    return (romaji[:v_idx], romaji[v_idx], is_long)


def _sub_cost(m1, m2):
    if m1 == m2:
        return 0.0
    c1, v1, l1 = _mora_cv(m1)
    c2, v2, l2 = _mora_cv(m2)
    if c1 == c2 and v1 == v2:
        return _COST_LENGTH                    # only length differs
    if c1 == c2:
        return _COST_SAME_CONSONANT            # same kana row, different vowel
    if v1 == v2:
        return _COST_SAME_VOWEL
    return _COST_UNRELATED


def mora_distance(kana_a, kana_b):
    """Weighted mora edit distance between two katakana strings."""
    a, b = to_morae(kana_a), to_morae(kana_b)
    n = len(b)
    prev = [j * _COST_INDEL for j in range(n + 1)]
    for i in range(1, len(a) + 1):
        cur = [i * _COST_INDEL]
        for j in range(1, n + 1):
            cur.append(min(
                prev[j - 1] + _sub_cost(a[i - 1], b[j - 1]),
                prev[j] + _COST_INDEL,
                cur[j - 1] + _COST_INDEL,
            ))
        prev = cur
    return prev[n]
