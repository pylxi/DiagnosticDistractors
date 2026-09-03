"""
Spelling-challenge distractor branch -- distractors for a *spelling*-focused
cloze (a second item type, separate from the meaning/context cloze the semantic
branch feeds). Here the question is whether a Japanese learner can recognize the
correct English spelling, so:

- distractors do NOT have to match the target's part of speech or CEFR level
  (form confusion doesn't care -- "bush"/"blush" are great look-alikes for
  "brush" even though one is a noun and the other is above A1), and
- distractors do NOT have to be real words: a katakana-influenced spelling like
  "burashi" (how a learner might write "brush" from its loanword reading) is one
  of the most authentic errors there is.

Two kinds of distractor, ranked together by how confusable they are:

1. Real-word look-alikes -- common English words within a small edit distance of
   the target, drawn from a general word list (pipeline/cache/english_words.txt),
   with a boost for edits that are classic Japanese-learner sound confusions
   (L/R, B/V, TH<->S, F/H), e.g. brush -> blush.
2. A transliteration distractor -- the target's katakana reading rendered back to
   romaji, preferring the authentic JMdict loanword (gairaigo) over the
   rule-based fallback because it is closer to what learners actually write:
   career -> "kyaria" (キャリア), not the literal "kariiru". Usually a non-word.
"""
import os

import Levenshtein
import pykakasi

from pipeline import gairaigo
from pipeline import eng_to_katakana
from pipeline import phonetic_swaps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDLIST_PATH = os.path.join(ROOT, "pipeline", "cache", "english_words.txt")

_kks = pykakasi.kakasi()
_WORDS = None

# A small blocklist so a vulgar edit-neighbour (glass -> "ass") isn't shown as a
# distractor in a learner-facing tool. Not exhaustive -- extend as needed.
_BLOCKLIST = {
    "ass", "arse", "shit", "crap", "piss", "damn", "sex", "sexy",
    "dick", "cock", "tits", "boob", "boobs", "porn", "slut", "whore",
}

# per-signal confusability scores, combined like the spelling branch (best signal
# plus a corroboration bonus for each additional signal that found the candidate)
_S_PHONETIC = 0.9        # a classic L/R, B/V, TH<->S, F/H sound-swap look-alike
_S_TRANSLIT = 0.85       # katakana-influenced spelling -- a very authentic error
_S_EDIT = {1: 0.7, 2: 0.45}
_CORROBORATION_WEIGHT = 0.4


def load_english_words():
    """Common English words for judging real-word look-alikes (cached)."""
    global _WORDS
    if _WORDS is None:
        with open(WORDLIST_PATH, encoding="utf-8") as f:
            _WORDS = {line.strip() for line in f if line.strip()}
    return _WORDS


def _romaji(kana):
    return "".join(x["hepburn"] for x in _kks.convert(kana))


def _inflections(w):
    """The target's own simple inflected forms, excluded as distractors -- a
    plural/tense of the target isn't a spelling confusion, just the same word."""
    forms = {w, w + "s", w + "es", w + "ed", w + "ing", w + "er", w + "est", w + "d"}
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        forms |= {w[:-1] + "ies", w[:-1] + "ied"}
    if w.endswith("e"):
        forms |= {w[:-1] + "ing", w + "d"}
    return forms


def transliteration(word):
    """The target's katakana reading as a romaji spelling, gairaigo-preferred.
    Returns {form, katakana, gairaigo} or None if no reading can be produced."""
    word = word.lower().strip()
    forms = [f for f in gairaigo.katakana_forms_for(word, include_non_eng_source=True)
             if gairaigo._primary_head(f) == word]
    if forms:
        kata, is_gairaigo = forms[0]["katakana"], True
    else:
        kata, is_gairaigo = eng_to_katakana.to_katakana(word), False
    if not kata:
        return None
    return {"form": _romaji(kata), "katakana": kata, "gairaigo": is_gairaigo}


def spelling_challenge_distractors(word, n=8, max_dist=2):
    word = word.lower().strip()
    words = load_english_words()
    phonetic_hits = {c.lower() for c, _rule in phonetic_swaps.candidates(word)}
    excluded = _inflections(word)   # the target and its own inflected forms

    merged = {}  # candidate -> {"sources": {signal: score}, "is_real_word", "katakana"}

    # 1. real-word look-alikes by edit distance
    for cand in words:
        if cand in excluded or cand in _BLOCKLIST or len(cand) < 3 \
                or abs(len(cand) - len(word)) > max_dist:
            continue
        dist = Levenshtein.distance(word, cand)
        if not (0 < dist <= max_dist):
            continue
        entry = merged.setdefault(cand, {"sources": {}, "is_real_word": True, "katakana": None})
        entry["sources"]["edit%d" % dist] = _S_EDIT[dist]
        if cand in phonetic_hits:
            entry["sources"]["phonetic"] = _S_PHONETIC

    # phonetic swaps that are real words but landed outside the edit window
    for cand in phonetic_hits:
        if cand in excluded or len(cand) < 3 or cand not in words:
            continue
        entry = merged.setdefault(cand, {"sources": {}, "is_real_word": True, "katakana": None})
        entry["sources"].setdefault("phonetic", _S_PHONETIC)

    # 2. the transliteration distractor (usually a non-word)
    tr = transliteration(word)
    if tr and tr["form"] != word:
        entry = merged.setdefault(tr["form"], {"sources": {}, "is_real_word": tr["form"] in words,
                                               "katakana": tr["katakana"]})
        entry["sources"]["transliteration"] = _S_TRANSLIT

    ranked = []
    for cand, info in merged.items():
        by_score = sorted(info["sources"], key=lambda s: info["sources"][s], reverse=True)
        scores = [info["sources"][s] for s in by_score]
        combined = scores[0] + _CORROBORATION_WEIGHT * sum(scores[1:])
        ranked.append({
            "word": cand,
            "is_real_word": info["is_real_word"],
            "katakana": info["katakana"],
            "source": by_score[0],
            "sources": by_score,
            "score": round(combined, 3),
        })
    ranked.sort(key=lambda c: (-c["score"], c["word"]))

    return {
        "target": word,
        "distractors": ranked[:n],
        "all_found": ranked,
        "sufficient": len(ranked) >= n,
    }


if __name__ == "__main__":
    import sys
    for w in (sys.argv[1:] or ["brush", "career", "talk", "light", "ask", "glass"]):
        out = spelling_challenge_distractors(w, n=8)
        print(f"\n=== {w} ===")
        for d in out["distractors"]:
            tag = "" if d["is_real_word"] else "  [non-word]"
            kata = f" ({d['katakana']})" if d["katakana"] else ""
            print(f"  {d['word']:12s} {d['score']:.2f}  {' + '.join(d['sources'])}{kata}{tag}")
