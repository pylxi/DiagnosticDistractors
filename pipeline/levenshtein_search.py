"""
Direct English-spelling Levenshtein search over the CEFR-J vocabulary.
Complements gairaigo.py (loanword/katakana confusions) and phonetic_swaps.py
(rule-based sound substitutions) by catching plain "looks similar on the
page" neighbors those two miss -- e.g. quiet/quite, though/through, form/from.
"""
import Levenshtein

from pipeline import cefr_lookup as cefr

def neighbors(word, target_pos=None, target_level=None, allow_adjacent=True,
              max_dist=3, top_n=15, exclude=None):
    word = word.lower().strip()
    exclude = {word} | {w.lower() for w in (exclude or [])}
    scored = []
    # Restrict the Levenshtein sweep to the cached pos/level pool rather than
    # measuring every headword in the vocabulary (see cefr.candidate_pool).
    # The pool is already gated by matches(), so we only need the distance
    # filter here; the result set is identical to the old full-vocab scan.
    for cand, level, pos in cefr.candidate_pool(target_pos, target_level,
                                                allow_adjacent=allow_adjacent):
        if cand in exclude:
            continue
        dist = Levenshtein.distance(word, cand)
        if not (0 < dist <= max_dist):
            continue
        scored.append({"word": cand, "distance": dist, "level": level, "pos": pos})
    scored.sort(key=lambda r: (r["distance"], r["word"]))
    # dedupe (a word can appear with multiple POS entries)
    seen, out = set(), []
    for r in scored:
        if r["word"] in seen:
            continue
        seen.add(r["word"])
        out.append(r)
        if len(out) >= top_n:
            break
    return out

if __name__ == "__main__":
    tests = [("quiet", "adjective", "A2"), ("though", "conjunction", "B1"),
             ("glass", "noun", "A1"), ("form", "noun", "A2")]
    for word, pos, level in tests:
        print(f"\n=== {word} ({pos}, {level}) ===")
        for r in neighbors(word, target_pos=pos, target_level=level, max_dist=2):
            print(" ", r)
