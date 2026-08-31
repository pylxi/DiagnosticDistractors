"""
Direct English-spelling Levenshtein search over the CEFR-J vocabulary.
Complements gairaigo.py (loanword/katakana confusions) and phonetic_swaps.py
(rule-based sound substitutions) by catching plain "looks similar on the
page" neighbors those two miss -- e.g. quiet/quite, though/through, form/from.
"""
import os
import sys

import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cefr_lookup as cefr

def neighbors(word, target_pos=None, target_level=None, allow_adjacent=True,
              max_dist=3, top_n=15, exclude=None):
    word = word.lower().strip()
    exclude = {word} | {w.lower() for w in (exclude or [])}
    scored = []
    for cand in cefr.all_words():
        if cand in exclude:
            continue
        dist = Levenshtein.distance(word, cand)
        if not (0 < dist <= max_dist):
            continue
        ok, level, pos = cefr.matches(cand, target_pos=target_pos,
                                       target_level=target_level,
                                       allow_adjacent=allow_adjacent)
        if ok:
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
