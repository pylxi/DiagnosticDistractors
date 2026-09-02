"""
Gairaigo / phonetic-loanword distractor lookup, built on the cached JMdict
loanword index (see build_loanword_index.py).

Two use cases:
1. exact_katakana_collisions(word) -- words that map to the SAME katakana
   reading as the target (e.g. "bus" and "bath" both -> basu). These are the
   strongest, most authentic L2 confusions: a Japanese learner genuinely
   cannot distinguish them by ear/kana.
2. near_katakana_neighbors(word, max_dist) -- words whose katakana reading is
   a small romaji edit-distance away (e.g. glass/grass/class), for when exact
   collisions don't yield enough candidates.
"""
import json
import os
from collections import defaultdict

import Levenshtein

from pipeline import kana_distance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "pipeline", "cache", "loanwords.json")

_records = None
_by_head = None

def _load():
    global _records, _by_head
    if _records is not None:
        return
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        _records = json.load(f)
    _by_head = defaultdict(list)
    for r in _records:
        for h in r["heads"]:
            _by_head[h].append(r)

def _primary_head(record):
    """The primary (first-listed) English gloss of a JMdict record, normalized
    the same way heads are (parenthetical stripped, lowercased). This is the
    word the katakana is really the transliteration of."""
    g = record["glosses"][0]
    i = g.find("(")
    return (g[:i] if i != -1 else g).strip().lower()

def katakana_forms_for(word, include_non_eng_source=False):
    """All JMdict loanword records whose English gloss matches `word`."""
    _load()
    word = word.lower().strip()
    hits = _by_head.get(word, [])
    if not include_non_eng_source:
        hits = [h for h in hits if not h["non_eng_source"]]
    return hits

def exact_katakana_collisions(word, exclude_words=None, allowed_heads=None):
    """Other English words that share the SAME katakana reading as `word`.

    If `allowed_heads` is given, only those English words are considered as
    candidates -- pass the CEFR-J pool so collisions are drawn from the actual
    answer space rather than all of JMdict's loanword glosses.
    """
    _load()
    exclude_words = exclude_words or set()
    exclude_words = {word.lower()} | {w.lower() for w in exclude_words}
    forms = katakana_forms_for(word, include_non_eng_source=True)
    collisions = defaultdict(set)  # katakana -> set of other english heads
    for form in forms:
        katakana = form["katakana"]
        for r in _records:
            if r["katakana"] == katakana:
                for h in r["heads"]:
                    if h in exclude_words:
                        continue
                    if allowed_heads is not None and h not in allowed_heads:
                        continue
                    collisions[katakana].add(h)
    return collisions

def near_katakana_neighbors(word, max_dist=2, top_n=15, exclude_words=None):
    """English words whose katakana reading is within `max_dist` romaji edits.

    Note: this matches on JMdict *records*, so it can return any English gloss
    attached to a near-reading record -- including a word whose own katakana
    would be written differently. For distractor generation over a known
    candidate set, prefer near_katakana_neighbors_among(), which matches each
    candidate on the reading used to write *that* word.
    """
    _load()
    exclude_words = exclude_words or set()
    exclude_words = {word.lower()} | {w.lower() for w in exclude_words}
    forms = katakana_forms_for(word, include_non_eng_source=False)
    if not forms:
        return []
    target_romajis = {f["romaji"] for f in forms}
    scored = []
    for r in _records:
        dist = min(Levenshtein.distance(tr, r["romaji"]) for tr in target_romajis)
        if 0 < dist <= max_dist:
            for h in r["heads"]:
                if h not in exclude_words:
                    scored.append((dist, h, r["katakana"], r["romaji"]))
    scored.sort(key=lambda x: x[0])
    seen = set()
    out = []
    for dist, head, katakana, romaji in scored:
        if head in seen:
            continue
        seen.add(head)
        out.append({"word": head, "distance": dist, "katakana": katakana, "romaji": romaji})
        if len(out) >= top_n:
            break
    return out

def near_katakana_neighbors_among(word, candidate_words, max_dist=1.0, top_n=15, exclude_words=None):
    """Among `candidate_words`, the ones whose OWN katakana loanword reading is
    within `max_dist` of `word`'s reading under the mora-aware distance
    (see kana_distance.mora_distance) -- so long/short-vowel and same-kana-row
    vowel swaps rank above genuine consonant differences, matching how a
    Japanese learner actually mishears the word.

    Two guards keep this precise:
    - each candidate is matched on the reading for which it is the PRIMARY gloss
      (the katakana is actually how *that* word is written) -- so "draw" (ドロー)
      is not surfaced as a neighbor of run (ラン) just because タイ ("tie") lists
      "draw" as a secondary gloss;
    - exact reading matches (distance 0) are left to exact_katakana_collisions.

    This is the precise search for building distractors over a fixed candidate
    set (e.g. the CEFR-J pool).
    """
    _load()
    exclude_words = {word.lower()} | {w.lower() for w in (exclude_words or set())}
    target_kata = {f["katakana"] for f in katakana_forms_for(word, include_non_eng_source=False)}
    if not target_kata:
        return []
    scored = []
    for cand in candidate_words:
        c = cand.lower().strip()
        if c in exclude_words:
            continue
        best = None  # (dist, katakana, romaji) for this candidate's closest form
        for f in katakana_forms_for(c, include_non_eng_source=True):
            if _primary_head(f) != c:
                continue
            dist = min(kana_distance.mora_distance(tk, f["katakana"]) for tk in target_kata)
            if 0 < dist <= max_dist and (best is None or dist < best[0]):
                best = (round(dist, 3), f["katakana"], f["romaji"])
        if best:
            scored.append({"word": c, "distance": best[0], "katakana": best[1], "romaji": best[2]})
    scored.sort(key=lambda r: (r["distance"], r["word"]))
    return scored[:top_n]

if __name__ == "__main__":
    import sys
    tests = sys.argv[1:] or ["glass", "bus", "light", "collar", "sign"]
    for w in tests:
        print(f"\n=== {w} ===")
        forms = katakana_forms_for(w, include_non_eng_source=True)
        print(f"  katakana forms: {[(f['katakana'], f['romaji'], 'NON-ENG' if f['non_eng_source'] else 'eng') for f in forms]}")
        collisions = exact_katakana_collisions(w)
        for kana, words in collisions.items():
            if words:
                print(f"  EXACT collision on {kana}: {sorted(words)}")
        neighbors = near_katakana_neighbors(w, max_dist=1)
        print(f"  near neighbors (edit<=1): {[(n['word'], n['distance']) for n in neighbors[:8]]}")
