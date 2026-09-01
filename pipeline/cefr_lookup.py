"""
Minimal CEFR-J + POS lookup, shared by the spelling and (later) semantic branches.
Exact-level match first, ±1 adjacent-level fallback when the caller asks for it.
"""
import csv
import os
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_INDEX = {lvl: i for i, lvl in enumerate(LEVELS)}

_lookup = None  # word -> list of (pos, level)

def _load():
    global _lookup
    if _lookup is not None:
        return
    _lookup = {}
    for lvl in LEVELS:
        path = os.path.join(ROOT, "data", "CEFR-J", lvl, "words.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pos = row["pos"].strip().lower()
                for variant in row["headword"].strip().lower().split("/"):
                    variant = variant.strip()
                    if variant:
                        _lookup.setdefault(variant, []).append((pos, lvl))

def entries(word):
    """All (pos, level) pairs for a word, or [] if it's not in CEFR-J."""
    _load()
    return _lookup.get(word.lower().strip(), [])

def levels_for(word):
    _load()
    return sorted({lvl for _, lvl in entries(word)}, key=LEVEL_INDEX.get)

def pos_for(word):
    _load()
    return sorted({pos for pos, _ in entries(word)})

def is_in_cefr_j(word):
    return bool(entries(word))

def all_words():
    """Every headword in the CEFR-J lists (for brute-force neighbor search)."""
    _load()
    return _lookup.keys()

def adjacent_levels(level, radius=1):
    i = LEVEL_INDEX[level]
    return [LEVELS[j] for j in range(max(0, i - radius), min(len(LEVELS), i + radius + 1))]

@lru_cache(maxsize=None)
def candidate_pool(target_pos=None, target_level=None, allow_adjacent=True):
    """
    The CEFR-J headwords eligible under `matches()` for this (pos, level)
    query, as a tuple of (word, matched_level, matched_pos) triples.

    This is exactly the accepted subset the branches used to rebuild by
    walking `all_words()` and calling `matches()` on every headword on every
    request. Here it's computed once per distinct
    (target_pos, target_level, allow_adjacent) and cached, so the semantic
    branch's per-request pool scan and the Levenshtein branch's per-request
    distance sweep both iterate a small pre-filtered pool instead of the full
    vocabulary -- which matters now that the web app runs the whole pipeline
    once per user click. Iteration order matches `_lookup` insertion order
    (i.e. `all_words()`), so callers that sort with ties resolve them
    identically to the pre-cache behavior.
    """
    _load()
    pool = []
    for word in _lookup:
        ok, level, pos = matches(word, target_pos=target_pos,
                                 target_level=target_level,
                                 allow_adjacent=allow_adjacent)
        if ok:
            pool.append((word, level, pos))
    return tuple(pool)

def matches(word, target_pos=None, target_level=None, allow_adjacent=True):
    """
    True if `word` is a real CEFR-J entry that satisfies the target POS
    (if given) at the target's exact level, or an adjacent level when
    allow_adjacent=True and no exact-level match exists.
    Returns (ok: bool, matched_level: str|None, matched_pos: str|None).
    """
    _load()
    cands = entries(word)
    if not cands:
        return False, None, None
    if target_pos:
        cands = [(p, l) for p, l in cands if p == target_pos.lower()]
        if not cands:
            return False, None, None
    if target_level is None:
        p, l = cands[0]
        return True, l, p
    exact = [(p, l) for p, l in cands if l == target_level]
    if exact:
        p, l = exact[0]
        return True, l, p
    if allow_adjacent:
        allowed = set(adjacent_levels(target_level, radius=1))
        near = [(p, l) for p, l in cands if l in allowed]
        if near:
            p, l = near[0]
            return True, l, p
    return False, None, None

if __name__ == "__main__":
    for w in ["run", "light", "right", "sink", "vest", "hat", "glass"]:
        print(w, "->", entries(w))
    print("\nmatches('right', target_pos='adjective', target_level='A1'):",
          matches("right", target_pos="adjective", target_level="A1"))
    print("matches('sink', target_pos='verb', target_level='A1'):",
          matches("sink", target_pos="verb", target_level="A1"))
