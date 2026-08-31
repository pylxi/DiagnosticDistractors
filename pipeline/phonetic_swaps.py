"""
English-spelling phonetic-swap rules for sounds Japanese learners commonly
conflate: L/R, B/V, TH->S/Z, F/H. Unlike the katakana-collision branch
(gairaigo.py), these operate directly on the English spelling, so they catch
confusions that don't share a loanword form at all (light/right is one
letter apart in English; katakana doesn't even distinguish them the same
way "collar/color" does).

Each rule is (from_substring, to_substring). For every occurrence of
`from_substring` in the target word, we try substituting just that
occurrence and keep the result only if it's a *different*, real CEFR-J word.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cefr_lookup as cefr

RULES = [
    ("l", "r"), ("r", "l"),           # light/right
    ("b", "v"), ("v", "b"),           # best/vest
    ("th", "s"), ("s", "th"),         # think/sink
    ("th", "z"), ("z", "th"),
    ("f", "h"), ("h", "f"),           # fat/hat
    ("si", "shi"), ("shi", "si"),
    ("v", "b"),
]

def _substitute_at(word, start, from_str, to_str):
    return word[:start] + to_str + word[start + len(from_str):]

def candidates(word):
    """
    Yield (candidate_word, rule) for every single-occurrence phonetic swap
    of `word`, without any CEFR/POS filtering yet.
    """
    word = word.lower()
    seen = set()
    for from_str, to_str in RULES:
        start = 0
        while True:
            idx = word.find(from_str, start)
            if idx == -1:
                break
            cand = _substitute_at(word, idx, from_str, to_str)
            start = idx + 1
            if cand != word and cand not in seen:
                seen.add(cand)
                yield cand, f"{from_str}->{to_str}"

def find_valid_swaps(word, target_pos=None, target_level=None, allow_adjacent=True):
    """
    Phonetic-swap candidates that are also real CEFR-J words matching the
    target's POS and CEFR level (with the same ±1 fallback as cefr_lookup).
    """
    out = []
    for cand, rule in candidates(word):
        ok, level, pos = cefr.matches(cand, target_pos=target_pos,
                                       target_level=target_level,
                                       allow_adjacent=allow_adjacent)
        if ok:
            out.append({"word": cand, "rule": rule, "level": level, "pos": pos})
    return out

if __name__ == "__main__":
    tests = [
        ("light", "adjective", "A1"),
        ("right", "adjective", "A1"),
        ("think", "verb", "A1"),
        ("fat", "adjective", "A1"),
        ("class", "noun", "A1"),
        ("glass", "noun", "A1"),
        ("board", "noun", "A2"),
    ]
    for word, pos, level in tests:
        raw = list(candidates(word))
        valid = find_valid_swaps(word, target_pos=pos, target_level=level)
        print(f"\n=== {word} ({pos}, {level}) ===")
        print(f"  raw candidates: {raw}")
        print(f"  valid (CEFR-J + POS + level): {valid}")
