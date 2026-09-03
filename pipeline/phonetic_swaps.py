"""
English-spelling phonetic-swap rules for sounds Japanese learners commonly
conflate: L/R, B/V, TH->S/Z, F/H. Unlike the katakana-collision branch
(gairaigo.py), these operate directly on the English spelling, so they catch
confusions that don't share a loanword form at all (light/right is one
letter apart in English; katakana doesn't even distinguish them the same
way "collar/color" does).

Each rule is (from_substring, to_substring). For every occurrence of
`from_substring` in the target word, we try substituting just that occurrence
and yield the result if it differs from the word. The caller (spelling_challenge)
decides which of these are real words worth keeping.
"""
RULES = [
    ("l", "r"), ("r", "l"),           # light/right
    ("b", "v"), ("v", "b"),           # best/vest
    ("th", "s"), ("s", "th"),         # think/sink
    ("th", "z"), ("z", "th"),
    ("f", "h"), ("h", "f"),           # fat/hat
    ("si", "shi"), ("shi", "si"),
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

if __name__ == "__main__":
    import sys
    for word in (sys.argv[1:] or ["light", "think", "fat", "glass", "board"]):
        print(f"  {word:8s} -> {list(candidates(word))}")
