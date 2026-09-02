"""
Rule-based English -> katakana transliteration, used as a FALLBACK reading for
the gairaigo branch when JMdict has no katakana form for a word (about 47% of
CEFR-J words -- e.g. "ask", which Japanese doesn't actually borrow as a
loanword, so it has no JMdict entry).

This is a heuristic grapheme-to-kana mapper, not a pronunciation model: it
approximates how an English word would be written in katakana by Japanese
phonotactics -- consonant clusters get epenthetic vowels (ask -> ア-ス-ク,
asuku), final consonants get a trailing vowel (bus -> バ-ス), L and R both
map to the ラ row, a short "u" is rendered as "a" (run -> ラ-ン), and the common
digraphs (sh/ch/th/igh/ee/oo/...) are handled. It gets the common short CEFR
words right and is only ever compared against other katakana readings by the
mora-aware distance, so consistency matters more than perfect authenticity:
ask/task/mask/desk all transliterate the same way, so they read as neighbours.
"""
import re
from functools import lru_cache

# consonant -> the five kana for that row (a, i, u, e, o columns)
_CV = {
    "k": ["カ", "キ", "ク", "ケ", "コ"], "g": ["ガ", "ギ", "グ", "ゲ", "ゴ"],
    "s": ["サ", "シ", "ス", "セ", "ソ"], "z": ["ザ", "ジ", "ズ", "ゼ", "ゾ"],
    "t": ["タ", "チ", "ツ", "テ", "ト"], "d": ["ダ", "ディ", "ドゥ", "デ", "ド"],
    "n": ["ナ", "ニ", "ヌ", "ネ", "ノ"], "h": ["ハ", "ヒ", "フ", "ヘ", "ホ"],
    "b": ["バ", "ビ", "ブ", "ベ", "ボ"], "p": ["パ", "ピ", "プ", "ペ", "ポ"],
    "m": ["マ", "ミ", "ム", "メ", "モ"], "r": ["ラ", "リ", "ル", "レ", "ロ"],
    "l": ["ラ", "リ", "ル", "レ", "ロ"], "y": ["ヤ", "イ", "ユ", "イェ", "ヨ"],
    "w": ["ワ", "ウィ", "ウ", "ウェ", "ウォ"], "f": ["ファ", "フィ", "フ", "フェ", "フォ"],
    "v": ["ヴァ", "ヴィ", "ヴ", "ヴェ", "ヴォ"],
    "S": ["シャ", "シ", "シュ", "シェ", "ショ"],   # sh
    "C": ["チャ", "チ", "チュ", "チェ", "チョ"],   # ch / tch
    "J": ["ジャ", "ジ", "ジュ", "ジェ", "ジョ"],   # j
}
_STANDALONE = ["ア", "イ", "ウ", "エ", "オ"]
# vowel token -> (column index for the preceding consonant, trailing kana)
_VOWELS = {
    "a": (0, ""), "i": (1, ""), "u": (0, ""), "e": (3, ""), "o": (4, ""),
    "Y": (0, "イ"),   # igh  -> ai
    "X": (1, "ー"),   # ee/ea -> long i
    "Q": (2, "ー"),   # oo   -> long u
    "Z": (3, "イ"),   # ai/ay -> ei
    "O": (4, "ー"),   # oa/au/aw -> long o
    "W": (0, "ウ"),   # ou/ow -> au
    "P": (4, "イ"),   # oi/oy
}
_EPEN = {"t": 4, "d": 4}   # epenthetic vowel column: o for t/d, u for the rest


def _coda(cons):
    return _CV[cons][_EPEN.get(cons, 2)]


def _normalize(word):
    w = "".join(c for c in word.lower() if c.isalpha())
    # consonant digraphs -> single tokens
    w = w.replace("tch", "C").replace("sch", "S")
    w = (w.replace("sh", "S").replace("ch", "C").replace("ph", "f")
          .replace("wh", "w").replace("ck", "k").replace("ng", "N")
          .replace("th", "s").replace("qu", "kw").replace("x", "ks"))
    # soft c (before e/i/y) -> s, otherwise c -> k; j -> J
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k").replace("j", "J")
    # vowel digraphs
    for a, b in [("igh", "Y"), ("ee", "X"), ("ea", "X"), ("oo", "Q"),
                 ("ai", "Z"), ("ay", "Z"), ("oa", "O"), ("au", "O"),
                 ("aw", "O"), ("ou", "W"), ("ow", "W"), ("oi", "P"), ("oy", "P")]:
        w = w.replace(a, b)
    # silent final e (magic e): drop when it follows a consonant
    if len(w) >= 2 and w[-1] == "e" and w[-2] not in _VOWELS:
        w = w[:-1]
    # y as a vowel: trailing y -> i
    if w.endswith("y"):
        w = w[:-1] + "i"
    # collapse doubled consonants (gemination isn't spelled out here)
    w = re.sub(r"([bdfghklmnprstvwzSCJ])\1", r"\1", w)
    return w


@lru_cache(maxsize=16384)
def to_katakana(word):
    """A heuristic katakana rendering of an English word (see module docstring)."""
    toks = _normalize(word)
    out = []
    pending = None   # an onset consonant waiting for its vowel
    for i, ch in enumerate(toks):
        if ch in _VOWELS:
            col, trail = _VOWELS[ch]
            out.append(_CV[pending][col] if pending else _STANDALONE[col])
            pending = None
            if trail:
                out.append(trail)
        elif ch == "N":                      # ng
            if pending:
                out.append(_coda(pending)); pending = None
            out.append("ング")
        elif ch == "n":                       # onset before a vowel, else moraic ン
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if pending:
                out.append(_coda(pending)); pending = None
            if nxt in _VOWELS:
                pending = "n"
            else:
                out.append("ン")
        elif ch in _CV:
            if pending:
                out.append(_coda(pending))
            pending = ch
        # anything else (stray symbol) is ignored
    if pending:
        out.append(_coda(pending))
    return "".join(out)


if __name__ == "__main__":
    import sys
    for w in (sys.argv[1:] or ["ask", "run", "bus", "task", "mask", "desk",
                               "glass", "light", "jump", "help", "sun", "cut",
                               "think", "cheese", "phone", "school"]):
        print(f"  {w:10s} -> {to_katakana(w)}")
