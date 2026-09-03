"""
Build pipeline/cache/english_words.txt: a set of common English words, used by
the spelling-challenge branch to decide which look-alike edits of a target are
real words (as opposed to the intentional non-word transliteration distractors).

Source is the same wiki-news-300d-1M.vec the semantic branch already needs --
its rows are frequency-ordered, so the top N are the most common English words.
We keep the lowercase alphabetic ones, which drops punctuation tokens, numbers,
and capitalized proper nouns.

Run once, after downloading the source vectors:
    python3 -m pipeline.build_english_wordlist
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(ROOT, "data", "fasttext", "wiki-news-300d-1M.vec")
OUT_PATH = os.path.join(ROOT, "pipeline", "cache", "english_words.txt")
TOP_N = 50000   # most-frequent rows to scan; ~28k survive the lowercase-alpha filter


def main(src_path=SRC_PATH, out_path=OUT_PATH, top_n=TOP_N):
    words = []
    seen = set()
    with open(src_path, encoding="utf-8") as f:
        next(f)  # header "<n_words> <dim>"
        for i, line in enumerate(f):
            if i >= top_n:
                break
            w = line.split(" ", 1)[0]
            if w.isalpha() and w.islower() and w not in seen:
                seen.add(w)
                words.append(w)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("\n".join(words) + "\n")
    print(f"wrote {out_path}: {len(words)} common English words (top {top_n} rows of wiki-news)")


if __name__ == "__main__":
    main()
