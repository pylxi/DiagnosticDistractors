"""
Build data/fasttext/pruned_cefr_j.vec: the subset of a full fastText word-vector
file restricted to the CEFR-J vocabulary, used by the semantic branch for its
FastText cosine-tiering stage.

Run once, after downloading the source vectors (see README "Data sources"):
    python3 -m pipeline.build_pruned_vectors

Reads:  data/fasttext/wiki-news-300d-1M.vec   (plain whole-word vectors)
Writes: data/fasttext/pruned_cefr_j.vec        (word2vec text format --
        a "<n_words> <dim>" header line, then "<word> <v1> ... <vdim>" per line)

Keeping only CEFR-J headwords shrinks the ~1M-word source to the few thousand
words the branch can ever score, so it loads in a moment instead of minutes.
The match is exact and case-sensitive; CEFR-J headwords are lowercased, so this
naturally drops capitalized proper-noun vectors.
"""
import os

from pipeline import cefr_lookup as cefr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(ROOT, "data", "fasttext", "wiki-news-300d-1M.vec")
OUT_PATH = os.path.join(ROOT, "data", "fasttext", "pruned_cefr_j.vec")


def main(src_path=SRC_PATH, out_path=OUT_PATH):
    words = set(cefr.all_words())
    kept = []
    with open(src_path, encoding="utf-8") as f:
        dim = int(next(f).split()[1])
        for line in f:
            if line.split(" ", 1)[0] in words:
                kept.append(line.rstrip("\n"))
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"{len(kept)} {dim}\n")
        for line in kept:
            out.write(line + "\n")
    pct = 100 * len(kept) // len(words) if words else 0
    print(f"wrote {out_path}: {len(kept)} CEFR-J words x {dim} dims "
          f"({len(kept)}/{len(words)} = {pct}% of CEFR-J vocabulary covered)")


if __name__ == "__main__":
    main()
