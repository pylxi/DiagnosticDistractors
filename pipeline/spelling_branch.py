"""
Spelling/phonetic distractor branch: runs all four surface-confusability
signals, pools their candidates, and ranks them by how confusable they really
are -- with cross-signal agreement as the top quality tier.

Signals (each contributes a confusability score in [0, 1]):
  - gairaigo_exact   -- identical katakana reading (bus/bath, light/right): 1.0
  - phonetic_swap    -- a single L/R, B/V, TH->S/Z, F/H swap lands on a real word
  - gairaigo_near    -- mora-aware katakana distance to a CEFR-J loanword
  - levenshtein      -- plain English-spelling edit distance over CEFR-J

Rather than a hard priority cascade that stops once a quota is met (which let a
fuzzy gairaigo_near match crowd out a strong levenshtein edit-1 neighbour like
run/ruin), every signal runs and the candidates are merged. A word's score is
its best signal plus a corroboration bonus for every other signal that also
found it, so a word that both *looks* and *sounds* like the target (found by
two signals -- e.g. run/rain, run/ring) outranks one that only lands near in a
single signal (run/phone). `funnel` records, per signal, how many raw
candidates it proposed and how many passed the CEFR-J gate; `n_corroborated`
counts words found by two or more signals.

FastText subword-embedding "visual neighbors" from the original design were
deliberately left out here -- see the note printed by --explain. The static
.vec file we have gives whole-word cosine similarity, which behaves like a
second semantic signal, not an orthographic one; it would blur exactly the
distinction this branch exists to keep clean.
"""
import sys

from pipeline import cefr_lookup as cefr
from pipeline import gairaigo
from pipeline import phonetic_swaps
from pipeline import levenshtein_search

FASTTEXT_NOTE = (
    "Note: skipping a FastText 'visual neighbor' fallback on purpose. "
    "pruned_cefr_j.vec stores plain whole-word vectors, not the subword "
    "n-gram model the original design assumed -- cosine similarity over it "
    "tracks meaning, not spelling, so it would just be a second, weaker "
    "semantic branch wearing a spelling-branch costume. Flagged for Lara "
    "rather than silently building something that doesn't do what it says."
)

# Per-signal confusability scores, in [0, 1]. A candidate accumulates one score
# per signal that found it; its final rank is the best score plus a
# corroboration bonus (_CORROBORATION_WEIGHT) for each additional signal.
_EXACT_SCORE = 1.0        # identical katakana reading -- indistinguishable by ear
_PHONETIC_SCORE = 0.85    # one confusable-sound swap lands on a real word
_CORROBORATION_WEIGHT = 0.4


def _near_score(mora_dist):
    """Mora-distance (0, ~1.0] -> confusability score. Closer reading = higher."""
    return max(0.0, 1.0 - mora_dist / 1.2)


def _lev_score(edit_dist):
    """English-spelling edit distance -> confusability score."""
    return {1: 0.8, 2: 0.55, 3: 0.35}.get(edit_dist, 0.2)


def spelling_distractors(word, target_pos=None, target_level=None, n=3, allow_adjacent=True):
    word = word.lower().strip()
    if target_pos is None or target_level is None:
        auto = cefr.entries(word)
        if not auto:
            raise ValueError(f"'{word}' is not in CEFR-J; pass target_pos/target_level explicitly")
        target_pos = target_pos or auto[0][0]
        target_level = target_level or auto[0][1]

    # The words that could pass the CEFR-J pos/level gate. Restricting the
    # gairaigo katakana search to this pool keeps it from spending its budget on
    # obscure JMdict loanwords that aren't in the answer space anyway.
    pool_words = {w for w, _, _ in cefr.candidate_pool(target_pos, target_level,
                                                       allow_adjacent=allow_adjacent)}

    # Gather raw (candidate, score) pairs from every signal.
    collisions = gairaigo.exact_katakana_collisions(word, allowed_heads=pool_words)
    raw = {
        "gairaigo_exact": [(w, _EXACT_SCORE) for ws in collisions.values() for w in ws],
        "phonetic_swap": [(it["word"], _PHONETIC_SCORE)
                          for it in phonetic_swaps.find_valid_swaps(
                              word, target_pos=target_pos, target_level=target_level,
                              allow_adjacent=allow_adjacent)],
        "gairaigo_near": [(it["word"], _near_score(it["distance"]))
                          for it in gairaigo.near_katakana_neighbors_among(
                              word, pool_words, top_n=30)],
        "levenshtein": [(it["word"], _lev_score(it["distance"]))
                        for it in levenshtein_search.neighbors(
                            word, target_pos=target_pos, target_level=target_level,
                            allow_adjacent=allow_adjacent, max_dist=3, top_n=30)],
    }

    # Gate each candidate once and merge the signals that found it.
    merged = {}  # word -> {"sources": {signal: score}, "level":, "pos":}
    funnel = []
    for source, items in raw.items():
        passed = 0
        for cand, score in items:
            cand = cand.lower().strip()
            if not cand or cand == word or " " in cand:
                # Skip the target itself and multi-word entries (e.g. "all
                # right"): a phrase isn't a plausible look-alike for a single
                # spelled target. (The semantic branch drops these via isalpha.)
                continue
            ok, level, pos = cefr.matches(cand, target_pos=target_pos,
                                           target_level=target_level,
                                           allow_adjacent=allow_adjacent)
            if not ok:
                continue
            passed += 1
            entry = merged.setdefault(cand, {"sources": {}, "level": level, "pos": pos})
            entry["sources"][source] = max(entry["sources"].get(source, 0.0), score)
        funnel.append({"source": source, "raw_candidates": len(items), "passed_cefr_gate": passed})

    # Rank by best signal + corroboration bonus for each additional signal.
    ranked = []
    for cand, info in merged.items():
        by_score = sorted(info["sources"], key=lambda s: info["sources"][s], reverse=True)
        scores = [info["sources"][s] for s in by_score]
        combined = scores[0] + _CORROBORATION_WEIGHT * sum(scores[1:])
        ranked.append({
            "word": cand, "level": info["level"], "pos": info["pos"],
            "source": by_score[0],          # primary (strongest) signal
            "sources": by_score,            # every signal that found it, best first
            "score": round(combined, 3),
        })
    ranked.sort(key=lambda c: (-c["score"], c["word"]))

    return {
        "target": word, "target_pos": target_pos, "target_level": target_level,
        "distractors": ranked[:n],
        "all_found": ranked,
        "sufficient": len(ranked) >= n,
        "funnel": funnel,
        "n_corroborated": sum(1 for c in ranked if len(c["sources"]) >= 2),
    }

if __name__ == "__main__":
    import json
    if "--explain" in sys.argv:
        print(FASTTEXT_NOTE)
        sys.exit(0)
    words = sys.argv[1:] or ["glass", "light", "bus", "collar", "quiet"]
    for w in words:
        out = spelling_distractors(w, n=8)
        print(f"\n=== {w} ({out['target_pos']}, {out['target_level']}) "
              f"sufficient={out['sufficient']} corroborated={out['n_corroborated']} ===")
        for f in out["funnel"]:
            print(f"  - {f['source']:<14} {f['raw_candidates']:>3} raw -> "
                  f"{f['passed_cefr_gate']:>2} passed CEFR gate")
        for d in out["all_found"]:
            marker = "*" if d in out["distractors"] else " "
            print(f"  {marker} {d['word']:<12} {d['score']:.2f}  via {' + '.join(d['sources'])}")
