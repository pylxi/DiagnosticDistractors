"""
Spelling/phonetic distractor branch: combines all four signals in priority
order (strongest, most authentic confusion first), filtering every candidate
through the CEFR-J POS/level check before it counts toward the quota.

Priority:
  1. gairaigo_exact   -- identical katakana reading (bus/bath, light/right)
  2. phonetic_swap    -- L/R, B/V, TH->S/Z, F/H substitution on the spelling
  3. gairaigo_near     -- mora-aware katakana distance to a CEFR-J loanword
  4. levenshtein       -- plain English-spelling edit distance over CEFR-J

Each signal only runs if the quota (`n`) isn't already met by higher-priority
signals -- `funnel` in the return value records, per signal, how many raw
candidates it proposed, how many survived the CEFR-J gate, and whether it
ran at all, so the whole cascade is visible rather than just the final pick.

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

def spelling_distractors(word, target_pos=None, target_level=None, n=3, allow_adjacent=True):
    word = word.lower().strip()
    if target_pos is None or target_level is None:
        auto = cefr.entries(word)
        if not auto:
            raise ValueError(f"'{word}' is not in CEFR-J; pass target_pos/target_level explicitly")
        target_pos = target_pos or auto[0][0]
        target_level = target_level or auto[0][1]

    seen = {word}
    results = []
    funnel = []

    # The words that could pass the CEFR-J pos/level gate below. Restricting the
    # gairaigo katakana search to this pool keeps it from spending its budget on
    # obscure JMdict loanwords that aren't in the answer space anyway.
    pool_words = {w for w, _, _ in cefr.candidate_pool(target_pos, target_level,
                                                       allow_adjacent=allow_adjacent)}

    def add(raw_items, source):
        raw_items = list(raw_items)
        before = len(results)
        for item in raw_items:
            cand = (item if isinstance(item, str) else item["word"]).lower().strip()
            if not cand or cand in seen:
                continue
            if " " in cand:
                # Skip multi-word CEFR-J entries (e.g. "all right"): a single
                # spelled word is the target, so a phrase isn't a plausible
                # look-alike distractor. (The semantic branch already excludes
                # these via its isalpha() check.)
                continue
            ok, level, pos = cefr.matches(cand, target_pos=target_pos,
                                           target_level=target_level,
                                           allow_adjacent=allow_adjacent)
            if ok:
                seen.add(cand)
                results.append({"word": cand, "source": source, "level": level, "pos": pos})
        funnel.append({
            "source": source, "ran": True,
            "raw_candidates": len(raw_items),
            "passed_cefr_gate": len(results) - before,
            "running_total": len(results),
        })

    collisions = gairaigo.exact_katakana_collisions(word, allowed_heads=pool_words)
    add([w for ws in collisions.values() for w in ws], "gairaigo_exact")

    if len(results) < n:
        add(phonetic_swaps.find_valid_swaps(word, target_pos=target_pos,
                                             target_level=target_level,
                                             allow_adjacent=allow_adjacent),
            "phonetic_swap")
    else:
        funnel.append({"source": "phonetic_swap", "ran": False, "reason": f"quota of {n} already met"})

    if len(results) < n:
        add(gairaigo.near_katakana_neighbors_among(word, pool_words, top_n=30,
                                                   exclude_words=seen),
            "gairaigo_near")
    else:
        funnel.append({"source": "gairaigo_near", "ran": False, "reason": f"quota of {n} already met"})

    if len(results) < n:
        add(levenshtein_search.neighbors(word, target_pos=target_pos, target_level=target_level,
                                          allow_adjacent=allow_adjacent, max_dist=3, top_n=30,
                                          exclude=seen),
            "levenshtein")
    else:
        funnel.append({"source": "levenshtein", "ran": False, "reason": f"quota of {n} already met"})

    return {
        "target": word, "target_pos": target_pos, "target_level": target_level,
        "distractors": results[:n],
        "all_found": results,
        "sufficient": len(results) >= n,
        "funnel": funnel,
    }

if __name__ == "__main__":
    import json
    if "--explain" in sys.argv:
        print(FASTTEXT_NOTE)
        sys.exit(0)
    words = sys.argv[1:] or ["glass", "light", "bus", "collar", "quiet"]
    for w in words:
        out = spelling_distractors(w, n=3)
        print(f"\n=== {w} ({out['target_pos']}, {out['target_level']}) sufficient={out['sufficient']} ===")
        for f in out["funnel"]:
            if f.get("ran", True) is False:
                print(f"  - {f['source']:<14} skipped ({f['reason']})")
            else:
                print(f"  - {f['source']:<14} {f['raw_candidates']:>3} raw -> "
                      f"{f['passed_cefr_gate']:>2} passed CEFR gate -> running total {f['running_total']}")
        for d in out["all_found"]:
            marker = "*" if d in out["distractors"] else " "
            print(f"  {marker} {d['word']:<12} via {d['source']:<14} ({d['pos']}, {d['level']})")
