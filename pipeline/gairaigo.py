"""
Gairaigo / loanword lookup over the cached JMdict loanword index (see
build_loanword_index.py). Used by the spelling-challenge branch to find a word's
authentic katakana reading (its gairaigo form) when it has one -- e.g.
career -> キャリア -- which makes a better transliteration distractor than the
rule-based fallback.
"""
import json
import os
from collections import defaultdict

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
    word the katakana is really the transliteration of -- so a record is only
    treated as *word*'s own reading when word is its primary gloss (career's
    キャリア, not "chat" which merely appears as a secondary gloss on talk's トーク)."""
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

if __name__ == "__main__":
    import sys
    for w in (sys.argv[1:] or ["glass", "bus", "career", "brush", "talk"]):
        forms = [f for f in katakana_forms_for(w, include_non_eng_source=True)
                 if _primary_head(f) == w]
        print(f"  {w:8s} -> {[(f['katakana'], f['romaji']) for f in forms]}")
