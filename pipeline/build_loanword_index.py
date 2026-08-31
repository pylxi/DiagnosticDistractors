"""
Build a cached index of English-sourced katakana loanwords from JMdict, for the
gairaigo/spelling-confusion branch. Run once (or whenever JMdict updates):
    python3 pipeline/build_loanword_index.py

Output: pipeline/cache/loanwords.json — a list of records:
    {katakana, romaji, glosses, non_eng_source, wasei, ent_seq}

Design notes (see project memory for the full writeup):
- JMdict's convention: a <gloss> or <lsource> with no xml:lang attribute defaults
  to English. A <gloss xml:lang="dut">/"fre"/"ger"/... is a translation into that
  language, NOT an English gloss, and must be excluded from reverse lookup or
  you get false matches (garasu's gloss list literally contains Russian and
  Hungarian translations mixed in with the English ones).
- An explicit <lsource xml:lang="XXX"> where XXX != eng means the loanword's
  *origin* is that language, not English (e.g. gArasu/glass is from Dutch
  "glas", not English "glass" -- they just happen to look alike). We keep
  these but flag them (non_eng_source=True) so callers can decide whether to
  treat them as authentic English gairaigo or just a phonetic neighbor.
- <misc>word usually written using kana alone</misc> etc. are orthography
  notes, not loanword-origin signals -- ignored here.
"""
import gzip
import json
import os
import time
import unicodedata

from lxml import etree
import pykakasi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JMDICT_PATH = os.path.join(ROOT, "data", "edict", "JMdict_e.xml.gz")
OUT_PATH = os.path.join(ROOT, "pipeline", "cache", "loanwords.json")
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

KATAKANA_RANGES = [(0x30A0, 0x30FF), (0x31F0, 0x31FF)]  # katakana + phonetic extensions

# JMdict <misc> values that mark a sense as unsuitable for a general-vocab
# distractor pool (slang/rare homographs, proper nouns, idioms, etc.) -- e.g.
# koppu's "cop / police officer" sense is tagged both "slang" and "rare term",
# distinct from its common "cup / tumbler" sense, and must not be pooled with it.
LOW_QUALITY_MISC = {
    "rare term", "slang", "obsolete term", "dated term", "archaism",
    "vulgar", "derogatory term", "organization name", "proverb",
    "idiomatic expression", "given name or surname", "place name",
    "unclassified name",
}

def is_katakana(s):
    if not s:
        return False
    for ch in s:
        cp = ord(ch)
        if not any(lo <= cp <= hi for lo, hi in KATAKANA_RANGES):
            return False
    return True

def strip_paren(gloss):
    """'glass (drinking vessel)' -> 'glass'"""
    idx = gloss.find("(")
    return (gloss[:idx] if idx != -1 else gloss).strip().lower()

def main():
    kks = pykakasi.kakasi()
    t0 = time.time()
    records = []
    n_entries = 0
    n_katakana_entries = 0

    with gzip.open(JMDICT_PATH, "rb") as f:
        context = etree.iterparse(f, events=("end",), tag="entry",
                                   load_dtd=True, resolve_entities=True, huge_tree=True)
        for _, entry in context:
            n_entries += 1
            ent_seq = entry.findtext("ent_seq")
            readings = [r.text for r in entry.findall("r_ele/reb")]
            katakana_readings = [r for r in readings if is_katakana(r)]
            if not katakana_readings:
                entry.clear()
                continue
            n_katakana_entries += 1

            eng_glosses = []
            non_eng_source = False
            wasei = False
            for sense in entry.findall("sense"):
                sense_misc = {m.text for m in sense.findall("misc") if m.text}
                if sense_misc & LOW_QUALITY_MISC:
                    # skip this sense's glosses entirely (e.g. slang/rare
                    # homographs like koppu="cop", unrelated to the common
                    # koppu="cup/tumbler" sense) -- but still check its
                    # lsource/wasei flags below since those describe the
                    # katakana form as a whole, not just this sense's gloss.
                    for ls in sense.findall("lsource"):
                        lang = ls.get(XML_LANG, "eng")
                        if lang != "eng":
                            non_eng_source = True
                        if ls.get("ls_wasei") == "y":
                            wasei = True
                    continue
                for ls in sense.findall("lsource"):
                    lang = ls.get(XML_LANG, "eng")
                    if lang != "eng":
                        non_eng_source = True
                    if ls.get("ls_wasei") == "y":
                        wasei = True
                for gloss in sense.findall("gloss"):
                    lang = gloss.get(XML_LANG, "eng")
                    if lang == "eng" and gloss.text:
                        eng_glosses.append(gloss.text)

            entry.clear()
            if not eng_glosses:
                continue

            katakana = katakana_readings[0]
            romaji = "".join(x["hepburn"] for x in kks.convert(katakana))
            records.append({
                "ent_seq": ent_seq,
                "katakana": katakana,
                "romaji": romaji,
                "glosses": eng_glosses,
                "heads": sorted({strip_paren(g) for g in eng_glosses}),
                "non_eng_source": non_eng_source,
                "wasei": wasei,
            })

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        json.dump(records, out, ensure_ascii=False)

    print(f"scanned {n_entries} entries, {n_katakana_entries} had katakana readings")
    print(f"indexed {len(records)} loanword records with English glosses")
    print(f"  non_eng_source flagged: {sum(r['non_eng_source'] for r in records)}")
    print(f"  wasei flagged: {sum(r['wasei'] for r in records)}")
    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
    print(f"done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
