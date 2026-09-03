"""
Step 1: verify every data source loads and looks sane.
Run from the DiagnosticDistractors folder: python3 pipeline/step1_load_data.py
"""
import csv
import gzip
import os
import sys
import time
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_cefr_j():
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    lookup = {}  # word -> list of (pos, level)
    counts = {}
    for lvl in levels:
        path = os.path.join(ROOT, "data", "CEFR-J", lvl, "words.csv")
        n = 0
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row["headword"].strip().lower()
                pos = row["pos"].strip().lower()
                # headwords like "a.m./A.M./am/AM" pack variants; split on '/'
                for variant in word.split("/"):
                    variant = variant.strip()
                    if not variant:
                        continue
                    lookup.setdefault(variant, []).append((pos, lvl))
                n += 1
        counts[lvl] = n
    return lookup, counts

def load_fasttext_vec(path, max_words=None):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        n_words, dim = int(header[0]), int(header[1])
        for i, line in enumerate(f):
            if max_words and i >= max_words:
                break
            parts = line.rstrip().split(" ")
            words.append(parts[0])
    return n_words, dim, words

def load_wordnet_ewn_sample():
    import yaml
    path = os.path.join(ROOT, "data", "wordnet_ewn", "entries-g.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    key = next(iter(data))
    return key, data[key]

def load_jmdict_sample(max_entries=5):
    from lxml import etree
    path = os.path.join(ROOT, "data", "edict", "JMdict_e.xml.gz")
    entries = []
    with gzip.open(path, "rb") as f:
        # JMdict genuinely uses its internal DTD entities (e.g. <misc>&rare;</misc>,
        # POS tags), so load_dtd/resolve_entities must stay on. Harden instead by
        # forbidding network entity resolution and keeping libxml2's built-in
        # resource limits (no huge_tree) against entity-expansion blow-ups.
        context = etree.iterparse(f, events=("end",), tag="entry",
                                   load_dtd=True, resolve_entities=True, no_network=True)
        for _, entry in context:
            kanji = [k.text for k in entry.findall("k_ele/keb")]
            readings = [r.text for r in entry.findall("r_ele/reb")]
            glosses = [g.text for g in entry.findall("sense/gloss")]
            pos_tags = [p.text for p in entry.findall("sense/pos")]
            entries.append((kanji, readings, glosses, pos_tags))
            entry.clear()
            if len(entries) >= max_entries:
                break
    return entries

def main():
    print("=== CEFR-J word lists ===")
    t0 = time.time()
    lookup, counts = load_cefr_j()
    for lvl, n in counts.items():
        print(f"  {lvl}: {n} rows")
    print(f"  unique headword variants: {len(lookup)}")
    print(f"  sample 'run' -> {lookup.get('run')}")
    print(f"  loaded in {time.time()-t0:.2f}s")

    print("\n=== FastText: pruned_cefr_j.vec ===")
    t0 = time.time()
    n_words, dim, words = load_fasttext_vec(os.path.join(ROOT, "data", "fasttext", "pruned_cefr_j.vec"))
    print(f"  header says {n_words} words x {dim} dims; read {len(words)} tokens")
    print(f"  sample tokens: {words[:8]}")
    print(f"  loaded in {time.time()-t0:.2f}s")

    print("\n=== FastText: wiki-news-300d-1M.vec (header + first 2000 tokens only) ===")
    t0 = time.time()
    n_words, dim, words = load_fasttext_vec(os.path.join(ROOT, "data", "fasttext", "wiki-news-300d-1M.vec"), max_words=2000)
    print(f"  header says {n_words} words x {dim} dims; sampled {len(words)} tokens")
    print(f"  sample tokens: {words[:8]}")
    print(f"  partial read in {time.time()-t0:.2f}s (full file is 2.2GB, not loaded here)")

    print("\n=== WordNet-EWN (entries-g.yaml sample) ===")
    t0 = time.time()
    key, entry = load_wordnet_ewn_sample()
    print(f"  sample key: {key}")
    print(f"  sample entry: {entry}")
    print(f"  loaded in {time.time()-t0:.2f}s")

    print("\n=== JMdict_e.xml.gz (first 5 entries) ===")
    t0 = time.time()
    entries = load_jmdict_sample()
    for kanji, readings, glosses, pos_tags in entries:
        print(f"  kanji={kanji} readings={readings} glosses={glosses[:3]} pos={pos_tags[:3]}")
    print(f"  loaded in {time.time()-t0:.2f}s")

    print("\nAll sources loaded OK.")

if __name__ == "__main__":
    main()
