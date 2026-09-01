# Data assumptions audit

What the pipeline assumes about its input data, where those assumptions could
bite, and the provenance of each source. Numbers below are from the CEFR-J
wordlist as loaded on 2026-09-01 (`pipeline/cefr_lookup.py`); regenerate them
if the wordlist is updated.

## CEFR-J wordlist (`data/CEFR-J/{A1..C2}/words.csv`)

**Provenance.** The CEFR-J wordlist from the Tono Lab, Tokyo University of
Foreign Studies — a CEFR-aligned English wordlist built for Japanese learners.
It is the candidate universe for *both* branches; nothing outside it can ever
become a distractor. External and gitignored (see README "Data sources").

**Shape as loaded.** 8,755 unique headwords. One CSV per level, so a word's
level is encoded by *which file* it appears in, and its POS comes from the
`pos` column. `cefr_lookup._load()` lowercases both, and splits headwords on
`/` into separate variants.

### Level values

Six levels, treated as a **linear ordinal scale** in this fixed order:

```
A1 < A2 < B1 < B2 < C1 < C2
```

Counts: A1 1200, A2 1443, B1 2487, B2 2859, C1 1095, C2 893.

- **Assumption: adjacency is symmetric and ±1.** `matches(allow_adjacent=True)`
  treats one level up or down as an acceptable substitute for the target level.
  This bakes in that the psychometric "distance" between, say, A2 and B1 is
  comparable to that between B1 and B2. The CEFR bands are not evenly spaced in
  learner effort, so ±1 is a pragmatic convenience, not a measured equivalence.
- **Assumption: a word has one level per (headword, POS) row.** A word that
  legitimately spans levels in different senses is represented only by whatever
  rows exist in the CSVs; there is no sense disambiguation.

### POS values

15 distinct POS tags actually present (not a generic noun/verb/adj/adv set):

```
noun 4968, adjective 2037, verb 1822, adverb 829, pronoun 83,
preposition 81, determiner 46, conjunction 38, number 30,
modal auxiliary 13, be-verb 11, interjection 9, do-verb 5,
have-verb 4, infinitive-to 1
```

- **Assumption: POS strings match exactly (case-insensitively).** `matches()`
  compares `target_pos.lower()` against the stored tag by string equality. A
  caller passing `"aux"` or `"modal"` will silently match nothing against
  `"modal auxiliary"`; callers must use the tags above verbatim. The
  fine-grained verb tags (`be-verb`, `do-verb`, `have-verb`) and
  `infinitive-to` are easy to get wrong.
- **Assumption: one word can carry multiple POS.** 1,032 headwords have more
  than one distinct POS. `entries()` returns all `(pos, level)` pairs;
  `matches()` filters to the requested POS and returns the first surviving
  pair. Downstream code that assumes a single POS per word will mis-handle
  these.

### Multi-word and multi-token headwords

- **148 headwords contain a space** (multi-word entries). The semantic branch
  still skips these: the `isalpha()` filter drops them, since scoring a phrase
  as a single blank fill isn't meaningful. Single words that split into
  multiple subwords, by contrast, *are* now scored (via a k-mask pass,
  length-normalized to the same [0, 1] scale — see `score_cefr_candidates()`),
  so the earlier "drop every multi-token word" gap is closed for single words.
  The spelling branch is unaffected throughout (it works on the surface form
  directly).

## FastText vectors (`data/fasttext/pruned_cefr_j.vec`)

**Provenance.** Derived, CEFR-J-scoped FastText vectors — 4,252 words × 300
dims. Used only by the semantic branch's cosine-tiering stage.

- **Assumption: partial coverage is expected.** ~4.25k vectors against 8.75k
  headwords means a large share of CEFR-J words have no vector. The branch
  already surfaces this (`n_scored_by_model` vs `n_with_vectors` in the debug
  output — e.g. 50 vs 49 for `ask`); candidates without a vector are scored by
  the MLM but cannot be placed in a cosine tier. Do not assume every scored
  candidate is tierable.
- The full Common Crawl subword model (`cc.en.300.bin.gz`) is a genuine
  subword-hashing model and would give a vector for *any* string, but it needs
  ~5GB+ RAM and is only queried via `fasttext_bin_probe.py`; the pruned `.vec`
  is what the branch uses at runtime.

## JMdict (`data/edict/JMdict_e.xml.gz`)

**Provenance.** JMdict, from the Electronic Dictionary Research and Development
Group. Source for the katakana-loanword (gairaigo) index built by
`build_loanword_index.py` into `pipeline/cache/loanwords.json`.

- **Assumption: katakana reading = phonetic collision key.** The gairaigo
  branch groups English words by shared katakana reading. A known hazard here
  is homograph collisions — unrelated senses sharing one reading — which
  produced a real bug already fixed in the spelling branch (see README Status).
  Any change to the loanword index should re-check for reading collisions.

## Extended WordNet (`data/wordnet_ewn/*.yaml`)

**Provenance.** Extended WordNet (Global WordNet Association). Present in the
tree but **not wired into either branch** — flagged as a future
candidate-generation signal. No current code depends on its shape.

## Pilot sentences (`data/pilot/`)

The only committed dataset (your own). Supplies the 44 target words and their
sentences for the batch run. Assumed to provide, per row, a target word with a
CEFR-J-valid POS/level and a sentence containing a blankable occurrence of the
word (`make_stem()` locates and masks it, including via inflection matching).
