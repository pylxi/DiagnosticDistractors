# Diagnostic Distractors

A distractor generator for English vocabulary diagnostic tests aimed at Japanese
learners, built around the CEFR-J proficiency framework. Given a target word,
its CEFR-J level/part of speech, and (for the semantic branch) a sentence, it
produces plausible wrong answers for a multiple-choice question — without an
LLM in the runtime pipeline itself.

## Why two branches

Distractors fail for two very different reasons, so there are two independent
generators, merged into one pool for human review:

- **Spelling / phonetic branch** (`pipeline/spelling_branch.py`) — catches
  confusions from how a word *looks or sounds* to a Japanese learner:
  identical katakana loanword readings (bus/bath), common sound-substitution
  errors (L/R, B/V, TH→S/Z, F/H), and plain English-spelling edit distance
  (quiet/quite). No model inference — pure lookup and rule-based generation
  over JMdict and the CEFR-J wordlist.
- **Semantic branch** (`pipeline/semantic_branch.py`) — catches confusions
  from *meaning in context*: a local DeBERTa-v3 masked-language model scores
  every CEFR-J word at the target's level for how well it fits the blanked
  sentence, then FastText cosine similarity sorts survivors into a spread of
  plausibility (near-miss / thematic / control) rather than three
  near-synonyms.

Both branches only ever draw candidates from the CEFR-J wordlist at the
learner's level (±1 as a fallback) — a distractor above the learner's level
isn't a distractor, it's a giveaway, so the level window is never widened to
fix a low-yield sentence.

See the live walkthroughs for more detail (real code, real intermediate
data, not a mockup):
- [Pipeline Trace](https://claude.ai/code/artifact/16bbb447-666e-4e9f-bbbf-f0c296e5b807) —
  stage-by-stage trace of both branches, including the pool-reduction funnel
  and copy-paste distractor export.
- [Dual-Pipeline Distractors](https://claude.ai/code/artifact/38c37115-83ec-4481-9efa-a45d07b7b692) —
  architecture diagram walking one example end to end.

## Layout

```
DiagnosticDistractors/
├── pipeline/                   # all pipeline code
│   ├── cefr_lookup.py          # CEFR-J word/POS/level lookup (used by both branches)
│   ├── gairaigo.py             # katakana loanword collision + near-neighbor lookup
│   ├── phonetic_swaps.py       # L/R, B/V, TH->S/Z, F/H substitution rules
│   ├── levenshtein_search.py   # plain English-spelling edit-distance search
│   ├── spelling_branch.py      # orchestrates the four spelling signals above
│   ├── build_loanword_index.py # builds pipeline/cache/loanwords.json from JMdict
│   ├── fasttext_bin_probe.py   # low-memory reader for the full cc.en.300.bin FastText model
│   ├── semantic_branch.py      # DeBERTa masking + CEFR-constrained scoring + tiering
│   ├── step1_load_data.py      # sanity-checks that every data source loads
│   └── cache/                  # generated — loanwords.json, semantic_branch_result.json
├── data/
│   ├── pilot/                  # your own pilot sentences (committed)
│   ├── CEFR-J/                 # external — see Data sources below (gitignored)
│   ├── fasttext/                # external (gitignored)
│   ├── edict/                   # external (gitignored)
│   └── wordnet_ewn/              # external, not yet wired into either branch (gitignored)
├── experiments/                 # ad-hoc scratch output, not part of the pipeline (gitignored)
└── requirements.txt
```

## Setup

The spelling branch and all the lookup/data-loading code need nothing beyond
`requirements.txt` and run fine in a lightweight environment. **The semantic
branch needs a real machine with network access**: it downloads
`microsoft/deberta-v3-large` on first use and needs enough RAM/disk to run
it. Run it from a normal desktop/laptop Python environment (a venv is
enough), not a constrained sandbox.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running it

```bash
# Confirm every data source loads (run this first after setup, or after
# moving/updating any data file)
python3 pipeline/step1_load_data.py

# Build the loanword index from JMdict (run once, or whenever JMdict updates)
python3 pipeline/build_loanword_index.py

# Spelling branch — no network/model needed
python3 pipeline/spelling_branch.py glass light bus collar quiet

# Semantic branch — needs the venv above with torch/transformers, and network
# access the first time (to download the model)
python3 pipeline/semantic_branch.py
```

`semantic_branch.py` currently runs against a 3-word sample of the pilot CSV
(see `sample_pilot()` in that file) and writes
`pipeline/cache/semantic_branch_result.json`. Running it against the full
pilot set is the next step (see Status below).

## Data sources

Only `data/pilot/` is committed to this repo (it's your own dataset). Every
other folder under `data/` is external, third-party-licensed, and/or too
large for git — download each into the path shown:

| Path | Source | Notes |
|---|---|---|
| `data/CEFR-J/{A1,A2,B1,B2,C1,C2}/words.csv` | CEFR-J wordlist (Tono Lab, Tokyo University of Foreign Studies) | ~8,755 unique headword/POS variants; the candidate universe for both branches |
| `data/fasttext/pruned_cefr_j.vec` | Derived FastText vectors, CEFR-J-scoped (4,252 words × 300 dims) | Used for semantic-branch cosine tiering |
| `data/fasttext/wiki-news-300d-1M.vec` | [fastText pre-trained vectors](https://fasttext.cc/docs/en/english-vectors.html) | Not currently used by either branch |
| `data/fasttext/cc.en.300.bin.gz` | [fastText Common Crawl vectors, subword model](https://fasttext.cc/docs/en/crawl-vectors.html) | Real subword-hashing model; query only via `pipeline/fasttext_bin_probe.py` — a normal load needs ~5GB+ RAM |
| `data/edict/JMdict_e.xml.gz` | [JMdict](https://www.edrdg.org/jmdict/j_jmdict.html) (Electronic Dictionary Research and Development Group) | Source for the loanword/gairaigo index |
| `data/wordnet_ewn/*.yaml` | [Extended WordNet](https://github.com/globalwordnet/english-wordnet) | Not yet wired into either branch — flagged as a candidate-generation signal to add later |

## Status

- **Spelling branch: done.** All four signals (gairaigo exact, phonetic swap,
  gairaigo near, Levenshtein) validated, including a fix for a JMdict
  homograph-collision bug (unrelated senses sharing one katakana reading).
- **Semantic branch: done for a 3-word sample**, confirmed end to end
  (masking → CEFR-constrained DeBERTa scoring → FastText cosine tiering).
  Fixed a real low-yield bug (candidate generation now scores CEFR-J words
  directly instead of filtering DeBERTa's unconstrained output) and a tier-
  visibility bug (candidates between the 70th–95th percentile were silently
  dropped from the debug output).
- **Not yet done:** running the full 44-word / 598-row pilot batch;
  WordNet-seeded candidates as a secondary semantic signal; multi-subword-
  token scoring in the semantic branch; merging both branches' output into
  one candidate pool with a human review step.

## A note on `experiments/`

`experiments/verb_embeddings.npy` + `verb_list.txt` were found alongside the
rest of the project files during this cleanup and moved here rather than
deleted — they look like DeBERTa embeddings for the 44 pilot verbs generated
outside the scripted pipeline. Kept for now in case they're still needed;
safe to delete if not.
