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
  from *meaning in context*: a local masked-language model (roberta-large by
  default) scores every CEFR-J word at the target's level for how well it fits the blanked
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
├── pyproject.toml              # packaging + deps (install with `pip install -e .`)
├── pipeline/                   # all pipeline code (installable package `pipeline`)
│   ├── __init__.py             # package marker
│   ├── cefr_lookup.py          # CEFR-J word/POS/level lookup (used by both branches)
│   ├── gairaigo.py             # katakana loanword collision + near-neighbor lookup
│   ├── phonetic_swaps.py       # L/R, B/V, TH->S/Z, F/H substitution rules
│   ├── levenshtein_search.py   # plain English-spelling edit-distance search
│   ├── spelling_branch.py      # orchestrates the four spelling signals above
│   ├── build_loanword_index.py # builds pipeline/cache/loanwords.json from JMdict
│   ├── build_pruned_vectors.py # builds data/fasttext/pruned_cefr_j.vec from wiki-news
│   ├── fasttext_bin_probe.py   # low-memory reader for the full cc.en.300.bin FastText model
│   ├── semantic_branch.py      # masked-LM masking + CEFR-constrained scoring + tiering
│   ├── step1_load_data.py      # sanity-checks that every data source loads
│   └── cache/                  # generated — loanwords.json, semantic_branch_result.json
├── data/
│   ├── pilot/                  # your own pilot sentences (committed)
│   ├── CEFR-J/                 # external — see Data sources below (gitignored)
│   ├── fasttext/                # external (gitignored)
│   ├── edict/                   # external (gitignored)
│   └── wordnet_ewn/              # external, not yet wired into either branch (gitignored)
├── experiments/                 # ad-hoc scratch output, not part of the pipeline (gitignored)
├── webapp/                      # live web app -- type a word+sentence, review, export (see webapp/README.md)
├── docs/                        # design notes (scoring objectives, data assumptions)
├── scripts/debug/               # one-off incident diagnostics (see scripts/debug/README.md)
├── tests/                       # pytest suite (`python3 -m pytest tests/`)
└── requirements.txt
```

## Setup

The project installs as a package (`pipeline`) via `pyproject.toml`. The
spelling branch and all the lookup/data-loading code run fine in a lightweight
environment (the core install). **The semantic branch needs a real machine
with network access**: it downloads its model (`roberta-large` by default) on
first use and needs enough RAM/disk to run it. Run it from a normal
desktop/laptop Python environment (a venv is enough), not a constrained
sandbox.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all]"            # everything: spelling + semantic + web + tests
```

Lighter installs are available via extras when you don't need the whole thing:

```bash
pip install -e .                   # core only — spelling branch + data tooling
pip install -e ".[semantic]"       # + the masked-LM semantic branch (torch/transformers)
pip install -e ".[web]"            # + the FastAPI web app
```

(`pip install -r requirements.txt` still works and installs the full set.)

## Running it

Modules are run with `python3 -m pipeline.<name>` (from the repo root, venv
active) now that the pipeline is a package:

```bash
# Confirm every data source loads (run this first after setup, or after
# moving/updating any data file)
python3 -m pipeline.step1_load_data

# Build the two generated data artifacts (run once, or whenever their source
# data changes): the loanword index from JMdict, and the CEFR-J-scoped
# FastText vectors from wiki-news-300d-1M.vec
python3 -m pipeline.build_loanword_index
python3 -m pipeline.build_pruned_vectors

# Spelling branch — no network/model needed
python3 -m pipeline.spelling_branch glass light bus collar quiet

# Semantic branch — needs the [semantic] extra (torch/transformers), and
# network access the first time (to download the model)
python3 -m pipeline.semantic_branch
```

`semantic_branch.py` currently runs against a 3-word sample of the pilot CSV
(see `sample_pilot()` in that file) and writes
`pipeline/cache/semantic_branch_result.json`. Running it against the full
pilot set is the next step (see Status below).

## Live web app

The pipeline is also available as a small local web app instead of just the two CLI scripts above -- type any word and the sentence it's used in, get real distractors from both branches, review/adjust which to keep, export a `.tsv`. Not limited to the 44 pilot words. See `webapp/README.md` for how to run it (needs the same venv as the semantic branch, plus `pip install fastapi uvicorn`).

## Design notes

- [docs/scoring-objectives.md](docs/scoring-objectives.md) — why the two
  branches optimize different, independent objectives (surface-form
  confusability vs. contextual plausibility) and are merged as a union rather
  than fused into one score.
- [docs/data-assumptions.md](docs/data-assumptions.md) — what the pipeline
  assumes about its input data (CEFR-J POS/level semantics, corpus provenance,
  vector coverage, tokenization gaps) and where those assumptions can bite.

## Data sources

Only `data/pilot/` is committed to this repo (it's your own dataset). Every
other folder under `data/` is external, third-party-licensed, and/or too
large for git — download each into the path shown:

| Path | Source | Notes |
|---|---|---|
| `data/CEFR-J/{A1,A2,B1,B2,C1,C2}/words.csv` | CEFR-J wordlist (Tono Lab, Tokyo University of Foreign Studies) | ~8,755 unique headword/POS variants; the candidate universe for both branches |
| `data/fasttext/pruned_cefr_j.vec` | **Generated** by `pipeline/build_pruned_vectors.py` — don't download | CEFR-J-scoped subset (~8,579 words × 300 dims) of wiki-news; used for semantic-branch cosine tiering |
| `data/fasttext/wiki-news-300d-1M.vec` | [fastText pre-trained vectors](https://fasttext.cc/docs/en/english-vectors.html) | **Required** — the source `build_pruned_vectors.py` prunes to CEFR-J |
| `data/fasttext/cc.en.300.bin.gz` | [fastText Common Crawl vectors, subword model](https://fasttext.cc/docs/en/crawl-vectors.html) | Real subword-hashing model; query only via `pipeline/fasttext_bin_probe.py` — a normal load needs ~5GB+ RAM |
| `data/edict/JMdict_e.xml.gz` | [JMdict](https://www.edrdg.org/jmdict/j_jmdict.html) (Electronic Dictionary Research and Development Group) | Source for the loanword/gairaigo index |
| `data/wordnet_ewn/*.yaml` | [Extended WordNet](https://github.com/globalwordnet/english-wordnet) | Not yet wired into either branch — flagged as a candidate-generation signal to add later |

## Status

- **Spelling branch: done.** All four signals (gairaigo exact, phonetic swap,
  gairaigo near, Levenshtein) validated, including a fix for a JMdict
  homograph-collision bug (unrelated senses sharing one katakana reading).
- **Semantic branch: architecture done**, confirmed end to end
  (masking → CEFR-constrained scoring → FastText cosine tiering). Fixed a
  real low-yield bug (candidate generation now scores CEFR-J words directly
  instead of filtering the model's unconstrained output) and a tier-
  visibility bug (candidates between the 70th–95th percentile were silently
  dropped from the debug output).
- **2026-09-01: found and fixed a serious bug** — `microsoft/deberta-v3-large`
  was pretrained with ELECTRA-style replaced-token-detection, not
  masked-language-modeling, so its fill-mask head was never trained;
  `transformers` silently bolted on a freshly random-initialized head every
  time it loaded (the "tied weights"/`cls.predictions.*` MISSING warning at
  load time, previously and incorrectly assumed benign). Every semantic-branch
  result generated before this date was scoring candidates against random
  noise, not real contextual fit. Switched the default model to
  `roberta-large` (a genuinely MLM-pretrained model) and confirmed
  reproducible, sensible output across separate process restarts. The 44-word
  pilot batch and the Pipeline Trace artifact's data need to be regenerated
  against the fixed model.
- **Not yet done:** WordNet-seeded candidates as a secondary semantic signal;
  multi-subword-token scoring in the semantic branch; merging both branches'
  output into one candidate pool with a human review step; a fuller refresh
  of this Status section (it still doesn't mention the live web app or the
  pytest test suite, both of which exist now).

## A note on `experiments/`

`experiments/verb_embeddings.npy` + `verb_list.txt` were found alongside the
rest of the project files during this cleanup and moved here rather than
deleted — they look like DeBERTa embeddings for the 44 pilot verbs generated
outside the scripted pipeline. Kept for now in case they're still needed;
safe to delete if not.
