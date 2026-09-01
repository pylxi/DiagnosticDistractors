# Diagnostic Distractors — live web app

Turns the pipeline into something you can actually use, not just browse:
type any word + the sentence it's used in, get real distractors from both
branches on demand, check/uncheck which ones to keep, export a `.tsv`.
Not limited to the 44 pilot words — any word, as long as it (or an
inflected form) appears in the sentence, and either it's in the CEFR-J
wordlist or you supply its level/POS yourself.

This has to run from a real machine with network access and the model
cached — the same reason `semantic_branch.py` and `run_batch.py` only ever
ran from your own Terminal, not from Claude's sandboxed tools. If you want
Claude Code to take this further (styling, new features, deploying it
somewhere), point it at this folder — everything it needs to know is here
and in `pipeline/`.

## Run it

```bash
cd ~/Documents/DiagnosticDistractors
source venv/bin/activate
pip install -e ".[semantic,web]"     # pipeline package + model + fastapi/uvicorn
uvicorn webapp.app:app --reload --port 8000
```

First boot loads the model (roberta-large by default; 10-20s) before the
server reports ready — that's expected, it only happens once. Then open
**http://127.0.0.1:8000/**.

## What it does

- `webapp/app.py` — a small FastAPI server. One real endpoint,
  `POST /api/generate`, which is almost exactly the per-word body of
  `pipeline/run_batch.py`'s loop: look up the word's CEFR-J level/POS (or
  use what you typed), locate it in the sentence via
  `semantic_branch.make_stem()`, then call `spelling_branch.spelling_distractors()`
  and `semantic_branch.semantic_distractors()` and return both results as
  JSON. No new pipeline logic — it calls the same functions `run_batch.py`
  and the Pipeline Trace artifact are built on.
- `webapp/static/index.html` — the page: an input form, then every
  candidate from both branches rendered as a checkbox (pre-checked to match
  what the pipeline would auto-pick), then an export button that builds a
  `.tsv` from whatever's checked and downloads it — client-side, no server
  round-trip needed for that part.

## Known edges

- A target word must be in the CEFR-J wordlist — that list is the candidate
  universe both branches draw from, so a word outside it is rejected rather
  than run with a made-up level/POS.
- A word with more than one CEFR-J sense (noun *and* verb, like `run` or
  `brush`) is not auto-picked — the app asks you to choose the part of speech
  (and level, if still ambiguous), instead of silently taking the first-listed
  sense. Words with a single CEFR-J entry fill in automatically.
- A target word with no FastText vector produces an empty semantic-branch
  result (spelling branch is unaffected) — the page says so rather than
  showing nothing with no explanation.
- Every request scores against the model fresh; nothing is cached per word,
  so repeat lookups take the same ~1-2s as the first one. Fine for one
  person reviewing questions; would need a cache layer to serve many users.

## Not done yet (fair game for Claude Code to pick up)

- No auth / not meant to be exposed to the internet as-is — it's a local
  review tool.
- No persistence — reviewed sets exist only until you export them; nothing
  is saved server-side.
- Styling is a plain, undecorated version of the Pipeline Trace artifact's
  look — functional, not polished.
