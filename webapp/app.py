"""
Diagnostic Distractors -- live web app.

Wraps the existing pipeline (pipeline/spelling_branch.py and
pipeline/semantic_branch.py) behind a small HTTP API so a user can type any
word + sentence, generate real distractors on demand, review/adjust which
ones to keep, and export a TSV -- instead of only browsing the 44 precomputed
pilot words like the static Pipeline Trace page does.

Must run from a real desktop Python environment with network access and the
project's venv active (torch/transformers/lxml/etc. -- see requirements.txt).
The model (roberta-large by default) loads once at startup so the first real request isn't slow.

Run from the repo root:
    source venv/bin/activate
    pip install fastapi uvicorn
    uvicorn webapp.app:app --reload --port 8000
Then open http://127.0.0.1:8000/
"""
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import cefr_lookup as cefr
from pipeline import spelling_branch
from pipeline import semantic_branch

app = FastAPI(title="Diagnostic Distractors")

_MODEL_NAME = "roberta-large"


@app.on_event("startup")
def preload_model():
    # Loads the model once when the server boots instead of on the first
    # request a real user makes -- otherwise whoever hits Generate first
    # eats a model-load delay for everyone after them.
    print(f"Loading {_MODEL_NAME} (first boot may take a while)...", flush=True)
    semantic_branch._get_model_and_tokenizer(_MODEL_NAME)
    print("Model loaded. Ready.", flush=True)


@app.get("/api/health")
def health():
    return {"ok": True, "model": _MODEL_NAME}


@app.get("/api/lookup")
def lookup(word: str):
    """The CEFR-J entries for a word, so the UI can tell whether a target is
    valid at all and whether it needs a part of speech chosen. A word with a
    single distinct POS resolves automatically; one with several (brush =
    noun+verb) needs the user to pick, so the POS control is only shown then."""
    word = word.lower().strip()
    entries = cefr.entries(word)
    return {
        "word": word,
        "in_cefr_j": bool(entries),
        "entries": [{"pos": p, "level": l} for p, l in entries],
        "pos_options": sorted({p for p, _ in entries}),
    }


class GenerateRequest(BaseModel):
    word: str
    sentence: str
    pos: Optional[str] = None
    level: Optional[str] = None


def _resolve_cefr_entry(word, entries, req_pos, req_level):
    """Pick the single CEFR-J (pos, level) this request refers to.

    `entries` is the word's list of (pos, level) pairs from CEFR-J (non-empty).
    We never guess when the word is ambiguous: many CEFR-J headwords carry more
    than one entry (run/brush/call are noun *and* verb), so silently taking the
    first one -- as the old auto-detect did -- picks the wrong sense more often
    than not. Instead, if the request doesn't narrow the word to exactly one
    CEFR-J entry, we ask the caller to disambiguate.
    """
    req_pos = (req_pos or "").strip().lower() or None
    req_level = (req_level or "").strip() or None

    matching = [
        (p, l) for p, l in entries
        if (req_pos is None or p == req_pos) and (req_level is None or l == req_level)
    ]
    if len(matching) == 1:
        return matching[0]

    options = ", ".join(f"{p}/{l}" for p, l in entries)
    if not matching:
        raise HTTPException(
            400,
            f"'{word}' has no CEFR-J entry matching "
            f"{req_pos or 'any pos'}/{req_level or 'any level'}. "
            f"Its CEFR-J entries are: {options}.",
        )
    raise HTTPException(
        400,
        f"'{word}' has more than one sense in CEFR-J ({options}); "
        f"choose the part of speech (and level, if still ambiguous) you mean.",
    )


@app.post("/api/generate")
def generate(req: GenerateRequest):
    word = req.word.lower().strip()
    sentence = req.sentence.strip()

    if not word or not sentence:
        raise HTTPException(400, "Both a word and a sentence are required.")

    # Target words must come from the CEFR-J wordlist -- that list is the whole
    # candidate universe both branches draw distractors from, so a target
    # outside it isn't allowed (no manual pos/level override to smuggle one in).
    entries = cefr.entries(word)
    if not entries:
        raise HTTPException(
            400,
            f"'{word}' isn't in the CEFR-J wordlist. Only CEFR-J words can be "
            f"used as target words.",
        )
    pos, level = _resolve_cefr_entry(word, entries, req.pos, req.level)

    stem, matched = semantic_branch.make_stem(sentence, word)
    if stem is None:
        raise HTTPException(
            400,
            f"Couldn't find '{word}' (or a recognized inflected form of it, "
            f"like -ed/-ing/-s) anywhere in that sentence.",
        )

    entry = {
        "word": word,
        "pos": pos,
        "level": level,
        "sentence": sentence,
        "stem": stem,
        "matched_surface_form": matched,
        "in_cefr_j": True,  # enforced above; kept for the frontend's shape
    }

    try:
        # n=8 (not the pipeline's default of 3) so the review step in the UI
        # has more than the bare minimum to choose from.
        entry["spelling"] = spelling_branch.spelling_distractors(
            word, target_pos=pos, target_level=level, n=8
        )
    except Exception as e:
        entry["spelling"] = {"error": str(e)}

    try:
        entry["semantic"] = semantic_branch.semantic_distractors(
            stem, word, target_pos=pos, target_level=level
        )
    except Exception as e:
        entry["semantic"] = {"error": str(e)}

    return entry


# Serve the frontend last, so it doesn't shadow the /api/* routes above.
app.mount(
    "/",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True),
    name="static",
)
