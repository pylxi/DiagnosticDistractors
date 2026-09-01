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


class GenerateRequest(BaseModel):
    word: str
    sentence: str
    pos: Optional[str] = None
    level: Optional[str] = None


@app.post("/api/generate")
def generate(req: GenerateRequest):
    word = req.word.lower().strip()
    sentence = req.sentence.strip()

    if not word or not sentence:
        raise HTTPException(400, "Both a word and a sentence are required.")

    pos, level = req.pos, req.level
    auto = cefr.entries(word)
    if not pos or not level:
        if not auto:
            raise HTTPException(
                400,
                f"'{word}' isn't in the CEFR-J wordlist, so its level/part of "
                f"speech can't be auto-detected. Enter a level and POS "
                f"yourself if you're sure this is the right target word.",
            )
        pos = pos or auto[0][0]
        level = level or auto[0][1]

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
        "in_cefr_j": bool(auto),
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
