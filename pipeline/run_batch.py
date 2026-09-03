"""
Run BOTH branches over every distinct word in the pilot CSV (first sentence
per word — 44 words) and write one combined JSON that the interactive HTML
app (Pipeline Trace) reads to let you pick any word from a dropdown and see
real output for it, instead of the hardcoded 3-word sample.

Must run in YOUR venv (semantic_branch needs torch/transformers). The
spelling branch has no such requirement but is run here too so everything
lands in one file.

Usage (from the DiagnosticDistractors folder, with your venv active):
    python3 -m pipeline.run_batch

Writes: pipeline/cache/batch_result.json
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib import metadata

from pipeline import spelling_challenge
from pipeline import semantic_branch

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "batch_result.json")
META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "batch_result.meta.json")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _provenance(n_words):
    """Fingerprint the model, dependencies, and source data behind this run, so a
    stale mix (old index + new vectors + a different model) is detectable."""
    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_words": n_words,
        # run_batch calls semantic_distractors with no model_name, so it uses
        # that function's default; keep this in sync if the default changes.
        "semantic_model": "roberta-large",
        "python": sys.version.split()[0],
        "packages": {p: ver(p) for p in ("torch", "transformers", "numpy",
                                          "python-Levenshtein", "pykakasi", "lxml")},
        "data_sha256": {
            "pruned_cefr_j.vec": _sha256(os.path.join(ROOT, "data", "fasttext", "pruned_cefr_j.vec")),
            "loanwords.json": _sha256(os.path.join(ROOT, "pipeline", "cache", "loanwords.json")),
            "pilot_csv": _sha256(semantic_branch.PILOT_CSV),
        },
    }
    return data


def main():
    rows = semantic_branch.sample_pilot(n_words=44)  # every distinct pilot word, first sentence
    results = []
    for i, row in enumerate(rows):
        word, pos, level, sentence = row["word"], row["pos"], row["level"], row["sentence"]
        print(f"[{i + 1}/{len(rows)}] {word} ...", flush=True)

        entry = {"word": word, "pos": pos, "level": level, "sentence": sentence}

        # Orthographic spelling-challenge distractors -- no model, no network,
        # no POS/level gate; always runs.
        try:
            entry["spelling"] = spelling_challenge.spelling_challenge_distractors(word, n=8)
        except Exception as e:
            entry["spelling"] = {"error": str(e)}

        # Semantic branch -- needs the target word blanked out of its sentence.
        stem, matched = semantic_branch.make_stem(sentence, word)
        if stem is None:
            entry["semantic"] = {"error": f"could not locate '{word}' (or an inflection) in its sentence"}
        else:
            entry["stem"] = stem
            entry["matched_surface_form"] = matched
            try:
                entry["semantic"] = semantic_branch.semantic_distractors(
                    stem, word, target_pos=pos, target_level=level
                )
            except Exception as e:
                entry["semantic"] = {"error": str(e)}

        results.append(entry)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(_provenance(len(results)), f, indent=2)
    print(f"\nwrote {OUT_PATH} ({len(results)} words)")
    print(f"wrote {META_PATH} (provenance: model, deps, data hashes)")


if __name__ == "__main__":
    main()
