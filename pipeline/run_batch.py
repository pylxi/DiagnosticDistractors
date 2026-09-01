"""
Run BOTH branches over every distinct word in the pilot CSV (first sentence
per word — 44 words) and write one combined JSON that the interactive HTML
app (Pipeline Trace) reads to let you pick any word from a dropdown and see
real output for it, instead of the hardcoded 3-word sample.

Must run in YOUR venv (semantic_branch needs torch/transformers). The
spelling branch has no such requirement but is run here too so everything
lands in one file.

Usage (from the DiagnosticDistractors folder, with your venv active):
    python3 pipeline/run_batch.py

Writes: pipeline/cache/batch_result.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spelling_branch
import semantic_branch

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "batch_result.json")


def main():
    rows = semantic_branch.sample_pilot(n_words=44)  # every distinct pilot word, first sentence
    results = []
    for i, row in enumerate(rows):
        word, pos, level, sentence = row["word"], row["pos"], row["level"], row["sentence"]
        print(f"[{i + 1}/{len(rows)}] {word} ...", flush=True)

        entry = {"word": word, "pos": pos, "level": level, "sentence": sentence}

        # Spelling branch -- no model, no network, always runs.
        try:
            entry["spelling"] = spelling_branch.spelling_distractors(
                word, target_pos=pos, target_level=level, n=3
            )
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
    print(f"\nwrote {OUT_PATH} ({len(results)} words)")


if __name__ == "__main__":
    main()
