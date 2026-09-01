"""Shared pytest notes.

Tests import the pipeline as a package (`from pipeline import cefr_lookup`,
etc.). The repo root is placed on sys.path via `[tool.pytest.ini_options]
pythonpath = ["."]` in pyproject.toml, so no path manipulation is needed here
(and `pip install -e .` makes the same imports work outside pytest).

None of these tests import torch/transformers -- semantic_branch.py only pulls
those in lazily inside score_cefr_candidates()/_get_model_and_tokenizer(), so
this whole suite runs without a GPU, without network access, and without the
ML deps installed. It DOES need the real data files (data/CEFR-J/,
data/fasttext/pruned_cefr_j.vec, pipeline/cache/loanwords.json) since these are
integration tests against real CEFR-J/JMdict content, not mocked fixtures --
run `python3 -m pipeline.build_loanword_index` first if loanwords.json is
missing.
"""
