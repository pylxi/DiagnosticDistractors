"""
Shared pytest setup: put pipeline/ on sys.path so tests can `import
cefr_lookup`, `import gairaigo`, etc. the same way the pipeline scripts
themselves do (they each do their own sys.path.insert at the top).

None of these tests import torch/transformers -- semantic_branch.py only
pulls those in lazily inside score_cefr_candidates()/_get_model_and_tokenizer(),
so this whole suite runs without a GPU, without network access, and without
the venv that has the ML deps installed. It DOES need the real data files
(data/CEFR-J/, data/fasttext/pruned_cefr_j.vec, pipeline/cache/loanwords.json)
since these are integration tests against real CEFR-J/JMdict content, not
mocked fixtures -- run `pipeline/build_loanword_index.py` first if
loanwords.json is missing.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(ROOT, "pipeline")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)
