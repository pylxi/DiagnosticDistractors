"""Diagnostic Distractors pipeline package.

Importable modules (see each module's docstring for details):

- ``cefr_lookup``          -- CEFR-J word/POS/level lookup, shared by both branches
- ``gairaigo``             -- katakana loanword collision + near-neighbor lookup
- ``phonetic_swaps``       -- L/R, B/V, TH->S/Z, F/H substitution rules
- ``levenshtein_search``   -- plain English-spelling edit-distance search
- ``spelling_branch``      -- orchestrates the four spelling signals above
- ``semantic_branch``      -- masked-LM scoring + FastText cosine tiering
- ``build_loanword_index`` -- builds pipeline/cache/loanwords.json from JMdict

Nothing heavy is imported here: torch/transformers are pulled in lazily inside
``semantic_branch`` only when the model is actually needed, so ``import
pipeline`` and the spelling-side modules stay dependency-light.
"""
