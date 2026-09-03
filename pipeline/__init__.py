"""Diagnostic Distractors pipeline package.

Importable modules (see each module's docstring for details):

- ``cefr_lookup``          -- CEFR-J word/POS/level lookup, shared by both branches
- ``gairaigo``             -- JMdict loanword (gairaigo) katakana lookup
- ``phonetic_swaps``       -- L/R, B/V, TH->S/Z, F/H substitution rules
- ``eng_to_katakana``      -- rule-based English->katakana transliteration fallback
- ``spelling_challenge``   -- orthographic distractors: look-alikes + transliteration
- ``semantic_branch``      -- masked-LM scoring + FastText cosine tiering
- ``build_loanword_index`` -- builds pipeline/cache/loanwords.json from JMdict

Nothing heavy is imported here: torch/transformers are pulled in lazily inside
``semantic_branch`` only when the model is actually needed, so ``import
pipeline`` and the spelling-side modules stay dependency-light.
"""
