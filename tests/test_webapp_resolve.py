"""
Tests for the web app's CEFR-J target-word gate and pos/level resolution
(webapp/app.py). These need fastapi/pydantic installed (the [web] extra); the
whole file skips itself cleanly where they aren't, like the model tests do.

The behavior under test: a target word must be in CEFR-J, and its (pos, level)
is resolved from CEFR-J without ever guessing -- a word that stays ambiguous is
rejected rather than silently tagged with its first-listed sense.
"""
import pytest

from pipeline import cefr_lookup as cefr

fastapi = pytest.importorskip("fastapi")
from fastapi import HTTPException
from webapp.app import _resolve_cefr_entry


def resolve(word, pos=None, level=None):
    return _resolve_cefr_entry(word, cefr.entries(word), pos, level)


def test_unambiguous_word_auto_fills_from_cefr_j():
    # 'ask' has a single CEFR-J entry, so nothing needs to be supplied.
    assert resolve("ask") == ("verb", "A1")


def test_multi_sense_word_is_rejected_when_nothing_disambiguates_it():
    # 'brush' is both noun/A1 and verb/A1; the old code silently took the
    # first (noun). Now it must be disambiguated rather than guessed.
    with pytest.raises(HTTPException) as exc:
        resolve("brush")
    assert exc.value.status_code == 400
    assert "more than one sense" in exc.value.detail


def test_multi_sense_word_resolves_once_pos_is_given():
    assert resolve("brush", pos="verb") == ("verb", "A1")
    assert resolve("run", pos="verb") == ("verb", "A1")


def test_pos_not_in_cefr_j_for_that_word_is_rejected():
    # 'run' is noun/verb only; asking for adjective is invalid.
    with pytest.raises(HTTPException) as exc:
        resolve("run", pos="adjective")
    assert exc.value.status_code == 400
    assert "no CEFR-J entry matching" in exc.value.detail


def test_case_insensitive_pos_match():
    assert resolve("brush", pos="VERB") == ("verb", "A1")
