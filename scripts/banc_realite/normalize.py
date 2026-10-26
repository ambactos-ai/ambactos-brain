# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic text normalization for gold ↔ memory matching."""
from __future__ import annotations

import re
import unicodedata


_TYPO_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u00ab": '"',
    "\u00bb": '"',
}
_DASHES = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
}


def normalize_text(text: str) -> str:
    """Lowercase, fold accents, unify punctuation and whitespace."""
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", text)
    for src, dst in {**_TYPO_QUOTES, **_DASHES}.items():
        value = value.replace(src, dst)
    value = value.lower()
    value = "".join(
        ch for ch in unicodedata.normalize("NFD", value)
        if unicodedata.category(ch) != "Mn"
    )
    value = re.sub(r"[*_`#>|]+", " ", value)
    value = re.sub(r"[^\w\s.,;:%/()+\-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def piece_basename(source: str) -> str:
    """Map gold `documents/foo.md` or area paths to `foo.md`."""
    path = source.replace("\\", "/").strip()
    if path.startswith("documents/"):
        path = path[len("documents/") :]
    if path.startswith("semantic-memory/pages/"):
        path = path[len("semantic-memory/pages/") :]
    if path.startswith("semantic-memory/draft/"):
        path = path[len("semantic-memory/draft/") :]
    return path.split("/")[-1]


def claim_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "body", "value", "content"):
            if key in value and isinstance(value[key], str):
                return value[key]
    return str(value)
