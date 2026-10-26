# SPDX-License-Identifier: AGPL-3.0-only
from banc_realite.normalize import normalize_text, piece_basename


def test_normalize_typography_and_case():
    assert normalize_text("Beacon Court — « 32.906 »") == "beacon court - 32.906"


def test_piece_basename_from_gold_path():
    assert piece_basename("documents/17-collection-register.md") == "17-collection-register.md"


def test_piece_basename_from_area_path():
    assert piece_basename("semantic-memory/pages/01-contract-summary.md") == "01-contract-summary.md"
