# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path

from banc_realite.relecture import (
    excerpt_around_quote,
    find_quote_line,
    format_numbered_excerpt,
    format_triplet,
)


def test_excerpt_around_known_quote(tmp_path: Path):
    piece = "sample.md"
    lines = ["alpha", "beta quote target gamma", "delta", "epsilon", "zeta"]
    (tmp_path / piece).write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = excerpt_around_quote(piece, "Quote TARGET", corpus=tmp_path)
    assert result["status"] == "ok"
    assert "beta quote target gamma" in result["texte"]
    assert "→" in result["texte"]
    assert "alpha" in result["texte"]
    assert "delta" in result["texte"]


def test_excerpt_quote_not_found(tmp_path: Path):
    piece = "sample.md"
    (tmp_path / piece).write_text("ligne unique\n", encoding="utf-8")
    result = excerpt_around_quote(piece, "absente du fichier", corpus=tmp_path)
    assert result["status"] == "citation_introuvable"
    assert "introuvable" in result["texte"].lower()


def test_excerpt_quote_on_first_line(tmp_path: Path):
    piece = "first.md"
    content = ["première ligne avec CITATION", "deuxième", "troisième"]
    (tmp_path / piece).write_text("\n".join(content) + "\n", encoding="utf-8")
    line = find_quote_line(piece, "citation", corpus=tmp_path)
    assert line == 1
    excerpt = format_numbered_excerpt(content, line, before=2, after=2)
    assert excerpt.startswith("→    1")
    assert "première ligne" in excerpt
    assert "deuxième" in excerpt


def test_format_triplet_modele_avec_attribut():
    claim = {
        "entity_id": "collection_novembre",
        "value": "2025-04-23",
        "qualifiers": {"attribute_id": "date_paiement_saisi"},
        "extracted_by_skill": "note-body",
    }
    info = format_triplet(claim)
    assert info["triplet"] == "collection novembre · date paiement saisi : 2025-04-23"
    assert info["sans_attribut"] is False


def test_format_triplet_modele_sans_attribut():
    claim = {
        "entity_id": "collection_register",
        "value": "657",
        "qualifiers": {},
        "extracted_by_skill": "note-body",
    }
    info = format_triplet(claim)
    assert info["triplet"] == "collection register : 657"
    assert info["sans_attribut"] is True


def test_format_triplet_tabulaire():
    claim = {
        "entity_id": "ignored",
        "value": "Section A / Key: Value",
        "qualifiers": {"tabular_row": 2},
        "extracted_by_skill": "tabulaire-deterministe",
    }
    info = format_triplet(claim)
    assert info["triplet"] == "Section A / Key: Value"
    assert info["sans_attribut"] is False
