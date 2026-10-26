# SPDX-License-Identifier: AGPL-3.0-only
"""Découpage déterministe des passages — inerte."""
from __future__ import annotations

from pathlib import Path

import pytest

from banc_realite.couche_brute_moreau import CouvertureDecoupeError, build_couche_brute_moreau
from banc_realite.passages import (
    MAX_CHARS,
    chunk_corpus,
    chunk_piece,
    lignes_non_couvertes,
    stats_duplication,
)


def test_chunk_deterministic(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = (
        "# Section A\n\n"
        "Paragraph one with enough text.\n\n"
        "## Section B\n\n"
        "| Col | Val |\n"
        "| --- | --- |\n"
        "| a | 1 |\n"
        "| b | 2 |\n"
    )
    (corpus / "01-demo.md").write_text(text, encoding="utf-8")
    first = chunk_piece("01-demo.md", corpus=corpus)
    second = chunk_piece("01-demo.md", corpus=corpus)
    assert first == second
    assert all(len(p.content) <= MAX_CHARS for p in first)
    assert any(p.section_path for p in first)


def test_table_row_never_split(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    long_cell = "x" * 500
    row = f"| {long_cell} | value |"
    (corpus / "02-table.md").write_text(f"# T\n\n{row}\n", encoding="utf-8")
    passages = chunk_piece("02-table.md", corpus=corpus)
    for passage in passages:
        for line in passage.content.splitlines():
            if line.strip().startswith("|"):
                assert line.count("|") >= 2




def test_tableau_recouvrement_large_ne_perd_pas_de_lignes(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    max_chars = 400
    overlap_chars = 200
    cell = "c" * 280
    rows = [f"| {cell} | v{i} |" for i in range(12)]
    (corpus / "table.md").write_text("# T\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    passages = chunk_piece(
        "table.md", corpus=corpus, max_chars=max_chars, overlap_chars=overlap_chars
    )
    assert lignes_non_couvertes(corpus, passages) == []
    source_lines = (corpus / "table.md").read_text(encoding="utf-8").splitlines()
    for passage in passages:
        first = passage.content.splitlines()[0]
        assert source_lines[passage.line_start - 1].strip() == first.strip()


def test_ligne_seule_hors_gabarit_comptee(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    max_chars = 200
    long_line = "| " + "x" * 350 + " | val |"
    (corpus / "wide.md").write_text(f"# T\n\n{long_line}\n", encoding="utf-8")
    passages = chunk_piece("wide.md", corpus=corpus, max_chars=max_chars)
    wide = [p for p in passages if long_line in p.content.splitlines()]
    assert len(wide) == 1
    assert len(wide[0].content) > max_chars
    assert wide[0].line_start == wide[0].line_end == 3
    stats = stats_duplication(passages, corpus_dir=corpus, max_chars=max_chars)
    assert stats["passages_hors_gabarit"] == 1
    assert lignes_non_couvertes(corpus, passages) == []


def test_build_refuse_lignes_non_couvertes_simulees(tmp_path: Path, monkeypatch):
    sorties = tmp_path / "couche-brute"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "01-demo.md").write_text("Contenu minimal.\n", encoding="utf-8")
    fake_dossier = type(
        "D",
        (),
        {"nom": "moreau", "corpus_dir": corpus, "vault_ref": "moreau"},
    )()

    def _fake_embed(_text: str) -> list[float]:
        return [0.1] * 8

    monkeypatch.setattr(
        "banc_realite.couche_brute_moreau.dossier_courant",
        lambda _nom=None: fake_dossier,
    )
    monkeypatch.setattr(
        "banc_realite.couche_brute_moreau.load_gold",
        lambda _dossier=None: {
            "assertions": [
                {"id": "MOR-A01", "piece": "01-demo.md", "quote": "Contenu minimal."},
            ]
        },
    )
    monkeypatch.setattr(
        "banc_realite.couche_brute_moreau.lignes_non_couvertes",
        lambda _corpus_dir, _passages: [("01-demo.md", 99)],
    )

    class _Conn:
        def execute(self, *_a, **_k):
            return self

    with pytest.raises(CouvertureDecoupeError) as exc:
        build_couche_brute_moreau(
            _Conn(), embed_fn=_fake_embed, sorties_dir=sorties, dossier="moreau"
        )
    assert "ligne_hors_passage" in str(exc.value)
