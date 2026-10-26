# SPDX-License-Identifier: AGPL-3.0-only
"""Protection écriture banc réel + contrôle verifier couche brute."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from banc_realite.couche_brute_moreau import (
    CoucheBruteChemins,
    build_couche_brute_moreau,
    regenerer_decoupe_config,
    verifier_couche_brute,
)
from banc_realite.dossiers import BANC_VAR


REAL_MOREAU_DECOUPE = BANC_VAR / "dossiers" / "moreau" / "couche-brute" / "decoupe-config.json"


class _FakeConnection:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows: list[dict] = list(rows or [])
        self._id = 0

    def execute(self, query, params=None):
        sql = " ".join(query.split())
        params = params or ()
        if sql.startswith("SET search_path"):
            return self
        if "ALTER TABLE" in sql or "CREATE INDEX" in sql or "UPDATE public.memory_chunks" in sql:
            return self
        if "INSERT INTO public.memory_chunks" in sql:
            self._id += 1
            piece, content, meta_json, _vec, vault_ref = params
            meta = json.loads(meta_json)
            self.rows.append(
                {
                    "id": self._id,
                    "source_ref": piece,
                    "content": content,
                    "metadata": meta,
                    "vault_ref": vault_ref,
                    "embedding_mistral": "[0.1]",
                }
            )
            return self
        if "embedding_mistral IS NULL" in sql:
            vault_ref = params[0]
            filtered = [
                r
                for r in self.rows
                if r.get("vault_ref") == vault_ref and r.get("embedding_mistral") is None
            ]
            return _Result(filtered)
        if "FROM public.memory_chunks" in sql and "vault_ref = %s" in sql:
            vault_ref = params[0]
            filtered = [r for r in self.rows if r.get("vault_ref") == vault_ref]
            return _Result(filtered)
        return _Result([])


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


def test_protection_ecriture_dossier_reel():
    with pytest.raises(AssertionError, match="écriture interdite"):
        REAL_MOREAU_DECOUPE.write_text('{"pollution": true}\n', encoding="utf-8")


def test_build_couche_brute_n_ecrit_pas_hors_tmp_path(tmp_path: Path, monkeypatch):
    sorties = tmp_path / "couche-brute"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    texte = "Texte unique pour isolation."
    (corpus / "01-demo.md").write_text(f"# A\n\n{texte}\n", encoding="utf-8")

    fake_dossier = type(
        "D",
        (),
        {
            "nom": "moreau",
            "corpus_dir": corpus,
            "vault_ref": "moreau",
        },
    )()

    monkeypatch.setattr(
        "banc_realite.couche_brute_moreau.dossier_courant",
        lambda _nom=None: fake_dossier,
    )
    monkeypatch.setattr(
        "banc_realite.couche_brute_moreau.load_gold",
        lambda _dossier=None: {
            "assertions": [{"id": "MOR-A01", "piece": "01-demo.md", "quote": texte}]
        },
    )

    conn = _FakeConnection()
    build_couche_brute_moreau(
        conn,
        dossier="moreau",
        sorties_dir=sorties,
        embed_fn=lambda _t: [0.1] * 8,
    )

    assert (sorties / "decoupe-config.json").is_file()
    assert (sorties / "build.json").is_file()
    assert not REAL_MOREAU_DECOUPE.with_name("pollution-test.json").exists()


def test_regenerer_decoupe_config_isolee(tmp_path: Path, monkeypatch):
    sorties = tmp_path / "couche-brute"
    monkeypatch.setattr(
        "banc_realite.couche_brute_moreau.load_gold",
        lambda _dossier=None: {
            "assertions": [{"id": "MOR-A01", "piece": "01-demo.md", "quote": "x" * 50}]
        },
    )
    decoupe = regenerer_decoupe_config("moreau", sorties_dir=sorties)
    assert decoupe["recouvrement_chars"] >= 49
    assert (sorties / "decoupe-config.json").is_file()


def _fixture_dossier(tmp_path: Path, monkeypatch, *, corpus_text: str, quote: str):
    sorties = tmp_path / "couche-brute"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    piece = "01-demo.md"
    (corpus / piece).write_text(corpus_text, encoding="utf-8")
    fake_dossier = type(
        "D",
        (),
        {"nom": "moreau", "corpus_dir": corpus, "vault_ref": "moreau"},
    )()
    monkeypatch.setattr(
        "banc_realite.couche_brute_moreau.dossier_courant",
        lambda _nom=None: fake_dossier,
    )
    monkeypatch.setattr(
        "banc_realite.couche_brute_moreau.load_gold",
        lambda _dossier=None: {
            "assertions": [{"id": "MOR-A01", "piece": piece, "quote": quote}]
        },
    )
    return sorties, fake_dossier


def test_verifier_echoue_si_passages_divergent_du_rapport(tmp_path: Path, monkeypatch):
    sorties, _d = _fixture_dossier(
        tmp_path,
        monkeypatch,
        corpus_text="# A\n\nTexte cohérent.\n",
        quote="Texte cohérent.",
    )
    chemins = CoucheBruteChemins.pour_dossier("moreau", sorties_dir=sorties)
    regenerer_decoupe_config("moreau", sorties_dir=sorties)
    chemins.build_json.write_text(
        json.dumps({"passages": 99, "recouvrement": {}}) + "\n",
        encoding="utf-8",
    )
    conn = _FakeConnection(
        [
            {
                "id": 1,
                "source_ref": "01-demo.md",
                "content": "Texte cohérent.",
                "metadata": {"piece": "01-demo.md", "line_start": 3, "line_end": 3},
                "vault_ref": "moreau",
                "embedding_mistral": "[0.1]",
            }
        ]
    )
    code, output = verifier_couche_brute(conn, "moreau", sorties_dir=sorties)
    assert code == 1
    assert "passages en base" in output


def test_verifier_echoue_si_vecteur_manquant(tmp_path: Path, monkeypatch):
    sorties, _d = _fixture_dossier(
        tmp_path,
        monkeypatch,
        corpus_text="# A\n\nTexte cohérent.\n",
        quote="Texte cohérent.",
    )
    regenerer_decoupe_config("moreau", sorties_dir=sorties)
    decoupe = json.loads((sorties / "decoupe-config.json").read_text(encoding="utf-8"))
    chemins = CoucheBruteChemins.pour_dossier("moreau", sorties_dir=sorties)
    chemins.build_json.write_text(
        json.dumps({"passages": 1, "recouvrement": decoupe, "citations_retrouvees": "1/1"})
        + "\n",
        encoding="utf-8",
    )
    conn = _FakeConnection(
        [
            {
                "id": 1,
                "source_ref": "01-demo.md",
                "content": "Texte cohérent.",
                "metadata": {"piece": "01-demo.md", "line_start": 3, "line_end": 3},
                "vault_ref": "moreau",
                "embedding_mistral": None,
            }
        ]
    )
    code, output = verifier_couche_brute(conn, "moreau", sorties_dir=sorties)
    assert code == 1
    assert "sans vecteur Mistral" in output


def test_verifier_echoue_si_couverture_incomplete(tmp_path: Path, monkeypatch):
    sorties, _d = _fixture_dossier(
        tmp_path,
        monkeypatch,
        corpus_text="# A\n\nTexte présent.\n",
        quote="citation absente",
    )
    regenerer_decoupe_config("moreau", sorties_dir=sorties)
    decoupe = json.loads((sorties / "decoupe-config.json").read_text(encoding="utf-8"))
    chemins = CoucheBruteChemins.pour_dossier("moreau", sorties_dir=sorties)
    chemins.build_json.write_text(
        json.dumps({"passages": 1, "recouvrement": decoupe}) + "\n",
        encoding="utf-8",
    )
    conn = _FakeConnection(
        [
            {
                "id": 1,
                "source_ref": "01-demo.md",
                "content": "Texte présent.",
                "metadata": {"piece": "01-demo.md", "line_start": 3, "line_end": 3},
                "vault_ref": "moreau",
                "embedding_mistral": "[0.1]",
            }
        ]
    )
    code, output = verifier_couche_brute(conn, "moreau", sorties_dir=sorties)
    assert code == 1
    assert "couverture citations" in output


def test_verifier_reussit_sur_dossier_coherent(tmp_path: Path, monkeypatch):
    sorties, _d = _fixture_dossier(
        tmp_path,
        monkeypatch,
        corpus_text="# A\n\nTexte cohérent pour verifier.\n",
        quote="Texte cohérent pour verifier.",
    )
    conn = _FakeConnection()
    build = build_couche_brute_moreau(
        conn,
        dossier="moreau",
        sorties_dir=sorties,
        embed_fn=lambda _t: [0.1] * 8,
    )
    assert build["citations_retrouvees"] == "1/1"
    code, output = verifier_couche_brute(conn, "moreau", sorties_dir=sorties)
    assert code == 0
    assert "Vérification couche brute MOREAU : OK" in output
