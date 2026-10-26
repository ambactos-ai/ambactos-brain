# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — étiquetage ontologie et RagFiltreOntologie."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for sub in ("scripts", "src"):
    entry = str(_ROOT / sub)
    if entry not in sys.path:
        sys.path.insert(0, entry)

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from banc_realite.etiquetage_ontologie import (
    entites_filtre_depuis_question,
    etiqueter_passages,
    filtrer_passages_par_entites_etiquettes,
)
from banc_realite.fournisseurs_contexte import RagFiltreOntologie


def _mini_ontologie() -> dict:
    return {
        "familles": [
            {
                "nom": "juridique",
                "regle": {"chemin": "JURIDIQUE"},
                "traitement": "lecture_simple",
            },
            {
                "nom": "travaux",
                "regle": {"chemin": "TRAVAUX"},
                "traitement": "reconstitution_entite",
            },
        ],
        "entites": [
            {
                "nom": "Alpha Corp",
                "type": "autre",
                "cles": ["ALPHA-1234", "SECRET#MASK"],
                "variantes": ["alpha corp"],
            },
            {
                "nom": "Beta SA",
                "type": "fournisseur",
                "cles": ["Beta SA"],
                "variantes": ["BÊTA sa"],
            },
            {
                "nom": "Gamma",
                "type": "autre",
                "cles": ["GAM"],
                "variantes": [],
            },
        ],
        "concepts": [
            {"nom": "loyer mensuel", "concept": "montant"},
            {"nom": "clause résolutoire", "concept": "regle"},
        ],
        "relations": [
            {
                "de_type": "autre",
                "verbe": "cite",
                "vers_type": "fournisseur",
                "exemple": {"de": "Alpha Corp", "vers": "Beta SA"},
            }
        ],
        "decisions": [],
        "sources": {},
    }


def _passages() -> list[dict]:
    return [
        {
            "id": "p1",
            "piece": "dossier/JURIDIQUE/bail.md",
            "texte": "Le contrat ALPHA-1234 fixe le loyer mensuel.",
        },
        {
            "id": "p2",
            "piece": "04 — TRAVAUX/devis.md",
            "texte": "Devis pour BÊTA sa sans autre entité.",
        },
        {
            "id": "p3",
            "piece": "autre.md",
            "texte": "SECRET#MASK ignoré ; mention de Gamma ici.",
        },
        {
            "id": "p4",
            "piece": "note.md",
            "texte": "Rien d'identifiable, sauf clause résolutoire.",
        },
    ]


def test_etiqueter_passages_famille_entites_concepts():
    ontologie = _mini_ontologie()
    out = etiqueter_passages(ontologie, _passages())
    assert out["p1"]["famille"] == "juridique"
    assert out["p2"]["famille"] == "travaux"
    assert out["p1"]["entites"] == ["Alpha Corp"]
    assert out["p2"]["entites"] == ["Beta SA"]
    assert "Gamma" in out["p3"]["entites"]
    assert "Alpha Corp" not in out["p3"]["entites"]
    assert "loyer mensuel" in out["p1"]["concepts"]
    assert "clause résolutoire" in out["p4"]["concepts"]


def test_entites_filtre_question_et_relation():
    ontologie = _mini_ontologie()
    directes, relations, filtre, repli = entites_filtre_depuis_question(
        "Question sur ALPHA-1234 et la suite",
        ontologie,
    )
    assert repli is False
    assert directes == ["Alpha Corp"]
    assert relations == ["Beta SA"]
    assert filtre == {"Alpha Corp", "Beta SA"}


def test_repli_sans_filtre_question():
    ontologie = _mini_ontologie()
    directes, relations, filtre, repli = entites_filtre_depuis_question(
        "Question générique sans clé",
        ontologie,
    )
    assert repli is True
    assert directes == []
    assert relations == []
    assert filtre == set()


def test_filtrer_passages_par_etiquettes():
    etiquettes = {
        "1": {"entites": ["Alpha Corp"]},
        "2": {"entites": ["Beta SA"]},
        "3": {"entites": []},
    }
    vivier = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    gardes = filtrer_passages_par_entites_etiquettes(vivier, {"Beta SA"}, etiquettes)
    assert [g["id"] for g in gardes] == ["2"]


def test_rag_filtre_ontologie_repli_et_filtre(tmp_path: Path):
    ontologie = _mini_ontologie()
    etiquettes = {
        "schema": "etiquettes-passages/1.0.0",
        "passages": {
            "10": {"entites": ["Alpha Corp"], "famille": "juridique", "concepts": []},
            "20": {"entites": ["Beta SA"], "famille": "travaux", "concepts": []},
        },
    }
    ontologie_path = tmp_path / "ontologie.v1.json"
    etiquettes_path = tmp_path / "etiquettes.v1.json"
    ontologie_path.write_text(json.dumps(ontologie), encoding="utf-8")
    etiquettes_path.write_text(json.dumps(etiquettes), encoding="utf-8")

    provider = RagFiltreOntologie(
        {"q?": [1.0, 0.0]},
        dossier="greenloop",
        etiquettes_path=etiquettes_path,
        ontologie_version="v1",
    )
    provider._ontologie = ontologie
    provider._etiquettes_passages = etiquettes["passages"]

    vivier = [
        {"id": "10", "text": "a", "rank": 1},
        {"id": "20", "text": "b", "rank": 2},
        {"id": "30", "text": "c", "rank": 3},
    ]
    provider._rag = MagicMock()
    provider._rag.contexte = MagicMock(return_value=vivier)

    out_repli = provider.contexte({"id": "q1", "question": "sans entité"}, k=5, connection=None)
    assert len(out_repli) == 3
    stats_repli = provider.stats_par_question["q1"]
    assert stats_repli["repli_sans_filtre"] is True

    out_filtre = provider.contexte(
        {"id": "q2", "question": "ALPHA-1234"},
        k=5,
        connection=None,
    )
    assert [item["id"] for item in out_filtre] == ["10", "20"]
    stats_filtre = provider.stats_par_question["q2"]
    assert stats_filtre["entites_filtre"] == ["Alpha Corp"]
    assert stats_filtre["entites_relations"] == ["Beta SA"]
    assert stats_filtre["repli_sans_filtre"] is False
