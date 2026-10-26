# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — composite objets + RAG + part lecture simple."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parents[3]
for sub in ("scripts", "src"):
    entry = str(_ROOT / sub)
    if entry not in sys.path:
        sys.path.insert(0, entry)

from banc_realite.fournisseurs_contexte import (
    BUDGET_ETAGE2_CHARS,
    PART_LIGNES,
    PART_LECTURE_SIMPLE,
    ObjetsSensPuisRagLectureSimple,
    ObjetsSensPuisRagLectureSimpleEntites,
    familles_lecture_simple,
    rendre_contexte,
)
from banc_realite.etiquetage_ontologie import entites_filtre_depuis_question
from banc_realite.recherche_entites import charger_index, objets_de_recherche

from tests.banc_realite.test_recherche_entites import _ecrire_fixture, _hit, _vecteurs_objets


def test_familles_lecture_simple_depuis_aiguillage():
    aiguillage = {
        "regles": [
            {"famille": "juridique", "traitement": "lecture_simple"},
            {"famille": "compta", "traitement": "registre_seul"},
            {"famille": "scan", "traitement": "ocr_lecture_simple"},
            {"famille": "juridique", "traitement": "reconstitution_entite"},
        ]
    }
    assert familles_lecture_simple(aiguillage) == frozenset({"juridique", "scan"})


def _passage(pid: str, text: str, *, source: str = "bail.md", line_start: int = 1) -> dict:
    item = _hit(text, source, 0.9, 1, line_start=line_start)
    item["id"] = pid
    return item


def test_composite_lecture_simple_budgets(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    question = {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"}

    etiquettes_path = tmp_path / "etiquettes.v1.json"
    etiquettes_path.write_text(
        json.dumps(
            {
                "schema": "etiquettes-passages/1.0.0",
                "passages": {
                    "ls1": {"famille": "juridique", "entites": [], "concepts": []},
                    "ls2": {"famille": "juridique", "entites": [], "concepts": []},
                    "rag1": {"famille": "travaux", "entites": [], "concepts": []},
                },
            }
        ),
        encoding="utf-8",
    )

    vivier = [
        _passage("ls1", "L" * 1500, line_start=10),
        _passage("ls2", "S" * 2000, line_start=20),
        _passage("rag1", "R" * 4000, line_start=30),
    ]
    rag_filtre = [_passage("rag1", "R" * 4000, line_start=30)]

    provider = ObjetsSensPuisRagLectureSimple(
        {question["question"]: [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
        etiquettes_path=etiquettes_path,
    )
    provider._familles_lecture_simple = frozenset({"juridique"})
    provider._rag._vivier = MagicMock(return_value=vivier)
    provider._rag.contexte = MagicMock(return_value=rag_filtre)

    elements = provider.contexte(question, k=12, connection=object())
    obj_chars = sum(len(str(e.get("text") or "")) for e in elements if e.get("kind") == "objet_sens")
    ls_chars = sum(
        len(str(e.get("text") or ""))
        for e in elements
        if e.get("kind") == "file_hit" and str(e.get("id")) in {"ls1", "ls2"}
    )
    assert obj_chars <= PART_LIGNES
    assert ls_chars <= PART_LECTURE_SIMPLE
    _rendered, kept = rendre_contexte(
        elements,
        budget_chars=BUDGET_ETAGE2_CHARS,
        question_id=question["id"],
        style="brain",
    )
    total = sum(len(str(e.get("text") or "")) for e in kept)
    assert total <= BUDGET_ETAGE2_CHARS
    stats = provider.stats_par_question["q1"]
    assert stats["repli_lecture_simple"] is False
    assert stats["car_lecture_simple"] == ls_chars


def test_repli_quand_aucun_passage_lecture_simple(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    question = {"id": "q2", "question": "Site Alpha tonnage January 2025 12.5"}

    etiquettes_path = tmp_path / "etiquettes.v1.json"
    etiquettes_path.write_text(
        json.dumps(
            {
                "schema": "etiquettes-passages/1.0.0",
                "passages": {
                    "rag1": {"famille": "travaux", "entites": [], "concepts": []},
                },
            }
        ),
        encoding="utf-8",
    )

    vivier = [_passage("rag1", "R" * 500, line_start=5)]
    rag_filtre = vivier

    provider = ObjetsSensPuisRagLectureSimple(
        {question["question"]: [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
        etiquettes_path=etiquettes_path,
    )
    provider._familles_lecture_simple = frozenset({"juridique"})
    provider._rag._vivier = MagicMock(return_value=vivier)
    provider._rag.contexte = MagicMock(return_value=rag_filtre)

    elements = provider.contexte(question, k=12, connection=object())
    stats = provider.stats_par_question["q2"]
    assert stats["repli_lecture_simple"] is True
    assert stats["n_lecture_simple"] == 0
    assert stats["car_lecture_simple"] == 0
    assert not any(str(e.get("id")) == "ls1" for e in elements)


def _mini_ontologie() -> dict:
    return {
        "familles": [],
        "entites": [
            {
                "nom": "Alpha Corp",
                "type": "autre",
                "cles": ["ALPHA-1234"],
                "variantes": [],
            },
            {
                "nom": "Beta SA",
                "type": "fournisseur",
                "cles": ["Beta SA"],
                "variantes": [],
            },
        ],
        "concepts": [],
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


def _ecrire_ontologie_etiquettes(tmp_path: Path) -> Path:
    ontologie_path = tmp_path / "ontologie.v1.json"
    etiquettes_path = tmp_path / "etiquettes.v1.json"
    ontologie_path.write_text(json.dumps(_mini_ontologie()), encoding="utf-8")
    etiquettes_path.write_text(
        json.dumps(
            {
                "schema": "etiquettes-passages/1.0.0",
                "passages": {
                    "ls_alpha": {
                        "famille": "juridique",
                        "entites": ["Alpha Corp"],
                        "concepts": [],
                    },
                    "ls_beta": {
                        "famille": "juridique",
                        "entites": ["Beta SA"],
                        "concepts": [],
                    },
                    "ls_gamma": {
                        "famille": "juridique",
                        "entites": ["Gamma"],
                        "concepts": [],
                    },
                    "rag1": {"famille": "travaux", "entites": [], "concepts": []},
                },
            }
        ),
        encoding="utf-8",
    )
    return etiquettes_path


def test_lecture_simple_entites_filtre_passages(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    etiquettes_path = _ecrire_ontologie_etiquettes(tmp_path)
    question = {
        "id": "q_ent",
        "question": "Détails sur ALPHA-1234 et la suite",
    }
    _, _, filtre, repli = entites_filtre_depuis_question(question["question"], _mini_ontologie())
    assert repli is False
    assert filtre == {"Alpha Corp", "Beta SA"}

    vivier = [
        _passage("ls_alpha", "A" * 400, line_start=1),
        _passage("ls_beta", "B" * 400, line_start=2),
        _passage("ls_gamma", "G" * 400, line_start=3),
        _passage("rag1", "R" * 200, line_start=4),
    ]
    provider = ObjetsSensPuisRagLectureSimpleEntites(
        {question["question"]: [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
        etiquettes_path=etiquettes_path,
    )
    provider._familles_lecture_simple = frozenset({"juridique"})
    provider._rag._vivier = MagicMock(return_value=vivier)
    provider._rag.contexte = MagicMock(return_value=[vivier[-1]])

    elements = provider.contexte(question, k=12, connection=object())
    ls_ids = {
        str(e.get("id"))
        for e in elements
        if e.get("kind") == "file_hit" and str(e.get("id", "")).startswith("ls_")
    }
    assert ls_ids <= {"ls_alpha", "ls_beta"}
    assert "ls_gamma" not in ls_ids
    stats = provider.stats_par_question["q_ent"]
    assert stats["repli_entites"] is False
    assert stats["entites_filtre"] == ["Alpha Corp"]
    assert stats["n_lecture_simple_filtre"] == 2


def test_lecture_simple_entites_repli_sans_entite_dans_question(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    etiquettes_path = _ecrire_ontologie_etiquettes(tmp_path)
    question = {"id": "q_repli", "question": "Question générique sans clé ontologique"}

    vivier = [
        _passage("ls_alpha", "A" * 300, line_start=1),
        _passage("ls_gamma", "G" * 300, line_start=2),
    ]
    provider = ObjetsSensPuisRagLectureSimpleEntites(
        {question["question"]: [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
        etiquettes_path=etiquettes_path,
    )
    provider._familles_lecture_simple = frozenset({"juridique"})
    provider._rag._vivier = MagicMock(return_value=vivier)
    provider._rag.contexte = MagicMock(return_value=[])

    elements = provider.contexte(question, k=12, connection=object())
    ls_ids = {
        str(e.get("id"))
        for e in elements
        if e.get("kind") == "file_hit" and str(e.get("id", "")).startswith("ls_")
    }
    assert ls_ids == {"ls_alpha", "ls_gamma"}
    stats = provider.stats_par_question["q_repli"]
    assert stats["repli_entites"] is True


def test_lecture_simple_entites_budget_5000(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    etiquettes_path = _ecrire_ontologie_etiquettes(tmp_path)
    question = {"id": "q5k", "question": "ALPHA-1234"}

    vivier = [
        _passage("ls_alpha", "A" * 2500, line_start=1),
        _passage("ls_beta", "B" * 2500, line_start=2),
        _passage("ls_gamma", "G" * 500, line_start=3),
    ]
    provider = ObjetsSensPuisRagLectureSimpleEntites(
        {question["question"]: [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
        etiquettes_path=etiquettes_path,
        part_lecture_simple=5_000,
    )
    provider._familles_lecture_simple = frozenset({"juridique"})
    provider._rag._vivier = MagicMock(return_value=vivier)
    provider._rag.contexte = MagicMock(return_value=[])

    elements = provider.contexte(question, k=12, connection=object())
    ls_chars = sum(
        len(str(e.get("text") or ""))
        for e in elements
        if e.get("kind") == "file_hit" and str(e.get("id", "")).startswith("ls_")
    )
    assert ls_chars <= 5_000
    _rendered, kept = rendre_contexte(
        elements,
        budget_chars=BUDGET_ETAGE2_CHARS,
        question_id=question["id"],
        style="brain",
    )
    total = sum(len(str(e.get("text") or "")) for e in kept)
    assert total <= BUDGET_ETAGE2_CHARS
