# SPDX-License-Identifier: AGPL-3.0-only
"""Instrument plafond du filtre par l'or — chemins et doubles injectés."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from banc_realite.dossiers import dossier_courant
from banc_realite.fournisseurs_contexte import K_VIVIER_FILTRE
from banc_realite.plafond_filtre_or import (
    AVERTISSEMENT,
    ENTETE,
    ecrire_sorties,
    mesurer_plafond,
    pieces_or_question,
    repertoire_plafonds,
    run_plafond,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "banc_realite"


def _hit(text: str, source: str, score: float, rank: int) -> dict:
    return {
        "kind": "file_hit",
        "memory_type": "semantic",
        "id": f"{source}:{rank}",
        "score": score,
        "rank": rank,
        "text": text,
        "source": source,
        "snippet": text[:280],
        "payload": {"metadata": {}},
        "provenance": {"source_file_path": source},
    }


def _gold_et_questions() -> tuple[list[dict], dict]:
    questions = [
        {
            "id": "gl-q001",
            "question": "alpha ?",
            "shape": "single",
            "supporting_assertions": ["a1"],
        },
        {
            "id": "gl-q002",
            "question": "beta ?",
            "shape": "single",
            "supporting_assertions": ["a2", "a3"],
        },
    ]
    gold = {
        "assertion_index": {
            "a1": {
                "id": "a1",
                "content": "alpha",
                "quote": "alpha",
                "source": "documents/a.md",
            },
            "a2": {
                "id": "a2",
                "content": "beta",
                "quote": "beta",
                "source": "b.md",
            },
            "a3": {
                "id": "a3",
                "content": "gamma",
                "quote": "gamma",
                "source": "documents/c.md",
            },
        },
        "assertions": [],
        "disputes": {},
        "situations": {},
        "trajectories": {},
    }
    return questions, gold


def test_pieces_or_depuis_assertions_d_appui() -> None:
    questions, gold = _gold_et_questions()
    assert pieces_or_question(questions[0], gold["assertion_index"]) == ["a.md"]
    assert pieces_or_question(questions[1], gold["assertion_index"]) == ["b.md", "c.md"]
    assert pieces_or_question({"supporting_assertions": []}, gold["assertion_index"]) == []


def test_plafond_restreint_vivier_aux_pieces_or(tmp_path, monkeypatch) -> None:
    questions, gold = _gold_et_questions()
    greenloop = dossier_courant("greenloop")
    monkeypatch.setattr(
        type(greenloop),
        "comparatif_dir",
        property(lambda self, _p=tmp_path: _p),
    )
    viviers = {
        "gl-q001": [
            _hit("alpha dans a", "a.md", 0.9, 1),
            _hit("bruit b", "b.md", 0.8, 2),
            _hit("encore a", "a.md", 0.7, 3),
        ],
        "gl-q002": [
            _hit("beta", "b.md", 0.9, 1),
            _hit("hors", "z.md", 0.8, 2),
            _hit("gamma", "c.md", 0.7, 3),
        ],
    }

    def vivier_fn(question):
        return list(viviers[question["id"]])

    payload = mesurer_plafond(
        dossier=greenloop,
        budgets=[12000, 24000],
        questions=questions,
        gold=gold,
        claims_index={},
        holder_table={},
        vivier_fn=vivier_fn,
    )
    assert payload["lit_or"] is True
    assert payload["ligne_du_comparatif"] is False
    assert payload["k_vivier"] == K_VIVIER_FILTRE
    assert payload["pieces_or_par_question"]["gl-q001"] == ["a.md"]
    assert payload["pieces_or_par_question"]["gl-q002"] == ["b.md", "c.md"]
    assert [run["budget"] for run in payload["result"]["runs"]] == [12000, 24000]
    row_q1 = next(
        row
        for row in payload["result"]["runs"][0]["questions"]
        if row["question_id"] == "gl-q001"
    )
    assert row_q1["evidence_count"] == 2
    assert row_q1["appui_tout_tient"] is True
    assert row_q1["appui_au_moins_une"] is True
    row_q2 = next(
        row
        for row in payload["result"]["runs"][0]["questions"]
        if row["question_id"] == "gl-q002"
    )
    assert row_q2["evidence_count"] == 2
    assert row_q2["appui_tout_tient"] is True
    agg = payload["result"]["runs"][0]["aggregate"]
    assert "par_shape" in agg["appui"]
    assert "deux_cotes_sur_28" in agg["disputes"]

    chemins = ecrire_sorties(payload, dossier=greenloop)
    md = Path(chemins["markdown"])
    js = Path(chemins["json"])
    assert md.parent == tmp_path / "plafonds"
    assert js.parent == tmp_path / "plafonds"
    assert md.name == "PLAFOND-FILTRE-OR.md"
    assert js.name == "plafond-filtre-or.json"
    assert not (tmp_path / "PLAFOND-FILTRE-OR.md").exists()
    assert not (tmp_path / "plafond-filtre-or.json").exists()
    texte = md.read_text(encoding="utf-8")
    assert AVERTISSEMENT in texte
    assert ENTETE in texte
    assert "lit l'or" in texte
    assert "pas une ligne du comparatif" in texte.lower()
    dumped = json.loads(js.read_text(encoding="utf-8"))
    assert dumped["lit_or"] is True
    assert dumped["ligne_du_comparatif"] is False
    assert dumped["avertissement"] == AVERTISSEMENT


def test_run_plafond_injecte_sans_cluster(tmp_path, monkeypatch) -> None:
    questions, gold = _gold_et_questions()
    greenloop = dossier_courant("greenloop")
    monkeypatch.setattr(
        type(greenloop),
        "comparatif_dir",
        property(lambda self, _p=tmp_path: _p),
    )
    payload = run_plafond(
        dossier=greenloop,
        budgets=[12000, 24000],
        questions=questions,
        gold=gold,
        claims_index={},
        holder_table={},
        vivier_fn=lambda q: [_hit("alpha dans a", "a.md", 1.0, 1)]
        if q["id"] == "gl-q001"
        else [_hit("beta", "b.md", 1.0, 1), _hit("gamma", "c.md", 0.9, 2)],
    )
    assert repertoire_plafonds(greenloop) == tmp_path / "plafonds"
    assert (tmp_path / "plafonds" / "PLAFOND-FILTRE-OR.md").is_file()
    assert payload["chemins"]["json"].endswith("plafond-filtre-or.json")


def _imports_du_module(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names.add(mod)
            for alias in node.names:
                names.add(f"{mod}.{alias.name}" if mod else alias.name)
                names.add(alias.name)
    return names


def test_plafond_non_importe_par_fournisseurs_ni_registre() -> None:
    for path in sorted(SCRIPTS.glob("*.py")):
        if path.name == "plafond_filtre_or.py":
            continue
        imported = _imports_du_module(path)
        assert "plafond_filtre_or" not in imported, path.name
        assert not any("plafond_filtre_or" in name for name in imported), path.name
    source_fournisseurs = (SCRIPTS / "fournisseurs_contexte.py").read_text(encoding="utf-8")
    source_mesure = (SCRIPTS / "mesure_comparatif_etage2.py").read_text(encoding="utf-8")
    source_temoin = (SCRIPTS / "temoin_brassage.py").read_text(encoding="utf-8")
    for source in (source_fournisseurs, source_mesure, source_temoin):
        assert "plafond_filtre_or" not in source
