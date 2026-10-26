# SPDX-License-Identifier: AGPL-3.0-only
"""Porte phrase — fonctions pures et métriques (doubles, sans base)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from banc_realite.mesure_porte_phrase import CONFIGURATIONS, _format_tableau
from banc_realite.porte_phrase import (
    compter_clauses_portees,
    compter_perte_selection,
    liste_clauses_sans_chiffre,
    phrases_du_passage,
    porte_phrase,
    score_phrase,
)


PASSAGE_MIXTE = """\
The customer remains responsible for contents until collection.
Only WT-series tickets are included; other forms are excluded.

| Service | Duration |
| --- | --- |
| Basic plan | 12 months |

- First bullet point with enough characters here.
- Short one.
* Another bullet with sufficient length for the test case.
1. Numbered bullet that should also be extracted as its own unit.
This restriction lasts for seven years after the service ends; extensions apply.
"""


def _file_hit(
    ident: str,
    text: str,
    *,
    piece: str = "contract.md",
    score: float = 0.9,
) -> dict:
    return {
        "kind": "file_hit",
        "id": ident,
        "text": text,
        "source": piece,
        "score": score,
        "provenance": {"source_file_path": piece},
    }


def test_phrases_du_passage_prose_tableau_puces():
    units = phrases_du_passage(PASSAGE_MIXTE)
    assert any("customer remains responsible" in unit for unit in units)
    assert any("restriction lasts for seven years" in unit for unit in units)
    assert any("Service" in unit and "Duration" in unit for unit in units)
    assert any(unit.startswith("- First bullet") for unit in units)
    assert any(unit.startswith("* Another bullet") for unit in units)
    assert any(unit.startswith("1. Numbered bullet") for unit in units)
    assert all("Short one." not in unit for unit in units)


def test_phrases_du_passage_borne_sans_couper_mot():
    long_word = "x" * 280
    unit = phrases_du_passage(f"This sentence contains {long_word} inside the passage.")
    assert len(unit) == 1
    assert unit[0].endswith(" …")
    assert not unit[0].endswith("x …")


def test_score_phrase_avec_et_sans_nombre():
    question = "How many months does the basic plan last for 12 customers?"
    with_number = "The basic plan lasts for 12 months in total for all users."
    without_number = "The basic plan lasts for many months in total for all users."
    assert score_phrase(question, with_number) > score_phrase(question, without_number)


def test_porte_phrase_ordre_limites_et_provenance():
    question = "customer responsible contents collection WT tickets"
    evidence = [
        _file_hit("p1", PASSAGE_MIXTE, piece="a.md", score=0.9),
        _file_hit("p2", "Second passage about WT-series ticket forms included in packet.", piece="b.md", score=0.5),
    ]
    out, stats = porte_phrase(question, evidence, n_par_passage=2, max_phrases=3)
    assert stats["unites_sans_vecteur"] == 0
    assert len(out) == 3
    assert all(item["kind"] == "phrase_hit" for item in out)
    assert out[0]["id"].startswith("phrase:p1:")
    assert out[0]["source"] == "a.md"
    assert out[0]["provenance"] == {"source_file_path": "a.md", "passage_id": "p1"}
    assert out[0]["score"] == score_phrase(question, out[0]["text"])
    ranks = [item["id"].split(":")[1] for item in out]
    assert ranks.index("p1") < ranks.index("p2")


def test_porte_phrase_score_zero_seulement_si_rien_de_mieux():
    evidence = [_file_hit("p1", "Alpha beta gamma delta epsilon zeta eta theta.", piece="x.md")]
    out, _stats = porte_phrase("quantum physics", evidence, n_par_passage=2, max_phrases=2)
    assert len(out) == 1


def test_compter_clauses_portees_et_perte_selection(monkeypatch, tmp_path):
    diagnostic = {
        "lignes": [
            {"question_id": "q1", "assertion_id": "a1", "verdict": "extrait_infidele"},
            {"question_id": "q1", "assertion_id": "a2", "verdict": "extrait_infidele"},
            {"question_id": "q2", "assertion_id": "a3", "verdict": "portee"},
        ]
    }
    path = tmp_path / "diag.json"
    path.write_text(json.dumps(diagnostic), encoding="utf-8")
    gold = {
        "assertion_index": {
            "a1": {"id": "a1", "quote": "customer remains responsible", "source": "doc.md"},
            "a2": {"id": "a2", "quote": "seven years after service ends", "source": "doc.md"},
            "a3": {"id": "a3", "quote": "amount 100 euros", "source": "doc.md"},
        }
    }
    clauses = liste_clauses_sans_chiffre(gold=gold, diagnostic_path=path)
    assert len(clauses) == 2

    claims_index = {}
    assertion_index = gold["assertion_index"]

    def fake_collect(assertion, evidence, _claims_index):
        text_blob = " ".join(item.get("text") or "" for item in evidence)
        needle = assertion["quote"]
        if needle in text_blob:
            return {"tient": [{"evidence_id": "x"}], "candidat": []}
        return {"tient": [], "candidat": []}

    monkeypatch.setattr("banc_realite.porte_phrase.collect_appui_states", fake_collect)

    evidence_phrases = [{"kind": "phrase_hit", "text": "customer remains responsible", "source": "doc.md"}]
    assert (
        compter_clauses_portees(
            "q1",
            evidence_phrases,
            clauses=clauses,
            assertion_index=assertion_index,
            claims_index=claims_index,
        )
        == 1
    )

    evidence_passages = [
        {"kind": "file_hit", "text": "customer remains responsible for contents", "source": "doc.md"},
        {"kind": "file_hit", "text": "seven years after service ends for everyone", "source": "doc.md"},
    ]
    question = {"supporting_assertions": ["a1", "a2"]}
    assert (
        compter_perte_selection(
            question,
            evidence_passages,
            evidence_phrases,
            assertion_index=assertion_index,
            claims_index=claims_index,
        )
        == 1
    )


def test_tableau_inclut_clauses_et_perte():
    results = []
    for config in CONFIGURATIONS:
        results.append(
            {
                **config,
                "disputes": {"deux_cotes_sur_28": 10, "reglage_sur_14": 5, "ecart_sur_14": 5},
                "appui": {"tout_tient_sur_141": 50, "au_moins_une_sur_141": 100},
                "taille": {
                    "elements": {"mediane": 12, "p90": 12, "max": 12},
                    "caracteres": {"p90": 2000, "plafond_8000": 0},
                },
                "reels": {"caracteres_reels": {"p90": 5000, "au_dessus_8000": 2}, "passages_tronques": 0},
                "caracteres_reels": {"mediane": 4000, "p90": 5000},
                "tokens_estimes": {"mediane": 1000.0, "p90": 1250.0},
                "tabulaires_portes": 3,
                "clauses_portees": 4,
                "clauses_total": 59,
                "perte_selection": 7 if config.get("phrases") else None,
                "unites_sans_vecteur": 0,
                "secondes_par_question": 1.5,
                "au_dessus_12000": 1,
                "questions": [{"au_dessus_12000": False}],
            }
        )
    tableau = _format_tableau(results)
    assert "clauses/59" in tableau
    assert ">12000" in tableau
    assert "perte_sel" in tableau
    assert "sans_vec" in tableau
