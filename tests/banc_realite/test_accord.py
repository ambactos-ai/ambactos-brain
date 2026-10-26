# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path

import pytest

from banc_realite import accord as acc


def test_cohens_kappa_known_values():
    labels = ["a", "b"]
    perfect = {"a": {"a": 5, "b": 0}, "b": {"a": 0, "b": 5}}
    assert acc.cohens_kappa(perfect, labels) == 1.0
    independent = {"a": {"a": 3, "b": 3}, "b": {"a": 3, "b": 3}}
    assert acc.cohens_kappa(independent, labels) == 0.0
    mixed = {"a": {"a": 4, "b": 1}, "b": {"a": 1, "b": 4}}
    assert round(acc.cohens_kappa(mixed, labels), 2) == 0.6


def _balanced_relecture_verdicts(partie: str) -> tuple[dict[str, list], dict[str, list]]:
    """Quatre quadrants équilibrés → kappa binaire ≈ 0 par partie."""
    lecteur_a: dict[str, list] = {}
    lecteur_b: dict[str, list] = {}
    if partie == "residu":
        prefix, pairs = "R", [("1", "1"), ("1", "3"), ("3", "1"), ("3", "3")]
    else:
        prefix, pairs = "H", [("V", "V"), ("V", "4"), ("4", "V"), ("4", "4")]
    idx = 1
    for a_verdict, b_verdict in pairs:
        for _ in range(10):
            ident = f"{prefix}{idx:02d}"
            lecteur_a[ident] = [a_verdict, ""]
            lecteur_b[ident] = [b_verdict, ""]
            idx += 1
    return lecteur_a, lecteur_b


def test_kappa_calcule_par_partie_sans_pool(tmp_path):
    residu_a, residu_b = _balanced_relecture_verdicts("residu")
    hors_a, hors_b = _balanced_relecture_verdicts("hors_or")
    lecteur_a = {"residu": residu_a, "hors_or": hors_a}
    lecteur_b = {"residu": residu_b, "hors_or": hors_b, "complet": True}
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps(lecteur_a), encoding="utf-8")
    b_path.write_text(json.dumps(lecteur_b), encoding="utf-8")
    payload = acc.compute_accord(
        lecteur_a_relecture_path=a_path,
        lecteur_b_relecture_path=b_path,
        lecteur_a_assertions_path=tmp_path / "missing-a.json",
        lecteur_b_assertions_path=tmp_path / "missing-b.json",
    )
    relecture = payload["relecture"]
    assert relecture["disponible"] is True
    assert "kappa" not in relecture
    assert "accord_brut" not in relecture
    assert abs(relecture["residu"]["binaire"]["kappa"]) < 0.01
    assert abs(relecture["hors_or"]["binaire"]["kappa"]) < 0.01


def test_refuse_incomplete_b(tmp_path):
    lecteur_a = {
        "residu": {"R01": [1, ""]},
        "hors_or": {"H01": ["V", ""]},
    }
    lecteur_b = {"residu": {"R01": [1, ""]}, "hors_or": {}}
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps(lecteur_a), encoding="utf-8")
    b_path.write_text(json.dumps(lecteur_b), encoding="utf-8")
    payload = acc.compute_accord(
        lecteur_a_relecture_path=a_path,
        lecteur_b_relecture_path=b_path,
        lecteur_a_assertions_path=tmp_path / "missing-a.json",
        lecteur_b_assertions_path=tmp_path / "missing-b.json",
    )
    assert payload["relecture"]["disponible"] is False


def test_refuse_relecture_b_incomplete_via_complet(tmp_path):
    lecteur_a = {
        "residu": {f"R{idx:02d}": [1, ""] for idx in range(1, 41)},
        "hors_or": {f"H{idx:02d}": ["V", ""] for idx in range(1, 41)},
    }
    lecteur_b = {
        "residu": {"R01": [1, ""]},
        "hors_or": {},
        "complet": False,
        "manquants": ["R02", "H01"],
    }
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps(lecteur_a), encoding="utf-8")
    b_path.write_text(json.dumps(lecteur_b), encoding="utf-8")
    payload = acc.compute_accord(
        lecteur_a_relecture_path=a_path,
        lecteur_b_relecture_path=b_path,
        lecteur_a_assertions_path=tmp_path / "missing-a.json",
        lecteur_b_assertions_path=tmp_path / "missing-b.json",
    )
    assert payload["relecture"]["disponible"] is False


def test_shuffle_key_is_stable():
    desaccords = [
        {
            "id": "R01",
            "partie": "residu",
            "lecteur_a": "1",
            "lecteur_b": "4",
            "justification_a": "a",
            "justification_b": "b",
        },
        {
            "id": "H01",
            "partie": "hors_or",
            "lecteur_a": "V",
            "lecteur_b": "5",
            "justification_a": "c",
            "justification_b": "d",
        },
    ]
    cards1, cle1 = acc.shuffle_lecture_order(desaccords)
    cards2, cle2 = acc.shuffle_lecture_order(desaccords)
    assert cle1 == cle2
    assert {card["card_id"] for card in cards1} == {"d01", "d02"}


def test_page_without_disagreement(tmp_path):
    donnees = {
        "residu": [
            {
                "content": "c",
                "quote": "q",
                "extrait_source": "e",
                "voisins": [],
            }
        ]
        + [
            {
                "content": "",
                "quote": "",
                "extrait_source": "",
                "voisins": [],
            }
        ]
        * 39,
        "hors_or": [
            {
                "triplet": "fait",
                "texte": "fait",
                "extrait_source": "e",
            }
        ]
        * 40,
    }
    donnees_path = tmp_path / "donnees.json"
    donnees_path.write_text(json.dumps(donnees), encoding="utf-8")
    lecteur_a = {
        "residu": {f"R{idx:02d}": [1, ""] for idx in range(1, 41)},
        "hors_or": {f"H{idx:02d}": ["V", ""] for idx in range(1, 41)},
    }
    lecteur_b = {
        "residu": {f"R{idx:02d}": [1, ""] for idx in range(1, 41)},
        "hors_or": {f"H{idx:02d}": ["V", ""] for idx in range(1, 41)},
    }
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps(lecteur_a), encoding="utf-8")
    b_path.write_text(json.dumps(lecteur_b), encoding="utf-8")
    accord_path = tmp_path / "accord.json"
    page_path = tmp_path / "page.html"
    cle_path = tmp_path / "cle.json"
    acc.write_accord_outputs(
        accord_path=accord_path,
        cle_path=cle_path,
        page_path=page_path,
        donnees_path=donnees_path,
        lecteur_a_relecture_path=a_path,
        lecteur_b_relecture_path=b_path,
        lecteur_a_assertions_path=tmp_path / "na.json",
        lecteur_b_assertions_path=tmp_path / "nb.json",
        write_page=True,
    )
    html = page_path.read_text(encoding="utf-8")
    assert "Aucun désaccord" in html
    assert 'id="d01"' not in html


def test_page_with_fake_b_verdicts(tmp_path):
    donnees = {
        "residu": [
            {
                "content": "contenu or",
                "quote": "citation",
                "extrait_source": "→    1  extrait",
                "voisins": [{"triplet": "voisin un", "texte": "voisin un"}],
            }
        ]
        + [
            {
                "content": "",
                "quote": "",
                "extrait_source": "",
                "voisins": [],
            }
        ]
        * 39,
        "hors_or": [
            {
                "triplet": "fait",
                "texte": "fait",
                "extrait_source": "extrait",
            }
        ]
        * 40,
    }
    donnees_path = tmp_path / "donnees.json"
    donnees_path.write_text(json.dumps(donnees), encoding="utf-8")
    lecteur_a = {
        "residu": {f"R{idx:02d}": [1, ""] for idx in range(1, 41)},
        "hors_or": {f"H{idx:02d}": ["V", ""] for idx in range(1, 41)},
    }
    lecteur_a["residu"]["R01"] = [1, "lecture A"]
    lecteur_b = {
        "residu": {f"R{idx:02d}": [1, ""] for idx in range(1, 41)},
        "hors_or": {f"H{idx:02d}": ["V", ""] for idx in range(1, 41)},
    }
    lecteur_b["residu"]["R01"] = [4, "lecture B"]
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps(lecteur_a), encoding="utf-8")
    b_path.write_text(json.dumps(lecteur_b), encoding="utf-8")
    page_path = tmp_path / "page.html"
    acc.write_accord_outputs(
        accord_path=tmp_path / "accord.json",
        cle_path=tmp_path / "cle.json",
        page_path=page_path,
        donnees_path=donnees_path,
        lecteur_a_relecture_path=a_path,
        lecteur_b_relecture_path=b_path,
        lecteur_a_assertions_path=tmp_path / "na.json",
        lecteur_b_assertions_path=tmp_path / "nb.json",
        write_page=True,
    )
    html = page_path.read_text(encoding="utf-8")
    assert 'id="d01"' in html
    assert "contenu or" in html
    assert "voisin un" in html
    assert "Lecture 1" in html
    assert "localStorage" in html
