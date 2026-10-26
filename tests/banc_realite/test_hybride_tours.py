# SPDX-License-Identifier: AGPL-3.0-only
"""Hybride tours — k_tour1 et configurations (doubles, sans base)."""

import pytest

from banc_realite.hybride_tours import CONFIGURATIONS, _format_tableau, verifier_prefixe_hybride
from banc_realite.trois_tours import alterner_resultats


def _hit(kind: str, ident: str, piece: str) -> dict:
    if kind == "claim_hit":
        return {
            "kind": kind,
            "id": ident,
            "source": piece,
            "provenance": {"source_file_path": piece},
        }
    return {
        "kind": kind,
        "id": ident,
        "source": piece,
        "provenance": {"source_ref": piece},
    }


def test_k_tour1_8_sur_12_garde_tour1_puis_alternance():
    t1 = [_hit("claim_hit", f"t1-{i}", f"p1.md") for i in range(8)]
    t2 = [_hit("claim_hit", f"t2-{i}", f"p2.md") for i in range(12)]
    t3 = [_hit("claim_hit", f"t3-{i}", f"p3.md") for i in range(12)]
    merged = alterner_resultats([t1, t2, t3], k=12, fusion="alternance", k_tour1=8)
    assert [item["id"] for item in merged[:8]] == [f"t1-{i}" for i in range(8)]
    assert [item["id"] for item in merged[8:]] == ["t2-0", "t3-0", "t2-1", "t3-1"]


def test_k_tour1_egal_k_ignore_tours_suivants():
    t1 = [_hit("claim_hit", f"t1-{i}", f"p1.md") for i in range(12)]
    t2 = [_hit("claim_hit", f"t2-{i}", f"p2.md") for i in range(12)]
    t3 = [_hit("claim_hit", f"t3-{i}", f"p3.md") for i in range(12)]
    merged = alterner_resultats([t1, t2, t3], k=12, fusion="alternance", k_tour1=12)
    assert [item["id"] for item in merged] == [f"t1-{i}" for i in range(12)]


def test_k_tour1_none_identique_a_alternance_actuelle():
    t1 = [_hit("claim_hit", "a", "p1.md"), _hit("claim_hit", "b", "p2.md")]
    t2 = [_hit("claim_hit", "c", "p3.md")]
    t3 = [_hit("claim_hit", "d", "p4.md"), _hit("claim_hit", "e", "p5.md")]
    tours = [t1, t2, t3]
    baseline = alterner_resultats(tours, k=5)
    merged = alterner_resultats(tours, k=5, k_tour1=None)
    assert [item["id"] for item in merged] == [item["id"] for item in baseline]


def test_k_tour1_superieur_a_k_value_error():
    with pytest.raises(ValueError, match="k_tour1"):
        alterner_resultats([[]], k=12, k_tour1=13)


def test_configurations_liste_exacte():
    assert [c["etiquette"] for c in CONFIGURATIONS] == [
        "fine_simple_k12",
        "fine_simple_k16",
        "fine_tours3_k12",
        "hybride_8_plus_4",
        "hybride_12_plus_4",
        "hybride_12_plus_6",
        "hybride_12_plus_8",
        "hybride_12_plus_6_tours2",
    ]
    assert len(CONFIGURATIONS) == 8


def test_tableau_une_ligne_par_configuration():
    results = []
    for config in CONFIGURATIONS:
        results.append(
            {
                **config,
                "disputes": {
                    "deux_cotes_sur_28": 10,
                    "reglage_sur_14": 5,
                    "ecart_sur_14": 5,
                },
                "appui": {
                    "tout_tient_sur_141": 50,
                    "au_moins_une_sur_141": 100,
                },
                "taille": {
                    "elements": {"mediane": 12, "p90": 12, "max": 12},
                    "caracteres": {"p90": 2000, "plafond_8000": 0},
                },
                "reels": {
                    "caracteres_reels": {"p90": 5000, "au_dessus_8000": 2},
                    "passages_tronques": 0,
                },
                "caracteres_reels": {"mediane": 4000, "p90": 5000},
                "tokens_estimes": {"mediane": 1000.0, "p90": 1250.0},
                "tabulaires_portes": 3,
            }
        )
    tableau = _format_tableau(results)
    lines = [line for line in tableau.strip().splitlines() if line and not line.startswith("-")]
    assert len(lines) == len(CONFIGURATIONS) + 1
    for config in CONFIGURATIONS:
        assert config["etiquette"] in tableau


def test_verifier_prefixe_hybride():
    evidence = {
        "fine_simple_k12": {
            "q1": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
            "q2": ["x"] * 12,
        },
        "hybride_12_plus_4": {
            "q1": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"],
            "q2": ["x"] * 16,
        },
    }
    assert verifier_prefixe_hybride(evidence) is True
    evidence["hybride_12_plus_4"]["q1"][0] = "z"
    assert verifier_prefixe_hybride(evidence) is False
