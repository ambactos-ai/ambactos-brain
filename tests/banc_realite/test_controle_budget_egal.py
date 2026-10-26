# SPDX-License-Identifier: AGPL-3.0-only
"""Contrôle à budget d'éléments égal — agrégation et tableau (doubles, sans base)."""

from banc_realite.controle_budget_egal import (
    CONFIGURATIONS,
    _aggregate_config_rows,
    _format_tableau,
)


def _row(
    qid: str,
    *,
    shape: str = "single",
    evidence_count: int = 8,
    caracteres: int = 1000,
    deux_cotes: bool = False,
    tout_tient: bool = True,
    au_moins_une: bool = True,
) -> dict:
    row = {
        "question_id": qid,
        "shape": shape,
        "evidence_count": evidence_count,
        "caracteres": caracteres,
        "plafond_caracteres": caracteres >= 8000,
        "appui_tout_tient": tout_tient,
        "appui_au_moins_une": au_moins_une,
    }
    if shape == "dispute":
        row["deux_cotes_dans_contexte"] = deux_cotes
    return row


def test_agregation_separe_reglage_et_ecart():
    reglage_ids = {"gl-q084", "gl-q086"}
    ecart_ids = {"gl-q096", "gl-q091"}
    dispute_rows = [
        _row("gl-q084", shape="dispute", deux_cotes=True, tout_tient=False, au_moins_une=True),
        _row("gl-q086", shape="dispute", deux_cotes=False, tout_tient=False, au_moins_une=False),
        _row("gl-q096", shape="dispute", deux_cotes=True, tout_tient=True, au_moins_une=True),
        _row("gl-q091", shape="dispute", deux_cotes=False, tout_tient=False, au_moins_une=True),
    ]
    all_rows = dispute_rows + [
        _row("gl-q001", shape="single", tout_tient=True, au_moins_une=True),
        _row("gl-q002", shape="single", tout_tient=False, au_moins_une=True),
    ]
    agg = _aggregate_config_rows(
        all_rows,
        dispute_rows,
        reglage_ids=reglage_ids,
        ecart_ids=ecart_ids,
    )
    assert agg["disputes"]["deux_cotes_sur_28"] == 2
    assert agg["disputes"]["reglage_sur_14"] == 1
    assert agg["disputes"]["ecart_sur_14"] == 1
    assert agg["appui"]["tout_tient_sur_141"] == 2
    assert agg["appui"]["au_moins_une_sur_141"] == 5
    assert agg["appui"]["par_shape"]["single"]["tout_tient"] == 1
    assert agg["appui"]["par_shape"]["single"]["au_moins_une"] == 2
    assert agg["appui"]["par_shape"]["dispute"]["tout_tient"] == 1
    assert agg["appui"]["par_shape"]["dispute"]["au_moins_une"] == 3


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
                    "elements": {"mediane": 8, "p90": 12, "max": 16},
                    "caracteres": {"mediane": 1000, "p90": 2000, "max": 3000, "plafond_8000": 0},
                },
            }
        )
    tableau = _format_tableau(results)
    lines = [line for line in tableau.strip().splitlines() if line and not line.startswith("-")]
    assert len(lines) == len(CONFIGURATIONS) + 1
    for config in CONFIGURATIONS:
        assert config["etiquette"] in tableau


def test_configurations_liste_exacte():
    assert [c["etiquette"] for c in CONFIGURATIONS] == [
        "temoin_k8",
        "temoin_k16",
        "tours_k8",
        "tours_k12",
        "tours_k16",
        "tours_k20",
        "tours_k24",
        "gagnante_k8_b16",
        "k12_b16",
    ]
    assert [(c["k"], c["mode"], c["budget"]) for c in CONFIGURATIONS] == [
        (8, "temoin", None),
        (16, "temoin", None),
        (8, "trois_tours", None),
        (12, "trois_tours", None),
        (16, "trois_tours", None),
        (20, "trois_tours", None),
        (24, "trois_tours", None),
        (8, "trois_tours_completion", 16),
        (12, "trois_tours_completion", 16),
    ]
