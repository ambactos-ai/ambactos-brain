# SPDX-License-Identifier: AGPL-3.0-only
"""Prix couche brute — présence sur rendu et configurations (doubles, sans base)."""

from unittest.mock import patch

from banc_realite.controle_budget_egal import _compile_question_row
from banc_realite.prix_couche_brute import CONFIGURATIONS, _format_tableau


def _fake_evidence(n: int) -> list[dict]:
    return [
        {
            "kind": "claim_hit",
            "text": f"élément {index}",
            "source": f"doc-{index}.md",
            "claim_id": f"claim-{index}",
        }
        for index in range(n)
    ]


def test_presence_sur_rendu_exclut_elements_ecartes():
    question = {
        "id": "gl-q001",
        "question": "test question",
        "shape": "single",
        "supporting_assertions": ["a1", "a2", "a3"],
    }
    evidence = _fake_evidence(3)
    appui_calls: list[list] = []

    def fake_appui(_question, ev, **_kwargs):
        appui_calls.append(list(ev))
        return {"tient": [aid for aid in ("a1", "a2", "a3") if aid in {f"a{i + 1}" for i, _ in enumerate(ev)}]}

    with (
        patch(
            "banc_realite.controle_budget_egal.build_evidence_for_mode",
            return_value=(evidence, {}),
        ),
        patch(
            "banc_realite.controle_budget_egal._render_markdown_with_budget",
            return_value=("rendu", 1),
        ),
        patch("banc_realite.controle_budget_egal._render_markdown", return_value="sans budget"),
        patch("banc_realite.controle_budget_egal._appui_summary", side_effect=fake_appui),
    ):
        row = _compile_question_row(
            question,
            connection=None,
            embed_cache={"test question": [0.0]},
            gold={"assertion_index": {}},
            claims_index={},
            assertion_index={},
            k=8,
            mode="temoin",
            budget=None,
            presence_sur_rendu=True,
        )

    assert len(appui_calls) == 1
    assert len(appui_calls[0]) == 2
    assert row["elements_ecartes"] == 1
    assert row["elements_rendus"] == 2
    assert row["appui_tout_tient"] is False


def test_presence_sur_rendu_false_compte_toute_evidence():
    question = {
        "id": "gl-q002",
        "question": "test question",
        "shape": "single",
        "supporting_assertions": ["a1", "a2", "a3"],
    }
    evidence = _fake_evidence(3)
    appui_calls: list[list] = []

    def fake_appui(_question, ev, **_kwargs):
        appui_calls.append(list(ev))
        return {"tient": ["a1", "a2", "a3"]}

    with (
        patch(
            "banc_realite.controle_budget_egal.build_evidence_for_mode",
            return_value=(evidence, {}),
        ),
        patch("banc_realite.controle_budget_egal._render_markdown", return_value="rendu"),
        patch("banc_realite.controle_budget_egal._appui_summary", side_effect=fake_appui),
    ):
        row = _compile_question_row(
            question,
            connection=None,
            embed_cache={"test question": [0.0]},
            gold={"assertion_index": {}},
            claims_index={},
            assertion_index={},
            k=8,
            mode="temoin",
            budget=None,
            presence_sur_rendu=False,
        )

    assert len(appui_calls) == 1
    assert len(appui_calls[0]) == 3
    assert "elements_ecartes" not in row
    assert "elements_rendus" not in row
    assert row["appui_tout_tient"] is True


def test_configurations_liste_exacte():
    assert [c["etiquette"] for c in CONFIGURATIONS] == [
        "fine_k12_tours3",
        "brute_k4",
        "brute_k6",
        "brute_k8",
        "brute_k12",
        "brute_k16",
        "brute_k8_b8000",
        "brute_k12_b8000",
        "brute_k16_b8000",
    ]
    assert len(CONFIGURATIONS) == 9
    assert CONFIGURATIONS[0] == {
        "etiquette": "fine_k12_tours3",
        "couche": "fine",
        "k": 12,
        "mode": "trois_tours",
        "n_tours": 3,
        "fusion": "alternance",
        "max_chars": 64000,
        "presence_sur_rendu": False,
    }
    assert CONFIGURATIONS[-1]["presence_sur_rendu"] is True
    assert CONFIGURATIONS[-1]["max_chars"] == 8000


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
                "rendu": {
                    "elements_rendus": {"mediane": 8, "max": 12},
                    "caracteres_rendus": {"mediane": 4000, "p90": 7500},
                    "tokens_estimes_mediane": 1000,
                    "elements_ecartes_total": 3,
                },
            }
        )
    tableau = _format_tableau(results)
    lines = [line for line in tableau.strip().splitlines() if line and not line.startswith("-")]
    assert len(lines) == len(CONFIGURATIONS) + 1
    for config in CONFIGURATIONS:
        assert config["etiquette"] in tableau
