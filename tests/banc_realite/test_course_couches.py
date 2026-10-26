# SPDX-License-Identifier: AGPL-3.0-only
"""Course des couches — configurations et tableau (doubles, sans base)."""

from banc_realite.course_couches import CONFIGURATIONS, _format_tableau


def test_configurations_liste_exacte():
    assert [c["etiquette"] for c in CONFIGURATIONS] == [
        "fine_tours2_k12",
        "fine_tours3_k12",
        "fine_tours4_k12",
        "fine_tours5_k12",
        "brute_temoin_k12",
        "brute_concat_k12",
        "brute_tours3_k12",
        "brute_tours5_k12",
        "brute_tours3_k20",
        "brute_fine_tours3_k12",
        "brute_fine_tours3_k20",
    ]
    assert len(CONFIGURATIONS) == 11


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
            }
        )
    tableau = _format_tableau(results)
    lines = [line for line in tableau.strip().splitlines() if line and not line.startswith("-")]
    assert len(lines) == len(CONFIGURATIONS) + 1
    for config in CONFIGURATIONS:
        assert config["etiquette"] in tableau
