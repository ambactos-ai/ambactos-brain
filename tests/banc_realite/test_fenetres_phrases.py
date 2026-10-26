# SPDX-License-Identifier: AGPL-3.0-only
"""Fenêtres de deux phrases — découpe, sélection et configurations."""

from __future__ import annotations

from banc_realite.mesure_porte_phrase import CONFIGURATIONS
from banc_realite.porte_phrase import (
    assertions_tenues_par_une_unite,
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


def _file_hit(ident: str, text: str, *, piece: str = "contract.md") -> dict:
    return {
        "kind": "file_hit",
        "id": ident,
        "text": text,
        "source": piece,
        "score": 0.9,
        "provenance": {"source_file_path": piece},
    }


def test_fenetre2_quatre_phrases_trois_unites_glissantes():
    prose = (
        "Alpha sentence one with enough characters here. "
        "Beta sentence two with enough characters here. "
        "Gamma sentence three with enough characters here. "
        "Delta sentence four with enough characters here."
    )
    units = phrases_du_passage(prose, fenetre=2)
    assert len(units) == 3
    assert "Alpha sentence one" in units[0] and "Beta sentence two" in units[0]
    assert "Beta sentence two" in units[1] and "Gamma sentence three" in units[1]
    assert "Gamma sentence three" in units[2] and "Delta sentence four" in units[2]


def test_fenetre2_une_phrase_une_unite():
    prose = "Single sentence with enough characters for one sliding window unit."
    units = phrases_du_passage(prose, fenetre=2)
    assert len(units) == 1
    assert "Single sentence" in units[0]


def test_fenetre2_tableau_et_puces_inchanges():
    units_f1 = phrases_du_passage(PASSAGE_MIXTE, fenetre=1)
    units_f2 = phrases_du_passage(PASSAGE_MIXTE, fenetre=2)
    table_units_f1 = [u for u in units_f1 if "Service" in u and "Duration" in u]
    table_units_f2 = [u for u in units_f2 if "Service" in u and "Duration" in u]
    assert table_units_f1 == table_units_f2
    bullets_f1 = [u for u in units_f1 if u.startswith("- First bullet")]
    bullets_f2 = [u for u in units_f2 if u.startswith("- First bullet")]
    assert bullets_f1 == bullets_f2


def test_fenetre2_borne_450_sans_couper_mot():
    long_word = "y" * 430
    unit = phrases_du_passage(
        f"First sentence with enough characters. Second sentence contains {long_word} inside.",
        fenetre=2,
    )
    assert len(unit) == 1
    assert unit[0].endswith(" …")
    assert not unit[0].endswith("y …")
    assert len(unit[0]) <= 453


def test_fenetre1_identique_jeu_111():
    assert phrases_du_passage(PASSAGE_MIXTE, fenetre=1) == phrases_du_passage(PASSAGE_MIXTE)


def test_porte_phrase_retire_chevauchement_fenetre2():
    passage = (
        "Alpha beta gamma delta epsilon zeta eta theta iota. "
        "Beta gamma delta epsilon zeta eta theta iota kappa. "
        "Gamma delta epsilon zeta eta theta iota kappa lambda. "
        "Delta epsilon zeta eta theta iota kappa lambda mu."
    )
    evidence = [_file_hit("p1", passage)]
    question = "beta gamma delta"
    hits, _stats = porte_phrase(
        question,
        evidence,
        fenetre=2,
        n_par_passage=3,
        max_phrases=3,
        mode="lexical",
    )
    assert len(hits) <= 2
    texts = [hit["text"] for hit in hits]
    assert not any(texts[0] in texts[1] and texts[0] != texts[1] for _ in [0]) if len(texts) == 2 else True


def test_assertions_tenues_par_une_unite_fenetre2():
    piece = (
        "The prior year's activity is rebuilt from monthly invoice transcriptions. "
        "It contains no daily visit dates or per-visit masses for that year."
    )
    assertion = {
        "id": "a1",
        "source": "doc.md",
        "quote": (
            "The prior year's activity is rebuilt from monthly invoice transcriptions. "
            "It contains no daily visit dates or per-visit masses for that year."
        ),
    }
    pieces_text = {"doc.md": piece}
    held_1 = assertions_tenues_par_une_unite([assertion], pieces_text, fenetre=1)
    held_2 = assertions_tenues_par_une_unite([assertion], pieces_text, fenetre=2)
    needle = next(iter(held_1))
    assert held_1[needle] is False
    assert held_2[needle] is True


def test_configurations_liste_exacte_fenetres():
    assert [c["etiquette"] for c in CONFIGURATIONS] == [
        "hybride_12_6",
        "hyb_vec_b8_n2",
        "hyb_vec_f2_b8_n2",
        "hyb_vec_f2_b8_n1",
        "hyb_vec_f2_b12_g8",
        "hyb_vec_f2_b12_g12",
        "hyb_lex_f2_b8_n2",
        "phrases_vec_f2_b12_n2",
    ]
