# SPDX-License-Identifier: AGPL-3.0-only
"""Brain banc g24/g36 — configurations budget-étendues (issue #127)."""

from __future__ import annotations

from banc_realite import etage3_config_gagnante as ecg
from banc_realite.dossiers import dossier_courant
from banc_realite.fournisseurs_contexte import BrainBanc, BrainBancG24, BrainBancG36
from banc_realite.mesure_comparatif_etage2 import K_FIXE, _instancier, k_effectif_fournisseur


def _comptes_config(config: dict) -> dict:
    return {
        "hybride_k": config["hybride"]["k"],
        "hybride_k_tour1": config["hybride"]["k_tour1"],
        "brute_k": config["brute_k"],
        "max_phrases": config["phrases"]["max_phrases"],
    }


def test_hybride_fenetres_g12_inchange():
    assert ecg.PORTE_PHRASE_HYBRIDE_FENETRES_G12 == {
        "etiquette": "hyb_vec_f2_b12_g12",
        "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
        "brute_k": 12,
        "phrases": {
            "n_par_passage": 2,
            "max_phrases": 12,
            "mode": "vectoriel",
            "selection": "globale",
            "fenetre": 2,
        },
    }


def test_configurations_g24_g36_existent_et_differencent_g12_par_comptes():
    origine = _comptes_config(ecg.PORTE_PHRASE_HYBRIDE_FENETRES_G12)
    g24 = _comptes_config(ecg.PORTE_PHRASE_HYBRIDE_FENETRES_G24)
    g36 = _comptes_config(ecg.PORTE_PHRASE_HYBRIDE_FENETRES_G36)

    assert g24 == {
        "hybride_k": 36,
        "hybride_k_tour1": 24,
        "brute_k": 24,
        "max_phrases": 24,
    }
    assert g36 == {
        "hybride_k": 54,
        "hybride_k_tour1": 36,
        "brute_k": 36,
        "max_phrases": 36,
    }

    for key in ("hybride_k", "hybride_k_tour1", "brute_k", "max_phrases"):
        assert origine[key] < g24[key] < g36[key]

    g12_phrases = ecg.PORTE_PHRASE_HYBRIDE_FENETRES_G12["phrases"]
    for variant in (ecg.PORTE_PHRASE_HYBRIDE_FENETRES_G24, ecg.PORTE_PHRASE_HYBRIDE_FENETRES_G36):
        phrases = variant["phrases"]
        assert phrases["n_par_passage"] == g12_phrases["n_par_passage"]
        assert phrases["mode"] == g12_phrases["mode"]
        assert phrases["selection"] == g12_phrases["selection"]
        assert phrases["fenetre"] == g12_phrases["fenetre"]

    assert ecg.CONFIGURATIONS["hybride_fenetres_g24"]["porte_phrase_config"] is ecg.PORTE_PHRASE_HYBRIDE_FENETRES_G24
    assert ecg.CONFIGURATIONS["hybride_fenetres_g36"]["porte_phrase_config"] is ecg.PORTE_PHRASE_HYBRIDE_FENETRES_G36


def test_fournisseurs_utilisent_configurations_attendues():
    moreau = dossier_courant("moreau")
    origine = BrainBanc({}, dossier=moreau)
    g24 = BrainBancG24({}, dossier=moreau)
    g36 = BrainBancG36({}, dossier=moreau)

    assert origine._config is ecg.CONFIGURATIONS["hybride_fenetres_g12"]["porte_phrase_config"]
    assert g24._config is ecg.CONFIGURATIONS["hybride_fenetres_g24"]["porte_phrase_config"]
    assert g36._config is ecg.CONFIGURATIONS["hybride_fenetres_g36"]["porte_phrase_config"]


def test_instancier_brain_banc_g24_g36():
    moreau = dossier_courant("moreau")
    g24 = _instancier("brain_banc_g24", {}, dossier=moreau)
    g36 = _instancier("brain_banc_g36", {}, dossier=moreau)
    assert g24.etiquette == "brain_banc_g24"
    assert g36.etiquette == "brain_banc_g36"


def test_k_fixe_et_effectif_par_variante():
    assert "brain_banc_g24" in K_FIXE
    assert "brain_banc_g36" in K_FIXE
    assert k_effectif_fournisseur("brain_banc") == 12
    assert k_effectif_fournisseur("brain_banc_g24") == 24
    assert k_effectif_fournisseur("brain_banc_g36") == 36
