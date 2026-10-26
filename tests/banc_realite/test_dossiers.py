# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — descripteurs de dossier banc comparatif."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from banc_realite import config
from banc_realite.dossiers import Dossier, compter_or, dossier_courant
from banc_realite.or_dossier import load_gold, verifier_or_coherent
from banc_realite.recherche_entites import chemin_reconstitution


def test_dossier_courant_defaut_greenloop():
    dossier = dossier_courant()
    assert dossier.nom == "greenloop"
    assert dossier.gold_dir == config.GOLD
    assert dossier.corpus_dir == config.CORPUS
    assert dossier.area_root == config.AREA
    assert dossier.vault_ref == config.VAULT_REF
    assert dossier.instance == config.INSTANCE
    assert dossier.smoke_question_id == config.SMOKE_QUESTION_ID


def test_dossier_courant_moreau():
    dossier = dossier_courant("moreau")
    assert dossier.nom == "moreau"
    assert dossier.gold_dir == config.BANC_ROOT / "dossiers" / "moreau" / "gold"
    assert dossier.corpus_dir == config.BANC_ROOT / "dossiers" / "moreau" / "documents"
    assert dossier.area_root == config.BANC_ROOT / "dossiers" / "moreau" / "aire"
    assert dossier.responsibility_context == "MOREAU"
    assert dossier.nature == "juridique"
    assert dossier.prefixe_question == "MOR-Q"
    assert dossier.mode_presence_principal == "verbatim"










def test_dossier_inconnu_leve_erreur_explicite():
    with pytest.raises(ValueError, match="dossier inconnu"):
        dossier_courant("inexistant")


def test_dossier_est_immuable():
    dossier = dossier_courant("greenloop")
    with pytest.raises(FrozenInstanceError):
        dossier.nom = "autre"  # type: ignore[misc]


@pytest.mark.skipif(
    not (config.BANC_ROOT / "dossiers" / "moreau" / "gold" / "questions.json").is_file(),
    reason="or MOREAU absent sur disque (~/.agents/var/banc-realite/dossiers/moreau/gold)",
)
def test_moreau_or_chargeable_par_descripteur():
    dossier = dossier_courant("moreau")
    counts = compter_or(dossier)
    assert counts == {"questions": 50, "assertions": 42, "pieces": 12}
