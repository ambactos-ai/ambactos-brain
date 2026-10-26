# SPDX-License-Identifier: AGPL-3.0-only
"""Les quatre lignes « filtre puis RAG » s'ajoutent après les existantes, sans les modifier."""
from __future__ import annotations

from pathlib import Path

import pytest

from banc_realite.mesure_comparatif_etage2 import (
    BRAIN_FOURNISSEURS,
    EXPECTED_VERBATIM,
    FOURNISSEUR_LABELS,
    K_EFFECTIF_FOURNISSEUR,
    K_FIXE,
    K_RAG_LARGE,
    K_VIVIER_FILTRE,
    REFERENCE_LINES,
    _fournisseur_label,
    _instancier,
)
from banc_realite.temoin_brassage import (
    APRES_RESSERREMENT,
    AVANT_RESSERREMENT,
    EXTRA_REFERENCE_LINES,
    REFERENCE_12000_TEMOIN_VERBATIM,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "banc_realite"

NOUVEAUX = (
    "rag_naif_large",
    "rag_filtre_pieces",
    "rag_filtre_situations",
    "rag_filtre_union",
)
NOUVEAUX_ENTITES = (
    "rag_filtre_entites",
    "entites_lignes",
    "entites_lignes_puis_rag",
)
NOUVEAUX_SENS = (
    "rag_filtre_sens",
    "objets_sens",
    "objets_sens_puis_rag",
)
NOUVEAUX_DESACCORDS = (
    "desaccords_puis_rag",
    "desaccords_puis_rag_filtre",
)
NOUVEAU_G12 = ("desaccords_puis_g12",)
LABELS_NOUVEAUX = {
    "rag_naif_large": "RAG naïf, contexte rempli",
    "rag_filtre_pieces": "filtre par le RAG puis RAG",
    "rag_filtre_situations": "filtre par les situations puis RAG",
    "rag_filtre_union": "deux filtres réunis puis RAG",
}
EXISTANTS_EXTRA = (
    "brain_banc_g24",
    "brain_banc_g36",
    "brain_banc_g48",
    "brain_banc_v1",
    "brain_banc_v2",
    "brain_banc_v3",
    "brain_banc_v4",
    "brain_mrfs",
    "brain_mrfs_lexical",
    "brain_mrfs_g12",
    "brain_situations",
    "brain_situations_g12",
    "brain_situations_lettre",
    "brain_situations_lettre_g12",
)
EXISTANTS_LABELS = {
    "brain_banc_v1": "porte 3 voisinage, 1 arête (atomes)",
    "brain_banc_v2": "porte 3 voisinage, 2 arêtes (atomes)",
    "brain_banc_v3": "porte 3 voisinage, 1 arête (phrases)",
    "brain_banc_v4": "porte 3 voisinage, 2 arêtes (phrases)",
    "brain_banc_calibre": "brain_banc (fenêtre calibrée sur l'or)",
    "brain_banc_g24": "Brain banc (g24, réglé sur le budget)",
    "brain_banc_g36": "Brain banc (g36, réglé sur le budget)",
    "brain_banc_g48": "Brain banc (g48, réglé sur le budget)",
    "brain_mrfs": "MRFS seul (constructeur de situations)",
    "brain_mrfs_lexical": "MRFS lexical seul (sans vecteurs)",
    "brain_mrfs_g12": "g12 puis MRFS (à plusieurs)",
    "brain_situations": "situations composites seules",
    "brain_situations_g12": "g12 puis situations composites (à plusieurs)",
    "brain_situations_lettre": "situations composites seules, lettre de la chronologie",
    "brain_situations_lettre_g12": (
        "g12 puis situations composites, lettre de la chronologie (à plusieurs)"
    ),
}
ORDRE_COMPARATIF = '''    order = [
        "rag_naif",
        "brain_prod",
        "brain_banc",
        "brain_banc_g24",
        "brain_banc_g36",
        "brain_banc_g48",
        "brain_banc_calibre",
        "brain_banc_v1",
        "brain_banc_v2",
        "brain_banc_v3",
        "brain_banc_v4",
        "mem0",
        "letta_archive",
    ]'''


def test_reference_lines_et_temoin_historiques_inchanges() -> None:
    assert REFERENCE_LINES == (
        ("brain_prod", 8, 12000),
        ("rag_naif", 12, 12000),
        ("brain_banc", None, 12000),
        ("mem0", 20, 12000),
    )
    assert EXPECTED_VERBATIM == {
        "brain_prod": 39,
        "rag_naif": 85,
        "brain_banc": 84,
        "mem0": 2,
    }
    assert REFERENCE_12000_TEMOIN_VERBATIM["greenloop"]["rag_naif"] == 46
    assert REFERENCE_12000_TEMOIN_VERBATIM["moreau"]["rag_naif"] == 14
    assert AVANT_RESSERREMENT == [
        ("brain_prod", "verbatim", 39, 4),
        ("brain_prod", "valeur", 49, 4),
        ("rag_naif", "verbatim", 85, 46),
        ("rag_naif", "valeur", 90, 48),
        ("brain_banc", "verbatim", 84, 23),
        ("brain_banc", "valeur", 89, 32),
        ("mem0", "verbatim", 2, 0),
        ("mem0", "valeur", 47, 19),
    ]
    assert APRES_RESSERREMENT == [
        ("brain_prod", "verbatim", 39, 4),
        ("brain_prod", "valeur", 37, 4),
        ("rag_naif", "verbatim", 85, 46),
        ("rag_naif", "valeur", 71, 43),
        ("brain_banc", "verbatim", 84, 23),
        ("brain_banc", "valeur", 75, 22),
        ("mem0", "verbatim", 2, 0),
        ("mem0", "valeur", 28, 15),
    ]


def test_labels_existants_inchanges_nouveaux_apres() -> None:
    for nom, label in EXISTANTS_LABELS.items():
        assert FOURNISSEUR_LABELS[nom] == label
    for nom, label in LABELS_NOUVEAUX.items():
        assert FOURNISSEUR_LABELS[nom] == label
        assert _fournisseur_label(nom, 12) == label
    cles = list(FOURNISSEUR_LABELS)
    for nom in EXISTANTS_LABELS:
        assert cles.index(nom) < cles.index("rag_naif_large")
    assert FOURNISSEUR_LABELS["desaccords_puis_g12"] == "désaccords en tête puis g12"
    assert _fournisseur_label("desaccords_puis_g12", 12) == "désaccords en tête puis g12"
    assert cles.index("desaccords_puis_rag") < cles.index("desaccords_puis_g12")
    assert cles.index("desaccords_puis_rag_filtre") < cles.index("desaccords_puis_g12")
    assert cles[-1] == "desaccords_puis_g12"


def test_extra_reference_ajoute_apres_existants() -> None:
    noms = [nom for nom, _k, _budget in EXTRA_REFERENCE_LINES]
    assert tuple(noms[: len(EXISTANTS_EXTRA)]) == EXISTANTS_EXTRA
    offset = len(EXISTANTS_EXTRA)
    assert tuple(noms[offset : offset + len(NOUVEAUX)]) == NOUVEAUX
    debut_entites = offset + len(NOUVEAUX)
    assert tuple(noms[debut_entites : debut_entites + len(NOUVEAUX_ENTITES)]) == NOUVEAUX_ENTITES
    debut_sens = debut_entites + len(NOUVEAUX_ENTITES)
    assert tuple(noms[debut_sens : debut_sens + len(NOUVEAUX_SENS)]) == NOUVEAUX_SENS
    debut_desaccords = debut_sens + len(NOUVEAUX_SENS)
    assert tuple(noms[debut_desaccords : debut_desaccords + len(NOUVEAUX_DESACCORDS)]) == NOUVEAUX_DESACCORDS
    assert tuple(noms[debut_desaccords + len(NOUVEAUX_DESACCORDS) :]) == NOUVEAU_G12
    for nom in (*NOUVEAUX, *NOUVEAUX_ENTITES, *NOUVEAUX_SENS, *NOUVEAUX_DESACCORDS, *NOUVEAU_G12):
        assert (nom, None, 12000) in EXTRA_REFERENCE_LINES


def test_k_fixe_et_effectif_des_quatre_lignes() -> None:
    for nom in NOUVEAUX:
        assert nom in K_FIXE
        assert nom in K_EFFECTIF_FOURNISSEUR
        assert nom not in BRAIN_FOURNISSEURS
    assert K_EFFECTIF_FOURNISSEUR["rag_naif_large"] == K_RAG_LARGE
    assert K_EFFECTIF_FOURNISSEUR["rag_filtre_pieces"] == K_VIVIER_FILTRE
    assert K_EFFECTIF_FOURNISSEUR["rag_filtre_situations"] == K_VIVIER_FILTRE
    assert K_EFFECTIF_FOURNISSEUR["rag_filtre_union"] == K_VIVIER_FILTRE


def test_instancier_quatre_lignes_apres_rag_naif() -> None:
    from banc_realite.dossiers import dossier_courant
    from banc_realite.fournisseurs_contexte import (
        RagFiltrePieces,
        RagFiltreSituations,
        RagFiltreUnion,
        RagNaifLarge,
    )

    moreau = dossier_courant("moreau")
    assert _instancier("rag_naif_large", {}, dossier=moreau).etiquette == "rag_naif_large"
    assert isinstance(_instancier("rag_naif_large", {}, dossier=moreau), RagNaifLarge)
    assert isinstance(_instancier("rag_filtre_pieces", {}, dossier=moreau), RagFiltrePieces)
    assert isinstance(
        _instancier("rag_filtre_situations", {}, dossier=moreau), RagFiltreSituations
    )
    assert isinstance(_instancier("rag_filtre_union", {}, dossier=moreau), RagFiltreUnion)
    with pytest.raises(ValueError, match="inconnu"):
        _instancier("plafond_filtre_or", {}, dossier=moreau)


def test_ordre_chargement_comparatif_inchange() -> None:
    source = (SCRIPTS / "mesure_comparatif_etage2.py").read_text(encoding="utf-8")
    assert ORDRE_COMPARATIF in source
