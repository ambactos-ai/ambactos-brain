# SPDX-License-Identifier: AGPL-3.0-only
"""Parité chemin matriciel vs cosine pour selectionner_par_le_sens."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import random
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts", "src"):
    entry = str(_ROOT / sub)
    if entry not in sys.path:
        sys.path.insert(0, entry)

from banc_realite.recherche_entites import (
    IndexBm25Preparation,
    _bm25,
    _lettre_objet,
    _score_lettre,
    _tokeniser_texte,
    charger_index,
    lire_question,
    matrice_depuis_cache,
    objets_de_recherche,
    preparer_objets,
    scores_lettre_prepares,
    selectionner_par_le_sens,
)


class _IndexMinimal:
    absentes: list = []
    _objets_cache = None

    def __init__(self, objets):
        self._objets = objets


class _ObjetMinimal:
    __slots__ = ("id", "empreinte", "sorte", "comparaison", "unite_id", "texte_recherche", "lignes", "payload")

    def __init__(self, ident: str, empreinte: str, texte: str):
        self.id = ident
        self.empreinte = empreinte
        self.sorte = "lignes"
        self.comparaison = ""
        self.unite_id = "u1"
        self.texte_recherche = texte
        self.lignes = ()
        self.payload = None


def _patch_index(monkeypatch: pytest.MonkeyPatch, objets: list[_ObjetMinimal]) -> None:
    index = _IndexMinimal(objets)

    def _objets_de_recherche(_index):
        return objets

    def _lire_question(_q, _index):
        from banc_realite.recherche_entites import ReconnaissanceQuestion

        return ReconnaissanceQuestion(
            cles=(),
            identifiants=(),
            dates=(),
            mois=(),
            annees=(),
            nombres=(),
            mots=(),
        )

    monkeypatch.setattr("banc_realite.recherche_entites.objets_de_recherche", _objets_de_recherche)
    monkeypatch.setattr("banc_realite.recherche_entites.lire_question", _lire_question)
    monkeypatch.setattr(
        "banc_realite.recherche_entites._unites_candidates",
        lambda *_a, **_k: (),
    )
    monkeypatch.setattr(
        "banc_realite.recherche_entites._unites_bonus",
        lambda *_a, **_k: set(),
    )
    monkeypatch.setattr(
        "banc_realite.recherche_entites._lignes_designees_objet",
        lambda *_a, **_k: [],
    )


def test_matrice_depuis_cache_vide() -> None:
    matrice, ordre = matrice_depuis_cache({}, [])
    assert ordre == []
    assert matrice.shape == (0, 0)


def test_selection_vectorisee_meme_ordre_et_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(158)
    dim = 16
    n = 50
    vecteurs: dict[str, list[float]] = {}
    objets: list[_ObjetMinimal] = []
    for i in range(n):
        empreinte = f"emp-{i}"
        vec = [rng.uniform(-1, 1) for _ in range(dim)]
        vecteurs[empreinte] = vec
        objets.append(_ObjetMinimal(f"obj-{i:03d}", empreinte, f"texte {i}"))
    question_vec = [rng.uniform(-1, 1) for _ in range(dim)]
    empreintes = [o.empreinte for o in objets]
    matrice, ordre = matrice_depuis_cache(vecteurs, empreintes)
    _patch_index(monkeypatch, objets)
    index = _IndexMinimal(objets)

    sel_ref = selectionner_par_le_sens(
        "question test",
        index,
        vecteur_question=question_vec,
        vecteurs_objets=vecteurs,
        matrice_objets=None,
    )
    sel_mat = selectionner_par_le_sens(
        "question test",
        index,
        vecteur_question=question_vec,
        vecteurs_objets=vecteurs,
        matrice_objets=(matrice, ordre),
    )
    ids_ref = [note.objet.id for note in sel_ref.objets_classes]
    ids_mat = [note.objet.id for note in sel_mat.objets_classes]
    assert ids_ref == ids_mat
    for ref, mat in zip(sel_ref.objets_classes, sel_mat.objets_classes):
        assert ref.objet.id == mat.objet.id
        assert abs(ref.score - mat.score) <= 1e-6


def test_bm25_index_meme_scores() -> None:
    documents = {
        "a": _tokeniser_texte("revenu locatif 2024"),
        "b": _tokeniser_texte("charges copropriété"),
    }
    query = ["revenu", "2024"]
    ref = _bm25(query, documents)
    prep = IndexBm25Preparation.depuis_documents(documents)
    rapide = prep.interroger(query)
    for uid in set(ref) | set(rapide):
        assert abs(ref.get(uid, 0.0) - rapide.get(uid, 0.0)) <= 1e-9


def test_scores_lettre_prepares_parite_echantillon() -> None:
    racine = (BANC_VAR / 'dossiers' / 'mrfs' / 'entites' / 'alpha_corp-ontologie-v1')
    if not racine.is_dir():
        pytest.skip("reconstitution alpha_corp absente sur ce VPS")
    index = charger_index(racine)
    preparation = preparer_objets(index)
    objets = objets_de_recherche(index)
    rng = random.Random(158)
    echantillon = rng.sample(objets, min(200, len(objets)))
    from banc_realite.or_dossier import load_questions

    questions = load_questions("alpha_corp")
    texte = str(next(q for q in questions if q["id"] == "GR-Q01")["question"])
    reconnaissance = lire_question(texte, index)
    prepares = scores_lettre_prepares(reconnaissance, preparation)
    for objet in echantillon:
        lettre = _lettre_objet(objet)
        attendu = _score_lettre(reconnaissance, lettre)
        assert prepares.get(objet.id, 0) == attendu


def test_selection_preparee_meme_ordre_que_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(158)
    dim = 16
    n = 50
    vecteurs: dict[str, list[float]] = {}
    objets: list[_ObjetMinimal] = []
    for i in range(n):
        empreinte = f"emp-{i}"
        vec = [rng.uniform(-1, 1) for _ in range(dim)]
        vecteurs[empreinte] = vec
        objets.append(_ObjetMinimal(f"obj-{i:03d}", empreinte, f"texte {i}"))
    question_vec = [rng.uniform(-1, 1) for _ in range(dim)]
    empreintes = [o.empreinte for o in objets]
    matrice, ordre = matrice_depuis_cache(vecteurs, empreintes)
    _patch_index(monkeypatch, objets)
    index = _IndexMinimal(objets)

    sel_ref = selectionner_par_le_sens(
        "question test",
        index,
        vecteur_question=question_vec,
        vecteurs_objets=vecteurs,
        matrice_objets=(matrice, ordre),
        preparation_objets=None,
    )
    from banc_realite.recherche_entites import PreparationObjets, IndexBm25Preparation

    docs = {o.id: _tokeniser_texte(o.texte_recherche) for o in objets}
    vide = IndexBm25Preparation.depuis_documents({})
    prep = PreparationObjets(
        objet_ids=tuple(o.id for o in objets),
        lettre_par_id={o.id: "" for o in objets},
        index_cles={},
        index_mots={},
        index_identifiants={},
        bm25_objets=IndexBm25Preparation.depuis_documents(docs),
        unites_par_entite={},
        unites_reconstituees=frozenset(),
        bm25_grandeurs=vide,
        bm25_unites_repli=vide,
    )
    sel_prep = selectionner_par_le_sens(
        "question test",
        index,
        vecteur_question=question_vec,
        vecteurs_objets=vecteurs,
        matrice_objets=(matrice, ordre),
        preparation_objets=prep,
    )
    assert [n.objet.id for n in sel_ref.objets_classes] == [
        n.objet.id for n in sel_prep.objets_classes
    ]


def test_prechauffer_tenant_sans_unite(tmp_path: Path) -> None:
    import json

    from brain.api_brain.recherche import prechauffer_tenant, vider_etat_tenant
    from brain.api_brain.tenants import TenantConfig

    reconst = tmp_path / "reconstitution"
    reconst.mkdir()
    (reconst / "registre.json").write_text(
        json.dumps({"unites": [], "entites": []}), encoding="utf-8"
    )
    (reconst / "embeddings-objets.json").write_text("{}", encoding="utf-8")
    cfg = TenantConfig(
        nom="vide",
        racine=tmp_path,
        reconstitution=reconst,
        ontologie=tmp_path / "ontologie.json",
        dossier_banc="greenloop",  # dossier connu : un dossier inconnu est refusé (158-S)
    )
    vider_etat_tenant("vide")
    etat = prechauffer_tenant(cfg)
    assert etat.matrice_objets is not None
    matrice, ordre = etat.matrice_objets
    assert ordre == []
    assert matrice.shape == (0, 0)
