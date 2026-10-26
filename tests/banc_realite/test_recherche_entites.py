# SPDX-License-Identifier: AGPL-3.0-only
"""Recherche filtrée par reconstitution — sélection et fournisseurs."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import json
from pathlib import Path

import pytest

from banc_realite.fournisseurs_contexte import (
    DesaccordsPuisG12,
    DesaccordsPuisRag,
    DesaccordsPuisRagFiltre,
    EntitesLignes,
    EntitesLignesPuisRag,
    K_RAG_LARGE,
    K_VIVIER_FILTRE,
    ObjetsSens,
    ObjetsSensPuisRag,
    PART_LIGNES,
    RagFiltreEntites,
    RagFiltreSens,
    filtrer_passages_par_pieces,
)
from banc_realite.mesure_comparatif_etage2 import (
    BRAIN_FOURNISSEURS,
    FOURNISSEUR_LABELS,
    K_EFFECTIF_FOURNISSEUR,
    K_FIXE,
    REFERENCE_LINES,
    _instancier,
)
from banc_realite.recherche_entites import (
    BM25_B,
    BM25_K1,
    K_GRANDEURS,
    K_UNITES_REPLI,
    N_DESACCORDS_EN_TETE,
    N_OBJETS_FILTRE,
    IndexReconstitution,
    charger_index,
    desaccords_en_tete,
    lire_question,
    objets_de_recherche,
    selectionner,
    selectionner_par_le_sens,
)
from banc_realite.temoin_brassage import EXTRA_REFERENCE_LINES


def _ecrire_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    registre = {
        "entites": [
            {
                "nom": "Site Alpha",
                "type": "site",
                "cles": ["Site Alpha", "SITE-ALP"],
                "variantes": [],
                "par_periode": True,
            }
        ],
        "unites": [
            {
                "id": "site-alpha-2025-01",
                "entite": "Site Alpha",
                "type": "site",
                "titre": "Site Alpha, mois de service 2025-01",
                "cles": ["Site Alpha", "SITE-ALP"],
                "variantes": [],
                "periode": ["2025-01"],
            },
            {
                "id": "site-alpha-2025-02",
                "entite": "Site Alpha",
                "type": "site",
                "titre": "Site Alpha, mois de service 2025-02",
                "cles": ["Site Alpha", "SITE-ALP"],
                "variantes": [],
                "periode": ["2025-02"],
            },
            {
                "id": "grandeur-tonnage-january",
                "entite": "tonnage mensuel",
                "type": "grandeur",
                "titre": "tonnage mensuel January 2025",
                "cles": [],
                "variantes": [],
                "periode": [],
                "lignes": [{"piece": "report.md", "lignes": [4]}],
            },
        ],
    }
    (root / "registre.json").write_text(json.dumps(registre, ensure_ascii=False, indent=2), encoding="utf-8")
    paquets = {
        "site-alpha-2025-01": {
            "titre": "Site Alpha, mois de service 2025-01",
            "cles": ["Site Alpha", "SITE-ALP"],
            "sections": {
                "entite": {
                    "report.md": {
                        "1": "Site Alpha collected 12.5 tonnes in January 2025",
                        "2": "Reference SITE-ALP-2025-01 applies",
                        "10": "Idle note on Site Alpha yard layout",
                        "11": "Idle note continued about the yard",
                        "13": "Orphan remark on a skip",
                    }
                },
                "document": {},
                "variantes": {},
                "emergent": {},
            },
        },
        "site-alpha-2025-02": {
            "titre": "Site Alpha, mois de service 2025-02",
            "cles": ["Site Alpha"],
            "sections": {
                "entite": {"report.md": {"3": "Site Alpha collected 10.0 tonnes in February 2025"}},
                "document": {},
                "variantes": {},
                "emergent": {},
            },
        },
        "grandeur-tonnage-january": {
            "titre": "tonnage mensuel January 2025",
            "cles": [],
            "sections": {
                "entite": {},
                "document": {},
                "variantes": {},
                "emergent": {"report.md": {"4": "Monthly tonnage January 2025 total 12.5"}},
            },
        },
    }
    (root / "paquets").mkdir(exist_ok=True)
    for uid, paquet in paquets.items():
        (root / "paquets" / f"{uid}.json").write_text(
            json.dumps(paquet, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    entites = {
        "site-alpha-2025-01": {
            "faits": [
                {
                    "quoi": "tonnage collecté",
                    "comparaison": "desaccord",
                    "versions": [
                        {
                            "piece": "report.md",
                            "lignes": [1],
                            "lettre": "Site Alpha collected 12.5 tonnes in January 2025",
                        },
                        {
                            "piece": "report.md",
                            "lignes": [2],
                            "lettre": "Reference SITE-ALP-2025-01 applies",
                        },
                    ],
                },
                {
                    "quoi": "etat interne A",
                    "comparaison": "concordant",
                    "versions": [
                        {
                            "piece": "report.md",
                            "lignes": [2],
                            "lettre": "Reference SITE-ALP-2025-01 applies",
                        }
                    ],
                },
            ],
            "regles": [
                {
                    "enonce": "la reference SITE-ALP s'applique au mois",
                    "piece": "report.md",
                    "lignes": [2],
                    "lettre": "Reference SITE-ALP-2025-01 applies",
                }
            ],
            "identites": [
                {
                    "variante": "SITE-ALP",
                    "piece": "report.md",
                    "lignes": [2],
                    "lettre": "Reference SITE-ALP-2025-01 applies",
                }
            ],
        },
        "site-alpha-2025-02": {
            "faits": [
                {
                    "quoi": "tonnage février",
                    "comparaison": "concordant",
                    "versions": [
                        {
                            "piece": "report.md",
                            "lignes": [3],
                            "lettre": "Site Alpha collected 10.0 tonnes in February 2025",
                        }
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
        "grandeur-tonnage-january": {
            "faits": [
                {
                    "quoi": "total mensuel",
                    "comparaison": "une_seule_source",
                    "versions": [
                        {
                            "piece": "report.md",
                            "lignes": [4],
                            "lettre": "Monthly tonnage January 2025 total 12.5",
                        }
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    }
    (root / "entites").mkdir(exist_ok=True)
    for uid, payload in entites.items():
        (root / "entites" / f"{uid}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def test_constantes_recherche_fixees_avant_mesure() -> None:
    assert K_GRANDEURS == 6
    assert K_UNITES_REPLI == 6
    assert N_OBJETS_FILTRE == 12
    assert N_DESACCORDS_EN_TETE == 3
    assert BM25_K1 == 1.5
    assert BM25_B == 0.75
    assert PART_LIGNES == 6_000


def test_lire_question_cles_identifiants_dates_mois(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    rec = lire_question(
        "For Site Alpha in January 2025, what was SITE-ALP-2025-01 and 12.5?",
        index,
    )
    assert "Site Alpha" in rec.cles
    assert "SITE-ALP-2025-01" in rec.identifiants
    assert "2025-01" in rec.mois
    assert "12.5" in rec.nombres
    assert "site" in rec.mots or "alpha" in rec.mots


def test_selection_filtre_periode_et_bonus_desaccord(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    sel = selectionner(
        "Site Alpha tonnage January 2025 12.5",
        index,
    )
    motifs = {item.id: item.motif for item in sel.unites_candidates}
    assert "site-alpha-2025-01" in motifs
    assert motifs["site-alpha-2025-01"] == "cle"
    assert "site-alpha-2025-02" not in motifs
    assert sel.faits_classes
    assert sel.faits_classes[0].score >= 5
    assert "report.md" in sel.pieces_designees


def test_selection_repli_bm25_sans_cle_ni_identifiant(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    sel = selectionner("Monthly tonnage January 2025 total", index)
    motifs = [item.motif for item in sel.unites_candidates]
    assert "repli" in motifs or "grandeur" in motifs


def test_charger_index_absent_leve(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reconstitution absente"):
        charger_index(tmp_path / "absent")


def _ecrire_fixture_unite_absente(root: Path) -> str:
    """Trois unités au registre, une sans fichier entites/<id>.json."""
    absent_id = "unite-sans-reconstitution"
    root.mkdir(parents=True, exist_ok=True)
    registre = {
        "entites": [
            {
                "nom": "Ent A",
                "type": "site",
                "cles": ["Ent A"],
                "variantes": [],
                "par_periode": False,
            }
        ],
        "unites": [
            {
                "id": "unite-ok-1",
                "entite": "Ent A",
                "type": "site",
                "titre": "Ent A période 1",
                "cles": ["Ent A"],
                "variantes": [],
                "periode": [],
            },
            {
                "id": "unite-ok-2",
                "entite": "Ent A",
                "type": "site",
                "titre": "Ent A période 2",
                "cles": ["Ent A"],
                "variantes": [],
                "periode": [],
            },
            {
                "id": absent_id,
                "entite": "Ent A",
                "type": "site",
                "titre": "Ent A sans reconstitution",
                "cles": ["Ent A"],
                "variantes": [],
                "periode": [],
            },
        ],
    }
    (root / "registre.json").write_text(
        json.dumps(registre, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "paquets").mkdir(exist_ok=True)
    (root / "entites").mkdir(exist_ok=True)
    paquet_vide = {
        "titre": "",
        "cles": [],
        "sections": {"entite": {}, "document": {}, "variantes": {}, "emergent": {}},
    }
    for uid in ("unite-ok-1", "unite-ok-2", absent_id):
        paquet = dict(paquet_vide)
        paquet["titre"] = uid
        (root / "paquets" / f"{uid}.json").write_text(
            json.dumps(paquet, ensure_ascii=False),
            encoding="utf-8",
        )
    for uid in ("unite-ok-1", "unite-ok-2"):
        (root / "entites" / f"{uid}.json").write_text(
            json.dumps({"faits": [], "regles": [], "identites": []}),
            encoding="utf-8",
        )
    return absent_id


def test_index_unites_absentes_exclues_de_la_selection(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    absent_id = _ecrire_fixture_unite_absente(root)
    index = charger_index(root)
    assert list(index.absentes) == [absent_id]

    sel = selectionner("Question sur Ent A", index)
    ids_candidates = {item.id for item in sel.unites_candidates}
    assert absent_id not in ids_candidates
    assert sel.unites_absentes == 1

    with pytest.raises(FileNotFoundError, match="reconstitution entité absente"):
        index.entite(absent_id)

    objets = objets_de_recherche(index)
    assert absent_id not in {objet.unite_id for objet in objets}


def _hit(text: str, source: str, score: float, rank: int, *, line_start: int = 1, line_end: int | None = None) -> dict:
    fin = line_end if line_end is not None else line_start
    return {
        "kind": "file_hit",
        "id": f"{source}:{rank}",
        "score": score,
        "rank": rank,
        "text": text,
        "source": source,
        "snippet": text[:280],
        "provenance": {
            "source_file_path": source,
            "line_start": line_start,
            "line_end": fin,
        },
    }


def test_rag_filtre_entites_repli_sans_piece(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    captured: list[int] = []

    def fake_vivier(self, question, *, k, connection):
        del question, connection
        captured.append(k)
        return [_hit("x", "other.md", 0.5, 1)]

    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.RagNaif.contexte",
        fake_vivier,
    )
    provider = RagFiltreEntites({}, reconstitution_dir=root)
    out = provider.contexte({"id": "q1", "question": "unknown topic xyz"}, k=12, connection=object())
    assert captured == [K_RAG_LARGE]
    assert out
    stats = provider.stats_par_question["q1"]
    assert stats["repli_sans_filtre"] is True


def test_rag_filtre_entites_filtre_pieces_designees(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)

    def fake_vivier(self, question, *, k, connection):
        del question, connection
        return [
            _hit("a", "report.md", 0.9, 1),
            _hit("b", "other.md", 0.8, 2),
        ]

    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.RagNaif.contexte",
        fake_vivier,
    )
    provider = RagFiltreEntites({}, reconstitution_dir=root)
    out = provider.contexte(
        {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"},
        k=12,
        connection=object(),
    )
    assert [item["source"] for item in out] == ["report.md"]
    stats = provider.stats_par_question["q1"]
    assert stats["repli_sans_filtre"] is False
    assert stats["pieces_designees"] == ["report.md"]


def test_entites_lignes_format_element(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    provider = EntitesLignes(reconstitution_dir=root)
    elements = provider.contexte(
        {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"},
        k=12,
        connection=object(),
    )
    assert elements
    item = elements[0]
    assert item["kind"] == "entite_ligne"
    assert set(item) >= {"id", "rank", "score", "text", "source", "snippet", "provenance"}
    assert "report.md:" in item["text"]
    assert item["provenance"]["source_file_path"] == "report.md"


def test_entites_lignes_puis_rag_dedup_ligne(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)

    def fake_vivier(self, question, *, k, connection):
        del question, connection
        return [
            _hit("dup", "report.md", 0.9, 1, line_start=1),
            _hit("new", "report.md", 0.8, 2, line_start=3),
        ]

    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.RagNaif.contexte",
        fake_vivier,
    )
    provider = EntitesLignesPuisRag({}, reconstitution_dir=root)
    elements = provider.contexte(
        {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"},
        k=12,
        connection=object(),
    )
    rag_items = [item for item in elements if item.get("kind") == "file_hit"]
    assert len(rag_items) == 1
    assert rag_items[0]["provenance"]["line_start"] == 3


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
LABELS_ENTITES = {
    "rag_filtre_entites": "reconstitution : filtre par entités puis RAG",
    "entites_lignes": "reconstitution : lignes désignées seules",
    "entites_lignes_puis_rag": "reconstitution : lignes désignées puis RAG filtré",
}
LABELS_SENS = {
    "rag_filtre_sens": "reconstitution : filtre par le sens puis RAG",
    "objets_sens": "reconstitution : objets classés par le sens",
    "objets_sens_puis_rag": "reconstitution : objets classés puis RAG filtré",
}
LABELS_DESACCORDS = {
    "desaccords_puis_rag": "désaccords en tête puis RAG",
    "desaccords_puis_rag_filtre": "désaccords en tête puis RAG filtré par le sens",
}
LABELS_G12 = {
    "desaccords_puis_g12": "désaccords en tête puis g12",
}


def test_reference_lines_inchangees_trois_lignes_entites_apres() -> None:
    assert REFERENCE_LINES == (
        ("brain_prod", 8, 12000),
        ("rag_naif", 12, 12000),
        ("brain_banc", None, 12000),
        ("mem0", 20, 12000),
    )
    noms = [nom for nom, _k, _budget in EXTRA_REFERENCE_LINES]
    assert noms[-9:-6] == list(NOUVEAUX_ENTITES)
    assert noms[-6:-3] == list(NOUVEAUX_SENS)
    assert noms[-3:-1] == list(NOUVEAUX_DESACCORDS)
    assert noms[-1:] == list(NOUVEAU_G12)
    for nom, label in LABELS_ENTITES.items():
        assert FOURNISSEUR_LABELS[nom] == label
    for nom, label in LABELS_SENS.items():
        assert FOURNISSEUR_LABELS[nom] == label
    for nom, label in LABELS_DESACCORDS.items():
        assert FOURNISSEUR_LABELS[nom] == label
    for nom, label in LABELS_G12.items():
        assert FOURNISSEUR_LABELS[nom] == label


def test_k_fixe_trois_lignes_entites() -> None:
    for nom in NOUVEAUX_ENTITES:
        assert nom in K_FIXE
        assert nom in K_EFFECTIF_FOURNISSEUR
        assert nom not in BRAIN_FOURNISSEURS
    assert K_EFFECTIF_FOURNISSEUR["rag_filtre_entites"] == K_VIVIER_FILTRE
    assert K_EFFECTIF_FOURNISSEUR["entites_lignes_puis_rag"] == K_VIVIER_FILTRE
    for nom in NOUVEAUX_SENS:
        assert nom in K_FIXE
        assert nom in K_EFFECTIF_FOURNISSEUR
        assert nom not in BRAIN_FOURNISSEURS
    assert K_EFFECTIF_FOURNISSEUR["rag_filtre_sens"] == K_VIVIER_FILTRE
    assert K_EFFECTIF_FOURNISSEUR["objets_sens"] == 0
    assert K_EFFECTIF_FOURNISSEUR["objets_sens_puis_rag"] == K_VIVIER_FILTRE
    for nom in NOUVEAUX_DESACCORDS:
        assert nom in K_FIXE
        assert nom in K_EFFECTIF_FOURNISSEUR
        assert nom not in BRAIN_FOURNISSEURS
    assert K_EFFECTIF_FOURNISSEUR["desaccords_puis_rag"] == K_RAG_LARGE
    assert K_EFFECTIF_FOURNISSEUR["desaccords_puis_rag_filtre"] == K_VIVIER_FILTRE
    for nom in NOUVEAU_G12:
        assert nom in K_FIXE
        assert nom in K_EFFECTIF_FOURNISSEUR
        assert nom not in BRAIN_FOURNISSEURS
    assert K_EFFECTIF_FOURNISSEUR["desaccords_puis_g12"] == K_EFFECTIF_FOURNISSEUR["brain_banc"]


def test_instancier_trois_lignes_entites(tmp_path: Path) -> None:
    from banc_realite.dossiers import dossier_courant

    moreau = dossier_courant("moreau")
    with pytest.raises(FileNotFoundError, match="reconstitution absente"):
        _instancier("rag_filtre_entites", {}, dossier=moreau)
    root = tmp_path / "greenloop"
    _ecrire_fixture(root)
    provider = RagFiltreEntites({}, dossier=moreau, reconstitution_dir=root)
    assert isinstance(provider, RagFiltreEntites)
    assert isinstance(
        EntitesLignes(reconstitution_dir=root),
        EntitesLignes,
    )


def test_selectionner_greenloop_lecture_seule() -> None:
    root = BANC_VAR / 'dossiers' / 'mrfs' / 'entites' / 'greenloop'
    if not (root / "registre.json").is_file():
        pytest.skip("reconstitution GreenLoop absente sur ce VPS")
    index = charger_index(root)
    sel = selectionner("What is certificate GL-CERT-JTS-R4A total 20.415?", index)
    assert sel.reconnaissance.identifiants or sel.reconnaissance.nombres
    assert sel.unites_candidates


def _lignes_paquet_entite_document(index) -> set[tuple[str, str, int]]:
    attendues: set[tuple[str, str, int]] = set()
    for unite in index.unites:
        uid = str(unite.get("id") or "")
        path = index.repertoire / "paquets" / f"{uid}.json"
        if not path.is_file():
            continue
        paquet = json.loads(path.read_text(encoding="utf-8"))
        sections = paquet.get("sections") or {}
        for nom in ("entite", "document"):
            for piece, lignes in (sections.get(nom) or {}).items():
                for num_s in lignes:
                    attendues.add((uid, str(piece), int(num_s)))
    return attendues


def test_objets_couvrent_toutes_les_lignes_entite_document(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    sortes = {objet.sorte for objet in objets}
    assert "fait" in sortes
    assert "regle" in sortes
    assert "lignes" in sortes
    ids = [objet.id for objet in objets]
    assert "site-alpha-2025-01#f1" in ids
    assert "site-alpha-2025-01#r1" in ids
    assert "site-alpha-2025-01#lreport.md:10" in ids
    assert "site-alpha-2025-01#lreport.md:13" in ids
    couvertes: set[tuple[str, str, int]] = set()
    for objet in objets:
        for ligne in objet.lignes:
            couvertes.add((objet.unite_id, ligne.piece, ligne.numero))
    assert _lignes_paquet_entite_document(index) <= couvertes
    reliquat_10 = next(objet for objet in objets if objet.id.endswith("#lreport.md:10"))
    assert [ligne.numero for ligne in reliquat_10.lignes] == [10, 11]
    assert reliquat_10.empreinte == __import__("hashlib").sha256(
        reliquat_10.texte_recherche.encode("utf-8")
    ).hexdigest()


def _vecteurs_objets(objets, *, desaccord=(1.0, 0.0), autre=(0.0, 1.0)):
    return {
        objet.empreinte: list(
            desaccord if objet.comparaison == "desaccord" else autre
        )
        for objet in objets
    }


def test_selectionner_par_le_sens_note_tous_les_objets(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    vecteurs = _vecteurs_objets(objets)
    sel = selectionner_par_le_sens(
        "Site Alpha tonnage January 2025 12.5",
        index,
        vecteur_question=[1.0, 0.0],
        vecteurs_objets=vecteurs,
    )
    assert len(sel.objets_classes) == len(objets)
    premier = sel.objets_classes[0].objet
    assert premier.sorte == "fait"
    assert premier.comparaison == "desaccord"
    assert sel.pieces_designees
    assert "report.md" in sel.pieces_designees


def test_selectionner_par_le_sens_vecteur_absent_leve(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    vecteurs = _vecteurs_objets(objets)
    vecteurs.pop(objets[0].empreinte)
    with pytest.raises(RuntimeError, match="embedding d'objet absent"):
        selectionner_par_le_sens(
            "Site Alpha tonnage January 2025 12.5",
            index,
            vecteur_question=[1.0, 0.0],
            vecteurs_objets=vecteurs,
        )


def test_selectionner_par_le_sens_desaccord_avant_id_a_score_egal(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    meme = [1.0, 0.0]
    vecteurs = {objet.empreinte: meme for objet in objets}
    sel = selectionner_par_le_sens(
        "zzzz without keys",
        index,
        vecteur_question=meme,
        vecteurs_objets=vecteurs,
    )
    faits = [note.objet for note in sel.objets_classes if note.objet.sorte == "fait"]
    grandeur = next(objet for objet in faits if objet.unite_id.startswith("grandeur"))
    desaccord = next(objet for objet in faits if objet.comparaison == "desaccord")
    rang = {note.objet.id: note.rang for note in sel.objets_classes}
    assert grandeur.id < desaccord.id
    assert rang[desaccord.id] < rang[grandeur.id]


def test_rag_filtre_sens_restreint_aux_pieces(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    question = {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"}
    embed_cache = {question["question"]: [1.0, 0.0]}
    vecteurs = _vecteurs_objets(objets)

    def fake_vivier(self, question_arg, *, k, connection):
        del question_arg, connection
        assert k == K_VIVIER_FILTRE
        return [
            _hit("a", "report.md", 0.9, 1),
            _hit("b", "other.md", 0.8, 2),
        ]

    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.RagNaif.contexte",
        fake_vivier,
    )
    provider = RagFiltreSens(
        embed_cache, reconstitution_dir=root, embeddings_objets=vecteurs
    )
    out = provider.contexte(question, k=12, connection=object())
    assert [item["source"] for item in out] == ["report.md"]


def test_objets_sens_un_element_par_objet_et_piece(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    question = {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"}
    provider = ObjetsSens(
        {question["question"]: [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
    )
    elements = provider.contexte(question, k=12, connection=object())
    assert elements
    assert elements[0]["kind"] == "objet_sens"
    assert elements[0]["text"].startswith("comparaison: desaccord")
    assert "report.md:" in elements[0]["text"]
    rangs = [item["rank"] for item in elements]
    assert rangs == sorted(rangs)


def test_objets_sens_cache_froid_refuse(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    provider = ObjetsSens(
        {},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
    )
    with pytest.raises(RuntimeError, match="cache froid"):
        provider.contexte(
            {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"},
            k=12,
            connection=object(),
        )


def test_objets_sens_puis_rag_dedup_ligne(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    question = {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"}

    def fake_vivier(self, question_arg, *, k, connection):
        del question_arg, connection
        return [
            _hit("dup", "report.md", 0.9, 1, line_start=1),
            _hit("new", "report.md", 0.8, 2, line_start=99),
        ]

    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.RagNaif.contexte",
        fake_vivier,
    )
    provider = ObjetsSensPuisRag(
        {question["question"]: [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
    )
    elements = provider.contexte(question, k=12, connection=object())
    rag_items = [item for item in elements if item.get("kind") == "file_hit"]
    assert len(rag_items) == 1
    assert rag_items[0]["provenance"]["line_start"] == 99


def test_lignes_entites_precedentes_inchangees(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)

    def fake_vivier(self, question_arg, *, k, connection):
        del question_arg, connection
        return [
            _hit("a", "report.md", 0.9, 1),
            _hit("b", "other.md", 0.8, 2),
        ]

    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.RagNaif.contexte",
        fake_vivier,
    )
    question = {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"}
    filtre = RagFiltreEntites({}, reconstitution_dir=root)
    out = filtre.contexte(question, k=12, connection=object())
    assert [item["source"] for item in out] == ["report.md"]
    lignes = EntitesLignes(reconstitution_dir=root).contexte(
        question, k=12, connection=object()
    )
    assert lignes[0]["kind"] == "entite_ligne"
    assert "comparaison:" not in lignes[0]["text"]


def test_desaccords_en_tete_premiers_faits_dans_ordre(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    sel = selectionner_par_le_sens(
        "Site Alpha tonnage January 2025 12.5",
        index,
        vecteur_question=[1.0, 0.0],
        vecteurs_objets=_vecteurs_objets(objets),
    )
    notes = desaccords_en_tete(sel)
    assert len(notes) == 1
    assert notes[0].objet.comparaison == "desaccord"
    assert notes[0].rang == sel.objets_classes[0].rang


def test_desaccords_puis_rag_tete_sans_plafond_et_stats(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    question = {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"}

    def fake_vivier(self, question_arg, *, k, connection):
        del question_arg, connection
        assert k == K_RAG_LARGE
        return [
            _hit("dup", "report.md", 0.9, 1, line_start=1),
            _hit("new", "report.md", 0.8, 2, line_start=99),
        ]

    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.RagNaif.contexte",
        fake_vivier,
    )
    provider = DesaccordsPuisRag(
        {question["question"]: [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
    )
    elements = provider.contexte(question, k=12, connection=object())
    tete = [item for item in elements if item.get("kind") == "objet_sens"]
    rag_items = [item for item in elements if item.get("kind") == "file_hit"]
    assert tete
    assert tete[0]["text"].startswith("comparaison: desaccord")
    assert len(rag_items) == 1
    assert rag_items[0]["provenance"]["line_start"] == 99
    stats = provider.stats_par_question["q1"]
    assert stats["desaccords_en_tete"] == ["site-alpha-2025-01#f1"]
    assert stats["taille_tete_chars"] == sum(len(item["text"]) for item in tete)


def test_desaccords_puis_rag_filtre_utilise_rag_filtre_sens(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    question = {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"}

    def fake_vivier(self, question_arg, *, k, connection):
        del question_arg, connection
        assert k == K_VIVIER_FILTRE
        return [
            _hit("dup", "report.md", 0.9, 1, line_start=1),
            _hit("new", "report.md", 0.8, 2, line_start=99),
        ]

    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.RagNaif.contexte",
        fake_vivier,
    )
    provider = DesaccordsPuisRagFiltre(
        {question["question"]: [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
    )
    elements = provider.contexte(question, k=12, connection=object())
    rag_items = [item for item in elements if item.get("kind") == "file_hit"]
    assert [item["source"] for item in rag_items] == ["report.md"]
    stats = provider.stats_par_question["q1"]
    assert stats["desaccords_en_tete"] == ["site-alpha-2025-01#f1"]
    assert stats["taille_tete_chars"] > 0


def test_desaccords_puis_rag_cache_froid_refuse(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    provider = DesaccordsPuisRag(
        {},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
    )
    with pytest.raises(RuntimeError, match="cache froid"):
        provider.contexte(
            {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"},
            k=12,
            connection=object(),
        )


def _phrase_sans_lignes(text: str, source: str) -> dict:
    return {
        "kind": "phrase_hit",
        "id": f"phrase:{source}:0",
        "text": text,
        "source": source,
        "score": 0.5,
        "provenance": {"source_file_path": source, "passage_id": "p0"},
    }


def test_desaccords_puis_g12_tete_puis_g12_filtre_lignes(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    question = {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"}
    sans_lignes = _phrase_sans_lignes("fenêtre sans numéros", "report.md")
    toutes_deja = _hit("dup", "report.md", 0.9, 1, line_start=1)
    partiel = _hit("partiel", "report.md", 0.8, 2, line_start=1, line_end=3)
    nouveau = _hit("new", "report.md", 0.7, 3, line_start=99)

    def fake_g12(self, question_arg, *, k, connection):
        del question_arg, connection
        return [sans_lignes, toutes_deja, partiel, nouveau]

    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.BrainBanc.contexte",
        fake_g12,
    )
    provider = DesaccordsPuisG12(
        {question["question"]: [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
    )
    elements = provider.contexte(question, k=12, connection=object())
    tete = [item for item in elements if item.get("kind") == "objet_sens"]
    suite = elements[len(tete) :]
    assert tete
    assert tete[0]["text"].startswith("comparaison: desaccord")
    assert sans_lignes in suite
    assert toutes_deja not in suite
    assert partiel in suite
    assert nouveau in suite
    assert suite == [sans_lignes, partiel, nouveau]
    stats = provider.stats_par_question["q1"]
    assert stats["desaccords_en_tete"] == ["site-alpha-2025-01#f1"]
    assert stats["taille_tete_chars"] == sum(len(item["text"]) for item in tete)
    assert stats["passages_rendus"] == len(elements)
    assert stats["vivier"] == 4
    assert stats["k_demande"] == K_EFFECTIF_FOURNISSEUR["brain_banc"]


def test_desaccords_puis_g12_meme_tete_que_rag(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    question = {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"}

    def fake_vivier(self, question_arg, *, k, connection):
        del question_arg, connection
        return [_hit("new", "report.md", 0.8, 2, line_start=99)]

    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.RagNaif.contexte",
        fake_vivier,
    )
    monkeypatch.setattr(
        "banc_realite.fournisseurs_contexte.BrainBanc.contexte",
        fake_vivier,
    )
    cache = {question["question"]: [1.0, 0.0]}
    vecteurs = _vecteurs_objets(objets)
    rag = DesaccordsPuisRag(
        cache, reconstitution_dir=root, embeddings_objets=vecteurs
    )
    g12 = DesaccordsPuisG12(
        cache, reconstitution_dir=root, embeddings_objets=vecteurs
    )
    tete_rag = [item for item in rag.contexte(question, k=12, connection=object()) if item.get("kind") == "objet_sens"]
    tete_g12 = [item for item in g12.contexte(question, k=12, connection=object()) if item.get("kind") == "objet_sens"]
    assert tete_rag == tete_g12
    stats_rag = rag.stats_par_question["q1"]
    stats_g12 = g12.stats_par_question["q1"]
    assert stats_g12["desaccords_en_tete"] == stats_rag["desaccords_en_tete"]
    assert stats_g12["taille_tete_chars"] == stats_rag["taille_tete_chars"]
    for cle in (
        "question_id",
        "reconnaissance",
        "unites_candidates",
        "pieces_designees",
        "n_lignes_designees",
        "repli_sans_filtre",
        "vivier",
        "passages_rendus",
        "k_demande",
        "desaccords_en_tete",
        "taille_tete_chars",
    ):
        assert cle in stats_g12


def test_desaccords_puis_g12_cache_froid_refuse(tmp_path: Path) -> None:
    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    provider = DesaccordsPuisG12(
        {},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
    )
    with pytest.raises(RuntimeError, match="cache froid"):
        provider.contexte(
            {"id": "q1", "question": "Site Alpha tonnage January 2025 12.5"},
            k=12,
            connection=object(),
        )


def test_desaccords_puis_g12_etiquette_et_instancier(tmp_path: Path) -> None:
    from banc_realite.dossiers import dossier_courant

    root = tmp_path / "dossier"
    _ecrire_fixture(root)
    index = charger_index(root)
    objets = objets_de_recherche(index)
    provider = DesaccordsPuisG12(
        {"q": [1.0, 0.0]},
        reconstitution_dir=root,
        embeddings_objets=_vecteurs_objets(objets),
    )
    assert provider.etiquette == "desaccords_puis_g12"
    moreau = dossier_courant("moreau")
    with pytest.raises(FileNotFoundError, match="reconstitution absente"):
        _instancier("desaccords_puis_g12", {}, dossier=moreau)
