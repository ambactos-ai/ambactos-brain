# SPDX-License-Identifier: AGPL-3.0-only
"""Mesure comparatif étage 2 — balayage k × budget par fournisseur."""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
from datetime import date
from pathlib import Path
from typing import Any, Literal

from .balayage_budget import _percentile
from .config import BANC_ROOT, export_runtime_env
from .dossiers import Dossier, dossier_courant
from .controle_budget_egal import _aggregate_config_rows
from .fidelite import load_gold
from .or_dossier import exiger_contextes_dossier, verifier_or_coherent
from .production import verifier_parite
from .fournisseurs_contexte import (
    COMPARATIF_DIR,
    BrainBanc,
    BrainBancCalibre,
    BrainMrfs,
    BrainMrfsG12,
    BrainMrfsLexical,
    BrainProd,
    BrainSituations,
    BrainSituationsG12,
    BrainSituationsLettre,
    BrainSituationsLettreG12,
    FournisseurFichier,
    K_LECTEUR_MRFS,
    K_RAG_LARGE,
    K_VIVIER_FILTRE,
    RagFiltrePieces,
    RagFiltreSituations,
    RagFiltreUnion,
    RagFiltreEntites,
    EntitesLignes,
    EntitesLignesPuisRag,
    RagFiltreSens,
    RagFiltreOntologie,
    ObjetsSens,
    ObjetsSensPuisRag,
    ObjetsSensPuisRagLectureSimple,
    ObjetsSensPuisRagLectureSimpleEntites,
    DesaccordsPuisRag,
    DesaccordsPuisRagFiltre,
    DesaccordsPuisG12,
    RagNaif,
    RagNaifLarge,
    comparatif_context_path,
    missing_embed_texts,
    rendre_contexte,
)
from .holders import build_holder_table
from .measure_disputes import _read_public_claims, _side_presence
from .measure_etage2 import _appui_summary, _load_embed_cache, _make_present_fn, load_questions
from .partage_disputes import partager
from .restitution import compute_verdict

RAPPORT_PATH = COMPARATIF_DIR / "RAPPORT-COMPARATIF-ETAGE2.md"
K_FIXE = frozenset(
    {
        "brain_banc",
        "brain_banc_v1",
        "brain_banc_v2",
        "brain_banc_v3",
        "brain_banc_v4",
        "brain_banc_calibre",
        "brain_banc_g24",
        "brain_banc_g36",
        "brain_banc_g48",
        "brain_mrfs",
        "brain_mrfs_lexical",
        "brain_mrfs_g12",
        "brain_situations",
        "brain_situations_g12",
        "brain_situations_lettre",
        "brain_situations_lettre_g12",
        "rag_naif_large",
        "rag_filtre_pieces",
        "rag_filtre_situations",
        "rag_filtre_union",
        "rag_filtre_entites",
        "entites_lignes",
        "entites_lignes_puis_rag",
        "rag_filtre_sens",
        "rag_filtre_ontologie",
        "objets_sens",
        "objets_sens_puis_rag",
        "objets_sens_puis_rag_lecture_simple",
        "objets_sens_puis_rag_lecture_simple_entites",
        "objets_sens_puis_rag_lecture_simple_entites_5000",
        "desaccords_puis_rag",
        "desaccords_puis_rag_filtre",
        "desaccords_puis_g12",
    }
)
K_PROD = 8
K_EFFECTIF_FOURNISSEUR: dict[str, int] = {
    "brain_banc": 12,
    "brain_banc_v1": 12,
    "brain_banc_v2": 12,
    "brain_banc_v3": 12,
    "brain_banc_v4": 12,
    "brain_banc_calibre": 12,
    "brain_banc_g24": 24,
    "brain_banc_g36": 36,
    "brain_banc_g48": 48,
    "brain_mrfs": K_LECTEUR_MRFS,
    "brain_mrfs_lexical": K_LECTEUR_MRFS,
    "brain_mrfs_g12": K_LECTEUR_MRFS,
    "brain_situations": K_LECTEUR_MRFS,
    "brain_situations_g12": K_LECTEUR_MRFS,
    "brain_situations_lettre": K_LECTEUR_MRFS,
    "brain_situations_lettre_g12": K_LECTEUR_MRFS,
    "rag_naif_large": K_RAG_LARGE,
    "rag_filtre_pieces": K_VIVIER_FILTRE,
    "rag_filtre_situations": K_VIVIER_FILTRE,
    "rag_filtre_union": K_VIVIER_FILTRE,
    "rag_filtre_entites": K_VIVIER_FILTRE,
    "entites_lignes": 0,
    "entites_lignes_puis_rag": K_VIVIER_FILTRE,
    "rag_filtre_sens": K_VIVIER_FILTRE,
    "rag_filtre_ontologie": K_VIVIER_FILTRE,
    "objets_sens": 0,
    "objets_sens_puis_rag": K_VIVIER_FILTRE,
    "objets_sens_puis_rag_lecture_simple": K_VIVIER_FILTRE,
    "objets_sens_puis_rag_lecture_simple_entites": K_VIVIER_FILTRE,
    "objets_sens_puis_rag_lecture_simple_entites_5000": K_VIVIER_FILTRE,
    "desaccords_puis_rag": K_RAG_LARGE,
    "desaccords_puis_rag_filtre": K_VIVIER_FILTRE,
    "desaccords_puis_g12": 12,
}
K_EFFECTIF_BANC = K_EFFECTIF_FOURNISSEUR
FOURNISSEUR_LABELS: dict[str, str] = {
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
    "rag_naif_large": "RAG naïf, contexte rempli",
    "rag_filtre_pieces": "filtre par le RAG puis RAG",
    "rag_filtre_situations": "filtre par les situations puis RAG",
    "rag_filtre_union": "deux filtres réunis puis RAG",
    "rag_filtre_entites": "reconstitution : filtre par entités puis RAG",
    "entites_lignes": "reconstitution : lignes désignées seules",
    "entites_lignes_puis_rag": "reconstitution : lignes désignées puis RAG filtré",
    "rag_filtre_sens": "reconstitution : filtre par le sens puis RAG",
    "rag_filtre_ontologie": "ontologie : filtre par entités littérales puis RAG",
    "objets_sens": "reconstitution : objets classés par le sens",
    "objets_sens_puis_rag": "reconstitution : objets classés puis RAG filtré",
    "objets_sens_puis_rag_lecture_simple": (
        "reconstitution : objets classés puis RAG filtré + lecture simple"
    ),
    "objets_sens_puis_rag_lecture_simple_entites": (
        "reconstitution : objets classés puis RAG filtré + lecture simple par entités"
    ),
    "objets_sens_puis_rag_lecture_simple_entites_5000": (
        "reconstitution : objets classés puis RAG filtré + lecture simple par entités (5 000)"
    ),
    "desaccords_puis_rag": "désaccords en tête puis RAG",
    "desaccords_puis_rag_filtre": "désaccords en tête puis RAG filtré par le sens",
    "desaccords_puis_g12": "désaccords en tête puis g12",
}
BRAIN_FOURNISSEURS = frozenset(
    {
        "brain_prod",
        "brain_banc",
        "brain_banc_v1",
        "brain_banc_v2",
        "brain_banc_v3",
        "brain_banc_v4",
        "brain_banc_calibre",
        "brain_banc_g24",
        "brain_banc_g36",
        "brain_banc_g48",
        "brain_mrfs",
        "brain_mrfs_lexical",
        "brain_mrfs_g12",
        "brain_situations",
        "brain_situations_g12",
        "brain_situations_lettre",
        "brain_situations_lettre_g12",
    }
)


def k_effectif_fournisseur(nom: str, k: int | None = None) -> int:
    if nom in K_EFFECTIF_FOURNISSEUR:
        return K_EFFECTIF_FOURNISSEUR[nom]
    return 12 if k is None else k
SHAPE_HORS_ETAGE2 = frozenset({"non_etabli"})
BUDGET_CORPUS_PCT = {"greenloop": 5.1, "moreau": 40.5}


def selection_questions_mesure(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Même sélection que l'agrégation appui de la mesure étage 2 (hors `non_etabli`)."""
    return [question for question in questions if question.get("shape") not in SHAPE_HORS_ETAGE2]


def questions_sans_appui(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [question for question in questions if not (question.get("supporting_assertions") or [])]


def _meta_mesure(dossier: Dossier, questions: list[dict[str, Any]]) -> dict[str, Any]:
    non_etabli = sum(1 for q in questions if q.get("shape") in SHAPE_HORS_ETAGE2)
    mesurables = len(questions) - non_etabli
    dispute_shape = "contradiction" if dossier.nature == "juridique" else "dispute"
    disputes_total = sum(1 for q in questions if q.get("shape") == dispute_shape)
    denominateur = 141 if dossier.nom == "greenloop" else mesurables
    return {
        "questions_total": len(questions),
        "questions_mesurables": mesurables,
        "non_etabli": non_etabli,
        "denominateur_appui": denominateur,
        "dispute_shape": dispute_shape,
        "disputes_total": disputes_total,
        "disputes_denominateur": 28 if dossier.nom == "greenloop" else disputes_total,
    }


def _rows_mesurables(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return selection_questions_mesure(rows)


def _denominateur_appui_key(dossier: Dossier) -> str:
    meta = _meta_mesure(dossier, [])
    return f"tout_tient_sur_{meta['denominateur_appui']}" if dossier.nom != "greenloop" else "tout_tient_sur_141"


FOURNISSEURS_RECONSTITUTION_DIR = frozenset(
    {
        "objets_sens",
        "objets_sens_puis_rag",
        "objets_sens_puis_rag_lecture_simple",
        "objets_sens_puis_rag_lecture_simple_entites",
        "objets_sens_puis_rag_lecture_simple_entites_5000",
        "entites_lignes",
        "entites_lignes_puis_rag",
        "rag_filtre_sens",
    }
)


def _fournisseur_desaccords_puis_rag(nom: str) -> bool:
    return nom.startswith("desaccords_puis_rag")


def _fournisseur_reconstitution_dir(nom: str) -> bool:
    return nom in FOURNISSEURS_RECONSTITUTION_DIR or _fournisseur_desaccords_puis_rag(nom)


def _fournisseur_embeddings_objets(nom: str) -> bool:
    return nom in {
        "objets_sens",
        "objets_sens_puis_rag",
        "objets_sens_puis_rag_lecture_simple",
        "objets_sens_puis_rag_lecture_simple_entites",
        "objets_sens_puis_rag_lecture_simple_entites_5000",
    } or _fournisseur_desaccords_puis_rag(nom)


def _etage2_json_filename(nom: str, suffixe: str | None) -> str:
    if suffixe:
        return f"etage2-{nom}@{suffixe}.json"
    return f"etage2-{nom}.json"


def _contextes_jsonl_filename(nom: str, suffixe: str | None) -> str:
    if suffixe:
        return f"contextes-{nom}@{suffixe}.jsonl"
    return f"contextes-{nom}.jsonl"


def etage2_resultat_path(
    nom: str,
    dossier: Dossier,
    suffixe: str | None = None,
) -> Path:
    return dossier.comparatif_dir / _etage2_json_filename(nom, suffixe)


def _opts_reconstitution_mesure(
    nom: str,
    *,
    reconstitution_dir: Path | None,
    embeddings_objets: dict[str, list[float]] | None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if reconstitution_dir is not None and _fournisseur_reconstitution_dir(nom):
        opts["reconstitution_dir"] = reconstitution_dir
    if embeddings_objets is not None and _fournisseur_embeddings_objets(nom):
        opts["embeddings_objets"] = embeddings_objets
    return opts


def _rapport_path(dossier: Dossier | None = None, suffixe: str | None = None) -> Path:
    current = dossier if dossier is not None else dossier_courant()
    if suffixe:
        return current.comparatif_dir / f"RAPPORT-COMPARATIF-ETAGE2@{suffixe}.md"
    return current.comparatif_dir / "RAPPORT-COMPARATIF-ETAGE2.md"


def _instancier(
    nom: str,
    embed_cache: dict[str, list[float]],
    *,
    dossier: Dossier | None = None,
    reconstitution_dir: Path | None = None,
    embeddings_objets: dict[str, list[float]] | None = None,
):
    current = dossier if dossier is not None else dossier_courant()
    rec = _opts_reconstitution_mesure(
        nom,
        reconstitution_dir=reconstitution_dir,
        embeddings_objets=embeddings_objets,
    )
    if nom == "rag_naif":
        return RagNaif(embed_cache, dossier=current)
    if nom == "rag_naif_large":
        return RagNaifLarge(embed_cache, dossier=current)
    if nom == "rag_filtre_pieces":
        return RagFiltrePieces(embed_cache, dossier=current)
    if nom == "rag_filtre_situations":
        return RagFiltreSituations(embed_cache, dossier=current)
    if nom == "rag_filtre_union":
        return RagFiltreUnion(embed_cache, dossier=current)
    if nom == "rag_filtre_entites":
        return RagFiltreEntites(embed_cache, dossier=current)
    if nom == "entites_lignes":
        return EntitesLignes(dossier=current, **rec)
    if nom == "entites_lignes_puis_rag":
        return EntitesLignesPuisRag(embed_cache, dossier=current, **rec)
    if nom == "rag_filtre_sens":
        return RagFiltreSens(embed_cache, dossier=current, **rec)
    if nom == "rag_filtre_ontologie":
        return RagFiltreOntologie(embed_cache, dossier=current)
    if nom == "objets_sens":
        return ObjetsSens(embed_cache, dossier=current, **rec)
    if nom == "objets_sens_puis_rag":
        return ObjetsSensPuisRag(embed_cache, dossier=current, **rec)
    if nom == "objets_sens_puis_rag_lecture_simple":
        return ObjetsSensPuisRagLectureSimple(embed_cache, dossier=current, **rec)
    if nom in {
        "objets_sens_puis_rag_lecture_simple_entites",
        "objets_sens_puis_rag_lecture_simple_entites_5000",
    }:
        part_ls = 5_000 if nom.endswith("_5000") else 3_000
        provider = ObjetsSensPuisRagLectureSimpleEntites(
            embed_cache,
            dossier=current,
            part_lecture_simple=part_ls,
            **rec,
        )
        provider.etiquette = nom
        return provider
    if nom == "desaccords_puis_rag":
        return DesaccordsPuisRag(embed_cache, dossier=current, **rec)
    if nom == "desaccords_puis_rag_filtre":
        return DesaccordsPuisRagFiltre(embed_cache, dossier=current, **rec)
    if nom == "desaccords_puis_g12":
        return DesaccordsPuisG12(embed_cache, dossier=current)
    if nom == "brain_prod":
        return BrainProd(embed_cache, dossier=current)
    if nom == "brain_banc":
        return BrainBanc(embed_cache, dossier=current)
    if nom == "brain_banc_v1":
        from .fournisseurs_contexte import BrainBancV1

        return BrainBancV1(embed_cache, dossier=current)
    if nom == "brain_banc_v2":
        from .fournisseurs_contexte import BrainBancV2

        return BrainBancV2(embed_cache, dossier=current)
    if nom == "brain_banc_v3":
        from .fournisseurs_contexte import BrainBancV3

        return BrainBancV3(embed_cache, dossier=current)
    if nom == "brain_banc_v4":
        from .fournisseurs_contexte import BrainBancV4

        return BrainBancV4(embed_cache, dossier=current)
    if nom == "brain_banc_calibre":
        return BrainBancCalibre(embed_cache, dossier=current)
    if nom == "brain_banc_g24":
        from .fournisseurs_contexte import BrainBancG24

        return BrainBancG24(embed_cache, dossier=current)
    if nom == "brain_banc_g36":
        from .fournisseurs_contexte import BrainBancG36

        return BrainBancG36(embed_cache, dossier=current)
    if nom == "brain_banc_g48":
        from .fournisseurs_contexte import BrainBancG48

        return BrainBancG48(embed_cache, dossier=current)
    if nom == "brain_mrfs":
        return BrainMrfs(embed_cache, dossier=current)
    if nom == "brain_mrfs_lexical":
        return BrainMrfsLexical(dossier=current)
    if nom == "brain_mrfs_g12":
        return BrainMrfsG12(embed_cache, dossier=current)
    if nom == "brain_situations":
        return BrainSituations(embed_cache, dossier=current)
    if nom == "brain_situations_g12":
        return BrainSituationsG12(embed_cache, dossier=current)
    if nom == "brain_situations_lettre":
        return BrainSituationsLettre(embed_cache, dossier=current)
    if nom == "brain_situations_lettre_g12":
        return BrainSituationsLettreG12(embed_cache, dossier=current)
    if nom == "mem0":
        return FournisseurFichier("mem0", dossier=current)
    if nom == "letta_archive":
        return FournisseurFichier(
            nom,
            comparatif_context_path(nom, dossier=current),
            dossier=current,
        )
    raise ValueError(f"fournisseur inconnu : {nom}")


def _render_style(nom: str) -> str:
    return "brain" if nom in BRAIN_FOURNISSEURS else "texte"


def _fournisseur_label(nom: str, k: int | None) -> str:
    if nom in FOURNISSEUR_LABELS:
        return FOURNISSEUR_LABELS[nom]
    if nom != "brain_prod":
        return nom
    effective_k = K_PROD if k is None else k
    if effective_k == K_PROD:
        return "brain_prod"
    return f"brain_prod (variante k={effective_k})"


def _disputes_denominators(
    dispute_rows: list[dict[str, Any]],
    *,
    reglage_ids: set[str],
    ecart_ids: set[str],
) -> dict[str, int]:
    reglage_rows = [row for row in dispute_rows if row["question_id"] in reglage_ids]
    ecart_rows = [row for row in dispute_rows if row["question_id"] in ecart_ids]
    return {
        "disputes_total": len(dispute_rows),
        "reglage_mesure": len(reglage_rows),
        "ecart_mesure": len(ecart_rows),
    }


def _appui_block_from_fields(
    rows: list[dict[str, Any]],
    *,
    tout_key: str,
    au_moins_key: str,
) -> dict[str, Any]:
    total = len(rows)
    tout_tient = sum(1 for row in rows if row.get(tout_key))
    au_moins_une = sum(1 for row in rows if row.get(au_moins_key))
    by_shape: dict[str, dict[str, int]] = {}
    for row in rows:
        shape = row["shape"]
        bucket = by_shape.setdefault(
            shape,
            {"questions": 0, "tout_tient": 0, "au_moins_une": 0},
        )
        bucket["questions"] += 1
        if row.get(tout_key):
            bucket["tout_tient"] += 1
        if row.get(au_moins_key):
            bucket["au_moins_une"] += 1
    return {
        "tout_tient_sur_141": tout_tient,
        "au_moins_une_sur_141": au_moins_une,
        "questions_total": total,
        "par_shape": by_shape,
    }


def _disputes_block_from_field(
    dispute_rows: list[dict[str, Any]],
    *,
    reglage_ids: set[str],
    ecart_ids: set[str],
    deux_cotes_key: str,
) -> dict[str, Any]:
    reglage_rows = [row for row in dispute_rows if row["question_id"] in reglage_ids]
    ecart_rows = [row for row in dispute_rows if row["question_id"] in ecart_ids]
    total = len(dispute_rows)
    return {
        "deux_cotes_sur_28": sum(1 for row in dispute_rows if row.get(deux_cotes_key)),
        "reglage_sur_14": sum(1 for row in reglage_rows if row.get(deux_cotes_key)),
        "ecart_sur_14": sum(1 for row in ecart_rows if row.get(deux_cotes_key)),
        "disputes_total": total,
        "reglage_total": len(reglage_ids),
        "ecart_total": len(ecart_ids),
    }


def _aggregate_config_rows_dual(
    rows: list[dict[str, Any]],
    dispute_rows: list[dict[str, Any]],
    *,
    reglage_ids: set[str],
    ecart_ids: set[str],
    dossier: Dossier | None = None,
) -> dict[str, Any]:
    current = dossier if dossier is not None else dossier_courant()
    mesurables = _rows_mesurables(rows)
    aggregate = _aggregate_config_rows(
        mesurables,
        dispute_rows,
        reglage_ids=reglage_ids,
        ecart_ids=ecart_ids,
    )
    meta = _meta_mesure(current, [{"shape": row["shape"]} for row in rows])
    denom = meta["denominateur_appui"]
    for block_name in ("appui",):
        block = aggregate[block_name]
        block["questions_total"] = denom
        block["tout_tient_sur_141"] = block["tout_tient_sur_141"]
        block[f"tout_tient_sur_{denom}"] = block["tout_tient_sur_141"]
        block[f"au_moins_une_sur_{denom}"] = block["au_moins_une_sur_141"]
    aggregate["appui_valeur"] = _appui_block_from_fields(
        mesurables,
        tout_key="appui_tout_tient_valeur",
        au_moins_key="appui_au_moins_une_valeur",
    )
    aggregate["appui_valeur"]["questions_total"] = denom
    aggregate["appui_valeur"][f"tout_tient_sur_{denom}"] = aggregate["appui_valeur"]["tout_tient_sur_141"]
    aggregate["appui_valeur"][f"au_moins_une_sur_{denom}"] = aggregate["appui_valeur"]["au_moins_une_sur_141"]
    aggregate["disputes_valeur"] = _disputes_block_from_field(
        dispute_rows,
        reglage_ids=reglage_ids,
        ecart_ids=ecart_ids,
        deux_cotes_key="deux_cotes_dans_contexte_valeur",
    )
    aggregate["disputes_valeur"].update(
        _disputes_denominators(
            dispute_rows,
            reglage_ids=reglage_ids,
            ecart_ids=ecart_ids,
        )
    )
    return aggregate


REFERENCE_LINES: tuple[tuple[str, int | None, int], ...] = (
    ("brain_prod", 8, 12000),
    ("rag_naif", 12, 12000),
    ("brain_banc", None, 12000),
    ("mem0", 20, 12000),
)
EXPECTED_VERBATIM: dict[str, int] = {
    "brain_prod": 39,
    "rag_naif": 85,
    "brain_banc": 84,
    "mem0": 2,
}
BIAS_NOTE = (
    "Note de biais : le mode valeur a été ajouté après observation du score mem0 (2/141). "
    "Le mode verbatim reste inchangé pour la comparabilité ; les deux colonnes sont toujours rapportées côte à côte."
)


def contextes_mesure_path(
    nom: str,
    dossier: Dossier | None = None,
    suffixe: str | None = None,
) -> Path:
    current = dossier if dossier is not None else dossier_courant()
    return current.comparatif_dir / _contextes_jsonl_filename(nom, suffixe)


def _signature_elements(elements: list[dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        (
            item.get("kind"),
            str(item.get("id") or ""),
            str(item.get("source") or ""),
            str(item.get("text") or "")[:240],
        )
        for item in elements
    )


def _verdict_tout_tient(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    mode_presence: Literal["verbatim", "valeur"],
) -> bool:
    supports = question.get("supporting_assertions") or []
    if not supports:
        return False
    summary = _appui_summary(
        question,
        evidence,
        assertion_index=assertion_index,
        claims_index=claims_index,
        mode_presence=mode_presence,
    )
    return len(summary["tient"]) == len(supports)


def _mem0_fichiers_k(ks: list[int]) -> set[int]:
    return {20 if k <= 20 else k for k in ks}


def _copier_contextes_mem0(nom: str, *, dossier: Dossier, ks: list[int] | None = None) -> Path:
    fichiers_k = _mem0_fichiers_k(ks or [20])
    dest = contextes_mesure_path(nom, dossier=dossier)
    dest.parent.mkdir(parents=True, exist_ok=True)
    primary = comparatif_context_path(nom, dossier=dossier, k=20)
    for fichier_k in sorted(fichiers_k):
        source = comparatif_context_path(nom, dossier=dossier, k=fichier_k)
        if not source.is_file():
            raise FileNotFoundError(f"contextes source absents : {source}")
        if fichier_k == 20 and source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        elif fichier_k != 20:
            sidecar = dest.parent / f"contextes-{nom}-k{fichier_k}.jsonl"
            if source.resolve() != sidecar.resolve():
                shutil.copy2(source, sidecar)
    return primary


def ecrire_contextes_mesure(
    nom: str,
    *,
    ks: list[int | None],
    questions: list[dict[str, Any]],
    elements_par_question_k: dict[tuple[str, int | None], list[dict[str, Any]]],
    dossier: Dossier | None = None,
    suffixe: str | None = None,
) -> Path:
    current = dossier if dossier is not None else dossier_courant()
    path = contextes_mesure_path(nom, dossier=current, suffixe=suffixe)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "meta": {
                    "fournisseur": nom,
                    "date": date.today().isoformat(),
                    "ks": [k for k in ks if k is not None],
                }
            },
            ensure_ascii=False,
        )
    ]
    for question in questions:
        qid = question["id"]
        for k in ks:
            elements = elements_par_question_k.get((qid, k))
            if elements is None:
                continue
            lines.append(
                json.dumps(
                    {"question_id": qid, "k": k, "elements": elements},
                    ensure_ascii=False,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def lire_contextes_mesure_bruts(
    nom: str,
    *,
    dossier: Dossier | None = None,
    suffixe: str | None = None,
) -> dict[tuple[str, int | None], list[dict[str, Any]]]:
    current = dossier if dossier is not None else dossier_courant()
    path = contextes_mesure_path(nom, dossier=current, suffixe=suffixe)
    if not path.is_file():
        raise FileNotFoundError(
            f"contextes persistés absents pour {nom} : {path} — rejouer la mesure étage 2"
        )
    index: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = 0
    if lines:
        first = json.loads(lines[0])
        if "meta" in first:
            start = 1
    for line in lines[start:]:
        row = json.loads(line)
        qid = str(row["question_id"])
        k = row.get("k")
        index[(qid, k)] = list(row.get("elements") or [])
    return index


def contextes_mesure_apres_budget(
    nom: str,
    questions: list[dict[str, Any]],
    *,
    k: int | None,
    budget: int,
    dossier: Dossier | None = None,
    suffixe: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    from .fournisseurs_contexte import _elements_to_evidence

    current = dossier if dossier is not None else dossier_courant()
    bruts = lire_contextes_mesure_bruts(nom, dossier=current, suffixe=suffixe)
    style = _render_style(nom)
    effective_k = k_effectif_fournisseur(nom, k)
    contexts: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        qid = question["id"]
        elements = bruts.get((qid, k))
        if elements is None:
            elements = bruts.get((qid, None))
        if elements is None:
            elements = bruts.get((qid, effective_k))
        if elements is None:
            raise KeyError(
                f"contextes {nom} incomplets : {qid} k={k} absent de "
                f"{contextes_mesure_path(nom, dossier=current, suffixe=suffixe)}"
            )
        if nom in ("mem0", "letta_archive") and elements and not elements[0].get("kind"):
            elements = _elements_to_evidence(elements[:effective_k])
        elif nom in K_FIXE:
            elements = list(elements)
        else:
            elements = list(elements[:effective_k])
        _rendered, kept = rendre_contexte(
            elements,
            budget_chars=budget,
            question_id=qid,
            style=style,
            dossier=current,
        )
        contexts[qid] = kept
    return contexts


def _detecter_instabilite_question(
    question: dict[str, Any],
    runs: list[list[dict[str, Any]]],
    *,
    nom: str,
    k: int | None,
    budget: int,
    style: str,
    dossier: Dossier,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
) -> list[str]:
    if len(runs) < 2:
        return []
    effective_k = 12 if k is None else k

    def _slice(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if nom in K_FIXE:
            return list(elements)
        return list(elements[:effective_k])

    def _kept(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sliced = _slice(elements)
        _rendered, kept = rendre_contexte(
            sliced,
            budget_chars=budget,
            question_id=question["id"],
            style=style,
            dossier=dossier,
        )
        return kept

    reference = runs[0]
    changes: list[str] = []
    ref_sig = _signature_elements(_slice(reference))
    ref_kept = _kept(reference)
    for index, candidate in enumerate(runs[1:], start=2):
        if _signature_elements(_slice(candidate)) != ref_sig:
            changes.append(f"éléments (exécution {index})")
        for mode in ("verbatim", "valeur"):
            if (
                _verdict_tout_tient(
                    question,
                    ref_kept,
                    assertion_index=assertion_index,
                    claims_index=claims_index,
                    mode_presence=mode,
                )
                != _verdict_tout_tient(
                    question,
                    _kept(candidate),
                    assertion_index=assertion_index,
                    claims_index=claims_index,
                    mode_presence=mode,
                )
            ):
                changes.append(f"verdict {mode} (exécution {index})")
    return sorted(set(changes))


def mesurer_stabilite_fournisseur(
    nom: str,
    *,
    connection,
    ks: list[int],
    budgets: list[int],
    questions: list[dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    embed_cache: dict[str, list[float]],
    stabilite: int,
    dossier: Dossier | None = None,
) -> dict[str, Any]:
    current = dossier if dossier is not None else dossier_courant()
    if nom in ("mem0", "letta_archive"):
        total = len(selection_questions_mesure(questions))
        return {
            "instables": 0,
            "total": total,
            "par_mode": {"verbatim": "0/0", "valeur": "0/0"},
            "details": [],
        }
    provider = _instancier(nom, embed_cache, dossier=current)
    style = _render_style(nom)
    mesurables = selection_questions_mesure(questions)
    total = len(mesurables)
    effective_ks = [None] if nom in K_FIXE else ks
    reference_budget = max(budgets)
    instables: set[str] = set()
    instables_verbatim: set[str] = set()
    instables_valeur: set[str] = set()
    details: list[dict[str, Any]] = []
    for k in effective_ks:
        effective_k = k_effectif_fournisseur(nom, k)
        for question in mesurables:
            runs: list[list[dict[str, Any]]] = []
            for _ in range(stabilite):
                runs.append(provider.contexte(question, k=effective_k, connection=connection))
            changes = _detecter_instabilite_question(
                question,
                runs,
                nom=nom,
                k=k,
                budget=reference_budget,
                style=style,
                dossier=current,
                assertion_index=assertion_index,
                claims_index=claims_index,
            )
            if not changes:
                continue
            instables.add(question["id"])
            if any(item.startswith("verdict verbatim") or item.startswith("éléments") for item in changes):
                instables_verbatim.add(question["id"])
            if any(item.startswith("verdict valeur") or item.startswith("éléments") for item in changes):
                instables_valeur.add(question["id"])
            details.append(
                {
                    "question_id": question["id"],
                    "k": k,
                    "budget": reference_budget,
                    "changements": changes,
                }
            )
    return {
        "instables": len(instables),
        "total": total,
        "par_mode": {
            "verbatim": f"{len(instables_verbatim)}/{total}",
            "valeur": f"{len(instables_valeur)}/{total}",
        },
        "details": details,
    }


def _format_section_reproductibilite(stabilite_par_fournisseur: dict[str, dict[str, Any]]) -> str:
    if not stabilite_par_fournisseur:
        return ""
    lines = [
        "",
        "## Reproductibilité",
        "",
        "Deux exécutions de recherche par fournisseur ; la première alimente les JSON et les contextes.",
        "",
        "| Fournisseur | mode | instables / total |",
        "| --- | --- | --- |",
    ]
    for nom in sorted(stabilite_par_fournisseur):
        payload = stabilite_par_fournisseur[nom]
        for mode, ratio in payload["par_mode"].items():
            lines.append(f"| {nom} | {mode} | {ratio} |")
    details = [
        item
        for payload in stabilite_par_fournisseur.values()
        for item in payload.get("details") or []
    ]
    if details:
        lines.extend(["", "### Questions instables", ""])
        for item in sorted(details, key=lambda row: (row["question_id"], str(row.get("k")))):
            changes = ", ".join(item["changements"])
            lines.append(
                f"- {item['question_id']} (k={item.get('k')}, budget={item.get('budget')}) : {changes}"
            )
    return "\n".join(lines)


def _compile_row(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    rendered: str,
    *,
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    holder_table: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    appui_verbatim = _appui_summary(
        question,
        evidence,
        assertion_index=assertion_index,
        claims_index=claims_index,
        mode_presence="verbatim",
    )
    appui_valeur = _appui_summary(
        question,
        evidence,
        assertion_index=assertion_index,
        claims_index=claims_index,
        mode_presence="valeur",
    )
    supports = question.get("supporting_assertions") or []
    present = _make_present_fn(evidence, assertion_index, claims_index, "plancher")
    verdict = compute_verdict(
        question,
        evidence,
        gold=gold,
        holder_table=holder_table,
        claims_index=claims_index,
        present=present,
    )
    row: dict[str, Any] = {
        "question_id": question["id"],
        "shape": question.get("shape") or "unknown",
        "evidence_count": len(evidence),
        "caracteres": len(rendered),
        "tokens_estimes": round(len(rendered) / 4, 2),
        "appui_tout_tient": bool(supports) and len(appui_verbatim["tient"]) == len(supports),
        "appui_au_moins_une": bool(appui_verbatim["tient"]),
        "appui_tout_tient_valeur": bool(supports) and len(appui_valeur["tient"]) == len(supports),
        "appui_au_moins_une_valeur": bool(appui_valeur["tient"]),
        "verdict": verdict.get("verdict"),
    }
    if question.get("shape") in ("dispute", "contradiction"):
        sides_verbatim = _side_presence(
            question, evidence, gold=gold, claims_index=claims_index, mode_presence="verbatim"
        )
        sides_valeur = _side_presence(
            question, evidence, gold=gold, claims_index=claims_index, mode_presence="valeur"
        )
        row["deux_cotes_dans_contexte"] = all(s["dans_contexte"] for s in sides_verbatim) if sides_verbatim else False
        row["deux_cotes_dans_contexte_valeur"] = (
            all(s["dans_contexte"] for s in sides_valeur) if sides_valeur else False
        )
    return row


def mesurer_fournisseur(
    nom: str,
    *,
    connection,
    ks: list[int],
    budgets: list[int],
    questions: list[dict[str, Any]],
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    holder_table: dict[str, list[dict[str, Any]]],
    embed_cache: dict[str, list[float]],
    reglage_ids: set[str],
    ecart_ids: set[str],
    dossier: Dossier | None = None,
    suffixe: str | None = None,
    reconstitution_dir: Path | None = None,
    embeddings_objets: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    current = dossier if dossier is not None else dossier_courant()
    if nom == "mem0":
        _copier_contextes_mem0(nom, dossier=current, ks=ks)
    elif nom == "letta_archive":
        _copier_contextes_mem0(nom, dossier=current)
    provider = _instancier(
        nom,
        embed_cache,
        dossier=current,
        reconstitution_dir=reconstitution_dir,
        embeddings_objets=embeddings_objets,
    )
    style = _render_style(nom)
    effective_ks = [None] if nom in K_FIXE else ks
    elements_par_question_k: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    if nom not in ("mem0", "letta_archive"):
        for k in effective_ks:
            effective_k = k_effectif_fournisseur(nom, k)
            for question in questions:
                elements = provider.contexte(question, k=effective_k, connection=connection)
                elements_par_question_k[(question["id"], k)] = list(elements)
        ecrire_contextes_mesure(
            nom,
            ks=effective_ks,
            questions=questions,
            elements_par_question_k=elements_par_question_k,
            dossier=current,
            suffixe=suffixe,
        )
    runs: list[dict[str, Any]] = []
    for budget in budgets:
        for k in effective_ks:
            rows: list[dict[str, Any]] = []
            for question in questions:
                effective_k = k_effectif_fournisseur(nom, k)
                if nom in ("mem0", "letta_archive"):
                    elements = provider.contexte(question, k=effective_k, connection=connection)
                else:
                    elements = elements_par_question_k[(question["id"], k)]
                rendered, kept = rendre_contexte(
                    elements,
                    budget_chars=budget,
                    question_id=question["id"],
                    style=style,
                    dossier=current,
                )
                rows.append(
                    _compile_row(
                        question,
                        kept,
                        rendered,
                        gold=gold,
                        claims_index=claims_index,
                        assertion_index=assertion_index,
                        holder_table=holder_table,
                    )
                )
            meta = _meta_mesure(current, questions)
            dispute_rows = [row for row in rows if row["shape"] == meta["dispute_shape"]]
            aggregate = _aggregate_config_rows_dual(
                rows,
                dispute_rows,
                reglage_ids=reglage_ids,
                ecart_ids=ecart_ids,
                dossier=current,
            )
            aggregate["disputes"].update(
                _disputes_denominators(
                    dispute_rows,
                    reglage_ids=reglage_ids,
                    ecart_ids=ecart_ids,
                )
            )
            chars = [int(row["caracteres"]) for row in rows]
            tokens = [float(row["tokens_estimes"]) for row in rows]
            aggregate["rendu"] = {
                "caracteres": {
                    "mediane": int(statistics.median(chars)) if chars else 0,
                    "p90": _percentile(chars, 90),
                },
                "tokens": {
                    "mediane": round(statistics.median(tokens), 2) if tokens else 0.0,
                    "p90": round(_percentile([int(t) for t in tokens], 90), 2),
                },
            }
            runs.append(
                {
                    "k": k,
                    "k_effectif": k_effectif_fournisseur(nom, k if k is not None else effective_k),
                    "budget": budget,
                    "questions": rows,
                    "aggregate": aggregate,
                }
            )
    payload: dict[str, Any] = {
        "fournisseur": nom,
        "k_fixe": nom in K_FIXE,
        "date": date.today().isoformat(),
        "runs": runs,
    }
    if isinstance(provider, FournisseurFichier) and provider.meta:
        payload["extracteur"] = provider.meta.get("extracteur")
        payload["modele"] = provider.meta.get("modele")
    out_path = etage2_resultat_path(nom, current, suffixe=suffixe)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _format_table_synthese(
    results: list[dict[str, Any]],
    *,
    dossier: Dossier,
    meta: dict[str, Any],
) -> str:
    denom = meta["denominateur_appui"]
    pct = BUDGET_CORPUS_PCT.get(dossier.nom, 0.0)
    lines = [
        "",
        "## Synthèse — budgets 4 000 / 8 000 / 12 000",
        "",
    ]
    if meta["non_etabli"]:
        lines.append(
            f"{meta['questions_mesurables']} questions mesurables à l'étage 2, "
            f"{meta['non_etabli']} `non_etabli` réservées à l'étage 3."
        )
        lines.append("")
    lines.extend(
        [
            f"Budget 12 000 caractères = {pct} % du corpus {dossier.responsibility_context}.",
            "",
            "| Fournisseur | k | budget | car. médiane | tout tient (verbatim) | tout tient (valeur) | ≥1 (verbatim) | ≥1 (valeur) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    budgets = sorted({run["budget"] for result in results for run in result["runs"]})
    for budget in budgets:
        for result in results:
            run = next((item for item in result["runs"] if item["budget"] == budget), None)
            if run is None:
                continue
            k_label = "figé" if result["k_fixe"] else str(run["k"])
            fournisseur = _fournisseur_label(result["fournisseur"], run["k"])
            appui = run["aggregate"]["appui"]
            appui_v = run["aggregate"]["appui_valeur"]
            tout_v = appui.get(f"tout_tient_sur_{denom}", appui["tout_tient_sur_141"])
            tout_val = appui_v.get(f"tout_tient_sur_{denom}", appui_v["tout_tient_sur_141"])
            au_moins_v = appui.get(f"au_moins_une_sur_{denom}", appui["au_moins_une_sur_141"])
            au_moins_val = appui_v.get(f"au_moins_une_sur_{denom}", appui_v["au_moins_une_sur_141"])
            med = run["aggregate"]["rendu"]["caracteres"]["mediane"]
            lines.append(
                f"| {fournisseur} | {k_label} | {budget} | {med} | "
                f"{tout_v}/{denom} | {tout_val}/{denom} | "
                f"{au_moins_v}/{denom} | {au_moins_val}/{denom} |"
            )
    return "\n".join(lines)


def _format_table_par_forme(
    results: list[dict[str, Any]],
    *,
    run_limite: int | None = None,
    dossier: Dossier | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    shapes: set[str] = set()
    for result in results:
        for run in result["runs"]:
            shapes.update(run["aggregate"]["appui"]["par_shape"].keys())
            shapes.update(run["aggregate"]["appui_valeur"]["par_shape"].keys())
    lines: list[str] = []
    if run_limite is not None:
        lines.extend(
            [
                f"RUN LIMITÉ À {run_limite} QUESTIONS — pas un résultat du banc",
                "",
            ]
        )
    current = dossier if dossier is not None else dossier_courant()
    measure_meta = meta or _meta_mesure(current, [])
    lines.extend(
        [
            f"# Rapport comparatif étage 2 — {current.responsibility_context}",
            "",
            BIAS_NOTE,
            "",
            f"Date : {date.today().isoformat()}",
        ]
    )
    if measure_meta["non_etabli"]:
        lines.append(
            f"{measure_meta['questions_mesurables']} questions mesurables à l'étage 2, "
            f"{measure_meta['non_etabli']} `non_etabli` réservées à l'étage 3."
        )
    lines.extend(
        [
            "",
            "## Par forme — tout tient / au moins une",
            "",
            "| Fournisseur | k | budget | forme | tout tient (verbatim) | tout tient (valeur) | ≥1 (verbatim) | ≥1 (valeur) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        for run in result["runs"]:
            k_label = "figé" if result["k_fixe"] else str(run["k"])
            fournisseur = _fournisseur_label(result["fournisseur"], run["k"])
            for shape in sorted(shapes):
                cell_v = run["aggregate"]["appui"]["par_shape"].get(shape, {})
                cell_val = run["aggregate"]["appui_valeur"]["par_shape"].get(shape, {})
                lines.append(
                    f"| {fournisseur} | {k_label} | {run['budget']} | {shape} | "
                    f"{cell_v.get('tout_tient', 0)}/{cell_v.get('questions', 0)} | "
                    f"{cell_val.get('tout_tient', 0)}/{cell_val.get('questions', 0)} | "
                    f"{cell_v.get('au_moins_une', 0)}/{cell_v.get('questions', 0)} | "
                    f"{cell_val.get('au_moins_une', 0)}/{cell_val.get('questions', 0)} |"
                )
    return "\n".join(lines)


def _find_reference_run(result: dict[str, Any], k: int | None, budget: int) -> dict[str, Any] | None:
    for run in result["runs"]:
        if run["budget"] != budget:
            continue
        if result["k_fixe"]:
            return run
        if run["k"] == k:
            return run
    return None


def _format_table_reference(
    results: list[dict[str, Any]],
    *,
    dossier: Dossier,
    meta: dict[str, Any],
) -> str:
    by_name = {result["fournisseur"]: result for result in results}
    denom = meta["denominateur_appui"]
    titre = (
        f"## Référence {dossier.responsibility_context} (budget 12 000)"
        if dossier.nom != "greenloop"
        else "## Référence GreenLoop (budget 12 000)"
    )
    lines = [
        "",
        titre,
        "",
        "| Fournisseur | k | tout tient (verbatim) | tout tient (valeur) | ≥1 (verbatim) | ≥1 (valeur) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for nom, k, budget in REFERENCE_LINES:
        result = by_name.get(nom)
        if result is None:
            lines.append(f"| {nom} | {k or 'figé'} | — | — | — | — |")
            continue
        run = _find_reference_run(result, k, budget)
        if run is None:
            lines.append(f"| {nom} | {k or 'figé'} | — | — | — | — |")
            continue
        appui = run["aggregate"]["appui"]
        appui_v = run["aggregate"]["appui_valeur"]
        k_label = "figé" if result["k_fixe"] else str(k)
        tout_v = appui.get(f"tout_tient_sur_{denom}", appui["tout_tient_sur_141"])
        tout_val = appui_v.get(f"tout_tient_sur_{denom}", appui_v["tout_tient_sur_141"])
        au_moins_v = appui.get(f"au_moins_une_sur_{denom}", appui["au_moins_une_sur_141"])
        au_moins_val = appui_v.get(f"au_moins_une_sur_{denom}", appui_v["au_moins_une_sur_141"])
        lines.append(
            f"| {nom} | {k_label} | "
            f"{tout_v}/{denom} | "
            f"{tout_val}/{denom} | "
            f"{au_moins_v}/{denom} | "
            f"{au_moins_val}/{denom} |"
        )
    return "\n".join(lines)


def _format_table_disputes(
    results: list[dict[str, Any]],
    *,
    dossier: Dossier,
    meta: dict[str, Any],
) -> str:
    total = meta["disputes_total"]
    if dossier.nom != "greenloop":
        lines = [
            "",
            f"## Contradictions {dossier.responsibility_context}",
            "",
            "MOREAU n'a pas servi au réglage — écart réglage/écart non rapporté (ADR 0023 §6).",
            "",
            "| Fournisseur | k | budget | deux_cotes (verbatim) | un_seul_cote (verbatim) | aucun (verbatim) | deux_cotes (valeur) | un_seul_cote (valeur) | aucun (valeur) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for result in results:
            for run in result["runs"]:
                k_label = "figé" if result["k_fixe"] else str(run["k"])
                fournisseur = _fournisseur_label(result["fournisseur"], run["k"])
                counts_v = {"deux_cotes": 0, "un_seul_cote": 0, "aucun": 0}
                counts_val = {"deux_cotes": 0, "un_seul_cote": 0, "aucun": 0}
                for row in run["questions"]:
                    if row["shape"] != meta["dispute_shape"]:
                        continue
                    if row.get("deux_cotes_dans_contexte"):
                        counts_v["deux_cotes"] += 1
                    elif row.get("appui_au_moins_une"):
                        counts_v["un_seul_cote"] += 1
                    else:
                        counts_v["aucun"] += 1
                    if row.get("deux_cotes_dans_contexte_valeur"):
                        counts_val["deux_cotes"] += 1
                    elif row.get("appui_au_moins_une_valeur"):
                        counts_val["un_seul_cote"] += 1
                    else:
                        counts_val["aucun"] += 1
                lines.append(
                    f"| {fournisseur} | {k_label} | {run['budget']} | "
                    f"{counts_v['deux_cotes']}/{total} | {counts_v['un_seul_cote']}/{total} | {counts_v['aucun']}/{total} | "
                    f"{counts_val['deux_cotes']}/{total} | {counts_val['un_seul_cote']}/{total} | {counts_val['aucun']}/{total} |"
                )
        return "\n".join(lines)

    lines = [
        "",
        "## Disputes — réglage / écart",
        "",
        "| Fournisseur | k | budget | 2 côtés (verbatim) | 2 côtés (valeur) | réglage (verbatim) | réglage (valeur) | écart (verbatim) | écart (valeur) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        for run in result["runs"]:
            k_label = "figé" if result["k_fixe"] else str(run["k"])
            fournisseur = _fournisseur_label(result["fournisseur"], run["k"])
            disputes = run["aggregate"]["disputes"]
            disputes_v = run["aggregate"]["disputes_valeur"]
            reglage_total = disputes.get("reglage_mesure", 0)
            ecart_total = disputes.get("ecart_mesure", 0)
            lines.append(
                f"| {fournisseur} | {k_label} | {run['budget']} | "
                f"{disputes['deux_cotes_sur_28']}/{total} | "
                f"{disputes_v['deux_cotes_sur_28']}/{total} | "
                f"{disputes['reglage_sur_14']}/{reglage_total} | "
                f"{disputes_v['reglage_sur_14']}/{reglage_total} | "
                f"{disputes['ecart_sur_14']}/{ecart_total} | "
                f"{disputes_v['ecart_sur_14']}/{ecart_total} |"
            )
    return "\n".join(lines)


def _format_table_couts(dossier: Dossier | None = None) -> str:
    from .tiers_mem0 import Mem0Chemins

    current = dossier if dossier is not None else dossier_courant()
    chemins = Mem0Chemins.pour(current)
    lines = [
        "",
        "## Coûts (estimations / journaux)",
        "",
        "| Poste | Source | Détail |",
        "| --- | --- | --- |",
    ]
    # Un fichier d'estimation par extracteur (revue Codex #123) ; l'ancien nom sans suffixe
    # n'est lu qu'à défaut.
    candidats = [chemins.sortie_a_blanc(ext) for ext in ("mistral", "cursor")]
    if current.nom == "greenloop":
        candidats.append(chemins.racine / "ingestion-mem0.a-blanc.json")
    for a_blanc in (c for c in candidats if c.is_file()):
        payload = json.loads(a_blanc.read_text(encoding="utf-8"))
        extracteur = payload.get("extracteur") or a_blanc.stem.replace("ingestion-mem0", "").strip("-") or "?"
        cout_ingestion = payload.get("cout_ingestion_eur")
        if cout_ingestion is None:
            detail_ingestion = payload.get("cout") or (
                f"quota Cursor, {payload.get('appels_prevus')} appels"
            )
        else:
            detail_ingestion = f"{payload.get('appels_prevus')} appels · {cout_ingestion} €"
        lines.append(f"| mem0 ingestion ({extracteur}) | a-blanc | {detail_ingestion} |")
        lines.append(
            f"| mem0 contextes ({extracteur}) | a-blanc | {payload.get('appels_contextes_prevus')} embeddings · "
            f"{payload.get('cout_contextes_eur')} € |"
        )
    lines.extend(
        [
            "",
            "Notes :",
            "- `brain_banc` : k ignoré (configuration `hybride_fenetres_g12`, arrivé tel quel).",
            "- `brain_banc (fenêtre calibrée sur l'or)` : k ignoré "
            "(configuration `hybride_fenetres_calibrees`, cache séparé).",
            "- `Brain banc (g24/g36/g48, réglé sur le budget)` : k ignoré "
            "(configurations `hybride_fenetres_g24` / `g36` / `g48` ; "
            "réglée sur le budget, pas comparable au réglage GreenLoop d'origine).",
            f"- `brain_prod` : ligne de production à k={K_PROD} ; autres k = variantes contre-factuelles.",
            "- mem0 / Letta archive : requête = texte de la question tel quel.",
            "",
        ]
    )
    return "\n".join(lines)


def _charger_resultats_comparatif_complets(
    dossier: Dossier,
    results: list[dict[str, Any]],
    suffixe: str | None = None,
) -> list[dict[str, Any]]:
    by_name = {result["fournisseur"]: result for result in results}
    suffixe_tag = f"@{suffixe}" if suffixe else None
    for path in sorted(dossier.comparatif_dir.glob("etage2-*.json")):
        stem = path.stem.replace("etage2-", "", 1)
        if suffixe_tag:
            if not stem.endswith(suffixe_tag):
                continue
            nom = stem[: -len(suffixe_tag)]
        elif "@" in stem:
            continue
        else:
            nom = stem
        if nom in by_name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("fournisseur") == nom:
            by_name[nom] = payload
    order = [
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
    ]
    ordered = [by_name[nom] for nom in order if nom in by_name]
    extras = sorted(nom for nom in by_name if nom not in order)
    ordered.extend(by_name[nom] for nom in extras)
    return ordered


def _run_reference(
    results: list[dict[str, Any]],
    nom: str,
    *,
    budget: int = 12000,
    k: int | None = None,
) -> dict[str, Any] | None:
    result = next((item for item in results if item["fournisseur"] == nom), None)
    if result is None:
        return None
    return _find_reference_run(result, k, budget)


def _meilleur_run_pour_budget(
    result: dict[str, Any],
    budget_target: int,
) -> dict[str, Any] | None:
    candidates = [run for run in result["runs"] if int(run["budget"]) == budget_target]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda run: abs(
            int(run["aggregate"]["rendu"]["caracteres"]["mediane"]) - budget_target
        ),
    )


def _plafond_rendu_fournisseur(result: dict[str, Any]) -> int | None:
    budgets = sorted({int(run["budget"]) for run in result["runs"]})
    medians: list[tuple[int, int]] = []
    for budget in budgets:
        run = _meilleur_run_pour_budget(result, budget)
        if run is None:
            continue
        medians.append(
            (budget, int(run["aggregate"]["rendu"]["caracteres"]["mediane"]))
        )
    if not medians:
        return None
    plafond = medians[0][0]
    prev_med = medians[0][1]
    for budget, med in medians[1:]:
        if med > prev_med:
            plafond = budget
            prev_med = med
        else:
            break
    return plafond


def _cellule_grille_budget_egal(
    run: dict[str, Any] | None,
    *,
    denom: int,
    temoin_lignes: list[dict[str, Any]] | None,
    fournisseur: str,
) -> dict[str, Any]:
    if run is None:
        return {
            "k": None,
            "verbatim": None,
            "valeur": None,
            "temoin": None,
            "ecart": None,
            "caracteres": None,
        }
    appui = run["aggregate"]["appui"]
    appui_v = run["aggregate"]["appui_valeur"]
    tout_v = appui.get(f"tout_tient_sur_{denom}", appui["tout_tient_sur_141"])
    tout_val = appui_v.get(f"tout_tient_sur_{denom}", appui_v["tout_tient_sur_141"])
    temoin_row = next(
        (
            row
            for row in temoin_lignes or []
            if row["fournisseur"] == fournisseur
            and row["mode"] == "verbatim"
            and row.get("budget") is not None
            and int(row["budget"]) == int(run["budget"])
            and (
                row.get("k") == run.get("k")
                or (row.get("k") is None and run.get("k") is None)
            )
        ),
        None,
    )
    temoin = int(temoin_row["temoin"]) if temoin_row else None
    ecart = int(tout_v) - temoin if temoin is not None else None
    k_label = run.get("k_effectif") if run.get("k") is None else run.get("k")
    return {
        "k": k_label,
        "verbatim": int(tout_v),
        "valeur": int(tout_val),
        "temoin": temoin,
        "ecart": ecart,
        "caracteres": int(run["aggregate"]["rendu"]["caracteres"]["mediane"]),
    }


def construire_grille_budget_egal(
    results: list[dict[str, Any]],
    *,
    budgets: list[int],
    dossier: Dossier,
    temoin_lignes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    meta = _meta_mesure(dossier, [])
    denom = meta["denominateur_appui"]
    grille: dict[str, dict[str, list[Any]]] = {}
    plafonds: dict[str, int | None] = {}
    for result in results:
        nom = result["fournisseur"]
        plafonds[nom] = _plafond_rendu_fournisseur(result)
        grille[nom] = {}
        for budget in budgets:
            run = _meilleur_run_pour_budget(result, budget)
            cell = _cellule_grille_budget_egal(
                run,
                denom=denom,
                temoin_lignes=temoin_lignes,
                fournisseur=nom,
            )
            grille[nom][str(budget)] = [
                cell["verbatim"],
                cell["valeur"],
                cell["temoin"],
                cell["ecart"],
                cell["caracteres"],
            ]
    return {"grille": grille, "plafonds_rendu": plafonds}


def _format_section_comparaison_budget_egal(
    results: list[dict[str, Any]],
    *,
    dossier: Dossier,
    budgets: list[int],
    temoin_lignes: list[dict[str, Any]] | None = None,
) -> str:
    payload = construire_grille_budget_egal(
        results,
        budgets=budgets,
        dossier=dossier,
        temoin_lignes=temoin_lignes,
    )
    meta = _meta_mesure(dossier, [])
    denom = meta["denominateur_appui"]
    lines = [
        "",
        "## Comparaison à unités et budget égaux",
        "",
        "À chaque budget cible : k ou variante dont la médiane de caractères rendus est la plus proche.",
        "",
        "| Budget | Fournisseur | k / variante | tout tient (verbatim) | tout tient (valeur) | "
        "témoin | écart | car. médiane |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for budget in budgets:
        for result in results:
            nom = result["fournisseur"]
            run = _meilleur_run_pour_budget(result, budget)
            cell = _cellule_grille_budget_egal(
                run,
                denom=denom,
                temoin_lignes=temoin_lignes,
                fournisseur=nom,
            )
            k_label = "figé" if result["k_fixe"] else str(cell["k"])
            tout_v = f"{cell['verbatim']}/{denom}" if cell["verbatim"] is not None else "—"
            tout_val = f"{cell['valeur']}/{denom}" if cell["valeur"] is not None else "—"
            temoin = str(cell["temoin"]) if cell["temoin"] is not None else "—"
            ecart = str(cell["ecart"]) if cell["ecart"] is not None else "—"
            chars = str(cell["caracteres"]) if cell["caracteres"] is not None else "—"
            label = _fournisseur_label(nom, run.get("k") if run else None)
            lines.append(
                f"| {budget} | {label} | {k_label} | {tout_v} | {tout_val} | "
                f"{temoin} | {ecart} | {chars} |"
            )
    lines.extend(["", "### Plafond de rendu", ""])
    for nom, plafond in sorted(payload["plafonds_rendu"].items()):
        label = _fournisseur_label(nom, None)
        lines.append(
            f"- {label} : budget {plafond if plafond is not None else '—'} "
            f"(au-delà, la médiane de caractères rendus ne croît plus)"
        )
    lines.append("")
    return "\n".join(lines)


def _format_section_plafond_forme(
    results: list[dict[str, Any]],
    *,
    dossier: Dossier,
    meta: dict[str, Any],
    temoin_lignes: list[dict[str, Any]] | None = None,
) -> str:
    if dossier.nom != "moreau":
        return ""
    from .embarquer_phrases import load_fenetre_calibree

    try:
        calibration = load_fenetre_calibree(dossier)
    except RuntimeError:
        return ""

    denom = meta["denominateur_appui"]
    budget = 12000
    origine = _run_reference(results, "brain_banc", budget=budget)
    calibre = _run_reference(results, "brain_banc_calibre", budget=budget)
    if origine is None and calibre is None:
        return ""

    temoin_by_key = {
        (row["fournisseur"], row["mode"]): row for row in temoin_lignes or []
    }

    def _cell(nom: str, mode: str, field: str) -> str:
        run = _run_reference(results, nom, budget=budget)
        if run is None:
            return "—"
        if field == "mediane":
            return str(run["aggregate"]["rendu"]["caracteres"]["mediane"])
        block = run["aggregate"]["appui" if mode == "verbatim" else "appui_valeur"]
        key = f"tout_tient_sur_{denom}"
        value = block.get(key, block["tout_tient_sur_141"])
        return f"{value}/{denom}"

    def _temoin(nom: str, mode: str) -> str:
        row = temoin_by_key.get((nom, mode))
        if row is None:
            return "—"
        return f"{row['temoin']}/{row['total']}"

    lines = [
        "",
        "## Plafond de forme",
        "",
        "Réglée sur le dossier mesuré, à lire comme un plafond de forme, pas comme une performance comparable.",
        "",
        f"- Fenêtre d'origine : **2 phrases** — couverture citations {calibration.get('fenetre_origine_2', '26/42')}",
        f"- Fenêtre calibrée : **{calibration['fenetre_phrases']} phrase(s)** — "
        f"couverture citations {calibration['citations_dans_une_fenetre']}",
        f"- Calcul : `{calibration.get('calcul', '')}`",
        "",
    ]
    if calibration.get("citations_manquantes_detail"):
        lines.append("Citations hors fenêtre calibrée :")
        for item in calibration["citations_manquantes_detail"]:
            lines.append(f"- {item['id']} : {item['motif']}")
        lines.append("")

    med_origine = _cell("brain_banc", "verbatim", "mediane")
    med_calibre = _cell("brain_banc_calibre", "verbatim", "mediane")
    lines.extend(
        [
            f"- Caractères médians rendus (budget {budget}) : origine {med_origine}, "
            f"calibrée {med_calibre}",
            "",
            f"| Ligne | k | budget | tout tient (verbatim) | témoin (verbatim) | "
            f"tout tient (valeur) | témoin (valeur) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            f"| brain_banc | figé | {budget} | "
            f"{_cell('brain_banc', 'verbatim', 'tout')} | {_temoin('brain_banc', 'verbatim')} | "
            f"{_cell('brain_banc', 'valeur', 'tout')} | {_temoin('brain_banc', 'valeur')} |",
            f"| brain_banc (fenêtre calibrée sur l'or) | figé | {budget} | "
            f"{_cell('brain_banc_calibre', 'verbatim', 'tout')} | "
            f"{_temoin('brain_banc_calibre', 'verbatim')} | "
            f"{_cell('brain_banc_calibre', 'valeur', 'tout')} | "
            f"{_temoin('brain_banc_calibre', 'valeur')} |",
            "",
        ]
    )
    return "\n".join(lines)


def ecrire_rapport(
    results: list[dict[str, Any]],
    *,
    run_limite: int | None = None,
    dossier: Dossier | None = None,
    questions: list[dict[str, Any]] | None = None,
    stabilite_par_fournisseur: dict[str, dict[str, Any]] | None = None,
    temoin_lignes: list[dict[str, Any]] | None = None,
    budgets_grille: list[int] | None = None,
    suffixe: str | None = None,
) -> None:
    current = dossier if dossier is not None else dossier_courant()
    merged = _charger_resultats_comparatif_complets(current, results, suffixe=suffixe)
    meta = _meta_mesure(current, questions or [])
    text = _format_table_par_forme(merged, run_limite=run_limite, dossier=current, meta=meta)
    text += _format_table_synthese(merged, dossier=current, meta=meta)
    text += _format_table_reference(merged, dossier=current, meta=meta)
    grille_budgets = budgets_grille or sorted(
        {int(run["budget"]) for result in merged for run in result["runs"]}
    )
    text += _format_section_comparaison_budget_egal(
        merged,
        dossier=current,
        budgets=grille_budgets,
        temoin_lignes=temoin_lignes,
    )
    text += _format_section_plafond_forme(
        merged,
        dossier=current,
        meta=meta,
        temoin_lignes=temoin_lignes,
    )
    text += _format_table_disputes(merged, dossier=current, meta=meta)
    text += _format_table_couts(current)
    if stabilite_par_fournisseur:
        text += _format_section_reproductibilite(stabilite_par_fournisseur)
    rapport = _rapport_path(current, suffixe=suffixe)
    rapport.parent.mkdir(parents=True, exist_ok=True)
    rapport.write_text(text + "\n", encoding="utf-8")


def _expected_configs(nom: str, ks: list[int], budgets: list[int]) -> set[tuple[int, int | None]]:
    effective_ks = [None] if nom in K_FIXE else ks
    return {(budget, k) for budget in budgets for k in effective_ks}


def _fichier_mesure_complet(
    path: Path,
    *,
    nom: str,
    ks: list[int],
    budgets: list[int],
    questions: list[dict[str, Any]],
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("fournisseur") != nom:
        return False
    runs = payload.get("runs") or []
    expected = _expected_configs(nom, ks, budgets)
    actual = {(int(run["budget"]), run.get("k")) for run in runs}
    if actual != expected:
        return False
    question_ids = {question["id"] for question in questions}
    for run in runs:
        rows = run.get("questions") or []
        if {row["question_id"] for row in rows} != question_ids:
            return False
        if len(rows) != len(questions):
            return False
    return True


def _piece_from_evidence(item: dict[str, Any]) -> str:
    source = str(item.get("source") or "").strip()
    if source:
        return Path(source).name
    provenance = item.get("provenance") or {}
    path = str(provenance.get("source_file_path") or provenance.get("piece") or "").strip()
    return Path(path).name if path else ""


def _pieces_sont_dossier(pieces: list[str], dossier: Dossier) -> bool:
    if not pieces:
        return False
    corpus = {path.name for path in dossier.corpus_dir.iterdir() if path.is_file()}
    return all(piece in corpus for piece in pieces)


def _fournisseur_contexte_preuve(
    nom: str,
    *,
    embed_cache: dict[str, list[float]],
    dossier: Dossier,
    k: int,
    connection,
) -> dict[str, Any]:
    provider = _instancier(nom, embed_cache, dossier=dossier)
    question = next(
        (item for item in load_questions(dossier) if item["id"] == dossier.smoke_question_id),
        None,
    )
    if question is None:
        raise RuntimeError(f"question fumée absente : {dossier.smoke_question_id}")
    effective_k = k_effectif_fournisseur(nom, k)
    elements = provider.contexte(question, k=effective_k, connection=connection)
    pieces = [_piece_from_evidence(item) for item in elements]
    return {
        "elements": len(elements),
        "pieces": pieces,
        "pieces_moreau": _pieces_sont_dossier(pieces, dossier) if dossier.nom == "moreau" else None,
        "vault_ref": dossier.vault_ref,
        "responsibility_context": dossier.responsibility_context,
    }


def _brain_prod_mor_q01_proof(
    *,
    embed_cache: dict[str, list[float]],
    dossier: Dossier,
    k: int,
) -> dict[str, Any]:
    from .infra import connect_existing, start_cluster

    questions = load_questions(dossier)
    question = next((q for q in questions if q["id"] == dossier.smoke_question_id), None)
    if question is None:
        raise RuntimeError(f"question fumée absente : {dossier.smoke_question_id}")
    start_cluster()
    from .production import restart_postgrest_public

    restart_postgrest_public(service_key=_brain_service_key())
    provider = BrainProd(embed_cache, dossier=dossier)
    connection = connect_existing()
    elements = provider.contexte(question, k=k, connection=connection)
    pieces = [_piece_from_evidence(item) for item in elements]
    return {
        "instance": dossier.instance,
        "vault_ref": dossier.vault_ref,
        "aire": str(dossier.area_root),
        "elements": len(elements),
        "pieces": pieces,
    }


def _brain_service_key() -> str:
    secrets = BANC_ROOT / "secrets.env"
    if secrets.is_file():
        for line in secrets.read_text(encoding="utf-8").splitlines():
            if line.startswith("BANC_BRAIN_SERVICE_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _chemins_verification(
    *,
    fournisseurs: list[str],
    dossier: Dossier,
) -> dict[str, str]:
    chemins: dict[str, str] = {
        "cache": str(dossier.cache_embeddings_questions),
        "sortie": str(dossier.comparatif_dir),
    }
    for nom in fournisseurs:
        if nom == "mem0":
            chemins[f"contextes_{nom}"] = str(comparatif_context_path("mem0", dossier=dossier, k=20))
            chemins[f"contextes_{nom}_k40"] = str(
                comparatif_context_path("mem0", dossier=dossier, k=40)
            )
        elif nom == "letta_archive":
            chemins[f"contextes_{nom}"] = str(comparatif_context_path(nom, dossier=dossier))
    return chemins


def run_mesure(
    *,
    fournisseurs: list[str],
    ks: list[int],
    budgets: list[int],
    limit_questions: int | None = None,
    dossier: Dossier | None = None,
    verifier_seulement: bool = False,
    reprendre: bool = False,
    forcer: list[str] | None = None,
    stabilite: int = 1,
    suffixe: str | None = None,
    reconstitution_dir: Path | None = None,
    embeddings_objets: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    from .infra import connect_existing, start_cluster
    from .production import restart_postgrest_public

    current = dossier if dossier is not None else dossier_courant()
    export_runtime_env()
    all_questions = load_questions(current)
    gold = load_gold(current)
    verifier_or_coherent(current, all_questions, gold)
    verifier_parite(current)
    questions = all_questions
    if limit_questions is not None:
        questions = questions[:limit_questions]
    embed_cache = _load_embed_cache(current)
    exiger_contextes_dossier(current, questions, embed_cache)
    missing = missing_embed_texts(questions, embed_cache)
    if missing:
        raise RuntimeError(f"embeddings manquants ({len(missing)}) : {missing[:3]}")
    forcer_set = set(forcer or [])
    if verifier_seulement:
        chemins = _chemins_verification(fournisseurs=fournisseurs, dossier=current)
        for nom in fournisseurs:
            if nom in ("mem0", "letta_archive"):
                provider = _instancier(nom, embed_cache, dossier=current)
                if nom == "mem0":
                    provider.ensure_fichiers(ks)
                else:
                    provider._load()
        payload: dict[str, Any] = {
            "status": "verifie",
            "dossier": current.nom,
            "chemins": chemins,
        }
        preuve: dict[str, Any] = {}
        needs_cluster = any(
            nom in fournisseurs
            for nom in BRAIN_FOURNISSEURS
            | {
                "rag_naif",
                "rag_naif_large",
                "rag_filtre_pieces",
                "rag_filtre_situations",
                "rag_filtre_union",
                "rag_filtre_entites",
                "entites_lignes_puis_rag",
                "rag_filtre_sens",
                "rag_filtre_ontologie",
                "objets_sens_puis_rag",
                "objets_sens_puis_rag_lecture_simple",
                "objets_sens_puis_rag_lecture_simple_entites",
                "objets_sens_puis_rag_lecture_simple_entites_5000",
                "desaccords_puis_rag",
                "desaccords_puis_rag_filtre",
                "desaccords_puis_g12",
            }
        )
        if needs_cluster:
            start_cluster()
            if "brain_prod" in fournisseurs:
                restart_postgrest_public(service_key=_brain_service_key())
            connection = connect_existing()
            effective_k = ks[0] if ks else K_PROD
            for nom in (
                "rag_naif",
                "rag_naif_large",
                "rag_filtre_pieces",
                "rag_filtre_situations",
                "rag_filtre_union",
                "rag_filtre_entites",
                "entites_lignes_puis_rag",
                "rag_filtre_sens",
                "rag_filtre_ontologie",
                "objets_sens_puis_rag",
                "objets_sens_puis_rag_lecture_simple",
                "objets_sens_puis_rag_lecture_simple_entites",
                "objets_sens_puis_rag_lecture_simple_entites_5000",
                "desaccords_puis_rag",
                "desaccords_puis_rag_filtre",
                "desaccords_puis_g12",
                *sorted(BRAIN_FOURNISSEURS),
            ):
                if nom not in fournisseurs:
                    continue
                if nom == "brain_prod":
                    preuve[nom] = _brain_prod_mor_q01_proof(
                        embed_cache=embed_cache,
                        dossier=current,
                        k=effective_k,
                    )
                else:
                    preuve[nom] = _fournisseur_contexte_preuve(
                        nom,
                        embed_cache=embed_cache,
                        dossier=current,
                        k=effective_k,
                        connection=connection,
                    )
        if preuve:
            payload["preuve_q01"] = preuve
        return payload
    assertion_index = gold["assertion_index"]
    start_cluster()
    if "brain_prod" in fournisseurs:
        restart_postgrest_public(service_key=_brain_service_key())
    connection = connect_existing()
    voisinage_fournisseurs = {"brain_banc_v1", "brain_banc_v2", "brain_banc_v3", "brain_banc_v4"}
    if voisinage_fournisseurs & set(fournisseurs):
        from .porte_voisinage import compter_voisin_hits_par_question, exiger_porte_voisinage_active

        probe_nom = next(
            nom
            for nom in ("brain_banc_v3", "brain_banc_v1", "brain_banc_v4", "brain_banc_v2")
            if nom in fournisseurs
        )
        probe = _instancier(probe_nom, embed_cache, dossier=current)
        stats_voisinage = compter_voisin_hits_par_question(
            questions,
            fournisseur=probe,
            connection=connection,
            k=k_effectif_fournisseur(probe_nom),
        )
        exiger_porte_voisinage_active(stats_voisinage, dossier=current.nom)
    claims = _read_public_claims(connection, vault_ref=current.vault_ref)
    claims_index = {str(c["claim_id"]): c for c in claims}
    holder_table, _, _ = build_holder_table(claims, gold)
    reglage, ecart = partager(load_questions(current))
    reglage_ids = {q["id"] for q in reglage}
    ecart_ids = {q["id"] for q in ecart}

    results = []
    stabilite_par_fournisseur: dict[str, dict[str, Any]] = {}
    for nom in fournisseurs:
        out_path = etage2_resultat_path(nom, current, suffixe=suffixe)
        if (
            reprendre
            and nom not in forcer_set
            and _fichier_mesure_complet(
                out_path,
                nom=nom,
                ks=ks,
                budgets=budgets,
                questions=questions,
            )
        ):
            results.append(json.loads(out_path.read_text(encoding="utf-8")))
            continue
        results.append(
            mesurer_fournisseur(
                nom,
                connection=connection,
                ks=ks,
                budgets=budgets,
                questions=questions,
                gold=gold,
                claims_index=claims_index,
                assertion_index=assertion_index,
                holder_table=holder_table,
                embed_cache=embed_cache,
                reglage_ids=reglage_ids,
                ecart_ids=ecart_ids,
                dossier=current,
                suffixe=suffixe,
                reconstitution_dir=reconstitution_dir,
                embeddings_objets=embeddings_objets,
            )
        )
        if stabilite > 1:
            stabilite_par_fournisseur[nom] = mesurer_stabilite_fournisseur(
                nom,
                connection=connection,
                ks=ks,
                budgets=budgets,
                questions=questions,
                assertion_index=assertion_index,
                claims_index=claims_index,
                embed_cache=embed_cache,
                stabilite=stabilite,
                dossier=current,
            )
    meta = _meta_mesure(current, questions)
    merged = _charger_resultats_comparatif_complets(current, results, suffixe=suffixe)
    ecrire_rapport(
        merged,
        run_limite=limit_questions,
        dossier=current,
        questions=questions,
        stabilite_par_fournisseur=stabilite_par_fournisseur,
        suffixe=suffixe,
    )
    return {
        "status": "complete",
        "fournisseurs": fournisseurs,
        "questions": len(questions),
        "dossier": current.nom,
        "questions_mesurables": meta["questions_mesurables"],
        "non_etabli": meta["non_etabli"],
        "stabilite": stabilite_par_fournisseur,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mesure comparatif étage 2")
    parser.add_argument("--fournisseurs", required=True, help="Liste séparée par des virgules")
    parser.add_argument("--k", required=True, help="Liste de k séparée par des virgules")
    parser.add_argument("--budgets", required=True, help="Liste de budgets caractères")
    parser.add_argument("--questions", type=int, default=None, help="Limiter le nombre de questions")
    parser.add_argument(
        "--dossier",
        default="greenloop",
        help="Nom du dossier (greenloop, moreau)",
    )
    parser.add_argument(
        "--verifier-seulement",
        action="store_true",
        help="S'arrête après garde-fous et résolution des chemins (sans score ni rapport)",
    )
    parser.add_argument(
        "--reprendre",
        action="store_true",
        help="Ne recalcule pas un fournisseur dont le JSON de sortie est déjà complet",
    )
    parser.add_argument(
        "--forcer",
        default="",
        help="Liste de fournisseurs à recalculer même si le JSON est complet",
    )
    parser.add_argument(
        "--stabilite",
        type=int,
        default=2,
        help="Nombre d'exécutions de recherche pour mesurer la reproductibilité (défaut 2)",
    )
    parser.add_argument(
        "--reconstitution-dir",
        type=Path,
        default=None,
        help="Répertoire de reconstitution pour les fournisseurs entités/objets.",
    )
    parser.add_argument(
        "--embeddings-objets",
        type=Path,
        default=None,
        help="Cache JSON des embeddings d'objets de reconstitution.",
    )
    parser.add_argument(
        "--suffixe",
        default=None,
        help="Suffixe des fichiers etage2-*, contextes-* et du rapport.",
    )
    args = parser.parse_args()
    fournisseurs = [item.strip() for item in args.fournisseurs.split(",") if item.strip()]
    ks = [int(item.strip()) for item in args.k.split(",") if item.strip()]
    budgets = [int(item.strip()) for item in args.budgets.split(",") if item.strip()]
    forcer = [item.strip() for item in args.forcer.split(",") if item.strip()]
    embeddings_objets = None
    if args.embeddings_objets is not None:
        embeddings_objets = json.loads(args.embeddings_objets.read_text(encoding="utf-8"))
    payload = run_mesure(
        fournisseurs=fournisseurs,
        ks=ks,
        budgets=budgets,
        limit_questions=args.questions,
        dossier=dossier_courant(args.dossier),
        verifier_seulement=args.verifier_seulement,
        reprendre=args.reprendre,
        forcer=forcer,
        stabilite=args.stabilite,
        suffixe=args.suffixe or None,
        reconstitution_dir=args.reconstitution_dir,
        embeddings_objets=embeddings_objets,
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
