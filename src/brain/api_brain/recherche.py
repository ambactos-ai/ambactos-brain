# SPDX-License-Identifier: AGPL-3.0-only
"""Embedding des questions et fournisseur de contexte composite."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from banc_realite.dossiers import Dossier
from banc_realite.embeddings_faits import (
    _embed_lot,
    charger_cache,
    make_compteur_embeddings,
)
from banc_realite.fournisseurs_contexte import (
    K_VIVIER_FILTRE,
    ObjetsSensPuisRag,
    PART_LIGNES,
    _RagFiltreBase,
    _elements_objets_sens,
    _passage_recouvre_lignes,
    _signatures_lignes_elements,
    _stats_entites,
    _texte_question,
    filtrer_passages_par_pieces,
)
from banc_realite.plafond_appels import CompteurAppels, PlafondAtteint
from banc_realite.recherche_entites import (
    IndexReconstitution,
    PreparationObjets,
    charger_index,
    matrice_depuis_cache,
    objets_de_recherche,
    preparer_objets,
    selectionner_par_le_sens,
)
from banc_realite.reconstitution_entites import verrou_publication

from .ontologie import _verrou_fichier_ontologie
from .tenants import (
    MSG_PLAFOND_NON_DECIDE,
    MSG_SANS_DEPOT_BRUT,
    TenantConfig,
    dossier_de_tenant,
)

BUDGET_DEFAUT = 12000
K_CONTEXTE = K_VIVIER_FILTRE
_CACHE_NOM = "embeddings-questions.json"
_INTERVALLE_CONTROLE_GENERATION_S = 2.0
_DELAI_GRACE_FERMETURE_S = 60.0
_JOURNAL = logging.getLogger(__name__)


def chemin_cache_questions(racine: Path) -> Path:
    return racine / _CACHE_NOM


def _cle_sha256(texte: str) -> str:
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def _charger_cache_disque(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): [float(x) for x in value]
        for key, value in payload.items()
        if isinstance(value, list)
    }


def _sauver_cache_disque(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(cache, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


ERREUR_PLAFOND_QUESTIONS_NON_DECIDE = (
    "plafond d'embeddings de questions non décidé pour ce tenant"
)
INDICATION_PLAFOND_QUESTIONS = (
    "le définir dans le registre des tenants (plafonds.embeddings_questions)"
)


def plafond_embeddings_questions_depuis_cfg(plafonds: dict[str, int]) -> int:
    return int(plafonds.get("embeddings_questions", 0))


def message_erreur_plafond_questions(exc: PlafondAtteint, *, plafond_cfg: int) -> str:
    if plafond_cfg <= 0:
        return f"{ERREUR_PLAFOND_QUESTIONS_NON_DECIDE} ; {INDICATION_PLAFOND_QUESTIONS}."
    return str(exc)


def make_compteur_questions(racine: Path, *, plafond: int) -> CompteurAppels:
    compteur = make_compteur_embeddings(constructeur_root=racine, plafond=plafond)
    compteur.definir_plafond(plafond)
    return compteur


def embed_question(
    texte: str,
    *,
    racine: Path,
    compteur_dossier: str,
    plafond: int,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
    compteur: CompteurAppels | None = None,
) -> tuple[dict[str, list[float]], bool]:
    """Retourne un cache {texte: vecteur} et si un appel réseau a été émis."""
    cache_path = chemin_cache_questions(racine)
    cle = _cle_sha256(texte)
    verrou = _verrou_fichier_ontologie(cache_path)
    with verrou:
        disque = _charger_cache_disque(cache_path)
        appel = False
        if cle not in disque:
            embedder_effectif = embedder
            if embedder_effectif is None:
                from banc_realite.compat_moteur import embed_many_with_retry, load_api_key

                load_api_key()
                embedder_effectif = embed_many_with_retry
            compteur_effectif = compteur or make_compteur_questions(racine, plafond=plafond)
            vecteurs = _embed_lot(
                [texte],
                embed_fn=embedder_effectif,
                compteur=compteur_effectif,
                dossier=compteur_dossier,
            )
            disque[cle] = vecteurs[0]
            _sauver_cache_disque(cache_path, disque)
            appel = True
        return {texte: disque[cle]}, appel


def tronquer_contexte(
    elements: list[dict[str, Any]], *, budget: int
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    total = 0
    for item in elements:
        longueur = len(str(item.get("text") or ""))
        if kept and total + longueur > budget:
            break
        if not kept and longueur > budget:
            tronque = dict(item)
            texte = str(item.get("text") or "")
            tronque["text"] = texte[:budget]
            tronque["tronque"] = True
            kept.append(tronque)
            total = budget
            break
        if total + longueur > budget:
            break
        kept.append(item)
        total += longueur
    return kept, total


def rechercher_contexte(
    *,
    question_texte: str,
    tenant_racine: Path,
    compteur_dossier: str,
    dossier: Dossier,
    reconstitution_dir: Path,
    connection: Any,
    embed_cache: dict[str, list[float]] | None = None,
    fournisseur: ObjetsSensPuisRag | None = None,
    budget: int = BUDGET_DEFAUT,
    plafond_embeddings: int = 0,
) -> dict[str, Any]:
    qid = f"api-{_cle_sha256(question_texte)[:12]}"
    question = {"id": qid, "question": question_texte}
    cache = embed_cache or {}
    if question_texte not in cache:
        cache_partiel, _ = embed_question(
            question_texte,
            racine=tenant_racine,
            compteur_dossier=compteur_dossier,
            plafond=plafond_embeddings,
        )
        cache = {**cache, **cache_partiel}
    provider = fournisseur or ObjetsSensPuisRag(
        cache,
        dossier=dossier,
        reconstitution_dir=reconstitution_dir,
    )
    elements = provider.contexte(question, k=K_CONTEXTE, connection=connection)
    contexte, caracteres = tronquer_contexte(elements, budget=budget)
    stats = provider.stats_par_question.get(qid) or provider.stats
    entites_filtre = stats.get("entites_filtre")
    payload: dict[str, Any] = {
        "question": question_texte,
        "contexte": contexte,
        "caracteres": caracteres,
    }
    if entites_filtre is not None:
        payload["entites_filtre"] = entites_filtre
    return payload


def charger_embeddings_objets(reconstitution_dir: Path) -> dict[str, list[float]]:
    path = reconstitution_dir / "embeddings-objets.json"
    return charger_cache(path)


class _FournisseurTenant(ObjetsSensPuisRag):
    """ObjetsSensPuisRag avec classement vectorisé préchargé."""

    def __init__(
        self,
        embed_cache: dict[str, list[float]],
        *,
        dossier: Dossier,
        reconstitution_dir: Path,
        embeddings_objets: dict[str, list[float]],
        matrice_objets: tuple[np.ndarray, list[str]],
        preparation_objets: PreparationObjets,
        index: IndexReconstitution | Any,
        objets: list[Any],
    ) -> None:
        objets_figes = list(objets)
        if isinstance(index, IndexReconstitution):
            index._objets_cache = objets_figes
        self._index = index
        self._objets = objets_figes
        self._embeddings_objets = embeddings_objets
        self._embeddings_objets_path = reconstitution_dir / "embeddings-objets.json"
        self._dossier = dossier
        self._embed_cache = embed_cache
        self._stats: dict[str, object] = {}
        self._stats_par_question: dict[str, dict[str, object]] = {}
        self._charger_cache_fn = charger_cache
        self._matrice_objets = matrice_objets
        self._preparation_objets = preparation_objets
        self._rag = _RagFiltreBase(embed_cache, dossier=dossier)

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        selection = self._selection(question)
        lignes_elements = _elements_objets_sens(selection)
        retenus: list[dict[str, Any]] = []
        total = 0
        for element in lignes_elements:
            longueur = len(str(element.get("text") or ""))
            if total + longueur > PART_LIGNES:
                break
            retenus.append(element)
            total += longueur
        deja = _signatures_lignes_elements(retenus)
        pieces = list(selection.pieces_designees)
        vivier = self._rag._vivier(question, k=K_VIVIER_FILTRE, connection=connection)
        rag_passages, _sans_piece = filtrer_passages_par_pieces(vivier, pieces)
        self._rag._enregistrer_stats(
            question,
            _stats_entites(
                question,
                selection,
                vivier=len(vivier),
                rendus=len(rag_passages),
                k_demande=K_VIVIER_FILTRE,
                repli_sans_filtre=False,
            ),
        )
        rag_retenus: list[dict[str, Any]] = []
        for item in rag_passages:
            if _passage_recouvre_lignes(item, deja):
                continue
            rag_retenus.append(item)
        elements = retenus + rag_retenus
        rag_stats = self._rag.stats_par_question.get(str(question.get("id") or ""), {})
        self._enregistrer_stats(
            question,
            _stats_entites(
                question,
                selection,
                vivier=int(rag_stats.get("vivier") or 0),
                rendus=len(elements),
                k_demande=int(rag_stats.get("k_demande") or K_VIVIER_FILTRE),
                repli_sans_filtre=bool(rag_stats.get("repli_sans_filtre")),
            ),
        )
        return elements

    def _selection(self, question: dict[str, Any]):
        if self._matrice_objets is not None:
            retenus = _objets_avec_matrice(
                objets_de_recherche(self._index),
                self._matrice_objets[1],
            )
            if isinstance(self._index, IndexReconstitution):
                self._index._objets_cache = retenus
            self._objets = retenus
        return selectionner_par_le_sens(
            _texte_question(question),
            self._index,
            vecteur_question=self._vecteur_question(question),
            vecteurs_objets=self._vecteurs_objets(),
            matrice_objets=self._matrice_objets,
            preparation_objets=self._preparation_objets,
        )


@dataclass
class EtatTenant:
    """Instantané immuable des ressources de recherche d'un tenant.

    Connexion, matrice, préparation, fournisseur et génération restent ceux
    de l'instantané. Un rechargement substitue un nouvel objet ; celui-ci
    ferme sa connexion quand plus aucune requête ne le référence.
    """

    embed_cache: dict[str, list[float]] = field(default_factory=dict)
    compteur_questions: CompteurAppels | None = None
    compteur_reconstitution: CompteurAppels | None = None
    fournisseur: ObjetsSensPuisRag | None = None
    connection: Any = None
    matrice_objets: tuple[np.ndarray, list[str]] | None = None
    embeddings_objets: dict[str, list[float]] | None = None
    preparation_objets: PreparationObjets | None = None
    generation_chargee: int | None = None
    dernier_controle_generation: float = 0.0
    _refs: int = field(default=0, compare=False, repr=False)
    _retire: bool = field(default=False, compare=False, repr=False)
    _ferme: bool = field(default=False, compare=False, repr=False)
    _verrou: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )
    _minuteur: threading.Timer | None = field(
        default=None, compare=False, repr=False
    )

    def acquerir(self) -> "EtatTenant":
        with self._verrou:
            self._refs += 1
        return self

    def relacher(self) -> None:
        with self._verrou:
            if self._refs > 0:
                self._refs -= 1
            if self._retire and self._refs <= 0:
                self._fermer_connexion_verrouille()

    def retirer(self) -> None:
        """Cet instantané n'est plus le courant ; fermer dès qu'il n'est plus tenu."""
        with self._verrou:
            self._retire = True
            if self._refs <= 0:
                self._fermer_connexion_verrouille()
            else:
                self._programmer_grace_verrouille()

    def _annuler_minuteur_verrouille(self) -> None:
        timer = self._minuteur
        self._minuteur = None
        if timer is None:
            return
        timer.cancel()
        with _verrou_minuteurs:
            _minuteurs.discard(timer)

    def _programmer_grace_verrouille(self) -> None:
        if self._minuteur is not None or self._ferme:
            return
        timer = threading.Timer(_DELAI_GRACE_FERMETURE_S, self._expirer_grace)
        timer.daemon = True
        self._minuteur = timer
        with _verrou_minuteurs:
            _minuteurs.add(timer)
        timer.start()

    def _expirer_grace(self) -> None:
        with self._verrou:
            self._fermer_connexion_verrouille()

    def _fermer_connexion_verrouille(self) -> None:
        self._annuler_minuteur_verrouille()
        if self._ferme:
            return
        self._ferme = True
        conn = self.connection
        self.connection = None
        if conn is None:
            return
        try:
            if not getattr(conn, "closed", False):
                conn.close()
        except Exception:
            pass


def compteur_questions_pour_tenant(
    racine: Path, etat: EtatTenant, *, plafond: int
) -> CompteurAppels:
    if etat.compteur_questions is None:
        etat.compteur_questions = make_compteur_questions(racine, plafond=plafond)
    else:
        etat.compteur_questions.definir_plafond(plafond)
    return etat.compteur_questions


def compteur_reconstitution_pour_tenant(cfg: TenantConfig) -> CompteurAppels:
    """Compteur de reconstitution : ``definir_plafond`` une fois par état de tenant."""
    from banc_realite.regimes_reconstitution import make_compteur_reconstitution

    etat = obtenir_etat_tenant(cfg.nom)
    plafond = cfg.plafond_reconstitution_appels()
    with etat._verrou:
        compteur = etat.compteur_reconstitution
        if compteur is None:
            compteur = make_compteur_reconstitution(cfg)
            etat.compteur_reconstitution = compteur
        elif compteur.plafond != plafond:
            compteur.definir_plafond(plafond)
        return compteur


_verrou_etats = threading.Lock()
_etats: dict[str, EtatTenant] = {}
_verrou_minuteurs = threading.Lock()
_minuteurs: set[threading.Timer] = set()
_dernieres_passes_sync: dict[str, str] = {}


def vider_etat_tenant(nom: str) -> None:
    with _verrou_etats:
        ancien = _etats.pop(nom, None)
    _dernieres_passes_sync.pop(nom, None)
    if ancien is not None:
        ancien.retirer()


def _generation_sur_disque(cfg: TenantConfig) -> int:
    path = cfg.chemin_marqueur_generation()
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    return int(payload.get("generation") or 0)


def _courant_apres_sync(cfg: TenantConfig) -> EtatTenant:
    """Instantané courant, remplacé si la génération sur disque a changé.

    Appeler sous ``_verrou_etats``. L'ancien instantané n'est pas muté.
    """
    nom = cfg.nom
    etat = _etats.get(nom)
    if etat is None:
        etat = EtatTenant()
        _etats[nom] = etat
    maintenant = time.monotonic()
    if (
        etat.generation_chargee is not None
        and maintenant - etat.dernier_controle_generation
        < _INTERVALLE_CONTROLE_GENERATION_S
    ):
        return etat
    etat.dernier_controle_generation = maintenant
    generation = _generation_sur_disque(cfg)
    if etat.generation_chargee is None:
        etat.generation_chargee = generation
        return etat
    if generation != etat.generation_chargee:
        nouveau = EtatTenant()
        nouveau.generation_chargee = generation
        nouveau.dernier_controle_generation = maintenant
        _etats[nom] = nouveau
        etat.retirer()
        return nouveau
    return etat


def acquerir_etat_tenant(cfg: TenantConfig) -> EtatTenant:
    """Référence sur l'instantané courant (+1) ; à relâcher en ``finally``."""
    with _verrou_etats:
        etat = _courant_apres_sync(cfg)
        etat.acquerir()
        return etat


def _objets_avec_vecteur(objets: list, embeddings: dict[str, list[float]]) -> list:
    retenus = []
    for obj in objets:
        if obj.empreinte in embeddings:
            retenus.append(obj)
        else:
            _JOURNAL.warning(
                "objet ignoré : empreinte absente du cache d'embeddings (%s, %s)",
                obj.id,
                obj.empreinte,
            )
    return retenus


def _objets_avec_matrice(objets: list, ordre: list[str]) -> list:
    dans_matrice = set(ordre)
    retenus = []
    for obj in objets:
        if obj.empreinte in dans_matrice:
            retenus.append(obj)
        else:
            _JOURNAL.warning(
                "objet ignoré : empreinte absente de la matrice (%s, %s)",
                obj.id,
                obj.empreinte,
            )
    return retenus


def _figer_objets(index: Any, objets: list) -> None:
    if isinstance(index, IndexReconstitution):
        index._objets_cache = objets


def _fabrique_defaut(cfg: TenantConfig, etat: EtatTenant) -> ObjetsSensPuisRag:
    reconst = cfg.chemin_reconstitution_effectif()
    with verrou_publication(reconst):
        embeddings = charger_embeddings_objets(reconst)
        index = charger_index(reconst)
        retenus = _objets_avec_vecteur(objets_de_recherche(index), embeddings)
        empreintes = [obj.empreinte for obj in retenus]
        matrice, ordre = matrice_depuis_cache(embeddings, empreintes)
        retenus = _objets_avec_matrice(retenus, ordre)
        _figer_objets(index, retenus)
        preparation = preparer_objets(index)
        etat.embeddings_objets = embeddings
        etat.matrice_objets = (matrice, ordre)
        etat.preparation_objets = preparation
        return _FournisseurTenant(
            etat.embed_cache,
            dossier=dossier_de_tenant(cfg),
            reconstitution_dir=reconst,
            embeddings_objets=embeddings,
            matrice_objets=(matrice, ordre),
            preparation_objets=preparation,
            index=index,
            objets=retenus,
        )


def _assurer_fournisseur(
    cfg: TenantConfig,
    etat: EtatTenant,
    cache_question: dict[str, list[float]],
    fabrique: Callable[[TenantConfig, dict[str, list[float]]], ObjetsSensPuisRag] | None,
) -> ObjetsSensPuisRag:
    etat.embed_cache.update(cache_question)
    with etat._verrou:
        if etat.fournisseur is not None:
            etat.fournisseur._embed_cache = etat.embed_cache
            rag = getattr(etat.fournisseur, "_rag", None)
            if rag is not None:
                rag._embed_cache = etat.embed_cache
            return etat.fournisseur
    if fabrique is not None:
        nouveau = fabrique(cfg, etat.embed_cache)
    else:
        nouveau = _fabrique_defaut(cfg, etat)
    with etat._verrou:
        if etat.fournisseur is None:
            etat.fournisseur = nouveau
        else:
            etat.fournisseur._embed_cache = etat.embed_cache
            rag = getattr(etat.fournisseur, "_rag", None)
            if rag is not None:
                rag._embed_cache = etat.embed_cache
        return etat.fournisseur


def obtenir_etat_tenant(nom: str) -> EtatTenant:
    with _verrou_etats:
        if nom not in _etats:
            _etats[nom] = EtatTenant()
        return _etats[nom]


def prechauffer_tenant(
    cfg: TenantConfig,
    *,
    embed_cache_initial: dict[str, list[float]] | None = None,
    fabrique: Callable[[TenantConfig, dict[str, list[float]]], ObjetsSensPuisRag] | None = None,
) -> EtatTenant:
    with _verrou_etats:
        etat = _courant_apres_sync(cfg)
        if embed_cache_initial:
            etat.embed_cache.update(embed_cache_initial)
        _assurer_fournisseur(cfg, etat, {}, fabrique)
        return etat


def obtenir_connection_tenant(
    etat: EtatTenant,
    dsn: str,
    ouvrir: Callable[[str], Any],
) -> Any:
    with etat._verrou:
        conn = etat.connection
        fermee = False
        if conn is not None:
            try:
                fermee = bool(conn.closed)
            except Exception:
                fermee = True
        if conn is None or fermee:
            etat.connection = ouvrir(dsn)
        return etat.connection


ENV_RECONSTITUTION_A_LA_DEMANDE = "BRAIN_API_RECONSTITUTION_A_LA_DEMANDE"
DELAI_RECONSTITUTION_S = 900
JOURNAL_RECONSTITUTION_LIGNES = 50
JOURNAL_SYNCHRONISATION_LIGNES = 50
PREFIXE_THREAD_RECONSTITUTION = "brain-reconstitution"
ReconstitueurFn = Callable[..., dict[str, Any]]
SynchroniseurFn = Callable[..., dict[str, Any]]


def essayer_verrou_synchronisation(cfg: TenantConfig, *, declencheur: str) -> TextIO | None:
    from banc_realite.synchroniser import essayer_verrou_synchronisation as _essayer

    return _essayer(cfg, declencheur=declencheur)


def relacher_verrou_synchronisation(handle: TextIO | None) -> None:
    from banc_realite.synchroniser import relacher_verrou_synchronisation as _relacher

    _relacher(handle)


def detenteur_verrou_synchronisation(cfg: TenantConfig) -> dict[str, Any] | None:
    from banc_realite.synchroniser import detenteur_verrou_synchronisation as _detenteur

    return _detenteur(cfg)


_verrou_files_reconstitution = threading.Lock()
_files_reconstitution: dict[str, "FileReconstitutionTenant"] = {}


@dataclass
class FileReconstitutionTenant:
    """File d'une reconstitution à la fois par tenant (mémoire + executor)."""

    verrou: threading.Lock = field(default_factory=threading.Lock)
    executor: ThreadPoolExecutor | None = None
    en_cours: set[str] = field(default_factory=set)
    en_attente: list[str] = field(default_factory=list)
    arrete: bool = False
    verrous_sync_en_attente: list[TextIO] = field(default_factory=list)
    verrous_sync_en_cours: list[TextIO] = field(default_factory=list)

    def ids_occupes(self) -> set[str]:
        with self.verrou:
            return set(self.en_cours) | set(self.en_attente)

    def snapshot(self) -> tuple[list[str], list[str]]:
        with self.verrou:
            return sorted(self.en_cours), list(self.en_attente)

    def soumettre(self, unite_id: str, tache: Callable[[], None]) -> bool:
        with self.verrou:
            if self.arrete:
                return False
            if unite_id in self.en_cours or unite_id in self.en_attente:
                return False
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=PREFIXE_THREAD_RECONSTITUTION,
                )
            self.en_attente.append(unite_id)
            self.executor.submit(self._executer, unite_id, tache)
            return True

    def soumettre_tache_sync(
        self, handle: TextIO, tache: Callable[[], None]
    ) -> bool:
        """Soumet une passe de synchronisation ; le verrou reste au runner."""
        with self.verrou:
            if self.arrete:
                return False
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=PREFIXE_THREAD_RECONSTITUTION,
                )
            self.verrous_sync_en_attente.append(handle)
            self.executor.submit(self._executer_sync, handle, tache)
            return True

    def _executer_sync(self, handle: TextIO, tache: Callable[[], None]) -> None:
        with self.verrou:
            if self.arrete:
                return
            try:
                self.verrous_sync_en_attente.remove(handle)
            except ValueError:
                return
            self.verrous_sync_en_cours.append(handle)
        try:
            tache()
        except Exception:
            _JOURNAL.exception("synchronisation en fond")
        finally:
            relacher_verrou_synchronisation(handle)
            with self.verrou:
                try:
                    self.verrous_sync_en_cours.remove(handle)
                except ValueError:
                    pass

    def _executer(self, unite_id: str, tache: Callable[[], None]) -> None:
        with self.verrou:
            try:
                self.en_attente.remove(unite_id)
            except ValueError:
                pass
            self.en_cours.add(unite_id)
        try:
            tache()
        except Exception:
            _JOURNAL.exception("reconstitution en fond de %s", unite_id)
        finally:
            with self.verrou:
                self.en_cours.discard(unite_id)

    def arreter(self) -> None:
        with self.verrou:
            self.arrete = True
            executor = self.executor
            self.executor = None
            en_file = list(self.verrous_sync_en_attente)
            self.verrous_sync_en_attente.clear()
        for handle in en_file:
            relacher_verrou_synchronisation(handle)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def _file_reconstitution(nom: str) -> FileReconstitutionTenant:
    with _verrou_files_reconstitution:
        file = _files_reconstitution.get(nom)
        if file is None or file.arrete:
            file = FileReconstitutionTenant()
            _files_reconstitution[nom] = file
        return file


def arreter_files_reconstitution() -> None:
    """Arrête tous les executors de reconstitution (fermeture de l'app)."""
    with _verrou_files_reconstitution:
        files = list(_files_reconstitution.values())
        _files_reconstitution.clear()
    _dernieres_passes_sync.clear()
    for file in files:
        file.arreter()


def reconstitution_desactivee(
    cfg: TenantConfig, *, force_desactivee: bool = False
) -> bool:
    if force_desactivee:
        return True
    if os.environ.get(ENV_RECONSTITUTION_A_LA_DEMANDE) == "0":
        return True
    return not cfg.reconstitution_a_la_demande


def _lire_journal_reconstitution(chemin: Path, *, n: int = JOURNAL_RECONSTITUTION_LIGNES) -> list[dict[str, Any]]:
    if not chemin.is_file() or n <= 0:
        return []
    lignes = chemin.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for brut in lignes[-n:]:
        texte = brut.strip()
        if not texte:
            continue
        try:
            payload = json.loads(texte)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def etat_reconstitution_tenant(cfg: TenantConfig) -> dict[str, Any]:
    """Journal (50 dernières lignes) et file mémoire du tenant."""
    with _verrou_files_reconstitution:
        file = _files_reconstitution.get(cfg.nom)
    if file is None:
        en_cours: list[str] = []
        en_attente: list[str] = []
    else:
        en_cours, en_attente = file.snapshot()
    return {
        "journal": _lire_journal_reconstitution(cfg.chemin_journal_reconstitution()),
        "en_cours": en_cours,
        "en_attente": en_attente,
    }


def _executer_unite_fond(
    reconstitueur: ReconstitueurFn,
    cfg: TenantConfig,
    unite_id: str,
) -> None:
    from banc_realite.regimes_reconstitution import (
        noter_deja_a_jour,
        unite_encore_candidate_demande,
        verrou_unite,
    )

    reconstitution = cfg.chemin_reconstitution_effectif()
    with verrou_unite(reconstitution, unite_id) as jeton:
        if jeton is None:
            return
        encore, unite = unite_encore_candidate_demande(reconstitution, unite_id)
        if not encore:
            noter_deja_a_jour(
                cfg,
                unite_id,
                declencheur="api",
                entite=str((unite or {}).get("entite") or ""),
            )
            return
    reconstitueur(
        cfg,
        unite_id,
        compteur=compteur_reconstitution_pour_tenant(cfg),
        delai_s=DELAI_RECONSTITUTION_S,
        declencheur="api",
        seulement_si_eligible=True,
    )


def declencher_reconstitution_demande(
    cfg: TenantConfig,
    entites: Sequence[str],
    *,
    reconstitueur: ReconstitueurFn | None = None,
    force_desactivee: bool = False,
) -> dict[str, Any]:
    """Réponse immédiate : file courante + lancement fond des unités candidates."""
    if reconstitution_desactivee(cfg, force_desactivee=force_desactivee):
        return {"desactivee": True}
    if cfg.plafond_reconstitution_appels() <= 0:
        return {"refus": "plafond non décidé"}

    from banc_realite.regimes_reconstitution import (
        charger_ontologie_tenant,
        republication_seule,
        reconstituer_unite_tenant,
        unites_candidates_demande,
    )
    from banc_realite.reconstitution_entites import lire_registre_reconstitution

    reconstitution = cfg.chemin_reconstitution_effectif()
    noms = [str(nom) for nom in entites if nom]
    if not (reconstitution / "registre.json").is_file():
        file = _file_reconstitution(cfg.nom)
        occupes = file.ids_occupes()
        return {
            "en_cours": sorted(occupes),
            "perimees": [],
            "lancees": [],
        }
    registre = lire_registre_reconstitution(reconstitution)
    ontologie = charger_ontologie_tenant(cfg)
    candidates = unites_candidates_demande(
        cfg, registre, ontologie, reconstitution, noms
    )
    perimees = sorted(
        {
            str(unite.get("id") or "")
            for unite in candidates
            if isinstance(unite.get("perimee"), Mapping) and unite.get("id")
        }
    )
    ids_entites = {
        str(unite.get("id") or "")
        for unite in registre.get("unites") or []
        if isinstance(unite, Mapping)
        and str(unite.get("entite") or "") in set(noms)
        and unite.get("id")
    }
    file = _file_reconstitution(cfg.nom)
    occupes = file.ids_occupes()
    compteur = compteur_reconstitution_pour_tenant(cfg)
    etat_compteur = compteur.lire()
    atteint = int(etat_compteur.get("emis") or 0) >= int(
        etat_compteur.get("plafond") or 0
    )
    fn = reconstitueur if reconstitueur is not None else reconstituer_unite_tenant
    lancees: list[str] = []
    refus: str | None = None
    for unite in candidates:
        uid = str(unite.get("id") or "")
        if not uid or uid in occupes:
            continue
        if atteint and not republication_seule(unite):
            refus = "plafond atteint"
            continue
        if file.soumettre(uid, partial(_executer_unite_fond, fn, cfg, uid)):
            lancees.append(uid)
            occupes.add(uid)
    occupes = file.ids_occupes()
    payload: dict[str, Any] = {
        "en_cours": sorted(occupes & ids_entites),
        "perimees": perimees,
        "lancees": lancees,
    }
    if refus:
        payload["refus"] = refus
    return payload


def etat_synchronisation_tenant(cfg: TenantConfig) -> dict[str, Any]:
    """Journal (50 dernières lignes), attente disque, et état de la passe."""
    attente_path = cfg.chemin_attente_synchronisation()
    attente: dict[str, Any] = {
        "pieces_a_remplacer": [],
        "pieces_supprimees": [],
        "depuis": "",
    }
    if attente_path.is_file():
        try:
            payload = json.loads(attente_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            attente = payload
    journal = _lire_journal_reconstitution(
        cfg.chemin_journal_synchronisation(),
        n=JOURNAL_SYNCHRONISATION_LIGNES,
    )
    derniere = None
    if journal:
        horodatage = journal[-1].get("horodatage")
        if horodatage:
            derniere = str(horodatage)
    if derniere is None:
        derniere = _dernieres_passes_sync.get(cfg.nom)
    detenteur = detenteur_verrou_synchronisation(cfg)
    return {
        "journal": journal,
        "attente": attente,
        "en_cours": detenteur is not None,
        "detenteur": detenteur,
        "derniere_passe": derniere,
    }


def _appeler_synchroniseur(
    fn: SynchroniseurFn,
    cfg: TenantConfig,
    *,
    a_blanc: bool,
) -> dict[str, Any]:
    """L'API tient déjà le verrou : la passe ne le reprend pas."""
    return fn(cfg, a_blanc=a_blanc, declencheur="api", prendre_verrou=False)


def declencher_synchronisation(
    cfg: TenantConfig,
    *,
    a_blanc: bool = False,
    synchroniseur: SynchroniseurFn | None = None,
) -> dict[str, Any]:
    """Lance une passe : refus immédiat, à blanc synchrone, sinon fond + verrou."""
    if cfg.depot_brut is None:
        return {"statut": "refus", "motif": MSG_SANS_DEPOT_BRUT}
    if cfg.plafond_embeddings_passages() <= 0:
        return {"statut": "refus", "motif": MSG_PLAFOND_NON_DECIDE}

    handle = essayer_verrou_synchronisation(cfg, declencheur="api")
    if handle is None:
        return {"statut": "en_cours"}

    if synchroniseur is not None:
        fn = synchroniseur
    else:
        from banc_realite.synchroniser import synchroniser as fn

    if a_blanc:
        try:
            return _appeler_synchroniseur(fn, cfg, a_blanc=True)
        finally:
            relacher_verrou_synchronisation(handle)

    passe = datetime.now(timezone.utc).isoformat()
    _dernieres_passes_sync[cfg.nom] = passe
    file = _file_reconstitution(cfg.nom)

    def tache() -> None:
        _appeler_synchroniseur(fn, cfg, a_blanc=False)

    if not file.soumettre_tache_sync(handle, tache):
        relacher_verrou_synchronisation(handle)
        return {"statut": "en_cours"}
    return {"statut": "lancee", "passe": passe}
