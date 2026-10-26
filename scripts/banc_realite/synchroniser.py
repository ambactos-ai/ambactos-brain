# SPDX-License-Identifier: AGPL-3.0-only
"""Passe idempotente de synchronisation tenant.

Manifeste → lecture → couche brute Postgres → étiquettes → unités périmées
→ ``synchronisation.jsonl``.
"""
from __future__ import annotations

import copy
import fcntl
import json
import os
import shutil
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from banc_realite.couche_brute_moreau import (
    charger_index_texte_vecteur,
    lister_passages_corpus,
    remplacer_passages_piece,
    sha256_texte,
    supprimer_passages_piece,
)
from banc_realite.etiquetage_ontologie import (
    charger_etiquettes_si_present,
    ecrire_etiquettes_atomique,
    reetiqueter,
    sha256_ontologie,
)
from banc_realite.etiquetage_ontologie import _charger_regles_aiguillage
from banc_realite.lecture_depot_brut import (
    appliquer_manifeste,
    inventorier_depot,
    lire_dossier_depot,
)
from banc_realite.lecture_drive import actualiser_pieces_tous, ecrire_json_atomique
from banc_realite.passages import Passage, chunk_piece
from banc_realite.plafond_appels import CompteurAppels, PlafondAtteint
from banc_realite.reconstitution_entites import (
    incrementer_generation_reconstitution,
    marquer_unites_perimees,
    verrou_registre,
)
from banc_realite.regimes_reconstitution import reconstituer_unites_continues
from brain.api_brain.tenants import (
    MSG_PLAFOND_NON_DECIDE,
    MSG_SANS_DEPOT_BRUT,
    TenantConfig,
)

DSN_DEFAUT = "postgresql://postgres@127.0.0.1:56102/postgres"
NOM_VERROU_SYNCHRONISATION = "synchronisation.lock"
MSG_PASSE_DEJA_EN_COURS = "passe déjà en cours"


class PasseDejaEnCours(Exception):
    """Le verrou de synchronisation du tenant est déjà tenu."""

    def __init__(self, message: str = MSG_PASSE_DEJA_EN_COURS) -> None:
        super().__init__(message)


def chemin_verrou_synchronisation(cfg: TenantConfig) -> Path:
    return cfg.chemin_reconstitution_effectif() / NOM_VERROU_SYNCHRONISATION


def essayer_verrou_synchronisation(
    cfg: TenantConfig, *, declencheur: str
) -> TextIO | None:
    """Flock exclusif non bloquant ; ``None`` si une passe tient déjà le verrou.

    Le fichier porte pid, déclencheur et horodatage du détenteur.
    """
    reconstitution = cfg.chemin_reconstitution_effectif()
    reconstitution.mkdir(parents=True, exist_ok=True)
    path = chemin_verrou_synchronisation(cfg)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    payload = {
        "pid": os.getpid(),
        "declencheur": declencheur,
        "horodatage": datetime.now(timezone.utc).isoformat(),
    }
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()
    return handle


def relacher_verrou_synchronisation(handle: TextIO | None) -> None:
    if handle is None:
        return
    try:
        if handle.closed:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        if not handle.closed:
            handle.close()
    except OSError:
        pass


def _lire_payload_verrou(path: Path) -> dict[str, Any]:
    try:
        brut = path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not brut:
        return {}
    try:
        payload = json.loads(brut.splitlines()[0])
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    try:
        out["pid"] = int(payload.get("pid"))
    except (TypeError, ValueError):
        pass
    declencheur = payload.get("declencheur")
    if declencheur:
        out["declencheur"] = str(declencheur)
    horodatage = payload.get("horodatage")
    if horodatage:
        out["horodatage"] = str(horodatage)
    return out


def detenteur_verrou_synchronisation(cfg: TenantConfig) -> dict[str, Any] | None:
    """Métadonnées du détenteur si le verrou est tenu, sinon ``None``."""
    path = chemin_verrou_synchronisation(cfg)
    if not path.is_file():
        return None
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            handle.close()
        except OSError:
            pass
        return _lire_payload_verrou(path)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        handle.close()
    except OSError:
        pass
    return None


def synchronisation_en_cours(cfg: TenantConfig) -> bool:
    return detenteur_verrou_synchronisation(cfg) is not None


def make_compteur_embeddings_passages(racine: Path, *, plafond: int) -> CompteurAppels:
    compteur = CompteurAppels(
        compteur_path=racine / "compteur-embeddings-passages.json",
        journal_path=racine / "appels-embeddings-passages.jsonl",
        plafond=plafond,
    )
    compteur.definir_plafond(plafond)
    return compteur


def _horodatage() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uniq(valeurs: list[str]) -> list[str]:
    vus: set[str] = set()
    out: list[str] = []
    for valeur in valeurs:
        if valeur and valeur not in vus:
            vus.add(valeur)
            out.append(valeur)
    return out


def classer_pieces(
    snapshot: dict[str, dict[str, Any]],
    registre: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Retourne (nouvelles, modifiées, supprimées) d'après les noms de pièce."""
    apres = {str(e.get("id") or ""): e for e in registre if e.get("id")}
    nouvelles: list[str] = []
    modifiees: list[str] = []
    supprimees: list[str] = []

    def _empreinte(entree: dict[str, Any]) -> str | None:
        lecture = entree.get("lecture")
        if isinstance(lecture, dict):
            emp = lecture.get("empreinte")
            return str(emp) if emp else None
        return None

    for identifiant, ancienne in snapshot.items():
        if identifiant not in apres:
            piece = ancienne.get("piece")
            if piece:
                supprimees.append(str(piece))
            continue
        nouvelle = apres[identifiant]
        ancienne_piece = ancienne.get("piece")
        nouvelle_piece = nouvelle.get("piece")
        if nouvelle_piece and not ancienne_piece:
            nouvelles.append(str(nouvelle_piece))
        elif nouvelle_piece and ancienne_piece:
            if str(ancienne_piece) != str(nouvelle_piece):
                supprimees.append(str(ancienne_piece))
                nouvelles.append(str(nouvelle_piece))
            elif _empreinte(ancienne) != _empreinte(nouvelle):
                modifiees.append(str(nouvelle_piece))

    for identifiant, nouvelle in apres.items():
        if identifiant not in snapshot and nouvelle.get("piece"):
            nouvelles.append(str(nouvelle["piece"]))

    return _uniq(nouvelles), _uniq(modifiees), _uniq(supprimees)


def _verifier_corpus_cache(cfg: TenantConfig) -> Path:
    cache_tous = cfg.chemin_cache_pieces_tous().resolve()
    pieces = cfg.pieces.resolve() if cfg.pieces is not None else None
    if pieces is None or pieces != cache_tous:
        raise ValueError(
            f"tenant «{cfg.nom}» : pieces ({pieces}) n'est pas le cache lecture "
            f"{cache_tous} — pas de copie silencieuse"
        )
    return cache_tous


def _ouvrir_connexion(cfg: TenantConfig):
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(
        cfg.dsn_effectif(DSN_DEFAUT),
        autocommit=True,
        row_factory=dict_row,
    )


def _embed_defaut(texte: str) -> list[float]:
    from banc_realite.compat_moteur import embed_with_retry

    return embed_with_retry(texte)


def _passages_pour_etiquettes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        identifiant = row.get("id")
        if identifiant is None:
            continue
        out.append(
            {
                "id": str(identifiant),
                "piece": str(row.get("source_ref") or ""),
                "texte": str(row.get("content") or ""),
            }
        )
    return out


def _ids_passages(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["id"]) for row in rows if row.get("id") is not None}


def _resume_vide(
    *,
    horodatage: str,
    lignes_lues: int,
    duree_s: float,
) -> dict[str, Any]:
    return {
        "horodatage": horodatage,
        "lignes_lues": lignes_lues,
        "pieces": {"nouvelles": [], "modifiees": [], "supprimees": []},
        "passages_remplaces": 0,
        "embeddings_emis": 0,
        "embeddings_reutilises": 0,
        "unites_perimees": [],
        "unites_refaites": [],
        "unites_echouees": [],
        "unites_a_refaire": [],
        "duree_reconstitution_s": 0.0,
        "duree_s": duree_s,
        "erreurs": [],
        "plafond_atteint": False,
        "en_attente": [],
    }


def _append_journal(chemin: Path, ligne: dict[str, Any]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ligne, ensure_ascii=False) + "\n")


def _clore_passe(
    cfg: TenantConfig,
    resume: dict[str, Any],
    *,
    a_blanc: bool,
    declencheur: str,
    journaliser: bool,
    a_ecrit: bool,
) -> dict[str, Any]:
    """Pose ``declencheur``, journalise, invalide l'état si la passe a écrit."""
    resume["declencheur"] = declencheur
    if a_blanc:
        return resume
    if journaliser:
        _append_journal(cfg.chemin_journal_synchronisation(), resume)
    if a_ecrit:
        incrementer_generation_reconstitution(cfg.reconstitution)
    return resume


def _charger_curseur(sortie_dir: Path) -> dict[str, Any]:
    path = sortie_dir / "depot-brut.curseur.json"
    if not path.is_file():
        return {"lignes_appliquees": 0, "derniere_ligne_sha256": ""}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"lignes_appliquees": 0}


def _liste_pieces(valeur: Any) -> list[str]:
    if not isinstance(valeur, list):
        return []
    return _uniq([str(item) for item in valeur if item])


def _attente_vide() -> dict[str, Any]:
    return {"pieces_a_remplacer": [], "pieces_supprimees": [], "depuis": ""}


def _charger_attente(chemin: Path) -> dict[str, Any]:
    """Lit ``synchronisation.en-attente.json`` ; listes vides si absent."""
    if not chemin.is_file():
        return _attente_vide()
    payload = json.loads(chemin.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return _attente_vide()
    return {
        "pieces_a_remplacer": _liste_pieces(payload.get("pieces_a_remplacer")),
        "pieces_supprimees": _liste_pieces(payload.get("pieces_supprimees")),
        "depuis": str(payload.get("depuis") or ""),
    }


def _supprimer_attente(chemin: Path) -> None:
    if chemin.is_file():
        chemin.unlink()


def _fichier_regles_aiguillage(cfg: TenantConfig, sortie: Path) -> Path:
    """Fichier de règles pour ``inventorier_depot`` (ontologie ou cache lecture)."""
    chemin = sortie / "aiguillage.json"
    ontologie = json.loads(cfg.ontologie.read_text(encoding="utf-8"))
    if not isinstance(ontologie, dict):
        ontologie = {}
    regles = _charger_regles_aiguillage(ontologie)
    ecrire_json_atomique(chemin, {"regles": regles})
    return chemin


def _stats_remplacement_a_blanc(
    passages: list[Passage],
    index_texte_vecteur: dict[str, list[float]],
    *,
    compteur: CompteurAppels | None,
    plafond: int,
) -> dict[str, Any]:
    """Compte embeddings sans appeler le fournisseur (passe ``a_blanc``)."""
    emis = 0
    reutilises = 0
    emis_persist = int(compteur.lire()["emis"]) if compteur is not None else 0
    for passage in passages:
        cle = sha256_texte(passage.content)
        connu = index_texte_vecteur.get(cle)
        if connu is not None:
            reutilises += 1
            continue
        if emis_persist + emis >= plafond:
            raise PlafondAtteint(emis_persist + emis, plafond)
        emis += 1
        index_texte_vecteur[cle] = [0.0]
    return {
        "passages_remplaces": len(passages),
        "embeddings_emis": emis,
        "embeddings_reutilises": reutilises,
    }


def _persister_attente(
    chemin: Path,
    *,
    pieces_a_remplacer: list[str],
    pieces_supprimees: list[str],
    depuis: str,
    a_blanc: bool,
) -> None:
    if a_blanc:
        return
    restant_r = _uniq(pieces_a_remplacer)
    restant_s = _uniq(pieces_supprimees)
    if not restant_r and not restant_s:
        _supprimer_attente(chemin)
        return
    ecrire_json_atomique(
        chemin,
        {
            "pieces_a_remplacer": restant_r,
            "pieces_supprimees": restant_s,
            "depuis": depuis,
        },
    )


def synchroniser(
    cfg: TenantConfig,
    *,
    a_blanc: bool = False,
    connection: Any | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    compteur: CompteurAppels | None = None,
    declencheur: str = "cli",
    prendre_verrou: bool = True,
) -> dict[str, Any]:
    """Exécute une passe de synchronisation pour ``cfg``.

    ``connection`` et ``embed_fn`` sont injectables (tests hors ligne).
    ``a_blanc`` calcule le résumé sans aucune écriture persistante.
    ``declencheur`` est journalisé (``cli`` | ``api`` | ``timer``).
    Le verrou par tenant est pris sauf si l'appelant le tient déjà
    (``prendre_verrou=False``).
    """
    started = time.monotonic()
    horodatage = _horodatage()
    if cfg.depot_brut is None:
        raise ValueError(MSG_SANS_DEPOT_BRUT)
    plafond = cfg.plafond_embeddings_passages()
    if plafond <= 0:
        raise ValueError(MSG_PLAFOND_NON_DECIDE)

    handle_verrou: TextIO | None = None
    if prendre_verrou:
        handle_verrou = essayer_verrou_synchronisation(
            cfg, declencheur=declencheur
        )
        if handle_verrou is None:
            raise PasseDejaEnCours()

    sortie_orig = cfg.registre_pieces_effectif()
    tmp_blanc: Path | None = None
    connexion_ouverte = False
    try:
        _verifier_corpus_cache(cfg)
        if a_blanc:
            tmp_blanc = Path(tempfile.mkdtemp(prefix="sync-a-blanc-"))
            if sortie_orig.exists():
                shutil.copytree(sortie_orig, tmp_blanc, dirs_exist_ok=True)
            sortie = tmp_blanc
        else:
            sortie = sortie_orig

        registre_path = sortie / "registre.json"
        if not registre_path.is_file():
            raise FileNotFoundError(f"registre de lecture absent : {registre_path}")
        registre: list[dict[str, Any]] = json.loads(
            registre_path.read_text(encoding="utf-8")
        )
        snapshot = {str(e["id"]): copy.deepcopy(e) for e in registre if e.get("id")}
        curseur_avant = _charger_curseur(sortie)
        manifeste_path = cfg.depot_brut / "manifeste.jsonl"
        lignes_manifeste = manifeste_path.read_text(encoding="utf-8").splitlines()
        offset_manifeste = int(curseur_avant.get("lignes_appliquees") or 0)
        manifeste_restant = any(
            ligne.strip() for ligne in lignes_manifeste[offset_manifeste:]
        )
        if manifeste_restant:
            regles_path = _fichier_regles_aiguillage(cfg, sortie)
            registre = inventorier_depot(
                depot_dir=cfg.depot_brut,
                regles_path=regles_path,
                sortie_dir=sortie,
            )
        stats_manifeste = appliquer_manifeste(cfg.depot_brut, registre, sortie)
        lignes_lues = int(stats_manifeste.get("lignes_appliquees") or 0) - int(
            curseur_avant.get("lignes_appliquees") or 0
        )
        if lignes_lues > 0:
            lire_dossier_depot(registre, cfg.depot_brut, sortie)
            actualiser_pieces_tous(sortie / "cache-lecture")

        nouvelles, modifiees, supprimees = classer_pieces(snapshot, registre)
        chemin_attente = cfg.chemin_attente_synchronisation()
        attente = _charger_attente(chemin_attente)
        a_remplacer = _uniq(
            [*nouvelles, *modifiees, *attente["pieces_a_remplacer"]]
        )
        supprimees_travail = _uniq([*supprimees, *attente["pieces_supprimees"]])
        deja_supprimees = set(supprimees_travail)
        a_remplacer = [piece for piece in a_remplacer if piece not in deja_supprimees]
        ontologie_sha = sha256_ontologie(cfg.ontologie)
        eti_path = cfg.chemin_etiquettes()
        existantes = charger_etiquettes_si_present(eti_path)
        ontologie_changee = str(existantes.get("ontologie_sha256") or "") != ontologie_sha

        if (
            not a_remplacer
            and not supprimees_travail
            and not ontologie_changee
        ):
            if not a_blanc:
                _supprimer_attente(chemin_attente)
            resume = _resume_vide(
                horodatage=horodatage,
                lignes_lues=max(lignes_lues, 0),
                duree_s=round(time.monotonic() - started, 3),
            )
            rec_path = cfg.reconstitution / "registre.json"
            journaliser = False
            a_ecrit = False
            if rec_path.is_file():
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
                if isinstance(rec, dict):
                    onto = json.loads(cfg.ontologie.read_text(encoding="utf-8"))
                    if not isinstance(onto, dict):
                        onto = {}
                    sources = onto.setdefault("sources", {})
                    if isinstance(sources, dict):
                        sources["drive_registre"] = str(sortie / "registre.json")
                    rec_stats = reconstituer_unites_continues(
                        cfg,
                        registre=rec,
                        ontologie=onto,
                        a_blanc=a_blanc,
                        declencheur="synchroniser",
                    )
                    resume.update(rec_stats)
                    if rec_stats.get("unites_refaites") or rec_stats.get(
                        "unites_echouees"
                    ):
                        a_ecrit = True
                        if not a_blanc:
                            resume["duree_s"] = round(time.monotonic() - started, 3)
                            journaliser = True
            return _clore_passe(
                cfg,
                resume,
                a_blanc=a_blanc,
                declencheur=declencheur,
                journaliser=journaliser,
                a_ecrit=a_ecrit,
            )

        corpus = sortie / "cache-lecture" / "pieces" / "tous"
        depuis_attente = str(attente.get("depuis") or "") or horodatage
        restant_remplacer = list(a_remplacer)
        restant_supprimer = list(supprimees_travail)

        def _sauver_attente() -> None:
            _persister_attente(
                chemin_attente,
                pieces_a_remplacer=restant_remplacer,
                pieces_supprimees=restant_supprimer,
                depuis=depuis_attente,
                a_blanc=a_blanc,
            )

        _sauver_attente()

        passages_remplaces = 0
        embeddings_emis = 0
        embeddings_reutilises = 0
        plafond_atteint = False
        ids_a_retirer: set[str] = set()
        pieces_etiquetees: list[str] = []
        pieces_supprimees_faites: list[str] = []
        traitement_interrompu = False

        try:
            conn = connection
            if conn is None:
                conn = _ouvrir_connexion(cfg)
                connexion_ouverte = True
            index = charger_index_texte_vecteur(conn, cfg.vault_ref)
            if a_blanc:
                index = dict(index)

            compteur_effectif = compteur
            if compteur_effectif is None and not a_blanc:
                compteur_effectif = make_compteur_embeddings_passages(
                    cfg.racine, plafond=plafond
                )
            compteur_simulation = compteur_effectif if compteur_effectif is not None else compteur

            def reserver() -> None:
                if compteur_effectif is not None:
                    compteur_effectif.reserver_emission()

            embedder = embed_fn if embed_fn is not None else _embed_defaut

            for piece in supprimees_travail:
                ids_a_retirer |= _ids_passages(
                    lister_passages_corpus(conn, cfg.vault_ref, piece)
                )
                if not a_blanc:
                    supprimer_passages_piece(conn, cfg.vault_ref, piece)
                pieces_supprimees_faites.append(piece)
                if piece in restant_supprimer:
                    restant_supprimer.remove(piece)

            for rang, piece in enumerate(a_remplacer):
                piece_path = corpus / piece
                if not piece_path.is_file():
                    if piece not in restant_remplacer:
                        restant_remplacer.append(piece)
                    continue
                passages = chunk_piece(piece, corpus=corpus)
                anciens = lister_passages_corpus(conn, cfg.vault_ref, piece)
                try:
                    if a_blanc:
                        stats_piece = _stats_remplacement_a_blanc(
                            passages,
                            index,
                            compteur=compteur_simulation,
                            plafond=plafond,
                        )
                    else:
                        stats_piece = remplacer_passages_piece(
                            conn,
                            vault_ref=cfg.vault_ref,
                            piece=piece,
                            passages=passages,
                            embed_fn=embedder,
                            index_texte_vecteur=index,
                            reserver_emission=reserver,
                            ecrire=True,
                        )
                except PlafondAtteint:
                    plafond_atteint = True
                    restant_remplacer = _uniq(
                        [piece for piece in a_remplacer[rang:] if piece not in deja_supprimees]
                    )
                    break
                ids_a_retirer |= _ids_passages(anciens)
                passages_remplaces += int(stats_piece["passages_remplaces"])
                embeddings_emis += int(stats_piece["embeddings_emis"])
                embeddings_reutilises += int(stats_piece["embeddings_reutilises"])
                pieces_etiquetees.append(piece)
                if piece in restant_remplacer:
                    restant_remplacer.remove(piece)

            ontologie = json.loads(cfg.ontologie.read_text(encoding="utf-8"))
            if not isinstance(ontologie, dict):
                ontologie = {}
            sources = ontologie.setdefault("sources", {})
            if isinstance(sources, dict):
                sources["drive_registre"] = str(sortie / "registre.json")

            if ontologie_changee:
                rows_eti = lister_passages_corpus(conn, cfg.vault_ref)
                payload_eti = reetiqueter(
                    ontologie,
                    _passages_pour_etiquettes(rows_eti),
                    existantes,
                    ontologie_sha256=ontologie_sha,
                    tout=True,
                )
            else:
                rows_eti = []
                for piece in pieces_etiquetees:
                    rows_eti.extend(lister_passages_corpus(conn, cfg.vault_ref, piece))
                payload_eti = reetiqueter(
                    ontologie,
                    _passages_pour_etiquettes(rows_eti),
                    existantes,
                    ontologie_sha256=ontologie_sha,
                    ids_a_retirer=ids_a_retirer,
                    tout=False,
                )
            if not a_blanc:
                ecrire_etiquettes_atomique(eti_path, payload_eti)

            unites_ids: list[str] = []
            remplacees = set(pieces_etiquetees)
            pieces_unites = _uniq(
                [
                    piece
                    for piece in (*modifiees, *attente["pieces_a_remplacer"])
                    if piece in remplacees
                ]
                + pieces_supprimees_faites
            )
            rec_path = cfg.reconstitution / "registre.json"
            rec: dict[str, Any] = {}
            if rec_path.is_file():
                contexte = (
                    verrou_registre(cfg.reconstitution)
                    if not a_blanc
                    else nullcontext()
                )
                with contexte:
                    loaded = json.loads(rec_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        rec = loaded
                        if pieces_unites:
                            unites_ids = marquer_unites_perimees(
                                rec,
                                paquets_dir=cfg.reconstitution / "paquets",
                                pieces=pieces_unites,
                                depuis=horodatage,
                                passe=horodatage,
                            )
                            if unites_ids and not a_blanc:
                                ecrire_json_atomique(rec_path, rec)

            rec_stats = reconstituer_unites_continues(
                cfg,
                registre=rec,
                ontologie=ontologie,
                a_blanc=a_blanc,
                declencheur="synchroniser",
            )

            _sauver_attente()
            en_attente = _uniq([*restant_remplacer, *restant_supprimer])

            resultat: dict[str, Any] = {
                "horodatage": horodatage,
                "lignes_lues": max(lignes_lues, 0),
                "pieces": {
                    "nouvelles": nouvelles,
                    "modifiees": modifiees,
                    "supprimees": supprimees,
                },
                "passages_remplaces": passages_remplaces,
                "embeddings_emis": embeddings_emis,
                "embeddings_reutilises": embeddings_reutilises,
                "unites_perimees": unites_ids,
                "unites_refaites": rec_stats.get("unites_refaites") or [],
                "unites_echouees": rec_stats.get("unites_echouees") or [],
                "unites_a_refaire": rec_stats.get("unites_a_refaire") or [],
                "duree_reconstitution_s": rec_stats.get("duree_reconstitution_s") or 0.0,
                "duree_s": round(time.monotonic() - started, 3),
                "erreurs": [],
                "plafond_atteint": plafond_atteint,
                "en_attente": en_attente,
            }
            return _clore_passe(
                cfg,
                resultat,
                a_blanc=a_blanc,
                declencheur=declencheur,
                journaliser=not a_blanc,
                a_ecrit=not a_blanc,
            )
        except BaseException:
            traitement_interrompu = True
            raise
        finally:
            if traitement_interrompu:
                _sauver_attente()
    finally:
        if connexion_ouverte:
            fermer = getattr(conn, "close", None)
            if callable(fermer):
                fermer()
        if tmp_blanc is not None:
            shutil.rmtree(tmp_blanc, ignore_errors=True)
        relacher_verrou_synchronisation(handle_verrou)
