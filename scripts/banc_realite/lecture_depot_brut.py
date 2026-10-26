# SPDX-License-Identifier: AGPL-3.0-only
"""Lecture depuis un dépôt brut connecteur (Drive + Gmail), sans réseau."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from banc_realite.lecture_drive import (
    EXPORT_DOCX,
    EXPORT_XLSX,
    ClientDrive,
    _base_et_piece,
    _deja_lu,
    _groupe_traitement,
    _lire_html_bytes,
    _retirer_piece_cache,
    actualiser_pieces_tous,
    aiguiller,
    charger_regles,
    construire_registre,
    ecrire_json_atomique,
    fusionner_registre,
    lire_dossier,
)

CURSEUR_MANIFESTE_NOM = "depot-brut.curseur.json"
MIME_MESSAGE = "message/rfc822"
PREFIX_GMAIL = "gmail:"


def valider_id_depot(identifiant: str) -> None:
    if not identifiant or ".." in identifiant or "/" in identifiant or "\\" in identifiant or "\0" in identifiant:
        raise ValueError(f"identifiant de fichier invalide: {identifiant!r}")


class ClientDriveDepot:
    """Client Drive v3 lu depuis un dépôt brut local."""

    def __init__(self, depot_dir: Path) -> None:
        self._depot = depot_dir.resolve()
        self._pieces_jointes: dict[str, Path] = {}
        self._indexer_pieces_jointes()

    def _indexer_pieces_jointes(self) -> None:
        messages_dir = self._depot / "gmail" / "messages"
        if not messages_dir.is_dir():
            return
        for path in messages_dir.glob("*.json"):
            message_id = path.stem
            valider_id_depot(message_id)
            payload = json.loads(path.read_text(encoding="utf-8"))
            prefix_attendu = f"gmail/pieces-jointes/{message_id}/"
            for pj in payload.get("pieces_jointes") or []:
                attachment_id = str(pj.get("attachment_id") or "")
                if not attachment_id:
                    continue
                valider_id_depot(attachment_id)
                rel = str(pj.get("fichier") or "").replace("\\", "/")
                if not rel:
                    continue
                rel_path = Path(rel)
                if rel_path.is_absolute() or ".." in rel_path.parts:
                    raise ValueError(f"chemin pièce jointe invalide: {rel!r}")
                if not rel.startswith(prefix_attendu):
                    raise ValueError(
                        f"chemin pièce jointe hors zone attendue {prefix_attendu!r}: {rel!r}"
                    )
                fichier = (self._depot / rel_path).resolve()
                try:
                    fichier.relative_to(self._depot)
                except ValueError:
                    raise ValueError(f"chemin pièce jointe hors dépôt: {rel!r}") from None
                if not fichier.is_file():
                    continue
                cle = f"{PREFIX_GMAIL}{message_id}:{attachment_id}"
                self._pieces_jointes[cle] = fichier

    def lister(self, drive_id: str) -> list[dict[str, Any]]:
        chemin = self._depot / "drive" / "fichiers.json"
        if not chemin.is_file():
            raise FileNotFoundError(chemin)
        fichiers = json.loads(chemin.read_text(encoding="utf-8"))
        # Un fichier à la corbeille (clé `trashed` du connecteur) n'entre jamais à l'inventaire,
        # même s'il traîne dans un instantané : la suppression fait foi par le manifeste.
        return [item for item in fichiers if not item.get("trashed")]

    def telecharger(self, file_id: str) -> bytes:
        if file_id.startswith(PREFIX_GMAIL):
            path = self._pieces_jointes.get(file_id)
            if path is None:
                raise FileNotFoundError(f"pièce jointe absente pour {file_id!r}")
            return path.read_bytes()
        valider_id_depot(file_id)
        path = self._depot / "drive" / "brut" / file_id
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes()

    def exporter(self, file_id: str, mime: str) -> bytes:
        valider_id_depot(file_id)
        if mime == EXPORT_XLSX:
            suffix = ".xlsx"
        elif mime == EXPORT_DOCX:
            suffix = ".docx"
        else:
            raise ValueError(f"mime d'export non supporté: {mime!r}")
        path = self._depot / "drive" / "export" / f"{file_id}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes()


def _internal_date_iso(internal_date: Any) -> str:
    if internal_date is None:
        return ""
    if isinstance(internal_date, str):
        return internal_date
    try:
        ms = int(internal_date)
    except (TypeError, ValueError):
        return str(internal_date)
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def construire_registre_gmail(depot_dir: Path) -> list[dict[str, Any]]:
    messages_dir = depot_dir / "gmail" / "messages"
    if not messages_dir.is_dir():
        return []
    entrees: list[dict[str, Any]] = []
    for path in sorted(messages_dir.glob("*.json")):
        message_id = path.stem
        valider_id_depot(message_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        thread_id = str(payload.get("thread_id") or "")
        subject = str(payload.get("subject") or "")
        taille = 0
        for pj in payload.get("pieces_jointes") or []:
            try:
                taille += int(pj.get("taille") or 0)
            except (TypeError, ValueError):
                pass
        entree_msg = {
            "id": f"{PREFIX_GMAIL}{message_id}",
            "nom": subject,
            "chemin": f"gmail/{thread_id}" if thread_id else "gmail",
            "mime": MIME_MESSAGE,
            "taille": taille,
            "modifie": _internal_date_iso(payload.get("internal_date")),
            "lien": "",
        }
        entrees.append(entree_msg)
        for pj in payload.get("pieces_jointes") or []:
            attachment_id = str(pj.get("attachment_id") or "")
            if not attachment_id:
                continue
            valider_id_depot(attachment_id)
            entrees.append(
                {
                    "id": f"{PREFIX_GMAIL}{message_id}:{attachment_id}",
                    "nom": str(pj.get("nom") or "piece"),
                    "chemin": f"gmail/{thread_id}/{message_id}" if thread_id else f"gmail/{message_id}",
                    "mime": str(pj.get("mime") or "application/octet-stream"),
                    "taille": int(pj.get("taille") or 0),
                    "modifie": entree_msg["modifie"],
                    "lien": "",
                    "lie_a": entree_msg["id"],
                }
            )
    entrees.sort(key=lambda e: (e["chemin"].lower(), e["nom"].lower()))
    return entrees


def _preserver_entrees_hors_instantane(
    ancien: list[dict[str, Any]],
    nouveau: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ids_nouveaux = {e["id"] for e in nouveau}
    fusion = list(nouveau)
    for entree in ancien:
        if entree["id"] not in ids_nouveaux:
            fusion.append(dict(entree))
    return fusion


def inventorier_depot(
    *,
    depot_dir: Path,
    regles_path: Path,
    sortie_dir: Path,
) -> list[dict[str, Any]]:
    depot_dir = depot_dir.resolve()
    client = ClientDriveDepot(depot_dir)
    fichiers = client.lister("")
    nouveau = construire_registre(fichiers) + construire_registre_gmail(depot_dir)
    regles = charger_regles(regles_path)
    aiguiller(nouveau, regles)
    sortie_dir.mkdir(parents=True, exist_ok=True)
    registre_path = sortie_dir / "registre.json"
    ancien: list[dict[str, Any]] = []
    if registre_path.is_file():
        ancien = json.loads(registre_path.read_text(encoding="utf-8"))
    nouveau_complet = _preserver_entrees_hors_instantane(ancien, nouveau)
    registre, _disparues = fusionner_registre(ancien, nouveau_complet)
    ecrire_json_atomique(registre_path, registre)
    return registre


def _charger_message(depot_dir: Path, message_id: str) -> dict[str, Any]:
    valider_id_depot(message_id)
    path = depot_dir / "gmail" / "messages" / f"{message_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _lire_corps_message(payload: dict[str, Any]) -> list[str]:
    corps_texte = payload.get("corps_texte")
    if isinstance(corps_texte, str) and corps_texte.strip():
        return corps_texte.splitlines()
    corps_html = payload.get("corps_html")
    if isinstance(corps_html, str) and corps_html.strip():
        lecture = _lire_html_bytes(corps_html.encode("utf-8"))
        return lecture.lignes
    return []


def _pieces_jointes_liees(
    registre: list[dict[str, Any]],
    message_entree_id: str,
    rang_par_id: dict[str, int],
) -> list[tuple[str, str]]:
    resultat: list[tuple[str, str]] = []
    for entree in registre:
        if entree.get("lie_a") != message_entree_id:
            continue
        nom = str(entree.get("nom") or "piece")
        rang = rang_par_id.get(entree["id"], 0)
        _, piece_nom = _base_et_piece(entree, rang)
        resultat.append((nom, piece_nom))
    return resultat


def lire_dossier_depot(
    registre: list[dict[str, Any]],
    depot_dir: Path,
    sortie_dir: Path,
    *,
    refaire: bool = False,
    seulement: set[str] | None = None,
    ocr_image: Any = None,
) -> None:
    depot_dir = depot_dir.resolve()
    client = ClientDriveDepot(depot_dir)
    rang_par_id = {entree["id"]: rang for rang, entree in enumerate(registre, start=1)}
    cache = sortie_dir / "cache-lecture"
    registre_path = sortie_dir / "registre.json"

    def ecrire_piece_mail(entree: dict[str, Any], rang: int) -> None:
        message_id = entree["id"][len(PREFIX_GMAIL) :]
        payload = _charger_message(depot_dir, message_id)
        headers = [
            f"De : {payload.get('from') or ''}",
            f"À : {payload.get('to') or ''}",
            f"Cc : {payload.get('cc') or ''}",
            f"Date : {payload.get('date') or ''}",
            f"Objet : {payload.get('subject') or ''}",
            f"Fil : {payload.get('thread_id') or ''}",
            "",
        ]
        lignes = headers + _lire_corps_message(payload)
        liees = _pieces_jointes_liees(registre, entree["id"], rang_par_id)
        if liees:
            if lignes and lignes[-1] != "":
                lignes.append("")
            lignes.append("Pièces jointes :")
            for nom, piece in liees:
                lignes.append(f"- {nom} → {piece}")
        texte = "\n".join(lignes)
        if texte and not texte.endswith("\n"):
            texte += "\n"
        traitement = entree.get("traitement") or ""
        groupe = _groupe_traitement(traitement)
        if not groupe:
            return
        pieces_dir = cache / "pieces" / groupe
        reperes_dir = cache / "reperes" / groupe
        pieces_dir.mkdir(parents=True, exist_ok=True)
        reperes_dir.mkdir(parents=True, exist_ok=True)
        base, piece_nom = _base_et_piece(entree, rang)
        piece_path = pieces_dir / piece_nom
        repere_path = reperes_dir / f"{base}.json"
        piece_path.write_text(texte, encoding="utf-8")
        repere_path.write_text("[]", encoding="utf-8")
        empreinte = entree.get("manifeste_sha256") or hashlib.sha256(texte.encode("utf-8")).hexdigest()
        entree["piece"] = piece_nom
        entree["groupe"] = groupe
        entree["lecture"] = {
            "lignes": len(lignes),
            "caracteres": len(texte.rstrip("\n")),
            "pages": 1,
            "ocr": False,
            "avertissements": [],
            "empreinte": empreinte,
            "modifie": entree.get("modifie") or "",
        }

    lire_dossier(
        registre,
        client,
        sortie_dir,
        refaire=refaire,
        seulement=seulement,
        ocr_image=ocr_image,
    )
    for rang, entree in enumerate(registre, start=1):
        if (entree.get("mime") or "") != MIME_MESSAGE:
            continue
        traitement = entree.get("traitement") or ""
        if traitement in ("registre_seul", "a_trancher"):
            continue
        groupe = _groupe_traitement(traitement)
        if not groupe:
            continue
        if seulement and traitement not in seulement:
            continue
        if not refaire and _deja_lu(entree, cache):
            continue
        try:
            ecrire_piece_mail(entree, rang)
        except Exception as exc:
            entree["lecture"] = {"erreur": f"{type(exc).__name__}: {exc}"}
            entree.pop("piece", None)
        ecrire_json_atomique(registre_path, registre)
    actualiser_pieces_tous(cache)


def _sha256_ligne(ligne: str) -> str:
    return hashlib.sha256(ligne.encode("utf-8")).hexdigest()


def _charger_curseur_manifeste(sortie_dir: Path) -> dict[str, Any]:
    path = sortie_dir / CURSEUR_MANIFESTE_NOM
    if not path.is_file():
        return {"lignes_appliquees": 0, "derniere_ligne_sha256": ""}
    return json.loads(path.read_text(encoding="utf-8"))


def _ecrire_curseur_manifeste(sortie_dir: Path, curseur: dict[str, Any]) -> None:
    ecrire_json_atomique(sortie_dir / CURSEUR_MANIFESTE_NOM, curseur)


def _id_manifeste_vers_registre(source: str, identifiant: str) -> str:
    if source == "gmail":
        if identifiant.startswith(PREFIX_GMAIL):
            return identifiant
        return f"{PREFIX_GMAIL}{identifiant}"
    return identifiant


def appliquer_manifeste(
    depot_dir: Path,
    registre: list[dict[str, Any]],
    sortie_dir: Path,
) -> dict[str, Any]:
    manifeste_path = depot_dir / "manifeste.jsonl"
    if not manifeste_path.is_file():
        raise FileNotFoundError(manifeste_path)
    lignes = manifeste_path.read_text(encoding="utf-8").splitlines()
    curseur = _charger_curseur_manifeste(sortie_dir)
    offset = int(curseur.get("lignes_appliquees") or 0)
    derniere_sha = str(curseur.get("derniere_ligne_sha256") or "")
    if offset > 0 and offset <= len(lignes):
        attendu = _sha256_ligne(lignes[offset - 1])
        if derniere_sha and attendu != derniere_sha:
            raise ValueError("manifeste réécrit : curseur incohérent avec la ligne précédente")
    stats: dict[str, int] = {"ajout": 0, "maj": 0, "suppression": 0, "lignes_appliquees": offset}
    par_id = {e["id"]: e for e in registre}
    registre_path = sortie_dir / "registre.json"
    cache = sortie_dir / "cache-lecture"
    registre_modifie = False

    for index in range(offset, len(lignes)):
        brute = lignes[index]
        numero_ligne = index + 1
        if not brute.strip():
            continue
        try:
            effet = json.loads(brute)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifeste ligne {numero_ligne} : JSON invalide ({exc})") from exc
        action = effet.get("action")
        source = str(effet.get("source") or "drive")
        identifiant = str(effet.get("id") or "")
        sha = str(effet.get("sha256") or "")
        reg_id = _id_manifeste_vers_registre(source, identifiant)

        if action == "suppression":
            entree = par_id.pop(reg_id, None)
            if entree is not None:
                _retirer_piece_cache(sortie_dir, entree)
                registre[:] = [e for e in registre if e["id"] != reg_id]
                stats["suppression"] += 1
                registre_modifie = True
        elif action in ("ajout", "maj"):
            entree = par_id.get(reg_id)
            if entree is not None:
                entree["manifeste_sha256"] = sha
                lecture = entree.get("lecture")
                if not (isinstance(lecture, dict) and lecture.get("empreinte") == sha):
                    _retirer_piece_cache(sortie_dir, entree)
                    entree.pop("piece", None)
                    entree.pop("groupe", None)
                    entree.pop("lecture", None)
                    registre_modifie = True
                stats["maj" if action == "maj" else "ajout"] += 1
        else:
            raise ValueError(f"manifeste ligne {numero_ligne} : action inconnue {action!r}")

        offset = index + 1
        curseur = {"lignes_appliquees": offset, "derniere_ligne_sha256": _sha256_ligne(brute)}
        ecrire_json_atomique(registre_path, registre)
        _ecrire_curseur_manifeste(sortie_dir, curseur)
        stats["lignes_appliquees"] = offset

    if registre_modifie:
        actualiser_pieces_tous(cache)
    return stats
