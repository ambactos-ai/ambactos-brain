# SPDX-License-Identifier: AGPL-3.0-only
"""Lecteur des situations composites : fichier JSON, classement hybride, rendu agent.

Le classement reprend `classer_faits` (SQL lexical, cosinus, RRF de la g12).
L'instrument rattache un élément à une seule pièce : une situation multi-pièces
est rendue en un élément par pièce, même rang, lettres de cette pièce seulement.
"""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .lecteur_faits_classes import (
    Fait,
    FaitNote,
    classer_faits,
    empreinte_texte,
)

DEFAULT_SITUATIONS_ROOT = (
    BANC_VAR / 'dossiers' / 'mrfs' / 'situations'
)
FICHIER_RECONCILIEES = "situations-reconciliees.json"
FICHIER_FUSIONNEES = "situations-fusionnees.json"


def chemin_situations(
    dossier: str,
    *,
    chemin: Path | str | None = None,
    situations_root: Path | None = None,
) -> Path:
    if chemin is not None:
        return Path(chemin)
    root = situations_root or DEFAULT_SITUATIONS_ROOT
    dossier_dir = root / dossier
    reconciliees = dossier_dir / FICHIER_RECONCILIEES
    if reconciliees.is_file():
        return reconciliees
    return dossier_dir / FICHIER_FUSIONNEES


def statistiques_chargement(
    dossier: str,
    *,
    chemin: Path | str | None = None,
    situations_root: Path | None = None,
    n_situations: int | None = None,
) -> dict[str, Any]:
    path = chemin_situations(
        dossier, chemin=chemin, situations_root=situations_root
    )
    stats: dict[str, Any] = {
        "fichier": path.name,
        "chemin": str(path),
        "reconcilie": path.name == FICHIER_RECONCILIEES,
    }
    if n_situations is not None:
        stats["n_situations"] = n_situations
    return stats


def charger_situations(
    dossier: str,
    *,
    chemin: Path | str | None = None,
    situations_root: Path | None = None,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    path = chemin_situations(
        dossier, chemin=chemin, situations_root=situations_root
    )
    if not path.is_file():
        raise RuntimeError(f"fichier de situations absent : {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"aucune situation dans {path}")
    situations = payload.get("situations")
    if not isinstance(situations, list) or not situations:
        raise RuntimeError(f"aucune situation dans {path}")
    loaded = list(situations)
    if stats is not None:
        stats.clear()
        stats.update(
            statistiques_chargement(
                dossier,
                chemin=path,
                situations_root=situations_root,
                n_situations=len(loaded),
            )
        )
    return loaded


def _cle_passage(
    piece: str,
    ligne_debut: Any,
    ligne_fin: Any,
) -> tuple[str, Any, Any]:
    return (piece, ligne_debut, ligne_fin)


def texte_classe(
    situation: Mapping[str, Any],
    *,
    lettres_chronologie: bool = False,
) -> str:
    """Titre, résumés, ce que chaque acteur dit, lettre des preuves."""
    parts: list[str] = []
    passages_lettre: set[tuple[str, Any, Any]] = set()
    titre = str(situation.get("titre") or "").strip()
    if titre:
        parts.append(titre)
    for resume in situation.get("resumes_par_piece") or []:
        if not isinstance(resume, Mapping):
            continue
        text = str(resume.get("resume") or "").strip()
        if text:
            parts.append(text)
    if lettres_chronologie:
        for moment in situation.get("chronologie") or []:
            if not isinstance(moment, Mapping):
                continue
            quoi = str(moment.get("quoi") or "").strip()
            if quoi:
                parts.append(quoi)
            lettre = str(moment.get("lettre") or "")
            if not lettre:
                continue
            piece = str(moment.get("piece") or "")
            key = _cle_passage(
                piece, moment.get("ligne_debut"), moment.get("ligne_fin")
            )
            if key in passages_lettre:
                continue
            parts.append(lettre)
            passages_lettre.add(key)
    for perception in situation.get("perceptions") or []:
        if not isinstance(perception, Mapping):
            continue
        dit = str(perception.get("ce_qu_il_dit") or "").strip()
        if dit:
            parts.append(dit)
        piece = str(perception.get("piece") or "")
        for preuve in perception.get("preuves") or []:
            if not isinstance(preuve, Mapping):
                continue
            lettre = str(preuve.get("lettre") or "")
            if not lettre:
                continue
            if lettres_chronologie:
                key = _cle_passage(
                    piece, preuve.get("ligne_debut"), preuve.get("ligne_fin")
                )
                if key in passages_lettre:
                    continue
                passages_lettre.add(key)
            parts.append(lettre)
    return "\n".join(parts)


def pieces_de(situation: Mapping[str, Any]) -> list[str]:
    raw = situation.get("pieces") or []
    pieces = [str(item) for item in raw if item]
    if pieces:
        return pieces
    seen: list[str] = []
    for perception in situation.get("perceptions") or []:
        if not isinstance(perception, Mapping):
            continue
        piece = str(perception.get("piece") or "").strip()
        if piece and piece not in seen:
            seen.append(piece)
    for moment in situation.get("chronologie") or []:
        if not isinstance(moment, Mapping):
            continue
        piece = str(moment.get("piece") or "").strip()
        if piece and piece not in seen:
            seen.append(piece)
    return seen


def _ligne_min(situation: Mapping[str, Any]) -> int:
    values: list[int] = []
    for perception in situation.get("perceptions") or []:
        if not isinstance(perception, Mapping):
            continue
        for preuve in perception.get("preuves") or []:
            if not isinstance(preuve, Mapping):
                continue
            debut = preuve.get("ligne_debut")
            if isinstance(debut, int):
                values.append(debut)
    for moment in situation.get("chronologie") or []:
        if not isinstance(moment, Mapping):
            continue
        debut = moment.get("ligne_debut")
        if isinstance(debut, int):
            values.append(debut)
    return min(values) if values else 0


def situation_vers_fait(
    situation: Mapping[str, Any],
    *,
    lettres_chronologie: bool = False,
) -> Fait:
    pieces = pieces_de(situation)
    source = pieces[0] if pieces else ""
    line = _ligne_min(situation)
    return Fait(
        id=str(situation["id"]),
        text=texte_classe(situation, lettres_chronologie=lettres_chronologie),
        source=source,
        line_start=line,
        line_end=line,
    )


def situations_vers_faits(
    situations: Sequence[Mapping[str, Any]],
    *,
    lettres_chronologie: bool = False,
) -> list[Fait]:
    return [
        situation_vers_fait(situation, lettres_chronologie=lettres_chronologie)
        for situation in situations
    ]


def _filtrer_piece(situation: Mapping[str, Any], piece: str) -> dict[str, Any]:
    payload = dict(situation)
    payload["chronologie"] = [
        moment
        for moment in situation.get("chronologie") or []
        if isinstance(moment, Mapping) and moment.get("piece") == piece
    ]
    payload["perceptions"] = [
        perception
        for perception in situation.get("perceptions") or []
        if isinstance(perception, Mapping) and perception.get("piece") == piece
    ]
    payload["points_de_desaccord"] = [
        item
        for item in situation.get("points_de_desaccord") or []
        if isinstance(item, Mapping) and item.get("piece") == piece
    ]
    payload["non_etabli"] = [
        item
        for item in situation.get("non_etabli") or []
        if isinstance(item, Mapping) and item.get("piece") == piece
    ]
    if "non_etabli_dans_la_piece_mais_etabli_dans_le_dossier" in situation:
        payload["non_etabli_dans_la_piece_mais_etabli_dans_le_dossier"] = [
            item
            for item in situation.get(
                "non_etabli_dans_la_piece_mais_etabli_dans_le_dossier"
            )
            or []
            if isinstance(item, Mapping) and item.get("piece") == piece
        ]
    return payload


def _est_reconcilie(situation: Mapping[str, Any]) -> bool:
    return "non_etabli_dans_la_piece_mais_etabli_dans_le_dossier" in situation


def _ligne_non_etabli_ancien(item: Mapping[str, Any]) -> str:
    return f"  - ({item['piece'][:2]}) {item['point']}"


def _ligne_non_etabli_dossier(item: Mapping[str, Any]) -> str:
    importance = str(item.get("importance") or "").strip()
    statut = str(item.get("statut") or "")
    suffix = f" [{importance}]" if importance else ""
    if statut == "incertain":
        suffix += " incertain"
    return f"  - ({item['piece'][:2]}) {item['point']}{suffix}"


def _ligne_etabli_ailleurs(item: Mapping[str, Any]) -> str:
    piece = str(item.get("piece") or "")
    etablies = ", ".join(str(nom) for nom in (item.get("pieces") or []) if nom)
    return (
        f"  - ({piece[:2]}) {item['point']} "
        f"(relevé dans {piece} ; établi par {etablies})"
    )


def _ref_lignes_passage(
    piece: str,
    ligne_debut: Any,
    ligne_fin: Any,
    *,
    lettre_deja: bool,
) -> str:
    ref = f"{piece}:{ligne_debut}-{ligne_fin}"
    if lettre_deja:
        ref += " (lettre ci-dessus)"
    return ref


def rendu_agent(
    situation: Mapping[str, Any],
    *,
    piece: str | None = None,
    lettres_chronologie: bool = False,
) -> str:
    """Rendu factuel, déterministe : aucune phrase n'est écrite par un modèle.

    Forme de `consolider.rendu_agent`. Si `piece` est fourni, seules les preuves
    (et la chronologie, les désaccords, le non établi) de cette pièce sont rendues.
    """
    s = _filtrer_piece(situation, piece) if piece is not None else dict(situation)
    natures = s.get("natures") or []
    pieces = situation.get("pieces") or []
    sources = s.get("sources") or []
    out = [
        f"SITUATION {s['id']} — {s['titre']}",
        f"natures: {', '.join(natures)} | pièces: {len(pieces)}",
        "sources: " + ", ".join(sources),
        "acteurs:",
    ]
    for acteur in s.get("acteurs") or []:
        roles = " ; ".join(
            f"{r['role']}{' [hors catalogue]' if r['hors_catalogue'] else ''} ({r['piece'][:2]})"
            for r in acteur.get("roles") or []
        )
        out.append(
            f"  - {acteur['nom']} | {acteur['type_entite']} | rôles: {roles}"
        )
    passages_lettre: set[tuple[str, Any, Any]] = set()
    out.append("chronologie:")
    for moment in s.get("chronologie") or []:
        piece_m = str(moment.get("piece") or "")
        key = _cle_passage(
            piece_m, moment.get("ligne_debut"), moment.get("ligne_fin")
        )
        lettre = str(moment.get("lettre") or "")
        if lettres_chronologie and lettre:
            deja = key in passages_lettre
            ref = _ref_lignes_passage(
                piece_m,
                moment.get("ligne_debut"),
                moment.get("ligne_fin"),
                lettre_deja=deja,
            )
            out.append(
                f"  - {moment['quand_texte']} [{moment.get('iso') or 'non résolue'}] : "
                f"{moment['quoi']} ({ref})"
            )
            if not deja:
                out.extend(f"      > {ligne}" for ligne in lettre.splitlines())
                passages_lettre.add(key)
        else:
            out.append(
                f"  - {moment['quand_texte']} [{moment.get('iso') or 'non résolue'}] : "
                f"{moment['quoi']} "
                f"({piece_m}:{moment.get('ligne_debut')}-{moment.get('ligne_fin')})"
            )
    out.append("perceptions:")
    for perception in s.get("perceptions") or []:
        piece_p = str(perception.get("piece") or "")
        refs_parts: list[str] = []
        for preuve in perception.get("preuves") or []:
            key = _cle_passage(
                piece_p, preuve.get("ligne_debut"), preuve.get("ligne_fin")
            )
            lettre = str(preuve.get("lettre") or "")
            deja = lettres_chronologie and key in passages_lettre and bool(lettre)
            refs_parts.append(
                _ref_lignes_passage(
                    piece_p,
                    preuve.get("ligne_debut"),
                    preuve.get("ligne_fin"),
                    lettre_deja=deja,
                )
            )
        refs = ", ".join(refs_parts)
        out.append(
            f"  - {perception['acteur']} | {perception['position']} | "
            f"{refs or 'sans preuve'}"
        )
        for preuve in perception.get("preuves") or []:
            lettre = str(preuve.get("lettre") or "")
            if not lettre:
                continue
            key = _cle_passage(
                piece_p, preuve.get("ligne_debut"), preuve.get("ligne_fin")
            )
            if lettres_chronologie and key in passages_lettre:
                continue
            out.extend(f"      > {ligne}" for ligne in lettre.splitlines())
            if lettres_chronologie:
                passages_lettre.add(key)
    if s.get("points_de_desaccord"):
        out.append("desaccords:")
        out.extend(
            f"  - ({item['piece'][:2]}) {item['point']}"
            for item in s["points_de_desaccord"]
        )
    if _est_reconcilie(s):
        if s.get("non_etabli"):
            out.append("non_etabli_dans_le_dossier:")
            out.extend(_ligne_non_etabli_dossier(item) for item in s["non_etabli"])
        etablis = s.get("non_etabli_dans_la_piece_mais_etabli_dans_le_dossier") or []
        if etablis:
            out.append("etabli_par_une_autre_piece:")
            out.extend(_ligne_etabli_ailleurs(item) for item in etablis)
    elif s.get("non_etabli"):
        out.append("non_etabli:")
        out.extend(_ligne_non_etabli_ancien(item) for item in s["non_etabli"])
    return "\n".join(out)


def lettres_de_piece(
    situation: Mapping[str, Any],
    piece: str,
    *,
    lettres_chronologie: bool = False,
) -> str:
    blocs: list[str] = []
    passages_lettre: set[tuple[str, Any, Any]] = set()
    if lettres_chronologie:
        for moment in situation.get("chronologie") or []:
            if not isinstance(moment, Mapping):
                continue
            if moment.get("piece") != piece:
                continue
            lettre = str(moment.get("lettre") or "")
            if not lettre:
                continue
            key = _cle_passage(
                piece, moment.get("ligne_debut"), moment.get("ligne_fin")
            )
            if key in passages_lettre:
                continue
            blocs.append(lettre)
            passages_lettre.add(key)
    for perception in situation.get("perceptions") or []:
        if not isinstance(perception, Mapping):
            continue
        if perception.get("piece") != piece:
            continue
        for preuve in perception.get("preuves") or []:
            if not isinstance(preuve, Mapping):
                continue
            lettre = str(preuve.get("lettre") or "")
            if not lettre:
                continue
            if lettres_chronologie:
                key = _cle_passage(
                    piece, preuve.get("ligne_debut"), preuve.get("ligne_fin")
                )
                if key in passages_lettre:
                    continue
                passages_lettre.add(key)
            blocs.append(lettre)
    return "\n".join(blocs)


def _lignes_piece(situation: Mapping[str, Any], piece: str) -> tuple[int | None, int | None]:
    debuts: list[int] = []
    fins: list[int] = []
    for perception in situation.get("perceptions") or []:
        if not isinstance(perception, Mapping) or perception.get("piece") != piece:
            continue
        for preuve in perception.get("preuves") or []:
            if not isinstance(preuve, Mapping):
                continue
            debut = preuve.get("ligne_debut")
            fin = preuve.get("ligne_fin")
            if isinstance(debut, int):
                debuts.append(debut)
            if isinstance(fin, int):
                fins.append(fin)
    if not debuts:
        return None, None
    return min(debuts), max(fins) if fins else None


def element_pour_piece(
    situation: Mapping[str, Any],
    piece: str,
    *,
    rank: int,
    score: float,
    lettres_chronologie: bool = False,
) -> dict[str, Any]:
    texte = rendu_agent(
        situation, piece=piece, lettres_chronologie=lettres_chronologie
    )
    lettres = lettres_de_piece(
        situation, piece, lettres_chronologie=lettres_chronologie
    )
    line_start, line_end = _lignes_piece(situation, piece)
    provenance: dict[str, Any] = {
        "source_file_path": piece,
        "citation": lettres,
        "situation_id": str(situation.get("id") or ""),
    }
    if isinstance(line_start, int):
        provenance["line_start"] = line_start
    if isinstance(line_end, int):
        provenance["line_end"] = line_end
    return {
        "kind": "mrfs_situation",
        "id": f"{piece}:{situation.get('id')}",
        "rank": int(rank),
        "score": float(score),
        "text": texte,
        "source": piece,
        "snippet": texte[:280],
        "provenance": provenance,
        "payload": {"metadata": provenance},
    }


def rendre_situations_classees(
    notes: Sequence[FaitNote],
    situations: Sequence[Mapping[str, Any]],
    *,
    lettres_chronologie: bool = False,
) -> list[dict[str, Any]]:
    par_id = {str(situation["id"]): situation for situation in situations}
    elements: list[dict[str, Any]] = []
    for rank, note in enumerate(notes, start=1):
        situation = par_id[note.fait.id]
        pieces = pieces_de(situation)
        if not pieces:
            elements.append(
                element_pour_piece(
                    situation,
                    "",
                    rank=rank,
                    score=note.score,
                    lettres_chronologie=lettres_chronologie,
                )
            )
            continue
        for piece in pieces:
            elements.append(
                element_pour_piece(
                    situation,
                    piece,
                    rank=rank,
                    score=note.score,
                    lettres_chronologie=lettres_chronologie,
                )
            )
    return elements


def _exiger_embeddings(
    faits: Sequence[Fait],
    embeddings: Mapping[str, Sequence[float]],
) -> None:
    for fait in faits:
        if fait.empreinte not in embeddings:
            raise RuntimeError(
                "embedding de situation absent "
                f"({fait.source}:{fait.line_start} {fait.empreinte[:12]}…)"
            )


def classer_situations(
    situations: Sequence[Mapping[str, Any]],
    question: str,
    *,
    connection=None,
    scorer_lexical: Callable[[Sequence[Fait], str], dict[str, float]] | None = None,
    question_vector: Sequence[float] | None = None,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    scores_lex: Mapping[str, float] | None = None,
    scores_vec: Mapping[str, float] | None = None,
    lettres_chronologie: bool = False,
) -> list[FaitNote]:
    """Note toutes les situations. Pas de vivier, pas de seuil, pas de k."""
    faits = situations_vers_faits(
        situations, lettres_chronologie=lettres_chronologie
    )
    if embeddings is not None:
        _exiger_embeddings(faits, embeddings)
    return classer_faits(
        faits,
        question,
        mode="hybride",
        connection=connection,
        scorer_lexical=scorer_lexical,
        question_vector=question_vector,
        embeddings=embeddings,
        scores_lex=scores_lex,
        scores_vec=scores_vec,
    )


def contexte_situations_composites(
    question: str,
    *,
    dossier: str,
    chemin: Path | str | None = None,
    situations_root: Path | None = None,
    situations: Sequence[Mapping[str, Any]] | None = None,
    connection=None,
    scorer_lexical: Callable[[Sequence[Fait], str], dict[str, float]] | None = None,
    question_vector: Sequence[float] | None = None,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    lettres_chronologie: bool = False,
) -> list[dict[str, Any]]:
    loaded = (
        list(situations)
        if situations is not None
        else charger_situations(
            dossier, chemin=chemin, situations_root=situations_root
        )
    )
    notes = classer_situations(
        loaded,
        question,
        connection=connection,
        scorer_lexical=scorer_lexical,
        question_vector=question_vector,
        embeddings=embeddings,
        lettres_chronologie=lettres_chronologie,
    )
    return rendre_situations_classees(
        notes, loaded, lettres_chronologie=lettres_chronologie
    )


def empreinte_situation(
    situation: Mapping[str, Any],
    *,
    lettres_chronologie: bool = False,
) -> str:
    return empreinte_texte(
        texte_classe(situation, lettres_chronologie=lettres_chronologie)
    )
