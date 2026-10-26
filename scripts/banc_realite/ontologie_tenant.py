# SPDX-License-Identifier: AGPL-3.0-only
"""Proposition d'ontologie tenant v0 — déterministe, sans modèle."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .normalize import normalize_text

SCHEMA = "ontologie-tenant/0.1.0"


def _titre_normalise(titre: str) -> str:
    base = unicodedata.normalize("NFKD", titre or "")
    sans_accents = "".join(ch for ch in base if not unicodedata.combining(ch))
    compact = re.sub(r"\s+", " ", sans_accents.lower()).strip()
    return compact


def _forme_cle(cle: str) -> str:
    return re.sub(r"\d", "#", cle)


def _extraire_regle(regle: Mapping[str, Any]) -> dict[str, str]:
    bloc: dict[str, str] = {}
    for cle in ("chemin", "nom", "mime"):
        valeur = regle.get(cle)
        if isinstance(valeur, str) and valeur:
            bloc[cle] = valeur
    return bloc


def _piece_de_entree(entree: Mapping[str, Any]) -> str:
    piece = entree.get("piece")
    if isinstance(piece, str) and piece:
        return piece
    nom = entree.get("nom") or ""
    if nom and not nom.endswith(".md"):
        return f"{nom}.md"
    return nom


def _famille_par_piece(
    drive_registre: list[dict[str, Any]],
) -> dict[str, str]:
    cartographie: dict[str, str] = {}
    for entree in drive_registre:
        famille = entree.get("famille") or ""
        piece = _piece_de_entree(entree)
        if piece:
            cartographie[piece] = famille
        nom = entree.get("nom") or ""
        if nom:
            cartographie.setdefault(nom, famille)
    return cartographie


def proposer_familles(
    drive_registre: list[dict[str, Any]],
    regles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comptes_regle = Counter(
        entree.get("regle")
        for entree in drive_registre
        if entree.get("regle") and not entree.get("doublon_de")
    )
    familles: list[dict[str, Any]] = []
    for index, regle in enumerate(regles, start=1):
        pieces = comptes_regle.get(index, 0)
        if pieces <= 0:
            continue
        familles.append(
            {
                "nom": regle.get("famille") or f"regle-{index}",
                "regle": _extraire_regle(regle),
                "traitement": regle.get("traitement") or "",
                "raison": regle.get("raison") or "",
                "pieces": pieces,
                "source": "proposee",
            }
        )
    a_trancher = sum(
        1
        for entree in drive_registre
        if entree.get("traitement") == "a_trancher" and not entree.get("doublon_de")
    )
    if a_trancher:
        familles.append(
            {
                "nom": "a_trancher",
                "regle": {},
                "traitement": "a_trancher",
                "raison": "",
                "pieces": a_trancher,
                "source": "proposee",
            }
        )
    familles.sort(key=lambda f: (f["nom"], json.dumps(f["regle"], sort_keys=True)))
    return familles


def proposer_types(entites_registre: Mapping[str, Any]) -> list[dict[str, Any]]:
    entites = entites_registre.get("entites") or []
    par_type: dict[str, list[dict[str, Any]]] = {}
    for entite in entites:
        if not isinstance(entite, dict):
            continue
        typ = entite.get("type") or "autre"
        par_type.setdefault(typ, []).append(entite)

    types_proposes: list[dict[str, Any]] = []
    for typ in sorted(par_type):
        membres = par_type[typ]
        formes = Counter(
            _forme_cle(cle)
            for ent in membres
            for cle in (ent.get("cles") or [])
            if isinstance(cle, str) and cle.strip()
        )
        cles_attendues = [forme for forme, _ in formes.most_common(10)]
        exemples = sorted(
            {ent.get("nom") or "" for ent in membres if ent.get("nom")},
            key=lambda s: s.lower(),
        )[:5]
        types_proposes.append(
            {
                "nom": typ,
                "exemples": exemples,
                "cles_attendues": cles_attendues,
                "entites": len(membres),
                "source": "proposee",
            }
        )
    return types_proposes


def _charger_faits_entites(entites_dir: Path) -> list[tuple[str, str, dict[str, Any]]]:
    if not entites_dir.is_dir():
        return []
    charge: list[tuple[str, str, dict[str, Any]]] = []
    for path in sorted(entites_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entite = payload.get("entite") or {}
        nom = entite.get("nom") or path.stem
        typ = entite.get("type") or "autre"
        for fait in payload.get("faits") or []:
            if isinstance(fait, dict):
                charge.append((nom, typ, fait))
    return charge


def _cles_normalisees_entite(entite: Mapping[str, Any]) -> set[str]:
    cles: set[str] = set()
    for cle in (entite.get("cles") or []) + (entite.get("variantes") or []):
        if isinstance(cle, str) and cle.strip():
            cles.add(normalize_text(cle))
    nom = entite.get("nom")
    if isinstance(nom, str) and nom.strip():
        cles.add(normalize_text(nom))
    return cles


def proposer_relations(
    entites_registre: Mapping[str, Any],
    entites_dir: Path,
    famille_par_piece: Mapping[str, str],
) -> list[dict[str, Any]]:
    entites = [
        e for e in (entites_registre.get("entites") or []) if isinstance(e, dict)
    ]
    cles_par_entite = {e["nom"]: _cles_normalisees_entite(e) for e in entites if e.get("nom")}
    type_par_entite = {e["nom"]: e.get("type") or "autre" for e in entites if e.get("nom")}

    compteur: Counter[tuple[str, str]] = Counter()
    exemples: dict[tuple[str, str], dict[str, Any]] = {}

    faits = _charger_faits_entites(entites_dir)
    for nom_b, typ_b, fait in faits:
        versions = fait.get("versions") or []
        familles_vues: set[str] = set()
        for version in versions:
            if not isinstance(version, dict):
                continue
            piece = version.get("piece") or ""
            famille = famille_par_piece.get(piece, "")
            if famille:
                familles_vues.add(famille)
            valeur = version.get("valeur") or ""
            if not isinstance(valeur, str):
                valeur = str(valeur)
            val_norm = normalize_text(valeur)
            lignes = version.get("lignes") or []
            for nom_a, cles_a in cles_par_entite.items():
                if nom_a == nom_b:
                    continue
                if any(cle and cle in val_norm for cle in cles_a if len(cle) >= 3):
                    paire = (type_par_entite.get(nom_a, "autre"), typ_b)
                    compteur[paire] += 1
                    exemples.setdefault(
                        paire,
                        {
                            "de": nom_a,
                            "vers": nom_b,
                            "piece": piece,
                            "lignes": list(lignes) if isinstance(lignes, list) else [],
                        },
                    )
        if len(familles_vues) >= 2:
            paire = (typ_b, typ_b)
            compteur[paire] += 1
            premiere = next((v for v in versions if isinstance(v, dict)), {})
            exemples.setdefault(
                paire,
                {
                    "de": nom_b,
                    "vers": nom_b,
                    "piece": premiere.get("piece") or "",
                    "lignes": list(premiere.get("lignes") or []),
                },
            )

    relations: list[dict[str, Any]] = []
    for (de_type, vers_type), occurrences in sorted(compteur.items()):
        relations.append(
            {
                "de_type": de_type,
                "verbe": "cite",
                "vers_type": vers_type,
                "occurrences": occurrences,
                "exemple": exemples[(de_type, vers_type)],
                "source": "proposee",
            }
        )
    return relations


def proposer_concepts(groupes_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    groupes = groupes_payload.get("groupes") or []
    par_titre: dict[str, list[dict[str, Any]]] = {}
    for groupe in groupes:
        if not isinstance(groupe, dict):
            continue
        titre = groupe.get("titre") or ""
        cle = _titre_normalise(titre)
        if not cle:
            continue
        par_titre.setdefault(cle, []).append(groupe)

    concepts: list[dict[str, Any]] = []
    for cle in sorted(par_titre):
        membres = par_titre[cle]
        concepts_freq = Counter(
            g.get("concept") or "autre" for g in membres if isinstance(g, dict)
        )
        concept = concepts_freq.most_common(1)[0][0] if concepts_freq else "autre"
        pieces: set[str] = set()
        for groupe in membres:
            for ref in groupe.get("lignes") or []:
                if isinstance(ref, dict):
                    piece = ref.get("piece")
                    if isinstance(piece, str) and piece:
                        pieces.add(piece)
        concepts.append(
            {
                "nom": cle,
                "concept": concept,
                "pieces": sorted(pieces),
                "groupes": len(membres),
                "source": "proposee",
            }
        )
    return concepts


def _periode_entite(
    nom: str,
    unites: list[dict[str, Any]],
) -> list[str] | None:
    periodes: list[str] = []
    for unite in unites:
        if unite.get("entite") != nom:
            continue
        bloc = unite.get("periode")
        if isinstance(bloc, list):
            for item in bloc:
                if isinstance(item, str) and item:
                    periodes.append(item)
        elif isinstance(bloc, str) and bloc:
            periodes.append(bloc)
    if not periodes:
        return None
    periodes.sort()
    return [periodes[0], periodes[-1]]


def _compter_faits_par_entite(entites_dir: Path) -> dict[str, int]:
    comptes: dict[str, int] = {}
    if not entites_dir.is_dir():
        return comptes
    for path in sorted(entites_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entite = payload.get("entite") or {}
        nom = entite.get("nom") or path.stem
        nb = len(payload.get("faits") or [])
        comptes[nom] = comptes.get(nom, 0) + nb
    return comptes


def _proposer_entites(
    entites_registre: Mapping[str, Any],
    entites_dir: Path,
) -> list[dict[str, Any]]:
    unites = [
        u for u in (entites_registre.get("unites") or []) if isinstance(u, dict)
    ]
    faits_par_nom = _compter_faits_par_entite(entites_dir)
    propositions: list[dict[str, Any]] = []
    for entite in entites_registre.get("entites") or []:
        if not isinstance(entite, dict) or not entite.get("nom"):
            continue
        nom = entite["nom"]
        propositions.append(
            {
                "nom": nom,
                "type": entite.get("type") or "autre",
                "cles": list(entite.get("cles") or []),
                "variantes": list(entite.get("variantes") or []),
                "periode": _periode_entite(nom, unites),
                "unites": sum(1 for u in unites if u.get("entite") == nom),
                "faits": faits_par_nom.get(nom, 0),
                "source": "proposee",
            }
        )
    propositions.sort(key=lambda e: e["nom"].lower())
    return propositions


PLAFOND_QUESTIONS = 40
CONCEPTS_QUESTIONS_INDIVIDUELLES = 15
SEUIL_SIMILARITE_VERSION = 0.60


def _normaliser_ligne(ligne: str) -> str:
    return re.sub(r"\s+", " ", (ligne or "").lower()).strip()


def _entree_piece_lue(entree: Mapping[str, Any]) -> bool:
    if entree.get("traitement") == "registre_seul":
        return False
    piece = entree.get("piece")
    if not isinstance(piece, str) or not piece:
        return False
    lecture = entree.get("lecture") or {}
    return isinstance(lecture, dict) and "erreur" not in lecture


def _chemin_piece_cache(
    drive_dir: Path | None,
    entree: Mapping[str, Any],
) -> Path | None:
    if drive_dir is None:
        return None
    piece = entree.get("piece")
    groupe = entree.get("groupe")
    if not isinstance(piece, str) or not piece:
        return None
    if not isinstance(groupe, str) or not groupe:
        from banc_realite.lecture_drive import _groupe_traitement

        groupe = _groupe_traitement(entree.get("traitement") or "") or ""
    if not groupe:
        return None
    return drive_dir / "cache-lecture" / "pieces" / groupe / piece


def _lignes_non_vides_piece(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    lignes: list[str] = []
    for brute in path.read_text(encoding="utf-8").splitlines():
        norm = _normaliser_ligne(brute)
        if norm:
            lignes.append(norm)
    return lignes


def _similarite_lignes(
    lignes_a: list[str],
    lignes_b: list[str],
) -> tuple[float, str | None]:
    if not lignes_a or not lignes_b:
        return 0.0, None
    compteur_a = Counter(lignes_a)
    compteur_b = Counter(lignes_b)
    commun = sum(
        min(compteur_a[ligne], compteur_b[ligne])
        for ligne in compteur_a
        if ligne in compteur_b
    )
    denom = min(len(lignes_a), len(lignes_b))
    if denom <= 0:
        return 0.0, None
    exemple: str | None = None
    for ligne in sorted(compteur_a.keys() & compteur_b.keys()):
        if min(compteur_a[ligne], compteur_b[ligne]) > 0:
            exemple = ligne
            break
    return commun / denom, exemple


def _cle_affichage_partagee(
    ea: Mapping[str, Any],
    eb: Mapping[str, Any],
    cles_partagees_norm: set[str],
) -> str:
    for entite in (ea, eb):
        for cle in (entite.get("cles") or []) + (entite.get("variantes") or []):
            if isinstance(cle, str) and cle.strip() and normalize_text(cle) in cles_partagees_norm:
                return cle.strip()
    return sorted(cles_partagees_norm)[0] if cles_partagees_norm else ""


def proposer_questions(
    drive_registre: list[dict[str, Any]],
    entites_registre: Mapping[str, Any],
    familles: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    types_entites: list[dict[str, Any]],
    *,
    drive_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions_brutes: list[dict[str, Any]] = []
    vus: set[tuple[str, str]] = set()

    def _ajouter(
        sujet: str,
        question: str,
        candidats: list[str],
        preuve: list[dict[str, Any]],
        pourquoi: str,
        **extra: Any,
    ) -> None:
        cle = (sujet, question)
        if cle in vus:
            return
        vus.add(cle)
        bloc: dict[str, Any] = {
            "sujet": sujet,
            "question": question,
            "candidats": candidats,
            "preuve": preuve,
            "pourquoi": pourquoi,
            "reponse": None,
        }
        bloc.update(extra)
        questions_brutes.append(bloc)

    entites = [
        e for e in (entites_registre.get("entites") or []) if isinstance(e, dict)
    ]
    entites_tri = sorted(entites, key=lambda e: (e.get("nom") or "").lower())
    for i, ea in enumerate(entites_tri):
        nom_a = ea.get("nom") or ""
        norm_nom_a = normalize_text(nom_a)
        cles_a = _cles_normalisees_entite(ea)
        for eb in entites_tri[i + 1 :]:
            nom_b = eb.get("nom") or ""
            norm_nom_b = normalize_text(nom_b)
            cles_b = _cles_normalisees_entite(eb)
            cle_partagee = cles_a & cles_b
            inclusion = (
                norm_nom_a
                and norm_nom_b
                and (norm_nom_a in norm_nom_b or norm_nom_b in norm_nom_a)
                and norm_nom_a != norm_nom_b
            )
            if not cle_partagee and not inclusion:
                continue
            preuve: list[dict[str, Any]] = [{"piece": "", "lignes": []}]
            if cle_partagee:
                label = _cle_affichage_partagee(ea, eb, cle_partagee)
                pourquoi = f'même clé "{label}" dans les deux'
            else:
                if norm_nom_a in norm_nom_b:
                    pourquoi = f'le nom « {nom_a} » est contenu dans « {nom_b} »'
                else:
                    pourquoi = f'le nom « {nom_b} » est contenu dans « {nom_a} »'
            _ajouter(
                "entites",
                f"Même entité ? ({nom_a} / {nom_b})",
                [nom_a, nom_b],
                preuve,
                pourquoi,
            )

    par_famille: dict[str, list[dict[str, Any]]] = {}
    for entree in drive_registre:
        if entree.get("doublon_de"):
            continue
        famille = entree.get("famille") or ""
        if not famille or entree.get("traitement") == "a_trancher":
            continue
        if not _entree_piece_lue(entree):
            continue
        par_famille.setdefault(famille, []).append(entree)

    for famille in sorted(par_famille):
        membres = sorted(
            par_famille[famille],
            key=lambda e: (_piece_de_entree(e).lower(), e.get("id") or ""),
        )
        for i, ea in enumerate(membres):
            piece_a = _piece_de_entree(ea)
            path_a = _chemin_piece_cache(drive_dir, ea)
            lignes_a = _lignes_non_vides_piece(path_a)
            for eb in membres[i + 1 :]:
                piece_b = _piece_de_entree(eb)
                path_b = _chemin_piece_cache(drive_dir, eb)
                lignes_b = _lignes_non_vides_piece(path_b)
                similarite, exemple = _similarite_lignes(lignes_a, lignes_b)
                if similarite < SEUIL_SIMILARITE_VERSION:
                    continue
                pct = int(round(similarite * 100))
                preuve_version = [
                    {"piece": piece_a, "lignes": [1]},
                    {"piece": piece_b, "lignes": [1]},
                ]
                if exemple:
                    preuve_version.append({"ligne_partagee": exemple})
                _ajouter(
                    "pieces",
                    f"Version de la même pièce ? ({piece_a} / {piece_b})",
                    [piece_a, piece_b],
                    preuve_version,
                    f"{pct} % de lignes identiques",
                    similarite=round(similarite, 4),
                )

    famille_par_piece = _famille_par_piece(drive_registre)
    concepts_eligibles = [
        c
        for c in concepts
        if len(c.get("pieces") or []) >= 3
    ]
    concepts_eligibles.sort(
        key=lambda c: (
            -len(c.get("pieces") or []),
            -int(c.get("groupes") or 0),
            c.get("nom") or "",
        )
    )
    for concept in concepts_eligibles[:CONCEPTS_QUESTIONS_INDIVIDUELLES]:
        pieces = concept.get("pieces") or []
        nb_familles = len(
            {famille_par_piece.get(p) for p in pieces if famille_par_piece.get(p)}
        )
        _ajouter(
            "concepts",
            f"Grandeur à suivre ? ({concept['nom']})",
            [concept.get("concept") or "autre"],
            [{"piece": pieces[0], "lignes": [1]}],
            f"présente dans {len(pieces)} pièces de {nb_familles} familles",
        )
    restants = concepts_eligibles[CONCEPTS_QUESTIONS_INDIVIDUELLES:]
    if restants:
        candidats_reliquat = sorted(c["nom"] for c in restants)
        _ajouter(
            "concepts",
            "D'autres grandeurs reviennent dans ≥ 3 pièces : lesquelles suivre ?",
            candidats_reliquat,
            [{"piece": (restants[0].get("pieces") or [""])[0], "lignes": [1]}],
            f"{len(candidats_reliquat)} grandeurs au-delà des "
            f"{CONCEPTS_QUESTIONS_INDIVIDUELLES} premières",
        )

    for typ in sorted(types_entites, key=lambda t: t["nom"]):
        if typ.get("entites") != 1:
            continue
        _ajouter(
            "types",
            f"Type utile ou cas isolé ? ({typ['nom']})",
            [typ["nom"]],
            [{"piece": "", "lignes": []}],
            f"une seule entité porte le type « {typ['nom']} »",
        )

    noms_familles = sorted(
        {f["nom"] for f in familles if f.get("nom") and f["nom"] != "a_trancher"}
    )
    for entree in sorted(
        drive_registre,
        key=lambda e: (_piece_de_entree(e).lower(), e.get("id") or ""),
    ):
        if entree.get("traitement") != "a_trancher" or entree.get("doublon_de"):
            continue
        piece = _piece_de_entree(entree)
        _ajouter(
            "familles",
            f"Quelle famille ? ({piece})",
            noms_familles,
            [{"piece": piece, "lignes": [1]}],
            "pièce en attente de classement (traitement a_trancher)",
        )

    for index, q in enumerate(questions_brutes, start=1):
        q["id"] = f"Q-{index:04d}"
    actives = questions_brutes[:PLAFOND_QUESTIONS]
    reserve = questions_brutes[PLAFOND_QUESTIONS:]
    return actives, reserve


def proposer_ontologie(
    drive_registre: list[dict[str, Any]],
    aiguillage: Mapping[str, Any],
    entites_registre: Mapping[str, Any],
    groupes: Mapping[str, Any],
    entites_dir: Path,
    *,
    dossier: str,
    genere_le: str,
    sources: Mapping[str, str] | None = None,
    drive_dir: Path | None = None,
) -> dict[str, Any]:
    regles = aiguillage.get("regles") or []
    familles = proposer_familles(drive_registre, regles)
    types_entites = proposer_types(entites_registre)
    entites = _proposer_entites(entites_registre, entites_dir)
    famille_par_piece = _famille_par_piece(drive_registre)
    relations = proposer_relations(entites_registre, entites_dir, famille_par_piece)
    concepts = proposer_concepts(groupes)
    questions, questions_en_reserve = proposer_questions(
        drive_registre,
        entites_registre,
        familles,
        concepts,
        types_entites,
        drive_dir=drive_dir,
    )
    return {
        "schema": SCHEMA,
        "dossier": dossier,
        "version": 0,
        "genere_le": genere_le,
        "sources": dict(sources or {}),
        "familles": familles,
        "types_entites": types_entites,
        "entites": entites,
        "relations": relations,
        "concepts": concepts,
        "questions": questions,
        "questions_en_reserve": questions_en_reserve,
        "decisions": [],
    }


def charger_entrees_drive(registre_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(registre_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("entrees") or payload.get("fichiers") or []


def charger_aiguillage(chemin: Path) -> dict[str, Any]:
    return json.loads(chemin.read_text(encoding="utf-8"))


def charger_registre_entites(chemin: Path) -> dict[str, Any]:
    return json.loads(chemin.read_text(encoding="utf-8"))


def charger_groupes(chemin: Path) -> dict[str, Any]:
    return json.loads(chemin.read_text(encoding="utf-8"))
