# SPDX-License-Identifier: AGPL-3.0-only
"""Reconstitution par entité — recensement, rassemblement, reconstitution."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import statistics
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .extraction_situations import _objet_json

ChatFn = Callable[[str, str], tuple[str, Any]]

ENTETE_DOC = 8
MAX_PAR_PIECE = 40
MAX_PAR_PIECE_PLAGE = 60
MAX_DOC = 220
MAX_EMERGENT_PAR_PIECE = 12
MAX_PIECES_VALEUR = 6
OCTETS_MAX_INVITE = 120_000
OCTETS_MAX_LOTS = 100_000
SEUIL_PLAGE = 12_000
MOTIFS_VARIANTE = ("declaree_par_le_modele", "cle_partagee", "autre_graphie")
CONCEPTS_RELIQUAT = (
    "mesure",
    "statut",
    "montant",
    "date",
    "obligation",
    "identite",
    "regle",
    "autre",
)

TERME = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
VALEUR = re.compile(r"(?<![\w.])\d{1,3}(?:[ ,]\d{3})*\.\d{2,3}(?![\w.])|\b[A-Z]{2,}(?:-[A-Z0-9]+){2,}\b")
DATE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])\b")
ID_PATTERN = re.compile(r"[A-Z]{2,}(?:-[A-Z0-9]+)+")
CLE_BEGAYEE = re.compile(r'"([^"]+)","\1":')
SEP_TABLEAU = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
MOT_TITRE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")

TITRES = {
    "document": "LA PIÈCE QUI EST LE DOCUMENT DE L'ENTITÉ, EN ENTIER",
    "entite": "LIGNES QUI PORTENT L'ENTITÉ",
    "variantes": "LIGNES DES VARIANTES DE NOM, TOUTES PÉRIODES",
    "emergent": "LIGNES TROUVÉES PAR CE QUI ÉMERGE : TERMES DÉFINIS, VALEURS DE L'ENTITÉ RETROUVÉES AILLEURS",
}

INVITE_RECENSEMENT = """Voici l'inventaire d'un dossier d'entreprise : ses pièces, les familles d'identifiants qu'on y
trouve, et les noms relevés lors d'une première lecture.

Dresse le registre des entités du dossier : ce qui existe dans l'entreprise et dont les pièces
parlent. Une entité est un objet durable ou un document : un site, un client, un fournisseur,
un contrat, une facture, un certificat, un compte. Un enregistrement unitaire (une visite, un
ticket, une tournée, une ligne de relevé) n'est pas une entité : il appartient à une entité.

Rends un seul objet JSON, sans texte autour :
{"entites": [
  {"nom": "nom canonique",
   "type": "site | client | fournisseur | contrat | facture | certificat | compte | personne | systeme | autre",
   "cles": ["chaînes exactes, telles qu'écrites dans les pièces, qui désignent l'entité : noms et identifiants"],
   "variantes": ["parmi les clés, les graphies dont l'identité avec l'entité n'est pas certaine"],
   "par_periode": true
  }
],
 "familles": [
  {"famille": "telle que donnée", "role": "entite | enregistrement",
   "gabarit_cle": "pour une famille d'entités nombreuses : null, et chaque identifiant devient une entité de ce type",
   "type": "le type des entités de cette famille"}
 ]}

Règles :
- "par_periode" vaut true quand l'entité est suivie dans le temps par de nombreux
  enregistrements (un site et ses collectes), false pour un document ou une personne.
- Pour une famille d'entités nombreuses (des factures), ne liste pas chaque identifiant dans
  "entites" : déclare la famille avec le rôle "entite".
- N'écris que des chaînes présentes dans l'inventaire. N'invente aucun identifiant."""

INVITE = """On a rassemblé, dans toutes les pièces d'un dossier d'entreprise, les lignes qui parlent
d'une même entité. Chaque ligne porte sa pièce et son numéro.

Reconstitue ce que le dossier établit sur cette entité. Tu ne tranches pas : tu rapportes ce
que chaque pièce dit, et tu compares.

Rends un seul objet JSON, sans texte autour :
{
 "entite": {"nom": "...", "type": "site | facture | certificat | contrat | compte | autre",
            "identifiants": ["..."]},
 "identites": [
   {"variante": "le nom tel qu'il est écrit",
    "jugement": "meme | non_resolue | distincte",
    "motif": "ce qui relie ou non ces lignes à l'entité : code, adresse, ticket, ou leur absence",
    "piece": "...", "lignes": [0]}
 ],
 "faits": [
   {"quoi": "ce qui est mesuré ou affirmé, en une phrase courte",
    "concept": "mesure | statut | montant | date | obligation | identite | autre",
    "versions": [{"piece": "...", "valeur": "recopiée caractère pour caractère",
                  "statut": "pese | estime | provisoire | corrige | declare | paye | du | non_precise",
                  "a_la_date": "date à laquelle la pièce l'affirme, ou null si la pièce ne la donne pas",
                  "lignes": [0]}],
    "comparaison": "concordant | desaccord | non_ordonnable | complementaire | une_seule_source",
    "explication": "une phrase"}
 ],
 "regles": [{"enonce": "en une phrase", "s_applique_a": "quel fait ou quel statut",
             "piece": "...", "lignes": [0]}],
 "non_etabli": [{"point": "ce que le dossier ne permet pas de dire sur cette entité",
                 "ce_qui_le_resoudrait": "..."}]
}

Règles :
- Une version par pièce et par fait. Si deux pièces donnent la même valeur, deux versions.
- "lignes" est une LISTE de numéros de ligne montrés ci-dessus, jamais un intervalle. Ne cite
  aucune ligne qui n'est pas montrée. Nom de fichier exact.
- Identité : juge chaque variante de nom occurrence par occurrence. Une variante peut être la
  même entité sur certaines lignes (code, adresse ou ticket communs) et non résolue sur
  d'autres (code absent, rien qui la relie). Fais alors deux entrées.
- États : compare aussi les états et leurs dates. Un même montant peut cacher un désaccord
  d'état : payé dans une pièce, encore dû dans une autre. Si les deux états ne portent pas de
  date qui permette de les ordonner, la comparaison est "non_ordonnable".
- Un désaccord se constate sur des valeurs, des statuts ou des états différents pour le même fait.
- Recopie nombres, dates et identifiants dans la forme exacte de la pièce.
- N'invente rien. Avant d'écrire qu'une chose n'est pas établie, vérifie qu'aucune ligne
  montrée ne la donne."""

INVITE_RELIQUAT = """Voici les lignes d'un dossier d'entreprise qui ne nomment aucune entité particulière : des
totaux, des tableaux de synthèse, des énoncés sur l'ensemble du compte. Chaque ligne porte sa
pièce et son numéro.

Regroupe les lignes qui parlent de la même grandeur ou du même fait, même quand les pièces ne
donnent pas la même valeur : c'est justement ce qu'on veut pouvoir comparer.

Rends un seul objet JSON, sans texte autour :
{"groupes": [
  {"titre": "la grandeur ou le fait, en quelques mots",
   "concept": "mesure | statut | montant | date | obligation | identite | regle | autre",
   "periode": "la période concernée telle qu'écrite, ou null",
   "lignes": [{"piece": "nom de fichier exact", "lignes": [0]}]}
]}

Règles :
- Un groupe réunit les versions d'une même grandeur dans toutes les pièces où elle apparaît.
- Une ligne peut appartenir à plusieurs groupes.
- Un tableau de synthèse par mois se range ligne par ligne : la ligne d'un mois rejoint le
  groupe de cette grandeur pour ce mois, et la ligne de total rejoint le groupe du total.
- Une règle ou une définition forme un groupe de concept "regle".
- N'utilise que des numéros de ligne montrés. "lignes" est une liste, jamais un intervalle."""


def _famille_identifiant(ident: str) -> str:
    reduit = re.sub(r"\d", "#", ident)
    segments = reduit.split("-")
    return "-".join(segments[:2])


def _slug(texte: str) -> str:
    base = texte.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base)
    return base.strip("-")


def _occurrences_cles_masquees(ligne: str, cles: list[str]) -> list[tuple[int, int, str]]:
    """Occurrences de clés dans une ligne, des plus longues aux plus courtes ; masquage inclusif."""
    ligne_l = ligne.lower()
    retenues: list[tuple[int, int, str]] = []
    vues: set[str] = set()
    uniques: list[str] = []
    for cle in cles:
        if not isinstance(cle, str) or not cle.strip():
            continue
        k = cle.lower()
        if k in vues:
            continue
        vues.add(k)
        uniques.append(cle)
    for cle in sorted(uniques, key=len, reverse=True):
        cle_l = cle.lower()
        pos = 0
        while pos <= len(ligne_l) - len(cle_l):
            idx = ligne_l.find(cle_l, pos)
            if idx == -1:
                break
            fin = idx + len(cle_l)
            avant_ok = idx == 0 or not ligne_l[idx - 1].isalnum()
            apres_ok = fin >= len(ligne_l) or not ligne_l[fin].isalnum()
            if avant_ok and apres_ok and not any(
                rs <= idx and fin <= re for rs, re, _ in retenues
            ):
                retenues.append((idx, fin, cle))
            pos = idx + 1
    return retenues


def ligne_porte_cle(ligne: str, cle: str, toutes_cles: list[str]) -> bool:
    """Vrai si la ligne porte cette clé après masquage des occurrences plus longues."""
    cle_l = cle.lower()
    return any(c.lower() == cle_l for _, _, c in _occurrences_cles_masquees(ligne, toutes_cles))


def index_cles_portees(
    corpus: Mapping[str, list[str]],
    toutes_cles: list[str],
) -> dict[tuple[str, int], frozenset[str]]:
    """Index (pièce, ligne) → clés portées après masquage, calculé une fois pour le corpus."""
    index: dict[tuple[str, int], frozenset[str]] = {}
    for piece, lignes in corpus.items():
        for i, ligne in enumerate(lignes):
            portees = frozenset(
                c.lower() for _, _, c in _occurrences_cles_masquees(ligne, toutes_cles)
            )
            if portees:
                index[(piece, i)] = portees
    return index


def _portees_index(
    index_cles: Mapping[tuple[str, int], frozenset[str]] | None,
    piece: str,
    i: int,
) -> frozenset[str] | None:
    if index_cles is None:
        return None
    return index_cles.get((piece, i), frozenset())


def _porte_cles(
    ligne: str,
    cles: list[str],
    toutes_cles: list[str],
    portees: frozenset[str] | None,
) -> bool:
    if not cles:
        return False
    if portees is not None:
        return any(c.lower() in portees for c in cles)
    return any(ligne_porte_cle(ligne, c, toutes_cles) for c in cles)


def _toutes_cles_entites(entites: list[Mapping[str, Any]]) -> list[str]:
    vues: dict[str, str] = {}
    for entite in entites:
        for cle in entite.get("cles") or []:
            if isinstance(cle, str) and cle.strip():
                k = cle.lower()
                if k not in vues or len(cle) > len(vues[k]):
                    vues[k] = cle
        for item in entite.get("variantes") or []:
            cle: str | None
            if isinstance(item, Mapping):
                raw = item.get("cle")
                cle = raw if isinstance(raw, str) else None
            elif isinstance(item, str):
                cle = item
            else:
                cle = None
            if cle and cle.strip():
                k = cle.lower()
                if k not in vues or len(cle) > len(vues[k]):
                    vues[k] = cle
    return list(vues.values())


def _cles_entite_sans_partagees(entite: Mapping[str, Any]) -> list[str]:
    partagees = {
        item["cle"]
        for item in entite.get("variantes") or []
        if isinstance(item, Mapping) and item.get("motif") == "cle_partagee"
    }
    return [c for c in entite.get("cles") or [] if c not in partagees]


def _cle_dans_corpus(
    cle: str,
    corpus: Mapping[str, list[str]],
    toutes_cles: list[str],
) -> bool:
    for lignes in corpus.values():
        for ligne in lignes:
            if ligne_porte_cle(ligne, cle, toutes_cles):
                return True
    return False


def _identifiants_corpus(corpus: Mapping[str, list[str]]) -> set[str]:
    trouves: set[str] = set()
    for lignes in corpus.values():
        trouves.update(ID_PATTERN.findall("\n".join(lignes)))
    return trouves


def charger_corpus(pieces_dir: Path) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    """Charge les pièces uniques ; exclut les doublons octet pour octet."""
    contenus: dict[bytes, str] = {}
    exclus: list[dict[str, str]] = []
    corpus: dict[str, list[str]] = {}
    for path in sorted(pieces_dir.iterdir()):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if raw in contenus:
            exclus.append({"piece": path.name, "identique_a": contenus[raw]})
            continue
        contenus[raw] = path.name
        corpus[path.name] = path.read_text(encoding="utf-8").splitlines()
    return corpus, exclus


def _entetes_pieces(corpus: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    pieces: list[dict[str, Any]] = []
    for nom in sorted(corpus):
        non_vides = [ligne for ligne in corpus[nom] if ligne.strip()]
        pieces.append({"nom": nom, "entete": non_vides[:9]})
    return pieces


def _familles_identifiants(corpus: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for nom, lignes in corpus.items():
        texte = "\n".join(lignes)
        for ident in set(ID_PATTERN.findall(texte)):
            fam = _famille_identifiant(ident)
            entree = stats.setdefault(
                fam,
                {"famille": fam, "effectif": 0, "exemples": [], "pieces": set()},
            )
            entree["effectif"] += 1
            entree["pieces"].add(nom)
            if len(entree["exemples"]) < 3 and ident not in entree["exemples"]:
                entree["exemples"].append(ident)
    return [
        {
            "famille": item["famille"],
            "effectif": item["effectif"],
            "exemples": item["exemples"],
            "pieces": len(item["pieces"]),
        }
        for item in sorted(stats.values(), key=lambda x: (-x["effectif"], x["famille"]))
    ]


def _acteurs_situations(situations_dir: Path | None) -> dict[str, list[dict[str, Any]]]:
    if situations_dir is None or not situations_dir.is_dir():
        return {}
    comptes: dict[str, dict[str, int]] = {}
    for path in sorted(situations_dir.glob("*.situations.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for situation in payload.get("situations") or []:
            for acteur in situation.get("acteurs") or []:
                nom = acteur.get("nom")
                if not isinstance(nom, str) or not nom.strip():
                    continue
                typ = acteur.get("type_entite") or "autre"
                comptes.setdefault(typ, {})
                comptes[typ][nom] = comptes[typ].get(nom, 0) + 1
    return {
        typ: [{"nom": nom, "effectif": effectif} for nom, effectif in sorted(valeurs.items())]
        for typ, valeurs in sorted(comptes.items())
    }


def construire_inventaire(
    *,
    pieces_dir: Path,
    situations_dir: Path | None = None,
) -> dict[str, Any]:
    corpus, exclus = charger_corpus(pieces_dir)
    return {
        "pieces": _entetes_pieces(corpus),
        "familles": _familles_identifiants(corpus),
        "acteurs": _acteurs_situations(situations_dir),
        "pieces_exclues": exclus,
        "corpus": corpus,
    }


def _serialiser_inventaire(inventaire: Mapping[str, Any]) -> str:
    payload = {
        "pieces": inventaire["pieces"],
        "familles": inventaire["familles"],
        "acteurs": inventaire.get("acteurs") or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


def _valider_entite(
    entite: Mapping[str, Any],
    corpus: Mapping[str, list[str]],
    ecartes: dict[str, list[str]],
    toutes_cles: list[str],
) -> dict[str, Any] | None:
    nom = entite.get("nom")
    if not isinstance(nom, str) or not nom.strip():
        ecartes.setdefault("entites", []).append("entité sans nom")
        return None
    cles_brutes = entite.get("cles") or []
    if not isinstance(cles_brutes, list):
        ecartes.setdefault("cles", []).append(f"{nom} : cles invalides")
        return None
    cles_valides: list[str] = []
    variantes_brutes = [
        v for v in (entite.get("variantes") or []) if isinstance(v, str) and v.strip()
    ]
    for cle in cles_brutes:
        if not isinstance(cle, str) or not cle.strip():
            continue
        if _cle_dans_corpus(cle, corpus, toutes_cles):
            cles_valides.append(cle)
        else:
            ecartes.setdefault("cles", []).append(f"{nom} : clé absente du corpus « {cle} »")
    if not cles_valides:
        ecartes.setdefault("entites", []).append(f"{nom} : aucune clé valide")
        return None
    variantes = [v for v in variantes_brutes if _cle_dans_corpus(v, corpus, toutes_cles)]
    return {
        "nom": nom.strip(),
        "type": entite.get("type") or "autre",
        "cles": cles_valides,
        "variantes": variantes,
        "par_periode": bool(entite.get("par_periode")),
    }


def _expandir_familles(
    familles: list[Any],
    corpus: Mapping[str, list[str]],
    entites: list[dict[str, Any]],
) -> None:
    couvertes = {cle.lower() for ent in entites for cle in ent["cles"]}
    for item in familles:
        if not isinstance(item, dict):
            continue
        if item.get("role") != "entite":
            continue
        fam = item.get("famille")
        typ = item.get("type") or "autre"
        if not isinstance(fam, str):
            continue
        for ident in sorted(_identifiants_corpus(corpus)):
            if _famille_identifiant(ident) != fam:
                continue
            if ident.lower() in couvertes:
                continue
            entites.append(
                {
                    "nom": ident,
                    "type": typ,
                    "cles": [ident],
                    "variantes": [],
                    "par_periode": False,
                }
            )
            couvertes.add(ident.lower())


def _cles_variantes(variantes: Any) -> list[str]:
    cles: list[str] = []
    for item in variantes or []:
        if isinstance(item, Mapping):
            cle = item.get("cle")
            if isinstance(cle, str) and cle.strip():
                cles.append(cle)
        elif isinstance(item, str) and item.strip():
            cles.append(item)
    return cles


def _est_autre_graphie(cle: str, nom: str) -> bool:
    if cle == nom:
        return False
    if ID_PATTERN.fullmatch(cle):
        return False
    if any(ch.isdigit() for ch in cle) or "@" in cle:
        return False
    return True


def _attribuer_variantes(entites: list[dict[str, Any]]) -> None:
    porteurs: dict[str, int] = {}
    for entite in entites:
        for cle in dict.fromkeys(entite["cles"]):
            porteurs[cle] = porteurs.get(cle, 0) + 1
    for entite in entites:
        vues: set[str] = set()
        result: list[dict[str, str]] = []

        def ajouter(cle: str, motif: str) -> None:
            if cle in vues:
                return
            vues.add(cle)
            result.append({"cle": cle, "motif": motif})

        for cle in dict.fromkeys(entite["cles"]):
            if porteurs.get(cle, 0) >= 2:
                ajouter(cle, "cle_partagee")
            elif _est_autre_graphie(cle, str(entite["nom"])):
                ajouter(cle, "autre_graphie")
        for cle in _cles_variantes(entite.get("variantes")):
            ajouter(cle, "declaree_par_le_modele")
        entite["variantes"] = result


def _titre_unite(entite: Mapping[str, Any], periodes: list[str]) -> str:
    if not periodes:
        return str(entite["nom"])
    if len(periodes) == 1:
        return f"{entite['nom']}, mois de service {periodes[0]}"
    return f"{entite['nom']}, mois de service {periodes[0]} à {periodes[-1]}"


def _identifiant_unite(entite: Mapping[str, Any], periodes: list[str]) -> str:
    typ = _slug(str(entite.get("type") or "autre"))
    id_cle = next((c for c in entite.get("cles") or [] if ID_PATTERN.fullmatch(c)), None)
    if id_cle:
        base = f"{typ}-{_slug(id_cle)}"
    else:
        base = f"{typ}-{_slug(str(entite['nom']))}"
    if periodes:
        suffix = f"{periodes[0]}_{periodes[-1]}"
        if not base.endswith(suffix):
            return f"{base}-{suffix}"
    return base


def _est_document_entite(
    lignes: list[str],
    cles: list[str],
    periode: list[str],
    toutes_cles: list[str],
) -> bool:
    if periode:
        return False
    tete = "\n".join(lignes[:ENTETE_DOC])
    return any(ligne_porte_cle(tete, c, toutes_cles) for c in cles)


def _prises_entite(
    lignes: list[str],
    piece: str,
    cles: list[str],
    periode: list[str],
    toutes_cles: list[str],
    index_cles: Mapping[tuple[str, int], frozenset[str]] | None = None,
) -> dict[int, str]:
    piece_ok = any(p in "\n".join(lignes[:12]) for p in periode) and not any(
        DATE.search(l) for l in lignes[12:40]
    )
    prises: dict[int, str] = {}
    for i, ligne in enumerate(lignes):
        if _porte_cles(
            ligne, cles, toutes_cles, _portees_index(index_cles, piece, i)
        ) and dans_periode(ligne, periode, piece_ok):
            ajouter(prises, lignes, i, piece, "porte l'entité")
    return prises


def _volume_entite(
    cles: list[str],
    periode: list[str],
    corpus: Mapping[str, list[str]],
    toutes_cles: list[str],
    index_cles: Mapping[tuple[str, int], frozenset[str]] | None = None,
) -> int:
    total = 0
    for piece, lignes in corpus.items():
        if _est_document_entite(lignes, cles, periode, toutes_cles):
            continue
        prises = _prises_entite(
            lignes, piece, cles, periode, toutes_cles, index_cles=index_cles
        )
        total += sum(len(lignes[i]) for i in prises)
    return total


def _plages_mois(
    mois: list[str],
    cles: list[str],
    corpus: Mapping[str, list[str]],
    toutes_cles: list[str],
    index_cles: Mapping[tuple[str, int], frozenset[str]] | None = None,
) -> list[list[str]]:
    plages: list[list[str]] = []
    courante: list[str] = []
    for m in mois:
        volume_avec = _volume_entite(
            cles, courante + [m], corpus, toutes_cles, index_cles=index_cles
        )
        if courante and volume_avec >= SEUIL_PLAGE:
            plages.append(courante)
            courante = [m]
        elif not courante and volume_avec >= SEUIL_PLAGE:
            plages.append([m])
            courante = []
        else:
            courante = courante + [m]
    if courante:
        plages.append(courante)
    return plages


def _decouper_unites(
    entites: list[dict[str, Any]],
    corpus: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    unites: list[dict[str, Any]] = []
    toutes_cles = _toutes_cles_entites(entites)
    index_cles = index_cles_portees(corpus, toutes_cles)
    for entite in entites:
        if entite.get("par_periode"):
            mois: set[str] = set()
            cles = _cles_entite_sans_partagees(entite)
            for piece, lignes in corpus.items():
                for i, ligne in enumerate(lignes):
                    if _porte_cles(
                        ligne,
                        cles,
                        toutes_cles,
                        _portees_index(index_cles, piece, i),
                    ):
                        mois.update(f"{a}-{m}" for a, m in DATE.findall(ligne))
            plages = (
                _plages_mois(
                    sorted(mois), cles, corpus, toutes_cles, index_cles=index_cles
                )
                if mois
                else [[]]
            )
        else:
            plages = [[]]
        for periode_liste in plages:
            ident = _identifiant_unite(entite, periode_liste)
            unites.append(
                {
                    "id": ident,
                    "entite": entite["nom"],
                    "type": entite["type"],
                    "titre": _titre_unite(entite, periode_liste),
                    "cles": list(entite["cles"]),
                    "variantes": list(entite.get("variantes") or []),
                    "periode": list(periode_liste),
                }
            )
    return unites


def valider_registre(
    payload: Mapping[str, Any],
    corpus: Mapping[str, list[str]],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    ecartes: dict[str, list[str]] = {}
    entites: list[dict[str, Any]] = []
    brutes = [e for e in payload.get("entites") or [] if isinstance(e, Mapping)]
    toutes_cles_brutes = [
        c
        for ent in brutes
        for c in list(ent.get("cles") or []) + list(ent.get("variantes") or [])
        if isinstance(c, str) and c.strip()
    ]
    for entite in brutes:
        valide = _valider_entite(entite, corpus, ecartes, toutes_cles_brutes)
        if valide is not None:
            entites.append(valide)
    familles = payload.get("familles") or []
    _expandir_familles(familles, corpus, entites)
    _attribuer_variantes(entites)
    unites = _decouper_unites(entites, corpus)
    return {"entites": entites, "unites": unites, "familles": familles}, ecartes


def _brut_recensement_lisible(texte: str) -> bool:
    try:
        _objet_json(texte, "entites")
        return True
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def recensement_brut_reutilisable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return _brut_recensement_lisible(path.read_text(encoding="utf-8"))
    except OSError:
        return False


def recenser(
    *,
    pieces_dir: Path,
    sortie_dir: Path,
    dossier: str,
    situations_dir: Path | None = None,
    chat_fn: ChatFn | None = None,
    compteur=None,
    delai_s: int = 900,
    refaire: bool = False,
) -> dict[str, Any]:
    inventaire = construire_inventaire(pieces_dir=pieces_dir, situations_dir=situations_dir)
    corpus = inventaire.pop("corpus")
    invite = _serialiser_inventaire(inventaire) + "\n\n" + INVITE_RECENSEMENT
    brut_path = sortie_dir / "recensement.brut.txt"
    brut_path.parent.mkdir(parents=True, exist_ok=True)
    brut: str | None = None
    appel = False
    if not refaire and recensement_brut_reutilisable(brut_path):
        brut = brut_path.read_text(encoding="utf-8")
    if brut is None:
        appel = True
        if brut_path.is_file():
            _ecarter_brut(brut_path, sortie_dir / "ecartes")
        if chat_fn is not None:
            if compteur is not None:
                compteur.reserver_emission()
            chat = chat_fn
        else:
            from .situations_composites import _chat_par_defaut

            chat = _chat_par_defaut(
                "recensement-entites",
                compteur=compteur,
                dossier=dossier,
                delai_s=delai_s,
            )
        brut, _meta = chat("", invite)
        brut_path.write_text(brut, encoding="utf-8")
    payload = _objet_json(brut, "entites")
    registre, ecartes = valider_registre(payload, corpus)
    sortie = {
        **registre,
        "ecartes": ecartes,
        "pieces_exclues": inventaire.get("pieces_exclues") or [],
    }
    (sortie_dir / "registre.json").write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**sortie, "appel": appel}


def est_tableau(ligne: str, piece: str) -> bool:
    return piece.endswith(".csv") or ligne.lstrip().startswith("|")


def est_separateur_tableau(ligne: str) -> bool:
    return bool(SEP_TABLEAU.match(ligne))


def entete_avant(lignes: list[str], index: int, piece: str) -> int | None:
    if piece.endswith(".csv"):
        return 0
    for i in range(index, -1, -1):
        if lignes[i].lstrip().startswith("|") and i + 1 < len(lignes) and re.match(
            r"^\s*\|?[\s:\-|]+\|?\s*$", lignes[i + 1]
        ):
            return i
        if not lignes[i].lstrip().startswith("|"):
            return None
    return None


def dans_periode(ligne: str, periode: list[str], piece_ok: bool) -> bool:
    if not periode:
        return True
    dates = {f"{a}-{m}" for a, m in DATE.findall(ligne)}
    if dates:
        return any(p in dates for p in periode)
    return piece_ok


def ajouter(
    prises: dict[int, str],
    lignes: list[str],
    i: int,
    piece: str,
    motif: str,
) -> None:
    prises.setdefault(i, motif)
    if est_tableau(lignes[i], piece):
        h = entete_avant(lignes, i, piece)
        if h is not None:
            prises.setdefault(h, "en-tête du tableau")
    else:
        for j in (i - 1, i + 1):
            if 0 <= j < len(lignes) and lignes[j].strip():
                prises.setdefault(j, "ligne voisine")


def config_unite(unite: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "titre": unite["titre"],
        "cles": list(unite["cles"]),
        "variantes": _cles_variantes(unite.get("variantes")),
        "periode": list(unite.get("periode") or []),
    }


def rassembler(
    unite: Mapping[str, Any],
    corpus: Mapping[str, list[str]],
    *,
    toutes_cles: list[str] | None = None,
    index_cles: Mapping[tuple[str, int], frozenset[str]] | None = None,
) -> dict[str, Any]:
    e = config_unite(unite)
    if toutes_cles is None:
        toutes_cles = list(e["cles"]) + list(e["variantes"])
    cles_entite = _cles_entite_sans_partagees(unite)
    tronque = False
    sections: dict[str, dict[str, dict[int, str]]] = {
        "entite": {},
        "document": {},
        "variantes": {},
        "emergent": {},
    }
    paquet_groupes: list[dict[str, Any]] = []
    if unite.get("type") == "grandeur":
        groupes_unite = unite.get("groupes")
        if groupes_unite:
            deja_vus: set[tuple[str, int]] = set()
            for grp in groupes_unite:
                if not isinstance(grp, Mapping):
                    continue
                contenu: dict[str, dict[str, str]] = {}
                for item in grp.get("lignes") or []:
                    if not isinstance(item, Mapping):
                        continue
                    piece = item.get("piece")
                    if not isinstance(piece, str) or piece not in corpus:
                        continue
                    for n in item.get("lignes") or []:
                        try:
                            i = int(n) - 1
                        except (TypeError, ValueError):
                            continue
                        if not (0 <= i < len(corpus[piece])) or not corpus[piece][i].strip():
                            continue
                        cle = (piece, i)
                        num = str(i + 1)
                        if cle in deja_vus:
                            texte = f"{piece}:{num}"
                        else:
                            texte = corpus[piece][i]
                            deja_vus.add(cle)
                        contenu.setdefault(piece, {})[num] = texte
                        sections["entite"].setdefault(piece, {})[i] = "ligne du groupe"
                if contenu:
                    paquet_groupes.append(
                        {
                            "titre": grp.get("titre") or "grandeur",
                            "lignes": {
                                p: {k: contenu[p][k] for k in sorted(contenu[p], key=int)}
                                for p in sorted(contenu)
                            },
                        }
                    )
        else:
            for item in unite.get("lignes") or []:
                if not isinstance(item, Mapping):
                    continue
                piece = item.get("piece")
                if not isinstance(piece, str) or piece not in corpus:
                    continue
                prises: dict[int, str] = {}
                for n in item.get("lignes") or []:
                    try:
                        i = int(n) - 1
                    except (TypeError, ValueError):
                        continue
                    if 0 <= i < len(corpus[piece]) and corpus[piece][i].strip():
                        prises[i] = "ligne du groupe"
                if prises:
                    cible = sections["entite"].setdefault(piece, {})
                    cible.update(prises)
        for piece in list(sections["entite"]):
            sections["entite"][piece] = dict(sorted(sections["entite"][piece].items()))
    else:
        for piece, lignes in corpus.items():
            if _est_document_entite(lignes, cles_entite, e["periode"], toutes_cles):
                sections["document"][piece] = {
                    i: "la pièce est le document de l'entité"
                    for i, l in enumerate(lignes[:MAX_DOC])
                    if l.strip()
                }
                continue
            prises = _prises_entite(
                lignes,
                piece,
                cles_entite,
                e["periode"],
                toutes_cles,
                index_cles=index_cles,
            )
            if prises:
                sections["entite"][piece] = dict(sorted(prises.items()))
            hors: dict[int, str] = {}
            for i, ligne in enumerate(lignes):
                if i in prises:
                    continue
                if _porte_cles(
                    ligne,
                    e["variantes"],
                    toutes_cles,
                    _portees_index(index_cles, piece, i),
                ):
                    ajouter(hors, lignes, i, piece, "variante de nom, hors période")
            if hors:
                triees = sorted(hors.items())
                if len(triees) > MAX_PAR_PIECE:
                    tronque = True
                    triees = triees[:MAX_PAR_PIECE]
                sections["variantes"][piece] = dict(triees)
    deja = {(p, i) for s in sections.values() for p, d in s.items() for i in d}
    texte_retenu = "\n".join(corpus[p][i] for p, i in deja)
    termes = sorted(set(TERME.findall(texte_retenu)))
    valeurs = sorted(set(VALEUR.findall(texte_retenu)) - set(e["cles"]))
    repartition = {
        v: {p for p, lignes in corpus.items() if any(v in l for l in lignes)} for v in valeurs
    }
    distinctives = [v for v in valeurs if 2 <= len(repartition[v]) <= MAX_PIECES_VALEUR]
    for piece, lignes in corpus.items():
        emer: dict[int, str] = {}
        for i, ligne in enumerate(lignes):
            if (piece, i) in deja or not ligne.strip():
                continue
            if not est_tableau(ligne, piece) and any(
                re.search(rf"\b{re.escape(t)}\b", ligne) for t in termes
            ):
                emer.setdefault(i, "définit un terme rencontré")
            elif any(v in ligne for v in distinctives) and dans_periode(ligne, e["periode"], True):
                ajouter(emer, lignes, i, piece, "porte une valeur de l'entité")
        emer = {i: m for i, m in emer.items() if (piece, i) not in deja}
        if emer:
            triees = sorted(emer.items())
            if len(triees) > MAX_EMERGENT_PAR_PIECE:
                tronque = True
                triees = triees[:MAX_EMERGENT_PAR_PIECE]
            sections["emergent"][piece] = dict(triees)
    sortie: dict[str, Any] = {
        "entite": unite["id"],
        "titre": e["titre"],
        "cles": e["cles"],
        "periode": e["periode"],
        "termes_emergents": termes,
        "valeurs_distinctives": distinctives,
        "sections": {},
        "taille": {},
        "tronque": tronque,
    }
    if paquet_groupes:
        sortie["groupes"] = paquet_groupes
    for nom_s, contenu in sections.items():
        sortie["sections"][nom_s] = {
            p: {str(i + 1): corpus[p][i] for i in sorted(d)} for p, d in contenu.items()
        }
        sortie["taille"][nom_s] = {
            "pieces": len(contenu),
            "lignes": sum(len(d) for d in contenu.values()),
            "caracteres": sum(len(corpus[p][i]) for p, d in contenu.items() for i in d),
        }
    return sortie


def presenter(paquet: Mapping[str, Any]) -> str:
    out = [f"ENTITÉ RECHERCHÉE : {paquet['titre']}", f"clés : {', '.join(paquet['cles'])}"]
    groupes = paquet.get("groupes")
    if groupes:
        for grp in groupes:
            out.append(f"\n=== {grp.get('titre') or 'grandeur'} ===")
            for piece, lignes in (grp.get("lignes") or {}).items():
                out.append(f"\n--- {piece}")
                out += [f"{int(n):4d}| {t}" for n, t in lignes.items()]
        return "\n".join(out)
    for section in ("document", "entite", "variantes", "emergent"):
        contenu = paquet["sections"].get(section) or {}
        if not contenu:
            continue
        out.append(f"\n=== {TITRES[section]} ===")
        for piece, lignes in contenu.items():
            out.append(f"\n--- {piece}")
            out += [f"{int(n):4d}| {t}" for n, t in lignes.items()]
    return "\n".join(out)


def reliquat(
    registre: Mapping[str, Any],
    corpus: Mapping[str, list[str]],
    *,
    toutes_cles: list[str] | None = None,
    index_cles: Mapping[tuple[str, int], frozenset[str]] | None = None,
    paquets: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Lignes non vides hors séparateurs, absentes des sections entite et document."""
    if toutes_cles is None:
        toutes_cles = _toutes_cles_entites(registre.get("entites") or [])
    prises: set[tuple[str, int]] = set()
    for unite in registre.get("unites") or []:
        if unite.get("type") == "grandeur":
            continue
        uid = unite["id"]
        if paquets is not None and uid in paquets:
            paquet = paquets[uid]
        else:
            paquet = rassembler(
                unite, corpus, toutes_cles=toutes_cles, index_cles=index_cles
            )
        for section in ("entite", "document"):
            for piece, lignes in (paquet.get("sections", {}).get(section) or {}).items():
                for n in lignes:
                    prises.add((piece, int(n) - 1))
    par_piece: dict[str, dict[str, dict[str, str]]] = {}
    n_lignes = 0
    n_car = 0
    for piece, lignes in corpus.items():
        for i, ligne in enumerate(lignes):
            if not ligne.strip() or est_separateur_tableau(ligne):
                continue
            if (piece, i) in prises:
                continue
            n_lignes += 1
            n_car += len(ligne)
            par_piece.setdefault(piece, {})[str(i + 1)] = {
                "texte": ligne,
                "motif": "reliquat",
            }
            if est_tableau(ligne, piece):
                h = entete_avant(lignes, i, piece)
                if h is None:
                    continue
                cle_h = str(h + 1)
                if cle_h not in par_piece.get(piece, {}):
                    par_piece.setdefault(piece, {})[cle_h] = {
                        "texte": lignes[h],
                        "motif": "rappel",
                    }
    par_trie = {
        p: {k: par_piece[p][k] for k in sorted(par_piece[p], key=int)}
        for p in sorted(par_piece)
    }
    return {"par_piece": par_trie, "lignes": n_lignes, "caracteres": n_car}


def presenter_reliquat(reste: Mapping[str, Any]) -> str:
    out: list[str] = []
    for piece, lignes in (reste.get("par_piece") or {}).items():
        out.append(f"--- {piece}")
        for n, item in lignes.items():
            texte = item["texte"] if isinstance(item, Mapping) else str(item)
            out.append(f"{int(n):4d}| {texte}")
    return "\n".join(out)


def _sous_reliquat(reste: Mapping[str, Any], pieces: list[str]) -> dict[str, Any]:
    par = {
        p: reste["par_piece"][p]
        for p in pieces
        if p in (reste.get("par_piece") or {})
    }
    n_lignes = 0
    n_car = 0
    for contenu in par.values():
        for item in contenu.values():
            if item.get("motif") == "rappel":
                continue
            n_lignes += 1
            n_car += len(item.get("texte") or "")
    return {"par_piece": par, "lignes": n_lignes, "caracteres": n_car}


def invite_reliquat(
    reste: Mapping[str, Any], *, invite: str = INVITE_RELIQUAT
) -> str:
    corps = presenter_reliquat(reste)
    if corps:
        return corps + "\n\n" + invite
    return invite


def decouper_lots_reliquat(
    reste: Mapping[str, Any],
    *,
    octets_max: int = OCTETS_MAX_LOTS,
    invite: str = INVITE_RELIQUAT,
) -> list[dict[str, Any]]:
    if not reste.get("par_piece"):
        return []

    def taille(r: Mapping[str, Any]) -> int:
        return len(invite_reliquat(r, invite=invite).encode("utf-8"))

    if taille(reste) <= octets_max:
        return [dict(reste)]
    lots: list[dict[str, Any]] = []
    courant: list[str] = []
    for piece in reste["par_piece"]:
        candidat = courant + [piece]
        if courant and taille(_sous_reliquat(reste, candidat)) > octets_max:
            lots.append(_sous_reliquat(reste, courant))
            courant = [piece]
        else:
            courant = candidat
    if courant:
        lots.append(_sous_reliquat(reste, courant))
    return lots


def _mots_titre(titre: str) -> set[str]:
    return {m.lower() for m in MOT_TITRE.findall(titre or "") if len(m) >= 3}


def _union_lignes_groupe(
    a: list[Mapping[str, Any]], b: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    par: dict[str, set[int]] = {}
    for item in list(a) + list(b):
        piece = item.get("piece")
        if not isinstance(piece, str):
            continue
        try:
            nums = {int(n) for n in item.get("lignes") or []}
        except (TypeError, ValueError):
            continue
        par.setdefault(piece, set()).update(nums)
    return [{"piece": p, "lignes": sorted(nums)} for p, nums in sorted(par.items())]


def _fusionnables(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a.get("concept") != b.get("concept"):
        return False
    if a.get("periode") != b.get("periode"):
        return False
    wa, wb = _mots_titre(str(a.get("titre") or "")), _mots_titre(str(b.get("titre") or ""))
    if not wa or not wb:
        return False
    return len(wa & wb) * 2 >= min(len(wa), len(wb))


def fusionner_groupes_entre_lots(
    groupes_par_lot: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    items: list[tuple[int, dict[str, Any]]] = [
        (i, g) for i, lot in enumerate(groupes_par_lot) for g in lot
    ]
    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if items[i][0] == items[j][0]:
                continue
            if _fusionnables(items[i][1], items[j][1]):
                union(i, j)
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    result: list[dict[str, Any]] = []
    for membres in sorted(clusters.values(), key=lambda m: m[0]):
        base = copy.deepcopy(items[membres[0]][1])
        for k in membres[1:]:
            base["lignes"] = _union_lignes_groupe(
                base.get("lignes") or [], items[k][1].get("lignes") or []
            )
        result.append(base)
    return result


def valider_groupes_reliquat(
    groupes: list[Any],
    reste: Mapping[str, Any],
    *,
    ajouter_reste: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    montrees: dict[str, set[int]] = {
        piece: {int(n) for n in lignes}
        for piece, lignes in (reste.get("par_piece") or {}).items()
    }
    ecartes: list[str] = []
    valides: list[dict[str, Any]] = []
    couvertes: dict[str, set[int]] = {p: set() for p in montrees}
    for groupe in groupes or []:
        if not isinstance(groupe, Mapping):
            continue
        titre = groupe.get("titre") if isinstance(groupe.get("titre"), str) else ""
        concept = groupe.get("concept")
        if concept not in CONCEPTS_RELIQUAT:
            concept = "autre"
        periode = groupe.get("periode")
        if periode in ("null", ""):
            periode = None
        refs: list[dict[str, Any]] = []
        for item in groupe.get("lignes") or []:
            if not isinstance(item, Mapping):
                continue
            piece = item.get("piece")
            try:
                nums = sorted({int(n) for n in item.get("lignes") or []})
            except (TypeError, ValueError):
                ecartes.append(f"{titre} : lignes invalides ({piece})")
                continue
            if piece not in montrees:
                ecartes.append(f"{titre} : pièce absente du reliquat ({piece})")
                continue
            bonnes = [n for n in nums if n in montrees[piece]]
            if len(bonnes) != len(nums):
                ecartes.append(
                    f"{titre} : {piece} lignes hors reliquat "
                    f"{sorted(set(nums) - set(bonnes))}"
                )
            if bonnes:
                refs.append({"piece": piece, "lignes": bonnes})
                couvertes.setdefault(piece, set()).update(bonnes)
        if refs:
            valides.append(
                {
                    "titre": titre or "grandeur",
                    "concept": concept,
                    "periode": periode,
                    "lignes": refs,
                }
            )
        elif groupe.get("lignes"):
            ecartes.append(f"{titre} : groupe vide écarté")
    if ajouter_reste:
        for piece, nums in montrees.items():
            manque = sorted(nums - couvertes.get(piece, set()))
            if manque:
                valides.append(
                    {
                        "titre": f"reliquat de {piece}",
                        "concept": "autre",
                        "periode": None,
                        "lignes": [{"piece": piece, "lignes": manque}],
                    }
                )
    return valides, ecartes


def _volume_refs(
    refs: list[Mapping[str, Any]], corpus: Mapping[str, list[str]]
) -> int:
    total = 0
    vus: set[tuple[str, int]] = set()
    for item in refs:
        piece = item.get("piece")
        if piece not in corpus:
            continue
        for n in item.get("lignes") or []:
            i = int(n) - 1
            cle = (piece, i)
            if cle in vus or not (0 <= i < len(corpus[piece])):
                continue
            vus.add(cle)
            total += len(corpus[piece][i])
    return total


def _mois_refs(
    refs: list[Mapping[str, Any]], corpus: Mapping[str, list[str]]
) -> set[str]:
    mois: set[str] = set()
    for item in refs:
        piece = item.get("piece")
        if piece not in corpus:
            continue
        for n in item.get("lignes") or []:
            i = int(n) - 1
            if 0 <= i < len(corpus[piece]):
                mois.update(f"{a}-{m}" for a, m in DATE.findall(corpus[piece][i]))
    return mois


def _partager_piece_periode(
    groupe: Mapping[str, Any],
    corpus: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str | None], list[int]] = {}
    for item in groupe.get("lignes") or []:
        piece = item.get("piece")
        if piece not in corpus:
            continue
        for n in item.get("lignes") or []:
            i = int(n) - 1
            if not (0 <= i < len(corpus[piece])):
                continue
            found = [f"{a}-{m}" for a, m in DATE.findall(corpus[piece][i])]
            if found:
                for mo in found:
                    buckets.setdefault((piece, mo), []).append(int(n))
            else:
                buckets.setdefault((piece, None), []).append(int(n))
    parts: list[dict[str, Any]] = []
    for (piece, mo), nums in sorted(
        buckets.items(), key=lambda x: (x[0][0], x[0][1] or "")
    ):
        parts.append(
            {
                "titre": groupe.get("titre") or "grandeur",
                "concept": groupe.get("concept") or "autre",
                "periode": mo,
                "lignes": [{"piece": piece, "lignes": sorted(set(nums))}],
            }
        )
    return parts


def _id_unique(base: str, vus: set[str]) -> str:
    if not base:
        base = "grandeur"
    if base not in vus:
        vus.add(base)
        return base
    i = 2
    while f"{base}-{i}" in vus:
        i += 1
    ident = f"{base}-{i}"
    vus.add(ident)
    return ident


def _periode_liste(periode: Any) -> list[str]:
    if periode in (None, "", "null"):
        return []
    if isinstance(periode, list):
        return [str(p) for p in periode if p not in (None, "")]
    return [str(periode)]


def _titre_grandeur(titre: str, periode: Any) -> str:
    periode_liste = _periode_liste(periode)
    if not periode_liste:
        return titre
    if len(periode_liste) == 1:
        return f"{titre}, {periode_liste[0]}"
    return f"{titre}, {periode_liste[0]} à {periode_liste[-1]}"


def _unite_grandeur(groupe: Mapping[str, Any], vus: set[str]) -> dict[str, Any]:
    titre_groupe = str(groupe.get("titre") or "grandeur")
    periode_liste = _periode_liste(groupe.get("periode"))
    ident = _identifiant_unite(
        {"nom": titre_groupe, "type": "grandeur", "cles": []},
        periode_liste,
    )
    ident = _id_unique(ident, vus)
    return {
        "id": ident,
        "entite": titre_groupe,
        "type": "grandeur",
        "titre": _titre_grandeur(titre_groupe, groupe.get("periode")),
        "cles": [],
        "variantes": [],
        "periode": periode_liste,
        "lignes": list(groupe.get("lignes") or []),
        "concept": groupe.get("concept") or "autre",
    }


def _premiere_piece_groupe(groupe: Mapping[str, Any]) -> str:
    pieces = [
        item.get("piece")
        for item in groupe.get("lignes") or []
        if isinstance(item, Mapping) and isinstance(item.get("piece"), str)
    ]
    return min(pieces) if pieces else ""


def _tri_groupes_petits(groupes: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        groupes,
        key=lambda g: (_premiere_piece_groupe(g), str(g.get("titre") or "")),
    )


def _unite_grandeur_lot(
    groupes: list[Mapping[str, Any]], rang: int, vus: set[str]
) -> dict[str, Any]:
    premier = groupes[0]
    titre_premier = str(premier.get("titre") or "grandeur")
    n_autres = len(groupes) - 1
    titre = (
        f"{titre_premier} et {n_autres} autres grandeurs"
        if n_autres > 0
        else titre_premier
    )
    ident = _id_unique(f"grandeur-lot-{rang:03d}-{_slug(titre_premier)}", vus)
    groupes_info = [
        {
            "titre": str(g.get("titre") or "grandeur"),
            "concept": g.get("concept") or "autre",
            "periode": g.get("periode"),
            "lignes": list(g.get("lignes") or []),
        }
        for g in groupes
    ]
    return {
        "id": ident,
        "entite": titre_premier,
        "type": "grandeur",
        "titre": titre,
        "cles": [],
        "variantes": [],
        "periode": [],
        "groupes": groupes_info,
        "concept": premier.get("concept") or "autre",
    }


def _empaqueter_groupes_petits(
    groupes: list[Mapping[str, Any]],
    corpus: Mapping[str, list[str]],
    vus: set[str],
) -> list[dict[str, Any]]:
    unites: list[dict[str, Any]] = []
    lot_courant: list[Mapping[str, Any]] = []
    rang = 1

    def flush() -> None:
        nonlocal rang
        if not lot_courant:
            return
        if len(lot_courant) == 1:
            unites.append(_unite_grandeur(lot_courant[0], vus))
        else:
            unites.append(_unite_grandeur_lot(lot_courant, rang, vus))
            rang += 1
        lot_courant.clear()

    for groupe in _tri_groupes_petits(groupes):
        refs = list(groupe.get("lignes") or [])
        union = _union_lignes_groupe(
            [item for g in lot_courant for item in g.get("lignes") or []],
            refs,
        )
        if lot_courant and _volume_refs(union, corpus) >= SEUIL_PLAGE:
            flush()
        lot_courant.append(groupe)
    flush()
    return unites


def unites_depuis_groupes(
    groupes: list[Mapping[str, Any]],
    corpus: Mapping[str, list[str]],
    ids_existants: set[str] | None = None,
) -> list[dict[str, Any]]:
    vus = set(ids_existants or [])
    unites: list[dict[str, Any]] = []
    petits: list[Mapping[str, Any]] = []
    for groupe in groupes:
        refs = list(groupe.get("lignes") or [])
        mois = _mois_refs(refs, corpus)
        if _volume_refs(refs, corpus) >= SEUIL_PLAGE:
            if len(mois) >= 2:
                for part in _partager_piece_periode(groupe, corpus):
                    unites.append(_unite_grandeur(part, vus))
            else:
                unites.append(_unite_grandeur(groupe, vus))
        else:
            petits.append(groupe)
    unites.extend(_empaqueter_groupes_petits(petits, corpus, vus))
    return unites


def _brut_reliquat_lisible(texte: str) -> bool:
    try:
        repare, _ = reparer_cles_begayees(texte)
        payload = _objet_json(repare, "groupes")
        return isinstance(payload.get("groupes"), list)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _chemins_bruts_reliquat(sortie_dir: Path, n_lots: int) -> list[Path]:
    return [sortie_dir / f"reliquat-lot-{i:02d}.brut.txt" for i in range(n_lots)]


def reliquat_bruts_reutilisables(sortie_dir: Path, n_lots: int) -> bool:
    if n_lots <= 0:
        return True
    for path in _chemins_bruts_reliquat(sortie_dir, n_lots):
        if not path.is_file():
            return False
        try:
            if not _brut_reliquat_lisible(path.read_text(encoding="utf-8")):
                return False
        except OSError:
            return False
    return True


def lire_objet_reliquat(brut: str) -> tuple[dict[str, Any], int]:
    repare, nombre = reparer_cles_begayees(brut)
    return _objet_json(repare, "groupes"), nombre


def reliquat_peut_reprendre(*, pieces_dir: Path, sortie_dir: Path) -> bool:
    registre_path = sortie_dir / "registre.json"
    if not registre_path.is_file():
        return False
    registre = json.loads(registre_path.read_text(encoding="utf-8"))
    corpus, _ = charger_corpus(pieces_dir)
    toutes_cles = _toutes_cles_entites(registre.get("entites") or [])
    index = index_cles_portees(corpus, toutes_cles)
    reste = reliquat(registre, corpus, toutes_cles=toutes_cles, index_cles=index)
    lots = decouper_lots_reliquat(reste)
    return reliquat_bruts_reutilisables(sortie_dir, len(lots))


def _ecarter_bruts_reliquat(sortie_dir: Path) -> None:
    for path in sorted(sortie_dir.glob("reliquat-lot-*.brut.txt")):
        _ecarter_brut(path, sortie_dir / "ecartes")


def regrouper_reliquat(
    *,
    pieces_dir: Path,
    sortie_dir: Path,
    dossier: str,
    chat_fn: ChatFn | None = None,
    compteur=None,
    delai_s: int = 900,
    refaire: bool = False,
) -> dict[str, Any]:
    registre_path = sortie_dir / "registre.json"
    if not registre_path.is_file():
        raise FileNotFoundError("registre.json absent — lancer recenser d'abord")
    registre = json.loads(registre_path.read_text(encoding="utf-8"))
    corpus, _ = charger_corpus(pieces_dir)
    toutes_cles = _toutes_cles_entites(registre.get("entites") or [])
    index = index_cles_portees(corpus, toutes_cles)
    reste = reliquat(registre, corpus, toutes_cles=toutes_cles, index_cles=index)
    lots = decouper_lots_reliquat(reste)
    sortie_dir.mkdir(parents=True, exist_ok=True)
    appel = False
    reparations = 0
    ecartes: list[str] = []
    groupes_par_lot: list[list[dict[str, Any]]] = []
    if refaire:
        _ecarter_bruts_reliquat(sortie_dir)
    if lots and (refaire or not reliquat_bruts_reutilisables(sortie_dir, len(lots))):
        appel = True
        if chat_fn is not None:
            chat = chat_fn
            reserver = compteur is not None
        else:
            from .situations_composites import _chat_par_defaut

            chat = _chat_par_defaut(
                "regroupement-reliquat",
                compteur=compteur,
                dossier=dossier,
                delai_s=delai_s,
            )
            reserver = False
        for i, lot in enumerate(lots):
            if reserver:
                compteur.reserver_emission()
            brut, _meta = chat("", invite_reliquat(lot))
            brut_path = sortie_dir / f"reliquat-lot-{i:02d}.brut.txt"
            brut_path.write_text(brut, encoding="utf-8")
    elif lots:
        for i, _lot in enumerate(lots):
            brut_path = sortie_dir / f"reliquat-lot-{i:02d}.brut.txt"
            brut = brut_path.read_text(encoding="utf-8")
            if not _brut_reliquat_lisible(brut):
                raise ValueError(f"réponse illisible pour le lot {i}")
    for i, lot in enumerate(lots):
        brut = (sortie_dir / f"reliquat-lot-{i:02d}.brut.txt").read_text(
            encoding="utf-8"
        )
        if not _brut_reliquat_lisible(brut):
            _ecarter_brut(
                sortie_dir / f"reliquat-lot-{i:02d}.brut.txt",
                sortie_dir / "ecartes",
            )
            raise ValueError(f"réponse illisible pour le lot {i}")
        payload, n_rep = lire_objet_reliquat(brut)
        reparations += n_rep
        bruts_groupes = payload.get("groupes") or []
        valides, ecartes_lot = valider_groupes_reliquat(
            bruts_groupes, lot, ajouter_reste=False
        )
        ecartes.extend(ecartes_lot)
        groupes_par_lot.append(valides)
    if lots:
        fusionnes = fusionner_groupes_entre_lots(groupes_par_lot)
        groupes, ecartes_final = valider_groupes_reliquat(fusionnes, reste)
        ecartes.extend(ecartes_final)
    else:
        groupes = []
    ids_existants = {
        u["id"] for u in registre.get("unites") or [] if u.get("type") != "grandeur"
    }
    unites_g = unites_depuis_groupes(groupes, corpus, ids_existants)
    registre["unites"] = [
        u for u in registre.get("unites") or [] if u.get("type") != "grandeur"
    ] + unites_g
    registre_path.write_text(
        json.dumps(registre, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sortie = {
        "groupes": groupes,
        "ecartes": ecartes,
        "reparations": reparations,
        "lots": len(lots),
        "appel": appel,
        "unites_grandeur": len(unites_g),
    }
    (sortie_dir / "groupes-reliquat.json").write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sortie


def reparer_cles_begayees(texte: str) -> tuple[str, int]:
    repare, nombre = CLE_BEGAYEE.subn(r'"\1":', texte)
    return repare, nombre


def _taille_invite(paquet: Mapping[str, Any]) -> int:
    return len((presenter(paquet) + "\n\n" + INVITE).encode("utf-8"))


def reduire_paquet(
    paquet: dict[str, Any],
    *,
    octets_max: int = OCTETS_MAX_INVITE,
) -> tuple[dict[str, Any], list[str]]:
    paquet = copy.deepcopy(paquet)
    reductions: list[str] = []
    if _taille_invite(paquet) <= octets_max:
        return paquet, reductions
    if paquet["sections"].get("emergent"):
        paquet["sections"]["emergent"] = {}
        paquet["taille"]["emergent"] = {"pieces": 0, "lignes": 0, "caracteres": 0}
        reductions.append("section emergent retirée")
    entite = paquet["sections"].setdefault("entite", {})
    while _taille_invite(paquet) > octets_max and entite:
        numeros: list[tuple[int, str]] = []
        for piece, lignes in entite.items():
            for num_str in lignes:
                numeros.append((int(num_str), piece))
        if not numeros:
            break
        numeros.sort()
        num, piece = numeros[0]
        del entite[piece][str(num)]
        if not entite[piece]:
            del entite[piece]
        reductions.append(f"entite {piece}:{num} retirée")
        paquet["taille"]["entite"] = {
            "pieces": len(entite),
            "lignes": sum(len(d) for d in entite.values()),
            "caracteres": sum(
                len(paquet["sections"]["entite"][p][n])
                for p, d in entite.items()
                for n in d
            ),
        }
    return paquet, reductions


def _reponse_lisible(brut: str) -> bool:
    try:
        repare, _ = reparer_cles_begayees(brut)
        _objet_json(repare, "faits")
        return True
    except ValueError:
        return False


def _ecarter_brut(brut_path: Path, ecartes_dir: Path) -> None:
    ecartes_dir.mkdir(parents=True, exist_ok=True)
    destination = ecartes_dir / brut_path.name
    if destination.exists():
        destination.unlink()
    brut_path.replace(destination)


def invalider_brut_unite(
    sortie_dir: Path,
    unite_id: str,
    *,
    maintenant: datetime | None = None,
) -> Path | None:
    """Déplace ``bruts/<id>.brut.txt`` vers ``ecartes/`` avec un horodatage."""
    brut_path = Path(sortie_dir) / "bruts" / f"{unite_id}.brut.txt"
    if not brut_path.is_file():
        return None
    ts = (maintenant or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    ecartes_dir = Path(sortie_dir) / "ecartes"
    ecartes_dir.mkdir(parents=True, exist_ok=True)
    destination = ecartes_dir / f"{unite_id}.brut.{ts}.txt"
    if destination.exists():
        destination = ecartes_dir / f"{unite_id}.brut.{ts}.{os.getpid()}.txt"
    brut_path.replace(destination)
    return destination


def rassembler_une_unite(
    unite: Mapping[str, Any],
    *,
    pieces_dir: Path,
    sortie_dir: Path,
    corpus: Mapping[str, list[str]] | None = None,
    toutes_cles: list[str] | None = None,
    index_cles: Mapping[tuple[str, int], frozenset[str]] | None = None,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Re-rassemble le paquet d'une seule unité et l'écrit sur disque.

    ``destination`` écrit ailleurs que ``paquets/<id>.json`` (paquet de
    référence). Absent : le paquet de référence est remplacé.
    """
    if corpus is None:
        corpus, _ = charger_corpus(pieces_dir)
    registre_path = Path(sortie_dir) / "registre.json"
    if toutes_cles is None:
        if not registre_path.is_file():
            raise FileNotFoundError("registre.json absent — lancer recenser d'abord")
        registre = json.loads(registre_path.read_text(encoding="utf-8"))
        toutes_cles = _toutes_cles_entites(registre.get("entites") or [])
    if index_cles is None:
        index_cles = index_cles_portees(corpus, toutes_cles)
    paquet = rassembler(unite, corpus, toutes_cles=toutes_cles, index_cles=index_cles)
    if destination is not None:
        cible = Path(destination)
    else:
        paquets_dir = Path(sortie_dir) / "paquets"
        paquets_dir.mkdir(parents=True, exist_ok=True)
        cible = paquets_dir / f"{unite['id']}.json"
    cible.parent.mkdir(parents=True, exist_ok=True)
    tmp = cible.parent / f"{cible.name}.{os.getpid()}.tmp"
    tmp.write_text(
        json.dumps(paquet, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    os.replace(tmp, cible)
    return paquet


@contextmanager
def verrou_registre(reconstitution_dir: Path) -> Iterator[None]:
    """Verrou exclusif pour lecture-modification-écriture de ``registre.json``."""
    reconstitution_dir.mkdir(parents=True, exist_ok=True)
    lock_path = reconstitution_dir / "registre.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def verrou_publication(reconstitution_dir: Path) -> Iterator[None]:
    """Verrou exclusif de publication (vue, pages, vecteurs, promotion, génération)."""
    reconstitution_dir.mkdir(parents=True, exist_ok=True)
    lock_path = reconstitution_dir / "publication.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def ecrire_json_atomique(path: Path, data: Any) -> None:
    """Écriture atomique avec fichier temporaire unique par processus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def lire_registre_reconstitution(reconstitution_dir: Path) -> dict[str, Any]:
    path = reconstitution_dir / "registre.json"
    if not path.is_file():
        raise FileNotFoundError(f"registre.json absent : {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("registre.json illisible")
    return payload


def persister_registre_reconstitution(
    reconstitution_dir: Path, registre: dict[str, Any]
) -> None:
    ecrire_json_atomique(reconstitution_dir / "registre.json", registre)


def modifier_registre(
    reconstitution_dir: Path,
    modifier: Callable[[dict[str, Any]], bool],
) -> bool:
    """Applique ``modifier`` sous verrou ; persiste si ``modifier`` renvoie ``True``."""
    with verrou_registre(reconstitution_dir):
        registre = lire_registre_reconstitution(reconstitution_dir)
        if not modifier(registre):
            return False
        persister_registre_reconstitution(reconstitution_dir, registre)
        return True


def lire_generation_reconstitution(reconstitution_dir: Path) -> int:
    path = reconstitution_dir / "generation.json"
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    return int(payload.get("generation") or 0)


def incrementer_generation_reconstitution(reconstitution_dir: Path) -> int:
    """Incrémente atomiquement le marqueur de génération (invalidation API inter-processus)."""
    with verrou_registre(reconstitution_dir):
        path = reconstitution_dir / "generation.json"
        generation = lire_generation_reconstitution(reconstitution_dir) + 1
        ecrire_json_atomique(
            path,
            {
                "generation": generation,
                "horodatage": datetime.now(timezone.utc).isoformat(),
            },
        )
        return generation


def retirer_perimee(registre: dict[str, Any], unite_id: str) -> bool:
    """Retire la clé ``perimee`` de l'unité ; ``True`` si elle était présente."""
    for unite in registre.get("unites") or []:
        if isinstance(unite, dict) and str(unite.get("id") or "") == unite_id:
            if "perimee" in unite:
                del unite["perimee"]
                return True
            return False
    return False


def retirer_perimee_persiste(reconstitution_dir: Path, unite_id: str) -> bool:
    """Retire ``perimee`` pour ``unite_id`` et persiste le registre sous verrou."""

    def modifier(registre: dict[str, Any]) -> bool:
        return retirer_perimee(registre, unite_id)

    return modifier_registre(reconstitution_dir, modifier)


def retirer_perimee_si_identique(
    reconstitution_dir: Path,
    unite_id: str,
    *,
    depuis: str | None,
) -> str:
    """Retire ``perimee`` seulement si ``depuis`` correspond encore.

    Retourne ``retire``, ``absent`` ou ``perime_depuis``. ``depuis is None``
    signifie qu'aucun marqueur n'était capturé à l'entrée : un marqueur
    apparu entre-temps est conservé.
    """
    etat = {"valeur": "absent"}

    def modifier(registre: dict[str, Any]) -> bool:
        for unite in registre.get("unites") or []:
            if not isinstance(unite, dict) or str(unite.get("id") or "") != unite_id:
                continue
            perimee = unite.get("perimee")
            if not isinstance(perimee, dict):
                etat["valeur"] = "absent"
                return False
            courant = perimee.get("depuis")
            courant_s = str(courant) if courant is not None else None
            if depuis is None or courant_s != str(depuis):
                etat["valeur"] = "perime_depuis"
                return False
            del unite["perimee"]
            etat["valeur"] = "retire"
            return True
        return False

    modifier_registre(reconstitution_dir, modifier)
    return etat["valeur"]


def chemin_objet_en_cours(reconstitution_dir: Path, unite_id: str) -> Path:
    return reconstitution_dir / "en-cours" / f"{unite_id}.json"


def chemin_objet_publie(reconstitution_dir: Path, unite_id: str) -> Path:
    return reconstitution_dir / "entites" / f"{unite_id}.json"


def publier_objet_en_cours(reconstitution_dir: Path, unite_id: str) -> None:
    """Remplace ``entites/<id>.json`` par l'objet mis en scène, atomiquement."""
    source = chemin_objet_en_cours(reconstitution_dir, unite_id)
    if not source.is_file():
        raise FileNotFoundError(f"objet en cours absent : {source}")
    cible = chemin_objet_publie(reconstitution_dir, unite_id)
    cible.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, cible)


def ecarter_objet_en_cours(
    reconstitution_dir: Path,
    unite_id: str,
    *,
    maintenant: datetime | None = None,
) -> Path | None:
    """Déplace ``en-cours/<id>.json`` vers ``ecartes/`` avec un horodatage."""
    source = chemin_objet_en_cours(reconstitution_dir, unite_id)
    if not source.is_file():
        return None
    ts = (maintenant or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    ecartes_dir = reconstitution_dir / "ecartes"
    ecartes_dir.mkdir(parents=True, exist_ok=True)
    destination = ecartes_dir / f"{unite_id}.objet.{ts}.json"
    if destination.exists():
        destination = ecartes_dir / f"{unite_id}.objet.{ts}.{os.getpid()}.json"
    source.replace(destination)
    return destination


ETAPES_PUBLICATION = frozenset({"pages", "vecteurs", "promotion", "generation"})


def marquer_echec_publication(
    reconstitution_dir: Path,
    unite_id: str,
    erreur: str,
    *,
    etape: str = "pages",
    version_entree: str | None = None,
) -> bool:
    """Enregistre ``publication_en_attente`` après un échec de publication.

    ``version_entree`` est le ``depuis`` capturé à l'entrée, ou l'horodatage
    de l'appel pour une unité non périmée. Ne modifie pas ``perimee``.
    ``etape`` indique l'étape manquante (``pages`` | ``vecteurs`` | ``promotion``
    | ``generation``). ``objet`` pointe vers ``en-cours/`` ou ``entites/``.
    """
    etape_eff = etape if etape in ETAPES_PUBLICATION else "pages"
    publie = chemin_objet_publie(reconstitution_dir, unite_id).is_file()
    if etape_eff in {"promotion", "generation"} and publie:
        objet = f"entites/{unite_id}.json"
    else:
        objet = f"en-cours/{unite_id}.json"
    version = version_entree or datetime.now(timezone.utc).isoformat()
    horodatage = datetime.now(timezone.utc).isoformat()

    def modifier(registre: dict[str, Any]) -> bool:
        for unite in registre.get("unites") or []:
            if not isinstance(unite, dict) or str(unite.get("id") or "") != unite_id:
                continue
            unite["publication_en_attente"] = {
                "version_entree": version,
                "etape": etape_eff,
                "objet": objet,
                "erreur": erreur,
                "horodatage": horodatage,
            }
            unite.pop("reconstitution_en_attente", None)
            return True
        return False

    return modifier_registre(reconstitution_dir, modifier)


def retirer_publication_en_attente(
    reconstitution_dir: Path, unite_id: str
) -> bool:
    """Retire ``publication_en_attente`` de l'unité après succès ou abandon."""

    def modifier(registre: dict[str, Any]) -> bool:
        for unite in registre.get("unites") or []:
            if not isinstance(unite, dict) or str(unite.get("id") or "") != unite_id:
                continue
            if "publication_en_attente" not in unite:
                return False
            del unite["publication_en_attente"]
            return True
        return False

    return modifier_registre(reconstitution_dir, modifier)


def enrichir_echecs(echecs_path: Path, unite_id: str, cause: str) -> list[dict[str, str]]:
    """Ajoute ou met à jour la cause d'échec de ``unite_id`` dans ``echecs.json``."""
    from banc_realite.lecture_drive import ecrire_json_atomique

    echecs: list[dict[str, str]] = []
    if echecs_path.is_file():
        brut = json.loads(echecs_path.read_text(encoding="utf-8"))
        if isinstance(brut, list):
            echecs = [dict(item) for item in brut if isinstance(item, Mapping)]
    trouve = False
    for item in echecs:
        if str(item.get("unite") or "") == unite_id:
            item["cause"] = cause
            trouve = True
            break
    if not trouve:
        echecs.append({"unite": unite_id, "cause": cause})
    ecrire_json_atomique(echecs_path, echecs)
    return echecs


def retirer_echec(echecs_path: Path, unite_id: str) -> None:
    """Retire l'unité de ``echecs.json`` après un succès."""
    from banc_realite.lecture_drive import ecrire_json_atomique

    if not echecs_path.is_file():
        return
    brut = json.loads(echecs_path.read_text(encoding="utf-8"))
    if not isinstance(brut, list):
        return
    filtrés = [
        item
        for item in brut
        if not (isinstance(item, Mapping) and str(item.get("unite") or "") == unite_id)
    ]
    if len(filtrés) == len(brut):
        return
    ecrire_json_atomique(echecs_path, filtrés)


def rattacher(
    payload: dict[str, Any],
    paquet: Mapping[str, Any],
    corpus: Mapping[str, list[str]],
) -> list[str]:
    montrees: dict[str, set[int]] = {}
    for contenu in paquet["sections"].values():
        for piece, lignes in contenu.items():
            montrees.setdefault(piece, set()).update(int(n) for n in lignes)
    textes = {piece: corpus[piece] for piece in montrees}
    rejets: list[str] = []

    def lettre(obj: dict[str, Any], ou: str) -> bool:
        piece = obj.get("piece")
        try:
            nums = sorted({int(n) for n in obj.get("lignes") or []})
        except (TypeError, ValueError):
            nums = []
        if piece not in montrees or not nums:
            rejets.append(f"{ou} : pièce ou lignes invalides ({piece})")
            return False
        bonnes = [n for n in nums if n in montrees[piece]]
        if len(bonnes) != len(nums):
            rejets.append(
                f"{ou} : {piece} lignes non montrées {sorted(set(nums) - set(bonnes))}"
            )
        if not bonnes:
            return False
        obj["lignes"] = bonnes
        obj["lettre"] = "\n".join(textes[piece][n - 1] for n in bonnes)
        return True

    for fait in payload.get("faits") or []:
        fait["versions"] = [
            v
            for v in fait.get("versions") or []
            if lettre(v, f"fait « {fait.get('quoi', '')[:40]} »")
        ]
    payload["faits"] = [f for f in payload.get("faits") or [] if f["versions"]]
    payload["regles"] = [r for r in payload.get("regles") or [] if lettre(r, "règle")]
    payload["identites"] = [
        i for i in payload.get("identites") or [] if lettre(i, "identité")
    ]
    return rejets


def lire_objet_reconstitution(brut: str) -> tuple[dict[str, Any], int]:
    repare, nombre = reparer_cles_begayees(brut)
    return _objet_json(repare, "faits"), nombre


def reconstituer_unite(
    *,
    unite: Mapping[str, Any],
    paquet: Mapping[str, Any],
    corpus: Mapping[str, list[str]],
    sortie_dir: Path,
    dossier: str,
    chat_fn: ChatFn | None = None,
    compteur=None,
    delai_s: int = 900,
    destination: Path | None = None,
) -> dict[str, Any]:
    from .situations_composites import _chat_par_defaut

    paquet_reduit, reductions = reduire_paquet(dict(paquet))
    invite = presenter(paquet_reduit) + "\n\n" + INVITE
    if len(invite.encode("utf-8")) > OCTETS_MAX_INVITE:
        raise ValueError(
            f"invite trop grande après réduction : {len(invite.encode('utf-8'))} octets"
        )
    bruts_dir = sortie_dir / "bruts"
    ecartes_dir = sortie_dir / "ecartes"
    brut_path = bruts_dir / f"{unite['id']}.brut.txt"
    if brut_path.is_file():
        brut = brut_path.read_text(encoding="utf-8")
    else:
        if chat_fn is not None:
            if compteur is not None:
                compteur.reserver_emission()
            chat = chat_fn
        else:
            chat = _chat_par_defaut(
                "reconstitution-entite",
                compteur=compteur,
                dossier=dossier,
                piece=str(unite["id"]),
                delai_s=delai_s,
            )
        brut, _meta = chat("", invite)
        bruts_dir.mkdir(parents=True, exist_ok=True)
        brut_path.write_text(brut, encoding="utf-8")
    if not _reponse_lisible(brut):
        _ecarter_brut(brut_path, ecartes_dir)
        raise ValueError(f"réponse illisible pour {unite['id']}")
    payload, reparations = lire_objet_reconstitution(brut)
    rejets = rattacher(payload, paquet_reduit, corpus)
    payload["rejets"] = rejets
    payload["reparations"] = reparations
    payload["reductions"] = reductions
    payload["invite_caracteres"] = len(invite)
    payload["unite"] = unite["id"]
    entites_dir = destination if destination is not None else (sortie_dir / "entites")
    entites_dir.mkdir(parents=True, exist_ok=True)
    objet_path = entites_dir / f"{unite['id']}.json"
    tmp = objet_path.with_suffix(objet_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    os.replace(tmp, objet_path)
    return payload


def _percentile(ordonnees: list[int], p: float) -> int:
    if not ordonnees:
        return 0
    k = max(1, math.ceil(p * len(ordonnees)))
    return ordonnees[min(len(ordonnees), k) - 1]


def _stats_tailles(tailles: list[int]) -> dict[str, int | float]:
    if not tailles:
        return {"mediane": 0, "p90": 0, "max": 0}
    ordonnees = sorted(tailles)
    return {
        "mediane": statistics.median(ordonnees),
        "p90": _percentile(ordonnees, 0.9),
        "max": ordonnees[-1],
    }


def _compter_variantes_par_motif(entites: list[Any]) -> dict[str, int]:
    comptes = {motif: 0 for motif in MOTIFS_VARIANTE}
    for entite in entites:
        for item in entite.get("variantes") or []:
            if not isinstance(item, Mapping):
                continue
            motif = item.get("motif")
            if not isinstance(motif, str):
                continue
            comptes[motif] = comptes.get(motif, 0) + 1
    return comptes


def resume_recensement(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entites": len(payload.get("entites") or []),
        "unites": len(payload.get("unites") or []),
        "variantes_par_motif": _compter_variantes_par_motif(payload.get("entites") or []),
        "ecartes": sum(len(v) for v in (payload.get("ecartes") or {}).values()),
        "appel": bool(payload.get("appel")),
    }


def _variantes_ontologie_en_chaines(variantes: Any) -> list[str]:
    chaines: list[str] = []
    for item in variantes or []:
        if isinstance(item, str) and item.strip():
            chaines.append(item)
        elif isinstance(item, Mapping):
            cle = item.get("cle")
            if isinstance(cle, str) and cle.strip():
                chaines.append(cle)
    return chaines


def recenser_depuis_ontologie(
    *,
    ontologie_path: Path,
    pieces_dir: Path,
    sortie_dir: Path,
    registre_reference_path: Path | None = None,
    groupes_reliquat_path: Path | None = None,
) -> dict[str, Any]:
    from .etiquetage_ontologie import _entites_actives_index, _fusions_depuis_decisions

    ontologie = json.loads(ontologie_path.read_text(encoding="utf-8"))
    fusions = _fusions_depuis_decisions(ontologie.get("decisions") or [])
    index = _entites_actives_index(ontologie)
    ref_par_nom: dict[str, Mapping[str, Any]] = {}
    if registre_reference_path is not None and registre_reference_path.is_file():
        reference = json.loads(registre_reference_path.read_text(encoding="utf-8"))
        for entite in reference.get("entites") or []:
            if isinstance(entite, Mapping) and isinstance(entite.get("nom"), str):
                ref_par_nom[entite["nom"]] = entite
    entites_brutes: list[dict[str, Any]] = []
    for nom in sorted(index):
        entite = index[nom]
        typ = str(entite.get("type") or "autre")
        ref = ref_par_nom.get(nom)
        if ref is not None and "par_periode" in ref:
            par_periode = bool(ref.get("par_periode"))
        else:
            par_periode = typ != "personne"
        entites_brutes.append(
            {
                "nom": nom,
                "type": typ,
                "cles": list(entite.get("cles") or []),
                "variantes": _variantes_ontologie_en_chaines(entite.get("variantes")),
                "par_periode": par_periode,
            }
        )
    corpus, pieces_exclues = charger_corpus(pieces_dir)
    registre, ecartes = valider_registre({"entites": entites_brutes, "familles": []}, corpus)
    ids_existants = {unite["id"] for unite in registre["unites"]}
    reliquat_path = groupes_reliquat_path
    if reliquat_path is None and registre_reference_path is not None:
        reliquat_path = registre_reference_path.parent / "groupes-reliquat.json"
    if reliquat_path is not None and reliquat_path.is_file():
        reliquat_payload = json.loads(reliquat_path.read_text(encoding="utf-8"))
        groupes = reliquat_payload.get("groupes") or []
        registre["unites"].extend(
            unites_depuis_groupes(groupes, corpus, ids_existants)
        )
    sortie = {
        **registre,
        "ecartes": ecartes,
        "pieces_exclues": pieces_exclues,
    }
    sortie_dir.mkdir(parents=True, exist_ok=True)
    (sortie_dir / "registre.json").write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    meta_ontologie = {
        "version": ontologie.get("version"),
        "entites": len(registre["entites"]),
        "unites": len(registre["unites"]),
        "fusions": [{"source": source, "cible": cible} for source, cible in sorted(fusions.items())],
    }
    (sortie_dir / "recensement.ontologie.json").write_text(
        json.dumps(meta_ontologie, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**sortie, "appel": False, "recensement_ontologie": meta_ontologie}


def pieces_referencees_par_paquet(paquet: Mapping[str, Any]) -> set[str]:
    """Pièces citées par ``sections.*.<piece>`` ou ``groupes[].lignes``."""
    pieces: set[str] = set()
    sections = paquet.get("sections") or {}
    if isinstance(sections, Mapping):
        for contenu in sections.values():
            if isinstance(contenu, Mapping):
                pieces.update(str(nom) for nom in contenu)
    for groupe in paquet.get("groupes") or []:
        if not isinstance(groupe, Mapping):
            continue
        lignes = groupe.get("lignes")
        if isinstance(lignes, Mapping):
            pieces.update(str(nom) for nom in lignes)
        elif isinstance(lignes, list):
            for item in lignes:
                if isinstance(item, Mapping) and item.get("piece"):
                    pieces.add(str(item["piece"]))
                elif isinstance(item, str) and item:
                    pieces.add(item)
    return pieces


def unites_perimees(registre: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Unités du registre portant la clé ``perimee``."""
    out: list[dict[str, Any]] = []
    for unite in registre.get("unites") or []:
        if isinstance(unite, Mapping) and isinstance(unite.get("perimee"), Mapping):
            out.append(dict(unite))
    return out


def marquer_unites_perimees(
    registre: dict[str, Any],
    *,
    paquets_dir: Path,
    pieces: Iterable[str],
    depuis: str,
    passe: str,
) -> list[str]:
    """Marque périmées les unités dont le paquet stocké référence une pièce touchée.

    Conserve ``entites/<id>.json``. Ne retire pas la clé ``perimee`` (hors de
    cette passe : reconstitution #174).
    """
    cibles = {str(p) for p in pieces if p}
    marques: list[str] = []
    if not cibles:
        return marques
    for unite in registre.get("unites") or []:
        if not isinstance(unite, dict):
            continue
        uid = str(unite.get("id") or "")
        if not uid:
            continue
        paquet_path = paquets_dir / f"{uid}.json"
        if not paquet_path.is_file():
            continue
        paquet = json.loads(paquet_path.read_text(encoding="utf-8"))
        if not isinstance(paquet, Mapping):
            continue
        touchees = sorted(pieces_referencees_par_paquet(paquet) & cibles)
        if not touchees:
            continue
        unite["perimee"] = {"depuis": depuis, "pieces": touchees, "passe": passe}
        marques.append(uid)
    return marques


def empreinte_paquet_reemploi(paquet: Mapping[str, Any]) -> str:
    sans_identite = {k: v for k, v in paquet.items() if k not in ("entite", "titre", "id")}
    canon = json.dumps(sans_identite, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def index_paquets_par_empreinte(ancien_dir: Path) -> dict[str, str]:
    paquets_dir = ancien_dir / "paquets"
    index: dict[str, str] = {}
    if not paquets_dir.is_dir():
        return index
    for path in sorted(paquets_dir.glob("*.json")):
        paquet = json.loads(path.read_text(encoding="utf-8"))
        index.setdefault(empreinte_paquet_reemploi(paquet), path.stem)
    return index


def _echecs_unites_ancien(ancien_dir: Path) -> set[str]:
    path = ancien_dir / "echecs.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return set()
    return {
        str(item.get("unite") or "")
        for item in payload
        if isinstance(item, Mapping) and item.get("unite")
    }


def copier_reconstitution_reemployee(
    unite: Mapping[str, Any],
    ancien_id: str,
    *,
    ancien_dir: Path,
    sortie_dir: Path,
) -> dict[str, Any]:
    source = ancien_dir / "entites" / f"{ancien_id}.json"
    payload = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    payload["unite"] = unite["id"]
    entite = payload.get("entite")
    if isinstance(entite, dict):
        entite["nom"] = unite.get("entite", entite.get("nom"))
        if unite.get("type"):
            entite["type"] = unite["type"]
    destination = sortie_dir / "entites" / f"{unite['id']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return payload


def tenter_reemployer_unite(
    unite: Mapping[str, Any],
    paquet: Mapping[str, Any],
    *,
    ancien_dir: Path,
    sortie_dir: Path,
    index_empreinte: Mapping[str, str],
    echecs_anciens: set[str],
) -> str | None:
    ancien_id = index_empreinte.get(empreinte_paquet_reemploi(paquet))
    if not ancien_id or ancien_id in echecs_anciens:
        return None
    if not (ancien_dir / "entites" / f"{ancien_id}.json").is_file():
        return None
    copier_reconstitution_reemployee(
        unite, ancien_id, ancien_dir=ancien_dir, sortie_dir=sortie_dir
    )
    return ancien_id


def prevoir_reemploi(
    *,
    pieces_dir: Path,
    sortie_dir: Path,
    ancien_dir: Path,
) -> dict[str, int]:
    registre = json.loads((sortie_dir / "registre.json").read_text(encoding="utf-8"))
    index = index_paquets_par_empreinte(ancien_dir)
    echecs = _echecs_unites_ancien(ancien_dir)
    corpus, _ = charger_corpus(pieces_dir)
    entites = registre.get("entites") or []
    toutes_cles = _toutes_cles_entites(entites)
    index_cles = index_cles_portees(corpus, toutes_cles)
    paquets_dir = sortie_dir / "paquets"
    reemployer = 0
    appeler = 0
    for unite in registre.get("unites") or []:
        uid = unite["id"]
        if (sortie_dir / "entites" / f"{uid}.json").is_file():
            continue
        paquet_path = paquets_dir / f"{uid}.json"
        if paquet_path.is_file():
            paquet = json.loads(paquet_path.read_text(encoding="utf-8"))
        else:
            paquet = rassembler(
                unite, corpus, toutes_cles=toutes_cles, index_cles=index_cles
            )
        ancien_id = index.get(empreinte_paquet_reemploi(paquet))
        if (
            ancien_id
            and ancien_id not in echecs
            and (ancien_dir / "entites" / f"{ancien_id}.json").is_file()
        ):
            reemployer += 1
        else:
            appeler += 1
    return {"reemployer": reemployer, "appeler": appeler}


def _empreinte_section_entite(paquet: Mapping[str, Any]) -> str:
    return json.dumps(
        paquet.get("sections", {}).get("entite") or {},
        ensure_ascii=False,
        sort_keys=True,
    )


def _paires_entite_identiques(paquets: Mapping[str, Mapping[str, Any]]) -> int:
    uids = sorted(paquets)
    total = 0
    for i, a in enumerate(uids):
        empreinte_a = _empreinte_section_entite(paquets[a])
        for b in uids[i + 1 :]:
            if empreinte_a == _empreinte_section_entite(paquets[b]):
                total += 1
    return total


def compter_dossier(
    *,
    pieces_dir: Path,
    sortie_dir: Path,
) -> dict[str, Any]:
    registre_path = sortie_dir / "registre.json"
    if not registre_path.is_file():
        raise FileNotFoundError("registre.json absent — lancer recenser d'abord")
    registre = json.loads(registre_path.read_text(encoding="utf-8"))
    unites = registre.get("unites") or []
    entites = registre.get("entites") or []
    toutes_cles = _toutes_cles_entites(entites)
    corpus, _ = charger_corpus(pieces_dir)
    index_cles = index_cles_portees(corpus, toutes_cles)
    paquets_dir = sortie_dir / "paquets"
    detail: dict[str, Any] = {}
    paquets: dict[str, dict[str, Any]] = {}
    appels_restants = 0
    tailles: list[int] = []
    tronquees = 0
    for unite in unites:
        uid = unite["id"]
        paquet_path = paquets_dir / f"{uid}.json"
        entree: dict[str, Any] = {"id": uid, "periode": list(unite.get("periode") or [])}
        if paquet_path.is_file():
            paquet = json.loads(paquet_path.read_text(encoding="utf-8"))
        else:
            paquet = rassembler(
                unite, corpus, toutes_cles=toutes_cles, index_cles=index_cles
            )
        paquets[uid] = paquet
        invite = presenter(paquet) + "\n\n" + INVITE
        entree["invite_octets"] = len(invite.encode("utf-8"))
        entree["paquet_caracteres"] = sum(
            t.get("caracteres", 0) for t in paquet.get("taille", {}).values()
        )
        entree["sections"] = {
            nom: {
                "lignes": paquet.get("taille", {}).get(nom, {}).get("lignes", 0),
                "caracteres": paquet.get("taille", {}).get(nom, {}).get("caracteres", 0),
            }
            for nom in ("document", "entite", "variantes", "emergent")
        }
        entree["tronque"] = bool(paquet.get("tronque"))
        if entree["tronque"]:
            tronquees += 1
        sortie = sortie_dir / "entites" / f"{uid}.json"
        restant = 0 if sortie.is_file() else 1
        entree["appels_restants"] = restant
        appels_restants += restant
        detail[uid] = entree
        tailles.append(entree["paquet_caracteres"])
    types = Counter((unite.get("type") or "autre") for unite in unites)
    reste = reliquat(
        registre,
        corpus,
        toutes_cles=toutes_cles,
        index_cles=index_cles,
        paquets=paquets,
    )
    lots = decouper_lots_reliquat(reste)
    par_piece = {
        piece: {
            "lignes": sum(
                1 for item in lignes.values() if item.get("motif") != "rappel"
            ),
            "caracteres": sum(
                len(item.get("texte") or "")
                for item in lignes.values()
                if item.get("motif") != "rappel"
            ),
        }
        for piece, lignes in reste["par_piece"].items()
    }
    return {
        "unites": len(unites),
        "unites_par_type": dict(sorted(types.items())),
        "tailles": _stats_tailles(tailles),
        "unites_tronquees": tronquees,
        "paires_entite_identiques": _paires_entite_identiques(paquets),
        "appels_restants": appels_restants,
        "detail": detail,
        "reliquat": {
            "lignes": reste["lignes"],
            "caracteres": reste["caracteres"],
            "par_piece": par_piece,
            "lots": len(lots),
        },
    }


def bilan(sortie_dir: Path) -> dict[str, Any]:
    entites_dir = sortie_dir / "entites"
    if not entites_dir.is_dir():
        return {
            "faits": 0,
            "comparaisons": {},
            "identites": {},
            "regles": 0,
            "non_etabli": 0,
            "rejets": 0,
            "reparations": 0,
            "echecs": [],
        }
    comparaisons: dict[str, int] = {}
    identites: dict[str, int] = {}
    total_faits = 0
    total_regles = 0
    total_non_etabli = 0
    total_rejets = 0
    total_reparations = 0
    for path in sorted(entites_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        faits = payload.get("faits") or []
        total_faits += len(faits)
        for fait in faits:
            comp = fait.get("comparaison") or "inconnu"
            comparaisons[comp] = comparaisons.get(comp, 0) + 1
        for identite in payload.get("identites") or []:
            jug = identite.get("jugement") or "inconnu"
            identites[jug] = identites.get(jug, 0) + 1
        total_regles += len(payload.get("regles") or [])
        total_non_etabli += len(payload.get("non_etabli") or [])
        total_rejets += len(payload.get("rejets") or [])
        total_reparations += int(payload.get("reparations") or 0)
    echecs_path = sortie_dir / "echecs.json"
    echecs = (
        json.loads(echecs_path.read_text(encoding="utf-8"))
        if echecs_path.is_file()
        else []
    )
    return {
        "faits": total_faits,
        "comparaisons": comparaisons,
        "identites": identites,
        "regles": total_regles,
        "non_etabli": total_non_etabli,
        "rejets": total_rejets,
        "reparations": total_reparations,
        "echecs": echecs,
    }
