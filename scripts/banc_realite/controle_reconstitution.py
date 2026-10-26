# SPDX-License-Identifier: AGPL-3.0-only
"""Contrôles déterministes d'une reconstitution par entité — sans lecture de l'or."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from .reconstitution_entites import SEP_TABLEAU, charger_corpus

DATE_JOUR = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
DATE_MOIS = re.compile(r"(?<!\d)\d{4}-\d{2}(?!\d)")
ID_PATTERN = re.compile(r"[A-Z]{2,}(?:-[A-Z0-9]+)+")
NOMBRE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"-?(?:\d{1,3}(?:[\u00a0\u202f ]\d{3})+|\d+)(?:[.,]\d+)?"
    r"(?![A-Za-z0-9])"
)
MOT_SIGNIFICATIF = re.compile(r"[A-Za-zÀ-ÿ]{3,}")
ESPACES_MILLIERS = re.compile(r"[\u00a0\u202f ]")

NOMS_CONTROLES = (
    "aucune_ligne_oubliee",
    "chaque_unite_reconstituee",
    "lettre_est_celle_de_la_piece",
    "concordant_est_vrai",
    "desaccord_est_vrai",
    "valeur_dans_la_lettre",
    "rien_du_paquet_laisse_de_cote",
    "desaccords_possibles_non_releves",
)

BLOQUANTS = {
    "aucune_ligne_oubliee",
    "chaque_unite_reconstituee",
    "lettre_est_celle_de_la_piece",
    "concordant_est_vrai",
    "desaccord_est_vrai",
    "valeur_dans_la_lettre",
    "rien_du_paquet_laisse_de_cote",
}


def normaliser_nombre(raw: str) -> str:
    s = raw.strip()
    signe = ""
    if s.startswith("-"):
        signe = "-"
        s = s[1:]
    s = ESPACES_MILLIERS.sub("", s)
    if "," in s and "." not in s:
        s = s.replace(",", ".", 1)
    return f"{signe}{s}"


def forme_comparaison(nombre: str) -> str:
    """Forme de comparaison : zéros de fin de partie décimale ignorés."""
    s = nombre.strip()
    signe = ""
    if s.startswith("-"):
        signe = "-"
        s = s[1:]
    if "." in s:
        entier, frac = s.split(".", 1)
        frac = frac.rstrip("0")
        entier = entier.lstrip("0") or "0"
        if not frac:
            return f"{signe}{entier}"
        return f"{signe}{entier}.{frac}"
    return f"{signe}{s.lstrip('0') or '0'}"


def _lectures_nombre(brut: str) -> tuple[frozenset[str], ...]:
    signe = ""
    s = brut
    if s.startswith("-"):
        signe = "-"
        s = s[1:]
    compact = ESPACES_MILLIERS.sub("", s)
    if "," in compact and "." not in compact:
        gauche, droite = compact.split(",", 1)
        decimal = f"{signe}{gauche}.{droite}"
        return (frozenset({decimal}), frozenset({f"{signe}{gauche}", droite}))
    return (frozenset({f"{signe}{compact}"}),)


def _extraire_masquer(texte: str, pattern: re.Pattern[str]) -> tuple[list[str], str]:
    trouves: list[str] = []
    chars = list(texte)
    for match in pattern.finditer(texte):
        trouves.append(match.group(0))
        for i in range(match.start(), match.end()):
            chars[i] = " "
    return trouves, "".join(chars)


def extraire_spans(texte: str) -> list[tuple[str, tuple[frozenset[str], ...]]]:
    """Dates, puis identifiants, puis nombres — chaque portion reconnue est retirée."""
    if not texte:
        return []
    spans: list[tuple[str, tuple[frozenset[str], ...]]] = []
    restant = texte
    dates_jour, restant = _extraire_masquer(restant, DATE_JOUR)
    dates_mois, restant = _extraire_masquer(restant, DATE_MOIS)
    identifiants, restant = _extraire_masquer(restant, ID_PATTERN)
    nombres, _ = _extraire_masquer(restant, NOMBRE_RE)
    for date in dates_jour + dates_mois:
        spans.append(("date", (frozenset({date}),)))
    for ident in identifiants:
        spans.append(("identifiant", (frozenset({ident}),)))
    for brut in nombres:
        spans.append(("nombre", _lectures_nombre(brut)))
    return spans


def jetons(texte: str) -> set[str]:
    result: set[str] = set()
    for _, lectures in extraire_spans(texte or ""):
        for lecture in lectures:
            result |= set(lecture)
    return result


def jetons_comparaison(texte: str) -> set[str]:
    result: set[str] = set()
    for kind, lectures in extraire_spans(texte or ""):
        for lecture in lectures:
            if kind == "nombre":
                result.update(forme_comparaison(tok) for tok in lecture)
            else:
                result |= set(lecture)
    return result


def extraire_nombres(texte: str) -> set[str]:
    result: set[str] = set()
    for kind, lectures in extraire_spans(texte or ""):
        if kind != "nombre":
            continue
        for lecture in lectures:
            result |= set(lecture)
    return result


def extraire_identifiants(texte: str) -> set[str]:
    return {tok for kind, lectures in extraire_spans(texte or "") if kind == "identifiant" for lecture in lectures for tok in lecture}


def _ancres(texte: str) -> set[str]:
    return {
        tok
        for kind, lectures in extraire_spans(texte or "")
        if kind in {"date", "identifiant"}
        for lecture in lectures
        for tok in lecture
    }


def _nombres_comparaison(texte: str) -> set[str]:
    return {
        forme_comparaison(tok)
        for kind, lectures in extraire_spans(texte or "")
        if kind == "nombre"
        for lecture in lectures
        for tok in lecture
    }


def _normaliser_texte(texte: str) -> str:
    return re.sub(r"\s+", " ", (texte or "").strip().lower())


def _reduire_espaces(texte: str) -> str:
    return re.sub(r"\s+", " ", (texte or "").strip())


def _resultat_controle(
    *,
    verifie: int,
    ecarts: list[dict[str, Any]],
    non_verifiables: int = 0,
    verdict: str | None = None,
    passes: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if passes is None:
        passes = verifie - len(ecarts)
    if verdict is None:
        verdict = "ok" if not ecarts else "ecart"
    payload: dict[str, Any] = {
        "verifie": verifie,
        "passes": passes,
        "ecarts": ecarts,
        "non_verifiables": non_verifiables,
        "verdict": verdict,
    }
    payload.update(extra)
    return payload


def _lignes_non_vides(
    corpus: dict[str, list[str]],
    pieces_exclues: set[str],
) -> list[tuple[str, int, str]]:
    lignes: list[tuple[str, int, str]] = []
    for piece, contenu in sorted(corpus.items()):
        if piece in pieces_exclues:
            continue
        for num, ligne in enumerate(contenu, start=1):
            if not ligne.strip():
                continue
            if SEP_TABLEAU.match(ligne):
                continue
            lignes.append((piece, num, ligne))
    return lignes


def _lignes_paquet_sections(
    paquet: dict[str, Any],
    *,
    sections: tuple[str, ...] = ("entite", "document", "variantes", "emergent"),
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for section in sections:
        contenu = (paquet.get("sections") or {}).get(section) or {}
        for piece, lignes in contenu.items():
            result.setdefault(piece, set()).update(int(n) for n in lignes)
    return result


def _lignes_entite_document(paquet: dict[str, Any]) -> dict[str, set[int]]:
    return _lignes_paquet_sections(paquet, sections=("entite", "document"))


def _lignes_couvertes_par_paquets(paquets: dict[str, dict[str, Any]]) -> set[tuple[str, int]]:
    couvert: set[tuple[str, int]] = set()
    for paquet in paquets.values():
        for piece, nums in _lignes_entite_document(paquet).items():
            for num in nums:
                couvert.add((piece, num))
    return couvert


def _lire_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _charger_paquets(paquets_dir: Path, uids: list[str]) -> dict[str, dict[str, Any]]:
    paquets: dict[str, dict[str, Any]] = {}
    for uid in uids:
        path = paquets_dir / f"{uid}.json"
        if not path.is_file():
            continue
        payload, _ = _lire_json(path)
        if isinstance(payload, dict):
            paquets[uid] = payload
    return paquets


def _charger_entites(entites_dir: Path, uids: list[str]) -> dict[str, dict[str, Any]]:
    entites: dict[str, dict[str, Any]] = {}
    for uid in uids:
        path = entites_dir / f"{uid}.json"
        if not path.is_file():
            continue
        payload, _ = _lire_json(path)
        if isinstance(payload, dict):
            entites[uid] = payload
    return entites


def _ecartes_unites(ecartes_dir: Path) -> set[str]:
    if not ecartes_dir.is_dir():
        return set()
    return {path.stem.replace(".brut", "") for path in ecartes_dir.glob("*.brut.txt")}


def _objets_avec_lignes(entite: dict[str, Any]) -> list[dict[str, Any]]:
    objets: list[dict[str, Any]] = []
    for fait in entite.get("faits") or []:
        for version in fait.get("versions") or []:
            objets.append(version)
    objets.extend(entite.get("regles") or [])
    objets.extend(entite.get("identites") or [])
    return objets


def _lettre_attendue(corpus: dict[str, list[str]], piece: str, nums: list[int]) -> str:
    return "\n".join(corpus[piece][n - 1] for n in sorted(nums))


def _controle_aucune_ligne_oubliee(
    corpus: dict[str, list[str]],
    pieces_exclues: set[str],
    paquets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    attendues = _lignes_non_vides(corpus, pieces_exclues)
    couvert = _lignes_couvertes_par_paquets(paquets)
    ecarts = [
        {"piece": piece, "ligne": num, "texte": texte}
        for piece, num, texte in attendues
        if (piece, num) not in couvert
    ]
    return _resultat_controle(verifie=len(attendues), ecarts=ecarts)


def _controle_chaque_unite_reconstituee(
    uids: list[str],
    entites: dict[str, dict[str, Any]],
    ecartes: set[str],
) -> dict[str, Any]:
    absentes = [uid for uid in uids if uid not in entites]
    reponses_ecartees = sorted(uid for uid in uids if uid in ecartes)
    ecarts = [{"unite": uid, "motif": "fichier absent"} for uid in absentes]
    ecarts.extend({"unite": uid, "motif": "reponse ecartee"} for uid in reponses_ecartees)
    return _resultat_controle(verifie=len(uids), ecarts=ecarts)


def _iter_objets_avec_contexte(entite: dict[str, Any]):
    for fait in entite.get("faits") or []:
        for version in fait.get("versions") or []:
            yield "version", version
    for regle in entite.get("regles") or []:
        yield "regle", regle
    for identite in entite.get("identites") or []:
        yield "identite", identite


def _controle_lettre_est_celle_de_la_piece(
    corpus: dict[str, list[str]],
    paquets: dict[str, dict[str, Any]],
    entites: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ecarts: list[dict[str, Any]] = []
    verifie = 0
    for uid, entite in entites.items():
        paquet = paquets.get(uid)
        if paquet is None:
            continue
        lignes_paquet = _lignes_paquet_sections(paquet)
        for kind, obj in _iter_objets_avec_contexte(entite):
            verifie += 1
            piece = obj.get("piece")
            try:
                nums = sorted({int(n) for n in obj.get("lignes") or []})
            except (TypeError, ValueError):
                nums = []
            if not isinstance(piece, str) or piece not in corpus:
                ecarts.append(
                    {
                        "unite": uid,
                        "type": kind,
                        "piece": piece,
                        "motif": "piece absente",
                    }
                )
                continue
            hors_paquet = [n for n in nums if n not in lignes_paquet.get(piece, set())]
            if hors_paquet:
                ecarts.append(
                    {
                        "unite": uid,
                        "type": kind,
                        "piece": piece,
                        "lignes": hors_paquet,
                        "motif": "ligne absente du paquet",
                    }
                )
            attendue = _lettre_attendue(corpus, piece, nums) if nums else ""
            if obj.get("lettre") != attendue:
                ecarts.append(
                    {
                        "unite": uid,
                        "type": kind,
                        "piece": piece,
                        "lignes": nums,
                        "motif": "lettre incorrecte",
                        "attendue": attendue,
                        "trouvee": obj.get("lettre"),
                    }
                )
    return _resultat_controle(verifie=verifie, ecarts=ecarts)


def _deux_a_deux_disjoints(ensembles: list[set[str]]) -> bool:
    for i, a in enumerate(ensembles):
        for b in ensembles[i + 1 :]:
            if a & b:
                return False
    return True


def _ensembles_versions(versions: list[dict[str, Any]]) -> tuple[list[set[str]], list[set[str]]]:
    bruts = [jetons(str(v.get("valeur") or "")) for v in versions]
    cmps = [jetons_comparaison(str(v.get("valeur") or "")) for v in versions]
    return bruts, cmps


def _motif_concordance(fait: dict[str, Any]) -> str | None:
    versions = fait.get("versions") or []
    if len(versions) < 2:
        return None
    _, cmps = _ensembles_versions(versions)
    avec = [e for e in cmps if e]
    if not avec:
        return "texte_seul"
    uniques = [set(s) for s in {frozenset(e) for e in avec}]
    if len(uniques) > 6 and _deux_a_deux_disjoints(uniques):
        return "serie_dans_un_fait"
    if len(avec) >= 2 and not set.intersection(*avec):
        return "jetons_disjoints"
    if len(avec) < len(cmps):
        return "version_sans_jeton"
    return None


def _zeros_fin_utilise(fait: dict[str, Any]) -> bool:
    versions = fait.get("versions") or []
    bruts, cmps = _ensembles_versions(versions)
    avec_brut = [e for e in bruts if e]
    avec_cmp = [e for e in cmps if e]
    if len(avec_cmp) < 2:
        return False
    if set.intersection(*avec_cmp) and (not avec_brut or not set.intersection(*avec_brut)):
        return True
    return False


def _controle_concordant_est_vrai(entites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ecarts: list[dict[str, Any]] = []
    non_verifiables = 0
    total = 0
    zeros_fin = 0
    motifs = {
        "texte_seul": 0,
        "version_sans_jeton": 0,
        "jetons_disjoints": 0,
        "serie_dans_un_fait": 0,
    }
    for uid, entite in entites.items():
        for fait in entite.get("faits") or []:
            if fait.get("comparaison") != "concordant":
                continue
            versions = fait.get("versions") or []
            if len(versions) < 2:
                continue
            total += 1
            motif = _motif_concordance(fait)
            if motif in {"texte_seul", "version_sans_jeton"}:
                non_verifiables += 1
                motifs[motif] += 1
                continue
            if motif in {"jetons_disjoints", "serie_dans_un_fait"}:
                motifs[motif] += 1
                ecarts.append(
                    {
                        "unite": uid,
                        "quoi": fait.get("quoi"),
                        "motif": motif,
                        "valeurs": [v.get("valeur") for v in versions],
                    }
                )
                continue
            if _zeros_fin_utilise(fait):
                zeros_fin += 1
    verifie = total - non_verifiables
    return _resultat_controle(
        verifie=verifie,
        ecarts=ecarts,
        non_verifiables=non_verifiables,
        egalite_zeros_fin=zeros_fin,
        motifs=motifs,
    )


def _desaccord_tient(fait: dict[str, Any]) -> bool:
    versions = fait.get("versions") or []
    if len(versions) < 2:
        return True
    jet = [jetons_comparaison(str(v.get("valeur") or "")) for v in versions]
    if len({frozenset(e) for e in jet}) >= 2:
        return True
    statuts = [str(v.get("statut") or "") for v in versions]
    if len(set(statuts)) >= 2:
        return True
    textes = [_normaliser_texte(str(v.get("valeur") or "")) for v in versions]
    if len(set(textes)) >= 2:
        return True
    return False


def _controle_desaccord_est_vrai(entites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ecarts: list[dict[str, Any]] = []
    verifie = 0
    for uid, entite in entites.items():
        for fait in entite.get("faits") or []:
            if fait.get("comparaison") != "desaccord":
                continue
            versions = fait.get("versions") or []
            if len(versions) < 2:
                continue
            verifie += 1
            if not _desaccord_tient(fait):
                ecarts.append(
                    {
                        "unite": uid,
                        "quoi": fait.get("quoi"),
                        "motif": "desaccord_non_verifie",
                        "valeurs": [v.get("valeur") for v in versions],
                        "statuts": [v.get("statut") for v in versions],
                    }
                )
    return _resultat_controle(verifie=verifie, ecarts=ecarts)


def _recopie_exacte(valeur: str, lettre: str) -> bool:
    reduite = _reduire_espaces(valeur)
    if not reduite:
        return False
    return reduite in _reduire_espaces(lettre)


def _lecture_presente(
    kind: str,
    lecture: frozenset[str],
    lettre_jetons: set[str],
    lettre_nombres_raw: set[str],
    lettre_nombres_cmp: set[str],
) -> tuple[bool, bool]:
    zeros = False
    for tok in lecture:
        if kind == "nombre":
            if tok in lettre_nombres_raw:
                continue
            if forme_comparaison(tok) in lettre_nombres_cmp:
                zeros = True
                continue
            return False, False
        if tok not in lettre_jetons:
            return False, False
    return True, zeros


def _valeur_dans_lettre(version: dict[str, Any]) -> tuple[bool, str | None, bool, bool]:
    valeur = str(version.get("valeur") or "")
    lettre = str(version.get("lettre") or "")
    recop = _recopie_exacte(valeur, lettre)
    spans = extraire_spans(valeur)
    lettre_spans = extraire_spans(lettre)
    lettre_jetons = jetons(lettre)
    lettre_nombres_raw = {
        tok for kind, lectures in lettre_spans if kind == "nombre" for lecture in lectures for tok in lecture
    }
    lettre_nombres_cmp = {forme_comparaison(tok) for tok in lettre_nombres_raw}
    zeros = False
    for kind, lectures in spans:
        ok_span = False
        zeros_span = False
        for lecture in lectures:
            present, z = _lecture_presente(
                kind, lecture, lettre_jetons, lettre_nombres_raw, lettre_nombres_cmp
            )
            if present:
                ok_span = True
                zeros_span = z
                break
        if not ok_span:
            manquant = next(iter(lectures[0]))
            return False, manquant, recop, False
        zeros = zeros or zeros_span
    return True, None, recop, zeros


def _controle_valeur_dans_la_lettre(entites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ecarts: list[dict[str, Any]] = []
    verifie = 0
    recopies = 0
    zeros_fin = 0
    for uid, entite in entites.items():
        for fait in entite.get("faits") or []:
            for version in fait.get("versions") or []:
                verifie += 1
                ok, manquant, recop, zeros = _valeur_dans_lettre(version)
                if recop:
                    recopies += 1
                if zeros:
                    zeros_fin += 1
                if not ok:
                    ecarts.append(
                        {
                            "unite": uid,
                            "quoi": fait.get("quoi"),
                            "motif": "valeur_absente_de_la_lettre",
                            "jeton_manquant": manquant,
                            "valeur": version.get("valeur"),
                            "lettre": version.get("lettre"),
                            "recopie_exacte": recop,
                        }
                    )
    part = round(recopies / verifie, 4) if verifie else 1.0
    return _resultat_controle(
        verifie=verifie,
        ecarts=ecarts,
        recopie_exacte=recopies,
        part_recopie_exacte=part,
        egalite_zeros_fin=zeros_fin,
    )


def _controle_rien_du_paquet_laisse_de_cote(
    paquets: dict[str, dict[str, Any]],
    entites: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ecarts: list[dict[str, Any]] = []
    verifie = 0
    for uid, paquet in paquets.items():
        attendues = _lignes_entite_document(paquet)
        citees: set[tuple[str, int]] = set()
        entite = entites.get(uid) or {}
        for obj in _objets_avec_lignes(entite):
            piece = obj.get("piece")
            if not isinstance(piece, str):
                continue
            try:
                nums = {int(n) for n in obj.get("lignes") or []}
            except (TypeError, ValueError):
                continue
            for num in nums:
                citees.add((piece, num))
        for nums in attendues.values():
            verifie += len(nums)
        non_citees = [
            {"piece": piece, "ligne": num}
            for piece in sorted(attendues)
            for num in sorted(attendues[piece])
            if (piece, num) not in citees
        ]
        if non_citees:
            total = sum(len(nums) for nums in attendues.values())
            nb_citees = total - len(non_citees)
            ecarts.append(
                {
                    "unite": uid,
                    "part": round(nb_citees / total, 4) if total else 1.0,
                    "non_citees": non_citees,
                }
            )
    return _resultat_controle(verifie=verifie, ecarts=ecarts)


def _nombres_fait(fait: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for version in fait.get("versions") or []:
        result |= _nombres_comparaison(str(version.get("valeur") or ""))
    return result


def _ancres_fait(fait: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for version in fait.get("versions") or []:
        result |= _ancres(str(version.get("valeur") or ""))
    return result


def _mots_quoi(quoi: Any) -> set[str]:
    return {m.lower() for m in MOT_SIGNIFICATIF.findall(str(quoi or ""))}


def _cle_paire(uid_a: str, fait_a: dict[str, Any], uid_b: str, fait_b: dict[str, Any]) -> frozenset:
    def _cle(uid: str, fait: dict[str, Any]) -> tuple:
        valeurs = tuple(str(v.get("valeur") or "") for v in (fait.get("versions") or []))
        return (uid, str(fait.get("quoi") or ""), str(fait.get("concept") or ""), valeurs)

    return frozenset({_cle(uid_a, fait_a), _cle(uid_b, fait_b)})


def _controle_desaccords_possibles_non_releves(
    entites: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    eligibles: list[tuple[str, dict[str, Any]]] = []
    for uid, entite in entites.items():
        for fait in entite.get("faits") or []:
            if fait.get("concept") not in {"mesure", "montant"}:
                continue
            if fait.get("comparaison") in {"desaccord", "non_ordonnable"}:
                continue
            eligibles.append((uid, fait))

    paires: list[tuple[int, dict[str, Any]]] = []
    vues: set[frozenset] = set()
    for i, (uid_a, fait_a) in enumerate(eligibles):
        mots_a = _mots_quoi(fait_a.get("quoi"))
        if not mots_a:
            continue
        ancres_a = _ancres_fait(fait_a)
        nums_a = _nombres_fait(fait_a)
        for uid_b, fait_b in eligibles[i + 1 :]:
            if uid_a == uid_b:
                continue
            if fait_a.get("concept") != fait_b.get("concept"):
                continue
            ancres_b = _ancres_fait(fait_b)
            if not (ancres_a & ancres_b):
                continue
            mots_b = _mots_quoi(fait_b.get("quoi"))
            if not mots_b:
                continue
            communs = mots_a & mots_b
            if len(communs) < len(mots_a) / 2 or len(communs) < len(mots_b) / 2:
                continue
            nums_b = _nombres_fait(fait_b)
            if nums_a == nums_b:
                continue
            if nums_a & nums_b:
                continue
            cle = _cle_paire(uid_a, fait_a, uid_b, fait_b)
            if cle in vues:
                continue
            vues.add(cle)
            paires.append(
                (
                    len(communs),
                    {
                        "unite_a": uid_a,
                        "quoi_a": fait_a.get("quoi"),
                        "unite_b": uid_b,
                        "quoi_b": fait_b.get("quoi"),
                        "mots_partages": sorted(communs),
                        "nombres_a": sorted(nums_a),
                        "nombres_b": sorted(nums_b),
                        "ancres": sorted(ancres_a & ancres_b),
                    },
                )
            )
    paires.sort(key=lambda item: (-item[0], item[1]["unite_a"], item[1]["unite_b"]))
    nb_paires = len(paires)
    ecarts = [item[1] for item in paires[:30]]
    return _resultat_controle(
        verifie=len(eligibles),
        ecarts=ecarts,
        passes=len(eligibles),
        verdict="a_relire" if nb_paires else "ok",
        nb_paires=nb_paires,
    )


def controler(pieces_dir: Path, sortie_dir: Path) -> dict[str, Any]:
    corpus, exclus_liste = charger_corpus(pieces_dir)
    pieces_exclues = {item["piece"] for item in exclus_liste}
    registre_path = sortie_dir / "registre.json"
    registre, err = _lire_json(registre_path)
    if registre is None:
        raise FileNotFoundError(f"registre illisible : {registre_path} ({err})")
    uids = [u["id"] for u in registre.get("unites") or [] if u.get("id")]
    paquets = _charger_paquets(sortie_dir / "paquets", uids)
    entites = _charger_entites(sortie_dir / "entites", uids)
    ecartes = _ecartes_unites(sortie_dir / "ecartes")

    controles = {
        "aucune_ligne_oubliee": _controle_aucune_ligne_oubliee(
            corpus, pieces_exclues, paquets
        ),
        "chaque_unite_reconstituee": _controle_chaque_unite_reconstituee(
            uids, entites, ecartes
        ),
        "lettre_est_celle_de_la_piece": _controle_lettre_est_celle_de_la_piece(
            corpus, paquets, entites
        ),
        "concordant_est_vrai": _controle_concordant_est_vrai(entites),
        "desaccord_est_vrai": _controle_desaccord_est_vrai(entites),
        "valeur_dans_la_lettre": _controle_valeur_dans_la_lettre(entites),
        "rien_du_paquet_laisse_de_cote": _controle_rien_du_paquet_laisse_de_cote(
            paquets, entites
        ),
        "desaccords_possibles_non_releves": _controle_desaccords_possibles_non_releves(
            entites
        ),
    }
    bloquant_ok = all(
        controles[nom]["verdict"] == "ok" for nom in BLOQUANTS
    )
    return {
        "bloquant_ok": bloquant_ok,
        "controles": controles,
    }


def _libelle_controle(nom: str) -> str:
    return {
        "aucune_ligne_oubliee": "Aucune ligne oubliée",
        "chaque_unite_reconstituee": "Chaque unité reconstituée",
        "lettre_est_celle_de_la_piece": "La lettre est celle de la pièce",
        "concordant_est_vrai": "« Concordant » est vrai",
        "desaccord_est_vrai": "« Désaccord » est vrai",
        "valeur_dans_la_lettre": "La valeur est dans la lettre",
        "rien_du_paquet_laisse_de_cote": "Rien du paquet laissé de côté",
        "desaccords_possibles_non_releves": "Désaccords possibles non relevés",
    }[nom]


def _nb_ecarts_affiche(ctrl: dict[str, Any]) -> int:
    if "nb_paires" in ctrl:
        return int(ctrl["nb_paires"])
    return len(ctrl.get("ecarts") or [])


def _rendre_markdown(resultat: dict[str, Any]) -> str:
    lignes = [
        "# Contrôle de reconstitution",
        "",
        "| Contrôle | Vérifiés | Passés | Écarts | Non vérifiables | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for nom in NOMS_CONTROLES:
        ctrl = resultat["controles"][nom]
        lignes.append(
            "| {libelle} | {verifie} | {passes} | {ecarts} | {non_verif} | {verdict} |".format(
                libelle=_libelle_controle(nom),
                verifie=ctrl["verifie"],
                passes=ctrl.get("passes", ctrl["verifie"] - len(ctrl["ecarts"])),
                ecarts=_nb_ecarts_affiche(ctrl),
                non_verif=ctrl.get("non_verifiables", 0),
                verdict=ctrl["verdict"],
            )
        )
    valeur = resultat["controles"]["valeur_dans_la_lettre"]
    recopie = valeur.get("recopie_exacte", 0)
    verifie_v = valeur.get("verifie") or 0
    part = valeur.get("part_recopie_exacte")
    if part is None:
        part = round(recopie / verifie_v, 4) if verifie_v else 1.0
    lignes.extend(
        [
            "",
            f"Part des valeurs recopiées exactement : {part:.0%} ({recopie}/{verifie_v}).",
        ]
    )
    zeros_6 = valeur.get("egalite_zeros_fin", 0)
    zeros_4 = resultat["controles"]["concordant_est_vrai"].get("egalite_zeros_fin", 0)
    if zeros_4 or zeros_6:
        lignes.append(
            f"Égalités par zéros de fin : {zeros_4} (concordant), {zeros_6} (valeur dans la lettre)."
        )
    lignes.extend(["", "## Détail des écarts", ""])
    for nom in NOMS_CONTROLES:
        ctrl = resultat["controles"][nom]
        if not ctrl["ecarts"]:
            continue
        lignes.append(f"### {_libelle_controle(nom)}")
        if nom == "desaccords_possibles_non_releves":
            nb = ctrl.get("nb_paires", len(ctrl["ecarts"]))
            if nb > len(ctrl["ecarts"]):
                lignes.append(f"Trente premières paires ({nb} au total).")
        if nom == "rien_du_paquet_laisse_de_cote":
            for ecart in ctrl["ecarts"]:
                lignes.append(
                    f"- unité {ecart.get('unite')} : part citée {ecart.get('part')}"
                )
                for ligne in (ecart.get("non_citees") or [])[:10]:
                    lignes.append(
                        f"  - {ligne.get('piece')}:{ligne.get('ligne')}"
                    )
                reste = ecart.get("non_citees") or []
                if len(reste) > 10:
                    lignes.append(
                        f"  - … {len(reste) - 10} lignes non citées supplémentaires"
                    )
        else:
            for ecart in ctrl["ecarts"]:
                lignes.append(f"- {json.dumps(ecart, ensure_ascii=False)}")
        lignes.append("")
    return "\n".join(lignes).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contrôles déterministes de reconstitution.")
    parser.add_argument("--pieces-dir", type=Path, required=True)
    parser.add_argument("--sortie-dir", type=Path, required=True)
    parser.add_argument("--rapport", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        resultat = controler(args.pieces_dir, args.sortie_dir)
    except Exception as exc:
        print(json.dumps({"status": "arret", "erreur": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(resultat, ensure_ascii=False, indent=2))
    if args.rapport is not None:
        args.rapport.parent.mkdir(parents=True, exist_ok=True)
        args.rapport.write_text(_rendre_markdown(resultat), encoding="utf-8")
    return 0 if resultat["bloquant_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
