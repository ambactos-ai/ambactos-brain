# SPDX-License-Identifier: AGPL-3.0-only
"""Pages Markdown par entité — état compilé et chronologie depuis la reconstitution."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from banc_realite.etiquetage_ontologie import _entites_actives_index
from banc_realite.reconstitution_entites import _identifiant_unite


def _format_lignes(lignes: list[Any]) -> str:
    nums = sorted({int(x) for x in lignes if x is not None})
    if not nums:
        return ""
    return "l. " + ", ".join(str(n) for n in nums)


def _format_citation(piece: str, lignes: list[Any]) -> str:
    if not piece:
        return ""
    fmt = _format_lignes(lignes)
    if fmt:
        return f"({piece}:{fmt})"
    return f"({piece})"


def _format_periode(periode: list[str]) -> str:
    if not periode:
        return "Sans période"
    if len(periode) == 1:
        return periode[0]
    return f"{periode[0]} → {periode[-1]}"


def _entite_slug(entite: Mapping[str, Any]) -> str:
    return _identifiant_unite(
        {
            "nom": entite.get("nom"),
            "type": entite.get("type") or "autre",
            "cles": entite.get("cles") or [],
        },
        [],
    )


def charger_reconstitution(reconstitution_dir: Path) -> dict[str, Any]:
    base = Path(reconstitution_dir)
    registre = json.loads((base / "registre.json").read_text(encoding="utf-8"))
    echecs_path = base / "echecs.json"
    echecs = (
        json.loads(echecs_path.read_text(encoding="utf-8")) if echecs_path.is_file() else []
    )
    reemploi_path = base / "reemploi.json"
    reemploi = (
        json.loads(reemploi_path.read_text(encoding="utf-8")) if reemploi_path.is_file() else {}
    )
    objets: dict[str, dict[str, Any]] = {}
    entites_dir = base / "entites"
    if entites_dir.is_dir():
        for path in sorted(entites_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            uid = str(payload.get("unite") or path.stem)
            objets[uid] = payload
    return {
        "registre": registre,
        "objets": objets,
        "echecs": echecs,
        "reemploi": reemploi,
    }


def regrouper_par_entite(
    registre: Mapping[str, Any],
    objets: Mapping[str, Mapping[str, Any]],
    echecs: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    echecs_map = {str(e.get("unite")): str(e.get("cause") or "") for e in echecs}
    par_nom: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"unites": [], "objets": {}, "echecs": {}}
    )
    for unite in registre.get("unites") or []:
        if not isinstance(unite, dict):
            continue
        nom = str(unite.get("entite") or "")
        if not nom:
            continue
        uid = str(unite.get("id") or "")
        bucket = par_nom[nom]
        bucket["unites"].append(unite)
        if uid in objets:
            bucket["objets"][uid] = dict(objets[uid])
        if uid in echecs_map:
            bucket["echecs"][uid] = echecs_map[uid]
    for nom in par_nom:
        par_nom[nom]["unites"].sort(key=lambda u: (u.get("periode") or [], str(u.get("id") or "")))
    return dict(par_nom)


def _versions_triees(versions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(v) for v in versions if isinstance(v, Mapping)],
        key=lambda v: (str(v.get("piece") or ""), str(v.get("valeur") or "")),
    )


def _valeurs_uniques(versions: list[Mapping[str, Any]]) -> set[str]:
    return {str(v.get("valeur") or "") for v in versions if isinstance(v, Mapping)}


def _fusionner_faits(objets: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    par_quoi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in objets.values():
        for fait in payload.get("faits") or []:
            if not isinstance(fait, dict):
                continue
            quoi = str(fait.get("quoi") or "")
            if not quoi:
                continue
            for version in fait.get("versions") or []:
                if isinstance(version, dict):
                    par_quoi[quoi].append(version)
    return dict(par_quoi)


def _ligne_fait(quoi: str, version: Mapping[str, Any]) -> str:
    piece = str(version.get("piece") or "")
    valeur = str(version.get("valeur") or "")
    lignes = list(version.get("lignes") or [])
    return f"**{quoi}** : {valeur} {_format_citation(piece, lignes)}".rstrip()


def _cle_regle(regle: Mapping[str, Any]) -> tuple[str, str, str, tuple[int, ...]]:
    return (
        str(regle.get("enonce") or ""),
        str(regle.get("s_applique_a") or ""),
        str(regle.get("piece") or ""),
        tuple(sorted(int(x) for x in regle.get("lignes") or [])),
    )


def _fusionner_regles(objets: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    vus: set[tuple[str, str, str, tuple[int, ...]]] = set()
    regles: list[dict[str, Any]] = []
    for payload in objets.values():
        for regle in payload.get("regles") or []:
            if not isinstance(regle, dict):
                continue
            key = _cle_regle(regle)
            if key in vus:
                continue
            vus.add(key)
            regles.append(dict(regle))
    regles.sort(
        key=lambda r: (
            str(r.get("enonce") or ""),
            str(r.get("piece") or ""),
            tuple(r.get("lignes") or []),
        )
    )
    return regles


def _fusionner_non_etabli(objets: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    vus: set[tuple[str, str]] = set()
    points: list[dict[str, Any]] = []
    for payload in objets.values():
        for item in payload.get("non_etabli") or []:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("point") or ""), str(item.get("ce_qui_le_resoudrait") or ""))
            if key in vus:
                continue
            vus.add(key)
            points.append(dict(item))
    points.sort(key=lambda p: (str(p.get("point") or ""), str(p.get("ce_qui_le_resoudrait") or "")))
    return points


def _fusionner_identites(objets: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    vus: set[tuple[str, str, str, str, tuple[int, ...]]] = set()
    identites: list[dict[str, Any]] = []
    for payload in objets.values():
        for ident in payload.get("identites") or []:
            if not isinstance(ident, dict):
                continue
            key = (
                str(ident.get("variante") or ""),
                str(ident.get("jugement") or ""),
                str(ident.get("motif") or ""),
                str(ident.get("piece") or ""),
                tuple(sorted(int(x) for x in ident.get("lignes") or [])),
            )
            if key in vus:
                continue
            vus.add(key)
            identites.append(dict(ident))
    identites.sort(
        key=lambda i: (
            str(i.get("variante") or ""),
            str(i.get("piece") or ""),
            tuple(i.get("lignes") or []),
        )
    )
    return identites


def _collecter_pieces(
    objets: Mapping[str, Mapping[str, Any]],
    relations: list[Mapping[str, Any]],
) -> dict[str, list[int]]:
    par_piece: dict[str, set[int]] = defaultdict(set)

    def ajouter(piece: str, lignes: list[Any]) -> None:
        if not piece:
            return
        for ligne in lignes or []:
            if ligne is not None:
                par_piece[piece].add(int(ligne))

    for payload in objets.values():
        for fait in payload.get("faits") or []:
            if not isinstance(fait, dict):
                continue
            for version in fait.get("versions") or []:
                if isinstance(version, dict):
                    ajouter(str(version.get("piece") or ""), list(version.get("lignes") or []))
        for regle in payload.get("regles") or []:
            if isinstance(regle, dict):
                ajouter(str(regle.get("piece") or ""), list(regle.get("lignes") or []))
        for ident in payload.get("identites") or []:
            if isinstance(ident, dict):
                ajouter(str(ident.get("piece") or ""), list(ident.get("lignes") or []))
    for relation in relations:
        ex = relation.get("exemple") or {}
        if isinstance(ex, dict):
            ajouter(str(ex.get("piece") or ""), list(ex.get("lignes") or []))
    return {p: sorted(nums) for p, nums in sorted(par_piece.items())}


def _relations_entite(nom: str, relations: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    retenues: list[dict[str, Any]] = []
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        ex = relation.get("exemple") or {}
        if not isinstance(ex, dict):
            continue
        de = ex.get("de")
        vers = ex.get("vers")
        if de != nom and vers != nom:
            continue
        retenues.append(dict(relation))
    retenues.sort(
        key=lambda r: (
            str(r.get("verbe") or ""),
            str((r.get("exemple") or {}).get("de") or ""),
            str((r.get("exemple") or {}).get("vers") or ""),
        )
    )
    return retenues


def _cause_courte(cause: str) -> str:
    texte = cause.strip()
    if texte.startswith("ValueError(") and texte.endswith(")"):
        texte = texte[len("ValueError(") : -1].strip("'\"")
    if texte.startswith("RuntimeError(") and texte.endswith(")"):
        texte = texte[len("RuntimeError(") : -1].strip("'\"")
    if len(texte) > 120:
        return texte[:117] + "..."
    return texte


def _ligne_relation(nom: str, relation: Mapping[str, Any]) -> str:
    verbe = str(relation.get("verbe") or "")
    ex = relation.get("exemple") or {}
    de = str(ex.get("de") or "")
    vers = str(ex.get("vers") or "")
    autre = vers if de == nom else de
    occ = int(relation.get("occurrences") or 0)
    piece = str(ex.get("piece") or "")
    cite = _format_citation(piece, list(ex.get("lignes") or []))
    inner = cite[1:-1] if cite.startswith("(") and cite.endswith(")") else cite
    return f"- {verbe} → {autre} ({occ} occurrences, ex. {inner})"


def _n_perimees(unites: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for u in unites if isinstance(u.get("perimee"), Mapping))


def rendre_page(
    entite: Mapping[str, Any],
    groupe: Mapping[str, Any],
    relations_ontologie: list[Mapping[str, Any]],
    version: int | str,
    date_gen: str,
    *,
    regime: str | None = None,
    date_reconstitution: str | None = None,
) -> str:
    nom = str(entite.get("nom") or "")
    typ = str(entite.get("type") or "autre")
    unites = list(groupe.get("unites") or [])
    objets: dict[str, Mapping[str, Any]] = dict(groupe.get("objets") or {})
    echecs_unite: dict[str, str] = dict(groupe.get("echecs") or {})
    n_perimees = _n_perimees(unites)
    date_etat = date_reconstitution or date_gen

    faits_fusionnes = _fusionner_faits(objets)
    regles_globales = _fusionner_regles(objets)
    cles_regles_globales = {_cle_regle(r) for r in regles_globales}
    non_etabli_global = _fusionner_non_etabli(objets)
    cles_non_global = {
        (str(g.get("point") or ""), str(g.get("ce_qui_le_resoudrait") or ""))
        for g in non_etabli_global
    }
    identites = _fusionner_identites(objets)
    relations = _relations_entite(nom, relations_ontologie)
    pieces_map = _collecter_pieces(objets, relations)

    convergents: list[str] = []
    desaccords: list[tuple[str, list[dict[str, Any]]]] = []
    for quoi in sorted(faits_fusionnes):
        versions = _versions_triees(faits_fusionnes[quoi])
        if len(_valeurs_uniques(versions)) <= 1:
            if versions:
                convergents.append(_ligne_fait(quoi, versions[0]))
        else:
            desaccords.append((quoi, versions))

    lignes: list[str] = [
        f"# {nom}",
        "",
        (
            f"type : {typ} · ontologie v{version} · {len(unites)} unités · "
            f"{len(pieces_map)} pièces citées · généré le {date_gen}"
        ),
        "",
    ]
    if n_perimees:
        lignes.append(
            f"État au {date_etat} · {n_perimees} unité(s) périmée(s)"
        )
        lignes.append("")
    if regime == "jamais":
        lignes.append("régime : jamais (pièces qui font foi seules)")
        lignes.append("")
    lignes.extend(
        [
            "## État compilé",
            "",
        ]
    )
    if convergents:
        lignes.extend(convergents)
        lignes.append("")
    if regles_globales:
        lignes.append("### Règles")
        for regle in regles_globales:
            piece = str(regle.get("piece") or "")
            lignes.append(
                f"- {regle.get('enonce')} — s'applique à {regle.get('s_applique_a')} "
                f"{_format_citation(piece, list(regle.get('lignes') or []))}".rstrip()
            )
        lignes.append("")
    if non_etabli_global:
        lignes.append("### Non établi")
        for item in non_etabli_global:
            lignes.append(
                f"- {item.get('point')} — le résoudrait : {item.get('ce_qui_le_resoudrait')}"
            )
        lignes.append("")

    if desaccords:
        lignes.append("## Désaccords")
        for quoi, versions in desaccords:
            lignes.append(f"### {quoi}")
            lignes.append("| pièce | valeur | statut | à la date | lignes |")
            lignes.append("|---|---|---|---|---|")
            for version in versions:
                a_la_date = version.get("a_la_date")
                date_cell = "" if a_la_date is None else str(a_la_date)
                fmt_l = _format_lignes(list(version.get("lignes") or []))
                valeur_cell = str(version.get("valeur")).replace("|", "\\|")
                lignes.append(
                    f"| {version.get('piece')} | {valeur_cell} | "
                    f"{version.get('statut')} | {date_cell} | {fmt_l} |"
                )
            lignes.append("")

    lignes.append("## Chronologie")
    for unite in unites:
        uid = str(unite.get("id") or "")
        periode = [str(p) for p in (unite.get("periode") or []) if p]
        titre = str(unite.get("titre") or nom)
        periode_label = _format_periode(periode)
        lignes.append(f"### {periode_label} — {titre}")
        if uid in echecs_unite:
            lignes.append(f"non reconstituée : {_cause_courte(echecs_unite[uid])}")
            lignes.append("")
            continue
        payload = objets.get(uid)
        if not payload:
            lignes.append("")
            continue
        for fait in payload.get("faits") or []:
            if not isinstance(fait, dict):
                continue
            quoi = str(fait.get("quoi") or "")
            versions = _versions_triees(list(fait.get("versions") or []))
            if len(_valeurs_uniques(versions)) <= 1:
                if versions:
                    lignes.append(_ligne_fait(quoi, versions[0]))
            else:
                for version in versions:
                    lignes.append(_ligne_fait(quoi, version))
        regles_u = [
            r
            for r in payload.get("regles") or []
            if isinstance(r, dict) and _cle_regle(r) not in cles_regles_globales
        ]
        if regles_u:
            lignes.append("### Règles")
            for regle in sorted(
                regles_u,
                key=lambda r: (str(r.get("enonce") or ""), str(r.get("piece") or "")),
            ):
                piece = str(regle.get("piece") or "")
                lignes.append(
                    f"- {regle.get('enonce')} — s'applique à {regle.get('s_applique_a')} "
                    f"{_format_citation(piece, list(regle.get('lignes') or []))}".rstrip()
                )
        non_u = [
            n
            for n in payload.get("non_etabli") or []
            if isinstance(n, dict)
            and (str(n.get("point") or ""), str(n.get("ce_qui_le_resoudrait") or ""))
            not in cles_non_global
        ]
        if non_u:
            lignes.append("### Non établi")
            for item in sorted(
                non_u,
                key=lambda p: (str(p.get("point") or ""), str(p.get("ce_qui_le_resoudrait") or "")),
            ):
                lignes.append(
                    f"- {item.get('point')} — le résoudrait : {item.get('ce_qui_le_resoudrait')}"
                )
        lignes.append("")

    if identites:
        lignes.append("## Identités")
        for ident in identites:
            piece = str(ident.get("piece") or "")
            lignes.append(
                f"- {ident.get('variante')} — {ident.get('jugement')} : {ident.get('motif')} "
                f"{_format_citation(piece, list(ident.get('lignes') or []))}".rstrip()
            )
        lignes.append("")

    if pieces_map:
        lignes.append("## Pièces citées")
        for piece, nums in pieces_map.items():
            fmt = _format_lignes(nums)
            if fmt:
                lignes.append(f"- {piece} (lignes : {fmt})")
            else:
                lignes.append(f"- {piece}")
        lignes.append("")

    if relations:
        lignes.append("## Entités liées")
        for relation in relations:
            lignes.append(_ligne_relation(nom, relation))
        lignes.append("")

    while lignes and lignes[-1] == "":
        lignes.pop()
    return "\n".join(lignes) + "\n"


def _stats_page(groupe: Mapping[str, Any]) -> tuple[int, int, int]:
    objets = dict(groupe.get("objets") or {})
    faits_fusionnes = _fusionner_faits(objets)
    n_faits = len(faits_fusionnes)
    n_desaccords = sum(
        1 for quoi in faits_fusionnes if len(_valeurs_uniques(faits_fusionnes[quoi])) > 1
    )
    n_non = len(_fusionner_non_etabli(objets))
    return n_faits, n_desaccords, n_non


def rendre_index(
    entites_actives: Mapping[str, Mapping[str, Any]],
    groupes: Mapping[str, Mapping[str, Any]],
    echecs: list[Mapping[str, Any]],
    version: int | str,
    date_gen: str,
) -> str:
    lignes = [
        "# Index des entités",
        "",
        f"Ontologie v{version} · généré le {date_gen}",
        "",
        "| entité | type | unités | faits | désaccords | non établi | périmées |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    entrees: list[tuple[str, str, str, int, int, int, int, int]] = []
    for nom, meta in entites_actives.items():
        typ = str(meta.get("type") or "autre")
        groupe = groupes.get(nom) or {"unites": [], "objets": {}}
        n_unites = len(groupe.get("unites") or [])
        n_faits, n_des, n_non = _stats_page(groupe)
        n_perimees = _n_perimees(list(groupe.get("unites") or []))
        slug = _entite_slug({"nom": nom, "type": typ, "cles": meta.get("cles") or []})
        entrees.append((typ, nom, slug, n_unites, n_faits, n_des, n_non, n_perimees))
    for typ, nom, slug, n_unites, n_faits, n_des, n_non, n_perimees in sorted(
        entrees, key=lambda e: (e[0], e[1])
    ):
        lignes.append(
            f"| [{nom}]({slug}.md) | {typ} | {n_unites} | {n_faits} | {n_des} | {n_non} | {n_perimees} |"
        )
    lignes.append("")
    if echecs:
        lignes.append("## Unités échouées")
        for item in sorted(echecs, key=lambda e: str(e.get("unite") or "")):
            lignes.append(f"- {item.get('unite')} : {_cause_courte(str(item.get('cause') or ''))}")
        lignes.append("")
    return "\n".join(lignes)


def generer_pages(
    reconstitution_dir: Path,
    ontologie: Mapping[str, Any],
    date_gen: str | None = None,
    *,
    entites: Sequence[str] | None = None,
    regime_par_entite: Mapping[str, str] | None = None,
    date_reconstitution: str | None = None,
    registre: Mapping[str, Any] | None = None,
    echecs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    date_effective = date_gen or date.today().isoformat()
    version = ontologie.get("version", 1)
    charge = charger_reconstitution(reconstitution_dir)
    registre_eff: Mapping[str, Any] = (
        charge["registre"] if registre is None else registre
    )
    echecs_eff: list[Mapping[str, Any]] = (
        list(charge["echecs"]) if echecs is None else list(echecs)
    )
    groupes = regrouper_par_entite(registre_eff, charge["objets"], echecs_eff)
    actives = _entites_actives_index(ontologie)
    relations = [r for r in (ontologie.get("relations") or []) if isinstance(r, dict)]
    pages_dir = Path(reconstitution_dir) / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cibles = set(entites) if entites is not None else None
    regimes = regime_par_entite or {}

    entites_sans_unite: list[str] = []
    n_pages = 0
    for nom in sorted(actives, key=lambda n: (actives[n].get("type") or "", n)):
        meta = actives[nom]
        groupe = groupes.get(nom) or {"unites": [], "objets": {}, "echecs": {}}
        if not groupe.get("unites"):
            entites_sans_unite.append(nom)
        if cibles is not None and nom not in cibles:
            continue
        entite = {"nom": nom, "type": meta.get("type"), "cles": meta.get("cles") or []}
        slug = _entite_slug(entite)
        contenu = rendre_page(
            entite,
            groupe,
            relations,
            version,
            date_effective,
            regime=regimes.get(nom),
            date_reconstitution=date_reconstitution,
        )
        (pages_dir / f"{slug}.md").write_text(contenu, encoding="utf-8")
        n_pages += 1

    index = rendre_index(actives, groupes, echecs_eff, version, date_effective)
    (pages_dir / "index.md").write_text(index, encoding="utf-8")

    return {
        "pages": n_pages,
        "entites_sans_unite": sorted(entites_sans_unite),
        "unites_echouees": len(echecs_eff),
    }
