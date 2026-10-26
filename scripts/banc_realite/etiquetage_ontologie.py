# SPDX-License-Identifier: AGPL-3.0-only
"""Étiquetage déterministe des passages par l'ontologie tenant v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "scripts", _ROOT / "src"):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from .couche_brute_moreau import lister_passages_dossier, passage_key
from .dossiers import BANC_VAR, dossier_courant
from .lecture_drive import _regle_correspond
from .normalize import normalize_text, piece_basename
from .ontologie_tenant import _piece_de_entree

SCHEMA_ETIQUETTES = "etiquettes-passages/1.0.0"
_MIN_CLE_CHARS = 4
_FUSION_EFFET = re.compile(r"^fusion\s+(.+?)\s+dans\s+(.+?)\s*$", re.IGNORECASE)


def _chemin_ontologie(dossier: str, version: str) -> Path:
    return BANC_VAR / "dossiers" / dossier / "ontologie" / f"ontologie.{version}.json"


def _chemin_etiquettes(dossier: str, version: str) -> Path:
    return BANC_VAR / "dossiers" / dossier / "ontologie" / f"etiquettes.{version}.json"


def chemin_etiquettes_depuis_ontologie(ontologie_path: Path) -> Path:
    """``etiquettes.v1.json`` dans le répertoire du fichier ontologie."""
    return ontologie_path.parent / "etiquettes.v1.json"


def sha256_ontologie(ontologie_path: Path) -> str:
    return hashlib.sha256(ontologie_path.read_bytes()).hexdigest()


def charger_etiquettes_si_present(chemin: Path) -> dict[str, Any]:
    if not chemin.is_file():
        return {
            "schema": SCHEMA_ETIQUETTES,
            "ontologie": "v1",
            "ontologie_sha256": "",
            "passages": {},
            "stats": {},
        }
    payload = json.loads(chemin.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {
            "schema": SCHEMA_ETIQUETTES,
            "ontologie": "v1",
            "ontologie_sha256": "",
            "passages": {},
            "stats": {},
        }
    payload.setdefault("passages", {})
    payload.setdefault("ontologie_sha256", "")
    return payload


def ecrire_etiquettes_atomique(chemin: Path, payload: dict[str, Any]) -> None:
    from .lecture_drive import ecrire_json_atomique

    chemin.parent.mkdir(parents=True, exist_ok=True)
    ecrire_json_atomique(chemin, payload)


def reetiqueter(
    ontologie: dict[str, Any],
    passages: Iterable[dict[str, Any]],
    existantes: Mapping[str, Any],
    *,
    ontologie_sha256: str,
    ids_a_retirer: Iterable[str] = (),
    tout: bool = False,
) -> dict[str, Any]:
    """Ré-étiquette tout ou les passages fournis, fusionne dans le fichier existant."""
    nouvelles = etiqueter_passages(ontologie, passages)
    if tout:
        passages_map: dict[str, Any] = dict(nouvelles)
    else:
        brut = existantes.get("passages") or {}
        passages_map = dict(brut) if isinstance(brut, Mapping) else {}
        for identifiant in ids_a_retirer:
            passages_map.pop(str(identifiant), None)
        passages_map.update(nouvelles)
    return {
        "schema": SCHEMA_ETIQUETTES,
        "ontologie": str(existantes.get("ontologie") or "v1"),
        "ontologie_sha256": ontologie_sha256,
        "passages": passages_map,
        "stats": _stats_etiquetage(passages_map, _entites_actives_index(ontologie)),
    }


def _chemin_aiguillage(drive_registre: Path) -> Path:
    return drive_registre.parent / "aiguillage.json"


def _mime_depuis_nom(nom: str) -> str:
    lower = nom.lower()
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith((".png", ".jpg", ".jpeg")):
        return "image/"
    if lower.endswith(".csv"):
        return "text/csv"
    if lower.endswith((".html", ".htm")):
        return "text/html"
    if lower.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if lower.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if lower.endswith(".md"):
        return "text/markdown"
    return ""


def _entree_depuis_piece(piece: str, registre_par_piece: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    base = piece_basename(piece)
    entree = registre_par_piece.get(base) or registre_par_piece.get(piece)
    if entree is not None:
        return dict(entree)
    chemin = piece.replace("\\", "/")
    nom = base or piece
    return {
        "chemin": chemin if "/" in chemin else "",
        "nom": nom,
        "mime": _mime_depuis_nom(nom),
    }


def _charger_regles_aiguillage(ontologie: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = ontologie.get("sources") or {}
    drive = sources.get("drive_registre")
    if isinstance(drive, str) and drive:
        path = _chemin_aiguillage(Path(drive))
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            regles = payload.get("regles") or []
            return [r for r in regles if isinstance(r, dict)]
    regles: list[dict[str, Any]] = []
    for famille in ontologie.get("familles") or []:
        if not isinstance(famille, dict):
            continue
        regle = dict(famille.get("regle") or {})
        if famille.get("nom"):
            regle.setdefault("famille", famille["nom"])
        if famille.get("traitement"):
            regle.setdefault("traitement", famille["traitement"])
        if regle:
            regles.append(regle)
    return regles


def _famille_pour_piece(
    piece: str,
    *,
    regles: list[dict[str, Any]],
    registre_par_piece: Mapping[str, Mapping[str, Any]],
) -> str | None:
    entree = _entree_depuis_piece(piece, registre_par_piece)
    base = piece_basename(piece)
    direct = registre_par_piece.get(base) or registre_par_piece.get(piece)
    if direct and direct.get("famille"):
        return str(direct["famille"]) or None
    for regle in regles:
        if _regle_correspond(entree, regle):
            famille = regle.get("famille") or ""
            return famille or None
    return None


def charger_registre_par_piece(ontologie: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index du registre Drive (si ``sources.drive_registre`` pointe un fichier)."""
    sources = ontologie.get("sources") or {}
    registre_par_piece: dict[str, dict[str, Any]] = {}
    drive = sources.get("drive_registre")
    if isinstance(drive, str) and Path(drive).is_file():
        registre = json.loads(Path(drive).read_text(encoding="utf-8"))
        for entree in registre:
            if not isinstance(entree, dict):
                continue
            piece = _piece_de_entree(entree)
            if piece:
                registre_par_piece[piece] = entree
            nom = entree.get("nom")
            if isinstance(nom, str) and nom:
                registre_par_piece.setdefault(nom, entree)
    return registre_par_piece


def famille_pour_piece(
    piece: str,
    ontologie: Mapping[str, Any],
    *,
    regles: list[dict[str, Any]] | None = None,
    registre_par_piece: Mapping[str, Mapping[str, Any]] | None = None,
) -> str | None:
    """Famille d'aiguillage d'une pièce (registre Drive, sinon règles d'ontologie)."""
    if regles is None:
        regles = _charger_regles_aiguillage(ontologie)
    if registre_par_piece is None:
        registre_par_piece = charger_registre_par_piece(ontologie)
    return _famille_pour_piece(
        piece, regles=regles, registre_par_piece=registre_par_piece
    )


def familles_pour_unite(
    paquet: Mapping[str, Any],
    ontologie: Mapping[str, Any],
    *,
    regles: list[dict[str, Any]] | None = None,
    registre_par_piece: Mapping[str, Mapping[str, Any]] | None = None,
) -> set[str]:
    """Union des familles nommées des pièces du paquet (les ``None`` sont omises)."""
    from banc_realite.reconstitution_entites import pieces_referencees_par_paquet

    if regles is None:
        regles = _charger_regles_aiguillage(ontologie)
    if registre_par_piece is None:
        registre_par_piece = charger_registre_par_piece(ontologie)
    familles: set[str] = set()
    for piece in pieces_referencees_par_paquet(paquet):
        famille = _famille_pour_piece(
            piece, regles=regles, registre_par_piece=registre_par_piece
        )
        if famille:
            familles.add(famille)
    return familles


def familles_des_pieces(
    pieces: Iterable[str],
    ontologie: Mapping[str, Any],
    *,
    regles: list[dict[str, Any]] | None = None,
    registre_par_piece: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str | None]:
    """Famille de chaque pièce, dans l'ordre, y compris les ``None``."""
    if regles is None:
        regles = _charger_regles_aiguillage(ontologie)
    if registre_par_piece is None:
        registre_par_piece = charger_registre_par_piece(ontologie)
    return [
        _famille_pour_piece(piece, regles=regles, registre_par_piece=registre_par_piece)
        for piece in pieces
    ]


def _fusions_depuis_decisions(decisions: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    fusions: dict[str, str] = {}
    for decision in decisions:
        effet = str(decision.get("effet") or "")
        match = _FUSION_EFFET.match(effet.strip())
        if match:
            fusions[match.group(1).strip()] = match.group(2).strip()
    return fusions


def _cle_utilisable(cle: str) -> bool:
    if not cle or not cle.strip():
        return False
    if "#" in cle:
        return False
    return len(normalize_text(cle)) >= _MIN_CLE_CHARS


def _cles_entite(entite: Mapping[str, Any]) -> list[str]:
    cles: list[str] = []
    for raw in (entite.get("cles") or []) + (entite.get("variantes") or []):
        if isinstance(raw, str) and _cle_utilisable(raw):
            cles.append(normalize_text(raw))
    nom = entite.get("nom")
    if isinstance(nom, str) and _cle_utilisable(nom):
        cles.append(normalize_text(nom))
    return cles


def _entites_actives_index(ontologie: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fusions = _fusions_depuis_decisions(ontologie.get("decisions") or [])
    sources_fusionnees = set(fusions)
    by_nom: dict[str, dict[str, Any]] = {}
    for entite in ontologie.get("entites") or []:
        if not isinstance(entite, dict):
            continue
        nom = entite.get("nom")
        if not isinstance(nom, str) or not nom.strip():
            continue
        if entite.get("retiree") or nom in sources_fusionnees:
            continue
        by_nom[nom] = dict(entite)
    for source, target in fusions.items():
        if source not in by_nom and target in by_nom:
            src = next(
                (e for e in ontologie.get("entites") or [] if isinstance(e, dict) and e.get("nom") == source),
                None,
            )
            if src is None:
                continue
            cles = list(by_nom[target].get("cles") or [])
            variantes = list(by_nom[target].get("variantes") or [])
            for cle in (src.get("cles") or []) + (src.get("variantes") or []):
                if isinstance(cle, str) and cle not in cles and cle not in variantes:
                    variantes.append(cle)
            by_nom[target] = {**by_nom[target], "cles": cles, "variantes": variantes}
    return by_nom


def _entites_dans_texte(texte: str, entites_index: Mapping[str, Mapping[str, Any]]) -> list[str]:
    norm = normalize_text(texte)
    if not norm:
        return []
    trouvees: list[str] = []
    for nom, entite in entites_index.items():
        for cle in _cles_entite(entite):
            if cle and cle in norm:
                trouvees.append(nom)
                break
    return sorted(set(trouvees))


def _concepts_dans_texte(texte: str, ontologie: Mapping[str, Any]) -> list[str]:
    norm = normalize_text(texte)
    if not norm:
        return []
    trouves: list[str] = []
    for concept in ontologie.get("concepts") or []:
        if not isinstance(concept, dict):
            continue
        nom = concept.get("nom")
        if not isinstance(nom, str) or not nom.strip():
            continue
        if normalize_text(nom) in norm:
            trouves.append(nom)
    return sorted(set(trouves))


def _graphe_cite(ontologie: Mapping[str, Any]) -> dict[str, set[str]]:
    graphe: dict[str, set[str]] = {}
    for relation in ontologie.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        if relation.get("verbe") != "cite":
            continue
        exemple = relation.get("exemple") or {}
        if not isinstance(exemple, dict):
            continue
        de = exemple.get("de")
        vers = exemple.get("vers")
        if not isinstance(de, str) or not isinstance(vers, str):
            continue
        graphe.setdefault(de, set()).add(vers)
        graphe.setdefault(vers, set()).add(de)
    return graphe


def etendre_entites_par_relations(
    entites: Iterable[str],
    ontologie: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    """Retourne (ensemble élargi, entités ajoutées par les relations cite)."""
    graphe = _graphe_cite(ontologie)
    base = set(entites)
    ajoutees: set[str] = set()
    for nom in list(base):
        for voisin in graphe.get(nom, ()):
            if voisin not in base:
                ajoutees.add(voisin)
                base.add(voisin)
    return base, ajoutees


def entites_filtre_depuis_question(
    question_texte: str,
    ontologie: Mapping[str, Any],
) -> tuple[list[str], list[str], set[str], bool]:
    """Directes, ajoutées par cite, ensemble filtre complet, repli si aucune directe."""
    index = _entites_actives_index(ontologie)
    directes = _entites_dans_texte(question_texte, index)
    if not directes:
        return [], [], set(), True
    elargi, ajoutees = etendre_entites_par_relations(directes, ontologie)
    actives = set(index)
    relations = sorted(n for n in ajoutees if n in actives)
    filtre_complet = {n for n in elargi if n in actives}
    return sorted(directes), relations, filtre_complet, False


def etiqueter_passages(
    ontologie: dict[str, Any],
    passages: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    entites_index = _entites_actives_index(ontologie)
    registre_par_piece = charger_registre_par_piece(ontologie)
    regles = _charger_regles_aiguillage(ontologie)

    out: dict[str, dict[str, Any]] = {}
    for passage in passages:
        pid = str(passage.get("id") or "")
        if not pid:
            continue
        piece = str(passage.get("piece") or "")
        texte = str(passage.get("texte") or passage.get("content") or "")
        famille = _famille_pour_piece(piece, regles=regles, registre_par_piece=registre_par_piece)
        out[pid] = {
            "famille": famille,
            "entites": _entites_dans_texte(texte, entites_index),
            "concepts": _concepts_dans_texte(texte, ontologie),
        }
    return out


def _stats_etiquetage(
    etiquettes: Mapping[str, Mapping[str, Any]],
    entites_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    passages = len(etiquettes)
    sans_famille = sum(1 for e in etiquettes.values() if not e.get("famille"))
    sans_entite = sum(1 for e in etiquettes.values() if not e.get("entites"))
    total_entites = sum(len(e.get("entites") or []) for e in etiquettes.values())
    moyenne = (total_entites / passages) if passages else 0.0
    vues: set[str] = set()
    for e in etiquettes.values():
        for nom in e.get("entites") or []:
            vues.add(str(nom))
    jamais_vues = sorted(set(entites_index) - vues)
    return {
        "passages": passages,
        "sans_famille": sans_famille,
        "sans_entite": sans_entite,
        "entites_par_passage_moyenne": round(moyenne, 4),
        "entites_jamais_vues": jamais_vues,
    }


def _passages_depuis_connection(connection, dossier_nom: str) -> list[dict[str, Any]]:
    dossier = dossier_courant(dossier_nom)
    rows = lister_passages_dossier(connection, dossier.vault_ref)
    passages: list[dict[str, Any]] = []
    for row in rows:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        piece = str(row.get("source_ref") or meta.get("piece") or "")
        line_start = int(meta.get("line_start") or 1)
        line_end = int(meta.get("line_end") or line_start)
        passages.append(
            {
                "id": str(row["id"]),
                "piece": piece,
                "texte": str(row.get("content") or ""),
                "passage_key": passage_key(piece, line_start, line_end),
                "line_start": line_start,
                "line_end": line_end,
            }
        )
    return passages


def etiqueter_dossier(
    dossier: str = "greenloop",
    version: str = "v1",
    *,
    connection=None,
) -> dict[str, Any]:
    ontologie_path = _chemin_ontologie(dossier, version)
    ontologie = json.loads(ontologie_path.read_text(encoding="utf-8"))
    if connection is None:
        from .infra import connect_existing

        connection = connect_existing()
    passages = _passages_depuis_connection(connection, dossier)
    etiquettes = etiqueter_passages(ontologie, passages)
    entites_index = _entites_actives_index(ontologie)
    payload = {
        "schema": SCHEMA_ETIQUETTES,
        "ontologie": version,
        "passages": etiquettes,
        "stats": _stats_etiquetage(etiquettes, entites_index),
    }
    sortie = _chemin_etiquettes(dossier, version)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def charger_etiquettes(chemin: Path) -> dict[str, Any]:
    return json.loads(chemin.read_text(encoding="utf-8"))


def filtrer_passages_par_entites_etiquettes(
    passages: list[dict[str, Any]],
    entites_filtre: set[str],
    etiquettes_par_passage: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not entites_filtre:
        return list(passages)
    gardes: list[dict[str, Any]] = []
    for item in passages:
        pid = str(item.get("id") or "")
        bloc = etiquettes_par_passage.get(pid) or {}
        entites = set(bloc.get("entites") or [])
        if entites & entites_filtre:
            gardes.append(item)
    return gardes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Étiquetage ontologie tenant → passages")
    parser.add_argument("--dossier", default="greenloop")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args(argv)
    payload = etiqueter_dossier(dossier=args.dossier, version=args.version)
    stats = payload.get("stats") or {}
    print(
        f"étiquettes écrites : {_chemin_etiquettes(args.dossier, args.version)} "
        f"({stats.get('passages')} passages)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
