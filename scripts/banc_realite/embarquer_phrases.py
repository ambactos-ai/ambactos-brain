# SPDX-License-Identifier: AGPL-3.0-only
"""Embarquement Mistral des unités phrase extraites des passages corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from banc_realite.compat_moteur import EMBED_DIM, embed_many_with_retry, load_api_key

from .config import BANC_ROOT, export_runtime_env
from .dossiers import Dossier, dossier_courant
from .moreau_or import normalize_verbatim
from .normalize import piece_basename
from .porte_phrase import phrases_du_passage

SORTIES_DIR = BANC_ROOT / "sorties"
DRY_RUN_PATH = SORTIES_DIR / "phrases-embeddings.dry-run.json"
CACHE_PATH = SORTIES_DIR / "phrases-embeddings.json"

MAX_UNITS = 25_000
MAX_CARACTERES = 3_000_000
MAX_EMBED_CALLS_DOSSIER = 1_500
BATCH_SIZE = 64
MISTRAL_EMBED_USD_PER_TOKEN = 0.10 / 1_000_000
FORMULE_COUT = "caracteres / 4 * 0.10 USD / 1_000_000 tokens (tarif public mistral-embed)"


def compter_unites_phrases(
    passages: list[tuple[str, str]] | list[tuple[str, str, str]],
    *,
    fenetre: int = 1,
) -> dict[str, Any]:
    """Passages (id, content) ou (id, piece, content) → statistiques dry-run (fonction pure)."""
    seen: dict[str, None] = {}
    caracteres = 0
    for row in passages:
        content = row[-1]
        for unit in phrases_du_passage(content, fenetre=fenetre):
            if unit in seen:
                continue
            seen[unit] = None
            caracteres += len(unit)
    unites = len(seen)
    tokens_estimes = caracteres / 4
    return {
        "passages": len(passages),
        "unites": unites,
        "caracteres": caracteres,
        "tokens_estimes": round(tokens_estimes, 2),
        "cout_estime_usd": round(tokens_estimes * MISTRAL_EMBED_USD_PER_TOKEN, 6),
        "formule_cout": FORMULE_COUT,
        "prix_unitaire_usd_par_token": MISTRAL_EMBED_USD_PER_TOKEN,
    }


def _fetch_passages(connection, vault_ref: str | None = None) -> list[tuple[str, str, str]]:
    if vault_ref is None or vault_ref == "greenloop":
        clause = " AND (vault_ref = %s OR vault_ref IS NULL)"
        params = (vault_ref or "greenloop",)
    else:
        clause = " AND vault_ref = %s"
        params = (vault_ref,)
    rows = connection.execute(
        f"""
        SELECT id, source_ref, content
        FROM public.memory_chunks
        WHERE source_type = 'corpus'
          {clause}
        ORDER BY id
        """,
        params,
    ).fetchall()
    return [
        (str(row["id"]), str(row.get("source_ref") or ""), str(row["content"] or ""))
        for row in rows
    ]


def compter_citations_dans_fenetres(
    passages: list[tuple[str, str, str]],
    assertions: list[dict[str, Any]],
    *,
    fenetre: int = 2,
) -> dict[str, Any]:
    """Combien des citations tiennent entières dans au moins une fenêtre glissante."""
    retrouvees = 0
    manquantes: list[str] = []
    for assertion in assertions:
        quote = str(assertion.get("quote") or "")
        piece = piece_basename(str(assertion.get("piece") or assertion.get("source") or ""))
        if not quote:
            manquantes.append(str(assertion.get("id") or ""))
            continue
        norm_quote = normalize_verbatim(quote)
        found = False
        for _passage_id, source_ref, content in passages:
            if piece and piece_basename(source_ref) != piece:
                continue
            for unit in phrases_du_passage(content, fenetre=fenetre):
                if norm_quote in normalize_verbatim(unit):
                    found = True
                    break
            if found:
                break
        if found:
            retrouvees += 1
        else:
            manquantes.append(str(assertion.get("id") or ""))
    total = len(assertions)
    return {
        "citations_total": total,
        "citations_dans_une_fenetre": f"{retrouvees}/{total}",
        "citations_manquantes": manquantes,
    }


def _citation_tient_dans_fenetre(
    assertion: dict[str, Any],
    passages: list[tuple[str, str, str]],
    *,
    fenetre: int,
) -> bool:
    quote = str(assertion.get("quote") or "")
    piece = piece_basename(str(assertion.get("piece") or assertion.get("source") or ""))
    if not quote:
        return False
    norm_quote = normalize_verbatim(quote)
    for _passage_id, source_ref, content in passages:
        if piece and piece_basename(source_ref) != piece:
            continue
        for unit in phrases_du_passage(content, fenetre=fenetre):
            if norm_quote in normalize_verbatim(unit):
                return True
    return False


def _motif_citation_hors_fenetre(
    assertion: dict[str, Any],
    passages: list[tuple[str, str, str]],
    *,
    fenetre: int,
    dossier: Dossier,
) -> str:
    piece = piece_basename(str(assertion.get("piece") or assertion.get("source") or ""))
    if piece in dossier.pieces_tabulaires:
        return "pièce tabulaire (unité tableau, pas fenêtre glissante prose)"
    quote = str(assertion.get("quote") or "")
    for _passage_id, source_ref, content in passages:
        if piece and piece_basename(source_ref) != piece:
            continue
        for unit in phrases_du_passage(content, fenetre=1):
            norm_unit = normalize_verbatim(unit)
            norm_quote = normalize_verbatim(quote)
            if norm_quote in norm_unit:
                return "citation plus longue que la borne caractères de la fenêtre"
        for unit in phrases_du_passage(content, fenetre=fenetre):
            if " …" in unit and normalize_verbatim(quote) not in normalize_verbatim(unit):
                return "fenêtre tronquée (borne caractères)"
    return "aucune fenêtre glissante ne couvre la citation dans la pièce"


def calculer_fenetre_depuis_or(
    passages: list[tuple[str, str, str]],
    assertions: list[dict[str, Any]],
    *,
    dossier: Dossier,
    fenetre_max: int = 20,
) -> dict[str, Any]:
    """Plus petite taille (phrases) telle que la citation la plus longue tienne dans une fenêtre."""
    longest_len = 0
    longest_id = ""
    longest_assertion: dict[str, Any] | None = None
    min_par_citation: dict[str, int | None] = {}
    for assertion in assertions:
        aid = str(assertion.get("id") or "")
        quote = str(assertion.get("quote") or assertion.get("content") or "")
        if len(quote) > longest_len:
            longest_len = len(quote)
            longest_id = aid
            longest_assertion = assertion
        needed: int | None = None
        for candidate in range(1, fenetre_max + 1):
            if _citation_tient_dans_fenetre(assertion, passages, fenetre=candidate):
                needed = candidate
                break
        min_par_citation[aid] = needed

    longest_min: int | None = None
    if longest_assertion is not None:
        for candidate in range(1, fenetre_max + 1):
            if _citation_tient_dans_fenetre(longest_assertion, passages, fenetre=candidate):
                longest_min = candidate
                break
    fenetre_phrases = longest_min or 1

    couverture = compter_citations_dans_fenetres(passages, assertions, fenetre=fenetre_phrases)
    manquantes_detail: list[dict[str, str]] = []
    for assertion in assertions:
        aid = str(assertion.get("id") or "")
        if aid in couverture["citations_manquantes"]:
            manquantes_detail.append(
                {
                    "id": aid,
                    "motif": _motif_citation_hors_fenetre(
                        assertion,
                        passages,
                        fenetre=fenetre_phrases,
                        dossier=dossier,
                    ),
                }
            )

    calcul = (
        f"min phrases pour citation la plus longue {longest_id!r} "
        f"({longest_len} car.) = {longest_min} "
        f"(borne unité {150 * (fenetre_phrases + 1)} car.)"
    )
    return {
        "fenetre_phrases": fenetre_phrases,
        "calcul": calcul,
        "citation_plus_longue_id": longest_id,
        "citation_plus_longue_chars": longest_len,
        "citation_plus_longue_min_phrases": longest_min,
        "min_phrases_par_citation": min_par_citation,
        "citations_total": couverture["citations_total"],
        "citations_dans_une_fenetre": couverture["citations_dans_une_fenetre"],
        "citations_manquantes": couverture["citations_manquantes"],
        "citations_manquantes_detail": manquantes_detail,
        "fenetre_origine_2": compter_citations_dans_fenetres(passages, assertions, fenetre=2)[
            "citations_dans_une_fenetre"
        ],
    }


def run_embarquer_phrases_calibrees(
    dossier: Dossier | str = "moreau",
    *,
    dry_run: bool = False,
    connection=None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    max_appels: int = MAX_EMBED_CALLS_DOSSIER,
) -> dict[str, Any]:
    from .infra import connect_existing, start_cluster
    from .or_dossier import load_gold

    current = dossier_courant(dossier) if isinstance(dossier, str) else dossier
    if current.nom == "greenloop":
        raise RuntimeError("fenêtres calibrées réservées aux dossiers hors GreenLoop")
    export_runtime_env()
    if connection is None:
        start_cluster()
        connection = connect_existing()

    cache_path = current.cache_phrases_embeddings_calibrees
    meta_path = current.meta_fenetre_calibree
    journal_path = cache_path.parent / "phrases-embeddings-calibrees-journal.jsonl"
    passages = _fetch_passages(connection, vault_ref=current.vault_ref)
    gold = load_gold(current)
    calibration = calculer_fenetre_depuis_or(
        passages,
        gold["assertions"],
        dossier=current,
    )
    fenetre = int(calibration["fenetre_phrases"])
    stats = compter_unites_phrases(passages, fenetre=fenetre)
    stats.update(calibration)
    stats["fenetre"] = fenetre
    stats["dossier"] = current.nom
    stats["vault_ref"] = current.vault_ref
    stats["cache_path"] = str(cache_path)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cache = _load_cache(cache_path)
    units_ordered: list[str] = []
    seen: set[str] = set()
    for _passage_id, _piece, content in passages:
        for unit in phrases_du_passage(content, fenetre=fenetre):
            if unit in seen:
                continue
            seen.add(unit)
            units_ordered.append(unit)
    pending = [unit for unit in units_ordered if unit not in cache]
    stats["unites_nouvelles"] = len(pending)
    stats["deja_cache"] = len(units_ordered) - len(pending)

    if dry_run:
        return {**stats, "dry_run": True}

    if stats["unites"] > MAX_UNITS:
        raise RuntimeError(f"refus : {stats['unites']} unités > seuil {MAX_UNITS}")

    if embed_fn is None:
        load_api_key()
        embed_fn = embed_many_with_retry

    appels = 0
    echecs = 0
    for offset in range(0, len(pending)):
        if appels >= max_appels:
            break
        batch = pending[offset : offset + BATCH_SIZE]
        if not batch:
            break
        try:
            vectors = embed_fn(batch)
        except Exception:
            echecs += len(batch)
            continue
        for unit, vector in zip(batch, vectors):
            cache[unit] = vector
        _write_cache(cache_path, cache)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "lot": appels + 1,
                        "unites": len(batch),
                        "dossier": current.nom,
                        "fenetre": fenetre,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        appels += 1

    return {
        **stats,
        "fenetres": len(units_ordered),
        "embeddings": len(cache),
        "appels_embedding": appels,
        "echecs": echecs,
        "plafond_atteint": appels >= max_appels and pending,
        "files_written": [str(cache_path), str(meta_path)],
    }


def load_fenetre_calibree(dossier: Dossier | str) -> dict[str, Any]:
    current = dossier_courant(dossier) if isinstance(dossier, str) else dossier
    path = current.meta_fenetre_calibree
    if not path.is_file():
        raise RuntimeError(f"métadonnées fenêtre calibrée absentes : {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "fenetre_phrases" not in payload:
        raise RuntimeError(f"métadonnées fenêtre calibrée invalides : {path}")
    return payload


def _load_cache(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"cache invalide : {path}")
    return payload


def _write_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def run_embarquer_phrases(
    *,
    dry_run: bool = False,
    fenetre: int = 1,
    connection=None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict[str, Any]:
    from .infra import connect_existing, start_cluster

    export_runtime_env()
    if connection is None:
        start_cluster()
        connection = connect_existing()

    passages = _fetch_passages(connection)
    stats = compter_unites_phrases(passages, fenetre=fenetre)
    stats["fenetre"] = fenetre
    stats["formule_cout"] = FORMULE_COUT
    stats["prix_unitaire_usd_par_token"] = MISTRAL_EMBED_USD_PER_TOKEN

    cache = _load_cache(CACHE_PATH)
    units_ordered: list[str] = []
    seen: set[str] = set()
    for _passage_id, _piece, content in passages:
        for unit in phrases_du_passage(content, fenetre=fenetre):
            if unit in seen:
                continue
            seen.add(unit)
            units_ordered.append(unit)
    pending = [unit for unit in units_ordered if unit not in cache]
    stats["unites_nouvelles"] = len(pending)
    stats["deja_cache"] = len(units_ordered) - len(pending)

    if dry_run:
        SORTIES_DIR.mkdir(parents=True, exist_ok=True)
        DRY_RUN_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return {**stats, "dry_run": True, "files_written": [str(DRY_RUN_PATH)]}

    if stats["unites"] > MAX_UNITS:
        raise RuntimeError(f"refus : {stats['unites']} unités > seuil {MAX_UNITS}")
    if stats["caracteres"] > MAX_CARACTERES:
        raise RuntimeError(f"refus : {stats['caracteres']} caractères > seuil {MAX_CARACTERES}")

    deja_cache = stats["deja_cache"]

    if embed_fn is None:
        load_api_key()
        embed_fn = embed_many_with_retry

    lots = 0
    echecs = 0
    dimension: int | None = None

    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset : offset + BATCH_SIZE]
        if not batch:
            continue
        try:
            vectors = embed_fn(batch)
        except Exception:
            echecs += len(batch)
            continue
        if dimension is None:
            if not vectors or len(vectors[0]) != EMBED_DIM:
                raise RuntimeError(
                    f"dimension attendue {EMBED_DIM}, obtenue {len(vectors[0]) if vectors else 0}"
                )
            dimension = EMBED_DIM
        for unit, vector in zip(batch, vectors):
            cache[unit] = vector
        _write_cache(CACHE_PATH, cache)
        lots += 1

    summary = {
        "unites": len(units_ordered),
        "unites_embarquees": len(units_ordered) - deja_cache - echecs,
        "lots": lots,
        "deja_cache": deja_cache,
        "echecs": echecs,
        "dimension": dimension or EMBED_DIM,
        "files_written": [str(CACHE_PATH)],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_embarquer_phrases_dossier(
    dossier: Dossier | str = "moreau",
    *,
    dry_run: bool = False,
    fenetre: int = 2,
    connection=None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    max_appels: int = MAX_EMBED_CALLS_DOSSIER,
) -> dict[str, Any]:
    from .infra import connect_existing, start_cluster
    from .or_dossier import load_gold

    current = dossier_courant(dossier) if isinstance(dossier, str) else dossier
    export_runtime_env()
    if connection is None:
        start_cluster()
        connection = connect_existing()

    cache_path = current.cache_phrases_embeddings
    journal_path = cache_path.parent / "phrases-embeddings-journal.jsonl"
    passages = _fetch_passages(connection, vault_ref=current.vault_ref)
    stats = compter_unites_phrases(passages, fenetre=fenetre)
    stats["fenetre"] = fenetre
    stats["dossier"] = current.nom
    stats["vault_ref"] = current.vault_ref
    stats["cache_path"] = str(cache_path)

    gold = load_gold(current)
    couverture = compter_citations_dans_fenetres(
        passages,
        gold["assertions"],
        fenetre=fenetre,
    )
    stats.update(couverture)

    cache = _load_cache(cache_path)
    units_ordered: list[str] = []
    seen: set[str] = set()
    for _passage_id, _piece, content in passages:
        for unit in phrases_du_passage(content, fenetre=fenetre):
            if unit in seen:
                continue
            seen.add(unit)
            units_ordered.append(unit)
    pending = [unit for unit in units_ordered if unit not in cache]
    stats["unites_nouvelles"] = len(pending)
    stats["deja_cache"] = len(units_ordered) - len(pending)

    if dry_run:
        return {**stats, "dry_run": True}

    if stats["unites"] > MAX_UNITS:
        raise RuntimeError(f"refus : {stats['unites']} unités > seuil {MAX_UNITS}")

    if embed_fn is None:
        load_api_key()
        embed_fn = embed_many_with_retry

    appels = 0
    echecs = 0
    for offset in range(0, len(pending)):
        if appels >= max_appels:
            break
        batch = pending[offset : offset + BATCH_SIZE]
        if not batch:
            break
        try:
            vectors = embed_fn(batch)
        except Exception:
            echecs += len(batch)
            continue
        for unit, vector in zip(batch, vectors):
            cache[unit] = vector
        _write_cache(cache_path, cache)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "lot": appels + 1,
                        "unites": len(batch),
                        "dossier": current.nom,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        appels += 1

    return {
        **stats,
        "fenetres": len(units_ordered),
        "embeddings": len(cache),
        "appels_embedding": appels,
        "echecs": echecs,
        "plafond_atteint": appels >= max_appels and pending,
        "files_written": [str(cache_path)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embarquer les unités phrase (Mistral mistral-embed).")
    parser.add_argument("--dry-run", action="store_true", help="Estimer coût sans appel réseau.")
    parser.add_argument("--fenetre", type=int, default=1, choices=(1, 2), help="Taille de fenêtre phrase.")
    parser.add_argument(
        "--calibrees",
        action="store_true",
        help="Calibrer la fenêtre sur l'or du dossier (MOREAU) et embarquer le cache dédié.",
    )
    parser.add_argument(
        "--dossier",
        default="moreau",
        help="Dossier cible pour --calibrees (défaut moreau).",
    )
    args = parser.parse_args(argv)
    try:
        if args.calibrees:
            run_embarquer_phrases_calibrees(args.dossier, dry_run=args.dry_run)
        else:
            run_embarquer_phrases(dry_run=args.dry_run, fenetre=args.fenetre)
    except Exception as exc:
        print(json.dumps({"status": "arret", "erreur": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
