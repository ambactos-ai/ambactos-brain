# SPDX-License-Identifier: AGPL-3.0-only
"""Embeddings Mistral des faits MRFS — cache, comptage à blanc, plafond avant requête.

Aucun appel réel n'est émis par `--a-blanc`. Sous pytest, un appel réel lève.
"""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "scripts", ROOT / "src"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.compat_moteur import MODEL_ID, embed_many_with_retry, load_api_key
from banc_realite.lecteur_faits_classes import Fait, charger_faits  # noqa: E402
from banc_realite.lecteur_situations_composites import (  # noqa: E402
    DEFAULT_SITUATIONS_ROOT,
    charger_situations,
    situations_vers_faits,
)
from banc_realite.plafond_appels import CompteurAppels, PlafondAtteint  # noqa: E402

DEFAULT_CONSTRUCTEUR_ROOT = (
    BANC_VAR / 'dossiers' / 'mrfs' / 'constructeur'
)
BATCH_SIZE = 64
DEFAULT_PLAFOND_REQUETES = 100
MISTRAL_EMBED_USD_PAR_TOKEN = 0.10 / 1_000_000
EUR_PAR_USD = 0.92
FORMULE_COUT = (
    "caractères / 4 × 0,10 USD / 1 000 000 tokens × 0,92 EUR/USD "
    "(tarif public mistral-embed)"
)
FOURNISSEUR_EMBED = "mistral"


class AppelEmbeddingsInterdit(RuntimeError):
    """Garde pytest — aucun appel réel à l'API d'embeddings."""


def cache_embeddings_faits_path(
    dossier: str,
    *,
    constructeur_root: Path | None = None,
) -> Path:
    root = constructeur_root or DEFAULT_CONSTRUCTEUR_ROOT
    return root / dossier / "embeddings-faits.json"


def cache_embeddings_situations_path(
    dossier: str,
    *,
    situations_root: Path | None = None,
) -> Path:
    root = situations_root or DEFAULT_SITUATIONS_ROOT
    return root / dossier / "embeddings-situations.json"


def cache_embeddings_situations_lettre_path(
    dossier: str,
    *,
    situations_root: Path | None = None,
) -> Path:
    root = situations_root or DEFAULT_SITUATIONS_ROOT
    return root / dossier / "embeddings-situations-lettre.json"


def make_compteur_embeddings_situations(
    *,
    situations_root: Path | None = None,
    plafond: int = DEFAULT_PLAFOND_REQUETES,
) -> CompteurAppels:
    root = situations_root or DEFAULT_SITUATIONS_ROOT
    compteur = CompteurAppels(
        compteur_path=root / "compteur-embeddings.json",
        journal_path=root / "appels-embeddings.jsonl",
        plafond=plafond,
    )
    compteur.plafond = plafond
    return compteur


def make_compteur_embeddings(
    *,
    constructeur_root: Path | None = None,
    plafond: int = DEFAULT_PLAFOND_REQUETES,
) -> CompteurAppels:
    root = constructeur_root or DEFAULT_CONSTRUCTEUR_ROOT
    compteur = CompteurAppels(
        compteur_path=root / "compteur-embeddings.json",
        journal_path=root / "appels-embeddings.jsonl",
        plafond=plafond,
    )
    compteur.plafond = plafond
    return compteur


def charger_cache(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"cache embeddings faits invalide : {path}")
    return {
        str(key): [float(x) for x in value]
        for key, value in payload.items()
        if isinstance(value, list)
    }


def ecrire_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def faits_a_embarquer(
    faits: Sequence[Fait],
    cache: Mapping[str, Sequence[float]],
) -> list[Fait]:
    pending: list[Fait] = []
    seen: set[str] = set()
    for fait in faits:
        key = fait.empreinte
        if key in cache or key in seen:
            continue
        seen.add(key)
        pending.append(fait)
    return pending


MAX_CAR_EMBEDDING_SITUATION = 12_000
MAX_CAR_REQUETE_SITUATIONS = 24_000


def _lots_par_volume(faits, *, batch_size: int):
    """Lots d'au plus `batch_size` textes et MAX_CAR_REQUETE_SITUATIONS caractères envoyés."""
    lot: list = []
    volume = 0
    for fait in faits:
        taille = min(len(fait.text), MAX_CAR_EMBEDDING_SITUATION)
        if lot and (len(lot) >= batch_size or volume + taille > MAX_CAR_REQUETE_SITUATIONS):
            yield lot
            lot, volume = [], 0
        lot.append(fait)
        volume += taille
    if lot:
        yield lot


def _compter_requetes_emission(
    pending: Sequence[Fait],
    *,
    batch_size: int,
    decouper_par_volume: bool,
) -> tuple[int, int]:
    if not pending:
        return 0, 0
    if decouper_par_volume:
        lots = list(_lots_par_volume(pending, batch_size=batch_size))
        n_requetes = len(lots)
        caracteres = sum(
            min(len(fait.text), MAX_CAR_EMBEDDING_SITUATION) for lot in lots for fait in lot
        )
        return n_requetes, caracteres
    caracteres = sum(len(fait.text) for fait in pending)
    n_requetes = (len(pending) + batch_size - 1) // batch_size
    return n_requetes, caracteres


def compter_a_blanc(
    faits: Sequence[Fait],
    *,
    cache: Mapping[str, Sequence[float]] | None = None,
    batch_size: int = BATCH_SIZE,
    dossier: str = "",
    decouper_par_volume: bool = False,
) -> dict[str, Any]:
    present = dict(cache or {})
    pending = faits_a_embarquer(faits, present)
    n_requetes, caracteres = _compter_requetes_emission(
        pending, batch_size=batch_size, decouper_par_volume=decouper_par_volume
    )
    tokens = caracteres / 4
    usd = tokens * MISTRAL_EMBED_USD_PAR_TOKEN
    n_deja_cache = sum(1 for fait in faits if fait.empreinte in present)
    return {
        "dossier": dossier,
        "modele": MODEL_ID,
        "n_faits": len(faits),
        "n_a_embarquer": len(pending),
        "n_deja_cache": n_deja_cache,
        "n_requetes": n_requetes,
        "batch_size": batch_size,
        "caracteres": caracteres,
        "tokens_estimes": round(tokens, 2),
        "cout_estime_usd": round(usd, 6),
        "cout_estime_eur": round(usd * EUR_PAR_USD, 6),
        "formule_cout": FORMULE_COUT,
        "dry_run": True,
    }


def _interdire_appel_reel() -> None:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise AppelEmbeddingsInterdit("appel embeddings interdit sous pytest")


def _journaliser(
    path: Path,
    *,
    emis: int,
    plafond: int,
    dossier: str,
    n_textes: int,
    duree_s: float,
    issue: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ligne = {
        "horodatage": datetime.now(timezone.utc).isoformat(),
        "fournisseur": FOURNISSEUR_EMBED,
        "modele": MODEL_ID,
        "dossier": dossier,
        "n_textes": n_textes,
        "emis": emis,
        "plafond": plafond,
        "duree_s": round(duree_s, 3),
        "issue": issue,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ligne, ensure_ascii=False) + "\n")


def _embed_lot(
    textes: list[str],
    *,
    embed_fn: Callable[[list[str]], list[list[float]]],
    compteur: CompteurAppels,
    dossier: str,
) -> list[list[float]]:
    emis = compteur.reserver_emission()
    started = time.monotonic()
    issue = "echec"
    try:
        vectors = embed_fn(textes)
        issue = "succes"
        return vectors
    except AppelEmbeddingsInterdit:
        issue = "interdit"
        raise
    except PlafondAtteint:
        issue = "plafond"
        raise
    except Exception:
        issue = "exception"
        raise
    finally:
        _journaliser(
            compteur.journal_path,
            emis=emis,
            plafond=compteur.plafond,
            dossier=dossier,
            n_textes=len(textes),
            duree_s=time.monotonic() - started,
            issue=issue,
        )


def embarquer_faits(
    dossier: str,
    *,
    a_blanc: bool = True,
    constructeur_root: Path | None = None,
    faits: Sequence[Fait] | None = None,
    cache: dict[str, list[float]] | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    plafond: int = DEFAULT_PLAFOND_REQUETES,
    batch_size: int = BATCH_SIZE,
    dsn: str | None = None,
) -> dict[str, Any]:
    root = constructeur_root or DEFAULT_CONSTRUCTEUR_ROOT
    cache_path = cache_embeddings_faits_path(dossier, constructeur_root=root)
    loaded = list(faits) if faits is not None else charger_faits(
        dossier, dsn=dsn or "postgresql://postgres@127.0.0.1:56102/postgres"
    )
    present = cache if cache is not None else charger_cache(cache_path)
    stats = compter_a_blanc(
        loaded, cache=present, batch_size=batch_size, dossier=dossier
    )
    stats["cache_path"] = str(cache_path)
    pending = faits_a_embarquer(loaded, present)
    if a_blanc:
        return stats
    if not pending:
        stats.update({"dry_run": False, "n_requetes_emises": 0, "n_cache": len(present)})
        return stats
    if embed_fn is None:
        _interdire_appel_reel()
        load_api_key()
        embed_fn = embed_many_with_retry

    compteur = make_compteur_embeddings(constructeur_root=root, plafond=plafond)
    # Le plafond demandé est toujours écrit : à la reprise, le compteur relit son
    # fichier, et un plafond resté en mémoire serait ignoré.
    if plafond is not None:
        compteur.definir_plafond(plafond)

    working = dict(present)
    appels = 0
    for offset in range(0, len(pending), batch_size):
        lot = pending[offset : offset + batch_size]
        if not lot:
            continue
        textes = [fait.text for fait in lot]
        vectors = _embed_lot(
            textes, embed_fn=embed_fn, compteur=compteur, dossier=dossier
        )
        for fait, vector in zip(lot, vectors):
            working[fait.empreinte] = vector
        ecrire_cache(cache_path, working)
        appels += 1
    stats.update(
        {
            "dry_run": False,
            "n_requetes_emises": appels,
            "n_cache": len(working),
        }
    )
    return stats


def embarquer_situations(
    dossier: str,
    *,
    a_blanc: bool = True,
    situations_root: Path | None = None,
    situations: Sequence[Mapping[str, Any]] | None = None,
    cache: dict[str, list[float]] | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    plafond: int = DEFAULT_PLAFOND_REQUETES,
    batch_size: int = BATCH_SIZE,
    chemin: Path | None = None,
    lettres_chronologie: bool = False,
) -> dict[str, Any]:
    root = situations_root or DEFAULT_SITUATIONS_ROOT
    if lettres_chronologie:
        cache_path = cache_embeddings_situations_lettre_path(
            dossier, situations_root=root
        )
    else:
        cache_path = cache_embeddings_situations_path(dossier, situations_root=root)
    loaded = (
        list(situations)
        if situations is not None
        else charger_situations(dossier, chemin=chemin, situations_root=root)
    )
    faits = situations_vers_faits(loaded, lettres_chronologie=lettres_chronologie)
    present = cache if cache is not None else charger_cache(cache_path)
    stats = compter_a_blanc(
        faits,
        cache=present,
        batch_size=batch_size,
        dossier=dossier,
        decouper_par_volume=True,
    )
    stats["cache_path"] = str(cache_path)
    stats["n_situations"] = len(loaded)
    pending = faits_a_embarquer(faits, present)
    if a_blanc:
        return stats
    if not pending:
        stats.update({"dry_run": False, "n_requetes_emises": 0, "n_cache": len(present)})
        return stats
    if embed_fn is None:
        _interdire_appel_reel()
        load_api_key()
        embed_fn = embed_many_with_retry

    compteur = make_compteur_embeddings_situations(
        situations_root=root, plafond=plafond
    )
    if plafond is not None:
        compteur.definir_plafond(plafond)

    working = dict(present)
    appels = 0
    for lot in _lots_par_volume(pending, batch_size=batch_size):
        # mistral-embed refuse plus de 8 192 tokens ; un registre en fait un par 1,85 caractère.
        textes = [fait.text[:MAX_CAR_EMBEDDING_SITUATION] for fait in lot]
        vectors = _embed_lot(
            textes, embed_fn=embed_fn, compteur=compteur, dossier=dossier
        )
        for fait, vector in zip(lot, vectors):
            working[fait.empreinte] = vector
        ecrire_cache(cache_path, working)
        appels += 1
    stats.update(
        {
            "dry_run": False,
            "n_requetes_emises": appels,
            "n_cache": len(working),
        }
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embeddings Mistral des faits MRFS")
    parser.add_argument("--dossier", required=True, help="greenloop ou moreau")
    parser.add_argument(
        "--a-blanc",
        action="store_true",
        help="Comptage sans aucun appel (défaut si --embarquer est absent)",
    )
    parser.add_argument(
        "--embarquer",
        action="store_true",
        help="Émet les requêtes manquantes (interdit sous pytest)",
    )
    parser.add_argument("--constructeur-root", type=Path, default=None)
    parser.add_argument("--situations-root", type=Path, default=None)
    parser.add_argument(
        "--situations",
        action="store_true",
        help="Embarque le texte classé des situations composites",
    )
    parser.add_argument(
        "--lettres-chronologie",
        action="store_true",
        help="Texte classé avec lettres de chronologie (cache séparé, avec --situations)",
    )
    parser.add_argument("--plafond", type=int, default=DEFAULT_PLAFOND_REQUETES)
    args = parser.parse_args(argv)
    if args.embarquer and args.a_blanc:
        parser.error("--a-blanc et --embarquer sont exclusifs")
    a_blanc = not args.embarquer
    if args.situations:
        stats = embarquer_situations(
            args.dossier,
            a_blanc=a_blanc,
            situations_root=args.situations_root,
            plafond=args.plafond,
            lettres_chronologie=args.lettres_chronologie,
        )
    else:
        stats = embarquer_faits(
            args.dossier,
            a_blanc=a_blanc,
            constructeur_root=args.constructeur_root,
            plafond=args.plafond,
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
