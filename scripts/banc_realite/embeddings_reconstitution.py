# SPDX-License-Identifier: AGPL-3.0-only
"""Embeddings Mistral des objets de reconstitution — cache, comptage à blanc, plafond.

Réutilise l'adaptateur, le compteur et le journal de `embeddings_faits.py --situations`.
Aucun appel réel n'est émis par `--a-blanc`. Sous pytest, un appel réel lève.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "scripts", ROOT / "src"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.embeddings_faits import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_PLAFOND_REQUETES,
    MAX_CAR_EMBEDDING_SITUATION,
    MODEL_ID,
    _embed_lot,
    _interdire_appel_reel,
    _lots_par_volume,
    charger_cache,
    compter_a_blanc,
    embed_many_with_retry,
    faits_a_embarquer,
    load_api_key,
    make_compteur_embeddings_situations,
)
from banc_realite.lecteur_situations_composites import (  # noqa: E402
    DEFAULT_SITUATIONS_ROOT,
)
from banc_realite.recherche_entites import (  # noqa: E402
    Objet,
    charger_index,
    chemin_reconstitution,
    objets_de_recherche,
)


CACHE_OBJETS = "embeddings-objets.json"
CACHE_OBJETS_LOCK = "embeddings-objets.lock"


def cache_embeddings_objets_path(
    dossier: str,
    *,
    reconstitution_dir: Path | None = None,
) -> Path:
    root = Path(reconstitution_dir) if reconstitution_dir is not None else chemin_reconstitution(dossier)
    return root / CACHE_OBJETS


# Un objet de recherche fait 275 caractères en médiane ; quelques blocs en font des dizaines
# de milliers. Au-delà de 1 500 caractères, seul le début est embarqué : le classement par
# les mots et par les clés lit toujours le texte entier.
MAX_CAR_TEXTE_OBJET = 1_500
MAX_CAR_REQUETE_OBJETS = 24_000


def _lots_objets(objets, *, batch_size: int):
    lot: list = []
    volume = 0
    for objet in objets:
        taille = min(len(objet.text), MAX_CAR_TEXTE_OBJET)
        if lot and (len(lot) >= batch_size or volume + taille > MAX_CAR_REQUETE_OBJETS):
            yield lot
            lot, volume = [], 0
        lot.append(objet)
        volume += taille
    if lot:
        yield lot


def _fusionner_cache_objets(
    cache_path: Path,
    nouveaux: Mapping[str, list[float]],
) -> dict[str, list[float]]:
    """Relit le cache sous verrou, y ajoute les clés de l'appelant, écrit atomiquement."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.parent / CACHE_OBJETS_LOCK
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            fusionne = charger_cache(cache_path)
            fusionne.update(nouveaux)
            tmp = cache_path.parent / (
                f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            tmp.write_text(json.dumps(fusionne, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cache_path)
            return fusionne
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _charger_objets(
    dossier: str,
    *,
    reconstitution_dir: Path | None = None,
    objets: Sequence[Objet] | None = None,
) -> list[Objet]:
    if objets is not None:
        return list(objets)
    root = Path(reconstitution_dir) if reconstitution_dir is not None else chemin_reconstitution(dossier)
    return objets_de_recherche(charger_index(root))


def compter_objets_a_blanc(
    objets: Sequence[Objet],
    *,
    cache: Mapping[str, Sequence[float]] | None = None,
    batch_size: int = BATCH_SIZE,
    dossier: str = "",
) -> dict[str, Any]:
    present = dict(cache or {})
    pending = faits_a_embarquer(objets, present)
    stats = compter_a_blanc(objets, cache=present, batch_size=batch_size, dossier=dossier)
    lots = list(_lots_objets(pending, batch_size=batch_size))
    caracteres_envoyes = sum(min(len(objet.text), MAX_CAR_TEXTE_OBJET) for objet in pending)
    par_sorte = Counter(objet.sorte for objet in objets)
    stats.update(
        {
            "n_objets": len(objets),
            "n_faits": par_sorte.get("fait", 0),
            "n_regles": par_sorte.get("regle", 0),
            "n_lignes": par_sorte.get("lignes", 0),
            "n_par_sorte": {
                "fait": par_sorte.get("fait", 0),
                "regle": par_sorte.get("regle", 0),
                "lignes": par_sorte.get("lignes", 0),
            },
            "n_requetes": len(lots),
            "caracteres_envoyes": caracteres_envoyes,
        }
    )
    return stats


def embarquer_objets(
    dossier: str,
    *,
    a_blanc: bool = True,
    reconstitution_dir: Path | None = None,
    situations_root: Path | None = None,
    objets: Sequence[Objet] | None = None,
    unites: Sequence[str] | None = None,
    cache: dict[str, list[float]] | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    plafond: int = DEFAULT_PLAFOND_REQUETES,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    cache_path = cache_embeddings_objets_path(dossier, reconstitution_dir=reconstitution_dir)
    loaded = _charger_objets(
        dossier, reconstitution_dir=reconstitution_dir, objets=objets
    )
    if unites is not None:
        cibles = {str(uid) for uid in unites}
        loaded = [objet for objet in loaded if objet.unite_id in cibles]
    present = cache if cache is not None else charger_cache(cache_path)
    stats = compter_objets_a_blanc(
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

    root_situations = situations_root or DEFAULT_SITUATIONS_ROOT
    compteur = make_compteur_embeddings_situations(
        situations_root=root_situations, plafond=plafond
    )
    if plafond is not None:
        compteur.definir_plafond(plafond)

    working = dict(present)
    ajoutes: dict[str, list[float]] = {}
    appels = 0
    for lot in _lots_objets(pending, batch_size=batch_size):
        textes = [objet.text[:MAX_CAR_TEXTE_OBJET] for objet in lot]
        vectors = _embed_lot(
            textes, embed_fn=embed_fn, compteur=compteur, dossier=dossier
        )
        for objet, vector in zip(lot, vectors):
            ajoutes[objet.empreinte] = vector
            working[objet.empreinte] = vector
        working = _fusionner_cache_objets(cache_path, ajoutes)
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
    parser = argparse.ArgumentParser(
        description="Embeddings Mistral des objets de reconstitution"
    )
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
    parser.add_argument("--reconstitution-dir", type=Path, default=None)
    parser.add_argument("--situations-root", type=Path, default=None)
    parser.add_argument("--plafond", type=int, default=DEFAULT_PLAFOND_REQUETES)
    args = parser.parse_args(argv)
    if args.embarquer and args.a_blanc:
        parser.error("--a-blanc et --embarquer sont exclusifs")
    a_blanc = not args.embarquer
    stats = embarquer_objets(
        args.dossier,
        a_blanc=a_blanc,
        reconstitution_dir=args.reconstitution_dir,
        situations_root=args.situations_root,
        plafond=args.plafond,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
