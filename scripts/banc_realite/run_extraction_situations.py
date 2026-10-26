#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CLI d'extraction de faits et rapprochement pour le constructeur MRFS."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

EXIT_BASE_INDISPONIBLE = 1
EXIT_PLAFOND = 3
EXIT_PIECE_VIDE = 4
EXIT_FOURNISSEUR = 5


def _pg_ready(dsn: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(dsn)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5432
    result = subprocess.run(
        ["pg_isready", "-h", host, "-p", str(port)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _parse_refaire(values: list[str] | None) -> set[tuple[str, int]]:
    refaire: set[tuple[str, int]] = set()
    for item in values or []:
        if ":" not in item:
            raise SystemExit(f"--refaire invalide (attendu piece:morceau) : {item!r}")
        piece, morceau_raw = item.rsplit(":", 1)
        try:
            morceau = int(morceau_raw)
        except ValueError as exc:
            raise SystemExit(f"morceau invalide dans --refaire : {item!r}") from exc
        refaire.add((piece, morceau))
    return refaire


def _journaliser_piece(
    dossier: str,
    line: dict[str, object],
    *,
    constructeur_root: Path,
) -> None:
    from banc_realite.extraction_situations import journal_pieces_path

    path = journal_pieces_path(dossier, constructeur_root=constructeur_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extraction de faits MRFS par morceaux")
    parser.add_argument("--dossier", required=True)
    parser.add_argument(
        "--pieces",
        default="toutes",
        help="Liste séparée par des virgules ou 'toutes'",
    )
    parser.add_argument("--max-pieces", type=int, default=0)
    parser.add_argument("--a-blanc", action="store_true")
    parser.add_argument("--rapprocher", action="store_true")
    parser.add_argument(
        "--revalider",
        action="store_true",
        help="Rejoue la validation sur les réponses conservées, sans appel, puis republie",
    )
    parser.add_argument(
        "--refaire",
        action="append",
        help="Force un morceau (piece:morceau), répétable",
    )
    parser.add_argument(
        "--plafond",
        type=int,
        help="Plafond d'appels cursor-agent (défaut 85, journalisé si explicite)",
    )
    parser.add_argument(
        "--constructeur-root",
        type=Path,
        help="Racine injectable des artefacts constructeur",
    )
    args = parser.parse_args()

    from banc_realite.constructeur_situations import PieceVide, process_dossier
    from banc_realite.dossiers import dossier_courant
    from banc_realite.extraction_situations import (
        DEFAULT_CONSTRUCTEUR_ROOT,
        build_chat_factory_builder,
        build_rapprochement_chat_factory,
        bilan_rapprochement,
        estimer_a_blanc,
        extraire_piece,
        faits_paths_dossier,
        groupes_path,
        lister_pieces,
        rapprocher_dossier,
        revalider_dossier,
    )
    from banc_realite.mrfs_support import BANC_PG_DSN
    from banc_realite.plafond_appels import PlafondAtteint, make_compteur
    from banc_realite.run_constructeur_situations import resolve_subject_id

    constructeur_root = args.constructeur_root or DEFAULT_CONSTRUCTEUR_ROOT
    pieces = lister_pieces(args.dossier, args.pieces)
    if args.max_pieces and args.max_pieces > 0:
        pieces = pieces[: args.max_pieces]

    if args.a_blanc:
        report = estimer_a_blanc(
            args.dossier, pieces, constructeur_root=constructeur_root
        )
        print(json.dumps(report, ensure_ascii=False))
        return 0

    if not _pg_ready(BANC_PG_DSN):
        print("PostgreSQL indisponible sur le cluster banc", file=sys.stderr)
        return EXIT_BASE_INDISPONIBLE

    subject_id = resolve_subject_id(args.dossier)
    compteur = make_compteur(
        constructeur_root=constructeur_root,
        plafond=args.plafond,
        plafond_explicite=args.plafond is not None,
    )
    desc = dossier_courant(args.dossier)
    pieces_dir = desc.corpus_dir
    refaire = _parse_refaire(args.refaire)

    if args.revalider:
        bilan = revalider_dossier(args.dossier, constructeur_root=constructeur_root)
        groupes = groupes_path(args.dossier, constructeur_root=constructeur_root)
        try:
            state = process_dossier(
                dossier=args.dossier,
                subject_id=subject_id,
                faits_paths=faits_paths_dossier(
                    args.dossier, constructeur_root=constructeur_root
                ),
                pieces_dir=pieces_dir,
                groupes_path=groupes if groupes.is_file() else None,
            )
        except PieceVide as exc:
            print(f"ARRÊT : {exc}", file=sys.stderr)
            return EXIT_PIECE_VIDE
        print(
            json.dumps(
                {
                    "dossier": args.dossier,
                    "revalidation": bilan,
                    "faits_publies": len(state.published),
                    "instantane": state.snapshot_id,
                    "appels": compteur.lire(),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.rapprocher:
        faits_paths = faits_paths_dossier(
            args.dossier, constructeur_root=constructeur_root
        )
        if not faits_paths:
            print("aucun fichier de faits pour le rapprochement", file=sys.stderr)
            return EXIT_PIECE_VIDE
        try:
            state = process_dossier(
                dossier=args.dossier,
                subject_id=subject_id,
                faits_paths=faits_paths,
                pieces_dir=pieces_dir,
                publier_instantane=False,
            )
        except PieceVide as exc:
            print(f"ARRÊT : {exc}", file=sys.stderr)
            return EXIT_PIECE_VIDE
        try:
            rapprocher_dossier(
                dossier=args.dossier,
                published=state.published,
                chat_factory_builder=build_rapprochement_chat_factory(
                    compteur, dossier=args.dossier
                ),
                constructeur_root=constructeur_root,
            )
        except PlafondAtteint as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_PLAFOND
        except RuntimeError as exc:
            print(f"ARRÊT fournisseur : {exc}", file=sys.stderr)
            return EXIT_FOURNISSEUR
        groupes = groupes_path(args.dossier, constructeur_root=constructeur_root)
        try:
            state = process_dossier(
                dossier=args.dossier,
                subject_id=subject_id,
                faits_paths=faits_paths,
                pieces_dir=pieces_dir,
                groupes_path=groupes if groupes.is_file() else None,
            )
        except PieceVide as exc:
            print(f"ARRÊT : {exc}", file=sys.stderr)
            return EXIT_PIECE_VIDE
        print(
            json.dumps(
                {
                    "dossier": args.dossier,
                    "rapprochement": bilan_rapprochement(
                        args.dossier, constructeur_root=constructeur_root
                    ),
                    "groupes": str(groupes),
                    "alternatives": sum(
                        1 for line in state.journal if line.get("alternatives")
                    ),
                    "appels": compteur.lire(),
                },
                ensure_ascii=False,
            )
        )
        return 0

    chat_builder = build_chat_factory_builder(
        compteur, dossier=args.dossier
    )
    total_appels = 0
    for piece in pieces:
        try:
            _etat, incomplete = extraire_piece(
                dossier=args.dossier,
                piece=piece,
                chat_factory_builder=chat_builder,
                refaire=refaire or None,
                constructeur_root=constructeur_root,
            )
        except PlafondAtteint as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_PLAFOND
        except RuntimeError as exc:
            # Panne du fournisseur : on s'arrête, on ne brûle pas le plafond sur la suite.
            print(f"ARRÊT fournisseur : {exc}", file=sys.stderr)
            return EXIT_FOURNISSEUR

        faits_paths = faits_paths_dossier(
            args.dossier, constructeur_root=constructeur_root
        )
        try:
            state = process_dossier(
                dossier=args.dossier,
                subject_id=subject_id,
                faits_paths=faits_paths,
                pieces_dir=pieces_dir,
                publier_instantane=piece == pieces[-1],
            )
        except PieceVide as exc:
            print(f"ARRÊT : {exc}", file=sys.stderr)
            return EXIT_PIECE_VIDE

        journal_line = next(
            (line for line in state.journal if line.get("piece") == piece), None
        )
        if journal_line is None:
            # Aucun fichier de faits pour la pièce : tous ses morceaux ont échoué.
            print(f"ARRÊT : aucun fait extrait pour {piece}", file=sys.stderr)
            return EXIT_PIECE_VIDE
        journal_line = {
            **journal_line,
            "dossier": args.dossier,
            "incomplete": incomplete,
            "appels": compteur.lire(),
        }
        _journaliser_piece(
            args.dossier, journal_line, constructeur_root=constructeur_root
        )
        total_appels = compteur.lire()["emis"]

    print(
        json.dumps(
            {
                "dossier": args.dossier,
                "pieces": len(pieces),
                "appels": compteur.lire(),
                "total_appels": total_appels,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
