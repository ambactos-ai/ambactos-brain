#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CLI du constructeur de situations MRFS (ADR 0025).

Un passage traite le dossier entier : tous les fichiers de faits donnés, puis les
rapprochements, puis un instantané de projection complet.
"""
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
EXIT_PIECE_VIDE = 4


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


def resolve_subject_id(dossier: str) -> str:
    """Sujet MRFS du dossier, lu au registre. Jamais le sujet de test par défaut."""
    from banc_realite.constructeur_situations import ALLOWED_LECTEUR_DOSSIERS
    from banc_realite.mrfs_support import (
        BANC_PG_DSN,
        BANC_REALITE_TEST_INSTANCE,
        ensure_test_subject_registered,
        load_subject_registry,
    )

    if dossier not in ALLOWED_LECTEUR_DOSSIERS:
        raise SystemExit(f"dossier non autorisé : {dossier!r}")
    if dossier == BANC_REALITE_TEST_INSTANCE:
        return ensure_test_subject_registered(dsn=BANC_PG_DSN)
    subject_id = load_subject_registry().get(dossier)
    if not subject_id:
        raise SystemExit(f"aucun sujet MRFS au registre pour le dossier {dossier!r}")
    return subject_id


def _faits_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Path]:
    paths = list(args.faits or [])
    if args.faits_dir:
        paths.extend(
            path
            for path in sorted(args.faits_dir.glob("*.json"))
            if not path.name.startswith("groupes")
        )
    if not paths:
        parser.error("--faits ou --faits-dir est requis")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error(f"fichiers de faits absents : {missing}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Constructeur de situations MRFS")
    parser.add_argument("--dossier", required=True, help="greenloop | moreau | banc_realite_test")
    parser.add_argument("--faits", type=Path, action="append", help="Fichier de faits (répétable)")
    parser.add_argument("--faits-dir", type=Path, help="Répertoire des fichiers de faits du dossier")
    parser.add_argument("--groupes", type=Path, help="Fichier JSON de rapprochements")
    parser.add_argument("--pieces-dir", type=Path, help="Répertoire des pièces sources")
    parser.add_argument(
        "--controle",
        action="store_true",
        help="Lecture seule : comptes par pièce relus en base (JSON)",
    )
    args = parser.parse_args()

    from banc_realite.constructeur_situations import (
        PieceVide,
        controle_dossier,
        process_dossier,
    )
    from banc_realite.mrfs_support import BANC_PG_DSN

    if not _pg_ready(BANC_PG_DSN):
        print("PostgreSQL indisponible sur le cluster banc", file=sys.stderr)
        return EXIT_BASE_INDISPONIBLE

    subject_id = resolve_subject_id(args.dossier)
    faits_paths = _faits_paths(args, parser)

    if args.controle:
        report = controle_dossier(
            dossier=args.dossier, subject_id=subject_id, faits_paths=faits_paths
        )
        print(json.dumps(report, ensure_ascii=False))
        return 0

    if not args.pieces_dir:
        parser.error("--pieces-dir est requis hors mode --controle")
    if args.groupes and not args.groupes.is_file():
        parser.error(f"fichier de groupes absent : {args.groupes}")

    try:
        state = process_dossier(
            dossier=args.dossier,
            subject_id=subject_id,
            faits_paths=faits_paths,
            pieces_dir=args.pieces_dir,
            groupes_path=args.groupes,
        )
    except PieceVide as exc:
        print(f"ARRÊT : {exc}", file=sys.stderr)
        return EXIT_PIECE_VIDE
    print(
        json.dumps(
            {
                "dossier": args.dossier,
                "sujet": subject_id,
                "pieces": len(state.journal),
                "faits_publies": len(state.published),
                "rejets": len(state.rejections),
                "enregistrements": len(state.projection_records),
                "instantane": state.snapshot_id,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
