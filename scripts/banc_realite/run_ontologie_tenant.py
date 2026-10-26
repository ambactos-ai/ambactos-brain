#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CLI proposition d'ontologie tenant v0."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

EXIT_ECHEC = 1


def _afficher_comptes(ontologie: dict[str, Any]) -> None:
    blocs = (
        ("familles", "familles"),
        ("types_entites", "types"),
        ("entites", "entités"),
        ("relations", "relations"),
        ("concepts", "concepts"),
    )
    for cle, label in blocs:
        print(f"  {label}: {len(ontologie.get(cle) or [])}")
    par_sujet = Counter(q.get("sujet") or "?" for q in ontologie.get("questions") or [])
    print("  questions:")
    for sujet in sorted(par_sujet):
        print(f"    {sujet}: {par_sujet[sujet]}")


def proposer(
    *,
    dossier: str,
    drive_dir: Path,
    entites_dir: Path,
    sortie: Path,
) -> int:
    from banc_realite.lecture_drive import ecrire_json_atomique
    from banc_realite.ontologie_tenant import (
        charger_aiguillage,
        charger_entrees_drive,
        charger_groupes,
        charger_registre_entites,
        proposer_ontologie,
    )

    registre_path = drive_dir / "registre.json"
    aiguillage_path = drive_dir / "aiguillage.json"
    entites_registre_path = entites_dir / "registre.json"
    groupes_path = entites_dir / "groupes-reliquat.json"
    faits_dir = entites_dir / "entites"

    if not registre_path.is_file():
        print(f"registre Drive absent : {registre_path}", file=sys.stderr)
        return EXIT_ECHEC
    if not aiguillage_path.is_file():
        print(f"aiguillage absent : {aiguillage_path}", file=sys.stderr)
        return EXIT_ECHEC
    if not entites_registre_path.is_file():
        print(f"registre entités absent : {entites_registre_path}", file=sys.stderr)
        return EXIT_ECHEC

    drive_registre = charger_entrees_drive(registre_path)
    aiguillage = charger_aiguillage(aiguillage_path)
    entites_registre = charger_registre_entites(entites_registre_path)
    groupes = (
        charger_groupes(groupes_path)
        if groupes_path.is_file()
        else {"groupes": []}
    )

    genere_le = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    ontologie = proposer_ontologie(
        drive_registre,
        aiguillage,
        entites_registre,
        groupes,
        faits_dir,
        dossier=dossier,
        genere_le=genere_le,
        sources={
            "drive_registre": str(registre_path),
            "entites_registre": str(entites_registre_path),
        },
        drive_dir=drive_dir,
    )
    sortie.parent.mkdir(parents=True, exist_ok=True)
    ecrire_json_atomique(sortie, ontologie)
    print(f"=== Ontologie v0 — {dossier} ===")
    _afficher_comptes(ontologie)
    print(f"écrit : {sortie}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Proposition d'ontologie tenant")
    sous = parser.add_subparsers(dest="commande", required=True)

    cmd = sous.add_parser("proposer", help="proposer ontologie v0")
    cmd.add_argument("--dossier", required=True)
    cmd.add_argument("--drive-dir", type=Path, required=True)
    cmd.add_argument("--entites-dir", type=Path, required=True)
    cmd.add_argument("--sortie", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.commande == "proposer":
        return proposer(
            dossier=args.dossier,
            drive_dir=args.drive_dir,
            entites_dir=args.entites_dir,
            sortie=args.sortie,
        )
    return EXIT_ECHEC


if __name__ == "__main__":
    raise SystemExit(main())
