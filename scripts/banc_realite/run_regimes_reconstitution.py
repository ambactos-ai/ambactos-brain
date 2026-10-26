#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CLI — reconstitution unitaire selon les régimes par famille d'un tenant."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "scripts", ROOT / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstitution unitaire d'un tenant Brain (régimes par famille)"
    )
    sous = parser.add_subparsers(dest="commande", required=True)
    refaire = sous.add_parser(
        "refaire",
        help="reconstituer une unité, ou toutes les unités périmées",
    )
    refaire.add_argument("--tenant", required=True, help="nom du tenant")
    refaire.add_argument(
        "--tenants-fichier",
        type=Path,
        metavar="JSON",
        help="fichier BRAIN_TENANTS (même forme que la variable d'environnement)",
    )
    cible = refaire.add_mutually_exclusive_group(required=True)
    cible.add_argument("--unite", metavar="ID", help="identifiant d'unité")
    cible.add_argument(
        "--perimees",
        action="store_true",
        help="toutes les unités marquées périmées",
    )
    refaire.add_argument(
        "--a-blanc",
        action="store_true",
        help="résoudre régime et lister, sans reconstituer",
    )
    refaire.add_argument(
        "--delai",
        type=int,
        default=900,
        help="délai par appel, en secondes (défaut 900)",
    )
    args = parser.parse_args(argv)

    from banc_realite.regimes_reconstitution import (
        make_compteur_reconstitution,
        reconstituer_unite_tenant,
    )
    from banc_realite.reconstitution_entites import unites_perimees
    from brain.api_brain.tenants import definir_tenants_fichier, resoudre_tenant

    if args.tenants_fichier is not None:
        definir_tenants_fichier(args.tenants_fichier.expanduser().resolve())

    cfg = resoudre_tenant(args.tenant)
    if cfg is None:
        print(f"tenant inconnu : {args.tenant}", file=sys.stderr)
        return 1

    rec_path = cfg.reconstitution / "registre.json"
    if not rec_path.is_file():
        print(f"registre.json absent : {rec_path}", file=sys.stderr)
        return 1
    registre = json.loads(rec_path.read_text(encoding="utf-8"))
    if args.unite:
        cibles = [args.unite]
    else:
        cibles = [str(u["id"]) for u in unites_perimees(registre)]

    compteur = None
    if not args.a_blanc:
        try:
            compteur = make_compteur_reconstitution(cfg)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    resultats: list[dict] = []
    code = 0
    for uid in cibles:
        try:
            resultat = reconstituer_unite_tenant(
                cfg,
                uid,
                compteur=compteur,
                delai_s=args.delai,
                declencheur="cli",
                a_blanc=args.a_blanc,
            )
        except Exception as exc:
            print(f"{uid} : {exc!r}", file=sys.stderr)
            code = 1
            continue
        resultats.append(resultat)
        statut = resultat.get("statut")
        if statut == "plafond":
            code = 3
            break
        if statut in {"echec", "echec_publication"}:
            code = 1

    print(json.dumps(resultats, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
