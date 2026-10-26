#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CLI de la passe synchroniser d'un tenant Brain.

Codes de sortie : 0 ok, 1 erreur, 2 passe déjà en cours, 3 plafond.
"""
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
        description="Passe synchroniser d'un tenant Brain (manifeste → lecture → couche brute)"
    )
    parser.add_argument("--tenant", required=True, help="nom du tenant")
    parser.add_argument(
        "--tenants-fichier",
        type=Path,
        metavar="JSON",
        help="fichier BRAIN_TENANTS (même forme que la variable d'environnement)",
    )
    parser.add_argument(
        "--a-blanc",
        action="store_true",
        help="tout calculer, rien écrire, imprimer le résumé",
    )
    parser.add_argument(
        "--declencheur",
        choices=("cli", "timer", "api"),
        default="cli",
        help="origine de la passe, journalisée dans synchronisation.jsonl",
    )
    args = parser.parse_args(argv)

    from banc_realite.synchroniser import PasseDejaEnCours, synchroniser
    from brain.api_brain.tenants import definir_tenants_fichier, resoudre_tenant

    if args.tenants_fichier is not None:
        definir_tenants_fichier(args.tenants_fichier.expanduser().resolve())

    cfg = resoudre_tenant(args.tenant)
    if cfg is None:
        print(f"tenant inconnu : {args.tenant}", file=sys.stderr)
        return 1
    try:
        resultat = synchroniser(
            cfg, a_blanc=args.a_blanc, declencheur=args.declencheur
        )
    except PasseDejaEnCours as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(resultat, ensure_ascii=False, indent=2))
    if resultat.get("plafond_atteint"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
