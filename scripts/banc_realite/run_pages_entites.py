#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CLI — génération des pages Markdown par entité."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.pages_entites import generer_pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pages Markdown par entité (reconstitution guidée)")
    parser.add_argument(
        "--reconstitution-dir",
        type=Path,
        required=True,
        help="Répertoire R (registre.json, entites/, echecs.json)",
    )
    parser.add_argument(
        "--ontologie",
        type=Path,
        required=True,
        help="Fichier ontologie.v1.json",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date ISO pour l'en-tête (déterminisme des tests)",
    )
    args = parser.parse_args(argv)
    ontologie = json.loads(args.ontologie.read_text(encoding="utf-8"))
    resultat = generer_pages(args.reconstitution_dir, ontologie, date_gen=args.date)
    print(json.dumps(resultat, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
