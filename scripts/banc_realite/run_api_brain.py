#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Lance l'API Brain v0 (uvicorn)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "scripts", ROOT / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="API Brain v0 par tenant")
    parser.add_argument("--hote", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=56110)
    parser.add_argument(
        "--prechauffer",
        nargs="*",
        metavar="TENANT",
        help="précharge l'état mémoire des tenants listés avant d'écouter",
    )
    parser.add_argument(
        "--tenants-fichier",
        type=Path,
        metavar="JSON",
        help="fichier BRAIN_TENANTS (même forme que la variable d'environnement)",
    )
    args = parser.parse_args(argv)

    import uvicorn

    from brain.api_brain.app import creer_app, prechauffer_tenants
    from brain.api_brain.tenants import definir_tenants_fichier

    if args.tenants_fichier is not None:
        definir_tenants_fichier(args.tenants_fichier.expanduser().resolve())

    if args.prechauffer:
        prechauffer_tenants(list(args.prechauffer))
    app = creer_app()
    uvicorn.run(app, host=args.hote, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
