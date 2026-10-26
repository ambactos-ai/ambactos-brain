#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Lance le serveur MCP mince branché sur l'API Brain v0.

En transport HTTP, l'écoute est limitée au loopback (127.0.0.1, ::1, localhost).
``BRAIN_MCP_JETON_ENTRANT`` protège l'accès via le middleware Bearer derrière une
passerelle TLS ; il n'autorise pas un bind HTTP brut hors loopback.
"""
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
    parser = argparse.ArgumentParser(
        description="Serveur MCP mince sur l'API Brain v0 (stdio ou HTTP streamable)"
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8646")
    parser.add_argument("--tenant", required=True)
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="stdio (défaut) ou http (streamable HTTP)",
    )
    parser.add_argument(
        "--hote",
        default="127.0.0.1",
        help="interface d'écoute HTTP (loopback uniquement ; jeton entrant = middleware derrière TLS)",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    from brain.mcp_api import creer_serveur, lire_jeton
    from brain.mcp_api.serveur import lancer_http_streamable, verifier_demarrage_http

    try:
        jeton = lire_jeton(args.tenant)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    serveur = creer_serveur(api_url=args.api_url, tenant=args.tenant, jeton=jeton)
    if args.transport == "stdio":
        serveur.run(transport="stdio")
    else:
        message = verifier_demarrage_http(args.hote)
        if message:
            print(message, file=sys.stderr)
            return 2
        import anyio

        anyio.run(
            lambda: lancer_http_streamable(serveur, host=args.hote, port=args.port)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
