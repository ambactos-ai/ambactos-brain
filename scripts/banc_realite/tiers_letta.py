# SPDX-License-Identifier: AGPL-3.0-only
"""Ingestion et contextes Letta archive pour le banc comparatif (venv tiers dédié)."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

from banc_realite.compat_moteur import EMBED_DIM, load_api_key

from .config import BANC_ROOT, export_runtime_env

COMPARATIF_DIR = BANC_ROOT / "sorties" / "comparatif"

LETTA_ROOT = Path.home() / ".agents" / "var" / "banc-realite" / "letta"
LETTA_DB = LETTA_ROOT / "letta.db"
ARRET_PATH = COMPARATIF_DIR / "_letta-arret.json"
JOURNAL_INGESTION = COMPARATIF_DIR / "appels-ingestion-letta.jsonl"
AGENT_NAME = "greenloop"
MIN_PORT = 56200
MISTRAL_BASE = "https://api.mistral.ai/v1"


def _load_questions() -> list[dict[str, Any]]:
    from .measure_etage2 import load_questions

    return load_questions()


def comparatif_context_path(etiquette: str) -> Path:
    return COMPARATIF_DIR / f"contextes-{etiquette}.jsonl"


def write_arret(cause: str) -> None:
    COMPARATIF_DIR.mkdir(parents=True, exist_ok=True)
    ARRET_PATH.write_text(json.dumps({"status": "arret", "cause": cause}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _port_libre(start: int = MIN_PORT) -> int:
    port = start
    while port < start + 200:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SOCKET_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"aucun port libre ≥ {start}")


def _attendre_serveur(port: int, *, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/v1/health")
            response = conn.getresponse()
            response.read()
            conn.close()
            if response.status == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Letta ne répond pas sur le port {port}")


def _state_path() -> Path:
    return LETTA_ROOT / "server-state.json"


def lancer_serveur() -> dict[str, Any]:
    LETTA_ROOT.mkdir(parents=True, exist_ok=True)
    port = _port_libre()
    api_key = load_api_key()
    env = {
        **dict(__import__("os").environ),
        "OPENAI_API_KEY": api_key,
        "OPENAI_API_BASE": MISTRAL_BASE,
        "LETTA_EMBEDDING_MODEL": "mistral-embed",
        "LETTA_EMBEDDING_DIM": str(EMBED_DIM),
    }
    log_path = LETTA_ROOT / "server.log"
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "letta",
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--db",
                str(LETTA_DB),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    try:
        _attendre_serveur(port)
    except Exception as exc:
        proc.terminate()
        proc.wait(timeout=5)
        raise exc
    state = {"pid": proc.pid, "port": port, "db": str(LETTA_DB)}
    _state_path().write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def arreter_serveur() -> None:
    state_path = _state_path()
    if not state_path.is_file():
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    pid = int(state.get("pid") or 0)
    if pid > 0:
        try:
            import os
            import signal

            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
        except OSError:
            pass


def _passages_brute(connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, source_ref, content, metadata
        FROM public.memory_chunks
        WHERE source_type = 'corpus'
        ORDER BY source_ref, id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _journaliser(chemin: Path, row: dict[str, Any]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _passages_journalises() -> set[str]:
    done: set[str] = set()
    if not JOURNAL_INGESTION.is_file():
        return done
    for line in JOURNAL_INGESTION.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        passage_id = row.get("passage_id")
        if passage_id:
            done.add(str(passage_id))
    return done


def ingerer(*, connection) -> dict[str, Any]:
    from letta_client import Letta

    state = json.loads(_state_path().read_text(encoding="utf-8"))
    client = Letta(base_url=f"http://127.0.0.1:{state['port']}")
    agents = client.agents.list()
    agent = next((a for a in agents if a.name == AGENT_NAME), None)
    if agent is None:
        agent = client.agents.create(name=AGENT_NAME)
    done = _passages_journalises()
    inserted = 0
    for row in _passages_brute(connection):
        passage_id = str(row["id"])
        if passage_id in done:
            continue
        started = time.monotonic()
        client.agents.passages.create(
            agent_id=agent.id,
            text=row.get("content") or "",
            metadata={"piece": row.get("source_ref") or "", "passage_id": passage_id},
        )
        inserted += 1
        _journaliser(
            JOURNAL_INGESTION,
            {
                "fournisseur": "letta_archive",
                "passage_id": passage_id,
                "piece": row.get("source_ref") or "",
                "secondes": round(time.monotonic() - started, 3),
            },
        )
    return {"status": "complete", "passages_inseres": inserted, "agent_id": agent.id}


def contextes(*, k: int) -> dict[str, Any]:
    from letta_client import Letta

    state = json.loads(_state_path().read_text(encoding="utf-8"))
    client = Letta(base_url=f"http://127.0.0.1:{state['port']}")
    agents = client.agents.list()
    agent = next(a for a in agents if a.name == AGENT_NAME)
    out_path = comparatif_context_path("letta_archive")
    COMPARATIF_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"meta": {"k": k}}, ensure_ascii=False) + "\n")
        for question in _load_questions():
            hits = client.agents.passages.search(
                agent_id=agent.id,
                query=question["question"],
                limit=k,
            )
            elements = []
            for rank, hit in enumerate(hits, 1):
                metadata = hit.metadata or {}
                elements.append(
                    {
                        "rank": rank,
                        "score": float(getattr(hit, "score", 0.0) or 0.0),
                        "text": hit.text or "",
                        "source": metadata.get("piece") or "",
                        "provenance": metadata,
                    }
                )
            handle.write(
                json.dumps({"question_id": question["id"], "elements": elements}, ensure_ascii=False) + "\n"
            )
    return {"status": "complete", "questions": len(_load_questions()), "sortie": str(out_path)}


def _try_import_letta_client() -> str | None:
    try:
        import letta_client  # noqa: F401
    except Exception as exc:
        return str(exc)
    return None


def _try_demarrer_serveur_archive() -> tuple[dict[str, Any] | None, str | None]:
    """Tente le serveur archive compatible letta_client ; échec → cause pour _letta-arret.json."""
    err = _try_import_letta_client()
    if err:
        return None, f"import letta_client impossible : {err}"
    # PyPI `letta` 0.32.x = Letta Code CLI, pas le serveur REST archive (SQLite, passages).
    try:
        import letta  # noqa: F401
    except ModuleNotFoundError:
        return None, (
            "module Python `letta` absent : le paquet PyPI `letta` installé est Letta Code CLI, "
            "pas le serveur REST archive attendu par letta_client (passages d'archive)."
        )
    return None, "serveur archive non implémenté dans ce lot"


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiers Letta archive — banc comparatif")
    sub = parser.add_subparsers(dest="commande", required=True)
    sub.add_parser("serveur", help="Lancer letta server sur un port libre ≥ 56200")
    sub.add_parser("ingerer", help="Insérer les passages memory_chunks en archive Letta")
    ctx = sub.add_parser("contextes", help="Recherche archive → contextes-letta_archive.jsonl")
    ctx.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    if args.commande != "serveur":
        export_runtime_env()
    if args.commande == "serveur":
        state, cause = _try_demarrer_serveur_archive()
        if cause:
            write_arret(cause)
            print(json.dumps({"status": "arret", "cause": cause}, ensure_ascii=False))
            return 1
        try:
            state = lancer_serveur()
            print(json.dumps({"status": "ok", **state}, ensure_ascii=False))
            return 0
        except Exception as exc:
            write_arret(str(exc))
            print(json.dumps({"status": "arret", "cause": str(exc)}, ensure_ascii=False))
            return 1
    err = _try_import_letta_client()
    if err:
        write_arret(f"import letta_client impossible : {err}")
        print(json.dumps({"status": "arret", "cause": err}, ensure_ascii=False))
        return 1
    if args.commande == "ingerer":
        from .infra import connect_existing, start_cluster

        start_cluster()
        connection = connect_existing()
        print(json.dumps(ingerer(connection=connection), ensure_ascii=False))
        return 0
    print(json.dumps(contextes(k=args.k), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
