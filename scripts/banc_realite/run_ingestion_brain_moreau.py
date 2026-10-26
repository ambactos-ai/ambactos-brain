#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrateur — verser MOREAU dans Brain (chemin ordinaire ingest_runner)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
for entry in (WORKTREE / "scripts", WORKTREE / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.config import MOREAU_INGESTION_SORTIES, export_runtime_env, write_banc_config_files
from banc_realite.ingest_runner import IngestionConfig
from banc_realite.ingestion_moreau import (
    count_completed_pieces,
    finalize_moreau_controls,
    fidelite_scores,
    run_moreau_brain_ingestion,
)
from banc_realite.infra import connect_existing, start_cluster, stop_cluster

PYTHON = sys.executable
GROK_RESULT = WORKTREE / "_grok-result.json"


def _run_tests() -> tuple[int, str]:
    proc = subprocess.run(
        [PYTHON, "-m", "pytest", "tests/banc_realite/", "-q"],
        cwd=str(WORKTREE),
        capture_output=True,
        text=True,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": f"{WORKTREE / 'src'}:{WORKTREE / 'scripts'}:{WORKTREE}",
        },
    )
    summary = ""
    for line in reversed((proc.stdout or "").splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            break
    if not summary:
        summary = (proc.stdout or proc.stderr or "pytest")[-500:]
    return proc.returncode, summary


def main() -> int:
    export_runtime_env()
    write_banc_config_files()
    MOREAU_INGESTION_SORTIES.mkdir(parents=True, exist_ok=True)
    cfg = IngestionConfig.for_moreau_brain()
    start_cluster()
    try:
        connection = connect_existing()
        if count_completed_pieces(cfg) < 12:
            run_moreau_brain_ingestion(connection)
        result = finalize_moreau_controls(connection)
        ingestion = result["ingestion"]
        fidelite = result["fidelite"]
        coverage = result["coverage"]
        isolation = result["isolation"]
    finally:
        stop_cluster()

    test_code, tests_summary = _run_tests()
    passed = 0
    if " passed" in tests_summary:
        try:
            passed = int(tests_summary.split(" passed")[0].rsplit(" ", 1)[-1])
        except ValueError:
            passed = 0

    scores = result["scores"]
    pieces_done = count_completed_pieces(cfg)
    calls = ingestion.get("appels_modele", 0)
    ok = (
        test_code == 0
        and fidelite.get("status") != "error"
        and isolation.get("isolation_ok")
        and ingestion.get("atomes", 0) > 0
        and pieces_done == 12
        and calls <= cfg.max_calls
    )
    if ok:
        payload = {
            "status": "complete",
            "pieces": pieces_done,
            "atomes": ingestion.get("atomes", 0),
            "appels_modele": ingestion.get("appels_modele", 0),
            "fournisseur": ingestion.get("fournisseur"),
            "embeddings": ingestion.get("embeddings", 0),
            "f1_global": scores["f1_global"],
            "f1_prose": scores["f1_prose"],
            "f1_tabulaire": scores["f1_tabulaire"],
            "or_porte_verbatim": coverage.get("or_porte_verbatim"),
            "or_porte_valeur": coverage.get("or_porte_valeur"),
            "manquantes": coverage.get("manquantes") or [],
            "isolation_ok": isolation.get("isolation_ok"),
            "tests": f"{passed} verts",
        }
        GROK_RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    print(
        json.dumps(
            {
                "status": "incomplete",
                "tests": tests_summary,
                "fidelite_status": fidelite.get("status"),
                "isolation_ok": isolation.get("isolation_ok"),
            },
            ensure_ascii=False,
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
