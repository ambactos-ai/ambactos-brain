#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrateur — verser MOREAU dans mem0 (issue #127 livrable 4)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
for entry in (WORKTREE / "scripts", WORKTREE / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.config import export_runtime_env
from banc_realite.dossiers import dossier_courant
from banc_realite.tiers_mem0 import (
    CURSOR_MODEL,
    Mem0Chemins,
    _compter_journal,
    compter_souvenirs_par_piece,
    contextes,
    ingerer,
    stats_longueur_souvenirs,
    verifier_isolation_mem0,
    write_rapport_ingestion_mem0,
)

PYTHON = sys.executable
GROK_RESULT = WORKTREE / "_grok-result.json"
MOREAU = dossier_courant("moreau")


def _run_tests() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/banc_realite/", "-q"],
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
    chemins = Mem0Chemins.pour(MOREAU)
    chemins.racine.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    journal = chemins.journal_ingestion("cursor")
    bilans = sum(
        1
        for line in journal.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("bilan")
    ) if journal.is_file() else 0
    if bilans < 12:
        ingestion = ingerer(
            executer=True,
            plafond_eur=999.0,
            extracteur="cursor",
            dossier=MOREAU,
        )
        if ingestion.get("status") != "complete" or ingestion.get("pieces_ingerees", 0) < 12:
            print(json.dumps({"status": "incomplete", "ingestion": ingestion}, ensure_ascii=False))
            return 1
    else:
        comptes = _compter_journal(journal, chemins.journal_contextes)
        par_piece = compter_souvenirs_par_piece(MOREAU, extracteur="cursor", model=CURSOR_MODEL)
        ingestion = {
            "status": "complete",
            "dossier": MOREAU.nom,
            "pieces_ingerees": bilans,
            "modele": CURSOR_MODEL,
            "extracteur": "cursor",
            "souvenirs_par_piece": par_piece,
            "souvenirs": sum(par_piece.values()),
            **comptes,
            "duree_s": 0,
        }

    contextes_result = contextes(k=20, extracteur="cursor", dossier=MOREAU)
    if contextes_result.get("status") != "complete" or contextes_result.get("questions", 0) < 50:
        print(json.dumps({"status": "incomplete", "contextes": contextes_result}, ensure_ascii=False))
        return 1

    isolation = verifier_isolation_mem0(extracteur="cursor", model=CURSOR_MODEL)
    longueurs = stats_longueur_souvenirs(MOREAU, extracteur="cursor", model=CURSOR_MODEL)
    comptes = _compter_journal(
        chemins.journal_ingestion("cursor"),
        chemins.journal_contextes,
    )
    ingestion = {**ingestion, **comptes}

    write_rapport_ingestion_mem0(
        dossier=MOREAU,
        extracteur="cursor",
        model=CURSOR_MODEL,
        ingestion=ingestion,
        contextes_result=contextes_result,
        isolation=isolation,
        longueurs=longueurs,
    )

    test_code, tests_summary = _run_tests()
    passed = 0
    if " passed" in tests_summary:
        try:
            passed = int(tests_summary.split(" passed")[0].rsplit(" ", 1)[-1])
        except ValueError:
            passed = 0

    ok = (
        test_code == 0
        and ingestion.get("pieces_ingerees", 0) == 12
        and contextes_result.get("questions", 0) == 50
        and isolation.get("isolation_ok")
        and ingestion.get("souvenirs", 0) > 0
    )
    if ok:
        payload = {
            "status": "complete",
            "pieces": 12,
            "morceaux": ingestion.get("appels_extraction", 0),
            "souvenirs": ingestion.get("souvenirs", 0),
            "appels_extraction": ingestion.get("appels_extraction", 0),
            "embeddings": comptes.get("embeddings", 0),
            "duree_s": round(time.monotonic() - started, 3),
            "echecs": comptes.get("echecs", 0),
            "souvenir_longueur": longueurs,
            "contextes": 50,
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
                "ingestion": ingestion,
                "contextes": contextes_result,
                "isolation_ok": isolation.get("isolation_ok"),
            },
            ensure_ascii=False,
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
