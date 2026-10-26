#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrateur — étage 3 sur les couches brute seule et brute+fine."""
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

from banc_realite.config import BANC_ROOT, export_runtime_env, write_banc_config_files
from banc_realite.etage3 import make_relay_chat_fn
from banc_realite.etage3_couches import (
    APPELS_PATH,
    ARRET_PATH,
    CONFIGS,
    INVITES_PATH,
    MAX_CALLS,
    NOTATIONS_PATH,
    RAPPORT_PATH,
    RESULT_PATH,
    SORTIES_ETAGE3_COUCHES,
    compile_and_prepare,
    load_invites_reponses,
    run_juge,
    run_noter,
    run_reponses,
    write_arret,
    write_report,
)
from banc_realite.infra import connect_existing, start_cluster, stop_cluster

PYTHON = sys.executable


def _run_tests() -> tuple[int, str]:
    proc = subprocess.run(
        [
            PYTHON,
            "-m",
            "pytest",
            "tests/banc_realite/test_etage3_couches.py",
            "tests/banc_realite/test_couche_brute_search.py",
            "tests/banc_realite/test_passages.py",
            "-q",
        ],
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
    SORTIES_ETAGE3_COUCHES.mkdir(parents=True, exist_ok=True)
    export_runtime_env()
    write_banc_config_files()

    test_code, tests_summary = _run_tests()
    if test_code != 0:
        write_arret("tests", tests_summary)
        return 1

    start_cluster()
    try:
        connection = connect_existing()
        prep = compile_and_prepare(connection)
    except Exception as exc:  # noqa: BLE001
        write_arret("compilation", str(exc))
        stop_cluster()
        return 1

    chat_fn = make_relay_chat_fn()
    try:
        resp = run_reponses(chat_fn)
    except Exception as exc:  # noqa: BLE001
        write_arret("reponses", str(exc))
        stop_cluster()
        return 1

    try:
        juge = run_juge(chat_fn)
    except Exception as exc:  # noqa: BLE001
        write_arret("juge", str(exc))
        stop_cluster()
        return 1

    appels = juge["appels"]
    if appels > MAX_CALLS:
        write_arret("budget", f"{appels} appels > plafond {MAX_CALLS}")
        stop_cluster()
        return 1

    try:
        notations = run_noter()
    except Exception as exc:  # noqa: BLE001
        write_arret("notation", str(exc))
        stop_cluster()
        return 1

    if not notations.get("complet"):
        write_arret("notation", f"incomplet : {notations.get('manquants')}")
        stop_cluster()
        return 1

    invites = load_invites_reponses()
    try:
        write_report(notations=notations, invites=invites)
    except Exception as exc:  # noqa: BLE001
        write_arret("rapport", str(exc))
        stop_cluster()
        return 1

    RESULT_PATH.write_text(
        json.dumps(
            {
                "status": "complete",
                "appels": appels,
                "configurations": list(CONFIGS),
                "tests": tests_summary,
                "stabilite": notations.get("stabilite"),
                "cadres": prep["cadres"],
                "invites": prep["invites"],
                "files": [
                    str(SORTIES_ETAGE3_COUCHES / "cadres.json"),
                    str(INVITES_PATH),
                    str(APPELS_PATH),
                    str(NOTATIONS_PATH),
                    str(RAPPORT_PATH),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stop_cluster()
    print(
        json.dumps(
            {
                "status": "complete",
                "appels": appels,
                "rapport": str(RAPPORT_PATH),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
