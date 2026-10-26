#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrateur — couche brute MOREAU (ingestion + couverture, sans mesure)."""
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

from banc_realite.config import export_runtime_env, write_banc_config_files
from banc_realite.couche_brute_moreau import (
    _duplication_greenloop,
    build_couche_brute_moreau,
    controle_couverture_retroactif,
    prechauffer_questions_moreau,
    write_rapport,
    write_rapport_couverture_decoupe,
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
    start_cluster()
    try:
        connection = connect_existing()
        greenloop_controle = controle_couverture_retroactif(connection, "greenloop")
        build = build_couche_brute_moreau(connection)
        questions = prechauffer_questions_moreau(
            appels_deja=int(build.get("appels_embedding_session") or 0)
        )
        write_rapport(build, questions)
        write_rapport_couverture_decoupe(build, greenloop_controle)
    finally:
        stop_cluster()

    code, tests_summary = _run_tests()
    passed = 0
    if " passed" in tests_summary:
        try:
            passed = int(tests_summary.split(" passed")[0].rsplit(" ", 1)[-1])
        except ValueError:
            passed = 0

    rec = build.get("recouvrement") or {}
    dup = build.get("duplication") or {}
    ok = (
        code == 0
        and build.get("status") != "plafond_atteint"
        and not questions.get("plafond_atteint")
        and build.get("citations_retrouvees") == "42/42"
    )
    if ok:
        gl_dup = _duplication_greenloop()
        payload = {
            "status": "complete",
            "passage_chars": rec.get("max_chars_passage"),
            "recouvrement_chars": rec.get("recouvrement_chars"),
            "moreau": build.get("citations_retrouvees"),
            "passages": build.get("passages"),
            "duplication_pct": dup.get("duplication_pct"),
            "duplication_greenloop_pct": gl_dup.get("duplication_pct"),
            "pieces_un_seul_passage": build.get("pieces_un_seul_passage"),
            "appels_embedding": build.get("appels_embedding_session", 0)
            + questions.get("warmed", 0),
            "tests": f"{passed} verts",
        }
        GROK_RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    print(
        json.dumps(
            {
                "status": "incomplete",
                "build": build.get("status"),
                "tests": tests_summary,
            },
            ensure_ascii=False,
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
