#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Lance l'ingestion MRFS étape 2 (pièces uniquement, cluster 56102)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_tests() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/mrfs",
        "tests/banc_realite",
        "-q",
        "--tb=no",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingestion MRFS banc — étape 2")
    parser.add_argument(
        "--dossier",
        choices=("greenloop", "moreau", "all"),
        default="all",
        help="Dossier à ingérer (défaut : all)",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Ne pas appeler le modèle — rapport et résultat seulement",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Ne pas relancer pytest à la fin",
    )
    args = parser.parse_args()

    from banc_realite.controle_ingestion_mrfs import (
        write_grok_result,
        write_rapport_ingestion_mrfs,
    )
    from banc_realite.ingestion_mrfs import (
        IngestionMrfsConfig,
        prepare_ingestion,
        run_full_ingestion,
        run_ingestion_mrfs,

        count_model_calls,
    )

    cfg = IngestionMrfsConfig()
    ingestion: dict | None = None
    if not args.skip_ingestion:
        registry, cfg = prepare_ingestion(cfg=cfg)
        if args.dossier == "all":
            ingestion = run_full_ingestion(cfg=cfg)
        else:
            ingestion = {
                "dossiers": {
                    args.dossier: run_ingestion_mrfs(
                        args.dossier,
                        cfg=cfg,
                        global_calls=lambda: count_model_calls(cfg),
                    )
                },
                "appels_modele": count_model_calls(cfg),
            }

    tests_passed = 0
    if not args.skip_tests:
        code = _run_tests()
        if code != 0:
            print(f"pytest failed with exit code {code}", file=sys.stderr)
            return code
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/mrfs", "tests/banc_realite", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        for line in reversed((out.stdout or "").splitlines()):
            if "passed" in line:
                try:
                    tests_passed = int(line.strip().split()[0])
                except (ValueError, IndexError):
                    tests_passed = 0
                break

    write_rapport_ingestion_mrfs(cfg=cfg, ingestion=ingestion)
    write_grok_result(cfg=cfg, tests_passed=tests_passed)
    root_result = ROOT / "_grok-result.json"
    root_result.write_text(cfg.result_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(cfg.result_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
