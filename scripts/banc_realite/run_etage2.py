#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrate étage 2 bras C — parity verify then restitution measurement."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[2]
for entry in (WORKTREE / "scripts", WORKTREE / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.config import (
    ARRET,
    BANC_ROOT,
    GOLD,
    REPORT,
    RESULT,
    SORTIES,
    SMOKE_QUESTION_ID,
    export_runtime_env,
    write_banc_config_files,
)
from banc_realite.infra import start_cluster, stop_cluster
from banc_realite.measure_etage2 import (
    PRECHAUFFAGE_STATUS_PATH,
    measure_output_dir,
    measure_status_path,
    prechauffer_questions,
    questions_journal_path,
)
from banc_realite.partie_a import run_partie_a, verify_parity_readonly

PYTHON = sys.executable


def write_arret(step: str, why: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"status": "arret", "a_quelle_etape": step, "pourquoi": why}
    if extra:
        payload.update(extra)
    ARRET.write_text(json.dumps(payload, ensure_ascii=False) + "\n")


def smoke_question_text() -> str:
    questions = json.loads((GOLD / "questions.json").read_text(encoding="utf-8"))["questions"]
    for question in questions:
        if question["id"] == SMOKE_QUESTION_ID:
            return question["question"]
    raise RuntimeError(f"smoke question {SMOKE_QUESTION_ID} not found")


def _systemd_prefix() -> list[str]:
    return [
        "systemd-run",
        "--user",
        "--wait",
        "--collect",
        "-p",
        "MemoryMax=3G",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "KillMode=control-group",
        "-p",
        "RuntimeMaxSec=1800",
    ]


def _runtime_env() -> dict[str, str]:
    export_runtime_env()
    env = {
        "PYTHONPATH": f"{WORKTREE / 'src'}:{WORKTREE / 'scripts'}:{WORKTREE}",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    import os

    for key in (
        "BRAIN_POSTGREST_URL",
        "BRAIN_POSTGREST_SERVICE_KEY",
        "BANC_BRAIN_SERVICE_KEY",
        "BRAIN_INSTANCES_PATH",
        "BRAIN_SUPABASE_CREDENTIALS_PATH",
        "BRAIN_CLAIMING_TABULAR",
        "BANC_MEASURE_OUTPUT_DIR",
    ):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def plan_etage2_steps(*, etape0: bool = False, purge_markers: bool = False) -> list[str]:
    """Pure schedule of orchestration steps — no subprocess, no I/O."""
    steps: list[str] = ["start_cluster"]
    if etape0:
        steps.append("run_partie_a")
        if purge_markers:
            steps.append("purge_markers")
    else:
        steps.append("verify_parity_readonly")
    steps.extend(
        [
            "prechauffer",
            "measure_standard",
            "measure_quotas_doux",
            "write_outputs",
            "stop_cluster",
        ]
    )
    return steps


def subprocess_run_kwargs(pipe: bool) -> dict[str, Any]:
    """Pure kwargs for subprocess.run — never mix capture_output with stdout/stderr."""
    base: dict[str, Any] = {"check": False, "text": True, "cwd": str(WORKTREE)}
    if pipe:
        return base
    return {**base, "capture_output": True}


def run_capped_subcommand(*, args: list[str], pipe: bool) -> subprocess.CompletedProcess[str]:
    cmd = _systemd_prefix()
    for key, value in _runtime_env().items():
        cmd.extend(["--setenv", f"{key}={value}"])
    cmd.extend([PYTHON, "-m", "scripts.banc_realite.measure_etage2", *args])
    return subprocess.run(cmd, **subprocess_run_kwargs(pipe=pipe))


def run_prechauffer(*, pipe: bool) -> dict[str, Any]:
    proc = run_capped_subcommand(args=["--prechauffer"], pipe=pipe)
    if proc.returncode != 0:
        raise RuntimeError("prechauffer failed: " + ((proc.stderr or proc.stdout or "")[-2000:]))
    if pipe and PRECHAUFFAGE_STATUS_PATH.is_file():
        return json.loads(PRECHAUFFAGE_STATUS_PATH.read_text(encoding="utf-8"))
    if pipe:
        return {"status": "complete"}
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip().startswith("{")]
    return json.loads(lines[-1])


def run_capped_measure(*, variante: str, parity: dict[str, Any], pipe: bool) -> dict[str, Any]:
    parity_path = measure_output_dir() / f"_parity-{variante}.json"
    parity_path.write_text(json.dumps(parity, ensure_ascii=False), encoding="utf-8")
    proc = run_capped_subcommand(args=["--variante", variante], pipe=pipe)
    status_path = measure_status_path(variante)
    if not status_path.is_file():
        raise RuntimeError("measure status file missing after subprocess")
    result = json.loads(status_path.read_text(encoding="utf-8"))
    if proc.returncode != 0 and result.get("status") != "arret":
        raise RuntimeError("measure subprocess failed: " + ((proc.stderr or proc.stdout or "")[-2000:]))
    result["parity"] = parity
    return result


def write_report(
    parity: dict[str, Any],
    standard: dict[str, Any] | None,
    quotas: dict[str, Any] | None,
    paires: list,
    *,
    prechauffage: dict[str, Any],
) -> None:
    agg8 = (standard or {}).get("aggregates", {})
    agg20 = (standard or {}).get("aggregates", {})
    search_health = (standard or {}).get("search_health") or {}
    body = f"""# Rapport — étage 2 bras C (restitution, reprise 3)

## Parité (cette exécution)

- Comptes brain/public : {parity.get('brain_total')} / {parity.get('public_total')}
- Par skill : brain={parity.get('brain_by_skill')} ; public={parity.get('public_by_skill')}
- Par lifecycle : brain={parity.get('brain_by_lifecycle')}
- Embeddings : {parity.get('brain_embeddings')}
- lex_tsv actifs nuls : {parity.get('lex_tsv_null_active')}
- Fonctions public : {parity.get('public_functions')}
- Fumée {SMOKE_QUESTION_ID} : strategy={json.dumps(parity.get('smoke', {}).get('strategy'), ensure_ascii=False)}

## Préchauffage embeddings questions

- Textes distincts requis : {prechauffage.get('textes_distincts_requis')}
- Textes distincts en cache : {prechauffage.get('textes_distincts_en_cache')}
- Exemple question nue : {json.dumps(prechauffage.get('exemple_question_nue'), ensure_ascii=False)}
- Exemple requête structurée : {json.dumps(prechauffage.get('exemple_requete_structuree'), ensure_ascii=False)}

{json.dumps(prechauffage, ensure_ascii=False, indent=2)}

## Santé des appels run_search (standard)

- Total : {search_health.get('run_search_total')}
- Sains : {search_health.get('run_search_sains')} (attendu 100 %)
- Dégradés : {search_health.get('run_search_degrades')}

## Instrument

- Table des détenteurs : {(standard or {}).get('holder_table_seconds')} s
- Tient comptés sans règle même pièce (F3) : {(standard or {}).get('tient_sans_regle_piece')}
- Paires à lire : {len(paires)} couples, {len({p['assertion']['id'] for p in paires})} assertions distinctes

## Restitution plancher / plafond par forme × difficulté (k=8)

{json.dumps(agg8.get('par_forme_difficulte_k8'), ensure_ascii=False, indent=2)}

## Restitution plancher / plafond par forme × difficulté (k=20)

{json.dumps(agg20.get('par_forme_difficulte_k20'), ensure_ascii=False, indent=2)}

## Disputes « un seul côté » (k=8 / k=20)

- k=8 : {json.dumps(agg8.get('disputes_un_seul_cote_k8'), ensure_ascii=False)}
- k=20 : {json.dumps(agg8.get('disputes_un_seul_cote_k20'), ensure_ascii=False)}

## Attribution B5 (supports manquants à k=20, plancher)

{json.dumps(agg8.get('causes_b5'), ensure_ascii=False, indent=2)}

## Bruit et caractères injectés

Voir `sorties/etage2-C.json` (bruit_plancher / bruit_plafond, caracteres_rendered par question).

## Variante quotas doux (k=8, plancher)

{json.dumps((quotas or {}).get('aggregates', {}).get('par_forme_difficulte_k8', {}), ensure_ascii=False, indent=2)}

## Déterminisme

Toutes les questions : {all(q.get('k8', {}).get('determinisme_ok') for q in (standard or {}).get('questions', []))}

## Point d'attention

L'index plein texte de production est en configuration `french` alors que GreenLoop est en anglais.
"""
    REPORT.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Étage 2 bras C")
    parser.add_argument("--etape0", action="store_true", help="Run partie A step 0 (not default)")
    parser.add_argument("--purge-marqueurs", action="store_true", help="Purge model marker claims in step 0")
    parser.add_argument("--pipe", action="store_true", help="Stream subprocess output instead of capturing")
    args = parser.parse_args()

    SORTIES.mkdir(parents=True, exist_ok=True)
    export_runtime_env()
    write_banc_config_files()

    start_cluster()
    try:
        if args.etape0:
            ok, partie_payload, why = run_partie_a(
                smoke_question_text(),
                purge_markers=args.purge_marqueurs,
            )
            parity = partie_payload.get("parity", {})
        else:
            ok, partie_payload, why = verify_parity_readonly(smoke_question_text())
            parity = partie_payload.get("parity", {})
    except Exception as exc:  # noqa: BLE001
        write_arret("A_exception", str(exc))
        stop_cluster()
        return 1
    if not ok:
        write_arret("A_parite", why or "parité non atteinte")
        stop_cluster()
        return 1

    try:
        prechauffage = run_prechauffer(pipe=args.pipe)
    except Exception as exc:  # noqa: BLE001
        write_arret("prechauffer", str(exc))
        stop_cluster()
        return 1

    try:
        standard = run_capped_measure(variante="standard", parity=parity, pipe=args.pipe)
    except Exception as exc:  # noqa: BLE001
        write_arret("B_mesure_standard", str(exc))
        stop_cluster()
        return 1
    if standard.get("status") == "arret":
        health = standard.get("search_health") or {}
        write_arret(
            standard.get("a_quelle_etape", "B_mesure_standard"),
            (
                f"{standard.get('pourquoi', 'mesure interrompue')}"
                f" — sains {health.get('run_search_sains', '?')}/{health.get('run_search_total', '?')}"
            ),
        )
        write_report(parity, None, None, [], prechauffage=prechauffage)
        stop_cluster()
        return 1

    try:
        quotas = run_capped_measure(variante="quotas-doux", parity=parity, pipe=args.pipe)
    except Exception as exc:  # noqa: BLE001
        write_arret("B_mesure_quotas", str(exc))
        stop_cluster()
        return 1
    if quotas.get("status") == "arret":
        health = quotas.get("search_health") or {}
        write_arret(
            quotas.get("a_quelle_etape", "B_mesure_quotas"),
            (
                f"{quotas.get('pourquoi', 'mesure interrompue')}"
                f" — sains {health.get('run_search_sains', '?')}/{health.get('run_search_total', '?')}"
            ),
        )
        write_report(parity, standard, None, standard.get("paires_a_lire", []), prechauffage=prechauffage)
        stop_cluster()
        return 1

    out_path = SORTIES / "etage2-C.json"
    quotas_path = SORTIES / "etage2-C-quotas-doux.json"
    paires_path = SORTIES / "etage2-paires-a-lire.json"
    paires = standard.get("paires_a_lire", [])
    out_path.write_text(json.dumps(standard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    quotas_path.write_text(json.dumps(quotas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paires_path.write_text(json.dumps(paires, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(parity, standard, quotas, paires, prechauffage=prechauffage)

    agg = standard.get("aggregates", {})
    search_health = standard.get("search_health") or {}
    RESULT.write_text(
        json.dumps(
            {
                "status": "complete",
                "parite_ok": True,
                "questions": len(standard.get("questions", [])),
                "par_forme_k8": agg.get("par_forme_difficulte_k8", {}),
                "par_forme_k20": agg.get("par_forme_difficulte_k20", {}),
                "disputes_un_seul_cote_k8": len(agg.get("disputes_un_seul_cote_k8", [])),
                "disputes_un_seul_cote_k20": len(agg.get("disputes_un_seul_cote_k20", [])),
                "paires_a_lire": len(paires),
                "assertions_paires": len({p["assertion"]["id"] for p in paires}),
                "holder_table_seconds": standard.get("holder_table_seconds"),
                "tient_sans_regle_piece": standard.get("tient_sans_regle_piece"),
                "run_search_total": search_health.get("run_search_total"),
                "run_search_sains": search_health.get("run_search_sains"),
                "determinisme_ok": all(
                    q.get("k8", {}).get("determinisme_ok") for q in standard.get("questions", [])
                ),
                "appels_modele": 0,
                "files_written": [
                    str(p.relative_to(BANC_ROOT))
                    for p in [
                        out_path,
                        quotas_path,
                        paires_path,
                        questions_journal_path("standard"),
                        questions_journal_path("quotas-doux"),
                        REPORT,
                        RESULT,
                    ]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    stop_cluster()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
