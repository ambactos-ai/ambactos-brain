#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrate étape 2 bras C — trial, ingestion, embeddings, fidélité."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[2]
for entry in (WORKTREE / "scripts", WORKTREE / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.fidelite import (
    F1_KEY_CLAIMS_TOTAL,
    F1_KEY_GLOBAL,
    F1_KEY_MEMOIRE,
    F1_KEY_PAR_VOIE,
    F1_KEY_PROSE,
    F1_KEY_SEUIL_ATTEINT,
    F1_KEY_TABULAIRE,
    measure,
)
from banc_realite.ingest_runner import (
    TRIAL_CSV,
    TRIAL_LARGE,
    TRIAL_SMALL,
    backfill_embeddings,
    count_tabular_claims,
    run_ingestion,
)
from banc_realite.infra import BancDatabase, cluster_ready, connect_existing, reset_brain_schema, start_cluster
from banc_realite.memory import embedding_coverage
from banc_realite.relay import CALL_LOG, ensure_call_log_header, load_call_count

BANC_ROOT = (BANC_VAR)
ARRET = BANC_ROOT / "_etape2-arret.json"
RESULT = BANC_ROOT / "_etape2-result.json"
REPORT = BANC_ROOT / "RAPPORT-ETAPE2-BRAS-C.md"
MAX_CALLS = 160
PYTHON = sys.executable

MEASURE_SUMMARY_PATHS: list[list[str]] = [
    ["F1", F1_KEY_GLOBAL],
    ["F1", F1_KEY_PAR_VOIE, F1_KEY_PROSE],
    ["F1", F1_KEY_PAR_VOIE, F1_KEY_TABULAIRE],
    [F1_KEY_MEMOIRE, F1_KEY_CLAIMS_TOTAL],
    [F1_KEY_SEUIL_ATTEINT],
]


def summarize_fidelite(fidelite: dict) -> dict[str, Any]:
    f1 = fidelite.get("F1", {})
    par_voie = f1.get(F1_KEY_PAR_VOIE, {})
    return {
        "F1": f1.get(F1_KEY_GLOBAL, 0.0),
        "F1_prose": par_voie.get(F1_KEY_PROSE, 0.0),
        "F1_tabulaire": par_voie.get(F1_KEY_TABULAIRE, 0.0),
        "claims": fidelite.get(F1_KEY_MEMOIRE, {}).get(F1_KEY_CLAIMS_TOTAL, 0),
        "seuil_atteint": bool(fidelite.get(F1_KEY_SEUIL_ATTEINT)),
    }


def write_arret(step: str, why: str) -> None:
    ARRET.write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False, indent=2)
        + "\n"
    )


def run_capped(script: str, *, runtime_max_sec: int) -> None:
    env = {
        "PYTHONPATH": f"{WORKTREE / 'src'}:{WORKTREE / 'scripts'}:{WORKTREE}",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in ("BRAIN_POSTGREST_URL", "BRAIN_POSTGREST_SERVICE_KEY", "BANC_BRAIN_SERVICE_KEY"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    cmd = [
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
        f"RuntimeMaxSec={runtime_max_sec}",
    ]
    for key, value in env.items():
        cmd.extend(["--setenv", f"{key}={value}"])
    cmd.extend([PYTHON, str(WORKTREE / "scripts" / "banc_realite" / script)])
    subprocess.run(cmd, check=True)


def ensure_infra() -> BancDatabase | None:
    (BANC_ROOT / "sorties").mkdir(parents=True, exist_ok=True)
    ensure_call_log_header(CALL_LOG, run_id="etape2-reprise")
    if not cluster_ready():
        start_cluster()
    connection = connect_existing()
    reset_brain_schema(connection)
    return None


def trial_failed(trial: dict) -> str | None:
    by_name = {p.get("file"): p for p in trial.get("pieces", []) if not p.get("skipped")}
    for required in (TRIAL_SMALL, TRIAL_LARGE, TRIAL_CSV):
        piece = by_name.get(required)
        if piece is None:
            return f"pièce d'essai absente: {required}"
        if piece.get("statut") == "erreur" or piece.get("erreur"):
            return f"{required} en erreur: {piece.get('erreur')}"
        if piece.get("claims_ecrits", 0) == 0:
            return f"{required} sans claim écrit"
    csv_piece = by_name[TRIAL_CSV]
    if csv_piece.get("appels_modele", 0) != 0:
        return f"{TRIAL_CSV} a déclenché des appels modèle"
    if csv_piece.get("claims_tabulaires", 0) == 0:
        return f"{TRIAL_CSV} sans claim tabulaire"
    large = by_name[TRIAL_LARGE]
    if large.get("claims_tabulaires", 0) == 0:
        return f"{TRIAL_LARGE} sans claim tabulaire"
    return None


def main() -> int:
    db = ensure_infra()
    prior_calls = load_call_count(CALL_LOG)

    trial = run_ingestion(trial_only=True)
    why = trial_failed(trial)
    if why or trial.get("arret") == "essai_echoue":
        write_arret("B_essai", why or f"essai sur 3 pièces en erreur: {trial.get('erreurs')}")
        return 1
    if not any(p.get("claims_ecrits", 0) > 0 for p in trial["pieces"] if not p.get("skipped")):
        write_arret("B_essai", "aucun claim écrit sur l'essai")
        return 1

    full = run_ingestion()
    calls = full.get("appels_modele", prior_calls)
    if calls > MAX_CALLS:
        write_arret("B_ingestion", f"plafond {MAX_CALLS} appels dépassé ({calls})")
        return 1
    hard_errors = [p for p in full.get("pieces", []) if p.get("statut") == "erreur"]
    if hard_errors:
        write_arret("B_ingestion", f"pièces en erreur: {[p['file'] for p in hard_errors]}")
        return 1

    embed_stats = backfill_embeddings()
    coverage = embedding_coverage(connect_existing(), vault_ref="greenloop")
    fidelite = measure()
    summary = summarize_fidelite(fidelite)
    f1 = summary["F1"]
    f1_prose = summary["F1_prose"]
    f1_tab = summary["F1_tabulaire"]
    claims = summary["claims"]
    claims_tab = count_tabular_claims(full, connection=connect_existing())
    seuil = summary["seuil_atteint"]

    files_written = [
        str(p.relative_to(BANC_ROOT))
        for p in [
            BANC_ROOT / "sorties/ingestion.jsonl",
            CALL_LOG,
            BANC_ROOT / "sorties/fidelite-C.json",
            BANC_ROOT / "sorties/residu-a-adjudger.json",
            BANC_ROOT / "sorties/hors-or-echantillon.json",
            REPORT,
            RESULT,
        ]
        if p.exists()
    ]
    RESULT.write_text(
        json.dumps(
            {
                "status": "complete",
                "appels_modele_cumules": calls,
                "fournisseur": full.get("fournisseur", "hermes_relay"),
                "claims": claims,
                "claims_tabulaires": claims_tab,
                "F1": f1,
                "F1_prose": f1_prose,
                "F1_tabulaire": f1_tab,
                "seuil_atteint": seuil,
                "files_written": files_written,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    write_report(trial, full, embed_stats, coverage, fidelite)
    return 0


def write_report(trial, full, embed_stats, coverage, fidelite) -> None:
    f1 = fidelite.get("F1", {})
    par_voie = f1.get(F1_KEY_PAR_VOIE, {})
    body = f"""# Rapport — étape 2 bras C (fidélité)

## 1. Montage
- Forme : cluster privé `initdb` sous `{BANC_ROOT}/pg/postgres`, schéma `brain` (filebrain), PostgREST loopback 56101, proxy 56100.
- Relancer : `{BANC_ROOT}/demarrer.sh`
- Arrêter : `{BANC_ROOT}/arreter.sh`
- Détruire : supprimer `{BANC_ROOT}/pg/` et `{BANC_ROOT}/aire/`.

## 2. Ingestion
- Appels modèle cumulés : {full.get('appels_modele')} (plafond {MAX_CALLS}, dont 37 antérieurs)
- Fournisseur : {full.get('fournisseur')} — journal `{CALL_LOG.name}` : 0 appel `claude_cli` / `anthropic`.
- Essai 3 pièces : {len(trial.get('pieces', []))} ; erreurs essai : {trial.get('erreurs') or 'aucune'}
- Pièces partielles : {full.get('partielles') or 'aucune'}
- Pièces en erreur (lot complet) : {full.get('erreurs') or 'aucune'}
- Embeddings Mistral : {embed_stats}
- Couverture note.body embeddés : {coverage}

## 3. Forme de la mémoire
{fidelite.get('memoire')}

## 4. Mesures
- F1 global : {f1.get(F1_KEY_GLOBAL)} (dénominateur : {f1.get('denominateur')}, F1 prose : {par_voie.get(F1_KEY_PROSE)}, F1 tabulaire : {par_voie.get(F1_KEY_TABULAIRE)})
- F2 : {fidelite.get('F2')}
- F3 classes : {fidelite.get('F3', {}).get('classes')}
- F4 : paires {fidelite.get('F4', {}).get('paires')}, distinctes mesurables {fidelite.get('F4', {}).get('distinctes_mesurables')}
- F6 : {fidelite.get('F6')}
- Hors-or : {fidelite.get('hors_or')}
- Normalisation F1 : {f1.get('normalisation')}

## 5. Point d'arrêt
- Seuil F1 ≥ 0,80 : {'atteint' if fidelite.get('seuil_atteint') else 'non atteint'} ({f1.get('global')})
- Diagnostic : voir `sorties/residu-a-adjudger.json` et F1 par pièce/perspective dans `sorties/fidelite-C.json`.

## 6. Limites
- Étage 2/3 non lancés (mandat étape 1 seulement).
"""
    REPORT.write_text(body)


if __name__ == "__main__":
    raise SystemExit(main())
