#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrate ingestion-2 — second GreenLoop corpus ingestion on corrected main."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[2]
for entry in (WORKTREE / "scripts", WORKTREE / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.config import (
    BANC_ROOT,
    GOLD,
    INGESTION2_SORTIES,
    SMOKE_QUESTION_ID,
    SORTIES,
    export_runtime_env,
    write_banc_config_files,
)
from banc_realite.fidelite import (
    F1_KEY_GLOBAL,
    F1_KEY_MEMOIRE,
    F1_KEY_PAR_VOIE,
    F1_KEY_PROSE,
    F1_KEY_TABULAIRE,
    measure,
)
from banc_realite.ingest_runner import (
    TRIAL_CSV,
    TRIAL_LARGE,
    TRIAL_SMALL,
    IngestionConfig,
    backfill_embeddings,
    clear_public_claims,
    count_tabular_claims,
    ingestion_status_path,
    list_corpus,
    run_ingestion,
)
from banc_realite.infra import PG_BIN, connect_existing, reset_brain_schema, start_cluster
from banc_realite.measure_etage2 import prechauffer_questions, questions_journal_path
from banc_realite.memory import embedding_coverage
from banc_realite.partie_a import _public_schema_ready, _reset_broken_public_schema
from banc_realite.production import (
    apply_production_schema,
    copy_claims_to_public,
    ensure_extensions,
    plan_brain_to_public_pipeline,
    restart_postgrest_brain,
    restart_postgrest_public,
    verify_parity,
)
from banc_realite.relay import load_call_count
from banc_realite.run_etage2 import run_capped_measure, subprocess_run_kwargs

ARRET = BANC_ROOT / "_ingestion2-arret.json"
RESULT = BANC_ROOT / "_ingestion2-result.json"
REPORT = BANC_ROOT / "RAPPORT-INGESTION-2.md"
DUMP_PATH = BANC_ROOT / "entrees" / "memoire-etage1-avant-ingestion-2.dump"
PYTHON = sys.executable
CFG = IngestionConfig.for_ingestion2()

PROTECTED_OUTPUTS = (
    CFG.ingest_log,
    CFG.call_log,
    INGESTION2_SORTIES / "fidelite-C.json",
    INGESTION2_SORTIES / "residu-a-adjudger.json",
    INGESTION2_SORTIES / "hors-or-echantillon.json",
    INGESTION2_SORTIES / "etage2-C.json",
    INGESTION2_SORTIES / "etage2-C-quotas-doux.json",
    INGESTION2_SORTIES / "etage2-paires-a-lire.json",
    REPORT,
    RESULT,
)


def write_arret(step: str, why: str) -> None:
    ARRET.write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False)
        + "\n"
    )


def _runtime_env() -> dict[str, str]:
    export_runtime_env()
    os.environ["BRAIN_CLAIMING_TABULAR"] = "1"
    write_banc_config_files()
    env = {
        "PYTHONPATH": f"{WORKTREE / 'src'}:{WORKTREE / 'scripts'}:{WORKTREE}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "BRAIN_CLAIMING_TABULAR": "1",
        "BANC_MEASURE_OUTPUT_DIR": str(INGESTION2_SORTIES),
    }
    for key in (
        "BRAIN_POSTGREST_URL",
        "BRAIN_POSTGREST_SERVICE_KEY",
        "BANC_BRAIN_SERVICE_KEY",
        "BRAIN_INSTANCES_PATH",
        "BRAIN_SUPABASE_CREDENTIALS_PATH",
    ):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def _service_key() -> str:
    secrets = BANC_ROOT / "secrets.env"
    if secrets.is_file():
        for line in secrets.read_text(encoding="utf-8").splitlines():
            if line.startswith("BANC_BRAIN_SERVICE_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def smoke_question_text() -> str:
    questions = json.loads((GOLD / "questions.json").read_text(encoding="utf-8"))["questions"]
    for question in questions:
        if question["id"] == SMOKE_QUESTION_ID:
            return question["question"]
    raise RuntimeError(f"smoke question {SMOKE_QUESTION_ID} not found")


def outputs_already_present() -> list[str]:
    return [str(path.relative_to(BANC_ROOT)) for path in PROTECTED_OUTPUTS if path.exists()]


def _verify_dump(path: Path) -> None:
    listed = subprocess.run(
        [str(PG_BIN / "pg_restore"), "--list", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    if "TABLE brain claims" not in listed.stdout and "TABLE DATA brain claims" not in listed.stdout:
        raise RuntimeError("pg_restore --list ne contient pas la table brain.claims")


def export_memory_dump() -> None:
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DUMP_PATH.exists():
        _verify_dump(DUMP_PATH)
        return
    subprocess.run(
        [
            str(PG_BIN / "pg_dump"),
            "-h",
            "127.0.0.1",
            "-p",
            "56102",
            "-U",
            "postgres",
            "-d",
            "postgres",
            "-n",
            "brain",
            "-n",
            "public",
            "-Fc",
            "-f",
            str(DUMP_PATH),
        ],
        check=True,
        env={**os.environ, "PGPASSWORD": ""},
    )
    _verify_dump(DUMP_PATH)


def empty_memory(connection) -> None:
    reset_brain_schema(connection)
    row = connection.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = 'claims'"
    ).fetchone()
    if row:
        clear_public_claims(connection)


def _systemd_ingestion_prefix() -> list[str]:
    return [
        "systemd-run",
        "--user",
        "--wait",
        "--collect",
        "--pipe",
        "-p",
        "MemoryMax=3G",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "KillMode=control-group",
        "-p",
        "RuntimeMaxSec=7200",
    ]


def build_ingestion_systemd_cmd(
    *,
    env: dict[str, str],
    python: str = PYTHON,
) -> list[str]:
    cmd = _systemd_ingestion_prefix()
    for key, value in env.items():
        cmd.extend(["--setenv", f"{key}={value}"])
    cmd.extend(
        [
            python,
            "-m",
            "scripts.banc_realite.ingest_runner",
            "all",
            "--ingestion2",
        ]
    )
    return cmd


def read_ingestion_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"fichier d'état ingestion absent: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"fichier d'état ingestion illisible: {path}: {exc}") from exc


def evaluate_ingestion_outcome(
    full: dict[str, Any],
    *,
    corpus_piece_count: int,
    calls: int,
    max_calls: int,
) -> tuple[str | None, list[str]]:
    pieces_partielles = list(full.get("partielles") or [])
    for piece in full.get("pieces", []):
        if piece.get("statut") == "partielle":
            name = piece.get("file")
            if name and name not in pieces_partielles:
                pieces_partielles.append(name)
    if full.get("arret"):
        return f"arrêt ingestion: {full['arret']}", pieces_partielles
    if calls >= max_calls:
        return f"plafond {max_calls} atteint ({calls})", pieces_partielles
    hard_errors = [p for p in full.get("pieces", []) if p.get("statut") == "erreur"]
    if hard_errors:
        return f"pièces en erreur: {[p['file'] for p in hard_errors]}", pieces_partielles
    processed = [p for p in full.get("pieces", []) if not p.get("skipped")]
    if len(processed) < corpus_piece_count:
        return (
            f"corpus incomplet: {len(processed)}/{corpus_piece_count} pièces traitées",
            pieces_partielles,
        )
    return None, pieces_partielles


def ingestion_duration_s(full: dict[str, Any], ingest_log: Path | None = None) -> float:
    if full.get("duree_s") is not None:
        return round(float(full["duree_s"]), 1)
    total = sum(
        float(piece.get("duree_s", 0))
        for piece in full.get("pieces", [])
        if not piece.get("skipped")
    )
    if total > 0:
        return round(total, 1)
    if ingest_log and ingest_log.is_file():
        journal_total = 0.0
        for line in ingest_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("skipped"):
                journal_total += float(record.get("duree_s", 0))
        return round(journal_total, 1)
    return 0.0


def format_piece_names(names: list[str] | None) -> str:
    if not names:
        return "aucune"
    return ", ".join(names)


def sum_tabular_rejections_from_journal(
    ingest_log: Path,
    *,
    run_id: str | None = None,
) -> tuple[int, int]:
    dropped = 0
    unanchored = 0
    if not ingest_log.is_file():
        return 0, 0
    for line in ingest_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if run_id is not None and record.get("run_id") != run_id:
            continue
        dropped += int(
            record.get("tabular_marker_claims_dropped")
            or record.get("marker_claims_dropped", 0)
        )
        unanchored += int(
            record.get("tabular_marker_claims_dropped_unanchored")
            or record.get("marker_claims_dropped_unanchored", 0)
        )
    return dropped, unanchored


def trial_failed(trial: dict[str, Any]) -> str | None:
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


def run_capped_full_ingestion() -> dict[str, Any]:
    env = _runtime_env()
    cmd = build_ingestion_systemd_cmd(env=env)
    proc = subprocess.run(cmd, **subprocess_run_kwargs(pipe=True))
    status_path = ingestion_status_path(CFG)
    if proc.returncode != 0 and not status_path.is_file():
        raise RuntimeError((proc.stderr or proc.stdout or "ingestion échouée")[-2000:])
    return read_ingestion_status(status_path)


def run_brain_to_public_pipeline(connection, smoke_question: str) -> tuple[bool, dict[str, Any], str | None]:
    steps = plan_brain_to_public_pipeline()
    payload: dict[str, Any] = {"steps": steps}
    restart_postgrest_brain()
    payload["postgrest_brain"] = "ok"
    missing_ext = ensure_extensions(connection)
    if missing_ext:
        return False, payload, f"extensions manquantes: {missing_ext}"
    if not _public_schema_ready(connection):
        _reset_broken_public_schema(connection)
        apply_production_schema(connection)
    copy_stats = copy_claims_to_public(connection)
    payload["copy"] = copy_stats
    service_key = _service_key()
    restart_postgrest_public(service_key=service_key)
    payload["postgrest_public"] = "ok"
    ok, parity, why = verify_parity(connection, smoke_question)
    payload["parity"] = parity
    return ok, payload, why


def memory_stats(connection) -> dict[str, Any]:
    total = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE lifecycle_status = 'active'"
    ).fetchone()["n"]
    tab = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE lifecycle_status = 'active' "
        "AND extracted_by_skill = 'tabulaire-deterministe'"
    ).fetchone()["n"]
    note_body = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE lifecycle_status = 'active' "
        "AND predicate = 'note.body'"
    ).fetchone()["n"]
    relation = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE lifecycle_status = 'active' "
        "AND predicate = 'relation.about'"
    ).fetchone()["n"]
    anchored = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE lifecycle_status = 'active' "
        "AND COALESCE(extracted_by_skill, '') != 'tabulaire-deterministe' "
        "AND source_span->>'line_origin' = 'anchored'"
    ).fetchone()["n"]
    model_total = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE lifecycle_status = 'active' "
        "AND COALESCE(extracted_by_skill, '') != 'tabulaire-deterministe'"
    ).fetchone()["n"]
    return {
        "total": int(total),
        "tabulaire": int(tab),
        "note_body": int(note_body),
        "relation_about": int(relation),
        "modele": int(model_total),
        "modele_anchored": int(anchored),
    }


def find_dispute_claims(connection) -> dict[str, Any]:
    needles = ("657", "11 collected sites", "11 sites")
    rows = connection.execute(
        "SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path, source_span "
        "FROM brain.claims WHERE lifecycle_status = 'active' "
        "AND source_file_path LIKE %s",
        ("%20-internal-service-review.md",),
    ).fetchall()
    hits = []
    for row in rows:
        text = json.dumps(row["value"], ensure_ascii=False) if not isinstance(row["value"], str) else row["value"]
        if any(needle.lower() in text.lower() for needle in needles):
            hits.append(
                {
                    "claim_id": str(row["claim_id"]),
                    "triplet": (row["entity_id"], row["predicate"], row["value"]),
                    "source_span": row.get("source_span"),
                }
            )
    return {"piece": "20-internal-service-review.md", "hits": hits}


def load_stage1_fidelite() -> dict[str, Any] | None:
    path = SORTIES / "fidelite-C.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage1_etage2() -> dict[str, Any] | None:
    path = SORTIES / "etage2-C.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(
    *,
    started: float,
    trial: dict[str, Any],
    full: dict[str, Any],
    embed_stats: dict[str, Any],
    coverage: dict[str, int],
    fidelite: dict[str, Any],
    parity_payload: dict[str, Any],
    standard: dict[str, Any],
    quotas: dict[str, Any],
    paires: list[dict[str, Any]],
    prechauffage: dict[str, Any],
    before_stats: dict[str, Any],
    after_stats: dict[str, Any],
    dispute_hits: dict[str, Any],
    ingest_log: Path | None = None,
) -> None:
    stage1_f1 = load_stage1_fidelite()
    stage1_e2 = load_stage1_etage2()
    calls = load_call_count(CFG.call_log)
    f1 = fidelite.get("F1", {})
    par_voie = f1.get(F1_KEY_PAR_VOIE, {})
    log_path = ingest_log or CFG.ingest_log
    duration_s = ingestion_duration_s(full, log_path)
    erreurs = full.get("erreurs") or []
    partielles = list(full.get("partielles") or [])
    for piece in full.get("pieces", []):
        if piece.get("statut") == "partielle":
            name = piece.get("file")
            if name and name not in partielles:
                partielles.append(name)
    dropped, unanchored = sum_tabular_rejections_from_journal(log_path)
    body = f"""# Rapport — ingestion 2 bras C

## Appels modèle et durée
- Total : {calls} (plafond {CFG.max_calls})
- Fournisseur : {full.get('fournisseur')}
- Durée : {duration_s} s
- Pièces en erreur : {format_piece_names(erreurs)}
- Pièces partielles : {format_piece_names(partielles)}

## Faits en mémoire
- Avant vidage (dump) : voir export `{DUMP_PATH.name}`
- Après ingestion : total {after_stats['total']}, tabulaires {after_stats['tabulaire']}, modèle {after_stats['modele']}
- note.body : {after_stats['note_body']} ; relation.about : {after_stats['relation_about']}
- Modèle line_origin=anchored : {after_stats['modele_anchored']} / {after_stats['modele']}
- Rejets tabulaires (somme journal) : dropped={dropped}, unanchored={unanchored}

## Dispute « 9 ou 11 sites »
{json.dumps(dispute_hits, ensure_ascii=False, indent=2)}

## Fidélité (graine 20260920)
- F1 global : {f1.get(F1_KEY_GLOBAL)} (étage 1 : {(stage1_f1 or {}).get('F1', {}).get('global')})
- F1 prose : {par_voie.get(F1_KEY_PROSE)} (étage 1 : {(stage1_f1 or {}).get('F1', {}).get('par_voie', {}).get('prose')})
- F1 tabulaire : {par_voie.get(F1_KEY_TABULAIRE)} (étage 1 : {(stage1_f1 or {}).get('F1', {}).get('par_voie', {}).get('tabulaire')})
- Claims : {fidelite.get(F1_KEY_MEMOIRE, {}).get('claims_total')}

## Embeddings
- Backfill : {embed_stats}
- Couverture note.body : {coverage}

## Parité
{json.dumps(parity_payload.get('parity', {}), ensure_ascii=False, indent=2)}

## Étage 2 bis (k=8 / k=20)
### Préchauffage
{json.dumps(prechauffage, ensure_ascii=False, indent=2)}

### Standard vs étage 2 précédent (`sorties/etage2-C.json`)
- k=8 : {json.dumps((standard or {}).get('aggregates', {}).get('par_forme_difficulte_k8'), ensure_ascii=False)}
- k=20 : {json.dumps((standard or {}).get('aggregates', {}).get('par_forme_difficulte_k20'), ensure_ascii=False)}
- Référence k=8 : {json.dumps((stage1_e2 or {}).get('aggregates', {}).get('par_forme_difficulte_k8'), ensure_ascii=False)}
- Référence k=20 : {json.dumps((stage1_e2 or {}).get('aggregates', {}).get('par_forme_difficulte_k20'), ensure_ascii=False)}
- Disputes un seul côté k=8 : {(standard or {}).get('aggregates', {}).get('disputes_un_seul_cote_k8')}
- Disputes un seul côté k=20 : {(standard or {}).get('aggregates', {}).get('disputes_un_seul_cote_k20')}
- Causes B5 k=20 : {json.dumps((standard or {}).get('aggregates', {}).get('causes_b5'), ensure_ascii=False)}
- Paires à lire : {len(paires)}

### Quotas doux k=8
{json.dumps((quotas or {}).get('aggregates', {}).get('par_forme_difficulte_k8'), ensure_ascii=False)}
"""
    REPORT.write_text(body, encoding="utf-8")


def main() -> int:
    started = time.monotonic()
    existing = outputs_already_present()
    if existing:
        write_arret("preflight", f"sorties déjà présentes: {existing}")
        return 1

    INGESTION2_SORTIES.mkdir(parents=True, exist_ok=True)
    _runtime_env()
    start_cluster()
    connection = connect_existing()

    try:
        export_memory_dump()
    except Exception as exc:  # noqa: BLE001
        write_arret("0_export", str(exc))
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    before_stats = memory_stats(connection)
    empty_memory(connection)
    restart_postgrest_brain()

    trial = run_ingestion(trial_only=True, config=CFG)
    why = trial_failed(trial)
    if why or trial.get("arret") == "essai_echoue":
        write_arret("3_essai", why or str(trial.get("erreurs")))
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    try:
        full = run_capped_full_ingestion()
    except Exception as exc:  # noqa: BLE001
        write_arret("4_ingestion", str(exc))
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    calls = load_call_count(CFG.call_log)
    corpus_piece_count = len(list_corpus())
    stop_reason, pieces_partielles = evaluate_ingestion_outcome(
        full,
        corpus_piece_count=corpus_piece_count,
        calls=calls,
        max_calls=CFG.max_calls,
    )
    if stop_reason:
        write_arret("4_ingestion", stop_reason)
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    embed_stats = backfill_embeddings(retry_attempts=10, retry_delay_s=30.0)
    coverage = embedding_coverage(connection, vault_ref="greenloop")

    ok, parity_payload, parity_why = run_brain_to_public_pipeline(connection, smoke_question_text())
    if not ok:
        write_arret("6_parite", parity_why or "parité non atteinte")
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    fidelite = measure(output_dir=INGESTION2_SORTIES)
    if fidelite.get("status") == "error":
        write_arret("7_fidelite", fidelite.get("message", "mesure en erreur"))
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    os.environ["BANC_MEASURE_OUTPUT_DIR"] = str(INGESTION2_SORTIES)
    prechauffage = prechauffer_questions()
    if prechauffage.get("warmed", 0) != 0:
        write_arret("8_prechauffage", f"cache froid: {prechauffage.get('warmed')} embeddings chauffés")
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    parity = parity_payload.get("parity", {})
    try:
        standard = run_capped_measure(variante="standard", parity=parity, pipe=False)
    except Exception as exc:  # noqa: BLE001
        write_arret("8_etage2_standard", str(exc))
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1
    if standard.get("status") == "arret":
        write_arret("8_etage2_standard", standard.get("pourquoi", "mesure interrompue"))
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    try:
        quotas = run_capped_measure(variante="quotas-doux", parity=parity, pipe=False)
    except Exception as exc:  # noqa: BLE001
        write_arret("8_etage2_quotas", str(exc))
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1
    if quotas.get("status") == "arret":
        write_arret("8_etage2_quotas", quotas.get("pourquoi", "mesure interrompue"))
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    out_path = INGESTION2_SORTIES / "etage2-C.json"
    quotas_path = INGESTION2_SORTIES / "etage2-C-quotas-doux.json"
    paires_path = INGESTION2_SORTIES / "etage2-paires-a-lire.json"
    paires = standard.get("paires_a_lire", [])
    out_path.write_text(json.dumps(standard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    quotas_path.write_text(json.dumps(quotas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paires_path.write_text(json.dumps(paires, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    after_stats = memory_stats(connection)
    dispute_hits = find_dispute_claims(connection)
    write_report(
        started=started,
        trial=trial,
        full=full,
        embed_stats=embed_stats,
        coverage=coverage,
        fidelite=fidelite,
        parity_payload=parity_payload,
        standard=standard,
        quotas=quotas,
        paires=paires,
        prechauffage=prechauffage,
        before_stats=before_stats,
        after_stats=after_stats,
        dispute_hits=dispute_hits,
        ingest_log=CFG.ingest_log,
    )

    f1_global = fidelite.get("F1", {}).get(F1_KEY_GLOBAL)
    claims_total = fidelite.get(F1_KEY_MEMOIRE, {}).get("claims_total", after_stats["total"])
    RESULT.write_text(
        json.dumps(
            {
                "status": "complete",
                "appels_modele": calls,
                "pieces_partielles": pieces_partielles,
                "claims": claims_total,
                "F1": f1_global,
                "files_written": [
                    str(p.relative_to(BANC_ROOT))
                    for p in [
                        DUMP_PATH,
                        CFG.ingest_log,
                        CFG.call_log,
                        INGESTION2_SORTIES / "fidelite-C.json",
                        INGESTION2_SORTIES / "residu-a-adjudger.json",
                        INGESTION2_SORTIES / "hors-or-echantillon.json",
                        out_path,
                        quotas_path,
                        paires_path,
                        questions_journal_path("standard"),
                        questions_journal_path("quotas-doux"),
                        REPORT,
                        RESULT,
                    ]
                    if p.exists()
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
