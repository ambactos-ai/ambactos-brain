#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrate ingestion-3 — ordinary path, full corpus, four measures."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date
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
    INGESTION3_MAX_EMBEDDINGS,
    INGESTION3_REPRISE_LOG,
    INGESTION3_SORTIES,
    SMOKE_QUESTION_ID,
    export_runtime_env,
    write_banc_config_files,
)
from banc_realite.fidelite import (
    F1_KEY_GLOBAL,
    F1_KEY_PAR_VOIE,
    F1_KEY_PROSE,
    F1_KEY_TABULAIRE,
    measure as measure_fidelite,
)
from banc_realite.ingest_runner import (
    TRIAL_CSV,
    TRIAL_LARGE,
    TRIAL_SMALL,
    IngestionConfig,
    VAULT_REF,
    backfill_embeddings,
    clear_public_claims,
    ingestion_status_path,
    list_corpus,
    run_ingestion,
    run_tabular_ordinary_reprise,
)
from banc_realite.infra import PG_BIN, connect_existing, reset_brain_schema, start_cluster
from banc_realite.measure_disputes import (
    COUCHES,
    KS,
    measure_etage2_disputes,
    _claims_by_piece,
    _read_public_claims,
)
from banc_realite.measure_etage2 import EMBED_CACHE_PATH
from banc_realite.measure_resolution import (
    PALIERS,
    build_confusion_for_palier,
    choose_best_palier,
    summarize_paliers,
)
from banc_realite.measure_tabulaire_v2 import count_by_scope, measure_completion
from banc_realite.memory import embedding_coverage
from banc_realite.partie_a import _public_schema_ready, _reset_broken_public_schema
from banc_realite.production import (
    apply_production_schema,
    copy_claims_to_public,
    ensure_extensions,
    restart_postgrest_brain,
    restart_postgrest_public,
    verify_parity,
)
from banc_realite.relay import load_call_count
from banc_realite.resolution import build_resolution_context, read_claims_from_db, write_inventaire
from banc_realite.run_ingestion2 import (
    build_ingestion_systemd_cmd,
    evaluate_ingestion_outcome,
    ingestion_duration_s,
    read_ingestion_status,
    run_brain_to_public_pipeline,
    sum_tabular_rejections_from_journal,
    trial_failed,
)
from banc_realite.run_etage2 import subprocess_run_kwargs
from banc_realite.holders import build_holder_table

ARRET = BANC_ROOT / "_ingestion3-arret.json"
RESULT = BANC_ROOT / "_ingestion3-result.json"
REPORT = INGESTION3_SORTIES / "RAPPORT-INGESTION-3.md"
DUMP_PATH = BANC_ROOT / "entrees" / "memoire-avant-ingestion-3.dump"
PYTHON = sys.executable
CFG = IngestionConfig.for_ingestion3()
EMBED_LOG = INGESTION3_SORTIES / "embedding-appels.jsonl"

BEFORE_REWRITE = {
    "F1": 0.9388,
    "F1_prose": None,
    "F1_tabulaire": None,
    "tabular_scope": {"row": 1499, "cell": 9122, "sans_scope": 443},
    "disputes_reconnues": 31,
    "pieges_en_dispute": 0,
    "trajectoires": 0,
    "indecidable": 12,
    "deux_cotes_28": 22,
    "faits_ramenes": 7007,
    "depuis_autre_piece": 4069,
}


def write_arret(step: str, why: str) -> None:
    ARRET.write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def completion_markers_present() -> list[str]:
    return [str(p.relative_to(BANC_ROOT)) for p in (RESULT, REPORT) if p.exists()]


def model_claim_count(connection) -> int:
    row = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims "
        "WHERE lifecycle_status = 'active' AND vault_ref = %s "
        "AND COALESCE(extracted_by_skill, '') != 'tabulaire-deterministe'",
        (VAULT_REF,),
    ).fetchone()
    return int(row["n"])


def _runtime_env() -> dict[str, str]:
    export_runtime_env()
    os.environ["BRAIN_CLAIMING_TABULAR"] = "1"
    write_banc_config_files()
    env = {
        "PYTHONPATH": f"{WORKTREE / 'src'}:{WORKTREE / 'scripts'}:{WORKTREE}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "BRAIN_CLAIMING_TABULAR": "1",
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


def smoke_question_text() -> str:
    questions = json.loads((GOLD / "questions.json").read_text(encoding="utf-8"))["questions"]
    for question in questions:
        if question["id"] == SMOKE_QUESTION_ID:
            return question["question"]
    raise RuntimeError(f"smoke question {SMOKE_QUESTION_ID} not found")


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


def build_ingestion3_systemd_cmd(env: dict[str, str]) -> list[str]:
    cmd = build_ingestion_systemd_cmd(env=env, python=PYTHON)
    if "--ingestion2" in cmd:
        idx = cmd.index("--ingestion2")
        cmd[idx] = "--ingestion3"
    else:
        cmd.append("--ingestion3")
    return cmd


def run_capped_full_ingestion() -> dict[str, Any]:
    env = _runtime_env()
    cmd = build_ingestion3_systemd_cmd(env=env)
    proc = subprocess.run(cmd, **subprocess_run_kwargs(pipe=True))
    status_path = ingestion_status_path(CFG)
    if proc.returncode != 0 and not status_path.is_file():
        raise RuntimeError((proc.stderr or proc.stdout or "ingestion échouée")[-2000:])
    return read_ingestion_status(status_path)


def _load_embed_cache() -> dict[str, list[float]]:
    if not EMBED_CACHE_PATH.is_file():
        raise RuntimeError("cache froid : etage2-question-embeddings.json absent")
    return json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))


def measure_resolution_paliers(connection) -> dict[str, Any]:
    inventaire = write_inventaire(connection)
    claims = read_claims_from_db(connection)
    ctx = build_resolution_context(inventaire, claims)
    from banc_realite.fidelite import load_gold

    gold = load_gold()
    by_piece = _claims_by_piece(claims)
    table: dict[str, dict[str, Any]] = {}
    for palier in PALIERS:
        table[palier] = build_confusion_for_palier(gold, by_piece, palier=palier, ctx=ctx)
    meilleur = choose_best_palier(table)
    return {
        "paliers": table,
        "resume": summarize_paliers(table),
        "meilleur_palier": meilleur,
    }


def best_deux_cotes(etage2: dict[str, Any]) -> int:
    best = 0
    for couche in COUCHES:
        for k in KS:
            block = etage2["par_couche"][couche][f"k{k}"]
            best = max(best, int(block["complet_avec"]))
    return best


def scope_parts(scope: dict[str, int]) -> dict[str, float]:
    total = sum(scope.values()) or 1
    return {key: round(value / total, 4) for key, value in scope.items()}


def regressions(after: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if after["F1"] < BEFORE_REWRITE["F1"]:
        lines.append(
            f"F1 global baisse : {BEFORE_REWRITE['F1']} → {after['F1']}"
        )
    if after["disputes_reconnues"] < BEFORE_REWRITE["disputes_reconnues"]:
        lines.append(
            f"disputes reconnues baissent : {BEFORE_REWRITE['disputes_reconnues']}/45 → "
            f"{after['disputes_reconnues']}/45"
        )
    if after["pieges_en_dispute"] > BEFORE_REWRITE["pieges_en_dispute"]:
        lines.append(
            f"pièges en dispute augmentent : {BEFORE_REWRITE['pieges_en_dispute']}/11 → "
            f"{after['pieges_en_dispute']}/11"
        )
    if after["deux_cotes_28"] < BEFORE_REWRITE["deux_cotes_28"]:
        lines.append(
            f"deux côtés sur 28 baissent : {BEFORE_REWRITE['deux_cotes_28']}/28 → "
            f"{after['deux_cotes_28']}/28"
        )
    if after["faits_ramenes"] < BEFORE_REWRITE["faits_ramenes"]:
        lines.append(
            f"complétion structurelle baisse : {BEFORE_REWRITE['faits_ramenes']} → "
            f"{after['faits_ramenes']} faits ramenés"
        )
    return lines


def write_report(
    *,
    full: dict[str, Any],
    calls: int,
    embed_stats: dict[str, Any],
    coverage: dict[str, int],
    fidelite: dict[str, Any],
    scopes: dict[str, Any],
    resolution: dict[str, Any],
    etage2: dict[str, Any],
    completion: dict[str, Any],
    parity_payload: dict[str, Any],
    dropped: int,
    unanchored: int,
    faits_ligne: int,
    faits_cellule: int,
    after: dict[str, Any],
) -> None:
    f1 = fidelite.get("F1", {})
    par_voie = f1.get(F1_KEY_PAR_VOIE, {})
    meilleur = resolution["meilleur_palier"]
    best_palier = resolution["paliers"][meilleur]
    scope_after = scopes.get("par_scope", {})
    scope_before = BEFORE_REWRITE["tabular_scope"]
    regress = regressions(after)

    lines = [
        "# Rapport — ingestion 3 (chemin ordinaire)",
        "",
        f"Date : {date.today().isoformat()}",
        "",
        "## Reprise tabulaire (Grok bloqué)",
        "",
        "- Voie prose **non** ré-ingérée : les 881 faits modèle datent de l'ingestion 2.",
        "- Seule la voie tabulaire a été remplacée par le chemin ordinaire, sans appel LLM.",
        "",
    ]
    if regress:
        lines.extend(["## Baisses constatées", ""] + [f"- {line}" for line in regress] + [""])

    lines.extend(
        [
            "## Ingestion",
            f"- Appels modèle : {calls} (plafond {CFG.max_calls})",
            f"- Durée : {ingestion_duration_s(full, CFG.ingest_log)} s",
            f"- Faits ligne (somme journal) : {faits_ligne}",
            f"- Faits cellule (somme journal) : {faits_cellule}",
            f"- Rejets marqueur tabulaire : dropped={dropped}, unanchored={unanchored}",
            f"- Embeddings : {embed_stats}",
            f"- Couverture note.body : {coverage}",
            "",
            "## 1. Fidélité",
            "",
            "| indicateur | avant (réécriture) | après (ordinaire) |",
            "| --- | --- | --- |",
            f"| F1 global | {BEFORE_REWRITE['F1']} | {f1.get(F1_KEY_GLOBAL)} |",
            f"| F1 prose | — | {par_voie.get(F1_KEY_PROSE)} |",
            f"| F1 tabulaire | — | {par_voie.get(F1_KEY_TABULAIRE)} |",
            f"| part row | {scope_parts(scope_before).get('row', 0):.1%} | {scope_parts(scope_after).get('row', 0):.1%} |",
            f"| part cell | {scope_parts(scope_before).get('cell', 0):.1%} | {scope_parts(scope_after).get('cell', 0):.1%} |",
            f"| part sans_scope | {scope_parts(scope_before).get('sans_scope', 0):.1%} | {scope_parts(scope_after).get('sans_scope', 0):.1%} |",
            "",
            "## 2. Tri contre l'or (palier par palier)",
            "",
            "| palier | disputes / 45 | pièges → dispute / 11 | trajectoires / 4 | indécidable |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in resolution["resume"]:
        palier = row["palier"]
        block = resolution["paliers"][palier]
        before_marker = "← avant" if palier == "P2" else ""
        lines.append(
            f"| {palier} {before_marker} | {block['disputes_reconnues']}/45 | "
            f"{block['pieges_en_dispute']}/11 | {block['trajectoires']}/4 | {block['indecidable']} |"
        )
    lines.extend(
        [
            "",
            f"Meilleur palier après : {meilleur} ({best_palier['disputes_reconnues']}/45 disputes).",
            "",
            "## 3. 28 questions dispute (verdict plancher, complet = deux côtés)",
            "",
            "| couche | k | sans trois tours | avec trois tours | Δ |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for couche in COUCHES:
        for k in KS:
            block = etage2["par_couche"][couche][f"k{k}"]
            lines.append(
                f"| {couche} | {k} | {block['complet_sans']}/28 | "
                f"{block['complet_avec']}/28 | {block['delta_complet']:+d} |"
            )
    lines.extend(
        [
            "",
            f"Meilleur deux côtés (avec tours) : {after['deux_cotes_28']}/28 "
            f"(avant réécriture : {BEFORE_REWRITE['deux_cotes_28']}/28).",
            "",
            "## 4. Complétion structurelle (faits cellule)",
            "",
            "| indicateur | avant (réécriture) | après (ordinaire) |",
            "| --- | --- | --- |",
            f"| faits ramenés | {BEFORE_REWRITE['faits_ramenes']} | {completion['faits_ramenes']} |",
            f"| depuis autre pièce | {BEFORE_REWRITE['depuis_autre_piece']} | {completion['depuis_autre_piece']} |",
            "",
            "## Parité brain → public",
            "",
            "```json",
            json.dumps(parity_payload.get("parity", {}), ensure_ascii=False, indent=2),
            "```",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sum_journal_scopes(
    ingest_log: Path,
    *,
    mode: str | None = None,
    run_id: str | None = None,
) -> tuple[int, int]:
    ligne = cellule = 0
    if not ingest_log.is_file():
        return 0, 0
    for line in ingest_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if run_id is not None and record.get("run_id") != run_id:
            continue
        if record.get("skipped"):
            continue
        if mode is not None and record.get("mode") != mode:
            continue
        ligne += int(record.get("faits_ligne", 0))
        cellule += int(record.get("faits_cellule", 0))
    return ligne, cellule


def main() -> int:
    started = time.monotonic()
    if completion_markers_present():
        write_arret("preflight", f"ingestion-3 déjà complète: {completion_markers_present()}")
        return 1

    INGESTION3_SORTIES.mkdir(parents=True, exist_ok=True)
    _runtime_env()
    start_cluster()
    connection = connect_existing()

    try:
        export_memory_dump()
    except Exception as exc:  # noqa: BLE001
        write_arret("0_export", str(exc))
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    model_before = model_claim_count(connection)
    if model_before < 881:
        write_arret(
            "0_reprise",
            f"faits modèle insuffisants ({model_before}/881) — restaurer le dump avant reprise",
        )
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    restart_postgrest_brain()
    reprise_cfg = IngestionConfig.for_ingestion3_reprise()
    calls_before = load_call_count(reprise_cfg.call_log)

    full = run_tabular_ordinary_reprise(config=reprise_cfg)
    calls = load_call_count(reprise_cfg.call_log) - calls_before
    if calls > 0 or full.get("arret") == "appel_modele_interdit":
        write_arret("1_tabulaire", f"appel modèle interdit détecté ({calls})")
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1
    if full.get("erreurs"):
        write_arret("1_tabulaire", f"pièces en erreur: {full['erreurs']}")
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1
    if model_claim_count(connection) != model_before:
        write_arret(
            "1_tabulaire",
            f"faits modèle altérés: {model_before} → {model_claim_count(connection)}",
        )
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    embed_stats = backfill_embeddings(
        retry_attempts=10,
        retry_delay_s=30.0,
        max_embeddings=INGESTION3_MAX_EMBEDDINGS,
        log_path=EMBED_LOG,
    )
    coverage = embedding_coverage(connection, vault_ref="greenloop")

    ok, parity_payload, parity_why = run_brain_to_public_pipeline(connection, smoke_question_text())
    if not ok:
        write_arret("3_parite", parity_why or "parité non atteinte")
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    fidelite = measure_fidelite(output_dir=INGESTION3_SORTIES)
    if fidelite.get("status") == "error":
        write_arret("4_fidelite", fidelite.get("message", "mesure en erreur"))
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
        return 1

    scopes = count_by_scope(connection)
    resolution = measure_resolution_paliers(connection)
    (INGESTION3_SORTIES / "resolution-paliers.json").write_text(
        json.dumps(resolution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    embed_cache = _load_embed_cache()
    from banc_realite.fidelite import load_gold

    gold = load_gold()
    public_claims = _read_public_claims(connection)
    claims_index = {str(c["claim_id"]): c for c in public_claims}
    build_holder_table(public_claims, gold)
    etage2 = measure_etage2_disputes(
        connection,
        embed_cache=embed_cache,
        gold=gold,
        claims_index=claims_index,
    )
    (INGESTION3_SORTIES / "etage2-disputes.json").write_text(
        json.dumps(etage2, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    completion = measure_completion(connection)
    (INGESTION3_SORTIES / "completion-structurelle.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    meilleur = resolution["meilleur_palier"]
    best_palier = resolution["paliers"][meilleur]
    dropped, unanchored = sum_tabular_rejections_from_journal(
        INGESTION3_REPRISE_LOG, run_id=full["run_id"]
    )
    faits_ligne, faits_cellule = sum_journal_scopes(
        INGESTION3_REPRISE_LOG,
        mode="tabulaire-ordinaire-reprise",
        run_id=full["run_id"],
    )
    f1_global = fidelite.get("F1", {}).get(F1_KEY_GLOBAL)
    after = {
        "F1": f1_global,
        "disputes_reconnues": best_palier["disputes_reconnues"],
        "pieges_en_dispute": best_palier["pieges_en_dispute"],
        "deux_cotes_28": best_deux_cotes(etage2),
        "faits_ramenes": completion["faits_ramenes"],
        "depuis_autre_piece": completion["depuis_autre_piece"],
    }

    write_report(
        full=full,
        calls=calls,
        embed_stats=embed_stats,
        coverage=coverage,
        fidelite=fidelite,
        scopes=scopes,
        resolution=resolution,
        etage2=etage2,
        completion=completion,
        parity_payload=parity_payload,
        dropped=dropped,
        unanchored=unanchored,
        faits_ligne=faits_ligne,
        faits_cellule=faits_cellule,
        after=after,
    )

    RESULT.write_text(
        json.dumps(
            {
                "status": "complete",
                "appels_modele": calls,
                "faits_ligne": faits_ligne,
                "faits_cellule": faits_cellule,
                "F1": f1_global,
                "disputes_reconnues": after["disputes_reconnues"],
                "pieges_en_dispute": after["pieges_en_dispute"],
                "deux_cotes_28": after["deux_cotes_28"],
                "duree_s": round(time.monotonic() - started, 1),
                "files_written": sorted(
                    str(p.relative_to(BANC_ROOT))
                    for p in [
                        DUMP_PATH,
                        INGESTION3_REPRISE_LOG,
                        reprise_cfg.call_log,
                        EMBED_LOG,
                        INGESTION3_SORTIES / "fidelite-C.json",
                        INGESTION3_SORTIES / "resolution-paliers.json",
                        INGESTION3_SORTIES / "etage2-disputes.json",
                        INGESTION3_SORTIES / "completion-structurelle.json",
                        REPORT,
                        RESULT,
                    ]
                    if p.exists()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
