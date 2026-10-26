# SPDX-License-Identifier: AGPL-3.0-only
"""Partie A — memory refresh and production-shaped public schema."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    from brain.claiming.tabular import TABULAR_SKILL, extract_tabular_claims
except ImportError:
    TABULAR_SKILL = None  # type: ignore[misc,assignment]
    extract_tabular_claims = None  # type: ignore[misc,assignment]

from .config import BANC_ROOT, SORTIES, VAULT_REF
from .ingest_runner import (
    backfill_embeddings,
    list_corpus,
    run_tabulaire_seul,
    source_path_for,
)
from .infra import connect_existing, start_cluster
from .production import (
    apply_production_schema,
    copy_claims_to_public,
    ensure_extensions,
    restart_postgrest_public,
    verify_parity,
)

PURGE_LOG = SORTIES / "purge-modele-tables.jsonl"


def _claim_span_within_table(claim: dict[str, Any], table_start: int, table_end: int) -> bool:
    span = claim.get("source_span") or {}
    line_start = span.get("line_start")
    line_end = span.get("line_end", line_start)
    if line_start is None or line_end is None:
        return False
    return int(line_start) >= int(table_start) and int(line_end) <= int(table_end)


def purge_model_marker_claims(connection) -> dict[str, Any]:
    """Émule ``_drop_marker_claims`` de main ; défaut connu : se fie aux lignes annoncées par le modèle.

    Delete model facts whose source_span falls entirely inside a masked table.
    """
    PURGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    before = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE lifecycle_status = 'active'"
    ).fetchone()["n"]
    deleted = 0
    for path in list_corpus():
        text = path.read_text(encoding="utf-8")
        parsed = extract_tabular_claims(file_name=path.name, text=text)
        if not parsed.tables:
            continue
        rows = connection.execute(
            "SELECT claim_id, entity_id, predicate, value, source_span, extracted_by_skill "
            "FROM brain.claims WHERE source_file_path = %s AND vault_ref = %s "
            "AND lifecycle_status = 'active' AND COALESCE(extracted_by_skill, '') != %s",
            (source_path_for(path.name), VAULT_REF, TABULAR_SKILL),
        ).fetchall()
        for row in rows:
            if any(
                _claim_span_within_table(dict(row), table.start_line, table.end_line)
                for table in parsed.tables
            ):
                connection.execute(
                    "DELETE FROM brain.claims WHERE claim_id = %s",
                    (row["claim_id"],),
                )
                deleted += 1
                record = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "file": path.name,
                    "claim_id": str(row["claim_id"]),
                    "entity_id": row["entity_id"],
                    "value": str(row["value"])[:500],
                }
                with PURGE_LOG.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    after = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE lifecycle_status = 'active'"
    ).fetchone()["n"]
    return {"before": int(before), "after": int(after), "deleted": deleted}


def memory_counts(connection) -> dict[str, Any]:
    total = connection.execute("SELECT COUNT(*) AS n FROM brain.claims").fetchone()["n"]
    active = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE lifecycle_status = 'active'"
    ).fetchone()["n"]
    tab = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims "
        "WHERE lifecycle_status = 'active' AND extracted_by_skill = %s",
        (TABULAR_SKILL,),
    ).fetchone()["n"]
    model = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims "
        "WHERE lifecycle_status = 'active' AND COALESCE(extracted_by_skill, '') != %s",
        (TABULAR_SKILL,),
    ).fetchone()["n"]
    emb = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims "
        "WHERE lifecycle_status = 'active' AND predicate = 'note.body' "
        "AND value_embedding_mistral IS NOT NULL"
    ).fetchone()["n"]
    return {
        "total": int(total),
        "active": int(active),
        "tabulaire": int(tab),
        "modele": int(model),
        "note_body_embedded": int(emb),
    }


def run_step0(connection, *, purge_markers: bool = False) -> dict[str, Any]:
    before = memory_counts(connection)
    tabulaire = run_tabulaire_seul()
    purge = purge_model_marker_claims(connection) if purge_markers else {"skipped": True}
    embedded = backfill_embeddings()
    after = memory_counts(connection)
    return {
        "before": before,
        "tabulaire": tabulaire,
        "purge": purge,
        "embedded": embedded,
        "after": after,
    }


def _public_schema_ready(connection) -> bool:
    row = connection.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = 'claims'"
    ).fetchone()
    return row is not None


def _reset_broken_public_schema(connection) -> None:
    row = connection.execute(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'public'"
    ).fetchone()
    if row and not _public_schema_ready(connection):
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")
        connection.execute("GRANT ALL ON SCHEMA public TO postgres")
        connection.execute("GRANT ALL ON SCHEMA public TO public")


def verify_parity_readonly(smoke_question: str) -> tuple[bool, dict[str, Any], str | None]:
    """Read-only parity check (default measurement path)."""
    connection = connect_existing()
    service_key = ""
    secrets = BANC_ROOT / "secrets.env"
    if secrets.is_file():
        for line in secrets.read_text(encoding="utf-8").splitlines():
            if line.startswith("BANC_BRAIN_SERVICE_KEY="):
                service_key = line.split("=", 1)[1].strip()
    restart_postgrest_public(service_key=service_key)
    ok, parity, why = verify_parity(connection, smoke_question)
    return ok, {"parity": parity}, why


def run_partie_a(
    smoke_question: str,
    *,
    purge_markers: bool = False,
) -> tuple[bool, dict[str, Any], str | None]:
    start_cluster()
    connection = connect_existing()
    step0 = run_step0(connection, purge_markers=purge_markers)
    missing_ext = ensure_extensions(connection)
    if missing_ext:
        return False, {"step0": step0, "missing_extensions": missing_ext}, (
            f"extensions manquantes: {missing_ext}"
        )
    if not _public_schema_ready(connection):
        _reset_broken_public_schema(connection)
        apply_production_schema(connection)
    copy_stats = copy_claims_to_public(connection)
    service_key = ""
    secrets = BANC_ROOT / "secrets.env"
    if secrets.is_file():
        for line in secrets.read_text(encoding="utf-8").splitlines():
            if line.startswith("BANC_BRAIN_SERVICE_KEY="):
                service_key = line.split("=", 1)[1].strip()
    restart_postgrest_public(service_key=service_key)
    ok, parity, why = verify_parity(connection, smoke_question)
    payload = {
        "step0": step0,
        "copy": copy_stats,
        "parity": parity,
    }
    return ok, payload, why
