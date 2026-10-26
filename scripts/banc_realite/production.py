# SPDX-License-Identifier: AGPL-3.0-only
"""Migrate banc DB to production-shaped public schema."""
from __future__ import annotations

import re
import subprocess
import time
from http.client import HTTPConnection
from typing import Any

from .config import BANC_ROOT, SCHEMA_PRODUCTION, SMOKE_QUESTION_ID, VAULT_REF
from .infra import PG_BIN, POSTGREST, pid_on_port, wait_postgrest_ready

EXPECTED_FUNCTIONS = frozenset(
    {
        "brain_claim_lex_text",
        "brain_jsonb_as_text",
        "brain_unaccent",
        "photo_upload_prepare_asset",
        "photo_upload_record_asset_outcome",
        "photo_upload_record_outcome",
        "recall_match_mistral",
        "search_claims_lexical",
        "set_updated_at",
    }
)


def _strip_schema_sql(raw: str) -> str:
    dump_set_prefixes = (
        "SET statement_timeout",
        "SET lock_timeout",
        "SET idle_in_transaction_session_timeout",
        "SET client_encoding",
        "SET standard_conforming_strings",
        "SET check_function_bodies",
        "SET xmloption",
        "SET client_min_messages",
        "SET row_security",
        "SET default_tablespace",
        "SET default_table_access_method",
    )
    lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("\\"):
            continue
        if line.strip().startswith("SELECT pg_catalog.set_config"):
            continue
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in dump_set_prefixes):
            continue
        if stripped.startswith("COMMENT ON SCHEMA"):
            continue
        lines.append(line)
    return "\n".join(lines)


def ensure_extensions(connection) -> list[str]:
    connection.execute("CREATE SCHEMA IF NOT EXISTS extensions;")
    missing: list[str] = []
    for ext in ("vector", "unaccent", "pgcrypto", "uuid-ossp"):
        try:
            connection.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}" WITH SCHEMA extensions;')
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{ext}: {exc}")
    return missing


def _extract_function_ddl(raw: str, name: str) -> str | None:
    pattern = rf"CREATE FUNCTION public\.{re.escape(name)}\(.*?\$\$;"
    match = re.search(pattern, raw, re.S)
    if not match:
        return None
    return match.group(0).replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1)


def _schema_blocks(raw: str) -> tuple[list[str], list[str], list[str]]:
    """Split pg_dump DDL into tables, functions, and the remainder."""
    blocks: list[str] = []
    current: list[str] = []
    starters = ("CREATE TABLE", "CREATE FUNCTION", "ALTER TABLE", "CREATE INDEX", "CREATE TRIGGER", "CREATE POLICY", "COMMENT", "CREATE SCHEMA")
    for line in raw.splitlines():
        if any(line.startswith(prefix) for prefix in starters):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    tables: list[str] = []
    functions: list[str] = []
    rest: list[str] = []
    for block in blocks:
        if block.startswith("CREATE TABLE"):
            tables.append(block)
        elif block.startswith("CREATE FUNCTION"):
            functions.append(block.replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1))
        else:
            rest.append(block)
    return tables, functions, rest


def apply_production_schema(connection) -> None:
    connection.execute("SET search_path TO public, extensions")
    raw = _strip_schema_sql(SCHEMA_PRODUCTION.read_text(encoding="utf-8"))
    raw = raw.replace("CREATE SCHEMA public;", "CREATE SCHEMA IF NOT EXISTS public;")
    helper_blocks: list[str] = []
    body = raw
    for helper in ("brain_jsonb_as_text", "brain_unaccent", "brain_claim_lex_text"):
        block = _extract_function_ddl(raw, helper)
        if block:
            helper_blocks.append(block)
            body = re.sub(
                rf"CREATE FUNCTION public\.{re.escape(helper)}\(.*?\$\$;\s*",
                "",
                body,
                count=1,
                flags=re.S,
            )
    tables, functions, rest = _schema_blocks(body)
    rest = [
        block
        for block in rest
        if not block.startswith("CREATE POLICY")
        and "ENABLE ROW LEVEL SECURITY" not in block
    ]
    sql_path = BANC_ROOT / "sorties" / "_schema-apply.sql"
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    phases = [
        "SET search_path TO public, extensions;\nCREATE SCHEMA IF NOT EXISTS public;\n",
        *helper_blocks,
        *tables,
        *functions,
        *rest,
    ]
    sql_path.write_text("\n\n".join(phases) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            str(PG_BIN / "psql"),
            "-h",
            "127.0.0.1",
            "-p",
            "56102",
            "-U",
            "postgres",
            "-d",
            "postgres",
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            str(sql_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "schema apply failed: " + (result.stderr or result.stdout)[-2000:]
        )


def _public_columns(connection) -> list[str]:
    rows = connection.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'claims' "
        "AND is_generated = 'NEVER' ORDER BY ordinal_position"
    ).fetchall()
    return [row["column_name"] for row in rows]


def _brain_columns(connection) -> list[str]:
    rows = connection.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'brain' AND table_name = 'claims' "
        "ORDER BY ordinal_position"
    ).fetchall()
    return [row["column_name"] for row in rows]


def copy_claims_to_public(connection) -> dict[str, int]:
    public_cols = set(_public_columns(connection))
    brain_cols = set(_brain_columns(connection))
    shared = sorted(public_cols & brain_cols)
    if not shared:
        raise RuntimeError("no shared columns between brain.claims and public.claims")
    connection.execute("TRUNCATE public.claim_assets, public.claims;")
    select_exprs = []
    for col in shared:
        if col == "extracted_by_actor":
            select_exprs.append("COALESCE(extracted_by_actor, '00000000-0000-4000-8000-b00000000001'::uuid)")
        elif col == "extracted_by_skill":
            select_exprs.append("COALESCE(extracted_by_skill, 'unknown')")
        elif col == "extracted_by_model":
            select_exprs.append("COALESCE(extracted_by_model, 'unknown')")
        else:
            select_exprs.append(col)
    cols_sql = ", ".join(shared)
    select_sql = ", ".join(select_exprs)
    connection.execute(
        f"INSERT INTO public.claims ({cols_sql}) "
        f"SELECT {select_sql} FROM brain.claims "
        "ON CONFLICT (claim_id) DO NOTHING"
    )
    brain_total = connection.execute("SELECT COUNT(*) AS n FROM brain.claims").fetchone()["n"]
    public_total = connection.execute("SELECT COUNT(*) AS n FROM public.claims").fetchone()["n"]
    return {"brain": int(brain_total), "public": int(public_total), "columns": len(shared)}


def restart_postgrest_brain(*, pg_port: int = 56102) -> None:
    postgrest_pid = pid_on_port(56101)
    if postgrest_pid is not None:
        subprocess.run(["kill", str(postgrest_pid)], check=False)
        time.sleep(0.5)
        if pid_on_port(56101) is not None:
            subprocess.run(["kill", "-9", str(postgrest_pid)], check=False)
            time.sleep(0.5)
    import os

    env = {
        **os.environ,
        "PGRST_DB_URI": f"postgresql://postgres@127.0.0.1:{pg_port}/postgres",
        "PGRST_DB_SCHEMAS": "brain",
        "PGRST_DB_ANON_ROLE": "postgres",
        "PGRST_DB_POOL": "4",
        "PGRST_SERVER_HOST": "127.0.0.1",
        "PGRST_SERVER_PORT": "56101",
    }
    log = BANC_ROOT / "pg" / "postgrest.log"
    with log.open("a") as handle:
        subprocess.Popen([str(POSTGREST)], stdout=handle, stderr=subprocess.STDOUT, env=env)
    wait_postgrest_ready(postgrest_port=56101, timeout_s=60.0)
    connection = HTTPConnection("127.0.0.1", 56101, timeout=5)
    connection.request("GET", "/claims?limit=0")
    response = connection.getresponse()
    response.read()
    connection.close()
    if response.status != 200:
        raise RuntimeError(f"PostgREST brain schema probe failed: {response.status}")


def restart_postgrest_public(*, service_key: str, pg_port: int = 56102) -> None:
    postgrest_pid = pid_on_port(56101)
    if postgrest_pid is not None:
        subprocess.run(["kill", str(postgrest_pid)], check=False)
        time.sleep(0.5)
        if pid_on_port(56101) is not None:
            subprocess.run(["kill", "-9", str(postgrest_pid)], check=False)
            time.sleep(0.5)
    import os

    env = {
        **os.environ,
        "PGRST_DB_URI": f"postgresql://postgres@127.0.0.1:{pg_port}/postgres",
        "PGRST_DB_SCHEMAS": "public",
        "PGRST_DB_EXTRA_SEARCH_PATH": "extensions",
        "PGRST_DB_ANON_ROLE": "postgres",
        "PGRST_DB_POOL": "4",
        "PGRST_SERVER_HOST": "127.0.0.1",
        "PGRST_SERVER_PORT": "56101",
    }
    log = BANC_ROOT / "pg" / "postgrest-public.log"
    with log.open("a") as handle:
        subprocess.Popen([str(POSTGREST)], stdout=handle, stderr=subprocess.STDOUT, env=env)
    wait_postgrest_ready(
        url="http://127.0.0.1:56100",
        service_key=service_key,
        postgrest_port=56101,
        timeout_s=60.0,
    )
    connection = HTTPConnection("127.0.0.1", 56101, timeout=5)
    connection.request("GET", "/claims?limit=0")
    response = connection.getresponse()
    response.read()
    connection.close()
    if response.status != 200:
        raise RuntimeError(f"PostgREST public schema probe failed: {response.status}")
    from .infra import ensure_proxy_running, is_running

    ensure_proxy_running()
    if not is_running()["56100"]:
        raise RuntimeError("proxy 56100 not running after postgrest restart")


def _count_by_vault_ref(connection, schema: str) -> dict[str, int]:
    rows = connection.execute(
        f"SELECT COALESCE(vault_ref, '(null)') AS k, COUNT(*) AS n "
        f"FROM {schema}.claims GROUP BY 1 ORDER BY 1"
    ).fetchall()
    return {row["k"]: int(row["n"]) for row in rows}


def _embeddings_by_vault_ref(connection, schema: str) -> dict[str, int]:
    rows = connection.execute(
        f"SELECT COALESCE(vault_ref, '(null)') AS k, COUNT(*) AS n "
        f"FROM {schema}.claims WHERE value_embedding_mistral IS NOT NULL "
        "GROUP BY 1 ORDER BY 1"
    ).fetchall()
    return {row["k"]: int(row["n"]) for row in rows}


def verifier_parite(dossier) -> None:
    """Échoue si brain.claims et public.claims divergent pour ce vault_ref."""
    from .dossiers import Dossier, dossier_courant
    from .infra import connect_existing

    current = dossier if isinstance(dossier, Dossier) else dossier_courant(dossier)
    vault_ref = current.vault_ref
    connection = connect_existing()
    brain_count = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE vault_ref = %s",
        (vault_ref,),
    ).fetchone()["n"]
    public_count = connection.execute(
        "SELECT COUNT(*) AS n FROM public.claims WHERE vault_ref = %s",
        (vault_ref,),
    ).fetchone()["n"]
    if int(brain_count) != int(public_count):
        raise RuntimeError(
            f"parité {vault_ref}: comptes brain={brain_count} public={public_count}"
        )
    brain_ids = {
        str(row["claim_id"])
        for row in connection.execute(
            "SELECT claim_id FROM brain.claims WHERE vault_ref = %s",
            (vault_ref,),
        ).fetchall()
    }
    public_ids = {
        str(row["claim_id"])
        for row in connection.execute(
            "SELECT claim_id FROM public.claims WHERE vault_ref = %s",
            (vault_ref,),
        ).fetchall()
    }
    missing_public = sorted(brain_ids - public_ids)
    missing_brain = sorted(public_ids - brain_ids)
    if missing_public or missing_brain:
        raise RuntimeError(
            f"parité {vault_ref}: claim_id manquants "
            f"public={missing_public[:5]} brain={missing_brain[:5]}"
        )
    brain_emb = {
        str(row["claim_id"]): row["has_embedding"]
        for row in connection.execute(
            "SELECT claim_id, value_embedding_mistral IS NOT NULL AS has_embedding "
            "FROM brain.claims WHERE vault_ref = %s",
            (vault_ref,),
        ).fetchall()
    }
    public_emb = {
        str(row["claim_id"]): row["has_embedding"]
        for row in connection.execute(
            "SELECT claim_id, value_embedding_mistral IS NOT NULL AS has_embedding "
            "FROM public.claims WHERE vault_ref = %s",
            (vault_ref,),
        ).fetchall()
    }
    vector_mismatch = sorted(
        claim_id for claim_id in brain_ids if brain_emb.get(claim_id) != public_emb.get(claim_id)
    )
    if vector_mismatch:
        raise RuntimeError(
            f"parité {vault_ref}: vecteurs divergents pour {vector_mismatch[:5]}"
        )


def parity_counts(connection) -> dict[str, Any]:
    def count_group(schema: str, group_col: str) -> dict[str, int]:
        rows = connection.execute(
            f"SELECT COALESCE({group_col}, '(null)') AS k, COUNT(*) AS n "
            f"FROM {schema}.claims GROUP BY 1 ORDER BY 1"
        ).fetchall()
        return {row["k"]: int(row["n"]) for row in rows}

    brain_by_vault_ref = _count_by_vault_ref(connection, "brain")
    public_by_vault_ref = _count_by_vault_ref(connection, "public")
    brain_embeddings_by_vault_ref = _embeddings_by_vault_ref(connection, "brain")
    public_embeddings_by_vault_ref = _embeddings_by_vault_ref(connection, "public")
    brain_total = connection.execute("SELECT COUNT(*) AS n FROM brain.claims").fetchone()["n"]
    public_total = connection.execute("SELECT COUNT(*) AS n FROM public.claims").fetchone()["n"]
    brain_skill = count_group("brain", "extracted_by_skill")
    public_skill = count_group("public", "extracted_by_skill")
    brain_lifecycle = count_group("brain", "lifecycle_status")
    public_lifecycle = count_group("public", "lifecycle_status")
    brain_emb = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims WHERE value_embedding_mistral IS NOT NULL"
    ).fetchone()["n"]
    public_emb = connection.execute(
        "SELECT COUNT(*) AS n FROM public.claims WHERE value_embedding_mistral IS NOT NULL"
    ).fetchone()["n"]
    lex_null = connection.execute(
        "SELECT COUNT(*) AS n FROM public.claims "
        "WHERE lifecycle_status = 'active' AND lex_tsv IS NULL"
    ).fetchone()["n"]
    func_rows = connection.execute(
        "SELECT proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = 'public' AND p.prokind = 'f'"
    ).fetchall()
    functions = sorted(row["proname"] for row in func_rows)
    return {
        "brain_total": int(brain_total),
        "public_total": int(public_total),
        "brain_by_vault_ref": brain_by_vault_ref,
        "public_by_vault_ref": public_by_vault_ref,
        "brain_embeddings_by_vault_ref": brain_embeddings_by_vault_ref,
        "public_embeddings_by_vault_ref": public_embeddings_by_vault_ref,
        "brain_by_skill": brain_skill,
        "public_by_skill": public_skill,
        "brain_by_lifecycle": brain_lifecycle,
        "public_by_lifecycle": public_lifecycle,
        "brain_embeddings": int(brain_emb),
        "public_embeddings": int(public_emb),
        "lex_tsv_null_active": int(lex_null),
        "public_functions": functions,
    }


def smoke_run_search(question: str) -> dict[str, Any]:
    from banc_realite.compat_moteur import run_search

    from .config import AREA, INSTANCE

    result = run_search(
        question,
        instance=INSTANCE,
        repo_root=AREA,
        top_k=8,
        vault_ref=VAULT_REF,
    )
    strategy = result.get("strategy") or {}
    warnings = list(result.get("warnings") or [])
    evidence = list(result.get("evidence") or [])
    kinds = {item.get("kind") for item in evidence}
    legs = set((strategy.get("legs") or {}).keys()) if isinstance(strategy.get("legs"), dict) else set()
    fallback_notes = [
        w
        for w in warnings
        if any(
            token in str(w).lower()
            for token in (
                "degraded",
                "repli",
                "rpc",
                "absent",
                "unavailable",
                "indisponible",
                "tranche 2000",
            )
        )
    ]
    return {
        "question_id": SMOKE_QUESTION_ID,
        "strategy": strategy,
        "warnings": warnings,
        "fallback_notes": fallback_notes,
        "evidence_kinds": sorted(k for k in kinds if k),
        "has_lexical": bool(
            {"claims_lexical", "claims_hybrid"} & legs
            or any(item.get("kind") == "claim_hit" for item in evidence)
        ),
        "has_vector": bool(
            {"claims_vector", "claims_hybrid"} & legs
            or any(item.get("kind") == "claim_hit" for item in evidence)
        ),
        "ok": not fallback_notes,
    }


def plan_brain_to_public_pipeline() -> list[str]:
    return [
        "postgrest_brain",
        "ingestion",
        "embeddings",
        "copy_public",
        "postgrest_public",
        "parity",
    ]


def verify_parity(connection, smoke_question: str) -> tuple[bool, dict[str, Any], str | None]:
    counts = parity_counts(connection)
    reasons: list[str] = []
    if counts["brain_by_vault_ref"] != counts["public_by_vault_ref"]:
        reasons.append(
            "vault_ref counts differ: "
            f"brain={counts['brain_by_vault_ref']} public={counts['public_by_vault_ref']}"
        )
    if counts["brain_embeddings_by_vault_ref"] != counts["public_embeddings_by_vault_ref"]:
        reasons.append(
            "embedding counts by vault_ref differ: "
            f"brain={counts['brain_embeddings_by_vault_ref']} "
            f"public={counts['public_embeddings_by_vault_ref']}"
        )
    if counts["brain_total"] != counts["public_total"]:
        reasons.append(
            f"total mismatch brain={counts['brain_total']} public={counts['public_total']}"
        )
    if counts["brain_by_skill"] != counts["public_by_skill"]:
        reasons.append("extracted_by_skill counts differ")
    if counts["brain_by_lifecycle"] != counts["public_by_lifecycle"]:
        reasons.append("lifecycle_status counts differ")
    if counts["brain_embeddings"] != counts["public_embeddings"]:
        reasons.append("embedding counts differ")
    if counts["lex_tsv_null_active"] != 0:
        reasons.append(f"{counts['lex_tsv_null_active']} active claims with null lex_tsv")
    actual_funcs = set(counts["public_functions"])
    if actual_funcs != EXPECTED_FUNCTIONS:
        missing = sorted(EXPECTED_FUNCTIONS - actual_funcs)
        extra = sorted(actual_funcs - EXPECTED_FUNCTIONS)
        if missing:
            reasons.append(f"missing functions: {missing}")
        if extra:
            reasons.append(f"extra functions: {extra}")
    smoke = smoke_run_search(smoke_question)
    counts["smoke"] = smoke
    if smoke["fallback_notes"]:
        reasons.append(f"smoke fallback notes: {smoke['fallback_notes']}")
    if not (smoke["has_lexical"] and smoke["has_vector"]):
        reasons.append("smoke missing lexical or vector leg")
    ok = not reasons
    return ok, counts, "; ".join(reasons) if reasons else None
