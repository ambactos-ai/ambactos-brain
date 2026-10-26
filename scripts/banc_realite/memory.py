# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only access to banc claims memory."""
from __future__ import annotations

from typing import Any

from .normalize import claim_text, piece_basename


def read_public_active_claims(connection, *, vault_ref: str | None = None) -> list[dict[str, Any]]:
    sql = (
        "SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path, "
        "source_span, extracted_by_skill, lifecycle_status, validation_state, extracted_at, "
        "created_at, value_embedding_mistral IS NOT NULL AS has_embedding "
        "FROM public.claims WHERE lifecycle_status = 'active'"
    )
    params: list[Any] = []
    if vault_ref is not None:
        sql += " AND vault_ref = %s"
        params.append(vault_ref)
    sql += " ORDER BY created_at, claim_id"
    rows = connection.execute(sql, params).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["text"] = claim_text(item.get("value"))
        item["piece"] = piece_basename(item.get("source_file_path") or "")
        out.append(item)
    return out


def read_active_claims(connection, *, vault_ref: str | None = None) -> list[dict[str, Any]]:
    sql = (
        "SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path, "
        "source_span, extracted_by_skill, lifecycle_status, validation_state, extracted_at, "
        "created_at, value_embedding_mistral IS NOT NULL AS has_embedding "
        "FROM brain.claims WHERE lifecycle_status = 'active'"
    )
    params: list[Any] = []
    if vault_ref is not None:
        sql += " AND vault_ref = %s"
        params.append(vault_ref)
    sql += " ORDER BY created_at, claim_id"
    rows = connection.execute(sql, params).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["text"] = claim_text(item.get("value"))
        item["piece"] = piece_basename(item.get("source_file_path") or "")
        out.append(item)
    return out


def memory_elements_by_piece(claims: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in claims:
        piece = row.get("piece") or ""
        grouped.setdefault(piece, []).append(row)
    return grouped


def embedding_coverage(connection, *, vault_ref: str | None = None) -> dict[str, int]:
    sql = (
        "SELECT COUNT(*) FILTER (WHERE predicate = 'note.body') AS note_body, "
        "COUNT(*) FILTER (WHERE predicate = 'note.body' AND value_embedding_mistral IS NOT NULL) "
        "AS note_body_embedded FROM brain.claims WHERE lifecycle_status = 'active'"
    )
    params: list[Any] = []
    if vault_ref is not None:
        sql += " AND vault_ref = %s"
        params.append(vault_ref)
    row = connection.execute(sql, params).fetchone()
    return {"note_body": int(row["note_body"]), "note_body_embedded": int(row["note_body_embedded"])}
