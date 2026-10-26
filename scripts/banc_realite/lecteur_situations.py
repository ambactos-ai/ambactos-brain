# SPDX-License-Identifier: AGPL-3.0-only
"""Lecteur MRFS pour l'instantané du constructeur de situations."""
from __future__ import annotations

from typing import Any, Mapping

try:
    from brain.mrfs.enums import SearchEvidenceKind, SearchMode, SearchPermission
except ImportError:
    SearchEvidenceKind = None  # type: ignore[misc,assignment]
    SearchMode = None  # type: ignore[misc,assignment]
    SearchPermission = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.projection_repository import PostgresProjectionSearchBackend
except ImportError:
    PostgresProjectionSearchBackend = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.search import (
        SearchActorContext,
        SearchExpansion,
        SearchHit,
        SearchPagination,
        SearchRequest,
    )
except ImportError:
    SearchActorContext = None  # type: ignore[misc,assignment]
    SearchExpansion = None  # type: ignore[misc,assignment]
    SearchHit = None  # type: ignore[misc,assignment]
    SearchPagination = None  # type: ignore[misc,assignment]
    SearchRequest = None  # type: ignore[misc,assignment]

from .config import BANC_ACTOR_ID
from .constructeur_situations import (
    ALLOWED_LECTEUR_DOSSIERS,
    CONSTRUCTEUR_PROJECTION_DEFINITION_ID,
    CONSTRUCTEUR_PROJECTION_DEFINITION_VERSION,
    _connect_factory,
    _ranking_profile,
    latest_constructeur_snapshot_id,
)
from .mrfs_support import BANC_PG_DSN, MRFS_SCHEMA, load_subject_registry


class ConstructeurSnapshotError(RuntimeError):
    """Instantané constructeur absent ou vide."""


def _subject_id_for_case(dossier: str, *, dsn: str) -> str:
    if dossier not in ALLOWED_LECTEUR_DOSSIERS:
        raise ValueError(
            f"dossier non autorisé : {dossier!r} "
            f"(attendu : {sorted(ALLOWED_LECTEUR_DOSSIERS)})"
        )
    registry = load_subject_registry()
    subject_id = registry.get(dossier)
    if not subject_id:
        raise ConstructeurSnapshotError(
            f"aucun sujet MRFS enregistré pour le dossier {dossier!r}"
        )
    return subject_id


def _latest_constructeur_snapshot_id(subject_id: str, *, dsn: str, schema: str) -> str:
    import psycopg

    snapshot_id = latest_constructeur_snapshot_id(subject_id, dsn=dsn, schema=schema)
    if snapshot_id is None:
        raise ConstructeurSnapshotError("aucun instantané constructeur pour ce sujet")
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        count = connection.execute(
            f"SELECT count(*) FROM {schema}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s "
            "AND record_key LIKE 'constructeur:%%'",
            (subject_id, snapshot_id),
        ).fetchone()[0]
    finally:
        connection.close()
    if int(count) == 0:
        raise ConstructeurSnapshotError("instantané constructeur vide")
    return snapshot_id


def _perspective_to_element(
    perspective: Mapping[str, Any],
    *,
    hit: SearchHit,
    rank: int,
) -> dict[str, Any] | None:
    text = str(perspective.get("text") or "").strip()
    source = str(perspective.get("source_file") or "").strip()
    if not text or not source:
        return None
    line_start = perspective.get("line_start")
    line_end = perspective.get("line_end")
    element: dict[str, Any] = {
        "kind": "mrfs_situation",
        "rank": rank,
        "score": float(hit.score),
        "text": text,
        "source": source,
        "enonciateur": perspective.get("enonciateur"),
        "date": perspective.get("date"),
        "alternatives": perspective.get("alternatives") or [],
        "provenance": {
            "source_file_path": source,
            "mrfs_object_kind": hit.kind.value,
            "mrfs_object_id": hit.object_ref.id,
        },
    }
    if isinstance(line_start, int):
        element["provenance"]["line_start"] = line_start
    if isinstance(line_end, int):
        element["provenance"]["line_end"] = line_end
    return element


def _hit_to_elements(hit: SearchHit, *, rank: int) -> list[dict[str, Any]]:
    summary = hit.perspective_summary or {}
    if isinstance(summary, Mapping) and isinstance(summary.get("propositions"), list):
        propositions = [dict(item) for item in summary["propositions"] if isinstance(item, Mapping)]
        if not propositions:
            return []
        shared_alternatives = summary.get("alternatives")
        elements: list[dict[str, Any]] = []
        for perspective in propositions:
            payload = dict(perspective)
            if shared_alternatives and not payload.get("alternatives"):
                payload["alternatives"] = [
                    alt
                    for alt in shared_alternatives
                    if alt.get("citation") != payload.get("text")
                ]
            element = _perspective_to_element(payload, hit=hit, rank=rank)
            if element is not None:
                elements.append(element)
        return elements
    perspective = dict(summary) if isinstance(summary, Mapping) else {}
    identity = dict(hit.identity_summary or {})
    if not perspective.get("text"):
        for key in ("text", "lexical_text", "proposition_text"):
            value = perspective.get(key) or identity.get(key)
            if isinstance(value, str) and value.strip():
                perspective["text"] = value.strip()
                break
    element = _perspective_to_element(perspective, hit=hit, rank=rank)
    return [element] if element is not None else []


def contexte_situations(
    question: str,
    *,
    dossier: str,
    k: int,
    dsn: str = BANC_PG_DSN,
) -> list[dict[str, Any]]:
    subject_id = _subject_id_for_case(dossier, dsn=dsn)
    snapshot_id = _latest_constructeur_snapshot_id(
        subject_id, dsn=dsn, schema=MRFS_SCHEMA
    )
    backend = PostgresProjectionSearchBackend(
        _connect_factory(dsn),
        subject_id=subject_id,
        schema=MRFS_SCHEMA,
        definition_id=CONSTRUCTEUR_PROJECTION_DEFINITION_ID,
        definition_version=CONSTRUCTEUR_PROJECTION_DEFINITION_VERSION,
        rebuild_projection=lambda observed: observed,
        snapshot_id=snapshot_id,
    )
    request = SearchRequest(
        subject_id=subject_id,
        query_text=question,
        evidence_kinds=(
            SearchEvidenceKind.SITUATION_HIT,
            SearchEvidenceKind.PROPOSITION_HIT,
            SearchEvidenceKind.ASSERTION_HIT,
        ),
        filters=(),
        retrieval_modes=(SearchMode.LEXICAL,),
        expand=SearchExpansion(),
        pagination=SearchPagination(limit=max(k, 1) * 4),
        ranking_profile=_ranking_profile(),
        actor_context=SearchActorContext(
            actor_id=str(BANC_ACTOR_ID),
            permissions=(SearchPermission.READ,),
        ),
    )
    leg = backend.search(request, SearchMode.LEXICAL)
    # Un même fait est indexé sous sa proposition, son assertion et sa situation :
    # le contexte ne le rend qu'une fois (ADR 0025 §11, répétition exacte retirée).
    elements: list[dict[str, Any]] = []
    seen: set[tuple[str, Any, Any, str]] = set()
    for rank, hit in enumerate(leg.hits, start=1):
        for element in _hit_to_elements(hit, rank=rank):
            provenance = element["provenance"]
            key = (
                element["source"],
                provenance.get("line_start"),
                provenance.get("line_end"),
                element["text"],
            )
            if key in seen:
                continue
            seen.add(key)
            elements.append(element)
            if len(elements) >= k:
                return elements
    return elements
