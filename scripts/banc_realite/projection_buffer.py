# SPDX-License-Identifier: AGPL-3.0-only
"""Buffer et publication S7 atomique des propositions MRFS issues des pièces."""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import brain.mrfs as mrfs
except ImportError:
    mrfs = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.enums import IntegrityStatus, ObjectKind, SearchEvidenceKind
except ImportError:
    IntegrityStatus = None  # type: ignore[misc,assignment]
    ObjectKind = None  # type: ignore[misc,assignment]
    SearchEvidenceKind = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.projection_repository import (
        PostgresProjectionRepository,
        ProjectionPublicationKind,
        ProjectionStagePublication,
    )
except ImportError:
    PostgresProjectionRepository = None  # type: ignore[misc,assignment]
    ProjectionPublicationKind = None  # type: ignore[misc,assignment]
    ProjectionStagePublication = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.refs import Checksum, ObjectRef
except ImportError:
    Checksum = None  # type: ignore[misc,assignment]
    ObjectRef = None  # type: ignore[misc,assignment]

from .config import BANC_ACTOR_ID
from .mrfs_support import BANC_PG_DSN, MRFS_SCHEMA, ensure_projection_snapshot


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_tests_importable() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def buffer_path(output_dir: Path, subject_id: str) -> Path:
    return output_dir / f"projection-buffer-{subject_id}.jsonl"


def append_buffered_record(
    output_dir: Path,
    subject_id: str,
    record: dict[str, Any],
) -> None:
    path = buffer_path(output_dir, subject_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_buffer(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _connect_transactionally(dsn: str = BANC_PG_DSN):
    import psycopg

    return psycopg.connect(dsn, autocommit=False)


def _projection_rebuild_authorizer(schema: str, subject_id: str, dsn: str = BANC_PG_DSN):
    import psycopg
    try:
        from brain.mrfs.projection_repository import ProjectionRebuildAuthorizationRequest
    except ImportError:
        ProjectionRebuildAuthorizationRequest = None  # type: ignore[misc,assignment]

    rebuilder_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"mrfs:test-rebuilder:{subject_id}"))

    def authorize(request: ProjectionRebuildAuthorizationRequest) -> None:
        if request.subject_id != subject_id:
            raise AssertionError("subject mismatch")
        connection = psycopg.connect(dsn, autocommit=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT set_config('mrfs.subject_id', %s, true)", (subject_id,))
                cursor.execute(
                    f'SELECT registration_activity_id FROM "{schema}".trusted_projection_rebuilders '
                    "WHERE subject_id = %s AND rebuilder_id = %s LIMIT 1",
                    (subject_id, rebuilder_id),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    authority_activity_id = str(existing[0])
                else:
                    cursor.execute(
                        f'SELECT activity_id FROM "{schema}".provenance_activities '
                        "WHERE subject_id = %s ORDER BY started_at, activity_id LIMIT 1",
                        (subject_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise RuntimeError(f"aucune activité pour {subject_id}")
                    authority_activity_id = str(row[0])
                    cursor.execute(
                        psycopg.sql.SQL(
                            "SELECT {}.register_projection_rebuilder("
                            "%s, %s, %s, %s, %s, %s)"
                        ).format(psycopg.sql.Identifier(schema)),
                        (
                            subject_id,
                            request.definition_id,
                            request.definition_version,
                            rebuilder_id,
                            authority_activity_id,
                            request.built_at,
                        ),
                    )
                authorization_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"mrfs:rebuild-authorization:{request.snapshot_id}:"
                        f"{request.snapshot_version}",
                    )
                )
                cursor.execute(
                    psycopg.sql.SQL(
                        "SELECT {}.authorize_projection_rebuild("
                        "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                        "%s, %s, %s, %s, %s, %s)"
                    ).format(psycopg.sql.Identifier(schema)),
                    (
                        subject_id,
                        authorization_id,
                        request.snapshot_id,
                        request.snapshot_version,
                        request.definition_id,
                        request.definition_version,
                        request.input_checksum.algorithm,
                        request.input_checksum.value,
                        request.assessment_status.value,
                        request.expected_record_checksum.algorithm,
                        request.expected_record_checksum.value,
                        request.expected_record_count,
                        request.expected_storage_checksum.algorithm,
                        request.expected_storage_checksum.value,
                        request.snapshot_activity_id,
                        rebuilder_id,
                        authority_activity_id,
                        request.built_at,
                    ),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    return authorize


def _projection_repository(schema: str, subject_id: str, dsn: str = BANC_PG_DSN):
    return PostgresProjectionRepository(
        lambda: _connect_transactionally(dsn),
        subject_id=subject_id,
        schema=schema,
        rebuild_projection=lambda observed: observed,
        authorize_rebuild=_projection_rebuild_authorizer(schema, subject_id, dsn),
    )


def _template_record(snapshot: mrfs.ProjectionSnapshot) -> mrfs.ProjectionRecord:
    for record in snapshot.records:
        if (
            record.kind is SearchEvidenceKind.PROPOSITION_HIT
            and record.object_ref.kind is ObjectKind.PROPOSITION
        ):
            return record
    raise RuntimeError(
        f"snapshot {snapshot.id} sans proposition_hit valide pour l'ingestion"
    )


def _record_from_buffer(
    payload: dict[str, Any],
    *,
    subject_id: str,
    template: mrfs.ProjectionRecord,
) -> mrfs.ProjectionRecord:
    record_key = str(payload["record_key"])
    verbatim = str(payload["verbatim"])
    checksum = Checksum("sha256", hashlib.sha256(verbatim.encode("utf-8")).hexdigest())
    return mrfs.ProjectionRecord(
        record_key=record_key,
        kind=SearchEvidenceKind.PROPOSITION_HIT,
        object_ref=template.object_ref,
        object_version=template.object_version,
        authority_refs=template.authority_refs,
        evidence_refs=template.evidence_refs,
        lexical_text=verbatim,
        search_fields=dict(payload.get("search_fields") or {}),
        perspective_summary=dict(payload.get("perspective_summary") or {}),
        identity_summary=dict(payload.get("identity_summary") or {}),
        derivation_summary=template.derivation_summary,
        factuality_summary=template.factuality_summary,
        payload_checksum=checksum,
        integrity_status=IntegrityStatus.VALID,
    )


def _ranking_profile() -> mrfs.RankingProfile:
    _ensure_tests_importable()
    from tests.mrfs.test_postgres_projection_repository import _ranking_profile as test_profile

    return test_profile()


def _extension_publication(
    current: mrfs.ProjectionSnapshot,
    new_records: tuple[mrfs.ProjectionRecord, ...],
    *,
    subject_id: str,
) -> ProjectionStagePublication:
    merged = tuple(current.records) + new_records
    now = datetime.now(timezone.utc)
    activity_id = str(uuid.uuid4())
    snapshot_id = str(uuid.uuid4())
    snapshot_ref = ObjectRef(snapshot_id, subject_id, ObjectKind.PROJECTION_SNAPSHOT)
    activity_ref = ObjectRef(activity_id, subject_id, ObjectKind.PROVENANCE_ACTIVITY)
    procedure = current.definition.procedure_ref
    activity_config = Checksum("sha256", "e" * 64)
    new_snapshot = replace(
        current,
        id=snapshot_id,
        records=merged,
        activity_ref=activity_ref,
        built_at=now,
    )
    uses = tuple(
        mrfs.ActivityUse(activity_id, item.ref, ordinal)
        for ordinal, item in enumerate(current.input_versions)
    )
    operation = f"rebuild:{len(merged)}:{hashlib.sha256(','.join(r.record_key for r in new_records).encode()).hexdigest()[:16]}"
    activity = mrfs.ProvenanceActivity(
        id=activity_id,
        subject_id=subject_id,
        activity_type=mrfs.ActivityType.PROJECT,
        actor_id=str(BANC_ACTOR_ID),
        procedure_refs=(procedure,),
        config_hash=activity_config,
        run_id=str(uuid.uuid4()),
        stage="mrfs-search-projection",
        started_at=now,
        ended_at=now,
        technical_state=mrfs.ActivityTechnicalState.SUCCEEDED,
        uses=uses,
        generations=(mrfs.ActivityGeneration(activity_id, snapshot_ref, 0),),
        errors=(),
    )
    return ProjectionStagePublication(
        publication_id=str(uuid.uuid4()),
        publication_kind=ProjectionPublicationKind.PROJECTION,
        workflow_id="mrfs.projection",
        workflow_version="1.0.0",
        stage_id=str(uuid.uuid4()),
        attempt_number=1,
        checkpoint_id=str(uuid.uuid4()),
        checkpoint_ordinal=0,
        checkpoint_token="projection:committed",
        activity=activity,
        input_versions=current.input_versions,
        output=new_snapshot,
        operation=operation,
        published_at=now,
        ranking_profiles=(_ranking_profile(),),
    )


def _ensure_ingestion_proposition_template(
    subject_id: str,
    *,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> None:
    import os
    import psycopg
    from dataclasses import replace

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT 1 FROM {schema}.projection_records "
            "WHERE subject_id = %s AND hit_kind = 'proposition_hit' "
            "AND snapshot_id = ("
            f"SELECT snapshot_id FROM {schema}.projection_snapshots "
            "WHERE subject_id = %s ORDER BY built_at DESC LIMIT 1)",
            (subject_id, subject_id),
        ).fetchone()
    finally:
        connection.close()
    if row is not None:
        return
    os.environ["MRFS_TEST_DSN"] = dsn
    from tests.mrfs.test_postgres_projection_repository import _projection_publication

    from .mrfs_support import _seed_reconstruction_for_subject

    reconstruction = _seed_reconstruction_for_subject(schema, subject_id)
    seed_publication = _projection_publication(reconstruction)
    proposition = next(
        record
        for record in seed_publication.output.records
        if record.kind.value == "proposition_hit"
    )
    repository = _projection_repository(schema, subject_id, dsn=dsn)
    snapshot_id = ensure_projection_snapshot(subject_id, dsn=dsn, schema=schema)
    current = repository.get_snapshot(snapshot_id)
    if any(item.record_key == proposition.record_key for item in current.records):
        return
    publication = _extension_publication(current, (proposition,), subject_id=subject_id)
    publication = replace(publication, operation="mrfs.ingest.ensure-template")
    repository.publish(publication)


def flush_projection_buffer(
    subject_id: str,
    *,
    output_dir: Path,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> int:
    """Publie atomiquement les propositions bufferisées pour un sujet."""
    _ensure_ingestion_proposition_template(subject_id, dsn=dsn, schema=schema)
    path = buffer_path(output_dir, subject_id)
    buffered = _load_buffer(path)
    if not buffered:
        return 0
    deduped: dict[str, dict[str, Any]] = {}
    for payload in buffered:
        deduped[str(payload["record_key"])] = payload
    snapshot_id = ensure_projection_snapshot(subject_id, dsn=dsn, schema=schema)
    repository = _projection_repository(schema, subject_id, dsn=dsn)
    current = repository.get_snapshot(snapshot_id)
    template = _template_record(current)
    existing_keys = {record.record_key for record in current.records}
    new_records = tuple(
        _record_from_buffer(payload, subject_id=subject_id, template=template)
        for record_key, payload in deduped.items()
        if record_key not in existing_keys
    )
    if not new_records:
        path.unlink(missing_ok=True)
        return 0
    publication = _extension_publication(current, new_records, subject_id=subject_id)
    repository.publish(publication)
    path.unlink(missing_ok=True)
    return len(new_records)
