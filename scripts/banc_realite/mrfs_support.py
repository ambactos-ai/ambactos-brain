# SPDX-License-Identifier: AGPL-3.0-only
"""MRFS banc-realite cluster helpers (private port 56102 only)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from brain.mrfs.contracts import (
        BANC_REALITE_INSTANCES,
        BANC_REALITE_PRODUCT,
        banc_realite_subject_id,
        validate_banc_realite_binding,
    )
except ImportError:
    BANC_REALITE_INSTANCES = ()  # type: ignore[misc,assignment]
    BANC_REALITE_PRODUCT = None  # type: ignore[misc,assignment]
    banc_realite_subject_id = None  # type: ignore[misc,assignment]
    validate_banc_realite_binding = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.enums import (
        SearchEvidenceKind,
        SearchMode,
        SearchPermission,
    )
except ImportError:
    SearchEvidenceKind = None  # type: ignore[misc,assignment]
    SearchMode = None  # type: ignore[misc,assignment]
    SearchPermission = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.product_adapters import require_banc_realite_reader_instance
except ImportError:
    require_banc_realite_reader_instance = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.projection_repository import PostgresProjectionSearchBackend
except ImportError:
    PostgresProjectionSearchBackend = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.refs import Checksum
except ImportError:
    Checksum = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.search import (
        RankingProfile,
        SearchActorContext,
        SearchExpansion,
        SearchHit,
        SearchPagination,
        SearchRequest,
    )
except ImportError:
    RankingProfile = None  # type: ignore[misc,assignment]
    SearchActorContext = None  # type: ignore[misc,assignment]
    SearchExpansion = None  # type: ignore[misc,assignment]
    SearchHit = None  # type: ignore[misc,assignment]
    SearchPagination = None  # type: ignore[misc,assignment]
    SearchRequest = None  # type: ignore[misc,assignment]

from .config import BANC_ACTOR_ID, BANC_ROOT, WORKTREE
from .dossiers import Dossier, dossier_courant

MRFS_SCHEMA = "mrfs_v2"
BANC_PG_DSN = "postgresql://postgres@127.0.0.1:56102/postgres"
SUBJECTS_PATH = BANC_ROOT / "dossiers" / "mrfs" / "subjects.json"
BANC_REALITE_TEST_INSTANCE = "banc_realite_test"
MEASUREMENT_INSTANCES = frozenset({"greenloop", "moreau"})
DEMO_RETIRED_PATH = BANC_ROOT / "dossiers" / "mrfs" / "demo-retired.json"
DEMO_RECORD_KEY = "demo:moreau:lumo-comportement"
DEMO_SOURCE = "07-lettre-licenciement.md"
DEMO_TEXT = (
    "Comportement inapproprié vis-à-vis du client LUMO SA le 19 décembre 2023, "
    "ayant porté atteinte à l'image de l'entreprise"
)
DEMO_QUERY = "comportement inapproprié LUMO SA 19 décembre 2023"
MAX_BUDGET_CHARS = 96000
def _ranking_profile() -> RankingProfile:
    from tests.mrfs.test_postgres_projection_repository import _ranking_profile as test_profile

    return test_profile()


@dataclass(slots=True)
class MrfsContextStats:
    subject_id: str | None
    instance_id: str
    empty_subject: bool
    hits: int
    corpus_chars: int
    rendered_chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "subject_id": self.subject_id,
            "instance_id": self.instance_id,
            "empty_subject": self.empty_subject,
            "hits": self.hits,
            "corpus_chars": self.corpus_chars,
            "rendered_chars": self.rendered_chars,
        }
        if self.empty_subject:
            payload["message"] = "sujet MRFS vide — aucun contexte versé"
        if self.corpus_chars:
            payload["corpus_pct"] = round(
                100.0 * self.rendered_chars / self.corpus_chars, 1
            )
        return payload


def _repo_root() -> Path:
    return WORKTREE


def load_subject_registry() -> dict[str, str]:
    if not SUBJECTS_PATH.is_file():
        return {}
    payload = json.loads(SUBJECTS_PATH.read_text(encoding="utf-8"))
    registry = payload.get("subjects") if isinstance(payload, dict) else payload
    if not isinstance(registry, dict):
        return {}
    return {str(key): str(value) for key, value in registry.items() if value}


def save_subject_registry(registry: dict[str, str]) -> None:
    SUBJECTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBJECTS_PATH.write_text(
        json.dumps({"subjects": registry}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _subject_has_projection(subject_id: str, *, dsn: str = BANC_PG_DSN) -> bool:
    import psycopg

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT 1 FROM {MRFS_SCHEMA}.projection_snapshots "
            "WHERE subject_id = %s LIMIT 1",
            (subject_id,),
        ).fetchone()
        return row is not None
    finally:
        connection.close()


def resolve_subject_id(dossier: Dossier) -> str | None:
    require_banc_realite_reader_instance(dossier.instance)
    registry = load_subject_registry()
    subject_id = registry.get(dossier.nom) or registry.get(dossier.instance)
    if not subject_id:
        return None
    if not _subject_has_projection(subject_id):
        return None
    return subject_id


def default_subject_id(dossier: Dossier) -> str:
    return banc_realite_subject_id(dossier.instance)


def ensure_mrfs_schema_installed(*, dsn: str = BANC_PG_DSN) -> list[dict[str, Any]]:
    script = _repo_root() / "scripts" / "install_mrfs_schema.py"
    env = {**os.environ, "PYTHONPATH": str(_repo_root() / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dsn",
            dsn,
            "--through",
            "10",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "install_mrfs_schema failed: " + (result.stderr or result.stdout)[-2000:]
        )
    return json.loads(result.stdout or "{}")


def _connect_factory(dsn: str) -> Callable[[], Any]:
    import psycopg

    def connect() -> Any:
        return psycopg.connect(dsn, autocommit=False)

    return connect


def _hit_text(hit: SearchHit) -> str:
    perspective = dict(hit.perspective_summary or {})
    identity = dict(hit.identity_summary or {})
    for key in ("text", "lexical_text", "proposition_text"):
        value = perspective.get(key) or identity.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _hit_source(hit: SearchHit) -> str:
    perspective = dict(hit.perspective_summary or {})
    identity = dict(hit.identity_summary or {})
    for key in ("source_file", "source_piece", "source"):
        value = perspective.get(key) or identity.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "source inconnue"


def _demo_retired(dossier_nom: str) -> bool:
    if not DEMO_RETIRED_PATH.is_file():
        return False
    payload = json.loads(DEMO_RETIRED_PATH.read_text(encoding="utf-8"))
    return bool((payload or {}).get(dossier_nom))


def _hit_line_range(hit: SearchHit) -> tuple[int | None, int | None]:
    perspective = dict(hit.perspective_summary or {})
    identity = dict(hit.identity_summary or {})
    line_start = perspective.get("line_start", identity.get("line_start"))
    line_end = perspective.get("line_end", identity.get("line_end"))
    start = int(line_start) if isinstance(line_start, int) else None
    end = int(line_end) if isinstance(line_end, int) else None
    return start, end


def hits_to_elements(
    hits: tuple[SearchHit, ...],
    *,
    dossier_nom: str | None = None,
) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        perspective = dict(hit.perspective_summary or {})
        if perspective.get("demo") and dossier_nom and _demo_retired(dossier_nom):
            continue
        source = _hit_source(hit)
        if source == "source inconnue":
            continue
        text = _hit_text(hit)
        if not text:
            continue
        line_start, line_end = _hit_line_range(hit)
        provenance: dict[str, Any] = {
            "source_file_path": source,
            "mrfs_object_kind": hit.kind.value,
            "mrfs_object_id": hit.object_ref.id,
        }
        if line_start is not None:
            provenance["line_start"] = line_start
        if line_end is not None:
            provenance["line_end"] = line_end
        elements.append(
            {
                "kind": "mrfs_situation",
                "rank": rank,
                "score": float(hit.score),
                "text": text,
                "source": source,
                "provenance": provenance,
            }
        )
    return elements


def _dedupe_search_hits(hits: tuple[SearchHit, ...], *, limit: int) -> tuple[SearchHit, ...]:
    """Deduplicate banc hits — ingested records often share template object_ref."""
    unique: list[SearchHit] = []
    seen: set[tuple[str, str, int | None, int | None]] = set()
    for hit in sorted(hits, key=lambda item: (-item.score, item.object_ref.id)):
        source = _hit_source(hit)
        text = _hit_text(hit)
        if not text or source == "source inconnue":
            continue
        line_start, line_end = _hit_line_range(hit)
        key = (source, text, line_start, line_end)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
        if len(unique) >= limit:
            break
    return tuple(unique)


def search_mrfs_hits(
    question_text: str,
    *,
    subject_id: str,
    k: int,
    connection=None,
    dsn: str = BANC_PG_DSN,
) -> tuple[SearchHit, ...]:
    del connection
    limit = max(k, 1)
    connect = _connect_factory(dsn)
    backend = PostgresProjectionSearchBackend(
        connect,
        subject_id=subject_id,
        schema=MRFS_SCHEMA,
        definition_id="mrfs.search.authority-index",
        definition_version="1.0.0",
        rebuild_projection=lambda observed: observed,
    )
    request = SearchRequest(
        subject_id=subject_id,
        query_text=question_text,
        evidence_kinds=(
            SearchEvidenceKind.SITUATION_HIT,
            SearchEvidenceKind.PROPOSITION_HIT,
            SearchEvidenceKind.ASSERTION_HIT,
        ),
        filters=(),
        retrieval_modes=(SearchMode.LEXICAL,),
        expand=SearchExpansion(),
        pagination=SearchPagination(limit=max(limit * 4, 8)),
        ranking_profile=_ranking_profile(),
        actor_context=SearchActorContext(
            actor_id=str(BANC_ACTOR_ID),
            permissions=(SearchPermission.READ,),
        ),
    )
    leg = backend.search(request, SearchMode.LEXICAL)
    return _dedupe_search_hits(leg.hits, limit=limit)


def corpus_char_budget(dossier: Dossier, budget_chars: int) -> int:
    from .mesure_comparatif_etage2 import BUDGET_CORPUS_PCT

    pct = float(BUDGET_CORPUS_PCT.get(dossier.nom, 0.0))
    return int(budget_chars * pct / 100.0) if pct else budget_chars


def ensure_subjects_registered(
    *,
    dsn: str = BANC_PG_DSN,
    instances: tuple[str, ...] = tuple(sorted(BANC_REALITE_INSTANCES)),
) -> dict[str, str]:
    import psycopg

    ensure_mrfs_schema_installed(dsn=dsn)
    registry = dict(load_subject_registry())
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        for instance in instances:
            validate_banc_realite_binding(
                product=BANC_REALITE_PRODUCT,
                instance_id=instance,
                scope_kind="memory_scope",
            )
            subject_id = registry.get(instance) or default_subject_id(
                dossier_courant(instance)
            )
            connection.execute(
                f"INSERT INTO {MRFS_SCHEMA}.subjects (subject_id) VALUES (%s) "
                "ON CONFLICT (subject_id) DO NOTHING",
                (subject_id,),
            )
            registry[instance] = subject_id
    finally:
        connection.close()
    save_subject_registry(registry)
    return registry


def _seed_reconstruction_for_subject(schema: str, subject_id: str):
    from tests.mrfs.test_postgres_governance import (
        _at,
        _complete_reconstruction_publication,
        _observation_with_entities,
        _payment_catalogs,
        _publish_catalogs,
    )
    from tests.mrfs.test_postgres_semantics import _ingestion_publication
    try:
        from brain.mrfs.ingestion_repository import PostgresIngestionRepository
    except ImportError:
        PostgresIngestionRepository = None  # type: ignore[misc,assignment]
    try:
        from brain.mrfs.semantic_repository import PostgresSemanticRepository
    except ImportError:
        PostgresSemanticRepository = None  # type: ignore[misc,assignment]

    try:
        from brain.mrfs.errors import MrfsError as _MrfsError
    except ImportError:
        _MrfsError = None  # type: ignore[misc,assignment]

    ingestion = _ingestion_publication(subject_id=subject_id)
    catalogs_and_refs = _payment_catalogs(_at())
    catalogs = catalogs_and_refs[0]
    try:
        _publish_catalogs(schema, catalogs)
    except _MrfsError as exc:
        if exc.code != "stale_catalog_base":
            raise
    connect = _connect_factory(os.environ.get("MRFS_TEST_DSN", BANC_PG_DSN))
    PostgresIngestionRepository(
        connect,
        subject_id=subject_id,
        schema=schema,
    ).publish(ingestion)
    observation = _observation_with_entities(ingestion, catalogs)
    PostgresSemanticRepository(
        connect,
        subject_id=subject_id,
        schema=schema,
    ).publish_observation(observation)
    reconstruction = _complete_reconstruction_publication(
        ingestion,
        observation,
        catalogs_and_refs,
    )
    PostgresSemanticRepository(
        connect,
        subject_id=subject_id,
        schema=schema,
    ).publish_reconstruction(reconstruction)
    return reconstruction




def ensure_projection_snapshot(
    subject_id: str,
    *,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> str:
    """Bootstrap S7 projection for a subject if absent, return snapshot_id."""
    import psycopg
    from dataclasses import replace

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT snapshot_id FROM {schema}.projection_snapshots "
            "WHERE subject_id = %s ORDER BY built_at DESC LIMIT 1",
            (subject_id,),
        ).fetchone()
        if row:
            return str(row[0])
    finally:
        connection.close()
    os.environ["MRFS_TEST_DSN"] = dsn
    root = Path(__file__).resolve().parents[2]
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tests.mrfs.test_postgres_projection_repository import (
        _projection_publication,
        _repository,
    )

    reconstruction = _seed_reconstruction_for_subject(schema, subject_id)
    publication = _projection_publication(reconstruction)
    _repository(schema, subject_id).publish(publication)
    return publication.output.id


def seed_moreau_demo_situation(*, dsn: str = BANC_PG_DSN, mark_demo: bool = True) -> int:
    """Publish one hand-written MOREAU proposition, then remove/mark it when done."""

    import psycopg

    ensure_mrfs_schema_installed(dsn=dsn)
    retired = {}
    if DEMO_RETIRED_PATH.is_file():
        retired = json.loads(DEMO_RETIRED_PATH.read_text(encoding="utf-8"))
    retired["moreau"] = False
    DEMO_RETIRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_RETIRED_PATH.write_text(
        json.dumps(retired, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    registry = ensure_subjects_registered(dsn=dsn)
    subject_id = registry["moreau"]
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        existing = connection.execute(
            f"SELECT 1 FROM {MRFS_SCHEMA}.projection_records "
            "WHERE subject_id = %s AND record_key = %s LIMIT 1",
            (subject_id, DEMO_RECORD_KEY),
        ).fetchone()
        if existing:
            return 1
        if _subject_has_projection(subject_id, dsn=dsn):
            snapshot_id = connection.execute(
                f"SELECT snapshot_id FROM {MRFS_SCHEMA}.projection_snapshots "
                "WHERE subject_id = %s ORDER BY built_at DESC LIMIT 1",
                (subject_id,),
            ).fetchone()[0]
            connection.execute(
                f"INSERT INTO {MRFS_SCHEMA}.projection_records ("
                "subject_id, snapshot_id, record_key, hit_kind, object_id, object_kind, "
                "object_version, payload_checksum_algorithm, payload_checksum_value, "
                "integrity_status, lexical_text, search_fields, perspective_summary, "
                "identity_summary, derivation_summary) VALUES ("
                "%s, %s, %s, 'proposition_hit', %s, 'proposition', 'demo.version.v1', "
                "'sha256', %s, 'valid', %s, %s::jsonb, %s::jsonb, %s::jsonb, NULL)",
                (
                    subject_id,
                    snapshot_id,
                    DEMO_RECORD_KEY,
                    "11111111-1111-4111-8111-111111111111",
                    "d" * 64,
                    DEMO_TEXT,
                    json.dumps(
                        {
                            "source_file": DEMO_SOURCE,
                            "content_kind": "event",
                            "temporal_role": "occurrence",
                        }
                    ),
                    json.dumps(
                        {
                            "text": DEMO_TEXT,
                            "source_file": DEMO_SOURCE,
                            "demo": mark_demo,
                            "temporal_role": "occurrence",
                            "temporal_anchor": "2023-12-19",
                        }
                    ),
                    json.dumps({"source_piece": DEMO_SOURCE, "demo": mark_demo}),
                ),
            )
            return 1
    finally:
        connection.close()
    os.environ["MRFS_TEST_DSN"] = dsn
    from dataclasses import replace

    from tests.mrfs.test_postgres_projection_repository import (
        _projection_publication,
        _repository,
    )

    reconstruction = _seed_reconstruction_for_subject(MRFS_SCHEMA, subject_id)
    publication = _projection_publication(reconstruction)
    proposition = replace(
        publication.output.records[0],
        record_key=DEMO_RECORD_KEY,
        lexical_text=DEMO_TEXT,
        perspective_summary={
            "text": DEMO_TEXT,
            "source_file": DEMO_SOURCE,
            "demo": mark_demo,
            "temporal_role": "occurrence",
            "temporal_anchor": "2023-12-19",
        },
        identity_summary={"source_piece": DEMO_SOURCE, "demo": mark_demo},
        search_fields={
            "source_file": DEMO_SOURCE,
            "content_kind": "event",
            "temporal_role": "occurrence",
        },
        factuality_summary=None,
    )
    snapshot = replace(
        publication.output,
        subject_id=subject_id,
        records=(proposition,) + publication.output.records[1:],
    )
    publication = replace(publication, output=snapshot)
    _repository(MRFS_SCHEMA, subject_id).publish(publication)
    return 1


def clear_moreau_demo_situation(*, dsn: str = BANC_PG_DSN) -> int:
    import psycopg

    registry = load_subject_registry()
    subject_id = registry.get("moreau") or default_subject_id(dossier_courant("moreau"))
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        count = connection.execute(
            f"SELECT count(*) FROM {MRFS_SCHEMA}.projection_records "
            "WHERE subject_id = %s AND record_key = %s",
            (subject_id, DEMO_RECORD_KEY),
        ).fetchone()[0]
    finally:
        connection.close()
    payload = {}
    if DEMO_RETIRED_PATH.is_file():
        payload = json.loads(DEMO_RETIRED_PATH.read_text(encoding="utf-8"))
    payload["moreau"] = True
    DEMO_RETIRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_RETIRED_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return int(count)


def require_source_file(piece: str) -> str:
    """Refuse toute écriture MRFS sans pièce source explicite."""
    source = str(piece or "").strip()
    if not source or source.lower() in {"inconnue", "unknown", "test.md"}:
        raise ValueError(f"source_file requis pour toute situation MRFS (refus : {piece!r})")
    return source


def ensure_test_subject_registered(*, dsn: str = BANC_PG_DSN) -> str:
    """Sujet hors profil de mesure — réservé aux tests hors ligne."""
    import psycopg

    ensure_mrfs_schema_installed(dsn=dsn)
    registry = dict(load_subject_registry())
    subject_id = registry.get(BANC_REALITE_TEST_INSTANCE) or str(
        uuid.uuid5(uuid.NAMESPACE_URL, "mrfs:banc-realite:test-subject")
    )
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        connection.execute(
            f"INSERT INTO {MRFS_SCHEMA}.subjects (subject_id) VALUES (%s) "
            "ON CONFLICT (subject_id) DO NOTHING",
            (subject_id,),
        )
    finally:
        connection.close()
    registry[BANC_REALITE_TEST_INSTANCE] = subject_id
    save_subject_registry(registry)
    ensure_projection_snapshot(subject_id, dsn=dsn)
    return subject_id


def list_polluted_projection_records(
    subject_id: str,
    *,
    dsn: str = BANC_PG_DSN,
) -> dict[str, list[str]]:
    import psycopg

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        snapshot_id = connection.execute(
            f"SELECT snapshot_id FROM {MRFS_SCHEMA}.projection_snapshots "
            "WHERE subject_id = %s ORDER BY built_at DESC LIMIT 1",
            (subject_id,),
        ).fetchone()[0]
        rows = connection.execute(
            f"SELECT record_key, perspective_summary->>'source_file' AS source_file "
            f"FROM {MRFS_SCHEMA}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s",
            (subject_id, snapshot_id),
        ).fetchall()
    finally:
        connection.close()
    test_md: list[str] = []
    inconnue: list[str] = []
    for record_key, source_file in rows:
        key = str(record_key)
        source = str(source_file or "").strip()
        if source == "test.md" or key.startswith("ingest:test:"):
            test_md.append(key)
        elif not source:
            inconnue.append(key)
    return {"test_md": sorted(set(test_md)), "inconnue": sorted(set(inconnue))}


def purge_polluted_projection_records(
    subject_id: str,
    *,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> dict[str, int]:
    """Retire test.md et enregistrements sans source_file via un nouveau snapshot S7."""
    from dataclasses import replace

    polluted = list_polluted_projection_records(subject_id, dsn=dsn)
    keys = set(polluted["test_md"]) | set(polluted["inconnue"])
    if not keys:
        return {"test_md": 0, "inconnue": 0, "removed": 0}
    os.environ["MRFS_TEST_DSN"] = dsn
    from .projection_buffer import _extension_publication, _projection_repository

    repository = _projection_repository(schema, subject_id, dsn=dsn)
    snapshot_id = ensure_projection_snapshot(subject_id, dsn=dsn, schema=schema)
    current = repository.get_snapshot(snapshot_id)
    kept = tuple(record for record in current.records if record.record_key not in keys)
    removed = len(current.records) - len(kept)
    if removed <= 0:
        return {
            "test_md": len(polluted["test_md"]),
            "inconnue": len(polluted["inconnue"]),
            "removed": 0,
        }
    publication = _extension_publication(current, (), subject_id=subject_id)
    publication = replace(
        publication,
        output=replace(publication.output, records=kept),
        operation="mrfs.purge.polluted-records",
    )
    repository.publish(publication)
    return {
        "test_md": len(polluted["test_md"]),
        "inconnue": len(polluted["inconnue"]),
        "removed": removed,
    }


def query_context(
    question: dict[str, Any],
    *,
    dossier: Dossier,
    k: int,
    connection=None,
) -> tuple[list[dict[str, Any]], MrfsContextStats]:
    require_banc_realite_reader_instance(dossier.instance)
    subject_id = resolve_subject_id(dossier)
    if not subject_id:
        stats = MrfsContextStats(
            subject_id=None,
            instance_id=dossier.instance,
            empty_subject=True,
            hits=0,
            corpus_chars=0,
        )
        return [], stats
    hits = search_mrfs_hits(
        str(question.get("question") or ""),
        subject_id=subject_id,
        k=k,
        connection=connection,
    )
    elements = hits_to_elements(hits, dossier_nom=dossier.nom)
    corpus_chars = sum(len(item.get("text") or "") for item in elements)
    stats = MrfsContextStats(
        subject_id=subject_id,
        instance_id=dossier.instance,
        empty_subject=not elements,
        hits=len(elements),
        corpus_chars=corpus_chars,
    )
    return elements, stats
