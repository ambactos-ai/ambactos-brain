# SPDX-License-Identifier: AGPL-3.0-only
"""Constructeur de situations MRFS depuis des fichiers de faits (ADR 0025)."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    import brain.mrfs as mrfs
except ImportError:
    mrfs = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.governance_repository import (
        GovernancePublicationKind,
        GovernanceStagePublication,
        PostgresIdentityRepository,
        PostgresTemporalRepository,
    )
except ImportError:
    GovernancePublicationKind = None  # type: ignore[misc,assignment]
    GovernanceStagePublication = None  # type: ignore[misc,assignment]
    PostgresIdentityRepository = None  # type: ignore[misc,assignment]
    PostgresTemporalRepository = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.ingestion_repository import (
        PostgresIngestionRepository,
        RepresentationStagePublication,
    )
except ImportError:
    PostgresIngestionRepository = None  # type: ignore[misc,assignment]
    RepresentationStagePublication = None  # type: ignore[misc,assignment]
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
    from brain.mrfs.semantic_repository import (
        ObservationStagePublication,
        PostgresSemanticRepository,
        ReconstructionStagePublication,
    )
except ImportError:
    ObservationStagePublication = None  # type: ignore[misc,assignment]
    PostgresSemanticRepository = None  # type: ignore[misc,assignment]
    ReconstructionStagePublication = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.selectors import make_text_position, make_text_quote
except ImportError:
    make_text_position = None  # type: ignore[misc,assignment]
    make_text_quote = None  # type: ignore[misc,assignment]

from .catalogue_banc import (
    catalog_refs_for_nature,
    frame_ref,
    install_catalogue_banc,
    nature_to_proposition_kind,
    nature_to_situation_kind,
)
from .mrfs_support import BANC_PG_DSN, MRFS_SCHEMA, require_source_file

CONSTRUCTEUR_ID_SEED = "v4"
_ID_NAMESPACE = uuid.UUID("e8f3c2d1-5a4b-6e7f-9c0d-1b2a3c4d5e6f")
SITUATION_VERSION = "constructeur/1.0.0"
WORKFLOW_ID = "banc-realite.constructeur-situations"
WORKFLOW_VERSION = "1.0.0"
RAPPROCHEMENT_PROFILE_ID = "identity.banc.constructeur.rapprochement"
RAPPROCHEMENT_PROFILE_VERSION = "1.0.0"
RAPPROCHEMENT_SIGNAL = "rapprochement_propose_par_modele"
RAPPROCHEMENT_CONFLICT = "rapprochement_non_retenu"
CONSTRUCTEUR_PROJECTION_DEFINITION_ID = "banc.constructeur.situations-index"
CONSTRUCTEUR_PROJECTION_DEFINITION_VERSION = "1.1.0"
CONSTRUCTEUR_RANKING_PROFILE_ID = "banc.constructeur.situations-ranking"
CONSTRUCTEUR_RANKING_PROFILE_VERSION = "1.0.0"
ALLOWED_LECTEUR_DOSSIERS = frozenset({"greenloop", "moreau", "banc_realite_test"})
VALID_NATURES = frozenset(
    {"evenement", "processus", "etat", "relation", "regle", "definition"}
)


@dataclass(frozen=True, slots=True)
class ValidatedFact:
    dossier: str
    piece: str
    morceau: int
    fact_id: str
    citation: str
    ligne_debut: int
    ligne_fin: int
    nature: str
    enonciateur: str
    date: Mapping[str, str | None] | None
    descripteur: str
    group_key: str | None = None


@dataclass(slots=True)
class Rejection:
    fact_id: str
    piece: str
    reason: str


@dataclass(slots=True)
class PublishedFact:
    validated: ValidatedFact
    proposition: mrfs.Proposition
    assertion: mrfs.Assertion
    situation: mrfs.Situation | None
    situation_link: mrfs.SituationLink | None
    candidate: mrfs.ReconstructionCandidate
    version_key: str = ""


@dataclass(slots=True)
class ConstructeurState:
    dossier: str
    subject_id: str
    dsn: str
    schema: str
    catalog: mrfs.CatalogBundle | None
    rejections: list[Rejection] = field(default_factory=list)
    published: list[PublishedFact] = field(default_factory=list)
    projection_records: list[mrfs.ProjectionRecord] = field(default_factory=list)
    piece_ingestions: dict[str, RepresentationStagePublication] = field(
        default_factory=dict
    )
    situation_ids: set[str] = field(default_factory=set)
    journal: list[dict[str, Any]] = field(default_factory=list)
    snapshot_id: str = ""


class PieceVide(RuntimeError):
    """Une pièce publiée à zéro proposition ou zéro situation (ADR 0025 §6)."""

    def __init__(self, piece: str, counts: Mapping[str, int]) -> None:
        super().__init__(f"pièce publiée à zéro : {piece} {dict(counts)}")
        self.piece = piece
        self.counts = dict(counts)


class _RefHolder:
    """Porte la référence d'un objet déjà publié, pour le chemin de reprise."""

    __slots__ = ("_ref",)

    def __init__(self, ref: mrfs.ObjectRef) -> None:
        self._ref = ref

    def as_ref(self) -> mrfs.ObjectRef:
        return self._ref


def _uid(*parts: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, "\0".join([CONSTRUCTEUR_ID_SEED, *parts])))


def _checksum(text: str) -> mrfs.Checksum:
    return mrfs.Checksum("sha256", hashlib.sha256(text.encode("utf-8")).hexdigest())


def _at(second: int = 0) -> datetime:
    return datetime(2026, 9, 27, 14, 0, second, tzinfo=timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def latest_constructeur_snapshot_id(
    subject_id: str,
    *,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> str | None:
    """Instantané constructeur courant : built_at puis nombre d'enregistrements."""
    import psycopg

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT ps.snapshot_id FROM {schema}.projection_snapshots ps "
            "WHERE ps.subject_id = %s AND ps.definition_id = %s "
            "AND ps.definition_version = %s "
            "ORDER BY ps.built_at DESC, "
            "(SELECT count(*) FROM "
            f"{schema}.projection_records pr "
            "WHERE pr.subject_id = ps.subject_id AND pr.snapshot_id = ps.snapshot_id "
            "AND pr.record_key LIKE 'constructeur:%%') DESC, "
            "ps.snapshot_id DESC LIMIT 1",
            (
                subject_id,
                CONSTRUCTEUR_PROJECTION_DEFINITION_ID,
                CONSTRUCTEUR_PROJECTION_DEFINITION_VERSION,
            ),
        ).fetchone()
    finally:
        connection.close()
    return str(row[0]) if row else None


def _connect_factory(dsn: str) -> Callable[[], Any]:
    import psycopg

    def connect() -> Any:
        return psycopg.connect(dsn, autocommit=False)

    return connect


def _ranking_profile() -> mrfs.RankingProfile:
    payload = json.dumps(
        {
            "profile_id": CONSTRUCTEUR_RANKING_PROFILE_ID,
            "version": CONSTRUCTEUR_RANKING_PROFILE_VERSION,
            "mode_weights": {
                mrfs.SearchMode.LEXICAL.value: 0.6,
                mrfs.SearchMode.STRUCTURED.value: 0.8,
                mrfs.SearchMode.IDENTIFIER.value: 1.0,
                mrfs.SearchMode.TEMPORAL.value: 0.7,
            },
            "max_per_kind": 20,
        },
        sort_keys=True,
    )
    return mrfs.RankingProfile(
        profile_id=CONSTRUCTEUR_RANKING_PROFILE_ID,
        version=CONSTRUCTEUR_RANKING_PROFILE_VERSION,
        checksum=_checksum(payload),
        mode_weights={
            mrfs.SearchMode.LEXICAL: 0.6,
            mrfs.SearchMode.STRUCTURED: 0.8,
            mrfs.SearchMode.IDENTIFIER: 1.0,
            mrfs.SearchMode.TEMPORAL: 0.7,
        },
        max_per_kind=20,
    )


def _constructeur_projection_definition() -> mrfs.ProjectionDefinition:
    record_kinds = (
        mrfs.SearchEvidenceKind.PROPOSITION_HIT,
        mrfs.SearchEvidenceKind.ASSERTION_HIT,
        mrfs.SearchEvidenceKind.SITUATION_HIT,
    )
    schema_payload = json.dumps(
        {
            "definition_id": CONSTRUCTEUR_PROJECTION_DEFINITION_ID,
            "version": CONSTRUCTEUR_PROJECTION_DEFINITION_VERSION,
            "record_kinds": [item.value for item in record_kinds],
        },
        sort_keys=True,
    )
    procedure = mrfs.ProcedureRef(
        str(uuid.uuid5(_ID_NAMESPACE, "banc.constructeur.projector")),
        "banc-constructeur-projector",
        "1.0.0",
        "rule",
    )
    return mrfs.ProjectionDefinition(
        definition_id=CONSTRUCTEUR_PROJECTION_DEFINITION_ID,
        version=CONSTRUCTEUR_PROJECTION_DEFINITION_VERSION,
        projection_kind=mrfs.ProjectionKind.SEARCH_INDEX,
        owner="banc-realite.constructeur",
        procedure_ref=procedure,
        schema_checksum=_checksum(schema_payload),
        authority_kinds=(
            mrfs.ObjectKind.PROPOSITION,
            mrfs.ObjectKind.ASSERTION,
            mrfs.ObjectKind.EVIDENCE_BUNDLE,
            mrfs.ObjectKind.SITUATION,
        ),
        record_kinds=record_kinds,
    )


def _projection_rebuild_authorizer(schema: str, subject_id: str, dsn: str):
    import psycopg
    try:
        from brain.mrfs.projection_repository import ProjectionRebuildAuthorizationRequest
    except ImportError:
        ProjectionRebuildAuthorizationRequest = None  # type: ignore[misc,assignment]

    rebuilder_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"constructeur-rebuilder:{subject_id}"))

    def authorize(request: ProjectionRebuildAuthorizationRequest) -> None:
        if request.subject_id != subject_id:
            raise AssertionError("subject mismatch")
        connection = psycopg.connect(dsn, autocommit=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT set_config('mrfs.subject_id', %s, true)", (subject_id,))
                cursor.execute(
                    f'SELECT registration_activity_id FROM "{schema}".trusted_projection_rebuilders '
                    "WHERE subject_id = %s AND rebuilder_id = %s "
                    "AND definition_id = %s AND definition_version = %s LIMIT 1",
                    (
                        subject_id,
                        rebuilder_id,
                        request.definition_id,
                        request.definition_version,
                    ),
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
                        f"constructeur:rebuild:{request.snapshot_id}:{request.snapshot_version}",
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


def load_faits_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON mal formé : {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON mal formé : racine attendue objet")
    return payload


def load_groupes_payload(raw: str) -> dict[str, list[list[str]]]:
    payload = load_faits_payload(raw)
    groupes = payload.get("groupes")
    if groupes is None:
        return {}
    if not isinstance(groupes, list):
        raise ValueError("groupes doit être une liste")
    return {"groupes": groupes}


def fact_ref(piece: str, fact_id: str) -> str:
    return f"{piece}#{fact_id}"


def parse_group_map(groupes: list[list[str]] | None) -> dict[str, str]:
    """Référence `piece#id` → clé de groupe, stable quel que soit l'ordre du fichier."""
    mapping: dict[str, str] = {}
    for group in groupes or []:
        if not isinstance(group, list):
            continue
        refs = sorted({str(item) for item in group})
        if len(refs) < 2:
            continue
        digest = hashlib.sha256("\n".join(refs).encode("utf-8")).hexdigest()[:16]
        key = f"groupe:{digest}"
        for ref in refs:
            if ref in mapping:
                raise ValueError(f"référence présente dans deux groupes : {ref}")
            mapping[ref] = key
    return mapping


def extract_piece_lines(piece_text: str) -> list[str]:
    return piece_text.splitlines()


def slice_lines(lines: list[str], start: int, end: int) -> str:
    if start < 1 or end < start:
        raise ValueError("bornes de lignes invalides")
    if end > len(lines):
        raise ValueError("ligne_fin hors pièce")
    return "\n".join(lines[start - 1 : end])


def citation_span_in_piece(
    piece_text: str, citation: str, line_start: int, line_end: int
) -> tuple[int, int]:
    lines = extract_piece_lines(piece_text)
    window = slice_lines(lines, line_start, line_end)
    offset = window.find(citation)
    if offset < 0:
        raise ValueError("citation absente des lignes annoncées")
    prefix = "\n".join(lines[: line_start - 1])
    if prefix:
        prefix += "\n"
    start = len(prefix) + offset
    return start, start + len(citation)


def validate_fact(
    payload: Mapping[str, Any],
    *,
    piece: str,
    piece_text: str,
    group_map: Mapping[str, str],
) -> ValidatedFact | Rejection:
    fact_id = str(payload.get("id") or "").strip()
    if not fact_id:
        return Rejection("", piece, "id manquant")
    citation = str(payload.get("citation") or "")
    if not citation:
        return Rejection(fact_id, piece, "citation vide")
    nature = str(payload.get("nature") or "").strip().lower()
    if nature not in VALID_NATURES:
        return Rejection(fact_id, piece, f"nature inconnue : {nature!r}")
    try:
        line_start = int(payload.get("ligne_debut"))
        line_end = int(payload.get("ligne_fin"))
    except (TypeError, ValueError):
        return Rejection(fact_id, piece, "lignes invalides")
    lines = extract_piece_lines(piece_text)
    if line_end > len(lines):
        return Rejection(fact_id, piece, "citation hors des lignes annoncées")
    try:
        slice_lines(lines, line_start, line_end)
    except ValueError:
        return Rejection(fact_id, piece, "bornes de lignes invalides")
    window = slice_lines(lines, line_start, line_end)
    if citation not in window:
        if citation not in piece_text:
            return Rejection(fact_id, piece, "citation absente de la pièce")
        return Rejection(fact_id, piece, "citation hors des lignes annoncées")
    enonciateur = str(payload.get("enonciateur") or "").strip()
    if not enonciateur:
        return Rejection(fact_id, piece, "énonciateur manquant")
    descripteur = str(payload.get("descripteur") or "").strip()
    if not descripteur:
        return Rejection(fact_id, piece, "descripteur manquant")
    date_payload = payload.get("date")
    date: Mapping[str, str | None] | None
    if date_payload is None:
        date = None
    elif isinstance(date_payload, dict):
        # Une valeur absente reste absente : `str(None)` fabriquerait la date « None ».
        date = {
            str(k): (None if v is None else str(v)) for k, v in date_payload.items()
        }
        if not any(date.values()):
            date = None
    else:
        return Rejection(fact_id, piece, "date invalide")
    dossier = str(payload.get("dossier") or "").strip()
    try:
        morceau = int(payload.get("morceau", 1))
    except (TypeError, ValueError):
        morceau = 1
    ref = fact_ref(piece, fact_id)
    return ValidatedFact(
        dossier=dossier,
        piece=piece,
        morceau=morceau,
        fact_id=fact_id,
        citation=citation,
        ligne_debut=line_start,
        ligne_fin=line_end,
        nature=nature,
        enonciateur=enonciateur,
        date=date,
        descripteur=descripteur,
        group_key=group_map.get(ref),
    )


def option_key(fact: ValidatedFact) -> str:
    """Clé d'option stable : une lettre d'abord, puis une empreinte de la référence.

    Un nom de pièce réel (`01-contrat-travail-cdi.md`) n'est pas un jeton admis par le
    domaine ; la référence lisible reste portée par le candidat et par l'index.
    """
    digest = hashlib.sha256(fact_ref(fact.piece, fact.fact_id).encode("utf-8")).hexdigest()
    return f"fait_{digest[:24]}"


def build_hypothesis_set(
    *,
    subject_id: str,
    facts: tuple[ValidatedFact, ...],
    candidates: tuple[mrfs.ReconstructionCandidate, ...],
    evidence_ref: mrfs.ObjectRef,
    activity_ref: mrfs.ObjectRef,
    scope_ref: mrfs.ObjectRef,
    hypothesis_id: str | None = None,
) -> mrfs.HypothesisSet:
    if len({fact.citation for fact in facts}) < 2:
        raise ValueError("HypothesisSet exige des citations différentes")
    members = tuple(
        mrfs.HypothesisOption(
            key=option_key(fact),
            kind=mrfs.HypothesisOptionKind.CANDIDATE,
            candidate_ref=candidate.as_ref(),
            evidence_for_refs=(evidence_ref,),
        )
        for fact, candidate in zip(facts, candidates, strict=True)
    )
    return mrfs.HypothesisSet(
        id=hypothesis_id or _uid(subject_id, "hypothesis", facts[0].group_key or ""),
        subject_id=subject_id,
        question="Quelle formulation conserve-t-on pour cette situation ?",
        scope_ref=scope_ref,
        members=members,
        created_by_activity_ref=activity_ref,
        created_at=_at(5),
    )


def _build_piece_ingestion(
    state: ConstructeurState,
    *,
    piece: str,
    piece_text: str,
    pieces_dir: Path,
) -> RepresentationStagePublication:
    if piece in state.piece_ingestions:
        return state.piece_ingestions[piece]
    require_source_file(piece)
    subject_id = state.subject_id
    checksum = mrfs.Checksum(
        "sha256", hashlib.sha256(piece_text.encode("utf-8")).hexdigest()
    )
    activity_id = _uid(state.subject_id, state.dossier, "ingest", piece)
    activity_ref = mrfs.ObjectRef(activity_id, subject_id, mrfs.ObjectKind.PROVENANCE_ACTIVITY)
    source_id = _uid(state.subject_id, state.dossier, "source", piece)
    representation_id = _uid(state.subject_id, state.dossier, "representation", piece)
    source = mrfs.Source(
        id=source_id,
        subject_id=subject_id,
        source_type=mrfs.SourceType.DOCUMENT,
        access_ref=str((pieces_dir / piece).resolve()),
        authority="owner",
        access_policy="subject_scoped",
        created_by_activity_ref=activity_ref,
        metadata={"constructeur_piece": piece},
    )
    representation = mrfs.SourceRepresentation(
        id=representation_id,
        subject_id=subject_id,
        source_ref=source.as_ref(),
        media_type=mrfs.MediaType.TEXT,
        format="text/markdown",
        version="constructeur/1",
        checksum=checksum,
        storage_ref=str((pieces_dir / piece).resolve()),
        kind=mrfs.RepresentationKind.ORIGINAL,
        produced_by_activity_ref=activity_ref,
        normalization_profile="nfc-lf",
        metadata={"constructeur_piece": piece},
    )
    quote_id = _uid(state.subject_id, state.dossier, piece, "quote-bootstrap")
    quote = make_text_quote(
        id=quote_id,
        subject_id=subject_id,
        representation_id=representation.id,
        exact=piece_text[: min(len(piece_text), 80)] or ".",
        language="fr",
    )
    run_id = _uid(state.subject_id, state.dossier, "run-ingest", piece)
    activity = mrfs.ProvenanceActivity(
        id=activity_id,
        subject_id=subject_id,
        activity_type=mrfs.ActivityType.INGEST,
        actor_id=_uid("actor", state.dossier),
        procedure_refs=(
            mrfs.ProcedureRef(_uid("proc-ingest"), WORKFLOW_ID, WORKFLOW_VERSION, "skill"),
        ),
        config_hash=checksum,
        run_id=run_id,
        stage="constructeur-ingest",
        started_at=_at(1),
        ended_at=_at(2),
        technical_state=mrfs.ActivityTechnicalState.SUCCEEDED,
        uses=(),
        generations=(
            mrfs.ActivityGeneration(activity_id, source.as_ref(), 0),
            mrfs.ActivityGeneration(activity_id, representation.as_ref(), 1),
            mrfs.ActivityGeneration(activity_id, quote.as_ref(), 2),
        ),
    )
    publication = RepresentationStagePublication(
        publication_id=_uid(state.subject_id, state.dossier, "pub-ingest", piece),
        workflow_id=WORKFLOW_ID,
        workflow_version=WORKFLOW_VERSION,
        stage_id=_uid(state.subject_id, state.dossier, "stage-ingest", piece),
        attempt_number=1,
        checkpoint_id=_uid(state.subject_id, state.dossier, "checkpoint-ingest", piece),
        checkpoint_ordinal=0,
        checkpoint_token="ingestion:committed",
        activity=activity,
        source=source,
        representations=(representation,),
        selectors=(quote,),
        published_at=_at(2),
    )
    PostgresIngestionRepository(
        _connect_factory(state.dsn),
        subject_id=subject_id,
        schema=state.schema,
    ).publish(publication)
    state.piece_ingestions[piece] = publication
    return publication


def _publish_fact_selectors(
    ingestion: RepresentationStagePublication,
    *,
    fact: ValidatedFact,
    quote: mrfs.TextQuoteSelector,
    position: mrfs.TextPositionSelector,
    state: ConstructeurState,
) -> None:
    subject_id = state.subject_id
    activity_id = _uid(subject_id, fact.dossier, "selectors", fact.piece, fact.fact_id)
    representation = ingestion.representations[0]
    activity = mrfs.ProvenanceActivity(
        id=activity_id,
        subject_id=state.subject_id,
        activity_type=mrfs.ActivityType.SEGMENT,
        actor_id=_uid("actor", fact.dossier),
        procedure_refs=(
            mrfs.ProcedureRef(_uid("proc-selectors"), WORKFLOW_ID, WORKFLOW_VERSION, "skill"),
        ),
        config_hash=_checksum(f"{fact.piece}:{fact.fact_id}:selectors"),
        run_id=_uid(subject_id, fact.dossier, "run-selectors", fact.piece, fact.fact_id),
        stage="constructeur-selectors",
        started_at=_at(2),
        ended_at=_at(3),
        technical_state=mrfs.ActivityTechnicalState.SUCCEEDED,
        uses=(mrfs.ActivityUse(activity_id, representation.as_ref(), 0),),
        generations=(
            mrfs.ActivityGeneration(activity_id, quote.as_ref(), 0),
            mrfs.ActivityGeneration(activity_id, position.as_ref(), 1),
        ),
    )
    publication = RepresentationStagePublication(
        publication_id=_uid(subject_id, fact.dossier, "pub-selectors", fact.piece, fact.fact_id),
        workflow_id=WORKFLOW_ID,
        workflow_version=WORKFLOW_VERSION,
        stage_id=_uid(subject_id, fact.dossier, "stage-selectors", fact.piece, fact.fact_id),
        attempt_number=1,
        checkpoint_id=_uid(subject_id, fact.dossier, "checkpoint-selectors", fact.piece, fact.fact_id),
        checkpoint_ordinal=1,
        checkpoint_token="text-selectors:committed",
        activity=activity,
        source=None,
        representations=(),
        selectors=(quote, position),
        published_at=_at(3),
    )
    PostgresIngestionRepository(
        _connect_factory(state.dsn),
        subject_id=state.subject_id,
        schema=state.schema,
    ).publish(publication)


def _build_observation(
    ingestion: RepresentationStagePublication,
    *,
    fact: ValidatedFact,
    quote: mrfs.TextQuoteSelector,
    position: mrfs.TextPositionSelector,
) -> ObservationStagePublication:
    subject_id = ingestion.subject_id
    activity_id = _uid(subject_id, fact.dossier, "observe", fact.piece, fact.fact_id)
    activity_ref = mrfs.ObjectRef(activity_id, subject_id, mrfs.ObjectKind.PROVENANCE_ACTIVITY)
    mention = mrfs.Mention(
        id=_uid(subject_id, fact.dossier, "mention", fact.piece, fact.fact_id),
        subject_id=subject_id,
        source_ref=ingestion.source.as_ref(),
        kind=mrfs.MentionKind.EXPLICIT,
        observation_type="constructeur_fact",
        observed_form=fact.citation,
        selector_refs=(quote.as_ref(), position.as_ref()),
        created_by_activity_ref=activity_ref,
        created_at=_at(3),
        language_or_modality="fr",
        metadata={
            "constructeur_piece": fact.piece,
            "constructeur_fact_id": fact.fact_id,
        },
    )
    evidence = mrfs.EvidenceBundle(
        id=_uid(subject_id, fact.dossier, "evidence", fact.piece, fact.fact_id),
        subject_id=subject_id,
        supported_object=mention.as_ref(),
        field_scope="observed_form",
        items=(
            mrfs.EvidenceItem(quote, mrfs.EvidenceRole.DIRECT, ordinal=0),
            mrfs.EvidenceItem(position, mrfs.EvidenceRole.DIRECT, ordinal=1),
        ),
        composed_by_activity_ref=activity_ref,
        integrity=mrfs.IntegrityStatus.VALID,
        access_policy="subject_scoped",
    )
    activity = mrfs.ProvenanceActivity(
        id=activity_id,
        subject_id=subject_id,
        activity_type=mrfs.ActivityType.EXTRACT,
        actor_id=_uid("actor", fact.dossier),
        procedure_refs=(
            mrfs.ProcedureRef(_uid("proc-observe"), WORKFLOW_ID, WORKFLOW_VERSION, "skill"),
        ),
        config_hash=_checksum(f"{fact.piece}:{fact.fact_id}:observe"),
        run_id=_uid(subject_id, fact.dossier, "run-observe", fact.piece, fact.fact_id),
        stage="constructeur-observation",
        started_at=_at(3),
        ended_at=_at(4),
        technical_state=mrfs.ActivityTechnicalState.SUCCEEDED,
        uses=(
            mrfs.ActivityUse(activity_id, ingestion.representations[0].as_ref(), 0),
            mrfs.ActivityUse(activity_id, quote.as_ref(), 1),
            mrfs.ActivityUse(activity_id, position.as_ref(), 2),
        ),
        generations=(
            mrfs.ActivityGeneration(activity_id, mention.as_ref(), 0),
            mrfs.ActivityGeneration(activity_id, evidence.as_ref(), 1),
        ),
    )
    return ObservationStagePublication(
        publication_id=_uid(subject_id, fact.dossier, "pub-observe", fact.piece, fact.fact_id),
        workflow_id=WORKFLOW_ID,
        workflow_version=WORKFLOW_VERSION,
        stage_id=_uid(subject_id, fact.dossier, "stage-observe", fact.piece, fact.fact_id),
        attempt_number=1,
        checkpoint_id=_uid(subject_id, fact.dossier, "checkpoint-observe", fact.piece, fact.fact_id),
        checkpoint_ordinal=2,
        checkpoint_token="observation:committed",
        activity=activity,
        mentions=(mention,),
        mention_versions=(mrfs.VersionedObjectRef(mention.as_ref(), "1"),),
        entities=(),
        entity_designations=(),
        evidence_bundles=(evidence,),
        published_at=_at(4),
    )


def _literal(text: str) -> mrfs.LiteralValue:
    return mrfs.LiteralValue(datatype="text", raw_value=text)


def _build_candidate(
    *,
    state: ConstructeurState,
    fact: ValidatedFact,
    ingestion: RepresentationStagePublication,
    observation: ObservationStagePublication,
    quote: mrfs.TextQuoteSelector,
    position: mrfs.TextPositionSelector,
) -> tuple[mrfs.ReconstructionCandidate, mrfs.EvidenceBundle, ...]:
    subject_id = state.subject_id
    concept_ref, enon_role, contenu_role, date_role, mapping_ref = catalog_refs_for_nature(
        fact.nature
    )
    activity_id = _uid(subject_id, fact.dossier, "reconstruct", fact.piece, fact.fact_id)
    activity_ref = mrfs.ObjectRef(activity_id, subject_id, mrfs.ObjectKind.PROVENANCE_ACTIVITY)
    proposition = mrfs.Proposition(
        id=_uid(subject_id, fact.dossier, "proposition", fact.piece, fact.fact_id),
        subject_id=subject_id,
        kind=nature_to_proposition_kind(fact.nature),
        concept_type=concept_ref,
        logical_scope="whole",
        created_by_activity_ref=activity_ref,
        created_at=_at(5),
        language="fr",
        metadata={
            "constructeur_piece": fact.piece,
            "constructeur_fact_id": fact.fact_id,
            "constructeur_dossier": fact.dossier,
        },
    )
    arg_ids = (
        _uid(subject_id, fact.dossier, "arg-enon", fact.piece, fact.fact_id),
        _uid(subject_id, fact.dossier, "arg-contenu", fact.piece, fact.fact_id),
        _uid(subject_id, fact.dossier, "arg-date", fact.piece, fact.fact_id),
    )
    evidence_ids_list = [
        _uid(subject_id, fact.dossier, "ev-enon", fact.piece, fact.fact_id),
        _uid(subject_id, fact.dossier, "ev-contenu", fact.piece, fact.fact_id),
    ]
    if fact.date:
        evidence_ids_list.append(_uid(subject_id, fact.dossier, "ev-date", fact.piece, fact.fact_id))
    evidence_ids_list.append(_uid(subject_id, fact.dossier, "ev-assertion", fact.piece, fact.fact_id))
    evidence_ids = tuple(evidence_ids_list)
    evidence_refs = tuple(
        mrfs.ObjectRef(item, subject_id, mrfs.ObjectKind.EVIDENCE_BUNDLE) for item in evidence_ids
    )
    arguments: list[mrfs.PropositionArgument] = [
        mrfs.PropositionArgument(
            id=arg_ids[0],
            subject_id=subject_id,
            proposition_ref=proposition.as_ref(),
            role=enon_role,
            filler=mrfs.ArgumentFiller(literal_value=_literal(fact.enonciateur)),
            ordinal=0,
            support_method=mrfs.ArgumentSupportMethod.DIRECT,
            evidence_bundle_refs=(evidence_refs[0],),
            created_by_activity_ref=activity_ref,
            created_at=_at(5),
        ),
        mrfs.PropositionArgument(
            id=arg_ids[1],
            subject_id=subject_id,
            proposition_ref=proposition.as_ref(),
            role=contenu_role,
            filler=mrfs.ArgumentFiller(literal_value=_literal(fact.citation)),
            ordinal=1,
            support_method=mrfs.ArgumentSupportMethod.DIRECT,
            evidence_bundle_refs=(evidence_refs[1],),
            created_by_activity_ref=activity_ref,
            created_at=_at(5),
        ),
    ]
    if fact.date:
        date_text = str(fact.date.get("texte") or fact.date.get("iso") or "")
        arguments.append(
            mrfs.PropositionArgument(
                id=arg_ids[2],
                subject_id=subject_id,
                proposition_ref=proposition.as_ref(),
                role=date_role,
                filler=mrfs.ArgumentFiller(literal_value=_literal(date_text)),
                ordinal=2,
                support_method=mrfs.ArgumentSupportMethod.DIRECT,
                evidence_bundle_refs=(evidence_refs[2],),
                created_by_activity_ref=activity_ref,
                created_at=_at(5),
            )
        )
    perspective = mrfs.SourcePerspective(
        id=_uid(subject_id, fact.dossier, "perspective", fact.piece, fact.fact_id),
        subject_id=subject_id,
        source_ref=ingestion.source.as_ref(),
        participant_ref=ingestion.source.as_ref(),
        role=mrfs.PerspectiveRole.AUTHOR,
        attribution_selector_refs=(quote.as_ref(), position.as_ref()),
        resolution_procedure=mrfs.ProcedureRef(
            _uid("proc-perspective"), WORKFLOW_ID, WORKFLOW_VERSION, "rule"
        ),
        created_by_activity_ref=activity_ref,
        created_at=_at(5),
        metadata={"enonciateur": fact.enonciateur},
    )
    profile = mrfs.AssertionEpistemicProfile(
        perspective_ref=perspective.as_ref(),
        proposition_ref=proposition.as_ref(),
        polarity=mrfs.Polarity.POSITIVE,
        commitment=mrfs.Commitment.COMMITTED,
        modal_operators=(),
        discourse_relation=mrfs.DiscourseRelation.ASSERT,
        linguistic_scope=mrfs.EpistemicScope(
            mrfs.ScopeKind.WHOLE_PROPOSITION, proposition.as_ref()
        ),
        attribution_selector_refs=(quote.as_ref(), position.as_ref()),
        evidence_bundle_refs=(evidence_refs[-1],),
        interpretation_state=mrfs.InterpretationState.SINGLE,
    )
    assertion = mrfs.Assertion(
        id=_uid(subject_id, fact.dossier, "assertion", fact.piece, fact.fact_id),
        subject_id=subject_id,
        proposition_ref=proposition.as_ref(),
        perspective_ref=perspective.as_ref(),
        profile=profile,
        mention_refs=(observation.mentions[0].as_ref(),),
        origin=mrfs.AssertionOrigin.EXPLICIT,
        created_by_activity_ref=activity_ref,
        created_at=_at(5),
        metadata={"constructeur_piece": fact.piece, "constructeur_fact_id": fact.fact_id},
    )
    supported = tuple(item.as_ref() for item in arguments) + (assertion.as_ref(),)
    bundles = tuple(
        mrfs.EvidenceBundle(
            id=evidence_ids[index],
            subject_id=subject_id,
            supported_object=supported_ref,
            field_scope="filler" if index < len(arguments) else "assertion",
            items=(
                mrfs.EvidenceItem(quote, mrfs.EvidenceRole.DIRECT, ordinal=0),
                mrfs.EvidenceItem(position, mrfs.EvidenceRole.DIRECT, ordinal=1),
            ),
            composed_by_activity_ref=activity_ref,
            integrity=mrfs.IntegrityStatus.VALID,
            access_policy="subject_scoped",
        )
        for index, (supported_ref, evidence_id) in enumerate(
            zip(supported, evidence_ids, strict=True)
        )
    )
    candidate = mrfs.ReconstructionCandidate(
        id=_uid(subject_id, fact.dossier, "candidate", fact.piece, fact.fact_id),
        key=f"{fact.piece}:{fact.fact_id}",
        subject_id=subject_id,
        lexical_frame_ref=frame_ref(),
        mapping_ref=mapping_ref,
        propositions=(proposition,),
        arguments=tuple(arguments),
        role_gaps=(),
        perspectives=(perspective,),
        assertions=(assertion,),
        created_by_activity_ref=activity_ref,
        created_at=_at(5),
    )
    return candidate, *bundles


def _publish_reconstruction(
    *,
    state: ConstructeurState,
    fact: ValidatedFact,
    ingestion: RepresentationStagePublication,
    observation: ObservationStagePublication,
    candidate: mrfs.ReconstructionCandidate,
    bundles: tuple[mrfs.EvidenceBundle, ...],
    alternative_set: mrfs.HypothesisSet | None = None,
) -> ReconstructionStagePublication:
    subject_id = state.subject_id
    procedure = mrfs.ProcedureRef(
        _uid("proc-reconstruct", fact.piece, fact.fact_id),
        WORKFLOW_ID,
        WORKFLOW_VERSION,
        "skill",
    )
    activity_ref = candidate.created_by_activity_ref
    request = mrfs.ReconstructionRequest(
        subject_id=subject_id,
        source_representation_refs=(
            mrfs.VersionedObjectRef(ingestion.representations[0].as_ref(), "constructeur/1"),
        ),
        mention_refs=(mrfs.VersionedObjectRef(observation.mentions[0].as_ref(), "1"),),
        entity_resolution_snapshot="constructeur:none",
        catalogs=state.catalog,
        procedure_refs=(procedure,),
        run_id=_uid(subject_id, fact.dossier, "run-reconstruct", fact.piece, fact.fact_id),
        reconstruction_activity_ref=activity_ref,
        evidence_bundles=bundles,
    )
    result = mrfs.reconstruct(
        request=request,
        candidates=(candidate,),
        alternative_set=alternative_set,
    )
    outputs = bundles + (candidate,) + candidate.propositions + candidate.arguments
    outputs += candidate.perspectives + candidate.assertions
    if alternative_set is not None:
        outputs += (alternative_set,)
    activity = mrfs.ProvenanceActivity(
        id=activity_ref.id,
        subject_id=subject_id,
        activity_type=mrfs.ActivityType.RECONSTRUCT,
        actor_id=_uid("actor", fact.dossier),
        procedure_refs=(procedure,),
        config_hash=_checksum(result.request_key),
        run_id=request.run_id,
        stage="constructeur-reconstruction",
        started_at=_at(5),
        ended_at=_at(6),
        technical_state=mrfs.ActivityTechnicalState.SUCCEEDED,
        uses=(
            mrfs.ActivityUse(activity_ref.id, ingestion.representations[0].as_ref(), 0),
            mrfs.ActivityUse(activity_ref.id, observation.mentions[0].as_ref(), 1),
        ),
        generations=tuple(
            mrfs.ActivityGeneration(activity_ref.id, item.as_ref(), ordinal)
            for ordinal, item in enumerate(outputs)
        ),
    )
    publication = ReconstructionStagePublication(
        publication_id=_uid(subject_id, fact.dossier, "pub-reconstruct", fact.piece, fact.fact_id),
        workflow_id=WORKFLOW_ID,
        workflow_version=WORKFLOW_VERSION,
        stage_id=_uid(subject_id, fact.dossier, "stage-reconstruct", fact.piece, fact.fact_id),
        attempt_number=1,
        checkpoint_id=_uid(subject_id, fact.dossier, "checkpoint-reconstruct", fact.piece, fact.fact_id),
        checkpoint_ordinal=2,
        checkpoint_token="reconstruction:committed",
        activity=activity,
        request=request,
        result=result,
        published_at=_at(6),
    )
    PostgresSemanticRepository(
        _connect_factory(state.dsn),
        subject_id=subject_id,
        schema=state.schema,
    ).publish_observation(observation)
    PostgresSemanticRepository(
        _connect_factory(state.dsn),
        subject_id=subject_id,
        schema=state.schema,
    ).publish_reconstruction(publication)
    return publication


def _situation_id_for_fact(subject_id: str, fact: ValidatedFact) -> str:
    """Une situation par fait ; le rapprochement ne change pas cette identité."""
    return _uid(subject_id, fact.dossier, "situation", fact.piece, fact.fact_id)


def _construction_procedure() -> mrfs.ProcedureRef:
    return mrfs.ProcedureRef(
        _uid("proc-declare-situation"),
        WORKFLOW_ID,
        WORKFLOW_VERSION,
        "procedure",
    )


def _rapprochement_identity_profile() -> mrfs.IdentityProfileSnapshot:
    policy = mrfs.ProcedureRef(
        _uid("proc-rapprochement-identity"),
        "banc-realite.constructeur-situations.rapprochement",
        RAPPROCHEMENT_PROFILE_VERSION,
        "policy",
    )
    checksum_payload = json.dumps(
        {
            "profile_id": RAPPROCHEMENT_PROFILE_ID,
            "rule": "rapprochement_propose_par_le_modele_non_evalue",
            "auto_link": False,
        },
        sort_keys=True,
    )
    return mrfs.IdentityProfileSnapshot(
        profile_id=RAPPROCHEMENT_PROFILE_ID,
        version=RAPPROCHEMENT_PROFILE_VERSION,
        target_kind=mrfs.IdentityTargetKind.SITUATION,
        domain="constructeur-banc",
        product_scope="banc-realite",
        positive_evidence_sets=((RAPPROCHEMENT_SIGNAL,),),
        contextual_signal_codes=(),
        weak_signal_codes=(),
        hard_conflict_codes=(RAPPROCHEMENT_CONFLICT,),
        policy_ref=policy,
        checksum=_checksum(checksum_payload),
        owner="banc-realite.constructeur",
        auto_link_authorization=None,
    )


def _proposition_exists(state: ConstructeurState, proposition_id: str) -> bool:
    import psycopg

    connection = psycopg.connect(state.dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT 1 FROM {state.schema}.propositions "
            "WHERE subject_id = %s AND proposition_id = %s LIMIT 1",
            (state.subject_id, proposition_id),
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def _find_cached_published(
    state: ConstructeurState, fact: ValidatedFact
) -> PublishedFact | None:
    for published in state.published:
        if (
            published.validated.piece == fact.piece
            and published.validated.fact_id == fact.fact_id
        ):
            return published
    return None


def _declared_creation_done(
    state: ConstructeurState, proposition_id: str, situation_id: str
) -> bool:
    import psycopg

    connection = psycopg.connect(state.dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT 1 FROM {state.schema}.situation_links "
            "WHERE subject_id = %s AND proposition_id = %s AND situation_id = %s "
            "AND decision_basis_kind = %s LIMIT 1",
            (
                state.subject_id,
                proposition_id,
                situation_id,
                mrfs.IdentityDecisionBasisKind.DECLARED_BY_SOURCE.value,
            ),
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def _situation_identity_hypothesis_exists(
    state: ConstructeurState, hypothesis_id: str
) -> bool:
    import psycopg

    connection = psycopg.connect(state.dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT 1 FROM {state.schema}.situation_identity_hypotheses "
            "WHERE subject_id = %s AND hypothesis_id = %s LIMIT 1",
            (state.subject_id, hypothesis_id),
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def _temporal_exists(state: ConstructeurState, temporal_id: str) -> bool:
    import psycopg

    connection = psycopg.connect(state.dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT 1 FROM {state.schema}.temporal_assertions "
            "WHERE subject_id = %s AND temporal_assertion_id = %s LIMIT 1",
            (state.subject_id, temporal_id),
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def _hypothesis_exists(state: ConstructeurState, hypothesis_id: str) -> bool:
    import psycopg

    connection = psycopg.connect(state.dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT 1 FROM {state.schema}.hypothesis_sets "
            "WHERE subject_id = %s AND hypothesis_set_id = %s LIMIT 1",
            (state.subject_id, hypothesis_id),
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def _reconstruction_version_for_proposition(
    state: ConstructeurState, proposition_id: str
) -> str | None:
    import psycopg

    connection = psycopg.connect(state.dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT version FROM {state.schema}.object_versions "
            "WHERE subject_id = %s AND object_id = %s AND object_kind = %s "
            "ORDER BY recorded_at DESC LIMIT 1",
            (state.subject_id, proposition_id, mrfs.ObjectKind.PROPOSITION.value),
        ).fetchone()
    finally:
        connection.close()
    return str(row[0]) if row else None


def _situation_exists(state: ConstructeurState, situation_id: str) -> bool:
    if situation_id in state.situation_ids:
        return True
    import psycopg

    connection = psycopg.connect(state.dsn, autocommit=True)
    try:
        row = connection.execute(
            f"SELECT 1 FROM {state.schema}.situations "
            "WHERE subject_id = %s AND situation_id = %s LIMIT 1",
            (state.subject_id, situation_id),
        ).fetchone()
    finally:
        connection.close()
    if row:
        state.situation_ids.add(situation_id)
    return row is not None


def _publish_situation_bundle(
    *,
    state: ConstructeurState,
    fact: ValidatedFact,
    reconstruction: ReconstructionStagePublication | _ReconstructionVersion,
    proposition: mrfs.Proposition,
    assertion: mrfs.Assertion,
    perspective: mrfs.SourcePerspective,
    bundles: tuple[mrfs.EvidenceBundle, ...],
    with_temporal: bool = True,
) -> tuple[mrfs.Situation | None, mrfs.SituationLink | None]:
    situation_kind = nature_to_situation_kind(fact.nature)
    if situation_kind is None:
        return None, None
    subject_id = state.subject_id
    concept_ref, _, _, _, _ = catalog_refs_for_nature(fact.nature)
    situation_id = _situation_id_for_fact(subject_id, fact)
    activity_id = _uid(
        subject_id, fact.dossier, "declare-situation", fact.piece, fact.fact_id
    )
    activity_ref = mrfs.ObjectRef(
        activity_id, subject_id, mrfs.ObjectKind.PROVENANCE_ACTIVITY
    )
    evidence_ref = bundles[-1].as_ref()

    def _held_situation() -> mrfs.Situation:
        return mrfs.Situation(
            id=situation_id,
            subject_id=subject_id,
            kind=situation_kind,
            concept_type=concept_ref,
            created_by_activity_ref=activity_ref,
            created_at=_at(8),
        )

    if _declared_creation_done(state, proposition.id, situation_id):
        situation = _held_situation()
        state.situation_ids.add(situation_id)
        if with_temporal:
            _publish_temporal_if_needed(
                state=state,
                fact=fact,
                situation=situation,
                perspective=perspective,
                evidence_ref=evidence_ref,
            )
        return situation, None
    if _situation_exists(state, situation_id):
        situation = _held_situation()
        if with_temporal:
            _publish_temporal_if_needed(
                state=state,
                fact=fact,
                situation=situation,
                perspective=perspective,
                evidence_ref=evidence_ref,
            )
        return situation, None
    procedure = _construction_procedure()
    link_id = _uid(subject_id, fact.dossier, "sit-link", fact.piece, fact.fact_id)
    link_ref = mrfs.ObjectRef(link_id, subject_id, mrfs.ObjectKind.SITUATION_LINK)
    situation_ref = mrfs.ObjectRef(situation_id, subject_id, mrfs.ObjectKind.SITUATION)
    activity = mrfs.ProvenanceActivity(
        id=activity_id,
        subject_id=subject_id,
        activity_type=mrfs.ActivityType.DECLARE_SITUATION,
        actor_id=_uid("actor", fact.dossier),
        procedure_refs=(procedure,),
        config_hash=_checksum(f"{proposition.id}:{situation_id}:declare"),
        run_id=_uid(
            subject_id, fact.dossier, "run-declare-situation", fact.piece, fact.fact_id
        ),
        stage="constructeur-declare-situation",
        started_at=_at(7),
        ended_at=_at(8),
        technical_state=mrfs.ActivityTechnicalState.SUCCEEDED,
        uses=(
            mrfs.ActivityUse(activity_id, proposition.as_ref(), 0),
            mrfs.ActivityUse(activity_id, evidence_ref, 1),
        ),
        generations=(
            mrfs.ActivityGeneration(activity_id, situation_ref, 0),
            mrfs.ActivityGeneration(activity_id, link_ref, 1),
        ),
    )
    situation, link = mrfs.declare_situation_from_proposition(
        mrfs.VersionedObjectRef(proposition.as_ref(), reconstruction.idempotency_key),
        situation_kind,
        concept_ref,
        procedure,
        (evidence_ref,),
        activity,
        situation_id=situation_id,
        link_id=link_id,
        version="1.0.0",
        declared_at=_at(8),
    )
    publication = GovernanceStagePublication(
        publication_id=_uid(
            subject_id, fact.dossier, "pub-declare-situation", fact.piece, fact.fact_id
        ),
        publication_kind=GovernancePublicationKind.IDENTITY,
        workflow_id=WORKFLOW_ID,
        workflow_version=WORKFLOW_VERSION,
        stage_id=_uid(
            subject_id, fact.dossier, "stage-declare-situation", fact.piece, fact.fact_id
        ),
        attempt_number=1,
        checkpoint_id=_uid(
            fact.dossier, "checkpoint-declare-situation", fact.piece, fact.fact_id
        ),
        checkpoint_ordinal=3,
        checkpoint_token="situation:declared",
        activity=activity,
        input_versions=(
            mrfs.VersionedObjectRef(
                proposition.as_ref(), reconstruction.idempotency_key
            ),
        ),
        outputs=(situation, link),
        output_versions=(
            mrfs.VersionedObjectRef(situation.as_ref(), SITUATION_VERSION),
            mrfs.VersionedObjectRef(link.as_ref(), link.version),
        ),
        operation="declare_situation",
        policy_or_rule_ref=procedure,
        evidence_snapshot_checksum=_checksum(f"{proposition.id}:{situation.id}:declare"),
        published_at=_at(8),
    )
    PostgresIdentityRepository(
        _connect_factory(state.dsn),
        subject_id=subject_id,
        schema=state.schema,
    ).publish(publication)
    state.situation_ids.add(situation_id)
    if with_temporal:
        _publish_temporal_if_needed(
            state=state,
            fact=fact,
            situation=situation,
            perspective=perspective,
            evidence_ref=evidence_ref,
        )
    return situation, link


def iso_day(value: object) -> str:
    """`AAAA-MM-JJ` si la valeur est un jour réel du calendrier, sinon chaîne vide.

    Une date partielle ou relative (« mardi », « dernier trimestre 2023 ») se publie non
    résolue, avec son texte d'origine ; elle n'est jamais complétée.
    """
    text = str(value or "").strip()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat() if len(text) == 10 else ""
    except ValueError:
        return ""


def _publish_temporal_if_needed(
    *,
    state: ConstructeurState,
    fact: ValidatedFact,
    situation: mrfs.Situation,
    perspective: mrfs.SourcePerspective,
    evidence_ref: mrfs.ObjectRef,
) -> None:
    if not fact.date:
        return
    iso = iso_day((fact.date or {}).get("iso"))
    raw = str((fact.date or {}).get("texte") or iso or "").strip()
    if not raw:
        return
    subject_id = state.subject_id
    temporal_id = _uid(subject_id, fact.dossier, "temporal", fact.piece, fact.fact_id)
    if _temporal_exists(state, temporal_id):
        return
    extent = mrfs.TemporalInstant(iso, mrfs.TemporalGranularity.DAY) if iso else None
    temporal_activity_id = _uid(subject_id, fact.dossier, "temporal-act", fact.piece, fact.fact_id)
    temporal_activity_ref = mrfs.ObjectRef(
        temporal_activity_id, subject_id, mrfs.ObjectKind.PROVENANCE_ACTIVITY
    )
    rule = mrfs.ProcedureRef(_uid("proc-temporal"), WORKFLOW_ID, WORKFLOW_VERSION, "rule")
    temporal = mrfs.TemporalAssertion(
        id=temporal_id,
        subject_id=subject_id,
        version="1.0.0",
        target_ref=situation.as_ref(),
        temporal_role="occurrence",
        temporal_role_version="1.0.0",
        raw_value=raw,
        extent=extent,
        # Non résolue : le texte d'origine est gardé, aucune valeur n'est inventée.
        granularity=(
            mrfs.TemporalGranularity.DAY if extent else mrfs.TemporalGranularity.OTHER
        ),
        calendar_id="gregorian",
        timezone_id="Europe/Paris" if extent else "unknown",
        uncertainty=mrfs.TemporalUncertainty("exact" if extent else "unresolved"),
        origin=mrfs.TemporalAssertionOrigin.EXPRESSED,
        resolution_state=(
            mrfs.TemporalResolutionState.RESOLVED
            if extent
            else mrfs.TemporalResolutionState.UNRESOLVED
        ),
        perspective_ref=perspective.as_ref(),
        evidence_refs=(evidence_ref,),
        resolution_rule=rule if extent else None,
        dependencies=(mrfs.VersionedObjectRef(situation.as_ref(), SITUATION_VERSION),),
        created_by_activity_ref=temporal_activity_ref,
        created_at=_at(10),
    )
    temporal_activity = mrfs.ProvenanceActivity(
        id=temporal_activity_id,
        subject_id=subject_id,
        activity_type=mrfs.ActivityType.RESOLVE_TIME,
        actor_id=_uid("actor", fact.dossier),
        procedure_refs=(rule,),
        config_hash=_checksum(f"{temporal_id}:{raw}"),
        run_id=_uid(subject_id, fact.dossier, "run-temporal", fact.piece, fact.fact_id),
        stage="constructeur-temporal",
        started_at=_at(9),
        ended_at=_at(10),
        technical_state=mrfs.ActivityTechnicalState.SUCCEEDED,
        uses=(
            mrfs.ActivityUse(temporal_activity_id, situation.as_ref(), 0),
            mrfs.ActivityUse(temporal_activity_id, evidence_ref, 1),
        ),
        generations=(mrfs.ActivityGeneration(temporal_activity_id, temporal.as_ref(), 0),),
    )
    temporal_publication = GovernanceStagePublication(
        publication_id=_uid(subject_id, fact.dossier, "pub-temporal", fact.piece, fact.fact_id),
        publication_kind=GovernancePublicationKind.TEMPORAL,
        workflow_id=WORKFLOW_ID,
        workflow_version=WORKFLOW_VERSION,
        stage_id=_uid(subject_id, fact.dossier, "stage-temporal", fact.piece, fact.fact_id),
        attempt_number=1,
        checkpoint_id=_uid(subject_id, fact.dossier, "checkpoint-temporal", fact.piece, fact.fact_id),
        checkpoint_ordinal=5,
        checkpoint_token="temporal:committed",
        activity=temporal_activity,
        input_versions=(mrfs.VersionedObjectRef(situation.as_ref(), SITUATION_VERSION),),
        outputs=(temporal,),
        output_versions=(mrfs.VersionedObjectRef(temporal.as_ref(), temporal.version),),
        operation="resolve_time",
        policy_or_rule_ref=rule,
        evidence_snapshot_checksum=_checksum(f"{temporal.id}:{temporal.version}:{raw}"),
        published_at=_at(10),
    )
    PostgresTemporalRepository(
        _connect_factory(state.dsn),
        subject_id=subject_id,
        schema=state.schema,
    ).publish(temporal_publication)


def _perspective_summary(
    fact: ValidatedFact,
    *,
    alternatives: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": fact.citation,
        "source_file": fact.piece,
        "line_start": fact.ligne_debut,
        "line_end": fact.ligne_fin,
        "enonciateur": fact.enonciateur,
        "date": fact.date,
    }
    if alternatives:
        payload["alternatives"] = alternatives
    return payload


def _projection_record(
    *,
    fact: ValidatedFact,
    kind: mrfs.SearchEvidenceKind,
    object_ref: mrfs.ObjectRef,
    object_version: str,
    authority_refs: tuple[mrfs.VersionedObjectRef, ...],
    evidence_refs: tuple[mrfs.ObjectRef, ...],
    alternatives: list[dict[str, Any]] | None = None,
) -> mrfs.ProjectionRecord:
    summary = _perspective_summary(fact, alternatives=alternatives)
    lexical = fact.citation
    checksum = mrfs.Checksum("sha256", hashlib.sha256(lexical.encode("utf-8")).hexdigest())
    suffix = kind.value.replace("_hit", "")
    return mrfs.ProjectionRecord(
        record_key=f"constructeur:{fact.piece}:{fact.fact_id}:{suffix}",
        kind=kind,
        object_ref=object_ref,
        object_version=object_version,
        authority_refs=authority_refs,
        evidence_refs=evidence_refs,
        lexical_text=lexical,
        search_fields={
            "source_file": fact.piece,
            "content_kind": fact.nature,
            "line_start": fact.ligne_debut,
            "line_end": fact.ligne_fin,
        },
        perspective_summary=summary,
        identity_summary={"constructeur_piece": fact.piece, "constructeur_fact_id": fact.fact_id},
        derivation_summary=None,
        factuality_summary=None,
        payload_checksum=checksum,
    )


class _ReconstructionVersion:
    __slots__ = ("idempotency_key",)

    def __init__(self, idempotency_key: str) -> None:
        self.idempotency_key = idempotency_key


def _collect_projection_records(
    *,
    fact: ValidatedFact,
    reconstruction: ReconstructionStagePublication | _ReconstructionVersion,
    proposition: mrfs.Proposition,
    assertion: mrfs.Assertion,
    bundles: tuple[mrfs.EvidenceBundle, ...],
    alternatives: list[dict[str, Any]] | None = None,
) -> list[mrfs.ProjectionRecord]:
    if bundles:
        evidence_ref = bundles[-1].as_ref()
    else:
        evidence_ref = mrfs.ObjectRef(
            _uid(
                proposition.subject_id,
                fact.dossier,
                "ev-assertion",
                fact.piece,
                fact.fact_id,
            ),
            proposition.subject_id,
            mrfs.ObjectKind.EVIDENCE_BUNDLE,
        )
    version = reconstruction.idempotency_key
    return [
        _projection_record(
            fact=fact,
            kind=mrfs.SearchEvidenceKind.PROPOSITION_HIT,
            object_ref=proposition.as_ref(),
            object_version=version,
            authority_refs=(
                mrfs.VersionedObjectRef(proposition.as_ref(), version),
                mrfs.VersionedObjectRef(assertion.as_ref(), version),
            ),
            evidence_refs=(evidence_ref,),
            alternatives=alternatives,
        ),
        _projection_record(
            fact=fact,
            kind=mrfs.SearchEvidenceKind.ASSERTION_HIT,
            object_ref=assertion.as_ref(),
            object_version=version,
            authority_refs=(mrfs.VersionedObjectRef(assertion.as_ref(), version),),
            evidence_refs=(evidence_ref,),
            alternatives=alternatives,
        ),
    ]


def _alternative_payload(fact: ValidatedFact) -> dict[str, Any]:
    return {
        "citation": fact.citation,
        "source_file": fact.piece,
        "line_start": fact.ligne_debut,
        "line_end": fact.ligne_fin,
        "enonciateur": fact.enonciateur,
        "date": dict(fact.date) if fact.date else None,
    }


def _other_options(fact: ValidatedFact, members: list[ValidatedFact]) -> list[dict[str, Any]] | None:
    others = [
        _alternative_payload(other)
        for other in members
        if not (other.piece == fact.piece and other.fact_id == fact.fact_id)
        and other.citation != fact.citation
    ]
    return others or None


def _situation_projection_record(
    *,
    situation: mrfs.Situation,
    facts: list[ValidatedFact],
    subject_id: str,
) -> mrfs.ProjectionRecord:
    lexical = "\n".join(fact.citation for fact in facts)
    summary: dict[str, Any] = {
        "propositions": [
            _perspective_summary(fact, alternatives=_other_options(fact, facts))
            for fact in facts
        ]
    }
    evidence_refs = tuple(
        mrfs.ObjectRef(
            _uid(subject_id, fact.dossier, "ev-assertion", fact.piece, fact.fact_id),
            subject_id,
            mrfs.ObjectKind.EVIDENCE_BUNDLE,
        )
        for fact in facts
    )
    return mrfs.ProjectionRecord(
        record_key=f"constructeur:situation:{situation.id}",
        kind=mrfs.SearchEvidenceKind.SITUATION_HIT,
        object_ref=situation.as_ref(),
        object_version=SITUATION_VERSION,
        authority_refs=(mrfs.VersionedObjectRef(situation.as_ref(), SITUATION_VERSION),),
        evidence_refs=evidence_refs,
        lexical_text=lexical,
        search_fields={"situation_id": situation.id},
        perspective_summary=summary,
        identity_summary={"situation_id": situation.id},
        derivation_summary=None,
        factuality_summary=None,
        payload_checksum=_checksum(lexical),
    )


def _fact_order(published: PublishedFact) -> tuple[str, str]:
    return (published.validated.piece, published.validated.fact_id)


def group_members(
    state: ConstructeurState, group_map: Mapping[str, str]
) -> dict[str, list[PublishedFact]]:
    """Groupes d'au moins deux faits publiés, membres triés (pièce, identifiant)."""
    by_ref = {
        fact_ref(item.validated.piece, item.validated.fact_id): item
        for item in state.published
    }
    groups: dict[str, list[PublishedFact]] = {}
    for ref, key in group_map.items():
        item = by_ref.get(ref)
        if item is not None:
            groups.setdefault(key, []).append(item)
    return {
        key: sorted(members, key=_fact_order)
        for key, members in groups.items()
        if len(members) >= 2
    }


def rebuild_projection_records(
    state: ConstructeurState, groups: Mapping[str, list[PublishedFact]]
) -> None:
    """Tous les enregistrements du dossier, recalculés depuis les faits publiés."""
    group_of: dict[tuple[str, str], str] = {}
    for key, members in groups.items():
        for member in members:
            group_of[_fact_order(member)] = key
    records: list[mrfs.ProjectionRecord] = []
    indexed_groups: set[str] = set()
    for published in sorted(state.published, key=_fact_order):
        fact = published.validated
        key = group_of.get(_fact_order(published))
        members = [item.validated for item in groups[key]] if key else []
        records.extend(
            _collect_projection_records(
                fact=fact,
                reconstruction=_ReconstructionVersion(published.version_key),
                proposition=published.proposition,
                assertion=published.assertion,
                bundles=(),
                alternatives=_other_options(fact, members) if members else None,
            )
        )
        if key is None:
            if published.situation is not None:
                records.append(
                    _situation_projection_record(
                        situation=published.situation,
                        facts=[fact],
                        subject_id=state.subject_id,
                    )
                )
            continue
        if key in indexed_groups:
            continue
        indexed_groups.add(key)
        canonical = next(
            (item for item in groups[key] if item.situation is not None), None
        )
        if canonical is not None and canonical.situation is not None:
            records.append(
                _situation_projection_record(
                    situation=canonical.situation,
                    facts=members,
                    subject_id=state.subject_id,
                )
            )
    state.projection_records = records


def _publish_hypothesis_for_group(
    state: ConstructeurState,
    *,
    group_key: str,
    members: list[PublishedFact],
) -> None:
    if len({item.validated.citation for item in members}) < 2:
        return
    hypothesis_id = _uid(state.subject_id, state.dossier, "hypothesis", group_key)
    if _hypothesis_exists(state, hypothesis_id):
        return
    members = [replace(item, validated=replace(item.validated, group_key=group_key)) for item in members]
    members_sorted = sorted(
        members, key=lambda item: (item.validated.piece, item.validated.fact_id)
    )
    first = members_sorted[0]
    subject_id = state.subject_id
    dossier = first.validated.dossier or state.dossier
    evidence_ref = mrfs.ObjectRef(
        _uid(subject_id, dossier, "ev-assertion", first.validated.piece, first.validated.fact_id),
        subject_id,
        mrfs.ObjectKind.EVIDENCE_BUNDLE,
    )
    scope_ref = mrfs.ObjectRef(
        _uid(subject_id, dossier, "mention", first.validated.piece, first.validated.fact_id),
        subject_id,
        mrfs.ObjectKind.MENTION,
    )
    representation_ref = mrfs.ObjectRef(
        _uid(subject_id, dossier, "representation", first.validated.piece),
        subject_id,
        mrfs.ObjectKind.SOURCE_REPRESENTATION,
    )
    activity_id = _uid(subject_id, dossier, "hypothesis-act", group_key)
    activity_ref = mrfs.ObjectRef(activity_id, state.subject_id, mrfs.ObjectKind.PROVENANCE_ACTIVITY)
    hypothesis = build_hypothesis_set(
        subject_id=state.subject_id,
        facts=tuple(item.validated for item in members_sorted),
        candidates=tuple(item.candidate for item in members_sorted),
        evidence_ref=evidence_ref,
        activity_ref=activity_ref,
        scope_ref=scope_ref,
        hypothesis_id=hypothesis_id,
    )
    procedure = mrfs.ProcedureRef(
        _uid("proc-hypothesis", group_key), WORKFLOW_ID, WORKFLOW_VERSION, "skill"
    )
    request = mrfs.ReconstructionRequest(
        subject_id=subject_id,
        source_representation_refs=(
            mrfs.VersionedObjectRef(representation_ref, "constructeur/1"),
        ),
        mention_refs=(mrfs.VersionedObjectRef(scope_ref, "1"),),
        entity_resolution_snapshot="constructeur:hypothesis",
        catalogs=state.catalog,
        procedure_refs=(procedure,),
        run_id=_uid(state.subject_id, state.dossier, "run-hypothesis", group_key),
        reconstruction_activity_ref=activity_ref,
        evidence_bundles=(),
    )
    request_key = mrfs.compute_reconstruction_key(request)
    result = mrfs.ReconstructionResult(
        request_key=request_key,
        activity_ref=activity_ref,
        candidates=(),
        propositions=(),
        arguments=(),
        role_gaps=(),
        perspectives=(),
        assertions=(),
        epistemic_profiles=(),
        product_assessments=(),
        hypothesis_sets=(hypothesis,),
    )
    activity = mrfs.ProvenanceActivity(
        id=activity_id,
        subject_id=state.subject_id,
        activity_type=mrfs.ActivityType.RECONSTRUCT,
        actor_id=_uid("actor", state.dossier),
        procedure_refs=(procedure,),
        config_hash=_checksum(request_key),
        run_id=_uid(subject_id, dossier, "run-hypothesis", group_key),
        stage="constructeur-hypothesis",
        started_at=_at(6),
        ended_at=_at(6),
        technical_state=mrfs.ActivityTechnicalState.SUCCEEDED,
        uses=(
            mrfs.ActivityUse(activity_id, representation_ref, 0),
            mrfs.ActivityUse(activity_id, scope_ref, 1),
        ),
        generations=(mrfs.ActivityGeneration(activity_id, hypothesis.as_ref(), 0),),
    )
    publication = ReconstructionStagePublication(
        publication_id=_uid(state.subject_id, state.dossier, "pub-hypothesis", group_key),
        workflow_id=WORKFLOW_ID,
        workflow_version=WORKFLOW_VERSION,
        stage_id=_uid(state.subject_id, state.dossier, "stage-hypothesis", group_key),
        attempt_number=1,
        checkpoint_id=_uid(state.subject_id, state.dossier, "checkpoint-hypothesis", group_key),
        checkpoint_ordinal=2,
        checkpoint_token="hypothesis:committed",
        activity=activity,
        request=request,
        result=result,
        published_at=_at(6),
    )
    PostgresSemanticRepository(
        _connect_factory(state.dsn),
        subject_id=state.subject_id,
        schema=state.schema,
    ).publish_reconstruction(publication)


def _published_from_fact(
    state: ConstructeurState, fact: ValidatedFact, *, version_key: str
) -> PublishedFact:
    """Références d'un fait déjà en base, recalculées depuis le fait — jamais relues."""
    piece, fact_id = fact.piece, fact.fact_id
    subject_id = state.subject_id
    dossier = fact.dossier or state.dossier

    def ref(kind: str, object_kind: mrfs.ObjectKind) -> mrfs.ObjectRef:
        return mrfs.ObjectRef(_uid(subject_id, dossier, kind, piece, fact_id), subject_id, object_kind)

    activity_ref = ref("reconstruct", mrfs.ObjectKind.PROVENANCE_ACTIVITY)
    candidate = mrfs.ReconstructionCandidate(
        id=_uid(subject_id, dossier, "candidate", piece, fact_id),
        key=f"{piece}:{fact_id}",
        subject_id=subject_id,
        lexical_frame_ref=frame_ref(),
        mapping_ref=catalog_refs_for_nature(fact.nature)[4],
        propositions=(),
        arguments=(),
        role_gaps=(),
        perspectives=(),
        assertions=(),
        created_by_activity_ref=activity_ref,
        created_at=_at(5),
    )
    proposition = mrfs.Proposition(
        id=_uid(subject_id, dossier, "proposition", piece, fact_id),
        subject_id=subject_id,
        kind=nature_to_proposition_kind(fact.nature),
        concept_type=catalog_refs_for_nature(fact.nature)[0],
        logical_scope="whole",
        created_by_activity_ref=activity_ref,
        created_at=_at(5),
    )
    perspective_ref = ref("perspective", mrfs.ObjectKind.SOURCE_PERSPECTIVE)
    assertion = mrfs.Assertion(
        id=_uid(subject_id, dossier, "assertion", piece, fact_id),
        subject_id=subject_id,
        proposition_ref=proposition.as_ref(),
        perspective_ref=perspective_ref,
        profile=mrfs.AssertionEpistemicProfile(
            perspective_ref=perspective_ref,
            proposition_ref=proposition.as_ref(),
            polarity=mrfs.Polarity.POSITIVE,
            commitment=mrfs.Commitment.COMMITTED,
            modal_operators=(),
            discourse_relation=mrfs.DiscourseRelation.ASSERT,
            linguistic_scope=mrfs.EpistemicScope(
                mrfs.ScopeKind.WHOLE_PROPOSITION, proposition.as_ref()
            ),
            attribution_selector_refs=(
                ref("quote", mrfs.ObjectKind.SELECTOR),
                ref("position", mrfs.ObjectKind.SELECTOR),
            ),
            evidence_bundle_refs=(ref("ev-assertion", mrfs.ObjectKind.EVIDENCE_BUNDLE),),
            interpretation_state=mrfs.InterpretationState.SINGLE,
        ),
        mention_refs=(ref("mention", mrfs.ObjectKind.MENTION),),
        origin=mrfs.AssertionOrigin.EXPLICIT,
        created_by_activity_ref=activity_ref,
        created_at=_at(5),
    )
    return PublishedFact(
        validated=fact,
        proposition=proposition,
        assertion=assertion,
        situation=None,
        situation_link=None,
        candidate=candidate,
        version_key=version_key,
    )


def _publish_equivalence_hypothesis(
    state: ConstructeurState,
    *,
    group_key: str,
    first: PublishedFact,
    other: PublishedFact,
    profile: mrfs.IdentityProfileSnapshot,
) -> None:
    if first.situation is None or other.situation is None:
        return
    subject_id = state.subject_id
    dossier = first.validated.dossier or state.dossier
    hypothesis_id = _uid(
        subject_id,
        dossier,
        "sit-equiv",
        group_key,
        other.validated.piece,
        other.validated.fact_id,
    )
    if _situation_identity_hypothesis_exists(state, hypothesis_id):
        return
    evidence_ref = mrfs.ObjectRef(
        _uid(
            subject_id,
            dossier,
            "ev-assertion",
            first.validated.piece,
            first.validated.fact_id,
        ),
        subject_id,
        mrfs.ObjectKind.EVIDENCE_BUNDLE,
    )
    activity_id = _uid(
        subject_id,
        dossier,
        "sit-equiv-act",
        group_key,
        other.validated.piece,
        other.validated.fact_id,
    )
    activity_ref = mrfs.ObjectRef(
        activity_id, subject_id, mrfs.ObjectKind.PROVENANCE_ACTIVITY
    )
    first_ref = first.situation.as_ref()
    other_ref = other.situation.as_ref()
    signal = mrfs.IdentitySignal(
        RAPPROCHEMENT_SIGNAL,
        mrfs.IdentitySignalKind.POSITIVE_DISCRIMINATING,
        (evidence_ref,),
    )
    hypothesis = mrfs.SituationIdentityHypothesis(
        id=hypothesis_id,
        subject_id=subject_id,
        operation=mrfs.SituationIdentityOperation.SITUATION_EQUIVALENCE,
        candidate_refs=(first_ref, other_ref),
        input_versions=(
            mrfs.VersionedObjectRef(first_ref, SITUATION_VERSION),
            mrfs.VersionedObjectRef(other_ref, SITUATION_VERSION),
        ),
        profile_id=profile.profile_id,
        profile_version=profile.version,
        signals=(signal,),
        remain_separate=True,
        created_by_activity_ref=activity_ref,
        created_at=_at(7),
    )
    activity = mrfs.ProvenanceActivity(
        id=activity_id,
        subject_id=subject_id,
        activity_type=mrfs.ActivityType.PROPOSE_IDENTITY,
        actor_id=_uid("actor", state.dossier),
        procedure_refs=(profile.policy_ref,),
        config_hash=profile.checksum,
        run_id=_uid(
            subject_id,
            dossier,
            "run-sit-equiv",
            group_key,
            other.validated.piece,
            other.validated.fact_id,
        ),
        stage="constructeur-situation-equivalence",
        started_at=_at(7),
        ended_at=_at(8),
        technical_state=mrfs.ActivityTechnicalState.SUCCEEDED,
        uses=(
            mrfs.ActivityUse(activity_id, first_ref, 0),
            mrfs.ActivityUse(activity_id, other_ref, 1),
            mrfs.ActivityUse(activity_id, evidence_ref, 2),
        ),
        generations=(mrfs.ActivityGeneration(activity_id, hypothesis.as_ref(), 0),),
    )
    publication = GovernanceStagePublication(
        publication_id=_uid(
            subject_id,
            dossier,
            "pub-sit-equiv",
            group_key,
            other.validated.piece,
            other.validated.fact_id,
        ),
        publication_kind=GovernancePublicationKind.IDENTITY,
        workflow_id=WORKFLOW_ID,
        workflow_version=WORKFLOW_VERSION,
        stage_id=_uid(
            subject_id,
            dossier,
            "stage-sit-equiv",
            group_key,
            other.validated.piece,
            other.validated.fact_id,
        ),
        attempt_number=1,
        checkpoint_id=_uid(
            dossier,
            "checkpoint-sit-equiv",
            group_key,
            other.validated.piece,
            other.validated.fact_id,
        ),
        checkpoint_ordinal=4,
        checkpoint_token="situation-equivalence:committed",
        activity=activity,
        input_versions=(
            mrfs.VersionedObjectRef(first_ref, SITUATION_VERSION),
            mrfs.VersionedObjectRef(other_ref, SITUATION_VERSION),
        ),
        outputs=(hypothesis,),
        output_versions=(mrfs.VersionedObjectRef(hypothesis.as_ref(), "1.0.0"),),
        operation="situation_equivalence",
        policy_or_rule_ref=profile.policy_ref,
        evidence_snapshot_checksum=_checksum(f"{first_ref.id}:{other_ref.id}:equiv"),
        identity_profiles=(profile,),
        published_at=_at(8),
    )
    PostgresIdentityRepository(
        _connect_factory(state.dsn),
        subject_id=subject_id,
        schema=state.schema,
    ).publish(publication)


def publish_groupes(
    state: ConstructeurState,
    groups: Mapping[str, list[PublishedFact]],
) -> None:
    """Alternatives (S4) et hypothèses d'équivalence non appliquées (S6)."""
    profile = _rapprochement_identity_profile()
    for key, members in groups.items():
        _publish_hypothesis_for_group(state, group_key=key, members=list(members))
        with_situation = [item for item in members if item.situation is not None]
        if len(with_situation) < 2:
            continue
        canonical = with_situation[0]
        for member in with_situation[1:]:
            _publish_equivalence_hypothesis(
                state,
                group_key=key,
                first=canonical,
                other=member,
                profile=profile,
            )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _records_fingerprint(rows: list[tuple[str, str, Any]]) -> str:
    payload = json.dumps(
        [
            {"key": key, "checksum": checksum, "summary": _json_safe(summary)}
            for key, checksum, summary in sorted(rows, key=lambda row: row[0])
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_fingerprint(state: ConstructeurState) -> str:
    return _records_fingerprint(
        [
            (record.record_key, record.payload_checksum.value, record.perspective_summary)
            for record in state.projection_records
        ]
    )


def _current_snapshot(state: ConstructeurState) -> tuple[str | None, str | None, int]:
    """(identifiant, empreinte, nombre d'instantanés du constructeur) relus en base."""
    import psycopg

    snapshot_id = latest_constructeur_snapshot_id(
        state.subject_id, dsn=state.dsn, schema=state.schema
    )
    connection = psycopg.connect(state.dsn, autocommit=True)
    try:
        total = connection.execute(
            f"SELECT count(*) FROM {state.schema}.projection_snapshots "
            "WHERE subject_id = %s AND definition_id = %s AND definition_version = %s",
            (
                state.subject_id,
                CONSTRUCTEUR_PROJECTION_DEFINITION_ID,
                CONSTRUCTEUR_PROJECTION_DEFINITION_VERSION,
            ),
        ).fetchone()[0]
        if snapshot_id is None:
            return None, None, int(total)
        rows = connection.execute(
            f"SELECT record_key, payload_checksum_value, perspective_summary "
            f"FROM {state.schema}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s",
            (state.subject_id, snapshot_id),
        ).fetchall()
    finally:
        connection.close()
    fingerprint = _records_fingerprint(
        [
            (
                str(key),
                str(checksum),
                summary if isinstance(summary, (dict, list)) else json.loads(summary or "null"),
            )
            for key, checksum, summary in rows
        ]
    )
    return snapshot_id, fingerprint, int(total)


def publish_projection_snapshot(state: ConstructeurState) -> str:
    """Publie l'instantané du dossier entier ; ne republie pas un contenu identique."""
    if not state.projection_records:
        return ""
    fingerprint = _state_fingerprint(state)
    current_id, current_fingerprint, total = _current_snapshot(state)
    if current_id is not None and current_fingerprint == fingerprint:
        return current_id
    subject_id = state.subject_id
    now = _utc_now()
    snapshot_key = f"{fingerprint}:{total + 1}"
    operation = f"constructeur:{snapshot_key}"
    activity_id = _uid(state.subject_id, state.dossier, "projection", "snapshot", snapshot_key)
    snapshot_id = _uid(state.subject_id, state.dossier, "snapshot", snapshot_key)
    activity_ref = mrfs.ObjectRef(activity_id, subject_id, mrfs.ObjectKind.PROVENANCE_ACTIVITY)
    snapshot_ref = mrfs.ObjectRef(snapshot_id, subject_id, mrfs.ObjectKind.PROJECTION_SNAPSHOT)
    definition = _constructeur_projection_definition()
    procedure = definition.procedure_ref
    input_versions = tuple(
        mrfs.VersionedObjectRef(record.object_ref, record.object_version)
        for record in state.projection_records
    )
    snapshot = mrfs.ProjectionSnapshot(
        id=snapshot_id,
        subject_id=subject_id,
        definition=definition,
        input_versions=input_versions,
        records=tuple(state.projection_records),
        activity_ref=activity_ref,
        built_at=now,
        build_state=mrfs.ProjectionBuildState.COMPLETE,
        missing_components=(),
    )
    activity = mrfs.ProvenanceActivity(
        id=activity_id,
        subject_id=subject_id,
        activity_type=mrfs.ActivityType.PROJECT,
        actor_id=_uid("actor", state.dossier),
        procedure_refs=(procedure,),
        config_hash=_checksum(snapshot_id),
        run_id=_uid(state.subject_id, state.dossier, "run-projection", snapshot_key),
        stage="constructeur-projection",
        started_at=now,
        ended_at=now,
        technical_state=mrfs.ActivityTechnicalState.SUCCEEDED,
        uses=tuple(
            mrfs.ActivityUse(activity_id, versioned.ref, ordinal)
            for ordinal, versioned in enumerate(input_versions)
        ),
        generations=(mrfs.ActivityGeneration(activity_id, snapshot_ref, 0),),
    )
    publication = ProjectionStagePublication(
        publication_id=_uid(state.subject_id, state.dossier, "pub-projection", snapshot_key),
        publication_kind=ProjectionPublicationKind.PROJECTION,
        workflow_id=WORKFLOW_ID,
        workflow_version=WORKFLOW_VERSION,
        stage_id=_uid(state.subject_id, state.dossier, "stage-projection", snapshot_key),
        attempt_number=1,
        checkpoint_id=_uid(state.subject_id, state.dossier, "checkpoint-projection", snapshot_key),
        checkpoint_ordinal=6,
        checkpoint_token="projection:committed",
        activity=activity,
        input_versions=input_versions,
        output=snapshot,
        operation=operation,
        published_at=now,
        ranking_profiles=(_ranking_profile(),),
    )
    repository = PostgresProjectionRepository(
        _connect_factory(state.dsn),
        subject_id=subject_id,
        schema=state.schema,
        rebuild_projection=lambda observed: observed,
        authorize_rebuild=_projection_rebuild_authorizer(state.schema, subject_id, state.dsn),
    )
    repository.publish(publication)
    return snapshot_id


def publish_validated_fact(
    state: ConstructeurState,
    *,
    fact: ValidatedFact,
    piece_text: str,
    pieces_dir: Path,
) -> PublishedFact:
    subject_id = state.subject_id
    cached = _find_cached_published(state, fact)
    if cached is not None:
        return cached
    proposition_id = _uid(subject_id, fact.dossier, "proposition", fact.piece, fact.fact_id)
    if _proposition_exists(state, proposition_id):
        version_key = _reconstruction_version_for_proposition(state, proposition_id)
        if not version_key:
            raise RuntimeError(
                f"proposition en base sans version : {fact.piece}#{fact.fact_id}"
            )
        published = _published_from_fact(state, fact, version_key=version_key)
        evidence_ref = mrfs.ObjectRef(
            _uid(subject_id, fact.dossier, "ev-assertion", fact.piece, fact.fact_id),
            subject_id,
            mrfs.ObjectKind.EVIDENCE_BUNDLE,
        )
        # Reprise : complète ce qu'un passage interrompu aurait laissé sans situation.
        situation, link = _publish_situation_bundle(
            state=state,
            fact=fact,
            reconstruction=_ReconstructionVersion(version_key),
            proposition=published.proposition,
            assertion=published.assertion,
            perspective=_RefHolder(published.assertion.perspective_ref),
            bundles=(_RefHolder(evidence_ref),),
        )
        published.situation = situation
        published.situation_link = link
        state.published.append(published)
        return published
    ingestion = _build_piece_ingestion(
        state, piece=fact.piece, piece_text=piece_text, pieces_dir=pieces_dir
    )
    representation = ingestion.representations[0]
    start, end = citation_span_in_piece(
        piece_text, fact.citation, fact.ligne_debut, fact.ligne_fin
    )
    quote = make_text_quote(
        id=_uid(subject_id, fact.dossier, "quote", fact.piece, fact.fact_id),
        subject_id=state.subject_id,
        representation_id=representation.id,
        exact=fact.citation,
        language="fr",
        metadata={
            "constructeur_piece": fact.piece,
            "line_start": fact.ligne_debut,
            "line_end": fact.ligne_fin,
        },
    )
    position = make_text_position(
        id=_uid(subject_id, fact.dossier, "position", fact.piece, fact.fact_id),
        subject_id=state.subject_id,
        representation_id=representation.id,
        start=start,
        end=end,
        normalization_profile=representation.normalization_profile,
    )
    _publish_fact_selectors(
        ingestion, fact=fact, quote=quote, position=position, state=state
    )
    observation = _build_observation(ingestion, fact=fact, quote=quote, position=position)
    candidate, *bundles = _build_candidate(
        state=state,
        fact=fact,
        ingestion=ingestion,
        observation=observation,
        quote=quote,
        position=position,
    )
    reconstruction = _publish_reconstruction(
        state=state,
        fact=fact,
        ingestion=ingestion,
        observation=observation,
        candidate=candidate,
        bundles=tuple(bundles),
    )
    proposition = candidate.propositions[0]
    assertion = candidate.assertions[0]
    situation, link = _publish_situation_bundle(
        state=state,
        fact=fact,
        reconstruction=reconstruction,
        proposition=proposition,
        assertion=assertion,
        perspective=candidate.perspectives[0],
        bundles=tuple(bundles),
    )
    published = PublishedFact(
        validated=fact,
        proposition=proposition,
        assertion=assertion,
        situation=situation,
        situation_link=link,
        candidate=candidate,
        version_key=reconstruction.idempotency_key,
    )
    state.published.append(published)
    return published


def count_fact_objects(
    state: ConstructeurState, facts: list[tuple[str, str]]
) -> dict[str, int]:
    """Comptes relus en base pour les faits donnés (pièce, identifiant) de ce constructeur."""
    import psycopg

    empty = {"propositions": 0, "situations": 0, "liens": 0, "temporalites": 0, "alternatives": 0}
    if not facts:
        return empty
    subject_id, schema = state.subject_id, state.schema
    proposition_ids = [
        _uid(subject_id, state.dossier, "proposition", piece, fact_id) for piece, fact_id in facts
    ]
    candidate_ids = [
        _uid(subject_id, state.dossier, "candidate", piece, fact_id) for piece, fact_id in facts
    ]
    connection = psycopg.connect(state.dsn, autocommit=True)
    try:
        def scalar(sql: str, ids: list[str]) -> int:
            return int(connection.execute(sql, (subject_id, ids)).fetchone()[0])

        return {
            "propositions": scalar(
                f"SELECT count(*) FROM {schema}.propositions "
                "WHERE subject_id = %s AND proposition_id::text = ANY(%s)",
                proposition_ids,
            ),
            "situations": scalar(
                f"SELECT count(DISTINCT situation_id) FROM {schema}.situation_links "
                "WHERE subject_id = %s AND proposition_id::text = ANY(%s)",
                proposition_ids,
            ),
            "liens": scalar(
                f"SELECT count(*) FROM {schema}.situation_links "
                "WHERE subject_id = %s AND proposition_id::text = ANY(%s)",
                proposition_ids,
            ),
            "temporalites": scalar(
                f"SELECT count(DISTINCT ta.temporal_assertion_id) "
                f"FROM {schema}.temporal_assertions ta "
                f"JOIN {schema}.situation_links sl ON sl.subject_id = ta.subject_id "
                "AND sl.situation_id = ta.target_object_id "
                "WHERE ta.subject_id = %s AND sl.proposition_id::text = ANY(%s)",
                proposition_ids,
            ),
            "alternatives": scalar(
                f"SELECT count(DISTINCT hypothesis_set_id) FROM {schema}.hypothesis_options "
                "WHERE subject_id = %s AND candidate_object_id::text = ANY(%s)",
                candidate_ids,
            ),
        }
    finally:
        connection.close()


def snapshot_record_kinds(state: ConstructeurState) -> dict[str, int]:
    import psycopg

    snapshot_id = latest_constructeur_snapshot_id(
        state.subject_id, dsn=state.dsn, schema=state.schema
    )
    if snapshot_id is None:
        return {}
    connection = psycopg.connect(state.dsn, autocommit=True)
    try:
        rows = connection.execute(
            f"SELECT hit_kind, count(*), "
            "count(*) FILTER (WHERE record_key NOT LIKE 'constructeur:%%') "
            f"FROM {state.schema}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s GROUP BY hit_kind",
            (state.subject_id, snapshot_id),
        ).fetchall()
    finally:
        connection.close()
    kinds = {str(kind): int(count) for kind, count, _ in rows}
    kinds["hors_constructeur"] = sum(int(foreign) for _, _, foreign in rows)
    return kinds


def new_state(
    *, dossier: str, subject_id: str, dsn: str = BANC_PG_DSN, schema: str = MRFS_SCHEMA
) -> ConstructeurState:
    if dossier not in ALLOWED_LECTEUR_DOSSIERS:
        raise ValueError(f"dossier non autorisé : {dossier!r}")
    catalog = install_catalogue_banc(dsn=dsn, schema=schema)
    return ConstructeurState(
        dossier=dossier,
        subject_id=subject_id,
        dsn=dsn,
        schema=schema,
        catalog=catalog,
    )


def load_dossier_facts(
    dossier: str, faits_paths: list[Path]
) -> dict[str, list[dict[str, Any]]]:
    """Faits bruts par pièce, tous morceaux confondus, dans l'ordre des fichiers."""
    by_piece: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(faits_paths):
        payload = load_faits_payload(path.read_text(encoding="utf-8"))
        if str(payload.get("dossier") or "") != dossier:
            raise ValueError(f"dossier du fichier de faits incompatible : {path.name}")
        piece = require_source_file(str(payload.get("piece") or ""))
        faits = payload.get("faits")
        if not isinstance(faits, list):
            raise ValueError(f"faits doit être une liste : {path.name}")
        by_piece.setdefault(piece, []).extend(
            item if isinstance(item, dict) else {"_invalide": item} for item in faits
        )
    return by_piece


def process_dossier(
    *,
    dossier: str,
    subject_id: str,
    faits_paths: list[Path],
    pieces_dir: Path,
    groupes_path: Path | None = None,
    publier_instantane: bool = True,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> ConstructeurState:
    """Un passage = le dossier entier : publication rejouable, puis un instantané complet.

    `publier_instantane=False` sert aux passages intermédiaires de l'extraction, pièce
    après pièce : les objets sont publiés et comptés, l'instantané attend le dernier passage.

    S'arrête à la première pièce publiée à zéro proposition ou zéro situation
    (`PieceVide`), sans publier d'instantané.
    """
    group_map: dict[str, str] = {}
    if groupes_path is not None:
        payload = load_groupes_payload(groupes_path.read_text(encoding="utf-8"))
        group_map = parse_group_map(payload.get("groupes"))
    by_piece = load_dossier_facts(dossier, faits_paths)
    state = new_state(dossier=dossier, subject_id=subject_id, dsn=dsn, schema=schema)
    for piece in sorted(by_piece):
        piece_path = pieces_dir / piece
        if not piece_path.is_file():
            raise FileNotFoundError(f"pièce absente : {piece_path}")
        piece_text = piece_path.read_text(encoding="utf-8")
        validated: list[ValidatedFact] = []
        rejected = 0
        seen: set[str] = set()
        for item in by_piece[piece]:
            result: ValidatedFact | Rejection
            if "_invalide" in item:
                result = Rejection("", piece, "fait non objet")
            else:
                result = validate_fact(
                    {**item, "dossier": dossier},
                    piece=piece,
                    piece_text=piece_text,
                    group_map=group_map,
                )
                if isinstance(result, ValidatedFact) and result.fact_id in seen:
                    result = Rejection(result.fact_id, piece, "identifiant en double dans la pièce")
            if isinstance(result, Rejection):
                state.rejections.append(result)
                rejected += 1
            else:
                seen.add(result.fact_id)
                validated.append(result)
        for fact in validated:
            publish_validated_fact(
                state, fact=fact, piece_text=piece_text, pieces_dir=pieces_dir
            )
        counts = count_fact_objects(state, [(piece, fact.fact_id) for fact in validated])
        line = {
            "piece": piece,
            "faits": len(by_piece[piece]),
            "valides": len(validated),
            "rejets": rejected,
            **counts,
        }
        state.journal.append(line)
        print(json.dumps(line, ensure_ascii=False), flush=True)
        if counts["propositions"] == 0 or counts["situations"] == 0:
            raise PieceVide(piece, counts)
    groups = group_members(state, group_map)
    publish_groupes(state, groups)
    rebuild_projection_records(state, groups)
    if publier_instantane:
        state.snapshot_id = publish_projection_snapshot(state)
    return state


def controle_dossier(
    *,
    dossier: str,
    subject_id: str,
    faits_paths: list[Path],
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> dict[str, Any]:
    """Comptes relus en base, par pièce, pour les faits des fichiers donnés. Lecture seule."""
    if dossier not in ALLOWED_LECTEUR_DOSSIERS:
        raise ValueError(f"dossier non autorisé : {dossier!r}")
    state = ConstructeurState(
        dossier=dossier,
        subject_id=subject_id,
        dsn=dsn,
        schema=schema,
        catalog=None,
    )
    by_piece = load_dossier_facts(dossier, faits_paths)
    pieces = {
        piece: count_fact_objects(
            state,
            [(piece, str(item.get("id") or "")) for item in items if item.get("id")],
        )
        for piece, items in sorted(by_piece.items())
    }
    return {
        "dossier": dossier,
        "sujet": subject_id,
        "pieces": pieces,
        # Comptés sur le dossier entier : une situation partagée entre pièces compte une fois.
        "totaux": count_fact_objects(
            state,
            [
                (piece, str(item.get("id") or ""))
                for piece, items in sorted(by_piece.items())
                for item in items
                if item.get("id")
            ],
        ),
        "instantane": snapshot_record_kinds(state),
    }
