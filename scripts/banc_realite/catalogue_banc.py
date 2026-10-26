# SPDX-License-Identifier: AGPL-3.0-only
"""Catalogue minimal actif pour le constructeur de situations (ADR 0025 §7)."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

try:
    import brain.mrfs as mrfs
except ImportError:
    mrfs = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.repository import PostgresCatalogRepository
except ImportError:
    PostgresCatalogRepository = None  # type: ignore[misc,assignment]

from .mrfs_support import BANC_PG_DSN, MRFS_SCHEMA

CATALOG_ID = "catalog.banc-realite.constructeur"
CATALOG_VERSION = "1.0.0"
NAMESPACE = "banc-realite.constructeur"
_ID_NAMESPACE = uuid.UUID("c4e8a1f2-9b3d-4e7a-8f6c-1d5e0a9b2c3d")

NATURES = (
    "evenement",
    "processus",
    "etat",
    "relation",
    "regle",
    "definition",
)

ROLE_ENONCIATEUR = "role.enonciateur"
ROLE_CONTENU = "role.contenu"
ROLE_DATE = "role.date"
FRAME_REF_ID = "frame.enonce-banc"


def _uid(*parts: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, "\0".join(parts)))


def _at(second: int = 0) -> datetime:
    return datetime(2026, 9, 27, 12, 0, second, tzinfo=timezone.utc)


def _header(
    ref: mrfs.CatalogEntryRef,
    *,
    label: str,
    definition: str,
    at: datetime,
) -> mrfs.CatalogEntryHeader:
    return mrfs.CatalogEntryHeader(
        ref=ref,
        namespace=NAMESPACE,
        status=mrfs.CatalogStatus.ACTIVE,
        label=label,
        definition=definition,
        owner="banc-realite.constructeur",
        changelog=("Installation catalogue constructeur.",),
        effective_at=at,
    )


def _role_ref(role_id: str) -> mrfs.CatalogEntryRef:
    return mrfs.CatalogEntryRef(
        catalog_id=CATALOG_ID,
        entry_id=role_id,
        version=CATALOG_VERSION,
        kind=mrfs.CatalogEntryKind.ROLE_DEFINITION,
    )


def _concept_ref(nature: str) -> mrfs.CatalogEntryRef:
    return mrfs.CatalogEntryRef(
        catalog_id=CATALOG_ID,
        entry_id=f"concept.{nature}",
        version=CATALOG_VERSION,
        kind=mrfs.CatalogEntryKind.CONCEPT_TYPE,
    )


def _mapping_ref(nature: str) -> mrfs.CatalogEntryRef:
    return mrfs.CatalogEntryRef(
        catalog_id=CATALOG_ID,
        entry_id=f"mapping.{nature}",
        version=CATALOG_VERSION,
        kind=mrfs.CatalogEntryKind.FRAME_CONCEPT_MAPPING,
    )


def nature_to_proposition_kind(nature: str) -> mrfs.PropositionKind:
    mapping = {
        "evenement": mrfs.PropositionKind.EVENT,
        "processus": mrfs.PropositionKind.PROCESS,
        "etat": mrfs.PropositionKind.STATE,
        "relation": mrfs.PropositionKind.RELATION,
        "regle": mrfs.PropositionKind.RULE,
        "definition": mrfs.PropositionKind.DEFINITION,
    }
    kind = mapping.get(nature)
    if kind is None:
        raise ValueError(f"nature inconnue : {nature!r}")
    return kind


def nature_to_situation_kind(nature: str) -> mrfs.SituationKind | None:
    mapping = {
        "evenement": mrfs.SituationKind.EVENT,
        "processus": mrfs.SituationKind.PROCESS,
        "etat": mrfs.SituationKind.STATE,
        "relation": mrfs.SituationKind.RELATION,
    }
    return mapping.get(nature)


def build_catalog_bundle(at: datetime | None = None) -> mrfs.CatalogBundle:
    at = at or _at()
    enon_ref = _role_ref(ROLE_ENONCIATEUR)
    contenu_ref = _role_ref(ROLE_CONTENU)
    date_ref = _role_ref(ROLE_DATE)
    role_refs = (enon_ref, contenu_ref, date_ref)
    roles = (
        mrfs.RoleDefinition(
            header=_header(
                enon_ref,
                label="Énonciateur",
                definition="Partie qui énonce le fait.",
                at=at,
            ),
            allowed_filler_kinds=(mrfs.RoleFillerKind.LITERAL,),
            allowed_type_refs=(),
            cardinality=mrfs.RoleCardinality(1, 1),
            semantic_centrality=mrfs.SemanticCentrality.CORE,
            expected_if_expressed=True,
            null_instantiations=(),
            product_requirement_refs=(),
        ),
        mrfs.RoleDefinition(
            header=_header(
                contenu_ref,
                label="Contenu",
                definition="Contenu énoncé du fait.",
                at=at,
            ),
            allowed_filler_kinds=(mrfs.RoleFillerKind.LITERAL,),
            allowed_type_refs=(),
            cardinality=mrfs.RoleCardinality(1, 1),
            semantic_centrality=mrfs.SemanticCentrality.CORE,
            expected_if_expressed=True,
            null_instantiations=(),
            product_requirement_refs=(),
        ),
        mrfs.RoleDefinition(
            header=_header(
                date_ref,
                label="Date",
                definition="Date associée au fait quand elle est donnée.",
                at=at,
            ),
            allowed_filler_kinds=(mrfs.RoleFillerKind.LITERAL,),
            allowed_type_refs=(),
            cardinality=mrfs.RoleCardinality(0, 1),
            semantic_centrality=mrfs.SemanticCentrality.PERIPHERAL,
            expected_if_expressed=False,
            null_instantiations=("indefinite",),
            product_requirement_refs=(),
        ),
    )
    frame_ref = mrfs.CatalogEntryRef(
        catalog_id=CATALOG_ID,
        entry_id=FRAME_REF_ID,
        version=CATALOG_VERSION,
        kind=mrfs.CatalogEntryKind.LEXICAL_FRAME,
    )
    frame = mrfs.LexicalFrame(
        header=_header(
            frame_ref,
            label="Énoncé banc",
            definition="Cadre lexical générique pour les faits du constructeur.",
            at=at,
        ),
        lexical_units=("enonce",),
        language="fr",
        domain="banc-realite",
        role_refs=role_refs,
        example_selector_refs=(),
        neighbor_frame_refs=(),
    )
    concepts: list[mrfs.ConceptType] = []
    mappings: list[mrfs.FrameConceptMapping] = []
    for nature in NATURES:
        concept_ref = _concept_ref(nature)
        prop_kind = nature_to_proposition_kind(nature)
        concepts.append(
            mrfs.ConceptType(
                header=_header(
                    concept_ref,
                    label=nature.replace("_", " ").title(),
                    definition=f"Concept constructeur pour la nature {nature}.",
                    at=at,
                ),
                content_kind=prop_kind,
                role_refs=role_refs,
                specializes_refs=(),
                compatible_refs=(),
                excludes_refs=(),
            )
        )
        mappings.append(
            mrfs.FrameConceptMapping(
                header=_header(
                    _mapping_ref(nature),
                    label=f"Mapping {nature}",
                    definition=f"Correspondance cadre → concept {nature}.",
                    at=at,
                ),
                lexical_frame_ref=frame_ref,
                concept_type_ref=concept_ref,
                role_mappings=(),
            )
        )
    checksum = hashlib.sha256(
        json.dumps([nature for nature in NATURES], sort_keys=True).encode()
    ).hexdigest()
    snapshot = mrfs.CatalogSnapshot(
        catalog_id=CATALOG_ID,
        namespace=NAMESPACE,
        version=CATALOG_VERSION,
        checksum=mrfs.Checksum("sha256", checksum),
        entries=tuple(roles) + (frame,) + tuple(concepts) + tuple(mappings),
    )
    return mrfs.CatalogBundle(snapshots=(snapshot,))


def catalog_refs_for_nature(
    nature: str,
) -> tuple[
    mrfs.CatalogEntryRef,
    mrfs.CatalogEntryRef,
    mrfs.CatalogEntryRef,
    mrfs.CatalogEntryRef,
    mrfs.CatalogEntryRef,
]:
    return (
        _concept_ref(nature),
        _role_ref(ROLE_ENONCIATEUR),
        _role_ref(ROLE_CONTENU),
        _role_ref(ROLE_DATE),
        _mapping_ref(nature),
    )


def frame_ref() -> mrfs.CatalogEntryRef:
    return mrfs.CatalogEntryRef(
        catalog_id=CATALOG_ID,
        entry_id=FRAME_REF_ID,
        version=CATALOG_VERSION,
        kind=mrfs.CatalogEntryKind.LEXICAL_FRAME,
    )


def _connect_factory(dsn: str) -> Callable[[], Any]:
    import psycopg

    def connect() -> Any:
        return psycopg.connect(dsn, autocommit=False)

    return connect


def install_catalogue_banc(
    *,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> mrfs.CatalogBundle:
    """Installation idempotente du catalogue constructeur sur le cluster banc."""
    try:
        from brain.mrfs.errors import MrfsError
    except ImportError:
        MrfsError = None  # type: ignore[misc,assignment]

    bundle = build_catalog_bundle()
    repository = PostgresCatalogRepository(_connect_factory(dsn), schema=schema)
    for snapshot in bundle.snapshots:
        publication = mrfs.CatalogPublication(
            publication_id=_uid("catalog-pub", snapshot.catalog_id),
            snapshot=snapshot,
            base_version=None,
            ordered_change_set=(f"install:{snapshot.catalog_id}",),
            editorial_decision_id=_uid("editorial", snapshot.catalog_id),
            activity_id=_uid("activity", snapshot.catalog_id),
            published_at=_at(3),
        )
        try:
            repository.publish(publication)
        except MrfsError as exc:
            if exc.code != "stale_catalog_base":
                raise
    return bundle
