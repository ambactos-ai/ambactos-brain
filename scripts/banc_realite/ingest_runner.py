# SPDX-License-Identifier: AGPL-3.0-only
"""Ingest dossier markdown into the banc Brain (GreenLoop par défaut)."""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL = "grok-4.5"

try:
    from brain.adapters.supabase_claims import (
        AtomicClaim,
        build_revision_envelope,
        compute_natural_key,
        write_claims_versioned,
    )
except ImportError:
    AtomicClaim = None  # type: ignore[misc,assignment]
    build_revision_envelope = None  # type: ignore[misc,assignment]
    compute_natural_key = None  # type: ignore[misc,assignment]
    write_claims_versioned = None  # type: ignore[misc,assignment]
try:
    from brain.claiming import CoarseToFineParams, extract_claims_coarse_to_fine
except ImportError:
    CoarseToFineParams = None  # type: ignore[misc,assignment]
    extract_claims_coarse_to_fine = None  # type: ignore[misc,assignment]
try:
    from brain.claiming.tabular import TABULAR_SKILL, tabular_enabled_from_env
except ImportError:
    TABULAR_SKILL = None  # type: ignore[misc,assignment]
    tabular_enabled_from_env = None  # type: ignore[misc,assignment]

from .dossiers import BANC_VAR, Dossier, dossier_courant
from .infra import BancDatabase, cluster_ready, connect_existing, reset_brain_schema, wait_postgrest_ready
from .config import (
    BANC_ROOT,
    INGESTION2_MAX_CALLS,
    INGESTION2_SORTIES,
    INGESTION3_MAX_CALLS,
    INGESTION3_REPRISE_LOG,
    INGESTION3_SORTIES,
    MOREAU_INGESTION_MAX_CALLS,
    MOREAU_INGESTION_MAX_EMBEDDINGS,
    MOREAU_INGESTION_SORTIES,
    SORTIES,
)
from .relay import CALL_LOG, ModelCallCapError, RelaySkillRunner, load_call_count


@dataclass(frozen=True, slots=True)
class IngestionContext:
    dossier: Dossier

    @property
    def corpus(self) -> Path:
        return self.dossier.corpus_dir

    @property
    def area(self) -> Path:
        return self.dossier.area_root

    @property
    def pages(self) -> Path:
        return self.area / "semantic-memory" / "pages"

    @property
    def vault_ref(self) -> str:
        return self.dossier.vault_ref

    @classmethod
    def for_name(cls, name: str) -> "IngestionContext":
        return cls(dossier=dossier_courant(name))


_DEFAULT_CTX = IngestionContext.for_name("greenloop")


@dataclass(frozen=True)
class IngestionConfig:
    output_dir: Path = SORTIES
    max_calls: int = 160
    include_prior_report: bool = True
    journal_path: Path | None = None
    dossier_name: str = "greenloop"
    max_embeddings: int | None = None
    call_log_name: str = "skill-calls.jsonl"

    @property
    def ctx(self) -> IngestionContext:
        return IngestionContext.for_name(self.dossier_name)

    @property
    def ingest_log(self) -> Path:
        if self.journal_path is not None:
            return self.journal_path
        return self.output_dir / "ingestion.jsonl"

    @property
    def call_log(self) -> Path:
        return self.output_dir / self.call_log_name

    @property
    def reprise_log(self) -> Path:
        return self.output_dir / "reprise.jsonl"

    @classmethod
    def for_ingestion2(cls) -> "IngestionConfig":
        return cls(
            output_dir=INGESTION2_SORTIES,
            max_calls=INGESTION2_MAX_CALLS,
            include_prior_report=False,
        )

    @classmethod
    def for_ingestion3(cls) -> "IngestionConfig":
        return cls(
            output_dir=INGESTION3_SORTIES,
            max_calls=INGESTION3_MAX_CALLS,
            include_prior_report=False,
        )

    @classmethod
    def for_ingestion3_reprise(cls) -> "IngestionConfig":
        return cls(
            output_dir=INGESTION3_SORTIES,
            max_calls=0,
            include_prior_report=False,
            journal_path=INGESTION3_REPRISE_LOG,
        )

    @classmethod
    def for_moreau_brain(cls) -> "IngestionConfig":
        return cls(
            output_dir=MOREAU_INGESTION_SORTIES,
            max_calls=MOREAU_INGESTION_MAX_CALLS,
            include_prior_report=False,
            dossier_name="moreau",
            max_embeddings=MOREAU_INGESTION_MAX_EMBEDDINGS,
            call_log_name="appels.jsonl",
        )


def ingestion_status_path(config: IngestionConfig | None = None) -> Path:
    cfg = config or IngestionConfig()
    return cfg.output_dir / "_ingestion-status.json"


def write_ingestion_status(summary: dict[str, Any], config: IngestionConfig | None = None) -> Path:
    path = ingestion_status_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


BANC_ROOT = (BANC_VAR)
CORPUS = (BANC_VAR / "dossiers" / "greenloop" / "documents")
AREA = BANC_ROOT / "aire"
PAGES = AREA / "semantic-memory" / "pages"
INGEST_LOG = Path(os.environ.get("INGEST_LOG", str(BANC_ROOT / "sorties" / "ingestion.jsonl")))
VAULT_REF = "greenloop"
MAX_CALLS = 160
TRIAL_SMALL = "26-weigh-ticket-quarry-tower-provisional.md"
TRIAL_LARGE = "24-customer-review.md"
TRIAL_CSV = "22-recorded-visits.csv"


def pgdata_exists() -> bool:
    return (BANC_ROOT / "pg" / "postgres" / "PG_VERSION").exists()


def list_markdown(ctx: IngestionContext | None = None) -> list[Path]:
    current = ctx or _DEFAULT_CTX
    return sorted(p for p in current.corpus.glob("*.md"))


def list_corpus(ctx: IngestionContext | None = None) -> list[Path]:
    current = ctx or _DEFAULT_CTX
    files = list_markdown(current)
    if tabular_enabled_from_env():
        files = sorted({*files, *current.corpus.glob("*.csv")})
    return files


def ensure_area(ctx: IngestionContext | None = None) -> None:
    current = ctx or _DEFAULT_CTX
    area = current.area
    area.mkdir(parents=True, exist_ok=True)
    marker = area / ".brain-area" / "area.id"
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(uuid.uuid4()) + "\n")
    pages = current.pages
    pages.mkdir(parents=True, exist_ok=True)
    for src in list_corpus(current):
        dst = pages / src.name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            dst.write_bytes(src.read_bytes())


def append_reprise_record(record: dict[str, Any], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def source_path_for(file_name: str) -> str:
    return f"semantic-memory/pages/{file_name}"


def last_ingest_record(file_name: str, *, ingest_log: Path | None = None) -> dict[str, Any] | None:
    log_path = ingest_log or INGEST_LOG
    if not log_path.is_file():
        return None
    last = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("file") == file_name:
            last = record
    return last


def claims_count_for_piece(
    connection,
    file_name: str,
    *,
    ctx: IngestionContext | None = None,
) -> int:
    current = ctx or _DEFAULT_CTX
    row = connection.execute(
        "SELECT COUNT(*) AS n FROM brain.claims "
        "WHERE lifecycle_status = 'active' AND source_file_path = %s AND vault_ref = %s",
        (source_path_for(file_name), current.vault_ref),
    ).fetchone()
    return int(row["n"])


def piece_status(
    connection,
    file_name: str,
    *,
    ingest_log: Path | None = None,
    ctx: IngestionContext | None = None,
) -> str:
    last = last_ingest_record(file_name, ingest_log=ingest_log)
    count = claims_count_for_piece(connection, file_name, ctx=ctx)
    if last and last.get("statut") == "complete" and count > 0 and not last.get("erreur"):
        return "complete"
    if last and last.get("statut") == "partielle" and count > 0:
        return "partial"
    if count > 0:
        return "partial"
    return "new"


def delete_claims_for_piece(
    connection,
    file_name: str,
    *,
    ctx: IngestionContext | None = None,
) -> None:
    current = ctx or _DEFAULT_CTX
    connection.execute(
        "DELETE FROM brain.claims WHERE source_file_path = %s AND vault_ref = %s",
        (source_path_for(file_name), current.vault_ref),
    )


def count_tabular_scopes_for_piece(
    connection,
    file_name: str,
    *,
    ctx: IngestionContext | None = None,
) -> tuple[int, int]:
    current = ctx or _DEFAULT_CTX
    rows = connection.execute(
        "SELECT COALESCE(qualifiers->>'tabular_scope', '') AS scope, COUNT(*) AS n "
        "FROM brain.claims WHERE lifecycle_status = 'active' AND source_file_path = %s "
        "AND vault_ref = %s AND extracted_by_skill = %s GROUP BY 1",
        (source_path_for(file_name), current.vault_ref, TABULAR_SKILL),
    ).fetchall()
    line = cell = 0
    for row in rows:
        if row["scope"] == "row":
            line = int(row["n"])
        elif row["scope"] == "cell":
            cell = int(row["n"])
    return line, cell


def write_claims_with_retry(
    *,
    claims: list[AtomicClaim],
    source_file_path: str,
    source_file_id: str,
    runner: RelaySkillRunner,
    result,
    db_url: str,
    db_key: str,
    skill_name: str | None = None,
) -> Any:
    wait_postgrest_ready(url=db_url, service_key=db_key)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return write_claims_versioned(
                claims,
                source_file_path=source_file_path,
                source_file_id=source_file_id,
                actor_id="banc-realite",
                skill_name=skill_name or result.prompt_version(),
                extracted_by_model=f"{runner.providers[-1] if runner.providers else 'hermes_relay'}:{MODEL}",
                revision=build_revision_envelope(
                    workflow_id="banc-realite",
                    actor_id="banc-realite",
                    prompt_version=result.prompt_version(),
                    model=MODEL,
                    rationale="banc-realite ingestion",
                ),
                supabase_url=db_url,
                supabase_service_key=db_key,
            )
        except Exception as exc:  # noqa: BLE001 — réessai PostgREST borné
            last_error = exc
            wait_postgrest_ready(url=db_url, service_key=db_key, timeout_s=30.0)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"PostgREST write failed after 3 attempts: {last_error}")


def ingest_piece(
    path: Path,
    *,
    db_url: str,
    db_key: str,
    runner: RelaySkillRunner,
    connection,
    ingest_log: Path | None = None,
    ctx: IngestionContext | None = None,
) -> dict[str, Any]:
    current = ctx or _DEFAULT_CTX
    text = path.read_text(encoding="utf-8")
    started = time.monotonic()
    calls_before = runner.calls
    record: dict[str, Any] = {
        "run_id": runner.run_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": path.name,
        "size": len(text.encode()),
        "zones": 0,
        "appels_modele": 0,
        "fournisseur": None,
        "claims_ecrits": 0,
        "claims_tabulaires": 0,
        "tables": 0,
        "lignes_tabulaires": 0,
        "lignes_illisibles": 0,
        "duree_s": 0.0,
        "statut": "erreur",
        "zones_manquantes": [],
        "erreur": None,
        "tabular_marker_claims_dropped": 0,
        "tabular_marker_claims_dropped_unanchored": 0,
        "faits_ligne": 0,
        "faits_cellule": 0,
    }
    log_path = ingest_log or INGEST_LOG
    try:
        params = CoarseToFineParams(tabular=tabular_enabled_from_env())
        result = extract_claims_coarse_to_fine(
            file_name=path.name,
            text=text,
            run_skill=runner.run_skill,
            params=params,
        )
        record["zones"] = len(result.zones)
        record["tables"] = result.tabular_tables
        record["lignes_tabulaires"] = result.tabular_rows
        record["lignes_illisibles"] = result.tabular_illisible_lines
        record["tabular_marker_claims_dropped"] = result.tabular_marker_claims_dropped
        record["tabular_marker_claims_dropped_unanchored"] = (
            result.tabular_marker_claims_dropped_unanchored
        )
        record["appels_modele"] = runner.calls - calls_before
        record["fournisseur"] = runner.primary_provider()

        failed_zones = [
            f"zone {call.zone.ordinal}"
            for call in result.fine_calls
            if call.error
        ]
        if result.coarse_run is None and result.errors and not path.name.lower().endswith(".csv"):
            failed_zones.insert(0, "coarse-map")

        tabular_rows = [r for r in result.claims if r.get("extracted_by_skill") == TABULAR_SKILL]
        prose_rows = [r for r in result.claims if r.get("extracted_by_skill") != TABULAR_SKILL]

        def to_atomic(rows: list[dict[str, Any]]) -> list[AtomicClaim]:
            out: list[AtomicClaim] = []
            for row in rows:
                if not row.get("entity_id") or not row.get("predicate") or row.get("value") in (None, ""):
                    continue
                out.append(
                    AtomicClaim(
                        entity_id=row["entity_id"],
                        predicate=row["predicate"],
                        value=row["value"],
                        qualifiers=row.get("qualifiers") or {},
                        modality=row.get("modality") or "asserted",
                        source_span=row.get("source_span"),
                        confidence=row.get("confidence") or "medium",
                        vault_ref=current.vault_ref,
                        source_kind="deterministic_extractor" if row.get("extracted_by_skill") == TABULAR_SKILL else None,
                    )
                )
            return out

        tabular_claims = to_atomic(tabular_rows)
        prose_claims = to_atomic(prose_rows)
        relative = source_path_for(path.name)
        source_file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"banc://{current.vault_ref}/{path.name}"))
        written_total = 0
        if tabular_claims:
            written = write_claims_with_retry(
                claims=tabular_claims,
                source_file_path=relative,
                source_file_id=source_file_id,
                runner=runner,
                result=result,
                db_url=db_url,
                db_key=db_key,
                skill_name=TABULAR_SKILL,
            )
            written_total += len(written.written) + len(written.unchanged)
            record["claims_tabulaires"] = written_total
        if prose_claims:
            written = write_claims_with_retry(
                claims=prose_claims,
                source_file_path=relative,
                source_file_id=source_file_id,
                runner=runner,
                result=result,
                db_url=db_url,
                db_key=db_key,
            )
            written_total += len(written.written) + len(written.unchanged)
        record["claims_ecrits"] = written_total
        line_count, cell_count = count_tabular_scopes_for_piece(connection, path.name, ctx=current)
        record["faits_ligne"] = line_count
        record["faits_cellule"] = cell_count
        claims = tabular_claims + prose_claims
        if claims and written_total != len({compute_natural_key(c) for c in claims}):
            raise RuntimeError("Writer did not account for all distinct incoming keys")
        record["zones_completees"] = [
            call.zone.ordinal for call in result.fine_calls if not call.error
        ]

        if failed_zones and claims:
            record["statut"] = "partielle"
            record["zones_manquantes"] = failed_zones
            record["erreur"] = "; ".join(result.errors) if result.errors else None
        elif failed_zones and not claims:
            record["statut"] = "erreur"
            record["zones_manquantes"] = failed_zones
            raise RuntimeError("; ".join(result.errors) if result.errors else "no claims extracted")
        elif result.errors:
            record["statut"] = "partielle" if claims else "erreur"
            record["zones_manquantes"] = failed_zones
            record["erreur"] = "; ".join(result.errors)
            if not claims:
                raise RuntimeError("; ".join(result.errors))
        else:
            record["statut"] = "complete"
    except ModelCallCapError as exc:
        record["erreur"] = str(exc)
        record["statut"] = "partielle"
        record["arret"] = "plafond_appels"
    except Exception as exc:  # noqa: BLE001 — journal d'ingestion borné
        record["erreur"] = str(exc)
        if record["statut"] != "partielle":
            record["statut"] = "erreur"
    record["appels_modele"] = runner.calls - calls_before
    record["duree_s"] = round(time.monotonic() - started, 2)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def run_ingestion(
    *,
    trial_only: bool = False,
    reset_db: bool = False,
    config: IngestionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or IngestionConfig()
    ctx = cfg.ctx
    ingest_log = cfg.ingest_log
    ensure_area(ctx)
    ingest_log.parent.mkdir(parents=True, exist_ok=True)
    db = None
    created = False
    if not cluster_ready():
        db = BancDatabase(resume=pgdata_exists())
        created = True
        db.export_env()
    connection = db.connection if db is not None else connect_existing()
    if reset_db:
        reset_brain_schema(connection)
    url = os.environ["BRAIN_POSTGREST_URL"]
    key = os.environ["BRAIN_POSTGREST_SERVICE_KEY"]
    runner = RelaySkillRunner(
        log_path=cfg.call_log,
        include_prior_report=cfg.include_prior_report,
        max_calls=cfg.max_calls if cfg.max_calls > 0 else None,
    )
    files = list_corpus(ctx)
    if trial_only:
        files = [ctx.corpus / TRIAL_SMALL, ctx.corpus / TRIAL_LARGE, ctx.corpus / TRIAL_CSV]
    summary = {
        "run_id": runner.run_id,
        "dossier": ctx.dossier.nom,
        "pieces": [],
        "appels_modele": runner.calls,
        "erreurs": [],
        "partielles": [],
        "skipped": [],
    }
    for path in files:
        if runner.calls >= cfg.max_calls:
            summary["arret"] = "plafond_appels"
            append_reprise_record(
                {
                    "run_id": runner.run_id,
                    "arret": "plafond_appels",
                    "piece": path.name,
                    "appels_modele": runner.calls,
                    "plafond": cfg.max_calls,
                },
                log_path=cfg.reprise_log,
            )
            break
        status = piece_status(connection, path.name, ingest_log=ingest_log, ctx=ctx)
        if status == "complete":
            skip = {
                "file": path.name,
                "skipped": True,
                "raison": "deja_complete",
                "claims_en_base": claims_count_for_piece(connection, path.name, ctx=ctx),
            }
            summary["pieces"].append(skip)
            summary["skipped"].append(path.name)
            with ingest_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({**skip, "run_id": runner.run_id}, ensure_ascii=False) + "\n")
            append_reprise_record(
                {**skip, "run_id": runner.run_id, "statut": "deja_complete"},
                log_path=cfg.reprise_log,
            )
            continue
        if status == "partial":
            delete_claims_for_piece(connection, path.name, ctx=ctx)
        record = ingest_piece(
            path,
            db_url=url,
            db_key=key,
            runner=runner,
            connection=connection,
            ingest_log=ingest_log,
            ctx=ctx,
        )
        summary["pieces"].append(record)
        summary["appels_modele"] = runner.calls
        append_reprise_record(
            {
                "run_id": runner.run_id,
                "piece": path.name,
                "statut": record.get("statut"),
                "zones_completees": record.get("zones_completees") or [],
                "zones_manquantes": record.get("zones_manquantes") or [],
                "appels_modele": record.get("appels_modele", 0),
            },
            log_path=cfg.reprise_log,
        )
        if record.get("arret") == "plafond_appels" or str(record.get("erreur") or "").startswith("plafond"):
            summary["arret"] = "plafond_appels"
            summary["partielles"].append(record["file"])
            break
        if record.get("statut") == "erreur":
            summary["erreurs"].append(record["file"])
            if trial_only:
                summary["arret"] = "essai_echoue"
                break
        elif record.get("statut") == "partielle":
            summary["partielles"].append(record["file"])
    runner.assert_no_claude()
    summary["fournisseur"] = runner.primary_provider()
    summary["duree_s"] = round(
        sum(
            float(piece.get("duree_s", 0))
            for piece in summary["pieces"]
            if not piece.get("skipped")
        ),
        1,
    )
    write_ingestion_status(summary, cfg)
    if db is not None and created:
        db.close()
    return summary


def delete_tabular_claims_for_piece(connection, file_name: str) -> int:
    row = connection.execute(
        "DELETE FROM brain.claims WHERE source_file_path = %s AND vault_ref = %s "
        "AND extracted_by_skill = %s RETURNING claim_id",
        (source_path_for(file_name), VAULT_REF, TABULAR_SKILL),
    ).fetchall()
    return len(row)


def tabular_count_from_journal(file_name: str) -> int:
    record = last_ingest_record(file_name)
    if record is None:
        return 0
    if record.get("claims_tabulaires") is not None:
        return int(record["claims_tabulaires"])
    if record.get("skipped"):
        return int(record.get("claims_en_base", 0))
    return 0


def count_tabular_claims(full: dict[str, Any], *, connection=None) -> int:
    if connection is not None:
        row = connection.execute(
            "SELECT COUNT(*) AS n FROM brain.claims "
            "WHERE lifecycle_status = 'active' AND vault_ref = %s "
            "AND extracted_by_skill = %s",
            (VAULT_REF, TABULAR_SKILL),
        ).fetchone()
        return int(row["n"])
    seen: set[str] = set()
    total = 0
    for piece in full.get("pieces", []):
        file_name = piece.get("file")
        if not file_name or file_name in seen:
            continue
        seen.add(file_name)
        total += tabular_count_from_journal(file_name)
    return total


def list_active_tabular_claims(connection, file_name: str) -> list[dict[str, Any]]:
    return connection.execute(
        "SELECT claim_id, entity_id, predicate, value, qualifiers, modality "
        "FROM brain.claims WHERE source_file_path = %s AND vault_ref = %s "
        "AND extracted_by_skill = %s AND lifecycle_status = 'active'",
        (source_path_for(file_name), VAULT_REF, TABULAR_SKILL),
    ).fetchall()


def delete_tabular_claims_by_ids(connection, claim_ids: list[str]) -> int:
    if not claim_ids:
        return 0
    connection.execute(
        "DELETE FROM brain.claims WHERE claim_id = ANY(%s)",
        (claim_ids,),
    )
    return len(claim_ids)


def sync_tabular_claims_for_piece(
    *,
    file_name: str,
    parsed_claims: list[dict[str, Any]],
    connection,
    write_fn,
    db_url: str | None = None,
    db_key: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "file": file_name,
        "claims_ecrits": 0,
        "claims_inchanges": 0,
        "claims_retires": 0,
        "statut": "erreur",
        "erreur": None,
    }
    claims: list[AtomicClaim] = []
    for row in parsed_claims:
        if not row.get("entity_id") or not row.get("predicate") or row.get("value") in (None, ""):
            continue
        claims.append(
            AtomicClaim(
                entity_id=row["entity_id"],
                predicate=row["predicate"],
                value=row["value"],
                qualifiers=row.get("qualifiers") or {},
                modality=row.get("modality") or "asserted",
                source_span=row.get("source_span"),
                confidence=row.get("confidence") or "medium",
                vault_ref=VAULT_REF,
                source_kind="deterministic_extractor",
            )
        )
    new_keys = {compute_natural_key(claim) for claim in claims}
    try:
        if claims:
            written = write_fn(
                claims=claims,
                file_name=file_name,
                db_url=db_url,
                db_key=db_key,
            )
            record["claims_ecrits"] = len(written.written)
            record["claims_inchanges"] = len(written.unchanged)
        stale_ids: list[str] = []
        for row in list_active_tabular_claims(connection, file_name):
            claim = AtomicClaim(
                entity_id=row["entity_id"],
                predicate=row["predicate"],
                value=row["value"],
                qualifiers=row.get("qualifiers") or {},
                modality=row.get("modality") or "asserted",
            )
            if compute_natural_key(claim) not in new_keys:
                stale_ids.append(str(row["claim_id"]))
        record["claims_retires"] = delete_tabular_claims_by_ids(connection, stale_ids)
        record["statut"] = "complete"
    except Exception as exc:  # noqa: BLE001 — journal borné
        record["erreur"] = str(exc)
        record["statut"] = "erreur"
    return record


class _ForbiddenModelRunner:
    """Bloque tout appel modèle — reprise tabulaire sans LLM."""

    calls = 0
    providers: list[str] = []

    def run_skill(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"appel modèle interdit en reprise tabulaire: {name}")

    def primary_provider(self) -> str | None:
        return None


def _tabular_claim_dicts_ordinary(path: Path, text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extrait les faits tabulaires par la phase tabulaire du chemin ordinaire."""
    relative = source_path_for(path.name)
    if path.name.lower().endswith(".csv"):
        params = CoarseToFineParams(tabular=True)
        result = extract_claims_coarse_to_fine(
            file_name=path.name,
            text=text,
            run_skill=_ForbiddenModelRunner().run_skill,
            params=params,
            source_path=relative,
        )
        rows = [r for r in result.claims if r.get("extracted_by_skill") == TABULAR_SKILL]
        return rows, {
            "tables": result.tabular_tables,
            "lignes_tabulaires": result.tabular_rows,
            "lignes_illisibles": result.tabular_illisible_lines,
            "tabular_marker_claims_dropped": result.tabular_marker_claims_dropped,
            "tabular_marker_claims_dropped_unanchored": (
                result.tabular_marker_claims_dropped_unanchored
            ),
        }
    try:
        from brain.claiming.structural import parse_structure
    except ImportError:
        parse_structure = None  # type: ignore[misc,assignment]
    try:
        from brain.claiming.tabular import extract_tabular_claims
    except ImportError:
        extract_tabular_claims = None  # type: ignore[misc,assignment]

    structure = parse_structure(text)
    parsed = extract_tabular_claims(
        file_name=path.name,
        text=text,
        structure=structure,
        source_path=relative,
    )
    return list(parsed.all_claims), {
        "tables": len(parsed.tables),
        "lignes_tabulaires": parsed.data_row_count,
        "lignes_illisibles": parsed.illisible_lines,
        "tabular_marker_claims_dropped": 0,
        "tabular_marker_claims_dropped_unanchored": 0,
    }


def ingest_tabular_ordinary_piece(
    path: Path,
    *,
    db_url: str,
    db_key: str,
    connection,
    ingest_log: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Ré-ingère les faits tabulaires d'une pièce par le chemin ordinaire, sans LLM."""
    text = path.read_text(encoding="utf-8")
    started = time.monotonic()
    record: dict[str, Any] = {
        "mode": "tabulaire-ordinaire-reprise",
        "run_id": run_id or str(uuid.uuid4()),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": path.name,
        "size": len(text.encode()),
        "tables": 0,
        "lignes_tabulaires": 0,
        "lignes_illisibles": 0,
        "claims_ecrits": 0,
        "claims_inchanges": 0,
        "claims_retires": 0,
        "claims_tabulaires": 0,
        "appels_modele": 0,
        "faits_ligne": 0,
        "faits_cellule": 0,
        "tabular_marker_claims_dropped": 0,
        "tabular_marker_claims_dropped_unanchored": 0,
        "duree_s": 0.0,
        "statut": "erreur",
        "erreur": None,
    }
    log_path = ingest_log or INGEST_LOG
    try:
        parsed_claims, meta = _tabular_claim_dicts_ordinary(path, text)
        record.update(meta)
        relative = source_path_for(path.name)
        source_file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"banc://{path.name}"))

        class _TabularResult:
            def prompt_version(self) -> str:
                return TABULAR_SKILL

        def _write_tabular(claims: list[AtomicClaim], **_kwargs):
            return write_claims_with_retry(
                claims=claims,
                source_file_path=relative,
                source_file_id=source_file_id,
                runner=_ForbiddenModelRunner(),
                result=_TabularResult(),
                db_url=db_url,
                db_key=db_key,
                skill_name=TABULAR_SKILL,
            )

        synced = sync_tabular_claims_for_piece(
            file_name=path.name,
            parsed_claims=parsed_claims,
            connection=connection,
            write_fn=_write_tabular,
            db_url=db_url,
            db_key=db_key,
        )
        line_count, cell_count = count_tabular_scopes_for_piece(connection, path.name)
        record.update(
            {
                "claims_ecrits": synced["claims_ecrits"],
                "claims_inchanges": synced["claims_inchanges"],
                "claims_retires": synced["claims_retires"],
                "claims_tabulaires": synced["claims_ecrits"] + synced["claims_inchanges"],
                "faits_ligne": line_count,
                "faits_cellule": cell_count,
                "statut": synced["statut"],
                "erreur": synced.get("erreur"),
            }
        )
    except Exception as exc:  # noqa: BLE001 — journal borné
        record["erreur"] = str(exc)
        record["statut"] = "erreur"
    record["duree_s"] = round(time.monotonic() - started, 2)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def run_tabular_ordinary_reprise(
    *,
    config: IngestionConfig | None = None,
) -> dict[str, Any]:
    from .config import export_runtime_env
    from .infra import start_cluster

    cfg = config or IngestionConfig.for_ingestion3()
    export_runtime_env()
    ensure_area()
    cfg.ingest_log.parent.mkdir(parents=True, exist_ok=True)
    start_cluster()
    connection = connect_existing()
    url = os.environ["BRAIN_POSTGREST_URL"]
    key = os.environ["BRAIN_POSTGREST_SERVICE_KEY"]
    calls_before = load_call_count(cfg.call_log)
    summary: dict[str, Any] = {
        "mode": "tabulaire-ordinaire-reprise",
        "run_id": str(uuid.uuid4()),
        "pieces": [],
        "appels_modele": 0,
        "erreurs": [],
    }
    for path in list_corpus():
        record = ingest_tabular_ordinary_piece(
            path,
            db_url=url,
            db_key=key,
            connection=connection,
            ingest_log=cfg.ingest_log,
            run_id=summary["run_id"],
        )
        summary["pieces"].append(record)
        if record.get("statut") == "erreur":
            summary["erreurs"].append(path.name)
    summary["appels_modele"] = load_call_count(cfg.call_log) - calls_before
    summary["duree_s"] = round(
        sum(float(piece.get("duree_s", 0)) for piece in summary["pieces"]),
        1,
    )
    if summary["appels_modele"] > 0:
        summary["arret"] = "appel_modele_interdit"
    write_ingestion_status(summary, cfg)
    return summary


def rewrite_tabular_piece(
    path: Path,
    *,
    db_url: str,
    db_key: str,
    connection,
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    started = time.monotonic()
    record: dict[str, Any] = {
        "mode": "tabulaire-seul",
        "run_id": str(uuid.uuid4()),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": path.name,
        "tables": 0,
        "lignes_tabulaires": 0,
        "lignes_illisibles": 0,
        "claims_supprimes": 0,
        "claims_ecrits": 0,
        "claims_inchanges": 0,
        "appels_modele": 0,
        "duree_s": 0.0,
        "statut": "erreur",
        "erreur": None,
    }
    try:
        try:
            from brain.claiming.tabular import extract_tabular_claims
        except ImportError:
            extract_tabular_claims = None  # type: ignore[misc,assignment]

        parsed = extract_tabular_claims(file_name=path.name, text=text)
        record["tables"] = len(parsed.tables)
        record["lignes_tabulaires"] = parsed.data_row_count
        record["lignes_illisibles"] = parsed.illisible_lines
        relative = source_path_for(path.name)
        source_file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"banc://{path.name}"))

        class _TabularResult:
            def prompt_version(self) -> str:
                return TABULAR_SKILL

        def _write_tabular(claims: list[AtomicClaim], **_kwargs):
            return write_claims_with_retry(
                claims=claims,
                source_file_path=relative,
                source_file_id=source_file_id,
                runner=RelaySkillRunner(),
                result=_TabularResult(),
                db_url=db_url,
                db_key=db_key,
                skill_name=TABULAR_SKILL,
            )

        synced = sync_tabular_claims_for_piece(
            file_name=path.name,
            parsed_claims=parsed.all_claims,
            connection=connection,
            write_fn=_write_tabular,
            db_url=db_url,
            db_key=db_key,
        )
        record.update(
            {
                "claims_ecrits": synced["claims_ecrits"],
                "claims_inchanges": synced["claims_inchanges"],
                "claims_supprimes": synced["claims_retires"],
                "statut": synced["statut"],
                "erreur": synced.get("erreur"),
            }
        )
    except Exception as exc:  # noqa: BLE001 — journal borné
        record["erreur"] = str(exc)
        record["statut"] = "erreur"
    record["duree_s"] = round(time.monotonic() - started, 2)
    with INGEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def run_tabulaire_seul() -> dict[str, Any]:
    from .config import export_runtime_env
    from .infra import start_cluster

    export_runtime_env()
    ensure_area()
    INGEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    start_cluster()
    connection = connect_existing()
    url = os.environ["BRAIN_POSTGREST_URL"]
    key = os.environ["BRAIN_POSTGREST_SERVICE_KEY"]
    summary = {"mode": "tabulaire-seul", "pieces": [], "appels_modele": 0, "erreurs": []}
    for path in list_corpus():
        record = rewrite_tabular_piece(path, db_url=url, db_key=key, connection=connection)
        summary["pieces"].append(record)
        if record.get("statut") == "erreur":
            summary["erreurs"].append(path.name)
    embedded = backfill_embeddings(
        max_embeddings=2500,
        log_path=BANC_ROOT / "sorties" / "tabulaire-v2" / "embedding-appels.jsonl",
    )
    summary["embedded"] = embedded["embedded"]
    summary["embeddings_cap"] = embedded.get("capped")
    return summary


def tabular_stats(connection) -> dict[str, Any]:
    row = connection.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(char_length(value::text)), 0) AS chars "
        "FROM brain.claims WHERE lifecycle_status = 'active' AND vault_ref = %s "
        "AND extracted_by_skill = %s",
        (VAULT_REF, TABULAR_SKILL),
    ).fetchone()
    tables = connection.execute(
        "SELECT COUNT(DISTINCT source_file_path) AS files FROM brain.claims "
        "WHERE lifecycle_status = 'active' AND vault_ref = %s AND extracted_by_skill = %s",
        (VAULT_REF, TABULAR_SKILL),
    ).fetchone()
    return {
        "faits_tabulaires": int(row["n"]),
        "chars_valeurs": int(row["chars"]),
        "pieces_avec_faits": int(tables["files"]),
    }


def backfill_embeddings(
    *,
    retry_attempts: int = 4,
    retry_delay_s: float = 1.0,
    max_embeddings: int | None = None,
    log_path: Path | None = None,
    vault_ref: str | None = None,
) -> dict[str, int]:
    from banc_realite.compat_moteur import MODEL_ID, MistralEmbeddingError, claim_embed_text, embed_texts, to_pgvector

    connection = connect_existing()
    sql = (
        "SELECT claim_id, entity_id, predicate, value, qualifiers "
        "FROM brain.claims WHERE lifecycle_status = 'active' "
        "AND predicate = 'note.body' AND value_embedding_mistral IS NULL "
    )
    params: list[Any] = []
    if vault_ref is not None:
        sql += "AND vault_ref = %s "
        params.append(vault_ref)
    sql += "ORDER BY claim_id"
    rows = connection.execute(sql, params).fetchall()
    original_count = len(rows)
    if max_embeddings is not None:
        rows = rows[:max_embeddings]
    embedded = 0
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
    for offset in range(0, len(rows), 16):
        batch = rows[offset : offset + 16]
        texts = [
            claim_embed_text(r["entity_id"], r["value"], r.get("qualifiers") or {})
            for r in batch
        ]
        vectors = _embed_batch_with_retry(
            texts,
            embed_fn=embed_texts,
            attempts=retry_attempts,
            delay_s=retry_delay_s,
            error_type=MistralEmbeddingError,
        )
        with connection.transaction():
            for row, vector in zip(batch, vectors):
                connection.execute(
                    "UPDATE brain.claims SET value_embedding_mistral = %s::extensions.vector, "
                    "embedded_mistral_at = now(), embedding_mistral_model_id = %s WHERE claim_id = %s",
                    (to_pgvector(vector), MODEL_ID, row["claim_id"]),
                )
                embedded += 1
                if log_handle is not None:
                    log_handle.write(
                        json.dumps(
                            {
                                "claim_id": str(row["claim_id"]),
                                "entity_id": row["entity_id"],
                                "batch_offset": offset,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    if log_handle is not None:
        log_handle.close()
    return {
        "embedded": embedded,
        "capped": max_embeddings,
        "skipped": max(0, original_count - embedded),
    }


def _embed_batch_with_retry(
    texts: list[str],
    *,
    embed_fn,
    attempts: int,
    delay_s: float,
    error_type: type[Exception],
) -> list[list[float]]:
    for attempt in range(attempts):
        try:
            return embed_fn(texts)
        except error_type as exc:
            message = str(exc)
            retryable = (
                "HTTP 429" in message
                or "HTTP 500" in message
                or "HTTP 502" in message
                or "HTTP 503" in message
                or "HTTP 504" in message
                or "network error" in message
            )
            if not retryable or attempt == attempts - 1:
                raise
            wait_s = 30.0 if "HTTP 429" in message else delay_s
            time.sleep(wait_s)
            if "HTTP 429" not in message:
                delay_s *= 2
    raise error_type("unreachable")


def clear_public_claims(connection) -> None:
    connection.execute("TRUNCATE public.claim_assets, public.claims;")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if "--ingestion3" in sys.argv:
        cfg = IngestionConfig.for_ingestion3()
    elif "--ingestion2" in sys.argv:
        cfg = IngestionConfig.for_ingestion2()
    else:
        cfg = IngestionConfig()
    if mode == "trial":
        print(
            json.dumps(
                run_ingestion(trial_only=True, reset_db="--reset" in sys.argv, config=cfg),
                indent=2,
            )
        )
    elif mode == "all":
        print(json.dumps(run_ingestion(reset_db="--reset" in sys.argv, config=cfg), indent=2))
    elif mode == "embed":
        print(json.dumps(backfill_embeddings(), indent=2))
    elif mode == "tabulaire-seul":
        print(json.dumps(run_tabulaire_seul(), indent=2))
    else:
        raise SystemExit(f"unknown mode: {mode}")
