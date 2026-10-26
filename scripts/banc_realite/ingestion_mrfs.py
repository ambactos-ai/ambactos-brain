# SPDX-License-Identifier: AGPL-3.0-only
"""Ingestion MRFS banc — pièces uniquement, jamais l'or (issue #137 étape 2)."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from brain.claiming import CoarseToFineParams, extract_claims_coarse_to_fine
except ImportError:
    CoarseToFineParams = None  # type: ignore[misc,assignment]
    extract_claims_coarse_to_fine = None  # type: ignore[misc,assignment]
try:
    from brain.claiming.mrfs_stage import publish_claiming_to_mrfs
except ImportError:
    publish_claiming_to_mrfs = None  # type: ignore[misc,assignment]
try:
    from brain.claiming.tabular import tabular_enabled_from_env
except ImportError:
    tabular_enabled_from_env = None  # type: ignore[misc,assignment]
try:
    from brain.mrfs.contracts import banc_realite_subject_id
except ImportError:
    banc_realite_subject_id = None  # type: ignore[misc,assignment]

from .config import BANC_ACTOR_ID, BANC_ROOT, export_runtime_env
from .dossiers import Dossier, dossier_courant
from .ingest_runner import append_reprise_record
from .mrfs_support import (
    BANC_PG_DSN,
    DEMO_RECORD_KEY,
    MRFS_SCHEMA,
    clear_moreau_demo_situation,
    ensure_mrfs_schema_installed,
    ensure_subjects_registered,
    load_subject_registry,
    require_source_file,
)
from .normalize import piece_basename
from .passages import Passage, chunk_piece
from .projection_buffer import append_buffered_record, flush_projection_buffer
from .relay import MODEL, ModelCallCapError, RelaySkillRunner

MRFS_DIR = BANC_ROOT / "dossiers" / "mrfs"
APPELS_LOG = MRFS_DIR / "appels-ingestion-mrfs.jsonl"
INGEST_LOG = MRFS_DIR / "ingestion-mrfs.jsonl"
REPRISE_LOG = MRFS_DIR / "reprise-ingestion-mrfs.jsonl"
RAPPORT_PATH = MRFS_DIR / "RAPPORT-INGESTION-MRFS.md"
RESULT_PATH = MRFS_DIR / "_grok-result.json"

CALL_CAPS: dict[str, int] = {"greenloop": 70, "moreau": 40}
TOTAL_CALL_CAP = 110
NEW_CALL_CAP = 110
FORBIDDEN_IMPORTS = frozenset({"or_dossier", "load_gold", "fidelite"})
PROTECTED_SUBJECTS = frozenset({"greenloop", "moreau"})


@dataclass(frozen=True, slots=True)
class IngestionMrfsConfig:
    output_dir: Path = MRFS_DIR
    appels_log: Path = APPELS_LOG
    ingest_log: Path = INGEST_LOG
    reprise_log: Path = REPRISE_LOG
    rapport_path: Path = RAPPORT_PATH
    result_path: Path = RESULT_PATH
    dsn: str = BANC_PG_DSN
    actor_id: str = str(BANC_ACTOR_ID)
    call_caps: dict[str, int] = field(default_factory=lambda: dict(CALL_CAPS))
    total_call_cap: int = TOTAL_CALL_CAP
    new_call_cap: int = NEW_CALL_CAP
    baseline_call_ids: frozenset[str] = frozenset()


def module_refuses_gold_access() -> tuple[bool, list[str]]:
    """Analyse statique : le module d'ingestion ne lit ni or_dossier ni gold/."""
    source_path = Path(__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] in FORBIDDEN_IMPORTS:
                violations.append(f"from {mod} import ...")
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    violations.append(f"from {mod} import {alias.name}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"open", "read_text"} and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    lowered = arg0.value.lower()
                    if "/gold" in lowered or lowered.endswith("gold"):
                        violations.append(f"open gold path: {arg0.value!r}")
    return not violations, violations


def passage_key(passage: Passage) -> str:
    return f"{passage.piece}:{passage.line_start}:{passage.line_end}"


def list_corpus_pieces(dossier: Dossier) -> list[str]:
    if not dossier.corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus absent : {dossier.corpus_dir}")
    return sorted(path.name for path in dossier.corpus_dir.iterdir() if path.is_file())


def list_passages(dossier: Dossier) -> list[Passage]:
    passages: list[Passage] = []
    for piece in list_corpus_pieces(dossier):
        passages.extend(chunk_piece(piece, corpus=dossier.corpus_dir))
    return passages


def _load_completed_keys(log_path: Path) -> set[str]:
    if not log_path.is_file():
        return set()
    done: set[str] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("statut") in {"complete", "skipped"} and record.get("passage_key"):
            done.add(str(record["passage_key"]))
    return done


def _append_ingest_record(record: dict[str, Any], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_appel_record(record: dict[str, Any], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _unique_call_ids(log_path: Path) -> set[str]:
    if not log_path.is_file():
        return set()
    ids: set[str] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        call_id = record.get("call_id")
        if call_id:
            ids.add(str(call_id))
    return ids


def count_model_calls(cfg: IngestionMrfsConfig | None = None) -> int:
    current = cfg or IngestionMrfsConfig()
    return len(_unique_call_ids(current.appels_log))


def count_new_model_calls(cfg: IngestionMrfsConfig | None = None) -> int:
    current = cfg or IngestionMrfsConfig()
    baseline = current.baseline_call_ids or frozenset()
    return len(_unique_call_ids(current.appels_log) - baseline)


def count_model_calls_for_dossier(
    dossier_name: str,
    *,
    cfg: IngestionMrfsConfig | None = None,
    new_only: bool = True,
) -> int:
    current = cfg or IngestionMrfsConfig()
    if not current.appels_log.is_file():
        return 0
    baseline = current.baseline_call_ids if new_only else frozenset()
    count = 0
    for line in current.appels_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        call_id = record.get("call_id")
        if not call_id:
            continue
        if str(call_id) in baseline:
            continue
        if record.get("dossier") == dossier_name:
            count += 1
    return count


def _claim_verbatim_text(claim: dict[str, Any], passage_text: str, piece_lines: list[str]) -> str:
    span = claim.get("source_span")
    if isinstance(span, dict):
        raw = span.get("raw_line") or span.get("exact") or span.get("quote")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        line_start = span.get("line_start")
        line_end = span.get("line_end", line_start)
        if isinstance(line_start, int) and isinstance(line_end, int) and piece_lines:
            start = max(1, line_start)
            end = min(len(piece_lines), line_end)
            excerpt = "\n".join(piece_lines[start - 1 : end]).strip()
            if excerpt:
                return excerpt
        start = span.get("start")
        end = span.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(passage_text):
            return passage_text[start:end].strip()
    value = claim.get("value")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _temporal_fields(claim: dict[str, Any]) -> dict[str, str]:
    qualifiers = claim.get("qualifiers") or {}
    out: dict[str, str] = {}
    for key in ("date", "since", "until", "period", "periode", "valid_from", "valid_to"):
        value = qualifiers.get(key)
        if isinstance(value, str) and value.strip():
            out[f"temporal_{key}"] = value.strip()
    return out


def _claim_record_key(
    *,
    subject_id: str,
    piece: str,
    passage: Passage,
    claim: dict[str, Any],
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "subject_id": subject_id,
                "piece": piece,
                "line_start": passage.line_start,
                "line_end": passage.line_end,
                "entity_id": claim.get("entity_id"),
                "predicate": claim.get("predicate"),
                "value": claim.get("value"),
                "span": claim.get("source_span"),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"ingest:{piece}:{passage.line_start}-{passage.line_end}:{digest}"





def _publish_claim_projection_records(
    *,
    connection,
    subject_id: str,
    piece: str,
    passage: Passage,
    claims: list[dict[str, Any]],
    passage_text: str,
    piece_lines: list[str],
    output_dir: Path,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> list[str]:
    if not claims:
        return []
    source_piece = require_source_file(piece)
    inserted: list[str] = []
    for claim in claims:
        verbatim = _claim_verbatim_text(claim, passage_text, piece_lines)
        if not verbatim:
            continue
        record_key = _claim_record_key(
            subject_id=subject_id,
            piece=piece,
            passage=passage,
            claim=claim,
        )
        temporal = _temporal_fields(claim)
        payload = {
            "record_key": record_key,
            "verbatim": verbatim,
            "search_fields": {
                "source_file": source_piece,
                "line_start": passage.line_start,
                "line_end": passage.line_end,
                **temporal,
            },
            "perspective_summary": {
                "text": verbatim,
                "source_file": source_piece,
                "line_start": passage.line_start,
                "line_end": passage.line_end,
                **temporal,
            },
            "identity_summary": {
                "source_piece": source_piece,
                "line_start": passage.line_start,
                "line_end": passage.line_end,
            },
        }
        append_buffered_record(output_dir, subject_id, payload)
        inserted.append(record_key)
    return inserted



def _mrfs_element_from_record(row: dict[str, Any]) -> dict[str, Any]:
    perspective = row.get("perspective_summary") or {}
    if isinstance(perspective, str):
        perspective = json.loads(perspective)
    identity = row.get("identity_summary") or {}
    if isinstance(identity, str):
        identity = json.loads(identity)
    source = (
        perspective.get("source_file")
        or identity.get("source_piece")
        or row.get("source_file")
        or ""
    )
    text = row.get("lexical_text") or perspective.get("text") or ""
    return {
        "kind": "mrfs_situation",
        "text": text,
        "source": source,
        "provenance": {"source_file_path": source, "record_key": row.get("record_key")},
    }


def list_projection_elements(
    subject_id: str,
    *,
    dsn: str = BANC_PG_DSN,
    exclude_demo: bool = True,
) -> list[dict[str, Any]]:
    import psycopg

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        snapshot_row = connection.execute(
            f"SELECT snapshot_id FROM {MRFS_SCHEMA}.projection_snapshots "
            "WHERE subject_id = %s ORDER BY built_at DESC LIMIT 1",
            (subject_id,),
        ).fetchone()
        if snapshot_row is None:
            return []
        snapshot_id = snapshot_row[0]
        rows = connection.execute(
            f"SELECT record_key, lexical_text, perspective_summary, identity_summary "
            f"FROM {MRFS_SCHEMA}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s",
            (subject_id, snapshot_id),
        ).fetchall()
    finally:
        connection.close()
    elements: list[dict[str, Any]] = []
    for row in rows:
        mapping = {
            "record_key": row[0],
            "lexical_text": row[1],
            "perspective_summary": row[2],
            "identity_summary": row[3],
        }
        if exclude_demo and mapping["record_key"] == DEMO_RECORD_KEY:
            continue
        element = _mrfs_element_from_record(mapping)
        if element["text"]:
            elements.append(element)
    return elements


def _stats_by_piece(elements: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements:
        piece = piece_basename(element.get("source") or "inconnue")
        counts[piece] = counts.get(piece, 0) + 1
    return counts


def _median_proposition_length(subject_id: str, *, dsn: str = BANC_PG_DSN) -> float:
    import psycopg

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        snapshot_row = connection.execute(
            f"SELECT snapshot_id FROM {MRFS_SCHEMA}.projection_snapshots "
            "WHERE subject_id = %s ORDER BY built_at DESC LIMIT 1",
            (subject_id,),
        ).fetchone()
        if snapshot_row is None:
            return 0.0
        snapshot_id = snapshot_row[0]
        rows = connection.execute(
            f"SELECT length(lexical_text) FROM {MRFS_SCHEMA}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s AND record_key <> %s",
            (subject_id, snapshot_id, DEMO_RECORD_KEY),
        ).fetchall()
    finally:
        connection.close()
    lengths = [int(row[0]) for row in rows if row[0]]
    return float(statistics.median(lengths)) if lengths else 0.0


class _JournalRelay:
    """Journalise chaque appel modèle avec un call_id unique."""

    def __init__(
        self,
        runner: RelaySkillRunner,
        appels_log: Path,
        *,
        dossier_name: str | None = None,
        piece: str | None = None,
        passage_key: str | None = None,
    ):
        self._runner = runner
        self._appels_log = appels_log
        self._dossier_name = dossier_name
        self._piece = piece
        self._passage_key = passage_key
        self._situations_published = 0

    def set_passage_context(
        self,
        *,
        dossier_name: str,
        piece: str,
        passage_key: str,
    ) -> None:
        self._dossier_name = dossier_name
        self._piece = piece
        self._passage_key = passage_key
        self._situations_published = 0

    def note_situations(self, count: int) -> None:
        self._situations_published = int(count)

    def _context_fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self._dossier_name:
            payload["dossier"] = self._dossier_name
        if self._piece:
            payload["piece"] = self._piece
        if self._passage_key:
            payload["passage_key"] = self._passage_key
        if self._situations_published:
            payload["situations_publiees"] = self._situations_published
        return payload

    def run_skill(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        call_id = str(uuid.uuid4())
        started = time.monotonic()
        try:
            payload = self._runner.run_skill(name, args)
        except Exception as exc:  # noqa: BLE001
            _append_appel_record(
                {
                    "call_id": call_id,
                    "skill": name,
                    "fournisseur": self._runner.primary_provider(),
                    "modele": MODEL,
                    "status": "erreur",
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "erreur": str(exc),
                    **self._context_fields(),
                },
                log_path=self._appels_log,
            )
            raise
        _append_appel_record(
            {
                "call_id": call_id,
                "skill": name,
                "fournisseur": payload.get("executor_provider") or self._runner.primary_provider(),
                "modele": payload.get("executor_model") or MODEL,
                "status": "ok",
                "duration_ms": int((time.monotonic() - started) * 1000),
                **self._context_fields(),
            },
            log_path=self._appels_log,
        )
        return payload


def ingest_passage(
    passage: Passage,
    *,
    dossier: Dossier,
    subject_id: str,
    runner: RelaySkillRunner,
    connection,
    cfg: IngestionMrfsConfig,
    run_skill: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    key = passage_key(passage)
    piece_path = dossier.corpus_dir / passage.piece
    passage_text = passage.content
    piece_lines = piece_path.read_text(encoding="utf-8").splitlines()
    calls_before = runner.calls
    record: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dossier": dossier.nom,
        "piece": passage.piece,
        "passage_key": key,
        "line_start": passage.line_start,
        "line_end": passage.line_end,
        "appels_modele": 0,
        "situations": 0,
        "statut": "erreur",
        "erreur": None,
    }
    skill_runner = run_skill or runner.run_skill
    if isinstance(run_skill, _JournalRelay):
        run_skill.set_passage_context(
            dossier_name=dossier.nom,
            piece=passage.piece,
            passage_key=key,
        )
    try:
        params = CoarseToFineParams(tabular=tabular_enabled_from_env())
        result = extract_claims_coarse_to_fine(
            file_name=passage.piece,
            text=passage_text,
            run_skill=skill_runner,
            params=params,
            source_path=f"documents/{passage.piece}#L{passage.line_start}-L{passage.line_end}",
        )
        storage_path = Path(
            f"documents/{passage.piece}#L{passage.line_start}-L{passage.line_end}"
        )
        os.environ.setdefault("BRAIN_MRFS_PROJECTION", "1")
        os.environ.setdefault("BRAIN_MRFS_DSN", cfg.dsn)
        mrfs_error = None
        try:
            publish_claiming_to_mrfs(
                result,
                text=passage_text,
                file_path=storage_path,
                vault_ref=subject_id,
                actor_id=cfg.actor_id,
                dsn=cfg.dsn,
                schema=MRFS_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001 — projection kernel non bloquante
            mrfs_error = str(exc)
            if "ingestion_identity_conflict" in mrfs_error:
                mrfs_error = None  # rejeu idempotent
        claims = [dict(row) for row in result.claims]
        inserted = _publish_claim_projection_records(
            connection=connection,
            subject_id=subject_id,
            piece=passage.piece,
            passage=passage,
            claims=claims,
            passage_text=passage_text,
            piece_lines=piece_lines,
            output_dir=cfg.output_dir,
            dsn=cfg.dsn,
        )
        record["situations"] = len(inserted)
        record["claims_extraits"] = len(claims)
        if isinstance(run_skill, _JournalRelay):
            run_skill.note_situations(len(inserted))
        if mrfs_error:
            record["mrfs_stage_erreur"] = mrfs_error
        if claims:
            record["statut"] = "complete"
        elif mrfs_error and "ingestion_identity_conflict" not in str(mrfs_error):
            record["statut"] = "erreur"
        elif result.errors:
            record["statut"] = "erreur"
            raise RuntimeError("; ".join(result.errors))
        else:
            record["statut"] = "complete"
    except ModelCallCapError as exc:
        record["erreur"] = str(exc)
        record["statut"] = "partielle"
        record["arret"] = "plafond_appels"
    except Exception as exc:  # noqa: BLE001
        record["erreur"] = str(exc)
        record["statut"] = "erreur"
    record["appels_modele"] = runner.calls - calls_before
    record["duree_s"] = round(time.monotonic() - started, 2)
    _append_ingest_record(record, log_path=cfg.ingest_log)
    return record


def run_ingestion_mrfs(
    dossier_name: str,
    *,
    cfg: IngestionMrfsConfig | None = None,
    global_calls: Callable[[], int] | None = None,
) -> dict[str, Any]:
    cfg = cfg or IngestionMrfsConfig()
    dossier = dossier_courant(dossier_name)
    cap = cfg.call_caps.get(dossier.nom, 0)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_mrfs_schema_installed(dsn=cfg.dsn)
    registry = ensure_subjects_registered(dsn=cfg.dsn)
    subject_id = registry[dossier.instance]
    import psycopg

    connection = psycopg.connect(cfg.dsn, autocommit=True)
    completed = _load_completed_keys(cfg.ingest_log)
    runner = RelaySkillRunner(
        log_path=cfg.output_dir / f"skill-calls-{dossier.nom}.jsonl",
        include_prior_report=False,
        max_calls=cap if cap > 0 else None,
    )
    journal_runner = _JournalRelay(
        runner,
        cfg.appels_log,
        dossier_name=dossier.nom,
    )
    summary: dict[str, Any] = {
        "dossier": dossier.nom,
        "subject_id": subject_id,
        "passages": [],
        "appels_modele": count_model_calls(cfg),
        "appels_nouveaux": count_new_model_calls(cfg),
        "situations": 0,
        "pieces_flushed": {},
    }
    current_piece: str | None = None
    try:
        for passage in list_passages(dossier):
            new_calls = count_new_model_calls(cfg)
            if global_calls and global_calls() >= cfg.new_call_cap:
                summary["arret"] = "plafond_appels_nouveaux"
                append_reprise_record(
                    {
                        "dossier": dossier.nom,
                        "passage_key": passage_key(passage),
                        "arret": "plafond_appels_nouveaux",
                        "plafond": cfg.new_call_cap,
                    },
                    log_path=cfg.reprise_log,
                )
                break
            if cap is not None and cap >= 0:
                dossier_calls = count_model_calls_for_dossier(dossier.nom, cfg=cfg)
                if dossier_calls >= cap:
                    summary["arret"] = "plafond_appels"
                    append_reprise_record(
                        {
                            "dossier": dossier.nom,
                            "passage_key": passage_key(passage),
                            "arret": "plafond_appels",
                            "plafond": cap,
                            "appels_dossier": dossier_calls,
                        },
                        log_path=cfg.reprise_log,
                    )
                    break
            key = passage_key(passage)
            if key in completed:
                skip = {
                    "passage_key": key,
                    "piece": passage.piece,
                    "skipped": True,
                    "statut": "skipped",
                    "raison": "deja_complete",
                }
                summary["passages"].append(skip)
                append_reprise_record(skip, log_path=cfg.reprise_log)
                continue
            if current_piece and passage.piece != current_piece:
                flushed = flush_projection_buffer(
                    subject_id,
                    output_dir=cfg.output_dir,
                    dsn=cfg.dsn,
                )
                if flushed:
                    summary["pieces_flushed"][current_piece] = flushed
            current_piece = passage.piece
            record = ingest_passage(
                passage,
                dossier=dossier,
                subject_id=subject_id,
                runner=runner,
                connection=connection,
                cfg=cfg,
                run_skill=journal_runner.run_skill,
            )
            summary["passages"].append(record)
            summary["situations"] += int(record.get("situations") or 0)
            summary["appels_modele"] = count_model_calls(cfg)
            summary["appels_nouveaux"] = count_new_model_calls(cfg)
            append_reprise_record(record, log_path=cfg.reprise_log)
            completed.add(key)
            if record.get("arret") == "plafond_appels":
                summary["arret"] = "plafond_appels"
                break
    finally:
        connection.close()
    if current_piece:
        flushed = flush_projection_buffer(subject_id, output_dir=cfg.output_dir, dsn=cfg.dsn)
        if flushed:
            summary["pieces_flushed"][current_piece] = flushed
    summary["situations_publiees"] = sum(summary["pieces_flushed"].values())
    runner.assert_no_claude()
    summary["fournisseur"] = runner.primary_provider()
    return summary


def prepare_ingestion(*, cfg: IngestionMrfsConfig | None = None) -> tuple[dict[str, str], IngestionMrfsConfig]:
    from dataclasses import replace

    current = cfg or IngestionMrfsConfig()
    export_runtime_env()
    os.environ["BRAIN_MRFS_PROJECTION"] = "1"
    os.environ["BRAIN_MRFS_DSN"] = current.dsn
    ensure_mrfs_schema_installed(dsn=current.dsn)
    clear_moreau_demo_situation(dsn=current.dsn)
    registry = ensure_subjects_registered(dsn=current.dsn)
    from .mrfs_support import ensure_test_subject_registered

    ensure_test_subject_registered(dsn=current.dsn)
    current = replace(
        current,
        baseline_call_ids=frozenset(_unique_call_ids(current.appels_log)),
    )
    return registry, current


def run_full_ingestion(*, cfg: IngestionMrfsConfig | None = None) -> dict[str, Any]:
    _, current = prepare_ingestion(cfg=cfg)
    results: dict[str, Any] = {"dossiers": {}, "appels_modele": 0, "appels_nouveaux": 0}
    for dossier_name in ("greenloop", "moreau"):
        results["dossiers"][dossier_name] = run_ingestion_mrfs(
            dossier_name,
            cfg=current,
            global_calls=lambda: count_new_model_calls(current),
        )
        results["appels_modele"] = count_model_calls(current)
        results["appels_nouveaux"] = count_new_model_calls(current)
        if results["appels_nouveaux"] >= current.new_call_cap:
            results["arret"] = "plafond_appels_nouveaux"
            break
    return results
