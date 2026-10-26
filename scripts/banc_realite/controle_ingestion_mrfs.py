# SPDX-License-Identifier: AGPL-3.0-only
"""Contrôles post-ingestion MRFS — l'or est lu ici seulement, jamais pendant l'ingestion."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

try:
    from brain.mrfs.contracts import banc_realite_subject_id
except ImportError:
    banc_realite_subject_id = None  # type: ignore[misc,assignment]

from .dossiers import Dossier, dossier_courant
from .fournisseurs_contexte import rendre_contexte
from .holders import _prepare_claims, classify_pair
from .ingestion_mrfs import (
    IngestionMrfsConfig,
    MRFS_DIR,
    count_model_calls,
    count_new_model_calls,
    count_model_calls_for_dossier,
    list_projection_elements,
    module_refuses_gold_access,
    _median_proposition_length,
    _stats_by_piece,
)
from .mrfs_support import (
    BANC_PG_DSN,
    MAX_BUDGET_CHARS,
    MRFS_SCHEMA,
    load_subject_registry,
    list_polluted_projection_records,
    purge_polluted_projection_records,
    query_context,
)
from .normalize import piece_basename

RAPPORT_PATH = MRFS_DIR / "RAPPORT-INGESTION-MRFS.md"
RESULT_PATH = MRFS_DIR / "_grok-result.json"

def _load_gold_for_coverage(dossier: Dossier) -> dict[str, Any]:
    """Lecture or en dehors du module d'ingestion — contrôle post-ingestion uniquement."""
    if dossier.nom == "greenloop":
        situations_blob = json.loads(
            (dossier.gold_dir / "situations.json").read_text(encoding="utf-8")
        )
        assertions: list[dict[str, Any]] = []
        for situation in situations_blob.get("situations") or []:
            for assertion in situation.get("assertions") or []:
                item = dict(assertion)
                item["situation_id"] = situation.get("id")
                assertions.append(item)
        return {"assertions": assertions}
    assertions_blob = json.loads(
        (dossier.gold_dir / "assertions.json").read_text(encoding="utf-8")
    )
    raw = assertions_blob.get("assertions") or assertions_blob
    assertions = []
    for item in raw:
        row = dict(item)
        piece = str(row.get("piece") or row.get("source") or "").strip()
        row["source"] = piece
        assertions.append(row)
    return {"assertions": assertions}


def measure_or_coverage_mrfs(
    dossier: Dossier,
    *,
    subject_id: str,
    dsn: str = BANC_PG_DSN,
) -> dict[str, Any]:
    gold = _load_gold_for_coverage(dossier)
    elements = list_projection_elements(subject_id, dsn=dsn)
    _, claims_index = _prepare_claims(
        [
            {
                "claim_id": element["provenance"]["record_key"],
                "text": element["text"],
                "source_file_path": element["source"],
                "piece": piece_basename(element["source"]),
            }
            for element in elements
        ]
    )
    assertions = gold["assertions"]
    total = len(assertions)
    verbatim_hits = 0
    valeur_hits = 0
    manquantes: list[dict[str, str]] = []
    for assertion in assertions:
        piece = piece_basename(assertion.get("source") or assertion.get("piece") or "")
        piece_elements = [e for e in elements if piece_basename(e.get("source") or "") == piece]
        tient_verbatim = any(
            classify_pair(assertion, element, claims_index, mode_presence="verbatim") == "tient"
            for element in piece_elements
        )
        tient_valeur = any(
            classify_pair(assertion, element, claims_index, mode_presence="valeur") == "tient"
            for element in piece_elements
        )
        if tient_verbatim:
            verbatim_hits += 1
        if tient_valeur:
            valeur_hits += 1
        if not tient_verbatim or not tient_valeur:
            causes: list[str] = []
            if not tient_verbatim:
                causes.append("verbatim")
            if not tient_valeur:
                causes.append("valeur")
            manquantes.append(
                {
                    "id": str(assertion.get("id") or ""),
                    "piece": piece,
                    "cause": "+".join(causes) if causes else "inconnue",
                }
            )
    return {
        "or_porte_verbatim": f"{verbatim_hits}/{total}",
        "or_porte_valeur": f"{valeur_hits}/{total}",
        "manquantes": manquantes,
        "elements": len(elements),
    }


def verify_subject_isolation(
    *,
    dsn: str = BANC_PG_DSN,
    registry: dict[str, str] | None = None,
) -> dict[str, Any]:
    import psycopg

    subjects = registry or load_subject_registry()
    greenloop_id = subjects.get("greenloop") or banc_realite_subject_id("greenloop")
    moreau_id = subjects.get("moreau") or banc_realite_subject_id("moreau")
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        gl_snapshot = connection.execute(
            f"SELECT snapshot_id FROM {MRFS_SCHEMA}.projection_snapshots "
            "WHERE subject_id = %s ORDER BY built_at DESC LIMIT 1",
            (greenloop_id,),
        ).fetchone()
        mo_snapshot = connection.execute(
            f"SELECT snapshot_id FROM {MRFS_SCHEMA}.projection_snapshots "
            "WHERE subject_id = %s ORDER BY built_at DESC LIMIT 1",
            (moreau_id,),
        ).fetchone()
        gl_rows = connection.execute(
            f"SELECT record_key, perspective_summary FROM {MRFS_SCHEMA}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s",
            (greenloop_id, gl_snapshot[0] if gl_snapshot else None),
        ).fetchall() if gl_snapshot else []
        mo_rows = connection.execute(
            f"SELECT record_key, perspective_summary FROM {MRFS_SCHEMA}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s",
            (moreau_id, mo_snapshot[0] if mo_snapshot else None),
        ).fetchall() if mo_snapshot else []
    finally:
        connection.close()
    gl_keys = {row[0] for row in gl_rows if str(row[0]).startswith("ingest:")}
    mo_keys = {row[0] for row in mo_rows if str(row[0]).startswith("ingest:")}
    overlap = gl_keys & mo_keys
    return {
        "isolation_ok": not overlap,
        "overlap_record_keys": sorted(overlap),
        "greenloop_records": len(gl_rows),
        "moreau_records": len(mo_rows),
        "greenloop_ingest_records": len(gl_keys),
        "moreau_ingest_records": len(mo_keys),
    }




def write_rapport_ingestion_mrfs(
    *,
    cfg: IngestionMrfsConfig | None = None,
    ingestion: dict[str, Any] | None = None,
) -> Path:
    current = cfg or IngestionMrfsConfig()
    registry = load_subject_registry()
    lines = [
        "# Rapport d'ingestion MRFS — étape 3",
        "",
        "## Chaîne",
        "",
        "- Lecture **pièces** (`documents/`) via `passages.chunk_piece`",
        "- Extraction coarse-to-fine Grok 4.5 (Cursor) puis `publish_claiming_to_mrfs`",
        "- Projection S7 (lexical) — **aucun embedding**",
        "- **L'or n'entre pas dans l'ingestion** (contrôle statique des imports)",
        "",
    ]
    ok, violations = module_refuses_gold_access()
    lines.append(f"- Preuve isolation or/ingestion : `{ok}`")
    if violations:
        lines.append(f"  - violations : {violations}")
    lines.append("")
    appels = count_model_calls(current)
    appels_nouveaux = count_new_model_calls(current)
    lines.extend(
        [
            f"## Appels modèle : {appels} total, {appels_nouveaux} nouveaux "
            f"(plafond nouveaux {current.new_call_cap})",
            "",
        ]
    )
    for nom in ("greenloop", "moreau"):
        lines.append(
            f"- `{nom}` nouveaux appels journal : "
            f"{count_model_calls_for_dossier(nom, cfg=current)} / {current.call_caps.get(nom, 0)}"
        )
    lines.append("")
    if ingestion:
        lines.append(f"- Journal ingestion : `{current.ingest_log}`")
        lines.append(f"- Journal appels : `{current.appels_log}`")
        lines.append("")
    for nom in ("greenloop", "moreau"):
        dossier = dossier_courant(nom)
        subject_id = registry.get(nom) or banc_realite_subject_id(nom)
        elements = list_projection_elements(subject_id, dsn=current.dsn)
        by_piece = _stats_by_piece(elements)
        cov = measure_or_coverage_mrfs(dossier, subject_id=subject_id, dsn=current.dsn)
        median = _median_proposition_length(subject_id, dsn=current.dsn)
        lines.extend(
            [
                f"## {nom}",
                "",
                f"- Situations (projection) : {len(elements)}",
                f"- Couverture or verbatim : {cov['or_porte_verbatim']}",
                f"- Couverture or valeur : {cov['or_porte_valeur']}",
                f"- Longueur médiane proposition : {median:.0f} car.",
                "",
                "### Situations par pièce",
                "",
            ]
        )
        for piece, count in sorted(by_piece.items()):
            lines.append(f"- `{piece}` : {count}")
        lines.append("")
        manquantes = cov.get("manquantes") or []
        if manquantes:
            lines.append("### Manquantes principales")
            lines.append("")
            by_piece_missing: dict[str, list[str]] = {}
            for item in manquantes[:30]:
                by_piece_missing.setdefault(item["piece"], []).append(
                    f"{item['id']} ({item['cause']})"
                )
            for piece, ids in sorted(by_piece_missing.items()):
                lines.append(f"- `{piece}` : {', '.join(ids[:5])}")
            lines.append("")
    isolation = verify_subject_isolation(dsn=current.dsn, registry=registry)
    lines.extend(
        [
            "## Isolation des sujets",
            "",
            f"- OK : {isolation['isolation_ok']}",
            f"- GreenLoop : {isolation['greenloop_records']} enregistrements",
            f"- MOREAU : {isolation['moreau_records']} enregistrements",
            "",
            "## Couverture via brain_mrfs.contexte (budget 96 000)",
            "",
        ]
    )
    for nom in ("greenloop", "moreau"):
        dossier = dossier_courant(nom)
        elements, stats = query_context(
            {"question": dossier.responsibility_context},
            dossier=dossier,
            k=50,
            connection=None,
        )
        rendered, kept = rendre_contexte(
            elements,
            budget_chars=MAX_BUDGET_CHARS,
            style="texte",
            dossier=dossier,
            corpus_chars=stats.corpus_chars,
        )
        lines.append(
            f"- `{nom}` : {len(kept)} éléments rendus, {len(rendered)} car. "
            f"(budget {MAX_BUDGET_CHARS})"
        )
    current.rapport_path.parent.mkdir(parents=True, exist_ok=True)
    current.rapport_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return current.rapport_path


def write_grok_result(
    *,
    cfg: IngestionMrfsConfig | None = None,
    tests_passed: int = 0,
    lecteur_repare: bool = False,
    purge: dict[str, int] | None = None,
    pieces_couvertes: dict[str, str] | None = None,
    status: str = "complete",
) -> Path:
    current = cfg or IngestionMrfsConfig()
    registry = load_subject_registry()
    situations: dict[str, int] = {}
    coverage_or: dict[str, str] = {}
    coverage_valeur: dict[str, str] = {}
    medians: list[float] = []
    piece_stats: dict[str, dict[str, int]] = {}
    for nom in ("greenloop", "moreau"):
        dossier = dossier_courant(nom)
        subject_id = registry.get(nom) or banc_realite_subject_id(nom)
        elements = list_projection_elements(subject_id, dsn=current.dsn)
        situations[nom] = len(elements)
        by_piece = _stats_by_piece(elements)
        piece_stats[nom] = by_piece
        cov = measure_or_coverage_mrfs(dossier, subject_id=subject_id, dsn=current.dsn)
        coverage_or[nom] = cov["or_porte_verbatim"]
        coverage_valeur[nom] = cov["or_porte_valeur"]
        medians.append(_median_proposition_length(subject_id, dsn=current.dsn))
    if pieces_couvertes is None:
        pieces_couvertes = {}
        for nom in ("greenloop", "moreau"):
            dossier = dossier_courant(nom)
            total_pieces = len(
                [path for path in dossier.corpus_dir.iterdir() if path.is_file()]
            )
            covered = len(piece_stats.get(nom) or {})
            pieces_couvertes[nom] = f"{covered}/{total_pieces}"
    payload = {
        "status": status,
        "lecteur_repare": lecteur_repare,
        "purge": purge or {"test_md": 0, "inconnue": 0},
        "appels_nouveaux": count_new_model_calls(current),
        "situations": situations,
        "pieces_couvertes": pieces_couvertes,
        "couverture_or": coverage_or,
        "couverture_valeur": coverage_valeur,
        "proposition_longueur_med": round(statistics.median(medians), 1) if medians else 0,
        "or_isole_de_l_ingestion": module_refuses_gold_access()[0],
        "tests": f"{tests_passed} verts",
    }
    current.result_path.parent.mkdir(parents=True, exist_ok=True)
    current.result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return current.result_path
