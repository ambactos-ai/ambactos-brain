# SPDX-License-Identifier: AGPL-3.0-only
"""Contrôles d'ingestion Brain — dossier juridique MOREAU."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .dossiers import dossier_courant
from .fidelite import F1_KEY_GLOBAL, F1_KEY_PAR_VOIE, F1_KEY_PROSE, F1_KEY_TABULAIRE, measure
from .holders import _claim_as_element, _prepare_claims, classify_pair
from .ingest_runner import IngestionConfig, backfill_embeddings, run_ingestion
from .memory import read_active_claims, read_public_active_claims
from .normalize import piece_basename
from .or_dossier import load_gold
from .relay import load_call_count

MOREAU_DOSSIER = dossier_courant("moreau")
GREENLOOP_DOSSIER = dossier_courant("greenloop")


def measure_or_coverage(connection, *, dossier: str = "moreau") -> dict[str, Any]:
    """Couverture des citations gold par les atomes extraits (verbatim et valeur)."""
    current = dossier_courant(dossier)
    gold = load_gold(current)
    claims = read_public_active_claims(connection, vault_ref=current.vault_ref)
    _, claims_index = _prepare_claims(claims)
    assertions = gold["assertions"]
    total = len(assertions)
    verbatim_hits = 0
    valeur_hits = 0
    manquantes: list[dict[str, str]] = []

    for assertion in assertions:
        piece = piece_basename(assertion.get("source") or assertion.get("piece") or "")
        piece_claims = [c for c in claims if c.get("piece") == piece]
        tient_verbatim = any(
            classify_pair(assertion, _claim_as_element(claim), claims_index, mode_presence="verbatim")
            == "tient"
            for claim in piece_claims
        )
        tient_valeur = any(
            classify_pair(assertion, _claim_as_element(claim), claims_index, mode_presence="valeur")
            == "tient"
            for claim in piece_claims
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
    }


def verify_isolation(connection) -> dict[str, Any]:
    """Vérifie l'isolation des aires MOREAU / GreenLoop dans brain.claims."""
    moreau_claims = read_active_claims(connection, vault_ref=MOREAU_DOSSIER.vault_ref)
    greenloop_claims = read_active_claims(connection, vault_ref=GREENLOOP_DOSSIER.vault_ref)
    moreau_ids = {str(c["claim_id"]) for c in moreau_claims}
    greenloop_ids = {str(c["claim_id"]) for c in greenloop_claims}
    overlap = moreau_ids & greenloop_ids
    greenloop_area = GREENLOOP_DOSSIER.area_root
    moreau_area = MOREAU_DOSSIER.area_root
    greenloop_semantic = (greenloop_area / "semantic-memory").is_dir()
    moreau_semantic = (moreau_area / "semantic-memory").is_dir()
    cross_vault = bool(
        connection.execute(
            "SELECT 1 FROM brain.claims WHERE lifecycle_status = 'active' "
            "AND vault_ref = %s AND source_file_path LIKE %s LIMIT 1",
            (MOREAU_DOSSIER.vault_ref, "%greenloop%"),
        ).fetchone()
        or connection.execute(
            "SELECT 1 FROM brain.claims WHERE lifecycle_status = 'active' "
            "AND vault_ref = %s AND source_file_path LIKE %s LIMIT 1",
            (GREENLOOP_DOSSIER.vault_ref, "%moreau%"),
        ).fetchone()
    )
    ok = (
        not overlap
        and not cross_vault
        and greenloop_semantic
        and moreau_semantic
        and moreau_area.resolve() != greenloop_area.resolve()
    )
    return {
        "isolation_ok": ok,
        "overlap_claim_ids": sorted(overlap),
        "moreau_atom_count": len(moreau_claims),
        "greenloop_atom_count": len(greenloop_claims),
        "greenloop_semantic_memory": greenloop_semantic,
    }


def count_completed_pieces(cfg: IngestionConfig | None = None) -> int:
    cfg = cfg or IngestionConfig.for_moreau_brain()
    if not cfg.ingest_log.is_file():
        return 0
    done: set[str] = set()
    for line in cfg.ingest_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        file_name = record.get("file")
        if not file_name:
            continue
        if record.get("skipped") or record.get("statut") == "complete":
            done.add(file_name)
    return len(done)


def finalize_moreau_controls(connection) -> dict[str, Any]:
    """Contrôles post-ingestion (fidélité, couverture or, isolation) sans ré-ingérer."""
    cfg = IngestionConfig.for_moreau_brain()
    calls = load_call_count(cfg.call_log)
    moreau_claims = read_active_claims(connection, vault_ref=MOREAU_DOSSIER.vault_ref)
    fidelite = measure(output_dir=cfg.output_dir, dossier="moreau")
    coverage = measure_or_coverage(connection, dossier="moreau")
    isolation = verify_isolation(connection)
    ingestion = {
        "ingestion": {"arret": "plafond_appels" if calls >= cfg.max_calls else None},
        "appels_modele": calls,
        "fournisseur": _primary_provider(cfg.call_log),
        "embeddings": _count_embeddings(cfg.output_dir),
        "atomes": len(moreau_claims),
        "pieces_traitees": count_completed_pieces(cfg),
    }
    write_rapport_ingestion_brain(
        output_dir=cfg.output_dir,
        ingestion=ingestion,
        fidelite=fidelite,
        coverage=coverage,
        isolation=isolation,
    )
    return {
        "ingestion": ingestion,
        "fidelite": fidelite,
        "coverage": coverage,
        "isolation": isolation,
        "scores": fidelite_scores(fidelite),
    }


def _primary_provider(call_log: Path) -> str | None:
    if not call_log.is_file():
        return None
    for line in reversed(call_log.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "end" and record.get("executor_provider"):
            return str(record["executor_provider"])
    return None


def _count_embeddings(output_dir: Path) -> int:
    path = output_dir / "embedding-appels.jsonl"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run_moreau_brain_ingestion(connection) -> dict[str, Any]:
    """Ingère les 12 pièces MOREAU par le chemin ordinaire (ingest_runner)."""
    cfg = IngestionConfig.for_moreau_brain()
    calls_before = load_call_count(cfg.call_log)
    summary = run_ingestion(config=cfg)
    calls = load_call_count(cfg.call_log) - calls_before
    embed_stats = backfill_embeddings(
        max_embeddings=cfg.max_embeddings,
        log_path=cfg.output_dir / "embedding-appels.jsonl",
        vault_ref=MOREAU_DOSSIER.vault_ref,
    )
    moreau_claims = read_active_claims(connection, vault_ref=MOREAU_DOSSIER.vault_ref)
    return {
        "ingestion": summary,
        "appels_modele": calls,
        "fournisseur": summary.get("fournisseur"),
        "embeddings": embed_stats.get("embedded", 0),
        "atomes": len(moreau_claims),
        "pieces_traitees": len(
            [p for p in summary.get("pieces", []) if not p.get("skipped") and p.get("statut") != "erreur"]
        ),
    }


def write_rapport_ingestion_brain(
    *,
    output_dir: Path,
    ingestion: dict[str, Any],
    fidelite: dict[str, Any],
    coverage: dict[str, Any],
    isolation: dict[str, Any],
) -> Path:
    f1 = fidelite.get("F1", {})
    par_voie = f1.get(F1_KEY_PAR_VOIE, {})
    lines = [
        "# Rapport — ingestion Brain MOREAU",
        "",
        f"Date : {date.today().isoformat()}",
        "",
        "## Ingestion",
        f"- Pièces : {ingestion.get('pieces_traitees', 0)}/12",
        f"- Atomes : {ingestion.get('atomes', 0)}",
        f"- Appels modèle : {ingestion.get('appels_modele', 0)} (plafond {IngestionConfig.for_moreau_brain().max_calls})",
        f"- Fournisseur : {ingestion.get('fournisseur')}",
        f"- Embeddings : {ingestion.get('embeddings', 0)}",
        "",
        "## 1. Fidélité",
        "",
        f"- F1 global : {f1.get(F1_KEY_GLOBAL)}",
        f"- F1 prose : {par_voie.get(F1_KEY_PROSE)}",
        f"- F1 tabulaire : {par_voie.get(F1_KEY_TABULAIRE)}",
        "",
        "## 2. Couverture de l'or (classify_pair)",
        "",
        f"- Verbatim : {coverage.get('or_porte_verbatim')}",
        f"- Valeur : {coverage.get('or_porte_valeur')}",
        "",
    ]
    manquantes = coverage.get("manquantes") or []
    if manquantes:
        lines.append("### Citations non portées")
        lines.append("")
        lines.append("| id | pièce | cause |")
        lines.append("| --- | --- | --- |")
        for row in manquantes:
            lines.append(f"| {row['id']} | {row['piece']} | {row['cause']} |")
        lines.append("")
    lines.extend(
        [
            "## 3. Isolation",
            "",
            f"- OK : {isolation.get('isolation_ok')}",
            f"- Atomes MOREAU : {isolation.get('moreau_atom_count')}",
            f"- Atomes GreenLoop : {isolation.get('greenloop_atom_count')}",
            f"- GreenLoop semantic-memory : {isolation.get('greenloop_semantic_memory')}",
        ]
    )
    if ingestion.get("ingestion", {}).get("arret") == "plafond_appels":
        lines.extend(
            [
                "",
                "## Arrêt",
                "",
                "Plafond d'appels modèle atteint — reprise via `reprise.jsonl`.",
            ]
        )
    report = output_dir / "RAPPORT-INGESTION-BRAIN.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def fidelite_scores(fidelite: dict[str, Any]) -> dict[str, float | None]:
    f1 = fidelite.get("F1", {})
    par_voie = f1.get(F1_KEY_PAR_VOIE, {})
    return {
        "f1_global": f1.get(F1_KEY_GLOBAL),
        "f1_prose": par_voie.get(F1_KEY_PROSE),
        "f1_tabulaire": par_voie.get(F1_KEY_TABULAIRE),
    }
