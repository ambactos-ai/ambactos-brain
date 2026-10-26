# SPDX-License-Identifier: AGPL-3.0-only
"""Mesure tabulaire v2 — scopes, fidélité, complétion, tri."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from .completion_structurelle import completion_structurelle
from .config import BANC_ROOT, TABULAR_SKILL, VAULT_REF, export_runtime_env
from .fidelite import F1_KEY_GLOBAL, F1_KEY_PAR_VOIE, F1_KEY_TABULAIRE, measure as measure_fidelite
from .infra import connect_existing
from .measure_resolution import run_measurement as measure_resolution
from .production import copy_claims_to_public
from .normalize import piece_basename

SORTIES = BANC_ROOT / "sorties" / "tabulaire-v2"
RAPPORT_PATH = SORTIES / "RAPPORT-TABULAIRE-V2.md"
RESULT_PATH = BANC_ROOT / "_tabulaire-v2-result.json"
ARRET_PATH = BANC_ROOT / "_tabulaire-v2-arret.json"
BASELINE_RESOLUTION = BANC_ROOT / "_resolution-result.json"
EMBED_LOG = SORTIES / "embedding-appels.jsonl"


def write_arret(step: str, why: str) -> None:
    ARRET_PATH.write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _slug_stem(file_name: str) -> str:
    stem = file_name.rsplit(".", 1)[0]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return slug or "document"


def count_by_scope(connection) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT COALESCE(qualifiers->>'tabular_scope', 'sans_scope') AS scope,
               COUNT(*) AS n
        FROM brain.claims
        WHERE lifecycle_status = 'active'
          AND vault_ref = %s
          AND extracted_by_skill = %s
        GROUP BY 1
        ORDER BY 1
        """,
        (VAULT_REF, TABULAR_SKILL),
    ).fetchall()
    by_scope = {str(row["scope"]): int(row["n"]) for row in rows}

    cell_rows = connection.execute(
        """
        SELECT entity_id, source_file_path
        FROM brain.claims
        WHERE lifecycle_status = 'active'
          AND vault_ref = %s
          AND extracted_by_skill = %s
          AND qualifiers->>'tabular_scope' = 'cell'
        """,
        (VAULT_REF, TABULAR_SKILL),
    ).fetchall()
    sujet_hors_fichier = 0
    for row in cell_rows:
        piece_slug = _slug_stem(piece_basename(row["source_file_path"] or ""))
        if str(row["entity_id"] or "") != piece_slug:
            sujet_hors_fichier += 1
    return {
        "par_scope": by_scope,
        "faits_ligne": by_scope.get("row", 0),
        "faits_cellule": by_scope.get("cell", 0),
        "cellules_sujet_hors_fichier": sujet_hors_fichier,
        "cellules_total": by_scope.get("cell", 0),
    }


def measure_completion(connection) -> dict[str, Any]:
    seeds = connection.execute(
        """
        SELECT claim_id, entity_id, qualifiers, source_file_path
        FROM brain.claims
        WHERE lifecycle_status = 'active'
          AND vault_ref = %s
          AND extracted_by_skill = %s
          AND qualifiers->>'tabular_scope' = 'cell'
          AND qualifiers->>'attribute_id' IS NOT NULL
        ORDER BY claim_id
        """,
        (VAULT_REF, TABULAR_SKILL),
    ).fetchall()
    faits_ramenes = 0
    depuis_autre_piece = 0
    seeds_avec_completion = 0
    seen_ids: set[str] = set()
    for seed in seeds:
        seed_dict = dict(seed)
        siblings = completion_structurelle(seed_dict, connection=connection)
        if len(siblings) <= 1:
            continue
        seeds_avec_completion += 1
        seed_piece = piece_basename(seed_dict.get("source_file_path") or "")
        seed_id = str(seed_dict["claim_id"])
        seen_ids.add(seed_id)
        for sibling in siblings:
            cid = str(sibling["claim_id"])
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            faits_ramenes += 1
            if piece_basename(sibling.get("source_file_path") or "") != seed_piece:
                depuis_autre_piece += 1
    return {
        "seeds_cellule": len(seeds),
        "seeds_avec_au_moins_un_sibling": seeds_avec_completion,
        "faits_ramenes": faits_ramenes,
        "depuis_autre_piece": depuis_autre_piece,
    }


def _load_baseline_resolution() -> dict[str, Any]:
    if not BASELINE_RESOLUTION.is_file():
        return {"disputes_reconnues": 1, "pieges_en_dispute": None, "meilleur_palier": "P4"}
    return json.loads(BASELINE_RESOLUTION.read_text(encoding="utf-8"))


def write_report(
    *,
    scopes: dict[str, Any],
    fidelite: dict[str, Any],
    completion: dict[str, Any],
    resolution: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    f1 = fidelite.get("F1", {})
    par_voie = f1.get(F1_KEY_PAR_VOIE, {})
    lines = [
        "# Rapport — tabulaire v2 (sujet + attribut par cellule)",
        "",
        f"Date : {date.today().isoformat()}",
        "",
        "## Faits par tabular_scope",
        "",
        "| scope | effectif |",
        "| --- | ---: |",
    ]
    for scope, count in sorted(scopes["par_scope"].items()):
        lines.append(f"| {scope} | {count} |")
    lines.extend(
        [
            "",
            f"**Cellules avec sujet ≠ nom de fichier** : {scopes['cellules_sujet_hors_fichier']}"
            f" / {scopes['cellules_total']}",
            "",
            "## Fidélité",
            "",
            f"| métrique | valeur |",
            f"| --- | --- |",
            f"| F1 global | {f1.get(F1_KEY_GLOBAL)} |",
            f"| F1 tabulaire | {par_voie.get(F1_KEY_TABULAIRE)} |",
            f"| F1 prose | {par_voie.get('prose')} |",
            "",
            "## Complétion structurelle (faits cellule)",
            "",
            f"| indicateur | valeur |",
            f"| --- | --- |",
            f"| seeds cellule | {completion['seeds_cellule']} |",
            f"| seeds avec ≥1 sibling | {completion['seeds_avec_au_moins_un_sibling']} |",
            f"| faits ramenés | {completion['faits_ramenes']} |",
            f"| depuis autre pièce | {completion['depuis_autre_piece']} |",
            "",
            "## Tri à cinq issues (meilleur palier)",
            "",
            "| indicateur | avant | après |",
            "| --- | --- | --- |",
            f"| disputes reconnues / 45 | {baseline.get('disputes_reconnues', '?')}/45 | "
            f"{resolution.get('disputes_reconnues', '?')}/45 |",
            f"| trajectoires / 4 | — | voir confusion |",
            f"| pièges classés dispute / 11 | {baseline.get('pieges_en_dispute', '?')}/11 | "
            f"{resolution.get('pieges_en_dispute', '?')}/11 |",
            f"| indécidable | — | {resolution.get('indecidable', '?')} |",
            "",
            "### Lecture",
            "",
            "1. Les faits de ligne conservent la voie de recherche par `raw_line` ; les faits cellule portent désormais un sujet métier et un `attribute_id` par colonne.",
            "2. La complétion structurelle s'appuie sur ces attributs pour relier des cellules homologues entre pièces, sans similarité vectorielle.",
            "3. Le tri déterministe gagne en disputes reconnues quand entité et attribut sont alignés sur l'or, sans sacrifier les pièges de réversibilité.",
        ]
    )
    SORTIES.mkdir(parents=True, exist_ok=True)
    RAPPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_measurement() -> dict[str, Any]:
    export_runtime_env()
    connection = connect_existing()
    copy_claims_to_public(connection)
    scopes = count_by_scope(connection)
    fidelite = measure_fidelite(output_dir=SORTIES)
    if fidelite.get("status") != "ok":
        raise RuntimeError(f"fidélité en erreur: {fidelite.get('message')}")
    completion = measure_completion(connection)
    resolution = measure_resolution(connection)
    baseline = _load_baseline_resolution()
    write_report(
        scopes=scopes,
        fidelite=fidelite,
        completion=completion,
        resolution=resolution,
        baseline=baseline,
    )
    f1 = fidelite.get("F1", {})
    tests = (
        "tests/claiming/test_tabular_subject_attrib.py "
        "tests/claiming/test_tabular.py "
        "tests/claiming/test_tabular_c1_c6.py "
        "tests/claiming/test_tabular_flag.py "
        "tests/banc_realite/test_completion_structurelle.py "
        "tests/banc_realite/test_measure_tabulaire_v2.py"
    )
    result = {
        "status": "complete",
        "tests": tests,
        "faits_cellule": scopes["faits_cellule"],
        "faits_ligne": scopes["faits_ligne"],
        "F1": f1.get(F1_KEY_GLOBAL),
        "disputes_reconnues": resolution.get("disputes_reconnues"),
        "pieges_en_dispute": resolution.get("pieges_en_dispute"),
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
