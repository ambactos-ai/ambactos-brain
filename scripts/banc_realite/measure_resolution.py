# SPDX-License-Identifier: AGPL-3.0-only
"""Mesure du tri à cinq issues, palier par palier, contre l'or."""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from .classer_couple import classer_couple
from .config import BANC_ROOT, export_runtime_env
from .fidelite import load_gold, match_assertion
from .normalize import claim_text, piece_basename
from .resolution import (
    PALIERS,
    SORTIES_RESOLUTION,
    build_inventaire_from_claims,
    build_resolution_context,
    read_claims_from_db,
    write_inventaire,
)

CONFUSION_PATH = SORTIES_RESOLUTION / "confusion-par-palier.json"
FAUX_RAPPROCHEMENTS_PATH = SORTIES_RESOLUTION / "faux-rapprochements.json"
RAPPORT_PATH = SORTIES_RESOLUTION / "RAPPORT-RESOLUTION.md"
RESULT_PATH = BANC_ROOT / "_resolution-result.json"
ARRET_PATH = BANC_ROOT / "_resolution-arret.json"


def write_arret(step: str, why: str) -> None:
    ARRET_PATH.write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _claims_by_piece(claims: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_piece: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        piece = piece_basename(claim.get("source_file_path") or "")
        by_piece.setdefault(piece, []).append(claim)
    return by_piece


def _claim_for_assertion(assertion: dict[str, Any], by_piece: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    piece = piece_basename(assertion.get("source") or "")
    hit, _ = match_assertion(assertion, by_piece.get(piece, []))
    return hit


def _enrich_from_version(claim: dict[str, Any], version: dict[str, Any]) -> dict[str, Any]:
    out = dict(claim)
    quals = dict(out.get("qualifiers") or {})
    if version.get("validity_start"):
        quals.setdefault("since", version["validity_start"])
    if version.get("validity_end"):
        quals.setdefault("valid_to", version["validity_end"])
    out["qualifiers"] = quals
    return out


def _pair_from_dispute(
    dispute: dict[str, Any],
    gold: dict[str, Any],
    by_piece: dict[str, list[dict[str, Any]]],
    *,
    palier: str,
    ctx,
) -> dict[str, Any] | None:
    claims: list[dict[str, Any]] = []
    for side in dispute.get("sides", []):
        assertion = gold["assertion_index"].get(side.get("assertion") or "")
        if not assertion:
            continue
        hit = _claim_for_assertion(assertion, by_piece)
        if hit:
            claims.append(hit)
    if len(claims) < 2:
        return None
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            result = classer_couple(claims[i], claims[j], palier=palier, resolution_ctx=ctx)
            if result["issue"] != "accord":
                result["entity_a"] = claims[i].get("entity_id")
                result["entity_b"] = claims[j].get("entity_id")
                result["attribute_a"] = (claims[i].get("qualifiers") or {}).get("attribute_id")
                result["attribute_b"] = (claims[j].get("qualifiers") or {}).get("attribute_id")
                return result
    result = classer_couple(claims[0], claims[1], palier=palier, resolution_ctx=ctx)
    result["entity_a"] = claims[0].get("entity_id")
    result["entity_b"] = claims[1].get("entity_id")
    result["attribute_a"] = (claims[0].get("qualifiers") or {}).get("attribute_id")
    result["attribute_b"] = (claims[1].get("qualifiers") or {}).get("attribute_id")
    return result


def _pair_from_trajectory(
    trajectory: dict[str, Any],
    gold: dict[str, Any],
    by_piece: dict[str, list[dict[str, Any]]],
    *,
    palier: str,
    ctx,
) -> dict[str, Any] | None:
    versions = trajectory.get("versions") or []
    if len(versions) < 2:
        return None
    left_v, right_v = versions[0], versions[1]
    left_a = gold["assertion_index"].get(left_v.get("assertion") or "")
    right_a = gold["assertion_index"].get(right_v.get("assertion") or "")
    if not left_a or not right_a:
        return None
    left_c = _claim_for_assertion(left_a, by_piece)
    right_c = _claim_for_assertion(right_a, by_piece)
    if not left_c or not right_c:
        return None
    result = classer_couple(
        _enrich_from_version(left_c, left_v),
        _enrich_from_version(right_c, right_v),
        palier=palier,
        resolution_ctx=ctx,
    )
    result["entity_a"] = left_c.get("entity_id")
    result["entity_b"] = right_c.get("entity_id")
    result["attribute_a"] = (left_c.get("qualifiers") or {}).get("attribute_id")
    result["attribute_b"] = (right_c.get("qualifiers") or {}).get("attribute_id")
    return result


def _pair_from_reversibility(
    case: dict[str, Any],
    gold: dict[str, Any],
    by_piece: dict[str, list[dict[str, Any]]],
    *,
    palier: str,
    ctx,
) -> dict[str, Any] | None:
    situations = case.get("situations") or []
    if len(situations) < 2:
        return None
    left_sit = next((s for s in gold["situations"] if s["id"] == situations[0]), None)
    right_sit = next((s for s in gold["situations"] if s["id"] == situations[1]), None)
    if not left_sit or not right_sit:
        return None
    left_a = (left_sit.get("assertions") or [None])[0]
    right_a = (right_sit.get("assertions") or [None])[0]
    if not left_a or not right_a:
        return None
    left_c = _claim_for_assertion(left_a, by_piece)
    right_c = _claim_for_assertion(right_a, by_piece)
    if not left_c or not right_c:
        return None
    result = classer_couple(left_c, right_c, palier=palier, resolution_ctx=ctx)
    result["entity_a"] = left_c.get("entity_id")
    result["entity_b"] = right_c.get("entity_id")
    result["attribute_a"] = (left_c.get("qualifiers") or {}).get("attribute_id")
    result["attribute_b"] = (right_c.get("qualifiers") or {}).get("attribute_id")
    return result


def build_confusion_for_palier(
    gold: dict[str, Any],
    by_piece: dict[str, list[dict[str, Any]]],
    *,
    palier: str,
    ctx,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    matrix: dict[str, Counter] = {
        "dispute": Counter(),
        "trajectoire": Counter(),
        "reversibility": Counter(),
    }
    indecidable = 0
    total = 0

    for dispute in gold.get("disputes", []):
        result = _pair_from_dispute(dispute, gold, by_piece, palier=palier, ctx=ctx)
        issue = result["issue"] if result else "indecidable"
        rows.append(
            {
                "id": dispute["id"],
                "gold": "dispute",
                "issue": issue,
                "raison": (result or {}).get("raison"),
                "entity_a": (result or {}).get("entity_a"),
                "entity_b": (result or {}).get("entity_b"),
                "attribute_a": (result or {}).get("attribute_a"),
                "attribute_b": (result or {}).get("attribute_b"),
            }
        )
        matrix["dispute"][issue] += 1
        total += 1
        if issue == "indecidable":
            indecidable += 1

    for trajectory in gold.get("trajectories", []):
        result = _pair_from_trajectory(trajectory, gold, by_piece, palier=palier, ctx=ctx)
        issue = result["issue"] if result else "indecidable"
        rows.append(
            {
                "id": trajectory["id"],
                "gold": "trajectoire",
                "issue": issue,
                "raison": (result or {}).get("raison"),
                "entity_a": (result or {}).get("entity_a"),
                "entity_b": (result or {}).get("entity_b"),
                "attribute_a": (result or {}).get("attribute_a"),
                "attribute_b": (result or {}).get("attribute_b"),
            }
        )
        matrix["trajectoire"][issue] += 1
        total += 1
        if issue == "indecidable":
            indecidable += 1

    for case in gold.get("reversibility_cases", []):
        result = _pair_from_reversibility(case, gold, by_piece, palier=palier, ctx=ctx)
        issue = result["issue"] if result else "indecidable"
        rows.append(
            {
                "id": case["id"],
                "gold": "reversibility",
                "issue": issue,
                "raison": (result or {}).get("raison"),
                "entity_a": (result or {}).get("entity_a"),
                "entity_b": (result or {}).get("entity_b"),
                "attribute_a": (result or {}).get("attribute_a"),
                "attribute_b": (result or {}).get("attribute_b"),
            }
        )
        matrix["reversibility"][issue] += 1
        total += 1
        if issue == "indecidable":
            indecidable += 1

    pieges = sum(1 for row in rows if row["gold"] == "reversibility" and row["issue"] == "dispute")
    disputes_reconnues = sum(1 for row in rows if row["gold"] == "dispute" and row["issue"] == "dispute")
    trajectoires = sum(1 for row in rows if row["gold"] == "trajectoire" and row["issue"] == "trajectoire")
    entites_rapprochees_accord = sum(
        1
        for row in rows
        if row["gold"] == "dispute" and row["issue"] == "accord"
    )

    return {
        "palier": palier,
        "details": rows,
        "matrice": {key: dict(counter) for key, counter in matrix.items()},
        "disputes_reconnues": disputes_reconnues,
        "trajectoires": trajectoires,
        "pieges_en_dispute": pieges,
        "entites_rapprochees_accord": entites_rapprochees_accord,
        "indecidable": indecidable,
        "part_indecidable": round(indecidable / total, 4) if total else 0.0,
        "total": total,
    }


def summarize_paliers(table: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for palier in [p for p in PALIERS if p in table]:
        block = table[palier]
        rows.append(
            {
                "palier": palier,
                "disputes_reconnues": f"{block['disputes_reconnues']}/45",
                "trajectoires": f"{block['trajectoires']}/4",
                "pieges_en_dispute": f"{block['pieges_en_dispute']}/11",
                "indecidable": block["indecidable"],
            }
        )
    return rows


def choose_best_palier(table: dict[str, dict[str, Any]]) -> str:
    baseline = table.get("P0", next(iter(table.values())))

    def score(palier: str) -> tuple[int, int, int, int]:
        block = table[palier]
        return (
            -block["pieges_en_dispute"],
            block["disputes_reconnues"],
            block["trajectoires"],
            -block["indecidable"],
        )

    candidates = [
        palier
        for palier in PALIERS
        if palier in table and table[palier]["pieges_en_dispute"] <= baseline["pieges_en_dispute"]
    ]
    if not candidates:
        candidates = [palier for palier in PALIERS if palier in table]
    return max(candidates, key=score)


def detect_false_merges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    false_merges: list[dict[str, Any]] = []
    for row in rows:
        if row.get("gold") in {"reversibility", "trajectoire"} and row.get("issue") == "dispute":
            false_merges.append(row)
    return false_merges


def _unresolved_dispute_examples(details: list[dict[str, Any]], limit: int = 3) -> list[dict[str, str]]:
    examples = []
    for row in details:
        if row["gold"] != "dispute" or row["issue"] == "dispute":
            continue
        examples.append(
            {
                "id": row["id"],
                "issue": row["issue"],
                "raison": row.get("raison") or "non classé",
                "entity_a": row.get("entity_a"),
                "entity_b": row.get("entity_b"),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def write_report(
    table: dict[str, dict[str, Any]],
    *,
    meilleur_palier: str,
    false_merges: list[dict[str, Any]],
    piege_inflation: dict[str, bool],
) -> None:
    lines = [
        "# Rapport — résolution d'entité et d'attribut",
        "",
        f"Date : {date.today().isoformat()}",
        "",
        "## Confusion par palier",
        "",
        "| palier | disputes reconnues / 45 | trajectoires / 4 | pièges classés dispute / 11 | indécidable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summarize_paliers(table):
        lines.append(
            f"| {row['palier']} | {row['disputes_reconnues']} | {row['trajectoires']} | "
            f"{row['pieges_en_dispute']} | {row['indecidable']} |"
        )
    for palier, inflated in piege_inflation.items():
        if inflated:
            lines.extend(
                [
                    "",
                    f"**Attention {palier}** : les disputes montent en transformant les pièges "
                    f"({table[palier]['pieges_en_dispute']}/11) — ce palier n'a pas résolu le problème, "
                    "il a fusionné.",
                ]
            )
    lines.extend(
        [
            "",
            f"**Meilleur palier** : {meilleur_palier}",
            "",
            "## Faux rapprochements (meilleur palier)",
            "",
        ]
    )
    if false_merges:
        for row in false_merges:
            lines.append(
                f"- `{row['id']}` : `{row.get('entity_a')}` + `{row.get('attribute_a')}` "
                f"↔ `{row.get('entity_b')}` + `{row.get('attribute_b')}` → {row['issue']}"
            )
    else:
        lines.append("- Aucun faux rapprochement détecté sur trajectoires et pièges.")
    unresolved = _unresolved_dispute_examples(table[meilleur_palier]["details"])
    if unresolved:
        lines.extend(["", "## Disputes non reconnues (exemples)", ""])
        for item in unresolved:
            lines.append(
                f"- `{item['id']}` ({item['issue']} — {item['raison']}) : "
                f"`{item['entity_a']}` vs `{item['entity_b']}`"
            )
    best = table[meilleur_palier]
    if best.get("entites_rapprochees_accord"):
        lines.extend(
            [
                "",
                "## Effet trompeur P4",
                "",
                (
                    f"{best['entites_rapprochees_accord']} disputes passent de « faits distincts » à "
                    f"« accord » après rapprochement d'entité : les en-têtes de pièce concordent, "
                    "mais les valeurs litigieuses de l'or ne sont pas dans les claims appariés "
                    "(lignes tabulaires sans `attribute_id`)."
                ),
                "",
                "## Ce qui manque encore (hors paliers)",
                "",
                "- Appariement assertion → ligne tabulaire (pas seulement l'en-tête de bloc).",
                "- `attribute_id` sur les faits tabulaires (mois/site) pour comparer les bonnes cellules.",
                "- Résolution d'entité par sujet métier (MRFS) quand le client est implicite dans la structure.",
            ]
        )
    RAPPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_measurement(connection) -> dict[str, Any]:
    export_runtime_env()
    inventaire = write_inventaire(connection)
    claims = read_claims_from_db(connection)
    ctx = build_resolution_context(inventaire, claims)
    gold = load_gold()
    by_piece = _claims_by_piece(claims)

    table: dict[str, dict[str, Any]] = {}
    for palier in PALIERS:
        table[palier] = build_confusion_for_palier(gold, by_piece, palier=palier, ctx=ctx)

    meilleur = choose_best_palier(table)
    false_merges = detect_false_merges(table[meilleur]["details"])
    piege_inflation = {
        palier: table[palier]["pieges_en_dispute"] > table["P0"]["pieges_en_dispute"]
        and table[palier]["disputes_reconnues"] > table["P0"]["disputes_reconnues"]
        for palier in PALIERS
    }

    payload = {
        "paliers": table,
        "resume": summarize_paliers(table),
        "meilleur_palier": meilleur,
        "faux_rapprochements": false_merges,
        "piege_inflation": piege_inflation,
    }
    SORTIES_RESOLUTION.mkdir(parents=True, exist_ok=True)
    CONFUSION_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    FAUX_RAPPROCHEMENTS_PATH.write_text(
        json.dumps(false_merges, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(table, meilleur_palier=meilleur, false_merges=false_merges, piege_inflation=piege_inflation)

    tests = (
        "tests/banc_realite/test_resolution.py "
        "tests/banc_realite/test_classer_couple.py "
        "tests/banc_realite/test_measure_resolution.py"
    )
    result = {
        "status": "complete",
        "tests": tests,
        "meilleur_palier": meilleur,
        "disputes_reconnues": table[meilleur]["disputes_reconnues"],
        "pieges_en_dispute": table[meilleur]["pieges_en_dispute"],
        "indecidable": table[meilleur]["indecidable"],
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
