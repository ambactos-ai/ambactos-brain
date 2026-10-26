# SPDX-License-Identifier: AGPL-3.0-only
"""Balayage déterministe du budget de complétion structurelle (fine, k=8, trois tours)."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from typing import Any

import psycopg.errors

from banc_realite.compat_moteur import _render_markdown

from . import etage3 as e3
from .config import BANC_ROOT, export_runtime_env
from .fidelite import load_gold
from .holders import build_holder_table
from .measure_disputes import (
    _read_public_claims,
    _side_presence,
    build_evidence_for_mode,
)
from .measure_etage2 import _load_embed_cache

BUDGETS: tuple[int | None, ...] = (8, 12, 16, 24, 32, None)
COUCHE = "fine"
TOP_K = 8
MODE = "trois_tours_completion"
MAX_RENDER_CHARS = 8000
THRESHOLDS = (8, 20, 50, 100)

SORTIES_DIR = BANC_ROOT / "sorties" / "balayage-budget"
RESULTAT_PATH = SORTIES_DIR / "resultat.json"
TABLEAU_PATH = SORTIES_DIR / "tableau.txt"
ARRET_PATH = BANC_ROOT / "_budget-arret.json"
RESULT_PATH = BANC_ROOT / "_budget-result.json"


def write_arret(why: str) -> None:
    ARRET_PATH.write_text(
        json.dumps({"status": "arret", "pourquoi": why}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _conversation_state() -> dict[str, Any]:
    return {"responsibility_context": "GreenLoop"}


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return round(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def _count_above(values: list[int], threshold: int) -> int:
    return sum(1 for value in values if value > threshold)


def _compile_question_row(
    question: dict[str, Any],
    *,
    connection,
    embed_cache: dict[str, list[float]],
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
    budget: int | None,
) -> dict[str, Any]:
    qid = question["id"]
    question_text = question["question"]
    if question_text not in embed_cache:
        raise RuntimeError(f"cache froid : embedding absent ({question_text[:60]}…)")
    evidence, _meta = build_evidence_for_mode(
        connection,
        question,
        couche=COUCHE,
        k=TOP_K,
        mode=MODE,
        embed_cache=embed_cache,
        budget=budget,
    )
    rendered = _render_markdown(
        "banc-realite",
        "GreenLoop",
        f"banc-etage3-{qid}",
        None,
        _conversation_state(),
        evidence,
        None,
    )
    row: dict[str, Any] = {
        "evidence_count": len(evidence),
        "caracteres": len(rendered),
        "plafond_caracteres": len(rendered) >= MAX_RENDER_CHARS,
    }
    if question.get("shape") == "dispute":
        sides = _side_presence(question, evidence, gold=gold, claims_index=claims_index)
        row["deux_cotes_dans_contexte"] = all(s["dans_contexte"] for s in sides) if sides else False
    return row


def _aggregate_rows(rows: list[dict[str, Any]], dispute_only: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [int(row["evidence_count"]) for row in rows]
    chars = [int(row["caracteres"]) for row in rows]
    deux_cotes = sum(1 for row in dispute_only if row.get("deux_cotes_dans_contexte"))
    return {
        "questions": len(rows),
        "disputes_deux_cotes": deux_cotes,
        "disputes_total": len(dispute_only),
        "elements": {
            "mediane": int(statistics.median(counts)) if counts else 0,
            "p90": _percentile(counts, 90),
            "max": max(counts) if counts else 0,
            "au_dessus_8": _count_above(counts, 8),
            "au_dessus_20": _count_above(counts, 20),
            "au_dessus_50": _count_above(counts, 50),
            "au_dessus_100": _count_above(counts, 100),
        },
        "caracteres": {
            "mediane": int(statistics.median(chars)) if chars else 0,
            "p90": _percentile(chars, 90),
            "max": max(chars) if chars else 0,
            "plafond_8000": sum(1 for value in chars if value >= MAX_RENDER_CHARS),
        },
    }


def _budget_label(budget: int | None) -> str:
    return "None" if budget is None else str(budget)


def _format_tableau(results: list[dict[str, Any]]) -> str:
    header = (
        "budget | disputes 2c/28 | méd.élém | p90 élém | max élém | >8 | >20 | >50 | >100 | "
        "méd.car. | p90 car. | max car. | plaf.8k"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        budget = row["budget"]
        elements = row["elements"]
        chars = row["caracteres"]
        lines.append(
            f"{budget:>4} | "
            f"{row['disputes_deux_cotes']:>2}/{row['disputes_total']:<2} | "
            f"{elements['mediane']:>7} | {elements['p90']:>7} | {elements['max']:>7} | "
            f"{elements['au_dessus_8']:>3} | {elements['au_dessus_20']:>3} | "
            f"{elements['au_dessus_50']:>3} | {elements['au_dessus_100']:>4} | "
            f"{chars['mediane']:>7} | {chars['p90']:>7} | {chars['max']:>7} | "
            f"{chars['plafond_8000']:>6}"
        )
    return "\n".join(lines) + "\n"


def run_balayage(*, essai: bool = False) -> dict[str, Any]:
    from .infra import connect_existing, start_cluster

    export_runtime_env()
    embed_cache = _load_embed_cache()
    if not embed_cache:
        raise RuntimeError("cache froid : etage2-question-embeddings.json absent ou vide")

    gold = load_gold()
    start_cluster()
    connection = connect_existing()
    claims = _read_public_claims(connection)
    claims_index = {str(c["claim_id"]): c for c in claims}
    build_holder_table(claims, gold)

    questions = e3.load_questions()
    if essai:
        dispute_questions = [q for q in questions if q.get("shape") == "dispute"][:3]
        questions = dispute_questions

    results: list[dict[str, Any]] = []
    for budget in BUDGETS:
        rows: list[dict[str, Any]] = []
        dispute_rows: list[dict[str, Any]] = []
        for question in questions:
            for attempt in range(2):
                try:
                    row = _compile_question_row(
                        question,
                        connection=connection,
                        embed_cache=embed_cache,
                        gold=gold,
                        claims_index=claims_index,
                        budget=budget,
                    )
                    break
                except Exception as exc:
                    if attempt == 0 and isinstance(exc, psycopg.errors.AdminShutdown):
                        start_cluster()
                        connection = connect_existing()
                        continue
                    raise
            rows.append(row)
            if question.get("shape") == "dispute":
                dispute_rows.append(row)
        payload = {"budget": _budget_label(budget), **_aggregate_rows(rows, dispute_rows)}
        results.append(payload)

    SORTIES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTAT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TABLEAU_PATH.write_text(_format_tableau(results), encoding="utf-8")
    return {
        "budgets": len(results),
        "questions": len(questions),
        "files": [str(RESULTAT_PATH), str(TABLEAU_PATH)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Balayage budget complétion structurelle.")
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Limiter au premier trio de questions dispute (pilote).",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        summary = run_balayage(essai=args.essai)
    except Exception as exc:
        write_arret(str(exc))
        print(json.dumps({"status": "arret", "erreur": str(exc)}, ensure_ascii=False))
        return 1
    elapsed = round(time.monotonic() - started, 2)
    payload = {
        "status": "complete",
        "date": date.today().isoformat(),
        "essai": args.essai,
        "secondes": elapsed,
        **summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
