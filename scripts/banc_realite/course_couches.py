# SPDX-License-Identifier: AGPL-3.0-only
"""Course équitable des couches — tours, fusion et rendu sans plafond."""
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
from .balayage_budget import _aggregate_rows, _percentile
from .config import BANC_ROOT, export_runtime_env
from .controle_budget_egal import _aggregate_config_rows
from .fidelite import load_gold
from .holders import build_holder_table
from .measure_disputes import (
    _read_public_claims,
    _side_presence,
    build_evidence_for_mode,
)
from .measure_etage2 import _appui_summary, _load_embed_cache
from .partage_disputes import partager

MAX_RENDER_CHARS = 8000
RENDER_SANS_BUDGET = 64000

CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {"etiquette": "fine_tours2_k12", "couche": "fine", "k": 12, "mode": "trois_tours", "n_tours": 2, "fusion": "alternance"},
    {"etiquette": "fine_tours3_k12", "couche": "fine", "k": 12, "mode": "trois_tours", "n_tours": 3, "fusion": "alternance"},
    {"etiquette": "fine_tours4_k12", "couche": "fine", "k": 12, "mode": "trois_tours", "n_tours": 4, "fusion": "alternance"},
    {"etiquette": "fine_tours5_k12", "couche": "fine", "k": 12, "mode": "trois_tours", "n_tours": 5, "fusion": "alternance"},
    {"etiquette": "brute_temoin_k12", "couche": "brute", "k": 12, "mode": "temoin", "n_tours": None, "fusion": None},
    {"etiquette": "brute_concat_k12", "couche": "brute", "k": 12, "mode": "trois_tours", "n_tours": 3, "fusion": "concat"},
    {"etiquette": "brute_tours3_k12", "couche": "brute", "k": 12, "mode": "trois_tours", "n_tours": 3, "fusion": "alternance"},
    {"etiquette": "brute_tours5_k12", "couche": "brute", "k": 12, "mode": "trois_tours", "n_tours": 5, "fusion": "alternance"},
    {"etiquette": "brute_tours3_k20", "couche": "brute", "k": 20, "mode": "trois_tours", "n_tours": 3, "fusion": "alternance"},
    {"etiquette": "brute_fine_tours3_k12", "couche": "brute_fine", "k": 12, "mode": "trois_tours", "n_tours": 3, "fusion": "alternance"},
    {"etiquette": "brute_fine_tours3_k20", "couche": "brute_fine", "k": 20, "mode": "trois_tours", "n_tours": 3, "fusion": "alternance"},
)

SORTIES_DIR = BANC_ROOT / "sorties" / "course-couches"
RESULTAT_PATH = SORTIES_DIR / "resultat.json"
TABLEAU_PATH = SORTIES_DIR / "tableau.txt"
ARRET_PATH = BANC_ROOT / "_course-arret.json"
RESULT_PATH = BANC_ROOT / "_course-result.json"


def write_arret(why: str) -> None:
    ARRET_PATH.write_text(
        json.dumps({"status": "arret", "pourquoi": why}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _conversation_state() -> dict[str, Any]:
    return {"responsibility_context": "GreenLoop"}


def _compile_question_row(
    question: dict[str, Any],
    *,
    connection,
    embed_cache: dict[str, list[float]],
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    couche: str,
    k: int,
    mode: str,
    n_tours: int | None,
    fusion: str | None,
) -> dict[str, Any]:
    qid = question["id"]
    question_text = question["question"]
    if question_text not in embed_cache:
        raise RuntimeError(f"cache froid : embedding absent ({question_text[:60]}…)")
    evidence_kwargs: dict[str, Any] = {
        "couche": couche,
        "k": k,
        "mode": mode,
        "embed_cache": embed_cache,
        "budget": None,
    }
    if mode == "trois_tours":
        evidence_kwargs["n_tours"] = n_tours if n_tours is not None else 3
        evidence_kwargs["fusion"] = fusion if fusion is not None else "alternance"
    evidence, _meta = build_evidence_for_mode(connection, question, **evidence_kwargs)
    rendered_sans_budget = _render_markdown(
        "banc-realite",
        "GreenLoop",
        f"banc-etage3-{qid}",
        None,
        _conversation_state(),
        evidence,
        None,
        max_chars=RENDER_SANS_BUDGET,
    )
    rendered = _render_markdown(
        "banc-realite",
        "GreenLoop",
        f"banc-etage3-{qid}",
        None,
        _conversation_state(),
        evidence,
        None,
        max_chars=MAX_RENDER_CHARS,
    )
    appui = _appui_summary(
        question,
        evidence,
        assertion_index=assertion_index,
        claims_index=claims_index,
    )
    supports = question.get("supporting_assertions") or []
    tout_tient = bool(supports) and len(appui["tient"]) == len(supports)
    au_moins_une = bool(appui["tient"])
    row: dict[str, Any] = {
        "question_id": qid,
        "shape": question.get("shape") or "unknown",
        "evidence_count": len(evidence),
        "caracteres": len(rendered),
        "caracteres_sans_budget": len(rendered_sans_budget),
        "au_dessus_8000": len(rendered_sans_budget) > 8000,
        "plafond_caracteres": len(rendered) >= MAX_RENDER_CHARS,
        "appui_tout_tient": tout_tient,
        "appui_au_moins_une": au_moins_une,
        "passages_tronques": rendered_sans_budget.count(" … [tronqué :"),
    }
    if question.get("shape") == "dispute":
        sides = _side_presence(question, evidence, gold=gold, claims_index=claims_index)
        row["deux_cotes_dans_contexte"] = all(s["dans_contexte"] for s in sides) if sides else False
    return row


def _taille_block_reels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chars = [int(row["caracteres_sans_budget"]) for row in rows]
    return {
        "caracteres_reels": {
            "mediane": int(statistics.median(chars)) if chars else 0,
            "p90": _percentile(chars, 90),
            "max": max(chars) if chars else 0,
            "au_dessus_8000": sum(1 for value in chars if value > 8000),
        },
        "passages_tronques": sum(int(row.get("passages_tronques", 0)) for row in rows),
    }


def _aggregate_config_rows_course(
    rows: list[dict[str, Any]],
    dispute_rows: list[dict[str, Any]],
    *,
    reglage_ids: set[str],
    ecart_ids: set[str],
) -> dict[str, Any]:
    base = _aggregate_config_rows(
        rows,
        dispute_rows,
        reglage_ids=reglage_ids,
        ecart_ids=ecart_ids,
    )
    return {**base, "reels": _taille_block_reels(rows)}


def _format_tableau(results: list[dict[str, Any]]) -> str:
    header = (
        "étiquette | 2c/28 | réglage/14 | écart/14 | tout/141 | ≥1/141 | "
        "méd.élém. | max élém. | p90 car. | plaf.8k | car. réels p90 | >8000"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        disputes = row["disputes"]
        appui = row["appui"]
        taille = row["taille"]
        reels = row["reels"]
        elements = taille["elements"]
        chars = taille["caracteres"]
        chars_reels = reels["caracteres_reels"]
        lines.append(
            f"{row['etiquette']:<24} | "
            f"{disputes['deux_cotes_sur_28']:>2}/28 | "
            f"{disputes['reglage_sur_14']:>2}/14 | "
            f"{disputes['ecart_sur_14']:>2}/14 | "
            f"{appui['tout_tient_sur_141']:>3}/141 | "
            f"{appui['au_moins_une_sur_141']:>3}/141 | "
            f"{elements['mediane']:>7} | {elements['max']:>7} | "
            f"{chars['p90']:>7} | {chars['plafond_8000']:>6} | "
            f"{chars_reels['p90']:>14} | {chars_reels['au_dessus_8000']:>5}"
        )
    return "\n".join(lines) + "\n"


def run_course(*, essai: bool = False) -> dict[str, Any]:
    from .infra import connect_existing, start_cluster

    export_runtime_env()
    embed_cache = _load_embed_cache()
    if not embed_cache:
        raise RuntimeError("cache froid : etage2-question-embeddings.json absent ou vide")

    gold = load_gold()
    assertion_index = gold["assertion_index"]
    start_cluster()
    connection = connect_existing()
    claims = _read_public_claims(connection)
    claims_index = {str(c["claim_id"]): c for c in claims}
    build_holder_table(claims, gold)

    all_questions = e3.load_questions()
    reglage, ecart = partager(all_questions)
    reglage_ids = {q["id"] for q in reglage}
    ecart_ids = {q["id"] for q in ecart}
    partage_payload = {
        "reglage": [q["id"] for q in reglage],
        "ecart": [q["id"] for q in ecart],
    }

    questions = all_questions
    if essai:
        dispute_questions = [q for q in all_questions if q.get("shape") == "dispute"][:3]
        questions = dispute_questions

    results: list[dict[str, Any]] = []
    for config in CONFIGURATIONS:
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
                        assertion_index=assertion_index,
                        couche=config["couche"],
                        k=config["k"],
                        mode=config["mode"],
                        n_tours=config.get("n_tours"),
                        fusion=config.get("fusion"),
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

        agg = _aggregate_config_rows_course(
            rows,
            dispute_rows,
            reglage_ids=reglage_ids,
            ecart_ids=ecart_ids,
        )
        payload = {**config, **agg}
        results.append(payload)

    SORTIES_DIR.mkdir(parents=True, exist_ok=True)
    output = {"partage": partage_payload, "configurations": results}
    RESULTAT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TABLEAU_PATH.write_text(_format_tableau(results), encoding="utf-8")
    return {
        "configurations": len(results),
        "questions": len(questions),
        "files": [str(RESULTAT_PATH), str(TABLEAU_PATH)],
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Course équitable des couches.")
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Limiter aux 3 premières disputes (pilote).",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        summary = run_course(essai=args.essai)
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
        **{k: v for k, v in summary.items() if k != "results"},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
