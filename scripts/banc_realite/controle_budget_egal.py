# SPDX-License-Identifier: AGPL-3.0-only
"""Contrôle à budget d'éléments égal — fine, neuf configurations, partage disputes figé."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from typing import Any

import psycopg.errors

from banc_realite.compat_moteur import _render_markdown, _render_markdown_with_budget

from . import etage3 as e3
from .balayage_budget import _aggregate_rows, _percentile
from .config import BANC_ROOT, export_runtime_env
from .fidelite import load_gold
from .holders import build_holder_table
from .measure_disputes import (
    _read_public_claims,
    _side_presence,
    build_evidence_for_mode,
)
from .measure_etage2 import _appui_summary, _load_embed_cache
from .partage_disputes import partager

COUCHE = "fine"
MAX_RENDER_CHARS = 8000

CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {"etiquette": "temoin_k8", "k": 8, "mode": "temoin", "budget": None},
    {"etiquette": "temoin_k16", "k": 16, "mode": "temoin", "budget": None},
    {"etiquette": "tours_k8", "k": 8, "mode": "trois_tours", "budget": None},
    {"etiquette": "tours_k12", "k": 12, "mode": "trois_tours", "budget": None},
    {"etiquette": "tours_k16", "k": 16, "mode": "trois_tours", "budget": None},
    {"etiquette": "tours_k20", "k": 20, "mode": "trois_tours", "budget": None},
    {"etiquette": "tours_k24", "k": 24, "mode": "trois_tours", "budget": None},
    {"etiquette": "gagnante_k8_b16", "k": 8, "mode": "trois_tours_completion", "budget": 16},
    {"etiquette": "k12_b16", "k": 12, "mode": "trois_tours_completion", "budget": 16},
)

SORTIES_DIR = BANC_ROOT / "sorties" / "controle-budget-egal"
RESULTAT_PATH = SORTIES_DIR / "resultat.json"
TABLEAU_PATH = SORTIES_DIR / "tableau.txt"
ARRET_PATH = BANC_ROOT / "_controle-budget-arret.json"
RESULT_PATH = BANC_ROOT / "_controle-budget-result.json"


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
    k: int,
    mode: str,
    budget: int | None,
    n_tours: int = 3,
    fusion: str = "alternance",
    max_chars: int = MAX_RENDER_CHARS,
    presence_sur_rendu: bool = False,
) -> dict[str, Any]:
    qid = question["id"]
    question_text = question["question"]
    if question_text not in embed_cache:
        raise RuntimeError(f"cache froid : embedding absent ({question_text[:60]}…)")
    evidence, _meta = build_evidence_for_mode(
        connection,
        question,
        couche=COUCHE,
        k=k,
        mode=mode,
        embed_cache=embed_cache,
        budget=budget,
        n_tours=n_tours,
        fusion=fusion,
    )
    if presence_sur_rendu:
        rendered, nb_ecartes = _render_markdown_with_budget(
            "banc-realite",
            "GreenLoop",
            f"banc-etage3-{qid}",
            None,
            _conversation_state(),
            evidence,
            None,
            max_chars=max_chars,
        )
        evidence_presence = evidence[: len(evidence) - nb_ecartes]
    else:
        rendered = _render_markdown(
            "banc-realite",
            "GreenLoop",
            f"banc-etage3-{qid}",
            None,
            _conversation_state(),
            evidence,
            None,
            max_chars=max_chars,
        )
        evidence_presence = evidence
        nb_ecartes = 0
    rendered_sans_budget = _render_markdown(
        "banc-realite",
        "GreenLoop",
        f"banc-etage3-{qid}",
        None,
        _conversation_state(),
        evidence,
        None,
        max_chars=64000,
    )
    appui = _appui_summary(
        question,
        evidence_presence,
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
    }
    if question.get("shape") == "dispute":
        sides = _side_presence(
            question,
            evidence_presence,
            gold=gold,
            claims_index=claims_index,
        )
        row["deux_cotes_dans_contexte"] = all(s["dans_contexte"] for s in sides) if sides else False
    if presence_sur_rendu:
        row["elements_rendus"] = len(evidence_presence)
        row["elements_ecartes"] = nb_ecartes
    return row


def _count_deux_cotes(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("deux_cotes_dans_contexte"))


def _disputes_block(
    all_dispute_rows: list[dict[str, Any]],
    *,
    reglage_ids: set[str],
    ecart_ids: set[str],
) -> dict[str, Any]:
    reglage_rows = [row for row in all_dispute_rows if row["question_id"] in reglage_ids]
    ecart_rows = [row for row in all_dispute_rows if row["question_id"] in ecart_ids]
    total = len(all_dispute_rows)
    return {
        "deux_cotes_sur_28": _count_deux_cotes(all_dispute_rows),
        "reglage_sur_14": _count_deux_cotes(reglage_rows),
        "ecart_sur_14": _count_deux_cotes(ecart_rows),
        "disputes_total": total,
        "reglage_total": len(reglage_ids),
        "ecart_total": len(ecart_ids),
    }


def _appui_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    tout_tient = sum(1 for row in rows if row["appui_tout_tient"])
    au_moins_une = sum(1 for row in rows if row["appui_au_moins_une"])
    by_shape: dict[str, dict[str, int]] = {}
    for row in rows:
        shape = row["shape"]
        bucket = by_shape.setdefault(
            shape,
            {"questions": 0, "tout_tient": 0, "au_moins_une": 0},
        )
        bucket["questions"] += 1
        if row["appui_tout_tient"]:
            bucket["tout_tient"] += 1
        if row["appui_au_moins_une"]:
            bucket["au_moins_une"] += 1
    return {
        "tout_tient_sur_141": tout_tient,
        "au_moins_une_sur_141": au_moins_une,
        "questions_total": total,
        "par_shape": by_shape,
    }


def _taille_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [int(row["evidence_count"]) for row in rows]
    chars = [int(row["caracteres"]) for row in rows]
    return {
        "elements": {
            "mediane": int(statistics.median(counts)) if counts else 0,
            "p90": _percentile(counts, 90),
            "max": max(counts) if counts else 0,
        },
        "caracteres": {
            "mediane": int(statistics.median(chars)) if chars else 0,
            "p90": _percentile(chars, 90),
            "max": max(chars) if chars else 0,
            "plafond_8000": sum(1 for value in chars if value >= MAX_RENDER_CHARS),
        },
    }


def _aggregate_config_rows(
    rows: list[dict[str, Any]],
    dispute_rows: list[dict[str, Any]],
    *,
    reglage_ids: set[str],
    ecart_ids: set[str],
) -> dict[str, Any]:
    return {
        "disputes": _disputes_block(dispute_rows, reglage_ids=reglage_ids, ecart_ids=ecart_ids),
        "appui": _appui_block(rows),
        "taille": _taille_block(rows),
        **_aggregate_rows(rows, dispute_rows),
    }


def _format_tableau(results: list[dict[str, Any]]) -> str:
    header = (
        "étiquette | 2c/28 | réglage/14 | écart/14 | tout/141 | ≥1/141 | "
        "méd.élém. | max élém. | p90 car. | plaf.8k"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        disputes = row["disputes"]
        appui = row["appui"]
        taille = row["taille"]
        elements = taille["elements"]
        chars = taille["caracteres"]
        lines.append(
            f"{row['etiquette']:<18} | "
            f"{disputes['deux_cotes_sur_28']:>2}/28 | "
            f"{disputes['reglage_sur_14']:>2}/14 | "
            f"{disputes['ecart_sur_14']:>2}/14 | "
            f"{appui['tout_tient_sur_141']:>3}/141 | "
            f"{appui['au_moins_une_sur_141']:>3}/141 | "
            f"{elements['mediane']:>7} | {elements['max']:>7} | "
            f"{chars['p90']:>7} | {chars['plafond_8000']:>6}"
        )
    return "\n".join(lines) + "\n"


def run_controle(*, essai: bool = False) -> dict[str, Any]:
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
    tours_k16_max_elements: int | None = None

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
                        k=config["k"],
                        mode=config["mode"],
                        budget=config["budget"],
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

        agg = _aggregate_config_rows(
            rows,
            dispute_rows,
            reglage_ids=reglage_ids,
            ecart_ids=ecart_ids,
        )
        payload = {**config, **agg}
        results.append(payload)

        if config["etiquette"] == "tours_k16" and not essai:
            max_elements = payload["taille"]["elements"]["max"]
            tours_k16_max_elements = max_elements
            if max_elements > config["k"]:
                raise RuntimeError(
                    f"tours_k16 rend {max_elements} éléments (> k={config['k']}) — pas un contrôle à budget égal"
                )

    SORTIES_DIR.mkdir(parents=True, exist_ok=True)
    output = {"partage": partage_payload, "configurations": results}
    RESULTAT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TABLEAU_PATH.write_text(_format_tableau(results), encoding="utf-8")
    return {
        "configurations": len(results),
        "questions": len(questions),
        "tours_k16_max_elements": tours_k16_max_elements,
        "files": [str(RESULTAT_PATH), str(TABLEAU_PATH)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contrôle à budget d'éléments égal (fine).")
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Limiter aux 3 premières disputes (pilote).",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        summary = run_controle(essai=args.essai)
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
