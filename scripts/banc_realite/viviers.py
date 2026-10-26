# SPDX-License-Identifier: AGPL-3.0-only
"""Mesure viviers larges — RRF sur pool fixe, coupé à k."""
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
from .balayage_budget import _percentile
from .config import BANC_ROOT, export_runtime_env
from .controle_budget_egal import _aggregate_config_rows
from .course_couches import _taille_block_reels
from .fidelite import claim_is_tabular, load_gold
from .holders import _strict_text_holds, build_holder_table, collect_appui_states
from .measure_disputes import (
    _read_public_claims,
    _side_presence,
    build_evidence_for_mode,
)
from .measure_etage2 import _appui_summary, _load_embed_cache
from .partage_disputes import partager

MAX_RENDER_CHARS = 8000
RENDER_SANS_BUDGET = 64000

TEMOIN_FINE_TOURS3 = {"deux_cotes_sur_28": 22, "tout_tient_sur_141": 48}
TEMOIN_HYBRIDE_TOURS2 = {"deux_cotes_sur_28": 24, "tout_tient_sur_141": 53}

CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {
        "etiquette": "fine_tours3_k12",
        "couche": "fine",
        "mode": "trois_tours",
        "k": 12,
        "n_tours": 3,
        "k_tour1": None,
        "vivier": None,
    },
    {
        "etiquette": "hybride_12_plus_6_tours2",
        "couche": "fine",
        "mode": "trois_tours",
        "k": 18,
        "n_tours": 2,
        "k_tour1": 12,
        "vivier": None,
    },
    {
        "etiquette": "fine_simple_k12_v200",
        "couche": "fine",
        "mode": "temoin",
        "k": 12,
        "n_tours": None,
        "k_tour1": None,
        "vivier": 200,
    },
    {
        "etiquette": "fine_tours3_k12_v200",
        "couche": "fine",
        "mode": "trois_tours",
        "k": 12,
        "n_tours": 3,
        "k_tour1": None,
        "vivier": 200,
    },
    {
        "etiquette": "hybride_12_plus_6_tours2_v200",
        "couche": "fine",
        "mode": "trois_tours",
        "k": 18,
        "n_tours": 2,
        "k_tour1": 12,
        "vivier": 200,
    },
    {
        "etiquette": "hybride_12_plus_6_tours2_v60",
        "couche": "fine",
        "mode": "trois_tours",
        "k": 18,
        "n_tours": 2,
        "k_tour1": 12,
        "vivier": 60,
    },
    {
        "etiquette": "fine_tours3_k16_v200",
        "couche": "fine",
        "mode": "trois_tours",
        "k": 16,
        "n_tours": 3,
        "k_tour1": None,
        "vivier": 200,
    },
)

SORTIES_DIR = BANC_ROOT / "sorties" / "viviers"
RESULTAT_PATH = SORTIES_DIR / "resultat.json"
TABLEAU_PATH = SORTIES_DIR / "tableau.txt"
ARRET_PATH = BANC_ROOT / "_viviers-arret.json"
RESULT_PATH = BANC_ROOT / "_viviers-result.json"


def write_arret(why: str) -> None:
    ARRET_PATH.write_text(
        json.dumps({"status": "arret", "pourquoi": why}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _conversation_state() -> dict[str, Any]:
    return {"responsibility_context": "GreenLoop"}


def _count_tabulaires_portes(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    claims: list[dict[str, Any]],
) -> int:
    count = 0
    for aid in question.get("supporting_assertions") or []:
        assertion = assertion_index[aid]
        states = collect_appui_states(assertion, evidence, claims_index)
        if not states["tient"]:
            continue
        tenants = [claim for claim in claims if _strict_text_holds(assertion, claim)]
        if any(claim_is_tabular(claim) for claim in tenants):
            count += 1
    return count


def _evidence_ids(evidence: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("id") or "") for item in evidence]


def _compile_question_row(
    question: dict[str, Any],
    *,
    connection,
    embed_cache: dict[str, list[float]],
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    claims: list[dict[str, Any]],
    couche: str,
    k: int,
    mode: str,
    n_tours: int | None,
    k_tour1: int | None,
    vivier: int | None,
) -> tuple[dict[str, Any], float]:
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
        "vivier": vivier,
    }
    if mode == "trois_tours":
        evidence_kwargs["n_tours"] = n_tours if n_tours is not None else 3
        evidence_kwargs["fusion"] = "alternance"
        if k_tour1 is not None:
            evidence_kwargs["k_tour1"] = k_tour1
    started = time.monotonic()
    evidence, _meta = build_evidence_for_mode(connection, question, **evidence_kwargs)
    elapsed = time.monotonic() - started
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
    deux_cotes = None
    if question.get("shape") == "dispute":
        sides = _side_presence(question, evidence, gold=gold, claims_index=claims_index)
        deux_cotes = all(s["dans_contexte"] for s in sides) if sides else False
    return (
        {
            "question_id": qid,
            "shape": question.get("shape") or "unknown",
            "deux_cotes": deux_cotes,
            "tout_tient": tout_tient,
            "au_moins_une": au_moins_une,
            "caracteres": len(rendered_sans_budget),
            "tabulaires_portes": _count_tabulaires_portes(
                question,
                evidence,
                assertion_index=assertion_index,
                claims_index=claims_index,
                claims=claims,
            ),
            "evidence_count": len(evidence),
            "caracteres_sans_budget": len(rendered_sans_budget),
            "au_dessus_8000": len(rendered_sans_budget) > 8000,
            "plafond_caracteres": len(rendered) >= MAX_RENDER_CHARS,
            "appui_tout_tient": tout_tient,
            "appui_au_moins_une": au_moins_une,
            "passages_tronques": rendered_sans_budget.count(" … [tronqué :"),
            "deux_cotes_dans_contexte": deux_cotes,
            "evidence_ids": _evidence_ids(evidence),
        },
        elapsed,
    )


def _aggregate_config_rows_viviers(
    rows: list[dict[str, Any]],
    dispute_rows: list[dict[str, Any]],
    *,
    reglage_ids: set[str],
    ecart_ids: set[str],
    secondes_par_question: float | None,
) -> dict[str, Any]:
    base = _aggregate_config_rows(
        rows,
        dispute_rows,
        reglage_ids=reglage_ids,
        ecart_ids=ecart_ids,
    )
    chars = [int(row["caracteres_sans_budget"]) for row in rows]
    tokens = [value / 4 for value in chars]
    payload = {
        **base,
        "reels": _taille_block_reels(rows),
        "tabulaires_portes": sum(int(row["tabulaires_portes"]) for row in rows),
        "tokens_estimes": {
            "mediane": round(statistics.median(tokens), 2) if tokens else 0.0,
            "p90": round(_percentile(chars, 90) / 4, 2) if chars else 0.0,
        },
        "caracteres_reels": {
            "mediane": int(statistics.median(chars)) if chars else 0,
            "p90": _percentile(chars, 90),
        },
    }
    if secondes_par_question is not None:
        payload["secondes_par_question"] = round(secondes_par_question, 3)
    return payload


def _format_tableau(results: list[dict[str, Any]]) -> str:
    header = (
        "étiquette | 2c/28 | réglage/14 | écart/14 | tout/141 | ≥1/141 | "
        "méd.élém. | car. réels méd·p90 | tokens÷4 | >8000 | tab.portés | s/q"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        disputes = row["disputes"]
        appui = row["appui"]
        taille = row["taille"]
        reels = row["reels"]
        elements = taille["elements"]
        chars_reels = row["caracteres_reels"]
        tokens = row["tokens_estimes"]
        spq = row.get("secondes_par_question")
        spq_str = f"{spq:>6.3f}" if spq is not None else "     —"
        lines.append(
            f"{row['etiquette']:<32} | "
            f"{disputes['deux_cotes_sur_28']:>2}/28 | "
            f"{disputes['reglage_sur_14']:>2}/14 | "
            f"{disputes['ecart_sur_14']:>2}/14 | "
            f"{appui['tout_tient_sur_141']:>3}/141 | "
            f"{appui['au_moins_une_sur_141']:>3}/141 | "
            f"{elements['mediane']:>7} | "
            f"{chars_reels['mediane']:>5}·{chars_reels['p90']:>5} | "
            f"{tokens['mediane']:>7} | "
            f"{reels['caracteres_reels']['au_dessus_8000']:>5} | "
            f"{row['tabulaires_portes']:>5} | "
            f"{spq_str}"
        )
    return "\n".join(lines) + "\n"


def _question_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": row["question_id"],
        "deux_cotes": row["deux_cotes"],
        "tout_tient": row["tout_tient"],
        "au_moins_une": row["au_moins_une"],
        "caracteres": row["caracteres"],
        "tabulaires_portes": row["tabulaires_portes"],
    }


def _verifier_temoins(results: list[dict[str, Any]]) -> None:
    by_etiquette = {row["etiquette"]: row for row in results}
    fine = by_etiquette.get("fine_tours3_k12")
    hybride = by_etiquette.get("hybride_12_plus_6_tours2")
    if fine is None or hybride is None:
        raise RuntimeError("configurations témoin absentes")
    if fine["disputes"]["deux_cotes_sur_28"] != TEMOIN_FINE_TOURS3["deux_cotes_sur_28"]:
        raise RuntimeError(
            "fine_tours3_k12 ne redonne pas 22/28 "
            f"(obtenu {fine['disputes']['deux_cotes_sur_28']}/28)"
        )
    if fine["appui"]["tout_tient_sur_141"] != TEMOIN_FINE_TOURS3["tout_tient_sur_141"]:
        raise RuntimeError(
            "fine_tours3_k12 ne redonne pas 48/141 "
            f"(obtenu {fine['appui']['tout_tient_sur_141']}/141)"
        )
    if hybride["disputes"]["deux_cotes_sur_28"] != TEMOIN_HYBRIDE_TOURS2["deux_cotes_sur_28"]:
        raise RuntimeError(
            "hybride_12_plus_6_tours2 ne redonne pas 24/28 "
            f"(obtenu {hybride['disputes']['deux_cotes_sur_28']}/28)"
        )
    if hybride["appui"]["tout_tient_sur_141"] != TEMOIN_HYBRIDE_TOURS2["tout_tient_sur_141"]:
        raise RuntimeError(
            "hybride_12_plus_6_tours2 ne redonne pas 53/141 "
            f"(obtenu {hybride['appui']['tout_tient_sur_141']}/141)"
        )


def run_viviers(*, essai: bool = False) -> dict[str, Any]:
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
        questions = [q for q in all_questions if q.get("shape") == "dispute"][:3]

    results: list[dict[str, Any]] = []
    tronques_total = 0
    v200_seconds: list[float] = []
    v200_questions = 0

    for config in CONFIGURATIONS:
        rows: list[dict[str, Any]] = []
        dispute_rows: list[dict[str, Any]] = []
        config_seconds = 0.0
        for question in questions:
            for attempt in range(2):
                try:
                    row, elapsed = _compile_question_row(
                        question,
                        connection=connection,
                        embed_cache=embed_cache,
                        gold=gold,
                        claims_index=claims_index,
                        assertion_index=assertion_index,
                        claims=claims,
                        couche=config["couche"],
                        k=config["k"],
                        mode=config["mode"],
                        n_tours=config.get("n_tours"),
                        k_tour1=config.get("k_tour1"),
                        vivier=config.get("vivier"),
                    )
                    break
                except Exception as exc:
                    if attempt == 0 and isinstance(exc, psycopg.errors.AdminShutdown):
                        start_cluster()
                        connection = connect_existing()
                        continue
                    raise
            rows.append(row)
            config_seconds += elapsed
            tronques_total += int(row.get("passages_tronques", 0))
            if question.get("shape") == "dispute":
                dispute_rows.append(row)

        secondes_par_question = config_seconds / len(questions) if questions else None
        if config.get("vivier") == 200:
            v200_seconds.append(config_seconds)
            v200_questions += len(questions)

        agg = _aggregate_config_rows_viviers(
            rows,
            dispute_rows,
            reglage_ids=reglage_ids,
            ecart_ids=ecart_ids,
            secondes_par_question=secondes_par_question,
        )
        payload = {
            **config,
            **agg,
            "questions": [_question_payload(row) for row in rows],
        }
        results.append(payload)

    if tronques_total != 0:
        raise RuntimeError(f"marqueurs tronqués détectés : {tronques_total}")

    if not essai:
        _verifier_temoins(results)

    SORTIES_DIR.mkdir(parents=True, exist_ok=True)
    output = {"partage": partage_payload, "configurations": results}
    RESULTAT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TABLEAU_PATH.write_text(_format_tableau(results), encoding="utf-8")

    secondes_par_question_v200 = None
    if v200_questions:
        secondes_par_question_v200 = round(sum(v200_seconds) / len(v200_seconds) / len(questions), 3)

    return {
        "configurations": len(results),
        "questions": len(questions),
        "files": [str(RESULTAT_PATH), str(TABLEAU_PATH)],
        "results": results,
        "essai": essai,
        "secondes_par_question_v200": secondes_par_question_v200,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mesure viviers larges RRF.")
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Limiter aux 3 premières disputes (pilote).",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        summary = run_viviers(essai=args.essai)
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
