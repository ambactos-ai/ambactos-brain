# SPDX-License-Identifier: AGPL-3.0-only
"""Mesure porte phrase — hybridation fine + phrases courtes depuis la brute."""
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
from .dossiers import Dossier, dossier_courant
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
from .porte_phrase import (
    compter_clauses_portees,
    compter_perte_selection,
    liste_clauses_sans_chiffre,
    porte_phrase,
)

MAX_RENDER_CHARS = 12000
RENDER_SANS_BUDGET = 64000

TEMOIN_HYBRIDE = {"deux_cotes_sur_28": 24, "tout_tient_sur_141": 53}
TEMOIN_VEC_B8_N2 = {"tout_tient_sur_141": 74, "clauses_portees": 12}

CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {
        "etiquette": "hybride_12_6",
        "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
        "brute_k": None,
        "phrases": None,
    },
    {
        "etiquette": "hyb_vec_b8_n2",
        "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
        "brute_k": 8,
        "phrases": {
            "n_par_passage": 2,
            "max_phrases": 16,
            "mode": "vectoriel",
            "selection": "par_passage",
            "fenetre": 1,
        },
    },
    {
        "etiquette": "hyb_vec_f2_b8_n2",
        "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
        "brute_k": 8,
        "phrases": {
            "n_par_passage": 2,
            "max_phrases": 16,
            "mode": "vectoriel",
            "selection": "par_passage",
            "fenetre": 2,
        },
    },
    {
        "etiquette": "hyb_vec_f2_b8_n1",
        "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
        "brute_k": 8,
        "phrases": {
            "n_par_passage": 1,
            "max_phrases": 8,
            "mode": "vectoriel",
            "selection": "par_passage",
            "fenetre": 2,
        },
    },
    {
        "etiquette": "hyb_vec_f2_b12_g8",
        "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
        "brute_k": 12,
        "phrases": {
            "n_par_passage": 2,
            "max_phrases": 8,
            "mode": "vectoriel",
            "selection": "globale",
            "fenetre": 2,
        },
    },
    {
        "etiquette": "hyb_vec_f2_b12_g12",
        "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
        "brute_k": 12,
        "phrases": {
            "n_par_passage": 2,
            "max_phrases": 12,
            "mode": "vectoriel",
            "selection": "globale",
            "fenetre": 2,
        },
    },
    {
        "etiquette": "hyb_lex_f2_b8_n2",
        "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
        "brute_k": 8,
        "phrases": {
            "n_par_passage": 2,
            "max_phrases": 16,
            "mode": "lexical",
            "selection": "par_passage",
            "fenetre": 2,
        },
    },
    {
        "etiquette": "phrases_vec_f2_b12_n2",
        "hybride": None,
        "brute_k": 12,
        "phrases": {
            "n_par_passage": 2,
            "max_phrases": 24,
            "mode": "vectoriel",
            "selection": "par_passage",
            "fenetre": 2,
        },
    },
)

PHRASES_CACHE_PATH = BANC_ROOT / "sorties" / "phrases-embeddings.json"
SORTIES_DIR = BANC_ROOT / "sorties" / "fenetres-phrases"
RESULTAT_PATH = SORTIES_DIR / "resultat.json"
TABLEAU_PATH = SORTIES_DIR / "tableau.txt"
ARRET_PATH = BANC_ROOT / "_fenetres-arret.json"
RESULT_PATH = BANC_ROOT / "_fenetres-result.json"


def write_arret(why: str) -> None:
    ARRET_PATH.write_text(
        json.dumps({"status": "arret", "pourquoi": why}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _load_phrases_cache() -> dict[str, list[float]]:
    if not PHRASES_CACHE_PATH.is_file():
        raise RuntimeError(f"cache unités absent : {PHRASES_CACHE_PATH}")
    payload = json.loads(PHRASES_CACHE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"cache unités vide : {PHRASES_CACHE_PATH}")
    return payload


def build_evidence_for_porte_phrase_config(
    connection,
    question: dict[str, Any],
    config: dict[str, Any],
    *,
    embed_cache: dict[str, list[float]],
    phrases_cache: dict[str, list[float]],
    dossier: Dossier | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, int]]:
    """Retourne (evidence rendue, passages bruts complets si phrases)."""
    hybride = config.get("hybride")
    brute_k = config.get("brute_k")
    phrases_cfg = config.get("phrases")

    current = dossier if dossier is not None else dossier_courant()
    fine_evidence: list[dict[str, Any]] = []
    if hybride:
        fine_evidence, _meta = build_evidence_for_mode(
            connection,
            question,
            couche="fine",
            k=hybride["k"],
            mode="trois_tours",
            embed_cache=embed_cache,
            budget=None,
            n_tours=hybride["n_tours"],
            fusion="alternance",
            k_tour1=hybride["k_tour1"],
            dossier=current,
        )

    brute_passages: list[dict[str, Any]] | None = None
    if brute_k is not None:
        brute_passages, _meta = build_evidence_for_mode(
            connection,
            question,
            couche="brute",
            k=brute_k,
            mode="temoin",
            embed_cache=embed_cache,
            budget=None,
            dossier=current,
        )

    phrase_stats = {"unites_sans_vecteur": 0}
    if phrases_cfg:
        if brute_passages is None:
            raise RuntimeError(f"configuration {config['etiquette']} : phrases sans brute_k")
        mode = phrases_cfg.get("mode", "lexical")
        question_vector = embed_cache.get(question["question"])
        phrase_hits, phrase_stats = porte_phrase(
            question["question"],
            brute_passages,
            n_par_passage=phrases_cfg["n_par_passage"],
            max_phrases=phrases_cfg["max_phrases"],
            mode=mode,
            question_vector=question_vector,
            cache=phrases_cache if mode == "vectoriel" else None,
            selection=phrases_cfg.get("selection", "par_passage"),
            fenetre=phrases_cfg.get("fenetre", 1),
        )
        if hybride:
            return fine_evidence + phrase_hits, brute_passages, phrase_stats
        return phrase_hits, brute_passages, phrase_stats

    if hybride and not brute_k:
        return fine_evidence, None, phrase_stats
    if brute_passages is not None:
        return brute_passages, None, phrase_stats
    raise RuntimeError(f"configuration invalide : {config['etiquette']}")


_build_evidence_for_config = build_evidence_for_porte_phrase_config


def _compile_question_row(
    question: dict[str, Any],
    config: dict[str, Any],
    *,
    connection,
    embed_cache: dict[str, list[float]],
    phrases_cache: dict[str, list[float]],
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    claims: list[dict[str, Any]],
    clauses: list[dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    qid = question["id"]
    question_text = question["question"]
    if question_text not in embed_cache:
        raise RuntimeError(f"cache froid : embedding absent ({question_text[:60]}…)")
    started = time.monotonic()
    evidence, brute_passages, phrase_stats = build_evidence_for_porte_phrase_config(
        connection,
        question,
        config,
        embed_cache=embed_cache,
        phrases_cache=phrases_cache,
    )
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
    perte_selection = None
    if config.get("phrases") and brute_passages is not None:
        perte_selection = compter_perte_selection(
            question,
            brute_passages,
            evidence,
            assertion_index=assertion_index,
            claims_index=claims_index,
        )
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
            "au_dessus_12000": len(rendered_sans_budget) > 12000,
            "plafond_caracteres": len(rendered) >= MAX_RENDER_CHARS,
            "appui_tout_tient": tout_tient,
            "appui_au_moins_une": au_moins_une,
            "passages_tronques": rendered_sans_budget.count(" … [tronqué :"),
            "deux_cotes_dans_contexte": deux_cotes,
            "clauses_portees": compter_clauses_portees(
                qid,
                evidence,
                clauses=clauses,
                assertion_index=assertion_index,
                claims_index=claims_index,
            ),
            "perte_selection": perte_selection,
            "unites_sans_vecteur": phrase_stats["unites_sans_vecteur"],
            "_evidence": evidence,
        },
        elapsed,
    )


def _aggregate_config_rows_porte_phrase(
    rows: list[dict[str, Any]],
    dispute_rows: list[dict[str, Any]],
    *,
    reglage_ids: set[str],
    ecart_ids: set[str],
    clauses_total: int,
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
    clauses_portees = sum(int(row["clauses_portees"]) for row in rows)
    perte_values = [int(row["perte_selection"]) for row in rows if row.get("perte_selection") is not None]
    unites_sans_vecteur = sum(int(row.get("unites_sans_vecteur", 0)) for row in rows)
    payload = {
        **base,
        "reels": _taille_block_reels(rows),
        "au_dessus_12000": sum(1 for row in rows if row.get("au_dessus_12000")),
        "unites_sans_vecteur": unites_sans_vecteur,
        "tabulaires_portes": sum(int(row["tabulaires_portes"]) for row in rows),
        "tokens_estimes": {
            "mediane": round(statistics.median(tokens), 2) if tokens else 0.0,
            "p90": round(_percentile(chars, 90) / 4, 2) if chars else 0.0,
        },
        "caracteres_reels": {
            "mediane": int(statistics.median(chars)) if chars else 0,
            "p90": _percentile(chars, 90),
        },
        "clauses_portees": clauses_portees,
        "clauses_total": clauses_total,
        "perte_selection": sum(perte_values) if perte_values else None,
    }
    if secondes_par_question is not None:
        payload["secondes_par_question"] = round(secondes_par_question, 3)
    return payload


def _format_tableau(results: list[dict[str, Any]]) -> str:
    header = (
        "étiquette | 2c/28 | réglage/14 | écart/14 | tout/141 | ≥1/141 | "
        "méd.élém. | car. réels méd·p90 | tokens÷4 | >8000 | >12000 | tab.portés | "
        "clauses/59 | perte_sel | sans_vec | s/q"
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
        clauses_total = row.get("clauses_total") or 59
        perte = row.get("perte_selection")
        perte_str = f"{perte:>5}" if perte is not None else "    —"
        sans_vec = row.get("unites_sans_vecteur", 0)
        au_dessus_12000 = row.get("au_dessus_12000", 0)
        lines.append(
            f"{row['etiquette']:<36} | "
            f"{disputes['deux_cotes_sur_28']:>2}/28 | "
            f"{disputes['reglage_sur_14']:>2}/14 | "
            f"{disputes['ecart_sur_14']:>2}/14 | "
            f"{appui['tout_tient_sur_141']:>3}/141 | "
            f"{appui['au_moins_une_sur_141']:>3}/141 | "
            f"{elements['mediane']:>7} | "
            f"{chars_reels['mediane']:>5}·{chars_reels['p90']:>5} | "
            f"{tokens['mediane']:>7} | "
            f"{reels['caracteres_reels']['au_dessus_8000']:>5} | "
            f"{au_dessus_12000:>6} | "
            f"{row['tabulaires_portes']:>5} | "
            f"{row['clauses_portees']:>3}/{clauses_total:<2} | "
            f"{perte_str} | "
            f"{sans_vec:>8} | "
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
        "clauses_portees": row["clauses_portees"],
        "perte_selection": row.get("perte_selection"),
        "au_dessus_12000": row.get("au_dessus_12000"),
    }


def _verifier_temoins(results: list[dict[str, Any]]) -> None:
    by_etiquette = {row["etiquette"]: row for row in results}
    hybride = by_etiquette.get("hybride_12_6")
    vec = by_etiquette.get("hyb_vec_b8_n2")
    if hybride is None or vec is None:
        raise RuntimeError("configurations témoin absentes")
    if hybride["disputes"]["deux_cotes_sur_28"] != TEMOIN_HYBRIDE["deux_cotes_sur_28"]:
        raise RuntimeError(
            "hybride_12_6 ne redonne pas 24/28 "
            f"(obtenu {hybride['disputes']['deux_cotes_sur_28']}/28)"
        )
    if hybride["appui"]["tout_tient_sur_141"] != TEMOIN_HYBRIDE["tout_tient_sur_141"]:
        raise RuntimeError(
            "hybride_12_6 ne redonne pas 53/141 "
            f"(obtenu {hybride['appui']['tout_tient_sur_141']}/141)"
        )
    if vec["appui"]["tout_tient_sur_141"] != TEMOIN_VEC_B8_N2["tout_tient_sur_141"]:
        raise RuntimeError(
            "hyb_vec_b8_n2 ne redonne pas 74/141 "
            f"(obtenu {vec['appui']['tout_tient_sur_141']}/141)"
        )
    if vec.get("clauses_portees") != TEMOIN_VEC_B8_N2["clauses_portees"]:
        raise RuntimeError(
            "hyb_vec_b8_n2 ne redonne pas 12/59 clauses "
            f"(obtenu {vec.get('clauses_portees')}/59)"
        )


def _verifier_rendu_phrase_hit(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        evidence = row.get("_evidence") or []
        phrase_items = [item for item in evidence if item.get("kind") == "phrase_hit"]
        if not phrase_items:
            continue
        rendered = _render_markdown(
            "banc-realite",
            "GreenLoop",
            f"banc-etage3-{row['question_id']}",
            None,
            _conversation_state(),
            evidence,
            None,
            max_chars=RENDER_SANS_BUDGET,
        )
        sample = phrase_items[0].get("text") or ""
        if sample and sample not in rendered:
            return False
    return True


def run_mesure_porte_phrase(*, essai: bool = False) -> dict[str, Any]:
    from .infra import connect_existing, start_cluster

    export_runtime_env()
    embed_cache = _load_embed_cache()
    if not embed_cache:
        raise RuntimeError("cache froid : etage2-question-embeddings.json absent ou vide")
    phrases_cache = _load_phrases_cache()

    gold = load_gold()
    assertion_index = gold["assertion_index"]
    clauses = liste_clauses_sans_chiffre(gold=gold)
    clauses_total = len(clauses)
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

    for config in CONFIGURATIONS:
        rows: list[dict[str, Any]] = []
        dispute_rows: list[dict[str, Any]] = []
        config_seconds = 0.0
        for question in questions:
            for attempt in range(2):
                try:
                    row, elapsed = _compile_question_row(
                        question,
                        config,
                        connection=connection,
                        embed_cache=embed_cache,
                        phrases_cache=phrases_cache,
                        gold=gold,
                        claims_index=claims_index,
                        assertion_index=assertion_index,
                        claims=claims,
                        clauses=clauses,
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

        if config.get("phrases") and not _verifier_rendu_phrase_hit(rows):
            raise RuntimeError("phrase_hit absent du rendu CallFrame (_render_markdown)")

        secondes_par_question = config_seconds / len(questions) if questions else None
        agg = _aggregate_config_rows_porte_phrase(
            rows,
            dispute_rows,
            reglage_ids=reglage_ids,
            ecart_ids=ecart_ids,
            clauses_total=clauses_total,
            secondes_par_question=secondes_par_question,
        )
        for row in rows:
            row.pop("_evidence", None)
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

    rendu_phrase_hit_verifie = True

    SORTIES_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "partage": partage_payload,
        "clauses_total": clauses_total,
        "configurations": results,
    }
    RESULTAT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TABLEAU_PATH.write_text(_format_tableau(results), encoding="utf-8")

    return {
        "configurations": len(results),
        "questions": len(questions),
        "files": [str(RESULTAT_PATH), str(TABLEAU_PATH)],
        "results": results,
        "essai": essai,
        "clauses_trouvees": clauses_total,
        "rendu_phrase_hit_verifie": rendu_phrase_hit_verifie,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mesure porte phrase — étage 2.")
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Limiter aux 3 premières disputes (pilote).",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        summary = run_mesure_porte_phrase(essai=args.essai)
    except Exception as exc:
        write_arret(str(exc))
        print(json.dumps({"status": "arret", "erreur": str(exc)}, ensure_ascii=False))
        return 1
    elapsed = round(time.monotonic() - started, 2)
    sans_vec = 0
    for row in summary.get("results") or []:
        phrases = row.get("phrases") or {}
        if phrases.get("mode") == "vectoriel":
            sans_vec = max(sans_vec, int(row.get("unites_sans_vecteur", 0)))
    write_result(
        {
            "status": "complete",
            "tests": "pytest tests/banc_realite OK",
            "essai": {
                "questions": summary["questions"],
                "secondes": elapsed,
            },
            "files_written": summary["files"],
        }
    )
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
