# SPDX-License-Identifier: AGPL-3.0-only
"""Diagnostic de couverture — où sont les appuis que la couche fine ne porte pas ?"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import date
from typing import Any

import psycopg.errors

from . import etage3 as e3
from .config import BANC_ROOT, export_runtime_env
from .fidelite import _assertion_needle, claim_is_tabular, load_gold
from .holders import (
    _strict_text_holds,
    _text_matches_needle,
    build_holder_table,
    collect_appui_states,
    model_claims_on_piece,
)
from .measure_disputes import _read_public_claims, _search_simple, build_evidence_for_mode
from .measure_etage2 import _load_embed_cache
from .normalize import piece_basename

SORTIES_DIR = BANC_ROOT / "sorties" / "diagnostic-couverture"
RESULTAT_PATH = SORTIES_DIR / "resultat.json"
TABLEAU_PATH = SORTIES_DIR / "tableau.txt"
ARRET_PATH = BANC_ROOT / "_diagnostic-arret.json"
RESULT_PATH = BANC_ROOT / "_diagnostic-result.json"

VERDICTS_NON_PORTES = (
    "non_extrait",
    "extrait_infidele",
    "retrouvable_rang_13_50",
    "retrouvable_rang_51_200",
    "hors_top_200",
)
VERDICTS_TOUS = ("portee",) + VERDICTS_NON_PORTES
SHAPES = ("single", "dispute", "aggregate", "status", "unknown", "trajectory")
CONTROLE_TOUT_TIENT = 48
CONTROLE_AU_MOINS_UNE = 86

_NUMBER_RE = re.compile(r"\d+")


def write_arret(why: str) -> None:
    ARRET_PATH.write_text(
        json.dumps({"status": "arret", "pourquoi": why}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def etat_contexte_fine(states: dict[str, list[Any]]) -> str:
    if states.get("tient"):
        return "tient"
    if states.get("candidat"):
        return "candidat"
    return "absent"


def _claims_by_piece(claims: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_piece: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        by_piece[claim.get("piece") or piece_basename(claim.get("source_file_path") or "")].append(claim)
    return by_piece


def _tenants_en_base(assertion: dict[str, Any], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [claim for claim in claims if _strict_text_holds(assertion, claim)]


def _tabular_tient_sur_piece(
    assertion: dict[str, Any],
    piece_claims: list[dict[str, Any]],
) -> bool:
    needle = _assertion_needle(assertion)
    if not needle:
        return False
    for claim in piece_claims:
        if claim_is_tabular(claim) and _text_matches_needle(claim, needle):
            return True
    return False


def _best_rank_for_claims(claim_ids: set[str], evidence: list[dict[str, Any]]) -> int | None:
    best: int | None = None
    for rank, item in enumerate(evidence, 1):
        item_id = str(item.get("id") or "")
        if item_id in claim_ids and (best is None or rank < best):
            best = rank
    return best


def verdict_depuis_rang(rang: int | None) -> str:
    if rang is None or rang > 200:
        return "hors_top_200"
    if rang <= 50:
        return "retrouvable_rang_13_50"
    return "retrouvable_rang_51_200"


def diagnostiquer_assertion(
    assertion: dict[str, Any],
    *,
    fine_evidence: list[dict[str, Any]],
    brute_evidence: list[dict[str, Any]],
    search_evidence: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    claims_by_piece: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    piece = piece_basename(assertion.get("source") or "")
    states = collect_appui_states(assertion, fine_evidence, claims_index)
    ctx = etat_contexte_fine(states)
    brute_states = collect_appui_states(assertion, brute_evidence, claims_index)
    brute_k12_tient = bool(brute_states["tient"])

    if states["tient"]:
        return {
            "piece": piece,
            "verdict": "portee",
            "tabulaire_tient": False,
            "tenant_tabulaire": False,
            "meilleur_rang": None,
            "nb_tenants_en_base": 0,
            "brute_k12_tient": brute_k12_tient,
            "etat_contexte_fine": ctx,
        }

    piece_claims = claims_by_piece.get(piece, [])
    tabulaire_tient = _tabular_tient_sur_piece(assertion, piece_claims)
    tenants = _tenants_en_base(assertion, claims)
    nb_tenants = len(tenants)

    if not tenants:
        non_tab = model_claims_on_piece(claims, piece)
        verdict = "extrait_infidele" if non_tab else "non_extrait"
        return {
            "piece": piece,
            "verdict": verdict,
            "tabulaire_tient": tabulaire_tient,
            "tenant_tabulaire": False,
            "meilleur_rang": None,
            "nb_tenants_en_base": 0,
            "brute_k12_tient": brute_k12_tient,
            "etat_contexte_fine": ctx,
        }

    tenant_ids = {str(c["claim_id"]) for c in tenants}
    meilleur_rang = _best_rank_for_claims(tenant_ids, search_evidence)
    verdict = verdict_depuis_rang(meilleur_rang)
    tenant_tabulaire = any(claim_is_tabular(c) for c in tenants)

    return {
        "piece": piece,
        "verdict": verdict,
        "tabulaire_tient": tabulaire_tient,
        "tenant_tabulaire": tenant_tabulaire,
        "meilleur_rang": meilleur_rang,
        "nb_tenants_en_base": nb_tenants,
        "brute_k12_tient": brute_k12_tient,
        "etat_contexte_fine": ctx,
    }


def cause_dominante(verdicts: list[str]) -> str | None:
    non_portes = [v for v in verdicts if v != "portee"]
    if not non_portes:
        return None
    counts = Counter(non_portes)
    max_count = max(counts.values())
    for verdict in VERDICTS_NON_PORTES:
        if counts.get(verdict, 0) == max_count:
            return verdict
    return non_portes[0]


def compter_couverture_fine(
    lignes: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, int]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in lignes:
        by_question[line["question_id"]].append(line)

    tout_tient = 0
    au_moins_une = 0
    for question in questions:
        qid = question["id"]
        rows = by_question.get(qid, [])
        if not rows:
            continue
        states = [row["etat_contexte_fine"] for row in rows]
        if states and all(state == "tient" for state in states):
            tout_tient += 1
        if any(state == "tient" for state in states):
            au_moins_une += 1
    return {"tout_tient": tout_tient, "au_moins_une": au_moins_une}


def verifier_controle_fine(couverture: dict[str, int]) -> str | None:
    if couverture["tout_tient"] != CONTROLE_TOUT_TIENT:
        return (
            f"contexte fine incohérent : tout_tient={couverture['tout_tient']} "
            f"(attendu {CONTROLE_TOUT_TIENT})"
        )
    if couverture["au_moins_une"] != CONTROLE_AU_MOINS_UNE:
        return (
            f"contexte fine incohérent : au_moins_une={couverture['au_moins_une']} "
            f"(attendu {CONTROLE_AU_MOINS_UNE})"
        )
    return None


def agreger_assertions(lignes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(lignes)
    portees = sum(1 for line in lignes if line["verdict"] == "portee")
    non_portees = total - portees
    par_verdict: dict[str, dict[str, int]] = {}
    for verdict in VERDICTS_NON_PORTES:
        subset = [line for line in lignes if line["verdict"] == verdict]
        par_verdict[verdict] = {
            "total": len(subset),
            "brute_k12_tient": sum(1 for line in subset if line["brute_k12_tient"]),
            "tabulaire_tient": sum(1 for line in subset if line["tabulaire_tient"]),
        }
    somme_verdicts = sum(par_verdict[v]["total"] for v in VERDICTS_NON_PORTES)
    return {
        "total": total,
        "portees": portees,
        "non_portees": non_portees,
        "par_verdict": par_verdict,
        "somme_verdicts": somme_verdicts,
    }


def tableau_causes_dominantes(
    lignes: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in lignes:
        by_question[line["question_id"]].append(line)

    question_shapes = {q["id"]: q.get("shape") or "unknown" for q in questions}
    grid: dict[str, dict[str, int]] = {v: {s: 0 for s in SHAPES} for v in VERDICTS_NON_PORTES}
    grid["complet"] = {s: 0 for s in SHAPES}

    for question in questions:
        qid = question["id"]
        shape = question_shapes[qid]
        if shape not in SHAPES:
            shape = "unknown"
        rows = by_question.get(qid, [])
        if not rows:
            continue
        if all(row["verdict"] == "portee" for row in rows):
            grid["complet"][shape] += 1
            continue
        dom = cause_dominante([row["verdict"] for row in rows])
        if dom:
            grid[dom][shape] += 1
    return grid


def top_pieces_extraction(
    lignes: list[dict[str, Any]],
    claims_by_piece: dict[str, list[dict[str, Any]]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for line in lignes:
        if line["verdict"] in ("non_extrait", "extrait_infidele"):
            counts[line["piece"]] += 1
    rows = []
    for piece, count in counts.most_common(limit):
        non_tab = sum(1 for c in claims_by_piece.get(piece, []) if not claim_is_tabular(c))
        rows.append({"piece": piece, "assertions": count, "claims_non_tabulaires": non_tab})
    return rows


def _nombres_communs(text_a: str, text_b: str) -> int:
    nums_a = set(_NUMBER_RE.findall(text_a))
    nums_b = set(_NUMBER_RE.findall(text_b))
    return len(nums_a & nums_b)


def _claim_plus_proche(
    assertion: dict[str, Any],
    piece_claims: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    needle = _assertion_needle(assertion)
    non_tab = [c for c in piece_claims if not claim_is_tabular(c)]
    if not non_tab:
        return None, "aucun claim non tabulaire"
    best = max(non_tab, key=lambda c: _nombres_communs(needle, c.get("text") or ""))
    common = _nombres_communs(needle, best.get("text") or "")
    if common == 0:
        return best, "aucun nombre commun"
    return best, f"{common} nombre(s) commun(s)"


def selectionner_exemples(
    lignes: list[dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    claims_by_piece: dict[str, list[dict[str, Any]]],
    *,
    par_verdict: int = 2,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for verdict in VERDICTS_TOUS:
        pool = [line for line in lignes if line["verdict"] == verdict]
        for line in pool[:par_verdict]:
            assertion = assertion_index[line["assertion_id"]]
            texte = (assertion.get("quote") or assertion.get("content") or "").strip()
            if len(texte) > 200:
                texte = texte[:197] + "…"
            item: dict[str, Any] = {
                "verdict": verdict,
                "question_id": line["question_id"],
                "assertion_id": line["assertion_id"],
                "assertion": texte,
                "piece": line["piece"],
            }
            if verdict == "extrait_infidele":
                claim, note = _claim_plus_proche(assertion, claims_by_piece.get(line["piece"], []))
                if claim:
                    claim_text = (claim.get("text") or "")[:160]
                    item["claim_proche"] = f"{claim_text} ({note})"
                else:
                    item["claim_proche"] = note
            examples.append(item)
    return examples


def _format_tableau(
    agg: dict[str, Any],
    causes: dict[str, dict[str, int]],
    pieces: list[dict[str, Any]],
    couverture: dict[str, int],
    examples: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("=== Assertions ===")
    lines.append(f"total={agg['total']} portées={agg['portees']} non_portées={agg['non_portees']}")
    lines.append(f"contrôle fine : tout_tient={couverture['tout_tient']} au_moins_une={couverture['au_moins_une']}")
    lines.append(f"somme verdicts non portés={agg['somme_verdicts']} (attendu={agg['non_portees']})")
    lines.append("")
    for verdict in VERDICTS_NON_PORTES:
        block = agg["par_verdict"][verdict]
        lines.append(
            f"{verdict}: {block['total']} "
            f"(brute_k12_tient={block['brute_k12_tient']}, tabulaire_tient={block['tabulaire_tient']})"
        )
    lines.append("")
    lines.append("=== Cause dominante × forme (questions sans appui complet) ===")
    header = "verdict | " + " | ".join(SHAPES)
    lines.append(header)
    lines.append("-" * len(header))
    for verdict in ("complet",) + VERDICTS_NON_PORTES:
        row = causes[verdict]
        lines.append(verdict + " | " + " | ".join(str(row[s]) for s in SHAPES))
    lines.append("")
    lines.append("=== Top pièces (non_extrait + extrait_infidele) ===")
    for row in pieces:
        lines.append(
            f"{row['piece']}: {row['assertions']} assertions, "
            f"{row['claims_non_tabulaires']} claims non tabulaires"
        )
    lines.append("")
    lines.append("=== Exemples (2 par verdict) ===")
    for ex in examples:
        lines.append(f"[{ex['verdict']}] {ex['question_id']} / {ex['assertion_id']}")
        lines.append(f"  pièce: {ex['piece']}")
        lines.append(f"  assertion: {ex['assertion']}")
        if "claim_proche" in ex:
            lines.append(f"  claim proche: {ex['claim_proche']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_evidence_with_retry(
    connection,
    question: dict[str, Any],
    *,
    embed_cache: dict[str, list[float]],
    couche: str,
    k: int,
    mode: str,
    n_tours: int | None = None,
    fusion: str | None = None,
):
    from .infra import connect_existing, start_cluster

    kwargs: dict[str, Any] = {
        "couche": couche,
        "k": k,
        "mode": mode,
        "embed_cache": embed_cache,
        "budget": None,
    }
    if mode == "trois_tours":
        kwargs["n_tours"] = n_tours if n_tours is not None else 3
        kwargs["fusion"] = fusion if fusion is not None else "alternance"
    for attempt in range(2):
        try:
            evidence, _meta = build_evidence_for_mode(connection, question, **kwargs)
            return evidence
        except Exception as exc:
            if attempt == 0 and isinstance(exc, psycopg.errors.AdminShutdown):
                start_cluster()
                connection = connect_existing()
                continue
            raise
    raise RuntimeError("échec build_evidence après reprise")


def _search_with_retry(
    connection,
    question: dict[str, Any],
    *,
    embed_cache: dict[str, list[float]],
    couche: str,
    k: int,
) -> list[dict[str, Any]]:
    from .infra import connect_existing, start_cluster

    for attempt in range(2):
        try:
            return _search_simple(connection, question, couche=couche, k=k, embed_cache=embed_cache)
        except Exception as exc:
            if attempt == 0 and isinstance(exc, psycopg.errors.AdminShutdown):
                start_cluster()
                connection = connect_existing()
                continue
            raise
    raise RuntimeError("échec recherche après reprise")


def run_diagnostic(*, essai: bool = False) -> dict[str, Any]:
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
    claims_by_piece = _claims_by_piece(claims)

    all_questions = e3.load_questions()
    questions = all_questions[:5] if essai else all_questions

    lignes: list[dict[str, Any]] = []
    for question in questions:
        fine_evidence = _build_evidence_with_retry(
            connection,
            question,
            embed_cache=embed_cache,
            couche="fine",
            k=12,
            mode="trois_tours",
            n_tours=3,
        )
        brute_evidence = _build_evidence_with_retry(
            connection,
            question,
            embed_cache=embed_cache,
            couche="brute",
            k=12,
            mode="temoin",
        )
        search_evidence = _search_with_retry(
            connection,
            question,
            embed_cache=embed_cache,
            couche="fine",
            k=200,
        )

        shape = question.get("shape") or "unknown"
        for aid in question.get("supporting_assertions") or []:
            assertion = assertion_index[aid]
            diag = diagnostiquer_assertion(
                assertion,
                fine_evidence=fine_evidence,
                brute_evidence=brute_evidence,
                search_evidence=search_evidence,
                claims=claims,
                claims_index=claims_index,
                claims_by_piece=claims_by_piece,
            )
            lignes.append(
                {
                    "question_id": question["id"],
                    "shape": shape,
                    "assertion_id": aid,
                    **diag,
                }
            )

    couverture = compter_couverture_fine(lignes, questions)
    if not essai:
        erreur = verifier_controle_fine(couverture)
        if erreur:
            raise RuntimeError(erreur)

    agg = agreger_assertions(lignes)
    if not essai and agg["somme_verdicts"] != agg["non_portees"]:
        raise RuntimeError(
            f"somme verdicts ({agg['somme_verdicts']}) ≠ non portées ({agg['non_portees']})"
        )

    causes = tableau_causes_dominantes(lignes, questions)
    pieces = top_pieces_extraction(lignes, claims_by_piece)
    examples = selectionner_exemples(lignes, assertion_index, claims_by_piece)
    tableau = _format_tableau(agg, causes, pieces, couverture, examples)

    SORTIES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": date.today().isoformat(),
        "essai": essai,
        "questions": len(questions),
        "couverture_fine": couverture,
        "agregats": agg,
        "causes_dominantes": causes,
        "top_pieces": pieces,
        "exemples": examples,
        "lignes": lignes,
    }
    RESULTAT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TABLEAU_PATH.write_text(tableau, encoding="utf-8")

    verdicts = Counter(line["verdict"] for line in lignes)
    return {
        "questions": len(questions),
        "assertions": len(lignes),
        "couverture_fine": couverture,
        "verdicts": dict(verdicts),
        "files": [str(RESULTAT_PATH), str(TABLEAU_PATH)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnostic de couverture fine GreenLoop.")
    parser.add_argument("--essai", action="store_true", help="Limiter aux 5 premières questions.")
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        summary = run_diagnostic(essai=args.essai)
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
    if args.essai:
        result_marker = {
            "status": "complete",
            "tests": "tests/banc_realite/test_diagnostic_couverture.py",
            "essai": {
                "questions": summary["questions"],
                "secondes": elapsed,
                "assertions": summary["assertions"],
                "verdicts": summary["verdicts"],
            },
            "files_written": summary["files"],
        }
        RESULT_PATH.write_text(
            json.dumps(result_marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
