# SPDX-License-Identifier: AGPL-3.0-only
"""Étage 2 verdicts by question shape (plancher / plafond via predicate)."""
from __future__ import annotations

from typing import Any, Callable

from .fidelite import claim_has_ordering_date, extract_ordering_date
from .normalize import piece_basename

PresentFn = Callable[[str], tuple[bool, int | None]]


def rendered_presence(claim: dict[str, Any], rendered: str) -> float:
    text = claim.get("text") or ""
    if not text or not rendered:
        return 0.0
    norm_claim = text.strip()
    if norm_claim in rendered:
        return 1.0
    words = [w for w in norm_claim.split() if len(w) > 3]
    if not words:
        return 0.0
    found = sum(1 for w in words if w in rendered)
    return round(found / len(words), 4)


def verdict_single(question: dict[str, Any], present: PresentFn) -> dict[str, Any]:
    supports = question.get("supporting_assertions") or []
    if not supports:
        return {"verdict": "aucun_support", "supports": []}
    aid = supports[0]
    injected, rank = present(aid)
    return {
        "verdict": "reussi" if injected else "echec",
        "rang": rank,
        "assertion": aid,
    }


def verdict_aggregate(question: dict[str, Any], present: PresentFn) -> dict[str, Any]:
    supports = question.get("supporting_assertions") or []
    if not supports:
        return {"verdict": "aucun_support", "supports": []}
    found = []
    for aid in supports:
        injected, rank = present(aid)
        found.append({"assertion": aid, "injecte": injected, "rang": rank})
    n_found = sum(1 for f in found if f["injecte"])
    if n_found == len(supports):
        verdict = "reussi"
    elif n_found > 0:
        verdict = "partiel"
    else:
        verdict = "echec"
    return {"verdict": verdict, "trouves": n_found, "attendus": len(supports), "supports": found}


def dispute_sides_for_question(
    question: dict[str, Any],
    disputes: list[dict[str, Any]],
) -> list[str]:
    qtext = (question.get("question") or "").strip()
    for dispute in disputes:
        if dispute.get("question", "").strip() == qtext:
            return [side["assertion"] for side in dispute.get("sides", [])]
    return list(question.get("supporting_assertions") or [])


def verdict_dispute(
    question: dict[str, Any],
    present: PresentFn,
    *,
    disputes: list[dict[str, Any]],
) -> dict[str, Any]:
    sides = dispute_sides_for_question(question, disputes)
    if not sides:
        return {"verdict": "aucun_cote"}
    injected_sides = []
    for aid in sides:
        injected, rank = present(aid)
        injected_sides.append({"assertion": aid, "injecte": injected, "rang": rank})
    n = sum(1 for s in injected_sides if s["injecte"])
    total = len(sides)
    if n == 0:
        verdict = "aucun"
    elif n == 1:
        verdict = "un_seul_cote"
    elif n == total:
        verdict = "complet"
    else:
        verdict = "partiel"
    return {"verdict": verdict, "cotes_injectes": n, "cotes_total": total, "sides": injected_sides}


def _situation_for_assertion(assertion_id: str, situations: list[dict[str, Any]]) -> dict[str, Any] | None:
    for situation in situations:
        for assertion in situation.get("assertions", []):
            if assertion.get("id") == assertion_id:
                return situation
    return None


def _trajectory_assertions(trajectory_id: str, trajectories: list[dict[str, Any]]) -> list[str]:
    for trajectory in trajectories:
        if trajectory.get("id") == trajectory_id:
            return [v["assertion"] for v in trajectory.get("versions", [])]
    return []


def verdict_status(
    question: dict[str, Any],
    present: PresentFn,
    *,
    situations: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
) -> dict[str, Any]:
    base = verdict_aggregate(question, present)
    if base.get("verdict") in {"aucun_support"}:
        return base
    stale_only = False
    for aid in question.get("supporting_assertions") or []:
        situation = _situation_for_assertion(aid, situations)
        traj_id = (situation or {}).get("trajectory")
        if not traj_id:
            continue
        traj_assertions = _trajectory_assertions(traj_id, trajectories)
        support_injected_flag, _ = present(aid)
        if not support_injected_flag:
            continue
        for other in traj_assertions:
            if other == aid:
                continue
            other_injected, _ = present(other)
            if other_injected:
                stale_only = True
                break
    if stale_only and base.get("verdict") == "reussi":
        base["version_perimee_seule"] = True
    return base


def verdict_trajectory(
    question: dict[str, Any],
    present: PresentFn,
    *,
    holder_table: dict[str, list[dict[str, Any]]],
    claims_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = verdict_aggregate(question, present)
    if base.get("verdict") == "aucun_support":
        return base
    dates = []
    for item in base.get("supports", []):
        if not item.get("injecte"):
            continue
        for holder in holder_table.get(item["assertion"], []):
            claim = claims_index.get(holder["claim_id"])
            if claim and claim_has_ordering_date(claim):
                dates.append(extract_ordering_date(claim))
    base["ordre_restituable"] = (
        len(dates) == sum(1 for s in base.get("supports", []) if s.get("injecte"))
        and len(dates) >= 2
        and dates == sorted(dates)
    )
    return base


SHAPES_JURIDIQUES_AGGREGATE = frozenset(
    {"citation", "date", "delai", "montant", "chronologie", "qualification"}
)


def verdict_juridique_support(question: dict[str, Any], present: PresentFn) -> dict[str, Any]:
    supports = question.get("supporting_assertions") or []
    if not supports:
        return {"verdict": "absent", "supports": []}
    found = []
    for aid in supports:
        injected, rank = present(aid)
        found.append({"assertion": aid, "injecte": injected, "rang": rank})
    n_found = sum(1 for item in found if item["injecte"])
    if n_found == len(supports):
        verdict = "complet"
    elif n_found > 0:
        verdict = "partiel"
    else:
        verdict = "absent"
    return {"verdict": verdict, "trouves": n_found, "attendus": len(supports), "supports": found}


def _piece_camp(piece: str) -> str:
    lowered = piece.lower()
    if any(token in lowered for token in ("defense", "conclusions", "employeur")):
        return "employeur"
    if any(token in lowered for token in ("assignation", "requete", "releve", "attestation", "salarie")):
        return "salariee"
    return "neutre"


def contradiction_cotes_for_question(
    question: dict[str, Any],
    gold: dict[str, Any],
) -> tuple[list[list[str]], str]:
    """Dérive deux côtés d'une contradiction MOREAU à partir des pièces."""
    assertion_index = gold["assertion_index"]
    aids = list(question.get("supporting_assertions") or [])
    if not aids:
        return [], "aucune assertion de soutien"
    camps: dict[str, list[str]] = {"employeur": [], "salariee": [], "neutre": []}
    for aid in aids:
        piece = piece_basename(str(assertion_index[aid].get("source") or ""))
        camps[_piece_camp(piece)].append(aid)
    if camps["employeur"] and camps["salariee"]:
        return [camps["salariee"], camps["employeur"]], (
            "côtés dérivés des pièces salariée vs employeur"
        )
    by_piece: dict[str, list[str]] = {}
    for aid in aids:
        piece = piece_basename(str(assertion_index[aid].get("source") or ""))
        by_piece.setdefault(piece, []).append(aid)
    pieces = sorted(by_piece)
    if len(pieces) >= 2:
        return [by_piece[pieces[0]], sum((by_piece[p] for p in pieces[1:]), [])], (
            f"côtés dérivés des pièces {pieces[0]!r} vs {pieces[1:]!r}"
        )
    mid = max(1, len(aids) // 2)
    return [aids[:mid], aids[mid:]], "côtés dérivés par moitié des assertions"


def verdict_contradiction_juridique(
    question: dict[str, Any],
    present: PresentFn,
    *,
    gold: dict[str, Any],
) -> dict[str, Any]:
    cotes, derivation = contradiction_cotes_for_question(question, gold)
    if len(cotes) < 2 or not cotes[0] or not cotes[1]:
        return {"verdict": "aucun", "derivation": derivation, "sides": []}
    sides_out = []
    cotes_presentes = 0
    for group in cotes:
        injected = []
        for aid in group:
            ok, rank = present(aid)
            injected.append({"assertion": aid, "injecte": ok, "rang": rank})
        if any(item["injecte"] for item in injected):
            cotes_presentes += 1
        sides_out.append(injected)
    if cotes_presentes == 0:
        verdict = "aucun"
    elif cotes_presentes == 1:
        verdict = "un_seul_cote"
    else:
        verdict = "deux_cotes"
    flat_sides = []
    for group in sides_out:
        for item in group:
            flat_sides.append(item)
    return {
        "verdict": verdict,
        "derivation": derivation,
        "cotes_injectes": cotes_presentes,
        "cotes_total": 2,
        "sides": flat_sides,
    }


def verdict_non_etabli(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": "hors_etage2",
        "question_id": question.get("id"),
        "shape": question.get("shape"),
    }


def verdict_unknown(question: dict[str, Any], evidence: list[dict[str, Any]], present: PresentFn) -> dict[str, Any]:
    supports = question.get("supporting_assertions") or []
    injected = []
    for aid in supports:
        ok, rank = present(aid)
        if ok:
            injected.append({"assertion": aid, "rang": rank})
    scores = [float(e.get("score") or 0.0) for e in evidence]
    return {
        "verdict": "non_mesurable_etage2",
        "n_evidence": len(evidence),
        "meilleur_score": max(scores) if scores else None,
        "supports_injectes": injected,
    }


def compute_verdict(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    gold: dict[str, Any],
    holder_table: dict[str, list[dict[str, Any]]],
    claims_index: dict[str, dict[str, Any]],
    present: PresentFn,
) -> dict[str, Any]:
    shape = question.get("shape") or "unknown"
    if shape == "single":
        return verdict_single(question, present)
    if shape == "aggregate":
        return verdict_aggregate(question, present)
    if shape == "dispute":
        return verdict_dispute(question, present, disputes=gold["disputes"])
    if shape == "status":
        return verdict_status(
            question,
            present,
            situations=gold["situations"],
            trajectories=gold["trajectories"],
        )
    if shape == "trajectory":
        return verdict_trajectory(
            question,
            present,
            holder_table=holder_table,
            claims_index=claims_index,
        )
    if shape in SHAPES_JURIDIQUES_AGGREGATE:
        return verdict_juridique_support(question, present)
    if shape == "contradiction":
        return verdict_contradiction_juridique(question, present, gold=gold)
    if shape == "non_etabli":
        return verdict_non_etabli(question)
    return verdict_unknown(question, evidence, present)
