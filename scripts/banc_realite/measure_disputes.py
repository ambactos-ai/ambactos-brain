# SPDX-License-Identifier: AGPL-3.0-only
"""Mesure disputes — confusion or, étage 2 rejoué, coût tokens."""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from banc_realite.compat_moteur import _build_structured_query, _normalize_evidence_item

from .classer_couple import classer_couple
from .completion_structurelle import enrichir_evidence_completion
from .config import BANC_ROOT, GOLD, SORTIES, VAULT_REF, export_runtime_env
from .dossiers import Dossier, dossier_courant
from .couche_brute_search import SEARCH_BY_CONFIG
from .fidelite import load_gold, match_assertion
from .holders import build_holder_table, assertion_present_in_evidence
from .measure_couche_brute import estimate_injected_tokens, load_questions
from .measure_etage2 import EMBED_CACHE_PATH
from .normalize import claim_text, piece_basename
from .restitution import compute_verdict, dispute_sides_for_question
from .trois_tours import COUCHE_TO_CONFIG, recherche_trois_tours

SORTIES_DISPUTES = BANC_ROOT / "sorties" / "disputes"
CONFUSION_PATH = SORTIES_DISPUTES / "confusion.json"
AVANT_APRES_PATH = SORTIES_DISPUTES / "etage2-avant-apres.json"
ABLATION_PATH = SORTIES_DISPUTES / "ablation-mecanisme.json"
QUESTIONS_DETAIL_PATH = SORTIES_DISPUTES / "questions-disputes-detail.json"
RAPPORT_PATH = SORTIES_DISPUTES / "RAPPORT-DISPUTES.md"
ARRET_PATH = BANC_ROOT / "_disputes-arret.json"
RESULT_PATH = BANC_ROOT / "_disputes-result.json"
VERIF_ARRET_PATH = BANC_ROOT / "_disputes-verif-arret.json"
VERIF_RESULT_PATH = BANC_ROOT / "_disputes-verif-result.json"

COUCHES = ("fine", "brute", "brute_fine")
KS = (8, 20)
MODES = ("temoin", "trois_tours", "completion", "trois_tours_completion")
REFERENCE_ETAGE3_BEST = 9
CAUSE_ZERO_BRUTE = "alternance_round_robin_ejecte_passages_tour1"


def write_arret(step: str, why: str, *, verif: bool = False) -> None:
    path = VERIF_ARRET_PATH if verif else ARRET_PATH
    path.write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _empty_verdict_counts() -> dict[str, int]:
    return {"complet": 0, "un_seul_cote": 0, "partiel": 0, "aucun": 0, "autre": 0}


def _count_verdict(verdict: str, bucket: dict[str, int]) -> None:
    key = verdict if verdict in bucket else "autre"
    bucket[key] += 1


def _search_raw(
    connection,
    question: dict[str, Any],
    *,
    couche: str,
    k: int,
    embed_cache: dict[str, list[float]],
    vivier: int | None = None,
    dossier: Dossier | None = None,
) -> list[dict[str, Any]]:
    current = dossier if dossier is not None else dossier_courant()
    qtext = question["question"]
    structured = _build_structured_query(
        qtext,
        current.responsibility_context,
        {"responsibility_context": current.responsibility_context},
    )
    config = COUCHE_TO_CONFIG[couche]
    kwargs: dict[str, Any] = {
        "query_vector": embed_cache[qtext],
        "top_k": k,
    }
    if dossier is not None:
        kwargs["vault_ref"] = current.vault_ref
    if vivier is not None:
        kwargs["vivier"] = vivier
    return SEARCH_BY_CONFIG[config](connection, structured, **kwargs)


def _search_simple(
    connection,
    question: dict[str, Any],
    *,
    couche: str,
    k: int,
    embed_cache: dict[str, list[float]],
    dossier: Dossier | None = None,
) -> list[dict[str, Any]]:
    return [
        _normalize_evidence_item(item)
        for item in _search_raw(
            connection,
            question,
            couche=couche,
            k=k,
            embed_cache=embed_cache,
            dossier=dossier,
        )
    ]


def _normalize_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_evidence_item(item) for item in evidence]


def build_evidence_for_mode(
    connection,
    question: dict[str, Any],
    *,
    couche: str,
    k: int,
    mode: str,
    embed_cache: dict[str, list[float]],
    budget: int | None = None,
    n_tours: int = 3,
    fusion: str = "alternance",
    k_tour1: int | None = None,
    vivier: int | None = None,
    dossier: Dossier | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in MODES:
        raise ValueError(f"mode inconnu: {mode}")
    qtext = question["question"]
    if qtext not in embed_cache:
        raise RuntimeError(f"cache froid : embedding absent ({qtext[:60]}…)")

    meta: dict[str, Any] = {
        "mode": mode,
        "tour2_vide": None,
        "completion": {"faits_ajoutes": 0, "faits_depuis_autre_piece": 0},
    }
    use_tours = mode in ("trois_tours", "trois_tours_completion")
    use_completion = mode in ("completion", "trois_tours_completion")

    if use_tours:
        result = recherche_trois_tours(
            qtext,
            k,
            couche,
            connection=connection,
            query_vector=embed_cache[qtext],
            n_tours=n_tours,
            fusion=fusion,
            k_tour1=k_tour1,
            vivier=vivier,
            dossier=dossier,
        )
        raw = list(result["raw_evidence"])
        meta["tour2_vide"] = bool(result["parametres"]["tour2_vide"])
    else:
        raw = _search_raw(
            connection,
            question,
            couche=couche,
            k=k,
            embed_cache=embed_cache,
            vivier=vivier,
            dossier=dossier,
        )

    if use_completion:
        raw, completion_stats = enrichir_evidence_completion(
            raw, connection=connection, budget=budget
        )
        meta["completion"] = completion_stats

    return _normalize_evidence(raw), meta


def _load_embed_cache() -> dict[str, list[float]]:
    if not EMBED_CACHE_PATH.is_file():
        raise RuntimeError("cache froid : etage2-question-embeddings.json absent")
    cache = json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))
    if not cache:
        raise RuntimeError("cache froid : etage2-question-embeddings.json vide")
    return cache


def _read_public_claims(connection, vault_ref: str | None = None) -> list[dict[str, Any]]:
    effective = vault_ref if vault_ref is not None else VAULT_REF
    rows = connection.execute(
        "SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path, "
        "source_span, extracted_by_skill, lifecycle_status "
        "FROM public.claims WHERE lifecycle_status = 'active' AND vault_ref = %s "
        "ORDER BY created_at, claim_id",
        (effective,),
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["text"] = claim_text(item.get("value"))
        item["piece"] = piece_basename(item.get("source_file_path") or "")
        out.append(item)
    return out


def _claims_by_piece(claims: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_piece: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        by_piece[claim["piece"]].append(claim)
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


def _pair_from_dispute(dispute: dict[str, Any], gold: dict[str, Any], by_piece: dict[str, list]) -> dict[str, Any] | None:
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
            result = classer_couple(claims[i], claims[j])
            if result["issue"] != "accord":
                return result
    return classer_couple(claims[0], claims[1])


def _pair_from_trajectory(trajectory: dict[str, Any], gold: dict[str, Any], by_piece: dict[str, list]) -> dict[str, Any] | None:
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
    return classer_couple(_enrich_from_version(left_c, left_v), _enrich_from_version(right_c, right_v))


def _pair_from_reversibility(case: dict[str, Any], gold: dict[str, Any], by_piece: dict[str, list]) -> dict[str, Any] | None:
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
    return classer_couple(left_c, right_c)


def build_confusion_matrix(gold: dict[str, Any], by_piece: dict[str, list]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    matrix: dict[str, Counter] = {
        "dispute": Counter(),
        "trajectoire": Counter(),
        "reversibility": Counter(),
    }
    indecidable = 0
    total = 0

    for dispute in gold.get("disputes", []):
        result = _pair_from_dispute(dispute, gold, by_piece)
        issue = result["issue"] if result else "indecidable"
        rows.append({"id": dispute["id"], "gold": "dispute", "issue": issue, "raison": (result or {}).get("raison")})
        matrix["dispute"][issue] += 1
        total += 1
        if issue == "indecidable":
            indecidable += 1

    for trajectory in gold.get("trajectories", []):
        result = _pair_from_trajectory(trajectory, gold, by_piece)
        issue = result["issue"] if result else "indecidable"
        rows.append({"id": trajectory["id"], "gold": "trajectoire", "issue": issue, "raison": (result or {}).get("raison")})
        matrix["trajectoire"][issue] += 1
        total += 1
        if issue == "indecidable":
            indecidable += 1

    for case in gold.get("reversibility_cases", []):
        result = _pair_from_reversibility(case, gold, by_piece)
        issue = result["issue"] if result else "indecidable"
        rows.append({"id": case["id"], "gold": "reversibility", "issue": issue, "raison": (result or {}).get("raison")})
        matrix["reversibility"][issue] += 1
        total += 1
        if issue == "indecidable":
            indecidable += 1

    part_indecidable = round(indecidable / total, 4) if total else 0.0
    return {
        "details": rows,
        "matrice": {gold_type: dict(counter) for gold_type, counter in matrix.items()},
        "part_indecidable": part_indecidable,
        "total": total,
        "indecidable": indecidable,
        "pieges_reversibility_en_dispute": sum(
            1 for row in rows if row["gold"] == "reversibility" and row["issue"] == "dispute"
        ),
    }


def _verdict_dispute_etage2(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
) -> str:
    def present(aid: str) -> tuple[bool, int | None]:
        assertion = gold["assertion_index"][aid]
        return assertion_present_in_evidence(assertion, evidence, claims_index, "plancher")

    block = compute_verdict(
        question,
        evidence,
        gold=gold,
        holder_table={},
        claims_index=claims_index,
        present=present,
    )
    return str(block.get("verdict") or "unknown")


def measure_etage2_disputes(
    connection,
    *,
    embed_cache: dict[str, list[float]],
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dispute_questions = [q for q in load_questions() if q.get("shape") == "dispute"]
    table: dict[str, Any] = {}
    tokens_trois: dict[str, list[float]] = defaultdict(list)
    tokens_simple: dict[str, list[float]] = defaultdict(list)

    for couche in COUCHES:
        table[couche] = {}
        for k in KS:
            sans = {"complet": 0, "un_seul_cote": 0, "partiel": 0, "aucun": 0, "autre": 0}
            avec = dict(sans)
            for question in dispute_questions:
                qid = question["id"]
                qtext = question["question"]
                if qtext not in embed_cache:
                    raise RuntimeError(f"cache froid : embedding absent ({qtext[:60]}…)")
                ev_simple = _search_simple(connection, question, couche=couche, k=k, embed_cache=embed_cache)
                ev_trois = recherche_trois_tours(
                    qtext,
                    k,
                    couche,
                    connection=connection,
                    query_vector=embed_cache[qtext],
                )["evidence"]
                v_simple = _verdict_dispute_etage2(
                    question, ev_simple, gold=gold, claims_index=claims_index
                )
                v_trois = _verdict_dispute_etage2(
                    question, ev_trois, gold=gold, claims_index=claims_index
                )
                key_simple = v_simple if v_simple in sans else "autre"
                key_trois = v_trois if v_trois in avec else "autre"
                sans[key_simple] += 1
                avec[key_trois] += 1
                tokens_simple[f"{couche}:k{k}"].append(estimate_injected_tokens(ev_simple))
                tokens_trois[f"{couche}:k{k}"].append(estimate_injected_tokens(ev_trois))
            table[couche][f"k{k}"] = {
                "sans_trois_tours": sans,
                "avec_trois_tours": avec,
                "complet_sans": sans["complet"],
                "complet_avec": avec["complet"],
                "delta_complet": avec["complet"] - sans["complet"],
                "tokens_sans": {
                    "total": round(sum(tokens_simple[f"{couche}:k{k}"]), 2),
                    "mediane": round(statistics.median(tokens_simple[f"{couche}:k{k}"]), 2),
                },
                "tokens_avec": {
                    "total": round(sum(tokens_trois[f"{couche}:k{k}"]), 2),
                    "mediane": round(statistics.median(tokens_trois[f"{couche}:k{k}"]), 2),
                },
            }
    return {
        "questions": len(dispute_questions),
        "reference_etage3_meilleur": REFERENCE_ETAGE3_BEST,
        "par_couche": table,
    }


def _side_presence(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
    mode_presence: str = "verbatim",
) -> list[dict[str, Any]]:
    def present(aid: str) -> tuple[bool, int | None]:
        assertion = gold["assertion_index"][aid]
        return assertion_present_in_evidence(
            assertion,
            evidence,
            claims_index,
            "plancher",
            mode_presence=mode_presence,
        )

    block = compute_verdict(
        question,
        evidence,
        gold=gold,
        holder_table={},
        claims_index=claims_index,
        present=present,
    )
    sides_out: list[dict[str, Any]] = []
    for side in block.get("sides") or []:
        aid = side["assertion"]
        assertion = gold["assertion_index"][aid]
        sides_out.append(
            {
                "assertion": aid,
                "piece": piece_basename(assertion.get("source") or ""),
                "dans_contexte": bool(side.get("injecte")),
                "rang": side.get("rang"),
            }
        )
    return sides_out


def compute_completion_deltas(par_couche: dict[str, Any]) -> dict[str, int]:
    """Delta de complétion calculé une fois à partir des totaux finaux (Z3)."""
    completion_vs_temoin = 0
    trois_tours_completion_vs_trois_tours = 0
    for couche in COUCHES:
        for k in KS:
            block = par_couche[couche][f"k{k}"]
            completion_vs_temoin += block["completion"]["complet"] - block["temoin"]["complet"]
            trois_tours_completion_vs_trois_tours += (
                block["trois_tours_completion"]["complet"] - block["trois_tours"]["complet"]
            )
    return {
        "completion_vs_temoin": completion_vs_temoin,
        "trois_tours_completion_vs_trois_tours": trois_tours_completion_vs_trois_tours,
    }


def count_tour2_vides(
    connection,
    *,
    embed_cache: dict[str, list[float]],
    shape: str,
    limit: int = 28,
    couche: str = "brute_fine",
    k: int = 8,
) -> dict[str, Any]:
    """Compte les tours 2 vides pour un shape donné (dispute ou single)."""
    questions = [q for q in load_questions() if q.get("shape") == shape][:limit]
    vides: list[str] = []
    for question in questions:
        qtext = question["question"]
        if qtext not in embed_cache:
            raise RuntimeError(f"cache froid : embedding absent ({qtext[:60]}…)")
        result = recherche_trois_tours(
            qtext,
            k,
            couche,
            connection=connection,
            query_vector=embed_cache[qtext],
        )
        if result["parametres"]["tour2_vide"]:
            vides.append(question["id"])
    return {
        "shape": shape,
        "total": len(questions),
        "tour2_vides": len(vides),
        "tour2_vides_questions": vides,
        "couche": couche,
        "k": k,
    }


def measure_ablation_mecanisme(
    connection,
    *,
    embed_cache: dict[str, list[float]],
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dispute_questions = [q for q in load_questions() if q.get("shape") == "dispute"]
    table: dict[str, Any] = {}
    tour2_vides_questions: set[str] = set()
    completion_total = {"faits_ajoutes": 0, "faits_depuis_autre_piece": 0}

    for couche in COUCHES:
        table[couche] = {}
        for k in KS:
            counts = {mode: _empty_verdict_counts() for mode in MODES}
            complet_by_mode = {mode: 0 for mode in MODES}
            for question in dispute_questions:
                for mode in MODES:
                    evidence, meta = build_evidence_for_mode(
                        connection,
                        question,
                        couche=couche,
                        k=k,
                        mode=mode,
                        embed_cache=embed_cache,
                    )
                    verdict = _verdict_dispute_etage2(
                        question, evidence, gold=gold, claims_index=claims_index
                    )
                    _count_verdict(verdict, counts[mode])
                    if verdict == "complet":
                        complet_by_mode[mode] += 1
                    if (
                        mode == "trois_tours"
                        and couche == "brute_fine"
                        and k == 8
                        and meta.get("tour2_vide")
                    ):
                        tour2_vides_questions.add(question["id"])
                    if mode in ("completion", "trois_tours_completion"):
                        completion_total["faits_ajoutes"] += meta["completion"]["faits_ajoutes"]
                        completion_total["faits_depuis_autre_piece"] += meta["completion"][
                            "faits_depuis_autre_piece"
                        ]
            table[couche][f"k{k}"] = {
                mode: {**counts[mode], "complet": complet_by_mode[mode]} for mode in MODES
            }

    completion_delta_complet = compute_completion_deltas(table)

    chiffre_tranche = (
        f"Complétion : {completion_total['faits_ajoutes']} faits ajoutés "
        f"({completion_total['faits_depuis_autre_piece']} depuis une autre pièce) ; "
        f"Δ complet completion−témoin={completion_delta_complet['completion_vs_temoin']}, "
        f"Δ trois_tours_completion−trois_tours="
        f"{completion_delta_complet['trois_tours_completion_vs_trois_tours']} "
        f"(sur {len(dispute_questions)*len(COUCHES)*len(KS)} mesures)."
    )
    return {
        "questions": len(dispute_questions),
        "modes": list(MODES),
        "couches": list(COUCHES),
        "ks": list(KS),
        "par_couche": table,
        "completion_agregat": completion_total,
        "completion_delta_complet": completion_delta_complet,
        "tour2_vides": len(tour2_vides_questions),
        "tour2_vides_questions": sorted(tour2_vides_questions),
        "chiffre_tranche_completion": chiffre_tranche,
        "cause_zero_brute": {
            "nom": CAUSE_ZERO_BRUTE,
            "nature": "defaut",
            "description": (
                "L'alternance round-robin sur des passages (file_hit) éjecte les meilleurs "
                "éléments du tour 1 au profit de passages d'autres pièces souvent hors sujet."
            ),
            "apres_correction": table["brute"],
        },
    }


def build_questions_detail(
    connection,
    *,
    embed_cache: dict[str, list[float]],
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    dispute_questions = [q for q in load_questions() if q.get("shape") == "dispute"]
    rows: list[dict[str, Any]] = []
    for question in dispute_questions:
        side_ids = dispute_sides_for_question(question, gold["disputes"])
        sides_gold = []
        for aid in side_ids:
            assertion = gold["assertion_index"][aid]
            sides_gold.append({"assertion": aid, "piece": piece_basename(assertion.get("source") or "")})
        row: dict[str, Any] = {
            "id": question["id"],
            "question": question["question"],
            "cotes_or": sides_gold,
            "par_mode_k8": {},
            "par_mode_k20": {},
        }
        for k in KS:
            mode_block: dict[str, Any] = {}
            for mode in MODES:
                evidence, meta = build_evidence_for_mode(
                    connection,
                    question,
                    couche="brute_fine",
                    k=k,
                    mode=mode,
                    embed_cache=embed_cache,
                )
                sides = _side_presence(question, evidence, gold=gold, claims_index=claims_index)
                mode_block[mode] = {
                    "verdict": _verdict_dispute_etage2(
                        question, evidence, gold=gold, claims_index=claims_index
                    ),
                    "deux_cotes_dans_contexte": all(s["dans_contexte"] for s in sides) if sides else False,
                    "cotes": sides,
                    "tour2_vide": meta.get("tour2_vide"),
                    "completion": meta.get("completion"),
                }
            target = row["par_mode_k8"] if k == 8 else row["par_mode_k20"]
            target.update(mode_block)
        rows.append(row)
    return rows


def write_report(
    confusion: dict[str, Any],
    etage2: dict[str, Any],
    *,
    ablation: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Rapport — disputes, trois tours, complétion structurelle",
        "",
        f"Date : {date.today().isoformat()}",
        "",
        "## Matrice de confusion (or → règle)",
        "",
        "| Or | Issue prédite | Effectif |",
        "| --- | --- | --- |",
    ]
    for gold_type, counter in confusion["matrice"].items():
        for issue, count in sorted(counter.items()):
            lines.append(f"| {gold_type} | {issue} | {count} |")
    lines.extend(
        [
            "",
            f"**Part d'indécidable** : {confusion['part_indecidable']:.1%} "
            f"({confusion['indecidable']}/{confusion['total']})",
            "",
            f"**Pièges réversibilité classés dispute** : {confusion['pieges_reversibility_en_dispute']}/"
            f"{len([r for r in confusion['details'] if r['gold'] == 'reversibility'])}",
            "",
            "## Étage 2 rejoué — 28 questions dispute (verdict plancher)",
            "",
            "| Couche | k | Complet sans tours | Complet avec tours | Δ | Tokens médian sans | Tokens médian avec |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for couche in COUCHES:
        for k in KS:
            block = etage2["par_couche"][couche][f"k{k}"]
            lines.append(
                f"| {couche} | {k} | {block['complet_sans']}/28 | {block['complet_avec']}/28 | "
                f"{block['delta_complet']:+d} | {block['tokens_sans']['mediane']} | "
                f"{block['tokens_avec']['mediane']} |"
            )
    if ablation:
        lines.extend(
            [
                "",
                "## Ablation — quatre modes (28 disputes, verdict plancher)",
                "",
                f"**Tour 2 vide** : {ablation['tour2_vides']}/28 questions "
                f"(trois_tours, brute_fine, k=8).",
                "",
                f"**Complétion** : {ablation['chiffre_tranche_completion']}",
                "",
                f"**Zéro couche brute** : {ablation['cause_zero_brute']['nom']} ({ablation['cause_zero_brute']['nature']}).",
                "",
                "| Couche | k | témoin | trois_tours | complétion | les deux |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for couche in COUCHES:
            for k in KS:
                block = ablation["par_couche"][couche][f"k{k}"]
                lines.append(
                    f"| {couche} | {k} | {block['temoin']['complet']}/28 | "
                    f"{block['trois_tours']['complet']}/28 | {block['completion']['complet']}/28 | "
                    f"{block['trois_tours_completion']['complet']}/28 |"
                )
    best = max(
        (
            (couche, k, etage2["par_couche"][couche][f"k{k}"]["complet_avec"])
            for couche in COUCHES
            for k in KS
        ),
        key=lambda row: row[2],
    )
    lines.extend(
        [
            "",
            f"Référence étage 3 (meilleur connu) : {REFERENCE_ETAGE3_BEST}/28 disputes avec les deux côtés.",
            "",
            "## Trois phrases",
            "",
            (
                "1. Les trois tours forcent l'alternance entre pièces déjà classées et pièces exclues des tours "
                "précédents : quand le tour 2 revient vide, l'absence confirme qu'une seule pièce porte le sujet."
            ),
            (
                "2. La complétion structurelle complète chaque fait par lecture indexée (même entité, même attribut) "
                "sans top-k ni similarité, pour récupérer les côtés absents du classement vectoriel."
            ),
            (
                f"3. Il reste {confusion['part_indecidable']:.0%} d'indécidable dans l'or mesuré ; "
                f"les trois tours portent le meilleur cas étage 2 à {best[2]}/28 "
                f"(couche {best[0]}, k={best[1]}) contre {REFERENCE_ETAGE3_BEST}/28 à l'étage 3 — "
                "le résidu indécidable et les cas où l'alternance dilue la couche brute seule iraient à Jev."
            ),
        ]
    )
    RAPPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_measurement(connection) -> dict[str, Any]:
    export_runtime_env()
    embed_cache = _load_embed_cache()
    gold = load_gold()
    claims = _read_public_claims(connection)
    claims_index = {str(c["claim_id"]): c for c in claims}
    by_piece = _claims_by_piece(claims)
    build_holder_table(claims, gold)

    SORTIES_DISPUTES.mkdir(parents=True, exist_ok=True)
    confusion = build_confusion_matrix(gold, by_piece)
    etage2 = measure_etage2_disputes(
        connection,
        embed_cache=embed_cache,
        gold=gold,
        claims_index=claims_index,
    )
    CONFUSION_PATH.write_text(json.dumps(confusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AVANT_APRES_PATH.write_text(json.dumps(etage2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(confusion, etage2)
    return {
        "confusion": confusion,
        "etage2": etage2,
        "tests": "tests/banc_realite/test_classer_couple.py tests/banc_realite/test_trois_tours.py tests/banc_realite/test_completion_structurelle.py",
    }


def run_verification(connection) -> dict[str, Any]:
    export_runtime_env()
    embed_cache = _load_embed_cache()
    gold = load_gold()
    claims = _read_public_claims(connection)
    claims_index = {str(c["claim_id"]): c for c in claims}
    by_piece = _claims_by_piece(claims)
    build_holder_table(claims, gold)

    SORTIES_DISPUTES.mkdir(parents=True, exist_ok=True)
    confusion = build_confusion_matrix(gold, by_piece)
    etage2 = measure_etage2_disputes(
        connection,
        embed_cache=embed_cache,
        gold=gold,
        claims_index=claims_index,
    )
    ablation = measure_ablation_mecanisme(
        connection,
        embed_cache=embed_cache,
        gold=gold,
        claims_index=claims_index,
    )
    questions_detail = build_questions_detail(
        connection,
        embed_cache=embed_cache,
        gold=gold,
        claims_index=claims_index,
    )

    CONFUSION_PATH.write_text(json.dumps(confusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AVANT_APRES_PATH.write_text(json.dumps(etage2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ABLATION_PATH.write_text(json.dumps(ablation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    QUESTIONS_DETAIL_PATH.write_text(
        json.dumps(questions_detail, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(confusion, etage2, ablation=ablation)

    tests = (
        "tests/banc_realite/test_classer_couple.py "
        "tests/banc_realite/test_trois_tours.py "
        "tests/banc_realite/test_completion_structurelle.py "
        "tests/banc_realite/test_measure_disputes_ablation.py"
    )
    return {
        "confusion": confusion,
        "etage2": etage2,
        "ablation": ablation,
        "questions_detail": questions_detail,
        "tests": tests,
        "cause_zero_brute": ablation["cause_zero_brute"]["nom"],
    }
