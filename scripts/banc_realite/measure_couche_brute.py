# SPDX-License-Identifier: AGPL-3.0-only
"""Ablation étage 2 — couche brute vs fine vs composite."""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from banc_realite.compat_moteur import _normalize_evidence_item, _render_markdown

from .config import BANC_ROOT, GOLD, SORTIES, VAULT_REF
from .couche_brute_search import SEARCH_BY_CONFIG
from .fidelite import load_gold
from .holders import (
    PieceCache,
    assertion_present_in_evidence,
    build_holder_table,
    classify_pair,
    collect_appui_states,
    format_claim_triplet,
    noise_count,
    overlap_index,
)
from .measure_etage2 import EMBED_CACHE_PATH, aggregate_entries
from .normalize import claim_text, piece_basename
from .restitution import compute_verdict

COUCHE_BRUTE_SORTIES = BANC_ROOT / "sorties" / "couche-brute"
CONFIGS = ("brute_seule", "fine_seule", "brute_fine")
KS = (8, 20)


def load_questions() -> list[dict[str, Any]]:
    blob = json.loads((GOLD / "questions.json").read_text(encoding="utf-8"))
    return blob["questions"]


def _load_embed_cache() -> dict[str, list[float]]:
    if not EMBED_CACHE_PATH.is_file():
        raise RuntimeError("cache froid : etage2-question-embeddings.json absent")
    return json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))


def _read_public_claims(connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path, "
        "source_span, extracted_by_skill, lifecycle_status "
        "FROM public.claims WHERE lifecycle_status = 'active' AND vault_ref = %s "
        "ORDER BY created_at, claim_id",
        (VAULT_REF,),
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["text"] = claim_text(item.get("value"))
        item["piece"] = piece_basename(item.get("source_file_path") or "")
        out.append(item)
    return out


def _make_present_fn(evidence, assertion_index, claims_index, mode):
    def present(aid: str) -> tuple[bool, int | None]:
        return assertion_present_in_evidence(assertion_index[aid], evidence, claims_index, mode)

    return present


def estimate_injected_tokens(evidence: list[dict[str, Any]]) -> float:
    """Tokens sur le rendu CallFrame réellement injecté (600 char/item, 8000 total)."""
    normalized: list[dict[str, Any]] = []
    for item in evidence:
        normalized.append(_normalize_evidence_item(item))
    rendered = _render_markdown(
        "banc-realite",
        "GreenLoop",
        "banc-tokens-estimate",
        None,
        {"responsibility_context": "GreenLoop"},
        normalized,
        None,
    )
    return round(len(rendered) / 4, 2)


def _estimate_tokens(evidence: list[dict[str, Any]]) -> float:
    return estimate_injected_tokens(evidence)


def _collect_paires(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    cache: PieceCache,
) -> list[dict[str, Any]]:
    paires: list[dict[str, Any]] = []
    for aid in question.get("supporting_assertions") or []:
        assertion = assertion_index[aid]
        if assertion_present_in_evidence(assertion, evidence, claims_index, "plancher")[0]:
            continue
        piece = piece_basename(assertion.get("source") or "")
        seen_claims: set[str] = set()
        for item in evidence:
            if classify_pair(assertion, item, claims_index) != "candidat":
                continue
            claim_id = str(item.get("id") or "")
            if claim_id in seen_claims:
                continue
            seen_claims.add(claim_id)
            claim = claims_index.get(claim_id) or {}
            paires.append(
                {
                    "assertion_id": aid,
                    "claim_id": claim_id,
                    "overlap": overlap_index(assertion, claim),
                    "question_id": question["id"],
                    "assertion": {
                        "id": aid,
                        "piece": piece,
                        "content": assertion.get("content"),
                        "quote": assertion.get("quote"),
                        "extrait": cache.excerpt(piece, assertion.get("quote") or ""),
                    },
                    "fait": {
                        "claim_id": claim_id,
                        "triplet": format_claim_triplet(claim),
                    },
                }
            )
    return paires


def _dedupe_paires(paires: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in paires:
        key = (row["assertion_id"], row["claim_id"])
        if key not in merged:
            merged[key] = {
                "assertion": row["assertion"],
                "fait": row["fait"],
                "overlap": row["overlap"],
                "questions": [row["question_id"]],
            }
        elif row["question_id"] not in merged[key]["questions"]:
            merged[key]["questions"].append(row["question_id"])
    out = list(merged.values())
    out.sort(key=lambda r: (r["assertion"]["id"], -r["overlap"]))
    return out


def _measure_block(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    gold: dict[str, Any],
    holder_table: dict[str, list[dict[str, Any]]],
    claims_index: dict[str, dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    support_ids = question.get("supporting_assertions") or []
    appuis = {}
    for aid in support_ids:
        appuis[aid] = collect_appui_states(
            gold["assertion_index"][aid], evidence, claims_index
        )
    verdicts = {}
    for mode in ("plancher", "plafond"):
        present = _make_present_fn(evidence, gold["assertion_index"], claims_index, mode)
        verdicts[mode] = compute_verdict(
            question,
            evidence,
            gold=gold,
            holder_table=holder_table,
            claims_index=claims_index,
            present=present,
        )
    return {
        f"verdict_plancher": verdicts["plancher"],
        f"verdict_plafond": verdicts["plafond"],
        "appuis": appuis,
        "n_evidence": len(evidence),
        "bruit_plancher": noise_count(
            evidence,
            support_ids,
            assertion_index=gold["assertion_index"],
            claims_index=claims_index,
            mode="plancher",
        ),
        "bruit_plafond": noise_count(
            evidence,
            support_ids,
            assertion_index=gold["assertion_index"],
            claims_index=claims_index,
            mode="plafond",
        ),
        "kinds": dict(Counter(item.get("kind") or "unknown" for item in evidence)),
        "tokens_estimes": _estimate_tokens(evidence),
        "evidence_ids": [e.get("id") for e in evidence],
    }


def _aggregate_config(entries: list[dict[str, Any]]) -> dict[str, Any]:
    base = aggregate_entries(entries)
    tokens_k8 = [e["k8"]["tokens_estimes"] for e in entries]
    tokens_k20 = [e["k20"]["tokens_estimes"] for e in entries]
    return {
        **base,
        "tokens_k8": {
            "total": round(sum(tokens_k8), 2),
            "mediane": round(statistics.median(tokens_k8), 2) if tokens_k8 else 0,
        },
        "tokens_k20": {
            "total": round(sum(tokens_k20), 2),
            "mediane": round(statistics.median(tokens_k20), 2) if tokens_k20 else 0,
        },
    }


def _success_rate(agg: dict[str, Any], k: int) -> float:
    data = agg.get(f"par_forme_difficulte_k{k}") or {}
    total = 0
    ok = 0
    for shape in data:
        for diff in data[shape]:
            for verdict, count in data[shape][diff].get("plancher", {}).items():
                total += count
                if verdict == "reussi":
                    ok += count
    return ok / total if total else 0.0


def _shape_success_rates(agg: dict[str, Any], k: int) -> dict[str, float]:
    data = agg.get(f"par_forme_difficulte_k{k}") or {}
    out: dict[str, float] = {}
    for shape in data:
        total = 0
        ok = 0
        for diff in data[shape]:
            for verdict, count in data[shape][diff].get("plancher", {}).items():
                total += count
                if verdict == "reussi":
                    ok += count
        if total:
            out[shape] = ok / total
    return out


def build_avant_apres(
    avant: dict[str, Any],
    apres: dict[str, Any],
) -> dict[str, Any]:
    shapes: set[str] = set()
    for cfg in CONFIGS:
        for blob in (avant, apres):
            agg = blob.get("configurations", {}).get(cfg, {}).get("aggregates") or {}
            shapes.update((agg.get("par_forme_difficulte_k8") or {}).keys())
    table: dict[str, Any] = {}
    for shape in sorted(shapes):
        row: dict[str, Any] = {}
        for config in CONFIGS:
            av_agg = avant["configurations"][config]["aggregates"]
            ap_agg = apres["configurations"][config]["aggregates"]
            av_rate = _shape_success_rates(av_agg, 8).get(shape)
            ap_rate = _shape_success_rates(ap_agg, 8).get(shape)
            row[config] = {
                "k8_plancher_reussi_avant": av_rate,
                "k8_plancher_reussi_apres": ap_rate,
                "k8_plancher_delta": (
                    round(ap_rate - av_rate, 4) if av_rate is not None and ap_rate is not None else None
                ),
                "tokens_k8_total_avant": av_agg["tokens_k8"]["total"],
                "tokens_k8_total_apres": ap_agg["tokens_k8"]["total"],
                "tokens_k8_mediane_avant": av_agg["tokens_k8"]["mediane"],
                "tokens_k8_mediane_apres": ap_agg["tokens_k8"]["mediane"],
            }
        table[shape] = row
    return {"par_forme_k8": table, "configs": list(CONFIGS)}


def measure_ablation(connection, *, avant_path: Path | None = None) -> dict[str, Any]:
    embed_cache = _load_embed_cache()
    gold = load_gold()
    claims = _read_public_claims(connection)
    claims_index = {str(c["claim_id"]): c for c in claims}
    cache = PieceCache()
    holder_table, _, _ = build_holder_table(claims, gold, cache=cache)
    questions = load_questions()

    by_config: dict[str, list[dict[str, Any]]] = {cfg: [] for cfg in CONFIGS}
    paires_by_config: dict[str, list[dict[str, Any]]] = {cfg: [] for cfg in CONFIGS}

    for question in questions:
        qtext = question["question"]
        if qtext not in embed_cache:
            raise RuntimeError(f"cache froid : embedding absent ({qtext[:60]}…)")
        qvec = embed_cache[qtext]
        entry_base = {
            "id": question["id"],
            "shape": question.get("shape"),
            "difficulty": question.get("difficulty"),
        }
        for config in CONFIGS:
            entry = dict(entry_base)
            search_fn = SEARCH_BY_CONFIG[config]
            for k in KS:
                evidence = search_fn(
                    connection,
                    qtext,
                    query_vector=qvec,
                    top_k=k,
                )
                entry[f"k{k}"] = _measure_block(
                    question,
                    evidence,
                    gold=gold,
                    holder_table=holder_table,
                    claims_index=claims_index,
                    k=k,
                )
                if k == 20:
                    paires_by_config[config].extend(
                        _collect_paires(
                            question,
                            evidence,
                            assertion_index=gold["assertion_index"],
                            claims_index=claims_index,
                            cache=cache,
                        )
                    )
            by_config[config].append(entry)

    callframe_path = SORTIES / "etage2-C.json"
    callframe_ref = None
    if callframe_path.is_file():
        callframe_ref = json.loads(callframe_path.read_text(encoding="utf-8"))

    result = {
        "status": "complete",
        "questions": len(questions),
        "configurations": {},
        "callframe_reference": {
            "path": str(callframe_path.relative_to(BANC_ROOT)) if callframe_path.is_file() else None,
            "note": (
                "La CallFrame complète (etage2-C) inclut claims_hybrid + file_hit lexicaux "
                "+ asset_hit ; « fine seule » ici = hybrid claims uniquement, sans passages "
                "ni assets."
            ),
            "aggregates_k8": (callframe_ref or {}).get("aggregates", {}).get("par_forme_difficulte_k8"),
            "aggregates_k20": (callframe_ref or {}).get("aggregates", {}).get("par_forme_difficulte_k20"),
        },
    }
    for config in CONFIGS:
        entries = by_config[config]
        result["configurations"][config] = {
            "questions": entries,
            "aggregates": _aggregate_config(entries),
            "paires_a_lire": _dedupe_paires(paires_by_config[config]),
        }

    out_path = COUCHE_BRUTE_SORTIES / "ablation.json"
    COUCHE_BRUTE_SORTIES.mkdir(parents=True, exist_ok=True)
    avant_source = avant_path or out_path
    if avant_source.is_file():
        avant_blob = json.loads(avant_source.read_text(encoding="utf-8"))
        avant_copy = COUCHE_BRUTE_SORTIES / "ablation-avant.json"
        if avant_source != avant_copy:
            avant_copy.write_text(json.dumps(avant_blob, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        apres_table = build_avant_apres(avant_blob, result)
        apres_path = COUCHE_BRUTE_SORTIES / "ablation-avant-apres.json"
        apres_path.write_text(json.dumps(apres_table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["avant_apres"] = str(apres_path.relative_to(BANC_ROOT))
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
