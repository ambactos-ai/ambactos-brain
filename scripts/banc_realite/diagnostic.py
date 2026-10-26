# SPDX-License-Identifier: AGPL-3.0-only
"""B5 attribution for missing supports at k = 20 (plancher)."""
from __future__ import annotations

from typing import Any

from banc_realite.compat_moteur import embed_with_retry, to_pgvector

from .config import VAULT_REF
from .holders import assertion_present_in_evidence, model_claims_on_piece
from .normalize import piece_basename


def _rank_in_list(claim_id: str, rows: list[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(rows, 1):
        row_id = str(row.get("claim_id") or row.get("id") or "")
        if row_id == claim_id:
            return idx
    return None


def _best_rank_in_fused(holder_ids: list[str], fused_ranking: list[dict[str, Any]]) -> int | None:
    ranks = [_rank_in_list(cid, fused_ranking) for cid in holder_ids]
    present = [r for r in ranks if r is not None]
    return min(present) if present else None


def _best_rank(holder_ids: list[str], rows: list[dict[str, Any]]) -> int | None:
    ranks = [_rank_in_list(cid, rows) for cid in holder_ids]
    present = [r for r in ranks if r is not None]
    return min(present) if present else None


def jambe_diagnostic(
    *,
    question_text: str,
    holder_ids: list[str],
    fused_ranking: list[dict[str, Any]],
    connection,
    question_vector: list[float] | None = None,
) -> dict[str, Any]:
    if not holder_ids:
        return {}
    lex_rows = connection.execute(
        "SELECT claim_id, rank FROM public.search_claims_lexical("
        "%s, %s::text[], %s, %s, %s, %s, %s, %s)",
        (
            question_text,
            ["note.body", "relation.about"],
            "active",
            None,
            None,
            VAULT_REF,
            "main",
            100,
        ),
    ).fetchall()
    vector = question_vector if question_vector is not None else embed_with_retry(question_text)
    vec_rows = connection.execute(
        "SELECT claim_id, similarity FROM public.recall_match_mistral("
        "%s::extensions.vector, %s, %s, %s, %s) ORDER BY similarity DESC",
        (to_pgvector(vector), 100, "note.body", None, VAULT_REF),
    ).fetchall()
    lex_rank = _best_rank(holder_ids, [dict(r) for r in lex_rows])
    vec_rank = _best_rank(holder_ids, [dict(r) for r in vec_rows])
    fused_rank = _best_rank_in_fused(holder_ids, fused_ranking)
    in_lex = lex_rank is not None
    in_vec = vec_rank is not None
    if not in_lex and not in_vec:
        jambe = "absent des deux jambes"
    elif fused_rank is None:
        jambe = "present dans une jambe, perdu à la fusion"
    else:
        jambe = "present dans le classement fusionné"
    return {
        "jambe": jambe,
        "rang_lexical": lex_rank,
        "rang_vectoriel": vec_rank,
        "rang_fusionne": fused_rank,
    }


def attribute_missing_support(
    *,
    assertion_id: str,
    question: dict[str, Any],
    evidence_k20: list[dict[str, Any]],
    fused_k50: list[dict[str, Any]],
    holder_table: dict[str, list[dict[str, Any]]],
    claims: list[dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    gold: dict[str, Any],
    connection,
    question_vector: list[float] | None = None,
    k: int = 20,
) -> dict[str, Any]:
    assertion = gold["assertion_index"][assertion_id]
    piece = piece_basename(assertion.get("source") or "")
    present, _ = assertion_present_in_evidence(assertion, evidence_k20, claims_index, "plancher")
    if present:
        return {"assertion": assertion_id, "cause": "present", "rang": None}

    strict_ids = [h["claim_id"] for h in holder_table.get(assertion_id, [])]
    model_on_piece = model_claims_on_piece(claims, piece)

    if not strict_ids and not model_on_piece:
        cause = "jamais en mémoire"
        best_rank = None
    elif not strict_ids:
        cause = "en mémoire, candidat seulement"
        best_rank = None
    else:
        best_rank = _best_rank_in_fused(strict_ids, fused_k50)
        if best_rank is None:
            cause = "en mémoire, hors du classement nu à 50"
        elif best_rank > k:
            cause = "dans le classement nu, coupé par k"
        else:
            cause = "dans le classement nu, écarté à la composition"

    out = {
        "assertion": assertion_id,
        "cause": cause,
        "rang": best_rank,
        "strict_holders": strict_ids,
    }
    out.update(
        jambe_diagnostic(
            question_text=question["question"],
            holder_ids=strict_ids,
            fused_ranking=fused_k50,
            connection=connection,
            question_vector=question_vector,
        )
    )
    return out


def collect_missing_diagnostics(
    question: dict[str, Any],
    evidence_k20: list[dict[str, Any]],
    fused_k50: list[dict[str, Any]],
    *,
    gold: dict[str, Any],
    holder_table: dict[str, list[dict[str, Any]]],
    claims: list[dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    connection,
    question_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for aid in question.get("supporting_assertions") or []:
        diag = attribute_missing_support(
            assertion_id=aid,
            question=question,
            evidence_k20=evidence_k20,
            fused_k50=fused_k50,
            holder_table=holder_table,
            claims=claims,
            claims_index=claims_index,
            gold=gold,
            connection=connection,
            question_vector=question_vector,
        )
        if diag.get("cause") != "present":
            out.append(diag)
    return out
