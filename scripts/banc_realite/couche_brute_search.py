# SPDX-License-Identifier: AGPL-3.0-only
"""Recherche hybride sur la couche brute (passages) et la couche fine (claims)."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from banc_realite.compat_moteur import to_pgvector
from banc_realite.compat_moteur import _HYBRID_RRF_K, _normalize_lexical_rank, _rrf

from .config import VAULT_REF

_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "avec",
        "dans",
        "pour",
        "les",
        "des",
        "une",
        "par",
        "sur",
        "est",
        "que",
        "qui",
    }
)
_DIGIT_GROUP = re.compile(r"(?<=\d)\s+(?=\d)")


def _fold(s: str) -> str:
    folded = "".join(
        c for c in unicodedata.normalize("NFD", (s or "").lower()) if unicodedata.category(c) != "Mn"
    )
    return _DIGIT_GROUP.sub("", folded)


def focus_tokens(query: str) -> list[str]:
    return [
        t
        for t in re.split(r"[^a-z0-9]+", _fold(query))
        if len(t) >= 3 and t not in _STOP
    ]


def _tsquery(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    return " & ".join(f"'{t}'" for t in tokens)


def build_lexical_tsquery(connection, query: str) -> str | None:
    """Aligné sur public.search_claims_lexical : french + brain_unaccent, sémantique OU."""
    row = connection.execute(
        """
        SELECT string_agg(quote_literal(lexeme), ' | ' ORDER BY lexeme) AS string_agg
        FROM unnest(
            to_tsvector('french', public.brain_unaccent(COALESCE(%s, '')))
        )
        """,
        (query,),
    ).fetchone()
    if not row:
        return None
    tsq = row["string_agg"] if isinstance(row, dict) else row[0]
    return tsq if tsq else None


def _stable_id_key(value: Any) -> str:
    return str(value or "")


def _exclude_sql(column: str, exclude_pieces: set[str] | None) -> tuple[str, list[str]]:
    if not exclude_pieces:
        return "", []
    placeholders = ", ".join("%s" for _ in exclude_pieces)
    return f" AND {column} NOT IN ({placeholders})", sorted(exclude_pieces)


def _vault_sql(vault_ref: str | None = None) -> tuple[str, list[str]]:
    """Filtre dossier : GreenLoop inclut les lignes legacy sans vault_ref."""
    ref = vault_ref if vault_ref is not None else VAULT_REF
    if ref == "greenloop":
        return " AND (vault_ref = %s OR vault_ref IS NULL)", [ref]
    return " AND vault_ref = %s", [ref]


def _vector_fetch_k(top_k: int, exclude_pieces: set[str] | None) -> int:
    """Sur-fetch quand des pièces sont exclues : recall_match_mistral limite avant le filtre."""
    if not exclude_pieces:
        return top_k
    return max(top_k * 10, top_k + len(exclude_pieces) * 5)


def completer_top_k_eligible(
    ranked: list[tuple[dict[str, Any], float]],
    *,
    exclude_pieces: set[str] | None,
    top_k: int,
    piece_key: str,
) -> list[tuple[dict[str, Any], float]]:
    """Filtre les pièces exclues avant de tronquer à k (Z1)."""
    if not exclude_pieces:
        return ranked[:top_k]
    out: list[tuple[dict[str, Any], float]] = []
    for row, score in ranked:
        if (row.get(piece_key) or "") in exclude_pieces:
            continue
        out.append((row, score))
        if len(out) >= top_k:
            break
    return out


def passage_to_evidence(row: dict[str, Any], score: float, rank: int) -> dict[str, Any]:
    piece = row.get("source_ref") or ""
    return {
        "kind": "file_hit",
        "memory_type": "semantic",
        "id": str(row["id"]),
        "score": float(score),
        "rank": int(rank),
        "text": row.get("content") or "",
        "source": piece,
        "snippet": (row.get("content") or "")[:280],
        "payload": {"metadata": row.get("metadata") or {}},
        "provenance": {"source_file_path": piece},
    }


def claim_to_evidence(row: dict[str, Any], score: float, rank: int) -> dict[str, Any]:
    from .normalize import claim_text

    text = claim_text(row.get("value"))
    return {
        "kind": "claim_hit",
        "memory_type": "semantic",
        "id": str(row["claim_id"]),
        "score": float(score),
        "rank": int(rank),
        "text": text,
        "source": row.get("source_file_path") or "",
        "snippet": text[:280],
        "payload": {"claim": row},
        "provenance": {"source_file_path": row.get("source_file_path")},
    }


def search_passages_lexical(
    connection,
    query: str,
    *,
    top_k: int,
    exclude_pieces: set[str] | None = None,
    vault_ref: str | None = None,
) -> list[dict[str, Any]]:
    tsq = build_lexical_tsquery(connection, query)
    if not tsq:
        return []
    exclude_clause, exclude_params = _exclude_sql("source_ref", exclude_pieces)
    vault_clause, vault_params = _vault_sql(vault_ref)
    rows = connection.execute(
        f"""
        SELECT id, source_ref, content, metadata,
               ts_rank_cd(lex_tsv, to_tsquery('simple'::regconfig, %s))::double precision AS rank
        FROM public.memory_chunks
        WHERE source_type = 'corpus'
          AND lex_tsv @@ to_tsquery('simple'::regconfig, %s)
          {vault_clause}
          {exclude_clause}
        ORDER BY rank DESC, id ASC
        LIMIT %s
        """,
        (tsq, tsq, *vault_params, *exclude_params, top_k),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, 1):
        score = _normalize_lexical_rank(row["rank"])
        if score <= 0:
            continue
        out.append(passage_to_evidence(dict(row), score, rank))
    return out


def search_passages_vector(
    connection,
    query_vector: list[float],
    *,
    top_k: int,
    exclude_pieces: set[str] | None = None,
    vault_ref: str | None = None,
) -> list[dict[str, Any]]:
    exclude_clause, exclude_params = _exclude_sql("source_ref", exclude_pieces)
    vault_clause, vault_params = _vault_sql(vault_ref)
    rows = connection.execute(
        f"""
        SELECT id, source_ref, content, metadata,
               1.0 - (embedding_mistral OPERATOR(extensions.<=>) %s::extensions.vector(1024)) AS similarity
        FROM public.memory_chunks
        WHERE source_type = 'corpus'
          AND embedding_mistral IS NOT NULL
          {vault_clause}
          {exclude_clause}
        ORDER BY embedding_mistral OPERATOR(extensions.<=>) %s::extensions.vector(1024), id ASC
        LIMIT %s
        """,
        (to_pgvector(query_vector), *vault_params, *exclude_params, to_pgvector(query_vector), top_k),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, 1):
        out.append(passage_to_evidence(dict(row), float(row["similarity"]), rank))
    return out


def _resolve_vivier(top_k: int, vivier: int | None) -> int:
    if vivier is None:
        return top_k * 2
    if vivier < top_k:
        raise ValueError(f"vivier ({vivier}) doit être ≥ top_k ({top_k})")
    return vivier


def search_passages_hybrid(
    connection,
    query: str,
    *,
    query_vector: list[float],
    top_k: int,
    exclude_pieces: set[str] | None = None,
    vivier: int | None = None,
    vault_ref: str | None = None,
) -> list[dict[str, Any]]:
    pool = _resolve_vivier(top_k, vivier)
    lex = search_passages_lexical(
        connection, query, top_k=pool, exclude_pieces=exclude_pieces, vault_ref=vault_ref
    )
    vec = search_passages_vector(
        connection, query_vector, top_k=pool, exclude_pieces=exclude_pieces, vault_ref=vault_ref
    )
    if not lex and not vec:
        return []
    lex = sorted(lex, key=lambda e: (-float(e["score"]), _stable_id_key(e.get("id"))))
    vec = sorted(vec, key=lambda e: (-float(e["score"]), _stable_id_key(e.get("id"))))
    fused = _rrf([lex, vec], key_fn=lambda e: e["id"], k=_HYBRID_RRF_K)[:top_k]
    for i, item in enumerate(fused, 1):
        item["rank"] = i
    return fused


def search_claims_lexical(
    connection,
    query: str,
    *,
    top_k: int,
    exclude_pieces: set[str] | None = None,
    vault_ref: str | None = None,
) -> list[dict[str, Any]]:
    tsq = build_lexical_tsquery(connection, query)
    if not tsq:
        return []
    exclude_clause, exclude_params = _exclude_sql("source_file_path", exclude_pieces)
    rows = connection.execute(
        f"""
        SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path,
               ts_rank_cd(lex_tsv, to_tsquery('simple'::regconfig, %s))::double precision AS rank
        FROM public.claims
        WHERE lifecycle_status = 'active'
          AND vault_ref = %s
          AND predicate = ANY(%s::text[])
          AND lex_tsv @@ to_tsquery('simple'::regconfig, %s)
          {exclude_clause}
        ORDER BY rank DESC, claim_id ASC
        LIMIT %s
        """,
        (
            tsq,
            vault_ref if vault_ref is not None else VAULT_REF,
            ["note.body", "relation.about"],
            tsq,
            *exclude_params,
            top_k,
        ),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, 1):
        score = _normalize_lexical_rank(row["rank"])
        if score <= 0:
            continue
        out.append(claim_to_evidence(dict(row), score, rank))
    return out


def search_claims_vector(
    connection,
    query_vector: list[float],
    *,
    top_k: int,
    exclude_pieces: set[str] | None = None,
    vault_ref: str | None = None,
) -> list[dict[str, Any]]:
    by_id: dict[str, tuple[dict[str, Any], float]] = {}
    vec = to_pgvector(query_vector)
    fetch_k = _vector_fetch_k(top_k, exclude_pieces)
    effective_vault = vault_ref if vault_ref is not None else VAULT_REF
    for pred in ("note.body", "relation.about"):
        rows = connection.execute(
            """
            SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path,
                   similarity
            FROM public.recall_match_mistral(
                %s::extensions.vector, %s, %s, %s, %s
            )
            """,
            (vec, fetch_k, pred, None, effective_vault),
        ).fetchall()
        for row in rows:
            item = dict(row)
            cid = str(item["claim_id"])
            sim = float(item["similarity"])
            prev = by_id.get(cid)
            if prev is None or sim > prev[1]:
                by_id[cid] = (item, sim)
    ranked = sorted(
        by_id.values(),
        key=lambda pair: (-pair[1], _stable_id_key(pair[0].get("claim_id"))),
    )
    eligible = completer_top_k_eligible(
        ranked,
        exclude_pieces=exclude_pieces,
        top_k=top_k,
        piece_key="source_file_path",
    )
    out: list[dict[str, Any]] = []
    for rank, (row, sim) in enumerate(eligible, 1):
        out.append(claim_to_evidence(row, sim, rank))
    return out


def search_claims_hybrid(
    connection,
    query: str,
    *,
    query_vector: list[float],
    top_k: int,
    exclude_pieces: set[str] | None = None,
    vivier: int | None = None,
    vault_ref: str | None = None,
) -> list[dict[str, Any]]:
    pool = _resolve_vivier(top_k, vivier)
    lex = search_claims_lexical(
        connection, query, top_k=pool, exclude_pieces=exclude_pieces, vault_ref=vault_ref
    )
    vec = search_claims_vector(
        connection,
        query_vector,
        top_k=pool,
        exclude_pieces=exclude_pieces,
        vault_ref=vault_ref,
    )
    if not lex and not vec:
        return []
    lex = sorted(lex, key=lambda e: (-float(e["score"]), _stable_id_key(e.get("id"))))
    vec = sorted(vec, key=lambda e: (-float(e["score"]), _stable_id_key(e.get("id"))))
    fused = _rrf([lex, vec], key_fn=lambda e: e["id"], k=_HYBRID_RRF_K)[:top_k]
    for i, item in enumerate(fused, 1):
        item["rank"] = i
    return fused


def search_brute_fine_hybrid(
    connection,
    query: str,
    *,
    query_vector: list[float],
    top_k: int,
    exclude_pieces: set[str] | None = None,
    vivier: int | None = None,
    vault_ref: str | None = None,
) -> list[dict[str, Any]]:
    brute_kwargs: dict[str, Any] = {
        "query_vector": query_vector,
        "top_k": top_k * 2,
        "exclude_pieces": exclude_pieces,
        "vault_ref": vault_ref,
    }
    if vivier is not None:
        brute_kwargs["vivier"] = vivier
    brute = search_passages_hybrid(
        connection,
        query,
        **brute_kwargs,
    )
    fine = search_claims_hybrid(
        connection,
        query,
        **brute_kwargs,
    )
    if not brute and not fine:
        return []
    fused = _rrf(
        [brute, fine],
        key_fn=lambda e: f"{e['kind']}:{e['id']}",
        k=_HYBRID_RRF_K,
    )[:top_k]
    for i, item in enumerate(fused, 1):
        item["rank"] = i
    return fused


SEARCH_BY_CONFIG: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "brute_seule": search_passages_hybrid,
    "fine_seule": search_claims_hybrid,
    "brute_fine": search_brute_fine_hybrid,
}
