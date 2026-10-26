# SPDX-License-Identifier: AGPL-3.0-only
"""Trois tours de recherche avec exclusion de pièces et alternance."""
from __future__ import annotations

from typing import Any

from banc_realite.compat_moteur import _build_structured_query, _normalize_evidence_item

from .couche_brute_search import SEARCH_BY_CONFIG
from .dossiers import Dossier, dossier_courant

COUCHE_TO_CONFIG = {
    "fine": "fine_seule",
    "brute": "brute_seule",
    "brute_fine": "brute_fine",
}

FUSION_MODES = frozenset({"alternance", "concat"})


def piece_de_evidence(item: dict[str, Any]) -> str:
    kind = item.get("kind")
    provenance = item.get("provenance") or {}
    if kind == "claim_hit":
        return str(item.get("source") or provenance.get("source_file_path") or "")
    return str(item.get("source") or provenance.get("source_ref") or "")


def pieces_du_tour(items: list[dict[str, Any]]) -> set[str]:
    return {piece for piece in (piece_de_evidence(item) for item in items) if piece}


def _concat_dedupe_tours(tours: list[list[dict[str, Any]]], *, k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for tour in tours:
        for item in tour:
            if len(out) >= k:
                return out
            key = (str(item.get("kind") or ""), str(item.get("id") or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
    return out


def _item_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("kind") or ""), str(item.get("id") or ""))


def _fusionner_tours_restants(
    tours: list[list[dict[str, Any]]],
    *,
    k: int,
    fusion: str,
    seen: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    if fusion == "concat":
        merged = _concat_dedupe_tours(tours, k=k)
    else:
        merged = []
        indices = [0] * len(tours)
        while len(merged) < k:
            progressed = False
            for tour_idx, tour in enumerate(tours):
                if indices[tour_idx] < len(tour):
                    merged.append(tour[indices[tour_idx]])
                    indices[tour_idx] += 1
                    progressed = True
                    if len(merged) >= k:
                        break
            if not progressed:
                break
    out: list[dict[str, Any]] = []
    for item in merged:
        if len(out) >= k:
            break
        key = _item_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def alterner_resultats(
    tours: list[list[dict[str, Any]]],
    *,
    k: int,
    fusion: str = "alternance",
    k_tour1: int | None = None,
) -> list[dict[str, Any]]:
    """Fusion explicite : alternance round-robin ou concat dédoublonné (tour 1 d'abord)."""
    if fusion not in FUSION_MODES:
        raise ValueError(f"fusion inconnue: {fusion}")
    if k_tour1 is None:
        if fusion == "concat":
            return _concat_dedupe_tours(tours, k=k)
        out: list[dict[str, Any]] = []
        indices = [0] * len(tours)
        while len(out) < k:
            progressed = False
            for tour_idx, tour in enumerate(tours):
                if indices[tour_idx] < len(tour):
                    out.append(tour[indices[tour_idx]])
                    indices[tour_idx] += 1
                    progressed = True
                    if len(out) >= k:
                        break
            if not progressed:
                break
        return out

    if not (1 <= k_tour1 <= k):
        raise ValueError(f"k_tour1 doit être entre 1 et k={k}, reçu {k_tour1}")
    if not tours:
        return []
    tour1 = list(tours[0][:k_tour1])
    seen = {_item_key(item) for item in tour1}
    remaining_k = k - len(tour1)
    if remaining_k <= 0 or len(tours) <= 1:
        return tour1[:k]
    rest = _fusionner_tours_restants(tours[1:], k=remaining_k, fusion=fusion, seen=seen)
    return tour1 + rest


def pieces_des_tours(tours: list[list[dict[str, Any]]]) -> dict[str, Any]:
    all_pieces: set[str] = set()
    for tour in tours:
        all_pieces |= pieces_du_tour(tour)
    tours_vides = [idx for idx, tour in enumerate(tours) if not tour]
    return {
        "pieces_distinctes": len(all_pieces),
        "tour2_vide": len(tours) > 1 and not tours[1],
        "tour3_vide": len(tours) > 2 and not tours[2],
        "tours_vides": tours_vides,
    }


def _search_tour(
    connection,
    *,
    query: str,
    query_vector: list[float],
    couche: str,
    top_k: int,
    exclude_pieces: set[str],
    vivier: int | None = None,
    vault_ref: str | None = None,
) -> list[dict[str, Any]]:
    config = COUCHE_TO_CONFIG[couche]
    search_fn = SEARCH_BY_CONFIG[config]
    kwargs: dict[str, Any] = {
        "query_vector": query_vector,
        "top_k": top_k,
        "exclude_pieces": exclude_pieces,
    }
    if vault_ref is not None:
        kwargs["vault_ref"] = vault_ref
    if vivier is not None:
        kwargs["vivier"] = vivier
    return search_fn(connection, query, **kwargs)


def recherche_trois_tours(
    question: str,
    k: int,
    couche: str,
    *,
    connection,
    query_vector: list[float],
    n_tours: int = 3,
    fusion: str = "alternance",
    k_tour1: int | None = None,
    vivier: int | None = None,
    dossier: Dossier | None = None,
) -> dict[str, Any]:
    if couche not in COUCHE_TO_CONFIG:
        raise ValueError(f"couche inconnue: {couche}")
    if n_tours < 1:
        raise ValueError(f"n_tours doit être ≥ 1, reçu {n_tours}")
    if fusion not in FUSION_MODES:
        raise ValueError(f"fusion inconnue: {fusion}")
    if k_tour1 is not None and not (1 <= k_tour1 <= k):
        raise ValueError(f"k_tour1 doit être entre 1 et k={k}, reçu {k_tour1}")

    current = dossier if dossier is not None else dossier_courant()
    structured_query = _build_structured_query(
        question,
        current.responsibility_context,
        {"responsibility_context": current.responsibility_context},
    )
    tours: list[list[dict[str, Any]]] = []
    exclude: set[str] = set()
    for tour_idx in range(n_tours):
        top_k = k_tour1 if tour_idx == 0 and k_tour1 is not None else k
        tour = _search_tour(
            connection,
            query=structured_query,
            query_vector=query_vector,
            couche=couche,
            top_k=top_k,
            exclude_pieces=exclude,
            vivier=vivier,
            vault_ref=current.vault_ref if dossier is not None else None,
        )
        tours.append(tour)
        exclude |= pieces_du_tour(tour)

    merged = alterner_resultats(tours, k=k, fusion=fusion, k_tour1=k_tour1)
    evidence = [_normalize_evidence_item(item) for item in merged]
    stats = pieces_des_tours(tours)
    parametres: dict[str, Any] = {
        "question": question,
        "k": k,
        "couche": couche,
        "config": COUCHE_TO_CONFIG[couche],
        "n_tours": n_tours,
        "fusion": fusion,
        **stats,
    }
    if k_tour1 is not None:
        parametres["k_tour1"] = k_tour1
    if n_tours >= 2:
        exclude2 = pieces_du_tour(tours[0])
        parametres["pieces_exclues_tour2"] = sorted(exclude2)
    if n_tours >= 3:
        exclude3 = pieces_du_tour(tours[0]) | pieces_du_tour(tours[1])
        parametres["pieces_exclues_tour3"] = sorted(exclude3)
    return {
        "evidence": evidence,
        "raw_evidence": merged,
        "tours": tours,
        "parametres": parametres,
    }
