# SPDX-License-Identifier: AGPL-3.0-only
"""Complétion structurelle par lecture indexée (même entité, même attribut)."""
from __future__ import annotations

from typing import Any

from .config import VAULT_REF
from .couche_brute_search import claim_to_evidence
from .normalize import claim_text, piece_basename


def completion_structurelle(
    fait: dict[str, Any],
    *,
    connection,
    vault_ref: str = VAULT_REF,
) -> list[dict[str, Any]]:
    """Retourne tous les faits actifs de même entity_id et attribute_id."""
    entity_id = fait.get("entity_id")
    qualifiers = fait.get("qualifiers") or {}
    attribute_id = qualifiers.get("attribute_id")
    if not entity_id or not attribute_id:
        return []
    rows = connection.execute(
        """
        SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path,
               source_span, extracted_by_skill, lifecycle_status
        FROM brain.claims
        WHERE lifecycle_status = 'active'
          AND vault_ref = %s
          AND entity_id = %s
          AND qualifiers->>'attribute_id' = %s
        ORDER BY claim_id ASC
        """,
        (vault_ref, entity_id, str(attribute_id)),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["text"] = claim_text(item.get("value"))
        out.append(item)
    return out


def _claim_from_evidence_item(item: dict[str, Any]) -> dict[str, Any] | None:
    if item.get("kind") != "claim_hit":
        return None
    payload = item.get("payload") or {}
    claim = payload.get("claim")
    if isinstance(claim, dict):
        return claim
    return None


def _seed_sibling_groups(
    evidence: list[dict[str, Any]],
    existing_ids: set[str],
    *,
    connection,
    vault_ref: str,
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    """Graines dans l'ordre d'apparition ; frères classés autre pièce / même pièce."""
    seeds_seen: set[str] = set()
    groups: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for item in evidence:
        claim = _claim_from_evidence_item(item)
        if claim is None:
            continue
        seed_id = str(claim.get("claim_id") or item.get("id") or "")
        if not seed_id or seed_id in seeds_seen:
            continue
        seeds_seen.add(seed_id)
        seed_piece = piece_basename(claim.get("source_file_path") or "")
        autre_piece: list[dict[str, Any]] = []
        meme_piece: list[dict[str, Any]] = []
        for sibling in completion_structurelle(claim, connection=connection, vault_ref=vault_ref):
            cid = str(sibling["claim_id"])
            if cid in existing_ids:
                continue
            sibling_piece = piece_basename(sibling.get("source_file_path") or "")
            if sibling_piece != seed_piece:
                autre_piece.append(sibling)
            else:
                meme_piece.append(sibling)
        groups.append((seed_piece, autre_piece, meme_piece))
    return groups


def _completion_sans_budget(
    evidence: list[dict[str, Any]],
    existing_ids: set[str],
    *,
    connection,
    vault_ref: str,
) -> tuple[list[dict[str, Any]], set[str], int]:
    """Comportement historique : tous les frères, ordre graine puis claim_id."""
    added_ids: set[str] = set()
    autres_piece = 0
    out = list(evidence)
    seeds: set[str] = set()
    for item in evidence:
        claim = _claim_from_evidence_item(item)
        if claim is None:
            continue
        seed_id = str(claim.get("claim_id") or item.get("id") or "")
        if not seed_id or seed_id in seeds:
            continue
        seeds.add(seed_id)
        seed_piece = piece_basename(claim.get("source_file_path") or "")
        for sibling in completion_structurelle(claim, connection=connection, vault_ref=vault_ref):
            cid = str(sibling["claim_id"])
            if cid in existing_ids or cid in added_ids:
                continue
            added_ids.add(cid)
            sibling_piece = piece_basename(sibling.get("source_file_path") or "")
            if sibling_piece != seed_piece:
                autres_piece += 1
            out.append(claim_to_evidence(sibling, score=0.0, rank=len(out) + 1))
    return out, added_ids, autres_piece


def _completion_sous_budget(
    evidence: list[dict[str, Any]],
    existing_ids: set[str],
    budget: int,
    groups: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], set[str], int, int]:
    """Sélection déterministe round-robin : autre pièce puis même pièce."""
    added_ids: set[str] = set()
    autres_piece = 0
    out = list(evidence)
    all_candidates: set[str] = set()
    for _seed_piece, autre_piece, meme_piece in groups:
        for sibling in autre_piece + meme_piece:
            all_candidates.add(str(sibling["claim_id"]))

    def _room() -> bool:
        return len(out) < budget

    for pass_idx in (1, 2):
        queues = [
            (group[0], group[1] if pass_idx == 1 else group[2]) for group in groups
        ]
        indices = [0] * len(queues)
        while _room():
            progress = False
            for i, (seed_piece, queue) in enumerate(queues):
                if not _room():
                    break
                while indices[i] < len(queue):
                    sibling = queue[indices[i]]
                    indices[i] += 1
                    cid = str(sibling["claim_id"])
                    if cid in existing_ids or cid in added_ids:
                        continue
                    added_ids.add(cid)
                    sibling_piece = piece_basename(sibling.get("source_file_path") or "")
                    if sibling_piece != seed_piece:
                        autres_piece += 1
                    out.append(claim_to_evidence(sibling, score=0.0, rank=len(out) + 1))
                    progress = True
                    break
            if not progress:
                break

    faits_ecartes = len(all_candidates - added_ids)
    return out, added_ids, autres_piece, faits_ecartes


def enrichir_evidence_completion(
    evidence: list[dict[str, Any]],
    *,
    connection,
    vault_ref: str = VAULT_REF,
    budget: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    """Complète l'évidence par lecture indexée sur chaque claim_hit."""
    existing_ids = {
        str(item.get("id"))
        for item in evidence
        if item.get("kind") == "claim_hit" and item.get("id")
    }
    if budget is None:
        out, added_ids, autres_piece = _completion_sans_budget(
            evidence, existing_ids, connection=connection, vault_ref=vault_ref
        )
        return out, {
            "faits_ajoutes": len(added_ids),
            "faits_depuis_autre_piece": autres_piece,
            "budget": None,
            "faits_ecartes_budget": 0,
        }

    if len(evidence) >= budget:
        groups = _seed_sibling_groups(
            evidence, existing_ids, connection=connection, vault_ref=vault_ref
        )
        all_candidates: set[str] = set()
        for _seed_piece, autre_piece, meme_piece in groups:
            for sibling in autre_piece + meme_piece:
                all_candidates.add(str(sibling["claim_id"]))
        return list(evidence), {
            "faits_ajoutes": 0,
            "faits_depuis_autre_piece": 0,
            "budget": budget,
            "faits_ecartes_budget": len(all_candidates),
        }

    groups = _seed_sibling_groups(
        evidence, existing_ids, connection=connection, vault_ref=vault_ref
    )
    out, added_ids, autres_piece, faits_ecartes = _completion_sous_budget(
        evidence, existing_ids, budget, groups
    )
    return out, {
        "faits_ajoutes": len(added_ids),
        "faits_depuis_autre_piece": autres_piece,
        "budget": budget,
        "faits_ecartes_budget": faits_ecartes,
    }
