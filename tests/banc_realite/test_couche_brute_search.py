# SPDX-License-Identifier: AGPL-3.0-only
"""Équivalence RRF et recherche couche brute — inerte."""
from __future__ import annotations

from banc_realite.couche_brute_search import focus_tokens
from banc_realite.compat_moteur import _HYBRID_RRF_K, _rrf


def _replicate_rrf(ranked_lists, key_fn, k=60):
    scores: dict[str, float] = {}
    by_key: dict[str, object] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            kid = key_fn(item)
            scores[kid] = scores.get(kid, 0.0) + 1.0 / (k + rank + 1)
            by_key.setdefault(kid, item)
    fused = []
    for kid in sorted(scores, key=lambda i: (-scores[i], i)):
        item = by_key[kid]
        if isinstance(item, dict):
            item = {**item, "score": scores[kid]}
        fused.append(item)
    return fused


def test_rrf_equivalent_to_brain():
    lex = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.5}]
    vec = [{"id": "b", "score": 0.8}, {"id": "c", "score": 0.7}]
    brain = _rrf([lex, vec], key_fn=lambda e: e["id"], k=_HYBRID_RRF_K)
    replica = _replicate_rrf([lex, vec], key_fn=lambda e: e["id"], k=_HYBRID_RRF_K)
    assert [e["id"] for e in brain] == [e["id"] for e in replica]
    assert [round(e["score"], 6) for e in brain] == [round(e["score"], 6) for e in replica]


def test_focus_tokens_min_length():
    assert "the" not in focus_tokens("the quick brown fox jumps")
    assert "quick" in focus_tokens("the quick brown fox jumps")


def test_brute_fine_merge_order():
    brute = [{"kind": "file_hit", "id": "p1", "score": 0.9}]
    fine = [{"kind": "claim_hit", "id": "c1", "score": 0.95}]
    fused = _rrf(
        [brute, fine],
        key_fn=lambda e: f"{e['kind']}:{e['id']}",
        k=_HYBRID_RRF_K,
    )
    assert len(fused) == 2
    assert {f"{e['kind']}:{e['id']}" for e in fused} == {"file_hit:p1", "claim_hit:c1"}


def test_completer_top_k_eligible_filtre_avant_limite():
    """Z1 : les k premiers exclus ne doivent pas vider le tour — compléter jusqu'à k."""
    from banc_realite.couche_brute_search import completer_top_k_eligible

    ranked = [
        ({"source_file_path": "a.md", "claim_id": "1"}, 1.0),
        ({"source_file_path": "a.md", "claim_id": "2"}, 0.9),
        ({"source_file_path": "a.md", "claim_id": "3"}, 0.8),
        ({"source_file_path": "b.md", "claim_id": "4"}, 0.7),
        ({"source_file_path": "b.md", "claim_id": "5"}, 0.6),
    ]
    eligible = completer_top_k_eligible(
        ranked,
        exclude_pieces={"a.md"},
        top_k=2,
        piece_key="source_file_path",
    )
    assert len(eligible) == 2
    assert all(row["source_file_path"] == "b.md" for row, _ in eligible)
