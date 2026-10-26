# SPDX-License-Identifier: AGPL-3.0-only
"""Paramètres n_tours et fusion — doubles sans base."""

from unittest.mock import MagicMock

from banc_realite import trois_tours
from banc_realite.trois_tours import alterner_resultats, pieces_des_tours, recherche_trois_tours


def _hit(kind: str, ident: str, piece: str) -> dict:
    if kind == "claim_hit":
        return {
            "kind": kind,
            "id": ident,
            "source": piece,
            "provenance": {"source_file_path": piece},
        }
    return {
        "kind": kind,
        "id": ident,
        "source": piece,
        "provenance": {"source_ref": piece},
    }


def _fake_search_factory():
    calls: list[dict] = []

    def search_fn(_connection, _query, *, query_vector, top_k, exclude_pieces):
        calls.append({"exclude": set(exclude_pieces), "top_k": top_k})
        idx = len(calls)
        if idx == 1:
            items = [_hit("claim_hit", f"t{idx}-a", f"p{idx}.md")]
        elif idx == 2:
            items = [_hit("claim_hit", f"t{idx}-a", f"p{idx}.md")]
        elif idx == 3:
            items = [_hit("claim_hit", f"t{idx}-a", f"p{idx}.md")]
        elif idx == 4:
            items = [_hit("claim_hit", f"t{idx}-a", f"p{idx}.md")]
        else:
            items = [_hit("claim_hit", f"t{idx}-a", f"p{idx}.md")]
        return items[:top_k]

    return search_fn, calls


def test_n_tours_1_une_seule_recherche(monkeypatch):
    search_fn, calls = _fake_search_factory()
    monkeypatch.setitem(trois_tours.SEARCH_BY_CONFIG, "fine_seule", search_fn)
    result = recherche_trois_tours(
        "question?",
        12,
        "fine",
        connection=MagicMock(),
        query_vector=[0.1],
        n_tours=1,
    )
    assert len(calls) == 1
    assert calls[0]["exclude"] == set()
    assert len(result["tours"]) == 1


def test_n_tours_5_exclut_pieces_des_quatre_premiers(monkeypatch):
    search_fn, calls = _fake_search_factory()
    monkeypatch.setitem(trois_tours.SEARCH_BY_CONFIG, "fine_seule", search_fn)
    recherche_trois_tours(
        "question?",
        12,
        "fine",
        connection=MagicMock(),
        query_vector=[0.1],
        n_tours=5,
    )
    assert len(calls) == 5
    assert calls[4]["exclude"] == {"p1.md", "p2.md", "p3.md", "p4.md"}


def test_fusion_concat_passages_reproduit_concat():
    t1 = [
        _hit("file_hit", "a", "p1.md"),
        _hit("file_hit", "b", "p1.md"),
        _hit("file_hit", "c", "p1.md"),
    ]
    t2 = [_hit("file_hit", "x", "p2.md"), _hit("file_hit", "y", "p2.md")]
    merged = alterner_resultats([t1, t2, []], k=4, fusion="concat")
    assert [item["id"] for item in merged] == ["a", "b", "c", "x"]


def test_fusion_alternance_passages_un_par_tour():
    t1 = [
        _hit("file_hit", "a", "p1.md"),
        _hit("file_hit", "b", "p1.md"),
    ]
    t2 = [_hit("file_hit", "x", "p2.md"), _hit("file_hit", "y", "p2.md")]
    t3 = [_hit("file_hit", "z", "p3.md")]
    merged = alterner_resultats([t1, t2, t3], k=4, fusion="alternance")
    assert [item["id"] for item in merged] == ["a", "x", "z", "b"]


def test_defauts_inchanges_sur_faits():
    t1 = [_hit("claim_hit", "a", "p1.md"), _hit("claim_hit", "b", "p2.md")]
    t2 = [_hit("claim_hit", "c", "p3.md")]
    t3 = [_hit("claim_hit", "d", "p4.md"), _hit("claim_hit", "e", "p5.md")]
    merged = alterner_resultats([t1, t2, t3], k=5)
    assert [item["id"] for item in merged] == ["a", "c", "d", "b", "e"]
    stats = pieces_des_tours([t1, t2, t3])
    assert stats["tours_vides"] == []
    assert stats["tour3_vide"] is False


def test_pieces_des_tours_rapporte_tours_vides():
    stats = pieces_des_tours([[_hit("claim_hit", "a", "p.md")], [], []])
    assert stats["tours_vides"] == [1, 2]
    assert stats["tour2_vide"] is True
    assert stats["tour3_vide"] is True
