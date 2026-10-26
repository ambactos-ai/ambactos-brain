# SPDX-License-Identifier: AGPL-3.0-only
"""Trois tours de recherche — alternance et exclusion de pièces."""

from banc_realite.trois_tours import alterner_resultats, piece_de_evidence, pieces_des_tours


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


def test_piece_de_evidence_claim_et_passage():
    claim = _hit("claim_hit", "c1", "documents/a.md")
    passage = _hit("file_hit", "p1", "documents/b.md")
    assert piece_de_evidence(claim) == "documents/a.md"
    assert piece_de_evidence(passage) == "documents/b.md"


def test_alterner_resultats_round_robin():
    t1 = [_hit("claim_hit", "a", "p1.md"), _hit("claim_hit", "b", "p2.md")]
    t2 = [_hit("file_hit", "c", "p3.md")]
    t3 = [_hit("claim_hit", "d", "p4.md"), _hit("claim_hit", "e", "p5.md")]
    merged = alterner_resultats([t1, t2, t3], k=5)
    assert [item["id"] for item in merged] == ["a", "c", "d", "b", "e"]


def test_alterner_respecte_k():
    t1 = [_hit("claim_hit", f"a{i}", f"p{i}.md") for i in range(4)]
    t2 = [_hit("claim_hit", f"b{i}", f"q{i}.md") for i in range(4)]
    t3 = [_hit("claim_hit", f"c{i}", f"r{i}.md") for i in range(4)]
    merged = alterner_resultats([t1, t2, t3], k=3)
    assert len(merged) == 3
    assert [item["id"] for item in merged] == ["a0", "b0", "c0"]


def test_pieces_des_tours_compte_distinctes():
    t1 = [_hit("claim_hit", "a", "p1.md"), _hit("claim_hit", "b", "p1.md")]
    t2 = [_hit("file_hit", "c", "p2.md")]
    t3 = []
    stats = pieces_des_tours([t1, t2, t3])
    assert stats["pieces_distinctes"] == 2
    assert stats["tour2_vide"] is False
    assert stats["tour3_vide"] is True


def test_alterner_file_hit_priorise_tour1():
    """Couche brute : pas d'alternance qui éjecte les passages du tour 1."""
    t1 = [
        _hit("file_hit", "a", "p1.md"),
        _hit("file_hit", "b", "p1.md"),
        _hit("file_hit", "c", "p1.md"),
    ]
    t2 = [_hit("file_hit", "x", "p2.md"), _hit("file_hit", "y", "p2.md")]
    merged = alterner_resultats([t1, t2, []], k=4, fusion="concat")
    assert [item["id"] for item in merged] == ["a", "b", "c", "x"]


def test_alterner_claim_hit_rest_en_round_robin():
    t1 = [_hit("claim_hit", "a", "p1.md"), _hit("claim_hit", "b", "p2.md")]
    t2 = [_hit("claim_hit", "c", "p3.md")]
    t3 = [_hit("claim_hit", "d", "p4.md")]
    merged = alterner_resultats([t1, t2, t3], k=4)
    assert [item["id"] for item in merged] == ["a", "c", "d", "b"]
