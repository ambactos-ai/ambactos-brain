# SPDX-License-Identifier: AGPL-3.0-only
"""Viviers larges — paramètre vivier et configurations (doubles, sans base)."""

from unittest.mock import MagicMock

import pytest

from banc_realite import couche_brute_search, measure_disputes, trois_tours
from banc_realite.couche_brute_search import (
    search_claims_hybrid,
    search_passages_hybrid,
)
from banc_realite.measure_disputes import build_evidence_for_mode
from banc_realite.trois_tours import recherche_trois_tours
from banc_realite.viviers import CONFIGURATIONS, _format_tableau


def _claim(ident: str, piece: str = "p.md") -> dict:
    return {
        "kind": "claim_hit",
        "id": ident,
        "source": piece,
        "provenance": {"source_file_path": piece},
        "score": 1.0,
        "rank": 1,
    }


def test_vivier_none_cherche_a_top_k_fois_deux(monkeypatch):
    lex_calls: list[int] = []
    vec_calls: list[int] = []

    def fake_lex(_connection, _query, *, top_k, exclude_pieces=None, vault_ref=None):
        lex_calls.append(top_k)
        return [_claim(f"lex-{i}") for i in range(top_k)]

    def fake_vec(_connection, _query_vector, *, top_k, exclude_pieces=None, vault_ref=None):
        vec_calls.append(top_k)
        return [_claim(f"vec-{i}", f"v{i}.md") for i in range(top_k)]

    monkeypatch.setattr(couche_brute_search, "search_claims_lexical", fake_lex)
    monkeypatch.setattr(couche_brute_search, "search_claims_vector", fake_vec)

    out = search_claims_hybrid(
        MagicMock(),
        "q",
        query_vector=[0.1],
        top_k=12,
        vivier=None,
    )
    assert lex_calls == [24]
    assert vec_calls == [24]
    assert len(out) == 12


def test_vivier_200_cherche_a_200_puis_coupe_a_top_k(monkeypatch):
    lex_calls: list[int] = []
    vec_calls: list[int] = []

    def fake_lex(_connection, _query, *, top_k, exclude_pieces=None, vault_ref=None):
        lex_calls.append(top_k)
        return [_claim(f"lex-{i}") for i in range(top_k)]

    def fake_vec(_connection, _query_vector, *, top_k, exclude_pieces=None, vault_ref=None):
        vec_calls.append(top_k)
        return [_claim(f"vec-{i}", f"v{i}.md") for i in range(top_k)]

    monkeypatch.setattr(couche_brute_search, "search_claims_lexical", fake_lex)
    monkeypatch.setattr(couche_brute_search, "search_claims_vector", fake_vec)

    out = search_claims_hybrid(
        MagicMock(),
        "q",
        query_vector=[0.1],
        top_k=12,
        vivier=200,
    )
    assert lex_calls == [200]
    assert vec_calls == [200]
    assert len(out) == 12


def test_vivier_inferieur_a_top_k_value_error():
    with pytest.raises(ValueError, match="vivier"):
        search_claims_hybrid(
            MagicMock(),
            "q",
            query_vector=[0.1],
            top_k=12,
            vivier=10,
        )


def test_vivier_transmis_a_passages_hybrid(monkeypatch):
    calls: list[int] = []

    def fake_lex(_connection, _query, *, top_k, exclude_pieces=None, vault_ref=None):
        calls.append(top_k)
        return [_claim("a")]

    def fake_vec(_connection, _query_vector, *, top_k, exclude_pieces=None, vault_ref=None):
        calls.append(top_k)
        return [_claim("b", "b.md")]

    monkeypatch.setattr(couche_brute_search, "search_passages_lexical", fake_lex)
    monkeypatch.setattr(couche_brute_search, "search_passages_vector", fake_vec)

    search_passages_hybrid(
        MagicMock(),
        "q",
        query_vector=[0.1],
        top_k=12,
        vivier=200,
    )
    assert calls == [200, 200]


def test_vivier_traverse_recherche_trois_tours(monkeypatch):
    calls: list[int | None] = []

    def fake_search(_connection, _query, *, query_vector, top_k, exclude_pieces=None, vivier=None):
        calls.append(vivier)
        return [_claim(f"t-{len(calls)}")]

    monkeypatch.setitem(trois_tours.SEARCH_BY_CONFIG, "fine_seule", fake_search)
    recherche_trois_tours(
        "question?",
        12,
        "fine",
        connection=MagicMock(),
        query_vector=[0.1],
        n_tours=2,
        vivier=200,
    )
    assert calls == [200, 200]


def test_vivier_traverse_build_evidence_for_mode(monkeypatch):
    calls: list[int | None] = []

    def fake_search(_connection, _query, *, query_vector, top_k, exclude_pieces=None, vivier=None):
        calls.append(vivier)
        return [_claim("x")]

    monkeypatch.setitem(measure_disputes.SEARCH_BY_CONFIG, "fine_seule", fake_search)
    question = {"question": "Q?"}
    embed_cache = {"Q?": [0.1]}
    build_evidence_for_mode(
        MagicMock(),
        question,
        couche="fine",
        k=12,
        mode="temoin",
        embed_cache=embed_cache,
        vivier=200,
    )
    assert calls == [200]


def test_vivier_traverse_build_evidence_trois_tours(monkeypatch):
    calls: list[int | None] = []

    def fake_search(_connection, _query, *, query_vector, top_k, exclude_pieces=None, vivier=None):
        calls.append(vivier)
        return [_claim(f"t-{len(calls)}")]

    monkeypatch.setitem(trois_tours.SEARCH_BY_CONFIG, "fine_seule", fake_search)
    question = {"question": "Q?"}
    embed_cache = {"Q?": [0.1]}
    build_evidence_for_mode(
        MagicMock(),
        question,
        couche="fine",
        k=12,
        mode="trois_tours",
        embed_cache=embed_cache,
        vivier=200,
    )
    assert calls == [200, 200, 200]


def test_configurations_liste_exacte():
    assert [c["etiquette"] for c in CONFIGURATIONS] == [
        "fine_tours3_k12",
        "hybride_12_plus_6_tours2",
        "fine_simple_k12_v200",
        "fine_tours3_k12_v200",
        "hybride_12_plus_6_tours2_v200",
        "hybride_12_plus_6_tours2_v60",
        "fine_tours3_k16_v200",
    ]
    assert len(CONFIGURATIONS) == 7


def test_tableau_inclut_secondes_par_question():
    results = []
    for config in CONFIGURATIONS:
        results.append(
            {
                **config,
                "disputes": {
                    "deux_cotes_sur_28": 10,
                    "reglage_sur_14": 5,
                    "ecart_sur_14": 5,
                },
                "appui": {
                    "tout_tient_sur_141": 50,
                    "au_moins_une_sur_141": 100,
                },
                "taille": {
                    "elements": {"mediane": 12, "p90": 12, "max": 12},
                    "caracteres": {"p90": 2000, "plafond_8000": 0},
                },
                "reels": {
                    "caracteres_reels": {"p90": 5000, "au_dessus_8000": 2},
                    "passages_tronques": 0,
                },
                "caracteres_reels": {"mediane": 4000, "p90": 5000},
                "tokens_estimes": {"mediane": 1000.0, "p90": 1250.0},
                "tabulaires_portes": 3,
                "secondes_par_question": 1.234 if config.get("vivier") == 200 else None,
            }
        )
    tableau = _format_tableau(results)
    assert "s/q" in tableau
    assert "1.234" in tableau
    lines = [line for line in tableau.strip().splitlines() if line and not line.startswith("-")]
    assert len(lines) == len(CONFIGURATIONS) + 1
