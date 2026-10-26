# SPDX-License-Identifier: AGPL-3.0-only
"""Corrections PR #99 — revue Codex Y1–Y6."""

from __future__ import annotations

from pathlib import Path

import pytest

from banc_realite.passages import MAX_CHARS, chunk_piece


# --- Y1 : jambe lexicale passages alignée sur search_claims_lexical (OR + french) ---


class _LexTsqueryConn:
    def execute(self, sql, params):
        class _Result:
            def fetchone(self_inner):
                return {"string_agg": "'heur' | 'suppl' | 'salair'"}

        return _Result()


def test_y1_lexical_tsquery_uses_or_not_and():
    from banc_realite.couche_brute_search import build_lexical_tsquery

    tsq = build_lexical_tsquery(
        _LexTsqueryConn(),
        "heures supplémentaires salaire",
    )
    assert tsq is not None
    assert " | " in tsq
    assert " & " not in tsq


def test_y1_missing_token_query_still_or_semantics():
    from banc_realite.couche_brute_search import build_lexical_tsquery

    tsq = build_lexical_tsquery(_LexTsqueryConn(), "heures xyzabsent salaire")
    assert tsq is not None
    assert tsq.count("|") >= 1


def test_y1_passages_lexical_does_not_use_focus_tokens_and():
    import inspect

    from banc_realite import couche_brute_search as cbs

    source = inspect.getsource(cbs.search_passages_lexical)
    assert "build_lexical_tsquery" in source
    assert "_tsquery(focus_tokens" not in source


# --- Y2 : jambe vectorielle faits sur note.body ET relation.about ---


class _VectorConn:
    def __init__(self) -> None:
        self.predicates: list[str] = []

    def execute(self, sql, params):
        self.predicates.append(params[2])

        class _Result:
            def fetchall(self_inner):
                pred = params[2]
                return [
                    {
                        "claim_id": f"id-{pred[:4]}",
                        "entity_id": "e1",
                        "predicate": pred,
                        "value": {"body": "x"},
                        "qualifiers": {},
                        "source_file_path": "a.md",
                        "similarity": 0.9 if pred == "note.body" else 0.95,
                    }
                ]

        return _Result()


def test_y2_vector_queries_both_predicates():
    from banc_realite.couche_brute_search import search_claims_vector

    conn = _VectorConn()
    hits = search_claims_vector(conn, [0.1] * 1024, top_k=5)
    assert set(conn.predicates) == {"note.body", "relation.about"}
    assert len(hits) == 2
    assert hits[0]["id"] == "id-rela"


# --- Y3 : coût mesuré sur le rendu CallFrame tronqué ---


def test_y3_tokens_measured_on_rendered_not_raw():
    from banc_realite.measure_couche_brute import estimate_injected_tokens

    raw_len = 3000
    evidence = [
        {
            "kind": "file_hit",
            "id": "p1",
            "score": 0.9,
            "rank": 1,
            "text": "x" * raw_len,
            "source": "doc.md",
        }
    ]
    tokens = estimate_injected_tokens(evidence)
    assert tokens < raw_len / 4
    assert tokens <= 8000 / 4


# --- Y4 : limite MAX_CHARS après recouvrement ---


def test_y4_overlap_does_not_exceed_max_chars(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    block_a = "A" * 1100
    block_b = "B" * 1100
    (corpus / "overlap.md").write_text(f"# T\n\n{block_a}\n\n{block_b}\n", encoding="utf-8")
    passages = chunk_piece("overlap.md", corpus=corpus)
    assert len(passages) >= 2
    for passage in passages:
        assert len(passage.content) <= MAX_CHARS


# --- Y5 : line_start cohérent avec le début du passage ---


def test_y5_line_start_matches_first_content_line(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lines = [f"line-{i:03d}: payload" for i in range(1, 80)]
    (corpus / "lines.md").write_text("# Section\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    source_lines = (corpus / "lines.md").read_text(encoding="utf-8").splitlines()
    passages = chunk_piece("lines.md", corpus=corpus)
    overlap_cases = [p for p in passages if p.line_start > 1]
    assert overlap_cases, "attendu au moins un passage avec recouvrement"
    for passage in overlap_cases:
        first_line = passage.content.splitlines()[0]
        assert source_lines[passage.line_start - 1].strip() == first_line.strip()


# --- Y6 : phrase conclusive dérivée du signe du calcul ---


def test_y6_three_sentences_when_all_deltas_negative():
    from banc_realite.run_couche_brute import _three_sentences

    ablation = {
        "configurations": {
            "brute_seule": {
                "aggregates": {
                    "par_forme_difficulte_k8": {
                        "single": {"easy": {"plancher": {"reussi": 10, "echec": 5}}},
                        "aggregate": {"easy": {"plancher": {"reussi": 1, "echec": 3}}},
                    },
                    "tokens_k8": {"total": 100},
                }
            },
            "fine_seule": {
                "aggregates": {
                    "par_forme_difficulte_k8": {
                        "single": {"easy": {"plancher": {"reussi": 20, "echec": 1}}},
                        "aggregate": {"easy": {"plancher": {"reussi": 5, "echec": 0}}},
                    },
                    "tokens_k8": {"total": 50},
                }
            },
            "brute_fine": {
                "aggregates": {
                    "par_forme_difficulte_k8": {
                        "single": {"easy": {"plancher": {"reussi": 12, "echec": 4}}},
                        "aggregate": {"easy": {"plancher": {"reussi": 2, "echec": 2}}},
                    },
                    "tokens_k8": {"total": 120},
                }
            },
        }
    }
    text = _three_sentences(ablation).lower()
    assert "gagne" not in text or "ne gagne pas" in text or "perd" in text
    assert "atomisation gagne" not in text
    assert "passages" in text or "brute" in text
