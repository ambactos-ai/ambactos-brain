# SPDX-License-Identifier: AGPL-3.0-only
"""Corrections PR #93 K1–K4 — tests écrits avant implémentation."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from banc_realite.config import VAULT_REF
from banc_realite.holders import build_holder_table, classify_pair
from banc_realite.measure_etage2 import SearchTracker


def _validate_subprocess_kwargs(kwargs: dict[str, Any]) -> None:
    """Mirror subprocess.run validation without launching a process."""
    capture_output = kwargs.get("capture_output", False)
    stdout = kwargs.get("stdout")
    stderr = kwargs.get("stderr")
    if capture_output and stdout is not None:
        raise ValueError("stdout must not be set when capture_output is True")
    if capture_output and stderr is not None:
        raise ValueError("stderr must not be set when capture_output is True")


# --- K1 : subprocess.run kwargs sans conflit capture_output / stdout ---


def test_k1_subprocess_kwargs_acceptable_for_pipe_true_and_false():
    from banc_realite.run_etage2 import subprocess_run_kwargs

    for pipe in (True, False):
        kwargs = subprocess_run_kwargs(pipe=pipe)
        _validate_subprocess_kwargs(kwargs)


def test_k1_default_path_uses_capture_not_conflicting_streams():
    from banc_realite.run_etage2 import subprocess_run_kwargs

    kwargs = subprocess_run_kwargs(pipe=False)
    assert kwargs.get("capture_output") is True
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs


# --- K2 : vault_ref passé à run_search pendant la mesure ---


def test_k2_search_tracker_passes_vault_ref():
    recorded: list[dict[str, Any]] = []

    def fake_run_search(query: str, **kwargs: Any) -> dict[str, Any]:
        recorded.append({"query": query, **kwargs})
        return {
            "evidence": [],
            "warnings": [],
            "strategy": {"fusion": "rrf_v1", "notes": []},
        }

    tracker = SearchTracker()
    with patch("banc_realite.measure_etage2.run_search", side_effect=fake_run_search):
        tracker.search_fn("test query", 8, embed_fn=lambda q: [0.1])

    assert recorded
    assert recorded[0]["vault_ref"] == VAULT_REF


# --- K3 : tient_sans_regle_piece compte les faits d'autres pièces ---


def _gold_two_pieces_shared_line() -> dict[str, Any]:
    assertion = {
        "id": "a1",
        "source": "semantic-memory/pages/x.md",
        "quote": "shared needle line",
        "content": "shared needle line",
    }
    return {
        "assertions": [assertion],
        "assertion_index": {"a1": assertion},
    }


def _claims_two_pieces_shared_line() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "c-a",
            "entity_id": "e",
            "predicate": "note.body",
            "value": "shared needle line",
            "source_file_path": "semantic-memory/pages/x.md",
            "extracted_by_skill": "tabulaire-deterministe",
            "source_span": {"raw_line": "prefix shared needle line suffix"},
            "qualifiers": {},
        },
        {
            "claim_id": "c-b",
            "entity_id": "e",
            "predicate": "note.body",
            "value": "shared needle line",
            "source_file_path": "semantic-memory/pages/y.md",
            "extracted_by_skill": "tabulaire-deterministe",
            "source_span": {"raw_line": "prefix shared needle line suffix"},
            "qualifiers": {},
        },
    ]


def test_k3_tient_sans_regle_counts_other_piece_claim():
    gold = _gold_two_pieces_shared_line()
    claims = _claims_two_pieces_shared_line()
    _, _, counter = build_holder_table(claims, gold)
    assert counter == 1

    claims_index = {str(c["claim_id"]): {**c, "piece": "y.md", "text": c["value"]} for c in claims}
    assertion = gold["assertion_index"]["a1"]
    element_b = {"kind": "claim_hit", "id": "c-b", "text": "shared needle line"}
    assert classify_pair(assertion, element_b, claims_index) == "non"


# --- K4 : diagnostic jambe — meilleur rang sur tous les détenteurs ---


class _FakeJambeConnection:
    def __init__(self, *, lex_rows: list[dict[str, Any]], vec_rows: list[dict[str, Any]]):
        self.lex_rows = lex_rows
        self.vec_rows = vec_rows

    def execute(self, query: str, params: tuple[Any, ...] | None = None):
        q = " ".join(query.split()).lower()
        if "search_claims_lexical" in q:
            return SimpleNamespace(fetchall=lambda: self.lex_rows)
        if "recall_match_mistral" in q:
            return SimpleNamespace(fetchall=lambda: self.vec_rows)
        raise AssertionError(f"unexpected query: {query}")


def test_k4_jambe_diagnostic_best_rank_across_holders():
    from banc_realite.diagnostic import jambe_diagnostic

    holder_ids = ["missing-holder", "present-holder"]
    fused = [{"id": f"noise-{i}"} for i in range(2)] + [{"id": "present-holder"}]
    connection = _FakeJambeConnection(
        lex_rows=[
            {"claim_id": "noise-lex-1"},
            {"claim_id": "noise-lex-2"},
            {"claim_id": "present-holder"},
        ],
        vec_rows=[
            {"claim_id": "noise-vec-1"},
            {"claim_id": "noise-vec-2"},
            {"claim_id": "present-holder", "similarity": 0.9},
        ],
    )
    diag = jambe_diagnostic(
        question_text="How many containers?",
        holder_ids=holder_ids,
        fused_ranking=fused,
        connection=connection,
        question_vector=[0.1, 0.2],
    )
    assert diag["rang_lexical"] == 3
    assert diag["rang_vectoriel"] == 3
    assert diag["rang_fusionne"] == 3
    assert diag["jambe"] == "present dans le classement fusionné"
