# SPDX-License-Identifier: AGPL-3.0-only
"""Mesure palier par palier — structure et faux rapprochements."""

from banc_realite.measure_resolution import (
    choose_best_palier,
    detect_false_merges,
    summarize_paliers,
)


def test_summarize_paliers_has_all_columns():
    rows = summarize_paliers(
        {
            "P0": {
                "disputes_reconnues": 1,
                "trajectoires": 0,
                "pieges_en_dispute": 1,
                "indecidable": 10,
            },
            "P4": {
                "disputes_reconnues": 5,
                "trajectoires": 2,
                "pieges_en_dispute": 3,
                "indecidable": 8,
            },
        }
    )
    assert rows[0]["palier"] == "P0"
    assert rows[-1]["palier"] == "P4"
    assert "disputes_reconnues" in rows[0]


def test_choose_best_palier_penalizes_trap_inflation():
    table = {
        "P2": {"disputes_reconnues": 40, "trajectoires": 1, "pieges_en_dispute": 11, "indecidable": 5},
        "P3": {"disputes_reconnues": 10, "trajectoires": 3, "pieges_en_dispute": 1, "indecidable": 8},
    }
    assert choose_best_palier(table) == "P3"


def test_detect_false_merges_flags_trap_promoted_to_dispute():
    rows = [
        {"id": "gl-r0009", "gold": "reversibility", "issue": "dispute", "entity_a": "a", "entity_b": "b"},
        {"id": "gl-d0001", "gold": "dispute", "issue": "dispute", "entity_a": "a", "entity_b": "b"},
    ]
    false_merges = detect_false_merges(rows)
    assert len(false_merges) == 1
    assert false_merges[0]["id"] == "gl-r0009"
