# SPDX-License-Identifier: AGPL-3.0-only
"""Mesure tabulaire v2 — structure des comptes."""

from banc_realite.measure_tabulaire_v2 import _slug_stem, count_by_scope


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _FakeConnection:
    def __init__(self, scope_rows, cell_rows):
        self.scope_rows = scope_rows
        self.cell_rows = cell_rows
        self.calls = 0

    def execute(self, sql, params=()):
        self.calls += 1
        if "GROUP BY" in sql:
            return _FakeCursor(self.scope_rows)
        return _FakeCursor(self.cell_rows)


def test_slug_stem_matches_file_entity():
    assert _slug_stem("17-collection-register.md") == "17_collection_register"


def test_count_by_scope_aggregates_rows_and_cells():
    conn = _FakeConnection(
        [{"scope": "row", "n": 3}, {"scope": "cell", "n": 5}],
        [
            {"entity_id": "quarry_tower", "source_file_path": "documents/rates.md"},
            {"entity_id": "rates", "source_file_path": "documents/rates.md"},
        ],
    )
    out = count_by_scope(conn)
    assert out["faits_ligne"] == 3
    assert out["faits_cellule"] == 5
    assert out["cellules_sujet_hors_fichier"] == 1
