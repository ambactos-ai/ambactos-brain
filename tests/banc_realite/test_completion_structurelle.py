# SPDX-License-Identifier: AGPL-3.0-only
"""Complétion structurelle par lecture indexée."""

from banc_realite.completion_structurelle import completion_structurelle, enrichir_evidence_completion


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.last_sql = ""
        self.last_params = ()

    def fetchall(self):
        return self.rows


class _FakeConnection:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params=()):
        self.last_sql = sql
        self.last_params = params
        return _FakeCursor(self.rows)


def test_completion_structurelle_requete_indexee():
    fait = {
        "entity_id": "ent-foo",
        "qualifiers": {"attribute_id": "attr-bar"},
        "vault_ref": "greenloop",
    }
    rows = [
        {"claim_id": "1", "entity_id": "ent-foo", "value": "a", "qualifiers": {"attribute_id": "attr-bar"}},
        {"claim_id": "2", "entity_id": "ent-foo", "value": "b", "qualifiers": {"attribute_id": "attr-bar"}},
    ]
    conn = _FakeConnection(rows)
    out = completion_structurelle(fait, connection=conn)
    assert len(out) == 2
    assert "entity_id = %s" in conn.last_sql
    assert "attribute_id" in conn.last_sql
    assert "LIMIT" not in conn.last_sql.upper()
    assert conn.last_params[0] == "greenloop"
    assert conn.last_params[1] == "ent-foo"
    assert conn.last_params[2] == "attr-bar"


def test_enrichir_evidence_completion_ajoute_sibling():
    seed = {
        "kind": "claim_hit",
        "id": "1",
        "payload": {
            "claim": {
                "claim_id": "1",
                "entity_id": "ent-foo",
                "source_file_path": "documents/a.md",
                "qualifiers": {"attribute_id": "attr-bar"},
                "value": "a",
            }
        },
    }
    rows = [
        {
            "claim_id": "1",
            "entity_id": "ent-foo",
            "value": "a",
            "qualifiers": {"attribute_id": "attr-bar"},
            "source_file_path": "documents/a.md",
        },
        {
            "claim_id": "2",
            "entity_id": "ent-foo",
            "value": "b",
            "qualifiers": {"attribute_id": "attr-bar"},
            "source_file_path": "documents/b.md",
        },
    ]
    conn = _FakeConnection(rows)
    enriched, stats = enrichir_evidence_completion([seed], connection=conn)
    assert stats["faits_ajoutes"] == 1
    assert stats["faits_depuis_autre_piece"] == 1
    assert len(enriched) == 2
