# SPDX-License-Identifier: AGPL-3.0-only
"""Budget de complétion structurelle."""

from banc_realite.completion_structurelle import enrichir_evidence_completion


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _FakeConnection:
    def __init__(self, rows_by_key: dict[tuple[str, str], list[dict]]):
        self.rows_by_key = rows_by_key

    def execute(self, sql, params=()):
        key = (params[1], params[2])
        return _FakeCursor(self.rows_by_key.get(key, []))


def _claim_row(
    claim_id: str,
    *,
    entity_id: str = "ent-foo",
    attribute_id: str = "attr-bar",
    value: str = "x",
    source: str = "documents/a.md",
) -> dict:
    return {
        "claim_id": claim_id,
        "entity_id": entity_id,
        "value": value,
        "qualifiers": {"attribute_id": attribute_id},
        "source_file_path": source,
    }


def _seed_item(claim: dict) -> dict:
    return {
        "kind": "claim_hit",
        "id": str(claim["claim_id"]),
        "payload": {"claim": claim},
    }


def _rows_for_entity(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    if not rows:
        return {}
    entity_id = rows[0]["entity_id"]
    attribute_id = rows[0]["qualifiers"]["attribute_id"]
    return {(entity_id, attribute_id): rows}


def test_budget_none_identique_au_comportement_actuel():
    seed = _claim_row("1", source="documents/a.md", value="a")
    rows = [
        _claim_row("1", source="documents/a.md", value="a"),
        _claim_row("2", source="documents/b.md", value="b"),
        _claim_row("3", source="documents/a.md", value="c"),
    ]
    conn = _FakeConnection(_rows_for_entity(rows))
    evidence = [_seed_item(seed)]
    enriched, stats = enrichir_evidence_completion(evidence, connection=conn, budget=None)
    assert len(enriched) == 3
    assert stats["faits_ajoutes"] == 2
    assert stats["faits_depuis_autre_piece"] == 1
    assert stats["budget"] is None
    assert stats["faits_ecartes_budget"] == 0


def test_budget_plafonne_le_total():
    seed = _claim_row("1", source="documents/a.md", value="a")
    rows = [
        _claim_row("1", source="documents/a.md", value="a"),
        _claim_row("2", source="documents/b.md", value="b"),
        _claim_row("3", source="documents/c.md", value="c"),
        _claim_row("4", source="documents/d.md", value="d"),
    ]
    conn = _FakeConnection(_rows_for_entity(rows))
    evidence = [_seed_item(seed)]
    enriched, stats = enrichir_evidence_completion(evidence, connection=conn, budget=3)
    assert len(enriched) <= 3
    assert stats["faits_ajoutes"] == 2
    assert stats["faits_ecartes_budget"] == 1


def test_evidence_deja_au_dessus_du_budget_inchangee():
    seed = _claim_row("1", source="documents/a.md", value="a")
    rows = [
        _claim_row("1", source="documents/a.md", value="a"),
        _claim_row("2", source="documents/b.md", value="b"),
    ]
    conn = _FakeConnection(_rows_for_entity(rows))
    evidence = [_seed_item(seed), _seed_item(_claim_row("9", source="documents/z.md"))]
    enriched, stats = enrichir_evidence_completion(evidence, connection=conn, budget=2)
    assert enriched == evidence
    assert stats["faits_ajoutes"] == 0
    assert stats["faits_ecartes_budget"] == 1


def test_priorite_autre_piece_avant_meme_piece():
    seed = _claim_row("1", source="documents/a.md", value="a")
    rows = [
        _claim_row("1", source="documents/a.md", value="a"),
        _claim_row("2", source="documents/a.md", value="meme"),
        _claim_row("3", source="documents/b.md", value="autre"),
    ]
    conn = _FakeConnection(_rows_for_entity(rows))
    evidence = [_seed_item(seed)]
    enriched, _stats = enrichir_evidence_completion(evidence, connection=conn, budget=2)
    assert len(enriched) == 2
    assert enriched[1]["id"] == "3"


def test_determinisme_deux_appels_identiques():
    seed = _claim_row("1", source="documents/a.md", value="a")
    rows = [
        _claim_row("1", source="documents/a.md", value="a"),
        _claim_row("2", source="documents/b.md", value="b"),
        _claim_row("3", source="documents/c.md", value="c"),
    ]
    conn = _FakeConnection(_rows_for_entity(rows))
    evidence = [_seed_item(seed)]
    first, stats1 = enrichir_evidence_completion(evidence, connection=conn, budget=3)
    second, stats2 = enrichir_evidence_completion(evidence, connection=conn, budget=3)
    assert first == second
    assert stats1 == stats2


def test_faits_ecartes_budget_compte_les_candidats_laisses():
    seed = _claim_row("1", source="documents/a.md", value="a")
    rows = [
        _claim_row("1", source="documents/a.md", value="a"),
        _claim_row("2", source="documents/b.md", value="b"),
        _claim_row("3", source="documents/c.md", value="c"),
        _claim_row("4", source="documents/d.md", value="d"),
    ]
    conn = _FakeConnection(_rows_for_entity(rows))
    evidence = [_seed_item(seed)]
    _enriched, stats = enrichir_evidence_completion(evidence, connection=conn, budget=1)
    assert stats["faits_ecartes_budget"] == 3
