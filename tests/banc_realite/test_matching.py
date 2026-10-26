# SPDX-License-Identifier: AGPL-3.0-only
from banc_realite.fidelite import match_assertion
from banc_realite.normalize import normalize_text


def test_match_assertion_by_quote_substring():
    assertion = {"quote": "WT-20250101-BC-AA", "content": "fallback"}
    elements = [{"text": "Recorded visit WT-20250101-BC-AA on 2025-01-01"}]
    hit, _ = match_assertion(assertion, elements)
    assert hit is elements[0]


def test_match_assertion_uses_content_when_quote_empty():
    assertion = {"quote": "", "content": "Seven containers"}
    elements = [{"text": "The site used seven containers total."}]
    hit, _ = match_assertion(assertion, elements)
    assert hit is elements[0]


def test_normalize_is_deterministic():
    a = normalize_text("Foo—Bar")
    b = normalize_text("Foo—Bar")
    assert a == b


def test_match_assertion_prefers_cell_over_row_when_both_exist():
    assertion = {
        "quote": "| Station House | 12 | 10 | 12 |",
        "content": "Station House in 2024-01 = 12.",
    }
    elements = [
        {
            "claim_id": "row-1",
            "text": "| Station House | 12 | 10 | 12 |",
            "entity_id": "24_customer_review",
            "qualifiers": {"tabular_scope": "row"},
        },
        {
            "claim_id": "cell-1",
            "text": "12",
            "entity_id": "station_house",
            "qualifiers": {"tabular_scope": "cell", "attribute_id": "collections", "month": "2024-01"},
        },
    ]
    hit, kind = match_assertion(assertion, elements)
    assert hit is elements[1]
    assert kind == "cellule"
    assert hit["entity_id"] == "station_house"
