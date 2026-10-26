# SPDX-License-Identifier: AGPL-3.0-only
from collections import Counter

from banc_realite.fidelite import match_assertion


def classify_dispute(sides, gold_index, by_piece):
    matched_elements = []
    for side in sides:
        assertion = gold_index[side["assertion"]]
        piece = assertion["source"].split("/")[-1]
        hit, _ = match_assertion(assertion, by_piece.get(piece, []))
        if hit:
            matched_elements.append(str(hit["claim_id"]))
    unique = len(set(matched_elements))
    total = len(sides)
    if unique == 0:
        return "aucun"
    if unique == 1 and len(matched_elements) > 1:
        return "cotes_fusionnes"
    if unique == total:
        return "tous_les_cotes"
    if unique == 1:
        return "un_seul_cote"
    return "partiel"


def test_dispute_all_sides_distinct():
    gold_index = {
        "a1": {"source": "documents/x.md", "quote": "two containers"},
        "a2": {"source": "documents/x.md", "quote": "three containers"},
    }
    by_piece = {
        "x.md": [
            {"claim_id": "1", "text": "two containers removed"},
            {"claim_id": "2", "text": "three containers removed"},
        ]
    }
    sides = [{"assertion": "a1"}, {"assertion": "a2"}]
    assert classify_dispute(sides, gold_index, by_piece) == "tous_les_cotes"


def test_dispute_merged_side():
    gold_index = {
        "a1": {"source": "documents/x.md", "quote": "two"},
        "a2": {"source": "documents/x.md", "quote": "three"},
    }
    by_piece = {"x.md": [{"claim_id": "1", "text": "two and three"}]}
    sides = [{"assertion": "a1"}, {"assertion": "a2"}]
    assert classify_dispute(sides, gold_index, by_piece) == "cotes_fusionnes"
