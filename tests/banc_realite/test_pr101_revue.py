# SPDX-License-Identifier: AGPL-3.0-only
"""PR101 V2–V4 — revue Codex : quantité, portée temporelle, appariement prose."""

from banc_realite.classer_couple import classer_couple, normaliser_quantite
from banc_realite.fidelite import match_assertion


def test_normaliser_quantite_dates_iso_ne_sont_pas_des_quantites():
    left = normaliser_quantite("2024-01-01")
    right = normaliser_quantite("2024-12-31")
    assert left is None
    assert right is None


def test_normaliser_quantite_vraies_quantites_distinctes():
    left = normaliser_quantite("12 t")
    right = normaliser_quantite("13 t")
    assert left is not None and right is not None
    assert left["valeur_base"] != right["valeur_base"]


def test_classer_couple_portee_temporelle_incomplete_est_indecidable():
    a = {
        "entity_id": "station_house",
        "value": "12",
        "qualifiers": {"attribute_id": "collections", "since": "2024-01", "site": "Station House"},
    }
    b = {
        "entity_id": "station_house",
        "value": "13",
        "qualifiers": {"attribute_id": "collections", "site": "Station House"},
    }
    out = classer_couple(a, b)
    assert out["issue"] == "indecidable"
    assert "portee" in out["raison"].lower() or "portée" in out["raison"].lower()


def test_match_assertion_prose_ne_pas_apparier_sur_un_seul_jeton():
    assertion = {
        "quote": "",
        "content": "In 2024 the amount was 500",
    }
    elements = [
        {
            "claim_id": "cell-year",
            "text": "2024",
            "entity_id": "station_house",
            "qualifiers": {"tabular_scope": "cell", "attribute_id": "collections"},
        },
        {
            "claim_id": "cell-amount",
            "text": "500",
            "entity_id": "station_house",
            "qualifiers": {"tabular_scope": "cell", "attribute_id": "collections"},
        },
    ]
    for element in elements:
        hit, kind = match_assertion(assertion, [element])
        assert hit is None
        assert kind is None


def test_classer_couple_sans_attribute_id_est_indecidable():
    a = {
        "entity_id": "22_recorded_visits",
        "value": "12",
        "qualifiers": {"tabular_scope": "row"},
    }
    b = {
        "entity_id": "22_recorded_visits",
        "value": "13",
        "qualifiers": {"tabular_scope": "row"},
    }
    out = classer_couple(a, b)
    assert out["issue"] == "indecidable"
    assert "attribut" in out["raison"].lower()


def test_match_assertion_tabulaire_prose_vrai_cas():
    assertion = {
        "quote": "",
        "content": "Station House in 2024-01 had 12 collections.",
    }
    elements = [
        {
            "claim_id": "cell-collections",
            "text": "12",
            "entity_id": "station_house",
            "qualifiers": {"tabular_scope": "cell", "attribute_id": "collections", "month": "2024-01"},
        }
    ]
    hit, kind = match_assertion(assertion, elements)
    assert hit is elements[0]
    assert kind == "cellule"
