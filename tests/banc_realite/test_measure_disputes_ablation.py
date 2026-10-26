# SPDX-License-Identifier: AGPL-3.0-only
"""Ablation disputes — modes et complétion (inerte)."""

from banc_realite.measure_disputes import MODES, _empty_verdict_counts, build_evidence_for_mode


class _FakeSearch:
    def __init__(self):
        self.calls = []

    def __call__(self, connection, query, *, query_vector, top_k, exclude_pieces=None):
        self.calls.append({"top_k": top_k, "exclude": sorted(exclude_pieces or [])})
        piece = "p1.md" if not exclude_pieces else "p2.md"
        return [
            {
                "kind": "claim_hit",
                "id": f"c-{len(self.calls)}",
                "score": 1.0,
                "text": "fact",
                "source": piece,
                "payload": {
                    "claim": {
                        "claim_id": f"c-{len(self.calls)}",
                        "entity_id": "ent-1",
                        "source_file_path": piece,
                        "qualifiers": {"attribute_id": "attr-1"},
                        "value": "x",
                    }
                },
            }
        ]


def test_modes_canoniques():
    assert MODES == ("temoin", "trois_tours", "completion", "trois_tours_completion")


def test_build_evidence_temoin_sans_completion(monkeypatch):
    search = _FakeSearch()
    monkeypatch.setitem(
        __import__("banc_realite.measure_disputes", fromlist=["SEARCH_BY_CONFIG"]).SEARCH_BY_CONFIG,
        "fine_seule",
        search,
    )

    def _no_completion(evidence, *, connection, vault_ref=None):
        raise AssertionError("completion ne doit pas être appelée en mode témoin")

    monkeypatch.setattr(
        "banc_realite.measure_disputes.enrichir_evidence_completion",
        _no_completion,
    )
    question = {"id": "gl-q072", "question": "Q?", "shape": "dispute"}
    evidence, meta = build_evidence_for_mode(
        object(),
        question,
        couche="fine",
        k=8,
        mode="temoin",
        embed_cache={"Q?": [0.1, 0.2]},
    )
    assert len(evidence) == 1
    assert meta["completion"]["faits_ajoutes"] == 0
    assert search.calls[0]["exclude"] == []


def test_empty_verdict_counts():
    bucket = _empty_verdict_counts()
    assert bucket["complet"] == 0
    assert set(bucket) == {"complet", "un_seul_cote", "partiel", "aucun", "autre"}


def test_compute_completion_deltas_une_fois_a_la_fin():
    """Z3 : le delta ne doit pas recompteter les gains précocents question par question."""
    from banc_realite.measure_disputes import compute_completion_deltas

    par_couche = {
        "fine": {
            "k8": {
                "temoin": {"complet": 0},
                "completion": {"complet": 2},
                "trois_tours": {"complet": 1},
                "trois_tours_completion": {"complet": 3},
            },
            "k20": {
                "temoin": {"complet": 0},
                "completion": {"complet": 1},
                "trois_tours": {"complet": 0},
                "trois_tours_completion": {"complet": 1},
            },
        },
        "brute": {
            "k8": {
                "temoin": {"complet": 0},
                "completion": {"complet": 0},
                "trois_tours": {"complet": 0},
                "trois_tours_completion": {"complet": 0},
            },
            "k20": {
                "temoin": {"complet": 0},
                "completion": {"complet": 0},
                "trois_tours": {"complet": 0},
                "trois_tours_completion": {"complet": 0},
            },
        },
        "brute_fine": {
            "k8": {
                "temoin": {"complet": 0},
                "completion": {"complet": 0},
                "trois_tours": {"complet": 0},
                "trois_tours_completion": {"complet": 0},
            },
            "k20": {
                "temoin": {"complet": 0},
                "completion": {"complet": 0},
                "trois_tours": {"complet": 0},
                "trois_tours_completion": {"complet": 0},
            },
        },
    }
    deltas = compute_completion_deltas(par_couche)
    assert deltas["completion_vs_temoin"] == 3
    assert deltas["trois_tours_completion_vs_trois_tours"] == 3

    wrong = 0
    cumul = {"temoin": 0, "completion": 0}
    for gain in (2, 1):
        cumul["completion"] += gain
        wrong += cumul["completion"] - cumul["temoin"]
    assert wrong == 5
