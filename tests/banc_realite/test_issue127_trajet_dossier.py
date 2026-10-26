# SPDX-License-Identifier: AGPL-3.0-only
"""Issue #127 — aucune constante GreenLoop sur le trajet de mesure MOREAU."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from banc_realite.dossiers import dossier_courant
from banc_realite.fournisseurs_contexte import BrainBanc, BrainProd, RagNaif
from banc_realite.mesure_porte_phrase import build_evidence_for_porte_phrase_config
from banc_realite.restitution import compute_verdict, verdict_juridique_support


class _SqlSpyConnection:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def execute(self, query: str, params=None):
        self.queries.append(" ".join(query.split()))
        return _Rows([])


class _Rows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


def _present_all(_aid: str):
    return True, 1


def _present_none(_aid: str):
    return False, None


def test_verdict_juridique_complet_partiel_absent():
    question = {"shape": "citation", "supporting_assertions": ["a1", "a2"]}
    complet = verdict_juridique_support(question, _present_all)
    assert complet["verdict"] == "complet"
    partiel = verdict_juridique_support(question, lambda aid: (aid == "a1", 1))
    assert partiel["verdict"] == "partiel"
    absent = verdict_juridique_support(question, _present_none)
    assert absent["verdict"] == "absent"


def test_verdict_non_etabli_hors_etage2():
    question = {"id": "MOR-Q30", "shape": "non_etabli", "supporting_assertions": []}
    block = compute_verdict(
        question,
        [],
        gold={"assertion_index": {}, "disputes": [], "situations": [], "trajectories": []},
        holder_table={},
        claims_index={},
        present=_present_none,
    )
    assert block["verdict"] == "hors_etage2"


def test_verdict_greenloop_dispute_non_regression():
    gold = {
        "assertion_index": {
            "a1": {"id": "a1", "source": "x.md", "quote": "one"},
            "a2": {"id": "a2", "source": "x.md", "quote": "two"},
        },
        "disputes": [
            {
                "id": "d1",
                "question": "How many?",
                "sides": [{"assertion": "a1"}, {"assertion": "a2"}],
            }
        ],
        "situations": [],
        "trajectories": [],
    }
    question = {"shape": "dispute", "question": "How many?", "supporting_assertions": ["a1", "a2"]}
    block = compute_verdict(
        question,
        [],
        gold=gold,
        holder_table={},
        claims_index={},
        present=lambda aid: (aid == "a1", 1),
    )
    assert block["verdict"] == "un_seul_cote"


@pytest.mark.parametrize(
    "provider_cls",
    [RagNaif, BrainProd, BrainBanc],
)
def test_aucune_constante_greenloop_sur_le_trajet(provider_cls, monkeypatch, tmp_path):
    moreau = dossier_courant("moreau")
    embed = {"q1": [0.1, 0.2]}
    question = {"id": "MOR-Q01", "question": "q1", "shape": "citation", "supporting_assertions": []}

    if provider_cls is BrainBanc:
        cache_path = tmp_path / "phrases.json"
        cache_path.write_text(json.dumps({"fenetre": [0.3, 0.4]}), encoding="utf-8")
        provider = provider_cls(embed, phrases_cache={"fenetre": [0.3, 0.4]}, dossier=moreau)
        monkeypatch.setattr(type(moreau), "cache_phrases_embeddings", property(lambda _s, p=cache_path: p))

        def fake_build(connection, q, config, *, embed_cache, phrases_cache, dossier=None):
            assert dossier is moreau
            return [{"kind": "file_hit", "id": "1", "source": "01-contrat-travail-cdi.md"}], None, {}

        monkeypatch.setattr(
            "banc_realite.fournisseurs_contexte.build_evidence_for_porte_phrase_config",
            fake_build,
        )
    elif provider_cls is BrainProd:
        monkeypatch.setattr(
            "banc_realite.fournisseurs_contexte.compile_cadre_prod",
            lambda *_a, **_k: {"evidence": [{"source": "01-contrat-travail-cdi.md"}]},
        )
        provider = provider_cls(embed, dossier=moreau)
    else:
        spy = _SqlSpyConnection()

        def fake_search(connection, query, *, query_vector, top_k, vault_ref=None):
            assert vault_ref == "moreau"
            assert "greenloop" not in str(query).lower()
            return [{"source": "01-contrat-travail-cdi.md"}]

        monkeypatch.setattr(
            "banc_realite.fournisseurs_contexte.search_passages_hybrid",
            fake_search,
        )
        provider = provider_cls(embed, dossier=moreau)
        out = provider.contexte(question, k=8, connection=spy)
        assert out
        for sql in spy.queries:
            assert "greenloop" not in sql.lower()
        assert moreau.responsibility_context == "MOREAU"
        return

    out = provider.contexte(question, k=8, connection=object())
    assert out
    assert moreau.responsibility_context == "MOREAU"
    opened = [str(path) for path in tmp_path.iterdir()] if provider_cls is BrainBanc else []
    assert not any("sorties" in path for path in opened)


def test_rag_naif_echoue_sans_vault_ref(monkeypatch):
    moreau = dossier_courant("moreau")
    captured: dict[str, Any] = {}

    def fake_search(connection, query, *, query_vector, top_k, vault_ref=None):
        captured["vault_ref"] = vault_ref
        return []

    monkeypatch.setattr("banc_realite.fournisseurs_contexte.search_passages_hybrid", fake_search)
    provider = RagNaif({"q1": [0.1]}, dossier=moreau)
    provider.contexte({"question": "q1"}, k=8, connection=object())
    assert captured.get("vault_ref") != "greenloop"
