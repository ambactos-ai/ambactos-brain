# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — issue #127 parité brain/public et pièces tabulaires par dossier."""
from __future__ import annotations

import pytest

from banc_realite.dossiers import dossier_courant
from banc_realite.fidelite import assertion_piece_tabular
from banc_realite.production import parity_counts, verifier_parite


GREENLOOP_TABULAR = frozenset(
    {
        "09-january-scale-export.md",
        "17-collection-register.md",
        "18-weigh-ticket-index.md",
        "24-customer-review.md",
        "28-collection-register-file-copy.md",
        "21-monthly-collections.csv",
        "22-recorded-visits.csv",
    }
)


class _ParityConnection:
    def __init__(self, brain_rows: list[dict], public_rows: list[dict]) -> None:
        self.brain_rows = list(brain_rows)
        self.public_rows = list(public_rows)

    def execute(self, query: str, params=None):
        sql = " ".join(query.split())
        params = params or ()
        if "FROM brain.claims WHERE vault_ref = %s" in sql and "COUNT(*)" in sql:
            vault_ref = params[0]
            return _Row(
                {
                    "n": sum(
                        1 for row in self.brain_rows if row.get("vault_ref") == vault_ref
                    )
                }
            )
        if "FROM public.claims WHERE vault_ref = %s" in sql and "COUNT(*)" in sql:
            vault_ref = params[0]
            return _Row(
                {
                    "n": sum(
                        1 for row in self.public_rows if row.get("vault_ref") == vault_ref
                    )
                }
            )
        if "SELECT claim_id FROM brain.claims WHERE vault_ref = %s" in sql:
            vault_ref = params[0]
            return _Rows(
                [
                    {"claim_id": row["claim_id"]}
                    for row in self.brain_rows
                    if row.get("vault_ref") == vault_ref
                ]
            )
        if "SELECT claim_id FROM public.claims WHERE vault_ref = %s" in sql:
            vault_ref = params[0]
            return _Rows(
                [
                    {"claim_id": row["claim_id"]}
                    for row in self.public_rows
                    if row.get("vault_ref") == vault_ref
                ]
            )
        if "value_embedding_mistral IS NOT NULL AS has_embedding" in sql:
            vault_ref = params[0]
            schema = "brain" if "brain.claims" in sql else "public"
            rows = self.brain_rows if schema == "brain" else self.public_rows
            return _Rows(
                [
                    {
                        "claim_id": row["claim_id"],
                        "has_embedding": row.get("has_embedding", False),
                    }
                    for row in rows
                    if row.get("vault_ref") == vault_ref
                ]
            )
        if "COALESCE(vault_ref" in sql and "value_embedding_mistral IS NOT NULL" in sql:
            schema = "brain" if "brain.claims" in sql else "public"
            rows = self.brain_rows if schema == "brain" else self.public_rows
            counts: dict[str, int] = {}
            for row in rows:
                if row.get("has_embedding"):
                    key = row.get("vault_ref") or "(null)"
                    counts[key] = counts.get(key, 0) + 1
            return _Rows([{"k": k, "n": n} for k, n in sorted(counts.items())])
        if "COALESCE(vault_ref" in sql:
            schema = "brain" if "brain.claims" in sql else "public"
            rows = self.brain_rows if schema == "brain" else self.public_rows
            counts = {}
            for row in rows:
                key = row.get("vault_ref") or "(null)"
                counts[key] = counts.get(key, 0) + 1
            return _Rows([{"k": k, "n": n} for k, n in sorted(counts.items())])
        if "GROUP BY 1" in sql:
            return _Rows([])
        if "COUNT(*) AS n FROM brain.claims" in sql and "vault_ref" not in sql:
            return _Row({"n": len(self.brain_rows)})
        if "COUNT(*) AS n FROM public.claims" in sql and "vault_ref" not in sql:
            return _Row({"n": len(self.public_rows)})
        if "extracted_by_skill" in sql:
            return _Rows([])
        if "lifecycle_status" in sql:
            return _Rows([])
        if "value_embedding_mistral IS NOT NULL" in sql and "vault_ref" not in sql:
            schema = "brain" if "brain.claims" in sql else "public"
            rows = self.brain_rows if schema == "brain" else self.public_rows
            return _Row({"n": sum(1 for row in rows if row.get("has_embedding"))})
        if "lex_tsv IS NULL" in sql:
            return _Row({"n": 0})
        if "FROM pg_proc" in sql:
            return _Rows([])
        raise AssertionError(f"unexpected query: {sql}")


class _Row:
    def __init__(self, data: dict):
        self._data = data

    def fetchone(self):
        return self._data


class _Rows:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


def _aligned_rows(vault_ref: str = "moreau") -> tuple[list[dict], list[dict]]:
    rows = [
        {"claim_id": "a1", "vault_ref": vault_ref, "has_embedding": True},
        {"claim_id": "a2", "vault_ref": vault_ref, "has_embedding": False},
    ]
    return rows, list(rows)


def test_verifier_parite_reussit_sur_tables_alignees(monkeypatch):
    brain, public = _aligned_rows()
    monkeypatch.setattr(
        "banc_realite.infra.connect_existing",
        lambda: _ParityConnection(brain, public),
    )
    verifier_parite("moreau")


def test_verifier_parite_echoue_sur_comptes_divergents(monkeypatch):
    brain, public = _aligned_rows()
    public = public[:1]
    monkeypatch.setattr(
        "banc_realite.infra.connect_existing",
        lambda: _ParityConnection(brain, public),
    )
    with pytest.raises(RuntimeError, match="comptes"):
        verifier_parite("moreau")


def test_verifier_parite_echoue_sur_claim_id_manquant(monkeypatch):
    brain, _public = _aligned_rows()
    public = [
        {"claim_id": "a1", "vault_ref": "moreau", "has_embedding": True},
        {"claim_id": "a3", "vault_ref": "moreau", "has_embedding": False},
    ]
    monkeypatch.setattr(
        "banc_realite.infra.connect_existing",
        lambda: _ParityConnection(brain, public),
    )
    with pytest.raises(RuntimeError, match="claim_id"):
        verifier_parite("moreau")


def test_verifier_parite_echoue_sur_vecteur_manquant(monkeypatch):
    brain, public = _aligned_rows()
    public = [{**row, "has_embedding": True} for row in public]
    monkeypatch.setattr(
        "banc_realite.infra.connect_existing",
        lambda: _ParityConnection(brain, public),
    )
    with pytest.raises(RuntimeError, match="vecteurs"):
        verifier_parite("moreau")


def test_parity_counts_rapporte_par_vault_ref():
    conn = _ParityConnection(
        [
            {"claim_id": "g1", "vault_ref": "greenloop", "has_embedding": True},
            {"claim_id": "m1", "vault_ref": "moreau", "has_embedding": True},
        ],
        [
            {"claim_id": "g1", "vault_ref": "greenloop", "has_embedding": True},
            {"claim_id": "m1", "vault_ref": "moreau", "has_embedding": True},
        ],
    )
    counts = parity_counts(conn)
    assert counts["brain_by_vault_ref"] == {"greenloop": 1, "moreau": 1}
    assert counts["public_by_vault_ref"] == {"greenloop": 1, "moreau": 1}
    assert counts["brain_embeddings_by_vault_ref"] == {"greenloop": 1, "moreau": 1}
    assert counts["public_embeddings_by_vault_ref"] == {"greenloop": 1, "moreau": 1}


def test_assertion_piece_tabular_moreau_bulletins():
    moreau = dossier_courant("moreau")
    assert assertion_piece_tabular(
        "documents/03-bulletins-paie-nov-dec-2023.md", dossier=moreau
    )
    assert assertion_piece_tabular("04-planning-heures-sup.md", dossier=moreau)
    assert not assertion_piece_tabular("01-contrat-travail-cdi.md", dossier=moreau)


def test_assertion_piece_tabular_greenloop_non_regression():
    greenloop = dossier_courant("greenloop")
    assert greenloop.pieces_tabulaires == GREENLOOP_TABULAR
    for piece in GREENLOOP_TABULAR:
        assert assertion_piece_tabular(piece, dossier=greenloop)
    assert not assertion_piece_tabular("01-other.md", dossier=greenloop)


def test_mesure_comparatif_refuse_sans_parite(monkeypatch):
    monkeypatch.setattr(
        "banc_realite.mesure_comparatif_etage2.export_runtime_env",
        lambda: None,
    )
    monkeypatch.setattr(
        "banc_realite.mesure_comparatif_etage2.load_questions",
        lambda _d: [{"id": "gl-q001"}],
    )
    monkeypatch.setattr(
        "banc_realite.mesure_comparatif_etage2.load_gold",
        lambda _d: {"assertions": [], "assertion_index": {}},
    )
    monkeypatch.setattr(
        "banc_realite.mesure_comparatif_etage2.verifier_or_coherent",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "banc_realite.mesure_comparatif_etage2.verifier_parite",
        lambda _d: (_ for _ in ()).throw(RuntimeError("parité moreau")),
    )
    from banc_realite.mesure_comparatif_etage2 import run_mesure

    with pytest.raises(RuntimeError, match="parité"):
        run_mesure(fournisseurs=["mem0"], ks=[8], budgets=[4000])


def test_temoin_brassage_refuse_sans_parite(monkeypatch):
    monkeypatch.setattr(
        "banc_realite.temoin_brassage.load_questions",
        lambda _d: [{"id": "gl-q001"}, {"id": "gl-q002"}],
    )
    monkeypatch.setattr(
        "banc_realite.temoin_brassage.load_gold",
        lambda _d: {"assertions": [], "assertion_index": {}},
    )
    monkeypatch.setattr(
        "banc_realite.temoin_brassage.verifier_or_coherent",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "banc_realite.temoin_brassage.verifier_parite",
        lambda _d: (_ for _ in ()).throw(RuntimeError("parité greenloop")),
    )
    from banc_realite.temoin_brassage import run_temoin_brassage

    with pytest.raises(RuntimeError, match="parité"):
        run_temoin_brassage(fournisseurs=["mem0"])
