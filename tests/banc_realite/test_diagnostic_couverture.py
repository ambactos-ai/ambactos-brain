# SPDX-License-Identifier: AGPL-3.0-only
"""Diagnostic de couverture — doubles purs, sans base ni sous-processus."""

from __future__ import annotations

from typing import Any

from banc_realite.diagnostic_couverture import (
    CONTROLE_AU_MOINS_UNE,
    CONTROLE_TOUT_TIENT,
    VERDICTS_NON_PORTES,
    agreger_assertions,
    cause_dominante,
    compter_couverture_fine,
    diagnostiquer_assertion,
    etat_contexte_fine,
    verdict_depuis_rang,
    verifier_controle_fine,
)


def _assertion(aid: str, source: str = "docs/a.md", quote: str = "needle text") -> dict[str, Any]:
    return {"id": aid, "source": source, "quote": quote, "content": quote}


def _claim(cid: str, piece: str, text: str, *, tabular: bool = False) -> dict[str, Any]:
    return {
        "claim_id": cid,
        "piece": piece,
        "source_file_path": piece,
        "text": text,
        "norm_text": text.lower(),
        "extracted_by_skill": "tabulaire-deterministe" if tabular else "claiming",
    }


def _evidence_claim(cid: str, text: str, source: str = "docs/a.md") -> dict[str, Any]:
    return {"kind": "claim_hit", "id": cid, "text": text, "source": source}


def _evidence_passage(text: str, source: str = "docs/a.md") -> dict[str, Any]:
    return {"kind": "passage", "id": f"p-{text[:8]}", "text": text, "source": source}


def _ctx(
    *,
    fine: list[dict[str, Any]],
    brute: list[dict[str, Any]] | None = None,
    search: list[dict[str, Any]] | None = None,
    claims: list[dict[str, Any]] | None = None,
    claims_index: dict[str, dict[str, Any]] | None = None,
    claims_by_piece: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    claims = claims or []
    claims_index = claims_index or {str(c["claim_id"]): c for c in claims}
    by_piece: dict[str, list[dict[str, Any]]] = claims_by_piece or {}
    if not by_piece:
        for claim in claims:
            by_piece.setdefault(claim["piece"], []).append(claim)
    return {
        "fine_evidence": fine,
        "brute_evidence": brute or [],
        "search_evidence": search or [],
        "claims": claims,
        "claims_index": claims_index,
        "claims_by_piece": by_piece,
    }


def test_etat_contexte_fine():
    assert etat_contexte_fine({"tient": [1], "candidat": []}) == "tient"
    assert etat_contexte_fine({"tient": [], "candidat": [1]}) == "candidat"
    assert etat_contexte_fine({"tient": [], "candidat": []}) == "absent"


def test_verdict_depuis_rang():
    assert verdict_depuis_rang(12) == "retrouvable_rang_13_50"
    assert verdict_depuis_rang(50) == "retrouvable_rang_13_50"
    assert verdict_depuis_rang(51) == "retrouvable_rang_51_200"
    assert verdict_depuis_rang(200) == "retrouvable_rang_51_200"
    assert verdict_depuis_rang(None) == "hors_top_200"
    assert verdict_depuis_rang(201) == "hors_top_200"


def test_diagnostiquer_assertion_portee():
    assertion = _assertion("a1", quote="needle text")
    claim = _claim("c1", "a.md", "needle text")
    ctx = _ctx(
        fine=[_evidence_claim("c1", "needle text", "docs/a.md")],
        brute=[],
        claims=[claim],
    )
    out = diagnostiquer_assertion(assertion, **ctx)
    assert out["verdict"] == "portee"
    assert out["etat_contexte_fine"] == "tient"


def test_diagnostiquer_assertion_non_extrait():
    assertion = _assertion("a1", source="docs/vide.md", quote="absent")
    ctx = _ctx(fine=[], brute=[], claims=[], claims_by_piece={"vide.md": []})
    out = diagnostiquer_assertion(assertion, **ctx)
    assert out["verdict"] == "non_extrait"
    assert out["nb_tenants_en_base"] == 0


def test_diagnostiquer_assertion_extrait_infidele():
    assertion = _assertion("a1", source="docs/a.md", quote="attendu exact")
    claim = _claim("c1", "a.md", "autre texte")
    ctx = _ctx(fine=[], brute=[], claims=[claim])
    out = diagnostiquer_assertion(assertion, **ctx)
    assert out["verdict"] == "extrait_infidele"


def test_diagnostiquer_assertion_retrouvable_rang_13_50():
    assertion = _assertion("a1", source="docs/a.md", quote="needle text")
    claim = _claim("c1", "a.md", "needle text")
    search = [_evidence_claim("x", "noise")] + [_evidence_claim("c1", "needle text")] * 20
    ctx = _ctx(fine=[], brute=[], search=search, claims=[claim])
    out = diagnostiquer_assertion(assertion, **ctx)
    assert out["verdict"] == "retrouvable_rang_13_50"
    assert out["meilleur_rang"] == 2


def test_diagnostiquer_assertion_retrouvable_rang_51_200():
    assertion = _assertion("a1", source="docs/a.md", quote="needle text")
    claim = _claim("c1", "a.md", "needle text")
    search = [_evidence_claim("x", "noise")] * 60 + [_evidence_claim("c1", "needle text")]
    ctx = _ctx(fine=[], brute=[], search=search, claims=[claim])
    out = diagnostiquer_assertion(assertion, **ctx)
    assert out["verdict"] == "retrouvable_rang_51_200"
    assert out["meilleur_rang"] == 61


def test_diagnostiquer_assertion_hors_top_200():
    assertion = _assertion("a1", source="docs/a.md", quote="needle text")
    claim = _claim("c1", "a.md", "needle text")
    ctx = _ctx(fine=[], brute=[], search=[_evidence_claim("x", "noise")], claims=[claim])
    out = diagnostiquer_assertion(assertion, **ctx)
    assert out["verdict"] == "hors_top_200"
    assert out["meilleur_rang"] is None


def test_diagnostiquer_assertion_tabulaire_tient():
    assertion = _assertion("a1", source="docs/tab.md", quote="cell value")
    tab = _claim("t1", "tab.md", "cell value", tabular=True)
    ctx = _ctx(fine=[], brute=[], claims=[tab], claims_by_piece={"tab.md": [tab]})
    out = diagnostiquer_assertion(assertion, **ctx)
    assert out["tabulaire_tient"] is True


def test_diagnostiquer_assertion_tenant_tabulaire():
    assertion = _assertion("a1", source="docs/tab.md", quote="cell value")
    tab = _claim("t1", "tab.md", "cell value", tabular=True)
    search = [_evidence_claim("t1", "cell value", "docs/tab.md")]
    ctx = _ctx(fine=[], brute=[], search=search, claims=[tab])
    out = diagnostiquer_assertion(assertion, **ctx)
    assert out["tenant_tabulaire"] is True


def test_diagnostiquer_assertion_brute_k12_tient():
    assertion = _assertion("a1", source="docs/a.md", quote="needle text")
    ctx = _ctx(
        fine=[],
        brute=[_evidence_passage("prefix needle text suffix", "docs/a.md")],
        claims=[],
        claims_by_piece={"a.md": []},
    )
    out = diagnostiquer_assertion(assertion, **ctx)
    assert out["brute_k12_tient"] is True


def test_cause_dominante_et_departsage():
    assert cause_dominante(["non_extrait", "extrait_infidele"]) == "non_extrait"
    assert cause_dominante(["extrait_infidele", "non_extrait"]) == "non_extrait"
    assert cause_dominante(["hors_top_200", "retrouvable_rang_51_200", "hors_top_200"]) == "hors_top_200"
    tied = ["retrouvable_rang_13_50", "retrouvable_rang_51_200"]
    assert cause_dominante(tied) == "retrouvable_rang_13_50"
    assert cause_dominante(["portee"]) is None


def test_agreger_assertions_brute_k12_tient():
    lignes = [
        {"verdict": "non_extrait", "brute_k12_tient": True, "tabulaire_tient": False},
        {"verdict": "non_extrait", "brute_k12_tient": False, "tabulaire_tient": True},
        {"verdict": "hors_top_200", "brute_k12_tient": True, "tabulaire_tient": False},
        {"verdict": "portee", "brute_k12_tient": False, "tabulaire_tient": False},
    ]
    agg = agreger_assertions(lignes)
    assert agg["portees"] == 1
    assert agg["non_portees"] == 3
    assert agg["somme_verdicts"] == 3
    assert agg["par_verdict"]["non_extrait"]["brute_k12_tient"] == 1
    assert agg["par_verdict"]["non_extrait"]["tabulaire_tient"] == 1
    assert agg["par_verdict"]["hors_top_200"]["brute_k12_tient"] == 1


def test_compter_couverture_fine():
    questions = [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}]
    lignes = [
        {"question_id": "q1", "etat_contexte_fine": "tient"},
        {"question_id": "q1", "etat_contexte_fine": "tient"},
        {"question_id": "q2", "etat_contexte_fine": "tient"},
        {"question_id": "q2", "etat_contexte_fine": "absent"},
        {"question_id": "q3", "etat_contexte_fine": "absent"},
    ]
    cov = compter_couverture_fine(lignes, questions)
    assert cov["tout_tient"] == 1
    assert cov["au_moins_une"] == 2


def test_verifier_controle_fine():
    assert verifier_controle_fine({"tout_tient": CONTROLE_TOUT_TIENT, "au_moins_une": CONTROLE_AU_MOINS_UNE}) is None
    assert verifier_controle_fine({"tout_tient": 0, "au_moins_une": 86}) is not None
    assert verifier_controle_fine({"tout_tient": 48, "au_moins_une": 0}) is not None


def test_six_verdicts_couvris():
    seen = set()
    assertion = _assertion("a1", source="docs/a.md", quote="needle text")
    claim = _claim("c1", "a.md", "needle text")
    claim_bad = _claim("c2", "a.md", "autre")

    seen.add(diagnostiquer_assertion(assertion, **_ctx(fine=[_evidence_claim("c1", "needle text")], claims=[claim]))["verdict"])
    seen.add(
        diagnostiquer_assertion(
            _assertion("a2", source="docs/v.md", quote="x"),
            **_ctx(fine=[], claims=[], claims_by_piece={"v.md": []}),
        )["verdict"]
    )
    seen.add(
        diagnostiquer_assertion(
            _assertion("a3", source="docs/a.md", quote="attendu"),
            **_ctx(fine=[], claims=[claim_bad]),
        )["verdict"]
    )
    search_50 = [_evidence_claim("c1", "needle text")]
    seen.add(diagnostiquer_assertion(assertion, **_ctx(fine=[], search=search_50, claims=[claim]))["verdict"])
    search_100 = [_evidence_claim("x", "n")] * 55 + [_evidence_claim("c1", "needle text")]
    seen.add(diagnostiquer_assertion(assertion, **_ctx(fine=[], search=search_100, claims=[claim]))["verdict"])
    seen.add(diagnostiquer_assertion(assertion, **_ctx(fine=[], search=[], claims=[claim]))["verdict"])

    assert seen == set(VERDICTS_NON_PORTES) | {"portee"}
