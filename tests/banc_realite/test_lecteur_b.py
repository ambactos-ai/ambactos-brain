# SPDX-License-Identifier: AGPL-3.0-only
import json
import os
from pathlib import Path

import pytest

from banc_realite import accord as acc
from banc_realite import lecteur_b as lb


@pytest.fixture
def sample_paths(tmp_path):
    donnees = {
        "seed": 20260920,
        "date": "2026-09-20",
        "residu": [
            {
                "id": "gl-a1",
                "piece": "pipe.md",
                "page": 1,
                "perspective": "hidden",
                "content": "Le total | est < 100> {ok}",
                "quote": "total | est",
                "extrait_source": "→    2  ligne avec | et <tags>",
                "extrait_status": "ok",
                "voisins": [
                    {
                        "claim_id": "cid-1",
                        "voie": "modele",
                        "triplet": "entite · attr : valeur",
                        "autres_qualifiers": {"site": "A"},
                        "texte": "fallback",
                    }
                ],
            }
        ],
        "hors_or": [
            {
                "claim_id": "cid-2",
                "voie": "tabulaire",
                "triplet": "Section A / Key: Value",
                "autres_qualifiers": {},
                "piece": "pipe.md",
                "lignes_source": "ligne 3",
                "extrait_source": "    3  extrait hors-or",
                "extrait_status": "ok",
                "texte": "valeur tabulaire",
            }
        ],
        "stats": {},
    }
    paires = [
        {
            "assertion": {
                "id": "gl-assert",
                "piece": "pipe.md",
                "content": "Contenu assertion",
                "quote": "Citation",
                "extrait": "→    4  extrait assertion",
            },
            "fait": {"claim_id": "f1", "triplet": "fait un"},
            "overlap": 0.5,
            "questions": ["gl-q001"],
        },
        {
            "assertion": {
                "id": "gl-assert",
                "piece": "pipe.md",
                "content": "Contenu assertion",
                "quote": "Citation",
                "extrait": "→    4  extrait assertion",
            },
            "fait": {"claim_id": "f2", "triplet": "fait deux"},
            "overlap": 0.2,
            "questions": ["gl-q001"],
        },
    ]
    donnees_path = tmp_path / "relecture-donnees.json"
    paires_path = tmp_path / "etage2-paires-a-lire.json"
    donnees_path.write_text(json.dumps(donnees, ensure_ascii=False), encoding="utf-8")
    paires_path.write_text(json.dumps(paires, ensure_ascii=False), encoding="utf-8")
    return {
        "donnees": donnees,
        "donnees_path": donnees_path,
        "paires_path": paires_path,
        "invites_path": tmp_path / "lecteur-b-invites.jsonl",
        "calls_path": tmp_path / "lecteur-b-calls.jsonl",
        "verdicts_relecture_path": tmp_path / "relecture-verdicts-lecteur-b.json",
        "verdicts_paires_path": tmp_path / "paires-verdicts-lecteur-b.json",
    }


def test_invites_are_blind_and_preserve_special_chars(sample_paths):
    invites = lb.build_invites(
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
    )
    assert len(invites) == 3
    residu = next(item for item in invites if item["id"] == "R01")
    assert "Le total | est < 100> {ok}" in residu["user"]
    assert "ligne avec | et <tags>" in residu["user"]
    assert lb.GRILLE_RESIDU in residu["user"]
    assert lb.GRILLE_HORS_OR in next(item for item in invites if item["id"] == "H01")["user"]
    assertion = next(item for item in invites if item["id"] == "A-gl-assert")
    assert "fait un" in assertion["user"]
    assert "fait deux" in assertion["user"]
    errors = lb.audit_invites_for_blindness(invites, donnees_path=sample_paths["donnees_path"])
    assert errors == []


def test_invites_fail_if_forbidden_tokens_present(sample_paths):
    invites = lb.build_invites(
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
    )
    invites[0]["user"] += "\nvoie tabulaire"
    errors = lb.audit_invites_for_blindness(invites, donnees_path=sample_paths["donnees_path"])
    assert any("voie" in err for err in errors)


def test_validate_relecture_response_and_illisible_retry():
    excerpt = "→    2  ligne avec | et <tags>"
    ok = lb.validate_relecture_response(
        {"verdict": "1", "lignes": [2], "manque": "", "justification": "ok"},
        partie="residu",
        excerpt=excerpt,
    )
    assert ok["ok"] is True
    assert ok["preuve_verifiable"] is True
    bad_lines = lb.validate_relecture_response(
        {"verdict": "1", "lignes": [99], "manque": "", "justification": "ok"},
        partie="residu",
        excerpt=excerpt,
    )
    assert bad_lines["preuve_verifiable"] is False

    calls = 0

    def chat(_system, user):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "pas du json", {}
        return '{"verdict":"4","lignes":[],"manque":"","justification":"absente"}', {}

    invite = {
        "id": "R01",
        "partie": "residu",
        "sha256": "x",
        "system": lb.SYSTEM_PROMPT,
        "user": "invite",
    }
    data = {"residu": [{"extrait_source": excerpt}], "hors_or": []}
    result = lb.adjudicate_one(
        invite,
        chat_fn=chat,
        calls_path=Path("/dev/null"),
        data=data,
        paires_grouped={},
    )
    assert calls == 2
    assert result["verdict"] == "4"


def test_illisible_after_two_bad_responses(tmp_path):
    calls = 0

    def chat(_system, _user):
        nonlocal calls
        calls += 1
        return "toujours invalide", {}

    invite = {
        "id": "H01",
        "partie": "hors_or",
        "sha256": "x",
        "system": lb.SYSTEM_PROMPT,
        "user": "invite",
    }
    calls_path = tmp_path / "calls.jsonl"
    lb.ensure_calls_header(calls_path)
    result = lb.adjudicate_one(
        invite,
        chat_fn=chat,
        calls_path=calls_path,
        data={"residu": [], "hors_or": [{"extrait_source": "    1  ligne"}]},
        paires_grouped={},
    )
    assert result["verdict"] == "illisible"
    assert calls == 2


def test_call_cap_and_resume(sample_paths, tmp_path):
    seen = {"count": 0}

    def chat(_system, _user):
        seen["count"] += 1
        return '{"verdict":"1","lignes":[2],"manque":"","justification":"ok"}', {}

    lb.run(
        chat,
        parties=("residu",),
        limite=1,
        dry_run=False,
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
        invites_path=sample_paths["invites_path"],
        calls_path=sample_paths["calls_path"],
        relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
        paires_verdicts_path=sample_paths["verdicts_paires_path"],
    )
    assert seen["count"] == 1
    lb.run(
        chat,
        parties=("residu",),
        limite=2,
        dry_run=False,
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
        invites_path=sample_paths["invites_path"],
        calls_path=sample_paths["calls_path"],
        relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
        paires_verdicts_path=sample_paths["verdicts_paires_path"],
    )
    assert seen["count"] == 1

    def chat_fail(_system, _user):
        seen["count"] += 1
        return "x", {}

    lb.MAX_CALLS = 1
    with pytest.raises(RuntimeError, match="plafond"):
        lb.run(
            chat_fail,
            parties=("residu", "hors_or"),
            dry_run=False,
            donnees_path=sample_paths["donnees_path"],
            paires_path=sample_paths["paires_path"],
            invites_path=sample_paths["invites_path"],
            calls_path=tmp_path / "calls-cap.jsonl",
            relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
            paires_verdicts_path=sample_paths["verdicts_paires_path"],
        )
    lb.MAX_CALLS = 140


def test_relay_factory_raises_under_pytest():
    os.environ["PYTEST_CURRENT_TEST"] = "tests/banc_realite/test_lecteur_b.py::test_relay"
    try:
        with pytest.raises(RuntimeError, match="interdit sous pytest"):
            lb.make_relay_chat_fn()
    finally:
        del os.environ["PYTEST_CURRENT_TEST"]


def test_dry_run_writes_invites(sample_paths):
    result = lb.run(
        lambda _s, _u: ("", {}),
        dry_run=True,
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
        invites_path=sample_paths["invites_path"],
        calls_path=sample_paths["calls_path"],
        relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
        paires_verdicts_path=sample_paths["verdicts_paires_path"],
    )
    assert result["invites"] == 3
    lines = sample_paths["invites_path"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


FORMAT_SUFFIX = {
    "residu": lb.FORMAT_CONSIGNE_RESIDU,
    "hors_or": lb.FORMAT_CONSIGNE_HORS_OR,
    "assertion": lb.FORMAT_CONSIGNE_ASSERTION,
}


def test_invites_end_with_partie_format_consigne(sample_paths):
    invites = lb.build_invites(
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
    )
    for invite in invites:
        suffix = FORMAT_SUFFIX[invite["partie"]]
        assert invite["user"].rstrip().endswith(suffix)


def test_resume_preserves_justification_from_journal(sample_paths):
    justification = "justification persistante du premier passage"

    def chat(_system, _user):
        return (
            json.dumps(
                {
                    "verdict": "1",
                    "lignes": [2],
                    "manque": "",
                    "justification": justification,
                }
            ),
            {},
        )

    lb.run(
        chat,
        parties=("residu",),
        limite=1,
        dry_run=False,
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
        invites_path=sample_paths["invites_path"],
        calls_path=sample_paths["calls_path"],
        relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
        paires_verdicts_path=sample_paths["verdicts_paires_path"],
    )
    lb.run(
        chat,
        parties=("residu", "hors_or", "assertion"),
        dry_run=False,
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
        invites_path=sample_paths["invites_path"],
        calls_path=sample_paths["calls_path"],
        relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
        paires_verdicts_path=sample_paths["verdicts_paires_path"],
    )
    verdicts = json.loads(sample_paths["verdicts_relecture_path"].read_text(encoding="utf-8"))
    assert verdicts["residu"]["R01"][1] == justification


def test_limite_omits_unprocessed_and_marks_incomplete(sample_paths):
    def chat(_system, _user):
        return '{"verdict":"1","lignes":[2],"manque":"","justification":"ok"}', {}

    lb.run(
        chat,
        parties=("residu", "hors_or"),
        limite=1,
        dry_run=False,
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
        invites_path=sample_paths["invites_path"],
        calls_path=sample_paths["calls_path"],
        relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
        paires_verdicts_path=sample_paths["verdicts_paires_path"],
    )
    verdicts = json.loads(sample_paths["verdicts_relecture_path"].read_text(encoding="utf-8"))
    assert "H01" not in verdicts["hors_or"]
    assert verdicts["complet"] is False
    assert "H01" in verdicts["manquants"]
    assert not acc.is_complete_b(verdicts, jeu="relecture")


def test_call_cap_counts_each_chat_attempt(sample_paths, tmp_path):
    calls = 0

    def chat(_system, _user):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "pas du json", {}
        return '{"verdict":"1","lignes":[2],"manque":"","justification":"ok"}', {}

    calls_path = tmp_path / "calls-cap.jsonl"
    lb.MAX_CALLS = 2
    lb.run(
        chat,
        parties=("residu",),
        limite=1,
        dry_run=False,
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
        invites_path=sample_paths["invites_path"],
        calls_path=calls_path,
        relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
        paires_verdicts_path=sample_paths["verdicts_paires_path"],
    )
    assert calls == 2
    assert lb.count_calls(calls_path) == 2

    calls_path2 = tmp_path / "calls-cap2.jsonl"

    def chat_retry_then_cap(_system, _user):
        nonlocal calls
        calls += 1
        if calls % 2 == 1:
            return "pas du json", {}
        return '{"verdict":"1","lignes":[2],"manque":"","justification":"ok"}', {}

    calls = 0
    with pytest.raises(RuntimeError, match="plafond"):
        lb.run(
            chat_retry_then_cap,
            parties=("residu", "hors_or"),
            limite=2,
            dry_run=False,
            donnees_path=sample_paths["donnees_path"],
            paires_path=sample_paths["paires_path"],
            invites_path=sample_paths["invites_path"],
            calls_path=calls_path2,
            relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
            paires_verdicts_path=sample_paths["verdicts_paires_path"],
        )
    lb.MAX_CALLS = 140


def test_resume_stops_when_invite_hash_changes(sample_paths):
    def chat(_system, _user):
        return '{"verdict":"1","lignes":[2],"manque":"","justification":"ok"}', {}

    lb.run(
        chat,
        parties=("residu",),
        limite=1,
        dry_run=False,
        donnees_path=sample_paths["donnees_path"],
        paires_path=sample_paths["paires_path"],
        invites_path=sample_paths["invites_path"],
        calls_path=sample_paths["calls_path"],
        relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
        paires_verdicts_path=sample_paths["verdicts_paires_path"],
    )
    sample_paths["donnees"]["residu"][0]["content"] = "contenu modifié"
    sample_paths["donnees_path"].write_text(
        json.dumps(sample_paths["donnees"], ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="invite modifiée"):
        lb.run(
            chat,
            parties=("residu",),
            limite=1,
            dry_run=False,
            donnees_path=sample_paths["donnees_path"],
            paires_path=sample_paths["paires_path"],
            invites_path=sample_paths["invites_path"],
            calls_path=sample_paths["calls_path"],
            relecture_verdicts_path=sample_paths["verdicts_relecture_path"],
            paires_verdicts_path=sample_paths["verdicts_paires_path"],
        )


def test_multiline_neighbor_is_single_numbered_block(sample_paths):
    sample_paths["donnees"]["residu"][0]["voisins"] = [
        {
            "claim_id": "cid-1",
            "voie": "modele",
            "triplet": "ligne un\nligne deux\nligne trois",
            "autres_qualifiers": {},
            "texte": "fallback",
        }
    ]
    sample_paths["donnees_path"].write_text(
        json.dumps(sample_paths["donnees"], ensure_ascii=False),
        encoding="utf-8",
    )
    residu = next(
        item
        for item in lb.build_invites(
            donnees_path=sample_paths["donnees_path"],
            paires_path=sample_paths["paires_path"],
        )
        if item["id"] == "R01"
    )
    block = residu["user"].split("Ce que la mémoire a de plus proche :", 1)[1]
    block = block.split(lb.FORMAT_CONSIGNE_RESIDU)[0].strip()
    assert block.startswith("1. ligne un\n    ligne deux\n    ligne trois")
    assert "\n2." not in block


def test_multiline_assertion_fact_is_single_numbered_block(sample_paths):
    paires = json.loads(sample_paths["paires_path"].read_text(encoding="utf-8"))
    paires.append(
        {
            "assertion": {
                "id": "gl-assert",
                "piece": "pipe.md",
                "content": "Contenu assertion",
                "quote": "Citation",
                "extrait": "→    4  extrait assertion",
            },
            "fait": {
                "claim_id": "f3",
                "triplet": "entête tabulaire\nvaleur ligne deux",
            },
            "overlap": 0.1,
            "questions": ["gl-q001"],
        }
    )
    sample_paths["paires_path"].write_text(json.dumps(paires, ensure_ascii=False), encoding="utf-8")
    assertion = next(
        item
        for item in lb.build_invites(
            donnees_path=sample_paths["donnees_path"],
            paires_path=sample_paths["paires_path"],
        )
        if item["id"] == "A-gl-assert"
    )
    facts = assertion["user"].split("Faits candidats :", 1)[1]
    facts = facts.split(lb.FORMAT_CONSIGNE_ASSERTION)[0].strip()
    assert "3. entête tabulaire\n    valeur ligne deux" in facts
    assert "\n4." not in facts
