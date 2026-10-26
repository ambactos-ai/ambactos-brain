# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne pour l'or juridique MOREAU."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from banc_realite.moreau_or import (
    FactCoverageError,
    IncompleteCitationError,
    QuoteNotFoundError,
    SupportBindingError,
    _reflow_chunks,
    build_assertions,
    build_questions,
    construire,
    find_quote_for_piece,
    normalize_verbatim,
    values_equivalent_for_coverage,
    verify_facts_coverage,
    verify_question_support_bindings,
    verify_quote,
    _quote_carries_date,
    _quote_carries_text,
)


def test_verify_quote_accepts_exact_substring():
    text = "La salariée perçoit 2 450,00 euros par mois."
    quote = "2 450,00 euros par mois."
    assert verify_quote(quote, text)


def test_verify_quote_rejects_invented_text():
    text = "La salariée perçoit 2 450,00 euros par mois."
    assert not verify_quote("2 450 euros mensuels", text)


def test_find_quote_rejects_invented_candidate():
    text = "Incident survenu en 1986 avec 64 heures reconnues."
    with pytest.raises(QuoteNotFoundError):
        find_quote_for_piece(
            "piece.md",
            text,
            ["86"],
            preferred_substrings=["86 heures exactes inventées"],
        )


def test_numeric_discriminant_not_matched_as_substring():
    text = "Réunion tenue en 2023 avec 64 heures."
    quote, _line = find_quote_for_piece("piece.md", text, ["64"], preferred_substrings=["64 heures"])
    assert "64 heures" in quote
    assert "2023" not in quote or "64 heures" in quote
    assert verify_quote(quote, text)


def test_assertion_keeps_document_date_form(tmp_path: Path):
    docs = tmp_path / "documents"
    docs.mkdir()
    piece = "01-contrat.md"
    text = "Fait à Orléans, le 15 septembre 2019, en deux exemplaires.\n"
    (docs / piece).write_text(text, encoding="utf-8")
    golden = tmp_path / "GOLDEN-faits.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "facts": [
                    {
                        "id": "F01",
                        "resume": "Embauche le 15 septembre 2019",
                        "pieces": ["01-contrat.pdf"],
                        "date": "2019-09-15",
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    corpus = {piece: text}
    meta = yaml.safe_load(golden.read_text(encoding="utf-8"))
    assertions = build_assertions(meta, corpus)
    assert assertions[0]["forme_date"] == "15 septembre 2019"
    assert "2019-09-15" not in assertions[0]["quote"]


def test_reflow_merges_three_line_prose_into_one_citation():
    text = (
        "L'Employeur engage la Salariée à compter du 15 septembre 2019, sous\n"
        "réserve de la visite d'information et de prévention, en qualité de\n"
        "Technicienne de maintenance industrielle, statut ETAM, classification niveau IV."
    )
    chunks = _reflow_chunks(text)
    prose = [chunk for chunk, _line, is_special in chunks if not is_special]
    assert len(prose) == 1
    assert prose[0].startswith("L'Employeur engage")
    assert prose[0].endswith("niveau IV.")


def test_table_line_stays_intact_after_reflow():
    text = "Heures supplémentaires 25 %                          12 h 20,00              240,00 €\n"
    chunks = _reflow_chunks(text)
    assert len(chunks) == 1
    assert chunks[0][0] == text.strip()
    assert chunks[0][2] is True


def test_construire_refuses_lowercase_prose_citation(tmp_path: Path, monkeypatch):
    from banc_realite import moreau_or

    docs = tmp_path / "documents"
    docs.mkdir()
    piece = "piece.md"
    (docs / piece).write_text("phrase 999 coupée en\nmilieu de mot.", encoding="utf-8")
    golden = tmp_path / "GOLDEN-faits.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "facts": [
                    {
                        "id": "F01",
                        "resume": "phrase 999 coupée",
                        "pieces": ["piece.pdf"],
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    gold_dir = tmp_path / "gold"
    monkeypatch.setattr(moreau_or, "GOLDEN_PATH", golden)
    monkeypatch.setattr(moreau_or, "DOCS_DIR", docs)
    monkeypatch.setattr(moreau_or, "GOLD_DIR", gold_dir)
    monkeypatch.setattr(moreau_or, "ASSERTIONS_PATH", gold_dir / "assertions.json")
    monkeypatch.setattr(moreau_or, "QUESTIONS_PATH", gold_dir / "questions.json")
    monkeypatch.setattr(moreau_or, "OR_JURIDIQUE_MD", gold_dir / "OR-JURIDIQUE.md")
    monkeypatch.setattr(moreau_or, "COUVERTURE_PATH", gold_dir / "couverture.json")
    monkeypatch.setattr(moreau_or, "RAPPORT_OR_PATH", gold_dir / "RAPPORT-OR.md")

    with pytest.raises((IncompleteCitationError, QuoteNotFoundError)):
        construire(
            golden_path=golden,
            docs_dir=docs,
            gold_dir=gold_dir,
            extraction_report=None,
        )


def test_verify_quote_normalizes_spaces_and_hyphen_breaks():
    piece = "La SAS ATELIER RIVIÈRE, immatriculée sous le n° 749 382 615, au\ncapital de 80 000 euros."
    quote = "La SAS ATELIER RIVIÈRE, immatriculée sous le n° 749 382 615, au capital de 80 000 euros."
    assert normalize_verbatim(piece) != piece
    assert verify_quote(quote, piece)


def test_non_etabli_question_has_no_supporting_assertions():
    from banc_realite.moreau_extraction import GOLDEN_PATH, load_corpus

    meta = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))
    corpus = load_corpus()
    assertions = build_assertions(meta, corpus)
    questions = build_questions(meta, assertions)
    empty = [q for q in questions if q["shape"] == "non_etabli" and not q["supporting_assertions"]]
    assert empty


def test_values_equivalent_for_coverage_amounts():
    assert values_equivalent_for_coverage("1 720 €", "1720,00", kind="montant")
    assert values_equivalent_for_coverage("1 720,00 €", "1720.00", kind="montant")


def test_document_date_satisfies_coverage_iso_does_not():
    assert _quote_carries_date("Convocation pour le 12 janvier 2024.", "2024-01-12")
    assert not _quote_carries_date("Convocation pour le 2024-01-12.", "2024-01-12")


def test_rg_compared_as_text_not_number():
    assert _quote_carries_text("R.G. n° 24/01984", "24/01984")
    assert not _quote_carries_text("R.G. n° 24/01984", "2401984")


def test_construire_refuses_incomplete_fact_coverage(tmp_path: Path, monkeypatch):
    from banc_realite import moreau_or

    docs = tmp_path / "documents"
    docs.mkdir()
    piece = "piece.md"
    (docs / piece).write_text(
        "Document du 12 janvier 2024 avec un montant de 1 720,00 euros.\n" + ("x" * 250),
        encoding="utf-8",
    )
    golden = tmp_path / "GOLDEN-faits.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "facts": [
                    {
                        "id": "F01",
                        "resume": "Fait du 12 janvier 2024 pour 1 720 €",
                        "pieces": ["piece.pdf"],
                        "montant_eur": 1720,
                        "date": "2024-01-12",
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    gold_dir = tmp_path / "gold"
    monkeypatch.setattr(moreau_or, "GOLDEN_PATH", golden)
    monkeypatch.setattr(moreau_or, "DOCS_DIR", docs)
    monkeypatch.setattr(moreau_or, "GOLD_DIR", gold_dir)
    monkeypatch.setattr(moreau_or, "ASSERTIONS_PATH", gold_dir / "assertions.json")
    monkeypatch.setattr(moreau_or, "QUESTIONS_PATH", gold_dir / "questions.json")
    monkeypatch.setattr(moreau_or, "OR_JURIDIQUE_MD", gold_dir / "OR-JURIDIQUE.md")
    monkeypatch.setattr(moreau_or, "COUVERTURE_PATH", gold_dir / "couverture.json")
    monkeypatch.setattr(moreau_or, "RAPPORT_OR_PATH", gold_dir / "RAPPORT-OR.md")
    monkeypatch.setattr(
        moreau_or,
        "build_assertions",
        lambda meta, corpus: [
            {
                "id": "MOR-A01",
                "fait": "F01",
                "piece": piece,
                "quote": "Document du 12 janvier 2024 avec du texte complémentaire sans montant.",
                "ligne": 1,
                "nature": "date",
                "verbatim_normalise": True,
                "forme_date": "12 janvier 2024",
            }
        ],
    )
    monkeypatch.setattr(moreau_or, "build_questions", lambda meta, assertions: [])

    with pytest.raises(FactCoverageError):
        construire(
            golden_path=golden,
            docs_dir=docs,
            gold_dir=gold_dir,
            extraction_report=None,
        )


def test_construire_refuses_support_without_answer_elements(tmp_path: Path, monkeypatch):
    from banc_realite import moreau_or

    docs = tmp_path / "documents"
    docs.mkdir()
    piece = "07-lettre.md"
    (docs / piece).write_text(
        "Orléans-sur-Clain, le 28 janvier 2024\n"
        "Nous notifions votre licenciement pour cause réelle et sérieuse.\n",
        encoding="utf-8",
    )
    golden = tmp_path / "GOLDEN-faits.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "facts": [
                    {
                        "id": "F09",
                        "resume": "Licenciement pour cause réelle et sérieuse",
                        "pieces": ["07-lettre.pdf"],
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    gold_dir = tmp_path / "gold"
    monkeypatch.setattr(moreau_or, "GOLDEN_PATH", golden)
    monkeypatch.setattr(moreau_or, "DOCS_DIR", docs)
    assertions = [
        {
            "id": "MOR-A98",
            "fait": "F09",
            "piece": piece,
            "quote": "Orléans-sur-Clain, le 28 janvier 2024",
            "ligne": 1,
            "nature": "date",
            "verbatim_normalise": True,
            "forme_date": "28 janvier 2024",
        },
        {
            "id": "MOR-A99",
            "fait": "F09",
            "piece": piece,
            "quote": "Nous notifions votre licenciement pour cause réelle et sérieuse.",
            "ligne": 2,
            "nature": "qualification",
            "verbatim_normalise": True,
        },
    ]
    questions = [
        {
            "id": "MOR-Q99",
            "shape": "qualification",
            "question": "Motif ?",
            "answer": "Licenciement pour cause réelle et sérieuse.",
            "supporting_assertions": ["MOR-A98"],
            "must_mention": ["cause réelle et sérieuse"],
            "must_not_claim": [],
            "difficulty": "easy",
            "why_it_matters": "test",
        }
    ]
    binding = verify_question_support_bindings(questions, {a["id"]: a for a in assertions})
    assert not binding["ok"]

    monkeypatch.setattr(moreau_or, "build_assertions", lambda meta, corpus: assertions)
    monkeypatch.setattr(moreau_or, "build_questions", lambda meta, assertions: questions)
    with pytest.raises(SupportBindingError):
        construire(
            golden_path=golden,
            docs_dir=docs,
            gold_dir=gold_dir,
            extraction_report=None,
        )
    assert not (gold_dir / "assertions.json").exists()


def test_construire_accepts_support_carrying_answer(tmp_path: Path, monkeypatch):
    from banc_realite import moreau_or

    docs = tmp_path / "documents"
    docs.mkdir()
    piece = "07-lettre.md"
    (docs / piece).write_text(
        "Orléans-sur-Clain, le 28 janvier 2024\n"
        "Nous notifions votre licenciement pour cause réelle et sérieuse.\n",
        encoding="utf-8",
    )
    golden = tmp_path / "GOLDEN-faits.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "facts": [
                    {
                        "id": "F09",
                        "resume": "Licenciement pour cause réelle et sérieuse",
                        "pieces": ["07-lettre.pdf"],
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    gold_dir = tmp_path / "gold"
    monkeypatch.setattr(moreau_or, "GOLDEN_PATH", golden)
    monkeypatch.setattr(moreau_or, "DOCS_DIR", docs)
    monkeypatch.setattr(
        moreau_or,
        "build_assertions",
        lambda meta, corpus: [
            {
                "id": "MOR-A99",
                "fait": "F09",
                "piece": piece,
                "quote": "Nous notifions votre licenciement pour cause réelle et sérieuse.",
                "ligne": 2,
                "nature": "qualification",
                "verbatim_normalise": True,
            }
        ],
    )
    monkeypatch.setattr(
        moreau_or,
        "build_questions",
        lambda meta, assertions: [
            {
                "id": "MOR-Q99",
                "shape": "qualification",
                "question": "Motif ?",
                "answer": "Licenciement pour cause réelle et sérieuse.",
                "supporting_assertions": ["MOR-A99"],
                "must_mention": ["cause réelle et sérieuse"],
                "must_not_claim": [],
                "difficulty": "easy",
                "why_it_matters": "test",
            }
        ],
    )
    summary = construire(
        golden_path=golden,
        docs_dir=docs,
        gold_dir=gold_dir,
        extraction_report=None,
    )
    assert summary["questions"] == 1
    assert (gold_dir / "assertions.json").exists()


def test_construire_writes_only_to_requested_gold_dir(tmp_path: Path, monkeypatch):
    from banc_realite import moreau_or

    docs = tmp_path / "documents"
    docs.mkdir()
    piece = "piece.md"
    (docs / piece).write_text(
        "Document du 12 janvier 2024 avec un montant de 1 720,00 euros.\n" + ("x" * 250),
        encoding="utf-8",
    )
    golden = tmp_path / "GOLDEN-faits.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "facts": [
                    {
                        "id": "F01",
                        "resume": "Fait du 12 janvier 2024 pour 1 720 €",
                        "pieces": ["piece.pdf"],
                        "montant_eur": 1720,
                        "date": "2024-01-12",
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    gold_dir = tmp_path / "gold"
    default_gold = tmp_path / "default-gold"
    default_gold.mkdir()
    for name in ("assertions.json", "questions.json", "OR-JURIDIQUE.md", "couverture.json", "RAPPORT-OR.md"):
        (default_gold / name).write_text("sentinel\n", encoding="utf-8")

    monkeypatch.setattr(moreau_or, "GOLDEN_PATH", golden)
    monkeypatch.setattr(moreau_or, "DOCS_DIR", docs)
    monkeypatch.setattr(moreau_or, "GOLD_DIR", default_gold)
    monkeypatch.setattr(moreau_or, "build_questions", lambda meta, assertions: [])
    construire(
        golden_path=golden,
        docs_dir=docs,
        gold_dir=gold_dir,
        extraction_report=None,
    )
    assert (gold_dir / "assertions.json").exists()
    assert (gold_dir / "questions.json").exists()
    for name in ("assertions.json", "questions.json", "OR-JURIDIQUE.md", "couverture.json", "RAPPORT-OR.md"):
        assert (default_gold / name).read_text(encoding="utf-8") == "sentinel\n"


def test_full_or_has_complete_fact_coverage():
    from banc_realite.moreau_extraction import GOLDEN_PATH, load_corpus

    meta = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))
    corpus = load_corpus()
    assertions = build_assertions(meta, corpus)
    report = verify_facts_coverage(meta, assertions)
    assert report["ok"]
    assert report["covered"] == report["total"] == 15
