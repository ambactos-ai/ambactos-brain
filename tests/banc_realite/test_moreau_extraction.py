# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne pour l'extraction MOREAU (sans PDF, sans réseau)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from banc_realite.moreau_extraction import (
    build_inventory,
    cmd_verifier,
    derive_discriminant_tokens,
    extract_eml_text,
    normalize_extraction_text,
    search_token,
    token_in_text,
    verify_facts,
    verify_manifest,
    write_extraction_report,
)


def _sha256_prefix(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def test_normalize_extraction_text():
    raw = "mot-\nsuite\u00a0fin\n\n\n\nligne"
    assert normalize_extraction_text(raw) == "motsuite fin\n\nligne\n"


def test_extract_eml_headers_and_body(tmp_path: Path):
    eml = tmp_path / "mail.eml"
    eml.write_text(
        "From: alice@example.com\n"
        "To: bob@example.com\n"
        "Subject: Test MOREAU\n"
        "Date: Mon, 8 Jan 2024 10:00:00 +0100\n"
        "Content-Type: text/plain; charset=utf-8\n"
        "\n"
        "Corps du message avec 86 heures.\n",
        encoding="utf-8",
    )
    text = extract_eml_text(eml)
    assert "From: alice@example.com" in text
    assert "To: bob@example.com" in text
    assert "Subject: Test MOREAU" in text
    assert "Date: Mon," in text and "Jan 2024" in text
    assert "86 heures" in text


def test_search_token_found_and_missing():
    corpus = {
        "01-alpha.md": "Claire MOREAU embauchée le 15 septembre 2019",
        "02-beta.md": "autre contenu",
    }
    found = search_token("2019", corpus, ["01-alpha.pdf"])
    assert found["status"] == "verifie"
    assert found["pieces"] == ["01-alpha.md"]

    elsewhere = search_token("autre", corpus, ["01-alpha.pdf"])
    assert elsewhere["status"] == "hors-piece"
    assert elsewhere["pieces"] == ["02-beta.md"]

    missing = search_token("introuvable", corpus, ["01-alpha.pdf"])
    assert missing["status"] == "absent"


def test_rg_reference_is_not_numeric_substring():
    text = "R.G. n° 24/01984"
    assert token_in_text("24/01984", text)
    assert not token_in_text("2401984", text)
    assert not token_in_text("86", "Incident survenu en 1986.")
    assert not token_in_text("3", "Convocation du 2 janvier 2023.")
    assert token_in_text("86", "TOTAL heures supplémentaires déclarées : 86 heures")
    assert token_in_text("1720", "Montant brut revendiqué : 1 720,00 euros")
    assert token_in_text("2450", "salaire brut 2 450,00 €")


def test_hors_piece_prevents_fact_verification(tmp_path: Path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "01-alpha.md").write_text("Claire MOREAU CDI 15 septembre 2019\n", encoding="utf-8")
    (docs / "02-beta.md").write_text("SIREN 749 382 615\n", encoding="utf-8")
    golden = tmp_path / "GOLDEN-faits.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "facts": [
                    {
                        "id": "F02",
                        "resume": "Employeur SIREN 749 382 615",
                        "pieces": ["01-alpha.pdf"],
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = verify_facts(golden_path=golden, docs_dir=docs)
    fact = report["facts"][0]
    assert report["facts_verified"] == 0
    assert not fact["ok"]
    assert any(t["status"] == "hors-piece" for t in fact["tokens"])


def test_verify_facts_report_with_fixture(tmp_path: Path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "01-alpha.md").write_text(
        "Claire MOREAU CDI 15 septembre 2019\nSIREN 749 382 615\n",
        encoding="utf-8",
    )
    (docs / "02-beta.md").write_text("Avenant 1 mars 2022 salaire 2 450 €\n", encoding="utf-8")
    golden = tmp_path / "GOLDEN-faits.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "rg": "24/01984",
                "facts": [
                    {
                        "id": "F01",
                        "resume": "Claire MOREAU embauchée en CDI le 15 septembre 2019",
                        "pieces": ["01-alpha.pdf"],
                        "date": "2019-09-15",
                    },
                    {
                        "id": "F02",
                        "resume": "Employeur SIREN 749 382 615",
                        "pieces": ["02-beta.pdf"],
                    },
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = verify_facts(golden_path=golden, docs_dir=docs)
    assert report["facts_total"] == 2
    assert report["facts_verified"] == 1
    f02_tokens = next(f for f in report["facts"] if f["id"] == "F02")["tokens"]
    assert any(t["status"] in {"absent", "hors-piece"} for t in f02_tokens)
    assert not next(f for f in report["facts"] if f["id"] == "F02")["ok"]
    out_md = tmp_path / "EXTRACTION.md"
    out_json = tmp_path / "extraction.json"
    import banc_realite.moreau_extraction as mod

    orig_md, orig_json = mod.EXTRACTION_MD, mod.EXTRACTION_JSON
    mod.EXTRACTION_MD, mod.EXTRACTION_JSON = out_md, out_json
    write_extraction_report(report)
    mod.EXTRACTION_MD, mod.EXTRACTION_JSON = orig_md, orig_json
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["facts_verified"] == 1


def test_derive_discriminant_tokens():
    fact = {
        "id": "F11",
        "resume": "Propos du 8 novembre 2023 « la porte est large »",
        "date": "2023-11-08",
        "pieces": ["09-attestation-collegue.pdf"],
    }
    tokens = derive_discriminant_tokens(fact, {"rg": "24/01984"})
    assert "la porte est large" in tokens
    assert "2023-11-08" in tokens


def test_manifest_malformed_line_fails(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "MANIFEST.sha256"
    manifest.write_text("deadbeef\n", encoding="utf-8")
    report = verify_manifest(manifest_path=manifest, source_dir=source)
    assert not report["ok"]
    assert report["results"][0]["status"] == "malformed"


def _write_piece(source: Path, name: str, content: bytes) -> Path:
    pieces = source / "pieces"
    pieces.mkdir(parents=True, exist_ok=True)
    path = pieces / name
    path.write_bytes(content)
    return path


def test_manifest_partial_coverage_fails(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    piece_a = _write_piece(source, "01-alpha.pdf", b"alpha content")
    _write_piece(source, "02-beta.pdf", b"beta content")
    manifest = source / "MANIFEST.sha256"
    manifest.write_text(f"{_sha256_prefix(piece_a)}  pieces/01-alpha.pdf\n", encoding="utf-8")
    report = verify_manifest(manifest_path=manifest, source_dir=source)
    assert not report["ok"]
    non_covered = [row for row in report["results"] if row["status"] == "non-couverte"]
    assert len(non_covered) == 1
    assert non_covered[0]["path"] == "pieces/02-beta.pdf"


def test_manifest_full_coverage_ok(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    piece_a = _write_piece(source, "01-alpha.pdf", b"alpha content")
    piece_b = _write_piece(source, "02-beta.eml", b"beta content")
    manifest = source / "MANIFEST.sha256"
    manifest.write_text(
        "\n".join(
            [
                f"{_sha256_prefix(piece_a)}  pieces/01-alpha.pdf",
                f"{_sha256_prefix(piece_b)}  pieces/02-beta.eml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = verify_manifest(manifest_path=manifest, source_dir=source)
    assert report["ok"]
    assert all(row["status"] == "ok" for row in report["results"])


def test_cmd_verifier_exit_code_and_report(tmp_path: Path, monkeypatch, capsys):
    import banc_realite.moreau_extraction as mod

    out_md = tmp_path / "EXTRACTION.md"
    out_json = tmp_path / "extraction.json"
    monkeypatch.setattr(mod, "EXTRACTION_MD", out_md)
    monkeypatch.setattr(mod, "EXTRACTION_JSON", out_json)

    calls = {"n": 0}

    def fake_verify_facts():
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "affaire": None,
                "rg": None,
                "facts_total": 2,
                "facts_verified": 1,
                "facts": [
                    {"id": "F01", "ok": True, "tokens": []},
                    {
                        "id": "F99",
                        "ok": False,
                        "tokens": [{"token": "introuvable", "status": "absent", "pieces": []}],
                    },
                ],
            }
        return {
            "affaire": None,
            "rg": None,
            "facts_total": 2,
            "facts_verified": 2,
            "facts": [
                {"id": "F01", "ok": True, "tokens": []},
                {"id": "F99", "ok": True, "tokens": []},
            ],
        }

    monkeypatch.setattr(mod, "verify_facts", fake_verify_facts)

    code_fail = cmd_verifier(argparse.Namespace())
    captured_fail = capsys.readouterr()
    assert code_fail == 1
    assert out_md.exists()
    assert out_json.exists()
    assert "F99" in captured_fail.out
    payload_fail = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload_fail["facts_verified"] == 1
    assert payload_fail["facts_total"] == 2

    code_ok = cmd_verifier(argparse.Namespace())
    captured_ok = capsys.readouterr()
    assert code_ok == 0
    assert "2/2" in captured_ok.out
    payload_ok = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload_ok["facts_verified"] == 2


def test_build_inventory(tmp_path: Path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "01-test.md").write_text("Document du 12 janvier 2024\n" + ("x" * 250), encoding="utf-8")
    inv = build_inventory(docs_dir=docs, out_path=tmp_path / "INVENTAIRE.md")
    assert inv["pieces"] == 1
    assert inv["caracteres"] > 200
    content = (tmp_path / "INVENTAIRE.md").read_text(encoding="utf-8")
    assert "01-test.md" in content
    assert "Total :" in content
