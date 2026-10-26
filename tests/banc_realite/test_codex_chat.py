# SPDX-License-Identifier: AGPL-3.0-only
"""Tests codex exec — fabrique make_codex_chat_fn et branchement juge."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from banc_realite import codex_chat as cc
from banc_realite import etage3 as e3
from banc_realite import etage3_comparatif as e3c
def _jsonl_stdout(text: str, *, input_tokens: int = 100, output_tokens: int = 5) -> str:
    return "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "sess-codex-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "item_0", "type": "agent_message", "text": text},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_input_tokens": 0,
                    },
                }
            ),
        ]
    )


def test_commande_construite_avec_options_attendues(tmp_path):
    captured: dict = {}

    def run_fn(argv, *, cwd, env, timeout):
        captured["argv"] = argv
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout"] = timeout
        return 0, _jsonl_stdout("OK"), ""

    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = cc.make_codex_chat_fn(model="gpt-5.6-sol", workdir=workdir, run_fn=run_fn)
    text, meta = chat("system prompt", "user prompt")

    assert text == "OK"
    assert meta["fournisseur"] == "codex-cli"
    assert meta["modele_demande"] == "gpt-5.6-sol"
    assert meta["usage"]["prompt_tokens"] == 100
    assert meta["usage"]["completion_tokens"] == 5
    assert captured["cwd"] == workdir
    assert captured["env"] == cc.CODEX_ENV
    argv = captured["argv"]
    assert argv[1] == "exec"
    assert "--skip-git-repo-check" in argv
    assert "--sandbox" in argv
    assert "read-only" in argv
    assert "-C" in argv
    assert str(workdir) in argv
    assert "--model" in argv
    assert "gpt-5.6-sol" in argv
    assert "--json" in argv
    assert argv[-1] == "system prompt\n\nuser prompt"


def test_refuse_slug_grok():
    with pytest.raises(RuntimeError, match="grok"):
        cc.assert_codex_model_slug_allowed("grok-4.7-high")


def test_refuse_slug_anthropic():
    with pytest.raises(RuntimeError, match="anthropic"):
        cc.assert_codex_model_slug_allowed("claude-sonnet-4")


def test_make_codex_chat_fn_refuse_grok(tmp_path):
    workdir = tmp_path / "vide"
    workdir.mkdir()
    with pytest.raises(RuntimeError, match="grok"):
        cc.make_codex_chat_fn(model="cursor-grok-4.5-high", workdir=workdir, run_fn=lambda *a, **k: (0, "", ""))


def test_extraction_json_sans_reparation():
    noisy = 'codex\n{"jugements":[],"refus":false}\ntokens used\n42'
    extracted = cc.extract_json_text(noisy)
    assert json.loads(extracted) == {"jugements": [], "refus": False}


def test_json_invalide_leve_sans_reparation():
    with pytest.raises(ValueError, match="JSON invalide"):
        cc.extract_json_text("pas du json du tout")


def test_juge_journalise_fournisseur_codex_cli(tmp_path):
    calls = tmp_path / "calls.jsonl"
    payload = {
        "jugements": [{"numero": 1, "verdict": "n_affirme_pas", "citation": ""}],
        "refus": False,
    }

    def run_fn(_argv, *, cwd, env, timeout):
        return 0, _jsonl_stdout(json.dumps(payload)), ""

    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = cc.make_codex_chat_fn(workdir=workdir, run_fn=run_fn)
    question = {
        "id": "MOR-Q01",
        "question": "Quelle est la date ?",
        "must_not_claim": ["Benali confirme."],
    }
    e3.adjudicate_judge(
        "Le dossier ne permet pas de répondre.",
        question,
        invite_key="juge-rag_naif-MOR-Q01",
        chat_fn=chat,
        calls_path=calls,
        modele="gpt-5.6-sol",
    )
    record = json.loads(calls.read_text(encoding="utf-8").splitlines()[-1])
    assert record["fournisseur"] == "codex-cli"
    assert record["modele_demande"] == "gpt-5.6-sol"
    assert record["final"] is True
    assert record["illisible"] is False


def test_json_invalide_journalise_illisible(tmp_path):
    calls = tmp_path / "calls.jsonl"

    def run_fn(_argv, *, cwd, env, timeout):
        return 0, _jsonl_stdout("pas du json"), ""

    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = cc.make_codex_chat_fn(workdir=workdir, run_fn=run_fn)
    question = {
        "id": "MOR-Q01",
        "question": "Quelle est la date ?",
        "must_not_claim": ["Benali confirme."],
    }
    result = e3.adjudicate_judge(
        "réponse",
        question,
        invite_key="juge-A-MOR-Q01",
        chat_fn=chat,
        calls_path=calls,
        modele="gpt-5.6-sol",
    )
    assert result["illisible"] is True
    lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["final"] is False
    assert json.loads(lines[-1])["fournisseur"] == "codex-cli"


def test_make_juge_chat_fn_codex(tmp_path, monkeypatch):
    monkeypatch.setattr(
        e3c,
        "make_codex_chat_fn",
        lambda **kwargs: (lambda s, u: ("ok", {"fournisseur": "codex-cli"})),
    )
    fn = e3c.make_juge_chat_fn("gpt-5.6-sol", fournisseur="codex")
    text, meta = fn("s", "u")
    assert text == "ok"
    assert meta["fournisseur"] == "codex-cli"


def test_factory_raises_under_pytest_without_run_fn():
    os.environ["PYTEST_CURRENT_TEST"] = "tests/banc_realite/test_codex_chat.py::test_factory"
    try:
        with pytest.raises(RuntimeError, match="interdit sous pytest"):
            cc.make_codex_chat_fn()
    finally:
        del os.environ["PYTEST_CURRENT_TEST"]
