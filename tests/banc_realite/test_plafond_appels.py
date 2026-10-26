# SPDX-License-Identifier: AGPL-3.0-only
"""Tests du compteur d'appels cursor-agent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from banc_realite.cursor_chat import make_cursor_chat_fn
from banc_realite.plafond_appels import CompteurAppels, PlafondAtteint, make_compteur


def test_reserver_emission_incremente_avant_appel(tmp_path: Path) -> None:
    compteur = CompteurAppels(
        compteur_path=tmp_path / "compteur-appels.json",
        journal_path=tmp_path / "appels.jsonl",
        plafond=3,
    )
    assert compteur.reserver_emission() == 1
    assert compteur.lire()["emis"] == 1


def test_run_fn_qui_leve_laisse_emis_incremente(tmp_path: Path) -> None:
    compteur = CompteurAppels(
        compteur_path=tmp_path / "compteur-appels.json",
        journal_path=tmp_path / "appels.jsonl",
        plafond=3,
    )

    def failing(_argv, *, cwd, env, timeout):
        raise RuntimeError("boom")

    wrapped = compteur.envelopper_run_fn(
        failing, dossier="d", piece="p.md", morceau=1, nature="extraction"
    )
    with pytest.raises(RuntimeError, match="boom"):
        wrapped([], cwd=tmp_path, env={}, timeout=1)
    assert compteur.lire()["emis"] == 1


def test_cursor_chat_retry_compte_deux_appels(tmp_path: Path, monkeypatch) -> None:
    compteur = CompteurAppels(
        compteur_path=tmp_path / "compteur-appels.json",
        journal_path=tmp_path / "appels.jsonl",
        plafond=5,
    )
    calls = 0

    def run_fn(_argv, *, cwd, env, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 1, "", "fail"
        return 0, json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": '{"faits": []}',
            }
        ), ""

    monkeypatch.setattr("banc_realite.cursor_chat.time.sleep", lambda _s: None)
    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = make_cursor_chat_fn(
        workdir=workdir,
        run_fn=compteur.envelopper_run_fn(
            run_fn,
            dossier="d",
            piece="p.md",
            morceau=1,
            nature="extraction",
        ),
    )
    text, _meta = chat("", "invite")
    assert text == '{"faits": []}'
    assert compteur.lire()["emis"] == 2


def test_plafond_atteint_sans_sous_processus(tmp_path: Path) -> None:
    compteur = CompteurAppels(
        compteur_path=tmp_path / "compteur-appels.json",
        journal_path=tmp_path / "appels.jsonl",
        plafond=1,
    )
    compteur.reserver_emission()
    launched = False

    def run_fn(_argv, *, cwd, env, timeout):
        nonlocal launched
        launched = True
        return 0, "", ""

    wrapped = compteur.envelopper_run_fn(
        run_fn, dossier="d", piece="p.md", morceau=1, nature="extraction"
    )
    with pytest.raises(PlafondAtteint):
        wrapped([], cwd=tmp_path, env={}, timeout=1)
    assert launched is False


def test_plafond_explicite_journalise(tmp_path: Path) -> None:
    compteur = make_compteur(
        constructeur_root=tmp_path,
        plafond=42,
        plafond_explicite=True,
    )
    assert compteur.lire()["plafond"] == 42


def test_journal_appels_une_ligne_par_sous_processus(tmp_path: Path) -> None:
    compteur = CompteurAppels(
        compteur_path=tmp_path / "compteur-appels.json",
        journal_path=tmp_path / "appels.jsonl",
        plafond=5,
    )

    def ok(_argv, *, cwd, env, timeout):
        return 0, "", ""

    wrapped = compteur.envelopper_run_fn(
        ok, dossier="greenloop", piece="a.md", morceau=2, nature="reprise"
    )
    wrapped([], cwd=tmp_path, env={}, timeout=1)
    lines = compteur.journal_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["fournisseur"] == "cursor-agent"
    assert record["modele"] == "cursor-grok-4.5-high"
    assert record["nature"] == "reprise"
    assert record["emis"] == 1
    assert record["plafond"] == 5
