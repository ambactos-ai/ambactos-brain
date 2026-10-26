# SPDX-License-Identifier: AGPL-3.0-only
"""Tests cursor-agent — fabrique make_cursor_chat_fn et choisir_chat_fn."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from banc_realite import cursor_chat as cc
from banc_realite import etage3_config_gagnante as ecg
from banc_realite.plafond_appels import CompteurAppels, PlafondAtteint


def _success_payload(result: str = "answer text") -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": result,
            "session_id": "sess-1",
            "request_id": "req-1",
            "duration_api_ms": 5912,
            "usage": {
                "inputTokens": 21426,
                "outputTokens": 128,
                "cacheReadTokens": 512,
                "cacheWriteTokens": 0,
            },
        }
    )


def test_success_returns_text_and_meta(tmp_path):
    captured: dict = {}

    def run_fn(argv, *, cwd, env, timeout):
        captured["argv"] = argv
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout"] = timeout
        return 0, _success_payload("**30 days after issue**"), ""

    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = cc.make_cursor_chat_fn(model="cursor-grok-4.5-high", workdir=workdir, run_fn=run_fn)
    system = "You are an agent."
    user = "When is payment due?"
    text, meta = chat(system, user)

    assert text == "**30 days after issue**"
    assert meta["fournisseur"] == "cursor-agent"
    assert meta["finish_reason"] is None
    assert meta["usage"]["prompt_tokens"] == 21426
    assert meta["usage"]["completion_tokens"] == 128
    assert meta["usage"]["cache_read_tokens"] == 512
    assert meta["usage"]["cache_write_tokens"] == 0
    assert captured["cwd"] == workdir
    assert captured["env"] == cc.CURSOR_ENV
    argv = captured["argv"]
    assert argv[:2] == [str(cc.CURSOR_AGENT), "--model"]
    assert "cursor-grok-4.5-high" in argv
    assert "--mode" in argv
    assert "ask" in argv
    assert "--output-format" in argv
    assert "json" in argv
    assert argv[-1] == system + "\n\n" + user


def test_is_error_then_success(monkeypatch, tmp_path):
    calls = 0

    def run_fn(_argv, *, cwd, env, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0, json.dumps({"type": "result", "subtype": "success", "is_error": True}), ""
        return 0, _success_payload("recovered"), ""

    sleeps: list[float] = []
    monkeypatch.setattr(cc.time, "sleep", lambda s: sleeps.append(s))

    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = cc.make_cursor_chat_fn(workdir=workdir, run_fn=run_fn)
    text, meta = chat("sys", "usr")

    assert calls == 2
    assert sleeps == [5.0]
    assert text == "recovered"
    assert meta["fournisseur"] == "cursor-agent"


def test_retry_plafond_atteint_non_enveloppe(monkeypatch, tmp_path):
    compteur = CompteurAppels(
        compteur_path=tmp_path / "compteur-appels.json",
        journal_path=tmp_path / "appels.jsonl",
        plafond=1,
    )
    appels = {"n": 0}

    def run_fn(_argv, *, cwd, env, timeout):
        appels["n"] += 1
        return 1, "", "fail"

    monkeypatch.setattr(cc.time, "sleep", lambda _s: None)
    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = cc.make_cursor_chat_fn(
        workdir=workdir,
        run_fn=compteur.envelopper_run_fn(
            run_fn,
            dossier="d",
            piece="p.md",
            morceau=None,
            nature="reconstitution-entite",
        ),
    )
    with pytest.raises(PlafondAtteint):
        chat("sys", "usr")
    assert appels["n"] == 1
    assert compteur.lire()["emis"] == 1


def test_two_failures_raise_after_two_calls(monkeypatch, tmp_path):
    calls = 0

    def run_fn(_argv, *, cwd, env, timeout):
        nonlocal calls
        calls += 1
        return 0, json.dumps({"type": "result", "subtype": "success", "is_error": True}), "err body"

    monkeypatch.setattr(cc.time, "sleep", lambda _s: None)

    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = cc.make_cursor_chat_fn(workdir=workdir, run_fn=run_fn)

    with pytest.raises(RuntimeError):
        chat("sys", "usr")
    assert calls == 2


def test_invalid_json_raises(tmp_path):
    def run_fn(_argv, *, cwd, env, timeout):
        return 0, "not json", ""

    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = cc.make_cursor_chat_fn(workdir=workdir, run_fn=run_fn)

    with pytest.raises(RuntimeError, match="JSON illisible"):
        chat("sys", "usr")


def test_nonempty_workdir_raises_at_factory(tmp_path):
    workdir = tmp_path / "sale"
    workdir.mkdir()
    (workdir / "fichier.txt").write_text("x", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non vide"):
        cc.make_cursor_chat_fn(workdir=workdir, run_fn=lambda *a, **k: (0, "", ""))


def test_factory_raises_under_pytest_without_run_fn():
    os.environ["PYTEST_CURRENT_TEST"] = "tests/banc_realite/test_cursor_chat.py::test_factory"
    try:
        with pytest.raises(RuntimeError, match="interdit sous pytest"):
            cc.make_cursor_chat_fn()
    finally:
        del os.environ["PYTEST_CURRENT_TEST"]


def test_choisir_chat_fn_relais():
    relais_called = False
    cursor_called = False

    def fabrique_relais():
        nonlocal relais_called
        relais_called = True
        return lambda s, u: ("r", {})

    def fabrique_cursor(modele):
        nonlocal cursor_called
        cursor_called = True
        return lambda s, u: ("c", {})

    fn = ecg.choisir_chat_fn("relais", "cursor-grok-4.5-high", fabrique_relais=fabrique_relais, fabrique_cursor=fabrique_cursor)
    assert relais_called
    assert not cursor_called
    assert fn("a", "b") == ("r", {})


def test_choisir_chat_fn_cursor():
    relais_called = False
    cursor_called = False
    seen_modele = None

    def fabrique_relais():
        nonlocal relais_called
        relais_called = True
        return lambda s, u: ("r", {})

    def fabrique_cursor(modele):
        nonlocal cursor_called, seen_modele
        cursor_called = True
        seen_modele = modele
        return lambda s, u: ("c", {})

    fn = ecg.choisir_chat_fn("cursor", "cursor-grok-4.5-high", fabrique_relais=fabrique_relais, fabrique_cursor=fabrique_cursor)
    assert cursor_called
    assert not relais_called
    assert seen_modele == "cursor-grok-4.5-high"
    assert fn("a", "b") == ("c", {})


def test_choisir_chat_fn_unknown_raises():
    with pytest.raises(ValueError, match="fournisseur inconnu"):
        ecg.choisir_chat_fn("autre", "x", fabrique_relais=lambda: None, fabrique_cursor=lambda m: None)


def _argv_proprietes(argv: list[str]) -> dict[str, list[str]]:
    props: dict[str, list[str]] = {}
    i = 0
    while i < len(argv):
        if argv[i] == "-p" and i + 1 < len(argv):
            raw = argv[i + 1]
            if "=" in raw:
                key, value = raw.split("=", 1)
                props.setdefault(key, []).append(value)
            i += 2
            continue
        i += 1
    return props


def test_argv_isole_masque_le_home_et_bind_lecture(tmp_path: Path) -> None:
    workdir = tmp_path / "invite"
    workdir.mkdir()
    inner = [str(cc.CURSOR_AGENT), "--model", "cursor-grok-4.5-high", "PROMPT"]
    argv = cc._construire_argv_isole(
        inner, workdir=workdir, timeout=42, env=cc.CURSOR_ENV
    )
    assert argv[:6] == [
        "systemd-run",
        "--user",
        "--wait",
        "--collect",
        "--pipe",
        "--quiet",
    ]
    props = _argv_proprietes(argv)
    assert props["TemporaryFileSystem"] == [cc.HOME_MASQUE]
    assert str(workdir) in props["BindReadOnlyPaths"]
    assert str(cc.CURSOR_AGENT) in props["BindReadOnlyPaths"]
    assert str(cc._repertoire_installation_cursor()) in props["BindReadOnlyPaths"]
    assert str(cc.CURSOR_AUTH_JSON) in props["BindReadOnlyPaths"]
    assert str(cc.CURSOR_AUTH_DIR) not in props["BindReadOnlyPaths"]
    assert props["PrivateTmp"] == ["yes"]
    assert props["NoNewPrivileges"] == ["yes"]
    assert props["ProtectSystem"] == ["strict"]
    assert props["MemoryMax"] == ["1G"]
    assert props["RuntimeMaxSec"] == ["42"]
    assert f"--working-directory={workdir}" in argv
    assert not any(arg.startswith("BindPaths=") for arg in argv)
    assert cc.HOME_MASQUE not in props.get("BindPaths", [])
    assert "--" in argv
    apres = argv[argv.index("--") + 1 :]
    assert apres[0] == str(cc._repertoire_installation_cursor() / cc.CURSOR_AGENT.name)
    assert apres[-1] == "PROMPT"


def test_run_fn_isole_commande_passee_a_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class _Resultat:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return _Resultat()

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    monkeypatch.setattr(cc, "ISOLE_WORKDIR_PARENT", tmp_path / "isole")
    code, stdout, stderr = cc._run_fn_isole(
        [str(cc.CURSOR_AGENT), "--trust", "bonjour"],
        cwd=tmp_path,
        env=cc.CURSOR_ENV,
        timeout=30,
    )
    assert code == 0
    assert stdout == "{}"
    assert stderr == ""
    argv = captured["argv"]
    props = _argv_proprietes(argv)
    assert props["TemporaryFileSystem"] == [cc.HOME_MASQUE]
    assert str(cc.CURSOR_AGENT) in props["BindReadOnlyPaths"]
    assert str(cc._repertoire_installation_cursor()) in props["BindReadOnlyPaths"]
    assert str(cc.CURSOR_AUTH_JSON) in props["BindReadOnlyPaths"]
    assert str(cc.CURSOR_AUTH_DIR) not in props["BindReadOnlyPaths"]
    assert any(
        Path(chemin).name == "invite.txt" or Path(chemin).name.startswith("invite-")
        for chemin in props["BindReadOnlyPaths"]
    )
    assert not any(arg.startswith("BindPaths=") for arg in argv)
    assert cc.HOME_MASQUE not in props.get("BindPaths", [])


def _cles_setenv(argv: list[str]) -> list[str]:
    cles: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--setenv" and i + 1 < len(argv):
            cles.append(argv[i + 1].split("=", 1)[0])
            i += 2
            continue
        if arg.startswith("--setenv="):
            cles.append(arg[len("--setenv=") :].split("=", 1)[0])
        if arg in {"-E", "--setenv"} and i + 1 < len(argv) and arg == "-E":
            cles.append(argv[i + 1].split("=", 1)[0])
            i += 2
            continue
        i += 1
    return cles


def test_run_fn_isole_env_minimal_path_home_lang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class _Resultat:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        return _Resultat()

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    monkeypatch.setattr(cc, "ISOLE_WORKDIR_PARENT", tmp_path / "isole")
    monkeypatch.setenv("LANG", "fr_FR.UTF-8")
    monkeypatch.setenv("CURSOR_FAKE_TOKEN", "x" * 32)
    cc._run_fn_isole(
        [str(cc.CURSOR_AGENT), "--trust", "bonjour"],
        cwd=tmp_path,
        env={**cc.CURSOR_ENV, "CURSOR_FAKE_TOKEN": "x" * 32},
        timeout=30,
    )
    cles = _cles_setenv(captured["argv"])
    assert sorted(cles) == ["HOME", "LANG", "PATH"]
    assert "CURSOR_FAKE_TOKEN" not in cles


def test_sortie_contenant_secret_auth_rejetee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "tok_test_contamination_cursor_abc123"
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"accessToken": secret, "refreshToken": "y" * 20}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cc, "CURSOR_AUTH_JSON", auth_path)
    monkeypatch.setattr(cc, "_secrets_env", lambda env=None: [])
    monkeypatch.setattr(cc.time, "sleep", lambda _s: None)

    def run_fn(_argv, *, cwd, env, timeout):
        return 0, _success_payload(f"reponse {secret} fin"), ""

    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = cc.make_cursor_chat_fn(workdir=workdir, run_fn=run_fn)
    with pytest.raises(RuntimeError, match="sortie contaminée par un secret"):
        chat("sys", "usr")


def test_sortie_secret_base64_ou_tronque_rejetee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import base64

    secret = "tok_test_contamination_cursor_abc123"
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"accessToken": secret}), encoding="utf-8")
    monkeypatch.setattr(cc, "CURSOR_AUTH_JSON", auth_path)
    monkeypatch.setattr(cc, "_secrets_env", lambda env=None: [])
    monkeypatch.setattr(cc.time, "sleep", lambda _s: None)

    def run_fn(_argv, *, cwd, env, timeout):
        return 0, _success_payload("fuite " + secret[:12]), ""

    workdir = tmp_path / "vide"
    workdir.mkdir()
    chat = cc.make_cursor_chat_fn(workdir=workdir, run_fn=run_fn)
    with pytest.raises(RuntimeError, match="sortie contaminée par un secret"):
        chat("sys", "usr")

    def run_fn_b64(_argv, *, cwd, env, timeout):
        encode = base64.b64encode(secret.encode("utf-8")).decode("ascii")
        return 0, _success_payload("fuite " + encode), ""

    chat_b64 = cc.make_cursor_chat_fn(workdir=workdir, run_fn=run_fn_b64)
    with pytest.raises(RuntimeError, match="sortie contaminée par un secret"):
        chat_b64("sys", "usr")


def test_run_fn_isole_refuse_si_systemd_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cc.shutil, "which", lambda _nom: None)
    with pytest.raises(RuntimeError, match="isolation indisponible"):
        cc._run_fn_isole(
            [str(cc.CURSOR_AGENT), "x"],
            cwd=tmp_path,
            env=cc.CURSOR_ENV,
            timeout=10,
        )


def test_run_fn_isole_refuse_si_demarrage_systemd_echoue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Resultat:
        returncode = 1
        stdout = ""
        stderr = "Failed to start transient service unit: Unit failed to load properly"

    monkeypatch.setattr(cc.subprocess, "run", lambda *a, **k: _Resultat())
    monkeypatch.setattr(cc, "ISOLE_WORKDIR_PARENT", tmp_path / "isole")
    with pytest.raises(RuntimeError, match="isolation indisponible"):
        cc._run_fn_isole(
            [str(cc.CURSOR_AGENT), "x"],
            cwd=tmp_path,
            env=cc.CURSOR_ENV,
            timeout=10,
        )
