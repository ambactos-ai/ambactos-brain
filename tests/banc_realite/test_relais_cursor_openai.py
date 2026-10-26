# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — relais OpenAI-compatible vers cursor-agent."""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from banc_realite import cursor_chat as cc
from banc_realite import relais_cursor_openai as relay


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _success_payload(result: str) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": result,
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
    )


@pytest.fixture
def relay_server(tmp_path):
    journal = tmp_path / "appels-relais-cursor.jsonl"
    port = _free_port()
    workdir = tmp_path / "vide"
    workdir.mkdir()
    responses: list[str] = ["plain answer"]
    active = {"count": 0, "max": 0}
    lock = threading.Lock()

    def chat_fn(system: str, user: str) -> tuple[str, dict]:
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            time.sleep(0.05)
            idx = min(len(responses) - 1, 0)
            text = responses[0] if len(responses) == 1 else responses.pop(0)
            active["count"] -= 1
        return text, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    server = relay.serve(port=port, model="cursor-grok-4.5-high", chat_fn=chat_fn, journal_path=journal)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield {"port": port, "journal": journal, "responses": responses, "active": active, "server": server}
    server.shutdown()
    server.server_close()


def _post_json(port: int, path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(port: int, path: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
        return response.status, response.read()


def test_healthz_and_models(relay_server):
    port = relay_server["port"]
    status, body = _get(port, "/healthz")
    assert status == 200
    assert body == b"ok\n"
    status, body = _get(port, "/v1/models")
    assert status == 200
    payload = json.loads(body.decode("utf-8"))
    assert payload["data"][0]["id"] == "cursor-grok-4.5-high"


def test_simple_text_completion(relay_server):
    port = relay_server["port"]
    payload = _post_json(
        port,
        "/v1/chat/completions",
        {
            "model": "cursor-grok-4.5-high",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "plain answer"
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_json_object_strips_fences(tmp_path):
    journal = tmp_path / "journal.jsonl"
    port = _free_port()
    workdir = tmp_path / "vide"
    workdir.mkdir()

    def run_fn(_argv, *, cwd, env, timeout):
        return 0, _success_payload('```json\n{"memory": []}\n```'), ""

    chat_fn = cc.make_cursor_chat_fn(workdir=workdir, run_fn=run_fn)
    server = relay.serve(port=port, model="cursor-grok-4.5-high", chat_fn=chat_fn, journal_path=journal)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = _post_json(
            port,
            "/v1/chat/completions",
            {
                "model": "cursor-grok-4.5-high",
                "messages": [{"role": "user", "content": "extract"}],
                "response_format": {"type": "json_object"},
            },
        )
        content = payload["choices"][0]["message"]["content"]
        assert json.loads(content) == {"memory": []}
        row = json.loads(journal.read_text(encoding="utf-8").strip())
        assert row["json_demande"] is True
        assert row["json_ok"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_json_missing_returns_502(tmp_path):
    journal = tmp_path / "journal.jsonl"
    port = _free_port()
    workdir = tmp_path / "vide"
    workdir.mkdir()

    def run_fn(_argv, *, cwd, env, timeout):
        return 0, _success_payload("pas du json"), ""

    chat_fn = cc.make_cursor_chat_fn(workdir=workdir, run_fn=run_fn)
    server = relay.serve(port=port, model="cursor-grok-4.5-high", chat_fn=chat_fn, journal_path=journal)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps(
            {
                "model": "cursor-grok-4.5-high",
                "messages": [{"role": "user", "content": "extract"}],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=10)
        assert exc_info.value.code == 502
        payload = json.loads(exc_info.value.read().decode("utf-8"))
        assert payload["error"]["type"] == "relay_error"
        row = json.loads(journal.read_text(encoding="utf-8").strip())
        assert row["json_ok"] is False
        assert row["statut"] == "502"
    finally:
        server.shutdown()
        server.server_close()


def test_concurrent_requests_are_serialized(tmp_path):
    journal = tmp_path / "journal.jsonl"
    port = _free_port()
    active = {"count": 0, "max": 0}
    lock = threading.Lock()

    def chat_fn(_system: str, _user: str) -> tuple[str, dict]:
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            time.sleep(0.1)
            active["count"] -= 1
        return "ok", {}

    server = relay.serve(port=port, model="cursor-grok-4.5-high", chat_fn=chat_fn, journal_path=journal)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[Exception] = []

    def worker():
        try:
            _post_json(
                port,
                "/v1/chat/completions",
                {"model": "cursor-grok-4.5-high", "messages": [{"role": "user", "content": "x"}]},
            )
        except Exception as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    server.shutdown()
    server.server_close()
    assert not errors
    assert active["max"] == 1
    assert len(journal.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_extract_json_helpers():
    assert relay._extract_json_text('{"a": 1}') == ('{"a": 1}', True)
    assert relay._extract_json_text("```json\n{\"b\":2}\n```")[1] is True
    assert relay._extract_json_text("aucun json") == ("{}", False)
