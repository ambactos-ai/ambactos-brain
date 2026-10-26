# SPDX-License-Identifier: AGPL-3.0-only
"""Injected run_skill — Hermes relay first, grok_cli fallback; never claude_cli."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import json
import os
import socket
import time
import urllib.error
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

try:
    from brain.providers.grok_cli import GrokCliProvider
except ImportError:
    GrokCliProvider = None  # type: ignore[misc,assignment]
try:
    from brain.skill_runtime.loader import load_skill
except ImportError:
    load_skill = None  # type: ignore[misc,assignment]
try:
    from brain.skill_runtime.runner import _format_args
except ImportError:
    _format_args = None  # type: ignore[misc,assignment]

RELAY_URL = "http://127.0.0.1:8645/v1/chat/completions"
MODEL = "grok-4.5"


class TruncationError(RuntimeError):
    """Sortie tronquée (finish_reason=length) — signal banc uniquement."""


class ModelCallCapError(RuntimeError):
    """Plafond d'appels modèle atteint — arrêt nommé."""


class RealModelCallForbiddenError(RuntimeError):
    """Garde pytest — aucun appel réel au modèle depuis les tests."""
SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"
CALL_LOG = (BANC_VAR / 'sorties' / 'skill-calls.jsonl')
TIMEOUT_S = 300
MAX_TOKENS = 8000
MAX_RETRIES = 2
PRIOR_CALLS_REPORT = 37


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_call_count(log_path: Path) -> int:
    if not log_path.is_file():
        return 0
    report_n = 0
    max_n = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "report" in record:
            report_n = int(record.get("n") or 0)
        if record.get("event") == "end" and record.get("call_n") is not None:
            max_n = max(max_n, int(record["call_n"]))
    return max(max_n, report_n)


def ensure_call_log_header(
    log_path: Path,
    run_id: str,
    *,
    include_prior_report: bool = True,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.is_file() and log_path.stat().st_size > 0:
        return
    if not include_prior_report:
        if not log_path.exists():
            log_path.write_text("", encoding="utf-8")
        return
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "report": "appels antérieurs à la reprise",
                    "n": PRIOR_CALLS_REPORT,
                    "run_id": run_id,
                    "ts": _iso_now(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


class RelaySkillRunner:
    def __init__(
        self,
        *,
        log_path: Path | None = None,
        run_id: str | None = None,
        include_prior_report: bool = True,
        max_calls: int | None = None,
    ):
        self.run_id = run_id or str(uuid.uuid4())
        self.log_path = log_path or CALL_LOG
        self.include_prior_report = include_prior_report
        self.max_calls = max_calls
        ensure_call_log_header(
            self.log_path,
            self.run_id,
            include_prior_report=include_prior_report,
        )
        self.calls = load_call_count(self.log_path)
        self.providers: list[str] = []
        self._grok = GrokCliProvider()

    def _append_log(self, record: dict[str, Any]) -> None:
        record.setdefault("run_id", self.run_id)
        record.setdefault("ts", _iso_now())
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run_skill(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise RealModelCallForbiddenError(
                "appel modèle réel interdit sous pytest — mocker RelaySkillRunner.run_skill"
            )
        if name not in {"coarse-map", "atomise-entite"}:
            raise ValueError(f"Unexpected extraction skill: {name}")
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise ModelCallCapError(f"plafond appels modèle atteint ({self.max_calls})")
        skill = load_skill(SKILLS_ROOT, name)
        messages = [
            {"role": "system", "content": skill.body},
            {"role": "user", "content": _format_args(args)},
        ]
        started = time.monotonic()
        last_error: Exception | None = None
        content: str | None = None
        usage: dict[str, Any] = {}
        provider = "hermes_relay"
        finish_reason: str | None = None

        for attempt in range(MAX_RETRIES + 1):
            call_n = self.calls + 1
            attempt_started = time.monotonic()
            self._append_log(
                {
                    "event": "start",
                    "call_n": call_n,
                    "skill": name,
                    "attempt": attempt + 1,
                    "executor_provider": provider,
                }
            )
            try:
                if provider == "hermes_relay":
                    content, usage, finish_reason = self._relay_chat(messages)
                else:
                    response = self._grok.chat(
                        system=skill.body,
                        user_message=_format_args(args),
                        model=MODEL,
                    )
                    content = response.content
                    usage = {
                        "prompt_tokens": response.tokens_input,
                        "completion_tokens": response.tokens_output,
                    }
                    finish_reason = "stop"
                if finish_reason == "length":
                    raise TruncationError("truncated output (finish_reason=length)")
                self.calls = call_n
                self.providers.append(provider)
                self._append_log(
                    {
                        "event": "end",
                        "call_n": call_n,
                        "skill": name,
                        "attempt": attempt + 1,
                        "status": "ok",
                        "executor_provider": provider,
                        "executor_model": MODEL,
                        "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                        "finish_reason": finish_reason,
                        "usage": usage,
                    }
                )
                last_error = None
                break
            except Exception as exc:
                if isinstance(exc, TruncationError):
                    self.calls = call_n
                    self._append_log(
                        {
                            "event": "end",
                            "call_n": call_n,
                            "skill": name,
                            "attempt": attempt + 1,
                            "status": "erreur",
                            "executor_provider": provider,
                            "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                            "finish_reason": "length",
                            "erreur": str(exc),
                        }
                    )
                    raise
                if isinstance(exc, (TimeoutError, socket.timeout)):
                    self.calls = call_n
                    self._append_log(
                        {
                            "event": "end",
                            "call_n": call_n,
                            "skill": name,
                            "attempt": attempt + 1,
                            "status": "timeout",
                            "executor_provider": provider,
                            "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                            "erreur": str(exc),
                        }
                    )
                    last_error = exc
                elif isinstance(exc, urllib.error.HTTPError):
                    self.calls = call_n
                    if exc.code in {502, 503, 504} and attempt < MAX_RETRIES:
                        if provider == "hermes_relay" and attempt == MAX_RETRIES - 1:
                            provider = "grok_cli"
                        self._append_log(
                            {
                                "event": "end",
                                "call_n": call_n,
                                "skill": name,
                                "attempt": attempt + 1,
                                "status": "erreur",
                                "executor_provider": provider,
                                "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                                "http_status": exc.code,
                                "erreur": str(exc),
                            }
                        )
                        last_error = exc
                        continue
                    self._append_log(
                        {
                            "event": "end",
                            "call_n": call_n,
                            "skill": name,
                            "attempt": attempt + 1,
                            "status": "erreur",
                            "executor_provider": provider,
                            "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                            "http_status": exc.code,
                            "erreur": str(exc),
                        }
                    )
                    last_error = exc
                else:
                    self.calls = call_n
                    self._append_log(
                        {
                            "event": "end",
                            "call_n": call_n,
                            "skill": name,
                            "attempt": attempt + 1,
                            "status": "erreur",
                            "executor_provider": provider,
                            "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                            "finish_reason": finish_reason,
                            "usage": usage or None,
                            "erreur": str(exc),
                        }
                    )
                    last_error = exc
            if attempt >= MAX_RETRIES:
                break

        if last_error is not None or content is None:
            raise RuntimeError(f"{name} failed after {MAX_RETRIES + 1} attempts: {last_error}")

        latency_ms = int((time.monotonic() - started) * 1000)
        return {
            "result": content,
            "skill_name": name,
            "tokens_input": int((usage or {}).get("prompt_tokens") or 0),
            "tokens_output": int((usage or {}).get("completion_tokens") or 0),
            "latency_ms": latency_ms,
            "executor_provider": provider,
            "executor_model": MODEL,
            "model_returned": MODEL,
            "model_identity_verified": False,
            "cost_eur": None,
        }

    def _relay_chat(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any], str | None]:
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        }
        request = Request(
            RELAY_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=TIMEOUT_S) as response:
            body = json.load(response)
        choice = body["choices"][0]
        return (
            choice["message"]["content"],
            body.get("usage") or {},
            choice.get("finish_reason"),
        )

    def primary_provider(self) -> str:
        if not self.providers:
            return "hermes_relay"
        if all(p == "hermes_relay" for p in self.providers):
            return "hermes_relay"
        if all(p == "grok_cli" for p in self.providers):
            return "grok_cli"
        return "hermes_relay+grok_cli"

    def assert_no_claude(self) -> None:
        forbidden = {"claude_cli", "anthropic", "claude"}
        seen = {p.lower() for p in self.providers}
        if seen & forbidden:
            raise RuntimeError(f"Forbidden provider in call log: {seen & forbidden}")
