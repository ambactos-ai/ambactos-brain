# SPDX-License-Identifier: AGPL-3.0-only
"""Étage 3 — appels modèle via codex exec (subscription ChatGPT)."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .etage3 import ChatFn

CODEX_BINARY = "codex"
CODEX_WORKDIR = BANC_VAR / 'codex-vide'
CODEX_ENV = {
    "HOME": str(Path.home()),
    "PATH": str(Path.home() / ".local" / "bin") + ':/usr/local/bin:/usr/bin:/bin',
}

QUOTA_MARKERS = (
    "rate limit",
    "rate_limit",
    "quota",
    "usage limit",
    "too many requests",
    "429",
)

MODEL_LINE_RE = re.compile(r"^model:\s*(.+)$", re.MULTILINE)
TOKENS_USED_RE = re.compile(r"tokens used\s*\n\s*([\d,]+)", re.IGNORECASE)

RunFn = Callable[..., tuple[int, str, str]]


class CodexQuotaAtteint(RuntimeError):
    """Quota ou limite de débit atteinte sur le compte Codex."""


def _default_run_fn(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"délai de {timeout} s dépassé"
    return result.returncode, result.stdout, result.stderr


def _ensure_empty_workdir(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    if any(workdir.iterdir()):
        raise RuntimeError(f"dossier de travail codex non vide: {workdir}")


def assert_codex_model_slug_allowed(model: str) -> None:
    lowered = model.lower()
    if "grok" in lowered:
        raise RuntimeError(f"modèle juge interdit (grok): {model}")
    if "anthropic" in lowered or "claude" in lowered:
        raise RuntimeError(f"modèle juge interdit (anthropic): {model}")


def _raise_if_quota(stdout: str, stderr: str, returncode: int) -> None:
    blob = f"{stdout}\n{stderr}".lower()
    if returncode != 0 and any(marker in blob for marker in QUOTA_MARKERS):
        raise CodexQuotaAtteint(f"quota codex-cli atteint: {stderr[:300] or stdout[:300]}")
    if returncode == 0:
        return
    if any(marker in blob for marker in QUOTA_MARKERS):
        raise CodexQuotaAtteint(f"quota codex-cli atteint: {stderr[:300] or stdout[:300]}")


def _first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_text(raw: str) -> str:
    """Extrait un objet JSON sans le réparer ; lève si aucun JSON valide."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    for candidate in (text, _first_balanced_json_object(text) or ""):
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    raise ValueError("JSON invalide ou absent")


def _parse_codex_jsonl(stdout: str) -> tuple[str, dict[str, Any]]:
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                text = str(item.get("text") or "")
                if text:
                    content_parts.append(text)
        elif event_type == "turn.completed":
            usage = event.get("usage") or {}
    if not content_parts:
        raise ValueError("aucun agent_message dans la sortie codex --json")
    return "".join(content_parts), usage


def _parse_model_from_output(stdout: str, stderr: str) -> str | None:
    for blob in (stdout, stderr):
        match = MODEL_LINE_RE.search(blob)
        if match:
            return match.group(1).strip()
    return None


def _parse_tokens_from_text(stdout: str, stderr: str) -> tuple[int | None, int | None]:
    for blob in (stdout, stderr):
        match = TOKENS_USED_RE.search(blob)
        if match:
            total = int(match.group(1).replace(",", ""))
            return None, total
    return None, None


def _meta_from_usage(
    *,
    model_demande: str,
    model_repondu: str | None,
    usage: dict[str, Any],
    session_id: str | None = None,
) -> dict[str, Any]:
    prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("output_tokens")
    cached = usage.get("cached_input_tokens") or 0
    meta_usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if cached:
        meta_usage["cache_read_tokens"] = cached
    total_tokens = None
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total_tokens = prompt_tokens + completion_tokens
    if total_tokens is not None:
        meta_usage["total_tokens"] = total_tokens
    return {
        "modele_demande": model_demande,
        "model_returned": model_repondu or model_demande,
        "fournisseur": "codex-cli",
        "finish_reason": None,
        "usage": meta_usage,
        "session_id": session_id,
    }


def make_codex_chat_fn(
    *,
    model: str = "gpt-5.6-sol",
    workdir: Path = CODEX_WORKDIR,
    timeout_s: int = 600,
    run_fn: RunFn | None = None,
) -> ChatFn:
    assert_codex_model_slug_allowed(model)
    if run_fn is None and os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("appel codex interdit sous pytest")

    binary = shutil.which(CODEX_BINARY)
    if binary is None:
        raise RuntimeError(f"binaire codex introuvable: {CODEX_BINARY}")

    _ensure_empty_workdir(workdir)
    runner = run_fn or _default_run_fn

    def _build_argv(prompt: str) -> list[str]:
        return [
            binary,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-C",
            str(workdir),
            "--model",
            model,
            "--json",
            prompt,
        ]

    def _invoke(system: str, user: str) -> tuple[int, str, str]:
        prompt = system + "\n\n" + user
        return runner(
            _build_argv(prompt),
            cwd=workdir,
            env=CODEX_ENV,
            timeout=timeout_s,
        )

    def _attempt(system: str, user: str) -> tuple[str, dict[str, Any]]:
        returncode, stdout, stderr = _invoke(system, user)
        _raise_if_quota(stdout, stderr, returncode)
        if returncode != 0:
            raise RuntimeError(f"codex code {returncode}: {stderr[:300] or stdout[:300]}")
        model_repondu = _parse_model_from_output(stdout, stderr)
        if model_repondu and model_repondu != model:
            raise RuntimeError(
                f"modèle codex inattendu: demandé {model}, obtenu {model_repondu}"
            )
        try:
            text, usage = _parse_codex_jsonl(stdout)
        except ValueError:
            text = stdout.strip() or stderr.strip()
            if not text:
                raise RuntimeError("sortie codex vide") from None
            prompt_fallback, total_fallback = _parse_tokens_from_text(stdout, stderr)
            usage = {
                "input_tokens": prompt_fallback,
                "output_tokens": total_fallback,
            }
        if text.lstrip().startswith("{") or "{" in text:
            try:
                text = extract_json_text(text)
            except ValueError:
                pass
        session_id = None
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started":
                session_id = event.get("thread_id")
                break
        return text, _meta_from_usage(
            model_demande=model,
            model_repondu=model_repondu or model,
            usage=usage,
            session_id=session_id,
        )

    def chat(system: str, user: str) -> tuple[str, dict[str, Any]]:
        try:
            return _attempt(system, user)
        except CodexQuotaAtteint:
            raise
        except RuntimeError:
            time.sleep(5)
            try:
                return _attempt(system, user)
            except RuntimeError as exc:
                raise RuntimeError(str(exc)) from exc

    return chat
