# SPDX-License-Identifier: AGPL-3.0-only
"""Relais OpenAI-compatible vers cursor-agent pour mem0 (issue #124)."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import argparse
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .config import BANC_ROOT
from .cursor_chat import make_cursor_chat_fn

COMPARATIF_DIR = BANC_ROOT / "sorties" / "comparatif"
JOURNAL_APPELS = COMPARATIF_DIR / "appels-relais-cursor.jsonl"
CURSOR_WORKDIR_DEFAULT = BANC_VAR / 'cursor-vide-relais'
JSON_SYSTEM_SUFFIX = (
    "Réponds uniquement par un objet JSON valide, sans texte avant ni après, "
    "sans balises markdown ni commentaires."
)

ChatFn = Callable[[str, str], tuple[str, dict[str, Any]]]


def _estimate_tokens(chars: int) -> int:
    return max(1, round(chars / 4))


def _journaliser(chemin: Path, row: dict[str, Any]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _concat_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system_parts: list[str] = []
    dialog_parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        prefix = {"user": "User", "assistant": "Assistant"}.get(role, role.capitalize())
        dialog_parts.append(f"{prefix}: {content}")
    return "\n\n".join(system_parts), "\n\n".join(dialog_parts)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def _first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
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


def _extract_json_text(raw: str) -> tuple[str, bool]:
    cleaned = _strip_code_fences(raw)
    for candidate in (cleaned, _first_balanced_json_object(cleaned) or ""):
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return candidate, True
        except json.JSONDecodeError:
            continue
    return "{}", False


def _tools_instruction(tools: list[dict[str, Any]], tool_choice: Any) -> str:
    payload = json.dumps({"tools": tools, "tool_choice": tool_choice}, ensure_ascii=False)
    return (
        "Le client attend une réponse structurée via tool_calls OpenAI. "
        f"Définitions : {payload}. "
        "Réponds par un objet JSON avec la clé tool_calls, chaque entrée ayant "
        'id, type="function", function.name et function.arguments (chaîne JSON).'
    )


def _parse_tool_calls(content: str) -> list[dict[str, Any]] | None:
    text, ok = _extract_json_text(content)
    if not ok:
        return None
    payload = json.loads(text)
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
        return payload["tool_calls"]
    return None


def _openai_completion(
    *,
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content if not tool_calls else content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_handler_class(
    *,
    chat_fn: ChatFn,
    model: str,
    journal_path: Path,
    call_lock: threading.Lock,
) -> type[BaseHTTPRequestHandler]:
    class RelayHandler(BaseHTTPRequestHandler):
        server_version = "relais-cursor-openai/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, status: int, body: bytes, content_type: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._send_text(200, b"ok\n")
                return
            if self.path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": model, "object": "model"}],
                    },
                )
                return
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            try:
                request = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"error": {"message": "JSON invalide", "type": "invalid_request_error"}})
                return

            messages = request.get("messages") or []
            response_format = request.get("response_format") or {}
            tools = request.get("tools")
            tool_choice = request.get("tool_choice", "auto")
            json_demande = isinstance(response_format, dict) and response_format.get("type") == "json_object"

            system, user = _concat_messages(messages)
            if json_demande:
                system = f"{system}\n\n{JSON_SYSTEM_SUFFIX}".strip() if system else JSON_SYSTEM_SUFFIX
            if tools:
                tools_hint = _tools_instruction(tools, tool_choice)
                system = f"{system}\n\n{tools_hint}".strip() if system else tools_hint

            car_entree = len(system) + len(user)
            started = time.monotonic()
            statut = "ok"
            json_ok = True
            content = ""
            tool_calls: list[dict[str, Any]] | None = None
            try:
                with call_lock:
                    raw_content, _meta = chat_fn(system, user)
                if tools:
                    tool_calls = _parse_tool_calls(raw_content)
                    if tool_calls:
                        content = raw_content if raw_content.strip() else ""
                    else:
                        # Outils demandés, aucun appel exploitable : erreur, jamais un 200 vide
                        # que mem0 prendrait pour une extraction réussie (revue Codex #123).
                        content, json_ok = _extract_json_text(raw_content)
                        json_ok = False
                        statut = "502"
                elif json_demande:
                    content, json_ok = _extract_json_text(raw_content)
                    if not json_ok:
                        statut = "502"
                else:
                    content = raw_content
            except Exception:
                statut = "erreur"
                json_ok = False
                content = "{}" if json_demande or tools else ""
                self._send_json(
                    500,
                    {"error": {"message": "échec cursor-agent", "type": "server_error"}},
                )
            finally:
                elapsed = round(time.monotonic() - started, 3)
                car_sortie = len(content)
                _journaliser(
                    journal_path,
                    {
                        "fournisseur": "cursor",
                        "modele": model,
                        "secondes": elapsed,
                        "car_entree": car_entree,
                        "car_sortie": car_sortie,
                        "json_demande": json_demande,
                        "json_ok": json_ok,
                        "statut": statut,
                    },
                )

            if statut == "erreur":
                return

            if (json_demande and not json_ok) or statut == "502":
                self._send_json(
                    502,
                    {
                        "error": {
                            "message": "JSON invalide ou aucun appel d'outil dans la sortie cursor-agent",
                            "type": "relay_error",
                        }
                    },
                )
                return

            prompt_tokens = _estimate_tokens(car_entree)
            completion_tokens = _estimate_tokens(car_sortie)
            payload = _openai_completion(
                model=model,
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tool_calls=tool_calls,
            )
            self._send_json(200, payload)

    return RelayHandler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int,
    model: str,
    chat_fn: ChatFn,
    journal_path: Path = JOURNAL_APPELS,
) -> ThreadingHTTPServer:
    call_lock = threading.Lock()
    handler = make_handler_class(
        chat_fn=chat_fn,
        model=model,
        journal_path=journal_path,
        call_lock=call_lock,
    )
    server = ThreadingHTTPServer((host, port), handler)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Relais OpenAI-compatible → cursor-agent")
    parser.add_argument("--port", type=int, default=56110)
    parser.add_argument("--model", default="cursor-grok-4.5-high")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--workdir", type=Path, default=CURSOR_WORKDIR_DEFAULT)
    args = parser.parse_args(argv)

    chat_fn = make_cursor_chat_fn(
        model=args.model,
        workdir=args.workdir,
        timeout_s=args.timeout_s,
    )
    server = serve(port=args.port, model=args.model, chat_fn=chat_fn)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
