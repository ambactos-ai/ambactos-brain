# SPDX-License-Identifier: AGPL-3.0-only
"""Étage 3 — appels modèle via cursor-agent (Grok servi par Cursor)."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from .etage3 import ChatFn

CURSOR_AGENT = (Path.home() / ".local" / "bin" / "cursor-agent")
CURSOR_WORKDIR = BANC_VAR / 'cursor-vide'
CURSOR_ENV = {
    "HOME": str(Path.home()),
    "PATH": str(Path.home() / ".local" / "bin") + ':/usr/local/bin:/usr/bin:/bin',
}
HOME_MASQUE = str(Path.home())
MSG_ISOLATION_INDISPONIBLE = "isolation indisponible"
MSG_SORTIE_CONTAMINEE = "sortie contaminée par un secret"
# Auth réelle : ~/.config/cursor/auth.json (accessToken / refreshToken).
# ~/.config/cursor-agent n'existe pas. ~/.cursor/cli-config.json a authInfo
# mais le dossier ~/.cursor contient mcp.json — on ne le monte pas.
# cursor-agent --help n'offre pas --auth-file : --api-key / CURSOR_API_KEY
# exposeraient le jeton dans l'env de l'unité. On monte le seul fichier
# auth.json (pas le répertoire) dans le HOME masqué.
# (--approve-mcps fait l'inverse) : HOME masqué + cwd sans mcp.json.
CURSOR_AUTH_DIR = Path.home() / ".config" / "cursor"
CURSOR_AUTH_JSON = CURSOR_AUTH_DIR / "auth.json"
ISOLE_WORKDIR_PARENT = Path.home() / ".agents/var/cursor-chat-isole"
_MARQUEURS_ENV_SECRET = ("TOKEN", "SECRET", "KEY", "JETON")
_log = logging.getLogger(__name__)
_MARQUEURS_ECHEC_DEMARRAGE = (
    "failed to connect to bus",
    "failed to start transient",
    "failed to start",
    "failed to run as unit",
    "systemd-run: command not found",
)

RunFn = Callable[..., tuple[int, str, str]]


def _repertoire_installation_cursor() -> Path:
    return Path(os.path.realpath(CURSOR_AGENT)).parent


def _chemins_bind_ro(workdir: Path) -> list[str]:
    return [
        str(workdir),
        str(CURSOR_AGENT),
        str(_repertoire_installation_cursor()),
        str(CURSOR_AUTH_JSON),
    ]


def _env_isole() -> dict[str, str]:
    """Environnement du service isolé : PATH, HOME temporaire, LANG."""
    return {
        "PATH": CURSOR_ENV["PATH"],
        "HOME": HOME_MASQUE,
        "LANG": os.environ.get("LANG") or "C.UTF-8",
    }


def _chaines_secretes(obj: Any) -> list[str]:
    if isinstance(obj, str) and len(obj) > 16:
        return [obj]
    if isinstance(obj, Mapping):
        trouvees: list[str] = []
        for valeur in obj.values():
            trouvees.extend(_chaines_secretes(valeur))
        return trouvees
    if isinstance(obj, list):
        trouvees = []
        for valeur in obj:
            trouvees.extend(_chaines_secretes(valeur))
        return trouvees
    return []


def _secrets_auth_json(auth_path: Path | None = None) -> list[str]:
    path = auth_path if auth_path is not None else CURSOR_AUTH_JSON
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    return _chaines_secretes(payload)


def _secrets_env(env: Mapping[str, str] | None = None) -> list[str]:
    source = os.environ if env is None else env
    secrets: list[str] = []
    for cle, valeur in source.items():
        nom = str(cle).upper()
        if not any(marqueur in nom for marqueur in _MARQUEURS_ENV_SECRET):
            continue
        if isinstance(valeur, str) and len(valeur) > 16:
            secrets.append(valeur)
    return secrets


def _secrets_a_filtrer() -> list[str]:
    return _secrets_auth_json() + _secrets_env()


def _sortie_contaminee(texte: str, secrets: list[str]) -> bool:
    if not texte or not secrets:
        return False
    for secret in secrets:
        if secret in texte:
            return True
        b64 = base64.b64encode(secret.encode("utf-8")).decode("ascii")
        if b64 in texte:
            return True
        if len(secret) >= 12:
            for i in range(0, len(secret) - 11):
                if secret[i : i + 12] in texte:
                    return True
    return False


def _assurer_sortie_propre(stdout: str, result: str) -> None:
    secrets = _secrets_a_filtrer()
    if _sortie_contaminee(stdout, secrets) or _sortie_contaminee(result, secrets):
        _log.error(MSG_SORTIE_CONTAMINEE)
        raise RuntimeError(MSG_SORTIE_CONTAMINEE)


def _binaire_systemd_run() -> str:
    trouve = shutil.which("systemd-run")
    if not trouve:
        raise RuntimeError(
            f"{MSG_ISOLATION_INDISPONIBLE} : systemd-run introuvable"
        )
    return trouve


def _env_hote_systemd() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in (
        "PATH",
        "USER",
        "LOGNAME",
        "LANG",
        "HOME",
        "DBUS_SESSION_BUS_ADDRESS",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    env["XDG_RUNTIME_DIR"] = os.environ.get("XDG_RUNTIME_DIR") or (
        f"/run/user/{os.getuid()}"
    )
    return env


def _construire_argv_isole(
    argv_cursor: list[str],
    *,
    workdir: Path,
    timeout: int,
    env: dict[str, str],
    systemd_bin: str = "systemd-run",
) -> list[str]:
    """Enveloppe cursor-agent dans systemd-run (home masqué, binds en lecture)."""
    binaire_reel = str(_repertoire_installation_cursor() / Path(CURSOR_AGENT).name)
    inner = list(argv_cursor)
    if inner:
        inner[0] = binaire_reel
    cmd: list[str] = [
        systemd_bin,
        "--user",
        "--wait",
        "--collect",
        "--pipe",
        "--quiet",
        "-p",
        f"TemporaryFileSystem={HOME_MASQUE}",
    ]
    for chemin in _chemins_bind_ro(workdir):
        cmd.extend(["-p", f"BindReadOnlyPaths={chemin}"])
    cmd.extend(
        [
            "-p",
            "PrivateTmp=yes",
            "-p",
            "NoNewPrivileges=yes",
            "-p",
            "ProtectSystem=strict",
            "-p",
            "MemoryMax=1G",
            "-p",
            f"RuntimeMaxSec={timeout}",
            f"--working-directory={workdir}",
        ]
    )
    for key, value in env.items():
        cmd.extend(["--setenv", f"{key}={value}"])
    cmd.append("--")
    cmd.extend(inner)
    return cmd


def _echec_demarrage_isolation(returncode: int, stderr: str) -> bool:
    if returncode in {127, 203}:
        return True
    texte = stderr.lower()
    return any(marqueur in texte for marqueur in _MARQUEURS_ECHEC_DEMARRAGE)


def _run_fn_isole(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[int, str, str]:
    """Lance cursor-agent dans un service utilisateur isolé ; pas de repli."""
    del cwd  # le cwd réel est un dossier temporaire qui ne contient que l'invite
    del env  # l'unité isolée ne reçoit que PATH / HOME temporaire / LANG
    systemd_bin = _binaire_systemd_run()
    ISOLE_WORKDIR_PARENT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="invite-", dir=str(ISOLE_WORKDIR_PARENT)
    ) as tmp:
        workdir = Path(tmp)
        invite = argv[-1] if argv else ""
        (workdir / "invite.txt").write_text(invite, encoding="utf-8")
        cmd = _construire_argv_isole(
            argv,
            workdir=workdir,
            timeout=timeout,
            env=_env_isole(),
            systemd_bin=systemd_bin,
        )
        try:
            result = subprocess.run(
                cmd,
                env=_env_hote_systemd(),
                timeout=timeout + 5,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{MSG_ISOLATION_INDISPONIBLE} : systemd-run introuvable"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"{MSG_ISOLATION_INDISPONIBLE} : {exc}"
            ) from exc
        except subprocess.TimeoutExpired:
            return 124, "", f"délai de {timeout} s dépassé"
        if _echec_demarrage_isolation(result.returncode, result.stderr):
            detail = (result.stderr or result.stdout or "").strip()[:200]
            raise RuntimeError(
                f"{MSG_ISOLATION_INDISPONIBLE} : {detail or f'code {result.returncode}'}"
            )
        return result.returncode, result.stdout, result.stderr


def _default_run_fn(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    isole: bool = False,
) -> tuple[int, str, str]:
    if isole:
        return _run_fn_isole(argv, cwd=cwd, env=env, timeout=timeout)
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        # Un dépassement vaut un échec ordinaire : la seconde tentative s'applique.
        return 124, "", f"délai de {timeout} s dépassé"
    return result.returncode, result.stdout, result.stderr


def _ensure_empty_workdir(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    if any(workdir.iterdir()):
        raise RuntimeError(f"dossier de travail cursor non vide: {workdir}")


def _parse_cursor_json(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("stdout vide")
    return json.loads(lines[-1])


def _meta_from_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
    usage_raw = payload.get("usage") or {}
    return {
        "model_returned": model,
        "fournisseur": "cursor-agent",
        "finish_reason": None,
        "usage": {
            "prompt_tokens": usage_raw.get("inputTokens"),
            "completion_tokens": usage_raw.get("outputTokens"),
            "cache_read_tokens": usage_raw.get("cacheReadTokens"),
            "cache_write_tokens": usage_raw.get("cacheWriteTokens"),
        },
        "session_id": payload.get("session_id"),
        "request_id": payload.get("request_id"),
        "duration_api_ms": payload.get("duration_api_ms"),
    }


def assert_cursor_model_available(model: str) -> None:
    result = subprocess.run(
        [str(CURSOR_AGENT), "--list-models"],
        capture_output=True,
        text=True,
        timeout=30,
        env=CURSOR_ENV,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cursor-agent --list-models a échoué ({result.returncode}): "
            f"{result.stderr[:200]}"
        )
    for line in result.stdout.splitlines():
        slug = line.split(" - ", 1)[0].strip()
        if slug == model:
            return
    raise RuntimeError(f"modèle cursor indisponible: {model}")


def make_cursor_chat_fn(
    *,
    model: str = "cursor-grok-4.5-high",
    workdir: Path = CURSOR_WORKDIR,
    timeout_s: int = 300,
    run_fn: RunFn | None = None,
    isole: bool = False,
) -> ChatFn:
    if run_fn is None and os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("appel cursor-agent interdit sous pytest")

    if not isole:
        _ensure_empty_workdir(workdir)
    if run_fn is None:
        runner = _run_fn_isole if isole else _default_run_fn
    else:
        runner = run_fn

    def _invoke(system: str, user: str) -> tuple[int, str, str]:
        prompt = system + "\n\n" + user
        argv = [
            str(CURSOR_AGENT),
            "--model",
            model,
            "--mode",
            "ask",
            "--print",
            "--output-format",
            "json",
            "--trust",
            prompt,
        ]
        return runner(
            argv,
            cwd=workdir,
            env=CURSOR_ENV,
            timeout=timeout_s,
        )

    def _attempt(system: str, user: str) -> tuple[str, dict[str, Any]]:
        returncode, stdout, stderr = _invoke(system, user)
        if returncode != 0:
            raise RuntimeError(
                f"cursor-agent code {returncode}: {stderr[:200]}"
            )
        try:
            payload = _parse_cursor_json(stdout)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"JSON illisible: {exc}") from exc
        if payload.get("type") != "result":
            raise RuntimeError(f"subtype {payload.get('subtype')}: {stderr[:200]}")
        if payload.get("subtype") != "success":
            raise RuntimeError(f"subtype {payload.get('subtype')}: {stderr[:200]}")
        if payload.get("is_error"):
            raise RuntimeError(f"subtype {payload.get('subtype')}: {stderr[:200]}")
        result = str(payload.get("result") or "").strip()
        if not result:
            raise RuntimeError(f"subtype {payload.get('subtype')}: {stderr[:200]}")
        _assurer_sortie_propre(stdout, result)
        return result, _meta_from_payload(payload, model)

    def chat(system: str, user: str) -> tuple[str, dict[str, Any]]:
        from .plafond_appels import PlafondAtteint

        try:
            return _attempt(system, user)
        except PlafondAtteint:
            raise
        except RuntimeError:
            time.sleep(5)
            try:
                return _attempt(system, user)
            except PlafondAtteint:
                raise
            except RuntimeError as exc:
                raise RuntimeError(str(exc)) from exc

    return chat
