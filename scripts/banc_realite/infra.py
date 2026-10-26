# SPDX-License-Identifier: AGPL-3.0-only
"""Ephemeral Brain database + PostgREST for banc-realite (private cluster)."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import json
import os
import re
import secrets
import subprocess
import threading
import time
import uuid
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PG_BIN = Path("/usr/lib/postgresql/16/bin")
POSTGREST = Path("/usr/local/bin/postgrest")
ENGINE_ROOT = Path(__file__).resolve().parents[2]
BANC_ROOT = (BANC_VAR)
PGDATA = BANC_ROOT / "pg" / "postgres"
STATE_FILE = BANC_ROOT / "pg" / "state.json"


def fresh_schema(engine: Path) -> str:
    native = (engine / "sql/filebrain_brain_schema.sql").read_text()
    table = native[
        native.index("CREATE SCHEMA IF NOT EXISTS brain;") : native.index(
            "\n);", native.index("CREATE TABLE IF NOT EXISTS brain.claims")
        )
        + 3
    ]
    rpc_start = native.index("CREATE OR REPLACE FUNCTION brain.recall_match_mistral(")
    rpc = native[rpc_start : native.index("$fn$;", rpc_start) + 5]
    additions = """
ALTER TABLE brain.claims ADD COLUMN IF NOT EXISTS branch_ref text NOT NULL DEFAULT 'main';
CREATE UNIQUE INDEX IF NOT EXISTS claims_vault_natural_key_active_uidx
    ON brain.claims (vault_ref, natural_key) WHERE lifecycle_status = 'active';
"""
    result = table + "\n" + additions + "\n" + rpc
    executable = re.sub(r"--[^\n]*", "", result)
    if re.search(r"\b(DROP|DELETE|TRUNCATE)\b", executable, re.I):
        raise ValueError("Destructive SQL outside the benchmark contract")
    return result


def wait_postgrest_ready(
    *,
    url: str | None = None,
    service_key: str | None = None,
    postgrest_port: int = 56101,
    timeout_s: float = 60.0,
) -> None:
    """Attend un GET /claims?limit=0 en 200 (pas seulement un port ouvert)."""
    deadline = time.monotonic() + timeout_s
    headers: dict[str, str] = {}
    if service_key:
        headers["Authorization"] = f"Bearer {service_key}"
    while time.monotonic() < deadline:
        for target in _postgrest_targets(url=url, postgrest_port=postgrest_port):
            connection = HTTPConnection(target["host"], target["port"], timeout=2)
            try:
                connection.request("GET", target["path"], headers=headers)
                response = connection.getresponse()
                response.read()
                if response.status == 200:
                    return
            except OSError:
                pass
            finally:
                connection.close()
        time.sleep(0.25)
    raise RuntimeError("PostgREST did not return 200 on GET /claims?limit=0")


def _postgrest_targets(*, url: str | None, postgrest_port: int) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = [
        {"host": "127.0.0.1", "port": postgrest_port, "path": "/claims?limit=0"},
    ]
    if url:
        targets.append(
            {"host": "127.0.0.1", "port": int(url.rsplit(":", 1)[-1]), "path": "/rest/v1/claims?limit=0"}
        )
    return targets


def reset_brain_schema(connection) -> None:
    ddl = fresh_schema(ENGINE_ROOT)
    connection.execute("DROP SCHEMA IF EXISTS brain CASCADE;")
    connection.execute(ddl)
    connection.execute("NOTIFY pgrst, 'reload schema';")
    wait_postgrest_ready(postgrest_port=56101, timeout_s=90.0)


class BancDatabase:
    """Private PostgreSQL + PostgREST; data under banc-realite/pg/."""

    def __init__(self, *, resume: bool = False, service_key: str | None = None):
        self.children: list[subprocess.Popen] = []
        self.logs = []
        self.proxy = None
        self.thread = None
        self.connection = None
        secrets_path = BANC_ROOT / "secrets.env"
        if service_key is None and secrets_path.is_file():
            for line in secrets_path.read_text().splitlines():
                if line.startswith("BANC_BRAIN_SERVICE_KEY="):
                    service_key = line.split("=", 1)[1]
                    break
        self.service_key = service_key or secrets.token_urlsafe(32)
        self.postgrest_port = 56101
        self.proxy_port = 56100
        self.pg_port = 56102
        self.url = f"http://127.0.0.1:{self.proxy_port}"
        self.data = BANC_ROOT / "pg"
        self.data.mkdir(parents=True, exist_ok=True)
        self._start(resume=resume)

    def _start(self, *, resume: bool) -> None:
        import psycopg
        from psycopg.rows import dict_row

        ddl = fresh_schema(ENGINE_ROOT)
        pgdata = PGDATA
        fresh = not (resume and (pgdata / "PG_VERSION").exists())
        if resume and (pgdata / "PG_VERSION").exists():
            if (self.data / "applied-schema.sql").read_text() != ddl:
                raise ValueError("Checkpoint schema differs from pinned Brain schema")
        else:
            if pgdata.exists():
                raise FileExistsError(f"Refusing to overwrite existing pgdata: {pgdata}")
            pgdata.mkdir(mode=0o700, parents=True)
            with (self.data / "initdb.log").open("w") as log:
                subprocess.run(
                    [str(PG_BIN / "initdb"), "-D", str(pgdata), "-U", "postgres",
                     "--auth=trust", "--no-locale", "--encoding=UTF8"],
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                )
        if fresh or not self._port_open(self.pg_port):
            self._spawn(
                [str(PG_BIN / "postgres"), "-D", str(pgdata), "-p", str(self.pg_port), "-k", str(self.data),
                 "-c", "listen_addresses=127.0.0.1", "-c", "shared_buffers=64MB",
                 "-c", "max_connections=20"],
                self.data / "postgres.log",
            )
        deadline = time.monotonic() + 30
        while True:
            try:
                self.connection = psycopg.connect(
                    f"postgresql://postgres@127.0.0.1:{self.pg_port}/postgres",
                    autocommit=True,
                    row_factory=dict_row,
                )
                break
            except psycopg.OperationalError:
                if time.monotonic() > deadline or self.children[0].poll() is not None:
                    raise RuntimeError("Private PostgreSQL did not start") from None
                time.sleep(0.1)
        if fresh or not (self.data / "applied-schema.sql").exists():
            (self.data / "applied-schema.sql").write_text(ddl)
            self.connection.execute(ddl)
        if fresh or not self._port_open(self.postgrest_port):
            env = {
            **os.environ,
            "PGRST_DB_URI": f"postgresql://postgres@127.0.0.1:{self.pg_port}/postgres",
            "PGRST_DB_SCHEMAS": "brain",
            "PGRST_DB_ANON_ROLE": "postgres",
            "PGRST_DB_POOL": "4",
            "PGRST_SERVER_HOST": "127.0.0.1",
            "PGRST_SERVER_PORT": str(self.postgrest_port),
        }
            self._spawn([str(POSTGREST)], self.data / "postgrest.log", env=env)
            wait_postgrest_ready(postgrest_port=self.postgrest_port, timeout_s=30.0)

        service_key = self.service_key
        postgrest_port = self.postgrest_port

        class Proxy(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def forward(self):
                if not secrets.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + service_key
                ):
                    self.send_error(401)
                    return
                if not self.path.startswith("/rest/v1/"):
                    self.send_error(404)
                    return
                upstream = HTTPConnection("127.0.0.1", postgrest_port, timeout=120)
                try:
                    body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    headers = {
                        k: v
                        for k, v in self.headers.items()
                        if k.lower() in {"content-type", "prefer", "accept", "range", "range-unit"}
                    }
                    upstream.request(self.command, self.path[len("/rest/v1") :], body=body, headers=headers)
                    response = upstream.getresponse()
                    payload = response.read()
                    self.send_response(response.status)
                    for name in ("Content-Type", "Content-Range", "Preference-Applied"):
                        if response.getheader(name):
                            self.send_header(name, response.getheader(name))
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                finally:
                    upstream.close()

            do_GET = do_POST = do_PATCH = forward

        if not self._port_open(self.proxy_port):
            self.proxy = ThreadingHTTPServer(("127.0.0.1", self.proxy_port), Proxy)
            self.thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
            self.thread.start()
        wait_postgrest_ready(url=self.url, service_key=self.service_key, postgrest_port=self.postgrest_port)
        STATE_FILE.write_text(
            json.dumps(
                {
                    "url": self.url,
                    "service_key_env": "BANC_BRAIN_SERVICE_KEY",
                    "pgdata": str(pgdata),
                    "postgrest_port": self.postgrest_port,
                },
                indent=2,
            )
            + "\n"
        )
        (BANC_ROOT / "secrets.env").write_text(
            f"BANC_BRAIN_SERVICE_KEY={self.service_key}\n"
            f"BRAIN_POSTGREST_URL={self.url}\n"
            f"BRAIN_POSTGREST_SERVICE_KEY={self.service_key}\n"
        )
        os.chmod(BANC_ROOT / "secrets.env", 0o600)

    @staticmethod
    def _port_open(port: int) -> bool:
        connection = HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.connect()
            return True
        except OSError:
            return False
        finally:
            connection.close()

    def _spawn(self, command, log_path, *, env=None):
        log = log_path.open("a")
        self.logs.append(log)
        self.children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env))

    def export_env(self) -> None:
        os.environ["BRAIN_POSTGREST_URL"] = self.url
        os.environ["BRAIN_POSTGREST_SERVICE_KEY"] = self.service_key

    def close(self):
        if self.proxy is not None:
            self.proxy.shutdown()
            self.proxy.server_close()
            if self.thread:
                self.thread.join(timeout=5)
            self.proxy = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        for child in reversed(self.children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        for log in self.logs:
            log.close()


def pid_on_port(port: int) -> int | None:
    """PID listening on a TCP port (same approach as arreter.sh)."""
    result = subprocess.run(
        ["ss", "-H", "-ltnp", f"sport = :{port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        match = re.search(r"pid=(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _load_service_key() -> str:
    secrets_path = BANC_ROOT / "secrets.env"
    if secrets_path.is_file():
        for line in secrets_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("BANC_BRAIN_SERVICE_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _tcp_open(port: int) -> bool:
    connection = HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.connect()
        return True
    except OSError:
        return False
    finally:
        connection.close()


def _postgrest_layer_ready() -> bool:
    connection = HTTPConnection("127.0.0.1", 56101, timeout=1)
    try:
        connection.request("GET", "/claims?limit=0")
        response = connection.getresponse()
        response.read()
        return response.status == 200
    except OSError:
        return False
    finally:
        connection.close()


def _proxy_layer_ready() -> bool:
    connection = HTTPConnection("127.0.0.1", 56100, timeout=1)
    try:
        connection.request(
            "GET",
            "/rest/v1/claims?limit=0",
            headers={"Authorization": "Bearer probe"},
        )
        response = connection.getresponse()
        response.read()
        return response.status in (200, 401)
    except OSError:
        return False
    finally:
        connection.close()


def is_running() -> dict[str, bool]:
    """État des trois couches du cluster (Postgres / PostgREST / proxy)."""
    return {
        "56102": _tcp_open(56102),
        "56101": _postgrest_layer_ready(),
        "56100": _proxy_layer_ready(),
    }


def cluster_ready() -> bool:
    return all(is_running().values())


def _spawn_detached(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log = log_path.open("a")
    subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )


def _start_postgres_if_needed() -> None:
    if _tcp_open(56102):
        return
    pgdata = PGDATA
    if not (pgdata / "PG_VERSION").exists():
        raise RuntimeError(f"pgdata absent : {pgdata}")
    _spawn_detached(
        [
            str(PG_BIN / "postgres"),
            "-D",
            str(pgdata),
            "-p",
            "56102",
            "-k",
            str(BANC_ROOT / "pg"),
            "-c",
            "listen_addresses=127.0.0.1",
            "-c",
            "shared_buffers=64MB",
            "-c",
            "max_connections=20",
        ],
        BANC_ROOT / "pg" / "postgres.log",
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _tcp_open(56102):
            return
        time.sleep(0.1)
    raise RuntimeError("PostgreSQL 56102 did not start")


def _start_postgrest_if_needed() -> None:
    if _postgrest_layer_ready():
        return
    env = {
        **os.environ,
        "PGRST_DB_URI": "postgresql://postgres@127.0.0.1:56102/postgres",
        "PGRST_DB_SCHEMAS": "brain",
        "PGRST_DB_ANON_ROLE": "postgres",
        "PGRST_DB_POOL": "4",
        "PGRST_SERVER_HOST": "127.0.0.1",
        "PGRST_SERVER_PORT": "56101",
    }
    _spawn_detached([str(POSTGREST)], BANC_ROOT / "pg" / "postgrest.log", env=env)
    wait_postgrest_ready(postgrest_port=56101, timeout_s=60.0)


def ensure_proxy_running() -> None:
    if _proxy_layer_ready():
        return
    service_key = _load_service_key()
    if not service_key:
        raise RuntimeError("BANC_BRAIN_SERVICE_KEY absent")
    import sys

    _spawn_detached(
        [
            sys.executable,
            "-m",
            "banc_realite.proxy_banc",
            "--port",
            "56100",
            "--upstream",
            "56101",
            "--service-key",
            service_key,
        ],
        BANC_ROOT / "pg" / "proxy.log",
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _proxy_layer_ready():
            return
        time.sleep(0.25)
    raise RuntimeError("proxy 56100 did not start")


def demarrer_cluster(*, postgres: bool = True) -> None:
    """Démarre les couches manquantes (détachées). Postgres optionnel si déjà debout."""
    if postgres:
        _start_postgres_if_needed()
    elif not _tcp_open(56102):
        raise RuntimeError("PostgreSQL 56102 requis mais absent")
    _start_postgrest_if_needed()
    ensure_proxy_running()
    service_key = _load_service_key()
    if service_key:
        (BANC_ROOT / "secrets.env").write_text(
            f"BANC_BRAIN_SERVICE_KEY={service_key}\n"
            f"BRAIN_POSTGREST_URL=http://127.0.0.1:56100\n"
            f"BRAIN_POSTGREST_SERVICE_KEY={service_key}\n"
        )
        os.chmod(BANC_ROOT / "secrets.env", 0o600)
    os.environ["BRAIN_POSTGREST_URL"] = "http://127.0.0.1:56100"
    if service_key:
        os.environ["BRAIN_POSTGREST_SERVICE_KEY"] = service_key


def start_cluster() -> None:
    """Relance uniquement les couches manquantes via demarrer.sh ; jamais Postgres si 56102 répond."""
    status = is_running()
    if all(status.values()):
        return
    if status["56102"]:
        demarrer_cluster(postgres=False)
        return
    subprocess.run([str(BANC_ROOT / "demarrer.sh")], check=True, cwd=str(ENGINE_ROOT))


def stop_cluster() -> None:
    """Stop the private banc cluster via arreter.sh (mandate entrypoint)."""
    subprocess.run([str(BANC_ROOT / "arreter.sh")], check=True)


def connect_existing():
    import psycopg
    from psycopg.rows import dict_row

    secrets_path = BANC_ROOT / "secrets.env"
    if secrets_path.is_file():
        for line in secrets_path.read_text().splitlines():
            if "=" in line and not os.environ.get(line.split("=", 1)[0]):
                key, value = line.split("=", 1)
                os.environ[key] = value
    connection = psycopg.connect(
        "postgresql://postgres@127.0.0.1:56102/postgres",
        autocommit=True,
        row_factory=dict_row,
    )
    return connection
