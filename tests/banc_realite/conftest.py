# SPDX-License-Identifier: AGPL-3.0-only
from banc_realite.dossiers import BANC_VAR
import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BANC_ROOT = (BANC_VAR)
BANC_SORTIES = BANC_ROOT / "sorties"
BANC_DOSSIERS = BANC_ROOT / "dossiers"
for path in (ROOT / "scripts", ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _blocked_subprocess(*args, **kwargs):
    raise RuntimeError(
        "subprocess interdit dans tests/banc_realite — un test tente de lancer un processus"
    )


def _blocked_os_system(command):
    raise RuntimeError(
        "os.system interdit dans tests/banc_realite — un test tente de lancer un processus"
    )


@pytest.fixture(autouse=True)
def block_process_launchers(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _blocked_subprocess)
    monkeypatch.setattr(subprocess, "Popen", _blocked_subprocess)
    monkeypatch.setattr(os, "system", _blocked_os_system)


@pytest.fixture(autouse=True)
def isolate_banc_outputs(tmp_path, monkeypatch):
    ingest_log = tmp_path / "ingestion.jsonl"
    monkeypatch.setenv("INGEST_LOG", str(ingest_log))
    try:
        monkeypatch.setattr("banc_realite.ingest_runner.INGEST_LOG", ingest_log)
    except Exception:
        pass

    tmp_resolved = tmp_path.resolve()

    def _is_forbidden_write(path: Path) -> bool:
        resolved = path.resolve()
        if resolved == tmp_resolved or tmp_resolved in resolved.parents:
            return False
        for blocked in (BANC_SORTIES, BANC_DOSSIERS):
            blocked_res = blocked.resolve()
            if resolved == blocked_res or blocked_res in resolved.parents:
                return True
        return False

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        path = Path(os.fspath(file)).resolve()
        if any(ch in mode for ch in ("w", "a", "x", "+")):
            if _is_forbidden_write(path):
                raise AssertionError(
                    "écriture interdite sous le banc réel pendant les tests : "
                    f"{path}"
                )
        return real_open(file, mode, *args, **kwargs)

    real_write_text = Path.write_text
    real_write_bytes = Path.write_bytes

    def guarded_write_text(self, data, *args, **kwargs):
        if _is_forbidden_write(self):
            raise AssertionError(
                "écriture interdite sous le banc réel pendant les tests : "
                f"{self.resolve()}"
            )
        return real_write_text(self, data, *args, **kwargs)

    def guarded_write_bytes(self, data, *args, **kwargs):
        if _is_forbidden_write(self):
            raise AssertionError(
                "écriture interdite sous le banc réel pendant les tests : "
                f"{self.resolve()}"
            )
        return real_write_bytes(self, data, *args, **kwargs)

    real_path_open = Path.open

    def guarded_path_open(self, mode="r", *args, **kwargs):
        if any(ch in str(mode) for ch in ("w", "a", "x", "+")):
            if _is_forbidden_write(self):
                raise AssertionError(
                    "écriture interdite sous le banc réel pendant les tests : "
                    f"{self.resolve()}"
                )
        return real_path_open(self, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    monkeypatch.setattr(Path, "write_text", guarded_write_text)
    monkeypatch.setattr(Path, "write_bytes", guarded_write_bytes)
    monkeypatch.setattr(Path, "open", guarded_path_open)
