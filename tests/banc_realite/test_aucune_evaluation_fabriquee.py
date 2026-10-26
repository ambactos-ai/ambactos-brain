# SPDX-License-Identifier: AGPL-3.0-only
"""Garde statique : le banc ne fabrique plus d'évaluation d'identité (ADR 0026 §8)."""
from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "banc_realite"
FORBIDDEN_CALLS = frozenset(
    {
        "authorize_auto_link",
        "IdentityEvaluationReport",
        "EvaluationManifest",
        "AutoLinkAuthorization",
    }
)


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _forbidden_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in FORBIDDEN_CALLS:
            found.append(f"{path.name}:{node.lineno}:{name}")
    return found


def test_banc_realite_scripts_do_not_fabricate_identity_evaluation() -> None:
    violations: list[str] = []
    for path in sorted(SCRIPTS_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        violations.extend(_forbidden_calls(path))
    assert violations == []
