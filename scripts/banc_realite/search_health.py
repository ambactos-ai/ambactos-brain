# SPDX-License-Identifier: AGPL-3.0-only
"""F1 — detect degraded run_search calls."""
from __future__ import annotations

from typing import Any

DEGRADED_MARKERS = ("unavailable", "degraded", "fallback", "tranche 2000")
BENIGN_WARNING = "assets_metadata_no_perceptual_memory"


def is_search_healthy(result: dict[str, Any]) -> tuple[bool, str | None]:
    strategy = result.get("strategy") or {}
    if strategy.get("fusion") != "rrf_v1":
        return False, f"fusion={strategy.get('fusion')}"
    for text in _warning_texts(result):
        lower = text.lower()
        if BENIGN_WARNING in lower:
            continue
        if any(marker in lower for marker in DEGRADED_MARKERS):
            return False, text
    return True, None


def _warning_texts(result: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for warning in result.get("warnings") or []:
        texts.append(str(warning))
    strategy = result.get("strategy") or {}
    for note in strategy.get("notes") or []:
        texts.append(str(note))
    return texts


def summarize_search_health(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    healthy = sum(1 for row in records if row.get("healthy"))
    return {
        "run_search_total": total,
        "run_search_sains": healthy,
        "run_search_degrades": total - healthy,
        "details_degrades": [row for row in records if not row.get("healthy")],
    }
