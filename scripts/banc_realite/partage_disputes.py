# SPDX-License-Identifier: AGPL-3.0-only
"""Partage figé des 28 disputes en moitié réglage / moitié écart."""
from __future__ import annotations

import random
from typing import Any

# Ne change jamais — fige le partage réglage / écart pour tout le banc.
GRAINE = 20260922


def partager(questions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    disputes = sorted(
        (q for q in questions if q.get("shape") == "dispute"),
        key=lambda q: q["id"],
    )
    rng = random.Random(GRAINE)
    shuffled = list(disputes)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    return shuffled[:mid], shuffled[mid:]
