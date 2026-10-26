# SPDX-License-Identifier: AGPL-3.0-only
"""Contrôle de découpe — assertions tenues par une unité phrase."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from .config import BANC_ROOT
from .fidelite import _assertion_needle, load_gold
from .holders import PieceCache
from .porte_phrase import assertions_tenues_par_une_unite, phrases_du_passage

SORTIES_DIR = BANC_ROOT / "sorties" / "fenetres-phrases"
RESULTAT_PATH = SORTIES_DIR / "controle_decoupe.json"
DIAGNOSTIC_PATH = BANC_ROOT / "sorties" / "diagnostic-couverture" / "resultat.json"

_NUMBER_RE = re.compile(r"\d+")


def _load_extractions_infideles(
    gold: dict[str, Any],
    diagnostic_path: Path,
) -> list[dict[str, Any]]:
    payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assertion_index = gold["assertion_index"]
    assertions: list[dict[str, Any]] = []
    for line in payload["lignes"]:
        if line.get("verdict") != "extrait_infidele":
            continue
        assertions.append(assertion_index[line["assertion_id"]])
    return assertions


def _pieces_text(assertions: list[dict[str, Any]], cache: PieceCache) -> dict[str, str]:
    from .normalize import piece_basename

    pieces: dict[str, str] = {}
    for assertion in assertions:
        piece = piece_basename(assertion.get("source") or "")
        if piece in pieces:
            continue
        lines = cache.get_lines(piece)
        pieces[piece] = "\n".join(lines) if lines is not None else ""
    return pieces


def _cause_probable(
    assertion: dict[str, Any],
    piece_text: str,
    *,
    fenetre: int,
) -> str:
    needle = _assertion_needle(assertion)
    if "|" in str(assertion.get("quote") or assertion.get("content") or ""):
        return "tableau"
    units = phrases_du_passage(piece_text, fenetre=fenetre)
    if not units:
        return "passage vide"
    longest = max(len(unit) for unit in units)
    if len(needle) > longest and fenetre == 2 and longest >= 450:
        return "coupe à 450"
    sentence_count = len(re.split(r"(?<=[.!?;])(?=\s|$)", piece_text.strip()))
    if fenetre == 2 and sentence_count > 2:
        return "plus de deux phrases"
    return "autre"


def run_controle_decoupe(
    *,
    diagnostic_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    gold = load_gold()
    diagnostic_path = diagnostic_path or DIAGNOSTIC_PATH
    output_path = output_path or RESULTAT_PATH
    assertions = _load_extractions_infideles(gold, diagnostic_path)
    cache = PieceCache()
    pieces_text = _pieces_text(assertions, cache)

    held_1 = assertions_tenues_par_une_unite(assertions, pieces_text, fenetre=1)
    held_2 = assertions_tenues_par_une_unite(assertions, pieces_text, fenetre=2)

    fenetre1 = sum(1 for assertion in assertions if held_1.get(_assertion_needle(assertion)))
    fenetre2 = sum(1 for assertion in assertions if held_2.get(_assertion_needle(assertion)))

    encore_coupees: list[dict[str, Any]] = []
    for assertion in assertions:
        needle = _assertion_needle(assertion)
        if held_2.get(needle, False):
            continue
        piece = assertion.get("source") or ""
        from .normalize import piece_basename

        piece_name = piece_basename(piece)
        encore_coupees.append(
            {
                "assertion_id": assertion.get("id"),
                "needle": needle[:120],
                "piece": piece_name,
                "cause_probable": _cause_probable(
                    assertion,
                    pieces_text.get(piece_name, ""),
                    fenetre=2,
                ),
            }
        )

    payload = {
        "assertions_total": len(assertions),
        "fenetre1": fenetre1,
        "fenetre2": fenetre2,
        "encore_coupees": encore_coupees,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"tenues par une unité (fenetre=1) : {fenetre1}/{len(assertions)}")
    print(f"tenues par une unité (fenetre=2) : {fenetre2}/{len(assertions)}")
    if encore_coupees:
        print("encore coupées à fenetre=2 :")
        for row in encore_coupees:
            print(f"  - {row['assertion_id']} ({row['cause_probable']}) : {row['needle'][:80]}")
    return {**payload, "files_written": [str(output_path)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contrôle découpe porte phrase.")
    parser.parse_args(argv)
    try:
        run_controle_decoupe()
    except Exception as exc:
        print(json.dumps({"status": "arret", "erreur": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
