# SPDX-License-Identifier: AGPL-3.0-only
"""Paramètres de découpe par dossier — taille de passage dérivée du recouvrement."""
from __future__ import annotations

from typing import Any

from .dossiers import Dossier, dossier_courant
from .passages import MAX_CHARS, OVERLAP_CHARS, calculer_recouvrement_depuis_or

# Règle : passage_chars >= ratio × recouvrement (issue #127).
RATIO_PASSAGE_RECUVREMENT = 4
ARRONDI_PASSAGE_CHARS = 50


def _arrondir_passage_chars(n: int) -> int:
    step = ARRONDI_PASSAGE_CHARS
    return ((n + step - 1) // step) * step


def parametres_decoupe_depuis_or(
    assertions: list[dict[str, Any]],
    *,
    ratio: int = RATIO_PASSAGE_RECUVREMENT,
    passage_chars_defaut: int = MAX_CHARS,
) -> dict[str, Any]:
    """Calcule recouvrement depuis l'or, puis taille de passage >= ratio × recouvrement."""
    rec = calculer_recouvrement_depuis_or(assertions, max_chars=None)
    overlap = int(rec["recouvrement_chars"])
    passage_requis = ratio * overlap if overlap > 0 else passage_chars_defaut
    passage_chars = max(passage_chars_defaut, passage_requis)
    if passage_chars > passage_chars_defaut:
        passage_chars = _arrondir_passage_chars(passage_chars)
    rec["max_chars_passage"] = passage_chars
    rec["ratio_passage_recouvrement"] = ratio
    rec["passage_chars_defaut"] = passage_chars_defaut
    rec["passage_requis_min"] = passage_requis
    rec["regle"] = (
        f"passage >= {ratio} × recouvrement "
        f"({passage_chars} >= {ratio} × {overlap})"
    )
    rec["passage_trop_petit"] = overlap > passage_chars // 2
    return rec


def parametres_decoupe_greenloop() -> dict[str, Any]:
    """GreenLoop — découpe historique inchangée (1 200 / 150)."""
    return {
        "max_chars_passage": MAX_CHARS,
        "recouvrement_chars": OVERLAP_CHARS,
        "ratio_passage_recouvrement": RATIO_PASSAGE_RECUVREMENT,
        "passage_chars_defaut": MAX_CHARS,
        "regle": f"historique fixe ({MAX_CHARS} / {OVERLAP_CHARS})",
        "passage_trop_petit": OVERLAP_CHARS > MAX_CHARS // 2,
        "calcul": f"overlap fixe {OVERLAP_CHARS} car.",
    }


def parametres_decoupe_dossier(
    dossier: Dossier | str,
    assertions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Paramètres de découpe pour un dossier du banc comparatif."""
    d = dossier_courant(dossier) if isinstance(dossier, str) else dossier
    if d.nom == "greenloop":
        return parametres_decoupe_greenloop()
    if assertions is None:
        from .or_dossier import load_gold

        assertions = load_gold(d)["assertions"]
    return parametres_decoupe_depuis_or(assertions)
