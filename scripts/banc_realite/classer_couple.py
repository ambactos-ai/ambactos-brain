# SPDX-License-Identifier: AGPL-3.0-only
"""Tri déterministe à cinq issues entre deux faits."""
from __future__ import annotations

import re
from typing import Any

from .fidelite import DATE_QUALIFIER_KEYS, extract_ordering_date
from .normalize import claim_text, normalize_text
from .resolution import (
    PALIERS,
    ResolutionContext,
    attributes_match,
    build_inventaire_from_claims,
    build_resolution_context,
    entities_match,
)

PORTEE_KEYS = frozenset(
    {
        "site",
        "location",
        "period",
        "periode",
        "instant",
        "month",
        "year",
    }
)
TEMPORAL_KEYS = frozenset(
    {
        "since",
        "valid_from",
        "valid_to",
        "periode_debut",
        "periode_fin",
        "date",
    }
)
SCOPE_KEYS = PORTEE_KEYS | TEMPORAL_KEYS

_UNIT_TO_KG = {
    "t": 1000.0,
    "ton": 1000.0,
    "tons": 1000.0,
    "tonne": 1000.0,
    "tonnes": 1000.0,
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
}

_ISO_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
_QUANTITY_RE = re.compile(
    r"(?P<num>\d[\d\s,\.]*)\s*(?P<unit>[a-zA-Z%°]+)?\s*$",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(
    r"^\s*(?P<currency>[A-Z]{3})\s+(?P<amount>\d[\d\s,\.]+)\s*$",
    re.IGNORECASE,
)


def _parse_number_token(raw: str, *, unit_hint: str | None = None) -> float:
    s = raw.strip().replace(" ", "").replace("\u00a0", "")
    if not s:
        raise ValueError("nombre vide")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        left, right = s.split(",", 1)
        if right.isdigit() and len(right) == 3:
            hint = (unit_hint or "").lower()
            if hint in {"t", "ton", "tons", "tonne", "tonnes"} and len(left) <= 2:
                s = f"{left}.{right}"
            elif hint in {"kg", "kilogram", "kilograms", "g", "gram", "grams"}:
                s = f"{left}.{right}"
            elif len(left) <= 2:
                s = f"{left}.{right}"
            else:
                s = f"{left}.{right}"
        else:
            s = s.replace(",", ".")
    return float(s)


def normaliser_quantite(text: str) -> dict[str, Any] | None:
    """Normalise une quantité textuelle vers kg (base du dossier GreenLoop)."""
    if not text:
        return None
    stripped = text.strip()
    if _ISO_DATE_RE.match(stripped):
        return None
    match = _QUANTITY_RE.fullmatch(stripped)
    if not match:
        return None
    unit = (match.group("unit") or "").lower().rstrip(".")
    try:
        value = _parse_number_token(match.group("num"), unit_hint=unit or None)
    except ValueError:
        return None
    if unit in _UNIT_TO_KG:
        return {
            "valeur_base": round(value * _UNIT_TO_KG[unit], 6),
            "unite_base": "kg",
            "valeur_brute": value,
            "unite_brute": unit,
        }
    if unit:
        return {
            "valeur_base": value,
            "unite_base": unit,
            "valeur_brute": value,
            "unite_brute": unit,
        }
    return {
        "valeur_base": value,
        "unite_base": "",
        "valeur_brute": value,
        "unite_brute": "",
    }


def _valeur_texte(fait: dict[str, Any]) -> str:
    value = fait.get("value")
    if isinstance(value, dict):
        return claim_text(value)
    return str(value or fait.get("text") or "")


def _qualifiers(fait: dict[str, Any]) -> dict[str, Any]:
    return dict(fait.get("qualifiers") or {})


def _scope_view(fait: dict[str, Any]) -> dict[str, str]:
    quals = _qualifiers(fait)
    out: dict[str, str] = {}
    for key in SCOPE_KEYS:
        if key in quals and quals[key] is not None:
            out[key] = normalize_text(str(quals[key]))
    return out


def _temporal_bounds(fait: dict[str, Any]) -> tuple[str | None, str | None]:
    quals = _qualifiers(fait)
    start = None
    end = None
    for key in ("since", "valid_from", "periode_debut", "period", "periode", "date"):
        if quals.get(key):
            start = str(quals[key]).strip()
            break
    if not start:
        start = extract_ordering_date(fait)
    for key in ("valid_to", "periode_fin"):
        if quals.get(key):
            end = str(quals[key]).strip()
            break
    return start, end


def _scopes_overlap(a: dict[str, str], b: dict[str, str]) -> bool:
    keys = set(a) | set(b)
    if not keys:
        return True
    for key in keys:
        va = a.get(key)
        vb = b.get(key)
        if va is None or vb is None:
            continue
        if va != vb:
            return False
    return True


def _scopes_incomplete(a: dict[str, str], b: dict[str, str]) -> bool:
    keys = PORTEE_KEYS & (set(a) | set(b))
    for key in keys:
        if (key in a) != (key in b):
            return True
    return False


def _temporal_scope_one_sided(a: dict[str, Any], b: dict[str, Any]) -> bool:
    quals_a = _qualifiers(a)
    quals_b = _qualifiers(b)
    has_a = any(quals_a.get(key) is not None for key in TEMPORAL_KEYS)
    has_b = any(quals_b.get(key) is not None for key in TEMPORAL_KEYS)
    return has_a != has_b


def _temporal_relation(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    a_start, a_end = _temporal_bounds(a)
    b_start, b_end = _temporal_bounds(b)
    if not a_start or not b_start:
        return None
    if a_end and b_start and a_end < b_start:
        return "ordonne"
    if b_end and a_start and b_end < a_start:
        return "ordonne"
    if a_end and b_start and a_end == b_start:
        return "ordonne"
    if b_end and a_start and b_end == a_start:
        return "ordonne"
    if a_start == b_start and (a_end or "") == (b_end or ""):
        return "identique"
    if a_start != b_start:
        return "disjoint"
    return None


def _parse_currency(text: str) -> dict[str, Any] | None:
    match = _CURRENCY_RE.match((text or "").strip())
    if not match:
        return None
    try:
        amount = _parse_number_token(match.group("amount"))
    except ValueError:
        return None
    return {"currency": match.group("currency").upper(), "amount": amount}


def _values_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    text_a = _valeur_texte(a)
    text_b = _valeur_texte(b)
    ca = _parse_currency(text_a)
    cb = _parse_currency(text_b)
    if ca and cb:
        return ca["currency"] == cb["currency"] and ca["amount"] == cb["amount"]
    if ca or cb:
        return False
    qa = normaliser_quantite(text_a)
    qb = normaliser_quantite(text_b)
    if qa and qb:
        unit_a = qa["unite_base"]
        unit_b = qb["unite_base"]
        if unit_a and unit_b and unit_a != unit_b:
            return False
        return qa["valeur_base"] == qb["valeur_base"]
    if qa or qb:
        return False
    return normalize_text(text_a) == normalize_text(text_b)


def classer_couple(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    palier: str = "P0",
    resolution_ctx: ResolutionContext | None = None,
) -> dict[str, Any]:
    """Retourne issue, raison, qualifiers_a, qualifiers_b."""
    if palier not in PALIERS:
        raise ValueError(f"palier inconnu: {palier}")
    ctx = resolution_ctx or build_resolution_context(build_inventaire_from_claims([]), [])

    entity_a = str(a.get("entity_id") or "")
    entity_b = str(b.get("entity_id") or "")
    quals_a = _qualifiers(a)
    quals_b = _qualifiers(b)
    attr_a = str(quals_a.get("attribute_id") or "")
    attr_b = str(quals_b.get("attribute_id") or "")

    if entity_a and entity_b and not entities_match(a, b, palier, ctx):
        return {
            "issue": "faits_distincts",
            "raison": "entités différentes",
            "qualifiers_a": quals_a,
            "qualifiers_b": quals_b,
        }
    if not attr_a or not attr_b:
        return {
            "issue": "indecidable",
            "raison": "attribut manquant sur au moins un des faits",
            "qualifiers_a": quals_a,
            "qualifiers_b": quals_b,
        }
    if not attributes_match(a, b, palier, ctx):
        return {
            "issue": "faits_distincts",
            "raison": "attributs différents",
            "qualifiers_a": quals_a,
            "qualifiers_b": quals_b,
        }

    scope_a = _scope_view(a)
    scope_b = _scope_view(b)
    for key in ("site", "location"):
        if key in scope_a and key in scope_b and scope_a[key] != scope_b[key]:
            return {
                "issue": "faits_distincts",
                "raison": f"portées {key} différentes",
                "qualifiers_a": quals_a,
                "qualifiers_b": quals_b,
            }

    if _values_equal(a, b):
        return {
            "issue": "accord",
            "raison": "valeurs égales après normalisation",
            "qualifiers_a": quals_a,
            "qualifiers_b": quals_b,
        }

    qa = normaliser_quantite(_valeur_texte(a))
    qb = normaliser_quantite(_valeur_texte(b))
    if qa and qb and qa["unite_base"] and qb["unite_base"] and qa["unite_base"] != qb["unite_base"]:
        return {
            "issue": "faits_distincts",
            "raison": "unités non convertibles",
            "qualifiers_a": quals_a,
            "qualifiers_b": quals_b,
        }

    if _scopes_incomplete(scope_a, scope_b) or _temporal_scope_one_sided(a, b):
        return {
            "issue": "indecidable",
            "raison": "portée incomplète d'un côté ou des deux",
            "qualifiers_a": quals_a,
            "qualifiers_b": quals_b,
        }

    temporal = _temporal_relation(a, b)
    if temporal in {"ordonne", "disjoint"}:
        return {
            "issue": "trajectoire",
            "raison": f"portées temporelles {temporal}es",
            "qualifiers_a": quals_a,
            "qualifiers_b": quals_b,
        }

    if _scopes_overlap(scope_a, scope_b):
        return {
            "issue": "dispute",
            "raison": "valeurs différentes sur une portée identique ou recouvrante",
            "qualifiers_a": quals_a,
            "qualifiers_b": quals_b,
        }

    return {
        "issue": "indecidable",
        "raison": "portée insuffisante pour trancher",
        "qualifiers_a": quals_a,
        "qualifiers_b": quals_b,
    }
