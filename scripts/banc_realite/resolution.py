# SPDX-License-Identifier: AGPL-3.0-only
"""Résolution déterministe d'entité et d'attribut par paliers cumulatifs."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from brain.claiming.form_aliases import _token_set, observed_form_aliases, slug_id
except ImportError:
    _token_set = None  # type: ignore[misc,assignment]
    observed_form_aliases = None  # type: ignore[misc,assignment]
    slug_id = None  # type: ignore[misc,assignment]

from .config import BANC_ROOT, VAULT_REF, export_runtime_env
from .normalize import claim_text

PALIERS = ("P0", "P1", "P2", "P3", "P4")
SORTIES_RESOLUTION = BANC_ROOT / "sorties" / "resolution"
INVENTAIRE_PATH = SORTIES_RESOLUTION / "inventaire.json"

_BASE_ROLE_PREFIXES = (
    "compte_",
    "account_",
    "invoice_",
    "facture_",
    "note_",
    "card_",
    "gl_",
    "receivables_",
)
_NUMERIC_PREFIX_RE = re.compile(r"^\d+_")
_YEAR_SUFFIX_RE = re.compile(r"_(19|20)\d{2}$")
_PERIOD_KEYS = frozenset({"period", "periode", "year", "month", "since", "valid_from"})
_DOCUMENT_TOKENS = frozenset(
    {
        "collection",
        "register",
        "review",
        "invoice",
        "ticket",
        "index",
        "export",
        "certificate",
        "weigh",
        "recorded",
        "visits",
        "monthly",
        "billing",
        "scale",
        "treatment",
        "internal",
        "service",
        "preparation",
        "transfer",
        "scenario",
        "file",
        "copy",
        "annual",
        "customer",
        "january",
        "register",
    }
)
_FR_EN_TOKENS = {
    "collectes": "collected",
    "collecte": "collection",
    "sites": "sites",
    "paiement": "payment",
    "nombre": "count",
    "montant": "amount",
    "date": "date",
    "periode": "period",
    "client": "customer",
    "devise": "currency",
    "visites": "visits",
    "echeance": "due",
    "reglement": "payment",
}


def normalize_entity_form(entity_id: str) -> str:
    return slug_id(entity_id or "")


def strip_role_prefix(entity_id: str, prefixes: tuple[str, ...] | list[str]) -> str:
    current = entity_id or ""
    changed = True
    while changed:
        changed = False
        if _NUMERIC_PREFIX_RE.match(current):
            current = _NUMERIC_PREFIX_RE.sub("", current, count=1)
            changed = True
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix) :]
                changed = True
                break
    return current


def _singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def normalize_entity_tokens(entity_id: str, prefixes: tuple[str, ...]) -> list[str]:
    stripped = strip_role_prefix(entity_id, prefixes)
    tokens = [t for t in stripped.split("_") if t]
    return sorted({_singularize(t) for t in tokens})


def _word_boundary_inclusion(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return longer.startswith(f"{shorter}_") or longer.endswith(f"_{shorter}")


def _is_document_slug(slug: str) -> bool:
    tokens = set(slug.split("_"))
    if tokens & _DOCUMENT_TOKENS:
        return True
    return bool(_NUMERIC_PREFIX_RE.match(slug))


def extract_customer_key(text: str) -> str | None:
    if not text:
        return None
    first_line = text.strip().split("\n", 1)[0].strip()
    candidates: list[str] = []
    if "—" in first_line:
        left, right = first_line.split("—", 1)
        candidates.append(right.split("/")[0].strip())
        candidates.append(left.strip())
    else:
        candidates.append(first_line.split("/")[0].strip())
    for cand in candidates:
        slug = slug_id(cand)
        if slug and not _is_document_slug(slug):
            return slug
    return None


def strip_year_period_suffix(attribute_id: str, qualifiers: dict[str, Any]) -> str:
    attr = attribute_id or ""
    if not attr:
        return attr
    year_match = _YEAR_SUFFIX_RE.search(attr)
    if not year_match:
        return attr
    period = next((str(qualifiers.get(key)) for key in _PERIOD_KEYS if qualifiers.get(key)), "")
    suffix_year = year_match.group(0).lstrip("_")
    if period and suffix_year == period:
        return _YEAR_SUFFIX_RE.sub("", attr)
    return attr


def _attr_canonical_tokens(attribute_id: str) -> frozenset[str]:
    tokens = [t for t in slug_id(attribute_id).split("_") if t]
    expanded: set[str] = set()
    for token in tokens:
        expanded.add(token)
        if token in _FR_EN_TOKENS:
            expanded.add(_FR_EN_TOKENS[token])
        for fr, en in _FR_EN_TOKENS.items():
            if token == en:
                expanded.add(fr)
    return frozenset(expanded)


def _discover_role_prefixes(entities: list[str]) -> list[str]:
    known = list(_BASE_ROLE_PREFIXES)
    entity_set = set(entities)
    discovered: set[str] = set()
    for entity in entities:
        for prefix in known:
            if entity.startswith(prefix):
                candidate = entity[len(prefix) :]
                if candidate in entity_set:
                    discovered.add(prefix)
    prefixes = sorted(set(known) | discovered, key=len, reverse=True)
    return prefixes


def _discover_bilingual_pairs(attributes: list[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    attrs = sorted({a for a in attributes if a})
    by_tokens: dict[frozenset[str], list[str]] = defaultdict(list)
    for attr in attrs:
        by_tokens[_attr_canonical_tokens(attr)].append(attr)
    for group in by_tokens.values():
        if len(group) < 2:
            continue
        canonical = min(group, key=len)
        for attr in group:
            pairs[attr] = canonical
    explicit = {
        "sites_collectes": "collected_sites",
        "collected_sites": "collected_sites",
        "sites_collectes_2024": "collected_sites",
        "date_paiement": "payment_date",
        "payment_date": "payment_date",
        "nombre_collectes": "collection_count",
        "collection_count": "collection_count",
    }
    for attr, target in explicit.items():
        if attr in attrs:
            pairs[attr] = target
    return pairs


def build_inventaire_from_claims(claims: list[dict[str, Any]]) -> dict[str, Any]:
    entities = Counter()
    attributes = Counter()
    entity_forms: dict[str, int] = Counter()
    attr_forms: dict[str, int] = Counter()
    prefix_hits: Counter[str] = Counter()

    for claim in claims:
        entity = str(claim.get("entity_id") or "")
        if entity:
            entities[entity] += 1
            entity_forms[normalize_entity_form(entity)] += 1
            for prefix in _BASE_ROLE_PREFIXES:
                if entity.startswith(prefix):
                    prefix_hits[prefix] += 1
            if _NUMERIC_PREFIX_RE.match(entity):
                prefix_hits["numeric_document_prefix"] += 1
        quals = claim.get("qualifiers") or {}
        attr = str(quals.get("attribute_id") or "")
        if attr:
            attributes[attr] += 1
            attr_forms[normalize_entity_form(attr)] += 1

    role_prefixes = _discover_role_prefixes(list(entities))
    bilingual = _discover_bilingual_pairs(list(attributes))

    return {
        "entities": {
            "count": len(entities),
            "forms": dict(entities.most_common()),
            "prefixes_frequents": dict(prefix_hits.most_common()),
            "role_prefixes": role_prefixes,
        },
        "attributes": {
            "count": len(attributes),
            "forms": dict(attributes.most_common(200)),
            "bilingual_pairs": bilingual,
        },
        "claims_total": len(claims),
    }


@dataclass
class ResolutionContext:
    inventaire: dict[str, Any]
    role_prefixes: tuple[str, ...] = field(default_factory=tuple)
    bilingual_attrs: dict[str, str] = field(default_factory=dict)
    entity_customer_keys: dict[str, set[str]] = field(default_factory=dict)
    entity_alias_tokens: dict[str, set[frozenset[str]]] = field(default_factory=dict)
    site_entity_slugs: dict[str, str] = field(default_factory=dict)


def build_resolution_context(
    inventaire: dict[str, Any],
    claims: list[dict[str, Any]],
) -> ResolutionContext:
    role_prefixes = tuple(inventaire.get("entities", {}).get("role_prefixes") or _BASE_ROLE_PREFIXES)
    bilingual = dict(inventaire.get("attributes", {}).get("bilingual_pairs") or {})
    entity_customer_keys: dict[str, set[str]] = defaultdict(set)
    entity_alias_tokens: dict[str, set[frozenset[str]]] = defaultdict(set)
    site_entity_slugs: dict[str, str] = {}

    for claim in claims:
        entity = str(claim.get("entity_id") or "")
        if not entity:
            continue
        text = claim_text(claim.get("value")) or str(claim.get("text") or "")
        customer = extract_customer_key(text)
        if customer:
            entity_customer_keys[entity].add(customer)
        entity_alias_tokens[entity].add(_token_set(entity))
        entity_alias_tokens[entity].add(_token_set(strip_role_prefix(entity, role_prefixes)))
        for alias in observed_form_aliases(label=entity, entity_id=entity, text=text):
            entity_alias_tokens[entity].add(_token_set(alias))
        site_match = re.search(r"—\s*([A-Za-z][A-Za-z0-9 ,'-]{1,40})\s+site\b", text, re.IGNORECASE)
        if site_match:
            site_slug = slug_id(site_match.group(1))
            if site_slug and len(site_slug.split("_")) <= 4:
                site_entity_slugs[entity] = site_slug

    return ResolutionContext(
        inventaire=inventaire,
        role_prefixes=role_prefixes,
        bilingual_attrs=bilingual,
        entity_customer_keys=dict(entity_customer_keys),
        entity_alias_tokens={k: set(v) for k, v in entity_alias_tokens.items()},
        site_entity_slugs=site_entity_slugs,
    )


def _entity_forms(entity_id: str, ctx: ResolutionContext) -> list[str]:
    forms = [normalize_entity_form(entity_id)]
    forms.append(normalize_entity_form(strip_role_prefix(entity_id, ctx.role_prefixes)))
    return [f for f in forms if f]


def entities_match(
    a: dict[str, Any],
    b: dict[str, Any],
    palier: str,
    ctx: ResolutionContext,
) -> bool:
    entity_a = str(a.get("entity_id") or "")
    entity_b = str(b.get("entity_id") or "")
    if not entity_a or not entity_b:
        return entity_a == entity_b
    if palier == "P0":
        return entity_a == entity_b

    forms_a = _entity_forms(entity_a, ctx)
    forms_b = _entity_forms(entity_b, ctx)

    if palier >= "P1" and any(x == y for x in forms_a for y in forms_b):
        return True

    if palier >= "P2":
        for left in forms_a:
            for right in forms_b:
                if _word_boundary_inclusion(left, right):
                    return True
        tokens_a = normalize_entity_tokens(entity_a, ctx.role_prefixes)
        tokens_b = normalize_entity_tokens(entity_b, ctx.role_prefixes)
        if tokens_a and tokens_b and tokens_a == tokens_b:
            return True

    if palier >= "P4":
        text_a = claim_text(a.get("value")) or str(a.get("text") or "")
        text_b = claim_text(b.get("value")) or str(b.get("text") or "")
        customer_a = extract_customer_key(text_a)
        customer_b = extract_customer_key(text_b)
        keys_a = set(ctx.entity_customer_keys.get(entity_a, set()))
        keys_b = set(ctx.entity_customer_keys.get(entity_b, set()))
        if customer_a:
            keys_a.add(customer_a)
        if customer_b:
            keys_b.add(customer_b)
        shared_customers = keys_a & keys_b
        if shared_customers:
            entity_slug_a = slug_id(strip_role_prefix(entity_a, ctx.role_prefixes))
            entity_slug_b = slug_id(strip_role_prefix(entity_b, ctx.role_prefixes))
            if _is_document_slug(entity_slug_a) and _is_document_slug(entity_slug_b):
                return True
            site_a = ctx.site_entity_slugs.get(entity_a)
            site_b = ctx.site_entity_slugs.get(entity_b)
            if site_a and site_b and site_a == site_b:
                return True

        aliases_a = set(ctx.entity_alias_tokens.get(entity_a, set()))
        aliases_b = set(ctx.entity_alias_tokens.get(entity_b, set()))
        for left in aliases_a:
            for right in aliases_b:
                if left and left == right and len(left) >= 2:
                    return True

    return False


def _resolve_attribute(attribute_id: str, qualifiers: dict[str, Any], palier: str, ctx: ResolutionContext) -> str:
    attr = attribute_id or ""
    if not attr:
        return attr
    if palier >= "P3":
        stripped = strip_year_period_suffix(attr, qualifiers)
        attr = ctx.bilingual_attrs.get(stripped, stripped)
        if attribute_id in ctx.bilingual_attrs:
            attr = ctx.bilingual_attrs[attribute_id]
    return normalize_entity_form(attr)


def attributes_match(
    a: dict[str, Any],
    b: dict[str, Any],
    palier: str,
    ctx: ResolutionContext,
) -> bool:
    quals_a = dict(a.get("qualifiers") or {})
    quals_b = dict(b.get("qualifiers") or {})
    attr_a = str(quals_a.get("attribute_id") or "")
    attr_b = str(quals_b.get("attribute_id") or "")
    if not attr_a or not attr_b:
        return attr_a == attr_b
    if palier == "P0":
        return attr_a == attr_b
    resolved_a = _resolve_attribute(attr_a, quals_a, palier, ctx)
    resolved_b = _resolve_attribute(attr_b, quals_b, palier, ctx)
    if resolved_a == resolved_b:
        if palier >= "P3" and (_YEAR_SUFFIX_RE.search(attr_a) or _YEAR_SUFFIX_RE.search(attr_b)):
            periods_a = {str(quals_a.get(key)) for key in _PERIOD_KEYS if quals_a.get(key)}
            periods_b = {str(quals_b.get(key)) for key in _PERIOD_KEYS if quals_b.get(key)}
            if periods_a and periods_b and periods_a != periods_b:
                return False
        return True
    if palier >= "P2" and _word_boundary_inclusion(resolved_a, resolved_b):
        return True
    if palier >= "P3":
        if _attr_canonical_tokens(resolved_a) != _attr_canonical_tokens(resolved_b):
            return False
        if _YEAR_SUFFIX_RE.search(attr_a) or _YEAR_SUFFIX_RE.search(attr_b):
            periods_a = {str(quals_a.get(key)) for key in _PERIOD_KEYS if quals_a.get(key)}
            periods_b = {str(quals_b.get(key)) for key in _PERIOD_KEYS if quals_b.get(key)}
            return not periods_a or not periods_b or periods_a == periods_b
        return True
    return False


def facts_same_subject_attribute(
    a: dict[str, Any],
    b: dict[str, Any],
    palier: str,
    ctx: ResolutionContext,
) -> bool:
    return entities_match(a, b, palier, ctx) and attributes_match(a, b, palier, ctx)


def read_claims_from_db(connection, *, vault_ref: str = VAULT_REF) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path, "
        "source_span "
        "FROM public.claims WHERE lifecycle_status = 'active' AND vault_ref = %s "
        "ORDER BY claim_id",
        (vault_ref,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["text"] = claim_text(item.get("value"))
        out.append(item)
    return out


def write_inventaire(connection) -> dict[str, Any]:
    export_runtime_env()
    claims = read_claims_from_db(connection)
    inventaire = build_inventaire_from_claims(claims)
    SORTIES_RESOLUTION.mkdir(parents=True, exist_ok=True)
    INVENTAIRE_PATH.write_text(json.dumps(inventaire, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return inventaire
