# SPDX-License-Identifier: AGPL-3.0-only
"""Porte 3 voisinage — expansion typée sur entités partagées et arêtes relation.about."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from .couche_brute_search import claim_to_evidence
from .fidelite import DATE_QUALIFIER_KEYS
from .moreau_or import normalize_verbatim
from .normalize import claim_text, piece_basename
from .porte_phrase import _phrases_du_passage_avec_fenetres
from .relecture import humanize_id, span_lines_label

VOISINAGE_QUOTA_V1 = 6
VOISINAGE_QUOTA_V2_TOTAL = 12
VOISINAGE_QUOTA_V2_EXTRA = VOISINAGE_QUOTA_V2_TOTAL - VOISINAGE_QUOTA_V1
VOISINAGE_FENETRE_PHRASES = 2
FORMES_PASSAGE_ENTIER = frozenset({"citation", "qualification", "chronologie", "delai"})

# Familles explicites (rapport mandat #135) : dates/échéances, montants, états.
FAMILLES_ATTRIBUTE: dict[str, tuple[str, ...]] = {
    "temporal": (
        "date",
        "delai",
        "echeance",
        "due",
        "periode",
        "period",
        "horaire",
        "heure",
        "fenetre",
        "jours",
        "retard",
        "convocation",
        "notification",
        "emission",
        "effet",
        "since",
        "valid_from",
        "valid_to",
        "month",
        "mois",
        "year",
        "deadline",
        "snapshot",
        "signature",
        "embauche",
        "licenciement",
        "entretien",
        "saisine",
        "remise",
        "transmission",
        "jour",
        "naissance",
        "depot",
        "ouverture",
        "revision",
        "service",
        "working",
        "collection_day",
        "issue",
    ),
    "monetary": (
        "montant",
        "amount",
        "salaire",
        "remuneration",
        "taux",
        "indemnite",
        "charge",
        "gbp",
        "net",
        "gross",
        "prime",
        "payment",
        "factur",
        "currency",
        "brut",
        "eur",
        "solde",
        "line_net",
        "net_a_payer",
        "salaire_base",
        "article_700",
        "dommages_interets",
        "heures",
        "kg",
        "tonnes",
        "visits",
        "collected",
        "masse",
        "capacite",
        "duree",
        "conges",
        "quantite",
        "quantity",
        "nombre",
        "count",
        "collections",
        "volume",
        "balance",
        "tax",
        "unit",
        "total",
        "mean",
        "raw",
        "weighed",
        "estimated",
        "containers",
        "capacity",
    ),
    "status": (
        "statut",
        "status",
        "etat",
        "condition",
        "sort",
        "qualification",
        "decision",
        "contestation",
        "exported",
        "regularise",
        "conteste",
        "deboutement",
        "rompe",
        "dispense",
        "tenue",
        "reading",
        "provisional",
        "mode",
        "motif",
        "type",
        "forme",
        "base_legale",
        "role",
        "qualite",
    ),
}

_FAMILLE_ORDER = tuple(FAMILLES_ATTRIBUTE.keys())

DATE_VALUE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
AMOUNT_VALUE_RE = re.compile(r"^-?\d+([.,]\d+)?$")

# Alias historique des tests / appelants.
FAMILLES_PREDICATS = FAMILLES_ATTRIBUTE


def familles_predicats_retenues() -> list[str]:
    return list(_FAMILLE_ORDER)


def _claim_from_evidence_item(item: dict[str, Any]) -> dict[str, Any] | None:
    if item.get("kind") != "claim_hit":
        return None
    payload = item.get("payload") or {}
    claim = payload.get("claim")
    if isinstance(claim, dict):
        return claim
    return None


def _attribute_id(claim: dict[str, Any]) -> str:
    return str((claim.get("qualifiers") or {}).get("attribute_id") or "")


def _normalized_value(claim: dict[str, Any]) -> str:
    return claim_text(claim.get("value")).strip()


def claim_valeur_forme(claim: dict[str, Any]) -> str | None:
    """Famille déduite de la forme de la valeur (date ou montant), jamais du prédicat."""
    value = _normalized_value(claim)
    if not value:
        return None
    if DATE_VALUE_RE.match(value):
        return "temporal"
    if AMOUNT_VALUE_RE.match(value):
        return "monetary"
    return None


def _famille_depuis_attribute_id(attribute_id: str) -> str | None:
    attr = attribute_id.lower()
    if not attr:
        return None
    tokens = {token for token in attr.split("_") if token}
    for famille in _FAMILLE_ORDER:
        for needle in FAMILLES_ATTRIBUTE[famille]:
            if needle in tokens or needle in attr:
                return famille
    return None


def claim_famille_predicat(claim: dict[str, Any]) -> str | None:
    """Retourne la famille (temporal, monetary, status) depuis attribute_id ou forme de valeur."""
    if claim.get("predicate") not in (None, "note.body"):
        return None
    famille = _famille_depuis_attribute_id(_attribute_id(claim))
    if famille is not None:
        return famille
    quals = claim.get("qualifiers") or {}
    if any(key in quals for key in DATE_QUALIFIER_KEYS):
        return "temporal"
    return claim_valeur_forme(claim)


def _line_number(claim: dict[str, Any]) -> int:
    span = claim.get("source_span") or {}
    for key in ("line", "line_start", "start_line", "line_number"):
        value = span.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def _line_label(claim: dict[str, Any]) -> str:
    label = span_lines_label(claim.get("source_span"))
    if label:
        return label
    line = _line_number(claim)
    return f"ligne {line}" if line else "ligne inconnue"


@dataclass(frozen=True)
class _FenetreLookup:
    phrase: str
    passage: str
    truncated: bool
    line_label: str


def _fenetre_tronquee(text: str) -> bool:
    return str(text or "").rstrip().endswith("…")


def _fenetre_autour_offset(
    content: str,
    line_offset: int,
    claim_value: Any,
    *,
    fenetre: int = VOISINAGE_FENETRE_PHRASES,
) -> tuple[str, bool]:
    lines = content.splitlines()
    if not lines:
        cleaned = " ".join(str(content or "").split())
        return cleaned, _fenetre_tronquee(cleaned)
    line_offset = max(0, min(line_offset, len(lines) - 1))
    needle = lines[line_offset].strip()
    norm_value = claim_text(claim_value).strip()
    units = _phrases_du_passage_avec_fenetres(content, fenetre=fenetre)
    if norm_value:
        for unit, _span in units:
            if norm_value in unit:
                return unit, _fenetre_tronquee(unit)
    if needle:
        for unit, _span in units:
            if needle in unit:
                return unit, _fenetre_tronquee(unit)
    for delta in range(len(lines)):
        for sign in (0, 1, -1):
            idx = line_offset + delta * sign
            if idx < 0 or idx >= len(lines):
                continue
            fragment = lines[idx].strip()
            if not fragment:
                continue
            for unit, _span in units:
                if fragment in unit:
                    return unit, _fenetre_tronquee(unit)
    if units:
        unit, _span = units[0]
        return unit, _fenetre_tronquee(unit)
    cleaned = " ".join(content.split())
    return cleaned, _fenetre_tronquee(cleaned)


def fetch_passages_with_metadata(connection, *, vault_ref: str) -> list[dict[str, Any]]:
    if vault_ref == "greenloop":
        clause = " AND (vault_ref = %s OR vault_ref IS NULL)"
        params = (vault_ref,)
    else:
        clause = " AND vault_ref = %s"
        params = (vault_ref,)
    rows = connection.execute(
        f"""
        SELECT id, source_ref, content, metadata
        FROM public.memory_chunks
        WHERE source_type = 'corpus'
          {clause}
        ORDER BY id
        """,
        params,
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        piece = piece_basename(str(row.get("source_ref") or meta.get("piece") or ""))
        line_start = int(meta.get("line_start") or 0)
        line_end = int(meta.get("line_end") or line_start)
        out.append(
            {
                "id": str(row["id"]),
                "piece": piece,
                "content": str(row.get("content") or ""),
                "line_start": line_start,
                "line_end": line_end,
            }
        )
    return out


class FenetreVoisinageIndex:
    """Index des fenêtres phrase (fenêtre=2) par pièce et ligne, depuis memory_chunks."""

    def __init__(self, passages: list[dict[str, Any]]) -> None:
        self._passages = list(passages)
        self._by_piece: dict[str, list[dict[str, Any]]] = {}
        for passage in self._passages:
            piece = str(passage.get("piece") or "")
            if piece:
                self._by_piece.setdefault(piece, []).append(passage)

    @classmethod
    def from_connection(cls, connection, *, vault_ref: str) -> FenetreVoisinageIndex:
        return cls(fetch_passages_with_metadata(connection, vault_ref=vault_ref))

    def _passage_pour_ligne(self, piece: str, line: int) -> dict[str, Any] | None:
        candidates = self._by_piece.get(piece_basename(piece), [])
        if not candidates:
            return None
        covering = [
            row
            for row in candidates
            if int(row.get("line_start") or 0) <= line <= int(row.get("line_end") or 0)
        ]
        if covering:
            return sorted(
                covering,
                key=lambda row: (
                    int(row.get("line_end") or 0) - int(row.get("line_start") or 0),
                    str(row.get("id") or ""),
                ),
            )[0]
        return sorted(candidates, key=lambda row: str(row.get("id") or ""))[0]

    def lookup(self, claim: dict[str, Any]) -> _FenetreLookup:
        piece = claim.get("piece") or piece_basename(claim.get("source_file_path") or "")
        line = _line_number(claim)
        passage = self._passage_pour_ligne(str(piece), line)
        line_label = _line_label(claim)
        if passage is None:
            return _FenetreLookup(
                phrase=claim_text(claim.get("value")),
                passage=claim_text(claim.get("value")),
                truncated=False,
                line_label=line_label,
            )
        line_start = int(passage.get("line_start") or 1)
        content = str(passage.get("content") or "")
        offset = max(0, line - line_start) if line else 0
        phrase, truncated = _fenetre_autour_offset(content, offset, claim.get("value"))
        return _FenetreLookup(
            phrase=phrase,
            passage=content.strip(),
            truncated=truncated,
            line_label=line_label,
        )


def _textes_porte2_phrases(evidence: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for item in evidence:
        if item.get("kind") != "phrase_hit":
            continue
        text = str(item.get("text") or "").strip()
        if text:
            out.add(normalize_verbatim(text))
    return out


def _phrase_voisin_normale(
    claim: dict[str, Any],
    *,
    index: FenetreVoisinageIndex,
    question_shape: str | None,
) -> str:
    lookup = index.lookup(claim)
    if question_shape in FORMES_PASSAGE_ENTIER or lookup.truncated:
        texte = lookup.passage
    else:
        texte = lookup.phrase
    return normalize_verbatim(texte)


def _texte_voisin_phrase(
    claim: dict[str, Any],
    *,
    index: FenetreVoisinageIndex,
    question_shape: str | None,
) -> tuple[str, str]:
    lookup = index.lookup(claim)
    if question_shape in FORMES_PASSAGE_ENTIER or lookup.truncated:
        return lookup.passage, lookup.line_label
    return lookup.phrase, lookup.line_label


def _date_sort_key(claim: dict[str, Any]) -> str:
    value = _normalized_value(claim)
    if DATE_VALUE_RE.match(value):
        return value
    return ""


def tri_claim_deterministe(claim: dict[str, Any]) -> tuple[Any, ...]:
    piece = claim.get("piece") or piece_basename(claim.get("source_file_path") or "")
    return (
        _date_sort_key(claim),
        piece,
        _line_number(claim),
        str(claim.get("claim_id") or ""),
    )


def _render_voisin_text(claim: dict[str, Any]) -> str:
    entity = humanize_id(claim.get("entity_id"))
    attribute = humanize_id(_attribute_id(claim) or claim.get("predicate"))
    value = claim_text(claim.get("value"))
    return f"{entity} {attribute} : {value}"


def _render_voisin_phrase_text(
    claim: dict[str, Any],
    *,
    phrase: str,
    line_label: str,
) -> str:
    entity = humanize_id(claim.get("entity_id"))
    attribute = humanize_id(_attribute_id(claim) or claim.get("predicate"))
    piece = claim.get("piece") or piece_basename(claim.get("source_file_path") or "")
    return f"{entity} —{attribute}→ · « {phrase} » ({piece}, {line_label})"


def _render_voisin_relation_text(
    claim: dict[str, Any],
    *,
    source_entity: str,
    edge_attribute: str,
    target_entity: str,
) -> str:
    entity = humanize_id(source_entity)
    target = humanize_id(target_entity)
    attribute = humanize_id(_attribute_id(claim))
    value = claim_text(claim.get("value"))
    piece = claim.get("piece") or piece_basename(claim.get("source_file_path") or "")
    edge = humanize_id(edge_attribute) if edge_attribute else edge_attribute
    return f"{entity} —{edge}→ {target} · {attribute} : {value} ({piece})"


def _render_voisin_relation_phrase_text(
    claim: dict[str, Any],
    *,
    source_entity: str,
    edge_attribute: str,
    target_entity: str,
    phrase: str,
    line_label: str,
) -> str:
    entity = humanize_id(source_entity)
    target = humanize_id(target_entity)
    piece = claim.get("piece") or piece_basename(claim.get("source_file_path") or "")
    edge = humanize_id(edge_attribute) if edge_attribute else edge_attribute
    return f"{entity} —{edge}→ {target} · « {phrase} » ({piece}, {line_label})"


def voisin_to_evidence(
    claim: dict[str, Any],
    *,
    rank: int,
    relation: dict[str, str] | None = None,
    phrase: str | None = None,
    line_label: str | None = None,
) -> dict[str, Any]:
    piece = claim.get("piece") or piece_basename(claim.get("source_file_path") or "")
    item = claim_to_evidence(claim, score=0.0, rank=rank)
    item["kind"] = "voisin_hit"
    if relation:
        if phrase is not None:
            item["text"] = _render_voisin_relation_phrase_text(
                claim,
                source_entity=relation["source_entity"],
                edge_attribute=relation["edge_attribute"],
                target_entity=relation["target_entity"],
                phrase=phrase,
                line_label=line_label or _line_label(claim),
            )
        else:
            item["text"] = _render_voisin_relation_text(
                claim,
                source_entity=relation["source_entity"],
                edge_attribute=relation["edge_attribute"],
                target_entity=relation["target_entity"],
            )
        voisinage = {
            "famille": claim_famille_predicat(claim),
            "arete": "relation.about",
            "source_entity": relation["source_entity"],
            "edge_attribute": relation["edge_attribute"],
            "target_entity": relation["target_entity"],
        }
    else:
        if phrase is not None:
            item["text"] = _render_voisin_phrase_text(
                claim,
                phrase=phrase,
                line_label=line_label or _line_label(claim),
            )
        else:
            item["text"] = _render_voisin_text(claim)
        voisinage = {"famille": claim_famille_predicat(claim), "arete": "entite_partagee"}
    if phrase is not None:
        voisinage["rendu"] = "phrase"
    item["source"] = piece
    item["snippet"] = item["text"][:280]
    payload = dict(item.get("payload") or {})
    payload["voisinage"] = voisinage
    item["payload"] = payload
    return item


def _ids_deja_presents(evidence: list[dict[str, Any]]) -> set[str]:
    present: set[str] = set()
    for item in evidence:
        kind = item.get("kind")
        if kind in {"claim_hit", "voisin_hit"} and item.get("id"):
            present.add(str(item["id"]))
            claim = _claim_from_evidence_item(item)
            if claim and claim.get("claim_id"):
                present.add(str(claim["claim_id"]))
    return present


def _claim_ids_porte1(evidence: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        if item.get("kind") != "claim_hit":
            continue
        claim = _claim_from_evidence_item(item)
        cid = str((claim or {}).get("claim_id") or item.get("id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        ordered.append(cid)
    return ordered


def _fetch_claims_by_ids(
    connection,
    claim_ids: list[str],
    *,
    vault_ref: str,
) -> list[dict[str, Any]]:
    if not claim_ids:
        return []
    rows = connection.execute(
        """
        SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path,
               source_span, extracted_by_skill, lifecycle_status, created_at
        FROM public.claims
        WHERE lifecycle_status = 'active'
          AND vault_ref = %s
          AND claim_id = ANY(%s)
        """,
        (vault_ref, claim_ids),
    ).fetchall()
    by_id = {str(_row_to_claim(dict(row))["claim_id"]): _row_to_claim(dict(row)) for row in rows}
    return [by_id[cid] for cid in claim_ids if cid in by_id]


def entites_porte1(
    evidence: list[dict[str, Any]],
    *,
    connection,
    vault_ref: str,
) -> list[str]:
    seed_claims = _fetch_claims_by_ids(connection, _claim_ids_porte1(evidence), vault_ref=vault_ref)
    seen: set[str] = set()
    ordered: list[str] = []
    for claim in seed_claims:
        entity = str(claim.get("entity_id") or "").strip()
        if not entity or entity in seen:
            continue
        seen.add(entity)
        ordered.append(entity)
    return ordered


def _row_to_claim(row: dict[str, Any]) -> dict[str, Any]:
    claim = dict(row)
    if claim.get("claim_id") is not None:
        claim["claim_id"] = str(claim["claim_id"])
    created = claim.get("created_at")
    if created is not None and not isinstance(created, str):
        claim["created_at"] = str(created)
    claim["text"] = claim_text(claim.get("value"))
    claim["piece"] = piece_basename(claim.get("source_file_path") or "")
    return claim


def _fetch_note_body_entities(
    connection,
    entity_ids: list[str],
    *,
    vault_ref: str,
) -> list[dict[str, Any]]:
    if not entity_ids:
        return []
    rows = connection.execute(
        """
        SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path,
               source_span, extracted_by_skill, lifecycle_status, created_at
        FROM public.claims
        WHERE lifecycle_status = 'active'
          AND vault_ref = %s
          AND predicate = 'note.body'
          AND entity_id = ANY(%s)
        """,
        (vault_ref, entity_ids),
    ).fetchall()
    return [_row_to_claim(dict(row)) for row in rows]


def _parse_relation_target(value: Any) -> str | None:
    text = str(value or "").strip()
    if text.startswith("note:"):
        target = text[5:].strip()
        return target or None
    return None


def _fetch_relations_about(
    connection,
    entity_ids: list[str],
    *,
    vault_ref: str,
) -> list[dict[str, Any]]:
    if not entity_ids:
        return []
    rows = connection.execute(
        """
        SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path,
               source_span, extracted_by_skill, lifecycle_status, created_at
        FROM public.claims
        WHERE lifecycle_status = 'active'
          AND vault_ref = %s
          AND predicate = 'relation.about'
          AND entity_id = ANY(%s)
        """,
        (vault_ref, entity_ids),
    ).fetchall()
    return [_row_to_claim(dict(row)) for row in rows]


def _candidats_filtres(
    claims: list[dict[str, Any]],
    *,
    exclude_ids: set[str],
    vault_ref: str,
    index: FenetreVoisinageIndex | None = None,
    question_shape: str | None = None,
    porte2_phrases: set[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_phrases: set[str] = set()
    for claim in claims:
        if claim.get("predicate") != "note.body":
            continue
        cid = str(claim.get("claim_id") or "")
        if not cid or cid in exclude_ids:
            continue
        if claim.get("vault_ref") not in (None, vault_ref):
            continue
        if claim_famille_predicat(claim) is None:
            continue
        if index is not None and porte2_phrases is not None:
            norm = _phrase_voisin_normale(claim, index=index, question_shape=question_shape)
            if norm in porte2_phrases or norm in seen_phrases:
                continue
            seen_phrases.add(norm)
        out.append(claim)
    out.sort(key=tri_claim_deterministe)
    return out


def _select_quota(candidates: list[dict[str, Any]], *, quota: int) -> list[dict[str, Any]]:
    return candidates[:quota]


def _candidats_relation_about(
    relations: list[dict[str, Any]],
    target_claims: list[dict[str, Any]],
    *,
    exclude_ids: set[str],
    vault_ref: str,
    index: FenetreVoisinageIndex | None = None,
    question_shape: str | None = None,
    porte2_phrases: set[str] | None = None,
) -> list[tuple[dict[str, Any], dict[str, str]]]:
    claims_by_entity: dict[str, list[dict[str, Any]]] = {}
    for claim in target_claims:
        entity = str(claim.get("entity_id") or "")
        if entity:
            claims_by_entity.setdefault(entity, []).append(claim)

    out: list[tuple[dict[str, Any], dict[str, str]]] = []
    seen_claim_ids: set[str] = set()
    seen_phrases: set[str] = set()
    relation_rows = sorted(
        relations,
        key=lambda row: (
            str(row.get("entity_id") or ""),
            _attribute_id(row),
            str(row.get("claim_id") or ""),
            str(row.get("value") or ""),
        ),
    )
    for relation in relation_rows:
        source_entity = str(relation.get("entity_id") or "")
        target_entity = _parse_relation_target(relation.get("value"))
        if not source_entity or not target_entity:
            continue
        edge_attribute = _attribute_id(relation)
        for claim in _candidats_filtres(
            claims_by_entity.get(target_entity, []),
            exclude_ids=exclude_ids,
            vault_ref=vault_ref,
            index=index,
            question_shape=question_shape,
            porte2_phrases=porte2_phrases,
        ):
            cid = str(claim.get("claim_id") or "")
            if cid in seen_claim_ids:
                continue
            if index is not None and porte2_phrases is not None:
                norm = _phrase_voisin_normale(claim, index=index, question_shape=question_shape)
                if norm in seen_phrases:
                    continue
                seen_phrases.add(norm)
            seen_claim_ids.add(cid)
            out.append(
                (
                    claim,
                    {
                        "source_entity": source_entity,
                        "edge_attribute": edge_attribute,
                        "target_entity": target_entity,
                    },
                )
            )
    out.sort(key=lambda pair: tri_claim_deterministe(pair[0]))
    return out


def table_attribute_ids(
    connection,
    *,
    vault_ref: str,
) -> list[dict[str, Any]]:
    """Inventaire des attribute_id note.body par dossier, avec famille retenue le cas échéant."""
    rows = connection.execute(
        """
        SELECT qualifiers->>'attribute_id' AS attribute_id, COUNT(*) AS count
        FROM public.claims
        WHERE lifecycle_status = 'active'
          AND vault_ref = %s
          AND predicate = 'note.body'
          AND qualifiers->>'attribute_id' IS NOT NULL
        GROUP BY 1
        ORDER BY count DESC, attribute_id
        """,
        (vault_ref,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        attribute_id = str(row["attribute_id"] or "")
        sample_claim = {
            "predicate": "note.body",
            "qualifiers": {"attribute_id": attribute_id},
            "value": "",
        }
        out.append(
            {
                "attribute_id": attribute_id,
                "count": int(row["count"]),
                "famille": claim_famille_predicat(sample_claim),
            }
        )
    return out


def compter_voisin_hits_par_question(
    questions: list[dict[str, Any]],
    *,
    fournisseur,
    connection,
    k: int,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    total = 0
    for question in questions:
        evidence = fournisseur.contexte(question, k=k, connection=connection)
        count = sum(1 for item in evidence if item.get("kind") == "voisin_hit")
        counts[str(question["id"])] = count
        total += count
    questions_avec = sum(1 for count in counts.values() if count > 0)
    return {
        "total": total,
        "questions_avec": questions_avec,
        "par_question": counts,
    }


class PorteVoisinageInerteError(RuntimeError):
    """Levée quand la porte ne rend aucun voisin_hit sur l'ensemble du dossier."""


def exiger_porte_voisinage_active(stats: dict[str, Any], *, dossier: str) -> None:
    if stats.get("total", 0) <= 0:
        raise PorteVoisinageInerteError(
            f"porte voisinage inerte sur {dossier} : 0 voisin_hit sur toutes les questions"
        )


def enrichir_porte_voisinage(
    evidence: list[dict[str, Any]],
    *,
    connection,
    vault_ref: str,
    arêtes: Literal[1, 2],
    rendu: Literal["atome", "phrase"] = "atome",
    question: dict[str, Any] | None = None,
    fenetre_index: FenetreVoisinageIndex | None = None,
) -> list[dict[str, Any]]:
    """Ajoute 6 (v1) ou 12 (v2) voisin_hit après les portes 1 et 2."""
    if arêtes not in (1, 2):
        raise ValueError(f"arêtes invalides : {arêtes}")
    exclude_ids = _ids_deja_presents(evidence)
    entities = entites_porte1(evidence, connection=connection, vault_ref=vault_ref)
    entity_claims = _fetch_note_body_entities(connection, entities, vault_ref=vault_ref)
    question_shape = str(question.get("shape") or "") if question else None
    index = fenetre_index
    porte2_phrases: set[str] | None = None
    if rendu == "phrase":
        index = index or FenetreVoisinageIndex.from_connection(connection, vault_ref=vault_ref)
        porte2_phrases = _textes_porte2_phrases(evidence)
    v1_candidates = _candidats_filtres(
        entity_claims,
        exclude_ids=exclude_ids,
        vault_ref=vault_ref,
        index=index,
        question_shape=question_shape,
        porte2_phrases=porte2_phrases,
    )
    v1_claims = _select_quota(v1_candidates, quota=VOISINAGE_QUOTA_V1)
    v1_ids = {str(c["claim_id"]) for c in v1_claims}
    exclude_after_v1 = exclude_ids | v1_ids

    out = list(evidence)
    base_rank = len(out)
    for offset, claim in enumerate(v1_claims, start=1):
        phrase = line_label = None
        if rendu == "phrase" and index is not None:
            phrase, line_label = _texte_voisin_phrase(
                claim,
                index=index,
                question_shape=question_shape,
            )
        out.append(
            voisin_to_evidence(
                claim,
                rank=base_rank + offset,
                phrase=phrase,
                line_label=line_label,
            )
        )

    if arêtes == 1:
        return out

    relations = _fetch_relations_about(connection, entities, vault_ref=vault_ref)
    target_entities = sorted(
        {
            target
            for relation in relations
            if (target := _parse_relation_target(relation.get("value")))
        }
    )
    target_claims = _fetch_note_body_entities(connection, target_entities, vault_ref=vault_ref)
    v2_pairs = _candidats_relation_about(
        relations,
        target_claims,
        exclude_ids=exclude_after_v1,
        vault_ref=vault_ref,
        index=index,
        question_shape=question_shape,
        porte2_phrases=porte2_phrases,
    )
    v2_pairs = v2_pairs[:VOISINAGE_QUOTA_V2_EXTRA]
    base_rank = len(out)
    for offset, (claim, relation) in enumerate(v2_pairs, start=1):
        phrase = line_label = None
        if rendu == "phrase" and index is not None:
            phrase, line_label = _texte_voisin_phrase(
                claim,
                index=index,
                question_shape=question_shape,
            )
        out.append(
            voisin_to_evidence(
                claim,
                rank=base_rank + offset,
                relation=relation,
                phrase=phrase,
                line_label=line_label,
            )
        )
    return out
