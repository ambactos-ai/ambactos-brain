# SPDX-License-Identifier: AGPL-3.0-only
"""Assertion ↔ evidence element states (tient / candidat / non)."""
from __future__ import annotations

import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from .config import CORPUS
from .fidelite import (
    _assertion_has_tabular_quote,
    _assertion_needle,
    _needle_row_tokens,
    claim_is_tabular,
)
from .normalize import claim_text, normalize_text, piece_basename

State = Literal["tient", "candidat", "non"]
ModePresence = Literal["verbatim", "valeur"]

_NBSP_RE = re.compile(r"[\u00a0\u202f]")
_RG_RE = re.compile(r"\b\d{2}/\d{5}\b")
_PIECE_REF_RE = re.compile(r"\bL\.\d{4}-\d+\b", re.IGNORECASE)
_ISO_DATE_IN_TEXT_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_SLASH_DATE_IN_TEXT_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
_FRENCH_DATE_RE = re.compile(
    r"\b(?:\d{1,2}|1er)\s+(?:janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|septembre|octobre|novembre|decembre|décembre)\s+\d{4}\b",
    re.IGNORECASE,
)
_NUMBER_SPAN_RE = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[\s\u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?(?!\d)"
)
_UPPER_IDENT_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}\b")
_UNIT_AFTER_RE = re.compile(
    r"^\s*(?:"
    r"%|€|"
    r"m\s*€|"
    r"millions?\s+d(?:e)?\s+euros?|"
    r"eur(?:o)?s?|"
    r"percent|"
    r"pour\s*cent|pourcent|"
    r"h(?:eures?)?|"
    r"jours?|days?"
    r")(?:\b|$)",
    re.IGNORECASE,
)
_LEXICAL_OVERLAP_MIN = 0.8

STOP_WORDS = frozenset(
    {
        "de",
        "du",
        "des",
        "le",
        "la",
        "les",
        "un",
        "une",
        "et",
        "en",
        "au",
        "aux",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "is",
        "are",
        "was",
        "were",
        "it",
        "its",
        "this",
        "that",
        "with",
        "from",
        "by",
        "at",
        "as",
        "or",
        "be",
        "been",
        "has",
        "have",
        "had",
    }
)


class PieceCache:
    """Cache piece reads and quote-line lookups (mandate: < 120 s holder table)."""

    def __init__(self, corpus: Path = CORPUS) -> None:
        self.corpus = corpus
        self._lines: dict[str, list[str]] = {}
        self._normalized: dict[str, list[str]] = {}
        self._quote_lines: dict[tuple[str, str], int | None] = {}
        self.read_count = 0

    def get_lines(self, piece: str) -> list[str] | None:
        if piece not in self._lines:
            path = self.corpus / piece
            if not path.is_file():
                return None
            self.read_count += 1
            raw = path.read_text(encoding="utf-8").splitlines()
            self._lines[piece] = raw
            self._normalized[piece] = [normalize_text(line) for line in raw]
        return self._lines[piece]

    def find_quote_line(self, piece: str, quote: str) -> int | None:
        needle = normalize_text(quote)
        if not needle:
            return None
        key = (piece, needle)
        if key not in self._quote_lines:
            lines = self.get_lines(piece)
            if lines is None:
                self._quote_lines[key] = None
            else:
                found = None
                for idx, norm in enumerate(self._normalized[piece], start=1):
                    if needle in norm:
                        found = idx
                        break
                self._quote_lines[key] = found
        return self._quote_lines[key]

    def excerpt(self, piece: str, quote: str, before: int = 3, after: int = 3) -> str:
        center = self.find_quote_line(piece, quote)
        lines = self.get_lines(piece)
        if center is None or lines is None:
            return ""
        start = max(1, center - before)
        end = min(len(lines), center + after)
        out: list[str] = []
        for num in range(start, end + 1):
            prefix = "→ " if num == center else "  "
            out.append(f"{prefix}{num:4d}  {lines[num - 1]}")
        return "\n".join(out)


def _claim_as_element(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "claim_hit",
        "id": str(claim.get("claim_id") or ""),
        "text": claim.get("text") or claim_text(claim.get("value")),
        "source": claim.get("source_file_path") or claim.get("piece") or "",
    }


def _claim_piece(claim: dict[str, Any]) -> str:
    return claim.get("piece") or piece_basename(claim.get("source_file_path") or "")


def _text_matches_needle(claim: dict[str, Any], needle: str) -> bool:
    if not needle:
        return False
    hay = claim.get("norm_text")
    if hay is None:
        hay = normalize_text(claim.get("text") or claim_text(claim.get("value")))
    if needle in hay:
        return True
    raw = claim.get("norm_raw_line")
    if raw is None:
        span = claim.get("source_span") or {}
        raw = normalize_text(span.get("raw_line") or "")
    return bool(raw and needle in raw)


def _normalize_for_match(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = _NBSP_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _parse_numeric_value(raw: str) -> float | None:
    token = _NBSP_RE.sub(" ", raw.strip())
    if not token:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        return None
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", token):
        return None
    if "/" in token:
        return None
    if _FRENCH_DATE_RE.search(token):
        return None
    cleaned = re.sub(r"[^\d.,\s\u00a0\u202f]", "", token)
    cleaned = _NBSP_RE.sub("", cleaned).replace(" ", "")
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        left, right = cleaned.split(",", 1)
        cleaned = f"{left}.{right}" if len(right) <= 2 else cleaned.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", cleaned):
        cleaned = cleaned.replace(".", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _numeric_values_in_text(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_SPAN_RE.finditer(text):
        parsed = _parse_numeric_value(match.group(0))
        if parsed is not None:
            values.append(parsed)
    return values


def _normalize_anchor_token(token: str) -> str | None:
    cleaned = _normalize_for_match(token)
    if not cleaned:
        return None
    if cleaned in {"%", "percent", "pourcent", "pour cent"}:
        return "percent"
    if cleaned in {"€", "eur", "euro", "euros"}:
        return "eur"
    if cleaned in {"m€", "m eur", "meur"}:
        return "million_eur"
    if cleaned.startswith("million") and "euro" in cleaned:
        return "million_eur"
    if cleaned in {"h", "heure", "heures", "hour", "hours"}:
        return "hours"
    if cleaned in {"jour", "jours", "day", "days"}:
        return "days"
    if len(cleaned) > 1:
        return f"word:{cleaned}"
    return None


def _anchors_compatible(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    if left == right:
        return True
    million = {"million_eur", "eur"}
    if left in million and right in million:
        return True
    return False


def _extract_number_anchor(text: str, start: int, end: int) -> str | None:
    after = text[end:]
    unit_match = _UNIT_AFTER_RE.match(after)
    if unit_match:
        return _normalize_anchor_token(unit_match.group(0).strip())
    before = text[:start]
    prev_tokens = re.findall(r"[A-Za-zÀ-ÿ%€]+", before)
    for token in reversed(prev_tokens):
        if _normalize_for_match(token) in STOP_WORDS:
            continue
        anchor = _normalize_anchor_token(token)
        if anchor is not None:
            return anchor
    return None


def _is_year_without_unit(value: float, anchor: str | None) -> bool:
    if anchor is not None and not anchor.startswith("word:"):
        return False
    if not value.is_integer():
        return False
    year = int(value)
    return 1900 <= year <= 2100


def _anchored_numbers_in_text(text: str) -> list[tuple[float, str | None]]:
    anchored: list[tuple[float, str | None]] = []
    for match in _NUMBER_SPAN_RE.finditer(text):
        parsed = _parse_numeric_value(match.group(0))
        if parsed is None:
            continue
        anchor = _extract_number_anchor(text, match.start(), match.end())
        if _is_year_without_unit(parsed, anchor):
            continue
        if all(
            not _number_values_equal(parsed, existing[0]) or existing[1] != anchor
            for existing in anchored
        ):
            anchored.append((parsed, anchor))
    return anchored


def _number_values_equal(left: float, right: float) -> bool:
    left_int = int(left) if left.is_integer() else None
    right_int = int(right) if right.is_integer() else None
    if left_int is not None and right_int is not None and left_int == right_int:
        return True
    return abs(left - right) < 0.001


def _anchored_number_present(target: float, anchor: str | None, text: str) -> bool:
    if anchor is None:
        return False
    for value, found_anchor in _anchored_numbers_in_text(text):
        if not _number_values_equal(target, value):
            continue
        if _anchors_compatible(anchor, found_anchor):
            return True
    return False


def _french_date_key(text: str) -> tuple[int, int, int] | None:
    match = _FRENCH_DATE_RE.search(text)
    if not match:
        return None
    day_raw, month_raw, year_raw = match.group(0).split()
    day = 1 if day_raw.lower() == "1er" else int(day_raw)
    month = {
        "janvier": 1,
        "fevrier": 2,
        "février": 2,
        "mars": 3,
        "avril": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "aout": 8,
        "août": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "decembre": 12,
        "décembre": 12,
    }[month_raw.lower()]
    return int(year_raw), month, day


def _slash_date_key(text: str) -> tuple[int, int, int] | None:
    match = _SLASH_DATE_IN_TEXT_RE.search(text)
    if not match:
        return None
    day, month, year = (int(part) for part in match.group(0).split("/"))
    return year, month, day


def _iso_date_key(text: str) -> tuple[int, int, int] | None:
    match = _ISO_DATE_IN_TEXT_RE.search(text)
    if not match:
        return None
    year, month, day = (int(part) for part in match.group(0).split("-"))
    return year, month, day


def _dates_in_text(text: str) -> list[tuple[int, int, int]]:
    keys: list[tuple[int, int, int]] = []
    for match in _FRENCH_DATE_RE.finditer(text):
        parsed = _french_date_key(match.group(0))
        if parsed:
            keys.append(parsed)
    for match in _SLASH_DATE_IN_TEXT_RE.finditer(text):
        parsed = _slash_date_key(match.group(0))
        if parsed:
            keys.append(parsed)
    for match in _ISO_DATE_IN_TEXT_RE.finditer(text):
        parsed = _iso_date_key(match.group(0))
        if parsed:
            keys.append(parsed)
    return keys


def _text_token_present(token: str, text: str) -> bool:
    needle = _normalize_for_match(token)
    if not needle:
        return False
    hay = _normalize_for_match(text)
    if needle in hay:
        return True
    compact_needle = re.sub(r"\s+", "", needle)
    compact_haystack = re.sub(r"\s+", "", hay)
    return compact_needle in compact_haystack


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for idx in range(start, min(end, len(chars))):
            chars[idx] = " "
    return "".join(chars)


def discriminants(texte: str) -> dict[str, list[Any]]:
    """Extract citation discriminants for value-mode presence."""
    raw = texte or ""
    textes: list[str] = []
    dates: list[tuple[int, int, int]] = []
    nombres: list[float] = []
    masked_spans: list[tuple[int, int]] = []

    for match in _RG_RE.finditer(raw):
        textes.append(match.group(0))
        masked_spans.append(match.span())

    for match in _PIECE_REF_RE.finditer(raw):
        textes.append(match.group(0))
        masked_spans.append(match.span())

    for match in _FRENCH_DATE_RE.finditer(raw):
        parsed = _french_date_key(match.group(0))
        if parsed and parsed not in dates:
            dates.append(parsed)
        masked_spans.append(match.span())

    for match in _SLASH_DATE_IN_TEXT_RE.finditer(raw):
        parsed = _slash_date_key(match.group(0))
        if parsed and parsed not in dates:
            dates.append(parsed)
        masked_spans.append(match.span())

    for match in _ISO_DATE_IN_TEXT_RE.finditer(raw):
        parsed = _iso_date_key(match.group(0))
        if parsed and parsed not in dates:
            dates.append(parsed)
        masked_spans.append(match.span())

    masked = _mask_spans(raw, masked_spans)
    for value, anchor in _anchored_numbers_in_text(masked):
        if all(
            not _number_values_equal(value, existing[0]) or existing[1] != anchor
            for existing in nombres
        ):
            nombres.append((value, anchor))

    for match in _UPPER_IDENT_RE.finditer(masked):
        token = match.group(0)
        if token not in textes:
            textes.append(token)

    return {"nombres": nombres, "dates": dates, "textes": textes}


def _has_discriminants(disc: dict[str, list[Any]]) -> bool:
    return bool(disc["nombres"] or disc["dates"] or disc["textes"])


def _content_words(text: str) -> list[str]:
    words: list[str] = []
    for raw in normalize_text(text).split():
        word = re.sub(r"[^\w]+", "", raw, flags=re.UNICODE)
        if len(word) > 1 and word not in STOP_WORDS:
            words.append(word)
    return words


def _significant_word_present(word: str, element_text: str) -> bool:
    return _text_token_present(word, element_text)


def _lexical_fallback_holds(citation: str, element_text: str) -> bool:
    cite_words = _content_words(citation)
    if not cite_words:
        return False
    present = sum(1 for word in cite_words if _significant_word_present(word, element_text))
    if present / len(cite_words) < _LEXICAL_OVERLAP_MIN:
        return False
    counts: dict[str, int] = defaultdict(int)
    for word in cite_words:
        counts[word] += 1
    for word, count in counts.items():
        if count == 1 and not _significant_word_present(word, element_text):
            return False
    return True


def _element_haystack(
    element: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
) -> str:
    kind = element.get("kind")
    if kind == "claim_hit":
        claim_id = str(element.get("id") or "")
        claim = claims_index.get(claim_id)
        if claim is not None:
            parts = [claim.get("text") or claim_text(claim.get("value"))]
            span = claim.get("source_span") or {}
            raw_line = span.get("raw_line") or ""
            if raw_line:
                parts.append(str(raw_line))
            return "\n".join(part for part in parts if part)
    return str(element.get("text") or "")


def _element_piece(
    element: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
) -> str:
    kind = element.get("kind")
    if kind == "claim_hit":
        claim_id = str(element.get("id") or "")
        claim = claims_index.get(claim_id)
        if claim is not None:
            return _claim_piece(claim)
    source = str(element.get("source") or "")
    return piece_basename(source) if source else ""


def _line_row_tokens(line: str) -> set[str]:
    raw = line.strip()
    if "|" in raw:
        return {normalize_text(part) for part in raw.split("|") if normalize_text(part)}
    if raw.count(",") >= 3:
        return {normalize_text(part) for part in raw.split(",") if normalize_text(part)}
    return set()


def _cell_is_numeric_token(token: str) -> bool:
    return _parse_numeric_value(token) is not None


def _tabular_anchor_tokens(row_tokens: set[str]) -> set[str]:
    return {token for token in row_tokens if not _cell_is_numeric_token(token) and len(token) > 1}


def _tabular_row_numbers(row_tokens: set[str]) -> list[float]:
    numbers: list[float] = []
    for token in row_tokens:
        parsed = _parse_numeric_value(token)
        if parsed is not None:
            numbers.append(parsed)
    return numbers


def _tabular_row_overlap(citation_tokens: set[str], line_tokens: set[str]) -> bool:
    anchors = _tabular_anchor_tokens(citation_tokens)
    if not anchors:
        return False
    overlap = anchors & line_tokens
    required = min(2, len(anchors))
    if len(overlap) >= required:
        return True
    return len(anchors) == 1 and bool(overlap)


def _tabular_numbers_present(citation_tokens: set[str], line_tokens: set[str]) -> bool:
    cite_numbers = _tabular_row_numbers(citation_tokens)
    if not cite_numbers:
        return True
    line_numbers = _tabular_row_numbers(line_tokens)
    return all(
        any(_number_values_equal(cite_number, line_number) for line_number in line_numbers)
        for cite_number in cite_numbers
    )


def _tabular_row_matches(citation_tokens: set[str], line: str) -> bool:
    line_tokens = _line_row_tokens(line)
    if not line_tokens:
        return False
    if not _tabular_row_overlap(citation_tokens, line_tokens):
        return False
    return _tabular_numbers_present(citation_tokens, line_tokens)


def _tabular_value_holds(citation: str, haystack: str, citation_tokens: set[str]) -> bool:
    for line in haystack.splitlines():
        if _tabular_row_matches(citation_tokens, line):
            return True
    if "\n" not in haystack and _tabular_row_matches(citation_tokens, haystack):
        return True
    return False


def _value_holds(
    assertion: dict[str, Any],
    element: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
) -> bool:
    piece_a = piece_basename(assertion.get("source") or "")
    element_piece = _element_piece(element, claims_index)
    if element_piece != piece_a:
        return False
    citation = str(assertion.get("quote") or assertion.get("content") or "")
    haystack = _element_haystack(element, claims_index)
    if not haystack.strip():
        return False
    if _assertion_has_tabular_quote(assertion):
        row_tokens = _needle_row_tokens(assertion)
        if not row_tokens:
            return False
        if not _tabular_value_holds(citation, haystack, row_tokens):
            return False
        disc = discriminants(citation)
        for date_key in disc["dates"]:
            if date_key not in _dates_in_text(haystack):
                return False
        for token in disc["textes"]:
            if not _text_token_present(token, haystack):
                return False
        return True
    disc = discriminants(citation)
    if not _has_discriminants(disc):
        return _lexical_fallback_holds(citation, haystack)
    for number, anchor in disc["nombres"]:
        if not _anchored_number_present(number, anchor, haystack):
            return False
    for date_key in disc["dates"]:
        if date_key not in _dates_in_text(haystack):
            return False
    for token in disc["textes"]:
        if not _text_token_present(token, haystack):
            return False
    return True


def _strict_text_holds(assertion: dict[str, Any], claim: dict[str, Any]) -> bool:
    piece_a = piece_basename(assertion.get("source") or "")
    if _claim_piece(claim) != piece_a:
        return False
    needle = _assertion_needle(assertion)
    return _text_matches_needle(claim, needle)


def _strict_text_holds_ignore_piece(assertion: dict[str, Any], claim: dict[str, Any]) -> bool:
    needle = _assertion_needle(assertion)
    return _text_matches_needle(claim, needle)


def _classify_pair_verbatim(
    assertion: dict[str, Any],
    element: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
) -> State:
    """Classify (assertion, CallFrame element) as tient / candidat / non."""
    piece_a = piece_basename(assertion.get("source") or "")
    needle = _assertion_needle(assertion)
    kind = element.get("kind")

    if kind == "claim_hit":
        claim_id = str(element.get("id") or "")
        claim = claims_index.get(claim_id)
        if claim is None:
            text = normalize_text(element.get("text") or "")
            if needle and needle in text and piece_a in str(element.get("source") or ""):
                return "tient"
            return "non"
        if _strict_text_holds(assertion, claim):
            return "tient"
        if not claim_is_tabular(claim) and _claim_piece(claim) == piece_a:
            return "candidat"
        return "non"

    text = normalize_text(element.get("text") or "")
    source = str(element.get("source") or "")
    if needle and needle in text and piece_a in source:
        return "tient"
    return "non"


def classify_pair(
    assertion: dict[str, Any],
    element: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
    *,
    mode_presence: str = "verbatim",
) -> State:
    if mode_presence == "verbatim":
        return _classify_pair_verbatim(assertion, element, claims_index)
    if _classify_pair_verbatim(assertion, element, claims_index) == "tient":
        return "tient"
    if _value_holds(assertion, element, claims_index):
        return "tient"
    return "non"


def present_predicate(mode: str):
    if mode == "plancher":
        return lambda state: state == "tient"
    return lambda state: state in ("tient", "candidat")


def assertion_present_in_evidence(
    assertion: dict[str, Any],
    evidence: list[dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    mode: str,
    *,
    mode_presence: str = "verbatim",
) -> tuple[bool, int | None]:
    accept = present_predicate(mode)
    for rank, item in enumerate(evidence, 1):
        if accept(classify_pair(assertion, item, claims_index, mode_presence=mode_presence)):
            return True, rank
    return False, None


def overlap_index(assertion: dict[str, Any], claim: dict[str, Any]) -> float:
    parts: list[str] = []
    parts.append(str(claim.get("entity_id") or ""))
    qualifiers = claim.get("qualifiers") or {}
    parts.append(str(qualifiers.get("attribute_id") or ""))
    parts.append(str(claim.get("value") or ""))
    for key, value in qualifiers.items():
        if key != "attribute_id":
            parts.append(str(value))
    claim_tokens = {
        w
        for part in parts
        for w in normalize_text(part).split()
        if len(w) > 1 and w not in STOP_WORDS and not re.fullmatch(r"\d+", w)
    }
    assertion_blob = normalize_text(
        f"{assertion.get('quote') or ''} {assertion.get('content') or ''}"
    )
    assertion_tokens = {
        w
        for w in assertion_blob.split()
        if len(w) > 1 and w not in STOP_WORDS and not re.fullmatch(r"\d+", w)
    }
    if not claim_tokens:
        return 0.0
    return round(len(claim_tokens & assertion_tokens) / len(claim_tokens), 4)


def format_claim_triplet(claim: dict[str, Any]) -> str:
    entity = str(claim.get("entity_id") or "")
    qualifiers = claim.get("qualifiers") or {}
    attribute = str(qualifiers.get("attribute_id") or claim.get("predicate") or "")
    value = claim_text(claim.get("value"))
    other_quals = {k: v for k, v in qualifiers.items() if k != "attribute_id"}
    base = f"{entity} · {attribute} : {value}"
    if other_quals:
        extras = ", ".join(f"{k}={v}" for k, v in sorted(other_quals.items()))
        return f"{base} ({extras})"
    return base


def _prepare_claims(claims: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    claims_index: dict[str, dict[str, Any]] = {}
    by_piece: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        enriched = dict(claim)
        enriched["piece"] = _claim_piece(claim)
        enriched["norm_text"] = normalize_text(claim.get("text") or claim_text(claim.get("value")))
        span = claim.get("source_span") or {}
        enriched["norm_raw_line"] = normalize_text(span.get("raw_line") or "")
        claims_index[str(claim["claim_id"])] = enriched
        by_piece[enriched["piece"]].append(enriched)
    return by_piece, claims_index


def build_holder_table(
    claims: list[dict[str, Any]],
    gold: dict[str, Any],
    *,
    cache: PieceCache | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], float, int]:
    _ = cache
    t0 = time.perf_counter()
    by_piece, claims_index = _prepare_claims(claims)
    piece_haystacks: dict[str, str] = {}
    for piece, piece_claims in by_piece.items():
        chunks: list[str] = []
        for claim in piece_claims:
            chunks.append(claim["norm_text"])
            if claim["norm_raw_line"]:
                chunks.append(claim["norm_raw_line"])
        piece_haystacks[piece] = "\n".join(chunks)

    table: dict[str, list[dict[str, Any]]] = {}
    tient_without_piece_rule = 0

    for assertion in gold["assertions"]:
        aid = assertion["id"]
        piece_a = piece_basename(assertion.get("source") or "")
        needle = _assertion_needle(assertion)
        holders: list[dict[str, Any]] = []
        if needle:
            for claim in by_piece.get(piece_a, []):
                element = _claim_as_element(claim)
                if classify_pair(assertion, element, claims_index) == "tient":
                    holders.append(
                        {
                            "claim_id": str(claim["claim_id"]),
                            "piece": claim["piece"],
                        }
                    )
            for other_piece, piece_claims in by_piece.items():
                if other_piece == piece_a:
                    continue
                if needle not in piece_haystacks.get(other_piece, ""):
                    continue
                for claim in piece_claims:
                    if _text_matches_needle(claim, needle):
                        tient_without_piece_rule += 1
        table[aid] = holders

    return table, time.perf_counter() - t0, tient_without_piece_rule


def model_claims_on_piece(claims: list[dict[str, Any]], piece: str) -> list[dict[str, Any]]:
    return [c for c in claims if (c.get("piece") or "") == piece and not claim_is_tabular(c)]


def collect_appui_states(
    assertion: dict[str, Any],
    evidence: list[dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    *,
    mode_presence: str = "verbatim",
) -> dict[str, list[dict[str, Any]]]:
    tient: list[dict[str, Any]] = []
    candidat: list[dict[str, Any]] = []
    for item in evidence:
        state = classify_pair(assertion, item, claims_index, mode_presence=mode_presence)
        row = {"evidence_id": item.get("id"), "kind": item.get("kind"), "state": state}
        if state == "tient":
            tient.append(row)
        elif state == "candidat":
            candidat.append(row)
    return {"tient": tient, "candidat": candidat}


def noise_count(
    evidence: list[dict[str, Any]],
    support_ids: list[str],
    *,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    mode: str,
    mode_presence: str = "verbatim",
) -> int:
    accept = present_predicate(mode)
    noise = 0
    for item in evidence:
        holds_any = False
        for aid in support_ids:
            assertion = assertion_index[aid]
            if accept(classify_pair(assertion, item, claims_index, mode_presence=mode_presence)):
                holds_any = True
                break
        if not holds_any:
            noise += 1
    return noise
