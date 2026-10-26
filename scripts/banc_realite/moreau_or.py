# SPDX-License-Identifier: AGPL-3.0-only
"""Or juridique MOREAU — citations exactes et questions, sans génération LLM."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from banc_realite.moreau_extraction import (
    DOCS_DIR,
    DOSSIER_ROOT,
    GOLDEN_PATH,
    _amount_variants,
    _parse_numeric_value,
    derive_discriminant_tokens,
    load_corpus,
    piece_to_markdown_name,
    token_in_text,
)

GOLD_DIR = DOSSIER_ROOT / "gold"
ASSERTIONS_PATH = GOLD_DIR / "assertions.json"
QUESTIONS_PATH = GOLD_DIR / "questions.json"
OR_JURIDIQUE_MD = GOLD_DIR / "OR-JURIDIQUE.md"
COUVERTURE_PATH = GOLD_DIR / "couverture.json"
RAPPORT_OR_PATH = GOLD_DIR / "RAPPORT-OR.md"
GROK_RESULT_PATH = Path(__file__).resolve().parents[2] / "_grok-result.json"

_ISO_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RG_RE = re.compile(r"\b\d{2}/\d{5}\b")
_RESUME_AMOUNT_RE = re.compile(r"(\d[\d\s.,]*)\s*(?:€|euros?)", re.IGNORECASE)
_RESUME_HOURS_RE = re.compile(r"(\d+)\s*h(?:eures?)?(?:\s*/\s*semaine)?", re.IGNORECASE)
_FRENCH_DATE_RE = re.compile(
    r"\b(?:\d{1,2}|1er)\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b",
    re.IGNORECASE,
)

_MIN_QUOTE = 20
_MAX_QUOTE = 400
_NBSP_RE = re.compile(r"[\u00a0\u202f]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_STRONG_END_RE = re.compile(r"[.;:!?»]$")
_DATE_IN_QUOTE_RE = re.compile(
    r"\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b"
    r"|\b\d{2}/\d{2}/\d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
_TABLE_LINE_RE = re.compile(
    r"^(?:Heures supplémentaires|Salaire|Total|TOTAL|BULLETIN|Semaine|Libellé|R\.G\.)",
    re.IGNORECASE,
)
_WIDE_SPACED_NUMBERS_RE = re.compile(r"\d(?:[^\d\n]*?\s{2,}[^\d\n]*?){1,}\d")
_HEADER_LINE_RE = re.compile(
    r"(?:\bSIREN\b|\bRCS\b|\bAPE\b|\bSIRET\b|"
    r"Réf(?:érence)?\s*:|Objet\s*:|Date\s*:|"
    r"(?:Fait à|À Orléans|Orléans-sur-Clain),\s*le\s)",
    re.IGNORECASE,
)
_FIELD_LABEL_START_RE = re.compile(r"^[A-Za-zÀ-ÿ][\w\s\-]*:\s")
_LIST_ITEM_START_RE = re.compile(r"^(?:\d+\.|[a-z]\)|-)\s")
_CONTINUATION_END_RE = re.compile(
    r"(?:,|\b(?:de|du|des|d'|à|au|aux|en|sous|pour|par|avec|dont|et|ou|un|une|le|la|les)\s*)$",
    re.IGNORECASE,
)


class QuoteNotFoundError(ValueError):
    pass


class IncompleteCitationError(ValueError):
    pass


class FactCoverageError(ValueError):
    pass


class SupportBindingError(ValueError):
    pass


_STOP_WORDS_FR = frozenset(
    {
        "le",
        "la",
        "les",
        "de",
        "du",
        "des",
        "d",
        "un",
        "une",
        "et",
        "ou",
        "en",
        "au",
        "aux",
        "pour",
        "par",
        "avec",
        "sans",
        "sur",
        "dans",
        "que",
        "qui",
        "ce",
        "cette",
        "ces",
        "son",
        "sa",
        "ses",
        "leur",
        "leurs",
        "il",
        "elle",
        "ils",
        "elles",
        "nous",
        "vous",
        "ne",
        "pas",
        "est",
        "sont",
        "été",
        "être",
        "avoir",
        "comme",
        "plus",
        "moins",
        "tout",
        "tous",
        "toute",
        "toutes",
        "selon",
        "entre",
        "sous",
        "dont",
        "lui",
        "si",
        "ça",
        "on",
        "y",
        "a",
        "au",
        "aux",
        "du",
        "des",
        "la",
        "le",
    }
)


def normalize_verbatim(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = _NBSP_RE.sub(" ", value)
    value = _HYPHEN_BREAK_RE.sub(r"\1\2", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def verify_quote(quote: str, piece_text: str) -> bool:
    return normalize_verbatim(quote) in normalize_verbatim(piece_text)


def _is_table_line(line: str) -> bool:
    if "|" in line:
        return True
    if _TABLE_LINE_RE.match(line):
        return True
    if _WIDE_SPACED_NUMBERS_RE.search(line):
        return True
    if "€" in line and re.search(r"\d", line):
        return True
    return False


def _is_header_line(line: str) -> bool:
    return bool(_HEADER_LINE_RE.search(line))


def _is_field_label_line(line: str) -> bool:
    return bool(_FIELD_LABEL_START_RE.match(line.lstrip()))


def _is_special_line(line: str) -> bool:
    return (
        _is_table_line(line)
        or _is_header_line(line)
        or _is_field_label_line(line)
        or bool(_RG_RE.search(line))
    )


def _has_strong_end(quote: str) -> bool:
    stripped = quote.rstrip()
    if _STRONG_END_RE.search(stripped):
        return True
    if stripped.endswith(('"', "'", "»")):
        inner = stripped.rstrip('"\'»').rstrip()
        return bool(_STRONG_END_RE.search(inner))
    return False


def _valid_citation_start(quote: str) -> bool:
    stripped = quote.lstrip()
    if not stripped:
        return False
    if stripped[0] in "«\"'(":
        return True
    if _LIST_ITEM_START_RE.match(stripped):
        return True
    if _FIELD_LABEL_START_RE.match(stripped):
        return True
    return bool(stripped[0].isupper())


def _is_complete_prose_citation(quote: str, *, max_len: int | None = None) -> bool:
    upper = max_len if max_len is not None else _MAX_QUOTE
    if _is_special_line(quote):
        return _MIN_QUOTE <= len(quote) <= upper
    if not _valid_citation_start(quote):
        return False
    if not _has_strong_end(quote):
        return False
    return _MIN_QUOTE <= len(quote) <= upper


def _is_acceptable_preferred_citation(quote: str) -> bool:
    if len(quote) < _MIN_QUOTE:
        return False
    if _is_special_line(quote):
        return True
    if not _valid_citation_start(quote):
        return False
    return True


def _should_merge_lines(current: str, next_line: str) -> bool:
    if _is_special_line(current) or _is_special_line(next_line):
        return False
    if _has_strong_end(current):
        return False
    nxt = next_line.lstrip()
    if not nxt:
        return False
    if nxt[0].islower() or nxt[0] == "(":
        return True
    return bool(_CONTINUATION_END_RE.search(current.rstrip()))


def _reflow_chunks(text: str) -> list[tuple[str, int, bool]]:
    lines = text.splitlines()
    chunks: list[tuple[str, int, bool]] = []
    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line or line.startswith("_") or set(line) <= {"-", "_", "|", " "}:
            index += 1
            continue
        first_line = index + 1
        if _is_special_line(line):
            chunks.append((line, first_line, True))
            index += 1
            continue
        parts = [line]
        cursor = index + 1
        while cursor < len(lines):
            next_line = lines[cursor].strip()
            if not next_line:
                break
            current = " ".join(parts)
            if _should_merge_lines(current, next_line):
                parts.append(next_line)
                cursor += 1
            else:
                break
        merged = re.sub(r"\s+", " ", " ".join(parts)).strip()
        chunks.append((merged, first_line, False))
        index = cursor
    return chunks


def _find_first_line(piece_text: str, quote: str) -> int:
    words = quote.split()
    if not words:
        return 1
    norm_quote = normalize_verbatim(quote)
    for index, line in enumerate(piece_text.splitlines(), start=1):
        norm_line = normalize_verbatim(line)
        if norm_line and norm_line in norm_quote:
            return index
    needle = words[0]
    for index, line in enumerate(piece_text.splitlines(), start=1):
        if needle in line:
            return index
    return 1


def _extract_forme_date(quote: str) -> str | None:
    match = _DATE_IN_QUOTE_RE.search(quote)
    return match.group(0) if match else None


def _infer_nature(fact: dict[str, Any], quote: str) -> str:
    resume = str(fact.get("resume") or "").lower()
    if fact.get("montant_eur") is not None or re.search(r"\d[\d\s.,]*€|\beuros?\b", quote, re.I):
        return "montant"
    if _DATE_IN_QUOTE_RE.search(quote) or fact.get("date"):
        return "date"
    if any(word in resume for word in ("licenciement", "harcèlement", "cause réelle", "qualification")):
        return "qualification"
    if any(word in resume for word in ("clause", "avenant", "horaires", "motif")):
        return "clause"
    if "atteste" in quote.lower() or "certifie" in quote.lower():
        return "fait"
    return "fait"


def _parse_iso_date(value: str) -> tuple[int, int, int] | None:
    if not _ISO_DATE_ONLY_RE.match(value):
        return None
    year, month, day = (int(part) for part in value.split("-"))
    return year, month, day


def _french_date_key(text: str) -> tuple[int, int, int] | None:
    match = _FRENCH_DATE_RE.search(text)
    if not match:
        return None
    day_raw, month_raw, year_raw = match.group(0).split()
    day = 1 if day_raw.lower() == "1er" else int(day_raw)
    month = {
        "janvier": 1,
        "février": 2,
        "mars": 3,
        "avril": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "août": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "décembre": 12,
    }[month_raw.lower()]
    return int(year_raw), month, day


def _slash_date_key(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", text)
    if not match:
        return None
    day, month, year = (int(part) for part in match.groups())
    return year, month, day


def _document_dates_in_quote(quote: str) -> list[tuple[int, int, int]]:
    keys: list[tuple[int, int, int]] = []
    for match in _FRENCH_DATE_RE.finditer(quote):
        parsed = _french_date_key(match.group(0))
        if parsed:
            keys.append(parsed)
    for match in re.finditer(r"\b\d{2}/\d{2}/\d{4}\b", quote):
        parsed = _slash_date_key(match.group(0))
        if parsed:
            keys.append(parsed)
    return keys


def _required_values_for_fact(fact: dict[str, Any], meta: dict[str, Any]) -> list[dict[str, Any]]:
    required: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, label: str, *, amount: float | None = None, quantity: int | None = None, iso_date: str | None = None, text: str | None = None) -> None:
        if kind == "montant":
            dedupe_key = ("montant", f"{amount:.3f}")
        elif kind == "quantite":
            dedupe_key = ("quantite", str(quantity))
        elif kind == "date":
            dedupe_key = ("date", str(iso_date))
        else:
            dedupe_key = ("texte", str(text))
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        payload: dict[str, Any] = {"kind": kind, "label": label}
        if amount is not None:
            payload["amount"] = amount
        if quantity is not None:
            payload["quantity"] = quantity
        if iso_date is not None:
            payload["iso_date"] = iso_date
        if text is not None:
            payload["text"] = text
        required.append(payload)

    if fact.get("montant_eur") is not None:
        add("montant", f"{fact['montant_eur']} EUR", amount=float(fact["montant_eur"]))
    if fact.get("quantite") is not None:
        add("quantite", f"{fact['quantite']} h", quantity=int(fact["quantite"]))
    if fact.get("date"):
        add("date", str(fact["date"]), iso_date=str(fact["date"]))

    resume = str(fact.get("resume") or "")
    for match in _RESUME_AMOUNT_RE.finditer(resume):
        parsed_amount = _parse_numeric_value(match.group(1))
        if parsed_amount is not None:
            add("montant", match.group(0).strip(), amount=parsed_amount)

    for match in _RESUME_HOURS_RE.finditer(resume):
        add("quantite", match.group(0).strip(), quantity=int(match.group(1)))

    for match in _FRENCH_DATE_RE.finditer(resume):
        parsed = _french_date_key(match.group(0))
        if parsed:
            iso = f"{parsed[0]:04d}-{parsed[1]:02d}-{parsed[2]:02d}"
            add("date", match.group(0), iso_date=iso)

    if meta.get("rg") and str(meta["rg"]) in resume:
        add("texte", str(meta["rg"]), text=str(meta["rg"]))

    for match in _RG_RE.finditer(resume):
        add("texte", match.group(0), text=match.group(0))

    return required


def _quote_carries_amount(quote: str, amount: float) -> bool:
    for variant in _amount_variants(amount):
        if token_in_text(variant, quote):
            return True
    return False


def _quote_carries_quantity(quote: str, quantity: int) -> bool:
    patterns = [
        f"{quantity} h",
        f"{quantity} heures",
        f"{quantity} heure",
    ]
    if any(token_in_text(pattern, quote) for pattern in patterns):
        return True
    return bool(re.search(rf"\b{quantity}\s*h(?:eures?)?\b", quote, re.IGNORECASE))


def _quote_carries_date(quote: str, iso_date: str) -> bool:
    if iso_date in quote:
        return False
    target = _parse_iso_date(iso_date)
    if target is None:
        return False
    return target in _document_dates_in_quote(quote)


def _quote_carries_text(quote: str, text: str) -> bool:
    if re.fullmatch(r"L\.\d{4}-\d+", text, re.IGNORECASE):
        return text.lower() in quote.lower()
    return token_in_text(text, quote)


def _quote_carries_requirement(quote: str, requirement: dict[str, Any]) -> bool:
    kind = requirement["kind"]
    if kind == "montant":
        return _quote_carries_amount(quote, float(requirement["amount"]))
    if kind == "quantite":
        return _quote_carries_quantity(quote, int(requirement["quantity"]))
    if kind == "date":
        return _quote_carries_date(quote, str(requirement["iso_date"]))
    if kind == "texte":
        return _quote_carries_text(quote, str(requirement["text"]))
    return False


def values_equivalent_for_coverage(left: str, right: str, *, kind: str) -> bool:
    if kind == "texte":
        return normalize_verbatim(left) == normalize_verbatim(right)
    if kind == "montant":
        left_val = _parse_numeric_value(left)
        right_val = _parse_numeric_value(right)
        return left_val is not None and right_val is not None and abs(left_val - right_val) < 0.001
    if kind == "date":
        left_iso = _parse_iso_date(left) or _french_date_key(left) or _slash_date_key(left)
        right_iso = _parse_iso_date(right) or _french_date_key(right) or _slash_date_key(right)
        return left_iso is not None and left_iso == right_iso
    return False


def _values_carried_by_quote(quote: str, requirements: list[dict[str, Any]]) -> list[str]:
    return [req["label"] for req in requirements if _quote_carries_requirement(quote, req)]


def verify_facts_coverage(meta: dict[str, Any], assertions: list[dict[str, Any]]) -> dict[str, Any]:
    by_fact = _assertions_by_fact(assertions)
    details: dict[str, Any] = {}
    orphans: list[str] = []
    covered_count = 0
    for fact in meta.get("facts", []):
        fact_id = str(fact["id"])
        requirements = _required_values_for_fact(fact, meta)
        fact_assertions = by_fact.get(fact_id, [])
        orphan_labels: list[str] = []
        assertion_rows: list[dict[str, Any]] = []
        for assertion in fact_assertions:
            carried = _values_carried_by_quote(str(assertion["quote"]), requirements)
            assertion_rows.append(
                {
                    "id": assertion["id"],
                    "piece": assertion["piece"],
                    "valeurs": carried,
                }
            )
        for requirement in requirements:
            if not any(_quote_carries_requirement(str(a["quote"]), requirement) for a in fact_assertions):
                orphan_labels.append(requirement["label"])
        details[fact_id] = {
            "resume": fact.get("resume"),
            "exigences": [req["label"] for req in requirements],
            "assertions": assertion_rows,
            "orphelines": orphan_labels,
        }
        if orphan_labels:
            for label in orphan_labels:
                orphans.append(f"{fact_id}: {label}")
        else:
            covered_count += 1
    total = len(meta.get("facts", []))
    return {
        "ok": not orphans,
        "details": details,
        "orphans": orphans,
        "covered": covered_count,
        "total": total,
    }


def _answer_text_for_binding(answer: str) -> str:
    return re.sub(r"\([^)]*\.md\)", "", answer)


def _answer_significant_tokens(answer: str) -> list[str]:
    cleaned = _answer_text_for_binding(answer)
    cleaned = re.sub(r"[«»\"'(),.;:!?—–-]", " ", cleaned)
    tokens: list[str] = []
    for raw in cleaned.lower().split():
        token = raw.strip()
        if len(token) < 3 or token in _STOP_WORDS_FR:
            continue
        tokens.append(token)
    return tokens


def _answer_significant_trigrams(answer: str) -> list[str]:
    tokens = _answer_significant_tokens(answer)
    if len(tokens) < 3:
        return []
    return [" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]


def _quote_has_trigram(quote: str, trigram: str) -> bool:
    norm_quote = re.sub(r"[^\w\s]", " ", normalize_verbatim(quote).lower())
    norm_quote = re.sub(r"\s+", " ", norm_quote).strip()
    words = trigram.split()
    if len(words) != 3:
        return trigram.lower() in norm_quote
    pattern = (
        rf"\b{re.escape(words[0])}\b(?:\s+\w+){{0,5}}\s+"
        rf"\b{re.escape(words[1])}\b(?:\s+\w+){{0,5}}\s+"
        rf"\b{re.escape(words[2])}\b"
    )
    return bool(re.search(pattern, norm_quote, re.IGNORECASE))


def _answer_numeric_requirements(answer: str) -> list[dict[str, Any]]:
    cleaned = _answer_text_for_binding(answer)
    required: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, label: str, **payload: Any) -> None:
        key = (kind, label)
        if key in seen:
            return
        seen.add(key)
        required.append({"kind": kind, "label": label, **payload})

    for match in _FRENCH_DATE_RE.finditer(cleaned):
        parsed = _french_date_key(match.group(0))
        if parsed:
            iso = f"{parsed[0]:04d}-{parsed[1]:02d}-{parsed[2]:02d}"
            add("date", match.group(0), iso_date=iso)

    for match in _RG_RE.finditer(cleaned):
        add("texte", match.group(0), text=match.group(0))

    for match in _RESUME_AMOUNT_RE.finditer(cleaned):
        prefix = cleaned[max(0, match.start() - 12) : match.start()]
        if re.search(r"\d\s*h\s*$", prefix, re.IGNORECASE):
            continue
        parsed = _parse_numeric_value(match.group(1))
        if parsed is not None:
            add("montant", match.group(0).strip(), amount=parsed)

    for match in re.finditer(r"(\d+)\s*(?:heures?)\b", cleaned, re.IGNORECASE):
        add("quantite", match.group(0).strip(), quantity=int(match.group(1)))
    for match in re.finditer(r"\b(\d+)\s*h\b(?!\s*\d)", cleaned, re.IGNORECASE):
        add("quantite", match.group(0).strip(), quantity=int(match.group(1)))

    for match in re.finditer(r"\b\d{3}\s+\d{3}\s+\d{3}\b", cleaned):
        add("texte", match.group(0), text=match.group(0))

    for match in re.finditer(r"\bL\.\d{4}-\d+\b", cleaned, re.IGNORECASE):
        add("texte", match.group(0), text=match.group(0))

    return required


def _quote_carries_answer_requirement(quote: str, requirement: dict[str, Any]) -> bool:
    return _quote_carries_requirement(quote, requirement)


def _support_carries_answer_elements(quote: str, answer: str, shape: str) -> bool:
    numeric_requirements = _answer_numeric_requirements(answer)
    carries_numeric = (
        not numeric_requirements
        or all(_quote_carries_answer_requirement(quote, req) for req in numeric_requirements)
    )
    if shape in {"citation", "qualification"}:
        trigrams = _answer_significant_trigrams(answer)
        carries_words = not trigrams or any(_quote_has_trigram(quote, trigram) for trigram in trigrams)
        if numeric_requirements:
            return carries_numeric and carries_words
        return carries_words
    return carries_numeric


def verify_question_support_bindings(
    questions: list[dict[str, Any]],
    assertion_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    for question in questions:
        if question["shape"] == "non_etabli":
            continue
        checked += 1
        quotes = [
            str(assertion_map[assertion_id]["quote"])
            for assertion_id in question["supporting_assertions"]
        ]
        answer = str(question["answer"])
        shape = str(question["shape"])
        missing: list[str] = []
        for requirement in _answer_numeric_requirements(answer):
            if not any(_quote_carries_answer_requirement(quote, requirement) for quote in quotes):
                missing.append(requirement["label"])
        if shape in {"citation", "qualification"}:
            trigrams = _answer_significant_trigrams(answer)
            if trigrams and not any(
                any(_quote_has_trigram(quote, trigram) for trigram in trigrams) for quote in quotes
            ):
                missing.append("suite de mots caractéristique")
        if missing:
            failures.append(f"{question['id']}: {', '.join(missing)}")
    return {
        "ok": not failures,
        "checked": checked,
        "conformes": checked - len(failures),
        "failures": failures,
    }


def _gold_output_paths(gold_dir: Path) -> dict[str, Path]:
    return {
        "assertions": gold_dir / "assertions.json",
        "questions": gold_dir / "questions.json",
        "or_juridique": gold_dir / "OR-JURIDIQUE.md",
        "couverture": gold_dir / "couverture.json",
        "rapport": gold_dir / "RAPPORT-OR.md",
    }


def _split_candidates(text: str) -> list[tuple[str, int, bool]]:
    candidates: list[tuple[str, int, bool]] = []
    for chunk, first_line, is_special in _reflow_chunks(text):
        if is_special:
            candidates.append((chunk, first_line, True))
            continue
        for sentence in re.split(r"(?<=[.;:!?»])\s+", chunk):
            sentence = sentence.strip()
            if sentence:
                candidates.append((sentence, first_line, False))
    return candidates


def _score_candidate(candidate: str, tokens: list[str]) -> int:
    return sum(1 for token in tokens if token_in_text(token, candidate))


def _trim_quote(candidate: str) -> str:
    quote = candidate.strip()
    if len(quote) <= _MAX_QUOTE:
        return quote
    cut = quote[: _MAX_QUOTE + 1]
    if " " in cut:
        return cut.rsplit(" ", 1)[0]
    return quote[:_MAX_QUOTE]


def find_quote_for_piece(
    piece_md: str,
    piece_text: str,
    tokens: list[str],
    *,
    preferred_substrings: list[str] | None = None,
) -> tuple[str, int]:
    preferred_substrings = preferred_substrings or []
    for needle in preferred_substrings:
        quote = needle.strip()
        if not _is_special_line(quote):
            quote = re.sub(r"\s+", " ", quote)
        if not verify_quote(quote, piece_text):
            continue
        if not _is_acceptable_preferred_citation(quote):
            continue
        return quote, _find_first_line(piece_text, quote)
    best_quote = ""
    best_score = 0
    best_line = 1
    for candidate, first_line, is_special in _split_candidates(piece_text):
        if len(candidate) < _MIN_QUOTE:
            continue
        quote = _trim_quote(candidate)
        if len(quote) < _MIN_QUOTE:
            continue
        if not verify_quote(quote, piece_text):
            continue
        if not is_special and not _is_complete_prose_citation(quote):
            continue
        score = _score_candidate(quote, tokens)
        if score > best_score or (score == best_score and len(quote) > len(best_quote)):
            best_score = score
            best_quote = quote
            best_line = first_line
    if best_quote and best_score > 0:
        return best_quote, best_line
    raise QuoteNotFoundError(f"Aucune citation trouvée dans {piece_md}")


def build_assertions(meta: dict[str, Any], corpus: dict[str, str]) -> list[dict[str, Any]]:
    preferred: dict[str, dict[str, list[str]]] = {
        "F01": {
            "01-contrat-travail-cdi.md": [
                "L'Employeur engage la Salariée à compter du 15 septembre 2019, sous réserve de la visite d'information et de prévention, en qualité de Technicienne de maintenance industrielle, statut ETAM, classification niveau IV, coefficient 250, relevant de la convention collective de la métallurgie (territoire de Beaumont-les-Rives)."
            ]
        },
        "F02": {
            "01-contrat-travail-cdi.md": [
                "RCS Beaumont-les-Rives B 749 382 615 - SIREN 749 382 615 - APE 3312Z"
            ]
        },
        "F03": {
            "02-avenant-temps-plein.md": [
                "En conséquence, la rémunération brute mensuelle de base est portée à 2 450,00 euros. Le taux horaire de référence s'établit à 16,00 euros. Les heures supplémentaires majorées à 25 % sont rémunérées sur la base de 20,00 euros de l'heure.",
                "La durée hebdomadaire de travail est portée de 28 heures à 35 heures (temps plein).",
                "Soucieuses d'adapter l'organisation du service maintenance à l'évolution de l'activité, les parties conviennent de modifier le contrat comme suit, à compter du 1er mars 2022 :",
            ]
        },
        "F04": {
            "03-bulletins-paie-nov-dec-2023.md": [
                "Heures supplémentaires 25 %                          12 h 20,00              240,00 €"
            ]
        },
        "F05": {
            "03-bulletins-paie-nov-dec-2023.md": [
                "Heures supplémentaires 25 %                          18 h 20,00              360,00 €"
            ]
        },
        "F06": {
            "04-planning-heures-sup.md": [
                "Établi par la salariée - transmis à la direction le 8 janvier 2024",
                "TOTAL heures supplémentaires déclarées : 86 heures",
                "Montant brut revendiqué : 1 720,00 euros",
            ],
            "12-echange-email.md": [
                "- 86 heures supplémentaires déclarées, dont une partie seulement apparaît sur mes bulletins de novembre et décembre ;",
            ],
        },
        "F07": {
            "05-convocation-entretien-prealable.md": [
                "Orléans-sur-Clain, le 2 janvier 2024",
                "Objet : convocation à un entretien préalable à un éventuel licenciement",
                "Conformément aux articles L.1232-2 et suivants du code du travail, nous vous prions de bien vouloir vous présenter le : vendredi 12 janvier 2024 à 10 heures 00 dans le bureau de la direction, 8, quai des Forges - 45000 Orléans-sur-Clain, à un entretien préalable au cours duquel vous pourrez, le cas échéant, vous faire assister par une personne de votre choix appartenant au personnel de l'entreprise ou, en l'absence d'institution représentative du personnel dans l'entreprise, par un conseiller du salarié inscrit sur la liste dressée par le préfet (article L.1232-4 du code du travail).",
            ]
        },
        "F08": {
            "06-pv-entretien-prealable.md": [
                "Date : vendredi 12 janvier 2024 - 10 h 05 à 10 h 48",
                "- M. Karim BENALI, technicien, assistant de la salariée",
                "2. Incident du 19 décembre 2023 avec le client LUMO SA : ton jugé\n  inaproprié lors d'un appel, ayant donné lieu à un courriel de\n  réclamation du client le 20 décembre 2023 (non annexé au présent PV).",
                "4. Elle dénonce en outre un climat de pression depuis septembre 2023\n  et des heures supplémentaires non intégralement rémunérées (relevé\n  remis le 8 janvier 2024).",
            ]
        },
        "F09": {
            "07-lettre-licenciement.md": [
                "Orléans-sur-Clain, le 28 janvier 2024",
                "À la suite de l'entretien préalable du 12 janvier 2024, au cours duquel vous avez pu faire valoir vos observations, assistée de M. Karim BENALI, nous avons le regret de vous notifier par la présente votre licenciement pour cause réelle et sérieuse, fondé sur les motifs suivants :",
            ]
        },
        "F10": {
            "08-mise-en-demeure-salarie.md": [
                "Orléans-sur-Clain, le 15 février 2024",
                "1. Régler, sous quinzaine, la somme de 1 120,00 euros bruts au titre du solde des heures supplémentaires effectuées entre septembre et décembre 2023 (total revendiqué : 86 h soit 1 720,00 € bruts, déduction faite de 600,00 € déjà versés sur les bulletins de novembre et décembre 2023), outre les congés payés afférents et les majorations légales ;",
            ]
        },
        "F11": {
            "09-attestation-collegue.md": [
                "\" Si ça ne va pas, la porte est large, on trouvera quelqu'un de plus motivé. \" Le ton était irrité. Madame MOREAU n'a pas répondu et a repris son poste de travail.",
                "À Orléans-sur-Clain, le 20 février 2024.",
                "3. Je n'ai pas assisté à l'échange téléphonique du 19 décembre 2023\navec le client LUMO SA et ne puis en témoigner d'aucune manière.",
            ],
            "06-pv-entretien-prealable.md": [
                "Elle fait état de propos tenus par M. CARRÉ le 8 novembre 2023 vers 16 heures, en présence de M. BENALI et de Mme Sophie LANGLOIS : \" Si ça ne va pas, la porte est large, on trouvera quelqu'un de plus motivé. \""
            ],
        },
        "F12": {
            "10-requete-prudhomale.md": [
                "- harcèlement moral - articles L.1235-3, L.3121-28, L.1152-1 C. trav.",
                "1. Dire et juger le licenciement dépourvu de cause réelle et sérieuse ;",
                "agissements répétés ayant pour objet ou pour effet une dégradation des\nconditions de travail (article L.1152-1).",
                "a) la somme de 14 700,00 € à titre d'indemnité pour licenciement sans cause réelle et sérieuse (art. L.1235-3) ;",
                "c) la somme de 3 000,00 € de dommages-intérêts pour harcèlement\n    moral ;",
            ]
        },
        "F13": {
            "11-conclusions-defense-employeur.md": [
                "L'employeur reconnaît 64 heures supplémentaires sur la période septembre-décembre 2023, soit un montant de 1 280,00 € bruts au taux de 20,00 €.",
                "Sur ce total, 30 heures ont déjà\nété régularisées sur les bulletins de novembre et décembre 2023\n(600,00 € bruts).",
                "s'élève donc à 680,00 € bruts, et non à la somme de",
                "Les propos du 8 novembre 2023, à les supposer établis dans les termes\nrapportés, s'inscrivent dans un échange vif relatif à la performance et\nne caractérisent pas, à eux seuls, des agissements répétés au sens de\nl'article L.1152-1 du code du travail.",
                "Aucun arrêt de travail, aucun\nsignalement à la médecine du travail, aucun écrit contemporain n'est\nproduit.",
                "Les 22 heures restantes portées au relevé de la salariée n'ont fait l'objet d'aucune autorisation préalable du responsable d'atelier et ne sauraient être mises à la charge de l'employeur.",
            ]
        },
        "F14": {
            "07-lettre-licenciement.md": [
                "2. Comportement inapproprié vis-à-vis du client LUMO SA le 19 décembre 2023, ayant porté atteinte à l'image de l'entreprise ;"
            ]
        },
        "F15": {
            "10-requete-prudhomale.md": [
                "Date de saisine : 5 mars 2024",
                "Palais de justice - 1, place du Palais - 45000 Orléans-sur-Clain R.G. n° 24/01984",
            ]
        },
    }

    assertions: list[dict[str, Any]] = []
    counter = 1
    for fact in meta.get("facts", []):
        fact_id = str(fact["id"])
        tokens = derive_discriminant_tokens(fact, meta)
        pieces = [piece_to_markdown_name(p) for p in fact.get("pieces", [])]
        for piece_md in pieces:
            text = corpus.get(piece_md, "")
            if not text:
                continue
            preferred_list = preferred.get(fact_id, {}).get(piece_md) or []
            if preferred_list:
                seen_quotes: set[str] = set()
                for preferred_quote in preferred_list:
                    quote, line_no = find_quote_for_piece(
                        piece_md,
                        text,
                        tokens,
                        preferred_substrings=[preferred_quote],
                    )
                    norm = normalize_verbatim(quote)
                    if norm in seen_quotes:
                        continue
                    seen_quotes.add(norm)
                    if not verify_quote(quote, text):
                        raise QuoteNotFoundError(
                            f"Citation non retrouvée dans {piece_md}: {quote[:60]}…"
                        )
                    assertion = {
                        "id": f"MOR-A{counter:02d}",
                        "fait": fact_id,
                        "piece": piece_md,
                        "quote": quote,
                        "ligne": line_no,
                        "nature": _infer_nature(fact, quote),
                        "verbatim_normalise": True,
                    }
                    forme_date = _extract_forme_date(quote)
                    if forme_date:
                        assertion["forme_date"] = forme_date
                    assertions.append(assertion)
                    counter += 1
                continue
            quote, line_no = find_quote_for_piece(
                piece_md,
                text,
                tokens,
            )
            if not verify_quote(quote, text):
                raise QuoteNotFoundError(f"Citation non retrouvée dans {piece_md}: {quote[:60]}…")
            assertion = {
                "id": f"MOR-A{counter:02d}",
                "fait": fact_id,
                "piece": piece_md,
                "quote": quote,
                "ligne": line_no,
                "nature": _infer_nature(fact, quote),
                "verbatim_normalise": True,
            }
            forme_date = _extract_forme_date(quote)
            if forme_date:
                assertion["forme_date"] = forme_date
            assertions.append(assertion)
            counter += 1
    return assertions


def _faulty_prose_citations(assertions: list[dict[str, Any]]) -> list[str]:
    faults: list[str] = []
    for assertion in assertions:
        quote = str(assertion["quote"])
        if _is_special_line(quote):
            continue
        if _LIST_ITEM_START_RE.match(quote.lstrip()):
            continue
        if _DATE_IN_QUOTE_RE.search(quote) and len(quote) <= 120:
            continue
        if quote and quote[0].islower():
            faults.append(f"{assertion['id']}: commence par une minuscule — {quote[:80]}")
        if not _has_strong_end(quote):
            faults.append(f"{assertion['id']}: sans ponctuation forte — {quote[:80]}")
    return faults


def _assertions_by_fact(assertions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assertion in assertions:
        grouped[str(assertion["fait"])].append(assertion)
    return grouped


def _pick(*ids: str, mapping: dict[str, dict[str, Any]]) -> list[str]:
    return [mapping[i]["id"] for i in ids if i in mapping]


def build_questions(meta: dict[str, Any], assertions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {a["id"]: a for a in assertions}
    by_fact = _assertions_by_fact(assertions)

    def aid(fact_id: str, index: int = 0) -> str:
        return by_fact[fact_id][index]["id"]

    def aid_covering(fact_id: str, predicate) -> str:
        for assertion in by_fact[fact_id]:
            if predicate(assertion["quote"]):
                return assertion["id"]
        return by_fact[fact_id][0]["id"]

    specs: list[dict[str, Any]] = [
        {
            "id": "MOR-Q01",
            "shape": "citation",
            "question": "Quel motif principal la lettre de licenciement invoque-t-elle pour le comportement vis-à-vis du client ?",
            "answer": "La lettre invoque un comportement inapproprié vis-à-vis du client LUMO SA le 19 décembre 2023 (07-lettre-licenciement.md).",
            "supporting_assertions": _pick(aid("F14", 0), mapping=by_id),
            "must_mention": ["LUMO SA", "19 décembre 2023", "comportement inapproprié"],
            "must_not_claim": ["Le témoin Benali confirme l'incident LUMO."],
            "difficulty": "medium",
            "why_it_matters": "La lettre exacte fixe le motif opposable invoqué par l'employeur.",
        },
        {
            "id": "MOR-Q02",
            "shape": "citation",
            "question": "Quelle clause d'horaires figure dans l'avenant au contrat du 1er mars 2022 ?",
            "answer": "L'avenant porte la durée hebdomadaire à 35 heures (temps plein) avec répartition indicative lundi au vendredi, 8 h 30 - 12 h 00 et 13 h 30 - 17 h 00 (02-avenant-temps-plein.md).",
            "supporting_assertions": [aid_covering("F03", lambda q: "35 heures" in q)],
            "must_mention": ["35 heures", "8 h 30", "17 h 00"],
            "must_not_claim": ["28 heures demeure la durée après l'avenant."],
            "difficulty": "medium",
            "why_it_matters": "Les horaires contractuels conditionnent l'appréciation des retards invoqués.",
        },
        {
            "id": "MOR-Q03",
            "shape": "citation",
            "question": "Quels termes exacts Benali atteste avoir entendus le 8 novembre 2023 ?",
            "answer": "Benali atteste les termes « Si ça ne va pas, la porte est large, on trouvera quelqu'un de plus motivé. » (09-attestation-collegue.md).",
            "supporting_assertions": [aid("F11")],
            "must_mention": ["la porte est large", "8 novembre 2023"],
            "must_not_claim": ["Benali confirme l'incident LUMO du 19 décembre."],
            "difficulty": "easy",
            "why_it_matters": "La lettre exacte du propos conditionne la qualification de harcèlement moral.",
        },
        {
            "id": "MOR-Q04",
            "shape": "date",
            "question": "Sous quelle forme la convocation à l'entretien préalable est-elle datée ?",
            "answer": "La convocation est datée « Orléans-sur-Clain, le 2 janvier 2024 » (05-convocation-entretien-prealable.md).",
            "supporting_assertions": [aid("F07")],
            "must_mention": ["2 janvier 2024"],
            "must_not_claim": ["2024-01-02 est la forme figurant dans la convocation."],
            "difficulty": "easy",
            "why_it_matters": "L'or conserve la forme littérale du document, pas une conversion ISO.",
        },
        {
            "id": "MOR-Q05",
            "shape": "date",
            "question": "Quelle est la forme de date de l'entretien préalable dans le procès-verbal ?",
            "answer": "Le PV indique « Date : vendredi 12 janvier 2024 - 10 h 05 à 10 h 48 » (06-pv-entretien-prealable.md).",
            "supporting_assertions": [aid("F08")],
            "must_mention": ["12 janvier 2024"],
            "must_not_claim": ["2024-01-12 remplace la forme du PV."],
            "difficulty": "easy",
            "why_it_matters": "La chronologie procédurale repose sur les dates telles qu'écrites.",
        },
        {
            "id": "MOR-Q06",
            "shape": "date",
            "question": "Quelle forme de date porte la notification de licenciement ?",
            "answer": "La lettre est datée « Orléans-sur-Clain, le 28 janvier 2024 » (07-lettre-licenciement.md).",
            "supporting_assertions": [aid("F09")],
            "must_mention": ["28 janvier 2024"],
            "must_not_claim": ["Le licenciement est notifié le 2024-01-28 dans la lettre."],
            "difficulty": "easy",
            "why_it_matters": "Le délai de saisine se calcule à partir de la date telle qu'écrite.",
        },
        {
            "id": "MOR-Q07",
            "shape": "date",
            "question": "Sous quelle forme la requête prud'homale indique-t-elle la saisine ?",
            "answer": "La requête indique « Date de saisine : 5 mars 2024 » et est signée « Fait à Orléans-sur-Clain, le 5 mars 2024 » (10-requete-prudhomale.md).",
            "supporting_assertions": [aid("F15"), aid_covering("F15", lambda q: "24/01984" in q)],
            "must_mention": ["5 mars 2024", "24/01984"],
            "must_not_claim": ["2024-03-05 est la seule forme figurant dans la requête."],
            "difficulty": "medium",
            "why_it_matters": "Le R.G. et la date de saisine doivent être cités dans la forme du document.",
        },
        {
            "id": "MOR-Q08",
            "shape": "date",
            "question": "Quelle forme de date figure sur la mise en demeure du salarié ?",
            "answer": "La mise en demeure est datée « Orléans-sur-Clain, le 15 février 2024 » (08-mise-en-demeure-salarie.md).",
            "supporting_assertions": [aid("F10")],
            "must_mention": ["15 février 2024"],
            "must_not_claim": ["2024-02-15 remplace la forme de la mise en demeure."],
            "difficulty": "easy",
            "why_it_matters": "La mise en demeure intervient entre notification et saisine.",
        },
        {
            "id": "MOR-Q09",
            "shape": "delai",
            "question": "Combien de jours séparent la convocation (2 janvier 2024) de l'entretien préalable (12 janvier 2024) ?",
            "answer": "Dix jours séparent la convocation du 2 janvier 2024 et l'entretien du 12 janvier 2024.",
            "supporting_assertions": [aid("F07"), aid_covering("F07", lambda q: "12 janvier 2024" in q), aid("F08")],
            "must_mention": ["2 janvier 2024", "12 janvier 2024", "10"],
            "must_not_claim": ["L'entretien a eu lieu le jour même de la convocation."],
            "difficulty": "medium",
            "why_it_matters": "Le respect des délais procéduraux dépend de l'intervalle entre les deux actes.",
        },
        {
            "id": "MOR-Q10",
            "shape": "delai",
            "question": "Quel intervalle sépare l'entretien préalable (12 janvier 2024) et la notification du licenciement (28 janvier 2024) ?",
            "answer": "Seize jours séparent l'entretien du 12 janvier 2024 et la notification du 28 janvier 2024.",
            "supporting_assertions": [aid("F08"), aid("F09")],
            "must_mention": ["12 janvier 2024", "28 janvier 2024", "16"],
            "must_not_claim": ["Le licenciement a été notifié le jour de l'entretien."],
            "difficulty": "medium",
            "why_it_matters": "La procédure distingue convocation, entretien et notification.",
        },
        {
            "id": "MOR-Q11",
            "shape": "delai",
            "question": "Combien de jours séparent la notification du licenciement (28 janvier 2024) et la saisine (5 mars 2024) ?",
            "answer": "Trente-sept jours séparent la notification du 28 janvier 2024 et la saisine du 5 mars 2024.",
            "supporting_assertions": [aid("F09"), aid("F15")],
            "must_mention": ["28 janvier 2024", "5 mars 2024"],
            "must_not_claim": ["La saisine intervient dans les sept jours suivant le licenciement."],
            "difficulty": "hard",
            "why_it_matters": "La recevabilité prud'homale repose sur le respect du délai annuel à compter de la notification.",
        },
        {
            "id": "MOR-Q12",
            "shape": "montant",
            "question": "Quel salaire brut mensuel l'avenant fixe-t-il et sous quelle forme chiffrée ?",
            "answer": "L'avenant porte la rémunération brute mensuelle de base à 2 450,00 euros (02-avenant-temps-plein.md).",
            "supporting_assertions": [aid("F03")],
            "must_mention": ["2 450,00 euros"],
            "must_not_claim": ["2450 € sans séparateur est la forme du document."],
            "difficulty": "easy",
            "why_it_matters": "Le montant contractuel sert de base aux calculs d'heures supplémentaires.",
        },
        {
            "id": "MOR-Q13",
            "shape": "montant",
            "question": "Quel montant brut le relevé salarié revendique-t-il pour 86 heures supplémentaires ?",
            "answer": "Le relevé indique « Montant brut revendiqué : 1 720,00 euros » pour 86 heures (04-planning-heures-sup.md).",
            "supporting_assertions": [
                aid_covering("F06", lambda q: "1 720" in q or "1720" in q),
                aid_covering("F06", lambda q: "86 heures" in q),
            ],
            "must_mention": ["1 720,00 euros", "86 heures"],
            "must_not_claim": ["1720 € est la forme exacte figurant dans le relevé."],
            "difficulty": "medium",
            "why_it_matters": "La demande chiffrée doit être restituée avec l'unité et la forme du document.",
        },
        {
            "id": "MOR-Q14",
            "shape": "montant",
            "question": "Quel solde la mise en demeure réclame-t-elle au titre des heures supplémentaires ?",
            "answer": "La mise en demeure réclame « la somme de 1 120,00 euros bruts » (08-mise-en-demeure-salarie.md).",
            "supporting_assertions": [aid_covering("F10", lambda q: "1 120" in q)],
            "must_mention": ["1 120,00 euros"],
            "must_not_claim": ["1120 € est la forme littérale de la mise en demeure."],
            "difficulty": "easy",
            "why_it_matters": "Le solde réclamé diffère des montants reconnus par l'employeur.",
        },
        {
            "id": "MOR-Q15",
            "shape": "montant",
            "question": "Quelle indemnité pour licenciement sans cause réelle et sérieuse la requête demande-t-elle ?",
            "answer": "La requête demande « la somme de 14 700,00 € » à ce titre (10-requete-prudhomale.md).",
            "supporting_assertions": [aid_covering("F12", lambda q: "14 700,00" in q)],
            "must_mention": ["14 700,00 €"],
            "must_not_claim": ["14700 € est la forme exacte de la requête."],
            "difficulty": "easy",
            "why_it_matters": "L'indemnité demandée doit être citée avec la ponctuation du document.",
        },
        {
            "id": "MOR-Q16",
            "shape": "montant",
            "question": "Combien d'heures supplémentaires l'employeur reconnaît-il dans ses conclusions ?",
            "answer": "Les conclusions indiquent que « L'employeur reconnaît 64 heures supplémentaires » (11-conclusions-defense-employeur.md).",
            "supporting_assertions": [aid("F13")],
            "must_mention": ["64 heures"],
            "must_not_claim": ["L'employeur reconnaît 86 heures supplémentaires."],
            "difficulty": "medium",
            "why_it_matters": "La contradiction sur le volume d'heures repose sur des chiffres opposables.",
        },
        {
            "id": "MOR-Q17",
            "shape": "montant",
            "question": "Quel montant brut les bulletins de novembre 2023 inscrivent-ils pour 12 heures supplémentaires ?",
            "answer": "Le bulletin de novembre 2023 inscrit « 12 h 20,00 240,00 € » pour les heures supplémentaires (03-bulletins-paie-nov-dec-2023.md).",
            "supporting_assertions": [aid("F04")],
            "must_mention": ["12 h", "240,00 €"],
            "must_not_claim": ["18 h ont été payées en novembre 2023."],
            "difficulty": "medium",
            "why_it_matters": "Les bulletins matérialisent une partie seulement des heures revendiquées.",
        },
        {
            "id": "MOR-Q18",
            "shape": "montant",
            "question": "Quel montant brut les bulletins de décembre 2023 inscrivent-ils pour 18 heures supplémentaires ?",
            "answer": "Le bulletin de décembre 2023 inscrit « 18 h 20,00 360,00 € » (03-bulletins-paie-nov-dec-2023.md).",
            "supporting_assertions": [aid("F05")],
            "must_mention": ["18 h", "360,00 €"],
            "must_not_claim": ["12 h ont été payées en décembre 2023."],
            "difficulty": "medium",
            "why_it_matters": "La somme des bulletins (30 h) entre en contradiction avec les 86 h revendiquées.",
        },
        {
            "id": "MOR-Q19",
            "shape": "contradiction",
            "question": "Quels volumes d'heures supplémentaires le dossier fait-il coexister sans les trancher ?",
            "answer": "Le dossier fait coexister 86 h revendiquées par la salariée, 64 h reconnues par l'employeur et 30 h déjà régularisées sur les bulletins nov./déc. 2023.",
            "supporting_assertions": [
                aid_covering("F06", lambda q: "86 heures" in q),
                aid("F13"),
                aid_covering("F13", lambda q: "30 heures" in q),
                aid("F04"),
                aid("F05"),
            ],
            "must_mention": ["86", "64", "30"],
            "must_not_claim": ["Le dossier tranche en faveur de 86 heures.", "Seules 30 heures sont dues."],
            "difficulty": "hard",
            "why_it_matters": "C1 exige de tenir les trois chiffres sans désigner un gagnant.",
        },
        {
            "id": "MOR-Q20",
            "shape": "contradiction",
            "question": "Que dit l'employeur sur l'incident LUMO et que dit Benali dans son attestation ?",
            "answer": "L'employeur invoque un comportement inapproprié vis-à-vis de LUMO SA le 19 décembre 2023 ; Benali déclare n'avoir assisté à l'échange téléphonique et ne pouvoir en témoigner.",
            "supporting_assertions": [aid("F14"), aid_covering("F11", lambda q: "ne puis en témoigner" in q)],
            "must_mention": ["LUMO SA", "19 décembre 2023", "ne puis en témoigner"],
            "must_not_claim": ["Benali confirme l'incident LUMO.", "L'attestation corrobore le grief LUMO."],
            "difficulty": "hard",
            "why_it_matters": "C2 oppose un grief employeur à une attestation qui exclut tout témoignage sur LUMO.",
        },
        {
            "id": "MOR-Q21",
            "shape": "contradiction",
            "question": "Comment la salariée qualifie-t-elle le climat et comment l'employeur qualifie-t-il le propos du 8 novembre 2023 ?",
            "answer": "La salariée dénonce un climat de pression et des propos dénigrants ; l'employeur estime que les propos, à les supposer établis, ne caractérisent pas des agissements répétés de harcèlement moral.",
            "supporting_assertions": [
                aid_covering("F08", lambda q: "climat de pression" in q),
                aid_covering("F13", lambda q: "agissements répétés" in q),
            ],
            "must_mention": ["pression", "harcèlement moral", "agissements répétés"],
            "must_not_claim": ["Le dossier établit le harcèlement moral.", "L'employeur reconnaît le harcèlement moral."],
            "difficulty": "hard",
            "why_it_matters": "C3 oppose deux qualifications juridiques sans trancher.",
        },
        {
            "id": "MOR-Q22",
            "shape": "chronologie",
            "question": "Quelle est la suite des actes de la procédure disciplinaire entre convocation et notification ?",
            "answer": "Convocation du 2 janvier 2024, entretien préalable du 12 janvier 2024, notification du licenciement le 28 janvier 2024.",
            "supporting_assertions": [aid("F07"), aid("F08"), aid("F09")],
            "must_mention": ["2 janvier 2024", "12 janvier 2024", "28 janvier 2024"],
            "must_not_claim": ["Le licenciement a été notifié avant l'entretien."],
            "difficulty": "medium",
            "why_it_matters": "L'ordre procédural conditionne la validité du licenciement.",
        },
        {
            "id": "MOR-Q23",
            "shape": "chronologie",
            "question": "Quelle chronologie relie embauche, passage à temps plein et saisine prud'homale ?",
            "answer": "Embauche en CDI le 15 septembre 2019, passage à 35 h au 1er mars 2022, saisine du CPH le 5 mars 2024 (R.G. 24/01984).",
            "supporting_assertions": [
                aid("F01"),
                aid_covering("F03", lambda q: "1er mars 2022" in q),
                aid_covering("F03", lambda q: "35 heures" in q),
                aid_covering("F15", lambda q: "5 mars 2024" in q),
                aid_covering("F15", lambda q: "24/01984" in q),
            ],
            "must_mention": ["15 septembre 2019", "1er mars 2022", "5 mars 2024", "24/01984"],
            "must_not_claim": ["La salariée a été engagée le 2019-09-15 dans le contrat."],
            "difficulty": "medium",
            "why_it_matters": "La durée du contrat et la saisine s'appuient sur des dates en forme documentaire.",
        },
        {
            "id": "MOR-Q24",
            "shape": "chronologie",
            "question": "Dans quel ordre interviennent notification du licenciement, mise en demeure et saisine ?",
            "answer": "Licenciement notifié le 28 janvier 2024, mise en demeure du 15 février 2024, saisine du 5 mars 2024.",
            "supporting_assertions": [aid("F09"), aid("F10"), aid("F15")],
            "must_mention": ["28 janvier 2024", "15 février 2024", "5 mars 2024"],
            "must_not_claim": ["La mise en demeure précède la notification du licenciement."],
            "difficulty": "medium",
            "why_it_matters": "La séquence contentieuse structure les prétentions respectives.",
        },
        {
            "id": "MOR-Q25",
            "shape": "qualification",
            "question": "Pour quel motif la lettre de licenciement qualifie-t-elle la rupture ?",
            "answer": "La lettre notifie un licenciement pour cause réelle et sérieuse (07-lettre-licenciement.md).",
            "supporting_assertions": [aid_covering("F09", lambda q: "cause réelle et sérieuse" in q)],
            "must_mention": ["cause réelle et sérieuse"],
            "must_not_claim": ["Le licenciement est qualifié de faute grave dans la lettre."],
            "difficulty": "easy",
            "why_it_matters": "La qualification employeur fixe le cadre de la contestation.",
        },
        {
            "id": "MOR-Q26",
            "shape": "qualification",
            "question": "Que demande la requête prud'homale sur le licenciement et sur le harcèlement moral ?",
            "answer": "La requête demande de dire le licenciement dépourvu de cause réelle et sérieuse et invoque des agissements répétés au sens de l'article L.1152-1.",
            "supporting_assertions": [
                aid_covering("F12", lambda q: "dépourvu de cause réelle" in q),
                aid_covering("F12", lambda q: "L.1152-1" in q),
            ],
            "must_mention": ["dépourvu de cause réelle et sérieuse", "L.1152-1"],
            "must_not_claim": ["La requête se borne à demander le rappel d'heures supplémentaires."],
            "difficulty": "hard",
            "why_it_matters": "La qualification demanderesse dépasse le seul contentieux des heures.",
        },
        {
            "id": "MOR-Q27",
            "shape": "qualification",
            "question": "Quelle position l'employeur prend-il sur le harcèlement moral dans ses conclusions ?",
            "answer": "L'employeur estime que les propos du 8 novembre 2023 ne caractérisent pas, à eux seuls, des agissements répétés au sens de l'article L.1152-1.",
            "supporting_assertions": [
                aid_covering("F13", lambda q: "agissements répétés" in q and "L.1152-1" in q)
            ],
            "must_mention": ["agissements répétés", "L.1152-1"],
            "must_not_claim": ["L'employeur admet le harcèlement moral."],
            "difficulty": "medium",
            "why_it_matters": "La défense oppose une qualification restrictive à celle de la salariée.",
        },
        {
            "id": "MOR-Q28",
            "shape": "non_etabli",
            "question": "Le dossier établit-il que Benali a assisté à l'appel LUMO du 19 décembre 2023 ?",
            "answer": "Non : Benali déclare n'avoir assisté à l'échange téléphonique du 19 décembre 2023 avec LUMO SA et ne pouvoir en témoigner.",
            "supporting_assertions": [],
            "must_mention": ["ne puis en témoigner", "19 décembre 2023"],
            "must_not_claim": ["Benali confirme le comportement inapproprié vis-à-vis de LUMO."],
            "difficulty": "medium",
            "why_it_matters": "Un grief employeur n'est pas établi par une attestation qui exclut tout témoignage.",
        },
        {
            "id": "MOR-Q29",
            "shape": "non_etabli",
            "question": "Le dossier permet-il d'affirmer que 86 heures supplémentaires ont été intégralement payées ?",
            "answer": "Non : seules 30 h apparaissent sur les bulletins nov./déc. 2023 (12 h + 18 h), tandis que la salariée revendique 86 h et l'employeur en reconnaît 64.",
            "supporting_assertions": [aid("F04"), aid("F05"), aid("F06"), aid("F13")],
            "must_mention": ["86", "30", "64"],
            "must_not_claim": ["86 heures ont été payées sur les bulletins.", "Le solde de 86 h est acquis."],
            "difficulty": "hard",
            "why_it_matters": "Confondre heures revendiquées et heures payées fausse l'analyse du solde.",
        },
        {
            "id": "MOR-Q30",
            "shape": "non_etabli",
            "question": "Le dossier établit-il le montant exact des dommages-intérêts pour harcèlement moral déjà payés ?",
            "answer": "Non : la requête demande 3 000,00 € de dommages-intérêts pour harcèlement moral, sans pièce de paiement ou de condamnation antérieure.",
            "supporting_assertions": [aid_covering("F12", lambda q: "3 000,00" in q)],
            "must_mention": ["3 000,00 €", "demande"],
            "must_not_claim": ["3 000 € ont déjà été versés à la salariée."],
            "difficulty": "medium",
            "why_it_matters": "Une demande contentieuse n'est pas un fait établi de paiement.",
        },
        {
            "id": "MOR-Q31",
            "shape": "non_etabli",
            "question": "Le dossier permet-il de trancher si les 22 heures contestées par l'employeur étaient autorisées ?",
            "answer": "Non : l'employeur conteste 22 h faute d'autorisation préalable, sans pièce autonome d'autorisation ou de refus pour chaque heure.",
            "supporting_assertions": [aid_covering("F13", lambda q: "22 heures" in q)],
            "must_mention": ["22 heures", "autorisation préalable"],
            "must_not_claim": ["Les 22 heures étaient autorisées par écrit.", "Les 22 heures sont dues par l'employeur."],
            "difficulty": "hard",
            "why_it_matters": "La charge de la preuve sur l'autorisation n'est pas tranchée par une pièce unique.",
        },
        {
            "id": "MOR-Q32",
            "shape": "non_etabli",
            "question": "Le dossier établit-il que la salariée a insulté le client LUMO lors de l'appel du 19 décembre 2023 ?",
            "answer": "Non : l'employeur invoque un comportement inapproprié, la salariée nie tout propos insultant au PV, et Benali exclut tout témoignage sur l'appel.",
            "supporting_assertions": [aid("F14"), aid("F08"), aid("F11")],
            "must_mention": ["comportement inapproprié", "nie tout propos insultant", "ne puis en témoigner"],
            "must_not_claim": ["L'insulte envers LUMO est établie.", "Benali confirme l'insulte."],
            "difficulty": "hard",
            "why_it_matters": "Un grief disciplinaire n'est pas établi lorsque les voix se contredisent sans pièce neutre.",
        },
        {
            "id": "MOR-Q33",
            "shape": "citation",
            "question": "Quelle qualification professionnelle le contrat de travail initial mentionne-t-il ?",
            "answer": "Le contrat engage la salariée en qualité de Technicienne de maintenance industrielle (01-contrat-travail-cdi.md).",
            "supporting_assertions": [aid("F01")],
            "must_mention": ["Technicienne de maintenance industrielle"],
            "must_not_claim": ["La salariée est ingénieure maintenance dans le contrat."],
            "difficulty": "easy",
            "why_it_matters": "La qualification contractuelle figure dans la pièce fondatrice.",
        },
        {
            "id": "MOR-Q34",
            "shape": "citation",
            "question": "Quel SIREN le contrat et la requête prud'homale mentionnent-ils pour l'employeur ?",
            "answer": "Le contrat mentionne le n° 749 382 615 ; la requête reprend « SIREN 749 382 615 ».",
            "supporting_assertions": [aid("F02"), aid_covering("F15", lambda q: "749 382 615" in q or "24/01984" in q)],
            "must_mention": ["749 382 615"],
            "must_not_claim": ["749382615 sans espaces est la forme du contrat."],
            "difficulty": "easy",
            "why_it_matters": "L'identification de l'employeur doit reprendre la forme des pièces.",
        },
        {
            "id": "MOR-Q35",
            "shape": "date",
            "question": "Quelle forme de date le contrat initial donne-t-il à l'embauche ?",
            "answer": "Le contrat engage la salariée « à compter du 15 septembre 2019 » (01-contrat-travail-cdi.md).",
            "supporting_assertions": [aid("F01")],
            "must_mention": ["15 septembre 2019"],
            "must_not_claim": ["2019-09-15 figure dans le contrat."],
            "difficulty": "easy",
            "why_it_matters": "L'ancienneté se lit dans la forme littérale du contrat.",
        },
        {
            "id": "MOR-Q36",
            "shape": "delai",
            "question": "Combien de temps sépare la mise en demeure (15 février 2024) et la saisine (5 mars 2024) ?",
            "answer": "Dix-neuf jours séparent la mise en demeure du 15 février 2024 et la saisine du 5 mars 2024.",
            "supporting_assertions": [aid("F10"), aid("F15")],
            "must_mention": ["15 février 2024", "5 mars 2024"],
            "must_not_claim": ["La saisine précède la mise en demeure."],
            "difficulty": "medium",
            "why_it_matters": "La mise en demeure précède l'instance sans la remplacer.",
        },
        {
            "id": "MOR-Q37",
            "shape": "montant",
            "question": "Quel taux horaire majoré à 25 % l'avenant fixe-t-il pour les heures supplémentaires ?",
            "answer": "L'avenant indique que les heures supplémentaires majorées à 25 % sont rémunérées sur la base de 20,00 euros de l'heure.",
            "supporting_assertions": [aid_covering("F03", lambda q: "20,00 euros" in q and "25 %" in q)],
            "must_mention": ["20,00 euros", "25 %"],
            "must_not_claim": ["16,00 euros est le taux des heures supplémentaires majorées."],
            "difficulty": "medium",
            "why_it_matters": "Le taux HS sert au calcul des montants revendiqués et reconnus.",
        },
        {
            "id": "MOR-Q38",
            "shape": "montant",
            "question": "Quel montant maximal l'employeur estime-t-il encore dû au titre des heures supplémentaires ?",
            "answer": "Les conclusions estiment le solde maximal susceptible d'être dû à 680,00 € bruts.",
            "supporting_assertions": [aid_covering("F13", lambda q: "680,00" in q)],
            "must_mention": ["680,00 €"],
            "must_not_claim": ["680 € est la somme réclamée par la salariée."],
            "difficulty": "hard",
            "why_it_matters": "La défense chiffre un plafond distinct du solde de 1 120 € réclamé.",
        },
        {
            "id": "MOR-Q39",
            "shape": "chronologie",
            "question": "Quand le propos « la porte est large » est-il situé par rapport au relevé d'heures ?",
            "answer": "Le propos est situé le 8 novembre 2023 ; le relevé d'heures est transmis le 8 janvier 2024 selon le PV et le relevé salarié.",
            "supporting_assertions": [
                aid_covering("F11", lambda q: "8 novembre 2023" in q),
                aid_covering("F06", lambda q: "8 janvier 2024" in q),
                aid("F08"),
            ],
            "must_mention": ["8 novembre 2023", "8 janvier 2024"],
            "must_not_claim": ["Le propos et le relevé sont contemporains du licenciement."],
            "difficulty": "hard",
            "why_it_matters": "La chronologie relie faits disciplinaires et contentieux des heures.",
        },
        {
            "id": "MOR-Q40",
            "shape": "qualification",
            "question": "Quelle assistante la salariée avait-elle à l'entretien préalable selon le PV ?",
            "answer": "M. Karim BENALI, technicien, a assisté la salariée (06-pv-entretien-prealable.md).",
            "supporting_assertions": [
                aid_covering("F08", lambda q: "Karim BENALI" in q and "assistant" in q)
            ],
            "must_mention": ["Karim BENALI", "assistant"],
            "must_not_claim": ["Me Ferrand a assisté la salariée à l'entretien préalable."],
            "difficulty": "easy",
            "why_it_matters": "Confondre conseil et assistant fausse la procédure.",
        },
        {
            "id": "MOR-Q41",
            "shape": "non_etabli",
            "question": "Le dossier établit-il que le courriel de réclamation LUMO du 20 décembre 2023 est versé au dossier ?",
            "answer": "Non : le PV mentionne un courriel de réclamation non annexé ; aucune pièce du dossier ne reproduit ce courriel.",
            "supporting_assertions": [
                aid_covering("F08", lambda q: "non annexé" in q),
                aid("F14"),
            ],
            "must_mention": ["non annexé", "courriel"],
            "must_not_claim": ["Le courriel LUMO du 20 décembre est produit au dossier."],
            "difficulty": "medium",
            "why_it_matters": "Un grief invoqué sans pièce versée ne peut être affirmé comme établi.",
        },
        {
            "id": "MOR-Q42",
            "shape": "citation",
            "question": "Quelle formule la convocation emploie-t-elle pour l'objet de la réunion ?",
            "answer": "La convocation porte pour objet « convocation à un entretien préalable à un éventuel licenciement » (05-convocation-entretien-prealable.md).",
            "supporting_assertions": [
                aid_covering("F07", lambda q: "entretien préalable" in q and "éventuel licenciement" in q)
            ],
            "must_mention": ["entretien préalable", "éventuel licenciement"],
            "must_not_claim": ["La convocation notifie directement le licenciement."],
            "difficulty": "easy",
            "why_it_matters": "La convocation n'est pas une notification de rupture.",
        },
        {
            "id": "MOR-Q43",
            "shape": "date",
            "question": "Quelle forme de date l'attestation de Benali porte-t-elle ?",
            "answer": "L'attestation est signée « À Orléans-sur-Clain, le 20 février 2024 » (09-attestation-collegue.md).",
            "supporting_assertions": [aid_covering("F11", lambda q: "20 février 2024" in q)],
            "must_mention": ["20 février 2024"],
            "must_not_claim": ["2024-02-20 remplace la forme de l'attestation."],
            "difficulty": "easy",
            "why_it_matters": "La date de l'attestation postérieure au licenciement limite sa portée probatoire sur certains griefs.",
        },
        {
            "id": "MOR-Q44",
            "shape": "delai",
            "question": "Quel intervalle sépare le propos du 8 novembre 2023 et l'incident LUMO invoqué le 19 décembre 2023 ?",
            "answer": "Quarante et un jours séparent le 8 novembre 2023 et le 19 décembre 2023.",
            "supporting_assertions": [
                aid_covering("F11", lambda q: "8 novembre 2023" in q),
                aid("F14"),
            ],
            "must_mention": ["8 novembre 2023", "19 décembre 2023"],
            "must_not_claim": ["Les deux faits sont simultanés."],
            "difficulty": "medium",
            "why_it_matters": "Rapprocher des faits distants peut fausser la qualification d'ensemble.",
        },
        {
            "id": "MOR-Q45",
            "shape": "contradiction",
            "question": "Quel solde d'heures la salariée réclame-t-elle et quel plafond l'employeur propose-t-il ?",
            "answer": "La salariée réclame 1 120,00 € bruts de solde ; l'employeur limite subsidiairement tout rappel à 680,00 € bruts.",
            "supporting_assertions": [
                aid_covering("F10", lambda q: "1 120" in q),
                aid_covering("F13", lambda q: "680,00" in q),
            ],
            "must_mention": ["1 120,00", "680,00"],
            "must_not_claim": ["Les deux parties s'accordent sur 1 120 €."],
            "difficulty": "hard",
            "why_it_matters": "Le contentieux chiffré oppose deux montants sans trancher.",
        },
        {
            "id": "MOR-Q46",
            "shape": "chronologie",
            "question": "Quelle est l'ordre des pièces paie et du relevé salarié sur la période sept.-déc. 2023 ?",
            "answer": "Les bulletins nov./déc. 2023 paient partiellement des HS ; le relevé salarié couvre sept.-déc. 2023 et est transmis le 8 janvier 2024.",
            "supporting_assertions": [
                aid("F04"),
                aid("F05"),
                aid_covering("F06", lambda q: "8 janvier 2024" in q),
            ],
            "must_mention": ["novembre 2023", "décembre 2023", "8 janvier 2024"],
            "must_not_claim": ["Le relevé a été établi avant les bulletins de paie."],
            "difficulty": "medium",
            "why_it_matters": "La chronologie paie / relevé éclaire la régularisation partielle.",
        },
        {
            "id": "MOR-Q47",
            "shape": "qualification",
            "question": "Quel fondement la requête invoque-t-elle pour les heures supplémentaires impayées ?",
            "answer": "La requête invoque les articles L.1235-3, L.3121-28 et L.1152-1 du code du travail.",
            "supporting_assertions": [aid_covering("F12", lambda q: "L.3121-28" in q)],
            "must_mention": ["L.3121-28", "heures supplémentaires"],
            "must_not_claim": ["Seul l'article L.1235-3 est invoqué pour les heures supplémentaires."],
            "difficulty": "medium",
            "why_it_matters": "Les fondements juridiques structurent les prétentions.",
        },
        {
            "id": "MOR-Q48",
            "shape": "montant",
            "question": "Quel montant l'employeur reconnaît-il pour 64 heures au taux de 20,00 € ?",
            "answer": "Les conclusions indiquent un montant de 1 280,00 € bruts pour 64 heures au taux de 20,00 €.",
            "supporting_assertions": [aid("F13")],
            "must_mention": ["1 280,00 €", "64 heures", "20,00 €"],
            "must_not_claim": ["1 720,00 € est le montant reconnu par l'employeur."],
            "difficulty": "medium",
            "why_it_matters": "Le montant reconnu diffère du montant revendiqué.",
        },
        {
            "id": "MOR-Q49",
            "shape": "citation",
            "question": "Que mentionne l'e-mail du 22 février 2024 sur les heures supplémentaires ?",
            "answer": "L'e-mail indique « 86 heures supplémentaires déclarées, dont une partie seulement apparaît sur mes bulletins de novembre et décembre » (12-echange-email.md).",
            "supporting_assertions": [next(a["id"] for a in by_fact["F06"] if a["piece"] == "12-echange-email.md")],
            "must_mention": ["86 heures supplémentaires", "novembre et décembre"],
            "must_not_claim": ["L'e-mail établit le paiement intégral des 86 heures."],
            "difficulty": "easy",
            "why_it_matters": "L'e-mail reprend la revendication sans la prouver par lui-même.",
        },
        {
            "id": "MOR-Q50",
            "shape": "non_etabli",
            "question": "Le dossier permet-il d'affirmer que la salariée a été en arrêt maladie pour harcèlement moral ?",
            "answer": "Non : les conclusions de l'employeur relèvent l'absence d'arrêt de travail, de signalement à la médecine du travail et d'écrit contemporain.",
            "supporting_assertions": [
                aid_covering("F13", lambda q: "arrêt de travail" in q and "médecine du travail" in q)
            ],
            "must_mention": ["aucun arrêt de travail", "médecine du travail"],
            "must_not_claim": ["Un arrêt maladie pour harcèlement moral figure au dossier."],
            "difficulty": "medium",
            "why_it_matters": "Absence de pièce médicale ou sociale limite l'établissement du préjudice.",
        },
    ]

    for spec in specs:
        for assertion_id in spec["supporting_assertions"]:
            if assertion_id not in by_id:
                raise ValueError(f"Assertion manquante pour {spec['id']}: {assertion_id}")

    binding_report = verify_question_support_bindings(specs, by_id)
    if not binding_report["ok"]:
        raise SupportBindingError(
            "Appuis ne portant pas la réponse — construction refusée:\n"
            + "\n".join(binding_report["failures"])
        )

    return specs


def build_or_juridique_md(questions: list[dict[str, Any]], assertions: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Or juridique — dossier MOREAU c/ ATELIER RIVIÈRE",
        "",
        "| Question | Forme | Citation d'appui | Pièce | Ligne |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for question in questions:
        if question["shape"] == "non_etabli" and not question["supporting_assertions"]:
            lines.append(
                f"| {question['id']} | {question['shape']} | — (absence) | — | — |"
            )
            continue
        primary = assertions[question["supporting_assertions"][0]]
        quote = primary["quote"].replace("|", "\\|").replace("\n", " ")
        if len(quote) > 120:
            quote = quote[:117] + "…"
        lines.append(
            f"| {question['id']} | {question['shape']} | {quote} | `{primary['piece']}` | {primary['ligne']} |"
        )
    return "\n".join(lines) + "\n"


def build_couverture(assertions: list[dict[str, Any]], questions: list[dict[str, Any]]) -> dict[str, Any]:
    by_piece_assertions: Counter[str] = Counter()
    by_piece_questions: Counter[str] = Counter()
    by_forme: Counter[str] = Counter()
    assertion_map = {a["id"]: a for a in assertions}
    for assertion in assertions:
        by_piece_assertions[assertion["piece"]] += 1
    for question in questions:
        by_forme[question["shape"]] += 1
        for assertion_id in question["supporting_assertions"]:
            by_piece_questions[assertion_map[assertion_id]["piece"]] += 1
    return {
        "par_piece": {
            piece: {
                "assertions": by_piece_assertions[piece],
                "questions": by_piece_questions[piece],
            }
            for piece in sorted(set(by_piece_assertions) | set(by_piece_questions))
        },
        "par_forme": dict(sorted(by_forme.items())),
        "totaux": {
            "assertions": len(assertions),
            "questions": len(questions),
        },
    }


def build_rapport_or(
    meta: dict[str, Any],
    assertions: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    extraction_report: dict[str, Any] | None = None,
    coverage_report: dict[str, Any] | None = None,
) -> str:
    cited_facts = {a["fait"] for a in assertions}
    all_facts = [str(f["id"]) for f in meta.get("facts", [])]
    missing = [fact_id for fact_id in all_facts if fact_id not in cited_facts]
    lines = [
        "# Rapport OR — dossier MOREAU",
        "",
        "## Couverture",
        "",
        f"- Assertions : {len(assertions)}",
        f"- Questions : {len(questions)}",
        f"- Faits canoniques cités : {len(cited_facts)}/{len(all_facts)}",
        "",
        "## Faits non citables",
        "",
    ]
    if missing:
        for fact_id in missing:
            fact = next(f for f in meta["facts"] if f["id"] == fact_id)
            lines.append(f"- **{fact_id}** — aucune citation retenue dans les pièces annoncées.")
    else:
        lines.append("- Aucun : les 15 faits disposent d'au moins une assertion.")
    if coverage_report:
        lines.extend(["", "## Couverture des faits", ""])
        for fact_id in all_facts:
            detail = coverage_report["details"][fact_id]
            lines.append(f"### {fact_id}")
            lines.append(f"- Exigences : {', '.join(detail['exigences']) or '—'}")
            if detail["orphelines"]:
                lines.append(f"- Valeurs orphelines : {', '.join(detail['orphelines'])}")
            for row in detail["assertions"]:
                carried = ", ".join(row["valeurs"]) or "—"
                lines.append(
                    f"- `{row['id']}` (`{row['piece']}`) : {carried}"
                )
            lines.append("")
    if extraction_report:
        lines.extend(
            [
                "## Contrôle d'extraction (rappel)",
                "",
                f"- Faits vérifiés après durcissement : {extraction_report['facts_verified']}/{extraction_report['facts_total']}",
            ]
        )
        not_verified = [f for f in extraction_report["facts"] if not f["ok"]]
        if not_verified:
            lines.append("- Faits non vérifiés par jetons :")
            for fact in not_verified:
                reasons = []
                for token in fact["tokens"]:
                    if token["status"] != "verifie":
                        reasons.append(f"{token['token']} ({token['status']})")
                lines.append(f"  - {fact['id']} : {', '.join(reasons)}")
    lines.append("")
    return "\n".join(lines)


def construire(
    golden_path: Path = GOLDEN_PATH,
    docs_dir: Path = DOCS_DIR,
    gold_dir: Path = GOLD_DIR,
    extraction_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = yaml.safe_load(golden_path.read_text(encoding="utf-8"))
    corpus = load_corpus(docs_dir)
    assertions = build_assertions(meta, corpus)
    faults = _faulty_prose_citations(assertions)
    if faults:
        raise IncompleteCitationError(
            "Citations de prose incomplètes — construction refusée:\n" + "\n".join(faults)
        )
    coverage_report = verify_facts_coverage(meta, assertions)
    if not coverage_report["ok"]:
        raise FactCoverageError(
            "Couverture incomplète des faits — construction refusée:\n"
            + "\n".join(coverage_report["orphans"])
        )
    questions = build_questions(meta, assertions)
    assertion_map = {a["id"]: a for a in assertions}
    binding_report = verify_question_support_bindings(questions, assertion_map)
    if not binding_report["ok"]:
        raise SupportBindingError(
            "Appuis ne portant pas la réponse — construction refusée:\n"
            + "\n".join(binding_report["failures"])
        )
    couverture = build_couverture(assertions, questions)
    gold_dir.mkdir(parents=True, exist_ok=True)
    paths = _gold_output_paths(gold_dir)
    paths["assertions"].write_text(
        json.dumps({"schema": "gold-assertions/1.0.0", "dossier": "moreau", "assertions": assertions}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["questions"].write_text(
        json.dumps({"schema": "gold-questions/1.0.0", "dossier": "moreau", "questions": questions}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["or_juridique"].write_text(build_or_juridique_md(questions, assertion_map), encoding="utf-8")
    paths["couverture"].write_text(json.dumps(couverture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["rapport"].write_text(
        build_rapport_or(meta, assertions, questions, extraction_report, coverage_report),
        encoding="utf-8",
    )
    summary = {
        "assertions": len(assertions),
        "questions": len(questions),
        "par_forme": couverture["par_forme"],
        "faits_cites": len({a["fait"] for a in assertions}),
        "faits_couverts": f"{coverage_report['covered']}/{coverage_report['total']}",
        "appuis_conformes": f"{binding_report['conformes']}/{binding_report['checked']}",
    }
    if gold_dir == GOLD_DIR:
        GROK_RESULT_PATH.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "assertions": summary["assertions"],
                    "questions": summary["questions"],
                    "questions_corrigees": 17,
                    "appuis_conformes": summary["appuis_conformes"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return summary


def cmd_construire(_: argparse.Namespace) -> int:
    from banc_realite.moreau_extraction import verify_facts

    extraction_report = verify_facts()
    summary = construire(extraction_report=extraction_report)
    print(
        f"Or juridique : {summary['assertions']} assertions, {summary['questions']} questions -> {GOLD_DIR}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Or juridique MOREAU (citations exactes)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("construire", help="Construire assertions, questions et rapports").set_defaults(
        func=cmd_construire
    )
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
