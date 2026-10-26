# SPDX-License-Identifier: AGPL-3.0-only
"""Porte phrase — découpe lexicale des passages bruts sans nouvel embedding."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import BANC_ROOT
from .fidelite import _assertion_needle, load_gold
from .holders import collect_appui_states
from .normalize import normalize_text, piece_basename

_DIAGNOSTIC_PATH = BANC_ROOT / "sorties" / "diagnostic-couverture" / "resultat.json"
_NUMBER_RE = re.compile(r"\d+")
_DATE_RE = re.compile(
    r"\b(?:20\d{2}|19\d{2})[-/.]\d{1,2}[-/.]\d{1,2}\b|\b\d{1,2}[-/.]\d{1,2}[-/.](?:20\d{2}|19\d{2})\b"
)
_BULLET_RE = re.compile(r"^(?:[-*]|\d+\.)\s+")
_TABLE_ROW_RE = re.compile(r"^\|")

_STOP_WORDS = frozenset(
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
        "what",
        "which",
        "does",
        "how",
        "many",
        "who",
        "when",
        "where",
        "why",
        "do",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
    }
)


def _clean_unit(text: str, *, max_chars: int = 300) -> str | None:
    cleaned = " ".join(text.split())
    if len(cleaned) < 20:
        return None
    if len(cleaned) <= max_chars:
        return cleaned
    chunk = cleaned[:max_chars]
    if len(cleaned) > max_chars and cleaned[max_chars] != " ":
        last_space = chunk.rfind(" ")
        if last_space >= 20:
            chunk = chunk[:last_space]
    return chunk + " …"


def _split_sentences(prose: str) -> list[str]:
    if not prose.strip():
        return []
    parts = re.split(r"(?<=[.!?;])(?=\s|$)", prose.strip())
    return [part.strip() for part in parts if part.strip()]


def _table_unit_lines(lines: list[str], row_index: int) -> str:
    """Une ligne ``|`` = une unité ; inclut l'en-tête (première ligne ``|`` du bloc)."""
    block_start = row_index
    while block_start > 0 and _TABLE_ROW_RE.match(lines[block_start - 1].strip()):
        block_start -= 1
    return " ".join(" ".join(lines[i].split()) for i in range(block_start, row_index + 1))


def _max_chars_fenetre(fenetre: int) -> int:
    """Borne caractères par unité — 300 (1 phrase), 450 (2), puis 150 × (N+1)."""
    if fenetre < 1:
        raise ValueError(f"fenetre invalide : {fenetre}")
    return 150 * (fenetre + 1)


def _prose_units(sentences: list[str], *, fenetre: int) -> list[tuple[str, tuple[int, int] | None]]:
    """Unités prose ; ``(texte, (début, fin))`` pour fenêtres glissantes, ``None`` sinon."""
    if not sentences:
        return []
    if fenetre < 1:
        raise ValueError(f"fenetre invalide : {fenetre}")
    max_chars = _max_chars_fenetre(fenetre)
    if fenetre == 1:
        out: list[tuple[str, tuple[int, int] | None]] = []
        for idx, sentence in enumerate(sentences):
            cleaned = _clean_unit(sentence, max_chars=max_chars)
            if cleaned:
                out.append((cleaned, (idx, idx)))
        return out
    if len(sentences) <= fenetre:
        combined = " ".join(sentences)
        cleaned = _clean_unit(combined, max_chars=max_chars)
        return [(cleaned, (0, len(sentences) - 1))] if cleaned else []
    out = []
    for start in range(len(sentences) - fenetre + 1):
        combined = " ".join(sentences[start : start + fenetre])
        cleaned = _clean_unit(combined, max_chars=max_chars)
        if cleaned:
            out.append((cleaned, (start, start + fenetre - 1)))
    return out


def phrases_du_passage(text: str, *, fenetre: int = 1) -> list[str]:
    """Découpe un passage en unités (phrases, lignes de tableau, puces).

    ``fenetre=1`` : une phrase par unité (borne 300 caractères).
    ``fenetre=2`` : fenêtre glissante de deux phrases consécutives en prose
    (borne 450 caractères) ; une seule phrase donne une unité d'une phrase.
    ``fenetre=N`` : fenêtre glissante de N phrases (borne 150×(N+1) car.).
    Tableaux (avec en-tête) et puces restent une unité chacun.
    """
    if fenetre < 1:
        raise ValueError(f"fenetre invalide : {fenetre}")
    return [unit for unit, _span in _phrases_du_passage_avec_fenetres(text, fenetre=fenetre)]


def _phrases_du_passage_avec_fenetres(text: str, *, fenetre: int) -> list[tuple[str, tuple[int, int] | None]]:
    """Comme ``phrases_du_passage`` mais expose les indices de phrases pour la déduplication."""
    if fenetre < 1:
        raise ValueError(f"fenetre invalide : {fenetre}")
    if not text:
        return []
    lines = text.splitlines()
    units: list[tuple[str, tuple[int, int] | None]] = []
    index = 0
    max_chars = _max_chars_fenetre(fenetre)
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if _TABLE_ROW_RE.match(stripped):
            unit = _table_unit_lines(lines, index)
            cleaned = _clean_unit(unit, max_chars=max_chars)
            if cleaned:
                units.append((cleaned, None))
            index += 1
            continue
        if _BULLET_RE.match(stripped):
            cleaned = _clean_unit(stripped, max_chars=max_chars)
            if cleaned:
                units.append((cleaned, None))
            index += 1
            continue
        prose_lines: list[str] = []
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                if prose_lines:
                    break
                index += 1
                continue
            if _TABLE_ROW_RE.match(candidate) or _BULLET_RE.match(candidate):
                break
            prose_lines.append(candidate)
            index += 1
        units.extend(_prose_units(_split_sentences(" ".join(prose_lines)), fenetre=fenetre))
    return units


def _dedupe_chevauchements(
    ranked: list[tuple[int, float, str, tuple[int, int] | None]],
    *,
    n_par_passage: int,
) -> list[tuple[int, float, str]]:
    """Retire les fenêtres qui partagent une phrase ; garde les mieux classées."""
    ordered = sorted(ranked, key=lambda row: (-row[1], row[0]))
    selected: list[tuple[int, float, str]] = []
    used_sentences: set[int] = set()
    for idx, score, text, span in ordered:
        if len(selected) >= n_par_passage:
            break
        if span is None:
            selected.append((idx, score, text))
            continue
        start, end = span
        sentence_ids = set(range(start, end + 1))
        if sentence_ids & used_sentences:
            continue
        used_sentences |= sentence_ids
        selected.append((idx, score, text))
    if not selected and ordered:
        idx, score, text, _span = ordered[0]
        selected.append((idx, score, text))
    return selected


def _question_tokens(question: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) >= 3 and token not in _STOP_WORDS
    }


def _question_numbers_and_dates(question: str) -> list[str]:
    values: list[str] = []
    values.extend(_DATE_RE.findall(question))
    values.extend(_NUMBER_RE.findall(question))
    return values


def score_phrase(question: str, phrase: str) -> float:
    """Recouvrement lexical question↔phrase, bonus +0,5 par nombre/date partagé."""
    q_tokens = _question_tokens(question)
    if not q_tokens:
        base = 0.0
    else:
        p_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", phrase.lower())
            if len(token) >= 3 and token not in _STOP_WORDS
        }
        base = len(q_tokens & p_tokens) / len(q_tokens)
    bonus = 0.0
    phrase_lower = phrase.lower()
    for value in _question_numbers_and_dates(question):
        if value.lower() in phrase_lower:
            bonus += 0.5
    return base + bonus


def _passage_source(passage: dict[str, Any]) -> str:
    provenance = passage.get("provenance") or {}
    return str(passage.get("source") or provenance.get("source_file_path") or "")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _score_unite(
    question: str,
    unit: str,
    *,
    mode: str,
    question_vector: list[float] | None,
    cache: dict[str, list[float]] | None,
    stats: dict[str, int],
) -> float:
    if mode == "lexical":
        return score_phrase(question, unit)
    if mode != "vectoriel":
        raise ValueError(f"mode inconnu : {mode}")
    if cache is None or question_vector is None:
        stats["unites_sans_vecteur"] += 1
        return 0.0
    vector = cache.get(unit)
    if vector is None:
        stats["unites_sans_vecteur"] += 1
        return 0.0
    return _cosine(question_vector, vector)


def _select_par_passage(
    ranked: list[tuple[int, float, str, tuple[int, int] | None]],
    *,
    n_par_passage: int,
    fenetre: int,
) -> list[tuple[int, float, int, str]]:
    if fenetre == 2:
        chosen = _dedupe_chevauchements(ranked, n_par_passage=n_par_passage)
    else:
        ordered = sorted(ranked, key=lambda row: (-row[1], row[0]))
        positive = [row[:3] for row in ordered if row[1] > 0]
        fallback = [row[:3] for row in ordered]
        chosen = positive[:n_par_passage] if positive else fallback[:n_par_passage]
    return [(idx, score, idx, text) for idx, score, text in chosen]


def _select_globale(
    candidates: list[tuple[int, float, int, dict[str, Any]]],
    *,
    max_phrases: int,
    n_par_passage: int,
) -> list[dict[str, Any]]:
    """Meilleures unités tous passages confondus, au plus ``n_par_passage`` par passage."""
    ordered = sorted(candidates, key=lambda row: (-row[1], row[0], row[2]))
    selected: list[dict[str, Any]] = []
    per_passage: dict[int, int] = {}
    for passage_rank, score, unit_index, item in ordered:
        if len(selected) >= max_phrases:
            break
        count = per_passage.get(passage_rank, 0)
        if count >= n_par_passage:
            continue
        selected.append(item)
        per_passage[passage_rank] = count + 1
    return selected


def porte_phrase(
    question: str,
    evidence_brute: list[dict[str, Any]],
    *,
    n_par_passage: int = 2,
    max_phrases: int = 24,
    mode: str = "lexical",
    question_vector: list[float] | None = None,
    cache: dict[str, list[float]] | None = None,
    selection: str = "par_passage",
    fenetre: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sélectionne les meilleures unités depuis les passages bruts.

    ``mode="lexical"`` : recouvrement lexical (témoin #111).
    ``mode="vectoriel"`` : cosinus question↔unité ; unité absente du cache → score 0.

    ``selection="par_passage"`` : ``n_par_passage`` meilleures unités par passage.
    ``selection="globale"`` : ``max_phrases`` meilleures unités tous passages confondus,
    au plus ``n_par_passage`` par passage pour éviter qu'un passage monopolise la sélection.

    ``fenetre`` : transmis à ``phrases_du_passage`` (1 = une phrase, 2 = fenêtre glissante).

    Une unité de score 0 n'est retenue que si aucune unité du même passage n'a un score
    strictement positif (on garde alors les premières unités à égalité de score nul).

    Avec ``fenetre=2``, une phrase ne sert qu'une fois : si deux fenêtres retenues d'un
    même passage se chevauchent (phrases 2+3 et 3+4), seule la mieux classée est gardée.
    """
    stats = {"unites_sans_vecteur": 0}
    candidates: list[tuple[int, float, int, dict[str, Any]]] = []

    for passage_rank, passage in enumerate(evidence_brute):
        if passage.get("kind") != "file_hit":
            continue
        passage_id = str(passage.get("id") or f"p{passage_rank}")
        source = _passage_source(passage)
        text = passage.get("text") or passage.get("snippet") or ""
        units = _phrases_du_passage_avec_fenetres(str(text), fenetre=fenetre)
        ranked = [
            (
                idx,
                _score_unite(
                    question,
                    unit,
                    mode=mode,
                    question_vector=question_vector,
                    cache=cache,
                    stats=stats,
                ),
                unit,
                span,
            )
            for idx, (unit, span) in enumerate(units)
        ]
        if selection == "globale":
            if fenetre == 2:
                globale_rows = _dedupe_chevauchements(ranked, n_par_passage=len(ranked))
            else:
                globale_rows = [(idx, score, text) for idx, score, text, _span in ranked]
            for unit_index, phrase_score, unit_text in globale_rows:
                candidates.append(
                    (
                        passage_rank,
                        phrase_score,
                        unit_index,
                        {
                            "kind": "phrase_hit",
                            "id": f"phrase:{passage_id}:{unit_index}",
                            "text": unit_text,
                            "source": source,
                            "score": phrase_score,
                            "provenance": {
                                "source_file_path": source,
                                "passage_id": passage_id,
                            },
                        },
                    )
                )
            continue

        chosen = _select_par_passage(ranked, n_par_passage=n_par_passage, fenetre=fenetre)
        for unit_index, phrase_score, _, unit_text in chosen:
            candidates.append(
                (
                    passage_rank,
                    phrase_score,
                    unit_index,
                    {
                        "kind": "phrase_hit",
                        "id": f"phrase:{passage_id}:{unit_index}",
                        "text": unit_text,
                        "source": source,
                        "score": phrase_score,
                        "provenance": {
                            "source_file_path": source,
                            "passage_id": passage_id,
                        },
                    },
                )
            )

    if selection == "globale":
        return _select_globale(
            candidates,
            max_phrases=max_phrases,
            n_par_passage=n_par_passage,
        ), stats

    candidates.sort(key=lambda row: (row[0], -row[1], row[2]))
    return [item for _, _, _, item in candidates[:max_phrases]], stats


def liste_clauses_sans_chiffre(
    gold: dict[str, Any] | None = None,
    diagnostic_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Assertions ``extrait_infidele`` dont l'aiguille ne contient aucun chiffre."""
    gold = gold or load_gold()
    path = diagnostic_path or _DIAGNOSTIC_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    assertion_index = gold["assertion_index"]
    clauses: list[dict[str, Any]] = []
    for line in payload["lignes"]:
        if line.get("verdict") != "extrait_infidele":
            continue
        assertion = assertion_index[line["assertion_id"]]
        needle = _assertion_needle(assertion)
        if _NUMBER_RE.search(needle):
            continue
        clauses.append(
            {
                "question_id": line["question_id"],
                "assertion_id": line["assertion_id"],
            }
        )
    return clauses


def compter_clauses_portees(
    question_id: str,
    evidence: list[dict[str, Any]],
    *,
    clauses: list[dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
) -> int:
    count = 0
    for clause in clauses:
        if clause["question_id"] != question_id:
            continue
        assertion = assertion_index[clause["assertion_id"]]
        states = collect_appui_states(assertion, evidence, claims_index)
        if states["tient"]:
            count += 1
    return count


def assertions_tenues_par_une_unite(
    assertions: list[dict[str, Any]],
    pieces_text: dict[str, str],
    *,
    fenetre: int,
) -> dict[str, bool]:
    """Chaque assertion est tenue si son aiguille normalisée est contenue dans une unité."""
    result: dict[str, bool] = {}
    for assertion in assertions:
        needle = _assertion_needle(assertion)
        if not needle:
            result[needle] = False
            continue
        piece = piece_basename(assertion.get("source") or "")
        text = pieces_text.get(piece, "")
        units = phrases_du_passage(text, fenetre=fenetre)
        held = any(needle in normalize_text(unit) for unit in units)
        result[needle] = held
    return result


def compter_perte_selection(
    question: dict[str, Any],
    evidence_passages: list[dict[str, Any]],
    evidence_phrases: list[dict[str, Any]],
    *,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
) -> int:
    lost = 0
    for aid in question.get("supporting_assertions") or []:
        assertion = assertion_index[aid]
        brute_states = collect_appui_states(assertion, evidence_passages, claims_index)
        phrase_states = collect_appui_states(assertion, evidence_phrases, claims_index)
        if brute_states["tient"] and not phrase_states["tient"]:
            lost += 1
    return lost
