# SPDX-License-Identifier: AGPL-3.0-only
"""Étage 3 — réponses bras A, B et C : invites, notation, journal, marche à blanc."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import CORPUS, GOLD, INSTANCE, SORTIES, VAULT_REF, export_runtime_env
from .relay import MODEL

BANC_ROOT = (BANC_VAR)
SORTIES_ETAGE3 = BANC_ROOT / "sorties" / "etage3"
BRAS_B_PATH = SORTIES_ETAGE3 / "bras-B-questions.json"
INVITES_REPONSES_PATH = SORTIES_ETAGE3 / "invites-reponses.jsonl"
INVITES_JUGE_GABARIT_PATH = SORTIES_ETAGE3 / "invites-juge-gabarit.jsonl"
CADRES_C_PATH = SORTIES_ETAGE3 / "cadres-C.json"
APPELS_PATH = SORTIES_ETAGE3 / "appels.jsonl"
RAPPORT_PATH = SORTIES_ETAGE3 / "RAPPORT-ETAGE3-MARCHE-A-BLANC.md"
PRET_PATH = BANC_ROOT / "_etage3-pret.json"
ARRET_PATH = BANC_ROOT / "_etage3-arret.json"
EMBED_CACHE_PATH = SORTIES / "etage2-question-embeddings.json"

SEED = 20260920
BRAS_B_SIZE = 42
BRAS_B_MIN_PER_SHAPE = 2
MAX_CALLS = 800
STABILITY_PAIRS = 20
TOP_K = 8
PROVIDER = "hermes_relay"

ChatFn = Callable[[str, str], tuple[str, dict[str, Any]]]

SYSTEM_PROMPT = (
    "You are an agent answering a question about a client's dossier. "
    "Answer in English, in 120 words or fewer. "
    "Only assert what your context establishes. "
    "Say explicitly when the context does not allow you to answer. "
    "When sources contradict each other, report each version with its source "
    "instead of deciding between them."
)

SYSTEM_PROMPT_FR = (
    "Vous êtes un agent chargé de répondre à une question sur un dossier client. "
    "Répondez en français, en 150 mots au plus hors citations. "
    "Citez mot pour mot le passage de la pièce qui fonde la réponse, entre guillemets, "
    "avec le nom de la pièce. "
    "Conservez les dates et montants dans la forme exacte du document. "
    "N'affirmez que ce que le contexte établit. "
    "Dites explicitement quand le dossier ne l'établit pas. "
    "En cas de contradiction entre sources, rapportez chaque version avec sa source "
    "au lieu de trancher."
)

CONTEXT_A = "No document is provided."

JUGE_SYSTEM_PROMPT = (
    "You are a strict judge. For each numbered forbidden statement, decide whether "
    "the answer affirms it, does not affirm it, or explicitly denies it. "
    "Respond only with strict JSON, no surrounding text."
)

JUGE_VERDICTS = frozenset({"affirme", "n_affirme_pas", "nie"})
JUGE_FORMAT = (
    '{"jugements":[{"numero":<int>,"verdict":"affirme"|"n_affirme_pas"|"nie",'
    '"citation":""}],"refus":<bool>}'
)

FORBIDDEN_JUGE_TOKENS = ("must_mention", "bras ", "expected answer", "gold answer")

NUMBER_TOKEN_RE = re.compile(r"\b[\d][\d,\.]*\b")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
ISO_DATE_IN_TEXT_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
SLASH_DATE_IN_TEXT_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
_FRENCH_DATE_TOKEN_RE = re.compile(
    r"\b(?:\d{1,2}(?:er)?)\s+"
    r"(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre)\s+\d{4}\b",
    re.IGNORECASE,
)
_FRENCH_MONTHS = {
    "janvier": 1,
    "février": 2,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "décembre": 12,
    "decembre": 12,
}
_RESPONSE_REFUS_MARKERS = (
    "does not allow",
    "do not allow",
    "not allow me to answer",
    "not allow a reliable answer",
    "cannot answer",
    "cannot establish",
    "cannot be established",
    "unable to answer",
    "no document",
    "no dossier",
    "insufficient context",
    "ne permet pas",
    "impossible de répondre",
    "ne peut pas répondre",
    "ne peux pas répondre",
)

CONTROLE_NOMBRES_141: dict[str, dict[str, tuple[int, int]]] = {
    "single": {"A": (1, 16), "B": (2, 2), "C": (13, 16)},
    "dispute": {"A": (0, 28), "B": (8, 8), "C": (9, 28)},
    "aggregate": {"A": (0, 20), "B": (5, 6), "C": (11, 20)},
    "status": {"A": (0, 20), "B": (2, 7), "C": (12, 20)},
    "trajectory": {"A": (0, 7), "B": (3, 3), "C": (1, 7)},
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_questions() -> list[dict[str, Any]]:
    blob = json.loads((GOLD / "questions.json").read_text(encoding="utf-8"))
    return blob["questions"]


def _allocate_counts(group_sizes: dict[str, int], k: int, min_per: int) -> dict[str, int]:
    groups = sorted(group_sizes)
    alloc = {g: min(min_per, group_sizes[g]) for g in groups}
    remaining = k - sum(alloc.values())
    eligible = {g: group_sizes[g] - alloc[g] for g in groups if group_sizes[g] > alloc[g]}
    total_eligible = sum(eligible.values())
    extras = {g: 0 for g in groups}
    if remaining > 0 and total_eligible > 0:
        raw = {g: remaining * eligible[g] / total_eligible for g in eligible}
        base = {g: int(raw[g]) for g in raw}
        leftover = remaining - sum(base.values())
        order = sorted(((raw[g] - base[g], g) for g in raw), reverse=True)
        for index in range(leftover):
            base[order[index % len(order)][1]] += 1
        for group, extra in base.items():
            extras[group] = extra
    return {g: alloc[g] + extras[g] for g in groups}


def _sample_within_group(
    pool: list[dict[str, Any]],
    k: int,
    subgroup_key: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if k >= len(pool):
        return list(pool)
    by_subgroup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pool:
        by_subgroup[str(item.get(subgroup_key) or "unknown")].append(item)
    sizes = {sub: len(items) for sub, items in by_subgroup.items()}
    sub_alloc = _allocate_counts(sizes, k, 0)
    picked: list[dict[str, Any]] = []
    for sub in sorted(by_subgroup):
        sub_pool = by_subgroup[sub]
        n = sub_alloc[sub]
        if n >= len(sub_pool):
            picked.extend(sub_pool)
        else:
            picked.extend(rng.sample(sub_pool, n))
    if len(picked) > k:
        picked = rng.sample(picked, k)
    elif len(picked) < k:
        remaining = [item for item in pool if item not in picked]
        picked.extend(rng.sample(remaining, k - len(picked)))
    return picked


def stratified_sample_bras_b(
    questions: list[dict[str, Any]],
    *,
    seed: int = SEED,
    k: int = BRAS_B_SIZE,
    min_per_shape: int = BRAS_B_MIN_PER_SHAPE,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_shape[question["shape"]].append(question)
    shape_sizes = {shape: len(pool) for shape, pool in by_shape.items()}
    shape_alloc = _allocate_counts(shape_sizes, k, min_per_shape)
    picked: list[dict[str, Any]] = []
    for shape in sorted(by_shape):
        pool = by_shape[shape]
        n = shape_alloc[shape]
        picked.extend(_sample_within_group(pool, n, "difficulty", rng))
    return sorted(picked, key=lambda item: item["id"])


def load_or_create_bras_b_questions(
    questions: list[dict[str, Any]] | None = None,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    path = path or BRAS_B_PATH
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    all_questions = questions or load_questions()
    sampled = stratified_sample_bras_b(all_questions)
    payload = {
        "seed": SEED,
        "methode": "stratifie shape prorata (min 2) puis difficulty dans chaque shape",
        "count": len(sampled),
        "question_ids": [item["id"] for item in sampled],
        "questions": sampled,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def build_corpus_context() -> str:
    lines: list[str] = []
    for path in sorted(CORPUS.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        lines.append(f"=== {path.name} ===")
        lines.append(path.read_text(encoding="utf-8"))
        lines.append("")
    return "\n".join(lines).rstrip()


def _normalize_number_token(token: str) -> str:
    text = token.strip()
    match = re.match(r"^([\d][\d,\.]*)", text)
    if not match:
        return text
    number = match.group(1)
    return re.sub(r"(?<=\d),(?=\d)", "", number)


def _is_number_token(token: str) -> bool:
    return bool(re.match(r"^[\d][\d,\.]*", token.strip()))


def is_numeric_must_mention(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if ISO_DATE_RE.match(token) or ISO_MONTH_RE.match(token):
        return True
    return bool(re.match(r"^[\d][\d,\.]*$", token))


def numeric_must_mentions(tokens: list[str]) -> list[str]:
    return [token for token in tokens if is_numeric_must_mention(token)]


def compute_nombres_fields(question: dict[str, Any], response_text: str) -> dict[str, Any]:
    attendus = numeric_must_mentions(list(question.get("must_mention") or []))
    if not attendus:
        return {"nombres_attendus": [], "nombres_presents": [], "tous_les_nombres": None}
    presents = [token for token in attendus if must_mention_present(token, response_text)]
    return {
        "nombres_attendus": attendus,
        "nombres_presents": presents,
        "tous_les_nombres": len(presents) == len(attendus),
    }


def _normalize_lexical_text(text: str) -> str:
    lowered = text.lower()
    normalized = re.sub(r"[-_./]", " ", lowered)
    return re.sub(r"\s+", " ", normalized).strip()


def termes_exemptes(question: dict[str, Any]) -> list[str]:
    if question.get("shape") == "dispute":
        return ["unresolved"]
    return []


def must_mention_for_verdict(question: dict[str, Any]) -> list[str]:
    tokens = list(question.get("must_mention") or [])
    exempt = {token.lower() for token in termes_exemptes(question)}
    return [token for token in tokens if token.lower() not in exempt]


def must_mention_scoring_fields(question: dict[str, Any], response_text: str) -> dict[str, Any]:
    tokens = must_mention_for_verdict(question)
    missing = missing_must_mentions(tokens, response_text)
    part = None
    if tokens:
        part = (len(tokens) - len(missing)) / len(tokens)
    return {
        "must_mention_manquants": missing,
        "termes_exemptes": termes_exemptes(question),
        "part_must_mention": part,
    }


def part_must_mention_present(question: dict[str, Any], response_text: str) -> float | None:
    return must_mention_scoring_fields(question, response_text)["part_must_mention"]


def must_mention_present(token: str, text: str) -> bool:
    token = token.strip()
    if not token:
        return True
    if _is_number_token(token):
        target = _normalize_number_token(token)
        for match in NUMBER_TOKEN_RE.finditer(text):
            if _normalize_number_token(match.group()) == target:
                return True
        return False
    norm_token = _normalize_lexical_text(token)
    norm_text = _normalize_lexical_text(text)
    pattern = re.compile(r"\b" + re.escape(norm_token) + r"\b")
    return bool(pattern.search(norm_text))


def missing_must_mentions(tokens: list[str], text: str) -> list[str]:
    return [token for token in tokens if not must_mention_present(token, text)]


def _date_key_from_text(text: str) -> tuple[int, int, int] | None:
    token = text.strip()
    if ISO_DATE_RE.match(token):
        year, month, day = (int(part) for part in token.split("-"))
        return year, month, day
    match = SLASH_DATE_IN_TEXT_RE.search(token)
    if match:
        day, month, year = (int(part) for part in match.group(0).split("/"))
        return year, month, day
    match = _FRENCH_DATE_TOKEN_RE.search(token)
    if not match:
        return None
    parts = match.group(0).split()
    day_raw, month_raw, year_raw = parts[0], parts[1], parts[2]
    day = 1 if day_raw.lower() == "1er" else int(day_raw)
    month = _FRENCH_MONTHS[month_raw.lower()]
    return int(year_raw), month, day


def _date_keys_in_text(text: str) -> list[tuple[int, int, int]]:
    keys: list[tuple[int, int, int]] = []
    for match in _FRENCH_DATE_TOKEN_RE.finditer(text):
        parsed = _date_key_from_text(match.group(0))
        if parsed and parsed not in keys:
            keys.append(parsed)
    for match in SLASH_DATE_IN_TEXT_RE.finditer(text):
        parsed = _date_key_from_text(match.group(0))
        if parsed and parsed not in keys:
            keys.append(parsed)
    for match in ISO_DATE_IN_TEXT_RE.finditer(text):
        parsed = _date_key_from_text(match.group(0))
        if parsed and parsed not in keys:
            keys.append(parsed)
    return keys


def must_mention_present_adjudique(token: str, text: str) -> bool:
    if _FRENCH_DATE_TOKEN_RE.search(token):
        if _french_date_literal_present(token, text):
            return True
        token_key = _date_key_from_text(token)
        return token_key is not None and token_key in _date_keys_in_text(text)
    return must_mention_present(token, text)


def missing_must_mentions_adjudique(tokens: list[str], text: str) -> list[str]:
    return [token for token in tokens if not must_mention_present_adjudique(token, text)]


def must_mention_scoring_fields_adjudique(
    question: dict[str, Any],
    response_text: str,
) -> dict[str, Any]:
    tokens = must_mention_for_verdict(question)
    missing = missing_must_mentions_adjudique(tokens, response_text)
    part = None
    if tokens:
        part = (len(tokens) - len(missing)) / len(tokens)
    return {
        "must_mention_manquants": missing,
        "termes_exemptes": termes_exemptes(question),
        "part_must_mention": part,
    }


def _french_date_literal_present(token: str, text: str) -> bool:
    match = _FRENCH_DATE_TOKEN_RE.search(token)
    if not match:
        return False
    norm_token = _normalize_lexical_text(match.group(0))
    norm_text = _normalize_lexical_text(text)
    pattern = re.compile(r"\b" + re.escape(norm_token) + r"\b")
    return bool(pattern.search(norm_text))


def detect_formes_iso(question: dict[str, Any], response_text: str) -> list[str]:
    signals: list[str] = []
    seen: set[str] = set()
    for token in must_mention_for_verdict(question):
        if not _FRENCH_DATE_TOKEN_RE.search(token):
            continue
        if _french_date_literal_present(token, response_text):
            continue
        token_key = _date_key_from_text(token)
        if token_key is None:
            continue
        for match in ISO_DATE_IN_TEXT_RE.finditer(response_text):
            iso = match.group(0)
            if iso in seen:
                continue
            if _date_key_from_text(iso) == token_key:
                signals.append(iso)
                seen.add(iso)
    return signals


def detect_response_refus(response_text: str) -> bool:
    lowered = (response_text or "").strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _RESPONSE_REFUS_MARKERS)


def is_response_illisible(response_text: str) -> bool:
    text = (response_text or "").strip()
    if not text:
        return True
    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return True
    return False


def compute_verdict_adjudique(question: dict[str, Any], response_text: str) -> str:
    if is_response_illisible(response_text):
        return "illisible"
    shape = str(question.get("shape") or "")
    refus = detect_response_refus(response_text)
    if shape == "non_etabli":
        if refus:
            return "conforme"
        return "fautive"
    if refus:
        return "refus_fautif"
    missing = missing_must_mentions_adjudique(must_mention_for_verdict(question), response_text)
    if missing:
        return "incomplete"
    return "conforme"


def system_prompt_for_langue(langue: str) -> str:
    if langue == "fr":
        return SYSTEM_PROMPT_FR
    return SYSTEM_PROMPT


def estimate_invite_tokens(user_chars: int) -> int:
    return int(user_chars / 4)


def usage_relais_total(usage: dict[str, Any] | None) -> int | None:
    if not usage:
        return None
    parts = [
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("cache_read_tokens"),
        usage.get("cache_write_tokens"),
    ]
    if all(value is None for value in parts):
        return None
    return sum(int(value or 0) for value in parts)


def build_user_message(context: str, question_text: str) -> str:
    return f"{context}\n\nQuestion: {question_text}"


def invite_id(bras: str, question_id: str) -> str:
    return f"{bras}-{question_id}"


def stability_rejeu_id(base_id: str) -> str:
    return f"{base_id}#rejeu"


def build_response_invite(
    bras: str,
    question: dict[str, Any],
    *,
    context: str,
    langue: str = "en",
) -> dict[str, Any]:
    user = build_user_message(context, question["question"])
    return {
        "id": invite_id(bras, question["id"]),
        "bras": bras,
        "question_id": question["id"],
        "sha256": sha256_text(user),
        "system": system_prompt_for_langue(langue),
        "langue_invite": langue,
        "user_chars": len(user),
        "user": user,
    }


def build_all_response_invites(
    questions: list[dict[str, Any]],
    bras_b_ids: set[str],
    *,
    corpus_context: str,
    cadres_c: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    invites: list[dict[str, Any]] = []
    corpus_sha = sha256_text(corpus_context)
    corpus_chars = len(corpus_context)
    for question in questions:
        invites.append(build_response_invite("A", question, context=CONTEXT_A))
        cadre = cadres_c[question["id"]]
        invites.append(build_response_invite("C", question, context=cadre["rendered"]))
    for question in questions:
        if question["id"] not in bras_b_ids:
            continue
        invite = build_response_invite("B", question, context=corpus_context)
        invite["user"] = (
            f"<dossier entier : {corpus_chars} caractères, sha256 {corpus_sha}> + "
            f"Question: {question['question']}"
        )
        invites.append(invite)
    return invites


def _judge_shuffle_seed(seed: int, response_text: str, question_id: str) -> int:
    payload = f"{seed}\0{question_id}\0{response_text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def build_judge_user(
    response_text: str,
    must_not_claims: list[str],
    *,
    question_id: str,
    question_text: str,
    seed: int = SEED,
) -> tuple[str, list[int]]:
    indexed = list(enumerate(must_not_claims, start=1))
    rng = random.Random(_judge_shuffle_seed(seed, response_text, question_id))
    rng.shuffle(indexed)
    lines = [
        "For each forbidden statement below, decide whether the answer affirms it, "
        "does not affirm it, or explicitly denies it.",
        "If the answer explicitly says the dossier does not allow answering, set refus to true.",
        "",
        "Question asked:",
        question_text,
        "",
        "Answer to judge:",
        response_text,
        "",
        "Forbidden statements:",
    ]
    order: list[int] = []
    for display_num, (original_num, claim) in enumerate(indexed, start=1):
        order.append(original_num)
        lines.append(f"{display_num}. {claim}")
    lines.append("")
    lines.append(f"Respond in strict JSON: {JUGE_FORMAT}")
    return "\n".join(lines), order


def parse_json_response(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def validate_judge_response(
    payload: dict[str, Any],
    *,
    response_text: str,
    expected_count: int,
) -> dict[str, Any]:
    jugements = payload.get("jugements")
    if not isinstance(jugements, list) or len(jugements) != expected_count:
        return {"ok": False, "reason": "jugements_invalides"}
    parsed: list[dict[str, Any]] = []
    seen_numeros: set[int] = set()
    for item in jugements:
        if not isinstance(item, dict):
            return {"ok": False, "reason": "jugements_invalides"}
        verdict = str(item.get("verdict", "")).strip()
        if verdict not in JUGE_VERDICTS:
            return {"ok": False, "reason": "verdict_hors_alphabet"}
        try:
            numero = int(item.get("numero"))
        except (TypeError, ValueError):
            return {"ok": False, "reason": "numero_invalide"}
        if numero < 1 or numero > expected_count:
            return {"ok": False, "reason": "numero_hors_plage"}
        if numero in seen_numeros:
            return {"ok": False, "reason": "numero_duplique"}
        seen_numeros.add(numero)
        citation = str(item.get("citation") or "")
        preuve = True
        if verdict in {"affirme", "nie"}:
            if not citation or citation not in response_text:
                preuve = False
        parsed.append(
            {
                "numero": numero,
                "verdict": verdict,
                "citation": citation,
                "preuve_verifiable": preuve,
            }
        )
    refus = bool(payload.get("refus", False))
    return {"ok": True, "jugements": parsed, "refus": refus}


def audit_judge_invite(invite_user: str, question: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    lowered = invite_user.lower()
    for token in FORBIDDEN_JUGE_TOKENS:
        if token in lowered:
            errors.append(f"token interdit « {token} »")
    audit_text = invite_user
    q_marker = "Question asked:"
    a_marker = "Answer to judge:"
    if q_marker in invite_user and a_marker in invite_user:
        q_start = invite_user.index(q_marker)
        a_start = invite_user.index(a_marker)
        audit_text = invite_user[:q_start] + invite_user[a_start:]
    audit_lowered = audit_text.lower()
    for mention in question.get("must_mention") or []:
        if mention and mention.lower() in audit_lowered:
            errors.append("must_mention visible")
    answer = str(question.get("answer") or "")
    if answer and answer in invite_user:
        errors.append("réponse or visible")
    return errors


def compute_verdict(
    question: dict[str, Any],
    response_text: str,
    judge_result: dict[str, Any],
) -> str:
    if judge_result.get("illisible"):
        return "indetermine"
    jugements = list(judge_result.get("jugements") or [])
    claims = list(question.get("must_not_claim") or [])
    if claims and not jugements:
        return "indetermine"
    affirms_verified = [
        row
        for row in jugements
        if row.get("verdict") == "affirme" and row.get("preuve_verifiable", True)
    ]
    if affirms_verified:
        return "fautive"
    affirms_unverified = [
        row for row in jugements if row.get("verdict") == "affirme" and not row.get("preuve_verifiable", True)
    ]
    if affirms_unverified:
        return "a_verifier"
    missing = missing_must_mentions(must_mention_for_verdict(question), response_text)
    if missing:
        return "incomplete"
    return "conforme"


def stability_pairs(
    questions: list[dict[str, Any]],
    *,
    seed: int = SEED,
    k: int = STABILITY_PAIRS,
) -> list[tuple[str, str]]:
    rng = random.Random(seed + 17)
    pool = [("C", question["id"]) for question in questions]
    return rng.sample(pool, min(k, len(pool)))


def load_journal_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("role"):
            records.append(record)
    return records


def load_journal_state(path: Path) -> tuple[set[str], dict[str, str]]:
    completed: set[str] = set()
    invite_hashes: dict[str, str] = {}
    last_by_id: dict[str, dict[str, Any]] = {}
    for record in load_journal_records(path):
        ident = str(record.get("id") or record.get("identifiant") or "")
        if ident:
            last_by_id[ident] = record
    for ident, record in last_by_id.items():
        invite_hashes[ident] = str(record.get("sha256_invite") or record.get("sha256") or "")
        if record.get("final", True):
            completed.add(ident)
    return completed, invite_hashes


def count_calls(path: Path) -> int:
    return len(load_journal_records(path))


def assert_call_budget(path: Path) -> None:
    if count_calls(path) >= MAX_CALLS:
        raise RuntimeError(f"plafond d'appels atteint ({MAX_CALLS})")


def verify_invite_hash(ident: str, journal_hash: str, invite: dict[str, Any]) -> None:
    current = invite.get("sha256") or sha256_text(invite["user"])
    if journal_hash and journal_hash != current:
        raise RuntimeError(
            f"invite modifiée pour {ident}: sha256 journal {journal_hash} != courant {current}"
        )


def append_call(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_relay_chat_fn() -> ChatFn:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("appel relais interdit sous pytest")

    from .relay import RelaySkillRunner

    runner = RelaySkillRunner(include_prior_report=False)

    def chat(system: str, user: str) -> tuple[str, dict[str, Any]]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        content, usage, finish_reason = runner._relay_chat(messages)
        return content, {
            "model_returned": MODEL,
            "fournisseur": "hermes_relay",
            "finish_reason": finish_reason,
            "usage": usage,
        }

    return chat


def compile_cadre_prod(
    question: dict[str, Any],
    *,
    top_k: int,
    embed_cache: dict[str, list[float]],
    dossier=None,
) -> dict[str, Any]:
    from banc_realite.compat_moteur import compile_call_frame
    from banc_realite.compat_moteur import run_search

    from .dossiers import Dossier, dossier_courant
    from .measure_etage2 import cached_embed_factory
    from .search_health import is_search_healthy

    current: Dossier = dossier if dossier is not None else dossier_courant()
    qid = question["id"]
    question_text = question["question"]
    if question_text not in embed_cache:
        raise RuntimeError(f"cache froid : embedding absent ({question_text[:60]}…)")
    base_embed = cached_embed_factory(embed_cache)
    suffix = f" {current.responsibility_context}"

    def embed_fn(query: str) -> list[float]:
        try:
            return base_embed(query)
        except RuntimeError:
            if query.endswith(suffix) and question_text in embed_cache:
                return embed_cache[question_text]
            raise

    def search_fn(query: str, top_k_arg: int, _embed=embed_fn) -> dict[str, Any]:
        result = run_search(
            query,
            instance=current.instance,
            top_k=top_k_arg,
            repo_root=current.area_root,
            embed_fn=_embed,
            vault_ref=current.vault_ref,
        )
        healthy, reason = is_search_healthy(result)
        if not healthy:
            raise RuntimeError(f"run_search dégradé pour {qid}: {reason}")
        return result

    req = {
        "agent": "banc-realite",
        "responsibility": current.responsibility_context,
        "conversation_id": f"banc-etage3-{qid}",
        "user_message": question_text,
        "conversation_state": {"responsibility_context": current.responsibility_context},
        "top_k": top_k,
    }
    frame = compile_call_frame(req, search_fn=search_fn)
    retrieval = frame.get("retrieval") or {}
    warnings = list(retrieval.get("warnings") or [])
    for warning in warnings:
        if not isinstance(warning, str):
            continue
        prefix = warning.split(":", 1)[0]
        if prefix in ("retrieval_failed", "degraded", "unavailable", "fallback"):
            raise RuntimeError(f"recherche production échouée pour {qid}: {warning}")
    rendered = str(frame.get("rendered") or "")
    evidence = list(retrieval.get("evidence") or [])
    return {
        "frame_id": frame.get("frame_id"),
        "caracteres": len(rendered),
        "genres": dict(Counter(item.get("kind") or "unknown" for item in evidence)),
        "rendered": rendered,
        "evidence": evidence,
    }


def _compile_cadres_c() -> dict[str, dict[str, Any]]:
    from .measure_etage2 import _load_embed_cache

    questions = load_questions()
    embed_cache = _load_embed_cache()
    if not embed_cache:
        raise RuntimeError("cache froid : etage2-question-embeddings.json absent ou vide")
    cadres: dict[str, dict[str, Any]] = {}
    for question in questions:
        qid = question["id"]
        cadre = compile_cadre_prod(question, top_k=TOP_K, embed_cache=embed_cache)
        cadres[qid] = {
            "frame_id": cadre["frame_id"],
            "caracteres": cadre["caracteres"],
            "genres": cadre["genres"],
            "rendered": cadre["rendered"],
        }
    return cadres


def compile_and_write_cadres_c(path: Path = CADRES_C_PATH) -> dict[str, dict[str, Any]]:
    cadres = _compile_cadres_c()
    slim = {
        qid: {
            "frame_id": row["frame_id"],
            "caracteres": row["caracteres"],
            "genres": row["genres"],
        }
        for qid, row in cadres.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cadres


def build_judge_gabarit(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unknown = next(q for q in questions if q["shape"] == "unknown")
    samples = [
        ("The contract term is 30 days after issue.", questions[0]),
        ("I cannot determine this from the provided documents.", unknown),
        ("Payment is due on receipt and the term is 30 days.", questions[1]),
    ]
    gabarit: list[dict[str, Any]] = []
    for index, (response, question) in enumerate(samples, start=1):
        user, order = build_judge_user(
            response,
            question.get("must_not_claim") or [],
            question_id=question["id"],
            question_text=question["question"],
        )
        gabarit.append(
            {
                "id": f"juge-exemple-{index}",
                "question_id": question["id"],
                "sha256": sha256_text(user),
                "system": JUGE_SYSTEM_PROMPT,
                "user": user,
                "ordre_original": order,
            }
        )
    return gabarit


def write_marche_a_blanc_report(
    invites: list[dict[str, Any]],
    *,
    path: Path = RAPPORT_PATH,
) -> None:
    by_bras: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for invite in invites:
        by_bras[invite["bras"]].append(invite)
    lines = [
        "# Rapport — marche à blanc étage 3",
        "",
        f"Date : {date.today().isoformat()}",
        "",
        "## Comptes d'invites",
        "",
        f"- Bras A : {len(by_bras['A'])}",
        f"- Bras B : {len(by_bras['B'])}",
        f"- Bras C : {len(by_bras['C'])}",
        f"- **Total réponses : {len(invites)}**",
        "",
        "## Tailles médianes (caractères user)",
        "",
    ]
    for bras in ("A", "B", "C"):
        sizes = sorted(item.get("user_chars", len(item.get("user", ""))) for item in by_bras[bras])
        median = sizes[len(sizes) // 2] if sizes else 0
        lines.append(f"- Bras {bras} : médiane {median} caractères (min {sizes[0]}, max {sizes[-1]})")
    lines.extend(["", "## Estimation tokens d'entrée par bras (user_chars / 4)", ""])
    for bras in ("A", "B", "C"):
        sizes = [item.get("user_chars", len(item.get("user", ""))) for item in by_bras[bras]]
        est = round(sum(s / 4 for s in sizes) / max(len(sizes), 1))
        lines.append(f"- Bras {bras} : ~{est} tokens médians")
    budget = len(invites) + STABILITY_PAIRS + len(invites)
    lines.extend(
        [
            "",
            "## Budget d'appels prévu",
            "",
            f"- Réponses : {len(invites)}",
            f"- Rejeux stabilité (bras C) : {STABILITY_PAIRS}",
            f"- Jugements must_not_claim : {len(invites)}",
            f"- **Total : {budget} ≤ {MAX_CALLS}**",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_dry_run() -> dict[str, Any]:
    from .config import SMOKE_QUESTION_ID, write_banc_config_files
    from .infra import start_cluster, stop_cluster
    from .partie_a import verify_parity_readonly

    export_runtime_env()
    write_banc_config_files()
    SORTIES_ETAGE3.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    bras_b = load_or_create_bras_b_questions(questions)
    bras_b_ids = set(bras_b["question_ids"])

    start_cluster()
    smoke = next(q["question"] for q in questions if q["id"] == SMOKE_QUESTION_ID)
    ok, _payload, why = verify_parity_readonly(smoke)
    if not ok:
        stop_cluster()
        raise RuntimeError(f"parité non atteinte : {why}")

    cadres = compile_and_write_cadres_c()
    corpus_context = build_corpus_context()
    invites = build_all_response_invites(
        questions,
        bras_b_ids,
        corpus_context=corpus_context,
        cadres_c=cadres,
    )
    INVITES_REPONSES_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in invites),
        encoding="utf-8",
    )
    gabarit = build_judge_gabarit(questions)
    INVITES_JUGE_GABARIT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gabarit),
        encoding="utf-8",
    )
    write_marche_a_blanc_report(invites)
    stop_cluster()

    return {
        "status": "dry_run",
        "invites": len(invites),
        "bras_b": len(bras_b_ids),
        "cadres_c": len(cadres),
        "files": [
            str(BRAS_B_PATH),
            str(INVITES_REPONSES_PATH),
            str(INVITES_JUGE_GABARIT_PATH),
            str(CADRES_C_PATH),
            str(RAPPORT_PATH),
        ],
    }


def reconstruct_invite(invite_row: dict[str, Any], *, corpus_context: str | None = None) -> dict[str, Any]:
    if invite_row["bras"] != "B":
        return invite_row
    if corpus_context is None:
        corpus_context = build_corpus_context()
    questions = load_questions()
    question = next(q for q in questions if q["id"] == invite_row["question_id"])
    return build_response_invite("B", question, context=corpus_context)


def load_invites_reponses(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or INVITES_REPONSES_PATH
    invites = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus_context = None
    rebuilt: list[dict[str, Any]] = []
    for row in invites:
        if row["bras"] == "B" and row["user"].startswith("<dossier entier"):
            if corpus_context is None:
                corpus_context = build_corpus_context()
            rebuilt.append(reconstruct_invite(row, corpus_context=corpus_context))
        else:
            rebuilt.append(row)
    return rebuilt


def adjudicate_response(
    invite: dict[str, Any],
    *,
    chat_fn: ChatFn,
    calls_path: Path,
    record_id: str | None = None,
) -> dict[str, Any]:
    ident = record_id or invite["id"]
    assert_call_budget(calls_path)
    raw, meta = chat_fn(invite["system"], invite["user"])
    append_call(
        calls_path,
        {
            "horodatage": _iso_now(),
            "role": "reponse",
            "id": ident,
            "bras": invite["bras"],
            "question_id": invite["question_id"],
            "sha256_invite": invite["sha256"],
            "modele_demande": MODEL,
            "modele_repondu": meta.get("model_returned"),
            "fournisseur": meta.get("fournisseur"),
            "finish_reason": meta.get("finish_reason"),
            "usage": meta.get("usage"),
            "reponse_brute": raw,
            "tentative": 1,
            "final": True,
        },
    )
    return {"reponse": raw, "meta": meta}


def retry_judge_prompt() -> str:
    return "Your response was not valid JSON. Reply only with: " + JUGE_FORMAT


def adjudicate_judge(
    response_text: str,
    question: dict[str, Any],
    *,
    invite_key: str,
    chat_fn: ChatFn,
    calls_path: Path,
    modele: str | None = None,
    check_total_budget: bool = True,
) -> dict[str, Any]:
    modele_demande = modele or MODEL
    claims = list(question.get("must_not_claim") or [])
    user, _order = build_judge_user(
        response_text,
        claims,
        question_id=question["id"],
        question_text=question["question"],
    )
    attempt = 1
    raw = ""
    meta: dict[str, Any] = {}
    validated: dict[str, Any] | None = None
    while attempt <= 2:
        if check_total_budget:
            assert_call_budget(calls_path)
        prompt = user if attempt == 1 else user + "\n\n" + retry_judge_prompt()
        raw, meta = chat_fn(JUGE_SYSTEM_PROMPT, prompt)
        payload = parse_json_response(raw)
        if payload is None:
            if attempt == 1:
                append_call(
                    calls_path,
                    {
                        "horodatage": _iso_now(),
                        "role": "juge",
                        "id": invite_key,
                        "question_id": question["id"],
                        "sha256_invite": sha256_text(user),
                        "modele_demande": modele_demande,
                        "modele_repondu": meta.get("model_returned"),
                        "fournisseur": meta.get("fournisseur"),
                        "finish_reason": meta.get("finish_reason"),
                        "usage": meta.get("usage"),
                        "reponse_brute": raw,
                        "tentative": attempt,
                        "final": False,
                    },
                )
                attempt += 1
                continue
            validated = {"ok": True, "jugements": [], "refus": False, "illisible": True}
            break
        validated = validate_judge_response(payload, response_text=response_text, expected_count=len(claims))
        if validated["ok"]:
            break
        if attempt == 1:
            append_call(
                calls_path,
                {
                    "horodatage": _iso_now(),
                    "role": "juge",
                    "id": invite_key,
                    "question_id": question["id"],
                    "sha256_invite": sha256_text(user),
                    "modele_demande": modele_demande,
                    "modele_repondu": meta.get("model_returned"),
                    "fournisseur": meta.get("fournisseur"),
                    "finish_reason": meta.get("finish_reason"),
                    "usage": meta.get("usage"),
                    "reponse_brute": raw,
                    "tentative": attempt,
                    "final": False,
                },
            )
            attempt += 1
            continue
        validated = {"ok": True, "jugements": [], "refus": False, "illisible": True}
        break
    append_call(
        calls_path,
        {
            "horodatage": _iso_now(),
            "role": "juge",
            "id": invite_key,
            "question_id": question["id"],
            "sha256_invite": sha256_text(user),
            "modele_demande": modele_demande,
            "modele_repondu": meta.get("model_returned"),
            "fournisseur": meta.get("fournisseur"),
            "finish_reason": meta.get("finish_reason"),
            "usage": meta.get("usage"),
            "reponse_brute": raw,
            "jugements": validated.get("jugements") if validated else [],
            "refus": validated.get("refus", False) if validated else False,
            "illisible": validated.get("illisible", False) if validated else True,
            "tentative": attempt,
            "final": True,
        },
    )
    return validated or {"ok": True, "jugements": [], "refus": False, "illisible": True}


def run_reponses(chat_fn: ChatFn, *, calls_path: Path | None = None) -> dict[str, Any]:
    calls_path = calls_path or APPELS_PATH
    invites = load_invites_reponses()
    completed, invite_hashes = load_journal_state(calls_path)
    stability = {invite_id("C", qid) for _bras, qid in stability_pairs(load_questions())}
    for invite in invites:
        ident = invite["id"]
        if ident in invite_hashes:
            verify_invite_hash(ident, invite_hashes[ident], invite)
        if ident in completed:
            continue
        adjudicate_response(invite, chat_fn=chat_fn, calls_path=calls_path)
        if ident in stability:
            rejeu_id = stability_rejeu_id(ident)
            if rejeu_id in invite_hashes:
                verify_invite_hash(rejeu_id, invite_hashes[rejeu_id], invite)
            if rejeu_id not in completed:
                adjudicate_response(
                    invite,
                    chat_fn=chat_fn,
                    calls_path=calls_path,
                    record_id=rejeu_id,
                )
    return {"status": "reponses", "appels": count_calls(calls_path)}


def _response_observations(calls_path: Path, base_id: str) -> list[dict[str, Any]]:
    rejeu_id = stability_rejeu_id(base_id)
    base_records: list[dict[str, Any]] = []
    rejeu_records: list[dict[str, Any]] = []
    for record in load_journal_records(calls_path):
        if record.get("role") != "reponse" or not record.get("final", True):
            continue
        ident = str(record["id"])
        if ident == base_id:
            base_records.append(record)
        elif ident == rejeu_id:
            rejeu_records.append(record)
    if rejeu_records:
        primary = base_records[0] if base_records else None
        secondary = rejeu_records[0]
        return [primary, secondary] if primary else [secondary]
    if len(base_records) >= 2:
        return base_records[:2]
    return base_records[:1]


def _latest_responses(calls_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in load_journal_records(calls_path):
        if record.get("role") != "reponse" or not record.get("final", True):
            continue
        ident = str(record["id"])
        if ident.endswith("#rejeu"):
            continue
        if ident not in latest:
            latest[ident] = record
    return latest


def run_juge(chat_fn: ChatFn, *, calls_path: Path | None = None) -> dict[str, Any]:
    calls_path = calls_path or APPELS_PATH
    invites = load_invites_reponses()
    questions = {q["id"]: q for q in load_questions()}
    responses = _latest_responses(calls_path)
    completed, _invite_hashes = load_journal_state(calls_path)
    for invite in invites:
        juge_id = f"juge-{invite['id']}"
        if juge_id in completed:
            continue
        if invite["id"] not in responses:
            continue
        response_text = str(responses[invite["id"]].get("reponse_brute") or "")
        adjudicate_judge(
            response_text,
            questions[invite["question_id"]],
            invite_key=juge_id,
            chat_fn=chat_fn,
            calls_path=calls_path,
        )
    return {"status": "juge", "appels": count_calls(calls_path)}


def _latest_judges(calls_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in load_journal_records(calls_path):
        if record.get("role") != "juge":
            continue
        ident = str(record["id"])
        if record.get("final", True):
            latest[ident] = record
    return latest


def _build_tableaux(
    notes: list[dict[str, Any]],
    *,
    bras_b_ids: set[str] | None = None,
) -> dict[str, Any]:
    tableaux: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "141": defaultdict(lambda: defaultdict(dict)),
        "bras_B": defaultdict(lambda: defaultdict(dict)),
    }
    for note in notes:
        shape = str(note.get("shape") or "unknown")
        bras = str(note["bras"])
        for scope, subset in (("141", True), ("bras_B", bras_b_ids and note["question_id"] in bras_b_ids)):
            if not subset:
                continue
            cell = tableaux[scope][shape][bras]
            if not cell:
                cell.update(
                    {
                        "conforme": 0,
                        "incomplete": 0,
                        "fautive": 0,
                        "a_verifier": 0,
                        "indetermine": 0,
                        "tous_les_nombres_true": 0,
                        "tous_les_nombres_attendus": 0,
                    }
                )
            verdict = str(note["verdict"])
            if verdict in cell:
                cell[verdict] += 1
            if note.get("tous_les_nombres") is not None:
                cell["tous_les_nombres_attendus"] += 1
                if note["tous_les_nombres"]:
                    cell["tous_les_nombres_true"] += 1
    result: dict[str, Any] = {}
    for scope, shapes in tableaux.items():
        result[scope] = {}
        for shape, by_bras in sorted(shapes.items()):
            result[scope][shape] = {}
            for bras, cell in sorted(by_bras.items()):
                attendus = cell.get("tous_les_nombres_attendus", 0)
                vrais = cell.get("tous_les_nombres_true", 0)
                result[scope][shape][bras] = {
                    "conforme": cell.get("conforme", 0),
                    "incomplete": cell.get("incomplete", 0),
                    "fautive": cell.get("fautive", 0),
                    "a_verifier": cell.get("a_verifier", 0),
                    "indetermine": cell.get("indetermine", 0),
                    "tous_les_nombres": f"{vrais}/{attendus}" if attendus else "0/0",
                }
    return result


def compute_stabilite(
    calls_path: Path,
    questions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    questions = questions or load_questions()
    qmap = {q["id"]: q for q in questions}
    pairs = stability_pairs(questions)
    text_identical = 0
    nombres_identical = 0
    total = 0
    for _bras, qid in pairs:
        base_id = invite_id("C", qid)
        observations = _response_observations(calls_path, base_id)
        if len(observations) < 2:
            continue
        total += 1
        t1 = str(observations[0].get("reponse_brute") or "")
        t2 = str(observations[1].get("reponse_brute") or "")
        if t1 == t2:
            text_identical += 1
        question = qmap[qid]
        n1 = compute_nombres_fields(question, t1)["tous_les_nombres"]
        n2 = compute_nombres_fields(question, t2)["tous_les_nombres"]
        if n1 == n2:
            nombres_identical += 1
    return {
        "couples": total,
        "reponses_identiques": f"{text_identical}/{total}",
        "taux_reponses_identiques": (text_identical / total) if total else None,
        "tous_les_nombres_identiques": f"{nombres_identical}/{total}",
        "taux_tous_les_nombres_identiques": (nombres_identical / total) if total else None,
    }


def verify_controle_nombres(tableaux: dict[str, Any]) -> str:
    scope = tableaux.get("141") or {}
    for shape, expected_by_bras in CONTROLE_NOMBRES_141.items():
        for bras, (numerator, denominator) in expected_by_bras.items():
            actual = scope.get(shape, {}).get(bras, {}).get("tous_les_nombres")
            expected = f"{numerator}/{denominator}"
            if actual != expected:
                return (
                    f"écart {shape}/{bras}: attendu {expected}, obtenu {actual}"
                )
    return "ok"


def run_noter(
    *,
    calls_path: Path | None = None,
    invites_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    calls_path = calls_path or APPELS_PATH
    invites = load_invites_reponses(invites_path)
    questions = {q["id"]: q for q in load_questions()}
    bras_b = load_or_create_bras_b_questions()
    bras_b_ids = set(bras_b["question_ids"])
    responses = _latest_responses(calls_path)
    juges = _latest_judges(calls_path)
    notes: list[dict[str, Any]] = []
    manquants: list[str] = []
    for invite in invites:
        ident = invite["id"]
        if ident not in responses or f"juge-{ident}" not in juges:
            manquants.append(ident)
            continue
        question = questions[invite["question_id"]]
        response_text = str(responses[ident].get("reponse_brute") or "")
        judge = juges[f"juge-{ident}"]
        verdict = compute_verdict(question, response_text, judge)
        nombres = compute_nombres_fields(question, response_text)
        scoring = must_mention_scoring_fields(question, response_text)
        notes.append(
            {
                "id": ident,
                "bras": invite["bras"],
                "question_id": invite["question_id"],
                "shape": question.get("shape"),
                "verdict": verdict,
                "must_mention_manquants": scoring["must_mention_manquants"],
                "termes_exemptes": scoring["termes_exemptes"],
                "part_must_mention": scoring["part_must_mention"],
                "nombres_attendus": nombres["nombres_attendus"],
                "nombres_presents": nombres["nombres_presents"],
                "tous_les_nombres": nombres["tous_les_nombres"],
                "refus": judge.get("refus", False),
            }
        )
    tableaux = _build_tableaux(notes, bras_b_ids=bras_b_ids)
    stabilite = compute_stabilite(calls_path)
    controle_nombres = verify_controle_nombres(tableaux)
    payload = {
        "date": date.today().isoformat(),
        "notes": notes,
        "tableaux": tableaux,
        "stabilite": stabilite,
        "controle_nombres": controle_nombres,
        "complet": not manquants,
        "manquants": sorted(manquants),
    }
    out = output_path or (SORTIES_ETAGE3 / "notations.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def run_depuis_journal(*, calls_path: Path | None = None, invites_path: Path | None = None) -> dict[str, Any]:
    calls_path = calls_path or APPELS_PATH
    invites = load_invites_reponses(invites_path)
    _, invite_hashes = load_journal_state(calls_path)
    for invite in invites:
        if invite["id"] in invite_hashes:
            verify_invite_hash(invite["id"], invite_hashes[invite["id"]], invite)
    payload = run_noter(calls_path=calls_path, invites_path=invites_path)
    return {
        "status": "depuis_journal",
        "traites": len(payload["notes"]),
        "complet": payload["complet"],
        "controle_nombres": payload.get("controle_nombres"),
        "stabilite": payload.get("stabilite"),
    }


def instructions_identical(invites: list[dict[str, Any]]) -> bool:
    systems = {invite["system"] for invite in invites}
    return len(systems) == 1 and systems.pop() == SYSTEM_PROMPT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Étage 3 — réponses et notation.")
    parser.add_argument("--dry-run", action="store_true", help="Marche à blanc : invites sans appel modèle.")
    parser.add_argument("--reponses", action="store_true", help="Collecter les réponses via chat_fn injecté.")
    parser.add_argument("--juge", action="store_true", help="Juger must_not_claim via chat_fn injecté.")
    parser.add_argument("--noter", action="store_true", help="Calculer les verdicts depuis le journal.")
    parser.add_argument(
        "--depuis-journal",
        action="store_true",
        help="Régénérer les notations depuis le journal sans appel modèle.",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        result = run_dry_run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.depuis_journal:
        result = run_depuis_journal()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.reponses:
        result = run_reponses(make_relay_chat_fn())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.juge:
        result = run_juge(make_relay_chat_fn())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.noter:
        result = run_noter()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
