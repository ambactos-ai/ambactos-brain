# SPDX-License-Identifier: AGPL-3.0-only
"""Second lecteur aveugle (lecteur B) — invites, appels injectés, marche à blanc."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .relay import MODEL, RELAY_URL

BANC_ROOT = (BANC_VAR)
SORTIES_DIR = BANC_ROOT / "sorties" / "ingestion-2"
DONNEES_PATH = SORTIES_DIR / "relecture-donnees.json"
PAIRES_PATH = SORTIES_DIR / "etage2-paires-a-lire.json"
INVITES_PATH = SORTIES_DIR / "lecteur-b-invites.jsonl"
CALLS_PATH = SORTIES_DIR / "lecteur-b-calls.jsonl"
VERDICTS_RELECTURE_PATH = SORTIES_DIR / "relecture-verdicts-lecteur-b.json"
VERDICTS_PAIRES_PATH = SORTIES_DIR / "paires-verdicts-lecteur-b.json"

SEED = 20260920
MAX_CALLS = 140
PROVIDER = "hermes_relay"

ChatFn = Callable[[str, str], tuple[str, dict[str, Any]]]

SYSTEM_PROMPT = (
    "Tu es un relecteur indépendant. Réponds uniquement en JSON strict, sans texte autour."
)

GRILLE_RESIDU = """\
L'assertion de l'or est-elle tenue par la mémoire affichée ?
1. Tenue, autoportante — un fait dit la même chose et se comprend seul (ses qualifiers comptent : une année portée par period, un site porté par site font partie du fait).
2. Tenue, mais fragment — un fait le dit, mais on ne peut pas le rattacher sans relire la source : il lui manque le sujet, la date ou qui parle.
3. Partielle — une partie y est, une information est perdue. Dire laquelle.
4. Absente — rien dans la mémoire affichée ne le dit.
Tu ne juges que les huit faits affichés."""

GRILLE_HORS_OR = """\
La source dit-elle ce fait ?
V — vrai : la source le dit.
3 — vrai mais fragment : exact, mais incompréhensible sans la source.
4 — faux ou déformé : la source ne dit pas cela.
5 — bruit : titre, mention de mise en page, métadonnée sans valeur, artefact d'outil."""

GRILLE_ASSERTION = """\
Lesquels de ces faits, seuls ou ensemble, disent ce que dit l'assertion ?
1 — tient
3 — tient en partie
4 — ne tient pas"""

BAREME_RESIDU = {
    "1": "tenue, autoportante",
    "2": "tenue, mais fragment",
    "3": "partielle",
    "4": "absente",
}
BAREME_HORS_OR = {
    "V": "vrai",
    "3": "vrai mais fragment",
    "4": "faux ou déformé",
    "5": "bruit",
}
BAREME_ASSERTION = {
    "1": "tient",
    "3": "tient en partie",
    "4": "ne tient pas",
}

FORBIDDEN_INVITE_TOKENS = ("voie", "modele", "tabulaire", "claim_id", "extracted_by_")

FORMAT_CONSIGNE_RESIDU = (
    '{"verdict":"1"|"2"|"3"|"4","lignes":[<numéros de ligne de l\'extrait>],'
    '"manque":"…","justification":"≤ 25 mots"}'
)
FORMAT_CONSIGNE_HORS_OR = (
    '{"verdict":"V"|"3"|"4"|"5","lignes":[<numéros de ligne de l\'extrait>],'
    '"manque":"","justification":"≤ 25 mots"}'
)
FORMAT_CONSIGNE_ASSERTION = (
    '{"verdict":"1"|"3"|"4","faits":[<numéros>],"manque":"…","justification":"≤ 25 mots"}'
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_qualifiers(qualifiers: dict[str, Any] | None) -> str:
    if not qualifiers:
        return ""
    parts = [f"{key} : {value}" for key, value in sorted(qualifiers.items())]
    return " (" + " ; ".join(parts) + ")"


def format_fact_line(fact: dict[str, Any]) -> str:
    label = fact.get("triplet") or fact.get("texte") or ""
    return f"{label}{format_qualifiers(fact.get('autres_qualifiers'))}"


def format_numbered_fact_block(numero: int, text: str) -> list[str]:
    lines = text.split("\n")
    block = [f"{numero}. {lines[0]}"]
    for continuation in lines[1:]:
        block.append(f"    {continuation}")
    return block


def delim_block(name: str, content: str) -> str:
    return f"<<<<<{name}>>>>>\n{content}\n<<<<<FIN_{name}>>>>>"


def relecture_ids(data: dict[str, Any]) -> list[tuple[str, str, int]]:
    items: list[tuple[str, str, int]] = []
    for idx in range(len(data["residu"])):
        items.append((f"R{idx + 1:02d}", "residu", idx))
    for idx in range(len(data["hors_or"])):
        items.append((f"H{idx + 1:02d}", "hors_or", idx))
    return items


def load_relecture_donnees(path: Path = DONNEES_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_residu_user(item: dict[str, Any]) -> str:
    lines = [
        GRILLE_RESIDU,
        "",
        f"Pièce : {item['piece']}",
        f"Page : {item.get('page')}",
        delim_block("CONTENT", item.get("content") or ""),
        delim_block("QUOTE", item.get("quote") or ""),
        delim_block("EXTRAIT_SOURCE", item.get("extrait_source") or ""),
        "",
        "Ce que la mémoire a de plus proche :",
    ]
    for index, neighbor in enumerate(item.get("voisins") or [], start=1):
        lines.extend(format_numbered_fact_block(index, format_fact_line(neighbor)))
    lines.append("")
    lines.append(f"Réponds en JSON strict : {FORMAT_CONSIGNE_RESIDU}")
    return "\n".join(lines)


def build_hors_or_user(item: dict[str, Any]) -> str:
    lines = [
        GRILLE_HORS_OR,
        "",
        f"Pièce : {item['piece']}",
        format_fact_line(item),
        f"Lignes source : {item.get('lignes_source') or ''}",
        delim_block("EXTRAIT_SOURCE", item.get("extrait_source") or ""),
        "",
        f"Réponds en JSON strict : {FORMAT_CONSIGNE_HORS_OR}",
    ]
    return "\n".join(lines)


def group_assertions_from_paires(paires: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    facts_seen: dict[str, set[str]] = defaultdict(set)
    for row in paires:
        assertion = row["assertion"]
        aid = assertion["id"]
        if aid not in grouped:
            grouped[aid] = {
                "id": aid,
                "piece": assertion.get("piece") or "",
                "content": assertion.get("content") or "",
                "quote": assertion.get("quote") or "",
                "extrait": assertion.get("extrait") or "",
                "faits": [],
            }
        claim_id = str(row["fait"]["claim_id"])
        if claim_id in facts_seen[aid]:
            continue
        facts_seen[aid].add(claim_id)
        grouped[aid]["faits"].append(
            {
                "numero": len(grouped[aid]["faits"]) + 1,
                "triplet": row["fait"].get("triplet") or row["fait"].get("texte") or "",
            }
        )
    return [grouped[aid] for aid in sorted(grouped)]


def build_assertion_user(item: dict[str, Any]) -> str:
    lines = [
        GRILLE_ASSERTION,
        "",
        f"Pièce : {item['piece']}",
        delim_block("CONTENT", item.get("content") or ""),
        delim_block("QUOTE", item.get("quote") or ""),
        delim_block("EXTRAIT", item.get("extrait") or ""),
        "",
        "Faits candidats :",
    ]
    for fact in item["faits"]:
        lines.extend(format_numbered_fact_block(fact["numero"], fact["triplet"]))
    lines.append("")
    lines.append(f"Réponds en JSON strict : {FORMAT_CONSIGNE_ASSERTION}")
    return "\n".join(lines)


def build_invites(
    *,
    donnees_path: Path = DONNEES_PATH,
    paires_path: Path = PAIRES_PATH,
) -> list[dict[str, Any]]:
    data = load_relecture_donnees(donnees_path)
    invites: list[dict[str, Any]] = []
    for ident, partie, index in relecture_ids(data):
        if partie == "residu":
            user = build_residu_user(data["residu"][index])
        else:
            user = build_hors_or_user(data["hors_or"][index])
        invites.append(
            {
                "id": ident,
                "partie": partie,
                "sha256": sha256_text(user),
                "system": SYSTEM_PROMPT,
                "user": user,
            }
        )
    paires = json.loads(paires_path.read_text(encoding="utf-8"))
    for item in group_assertions_from_paires(paires):
        user = build_assertion_user(item)
        invites.append(
            {
                "id": f"A-{item['id']}",
                "partie": "assertion",
                "sha256": sha256_text(user),
                "system": SYSTEM_PROMPT,
                "user": user,
            }
        )
    return invites


def extract_line_numbers(excerpt: str) -> set[int]:
    numbers: set[int] = set()
    for match in re.finditer(r"^\s*(?:→\s*)?(\d{1,5})\s{2,}", excerpt, flags=re.MULTILINE):
        numbers.add(int(match.group(1)))
    return numbers


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


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def validate_relecture_response(
    payload: dict[str, Any],
    *,
    partie: str,
    excerpt: str,
) -> dict[str, Any]:
    alphabet = set(BAREME_RESIDU) if partie == "residu" else set(BAREME_HORS_OR)
    verdict = str(payload.get("verdict", "")).strip()
    if verdict not in alphabet:
        return {"ok": False, "reason": "verdict_hors_alphabet"}
    lignes_raw = payload.get("lignes")
    if not isinstance(lignes_raw, list):
        return {"ok": False, "reason": "lignes_invalides"}
    lignes: list[int] = []
    for item in lignes_raw:
        try:
            lignes.append(int(item))
        except (TypeError, ValueError):
            return {"ok": False, "reason": "lignes_invalides"}
    manque = str(payload.get("manque") or "")
    justification = str(payload.get("justification") or "")
    if word_count(justification) > 25:
        return {"ok": False, "reason": "justification_trop_longue"}
    if partie == "residu" and verdict != "3" and manque:
        manque = ""
    if partie == "residu" and verdict == "3" and not manque.strip():
        return {"ok": False, "reason": "manque_requis"}
    valid_lines = extract_line_numbers(excerpt)
    preuve_verifiable = all(num in valid_lines for num in lignes)
    return {
        "ok": True,
        "verdict": verdict,
        "lignes": lignes,
        "manque": manque,
        "justification": justification,
        "preuve_verifiable": preuve_verifiable,
    }


def validate_assertion_response(payload: dict[str, Any], *, max_fact: int) -> dict[str, Any]:
    verdict = str(payload.get("verdict", "")).strip()
    if verdict not in BAREME_ASSERTION:
        return {"ok": False, "reason": "verdict_hors_alphabet"}
    faits_raw = payload.get("faits")
    if not isinstance(faits_raw, list):
        return {"ok": False, "reason": "faits_invalides"}
    faits: list[int] = []
    for item in faits_raw:
        try:
            num = int(item)
        except (TypeError, ValueError):
            return {"ok": False, "reason": "faits_invalides"}
        if num < 1 or num > max_fact:
            return {"ok": False, "reason": "faits_hors_plage"}
        faits.append(num)
    manque = str(payload.get("manque") or "")
    justification = str(payload.get("justification") or "")
    if word_count(justification) > 25:
        return {"ok": False, "reason": "justification_trop_longue"}
    return {
        "ok": True,
        "verdict": verdict,
        "faits": faits,
        "manque": manque,
        "justification": justification,
        "preuve_verifiable": True,
    }


def retry_prompt(partie: str) -> str:
    if partie == "assertion":
        return (
            "Ta réponse n'était pas un JSON valide. Réponds uniquement avec : "
            '{"verdict":"1"|"3"|"4","faits":[<numéros>],"manque":"…","justification":"≤ 25 mots"}'
        )
    if partie == "residu":
        return (
            "Ta réponse n'était pas un JSON valide. Réponds uniquement avec : "
            '{"verdict":"1"|"2"|"3"|"4","lignes":[<numéros>],"manque":"…","justification":"≤ 25 mots"}'
        )
    return (
        "Ta réponse n'était pas un JSON valide. Réponds uniquement avec : "
        '{"verdict":"V"|"3"|"4"|"5","lignes":[<numéros>],"manque":"","justification":"≤ 25 mots"}'
    )


def partie_for_id(ident: str) -> str:
    if ident.startswith("A-"):
        return "assertion"
    if ident.startswith("H"):
        return "hors_or"
    return "residu"


def fields_from_parsed_response(
    payload: dict[str, Any] | None,
    *,
    partie: str,
    fallback_verdict: str | None = None,
) -> dict[str, Any]:
    if payload is None:
        return {
            "verdict": fallback_verdict or "illisible",
            "manque": "",
            "justification": "",
            "faits": [],
            "preuve_verifiable": False,
        }
    if partie == "assertion":
        return {
            "verdict": str(payload.get("verdict", fallback_verdict or "illisible")),
            "manque": str(payload.get("manque") or ""),
            "justification": str(payload.get("justification") or ""),
            "faits": list(payload.get("faits") or []),
            "preuve_verifiable": True,
        }
    return {
        "verdict": str(payload.get("verdict", fallback_verdict or "illisible")),
        "manque": str(payload.get("manque") or ""),
        "justification": str(payload.get("justification") or ""),
        "faits": [],
        "preuve_verifiable": bool(payload.get("preuve_verifiable", True)),
    }


def load_journal_records(calls_path: Path) -> list[dict[str, Any]]:
    if not calls_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in calls_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("identifiant"):
            records.append(record)
    return records


def load_journal_state(calls_path: Path) -> tuple[set[str], dict[str, str]]:
    completed: set[str] = set()
    invite_hashes: dict[str, str] = {}
    last_by_id: dict[str, dict[str, Any]] = {}
    for record in load_journal_records(calls_path):
        ident = str(record["identifiant"])
        last_by_id[ident] = record
    for ident, record in last_by_id.items():
        invite_hashes[ident] = str(record.get("sha256_invite") or "")
        if record.get("final", True):
            completed.add(ident)
    return completed, invite_hashes


def load_existing_call_ids(calls_path: Path) -> set[str]:
    return load_journal_state(calls_path)[0]


def ensure_calls_header(calls_path: Path) -> None:
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    if calls_path.is_file() and calls_path.stat().st_size > 0:
        return
    with calls_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "fournisseur": PROVIDER,
                    "modele_demande": MODEL,
                    "relay_url": RELAY_URL,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def append_call(calls_path: Path, record: dict[str, Any]) -> None:
    with calls_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_results_from_calls(calls_path: Path) -> dict[str, dict[str, Any]]:
    last_by_id: dict[str, dict[str, Any]] = {}
    for record in load_journal_records(calls_path):
        last_by_id[str(record["identifiant"])] = record
    latest: dict[str, dict[str, Any]] = {}
    for ident, record in last_by_id.items():
        if not record.get("final", True):
            continue
        partie = partie_for_id(ident)
        stored = {
            "verdict": record.get("verdict"),
            "preuve_verifiable": record.get("preuve_verifiable", False),
            "manque": record.get("manque", ""),
            "justification": record.get("justification", ""),
            "faits": record.get("faits") or [],
        }
        if not stored["justification"] and record.get("reponse_brute"):
            parsed = parse_json_response(str(record["reponse_brute"]))
            extracted = fields_from_parsed_response(
                parsed,
                partie=partie,
                fallback_verdict=str(record.get("verdict") or "illisible"),
            )
            stored.update(extracted)
            if record.get("preuve_verifiable") is not None:
                stored["preuve_verifiable"] = record.get("preuve_verifiable", False)
        latest[ident] = stored
    return latest


def count_calls(calls_path: Path) -> int:
    return len(load_journal_records(calls_path))


def assert_call_budget(calls_path: Path) -> None:
    if count_calls(calls_path) >= MAX_CALLS:
        raise RuntimeError(f"plafond d'appels atteint ({MAX_CALLS})")


def verify_invite_hash(ident: str, journal_hash: str, invite: dict[str, Any]) -> None:
    current = invite["sha256"]
    if journal_hash and journal_hash != current:
        raise RuntimeError(
            f"invite modifiée pour {ident}: sha256 journal {journal_hash} != courant {current}"
        )


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
            "finish_reason": finish_reason,
            "usage": usage,
        }

    return chat


def adjudicate_one(
    invite: dict[str, Any],
    *,
    chat_fn: ChatFn,
    calls_path: Path,
    data: dict[str, Any],
    paires_grouped: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ident = invite["id"]
    partie = invite["partie"]
    attempt = 1
    user = invite["user"]
    raw = ""
    meta: dict[str, Any] = {}
    validated: dict[str, Any] | None = None

    while attempt <= 2:
        assert_call_budget(calls_path)
        raw, meta = chat_fn(SYSTEM_PROMPT, user)
        payload = parse_json_response(raw)
        if payload is None:
            if attempt == 1:
                append_call(
                    calls_path,
                    {
                        "horodatage": _iso_now(),
                        "identifiant": ident,
                        "sha256_invite": invite["sha256"],
                        "modele_demande": MODEL,
                        "modele_repondu": meta.get("model_returned"),
                        "finish_reason": meta.get("finish_reason"),
                        "usage": meta.get("usage"),
                        "reponse_brute": raw,
                        "verdict": None,
                        "manque": "",
                        "justification": "",
                        "faits": [],
                        "preuve_verifiable": False,
                        "tentative": attempt,
                        "final": False,
                    },
                )
                user = invite["user"] + "\n\n" + retry_prompt(partie)
                attempt += 1
                continue
            validated = {
                "ok": True,
                "verdict": "illisible",
                "lignes": [],
                "manque": "",
                "justification": "",
                "faits": [],
                "preuve_verifiable": False,
            }
            append_call(
                calls_path,
                {
                    "horodatage": _iso_now(),
                    "identifiant": ident,
                    "sha256_invite": invite["sha256"],
                    "modele_demande": MODEL,
                    "modele_repondu": meta.get("model_returned"),
                    "finish_reason": meta.get("finish_reason"),
                    "usage": meta.get("usage"),
                    "reponse_brute": raw,
                    "verdict": "illisible",
                    "manque": "",
                    "justification": "",
                    "faits": [],
                    "preuve_verifiable": False,
                    "tentative": attempt,
                    "final": True,
                },
            )
            break
        if partie == "assertion":
            assertion_id = ident[2:] if ident.startswith("A-") else ident
            max_fact = len(paires_grouped[assertion_id]["faits"])
            validated = validate_assertion_response(payload, max_fact=max_fact)
        else:
            index = int(ident[1:]) - 1
            if partie == "residu":
                excerpt = data["residu"][index].get("extrait_source") or ""
            else:
                excerpt = data["hors_or"][index].get("extrait_source") or ""
            validated = validate_relecture_response(payload, partie=partie, excerpt=excerpt)
        if validated["ok"]:
            append_call(
                calls_path,
                {
                    "horodatage": _iso_now(),
                    "identifiant": ident,
                    "sha256_invite": invite["sha256"],
                    "modele_demande": MODEL,
                    "modele_repondu": meta.get("model_returned"),
                    "finish_reason": meta.get("finish_reason"),
                    "usage": meta.get("usage"),
                    "reponse_brute": raw,
                    "verdict": validated["verdict"],
                    "manque": validated.get("manque", ""),
                    "justification": validated.get("justification", ""),
                    "faits": validated.get("faits") or [],
                    "preuve_verifiable": validated.get("preuve_verifiable", False),
                    "tentative": attempt,
                    "final": True,
                },
            )
            break
        if attempt == 1:
            append_call(
                calls_path,
                {
                    "horodatage": _iso_now(),
                    "identifiant": ident,
                    "sha256_invite": invite["sha256"],
                    "modele_demande": MODEL,
                    "modele_repondu": meta.get("model_returned"),
                    "finish_reason": meta.get("finish_reason"),
                    "usage": meta.get("usage"),
                    "reponse_brute": raw,
                    "verdict": None,
                    "manque": str(payload.get("manque") or ""),
                    "justification": str(payload.get("justification") or ""),
                    "faits": list(payload.get("faits") or []),
                    "preuve_verifiable": False,
                    "tentative": attempt,
                    "final": False,
                },
            )
            user = invite["user"] + "\n\n" + retry_prompt(partie)
            attempt += 1
            continue
        validated = {
            "ok": True,
            "verdict": "illisible",
            "lignes": [],
            "manque": "",
            "justification": "",
            "faits": [],
            "preuve_verifiable": False,
        }
        append_call(
            calls_path,
            {
                "horodatage": _iso_now(),
                "identifiant": ident,
                "sha256_invite": invite["sha256"],
                "modele_demande": MODEL,
                "modele_repondu": meta.get("model_returned"),
                "finish_reason": meta.get("finish_reason"),
                "usage": meta.get("usage"),
                "reponse_brute": raw,
                "verdict": "illisible",
                "manque": "",
                "justification": "",
                "faits": [],
                "preuve_verifiable": False,
                "tentative": attempt,
                "final": True,
            },
        )
        break

    if validated is None:
        validated = {
            "ok": True,
            "verdict": "illisible",
            "manque": "",
            "justification": "",
            "faits": [],
            "preuve_verifiable": False,
        }
    return validated


def invites_for_parties(
    invites: list[dict[str, Any]],
    *,
    parties: tuple[str, ...],
    limite: int | None = None,
) -> list[dict[str, Any]]:
    selected = [invite for invite in invites if invite["partie"] in parties]
    if limite is not None:
        selected = selected[:limite]
    return selected


def expected_scope_ids(
    invites: list[dict[str, Any]],
    *,
    parties: tuple[str, ...],
) -> list[str]:
    return [invite["id"] for invite in invites if invite["partie"] in parties]


def write_verdict_outputs(
    *,
    data: dict[str, Any],
    paires_grouped: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    relecture_path: Path,
    paires_path: Path,
    scope_ids: list[str] | None = None,
) -> None:
    residu_out: dict[str, list[Any]] = {}
    hors_or_out: dict[str, list[Any]] = {}
    illisibles: list[str] = []
    preuves_non_verifiables: list[str] = []
    compte_residu: dict[str, int] = defaultdict(int)
    compte_hors_or: dict[str, int] = defaultdict(int)
    manquants: list[str] = []

    relecture_scope = {
        ident
        for ident in (scope_ids or [])
        if ident.startswith("R") or ident.startswith("H")
    }
    if scope_ids is None:
        relecture_scope = {ident for ident, _partie, _index in relecture_ids(data)}

    for ident, partie, _index in relecture_ids(data):
        if ident not in relecture_scope:
            continue
        if ident not in results:
            manquants.append(ident)
            continue
        result = results[ident]
        verdict = str(result.get("verdict", "illisible"))
        note = result.get("manque") or result.get("justification") or ""
        if verdict == "illisible":
            illisibles.append(ident)
        elif not result.get("preuve_verifiable", True):
            preuves_non_verifiables.append(ident)
        if partie == "residu":
            residu_out[ident] = [verdict, note]
            compte_residu[verdict] += 1
        else:
            hors_or_out[ident] = [verdict, note]
            compte_hors_or[verdict] += 1

    expected_relecture = sorted(relecture_scope)
    relecture_complet = not manquants and len(residu_out) + len(hors_or_out) == len(expected_relecture)

    relecture_path.write_text(
        json.dumps(
            {
                "annotateur": "grok-4.5 par le relais local, lecteur B, aveugle au lecteur A et à la voie",
                "date": date.today().isoformat(),
                "graine": SEED,
                "bareme_residu": BAREME_RESIDU,
                "bareme_hors_or": BAREME_HORS_OR,
                "residu": residu_out,
                "hors_or": hors_or_out,
                "compte_residu": dict(compte_residu),
                "compte_hors_or": dict(compte_hors_or),
                "illisibles": illisibles,
                "preuves_non_verifiables": preuves_non_verifiables,
                "complet": relecture_complet,
                "manquants": sorted(manquants),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    par_assertion: dict[str, Any] = {}
    paires_rows: list[dict[str, Any]] = []
    compte_assertions: dict[str, int] = defaultdict(int)
    illisibles_assertions: list[str] = []
    assertion_scope = {ident for ident in (scope_ids or []) if ident.startswith("A-")}
    if scope_ids is None:
        assertion_scope = {f"A-{aid}" for aid in paires_grouped}
    manquants_assertions: list[str] = []

    for aid in sorted(paires_grouped):
        ident = f"A-{aid}"
        if ident not in assertion_scope:
            continue
        if ident not in results:
            manquants_assertions.append(ident)
            continue
        result = results[ident]
        verdict = str(result.get("verdict", "illisible"))
        note = result.get("manque") or result.get("justification") or ""
        if verdict == "illisible":
            illisibles_assertions.append(ident)
        par_assertion[aid] = verdict if verdict != "illisible" else "illisible"
        compte_assertions[verdict] += 1
        paires_rows.append(
            {
                "id": ident,
                "assertion": aid,
                "verdict": verdict,
                "faits": result.get("faits") or [],
                "note": note,
            }
        )

    paires_path.write_text(
        json.dumps(
            {
                "annotateur": "grok-4.5 par le relais local, lecteur B, aveugle au lecteur A",
                "date": date.today().isoformat(),
                "graine": SEED,
                "bareme": BAREME_ASSERTION,
                "par_assertion": par_assertion,
                "paires": paires_rows,
                "compte_assertions": dict(compte_assertions),
                "illisibles": illisibles_assertions,
                "complet": not manquants_assertions and len(par_assertion) == len(assertion_scope),
                "manquants": sorted(manquants_assertions),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run(
    chat_fn: ChatFn,
    *,
    parties: tuple[str, ...] = ("residu", "hors_or", "assertion"),
    limite: int | None = None,
    dry_run: bool = False,
    donnees_path: Path = DONNEES_PATH,
    paires_path: Path = PAIRES_PATH,
    invites_path: Path = INVITES_PATH,
    calls_path: Path = CALLS_PATH,
    relecture_verdicts_path: Path = VERDICTS_RELECTURE_PATH,
    paires_verdicts_path: Path = VERDICTS_PAIRES_PATH,
) -> dict[str, Any]:
    invites = build_invites(donnees_path=donnees_path, paires_path=paires_path)
    if dry_run:
        invites_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in invites),
            encoding="utf-8",
        )
        return {"status": "dry_run", "invites": len(invites), "path": str(invites_path)}

    data = load_relecture_donnees(donnees_path)
    paires = group_assertions_from_paires(json.loads(paires_path.read_text(encoding="utf-8")))
    paires_grouped = {item["id"]: item for item in paires}
    selected = invites_for_parties(invites, parties=parties, limite=limite)

    ensure_calls_header(calls_path)
    existing, invite_hashes = load_journal_state(calls_path)
    results = load_results_from_calls(calls_path)
    scope_ids = expected_scope_ids(invites, parties=parties)

    for invite in selected:
        ident = invite["id"]
        if ident in invite_hashes:
            verify_invite_hash(ident, invite_hashes[ident], invite)
        if ident in existing:
            continue
        results[ident] = adjudicate_one(
            invite,
            chat_fn=chat_fn,
            calls_path=calls_path,
            data=data,
            paires_grouped=paires_grouped,
        )

    write_verdict_outputs(
        data=data,
        paires_grouped=paires_grouped,
        results=results,
        relecture_path=relecture_verdicts_path,
        paires_path=paires_verdicts_path,
        scope_ids=scope_ids,
    )
    return {
        "status": "complete",
        "appels": count_calls(calls_path),
        "invites_total": len(invites),
        "traites": len(results),
    }


def run_from_journal(
    *,
    donnees_path: Path = DONNEES_PATH,
    paires_path: Path = PAIRES_PATH,
    calls_path: Path = CALLS_PATH,
    relecture_verdicts_path: Path = VERDICTS_RELECTURE_PATH,
    paires_verdicts_path: Path = VERDICTS_PAIRES_PATH,
) -> dict[str, Any]:
    invites = build_invites(donnees_path=donnees_path, paires_path=paires_path)
    data = load_relecture_donnees(donnees_path)
    paires = group_assertions_from_paires(json.loads(paires_path.read_text(encoding="utf-8")))
    paires_grouped = {item["id"]: item for item in paires}
    _, invite_hashes = load_journal_state(calls_path)
    for invite in invites:
        if invite["id"] in invite_hashes:
            verify_invite_hash(invite["id"], invite_hashes[invite["id"]], invite)
    results = load_results_from_calls(calls_path)
    write_verdict_outputs(
        data=data,
        paires_grouped=paires_grouped,
        results=results,
        relecture_path=relecture_verdicts_path,
        paires_path=paires_verdicts_path,
        scope_ids=[invite["id"] for invite in invites],
    )
    return {
        "status": "depuis_journal",
        "appels": count_calls(calls_path),
        "traites": len(results),
        "invites_total": len(invites),
    }


def verify_invites_command(
    *,
    donnees_path: Path = DONNEES_PATH,
    paires_path: Path = PAIRES_PATH,
    lecteur_a_relecture_path: Path | None = None,
) -> dict[str, Any]:
    invites = build_invites(donnees_path=donnees_path, paires_path=paires_path)
    errors = audit_invites_for_blindness(
        invites,
        donnees_path=donnees_path,
        lecteur_a_relecture_path=lecteur_a_relecture_path,
    )
    format_errors = []
    for invite in invites:
        suffix = {
            "residu": FORMAT_CONSIGNE_RESIDU,
            "hors_or": FORMAT_CONSIGNE_HORS_OR,
            "assertion": FORMAT_CONSIGNE_ASSERTION,
        }[invite["partie"]]
        if not invite["user"].rstrip().endswith(suffix):
            format_errors.append(f"{invite['id']}: format de consigne incorrect")
    all_errors = errors + format_errors
    return {
        "status": "ok" if not all_errors else "erreur",
        "invites": len(invites),
        "erreurs": all_errors,
    }


def audit_invites_for_blindness(
    invites: list[dict[str, Any]],
    *,
    donnees_path: Path,
    lecteur_a_relecture_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    data = load_relecture_donnees(donnees_path)
    claim_ids = {
        str(neighbor.get("claim_id"))
        for item in data["residu"]
        for neighbor in item.get("voisins") or []
    }
    claim_ids.update(str(item.get("claim_id")) for item in data["hors_or"])
    justifications_a: set[str] = set()
    if lecteur_a_relecture_path and lecteur_a_relecture_path.is_file():
        lecteur_a = json.loads(lecteur_a_relecture_path.read_text(encoding="utf-8"))
        for section in ("residu", "hors_or"):
            for verdict in lecteur_a.get(section, {}).values():
                if isinstance(verdict, list) and len(verdict) > 1 and verdict[1]:
                    justifications_a.add(str(verdict[1]))
    for invite in invites:
        blob = invite["user"]
        lowered = blob.lower()
        for token in FORBIDDEN_INVITE_TOKENS:
            if token in lowered:
                errors.append(f"{invite['id']}: token interdit « {token} »")
        for claim_id in claim_ids:
            if claim_id and claim_id in blob:
                errors.append(f"{invite['id']}: claim_id visible")
        for justification in justifications_a:
            if justification and justification in blob:
                errors.append(f"{invite['id']}: justification lecteur A visible")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Lecteur B — marche à blanc ou exécution injectée.")
    parser.add_argument("--dry-run", action="store_true", help="Écrire les invites sans appeler de modèle.")
    parser.add_argument(
        "--depuis-journal",
        action="store_true",
        help="Régénérer les verdicts depuis le journal existant, sans appel modèle.",
    )
    parser.add_argument(
        "--verifier-invites",
        action="store_true",
        help="Vérifier l'aveuglement et le format des invites réelles.",
    )
    args = parser.parse_args()
    if args.verifier_invites:
        lecteur_a_path = SORTIES_DIR / "relecture-verdicts-claude.json"
        result = verify_invites_command(
            lecteur_a_relecture_path=lecteur_a_path if lecteur_a_path.is_file() else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["status"] == "ok" else 1)
    if args.depuis_journal:
        result = run_from_journal()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = run(make_relay_chat_fn(), dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
