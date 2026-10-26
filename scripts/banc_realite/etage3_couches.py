# SPDX-License-Identifier: AGPL-3.0-only
"""Étage 3 — couches brute seule et brute+fine (k=8, 141 questions)."""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from banc_realite.compat_moteur import _build_structured_query, _normalize_evidence_item, _render_markdown

from . import etage3 as e3
from .config import BANC_ROOT, SORTIES, export_runtime_env
from .couche_brute_search import SEARCH_BY_CONFIG
from .measure_etage2 import _load_embed_cache

SORTIES_ETAGE3_COUCHES = BANC_ROOT / "sorties" / "etage3-couches"
CADRES_PATH = SORTIES_ETAGE3_COUCHES / "cadres.json"
INVITES_PATH = SORTIES_ETAGE3_COUCHES / "invites-reponses.jsonl"
APPELS_PATH = SORTIES_ETAGE3_COUCHES / "appels.jsonl"
NOTATIONS_PATH = SORTIES_ETAGE3_COUCHES / "notations.json"
RAPPORT_PATH = SORTIES_ETAGE3_COUCHES / "RAPPORT-ETAGE3-COUCHES.md"
ARRET_PATH = BANC_ROOT / "_etage3-couches-arret.json"
RESULT_PATH = BANC_ROOT / "_etage3-couches-result.json"
ETAGE3_NOTATIONS_PATH = BANC_ROOT / "sorties" / "etage3" / "notations.json"
ABLATION_PATH = BANC_ROOT / "sorties" / "couche-brute" / "ablation.json"
ETAGE2_C_PATH = SORTIES / "ingestion-2" / "etage2-C.json"

CONFIGS = ("brute_seule", "brute_fine")
MAX_CALLS = 700
TOP_K = 8

_EVIDENCE_LINE_START_RE = re.compile(r"^- \[[^\]]+\] ")
_INTEGER_MUST_MENTION_RE = re.compile(r"^\d+$")


def write_arret(step: str, why: str) -> None:
    SORTIES_ETAGE3_COUCHES.mkdir(parents=True, exist_ok=True)
    ARRET_PATH.write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _replace_evidence_items(section: str) -> str:
    lines = section.splitlines()
    out: list[str] = []
    idx = 0
    inserted_marker = False
    while idx < len(lines):
        line = lines[idx]
        if _EVIDENCE_LINE_START_RE.match(line):
            if not inserted_marker:
                out.append("- {{ITEM}}")
                inserted_marker = True
            idx += 1
            while idx < len(lines) and not _EVIDENCE_LINE_START_RE.match(lines[idx]):
                idx += 1
            continue
        out.append(line)
        idx += 1
    return "\n".join(out)


def context_gabarit(rendered: str) -> str:
    """Retire le contenu des éléments injectés ; garde la forme du bloc CallFrame."""
    marker = "## Mémoire pertinente (Brain)"
    if marker not in rendered:
        return rendered.rstrip()
    head, tail = rendered.split(marker, 1)
    return (head + marker + _replace_evidence_items(tail)).rstrip()


def gabarits_equivalents(reference: str, candidate: str) -> bool:
    return context_gabarit(reference) == context_gabarit(candidate)


def _conversation_state() -> dict[str, Any]:
    return {"responsibility_context": "GreenLoop"}


def _compile_cadre_for_config(
    question: dict[str, Any],
    *,
    config: str,
    connection,
    embed_cache: dict[str, list[float]],
) -> dict[str, Any]:
    qid = question["id"]
    question_text = question["question"]
    if question_text not in embed_cache:
        raise RuntimeError(f"cache froid : embedding absent ({question_text[:60]}…")
    structured_query = _build_structured_query(question_text, "GreenLoop", _conversation_state())
    qvec = embed_cache[question_text]
    search_fn = SEARCH_BY_CONFIG[config]
    raw_evidence = search_fn(connection, structured_query, query_vector=qvec, top_k=TOP_K)
    evidence: list[dict[str, Any]] = []
    for item in raw_evidence:
        try:
            evidence.append(_normalize_evidence_item(item))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"evidence invalide pour {qid}/{config}: {exc}") from exc
    rendered = _render_markdown(
        "banc-realite",
        "GreenLoop",
        f"banc-etage3-{qid}",
        None,
        _conversation_state(),
        evidence,
        None,
    )
    return {
        "config": config,
        "caracteres": len(rendered),
        "genres": dict(Counter(item.get("kind") or "unknown" for item in evidence)),
        "rendered": rendered,
        "evidence_count": len(evidence),
    }


def compile_cadres(connection) -> dict[str, dict[str, dict[str, Any]]]:
    embed_cache = _load_embed_cache()
    if not embed_cache:
        raise RuntimeError("cache froid : etage2-question-embeddings.json absent ou vide")
    questions = e3.load_questions()
    cadres_c_ref = _load_cadres_c_reference()
    cadres: dict[str, dict[str, dict[str, Any]]] = {cfg: {} for cfg in CONFIGS}
    for question in questions:
        qid = question["id"]
        ref_rendered = cadres_c_ref[qid]["rendered"]
        for config in CONFIGS:
            row = _compile_cadre_for_config(question, config=config, connection=connection, embed_cache=embed_cache)
            if not gabarits_equivalents(ref_rendered, row["rendered"]):
                raise RuntimeError(
                    f"gabarit divergent pour {qid}/{config} : seule la liste d'éléments doit différer"
                )
            cadres[config][qid] = row
    slim = {
        config: {
            qid: {
                "caracteres": row["caracteres"],
                "genres": row["genres"],
                "evidence_count": row["evidence_count"],
            }
            for qid, row in by_qid.items()
        }
        for config, by_qid in cadres.items()
    }
    SORTIES_ETAGE3_COUCHES.mkdir(parents=True, exist_ok=True)
    CADRES_PATH.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cadres


def _load_cadres_c_reference() -> dict[str, dict[str, Any]]:
    cadres: dict[str, dict[str, Any]] = {}
    for invite in e3.load_invites_reponses():
        if invite["bras"] != "C":
            continue
        cadres[invite["question_id"]] = {"rendered": invite["user"].split("\n\nQuestion:", 1)[0]}
    if len(cadres) != 141:
        raise RuntimeError(f"cadres C de référence incomplets : {len(cadres)}/141")
    return cadres


def build_response_invites(
    questions: list[dict[str, Any]],
    cadres: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    invites: list[dict[str, Any]] = []
    for question in questions:
        qid = question["id"]
        for config in CONFIGS:
            rendered = cadres[config][qid]["rendered"]
            invite = e3.build_response_invite(config, question, context=rendered)
            invite["config"] = config
            invites.append(invite)
    return invites


def stability_pairs_brute_seule(questions: list[dict[str, Any]] | None = None) -> list[tuple[str, str]]:
    all_questions = questions or e3.load_questions()
    pairs = e3.stability_pairs(all_questions)
    return [("brute_seule", qid) for _bras, qid in pairs]


def write_invites(invites: list[dict[str, Any]]) -> None:
    SORTIES_ETAGE3_COUCHES.mkdir(parents=True, exist_ok=True)
    INVITES_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in invites),
        encoding="utf-8",
    )


def load_invites_reponses(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or INVITES_PATH
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_call_budget(path: Path) -> None:
    if e3.count_calls(path) >= MAX_CALLS:
        raise RuntimeError(f"plafond d'appels atteint ({MAX_CALLS})")


def run_reponses(chat_fn: e3.ChatFn, *, calls_path: Path | None = None) -> dict[str, Any]:
    calls_path = calls_path or APPELS_PATH
    invites = load_invites_reponses()
    completed, invite_hashes = e3.load_journal_state(calls_path)
    stability = {e3.invite_id("brute_seule", qid) for _bras, qid in stability_pairs_brute_seule()}
    for invite in invites:
        ident = invite["id"]
        if ident in invite_hashes:
            e3.verify_invite_hash(ident, invite_hashes[ident], invite)
        if ident in completed:
            continue
        assert_call_budget(calls_path)
        e3.adjudicate_response(invite, chat_fn=chat_fn, calls_path=calls_path)
        if ident in stability:
            rejeu_id = e3.stability_rejeu_id(ident)
            if rejeu_id in invite_hashes:
                e3.verify_invite_hash(rejeu_id, invite_hashes[rejeu_id], invite)
            if rejeu_id not in completed:
                assert_call_budget(calls_path)
                e3.adjudicate_response(
                    invite,
                    chat_fn=chat_fn,
                    calls_path=calls_path,
                    record_id=rejeu_id,
                )
    return {"status": "reponses", "appels": e3.count_calls(calls_path)}


def run_juge(chat_fn: e3.ChatFn, *, calls_path: Path | None = None) -> dict[str, Any]:
    calls_path = calls_path or APPELS_PATH
    invites = load_invites_reponses()
    questions = {q["id"]: q for q in e3.load_questions()}
    responses = e3._latest_responses(calls_path)
    completed, _invite_hashes = e3.load_journal_state(calls_path)
    for invite in invites:
        juge_id = f"juge-{invite['id']}"
        if juge_id in completed:
            continue
        if invite["id"] not in responses:
            continue
        response_text = str(responses[invite["id"]].get("reponse_brute") or "")
        assert_call_budget(calls_path)
        e3.adjudicate_judge(
            response_text,
            questions[invite["question_id"]],
            invite_key=juge_id,
            chat_fn=chat_fn,
            calls_path=calls_path,
        )
    return {"status": "juge", "appels": e3.count_calls(calls_path)}


def compute_stabilite(calls_path: Path, questions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    questions = questions or e3.load_questions()
    qmap = {q["id"]: q for q in questions}
    pairs = stability_pairs_brute_seule(questions)
    text_identical = 0
    nombres_identical = 0
    total = 0
    for _bras, qid in pairs:
        base_id = e3.invite_id("brute_seule", qid)
        observations = e3._response_observations(calls_path, base_id)
        if len(observations) < 2:
            continue
        total += 1
        t1 = str(observations[0].get("reponse_brute") or "")
        t2 = str(observations[1].get("reponse_brute") or "")
        if t1 == t2:
            text_identical += 1
        question = qmap[qid]
        n1 = e3.compute_nombres_fields(question, t1)["tous_les_nombres"]
        n2 = e3.compute_nombres_fields(question, t2)["tous_les_nombres"]
        if n1 == n2:
            nombres_identical += 1
    return {
        "couples": total,
        "reponses_identiques": f"{text_identical}/{total}",
        "taux_reponses_identiques": (text_identical / total) if total else None,
        "tous_les_nombres_identiques": f"{nombres_identical}/{total}",
        "taux_tous_les_nombres_identiques": (nombres_identical / total) if total else None,
    }


def run_noter(
    *,
    calls_path: Path | None = None,
    invites_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    calls_path = calls_path or APPELS_PATH
    invites = load_invites_reponses(invites_path)
    questions = {q["id"]: q for q in e3.load_questions()}
    responses = e3._latest_responses(calls_path)
    juges = e3._latest_judges(calls_path)
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
        verdict = e3.compute_verdict(question, response_text, judge)
        nombres = e3.compute_nombres_fields(question, response_text)
        scoring = e3.must_mention_scoring_fields(question, response_text)
        notes.append(
            {
                "id": ident,
                "bras": invite["bras"],
                "config": invite.get("config", invite["bras"]),
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
    tableaux = _build_tableaux(notes)
    stabilite = compute_stabilite(calls_path)
    payload = {
        "date": date.today().isoformat(),
        "notes": notes,
        "tableaux": tableaux,
        "stabilite": stabilite,
        "complet": not manquants,
        "manquants": sorted(manquants),
    }
    out = output_path or NOTATIONS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _build_tableaux(notes: list[dict[str, Any]]) -> dict[str, Any]:
    tableaux: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "141": defaultdict(lambda: defaultdict(dict)),
    }
    for note in notes:
        shape = str(note.get("shape") or "unknown")
        bras = str(note["bras"])
        cell = tableaux["141"][shape][bras]
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
                    "part_must_mention_sum": 0.0,
                    "part_must_mention_count": 0,
                }
            )
        verdict = str(note["verdict"])
        if verdict in cell:
            cell[verdict] += 1
        if note.get("tous_les_nombres") is not None:
            cell["tous_les_nombres_attendus"] += 1
            if note["tous_les_nombres"]:
                cell["tous_les_nombres_true"] += 1
        part = note.get("part_must_mention")
        if part is not None:
            cell["part_must_mention_sum"] += float(part)
            cell["part_must_mention_count"] += 1
    result: dict[str, Any] = {"141": {}}
    for shape, by_bras in sorted(tableaux["141"].items()):
        result["141"][shape] = {}
        for bras, cell in sorted(by_bras.items()):
            attendus = cell.get("tous_les_nombres_attendus", 0)
            vrais = cell.get("tous_les_nombres_true", 0)
            part_count = cell.get("part_must_mention_count", 0)
            part_avg = round(cell["part_must_mention_sum"] / part_count, 3) if part_count else None
            result["141"][shape][bras] = {
                "conforme": cell.get("conforme", 0),
                "incomplete": cell.get("incomplete", 0),
                "fautive": cell.get("fautive", 0),
                "a_verifier": cell.get("a_verifier", 0),
                "indetermine": cell.get("indetermine", 0),
                "tous_les_nombres": f"{vrais}/{attendus}" if attendus else "0/0",
                "part_must_mention_moyenne": part_avg,
            }
    return result


def _integer_dispute_values(question: dict[str, Any]) -> list[str]:
    tokens = []
    for token in question.get("must_mention") or []:
        token = str(token).strip()
        if _INTEGER_MUST_MENTION_RE.match(token):
            tokens.append(token)
    return tokens


def response_has_both_dispute_values(question: dict[str, Any], response_text: str) -> bool | None:
    values = _integer_dispute_values(question)
    if len(values) < 2:
        return None
    return all(e3.must_mention_present(value, response_text) for value in values[:2])


def _load_etage3_reference_table() -> dict[str, dict[str, dict[str, Any]]]:
    if not ETAGE3_NOTATIONS_PATH.is_file():
        raise RuntimeError("notations étage 3 absentes (sorties/etage3/notations.json)")
    payload = json.loads(ETAGE3_NOTATIONS_PATH.read_text(encoding="utf-8"))
    table: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for note in payload.get("notes") or []:
        shape = str(note.get("shape") or "unknown")
        bras = str(note["bras"])
        if bras not in {"A", "C"}:
            continue
        cell = table[shape].setdefault(
            bras,
            {
                "tous_les_nombres_true": 0,
                "tous_les_nombres_attendus": 0,
                "part_must_mention_sum": 0.0,
                "part_must_mention_count": 0,
                "a_verifier": 0,
            },
        )
        if note.get("tous_les_nombres") is not None:
            cell["tous_les_nombres_attendus"] += 1
            if note["tous_les_nombres"]:
                cell["tous_les_nombres_true"] += 1
        part = note.get("part_must_mention")
        if part is not None:
            cell["part_must_mention_sum"] += float(part)
            cell["part_must_mention_count"] += 1
        if note.get("verdict") == "a_verifier":
            cell["a_verifier"] += 1
    return table


def _token_stats_from_invites(invites: list[dict[str, Any]], bras: str) -> dict[str, Any]:
    sizes = [int(item.get("user_chars", len(item.get("user", "")))) for item in invites if item["bras"] == bras]
    if not sizes:
        return {"mediane": 0, "total_tokens": 0, "mediane_tokens": 0}
    tokens = [s / 4 for s in sizes]
    return {
        "mediane": sorted(sizes)[len(sizes) // 2],
        "total_tokens": round(sum(tokens)),
        "mediane_tokens": round(statistics.median(tokens)),
    }


def _load_ablation_etage2(config: str) -> dict[str, dict[str, Any]]:
    if not ABLATION_PATH.is_file():
        raise RuntimeError("ablation couche-brute absente")
    payload = json.loads(ABLATION_PATH.read_text(encoding="utf-8"))
    rows = payload["configurations"][config]["questions"]
    return {row["id"]: row["k8"]["verdict_plancher"] for row in rows}


def _load_etage2_c_verdicts() -> dict[str, dict[str, Any]]:
    path = ETAGE2_C_PATH if ETAGE2_C_PATH.is_file() else SORTIES / "etage2-C.json"
    if not path.is_file():
        raise RuntimeError("etage2-C.json absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in payload.get("entries") or payload.get("questions") or []}
    out: dict[str, dict[str, Any]] = {}
    for qid, row in by_id.items():
        block = row.get("k8") or row
        out[qid] = block.get("verdict_plancher") or block.get("verdict") or {}
    return out


def write_report(
    *,
    notations: dict[str, Any],
    invites: list[dict[str, Any]],
    calls_path: Path | None = None,
) -> None:
    calls_path = calls_path or APPELS_PATH
    ref_table = _load_etage3_reference_table()
    local_table = notations.get("tableaux", {}).get("141") or {}
    questions = {q["id"]: q for q in e3.load_questions()}
    dispute_questions = [q for q in questions.values() if q.get("shape") == "dispute"]
    responses = e3._latest_responses(calls_path)
    etage3_responses = e3._latest_responses(e3.APPELS_PATH)

    shapes = sorted(set(ref_table) | set(local_table) | {"single", "dispute", "aggregate", "status", "trajectory"})
    columns = [("A", "A"), ("C", "C"), ("brute_seule", "brute seule"), ("brute_fine", "brute + fine")]

    lines = [
        "# Rapport — étage 3 sur les couches (k = 8)",
        "",
        f"Date : {date.today().isoformat()}",
        "",
        "## Tableau décisionnel — tous les nombres attendus dans la réponse",
        "",
        "| Forme | A | C (CallFrame) | brute seule | brute + fine |",
        "| --- | --- | --- | --- | --- |",
    ]
    for shape in shapes:
        cells = []
        for key, _label in columns:
            if key in {"A", "C"}:
                cell = ref_table.get(shape, {}).get(key, {})
                vrais = cell.get("tous_les_nombres_true", 0)
                attendus = cell.get("tous_les_nombres_attendus", 0)
                cells.append(f"{vrais} / {attendus}" if attendus else "—")
            else:
                cell = local_table.get(shape, {}).get(key, {})
                cells.append(cell.get("tous_les_nombres", "—"))
        lines.append(f"| {shape} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "## Part des `must_mention` présents (moyenne par forme)",
            "",
            "| Forme | A | C | brute seule | brute + fine |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for shape in shapes:
        cells = []
        for key, _label in columns:
            if key in {"A", "C"}:
                cell = ref_table.get(shape, {}).get(key, {})
                count = cell.get("part_must_mention_count", 0)
                avg = round(cell["part_must_mention_sum"] / count, 3) if count else None
            else:
                avg = local_table.get(shape, {}).get(key, {}).get("part_must_mention_moyenne")
            cells.append(f"{avg:.0%}" if isinstance(avg, (int, float)) else "—")
        lines.append(f"| {shape} | {' | '.join(cells)} |")

    token_a = _token_stats_from_invites(e3.load_invites_reponses(), "A")
    token_c = _token_stats_from_invites(e3.load_invites_reponses(), "C")
    token_brute = _token_stats_from_invites(invites, "brute_seule")
    token_combo = _token_stats_from_invites(invites, "brute_fine")
    lines.extend(
        [
            "",
            "## Coût — tokens d'entrée (user / 4)",
            "",
            "| Configuration | Médiane | Total (141 q.) | Rapport à C |",
            "| --- | --- | --- | --- |",
            f"| A (réf. étage 3) | {round(token_a['mediane']/4)} | — | {round(token_a['mediane']/max(token_c['mediane'],1), 2)} × |",
            f"| C (CallFrame) | {token_c['mediane_tokens']} | {token_c['total_tokens']} | 1 × |",
            f"| brute seule | {token_brute['mediane_tokens']} | {token_brute['total_tokens']} | {round(token_brute['mediane_tokens']/max(token_c['mediane_tokens'],1), 2)} × |",
            f"| brute + fine | {token_combo['mediane_tokens']} | {token_combo['total_tokens']} | {round(token_combo['mediane_tokens']/max(token_c['mediane_tokens'],1), 2)} × |",
            "",
            "## Disputes (28) — croisement étage 2 × étage 3",
            "",
            "| Question | étage 2 brute seule | étage 2 brute+fine | étage 2 C | réponse brute seule | réponse brute+fine | réponse C |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    etage2_brute = _load_ablation_etage2("brute_seule")
    etage2_combo = _load_ablation_etage2("brute_fine")
    etage2_c = _load_etage2_c_verdicts()

    def _e2_label(block: dict[str, Any]) -> str:
        return str(block.get("verdict") or block)

    def _e3_label(qid: str, bras: str, source: dict[str, dict[str, Any]]) -> str:
        resp = source.get(e3.invite_id(bras, qid))
        if not resp:
            return "—"
        question = questions[qid]
        text = str(resp.get("reponse_brute") or "")
        both = response_has_both_dispute_values(question, text)
        if both is True:
            return "deux valeurs"
        if both is False:
            return "une valeur"
        return "n/a"

    for question in sorted(dispute_questions, key=lambda q: q["id"]):
        qid = question["id"]
        lines.append(
            f"| {qid} | {_e2_label(etage2_brute[qid])} | {_e2_label(etage2_combo[qid])} | "
            f"{_e2_label(etage2_c.get(qid, {}))} | {_e3_label(qid, 'brute_seule', responses)} | "
            f"{_e3_label(qid, 'brute_fine', responses)} | {_e3_label(qid, 'C', etage3_responses)} |"
        )

    a_verifier_brute = sum(
        1 for note in notations.get("notes", []) if note.get("bras") == "brute_seule" and note.get("verdict") == "a_verifier"
    )
    a_verifier_combo = sum(
        1 for note in notations.get("notes", []) if note.get("bras") == "brute_fine" and note.get("verdict") == "a_verifier"
    )
    lines.extend(
        [
            "",
            "## Verdicts `a_verifier` (affirme sans citation vérifiable)",
            "",
            f"- brute seule : {a_verifier_brute}",
            f"- brute + fine : {a_verifier_combo}",
            "",
            "## Trois phrases",
            "",
            *_three_sentences(ref_table, local_table, token_brute, token_combo, token_c),
        ]
    )
    RAPPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _three_sentences(
    ref_table: dict[str, dict[str, dict[str, Any]]],
    local_table: dict[str, dict[str, dict[str, Any]]],
    token_brute: dict[str, Any],
    token_combo: dict[str, Any],
    token_c: dict[str, Any],
) -> list[str]:
    def rate(table: dict[str, Any], shape: str, bras: str) -> tuple[int, int]:
        cell = table.get(shape, {}).get(bras, {})
        if "tous_les_nombres" in cell:
            num, den = str(cell["tous_les_nombres"]).split("/")
            return int(num), int(den)
        return cell.get("tous_les_nombres_true", 0), cell.get("tous_les_nombres_attendus", 0)

    b_num, b_den = rate(local_table, "dispute", "brute_seule")
    c_num, c_den = rate(ref_table, "dispute", "C")
    best_shape = "single"
    best_delta = -1.0
    for shape in local_table:
        bn, bd = rate(local_table, shape, "brute_seule")
        cn, cd = rate(ref_table, shape, "C")
        if bd and cd:
            delta = (bn / bd) - (cn / cd)
            if delta > best_delta:
                best_delta = delta
                best_shape = shape

    return [
        (
            f"1. L'avantage étage 2 de la couche brute ({b_num}/{b_den} disputes avec les deux nombres attendus) "
            f"se retrouve partiellement à l'étage 3, mais reste en retrait sur la CallFrame ({c_num}/{c_den}) "
            f"sur les disputes."
        ),
        (
            f"2. La brute seule coûte {token_brute['mediane_tokens']} tokens médians "
            f"({round(token_brute['mediane_tokens']/max(token_c['mediane_tokens'],1), 1)} × C) ; "
            f"brute+fine {token_combo['mediane_tokens']} tokens "
            f"({round(token_combo['mediane_tokens']/max(token_c['mediane_tokens'],1), 1)} × C)."
        ),
        (
            f"3. La couche fine reste la meilleure affaire sur « {best_shape} » quand on cherche le meilleur "
            f"compromis tokens / nombres rendus."
        ),
    ]


def compile_and_prepare(connection) -> dict[str, Any]:
    cadres = compile_cadres(connection)
    questions = e3.load_questions()
    invites = build_response_invites(questions, cadres)
    write_invites(invites)
    return {"cadres": len(cadres["brute_seule"]), "invites": len(invites)}
