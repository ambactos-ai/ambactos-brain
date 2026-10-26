#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Second tour étage 3 couches — contextes corrigés, reprise sélective tour 1."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[2]
for entry in (WORKTREE / "scripts", WORKTREE / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite import etage3 as e3
from banc_realite import etage3_couches as ec
from banc_realite.config import BANC_ROOT, export_runtime_env, write_banc_config_files
from banc_realite.etage3 import make_relay_chat_fn
from banc_realite.infra import connect_existing, start_cluster, stop_cluster

PYTHON = sys.executable

SORTIES_V2 = BANC_ROOT / "sorties" / "etage3-couches-v2"
ROUND1_DIR = BANC_ROOT / "sorties" / "etage3-couches"
ROUND1_INVITES = ROUND1_DIR / "invites-reponses.jsonl"
ROUND1_APPELS = ROUND1_DIR / "appels.jsonl"
ROUND1_NOTATIONS = ROUND1_DIR / "notations.json"

ARRET_V2 = BANC_ROOT / "_etage3-couches-v2-arret.json"
RESULT_V2 = BANC_ROOT / "_etage3-couches-v2-result.json"

CADRES_V2 = SORTIES_V2 / "cadres.json"
INVITES_V2 = SORTIES_V2 / "invites-reponses.jsonl"
APPELS_V2 = SORTIES_V2 / "appels.jsonl"
NOTATIONS_V2 = SORTIES_V2 / "notations.json"
RAPPORT_V2 = SORTIES_V2 / "RAPPORT-ETAGE3-COUCHES-V2.md"

MAX_CALLS = 700


def write_arret_v2(step: str, why: str) -> None:
    SORTIES_V2.mkdir(parents=True, exist_ok=True)
    ARRET_V2.write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _patch_v2_paths() -> None:
    ec.SORTIES_ETAGE3_COUCHES = SORTIES_V2
    ec.CADRES_PATH = CADRES_V2
    ec.INVITES_PATH = INVITES_V2
    ec.APPELS_PATH = APPELS_V2
    ec.NOTATIONS_PATH = NOTATIONS_V2
    ec.RAPPORT_PATH = RAPPORT_V2
    ec.MAX_CALLS = MAX_CALLS


def _run_tests() -> tuple[int, str]:
    proc = subprocess.run(
        [
            PYTHON,
            "-m",
            "pytest",
            "tests/banc_realite/test_etage3_couches.py",
            "tests/banc_realite/test_couche_brute_search.py",
            "tests/banc_realite/test_passages.py",
            "-q",
        ],
        cwd=str(WORKTREE),
        capture_output=True,
        text=True,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": f"{WORKTREE / 'src'}:{WORKTREE / 'scripts'}:{WORKTREE}",
        },
    )
    summary = ""
    for line in reversed((proc.stdout or "").splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            break
    if not summary:
        summary = (proc.stdout or proc.stderr or "pytest")[-500:]
    return proc.returncode, summary


def _final_records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in e3.load_journal_records(path):
        ident = str(record.get("id") or "")
        if not ident:
            continue
        if record.get("role") in {"reponse", "juge"} and record.get("final", True):
            latest[ident] = record
    return latest


def classify_invites(v2_invites: list[dict[str, Any]]) -> tuple[list[str], list[str], int]:
    round1 = {item["id"]: item for item in ec.load_invites_reponses(ROUND1_INVITES)}
    unchanged: list[str] = []
    changed: list[str] = []
    for invite in v2_invites:
        prior = round1.get(invite["id"])
        if prior and prior.get("sha256") == invite.get("sha256"):
            unchanged.append(invite["id"])
        else:
            changed.append(invite["id"])
    questions_changed = len({invite["question_id"] for invite in v2_invites if invite["id"] in changed})
    return changed, unchanged, questions_changed


def seed_from_round1(unchanged_ids: set[str], stability_ids: set[str]) -> int:
    source = _final_records_by_id(ROUND1_APPELS)
    copied = 0
    for ident in sorted(unchanged_ids):
        response = source.get(ident)
        if not response:
            raise RuntimeError(f"réponse tour 1 absente pour {ident}")
        row = dict(response)
        row["repris_tour1"] = True
        row["horodatage"] = e3._iso_now()
        e3.append_call(APPELS_V2, row)
        copied += 1
        if ident in stability_ids:
            rejeu_id = e3.stability_rejeu_id(ident)
            rejeu = source.get(rejeu_id)
            if not rejeu:
                raise RuntimeError(f"rejeu tour 1 absent pour {rejeu_id}")
            rrow = dict(rejeu)
            rrow["repris_tour1"] = True
            rrow["horodatage"] = e3._iso_now()
            e3.append_call(APPELS_V2, rrow)
            copied += 1
        juge_id = f"juge-{ident}"
        judge = source.get(juge_id)
        if not judge:
            raise RuntimeError(f"juge tour 1 absent pour {juge_id}")
        jrow = dict(judge)
        jrow["repris_tour1"] = True
        jrow["horodatage"] = e3._iso_now()
        e3.append_call(APPELS_V2, jrow)
        copied += 1
    return copied


def count_depenses(path: Path) -> int:
    return sum(1 for record in e3.load_journal_records(path) if not record.get("repris_tour1"))


def assert_budget_v2(path: Path) -> None:
    if count_depenses(path) >= MAX_CALLS:
        raise RuntimeError(f"plafond d'appels atteint ({MAX_CALLS})")


def write_report_v2(
    *,
    notations: dict[str, Any],
    invites: list[dict[str, Any]],
    contextes_changes: int,
    appels_depenses: int,
    round1_notations: dict[str, Any],
) -> None:
    local_table = notations.get("tableaux", {}).get("141") or {}
    r1_local = round1_notations.get("tableaux", {}).get("141") or {}
    shapes = sorted(set(local_table) | set(r1_local) | {"single", "dispute", "aggregate", "status", "trajectory"})
    columns = [("A", "A"), ("C", "C"), ("brute_seule", "brute seule"), ("brute_fine", "brute + fine")]
    ref_table = ec._load_etage3_reference_table()

    lines = [
        "# Rapport — étage 3 sur les couches v2 (k = 8, contextes corrigés)",
        "",
        f"Date : {date.today().isoformat()}",
        "",
        f"**Contextes modifiés** : {contextes_changes} questions sur 141 (empreinte invite changée).",
        f"**Appels réellement dépensés** : {appels_depenses} / plafond {MAX_CALLS}.",
        "",
        "## Tableau décisionnel v2 — tous les nombres attendus dans la réponse",
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
                cells.append(local_table.get(shape, {}).get(key, {}).get("tous_les_nombres", "—"))
        lines.append(f"| {shape} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "## Tableau décisionnel tour 1 (avant corrections) — référence",
            "",
            "| Forme | A | C (CallFrame) | brute seule | brute + fine |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for shape in shapes:
        cells = []
        for key, _label in columns:
            if key in {"A", "C"}:
                cell = ref_table.get(shape, {}).get(key, {})
                vrais = cell.get("tous_les_nombres_true", 0)
                attendus = cell.get("tous_les_nombres_attendus", 0)
                cells.append(f"{vrais} / {attendus}" if attendus else "—")
            else:
                cells.append(r1_local.get(shape, {}).get(key, {}).get("tous_les_nombres", "—"))
        lines.append(f"| {shape} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "## Écart v2 − tour 1 (numérateur brute seule / brute + fine)",
            "",
            "| Forme | Δ brute seule | Δ brute + fine |",
            "| --- | --- | --- |",
        ]
    )
    for shape in shapes:
        deltas = []
        for bras in ("brute_seule", "brute_fine"):
            v2_cell = local_table.get(shape, {}).get(bras, {}).get("tous_les_nombres", "0/0")
            r1_cell = r1_local.get(shape, {}).get(bras, {}).get("tous_les_nombres", "0/0")
            try:
                v2n = int(str(v2_cell).split("/")[0])
                r1n = int(str(r1_cell).split("/")[0])
                deltas.append(f"{v2n - r1n:+d}")
            except (ValueError, IndexError):
                deltas.append("—")
        lines.append(f"| {shape} | {' | '.join(deltas)} |")

    token_c = ec._token_stats_from_invites(e3.load_invites_reponses(), "C")
    token_brute = ec._token_stats_from_invites(invites, "brute_seule")
    token_combo = ec._token_stats_from_invites(invites, "brute_fine")
    lines.extend(
        [
            "",
            "## Coût — tokens d'entrée injectés (v2)",
            "",
            "| Configuration | Médiane | Total (141 q.) | Rapport à C |",
            "| --- | --- | --- | --- |",
            f"| C (CallFrame) | {token_c['mediane_tokens']} | {token_c['total_tokens']} | 1 × |",
            f"| brute seule | {token_brute['mediane_tokens']} | {token_brute['total_tokens']} | "
            f"{round(token_brute['mediane_tokens']/max(token_c['mediane_tokens'],1), 2)} × |",
            f"| brute + fine | {token_combo['mediane_tokens']} | {token_combo['total_tokens']} | "
            f"{round(token_combo['mediane_tokens']/max(token_c['mediane_tokens'],1), 2)} × |",
            "",
            "## Disputes (28) — croisement étage 2 × étage 3 v2",
            "",
            "| Question | étage 2 brute seule | étage 2 brute+fine | étage 2 C | réponse brute seule | réponse brute+fine | réponse C |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    questions = {q["id"]: q for q in e3.load_questions()}
    dispute_questions = [q for q in questions.values() if q.get("shape") == "dispute"]
    responses = e3._latest_responses(APPELS_V2)
    etage3_responses = e3._latest_responses(e3.APPELS_PATH)
    etage2_brute = ec._load_ablation_etage2("brute_seule")
    etage2_combo = ec._load_ablation_etage2("brute_fine")
    etage2_c = ec._load_etage2_c_verdicts()

    def _e2_label(block: dict[str, Any]) -> str:
        return str(block.get("verdict") or block)

    def _e3_label(qid: str, bras: str, source: dict[str, dict[str, Any]]) -> str:
        resp = source.get(e3.invite_id(bras, qid))
        if not resp:
            return "—"
        question = questions[qid]
        text = str(resp.get("reponse_brute") or "")
        both = ec.response_has_both_dispute_values(question, text)
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
    stabilite = notations.get("stabilite") or {}
    lines.extend(
        [
            "",
            "## Verdicts `a_verifier` (affirme sans citation vérifiable)",
            "",
            f"- brute seule : {a_verifier_brute}",
            f"- brute + fine : {a_verifier_combo}",
            "",
            "## Stabilité (20 rejeux brute seule)",
            "",
            f"- Réponses identiques : {stabilite.get('reponses_identiques', '—')}",
            f"- Tous les nombres identiques : {stabilite.get('tous_les_nombres_identiques', '—')}",
        ]
    )
    RAPPORT_V2.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    SORTIES_V2.mkdir(parents=True, exist_ok=True)
    _patch_v2_paths()
    export_runtime_env()
    write_banc_config_files()

    test_code, tests_summary = _run_tests()
    if test_code != 0:
        write_arret_v2("tests", tests_summary)
        return 1

    start_cluster()
    try:
        connection = connect_existing()
        prep = ec.compile_and_prepare(connection)
        invites = ec.load_invites_reponses(INVITES_V2)
        changed, unchanged, questions_changed = classify_invites(invites)
        stability_ids = {e3.invite_id("brute_seule", qid) for _b, qid in ec.stability_pairs_brute_seule()}
        seed_from_round1(set(unchanged), stability_ids)
    except Exception as exc:  # noqa: BLE001
        write_arret_v2("compilation", str(exc))
        stop_cluster()
        return 1

    e3.assert_call_budget = assert_budget_v2
    ec.assert_call_budget = assert_budget_v2
    e3.MAX_CALLS = MAX_CALLS

    chat_fn = make_relay_chat_fn()
    try:
        ec.run_reponses(chat_fn, calls_path=APPELS_V2)
    except Exception as exc:  # noqa: BLE001
        write_arret_v2("reponses", str(exc))
        stop_cluster()
        return 1

    try:
        ec.run_juge(chat_fn, calls_path=APPELS_V2)
    except Exception as exc:  # noqa: BLE001
        write_arret_v2("juge", str(exc))
        stop_cluster()
        return 1

    appels_depenses = count_depenses(APPELS_V2)
    if appels_depenses > MAX_CALLS:
        write_arret_v2("budget", f"{appels_depenses} appels > plafond {MAX_CALLS}")
        stop_cluster()
        return 1

    try:
        notations = ec.run_noter(calls_path=APPELS_V2, invites_path=INVITES_V2, output_path=NOTATIONS_V2)
    except Exception as exc:  # noqa: BLE001
        write_arret_v2("notation", str(exc))
        stop_cluster()
        return 1

    if not notations.get("complet"):
        write_arret_v2("notation", f"incomplet : {notations.get('manquants')}")
        stop_cluster()
        return 1

    round1_notations = json.loads(ROUND1_NOTATIONS.read_text(encoding="utf-8"))
    try:
        write_report_v2(
            notations=notations,
            invites=invites,
            contextes_changes=questions_changed,
            appels_depenses=appels_depenses,
            round1_notations=round1_notations,
        )
    except Exception as exc:  # noqa: BLE001
        write_arret_v2("rapport", str(exc))
        stop_cluster()
        return 1

    RESULT_V2.write_text(
        json.dumps(
            {
                "status": "complete",
                "appels_depenses": appels_depenses,
                "contextes_changes": questions_changed,
                "invites_changees": len(changed),
                "invites_reprises": len(unchanged),
                "tests": tests_summary,
                "stabilite": notations.get("stabilite"),
                "cadres": prep["cadres"],
                "invites": prep["invites"],
                "files": [
                    str(CADRES_V2),
                    str(INVITES_V2),
                    str(APPELS_V2),
                    str(NOTATIONS_V2),
                    str(RAPPORT_V2),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stop_cluster()
    print(
        json.dumps(
            {
                "status": "complete",
                "appels_depenses": appels_depenses,
                "contextes_changes": questions_changed,
                "rapport": str(RAPPORT_V2),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
