#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrateur — construire la couche brute et mesurer l'ablation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[2]
for entry in (WORKTREE / "scripts", WORKTREE / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.config import BANC_ROOT, export_runtime_env, write_banc_config_files
from banc_realite.couche_brute_build import COUCHE_BRUTE_SORTIES, build_couche_brute
from banc_realite.couche_brute_schema import DIVERGENCE_NOTE
from banc_realite.infra import connect_existing, start_cluster, stop_cluster
from banc_realite.measure_couche_brute import CONFIGS, build_avant_apres, measure_ablation

ARRET = BANC_ROOT / "_couche-brute-arret.json"
RESULT = BANC_ROOT / "_couche-brute-result.json"
REPORT = COUCHE_BRUTE_SORTIES / "RAPPORT-COUCHE-BRUTE.md"
PYTHON = sys.executable


def write_arret(step: str, why: str) -> None:
    COUCHE_BRUTE_SORTIES.mkdir(parents=True, exist_ok=True)
    ARRET.write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _run_tests() -> tuple[int, str]:
    proc = subprocess.run(
        [
            PYTHON,
            "-m",
            "pytest",
            "tests/banc_realite/test_passages.py",
            "tests/banc_realite/test_couche_brute_search.py",
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
        if "passed" in line or "failed" in line:
            summary = line.strip()
            break
    if not summary:
        summary = (proc.stdout or proc.stderr or "pytest")[-500:]
    return proc.returncode, summary


def _table_md(aggregates: dict[str, Any], k_label: str) -> str:
    key = f"par_forme_difficulte_{k_label}"
    data = aggregates.get(key) or {}
    lines = [
        f"| forme | difficulté | plancher | plafond |",
        f"| --- | --- | --- | --- |",
    ]
    for shape in sorted(data):
        for diff in sorted(data[shape]):
            plancher = data[shape][diff].get("plancher", {})
            plafond = data[shape][diff].get("plafond", {})
            lines.append(
                f"| {shape} | {diff} | {json.dumps(plancher, ensure_ascii=False)} | "
                f"{json.dumps(plafond, ensure_ascii=False)} |"
            )
    return "\n".join(lines)


def _avant_apres_table_md(ablation: dict[str, Any]) -> str:
    apres_rel = ablation.get("avant_apres")
    if not apres_rel:
        return "_Tableau avant/après indisponible (pas de mesure antérieure)._"
    apres_path = BANC_ROOT / apres_rel
    if not apres_path.is_file():
        return "_Tableau avant/après indisponible (fichier absent)._"
    payload = json.loads(apres_path.read_text(encoding="utf-8"))
    table = payload.get("par_forme_k8") or {}
    lines = [
        "| forme | config | réussite k8 avant | réussite k8 après | Δ réussite | tokens k8 méd. avant | tokens k8 méd. après |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for shape in sorted(table):
        for config in CONFIGS:
            cell = table[shape].get(config) or {}
            av = cell.get("k8_plancher_reussi_avant")
            ap = cell.get("k8_plancher_reussi_apres")
            delta = cell.get("k8_plancher_delta")
            av_s = f"{av:.1%}" if av is not None else "—"
            ap_s = f"{ap:.1%}" if ap is not None else "—"
            delta_s = f"{delta:+.1%}" if delta is not None else "—"
            lines.append(
                f"| {shape} | {config} | {av_s} | {ap_s} | {delta_s} | "
                f"{cell.get('tokens_k8_mediane_avant', '—')} | {cell.get('tokens_k8_mediane_apres', '—')} |"
            )
    return "\n".join(lines)


def write_report(build: dict[str, Any], ablation: dict[str, Any], tests_summary: str) -> None:
    reserve = (
        "Réserve d'instrument : « tient » exige la citation littérale, ce qui favorise "
        "structurellement les passages ; l'arbitre reste l'étape 3."
    )
    sections = [
        "# Rapport — couche brute (passages)",
        "",
        reserve,
        "",
        "## Divergence banc / production",
        "",
        DIVERGENCE_NOTE,
        "",
        "## Passages produits",
        "",
        f"- Nombre : {build.get('passages')}",
        f"- Longueur médiane : {build.get('longueur_mediane')} caractères",
        f"- Appels embedding Mistral : {build.get('appels_embedding')}",
        "",
    ]
    for config in CONFIGS:
        agg = ablation["configurations"][config]["aggregates"]
        sections.extend(
            [
                f"## {config} — tokens injectés",
                "",
                f"- k=8 : total {agg['tokens_k8']['total']}, médiane {agg['tokens_k8']['mediane']}",
                f"- k=20 : total {agg['tokens_k20']['total']}, médiane {agg['tokens_k20']['mediane']}",
                "",
                f"### Tableau forme × difficulté (k=8)",
                "",
                _table_md(agg, "k8"),
                "",
                f"### Tableau forme × difficulté (k=20)",
                "",
                _table_md(agg, "k20"),
                "",
            ]
        )

    ref = ablation.get("callframe_reference") or {}
    sections.extend(
        [
            "## Contrôle CallFrame vs fine seule",
            "",
            ref.get("note", ""),
            "",
            "### CallFrame (etage2-C) k=8",
            "",
            json.dumps(ref.get("aggregates_k8"), ensure_ascii=False, indent=2),
            "",
            "### fine seule (cette mesure) k=8",
            "",
            json.dumps(
                ablation["configurations"]["fine_seule"]["aggregates"].get("par_forme_difficulte_k8"),
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "## Ablation avant / après corrections PR99 (k=8, par forme)",
            "",
            _avant_apres_table_md(ablation),
            "",
            "## Trois phrases",
            "",
            _three_sentences(ablation),
            "",
            f"## Tests\n\n{tests_summary}",
        ]
    )
    REPORT.write_text("\n".join(sections) + "\n", encoding="utf-8")


def _three_sentences(ablation: dict[str, Any]) -> str:
    configs = ablation["configurations"]
    brute = configs["brute_seule"]["aggregates"]
    fine = configs["fine_seule"]["aggregates"]
    combo = configs["brute_fine"]["aggregates"]

    def success_rate(agg: dict[str, Any], k: int) -> float:
        data = agg.get(f"par_forme_difficulte_k{k}") or {}
        total = 0
        ok = 0
        for shape in data:
            for diff in data[shape]:
                for verdict, count in data[shape][diff].get("plancher", {}).items():
                    total += count
                    if verdict == "reussi":
                        ok += count
        return ok / total if total else 0.0

    brute_gain = success_rate(combo, 8) - success_rate(fine, 8)
    brute_cost = combo["tokens_k8"]["total"] - fine["tokens_k8"]["total"]
    best_shape = "single"
    best_delta = None
    for shape in (brute.get("par_forme_difficulte_k8") or {}):
        b = (brute["par_forme_difficulte_k8"].get(shape) or {}).get("easy", {}).get("plancher", {})
        f = (fine["par_forme_difficulte_k8"].get(shape) or {}).get("easy", {}).get("plancher", {})
        delta = b.get("reussi", 0) - f.get("reussi", 0)
        if best_delta is None or delta > best_delta:
            best_delta = delta
            best_shape = shape

    if brute_gain > 0:
        phrase1 = (
            f"1. La couche brute apporte un reliquat lexical que la fine n'a pas : "
            f"brute+fine gagne {brute_gain:.1%} de réussites plancher (k=8) vs fine seule."
        )
    else:
        phrase1 = (
            f"1. La couche brute n'améliore pas le composite : brute+fine perd "
            f"{abs(brute_gain):.1%} de réussites plancher (k=8) vs fine seule."
        )

    phrase2 = (
        f"2. Le composite coûte {brute_cost:.0f} tokens estimés de plus (k=8, total) "
        f"que la fine seule."
        if brute_cost >= 0
        else (
            f"2. Le composite coûte {abs(brute_cost):.0f} tokens estimés de moins (k=8, total) "
            f"que la fine seule."
        )
    )

    if best_delta is not None and best_delta > 0:
        phrase3 = (
            f"3. Les passages gagnent surtout sur « {best_shape} » (+{best_delta} réussites easy) "
            f"là où l'atomisation laisse des assertions absentes."
        )
    elif best_delta is not None and best_delta < 0:
        phrase3 = (
            f"3. L'atomisation reste devant sur « {best_shape} » ({best_delta} réussites easy) : "
            f"les passages n'y rattrapent pas le reliquat attendu."
        )
    else:
        phrase3 = (
            f"3. Sur « {best_shape} », passages et atomisation se valent sur le plancher easy ; "
            f"le reliquat lexical ne se traduit pas en écart net."
        )

    return f"{phrase1}\n{phrase2}\n{phrase3}"


def main() -> int:
    COUCHE_BRUTE_SORTIES.mkdir(parents=True, exist_ok=True)
    export_runtime_env()
    write_banc_config_files()

    start_cluster()
    try:
        connection = connect_existing()
        build = build_couche_brute(connection)
    except Exception as exc:  # noqa: BLE001
        write_arret("construction", str(exc))
        stop_cluster()
        return 1

    try:
        ablation = measure_ablation(connection)
    except Exception as exc:  # noqa: BLE001
        write_arret("ablation", str(exc))
        stop_cluster()
        return 1

    test_code, tests_summary = _run_tests()
    if test_code != 0:
        write_arret("tests", tests_summary)
        stop_cluster()
        return 1

    write_report(build, ablation, tests_summary)
    RESULT.write_text(
        json.dumps(
            {
                "status": "complete",
                "passages": build.get("passages"),
                "appels_embedding": build.get("appels_embedding"),
                "tests": tests_summary,
                "tableaux": "sorties/couche-brute/ablation.json",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stop_cluster()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
