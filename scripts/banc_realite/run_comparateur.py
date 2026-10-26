# SPDX-License-Identifier: AGPL-3.0-only
"""Point d'entrée — comparateur Z1/Z2/Z3 + rejouage or et disputes (déterministe)."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
for entry in (WORKTREE / "scripts", WORKTREE / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.config import BANC_ROOT, export_runtime_env
from banc_realite.infra import connect_existing, start_cluster
from banc_realite.measure_disputes import (
    ABLATION_PATH,
    measure_ablation_mecanisme,
    measure_etage2_disputes,
    count_tour2_vides,
)
from banc_realite.measure_etage2 import EMBED_CACHE_PATH
from banc_realite.measure_resolution import run_measurement as measure_resolution
from banc_realite.production import copy_claims_to_public

RAPPORT_PATH = BANC_ROOT / "RAPPORT-COMPARATEUR.md"
RESULT_PATH = BANC_ROOT / "_comparateur-result.json"
ARRET_PATH = BANC_ROOT / "_comparateur-arret.json"
BASELINE_DISPUTES = 2


def write_arret(step: str, why: str) -> None:
    ARRET_PATH.write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _run_tests() -> str:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/banc_realite/test_couche_brute_search.py::test_completer_top_k_eligible_filtre_avant_limite",
        "tests/banc_realite/test_classer_couple.py::test_classer_couple_pas_accord_unites_differentes",
        "tests/banc_realite/test_classer_couple.py::test_classer_couple_pas_accord_montants_devise_differents",
        "tests/banc_realite/test_measure_disputes_ablation.py::test_compute_completion_deltas_une_fois_a_la_fin",
        "tests/banc_realite/test_trois_tours.py",
        "tests/banc_realite/test_resolution.py",
        "-q",
    ]
    proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tests échoués:\n{proc.stdout}\n{proc.stderr}")
    return " ".join(cmd[cmd.index("-m") + 2 : cmd.index("-q")])


def _load_embed_cache() -> dict:
    if not EMBED_CACHE_PATH.is_file():
        raise RuntimeError("cache froid : etage2-question-embeddings.json absent")
    return json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))


def _unresolved_examples(details: list[dict], limit: int = 3) -> list[dict]:
    out = []
    for row in details:
        if row.get("gold") != "dispute" or row.get("issue") == "dispute":
            continue
        out.append(
            {
                "id": row["id"],
                "issue": row["issue"],
                "raison": row.get("raison") or "",
                "entity_a": row.get("entity_a"),
                "entity_b": row.get("entity_b"),
            }
        )
        if len(out) >= limit:
            break
    return out


def write_report(
    *,
    resolution: dict,
    resolution_table: dict,
    etage2: dict,
    ablation: dict,
    tour2_disputes: dict,
    tour2_single: dict,
    examples: list[dict],
) -> None:
    best = resolution_table[resolution["meilleur_palier"]]
    lines = [
        "# Rapport — comparateur (Z1/Z2/Z3)",
        "",
        f"Date : {date.today().isoformat()}",
        "",
        "## 1. Tri à cinq issues contre l'or",
        "",
        "| indicateur | avant (baseline) | après |",
        "| --- | --- | --- |",
        f"| disputes reconnues / 45 | {BASELINE_DISPUTES}/45 | {best['disputes_reconnues']}/45 |",
        f"| trajectoires / 4 | — | {best['trajectoires']}/4 |",
        f"| pièges classés dispute / 11 | — | {best['pieges_en_dispute']}/11 |",
        f"| indécidable | — | {best['indecidable']} |",
        "",
        "## 2. 28 questions dispute — k=8 et k=20, avec/sans trois tours",
        "",
        "| Couche | k | Complet sans tours | Complet avec tours | Δ |",
        "| --- | --- | --- | --- | --- |",
    ]
    for couche in ("fine", "brute", "brute_fine"):
        for k in (8, 20):
            block = etage2["par_couche"][couche][f"k{k}"]
            lines.append(
                f"| {couche} | {k} | {block['complet_sans']}/28 | "
                f"{block['complet_avec']}/28 | {block['delta_complet']:+d} |"
            )
    lines.extend(
        [
            "",
            "### Ablation quatre modes (complet / 28)",
            "",
            "| Couche | k | témoin | trois_tours | complétion | les deux |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for couche in ("fine", "brute", "brute_fine"):
        for k in (8, 20):
            block = ablation["par_couche"][couche][f"k{k}"]
            lines.append(
                f"| {couche} | {k} | {block['temoin']['complet']}/28 | "
                f"{block['trois_tours']['complet']}/28 | {block['completion']['complet']}/28 | "
                f"{block['trois_tours_completion']['complet']}/28 |"
            )
    lines.extend(
        [
            "",
            f"Complétion : {ablation['chiffre_tranche_completion']}",
            "",
            "## 3. Tours 2 vides (brute_fine, k=8)",
            "",
            f"| shape | total | tour 2 vide |",
            f"| --- | ---: | ---: |",
            f"| dispute | {tour2_disputes['total']} | {tour2_disputes['tour2_vides']} |",
            f"| single | {tour2_single['total']} | {tour2_single['tour2_vides']} |",
        ]
    )
    if best["disputes_reconnues"] <= BASELINE_DISPUTES and examples:
        lines.extend(["", "## Disputes non reconnues — trois exemples", ""])
        for ex in examples:
            lines.append(
                f"- `{ex['id']}` → **{ex['issue']}** ({ex['raison']}) : "
                f"`{ex['entity_a']}` vs `{ex['entity_b']}`"
            )
    RAPPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    export_runtime_env()
    try:
        tests = _run_tests()
        start_cluster()
        connection = connect_existing()
        copy_claims_to_public(connection)
        embed_cache = _load_embed_cache()

        from banc_realite.fidelite import load_gold
        from banc_realite.holders import build_holder_table
        from banc_realite.measure_disputes import _claims_by_piece, _read_public_claims

        gold = load_gold()
        claims = _read_public_claims(connection)
        claims_index = {str(c["claim_id"]): c for c in claims}
        build_holder_table(claims, gold)

        resolution = measure_resolution(connection)
        from banc_realite.measure_resolution import CONFUSION_PATH

        resolution_payload = json.loads(CONFUSION_PATH.read_text(encoding="utf-8"))
        resolution_table = resolution_payload["paliers"]

        etage2 = measure_etage2_disputes(
            connection,
            embed_cache=embed_cache,
            gold=gold,
            claims_index=claims_index,
        )
        ablation = measure_ablation_mecanisme(
            connection,
            embed_cache=embed_cache,
            gold=gold,
            claims_index=claims_index,
        )
        ABLATION_PATH.write_text(json.dumps(ablation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        tour2_disputes = count_tour2_vides(connection, embed_cache=embed_cache, shape="dispute")
        tour2_single = count_tour2_vides(connection, embed_cache=embed_cache, shape="single")

        examples = _unresolved_examples(resolution_table[resolution["meilleur_palier"]]["details"])

        write_report(
            resolution=resolution,
            resolution_table=resolution_table,
            etage2=etage2,
            ablation=ablation,
            tour2_disputes=tour2_disputes,
            tour2_single=tour2_single,
            examples=examples,
        )

        connection.close()
        result = {
            "status": "complete",
            "tests": tests,
            "disputes_reconnues": resolution["disputes_reconnues"],
            "pieges_en_dispute": resolution["pieges_en_dispute"],
            "tours2_vides_disputes": tour2_disputes["tour2_vides"],
            "tours2_vides_single": tour2_single["tour2_vides"],
            "meilleur_palier": resolution["meilleur_palier"],
            "completion_delta": ablation["completion_delta_complet"],
        }
        RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001
        write_arret("comparateur", str(exc))
        return 1
    finally:
        subprocess.run([str(BANC_ROOT / "arreter.sh")], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
