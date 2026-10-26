# SPDX-License-Identifier: AGPL-3.0-only
"""Plafond du filtre par l'or — instrument de mesure, pas une ligne du comparatif.

Cet outil lit l'or. Les pièces retenues pour chaque question sont celles où l'or
place les assertions d'appui (`supporting_assertions` → `source`). Le RAG
(vivier de 200) est restreint à ces pièces. Les sorties le disent en toutes
lettres. Aucun fournisseur ni registre n'importe ce module.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Callable

from .dossiers import Dossier, dossier_courant
from .fidelite import load_gold
from .fournisseurs_contexte import (
    K_VIVIER_FILTRE,
    RagNaif,
    filtrer_passages_par_pieces,
    rendre_contexte,
)
from .holders import build_holder_table
from .measure_disputes import _read_public_claims
from .measure_etage2 import _load_embed_cache, load_questions
from .mesure_comparatif_etage2 import (
    _aggregate_config_rows_dual,
    _compile_row,
    _disputes_denominators,
    _format_table_disputes,
    _format_table_par_forme,
    _meta_mesure,
    selection_questions_mesure,
)
from .normalize import piece_basename
from .or_dossier import verifier_or_coherent
from .partage_disputes import partager
from .production import verifier_parite

AVERTISSEMENT = (
    "Cet instrument de mesure lit l'or. Ce n'est pas une ligne du comparatif."
)
ENTETE = (
    "Les pièces retenues pour chaque question sont celles où l'or place les "
    "assertions d'appui (`supporting_assertions` → `source`). Le RAG "
    f"(vivier de {K_VIVIER_FILTRE}) est restreint à ces pièces."
)

VivierFn = Callable[[dict[str, Any]], list[dict[str, Any]]]


def pieces_or_question(
    question: dict[str, Any],
    assertion_index: dict[str, dict[str, Any]],
) -> list[str]:
    """Pièces où l'or place les assertions d'appui, ordre de première apparition."""
    retenues: list[str] = []
    seen: set[str] = set()
    for aid in question.get("supporting_assertions") or []:
        assertion = assertion_index.get(str(aid)) or {}
        source = str(assertion.get("source") or "").strip()
        if not source:
            continue
        piece = piece_basename(source)
        if not piece or piece in seen:
            continue
        seen.add(piece)
        retenues.append(piece)
    return retenues


def repertoire_plafonds(dossier: Dossier) -> Path:
    return dossier.comparatif_dir / "plafonds"


def _vivier_par_defaut(
    question: dict[str, Any],
    *,
    rag: RagNaif,
    connection,
) -> list[dict[str, Any]]:
    return rag.contexte(question, k=K_VIVIER_FILTRE, connection=connection)


def mesurer_plafond(
    *,
    dossier: Dossier,
    budgets: list[int],
    questions: list[dict[str, Any]],
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
    holder_table: dict[str, list[dict[str, Any]]],
    vivier_fn: VivierFn,
) -> dict[str, Any]:
    """Calcule le plafond avec les fonctions de mesure du banc. Lit l'or."""
    assertion_index = gold["assertion_index"]
    meta = _meta_mesure(dossier, questions)
    reglage, ecart = partager(questions)
    reglage_ids = {q["id"] for q in reglage}
    ecart_ids = {q["id"] for q in ecart}
    pieces_par_question: dict[str, list[str]] = {}
    elements_par_question: dict[str, list[dict[str, Any]]] = {}
    sans_piece_par_question: dict[str, int] = {}
    for question in questions:
        pieces = pieces_or_question(question, assertion_index)
        pieces_par_question[str(question["id"])] = pieces
        vivier = vivier_fn(question)
        gardes, sans_piece = filtrer_passages_par_pieces(vivier, pieces)
        elements_par_question[str(question["id"])] = gardes
        sans_piece_par_question[str(question["id"])] = sans_piece
    runs: list[dict[str, Any]] = []
    for budget in budgets:
        rows: list[dict[str, Any]] = []
        for question in questions:
            elements = elements_par_question[str(question["id"])]
            rendered, kept = rendre_contexte(
                elements,
                budget_chars=budget,
                question_id=str(question["id"]),
                style="texte",
                dossier=dossier,
            )
            rows.append(
                _compile_row(
                    question,
                    kept,
                    rendered,
                    gold=gold,
                    claims_index=claims_index,
                    assertion_index=assertion_index,
                    holder_table=holder_table,
                )
            )
        dispute_rows = [row for row in rows if row["shape"] == meta["dispute_shape"]]
        aggregate = _aggregate_config_rows_dual(
            rows,
            dispute_rows,
            reglage_ids=reglage_ids,
            ecart_ids=ecart_ids,
            dossier=dossier,
        )
        aggregate["disputes"].update(
            _disputes_denominators(
                dispute_rows,
                reglage_ids=reglage_ids,
                ecart_ids=ecart_ids,
            )
        )
        runs.append(
            {
                "k": None,
                "k_effectif": K_VIVIER_FILTRE,
                "budget": budget,
                "questions": rows,
                "aggregate": aggregate,
            }
        )
    result = {
        "fournisseur": "plafond_filtre_or",
        "k_fixe": True,
        "runs": runs,
    }
    return {
        "instrument": "plafond_filtre_or",
        "lit_or": True,
        "ligne_du_comparatif": False,
        "avertissement": AVERTISSEMENT,
        "entete": ENTETE,
        "dossier": dossier.nom,
        "date": date.today().isoformat(),
        "budgets": list(budgets),
        "k_vivier": K_VIVIER_FILTRE,
        "questions_total": len(questions),
        "questions_mesurables": len(selection_questions_mesure(questions)),
        "meta": meta,
        "pieces_or_par_question": pieces_par_question,
        "passages_sans_piece_par_question": sans_piece_par_question,
        "result": result,
    }


def _table_synthese(payload: dict[str, Any]) -> str:
    result = payload["result"]
    meta = payload["meta"]
    denom = meta["denominateur_appui"]
    lines = [
        "## Tout tient / au moins une",
        "",
        "| budget | tout tient (verbatim) | tout tient (valeur) | ≥1 (verbatim) | ≥1 (valeur) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in result["runs"]:
        appui = run["aggregate"]["appui"]
        appui_v = run["aggregate"]["appui_valeur"]
        tout_v = appui.get(f"tout_tient_sur_{denom}", appui["tout_tient_sur_141"])
        tout_val = appui_v.get(f"tout_tient_sur_{denom}", appui_v["tout_tient_sur_141"])
        au_moins_v = appui.get(f"au_moins_une_sur_{denom}", appui["au_moins_une_sur_141"])
        au_moins_val = appui_v.get(f"au_moins_une_sur_{denom}", appui_v["au_moins_une_sur_141"])
        lines.append(
            f"| {run['budget']} | {tout_v}/{denom} | {tout_val}/{denom} | "
            f"{au_moins_v}/{denom} | {au_moins_val}/{denom} |"
        )
    return "\n".join(lines)


def formater_rapport(payload: dict[str, Any], *, dossier: Dossier) -> str:
    result = payload["result"]
    meta = payload["meta"]
    parties = [
        "# Plafond du filtre par l'or",
        "",
        AVERTISSEMENT,
        "",
        ENTETE,
        "",
        f"Date : {payload['date']}",
        f"Dossier : {payload['dossier']}",
        f"Budgets : {', '.join(str(b) for b in payload['budgets'])}",
        f"Vivier RAG : {payload['k_vivier']} passages",
        "",
        _table_synthese(payload),
        "",
        _format_table_par_forme(
            [result],
            dossier=dossier,
            meta=meta,
        ),
        _format_table_disputes([result], dossier=dossier, meta=meta),
        "",
    ]
    return "\n".join(parties).strip() + "\n"


def ecrire_sorties(payload: dict[str, Any], *, dossier: Dossier) -> dict[str, str]:
    out_dir = repertoire_plafonds(dossier)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "PLAFOND-FILTRE-OR.md"
    json_path = out_dir / "plafond-filtre-or.json"
    md_path.write_text(formater_rapport(payload, dossier=dossier), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"markdown": str(md_path), "json": str(json_path)}


def run_plafond(
    *,
    dossier: Dossier,
    budgets: list[int],
    connection=None,
    questions: list[dict[str, Any]] | None = None,
    gold: dict[str, Any] | None = None,
    claims_index: dict[str, dict[str, Any]] | None = None,
    holder_table: dict[str, list[dict[str, Any]]] | None = None,
    embed_cache: dict[str, list[float]] | None = None,
    vivier_fn: VivierFn | None = None,
) -> dict[str, Any]:
    all_questions = questions if questions is not None else load_questions(dossier)
    gold_payload = gold if gold is not None else load_gold(dossier)
    if questions is None:
        verifier_or_coherent(dossier, all_questions, gold_payload)
        verifier_parite(dossier)
    if vivier_fn is None or claims_index is None:
        if connection is None:
            from .config import export_runtime_env
            from .infra import connect_existing, start_cluster

            export_runtime_env()
            start_cluster()
            connection = connect_existing()
        if claims_index is None:
            claims = _read_public_claims(connection, vault_ref=dossier.vault_ref)
            claims_index = {str(c["claim_id"]): c for c in claims}
        if vivier_fn is None:
            cache = embed_cache if embed_cache is not None else _load_embed_cache(dossier)
            rag = RagNaif(cache, dossier=dossier)

            def vivier_fn(question: dict[str, Any], _rag=rag, _conn=connection) -> list[dict[str, Any]]:
                return _vivier_par_defaut(question, rag=_rag, connection=_conn)

    assert claims_index is not None
    assert vivier_fn is not None
    if holder_table is None:
        claims_list = list(claims_index.values())
        holder_table, _, _ = build_holder_table(claims_list, gold_payload)
    payload = mesurer_plafond(
        dossier=dossier,
        budgets=budgets,
        questions=all_questions,
        gold=gold_payload,
        claims_index=claims_index,
        holder_table=holder_table,
        vivier_fn=vivier_fn,
    )
    chemins = ecrire_sorties(payload, dossier=dossier)
    payload["chemins"] = chemins
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plafond du filtre par l'or — instrument de mesure qui lit l'or. "
            "Pas une ligne du comparatif."
        )
    )
    parser.add_argument(
        "--dossier",
        required=True,
        choices=("greenloop", "moreau"),
        help="Nom du dossier (greenloop, moreau)",
    )
    parser.add_argument(
        "--budgets",
        default="12000,24000",
        help="Liste de budgets caractères (défaut : 12000,24000)",
    )
    args = parser.parse_args()
    budgets = [int(item.strip()) for item in args.budgets.split(",") if item.strip()]
    payload = run_plafond(
        dossier=dossier_courant(args.dossier),
        budgets=budgets,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "instrument": "plafond_filtre_or",
                "lit_or": True,
                "ligne_du_comparatif": False,
                "dossier": payload["dossier"],
                "chemins": payload["chemins"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
