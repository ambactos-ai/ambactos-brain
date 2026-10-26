# SPDX-License-Identifier: AGPL-3.0-only
"""Témoin de brassage — mesure du bruit du mode présence valeur / verbatim."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

from .dossiers import Dossier, dossier_courant
from .fidelite import load_gold
from .or_dossier import verifier_or_coherent
from .production import verifier_parite
from .holders import classify_pair, discriminants
from .measure_disputes import _read_public_claims
from .measure_etage2 import _appui_summary, load_questions
from .mesure_comparatif_etage2 import (
    K_FIXE,
    REFERENCE_LINES,
    _charger_resultats_comparatif_complets,
    _find_reference_run,
    _format_section_reproductibilite,
    _fournisseur_label,
    _meilleur_run_pour_budget,
    _meta_mesure,
    _rows_mesurables,
    contextes_mesure_apres_budget,
    ecrire_rapport,
    k_effectif_fournisseur,
    questions_sans_appui,
    selection_questions_mesure,
)
from .normalize import piece_basename

GROK_RESULT_PATH = Path(__file__).resolve().parents[2] / "_grok-result.json"

ModePresence = Literal["verbatim", "valeur"]

EXTRA_REFERENCE_LINES: tuple[tuple[str, int | None, int], ...] = (
    ("brain_banc_g24", None, 12000),
    ("brain_banc_g36", None, 12000),
    ("brain_banc_g48", None, 12000),
    ("brain_banc_v1", None, 12000),
    ("brain_banc_v2", None, 12000),
    ("brain_banc_v3", None, 12000),
    ("brain_banc_v4", None, 12000),
    ("brain_mrfs", None, 12000),
    ("brain_mrfs_lexical", None, 12000),
    ("brain_mrfs_g12", None, 12000),
    ("brain_situations", None, 12000),
    ("brain_situations_g12", None, 12000),
    ("brain_situations_lettre", None, 12000),
    ("brain_situations_lettre_g12", None, 12000),
    ("rag_naif_large", None, 12000),
    ("rag_filtre_pieces", None, 12000),
    ("rag_filtre_situations", None, 12000),
    ("rag_filtre_union", None, 12000),
    ("rag_filtre_entites", None, 12000),
    ("entites_lignes", None, 12000),
    ("entites_lignes_puis_rag", None, 12000),
    ("rag_filtre_sens", None, 12000),
    ("objets_sens", None, 12000),
    ("objets_sens_puis_rag", None, 12000),
    ("desaccords_puis_rag", None, 12000),
    ("desaccords_puis_rag_filtre", None, 12000),
    ("desaccords_puis_g12", None, 12000),
)
REFERENCE_K_OVERRIDE: dict[str, dict[str, int | None]] = {
    "moreau": {"brain_prod": 12},
}
REFERENCE_12000_TEMOIN_VERBATIM: dict[str, dict[str, int]] = {
    "greenloop": {
        "brain_prod": 4,
        "rag_naif": 46,
        "brain_banc": 23,
        "mem0": 0,
    },
    "moreau": {
        "brain_prod": 0,
        "rag_naif": 14,
        "brain_banc": 3,
        "mem0": 0,
    },
}


def appariement_brassage(question_ids: list[str]) -> dict[str, str]:
    """Rotation déterministe : question i reçoit le contexte de i+1 (dernière → première)."""
    ordered = sorted(question_ids)
    if len(ordered) < 2:
        raise ValueError("au moins deux questions requises pour le brassage")
    return {ordered[index]: ordered[(index + 1) % len(ordered)] for index in range(len(ordered))}


def _metric_keys(mode_presence: ModePresence) -> tuple[str, str]:
    if mode_presence == "verbatim":
        return "appui_tout_tient", "appui_au_moins_une"
    return "appui_tout_tient_valeur", "appui_au_moins_une_valeur"


def _question_holds(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    mode_presence: ModePresence,
    metric: Literal["tout_tient", "au_moins_une"],
) -> bool:
    summary = _appui_summary(
        question,
        evidence,
        assertion_index=assertion_index,
        claims_index=claims_index,
        mode_presence=mode_presence,
    )
    supports = question.get("supporting_assertions") or []
    if not supports:
        return False
    if metric == "tout_tient":
        return len(summary["tient"]) == len(supports)
    return bool(summary["tient"])


def _element_haystack(element: dict[str, Any], claims_index: dict[str, dict[str, Any]]) -> str:
    if element.get("kind") == "claim_hit":
        claim_id = str(element.get("id") or "")
        claim = claims_index.get(claim_id)
        if claim is not None:
            parts = [str(claim.get("text") or "")]
            span = claim.get("source_span") or {}
            raw_line = span.get("raw_line") or ""
            if raw_line:
                parts.append(str(raw_line))
            return "\n".join(part for part in parts if part)
    return str(element.get("text") or "")


def _element_piece(element: dict[str, Any], claims_index: dict[str, dict[str, Any]]) -> str:
    if element.get("kind") == "claim_hit":
        claim_id = str(element.get("id") or "")
        claim = claims_index.get(claim_id)
        if claim is not None:
            return piece_basename(claim.get("source_file_path") or claim.get("piece") or "")
    source = str(element.get("source") or "")
    return piece_basename(source) if source else ""


def discriminant_suffisant(
    assertion: dict[str, Any],
    element: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
) -> str:
    """Libellé du discriminant (ou repli) qui explique un faux positif valeur."""
    piece_a = piece_basename(assertion.get("source") or "")
    if _element_piece(element, claims_index) != piece_a:
        return "pièce incorrecte (ne devrait pas tenir)"
    citation = str(assertion.get("quote") or assertion.get("content") or "")
    disc = discriminants(citation)
    parts: list[str] = []
    for number, anchor in disc["nombres"]:
        label = f"nombre {number:g}"
        if anchor:
            label += f" ({anchor})"
        parts.append(label)
    for date_key in disc["dates"]:
        parts.append(f"date {date_key[2]:02d}/{date_key[1]:02d}/{date_key[0]}")
    for token in disc["textes"]:
        parts.append(f"texte «{token}»")
    if not parts:
        return "repli lexical (recouvrance substantielle)"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts)


def _assertion_tient_valeur(
    assertion: dict[str, Any],
    evidence: list[dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None]:
    for item in evidence:
        if classify_pair(assertion, item, claims_index, mode_presence="valeur") == "tient":
            return True, item
    return False, None


def compter_brassage(
    questions: list[dict[str, Any]],
    contexts: dict[str, list[dict[str, Any]]],
    *,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    mode_presence: ModePresence,
    metric: Literal["tout_tient", "au_moins_une"] = "tout_tient",
    pairing: dict[str, str] | None = None,
) -> dict[str, Any]:
    ordered_ids = sorted(q["id"] for q in questions)
    pairing = pairing or appariement_brassage(ordered_ids)
    signal = 0
    temoin = 0
    for question in questions:
        qid = question["id"]
        own = contexts.get(qid, [])
        foreign = contexts.get(pairing[qid], [])
        if _question_holds(
            question,
            own,
            assertion_index=assertion_index,
            claims_index=claims_index,
            mode_presence=mode_presence,
            metric=metric,
        ):
            signal += 1
        if _question_holds(
            question,
            foreign,
            assertion_index=assertion_index,
            claims_index=claims_index,
            mode_presence=mode_presence,
            metric=metric,
        ):
            temoin += 1
    return {
        "signal": signal,
        "temoin": temoin,
        "ecart": signal - temoin,
        "total": len(questions),
        "pairing": pairing,
    }


def assertions_temoin_valeur(
    questions: list[dict[str, Any]],
    contexts: dict[str, list[dict[str, Any]]],
    *,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    pairing: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    ordered_ids = sorted(q["id"] for q in questions)
    pairing = pairing or appariement_brassage(ordered_ids)
    rows: list[dict[str, str]] = []
    for question in questions:
        qid = question["id"]
        foreign = contexts.get(pairing[qid], [])
        for aid in question.get("supporting_assertions") or []:
            assertion = assertion_index[aid]
            holds, element = _assertion_tient_valeur(assertion, foreign, claims_index)
            if not holds or element is None:
                continue
            citation = str(assertion.get("quote") or assertion.get("content") or "")
            rows.append(
                {
                    "assertion_id": aid,
                    "question_id": qid,
                    "contexte_de": pairing[qid],
                    "piece": piece_basename(assertion.get("source") or ""),
                    "citation": citation[:80] + ("…" if len(citation) > 80 else ""),
                    "discriminant": discriminant_suffisant(assertion, element, claims_index),
                }
            )
    return rows


def charger_signal_etage2(
    fournisseur: str,
    *,
    k: int | None,
    budget: int,
    k_fixe: bool,
    metric: Literal["tout_tient", "au_moins_une"] = "tout_tient",
    dossier: Dossier | None = None,
) -> dict[str, int]:
    current = dossier if dossier is not None else dossier_courant()
    path = current.comparatif_dir / f"etage2-{fournisseur}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    run = _find_reference_run(payload, k, budget)
    if run is None:
        raise FileNotFoundError(f"run de référence absent pour {fournisseur} k={k} budget={budget}")
    mesurables = _rows_mesurables(run["questions"])
    out: dict[str, int] = {}
    for mode_presence in ("verbatim", "valeur"):
        tout_key, au_moins_key = _metric_keys(mode_presence)
        key = tout_key if metric == "tout_tient" else au_moins_key
        out[mode_presence] = sum(1 for row in mesurables if row.get(key))
    return out


def _effective_k(nom: str, k: int | None) -> int:
    return k_effectif_fournisseur(nom, k)


def charger_contextes_mesure(
    nom: str,
    questions: list[dict[str, Any]],
    *,
    k: int | None,
    budget: int,
    dossier: Dossier | None = None,
) -> dict[str, list[dict[str, Any]]]:
    current = dossier if dossier is not None else dossier_courant()
    return contextes_mesure_apres_budget(
        nom,
        questions,
        k=k,
        budget=budget,
        dossier=current,
    )


def _lecture_ecarts(
    lignes: list[dict[str, Any]],
    *,
    fournisseurs_attendus: list[str] | None = None,
) -> str:
    mesures = sorted({row["fournisseur"] for row in lignes})
    phrases: list[str] = []
    if fournisseurs_attendus and set(mesures) != set(fournisseurs_attendus):
        manquants = sorted(set(fournisseurs_attendus) - set(mesures))
        phrases.append(
            f"Exécution partielle : {len(mesures)} fournisseur(s) mesuré(s) "
            f"({', '.join(mesures)}). Non mesurés : {', '.join(manquants)}."
        )
    by_key = {(row["fournisseur"], row["mode"]): row for row in lignes}
    for nom in mesures:
        val = by_key.get((nom, "valeur"))
        if val is None:
            continue
        phrases.append(
            f"{nom} : signal valeur {val['signal']}/{val['total']}, "
            f"témoin {val['temoin']}/{val['total']} (écart {val['ecart']})."
        )
    return " ".join(phrases) if phrases else "_Aucune mesure disponible._"


def formater_tableau(lignes: list[dict[str, Any]]) -> str:
    header = (
        "| fournisseur | mode | signal (vrai appariement) | témoin (brassé) | écart |\n"
        "| --- | --- | --- | --- | --- |"
    )
    body = [
        f"| {row['fournisseur']} | {row['mode']} | "
        f"{row['signal']}/{row['total']} | {row['temoin']}/{row['total']} | {row['ecart']} |"
        for row in lignes
    ]
    return "\n".join([header, *body])


def formater_grille_temoin(
    lignes: list[dict[str, Any]],
    *,
    budgets: list[int],
    dossier: Dossier,
) -> str:
    by_key = {
        (row["fournisseur"], int(row["budget"]), row["mode"]): row
        for row in lignes
        if row.get("budget") is not None
    }
    lines = [
        "",
        "## Grille témoin (budget × fournisseur)",
        "",
        "k ou variante : même règle que la grille « à unités et budget égaux ». "
        "Métrique : **tout tient**, mode verbatim (colonne témoin) et valeur (colonne témoin valeur).",
        "",
        "| Budget | Fournisseur | k / variante | témoin (verbatim) | témoin (valeur) | écart (verbatim) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    fournisseurs = sorted({row["fournisseur"] for row in lignes if row.get("budget") is not None})
    for budget in budgets:
        for nom in fournisseurs:
            verb = by_key.get((nom, budget, "verbatim"))
            val = by_key.get((nom, budget, "valeur"))
            if verb is None and val is None:
                continue
            k = verb.get("k") if verb is not None else val.get("k")
            k_fixe = nom in K_FIXE
            k_label = "figé" if k_fixe else str(k)
            label = _fournisseur_label(nom, k)
            total = (verb or val)["total"]
            temoin_v = f"{verb['temoin']}/{total}" if verb is not None else "—"
            temoin_val = f"{val['temoin']}/{total}" if val is not None else "—"
            ecart = str(verb["ecart"]) if verb is not None else "—"
            lines.append(
                f"| {budget} | {label} | {k_label} | {temoin_v} | {temoin_val} | {ecart} |"
            )
    lines.append("")
    return "\n".join(lines)


def verifier_reference_12000_inchangee(
    lignes: list[dict[str, Any]],
    *,
    dossier: Dossier,
    fournisseurs: list[str] | None = None,
) -> bool:
    attendus = REFERENCE_12000_TEMOIN_VERBATIM.get(dossier.nom, {})
    for nom, attendu in attendus.items():
        if fournisseurs is not None and nom not in fournisseurs:
            continue
        row = next(
            (
                item
                for item in lignes
                if item["fournisseur"] == nom
                and item["mode"] == "verbatim"
                and int(item.get("budget") or 12000) == 12000
            ),
            None,
        )
        if row is None or int(row["temoin"]) != attendu:
            return False
    return True


def formater_assertions_scandaleuses(rows: list[dict[str, str]], limit: int = 20) -> str:
    if not rows:
        return "_Aucune assertion tenue par le témoin en mode valeur._\n"
    lines = [
        "",
        "## Assertions tenues par le témoin (mode valeur)",
        "",
        "| assertion | pièce | citation (80 car.) | discriminant |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| {row['assertion_id']} | {row['piece']} | {row['citation']} | {row['discriminant']} |"
        )
    if len(rows) > limit:
        lines.append(f"\n_… et {len(rows) - limit} autres assertions._")
    return "\n".join(lines) + "\n"


TRACE_RESSERREMENT = (
    "Instrument resserré le 2026-09-24 sur revue indépendante (#129) ; "
    "les chiffres « avant » ci-dessous restent la référence publiée avant ce resserrement."
)
TRACE_JSON_REGENERES = "JSON régénérés le 25/09 avec l'instrument courant."

AVANT_RESSERREMENT: list[tuple[str, str, int, int]] = [
    ("brain_prod", "verbatim", 39, 4),
    ("brain_prod", "valeur", 49, 4),
    ("rag_naif", "verbatim", 85, 46),
    ("rag_naif", "valeur", 90, 48),
    ("brain_banc", "verbatim", 84, 23),
    ("brain_banc", "valeur", 89, 32),
    ("mem0", "verbatim", 2, 0),
    ("mem0", "valeur", 47, 19),
]

APRES_RESSERREMENT: list[tuple[str, str, int, int]] = [
    ("brain_prod", "verbatim", 39, 4),
    ("brain_prod", "valeur", 37, 4),
    ("rag_naif", "verbatim", 85, 46),
    ("rag_naif", "valeur", 71, 43),
    ("brain_banc", "verbatim", 84, 23),
    ("brain_banc", "valeur", 75, 22),
    ("mem0", "verbatim", 2, 0),
    ("mem0", "valeur", 28, 15),
]


def formater_tableau_avant_apres(
    lignes: list[dict[str, Any]],
    *,
    avant: list[tuple[str, str, int, int]] | None = None,
    apres_resserrement: list[tuple[str, str, int, int]] | None = None,
) -> str:
    avant_rows = avant or AVANT_RESSERREMENT
    resserrement_rows = apres_resserrement or APRES_RESSERREMENT
    by_key = {(row["fournisseur"], row["mode"]): row for row in lignes}
    resserrement_by_key = {
        (fournisseur, mode): (signal, temoin)
        for fournisseur, mode, signal, temoin in resserrement_rows
    }
    header = (
        "| fournisseur | mode | signal avant | témoin avant | "
        "signal après resserrement | témoin après resserrement | "
        "signal après correction | témoin après correction | écart correction |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    body: list[str] = []
    for fournisseur, mode, signal_avant, temoin_avant in avant_rows:
        signal_resserrement, temoin_resserrement = resserrement_by_key.get(
            (fournisseur, mode),
            ("—", "—"),
        )
        apres = by_key.get((fournisseur, mode), {})
        signal_correction = apres.get("signal", "—")
        temoin_correction = apres.get("temoin", "—")
        ecart_correction = apres.get("ecart", "—")
        body.append(
            f"| {fournisseur} | {mode} | {signal_avant} | {temoin_avant} | "
            f"{signal_resserrement} | {temoin_resserrement} | "
            f"{signal_correction} | {temoin_correction} | {ecart_correction} |"
        )
    return "\n".join([header, *body])


def _formater_hors_etage2(
    *,
    non_etabli: list[dict[str, Any]],
    sans_appui: list[dict[str, Any]],
) -> str:
    lines = ["", "## Hors étage 2", ""]
    if non_etabli:
        ids = ", ".join(question["id"] for question in sorted(non_etabli, key=lambda q: q["id"]))
        lines.append(f"- `non_etabli` ({len(non_etabli)}) : {ids}.")
    if sans_appui:
        ids = ", ".join(
            f"{question['id']} ({question.get('shape') or 'unknown'})"
            for question in sorted(sans_appui, key=lambda q: q["id"])
        )
        lines.append(
            f"- sans assertion d'appui ({len(sans_appui)}) : {ids} "
            "(jamais « tout tient », listées pour traçabilité)."
        )
    if len(lines) == 3:
        lines.append("_Aucune question hors périmètre mesurable._")
    return "\n".join(lines)


def ecrire_temoin_rapport(
    lignes: list[dict[str, Any]],
    assertions_par_fournisseur: dict[str, list[dict[str, str]]],
    *,
    fournisseurs_attendus: list[str] | None = None,
    meta_mesure: dict[str, Any] | None = None,
    hors_etage2: dict[str, list[dict[str, Any]]] | None = None,
    stabilite_par_fournisseur: dict[str, dict[str, Any]] | None = None,
    grille_lignes: list[dict[str, Any]] | None = None,
    budgets_grille: list[int] | None = None,
    dossier: Dossier | None = None,
) -> None:
    current = dossier if dossier is not None else dossier_courant()
    temoin_path = current.comparatif_dir / "TEMOIN-BRASSAGE.md"
    current.comparatif_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_mesure or _meta_mesure(current, load_questions(current))
    denom = meta["denominateur_appui"]
    hors = hors_etage2 or {
        "non_etabli": [q for q in load_questions(current) if q.get("shape") == "non_etabli"],
        "sans_appui": questions_sans_appui(load_questions(current)),
    }
    historique_greenloop = current.nom == "greenloop"
    text = [
        "# Témoin de brassage",
        "",
        f"Date : {date.today().isoformat()}",
        "",
    ]
    if historique_greenloop:
        text.extend([TRACE_RESSERREMENT, TRACE_JSON_REGENERES, ""])
    else:
        text.extend(["MOREAU n'a pas servi au réglage.", ""])
    text.extend(
        [
            "Appariement : rotation d'un cran sur les questions triées par identifiant "
            "(question *i* évaluée contre le contexte de *i+1*).",
            "",
            "Métrique : **tout tient** (toutes les assertions de soutien présentes dans le contexte).",
            "",
            f"Dénominateur appui : **{denom}** (même sélection que la mesure étage 2).",
            "",
            _formater_hors_etage2(
                non_etabli=hors["non_etabli"],
                sans_appui=hors["sans_appui"],
            ),
            "",
        ]
    )
    if historique_greenloop:
        text.extend(
            [
                "## Avant / après resserrement / après correction invariant",
                "",
                formater_tableau_avant_apres(lignes),
                "",
            ]
        )
    text.extend(
        [
            formater_tableau(lignes),
            "",
            "## Lecture",
            "",
            _lecture_ecarts(lignes, fournisseurs_attendus=fournisseurs_attendus),
            "",
        ]
    )
    if historique_greenloop:
        for nom in sorted(assertions_par_fournisseur):
            rows = assertions_par_fournisseur[nom]
            if not rows:
                continue
            text.append(f"### {nom}")
            text.append(formater_assertions_scandaleuses(rows, limit=50).rstrip())
            text.append("")
    if stabilite_par_fournisseur:
        text.append(_format_section_reproductibilite(stabilite_par_fournisseur))
        text.append("")
    if grille_lignes and budgets_grille:
        text.append(
            formater_grille_temoin(
                grille_lignes,
                budgets=budgets_grille,
                dossier=current,
            ).strip()
        )
        text.append("")
    temoin_path.write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")


def injecter_section_rapport_comparatif(
    lignes: list[dict[str, Any]],
    *,
    fournisseurs_attendus: list[str] | None = None,
    dossier: Dossier | None = None,
) -> None:
    current = dossier if dossier is not None else dossier_courant()
    rapport_path = current.comparatif_dir / "RAPPORT-COMPARATIF-ETAGE2.md"
    marker = "## Témoin de brassage"
    section = [
        "",
        marker,
        "",
        f"Date : {date.today().isoformat()}",
        "",
        TRACE_RESSERREMENT,
        "",
        "### Avant / après resserrement / après correction invariant",
        "",
        formater_tableau_avant_apres(lignes),
        "",
        formater_tableau(lignes),
        "",
        _lecture_ecarts(lignes, fournisseurs_attendus=fournisseurs_attendus),
        "",
    ]
    if rapport_path.is_file():
        existing = rapport_path.read_text(encoding="utf-8")
        if marker in existing:
            before = existing.split(marker, 1)[0].rstrip()
            rapport_path.write_text(before + "\n".join(section), encoding="utf-8")
            return
        rapport_path.write_text(existing.rstrip() + "\n".join(section), encoding="utf-8")
    else:
        rapport_path.parent.mkdir(parents=True, exist_ok=True)
        rapport_path.write_text("\n".join(section).lstrip(), encoding="utf-8")


def mesurer_fournisseur_temoin(
    nom: str,
    questions_mesure: list[dict[str, Any]],
    *,
    k: int | None,
    budget: int,
    k_fixe: bool,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    contexts: dict[str, list[dict[str, Any]]],
    signal_etage2: dict[str, int],
    denominateur: int,
    verifier_signal: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    pairing = appariement_brassage(sorted(q["id"] for q in questions_mesure))
    lignes: list[dict[str, Any]] = []
    for mode_presence in ("verbatim", "valeur"):
        recomputed = compter_brassage(
            questions_mesure,
            contexts,
            assertion_index=assertion_index,
            claims_index=claims_index,
            mode_presence=mode_presence,
            metric="tout_tient",
            pairing=pairing,
        )
        signal = signal_etage2[mode_presence]
        temoin = recomputed["temoin"]
        if verifier_signal and recomputed["signal"] != signal:
            raise AssertionError(
                f"signal {mode_presence} recomputé ({recomputed['signal']}) ≠ etage2 ({signal}) "
                f"pour {nom}"
            )
        if recomputed["total"] != denominateur:
            raise AssertionError(
                f"dénominateur témoin ({recomputed['total']}) ≠ mesure ({denominateur}) pour {nom}"
            )
        lignes.append(
            {
                "fournisseur": nom,
                "mode": mode_presence,
                "signal": signal,
                "temoin": temoin,
                "ecart": signal - temoin,
                "total": denominateur,
            }
        )
    assertions = assertions_temoin_valeur(
        questions_mesure,
        contexts,
        assertion_index=assertion_index,
        claims_index=claims_index,
        pairing=pairing,
    )
    return lignes, assertions


def _reference_config(
    dossier: Dossier,
    fournisseurs: list[str],
) -> dict[str, tuple[int | None, int, bool]]:
    ref_by_name = {
        nom: (k, budget, nom in K_FIXE) for nom, k, budget in REFERENCE_LINES
    }
    for nom, k, budget in EXTRA_REFERENCE_LINES:
        ref_by_name.setdefault(nom, (k, budget, nom in K_FIXE))
    for nom, k in REFERENCE_K_OVERRIDE.get(dossier.nom, {}).items():
        if nom in ref_by_name:
            _k, budget, k_fixe = ref_by_name[nom]
            ref_by_name[nom] = (k, budget, k_fixe)
    if (
        dossier.nom == "moreau"
        and "brain_banc_calibre" in fournisseurs
        and "brain_banc_calibre" not in ref_by_name
    ):
        ref_by_name["brain_banc_calibre"] = (None, 12000, True)
    return ref_by_name


def _mesurer_reference_lignes(
    fournisseurs: list[str],
    *,
    ref_by_name: dict[str, tuple[int | None, int, bool]],
    questions: list[dict[str, Any]],
    questions_mesure: list[dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    denominateur: int,
    dossier: Dossier,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    lignes: list[dict[str, Any]] = []
    assertions_par_fournisseur: dict[str, list[dict[str, str]]] = {}
    for nom in fournisseurs:
        if nom not in ref_by_name:
            raise ValueError(f"fournisseur hors référence : {nom}")
        k, budget, k_fixe = ref_by_name[nom]
        signal_etage2 = charger_signal_etage2(
            nom, k=k, budget=budget, k_fixe=k_fixe, dossier=dossier
        )
        contexts = charger_contextes_mesure(
            nom,
            questions,
            k=k,
            budget=budget,
            dossier=dossier,
        )
        rows, assertions = mesurer_fournisseur_temoin(
            nom,
            questions_mesure,
            k=k,
            budget=budget,
            k_fixe=k_fixe,
            assertion_index=assertion_index,
            claims_index=claims_index,
            contexts=contexts,
            signal_etage2=signal_etage2,
            denominateur=denominateur,
        )
        for row in rows:
            row["budget"] = budget
            row["k"] = k
        lignes.extend(rows)
        assertions_par_fournisseur[nom] = assertions
    return lignes, assertions_par_fournisseur


def _signal_depuis_run(run: dict[str, Any], denominateur: int) -> dict[str, int]:
    appui = run["aggregate"]["appui"]
    appui_v = run["aggregate"]["appui_valeur"]
    key = f"tout_tient_sur_{denominateur}"
    return {
        "verbatim": int(appui.get(key, appui["tout_tient_sur_141"])),
        "valeur": int(appui_v.get(key, appui_v["tout_tient_sur_141"])),
    }


def _mesurer_grille_temoin(
    fournisseurs: list[str],
    budgets: list[int],
    results: list[dict[str, Any]],
    *,
    questions: list[dict[str, Any]],
    questions_mesure: list[dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    denominateur: int,
    dossier: Dossier,
) -> list[dict[str, Any]]:
    by_name = {result["fournisseur"]: result for result in results}
    grille_lignes: list[dict[str, Any]] = []
    for nom in fournisseurs:
        result = by_name.get(nom)
        if result is None:
            continue
        for budget in budgets:
            run = _meilleur_run_pour_budget(result, budget)
            if run is None:
                continue
            k = run.get("k")
            signal_etage2 = _signal_depuis_run(run, denominateur)
            contexts = charger_contextes_mesure(
                nom,
                questions,
                k=k,
                budget=budget,
                dossier=dossier,
            )
            rows, _assertions = mesurer_fournisseur_temoin(
                nom,
                questions_mesure,
                k=k,
                budget=budget,
                k_fixe=result["k_fixe"],
                assertion_index=assertion_index,
                claims_index=claims_index,
                contexts=contexts,
                signal_etage2=signal_etage2,
                denominateur=denominateur,
                verifier_signal=False,
            )
            for row in rows:
                row["budget"] = budget
                row["k"] = k
            grille_lignes.extend(rows)
    return grille_lignes


def run_temoin_brassage(
    *,
    fournisseurs: list[str] | None = None,
    budgets: list[int] | None = None,
    dossier: Dossier | None = None,
    stabilite_par_fournisseur: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    current = dossier if dossier is not None else dossier_courant()
    fournisseurs = fournisseurs or [nom for nom, _k, _budget in REFERENCE_LINES]
    if current.nom == "moreau":
        cal_path = current.comparatif_dir / "etage2-brain_banc_calibre.json"
        if cal_path.is_file() and "brain_banc_calibre" not in fournisseurs:
            fournisseurs = [*fournisseurs, "brain_banc_calibre"]
    questions = load_questions(current)
    questions_mesure = selection_questions_mesure(questions)
    meta_mesure = _meta_mesure(current, questions)
    denominateur = meta_mesure["denominateur_appui"]
    hors_etage2 = {
        "non_etabli": [q for q in questions if q.get("shape") == "non_etabli"],
        "sans_appui": questions_sans_appui(questions),
    }
    gold = load_gold(current)
    verifier_or_coherent(current, questions, gold)
    verifier_parite(current)
    assertion_index = gold["assertion_index"]
    claims_index: dict[str, dict[str, Any]] = {}
    from .config import export_runtime_env
    from .infra import connect_existing, start_cluster

    export_runtime_env()
    start_cluster()
    connection = connect_existing()
    claims = _read_public_claims(connection, vault_ref=current.vault_ref)
    claims_index = {str(c["claim_id"]): c for c in claims}

    ref_by_name = _reference_config(current, fournisseurs)
    reference_lignes, assertions_par_fournisseur = _mesurer_reference_lignes(
        fournisseurs,
        ref_by_name=ref_by_name,
        questions=questions,
        questions_mesure=questions_mesure,
        assertion_index=assertion_index,
        claims_index=claims_index,
        denominateur=denominateur,
        dossier=current,
    )
    if not verifier_reference_12000_inchangee(
        reference_lignes,
        dossier=current,
        fournisseurs=fournisseurs,
    ):
        attendus = REFERENCE_12000_TEMOIN_VERBATIM.get(current.nom, {})
        obtenus = {
            row["fournisseur"]: row["temoin"]
            for row in reference_lignes
            if row["mode"] == "verbatim" and int(row.get("budget") or 12000) == 12000
        }
        raise RuntimeError(
            f"témoin référence 12 000 modifié pour {current.nom} : "
            f"attendu {attendus}, obtenu {obtenus}"
        )

    merged = _charger_resultats_comparatif_complets(current, [])
    fournisseurs_grille = [
        result["fournisseur"]
        for result in merged
        if result["fournisseur"] not in {"letta_archive"}
    ]
    budgets_grille = budgets or sorted(
        {int(run["budget"]) for result in merged for run in result["runs"]}
    )
    grille_lignes = _mesurer_grille_temoin(
        fournisseurs_grille,
        budgets_grille,
        merged,
        questions=questions,
        questions_mesure=questions_mesure,
        assertion_index=assertion_index,
        claims_index=claims_index,
        denominateur=denominateur,
        dossier=current,
    )

    ecrire_temoin_rapport(
        reference_lignes,
        assertions_par_fournisseur,
        fournisseurs_attendus=fournisseurs,
        meta_mesure=meta_mesure,
        hors_etage2=hors_etage2,
        stabilite_par_fournisseur=stabilite_par_fournisseur,
        grille_lignes=grille_lignes,
        budgets_grille=budgets_grille,
        dossier=current,
    )
    if current.nom == "greenloop":
        injecter_section_rapport_comparatif(
            reference_lignes,
            fournisseurs_attendus=fournisseurs,
            dossier=current,
        )
    ecrire_rapport(
        merged,
        dossier=current,
        questions=questions,
        temoin_lignes=grille_lignes,
        budgets_grille=budgets_grille,
    )

    temoin_verbatim: dict[str, int] = {}
    temoin_valeur: dict[str, int] = {}
    for row in reference_lignes:
        if row["mode"] == "verbatim":
            temoin_verbatim[row["fournisseur"]] = row["temoin"]
        else:
            temoin_valeur[row["fournisseur"]] = row["temoin"]

    apres: dict[str, dict[str, list[int]]] = {}
    for row in reference_lignes:
        nom = row["fournisseur"]
        mode = row["mode"]
        apres.setdefault(nom, {})[mode] = [row["signal"], row["temoin"]]

    temoin_grille: dict[str, dict[str, list[int]]] = {}
    for row in grille_lignes:
        if row["mode"] != "verbatim":
            continue
        nom = row["fournisseur"]
        budget_key = str(int(row["budget"]))
        temoin_grille.setdefault(nom, {})[budget_key] = [
            int(row["temoin"]),
            int(next(
                item["temoin"]
                for item in grille_lignes
                if item["fournisseur"] == nom
                and item["mode"] == "valeur"
                and int(item["budget"]) == int(row["budget"])
            )),
        ]

    avant_verbatim = {
        fournisseur: signal
        for fournisseur, mode, signal, _temoin in AVANT_RESSERREMENT
        if mode == "verbatim"
    }
    verbatim_inchange = all(
        row["signal"] == avant_verbatim[row["fournisseur"]]
        for row in reference_lignes
        if row["mode"] == "verbatim" and row["fournisseur"] in avant_verbatim
    )
    valeur_jamais_sous_verbatim = all(
        row["signal"]
        >= next(
            s
            for r in reference_lignes
            if r["fournisseur"] == row["fournisseur"] and r["mode"] == "verbatim"
            for s in [r["signal"]]
        )
        for row in reference_lignes
        if row["mode"] == "valeur"
    )
    return {
        "status": "complete",
        "lignes": reference_lignes,
        "grille_lignes": grille_lignes,
        "apres": apres,
        "temoin_grille": temoin_grille,
        "verbatim_inchange": verbatim_inchange,
        "valeur_jamais_sous_verbatim": valeur_jamais_sous_verbatim,
        "reference_12000_inchangee": True,
        "temoin_verbatim": temoin_verbatim,
        "temoin_valeur": temoin_valeur,
        "assertions_par_fournisseur": assertions_par_fournisseur,
        "dossier": current.nom,
        "denominateur_appui": denominateur,
        "meta_mesure": meta_mesure,
        "hors_etage2": hors_etage2,
        "budgets_grille": budgets_grille,
    }


def compter_invariant_ecarts(dossier: Dossier | None = None) -> dict[str, int]:
    current = dossier if dossier is not None else dossier_courant()
    ecarts: dict[str, int] = {}
    for nom in ("greenloop", "moreau"):
        gold = load_gold(nom)
        count = 0
        for assertion in gold["assertions"]:
            quote = str(assertion.get("quote") or assertion.get("content") or "")
            source = str(assertion.get("source") or "")
            element = {
                "kind": "file_hit",
                "id": f"inv:{assertion['id']}",
                "text": quote,
                "source": source,
            }
            if classify_pair(assertion, element, {}, mode_presence="verbatim") != "tient":
                continue
            if classify_pair(assertion, element, {}, mode_presence="valeur") != "tient":
                count += 1
        ecarts[nom] = count
    return ecarts


def ecrire_grok_result(
    *,
    tests_passed: int,
    greenloop_payload: dict[str, Any],
    moreau_payload: dict[str, Any],
    reference_12000_inchangee: bool = True,
) -> None:
    def _temoin_grille(source: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
        return source.get("temoin_grille") or {}

    body: dict[str, Any] = {
        "status": "complete",
        "temoin": {
            "greenloop": _temoin_grille(greenloop_payload),
            "moreau": _temoin_grille(moreau_payload),
        },
        "reference_12000_inchangee": reference_12000_inchangee,
        "tests": f"{tests_passed} verts",
    }
    GROK_RESULT_PATH.write_text(json.dumps(body, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Témoin de brassage comparatif étage 2")
    parser.add_argument(
        "--fournisseurs",
        default=",".join(nom for nom, _k, _budget in REFERENCE_LINES),
        help="Liste séparée par des virgules",
    )
    parser.add_argument(
        "--budgets",
        default="",
        help="Liste de budgets caractères pour la grille témoin (défaut : tous les runs etage2)",
    )
    parser.add_argument(
        "--tests-passed",
        type=int,
        default=0,
        help="Nombre de tests verts (pour _grok-result.json)",
    )
    parser.add_argument(
        "--dossier",
        default="greenloop",
        help="Nom du dossier (greenloop, moreau)",
    )
    args = parser.parse_args()
    fournisseurs = [item.strip() for item in args.fournisseurs.split(",") if item.strip()]
    budgets = [int(item.strip()) for item in args.budgets.split(",") if item.strip()] or None
    payload = run_temoin_brassage(
        fournisseurs=fournisseurs,
        budgets=budgets,
        dossier=dossier_courant(args.dossier),
    )
    if args.tests_passed > 0:
        ecrire_grok_result(
            tests_passed=args.tests_passed,
            greenloop_payload=payload if args.dossier == "greenloop" else {},
            moreau_payload=payload if args.dossier == "moreau" else {},
            reference_12000_inchangee=payload.get("reference_12000_inchangee", True),
        )
    print(json.dumps({"status": payload["status"], "fournisseurs": fournisseurs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
