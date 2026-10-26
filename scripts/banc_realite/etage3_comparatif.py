# SPDX-License-Identifier: AGPL-3.0-only
"""Étage 3 comparatif — réponses par fournisseur de contexte (MOREAU d'abord)."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .config import BANC_ROOT
from .codex_chat import CodexQuotaAtteint, assert_codex_model_slug_allowed, make_codex_chat_fn
from .cursor_chat import assert_cursor_model_available, make_cursor_chat_fn
from .dossiers import Dossier, dossier_courant
from .etage3 import (
    CONTEXT_A,
    ChatFn,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FR,
    adjudicate_judge,
    append_call,
    build_response_invite,
    compute_verdict_adjudique,
    count_calls,
    detect_formes_iso,
    detect_response_refus,
    estimate_invite_tokens,
    invite_id as _invite_id,
    is_response_illisible,
    load_journal_records,
    load_journal_state,
    must_mention_for_verdict,
    must_mention_present,
    must_mention_scoring_fields_adjudique,
    system_prompt_for_langue,
    usage_relais_total,
    verify_invite_hash,
)
from .fournisseurs_contexte import _elements_to_evidence, rendre_contexte
from .holders import (
    _anchored_number_present,
    _dates_in_text,
    _text_token_present,
    discriminants,
)
from .mesure_comparatif_etage2 import (
    K_FIXE,
    REFERENCE_LINES,
    _find_reference_run,
    _meta_mesure,
    _render_style,
    k_effectif_fournisseur,
    lire_contextes_mesure_bruts,
)
from .normalize import normalize_text, piece_basename
from .or_dossier import load_gold, load_questions

REPONDANT_MODEL = "cursor-grok-4.5-high"
JUGE_MODEL = "gpt-5.6-sol-high"
JUGE_MODEL_CODEX = "gpt-5.6-sol"
MAX_REPONDANT_CALLS = 320
MAX_JUGE_CALLS = 260
STABILITY_QUESTIONS = 5
STABILITY_REJEUX = 3
SEED = 20260925
GROK_RESULT_PATH = Path(__file__).resolve().parents[2] / "_grok-result.json"
REFERENCE_GREENLOOP_ETAGE3_20260923 = {
    "label": "fine hybride k12+k6 (2026-09-23, juge Grok 4.5 adjudication #106)",
    "juge": "Grok 4.5 (adjudication #106)",
    "conformes": "75/141",
    "deux_cotes": "26/28",
}

BRAS_A = "A"
FOURNISSEURS: tuple[tuple[str, int | None, int], ...] = REFERENCE_LINES
_plafond_repondant: int = MAX_REPONDANT_CALLS
_plafond_juge: int = MAX_JUGE_CALLS
ALL_BRAS: tuple[tuple[str, int | None, int | None], ...] = (
    (BRAS_A, None, None),
    *FOURNISSEURS,
)
FOURNISSEUR_CONTEXTE_BRAS = frozenset(nom for nom, _k, _b in FOURNISSEURS)


def reference_lines(dossier: Dossier) -> tuple[tuple[str, int | None, int], ...]:
    """Lignes de référence étage 2 pour l'étage 3 comparatif (GreenLoop : rag_naif k=12, etc.)."""
    _ = dossier
    return REFERENCE_LINES


def plafond_avec_marge(comptage: int) -> int:
    return math.ceil(comptage * 1.03)


def init_plafonds(
    dossier: Dossier,
    *,
    plafond_repondant: int | None = None,
    plafond_juge: int | None = None,
    lignes_supplementaires: Sequence[tuple[str, int | None, int]] = (),
) -> dict[str, int]:
    global _plafond_repondant, _plafond_juge
    questions = load_questions(dossier)
    comptage = comptage_a_blanc(questions)
    extra = (
        comptage_lignes(questions, lignes_supplementaires)
        if lignes_supplementaires
        else {"repondant": 0, "juge": 0}
    )
    total_repondant = comptage["repondant"] + extra["repondant"]
    total_juge = comptage["juge"] + extra["juge"]
    _plafond_repondant = (
        plafond_repondant
        if plafond_repondant is not None
        else plafond_avec_marge(total_repondant)
    )
    _plafond_juge = (
        plafond_juge if plafond_juge is not None else plafond_avec_marge(total_juge)
    )
    return {
        "repondant": _plafond_repondant,
        "juge": _plafond_juge,
        "comptage_a_blanc": comptage["repondant"],
        "comptage_juge_a_blanc": comptage["juge"],
    }


def plafond_repondant_actif() -> int:
    return _plafond_repondant


def plafond_juge_actif() -> int:
    return _plafond_juge


def _shape_refus_attendu(shape: str) -> bool:
    return shape in ("non_etabli", "unknown")


def _shape_deux_cotes(shape: str, dossier: Dossier) -> bool:
    if dossier.nature == "juridique":
        return shape == "contradiction"
    return shape == "dispute"


def etage3_dir(dossier: Dossier | None = None) -> Path:
    current = dossier if dossier is not None else dossier_courant("moreau")
    if current.nom == "greenloop":
        return BANC_ROOT / "sorties" / "etage3-comparatif"
    return BANC_ROOT / "dossiers" / current.nom / "etage3"


def etage3_v1_dir(dossier: Dossier | None = None) -> Path:
    current = dossier if dossier is not None else dossier_courant("moreau")
    return BANC_ROOT / "dossiers" / current.nom / "etage3-v1-anglais"


def invites_path(dossier: Dossier | None = None) -> Path:
    return etage3_dir(dossier) / "invites-reponses.jsonl"


def appels_path(dossier: Dossier | None = None) -> Path:
    return etage3_dir(dossier) / "appels.jsonl"


def rapport_path(dossier: Dossier | None = None) -> Path:
    return etage3_dir(dossier) / "RAPPORT-ETAGE3-COMPARATIF.md"


def rapport_blanc_path(dossier: Dossier | None = None) -> Path:
    return etage3_dir(dossier) / "RAPPORT-ETAGE3-A-BLANC.md"


def notations_path(bras: str, dossier: Dossier | None = None) -> Path:
    return etage3_dir(dossier) / f"notations-{bras}.json"


def rapport_situations_path(dossier: Dossier | None = None) -> Path:
    return etage3_dir(dossier) / "RAPPORT-ETAGE3-SITUATIONS.md"


def k_pour_ligne_supplementaire(nom: str) -> int | None:
    if nom in K_FIXE:
        return None
    for ligne_nom, k, _budget in REFERENCE_LINES:
        if ligne_nom == nom:
            return k
    return k_effectif_fournisseur(nom)


def parse_lignes_supplementaires(
    spec: str | None,
) -> tuple[tuple[str, int | None, int], ...]:
    if not spec or not str(spec).strip():
        return ()
    lignes: list[tuple[str, int | None, int]] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"ligne supplémentaire mal formée : {part}")
        nom, budget_s = part.split(":", 1)
        nom = nom.strip()
        budget = int(budget_s.strip())
        lignes.append((nom, k_pour_ligne_supplementaire(nom), budget))
    return tuple(lignes)


def lignes_contexte(
    lignes_supplementaires: Sequence[tuple[str, int | None, int]] | None = None,
) -> tuple[tuple[str, int | None, int], ...]:
    if not lignes_supplementaires:
        return FOURNISSEURS
    return tuple(FOURNISSEURS) + tuple(lignes_supplementaires)


def invite_key(bras: str, question_id: str) -> str:
    return _invite_id(bras, question_id)


def stability_rejeu_id(base_id: str, index: int) -> str:
    return f"{base_id}#rejeu-{index}"


def count_calls_by_role(path: Path, role: str) -> int:
    seen: set[str] = set()
    count = 0
    for record in load_journal_records(path):
        if record.get("role") != role or not record.get("final", True):
            continue
        ident = str(record["id"])
        if ident in seen:
            continue
        seen.add(ident)
        count += 1
    return count


def assert_repondant_budget(path: Path) -> None:
    plafond = plafond_repondant_actif()
    if count_calls_by_role(path, "reponse") >= plafond:
        raise RuntimeError(f"plafond répondant atteint ({plafond})")


def assert_juge_budget(path: Path) -> None:
    plafond = plafond_juge_actif()
    if count_calls_by_role(path, "juge") >= plafond:
        raise RuntimeError(f"plafond juge atteint ({plafond})")


def _model_family(slug: str) -> str:
    lowered = slug.lower()
    if "grok" in lowered:
        return "grok"
    if "sol" in lowered:
        return "sol"
    return lowered.split("-", 1)[0]


def assert_judge_model_allowed(
    repondant_model: str,
    juge_model: str,
    *,
    meme_famille_accepte: bool = False,
) -> None:
    if repondant_model == juge_model:
        raise RuntimeError("le juge ne peut pas être le même modèle que le répondant")
    if meme_famille_accepte:
        return
    if _model_family(repondant_model) == _model_family(juge_model):
        raise RuntimeError(
            "le juge ne peut pas appartenir à la même famille de modèle que le répondant "
            f"({repondant_model} / {juge_model}) — utiliser --juge-meme-famille-accepte pour "
            "documenter une exception"
        )


def assert_judge_model_distinct(repondant_model: str, juge_model: str) -> None:
    assert_judge_model_allowed(repondant_model, juge_model)


def build_rendered_contexts(
    nom: str,
    questions: list[dict[str, Any]],
    *,
    k: int | None,
    budget: int,
    dossier: Dossier,
) -> dict[str, str]:
    bruts = lire_contextes_mesure_bruts(nom, dossier=dossier)
    style = _render_style(nom)
    effective_k = k_effectif_fournisseur(nom, k)
    rendered: dict[str, str] = {}
    for question in questions:
        qid = question["id"]
        elements = bruts.get((qid, k))
        if elements is None:
            elements = bruts.get((qid, None))
        if elements is None:
            elements = bruts.get((qid, effective_k))
        if elements is None:
            raise KeyError(
                f"contextes {nom} incomplets : {qid} k={k} absent — "
                f"rejouer la mesure étage 2, aucune recherche à l'étage 3"
            )
        if nom in ("mem0", "letta_archive") and elements and not elements[0].get("kind"):
            elements = _elements_to_evidence(elements[:effective_k])
        elif nom in K_FIXE:
            elements = list(elements)
        else:
            elements = list(elements[:effective_k])
        text, _kept = rendre_contexte(
            elements,
            budget_chars=budget,
            question_id=qid,
            style=style,
            dossier=dossier,
        )
        rendered[qid] = text
    return rendered


def archiver_etage3_v1(dossier: Dossier | None = None) -> Path:
    current = dossier if dossier is not None else dossier_courant("moreau")
    src = etage3_dir(current)
    dst = etage3_v1_dir(current)
    if dst.exists():
        return dst
    if not src.is_dir():
        raise FileNotFoundError(f"étage 3 absent : {src}")
    src.rename(dst)
    return dst


_LETTRE_VERBATIM_MIN_SIGNIFICANT = 40

_LETTRE_AVOU_IGNORANCE_MARKERS = (
    "ne l'etablissent pas",
    "ne l'etablit pas",
    "ne precise pas",
    "ne precisent pas",
    "sans citer le libelle",
)

REGLE_LETTRE_LIGNES = (
    "1. Passage verbatim d'au moins 40 caractères significatifs de la citation "
    "(normalisation espaces et guillemets ; extrait réponse → sous-chaîne citation).",
    "2. Tous les discriminants de la citation présents dans la réponse "
    "(nombres ancrés, dates, identifiants — `discriminants()`).",
    "3. Pièce d'appui nommée dans la réponse.",
    "4. Pas d'aveu d'ignorance sur le point (« ne l'établit pas », « ne précise pas », "
    "« sans citer le libellé », « par ex. » avec formes candidates).",
)


def _piece_named_in_response(piece: str, response_text: str) -> bool:
    base = piece_basename(piece)
    return bool(base) and base in response_text


def _significant_char_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


def _passage_verbatim_dans_citation(
    response_text: str,
    citation: str,
    *,
    min_significant: int = _LETTRE_VERBATIM_MIN_SIGNIFICANT,
) -> bool:
    norm_cite = normalize_text(citation)
    norm_resp = normalize_text(response_text)
    if not norm_cite or not norm_resp:
        return False
    cite_sig = _significant_char_count(norm_cite)
    required = min(min_significant, cite_sig)
    n = len(norm_cite)
    for i in range(n):
        for j in range(i + 1, n + 1):
            chunk = norm_cite[i:j]
            if _significant_char_count(chunk) < required:
                continue
            if chunk in norm_resp:
                return True
    return False


def _discriminants_dans_reponse(citation: str, response_text: str) -> bool:
    disc = discriminants(citation)
    for number, anchor in disc["nombres"]:
        if anchor is None:
            continue
        if not _anchored_number_present(number, anchor, response_text):
            return False
    for date_key in disc["dates"]:
        if date_key not in _dates_in_text(response_text):
            return False
    for token in disc["textes"]:
        if not _text_token_present(token, response_text):
            return False
    return True


def _par_ex_formes_candidates(response_text: str) -> bool:
    lowered = response_text.lower()
    idx = lowered.find("par ex.")
    if idx < 0:
        idx = lowered.find("par exemple")
    if idx < 0:
        return False
    tail = response_text[idx:]
    if len(re.findall(r"«[^»]+»", tail)) >= 2:
        return True
    return len(re.findall(r'"[^"]{3,}"', tail)) >= 2


def reponse_avoue_ignorance_lettre(response_text: str) -> bool:
    norm = normalize_text(response_text)
    if any(marker in norm for marker in _LETTRE_AVOU_IGNORANCE_MARKERS):
        return True
    return _par_ex_formes_candidates(response_text)


def _assertion_lettre_dans_reponse(
    assertion: dict[str, Any],
    response_text: str,
) -> bool:
    piece = str(assertion.get("source") or assertion.get("piece") or "")
    if not _piece_named_in_response(piece, response_text):
        return False
    citation = str(assertion.get("quote") or assertion.get("content") or "")
    if not _passage_verbatim_dans_citation(response_text, citation):
        return False
    if not _discriminants_dans_reponse(citation, response_text):
        return False
    return True


def _detect_refus_attendu(question: dict[str, Any], response_text: str) -> bool:
    if detect_response_refus(response_text):
        return True
    if str(question.get("shape") or "") == "unknown":
        lowered = normalize_text(response_text)
        return "cannot determine" in lowered or "can not determine" in lowered
    return False


def compute_verdict_valeur_comparatif(
    question: dict[str, Any],
    response_text: str,
) -> str:
    shape = str(question.get("shape") or "")
    if shape == "unknown":
        if is_response_illisible(response_text):
            return "illisible"
        if _detect_refus_attendu(question, response_text):
            return "conforme"
        return "fautive"
    verdict = compute_verdict_adjudique(question, response_text)
    if (
        verdict == "conforme"
        and not _shape_refus_attendu(shape)
        and reponse_avoue_ignorance_lettre(response_text)
    ):
        return "incomplete"
    return verdict


def compute_verdict_lettre(
    question: dict[str, Any],
    response_text: str,
    assertion_index: dict[str, dict[str, Any]],
) -> str:
    verdict = compute_verdict_valeur_comparatif(question, response_text)
    if verdict != "conforme":
        return verdict
    if reponse_avoue_ignorance_lettre(response_text):
        return "incomplete"
    supports = question.get("supporting_assertions") or []
    for aid in supports:
        assertion = assertion_index.get(aid)
        if assertion and _assertion_lettre_dans_reponse(assertion, response_text):
            return "conforme"
    return "incomplete"


def build_all_invites(
    questions: list[dict[str, Any]],
    *,
    dossier: Dossier,
    lignes_supplementaires: Sequence[tuple[str, int | None, int]] = (),
) -> list[dict[str, Any]]:
    invites: list[dict[str, Any]] = []
    lignes = lignes_contexte(lignes_supplementaires)
    contexts_by_bras: dict[str, dict[str, str]] = {}
    for nom, k, budget in lignes:
        contexts_by_bras[nom] = build_rendered_contexts(
            nom,
            questions,
            k=k,
            budget=budget,
            dossier=dossier,
        )
    for question in questions:
        invites.append(
            _decorate_invite(
                build_response_invite(
                    BRAS_A,
                    question,
                    context=CONTEXT_A,
                    langue=dossier.langue,
                ),
                fournisseur_contexte=None,
                k=None,
                budget=None,
            )
        )
    for nom, k, budget in lignes:
        contexts = contexts_by_bras[nom]
        for question in questions:
            invites.append(
                _decorate_invite(
                    build_response_invite(
                        nom,
                        question,
                        context=contexts[question["id"]],
                        langue=dossier.langue,
                    ),
                    fournisseur_contexte=nom,
                    k=k,
                    budget=budget,
                )
            )
    return invites


def _decorate_invite(
    invite: dict[str, Any],
    *,
    fournisseur_contexte: str | None,
    k: int | None,
    budget: int | None,
) -> dict[str, Any]:
    row = dict(invite)
    row["fournisseur_contexte"] = fournisseur_contexte
    row["k"] = k
    row["budget"] = budget
    return row


def stability_questions(
    questions: list[dict[str, Any]],
    *,
    seed: int = SEED,
    k: int = STABILITY_QUESTIONS,
) -> list[str]:
    rng = random.Random(seed + 31)
    pool = [question["id"] for question in questions]
    return sorted(rng.sample(pool, min(k, len(pool))))


def comptage_a_blanc(questions: list[dict[str, Any]] | None = None) -> dict[str, int]:
    questions = questions or load_questions(dossier_courant("moreau"))
    reponses = len(ALL_BRAS) * len(questions)
    reponses += STABILITY_QUESTIONS * len(FOURNISSEURS) * STABILITY_REJEUX
    juges = len(ALL_BRAS) * len(questions)
    return {"repondant": reponses, "juge": juges}


def comptage_lignes(
    questions: list[dict[str, Any]],
    lignes: Sequence[tuple[str, int | None, int]],
    *,
    include_a: bool = False,
) -> dict[str, int]:
    n_bras = len(lignes) + (1 if include_a else 0)
    repondant = n_bras * len(questions)
    repondant += STABILITY_QUESTIONS * len(lignes) * STABILITY_REJEUX
    juge = n_bras * len(questions)
    return {"repondant": repondant, "juge": juge}


def ids_repondant_prevus(
    questions: list[dict[str, Any]],
    lignes: Sequence[tuple[str, int | None, int]],
    *,
    include_a: bool = False,
    include_stability: bool = True,
) -> list[str]:
    ids: list[str] = []
    if include_a:
        ids.extend(invite_key(BRAS_A, question["id"]) for question in questions)
    for nom, _k, _budget in lignes:
        ids.extend(invite_key(nom, question["id"]) for question in questions)
        if include_stability:
            for qid in stability_questions(questions):
                ident = invite_key(nom, qid)
                ids.extend(
                    stability_rejeu_id(ident, index)
                    for index in range(1, STABILITY_REJEUX + 1)
                )
    return ids


def ids_juge_prevus(
    questions: list[dict[str, Any]],
    lignes: Sequence[tuple[str, int | None, int]],
    *,
    include_a: bool = False,
) -> list[str]:
    ids: list[str] = []
    if include_a:
        ids.extend(f"juge-{invite_key(BRAS_A, question['id'])}" for question in questions)
    for nom, _k, _budget in lignes:
        ids.extend(f"juge-{invite_key(nom, question['id'])}" for question in questions)
    return ids


def comptage_restant(
    questions: list[dict[str, Any]],
    lignes: Sequence[tuple[str, int | None, int]],
    calls_path: Path,
    *,
    include_a: bool = False,
) -> dict[str, int]:
    completed, _hashes = load_journal_state(calls_path)
    repondant_ids = ids_repondant_prevus(questions, lignes, include_a=include_a)
    juge_ids = ids_juge_prevus(questions, lignes, include_a=include_a)
    return {
        "repondant": sum(1 for ident in repondant_ids if ident not in completed),
        "juge": sum(1 for ident in juge_ids if ident not in completed),
    }


def write_dry_run_report(
    invites: list[dict[str, Any]],
    *,
    comptage: dict[str, int],
    dossier: Dossier,
    repondant_model: str,
    juge_model: str,
    plafonds: dict[str, int] | None = None,
    path: Path | None = None,
) -> None:
    out = path or rapport_path(dossier)
    plafonds = plafonds or {
        "repondant": plafond_repondant_actif(),
        "juge": plafond_juge_actif(),
    }
    by_bras: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for invite in invites:
        by_bras[str(invite["bras"])].append(invite)
    etage2_col = _etage2_tout_tient_par_bras(dossier)
    lines = [
        "# Rapport — étage 3 comparatif",
        "",
        f"Date : {date.today().isoformat()}",
        f"Dossier : {dossier.nom}",
        "",
        "## Modèles",
        "",
        f"- Répondant : `{repondant_model}`",
        f"- Juge : `{juge_model}`",
        "",
        "## Comptage à blanc (avant tout appel)",
        "",
        f"- Appels répondant prévus : {comptage['repondant']} (plafond {plafonds['repondant']})",
        f"- Appels juge prévus : {comptage['juge']} (plafond {plafonds['juge']})",
        "",
        f"Langue invite : `{dossier.langue}`",
        "",
        "## Invites par bras",
        "",
        "| Bras | k | budget | invites | tokens invite médians (est.) | tout tient (verbatim) étage 2 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    order = [BRAS_A] + [nom for nom, _k, _b in FOURNISSEURS]
    for bras in order:
        rows = by_bras[bras]
        if bras == BRAS_A:
            k_label = "—"
            budget_label = "—"
            etage2 = "—"
        else:
            spec = next(item for item in FOURNISSEURS if item[0] == bras)
            k_label = "figé" if spec[1] is None else str(spec[1])
            budget_label = str(spec[2])
            etage2 = etage2_col.get(bras, "—")
        tokens = sorted(estimate_invite_tokens(int(invite.get("user_chars", 0))) for invite in rows)
        median = tokens[len(tokens) // 2] if tokens else 0
        lines.append(
            f"| {bras} | {k_label} | {budget_label} | {len(rows)} | {median} | {etage2} |"
        )
    lines.extend(
        [
            "",
            f"Stabilité : {STABILITY_QUESTIONS} questions × {STABILITY_REJEUX} rejeux × "
            f"{len(FOURNISSEURS)} fournisseurs = "
            f"{STABILITY_QUESTIONS * STABILITY_REJEUX * len(FOURNISSEURS)} appels répondant supplémentaires.",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def _etage2_tout_tient_par_bras(dossier: Dossier) -> dict[str, str]:
    meta = _meta_mesure(dossier, load_questions(dossier))
    denom = meta["denominateur_appui"]
    out: dict[str, str] = {}
    for nom, k, budget in FOURNISSEURS:
        path = dossier.comparatif_dir / f"etage2-{nom}.json"
        if not path.is_file():
            out[nom] = "—"
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = _find_reference_run(payload, k, budget)
        if run is None:
            out[nom] = "—"
            continue
        block = run["aggregate"]["appui"]
        key = f"tout_tient_sur_{denom}"
        value = block.get(key, block.get("tout_tient_sur_141", 0))
        out[nom] = f"{value}/{denom}"
    return out


def resolve_juge_modele(juge_modele: str | None, *, fournisseur: str) -> str:
    if juge_modele is not None:
        return juge_modele
    if fournisseur == "codex":
        return JUGE_MODEL_CODEX
    return JUGE_MODEL


def run_dry_run(
    dossier: Dossier | None = None,
    *,
    lignes_supplementaires: Sequence[tuple[str, int | None, int]] = (),
    juge_modele: str | None = None,
    juge_fournisseur: str = "cursor",
) -> dict[str, Any]:
    current = dossier if dossier is not None else dossier_courant("moreau")
    plafonds = init_plafonds(current, lignes_supplementaires=lignes_supplementaires)
    juge_slug = resolve_juge_modele(juge_modele, fournisseur=juge_fournisseur)
    assert_cursor_model_available(REPONDANT_MODEL)
    if juge_fournisseur == "cursor":
        assert_cursor_model_available(juge_slug)
    assert_judge_model_distinct(REPONDANT_MODEL, juge_slug)
    questions = load_questions(current)
    if lignes_supplementaires:
        extra_noms = {nom for nom, _k, _b in lignes_supplementaires}
        path = invites_path(current)
        existing = load_invites(path, current) if path.is_file() else build_all_invites(
            questions, dossier=current
        )
        kept = [invite for invite in existing if invite.get("bras") not in extra_noms]
        extra_invites = []
        for nom, k, budget in lignes_supplementaires:
            contexts = build_rendered_contexts(
                nom, questions, k=k, budget=budget, dossier=current
            )
            for question in questions:
                extra_invites.append(
                    _decorate_invite(
                        build_response_invite(
                            nom,
                            question,
                            context=contexts[question["id"]],
                            langue=current.langue,
                        ),
                        fournisseur_contexte=nom,
                        k=k,
                        budget=budget,
                    )
                )
        invites = kept + extra_invites
    else:
        invites = build_all_invites(
            questions,
            dossier=current,
            lignes_supplementaires=(),
        )
    comptage = comptage_a_blanc(questions)
    out_dir = etage3_dir(current)
    out_dir.mkdir(parents=True, exist_ok=True)
    invites_path(current).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in invites),
        encoding="utf-8",
    )
    result: dict[str, Any] = {
        "status": "dry_run",
        "dossier": current.nom,
        "invites": len(invites),
        "comptage_a_blanc": comptage,
        "plafonds": {"repondant": plafonds["repondant"], "juge": plafonds["juge"]},
        "slugs": {"repondant": REPONDANT_MODEL, "juge": juge_slug},
        "files": [str(invites_path(current)), str(rapport_blanc_path(current))],
    }
    if lignes_supplementaires:
        calls = appels_path(current)
        restant_ref = comptage_restant(
            questions, FOURNISSEURS, calls, include_a=True
        )
        restant_extra = comptage_restant(
            questions, lignes_supplementaires, calls, include_a=False
        )
        result["comptage_reference_restant"] = restant_ref
        result["comptage_supplementaires"] = restant_extra
        result["comptage_total_restant"] = {
            "repondant": restant_ref["repondant"] + restant_extra["repondant"],
            "juge": restant_ref["juge"] + restant_extra["juge"],
        }
        result["lignes_supplementaires"] = [
            [nom, k, budget] for nom, k, budget in lignes_supplementaires
        ]
        result["files"] = [str(invites_path(current))]
        return result
    write_dry_run_report(
        invites,
        comptage=comptage,
        dossier=current,
        repondant_model=REPONDANT_MODEL,
        juge_model=juge_slug,
        plafonds={"repondant": plafonds["repondant"], "juge": plafonds["juge"]},
        path=rapport_blanc_path(current),
    )
    return result


def load_invites(path: Path | None = None, dossier: Dossier | None = None) -> list[dict[str, Any]]:
    path = path or invites_path(dossier)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assurer_invites_supplementaires(
    dossier: Dossier,
    lignes_supplementaires: Sequence[tuple[str, int | None, int]],
    *,
    path: Path | None = None,
    questions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    path = path or invites_path(dossier)
    existing = load_invites(path, dossier) if path.is_file() else []
    if not lignes_supplementaires:
        return existing
    have = {str(invite.get("bras")) for invite in existing}
    missing = [ligne for ligne in lignes_supplementaires if ligne[0] not in have]
    if not missing:
        return existing
    questions = questions or load_questions(dossier)
    extra: list[dict[str, Any]] = []
    for nom, k, budget in missing:
        contexts = build_rendered_contexts(
            nom, questions, k=k, budget=budget, dossier=dossier
        )
        for question in questions:
            extra.append(
                _decorate_invite(
                    build_response_invite(
                        nom,
                        question,
                        context=contexts[question["id"]],
                        langue=dossier.langue,
                    ),
                    fournisseur_contexte=nom,
                    k=k,
                    budget=budget,
                )
            )
    combined = existing + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined),
        encoding="utf-8",
    )
    return combined


def adjudicate_response(
    invite: dict[str, Any],
    *,
    chat_fn: ChatFn,
    calls_path: Path,
    record_id: str | None = None,
    modele: str = REPONDANT_MODEL,
) -> dict[str, Any]:
    ident = record_id or invite["id"]
    assert_repondant_budget(calls_path)
    raw, meta = chat_fn(invite["system"], invite["user"])
    append_call(
        calls_path,
        {
            "horodatage": date.today().isoformat(),
            "role": "reponse",
            "id": ident,
            "bras": invite["bras"],
            "question_id": invite["question_id"],
            "fournisseur_contexte": invite.get("fournisseur_contexte"),
            "sha256_invite": invite["sha256"],
            "modele_demande": modele,
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


def adjudicate_judge_call(
    response_text: str,
    question: dict[str, Any],
    *,
    invite_key_value: str,
    chat_fn: ChatFn,
    calls_path: Path,
    modele: str = JUGE_MODEL,
    meme_famille_accepte: bool = False,
) -> dict[str, Any]:
    assert_juge_budget(calls_path)
    assert_judge_model_allowed(
        REPONDANT_MODEL,
        modele,
        meme_famille_accepte=meme_famille_accepte,
    )
    return adjudicate_judge(
        response_text,
        question,
        invite_key=invite_key_value,
        chat_fn=chat_fn,
        calls_path=calls_path,
        modele=modele,
        check_total_budget=False,
    )


def _stability_targets(
    questions: list[dict[str, Any]],
    lignes: Sequence[tuple[str, int | None, int]] | None = None,
) -> dict[str, set[str]]:
    qids = set(stability_questions(questions))
    actives = FOURNISSEURS if lignes is None else lignes
    return {
        nom: {invite_key(nom, qid) for qid in qids}
        for nom, _k, _b in actives
    }


def run_reponses(
    chat_fn: ChatFn,
    *,
    dossier: Dossier | None = None,
    calls_path: Path | None = None,
    modele: str = REPONDANT_MODEL,
    plafond_repondant: int | None = None,
    plafond_juge: int | None = None,
    lignes_supplementaires: Sequence[tuple[str, int | None, int]] = (),
) -> dict[str, Any]:
    current = dossier if dossier is not None else dossier_courant("moreau")
    init_plafonds(
        current,
        plafond_repondant=plafond_repondant,
        plafond_juge=plafond_juge,
        lignes_supplementaires=lignes_supplementaires,
    )
    calls_path = calls_path or appels_path(current)
    if lignes_supplementaires:
        invites = _assurer_invites_supplementaires(
            current, lignes_supplementaires
        )
        extra_noms = {nom for nom, _k, _b in lignes_supplementaires}
        invites = [invite for invite in invites if invite.get("bras") in extra_noms]
        stability = _stability_targets(
            load_questions(current),
            lignes_supplementaires,
        )
    else:
        invites = load_invites(dossier=current)
        stability = _stability_targets(load_questions(current))
    completed, invite_hashes = load_journal_state(calls_path)
    for invite in invites:
        ident = invite["id"]
        if ident in invite_hashes:
            verify_invite_hash(ident, invite_hashes[ident], invite)
        if ident in completed:
            continue
        adjudicate_response(invite, chat_fn=chat_fn, calls_path=calls_path, modele=modele)
        if invite["bras"] in stability and ident in stability[invite["bras"]]:
            for index in range(1, STABILITY_REJEUX + 1):
                rejeu_id = stability_rejeu_id(ident, index)
                if rejeu_id in invite_hashes:
                    verify_invite_hash(rejeu_id, invite_hashes[rejeu_id], invite)
                if rejeu_id in completed:
                    continue
                adjudicate_response(
                    invite,
                    chat_fn=chat_fn,
                    calls_path=calls_path,
                    record_id=rejeu_id,
                    modele=modele,
                )
    return {
        "status": "reponses",
        "appels_repondant": count_calls_by_role(calls_path, "reponse"),
        "appels_total": count_calls(calls_path),
    }


def _latest_responses(calls_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in load_journal_records(calls_path):
        if record.get("role") != "reponse" or not record.get("final", True):
            continue
        ident = str(record["id"])
        if "#rejeu-" in ident:
            continue
        latest[ident] = record
    return latest


def run_juge(
    chat_fn: ChatFn,
    *,
    dossier: Dossier | None = None,
    calls_path: Path | None = None,
    modele: str = JUGE_MODEL,
    meme_famille_accepte: bool = False,
    plafond_repondant: int | None = None,
    plafond_juge: int | None = None,
    lignes_supplementaires: Sequence[tuple[str, int | None, int]] = (),
) -> dict[str, Any]:
    current = dossier if dossier is not None else dossier_courant("moreau")
    init_plafonds(
        current,
        plafond_repondant=plafond_repondant,
        plafond_juge=plafond_juge,
        lignes_supplementaires=lignes_supplementaires,
    )
    assert_judge_model_allowed(
        REPONDANT_MODEL,
        modele,
        meme_famille_accepte=meme_famille_accepte,
    )
    calls_path = calls_path or appels_path(current)
    if lignes_supplementaires:
        invites = _assurer_invites_supplementaires(
            current, lignes_supplementaires
        )
        extra_noms = {nom for nom, _k, _b in lignes_supplementaires}
        invites = [invite for invite in invites if invite.get("bras") in extra_noms]
    else:
        invites = load_invites(dossier=current)
    questions = {q["id"]: q for q in load_questions(current)}
    responses = _latest_responses(calls_path)
    completed, _invite_hashes = load_journal_state(calls_path)
    for invite in invites:
        juge_id = f"juge-{invite['id']}"
        if juge_id in completed:
            continue
        if invite["id"] not in responses:
            continue
        response_text = str(responses[invite["id"]].get("reponse_brute") or "")
        adjudicate_judge_call(
            response_text,
            questions[invite["question_id"]],
            invite_key_value=juge_id,
            chat_fn=chat_fn,
            calls_path=calls_path,
            modele=modele,
        )
    return {
        "status": "juge",
        "appels_juge": count_calls_by_role(calls_path, "juge"),
        "appels_total": count_calls(calls_path),
    }


def compute_verdict_comparatif(
    question: dict[str, Any],
    response_text: str,
    judge_result: dict[str, Any] | None = None,
) -> str:
    return compute_verdict_adjudique(question, response_text)


def format_colonne_juge(juge_count: int, total_jugements: int, fautes: int) -> str:
    if juge_count <= 0:
        return f"en attente (0/{total_jugements} jugements)"
    return f"{fautes} ({juge_count}/{total_jugements} jugés)"


def compute_deux_cotes(
    question: dict[str, Any],
    response_text: str,
    *,
    dossier: Dossier | None = None,
) -> bool:
    current = dossier if dossier is not None else dossier_courant("moreau")
    shape = str(question.get("shape") or "")
    if not _shape_deux_cotes(shape, current):
        return False
    tokens = must_mention_for_verdict(question)
    if len(tokens) < 2:
        return False
    present = sum(1 for token in tokens if must_mention_present(token, response_text))
    return present >= 2


def count_fautes_juge(judge_result: dict[str, Any]) -> int:
    return sum(
        1
        for row in judge_result.get("jugements") or []
        if row.get("verdict") == "affirme" and row.get("preuve_verifiable", True)
    )


def compute_stabilite_bras(
    calls_path: Path,
    bras: str,
    questions: list[dict[str, Any]],
    *,
    qmap: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    qmap = qmap or {q["id"]: q for q in questions}
    qids = stability_questions(questions)
    stable = 0
    total = 0
    by_id: dict[str, list[str]] = defaultdict(list)
    for record in load_journal_records(calls_path):
        if record.get("role") != "reponse" or not record.get("final", True):
            continue
        ident = str(record["id"])
        if not ident.startswith(f"{bras}-") or "#rejeu-" not in ident:
            continue
        base = ident.split("#rejeu-", 1)[0]
        by_id[base].append(str(record.get("reponse_brute") or ""))
    for qid in qids:
        base = invite_key(bras, qid)
        texts = by_id.get(base) or []
        if len(texts) < STABILITY_REJEUX:
            continue
        question = qmap[qid]
        verdicts = [compute_verdict_adjudique(question, text) for text in texts[:STABILITY_REJEUX]]
        total += 1
        if len(set(verdicts)) == 1:
            stable += 1
    return {"stables": stable, "total": total, "ratio": f"{stable}/{STABILITY_QUESTIONS}"}


def _aggregate_bras(
    notes: list[dict[str, Any]],
    *,
    bras: str,
    questions: list[dict[str, Any]],
    calls_path: Path,
    juge_count: int,
    total_jugements: int,
    dossier: Dossier,
) -> dict[str, Any]:
    subset = [note for note in notes if note["bras"] == bras]
    refus_attendu_ids = {
        q["id"] for q in questions if _shape_refus_attendu(str(q.get("shape") or ""))
    }
    deux_cotes_ids = {
        q["id"] for q in questions if _shape_deux_cotes(str(q.get("shape") or ""), dossier)
    }
    conforme_valeur = sum(1 for note in subset if note["verdict_valeur"] == "conforme")
    conforme_lettre = sum(1 for note in subset if note["verdict_lettre"] == "conforme")
    refus_attendu = sum(
        1
        for note in subset
        if note["question_id"] in refus_attendu_ids and note["verdict_valeur"] == "conforme"
    )
    refus_fautifs = sum(1 for note in subset if note["verdict_valeur"] == "refus_fautif")
    fautes_juge = sum(int(note.get("fautes_juge") or 0) for note in subset)
    deux_cotes = sum(
        1
        for note in subset
        if note["question_id"] in deux_cotes_ids and note.get("deux_cotes")
    )
    formes_iso = sum(len(note.get("formes_iso") or []) for note in subset)
    illisibles = sum(1 for note in subset if note["verdict_valeur"] == "illisible")
    par_forme: dict[str, Counter[str]] = defaultdict(Counter)
    for note in subset:
        par_forme[str(note.get("shape") or "unknown")][note["verdict_valeur"]] += 1
    tokens_invite = [
        t for t in (note.get("tokens_invite") for note in subset) if t is not None
    ]
    usage_relais = [
        t for t in (note.get("usage_relais") for note in subset) if t is not None
    ]
    qmap = {q["id"]: q for q in questions}
    stabilite = compute_stabilite_bras(calls_path, bras, questions, qmap=qmap)
    return {
        "conforme_valeur": f"{conforme_valeur}/{len(questions)}",
        "conforme_lettre": f"{conforme_lettre}/{len(questions)}",
        "conformes": f"{conforme_valeur}/{len(questions)}",
        "par_forme": {shape: dict(counter) for shape, counter in sorted(par_forme.items())},
        "refus_non_etabli": f"{refus_attendu}/{len(refus_attendu_ids)}",
        "refus_unknown": f"{refus_attendu}/{len(refus_attendu_ids)}",
        "refus_fautifs": refus_fautifs,
        "fautes_juge": format_colonne_juge(juge_count, total_jugements, fautes_juge),
        "fautes_juge_n": fautes_juge,
        "deux_cotes": f"{deux_cotes}/{len(deux_cotes_ids)}",
        "stabilite": stabilite["ratio"],
        "tokens_invite_medians": int(statistics.median(tokens_invite)) if tokens_invite else 0,
        "usage_relais_medians": int(statistics.median(usage_relais)) if usage_relais else 0,
        "tokens_medians": int(statistics.median(tokens_invite)) if tokens_invite else 0,
        "formes_iso": formes_iso,
        "illisibles": illisibles,
    }


def run_noter(
    *,
    dossier: Dossier | None = None,
    calls_path: Path | None = None,
    invites_path_arg: Path | None = None,
    plafond_repondant: int | None = None,
    plafond_juge: int | None = None,
    lignes_supplementaires: Sequence[tuple[str, int | None, int]] = (),
) -> dict[str, Any]:
    current = dossier if dossier is not None else dossier_courant("moreau")
    init_plafonds(
        current,
        plafond_repondant=plafond_repondant,
        plafond_juge=plafond_juge,
        lignes_supplementaires=lignes_supplementaires,
    )
    calls_path = calls_path or appels_path(current)
    if lignes_supplementaires:
        invites = _assurer_invites_supplementaires(
            current,
            lignes_supplementaires,
            path=invites_path_arg,
        )
        extra_noms = {nom for nom, _k, _b in lignes_supplementaires}
        invites = [invite for invite in invites if invite.get("bras") in extra_noms]
    else:
        invites = load_invites(invites_path_arg, dossier=current)
        ref_bras = {BRAS_A} | {nom for nom, _k, _b in FOURNISSEURS}
        invites = [invite for invite in invites if invite.get("bras") in ref_bras]
        invites = list({invite["id"]: invite for invite in invites}.values())
    questions = load_questions(current)
    qmap = {q["id"]: q for q in questions}
    assertion_index = load_gold(current)["assertion_index"]
    responses = _latest_responses(calls_path)
    juges: dict[str, dict[str, Any]] = {}
    for record in load_journal_records(calls_path):
        if record.get("role") == "juge" and record.get("final", True):
            juges[str(record["id"])] = record
    invite_ids = {invite["id"] for invite in invites}
    juges = {
        ident: record
        for ident, record in juges.items()
        if ident.removeprefix("juge-") in invite_ids
    }
    juge_count = len(juges)
    total_jugements = len(invites)
    notes: list[dict[str, Any]] = []
    manquants: list[str] = []
    illisibles: list[str] = []
    for invite in invites:
        ident = invite["id"]
        if ident not in responses:
            manquants.append(ident)
            continue
        question = qmap[invite["question_id"]]
        response_text = str(responses[ident].get("reponse_brute") or "")
        verdict_valeur = compute_verdict_valeur_comparatif(question, response_text)
        verdict_lettre = compute_verdict_lettre(question, response_text, assertion_index)
        scoring = must_mention_scoring_fields_adjudique(question, response_text)
        judge = juges.get(f"juge-{ident}")
        usage = responses[ident].get("usage") or {}
        iso_signals = detect_formes_iso(question, response_text)
        note = {
            "id": ident,
            "bras": invite["bras"],
            "question_id": invite["question_id"],
            "shape": question.get("shape"),
            "verdict": verdict_valeur,
            "verdict_valeur": verdict_valeur,
            "verdict_lettre": verdict_lettre,
            "must_mention_manquants": scoring["must_mention_manquants"],
            "refus": detect_response_refus(response_text),
            "fautes_juge": count_fautes_juge(judge) if judge else None,
            "deux_cotes": compute_deux_cotes(question, response_text, dossier=current),
            "formes_iso": iso_signals,
            "tokens_invite": estimate_invite_tokens(int(invite.get("user_chars", 0))),
            "usage_relais": usage_relais_total(usage),
            "tokens_entree": estimate_invite_tokens(int(invite.get("user_chars", 0))),
        }
        notes.append(note)
        if verdict_valeur == "illisible":
            illisibles.append(ident)
    bras_ordre: list[str]
    if lignes_supplementaires:
        bras_ordre = [nom for nom, _k, _b in lignes_supplementaires]
    else:
        bras_ordre = [BRAS_A] + [nom for nom, _k, _b in FOURNISSEURS]
    by_bras: dict[str, Any] = {}
    for bras in bras_ordre:
        payload = {
            "date": date.today().isoformat(),
            "bras": bras,
            "notes": [note for note in notes if note["bras"] == bras],
            "agregat": _aggregate_bras(
                notes,
                bras=bras,
                questions=questions,
                calls_path=calls_path,
                juge_count=juge_count,
                total_jugements=total_jugements,
                dossier=current,
            ),
        }
        by_bras[bras] = payload
        notations_path(bras, current).parent.mkdir(parents=True, exist_ok=True)
        notations_path(bras, current).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if lignes_supplementaires:
        write_situations_report(
            by_bras,
            dossier=current,
            questions=questions,
            extra_noms=tuple(nom for nom, _k, _b in lignes_supplementaires),
            juge_count=juge_count,
            total_jugements=total_jugements,
            langue_invite=current.langue,
        )
    else:
        write_final_report(
            by_bras,
            dossier=current,
            juge_count=juge_count,
            total_jugements=total_jugements,
            langue_invite=current.langue,
            v1_archivee=etage3_v1_dir(current).is_dir(),
        )
    return {
        "status": "noter",
        "notes": len(notes),
        "manquants": sorted(manquants),
        "illisibles": sorted(illisibles),
        "complet": not manquants,
        "juge": format_colonne_juge(juge_count, total_jugements, sum(int(n.get("fautes_juge") or 0) for n in notes)),
        "agregats": {bras: payload["agregat"] for bras, payload in by_bras.items()},
    }


def write_final_report(
    by_bras: dict[str, Any],
    *,
    dossier: Dossier,
    juge_count: int,
    total_jugements: int,
    langue_invite: str = "fr",
    v1_archivee: bool = False,
) -> None:
    etage2_col = _etage2_tout_tient_par_bras(dossier)
    total_fautes = sum(
        int(note.get("fautes_juge") or 0)
        for payload in by_bras.values()
        for note in payload.get("notes", [])
    )
    refus_col = "refus unknown" if dossier.nom == "greenloop" else "refus non établi"
    refus_key = "refus_unknown" if dossier.nom == "greenloop" else "refus_non_etabli"
    lines = [
        "# Rapport — étage 3 comparatif",
        "",
        f"Date : {date.today().isoformat()}",
        f"Dossier : {dossier.nom}",
        f"Langue invite : {langue_invite}",
        "",
        f"Juge must_not_claim : {format_colonne_juge(juge_count, total_jugements, total_fautes)}",
        "",
    ]
    if dossier.nom == "greenloop":
        ref = REFERENCE_GREENLOOP_ETAGE3_20260923
        lines.extend(
            [
                "## Colonne principale",
                "",
                "Dossier business GreenLoop : la colonne principale est `conforme_valeur` "
                "(`conformes`). `conforme_lettre` est calculée avec la même règle à quatre "
                "conditions mais n'est pas la colonne business du dossier.",
                "",
                "## Référence du 23/09 (hors bras homogène)",
                "",
                f"Ligne nommée : {ref['label']}.",
                f"Juge de référence : {ref['juge']}.",
                f"Conformes : {ref['conformes']}. Disputes des deux côtés : {ref['deux_cotes']}.",
                "",
            ]
        )
    lines.extend(
        [
        "## Règle conforme_lettre",
        "",
        *REGLE_LETTRE_LIGNES,
        "",
        f"| Bras | conforme_valeur | conforme_lettre | {refus_col} | refus fautifs | fautes juge | deux côtés | stabilité | tokens invite médians | usage relais médian | formes ISO | illisibles | tout tient (verbatim) étage 2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    order = [BRAS_A] + [nom for nom, _k, _b in FOURNISSEURS]
    for bras in order:
        ag = by_bras[bras]["agregat"]
        etage2 = etage2_col.get(bras, "—") if bras != BRAS_A else "—"
        lines.append(
            f"| {bras} | {ag['conforme_valeur']} | {ag['conforme_lettre']} | {ag[refus_key]} | "
            f"{ag['refus_fautifs']} | {ag['fautes_juge']} | {ag['deux_cotes']} | {ag['stabilite']} | "
            f"{ag['tokens_invite_medians']} | {ag['usage_relais_medians']} | {ag['formes_iso']} | "
            f"{ag['illisibles']} | {etage2} |"
        )
    if v1_archivee:
        lines.extend(
            [
                "",
                "Passe v1 anglaise (valeur seule) archivée dans `etage3-v1-anglais/` "
                "(A 7, brain_prod 11, rag_naif 15, brain_banc 12, mem0 21 conformes /50).",
            ]
        )
    lines.append("")
    lines.append("## Par forme")
    lines.append("")
    for bras in order:
        ag = by_bras[bras]["agregat"]
        lines.append(f"### {bras}")
        lines.append("")
        for shape, counts in sorted(ag["par_forme"].items()):
            parts = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            lines.append(f"- {shape} : {parts}")
        lines.append("")
    if juge_count <= 0:
        lines.extend(
            [
                "## Juge",
                "",
                "Notation adjugée depuis les réponses seules. "
                f"Colonne fautes juge : {format_colonne_juge(0, total_jugements, 0)}.",
                "",
            ]
        )
    rapport_path(dossier).write_text("\n".join(lines), encoding="utf-8")


def _charger_notations_existantes(
    dossier: Dossier,
    bras: str,
) -> dict[str, Any] | None:
    path = notations_path(bras, dossier)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def write_situations_report(
    extra_by_bras: dict[str, Any],
    *,
    dossier: Dossier,
    questions: list[dict[str, Any]],
    extra_noms: Sequence[str],
    juge_count: int,
    total_jugements: int,
    langue_invite: str = "fr",
) -> None:
    by_bras: dict[str, Any] = {}
    order = [BRAS_A] + [nom for nom, _k, _b in FOURNISSEURS] + list(extra_noms)
    for bras in order:
        if bras in extra_by_bras:
            by_bras[bras] = extra_by_bras[bras]
            continue
        existing = _charger_notations_existantes(dossier, bras)
        if existing is not None:
            by_bras[bras] = existing
    etage2_col = _etage2_tout_tient_par_bras(dossier)
    total_fautes = sum(
        int(note.get("fautes_juge") or 0)
        for payload in by_bras.values()
        for note in payload.get("notes", [])
    )
    refus_col = "refus unknown" if dossier.nom == "greenloop" else "refus non établi"
    refus_key = "refus_unknown" if dossier.nom == "greenloop" else "refus_non_etabli"
    lines = [
        "# Rapport — étage 3 situations",
        "",
        f"Date : {date.today().isoformat()}",
        f"Dossier : {dossier.nom}",
        f"Langue invite : {langue_invite}",
        "",
        f"Juge must_not_claim : {format_colonne_juge(juge_count, total_jugements, total_fautes)}",
        "",
        "## Règle conforme_lettre",
        "",
        *REGLE_LETTRE_LIGNES,
        "",
        f"| Bras | conforme_valeur | conforme_lettre | {refus_col} | refus fautifs | fautes juge | deux côtés | stabilité | tokens invite médians | usage relais médian | formes ISO | illisibles | tout tient (verbatim) étage 2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for bras in order:
        if bras not in by_bras:
            continue
        ag = by_bras[bras]["agregat"]
        etage2 = etage2_col.get(bras, "—") if bras != BRAS_A else "—"
        lines.append(
            f"| {bras} | {ag.get('conforme_valeur', '—')} | {ag.get('conforme_lettre', '—')} | "
            f"{ag.get(refus_key, ag.get('refus_non_etabli', '—'))} | "
            f"{ag.get('refus_fautifs', '—')} | {ag.get('fautes_juge', '—')} | "
            f"{ag.get('deux_cotes', '—')} | {ag.get('stabilite', '—')} | "
            f"{ag.get('tokens_invite_medians', ag.get('tokens_medians', '—'))} | "
            f"{ag.get('usage_relais_medians', '—')} | {ag.get('formes_iso', '—')} | "
            f"{ag.get('illisibles', '—')} | {etage2} |"
        )
    lines.append("")
    lines.append("## Par forme")
    lines.append("")
    for bras in order:
        if bras not in by_bras:
            continue
        ag = by_bras[bras]["agregat"]
        lines.append(f"### {bras}")
        lines.append("")
        for shape, counts in sorted((ag.get("par_forme") or {}).items()):
            parts = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            lines.append(f"- {shape} : {parts}")
        lines.append("")
    cibles = [
        question
        for question in questions
        if str(question.get("shape") or "") in {"contradiction", "non_etabli"}
    ]
    lines.extend(
        [
            "## Contradiction et non établi",
            "",
            "Pour chaque question de ces formes, le verdict de chaque ligne.",
            "",
        ]
    )
    notes_par_bras: dict[str, dict[str, dict[str, Any]]] = {}
    for bras, payload in by_bras.items():
        notes_par_bras[bras] = {
            str(note.get("question_id")): note for note in payload.get("notes") or []
        }
    for question in cibles:
        qid = question["id"]
        shape = question.get("shape")
        lines.append(f"### {qid} ({shape})")
        lines.append("")
        lines.append("| Ligne | verdict |")
        lines.append("| --- | --- |")
        for bras in order:
            note = notes_par_bras.get(bras, {}).get(qid)
            verdict = (note or {}).get("verdict_valeur") or (note or {}).get("verdict") or "—"
            lines.append(f"| {bras} | {verdict} |")
        lines.append("")
    rapport_situations_path(dossier).write_text("\n".join(lines), encoding="utf-8")


def write_grok_result(
    *,
    agregats: dict[str, Any],
    juge: str,
    tests: int,
    status: str = "complete",
    comptage: dict[str, int] | None = None,
    appels: dict[str, int] | None = None,
    juge_model: str = JUGE_MODEL,
    langue_invite: str = "fr",
    appels_repondant: int | None = None,
    v1_anglais_archivee: bool = False,
    dossier_nom: str = "moreau",
    main_repare: bool = False,
) -> Path:
    dossier_key = "greenloop" if dossier_nom == "greenloop" else "moreau"
    bras_payload = {
        bras: {
            "conformes": agregats[bras]["conformes"],
            "conforme_lettre": agregats[bras]["conforme_lettre"],
            "refus_unknown": agregats[bras]["refus_unknown"],
            "refus_fautifs": agregats[bras]["refus_fautifs"],
            "fautes_juge": agregats[bras]["fautes_juge_n"],
            "deux_cotes": agregats[bras]["deux_cotes"],
            "stabilite": agregats[bras]["stabilite"],
            "tokens_invite_med": agregats[bras]["tokens_invite_medians"],
        }
        for bras in agregats
    }
    payload: dict[str, Any] = {
        "status": status,
        "main_repare": main_repare,
        "langue_invite": langue_invite,
        dossier_key: bras_payload,
        "v1_anglais_archivee": v1_anglais_archivee,
        "tests": f"{tests} verts",
        "juge": juge,
    }
    if appels_repondant is not None:
        payload["appels_repondant"] = appels_repondant
    if comptage is not None:
        payload["comptage"] = comptage
    if appels is not None:
        payload["appels"] = appels
        payload["slugs"] = {"repondant": REPONDANT_MODEL, "juge": juge_model}
    GROK_RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return GROK_RESULT_PATH


def instructions_identical(invites: list[dict[str, Any]], *, langue: str = "en") -> bool:
    expected = system_prompt_for_langue(langue)
    return all(invite["system"] == expected for invite in invites)


def make_repondant_chat_fn() -> ChatFn:
    assert_cursor_model_available(REPONDANT_MODEL)
    return make_cursor_chat_fn(model=REPONDANT_MODEL)


def make_juge_chat_fn(
    juge_model: str = JUGE_MODEL,
    *,
    fournisseur: str = "cursor",
    meme_famille_accepte: bool = False,
) -> ChatFn:
    assert_judge_model_allowed(
        REPONDANT_MODEL,
        juge_model,
        meme_famille_accepte=meme_famille_accepte,
    )
    if fournisseur == "cursor":
        assert_cursor_model_available(juge_model)
        return make_cursor_chat_fn(model=juge_model)
    if fournisseur == "codex":
        assert_codex_model_slug_allowed(juge_model)
        return make_codex_chat_fn(model=juge_model)
    raise ValueError(f"fournisseur juge inconnu: {fournisseur}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Étage 3 comparatif — fournisseurs de contexte.")
    parser.add_argument("--dry-run", action="store_true", help="Invites et comptage sans appel modèle.")
    parser.add_argument("--reponses", action="store_true", help="Collecter les réponses (répondant Grok 4.5).")
    parser.add_argument("--juge", action="store_true", help="Juger must_not_claim (GPT-5.6 Sol).")
    parser.add_argument(
        "--juge-fournisseur",
        default="cursor",
        choices=("cursor", "codex"),
        help="Fournisseur du juge must_not_claim (défaut: cursor).",
    )
    parser.add_argument(
        "--juge-modele",
        default=None,
        help=f"Slug du juge must_not_claim (défaut: {JUGE_MODEL} ou {JUGE_MODEL_CODEX} selon le fournisseur).",
    )
    parser.add_argument(
        "--juge-meme-famille-accepte",
        action="store_true",
        help="Autorise un juge de la même famille que le répondant (exception documentée).",
    )
    parser.add_argument("--noter", action="store_true", help="Calculer les verdicts depuis le journal.")
    parser.add_argument(
        "--lignes-supplementaires",
        default="",
        help="Lignes hors référence, nom:budget[,nom:budget…] "
        "(ex. brain_situations:24000,brain_situations_g12:24000).",
    )
    parser.add_argument("--dossier", default="moreau", help="Dossier mesuré (défaut: moreau).")
    parser.add_argument(
        "--plafond-repondant",
        type=int,
        default=None,
        help="Plafond d'IDs répondant uniques (défaut : comptage à blanc + 3 %%).",
    )
    parser.add_argument(
        "--plafond-juge",
        type=int,
        default=None,
        help="Plafond d'IDs juge uniques (défaut : comptage à blanc + 3 %%).",
    )
    args = parser.parse_args(argv)
    dossier = dossier_courant(args.dossier)
    extra = parse_lignes_supplementaires(args.lignes_supplementaires)
    juge_modele = resolve_juge_modele(args.juge_modele, fournisseur=args.juge_fournisseur)
    plafond_kwargs = {
        "plafond_repondant": args.plafond_repondant,
        "plafond_juge": args.plafond_juge,
        "lignes_supplementaires": extra,
    }
    if args.dry_run:
        result = run_dry_run(
            dossier,
            lignes_supplementaires=extra,
            juge_modele=args.juge_modele,
            juge_fournisseur=args.juge_fournisseur,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.reponses:
        result = run_reponses(make_repondant_chat_fn(), dossier=dossier, **plafond_kwargs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.juge:
        try:
            result = run_juge(
                make_juge_chat_fn(
                    juge_modele,
                    fournisseur=args.juge_fournisseur,
                    meme_famille_accepte=args.juge_meme_famille_accepte,
                ),
                dossier=dossier,
                modele=juge_modele,
                meme_famille_accepte=args.juge_meme_famille_accepte,
                **plafond_kwargs,
            )
        except CodexQuotaAtteint as exc:
            print(json.dumps({"status": "quota_codex", "erreur": str(exc)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.noter:
        result = run_noter(dossier=dossier, **plafond_kwargs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
