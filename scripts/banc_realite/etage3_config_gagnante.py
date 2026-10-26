# SPDX-License-Identifier: AGPL-3.0-only
"""Étage 3 — lanceur paramétrable (fine k=12 ou hybride + fenêtres, 141 questions)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import urllib.error

import psycopg.errors
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from banc_realite.compat_moteur import _render_markdown, _render_markdown_with_budget

from . import etage3 as e3
from .config import BANC_ROOT, export_runtime_env, write_banc_config_files
from .etage3_couches import (
    _load_cadres_c_reference,
    gabarits_equivalents,
)
from .fidelite import load_gold
from .holders import build_holder_table
from .measure_disputes import (
    _read_public_claims,
    _side_presence,
    build_evidence_for_mode,
)
from .measure_etage2 import _load_embed_cache
from .mesure_porte_phrase import (
    PHRASES_CACHE_PATH,
    build_evidence_for_porte_phrase_config,
)
from .partage_disputes import partager
from .relay import MODEL, RELAY_URL

CONFIG = "fine_tours_k12"
COUCHE = "fine"
MODE = "trois_tours"
TOP_K = 12
# Contrôle à budget égal (controle_budget_egal.py) : k=12, trois tours, sans complétion.
MAX_RENDER_CHARS = 8000
MAX_CALLS = 700
STABILITY_PAIRS = e3.STABILITY_PAIRS
HYBRIDE_MARKER_DIR = Path.home() / ".agents" / "var" / "banc-realite"

PORTE_PHRASE_HYBRIDE_FENETRES_G12: dict[str, Any] = {
    "etiquette": "hyb_vec_f2_b12_g12",
    "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
    "brute_k": 12,
    "phrases": {
        "n_par_passage": 2,
        "max_phrases": 12,
        "mode": "vectoriel",
        "selection": "globale",
        "fenetre": 2,
    },
}

PORTE_PHRASE_HYBRIDE_FENETRES_G24: dict[str, Any] = {
    "etiquette": "hyb_vec_f2_b24_g24",
    "hybride": {"k": 36, "n_tours": 2, "k_tour1": 24},
    "brute_k": 24,
    "phrases": {
        "n_par_passage": 2,
        "max_phrases": 24,
        "mode": "vectoriel",
        "selection": "globale",
        "fenetre": 2,
    },
}

PORTE_PHRASE_HYBRIDE_FENETRES_G36: dict[str, Any] = {
    "etiquette": "hyb_vec_f2_b36_g36",
    "hybride": {"k": 54, "n_tours": 2, "k_tour1": 36},
    "brute_k": 36,
    "phrases": {
        "n_par_passage": 2,
        "max_phrases": 36,
        "mode": "vectoriel",
        "selection": "globale",
        "fenetre": 2,
    },
}

PORTE_PHRASE_HYBRIDE_FENETRES_G48: dict[str, Any] = {
    "etiquette": "hyb_vec_f2_b48_g48",
    "hybride": {"k": 72, "n_tours": 2, "k_tour1": 48},
    "brute_k": 48,
    "phrases": {
        "n_par_passage": 2,
        "max_phrases": 48,
        "mode": "vectoriel",
        "selection": "globale",
        "fenetre": 2,
    },
}

PORTE_PHRASE_HYBRIDE_FENETRES_CALIBREES: dict[str, Any] = {
    "etiquette": "hyb_vec_cal_b12_g12",
    "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
    "brute_k": 12,
    "phrases": {
        "n_par_passage": 2,
        "max_phrases": 12,
        "mode": "vectoriel",
        "selection": "globale",
        "fenetre": 2,
    },
}

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "fine_tours_k12": {
        "sorties_slug": "etage3-config-gagnante",
        "arret_basename": "_etage3-config-arret",
        "result_basename": "_etage3-config-result",
        "marker_dir": BANC_ROOT,
        "rapport_name": "RAPPORT-CONFIG-GAGNANTE.md",
        "kind": "fine_tours",
        "couche": "fine",
        "mode": "trois_tours",
        "top_k": 12,
        "deux_cotes_attendu": 22,
        "verifier_gabarit": True,
    },
    "hybride_fenetres_g12": {
        "sorties_slug": "etage3-hybride-fenetres",
        "arret_basename": "_etage3-hybride-arret",
        "result_basename": "_etage3-hybride-result",
        "marker_dir": HYBRIDE_MARKER_DIR,
        "rapport_name": "RAPPORT-HYBRIDE-FENETRES.md",
        "kind": "porte_phrase",
        "porte_phrase_config": PORTE_PHRASE_HYBRIDE_FENETRES_G12,
        "deux_cotes_attendu": 24,
        "verifier_gabarit": False,
        "reference_notations": BANC_ROOT / "sorties" / "etage3-config-gagnante" / "notations.json",
    },
    "hybride_fenetres_g24": {
        "sorties_slug": "etage3-hybride-fenetres-g24",
        "arret_basename": "_etage3-hybride-g24-arret",
        "result_basename": "_etage3-hybride-g24-result",
        "marker_dir": HYBRIDE_MARKER_DIR,
        "rapport_name": "RAPPORT-HYBRIDE-FENETRES-G24.md",
        "kind": "porte_phrase",
        "porte_phrase_config": PORTE_PHRASE_HYBRIDE_FENETRES_G24,
        "deux_cotes_attendu": 48,
        "verifier_gabarit": False,
        "reference_notations": BANC_ROOT / "sorties" / "etage3-config-gagnante" / "notations.json",
    },
    "hybride_fenetres_g36": {
        "sorties_slug": "etage3-hybride-fenetres-g36",
        "arret_basename": "_etage3-hybride-g36-arret",
        "result_basename": "_etage3-hybride-g36-result",
        "marker_dir": HYBRIDE_MARKER_DIR,
        "rapport_name": "RAPPORT-HYBRIDE-FENETRES-G36.md",
        "kind": "porte_phrase",
        "porte_phrase_config": PORTE_PHRASE_HYBRIDE_FENETRES_G36,
        "deux_cotes_attendu": 72,
        "verifier_gabarit": False,
        "reference_notations": BANC_ROOT / "sorties" / "etage3-config-gagnante" / "notations.json",
    },
    "hybride_fenetres_g48": {
        "sorties_slug": "etage3-hybride-fenetres-g48",
        "arret_basename": "_etage3-hybride-g48-arret",
        "result_basename": "_etage3-hybride-g48-result",
        "marker_dir": HYBRIDE_MARKER_DIR,
        "rapport_name": "RAPPORT-HYBRIDE-FENETRES-G48.md",
        "kind": "porte_phrase",
        "porte_phrase_config": PORTE_PHRASE_HYBRIDE_FENETRES_G48,
        "deux_cotes_attendu": 96,
        "verifier_gabarit": False,
        "reference_notations": BANC_ROOT / "sorties" / "etage3-config-gagnante" / "notations.json",
    },
    "hybride_fenetres_calibrees": {
        "sorties_slug": "etage3-hybride-fenetres-calibrees",
        "arret_basename": "_etage3-hybride-cal-arret",
        "result_basename": "_etage3-hybride-cal-result",
        "marker_dir": HYBRIDE_MARKER_DIR,
        "rapport_name": "RAPPORT-HYBRIDE-FENETRES-CALIBREES.md",
        "kind": "porte_phrase",
        "porte_phrase_config": PORTE_PHRASE_HYBRIDE_FENETRES_CALIBREES,
        "deux_cotes_attendu": 24,
        "verifier_gabarit": False,
        "reference_notations": BANC_ROOT / "sorties" / "etage3-config-gagnante" / "notations.json",
    },
}

_active_config_name = CONFIG


def config_paths(config_name: str) -> dict[str, Path]:
    spec = CONFIGURATIONS[config_name]
    sorties = BANC_ROOT / "sorties" / spec["sorties_slug"]
    marker_dir = Path(spec["marker_dir"])
    return {
        "sorties_config": sorties,
        "cadres": sorties / "cadres.json",
        "invites": sorties / "invites-reponses.jsonl",
        "appels": sorties / "appels.jsonl",
        "notations": sorties / "notations.json",
        "rapport": sorties / spec["rapport_name"],
        "noter_rapport": sorties / "RAPPORT-NOTATIONS.md",
        "arret": marker_dir / f"{spec['arret_basename']}.json",
        "result": marker_dir / f"{spec['result_basename']}.json",
    }


def select_config(config_name: str) -> None:
    global _active_config_name
    if config_name not in CONFIGURATIONS:
        raise ValueError(f"configuration inconnue: {config_name}")
    _active_config_name = config_name


def active_config_name() -> str:
    return _active_config_name


def active_paths() -> dict[str, Path]:
    return config_paths(_active_config_name)


_fine_paths = config_paths("fine_tours_k12")
SORTIES_CONFIG = _fine_paths["sorties_config"]
CADRES_PATH = _fine_paths["cadres"]
INVITES_PATH = _fine_paths["invites"]
APPELS_PATH = _fine_paths["appels"]
NOTATIONS_PATH = _fine_paths["notations"]
RAPPORT_PATH = _fine_paths["rapport"]
ARRET_PATH = _fine_paths["arret"]
RESULT_PATH = _fine_paths["result"]


def write_arret(step: str, why: str, *, config_name: str | None = None) -> None:
    paths = config_paths(config_name or _active_config_name)
    paths["sorties_config"].mkdir(parents=True, exist_ok=True)
    paths["arret"].parent.mkdir(parents=True, exist_ok=True)
    paths["arret"].write_text(
        json.dumps({"status": "arret", "a_quelle_etape": step, "pourquoi": why}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _load_phrases_cache() -> dict[str, list[float]]:
    if not PHRASES_CACHE_PATH.is_file():
        raise RuntimeError(f"cache unités absent : {PHRASES_CACHE_PATH}")
    payload = json.loads(PHRASES_CACHE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"cache unités vide : {PHRASES_CACHE_PATH}")
    return payload


def _build_evidence_for_active_config(
    connection,
    question: dict[str, Any],
    *,
    embed_cache: dict[str, list[float]],
    phrases_cache: dict[str, list[float]] | None = None,
    config_name: str | None = None,
) -> list[dict[str, Any]]:
    spec = CONFIGURATIONS[config_name or _active_config_name]
    if spec["kind"] == "fine_tours":
        evidence, _meta = build_evidence_for_mode(
            connection,
            question,
            couche=spec["couche"],
            k=spec["top_k"],
            mode=spec["mode"],
            embed_cache=embed_cache,
        )
        return evidence
    if spec["kind"] == "porte_phrase":
        if phrases_cache is None:
            phrases_cache = _load_phrases_cache()
        evidence, _brute, _stats = build_evidence_for_porte_phrase_config(
            connection,
            question,
            spec["porte_phrase_config"],
            embed_cache=embed_cache,
            phrases_cache=phrases_cache,
        )
        return evidence
    raise RuntimeError(f"kind inconnu : {spec['kind']}")


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return round(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def _count_above(values: list[int], threshold: int) -> int:
    return sum(1 for value in values if value > threshold)


def aggregate_distribution(cadres_slim: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = [int(row["evidence_count"]) for row in cadres_slim.values()]
    chars = [int(row["caracteres"]) for row in cadres_slim.values()]
    return {
        "elements": {
            "mediane": int(statistics.median(counts)) if counts else 0,
            "p90": _percentile(counts, 90),
            "max": max(counts) if counts else 0,
            "au_dessus_8": _count_above(counts, 8),
            "au_dessus_12": _count_above(counts, 12),
            "au_dessus_16": _count_above(counts, 16),
        },
        "caracteres": {
            "mediane": int(statistics.median(chars)) if chars else 0,
            "p90": _percentile(chars, 90),
            "max": max(chars) if chars else 0,
            "plafond_8000": sum(1 for value in chars if value >= MAX_RENDER_CHARS),
        },
    }




def gabarit_socle_equivalent(reference: str, candidate: str) -> bool:
    """Même socle CallFrame que le bras C ; la liste d'éléments peut s'allonger (complétion)."""
    return gabarits_equivalents(reference, candidate)


def choisir_chat_fn(
    fournisseur: str,
    modele: str,
    *,
    fabrique_relais: Callable[[], e3.ChatFn],
    fabrique_cursor: Callable[[str], e3.ChatFn],
) -> e3.ChatFn:
    if fournisseur == "relais":
        return fabrique_relais()
    if fournisseur == "cursor":
        return fabrique_cursor(modele)
    raise ValueError(f"fournisseur inconnu: {fournisseur}")


def assert_relay_responds_200() -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    request = Request(
        RELAY_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"relais indisponible (HTTP {response.status}) — démarrage refusé")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"relais indisponible (HTTP {exc.code}) — démarrage refusé") from exc
    except OSError as exc:
        raise RuntimeError(f"relais indisponible ({exc}) — démarrage refusé") from exc


def _conversation_state() -> dict[str, Any]:
    return {"responsibility_context": "GreenLoop"}


def _compile_cadre_for_question(
    question: dict[str, Any],
    *,
    connection,
    embed_cache: dict[str, list[float]],
    gold: dict[str, Any],
    claims_index: dict[str, dict[str, Any]],
    phrases_cache: dict[str, list[float]] | None = None,
    config_name: str | None = None,
) -> dict[str, Any]:
    config_name = config_name or _active_config_name
    spec = CONFIGURATIONS[config_name]
    qid = question["id"]
    question_text = question["question"]
    if question_text not in embed_cache:
        raise RuntimeError(f"cache froid : embedding absent ({question_text[:60]}…)")
    evidence = _build_evidence_for_active_config(
        connection,
        question,
        embed_cache=embed_cache,
        phrases_cache=phrases_cache,
        config_name=config_name,
    )
    rendered, elements_ecartes = _render_markdown_with_budget(
        "banc-realite",
        "GreenLoop",
        f"banc-etage3-{qid}",
        None,
        _conversation_state(),
        evidence,
        None,
    )
    row: dict[str, Any] = {
        "config": config_name,
        "caracteres": len(rendered),
        "genres": dict(Counter(item.get("kind") or "unknown" for item in evidence)),
        "rendered": rendered,
        "evidence_count": len(evidence),
        "elements_ecartes": elements_ecartes,
        "passages_tronques": rendered.count(" … [tronqué :"),
    }
    if question.get("shape") == "dispute":
        sides = _side_presence(question, evidence, gold=gold, claims_index=claims_index)
        row["deux_cotes_dans_contexte"] = all(s["dans_contexte"] for s in sides) if sides else False
    return row


def compile_cadres(connection, *, config_name: str | None = None) -> dict[str, dict[str, Any]]:
    config_name = config_name or _active_config_name
    spec = CONFIGURATIONS[config_name]
    paths = config_paths(config_name)
    embed_cache = _load_embed_cache()
    if not embed_cache:
        raise RuntimeError("cache froid : etage2-question-embeddings.json absent ou vide")
    phrases_cache: dict[str, list[float]] | None = None
    if spec["kind"] == "porte_phrase":
        phrases_cache = _load_phrases_cache()
    gold = load_gold()
    claims = _read_public_claims(connection)
    claims_index = {str(c["claim_id"]): c for c in claims}
    build_holder_table(claims, gold)
    cadres_c_ref = _load_cadres_c_reference() if spec.get("verifier_gabarit") else None
    questions = e3.load_questions()
    cadres: dict[str, dict[str, Any]] = {}
    from .infra import connect_existing, start_cluster

    for question in questions:
        qid = question["id"]
        for attempt in range(2):
            try:
                row = _compile_cadre_for_question(
                    question,
                    connection=connection,
                    embed_cache=embed_cache,
                    gold=gold,
                    claims_index=claims_index,
                    phrases_cache=phrases_cache,
                    config_name=config_name,
                )
                break
            except Exception as exc:
                if attempt == 0 and isinstance(exc, psycopg.errors.AdminShutdown):
                    start_cluster()
                    connection = connect_existing()
                    continue
                raise
        if cadres_c_ref is not None:
            ref_rendered = cadres_c_ref[qid]["rendered"]
            if not gabarit_socle_equivalent(ref_rendered, row["rendered"]):
                raise RuntimeError(
                    f"gabarit divergent pour {qid}/{config_name} : seule la liste d'éléments doit différer"
                )
        cadres[qid] = row
    slim = {
        qid: {
            "caracteres": row["caracteres"],
            "genres": row["genres"],
            "evidence_count": row["evidence_count"],
            **(
                {"deux_cotes_dans_contexte": row["deux_cotes_dans_contexte"]}
                if "deux_cotes_dans_contexte" in row
                else {}
            ),
            **({"elements_ecartes": row["elements_ecartes"]} if row.get("elements_ecartes") else {}),
        }
        for qid, row in cadres.items()
    }
    paths["sorties_config"].mkdir(parents=True, exist_ok=True)
    paths["cadres"].write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cadres


def build_response_invites(
    questions: list[dict[str, Any]],
    cadres: dict[str, dict[str, Any]],
    *,
    config_name: str | None = None,
) -> list[dict[str, Any]]:
    config_name = config_name or _active_config_name
    invites: list[dict[str, Any]] = []
    for question in questions:
        qid = question["id"]
        rendered = cadres[qid]["rendered"]
        invite = e3.build_response_invite(config_name, question, context=rendered)
        invite["config"] = config_name
        invites.append(invite)
    return invites


def stability_pairs_config(
    questions: list[dict[str, Any]] | None = None,
    *,
    config_name: str | None = None,
) -> list[tuple[str, str]]:
    config_name = config_name or _active_config_name
    all_questions = questions or e3.load_questions()
    pairs = e3.stability_pairs(all_questions)
    return [(config_name, qid) for _bras, qid in pairs]


def write_invites(invites: list[dict[str, Any]], *, config_name: str | None = None) -> None:
    paths = config_paths(config_name or _active_config_name)
    paths["sorties_config"].mkdir(parents=True, exist_ok=True)
    paths["invites"].write_text(
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
    paths = active_paths()
    calls_path = calls_path or paths["appels"]
    config_name = _active_config_name
    prior_max = e3.MAX_CALLS
    e3.MAX_CALLS = MAX_CALLS
    try:
        invites = load_invites_reponses(paths["invites"])
        completed, invite_hashes = e3.load_journal_state(calls_path)
        stability = {e3.invite_id(config_name, qid) for _bras, qid in stability_pairs_config()}
        for invite in invites:
            ident = invite["id"]
            if ident in invite_hashes:
                e3.verify_invite_hash(ident, invite_hashes[ident], invite)
            if ident in completed:
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
    finally:
        e3.MAX_CALLS = prior_max


def run_juge(chat_fn: e3.ChatFn, *, calls_path: Path | None = None) -> dict[str, Any]:
    paths = active_paths()
    calls_path = calls_path or paths["appels"]
    prior_max = e3.MAX_CALLS
    e3.MAX_CALLS = MAX_CALLS
    try:
        invites = load_invites_reponses(paths["invites"])
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
    finally:
        e3.MAX_CALLS = prior_max


def compute_stabilite(
    calls_path: Path,
    questions: list[dict[str, Any]] | None = None,
    *,
    config_name: str | None = None,
) -> dict[str, Any]:
    config_name = config_name or _active_config_name
    questions = questions or e3.load_questions()
    qmap = {q["id"]: q for q in questions}
    pairs = stability_pairs_config(questions, config_name=config_name)
    text_identical = 0
    nombres_identical = 0
    total = 0
    for _bras, qid in pairs:
        base_id = e3.invite_id(config_name, qid)
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


def _user_char_stats(invites: list[dict[str, Any]]) -> dict[str, int]:
    sizes = [int(item.get("user_chars", len(item.get("user", "")))) for item in invites]
    return {
        "mediane": int(statistics.median(sizes)) if sizes else 0,
        "p90": _percentile(sizes, 90),
    }


def _notations_metrics(payload: dict[str, Any], *, config_name: str) -> dict[str, Any]:
    notes = payload.get("notes") or []
    reglage, ecart = partager(e3.load_questions())
    reglage_ids = {q["id"] for q in reglage}
    ecart_ids = {q["id"] for q in ecart}
    by_qid = {n["question_id"]: n for n in notes}
    disputes = [n for n in notes if n.get("shape") == "dispute"]
    conformes = sum(1 for n in notes if n.get("verdict") == "conforme")
    refus = sum(1 for n in notes if n.get("refus"))
    disputes_conformes = sum(1 for n in disputes if n.get("verdict") == "conforme")
    reglage_conf = sum(1 for qid in reglage_ids if by_qid.get(qid, {}).get("verdict") == "conforme")
    ecart_conf = sum(1 for qid in ecart_ids if by_qid.get(qid, {}).get("verdict") == "conforme")
    tous_les_nombres = sum(1 for n in notes if n.get("tous_les_nombres"))
    tous_les_nombres_attendus = sum(1 for n in notes if n.get("nombres_attendus"))
    tableaux = e3._build_tableaux(notes)
    par_forme: dict[str, dict[str, Any]] = {}
    for shape, by_bras in sorted((tableaux.get("141") or {}).items()):
        cell = by_bras.get(config_name) or next(iter(by_bras.values()), {})
        par_forme[shape] = {
            "conforme": cell.get("conforme", 0),
            "tous_les_nombres": cell.get("tous_les_nombres", "0/0"),
        }
    return {
        "conformes": conformes,
        "disputes_conformes": disputes_conformes,
        "reglage_conformes": reglage_conf,
        "ecart_conformes": ecart_conf,
        "refus": refus,
        "tous_les_nombres": f"{tous_les_nombres}/{tous_les_nombres_attendus}",
        "par_forme": par_forme,
        "stabilite": payload.get("stabilite") or {},
    }


def write_noter_report(
    local_payload: dict[str, Any],
    *,
    reference_path: Path,
    output_path: Path,
    local_name: str,
    reference_name: str,
) -> None:
    if not reference_path.is_file():
        raise RuntimeError(f"notations de référence absentes : {reference_path}")
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    local_metrics = _notations_metrics(local_payload, config_name=local_name)
    reference_metrics = _notations_metrics(reference_payload, config_name=reference_name)
    shapes = sorted(set(local_metrics["par_forme"]) | set(reference_metrics["par_forme"]))
    lines = [
        f"# Rapport — notation étage 3 (`{local_name}` vs `{reference_name}`)",
        "",
        f"Date : {date.today().isoformat()}",
        "",
        "## Synthèse",
        "",
        "| Mesure | " + local_name + " | " + reference_name + " |",
        "| --- | --- | --- |",
        f"| Conformes /141 | {local_metrics['conformes']}/141 | {reference_metrics['conformes']}/141 |",
        (
            f"| Disputes conformes /28 | {local_metrics['disputes_conformes']}/28 "
            f"({local_metrics['reglage_conformes']}/14 · {local_metrics['ecart_conformes']}/14) | "
            f"{reference_metrics['disputes_conformes']}/28 "
            f"({reference_metrics['reglage_conformes']}/14 · {reference_metrics['ecart_conformes']}/14) |"
        ),
        f"| Refus | {local_metrics['refus']} | {reference_metrics['refus']} |",
        f"| Tous les nombres | {local_metrics['tous_les_nombres']} | {reference_metrics['tous_les_nombres']} |",
        "",
        "## Par forme",
        "",
        "| Forme | Conformes (" + local_name + ") | Conformes (" + reference_name + ") | "
        "Tous les nombres (" + local_name + ") | Tous les nombres (" + reference_name + ") |",
        "| --- | --- | --- | --- | --- |",
    ]
    for shape in shapes:
        local_shape = local_metrics["par_forme"].get(shape, {})
        ref_shape = reference_metrics["par_forme"].get(shape, {})
        lines.append(
            f"| {shape} | {local_shape.get('conforme', 0)} | {ref_shape.get('conforme', 0)} | "
            f"{local_shape.get('tous_les_nombres', '0/0')} | {ref_shape.get('tous_les_nombres', '0/0')} |"
        )
    local_stab = local_metrics["stabilite"]
    ref_stab = reference_metrics["stabilite"]
    lines.extend(
        [
            "",
            "## Stabilité",
            "",
            "| Mesure | " + local_name + " | " + reference_name + " |",
            "| --- | --- | --- |",
            f"| Réponses identiques | {local_stab.get('reponses_identiques', '—')} | "
            f"{ref_stab.get('reponses_identiques', '—')} |",
            f"| Tous les nombres identiques | {local_stab.get('tous_les_nombres_identiques', '—')} | "
            f"{ref_stab.get('tous_les_nombres_identiques', '—')} |",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_noter(
    *,
    calls_path: Path | None = None,
    invites_path: Path | None = None,
    output_path: Path | None = None,
    config_name: str | None = None,
) -> dict[str, Any]:
    config_name = config_name or _active_config_name
    spec = CONFIGURATIONS[config_name]
    paths = config_paths(config_name)
    calls_path = calls_path or paths["appels"]
    invites = load_invites_reponses(invites_path or paths["invites"])
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
    stabilite = compute_stabilite(calls_path, config_name=config_name)
    payload = {
        "date": date.today().isoformat(),
        "config": config_name,
        "notes": notes,
        "stabilite": stabilite,
        "complet": not manquants,
        "manquants": sorted(manquants),
    }
    out = output_path or paths["notations"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reference_path = spec.get("reference_notations")
    if reference_path:
        write_noter_report(
            payload,
            reference_path=Path(reference_path),
            output_path=paths["noter_rapport"],
            local_name=config_name,
            reference_name="fine_tours_k12",
        )
    return payload


def estimate_cost(invites: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [int(item.get("user_chars", len(item.get("user", "")))) for item in invites]
    tokens = [s / 4 for s in sizes]
    n_invites = len(invites)
    appels_prevus = n_invites + STABILITY_PAIRS + n_invites
    token_c_sizes = [
        int(item.get("user_chars", len(item.get("user", ""))))
        for item in e3.load_invites_reponses()
        if item["bras"] == "C"
    ]
    token_c_median = round(statistics.median(token_c_sizes) / 4) if token_c_sizes else 0
    median_tokens = round(statistics.median(tokens)) if tokens else 0
    return {
        "appels_prevus": appels_prevus,
        "reponses": n_invites,
        "rejeux_stabilite": STABILITY_PAIRS,
        "jugements": n_invites,
        "tokens_medians": median_tokens,
        "tokens_medians_c": token_c_median,
        "rapport_tokens_c": round(median_tokens / max(token_c_median, 1), 2),
    }


def write_report(
    invites: list[dict[str, Any]],
    cadres_slim: dict[str, dict[str, Any]],
    *,
    config_name: str | None = None,
    cadres_full: dict[str, dict[str, Any]] | None = None,
) -> None:
    config_name = config_name or _active_config_name
    spec = CONFIGURATIONS[config_name]
    paths = config_paths(config_name)
    cost = estimate_cost(invites)
    distribution = aggregate_distribution(cadres_slim)
    dispute_rows = [
        (qid, row)
        for qid, row in cadres_slim.items()
        if "deux_cotes_dans_contexte" in row
    ]
    deux_cotes = sum(1 for _qid, row in dispute_rows if row["deux_cotes_dans_contexte"])
    elements = distribution["elements"]
    chars = distribution["caracteres"]
    user_chars = _user_char_stats(invites)
    cadres_elements_ecartes = sum(1 for row in (cadres_full or {}).values() if int(row.get("elements_ecartes", 0)) > 0)
    tronques = sum(int(row.get("passages_tronques", 0)) for row in (cadres_full or {}).values())
    if spec["kind"] == "fine_tours":
        config_line = f"Configuration : `{config_name}` — couche fine, k = {spec['top_k']}, trois tours (sans complétion)"
        title = "# Rapport — marche à blanc étage 3 (configuration gagnante)"
    else:
        config_line = f"Configuration : `{config_name}` — hybride 12+6 + 12 fenêtres vectorielles globales"
        title = "# Rapport — marche à blanc étage 3 (hybride fenêtres)"
    lines = [
        title,
        "",
        f"Date : {date.today().isoformat()}",
        "",
        config_line,
        "",
        "## Invites",
        "",
        f"- Questions : {len(invites)}",
        f"- user_chars médiane / p90 : {user_chars['mediane']} / {user_chars['p90']}",
    ]
    if spec.get("verifier_gabarit"):
        lines.append("- Gabarit : identique au bras C (CallFrame)")
    lines.extend(
        [
            "",
            "## Distribution du contexte",
            "",
            "| Mesure | Éléments | Caractères |",
            "| --- | --- | --- |",
            f"| Médiane | {elements['mediane']} | {chars['mediane']} |",
            f"| 90ᵉ centile | {elements['p90']} | {chars['p90']} |",
            f"| Maximum | {elements['max']} | {chars['max']} |",
            "",
            f"- Contextes au-dessus de 8 éléments : {elements['au_dessus_8']}",
            f"- Contextes au-dessus de 12 éléments : {elements['au_dessus_12']}",
            f"- Contextes au-dessus de 16 éléments : {elements['au_dessus_16']}",
            f"- Contextes au plafond {MAX_RENDER_CHARS} caractères : {chars['plafond_8000']}",
            f"- Cadres avec éléments non injectés (budget 12 000) : {cadres_elements_ecartes}",
            f"- Marqueurs « tronqué » : {tronques}",
            "",
            "## Contexte disputes (étage 2, verdict plancher)",
            "",
            f"- Deux côtés dans le contexte : {deux_cotes}/28",
            "",
            "## Coût estimé (avant rechargement Grok)",
            "",
            "| Poste | Appels |",
            "| --- | --- |",
            f"| Réponses | {cost['reponses']} |",
            f"| Rejeux stabilité | {cost['rejeux_stabilite']} |",
            f"| Jugements must_not_claim | {cost['jugements']} |",
            f"| **Total** | **{cost['appels_prevus']}** ≤ {MAX_CALLS} |",
            "",
            "## Tokens d'entrée (user / 4)",
            "",
            f"- Médiane `{config_name}` : {cost['tokens_medians']} tokens",
            f"- Médiane bras C : {cost['tokens_medians_c']} tokens",
            f"- Rapport à C : {cost['rapport_tokens_c']} ×",
            "",
        ]
    )
    paths["rapport"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_and_prepare(connection, *, config_name: str | None = None) -> dict[str, Any]:
    config_name = config_name or _active_config_name
    spec = CONFIGURATIONS[config_name]
    paths = config_paths(config_name)
    cadres = compile_cadres(connection, config_name=config_name)
    questions = e3.load_questions()
    invites = build_response_invites(questions, cadres, config_name=config_name)
    write_invites(invites, config_name=config_name)
    slim = json.loads(paths["cadres"].read_text(encoding="utf-8"))
    distribution = aggregate_distribution(slim)
    deux_cotes = sum(1 for row in slim.values() if row.get("deux_cotes_dans_contexte"))
    attendu = int(spec["deux_cotes_attendu"])
    if deux_cotes != attendu:
        raise RuntimeError(
            f"divergence disputes deux côtés : {deux_cotes}/28 attendu {attendu}/28 ({config_name})"
        )
    if spec["kind"] == "fine_tours" and distribution["caracteres"]["plafond_8000"] != 0:
        raise RuntimeError(
            f"{distribution['caracteres']['plafond_8000']} contexte(s) au plafond {MAX_RENDER_CHARS} caractères"
        )
    tronques = sum(int(row.get("passages_tronques", 0)) for row in cadres.values())
    if tronques != 0:
        raise RuntimeError(f"marqueurs tronqués détectés : {tronques}")
    write_report(invites, slim, config_name=config_name, cadres_full=cadres)
    cost = estimate_cost(invites)
    user_chars = _user_char_stats(invites)
    cadres_elements_ecartes = sum(1 for row in cadres.values() if int(row.get("elements_ecartes", 0)) > 0)
    return {
        "cadres": len(cadres),
        "invites": len(invites),
        "deux_cotes_disputes": deux_cotes,
        "distribution": distribution,
        "user_chars_mediane": user_chars["mediane"],
        "user_chars_p90": user_chars["p90"],
        "cadres_elements_ecartes": cadres_elements_ecartes,
        "tronques": tronques,
        **cost,
    }


def run_marche_a_blanc(*, config_name: str | None = None) -> dict[str, Any]:
    from .infra import connect_existing, start_cluster

    config_name = config_name or _active_config_name
    paths = config_paths(config_name)
    export_runtime_env()
    write_banc_config_files()
    paths["sorties_config"].mkdir(parents=True, exist_ok=True)
    start_cluster()
    connection = connect_existing()
    return compile_and_prepare(connection, config_name=config_name)


def _run_tests() -> tuple[int, str]:
    import subprocess
    import sys

    worktree = Path(__file__).resolve().parents[2]
    python = sys.executable
    proc = subprocess.run(
        [python, "-m", "pytest", "tests/banc_realite", "-q"],
        cwd=str(worktree),
        capture_output=True,
        text=True,
        env={
            **dict(os.environ),
            "PYTHONPATH": f"{worktree / 'src'}:{worktree / 'scripts'}:{worktree}",
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


def _dry_run_marker_payload(prep: dict[str, Any], tests_summary: str, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "status": "complete",
        "tests": tests_summary,
        "dry_run": {
            "invites": prep["invites"],
            "user_chars_mediane": prep["user_chars_mediane"],
            "user_chars_p90": prep["user_chars_p90"],
            "tokens_mediane": prep["tokens_medians"],
            "cadres_elements_ecartes": prep["cadres_elements_ecartes"],
            "tronques": prep["tronques"],
            "deux_cotes_etage2": f"{prep['deux_cotes_disputes']}/28",
        },
        "files_written": [
            str(paths["cadres"]),
            str(paths["invites"]),
            str(paths["rapport"]),
        ],
    }


def _write_marker_result(payload: dict[str, Any], *, config_name: str) -> None:
    paths = config_paths(config_name)
    paths["result"].parent.mkdir(parents=True, exist_ok=True)
    paths["result"].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_prepare() -> dict[str, Any]:
    paths = active_paths()
    test_code, tests_summary = _run_tests()
    if test_code != 0:
        write_arret("tests", tests_summary)
        raise RuntimeError(tests_summary)
    try:
        prep = run_marche_a_blanc()
    except Exception as exc:
        write_arret("compilation", str(exc))
        raise
    payload = {
        "status": "pret",
        "invites": prep["invites"],
        "appels_modele": 0,
        "appels_prevus": prep["appels_prevus"],
        "tokens_medians": prep["tokens_medians"],
        "deux_cotes_disputes": prep["deux_cotes_disputes"],
        "suite": tests_summary,
        "files": [
            str(paths["cadres"]),
            str(paths["invites"]),
            str(paths["rapport"]),
        ],
    }
    _write_marker_result(payload, config_name=_active_config_name)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Étage 3 — lanceur paramétrable (fine k=12 ou hybride fenêtres).")
    parser.add_argument(
        "--config",
        choices=tuple(CONFIGURATIONS),
        default="fine_tours_k12",
        help="Configuration à exécuter (défaut: fine_tours_k12).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compiler les cadres et écrire les invites sans appel modèle (défaut).",
    )
    parser.add_argument("--reponses", action="store_true", help="Collecter les réponses via le relais.")
    parser.add_argument("--juge", action="store_true", help="Juger must_not_claim via le relais.")
    parser.add_argument("--noter", action="store_true", help="Calculer les verdicts depuis le journal.")
    parser.add_argument(
        "--fournisseur",
        choices=("relais", "cursor"),
        default="relais",
        help="Fournisseur modèle pour --reponses / --juge (défaut: relais).",
    )
    parser.add_argument(
        "--modele",
        default="cursor-grok-4.5-high",
        help="Modèle cursor-agent (uniquement avec --fournisseur cursor).",
    )
    args = parser.parse_args(argv)
    select_config(args.config)
    paths = active_paths()

    if args.reponses or args.juge:
        from .cursor_chat import assert_cursor_model_available, make_cursor_chat_fn

        try:
            if args.fournisseur == "cursor":
                assert_cursor_model_available(args.modele)
            else:
                assert_relay_responds_200()
            chat_fn = choisir_chat_fn(
                args.fournisseur,
                args.modele,
                fabrique_relais=e3.make_relay_chat_fn,
                fabrique_cursor=lambda modele: make_cursor_chat_fn(model=modele),
            )
        except (RuntimeError, ValueError) as exc:
            print(json.dumps({"status": "refuse", "erreur": str(exc)}, ensure_ascii=False))
            return 1
        if args.reponses:
            result = run_reponses(chat_fn)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        result = run_juge(chat_fn)
        print(json.dumps(result, ensure_ascii=False))
        return 0

    if args.noter:
        result = run_noter()
        print(json.dumps(result, ensure_ascii=False))
        return 0

    if args.dry_run:
        test_code, tests_summary = _run_tests()
        if test_code != 0:
            write_arret("tests", tests_summary)
            if args.config == "hybride_fenetres_g12":
                paths["arret"].parent.mkdir(parents=True, exist_ok=True)
                paths["arret"].write_text(
                    json.dumps({"status": "arret", "pourquoi": tests_summary}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps({"status": "arret", "erreur": tests_summary}, ensure_ascii=False))
            return 1
        try:
            prep = run_marche_a_blanc(config_name=args.config)
        except Exception as exc:
            write_arret("compilation", str(exc))
            if args.config == "hybride_fenetres_g12":
                paths["arret"].parent.mkdir(parents=True, exist_ok=True)
                paths["arret"].write_text(
                    json.dumps({"status": "arret", "pourquoi": str(exc)}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps({"status": "arret", "erreur": str(exc)}, ensure_ascii=False))
            return 1
        if args.config == "hybride_fenetres_g12":
            marker = _dry_run_marker_payload(prep, tests_summary, paths)
            _write_marker_result(marker, config_name=args.config)
            print(json.dumps(marker, ensure_ascii=False, indent=2))
            return 0
        payload = {
            "status": "pret",
            "invites": prep["invites"],
            "appels_modele": 0,
            "appels_prevus": prep["appels_prevus"],
            "tokens_medians": prep["tokens_medians"],
            "deux_cotes_disputes": prep["deux_cotes_disputes"],
            "files": [str(paths["cadres"]), str(paths["invites"]), str(paths["rapport"])],
        }
        _write_marker_result(payload, config_name=args.config)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    try:
        result = run_prepare()
    except RuntimeError as exc:
        print(json.dumps({"status": "arret", "erreur": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
