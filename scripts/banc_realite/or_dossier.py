# SPDX-License-Identifier: AGPL-3.0-only
"""Chargement de l'or par dossier et garde-fous de mesure."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .dossiers import Dossier, dossier_courant
from .normalize import piece_basename

_PREFIX_RE = re.compile(r"^[A-Za-z]+-[A-Za-z]")


def _prefixe_trouve(question_id: str) -> str:
    match = _PREFIX_RE.match(question_id or "")
    return match.group(0) if match else (question_id or "")


def _load_gold_greenloop(gold_dir: Path) -> dict[str, Any]:
    situations_blob = json.loads((gold_dir / "situations.json").read_text(encoding="utf-8"))
    assertions: list[dict[str, Any]] = []
    assertion_index: dict[str, dict[str, Any]] = {}
    for situation in situations_blob["situations"]:
        for assertion in situation.get("assertions", []):
            item = dict(assertion)
            item["situation_id"] = situation["id"]
            assertions.append(item)
            assertion_index[item["id"]] = item
    return {
        "situations": situations_blob["situations"],
        "disputes": situations_blob.get("disputes", []),
        "trajectories": situations_blob.get("trajectories", []),
        "reversibility_cases": situations_blob.get("reversibility_cases", []),
        "assertions": assertions,
        "assertion_index": assertion_index,
    }


def _load_gold_moreau(gold_dir: Path) -> dict[str, Any]:
    assertions_blob = json.loads((gold_dir / "assertions.json").read_text(encoding="utf-8"))
    questions_blob = json.loads((gold_dir / "questions.json").read_text(encoding="utf-8"))
    raw_assertions = assertions_blob.get("assertions") or assertions_blob
    assertions: list[dict[str, Any]] = []
    assertion_index: dict[str, dict[str, Any]] = {}
    for raw in raw_assertions:
        item = dict(raw)
        piece = str(item.get("piece") or item.get("source") or "").strip()
        item["source"] = piece
        assertions.append(item)
        assertion_index[item["id"]] = item
    questions = questions_blob.get("questions") or questions_blob
    disputes = [q for q in questions if q.get("shape") == "contradiction"]
    return {
        "situations": [],
        "disputes": disputes,
        "trajectories": [],
        "reversibility_cases": [],
        "assertions": assertions,
        "assertion_index": assertion_index,
    }


def load_gold(dossier: str | Dossier | None = None) -> dict[str, Any]:
    """Charge l'or depuis le gold_dir du descripteur (GreenLoop par défaut)."""
    current = dossier if isinstance(dossier, Dossier) else dossier_courant(dossier)
    gold_dir = current.gold_dir
    if current.nom == "greenloop":
        return _load_gold_greenloop(gold_dir)
    return _load_gold_moreau(gold_dir)


def load_questions(dossier: str | Dossier | None = None) -> list[dict[str, Any]]:
    """Charge les questions depuis le gold_dir du descripteur (GreenLoop par défaut)."""
    current = dossier if isinstance(dossier, Dossier) else dossier_courant(dossier)
    blob = json.loads((current.gold_dir / "questions.json").read_text(encoding="utf-8"))
    return blob["questions"]


def _pieces_corpus(dossier: Dossier) -> set[str]:
    if not dossier.corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus absent : {dossier.corpus_dir}")
    return {path.name for path in dossier.corpus_dir.iterdir() if path.is_file()}


def verifier_or_coherent(
    dossier: Dossier,
    questions: list[dict[str, Any]],
    gold: dict[str, Any],
) -> None:
    """Refuse toute mesure si l'or ne correspond pas au dossier demandé."""
    attendu = dossier.prefixe_question
    for question in questions:
        qid = str(question.get("id") or "")
        if not qid.startswith(attendu):
            trouve = _prefixe_trouve(qid)
            raise ValueError(
                f"or incohérent pour dossier {dossier.nom!r} : "
                f"préfixe attendu {attendu!r}, trouvé {trouve!r} (question {qid!r})"
            )

    corpus_pieces = _pieces_corpus(dossier)
    for assertion in gold.get("assertions") or []:
        piece = piece_basename(str(assertion.get("source") or assertion.get("piece") or ""))
        if piece and piece not in corpus_pieces:
            raise ValueError(
                f"or incohérent pour dossier {dossier.nom!r} : "
                f"pièce {piece!r} absente du corpus {dossier.corpus_dir}"
            )


def exiger_contextes_dossier(
    dossier: Dossier,
    questions: list[dict[str, Any]],
    embed_cache: dict[str, list[float]],
) -> None:
    """Refuse la mesure si les contextes du dossier n'ont pas encore été produits."""
    if dossier.nom == "greenloop":
        return
    missing = sorted(q["question"] for q in questions if q["question"] not in embed_cache)
    if missing:
        raise RuntimeError(
            f"aucun contexte {dossier.responsibility_context} n'a encore été produit"
        )
