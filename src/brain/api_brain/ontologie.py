# SPDX-License-Identifier: AGPL-3.0-only
"""Lecture et décisions sur l'ontologie tenant."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from banc_realite.etiquetage_ontologie import _entites_actives_index

_verrous_ontologie: dict[str, threading.Lock] = {}
_verrou_registre_verrous = threading.Lock()


def _verrou_fichier_ontologie(path: Path) -> threading.Lock:
    cle = str(path.resolve())
    with _verrou_registre_verrous:
        if cle not in _verrous_ontologie:
            _verrous_ontologie[cle] = threading.Lock()
        return _verrous_ontologie[cle]


def _ecrire_ontologie_atomique(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def charger_ontologie(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ontologie invalide")
    return payload


def vue_ontologie(ontologie: Mapping[str, Any]) -> dict[str, Any]:
    actives = _entites_actives_index(ontologie)
    entites = []
    for nom in sorted(actives, key=lambda n: (actives[n].get("type") or "", n)):
        meta = actives[nom]
        entites.append(
            {
                "nom": nom,
                "type": meta.get("type") or "autre",
                "cles": list(meta.get("cles") or []),
                "variantes": list(meta.get("variantes") or []),
            }
        )
    decisions = ontologie.get("decisions") or []
    return {
        "version": ontologie.get("version"),
        "entites": entites,
        "relations": list(ontologie.get("relations") or []),
        "concepts": list(ontologie.get("concepts") or []),
        "decisions": len(decisions) if isinstance(decisions, list) else 0,
    }


def _ids_decidees(decisions: list[Mapping[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for decision in decisions:
        qid = decision.get("question_id")
        if isinstance(qid, str) and qid.strip():
            ids.add(qid.strip())
    return ids


def _annoter_questions(
    questions: list[Mapping[str, Any]],
    decidees: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id") or "")
        bloc = dict(item)
        bloc["statut"] = "decidee" if qid in decidees else "ouverte"
        out.append(bloc)
    return out


def vue_questions(ontologie: Mapping[str, Any]) -> dict[str, Any]:
    decidees = _ids_decidees(list(ontologie.get("decisions") or []))
    questions = _annoter_questions(list(ontologie.get("questions") or []), decidees)
    reserve = _annoter_questions(
        list(ontologie.get("questions_en_reserve") or []), decidees
    )
    return {"questions": questions, "questions_en_reserve": reserve}


def ajouter_decision(
    path: Path,
    *,
    question_id: str | None,
    texte: str,
    effet: str | None,
    decideur: str,
) -> dict[str, Any]:
    verrou = _verrou_fichier_ontologie(path)
    with verrou:
        ontologie = charger_ontologie(path)
        decisions = list(ontologie.get("decisions") or [])
        horodatage = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        bloc: dict[str, Any] = {
            "horodatage": horodatage,
            "texte": texte,
            "decideur": decideur,
        }
        if question_id:
            bloc["question_id"] = question_id
        if effet is not None:
            bloc["effet"] = effet
        decisions.append(bloc)
        ontologie["decisions"] = decisions
        _ecrire_ontologie_atomique(path, ontologie)
        return {"version": ontologie.get("version"), "decisions": len(decisions)}
