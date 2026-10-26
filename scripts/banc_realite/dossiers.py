# SPDX-License-Identifier: AGPL-3.0-only
"""Descripteurs de dossier pour le banc comparatif."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

BANC_VAR = Path(os.environ.get("BANC_VAR", "./var"))

NatureDossier = Literal["business", "juridique"]
ModePresencePrincipal = Literal["valeur", "verbatim"]
LangueDossier = Literal["en", "fr"]


@dataclass(frozen=True, slots=True)
class Dossier:
    nom: str
    gold_dir: Path
    corpus_dir: Path
    area_root: Path
    responsibility_context: str
    vault_ref: str
    instance: str
    prefixe_question: str
    smoke_question_id: str
    nature: NatureDossier
    mode_presence_principal: ModePresencePrincipal
    langue: LangueDossier
    pieces_tabulaires: frozenset[str]

    @property
    def cache_embeddings_questions(self) -> Path:
        """Cache d'embeddings des questions — GreenLoop sous sorties/, MOREAU sous couche-brute/."""
        if self.nom == "greenloop":
            return BANC_VAR / "sorties" / "etage2-question-embeddings.json"
        return BANC_VAR / "dossiers" / self.nom / "couche-brute" / "etage2-question-embeddings.json"

    @property
    def comparatif_dir(self) -> Path:
        """Répertoire des artefacts comparatif — GreenLoop sous sorties/comparatif/, MOREAU dédié."""
        if self.nom == "greenloop":
            return BANC_VAR / "sorties" / "comparatif"
        return BANC_VAR / "dossiers" / self.nom / "comparatif"

    @property
    def cache_phrases_embeddings(self) -> Path:
        """Cache des fenêtres phrase — GreenLoop sous sorties/, MOREAU sous couche-brute/."""
        if self.nom == "greenloop":
            return BANC_VAR / "sorties" / "phrases-embeddings.json"
        return BANC_VAR / "dossiers" / self.nom / "couche-brute" / "phrases-embeddings.json"

    @property
    def cache_phrases_embeddings_calibrees(self) -> Path:
        """Cache fenêtres calibrées sur l'or du dossier — MOREAU uniquement."""
        if self.nom == "greenloop":
            return BANC_VAR / "sorties" / "phrases-embeddings-calibrees.json"
        return (
            BANC_VAR
            / "dossiers"
            / self.nom
            / "couche-brute"
            / "phrases-embeddings-calibrees.json"
        )

    @property
    def meta_fenetre_calibree(self) -> Path:
        """Métadonnées de calibration (N phrases, calcul, couverture citations)."""
        return self.cache_phrases_embeddings_calibrees.parent / "fenetre-calibree.json"


DOSSIERS: dict[str, Dossier] = {
    "greenloop": Dossier(
        nom="greenloop",
        gold_dir=BANC_VAR / "dossiers" / "greenloop" / "gold",
        corpus_dir=BANC_VAR / "dossiers" / "greenloop" / "documents",
        area_root=BANC_VAR / "dossiers" / "greenloop" / "aire",
        responsibility_context="GreenLoop",
        vault_ref="greenloop",
        instance="greenloop",
        prefixe_question="gl-q",
        smoke_question_id="gl-q072",
        nature="business",
        mode_presence_principal="valeur",
        langue="en",
        pieces_tabulaires=frozenset(
            {
                "09-january-scale-export.md",
                "17-collection-register.md",
                "18-weigh-ticket-index.md",
                "24-customer-review.md",
                "28-collection-register-file-copy.md",
                "21-monthly-collections.csv",
                "22-recorded-visits.csv",
            }
        ),
    ),
    "moreau": Dossier(
        nom="moreau",
        gold_dir=BANC_VAR / "dossiers" / "moreau" / "gold",
        corpus_dir=BANC_VAR / "dossiers" / "moreau" / "documents",
        area_root=BANC_VAR / "dossiers" / "moreau" / "aire",
        responsibility_context="MOREAU",
        vault_ref="moreau",
        instance="moreau",
        prefixe_question="MOR-Q",
        smoke_question_id="MOR-Q01",
        nature="juridique",
        mode_presence_principal="verbatim",
        langue="fr",
        pieces_tabulaires=frozenset(
            {
                "03-bulletins-paie-nov-dec-2023.md",
                "04-planning-heures-sup.md",
            }
        ),
    ),
}


def dossier_courant(nom: str | None = None) -> Dossier:
    """Rend le descripteur du dossier nommé, GreenLoop par défaut."""
    key = nom if nom is not None else "greenloop"
    try:
        return DOSSIERS[key]
    except KeyError:
        known = ", ".join(sorted(DOSSIERS))
        raise ValueError(f"dossier inconnu : {key!r} (attendu : {known})") from None


def compter_or(dossier: Dossier) -> dict[str, int]:
    """Compte questions, assertions et pièces du corpus pour un dossier."""
    questions_path = dossier.gold_dir / "questions.json"
    if not questions_path.is_file():
        raise FileNotFoundError(f"questions absentes : {questions_path}")
    questions_blob = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = questions_blob.get("questions") or questions_blob

    if dossier.nom == "greenloop":
        situations_path = dossier.gold_dir / "situations.json"
        if not situations_path.is_file():
            raise FileNotFoundError(f"situations absentes : {situations_path}")
        situations_blob = json.loads(situations_path.read_text(encoding="utf-8"))
        assertions = sum(
            len(situation.get("assertions") or [])
            for situation in situations_blob.get("situations") or []
        )
    else:
        assertions_path = dossier.gold_dir / "assertions.json"
        if not assertions_path.is_file():
            raise FileNotFoundError(f"assertions absentes : {assertions_path}")
        assertions_blob = json.loads(assertions_path.read_text(encoding="utf-8"))
        assertions = len(assertions_blob.get("assertions") or assertions_blob)

    if not dossier.corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus absent : {dossier.corpus_dir}")
    pieces = sum(1 for path in dossier.corpus_dir.iterdir() if path.is_file())

    return {
        "questions": len(questions),
        "assertions": assertions,
        "pieces": pieces,
    }
