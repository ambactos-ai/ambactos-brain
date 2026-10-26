# SPDX-License-Identifier: AGPL-3.0-only
"""Compteur persistant d'appels cursor-agent pour l'extraction MRFS."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import fcntl
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cursor_chat import RunFn

DEFAULT_CONSTRUCTEUR_ROOT = (
    BANC_VAR / 'dossiers' / 'mrfs' / 'constructeur'
)
DEFAULT_PLAFOND = 85
DEFAULT_MODEL = "cursor-grok-4.5-high"
FOURNISSEUR = "cursor-agent"


class PlafondAtteint(RuntimeError):
    """Plafond d'appels atteint — aucun sous-processus ne doit être lancé."""

    def __init__(self, emis: int, plafond: int) -> None:
        super().__init__(f"plafond d'appels atteint ({emis}/{plafond})")
        self.emis = emis
        self.plafond = plafond


@dataclass
class CompteurAppels:
    compteur_path: Path = field(
        default_factory=lambda: DEFAULT_CONSTRUCTEUR_ROOT / "compteur-appels.json"
    )
    journal_path: Path = field(
        default_factory=lambda: DEFAULT_CONSTRUCTEUR_ROOT / "appels.jsonl"
    )
    plafond: int = DEFAULT_PLAFOND
    plafond_journalise: bool = False

    def _ouvrir_exclusif(self):
        self.compteur_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.compteur_path.open("a+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _lire_payload_verrouille(self, handle) -> tuple[dict[str, Any], bool]:
        handle.seek(0)
        raw = handle.read().strip()
        payload = json.loads(raw) if raw else {"plafond": self.plafond, "emis": 0}
        return payload, not raw

    def lire(self) -> dict[str, int]:
        """Lit le compteur sous le même verrou exclusif que les écritures."""
        with self._ouvrir_exclusif() as handle:
            payload, _vide = self._lire_payload_verrouille(handle)
            return {
                "plafond": int(payload.get("plafond", self.plafond)),
                "emis": int(payload.get("emis", 0)),
            }

    def definir_plafond(self, plafond: int, *, journaliser: bool = False) -> None:
        self.plafond = plafond
        self.plafond_journalise = journaliser
        with self._ouvrir_exclusif() as handle:
            payload, vide = self._lire_payload_verrouille(handle)
            if not vide and int(payload.get("plafond", plafond)) == plafond:
                return
            payload["plafond"] = plafond
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()

    def reserver_emission(self) -> int:
        """Incrémente et persiste avant le lancement du sous-processus."""
        self.compteur_path.parent.mkdir(parents=True, exist_ok=True)
        with self.compteur_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            raw = handle.read().strip()
            payload = json.loads(raw) if raw else {"plafond": self.plafond, "emis": 0}
            plafond = int(payload.get("plafond", self.plafond))
            emis = int(payload.get("emis", 0))
            if emis >= plafond:
                raise PlafondAtteint(emis, plafond)
            emis += 1
            payload["plafond"] = plafond
            payload["emis"] = emis
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            return emis

    def journaliser(
        self,
        *,
        emis: int,
        dossier: str,
        piece: str,
        morceau: int | None,
        nature: str,
        duree_s: float,
        issue: str,
        jetons: dict[str, Any] | None = None,
    ) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        etat = self.lire()
        ligne = {
            "horodatage": datetime.now(timezone.utc).isoformat(),
            "fournisseur": FOURNISSEUR,
            "modele": DEFAULT_MODEL,
            "dossier": dossier,
            "piece": piece,
            "morceau": morceau,
            "nature": nature,
            "emis": emis,
            "plafond": etat["plafond"],
            "duree_s": round(duree_s, 3),
            "jetons": jetons or {},
            "issue": issue,
        }
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ligne, ensure_ascii=False) + "\n")

    def envelopper_run_fn(
        self,
        run_fn: RunFn,
        *,
        dossier: str,
        piece: str,
        morceau: int | None,
        nature: str,
    ) -> RunFn:
        compteur = self

        def wrapped(
            argv: list[str], *, cwd: Path, env: dict[str, str], timeout: int
        ) -> tuple[int, str, str]:
            emis = compteur.reserver_emission()
            started = time.monotonic()
            issue = "echec"
            try:
                code, stdout, stderr = run_fn(
                    argv, cwd=cwd, env=env, timeout=timeout
                )
                issue = "succes" if code == 0 else "echec"
                return code, stdout, stderr
            except Exception:
                issue = "exception"
                raise
            finally:
                compteur.journaliser(
                    emis=emis,
                    dossier=dossier,
                    piece=piece,
                    morceau=morceau,
                    nature=nature,
                    duree_s=time.monotonic() - started,
                    issue=issue,
                )

        return wrapped


def make_compteur(
    *,
    constructeur_root: Path | None = None,
    plafond: int | None = None,
    plafond_explicite: bool = False,
) -> CompteurAppels:
    root = constructeur_root or DEFAULT_CONSTRUCTEUR_ROOT
    compteur = CompteurAppels(
        compteur_path=root / "compteur-appels.json",
        journal_path=root / "appels.jsonl",
    )
    if plafond is not None:
        compteur.definir_plafond(plafond, journaliser=plafond_explicite)
    else:
        etat = compteur.lire()
        compteur.plafond = etat["plafond"]
    return compteur
