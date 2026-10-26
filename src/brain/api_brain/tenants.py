# SPDX-License-Identifier: AGPL-3.0-only
"""Résolution des tenants Brain et jetons d'accès."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from banc_realite.dossiers import BANC_VAR, DOSSIERS, Dossier, LangueDossier
from banc_realite.recherche_entites import chemin_reconstitution

_ENV_TENANTS = "BRAIN_TENANTS"
_CACHE_QUESTIONS_NOM = "embeddings-questions.json"
_TOKEN_PREFIX = "BRAIN_TENANT_"
_TOKEN_SUFFIX = "_JETON"
REGISTRE_PIECES_DEFAUT = "lecture-drive"
JOURNAL_SYNCHRONISATION_NOM = "synchronisation.jsonl"
JOURNAL_RECONSTITUTION_NOM = "reconstitution.jsonl"
ATTENTE_SYNCHRONISATION_NOM = "synchronisation.en-attente.json"
COMPTEUR_EMBEDDINGS_PASSAGES_NOM = "compteur-embeddings-passages.json"
COMPTEUR_RECONSTITUTION_NOM = "compteur-reconstitution.json"
GENERATION_RECONSTITUTION_NOM = "generation.json"
ETIQUETTES_NOM = "etiquettes.v1.json"
MSG_SANS_DEPOT_BRUT = "tenant sans dépôt brut"
MSG_PLAFOND_NON_DECIDE = "plafond non décidé"

REGIMES_VALIDES = frozenset({"continu", "a_la_demande", "jamais"})
REGIME_DEFAUT = "a_la_demande"
_ORDRE_REGIME = {"jamais": 0, "a_la_demande": 1, "continu": 2}

_tenants_fichier: Path | None = None


def definir_tenants_fichier(chemin: Path | None) -> None:
    """Chemin JSON optionnel (ex. --tenants-fichier de run_api_brain)."""
    global _tenants_fichier
    _tenants_fichier = chemin


@dataclass(frozen=True, slots=True)
class TenantDossier(Dossier):
    """Dossier dérivé d'un tenant : cache des questions sous la racine du tenant."""

    tenant_racine: Path = field(kw_only=True)

    @property
    def cache_embeddings_questions(self) -> Path:
        return self.tenant_racine / _CACHE_QUESTIONS_NOM


@dataclass(frozen=True, slots=True)
class TenantConfig:
    """Configuration d'un tenant Brain.

    Table des chemins et du stockage (clés optionnelles de BRAIN_TENANTS) :

    depot_brut
        Répertoire du dépôt brut du connecteur. Sans lui, ``synchroniser``
        répond « tenant sans dépôt brut » et ne fait rien.
    registre_pieces
        Répertoire de lecture (``registre.json``, ``cache-lecture``,
        ``depot-brut.curseur.json``). Défaut : ``racine/lecture-drive``.
    pieces
        Corpus publié. Pour ``synchroniser`` doit être
        ``registre_pieces/cache-lecture/pieces/tous``.
    ontologie
        Fichier ontologie. ``etiquettes.v1.json`` est dans le même répertoire.
    reconstitution
        ``registre.json``, ``paquets/``, ``entites/``.
    journal
        ``racine/synchronisation.jsonl``.
    attente
        ``racine/synchronisation.en-attente.json``.
    couche brute
        Postgres ``public.memory_chunks`` filtré par ``vault_ref``, DSN
        ``dsn_effectif``.
    plafonds.embeddings_passages
        Obligatoire dès que ``depot_brut`` est renseigné (sinon « plafond
        non décidé »). Compteur : ``racine/compteur-embeddings-passages.json``.
    plafonds.reconstitution_appels
        Obligatoire dès qu'une famille est ``continu`` ou qu'on appelle
        ``reconstituer_unite_tenant``. Compteur :
        ``racine/compteur-reconstitution.json``.
    regimes
        Map famille → ``continu`` | ``a_la_demande`` | ``jamais``. Toute
        famille absente ou ``None`` vaut ``a_la_demande``.
    reconstitution_a_la_demande
        Si ``false``, l'API ne déclenche aucune reconstitution à la demande.
        Défaut ``true``. ``BRAIN_API_RECONSTITUTION_A_LA_DEMANDE=0`` prime.
    """

    nom: str
    racine: Path
    reconstitution: Path
    ontologie: Path
    vault_ref: str = ""
    pieces: Path | None = None
    dossier_banc: str | None = None
    registre_pieces: Path | None = None
    dsn: str | None = None
    langue: LangueDossier = "fr"
    pieces_tabulaires: frozenset[str] = field(default_factory=frozenset)
    plafonds: dict[str, int] = field(default_factory=dict)
    responsibility_context: str = ""
    depot_brut: Path | None = None
    regimes: dict[str, str] = field(default_factory=dict)
    reconstitution_a_la_demande: bool = True

    def chemin_ontologie_effectif(self) -> Path:
        return self.ontologie

    def chemin_reconstitution_effectif(self) -> Path:
        return self.reconstitution

    def chemin_marqueur_generation(self) -> Path:
        return self.reconstitution / GENERATION_RECONSTITUTION_NOM

    def dsn_effectif(self, defaut: str) -> str:
        return self.dsn if self.dsn else defaut

    def registre_pieces_effectif(self) -> Path:
        if self.registre_pieces is not None:
            return self.registre_pieces
        return self.racine / REGISTRE_PIECES_DEFAUT

    def chemin_cache_pieces_tous(self) -> Path:
        return self.registre_pieces_effectif() / "cache-lecture" / "pieces" / "tous"

    def chemin_etiquettes(self) -> Path:
        return self.ontologie.parent / ETIQUETTES_NOM

    def chemin_journal_synchronisation(self) -> Path:
        return self.racine / JOURNAL_SYNCHRONISATION_NOM

    def chemin_attente_synchronisation(self) -> Path:
        return self.racine / ATTENTE_SYNCHRONISATION_NOM

    def chemin_compteur_embeddings_passages(self) -> Path:
        return self.racine / COMPTEUR_EMBEDDINGS_PASSAGES_NOM

    def plafond_embeddings_passages(self) -> int:
        return int(self.plafonds.get("embeddings_passages") or 0)

    def chemin_journal_reconstitution(self) -> Path:
        return self.racine / JOURNAL_RECONSTITUTION_NOM

    def chemin_compteur_reconstitution(self) -> Path:
        return self.racine / COMPTEUR_RECONSTITUTION_NOM

    def plafond_reconstitution_appels(self) -> int:
        return int(self.plafonds.get("reconstitution_appels") or 0)

    def a_famille_continue(self) -> bool:
        return any(valeur == "continu" for valeur in self.regimes.values())

    def regime_pour(self, famille: str | None) -> str:
        """Régime de ``famille`` ; défaut ``a_la_demande`` si absente ou ``None``."""
        if not famille:
            return REGIME_DEFAUT
        valeur = self.regimes.get(famille)
        if valeur is None:
            return REGIME_DEFAUT
        return valeur

    def regime_le_plus_exigeant(self, familles: Iterable[str | None]) -> str:
        """``continu`` > ``a_la_demande`` > ``jamais`` ; liste vide → défaut."""
        items = list(familles)
        if not items:
            return REGIME_DEFAUT
        meilleur = "jamais"
        rang = _ORDRE_REGIME[meilleur]
        for famille in items:
            candidat = self.regime_pour(famille)
            rang_candidat = _ORDRE_REGIME[candidat]
            if rang_candidat > rang:
                meilleur = candidat
                rang = rang_candidat
        return meilleur

    def __post_init__(self) -> None:
        pieces = self.pieces
        vault_ref = self.vault_ref
        if self.dossier_banc and self.dossier_banc in DOSSIERS:
            ref = DOSSIERS[self.dossier_banc]
            if not vault_ref:
                vault_ref = ref.vault_ref
            if pieces is None:
                pieces = ref.corpus_dir
            if not self.responsibility_context:
                object.__setattr__(self, "responsibility_context", ref.responsibility_context)
        elif vault_ref in DOSSIERS and not self.responsibility_context:
            object.__setattr__(
                self, "responsibility_context", DOSSIERS[vault_ref].responsibility_context
            )
        if not vault_ref or pieces is None:
            raise TenantIncompletError(
                f"tenant «{self.nom}» incomplet : vault_ref et pieces requis"
            )
        if not self.responsibility_context:
            object.__setattr__(self, "responsibility_context", self.nom)
        object.__setattr__(self, "vault_ref", vault_ref)
        object.__setattr__(self, "pieces", pieces)


def normaliser_nom_tenant_env(tenant: str) -> str:
    """Normalise un identifiant tenant pour les variables d'environnement (A-Z, 0-9, _)."""
    return re.sub(r"[^a-zA-Z0-9]", "_", tenant).upper()


def _nom_env(tenant: str) -> str:
    return normaliser_nom_tenant_env(tenant)


class CollisionNomsTenantsError(ValueError):
    """Deux tenants distincts partagent le même préfixe de variable d'environnement."""


class DossierBancInconnuError(ValueError):
    """Un tenant référence un dossier_banc absent du registre DOSSIERS (compatibilité banc)."""


class TenantIncompletError(ValueError):
    """Entrée BRAIN_TENANTS sans dossier_banc et champs obligatoires manquants."""


class RegimeInconnuError(ValueError):
    """Valeur de régime hors ``continu`` | ``a_la_demande`` | ``jamais``."""


def _lire_jeton(tenant: str) -> str | None:
    cle = f"{_TOKEN_PREFIX}{_nom_env(tenant)}{_TOKEN_SUFFIX}"
    valeur = os.environ.get(cle)
    if valeur is None or valeur == "":
        return None
    return valeur


def jeton_configure(tenant: str) -> bool:
    return _lire_jeton(tenant) is not None


def verifier_jeton(tenant: str, propose: str | None) -> bool:
    attendu = _lire_jeton(tenant)
    if attendu is None:
        return False
    if propose is None:
        return False
    import hmac

    return hmac.compare_digest(propose, attendu)


def _racine_dossier_banc(nom: str, ref: Dossier) -> tuple[Path, bool]:
    """Racine du dossier banc ; bool = dérivée de BANC_VAR/dossiers/<nom>."""
    sous_banc = BANC_VAR / "dossiers" / nom
    if sous_banc.is_dir():
        return sous_banc.resolve(), True
    return ref.corpus_dir.parent.resolve(), False


def _ontologie_par_defaut(bloc: dict[str, Any]) -> bool:
    raw = bloc.get("ontologie")
    if raw is None:
        return True
    if not isinstance(raw, str) or not raw.strip():
        return True
    return raw.strip() == "ontologie/ontologie.v1.json"


def _resoudre_chemin(racine: Path, valeur: str) -> Path:
    path = Path(valeur).expanduser()
    if path.is_absolute():
        return path
    return (racine / path).resolve()


def _pieces_tabulaires_depuis_bloc(bloc: dict[str, Any], ref: Dossier | None) -> frozenset[str]:
    raw = bloc.get("pieces_tabulaires")
    if isinstance(raw, list):
        return frozenset(str(x) for x in raw)
    if ref is not None:
        return ref.pieces_tabulaires
    return frozenset()


def _plafonds_depuis_bloc(bloc: dict[str, Any]) -> dict[str, int]:
    raw = bloc.get("plafonds")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for cle, val in raw.items():
        if isinstance(cle, str) and isinstance(val, int):
            out[cle] = val
    return out


def _reconstitution_a_la_demande_depuis_bloc(bloc: dict[str, Any]) -> bool:
    raw = bloc.get("reconstitution_a_la_demande")
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    return True


def _regimes_depuis_bloc(nom: str, bloc: dict[str, Any]) -> dict[str, str]:
    raw = bloc.get("regimes")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RegimeInconnuError(
            f"tenant «{nom}» : regimes doit être un objet famille → régime"
        )
    out: dict[str, str] = {}
    for cle, val in raw.items():
        if not isinstance(cle, str) or not isinstance(val, str) or val not in REGIMES_VALIDES:
            raise RegimeInconnuError(
                f"tenant «{nom}» : régime inconnu pour la famille «{cle}» "
                f"(«{val}» ; attendu : continu, a_la_demande, jamais)"
            )
        out[cle] = val
    return out


def _langue_depuis_bloc(bloc: dict[str, Any], ref: Dossier | None) -> LangueDossier:
    raw = bloc.get("langue")
    if raw in ("fr", "en"):
        return raw
    if ref is not None:
        return ref.langue
    return "fr"


def _construire_tenant(nom: str, bloc: dict[str, Any]) -> TenantConfig:
    """Construit un tenant ; seul point de lecture de DOSSIERS (raccourci dossier_banc)."""
    dossier_banc_raw = bloc.get("dossier_banc")
    ref: Dossier | None = None
    dossier_banc: str | None = None

    if dossier_banc_raw is not None and str(dossier_banc_raw).strip() != "":
        dossier_banc = str(dossier_banc_raw)
        if dossier_banc not in DOSSIERS:
            connus = ", ".join(sorted(DOSSIERS))
            raise DossierBancInconnuError(
                f"dossier_banc invalide pour le tenant «{nom}» : "
                f"«{dossier_banc}» (attendu : {connus})"
            )
        ref = DOSSIERS[dossier_banc]

    racine_via_banc_var = False
    racine_raw = bloc.get("racine")
    if isinstance(racine_raw, str) and racine_raw.strip():
        racine = Path(racine_raw).expanduser().resolve()
    elif ref is not None and dossier_banc is not None:
        racine, racine_via_banc_var = _racine_dossier_banc(dossier_banc, ref)
    else:
        raise TenantIncompletError(
            f"tenant «{nom}» incomplet : «racine» obligatoire sans raccourci dossier_banc"
        )

    reconstitution_raw = bloc.get("reconstitution")
    if isinstance(reconstitution_raw, str) and reconstitution_raw.strip():
        reconstitution = _resoudre_chemin(racine, reconstitution_raw)
    elif ref is not None and dossier_banc is not None:
        reconstitution = chemin_reconstitution(dossier_banc).resolve()
    else:
        raise TenantIncompletError(
            f"tenant «{nom}» incomplet : «reconstitution» obligatoire sans raccourci dossier_banc"
        )

    ontologie_raw = bloc.get("ontologie", "ontologie/ontologie.v1.json")
    if not isinstance(ontologie_raw, str):
        ontologie_raw = "ontologie/ontologie.v1.json"
    ontologie = _resoudre_chemin(racine, ontologie_raw)
    if (
        ref is not None
        and dossier_banc is not None
        and racine_via_banc_var
        and _ontologie_par_defaut(bloc)
        and not ontologie.is_file()
    ):
        raise TenantIncompletError(
            f"tenant «{nom}» incomplet : ontologie absente pour le raccourci "
            f"dossier_banc «{dossier_banc}» (attendu : {ontologie})"
        )

    pieces_raw = bloc.get("pieces")
    if isinstance(pieces_raw, str) and pieces_raw.strip():
        pieces = _resoudre_chemin(racine, pieces_raw)
    elif ref is not None:
        pieces = ref.corpus_dir
    else:
        raise TenantIncompletError(
            f"tenant «{nom}» incomplet : «pieces» obligatoire sans raccourci dossier_banc"
        )

    vault_ref_raw = bloc.get("vault_ref")
    if isinstance(vault_ref_raw, str) and vault_ref_raw.strip():
        vault_ref = vault_ref_raw.strip()
    elif ref is not None:
        vault_ref = ref.vault_ref
    else:
        raise TenantIncompletError(
            f"tenant «{nom}» incomplet : «vault_ref» obligatoire sans raccourci dossier_banc"
        )

    registre_raw = bloc.get("registre_pieces")
    registre_pieces: Path | None = None
    if isinstance(registre_raw, str) and registre_raw.strip():
        registre_pieces = _resoudre_chemin(racine, registre_raw)

    depot_raw = bloc.get("depot_brut")
    depot_brut: Path | None = None
    if isinstance(depot_raw, str) and depot_raw.strip():
        depot_brut = _resoudre_chemin(racine, depot_raw)

    dsn_raw = bloc.get("dsn")
    dsn = str(dsn_raw).strip() if isinstance(dsn_raw, str) and dsn_raw.strip() else None

    if dossier_banc is None and vault_ref in DOSSIERS:
        dossier_banc = vault_ref

    if ref is None and vault_ref in DOSSIERS:
        ref = DOSSIERS[vault_ref]

    resp_raw = bloc.get("responsibility_context")
    if isinstance(resp_raw, str) and resp_raw.strip():
        responsibility_context = resp_raw.strip()
    elif ref is not None:
        responsibility_context = ref.responsibility_context
    else:
        responsibility_context = nom

    return TenantConfig(
        nom=nom,
        racine=racine,
        reconstitution=reconstitution,
        ontologie=ontologie,
        vault_ref=vault_ref,
        pieces=pieces,
        dossier_banc=dossier_banc,
        registre_pieces=registre_pieces,
        dsn=dsn,
        langue=_langue_depuis_bloc(bloc, ref),
        pieces_tabulaires=_pieces_tabulaires_depuis_bloc(bloc, ref),
        plafonds=_plafonds_depuis_bloc(bloc),
        responsibility_context=responsibility_context,
        depot_brut=depot_brut,
        regimes=_regimes_depuis_bloc(nom, bloc),
        reconstitution_a_la_demande=_reconstitution_a_la_demande_depuis_bloc(bloc),
    )


def dossier_de_tenant(cfg: TenantConfig) -> TenantDossier:
    """Construit le Dossier passé à la recherche API depuis la configuration tenant."""
    assert cfg.pieces is not None
    ref = DOSSIERS.get(cfg.dossier_banc) if cfg.dossier_banc else None
    if ref is not None:
        return TenantDossier(
            nom=ref.nom,
            gold_dir=ref.gold_dir,
            corpus_dir=cfg.pieces,
            area_root=ref.area_root,
            responsibility_context=cfg.responsibility_context or ref.responsibility_context,
            vault_ref=cfg.vault_ref,
            instance=ref.instance,
            prefixe_question=ref.prefixe_question,
            smoke_question_id=ref.smoke_question_id,
            nature=ref.nature,
            mode_presence_principal=ref.mode_presence_principal,
            langue=cfg.langue,
            pieces_tabulaires=cfg.pieces_tabulaires or ref.pieces_tabulaires,
            tenant_racine=cfg.racine,
        )
    return TenantDossier(
        nom=cfg.nom,
        gold_dir=cfg.racine / ".brain-api" / "gold",
        corpus_dir=cfg.pieces,
        area_root=cfg.racine / ".brain-api" / "aire",
        responsibility_context=cfg.responsibility_context or cfg.nom,
        vault_ref=cfg.vault_ref,
        instance=cfg.nom,
        prefixe_question="api",
        smoke_question_id="api-smoke",
        nature="business",
        mode_presence_principal="valeur",
        langue=cfg.langue,
        pieces_tabulaires=cfg.pieces_tabulaires,
        tenant_racine=cfg.racine,
    )


def _tenant_depuis_env_prefix(nom: str) -> TenantConfig | None:
    prefix = f"{_TOKEN_PREFIX}{_nom_env(nom)}_"
    racine_raw = os.environ.get(f"{prefix}RACINE")
    if not racine_raw:
        return None
    bloc: dict[str, Any] = {
        "racine": racine_raw,
        "reconstitution": os.environ.get(f"{prefix}RECONSTITUTION", ""),
        "ontologie": os.environ.get(f"{prefix}ONTOLOGIE", "ontologie/ontologie.v1.json"),
    }
    dossier_banc = os.environ.get(f"{prefix}DOSSIER_BANC")
    if dossier_banc:
        bloc["dossier_banc"] = dossier_banc
    vault = os.environ.get(f"{prefix}VAULT_REF")
    if vault:
        bloc["vault_ref"] = vault
    pieces = os.environ.get(f"{prefix}PIECES")
    if pieces:
        bloc["pieces"] = pieces
    depot_brut = os.environ.get(f"{prefix}DEPOT_BRUT")
    if depot_brut:
        bloc["depot_brut"] = depot_brut
    try:
        return _construire_tenant(nom, bloc)
    except (TenantIncompletError, DossierBancInconnuError):
        return None


def _tenant_depuis_json(nom: str, bloc: dict[str, Any]) -> TenantConfig:
    return _construire_tenant(nom, bloc)


def _verifier_collisions_noms_tenants(noms: list[str]) -> None:
    vu: dict[str, str] = {}
    for nom in noms:
        cle = normaliser_nom_tenant_env(nom)
        if cle in vu:
            autre = vu[cle]
            raise CollisionNomsTenantsError(
                f"collision de noms de tenant : «{nom}» et «{autre}» "
                f"produisent le même préfixe {_TOKEN_PREFIX}{cle}{_TOKEN_SUFFIX}"
            )
        vu[cle] = nom


def _charger_payload_tenants() -> dict[str, Any]:
    if _tenants_fichier is not None and _tenants_fichier.is_file():
        try:
            payload = json.loads(_tenants_fichier.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    raw = os.environ.get(_ENV_TENANTS)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def charger_registre_tenants() -> dict[str, TenantConfig]:
    tenants: dict[str, TenantConfig] = {}
    payload = _charger_payload_tenants()
    erreurs: list[ValueError] = []
    for nom, bloc in payload.items():
        if not isinstance(nom, str) or not isinstance(bloc, dict):
            continue
        try:
            tenants[nom] = _tenant_depuis_json(nom, bloc)
        except (TenantIncompletError, DossierBancInconnuError) as exc:
            erreurs.append(exc)
    for cle, valeur in os.environ.items():
        if not cle.startswith(_TOKEN_PREFIX) or not cle.endswith("_RACINE"):
            continue
        milieu = cle[len(_TOKEN_PREFIX) : -len("_RACINE")]
        if not milieu:
            continue
        nom = milieu.lower()
        if nom in tenants:
            continue
        cfg = _tenant_depuis_env_prefix(nom)
        if cfg is not None:
            tenants[nom] = cfg
    if erreurs:
        raise erreurs[0]
    _verifier_collisions_noms_tenants(list(tenants))
    return tenants


def resoudre_tenant(nom: str) -> TenantConfig | None:
    registre = charger_registre_tenants()
    return registre.get(nom)
