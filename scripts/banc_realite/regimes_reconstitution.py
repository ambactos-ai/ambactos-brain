# SPDX-License-Identifier: AGPL-3.0-only
"""Régimes par famille et reconstitution unitaire réutilisable (brain#174)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from banc_realite.etiquetage_ontologie import (
    charger_registre_par_piece,
    famille_pour_piece,
)
from banc_realite.etiquetage_ontologie import _charger_regles_aiguillage
from banc_realite.pages_entites import generer_pages
from banc_realite.plafond_appels import CompteurAppels, PlafondAtteint
from banc_realite.reconstitution_entites import (
    ETAPES_PUBLICATION,
    charger_corpus,
    chemin_objet_en_cours,
    chemin_objet_publie,
    ecarter_objet_en_cours,
    enrichir_echecs,
    incrementer_generation_reconstitution,
    invalider_brut_unite,
    lire_registre_reconstitution,
    marquer_echec_publication,
    modifier_registre,
    pieces_referencees_par_paquet,
    persister_registre_reconstitution,
    publier_objet_en_cours,
    rassembler_une_unite,
    reconstituer_unite,
    retirer_echec,
    retirer_perimee_si_identique,
    retirer_publication_en_attente,
    unites_perimees,
    verrou_publication,
    verrou_registre,
)
from brain.api_brain.tenants import (
    MSG_PLAFOND_NON_DECIDE,
    TenantConfig,
)

ChatFn = Callable[[str, str], tuple[str, Any]]
VERROU_MAX_S = 2 * 3600
DECLENCHEURS = frozenset({"synchroniser", "api", "cli"})
ETAPES_FAITES = frozenset({"vecteurs", "promotion", "paquet", "generation"})
REGIMES_DEMANDE = frozenset({"a_la_demande", "continu"})


def _chat_reconstitution_isole(
    delai_s: int,
    *,
    compteur: CompteurAppels,
    dossier: str,
    piece: str,
) -> ChatFn:
    """Grok via cursor-agent isolé ; une réservation de plafond par tentative."""
    cache: dict[str, ChatFn] = {}

    def chat(system: str, user: str) -> tuple[str, Any]:
        fn = cache.get("fn")
        if fn is None:
            from banc_realite.cursor_chat import _run_fn_isole, make_cursor_chat_fn

            wrapped = compteur.envelopper_run_fn(
                _run_fn_isole,
                dossier=dossier,
                piece=piece,
                morceau=None,
                nature="reconstitution-entite",
            )
            fn = make_cursor_chat_fn(
                run_fn=wrapped, isole=True, timeout_s=delai_s
            )
            cache["fn"] = fn
        return fn(system, user)

    return chat


def make_compteur_reconstitution(cfg: TenantConfig) -> CompteurAppels:
    """Compteur d'appels modèle pour la reconstitution unitaire du tenant."""
    plafond = cfg.plafond_reconstitution_appels()
    if plafond <= 0:
        raise ValueError(MSG_PLAFOND_NON_DECIDE)
    compteur = CompteurAppels(
        compteur_path=cfg.chemin_compteur_reconstitution(),
        journal_path=cfg.racine / "appels-reconstitution.jsonl",
        plafond=plafond,
    )
    compteur.definir_plafond(plafond)
    return compteur


def charger_ontologie_tenant(cfg: TenantConfig) -> dict[str, Any]:
    """Ontologie du tenant avec ``sources.drive_registre`` pointé s'il existe."""
    if not cfg.ontologie.is_file():
        return {}
    ontologie = json.loads(cfg.ontologie.read_text(encoding="utf-8"))
    if not isinstance(ontologie, dict):
        ontologie = {}
    sources = ontologie.setdefault("sources", {})
    if isinstance(sources, dict):
        registre = cfg.registre_pieces_effectif() / "registre.json"
        if registre.is_file():
            sources["drive_registre"] = str(registre)
    return ontologie


def charger_paquet_unite(reconstitution: Path, unite_id: str) -> dict[str, Any]:
    path = reconstitution / "paquets" / f"{unite_id}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _chemin_paquet_en_cours(reconstitution: Path, unite_id: str) -> Path:
    return reconstitution / "en-cours" / f"{unite_id}.paquet.json"


def _promouvoir_paquet_en_cours(reconstitution: Path, unite_id: str) -> None:
    """Remplace ``paquets/<id>.json`` par le paquet mis en scène, atomiquement."""
    source = _chemin_paquet_en_cours(reconstitution, unite_id)
    if not source.is_file():
        return
    cible = reconstitution / "paquets" / f"{unite_id}.json"
    cible.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, cible)


def _ecarter_paquet_en_cours(
    reconstitution: Path,
    unite_id: str,
    *,
    maintenant: datetime | None = None,
) -> Path | None:
    """Déplace ``en-cours/<id>.paquet.json`` vers ``ecartes/``."""
    source = _chemin_paquet_en_cours(reconstitution, unite_id)
    if not source.is_file():
        return None
    ts = (maintenant or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    ecartes_dir = reconstitution / "ecartes"
    ecartes_dir.mkdir(parents=True, exist_ok=True)
    destination = ecartes_dir / f"{unite_id}.paquet.{ts}.json"
    if destination.exists():
        destination = ecartes_dir / f"{unite_id}.paquet.{ts}.{os.getpid()}.json"
    source.replace(destination)
    return destination


def familles_et_regime_unite(
    cfg: TenantConfig,
    paquet: Mapping[str, Any],
    ontologie: Mapping[str, Any],
) -> tuple[list[str], str]:
    """Familles nommées du paquet et régime le plus exigeant (défaut inclus)."""
    pieces = sorted(pieces_referencees_par_paquet(paquet))
    if not pieces:
        return [], cfg.regime_pour(None)
    regles = _charger_regles_aiguillage(ontologie)
    registre_par_piece = charger_registre_par_piece(ontologie)
    par_piece: list[str | None] = [
        famille_pour_piece(
            piece,
            ontologie,
            regles=regles,
            registre_par_piece=registre_par_piece,
        )
        for piece in pieces
    ]
    nommees = sorted({famille for famille in par_piece if famille})
    return nommees, cfg.regime_le_plus_exigeant(par_piece)


def regime_par_entite(
    cfg: TenantConfig,
    registre: Mapping[str, Any],
    ontologie: Mapping[str, Any],
    reconstitution: Path,
) -> dict[str, str]:
    """Régime le plus exigeant de chaque entité (union des familles de ses unités)."""
    par_entite: dict[str, list[str | None]] = {}
    for unite in registre.get("unites") or []:
        if not isinstance(unite, Mapping):
            continue
        nom = str(unite.get("entite") or "")
        uid = str(unite.get("id") or "")
        if not nom or not uid:
            continue
        paquet = charger_paquet_unite(reconstitution, uid)
        pieces = pieces_referencees_par_paquet(paquet) if paquet else set()
        if not pieces:
            par_entite.setdefault(nom, []).append(None)
            continue
        regles = _charger_regles_aiguillage(ontologie)
        registre_par_piece = charger_registre_par_piece(ontologie)
        for piece in pieces:
            par_entite.setdefault(nom, []).append(
                famille_pour_piece(
                    piece,
                    ontologie,
                    regles=regles,
                    registre_par_piece=registre_par_piece,
                )
            )
    return {
        nom: cfg.regime_le_plus_exigeant(familles)
        for nom, familles in par_entite.items()
    }


def _horodatage() -> str:
    return datetime.now(timezone.utc).isoformat()


def _verrou_perime(path: Path, *, maintenant: datetime | None = None) -> bool:
    now = maintenant or datetime.now(timezone.utc)
    ts: datetime | None = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("horodatage") if isinstance(payload, dict) else None
        if isinstance(raw, str) and raw:
            ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        ts = None
    if ts is None:
        try:
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return True
    return (now - ts).total_seconds() > VERROU_MAX_S


@contextmanager
def verrou_unite(reconstitution: Path, unite_id: str) -> Iterator[Path | None]:
    """Création exclusive de ``verrous/<id>.lock`` ; ``None`` si déjà tenu."""
    verrous = reconstitution / "verrous"
    verrous.mkdir(parents=True, exist_ok=True)
    path = verrous / f"{unite_id}.lock"
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            if _verrou_perime(path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            yield None
            return
    try:
        os.write(
            fd,
            json.dumps(
                {"pid": os.getpid(), "horodatage": _horodatage()},
                ensure_ascii=False,
            ).encode("utf-8"),
        )
        os.close(fd)
        yield path
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _append_journal(chemin: Path, ligne: dict[str, Any]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ligne, ensure_ascii=False) + "\n")


def noter_deja_a_jour(
    cfg: TenantConfig,
    unite_id: str,
    *,
    declencheur: str,
    entite: str = "",
    familles: Sequence[str] | None = None,
    regime: str = "",
) -> dict[str, Any]:
    """Journalise un saut de reconstitution : l'unité n'est plus candidate."""
    base: dict[str, Any] = {
        "unite": unite_id,
        "entite": entite,
        "familles": list(familles or []),
        "regime": regime,
        "declencheur": declencheur,
    }
    ligne = {
        **base,
        "horodatage": _horodatage(),
        "duree_murale_s": 0.0,
        "appels": 0,
        "statut": "deja_a_jour",
        "erreur": "",
    }
    _append_journal(cfg.chemin_journal_reconstitution(), ligne)
    return {**base, "statut": "deja_a_jour"}


def _lire_registre(reconstitution: Path) -> dict[str, Any]:
    with verrou_registre(reconstitution):
        return lire_registre_reconstitution(reconstitution)


def _unite_du_registre(registre: Mapping[str, Any], unite_id: str) -> dict[str, Any]:
    for unite in registre.get("unites") or []:
        if isinstance(unite, dict) and str(unite.get("id") or "") == unite_id:
            return unite
    raise KeyError(f"unité inconnue : {unite_id}")


def _dossier_tenant(cfg: TenantConfig) -> str:
    return cfg.dossier_banc or cfg.vault_ref or cfg.nom


def _plafond_atteint(compteur: CompteurAppels | None) -> bool:
    if compteur is None:
        return False
    etat = compteur.lire()
    return int(etat.get("emis") or 0) >= int(etat.get("plafond") or 0)


def _invalider_etat_api(nom: str) -> None:
    try:
        from brain.api_brain.recherche import vider_etat_tenant
    except Exception:
        return
    vider_etat_tenant(nom)


def _persister_registre(reconstitution: Path, registre: dict[str, Any]) -> None:
    with verrou_registre(reconstitution):
        persister_registre_reconstitution(reconstitution, registre)


def _normaliser_etapes_faites(brut: object) -> set[str]:
    faits: set[str] = set()
    if isinstance(brut, (list, tuple, set, frozenset)):
        for item in brut:
            nom = str(item)
            if nom in ETAPES_FAITES:
                faits.add(nom)
    return faits


def _etapes_faites_de(
    attente: Mapping[str, Any],
    reconstitution: Path,
    unite_id: str,
) -> set[str]:
    """Ensemble monotone des étapes déjà réussies (jamais rétréci)."""
    faits = _normaliser_etapes_faites(attente.get("etapes_faites"))
    etape = str(attente.get("etape") or "")
    if etape in {"promotion", "paquet", "generation"}:
        faits.add("vecteurs")
    if etape in {"paquet", "generation"}:
        faits.add("promotion")
    if etape == "generation":
        faits.add("paquet")
    if (
        chemin_objet_publie(reconstitution, unite_id).is_file()
        and not chemin_objet_en_cours(reconstitution, unite_id).is_file()
    ):
        faits.add("vecteurs")
        faits.add("promotion")
    return faits


def _persister_etapes_faites(
    reconstitution: Path,
    unite_id: str,
    etapes_faites: set[str],
    *,
    familles: Sequence[str] | None = None,
    regime: str | None = None,
    etape: str | None = None,
) -> None:
    faits = sorted(etapes_faites & ETAPES_FAITES)
    familles_eff = [str(item) for item in familles] if familles is not None else None
    etape_eff = etape if etape in ETAPES_FAITES or etape in ETAPES_PUBLICATION else None

    def modifier(registre: dict[str, Any]) -> bool:
        for unite in registre.get("unites") or []:
            if not isinstance(unite, dict) or str(unite.get("id") or "") != unite_id:
                continue
            attente = unite.get("publication_en_attente")
            if not isinstance(attente, dict):
                return False
            attente["etapes_faites"] = faits
            if familles_eff is not None:
                attente["familles"] = familles_eff
            if regime:
                attente["regime"] = regime
            if etape_eff:
                attente["etape"] = etape_eff
            if "promotion" in faits or "paquet" in faits or "generation" in faits:
                attente["objet"] = f"entites/{unite_id}.json"
            return True
        return False

    modifier_registre(reconstitution, modifier)


def _publication_en_attente_coherente(unite: Mapping[str, Any]) -> bool:
    """True si ``publication_en_attente.version_entree`` correspond au ``perimee``.

    Absence de ``perimee`` : l'attente reste cohérente. Un ``perimee`` plus
    récent invalide l'attente et force une reconstitution à neuf.
    """
    attente = unite.get("publication_en_attente")
    if not isinstance(attente, Mapping):
        return False
    perimee = unite.get("perimee")
    if not isinstance(perimee, Mapping):
        return True
    depuis = perimee.get("depuis")
    depuis_s = str(depuis) if depuis is not None else None
    version = attente.get("version_entree")
    version_s = str(version) if version is not None else None
    return version_s == depuis_s


def _familles_et_regime_marque(
    unite: Mapping[str, Any],
    cle: str,
) -> tuple[list[str], str] | None:
    """Régime et familles mémorisés sur ``cle``, s'ils existent."""
    attente = unite.get(cle)
    if not isinstance(attente, Mapping):
        return None
    regime = attente.get("regime")
    if not isinstance(regime, str) or not regime:
        return None
    brut = attente.get("familles")
    if isinstance(brut, list):
        familles = [str(item) for item in brut if str(item)]
    else:
        familles = []
    return familles, regime


def _familles_et_regime_attente(
    unite: Mapping[str, Any],
) -> tuple[list[str], str] | None:
    """Régime et familles mémorisés sur ``publication_en_attente``, s'ils existent."""
    return _familles_et_regime_marque(unite, "publication_en_attente")


def _reconstitution_en_attente_presente(unite: Mapping[str, Any]) -> bool:
    return isinstance(unite.get("reconstitution_en_attente"), Mapping)


def _familles_et_regime_reconstitution_attente(
    unite: Mapping[str, Any],
) -> tuple[list[str], str] | None:
    """Régime et familles mémorisés sur ``reconstitution_en_attente``."""
    return _familles_et_regime_marque(unite, "reconstitution_en_attente")


def _marquer_reconstitution_en_attente(
    reconstitution: Path,
    unite_id: str,
    *,
    regime: str,
    familles: Sequence[str],
    version_entree: str,
    depuis: str | None,
) -> bool:
    """Enregistre le régime déclencheur sous verrou, avant l'appel modèle."""

    def modifier(registre: dict[str, Any]) -> bool:
        for unite in registre.get("unites") or []:
            if not isinstance(unite, dict) or str(unite.get("id") or "") != unite_id:
                continue
            unite["reconstitution_en_attente"] = {
                "regime": regime,
                "familles": [str(item) for item in familles],
                "version_entree": version_entree,
                "depuis": depuis,
            }
            return True
        return False

    return modifier_registre(reconstitution, modifier)


def _retirer_reconstitution_en_attente(
    reconstitution: Path, unite_id: str
) -> bool:
    """Retire ``reconstitution_en_attente`` après succès de publication."""

    def modifier(registre: dict[str, Any]) -> bool:
        for unite in registre.get("unites") or []:
            if not isinstance(unite, dict) or str(unite.get("id") or "") != unite_id:
                continue
            if "reconstitution_en_attente" not in unite:
                return False
            del unite["reconstitution_en_attente"]
            return True
        return False

    return modifier_registre(reconstitution, modifier)


def _regime_depuis_attente_ou_paquet(
    cfg: TenantConfig,
    unite: Mapping[str, Any],
    reconstitution: Path,
    ontologie: Mapping[str, Any],
) -> tuple[list[str], str]:
    """Régime mémorisé sur une attente, sinon résolution depuis le paquet."""
    if _publication_en_attente_coherente(unite):
        memorises = _familles_et_regime_attente(unite)
        if memorises is not None:
            return memorises
        return [], ""
    memorises = _familles_et_regime_reconstitution_attente(unite)
    if memorises is not None:
        return memorises
    paquet = charger_paquet_unite(reconstitution, str(unite.get("id") or ""))
    return familles_et_regime_unite(cfg, paquet, ontologie)


def _etape_reprise(reconstitution: Path, unite: Mapping[str, Any]) -> str:
    """Première étape de publication encore à faire (hors rendu des pages)."""
    attente = unite.get("publication_en_attente")
    if not isinstance(attente, Mapping):
        return "pages"
    uid = str(unite.get("id") or "")
    faits = _etapes_faites_de(attente, reconstitution, uid)
    if "generation" in faits:
        return "generation"
    if "paquet" in faits:
        return "generation"
    if "promotion" in faits:
        return "generation"
    if "vecteurs" in faits:
        return "promotion"
    etape = str(attente.get("etape") or "")
    if etape not in ETAPES_PUBLICATION:
        return "pages"
    if etape == "generation" and chemin_objet_publie(reconstitution, uid).is_file():
        return "generation"
    if etape == "promotion":
        if chemin_objet_en_cours(reconstitution, uid).is_file():
            return "promotion"
        if chemin_objet_publie(reconstitution, uid).is_file():
            return "generation"
        return "pages"
    if chemin_objet_en_cours(reconstitution, uid).is_file():
        return etape
    return "pages"


def _capturer_perimee(
    unite: Mapping[str, Any],
) -> tuple[str | None, Any]:
    perimee = unite.get("perimee")
    if not isinstance(perimee, Mapping):
        return None, None
    brut = perimee.get("depuis")
    depuis = str(brut) if brut is not None else None
    return depuis, perimee.get("passe")


@contextmanager
def _vue_en_cours(
    reconstitution: Path,
    unite_id: str,
    *,
    registre: Mapping[str, Any] | None = None,
    echecs: Sequence[Mapping[str, Any]] | None = None,
) -> Iterator[Path]:
    """Répertoire temporaire : objets publiés, l'unité overlayée depuis ``en-cours/``."""
    en_cours = chemin_objet_en_cours(reconstitution, unite_id)
    publie = chemin_objet_publie(reconstitution, unite_id)
    if not en_cours.is_file() and not publie.is_file():
        raise FileNotFoundError(
            f"objet en cours ou publié absent : {en_cours}"
        )
    with tempfile.TemporaryDirectory(prefix="vue-en-cours-") as tmp_s:
        tmp = Path(tmp_s)
        if registre is not None:
            (tmp / "registre.json").write_text(
                json.dumps(registre, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(reconstitution / "registre.json", tmp / "registre.json")
        if echecs is not None:
            (tmp / "echecs.json").write_text(
                json.dumps(list(echecs), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        elif (reconstitution / "echecs.json").is_file():
            shutil.copy2(reconstitution / "echecs.json", tmp / "echecs.json")
        reemploi = reconstitution / "reemploi.json"
        if reemploi.is_file():
            shutil.copy2(reemploi, tmp / "reemploi.json")
        paquets_src = reconstitution / "paquets"
        paquets_dst = tmp / "paquets"
        if paquets_src.is_dir():
            paquets_dst.mkdir()
            for path in paquets_src.glob("*.json"):
                shutil.copy2(path, paquets_dst / path.name)
        paquet_en_cours = _chemin_paquet_en_cours(reconstitution, unite_id)
        if paquet_en_cours.is_file():
            paquets_dst.mkdir(exist_ok=True)
            shutil.copy2(paquet_en_cours, paquets_dst / f"{unite_id}.json")
        entites_dst = tmp / "entites"
        entites_dst.mkdir()
        entites_src = reconstitution / "entites"
        if entites_src.is_dir():
            for path in entites_src.glob("*.json"):
                shutil.copy2(path, entites_dst / path.name)
        if en_cours.is_file():
            shutil.copy2(en_cours, entites_dst / f"{unite_id}.json")
        yield tmp


def _registre_sans_perimee(
    registre: Mapping[str, Any], unite_id: str
) -> dict[str, Any]:
    copie = dict(registre)
    unites: list[Any] = []
    for unite in registre.get("unites") or []:
        if isinstance(unite, dict):
            item = dict(unite)
            if str(item.get("id") or "") == unite_id:
                item.pop("perimee", None)
                item.pop("publication_en_attente", None)
                item.pop("reconstitution_en_attente", None)
            unites.append(item)
        else:
            unites.append(unite)
    copie["unites"] = unites
    return copie


def _echecs_sans_unite(echecs_path: Path, unite_id: str) -> list[dict[str, Any]]:
    if not echecs_path.is_file():
        return []
    brut = json.loads(echecs_path.read_text(encoding="utf-8"))
    if not isinstance(brut, list):
        return []
    return [
        dict(item)
        for item in brut
        if isinstance(item, Mapping) and str(item.get("unite") or "") != unite_id
    ]


def reconstituer_unite_tenant(
    cfg: TenantConfig,
    unite_id: str,
    *,
    compteur: CompteurAppels | None,
    delai_s: int,
    chat: ChatFn | None = None,
    declencheur: str = "cli",
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    a_blanc: bool = False,
    ontologie: Mapping[str, Any] | None = None,
    seulement_si_eligible: bool = False,
) -> dict[str, Any]:
    """Invalide, reconstitue, régénère pages/vecteurs d'une unité, journalise.

    Verrou tenu → ``{"statut": "en_cours"}`` sans rien faire. Plafond atteint
    → ``{"statut": "plafond"}``. Régime ``jamais`` → pas d'appel modèle.
    Une ``publication_en_attente`` cohérente reste une republication seule,
    y compris si le paquet courant est vide ou de régime ``jamais``.
    Une ``reconstitution_en_attente`` conserve le régime déclencheur tant que
    le modèle n'a pas abouti.
    ``seulement_si_eligible=True`` (API) relit le registre sous ``verrou_unite``
    et saute si l'unité n'est plus candidate (``deja_a_jour``, pas d'appel
    modèle). Le CLI ``refaire --unite`` garde ``False``.
    """
    if declencheur not in DECLENCHEURS:
        declencheur = "cli"

    reconstitution = cfg.reconstitution
    registre = _lire_registre(reconstitution)
    unite = _unite_du_registre(registre, unite_id)
    ontologie_eff = dict(ontologie) if ontologie is not None else charger_ontologie_tenant(cfg)
    attente_coherente = _publication_en_attente_coherente(unite)
    republier_seulement = attente_coherente
    familles, regime = _regime_depuis_attente_ou_paquet(
        cfg, unite, reconstitution, ontologie_eff
    )
    entite = str(unite.get("entite") or "")
    base: dict[str, Any] = {
        "unite": unite_id,
        "entite": entite,
        "familles": familles,
        "regime": regime,
        "declencheur": declencheur,
    }

    if a_blanc:
        return {**base, "statut": "a_blanc"}

    if compteur is None or cfg.plafond_reconstitution_appels() <= 0:
        raise ValueError(MSG_PLAFOND_NON_DECIDE)

    if not attente_coherente and regime == "jamais":
        ligne = {
            **base,
            "horodatage": _horodatage(),
            "duree_murale_s": 0.0,
            "appels": 0,
            "statut": "jamais",
            "erreur": "",
        }
        _append_journal(cfg.chemin_journal_reconstitution(), ligne)
        return {**base, "statut": "jamais"}

    if not republier_seulement and _plafond_atteint(compteur):
        ligne = {
            **base,
            "horodatage": _horodatage(),
            "duree_murale_s": 0.0,
            "appels": 0,
            "statut": "plafond",
            "erreur": "plafond d'appels atteint",
        }
        _append_journal(cfg.chemin_journal_reconstitution(), ligne)
        return {**base, "statut": "plafond"}

    with verrou_unite(reconstitution, unite_id) as jeton:
        if jeton is None:
            return {**base, "statut": "en_cours"}
        if seulement_si_eligible:
            registre_frais = _lire_registre(reconstitution)
            try:
                unite_fraiche = _unite_du_registre(registre_frais, unite_id)
            except KeyError:
                return noter_deja_a_jour(
                    cfg,
                    unite_id,
                    declencheur=declencheur,
                    entite=entite,
                    familles=familles,
                    regime=regime,
                )
            if not _candidate_demande(unite_fraiche, reconstitution):
                return noter_deja_a_jour(
                    cfg,
                    unite_id,
                    declencheur=declencheur,
                    entite=str(unite_fraiche.get("entite") or entite),
                    familles=familles,
                    regime=regime,
                )
        chat_interne = chat is None
        chat_effectif = (
            chat
            if chat is not None
            else _chat_reconstitution_isole(
                delai_s,
                compteur=compteur,
                dossier=_dossier_tenant(cfg),
                piece=unite_id,
            )
        )
        return _executer_reconstitution(
            cfg,
            unite=unite,
            registre=registre,
            ontologie=ontologie_eff,
            familles=familles,
            regime=regime,
            compteur=compteur,
            delai_s=delai_s,
            chat=chat_effectif,
            declencheur=declencheur,
            embed_fn=embed_fn,
            republier_seulement=republier_seulement,
            reserver_emission_unite=not chat_interne,
        )


def _executer_reconstitution(
    cfg: TenantConfig,
    *,
    unite: dict[str, Any],
    registre: dict[str, Any],
    ontologie: Mapping[str, Any],
    familles: list[str],
    regime: str,
    compteur: CompteurAppels,
    delai_s: int,
    chat: ChatFn | None,
    declencheur: str,
    embed_fn: Callable[[list[str]], list[list[float]]] | None,
    republier_seulement: bool,
    reserver_emission_unite: bool = True,
) -> dict[str, Any]:
    reconstitution = cfg.reconstitution
    unite_id = str(unite["id"])
    registre = _lire_registre(reconstitution)
    unite = _unite_du_registre(registre, unite_id)
    attente_coherente = _publication_en_attente_coherente(unite)
    republier_seulement = attente_coherente
    if attente_coherente:
        memorises = _familles_et_regime_attente(unite)
        if memorises is not None:
            familles, regime = memorises
    else:
        memorises = _familles_et_regime_reconstitution_attente(unite)
        if memorises is not None:
            familles, regime = memorises
    entite = str(unite.get("entite") or "")
    pieces_dir = cfg.chemin_cache_pieces_tous()
    if not pieces_dir.is_dir() and cfg.pieces is not None:
        pieces_dir = cfg.pieces
    dossier = _dossier_tenant(cfg)
    emis_avant = int(compteur.lire().get("emis") or 0)
    started = time.monotonic()
    statut = "echec"
    erreur = ""
    reconstruction_ok = False
    depuis_capture, _passe_capture = _capturer_perimee(unite)
    attente = unite.get("publication_en_attente")
    if (
        republier_seulement
        and isinstance(attente, Mapping)
        and attente.get("version_entree") is not None
    ):
        version_entree = str(attente.get("version_entree"))
    else:
        version_entree = depuis_capture or _horodatage()
    etape_reprise = (
        _etape_reprise(reconstitution, unite) if republier_seulement else "pages"
    )
    try:
        if not republier_seulement:
            ecarter_objet_en_cours(reconstitution, unite_id)
            _ecarter_paquet_en_cours(reconstitution, unite_id)
            retirer_publication_en_attente(reconstitution, unite_id)
            invalider_brut_unite(reconstitution, unite_id)
            _marquer_reconstitution_en_attente(
                reconstitution,
                unite_id,
                regime=regime,
                familles=familles,
                version_entree=version_entree,
                depuis=depuis_capture,
            )
            paquet = rassembler_une_unite(
                unite,
                pieces_dir=pieces_dir,
                sortie_dir=reconstitution,
                destination=_chemin_paquet_en_cours(reconstitution, unite_id),
            )
            corpus: Mapping[str, list[str]] = {}
            if pieces_dir.is_dir():
                corpus, _ = charger_corpus(pieces_dir)
            reconstituer_unite(
                unite=unite,
                paquet=paquet,
                corpus=corpus,
                sortie_dir=reconstitution,
                dossier=dossier,
                chat_fn=chat,
                compteur=compteur if reserver_emission_unite else None,
                delai_s=delai_s,
                destination=reconstitution / "en-cours",
            )
        reconstruction_ok = True
    except PlafondAtteint as exc:
        statut = "plafond"
        erreur = str(exc)
        enrichir_echecs(reconstitution / "echecs.json", unite_id, repr(exc))
    except Exception as exc:
        statut = "echec"
        erreur = repr(exc)
        enrichir_echecs(reconstitution / "echecs.json", unite_id, repr(exc))
    if reconstruction_ok:
        etape_echec = etape_reprise if etape_reprise in ETAPES_PUBLICATION else "pages"
        etapes_faites: set[str] = set()
        if republier_seulement and isinstance(attente, Mapping):
            etapes_faites = _etapes_faites_de(attente, reconstitution, unite_id)
        try:
            with verrou_publication(reconstitution):
                sauter_vecteurs = bool(
                    etapes_faites
                    & {"vecteurs", "promotion", "paquet", "generation"}
                )
                registre = _lire_registre(reconstitution)
                regimes = regime_par_entite(
                    cfg, registre, ontologie, reconstitution
                )
                registre_cible = _registre_sans_perimee(registre, unite_id)
                echecs_cibles = _echecs_sans_unite(
                    reconstitution / "echecs.json", unite_id
                )
                from banc_realite.recherche_entites import (
                    charger_index,
                    objets_de_recherche,
                )

                pages_generees: list[tuple[str, bytes]] = []
                objets: list[Any] = []
                etape_echec = "pages"
                with _vue_en_cours(
                    reconstitution,
                    unite_id,
                    registre=registre_cible,
                    echecs=echecs_cibles,
                ) as vue:
                    if entite:
                        generer_pages(
                            vue,
                            ontologie,
                            entites=[entite],
                            regime_par_entite=regimes,
                            registre=registre_cible,
                            echecs=echecs_cibles,
                        )
                        pages_src = vue / "pages"
                        if pages_src.is_dir():
                            pages_generees = [
                                (path.name, path.read_bytes())
                                for path in pages_src.glob("*.md")
                            ]
                    if not sauter_vecteurs:
                        objets = [
                            obj
                            for obj in objets_de_recherche(charger_index(vue))
                            if obj.unite_id == unite_id
                        ]
                if not sauter_vecteurs:
                    from banc_realite.embeddings_reconstitution import embarquer_objets

                    etape_echec = "vecteurs"
                    sous_pytest = "pytest" in sys.modules
                    embarquer_objets(
                        dossier,
                        reconstitution_dir=reconstitution,
                        situations_root=cfg.racine,
                        unites=[unite_id],
                        objets=objets,
                        a_blanc=embed_fn is None and sous_pytest,
                        embed_fn=embed_fn,
                    )
                etapes_faites.add("vecteurs")
                if pages_generees:
                    pages_dst = reconstitution / "pages"
                    pages_dst.mkdir(parents=True, exist_ok=True)
                    for nom, contenu in pages_generees:
                        (pages_dst / nom).write_bytes(contenu)
                etape_echec = "promotion"
                if "promotion" not in etapes_faites:
                    if chemin_objet_en_cours(reconstitution, unite_id).is_file():
                        publier_objet_en_cours(reconstitution, unite_id)
                if (
                    chemin_objet_publie(reconstitution, unite_id).is_file()
                    and not chemin_objet_en_cours(reconstitution, unite_id).is_file()
                ):
                    etapes_faites.add("promotion")
                if _chemin_paquet_en_cours(reconstitution, unite_id).is_file():
                    etape_echec = "paquet"
                    _promouvoir_paquet_en_cours(reconstitution, unite_id)
                if not _chemin_paquet_en_cours(reconstitution, unite_id).is_file():
                    etapes_faites.add("paquet")
                if "generation" not in etapes_faites:
                    etape_echec = "generation"
                    incrementer_generation_reconstitution(reconstitution)
                    etapes_faites.add("generation")
                _invalider_etat_api(cfg.nom)
                retrait = retirer_perimee_si_identique(
                    reconstitution, unite_id, depuis=depuis_capture
                )
                retirer_publication_en_attente(reconstitution, unite_id)
                _retirer_reconstitution_en_attente(reconstitution, unite_id)
                retirer_echec(reconstitution / "echecs.json", unite_id)
                statut = (
                    "succes_perime_depuis"
                    if retrait == "perime_depuis"
                    else "succes"
                )
        except Exception as exc:
            erreur = repr(exc)
            statut = "echec_publication"
            marquer_echec_publication(
                reconstitution,
                unite_id,
                erreur,
                etape=etape_echec,
                version_entree=version_entree,
            )
            _persister_etapes_faites(
                reconstitution,
                unite_id,
                etapes_faites,
                familles=familles,
                regime=regime,
                etape=etape_echec,
            )
    duree = round(time.monotonic() - started, 3)
    emis_apres = int(compteur.lire().get("emis") or 0)
    ligne = {
        "horodatage": _horodatage(),
        "unite": unite_id,
        "entite": entite,
        "familles": familles,
        "regime": regime,
        "declencheur": declencheur,
        "duree_murale_s": duree,
        "appels": max(0, emis_apres - emis_avant),
        "statut": statut,
        "erreur": erreur,
    }
    _append_journal(cfg.chemin_journal_reconstitution(), ligne)
    return {
        "unite": unite_id,
        "entite": entite,
        "familles": familles,
        "regime": regime,
        "declencheur": declencheur,
        "duree_murale_s": duree,
        "appels": ligne["appels"],
        "statut": statut,
        "erreur": erreur,
    }


def unites_perimees_continues(
    cfg: TenantConfig,
    registre: Mapping[str, Any],
    ontologie: Mapping[str, Any],
    reconstitution: Path,
) -> list[dict[str, Any]]:
    """Unités marquées ``perimee`` en régime ``continu``, ou à republication seule.

    Une ``publication_en_attente`` dont ``version_entree`` est cohérent avec
    ``perimee.depuis`` reste éligible, avant toute résolution de régime depuis
    le paquet courant (paquet vide et régime ``jamais`` compris). Une
    ``reconstitution_en_attente`` reste éligible quel que soit le régime
    recalculé depuis le paquet.
    """
    cibles: list[dict[str, Any]] = []
    for unite in unites_perimees(registre):
        uid = str(unite.get("id") or "")
        if not uid:
            continue
        if _publication_en_attente_coherente(unite):
            cibles.append(unite)
            continue
        if _reconstitution_en_attente_presente(unite):
            cibles.append(unite)
            continue
        paquet = charger_paquet_unite(reconstitution, uid)
        _familles, regime = familles_et_regime_unite(cfg, paquet, ontologie)
        if regime == "continu":
            cibles.append(unite)
    return cibles


def reconstituer_unites_continues(
    cfg: TenantConfig,
    *,
    registre: Mapping[str, Any] | None = None,
    ontologie: Mapping[str, Any] | None = None,
    a_blanc: bool = False,
    compteur: CompteurAppels | None = None,
    delai_s: int = 900,
    chat: ChatFn | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    declencheur: str = "synchroniser",
) -> dict[str, Any]:
    """Reconstitue séquentiellement les unités périmées en régime ``continu``.

    Un statut ``plafond`` n'interrompt pas la boucle : les cibles suivantes
    en republication seule ou en attente de publication sont encore visitées ;
    un nouvel appel modèle est sauté par ``reconstituer_unite_tenant``.
    """
    reconstitution = cfg.reconstitution
    rec = dict(registre) if registre is not None else (
        _lire_registre(reconstitution) if (reconstitution / "registre.json").is_file() else {}
    )
    onto = dict(ontologie) if ontologie is not None else charger_ontologie_tenant(cfg)
    cibles = unites_perimees_continues(cfg, rec, onto, reconstitution)
    ids = [str(u["id"]) for u in cibles]
    if a_blanc:
        return {
            "unites_a_refaire": ids,
            "unites_refaites": [],
            "unites_echouees": [],
            "duree_reconstitution_s": 0.0,
        }
    if not ids:
        return {
            "unites_a_refaire": [],
            "unites_refaites": [],
            "unites_echouees": [],
            "duree_reconstitution_s": 0.0,
        }
    compteur_eff = compteur if compteur is not None else make_compteur_reconstitution(cfg)
    started = time.monotonic()
    refaites: list[str] = []
    echouees: list[str] = []
    for uid in ids:
        resultat = reconstituer_unite_tenant(
            cfg,
            uid,
            compteur=compteur_eff,
            delai_s=delai_s,
            chat=chat,
            declencheur=declencheur,
            embed_fn=embed_fn,
            ontologie=onto,
        )
        statut = str(resultat.get("statut") or "")
        if statut == "succes":
            refaites.append(uid)
        elif statut == "en_cours":
            continue
        else:
            echouees.append(uid)
    return {
        "unites_a_refaire": [],
        "unites_refaites": refaites,
        "unites_echouees": echouees,
        "duree_reconstitution_s": round(time.monotonic() - started, 3),
    }


def republication_seule(unite: Mapping[str, Any]) -> bool:
    """True si l'unité n'a plus besoin d'appel modèle (publication en attente cohérente)."""
    return _publication_en_attente_coherente(unite)


def _jamais_reconstituee(reconstitution: Path, unite_id: str) -> bool:
    return not chemin_objet_publie(reconstitution, unite_id).is_file()


def unite_encore_candidate_demande(
    reconstitution: Path,
    unite_id: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Relit le registre : candidate si périmée, jamais reconstituée, ou en attente."""
    if not unite_id or not (reconstitution / "registre.json").is_file():
        return False, None
    registre = lire_registre_reconstitution(reconstitution)
    for unite in registre.get("unites") or []:
        if isinstance(unite, Mapping) and str(unite.get("id") or "") == unite_id:
            copie = dict(unite)
            return _candidate_demande(copie, reconstitution), copie
    return False, None


def _candidate_demande(unite: Mapping[str, Any], reconstitution: Path) -> bool:
    """Périmée, jamais reconstituée, ou en attente de publication/reconstitution."""
    uid = str(unite.get("id") or "")
    if not uid:
        return False
    if isinstance(unite.get("perimee"), Mapping):
        return True
    if _jamais_reconstituee(reconstitution, uid):
        return True
    if isinstance(unite.get("publication_en_attente"), Mapping):
        return True
    if isinstance(unite.get("reconstitution_en_attente"), Mapping):
        return True
    return False


def unites_candidates_demande(
    cfg: TenantConfig,
    registre: Mapping[str, Any],
    ontologie: Mapping[str, Any],
    reconstitution: Path,
    entites: Sequence[str],
) -> list[dict[str, Any]]:
    """Unités à reconstituer à la demande pour les entités données.

    Candidats : périmées, jamais reconstituées (sans ``entites/<id>.json``),
    en attente de publication ou de reconstitution. Régime ``jamais`` exclu ;
    ``a_la_demande`` et ``continu`` restent éligibles. Le régime mémorisé sur
    une attente l'emporte sur le paquet courant.
    """
    noms = {str(nom) for nom in entites if nom}
    if not noms:
        return []
    cibles: list[dict[str, Any]] = []
    for unite in registre.get("unites") or []:
        if not isinstance(unite, Mapping):
            continue
        if str(unite.get("entite") or "") not in noms:
            continue
        if not _candidate_demande(unite, reconstitution):
            continue
        _familles, regime = _regime_depuis_attente_ou_paquet(
            cfg, unite, reconstitution, ontologie
        )
        if regime == "jamais":
            continue
        if regime and regime not in REGIMES_DEMANDE:
            continue
        cibles.append(dict(unite))
    return cibles
