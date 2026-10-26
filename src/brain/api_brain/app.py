# SPDX-License-Identifier: AGPL-3.0-only
"""Application FastAPI Brain v0."""
from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from banc_realite.fournisseurs_contexte import ObjetsSensPuisRag
from banc_realite.pages_entites import (
    _entite_slug,
    _stats_page,
    charger_reconstitution,
    generer_pages,
    regrouper_par_entite,
)
from banc_realite.etiquetage_ontologie import (
    _entites_actives_index,
    entites_filtre_depuis_question,
)

from .ontologie import (
    ajouter_decision,
    charger_ontologie,
    vue_ontologie,
    vue_questions,
)
from banc_realite.plafond_appels import PlafondAtteint

from .recherche import (
    BUDGET_DEFAUT,
    ReconstitueurFn,
    SynchroniseurFn,
    _assurer_fournisseur,
    acquerir_etat_tenant,
    arreter_files_reconstitution,
    compteur_questions_pour_tenant,
    declencher_reconstitution_demande,
    declencher_synchronisation,
    embed_question,
    etat_reconstitution_tenant,
    etat_synchronisation_tenant,
    message_erreur_plafond_questions,
    obtenir_connection_tenant,
    plafond_embeddings_questions_depuis_cfg,
    prechauffer_tenant,
    rechercher_contexte,
    vider_etat_tenant,
)
from .tenants import (
    CollisionNomsTenantsError,
    DossierBancInconnuError,
    TenantConfig,
    TenantIncompletError,
    charger_registre_tenants,
    dossier_de_tenant,
    jeton_configure,
    resoudre_tenant,
    verifier_jeton,
)

BANC_DSN_DEFAUT = "postgresql://postgres@127.0.0.1:56102/postgres"

EmbedderFn = Callable[[list[str]], list[list[float]]]
FournisseurFabrique = Callable[[TenantConfig, dict[str, list[float]]], ObjetsSensPuisRag]
ConnectionFactory = Callable[[str], Any]


class DecisionBody(BaseModel):
    question_id: str | None = None
    texte: str
    effet: str | None = None
    decideur: str


class RegenererBody(BaseModel):
    date: str | None = None


class RechercherBody(BaseModel):
    question: str
    budget: int | None = Field(default=None, ge=1)


class SynchroniserBody(BaseModel):
    a_blanc: bool = False


def _erreur(code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"erreur": message})


def _extraire_bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization") or ""
    prefix = "Bearer "
    if auth.startswith(prefix):
        return auth[len(prefix) :].strip()
    return None


def _index_entites(tenant: TenantConfig) -> list[dict[str, Any]]:
    ontologie = charger_ontologie(tenant.chemin_ontologie_effectif())
    actives = _entites_actives_index(ontologie)
    charge = charger_reconstitution(tenant.chemin_reconstitution_effectif())
    groupes = regrouper_par_entite(charge["registre"], charge["objets"], charge["echecs"])
    entrees: list[dict[str, Any]] = []
    for nom in sorted(actives, key=lambda n: (actives[n].get("type") or "", n)):
        meta = actives[nom]
        typ = str(meta.get("type") or "autre")
        groupe = groupes.get(nom) or {"unites": [], "objets": {}}
        n_unites = len(groupe.get("unites") or [])
        n_faits, n_des, n_non = _stats_page(groupe)
        slug = _entite_slug({"nom": nom, "type": typ, "cles": meta.get("cles") or []})
        entrees.append(
            {
                "slug": slug,
                "nom": nom,
                "type": typ,
                "unites": n_unites,
                "faits": n_faits,
                "desaccords": n_des,
                "non_etabli": n_non,
            }
        )
    return entrees


def _champ_reconstitution(
    cfg: TenantConfig,
    entites: list[str],
    request: Request,
) -> dict[str, Any]:
    return declencher_reconstitution_demande(
        cfg,
        entites,
        reconstitueur=getattr(request.app.state, "reconstitueur", None),
        force_desactivee=bool(
            getattr(request.app.state, "reconstitution_force_off", False)
        ),
    )


@asynccontextmanager
async def _duree_vie(_app: FastAPI) -> AsyncIterator[None]:
    yield
    arreter_files_reconstitution()


def creer_app(
    *,
    embedder: EmbedderFn | None = None,
    fournisseur_fabrique: FournisseurFabrique | None = None,
    connection_factory: ConnectionFactory | None = None,
    reconstitueur: ReconstitueurFn | None = None,
    reconstitution_a_la_demande: bool | None = None,
    synchroniseur: SynchroniseurFn | None = None,
) -> FastAPI:
    app = FastAPI(title="Brain API v0", lifespan=_duree_vie)
    app.state.reconstitueur = reconstitueur
    app.state.reconstitution_force_off = reconstitution_a_la_demande is False
    app.state.synchroniseur = synchroniseur

    def _depend_auth(tenant_nom: str, request: Request) -> TenantConfig:
        try:
            cfg = resoudre_tenant(tenant_nom)
        except (CollisionNomsTenantsError, DossierBancInconnuError, TenantIncompletError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if cfg is None:
            raise HTTPException(status_code=404, detail="tenant inconnu")
        if not jeton_configure(tenant_nom):
            raise HTTPException(status_code=503, detail="jeton non configuré")
        token = _extraire_bearer(request)
        if not verifier_jeton(tenant_nom, token):
            raise HTTPException(status_code=401, detail="jeton invalide")
        return cfg

    @app.exception_handler(HTTPException)
    def _http_exc(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        message = detail if isinstance(detail, str) else str(detail)
        return _erreur(exc.status_code, message)

    @app.get("/sante")
    def sante() -> dict[str, Any]:
        try:
            registre = charger_registre_tenants()
        except (CollisionNomsTenantsError, DossierBancInconnuError, TenantIncompletError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"ok": True, "tenants": sorted(registre)}

    @app.get("/tenants/{tenant}/ontologie")
    def get_ontologie_route(tenant: str, request: Request) -> dict[str, Any]:
        cfg = _depend_auth(tenant, request)
        ontologie = charger_ontologie(cfg.chemin_ontologie_effectif())
        return vue_ontologie(ontologie)

    @app.get("/tenants/{tenant}/ontologie/questions")
    def get_questions(tenant: str, request: Request) -> dict[str, Any]:
        cfg = _depend_auth(tenant, request)
        ontologie = charger_ontologie(cfg.chemin_ontologie_effectif())
        return vue_questions(ontologie)

    @app.post("/tenants/{tenant}/ontologie/decisions")
    def post_decision(tenant: str, body: DecisionBody, request: Request) -> dict[str, Any]:
        cfg = _depend_auth(tenant, request)
        if not body.texte.strip():
            raise HTTPException(status_code=400, detail="texte requis")
        if not body.decideur.strip():
            raise HTTPException(status_code=400, detail="decideur requis")
        return ajouter_decision(
            cfg.chemin_ontologie_effectif(),
            question_id=body.question_id,
            texte=body.texte,
            effet=body.effet,
            decideur=body.decideur,
        )

    @app.get("/tenants/{tenant}/entites")
    def get_entites(tenant: str, request: Request) -> list[dict[str, Any]]:
        cfg = _depend_auth(tenant, request)
        return _index_entites(cfg)

    @app.get("/tenants/{tenant}/entites/{slug}")
    def get_entite(
        tenant: str,
        slug: str,
        request: Request,
        format: str | None = None,
    ) -> dict[str, Any]:
        cfg = _depend_auth(tenant, request)
        ontologie = charger_ontologie(cfg.chemin_ontologie_effectif())
        actives = _entites_actives_index(ontologie)
        charge = charger_reconstitution(cfg.chemin_reconstitution_effectif())
        groupes = regrouper_par_entite(charge["registre"], charge["objets"], charge["echecs"])
        for nom, meta in actives.items():
            typ = str(meta.get("type") or "autre")
            ent_slug = _entite_slug(
                {"nom": nom, "type": typ, "cles": meta.get("cles") or []}
            )
            if ent_slug != slug:
                continue
            groupe = groupes.get(nom) or {"unites": [], "objets": {}, "echecs": {}}
            reconstitution = _champ_reconstitution(cfg, [nom], request)
            if format == "json":
                return {
                    "slug": slug,
                    "nom": nom,
                    **groupe,
                    "reconstitution": reconstitution,
                }
            pages_dir = cfg.chemin_reconstitution_effectif() / "pages"
            page_path = pages_dir / f"{slug}.md"
            if page_path.is_file():
                markdown = page_path.read_text(encoding="utf-8")
            else:
                from banc_realite.pages_entites import rendre_page

                relations = [
                    r for r in (ontologie.get("relations") or []) if isinstance(r, dict)
                ]
                entite = {"nom": nom, "type": typ, "cles": meta.get("cles") or []}
                markdown = rendre_page(
                    entite,
                    groupe,
                    relations,
                    ontologie.get("version", 1),
                    date.today().isoformat(),
                )
            return {
                "slug": slug,
                "nom": nom,
                "markdown": markdown,
                "reconstitution": reconstitution,
            }
        raise HTTPException(status_code=404, detail="entité absente")

    @app.post("/tenants/{tenant}/pages/regenerer")
    def post_regenerer(tenant: str, body: RegenererBody, request: Request) -> dict[str, Any]:
        cfg = _depend_auth(tenant, request)
        ontologie = charger_ontologie(cfg.chemin_ontologie_effectif())
        return generer_pages(
            cfg.chemin_reconstitution_effectif(),
            ontologie,
            date_gen=body.date,
        )

    @app.post("/tenants/{tenant}/rechercher")
    def post_rechercher(tenant: str, body: RechercherBody, request: Request) -> dict[str, Any]:
        cfg = _depend_auth(tenant, request)
        texte = body.question.strip()
        if not texte:
            raise HTTPException(status_code=400, detail="question requise")
        budget = body.budget if body.budget is not None else BUDGET_DEFAUT
        plafond_embeddings = plafond_embeddings_questions_depuis_cfg(cfg.plafonds)
        etat = acquerir_etat_tenant(cfg)
        try:
            try:
                cache, _ = embed_question(
                    texte,
                    racine=cfg.racine,
                    compteur_dossier=cfg.vault_ref,
                    plafond=plafond_embeddings,
                    embedder=embedder,
                    compteur=compteur_questions_pour_tenant(
                        cfg.racine, etat, plafond=plafond_embeddings
                    ),
                )
            except PlafondAtteint as exc:
                return _erreur(
                    503,
                    message_erreur_plafond_questions(exc, plafond_cfg=plafond_embeddings),
                )
            dsn = cfg.dsn_effectif(os.environ.get("BRAIN_BANC_DSN", BANC_DSN_DEFAUT))
            if connection_factory is not None:
                ouvrir = connection_factory
            else:
                import psycopg
                from psycopg.rows import dict_row

                ouvrir = lambda d: psycopg.connect(d, autocommit=True, row_factory=dict_row)
            fournisseur = _assurer_fournisseur(cfg, etat, cache, fournisseur_fabrique)
            connection = obtenir_connection_tenant(etat, dsn, ouvrir)
            payload = rechercher_contexte(
                question_texte=texte,
                tenant_racine=cfg.racine,
                compteur_dossier=cfg.vault_ref,
                dossier=dossier_de_tenant(cfg),
                reconstitution_dir=cfg.chemin_reconstitution_effectif(),
                connection=connection,
                embed_cache=etat.embed_cache,
                fournisseur=fournisseur,
                budget=budget,
                plafond_embeddings=plafond_embeddings,
            )
            ontologie = charger_ontologie(cfg.chemin_ontologie_effectif())
            directes, _relations, _filtre, _repli = entites_filtre_depuis_question(
                texte, ontologie
            )
            payload["reconstitution"] = _champ_reconstitution(cfg, directes, request)
            return payload
        finally:
            etat.relacher()

    @app.get("/tenants/{tenant}/reconstitution")
    def get_reconstitution(tenant: str, request: Request) -> dict[str, Any]:
        cfg = _depend_auth(tenant, request)
        return etat_reconstitution_tenant(cfg)

    @app.post("/tenants/{tenant}/synchroniser")
    def post_synchroniser(
        tenant: str,
        request: Request,
        body: SynchroniserBody = Body(default_factory=SynchroniserBody),
    ) -> dict[str, Any]:
        cfg = _depend_auth(tenant, request)
        a_blanc = bool(body.a_blanc)
        return declencher_synchronisation(
            cfg,
            a_blanc=a_blanc,
            synchroniseur=getattr(request.app.state, "synchroniseur", None),
        )

    @app.get("/tenants/{tenant}/synchronisation")
    def get_synchronisation(tenant: str, request: Request) -> dict[str, Any]:
        cfg = _depend_auth(tenant, request)
        return etat_synchronisation_tenant(cfg)

    @app.post("/tenants/{tenant}/recharger")
    def post_recharger(tenant: str, request: Request) -> dict[str, Any]:
        _depend_auth(tenant, request)
        vider_etat_tenant(tenant)
        return {"ok": True, "tenant": tenant}

    return app


def prechauffer_tenants(
    noms: list[str],
    *,
    embedder: EmbedderFn | None = None,
    fournisseur_fabrique: FournisseurFabrique | None = None,
) -> None:
    for nom in noms:
        cfg = resoudre_tenant(nom)
        if cfg is None:
            raise ValueError(f"tenant inconnu : {nom}")
        prechauffer_tenant(cfg, fabrique=fournisseur_fabrique)
