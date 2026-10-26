# SPDX-License-Identifier: AGPL-3.0-only
"""Serveur MCP mince : chaque outil appelle l'API Brain v0 du tenant."""
from __future__ import annotations

import hmac
import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # mcp 2.x : FastMCP est devenu MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP

from mcp.types import CallToolResult, TextContent

DELAI_COURT_S = 30.0
DELAI_LONG_S = 120.0
OUTILS_DELAI_LONG = frozenset(
    {"brain_rechercher", "brain_pages_regenerer", "brain_synchroniser"}
)
NOM_VARIABLE_JETON_ENTRANT = "BRAIN_MCP_JETON_ENTRANT"
HOTES_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
_CORPS_401_JETON_ENTRANT = b'{"code":401,"detail":"jeton entrant invalide"}'


def hote_est_loopback(hote: str) -> bool:
    """Vrai si l'hôte HTTP est une interface locale uniquement."""
    return hote in HOTES_LOOPBACK


def lire_jeton_entrant() -> str | None:
    """Lit ``BRAIN_MCP_JETON_ENTRANT`` ; ``None`` si absente ou vide."""
    valeur = os.environ.get(NOM_VARIABLE_JETON_ENTRANT)
    if valeur is None or valeur == "":
        return None
    return valeur


def verifier_demarrage_http(
    hote: str,
    jeton_entrant: str | None | object = ...,
) -> str | None:
    """Retourne un message d'erreur si le bind HTTP est refusé, sinon ``None``.

    Le transport HTTP brut n'écoute que sur loopback (127.0.0.1, ::1, localhost).
    ``BRAIN_MCP_JETON_ENTRANT`` ne permet pas un bind distant : il est vérifié par
    le middleware ``Authorization: Bearer`` lorsque le service est exposé derrière
    une passerelle TLS devant le loopback.
    """
    if hote_est_loopback(hote):
        return None
    return (
        f"refus : transport HTTP brut hors loopback (--hote {hote!r}) ; "
        "exposer derrière une passerelle TLS devant 127.0.0.1"
    )


class MiddlewareJetonEntrant:
    """ASGI : exige ``Authorization: Bearer`` quand le jeton entrant est configuré."""

    def __init__(self, app: Any, jeton_attendu: str) -> None:
        self._app = app
        self._jeton_attendu = jeton_attendu

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        entetes = {
            cle.decode("latin-1").lower(): valeur.decode("latin-1")
            for cle, valeur in scope.get("headers", ())
        }
        auth = entetes.get("authorization", "")
        jeton = auth[7:] if auth.startswith("Bearer ") else ""
        if not jeton or not hmac.compare_digest(jeton, self._jeton_attendu):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [[b"content-type", b"application/json"]],
                }
            )
            await send({"type": "http.response.body", "body": _CORPS_401_JETON_ENTRANT})
            return
        await self._app(scope, receive, send)


async def lancer_http_streamable(
    serveur: FastMCP,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Démarre le transport streamable HTTP, avec middleware jeton entrant si configuré."""
    import uvicorn

    app: Any = serveur.streamable_http_app(host=host)
    jeton_entrant = lire_jeton_entrant()
    if jeton_entrant is not None:
        app = MiddlewareJetonEntrant(app, jeton_entrant)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=serveur.settings.log_level.lower(),
    )
    await uvicorn.Server(config).serve()


def nom_variable_jeton(tenant: str) -> str:
    """Nom d'environnement du jeton, même convention que l'API Brain."""
    maj = re.sub(r"[^a-zA-Z0-9]", "_", tenant).upper()
    return f"BRAIN_TENANT_{maj}_JETON"


def lire_jeton(tenant: str) -> str:
    """Lit ``BRAIN_TENANT_<TENANT_MAJ>_JETON`` ; lève si absent."""
    cle = nom_variable_jeton(tenant)
    valeur = os.environ.get(cle)
    if valeur is None or valeur == "":
        raise RuntimeError(f"jeton absent : {cle}")
    return valeur


def delai_pour(outil: str) -> float:
    """Délai httpx de l'outil : 120 s pour recherche et régénération, 30 s ailleurs."""
    if outil in OUTILS_DELAI_LONG:
        return DELAI_LONG_S
    return DELAI_COURT_S


def _detail_api(reponse: httpx.Response) -> str:
    try:
        payload = reponse.json()
    except ValueError:
        return reponse.text
    if isinstance(payload, dict):
        if "detail" in payload:
            detail = payload["detail"]
            return detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
        if "erreur" in payload:
            return str(payload["erreur"])
    return reponse.text


def _erreur_mcp(code: int, detail: str) -> CallToolResult:
    payload = {"code": code, "detail": detail}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structuredContent=payload,
        isError=True,
    )


def _succes_mcp(donnees: Any) -> CallToolResult:
    if isinstance(donnees, str):
        texte = donnees
        structure: dict[str, Any] | None = None
    else:
        texte = json.dumps(donnees, ensure_ascii=False)
        structure = donnees if isinstance(donnees, dict) else None
    return CallToolResult(
        content=[TextContent(type="text", text=texte)],
        structuredContent=structure,
        isError=False,
    )


def creer_serveur(
    *,
    api_url: str,
    tenant: str,
    jeton: str | None = None,
    client_http: httpx.AsyncClient | None = None,
) -> FastMCP:
    """Construit le serveur MCP branché sur l'API Brain du tenant."""
    if jeton is None:
        jeton = lire_jeton(tenant)
    base = api_url.rstrip("/")
    prefixe = f"/tenants/{tenant}"

    mcp = FastMCP(
        "brain-api",
        instructions=(
            "Serveur MCP mince sur l'API Brain v0 : chaque outil relaie "
            "l'appel HTTP du tenant, avec son jeton, sans logique métier."
        ),
    )

    @asynccontextmanager
    async def _session(timeout: float) -> AsyncIterator[httpx.AsyncClient]:
        if client_http is not None:
            yield client_http
            return
        async with httpx.AsyncClient(base_url=base, timeout=timeout) as http:
            yield http

    async def _appeler(
        methode: str,
        chemin: str,
        *,
        outil: str,
        corps: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        markdown: bool = False,
    ) -> CallToolResult:
        timeout = delai_pour(outil)
        try:
            async with _session(timeout) as http:
                reponse = await http.request(
                    methode,
                    chemin,
                    json=corps,
                    params=params,
                    headers={"Authorization": f"Bearer {jeton}"},
                    timeout=timeout,
                )
        except httpx.HTTPError as exc:
            return _erreur_mcp(503, str(exc))
        if reponse.status_code >= 400:
            return _erreur_mcp(reponse.status_code, _detail_api(reponse))
        if markdown:
            try:
                payload = reponse.json()
            except ValueError:
                return _succes_mcp(reponse.text)
            if isinstance(payload, dict) and "markdown" in payload:
                return _succes_mcp(str(payload["markdown"]))
            return _succes_mcp(payload)
        try:
            return _succes_mcp(reponse.json())
        except ValueError:
            return _succes_mcp(reponse.text)

    @mcp.tool(structured_output=False)
    async def brain_sante() -> CallToolResult:
        """État de l'API Brain : ok et liste des tenants chargés."""
        return await _appeler("GET", "/sante", outil="brain_sante")

    @mcp.tool(structured_output=False)
    async def brain_ontologie() -> CallToolResult:
        """Ontologie du tenant : entités, relations, concepts et version."""
        return await _appeler("GET", f"{prefixe}/ontologie", outil="brain_ontologie")

    @mcp.tool(structured_output=False)
    async def brain_ontologie_questions() -> CallToolResult:
        """Questions ouvertes et décidées de l'ontologie du tenant."""
        return await _appeler(
            "GET",
            f"{prefixe}/ontologie/questions",
            outil="brain_ontologie_questions",
        )

    @mcp.tool(structured_output=False)
    async def brain_ontologie_decider(
        question_id: str,
        texte: str,
        decideur: str,
    ) -> CallToolResult:
        """Enregistre une décision d'ontologie (question, texte, décideur)."""
        return await _appeler(
            "POST",
            f"{prefixe}/ontologie/decisions",
            outil="brain_ontologie_decider",
            corps={
                "question_id": question_id,
                "texte": texte,
                "decideur": decideur,
            },
        )

    @mcp.tool(structured_output=False)
    async def brain_entites() -> CallToolResult:
        """Index des entités actives du tenant (slug, nom, type, compteurs)."""
        return await _appeler("GET", f"{prefixe}/entites", outil="brain_entites")

    @mcp.tool(structured_output=False)
    async def brain_page(
        slug: str,
        format: Literal["json", "markdown"] = "markdown",
    ) -> CallToolResult:
        """Page d'une entité : JSON reconstitué ou Markdown."""
        return await _appeler(
            "GET",
            f"{prefixe}/entites/{slug}",
            outil="brain_page",
            params={"format": format},
            markdown=(format == "markdown"),
        )

    @mcp.tool(structured_output=False)
    async def brain_pages_regenerer() -> CallToolResult:
        """Régénère toutes les pages Markdown des entités du tenant."""
        return await _appeler(
            "POST",
            f"{prefixe}/pages/regenerer",
            outil="brain_pages_regenerer",
            corps={},
        )

    @mcp.tool(structured_output=False)
    async def brain_rechercher(question: str, budget: int | None = None) -> CallToolResult:
        """Recherche un contexte composite cité pièce et ligne, pas une réponse rédigée."""
        corps: dict[str, Any] = {"question": question}
        if budget is not None:
            corps["budget"] = budget
        return await _appeler(
            "POST",
            f"{prefixe}/rechercher",
            outil="brain_rechercher",
            corps=corps,
        )

    @mcp.tool(structured_output=False)
    async def brain_recharger() -> CallToolResult:
        """Vide le cache mémoire du tenant ; la prochaine requête le recharge."""
        return await _appeler("POST", f"{prefixe}/recharger", outil="brain_recharger")

    @mcp.tool(structured_output=False)
    async def brain_reconstitution() -> CallToolResult:
        """Journal de reconstitution (50 dernières lignes) et file du tenant."""
        return await _appeler(
            "GET",
            f"{prefixe}/reconstitution",
            outil="brain_reconstitution",
        )

    @mcp.tool(structured_output=False)
    async def brain_synchroniser(a_blanc: bool = False) -> CallToolResult:
        """Lance une passe de synchronisation du tenant (fond, ou à blanc synchrone)."""
        return await _appeler(
            "POST",
            f"{prefixe}/synchroniser",
            outil="brain_synchroniser",
            corps={"a_blanc": a_blanc},
        )

    @mcp.tool(structured_output=False)
    async def brain_synchronisation() -> CallToolResult:
        """Journal de synchronisation (50 dernières lignes), attente et état."""
        return await _appeler(
            "GET",
            f"{prefixe}/synchronisation",
            outil="brain_synchronisation",
        )

    return mcp
