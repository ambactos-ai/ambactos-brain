# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne du serveur MCP mince — API remplacée par MockTransport."""
from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts", "src"):
    entry = str(_ROOT / sub)
    if entry not in sys.path:
        sys.path.insert(0, entry)

try:
    import httpx
    from mcp import Client

    from brain.mcp_api import creer_serveur, lire_jeton
    from brain.mcp_api.serveur import (
        DELAI_COURT_S,
        DELAI_LONG_S,
        MiddlewareJetonEntrant,
        _erreur_mcp,
        delai_pour,
        hote_est_loopback,
        lire_jeton_entrant,
        nom_variable_jeton,
        verifier_demarrage_http,
    )

    _MCP = True
except ImportError:
    _MCP = False

pytestmark = pytest.mark.skipif(not _MCP, reason="mcp ou httpx absent")

NOMS_OUTILS = [
    "brain_sante",
    "brain_ontologie",
    "brain_ontologie_questions",
    "brain_ontologie_decider",
    "brain_entites",
    "brain_page",
    "brain_pages_regenerer",
    "brain_rechercher",
    "brain_recharger",
    "brain_reconstitution",
    "brain_synchroniser",
    "brain_synchronisation",
]


def _texte(resultat: Any) -> str:
    return "".join(getattr(bloc, "text", "") for bloc in (resultat.content or []))


def _json(resultat: Any) -> Any:
    return json.loads(_texte(resultat))


def _handler(requetes: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requetes.append(request)
        path = request.url.path
        methode = request.method
        auth = request.headers.get("authorization")
        if path != "/sante" and auth != "Bearer secret-demo":
            return httpx.Response(401, json={"erreur": "jeton invalide"})
        if methode == "GET" and path == "/sante":
            return httpx.Response(200, json={"ok": True, "tenants": ["demo"]})
        if methode == "GET" and path == "/tenants/demo/ontologie":
            return httpx.Response(200, json={"version": 2, "entites": [{"nom": "Alpha SA"}]})
        if methode == "GET" and path == "/tenants/demo/ontologie/questions":
            return httpx.Response(200, json={"questions": [{"id": "Q-0001"}]})
        if methode == "POST" and path == "/tenants/demo/ontologie/decisions":
            corps = json.loads(request.content)
            return httpx.Response(200, json={"decisions": 1, "recu": corps})
        if methode == "GET" and path == "/tenants/demo/entites":
            return httpx.Response(
                200,
                json=[{"slug": "client-alpha-sa", "nom": "Alpha SA", "type": "client"}],
            )
        if methode == "GET" and path.startswith("/tenants/demo/entites/"):
            slug = path.rsplit("/", 1)[-1]
            if slug == "absente":
                return httpx.Response(404, json={"erreur": "entité absente"})
            fmt = request.url.params.get("format")
            if fmt == "json":
                return httpx.Response(
                    200,
                    json={"slug": slug, "nom": "Alpha SA", "unites": []},
                )
            return httpx.Response(
                200,
                json={"slug": slug, "nom": "Alpha SA", "markdown": "# Alpha SA\n"},
            )
        if methode == "POST" and path == "/tenants/demo/pages/regenerer":
            corps = json.loads(request.content or b"{}")
            return httpx.Response(200, json={"pages": 2, "corps": corps})
        if methode == "POST" and path == "/tenants/demo/rechercher":
            corps = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "question": corps["question"],
                    "passages": [{"piece": "contrat.md", "ligne": 12}],
                },
            )
        if methode == "POST" and path == "/tenants/demo/recharger":
            return httpx.Response(200, json={"ok": True, "tenant": "demo"})
        if methode == "GET" and path == "/tenants/demo/reconstitution":
            return httpx.Response(
                200,
                json={"journal": [{"unite": "u1"}], "en_cours": ["u1"], "en_attente": []},
            )
        if methode == "POST" and path == "/tenants/demo/synchroniser":
            corps = json.loads(request.content or b"{}")
            if corps.get("a_blanc"):
                return httpx.Response(
                    200,
                    json={"horodatage": "h", "passages_remplaces": 0, "a_blanc": True},
                )
            return httpx.Response(
                200,
                json={"statut": "lancee", "passe": "2026-01-01T00:00:00+00:00"},
            )
        if methode == "GET" and path == "/tenants/demo/synchronisation":
            return httpx.Response(
                200,
                json={
                    "journal": [{"declencheur": "api"}],
                    "attente": {},
                    "en_cours": False,
                    "derniere_passe": "h",
                },
            )
        return httpx.Response(404, json={"erreur": "inconnue"})

    return handler


@pytest.fixture
def banc() -> Iterator[tuple[Any, list[httpx.Request]]]:
    requetes: list[httpx.Request] = []
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(requetes)),
        base_url="http://brain.test",
    )
    serveur = creer_serveur(
        api_url="http://brain.test",
        tenant="demo",
        jeton="secret-demo",
        client_http=http,
    )
    yield serveur, requetes
    asyncio.run(http.aclose())


def _lister(serveur: Any) -> list[Any]:
    async def _run() -> list[Any]:
        async with Client(serveur) as client:
            resultat = await client.list_tools()
            return list(resultat.tools)

    return asyncio.run(_run())


def _appeler(serveur: Any, nom: str, arguments: dict[str, Any] | None = None) -> Any:
    async def _run() -> Any:
        async with Client(serveur) as client:
            return await client.call_tool(nom, arguments or {})

    return asyncio.run(_run())


def test_tools_list_en_memoire(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, _ = banc
    outils = _lister(serveur)
    noms = [outil.name for outil in outils]
    assert noms == NOMS_OUTILS
    descriptions = {outil.name: outil.description or "" for outil in outils}
    assert "contexte composite cité pièce et ligne, pas une réponse rédigée" in descriptions[
        "brain_rechercher"
    ]


def test_brain_sante(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, requetes = banc
    resultat = _appeler(serveur, "brain_sante")
    assert resultat.is_error is False
    assert _json(resultat) == {"ok": True, "tenants": ["demo"]}
    assert requetes[-1].method == "GET"
    assert requetes[-1].url.path == "/sante"


def test_brain_ontologie_et_questions(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, requetes = banc
    ontologie = _appeler(serveur, "brain_ontologie")
    assert _json(ontologie)["version"] == 2
    assert requetes[-1].headers["authorization"] == "Bearer secret-demo"
    questions = _appeler(serveur, "brain_ontologie_questions")
    assert _json(questions)["questions"][0]["id"] == "Q-0001"


def test_brain_ontologie_decider(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, requetes = banc
    resultat = _appeler(
        serveur,
        "brain_ontologie_decider",
        {"question_id": "Q-0001", "texte": "note", "decideur": "humain"},
    )
    assert resultat.is_error is False
    recu = json.loads(requetes[-1].content)
    assert recu == {"question_id": "Q-0001", "texte": "note", "decideur": "humain"}
    assert _json(resultat)["decisions"] == 1


def test_brain_entites(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, _ = banc
    resultat = _appeler(serveur, "brain_entites")
    assert _json(resultat)[0]["slug"] == "client-alpha-sa"


def test_brain_page_json_et_markdown(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, requetes = banc
    markdown = _appeler(
        serveur, "brain_page", {"slug": "client-alpha-sa", "format": "markdown"}
    )
    assert markdown.is_error is False
    assert _texte(markdown) == "# Alpha SA\n"
    assert requetes[-1].url.params.get("format") == "markdown"
    js = _appeler(serveur, "brain_page", {"slug": "client-alpha-sa", "format": "json"})
    assert _json(js)["unites"] == []


def test_brain_pages_regenerer_corps_vide(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, requetes = banc
    resultat = _appeler(serveur, "brain_pages_regenerer")
    assert _json(resultat)["pages"] == 2
    assert json.loads(requetes[-1].content) == {}


def test_brain_rechercher(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, requetes = banc
    resultat = _appeler(
        serveur, "brain_rechercher", {"question": "qui est Alpha ?", "budget": 8}
    )
    assert _json(resultat)["passages"][0]["ligne"] == 12
    corps = json.loads(requetes[-1].content)
    assert corps == {"question": "qui est Alpha ?", "budget": 8}


def test_brain_recharger(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, requetes = banc
    resultat = _appeler(serveur, "brain_recharger")
    assert _json(resultat) == {"ok": True, "tenant": "demo"}
    assert requetes[-1].method == "POST"
    assert requetes[-1].url.path == "/tenants/demo/recharger"


def test_brain_reconstitution(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, requetes = banc
    resultat = _appeler(serveur, "brain_reconstitution")
    assert resultat.is_error is False
    assert _json(resultat)["journal"][0]["unite"] == "u1"
    assert requetes[-1].method == "GET"
    assert requetes[-1].url.path == "/tenants/demo/reconstitution"


def test_brain_synchroniser(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, requetes = banc
    resultat = _appeler(serveur, "brain_synchroniser")
    assert resultat.is_error is False
    assert _json(resultat)["statut"] == "lancee"
    assert requetes[-1].method == "POST"
    assert requetes[-1].url.path == "/tenants/demo/synchroniser"
    assert json.loads(requetes[-1].content) == {"a_blanc": False}
    blanc = _appeler(serveur, "brain_synchroniser", {"a_blanc": True})
    assert _json(blanc)["a_blanc"] is True
    assert json.loads(requetes[-1].content) == {"a_blanc": True}


def test_brain_synchronisation(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, requetes = banc
    resultat = _appeler(serveur, "brain_synchronisation")
    assert resultat.is_error is False
    assert _json(resultat)["journal"][0]["declencheur"] == "api"
    assert requetes[-1].method == "GET"
    assert requetes[-1].url.path == "/tenants/demo/synchronisation"


def test_erreur_http_devient_resultat_mcp(banc: tuple[Any, list[httpx.Request]]) -> None:
    serveur, _ = banc
    resultat = _appeler(
        serveur, "brain_page", {"slug": "absente", "format": "markdown"}
    )
    assert resultat.is_error is True
    payload = _json(resultat)
    assert payload["code"] == 404
    assert payload["detail"] == "entité absente"


def test_erreur_mcp_champs_protocole() -> None:
    resultat = _erreur_mcp(404, "entité absente")
    protocole = resultat.model_dump(by_alias=True)
    assert protocole["isError"] is True
    assert protocole["structuredContent"] == {"code": 404, "detail": "entité absente"}


def test_erreur_reseau_attrapee() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connexion refusée")

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://brain.test",
    )
    serveur = creer_serveur(
        api_url="http://brain.test",
        tenant="demo",
        jeton="secret-demo",
        client_http=http,
    )
    try:
        resultat = _appeler(serveur, "brain_sante")
    finally:
        asyncio.run(http.aclose())
    assert resultat.is_error is True
    payload = _json(resultat)
    assert payload["code"] == 503
    assert "connexion refusée" in payload["detail"]


def test_jeton_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAIN_TENANT_DEMO_JETON", raising=False)
    with pytest.raises(RuntimeError, match="jeton absent : BRAIN_TENANT_DEMO_JETON"):
        lire_jeton("demo")
    monkeypatch.setenv("BRAIN_TENANT_ALPHA_CORP_JETON", "x")
    assert nom_variable_jeton("alpha-corp") == "BRAIN_TENANT_ALPHA_CORP_JETON"
    assert lire_jeton("alpha-corp") == "x"


def test_delais() -> None:
    assert delai_pour("brain_rechercher") == DELAI_LONG_S
    assert delai_pour("brain_pages_regenerer") == DELAI_LONG_S
    assert delai_pour("brain_sante") == DELAI_COURT_S
    assert DELAI_LONG_S == 120.0
    assert DELAI_COURT_S == 30.0


def test_bind_hors_loopback_sans_jeton_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    from banc_realite.run_mcp_api import main

    monkeypatch.delenv("BRAIN_MCP_JETON_ENTRANT", raising=False)
    monkeypatch.setenv("BRAIN_TENANT_DEMO_JETON", "secret-demo")
    code = main(
        [
            "--tenant",
            "demo",
            "--transport",
            "http",
            "--hote",
            "0.0.0.0",
            "--port",
            "1",
        ]
    )
    assert code == 2


def test_bind_hors_loopback_avec_jeton_refuse() -> None:
    message = verifier_demarrage_http("0.0.0.0", "jeton-entrant")
    assert message is not None
    assert "transport HTTP brut hors loopback" in message
    assert hote_est_loopback("127.0.0.1")
    assert not hote_est_loopback("0.0.0.0")


def test_middleware_jeton_entrant() -> None:
    async def app_amont(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = MiddlewareJetonEntrant(app_amont, "bon-jeton")

    async def _requete(headers: dict[str, str] | None = None) -> httpx.Response:
        entetes = []
        if headers:
            for cle, valeur in headers.items():
                entetes.append((cle.lower().encode(), valeur.encode()))
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": entetes,
        }
        corps = b""
        statut = 500

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            nonlocal corps, statut
            if message["type"] == "http.response.start":
                statut = message["status"]
            elif message["type"] == "http.response.body":
                corps += message.get("body", b"")

        await middleware(scope, receive, send)
        return httpx.Response(statut, content=corps)

    sans_auth = asyncio.run(_requete())
    assert sans_auth.status_code == 401
    assert sans_auth.json() == {"code": 401, "detail": "jeton entrant invalide"}

    mauvais = asyncio.run(_requete({"Authorization": "Bearer mauvais"}))
    assert mauvais.status_code == 401

    bon = asyncio.run(_requete({"Authorization": "Bearer bon-jeton"}))
    assert bon.status_code == 200
    assert bon.text == "ok"


def test_lire_jeton_entrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAIN_MCP_JETON_ENTRANT", raising=False)
    assert lire_jeton_entrant() is None
    monkeypatch.setenv("BRAIN_MCP_JETON_ENTRANT", "")
    assert lire_jeton_entrant() is None
    monkeypatch.setenv("BRAIN_MCP_JETON_ENTRANT", "entrant")
    assert lire_jeton_entrant() == "entrant"


def test_lanceur_help() -> None:
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "src:scripts"
    proc = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "banc_realite" / "run_mcp_api.py"),
            "--help",
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--tenant" in proc.stdout
    assert "--transport" in proc.stdout
    assert "--api-url" in proc.stdout
    assert "stdio" in proc.stdout
    assert "loopback" in proc.stdout
