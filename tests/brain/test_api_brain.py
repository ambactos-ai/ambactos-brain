# SPDX-License-Identifier: AGPL-3.0-only
"""Tests API Brain v0 — tenant minimal hors ligne."""
from __future__ import annotations

import fcntl
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts", "src"):
    entry = str(_ROOT / sub)
    if entry not in sys.path:
        sys.path.insert(0, entry)

try:
    from fastapi.testclient import TestClient

    _FASTAPI = True
except ImportError:
    _FASTAPI = False

pytestmark = pytest.mark.skipif(not _FASTAPI, reason="FastAPI absent")


def _ecrire_tenant(tmp_path: Path) -> Path:
    racine = tmp_path / "tenant-demo"
    racine.mkdir()
    reconst = racine / "reconstitution"
    reconst.mkdir()
    ontologie_dir = racine / "ontologie"
    ontologie_dir.mkdir()
    ontologie = {
        "version": 2,
        "decisions": [],
        "entites": [
            {
                "nom": "Alpha SA",
                "type": "client",
                "cles": ["Alpha SA"],
                "variantes": [],
            },
            {
                "nom": "Beta Corp",
                "type": "fournisseur",
                "cles": ["Beta Corp"],
                "variantes": [],
            },
            {
                "nom": "Gamma Old",
                "type": "autre",
                "cles": ["Gamma Old"],
                "variantes": [],
            },
        ],
        "relations": [],
        "concepts": [],
        "questions": [
            {
                "id": "Q-0001",
                "sujet": "test",
                "question": "Faut-il fusionner Gamma ?",
                "candidats": [],
                "preuve": [],
                "pourquoi": "test",
            }
        ],
        "questions_en_reserve": [],
    }
    (ontologie_dir / "ontologie.v1.json").write_text(
        json.dumps(ontologie, ensure_ascii=False), encoding="utf-8"
    )
    registre = {
        "unites": [
            {
                "id": "client-alpha-sa",
                "entite": "Alpha SA",
                "type": "client",
                "titre": "Alpha",
                "cles": ["Alpha SA"],
                "variantes": [],
                "periode": [],
            },
            {
                "id": "fournisseur-beta-corp",
                "entite": "Beta Corp",
                "type": "fournisseur",
                "titre": "Beta",
                "cles": ["Beta Corp"],
                "variantes": [],
                "periode": [],
            },
        ],
    }
    (reconst / "registre.json").write_text(json.dumps(registre), encoding="utf-8")
    (reconst / "echecs.json").write_text("[]", encoding="utf-8")
    (reconst / "reemploi.json").write_text("{}", encoding="utf-8")
    entites = reconst / "entites"
    entites.mkdir()
    for uid, nom in (
        ("client-alpha-sa", "Alpha SA"),
        ("fournisseur-beta-corp", "Beta Corp"),
    ):
        obj = {
            "entite": {"nom": nom, "type": "client" if "alpha" in uid else "fournisseur"},
            "faits": [],
            "regles": [],
            "non_etabli": [],
            "unite": uid,
        }
        (entites / f"{uid}.json").write_text(json.dumps(obj), encoding="utf-8")
    return racine


def _config_env(
    monkeypatch: pytest.MonkeyPatch,
    racine: Path,
    *,
    reconstitution_appels: int | None = None,
    regimes: dict[str, str] | None = None,
    reconstitution_a_la_demande: bool | None = None,
    depot_brut: Path | None = None,
    embeddings_passages: int | None = None,
) -> None:
    plafonds: dict[str, int] = {"embeddings_questions": 10_000}
    if reconstitution_appels is not None:
        plafonds["reconstitution_appels"] = reconstitution_appels
    if embeddings_passages is not None:
        plafonds["embeddings_passages"] = embeddings_passages
    bloc: dict[str, object] = {
        "racine": str(racine),
        "reconstitution": str(racine / "reconstitution"),
        "ontologie": str(racine / "ontologie" / "ontologie.v1.json"),
        "dossier_banc": "greenloop",
        "plafonds": plafonds,
    }
    if regimes is not None:
        bloc["regimes"] = regimes
    if reconstitution_a_la_demande is not None:
        bloc["reconstitution_a_la_demande"] = reconstitution_a_la_demande
    if depot_brut is not None:
        bloc["depot_brut"] = str(depot_brut)
    payload = {"demo": bloc}
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    monkeypatch.setenv("BRAIN_TENANT_DEMO_JETON", "secret-demo")


@pytest.fixture
def client_demo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    racine = _ecrire_tenant(tmp_path)
    _config_env(monkeypatch, racine)
    from brain.api_brain import creer_app

    return TestClient(creer_app()), racine


def test_jeton_401_503(client_demo, monkeypatch: pytest.MonkeyPatch):
    client, _ = client_demo
    assert client.get("/tenants/demo/ontologie").status_code == 401
    monkeypatch.delenv("BRAIN_TENANT_DEMO_JETON", raising=False)
    assert client.get(
        "/tenants/demo/ontologie",
        headers={"Authorization": "Bearer secret-demo"},
    ).status_code == 503


def test_ontologie_fusion(client_demo):
    client, racine = client_demo
    headers = {"Authorization": "Bearer secret-demo"}
    avant = client.get("/tenants/demo/ontologie", headers=headers).json()
    noms_avant = {e["nom"] for e in avant["entites"]}
    assert "Gamma Old" in noms_avant

    client.post(
        "/tenants/demo/ontologie/decisions",
        headers=headers,
        json={
            "question_id": "Q-0001",
            "texte": "fusion acceptée",
            "effet": "fusion Gamma Old dans Beta Corp",
            "decideur": "test",
        },
    )
    apres = client.get("/tenants/demo/ontologie", headers=headers).json()
    noms_apres = {e["nom"] for e in apres["entites"]}
    assert "Gamma Old" not in noms_apres
    assert "Beta Corp" in noms_apres


def test_decision_atomique_et_question_decidee(client_demo):
    client, racine = client_demo
    headers = {"Authorization": "Bearer secret-demo"}
    path = racine / "ontologie" / "ontologie.v1.json"
    avant = path.read_text(encoding="utf-8")

    rep = client.post(
        "/tenants/demo/ontologie/decisions",
        headers=headers,
        json={
            "question_id": "Q-0001",
            "texte": "note",
            "decideur": "humain",
        },
    )
    assert rep.status_code == 200
    assert rep.json()["decisions"] == 1
    apres = json.loads(path.read_text(encoding="utf-8"))
    assert len(apres["decisions"]) == 1
    assert apres["decisions"][0]["decideur"] == "humain"
    assert avant != path.read_text(encoding="utf-8")

    questions = client.get("/tenants/demo/ontologie/questions", headers=headers).json()
    q1 = next(q for q in questions["questions"] if q["id"] == "Q-0001")
    assert q1["statut"] == "decidee"


def test_entites_et_slug(client_demo):
    client, _ = client_demo
    headers = {"Authorization": "Bearer secret-demo"}
    index = client.get("/tenants/demo/entites", headers=headers).json()
    assert isinstance(index, list) and len(index) >= 2
    slug = next(e["slug"] for e in index if e["nom"] == "Alpha SA")
    page = client.get(f"/tenants/demo/entites/{slug}", headers=headers)
    assert page.status_code == 200
    assert "markdown" in page.json()
    assert client.get("/tenants/demo/entites/inexistant-zzz", headers=headers).status_code == 404


def test_regenerer_deterministe(client_demo):
    client, _ = client_demo
    headers = {"Authorization": "Bearer secret-demo"}
    r1 = client.post("/tenants/demo/pages/regenerer", headers=headers, json={"date": "2026-01-01"})
    r2 = client.post("/tenants/demo/pages/regenerer", headers=headers, json={"date": "2026-01-01"})
    assert r1.status_code == 200
    assert r1.json() == r2.json()


def test_rechercher_injecte(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from brain.api_brain.recherche import vider_etat_tenant

    vider_etat_tenant("demo")
    racine = _ecrire_tenant(tmp_path)
    _config_env(monkeypatch, racine)

    def faux_embedder(textes: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in textes]

    mock_provider = MagicMock()
    mock_provider.contexte.return_value = [{"text": "ctx", "piece": "p.md", "lignes": [1]}]
    mock_provider.stats_par_question = {}
    mock_provider.stats = {}

    def fabrique(cfg, cache):
        return mock_provider

    connexion = MagicMock()
    from brain.api_brain import creer_app

    app = creer_app(
        embedder=faux_embedder,
        fournisseur_fabrique=fabrique,
        connection_factory=lambda _dsn: connexion,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret-demo"}
    rep = client.post(
        "/tenants/demo/rechercher",
        headers=headers,
        json={"question": "Quelle est la date ?"},
    )
    assert rep.status_code == 200
    body = rep.json()
    assert body["contexte"][0]["text"] == "ctx"
    assert body["caracteres"] == 3
    mock_provider.contexte.assert_called_once()
    connexion.close.assert_not_called()


def test_rechercher_recharge_si_generation_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain.recherche import vider_etat_tenant

    vider_etat_tenant("demo")
    racine = _ecrire_tenant(tmp_path)
    _config_env(monkeypatch, racine)
    appels = {"n": 0}

    def faux_embedder(textes: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in textes]

    mock_provider = MagicMock()
    mock_provider.contexte.return_value = [{"text": "ctx", "piece": "p.md", "lignes": [1]}]
    mock_provider.stats_par_question = {}
    mock_provider.stats = {}

    def fabrique(cfg, cache):
        appels["n"] += 1
        return mock_provider

    from brain.api_brain import creer_app

    client = TestClient(
        creer_app(embedder=faux_embedder, fournisseur_fabrique=fabrique)
    )
    headers = {"Authorization": "Bearer secret-demo"}
    payload = {"question": "Quelle est la date ?"}
    assert client.post("/tenants/demo/rechercher", headers=headers, json=payload).status_code == 200
    gen_path = racine / "reconstitution" / "generation.json"
    gen_path.write_text(
        json.dumps({"generation": 2, "horodatage": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    import time

    time.sleep(2.1)
    assert client.post("/tenants/demo/rechercher", headers=headers, json=payload).status_code == 200
    assert appels["n"] == 2


def test_rechercher_etat_reutilise_fabrique(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from brain.api_brain.recherche import vider_etat_tenant

    vider_etat_tenant("demo")
    racine = _ecrire_tenant(tmp_path)
    _config_env(monkeypatch, racine)
    appels = {"n": 0}

    def faux_embedder(textes: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in textes]

    mock_provider = MagicMock()
    mock_provider.contexte.return_value = [{"text": "ctx", "piece": "p.md", "lignes": [1]}]
    mock_provider.stats_par_question = {}
    mock_provider.stats = {}

    def fabrique(cfg, cache):
        appels["n"] += 1
        return mock_provider

    connexion = MagicMock()
    connexion.closed = False
    from brain.api_brain import creer_app

    client = TestClient(
        creer_app(
            embedder=faux_embedder,
            fournisseur_fabrique=fabrique,
            connection_factory=lambda _dsn: connexion,
        )
    )
    headers = {"Authorization": "Bearer secret-demo"}
    payload = {"question": "Quelle est la date ?"}
    assert client.post("/tenants/demo/rechercher", headers=headers, json=payload).status_code == 200
    assert client.post("/tenants/demo/rechercher", headers=headers, json=payload).status_code == 200
    assert appels["n"] == 1
    assert mock_provider.contexte.call_count == 2


def test_recharger_vide_etat(client_demo):
    client, _ = client_demo
    headers = {"Authorization": "Bearer secret-demo"}
    assert client.post("/tenants/demo/recharger", headers=headers).json()["ok"] is True


def test_recharger_401_sans_jeton(client_demo):
    client, _ = client_demo
    assert client.post("/tenants/demo/recharger").status_code == 401


def test_collision_noms_tenants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brain.api_brain.tenants import CollisionNomsTenantsError, charger_registre_tenants

    racine_a = tmp_path / "a"
    racine_b = tmp_path / "b"
    racine_a.mkdir()
    racine_b.mkdir()
    payload = {
        "acme-prod": {
            "racine": str(racine_a),
            "reconstitution": str(racine_a),
            "ontologie": str(racine_a / "o.json"),
            "dossier_banc": "greenloop",
        },
        "acme_prod": {
            "racine": str(racine_b),
            "reconstitution": str(racine_b),
            "ontologie": str(racine_b / "o.json"),
            "dossier_banc": "greenloop",
        },
    }
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    with pytest.raises(CollisionNomsTenantsError):
        charger_registre_tenants()


def test_tenant_unique_charge_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brain.api_brain.tenants import charger_registre_tenants

    racine = tmp_path / "solo"
    racine.mkdir()
    payload = {
        "acme-prod": {
            "racine": str(racine),
            "reconstitution": str(racine),
            "ontologie": str(racine / "o.json"),
            "dossier_banc": "greenloop",
        },
    }
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    assert len(charger_registre_tenants()) == 1


def test_dossier_banc_inconnu_refuse_rechercher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    racine = _ecrire_tenant(tmp_path)
    payload = {
        "demo": {
            "racine": str(racine),
            "reconstitution": str(racine / "reconstitution"),
            "ontologie": str(racine / "ontologie" / "ontologie.v1.json"),
            "dossier_banc": "inexistant-zzz",
        }
    }
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    monkeypatch.setenv("BRAIN_TENANT_DEMO_JETON", "secret-demo")
    fabrique_appelee = False

    def fabrique(_cfg, _cache):
        nonlocal fabrique_appelee
        fabrique_appelee = True
        return MagicMock()

    from brain.api_brain import creer_app

    client = TestClient(creer_app(fournisseur_fabrique=fabrique))
    rep = client.post(
        "/tenants/demo/rechercher",
        headers={"Authorization": "Bearer secret-demo"},
        json={"question": "test"},
    )
    assert rep.status_code == 503
    assert "dossier_banc" in rep.json()["erreur"]
    assert fabrique_appelee is False


def test_fabrique_defaut_passe_dossier_vault_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from unittest.mock import patch

    from banc_realite.dossiers import DOSSIERS
    from brain.api_brain.recherche import EtatTenant, _fabrique_defaut
    from brain.api_brain.tenants import charger_registre_tenants

    racine = _ecrire_tenant(tmp_path)
    _config_env(monkeypatch, racine)
    cfg = charger_registre_tenants()["demo"]
    etat = EtatTenant()
    dossier_capture: list = []

    def faux_fournisseur(*_args, **kwargs):
        dossier_capture.append(kwargs["dossier"])
        return MagicMock()

    with (
        patch("brain.api_brain.recherche.charger_embeddings_objets", return_value={}),
        patch("brain.api_brain.recherche.charger_index", return_value={}),
        patch("brain.api_brain.recherche.objets_de_recherche", return_value=[]),
        patch("brain.api_brain.recherche.matrice_depuis_cache", return_value=([], [])),
        patch("brain.api_brain.recherche.preparer_objets", return_value=MagicMock()),
        patch("brain.api_brain.recherche._FournisseurTenant", side_effect=faux_fournisseur),
    ):
        _fabrique_defaut(cfg, etat)

    assert len(dossier_capture) == 1
    attendu = DOSSIERS["greenloop"]
    assert dossier_capture[0].vault_ref == attendu.vault_ref
    assert dossier_capture[0].corpus_dir == attendu.corpus_dir


def test_decisions_concurrentes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    racine = _ecrire_tenant(tmp_path)
    _config_env(monkeypatch, racine)
    from brain.api_brain import creer_app

    client = TestClient(creer_app())
    headers = {"Authorization": "Bearer secret-demo"}

    def poster(i: int) -> int:
        rep = client.post(
            "/tenants/demo/ontologie/decisions",
            headers=headers,
            json={"texte": f"d{i}", "decideur": f"thread-{i % 8}"},
        )
        assert rep.status_code == 200
        return rep.json()["decisions"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(poster, i) for i in range(20)]
        compteurs = sorted(f.result() for f in as_completed(futs))
    assert compteurs == list(range(1, 21))
    path = racine / "ontologie" / "ontologie.v1.json"
    decisions = json.loads(path.read_text(encoding="utf-8"))["decisions"]
    assert len(decisions) == 20


def test_tronquer_contexte_respecte_budget() -> None:
    from brain.api_brain.recherche import tronquer_contexte

    elements = [{"text": "x" * 500, "piece": "p.md"}]
    ctx, car = tronquer_contexte(elements, budget=100)
    assert car == 100
    assert len(ctx[0]["text"]) == 100
    assert ctx[0].get("tronque") is True


def test_rechercher_budget_premier_element(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brain.api_brain.recherche import vider_etat_tenant

    vider_etat_tenant("demo")
    racine = _ecrire_tenant(tmp_path)
    _config_env(monkeypatch, racine)

    mock_provider = MagicMock()
    mock_provider.contexte.return_value = [{"text": "y" * 500, "piece": "p.md", "lignes": [1]}]
    mock_provider.stats_par_question = {}
    mock_provider.stats = {}

    from brain.api_brain import creer_app

    client = TestClient(
        creer_app(
            embedder=lambda textes: [[0.1, 0.2] for _ in textes],
            fournisseur_fabrique=lambda _cfg, _cache: mock_provider,
            connection_factory=lambda _dsn: MagicMock(closed=False),
        )
    )
    headers = {"Authorization": "Bearer secret-demo"}
    rep = client.post(
        "/tenants/demo/rechercher",
        headers=headers,
        json={"question": "test budget", "budget": 100},
    )
    assert rep.status_code == 200
    body = rep.json()
    assert body["caracteres"] == 100
    assert len(body["contexte"][0]["text"]) == 100


def test_embed_question_cache_concurrent_threads(tmp_path: Path) -> None:
    from brain.api_brain.recherche import (
        _charger_cache_disque,
        chemin_cache_questions,
        embed_question,
    )

    racine = tmp_path / "tenant"
    racine.mkdir()

    def embedder(textes: list[str]) -> list[list[float]]:
        return [[float(len(texte))] for texte in textes]

    questions = [f"question distincte numero {index}" for index in range(20)]
    erreurs: list[BaseException] = []

    def worker(question: str) -> None:
        try:
            embed_question(
                question,
                racine=racine,
                compteur_dossier="moreau",
                plafond=10_000,
                embedder=embedder,
            )
        except BaseException as exc:
            erreurs.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, q) for q in questions]
        for fut in as_completed(futures):
            fut.result()

    assert erreurs == []
    cache = _charger_cache_disque(chemin_cache_questions(racine))
    assert len(cache) == 20


def _tenant_complet_sans_banc(racine: Path) -> dict:
    pieces = racine / "pieces"
    pieces.mkdir(exist_ok=True)
    return {
        "racine": str(racine),
        "reconstitution": str(racine / "reconstitution"),
        "ontologie": str(racine / "ontologie" / "ontologie.v1.json"),
        "pieces": str(pieces),
        "vault_ref": "tenant-saas",
    }


def test_tenant_complet_sans_dossier_banc_passe_dossier_fabrique(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from unittest.mock import patch

    from brain.api_brain.recherche import EtatTenant, _fabrique_defaut
    from brain.api_brain.tenants import charger_registre_tenants

    racine = _ecrire_tenant(tmp_path)
    payload = {"demo": _tenant_complet_sans_banc(racine)}
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    cfg = charger_registre_tenants()["demo"]
    etat = EtatTenant()
    dossier_capture: list = []

    def faux_fournisseur(*_args, **kwargs):
        dossier_capture.append(kwargs["dossier"])
        return MagicMock()

    with (
        patch("brain.api_brain.recherche.charger_embeddings_objets", return_value={}),
        patch("brain.api_brain.recherche.charger_index", return_value={}),
        patch("brain.api_brain.recherche.objets_de_recherche", return_value=[]),
        patch("brain.api_brain.recherche.matrice_depuis_cache", return_value=([], [])),
        patch("brain.api_brain.recherche.preparer_objets", return_value=MagicMock()),
        patch("brain.api_brain.recherche._FournisseurTenant", side_effect=faux_fournisseur),
    ):
        _fabrique_defaut(cfg, etat)

    assert dossier_capture[0].vault_ref == "tenant-saas"
    assert dossier_capture[0].corpus_dir == racine / "pieces"


def test_tenant_dossier_banc_seul_compat_greenloop(monkeypatch: pytest.MonkeyPatch) -> None:
    from banc_realite.dossiers import DOSSIERS
    from brain.api_brain.tenants import charger_registre_tenants

    payload = {"gl": {"dossier_banc": "greenloop"}}
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    cfg = charger_registre_tenants()["gl"]
    ref = DOSSIERS["greenloop"]
    assert cfg.vault_ref == ref.vault_ref
    assert cfg.pieces == ref.corpus_dir


def test_chemin_cache_reference_greenloop(monkeypatch: pytest.MonkeyPatch) -> None:
    from banc_realite.dossiers import BANC_VAR, DOSSIERS
    from banc_realite.mesurer_api_brain import chemin_cache_reference
    from brain.api_brain.tenants import charger_registre_tenants

    monkeypatch.setenv("BRAIN_TENANTS", json.dumps({"gl": {"dossier_banc": "greenloop"}}))
    cfg = charger_registre_tenants()["gl"]
    chemin = chemin_cache_reference(cfg, None)
    attendu = DOSSIERS["greenloop"].cache_embeddings_questions
    assert chemin == attendu
    assert chemin == BANC_VAR / "sorties" / "etage2-question-embeddings.json"




def test_tenant_incomplet_sans_dossier_banc_503(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    racine = tmp_path / "incomplet"
    racine.mkdir()
    payload = {
        "demo": {
            "racine": str(racine),
            "reconstitution": str(racine),
            "ontologie": str(racine / "o.json"),
        }
    }
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    monkeypatch.setenv("BRAIN_TENANT_DEMO_JETON", "secret-demo")
    from brain.api_brain import creer_app

    client = TestClient(creer_app())
    rep = client.post(
        "/tenants/demo/rechercher",
        headers={"Authorization": "Bearer secret-demo"},
        json={"question": "test"},
    )
    assert rep.status_code == 503


def test_chemins_relatifs_resolus_contre_racine(tmp_path: Path) -> None:
    from brain.api_brain.tenants import charger_registre_tenants

    racine = tmp_path / "rac"
    racine.mkdir()
    (racine / "reconstitution").mkdir()
    (racine / "pieces").mkdir()
    payload = {
        "t": {
            "racine": str(racine),
            "reconstitution": "reconstitution",
            "ontologie": "ontologie/ontologie.v1.json",
            "pieces": "pieces",
            "vault_ref": "v",
        }
    }
    import os

    os.environ["BRAIN_TENANTS"] = json.dumps(payload)
    try:
        cfg = charger_registre_tenants()["t"]
    finally:
        del os.environ["BRAIN_TENANTS"]
    assert cfg.reconstitution == (racine / "reconstitution").resolve()
    assert cfg.pieces == (racine / "pieces").resolve()


def test_plafond_embeddings_questions_atteint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from banc_realite.plafond_appels import PlafondAtteint
    from brain.api_brain.recherche import vider_etat_tenant

    vider_etat_tenant("demo")
    racine = _ecrire_tenant(tmp_path)
    payload = {
        "demo": {
            "racine": str(racine),
            "reconstitution": str(racine / "reconstitution"),
            "ontologie": str(racine / "ontologie" / "ontologie.v1.json"),
            "dossier_banc": "greenloop",
            "plafonds": {"embeddings_questions": 2},
        }
    }
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    monkeypatch.setenv("BRAIN_TENANT_DEMO_JETON", "secret-demo")

    appels = 0

    def faux_embedder(textes: list[str]) -> list[list[float]]:
        nonlocal appels
        appels += len(textes)
        return [[0.1, 0.2, 0.3] for _ in textes]

    mock_provider = MagicMock()
    mock_provider.contexte.return_value = [{"text": "ctx", "piece": "p.md", "lignes": [1]}]
    mock_provider.stats_par_question = {}
    mock_provider.stats = {}

    from brain.api_brain import creer_app

    client = TestClient(
        creer_app(
            embedder=faux_embedder,
            fournisseur_fabrique=lambda _cfg, _cache: mock_provider,
            connection_factory=lambda _dsn: MagicMock(closed=False),
        )
    )
    headers = {"Authorization": "Bearer secret-demo"}

    for i in range(2):
        rep = client.post(
            "/tenants/demo/rechercher",
            headers=headers,
            json={"question": f"question distincte {i}"},
        )
        assert rep.status_code == 200

    rep = client.post(
        "/tenants/demo/rechercher",
        headers=headers,
        json={"question": "question distincte 2"},
    )
    assert rep.status_code == 503
    assert rep.json()["erreur"] == str(PlafondAtteint(2, 2))
    assert appels == 2


def test_plafond_embeddings_questions_non_decide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain.recherche import ERREUR_PLAFOND_QUESTIONS_NON_DECIDE, vider_etat_tenant

    vider_etat_tenant("demo")
    racine = _ecrire_tenant(tmp_path)
    payload = {
        "demo": {
            "racine": str(racine),
            "reconstitution": str(racine / "reconstitution"),
            "ontologie": str(racine / "ontologie" / "ontologie.v1.json"),
            "dossier_banc": "greenloop",
        }
    }
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    monkeypatch.setenv("BRAIN_TENANT_DEMO_JETON", "secret-demo")

    from brain.api_brain import creer_app

    client = TestClient(
        creer_app(
            embedder=lambda textes: [[0.1] for _ in textes],
            fournisseur_fabrique=lambda _cfg, _cache: MagicMock(),
            connection_factory=lambda _dsn: MagicMock(closed=False),
        )
    )
    rep = client.post(
        "/tenants/demo/rechercher",
        headers={"Authorization": "Bearer secret-demo"},
        json={"question": "première question"},
    )
    assert rep.status_code == 503
    assert ERREUR_PLAFOND_QUESTIONS_NON_DECIDE in rep.json()["erreur"]


def test_compteur_existant_realigne_plafond_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain.recherche import make_compteur_questions, vider_etat_tenant

    vider_etat_tenant("demo")
    racine = _ecrire_tenant(tmp_path)
    (racine / "compteur-embeddings.json").write_text(
        json.dumps({"plafond": 10_000, "emis": 0}), encoding="utf-8"
    )
    payload = {
        "demo": {
            "racine": str(racine),
            "reconstitution": str(racine / "reconstitution"),
            "ontologie": str(racine / "ontologie" / "ontologie.v1.json"),
            "dossier_banc": "greenloop",
            "plafonds": {"embeddings_questions": 390},
        }
    }
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    monkeypatch.setenv("BRAIN_TENANT_DEMO_JETON", "secret-demo")

    make_compteur_questions(racine, plafond=390)
    payload_compteur = json.loads(
        (racine / "compteur-embeddings.json").read_text(encoding="utf-8")
    )
    assert payload_compteur["plafond"] == 390
    assert payload_compteur["emis"] == 0


def test_tenants_fichier_charge_registre(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain.tenants import charger_registre_tenants, definir_tenants_fichier

    racine = tmp_path / "fichier"
    racine.mkdir()
    (racine / "reconstitution").mkdir()
    (racine / "pieces").mkdir()
    fichier = tmp_path / "tenants.json"
    fichier.write_text(
        json.dumps(
            {
                "depuis-fichier": {
                    "racine": str(racine),
                    "reconstitution": "reconstitution",
                    "pieces": "pieces",
                    "vault_ref": "vf",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("BRAIN_TENANTS", raising=False)
    definir_tenants_fichier(fichier)
    cfg = charger_registre_tenants()["depuis-fichier"]
    assert cfg.vault_ref == "vf"
    definir_tenants_fichier(None)


def test_regimes_charges_et_defaut(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brain.api_brain.tenants import charger_registre_tenants

    racine = tmp_path / "r"
    racine.mkdir()
    (racine / "reconstitution").mkdir()
    (racine / "pieces").mkdir()
    payload = {
        "t": {
            "racine": str(racine),
            "reconstitution": "reconstitution",
            "ontologie": "ontologie/ontologie.v1.json",
            "pieces": "pieces",
            "vault_ref": "v",
            "regimes": {"juridique": "jamais", "travaux": "continu"},
        }
    }
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    cfg = charger_registre_tenants()["t"]
    assert cfg.regime_pour("juridique") == "jamais"
    assert cfg.regime_pour("travaux") == "continu"
    assert cfg.regime_pour("absente") == "a_la_demande"
    assert cfg.regime_pour(None) == "a_la_demande"


def test_regimes_valeur_inconnue_refuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain.tenants import RegimeInconnuError, charger_registre_tenants

    racine = tmp_path / "r"
    racine.mkdir()
    (racine / "reconstitution").mkdir()
    (racine / "pieces").mkdir()
    payload = {
        "t": {
            "racine": str(racine),
            "reconstitution": "reconstitution",
            "pieces": "pieces",
            "vault_ref": "v",
            "regimes": {"juridique": "parfois"},
        }
    }
    monkeypatch.setenv("BRAIN_TENANTS", json.dumps(payload))
    with pytest.raises(RegimeInconnuError, match="parfois"):
        charger_registre_tenants()


def _objet_avec_fait(nom: str, typ: str, uid: str, quoi: str) -> dict:
    return {
        "entite": {"nom": nom, "type": typ},
        "faits": [
            {
                "quoi": quoi,
                "comparaison": "une_seule_source",
                "versions": [
                    {
                        "piece": "a.md",
                        "valeur": quoi,
                        "statut": "declare",
                        "lignes": [1],
                        "lettre": quoi,
                    }
                ],
            }
        ],
        "regles": [],
        "non_etabli": [],
        "identites": [],
        "unite": uid,
    }


def test_objet_en_cours_non_servi_par_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain.recherche import vider_etat_tenant

    vider_etat_tenant("demo")
    racine = _ecrire_tenant(tmp_path)
    reconst = racine / "reconstitution"
    en_cours = reconst / "en-cours"
    en_cours.mkdir()
    (en_cours / "client-alpha-sa.json").write_text(
        json.dumps(
            _objet_avec_fait(
                "Alpha SA", "client", "client-alpha-sa", "secret-en-cours-ne-pas-servir"
            )
        ),
        encoding="utf-8",
    )
    _config_env(monkeypatch, racine)
    from brain.api_brain import creer_app

    client = TestClient(creer_app())
    headers = {"Authorization": "Bearer secret-demo"}
    index = client.get("/tenants/demo/entites", headers=headers).json()
    slug = next(e["slug"] for e in index if e["nom"] == "Alpha SA")
    page = client.get(
        f"/tenants/demo/entites/{slug}",
        headers=headers,
        params={"format": "json"},
    )
    assert page.status_code == 200
    blob = json.dumps(page.json(), ensure_ascii=False)
    assert "secret-en-cours-ne-pas-servir" not in blob


def test_api_demarre_sans_vecteur_objet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from banc_realite.recherche_entites import charger_index, objets_de_recherche
    from brain.api_brain.recherche import EtatTenant, _fabrique_defaut, vider_etat_tenant
    from brain.api_brain.tenants import charger_registre_tenants

    vider_etat_tenant("demo")
    racine = _ecrire_tenant(tmp_path)
    reconst = racine / "reconstitution"
    (reconst / "entites" / "client-alpha-sa.json").write_text(
        json.dumps(
            _objet_avec_fait("Alpha SA", "client", "client-alpha-sa", "fait-alpha-vectorise")
        ),
        encoding="utf-8",
    )
    (reconst / "entites" / "fournisseur-beta-corp.json").write_text(
        json.dumps(
            _objet_avec_fait(
                "Beta Corp", "fournisseur", "fournisseur-beta-corp", "fait-beta-sans-vecteur"
            )
        ),
        encoding="utf-8",
    )
    index = charger_index(reconst)
    cache: dict[str, list[float]] = {}
    for obj in objets_de_recherche(index):
        if obj.unite_id == "client-alpha-sa":
            cache[obj.empreinte] = [0.1, 0.2, 0.3, 0.4]
    (reconst / "embeddings-objets.json").write_text(
        json.dumps(cache), encoding="utf-8"
    )
    _config_env(monkeypatch, racine)
    cfg = charger_registre_tenants()["demo"]
    caplog.set_level(logging.WARNING)
    etat = EtatTenant()
    fournisseur = _fabrique_defaut(cfg, etat)
    ordre = etat.matrice_objets[1] if etat.matrice_objets is not None else []
    assert all(emp in cache for emp in ordre)
    assert any("empreinte absente" in rec.message for rec in caplog.records)
    ids_index = {obj.id for obj in objets_de_recherche(fournisseur._index)}
    assert "fournisseur-beta-corp#f1" not in ids_index

    def fabrique(_cfg, cache):
        fournisseur._embed_cache = cache
        fournisseur._rag = MagicMock()
        fournisseur._rag._vivier.return_value = []
        fournisseur._rag.stats_par_question = {}
        fournisseur._rag._enregistrer_stats = MagicMock()
        return fournisseur

    from brain.api_brain import creer_app

    client = TestClient(
        creer_app(
            embedder=lambda textes: [[0.1, 0.2, 0.3, 0.4] for _ in textes],
            fournisseur_fabrique=fabrique,
            connection_factory=lambda _dsn: MagicMock(closed=False),
        )
    )
    rep = client.post(
        "/tenants/demo/rechercher",
        headers={"Authorization": "Bearer secret-demo"},
        json={"question": "fait-beta-sans-vecteur"},
    )
    assert rep.status_code == 200
    contexte = json.dumps(rep.json().get("contexte") or [], ensure_ascii=False)
    assert "fait-beta-sans-vecteur" not in contexte
    assert "fournisseur-beta-corp" not in contexte


def _objet_sans_fait(nom: str, typ: str, uid: str) -> dict:
    return {
        "entite": {"nom": nom, "type": typ},
        "faits": [],
        "regles": [],
        "non_etabli": [],
        "identites": [],
        "unite": uid,
    }


def test_fabrique_defaut_instantane_sous_verrou_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging
    from contextlib import contextmanager

    from banc_realite.recherche_entites import (
        charger_index as charger_index_reel,
        objets_de_recherche,
    )
    from banc_realite.reconstitution_entites import (
        verrou_publication as verrou_publication_reel,
    )
    from brain.api_brain import recherche as mod_recherche
    from brain.api_brain.recherche import EtatTenant, _fabrique_defaut, vider_etat_tenant
    from brain.api_brain.tenants import charger_registre_tenants

    vider_etat_tenant("demo")
    racine = _ecrire_tenant(tmp_path)
    reconst = racine / "reconstitution"
    payload_alpha = _objet_avec_fait(
        "Alpha SA", "client", "client-alpha-sa", "fait-alpha-vectorise"
    )
    payload_beta = _objet_avec_fait(
        "Beta Corp", "fournisseur", "fournisseur-beta-corp", "fait-beta-publie"
    )
    (reconst / "entites" / "client-alpha-sa.json").write_text(
        json.dumps(payload_alpha), encoding="utf-8"
    )
    (reconst / "entites" / "fournisseur-beta-corp.json").write_text(
        json.dumps(payload_beta), encoding="utf-8"
    )
    index_complet = charger_index_reel(reconst)
    cache: dict[str, list[float]] = {}
    for obj in objets_de_recherche(index_complet):
        if obj.unite_id == "client-alpha-sa":
            cache[obj.empreinte] = [0.1, 0.2, 0.3, 0.4]
        elif obj.unite_id == "fournisseur-beta-corp":
            cache[obj.empreinte] = [0.4, 0.3, 0.2, 0.1]
    (reconst / "embeddings-objets.json").write_text(
        json.dumps(cache), encoding="utf-8"
    )
    (reconst / "entites" / "fournisseur-beta-corp.json").write_text(
        json.dumps(
            _objet_sans_fait("Beta Corp", "fournisseur", "fournisseur-beta-corp")
        ),
        encoding="utf-8",
    )

    lectures_index = {"n": 0}

    def charger_index_compte(repertoire):
        lectures_index["n"] += 1
        return charger_index_reel(repertoire)

    matrice_origine = mod_recherche.matrice_depuis_cache

    def matrice_puis_publier(vecteurs, empreintes):
        resultat = matrice_origine(vecteurs, empreintes)
        (reconst / "entites" / "fournisseur-beta-corp.json").write_text(
            json.dumps(payload_beta), encoding="utf-8"
        )
        return resultat

    prises_verrou: list[Path] = []

    @contextmanager
    def verrou_compte(repertoire):
        prises_verrou.append(Path(repertoire))
        with verrou_publication_reel(repertoire):
            yield

    monkeypatch.setattr(mod_recherche, "charger_index", charger_index_compte)
    monkeypatch.setattr(mod_recherche, "matrice_depuis_cache", matrice_puis_publier)
    monkeypatch.setattr(mod_recherche, "verrou_publication", verrou_compte)
    _config_env(monkeypatch, racine)
    cfg = charger_registre_tenants()["demo"]
    caplog.set_level(logging.WARNING)
    etat = EtatTenant()
    fournisseur = _fabrique_defaut(cfg, etat)

    assert prises_verrou == [reconst]
    assert lectures_index["n"] == 1
    ids_index = {obj.id for obj in objets_de_recherche(fournisseur._index)}
    assert "fournisseur-beta-corp#f1" not in ids_index
    assert any(ident.startswith("client-alpha-sa#") for ident in ids_index)
    ordre = etat.matrice_objets[1] if etat.matrice_objets is not None else []
    assert all(emp in cache for emp in ordre)

    hors_matrice = [
        obj
        for obj in objets_de_recherche(charger_index_reel(reconst))
        if obj.unite_id == "fournisseur-beta-corp"
    ]
    assert hors_matrice
    fournisseur._index._objets_cache = list(fournisseur._objets) + hors_matrice
    fournisseur._embed_cache = {"fait-alpha-vectorise": [0.1, 0.2, 0.3, 0.4]}
    selection = fournisseur._selection(
        {"id": "q-matrice", "question": "fait-alpha-vectorise"}
    )
    assert selection is not None
    assert any("empreinte absente de la matrice" in rec.message for rec in caplog.records)
    ids_apres = {obj.id for obj in objets_de_recherche(fournisseur._index)}
    assert "fournisseur-beta-corp#f1" not in ids_apres


class _ConnexionDouble:
    def __init__(self, nom: str) -> None:
        self.nom = nom
        self.closed = False
        self.n_close = 0

    def close(self) -> None:
        self.n_close += 1
        self.closed = True


def test_rechargement_preserve_instantane_en_cours() -> None:
    from brain.api_brain.recherche import (
        obtenir_connection_tenant,
        obtenir_etat_tenant,
        vider_etat_tenant,
    )

    vider_etat_tenant("demo")
    ouverts: list[_ConnexionDouble] = []

    def ouvrir(dsn: str) -> _ConnexionDouble:
        conn = _ConnexionDouble(dsn)
        ouverts.append(conn)
        return conn

    etat_a = obtenir_etat_tenant("demo")
    etat_a.acquerir()
    try:
        conn_a = obtenir_connection_tenant(etat_a, "dsn-a", ouvrir)
        vider_etat_tenant("demo")
        assert conn_a.closed is False
        assert conn_a.n_close == 0

        etat_b = obtenir_etat_tenant("demo")
        etat_b.acquerir()
        try:
            assert etat_b is not etat_a
            conn_b = obtenir_connection_tenant(etat_b, "dsn-b", ouvrir)
            assert conn_b is not conn_a
            assert conn_b.closed is False
            etat_a.relacher()
            etat_a = None
            assert conn_a.n_close == 1
            assert conn_a.closed is True
            assert conn_b.closed is False
        finally:
            etat_b.relacher()
        assert conn_b.closed is False
    finally:
        if etat_a is not None:
            etat_a.relacher()
        vider_etat_tenant("demo")


_HEADERS_DEMO = {"Authorization": "Bearer secret-demo"}


def _marquer_perimee(racine: Path, uid: str) -> None:
    path = racine / "reconstitution" / "registre.json"
    registre = json.loads(path.read_text(encoding="utf-8"))
    for unite in registre.get("unites") or []:
        if unite.get("id") == uid:
            unite["perimee"] = {
                "depuis": "2026-01-01T00:00:00+00:00",
                "pieces": ["x.md"],
                "passe": "p1",
            }
    path.write_text(json.dumps(registre), encoding="utf-8")


def _poser_paquet_juridique(racine: Path, uid: str = "client-alpha-sa") -> None:
    paquets = racine / "reconstitution" / "paquets"
    paquets.mkdir(exist_ok=True)
    (paquets / f"{uid}.json").write_text(
        json.dumps(
            {"sections": {"document": {"dossier/JURIDIQUE/kbis.md": {"1": "x"}}}}
        ),
        encoding="utf-8",
    )
    onto_path = racine / "ontologie" / "ontologie.v1.json"
    onto = json.loads(onto_path.read_text(encoding="utf-8"))
    onto["familles"] = [
        {
            "nom": "juridique",
            "regle": {"chemin": "JURIDIQUE"},
            "traitement": "lecture_simple",
        }
    ]
    onto_path.write_text(json.dumps(onto, ensure_ascii=False), encoding="utf-8")


def _slug_alpha(client: TestClient) -> str:
    index = client.get("/tenants/demo/entites", headers=_HEADERS_DEMO).json()
    return next(e["slug"] for e in index if e["nom"] == "Alpha SA")


def _app_recherche(reconstitueur=None, **kwargs):
    from brain.api_brain import creer_app

    def faux_embedder(textes: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in textes]

    mock_provider = MagicMock()
    mock_provider.contexte.return_value = [{"text": "ctx", "piece": "p.md", "lignes": [1]}]
    mock_provider.stats_par_question = {}
    mock_provider.stats = {}

    def fabrique(_cfg, _cache):
        return mock_provider

    connexion = MagicMock()
    connexion.closed = False
    return creer_app(
        embedder=faux_embedder,
        fournisseur_fabrique=fabrique,
        connection_factory=lambda _dsn: connexion,
        reconstitueur=reconstitueur,
        **kwargs,
    )


def test_entite_perimee_lance_en_fond_sans_doublon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _marquer_perimee(racine, "client-alpha-sa")
    _config_env(monkeypatch, racine, reconstitution_appels=10)
    appels: list[str] = []
    demarre = threading.Event()
    poursuivre = threading.Event()

    def double(_cfg, unite_id: str, **_kwargs):
        appels.append(unite_id)
        demarre.set()
        assert poursuivre.wait(timeout=5)
        return {"statut": "succes", "unite": unite_id}

    app = creer_app(reconstitueur=double)
    try:
        with TestClient(app) as client:
            slug = _slug_alpha(client)
            r1 = client.get(f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO)
            assert r1.status_code == 200
            rec1 = r1.json()["reconstitution"]
            assert "client-alpha-sa" in rec1["lancees"]
            assert "client-alpha-sa" in rec1["perimees"]
            assert demarre.wait(timeout=5)
            r2 = client.get(f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO)
            rec2 = r2.json()["reconstitution"]
            assert "client-alpha-sa" in rec2["en_cours"]
            assert rec2["lancees"] == []
            assert appels == ["client-alpha-sa"]
            poursuivre.set()
    finally:
        poursuivre.set()
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_regime_jamais_ne_lance_rien(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _marquer_perimee(racine, "client-alpha-sa")
    _poser_paquet_juridique(racine)
    _config_env(
        monkeypatch,
        racine,
        reconstitution_appels=10,
        regimes={"juridique": "jamais"},
    )
    appels: list[str] = []

    def double(_cfg, unite_id: str, **_kwargs):
        appels.append(unite_id)
        return {"statut": "succes", "unite": unite_id}

    app = creer_app(reconstitueur=double)
    try:
        with TestClient(app) as client:
            slug = _slug_alpha(client)
            rec = client.get(
                f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO
            ).json()["reconstitution"]
            assert rec["lancees"] == []
            assert rec["en_cours"] == []
            assert appels == []
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_plafond_non_decide_refus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _marquer_perimee(racine, "client-alpha-sa")
    _config_env(monkeypatch, racine)
    appels: list[str] = []

    def double(_cfg, unite_id: str, **_kwargs):
        appels.append(unite_id)
        return {"statut": "succes", "unite": unite_id}

    app = creer_app(reconstitueur=double)
    try:
        with TestClient(app) as client:
            slug = _slug_alpha(client)
            rec = client.get(
                f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO
            ).json()["reconstitution"]
            assert rec == {"refus": "plafond non décidé"}
            assert appels == []
            assert client.get(
                f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO
            ).status_code == 200
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_plafond_atteint_refus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _marquer_perimee(racine, "client-alpha-sa")
    _config_env(monkeypatch, racine, reconstitution_appels=1)
    (racine / "compteur-reconstitution.json").write_text(
        json.dumps({"plafond": 1, "emis": 1}) + "\n", encoding="utf-8"
    )
    appels: list[str] = []

    def double(_cfg, unite_id: str, **_kwargs):
        appels.append(unite_id)
        return {"statut": "succes", "unite": unite_id}

    app = creer_app(reconstitueur=double)
    try:
        with TestClient(app) as client:
            slug = _slug_alpha(client)
            rec = client.get(
                f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO
            ).json()["reconstitution"]
            assert rec["refus"] == "plafond atteint"
            assert rec["lancees"] == []
            assert appels == []
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_desactivation_env_et_tenant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _marquer_perimee(racine, "client-alpha-sa")
    _config_env(monkeypatch, racine, reconstitution_appels=10)
    monkeypatch.setenv("BRAIN_API_RECONSTITUTION_A_LA_DEMANDE", "0")
    appels: list[str] = []

    def double(_cfg, unite_id: str, **_kwargs):
        appels.append(unite_id)
        return {"statut": "succes", "unite": unite_id}

    app = creer_app(reconstitueur=double)
    try:
        with TestClient(app) as client:
            slug = _slug_alpha(client)
            rec = client.get(
                f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO
            ).json()["reconstitution"]
            assert rec == {"desactivee": True}
            assert appels == []
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")

    monkeypatch.delenv("BRAIN_API_RECONSTITUTION_A_LA_DEMANDE", raising=False)
    _config_env(
        monkeypatch,
        racine,
        reconstitution_appels=10,
        reconstitution_a_la_demande=False,
    )
    app = creer_app(reconstitueur=double)
    try:
        with TestClient(app) as client:
            slug = _slug_alpha(client)
            rec = client.get(
                f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO
            ).json()["reconstitution"]
            assert rec == {"desactivee": True}
            assert appels == []
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_rechercher_cle_litterale_lance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _marquer_perimee(racine, "client-alpha-sa")
    _config_env(monkeypatch, racine, reconstitution_appels=10)
    appels: list[str] = []
    demarre = threading.Event()
    poursuivre = threading.Event()

    def double(_cfg, unite_id: str, **_kwargs):
        appels.append(unite_id)
        demarre.set()
        assert poursuivre.wait(timeout=5)
        return {"statut": "succes", "unite": unite_id}

    app = _app_recherche(reconstitueur=double)
    try:
        with TestClient(app) as client:
            sans = client.post(
                "/tenants/demo/rechercher",
                headers=_HEADERS_DEMO,
                json={"question": "Quelle est la date ?"},
            )
            assert sans.status_code == 200
            rec_sans = sans.json()["reconstitution"]
            assert rec_sans["lancees"] == []
            avec = client.post(
                "/tenants/demo/rechercher",
                headers=_HEADERS_DEMO,
                json={"question": "Où en est Alpha SA ?"},
            )
            assert avec.status_code == 200
            rec_avec = avec.json()["reconstitution"]
            assert "client-alpha-sa" in rec_avec["lancees"]
            assert demarre.wait(timeout=5)
            assert appels == ["client-alpha-sa"]
            poursuivre.set()
    finally:
        poursuivre.set()
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_get_reconstitution_journal_et_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _marquer_perimee(racine, "client-alpha-sa")
    _config_env(monkeypatch, racine, reconstitution_appels=10)
    journal = racine / "reconstitution.jsonl"
    journal.write_text(
        "".join(
            json.dumps({"unite": f"u{i}", "statut": "succes"}) + "\n" for i in range(3)
        ),
        encoding="utf-8",
    )
    appels: list[str] = []
    demarre = threading.Event()
    poursuivre = threading.Event()

    def double(_cfg, unite_id: str, **_kwargs):
        appels.append(unite_id)
        demarre.set()
        assert poursuivre.wait(timeout=5)
        return {"statut": "succes", "unite": unite_id}

    app = creer_app(reconstitueur=double)
    try:
        with TestClient(app) as client:
            slug = _slug_alpha(client)
            client.get(f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO)
            assert demarre.wait(timeout=5)
            rec = client.get(
                "/tenants/demo/reconstitution", headers=_HEADERS_DEMO
            ).json()
            unites_journal = [ligne["unite"] for ligne in rec["journal"]]
            assert unites_journal == ["u0", "u1", "u2"]
            occupes = set(rec["en_cours"]) | set(rec["en_attente"])
            assert "client-alpha-sa" in occupes
            poursuivre.set()
    finally:
        poursuivre.set()
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_fabrique_mesurer_api_brain_ne_declenche_pas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from banc_realite.mesurer_api_brain import creer_app_mesure
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _marquer_perimee(racine, "client-alpha-sa")
    _config_env(monkeypatch, racine, reconstitution_appels=10)
    appels: list[str] = []

    def double(_cfg, unite_id: str, **_kwargs):
        appels.append(unite_id)
        return {"statut": "succes", "unite": unite_id}

    app = creer_app_mesure(reconstitueur=double)
    try:
        with TestClient(app) as client:
            slug = _slug_alpha(client)
            rec = client.get(
                f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO
            ).json()["reconstitution"]
            assert rec == {"desactivee": True}
            assert appels == []
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_executor_arrete_a_la_fermeture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import (
        PREFIXE_THREAD_RECONSTITUTION,
        arreter_files_reconstitution,
        vider_etat_tenant,
    )

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _marquer_perimee(racine, "client-alpha-sa")
    _config_env(monkeypatch, racine, reconstitution_appels=10)
    fini = threading.Event()

    def double(_cfg, unite_id: str, **_kwargs):
        fini.set()
        return {"statut": "succes", "unite": unite_id}

    app = creer_app(reconstitueur=double)
    with TestClient(app) as client:
        slug = _slug_alpha(client)
        rec = client.get(
            f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO
        ).json()["reconstitution"]
        assert "client-alpha-sa" in rec["lancees"]
        assert fini.wait(timeout=5)
    deadline = time.monotonic() + 2
    restants: list[str] = ["?"]
    while time.monotonic() < deadline:
        restants = [
            t.name
            for t in threading.enumerate()
            if t.name.startswith(PREFIXE_THREAD_RECONSTITUTION)
        ]
        if not restants:
            break
        time.sleep(0.02)
    assert restants == []
    vider_etat_tenant("demo")


def test_seconde_soumission_apres_publication_ne_rappelle_pas_modele(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import (
        _file_reconstitution,
        arreter_files_reconstitution,
        vider_etat_tenant,
    )

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _marquer_perimee(racine, "client-alpha-sa")
    _config_env(monkeypatch, racine, reconstitution_appels=10)
    appels: list[str] = []
    premier_fini = threading.Event()

    def double(_cfg, unite_id: str, **_kwargs):
        appels.append(unite_id)
        path = racine / "reconstitution" / "registre.json"
        rec = json.loads(path.read_text(encoding="utf-8"))
        for unite in rec.get("unites") or []:
            if unite.get("id") == unite_id:
                unite.pop("perimee", None)
        path.write_text(json.dumps(rec), encoding="utf-8")
        premier_fini.set()
        return {"statut": "succes", "unite": unite_id}

    def _file_vide() -> None:
        file = _file_reconstitution("demo")
        deadline = time.monotonic() + 5
        while file.ids_occupes() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert file.ids_occupes() == set()

    app = creer_app(reconstitueur=double)
    try:
        with TestClient(app) as client:
            slug = _slug_alpha(client)
            rec1 = client.get(
                f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO
            ).json()["reconstitution"]
            assert "client-alpha-sa" in rec1["lancees"]
            assert premier_fini.wait(timeout=5)
            _file_vide()
            assert appels == ["client-alpha-sa"]
            stale = [
                {
                    "id": "client-alpha-sa",
                    "entite": "Alpha SA",
                    "perimee": {
                        "depuis": "2026-01-01T00:00:00+00:00",
                        "pieces": ["x.md"],
                        "passe": "p1",
                    },
                }
            ]
            monkeypatch.setattr(
                "banc_realite.regimes_reconstitution.unites_candidates_demande",
                lambda *_a, **_k: stale,
            )
            rec2 = client.get(
                f"/tenants/demo/entites/{slug}", headers=_HEADERS_DEMO
            ).json()["reconstitution"]
            assert "client-alpha-sa" in rec2["lancees"]
            _file_vide()
            assert appels == ["client-alpha-sa"]
            journal = racine / "reconstitution.jsonl"
            lignes = [
                json.loads(ligne)
                for ligne in journal.read_text(encoding="utf-8").splitlines()
                if ligne
            ]
            assert lignes[-1]["statut"] == "deja_a_jour"
            assert lignes[-1]["unite"] == "client-alpha-sa"
            assert lignes[-1]["appels"] == 0
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def _attendre_plus_en_cours(client: TestClient) -> dict:
    deadline = time.monotonic() + 5
    rec: dict = {}
    while time.monotonic() < deadline:
        rec = client.get(
            "/tenants/demo/synchronisation", headers=_HEADERS_DEMO
        ).json()
        if not rec.get("en_cours"):
            return rec
        time.sleep(0.02)
    return rec


def test_synchroniser_fond_reponse_immediate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    depot = racine / "depot-brut"
    depot.mkdir()
    _config_env(monkeypatch, racine, depot_brut=depot, embeddings_passages=10)
    demarre = threading.Event()
    poursuivre = threading.Event()
    recu: dict[str, object] = {}

    def double(cfg, **kwargs):
        recu.update(kwargs)
        ligne = {
            "declencheur": kwargs.get("declencheur"),
            "horodatage": "2026-01-01T00:00:00+00:00",
        }
        journal = cfg.chemin_journal_synchronisation()
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ligne, ensure_ascii=False) + "\n")
        demarre.set()
        assert poursuivre.wait(timeout=5)
        return ligne

    app = creer_app(synchroniseur=double)
    try:
        with TestClient(app) as client:
            t0 = time.monotonic()
            r1 = client.post(
                "/tenants/demo/synchroniser",
                headers=_HEADERS_DEMO,
                json={},
            )
            assert time.monotonic() - t0 < 1.0
            assert r1.status_code == 200
            body = r1.json()
            assert body["statut"] == "lancee"
            assert body["passe"]
            assert demarre.wait(timeout=5)
            r2 = client.post(
                "/tenants/demo/synchroniser",
                headers=_HEADERS_DEMO,
                json={},
            )
            assert r2.status_code == 200
            assert r2.json() == {"statut": "en_cours"}
            etat = client.get(
                "/tenants/demo/synchronisation", headers=_HEADERS_DEMO
            ).json()
            assert etat["en_cours"] is True
            assert etat["detenteur"]["declencheur"] == "api"
            assert recu.get("declencheur") == "api"
            poursuivre.set()
            etat = _attendre_plus_en_cours(client)
            assert etat["en_cours"] is False
            assert etat["journal"][-1]["declencheur"] == "api"
    finally:
        poursuivre.set()
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_synchroniser_en_cours_si_verrou_tenu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    depot = racine / "depot-brut"
    depot.mkdir()
    _config_env(monkeypatch, racine, depot_brut=depot, embeddings_passages=10)
    appels: list[str] = []

    def double(_cfg, **_kwargs):
        appels.append("oui")
        return {"horodatage": "h"}

    app = creer_app(synchroniseur=double)
    lock_path = racine / "reconstitution" / "synchronisation.lock"
    try:
        with TestClient(app) as client, lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            r = client.post(
                "/tenants/demo/synchroniser",
                headers=_HEADERS_DEMO,
                json={},
            )
            assert r.status_code == 200
            assert r.json() == {"statut": "en_cours"}
            assert appels == []
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_synchroniser_refus_sans_depot_brut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _config_env(monkeypatch, racine, embeddings_passages=10)
    appels: list[str] = []

    def double(_cfg, **_kwargs):
        appels.append("oui")
        return {}

    app = creer_app(synchroniseur=double)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/tenants/demo/synchroniser",
                headers=_HEADERS_DEMO,
                json={},
            )
            assert r.status_code == 200
            assert r.json() == {
                "statut": "refus",
                "motif": "tenant sans dépôt brut",
            }
            assert appels == []
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_synchroniser_refus_plafond_non_decide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    depot = racine / "depot-brut"
    depot.mkdir()
    _config_env(monkeypatch, racine, depot_brut=depot)
    appels: list[str] = []

    def double(_cfg, **_kwargs):
        appels.append("oui")
        return {}

    app = creer_app(synchroniseur=double)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/tenants/demo/synchroniser",
                headers=_HEADERS_DEMO,
                json={},
            )
            assert r.status_code == 200
            assert r.json() == {
                "statut": "refus",
                "motif": "plafond non décidé",
            }
            assert appels == []
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_synchroniser_a_blanc_synchrone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    depot = racine / "depot-brut"
    depot.mkdir()
    _config_env(monkeypatch, racine, depot_brut=depot, embeddings_passages=10)
    recu: dict[str, object] = {}

    def double(_cfg, **kwargs):
        recu.update(kwargs)
        return {
            "horodatage": "2026-01-01T00:00:00+00:00",
            "passages_remplaces": 4,
            "declencheur": kwargs.get("declencheur"),
        }

    app = creer_app(synchroniseur=double)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/tenants/demo/synchroniser",
                headers=_HEADERS_DEMO,
                json={"a_blanc": True},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["passages_remplaces"] == 4
            assert body["declencheur"] == "api"
            assert recu.get("a_blanc") is True
            assert "statut" not in body
            r2 = client.post(
                "/tenants/demo/synchroniser",
                headers=_HEADERS_DEMO,
                json={},
            )
            assert r2.status_code == 200
            assert r2.json()["statut"] == "lancee"
            _attendre_plus_en_cours(client)
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_get_synchronisation_journal_et_attente(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    _config_env(monkeypatch, racine)
    journal = racine / "synchronisation.jsonl"
    journal.write_text(
        "".join(json.dumps({"n": i, "horodatage": f"t{i}"}) + "\n" for i in range(52)),
        encoding="utf-8",
    )
    attente = {
        "pieces_a_remplacer": ["a.md"],
        "pieces_supprimees": [],
        "depuis": "2026-01-01T00:00:00+00:00",
    }
    (racine / "synchronisation.en-attente.json").write_text(
        json.dumps(attente), encoding="utf-8"
    )
    app = creer_app(synchroniseur=lambda *_a, **_k: {})
    try:
        with TestClient(app) as client:
            rec = client.get(
                "/tenants/demo/synchronisation", headers=_HEADERS_DEMO
            ).json()
            assert [ligne["n"] for ligne in rec["journal"]] == list(range(2, 52))
            assert rec["attente"]["pieces_a_remplacer"] == ["a.md"]
            assert rec["en_cours"] is False
            assert rec["detenteur"] is None
            assert rec["derniere_passe"] == "t51"
    finally:
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_verrou_partage_cli_refuse_etat_montre_detenteur(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    from banc_realite.run_synchroniser import main
    from banc_realite.synchroniser import (
        essayer_verrou_synchronisation,
        relacher_verrou_synchronisation,
    )
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant
    from brain.api_brain.tenants import resoudre_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    depot = racine / "depot-brut"
    depot.mkdir()
    _config_env(monkeypatch, racine, depot_brut=depot, embeddings_passages=10)
    cfg = resoudre_tenant("demo")
    assert cfg is not None
    handle = essayer_verrou_synchronisation(cfg, declencheur="timer")
    assert handle is not None
    app = creer_app(synchroniseur=lambda *_a, **_k: {})
    try:
        with TestClient(app) as client:
            rec = client.get(
                "/tenants/demo/synchronisation", headers=_HEADERS_DEMO
            ).json()
            assert rec["en_cours"] is True
            assert rec["detenteur"]["declencheur"] == "timer"
            assert rec["detenteur"]["pid"]
            assert rec["detenteur"]["horodatage"]
            assert main(["--tenant", "demo"]) == 2
            assert "passe déjà en cours" in capsys.readouterr().err
            r = client.post(
                "/tenants/demo/synchroniser",
                headers=_HEADERS_DEMO,
                json={},
            )
            assert r.status_code == 200
            assert r.json() == {"statut": "en_cours"}
    finally:
        relacher_verrou_synchronisation(handle)
        arreter_files_reconstitution()
        vider_etat_tenant("demo")


def test_arreter_preserve_verrou_passe_en_cours(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from banc_realite.synchroniser import (
        essayer_verrou_synchronisation,
        relacher_verrou_synchronisation,
    )
    from brain.api_brain import creer_app
    from brain.api_brain.recherche import arreter_files_reconstitution, vider_etat_tenant
    from brain.api_brain.tenants import resoudre_tenant

    vider_etat_tenant("demo")
    arreter_files_reconstitution()
    racine = _ecrire_tenant(tmp_path)
    depot = racine / "depot-brut"
    depot.mkdir()
    _config_env(monkeypatch, racine, depot_brut=depot, embeddings_passages=10)
    demarre = threading.Event()
    poursuivre = threading.Event()

    def double(_cfg, **_kwargs):
        demarre.set()
        assert poursuivre.wait(timeout=5)
        return {"horodatage": "h"}

    app = creer_app(synchroniseur=double)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/tenants/demo/synchroniser",
                headers=_HEADERS_DEMO,
                json={},
            )
            assert r.json()["statut"] == "lancee"
            assert demarre.wait(timeout=5)
        cfg = resoudre_tenant("demo")
        assert cfg is not None
        assert essayer_verrou_synchronisation(cfg, declencheur="cli") is None
    finally:
        poursuivre.set()
        cfg = resoudre_tenant("demo")
        pris = None
        if cfg is not None:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                pris = essayer_verrou_synchronisation(cfg, declencheur="cli")
                if pris is not None:
                    relacher_verrou_synchronisation(pris)
                    break
                time.sleep(0.02)
        arreter_files_reconstitution()
        vider_etat_tenant("demo")
        assert pris is not None
