# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — régimes par famille et reconstitution unitaire (#174)."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from banc_realite.etiquetage_ontologie import famille_pour_piece, familles_pour_unite
from banc_realite.plafond_appels import CompteurAppels
from banc_realite.regimes_reconstitution import (
    VERROU_MAX_S,
    charger_ontologie_tenant,
    charger_paquet_unite,
    familles_et_regime_unite,
    reconstituer_unite_tenant,
    reconstituer_unites_continues,
    unites_perimees_continues,
)
from banc_realite.reconstitution_entites import (
    ecrire_json_atomique,
    incrementer_generation_reconstitution,
    lire_generation_reconstitution,
    lire_registre_reconstitution,
    modifier_registre,
    retirer_perimee_persiste,
    unites_perimees,
)
from brain.api_brain.tenants import (
    MSG_PLAFOND_NON_DECIDE,
    REGIME_DEFAUT,
    TenantConfig,
)


def _ontologie() -> dict:
    return {
        "version": 1,
        "familles": [
            {
                "nom": "juridique",
                "regle": {"chemin": "JURIDIQUE"},
                "traitement": "lecture_simple",
            },
            {
                "nom": "travaux",
                "regle": {"chemin": "TRAVAUX"},
                "traitement": "reconstitution_entite",
            },
        ],
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
        ],
        "relations": [],
        "decisions": [],
        "sources": {},
    }


def _monde(tmp_path: Path, *, regimes: dict[str, str] | None = None) -> dict:
    racine = tmp_path / "tenant"
    lecture = racine / "lecture-drive"
    cache_tous = lecture / "cache-lecture" / "pieces" / "tous"
    reconst = racine / "reconstitution"
    ontologie_dir = racine / "ontologie"
    for path in (cache_tous, reconst / "paquets", reconst / "entites", reconst / "bruts"):
        path.mkdir(parents=True)
    (cache_tous / "JURIDIQUE-kbis.md").write_text("Alpha SA kbis\n", encoding="utf-8")
    (cache_tous / "TRAVAUX-devis.md").write_text("Alpha SA devis\n", encoding="utf-8")
    (cache_tous / "autre.md").write_text("Beta Corp note\n", encoding="utf-8")
    ontologie_path = ontologie_dir / "ontologie.v1.json"
    ontologie_dir.mkdir(parents=True, exist_ok=True)
    ontologie_path.write_text(
        json.dumps(_ontologie(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    unites = [
        {
            "id": "client-alpha-sa",
            "entite": "Alpha SA",
            "type": "client",
            "titre": "Alpha SA",
            "cles": ["Alpha SA"],
            "variantes": [],
            "periode": [],
            "perimee": {
                "depuis": "2026-01-01T00:00:00+00:00",
                "pieces": ["JURIDIQUE-kbis.md"],
                "passe": "p1",
            },
        },
        {
            "id": "fournisseur-beta-corp",
            "entite": "Beta Corp",
            "type": "fournisseur",
            "titre": "Beta Corp",
            "cles": ["Beta Corp"],
            "variantes": [],
            "periode": [],
        },
    ]
    registre = {
        "entites": _ontologie()["entites"],
        "unites": unites,
        "familles": [],
    }
    (reconst / "registre.json").write_text(
        json.dumps(registre, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paquet_alpha = {
        "marqueur": "ancien",
        "sections": {
            "document": {
                "dossier/JURIDIQUE/kbis.md": {"1": "ancien"},
                "04 — TRAVAUX/devis.md": {"1": "ancien"},
            }
        },
    }
    (reconst / "paquets" / "client-alpha-sa.json").write_text(
        json.dumps(paquet_alpha, ensure_ascii=False), encoding="utf-8"
    )
    (reconst / "paquets" / "fournisseur-beta-corp.json").write_text(
        json.dumps(
            {"sections": {"document": {"autre.md": {"1": "note"}}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reconst / "bruts" / "client-alpha-sa.brut.txt").write_text(
        "brut-obsolète", encoding="utf-8"
    )
    for uid, nom, typ in (
        ("client-alpha-sa", "Alpha SA", "client"),
        ("fournisseur-beta-corp", "Beta Corp", "fournisseur"),
    ):
        (reconst / "entites" / f"{uid}.json").write_text(
            json.dumps(
                {
                    "entite": {"nom": nom, "type": typ},
                    "faits": [],
                    "regles": [],
                    "non_etabli": [],
                    "identites": [],
                    "unite": uid,
                }
            ),
            encoding="utf-8",
        )
    cfg = TenantConfig(
        nom="demo",
        racine=racine,
        reconstitution=reconst,
        ontologie=ontologie_path,
        vault_ref="regimes-test",
        pieces=cache_tous,
        registre_pieces=lecture,
        plafonds={"reconstitution_appels": 10},
        regimes=regimes or {"juridique": "jamais", "travaux": "continu"},
    )
    compteur = CompteurAppels(
        compteur_path=racine / "compteur-reconstitution.json",
        journal_path=racine / "appels-reconstitution.jsonl",
        plafond=10,
    )
    compteur.definir_plafond(10)
    return {
        "cfg": cfg,
        "compteur": compteur,
        "reconst": reconst,
        "racine": racine,
        "cache_tous": cache_tous,
    }


def _double_succes(**kwargs):
    unite = kwargs["unite"]
    sortie = kwargs["sortie_dir"]
    destination = kwargs.get("destination")
    payload = {
        "entite": {"nom": unite.get("entite") or "Alpha SA", "type": "client"},
        "faits": [
            {
                "quoi": "refait",
                "versions": [
                    {
                        "piece": "JURIDIQUE-kbis.md",
                        "valeur": "ok",
                        "statut": "declare",
                        "lignes": [1],
                    }
                ],
            }
        ],
        "regles": [],
        "non_etabli": [],
        "identites": [],
        "unite": unite["id"],
    }
    cible_dir = Path(destination) if destination is not None else (sortie / "entites")
    cible_dir.mkdir(parents=True, exist_ok=True)
    (cible_dir / f"{unite['id']}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _poser_objet_en_cours(reconst: Path, unite_id: str) -> Path:
    source = reconst / "entites" / f"{unite_id}.json"
    cible_dir = reconst / "en-cours"
    cible_dir.mkdir(parents=True, exist_ok=True)
    cible = cible_dir / f"{unite_id}.json"
    cible.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return cible


def test_regime_par_famille_et_defaut(tmp_path: Path) -> None:
    cfg = TenantConfig(
        nom="t",
        racine=tmp_path,
        reconstitution=tmp_path / "r",
        ontologie=tmp_path / "o.json",
        vault_ref="v",
        pieces=tmp_path / "p",
        regimes={"juridique": "jamais", "travaux": "continu"},
    )
    assert cfg.regime_pour("juridique") == "jamais"
    assert cfg.regime_pour("travaux") == "continu"
    assert cfg.regime_pour("absente") == REGIME_DEFAUT
    assert cfg.regime_pour(None) == "a_la_demande"
    assert cfg.regime_pour("") == "a_la_demande"


def test_regime_le_plus_exigeant(tmp_path: Path) -> None:
    cfg = TenantConfig(
        nom="t",
        racine=tmp_path,
        reconstitution=tmp_path / "r",
        ontologie=tmp_path / "o.json",
        vault_ref="v",
        pieces=tmp_path / "p",
        regimes={"juridique": "jamais", "travaux": "continu", "mails": "a_la_demande"},
    )
    assert cfg.regime_le_plus_exigeant(["juridique", "travaux"]) == "continu"
    assert cfg.regime_le_plus_exigeant(["juridique", None]) == "a_la_demande"
    assert cfg.regime_le_plus_exigeant(["juridique"]) == "jamais"
    assert cfg.regime_le_plus_exigeant([]) == "a_la_demande"
    assert cfg.regime_le_plus_exigeant([None]) == "a_la_demande"


def test_famille_unite_depuis_paquet() -> None:
    ontologie = _ontologie()
    paquet = {
        "sections": {
            "document": {
                "dossier/JURIDIQUE/kbis.md": {"1": "x"},
                "04 — TRAVAUX/devis.md": {"1": "y"},
            }
        }
    }
    assert famille_pour_piece("dossier/JURIDIQUE/kbis.md", ontologie) == "juridique"
    assert famille_pour_piece("04 — TRAVAUX/devis.md", ontologie) == "travaux"
    assert familles_pour_unite(paquet, ontologie) == {"juridique", "travaux"}
    cfg = TenantConfig(
        nom="t",
        racine=Path("/tmp"),
        reconstitution=Path("/tmp/r"),
        ontologie=Path("/tmp/o.json"),
        vault_ref="v",
        pieces=Path("/tmp/p"),
        regimes={"juridique": "jamais", "travaux": "continu"},
    )
    nommees, regime = familles_et_regime_unite(cfg, paquet, ontologie)
    assert nommees == ["juridique", "travaux"]
    assert regime == "continu"


def test_invalidation_brut_et_paquet_rassemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", _double_succes
    )
    ancien_paquet = (monde["reconst"] / "paquets" / "client-alpha-sa.json").read_text(
        encoding="utf-8"
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
    )
    assert resultat["statut"] == "succes"
    assert not (monde["reconst"] / "bruts" / "client-alpha-sa.brut.txt").is_file()
    ecartes = list((monde["reconst"] / "ecartes").glob("client-alpha-sa.brut.*.txt"))
    assert len(ecartes) == 1
    assert "brut-obsolète" in ecartes[0].read_text(encoding="utf-8")
    nouveau = (monde["reconst"] / "paquets" / "client-alpha-sa.json").read_text(
        encoding="utf-8"
    )
    assert nouveau != ancien_paquet
    assert "marqueur" not in json.loads(nouveau)


def test_succes_retire_perimee_pages_entite_seule_vecteurs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from banc_realite.pages_entites import generer_pages

    monde = _monde(tmp_path)
    generer_pages(monde["reconst"], _ontologie(), date_gen="2026-01-01")
    page_beta = (monde["reconst"] / "pages" / "fournisseur-beta-corp.md").read_bytes()
    page_alpha_avant = (monde["reconst"] / "pages" / "client-alpha-sa.md").read_bytes()
    embarques: list[list[str] | None] = []

    def faux_embarquer(dossier: str, **kwargs):
        embarques.append(list(kwargs.get("unites") or []))
        return {"n_requetes_emises": 0}

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", _double_succes
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets", faux_embarquer
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat["statut"] == "succes"
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    assert unites_perimees(rec) == []
    assert (monde["reconst"] / "pages" / "fournisseur-beta-corp.md").read_bytes() == page_beta
    assert (monde["reconst"] / "pages" / "client-alpha-sa.md").read_bytes() != page_alpha_avant
    assert embarques == [["client-alpha-sa"]]
    journal = monde["cfg"].chemin_journal_reconstitution()
    lignes = [json.loads(l) for l in journal.read_text(encoding="utf-8").splitlines() if l]
    assert len(lignes) == 1
    assert "duree_murale_s" in lignes[0]
    assert isinstance(lignes[0]["duree_murale_s"], (int, float))
    assert lignes[0]["statut"] == "succes"
    assert lignes[0]["declencheur"] == "cli"


def test_echec_garde_perimee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monde = _monde(tmp_path)

    def faux_echec(**_kwargs):
        raise ValueError("réponse illisible")

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_echec
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
    )
    assert resultat["statut"] == "echec"
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    ids = {u["id"] for u in unites_perimees(rec)}
    assert "client-alpha-sa" in ids
    echecs = json.loads((monde["reconst"] / "echecs.json").read_text(encoding="utf-8"))
    assert any(e.get("unite") == "client-alpha-sa" for e in echecs)


def test_verrou_tenu_en_cours(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monde = _monde(tmp_path)
    verrous = monde["reconst"] / "verrous"
    verrous.mkdir(parents=True)
    (verrous / "client-alpha-sa.lock").write_text(
        json.dumps(
            {"pid": 1, "horodatage": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    appels = {"n": 0}

    def faux(**_k):
        appels["n"] += 1
        return {}

    monkeypatch.setattr("banc_realite.regimes_reconstitution.reconstituer_unite", faux)
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
    )
    assert resultat["statut"] == "en_cours"
    assert appels["n"] == 0
    assert (monde["reconst"] / "bruts" / "client-alpha-sa.brut.txt").is_file()


def test_verrou_perime_ignore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monde = _monde(tmp_path)
    verrous = monde["reconst"] / "verrous"
    verrous.mkdir(parents=True)
    ancien = datetime.now(timezone.utc) - timedelta(seconds=VERROU_MAX_S + 60)
    (verrous / "client-alpha-sa.lock").write_text(
        json.dumps({"pid": 1, "horodatage": ancien.isoformat()}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", _double_succes
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
    )
    assert resultat["statut"] == "succes"
    assert not (verrous / "client-alpha-sa.lock").is_file()


def test_plafond_atteint_arret_propre(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    monde["compteur"].reserver_emission()
    etat = monde["compteur"].lire()
    monde["compteur"].definir_plafond(int(etat["emis"]))
    appels = {"n": 0}

    def faux(**_k):
        appels["n"] += 1
        return {}

    monkeypatch.setattr("banc_realite.regimes_reconstitution.reconstituer_unite", faux)
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
    )
    assert resultat["statut"] == "plafond"
    assert appels["n"] == 0
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    assert any(u.get("id") == "client-alpha-sa" for u in unites_perimees(rec))


def test_echec_publication_garde_perimee_reprise_sans_modele(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    appels_modele = {"n": 0}
    appels_embarquer = {"n": 0}

    def faux_reconst(**kwargs):
        appels_modele["n"] += 1
        return _double_succes(**kwargs)

    def faux_embarquer(*_args, **_kwargs):
        appels_embarquer["n"] += 1
        if appels_embarquer["n"] == 1:
            raise RuntimeError("embed indisponible")
        return {"n_requetes_emises": 0}

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets", faux_embarquer
    )
    resultat1 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat1["statut"] == "echec_publication"
    assert appels_embarquer["n"] == 1
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    assert "publication" not in alpha["perimee"]
    attente = alpha["publication_en_attente"]
    assert attente["objet"] == "en-cours/client-alpha-sa.json"
    assert attente["etape"] == "vecteurs"
    assert attente["version_entree"] == alpha["perimee"]["depuis"]
    assert appels_modele["n"] == 1
    en_cours = monde["reconst"] / "en-cours" / "client-alpha-sa.json"
    publie = monde["reconst"] / "entites" / "client-alpha-sa.json"
    assert en_cours.is_file()
    assert "refait" in en_cours.read_text(encoding="utf-8")
    assert "refait" not in publie.read_text(encoding="utf-8")

    time.sleep(0.05)
    resultat2 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat2["statut"] == "succes"
    assert appels_modele["n"] == 1
    assert appels_embarquer["n"] == 2
    rec2 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    assert unites_perimees(rec2) == []
    alpha2 = next(u for u in rec2["unites"] if u["id"] == "client-alpha-sa")
    assert "publication_en_attente" not in alpha2
    assert (monde["reconst"] / "generation.json").is_file()
    assert not en_cours.is_file()
    assert "refait" in publie.read_text(encoding="utf-8")


def test_retirer_perimee_concurrent_deux_unites(tmp_path: Path) -> None:
    reconst = tmp_path / "reconstitution"
    reconst.mkdir()
    registre = {
        "entites": [],
        "unites": [
            {
                "id": "u1",
                "perimee": {"depuis": "2026-01-01T00:00:00+00:00", "pieces": [], "passe": "p"},
            },
            {
                "id": "u2",
                "perimee": {"depuis": "2026-01-01T00:00:00+00:00", "pieces": [], "passe": "p"},
            },
        ],
        "familles": [],
    }
    ecrire_json_atomique(reconst / "registre.json", registre)
    barriere = threading.Barrier(2)
    erreurs: list[str] = []

    def retirer(uid: str) -> None:
        try:
            barriere.wait()
            assert retirer_perimee_persiste(reconst, uid)
        except Exception as exc:
            erreurs.append(str(exc))

    t1 = threading.Thread(target=retirer, args=("u1",))
    t2 = threading.Thread(target=retirer, args=("u2",))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not erreurs
    final = lire_registre_reconstitution(reconst)
    assert unites_perimees(final) == []


def test_ecrire_json_atomique_temporaires_distincts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    cible = tmp_path / "registre.json"
    tmp_vus: list[str] = []
    original_replace = os.replace

    def trace_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        tmp_vus.append(Path(src).name)
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", trace_replace)
    ecrire_json_atomique(cible, {"a": 1})
    ecrire_json_atomique(cible, {"b": 2})
    assert len(tmp_vus) == 2
    assert tmp_vus[0] != tmp_vus[1]
    assert all(".tmp" in nom for nom in tmp_vus)


def test_plafond_atteint_n_empeche_pas_la_republication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    alpha["publication_en_attente"] = {
        "version_entree": alpha["perimee"]["depuis"],
        "etape": "vecteurs",
        "objet": "en-cours/client-alpha-sa.json",
        "erreur": "embed indisponible",
        "horodatage": "2026-01-02T00:00:00+00:00",
    }
    (monde["reconst"] / "registre.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _poser_objet_en_cours(monde["reconst"], "client-alpha-sa")
    monde["compteur"].reserver_emission()
    etat = monde["compteur"].lire()
    monde["compteur"].definir_plafond(int(etat["emis"]))
    appels_modele = {"n": 0}

    def faux_modele(**_kwargs):
        appels_modele["n"] += 1
        raise AssertionError("republication : pas d'appel modèle")

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_modele
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets",
        lambda *_a, **_k: {"n_requetes_emises": 0},
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat["statut"] == "succes"
    assert appels_modele["n"] == 0
    rec2 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    assert unites_perimees(rec2) == []


def test_succes_pages_sans_perimee_echec_publication_garde_marqueurs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from banc_realite.pages_entites import generer_pages

    monde = _monde(tmp_path)
    (monde["reconst"] / "echecs.json").write_text(
        json.dumps(
            [{"unite": "client-alpha-sa", "cause": "ValueError('publication précédente')"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    generer_pages(monde["reconst"], _ontologie(), date_gen="2026-01-01")
    page_avant = (monde["reconst"] / "pages" / "client-alpha-sa.md").read_text(
        encoding="utf-8"
    )
    assert "unité(s) périmée(s)" in page_avant
    assert "non reconstituée" in page_avant

    appels_embarquer = {"n": 0}

    def faux_embarquer(*_args, **_kwargs):
        appels_embarquer["n"] += 1
        if appels_embarquer["n"] == 1:
            raise RuntimeError("embed indisponible")
        return {"n_requetes_emises": 0}

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", _double_succes
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets", faux_embarquer
    )
    resultat1 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat1["statut"] == "echec_publication"
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    assert isinstance(alpha.get("perimee"), dict)
    assert "publication" not in alpha["perimee"]
    assert alpha["publication_en_attente"]["etape"] == "vecteurs"
    echecs = json.loads((monde["reconst"] / "echecs.json").read_text(encoding="utf-8"))
    assert any(e.get("unite") == "client-alpha-sa" for e in echecs)

    time.sleep(0.05)
    resultat2 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat2["statut"] == "succes"
    rec2 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    assert unites_perimees(rec2) == []
    page = (monde["reconst"] / "pages" / "client-alpha-sa.md").read_text(encoding="utf-8")
    index = (monde["reconst"] / "pages" / "index.md").read_text(encoding="utf-8")
    assert "unité(s) périmée(s)" not in page
    assert "non reconstituée" not in page
    ligne_alpha = [l for l in index.splitlines() if l.startswith("| [Alpha SA]")][0]
    assert ligne_alpha.endswith("| 0 |")
    assert "## Unités échouées" not in index
    echecs2 = json.loads((monde["reconst"] / "echecs.json").read_text(encoding="utf-8"))
    assert not any(e.get("unite") == "client-alpha-sa" for e in echecs2)


def test_matrice_depuis_cache_ignore_empreinte_absente(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from banc_realite.recherche_entites import matrice_depuis_cache

    caplog.set_level(logging.WARNING)
    matrice, ordre = matrice_depuis_cache({"a": [1.0, 0.0]}, ["a", "manquante"])
    assert ordre == ["a"]
    assert matrice.shape == (1, 2)
    assert any("manquante" in rec.message for rec in caplog.records)


def test_plafond_non_decide_leve(tmp_path: Path) -> None:
    monde = _monde(tmp_path)
    cfg = TenantConfig(
        nom="demo",
        racine=monde["racine"],
        reconstitution=monde["reconst"],
        ontologie=monde["cfg"].ontologie,
        vault_ref="regimes-test",
        pieces=monde["cache_tous"],
        registre_pieces=monde["cfg"].registre_pieces,
        plafonds={},
        regimes={"travaux": "continu"},
    )
    with pytest.raises(ValueError, match=MSG_PLAFOND_NON_DECIDE):
        reconstituer_unite_tenant(
            cfg, "client-alpha-sa", compteur=monde["compteur"], delai_s=30
        )


def test_echec_publication_unite_non_perimee_reprise_sans_modele(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    del alpha["perimee"]
    (monde["reconst"] / "registre.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    appels_modele = {"n": 0}
    appels_embarquer = {"n": 0}

    def faux_reconst(**kwargs):
        appels_modele["n"] += 1
        return _double_succes(**kwargs)

    def faux_embarquer(*_args, **_kwargs):
        appels_embarquer["n"] += 1
        if appels_embarquer["n"] == 1:
            raise RuntimeError("embed indisponible")
        return {"n_requetes_emises": 0}

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets", faux_embarquer
    )
    resultat1 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat1["statut"] == "echec_publication"
    rec1 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha1 = next(u for u in rec1["unites"] if u["id"] == "client-alpha-sa")
    assert "perimee" not in alpha1
    attente = alpha1["publication_en_attente"]
    assert attente["objet"] == "en-cours/client-alpha-sa.json"
    assert attente["etape"] == "vecteurs"
    assert attente.get("version_entree")
    assert (monde["reconst"] / "en-cours" / "client-alpha-sa.json").is_file()
    assert appels_modele["n"] == 1

    resultat2 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat2["statut"] == "succes"
    assert appels_modele["n"] == 1
    assert appels_embarquer["n"] == 2
    rec2 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    assert unites_perimees(rec2) == []
    alpha2 = next(u for u in rec2["unites"] if u["id"] == "client-alpha-sa")
    assert "publication_en_attente" not in alpha2


def _tenants_fichier(monde: dict, tmp_path: Path) -> Path:
    path = tmp_path / "tenants.json"
    path.write_text(
        json.dumps(
            {
                "demo": {
                    "racine": str(monde["racine"]),
                    "reconstitution": str(monde["reconst"]),
                    "ontologie": str(monde["cfg"].ontologie),
                    "vault_ref": "regimes-test",
                    "pieces": str(monde["cache_tous"]),
                    "plafonds": {"reconstitution_appels": 10},
                    "regimes": {"juridique": "jamais", "travaux": "continu"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _appeler_cli(argv: list[str]) -> int:
    from banc_realite.run_regimes_reconstitution import main
    from brain.api_brain.tenants import definir_tenants_fichier

    try:
        return main(argv)
    finally:
        definir_tenants_fichier(None)


def test_cli_echec_publication_code_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    tenants = _tenants_fichier(monde, tmp_path)

    def faux(*_a, **_k):
        return {
            "unite": "client-alpha-sa",
            "entite": "Alpha SA",
            "familles": [],
            "regime": "continu",
            "declencheur": "cli",
            "statut": "echec_publication",
            "erreur": "embed",
            "appels": 0,
            "duree_murale_s": 0.0,
        }

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite_tenant", faux
    )
    code = _appeler_cli(
        [
            "refaire",
            "--tenant",
            "demo",
            "--tenants-fichier",
            str(tenants),
            "--unite",
            "client-alpha-sa",
        ]
    )
    assert code == 1


def test_cli_plafond_code_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monde = _monde(tmp_path)
    tenants = _tenants_fichier(monde, tmp_path)

    def faux(*_a, **_k):
        return {
            "unite": "client-alpha-sa",
            "entite": "Alpha SA",
            "familles": [],
            "regime": "continu",
            "declencheur": "cli",
            "statut": "plafond",
            "erreur": "plafond d'appels atteint",
            "appels": 0,
            "duree_murale_s": 0.0,
        }

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite_tenant", faux
    )
    code = _appeler_cli(
        [
            "refaire",
            "--tenant",
            "demo",
            "--tenants-fichier",
            str(tenants),
            "--unite",
            "client-alpha-sa",
        ]
    )
    assert code == 3


def test_cli_perimees_echec_code_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    tenants = _tenants_fichier(monde, tmp_path)

    def faux(*_a, **_k):
        return {
            "unite": "client-alpha-sa",
            "entite": "Alpha SA",
            "familles": [],
            "regime": "continu",
            "declencheur": "cli",
            "statut": "echec",
            "erreur": "illisible",
            "appels": 0,
            "duree_murale_s": 0.0,
        }

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite_tenant", faux
    )
    code = _appeler_cli(
        [
            "refaire",
            "--tenant",
            "demo",
            "--tenants-fichier",
            str(tenants),
            "--perimees",
        ]
    )
    assert code == 1


def test_reconstituer_unite_tenant_isole_par_defaut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    vu: dict = {}

    def faux_make(*, isole: bool = False, **_kwargs):
        vu["isole"] = isole
        return lambda _s, _u: ("ok", {})

    def faux_reconstituer(**kwargs):
        chat_fn = kwargs.get("chat_fn")
        vu["chat_fn"] = chat_fn
        assert chat_fn is not None
        chat_fn("sys", "usr")
        return _double_succes(**kwargs)

    monkeypatch.setattr("banc_realite.cursor_chat.make_cursor_chat_fn", faux_make)
    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite",
        faux_reconstituer,
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
    )
    assert resultat["statut"] == "succes"
    assert vu["isole"] is True
    assert vu["chat_fn"] is not None


def _payload_cursor_ok(result: str = '{"ok": true}') -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": result,
            "session_id": "sess-1",
            "request_id": "req-1",
            "duration_api_ms": 1,
            "usage": {},
        }
    )


def test_chat_isole_deux_tentatives_deux_emissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    appels = {"n": 0}

    def run_fn(_argv, *, cwd, env, timeout):
        appels["n"] += 1
        if appels["n"] == 1:
            return 1, "", "fail"
        return 0, _payload_cursor_ok(), ""

    monkeypatch.setattr("banc_realite.cursor_chat._run_fn_isole", run_fn)
    monkeypatch.setattr("banc_realite.cursor_chat.time.sleep", lambda _s: None)

    def faux_reconstituer(**kwargs):
        chat_fn = kwargs["chat_fn"]
        assert chat_fn is not None
        chat_fn("sys", "usr")
        return _double_succes(**kwargs)

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconstituer
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
    )
    assert resultat["statut"] == "succes"
    assert appels["n"] == 2
    assert monde["compteur"].lire()["emis"] == 2
    journal = monde["compteur"].journal_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(journal) == 2


def test_chat_isole_seconde_tentative_refusee_plafond(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    monde["compteur"].definir_plafond(1)
    appels = {"n": 0}

    def run_fn(_argv, *, cwd, env, timeout):
        appels["n"] += 1
        return 1, "", "fail"

    monkeypatch.setattr("banc_realite.cursor_chat._run_fn_isole", run_fn)
    monkeypatch.setattr("banc_realite.cursor_chat.time.sleep", lambda _s: None)

    def faux_reconstituer(**kwargs):
        kwargs["chat_fn"]("sys", "usr")
        return _double_succes(**kwargs)

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconstituer
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
    )
    assert resultat["statut"] == "plafond"
    assert appels["n"] == 1
    assert monde["compteur"].lire()["emis"] == 1


def test_echec_generation_reprise_sans_modele_ni_vecteurs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    appels_modele = {"n": 0}
    appels_embarquer = {"n": 0}
    appels_gen = {"n": 0}

    def faux_reconst(**kwargs):
        appels_modele["n"] += 1
        return _double_succes(**kwargs)

    def faux_embarquer(*_args, **_kwargs):
        appels_embarquer["n"] += 1
        return {"n_requetes_emises": 0}

    def faux_gen(reconstitution_dir: Path) -> int:
        appels_gen["n"] += 1
        if appels_gen["n"] == 1:
            raise RuntimeError("generation.json indisponible")
        return incrementer_generation_reconstitution(reconstitution_dir)

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets", faux_embarquer
    )
    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.incrementer_generation_reconstitution",
        faux_gen,
    )
    resultat1 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat1["statut"] == "echec_publication"
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    assert "publication" not in alpha["perimee"]
    attente = alpha["publication_en_attente"]
    assert attente["etape"] == "generation"
    assert attente["objet"] == "entites/client-alpha-sa.json"
    assert attente["version_entree"] == alpha["perimee"]["depuis"]
    en_cours = monde["reconst"] / "en-cours" / "client-alpha-sa.json"
    publie = monde["reconst"] / "entites" / "client-alpha-sa.json"
    assert not en_cours.is_file()
    assert "refait" in publie.read_text(encoding="utf-8")
    assert lire_generation_reconstitution(monde["reconst"]) == 0

    resultat2 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat2["statut"] == "succes"
    assert appels_modele["n"] == 1
    assert appels_embarquer["n"] == 1
    assert appels_gen["n"] == 2
    rec2 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    assert unites_perimees(rec2) == []
    alpha2 = next(u for u in rec2["unites"] if u["id"] == "client-alpha-sa")
    assert "publication_en_attente" not in alpha2
    assert lire_generation_reconstitution(monde["reconst"]) == 1
    assert not en_cours.is_file()


def test_echec_promotion_paquet_reprise_sans_modele(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from banc_realite.regimes_reconstitution import (
        _promouvoir_paquet_en_cours as vrai_promouvoir,
    )

    monde = _monde(tmp_path)
    ancien_paquet = (monde["reconst"] / "paquets" / "client-alpha-sa.json").read_text(
        encoding="utf-8"
    )
    appels_modele = {"n": 0}
    appels_paquet = {"n": 0}

    def faux_reconst(**kwargs):
        appels_modele["n"] += 1
        return _double_succes(**kwargs)

    def faux_promouvoir(reconstitution: Path, unite_id: str) -> None:
        appels_paquet["n"] += 1
        if appels_paquet["n"] == 1:
            raise RuntimeError("paquet indisponible")
        vrai_promouvoir(reconstitution, unite_id)

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution._promouvoir_paquet_en_cours",
        faux_promouvoir,
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets",
        lambda *_a, **_k: {"n_requetes_emises": 0},
    )
    resultat1 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat1["statut"] == "echec_publication"
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    attente = alpha["publication_en_attente"]
    assert attente["etape"] == "paquet"
    assert "paquet" not in set(attente.get("etapes_faites") or [])
    assert "promotion" in set(attente.get("etapes_faites") or [])
    en_cours = monde["reconst"] / "en-cours" / "client-alpha-sa.json"
    publie = monde["reconst"] / "entites" / "client-alpha-sa.json"
    paquet_en_cours = monde["reconst"] / "en-cours" / "client-alpha-sa.paquet.json"
    assert not en_cours.is_file()
    assert "refait" in publie.read_text(encoding="utf-8")
    assert paquet_en_cours.is_file()
    assert (monde["reconst"] / "paquets" / "client-alpha-sa.json").read_text(
        encoding="utf-8"
    ) == ancien_paquet
    assert lire_generation_reconstitution(monde["reconst"]) == 0
    assert appels_modele["n"] == 1
    assert appels_paquet["n"] == 1

    resultat2 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat2["statut"] == "succes"
    assert appels_modele["n"] == 1
    assert appels_paquet["n"] == 2
    rec2 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    assert unites_perimees(rec2) == []
    alpha2 = next(u for u in rec2["unites"] if u["id"] == "client-alpha-sa")
    assert "publication_en_attente" not in alpha2
    assert not paquet_en_cours.is_file()
    nouveau = (monde["reconst"] / "paquets" / "client-alpha-sa.json").read_text(
        encoding="utf-8"
    )
    assert nouveau != ancien_paquet
    assert "marqueur" not in json.loads(nouveau)
    assert lire_generation_reconstitution(monde["reconst"]) == 1


def test_reprise_echec_pages_preserve_etapes_faites_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from banc_realite.pages_entites import generer_pages as vrai_generer_pages

    monde = _monde(tmp_path)
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    alpha["publication_en_attente"] = {
        "version_entree": alpha["perimee"]["depuis"],
        "etape": "generation",
        "etapes_faites": ["vecteurs", "promotion"],
        "objet": "entites/client-alpha-sa.json",
        "erreur": "generation.json indisponible",
        "horodatage": "2026-01-02T00:00:00+00:00",
    }
    (monde["reconst"] / "registre.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    appels_modele = {"n": 0}
    appels_embarquer = {"n": 0}

    def faux_modele(**_kwargs):
        appels_modele["n"] += 1
        raise AssertionError("reprise : pas d'appel modèle")

    def faux_embarquer(*_args, **_kwargs):
        appels_embarquer["n"] += 1
        return {"n_requetes_emises": 0}

    def faux_pages(*_args, **_kwargs):
        raise RuntimeError("rendu pages indisponible")

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_modele
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets", faux_embarquer
    )
    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.generer_pages", faux_pages
    )
    resultat1 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat1["statut"] == "echec_publication"
    rec1 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha1 = next(u for u in rec1["unites"] if u["id"] == "client-alpha-sa")
    faits = set(alpha1["publication_en_attente"]["etapes_faites"])
    assert "promotion" in faits
    assert appels_modele["n"] == 0
    assert appels_embarquer["n"] == 0

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.generer_pages", vrai_generer_pages
    )
    resultat2 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat2["statut"] == "succes"
    assert appels_modele["n"] == 0
    assert appels_embarquer["n"] == 0


def test_marqueur_perimee_reecrit_pendant_reconstitution_conserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    nouveau_depuis = "2026-06-15T12:00:00+00:00"

    def faux_reconst(**kwargs):
        def modifier(registre: dict) -> bool:
            for unite in registre.get("unites") or []:
                if isinstance(unite, dict) and unite.get("id") == "client-alpha-sa":
                    unite["perimee"] = {
                        "depuis": nouveau_depuis,
                        "pieces": ["TRAVAUX-devis.md"],
                        "passe": "p-nouveau",
                    }
                    return True
            return False

        modifier_registre(monde["reconst"], modifier)
        return _double_succes(**kwargs)

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets",
        lambda *_a, **_k: {"n_requetes_emises": 0},
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat["statut"] == "succes_perime_depuis"
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    assert alpha["perimee"]["depuis"] == nouveau_depuis
    assert alpha["perimee"]["passe"] == "p-nouveau"
    assert any(u.get("id") == "client-alpha-sa" for u in unites_perimees(rec))
    journal = monde["cfg"].chemin_journal_reconstitution()
    lignes = [json.loads(l) for l in journal.read_text(encoding="utf-8").splitlines() if l]
    assert lignes[-1]["statut"] == "succes_perime_depuis"


def test_deux_unites_meme_entite_publication_parallele(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    reconst = monde["reconst"]
    rec = json.loads((reconst / "registre.json").read_text(encoding="utf-8"))
    rec["unites"].append(
        {
            "id": "client-alpha-sa-avenant",
            "entite": "Alpha SA",
            "type": "client",
            "titre": "Alpha SA avenant",
            "cles": ["Alpha SA"],
            "variantes": [],
            "periode": [],
            "perimee": {
                "depuis": "2026-02-01T00:00:00+00:00",
                "pieces": ["TRAVAUX-devis.md"],
                "passe": "p2",
            },
        }
    )
    (reconst / "registre.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reconst / "paquets" / "client-alpha-sa-avenant.json").write_text(
        (reconst / "paquets" / "client-alpha-sa.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (reconst / "entites" / "client-alpha-sa-avenant.json").write_text(
        json.dumps(
            {
                "entite": {"nom": "Alpha SA", "type": "client"},
                "faits": [],
                "regles": [],
                "non_etabli": [],
                "identites": [],
                "unite": "client-alpha-sa-avenant",
            }
        ),
        encoding="utf-8",
    )
    barriere = threading.Barrier(2, timeout=10)

    def faux_reconst(**kwargs):
        unite = kwargs["unite"]
        quoi = (
            "contrat-cadre"
            if unite["id"] == "client-alpha-sa"
            else "avenant-travaux"
        )
        payload = {
            "entite": {"nom": "Alpha SA", "type": "client"},
            "faits": [
                {
                    "quoi": quoi,
                    "versions": [
                        {
                            "piece": "TRAVAUX-devis.md",
                            "valeur": "ok",
                            "statut": "declare",
                            "lignes": [1],
                        }
                    ],
                }
            ],
            "regles": [],
            "non_etabli": [],
            "identites": [],
            "unite": unite["id"],
        }
        destination = Path(kwargs["destination"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{unite['id']}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        barriere.wait()
        return payload

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets",
        lambda *_a, **_k: {"n_requetes_emises": 0},
    )
    resultats: dict[str, dict] = {}
    erreurs: list[str] = []

    def lancer(uid: str) -> None:
        try:
            resultats[uid] = reconstituer_unite_tenant(
                monde["cfg"],
                uid,
                compteur=monde["compteur"],
                delai_s=30,
                declencheur="cli",
                embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
            )
        except Exception as exc:
            erreurs.append(f"{uid}: {exc}")

    t1 = threading.Thread(target=lancer, args=("client-alpha-sa",))
    t2 = threading.Thread(target=lancer, args=("client-alpha-sa-avenant",))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)
    assert not erreurs
    assert resultats["client-alpha-sa"]["statut"] == "succes"
    assert resultats["client-alpha-sa-avenant"]["statut"] == "succes"
    page = (reconst / "pages" / "client-alpha-sa.md").read_text(encoding="utf-8")
    assert "contrat-cadre" in page
    assert "avenant-travaux" in page
    index = (reconst / "pages" / "index.md").read_text(encoding="utf-8")
    ligne_alpha = [l for l in index.splitlines() if l.startswith("| [Alpha SA]")][0]
    cellules = [c.strip() for c in ligne_alpha.split("|")]
    # | entité | type | unités | faits | ...
    assert cellules[3] == "2"


def _ecrire_objet_en_cours(kwargs: dict, quoi: str) -> dict:
    unite = kwargs["unite"]
    payload = {
        "entite": {"nom": unite.get("entite") or "Alpha SA", "type": "client"},
        "faits": [
            {
                "quoi": quoi,
                "versions": [
                    {
                        "piece": "TRAVAUX-devis.md",
                        "valeur": "ok",
                        "statut": "declare",
                        "lignes": [1],
                    }
                ],
            }
        ],
        "regles": [],
        "non_etabli": [],
        "identites": [],
        "unite": unite["id"],
    }
    destination = Path(kwargs["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{unite['id']}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _poser_avenant_alpha(reconst: Path) -> None:
    rec = json.loads((reconst / "registre.json").read_text(encoding="utf-8"))
    rec["unites"].append(
        {
            "id": "client-alpha-sa-avenant",
            "entite": "Alpha SA",
            "type": "client",
            "titre": "Alpha SA avenant",
            "cles": ["Alpha SA"],
            "variantes": [],
            "periode": [],
            "perimee": {
                "depuis": "2026-02-01T00:00:00+00:00",
                "pieces": ["TRAVAUX-devis.md"],
                "passe": "p2",
            },
        }
    )
    (reconst / "registre.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reconst / "paquets" / "client-alpha-sa-avenant.json").write_text(
        (reconst / "paquets" / "client-alpha-sa.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (reconst / "entites" / "client-alpha-sa-avenant.json").write_text(
        json.dumps(
            {
                "entite": {"nom": "Alpha SA", "type": "client"},
                "faits": [],
                "regles": [],
                "non_etabli": [],
                "identites": [],
                "unite": "client-alpha-sa-avenant",
            }
        ),
        encoding="utf-8",
    )


def test_reprise_apres_perimee_plus_recente_reconstitue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    appels_modele = {"n": 0}
    appels_embarquer = {"n": 0}

    def faux_reconst(**kwargs):
        appels_modele["n"] += 1
        quoi = "ancien-objet" if appels_modele["n"] == 1 else "objet-neuf"
        return _ecrire_objet_en_cours(kwargs, quoi)

    def faux_embarquer(*_args, **_kwargs):
        appels_embarquer["n"] += 1
        if appels_embarquer["n"] == 1:
            raise RuntimeError("embed indisponible")
        return {"n_requetes_emises": 0}

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets", faux_embarquer
    )
    resultat1 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat1["statut"] == "echec_publication"
    assert appels_modele["n"] == 1
    en_cours = monde["reconst"] / "en-cours" / "client-alpha-sa.json"
    assert "ancien-objet" in en_cours.read_text(encoding="utf-8")

    nouveau_depuis = "2026-09-01T00:00:00+00:00"

    def poser_perimee_plus_recente(registre: dict) -> bool:
        for unite in registre.get("unites") or []:
            if isinstance(unite, dict) and unite.get("id") == "client-alpha-sa":
                unite["perimee"] = {
                    "depuis": nouveau_depuis,
                    "pieces": ["TRAVAUX-devis.md"],
                    "passe": "p-nouveau",
                }
                return True
        return False

    modifier_registre(monde["reconst"], poser_perimee_plus_recente)
    resultat2 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat2["statut"] == "succes"
    assert appels_modele["n"] == 2
    publie = (monde["reconst"] / "entites" / "client-alpha-sa.json").read_text(
        encoding="utf-8"
    )
    assert "objet-neuf" in publie
    assert "ancien-objet" not in publie
    assert not en_cours.is_file()
    ecartes = list((monde["reconst"] / "ecartes").glob("client-alpha-sa.objet.*.json"))
    assert len(ecartes) == 1
    assert "ancien-objet" in ecartes[0].read_text(encoding="utf-8")


def test_reprise_promotion_regenere_pages_avec_fait_concurrent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from banc_realite.reconstitution_entites import (
        publier_objet_en_cours as vrai_publier,
    )

    monde = _monde(tmp_path)
    _poser_avenant_alpha(monde["reconst"])
    promotions_a = {"n": 0}

    def faux_reconst(**kwargs):
        uid = kwargs["unite"]["id"]
        quoi = "fait-a" if uid == "client-alpha-sa" else "fait-b"
        return _ecrire_objet_en_cours(kwargs, quoi)

    def faux_publier(reconstitution_dir: Path, unite_id: str) -> None:
        if unite_id == "client-alpha-sa" and promotions_a["n"] == 0:
            promotions_a["n"] += 1
            raise RuntimeError("promotion indisponible")
        vrai_publier(reconstitution_dir, unite_id)

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.publier_objet_en_cours", faux_publier
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets",
        lambda *_a, **_k: {"n_requetes_emises": 0},
    )
    resultat_a1 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat_a1["statut"] == "echec_publication"
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    assert alpha["publication_en_attente"]["etape"] == "promotion"
    resultat_b = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa-avenant",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat_b["statut"] == "succes"
    resultat_a2 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat_a2["statut"] == "succes"
    page = (monde["reconst"] / "pages" / "client-alpha-sa.md").read_text(
        encoding="utf-8"
    )
    assert "fait-a" in page
    assert "fait-b" in page


def test_refaire_unite_a_jour_invalide_brut_et_appelle_modele(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    del alpha["perimee"]
    (monde["reconst"] / "registre.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ancien_paquet = (monde["reconst"] / "paquets" / "client-alpha-sa.json").read_text(
        encoding="utf-8"
    )
    appels_modele = {"n": 0}

    def faux_reconst(**kwargs):
        appels_modele["n"] += 1
        return _double_succes(**kwargs)

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="cli",
    )
    assert resultat["statut"] == "succes"
    assert appels_modele["n"] == 1
    assert not (monde["reconst"] / "bruts" / "client-alpha-sa.brut.txt").is_file()
    ecartes = list((monde["reconst"] / "ecartes").glob("client-alpha-sa.brut.*.txt"))
    assert len(ecartes) == 1
    assert "brut-obsolète" in ecartes[0].read_text(encoding="utf-8")
    nouveau = (monde["reconst"] / "paquets" / "client-alpha-sa.json").read_text(
        encoding="utf-8"
    )
    assert nouveau != ancien_paquet
    assert "marqueur" not in json.loads(nouveau)


def _echec_publication_piece_continue_supprimee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    """Famille continue : dernière pièce retirée, reconstitution ok, publication échoue."""
    monde = _monde(tmp_path)
    (monde["cache_tous"] / "TRAVAUX-devis.md").unlink()
    appels_modele = {"n": 0}
    appels_embarquer = {"n": 0}

    def faux_reconst(**kwargs):
        appels_modele["n"] += 1
        return _double_succes(**kwargs)

    def faux_embarquer(*_args, **_kwargs):
        appels_embarquer["n"] += 1
        if appels_embarquer["n"] == 1:
            raise RuntimeError("embed indisponible")
        return {"n_requetes_emises": 0}

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets", faux_embarquer
    )
    resultat = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="synchroniser",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
    )
    assert resultat["statut"] == "echec_publication"
    assert resultat["regime"] == "continu"
    assert appels_modele["n"] == 1
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    attente = alpha["publication_en_attente"]
    assert attente["regime"] == "continu"
    assert "travaux" in attente["familles"]
    assert "reconstitution_en_attente" not in alpha
    monde["appels_modele"] = appels_modele
    monde["appels_embarquer"] = appels_embarquer
    return monde


def test_publication_en_attente_reste_eligible_paquet_reecrit_passe_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _echec_publication_piece_continue_supprimee(tmp_path, monkeypatch)
    (monde["reconst"] / "paquets" / "client-alpha-sa.json").write_text(
        "{}\n", encoding="utf-8"
    )
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    onto = charger_ontologie_tenant(monde["cfg"])
    cibles = unites_perimees_continues(monde["cfg"], rec, onto, monde["reconst"])
    assert any(u.get("id") == "client-alpha-sa" for u in cibles)

    resultat = reconstituer_unites_continues(
        monde["cfg"],
        compteur=monde["compteur"],
        delai_s=30,
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
        declencheur="synchroniser",
    )
    assert "client-alpha-sa" in resultat["unites_refaites"]
    assert monde["appels_modele"]["n"] == 1
    rec2 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    assert unites_perimees(rec2) == []
    alpha2 = next(u for u in rec2["unites"] if u["id"] == "client-alpha-sa")
    assert "publication_en_attente" not in alpha2


def test_publication_en_attente_reste_eligible_paquet_reecrit_refaire_unite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _echec_publication_piece_continue_supprimee(tmp_path, monkeypatch)
    tenants = _tenants_fichier(monde, tmp_path)
    code = _appeler_cli(
        [
            "refaire",
            "--tenant",
            "demo",
            "--tenants-fichier",
            str(tenants),
            "--unite",
            "client-alpha-sa",
        ]
    )
    assert code == 0
    assert monde["appels_modele"]["n"] == 1
    rec2 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    assert unites_perimees(rec2) == []
    alpha2 = next(u for u in rec2["unites"] if u["id"] == "client-alpha-sa")
    assert "publication_en_attente" not in alpha2


def test_echec_modele_preserve_paquet_reference_et_retente(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    (monde["cache_tous"] / "TRAVAUX-devis.md").unlink()
    ancien = (monde["reconst"] / "paquets" / "client-alpha-sa.json").read_text(
        encoding="utf-8"
    )
    appels = {"n": 0}

    def faux_echec(**_kwargs):
        appels["n"] += 1
        raise TimeoutError("délai modèle")

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_echec
    )
    resultat1 = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="synchroniser",
    )
    assert resultat1["statut"] == "echec"
    assert appels["n"] == 1
    assert (monde["reconst"] / "paquets" / "client-alpha-sa.json").read_text(
        encoding="utf-8"
    ) == ancien
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    alpha = next(u for u in rec["unites"] if u["id"] == "client-alpha-sa")
    attente = alpha["reconstitution_en_attente"]
    assert attente["regime"] == "continu"
    assert "travaux" in attente["familles"]
    assert attente["version_entree"] == alpha["perimee"]["depuis"]
    assert attente["depuis"] == alpha["perimee"]["depuis"]
    assert (monde["reconst"] / "en-cours" / "client-alpha-sa.paquet.json").is_file()

    (monde["reconst"] / "paquets" / "client-alpha-sa.json").write_text(
        "{}\n", encoding="utf-8"
    )
    onto = charger_ontologie_tenant(monde["cfg"])
    rec2 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    cibles = unites_perimees_continues(monde["cfg"], rec2, onto, monde["reconst"])
    assert any(u.get("id") == "client-alpha-sa" for u in cibles)

    resultat2 = reconstituer_unites_continues(
        monde["cfg"],
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="synchroniser",
    )
    assert "client-alpha-sa" in resultat2["unites_echouees"]
    assert appels["n"] == 2


def test_plafond_n_interrompt_pas_la_publication_en_attente_suivante(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    rec = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    beta = next(u for u in rec["unites"] if u["id"] == "fournisseur-beta-corp")
    beta["perimee"] = {
        "depuis": "2026-01-01T00:00:00+00:00",
        "pieces": ["autre.md"],
        "passe": "p1",
    }
    beta["publication_en_attente"] = {
        "version_entree": beta["perimee"]["depuis"],
        "etape": "vecteurs",
        "objet": "en-cours/fournisseur-beta-corp.json",
        "erreur": "embed indisponible",
        "horodatage": "2026-01-02T00:00:00+00:00",
        "regime": "continu",
        "familles": [],
    }
    (monde["reconst"] / "registre.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _poser_objet_en_cours(monde["reconst"], "fournisseur-beta-corp")
    monde["compteur"].reserver_emission()
    etat = monde["compteur"].lire()
    monde["compteur"].definir_plafond(int(etat["emis"]))
    appels_modele = {"n": 0}

    def faux_modele(**_kwargs):
        appels_modele["n"] += 1
        raise AssertionError("plafond : pas d'appel modèle")

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_modele
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets",
        lambda *_a, **_k: {"n_requetes_emises": 0},
    )
    resultat = reconstituer_unites_continues(
        monde["cfg"],
        compteur=monde["compteur"],
        delai_s=30,
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
        declencheur="synchroniser",
    )
    assert "client-alpha-sa" in resultat["unites_echouees"]
    assert "fournisseur-beta-corp" in resultat["unites_refaites"]
    assert appels_modele["n"] == 0
    rec2 = json.loads((monde["reconst"] / "registre.json").read_text(encoding="utf-8"))
    beta2 = next(u for u in rec2["unites"] if u["id"] == "fournisseur-beta-corp")
    assert "publication_en_attente" not in beta2
    assert "perimee" not in beta2


def test_lire_definir_plafond_16_threads_sans_exception(tmp_path: Path) -> None:
    path = tmp_path / "compteur-appels.json"
    journal = tmp_path / "appels.jsonl"
    erreurs: list[BaseException] = []
    garde = threading.Lock()

    def travail() -> None:
        compteur = CompteurAppels(compteur_path=path, journal_path=journal)
        try:
            for k in range(32):
                if k % 2 == 0:
                    compteur.definir_plafond(10 + (k % 5))
                else:
                    compteur.lire()
        except BaseException as exc:
            with garde:
                erreurs.append(exc)

    threads = [threading.Thread(target=travail) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()
    assert erreurs == []


def test_seulement_si_eligible_deuxieme_appel_sans_modele(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monde = _monde(tmp_path)
    appels_modele = {"n": 0}

    def faux_reconst(**kwargs):
        appels_modele["n"] += 1
        return _double_succes(**kwargs)

    monkeypatch.setattr(
        "banc_realite.regimes_reconstitution.reconstituer_unite", faux_reconst
    )
    monkeypatch.setattr(
        "banc_realite.embeddings_reconstitution.embarquer_objets",
        lambda *_a, **_k: {"n_requetes_emises": 0},
    )
    premier = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="api",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
        seulement_si_eligible=True,
    )
    assert premier["statut"] == "succes"
    assert appels_modele["n"] == 1
    second = reconstituer_unite_tenant(
        monde["cfg"],
        "client-alpha-sa",
        compteur=monde["compteur"],
        delai_s=30,
        declencheur="api",
        embed_fn=lambda textes: [[0.1] * 4 for _ in textes],
        seulement_si_eligible=True,
    )
    assert second["statut"] == "deja_a_jour"
    assert appels_modele["n"] == 1
    journal = monde["cfg"].chemin_journal_reconstitution()
    lignes = [
        json.loads(ligne)
        for ligne in journal.read_text(encoding="utf-8").splitlines()
        if ligne
    ]
    assert lignes[-1]["statut"] == "deja_a_jour"
    assert lignes[-1]["appels"] == 0
