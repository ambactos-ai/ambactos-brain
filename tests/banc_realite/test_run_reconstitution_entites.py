# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — run_reconstitution_entites."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from banc_realite.plafond_appels import CompteurAppels, PlafondAtteint
from banc_realite.run_reconstitution_entites import (
    EXIT_ECHEC,
    EXIT_PLAFOND,
    compter_commande,
    main,
    rassembler_dossier,
    recenser_dossier,
    reconstituer_dossier,
    regrouper_dossier,
)


def _ecrire_pieces(tmp_path: Path) -> Path:
    pieces_dir = tmp_path / "pieces"
    pieces_dir.mkdir()
    (pieces_dir / "a.md").write_text("Mill Road\nINV-2025-065\n", encoding="utf-8")
    (pieces_dir / "b.md").write_text("INV-2025-069\n", encoding="utf-8")
    return pieces_dir


def test_recenser_refuse_sans_chat(tmp_path: Path) -> None:
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    with pytest.raises(RuntimeError, match="pytest"):
        recenser_dossier(
            pieces_dir=_ecrire_pieces(tmp_path),
            sortie_dir=sortie_dir,
            dossier="moreau",
        )


def test_recenser_avec_chat_injecte(tmp_path: Path) -> None:
    pieces_dir = _ecrire_pieces(tmp_path)
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    brut = json.dumps(
        {
            "entites": [
                {
                    "nom": "Mill Road",
                    "type": "site",
                    "cles": ["Mill Road"],
                    "variantes": [],
                    "par_periode": False,
                }
            ],
            "familles": [],
        }
    )

    def chat_fn(_system: str, _user: str) -> tuple[str, dict]:
        return brut, {}

    assert (
        recenser_dossier(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier="moreau",
            chat_fn=chat_fn,
        )
        == 0
    )
    registre = json.loads((sortie_dir / "registre.json").read_text(encoding="utf-8"))
    assert len(registre["entites"]) == 1
    assert registre["unites"][0]["cles"] == ["Mill Road"]


def test_recenser_imprime_variantes_et_appel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pieces_dir = _ecrire_pieces(tmp_path)
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    brut = json.dumps(
        {
            "entites": [
                {
                    "nom": "Mill Road",
                    "type": "site",
                    "cles": ["Mill Road"],
                    "variantes": [],
                    "par_periode": False,
                }
            ],
            "familles": [],
        }
    )
    (sortie_dir / "recensement.brut.txt").write_text(brut, encoding="utf-8")
    assert (
        recenser_dossier(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier="moreau",
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["entites"] == 1
    assert payload["unites"] == 1
    assert payload["appel"] is False
    assert "variantes_par_motif" in payload
    assert payload["ecartes"] == 0


def test_recenser_refaire_via_cli_appelle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pieces_dir = _ecrire_pieces(tmp_path)
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    (sortie_dir / "recensement.brut.txt").write_text(
        json.dumps({"entites": [], "familles": []}), encoding="utf-8"
    )
    brut = json.dumps(
        {
            "entites": [
                {
                    "nom": "Mill Road",
                    "type": "site",
                    "cles": ["Mill Road"],
                    "variantes": [],
                    "par_periode": False,
                }
            ],
            "familles": [],
        }
    )
    appels = {"n": 0}

    def chat_fn(_system: str, _user: str) -> tuple[str, dict]:
        appels["n"] += 1
        return brut, {}

    assert (
        recenser_dossier(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier="moreau",
            chat_fn=chat_fn,
            refaire=True,
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["appel"] is True
    assert appels["n"] == 1
    assert (sortie_dir / "ecartes" / "recensement.brut.txt").is_file()


def test_compter_et_rassembler(tmp_path: Path) -> None:
    pieces_dir = _ecrire_pieces(tmp_path)
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    registre = {
        "entites": [],
        "unites": [
            {
                "id": "site-mill-road",
                "titre": "Mill Road",
                "cles": ["Mill Road"],
                "variantes": [],
                "periode": [],
            }
        ],
        "familles": [],
        "ecartes": {},
    }
    (sortie_dir / "registre.json").write_text(json.dumps(registre), encoding="utf-8")
    assert compter_commande(pieces_dir=pieces_dir, sortie_dir=sortie_dir) == 0
    assert rassembler_dossier(pieces_dir=pieces_dir, sortie_dir=sortie_dir) == 0
    assert (sortie_dir / "paquets" / "site-mill-road.json").is_file()


def test_reconstituer_plafond_atteint(tmp_path: Path) -> None:
    pieces_dir = _ecrire_pieces(tmp_path)
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    unite = {
        "id": "site-mill-road",
        "titre": "Mill Road",
        "cles": ["Mill Road"],
        "variantes": [],
        "periode": [],
    }
    registre = {"entites": [], "unites": [unite], "familles": [], "ecartes": {}}
    (sortie_dir / "registre.json").write_text(json.dumps(registre), encoding="utf-8")
    compteur = CompteurAppels(
        compteur_path=sortie_dir / "compteur-appels.json",
        journal_path=sortie_dir / "appels.jsonl",
        plafond=0,
    )

    def chat_fn(_system: str, _user: str) -> tuple[str, dict]:
        raise PlafondAtteint(0, 0)

    code = reconstituer_dossier(
        pieces_dir=pieces_dir,
        sortie_dir=sortie_dir,
        dossier="moreau",
        chat_fn=chat_fn,
        compteur=compteur,
    )
    assert code == EXIT_PLAFOND


def test_parallele_trois_unites_compteur_exact(tmp_path: Path) -> None:
    pieces_dir = _ecrire_pieces(tmp_path)
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    unites = [
        {
            "id": f"unite-{idx}",
            "titre": f"Entité {idx}",
            "cles": [f"INV-2025-06{idx}"],
            "variantes": [],
            "periode": [],
        }
        for idx in range(3)
    ]
    registre = {"entites": [], "unites": unites, "familles": [], "ecartes": {}}
    (sortie_dir / "registre.json").write_text(json.dumps(registre), encoding="utf-8")
    compteur = CompteurAppels(
        compteur_path=sortie_dir / "compteur-appels.json",
        journal_path=sortie_dir / "appels.jsonl",
        plafond=10,
    )
    appels: list[str] = []

    def chat_fn(_system: str, _user: str) -> tuple[str, dict]:
        payload = json.dumps(
            {
                "entite": {"nom": "x", "type": "autre", "identifiants": []},
                "faits": [],
                "identites": [],
                "regles": [],
                "non_etabli": [],
            }
        )
        for unite in unites:
            if f"ENTITÉ RECHERCHÉE : {unite['titre']}" in _user:
                appels.append(unite["id"])
                break
        return payload, {}

    assert (
        reconstituer_dossier(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier="moreau",
            parallele=3,
            chat_fn=chat_fn,
            compteur=compteur,
        )
        == 0
    )
    assert sorted(appels) == ["unite-0", "unite-1", "unite-2"]
    assert compteur.lire()["emis"] == 3
    for unite in unites:
        assert (sortie_dir / "entites" / f"{unite['id']}.json").is_file()


def test_main_bilan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sortie_dir = tmp_path / "sortie"
    entites_dir = sortie_dir / "entites"
    entites_dir.mkdir(parents=True)
    (entites_dir / "a.json").write_text(
        json.dumps(
            {
                "faits": [{"comparaison": "concordant"}],
                "identites": [{"jugement": "meme"}],
                "regles": [{}],
                "non_etabli": [{}],
                "rejets": [],
                "reparations": 2,
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "bilan",
            "--dossier",
            "moreau",
            "--pieces-dir",
            str(tmp_path / "pieces"),
            "--sortie-dir",
            str(sortie_dir),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["faits"] == 1
    assert payload["comparaisons"]["concordant"] == 1
    assert payload["reparations"] == 2


def test_regrouper_refuse_sans_chat(tmp_path: Path) -> None:
    pieces_dir = _ecrire_pieces(tmp_path)
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    registre = {
        "entites": [],
        "unites": [
            {
                "id": "site-mill-road",
                "titre": "Mill Road",
                "cles": ["Mill Road"],
                "variantes": [],
                "periode": [],
            }
        ],
        "familles": [],
        "ecartes": {},
    }
    (sortie_dir / "registre.json").write_text(json.dumps(registre), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pytest"):
        regrouper_dossier(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier="moreau",
        )


def test_regrouper_avec_chat_injecte_et_compteur(tmp_path: Path) -> None:
    pieces_dir = _ecrire_pieces(tmp_path)
    (pieces_dir / "c.md").write_text("Total for 2024: 13.334 t\n", encoding="utf-8")
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    registre = {
        "entites": [
            {
                "nom": "Mill Road",
                "type": "site",
                "cles": ["Mill Road"],
                "variantes": [],
                "par_periode": False,
            }
        ],
        "unites": [
            {
                "id": "site-mill-road",
                "entite": "Mill Road",
                "type": "site",
                "titre": "Mill Road",
                "cles": ["Mill Road"],
                "variantes": [],
                "periode": [],
            }
        ],
        "familles": [],
        "ecartes": {},
    }
    (sortie_dir / "registre.json").write_text(json.dumps(registre), encoding="utf-8")
    compteur = CompteurAppels(
        compteur_path=sortie_dir / "compteur-appels.json",
        journal_path=sortie_dir / "appels.jsonl",
        plafond=10,
    )
    appels = {"n": 0}

    def chat_fn(_system: str, user: str) -> tuple[str, dict]:
        appels["n"] += 1
        assert "Voici les lignes d'un dossier d'entreprise" in user
        return (
            json.dumps(
                {
                    "groupes": [
                        {
                            "titre": "total 2024",
                            "concept": "mesure",
                            "periode": "2024",
                            "lignes": [{"piece": "c.md", "lignes": [1]}],
                        }
                    ]
                }
            ),
            {},
        )

    assert (
        regrouper_dossier(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier="moreau",
            chat_fn=chat_fn,
            compteur=compteur,
        )
        == 0
    )
    assert appels["n"] == 1
    assert compteur.lire()["emis"] == 1
    assert (sortie_dir / "groupes-reliquat.json").is_file()
    registre2 = json.loads((sortie_dir / "registre.json").read_text(encoding="utf-8"))
    assert any(u.get("type") == "grandeur" for u in registre2["unites"])
    assert rassembler_dossier(pieces_dir=pieces_dir, sortie_dir=sortie_dir) == 0
    grandeurs = [u for u in registre2["unites"] if u.get("type") == "grandeur"]
    assert (sortie_dir / "paquets" / f"{grandeurs[0]['id']}.json").is_file()


def test_compter_affiche_reliquat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pieces_dir = _ecrire_pieces(tmp_path)
    (pieces_dir / "c.md").write_text("Total for 2024: 13.334 t\n", encoding="utf-8")
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    registre = {
        "entites": [],
        "unites": [
            {
                "id": "site-mill-road",
                "titre": "Mill Road",
                "cles": ["Mill Road"],
                "variantes": [],
                "periode": [],
            }
        ],
        "familles": [],
        "ecartes": {},
    }
    (sortie_dir / "registre.json").write_text(json.dumps(registre), encoding="utf-8")
    assert compter_commande(pieces_dir=pieces_dir, sortie_dir=sortie_dir) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "reliquat" in payload
    assert payload["reliquat"]["lignes"] >= 1
    assert "lots" in payload["reliquat"]
    assert "par_piece" in payload["reliquat"]


def test_reconstituer_unite_grandeur(tmp_path: Path) -> None:
    pieces_dir = _ecrire_pieces(tmp_path)
    (pieces_dir / "c.md").write_text("Total for 2024: 13.334 t\n", encoding="utf-8")
    sortie_dir = tmp_path / "sortie"
    sortie_dir.mkdir()
    unite = {
        "id": "grandeur-total-2024",
        "entite": "total 2024",
        "type": "grandeur",
        "titre": "total 2024, 2024",
        "cles": [],
        "variantes": [],
        "periode": ["2024"],
        "lignes": [{"piece": "c.md", "lignes": [1]}],
        "concept": "mesure",
    }
    registre = {"entites": [], "unites": [unite], "familles": [], "ecartes": {}}
    (sortie_dir / "registre.json").write_text(json.dumps(registre), encoding="utf-8")

    def chat_fn(_system: str, user: str) -> tuple[str, dict]:
        assert "ENTITÉ RECHERCHÉE : total 2024, 2024" in user
        return (
            json.dumps(
                {
                    "entite": {"nom": "total", "type": "autre", "identifiants": []},
                    "faits": [
                        {
                            "quoi": "total",
                            "concept": "mesure",
                            "versions": [
                                {
                                    "piece": "c.md",
                                    "valeur": "13.334 t",
                                    "statut": "declare",
                                    "a_la_date": None,
                                    "lignes": [1],
                                }
                            ],
                            "comparaison": "une_seule_source",
                            "explication": "x",
                        }
                    ],
                    "identites": [],
                    "regles": [],
                    "non_etabli": [],
                }
            ),
            {},
        )

    assert (
        reconstituer_dossier(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier="moreau",
            chat_fn=chat_fn,
        )
        == 0
    )
    assert (sortie_dir / "entites" / "grandeur-total-2024.json").is_file()
