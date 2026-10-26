# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — run_situations_composites."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from banc_realite.run_situations_composites import (
    EXIT_ECHEC,
    EXIT_PLAFOND,
    appliquer_dossier,
    compter_dossier,
    extraire_dossier,
    fusionner_dossier,
    main,
    rapprocher_dossier,
    reconcilier_dossier,
)
from banc_realite.situations_composites import (
    INVITE_RAPPROCHEMENT,
    INVITE_RECONCILIATION,
    REGLE_TABLEAUX,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "situations"


def test_compter_deduit_les_pieces_deja_extraites(tmp_path: Path) -> None:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    par_piece = sortie_dir / "par-piece"
    pieces_dir.mkdir()
    par_piece.mkdir(parents=True)
    (pieces_dir / "piece-a.md").write_text((FIXTURES / "piece-a.md").read_text(encoding="utf-8"), encoding="utf-8")
    (pieces_dir / "piece-b.md").write_text((FIXTURES / "piece-b.md").read_text(encoding="utf-8"), encoding="utf-8")
    (par_piece / "piece-a.md.situations.json").write_text(
        (FIXTURES / "piece-a.situations.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    rapport = compter_dossier(
        pieces_dir=pieces_dir,
        sortie_dir=sortie_dir,
        dossier="moreau",
    )
    assert rapport["appels_extraction_restants"] == 1
    assert rapport["pieces"]["piece-a.md"]["appels_extraction_restants"] == 0
    assert rapport["pieces"]["piece-b.md"]["appels_extraction_restants"] == 1
    assert rapport["pieces"]["piece-a.md"]["remarques_non_etabli"] == 1


def test_extraire_ecrit_par_piece_et_refuse_sans_chat(tmp_path: Path) -> None:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    pieces_dir.mkdir()
    (pieces_dir / "piece-a.md").write_text((FIXTURES / "piece-a.md").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pytest"):
        extraire_dossier(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier="moreau",
        )


def test_extraire_avec_chat_injecte(tmp_path: Path) -> None:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    pieces_dir.mkdir()
    texte = (FIXTURES / "piece-a.md").read_text(encoding="utf-8")
    (pieces_dir / "piece-a.md").write_text(texte, encoding="utf-8")
    brut = json.dumps(
        {
            "situations": [
                {
                    "id": "s1",
                    "perceptions": [{"preuves": [{"ligne_debut": 3, "ligne_fin": 3}]}],
                    "chronologie": [{"ligne_debut": 3, "ligne_fin": 3}],
                }
            ]
        }
    )

    def chat_fn(_system: str, _user: str) -> tuple[str, dict]:
        return brut, {}

    code = extraire_dossier(
        pieces_dir=pieces_dir,
        sortie_dir=sortie_dir,
        dossier="moreau",
        chat_fn=chat_fn,
    )
    assert code == 0
    sortie = sortie_dir / "par-piece" / "piece-a.md.situations.json"
    assert sortie.is_file()
    payload = json.loads(sortie.read_text(encoding="utf-8"))
    assert payload["situations"][0]["id"] == "m1.s1"


def test_extraire_plafond_atteint(tmp_path: Path) -> None:
    from banc_realite.plafond_appels import CompteurAppels, PlafondAtteint

    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    pieces_dir.mkdir()
    (pieces_dir / "piece-a.md").write_text("ligne\n", encoding="utf-8")
    compteur = CompteurAppels(
        compteur_path=sortie_dir / "compteur-appels.json",
        journal_path=sortie_dir / "appels.jsonl",
        plafond=0,
    )

    def chat_fn(_system: str, _user: str) -> tuple[str, dict]:
        raise PlafondAtteint(0, 0)

    code = extraire_dossier(
        pieces_dir=pieces_dir,
        sortie_dir=sortie_dir,
        dossier="moreau",
        chat_fn=chat_fn,
        compteur=compteur,
    )
    assert code == EXIT_PLAFOND


def test_rapprocher_fusionner_reconcilier_appliquer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    par_piece = sortie_dir / "par-piece"
    pieces_dir.mkdir()
    par_piece.mkdir(parents=True)
    for name in ("piece-a.md", "piece-b.md"):
        (pieces_dir / name).write_text((FIXTURES / name).read_text(encoding="utf-8"), encoding="utf-8")
        shutil_copy = FIXTURES / f"{name.replace('.md', '.situations.json')}"
        (par_piece / f"{name}.situations.json").write_text(
            shutil_copy.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    def chat_fn(_system: str, user: str) -> tuple[str, dict]:
        assert INVITE_RAPPROCHEMENT in user
        return json.dumps(
            {
                "groupes": [
                    {
                        "titre": "Retards de décembre 2023",
                        "references": ["piece-a.md#s1", "piece-b.md#s1"],
                    }
                ]
            }
        ), {}

    assert rapprocher_dossier(sortie_dir=sortie_dir, dossier="moreau", chat_fn=chat_fn) == 0
    resume_rapprochement = json.loads(capsys.readouterr().out)
    assert resume_rapprochement == {"groupes": 1, "references_ecartees": 0}
    assert (sortie_dir / "rapprochement.brut.txt").is_file()
    assert fusionner_dossier(sortie_dir=sortie_dir) == 0
    assert (sortie_dir / "situations-fusionnees.json").is_file()

    def chat_reco(_system: str, user: str) -> tuple[str, dict]:
        matiere = json.loads(user.split(INVITE_RECONCILIATION, 1)[1])
        return json.dumps(
            {
                "remarques": [
                    {
                        "id": item["id"],
                        "portee_dossier": "non_etabli",
                        "pieces": [],
                        "importance": "faible",
                        "motif": "reste",
                    }
                    for item in matiere["remarques"]
                ]
            }
        ), {}

    assert (
        reconcilier_dossier(
            sortie_dir=sortie_dir,
            pieces_dir=pieces_dir,
            dossier="moreau",
            chat_fn=chat_reco,
        )
        == 0
    )
    resume_reconciliation = json.loads(capsys.readouterr().out)
    assert resume_reconciliation["verdicts"] >= 1
    assert resume_reconciliation["ecartes"] == 0
    assert resume_reconciliation["sans_verdict"] == 0
    assert appliquer_dossier(sortie_dir=sortie_dir) == 0
    assert (sortie_dir / "situations-reconciliees.json").is_file()


def test_main_compter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    pieces_dir.mkdir()
    (pieces_dir / "piece-a.md").write_text("abc\n", encoding="utf-8")
    code = main(
        [
            "compter",
            "--pieces-dir",
            str(pieces_dir),
            "--sortie-dir",
            str(sortie_dir),
            "--dossier",
            "moreau",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["appels_extraction_restants"] == 1


def _registre_markdown() -> str:
    return "\n".join(
        [
            "# Registre",
            "",
            "| site | date | poids |",
            "| --- | --- | --- |",
            "| Alpha | 2024-01-01 | 10 |",
            "| Alpha | 2024-01-02 | 12 |",
        ]
    )


def test_extraire_transmet_regle_tableaux(tmp_path: Path) -> None:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    pieces_dir.mkdir()
    (pieces_dir / "registre.md").write_text(_registre_markdown(), encoding="utf-8")
    invites: list[str] = []

    def chat_fn(_system: str, user: str) -> tuple[str, dict]:
        invites.append(user)
        return json.dumps({"situations": [{"id": "s1", "perceptions": [], "chronologie": []}]}), {}

    assert (
        extraire_dossier(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier="greenloop",
            chat_fn=chat_fn,
            regle_tableaux=True,
        )
        == 0
    )
    assert invites
    assert invites[0].endswith(REGLE_TABLEAUX)


def test_extraire_sans_regle_tableaux_n_ajoute_pas_la_regle(tmp_path: Path) -> None:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    pieces_dir.mkdir()
    (pieces_dir / "registre.md").write_text(_registre_markdown(), encoding="utf-8")
    invites: list[str] = []

    def chat_fn(_system: str, user: str) -> tuple[str, dict]:
        invites.append(user)
        return json.dumps({"situations": [{"id": "s1", "perceptions": [], "chronologie": []}]}), {}

    extraire_dossier(
        pieces_dir=pieces_dir,
        sortie_dir=sortie_dir,
        dossier="greenloop",
        chat_fn=chat_fn,
    )
    assert REGLE_TABLEAUX not in invites[0]


def test_extraire_refaire_deplace_les_bruts(tmp_path: Path) -> None:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    pieces_dir.mkdir()
    (pieces_dir / "piece-a.md").write_text("ligne\n", encoding="utf-8")
    bruts = sortie_dir / "bruts"
    bruts.mkdir(parents=True)
    brut = bruts / "piece-a.md.m1.brut.txt"
    brut.write_text('{"situations": [{"id": "ancien"}]}', encoding="utf-8")
    (sortie_dir / "par-piece").mkdir(parents=True)
    (sortie_dir / "par-piece" / "piece-a.md.situations.json").write_text(
        json.dumps({"piece": "piece-a.md", "situations": [{"id": "m1.ancien"}]}),
        encoding="utf-8",
    )
    appels = 0

    def chat_fn(_system: str, _user: str) -> tuple[str, dict]:
        nonlocal appels
        appels += 1
        return json.dumps({"situations": [{"id": "s1", "perceptions": [], "chronologie": []}]}), {}

    code = extraire_dossier(
        pieces_dir=pieces_dir,
        sortie_dir=sortie_dir,
        dossier="moreau",
        chat_fn=chat_fn,
        refaire=True,
    )
    assert code == 0
    assert appels == 1
    assert not (bruts / "piece-a.md.m1.brut.txt").is_file() or (
        json.loads((bruts / "piece-a.md.m1.brut.txt").read_text(encoding="utf-8"))["situations"][0]["id"]
        == "s1"
    )
    ecartes = list((bruts / "ecartes").glob("piece-a.md.m1.*.brut.txt"))
    assert len(ecartes) == 1
    assert "ancien" in ecartes[0].read_text(encoding="utf-8")


def test_extraire_n_ecrit_pas_la_piece_si_un_morceau_est_illisible(tmp_path: Path) -> None:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    pieces_dir.mkdir()
    (pieces_dir / "piece-a.md").write_text("ligne une\nligne deux\n", encoding="utf-8")

    def chat_fn(_system: str, _user: str) -> tuple[str, dict]:
        return "pas un json de situations", {}

    code = extraire_dossier(
        pieces_dir=pieces_dir,
        sortie_dir=sortie_dir,
        dossier="moreau",
        chat_fn=chat_fn,
    )
    assert code == EXIT_ECHEC
    assert not (sortie_dir / "par-piece" / "piece-a.md.situations.json").is_file()
    ecartes = list((sortie_dir / "bruts" / "ecartes").glob("piece-a.md.m1.*.brut.txt"))
    assert len(ecartes) == 1


def test_compter_morceaux_tableau_et_couverture(tmp_path: Path) -> None:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    par_piece = sortie_dir / "par-piece"
    pieces_dir.mkdir()
    par_piece.mkdir(parents=True)
    registre = _registre_markdown()
    (pieces_dir / "registre.md").write_text(registre, encoding="utf-8")
    (pieces_dir / "prose.md").write_text("une ligne de prose\n", encoding="utf-8")
    (par_piece / "registre.md.situations.json").write_text(
        json.dumps(
            {
                "piece": "registre.md",
                "situations": [
                    {
                        "id": "m1.s1",
                        "nature": "processus",
                        "titre": "Pesées Alpha",
                        "non_etabli": [],
                        "perceptions": [{"preuves": [{"ligne_debut": 5, "ligne_fin": 5}]}],
                        "chronologie": [{"ligne_debut": 5, "ligne_fin": 6}],
                    }
                ],
                "couverture": {
                    "lignes_de_donnees": 2,
                    "citees": 2,
                    "non_citees": [],
                },
            }
        ),
        encoding="utf-8",
    )
    rapport = compter_dossier(
        pieces_dir=pieces_dir,
        sortie_dir=sortie_dir,
        dossier="greenloop",
    )
    assert rapport["pieces"]["registre.md"]["morceaux_avec_lignes_de_tableau"] == 1
    assert rapport["pieces"]["prose.md"]["morceaux_avec_lignes_de_tableau"] == 0
    assert rapport["pieces"]["registre.md"]["couverture"] == {
        "lignes_de_donnees": 2,
        "citees": 2,
        "non_citees": [],
    }
    assert "couverture" not in rapport["pieces"]["prose.md"]


def _dossier_deux_pieces(tmp_path: Path) -> tuple[Path, Path]:
    pieces_dir = tmp_path / "pieces"
    sortie_dir = tmp_path / "sortie"
    par_piece = sortie_dir / "par-piece"
    pieces_dir.mkdir()
    par_piece.mkdir(parents=True)
    for name in ("piece-a.md", "piece-b.md"):
        (pieces_dir / name).write_text(
            (FIXTURES / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
        (par_piece / f"{name}.situations.json").write_text(
            (FIXTURES / f"{name.replace('.md', '.situations.json')}").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
    return pieces_dir, sortie_dir


def test_compter_affiche_octets_invite_rapprochement_avec_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pieces_dir, sortie_dir = _dossier_deux_pieces(tmp_path)
    code = main(
        [
            "compter",
            "--pieces-dir",
            str(pieces_dir),
            "--sortie-dir",
            str(sortie_dir),
            "--dossier",
            "moreau",
            "--dates-max",
            "4",
            "--resume-max",
            "150",
            "--alias",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["invite_rapprochement_octets"] > 0
    assert payload["invite_rapprochement_sous_120000"] is True
    rapport = compter_dossier(
        pieces_dir=pieces_dir,
        sortie_dir=sortie_dir,
        dossier="moreau",
        dates_max=4,
        resume_max=150,
        alias=True,
    )
    assert rapport["invite_rapprochement_octets"] == payload["invite_rapprochement_octets"]


def test_rapprocher_dossier_transmet_alias(tmp_path: Path) -> None:
    _pieces_dir, sortie_dir = _dossier_deux_pieces(tmp_path)
    invites: list[str] = []

    def chat_fn(_system: str, user: str) -> tuple[str, dict]:
        invites.append(user)
        return json.dumps(
            {
                "groupes": [
                    {
                        "titre": "Retards de décembre 2023",
                        "references": ["r001", "r002"],
                    }
                ]
            }
        ), {}

    assert (
        rapprocher_dossier(
            sortie_dir=sortie_dir, dossier="moreau", chat_fn=chat_fn, alias=True
        )
        == 0
    )
    assert invites[0].startswith("r001 |")
    assert "piece-a.md#s1" not in invites[0]
    payload = json.loads((sortie_dir / "groupes.json").read_text(encoding="utf-8"))
    assert payload["groupes"][0]["references"] == ["piece-a.md#s1", "piece-b.md#s1"]
    assert payload["alias"]["r001"] == "piece-a.md#s1"
    assert (sortie_dir / "rapprochement.brut.txt.alias.json").is_file()
