# SPDX-License-Identifier: AGPL-3.0-only
"""Tests des contrôles déterministes de reconstitution par entité."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from banc_realite.controle_reconstitution import (
    BLOQUANTS,
    controler,
    extraire_nombres,
    forme_comparaison,
    jetons,
    normaliser_nombre,
)


def _ecrire_registre(sortie: Path, uids: list[str]) -> None:
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "registre.json").write_text(
        json.dumps({"unites": [{"id": uid} for uid in uids]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _ecrire_paquet(
    sortie: Path,
    uid: str,
    *,
    entite: dict | None = None,
    document: dict | None = None,
) -> None:
    paquets = sortie / "paquets"
    paquets.mkdir(parents=True, exist_ok=True)
    payload = {
        "entite": uid,
        "sections": {
            "entite": entite or {},
            "document": document or {},
            "variantes": {},
            "emergent": {},
        },
    }
    (paquets / f"{uid}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _ecrire_entite(sortie: Path, uid: str, payload: dict) -> None:
    entites = sortie / "entites"
    entites.mkdir(parents=True, exist_ok=True)
    (entites / f"{uid}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _corpus(pieces: Path) -> dict[str, list[str]]:
    return {
        path.name: path.read_text(encoding="utf-8").splitlines()
        for path in sorted(pieces.iterdir())
        if path.is_file()
    }


def _entite_citant_tout(
    pieces: Path,
    *,
    entite: dict | None = None,
    document: dict | None = None,
    extra: dict | None = None,
) -> dict:
    corpus = _corpus(pieces)
    payload: dict = {"faits": [], "regles": [], "identites": []}
    if extra:
        payload.update(extra)
    for section in (entite or {}, document or {}):
        for piece, lignes in section.items():
            nums = sorted(int(n) for n in lignes)
            lettre = "\n".join(corpus[piece][n - 1] for n in nums)
            payload["faits"].append(
                {
                    "versions": [
                        {
                            "piece": piece,
                            "lignes": nums,
                            "lettre": lettre,
                            "valeur": "x",
                        }
                    ]
                }
            )
    return payload


def _base(tmp_path: Path) -> tuple[Path, Path]:
    pieces = tmp_path / "pieces"
    sortie = tmp_path / "sortie"
    pieces.mkdir()
    sortie.mkdir()
    return pieces, sortie


def test_module_n_importe_pas_or_dossier() -> None:
    source = Path(__file__).resolve().parents[2] / "scripts/banc_realite/controle_reconstitution.py"
    texte = source.read_text(encoding="utf-8")
    assert "or_dossier" not in texte
    assert "moreau_or" not in texte
    assert "load_gold" not in texte
    assert "fidelite" not in texte


def test_normaliser_nombre() -> None:
    assert normaliser_nombre("1 234,50") == "1234.50"
    assert normaliser_nombre("20.415") == "20.415"


def test_controle_1_aucune_ligne_oubliee_ok(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("ligne une\nligne deux\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "ligne une", "2": "ligne deux"}})
    _ecrire_entite(
        sortie,
        "u1",
        _entite_citant_tout(pieces, entite={"a.md": {"1": "ligne une", "2": "ligne deux"}}),
    )
    ctrl = controler(pieces, sortie)["controles"]["aucune_ligne_oubliee"]
    assert ctrl["verdict"] == "ok"
    assert ctrl["ecarts"] == []


def test_controle_1_aucune_ligne_oubliee_ecart(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("ligne une\nligne deux\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "ligne une"}})
    _ecrire_entite(
        sortie,
        "u1",
        _entite_citant_tout(pieces, entite={"a.md": {"1": "ligne une"}}),
    )
    ctrl = controler(pieces, sortie)["controles"]["aucune_ligne_oubliee"]
    assert ctrl["verdict"] == "ecart"
    assert ctrl["ecarts"] == [{"piece": "a.md", "ligne": 2, "texte": "ligne deux"}]


def test_controle_2_chaque_unite_reconstituee_ok(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("texte\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "texte"}})
    _ecrire_entite(
        sortie,
        "u1",
        _entite_citant_tout(pieces, entite={"a.md": {"1": "texte"}}),
    )
    ctrl = controler(pieces, sortie)["controles"]["chaque_unite_reconstituee"]
    assert ctrl["verdict"] == "ok"


def test_controle_2_chaque_unite_reconstituee_ecart(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("texte\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1", "u2"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "texte"}})
    _ecrire_paquet(sortie, "u2", entite={"a.md": {"1": "texte"}})
    _ecrire_entite(
        sortie,
        "u1",
        _entite_citant_tout(pieces, entite={"a.md": {"1": "texte"}}),
    )
    ctrl = controler(pieces, sortie)["controles"]["chaque_unite_reconstituee"]
    assert ctrl["verdict"] == "ecart"
    assert any(e["unite"] == "u2" and e["motif"] == "fichier absent" for e in ctrl["ecarts"])


def test_controle_3_lettre_ok(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("alpha\nbeta\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "alpha", "2": "beta"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1, 2],
                            "lettre": "alpha\nbeta",
                            "valeur": "x",
                        }
                    ]
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["lettre_est_celle_de_la_piece"]
    assert ctrl["verdict"] == "ok"


def test_controle_3_lettre_ecart(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("alpha\nbeta\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "alpha", "2": "beta"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1, 2],
                            "lettre": "alpha seulement",
                            "valeur": "x",
                        }
                    ]
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["lettre_est_celle_de_la_piece"]
    assert ctrl["verdict"] == "ecart"
    assert any(e["motif"] == "lettre incorrecte" for e in ctrl["ecarts"])


def test_controle_4_concordant_ok(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("l1\nl2\n", encoding="utf-8")
    (pieces / "b.md").write_text("l3\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(
        sortie,
        "u1",
        entite={"a.md": {"1": "l1", "2": "l2"}, "b.md": {"1": "l3"}},
    )
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "quoi": "montant",
                    "comparaison": "concordant",
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": "l1",
                            "valeur": "1 000,00 EUR",
                        },
                        {
                            "piece": "b.md",
                            "lignes": [1],
                            "lettre": "l3",
                            "valeur": "1000.00 EUR",
                        },
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["concordant_est_vrai"]
    assert ctrl["verdict"] == "ok"


def test_controle_4_concordant_ecart(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("l1\nl2\n", encoding="utf-8")
    (pieces / "b.md").write_text("l3\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(
        sortie,
        "u1",
        entite={"a.md": {"1": "l1", "2": "l2"}, "b.md": {"1": "l3"}},
    )
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "quoi": "montant",
                    "comparaison": "concordant",
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": "l1",
                            "valeur": "100 EUR",
                        },
                        {
                            "piece": "b.md",
                            "lignes": [1],
                            "lettre": "l3",
                            "valeur": "200 EUR",
                        },
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["concordant_est_vrai"]
    assert ctrl["verdict"] == "ecart"
    assert ctrl["ecarts"][0]["motif"] == "jetons_disjoints"


def test_controle_5_desaccord_ok(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("l1\n", encoding="utf-8")
    (pieces / "b.md").write_text("l2\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(
        sortie,
        "u1",
        entite={"a.md": {"1": "l1"}, "b.md": {"1": "l2"}},
    )
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "quoi": "montant",
                    "comparaison": "desaccord",
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": "l1",
                            "valeur": "100 EUR",
                            "statut": "du",
                        },
                        {
                            "piece": "b.md",
                            "lignes": [1],
                            "lettre": "l2",
                            "valeur": "200 EUR",
                            "statut": "du",
                        },
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["desaccord_est_vrai"]
    assert ctrl["verdict"] == "ok"


def test_controle_5_desaccord_ecart(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("l1\n", encoding="utf-8")
    (pieces / "b.md").write_text("l2\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(
        sortie,
        "u1",
        entite={"a.md": {"1": "l1"}, "b.md": {"1": "l2"}},
    )
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "quoi": "montant",
                    "comparaison": "desaccord",
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": "l1",
                            "valeur": "100 EUR",
                            "statut": "du",
                        },
                        {
                            "piece": "b.md",
                            "lignes": [1],
                            "lettre": "l2",
                            "valeur": "100 EUR",
                            "statut": "du",
                        },
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["desaccord_est_vrai"]
    assert ctrl["verdict"] == "ecart"
    assert ctrl["ecarts"][0]["motif"] == "desaccord_non_verifie"


def test_controle_6_valeur_dans_lettre_ok(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("Total 20.415 t\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "Total 20.415 t"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": "Total 20.415 t",
                            "valeur": "20.415 t",
                        }
                    ]
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["valeur_dans_la_lettre"]
    assert ctrl["verdict"] == "ok"


def test_controle_6_valeur_dans_lettre_ecart(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("Total 20.415 t\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "Total 20.415 t"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": "Total 20.415 t",
                            "valeur": "99.999 t",
                        }
                    ]
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["valeur_dans_la_lettre"]
    assert ctrl["verdict"] == "ecart"
    assert ctrl["ecarts"][0]["motif"] == "valeur_absente_de_la_lettre"


def test_controle_7_paquet_non_cite_ok(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("alpha\nbeta\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "alpha", "2": "beta"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1, 2],
                            "lettre": "alpha\nbeta",
                            "valeur": "x",
                        }
                    ]
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["rien_du_paquet_laisse_de_cote"]
    assert ctrl["verdict"] == "ok"


def test_controle_7_paquet_non_cite_ecart(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("alpha\nbeta\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "alpha", "2": "beta"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": "alpha",
                            "valeur": "x",
                        }
                    ]
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["rien_du_paquet_laisse_de_cote"]
    assert ctrl["verdict"] == "ecart"
    assert ctrl["ecarts"][0]["non_citees"] == [{"piece": "a.md", "ligne": 2}]


def test_controle_8_desaccords_possibles_ok(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("l1\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "l1"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "quoi": "tonnage total mensuel",
                    "concept": "mesure",
                    "comparaison": "une_seule_source",
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": "l1",
                            "valeur": "10 t",
                        }
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["desaccords_possibles_non_releves"]
    assert ctrl["verdict"] == "ok"
    assert ctrl["ecarts"] == []


def test_controle_8_desaccords_possibles_ecart(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("l1\nl2\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1", "u2"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "l1"}})
    _ecrire_paquet(sortie, "u2", entite={"a.md": {"2": "l2"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "quoi": "tonnage total mensuel",
                    "concept": "mesure",
                    "comparaison": "une_seule_source",
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": "l1",
                            "valeur": "10 t le 2024-01",
                        }
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    _ecrire_entite(
        sortie,
        "u2",
        {
            "faits": [
                {
                    "quoi": "tonnage total mensuel du site",
                    "concept": "mesure",
                    "comparaison": "concordant",
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [2],
                            "lettre": "l2",
                            "valeur": "20 t le 2024-01",
                        }
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["desaccords_possibles_non_releves"]
    assert ctrl["verdict"] == "a_relire"
    assert len(ctrl["ecarts"]) >= 1


def test_bloquant_ok_aggregue(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("alpha\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "alpha"}})
    _ecrire_entite(
        sortie,
        "u1",
        _entite_citant_tout(pieces, entite={"a.md": {"1": "alpha"}}),
    )
    resultat = controler(pieces, sortie)
    assert resultat["bloquant_ok"] is True
    assert all(resultat["controles"][nom]["verdict"] == "ok" for nom in BLOQUANTS)


def test_cli_rapport_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from banc_realite.controle_reconstitution import main

    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("alpha\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "alpha"}})
    _ecrire_entite(
        sortie,
        "u1",
        _entite_citant_tout(pieces, entite={"a.md": {"1": "alpha"}}),
    )
    rapport = tmp_path / "rapport.md"
    code = main(["--pieces-dir", str(pieces), "--sortie-dir", str(sortie), "--rapport", str(rapport)])
    assert code == 0
    assert rapport.is_file()
    texte = rapport.read_text(encoding="utf-8")
    assert "| Contrôle | Vérifiés | Passés | Écarts | Non vérifiables | Verdict |" in texte
    out = capsys.readouterr().out
    assert '"bloquant_ok": true' in out


def test_extraire_nombres_partage_concordance() -> None:
    assert "1000.00" in extraire_nombres("1 000,00")
    assert "1000.00" in extraire_nombres("1000.00")


def test_jetons_date_aaaa_mm_jj_pas_eclatee() -> None:
    lettre = "The collection agreement was signed on 2023-05-15. Service began on 2023-05-24."
    j = jetons(lettre)
    assert "2023-05-15" in j
    assert "2023-05-24" in j
    assert "2023" not in j
    assert "15" not in j
    assert jetons("2023-05-15") == {"2023-05-15"}


def test_jetons_identifiant_et_periode() -> None:
    valeur = "GL-CERT-JTS-R4A ; Requested review period: 2024-01 through 2025-04"
    j = jetons(valeur)
    assert "GL-CERT-JTS-R4A" in j
    assert "2024-01" in j
    assert "2025-04" in j


def test_jetons_virgule_deux_lectures() -> None:
    j = jetons("1,88")
    assert "1.88" in j
    assert "1" in j
    assert "88" in j


def test_jetons_signe_moins_conserve() -> None:
    assert "-3.170" in jetons("-3.170")
    assert "-3.170" in jetons("raw_kg -3.170 kg")


def test_forme_comparaison_zeros_fin() -> None:
    assert forme_comparaison("31.380") == forme_comparaison("31.38")
    assert forme_comparaison("0.000") == forme_comparaison("0")
    assert "0.000" in jetons("0.000")
    assert "0" in jetons("0")
    assert jetons("0.000") != jetons("0")


def test_controle_6_date_dans_lettre(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    lettre = "The collection agreement was signed on 2023-05-15. Service began on 2023-05-24."
    (pieces / "a.md").write_text(lettre + "\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": lettre}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": lettre,
                            "valeur": "2023-05-15",
                        }
                    ]
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["valeur_dans_la_lettre"]
    assert ctrl["verdict"] == "ok"
    assert ctrl["ecarts"] == []
    assert ctrl["recopie_exacte"] == 1


def test_controle_6_identifiant_et_periode(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    lettre = (
        "Account: Copper Kettle / Juniper Table Services Ltd\n"
        "Requested review period: 2024-01 through 2025-04.\n"
        "Certificate reference GL-CERT-JTS-R4A was first drafted for an annual review with a later four-month update."
    )
    valeur = "GL-CERT-JTS-R4A ; Requested review period: 2024-01 through 2025-04"
    (pieces / "a.md").write_text(lettre + "\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": lettre.split("\n")[0]}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": lettre,
                            "valeur": valeur,
                        }
                    ]
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["valeur_dans_la_lettre"]
    assert ctrl["verdict"] == "ok"
    assert ctrl["ecarts"] == []


def test_controle_6_virgule_presente_dans_csv(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    lettre = "| 1 | 88 L |"
    (pieces / "a.md").write_text(lettre + "\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": lettre}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": lettre,
                            "valeur": "1,88",
                        }
                    ]
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["valeur_dans_la_lettre"]
    assert ctrl["verdict"] == "ok"
    assert ctrl["ecarts"] == []


def test_controle_4_texte_seul_pas_un_ecart(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("paid\nSettled, per ledger\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "paid", "2": "Settled, per ledger"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "quoi": "statut",
                    "comparaison": "concordant",
                    "versions": [
                        {"piece": "a.md", "lignes": [1], "lettre": "paid", "valeur": "paid"},
                        {
                            "piece": "a.md",
                            "lignes": [2],
                            "lettre": "Settled, per ledger",
                            "valeur": "Settled, per ledger",
                        },
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["concordant_est_vrai"]
    assert ctrl["verdict"] == "ok"
    assert ctrl["ecarts"] == []
    assert ctrl["non_verifiables"] == 1
    assert ctrl["motifs"]["texte_seul"] == 1
    assert ctrl["verifie"] == 0


def test_controle_4_serie_quatorze_factures(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    refs = [f"INV-2024-{i:03d}" for i in range(1, 15)]
    (pieces / "a.md").write_text("\n".join(refs) + "\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {str(i): refs[i - 1] for i in range(1, 15)}})
    versions = [
        {"piece": "a.md", "lignes": [i], "lettre": refs[i - 1], "valeur": refs[i - 1]}
        for i in range(1, 15)
    ]
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "quoi": "références de facture",
                    "comparaison": "concordant",
                    "versions": versions,
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["concordant_est_vrai"]
    assert ctrl["verdict"] == "ecart"
    assert ctrl["ecarts"][0]["motif"] == "serie_dans_un_fait"
    assert ctrl["motifs"]["serie_dans_un_fait"] == 1


def test_controle_4_31_380_egal_31_38(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("31.380\n31.38\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "31.380", "2": "31.38"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "quoi": "masse",
                    "comparaison": "concordant",
                    "versions": [
                        {"piece": "a.md", "lignes": [1], "lettre": "31.380", "valeur": "31.380"},
                        {"piece": "a.md", "lignes": [2], "lettre": "31.38", "valeur": "31.38"},
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["concordant_est_vrai"]
    assert ctrl["verdict"] == "ok"
    assert ctrl["ecarts"] == []
    assert ctrl["egalite_zeros_fin"] == 1


def test_controle_8_nombre_commun_rejetee(tmp_path: Path) -> None:
    pieces, sortie = _base(tmp_path)
    (pieces / "a.md").write_text("l1\nl2\n", encoding="utf-8")
    _ecrire_registre(sortie, ["u1", "u2"])
    _ecrire_paquet(sortie, "u1", entite={"a.md": {"1": "l1"}})
    _ecrire_paquet(sortie, "u2", entite={"a.md": {"2": "l2"}})
    _ecrire_entite(
        sortie,
        "u1",
        {
            "faits": [
                {
                    "quoi": "tonnage total mensuel",
                    "concept": "mesure",
                    "comparaison": "une_seule_source",
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [1],
                            "lettre": "l1",
                            "valeur": "10 t le 2024-01",
                        }
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    _ecrire_entite(
        sortie,
        "u2",
        {
            "faits": [
                {
                    "quoi": "tonnage total mensuel du site",
                    "concept": "mesure",
                    "comparaison": "concordant",
                    "versions": [
                        {
                            "piece": "a.md",
                            "lignes": [2],
                            "lettre": "l2",
                            "valeur": "10 t le 2024-01",
                        }
                    ],
                }
            ],
            "regles": [],
            "identites": [],
        },
    )
    ctrl = controler(pieces, sortie)["controles"]["desaccords_possibles_non_releves"]
    assert ctrl["verdict"] == "ok"
    assert ctrl["ecarts"] == []
    assert ctrl["nb_paires"] == 0
