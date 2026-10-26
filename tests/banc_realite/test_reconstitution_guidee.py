# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — reconstitution guidée par ontologie et réemploi."""
from __future__ import annotations

import json
from pathlib import Path

from banc_realite.mesure_comparatif_etage2 import (
    _contextes_jsonl_filename,
    _etage2_json_filename,
    contextes_mesure_path,
    etage2_resultat_path,
)
from banc_realite.reconstitution_entites import (
    empreinte_paquet_reemploi,
    index_paquets_par_empreinte,
    rassembler,
    recenser_depuis_ontologie,
)
from banc_realite.run_reconstitution_entites import _executer_unite


def _ecrire_piece(pieces_dir: Path, nom: str, lignes: list[str]) -> None:
    pieces_dir.mkdir(parents=True, exist_ok=True)
    (pieces_dir / nom).write_text("\n".join(lignes) + "\n", encoding="utf-8")


def test_recenser_depuis_ontologie_fusion_et_par_periode(tmp_path: Path) -> None:
    pieces = tmp_path / "pieces"
    _ecrire_piece(
        pieces,
        "dossier.txt",
        [
            "Client Alpha SA opère sur le site Nord.",
            "Facture FAC-001 pour Client Alpha SA en 2025-03.",
            "Contact : Marie Dupont, responsable.",
        ],
    )
    ontologie = {
        "version": 1,
        "decisions": [{"effet": "fusion Client Beta dans Client Alpha SA"}],
        "entites": [
            {
                "nom": "Client Alpha SA",
                "type": "client",
                "cles": ["Client Alpha SA", "FAC-001"],
                "variantes": [],
                "retiree": False,
            },
            {
                "nom": "Client Beta",
                "type": "client",
                "cles": ["Client Beta"],
                "variantes": [],
                "retiree": False,
            },
            {
                "nom": "Marie Dupont",
                "type": "personne",
                "cles": ["Marie Dupont"],
                "variantes": [],
                "retiree": False,
            },
        ],
    }
    ont_path = tmp_path / "ontologie.json"
    ont_path.write_text(json.dumps(ontologie, ensure_ascii=False), encoding="utf-8")
    ref_registre = {
        "entites": [
            {
                "nom": "Client Alpha SA",
                "type": "client",
                "cles": ["Client Alpha SA"],
                "variantes": [],
                "par_periode": True,
            }
        ]
    }
    ref_path = tmp_path / "ref-registre.json"
    ref_path.write_text(json.dumps(ref_registre, ensure_ascii=False), encoding="utf-8")
    sortie = tmp_path / "sortie"
    payload = recenser_depuis_ontologie(
        ontologie_path=ont_path,
        pieces_dir=pieces,
        sortie_dir=sortie,
        registre_reference_path=ref_path,
    )
    alpha = next(e for e in payload["entites"] if e["nom"] == "Client Alpha SA")
    assert "Client Beta" not in {e["nom"] for e in payload["entites"]}
    assert alpha["par_periode"] is True
    personne = next(e for e in payload["entites"] if e["nom"] == "Marie Dupont")
    assert personne["par_periode"] is False
    meta = json.loads((sortie / "recensement.ontologie.json").read_text(encoding="utf-8"))
    assert meta["entites"] == len(payload["entites"])
    assert meta["fusions"]


def test_reemploi_empreinte_sans_appel(tmp_path: Path) -> None:
    pieces = tmp_path / "pieces"
    _ecrire_piece(pieces, "note.txt", ["Ligne stable pour entité TestCo."])
    ancien = tmp_path / "ancien"
    nouveau = tmp_path / "nouveau"
    unite = {
        "id": "client-testco",
        "entite": "TestCo",
        "type": "client",
        "titre": "TestCo",
        "cles": ["TestCo"],
        "variantes": [],
        "periode": [],
    }
    from banc_realite.reconstitution_entites import charger_corpus

    corpus, _ = charger_corpus(pieces)
    paquet = rassembler(unite, corpus)
    (ancien / "paquets").mkdir(parents=True)
    (ancien / "paquets" / "client-testco.json").write_text(
        json.dumps(paquet, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    reconstitution = {
        "entite": {"nom": "TestCo", "type": "client", "identifiants": []},
        "identites": [],
        "faits": [],
        "regles": [],
        "non_etabli": [],
        "unite": "client-testco",
    }
    (ancien / "entites").mkdir(parents=True)
    (ancien / "entites" / "client-testco.json").write_text(
        json.dumps(reconstitution, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    index = index_paquets_par_empreinte(ancien)
    emis_avant = 0

    def chat_fn(_a: str, _b: str):
        nonlocal emis_avant
        emis_avant += 1
        return ("{}", {})

    class Compteur:
        def reserver_emission(self) -> int:
            nonlocal emis_avant
            emis_avant += 1
            return 1

    (nouveau / "paquets").mkdir(parents=True)
    (nouveau / "paquets" / "client-testco.json").write_text(
        json.dumps(paquet, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    _executer_unite(
        unite,
        pieces_dir=pieces,
        sortie_dir=nouveau,
        dossier="alpha_corp",
        chat_fn=chat_fn,
        compteur=Compteur(),
        delai_s=1,
        reutiliser_dir=ancien,
        index_empreinte=index,
        echecs_anciens=set(),
        reemploi_trace={},
    )
    assert emis_avant == 0
    assert (nouveau / "entites" / "client-testco.json").is_file()
    copie = json.loads((nouveau / "entites" / "client-testco.json").read_text(encoding="utf-8"))
    assert copie["unite"] == "client-testco"


def test_paquet_different_declenche_appel(tmp_path: Path) -> None:
    pieces = tmp_path / "pieces"
    _ecrire_piece(pieces, "note.txt", ["Première version TestCo.", "Deuxième ligne TestCo."])
    ancien = tmp_path / "ancien"
    nouveau = tmp_path / "nouveau"
    unite = {
        "id": "client-testco",
        "entite": "TestCo",
        "type": "client",
        "titre": "TestCo",
        "cles": ["TestCo"],
        "variantes": [],
        "periode": [],
    }
    from banc_realite.reconstitution_entites import charger_corpus

    corpus, _ = charger_corpus(pieces)
    paquet_ancien = rassembler(unite, corpus)
    (ancien / "paquets").mkdir(parents=True)
    (ancien / "paquets" / "client-testco.json").write_text(
        json.dumps(paquet_ancien, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    (ancien / "entites").mkdir(parents=True)
    (ancien / "entites" / "client-testco.json").write_text(
        json.dumps({"entite": {"nom": "TestCo"}, "faits": [], "unite": "client-testco"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    paquet_nouveau = dict(paquet_ancien)
    paquet_nouveau["titre"] = "autre titre"
    assert empreinte_paquet_reemploi(paquet_ancien) == empreinte_paquet_reemploi(paquet_nouveau)
    paquet_nouveau["sections"] = {"entite": {"note.txt": {"1": "texte changé"}}}
    assert empreinte_paquet_reemploi(paquet_ancien) != empreinte_paquet_reemploi(paquet_nouveau)
    (nouveau / "paquets").mkdir(parents=True)
    (nouveau / "paquets" / "client-testco.json").write_text(
        json.dumps(paquet_nouveau, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    appels = 0

    def chat_fn(_a: str, _b: str):
        return (
            json.dumps(
                {
                    "entite": {"nom": "TestCo", "type": "client", "identifiants": []},
                    "identites": [],
                    "faits": [],
                    "regles": [],
                    "non_etabli": [],
                }
            ),
            {},
        )

    class Compteur:
        def reserver_emission(self) -> int:
            nonlocal appels
            appels += 1
            return appels

    index = index_paquets_par_empreinte(ancien)
    _executer_unite(
        unite,
        pieces_dir=pieces,
        sortie_dir=nouveau,
        dossier="alpha_corp",
        chat_fn=chat_fn,
        compteur=Compteur(),
        delai_s=1,
        reutiliser_dir=ancien,
        index_empreinte=index,
        echecs_anciens=set(),
        reemploi_trace={},
    )
    assert appels == 1


