# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — pages Markdown par entité."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts", "src"):
    entry = str(_ROOT / sub)
    if entry not in sys.path:
        sys.path.insert(0, entry)

from banc_realite.pages_entites import generer_pages, rendre_page, regrouper_par_entite


def _ecrire_reconstitution(base: Path) -> None:
    registre = {
        "entites": [
            {"nom": "Alpha SA", "type": "client", "cles": ["Alpha SA"], "par_periode": True},
            {"nom": "Beta Corp", "type": "fournisseur", "cles": ["Beta Corp"], "par_periode": False},
        ],
        "unites": [
            {
                "id": "client-alpha-sa-2025-01_2025-02",
                "entite": "Alpha SA",
                "type": "client",
                "titre": "Alpha SA période 1",
                "cles": ["Alpha SA"],
                "variantes": [],
                "periode": ["2025-01", "2025-02"],
            },
            {
                "id": "fournisseur-beta-corp",
                "entite": "Beta Corp",
                "type": "fournisseur",
                "titre": "Beta Corp document",
                "cles": ["Beta Corp"],
                "variantes": [],
                "periode": [],
            },
            {
                "id": "client-alpha-sa-2025-03_2025-03",
                "entite": "Alpha SA",
                "type": "client",
                "titre": "Alpha SA période 2",
                "cles": ["Alpha SA"],
                "variantes": [],
                "periode": ["2025-03", "2025-03"],
            },
        ],
    }
    (base / "registre.json").write_text(json.dumps(registre, ensure_ascii=False), encoding="utf-8")
    echecs = [
        {
            "unite": "client-alpha-sa-2025-03_2025-03",
            "cause": "ValueError('réponse illisible')",
        }
    ]
    (base / "echecs.json").write_text(json.dumps(echecs, ensure_ascii=False), encoding="utf-8")
    (base / "reemploi.json").write_text("{}", encoding="utf-8")
    entites = base / "entites"
    entites.mkdir(parents=True)
    alpha_obj = {
        "entite": {"nom": "Alpha SA", "type": "client", "identifiants": []},
        "identites": [],
        "faits": [
            {
                "quoi": "Montant convenu",
                "concept": "montant",
                "versions": [
                    {
                        "piece": "facture-a.md",
                        "valeur": "100,00",
                        "statut": "declare",
                        "a_la_date": None,
                        "lignes": [8, 9],
                    }
                ],
            },
            {
                "quoi": "Statut paiement",
                "concept": "statut",
                "versions": [
                    {
                        "piece": "facture-a.md",
                        "valeur": "paye",
                        "statut": "paye",
                        "a_la_date": "2025-01-15",
                        "lignes": [10],
                    },
                    {
                        "piece": "facture-b.md",
                        "valeur": "du",
                        "statut": "du",
                        "a_la_date": "2025-02-01",
                        "lignes": [3],
                    },
                ],
            },
        ],
        "regles": [],
        "non_etabli": [{"point": "Date de clôture", "ce_qui_le_resoudrait": "bilan 2025"}],
        "unite": "client-alpha-sa-2025-01_2025-02",
    }
    (entites / "client-alpha-sa-2025-01_2025-02.json").write_text(
        json.dumps(alpha_obj, ensure_ascii=False), encoding="utf-8"
    )
    beta_obj = {
        "entite": {"nom": "Beta Corp", "type": "fournisseur", "identifiants": []},
        "identites": [
            {
                "variante": "BETA",
                "jugement": "meme",
                "motif": "même en-tête",
                "piece": "devis.md",
                "lignes": [1],
            }
        ],
        "faits": [],
        "regles": [],
        "non_etabli": [],
        "unite": "fournisseur-beta-corp",
    }
    (entites / "fournisseur-beta-corp.json").write_text(
        json.dumps(beta_obj, ensure_ascii=False), encoding="utf-8"
    )


def _mini_ontologie() -> dict:
    return {
        "version": 1,
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
        ],
        "relations": [
            {
                "de_type": "client",
                "verbe": "cite",
                "vers_type": "fournisseur",
                "occurrences": 2,
                "exemple": {
                    "de": "Alpha SA",
                    "vers": "Beta Corp",
                    "piece": "facture-a.md",
                    "lignes": [5],
                },
                "source": "test",
            }
        ],
    }


def test_pages_entites_comportement(tmp_path: Path) -> None:
    r_dir = tmp_path / "r"
    r_dir.mkdir()
    _ecrire_reconstitution(r_dir)
    ontologie = _mini_ontologie()
    date_fixe = "2026-01-01"

    stats1 = generer_pages(r_dir, ontologie, date_gen=date_fixe)
    page_alpha = (r_dir / "pages" / "client-alpha-sa.md").read_text(encoding="utf-8")
    page_beta = (r_dir / "pages" / "fournisseur-beta-corp.md").read_text(encoding="utf-8")
    index = (r_dir / "pages" / "index.md").read_text(encoding="utf-8")

    assert "**Montant convenu** : 100,00 (facture-a.md:l. 8, 9)" in page_alpha
    assert "## Désaccords" in page_alpha
    assert "### Statut paiement" in page_alpha
    assert "| facture-a.md | paye |" in page_alpha
    assert "non reconstituée" in page_alpha
    assert "réponse illisible" in page_alpha

    etat, _, apres_etat = page_alpha.partition("## État compilé")
    bloc_etat = apres_etat.split("## Désaccords")[0]
    assert "Statut paiement" not in bloc_etat
    assert "Montant convenu" in bloc_etat

    assert "cite → Beta Corp" in page_alpha
    assert "l. 1" in page_beta

    lignes_index = [l for l in index.splitlines() if l.startswith("| [")]
    noms = [l.split("](")[0].removeprefix("| [") for l in lignes_index]
    types_par_nom = {e["nom"]: e["type"] for e in ontologie["entites"]}
    assert noms == sorted(noms, key=lambda n: (types_par_nom[n], n))

    stats2 = generer_pages(r_dir, ontologie, date_gen=date_fixe)
    assert stats1 == stats2
    assert (r_dir / "pages" / "client-alpha-sa.md").read_text(encoding="utf-8") == page_alpha

    assert stats1["unites_echouees"] == 1
    assert stats1["pages"] == 2


def test_regrouper_et_rendre_page_minimal(tmp_path: Path) -> None:
    from banc_realite.pages_entites import charger_reconstitution

    r_dir = tmp_path / "r"
    r_dir.mkdir()
    _ecrire_reconstitution(r_dir)
    charge = charger_reconstitution(r_dir)
    groupes = regrouper_par_entite(charge["registre"], charge["objets"], charge["echecs"])
    assert len(groupes["Alpha SA"]["unites"]) == 2
    ontologie = _mini_ontologie()
    texte = rendre_page(
        {"nom": "Alpha SA", "type": "client", "cles": ["Alpha SA"]},
        groupes["Alpha SA"],
        ontologie["relations"],
        1,
        "2026-01-01",
    )
    assert "(facture-a.md:l. 8, 9)" in texte


def test_banniere_perimees_etiquette_jamais_et_index(tmp_path: Path) -> None:
    r_dir = tmp_path / "r"
    r_dir.mkdir()
    _ecrire_reconstitution(r_dir)
    registre = json.loads((r_dir / "registre.json").read_text(encoding="utf-8"))
    for unite in registre["unites"]:
        if unite["entite"] == "Alpha SA":
            unite["perimee"] = {
                "depuis": "2026-01-01T00:00:00+00:00",
                "pieces": ["facture-a.md"],
                "passe": "p",
            }
    (r_dir / "registre.json").write_text(
        json.dumps(registre, ensure_ascii=False), encoding="utf-8"
    )
    ontologie = _mini_ontologie()
    stats = generer_pages(
        r_dir,
        ontologie,
        date_gen="2026-02-02",
        regime_par_entite={"Beta Corp": "jamais"},
    )
    page_alpha = (r_dir / "pages" / "client-alpha-sa.md").read_text(encoding="utf-8")
    page_beta = (r_dir / "pages" / "fournisseur-beta-corp.md").read_text(encoding="utf-8")
    index = (r_dir / "pages" / "index.md").read_text(encoding="utf-8")

    assert "État au 2026-02-02 · 2 unité(s) périmée(s)" in page_alpha
    assert "régime : jamais (pièces qui font foi seules)" in page_beta
    assert "État au" not in page_beta
    assert "| entité | type | unités | faits | désaccords | non établi | périmées |" in index
    ligne_alpha = [l for l in index.splitlines() if l.startswith("| [Alpha SA]")][0]
    assert ligne_alpha.endswith("| 2 |")
    ligne_beta = [l for l in index.splitlines() if l.startswith("| [Beta Corp]")][0]
    assert ligne_beta.endswith("| 0 |")
    assert stats["pages"] == 2


def test_generer_pages_entite_seule_laisse_les_autres(tmp_path: Path) -> None:
    r_dir = tmp_path / "r"
    r_dir.mkdir()
    _ecrire_reconstitution(r_dir)
    ontologie = _mini_ontologie()
    generer_pages(r_dir, ontologie, date_gen="2026-01-01")
    beta_avant = (r_dir / "pages" / "fournisseur-beta-corp.md").read_bytes()
    stats = generer_pages(
        r_dir, ontologie, date_gen="2026-03-03", entites=["Alpha SA"]
    )
    assert stats["pages"] == 1
    assert (r_dir / "pages" / "fournisseur-beta-corp.md").read_bytes() == beta_avant
    alpha = (r_dir / "pages" / "client-alpha-sa.md").read_text(encoding="utf-8")
    assert "généré le 2026-03-03" in alpha
