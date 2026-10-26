# SPDX-License-Identifier: AGPL-3.0-only
"""Embeddings des objets de reconstitution — à blanc, cache, plafond, garde pytest."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from banc_realite.embeddings_faits import (
    AppelEmbeddingsInterdit,
    BATCH_SIZE,
    MODEL_ID,
)
from banc_realite.embeddings_reconstitution import (
    compter_objets_a_blanc,
    embarquer_objets,
)
from banc_realite.plafond_appels import PlafondAtteint
from banc_realite.recherche_entites import LigneObjet, Objet


def _objet(text: str, ident: str, unite_id: str = "u") -> Objet:
    return Objet(
        id=ident,
        unite_id=unite_id,
        sorte="fait",
        comparaison="concordant",
        lignes=(LigneObjet(piece="a.md", numero=1, lettre=text),),
        texte_recherche=text,
    )


def test_a_blanc_sans_appel(tmp_path: Path) -> None:
    objets = [_objet(f"lettre {index} " * 20, f"u#f{index}") for index in range(5)]
    calls: list[list[str]] = []

    def embed_fn(textes: list[str]) -> list[list[float]]:
        calls.append(list(textes))
        raise AssertionError("aucun appel en à-blanc")

    stats = embarquer_objets(
        "moreau",
        a_blanc=True,
        reconstitution_dir=tmp_path / "entites",
        situations_root=tmp_path,
        objets=objets,
        cache={},
        embed_fn=embed_fn,
        batch_size=2,
    )
    assert calls == []
    assert stats["dry_run"] is True
    assert stats["n_objets"] == 5
    assert stats["n_a_embarquer"] == 5
    assert stats["n_requetes"] == 3
    assert stats["n_faits"] == 5
    assert stats["modele"] == MODEL_ID
    assert not (tmp_path / "compteur-embeddings.json").exists()


def test_plafond_compte_avant_la_requete(tmp_path: Path) -> None:
    objets = [_objet(f"texte distinct {index} abcdef", f"u#f{index}") for index in range(3)]
    seen_emis: list[int] = []

    def embed_fn(textes: list[str]) -> list[list[float]]:
        payload = json.loads((tmp_path / "compteur-embeddings.json").read_text(encoding="utf-8"))
        seen_emis.append(int(payload["emis"]))
        assert payload["emis"] >= 1
        return [[0.1, 0.2] for _ in textes]

    with pytest.raises(PlafondAtteint):
        embarquer_objets(
            "moreau",
            a_blanc=False,
            reconstitution_dir=tmp_path / "entites",
            situations_root=tmp_path,
            objets=objets,
            cache={},
            embed_fn=embed_fn,
            plafond=1,
            batch_size=2,
        )
    assert seen_emis == [1]
    journal = (tmp_path / "appels-embeddings.jsonl").read_text(encoding="utf-8")
    assert "mistral" in journal
    assert MODEL_ID in journal


def test_cache_relu_sans_appel(tmp_path: Path) -> None:
    objets = [_objet("une lettre déjà embarquée ici", "u#f1")]
    cache = {objets[0].empreinte: [0.1, 0.2, 0.3]}

    def embed_fn(textes: list[str]) -> list[list[float]]:
        raise AssertionError("cache hit : pas d'appel")

    stats = embarquer_objets(
        "moreau",
        a_blanc=False,
        reconstitution_dir=tmp_path / "entites",
        situations_root=tmp_path,
        objets=objets,
        cache=cache,
        embed_fn=embed_fn,
    )
    assert stats["n_a_embarquer"] == 0
    assert stats["n_requetes_emises"] == 0
    assert stats["n_deja_cache"] == 1


def test_pytest_interdit_appel_reel(tmp_path: Path) -> None:
    objets = [_objet("lettre à embarquer maintenant", "u#f1")]
    assert os.environ.get("PYTEST_CURRENT_TEST")
    with pytest.raises(AppelEmbeddingsInterdit, match="interdit sous pytest"):
        embarquer_objets(
            "moreau",
            a_blanc=False,
            reconstitution_dir=tmp_path / "entites",
            situations_root=tmp_path,
            objets=objets,
            cache={},
            embed_fn=None,
        )


def test_cache_ecrit_apres_chaque_requete(tmp_path: Path) -> None:
    reconstitution = tmp_path / "entites"
    objets = [_objet(f"texte distinct {index} abcdef", f"u#f{index}") for index in range(3)]
    ecritures: list[int] = []

    def embed_fn(textes: list[str]) -> list[list[float]]:
        cache_path = reconstitution / "embeddings-objets.json"
        if cache_path.is_file():
            ecritures.append(len(json.loads(cache_path.read_text(encoding="utf-8"))))
        else:
            ecritures.append(0)
        return [[0.1, 0.2] for _ in textes]

    embarquer_objets(
        "moreau",
        a_blanc=False,
        reconstitution_dir=reconstitution,
        situations_root=tmp_path,
        objets=objets,
        cache={},
        embed_fn=embed_fn,
        batch_size=2,
    )
    cache = json.loads((reconstitution / "embeddings-objets.json").read_text(encoding="utf-8"))
    assert len(cache) == 3
    assert ecritures == [0, 2]


def test_lots_bornes_a_64_textes_et_24000_caracteres(tmp_path: Path) -> None:
    # Chaque texte est borné à 1 500 caractères envoyés : seize textes font 24 000.
    gros = [_objet(("x" * 10_000) + str(index), f"u#f{index}") for index in range(17)]
    stats = compter_objets_a_blanc(gros, cache={}, batch_size=BATCH_SIZE)
    assert stats["n_requetes"] == 2
    assert stats["n_a_embarquer"] == 17

    appels: list[int] = []

    def embed_fn(textes: list[str]) -> list[list[float]]:
        appels.append(len(textes))
        assert sum(len(texte) for texte in textes) <= 24_000
        return [[0.1] for _ in textes]

    embarquer_objets(
        "moreau",
        a_blanc=False,
        reconstitution_dir=tmp_path / "entites",
        situations_root=tmp_path,
        objets=gros,
        cache={},
        embed_fn=embed_fn,
        batch_size=64,
    )
    assert appels == [16, 1]


def test_texte_envoye_borne_a_1500(tmp_path: Path) -> None:
    objet = _objet("y" * 13_000, "u#f1")
    recus: list[int] = []

    def embed_fn(textes: list[str]) -> list[list[float]]:
        recus.extend(len(texte) for texte in textes)
        return [[0.1] for _ in textes]

    embarquer_objets(
        "moreau",
        a_blanc=False,
        reconstitution_dir=tmp_path / "entites",
        situations_root=tmp_path,
        objets=[objet],
        cache={},
        embed_fn=embed_fn,
    )
    assert recus == [1_500]


def test_cache_fusionne_deux_unites_concurrentes(tmp_path: Path) -> None:
    reconstitution = tmp_path / "reconstitution"
    reconstitution.mkdir()
    objet_a = _objet("vecteur distinct unité alpha abcdef", "a#f1", unite_id="u-a")
    objet_b = _objet("vecteur distinct unité beta ghijkl", "b#f1", unite_id="u-b")
    objets = [objet_a, objet_b]
    a_dans_embed = threading.Event()
    b_a_ecrit = threading.Event()
    erreurs: list[str] = []

    def embed_a(textes: list[str]) -> list[list[float]]:
        a_dans_embed.set()
        assert b_a_ecrit.wait(timeout=5)
        return [[1.0, 0.0] for _ in textes]

    def embed_b(textes: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in textes]

    def run_a() -> None:
        try:
            embarquer_objets(
                "moreau",
                a_blanc=False,
                reconstitution_dir=reconstitution,
                situations_root=tmp_path,
                objets=objets,
                unites=["u-a"],
                embed_fn=embed_a,
            )
        except Exception as exc:
            erreurs.append(repr(exc))

    def run_b() -> None:
        try:
            assert a_dans_embed.wait(timeout=5)
            embarquer_objets(
                "moreau",
                a_blanc=False,
                reconstitution_dir=reconstitution,
                situations_root=tmp_path,
                objets=objets,
                unites=["u-b"],
                embed_fn=embed_b,
            )
            b_a_ecrit.set()
        except Exception as exc:
            erreurs.append(repr(exc))
            b_a_ecrit.set()

    t_a = threading.Thread(target=run_a)
    t_b = threading.Thread(target=run_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=8)
    t_b.join(timeout=8)
    assert not t_a.is_alive() and not t_b.is_alive()
    assert erreurs == []
    cache_path = reconstitution / "embeddings-objets.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert objet_a.empreinte in cache
    assert objet_b.empreinte in cache
    assert cache[objet_a.empreinte] == [1.0, 0.0]
    assert cache[objet_b.empreinte] == [0.0, 1.0]
    assert (reconstitution / "embeddings-objets.lock").is_file()
