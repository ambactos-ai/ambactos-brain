# SPDX-License-Identifier: AGPL-3.0-only
"""Tests hors ligne — porte 3 voisinage (issue #135)."""
from __future__ import annotations

import pytest

from banc_realite import etage3_config_gagnante as ecg
from banc_realite.dossiers import dossier_courant
from banc_realite.fournisseurs_contexte import BrainBanc, BrainBancV1, BrainBancV2
from banc_realite.mesure_comparatif_etage2 import K_FIXE, _instancier, k_effectif_fournisseur
from banc_realite.porte_voisinage import (
    FORMES_PASSAGE_ENTIER,
    VOISINAGE_QUOTA_V1,
    VOISINAGE_QUOTA_V2_TOTAL,
    FenetreVoisinageIndex,
    PorteVoisinageInerteError,
    _fenetre_autour_offset,
    _textes_porte2_phrases,
    claim_famille_predicat,
    claim_valeur_forme,
    enrichir_porte_voisinage,
    entites_porte1,
    exiger_porte_voisinage_active,
    familles_predicats_retenues,
    tri_claim_deterministe,
)


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _FakeConnection:
    def __init__(
        self,
        entity_rows=None,
        relation_rows=None,
        target_rows=None,
        seed_rows=None,
    ):
        self.entity_rows = entity_rows or []
        self.relation_rows = relation_rows or []
        self.target_rows = target_rows or []
        self.seed_rows = seed_rows or []
        self.relation_calls = 0
        self.target_calls = 0

    def execute(self, sql, params=()):
        sql_norm = " ".join(sql.split())
        if "claim_id = ANY" in sql_norm:
            return _FakeCursor(self.seed_rows)
        if "predicate = 'relation.about'" in sql_norm:
            self.relation_calls += 1
            return _FakeCursor(self.relation_rows)
        if "entity_id = ANY" in sql_norm and params and params[1] == ["ent-cible"]:
            self.target_calls += 1
            return _FakeCursor(self.target_rows)
        if "entity_id = ANY" in sql_norm:
            return _FakeCursor(self.entity_rows)
        return _FakeCursor([])


def _claim(
    claim_id: str,
    *,
    entity_id: str = "ent-a",
    attribute_id: str = "date_effet",
    value: str = "2024-01-01",
    piece: str = "a.md",
    created_at: str = "2024-01-01",
    line: int = 1,
    vault_ref: str = "greenloop",
    predicate: str = "note.body",
):
    return {
        "claim_id": claim_id,
        "entity_id": entity_id,
        "predicate": predicate,
        "value": value,
        "qualifiers": {"attribute_id": attribute_id},
        "source_file_path": piece,
        "source_span": {"line": line},
        "created_at": created_at,
        "vault_ref": vault_ref,
        "piece": piece,
    }


def _relation(
    claim_id: str,
    *,
    entity_id: str,
    attribute_id: str,
    target_entity: str,
    vault_ref: str = "greenloop",
):
    return _claim(
        claim_id,
        entity_id=entity_id,
        attribute_id=attribute_id,
        value=f"note:{target_entity}",
        predicate="relation.about",
        vault_ref=vault_ref,
    )


def _seed_evidence(claim_id: str = "seed-1", entity_id: str = "ent-a") -> list[dict]:
    return [
        {
            "kind": "claim_hit",
            "id": claim_id,
        },
        {
            "kind": "phrase_hit",
            "id": "phrase-1",
            "text": "fenêtre",
            "source": "a.md",
        },
    ]


def _seed_connection(entity_rows, relation_rows=None, target_rows=None):
    seed = _claim("seed-1", entity_id="ent-a", attribute_id="note_body", value="seed")
    return _FakeConnection(
        entity_rows=entity_rows,
        relation_rows=relation_rows or [],
        target_rows=target_rows or [],
        seed_rows=[seed],
    )


def test_familles_depuis_attribute_id_et_valeur():
    assert familles_predicats_retenues() == ["temporal", "monetary", "status"]
    assert claim_famille_predicat(_claim("1", attribute_id="montant_brut")) == "monetary"
    assert claim_famille_predicat(_claim("2", attribute_id="statut_export")) == "status"
    assert claim_famille_predicat(_claim("3", attribute_id="date_echeance")) == "temporal"
    assert claim_famille_predicat(_claim("4", attribute_id="nom_client", value="Dupont")) is None
    assert claim_famille_predicat(_claim("5", attribute_id="signataire", value="2023-11-30")) == "temporal"
    assert claim_famille_predicat(_claim("6", attribute_id="signataire", value="5")) == "monetary"
    assert claim_valeur_forme(_claim("7", attribute_id="x", value="rejet")) is None
    assert claim_famille_predicat(_relation("r1", entity_id="ent-a", attribute_id="signataire", target_entity="ent-b")) is None


def test_tri_deterministe_date_puis_piece_puis_ligne():
    later = _claim("c2", attribute_id="date_effet", value="2024-02-01", piece="b.md", line=1)
    earlier = _claim("c1", attribute_id="date_effet", value="2024-01-01", piece="a.md", line=9)
    assert tri_claim_deterministe(earlier) < tri_claim_deterministe(later)


def test_v1_quota_familles_dedup_determinisme():
    entity_rows = [
        _claim("c1", attribute_id="date_effet", value="2024-02-01", line=2),
        _claim("c2", attribute_id="montant_brut", value="1200", line=1),
        _claim("c3", attribute_id="statut_export", value="exported", line=1),
        _claim("c4", attribute_id="signataire", value="2024-03-01", line=1),
        _claim("c5", attribute_id="date_signature", value="2024-01-03", line=1),
        _claim("c6", attribute_id="montant_net", value="99", line=1),
        _claim("c7", attribute_id="heures_supplementaires", value="12", line=1),
        _claim("c8", attribute_id="nom_client", value="foo", line=1),
    ]
    conn = _seed_connection(entity_rows)
    evidence = _seed_evidence()
    out1 = enrichir_porte_voisinage(evidence, connection=conn, vault_ref="greenloop", arêtes=1)
    out2 = enrichir_porte_voisinage(evidence, connection=conn, vault_ref="greenloop", arêtes=1)
    voisins1 = [item for item in out1 if item["kind"] == "voisin_hit"]
    voisins2 = [item for item in out2 if item["kind"] == "voisin_hit"]
    assert len(voisins1) == VOISINAGE_QUOTA_V1
    assert [item["id"] for item in voisins1] == [item["id"] for item in voisins2]
    assert "seed-1" not in {item["id"] for item in voisins1}
    assert all(claim_famille_predicat(item["payload"]["claim"]) for item in voisins1)
    assert all(item["payload"]["voisinage"]["arete"] == "entite_partagee" for item in voisins1)


def test_v2_suit_relation_about_et_rend_l_arete():
    entity_rows = [
        _claim(f"c{i}", attribute_id=f"date_{i}", value=f"2024-01-{i:02d}")
        for i in range(1, 8)
    ]
    relation_rows = [
        _relation("r1", entity_id="ent-a", attribute_id="signataire", target_entity="ent-cible"),
    ]
    target_rows = [
        _claim(f"t{i}", entity_id="ent-cible", attribute_id="montant_brut", value=str(100 + i), piece="c.md")
        for i in range(1, 8)
    ]
    conn = _seed_connection(entity_rows, relation_rows, target_rows)
    evidence = _seed_evidence()
    out = enrichir_porte_voisinage(evidence, connection=conn, vault_ref="greenloop", arêtes=2)
    voisins = [item for item in out if item["kind"] == "voisin_hit"]
    assert len(voisins) == VOISINAGE_QUOTA_V2_TOTAL
    assert conn.relation_calls == 1
    assert conn.target_calls == 1
    relation_hits = [item for item in voisins if item["payload"]["voisinage"]["arete"] == "relation.about"]
    assert len(relation_hits) == 6
    assert "—signataire→" in relation_hits[0]["text"] or "—Signataire→" in relation_hits[0]["text"]


def test_entites_uniquement_depuis_porte1():
    conn = _seed_connection([])
    evidence = _seed_evidence()
    assert entites_porte1(evidence, connection=conn, vault_ref="greenloop") == ["ent-a"]


def test_porte_inerte_fait_echouer_la_mesure():
    with pytest.raises(PorteVoisinageInerteError):
        exiger_porte_voisinage_active({"total": 0, "questions_avec": 0}, dossier="greenloop")


def test_hybride_fenetres_g12_inchange_bit_pour_bit():
    assert ecg.PORTE_PHRASE_HYBRIDE_FENETRES_G12 == {
        "etiquette": "hyb_vec_f2_b12_g12",
        "hybride": {"k": 18, "n_tours": 2, "k_tour1": 12},
        "brute_k": 12,
        "phrases": {
            "n_par_passage": 2,
            "max_phrases": 12,
            "mode": "vectoriel",
            "selection": "globale",
            "fenetre": 2,
        },
    }


def test_instancier_brain_banc_v1_v2():
    moreau = dossier_courant("moreau")
    v1 = _instancier("brain_banc_v1", {}, dossier=moreau)
    v2 = _instancier("brain_banc_v2", {}, dossier=moreau)
    assert v1.etiquette == "brain_banc_v1"
    assert v2.etiquette == "brain_banc_v2"
    assert v1._config is ecg.CONFIGURATIONS["hybride_fenetres_g12"]["porte_phrase_config"]
    assert BrainBanc({}, dossier=moreau)._config is v1._config


def test_k_fixe_v1_v2():
    assert "brain_banc_v1" in K_FIXE
    assert "brain_banc_v2" in K_FIXE
    assert k_effectif_fournisseur("brain_banc_v1") == 12
    assert k_effectif_fournisseur("brain_banc_v2") == 12


def test_fournisseurs_v1_v2_utilisent_g12():
    moreau = dossier_courant("moreau")
    assert BrainBancV1({}, dossier=moreau)._config is BrainBanc({}, dossier=moreau)._config
    assert BrainBancV2({}, dossier=moreau)._config is BrainBanc({}, dossier=moreau)._config


def test_fenetre_autour_offset_rend_deux_phrases():
    content = (
        "Madame MOREAU a été engagée le 15 septembre 2019 en CDI à temps partiel (28 h). "
        "Un avenant du 1er mars 2022 la fait passer à temps plein (35 h)."
    )
    phrase, truncated = _fenetre_autour_offset(content, 0, "15 septembre 2019")
    assert "15 septembre 2019" in phrase
    assert "avenant" in phrase.lower()
    assert not truncated


def test_voisin_phrase_rend_fenetre_par_piece_et_ligne():
    passage = (
        "Madame MOREAU a été engagée le 15 septembre 2019 en CDI à temps partiel (28 h). "
        "Un avenant du 1er mars 2022 la fait passer à temps plein (35 h)."
    )
    index = FenetreVoisinageIndex(
        [
            {
                "id": "p1",
                "piece": "contrat.md",
                "content": passage,
                "line_start": 10,
                "line_end": 11,
            }
        ]
    )
    claim = _claim(
        "c-embauche",
        attribute_id="date_embauche",
        value="15 septembre 2019",
        piece="contrat.md",
        line=10,
    )
    entity_rows = [claim]
    conn = _seed_connection(entity_rows)
    evidence = [
        *_seed_evidence(),
        {
            "kind": "phrase_hit",
            "id": "phrase-2",
            "text": "Autre fenêtre déjà rendue par la porte 2.",
            "source": "contrat.md",
        },
    ]
    question = {"id": "q1", "shape": "fait"}
    out = enrichir_porte_voisinage(
        evidence,
        connection=conn,
        vault_ref="greenloop",
        arêtes=1,
        rendu="phrase",
        question=question,
        fenetre_index=index,
    )
    voisins = [item for item in out if item["kind"] == "voisin_hit"]
    assert len(voisins) == 1
    assert "«" in voisins[0]["text"]
    assert "15 septembre 2019" in voisins[0]["text"]
    assert "(contrat.md, ligne 10)" in voisins[0]["text"]
    assert voisins[0]["payload"]["voisinage"]["rendu"] == "phrase"


def test_voisin_phrase_passage_entier_pour_chronologie():
    passage = (
        "Madame MOREAU a été engagée le 15 septembre 2019 en CDI à temps partiel (28 h). "
        "Un avenant du 1er mars 2022 la fait passer à temps plein (35 h)."
    )
    index = FenetreVoisinageIndex(
        [
            {
                "id": "p1",
                "piece": "contrat.md",
                "content": passage,
                "line_start": 1,
                "line_end": 2,
            }
        ]
    )
    claim = _claim(
        "c-embauche",
        attribute_id="date_embauche",
        value="15 septembre 2019",
        piece="contrat.md",
        line=1,
    )
    conn = _seed_connection([claim])
    question = {"id": "q-chrono", "shape": "chronologie"}
    out = enrichir_porte_voisinage(
        _seed_evidence(),
        connection=conn,
        vault_ref="greenloop",
        arêtes=1,
        rendu="phrase",
        question=question,
        fenetre_index=index,
    )
    voisin = next(item for item in out if item["kind"] == "voisin_hit")
    assert "1er mars 2022" in voisin["text"]
    assert "15 septembre 2019" in voisin["text"]


def test_voisin_phrase_dedup_porte2_et_determinisme():
    passage = (
        "Le salaire brut est de 2 400 euros par mois.\n"
        "Les heures supplémentaires sont rémunérées au taux légal.\n"
        "La période d'essai court sur deux mois renouvelable."
    )
    fenetre_porte2 = (
        "Le salaire brut est de 2 400 euros par mois. "
        "Les heures supplémentaires sont rémunérées au taux légal."
    )
    index = FenetreVoisinageIndex(
        [
            {
                "id": "p1",
                "piece": "a.md",
                "content": passage,
                "line_start": 1,
                "line_end": 3,
            }
        ]
    )
    claims = [
        _claim("c1", attribute_id="montant_brut", value="2400", line=1),
        _claim("c2", attribute_id="periode_essai", value="deux mois", line=3),
    ]
    conn = _seed_connection(claims)
    evidence = [
        *_seed_evidence(),
        {"kind": "phrase_hit", "id": "phrase-1", "text": fenetre_porte2, "source": "a.md"},
    ]
    assert fenetre_porte2 in {item["text"] for item in evidence if item["kind"] == "phrase_hit"}
    assert _textes_porte2_phrases(evidence)
    question = {"id": "q1", "shape": "montant"}
    out1 = enrichir_porte_voisinage(
        evidence,
        connection=conn,
        vault_ref="greenloop",
        arêtes=1,
        rendu="phrase",
        question=question,
        fenetre_index=index,
    )
    out2 = enrichir_porte_voisinage(
        evidence,
        connection=conn,
        vault_ref="greenloop",
        arêtes=1,
        rendu="phrase",
        question=question,
        fenetre_index=index,
    )
    voisins1 = [item for item in out1 if item["kind"] == "voisin_hit"]
    voisins2 = [item for item in out2 if item["kind"] == "voisin_hit"]
    assert len(voisins1) == 1
    assert [item["id"] for item in voisins1] == [item["id"] for item in voisins2]
    assert all("2 400" not in item["text"] for item in voisins1)
    assert any("période d'essai" in item["text"] for item in voisins1)


def test_formes_clause_passage_entier():
    assert "chronologie" in FORMES_PASSAGE_ENTIER
    assert "citation" in FORMES_PASSAGE_ENTIER
