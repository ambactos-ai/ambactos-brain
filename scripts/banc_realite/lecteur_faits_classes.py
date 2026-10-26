# SPDX-License-Identifier: AGPL-3.0-only
"""Lecteur MRFS classé : tous les faits de l'instantané, jambes lexicale et vectorielle.

Le domaine n'est pas appelé. Le classement vit dans le banc (ADR 0025 §8, #144).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from banc_realite.compat_moteur import _HYBRID_RRF_K, _rrf

from .constructeur_situations import latest_constructeur_snapshot_id
from .couche_brute_search import build_lexical_tsquery
from .lecteur_situations import (
    ConstructeurSnapshotError,
    _latest_constructeur_snapshot_id,
    _subject_id_for_case,
)
from .mrfs_support import BANC_PG_DSN, MRFS_SCHEMA

ModeClassement = Literal["lexical", "hybride"]

_DIGIT_GROUP = re.compile(r"(?<=\d)\s+(?=\d)")


def empreinte_texte(texte: str) -> str:
    return hashlib.sha256((texte or "").encode("utf-8")).hexdigest()


def _fold(s: str) -> str:
    folded = "".join(
        c
        for c in unicodedata.normalize("NFD", (s or "").lower())
        if unicodedata.category(c) != "Mn"
    )
    return _DIGIT_GROUP.sub("", folded)


def _jetons(texte: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", _fold(texte)) if len(token) >= 2}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True, slots=True)
class Fait:
    id: str
    text: str
    source: str
    line_start: int | None = None
    line_end: int | None = None
    enonciateur: Any = None
    date: Any = None
    alternatives: tuple[dict[str, Any], ...] = ()
    object_id: str = ""
    fact_id: str = ""

    @property
    def empreinte(self) -> str:
        return empreinte_texte(self.text)


@dataclass(frozen=True, slots=True)
class FaitNote:
    fait: Fait
    score: float
    rang_lexical: int | None
    rang_vectoriel: int | None


def _line_key(value: Any) -> int:
    return int(value) if isinstance(value, int) else 0


def _tie_key(fait: Fait) -> tuple[str, int, str]:
    return (fait.source, _line_key(fait.line_start), fait.id)


def _connexion(dsn: str):
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)


def _cell(row: Any, key: str | int, index: int) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    return row[index]


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _fait_depuis_enregistrement(
    record_key: str,
    perspective: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    object_id: str = "",
) -> Fait | None:
    text = str(perspective.get("text") or "").strip()
    source = str(perspective.get("source_file") or identity.get("constructeur_piece") or "").strip()
    if not text or not source:
        return None
    alternatives = tuple(
        dict(alt)
        for alt in (perspective.get("alternatives") or [])
        if isinstance(alt, Mapping)
    )
    line_start = perspective.get("line_start")
    line_end = perspective.get("line_end")
    return Fait(
        id=str(record_key),
        text=text,
        source=source,
        line_start=line_start if isinstance(line_start, int) else None,
        line_end=line_end if isinstance(line_end, int) else None,
        enonciateur=perspective.get("enonciateur"),
        date=perspective.get("date"),
        alternatives=alternatives,
        object_id=str(object_id or ""),
        fact_id=str(identity.get("constructeur_fact_id") or ""),
    )


def charger_faits(
    dossier: str,
    *,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
    connection=None,
) -> list[Fait]:
    """Tous les `proposition_hit` de l'instantané constructeur courant."""
    subject_id = _subject_id_for_case(dossier, dsn=dsn)
    if connection is None:
        snapshot_id = _latest_constructeur_snapshot_id(
            subject_id, dsn=dsn, schema=schema
        )
    else:
        snapshot_id = latest_constructeur_snapshot_id(
            subject_id, dsn=dsn, schema=schema
        )
        if snapshot_id is None:
            raise ConstructeurSnapshotError("aucun instantané constructeur pour ce sujet")
    own = False
    conn = connection
    if conn is None:
        conn = _connexion(dsn)
        own = True
    try:
        rows = conn.execute(
            f"SELECT record_key, object_id, perspective_summary, identity_summary "
            f"FROM {schema}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s "
            "AND hit_kind = 'proposition_hit' "
            "AND record_key LIKE 'constructeur:%%' "
            "ORDER BY record_key",
            (subject_id, snapshot_id),
        ).fetchall()
    finally:
        if own:
            conn.close()
    if not rows:
        raise ConstructeurSnapshotError("instantané constructeur vide")
    faits: list[Fait] = []
    for row in rows:
        perspective = _as_mapping(_cell(row, "perspective_summary", 2))
        identity = _as_mapping(_cell(row, "identity_summary", 3))
        fait = _fait_depuis_enregistrement(
            str(_cell(row, "record_key", 0)),
            perspective,
            identity,
            object_id=str(_cell(row, "object_id", 1) or ""),
        )
        if fait is not None:
            faits.append(fait)
    if not faits:
        raise ConstructeurSnapshotError("instantané constructeur vide")
    return faits


def scorer_lexical_python(faits: Sequence[Fait], question: str) -> dict[str, float]:
    """OU des mots : un fait sans jeton commun n'a pas de rang lexical."""
    q_tokens = _jetons(question)
    if not q_tokens:
        return {}
    scores: dict[str, float] = {}
    for fait in faits:
        overlap = q_tokens & _jetons(fait.text)
        if overlap:
            scores[fait.id] = float(len(overlap))
    return scores


def scorer_lexical_sql(
    connection,
    faits: Sequence[Fait],
    question: str,
) -> dict[str, float]:
    tsq = build_lexical_tsquery(connection, question)
    if not tsq or not faits:
        return {}
    ids = [fait.id for fait in faits]
    texts = [fait.text for fait in faits]
    rows = connection.execute(
        """
        SELECT t.id,
               ts_rank_cd(
                   to_tsvector('french', public.brain_unaccent(t.text)),
                   to_tsquery('simple', %s)
               )::double precision AS rank
        FROM unnest(%s::text[], %s::text[]) AS t(id, text)
        WHERE to_tsvector('french', public.brain_unaccent(t.text))
              @@ to_tsquery('simple', %s)
        """,
        (tsq, ids, texts, tsq),
    ).fetchall()
    scores: dict[str, float] = {}
    for row in rows:
        score = float(_cell(row, "rank", 1))
        if score > 0:
            scores[str(_cell(row, "id", 0))] = score
    return scores


def scores_lexicaux(
    faits: Sequence[Fait],
    question: str,
    *,
    connection=None,
    scorer: Callable[[Sequence[Fait], str], dict[str, float]] | None = None,
) -> dict[str, float]:
    if scorer is not None:
        return scorer(faits, question)
    if connection is None:
        raise RuntimeError(
            "classement lexical sans connexion : passer une connexion au cluster du banc "
            "ou injecter un scorer"
        )
    return scorer_lexical_sql(connection, faits, question)


def scores_vectoriels(
    faits: Sequence[Fait],
    question_vector: Sequence[float],
    embeddings: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    import numpy as np

    vectors = []
    for fait in faits:
        vector = embeddings.get(fait.empreinte)
        if vector is None:
            raise RuntimeError(
                "embedding de fait absent "
                f"({fait.source}:{fait.line_start} {fait.empreinte[:12]}…)"
            )
        vectors.append(vector)
    if not vectors:
        return {}
    matrix = np.asarray(vectors, dtype=np.float64)
    query = np.asarray(question_vector, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    values = np.divide(dots, norms, out=np.zeros_like(dots), where=norms != 0.0)
    return {fait.id: float(value) for fait, value in zip(faits, values)}


def _ordonner(scores: Mapping[str, float], par_id: Mapping[str, Fait]) -> list[Fait]:
    return sorted(
        (par_id[ident] for ident in scores if ident in par_id),
        key=lambda fait: (-float(scores[fait.id]), *_tie_key(fait)),
    )


def _rangs(ordered: Sequence[Fait]) -> dict[str, int]:
    return {fait.id: index for index, fait in enumerate(ordered, start=1)}


def _items_rrf(ordered: Sequence[Fait]) -> list[dict[str, Any]]:
    return [
        {
            "id": fait.id,
            "source": fait.source,
            "line_start": _line_key(fait.line_start),
        }
        for fait in ordered
    ]


def classer_faits(
    faits: Sequence[Fait],
    question: str,
    *,
    mode: ModeClassement = "hybride",
    connection=None,
    scorer_lexical: Callable[[Sequence[Fait], str], dict[str, float]] | None = None,
    question_vector: Sequence[float] | None = None,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    scores_lex: Mapping[str, float] | None = None,
    scores_vec: Mapping[str, float] | None = None,
) -> list[FaitNote]:
    """Note tous les faits. Pas de vivier, pas de seuil, pas de k."""
    par_id = {fait.id: fait for fait in faits}
    if scores_lex is None:
        scores_lex = scores_lexicaux(
            faits, question, connection=connection, scorer=scorer_lexical
        )
    lex_ordered = _ordonner(scores_lex, par_id)
    lex_ranks = _rangs(lex_ordered)

    vec_ordered: list[Fait] = []
    vec_ranks: dict[str, int] = {}
    if mode == "hybride":
        if scores_vec is None:
            if question_vector is None:
                raise RuntimeError("embedding de question absent")
            if embeddings is None:
                raise RuntimeError("embeddings de faits absents")
            scores_vec = scores_vectoriels(faits, question_vector, embeddings)
        vec_ordered = _ordonner(scores_vec, par_id)
        vec_ranks = _rangs(vec_ordered)

    if mode == "lexical":
        return [
            FaitNote(
                fait=fait,
                score=float(scores_lex[fait.id]),
                rang_lexical=lex_ranks[fait.id],
                rang_vectoriel=None,
            )
            for fait in lex_ordered
        ]

    fused = _rrf(
        [_items_rrf(lex_ordered), _items_rrf(vec_ordered)],
        key_fn=lambda item: item["id"],
        k=_HYBRID_RRF_K,
    )
    fused = sorted(
        fused,
        key=lambda item: (
            -float(item["score"]),
            str(item.get("source") or ""),
            int(item.get("line_start") or 0),
            str(item["id"]),
        ),
    )
    notes: list[FaitNote] = []
    for item in fused:
        fait = par_id[str(item["id"])]
        notes.append(
            FaitNote(
                fait=fait,
                score=float(item["score"]),
                rang_lexical=lex_ranks.get(fait.id),
                rang_vectoriel=vec_ranks.get(fait.id),
            )
        )
    return notes


def fait_vers_payload(fait: Fait, *, rank: int, score: float) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "source_file_path": fait.source,
        "mrfs_object_kind": "proposition_hit",
    }
    if fait.object_id:
        provenance["mrfs_object_id"] = fait.object_id
    if isinstance(fait.line_start, int):
        provenance["line_start"] = fait.line_start
    if isinstance(fait.line_end, int):
        provenance["line_end"] = fait.line_end
    return {
        "kind": "mrfs_situation",
        "rank": rank,
        "score": float(score),
        "text": fait.text,
        "source": fait.source,
        "enonciateur": fait.enonciateur,
        "date": fait.date,
        "alternatives": [dict(alt) for alt in fait.alternatives],
        "provenance": provenance,
    }


def groupes_de_faits(faits: Sequence[Fait]) -> dict[str, str]:
    """Identifiant de fait → représentant de son groupe de rapprochement.

    Deux faits sont du même groupe dès que l'un cite l'autre en alternative. Les
    composantes sont calculées sur tous les faits : deux membres d'un groupe qui portent
    la même citation ne se citent pas l'un l'autre, mais un troisième les cite tous deux.
    """
    parent = {fait.id: fait.id for fait in faits}

    def find(ident: str) -> str:
        while parent[ident] != ident:
            parent[ident] = parent[parent[ident]]
            ident = parent[ident]
        return ident

    par_cle: dict[tuple[str, int, str], list[str]] = {}
    for fait in faits:
        par_cle.setdefault(
            (fait.source, _line_key(fait.line_start), fait.text), []
        ).append(fait.id)
    for fait in faits:
        for alt in fait.alternatives:
            cle = (
                str(alt.get("source_file") or alt.get("source") or ""),
                _line_key(alt.get("line_start")),
                str(alt.get("citation") or alt.get("text") or ""),
            )
            for autre in par_cle.get(cle, ()):
                a, b = find(fait.id), find(autre)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return {ident: find(ident) for ident in parent}


def rendre_par_groupe(
    notes: Iterable[FaitNote], *, faits: Sequence[Fait] | None = None
) -> list[dict[str, Any]]:
    """Un élément par groupe : les membres suivants sont sautés."""
    from .fournisseurs_contexte import formater_element_mrfs

    notes = list(notes)
    groupes = groupes_de_faits(faits if faits is not None else [n.fait for n in notes])
    seen: set[str] = set()
    elements: list[dict[str, Any]] = []
    rank = 0
    for note in notes:
        cle = groupes.get(note.fait.id, note.fait.id)
        if cle in seen:
            continue
        seen.add(cle)
        rank += 1
        elements.append(
            formater_element_mrfs(
                fait_vers_payload(note.fait, rank=rank, score=note.score),
                rank=rank,
            )
        )
    return elements


def contexte_faits_classes(
    question: str,
    *,
    dossier: str,
    mode: ModeClassement = "hybride",
    dsn: str = BANC_PG_DSN,
    connection=None,
    faits: Sequence[Fait] | None = None,
    question_vector: Sequence[float] | None = None,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    scorer_lexical: Callable[[Sequence[Fait], str], dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    own = False
    conn = connection
    loaded = list(faits) if faits is not None else None
    if loaded is None and conn is None:
        conn = _connexion(dsn)
        own = True
    try:
        if loaded is None:
            loaded = charger_faits(dossier, dsn=dsn, connection=conn)
        notes = classer_faits(
            loaded,
            question,
            mode=mode,
            connection=conn,
            scorer_lexical=scorer_lexical,
            question_vector=question_vector,
            embeddings=embeddings,
        )
        return rendre_par_groupe(notes, faits=loaded)
    finally:
        if own and conn is not None:
            conn.close()
