# SPDX-License-Identifier: AGPL-3.0-only
"""Couche brute MOREAU — découpage, embeddings Mistral, couverture citations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from banc_realite.compat_moteur import MODEL_ID, MistralEmbeddingError, embed_with_retry, to_pgvector

from .couche_brute_schema import apply_schema
from .dossiers import BANC_VAR, Dossier, dossier_courant
from .embarquer_phrases import FORMULE_COUT, MISTRAL_EMBED_USD_PER_TOKEN
from .moreau_or import normalize_verbatim, verify_quote
from .normalize import piece_basename
from .or_dossier import load_gold
from .decoupe_dossier import parametres_decoupe_dossier, parametres_decoupe_greenloop
from .passages import (
    OVERLAP_CHARS,
    MAX_CHARS,
    Passage,
    chunk_corpus,
    compter_pieces_un_seul_passage,
    lignes_non_couvertes,
    stats_duplication,
)

MAX_EMBED_CALLS = 300
COUVERTURE_RAPPORT = (BANC_VAR / 'sorties' / 'comparatif' / 'COUVERTURE-DECOUPE.md')


@dataclass(frozen=True, slots=True)
class CoucheBruteChemins:
    """Artefacts couche brute — tous les chemins dérivés du dossier passé."""

    dossier: Dossier
    sorties: Path

    @classmethod
    def pour_dossier(
        cls,
        dossier: Dossier | str,
        *,
        sorties_dir: Path | None = None,
    ) -> CoucheBruteChemins:
        d = dossier_courant(dossier) if isinstance(dossier, str) else dossier
        sorties = (
            sorties_dir
            if sorties_dir is not None
            else BANC_VAR / "dossiers" / d.nom / "couche-brute"
        )
        return cls(dossier=d, sorties=sorties)

    @property
    def embed_log(self) -> Path:
        return self.sorties / "embedding-appels.jsonl"

    @property
    def reprise_log(self) -> Path:
        return self.sorties / "ingestion-reprise.jsonl"

    @property
    def question_embed_cache(self) -> Path:
        return self.sorties / "etage2-question-embeddings.json"

    @property
    def rapport(self) -> Path:
        return self.sorties / "RAPPORT-COUCHE-BRUTE.md"

    @property
    def decoupe_config(self) -> Path:
        return self.sorties / "decoupe-config.json"

    @property
    def build_json(self) -> Path:
        return self.sorties / "build.json"


class CouvertureDecoupeError(RuntimeError):
    """Porte — une citation de l'or n'est contenue entière par aucun passage."""


def passage_key(piece: str, line_start: int, line_end: int) -> str:
    return f"{piece}#{line_start}-{line_end}"


def embed_cache_path(dossier_nom: str = "moreau") -> Path:
    if dossier_nom == "greenloop":
        from .config import SORTIES

        return SORTIES / "etage2-question-embeddings.json"
    return CoucheBruteChemins.pour_dossier(dossier_nom).question_embed_cache


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_reprise_done(chemins: CoucheBruteChemins) -> set[str]:
    done: set[str] = set()
    if not chemins.reprise_log.is_file():
        return done
    for line in chemins.reprise_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        passage = str(row.get("passage") or "")
        if passage:
            done.add(passage)
    return done


def _load_db_done(connection, vault_ref: str) -> set[str]:
    rows = connection.execute(
        """
        SELECT source_ref, metadata
        FROM public.memory_chunks
        WHERE source_type = 'corpus'
          AND vault_ref = %s
        """,
        (vault_ref,),
    ).fetchall()
    done: set[str] = set()
    for row in rows:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        piece = row.get("source_ref") or meta.get("piece") or ""
        line_start = int(meta.get("line_start") or 0)
        line_end = int(meta.get("line_end") or 0)
        if piece and line_start and line_end:
            done.add(passage_key(piece, line_start, line_end))
    return done


SQL_DELETE_PIECE = """
DELETE FROM public.memory_chunks
WHERE source_type = 'corpus' AND vault_ref = %s AND source_ref = %s
"""

SQL_INSERT_PASSAGE = """
INSERT INTO public.memory_chunks (
    source_type, source_ref, content, metadata, embedding_mistral, vault_ref
) VALUES ('corpus', %s, %s, %s::jsonb, %s::extensions.vector, %s)
"""

SQL_INDEX_TEXTE_VECTEUR = """
SELECT content, embedding_mistral
FROM public.memory_chunks
WHERE source_type = 'corpus' AND vault_ref = %s
"""

SQL_LISTER_PASSAGES = """
SELECT id, source_ref, content, metadata
FROM public.memory_chunks
WHERE source_type = 'corpus' AND vault_ref = %s
ORDER BY id
"""

SQL_LISTER_PASSAGES_PIECE = """
SELECT id, source_ref, content, metadata
FROM public.memory_chunks
WHERE source_type = 'corpus' AND vault_ref = %s AND source_ref = %s
ORDER BY id
"""


def sha256_texte(texte: str) -> str:
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def _vecteur_depuis_cellule(brut: Any) -> list[float] | None:
    if brut is None:
        return None
    if isinstance(brut, list):
        return [float(x) for x in brut]
    if hasattr(brut, "tolist"):
        return [float(x) for x in brut.tolist()]
    if isinstance(brut, str):
        texte = brut.strip()
        if texte.startswith("[") and texte.endswith("]"):
            interieur = texte[1:-1].strip()
            if not interieur:
                return []
            return [float(x) for x in interieur.split(",")]
    return None


@contextmanager
def _transaction(connection) -> Iterator[None]:
    txn = getattr(connection, "transaction", None)
    if callable(txn):
        with txn():
            yield
        return
    connection.execute("BEGIN")
    try:
        yield
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


def charger_index_texte_vecteur(connection, vault_ref: str) -> dict[str, list[float]]:
    """Index sha256(texte) → vecteur déjà stocké pour ce tenant."""
    rows = connection.execute(SQL_INDEX_TEXTE_VECTEUR, (vault_ref,)).fetchall()
    index: dict[str, list[float]] = {}
    for row in rows:
        contenu = str(row.get("content") or "")
        vecteur = _vecteur_depuis_cellule(row.get("embedding_mistral"))
        if not contenu or vecteur is None:
            continue
        index.setdefault(sha256_texte(contenu), vecteur)
    return index


def lister_passages_corpus(
    connection,
    vault_ref: str,
    piece: str | None = None,
) -> list[dict[str, Any]]:
    if piece is None:
        rows = connection.execute(SQL_LISTER_PASSAGES, (vault_ref,)).fetchall()
    else:
        rows = connection.execute(SQL_LISTER_PASSAGES_PIECE, (vault_ref, piece)).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        out.append(
            {
                "id": row.get("id"),
                "source_ref": row.get("source_ref") or "",
                "content": row.get("content") or "",
                "metadata": meta,
            }
        )
    return out


def supprimer_passages_piece(connection, vault_ref: str, piece: str) -> int:
    n = len(lister_passages_corpus(connection, vault_ref, piece))
    with _transaction(connection):
        connection.execute(SQL_DELETE_PIECE, (vault_ref, piece))
    return n


def remplacer_passages_piece(
    connection,
    *,
    vault_ref: str,
    piece: str,
    passages: list[Passage],
    embed_fn: Callable[[str], list[float]],
    index_texte_vecteur: dict[str, list[float]],
    reserver_emission: Callable[[], Any] | None = None,
    ecrire: bool = True,
) -> dict[str, Any]:
    """Remplace les passages d'une pièce dans une transaction.

    Les embeddings sont préparés avant le DELETE : un plafond atteint laisse
    la pièce intacte. ``ecrire=False`` calcule sans toucher à la base.
    """
    vecteurs: list[list[float]] = []
    emis = 0
    reutilises = 0
    for passage in passages:
        cle = sha256_texte(passage.content)
        connu = index_texte_vecteur.get(cle)
        if connu is not None:
            vecteurs.append(connu)
            reutilises += 1
            continue
        if reserver_emission is not None:
            reserver_emission()
        vecteur = embed_fn(passage.content)
        index_texte_vecteur[cle] = vecteur
        vecteurs.append(vecteur)
        emis += 1

    if ecrire:
        with _transaction(connection):
            connection.execute(SQL_DELETE_PIECE, (vault_ref, piece))
            for passage, vecteur in zip(passages, vecteurs, strict=True):
                connection.execute(
                    SQL_INSERT_PASSAGE,
                    (
                        passage.piece,
                        passage.content,
                        json.dumps(passage.to_metadata(), ensure_ascii=False),
                        to_pgvector(vecteur),
                        vault_ref,
                    ),
                )

    return {
        "passages_remplaces": len(passages),
        "embeddings_emis": emis,
        "embeddings_reutilises": reutilises,
    }


def _append_embed_log(chemins: CoucheBruteChemins, entry: dict[str, Any]) -> None:
    _append_jsonl(chemins.embed_log, entry)


def _journaliser_reprise(chemins: CoucheBruteChemins, piece: str, passage: Passage) -> None:
    _append_jsonl(
        chemins.reprise_log,
        {
            "piece": piece,
            "passage": passage_key(piece, passage.line_start, passage.line_end),
            "line_start": passage.line_start,
            "line_end": passage.line_end,
        },
    )


def _embed_text(
    text: str,
    *,
    call_no: int,
    chemins: CoucheBruteChemins,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[float]:
    if embed_fn is not None:
        vector = embed_fn(text)
        _append_embed_log(
            chemins,
            {"appel": call_no, "chars": len(text), "tentative": 1, "status": "ok"},
        )
        return vector
    for attempt in range(10):
        try:
            vector = embed_with_retry(text, attempts=1)
            _append_embed_log(
                chemins,
                {
                    "appel": call_no,
                    "chars": len(text),
                    "tentative": attempt + 1,
                    "status": "ok",
                },
            )
            return vector
        except MistralEmbeddingError as exc:
            message = str(exc)
            _append_embed_log(
                chemins,
                {
                    "appel": call_no,
                    "chars": len(text),
                    "tentative": attempt + 1,
                    "status": "erreur",
                    "message": message[:200],
                },
            )
            if "HTTP 429" in message and attempt < 9:
                time.sleep(30)
                continue
            raise
    raise MistralEmbeddingError("échec embedding après 10 tentatives 429")


def controle_couverture_citations(
    passages: list[Passage],
    assertions: list[dict[str, Any]],
    *,
    corpus_dir: Path | None = None,
) -> dict[str, Any]:
    """Vérifie que chaque citation gold apparaît verbatim dans au moins un passage."""
    by_piece: dict[str, list[Passage]] = {}
    for passage in passages:
        piece_key = piece_basename(passage.piece)
        by_piece.setdefault(piece_key, []).append(passage)

    piece_text_cache: dict[str, str] = {}

    def piece_text(piece: str) -> str:
        if piece not in piece_text_cache:
            if corpus_dir is not None:
                path = corpus_dir / piece
                piece_text_cache[piece] = path.read_text(encoding="utf-8") if path.is_file() else ""
            else:
                piece_text_cache[piece] = "\n".join(p.content for p in by_piece.get(piece, []))
        return piece_text_cache[piece]

    retrouvees: set[str] = set()
    manquantes: list[dict[str, Any]] = []
    coupees: list[dict[str, Any]] = []

    for assertion in assertions:
        quote = str(assertion.get("quote") or "")
        piece = piece_basename(
            str(assertion.get("piece") or assertion.get("source") or "")
        )
        aid = str(assertion.get("id") or "")
        if not quote or not piece:
            manquantes.append(
                {
                    "id": aid,
                    "piece": piece,
                    "cause": "assertion_incomplete",
                }
            )
            continue
        norm_quote = normalize_verbatim(quote)
        piece_passages = by_piece.get(piece, [])
        in_single = any(
            norm_quote in normalize_verbatim(p.content) for p in piece_passages
        )
        if in_single:
            retrouvees.add(aid)
            continue
        if verify_quote(quote, piece_text(piece)):
            coupees.append(
                {
                    "id": aid,
                    "piece": piece,
                    "cause": "citation_coupee_par_decoupe",
                    "quote_debut": quote[:80],
                }
            )
        else:
            manquantes.append(
                {
                    "id": aid,
                    "piece": piece,
                    "cause": "citation_absente",
                    "quote_debut": quote[:80],
                }
            )

    total = len(assertions)
    found_count = len(retrouvees)
    all_missing = manquantes + coupees
    manquantes_par_piece: dict[str, int] = {}
    for item in all_missing:
        piece_name = str(item.get("piece") or "")
        if piece_name:
            manquantes_par_piece[piece_name] = manquantes_par_piece.get(piece_name, 0) + 1
    return {
        "citations_total": total,
        "citations_retrouvees": f"{found_count}/{total}",
        "citations_retrouvees_ids": sorted(retrouvees),
        "citations_manquantes": all_missing,
        "citations_coupees": coupees,
        "citations_absentes": manquantes,
        "manquantes_par_piece": manquantes_par_piece,
    }


def verifier_couverture_citations(result: dict[str, Any]) -> None:
    """Porte — refuse la construction si une citation n'est pas couverte."""
    manquantes = result.get("citations_manquantes") or []
    if not manquantes:
        return
    noms = ", ".join(
        f"{item['id']} ({item.get('cause', '?')})" for item in manquantes[:20]
    )
    extra = ""
    if len(manquantes) > 20:
        extra = f" … et {len(manquantes) - 20} autres"
    raise CouvertureDecoupeError(
        f"Couverture citations incomplète — construction refusée : {noms}{extra}"
    )


def verifier_couverture_lignes(manquantes: list[tuple[str, int]]) -> None:
    """Porte — refuse la construction si une ligne source n'apparaît dans aucun passage."""
    if not manquantes:
        return
    noms = ", ".join(f"{piece}:{line_no}" for piece, line_no in manquantes[:10])
    extra = ""
    if len(manquantes) > 10:
        extra = f" … et {len(manquantes) - 10} autres"
    raise CouvertureDecoupeError(
        "Lignes hors passage — construction refusée "
        f"(cause ligne_hors_passage) : {noms}{extra}"
    )


def _passages_depuis_db(connection, vault_ref: str) -> list[Passage]:
    rows = lister_passages_dossier(connection, vault_ref)
    return [
        Passage(
            piece=str(row.get("source_ref") or ""),
            content=str(row.get("content") or ""),
            line_start=1,
            line_end=1,
            section_path="",
        )
        for row in rows
    ]


def controle_couverture_retroactif(
    connection,
    dossier_nom: str,
) -> dict[str, Any]:
    """Contrôle de couverture sur une couche brute déjà construite, sans la reconstruire."""
    from .dossiers import dossier_courant

    dossier = dossier_courant(dossier_nom)
    gold = load_gold(dossier)
    passages = _passages_depuis_db(connection, dossier.vault_ref)
    result = controle_couverture_citations(
        passages,
        gold["assertions"],
        corpus_dir=dossier.corpus_dir,
    )
    decoupe = parametres_decoupe_dossier(dossier, gold["assertions"])
    return {
        "dossier": dossier.nom,
        "retroactif": True,
        "passages_en_base": len(passages),
        "recouvrement_requis": decoupe,
        **result,
    }


def _stats_par_piece(
    passages: list[Passage],
    appels_par_piece: dict[str, int],
) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for passage in passages:
        bucket = stats.setdefault(
            passage.piece,
            {"piece": passage.piece, "passages": 0, "caracteres": 0, "appels_embedding": 0},
        )
        bucket["passages"] += 1
        bucket["caracteres"] += len(passage.content)
    for piece, appels in appels_par_piece.items():
        if piece not in stats:
            stats[piece] = {
                "piece": piece,
                "passages": 0,
                "caracteres": 0,
                "appels_embedding": 0,
            }
        stats[piece]["appels_embedding"] = appels
    return [stats[k] for k in sorted(stats)]


def _estimer_cout(appels: int, caracteres: int) -> float:
    tokens = caracteres / 4
    return round(tokens * MISTRAL_EMBED_USD_PER_TOKEN, 6)


def _lire_decoupe_config(chemins: CoucheBruteChemins) -> dict[str, Any] | None:
    if not chemins.decoupe_config.is_file():
        return None
    return json.loads(chemins.decoupe_config.read_text(encoding="utf-8"))


def _ecrire_decoupe_config(chemins: CoucheBruteChemins, config: dict[str, Any]) -> None:
    chemins.sorties.mkdir(parents=True, exist_ok=True)
    chemins.decoupe_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def regenerer_decoupe_config(
    dossier: Dossier | str = "moreau",
    *,
    sorties_dir: Path | None = None,
) -> dict[str, Any]:
    """Régénère decoupe-config.json depuis l'or, sans toucher à la base."""
    chemins = CoucheBruteChemins.pour_dossier(dossier, sorties_dir=sorties_dir)
    gold = load_gold(chemins.dossier)
    decoupe = parametres_decoupe_dossier(chemins.dossier, gold["assertions"])
    _ecrire_decoupe_config(chemins, decoupe)
    return decoupe


def _purger_couche_brute_dossier(connection, chemins: CoucheBruteChemins) -> None:
    connection.execute(
        """
        DELETE FROM public.memory_chunks
        WHERE source_type = 'corpus'
          AND vault_ref = %s
        """,
        (chemins.dossier.vault_ref,),
    )
    if chemins.reprise_log.is_file():
        chemins.reprise_log.unlink()


def build_couche_brute_moreau(
    connection,
    *,
    dossier: Dossier | str = "moreau",
    sorties_dir: Path | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> dict[str, Any]:
    chemins = CoucheBruteChemins.pour_dossier(dossier, sorties_dir=sorties_dir)
    d = chemins.dossier
    chemins.sorties.mkdir(parents=True, exist_ok=True)
    apply_schema(connection)

    gold = load_gold(d)
    decoupe_info = parametres_decoupe_dossier(d, gold["assertions"])
    overlap_chars = int(decoupe_info["recouvrement_chars"])
    max_chars = int(decoupe_info["max_chars_passage"])
    previous = _lire_decoupe_config(chemins)
    if previous is not None and (
        int(previous.get("recouvrement_chars") or 0) != overlap_chars
        or int(previous.get("max_chars_passage") or 0) != max_chars
    ):
        _purger_couche_brute_dossier(connection, chemins)
    _ecrire_decoupe_config(chemins, decoupe_info)

    all_passages = chunk_corpus(
        corpus=d.corpus_dir,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    verifier_couverture_lignes(lignes_non_couvertes(d.corpus_dir, all_passages))
    duplication = stats_duplication(
        all_passages, corpus_dir=d.corpus_dir, max_chars=max_chars
    )
    pieces_un_passage = compter_pieces_un_seul_passage(all_passages, d.corpus_dir)
    couverture_prevue = controle_couverture_citations(
        all_passages,
        gold["assertions"],
        corpus_dir=d.corpus_dir,
    )
    verifier_couverture_citations(couverture_prevue)

    done = _load_reprise_done(chemins) | _load_db_done(connection, d.vault_ref)
    pending = [
        p
        for p in all_passages
        if passage_key(p.piece, p.line_start, p.line_end) not in done
    ]

    appels = 0
    inserted = 0
    skipped = len(all_passages) - len(pending)
    appels_par_piece: dict[str, int] = {}
    caracteres_embedes = 0
    plafond_atteint = False

    for passage in pending:
        appels += 1
        if appels > MAX_EMBED_CALLS:
            plafond_atteint = True
            break
        vector = _embed_text(
            passage.content, call_no=appels, chemins=chemins, embed_fn=embed_fn
        )
        caracteres_embedes += len(passage.content)
        started = time.monotonic()
        connection.execute(
            """
            INSERT INTO public.memory_chunks (
                source_type, source_ref, content, metadata, embedding_mistral, vault_ref
            ) VALUES ('corpus', %s, %s, %s::jsonb, %s::extensions.vector, %s)
            """,
            (
                passage.piece,
                passage.content,
                json.dumps(passage.to_metadata(), ensure_ascii=False),
                to_pgvector(vector),
                d.vault_ref,
            ),
        )
        inserted += 1
        appels_par_piece[passage.piece] = appels_par_piece.get(passage.piece, 0) + 1
        _journaliser_reprise(chemins, passage.piece, passage)
        elapsed = time.monotonic() - started
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

    stored_passages = _fetch_passages_from_db(connection, d.vault_ref)
    couverture = controle_couverture_citations(
        stored_passages,
        gold["assertions"],
        corpus_dir=d.corpus_dir,
    )
    if not plafond_atteint:
        verifier_couverture_citations(couverture)
    lengths = sorted(len(p.content) for p in stored_passages)
    median_len = lengths[len(lengths) // 2] if lengths else 0
    pieces = sorted({p.piece for p in stored_passages})

    payload = {
        "status": "plafond_atteint" if plafond_atteint else "complete",
        "pieces": len(pieces),
        "passages": len(stored_passages),
        "passages_inseres": inserted,
        "passages_reprise": skipped,
        "longueur_mediane": median_len,
        "appels_embedding": appels if not plafond_atteint else MAX_EMBED_CALLS,
        "appels_embedding_session": appels,
        "cout_estime_usd": _estimer_cout(appels, caracteres_embedes),
        "formule_cout": FORMULE_COUT,
        "modele": MODEL_ID,
        "vault_ref": d.vault_ref,
        "par_piece": _stats_par_piece(stored_passages, appels_par_piece),
        "recouvrement": decoupe_info,
        "duplication": duplication,
        "pieces_un_seul_passage": pieces_un_passage,
        **couverture,
    }
    chemins.build_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _fetch_passages_from_db(connection, vault_ref: str) -> list[Passage]:
    rows = connection.execute(
        """
        SELECT source_ref, content, metadata
        FROM public.memory_chunks
        WHERE source_type = 'corpus'
          AND vault_ref = %s
        ORDER BY source_ref, id
        """,
        (vault_ref,),
    ).fetchall()
    out: list[Passage] = []
    for row in rows:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        out.append(
            Passage(
                piece=row.get("source_ref") or meta.get("piece") or "",
                content=row.get("content") or "",
                line_start=int(meta.get("line_start") or 1),
                line_end=int(meta.get("line_end") or 1),
                section_path=str(meta.get("section_path") or ""),
            )
        )
    return out


def _load_question_cache(chemins: CoucheBruteChemins) -> dict[str, list[float]]:
    if chemins.question_embed_cache.is_file():
        return json.loads(chemins.question_embed_cache.read_text(encoding="utf-8"))
    return {}


def _write_question_cache(chemins: CoucheBruteChemins, cache: dict[str, list[float]]) -> None:
    chemins.sorties.mkdir(parents=True, exist_ok=True)
    chemins.question_embed_cache.write_text(json.dumps(cache), encoding="utf-8")


def prechauffer_questions_moreau(
    *,
    dossier: Dossier | str = "moreau",
    sorties_dir: Path | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    appels_deja: int = 0,
) -> dict[str, Any]:
    from .or_dossier import load_questions

    chemins = CoucheBruteChemins.pour_dossier(dossier, sorties_dir=sorties_dir)
    cache = _load_question_cache(chemins)
    questions = load_questions(chemins.dossier.nom)
    to_warm = [q["question"] for q in questions if q["question"] not in cache]
    appels = appels_deja
    warmed = 0
    plafond_atteint = False

    for text in to_warm:
        appels += 1
        if appels > MAX_EMBED_CALLS:
            plafond_atteint = True
            break
        cache[text] = _embed_text(text, call_no=appels, chemins=chemins, embed_fn=embed_fn)
        _write_question_cache(chemins, cache)
        warmed += 1
        time.sleep(max(0.0, 1.0))

    return {
        "questions_total": len(questions),
        "warmed": warmed,
        "skipped": len(to_warm) - warmed,
        "appels_embedding": appels,
        "plafond_atteint": plafond_atteint,
        "cache_path": str(chemins.question_embed_cache),
    }


def _corpus_chars(dossier_nom: str) -> int:
    from .dossiers import dossier_courant

    dossier = dossier_courant(dossier_nom)
    total = 0
    for path in dossier.corpus_dir.iterdir():
        if path.is_file():
            total += len(path.read_text(encoding="utf-8"))
    return total


def _duplication_greenloop() -> dict[str, Any]:
    from .dossiers import dossier_courant

    gl = dossier_courant("greenloop")
    gl_decoupe = parametres_decoupe_greenloop()
    gl_passages = chunk_corpus(
        corpus=gl.corpus_dir,
        max_chars=int(gl_decoupe["max_chars_passage"]),
        overlap_chars=int(gl_decoupe["recouvrement_chars"]),
    )
    return stats_duplication(gl_passages, corpus_dir=gl.corpus_dir)


def write_rapport_couverture_decoupe(
    moreau_build: dict[str, Any] | None,
    greenloop_controle: dict[str, Any],
) -> Path:
    """Rapport comparatif MOREAU / GreenLoop — couverture citations par la découpe."""
    COUVERTURE_RAPPORT.parent.mkdir(parents=True, exist_ok=True)
    budget_chars = 12_000
    gl_corpus = _corpus_chars("greenloop")
    mo_corpus = _corpus_chars("moreau")
    gl_passages_db = int(greenloop_controle.get("passages_en_base") or 0)
    mo_passages = int((moreau_build or {}).get("passages") or 0)
    gl_budget_pct = round(100.0 * budget_chars / gl_corpus, 1) if gl_corpus else 0.0
    mo_budget_pct = round(100.0 * budget_chars / mo_corpus, 1) if mo_corpus else 0.0
    lines = [
        "# Couverture découpe — MOREAU vs GreenLoop",
        "",
        "Contrôle : chaque citation de l'or doit être contenue **entière** "
        "dans au moins un passage de la couche brute.",
        "",
        "## Budget lecture vs corpus",
        "",
        "| dossier | corpus (car.) | passages | budget 12 000 = X % du corpus |",
        "| --- | --- | --- | --- |",
        f"| GreenLoop | {gl_corpus} | {gl_passages_db} | {gl_budget_pct} % |",
        f"| MOREAU | {mo_corpus} | {mo_passages} | {mo_budget_pct} % |",
        "",
        "## MOREAU",
        "",
    ]
    if moreau_build:
        rec = moreau_build.get("recouvrement") or {}
        dup = moreau_build.get("duplication") or {}
        lines.extend(
            [
                f"- Couverture : **{moreau_build.get('citations_retrouvees')}**",
                f"- Règle : {rec.get('regle', '')}",
                f"- Taille de passage : **{rec.get('max_chars_passage')} caractères**",
                f"- Recouvrement retenu : **{rec.get('recouvrement_chars')} caractères**",
                f"- Calcul : `{rec.get('calcul', '')}`",
                f"- Citation la plus longue : {rec.get('citation_plus_longue_id')} "
                f"({rec.get('citation_plus_longue_chars')} car.)",
                f"- Marge : {rec.get('marge_chars')} car.",
                f"- Pièces en un seul passage : {moreau_build.get('pieces_un_seul_passage')}",
                f"- Caractères dupliqués : {dup.get('caracteres_dupliques')} "
                f"({dup.get('duplication_pct')} % des caractères indexés)",
            ]
        )
        if rec.get("passage_trop_petit"):
            lines.append(
                "- **Alerte** : le recouvrement requis dépasse la moitié de la taille "
                f"d'un passage ({rec.get('max_chars_passage')} car.) — les passages sont "
                "trop petits pour ce dossier."
            )
        manquantes = moreau_build.get("citations_manquantes") or []
        if manquantes:
            lines.append("")
            lines.append("### Citations non couvertes")
            for item in manquantes:
                lines.append(f"- `{item['id']}` ({item['piece']}) : {item.get('cause')}")
    else:
        lines.append("- Construction MOREAU non disponible.")

    lines.extend(["", "## GreenLoop (contrôle rétroactif, couche brute inchangée)", ""])
    gl_rec = greenloop_controle.get("recouvrement_requis") or {}
    gl_dup = _duplication_greenloop()
    lines.extend(
        [
            f"- Couverture : **{greenloop_controle.get('citations_retrouvees')}** "
            f"({greenloop_controle.get('passages_en_base')} passages en base)",
            f"- Règle : {gl_rec.get('regle', f'historique fixe ({MAX_CHARS} / {OVERLAP_CHARS})')}",
            f"- Taille de passage : {gl_rec.get('max_chars_passage', MAX_CHARS)} car.",
            f"- Recouvrement : {gl_rec.get('recouvrement_chars', OVERLAP_CHARS)} car.",
            f"- Duplication (découpe actuelle) : {gl_dup.get('duplication_pct')} %",
        ]
    )
    gl_manquantes = greenloop_controle.get("manquantes_par_piece") or {}
    if gl_manquantes:
        lines.append("")
        lines.append("### Répartition des citations non couvertes par pièce")
        lines.append("")
        lines.append("| pièce | manquantes |")
        lines.append("| --- | --- |")
        for piece in sorted(gl_manquantes):
            lines.append(f"| {piece} | {gl_manquantes[piece]} |")
        total_manquantes = len(greenloop_controle.get("citations_manquantes") or [])
        lines.append("")
        lines.append(
            f"**{total_manquantes}** citations de l'or GreenLoop n'apparaissent entières "
            "dans aucun passage de la couche brute actuelle. Les mesures publiées "
            "(dont le 85/141 du RAG naïf) peuvent compter des zéros dus à la découpe, "
            "pas à la mémoire des fournisseurs."
        )
    else:
        lines.append("")
        lines.append("Toutes les citations GreenLoop sont couvertes par la découpe actuelle.")

    COUVERTURE_RAPPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return COUVERTURE_RAPPORT


def write_rapport(
    build: dict[str, Any],
    questions: dict[str, Any],
    *,
    dossier: Dossier | str = "moreau",
    sorties_dir: Path | None = None,
) -> None:
    chemins = CoucheBruteChemins.pour_dossier(dossier, sorties_dir=sorties_dir)
    vault_ref = chemins.dossier.vault_ref
    lines = [
        "# Rapport — couche brute MOREAU",
        "",
        f"- Dossier : `{vault_ref}` (colonne `memory_chunks.vault_ref`)",
        f"- Pièces : {build.get('pieces')}",
        f"- Passages : {build.get('passages')}",
        f"- Appels embedding (session) : {build.get('appels_embedding_session')}",
        f"- Coût estimé : {build.get('cout_estime_usd')} USD ({build.get('formule_cout')})",
        f"- Couverture citations : **{build.get('citations_retrouvees')}**",
    ]
    rec = build.get("recouvrement") or {}
    dup = build.get("duplication") or {}
    if rec:
        lines.append(f"- Recouvrement : {rec.get('recouvrement_chars')} car. ({rec.get('calcul', '')})")
    if dup:
        lines.append(
            f"- Duplication : {dup.get('caracteres_dupliques')} car. "
            f"({dup.get('duplication_pct')} %)"
        )
    lines.extend(
        [
            "",
            "## Par pièce",
            "",
            "| pièce | passages | caractères | appels embedding |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in build.get("par_piece") or []:
        lines.append(
            f"| {row['piece']} | {row['passages']} | {row['caracteres']} | "
            f"{row.get('appels_embedding', 0)} |"
        )

    lines.extend(["", "## Contrôle de couverture", ""])
    coupees = build.get("citations_coupees") or []
    absentes = build.get("citations_absentes") or []
    if not coupees and not absentes:
        lines.append("Toutes les citations gold sont retrouvées mot pour mot dans un passage.")
    else:
        if coupees:
            lines.append("### Citations coupées par la découpe")
            for item in coupees:
                lines.append(
                    f"- `{item['id']}` ({item['piece']}) : {item.get('quote_debut', '')}…"
                )
        if absentes:
            lines.append("### Citations absentes")
            for item in absentes:
                lines.append(
                    f"- `{item['id']}` ({item['piece']}) : {item.get('quote_debut', '')}…"
                )

    lines.extend(
        [
            "",
            "## Questions MOREAU",
            "",
            f"- Cache : `{questions.get('cache_path')}`",
            f"- Questions préchauffées cette session : {questions.get('warmed')}",
            f"- Questions déjà en cache : {questions.get('skipped')}",
        ]
    )
    if build.get("status") == "plafond_atteint" or questions.get("plafond_atteint"):
        lines.append("")
        lines.append(
            f"**Plafond atteint** ({MAX_EMBED_CALLS} appels) — ingestion incomplète, reprise possible."
        )

    chemins.rapport.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cles_passages(passages: list[Passage]) -> set[str]:
    return {passage_key(p.piece, p.line_start, p.line_end) for p in passages}


def _passages_attendus(dossier: Dossier, decoupe: dict[str, Any]) -> list[Passage]:
    return chunk_corpus(
        corpus=dossier.corpus_dir,
        max_chars=int(decoupe["max_chars_passage"]),
        overlap_chars=int(decoupe["recouvrement_chars"]),
    )


def _passages_sans_vecteur(connection, vault_ref: str) -> list[str]:
    rows = connection.execute(
        """
        SELECT source_ref, metadata
        FROM public.memory_chunks
        WHERE source_type = 'corpus'
          AND vault_ref = %s
          AND embedding_mistral IS NULL
        """,
        (vault_ref,),
    ).fetchall()
    keys: list[str] = []
    for row in rows:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        piece = row.get("source_ref") or meta.get("piece") or ""
        line_start = int(meta.get("line_start") or 0)
        line_end = int(meta.get("line_end") or 0)
        if piece and line_start and line_end:
            keys.append(passage_key(piece, line_start, line_end))
    return sorted(keys)


def _decoupe_egale(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if a is None or b is None:
        return False
    for key in ("recouvrement_chars", "max_chars_passage", "citation_plus_longue_id"):
        if str(a.get(key) or "") != str(b.get(key) or ""):
            return False
    return int(a.get("citation_plus_longue_chars") or 0) == int(
        b.get("citation_plus_longue_chars") or 0
    )


def verifier_couche_brute(
    connection,
    dossier: Dossier | str = "moreau",
    *,
    sorties_dir: Path | None = None,
) -> tuple[int, str]:
    """Contrôle d'intégrité couche brute — retourne (code, sortie texte)."""
    chemins = CoucheBruteChemins.pour_dossier(dossier, sorties_dir=sorties_dir)
    d = chemins.dossier
    gold = load_gold(d)
    total_citations = len(gold["assertions"])
    erreurs: list[str] = []

    decoupe_config = _lire_decoupe_config(chemins)
    build: dict[str, Any] = {}
    if chemins.build_json.is_file():
        build = json.loads(chemins.build_json.read_text(encoding="utf-8"))

    passages_db = _fetch_passages_from_db(connection, d.vault_ref)
    nb_db = len(passages_db)
    nb_rapport = int(build.get("passages") or 0)
    if nb_rapport and nb_db != nb_rapport:
        erreurs.append(
            f"passages en base ({nb_db}) ≠ build.json ({nb_rapport})"
        )

    sans_vecteur = _passages_sans_vecteur(connection, d.vault_ref)
    if sans_vecteur:
        preview = ", ".join(sans_vecteur[:5])
        extra = f" … +{len(sans_vecteur) - 5}" if len(sans_vecteur) > 5 else ""
        erreurs.append(f"{len(sans_vecteur)} passage(s) sans vecteur Mistral : {preview}{extra}")

    decoupe_attendue = parametres_decoupe_dossier(d, gold["assertions"])
    if decoupe_config is None:
        erreurs.append("decoupe-config.json absent")
    elif not _decoupe_egale(decoupe_config, decoupe_attendue):
        erreurs.append(
            "decoupe-config.json diverge de l'or "
            f"(fichier {decoupe_config.get('recouvrement_chars')}/"
            f"{decoupe_config.get('max_chars_passage')}, "
            f"attendu {decoupe_attendue['recouvrement_chars']}/"
            f"{decoupe_attendue['max_chars_passage']})"
        )

    recouvrement = decoupe_config or decoupe_attendue
    attendus = _passages_attendus(d, recouvrement)
    cles_db = _cles_passages(passages_db)
    cles_attendues = _cles_passages(attendus)
    if cles_db != cles_attendues:
        manquants = sorted(cles_attendues - cles_db)
        surplus = sorted(cles_db - cles_attendues)
        detail = ""
        if manquants:
            detail += f" manquants en base : {', '.join(manquants[:3])}"
        if surplus:
            detail += f" surplus en base : {', '.join(surplus[:3])}"
        erreurs.append(f"découpe en base diverge de la configuration —{detail}")

    rec_build = build.get("recouvrement") or {}
    if decoupe_config and rec_build and not _decoupe_egale(decoupe_config, rec_build):
        erreurs.append("decoupe-config.json diverge de build.json (recouvrement)")

    couverture = controle_couverture_citations(
        passages_db, gold["assertions"], corpus_dir=d.corpus_dir
    )
    retrouvees = str(couverture.get("citations_retrouvees") or "")
    attendu_couverture = f"{total_citations}/{total_citations}"
    if retrouvees != attendu_couverture:
        erreurs.append(f"couverture citations {retrouvees} (attendu {attendu_couverture})")

    rec = decoupe_attendue
    if erreurs:
        lines = [
            f"Vérification couche brute {d.nom.upper()} : ÉCHEC",
            *[f"- {e}" for e in erreurs],
        ]
        return 1, "\n".join(lines)

    lines = [
        f"Vérification couche brute {d.nom.upper()} : OK",
        f"- Passages en base : {nb_db} (build.json : {nb_rapport or nb_db})",
        f"- Vecteurs Mistral : {nb_db}/{nb_db}",
        f"- Découpe : recouvrement {rec['recouvrement_chars']} car., "
        f"passage {rec['max_chars_passage']} car.",
        f"- Citation la plus longue : {rec['citation_plus_longue_id']} "
        f"({rec['citation_plus_longue_chars']} car.)",
        f"- Couverture citations : {retrouvees}",
    ]
    return 0, "\n".join(lines)


def lister_passages_dossier(connection, vault_ref: str) -> list[dict[str, Any]]:
    if vault_ref == "greenloop":
        clause = " AND (vault_ref = %s OR vault_ref IS NULL)"
        params: tuple[Any, ...] = (vault_ref,)
    else:
        clause = " AND vault_ref = %s"
        params = (vault_ref,)
    rows = connection.execute(
        f"""
        SELECT id, source_ref, content, vault_ref
        FROM public.memory_chunks
        WHERE source_type = 'corpus'
          {clause}
        ORDER BY id
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def cmd_verifier(args: argparse.Namespace) -> int:
    from .infra import connect_existing

    connection = connect_existing()
    code, output = verifier_couche_brute(connection, args.dossier)
    print(output)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Couche brute dossier juridique MOREAU")
    sub = parser.add_subparsers(dest="command", required=True)
    verifier = sub.add_parser("verifier", help="Contrôle d'intégrité couche brute")
    verifier.add_argument("--dossier", default="moreau")
    verifier.set_defaults(func=cmd_verifier)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
