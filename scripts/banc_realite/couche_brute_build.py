# SPDX-License-Identifier: AGPL-3.0-only
"""Construire la couche brute : découpage, schéma, embeddings Mistral."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from banc_realite.compat_moteur import MODEL_ID, MistralEmbeddingError, embed_with_retry, to_pgvector

from .config import BANC_ROOT
from .couche_brute_schema import apply_schema
from .passages import chunk_corpus

COUCHE_BRUTE_SORTIES = BANC_ROOT / "sorties" / "couche-brute"
EMBED_LOG = COUCHE_BRUTE_SORTIES / "embedding-appels.jsonl"
MAX_EMBED_CALLS = 800


def _append_embed_log(entry: dict[str, Any]) -> None:
    COUCHE_BRUTE_SORTIES.mkdir(parents=True, exist_ok=True)
    with EMBED_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _embed_passage(text: str, *, call_no: int) -> list[float]:
    for attempt in range(10):
        try:
            vector = embed_with_retry(text, attempts=1)
            _append_embed_log(
                {
                    "appel": call_no,
                    "chars": len(text),
                    "tentative": attempt + 1,
                    "status": "ok",
                }
            )
            return vector
        except MistralEmbeddingError as exc:
            message = str(exc)
            _append_embed_log(
                {
                    "appel": call_no,
                    "chars": len(text),
                    "tentative": attempt + 1,
                    "status": "erreur",
                    "message": message[:200],
                }
            )
            if "HTTP 429" in message and attempt < 9:
                time.sleep(30)
                continue
            raise
    raise MistralEmbeddingError("échec embedding après 10 tentatives 429")


def build_couche_brute(connection) -> dict[str, Any]:
    COUCHE_BRUTE_SORTIES.mkdir(parents=True, exist_ok=True)
    apply_schema(connection)
    connection.execute("DELETE FROM public.memory_chunks WHERE source_type = 'corpus'")

    passages = chunk_corpus()
    lengths = sorted(len(p.content) for p in passages)
    median_len = lengths[len(lengths) // 2] if lengths else 0

    appels = 0
    inserted = 0
    for passage in passages:
        appels += 1
        if appels > MAX_EMBED_CALLS:
            raise RuntimeError(f"plafond embedding atteint ({MAX_EMBED_CALLS})")
        vector = _embed_passage(passage.content, call_no=appels)
        started = time.monotonic()
        connection.execute(
            """
            INSERT INTO public.memory_chunks (
                source_type, source_ref, content, metadata, embedding_mistral
            ) VALUES ('corpus', %s, %s, %s::jsonb, %s::extensions.vector)
            """,
            (
                passage.piece,
                passage.content,
                json.dumps(passage.to_metadata(), ensure_ascii=False),
                to_pgvector(vector),
            ),
        )
        inserted += 1
        elapsed = time.monotonic() - started
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

    payload = {
        "passages": inserted,
        "longueur_mediane": median_len,
        "appels_embedding": appels,
        "modele": MODEL_ID,
    }
    (COUCHE_BRUTE_SORTIES / "build.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
