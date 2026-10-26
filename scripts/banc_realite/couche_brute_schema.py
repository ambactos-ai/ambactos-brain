# SPDX-License-Identifier: AGPL-3.0-only
"""Schéma banc pour public.memory_chunks (divergence documentée vs production)."""
from __future__ import annotations

SCHEMA_SQL = """
-- Banc : embedding Vertex (1408) reste en place mais inutilisé ; Mistral en colonne dédiée.
ALTER TABLE public.memory_chunks
    ALTER COLUMN embedding DROP NOT NULL;

ALTER TABLE public.memory_chunks
    ADD COLUMN IF NOT EXISTS embedding_mistral extensions.vector(1024);

ALTER TABLE public.memory_chunks
    ADD COLUMN IF NOT EXISTS lex_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector(
            'french'::regconfig,
            public.brain_unaccent(content)
        )
    ) STORED;

CREATE INDEX IF NOT EXISTS memory_chunks_lex_tsv_idx
    ON public.memory_chunks
    USING gin (lex_tsv);

CREATE INDEX IF NOT EXISTS memory_chunks_embedding_mistral_idx
    ON public.memory_chunks
    USING hnsw (embedding_mistral extensions.vector_cosine_ops)
    WHERE embedding_mistral IS NOT NULL;

ALTER TABLE public.memory_chunks
    ADD COLUMN IF NOT EXISTS vault_ref text;

UPDATE public.memory_chunks
    SET vault_ref = 'greenloop'
    WHERE source_type = 'corpus'
      AND vault_ref IS NULL;

CREATE INDEX IF NOT EXISTS memory_chunks_vault_ref_corpus_idx
    ON public.memory_chunks (vault_ref, source_ref)
    WHERE source_type = 'corpus';
"""

DIVERGENCE_NOTE = (
    "Production : `memory_chunks.embedding` est `vector(1408)` NOT NULL (Vertex), sans "
    "`embedding_mistral` ni `lex_tsv`. Banc : `embedding` nullable et non alimenté ; "
    "recherche sur `embedding_mistral vector(1024)` et `lex_tsv` (french + brain_unaccent, "
    "comme `public.claims`)."
)


def apply_schema(connection) -> None:
    connection.execute("SET search_path TO public, extensions")
    connection.execute(SCHEMA_SQL)
