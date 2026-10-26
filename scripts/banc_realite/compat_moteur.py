# SPDX-License-Identifier: AGPL-3.0-only
"""Helpers du moteur public — extraits des modules non publiés (ancien chemin)."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

MODEL_ID = "mistral-embed"
EMBED_DIM = 1024
_HYBRID_RRF_K = 60
_DEFAULT_RENDER_MAX_CHARS = 12000


class MistralEmbeddingError(RuntimeError):
    pass


class AppelEmbeddingsInterdit(RuntimeError):
    """Garde pytest — aucun appel réel à l'API d'embeddings."""


def _sous_pytest() -> bool:
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
        or os.environ.get("PYTEST_VERSION") is not None
    )


def claim_embed_text(
    entity_id: str | None,
    value: Any,
    qualifiers: dict[str, Any] | None = None,
) -> str:
    v = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    parts = [(entity_id or "").replace("_", " ").strip()]
    attribute = (qualifiers or {}).get("attribute_id")
    if attribute:
        parts.append(str(attribute).replace("_", " ").strip())
    ctx = " ".join(p for p in parts if p)
    return f"{ctx} : {v}" if ctx else v


def to_pgvector(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in values) + "]"


def load_api_key(api_key: str | None = None) -> str:
    if api_key:
        return api_key
    for name in ("MISTRAL_API_KEY", "BRAIN_MISTRAL_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise MistralEmbeddingError("MISTRAL_API_KEY not set")


def embed_many_with_retry(texts: list[str], **_kwargs: Any) -> list[list[float]]:
    if _sous_pytest():
        raise AppelEmbeddingsInterdit("interdit sous pytest")
    raise MistralEmbeddingError("fournir un embedder injecté ; pas d'appel réseau par défaut")


def embed_with_retry(text: str, **kwargs: Any) -> list[float]:
    return embed_many_with_retry([text], **kwargs)[0]


def embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
    return embed_many_with_retry(texts, **kwargs)


def _normalize_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    text = item.get("text")
    if text is None:
        text = item.get("snippet")
    if text is not None and not isinstance(text, str):
        text = str(text)
    if text and len(text) > 2000:
        text = text[:2000]
    out = dict(item)
    if text is not None:
        out["text"] = text
    score = item.get("score")
    if score is not None:
        out["score"] = float(score)
    fusion = item.get("fusion_score")
    if fusion is not None:
        out["fusion_score"] = float(fusion)
    return out


def __getattr__(name: str):
    def _manquant(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{name} n'est pas publié dans le moteur public")

    if name.isidentifier():
        return _manquant
    raise AttributeError(name)


def _rrf(ranked_lists: list[list[Any]], key_fn, k: int = 60) -> list[Any]:
    scores: dict[str, float] = defaultdict(float)
    by_key: dict[str, Any] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            kid = key_fn(item)
            scores[kid] += 1.0 / (k + rank + 1)
            by_key.setdefault(kid, item)
    fused: list[Any] = []
    for kid in sorted(scores, key=lambda i: (-scores[i], i)):
        item = by_key[kid]
        if isinstance(item, dict):
            item = {**item, "score": scores[kid]}
        fused.append(item)
    return fused


def _normalize_lexical_rank(rank: float) -> float:
    r = max(0.0, float(rank))
    return r / (r + 1.0)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _build_structured_query(
    user_message: str,
    responsibility: str,
    conversation_state: dict[str, Any],
) -> str:
    parts: list[str] = []
    if user_message:
        parts.append(user_message)
    for key in ("active_topic", "active_intention"):
        value = conversation_state.get(key)
        if value:
            parts.append(value)
    resp_label = responsibility.replace("_", " ")
    if resp_label:
        parts.append(resp_label)
    entities = _dedupe_preserve_order(conversation_state.get("referenced_entities") or [])
    for entity in entities:
        if entity:
            parts.append(entity)
    return " ".join(parts)[:512]


def _format_evidence_bullet(item: dict[str, Any]) -> str:
    kind = item.get("kind") or "unknown"
    text = item.get("text") or ""
    source = item.get("source")
    if source:
        return f"- [{kind}] {text} ({source})"
    return f"- [{kind}] {text}"


def _render_markdown_with_budget(
    agent: str,
    responsibility: str,
    conversation_id: str,
    run: dict[str, str | None] | None,
    conversation_state: dict[str, Any],
    evidence: list[dict[str, Any]],
    judgment: dict[str, Any] | None = None,
    *,
    max_chars: int = _DEFAULT_RENDER_MAX_CHARS,
) -> tuple[str, int]:
    del judgment
    lignes = [
        f"Agent: {agent}",
        f"Responsabilité: {responsibility.replace('_', ' ')}",
        f"Conversation: {conversation_id}",
    ]
    if run:
        lignes.append("Run: " + " ".join(str(v) for v in run.values() if v))
    for item in evidence:
        lignes.append(_format_evidence_bullet(item))
    texte = "\n".join(lignes)
    if len(texte) > max_chars:
        return texte[:max_chars], max_chars
    return texte, len(texte)


def _render_markdown(
    agent: str,
    responsibility: str,
    conversation_id: str,
    run: dict[str, str | None] | None,
    conversation_state: dict[str, Any],
    evidence: list[dict[str, Any]],
    judgment: dict[str, Any] | None = None,
    *,
    max_chars: int = _DEFAULT_RENDER_MAX_CHARS,
) -> str:
    rendu, _ = _render_markdown_with_budget(
        agent,
        responsibility,
        conversation_id,
        run,
        conversation_state,
        evidence,
        judgment,
        max_chars=max_chars,
    )
    return rendu


def run_search(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("run_search n'est pas publié dans le moteur public")


def compile_call_frame(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("compile_call_frame n'est pas publié dans le moteur public")
