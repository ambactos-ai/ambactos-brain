# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptateurs fournisseurs de contexte pour le banc comparatif étage 2."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from banc_realite.compat_moteur import _build_structured_query, _render_markdown_with_budget

from .config import BANC_ROOT, SORTIES
from .dossiers import BANC_VAR, DOSSIERS, Dossier, dossier_courant
from .couche_brute_search import search_passages_hybrid
from .etage3_config_gagnante import CONFIGURATIONS as ETAGE3_CONFIGS
from .etage3 import compile_cadre_prod
from .measure_etage2 import _load_embed_cache, cached_embed_factory
from .mesure_porte_phrase import build_evidence_for_porte_phrase_config
from .normalize import piece_basename
from .porte_voisinage import enrichir_porte_voisinage

COMPARATIF_DIR = BANC_ROOT / "sorties" / "comparatif"


@runtime_checkable
class FournisseurContexte(Protocol):
    etiquette: str

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        ...


def _conversation_state(dossier: Dossier | None = None) -> dict[str, Any]:
    current = dossier if dossier is not None else dossier_courant()
    return {"responsibility_context": current.responsibility_context}


def _require_question_vector(question: dict[str, Any], embed_cache: dict[str, list[float]]) -> list[float]:
    text = question["question"]
    if text not in embed_cache:
        raise RuntimeError(f"cache froid : embedding absent ({text[:60]}…)")
    return embed_cache[text]


def _elements_to_evidence(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in elements:
        rank = int(item.get("rank") or len(out) + 1)
        source = item.get("source") or ""
        out.append(
            {
                "kind": "file_hit",
                "memory_type": "semantic",
                "id": f"{source}:{rank}",
                "score": float(item.get("score") or 0.0),
                "rank": rank,
                "text": item.get("text") or "",
                "source": source,
                "snippet": (item.get("text") or "")[:280],
                "payload": {"metadata": item.get("provenance") or {}},
                "provenance": item.get("provenance") or {"source_file_path": source},
            }
        )
    return out


def _source_etiquette(item: dict[str, Any]) -> str:
    source = str(item.get("source") or "").strip()
    if not source:
        provenance = item.get("provenance") or {}
        source = str(provenance.get("source_file_path") or "").strip()
    return source or "source inconnue"


def _render_texte_simple(elements: list[dict[str, Any]], *, budget_chars: int) -> tuple[str, list[dict[str, Any]]]:
    kept = list(elements)
    while kept:
        rendered = "\n\n".join(
            f"- [{_source_etiquette(item)}] {item.get('text') or ''}" for item in kept
        ).strip()
        if len(rendered) <= budget_chars:
            return rendered, kept
        kept.pop()
    return "", []


def rendre_contexte(
    elements: list[dict[str, Any]],
    *,
    budget_chars: int,
    question_id: str = "banc-comparatif",
    style: str = "brain",
    dossier: Dossier | None = None,
    corpus_chars: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Rendu texte + éléments gardés sous le budget caractères."""
    current = dossier if dossier is not None else dossier_courant()
    if style == "brain":
        rendered, dropped = _render_markdown_with_budget(
            "banc-realite",
            current.responsibility_context,
            f"banc-comparatif-{question_id}",
            None,
            _conversation_state(current),
            elements,
            None,
            max_chars=budget_chars,
        )
        kept = elements[: max(len(elements) - dropped, 0)]
        if corpus_chars and corpus_chars > 0:
            pct = round(100.0 * len(rendered) / corpus_chars, 1)
            rendered = f"{rendered}\n\n_Part du corpus MRFS versée : {pct} %_"
        return rendered, kept
    rendered, kept = _render_texte_simple(elements, budget_chars=budget_chars)
    if corpus_chars and corpus_chars > 0 and rendered:
        pct = round(100.0 * len(rendered) / corpus_chars, 1)
        rendered = f"{rendered}\n\nPart du corpus MRFS versée : {pct} %"
    return rendered, kept


class RagNaif:
    etiquette = "rag_naif"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        qtext = question["question"]
        vector = _require_question_vector(question, self._embed_cache)
        structured = _build_structured_query(
            qtext,
            self._dossier.responsibility_context,
            {"responsibility_context": self._dossier.responsibility_context},
        )
        return search_passages_hybrid(
            connection,
            structured,
            query_vector=vector,
            top_k=k,
            vault_ref=self._dossier.vault_ref,
        )


class BrainProd:
    etiquette = "brain_prod"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del connection
        _require_question_vector(question, self._embed_cache)
        cadre = compile_cadre_prod(
            question,
            top_k=k,
            embed_cache=self._embed_cache,
            dossier=self._dossier,
        )
        return list(cadre.get("evidence") or [])


class BrainBanc:
    etiquette = "brain_banc"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )
        self._phrases_cache = phrases_cache
        self._config = ETAGE3_CONFIGS["hybride_fenetres_g12"]["porte_phrase_config"]

    def _phrases(self) -> dict[str, list[float]]:
        if self._phrases_cache is not None:
            return self._phrases_cache
        cache_path = self._dossier.cache_phrases_embeddings
        if not cache_path.is_file():
            raise RuntimeError(f"cache unités absent : {cache_path}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not payload:
            raise RuntimeError(f"cache unités vide : {cache_path}")
        self._phrases_cache = payload
        return payload

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        _require_question_vector(question, self._embed_cache)
        evidence, _brute, _stats = build_evidence_for_porte_phrase_config(
            connection,
            question,
            self._config,
            embed_cache=self._embed_cache,
            phrases_cache=self._phrases(),
            dossier=self._dossier,
        )
        return evidence


class BrainBancG24:
    etiquette = "brain_banc_g24"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )
        self._phrases_cache = phrases_cache
        self._config = ETAGE3_CONFIGS["hybride_fenetres_g24"]["porte_phrase_config"]

    def _phrases(self) -> dict[str, list[float]]:
        if self._phrases_cache is not None:
            return self._phrases_cache
        cache_path = self._dossier.cache_phrases_embeddings
        if not cache_path.is_file():
            raise RuntimeError(f"cache unités absent : {cache_path}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not payload:
            raise RuntimeError(f"cache unités vide : {cache_path}")
        self._phrases_cache = payload
        return payload

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        _require_question_vector(question, self._embed_cache)
        evidence, _brute, _stats = build_evidence_for_porte_phrase_config(
            connection,
            question,
            self._config,
            embed_cache=self._embed_cache,
            phrases_cache=self._phrases(),
            dossier=self._dossier,
        )
        return evidence


class BrainBancG48:
    etiquette = "brain_banc_g48"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )
        self._phrases_cache = phrases_cache
        self._config = ETAGE3_CONFIGS["hybride_fenetres_g48"]["porte_phrase_config"]

    def _phrases(self) -> dict[str, list[float]]:
        if self._phrases_cache is not None:
            return self._phrases_cache
        cache_path = self._dossier.cache_phrases_embeddings
        if not cache_path.is_file():
            raise RuntimeError(f"cache unités absent : {cache_path}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not payload:
            raise RuntimeError(f"cache unités vide : {cache_path}")
        self._phrases_cache = payload
        return payload

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        _require_question_vector(question, self._embed_cache)
        evidence, _brute, _stats = build_evidence_for_porte_phrase_config(
            connection,
            question,
            self._config,
            embed_cache=self._embed_cache,
            phrases_cache=self._phrases(),
            dossier=self._dossier,
        )
        return evidence


class BrainBancG36:
    etiquette = "brain_banc_g36"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )
        self._phrases_cache = phrases_cache
        self._config = ETAGE3_CONFIGS["hybride_fenetres_g36"]["porte_phrase_config"]

    def _phrases(self) -> dict[str, list[float]]:
        if self._phrases_cache is not None:
            return self._phrases_cache
        cache_path = self._dossier.cache_phrases_embeddings
        if not cache_path.is_file():
            raise RuntimeError(f"cache unités absent : {cache_path}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not payload:
            raise RuntimeError(f"cache unités vide : {cache_path}")
        self._phrases_cache = payload
        return payload

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        _require_question_vector(question, self._embed_cache)
        evidence, _brute, _stats = build_evidence_for_porte_phrase_config(
            connection,
            question,
            self._config,
            embed_cache=self._embed_cache,
            phrases_cache=self._phrases(),
            dossier=self._dossier,
        )
        return evidence


class BrainBancV1:
    etiquette = "brain_banc_v1"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )
        self._phrases_cache = phrases_cache
        self._config = ETAGE3_CONFIGS["hybride_fenetres_g12"]["porte_phrase_config"]

    def _phrases(self) -> dict[str, list[float]]:
        if self._phrases_cache is not None:
            return self._phrases_cache
        cache_path = self._dossier.cache_phrases_embeddings
        if not cache_path.is_file():
            raise RuntimeError(f"cache unités absent : {cache_path}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not payload:
            raise RuntimeError(f"cache unités vide : {cache_path}")
        self._phrases_cache = payload
        return payload

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        _require_question_vector(question, self._embed_cache)
        evidence, _brute, _stats = build_evidence_for_porte_phrase_config(
            connection,
            question,
            self._config,
            embed_cache=self._embed_cache,
            phrases_cache=self._phrases(),
            dossier=self._dossier,
        )
        return enrichir_porte_voisinage(
            evidence,
            connection=connection,
            vault_ref=self._dossier.vault_ref,
            arêtes=1,
        )


class BrainBancV3:
    etiquette = "brain_banc_v3"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )
        self._phrases_cache = phrases_cache
        self._config = ETAGE3_CONFIGS["hybride_fenetres_g12"]["porte_phrase_config"]

    def _phrases(self) -> dict[str, list[float]]:
        if self._phrases_cache is not None:
            return self._phrases_cache
        cache_path = self._dossier.cache_phrases_embeddings
        if not cache_path.is_file():
            raise RuntimeError(f"cache unités absent : {cache_path}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not payload:
            raise RuntimeError(f"cache unités vide : {cache_path}")
        self._phrases_cache = payload
        return payload

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        _require_question_vector(question, self._embed_cache)
        evidence, _brute, _stats = build_evidence_for_porte_phrase_config(
            connection,
            question,
            self._config,
            embed_cache=self._embed_cache,
            phrases_cache=self._phrases(),
            dossier=self._dossier,
        )
        return enrichir_porte_voisinage(
            evidence,
            connection=connection,
            vault_ref=self._dossier.vault_ref,
            arêtes=1,
            rendu="phrase",
            question=question,
        )


class BrainBancV4:
    etiquette = "brain_banc_v4"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )
        self._phrases_cache = phrases_cache
        self._config = ETAGE3_CONFIGS["hybride_fenetres_g12"]["porte_phrase_config"]

    def _phrases(self) -> dict[str, list[float]]:
        if self._phrases_cache is not None:
            return self._phrases_cache
        cache_path = self._dossier.cache_phrases_embeddings
        if not cache_path.is_file():
            raise RuntimeError(f"cache unités absent : {cache_path}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not payload:
            raise RuntimeError(f"cache unités vide : {cache_path}")
        self._phrases_cache = payload
        return payload

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        _require_question_vector(question, self._embed_cache)
        evidence, _brute, _stats = build_evidence_for_porte_phrase_config(
            connection,
            question,
            self._config,
            embed_cache=self._embed_cache,
            phrases_cache=self._phrases(),
            dossier=self._dossier,
        )
        return enrichir_porte_voisinage(
            evidence,
            connection=connection,
            vault_ref=self._dossier.vault_ref,
            arêtes=2,
            rendu="phrase",
            question=question,
        )


class BrainBancV2:
    etiquette = "brain_banc_v2"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )
        self._phrases_cache = phrases_cache
        self._config = ETAGE3_CONFIGS["hybride_fenetres_g12"]["porte_phrase_config"]

    def _phrases(self) -> dict[str, list[float]]:
        if self._phrases_cache is not None:
            return self._phrases_cache
        cache_path = self._dossier.cache_phrases_embeddings
        if not cache_path.is_file():
            raise RuntimeError(f"cache unités absent : {cache_path}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not payload:
            raise RuntimeError(f"cache unités vide : {cache_path}")
        self._phrases_cache = payload
        return payload

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        _require_question_vector(question, self._embed_cache)
        evidence, _brute, _stats = build_evidence_for_porte_phrase_config(
            connection,
            question,
            self._config,
            embed_cache=self._embed_cache,
            phrases_cache=self._phrases(),
            dossier=self._dossier,
        )
        return enrichir_porte_voisinage(
            evidence,
            connection=connection,
            vault_ref=self._dossier.vault_ref,
            arêtes=2,
        )


class BrainBancCalibre:
    etiquette = "brain_banc_calibre"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._dossier = dossier if dossier is not None else dossier_courant()
        self._embed_cache = (
            embed_cache if embed_cache is not None else _load_embed_cache(self._dossier)
        )
        self._phrases_cache = phrases_cache
        from .embarquer_phrases import load_fenetre_calibree

        meta = load_fenetre_calibree(self._dossier)
        self._config = json.loads(
            json.dumps(ETAGE3_CONFIGS["hybride_fenetres_calibrees"]["porte_phrase_config"])
        )
        self._config["phrases"]["fenetre"] = int(meta["fenetre_phrases"])

    def _phrases(self) -> dict[str, list[float]]:
        if self._phrases_cache is not None:
            return self._phrases_cache
        cache_path = self._dossier.cache_phrases_embeddings_calibrees
        if not cache_path.is_file():
            raise RuntimeError(f"cache unités calibrées absent : {cache_path}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not payload:
            raise RuntimeError(f"cache unités calibrées vide : {cache_path}")
        self._phrases_cache = payload
        return payload

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        _require_question_vector(question, self._embed_cache)
        evidence, _brute, _stats = build_evidence_for_porte_phrase_config(
            connection,
            question,
            self._config,
            embed_cache=self._embed_cache,
            phrases_cache=self._phrases(),
            dossier=self._dossier,
        )
        return evidence


class FournisseurFichier:
    def __init__(
        self,
        etiquette: str,
        chemin: Path | str | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self.etiquette = etiquette
        if etiquette == "mem0":
            if chemin is not None:
                raise ValueError("mem0 résout son fichier par k — ne pas passer chemin")
            self.chemin = None
        else:
            if chemin is None:
                raise ValueError(f"chemin requis pour {etiquette}")
            self.chemin = Path(chemin)
        self._dossier = dossier
        self._index_by_path: dict[Path, dict[str, list[dict[str, Any]]]] = {}
        self._k_max_by_path: dict[Path, int] = {}
        self._meta_by_path: dict[Path, dict[str, Any]] = {}

    def _exiger_prefixe_dossier(self, question_ids: list[str]) -> None:
        if self._dossier is None or self._dossier.nom == "greenloop":
            return
        prefix = self._dossier.prefixe_question
        invalid = sorted(qid for qid in question_ids if not str(qid).startswith(prefix))
        if invalid:
            raise ValueError(
                f"contextes {self.etiquette} incohérents pour dossier {self._dossier.nom!r} : "
                f"préfixe attendu {prefix!r}, trouvé sur {invalid[:3]}"
                f"{'…' if len(invalid) > 3 else ''} dans {self.chemin}"
            )

    def _chemin_pour_k(self, k: int) -> Path:
        if self.etiquette == "mem0":
            from .tiers_mem0 import comparatif_context_path as mem0_context_path

            return mem0_context_path("mem0", dossier=self._dossier, k=k)
        assert self.chemin is not None
        return self.chemin

    @property
    def meta(self) -> dict[str, Any]:
        chemin = self._chemin_pour_k(20)
        self._load(chemin)
        return dict(self._meta_by_path.get(chemin) or {})

    def ensure_fichiers(self, ks: list[int]) -> None:
        fichiers = {self._chemin_pour_k(k) for k in ks}
        for chemin in fichiers:
            self._load(chemin)

    def _load(self, chemin: Path | None = None) -> dict[str, list[dict[str, Any]]]:
        if chemin is None:
            chemin = self._chemin_pour_k(20) if self.etiquette == "mem0" else self.chemin
        assert chemin is not None
        if chemin in self._index_by_path:
            return self._index_by_path[chemin]
        # Un fichier absent ou une question manquante est une mesure fausse (contexte vide
        # compté comme un vrai zéro) : on refuse, on ne devine pas (revue Codex #123).
        if not chemin.is_file():
            raise FileNotFoundError(
                f"contextes {self.etiquette} absents pour k demandé : {chemin} — "
                f"lancer `tiers_mem0.py contextes --k …` d'abord"
            )
        index: dict[str, list[dict[str, Any]]] = {}
        k_max = 20
        meta: dict[str, Any] = {}
        lines = [line for line in chemin.read_text(encoding="utf-8").splitlines() if line.strip()]
        start = 0
        if lines:
            first = json.loads(lines[0])
            if "meta" in first:
                meta = dict(first["meta"])
                k_max = int(meta.get("k") or 20)
                start = 1
        for line in lines[start:]:
            row = json.loads(line)
            index[str(row["question_id"])] = list(row.get("elements") or [])
        self._exiger_prefixe_dossier(list(index))
        self._k_max_by_path[chemin] = k_max
        self._meta_by_path[chemin] = meta
        self._index_by_path[chemin] = index
        return index

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del connection
        chemin = self._chemin_pour_k(k)
        index = self._load(chemin)
        self._exiger_prefixe_dossier([str(question["id"])])
        k_max = self._k_max_by_path[chemin]
        if k > k_max:
            raise ValueError(
                f"contextes {self.etiquette} : k={k} > k_max={k_max} dans {chemin}"
            )
        if question["id"] not in index:
            raise KeyError(
                f"contextes {self.etiquette} incomplets : {question['id']} absent de {chemin}"
            )
        return _elements_to_evidence(index[question["id"]][:k])


def missing_embed_texts(questions: list[dict[str, Any]], embed_cache: dict[str, list[float]]) -> list[str]:
    return sorted(q["question"] for q in questions if q["question"] not in embed_cache)


def embed_factory_or_raise(
    embed_cache: dict[str, list[float]] | None = None,
    *,
    dossier: Dossier | None = None,
):
    cache = embed_cache if embed_cache is not None else _load_embed_cache(dossier)
    return cached_embed_factory(cache), cache


def comparatif_context_path(
    etiquette: str,
    dossier: Dossier | None = None,
    *,
    k: int = 20,
) -> Path:
    current = dossier if dossier is not None else dossier_courant()
    if etiquette == "mem0":
        from .tiers_mem0 import comparatif_context_path as mem0_context_path

        return mem0_context_path(etiquette, dossier=current, k=k)
    return current.comparatif_dir / f"contextes-{etiquette}.jsonl"


def default_sorties_path(dossier: Dossier | None = None) -> Path:
    current = dossier if dossier is not None else dossier_courant()
    return current.cache_embeddings_questions


# Ancien plafond de pagination du domaine (k×4 ≤ 100). Le lecteur classé
# note tous les faits ; k n'est plus une limite de recherche. Conservé pour
# les registres d'étage 2. `rendre_contexte` coupe au budget.
K_LECTEUR_MRFS = 25
INTITULE_RAPPROCHEMENT = "Rapprochement proposé"

# Filtre puis RAG — constantes fixées avant toute mesure (#154).
K_RAG_LARGE = 60
K_VIVIER_FILTRE = 200
N_PIECES_FILTRE = 3
N_MEILLEURS_PASSAGES = 3
N_PIECES_UNION = N_PIECES_FILTRE * 2


def _texte_question(question: dict[str, Any] | str) -> str:
    if isinstance(question, str):
        return question
    return str(question.get("question") or "")


def _nom_dossier(dossier: Dossier | str | None) -> str:
    if isinstance(dossier, str):
        return dossier
    if dossier is None:
        return dossier_courant().nom
    return dossier.nom


def _piece_element(item: dict[str, Any]) -> str:
    source = str(item.get("source") or "").strip()
    if not source:
        provenance = item.get("provenance") or {}
        source = str(
            provenance.get("source_file_path") or provenance.get("piece") or ""
        ).strip()
    return piece_basename(source)


def _lignes_lisibles(start: Any, end: Any) -> str:
    if isinstance(start, int) and isinstance(end, int):
        if start == end:
            return f"ligne {start}"
        return f"lignes {start}–{end}"
    if isinstance(start, int):
        return f"ligne {start}"
    return ""


def _date_lisible(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, Mapping):
        return str(value.get("texte") or value.get("iso") or "").strip()
    text = str(value).strip()
    if text in {"None", "none"}:
        return ""
    return text


def _ligne_provenance(
    *,
    source: str,
    line_start: Any,
    line_end: Any,
    enonciateur: Any,
    date: Any,
) -> str:
    parts = [f"Pièce {source}"]
    lignes = _lignes_lisibles(line_start, line_end)
    if lignes:
        parts.append(lignes)
    if enonciateur:
        parts.append(f"énoncé par {enonciateur}")
    date_txt = _date_lisible(date)
    if date_txt:
        parts.append(date_txt)
    return ", ".join(parts)


def _ligne_alternative(alt: Mapping[str, Any]) -> str:
    citation = str(alt.get("citation") or alt.get("text") or "").strip()
    source = str(alt.get("source_file") or alt.get("source") or "").strip() or "source inconnue"
    ligne = _ligne_provenance(
        source=source,
        line_start=alt.get("line_start"),
        line_end=alt.get("line_end"),
        enonciateur=alt.get("enonciateur"),
        date=alt.get("date"),
    )
    if citation:
        return f"- {citation} — {ligne}"
    return f"- {ligne}"


def formater_element_mrfs(fait: dict[str, Any], *, rank: int | None = None) -> dict[str, Any]:
    """Lettre + provenance lisible ; alternatives dans le même élément."""
    lettre = str(fait.get("text") or "").strip()
    source = str(fait.get("source") or "").strip()
    provenance = dict(fait.get("provenance") or {})
    if source:
        provenance.setdefault("source_file_path", source)
    provenance["citation"] = lettre
    effective_rank = int(rank if rank is not None else fait.get("rank") or 1)
    blocs = [
        lettre,
        "",
        _ligne_provenance(
            source=source,
            line_start=provenance.get("line_start"),
            line_end=provenance.get("line_end"),
            enonciateur=fait.get("enonciateur"),
            date=fait.get("date"),
        ),
    ]
    alternatives = [
        dict(alt)
        for alt in (fait.get("alternatives") or [])
        if isinstance(alt, Mapping)
    ]
    if alternatives:
        blocs.append(f"{INTITULE_RAPPROCHEMENT} :")
        blocs.extend(_ligne_alternative(alt) for alt in alternatives)
    return {
        "kind": "mrfs_situation",
        "id": f"{source}:{effective_rank}",
        "rank": effective_rank,
        "score": float(fait.get("score") or 0.0),
        "text": "\n".join(blocs).strip(),
        "source": source,
        "snippet": lettre[:280],
        "enonciateur": fait.get("enonciateur"),
        "date": fait.get("date"),
        "alternatives": alternatives,
        "provenance": provenance,
        "payload": {"metadata": provenance},
    }


def fusionner_g12_puis_mrfs(
    g12: list[dict[str, Any]], mrfs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """g12 d'abord, inchangée ; MRFS ensuite. Seule la répétition exacte même pièce est retirée."""
    deja: dict[str, set[str]] = {}
    for item in g12:
        deja.setdefault(_piece_element(item), set()).add(str(item.get("text") or ""))
    retenus: list[dict[str, Any]] = []
    for item in mrfs:
        piece = _piece_element(item)
        provenance = item.get("provenance") or {}
        lettre = str(provenance.get("citation") or "").strip()
        if not lettre:
            lettre = str(item.get("text") or "").split("\n", 1)[0].strip()
        if lettre and lettre in deja.get(piece, set()):
            continue
        retenus.append(item)
    return list(g12) + retenus


class BrainMrfs:
    etiquette = "brain_mrfs"
    _mode = "hybride"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        faits: list[Any] | None = None,
        embeddings_faits: dict[str, list[float]] | None = None,
        embeddings_faits_path: Path | None = None,
        constructeur_root: Path | None = None,
        scorer_lexical: Any = None,
    ) -> None:
        self._nom = _nom_dossier(dossier)
        if isinstance(dossier, Dossier):
            self._dossier = dossier
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            self._dossier = dossier_courant(dossier)
        elif dossier is None:
            self._dossier = dossier_courant()
        else:
            self._dossier = None
        self._embed_cache = embed_cache
        self._faits = list(faits) if faits is not None else None
        self._embeddings_faits = embeddings_faits
        self._embeddings_faits_path = embeddings_faits_path
        self._constructeur_root = constructeur_root
        self._scorer_lexical = scorer_lexical
        self._stats: dict[str, object] = {}

    @property
    def stats(self) -> dict[str, object]:
        return dict(self._stats)

    def _question_vector(self, question: dict[str, Any]) -> list[float]:
        cache = self._embed_cache
        if cache is None:
            if self._dossier is None:
                raise RuntimeError("cache froid : embedding de question absent")
            cache = _load_embed_cache(self._dossier)
            self._embed_cache = cache
        return _require_question_vector(question, cache)

    def _charger_embeddings_faits(self) -> dict[str, list[float]]:
        if self._embeddings_faits is not None:
            return self._embeddings_faits
        from .embeddings_faits import cache_embeddings_faits_path, charger_cache

        path = self._embeddings_faits_path or cache_embeddings_faits_path(
            self._nom, constructeur_root=self._constructeur_root
        )
        cache = charger_cache(path)
        if not cache:
            raise RuntimeError(f"embeddings de faits absents : {path}")
        self._embeddings_faits = cache
        return cache

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        from .lecteur_faits_classes import (
            _connexion,
            charger_faits,
            contexte_faits_classes,
        )
        from .mrfs_support import BANC_PG_DSN, MAX_BUDGET_CHARS

        # Le classement lexical se fait toujours en SQL, sur le cluster du banc : la
        # connexion de la mesure peut viser une autre base, et un repli sur un autre
        # calcul selon l'état du cache rendrait deux contextes pour une même question.
        del connection
        own = False
        conn = None
        if self._scorer_lexical is None or self._faits is None:
            conn = _connexion(BANC_PG_DSN)
            own = True
        try:
            faits = (
                self._faits
                if self._faits is not None
                else charger_faits(self._nom, connection=conn)
            )
            if self._faits is None:
                self._faits = list(faits)
            kwargs: dict[str, Any] = {
                "dossier": self._nom,
                "mode": self._mode,
                "connection": conn,
                "faits": faits,
                "scorer_lexical": self._scorer_lexical,
            }
            if self._mode == "hybride":
                kwargs["question_vector"] = self._question_vector(question)
                kwargs["embeddings"] = self._charger_embeddings_faits()
            elements = contexte_faits_classes(_texte_question(question), **kwargs)
        finally:
            if own and conn is not None:
                conn.close()
        self._stats = {
            "hits": len(elements),
            "n_faits": len(faits),
            "budget_chars": MAX_BUDGET_CHARS,
            "dossier": self._nom,
            "mode": self._mode,
        }
        return elements


class BrainMrfsLexical(BrainMrfs):
    etiquette = "brain_mrfs_lexical"
    _mode = "lexical"


class BrainMrfsG12:
    etiquette = "brain_mrfs_g12"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
        faits: list[Any] | None = None,
        embeddings_faits: dict[str, list[float]] | None = None,
    ) -> None:
        current = dossier if dossier is not None else dossier_courant()
        self._g12 = BrainBanc(embed_cache, phrases_cache, dossier=current)
        self._mrfs = BrainMrfs(
            embed_cache,
            dossier=current,
            faits=faits,
            embeddings_faits=embeddings_faits,
        )

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        g12 = self._g12.contexte(question, k=k, connection=connection)
        mrfs = self._mrfs.contexte(question, k=k, connection=connection)
        return fusionner_g12_puis_mrfs(g12, mrfs)


class BrainSituations:
    etiquette = "brain_situations"
    _lettres_chronologie = False

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        situations: list[Any] | None = None,
        embeddings_situations: dict[str, list[float]] | None = None,
        embeddings_situations_path: Path | None = None,
        situations_root: Path | None = None,
        chemin: Path | None = None,
        scorer_lexical: Any = None,
    ) -> None:
        self._nom = _nom_dossier(dossier)
        if isinstance(dossier, Dossier):
            self._dossier = dossier
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            self._dossier = dossier_courant(dossier)
        elif dossier is None:
            self._dossier = dossier_courant()
        else:
            self._dossier = None
        self._embed_cache = embed_cache
        self._situations = list(situations) if situations is not None else None
        self._embeddings_situations = embeddings_situations
        self._embeddings_situations_path = embeddings_situations_path
        self._situations_root = situations_root
        self._chemin = chemin
        self._scorer_lexical = scorer_lexical
        self._stats: dict[str, object] = {}

    @property
    def stats(self) -> dict[str, object]:
        return dict(self._stats)

    def _question_vector(self, question: dict[str, Any]) -> list[float]:
        cache = self._embed_cache
        if cache is None:
            if self._dossier is None:
                raise RuntimeError("cache froid : embedding de question absent")
            cache = _load_embed_cache(self._dossier)
            self._embed_cache = cache
        return _require_question_vector(question, cache)

    def _cache_embeddings_path(self) -> Path:
        from .embeddings_faits import (
            cache_embeddings_situations_lettre_path,
            cache_embeddings_situations_path,
        )

        if self._lettres_chronologie:
            return cache_embeddings_situations_lettre_path(
                self._nom, situations_root=self._situations_root
            )
        return cache_embeddings_situations_path(
            self._nom, situations_root=self._situations_root
        )

    def _charger_embeddings_situations(self) -> dict[str, list[float]]:
        if self._embeddings_situations is not None:
            return self._embeddings_situations
        from .embeddings_faits import charger_cache

        path = self._embeddings_situations_path or self._cache_embeddings_path()
        cache = charger_cache(path)
        if not cache:
            raise RuntimeError(f"embeddings de situations absents : {path}")
        self._embeddings_situations = cache
        return cache

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        from .lecteur_faits_classes import _connexion
        from .lecteur_situations_composites import (
            charger_situations,
            contexte_situations_composites,
        )
        from .mrfs_support import BANC_PG_DSN, MAX_BUDGET_CHARS

        del connection
        own = False
        conn = None
        if self._scorer_lexical is None:
            conn = _connexion(BANC_PG_DSN)
            own = True
        try:
            situations = (
                self._situations
                if self._situations is not None
                else charger_situations(
                    self._nom,
                    chemin=self._chemin,
                    situations_root=self._situations_root,
                )
            )
            if self._situations is None:
                self._situations = list(situations)
            elements = contexte_situations_composites(
                _texte_question(question),
                dossier=self._nom,
                chemin=self._chemin,
                situations_root=self._situations_root,
                situations=situations,
                connection=conn,
                scorer_lexical=self._scorer_lexical,
                question_vector=self._question_vector(question),
                embeddings=self._charger_embeddings_situations(),
                lettres_chronologie=self._lettres_chronologie,
            )
        finally:
            if own and conn is not None:
                conn.close()
        self._stats = {
            "hits": len(elements),
            "n_situations": len(situations),
            "budget_chars": MAX_BUDGET_CHARS,
            "dossier": self._nom,
        }
        return elements


class BrainSituationsLettre(BrainSituations):
    etiquette = "brain_situations_lettre"
    _lettres_chronologie = True


class BrainSituationsG12:
    etiquette = "brain_situations_g12"
    _situations_cls = BrainSituations

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
        situations: list[Any] | None = None,
        embeddings_situations: dict[str, list[float]] | None = None,
        scorer_lexical: Any = None,
        situations_root: Path | None = None,
        chemin: Path | None = None,
    ) -> None:
        current = dossier if dossier is not None else dossier_courant()
        self._g12 = BrainBanc(embed_cache, phrases_cache, dossier=current)
        self._situations = self._situations_cls(
            embed_cache,
            dossier=current,
            situations=situations,
            embeddings_situations=embeddings_situations,
            situations_root=situations_root,
            chemin=chemin,
            scorer_lexical=scorer_lexical,
        )

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        g12 = self._g12.contexte(question, k=k, connection=connection)
        situations = self._situations.contexte(question, k=k, connection=connection)
        return fusionner_g12_puis_mrfs(g12, situations)


class BrainSituationsLettreG12(BrainSituationsG12):
    etiquette = "brain_situations_lettre_g12"
    _situations_cls = BrainSituationsLettre


def piece_depuis_provenance(item: Mapping[str, Any]) -> str | None:
    """Nom de pièce lu dans la provenance du passage ; None si absente."""
    provenance = item.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    raw = provenance.get("source_file_path") or provenance.get("piece") or ""
    text = str(raw).strip()
    if not text:
        return None
    return piece_basename(text)


def noter_pieces_par_passages(
    passages: list[dict[str, Any]],
    *,
    n_meilleurs: int = N_MEILLEURS_PASSAGES,
) -> tuple[list[str], int]:
    """Classe les pièces par la somme des n meilleurs scores. Second : passages sans pièce."""
    scores: dict[str, list[float]] = {}
    premier_rang: dict[str, int] = {}
    sans_piece = 0
    for item in passages:
        piece = piece_depuis_provenance(item)
        if piece is None:
            sans_piece += 1
            continue
        scores.setdefault(piece, []).append(float(item.get("score") or 0.0))
        premier_rang.setdefault(piece, int(item.get("rank") or 10**9))
    notes = {
        piece: sum(sorted(vals, reverse=True)[:n_meilleurs])
        for piece, vals in scores.items()
    }
    ordered = sorted(notes, key=lambda piece: (-notes[piece], premier_rang[piece], piece))
    return ordered, sans_piece


def pieces_depuis_classement_situations(
    notes: list[Any],
    situations: list[Mapping[str, Any]],
    *,
    n_pieces: int = N_PIECES_FILTRE,
) -> list[str]:
    """Parcourt le classement et retient les pièces dans l'ordre d'apparition."""
    from .lecteur_situations_composites import pieces_de

    by_id = {str(situation.get("id") or ""): situation for situation in situations}
    retenues: list[str] = []
    seen: set[str] = set()
    for note in notes:
        fait = getattr(note, "fait", None)
        sid = str(getattr(fait, "id", "") or "")
        situation = by_id.get(sid)
        if situation is None:
            continue
        for raw in pieces_de(situation):
            piece = piece_basename(str(raw)) if raw else ""
            if not piece or piece in seen:
                continue
            seen.add(piece)
            retenues.append(piece)
            if len(retenues) >= n_pieces:
                return retenues
    return retenues


def reunir_pieces(*listes: list[str], plafond: int = N_PIECES_UNION) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for liste in listes:
        for piece in liste:
            if not piece or piece in seen:
                continue
            seen.add(piece)
            out.append(piece)
            if len(out) >= plafond:
                return out
    return out


def filtrer_passages_par_pieces(
    passages: list[dict[str, Any]],
    pieces: list[str] | set[str],
) -> tuple[list[dict[str, Any]], int]:
    """Garde les passages des pièces retenues, ordre d'origine. Écarte sans provenance."""
    wanted = set(pieces)
    out: list[dict[str, Any]] = []
    sans_piece = 0
    for item in passages:
        piece = piece_depuis_provenance(item)
        if piece is None:
            sans_piece += 1
            continue
        if piece in wanted:
            out.append(item)
    return out, sans_piece


def _stats_filtre(
    question: dict[str, Any],
    *,
    pieces_retenues: list[str],
    sans_piece: int,
    vivier: int,
    rendus: int,
    k_demande: int,
) -> dict[str, object]:
    return {
        "question_id": str(question.get("id") or ""),
        "pieces_retenues": list(pieces_retenues),
        "passages_sans_piece": sans_piece,
        "vivier": vivier,
        "passages_rendus": rendus,
        "k_demande": k_demande,
    }


class _RagFiltreBase:
    """Vivier RAG + stats par question. Cache froid refusé comme RagNaif."""

    etiquette = ""

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | None = None,
    ) -> None:
        self._rag = RagNaif(embed_cache, dossier=dossier)
        self._stats: dict[str, object] = {}
        self._stats_par_question: dict[str, dict[str, object]] = {}

    @property
    def stats(self) -> dict[str, object]:
        return dict(self._stats)

    @property
    def stats_par_question(self) -> dict[str, dict[str, object]]:
        return {qid: dict(payload) for qid, payload in self._stats_par_question.items()}

    def _vivier(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        return self._rag.contexte(question, k=k, connection=connection)

    def _enregistrer_stats(self, question: dict[str, Any], payload: dict[str, object]) -> None:
        self._stats = payload
        qid = str(question.get("id") or "")
        if qid:
            self._stats_par_question[qid] = dict(payload)


class RagNaifLarge(_RagFiltreBase):
    etiquette = "rag_naif_large"

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        vivier = self._vivier(question, k=K_RAG_LARGE, connection=connection)
        pieces: list[str] = []
        seen: set[str] = set()
        gardes: list[dict[str, Any]] = []
        sans_piece = 0
        for item in vivier:
            piece = piece_depuis_provenance(item)
            if piece is None:
                sans_piece += 1
                continue
            gardes.append(item)
            if piece not in seen:
                seen.add(piece)
                pieces.append(piece)
        self._enregistrer_stats(
            question,
            _stats_filtre(
                question,
                pieces_retenues=pieces,
                sans_piece=sans_piece,
                vivier=len(vivier),
                rendus=len(gardes),
                k_demande=K_RAG_LARGE,
            ),
        )
        return gardes


class RagFiltrePieces(_RagFiltreBase):
    etiquette = "rag_filtre_pieces"

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        vivier = self._vivier(question, k=K_VIVIER_FILTRE, connection=connection)
        classees, sans_piece = noter_pieces_par_passages(vivier)
        retenues = classees[:N_PIECES_FILTRE]
        gardes, _sans = filtrer_passages_par_pieces(vivier, retenues)
        self._enregistrer_stats(
            question,
            _stats_filtre(
                question,
                pieces_retenues=retenues,
                sans_piece=sans_piece,
                vivier=len(vivier),
                rendus=len(gardes),
                k_demande=K_VIVIER_FILTRE,
            ),
        )
        return gardes


class _SituationsPourFiltre:
    """Charge et classe les situations comme `brain_situations` (même cache)."""

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        situations: list[Any] | None = None,
        embeddings_situations: dict[str, list[float]] | None = None,
        embeddings_situations_path: Path | None = None,
        situations_root: Path | None = None,
        chemin: Path | None = None,
        scorer_lexical: Any = None,
    ) -> None:
        self._inner = BrainSituations(
            embed_cache,
            dossier=dossier,
            situations=situations,
            embeddings_situations=embeddings_situations,
            embeddings_situations_path=embeddings_situations_path,
            situations_root=situations_root,
            chemin=chemin,
            scorer_lexical=scorer_lexical,
        )

    def classer(self, question: dict[str, Any]) -> tuple[list[Any], list[Any]]:
        from .lecteur_faits_classes import _connexion
        from .lecteur_situations_composites import charger_situations, classer_situations
        from .mrfs_support import BANC_PG_DSN

        inner = self._inner
        own = False
        conn = None
        if inner._scorer_lexical is None:
            conn = _connexion(BANC_PG_DSN)
            own = True
        try:
            situations = (
                inner._situations
                if inner._situations is not None
                else charger_situations(
                    inner._nom,
                    chemin=inner._chemin,
                    situations_root=inner._situations_root,
                )
            )
            if inner._situations is None:
                inner._situations = list(situations)
            notes = classer_situations(
                situations,
                _texte_question(question),
                connection=conn,
                scorer_lexical=inner._scorer_lexical,
                question_vector=inner._question_vector(question),
                embeddings=inner._charger_embeddings_situations(),
            )
        finally:
            if own and conn is not None:
                conn.close()
        return notes, list(situations)


class RagFiltreSituations(_RagFiltreBase):
    etiquette = "rag_filtre_situations"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        situations: list[Any] | None = None,
        embeddings_situations: dict[str, list[float]] | None = None,
        embeddings_situations_path: Path | None = None,
        situations_root: Path | None = None,
        chemin: Path | None = None,
        scorer_lexical: Any = None,
    ) -> None:
        current = None
        if isinstance(dossier, Dossier):
            current = dossier
        elif dossier is None:
            current = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            current = dossier_courant(dossier)
        super().__init__(embed_cache, dossier=current)
        self._chargeur = _SituationsPourFiltre(
            embed_cache,
            dossier=dossier,
            situations=situations,
            embeddings_situations=embeddings_situations,
            embeddings_situations_path=embeddings_situations_path,
            situations_root=situations_root,
            chemin=chemin,
            scorer_lexical=scorer_lexical,
        )

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        vivier = self._vivier(question, k=K_VIVIER_FILTRE, connection=connection)
        notes, situations = self._chargeur.classer(question)
        retenues = pieces_depuis_classement_situations(notes, situations)
        gardes, sans_piece = filtrer_passages_par_pieces(vivier, retenues)
        self._enregistrer_stats(
            question,
            _stats_filtre(
                question,
                pieces_retenues=retenues,
                sans_piece=sans_piece,
                vivier=len(vivier),
                rendus=len(gardes),
                k_demande=K_VIVIER_FILTRE,
            ),
        )
        return gardes


class RagFiltreUnion(RagFiltreSituations):
    etiquette = "rag_filtre_union"

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        vivier = self._vivier(question, k=K_VIVIER_FILTRE, connection=connection)
        classees, _sans_note = noter_pieces_par_passages(vivier)
        pieces_rag = classees[:N_PIECES_FILTRE]
        notes, situations = self._chargeur.classer(question)
        pieces_sit = pieces_depuis_classement_situations(notes, situations)
        retenues = reunir_pieces(pieces_rag, pieces_sit)
        gardes, sans_piece = filtrer_passages_par_pieces(vivier, retenues)
        self._enregistrer_stats(
            question,
            _stats_filtre(
                question,
                pieces_retenues=retenues,
                sans_piece=sans_piece,
                vivier=len(vivier),
                rendus=len(gardes),
                k_demande=K_VIVIER_FILTRE,
            ),
        )
        return gardes


# Reconstitution puis recherche — constantes fixées avant toute mesure (#154-O, #154-P).
PART_LIGNES = 6_000
PART_LECTURE_SIMPLE = 3_000
BUDGET_ETAGE2_CHARS = 12_000


def _repertoire_reconstitution(
    dossier: Dossier | str | None,
    reconstitution_dir: Path | None,
) -> Path:
    if reconstitution_dir is not None:
        root = Path(reconstitution_dir).expanduser()
    else:
        from .recherche_entites import chemin_reconstitution

        root = chemin_reconstitution(_nom_dossier(dossier))
    if not (root / "registre.json").is_file():
        raise FileNotFoundError(
            f"reconstitution absente pour {_nom_dossier(dossier)} : {root / 'registre.json'}"
        )
    return root


def _signature_ligne_designee(piece: str, lignes: tuple[int, ...] | list[int]) -> tuple[str, tuple[int, ...]]:
    return piece_basename(piece), tuple(sorted(int(n) for n in lignes))


def _signatures_lignes_elements(elements: list[dict[str, Any]]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for item in elements:
        provenance = item.get("provenance") or {}
        piece = piece_basename(
            str(provenance.get("source_file_path") or provenance.get("piece") or item.get("source") or "")
        )
        start = provenance.get("line_start")
        end = provenance.get("line_end")
        if piece and isinstance(start, int):
            fin = end if isinstance(end, int) else start
            for num in range(start, fin + 1):
                out.add((piece, num))
    return out


def _passage_recouvre_lignes(
    item: Mapping[str, Any], deja: set[tuple[str, int]]
) -> bool:
    signature = _signature_passage_rag(item)
    if signature is None:
        return False
    piece, nums = signature
    return any((piece, num) in deja for num in nums)


def _passage_lignes_toutes_deja(
    item: Mapping[str, Any], deja: set[tuple[str, int]]
) -> bool:
    """True si toutes les lignes du passage sont déjà dans `deja`. Sans numéros → False."""
    signature = _signature_passage_rag(item)
    if signature is None:
        return False
    piece, nums = signature
    return bool(nums) and all((piece, num) in deja for num in nums)


def _signature_passage_rag(item: Mapping[str, Any]) -> tuple[str, tuple[int, ...]] | None:
    provenance = item.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    piece = piece_basename(
        str(provenance.get("source_file_path") or provenance.get("piece") or item.get("source") or "")
    )
    start = provenance.get("line_start")
    end = provenance.get("line_end")
    if not piece or not isinstance(start, int):
        return None
    if isinstance(end, int) and end != start:
        nums = list(range(start, end + 1))
    else:
        nums = [start]
    return _signature_ligne_designee(piece, nums)


def _texte_version(version: Mapping[str, Any]) -> str:
    piece = str(version.get("piece") or "")
    lignes = version.get("lignes") or []
    nums = ",".join(str(int(n)) for n in lignes)
    lettre = str(version.get("lettre") or "")
    return f"{piece}:{nums}\n{lettre}".strip()


def _elements_entites_lignes(selection) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for fait_classe in selection.faits_classes:
        fait = fait_classe.fait
        versions = list(fait.get("versions") or [])
        par_piece: dict[str, list[Mapping[str, Any]]] = {}
        for version in versions:
            piece = str(version.get("piece") or "")
            if not piece:
                continue
            par_piece.setdefault(piece, []).append(version)
        for piece, versions_piece in par_piece.items():
            blocs = [_texte_version(version) for version in versions_piece]
            texte = "\n\n".join(blocs).strip()
            if not texte:
                continue
            lignes = sorted(
                {
                    int(n)
                    for version in versions_piece
                    for n in version.get("lignes") or []
                }
            )
            provenance = {
                "source_file_path": piece,
                "citation": texte,
                "line_start": lignes[0] if lignes else None,
                "line_end": lignes[-1] if lignes else None,
            }
            elements.append(
                {
                    "kind": "entite_ligne",
                    "id": f"{fait_classe.unite_id}:{fait.get('quoi', '')[:40]}:{piece}",
                    "rank": fait_classe.rang,
                    "score": float(fait_classe.score),
                    "text": texte,
                    "source": piece,
                    "snippet": texte[:280],
                    "provenance": provenance,
                    "payload": {"metadata": provenance},
                }
            )
    return elements


def familles_lecture_simple(aiguillage: Mapping[str, Any]) -> frozenset[str]:
    """Familles dont au moins une règle d'aiguillage est en lecture simple."""
    out: set[str] = set()
    for regle in aiguillage.get("regles") or []:
        if not isinstance(regle, Mapping):
            continue
        famille = str(regle.get("famille") or "").strip()
        if not famille:
            continue
        traitement = str(regle.get("traitement") or "")
        if traitement.startswith("lecture_simple") or traitement.startswith("ocr_lecture_simple"):
            out.add(famille)
    return frozenset(out)


def _chemin_aiguillage_dossier(dossier: Dossier) -> Path:
    return BANC_VAR / "dossiers" / "drive" / dossier.nom / "aiguillage.json"


def _filtrer_passages_par_familles_lecture_simple(
    passages: list[dict[str, Any]],
    familles: frozenset[str],
    etiquettes_par_passage: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not familles:
        return []
    gardes: list[dict[str, Any]] = []
    for item in passages:
        pid = str(item.get("id") or "")
        bloc = etiquettes_par_passage.get(pid) or {}
        if str(bloc.get("famille") or "") in familles:
            gardes.append(item)
    return gardes


def _longueur_passage_contexte(item: Mapping[str, Any]) -> int:
    return len(str(item.get("text") or ""))


def _retenir_passages_sous_budget(
    passages: list[dict[str, Any]],
    *,
    budget_chars: int,
    deja: set[tuple[str, int]],
) -> tuple[list[dict[str, Any]], int]:
    retenus: list[dict[str, Any]] = []
    total = 0
    for item in passages:
        if _passage_recouvre_lignes(item, deja):
            continue
        longueur = _longueur_passage_contexte(item)
        if longueur <= 0:
            continue
        if total + longueur > budget_chars:
            break
        retenus.append(item)
        total += longueur
    return retenus, total


def _etendre_deja_passages_rag(deja: set[tuple[str, int]], passages: list[dict[str, Any]]) -> None:
    for item in passages:
        signature = _signature_passage_rag(item)
        if signature is None:
            continue
        piece, nums = signature
        for num in nums:
            deja.add((piece, num))


def _stats_entites(
    question: dict[str, Any],
    selection,
    *,
    vivier: int,
    rendus: int,
    k_demande: int,
    repli_sans_filtre: bool,
    unites_absentes: int | None = None,
    n_lecture_simple: int = 0,
    car_lecture_simple: int = 0,
    repli_lecture_simple: bool = False,
    entites_filtre: list[str] | None = None,
    n_lecture_simple_filtre: int | None = None,
    repli_entites: bool | None = None,
) -> dict[str, object]:
    nb_absentes = (
        unites_absentes
        if unites_absentes is not None
        else int(getattr(selection, "unites_absentes", 0))
    )
    payload: dict[str, object] = {
        "question_id": str(question.get("id") or ""),
        "reconnaissance": selection.reconnaissance.as_dict(),
        "unites_candidates": [
            {"id": candidate.id, "motif": candidate.motif}
            for candidate in selection.unites_candidates
        ],
        "pieces_designees": list(selection.pieces_designees),
        "n_lignes_designees": len(selection.lignes_designees),
        "repli_sans_filtre": repli_sans_filtre,
        "vivier": vivier,
        "passages_rendus": rendus,
        "k_demande": k_demande,
        "unites_absentes": nb_absentes,
        "n_lecture_simple": n_lecture_simple,
        "car_lecture_simple": car_lecture_simple,
        "repli_lecture_simple": repli_lecture_simple,
    }
    if entites_filtre is not None:
        payload["entites_filtre"] = list(entites_filtre)
    if n_lecture_simple_filtre is not None:
        payload["n_lecture_simple_filtre"] = n_lecture_simple_filtre
    if repli_entites is not None:
        payload["repli_entites"] = repli_entites
    return payload


class _EntitesRechercheBase:
    """Sélection par reconstitution — stats par question, répertoire injectable."""

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
    ) -> None:
        from .recherche_entites import charger_index

        root = _repertoire_reconstitution(dossier, reconstitution_dir)
        self._index = charger_index(root)
        self._stats: dict[str, object] = {}
        self._stats_par_question: dict[str, dict[str, object]] = {}

    @property
    def stats(self) -> dict[str, object]:
        return dict(self._stats)

    @property
    def stats_par_question(self) -> dict[str, dict[str, object]]:
        return {qid: dict(payload) for qid, payload in self._stats_par_question.items()}

    def _selection(self, question: dict[str, Any]):
        from .recherche_entites import selectionner

        return selectionner(_texte_question(question), self._index)

    def _enregistrer_stats(self, question: dict[str, Any], payload: dict[str, object]) -> None:
        self._stats = payload
        qid = str(question.get("id") or "")
        if qid:
            self._stats_par_question[qid] = dict(payload)


class RagFiltreEntites(_RagFiltreBase, _EntitesRechercheBase):
    etiquette = "rag_filtre_entites"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
    ) -> None:
        current = None
        if isinstance(dossier, Dossier):
            current = dossier
        elif dossier is None:
            current = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            current = dossier_courant(dossier)
        _RagFiltreBase.__init__(self, embed_cache, dossier=current)
        _EntitesRechercheBase.__init__(
            self,
            embed_cache,
            dossier=dossier,
            reconstitution_dir=reconstitution_dir,
        )

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        selection = self._selection(question)
        pieces = list(selection.pieces_designees)
        repli = not pieces
        if repli:
            vivier = self._vivier(question, k=K_RAG_LARGE, connection=connection)
            gardes = vivier
            k_demande = K_RAG_LARGE
        else:
            vivier = self._vivier(question, k=K_VIVIER_FILTRE, connection=connection)
            gardes, _sans_piece = filtrer_passages_par_pieces(vivier, pieces)
            k_demande = K_VIVIER_FILTRE
        self._enregistrer_stats(
            question,
            _stats_entites(
                question,
                selection,
                vivier=len(vivier),
                rendus=len(gardes),
                k_demande=k_demande,
                repli_sans_filtre=repli,
            ),
        )
        return gardes


class EntitesLignes(_EntitesRechercheBase):
    etiquette = "entites_lignes"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
    ) -> None:
        del embed_cache
        super().__init__(None, dossier=dossier, reconstitution_dir=reconstitution_dir)

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k, connection
        selection = self._selection(question)
        elements = _elements_entites_lignes(selection)
        self._enregistrer_stats(
            question,
            _stats_entites(
                question,
                selection,
                vivier=0,
                rendus=len(elements),
                k_demande=0,
                repli_sans_filtre=False,
            ),
        )
        return elements


class EntitesLignesPuisRag(_EntitesRechercheBase):
    etiquette = "entites_lignes_puis_rag"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
    ) -> None:
        current = None
        if isinstance(dossier, Dossier):
            current = dossier
        elif dossier is None:
            current = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            current = dossier_courant(dossier)
        super().__init__(None, dossier=dossier, reconstitution_dir=reconstitution_dir)
        self._rag = RagFiltreEntites(embed_cache, dossier=current, reconstitution_dir=reconstitution_dir)

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        selection = self._selection(question)
        lignes_elements = _elements_entites_lignes(selection)
        retenus: list[dict[str, Any]] = []
        total = 0
        for element in lignes_elements:
            longueur = len(str(element.get("text") or ""))
            if total + longueur > PART_LIGNES:
                break
            retenus.append(element)
            total += longueur
        deja = _signatures_lignes_elements(retenus)
        rag_passages = self._rag.contexte(question, k=K_VIVIER_FILTRE, connection=connection)
        rag_retenus: list[dict[str, Any]] = []
        for item in rag_passages:
            if _passage_recouvre_lignes(item, deja):
                continue
            rag_retenus.append(item)
        elements = retenus + rag_retenus
        rag_stats = self._rag.stats_par_question.get(str(question.get("id") or ""), {})
        self._enregistrer_stats(
            question,
            _stats_entites(
                question,
                selection,
                vivier=int(rag_stats.get("vivier") or 0),
                rendus=len(elements),
                k_demande=int(rag_stats.get("k_demande") or K_VIVIER_FILTRE),
                repli_sans_filtre=bool(rag_stats.get("repli_sans_filtre")),
            ),
        )
        return elements


def _groupes_lignes_objet(lignes) -> dict[str, list[tuple[tuple[int, ...], tuple[str, ...]]]]:
    par_piece: dict[str, list] = {}
    ordre: list[str] = []
    for ligne in lignes:
        piece = str(ligne.piece or "")
        if not piece:
            continue
        if piece not in par_piece:
            ordre.append(piece)
            par_piece[piece] = []
        par_piece[piece].append(ligne)
    groupes: dict[str, list[tuple[tuple[int, ...], tuple[str, ...]]]] = {}
    for piece in ordre:
        items = sorted(par_piece[piece], key=lambda item: item.numero)
        blocs: list[list] = []
        courant: list = []
        for item in items:
            if courant and item.numero != courant[-1].numero + 1:
                blocs.append(courant)
                courant = [item]
            else:
                courant.append(item)
        if courant:
            blocs.append(courant)
        groupes[piece] = [
            (tuple(item.numero for item in bloc), tuple(item.lettre for item in bloc))
            for bloc in blocs
        ]
    return {piece: groupes[piece] for piece in ordre}


def _texte_objet_piece(objet, piece: str, groupes_piece) -> str:
    parts: list[str] = []
    if objet.sorte == "fait":
        parts.append(f"comparaison: {objet.comparaison}")
    for nums, lettres in groupes_piece:
        nums_s = ",".join(str(n) for n in nums)
        bloc = f"{piece}:{nums_s}\n" + "\n".join(lettres)
        parts.append(bloc.strip())
    return "\n\n".join(p for p in parts if p).strip()


def _elements_objets_sens_notes(
    notes,
    *,
    budget_chars: int | None = None,
) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    rendu = 0
    for note in notes:
        if budget_chars is not None and rendu > budget_chars:
            break
        objet = note.objet
        groupes = _groupes_lignes_objet(objet.lignes)
        for piece, groupes_piece in groupes.items():
            texte = _texte_objet_piece(objet, piece, groupes_piece)
            if not texte:
                continue
            rendu += len(texte)
            nums = [n for nums, _lettres in groupes_piece for n in nums]
            provenance = {
                "source_file_path": piece,
                "citation": texte,
                "line_start": nums[0] if nums else None,
                "line_end": nums[-1] if nums else None,
                "objet_id": objet.id,
                "sorte": objet.sorte,
            }
            elements.append(
                {
                    "kind": "objet_sens",
                    "id": f"{objet.id}:{piece}",
                    "rank": note.rang,
                    "score": float(note.score),
                    "text": texte,
                    "source": piece,
                    "snippet": texte[:280],
                    "provenance": provenance,
                    "payload": {"metadata": provenance},
                }
            )
    return elements


def _elements_objets_sens(selection) -> list[dict[str, Any]]:
    # Tous les objets sont classés ; seuls les premiers sont rendus, jusqu'au plus grand
    # budget du banc. Rendre les 2 466 objets pour chaque question épuise la mémoire.
    from .mrfs_support import MAX_BUDGET_CHARS

    return _elements_objets_sens_notes(selection.objets_classes, budget_chars=MAX_BUDGET_CHARS)


class _SensRechercheBase:
    """Classement par le sens — vecteur de question du cache du banc, cache froid refusé."""

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
        embeddings_objets: dict[str, list[float]] | None = None,
    ) -> None:
        from .embeddings_faits import charger_cache
        from .recherche_entites import charger_index, objets_de_recherche

        root = _repertoire_reconstitution(dossier, reconstitution_dir)
        self._index = charger_index(root)
        self._objets = objets_de_recherche(self._index)
        self._embeddings_objets = embeddings_objets
        self._embeddings_objets_path = root / "embeddings-objets.json"
        if isinstance(dossier, Dossier):
            self._dossier = dossier
        elif dossier is None:
            self._dossier = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            self._dossier = dossier_courant(dossier)
        else:
            self._dossier = None
        self._embed_cache = embed_cache
        self._stats: dict[str, object] = {}
        self._stats_par_question: dict[str, dict[str, object]] = {}
        self._charger_cache_fn = charger_cache

    @property
    def stats(self) -> dict[str, object]:
        return dict(self._stats)

    @property
    def stats_par_question(self) -> dict[str, dict[str, object]]:
        return {qid: dict(payload) for qid, payload in self._stats_par_question.items()}

    def _vecteur_question(self, question: dict[str, Any]) -> list[float]:
        cache = self._embed_cache
        if cache is None:
            if self._dossier is None:
                raise RuntimeError("cache froid : embedding de question absent")
            from .measure_etage2 import _load_embed_cache

            cache = _load_embed_cache(self._dossier)
            self._embed_cache = cache
        return _require_question_vector(question, cache)

    def _vecteurs_objets(self) -> dict[str, list[float]]:
        if self._embeddings_objets is not None:
            return self._embeddings_objets
        cache = self._charger_cache_fn(self._embeddings_objets_path)
        self._embeddings_objets = cache
        return cache

    def _selection(self, question: dict[str, Any]):
        from .recherche_entites import selectionner_par_le_sens

        return selectionner_par_le_sens(
            _texte_question(question),
            self._index,
            vecteur_question=self._vecteur_question(question),
            vecteurs_objets=self._vecteurs_objets(),
        )

    def _enregistrer_stats(self, question: dict[str, Any], payload: dict[str, object]) -> None:
        self._stats = payload
        qid = str(question.get("id") or "")
        if qid:
            self._stats_par_question[qid] = dict(payload)


def _stats_ontologie(
    question: dict[str, Any],
    *,
    entites_filtre: list[str],
    entites_relations: list[str],
    vivier: int,
    rendus: int,
    repli_sans_filtre: bool,
    etiquettes_version: str,
) -> dict[str, object]:
    return {
        "question_id": str(question.get("id") or ""),
        "entites_filtre": list(entites_filtre),
        "entites_relations": list(entites_relations),
        "repli_sans_filtre": repli_sans_filtre,
        "vivier": vivier,
        "passages_rendus": rendus,
        "k_demande": K_VIVIER_FILTRE,
        "etiquettes_version": etiquettes_version,
    }


class RagFiltreOntologie(_RagFiltreBase):
    etiquette = "rag_filtre_ontologie"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        etiquettes_path: Path | None = None,
        ontologie_version: str = "v1",
    ) -> None:
        from .etiquetage_ontologie import _chemin_etiquettes, charger_etiquettes

        current = None
        if isinstance(dossier, Dossier):
            current = dossier
        elif dossier is None:
            current = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            current = dossier_courant(dossier)
        _RagFiltreBase.__init__(self, embed_cache, dossier=current)
        self._ontologie_version = ontologie_version
        nom = current.nom if current is not None else "greenloop"
        path = etiquettes_path or _chemin_etiquettes(nom, ontologie_version)
        payload = charger_etiquettes(path)
        self._etiquettes_passages: dict[str, dict[str, Any]] = dict(payload.get("passages") or {})
        ontologie_path = path.parent / f"ontologie.{ontologie_version}.json"
        self._ontologie: dict[str, Any] = json.loads(ontologie_path.read_text(encoding="utf-8"))

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        from .etiquetage_ontologie import (
            entites_filtre_depuis_question,
            filtrer_passages_par_entites_etiquettes,
        )

        texte = str(question.get("question") or "")
        directes, relations, filtre_complet, repli = entites_filtre_depuis_question(
            texte,
            self._ontologie,
        )
        vivier = self._vivier(question, k=K_VIVIER_FILTRE, connection=connection)
        if repli:
            gardes = vivier
        else:
            gardes = filtrer_passages_par_entites_etiquettes(
                vivier,
                filtre_complet,
                self._etiquettes_passages,
            )
        self._enregistrer_stats(
            question,
            _stats_ontologie(
                question,
                entites_filtre=directes,
                entites_relations=relations,
                vivier=len(vivier),
                rendus=len(gardes),
                repli_sans_filtre=repli,
                etiquettes_version=self._ontologie_version,
            ),
        )
        return gardes


class RagFiltreSens(_RagFiltreBase, _SensRechercheBase):
    etiquette = "rag_filtre_sens"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
        embeddings_objets: dict[str, list[float]] | None = None,
    ) -> None:
        current = None
        if isinstance(dossier, Dossier):
            current = dossier
        elif dossier is None:
            current = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            current = dossier_courant(dossier)
        _RagFiltreBase.__init__(self, embed_cache, dossier=current)
        _SensRechercheBase.__init__(
            self,
            embed_cache,
            dossier=dossier,
            reconstitution_dir=reconstitution_dir,
            embeddings_objets=embeddings_objets,
        )

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        selection = self._selection(question)
        pieces = list(selection.pieces_designees)
        vivier = self._vivier(question, k=K_VIVIER_FILTRE, connection=connection)
        gardes, _sans_piece = filtrer_passages_par_pieces(vivier, pieces)
        self._enregistrer_stats(
            question,
            _stats_entites(
                question,
                selection,
                vivier=len(vivier),
                rendus=len(gardes),
                k_demande=K_VIVIER_FILTRE,
                repli_sans_filtre=False,
            ),
        )
        return gardes


class ObjetsSens(_SensRechercheBase):
    etiquette = "objets_sens"

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k, connection
        selection = self._selection(question)
        elements = _elements_objets_sens(selection)
        self._enregistrer_stats(
            question,
            _stats_entites(
                question,
                selection,
                vivier=0,
                rendus=len(elements),
                k_demande=0,
                repli_sans_filtre=False,
            ),
        )
        return elements


class ObjetsSensPuisRag(_SensRechercheBase):
    etiquette = "objets_sens_puis_rag"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
        embeddings_objets: dict[str, list[float]] | None = None,
    ) -> None:
        super().__init__(
            embed_cache,
            dossier=dossier,
            reconstitution_dir=reconstitution_dir,
            embeddings_objets=embeddings_objets,
        )
        current = None
        if isinstance(dossier, Dossier):
            current = dossier
        elif dossier is None:
            current = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            current = dossier_courant(dossier)
        self._rag = RagFiltreSens(
            embed_cache,
            dossier=current,
            reconstitution_dir=reconstitution_dir,
            embeddings_objets=embeddings_objets,
        )

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        selection = self._selection(question)
        lignes_elements = _elements_objets_sens(selection)
        retenus: list[dict[str, Any]] = []
        total = 0
        for element in lignes_elements:
            longueur = len(str(element.get("text") or ""))
            if total + longueur > PART_LIGNES:
                break
            retenus.append(element)
            total += longueur
        deja = _signatures_lignes_elements(retenus)
        rag_passages = self._rag.contexte(question, k=K_VIVIER_FILTRE, connection=connection)
        rag_retenus: list[dict[str, Any]] = []
        for item in rag_passages:
            if _passage_recouvre_lignes(item, deja):
                continue
            rag_retenus.append(item)
        elements = retenus + rag_retenus
        rag_stats = self._rag.stats_par_question.get(str(question.get("id") or ""), {})
        self._enregistrer_stats(
            question,
            _stats_entites(
                question,
                selection,
                vivier=int(rag_stats.get("vivier") or 0),
                rendus=len(elements),
                k_demande=int(rag_stats.get("k_demande") or K_VIVIER_FILTRE),
                repli_sans_filtre=bool(rag_stats.get("repli_sans_filtre")),
            ),
        )
        return elements


class ObjetsSensPuisRagLectureSimple(_SensRechercheBase):
    etiquette = "objets_sens_puis_rag_lecture_simple"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
        embeddings_objets: dict[str, list[float]] | None = None,
        etiquettes_path: Path | None = None,
        ontologie_version: str = "v1",
        part_lecture_simple: int = PART_LECTURE_SIMPLE,
    ) -> None:
        super().__init__(
            embed_cache,
            dossier=dossier,
            reconstitution_dir=reconstitution_dir,
            embeddings_objets=embeddings_objets,
        )
        self._part_lecture_simple = part_lecture_simple
        from .etiquetage_ontologie import _chemin_etiquettes, charger_etiquettes
        from .ontologie_tenant import charger_aiguillage

        current = None
        if isinstance(dossier, Dossier):
            current = dossier
        elif dossier is None:
            current = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            current = dossier_courant(dossier)
        self._rag = RagFiltreSens(
            embed_cache,
            dossier=current,
            reconstitution_dir=reconstitution_dir,
            embeddings_objets=embeddings_objets,
        )
        nom = current.nom if current is not None else "greenloop"
        path = etiquettes_path or _chemin_etiquettes(nom, ontologie_version)
        payload = charger_etiquettes(path)
        self._etiquettes_passages: dict[str, dict[str, Any]] = dict(payload.get("passages") or {})
        if current is not None and _chemin_aiguillage_dossier(current).is_file():
            self._familles_lecture_simple = familles_lecture_simple(
                charger_aiguillage(_chemin_aiguillage_dossier(current))
            )
        else:
            self._familles_lecture_simple = frozenset()

    def _preparer_candidats_lecture_simple(
        self,
        question: dict[str, Any],
        candidats_ls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, object]]:
        return candidats_ls, {}

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        selection = self._selection(question)
        lignes_elements = _elements_objets_sens(selection)
        retenus: list[dict[str, Any]] = []
        total_obj = 0
        for element in lignes_elements:
            longueur = len(str(element.get("text") or ""))
            if total_obj + longueur > PART_LIGNES:
                break
            retenus.append(element)
            total_obj += longueur
        deja = _signatures_lignes_elements(retenus)

        vivier = self._rag._vivier(question, k=K_VIVIER_FILTRE, connection=connection)
        candidats_ls = _filtrer_passages_par_familles_lecture_simple(
            vivier,
            self._familles_lecture_simple,
            self._etiquettes_passages,
        )
        candidats_ls, extra_lecture = self._preparer_candidats_lecture_simple(question, candidats_ls)
        repli_lecture_simple = len(candidats_ls) == 0
        ls_retenus: list[dict[str, Any]] = []
        car_ls = 0
        part_ls = self._part_lecture_simple
        if not repli_lecture_simple:
            ls_retenus, car_ls = _retenir_passages_sous_budget(
                candidats_ls,
                budget_chars=part_ls,
                deja=deja,
            )
            _etendre_deja_passages_rag(deja, ls_retenus)

        if repli_lecture_simple:
            rag_budget = BUDGET_ETAGE2_CHARS - total_obj
        else:
            rag_budget = BUDGET_ETAGE2_CHARS - total_obj - part_ls + car_ls

        rag_passages = self._rag.contexte(question, k=K_VIVIER_FILTRE, connection=connection)
        rag_retenus: list[dict[str, Any]] = []
        rag_chars = 0
        for item in rag_passages:
            if _passage_recouvre_lignes(item, deja):
                continue
            longueur = _longueur_passage_contexte(item)
            if longueur <= 0:
                continue
            if rag_chars + longueur > rag_budget:
                break
            rag_retenus.append(item)
            rag_chars += longueur

        elements = retenus + ls_retenus + rag_retenus
        rag_stats = self._rag.stats_par_question.get(str(question.get("id") or ""), {})
        stats_extra: dict[str, object] = {}
        for key in ("entites_filtre", "n_lecture_simple_filtre", "repli_entites"):
            if key in extra_lecture:
                stats_extra[key] = extra_lecture[key]
        self._enregistrer_stats(
            question,
            _stats_entites(
                question,
                selection,
                vivier=len(vivier),
                rendus=len(elements),
                k_demande=int(rag_stats.get("k_demande") or K_VIVIER_FILTRE),
                repli_sans_filtre=bool(rag_stats.get("repli_sans_filtre")),
                n_lecture_simple=len(ls_retenus),
                car_lecture_simple=car_ls,
                repli_lecture_simple=repli_lecture_simple,
                entites_filtre=stats_extra.get("entites_filtre")
                if "entites_filtre" in stats_extra
                else None,
                n_lecture_simple_filtre=stats_extra.get("n_lecture_simple_filtre")
                if "n_lecture_simple_filtre" in stats_extra
                else None,
                repli_entites=stats_extra.get("repli_entites")
                if "repli_entites" in stats_extra
                else None,
            ),
        )
        return elements


class ObjetsSensPuisRagLectureSimpleEntites(ObjetsSensPuisRagLectureSimple):
    etiquette = "objets_sens_puis_rag_lecture_simple_entites"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
        embeddings_objets: dict[str, list[float]] | None = None,
        etiquettes_path: Path | None = None,
        ontologie_version: str = "v1",
        part_lecture_simple: int = PART_LECTURE_SIMPLE,
    ) -> None:
        super().__init__(
            embed_cache,
            dossier=dossier,
            reconstitution_dir=reconstitution_dir,
            embeddings_objets=embeddings_objets,
            etiquettes_path=etiquettes_path,
            ontologie_version=ontologie_version,
            part_lecture_simple=part_lecture_simple,
        )
        from .etiquetage_ontologie import _chemin_etiquettes

        current = None
        if isinstance(dossier, Dossier):
            current = dossier
        elif dossier is None:
            current = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            current = dossier_courant(dossier)
        nom = current.nom if current is not None else "greenloop"
        path = etiquettes_path or _chemin_etiquettes(nom, ontologie_version)
        ontologie_path = path.parent / f"ontologie.{ontologie_version}.json"
        self._ontologie: dict[str, Any] = json.loads(ontologie_path.read_text(encoding="utf-8"))

    def _preparer_candidats_lecture_simple(
        self,
        question: dict[str, Any],
        candidats_ls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, object]]:
        from .etiquetage_ontologie import (
            entites_filtre_depuis_question,
            filtrer_passages_par_entites_etiquettes,
        )

        texte = str(question.get("question") or "")
        directes, _relations, filtre_complet, repli = entites_filtre_depuis_question(
            texte,
            self._ontologie,
        )
        extra: dict[str, object] = {
            "entites_filtre": list(directes),
            "repli_entites": repli,
        }
        if repli:
            extra["n_lecture_simple_filtre"] = len(candidats_ls)
            return candidats_ls, extra
        filtres = filtrer_passages_par_entites_etiquettes(
            candidats_ls,
            filtre_complet,
            self._etiquettes_passages,
        )
        extra["n_lecture_simple_filtre"] = len(filtres)
        return filtres, extra


def _stats_desaccords_puis_rag(
    question: dict[str, Any],
    selection,
    *,
    desaccords_en_tete: list[str],
    taille_tete_chars: int,
    vivier: int,
    rendus: int,
    k_demande: int,
) -> dict[str, object]:
    payload = _stats_entites(
        question,
        selection,
        vivier=vivier,
        rendus=rendus,
        k_demande=k_demande,
        repli_sans_filtre=False,
    )
    payload["desaccords_en_tete"] = list(desaccords_en_tete)
    payload["taille_tete_chars"] = taille_tete_chars
    return payload


class _DesaccordsPuisRagBase(_SensRechercheBase):
    """Désaccords en tête (toutes versions) puis RAG — cache froid refusé."""

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
        embeddings_objets: dict[str, list[float]] | None = None,
    ) -> None:
        super().__init__(
            embed_cache,
            dossier=dossier,
            reconstitution_dir=reconstitution_dir,
            embeddings_objets=embeddings_objets,
        )
        current = None
        if isinstance(dossier, Dossier):
            current = dossier
        elif dossier is None:
            current = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            current = dossier_courant(dossier)
        self._rag = self._construire_rag(
            embed_cache,
            dossier=current,
            reconstitution_dir=reconstitution_dir,
            embeddings_objets=embeddings_objets,
        )

    def _construire_rag(
        self,
        embed_cache: dict[str, list[float]] | None,
        *,
        dossier: Dossier | None,
        reconstitution_dir: Path | None,
        embeddings_objets: dict[str, list[float]] | None,
    ):
        raise NotImplementedError

    def _tete(self, selection) -> tuple[list[dict[str, Any]], list[str], int]:
        from .recherche_entites import desaccords_en_tete

        notes = desaccords_en_tete(selection)
        elements = _elements_objets_sens_notes(notes, budget_chars=None)
        ids = [note.objet.id for note in notes]
        taille = sum(len(str(element.get("text") or "")) for element in elements)
        return elements, ids, taille

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        selection = self._selection(question)
        tete, ids, taille = self._tete(selection)
        deja = _signatures_lignes_elements(tete)
        rag_passages = self._rag.contexte(question, k=0, connection=connection)
        rag_retenus: list[dict[str, Any]] = []
        for item in rag_passages:
            if _passage_recouvre_lignes(item, deja):
                continue
            rag_retenus.append(item)
        elements = tete + rag_retenus
        rag_stats = self._rag.stats_par_question.get(str(question.get("id") or ""), {})
        self._enregistrer_stats(
            question,
            _stats_desaccords_puis_rag(
                question,
                selection,
                desaccords_en_tete=ids,
                taille_tete_chars=taille,
                vivier=int(rag_stats.get("vivier") or 0),
                rendus=len(elements),
                k_demande=int(rag_stats.get("k_demande") or 0),
            ),
        )
        return elements


class DesaccordsPuisRag(_DesaccordsPuisRagBase):
    etiquette = "desaccords_puis_rag"

    def _construire_rag(
        self,
        embed_cache: dict[str, list[float]] | None,
        *,
        dossier: Dossier | None,
        reconstitution_dir: Path | None,
        embeddings_objets: dict[str, list[float]] | None,
    ):
        return RagNaifLarge(embed_cache, dossier=dossier)


class DesaccordsPuisRagFiltre(_DesaccordsPuisRagBase):
    etiquette = "desaccords_puis_rag_filtre"

    def _construire_rag(
        self,
        embed_cache: dict[str, list[float]] | None,
        *,
        dossier: Dossier | None,
        reconstitution_dir: Path | None,
        embeddings_objets: dict[str, list[float]] | None,
    ):
        return RagFiltreSens(
            embed_cache,
            dossier=dossier,
            reconstitution_dir=reconstitution_dir,
            embeddings_objets=embeddings_objets,
        )


class DesaccordsPuisG12(_SensRechercheBase):
    """Désaccords en tête (toutes versions) puis g12 — cache froid refusé."""

    etiquette = "desaccords_puis_g12"

    def __init__(
        self,
        embed_cache: dict[str, list[float]] | None = None,
        phrases_cache: dict[str, list[float]] | None = None,
        *,
        dossier: Dossier | str | None = None,
        reconstitution_dir: Path | None = None,
        embeddings_objets: dict[str, list[float]] | None = None,
    ) -> None:
        super().__init__(
            embed_cache,
            dossier=dossier,
            reconstitution_dir=reconstitution_dir,
            embeddings_objets=embeddings_objets,
        )
        current = None
        if isinstance(dossier, Dossier):
            current = dossier
        elif dossier is None:
            current = dossier_courant()
        elif isinstance(dossier, str) and dossier in DOSSIERS:
            current = dossier_courant(dossier)
        self._g12 = BrainBanc(embed_cache, phrases_cache, dossier=current)

    def _tete(self, selection) -> tuple[list[dict[str, Any]], list[str], int]:
        from .recherche_entites import desaccords_en_tete

        notes = desaccords_en_tete(selection)
        elements = _elements_objets_sens_notes(notes, budget_chars=None)
        ids = [note.objet.id for note in notes]
        taille = sum(len(str(element.get("text") or "")) for element in elements)
        return elements, ids, taille

    def contexte(self, question: dict[str, Any], *, k: int, connection) -> list[dict[str, Any]]:
        del k
        selection = self._selection(question)
        tete, ids, taille = self._tete(selection)
        deja = _signatures_lignes_elements(tete)
        g12_passages = self._g12.contexte(question, k=0, connection=connection)
        g12_retenus: list[dict[str, Any]] = []
        for item in g12_passages:
            if _passage_lignes_toutes_deja(item, deja):
                continue
            g12_retenus.append(item)
        elements = tete + g12_retenus
        self._enregistrer_stats(
            question,
            _stats_desaccords_puis_rag(
                question,
                selection,
                desaccords_en_tete=ids,
                taille_tete_chars=taille,
                vivier=len(g12_passages),
                rendus=len(elements),
                k_demande=12,
            ),
        )
        return elements
