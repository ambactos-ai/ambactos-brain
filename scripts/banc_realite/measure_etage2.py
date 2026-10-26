# SPDX-License-Identifier: AGPL-3.0-only
"""Measure étage 2 restitution for all GreenLoop questions."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from banc_realite.compat_moteur import MistralEmbeddingError, embed_with_retry
from banc_realite.compat_moteur import _build_structured_query, compile_call_frame
from banc_realite.compat_moteur import run_search

WORKTREE = Path(__file__).resolve().parents[2]
for entry in (WORKTREE / "scripts", WORKTREE / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

try:
    from eval_call_frame_injection import determinisme_ok
except ImportError:
    determinisme_ok = None  # type: ignore[misc,assignment]

from .config import AREA, GOLD, INSTANCE, SORTIES, VAULT_REF, export_runtime_env
from .dossiers import Dossier, dossier_courant
from .diagnostic import collect_missing_diagnostics
from .fidelite import load_gold
from .holders import (
    PieceCache,
    assertion_present_in_evidence,
    build_holder_table,
    classify_pair,
    collect_appui_states,
    format_claim_triplet,
    noise_count,
    overlap_index,
)
from .normalize import claim_text, piece_basename
from .restitution import compute_verdict, rendered_presence
from .search_health import is_search_healthy, summarize_search_health

EMBED_CACHE_PATH = SORTIES / "etage2-question-embeddings.json"


def measure_output_dir() -> Path:
    custom = os.environ.get("BANC_MEASURE_OUTPUT_DIR")
    return Path(custom) if custom else SORTIES


def prechauffage_status_path() -> Path:
    return measure_output_dir() / "_prechauffage-status.json"


PRECHAUFFAGE_STATUS_PATH = SORTIES / "_prechauffage-status.json"


def questions_journal_path(label: str) -> Path:
    base = measure_output_dir()
    if label == "quotas-doux":
        return base / "etage2-C-quotas-doux.questions.jsonl"
    return base / "etage2-C.questions.jsonl"


def measure_status_path(label: str) -> Path:
    return measure_output_dir() / f"_measure-{label}-status.json"


def load_questions(dossier=None) -> list[dict[str, Any]]:
    from .or_dossier import load_questions as _load_questions_dossier

    return _load_questions_dossier(dossier)


def _read_public_claims(connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path, "
        "source_span, extracted_by_skill, lifecycle_status "
        "FROM public.claims WHERE lifecycle_status = 'active' AND vault_ref = %s "
        "ORDER BY created_at, claim_id",
        (VAULT_REF,),
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["text"] = claim_text(item.get("value"))
        item["piece"] = piece_basename(item.get("source_file_path") or "")
        out.append(item)
    return out


def _make_present_fn(
    evidence: list[dict[str, Any]],
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    mode: str,
    *,
    mode_presence: str = "verbatim",
):
    def present(aid: str) -> tuple[bool, int | None]:
        return assertion_present_in_evidence(
            assertion_index[aid],
            evidence,
            claims_index,
            mode,
            mode_presence=mode_presence,
        )

    return present


def _appui_summary(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    mode_presence: str = "verbatim",
) -> dict[str, Any]:
    supports = question.get("supporting_assertions") or []
    tient_ids: list[str] = []
    candidat_ids: list[str] = []
    absents: list[str] = []
    detail: dict[str, Any] = {}
    for aid in supports:
        assertion = assertion_index[aid]
        states = collect_appui_states(
            assertion,
            evidence,
            claims_index,
            mode_presence=mode_presence,
        )
        detail[aid] = states
        if states["tient"]:
            tient_ids.append(aid)
        elif states["candidat"]:
            candidat_ids.append(aid)
        else:
            absents.append(aid)
    return {
        "tient": tient_ids,
        "candidat": candidat_ids,
        "absents": absents,
        "detail": detail,
    }


def _collect_paires(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    assertion_index: dict[str, dict[str, Any]],
    claims_index: dict[str, dict[str, Any]],
    cache: PieceCache,
) -> list[dict[str, Any]]:
    paires: list[dict[str, Any]] = []
    for aid in question.get("supporting_assertions") or []:
        assertion = assertion_index[aid]
        if assertion_present_in_evidence(assertion, evidence, claims_index, "plancher")[0]:
            continue
        piece = piece_basename(assertion.get("source") or "")
        seen_claims: set[str] = set()
        for item in evidence:
            if classify_pair(assertion, item, claims_index) != "candidat":
                continue
            claim_id = str(item.get("id") or "")
            if claim_id in seen_claims:
                continue
            seen_claims.add(claim_id)
            claim = claims_index.get(claim_id) or {}
            paires.append(
                {
                    "assertion_id": aid,
                    "claim_id": claim_id,
                    "overlap": overlap_index(assertion, claim),
                    "question_id": question["id"],
                    "assertion": {
                        "id": aid,
                        "piece": piece,
                        "content": assertion.get("content"),
                        "quote": assertion.get("quote"),
                        "extrait": cache.excerpt(piece, assertion.get("quote") or ""),
                    },
                    "fait": {
                        "claim_id": claim_id,
                        "triplet": format_claim_triplet(claim),
                    },
                }
            )
    return paires


def aggregate_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_forme_diff_k8: dict[str, dict[str, dict[str, dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: {"plancher": defaultdict(int), "plafond": defaultdict(int)})
    )
    by_forme_diff_k20: dict[str, dict[str, dict[str, dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: {"plancher": defaultdict(int), "plafond": defaultdict(int)})
    )
    un_seul_cote_k8: list[str] = []
    un_seul_cote_k20: list[str] = []
    causes = Counter()
    for entry in entries:
        shape = entry.get("shape") or "unknown"
        diff = entry.get("difficulty") or "unknown"
        for k_label, store in (("k8", by_forme_diff_k8), ("k20", by_forme_diff_k20)):
            block = entry[k_label]
            for mode in ("plancher", "plafond"):
                vkey = block[f"verdict_{mode}"].get("verdict", "unknown")
                store[shape][diff][mode][vkey] += 1
            if shape == "dispute":
                if block["verdict_plancher"].get("verdict") == "un_seul_cote":
                    if k_label == "k8":
                        un_seul_cote_k8.append(entry["id"])
                    else:
                        un_seul_cote_k20.append(entry["id"])
        for diag in entry.get("attribution_b5", []):
            causes[diag.get("cause") or "unknown"] += 1
    return {
        "par_forme_difficulte_k8": {
            shape: {
                diff: {mode: dict(vals) for mode, vals in modes.items()}
                for diff, modes in diffs.items()
            }
            for shape, diffs in by_forme_diff_k8.items()
        },
        "par_forme_difficulte_k20": {
            shape: {
                diff: {mode: dict(vals) for mode, vals in modes.items()}
                for diff, modes in diffs.items()
            }
            for shape, diffs in by_forme_diff_k20.items()
        },
        "disputes_un_seul_cote_k8": un_seul_cote_k8,
        "disputes_un_seul_cote_k20": un_seul_cote_k20,
        "causes_b5": dict(causes),
    }


class SearchTracker:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def search_fn(self, query: str, top_k: int, *, embed_fn) -> dict[str, Any]:
        result = run_search(
            query,
            instance=INSTANCE,
            top_k=top_k,
            repo_root=AREA,
            embed_fn=embed_fn,
            vault_ref=VAULT_REF,
        )
        healthy, reason = is_search_healthy(result)
        self.records.append(
            {
                "query_prefix": query[:80],
                "top_k": top_k,
                "healthy": healthy,
                "reason": reason,
                "fusion": (result.get("strategy") or {}).get("fusion"),
                "warnings": list(result.get("warnings") or []),
                "notes": list((result.get("strategy") or {}).get("notes") or []),
            }
        )
        return result


def _embed_cache_path(dossier: Dossier | None = None) -> Path:
    current = dossier if dossier is not None else dossier_courant()
    return current.cache_embeddings_questions


def _load_embed_cache(dossier: Dossier | None = None) -> dict[str, list[float]]:
    path = _embed_cache_path(dossier)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_embed_cache(cache: dict[str, list[float]], dossier: Dossier | None = None) -> None:
    path = _embed_cache_path(dossier)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


def cached_embed_factory(cache: dict[str, list[float]]):
    def cached_embed(query: str) -> list[float]:
        if query not in cache:
            raise RuntimeError(f"cache froid : embedding absent ({query[:60]}…)")
        return cache[query]

    return cached_embed


def _fake_search_result() -> dict[str, Any]:
    return {
        "evidence": [],
        "warnings": [],
        "strategy": {"fusion": "rrf_v1", "notes": []},
    }


def collect_required_embed_texts(questions: list[dict[str, Any]]) -> set[str]:
    """Recording pass: texts that run_search will pass to embed_fn during measurement."""
    required: set[str] = set()

    def recording_embed_fn(query: str) -> list[float]:
        required.add(query)
        raise RuntimeError("enregistrement seulement")

    def recording_search_fn(query: str, top_k: int) -> dict[str, Any]:
        try:
            recording_embed_fn(query)
        except RuntimeError:
            pass
        return _fake_search_result()

    for question in questions:
        qid = question["id"]
        question_text = question["question"]
        req = {
            "agent": "banc-realite",
            "responsibility": "GreenLoop",
            "conversation_id": f"banc-etage2-{qid}-k8",
            "user_message": question_text,
            "conversation_state": {"responsibility_context": "GreenLoop"},
            "top_k": 8,
        }
        compile_call_frame(req, search_fn=recording_search_fn)
        try:
            recording_embed_fn(question_text)
        except RuntimeError:
            pass
    return required


def embed_question_slow(text: str) -> list[float]:
    for attempt in range(10):
        try:
            return embed_with_retry(text, attempts=1)
        except MistralEmbeddingError as exc:
            message = str(exc)
            if "HTTP 429" in message and attempt < 9:
                time.sleep(30)
                continue
            raise
    raise MistralEmbeddingError("échec embedding après 10 tentatives 429")


def prechauffer_questions() -> dict[str, Any]:
    measure_output_dir().mkdir(parents=True, exist_ok=True)
    cache = _load_embed_cache()
    questions = load_questions()
    required = collect_required_embed_texts(questions)
    example = questions[0]
    example_nue = example["question"]
    example_struct = _build_structured_query(
        example_nue,
        "GreenLoop",
        {"responsibility_context": "GreenLoop"},
    )
    to_warm = sorted(text for text in required if text not in cache)
    warmed = 0
    for text in to_warm:
        started = time.monotonic()
        cache[text] = embed_question_slow(text)
        _write_embed_cache(cache)
        warmed += 1
        elapsed = time.monotonic() - started
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
    payload = {
        "status": "complete",
        "questions_total": len(questions),
        "textes_distincts_requis": len(required),
        "textes_distincts_en_cache": len(cache),
        "warmed": warmed,
        "skipped": len(required) - len(to_warm),
        "exemple_question_nue": example_nue,
        "exemple_requete_structuree": example_struct,
        "cache_path": str(EMBED_CACHE_PATH),
    }
    prechauffage_status_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _load_journal_entries(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("degradee"):
            continue
        if not entry.get("search_sain", True):
            continue
        completed[entry["id"]] = entry
    return completed


def _append_journal_entry(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def measure_all(
    connection,
    *,
    population_caps: dict[str, float] | None = None,
    label: str = "standard",
    parity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if population_caps:
        os.environ["BRAIN_SEARCH_POPULATION_CAPS"] = json.dumps(population_caps)
    else:
        os.environ.pop("BRAIN_SEARCH_POPULATION_CAPS", None)

    gold = load_gold()
    claims = _read_public_claims(connection)
    claims_index = {str(c["claim_id"]): c for c in claims}
    cache = PieceCache()
    holder_table, holder_seconds, tient_without_piece = build_holder_table(claims, gold, cache=cache)

    embed_cache = _load_embed_cache()
    tracker = SearchTracker()

    questions = load_questions()
    journal_path = questions_journal_path(label)
    completed = _load_journal_entries(journal_path)
    entries: list[dict[str, Any]] = []
    all_paires: list[dict[str, Any]] = []
    degraded_questions: list[dict[str, Any]] = []

    for question in questions:
        qid = question["id"]
        if qid in completed:
            entry = completed[qid]
            entries.append(entry)
            if "k20" in entry:
                all_paires.extend(
                    _collect_paires(
                        question,
                        _evidence_from_entry(entry),
                        assertion_index=gold["assertion_index"],
                        claims_index=claims_index,
                        cache=cache,
                    )
                )
            continue

        entry: dict[str, Any] = {
            "id": qid,
            "shape": question.get("shape"),
            "difficulty": question.get("difficulty"),
        }
        question_text = question["question"]
        embed_fn = cached_embed_factory(embed_cache)

        def search_fn(query: str, top_k: int) -> dict[str, Any]:
            return tracker.search_fn(query, top_k, embed_fn=embed_fn)

        question_degraded = False
        degrade_reasons: list[str] = []
        evidence_k20: list[dict[str, Any]] = []

        for k in (8, 20):
            req = {
                "agent": "banc-realite",
                "responsibility": "GreenLoop",
                "conversation_id": f"banc-etage2-{qid}-k{k}",
                "user_message": question["question"],
                "conversation_state": {"responsibility_context": "GreenLoop"},
                "top_k": k,
            }
            t0 = time.perf_counter()
            frame_a = compile_call_frame(req, search_fn=search_fn)
            frame_b = compile_call_frame(req, search_fn=search_fn)
            latence_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            evidence = list((frame_a.get("retrieval") or {}).get("evidence") or [])
            if k == 20:
                evidence_k20 = evidence
            rendered = str(frame_a.get("rendered") or "")
            support_ids = question.get("supporting_assertions") or []
            appuis = _appui_summary(
                question,
                evidence,
                assertion_index=gold["assertion_index"],
                claims_index=claims_index,
            )
            verdicts = {}
            for mode in ("plancher", "plafond"):
                present = _make_present_fn(evidence, gold["assertion_index"], claims_index, mode)
                verdicts[mode] = compute_verdict(
                    question,
                    evidence,
                    gold=gold,
                    holder_table=holder_table,
                    claims_index=claims_index,
                    present=present,
                )
            presence = []
            for aid in support_ids:
                for holder in holder_table.get(aid, []):
                    claim = claims_index.get(holder["claim_id"])
                    if claim:
                        presence.append(
                            {
                                "assertion": aid,
                                "claim_id": holder["claim_id"],
                                "part_texte_dans_rendered": rendered_presence(claim, rendered),
                            }
                        )
            entry[f"k{k}"] = {
                "verdict_plancher": verdicts["plancher"],
                "verdict_plafond": verdicts["plafond"],
                "appuis": appuis,
                "n_evidence": len(evidence),
                "bruit_plancher": noise_count(
                    evidence,
                    support_ids,
                    assertion_index=gold["assertion_index"],
                    claims_index=claims_index,
                    mode="plancher",
                ),
                "bruit_plafond": noise_count(
                    evidence,
                    support_ids,
                    assertion_index=gold["assertion_index"],
                    claims_index=claims_index,
                    mode="plafond",
                ),
                "kinds": dict(Counter(item.get("kind") or "unknown" for item in evidence)),
                "caracteres_rendered": len(rendered),
                "tokens_estimes": round(len(rendered) / 4, 2),
                "determinisme_ok": determinisme_ok(frame_a.get("frame_id"), frame_b.get("frame_id")),
                "latence_ms": latence_ms,
                "frame_id": frame_a.get("frame_id"),
                "presence_rendered": presence,
                "evidence_ids": [e.get("id") for e in evidence],
            }
            if k == 20:
                all_paires.extend(
                    _collect_paires(
                        question,
                        evidence,
                        assertion_index=gold["assertion_index"],
                        claims_index=claims_index,
                        cache=cache,
                    )
                )

        raw50 = search_fn(question["question"], 50)
        fused50 = list(raw50.get("evidence") or [])
        entry["classement_k50"] = [
            {"rang": i, "id": item.get("id"), "kind": item.get("kind"), "score": item.get("score")}
            for i, item in enumerate(fused50, 1)
        ]
        if question_text not in embed_cache:
            raise RuntimeError(f"cache froid : embedding absent pour la question ({question_text[:60]}…)")
        question_vector = embed_cache[question_text]
        entry["attribution_b5"] = collect_missing_diagnostics(
            question,
            evidence_k20,
            fused50,
            gold=gold,
            holder_table=holder_table,
            claims=claims,
            claims_index=claims_index,
            connection=connection,
            question_vector=question_vector,
        )

        recent = tracker.records[-4:]
        unhealthy = [row for row in recent if not row["healthy"]]
        if unhealthy:
            question_degraded = True
            degrade_reasons = [row["reason"] or "search degrade" for row in unhealthy]

        entry["degradee"] = question_degraded
        entry["search_sain"] = not question_degraded
        if question_degraded:
            entry["degrade_raisons"] = degrade_reasons
            degraded_questions.append({"id": qid, "raisons": degrade_reasons})

        _append_journal_entry(journal_path, entry)
        entries.append(entry)

    search_health = summarize_search_health(tracker.records)
    paires_dedup = _dedupe_paires(all_paires)

    if degraded_questions or search_health["run_search_degrades"] > 0:
        payload = {
            "status": "arret",
            "a_quelle_etape": "B_mesure",
            "pourquoi": "appels run_search degrades",
            "label": label,
            "questions_degradees": degraded_questions,
            "search_health": search_health,
            "holder_table_seconds": round(holder_seconds, 3),
            "tient_sans_regle_piece": tient_without_piece,
        }
        measure_status_path(label).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload

    result = {
        "status": "complete",
        "label": label,
        "parity": parity,
        "holder_table_seconds": round(holder_seconds, 3),
        "tient_sans_regle_piece": tient_without_piece,
        "search_health": search_health,
        "questions": entries,
        "paires_a_lire": paires_dedup,
        "aggregates": aggregate_entries(entries),
    }
    measure_status_path(label).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _evidence_from_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    ids = entry.get("k20", {}).get("evidence_ids") or []
    return [{"id": item_id, "kind": "claim_hit"} for item_id in ids]


def _dedupe_paires(paires: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in paires:
        key = (row["assertion_id"], row["claim_id"])
        if key not in merged:
            merged[key] = {
                "assertion": row["assertion"],
                "fait": row["fait"],
                "overlap": row["overlap"],
                "questions": [row["question_id"]],
            }
        elif row["question_id"] not in merged[key]["questions"]:
            merged[key]["questions"].append(row["question_id"])
    out = list(merged.values())
    out.sort(key=lambda r: (r["assertion"]["id"], -r["overlap"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure étage 2 restitution")
    parser.add_argument("--variante", choices=["standard", "quotas-doux"], default="standard")
    parser.add_argument("--prechauffer", action="store_true", help="Warm question embeddings slowly")
    args = parser.parse_args()
    export_runtime_env()
    measure_output_dir().mkdir(parents=True, exist_ok=True)

    if args.prechauffer:
        payload = prechauffer_questions()
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    from .infra import connect_existing

    parity_path = measure_output_dir() / f"_parity-{args.variante}.json"
    parity = None
    if parity_path.is_file():
        parity = json.loads(parity_path.read_text(encoding="utf-8"))

    caps = None
    if args.variante == "quotas-doux":
        caps = {"file_hit": 0.375, "asset_hit": 0.25}
    result = measure_all(
        connect_existing(),
        population_caps=caps,
        label=args.variante,
        parity=parity,
    )
    print(json.dumps({"status": result.get("status"), "label": args.variante}, ensure_ascii=False))
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
