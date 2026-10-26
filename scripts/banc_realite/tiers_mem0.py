# SPDX-License-Identifier: AGPL-3.0-only
"""Ingestion et contextes mem0 pour le banc comparatif (venv tiers dédié)."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from banc_realite.compat_moteur import EMBED_DIM

from .config import BANC_ROOT, export_runtime_env
from .dossiers import BANC_VAR, Dossier, dossier_courant

COMPARATIF_DIR = BANC_ROOT / "sorties" / "comparatif"

# Tarifs Mistral — vérifier https://docs.mistral.ai/deployment/laplateforme/pricing/ (2026-09-23)
PRIX_MISTRAL_LARGE_INPUT_EUR_MTOK = 2.0
PRIX_MISTRAL_LARGE_OUTPUT_EUR_MTOK = 6.0
PRIX_MISTRAL_EMBED_EUR_MTOK = 0.1
TOKENS_PAR_CARACTERE = 0.25
MAX_MORCEAU_CHARS = 6000
PG_PORT = 56102
PG_DSN = f"postgresql://postgres@127.0.0.1:{PG_PORT}/postgres"
DB_NAME = "tiers_mem0"
MODELES_PREFERES = ("mistral-large-latest", "mistral-medium-latest")
TENTATIVES_429 = 6
PAUSE_ENTRE_APPELS_S = 1.5
MOREAU_MAX_EXTRACTION = 80
MOREAU_MAX_EMBEDDINGS = 1000
MAX_EMBEDDINGS_CONTEXTES = 300
MISTRAL_BASE = "https://api.mistral.ai/v1"
CURSOR_RELAY_PORT = 56110
CURSOR_RELAY_BASE = f"http://127.0.0.1:{CURSOR_RELAY_PORT}/v1"
CURSOR_MODEL = "cursor-grok-4.5-high"
EXTRACTEURS = ("mistral", "cursor")


def _resolve_dossier(dossier: Dossier | str | None) -> Dossier:
    if isinstance(dossier, Dossier):
        return dossier
    return dossier_courant(dossier)


@dataclass(frozen=True, slots=True)
class Mem0Chemins:
    """Artefacts mem0 — GreenLoop conserve sorties/comparatif, MOREAU sous dossiers/<nom>/mem0/."""

    dossier: Dossier

    @classmethod
    def pour(cls, dossier: Dossier | str | None = None) -> Mem0Chemins:
        return cls(_resolve_dossier(dossier))

    @property
    def racine(self) -> Path:
        if self.dossier.nom == "greenloop":
            return COMPARATIF_DIR
        return BANC_VAR / "dossiers" / self.dossier.nom / "mem0"

    def journal_ingestion(self, extracteur: str) -> Path:
        if self.dossier.nom == "greenloop":
            return self.racine / f"appels-ingestion-mem0-{extracteur}.jsonl"
        return self.racine / "appels-ingestion-mem0.jsonl"

    def sortie_a_blanc(self, extracteur: str) -> Path:
        if self.dossier.nom == "greenloop":
            return self.racine / f"ingestion-mem0-{extracteur}.a-blanc.json"
        return self.racine / "ingestion-mem0.a-blanc.json"

    @property
    def contextes_path(self) -> Path:
        return self.racine / "contextes-mem0.jsonl"

    def contextes_path_for_k(self, k: int) -> Path:
        if k == 20:
            return self.contextes_path
        return self.racine / f"contextes-mem0-k{k}.jsonl"

    def journal_contextes_for_k(self, k: int) -> Path:
        if k == 20:
            return self.racine / "appels-contextes-mem0.jsonl"
        return self.racine / f"appels-contextes-mem0-k{k}.jsonl"

    @property
    def journal_contextes(self) -> Path:
        return self.journal_contextes_for_k(20)

    @property
    def rapport(self) -> Path:
        return self.racine / "RAPPORT-INGESTION-MEM0.md"


def comparatif_context_path(
    etiquette: str,
    dossier: Dossier | str | None = None,
    *,
    k: int = 20,
) -> Path:
    if etiquette == "mem0":
        fichier_k = 20 if k <= 20 else k
        return Mem0Chemins.pour(dossier).contextes_path_for_k(fichier_k)
    return COMPARATIF_DIR / f"contextes-{etiquette}.jsonl"


def _plafonds(dossier: Dossier) -> dict[str, int] | None:
    if dossier.nom == "moreau":
        return {"extraction": MOREAU_MAX_EXTRACTION, "embeddings": MOREAU_MAX_EMBEDDINGS}
    return None


def _collection_name(dossier: Dossier, extracteur: str, model: str) -> str:
    return f"mem0_{dossier.nom}_{extracteur}_{_slug(model)}"


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _load_questions(dossier: Dossier | str | None = None) -> list[dict[str, Any]]:
    from .measure_etage2 import load_questions

    return load_questions(dossier)


def lister_pieces(dossier: Dossier | str | None = None) -> list[Path]:
    current = _resolve_dossier(dossier)
    return sorted(path for path in current.corpus_dir.iterdir() if path.is_file())


def decouper_piece(text: str, *, max_chars: int = MAX_MORCEAU_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= max_chars:
            current = para
        else:
            for start in range(0, len(para), max_chars):
                chunks.append(para[start : start + max_chars])
            current = ""
    if current:
        chunks.append(current)
    return chunks


def estimer_a_blanc(
    *,
    extracteur: str = "mistral",
    dossier: Dossier | str | None = None,
) -> dict[str, Any]:
    current = _resolve_dossier(dossier)
    chemins = Mem0Chemins.pour(current)
    pieces = lister_pieces(current)
    total_chars = 0
    morceaux = 0
    detail: list[dict[str, Any]] = []
    for piece in pieces:
        text = piece.read_text(encoding="utf-8")
        parts = decouper_piece(text)
        total_chars += len(text)
        morceaux += len(parts)
        detail.append({"piece": piece.name, "caracteres": len(text), "morceaux": len(parts)})
    tokens_estimes = round(total_chars * TOKENS_PAR_CARACTERE)
    tokens_ingestion = morceaux * 2000
    tokens_contextes = len(_load_questions(current)) * 20 * 50
    cout_ingestion = (
        tokens_ingestion / 1_000_000 * PRIX_MISTRAL_LARGE_INPUT_EUR_MTOK
        + tokens_ingestion / 1_000_000 * PRIX_MISTRAL_LARGE_OUTPUT_EUR_MTOK * 0.2
    )
    cout_contextes = tokens_contextes / 1_000_000 * PRIX_MISTRAL_EMBED_EUR_MTOK
    payload = {
        "date": time.strftime("%Y-%m-%d"),
        "dossier": current.nom,
        "extracteur": extracteur,
        "pieces": len(pieces),
        "caracteres": total_chars,
        "tokens_estimes_corpus": tokens_estimes,
        "appels_prevus": morceaux,
        "appels_contextes_prevus": len(_load_questions(current)),
        "cout_contextes_eur": round(cout_contextes, 4),
        "detail_pieces": detail,
        "note_requete": "Le texte de requête mem0 est la question telle quelle, pas la requête structurée CallFrame.",
    }
    if extracteur == "cursor":
        payload["cout_ingestion_eur"] = None
        payload["cout_total_eur"] = None
        payload["cout"] = f"quota Cursor, {morceaux} appels prévus"
        payload["modele_extraction"] = CURSOR_MODEL
    else:
        payload["cout_ingestion_eur"] = round(cout_ingestion, 4)
        payload["cout_total_eur"] = round(cout_ingestion + cout_contextes, 4)
    chemins.racine.mkdir(parents=True, exist_ok=True)
    chemins.sortie_a_blanc(extracteur).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _journaliser(chemin: Path, row: dict[str, Any]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_results(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        rows = payload.get("results", [])
        return rows if isinstance(rows, list) else []
    if isinstance(payload, list):
        return payload
    return []


def _morceaux_journalises(journal: Path, modele: str | None = None) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not journal.is_file():
        return done
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("bilan") or row.get("erreur"):
            continue
        if modele is not None and row.get("modele") not in (None, modele):
            continue
        piece = row.get("piece")
        morceau = row.get("morceau")
        if piece is not None and morceau is not None:
            done.add((str(piece), int(morceau)))
    return done


def _pieces_avec_bilan(journal: Path) -> set[str]:
    done: set[str] = set()
    if not journal.is_file():
        return done
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("bilan") and row.get("piece"):
            done.add(str(row["piece"]))
    return done


def _compter_contextes_journal(journal_contextes: Path) -> int:
    appels = 0
    if not journal_contextes.is_file():
        return appels
    for line in journal_contextes.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("question_id"):
            appels += 1
    return appels


def _compter_journal(
    journal_ingestion: Path,
    journal_contextes: Path,
) -> dict[str, int]:
    extraction = 0
    embeddings = 0
    echecs = 0
    if journal_ingestion.is_file():
        for line in journal_ingestion.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("erreur"):
                echecs += 1
                continue
            if row.get("bilan"):
                continue
            if row.get("morceau") is not None:
                extraction += 1
                embeddings += int(row.get("souvenirs") or 0)
    if journal_contextes.is_file():
        for line in journal_contextes.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("question_id"):
                embeddings += 1
    return {"appels_extraction": extraction, "embeddings": embeddings, "echecs": echecs}


def _choisir_modele_mistral(api_key: str) -> str:
    request = urllib.request.Request(
        f"{MISTRAL_BASE}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GET /v1/models a échoué (HTTP {exc.code})") from exc
    ids = [row.get("id") for row in payload.get("data") or [] if row.get("id")]
    for candidate in MODELES_PREFERES:
        if candidate in ids:
            return candidate
    raise RuntimeError(
        f"aucun des modèles préférés {MODELES_PREFERES} n'est disponible via GET /v1/models"
    )


def _avec_reprise_429(appel, etiquette: str) -> Any:
    last_exc: Exception | None = None
    for tentative in range(TENTATIVES_429):
        try:
            result = appel()
            time.sleep(PAUSE_ENTRE_APPELS_S)
            return result
        except Exception as exc:
            message = str(exc)
            if "429" not in message and "rate_limit" not in message.lower():
                raise
            last_exc = exc
            time.sleep(2 ** (tentative + 1))
    raise RuntimeError(f"429 persistant sur {etiquette} après {TENTATIVES_429} tentatives") from last_exc


def _add_avec_reprise(memory, chunk: str, piece_name: str, user_id: str) -> dict[str, Any]:
    return _avec_reprise_429(
        lambda: memory.add(chunk, user_id=user_id, metadata={"piece": piece_name}),
        piece_name,
    )


def _ensure_database(connection) -> None:
    exists = connection.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,)).fetchone()
    if not exists:
        connection.execute(f'CREATE DATABASE "{DB_NAME}"')
    import psycopg

    db = psycopg.connect(f"postgresql://postgres@127.0.0.1:{PG_PORT}/{DB_NAME}", autocommit=True)
    try:
        db.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        db.close()


def _build_memory_config(
    api_key: str,
    model: str,
    *,
    extracteur: str = "mistral",
    dossier: Dossier | str | None = None,
) -> dict[str, Any]:
    current = _resolve_dossier(dossier)
    if extracteur == "cursor":
        llm_config = {
            "model": CURSOR_MODEL,
            "api_key": "relais",
            "openai_base_url": CURSOR_RELAY_BASE,
        }
    else:
        llm_config = {
            "model": model,
            "api_key": api_key,
            "openai_base_url": MISTRAL_BASE,
        }
    return {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "dbname": DB_NAME,
                "user": "postgres",
                "password": "",
                "host": "127.0.0.1",
                "port": PG_PORT,
                "collection_name": _collection_name(current, extracteur, model),
                "embedding_model_dims": EMBED_DIM,
            },
        },
        "llm": {
            "provider": "openai",
            "config": llm_config,
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "mistral-embed",
                "api_key": api_key,
                "openai_base_url": MISTRAL_BASE,
            },
        },
    }


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[rank]


def stats_longueur_souvenirs(
    dossier: Dossier | str,
    *,
    extracteur: str,
    model: str,
) -> dict[str, int]:
    current = _resolve_dossier(dossier)
    table = _collection_name(current, extracteur, model)
    import psycopg

    lengths: list[int] = []
    try:
        conn = psycopg.connect(f"postgresql://postgres@127.0.0.1:{PG_PORT}/{DB_NAME}")
        rows = conn.execute(
            f"SELECT length(payload->>'data') AS n FROM \"{table}\" "
            f"WHERE payload->>'user_id' = %s AND payload->>'data' IS NOT NULL",
            (current.nom,),
        ).fetchall()
        conn.close()
        lengths = [int(row[0]) for row in rows if row[0] is not None]
    except Exception:
        lengths = []
    if not lengths:
        return {"med": 0, "p90": 0, "max": 0}
    return {
        "med": int(statistics.median(lengths)),
        "p90": _percentile(lengths, 90),
        "max": max(lengths),
    }


def verifier_isolation_mem0(
    *,
    extracteur: str = "cursor",
    model: str = CURSOR_MODEL,
) -> dict[str, Any]:
    import psycopg

    greenloop = dossier_courant("greenloop")
    moreau = dossier_courant("moreau")
    tables = {
        "greenloop": _collection_name(greenloop, extracteur, model),
        "moreau": _collection_name(moreau, extracteur, model),
    }
    counts: dict[str, int] = {}
    cross: dict[str, int] = {}
    conn = psycopg.connect(f"postgresql://postgres@127.0.0.1:{PG_PORT}/{DB_NAME}")
    try:
        for label, table in tables.items():
            try:
                counts[label] = conn.execute(
                    f"SELECT COUNT(*) FROM \"{table}\" WHERE payload->>'user_id' = %s",
                    (label,),
                ).fetchone()[0]
            except Exception:
                counts[label] = 0
            other = "moreau" if label == "greenloop" else "greenloop"
            try:
                cross[f"{label}_contient_{other}"] = conn.execute(
                    f"SELECT COUNT(*) FROM \"{table}\" WHERE payload->>'user_id' = %s",
                    (other,),
                ).fetchone()[0]
            except Exception:
                cross[f"{label}_contient_{other}"] = 0
    finally:
        conn.close()
    isolation_ok = all(value == 0 for value in cross.values())
    return {
        "isolation_ok": isolation_ok,
        "collections": tables,
        "counts": counts,
        "cross_user_id": cross,
    }


def compter_souvenirs_par_piece(
    dossier: Dossier | str,
    *,
    extracteur: str,
    model: str,
) -> dict[str, int]:
    current = _resolve_dossier(dossier)
    table = _collection_name(current, extracteur, model)
    import psycopg

    out: dict[str, int] = {}
    conn = psycopg.connect(f"postgresql://postgres@127.0.0.1:{PG_PORT}/{DB_NAME}")
    try:
        rows = conn.execute(
            f"SELECT payload->>'piece' AS piece, COUNT(*) AS n FROM \"{table}\" "
            f"WHERE payload->>'user_id' = %s GROUP BY 1",
            (current.nom,),
        ).fetchall()
        for piece, count in rows:
            if piece:
                out[str(piece)] = int(count)
    except Exception:
        pass
    finally:
        conn.close()
    return out


def ingerer(
    *,
    executer: bool,
    plafond_eur: float,
    extracteur: str = "mistral",
    dossier: Dossier | str | None = None,
) -> dict[str, Any]:
    if extracteur not in EXTRACTEURS:
        raise ValueError(f"extracteur inconnu: {extracteur}")
    current = _resolve_dossier(dossier)
    chemins = Mem0Chemins.pour(current)
    plafonds = _plafonds(current)
    estimation = estimer_a_blanc(extracteur=extracteur, dossier=current)
    if extracteur == "mistral" and estimation["cout_ingestion_eur"] > plafond_eur:
        raise RuntimeError(
            f"estimation {estimation['cout_ingestion_eur']} € > plafond {plafond_eur} € — ingestion refusée"
        )
    if not executer:
        return {"status": "refuse", "pourquoi": "--executer absent", "estimation": estimation}

    from banc_realite.compat_moteur import load_api_key

    import psycopg

    api_key = load_api_key()
    model = CURSOR_MODEL if extracteur == "cursor" else _choisir_modele_mistral(api_key)
    admin = psycopg.connect(PG_DSN, autocommit=True)
    try:
        _ensure_database(admin)
    finally:
        admin.close()

    from mem0 import Memory

    memory = Memory.from_config(
        _build_memory_config(api_key, model, extracteur=extracteur, dossier=current)
    )
    journal_ingestion = chemins.journal_ingestion(extracteur)
    done_morceaux = _morceaux_journalises(journal_ingestion, model)
    bilans = _pieces_avec_bilan(journal_ingestion)
    ingerees = 0
    started = time.monotonic()
    user_id = current.nom

    for piece in lister_pieces(current):
        text = piece.read_text(encoding="utf-8")
        chunks = decouper_piece(text)
        piece_started = time.monotonic()
        for idx, chunk in enumerate(chunks):
            key = (piece.name, idx)
            if key in done_morceaux:
                continue
            comptes = _compter_journal(journal_ingestion, chemins.journal_contextes)
            if plafonds and comptes["appels_extraction"] >= plafonds["extraction"]:
                return {
                    "status": "arret",
                    "cause": "plafond_appels_extraction",
                    "dossier": current.nom,
                    "pieces_ingerees": ingerees,
                    "modele": model,
                    "extracteur": extracteur,
                    **comptes,
                    "duree_s": round(time.monotonic() - started, 3),
                }
            if plafonds and comptes["embeddings"] >= plafonds["embeddings"]:
                return {
                    "status": "arret",
                    "cause": "plafond_embeddings",
                    "dossier": current.nom,
                    "pieces_ingerees": ingerees,
                    "modele": model,
                    "extracteur": extracteur,
                    **comptes,
                    "duree_s": round(time.monotonic() - started, 3),
                }
            chunk_started = time.monotonic()
            try:
                result = _add_avec_reprise(memory, chunk, piece.name, user_id)
            except Exception as exc:
                _journaliser(
                    journal_ingestion,
                    {
                        "fournisseur": "mem0",
                        "dossier": current.nom,
                        "extracteur": extracteur,
                        "modele": model,
                        "piece": piece.name,
                        "morceau": idx,
                        "morceaux": len(chunks),
                        "erreur": str(exc),
                        "secondes": round(time.monotonic() - chunk_started, 3),
                    },
                )
                continue
            chunk_souvenirs = len(_extract_results(result))
            usage = result.get("usage") or {} if isinstance(result, dict) else {}
            _journaliser(
                journal_ingestion,
                {
                    "fournisseur": "mem0",
                    "dossier": current.nom,
                    "extracteur": extracteur,
                    "modele": model,
                    "piece": piece.name,
                    "morceau": idx,
                    "morceaux": len(chunks),
                    "souvenirs": chunk_souvenirs,
                    "usage": {
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "total_tokens": int(usage.get("total_tokens") or 0),
                    },
                    "secondes": round(time.monotonic() - chunk_started, 3),
                },
            )
            done_morceaux.add(key)
        if piece.name in bilans:
            continue
        if not all((piece.name, idx) in done_morceaux for idx in range(len(chunks))):
            continue
        souvenirs = 0
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for line in journal_ingestion.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("piece") == piece.name
                and not row.get("bilan")
                and row.get("morceau") is not None
                and not row.get("erreur")
            ):
                souvenirs += int(row.get("souvenirs") or 0)
                chunk_usage = row.get("usage") or {}
                for usage_key in usage_total:
                    usage_total[usage_key] += int(chunk_usage.get(usage_key) or 0)
        _journaliser(
            journal_ingestion,
            {
                "fournisseur": "mem0",
                "dossier": current.nom,
                "extracteur": extracteur,
                "modele": model,
                "piece": piece.name,
                "bilan": True,
                "morceaux": len(chunks),
                "souvenirs": souvenirs,
                "usage": usage_total,
                "secondes": round(time.monotonic() - piece_started, 3),
            },
        )
        bilans.add(piece.name)
        ingerees += 1
    comptes = _compter_journal(journal_ingestion, chemins.journal_contextes)
    par_piece = compter_souvenirs_par_piece(current, extracteur=extracteur, model=model)
    return {
        "status": "complete",
        "dossier": current.nom,
        "pieces_ingerees": ingerees,
        "modele": model,
        "extracteur": extracteur,
        "souvenirs_par_piece": par_piece,
        "souvenirs": sum(par_piece.values()),
        **comptes,
        "duree_s": round(time.monotonic() - started, 3),
    }


def contextes(
    *,
    k: int,
    extracteur: str = "mistral",
    dossier: Dossier | str | None = None,
) -> dict[str, Any]:
    if extracteur not in EXTRACTEURS:
        raise ValueError(f"extracteur inconnu: {extracteur}")
    current = _resolve_dossier(dossier)
    chemins = Mem0Chemins.pour(current)
    plafonds = _plafonds(current)
    fichier_k = 20 if k <= 20 else k
    out_path = chemins.contextes_path_for_k(fichier_k)
    journal = chemins.journal_contextes_for_k(fichier_k)
    chemins.racine.mkdir(parents=True, exist_ok=True)
    if fichier_k == 20:
        _verifier_extracteur_contextes(out_path, extracteur)

    from banc_realite.compat_moteur import load_api_key

    from mem0 import Memory

    api_key = load_api_key()
    model = CURSOR_MODEL if extracteur == "cursor" else _choisir_modele_mistral(api_key)
    memory = Memory.from_config(
        _build_memory_config(api_key, model, extracteur=extracteur, dossier=current)
    )
    user_id = current.nom
    appels = 0
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "meta": {
                        "k": fichier_k,
                        "extracteur": extracteur,
                        "modele": model,
                        "dossier": current.nom,
                    }
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for question in _load_questions(current):
            comptes = _compter_journal(
                chemins.journal_ingestion(extracteur),
                journal,
            )
            if _compter_contextes_journal(journal) >= MAX_EMBEDDINGS_CONTEXTES:
                return {
                    "status": "arret",
                    "cause": "plafond_embeddings_contextes",
                    "questions": appels,
                    "sortie": str(out_path),
                    "extracteur": extracteur,
                    "modele_llm": model,
                    "dossier": current.nom,
                    "embeddings_contextes": _compter_contextes_journal(journal),
                    **comptes,
                }
            if plafonds and comptes["embeddings"] >= plafonds["embeddings"]:
                return {
                    "status": "arret",
                    "cause": "plafond_embeddings",
                    "questions": appels,
                    "sortie": str(out_path),
                    "extracteur": extracteur,
                    "modele_llm": model,
                    "dossier": current.nom,
                    **comptes,
                }
            started = time.monotonic()
            hits = _avec_reprise_429(
                # mem0 2.1 : `user_id` et `limit` ne sont plus des paramètres directs de
                # search() — filtres explicites et `top_k` (ValueError sinon).
                lambda: memory.search(
                    question["question"], filters={"user_id": user_id}, top_k=k
                ),
                question["id"],
            )
            rows = _extract_results(hits)
            elements = []
            for rank, hit in enumerate(rows, 1):
                metadata = hit.get("metadata") or {}
                elements.append(
                    {
                        "rank": rank,
                        "score": float(hit.get("score") or 0.0),
                        "text": hit.get("memory") or hit.get("text") or "",
                        "source": metadata.get("piece") or "",
                        "provenance": metadata,
                    }
                )
            handle.write(
                json.dumps({"question_id": question["id"], "elements": elements}, ensure_ascii=False) + "\n"
            )
            appels += 1
            _journaliser(
                journal,
                {
                    "fournisseur": "mem0",
                    "dossier": current.nom,
                    "extracteur": extracteur,
                    "modele": "mistral-embed",
                    "question_id": question["id"],
                    "k": k,
                    "secondes": round(time.monotonic() - started, 3),
                },
            )
    return {
        "status": "complete",
        "questions": appels,
        "sortie": str(out_path),
        "extracteur": extracteur,
        "modele_llm": model,
        "dossier": current.nom,
    }


def _lire_meta_contextes(chemin: Path) -> dict[str, Any] | None:
    if not chemin.is_file():
        return None
    for line in chemin.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "meta" in row:
            return dict(row["meta"])
    return None


def _verifier_extracteur_contextes(chemin: Path, extracteur: str) -> None:
    meta = _lire_meta_contextes(chemin)
    if meta is None:
        return
    existant = meta.get("extracteur")
    if existant and existant != extracteur:
        raise RuntimeError(
            f"contextes mem0 déjà produits par extracteur « {existant} » "
            f"({chemin}) — refus d'écraser avec « {extracteur} »"
        )


def write_rapport_ingestion_mem0(
    *,
    dossier: Dossier | str,
    extracteur: str,
    model: str,
    ingestion: dict[str, Any],
    contextes_result: dict[str, Any],
    isolation: dict[str, Any],
    longueurs: dict[str, int],
) -> Path:
    current = _resolve_dossier(dossier)
    chemins = Mem0Chemins.pour(current)
    par_piece = ingestion.get("souvenirs_par_piece") or compter_souvenirs_par_piece(
        current, extracteur=extracteur, model=model
    )
    lines = [
        "# Rapport — ingestion mem0 MOREAU",
        "",
        f"Date : {time.strftime('%Y-%m-%d')}",
        f"Dossier : {current.nom}",
        f"Extracteur : {extracteur} ({model})",
        "",
        "## Ingestion",
        f"- Pièces ingérées : {ingestion.get('pieces_ingerees', 0)}/12",
        f"- Morceaux traités : {ingestion.get('appels_extraction', 0)}",
        f"- Souvenirs créés : {ingestion.get('souvenirs', sum(par_piece.values()))}",
        f"- Appels extraction : {ingestion.get('appels_extraction', 0)} (plafond {MOREAU_MAX_EXTRACTION})",
        f"- Embeddings : {ingestion.get('embeddings', 0)} (plafond {MOREAU_MAX_EMBEDDINGS})",
        f"- Durée : {ingestion.get('duree_s', 0)} s",
        f"- Échecs journalisés : {ingestion.get('echecs', 0)}",
        "",
        "### Souvenirs par pièce",
        "",
        "| pièce | souvenirs |",
        "| --- | ---: |",
    ]
    for piece, count in sorted(par_piece.items()):
        lines.append(f"| {piece} | {count} |")
    lines.extend(
        [
            "",
            "## Longueur des souvenirs (caractères)",
            "",
            f"- Médiane : {longueurs.get('med', 0)}",
            f"- p90 : {longueurs.get('p90', 0)}",
            f"- max : {longueurs.get('max', 0)}",
            "",
            "## Contextes",
            "",
            f"- Questions : {contextes_result.get('questions', 0)}",
            "- k : 20",
            f"- Fichier : {contextes_result.get('sortie', chemins.contextes_path)}",
            "",
            "## Isolation GreenLoop / MOREAU",
            "",
            f"- OK : {isolation.get('isolation_ok')}",
            f"- Collections : {json.dumps(isolation.get('collections', {}), ensure_ascii=False)}",
            f"- Comptes : {json.dumps(isolation.get('counts', {}), ensure_ascii=False)}",
            f"- Fuites user_id : {json.dumps(isolation.get('cross_user_id', {}), ensure_ascii=False)}",
        ]
    )
    chemins.racine.mkdir(parents=True, exist_ok=True)
    chemins.rapport.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return chemins.rapport


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiers mem0 — banc comparatif")
    sub = parser.add_subparsers(dest="commande", required=True)
    ab = sub.add_parser("a-blanc", help="Estimation sans appel réseau")
    ab.add_argument("--extracteur", choices=EXTRACTEURS, default="mistral")
    ab.add_argument("--dossier", default="greenloop", help="Nom du dossier (greenloop, moreau)")
    ing = sub.add_parser("ingerer", help="Ingestion mem0 sur le cluster du banc")
    ing.add_argument("--executer", action="store_true")
    ing.add_argument("--plafond-eur", type=float, required=True)
    ing.add_argument("--extracteur", choices=EXTRACTEURS, default="mistral")
    ing.add_argument("--dossier", default="greenloop", help="Nom du dossier (greenloop, moreau)")
    ctx = sub.add_parser("contextes", help="Extraire les contextes question par question")
    ctx.add_argument("--k", type=int, default=20)
    ctx.add_argument("--extracteur", choices=EXTRACTEURS, default="mistral")
    ctx.add_argument("--dossier", default="greenloop", help="Nom du dossier (greenloop, moreau)")
    args = parser.parse_args()
    export_runtime_env()
    dossier = args.dossier
    if args.commande == "a-blanc":
        print(json.dumps(estimer_a_blanc(extracteur=args.extracteur, dossier=dossier), ensure_ascii=False))
        return 0
    if args.commande == "ingerer":
        print(
            json.dumps(
                ingerer(
                    executer=args.executer,
                    plafond_eur=args.plafond_eur,
                    extracteur=args.extracteur,
                    dossier=dossier,
                ),
                ensure_ascii=False,
            )
        )
        return 0
    print(json.dumps(contextes(k=args.k, extracteur=args.extracteur, dossier=dossier), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
