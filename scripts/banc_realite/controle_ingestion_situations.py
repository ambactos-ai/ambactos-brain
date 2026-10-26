#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Contrôles d'ingestion des situations MRFS — lecture seule, l'or ne sert qu'ici."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "scripts", ROOT / "src"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from banc_realite.constructeur_situations import (  # noqa: E402
    ALLOWED_LECTEUR_DOSSIERS,
    controle_dossier,
    extract_piece_lines,
    latest_constructeur_snapshot_id,
)
from banc_realite.dossiers import Dossier, dossier_courant  # noqa: E402
from banc_realite.extraction_situations import (  # noqa: E402
    DEFAULT_CONSTRUCTEUR_ROOT,
    charger_etat,
    constructeur_dossier_root,
    faits_dir,
)
from banc_realite.embeddings_faits import (  # noqa: E402
    cache_embeddings_faits_path,
    charger_cache,
)
from banc_realite.fournisseurs_contexte import (  # noqa: E402
    BrainMrfs,
    BrainMrfsLexical,
    rendre_contexte,
)
from banc_realite.lecteur_faits_classes import charger_faits, empreinte_texte  # noqa: E402
from banc_realite.holders import assertion_present_in_evidence  # noqa: E402
from banc_realite.mrfs_support import BANC_PG_DSN, MAX_BUDGET_CHARS, MRFS_SCHEMA, load_subject_registry  # noqa: E402
from banc_realite.normalize import piece_basename  # noqa: E402
from banc_realite.or_dossier import load_gold, load_questions  # noqa: E402

DENOMINATEURS_OR = {"greenloop": 2403, "moreau": 42}
FR_MARQUEURS = frozenset(
    {
        "le",
        "la",
        "les",
        "de",
        "des",
        "un",
        "une",
        "et",
        "est",
        "dans",
        "pour",
        "que",
        "qui",
        "pas",
        "sur",
        "du",
        "au",
        "aux",
        "en",
        "par",
        "avec",
        "son",
        "sa",
        "ses",
    }
)
EN_MARQUEURS = frozenset(
    {
        "the",
        "and",
        "of",
        "to",
        "in",
        "for",
        "is",
        "that",
        "with",
        "on",
        "as",
        "by",
        "from",
        "this",
        "was",
        "are",
        "be",
        "or",
        "an",
        "at",
    }
)


def _pg_ready(dsn: str) -> bool:
    parsed = urlparse(dsn)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5432
    result = subprocess.run(
        ["pg_isready", "-h", host, "-p", str(port)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _mots(text: str) -> list[str]:
    return [token.lower() for token in text.replace("'", " ").split() if token]


def detecter_langue(textes: list[str]) -> str:
    fr = 0
    en = 0
    for texte in textes:
        for token in _mots(texte):
            if token in FR_MARQUEURS:
                fr += 1
            if token in EN_MARQUEURS:
                en += 1
    if fr == 0 and en == 0:
        return "indeterminee"
    if fr > en:
        return "fr"
    if en > fr:
        return "en"
    return "mixte"


def _agreger_etat_piece(piece_state: dict[str, Any]) -> dict[str, int]:
    rendus = valides = rejetes = doublons = recales = 0
    morceaux = piece_state.get("morceaux") or {}
    for raw in morceaux.values():
        if not isinstance(raw, dict):
            continue
        rendus += int(raw.get("faits_rendus") or 0)
        valides += int(raw.get("faits_valides") or 0)
        rejetes += int(raw.get("faits_rejetes") or 0)
        doublons += int(raw.get("faits_doublons") or 0)
        recales += int(raw.get("lettres_recalees") or 0)
    return {
        "faits_rendus": rendus,
        "faits_valides": valides,
        "faits_rejetes": rejetes,
        "faits_doublons": doublons,
        "faits_recales": recales,
    }


def _charger_faits(dossier: str, *, constructeur_root: Path) -> dict[str, list[dict[str, Any]]]:
    directory = faits_dir(dossier, constructeur_root=constructeur_root)
    by_piece: dict[str, list[dict[str, Any]]] = {}
    if not directory.is_dir():
        return by_piece
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        piece = str(payload.get("piece") or "")
        faits = payload.get("faits")
        if not piece or not isinstance(faits, list):
            continue
        by_piece.setdefault(piece, []).extend(
            item for item in faits if isinstance(item, dict)
        )
    return by_piece


def _couverture_lignes(piece_text: str, faits: list[dict[str, Any]]) -> dict[str, Any]:
    lines = extract_piece_lines(piece_text)
    non_vides = {index for index, line in enumerate(lines, start=1) if line.strip()}
    couvertes: set[int] = set()
    longueurs: list[int] = []
    for fait in faits:
        citation = str(fait.get("citation") or "")
        if citation:
            longueurs.append(len(citation))
        start = fait.get("ligne_debut")
        end = fait.get("ligne_fin")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        for numero in range(start, end + 1):
            if numero in non_vides:
                couvertes.add(numero)
    total = len(non_vides)
    return {
        "lignes_non_vides": total,
        "lignes_couvertes": len(couvertes),
        "part_lignes": round(len(couvertes) / total, 4) if total else 0.0,
        "longueur_mediane_citations": (
            statistics.median(longueurs) if longueurs else 0
        ),
    }


def _faits_paths(dossier: str, *, constructeur_root: Path) -> list[Path]:
    directory = faits_dir(dossier, constructeur_root=constructeur_root)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def compter_instantane(
    subject_id: str,
    *,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> dict[str, Any]:
    import psycopg

    snapshot_id = latest_constructeur_snapshot_id(subject_id, dsn=dsn, schema=schema)
    if snapshot_id is None:
        return {
            "snapshot_id": None,
            "par_nature": {},
            "hors_constructeur": 0,
            "ingest": 0,
        }
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        rows = connection.execute(
            f"SELECT hit_kind, count(*) FROM {schema}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s GROUP BY hit_kind",
            (subject_id, snapshot_id),
        ).fetchall()
        hors = connection.execute(
            f"SELECT count(*) FROM {schema}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s "
            "AND record_key NOT LIKE 'constructeur:%%'",
            (subject_id, snapshot_id),
        ).fetchone()[0]
        ingest = connection.execute(
            f"SELECT count(*) FROM {schema}.projection_records "
            "WHERE subject_id = %s AND snapshot_id = %s "
            "AND record_key LIKE 'ingest:%%'",
            (subject_id, snapshot_id),
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "snapshot_id": snapshot_id,
        "par_nature": {str(kind): int(count) for kind, count in rows},
        "hors_constructeur": int(hors),
        "ingest": int(ingest),
    }


def isolation_constructeur(
    dossier: str,
    *,
    dsn: str = BANC_PG_DSN,
    schema: str = MRFS_SCHEMA,
) -> dict[str, Any]:
    import psycopg

    registry = load_subject_registry()
    subject_id = registry.get(dossier)
    autres = {
        nom: sid
        for nom, sid in registry.items()
        if nom != dossier and nom in ALLOWED_LECTEUR_DOSSIERS
    }
    pieces = set()
    desc = dossier_courant(dossier) if dossier in {"greenloop", "moreau"} else None
    if desc is not None and desc.corpus_dir.is_dir():
        pieces = {path.name for path in desc.corpus_dir.iterdir() if path.is_file()}
    connection = psycopg.connect(dsn, autocommit=True)
    fuites: list[dict[str, str]] = []
    try:
        for nom, autre_id in sorted(autres.items()):
            snapshot_id = latest_constructeur_snapshot_id(
                autre_id, dsn=dsn, schema=schema
            )
            if snapshot_id is None:
                continue
            rows = connection.execute(
                f"SELECT record_key, identity_summary FROM {schema}.projection_records "
                "WHERE subject_id = %s AND snapshot_id = %s "
                "AND record_key LIKE 'constructeur:%%'",
                (autre_id, snapshot_id),
            ).fetchall()
            for record_key, identity in rows:
                identity = identity or {}
                piece = str(
                    identity.get("constructeur_piece")
                    or identity.get("source_piece")
                    or ""
                )
                if piece and piece in pieces:
                    fuites.append(
                        {
                            "sujet": nom,
                            "record_key": str(record_key),
                            "piece": piece,
                        }
                    )
                    continue
                if pieces and any(
                    f":{name}:" in str(record_key) or str(record_key).startswith(f"constructeur:{name}:")
                    for name in pieces
                ):
                    fuites.append(
                        {
                            "sujet": nom,
                            "record_key": str(record_key),
                            "piece": "record_key",
                        }
                    )
        if subject_id:
            snapshot_id = latest_constructeur_snapshot_id(
                subject_id, dsn=dsn, schema=schema
            )
            autres_pieces: set[str] = set()
            for nom in autres:
                autre_desc = dossier_courant(nom) if nom in {"greenloop", "moreau"} else None
                if autre_desc is not None and autre_desc.corpus_dir.is_dir():
                    autres_pieces.update(
                        path.name
                        for path in autre_desc.corpus_dir.iterdir()
                        if path.is_file()
                    )
            if snapshot_id is not None and autres_pieces:
                rows = connection.execute(
                    f"SELECT record_key, identity_summary FROM {schema}.projection_records "
                    "WHERE subject_id = %s AND snapshot_id = %s "
                    "AND record_key LIKE 'constructeur:%%'",
                    (subject_id, snapshot_id),
                ).fetchall()
                for record_key, identity in rows:
                    identity = identity or {}
                    piece = str(identity.get("constructeur_piece") or "")
                    if piece in autres_pieces:
                        fuites.append(
                            {
                                "sujet": dossier,
                                "record_key": str(record_key),
                                "piece": piece,
                            }
                        )
    finally:
        connection.close()
    return {
        "ok": not fuites,
        "fuites": fuites[:20],
        "n_fuites": len(fuites),
    }


def couverture_or(
    dossier: Dossier,
    elements_par_question: dict[str, list[dict[str, Any]]],
    *,
    gold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold = gold if gold is not None else load_gold(dossier)
    assertions = list(gold.get("assertions") or [])
    total = len(assertions)
    attendu = DENOMINATEURS_OR.get(dossier.nom)
    verbatim_ids: set[str] = set()
    valeur_ids: set[str] = set()
    claims_index: dict[str, dict[str, Any]] = {}
    for _qid, evidence in elements_par_question.items():
        pieces = {
            piece_basename(str(item.get("source") or ""))
            for item in evidence
            if item.get("source")
        }
        for assertion in assertions:
            aid = str(assertion.get("id") or "")
            piece = piece_basename(
                str(assertion.get("source") or assertion.get("piece") or "")
            )
            if piece and pieces and piece not in pieces:
                continue
            if aid not in verbatim_ids:
                tient, _rank = assertion_present_in_evidence(
                    assertion,
                    evidence,
                    claims_index,
                    "plancher",
                    mode_presence="verbatim",
                )
                if tient:
                    verbatim_ids.add(aid)
            if aid not in valeur_ids:
                tient, _rank = assertion_present_in_evidence(
                    assertion,
                    evidence,
                    claims_index,
                    "plancher",
                    mode_presence="valeur",
                )
                if tient:
                    valeur_ids.add(aid)
    return {
        "total_lu": total,
        "denominateur_attendu": attendu,
        "ecart_denominateur": None if attendu is None else total - attendu,
        "verbatim": f"{len(verbatim_ids)}/{total}",
        "valeur": f"{len(valeur_ids)}/{total}",
        "n_verbatim": len(verbatim_ids),
        "n_valeur": len(valeur_ids),
    }


def _rendre_markdown(rapport: dict[str, Any]) -> str:
    dossier = rapport["dossier"]
    lines = [
        f"# Contrôle d'ingestion — {dossier}",
        "",
        "## Par pièce",
        "",
        "| Pièce | rendus | publiés (valides) | recalés | rejetés | doublons | propositions | situations | temporalités | alternatives | part lignes | médiane citation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for piece in rapport.get("pieces") or []:
        lines.append(
            f"| `{piece['piece']}` | {piece['faits_rendus']} | {piece['faits_valides']} | "
            f"{piece['faits_recales']} | {piece['faits_rejetes']} | {piece['faits_doublons']} | "
            f"{piece.get('propositions', 0)} | {piece.get('situations', 0)} | "
            f"{piece.get('temporalites', 0)} | {piece.get('alternatives', 0)} | "
            f"{piece.get('part_lignes', 0):.1%} | {piece.get('longueur_mediane_citations', 0)} |"
        )
    instantane = rapport.get("instantane") or {}
    lines.extend(
        [
            "",
            "## Instantané courant",
            "",
            f"- Identifiant : `{instantane.get('snapshot_id')}` (dernier, non figé)",
            f"- Par nature : {instantane.get('par_nature') or {}}",
            f"- Hors `constructeur:` : {instantane.get('hors_constructeur', 'non vérifié')}",
            f"- `ingest:%` : {instantane.get('ingest', 'non vérifié')}",
            "",
            "## Lecteur par question (96 000)",
            "",
        ]
    )
    fournisseurs = rapport.get("fournisseurs") or {}
    if not fournisseurs and rapport.get("questions_zero") is not None:
        fournisseurs = {
            "brain_mrfs": {
                "questions_zero": rapport.get("questions_zero") or [],
                "couverture_or": rapport.get("couverture_or") or {},
            }
        }
    for nom, bloc in fournisseurs.items():
        lines.append(f"### `{nom}`")
        lines.append("")
        statut = bloc.get("statut")
        if statut and statut != "ok":
            lines.append(f"- Statut : {statut}")
            lines.append("")
            continue
        zeros = list(bloc.get("questions_zero") or [])
        lines.append(f"- Questions : {rapport.get('n_questions', 0)}")
        lines.append(f"- Questions à zéro élément : {len(zeros)}")
        if 0 < len(zeros) < 20:
            for qid in zeros:
                lines.append(f"- `{qid}`")
        couverture = bloc.get("couverture_or") or {}
        lines.extend(
            [
                f"- Verbatim : {couverture.get('verbatim', 'non vérifié')}",
                f"- Valeur : {couverture.get('valeur', 'non vérifié')}",
                "",
            ]
        )
    couverture = rapport.get("couverture_or") or {}
    if couverture and not fournisseurs:
        lines.extend(
            [
                "## Couverture de l'or (96 000)",
                "",
                f"- Assertions lues : {couverture.get('total_lu', 'non vérifié')}",
                f"- Dénominateur attendu : {couverture.get('denominateur_attendu')}",
                f"- Écart : {couverture.get('ecart_denominateur')}",
                f"- Verbatim : {couverture.get('verbatim', 'non vérifié')}",
                f"- Valeur : {couverture.get('valeur', 'non vérifié')}",
            ]
        )
    langue = rapport.get("langue") or {}
    lines.extend(
        [
            "",
            "## Langue",
            "",
            f"- Questions : {langue.get('questions')}",
            f"- Pièces : {langue.get('pieces')}",
            f"- Lecteur lexical : {langue.get('avertissement') or 'alignées'}",
            "",
            "## Isolation",
            "",
            f"- OK : {(rapport.get('isolation') or {}).get('ok')}",
            f"- Fuites : {(rapport.get('isolation') or {}).get('n_fuites', 0)}",
            "",
        ]
    )
    return "\n".join(lines)


def _mesurer_lecteur(
    lecteur: Any,
    questions: list[dict[str, Any]],
    *,
    dossier: Dossier,
    gold: dict[str, Any],
) -> dict[str, Any]:
    elements_par_question: dict[str, list[dict[str, Any]]] = {}
    n_par_question: dict[str, int] = {}
    questions_zero: list[str] = []
    for question in questions:
        qid = str(question["id"])
        elements = lecteur.contexte(question, k=0, connection=None)
        _rendered, kept = rendre_contexte(
            elements,
            budget_chars=MAX_BUDGET_CHARS,
            question_id=qid,
            style="brain",
            dossier=dossier,
        )
        elements_par_question[qid] = kept
        n_par_question[qid] = len(kept)
        if not kept:
            questions_zero.append(qid)
    return {
        "statut": "ok",
        "elements_par_question": n_par_question,
        "questions_zero": questions_zero,
        "couverture_or": couverture_or(dossier, elements_par_question, gold=gold),
    }


def _fournisseurs_defaut(
    dossier: str,
    current: Dossier,
    *,
    dsn: str,
    constructeur_root: Path | None,
) -> tuple[dict[str, Any], str]:
    try:
        faits = charger_faits(dossier, dsn=dsn)
    except Exception:
        faits = None
    lexical = BrainMrfsLexical(dossier=current, faits=faits)
    providers: dict[str, Any] = {"brain_mrfs_lexical": lexical}
    statut_embeddings = "embeddings absents"
    cache_path = cache_embeddings_faits_path(dossier, constructeur_root=constructeur_root)
    cache = charger_cache(cache_path) if cache_path.is_file() else {}
    if faits and cache and all(empreinte_texte(fait.text) in cache for fait in faits):
        providers["brain_mrfs"] = BrainMrfs(
            dossier=current,
            faits=faits,
            embeddings_faits=cache,
        )
        statut_embeddings = "ok"
    return providers, statut_embeddings


def controler_ingestion(
    dossier: str,
    *,
    dsn: str = BANC_PG_DSN,
    constructeur_root: Path | None = None,
    sortie: Path | None = None,
    provider: BrainMrfs | None = None,
    providers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dossier not in {"greenloop", "moreau"}:
        raise ValueError(f"dossier non autorisé : {dossier!r}")
    current = dossier_courant(dossier)
    root = constructeur_root or DEFAULT_CONSTRUCTEUR_ROOT
    out_dir = sortie or (constructeur_dossier_root(dossier, constructeur_root=root) / "controle")
    etat = charger_etat(dossier, constructeur_root=root)
    faits_by_piece = _charger_faits(dossier, constructeur_root=root)
    registry = load_subject_registry()
    subject_id = registry.get(dossier)
    if not subject_id:
        raise RuntimeError(f"aucun sujet MRFS enregistré pour {dossier}")
    base = controle_dossier(
        dossier=dossier,
        subject_id=subject_id,
        faits_paths=_faits_paths(dossier, constructeur_root=root),
        dsn=dsn,
    )
    par_piece_base = base.get("pieces") or {}
    pieces_rapport: list[dict[str, Any]] = []
    corpus_names = sorted(
        path.name for path in current.corpus_dir.iterdir() if path.is_file()
    ) if current.corpus_dir.is_dir() else []
    for piece in corpus_names:
        counts = _agreger_etat_piece((etat.get("pieces") or {}).get(piece) or {})
        faits = faits_by_piece.get(piece) or []
        piece_path = current.corpus_dir / piece
        piece_text = piece_path.read_text(encoding="utf-8") if piece_path.is_file() else ""
        couverture = _couverture_lignes(piece_text, faits)
        db_counts = par_piece_base.get(piece) or {}
        pieces_rapport.append(
            {
                "piece": piece,
                **counts,
                "propositions": db_counts.get("propositions", 0),
                "situations": db_counts.get("situations", 0),
                "temporalites": db_counts.get("temporalites", 0),
                "alternatives": db_counts.get("alternatives", 0),
                **couverture,
            }
        )
    instantane = compter_instantane(subject_id, dsn=dsn)
    questions = load_questions(current)
    gold = load_gold(current)
    statut_embeddings = "ok"
    if providers is None:
        if provider is not None:
            providers = {"brain_mrfs_lexical": provider}
        else:
            providers, statut_embeddings = _fournisseurs_defaut(
                dossier, current, dsn=dsn, constructeur_root=root
            )
    blocs: dict[str, Any] = {}
    for nom, lecteur in providers.items():
        blocs[nom] = _mesurer_lecteur(
            lecteur, questions, dossier=current, gold=gold
        )
    if "brain_mrfs" not in blocs:
        blocs["brain_mrfs"] = {"statut": statut_embeddings, "questions_zero": [], "couverture_or": {}}
    lexical_bloc = blocs.get("brain_mrfs_lexical") or next(iter(blocs.values()))
    questions_zero = list(lexical_bloc.get("questions_zero") or [])
    n_par_question = dict(lexical_bloc.get("elements_par_question") or {})
    couverture = dict(lexical_bloc.get("couverture_or") or {})
    textes_questions = [str(q.get("question") or "") for q in questions]
    textes_pieces: list[str] = []
    for piece in corpus_names:
        path = current.corpus_dir / piece
        if path.is_file():
            textes_pieces.append(path.read_text(encoding="utf-8")[:4000])
    langue_questions = detecter_langue(textes_questions)
    langue_pieces = detecter_langue(textes_pieces)
    avertissement = ""
    if langue_questions != langue_pieces and langue_questions != "indeterminee":
        avertissement = (
            f"questions en {langue_questions}, pièces en {langue_pieces} "
            f"({len(questions_zero)} questions à zéro élément)"
        )
    isolation = isolation_constructeur(dossier, dsn=dsn)
    rapport = {
        "dossier": dossier,
        "sujet": subject_id,
        "budget": MAX_BUDGET_CHARS,
        "pieces": pieces_rapport,
        "instantane": instantane,
        "n_questions": len(questions),
        "elements_par_question": n_par_question,
        "questions_zero": questions_zero,
        "couverture_or": couverture,
        "fournisseurs": blocs,
        "embeddings_faits": statut_embeddings,
        "langue": {
            "questions": langue_questions,
            "pieces": langue_pieces,
            "declaree": current.langue,
            "avertissement": avertissement,
        },
        "isolation": isolation,
        "base": {"totaux": base.get("totaux"), "instantane": base.get("instantane")},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "controle.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "controle.md").write_text(_rendre_markdown(rapport) + "\n", encoding="utf-8")
    rapport["sortie"] = str(out_dir)
    return rapport


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contrôle d'ingestion des situations MRFS")
    parser.add_argument("--dossier", required=True, help="greenloop ou moreau")
    parser.add_argument("--dsn", default=BANC_PG_DSN)
    parser.add_argument("--constructeur-root", type=Path, default=None)
    parser.add_argument("--sortie", type=Path, default=None)
    args = parser.parse_args(argv)
    if not _pg_ready(args.dsn):
        print("PostgreSQL indisponible sur le cluster banc", file=sys.stderr)
        return 1
    rapport = controler_ingestion(
        args.dossier,
        dsn=args.dsn,
        constructeur_root=args.constructeur_root,
        sortie=args.sortie,
    )
    print(json.dumps(
        {
            "dossier": rapport["dossier"],
            "questions_zero": len(rapport["questions_zero"]),
            "couverture_or": rapport["couverture_or"],
            "sortie": rapport["sortie"],
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
