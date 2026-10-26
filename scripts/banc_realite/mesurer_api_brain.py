#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Mesure latence et parité de contexte de POST /tenants/{t}/rechercher."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from brain.api_brain.tenants import TenantConfig

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "scripts", ROOT / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

QUESTIONS_OR = ("GR-Q01", "GR-Q02", "GR-Q06", "GR-Q22", "GR-Q30")
FOURNISSEUR = "objets_sens_puis_rag"
SUFFIXE_CONTEXTE = "ontologie-v1"


def creer_app_mesure(**kwargs: Any) -> Any:
    """Fabrique d'app du banc : reconstitution à la demande toujours éteinte."""
    from brain.api_brain import creer_app

    kwargs["reconstitution_a_la_demande"] = False
    return creer_app(**kwargs)


def _cle_sha256(texte: str) -> str:
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def _sync_cache_questions_api(racine: Path, cache_banc: dict[str, list[float]]) -> None:
    path = racine / "embeddings-questions.json"
    disque: dict[str, list[float]] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            disque = {
                str(k): [float(x) for x in v]
                for k, v in payload.items()
                if isinstance(v, list)
            }
    for texte, vecteur in cache_banc.items():
        disque[_cle_sha256(texte)] = vecteur
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(disque, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def chemin_cache_reference(cfg: TenantConfig, cache_reference: Path | None = None) -> Path:
    """Chemin du cache d'embeddings questions du banc (jamais le cache API du tenant)."""
    from banc_realite.dossiers import DOSSIERS

    if cfg.dossier_banc and cfg.dossier_banc in DOSSIERS:
        return DOSSIERS[cfg.dossier_banc].cache_embeddings_questions
    if cache_reference is not None:
        return cache_reference.expanduser().resolve()
    raise ValueError(
        "cache de référence requis : préciser --cache-reference <json> "
        "ou un tenant avec dossier_banc connu"
    )


def _comparer_contextes(
    api_elements: list[dict[str, Any]],
    or_elements: list[dict[str, Any]],
) -> bool:
    if len(api_elements) != len(or_elements):
        return False
    for gauche, droite in zip(api_elements, or_elements):
        if str(gauche.get("text") or "") != str(droite.get("text") or ""):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mesure API Brain rechercher")
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument(
        "--profil",
        metavar="QUESTION_ID",
        help="cProfile sur une seule question (ex. GR-Q01), imprime les 30 premières lignes",
    )
    parser.add_argument("--tenant", default="greenloop")
    parser.add_argument(
        "--cache-reference",
        type=Path,
        default=None,
        metavar="JSON",
        help="cache banc des embeddings questions (obligatoire sans dossier_banc)",
    )
    args = parser.parse_args(argv)

    from banc_realite.dossiers import dossier_courant
    from banc_realite.mesure_comparatif_etage2 import lire_contextes_mesure_bruts
    from banc_realite.or_dossier import load_questions
    from brain.api_brain.recherche import (
        BUDGET_DEFAUT,
        make_compteur_questions,
        plafond_embeddings_questions_depuis_cfg,
        prechauffer_tenant,
        tronquer_contexte,
    )
    from brain.api_brain.tenants import resoudre_tenant
    from fastapi.testclient import TestClient

    cfg = resoudre_tenant(args.tenant)
    if cfg is None:
        print(json.dumps({"erreur": f"tenant inconnu : {args.tenant}"}), file=sys.stderr)
        return 2

    from brain.api_brain.tenants import dossier_de_tenant

    dossier = dossier_de_tenant(cfg)  # même résolution que l'API (158-U), plus DOSSIERS
    try:
        cache_banc_path = chemin_cache_reference(cfg, args.cache_reference)
    except ValueError as exc:
        print(json.dumps({"erreur": str(exc)}), file=sys.stderr)
        return 2
    if not cache_banc_path.is_file():
        print(
            json.dumps(
                {"erreur": f"cache de référence absent : {cache_banc_path}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    cache_banc = json.loads(cache_banc_path.read_text(encoding="utf-8"))
    _sync_cache_questions_api(cfg.racine, cache_banc)

    plafond_embeddings = plafond_embeddings_questions_depuis_cfg(cfg.plafonds)
    compteur = make_compteur_questions(cfg.racine, plafond=plafond_embeddings)
    avant = compteur.lire()

    prechauffer_tenant(cfg, embed_cache_initial=dict(cache_banc))
    app = creer_app_mesure()
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {_jeton(args.tenant)}"}

    toutes = load_questions(dossier)
    par_id = {q["id"]: q for q in toutes}
    if args.profil:
        return _profil_une_question(
            client, headers, args.tenant, par_id, args.profil, compteur, avant
        )
    if args.questions > len(QUESTIONS_OR):
        ids = [q["id"] for q in toutes[: args.questions]]
    else:
        ids = list(QUESTIONS_OR[: args.questions])
    bruts = lire_contextes_mesure_bruts(
        FOURNISSEUR, dossier=dossier, suffixe=SUFFIXE_CONTEXTE
    )

    resultats: list[dict[str, Any]] = []
    for qid in ids:
        question = par_id[qid]
        texte = str(question["question"])
        or_elements = bruts.get((qid, None))
        if or_elements is None:
            or_elements = next(
                (elts for (rid, _k), elts in bruts.items() if rid == qid),
                None,
            )
        if or_elements is None:
            print(json.dumps({"erreur": f"contexte or absent pour {qid}"}), file=sys.stderr)
            return 2

        t0 = time.perf_counter()
        rep1 = client.post(
            f"/tenants/{args.tenant}/rechercher",
            headers=headers,
            json={"question": texte, "budget": BUDGET_DEFAUT},
        )
        t1 = time.perf_counter()
        if rep1.status_code != 200:
            print(
                json.dumps({"erreur": rep1.status_code, "detail": rep1.text}),
                file=sys.stderr,
            )
            return 2
        rep2 = client.post(
            f"/tenants/{args.tenant}/rechercher",
            headers=headers,
            json={"question": texte, "budget": BUDGET_DEFAUT},
        )
        t2 = time.perf_counter()
        if rep2.status_code != 200:
            print(
                json.dumps({"erreur": rep2.status_code, "detail": rep2.text}),
                file=sys.stderr,
            )
            return 2
        ctx = rep2.json().get("contexte") or []
        or_budget, _ = tronquer_contexte(list(or_elements), budget=BUDGET_DEFAUT)
        identique = _comparer_contextes(ctx, or_budget)
        resultats.append(
            {
                "question_id": qid,
                "t1_s": round(t1 - t0, 4),
                "t2_s": round(t2 - t1, 4),
                "identique": identique,
                "n_elements": len(ctx),
            }
        )

    apres = compteur.lire()
    meta = {"embeddings_avant": avant, "embeddings_apres": apres}
    print(json.dumps({"mesures": resultats, "meta": meta}, ensure_ascii=False, indent=2))
    return 0


def _profil_une_question(
    client: Any,
    headers: dict[str, str],
    tenant: str,
    par_id: dict[str, dict[str, Any]],
    qid: str,
    compteur: Any,
    avant: int,
) -> int:
    import cProfile
    import pstats
    from io import StringIO

    from brain.api_brain.recherche import BUDGET_DEFAUT

    question = par_id.get(qid)
    if question is None:
        print(json.dumps({"erreur": f"question inconnue : {qid}"}), file=sys.stderr)
        return 2
    texte = str(question["question"])

    def _appel() -> None:
        rep = client.post(
            f"/tenants/{tenant}/rechercher",
            headers=headers,
            json={"question": texte, "budget": BUDGET_DEFAUT},
        )
        if rep.status_code != 200:
            raise RuntimeError(f"{rep.status_code}: {rep.text}")

    prof = cProfile.Profile()
    prof.enable()
    _appel()
    prof.disable()
    stream = StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(30)
    apres = compteur.lire()
    print(stream.getvalue())
    print(
        json.dumps(
            {"question_id": qid, "embeddings_avant": avant, "embeddings_apres": apres},
            ensure_ascii=False,
        )
    )
    return 0


def _jeton(tenant: str) -> str:
    import os
    import re

    cle = "BRAIN_TENANT_" + re.sub(r"[^a-zA-Z0-9]", "_", tenant).upper() + "_JETON"
    valeur = os.environ.get(cle)
    if not valeur:
        raise RuntimeError(f"jeton absent : {cle}")
    return valeur


if __name__ == "__main__":
    raise SystemExit(main())
