#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CLI reconstitution par entité — recensement, rassemblement, reconstitution."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

EXIT_ECHEC = 1
EXIT_PLAFOND = 3


def _refuse_appel_reel_sous_pytest(chat_fn: Callable[..., Any] | None) -> None:
    if chat_fn is not None:
        return
    if "pytest" in sys.modules and os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("appel réel interdit sous pytest")


def _lire_registre(sortie_dir: Path) -> dict[str, Any]:
    path = sortie_dir / "registre.json"
    if not path.is_file():
        raise FileNotFoundError("registre.json absent — lancer recenser d'abord")
    return json.loads(path.read_text(encoding="utf-8"))


def _situations_dir(sortie_dir: Path) -> Path | None:
    candidat = sortie_dir / "par-piece"
    return candidat if candidat.is_dir() else None


def recenser_dossier(
    *,
    pieces_dir: Path,
    sortie_dir: Path,
    dossier: str,
    chat_fn: Callable[[str, str], tuple[str, Any]] | None = None,
    compteur=None,
    delai_s: int = 900,
    refaire: bool = False,
    depuis_ontologie: Path | None = None,
    registre_reference: Path | None = None,
) -> int:
    from banc_realite.plafond_appels import PlafondAtteint
    from banc_realite.reconstitution_entites import (
        recenser,
        recenser_depuis_ontologie,
        recensement_brut_reutilisable,
        resume_recensement,
    )

    if depuis_ontologie is not None:
        try:
            payload = recenser_depuis_ontologie(
                ontologie_path=depuis_ontologie,
                pieces_dir=pieces_dir,
                sortie_dir=sortie_dir,
                registre_reference_path=registre_reference,
            )
        except Exception as exc:
            print(f"échec : {exc!r}", file=sys.stderr)
            return EXIT_ECHEC
        print(json.dumps(resume_recensement(payload), ensure_ascii=False))
        return 0

    if refaire or not recensement_brut_reutilisable(sortie_dir / "recensement.brut.txt"):
        _refuse_appel_reel_sous_pytest(chat_fn)
    try:
        payload = recenser(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier=dossier,
            situations_dir=_situations_dir(sortie_dir),
            chat_fn=chat_fn,
            compteur=compteur,
            delai_s=delai_s,
            refaire=refaire,
        )
    except PlafondAtteint as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_PLAFOND
    except Exception as exc:
        print(f"échec : {exc!r}", file=sys.stderr)
        return EXIT_ECHEC
    print(json.dumps(resume_recensement(payload), ensure_ascii=False))
    return 0


def compter_commande(*, pieces_dir: Path, sortie_dir: Path) -> int:
    from banc_realite.reconstitution_entites import compter_dossier

    try:
        rapport = compter_dossier(pieces_dir=pieces_dir, sortie_dir=sortie_dir)
    except Exception as exc:
        print(f"échec : {exc!r}", file=sys.stderr)
        return EXIT_ECHEC
    print(json.dumps(rapport, ensure_ascii=False, indent=2))
    return 0


def rassembler_dossier(*, pieces_dir: Path, sortie_dir: Path) -> int:
    from banc_realite.reconstitution_entites import (
        _toutes_cles_entites,
        charger_corpus,
        index_cles_portees,
        rassembler,
    )

    try:
        registre = _lire_registre(sortie_dir)
        corpus, exclus = charger_corpus(pieces_dir)
        toutes_cles = _toutes_cles_entites(registre.get("entites") or [])
        index_cles = index_cles_portees(corpus, toutes_cles)
        paquets_dir = sortie_dir / "paquets"
        paquets_dir.mkdir(parents=True, exist_ok=True)
        for unite in registre.get("unites") or []:
            paquet = rassembler(
                unite, corpus, toutes_cles=toutes_cles, index_cles=index_cles
            )
            (paquets_dir / f"{unite['id']}.json").write_text(
                json.dumps(paquet, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        print(f"échec : {exc!r}", file=sys.stderr)
        return EXIT_ECHEC
    print(
        json.dumps(
            {
                "unites": len(registre.get("unites") or []),
                "pieces_exclues": len(exclus),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _executer_unite(
    unite: dict[str, Any],
    *,
    pieces_dir: Path,
    sortie_dir: Path,
    dossier: str,
    chat_fn: Callable[[str, str], tuple[str, Any]] | None,
    compteur,
    delai_s: int,
    reutiliser_dir: Path | None = None,
    index_empreinte: dict[str, str] | None = None,
    echecs_anciens: set[str] | None = None,
    reemploi_trace: dict[str, str] | None = None,
) -> None:
    from banc_realite.reconstitution_entites import (
        _toutes_cles_entites,
        charger_corpus,
        index_cles_portees,
        rassembler,
        reconstituer_unite,
        tenter_reemployer_unite,
    )

    paquet_path = sortie_dir / "paquets" / f"{unite['id']}.json"
    if paquet_path.is_file():
        paquet = json.loads(paquet_path.read_text(encoding="utf-8"))
    else:
        corpus, _ = charger_corpus(pieces_dir)
        registre = _lire_registre(sortie_dir)
        toutes_cles = _toutes_cles_entites(registre.get("entites") or [])
        index_cles = index_cles_portees(corpus, toutes_cles)
        paquet = rassembler(
            unite, corpus, toutes_cles=toutes_cles, index_cles=index_cles
        )
    if (
        reutiliser_dir is not None
        and index_empreinte is not None
        and echecs_anciens is not None
    ):
        ancien_id = tenter_reemployer_unite(
            unite,
            paquet,
            ancien_dir=reutiliser_dir,
            sortie_dir=sortie_dir,
            index_empreinte=index_empreinte,
            echecs_anciens=echecs_anciens,
        )
        if ancien_id is not None:
            if reemploi_trace is not None:
                reemploi_trace[unite["id"]] = ancien_id
            return
    corpus, _ = charger_corpus(pieces_dir)
    reconstituer_unite(
        unite=unite,
        paquet=paquet,
        corpus=corpus,
        sortie_dir=sortie_dir,
        dossier=dossier,
        chat_fn=chat_fn,
        compteur=compteur,
        delai_s=delai_s,
    )


def reconstituer_dossier(
    *,
    pieces_dir: Path,
    sortie_dir: Path,
    dossier: str,
    unites: str | None = None,
    parallele: int = 1,
    chat_fn: Callable[[str, str], tuple[str, Any]] | None = None,
    compteur=None,
    delai_s: int = 900,
    reutiliser: Path | None = None,
) -> int:
    from banc_realite.plafond_appels import PlafondAtteint
    from banc_realite.reconstitution_entites import (
        _echecs_unites_ancien,
        index_paquets_par_empreinte,
    )

    _refuse_appel_reel_sous_pytest(chat_fn)
    try:
        registre = _lire_registre(sortie_dir)
    except Exception as exc:
        print(f"échec : {exc!r}", file=sys.stderr)
        return EXIT_ECHEC
    selection = None
    if unites:
        selection = {item.strip() for item in unites.split(",") if item.strip()}
    cibles = [
        unite
        for unite in registre.get("unites") or []
        if selection is None or unite["id"] in selection
    ]
    index_empreinte: dict[str, str] | None = None
    echecs_anciens: set[str] | None = None
    reemploi_trace: dict[str, str] = {}
    if reutiliser is not None:
        index_empreinte = index_paquets_par_empreinte(reutiliser)
        echecs_anciens = _echecs_unites_ancien(reutiliser)
    echecs: list[dict[str, str]] = []
    plafond = False
    workers = max(1, parallele)
    kwargs_communs = {
        "pieces_dir": pieces_dir,
        "sortie_dir": sortie_dir,
        "dossier": dossier,
        "chat_fn": chat_fn,
        "compteur": compteur,
        "delai_s": delai_s,
        "reutiliser_dir": reutiliser,
        "index_empreinte": index_empreinte,
        "echecs_anciens": echecs_anciens,
        "reemploi_trace": reemploi_trace,
    }
    if workers == 1:
        for unite in cibles:
            sortie = sortie_dir / "entites" / f"{unite['id']}.json"
            if sortie.is_file():
                continue
            try:
                _executer_unite(unite, **kwargs_communs)
            except PlafondAtteint as exc:
                print(str(exc), file=sys.stderr)
                plafond = True
                break
            except Exception as exc:
                print(f"{unite['id']} : {exc!r}", file=sys.stderr)
                echecs.append({"unite": unite["id"], "cause": repr(exc)})
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for unite in cibles:
                if (sortie_dir / "entites" / f"{unite['id']}.json").is_file():
                    continue
                futures[pool.submit(_executer_unite, unite, **kwargs_communs)] = unite
            for fut in as_completed(futures):
                unite = futures[fut]
                try:
                    fut.result()
                except PlafondAtteint as exc:
                    print(str(exc), file=sys.stderr)
                    plafond = True
                    break
                except Exception as exc:
                    print(f"{unite['id']} : {exc!r}", file=sys.stderr)
                    echecs.append({"unite": unite["id"], "cause": repr(exc)})
    if reemploi_trace:
        (sortie_dir / "reemploi.json").write_text(
            json.dumps(reemploi_trace, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if echecs:
        (sortie_dir / "echecs.json").write_text(
            json.dumps(echecs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if plafond:
        return EXIT_PLAFOND
    return EXIT_ECHEC if echecs else 0


def bilan_commande(*, sortie_dir: Path) -> int:
    from banc_realite.reconstitution_entites import bilan

    print(json.dumps(bilan(sortie_dir), ensure_ascii=False, indent=2))
    return 0


def regrouper_dossier(
    *,
    pieces_dir: Path,
    sortie_dir: Path,
    dossier: str,
    chat_fn: Callable[[str, str], tuple[str, Any]] | None = None,
    compteur=None,
    delai_s: int = 900,
    refaire: bool = False,
) -> int:
    from banc_realite.plafond_appels import PlafondAtteint
    from banc_realite.reconstitution_entites import (
        regrouper_reliquat,
        reliquat_peut_reprendre,
    )

    if refaire or not reliquat_peut_reprendre(
        pieces_dir=pieces_dir, sortie_dir=sortie_dir
    ):
        _refuse_appel_reel_sous_pytest(chat_fn)
    try:
        payload = regrouper_reliquat(
            pieces_dir=pieces_dir,
            sortie_dir=sortie_dir,
            dossier=dossier,
            chat_fn=chat_fn,
            compteur=compteur,
            delai_s=delai_s,
            refaire=refaire,
        )
    except PlafondAtteint as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_PLAFOND
    except Exception as exc:
        print(f"échec : {exc!r}", file=sys.stderr)
        return EXIT_ECHEC
    print(
        json.dumps(
            {
                "groupes": len(payload.get("groupes") or []),
                "unites_grandeur": payload.get("unites_grandeur", 0),
                "lots": payload.get("lots", 0),
                "appel": bool(payload.get("appel")),
                "reparations": payload.get("reparations", 0),
                "ecartes": len(payload.get("ecartes") or []),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstitution par entité")
    parser.add_argument(
        "commande",
        choices=(
            "recenser",
            "compter",
            "rassembler",
            "regrouper",
            "reconstituer",
            "bilan",
        ),
    )
    parser.add_argument("--dossier", required=True)
    parser.add_argument("--pieces-dir", type=Path, required=True)
    parser.add_argument("--sortie-dir", type=Path, required=True)
    parser.add_argument("--unites", default="", help="Liste d'unités pour reconstituer")
    parser.add_argument(
        "--parallele",
        type=int,
        default=1,
        help="Unités reconstituées en parallèle (défaut 1).",
    )
    parser.add_argument(
        "--delai",
        type=int,
        default=900,
        help="Délai par appel, en secondes.",
    )
    parser.add_argument(
        "--constructeur-root",
        type=Path,
        default=None,
        help="Répertoire du compteur d'appels (défaut : compteur du chantier).",
    )
    parser.add_argument(
        "--refaire",
        action="store_true",
        help="Forcer un nouvel appel de recensement ou de regroupement (l'ancien brut va dans ecartes/).",
    )
    parser.add_argument(
        "--depuis-ontologie",
        type=Path,
        default=None,
        help="Recenser sans modèle à partir d'une ontologie JSON.",
    )
    parser.add_argument(
        "--registre-reference",
        type=Path,
        default=None,
        help="Registre de référence (par_periode, groupes-reliquat).",
    )
    parser.add_argument(
        "--reutiliser",
        type=Path,
        default=None,
        help="Répertoire de reconstitution antérieur pour réemployer les unités identiques.",
    )
    args = parser.parse_args(argv)
    args.sortie_dir.mkdir(parents=True, exist_ok=True)

    compteur = None
    if args.commande in ("recenser", "reconstituer", "regrouper"):
        from banc_realite.plafond_appels import make_compteur

        compteur = make_compteur(constructeur_root=args.constructeur_root)

    if args.commande == "recenser":
        return recenser_dossier(
            pieces_dir=args.pieces_dir,
            sortie_dir=args.sortie_dir,
            dossier=args.dossier,
            compteur=compteur,
            delai_s=args.delai,
            refaire=args.refaire,
            depuis_ontologie=args.depuis_ontologie,
            registre_reference=args.registre_reference,
        )
    if args.commande == "compter":
        return compter_commande(pieces_dir=args.pieces_dir, sortie_dir=args.sortie_dir)
    if args.commande == "rassembler":
        return rassembler_dossier(pieces_dir=args.pieces_dir, sortie_dir=args.sortie_dir)
    if args.commande == "regrouper":
        return regrouper_dossier(
            pieces_dir=args.pieces_dir,
            sortie_dir=args.sortie_dir,
            dossier=args.dossier,
            compteur=compteur,
            delai_s=args.delai,
            refaire=args.refaire,
        )
    if args.commande == "reconstituer":
        return reconstituer_dossier(
            pieces_dir=args.pieces_dir,
            sortie_dir=args.sortie_dir,
            dossier=args.dossier,
            unites=args.unites or None,
            parallele=args.parallele,
            compteur=compteur,
            delai_s=args.delai,
            reutiliser=args.reutiliser,
        )
    if args.commande == "bilan":
        return bilan_commande(sortie_dir=args.sortie_dir)
    return EXIT_ECHEC


if __name__ == "__main__":
    raise SystemExit(main())
