#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CLI situations composites par dossier — extraction, rapprochement, fusion."""
from __future__ import annotations

import argparse
import json
import os
import sys
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


def _lister_pieces(pieces_dir: Path, selection: str | None) -> list[Path]:
    if selection:
        noms = [item.strip() for item in selection.split(",") if item.strip()]
        return [pieces_dir / nom for nom in noms]
    return sorted(path for path in pieces_dir.iterdir() if path.is_file())


def compter_dossier(
    *,
    pieces_dir: Path,
    sortie_dir: Path,
    dossier: str,
    max_car: int = 8000,
    dates_max: int | None = None,
    resume_max: int | None = None,
    alias: bool = False,
) -> dict[str, Any]:
    from banc_realite.situations_composites import (
        INVITE_RAPPROCHEMENT,
        LOT_RECONCILIATION,
        OCTETS_MAX_INVITE,
        charger_par_piece,
        construire_invite_rapprochement,
        couverture_des_lignes,
        decouper_piece,
        lignes_de_donnees_tableau,
        remarques_numerotees,
    )

    par_piece = sortie_dir / "par-piece"
    detail: dict[str, Any] = {}
    total_appels = 0
    for path in sorted(pieces_dir.iterdir()):
        if not path.is_file():
            continue
        texte = path.read_text(encoding="utf-8")
        morceaux = decouper_piece(texte, max_car=max_car)
        donnees = lignes_de_donnees_tableau(path.name, texte)
        morceaux_tableau = sum(
            1 for debut, fin in morceaux if any(debut <= numero <= fin for numero in donnees)
        )
        sortie = par_piece / f"{path.name}.situations.json"
        restants = 0 if sortie.is_file() else len(morceaux)
        total_appels += restants
        entree: dict[str, Any] = {
            "caracteres": len(texte),
            "morceaux": len(morceaux),
            "morceaux_avec_lignes_de_tableau": morceaux_tableau,
            "appels_extraction_restants": restants,
        }
        if sortie.is_file():
            payload = json.loads(sortie.read_text(encoding="utf-8"))
            situations = payload.get("situations") or []
            entree["situations"] = len(situations)
            entree["remarques_non_etabli"] = sum(
                len(situation.get("non_etabli") or []) for situation in situations
            )
            if "couverture" in payload:
                entree["couverture"] = payload["couverture"]
            else:
                entree["couverture"] = couverture_des_lignes(donnees, situations)
        detail[path.name] = entree

    rapport: dict[str, Any] = {
        "dossier": dossier,
        "pieces": detail,
        "appels_extraction_restants": total_appels,
    }
    if par_piece.is_dir() and any(par_piece.glob("*.situations.json")):
        try:
            situations = charger_par_piece(par_piece)
        except KeyError:
            situations = {}
        if situations:
            invite, _table = construire_invite_rapprochement(
                situations,
                dates_max=dates_max,
                resume_max=resume_max,
                alias=alias,
            )
            suffixe = "\n\n" + INVITE_RAPPROCHEMENT
            descripteurs = invite[: -len(suffixe)] if invite.endswith(suffixe) else invite
            rapport["descripteurs_rapprochement_caracteres"] = len(descripteurs) + (
                1 if descripteurs else 0
            )
            taille = len(invite.encode("utf-8"))
            rapport["invite_rapprochement_octets"] = taille
            rapport["invite_rapprochement_sous_120000"] = taille <= OCTETS_MAX_INVITE
        fusion_path = sortie_dir / "situations-fusionnees.json"
        if fusion_path.is_file():
            fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
            remarques = remarques_numerotees(fusion.get("situations") or [])
            rapport["lots_reconciliation"] = (
                len(remarques) + LOT_RECONCILIATION - 1
            ) // LOT_RECONCILIATION
    return rapport


def extraire_dossier(
    *,
    pieces_dir: Path,
    sortie_dir: Path,
    dossier: str,
    pieces: str | None = None,
    max_car: int = 8000,
    formes_exactes: bool = False,
    refaire: bool = False,
    delai_s: int = 300,
    chat_fn: Callable[[str, str], tuple[str, Any]] | None = None,
    compteur=None,
    bruts_dir: Path | None = None,
    regle_tableaux: bool = False,
) -> int:
    from banc_realite.plafond_appels import PlafondAtteint
    from banc_realite.situations_composites import ecarter_bruts_piece, extraire_piece_par_morceaux

    _refuse_appel_reel_sous_pytest(chat_fn)
    par_piece = sortie_dir / "par-piece"
    par_piece.mkdir(parents=True, exist_ok=True)
    bruts = bruts_dir or (sortie_dir / "bruts")
    for path in _lister_pieces(pieces_dir, pieces):
        if not path.is_file():
            return EXIT_ECHEC
        sortie = par_piece / f"{path.name}.situations.json"
        if sortie.is_file() and not refaire:
            continue
        if refaire:
            ecarter_bruts_piece(bruts, path.name)
        try:
            payload = extraire_piece_par_morceaux(
                path.name,
                path.read_text(encoding="utf-8"),
                max_car=max_car,
                chat_fn=chat_fn,
                compteur=compteur,
                dossier=dossier,
                bruts_dir=bruts,
                formes_exactes=formes_exactes,
                delai_s=delai_s,
                regle_tableaux=regle_tableaux,
            )
        except PlafondAtteint as exc:
            print(f"plafond atteint sur {path.name} : {exc}", file=sys.stderr)
            return EXIT_PLAFOND
        except Exception as exc:
            print(f"échec sur {path.name} : {exc!r}", file=sys.stderr)
            return EXIT_ECHEC
        sortie.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def rapprocher_dossier(
    *,
    sortie_dir: Path,
    dossier: str,
    chat_fn: Callable[[str, str], tuple[str, Any]] | None = None,
    compteur=None,
    delai_s: int = 300,
    dates_max: int | None = None,
    resume_max: int | None = None,
    alias: bool = False,
) -> int:
    from banc_realite.situations_composites import charger_par_piece, rapprocher

    _refuse_appel_reel_sous_pytest(chat_fn)
    par_piece = sortie_dir / "par-piece"
    situations = charger_par_piece(par_piece)
    try:
        payload = rapprocher(
            situations,
            chat_fn=chat_fn,
            compteur=compteur,
            dossier=dossier,
            delai_s=delai_s,
            brut_path=sortie_dir / "rapprochement.brut.txt",
            dates_max=dates_max,
            resume_max=resume_max,
            alias=alias,
        )
    except Exception as exc:
        print(f"échec : {exc!r}", file=sys.stderr)
        return EXIT_ECHEC
    (sortie_dir / "groupes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "groupes": len(payload["groupes"]),
                "references_ecartees": len(payload["references_ecartees"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def fusionner_dossier(*, sortie_dir: Path) -> int:
    from banc_realite.situations_composites import charger_par_piece, fusionner

    par_piece = sortie_dir / "par-piece"
    groupes_path = sortie_dir / "groupes.json"
    if not groupes_path.is_file():
        return EXIT_ECHEC
    situations = charger_par_piece(par_piece)
    groupes = json.loads(groupes_path.read_text(encoding="utf-8"))["groupes"]
    fusionnees = fusionner(situations, groupes)
    (sortie_dir / "situations-fusionnees.json").write_text(
        json.dumps({"situations": fusionnees}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def reconcilier_dossier(
    *,
    sortie_dir: Path,
    pieces_dir: Path,
    dossier: str,
    chat_fn: Callable[[str, str], tuple[str, Any]] | None = None,
    compteur=None,
    delai_s: int = 300,
) -> int:
    from banc_realite.situations_composites import reconcilier

    _refuse_appel_reel_sous_pytest(chat_fn)
    fusion_path = sortie_dir / "situations-fusionnees.json"
    if not fusion_path.is_file():
        return EXIT_ECHEC
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    try:
        payload = reconcilier(
            par_piece_dir=sortie_dir / "par-piece",
            pieces_dir=pieces_dir,
            fusion=fusion,
            chat_fn=chat_fn,
            compteur=compteur,
            dossier=dossier,
            brut_dir=sortie_dir / "reconciliation-bruts",
            delai_s=delai_s,
        )
    except Exception as exc:
        print(f"échec : {exc!r}", file=sys.stderr)
        return EXIT_ECHEC
    (sortie_dir / "reconciliation-verdicts.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdicts": len(payload["verdicts"]),
                "ecartes": len(payload["ecartes"]),
                "sans_verdict": len(payload["sans_verdict"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def appliquer_dossier(*, sortie_dir: Path) -> int:
    from banc_realite.situations_composites import appliquer_reconciliation

    fusion_path = sortie_dir / "situations-fusionnees.json"
    verdicts_path = sortie_dir / "reconciliation-verdicts.json"
    if not fusion_path.is_file() or not verdicts_path.is_file():
        return EXIT_ECHEC
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    sortie = appliquer_reconciliation(fusion, verdicts)
    (sortie_dir / "situations-reconciliees.json").write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Situations composites par dossier")
    parser.add_argument("commande", choices=("compter", "extraire", "rapprocher", "fusionner", "reconcilier", "appliquer"))
    parser.add_argument("--pieces-dir", type=Path, required=True)
    parser.add_argument("--sortie-dir", type=Path, required=True)
    parser.add_argument("--dossier", required=True)
    parser.add_argument("--pieces", default="", help="Liste de pièces pour extraire")
    parser.add_argument("--max-car", type=int, default=8000)
    parser.add_argument("--formes-exactes", action="store_true")
    parser.add_argument("--regle-tableaux", action="store_true")
    parser.add_argument("--refaire", action="store_true")
    parser.add_argument("--delai", type=int, default=300, help="Délai par appel, en secondes.")
    parser.add_argument(
        "--dates-max",
        type=int,
        default=None,
        help="Nombre max de dates par descripteur de rapprochement.",
    )
    parser.add_argument(
        "--resume-max",
        type=int,
        default=None,
        help="Longueur max du résumé par descripteur de rapprochement.",
    )
    parser.add_argument(
        "--alias",
        action="store_true",
        help="Remplacer les références par des alias courts (r001, r002…).",
    )
    parser.add_argument(
        "--constructeur-root",
        type=Path,
        default=None,
        help="Répertoire du compteur d'appels (défaut : compteur du chantier).",
    )
    args = parser.parse_args(argv)

    args.sortie_dir.mkdir(parents=True, exist_ok=True)
    compteur = None
    if args.commande in ("extraire", "rapprocher", "reconcilier"):
        from banc_realite.plafond_appels import make_compteur

        # Le lanceur ne fixe ni ne relève le plafond : il lit celui du compteur.
        compteur = make_compteur(constructeur_root=args.constructeur_root)

    if args.commande == "compter":
        rapport = compter_dossier(
            pieces_dir=args.pieces_dir,
            sortie_dir=args.sortie_dir,
            dossier=args.dossier,
            max_car=args.max_car,
            dates_max=args.dates_max,
            resume_max=args.resume_max,
            alias=args.alias,
        )
        print(json.dumps(rapport, ensure_ascii=False, indent=2))
        return 0
    if args.commande == "extraire":
        return extraire_dossier(
            pieces_dir=args.pieces_dir,
            sortie_dir=args.sortie_dir,
            dossier=args.dossier,
            pieces=args.pieces or None,
            max_car=args.max_car,
            formes_exactes=args.formes_exactes,
            refaire=args.refaire,
            delai_s=args.delai,
            compteur=compteur,
            regle_tableaux=args.regle_tableaux,
        )
    if args.commande == "rapprocher":
        return rapprocher_dossier(
            sortie_dir=args.sortie_dir,
            dossier=args.dossier,
            compteur=compteur,
            delai_s=args.delai,
            dates_max=args.dates_max,
            resume_max=args.resume_max,
            alias=args.alias,
        )
    if args.commande == "fusionner":
        return fusionner_dossier(sortie_dir=args.sortie_dir)
    if args.commande == "reconcilier":
        return reconcilier_dossier(
            sortie_dir=args.sortie_dir,
            pieces_dir=args.pieces_dir,
            dossier=args.dossier,
            compteur=compteur,
            delai_s=args.delai,
        )
    if args.commande == "appliquer":
        return appliquer_dossier(sortie_dir=args.sortie_dir)
    return EXIT_ECHEC


if __name__ == "__main__":
    raise SystemExit(main())
