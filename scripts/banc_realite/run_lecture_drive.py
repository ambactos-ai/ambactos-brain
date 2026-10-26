#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CLI lecture Drive — inventaire, lecture de passage, bilan, purge."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

EXIT_ECHEC = 1
EXIT_JETON = 2


def _jeton_drive() -> str:
    jeton = os.environ.get("BANC_DRIVE_ACCESS_TOKEN", "").strip()
    if not jeton:
        print(
            "Variable d'environnement BANC_DRIVE_ACCESS_TOKEN absente ou vide.",
            file=sys.stderr,
        )
        sys.exit(EXIT_JETON)
    return jeton


def _afficher_comptes_traitements(registre: list[dict[str, Any]]) -> None:
    from banc_realite.lecture_drive import compte_traitements

    comptes = compte_traitements(registre)
    print("=== Par traitement ===")
    for traitement in sorted(comptes["traitement"]):
        bloc = comptes["traitement"][traitement]
        print(f"  {traitement}: {bloc['fichiers']} fichiers, {bloc['volume']} octets")
    print("=== Par famille ===")
    for famille in sorted(comptes["famille"]):
        bloc = comptes["famille"][famille]
        print(f"  {famille}: {bloc['fichiers']} fichiers, {bloc['volume']} octets")
    a_trancher = [
        entree
        for entree in registre
        if entree.get("traitement") == "a_trancher"
    ]
    if a_trancher:
        print("=== À trancher ===")
        for entree in a_trancher:
            print(f"  {entree.get('chemin', '')}/{entree.get('nom', '')} ({entree.get('mime', '')})")


def _afficher_comptes_lecture(registre: list[dict[str, Any]], sortie_dir: Path | None = None) -> None:
    from banc_realite.lecture_drive import compte_lecture, compte_lecture_par_groupe

    comptes = compte_lecture(registre)
    print("=== Lecture par famille ===")
    for famille in sorted(comptes):
        bloc = comptes[famille]
        print(
            f"  {famille}: {bloc['fichiers_lus']} lus, "
            f"{bloc['lignes']} lignes, {bloc['caracteres']} caractères, "
            f"{bloc['ocr']} OCR, {bloc['avertissements']} avertissements, "
            f"{bloc['erreurs']} erreurs"
        )
    if sortie_dir is not None:
        par_groupe = compte_lecture_par_groupe(registre, sortie_dir)
        if par_groupe:
            print("=== Lecture par groupe ===")
            for groupe in sorted(par_groupe):
                bloc = par_groupe[groupe]
                print(
                    f"  {groupe}: {bloc['pieces']} pièces, "
                    f"{bloc['lignes']} lignes, {bloc['caracteres']} caractères, "
                    f"--pieces-dir {bloc['pieces_dir']}"
                )


def inventorier(
    *,
    drive_id: str,
    regles_path: Path,
    sortie_dir: Path,
    depot_brut: Path | None = None,
) -> int:
    if depot_brut is not None:
        from banc_realite.lecture_depot_brut import inventorier_depot

        registre = inventorier_depot(
            depot_dir=depot_brut,
            regles_path=regles_path,
            sortie_dir=sortie_dir,
        )
        _afficher_comptes_traitements(registre)
        return 0

    from banc_realite.lecture_drive import (
        ClientDriveHttp,
        aiguiller,
        charger_regles,
        construire_registre,
        ecrire_json_atomique,
        fusionner_registre,
        _retirer_piece_cache,
    )

    client = ClientDriveHttp(_jeton_drive())
    fichiers = client.lister(drive_id)
    nouveau = construire_registre(fichiers)
    regles = charger_regles(regles_path)
    aiguiller(nouveau, regles)
    sortie_dir.mkdir(parents=True, exist_ok=True)
    registre_path = sortie_dir / "registre.json"
    ancien: list[dict[str, Any]] = []
    if registre_path.is_file():
        ancien = json.loads(registre_path.read_text(encoding="utf-8"))
    registre, disparues = fusionner_registre(ancien, nouveau)
    if disparues:
        print("=== Disparus du Drive ===")
        for entree in disparues:
            print(f"  {entree.get('nom', '')} ({entree.get('chemin', '')})")
            _retirer_piece_cache(sortie_dir, entree)
    ecrire_json_atomique(registre_path, registre)
    _afficher_comptes_traitements(registre)
    return 0


def lire(
    *,
    sortie_dir: Path,
    refaire: bool = False,
    seulement: list[str] | None = None,
    depot_brut: Path | None = None,
) -> int:
    registre_path = sortie_dir / "registre.json"
    if not registre_path.is_file():
        print(f"Registre absent: {registre_path}", file=sys.stderr)
        return EXIT_ECHEC
    registre = json.loads(registre_path.read_text(encoding="utf-8"))
    filtre = set(seulement) if seulement else None
    if depot_brut is not None:
        from banc_realite.lecture_depot_brut import lire_dossier_depot

        lire_dossier_depot(
            registre,
            depot_brut,
            sortie_dir,
            refaire=refaire,
            seulement=filtre,
        )
    else:
        from banc_realite.lecture_drive import ClientDriveHttp, lire_dossier

        client = ClientDriveHttp(_jeton_drive())
        lire_dossier(registre, client, sortie_dir, refaire=refaire, seulement=filtre)
    _afficher_comptes_lecture(registre, sortie_dir)
    return 0


def appliquer_manifeste_cmd(
    *,
    depot_brut: Path,
    sortie_dir: Path,
) -> int:
    from banc_realite.lecture_depot_brut import appliquer_manifeste

    registre_path = sortie_dir / "registre.json"
    if not registre_path.is_file():
        print(f"Registre absent: {registre_path}", file=sys.stderr)
        return EXIT_ECHEC
    registre = json.loads(registre_path.read_text(encoding="utf-8"))
    stats = appliquer_manifeste(depot_brut, registre, sortie_dir)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


def bilan(sortie_dir: Path) -> int:
    registre_path = sortie_dir / "registre.json"
    if not registre_path.is_file():
        print(f"Registre absent: {registre_path}", file=sys.stderr)
        return EXIT_ECHEC
    registre = json.loads(registre_path.read_text(encoding="utf-8"))
    _afficher_comptes_traitements(registre)
    _afficher_comptes_lecture(registre, sortie_dir)
    return 0


def purger_cmd(sortie_dir: Path) -> int:
    from banc_realite.lecture_drive import purger

    purger(sortie_dir)
    print("Cache de lecture purgé.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lecture Drive sur place")
    parser.add_argument(
        "commande",
        choices=("inventorier", "lire", "bilan", "purger", "appliquer-manifeste"),
    )
    parser.add_argument("--sortie-dir", type=Path, required=True)
    parser.add_argument("--drive-id", default="", help="ID du Drive partagé (inventorier)")
    parser.add_argument("--regles", type=Path, default=None, help="Fichier JSON de règles")
    parser.add_argument("--refaire", action="store_true", help="Relire tous les fichiers")
    parser.add_argument(
        "--seulement",
        nargs="+",
        default=None,
        help="Limiter la lecture à certains traitements",
    )
    parser.add_argument(
        "--depot-brut",
        type=Path,
        default=None,
        help="Racine tenant du dépôt brut (sans réseau)",
    )
    args = parser.parse_args(argv)

    if args.commande == "appliquer-manifeste":
        if args.depot_brut is None:
            print("--depot-brut requis pour appliquer-manifeste", file=sys.stderr)
            return EXIT_ECHEC
        return appliquer_manifeste_cmd(depot_brut=args.depot_brut, sortie_dir=args.sortie_dir)

    if args.commande == "inventorier":
        if args.depot_brut is None and not args.drive_id:
            print("--drive-id requis pour inventorier (sans --depot-brut)", file=sys.stderr)
            return EXIT_ECHEC
        if args.regles is None:
            print("--regles requis pour inventorier", file=sys.stderr)
            return EXIT_ECHEC
        return inventorier(
            drive_id=args.drive_id,
            regles_path=args.regles,
            sortie_dir=args.sortie_dir,
            depot_brut=args.depot_brut,
        )
    if args.commande == "lire":
        return lire(
            sortie_dir=args.sortie_dir,
            refaire=args.refaire,
            seulement=args.seulement,
            depot_brut=args.depot_brut,
        )
    if args.commande == "bilan":
        return bilan(args.sortie_dir)
    return purger_cmd(args.sortie_dir)


if __name__ == "__main__":
    raise SystemExit(main())
