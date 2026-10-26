# SPDX-License-Identifier: AGPL-3.0-only
"""Constructeur de situations composites — portage du prototype hors dépôt.

Extraction par pièce, rapprochement, fusion, numérotation des preuves,
contrôle de la prose. Les trois invites sont celles du prototype, mot pour mot.
Chemins, fonction de chat et compteur injectables. Aucun appel réel depuis
les tests.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .extraction_situations import _objet_json

ChatFn = Callable[[str, str], tuple[str, Any]]

ROLES = (
    "account, actor, amount, approver, asset, audience, authority, basis, beneficiary, cause, "
    "channel, compensation, condition, control, counterparty, currency, customer, data, "
    "deadline, decision_maker, document, duration, end_time, evidence, impact, location, "
    "manager, matter, method, object, outcome, product_service, project, provider, quantity, "
    "reason, recipient, requester, responsible, risk, scope, sender, start_time, status, "
    "subject, supplier, target, work_content, worker"
)
CONCEPTS = (
    "approval, change, commitment, decision, goal, incident, notification, obligation, plan, "
    "request, review"
)
TYPES_ENTITES = (
    "person, organization, organizational-unit, party, agreement, document, location, "
    "product-or-service, project, system, asset, account, dataset"
)

INVITE_EXTRACTION = f"""Tu reconstitues la réalité décrite par une pièce d'un dossier.
Chaque ligne de la pièce est précédée de son numéro.

L'unité que tu rends est la SITUATION : quelque chose qui a eu lieu, qui dure, ou qui lie
des personnes dans le monde — pas une phrase de la pièce. Une situation réunit tout ce que
la pièce en dit, par qui que ce soit. Deux parties qui parlent du même événement parlent de
la MÊME situation, même si elles se contredisent.

Pour chaque situation, rends :
- "id" : "s1", "s2", …
- "titre" : un intitulé court et neutre ;
- "nature" : "evenement" | "processus" | "etat" | "relation" ;
- "concept" : un concept de cette liste s'il convient vraiment, sinon null : {CONCEPTS} ;
- "resume_neutre" : deux phrases au plus, sans prendre parti, sans rien ajouter à la pièce ;
- "acteurs" : chaque personne ou organisation impliquée, avec
    "nom" (tel qu'écrit dans la pièce),
    "type_entite" (un de : {TYPES_ENTITES}),
    "role" (son rôle DANS cette situation ; prends-le dans cette liste s'il convient :
      {ROLES} ; sinon écris un rôle court en français),
    "role_hors_catalogue" (true si tu as écrit un rôle hors de la liste) ;
- "chronologie" : les moments de la situation, dans l'ordre, chacun avec
    "quand_texte" (la date ou l'heure telle qu'écrite), "iso" ("AAAA-MM-JJ" si le jour est
    complet, sinon null), "quoi" (ce qui se passe à ce moment), "ligne_debut", "ligne_fin" ;
- "perceptions" : ce que CHAQUE acteur dit de la situation, une entrée par acteur et par
    prise de position, avec
    "acteur" (son nom),
    "position" : "affirme" | "conteste" | "nie" | "reconnait" | "rapporte" | "produit_une_piece",
    "ce_qu_il_dit" (une à deux phrases fidèles, sans rien ajouter),
    "preuves" : liste de {{"ligne_debut": n, "ligne_fin": m}} — les lignes ENTIÈRES de la pièce
    qui portent cette prise de position ;
- "points_de_desaccord" : ce sur quoi les acteurs se contredisent, en phrases courtes ;
- "non_etabli" : ce que la pièce annonce sans le montrer (pièce non annexée, conditionnel,
    ouï-dire), en phrases courtes ;
- "liens" : liste de {{"vers": "s2", "nature": "precede" | "suit" | "cause" | "repond_a" |
    "fait_partie_de"}} vers d'autres situations de ta réponse.

Règles :
- N'invente rien. Ce que la pièce ne dit pas n'existe pas.
- Ne tranche pas entre les acteurs.
- Ne recopie pas le texte dans "preuves" : donne seulement les numéros de ligne.

Réponds par un seul objet JSON, sans texte autour : {{"situations": [ … ]}}"""

REGLE_FORMES_EXACTES = (
    '\n\nRègle supplémentaire : dans "quoi", "ce_qu_il_dit", "resume" et "titre", recopie les nombres,\n'
    "montants, dates et identifiants dans la forme exacte de la pièce, caractère pour caractère\n"
    " (« 16.099 » reste « 16.099 », « 2025-01-08 » reste « 2025-01-08 »). Ne convertis pas les\n"
    "séparateurs décimaux, ne traduis pas les unités, ne reformate pas les dates."
)

REGLE_TABLEAUX = (
    "Règle pour les tableaux et registres : quand le passage est un tableau dont chaque ligne est\n"
    "un enregistrement (visite, pesée, facture, paiement), ne fais pas une situation par ligne.\n"
    "Regroupe les lignes par entité suivie (site, client, contrat, compte) : une situation par\n"
    "entité pour le passage, et chaque ligne du tableau devient un moment de sa chronologie, avec\n"
    "son numéro de ligne. Chaque ligne du tableau figure dans une chronologie, une seule fois. Une\n"
    "ligne qui porte une anomalie (valeur nulle, négative, provisoire, corrigée, non pesée, doublon)\n"
    "est aussi rapportée dans une perception, avec sa ligne en preuve. N'écris aucun texte avant ni\n"
    "après l'objet JSON."
)

INVITE_PROSE_PIECE = """Tu écris pour un avocat qui interroge un dossier. On te donne UNE situation,
déjà reconstituée : son type, ses acteurs et leurs rôles, sa chronologie, ce que chacun en
dit, et les passages de la pièce qui le prouvent. Chaque passage porte un identifiant : e1, e2…

Écris un texte de six à dix phrases, en français, en prose suivie, sans liste ni titre, qui
dit : de quoi il s'agit, qui est impliqué et à quel titre, ce qui s'est passé et quand, ce que
chaque partie en dit, sur quoi elles s'opposent, et ce qui n'est pas établi.

Règles :
- N'écris AUCUN fait, date, heure, montant, nom ou qualification qui ne soit pas dans la
  situation fournie.
- Ne cite jamais la pièce toi-même : n'emploie aucun guillemet. Pour appuyer une phrase sur un
  passage, écris son identifiant entre crochets à la fin de la phrase, ainsi : [e3]. Le
  passage exact sera affiché sous ton texte.
- Chaque affirmation sur ce qu'une partie dit porte au moins un identifiant.
- Ne tranche pas entre les parties. Emploie selon, d'après, affirme, conteste.
- Commence directement par le texte. Pas de phrase sur ce que tu vas faire, pas
  d'introduction, pas de conclusion, pas de conseil juridique.

Réponds par le texte seul."""

INVITE_RAPPROCHEMENT = """Voici des situations reconstituées à partir de plusieurs pièces d'un même dossier,
une par ligne : référence | nature | titre | dates | acteurs | résumé.

Regroupe les références qui décrivent LA MÊME situation du monde : le même événement, le même
état ou la même relation, vu par des pièces différentes. Deux pièces qui se contredisent sur
une même situation vont dans le même groupe.

Ne regroupe PAS deux situations parce qu'elles touchent au même sujet ou à la même procédure.
Un entretien préalable et le licenciement qui le suit sont deux situations. Des retards et
l'incident avec un client sont deux situations, même s'ils fondent le même licenciement.

Réponds par un seul objet JSON, sans texte autour :
{"groupes": [{"titre": "intitulé court et neutre", "references": ["piece#s1", "piece#s3"]}]}
N'y mets que les groupes d'au moins deux références. N'invente aucune référence."""

INVITE_RECONCILIATION = """On a lu les pièces d'un dossier une par une. En lisant chaque pièce, on a noté ce
qu'elle annonce sans le montrer : une pièce citée mais non jointe, un fait au conditionnel, un
témoin cité mais non entendu. Ces remarques sont vraies à l'échelle de LA PIÈCE lue.

Voici l'inventaire du dossier entier, puis les remarques. Pour chaque remarque, dis ce qu'elle
devient à l'échelle du DOSSIER.

Pour chaque remarque, rends :
- "id" : son identifiant ;
- "portee_dossier" :
    "etabli_par_une_piece" — une pièce de l'inventaire est ce qui manquait, ou le montre ;
    "non_etabli" — rien dans l'inventaire ne l'établit ;
    "incertain" — l'inventaire ne permet pas de le dire ;
- "pieces" : les noms de fichier, pris dans l'inventaire, qui l'établissent (liste vide sinon) ;
- "importance" : "haute" si la remarque touche un fait disputé entre les parties ou une
    preuve sur laquelle une partie s'appuie ; "moyenne" si elle touche un fait du dossier sans
    dispute ; "faible" si elle touche un renvoi de forme (texte de loi, règlement, barème,
    mention administrative) ;
- "motif" : une phrase courte.

Règles :
- Ne te fie qu'à l'inventaire. Si une pièce de l'inventaire a le bon intitulé mais que tu ne
  peux pas savoir si elle contient le détail demandé, réponds "incertain" et nomme-la.
- N'invente aucun nom de fichier.

Réponds par un seul objet JSON, sans texte autour : {"remarques": [ … ]}"""

LOT_RECONCILIATION = 70
OCTETS_MAX_INVITE = 120_000
CHAMP_PORTEE_MODELE = "portee_dossier"
ORDRE_IMPORTANCE = {"haute": 0, "moyenne": 1, "faible": 2, None: 3}

INVITE_PROSE = """Tu écris pour un avocat qui interroge un dossier. On te donne UNE situation,
reconstituée à partir de PLUSIEURS pièces : son type, ses acteurs et leurs rôles, sa
chronologie, ce que chacun en dit dans chaque pièce, et les passages qui le prouvent. Chaque
passage porte un identifiant : e1, e2…

Écris un texte de huit à douze phrases, en français, en prose suivie, sans liste ni titre, qui
dit : de quoi il s'agit, qui est impliqué et à quel titre, ce qui s'est passé et quand, ce que
chaque partie en dit et dans quelle pièce, comment le récit évolue d'une pièce à l'autre, sur
quoi les parties s'opposent, et ce qui n'est pas établi.

Règles :
- N'écris AUCUN fait, date, heure, montant, nom ou qualification qui ne soit pas dans la
  situation fournie.
- Ne cite jamais une pièce toi-même : n'emploie aucun guillemet. Pour appuyer une phrase sur
  un passage, écris son identifiant entre crochets à la fin de la phrase, ainsi : [e3]. Le
  passage exact sera affiché sous ton texte, avec sa pièce et ses lignes.
- Nomme la pièce quand tu dis qui affirme quoi : le procès-verbal, la lettre de licenciement,
  la requête, les conclusions.
- Chaque affirmation sur ce qu'une partie dit porte au moins un identifiant.
- Ne tranche pas entre les parties.
- Commence directement par le texte. Pas de phrase sur ce que tu vas faire, pas
  d'introduction, pas de conclusion, pas de conseil juridique.

Réponds par le texte seul."""


def _reduit(texte: str) -> str:
    return re.sub(r"\s+", " ", texte).strip()


def piece_numerotee(piece_text: str) -> str:
    return "\n".join(
        f"{n:4d}| {ligne}" for n, ligne in enumerate(piece_text.splitlines(), start=1)
    )


def _est_titre_markdown(ligne: str) -> bool:
    return bool(re.match(r"^\s*#{1,6}\s", ligne))


def decouper_piece(piece_text: str, *, max_car: int = 8000) -> list[tuple[int, int]]:
    """Découpe une pièce en morceaux contigus, sans trou ni recouvrement."""
    lignes = piece_text.splitlines()
    total = len(lignes)
    if total == 0:
        return [(1, 1)] if piece_text == "" else []

    morceaux: list[tuple[int, int]] = []
    debut = 0
    while debut < total:
        if len(lignes[debut]) > max_car:
            morceaux.append((debut + 1, debut + 1))
            debut += 1
            continue

        fin = debut
        taille = len(lignes[debut])
        while fin + 1 < total:
            suivante = lignes[fin + 1]
            if len(suivante) > max_car:
                break
            candidate = taille + 1 + len(suivante)
            if candidate > max_car:
                break
            fin += 1
            taille = candidate

        coupure = fin
        if fin < total - 1:
            for idx in range(fin, debut, -1):
                if not lignes[idx].strip():
                    coupure = idx
                    break
            if coupure == fin:
                for idx in range(debut + 1, fin + 1):
                    if _est_titre_markdown(lignes[idx]):
                        coupure = idx - 1
                        break
            if coupure < debut:
                coupure = debut

        morceaux.append((debut + 1, coupure + 1))
        debut = coupure + 1
    return morceaux


def _est_separateur_tableau_markdown(ligne: str) -> bool:
    stripped = ligne.strip()
    if "|" not in stripped:
        return False
    return bool(re.match(r"^\|?[\s:\-|]+\|?$", stripped))


def _detecter_tous_entetes_tableau(lignes: Sequence[str]) -> list[list[tuple[int, str]]]:
    groupes: list[list[tuple[int, str]]] = []
    index = 0
    total = len(lignes)
    while index < total:
        ligne = lignes[index]
        if "|" in ligne and index + 1 < total and _est_separateur_tableau_markdown(lignes[index + 1]):
            groupes.append([(index + 1, lignes[index]), (index + 2, lignes[index + 1])])
            index += 2
            continue
        index += 1
    if groupes:
        return groupes
    for index, ligne in enumerate(lignes):
        if not ligne.strip():
            continue
        if any(sep in ligne for sep in (",", ";", "\t")):
            return [[(index + 1, ligne)]]
        break
    return []


def _detecter_entetes_tableau(lignes: Sequence[str]) -> list[tuple[int, str]]:
    tous = _detecter_tous_entetes_tableau(lignes)
    return list(tous[0]) if tous else []


def lignes_de_donnees_tableau(piece: str, piece_text: str) -> list[int]:
    lignes = piece_text.splitlines()
    entetes_nums = {
        numero for groupe in _detecter_tous_entetes_tableau(lignes) for numero, _texte in groupe
    }
    if piece.lower().endswith(".csv"):
        return [index for index, _ligne in enumerate(lignes, start=1) if index not in entetes_nums]
    numeros: list[int] = []
    for index, ligne in enumerate(lignes, start=1):
        if index in entetes_nums:
            continue
        if ligne.lstrip().startswith("|") and not _est_separateur_tableau_markdown(ligne):
            numeros.append(index)
    return numeros


def _ligne_appartient_a_un_tableau(numero: int, lignes: Sequence[str], piece: str) -> bool:
    if not (1 <= numero <= len(lignes)):
        return False
    if piece.lower().endswith(".csv"):
        return True
    ligne = lignes[numero - 1]
    if ligne.lstrip().startswith("|") or _est_separateur_tableau_markdown(ligne):
        return True
    if numero < len(lignes) and "|" in ligne and _est_separateur_tableau_markdown(lignes[numero]):
        return True
    return False


def _entetes_rappel_morceau(
    lignes: Sequence[str],
    *,
    ligne_debut: int,
    ligne_fin: int,
    piece: str,
) -> list[tuple[int, str]]:
    if not lignes or not (1 <= ligne_debut <= len(lignes)):
        return []
    if not _ligne_appartient_a_un_tableau(ligne_debut, lignes, piece):
        return []
    donnees = [
        numero
        for numero in lignes_de_donnees_tableau(piece, "\n".join(lignes))
        if ligne_debut <= numero <= ligne_fin
    ]
    if not donnees:
        return []
    premiere = donnees[0]
    dernier: list[tuple[int, str]] = []
    for groupe in _detecter_tous_entetes_tableau(lignes):
        if groupe[0][0] < premiere:
            dernier = list(groupe)
    return dernier


def couverture_des_lignes(
    lignes_donnees: Sequence[int],
    situations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cites: set[int] = set()
    for situation in situations:
        for moment in situation.get("chronologie") or []:
            try:
                debut = int(moment["ligne_debut"])
                fin = int(moment["ligne_fin"])
            except (KeyError, TypeError, ValueError):
                continue
            cites.update(range(debut, fin + 1))
        for perception in situation.get("perceptions") or []:
            for preuve in perception.get("preuves") or []:
                try:
                    debut = int(preuve["ligne_debut"])
                    fin = int(preuve["ligne_fin"])
                except (KeyError, TypeError, ValueError):
                    continue
                cites.update(range(debut, fin + 1))
    donnees = list(lignes_donnees)
    citees = [numero for numero in donnees if numero in cites]
    non_citees = [numero for numero in donnees if numero not in cites]
    return {
        "lignes_de_donnees": len(donnees),
        "citees": len(citees),
        "non_citees": non_citees,
    }


def _reponse_lisible(brut: str) -> bool:
    try:
        _objet_json(brut, "situations")
    except (ValueError, TypeError):
        return False
    return True


def _reponse_rapprochement_lisible(brut: str) -> bool:
    try:
        _objet_json(brut, "groupes")
    except (ValueError, TypeError):
        return False
    return True


def _reponse_reconciliation_lisible(brut: str) -> bool:
    try:
        _objet_json(brut, "remarques")
    except (ValueError, TypeError):
        return False
    return True


def _ecarter_brut_fichier(brut_path: Path, *, horodatage: str | None = None) -> Path:
    brut_path = Path(brut_path)
    stamp = horodatage or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = brut_path.with_name(f"{brut_path.stem}.{stamp}.ecarte.txt")
    suffixe = 1
    while dest.exists():
        suffixe += 1
        dest = brut_path.with_name(f"{brut_path.stem}.{stamp}-{suffixe}.ecarte.txt")
    if brut_path.is_file():
        brut_path.rename(dest)
    return dest


def ecarter_reponse_brute(
    brut_path: Path,
    *,
    piece: str,
    rang: int,
    bruts_dir: Path,
    horodatage: str | None = None,
) -> Path:
    ecartes = Path(bruts_dir) / "ecartes"
    ecartes.mkdir(parents=True, exist_ok=True)
    stamp = horodatage or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = ecartes / f"{piece}.m{rang}.{stamp}.brut.txt"
    suffixe = 1
    while dest.exists():
        suffixe += 1
        dest = ecartes / f"{piece}.m{rang}.{stamp}-{suffixe}.brut.txt"
    brut_path = Path(brut_path)
    if brut_path.is_file():
        brut_path.rename(dest)
    return dest


def ecarter_bruts_piece(bruts_dir: Path | str, piece: str) -> list[Path]:
    dossier = Path(bruts_dir)
    if not dossier.is_dir():
        return []
    motif = re.compile(rf"^{re.escape(piece)}\.m(\d+)\.brut\.txt$")
    deplaces: list[Path] = []
    for path in sorted(dossier.iterdir()):
        if not path.is_file():
            continue
        match = motif.match(path.name)
        if match is None:
            continue
        deplaces.append(
            ecarter_reponse_brute(path, piece=piece, rang=int(match.group(1)), bruts_dir=dossier)
        )
    return deplaces


def _numeroter_lignes(
    lignes: Sequence[str],
    *,
    ligne_debut: int,
    rappels: Sequence[tuple[int, str]] = (),
) -> str:
    rows: list[str] = []
    for numero, texte in rappels:
        rows.append(f"(rappel, ligne {numero})")
        rows.append(f"{numero:4d}| {texte}")
    for offset, ligne in enumerate(lignes):
        rows.append(f"{ligne_debut + offset:4d}| {ligne}")
    return "\n".join(rows)


def _construire_invite_extraction(
    piece: str,
    corps: str,
    *,
    formes_exactes: bool = False,
    regle_tableaux: bool = False,
) -> str:
    invite = f"Pièce : {piece}\n\n{corps}\n\n{INVITE_EXTRACTION}"
    if formes_exactes:
        invite += REGLE_FORMES_EXACTES
    if regle_tableaux:
        invite += "\n\n" + REGLE_TABLEAUX
    return invite


def _morceau_a_lignes_de_donnees(
    piece: str,
    piece_text: str,
    *,
    ligne_debut: int,
    ligne_fin: int,
    total: int,
    max_car: int,
) -> bool:
    donnees = lignes_de_donnees_tableau(piece, piece_text)
    if total == 1 and len(piece_text) <= max_car:
        return bool(donnees)
    return any(ligne_debut <= numero <= ligne_fin for numero in donnees)


def _construire_invite_morceau(
    piece: str,
    piece_text: str,
    rang: int,
    total: int,
    ligne_debut: int,
    ligne_fin: int,
    *,
    entetes: Sequence[tuple[int, str]],
    max_car: int,
    formes_exactes: bool = False,
    regle_tableaux: bool = False,
) -> str:
    appliquer_tableaux = bool(
        regle_tableaux
        and _morceau_a_lignes_de_donnees(
            piece,
            piece_text,
            ligne_debut=ligne_debut,
            ligne_fin=ligne_fin,
            total=total,
            max_car=max_car,
        )
    )
    if total == 1 and len(piece_text) <= max_car:
        return _construire_invite_extraction(
            piece,
            piece_numerotee(piece_text),
            formes_exactes=formes_exactes,
            regle_tableaux=appliquer_tableaux,
        )
    lignes = piece_text.splitlines()[ligne_debut - 1 : ligne_fin]
    rappels = [
        (numero, texte)
        for numero, texte in entetes
        if numero < ligne_debut or numero > ligne_fin
    ]
    corps = _numeroter_lignes(lignes, ligne_debut=ligne_debut, rappels=rappels)
    entete = f"Pièce : {piece}\nmorceau {rang}/{total}, lignes {ligne_debut}-{ligne_fin}\n\n"
    invite = f"{entete}{corps}\n\n{INVITE_EXTRACTION}"
    if formes_exactes:
        invite += REGLE_FORMES_EXACTES
    if appliquer_tableaux:
        invite += "\n\n" + REGLE_TABLEAUX
    return invite


def _ligne_autorisee(
    numero: int,
    *,
    ligne_debut: int,
    ligne_fin: int,
    rappel: set[int],
) -> bool:
    return ligne_debut <= numero <= ligne_fin or numero in rappel


def _intervalle_autorise(
    debut: int,
    fin: int,
    *,
    ligne_debut: int,
    ligne_fin: int,
    rappel: set[int],
) -> bool:
    return _ligne_autorisee(debut, ligne_debut=ligne_debut, ligne_fin=ligne_fin, rappel=rappel) and _ligne_autorisee(
        fin, ligne_debut=ligne_debut, ligne_fin=ligne_fin, rappel=rappel
    )


def _prefixer_ids_situations(situations: Sequence[dict[str, Any]], rang: int) -> list[dict[str, Any]]:
    prefixe = f"m{rang}."
    sortie: list[dict[str, Any]] = []
    for situation in situations:
        copie = json.loads(json.dumps(situation))
        copie["id"] = f"{prefixe}{copie['id']}"
        for lien in copie.get("liens") or []:
            vers = lien.get("vers")
            if isinstance(vers, str) and vers and not vers.startswith("m"):
                lien["vers"] = f"{prefixe}{vers}"
        sortie.append(copie)
    return sortie


def _filtrer_lignes_hors_morceau(
    situations: Sequence[dict[str, Any]],
    *,
    ligne_debut: int,
    ligne_fin: int,
    rappel: set[int],
    rejets: list[str],
) -> list[dict[str, Any]]:
    gardees: list[dict[str, Any]] = []
    for situation in situations:
        copie = json.loads(json.dumps(situation))
        for perception in copie.get("perceptions") or []:
            preuves_gardees = []
            for preuve in perception.get("preuves") or []:
                try:
                    debut = int(preuve["ligne_debut"])
                    fin = int(preuve["ligne_fin"])
                except (KeyError, TypeError, ValueError):
                    rejets.append(f"{copie.get('id')} : preuve rejetée (lignes invalides)")
                    continue
                if _intervalle_autorise(
                    debut, fin, ligne_debut=ligne_debut, ligne_fin=ligne_fin, rappel=rappel
                ):
                    preuves_gardees.append(preuve)
                else:
                    rejets.append(
                        f"{copie.get('id')} : preuve rejetée (lignes {debut}-{fin} hors morceau)"
                    )
            perception["preuves"] = preuves_gardees
        chronologie_gardee = []
        for moment in copie.get("chronologie") or []:
            try:
                debut = int(moment["ligne_debut"])
                fin = int(moment["ligne_fin"])
            except (KeyError, TypeError, ValueError):
                rejets.append(f"{copie.get('id')} : moment rejeté (lignes invalides)")
                continue
            if _intervalle_autorise(
                debut, fin, ligne_debut=ligne_debut, ligne_fin=ligne_fin, rappel=rappel
            ):
                chronologie_gardee.append(moment)
            else:
                rejets.append(
                    f"{copie.get('id')} : moment rejeté (lignes {debut}-{fin} hors morceau)"
                )
        copie["chronologie"] = chronologie_gardee
        gardees.append(copie)
    return gardees


def passage(lignes: Sequence[str], debut: int, fin: int) -> str:
    if not (1 <= debut <= fin <= len(lignes)):
        raise ValueError(f"lignes hors pièce : {debut}-{fin}")
    return "\n".join(lignes[debut - 1 : fin])


def rattacher_lettres(
    payload: dict[str, Any],
    piece: str,
    piece_text: str,
) -> dict[str, Any]:
    lignes = piece_text.splitlines()
    rejets: list[str] = []
    for situation in payload.get("situations") or []:
        for perception in situation.get("perceptions") or []:
            gardees = []
            for preuve in perception.get("preuves") or []:
                try:
                    preuve["lettre"] = passage(
                        lignes, int(preuve["ligne_debut"]), int(preuve["ligne_fin"])
                    )
                    gardees.append(preuve)
                except (ValueError, KeyError, TypeError) as exc:
                    rejets.append(f"{situation.get('id')} : preuve rejetée ({exc})")
            perception["preuves"] = gardees
        for moment in situation.get("chronologie") or []:
            try:
                moment["lettre"] = passage(
                    lignes, int(moment["ligne_debut"]), int(moment["ligne_fin"])
                )
            except (ValueError, KeyError, TypeError) as exc:
                rejets.append(
                    f"{situation.get('id')} : moment sans lignes valides ({exc})"
                )
                moment["lettre"] = None
    payload["piece"] = piece
    payload["rejets"] = rejets
    return payload


def _chat_par_defaut(
    nature: str,
    *,
    compteur=None,
    dossier: str = "moreau",
    piece: str = "",
    morceau: int | None = None,
    delai_s: int = 300,
) -> ChatFn:
    from .cursor_chat import _default_run_fn, make_cursor_chat_fn
    from .plafond_appels import make_compteur

    compteur = compteur or make_compteur()
    run_fn = compteur.envelopper_run_fn(
        _default_run_fn, dossier=dossier, piece=piece, morceau=morceau, nature=nature
    )
    return make_cursor_chat_fn(run_fn=run_fn, timeout_s=delai_s)


def extraire_piece(
    piece: str,
    piece_text: str,
    *,
    chat_fn: ChatFn | None = None,
    compteur=None,
    dossier: str = "moreau",
) -> dict[str, Any]:
    invite = _construire_invite_extraction(piece, piece_numerotee(piece_text))
    chat = chat_fn or _chat_par_defaut(
        "prototype-situations", compteur=compteur, dossier=dossier, piece=piece
    )
    brut, _meta = chat("", invite)
    payload = _objet_json(brut, "situations")
    return rattacher_lettres(payload, piece, piece_text)


def extraire_piece_par_morceaux(
    piece: str,
    piece_text: str,
    *,
    max_car: int = 8000,
    chat_fn: ChatFn | None = None,
    compteur=None,
    dossier: str = "moreau",
    bruts_dir: Path | str | None = None,
    formes_exactes: bool = False,
    delai_s: int = 300,
    regle_tableaux: bool = False,
) -> dict[str, Any]:
    morceaux = decouper_piece(piece_text, max_car=max_car)
    vide = {"lignes_de_donnees": 0, "citees": 0, "non_citees": []}
    if not morceaux:
        rattache = rattacher_lettres({"situations": []}, piece, piece_text)
        rattache["couverture"] = dict(vide)
        rattache["morceaux"] = []
        return rattache
    lignes = piece_text.splitlines()
    toutes_donnees = lignes_de_donnees_tableau(piece, piece_text)
    rejets: list[str] = []
    situations: list[dict[str, Any]] = []
    details_morceaux: list[dict[str, Any]] = []
    bruts_path = Path(bruts_dir) if bruts_dir is not None else None
    if bruts_path is not None:
        bruts_path.mkdir(parents=True, exist_ok=True)

    for rang, (ligne_debut, ligne_fin) in enumerate(morceaux, start=1):
        brut_file = (
            bruts_path / f"{piece}.m{rang}.brut.txt" if bruts_path is not None else None
        )
        entetes = _entetes_rappel_morceau(
            lignes, ligne_debut=ligne_debut, ligne_fin=ligne_fin, piece=piece
        )
        numeros_rappel = {numero for numero, _texte in entetes}
        recu = False
        brut: str | None = None
        if brut_file is not None and brut_file.is_file():
            candidat = brut_file.read_text(encoding="utf-8")
            if _reponse_lisible(candidat):
                brut = candidat
            else:
                ecarter_reponse_brute(
                    brut_file, piece=piece, rang=rang, bruts_dir=bruts_path
                )
        if brut is None:
            invite = _construire_invite_morceau(
                piece,
                piece_text,
                rang,
                len(morceaux),
                ligne_debut,
                ligne_fin,
                entetes=entetes,
                max_car=max_car,
                formes_exactes=formes_exactes,
                regle_tableaux=regle_tableaux,
            )
            if chat_fn is not None:
                chat = chat_fn
            else:
                chat = _chat_par_defaut(
                    "prototype-situations",
                    compteur=compteur,
                    dossier=dossier,
                    piece=piece,
                    morceau=rang,
                    delai_s=delai_s,
                )
            brut, _meta = chat("", invite)
            recu = True
            if brut_file is not None:
                brut_file.write_text(brut, encoding="utf-8")
        try:
            payload = _objet_json(brut, "situations")
        except ValueError:
            if recu and bruts_path is not None and brut_file is not None and brut_file.is_file():
                ecarter_reponse_brute(
                    brut_file, piece=piece, rang=rang, bruts_dir=bruts_path
                )
            raise
        morceau_situations = _filtrer_lignes_hors_morceau(
            payload.get("situations") or [],
            ligne_debut=ligne_debut,
            ligne_fin=ligne_fin,
            rappel=numeros_rappel,
            rejets=rejets,
        )
        morceau_situations = _prefixer_ids_situations(morceau_situations, rang)
        situations.extend(morceau_situations)
        donnees_morceau = [n for n in toutes_donnees if ligne_debut <= n <= ligne_fin]
        details_morceaux.append(
            {
                "rang": rang,
                "ligne_debut": ligne_debut,
                "ligne_fin": ligne_fin,
                "situations": morceau_situations,
                "couverture": couverture_des_lignes(donnees_morceau, morceau_situations),
            }
        )

    couverture_piece = {
        "lignes_de_donnees": sum(item["couverture"]["lignes_de_donnees"] for item in details_morceaux),
        "citees": sum(item["couverture"]["citees"] for item in details_morceaux),
        "non_citees": [n for item in details_morceaux for n in item["couverture"]["non_citees"]],
    }
    payload = {
        "situations": situations,
        "morceaux": details_morceaux,
        "rejets": rejets,
        "couverture": couverture_piece,
    }
    rattache = rattacher_lettres(payload, piece, piece_text)
    rattache["rejets"] = rejets + list(rattache.get("rejets") or [])
    rattache["morceaux"] = details_morceaux
    rattache["couverture"] = couverture_piece
    return rattache


def charger_par_piece(par_piece_dir: Path) -> dict[str, dict[str, Any]]:
    situations: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(par_piece_dir).glob("*.situations.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        piece = payload["piece"]
        for situation in payload["situations"]:
            situations[f"{piece}#{situation['id']}"] = {**situation, "piece": piece}
    return situations


def _couper_resume(texte: str, resume_max: int) -> str:
    if len(texte) <= resume_max:
        return texte
    coupe = texte[:resume_max]
    espace = coupe.rfind(" ")
    if espace == -1:
        return coupe + "…"
    return coupe[:espace] + "…"


def ligne_descripteur(
    ref: str,
    situation: Mapping[str, Any],
    *,
    dates_max: int | None = None,
    resume_max: int | None = None,
) -> str:
    dates_brutes = [
        moment.get("iso") or moment.get("quand_texte") or ""
        for moment in situation.get("chronologie") or []
    ]
    if dates_max is not None and len(dates_brutes) > dates_max:
        n_premieres = max(dates_max - 1, 0)
        dates = ", ".join([*dates_brutes[:n_premieres], "…", dates_brutes[-1]])
        dates = f"{dates} ({len(dates_brutes)} dates)"
    else:
        dates = ", ".join(dates_brutes)
    acteurs = ", ".join(acteur["nom"] for acteur in situation.get("acteurs") or [])
    resume = situation.get("resume_neutre", "")
    if resume_max is not None:
        resume = _couper_resume(resume, resume_max)
    return (
        f"{ref} | {situation['nature']} | {situation['titre']} | {dates} | "
        f"{acteurs} | {resume}"
    )


def construire_invite_rapprochement(
    situations: Mapping[str, dict[str, Any]],
    *,
    dates_max: int | None = None,
    resume_max: int | None = None,
    alias: bool = False,
) -> tuple[str, dict[str, str]]:
    table: dict[str, str] = {}
    lignes: list[str] = []
    for index, (ref, situation) in enumerate(situations.items(), start=1):
        etiquette = ref
        if alias:
            etiquette = f"r{index:03d}"
            table[etiquette] = ref
        lignes.append(
            ligne_descripteur(
                etiquette, situation, dates_max=dates_max, resume_max=resume_max
            )
        )
    invite = "\n".join(lignes) + "\n\n" + INVITE_RAPPROCHEMENT
    return invite, table


def _chemin_table_alias(brut_path: Path) -> Path:
    return Path(str(brut_path) + ".alias.json")


def _ecrire_table_alias(path: Path, table: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(table), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _exiger_table_alias_concordante(path: Path, table: Mapping[str, str]) -> None:
    if not path.is_file():
        _ecrire_table_alias(path, table)
        return
    disque = json.loads(path.read_text(encoding="utf-8"))
    if disque != dict(table):
        raise ValueError(
            f"table d'alias sur disque différente de la table recalculée ({path})"
        )


def _retraduire_references_alias(
    payload: Mapping[str, Any],
    table: Mapping[str, str],
) -> tuple[dict[str, Any], list[str]]:
    inconnus: list[str] = []
    groupes: list[dict[str, Any]] = []
    for groupe in payload.get("groupes") or []:
        refs: list[str] = []
        for ref in groupe.get("references") or []:
            if ref in table:
                refs.append(table[ref])
            else:
                inconnus.append(ref)
        copie = dict(groupe)
        copie["references"] = refs
        groupes.append(copie)
    return {"groupes": groupes}, inconnus


def rapprocher(
    situations: Mapping[str, dict[str, Any]],
    *,
    chat_fn: ChatFn | None = None,
    compteur=None,
    dossier: str = "moreau",
    delai_s: int = 300,
    brut_path: Path | None = None,
    dates_max: int | None = None,
    resume_max: int | None = None,
    alias: bool = False,
    octets_max: int = OCTETS_MAX_INVITE,
) -> dict[str, Any]:
    invite, table = construire_invite_rapprochement(
        situations, dates_max=dates_max, resume_max=resume_max, alias=alias
    )
    brut: str | None = None
    fichier_brut = Path(brut_path) if brut_path is not None else None
    alias_path = _chemin_table_alias(fichier_brut) if fichier_brut is not None else None
    if fichier_brut is not None and fichier_brut.is_file():
        candidat = fichier_brut.read_text(encoding="utf-8")
        if _reponse_rapprochement_lisible(candidat):
            if alias:
                assert alias_path is not None
                _exiger_table_alias_concordante(alias_path, table)
            brut = candidat
        else:
            _ecarter_brut_fichier(fichier_brut)
    if brut is None:
        taille = len(invite.encode("utf-8"))
        if taille > octets_max:
            raise ValueError(
                f"invite de rapprochement trop grande : {taille} octets "
                f"(plafond {octets_max})"
            )
        chat = chat_fn or _chat_par_defaut(
            "prototype-rapprochement",
            compteur=compteur,
            dossier=dossier,
            delai_s=delai_s,
        )
        brut, _meta = chat("", invite)
        if fichier_brut is not None:
            fichier_brut.parent.mkdir(parents=True, exist_ok=True)
            fichier_brut.write_text(brut, encoding="utf-8")
            if alias:
                assert alias_path is not None
                _ecrire_table_alias(alias_path, table)
    payload = _objet_json(brut, "groupes")
    inconnues: list[str] = []
    if alias:
        payload, inconnues = _retraduire_references_alias(payload, table)
    groupes: list[dict[str, Any]] = []
    vues: set[str] = set()
    for groupe in payload["groupes"]:
        refs = []
        for ref in groupe.get("references") or []:
            if ref not in situations:
                inconnues.append(ref)
            elif ref in vues:
                inconnues.append(f"{ref} (déjà dans un autre groupe)")
            else:
                refs.append(ref)
                vues.add(ref)
        if len(refs) >= 2:
            groupes.append({"titre": groupe.get("titre", ""), "references": refs})
    resultat: dict[str, Any] = {"groupes": groupes, "references_ecartees": inconnues}
    if alias:
        resultat["alias"] = table
    return resultat


def fusionner(
    situations: Mapping[str, dict[str, Any]],
    groupes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Réunit sans réécrire : chaque élément garde sa pièce d'origine."""
    groupes_travail = [dict(groupe) for groupe in groupes]
    rapprochees = {
        ref for groupe in groupes_travail for ref in groupe["references"]
    }
    for ref, situation in situations.items():
        if ref not in rapprochees:
            groupes_travail.append(
                {"titre": situation["titre"], "references": [ref]}
            )
    fusionnees: list[dict[str, Any]] = []
    for numero, groupe in enumerate(groupes_travail, start=1):
        membres = [situations[ref] for ref in groupe["references"]]
        acteurs: dict[str, dict[str, Any]] = {}
        for membre in membres:
            for acteur in membre.get("acteurs") or []:
                cle = re.sub(r"[^a-z]", "", acteur["nom"].lower())
                fiche = acteurs.setdefault(
                    cle,
                    {
                        "nom": acteur["nom"],
                        "type_entite": acteur["type_entite"],
                        "roles": [],
                    },
                )
                role = {
                    "role": acteur["role"],
                    "piece": membre["piece"],
                    "hors_catalogue": bool(acteur.get("role_hors_catalogue")),
                }
                if role not in fiche["roles"]:
                    fiche["roles"].append(role)
        chronologie = [
            {**moment, "piece": membre["piece"]}
            for membre in membres
            for moment in membre.get("chronologie") or []
        ]
        chronologie.sort(
            key=lambda moment: (
                moment.get("iso") or "9999",
                moment["piece"],
                moment.get("ligne_debut") or 0,
            )
        )
        fusionnees.append(
            {
                "id": f"g{numero}",
                "titre": groupe["titre"],
                "natures": sorted({membre["nature"] for membre in membres}),
                "pieces": sorted({membre["piece"] for membre in membres}),
                "sources": list(groupe["references"]),
                "resumes_par_piece": [
                    {
                        "piece": membre["piece"],
                        "resume": membre.get("resume_neutre", ""),
                    }
                    for membre in membres
                ],
                "acteurs": list(acteurs.values()),
                "chronologie": chronologie,
                "perceptions": [
                    {**perception, "piece": membre["piece"]}
                    for membre in membres
                    for perception in membre.get("perceptions") or []
                ],
                "points_de_desaccord": [
                    {"piece": membre["piece"], "point": point}
                    for membre in membres
                    for point in membre.get("points_de_desaccord") or []
                ],
                "non_etabli": [
                    {"piece": membre["piece"], "point": point}
                    for membre in membres
                    for point in membre.get("non_etabli") or []
                ],
            }
        )
    return fusionnees


def numeroter_preuves_piece(situation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Un identifiant par passage distinct de la situation, dans l'ordre de lecture."""
    index: dict[tuple[int, int], str] = {}
    preuves: dict[str, dict[str, Any]] = {}
    for bloc in situation.get("perceptions") or []:
        for preuve in bloc.get("preuves") or []:
            cle = (int(preuve["ligne_debut"]), int(preuve["ligne_fin"]))
            if cle not in index:
                index[cle] = f"e{len(index) + 1}"
                preuves[index[cle]] = {
                    "ligne_debut": cle[0],
                    "ligne_fin": cle[1],
                    "lettre": preuve["lettre"],
                }
            preuve["id"] = index[cle]
    for moment in situation.get("chronologie") or []:
        if moment.get("lettre") is None:
            continue
        cle = (int(moment["ligne_debut"]), int(moment["ligne_fin"]))
        if cle not in index:
            index[cle] = f"e{len(index) + 1}"
            preuves[index[cle]] = {
                "ligne_debut": cle[0],
                "ligne_fin": cle[1],
                "lettre": moment["lettre"],
            }
        moment["id"] = index[cle]
    return preuves


def numeroter(situation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[tuple[str, int, int], str] = {}
    preuves: dict[str, dict[str, Any]] = {}

    def ident(piece: str, debut: int, fin: int, lettre: str) -> str:
        cle = (piece, debut, fin)
        if cle not in index:
            index[cle] = f"e{len(index) + 1}"
            preuves[index[cle]] = {
                "piece": piece,
                "ligne_debut": debut,
                "ligne_fin": fin,
                "lettre": lettre,
            }
        return index[cle]

    for perception in situation.get("perceptions") or []:
        for preuve in perception.get("preuves") or []:
            preuve["id"] = ident(
                perception["piece"],
                int(preuve["ligne_debut"]),
                int(preuve["ligne_fin"]),
                preuve["lettre"],
            )
    for moment in situation.get("chronologie") or []:
        if moment.get("lettre"):
            moment["id"] = ident(
                moment["piece"],
                int(moment["ligne_debut"]),
                int(moment["ligne_fin"]),
                moment["lettre"],
            )
    return preuves


def rendu_agent(situation: Mapping[str, Any]) -> str:
    from .lecteur_situations_composites import rendu_agent as rendu

    return rendu(situation)


def controler_prose_v2(
    prose: str,
    preuves: Mapping[str, Mapping[str, Any]],
    piece_text: str,
) -> dict[str, Any]:
    piece = _reduit(piece_text)
    renvois = re.findall(r"\[(e\d+)\]", prose)
    sans_renvois = re.sub(r"\[e\d+\]", "", prose)
    nombres = set(re.findall(r"\d+(?:[,.]\d+)?", sans_renvois))
    phrases = [p for p in re.split(r"(?<=[.!?])\s+", sans_renvois.strip()) if p]
    return {
        "renvois": len(renvois),
        "renvois_inconnus": sorted({r for r in renvois if r not in preuves}),
        "guillemets": len(re.findall(r"[«»\"“”]", prose)),
        "nombres_absents_de_la_piece": sorted(n for n in nombres if n not in piece),
        "phrases": len(phrases),
        "parle_de_lui": bool(re.match(r"\s*(Je |J'|Voici|D'accord)", prose)),
    }


def controler(
    prose: str,
    preuves: Mapping[str, Mapping[str, Any]],
    pieces: Sequence[str],
    *,
    piece_texts: Mapping[str, str],
) -> dict[str, Any]:
    corpus = " ".join(_reduit(piece_texts[piece]) for piece in pieces)
    renvois = re.findall(r"\[(e\d+)\]", prose)
    nu = re.sub(r"\[e\d+\]", "", prose)
    nombres = set(re.findall(r"\d+(?:[,.]\d+)?", nu))
    return {
        "renvois": len(renvois),
        "renvois_inconnus": sorted({r for r in renvois if r not in preuves}),
        "guillemets": len(re.findall(r"[«»\"“”]", prose)),
        "nombres_absents_des_pieces": sorted(n for n in nombres if n not in corpus),
        "phrases": len([p for p in re.split(r"(?<=[.!?])\s+", nu.strip()) if p]),
        "parle_de_lui": bool(re.match(r"\s*(Je |J'|Voici|D'accord)", prose)),
    }


def rendre_prose(
    situation: dict[str, Any],
    *,
    chat_fn: ChatFn | None = None,
    compteur=None,
    dossier: str = "moreau",
    piece_texts: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any]]:
    preuves = numeroter(situation)
    chat = chat_fn or _chat_par_defaut(
        "prototype-prose-fusion", compteur=compteur, dossier=dossier
    )
    matiere = json.dumps(situation, ensure_ascii=False, indent=1)
    prose, _meta = chat("", f"{INVITE_PROSE}\n\nSituation :\n{matiere}")
    prose = prose.strip()
    texts = piece_texts or {}
    controle = controler(prose, preuves, situation.get("pieces") or [], piece_texts=texts)
    return prose, preuves, controle


def invite_reconciliation(portee: str = "dossier") -> str:
    """Invite de réconciliation. Pour `dossier`, copie mot pour mot du prototype."""
    if portee == "dossier":
        return INVITE_RECONCILIATION
    texte = INVITE_RECONCILIATION.replace(CHAMP_PORTEE_MODELE, f"portee_{portee}")
    texte = texte.replace("DOSSIER", portee.upper())
    texte = texte.replace("dossier", portee)
    return texte


def inventaire_perimetre(
    par_piece_dir: Path,
    *,
    piece_texts: Mapping[str, str] | None = None,
    pieces_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Ce que le périmètre contient : nom, intitulé lu dans la pièce, situations relevées."""
    lignes: list[dict[str, Any]] = []
    for path in sorted(Path(par_piece_dir).glob("*.situations.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        piece = payload["piece"]
        if piece_texts is not None and piece in piece_texts:
            texte = piece_texts[piece].splitlines()
        elif pieces_dir is not None:
            texte = (Path(pieces_dir) / piece).read_text(encoding="utf-8").splitlines()
        else:
            raise ValueError(f"texte de pièce absent : {piece}")
        entete = [
            ligne.strip()
            for ligne in texte
            if ligne.strip() and not set(ligne.strip()) <= set("-_=|: ")
        ][:9]
        lignes.append(
            {
                "fichier": piece,
                "premieres_lignes": entete,
                "situations": [situation["titre"] for situation in payload["situations"]],
            }
        )
    return lignes


def remarques_numerotees(
    situations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for situation in situations:
        for rang, item in enumerate(situation.get("non_etabli") or [], start=1):
            if isinstance(item, Mapping):
                piece = str(item.get("piece") or "")
                point = str(item.get("point") or "")
            else:
                piece = ""
                point = str(item)
            out.append(
                {
                    "id": f"{situation['id']}.n{rang}",
                    "situation": situation["titre"],
                    "relevee_dans": piece,
                    "remarque": point,
                }
            )
    return out


def reconcilier(
    *,
    inventaire: Sequence[Mapping[str, Any]] | None = None,
    par_piece_dir: Path | str | None = None,
    piece_texts: Mapping[str, str] | None = None,
    pieces_dir: Path | str | None = None,
    fusion: Mapping[str, Any] | None = None,
    situations: Sequence[Mapping[str, Any]] | None = None,
    chat_fn: ChatFn | None = None,
    compteur=None,
    dossier: str = "moreau",
    portee: str = "dossier",
    lot: int = LOT_RECONCILIATION,
    brut_dir: Path | str | None = None,
    delai_s: int = 300,
    octets_max: int = OCTETS_MAX_INVITE,
) -> dict[str, Any]:
    """Passe de réconciliation : inventaire + remarques, aucun contenu de pièce entier.

    Chat, chemins et compteur injectables. Aucun appel réel depuis les tests.
    """
    if inventaire is None:
        if par_piece_dir is None:
            raise ValueError("inventaire ou par_piece_dir requis")
        inventaire = inventaire_perimetre(
            Path(par_piece_dir), piece_texts=piece_texts, pieces_dir=pieces_dir
        )
    if situations is None:
        if fusion is None:
            raise ValueError("fusion ou situations requis")
        situations = list(fusion["situations"])
    connus = {item["fichier"] for item in inventaire}
    toutes = remarques_numerotees(situations)
    invite = invite_reconciliation(portee)
    champ_portee = f"portee_{portee}"
    chat: ChatFn | None = chat_fn
    verdicts: dict[str, dict[str, Any]] = {}
    ecartes: list[str] = []
    bruts_path = Path(brut_dir) if brut_dir is not None else None
    for debut in range(0, len(toutes), lot):
        lot_items = toutes[debut : debut + lot]
        matiere = json.dumps(
            {"inventaire": list(inventaire), "remarques": lot_items},
            ensure_ascii=False,
            indent=1,
        )
        brut: str | None = None
        brut_file = (
            bruts_path / f"reconciliation.lot{debut // lot + 1}.brut.txt"
            if bruts_path is not None
            else None
        )
        if brut_file is not None and brut_file.is_file():
            candidat = brut_file.read_text(encoding="utf-8")
            if _reponse_reconciliation_lisible(candidat):
                brut = candidat
            else:
                _ecarter_brut_fichier(brut_file)
        if brut is None:
            texte_invite = f"{invite}\n\n{matiere}"
            taille = len(texte_invite.encode("utf-8"))
            if taille > octets_max:
                raise ValueError(
                    f"invite de réconciliation trop grande : {taille} octets "
                    f"(plafond {octets_max})"
                )
            if chat is None:
                chat = _chat_par_defaut(
                    "prototype-reconciliation",
                    compteur=compteur,
                    dossier=dossier,
                    delai_s=delai_s,
                )
            brut, _meta = chat("", texte_invite)
            if brut_file is not None:
                brut_file.parent.mkdir(parents=True, exist_ok=True)
                brut_file.write_text(brut, encoding="utf-8")
        attendus = {item["id"] for item in lot_items}
        for verdict in _objet_json(brut, "remarques")["remarques"]:
            ident = str(verdict.get("id"))
            if ident not in attendus:
                ecartes.append(f"{ident} : identifiant inconnu")
                continue
            pieces_brutes = [p for p in (verdict.get("pieces") or [])]
            pieces = [p for p in pieces_brutes if p in connus]
            inconnues = [p for p in pieces_brutes if p not in connus]
            if inconnues:
                ecartes.append(f"{ident} : pièce inconnue {inconnues}")
            statut = verdict.get(champ_portee, verdict.get(CHAMP_PORTEE_MODELE))
            if statut == "etabli_par_une_piece" and not pieces:
                statut = "incertain"
                ecartes.append(
                    f"{ident} : « établi » sans pièce connue, ramené à incertain"
                )
            verdicts[ident] = {
                "portee": portee,
                "statut": statut,
                "pieces": pieces,
                "importance": verdict.get("importance"),
                "motif": verdict.get("motif"),
            }
    manquants = sorted({item["id"] for item in toutes} - set(verdicts))
    return {
        "portee": portee,
        "verdicts": verdicts,
        "ecartes": ecartes,
        "sans_verdict": manquants,
    }


def appliquer_reconciliation(
    fusion: Mapping[str, Any],
    verdicts_payload: Mapping[str, Any],
    *,
    portee: str | None = None,
) -> dict[str, Any]:
    """Applique les verdicts sans effacer la remarque d'origine : elle change de rubrique."""
    del portee
    sortie = json.loads(json.dumps(fusion))
    verdicts = verdicts_payload.get("verdicts") or {}
    for situation in sortie["situations"]:
        restants: list[dict[str, Any]] = []
        etablis: list[dict[str, Any]] = []
        for rang, item in enumerate(situation.get("non_etabli") or [], start=1):
            origine = dict(item) if isinstance(item, Mapping) else {"point": item}
            verdict = verdicts.get(f"{situation['id']}.n{rang}")
            enrichi = {
                **origine,
                "releve_a_la_portee": "piece",
                **(verdict or {"statut": "sans_verdict"}),
            }
            if verdict and verdict.get("statut") == "etabli_par_une_piece":
                etablis.append(enrichi)
            else:
                restants.append(enrichi)
        restants.sort(key=lambda x: ORDRE_IMPORTANCE.get(x.get("importance"), 3))
        situation["non_etabli_dans_la_piece_mais_etabli_dans_le_dossier"] = etablis
        situation["non_etabli"] = restants
    return sortie
