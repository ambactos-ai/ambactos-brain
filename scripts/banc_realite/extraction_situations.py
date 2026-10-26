# SPDX-License-Identifier: AGPL-3.0-only
"""Extraction de faits par morceaux pour le constructeur MRFS (ADR 0025 §3, 5, 6, 10)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .constructeur_situations import (
    Rejection,
    ValidatedFact,
    VALID_NATURES,
    extract_piece_lines,
    fact_ref,
    load_groupes_payload,
    slice_lines,
    validate_fact,
)
from .cursor_chat import ChatFn, make_cursor_chat_fn
from .dossiers import BANC_VAR, dossier_courant
from .plafond_appels import CompteurAppels, PlafondAtteint, make_compteur

MAX_MORCEAU_CHARS = 8000
RAPPROCHEMENT_OVERLAP_RATIO = 0.20
DEFAULT_CONSTRUCTEUR_ROOT = (
    BANC_VAR / 'dossiers' / 'mrfs' / 'constructeur'
)

EXTRACTION_PROMPT = """Tu lis un extrait d'une pièce d'un dossier. Chaque ligne est précédée de son numéro.
Relève CHAQUE fait énoncé dans l'extrait : événement, état, relation, processus, règle ou
clause, définition. Un fait par énoncé. N'en omets aucun, n'en invente aucun.

Pour chaque fait, rends :
- "id" : "f1", "f2", … dans l'ordre de lecture ;
- "citation" : le passage EXACT de la pièce, copié caractère pour caractère, sans le
  préfixe de numéro de ligne, sans correction, sans traduction, sans points de suspension.
  Pour une ligne de tableau ou de CSV, la citation est la ligne entière ;
- "ligne_debut", "ligne_fin" : numéros de ligne de la citation, bornes incluses ;
- "nature" : "evenement" | "processus" | "etat" | "relation" | "regle" | "definition" ;
- "enonciateur" : la personne ou l'organisation qui énonce le fait, telle qu'écrite dans la
  pièce ; si la pièce ne le dit pas : "auteur de la pièce" ;
- "date" : {"texte": la date telle qu'écrite, "iso": "AAAA-MM-JJ" si elle est complète, sinon
  null}, ou null si le fait n'est pas daté ;
- "descripteur" : une ligne, qui / quoi / quand, dans la langue de la pièce.

Réponds par un seul objet JSON, sans texte autour : {"faits": [ … ]}"""

REPRISE_JSON = (
    "Ta réponse précédente n'était pas un JSON valide au format demandé."
)

RAPPROCHEMENT_PROMPT = """Voici des faits relevés dans plusieurs pièces d'un même dossier, un par ligne :
référence | énonciateur | date | descripteur.
Regroupe les références qui parlent du MÊME événement, état ou relation du monde — y compris
quand elles le décrivent différemment ou se contredisent sur un montant, une date ou une
qualification. Ne regroupe pas deux faits seulement parce qu'ils se ressemblent.
Réponds par un seul objet JSON, sans texte autour :
{"groupes": [["ref1", "ref2"], …]}
N'y mets que les groupes d'au moins deux références. N'invente aucune référence."""

DOUBLON = "identifiant en double dans la pièce"

NATURES_RAPPROCHEMENT = frozenset({"evenement", "processus", "etat", "relation"})


@dataclass(frozen=True, slots=True)
class Morceau:
    index: int
    ligne_debut: int
    ligne_fin: int
    lignes: tuple[str, ...]

    @property
    def texte_brut(self) -> str:
        return "\n".join(self.lignes)


@dataclass(slots=True)
class MorceauEtat:
    statut: str  # fait | echec | a_faire
    fichier: str = ""
    faits_rendus: int = 0
    faits_valides: int = 0
    faits_rejetes: int = 0
    faits_doublons: int = 0
    lettres_recalees: int = 0
    lignes_corrigees: int = 0


class ChatFactory(Protocol):
    def __call__(self, nature: str) -> ChatFn: ...


def constructeur_dossier_root(
    dossier: str, *, constructeur_root: Path | None = None
) -> Path:
    root = constructeur_root or DEFAULT_CONSTRUCTEUR_ROOT
    return root / dossier


def faits_dir(dossier: str, *, constructeur_root: Path | None = None) -> Path:
    return constructeur_dossier_root(dossier, constructeur_root=constructeur_root) / "faits"


def faits_path(
    dossier: str, piece: str, morceau: int, *, constructeur_root: Path | None = None
) -> Path:
    stem = Path(piece).name
    return faits_dir(dossier, constructeur_root=constructeur_root) / f"{stem}.m{morceau:02d}.json"


def brut_path(
    dossier: str, piece: str, morceau: int, *, constructeur_root: Path | None = None
) -> Path:
    return faits_path(
        dossier, piece, morceau, constructeur_root=constructeur_root
    ).with_suffix(".brut.txt")


def etat_path(dossier: str, *, constructeur_root: Path | None = None) -> Path:
    return constructeur_dossier_root(
        dossier, constructeur_root=constructeur_root
    ) / "etat-extraction.json"


def groupes_path(dossier: str, *, constructeur_root: Path | None = None) -> Path:
    return constructeur_dossier_root(
        dossier, constructeur_root=constructeur_root
    ) / "groupes.json"


def journal_pieces_path(
    dossier: str, *, constructeur_root: Path | None = None
) -> Path:
    return constructeur_dossier_root(
        dossier, constructeur_root=constructeur_root
    ) / "journal-pieces.jsonl"


def decouper_piece_extraction(piece_text: str) -> list[Morceau]:
    """Découpe sans recouvrement, au plus 8000 caractères, en fin de ligne."""
    lines = extract_piece_lines(piece_text)
    if not lines:
        return [Morceau(1, 1, 1, ("",))] if piece_text == "" else []

    morceaux: list[Morceau] = []
    current: list[str] = []
    current_chars = 0
    start_line = 1
    index = 1

    def flush(end_line: int) -> None:
        nonlocal index, current, current_chars, start_line
        if not current:
            return
        morceaux.append(
            Morceau(
                index=index,
                ligne_debut=start_line,
                ligne_fin=end_line,
                lignes=tuple(current),
            )
        )
        index += 1
        current = []
        current_chars = 0

    for line_no, line in enumerate(lines, start=1):
        line_len = len(line)
        if line_len > MAX_MORCEAU_CHARS:
            flush(line_no - 1)
            morceaux.append(Morceau(index, line_no, line_no, (line,)))
            index += 1
            start_line = line_no + 1
            continue

        extra = line_len + (1 if current else 0)
        if current and current_chars + extra > MAX_MORCEAU_CHARS:
            flush(line_no - 1)
            start_line = line_no
            current = [line]
            current_chars = line_len
        else:
            if not current:
                start_line = line_no
            current.append(line)
            current_chars += extra

    if current:
        flush(len(lines))
    return morceaux


def formater_morceau(morceau: Morceau) -> str:
    return "\n".join(
        f"{num:4d}| {line}" for num, line in enumerate(morceau.lignes, start=morceau.ligne_debut)
    )


def construire_invite_extraction(piece: str, morceau: Morceau, total_morceaux: int) -> str:
    header = f"Pièce : {piece} — morceau {morceau.index}/{total_morceaux}\n\n"
    return header + formater_morceau(morceau) + "\n\n" + EXTRACTION_PROMPT


def _strip_markdown_json(text: str) -> str:
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", stripped, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return stripped


def _objet_json(text: str, cle: str) -> dict[str, Any]:
    """Le premier objet JSON de la réponse qui porte la clé attendue.

    Le modèle écrit parfois une phrase avant ou après son JSON. Ce texte autour est
    ignoré ; le contenu de l'objet, lui, n'est jamais retouché.
    """
    raw = _strip_markdown_json(text)
    decoder = json.JSONDecoder()
    position = raw.find("{")
    derniere_erreur: Exception | None = None
    while position >= 0:
        try:
            payload, _end = decoder.raw_decode(raw, position)
        except json.JSONDecodeError as exc:
            derniere_erreur = exc
        else:
            if isinstance(payload, dict) and cle in payload:
                return payload
        position = raw.find("{", position + 1)
    raise ValueError(f"aucun objet JSON portant « {cle} » : {derniere_erreur}")


def _validate_date(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("date invalide")
    texte = value.get("texte")
    iso = value.get("iso")
    if texte is not None and not isinstance(texte, str):
        raise ValueError("date.texte invalide")
    if iso is not None and not isinstance(iso, str):
        raise ValueError("date.iso invalide")
    return {"texte": texte, "iso": iso}


def _fait_bien_forme(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("fait non objet")
    fact_id = item.get("id")
    citation = item.get("citation")
    nature = item.get("nature")
    enonciateur = item.get("enonciateur")
    descripteur = item.get("descripteur")
    if not isinstance(fact_id, str) or not fact_id.strip():
        raise ValueError("id manquant")
    if not isinstance(citation, str) or not citation:
        raise ValueError("citation vide")
    if not isinstance(nature, str) or nature.strip().lower() not in VALID_NATURES:
        raise ValueError(f"nature inconnue : {nature!r}")
    if not isinstance(enonciateur, str) or not enonciateur.strip():
        raise ValueError("énonciateur manquant")
    if not isinstance(descripteur, str) or not descripteur.strip():
        raise ValueError("descripteur manquant")
    try:
        start = int(item.get("ligne_debut"))
        end = int(item.get("ligne_fin"))
    except (TypeError, ValueError) as exc:
        raise ValueError("lignes invalides") from exc
    return {
        "id": fact_id.strip(),
        "citation": citation,
        "ligne_debut": start,
        "ligne_fin": end,
        "nature": nature.strip().lower(),
        "enonciateur": enonciateur.strip(),
        "date": _validate_date(item.get("date")),
        "descripteur": descripteur.strip(),
    }


def parse_reponse_extraction(
    text: str, *, mal_formes: list[str] | None = None
) -> list[dict[str, Any]]:
    """Faits de la réponse. Sans `mal_formes`, un fait mal formé invalide la réponse ;
    avec, il est écarté seul et son motif y est consigné."""
    payload = _objet_json(text, "faits")
    faits = payload.get("faits")
    if not isinstance(faits, list):
        raise ValueError("faits doit être une liste")
    parsed: list[dict[str, Any]] = []
    for item in faits:
        try:
            parsed.append(_fait_bien_forme(item))
        except ValueError as exc:
            if mal_formes is None:
                raise
            mal_formes.append(str(exc))
    return parsed


def identifiant_fait(morceau: int, fact: dict[str, Any]) -> str:
    """Identifiant tiré du contenu : une nouvelle extraction du même morceau rend les
    mêmes identifiants aux mêmes faits, et de nouveaux aux faits qui ont changé. Un
    identifiant de rang (`f1`, `f2`…) rattacherait une citation nouvelle à une
    proposition déjà publiée avec une autre lettre."""
    base = f"{fact['ligne_debut']}:{fact['ligne_fin']}:{fact['citation']}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:10]
    return f"m{morceau}-{digest}"


def _espaces_reduits(text: str) -> tuple[str, list[int]]:
    """Le texte avec ses suites d'espaces réduites à une espace, et, pour chaque
    caractère gardé, sa position dans le texte d'origine."""
    chars: list[str] = []
    index: list[int] = []
    for position, char in enumerate(text):
        if char.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                index.append(position)
        else:
            chars.append(char)
            index.append(position)
    if chars and chars[-1] == " ":
        chars.pop()
        index.pop()
    return "".join(chars), index


def _ligne(piece_text: str, position: int) -> int:
    """Numéro de ligne d'une position, dans la numérotation montrée au modèle.

    Les lignes sont celles de `str.splitlines` : un saut de page compte comme une fin de
    ligne. Compter les seuls retours à la ligne décalerait tout ce qui suit un saut de page.
    """
    fin = 0
    for numero, ligne in enumerate(piece_text.splitlines(keepends=True), start=1):
        fin += len(ligne)
        if position < fin:
            return numero
    return max(1, len(piece_text.splitlines()))


def recaler_lettre(
    fact: dict[str, Any], *, piece_text: str
) -> tuple[dict[str, Any], bool]:
    """ADR 0025 §3, amendé : la lettre publiée est découpée dans la pièce par le code.

    Quand la citation rendue ne diffère de la pièce que par la mise en page (retours à
    la ligne, indentation), le passage est repéré sans tenir compte des espaces, puis
    recopié de la pièce, caractère pour caractère. Tout autre écart — une lettre, une
    apostrophe, un chiffre — laisse le fait en l'état, donc rejeté.
    """
    citation = fact["citation"]
    if citation in piece_text:
        return fact, False
    cible, _ = _espaces_reduits(citation)
    if not cible:
        return fact, False
    reduit, index = _espaces_reduits(piece_text)
    spans: list[tuple[int, int]] = []
    start = reduit.find(cible)
    while start >= 0:
        spans.append((index[start], index[start + len(cible) - 1] + 1))
        start = reduit.find(cible, start + 1)
    if len(spans) > 1:
        # Plusieurs passages identiques : seules les lignes annoncées départagent.
        spans = [
            span
            for span in spans
            if _ligne(piece_text, span[0]) <= fact["ligne_fin"]
            and _ligne(piece_text, span[1] - 1) >= fact["ligne_debut"]
        ]
    if len(spans) != 1:
        return fact, False
    debut, fin = spans[0]
    recale = dict(fact)
    recale["citation"] = piece_text[debut:fin]
    recale["ligne_debut"] = _ligne(piece_text, debut)
    recale["ligne_fin"] = _ligne(piece_text, fin - 1)
    recale["lettre_recalee"] = True
    return recale, True


def _occurrences_uniques(piece_text: str, citation: str) -> tuple[int, int] | None:
    first = piece_text.find(citation)
    if first < 0:
        return None
    second = piece_text.find(citation, first + len(citation))
    if second >= 0:
        return None
    return _ligne(piece_text, first), _ligne(piece_text, first + len(citation) - 1)


def corriger_lignes_fact(
    fact: dict[str, Any], *, piece: str, piece_text: str
) -> tuple[dict[str, Any], bool]:
    probe = validate_fact(
        {**fact, "dossier": "", "morceau": 1},
        piece=piece,
        piece_text=piece_text,
        group_map={},
    )
    if isinstance(probe, ValidatedFact):
        return fact, False
    if probe.reason not in {
        "citation hors des lignes annoncées",
        "citation absente de la pièce",
    }:
        return fact, False
    span = _occurrences_uniques(piece_text, fact["citation"])
    if span is None:
        return fact, False
    corrected = dict(fact)
    corrected["ligne_debut"], corrected["ligne_fin"] = span
    return corrected, True


def valider_faits_morceau(
    faits: list[dict[str, Any]],
    *,
    dossier: str,
    piece: str,
    morceau: int,
    piece_text: str,
) -> tuple[list[dict[str, Any]], list[Rejection], int]:
    valides: list[dict[str, Any]] = []
    rejets: list[Rejection] = []
    lignes_corrigees = 0
    seen: set[str] = set()
    for fact in faits:
        fact, _recale = recaler_lettre(fact, piece_text=piece_text)
        corrected, fixed = corriger_lignes_fact(fact, piece=piece, piece_text=piece_text)
        if fixed:
            lignes_corrigees += 1
        # Après la correction des lignes : l'identifiant porte les lignes retenues.
        corrected = {**corrected, "id": identifiant_fait(morceau, corrected)}
        payload = {**corrected, "dossier": dossier, "morceau": morceau}
        result = validate_fact(payload, piece=piece, piece_text=piece_text, group_map={})
        if isinstance(result, Rejection):
            rejets.append(result)
            continue
        if result.fact_id in seen:
            rejets.append(
                Rejection(result.fact_id, piece, DOUBLON)
            )
            continue
        seen.add(result.fact_id)
        valides.append(payload)
    return valides, rejets, lignes_corrigees


def ecrire_fichier_faits(
    path: Path,
    *,
    dossier: str,
    piece: str,
    morceau: int,
    faits: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dossier": dossier,
        "piece": piece,
        "morceau": morceau,
        "faits": faits,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def charger_etat(dossier: str, *, constructeur_root: Path | None = None) -> dict[str, Any]:
    path = etat_path(dossier, constructeur_root=constructeur_root)
    if not path.is_file():
        return {"dossier": dossier, "pieces": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"dossier": dossier, "pieces": {}}
    payload.setdefault("pieces", {})
    return payload


def sauver_etat(
    dossier: str, etat: dict[str, Any], *, constructeur_root: Path | None = None
) -> None:
    path = etat_path(dossier, constructeur_root=constructeur_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(etat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _morceau_etat(etat: dict[str, Any], piece: str, morceau: int) -> MorceauEtat:
    pieces = etat.setdefault("pieces", {})
    piece_state = pieces.setdefault(piece, {})
    morceaux = piece_state.setdefault("morceaux", {})
    key = str(morceau)
    raw = morceaux.get(key) or {}
    return MorceauEtat(
        statut=str(raw.get("statut") or "a_faire"),
        fichier=str(raw.get("fichier") or ""),
        faits_rendus=int(raw.get("faits_rendus") or 0),
        faits_valides=int(raw.get("faits_valides") or 0),
        faits_rejetes=int(raw.get("faits_rejetes") or 0),
        faits_doublons=int(raw.get("faits_doublons") or 0),
        lettres_recalees=int(raw.get("lettres_recalees") or 0),
        lignes_corrigees=int(raw.get("lignes_corrigees") or 0),
    )


def _maj_morceau_etat(
    etat: dict[str, Any],
    piece: str,
    morceau: int,
    morceau_etat: MorceauEtat,
) -> None:
    etat.setdefault("pieces", {}).setdefault(piece, {}).setdefault("morceaux", {})[
        str(morceau)
    ] = {
        "statut": morceau_etat.statut,
        "fichier": morceau_etat.fichier,
        "faits_rendus": morceau_etat.faits_rendus,
        "faits_valides": morceau_etat.faits_valides,
        "faits_rejetes": morceau_etat.faits_rejetes,
        "faits_doublons": morceau_etat.faits_doublons,
        "lettres_recalees": morceau_etat.lettres_recalees,
        "lignes_corrigees": morceau_etat.lignes_corrigees,
    }


def lister_pieces(dossier: str, selection: str) -> list[str]:
    desc = dossier_courant(dossier)
    all_pieces = sorted(path.name for path in desc.corpus_dir.iterdir() if path.is_file())
    if selection.strip().lower() == "toutes":
        return all_pieces
    chosen = [item.strip() for item in selection.split(",") if item.strip()]
    missing = [name for name in chosen if name not in all_pieces]
    if missing:
        raise FileNotFoundError(f"pièces absentes du corpus : {missing}")
    return sorted(chosen)


def estimer_a_blanc(
    dossier: str,
    pieces: list[str],
    *,
    constructeur_root: Path | None = None,
) -> dict[str, Any]:
    desc = dossier_courant(dossier)
    detail: list[dict[str, Any]] = []
    total_morceaux = 0
    total_chars = 0
    refs_rapp: list[str] = []
    for piece in pieces:
        text = (desc.corpus_dir / piece).read_text(encoding="utf-8")
        morceaux = decouper_piece_extraction(text)
        total_morceaux += len(morceaux)
        total_chars += len(text)
        detail.append(
            {
                "piece": piece,
                "caracteres": len(text),
                "morceaux": len(morceaux),
            }
        )
        faits_dir_path = faits_dir(dossier, constructeur_root=constructeur_root)
        if faits_dir_path.is_dir():
            for path in sorted(faits_dir_path.glob(f"{piece}.m*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                for fact in payload.get("faits") or []:
                    if str(fact.get("nature") or "") in NATURES_RAPPROCHEMENT:
                        refs_rapp.append(f"{piece}#{fact.get('id')}")
    if refs_rapp:
        lignes_rapp = [
            formater_ligne_rapprochement(ref, "", None, ref) for ref in refs_rapp
        ]
        lots_rapp = len(decouper_lots_rapprochement(refs_rapp, lignes_rapp))
    else:
        lots_rapp = 1 if total_morceaux else 0
    return {
        "dossier": dossier,
        "pieces": detail,
        "total_caracteres": total_chars,
        "total_morceaux": total_morceaux,
        "lots_rapprochement": lots_rapp,
        "appels_prevus": total_morceaux + lots_rapp,
    }


def _appeler_modele(
    chat: ChatFn,
    invite: str,
    *,
    reprise: bool = False,
) -> tuple[str, dict[str, Any]]:
    suffix = f"\n\n{REPRISE_JSON}" if reprise else ""
    return chat("", invite + suffix)


def extraire_morceau(
    *,
    dossier: str,
    piece: str,
    piece_text: str,
    morceau: Morceau,
    total_morceaux: int,
    chat_factory: ChatFactory,
    constructeur_root: Path | None = None,
    relire_brut: bool = False,
) -> MorceauEtat:
    invite = construire_invite_extraction(piece, morceau, total_morceaux)
    brut = ""
    parsed: list[dict[str, Any]] | None = None
    mal_formes: list[str] = []
    # Une réponse déjà payée et conservée est relue avant tout nouvel appel.
    conserve = brut_path(dossier, piece, morceau.index, constructeur_root=constructeur_root)
    if relire_brut and conserve.is_file():
        try:
            brut = conserve.read_text(encoding="utf-8")
            parsed = parse_reponse_extraction(brut, mal_formes=mal_formes)
        except ValueError:
            brut, parsed = "", None
    for attempt, (reprise, nature) in enumerate(
        ((False, "extraction"), (True, "reprise")) if parsed is None else ()
    ):
        chat = chat_factory(nature)
        try:
            brut, _meta = _appeler_modele(chat, invite, reprise=reprise)
            parsed = parse_reponse_extraction(brut, mal_formes=mal_formes)
            break
        except ValueError:
            # Chaque réponse payée est conservée, même refusée.
            refusee = brut_path(
                dossier, piece, morceau.index, constructeur_root=constructeur_root
            ).with_suffix(f".refus{attempt + 1}.txt")
            refusee.parent.mkdir(parents=True, exist_ok=True)
            refusee.write_text(brut, encoding="utf-8")
            if attempt == 1:
                path_brut = brut_path(
                    dossier, piece, morceau.index, constructeur_root=constructeur_root
                )
                path_brut.parent.mkdir(parents=True, exist_ok=True)
                path_brut.write_text(brut, encoding="utf-8")
                return MorceauEtat(statut="echec", fichier=str(path_brut))

    assert parsed is not None
    valides, rejets, lignes_corrigees = valider_faits_morceau(
        parsed,
        dossier=dossier,
        piece=piece,
        morceau=morceau.index,
        piece_text=piece_text,
    )
    out_path = faits_path(
        dossier, piece, morceau.index, constructeur_root=constructeur_root
    )
    ecrire_fichier_faits(
        out_path,
        dossier=dossier,
        piece=piece,
        morceau=morceau.index,
        faits=valides,
    )
    brut_path(
        dossier, piece, morceau.index, constructeur_root=constructeur_root
    ).write_text(brut, encoding="utf-8")
    return MorceauEtat(
        statut="fait",
        fichier=str(out_path),
        faits_rendus=len(parsed) + len(mal_formes),
        faits_valides=len(valides),
        # Un doublon est la même lettre rendue deux fois : rien n'est perdu.
        faits_rejetes=len(mal_formes)
        + sum(1 for rejet in rejets if rejet.reason != DOUBLON),
        faits_doublons=sum(1 for rejet in rejets if rejet.reason == DOUBLON),
        lettres_recalees=sum(1 for fait in valides if fait.get("lettre_recalee")),
        lignes_corrigees=lignes_corrigees,
    )


def extraire_piece(
    *,
    dossier: str,
    piece: str,
    chat_factory_builder: Callable[[str, int], ChatFactory],
    refaire: set[tuple[str, int]] | None = None,
    constructeur_root: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    desc = dossier_courant(dossier)
    piece_path = desc.corpus_dir / piece
    piece_text = piece_path.read_text(encoding="utf-8")
    morceaux = decouper_piece_extraction(piece_text)
    etat = charger_etat(dossier, constructeur_root=constructeur_root)
    incomplete = False
    for morceau in morceaux:
        courant = _morceau_etat(etat, piece, morceau.index)
        force = refaire is not None and (piece, morceau.index) in refaire
        if courant.statut == "fait" and not force:
            continue
        try:
            result = extraire_morceau(
                dossier=dossier,
                piece=piece,
                piece_text=piece_text,
                morceau=morceau,
                total_morceaux=len(morceaux),
                chat_factory=chat_factory_builder(piece, morceau.index),
                constructeur_root=constructeur_root,
                relire_brut=courant.statut == "echec" and not force,
            )
        except PlafondAtteint:
            raise
        if result.statut == "echec":
            incomplete = True
        _maj_morceau_etat(etat, piece, morceau.index, result)
        sauver_etat(dossier, etat, constructeur_root=constructeur_root)
    return etat, incomplete


class AppelInterdit(RuntimeError):
    """La revalidation relit les réponses conservées ; elle n'appelle jamais le modèle."""


def _chat_interdit(nature: str) -> ChatFn:
    raise AppelInterdit(f"appel de modèle interdit en revalidation ({nature})")


def revalider_dossier(
    dossier: str, *, constructeur_root: Path | None = None
) -> dict[str, Any]:
    """Rejoue la validation sur les réponses brutes conservées. Aucun appel de modèle."""
    desc = dossier_courant(dossier)
    etat = charger_etat(dossier, constructeur_root=constructeur_root)
    bilan = {"morceaux": 0, "valides_avant": 0, "valides_apres": 0, "lettres_recalees": 0, "rejets": 0}
    for piece in sorted(etat.get("pieces", {})):
        piece_text = (desc.corpus_dir / piece).read_text(encoding="utf-8")
        morceaux = {item.index: item for item in decouper_piece_extraction(piece_text)}
        for key in sorted(etat["pieces"][piece].get("morceaux", {}), key=int):
            morceau = morceaux.get(int(key))
            conserve = brut_path(dossier, piece, int(key), constructeur_root=constructeur_root)
            if morceau is None or not conserve.is_file():
                continue
            avant = _morceau_etat(etat, piece, int(key))
            apres = extraire_morceau(
                dossier=dossier,
                piece=piece,
                piece_text=piece_text,
                morceau=morceau,
                total_morceaux=len(morceaux),
                chat_factory=_chat_interdit,
                constructeur_root=constructeur_root,
                relire_brut=True,
            )
            _maj_morceau_etat(etat, piece, int(key), apres)
            bilan["morceaux"] += 1
            bilan["valides_avant"] += avant.faits_valides
            bilan["valides_apres"] += apres.faits_valides
            bilan["lettres_recalees"] += apres.lettres_recalees
            bilan["rejets"] += apres.faits_rejetes
    sauver_etat(dossier, etat, constructeur_root=constructeur_root)
    return bilan


def faits_paths_dossier(
    dossier: str, *, constructeur_root: Path | None = None
) -> list[Path]:
    directory = faits_dir(dossier, constructeur_root=constructeur_root)
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def _tri_rapprochement_key(
    ref: str, descripteur: str, date_iso: str | None
) -> tuple[str, tuple[int, ...], str]:
    numbers = tuple(int(item) for item in re.findall(r"\d+", descripteur))
    return (date_iso or "", numbers, descripteur)


def formater_ligne_rapprochement(
    ref: str, enonciateur: str, date: dict[str, Any] | None, descripteur: str
) -> str:
    date_text = ""
    if isinstance(date, dict):
        date_text = str(date.get("iso") or date.get("texte") or "")
    return f"{ref} | {enonciateur} | {date_text} | {descripteur}"


def decouper_lots_rapprochement(
    refs: list[str], lignes: list[str]
) -> list[list[str]]:
    if not refs:
        return []
    overlap = int(MAX_MORCEAU_CHARS * RAPPROCHEMENT_OVERLAP_RATIO)
    lots: list[list[str]] = []
    start = 0
    n = len(lignes)
    while start < n:
        total = 0
        end = start
        while end < n:
            add = len(lignes[end]) + (1 if end > start else 0)
            if total + add > MAX_MORCEAU_CHARS and end > start:
                break
            total += add
            end += 1
        if end == start:
            end = start + 1
        lots.append(refs[start:end])
        if end >= n:
            break
        back_chars = 0
        new_start = end
        while new_start > start and back_chars < overlap:
            new_start -= 1
            back_chars += len(lignes[new_start]) + 1
        start = new_start if new_start > start else end
    return lots


def preparer_entree_rapprochement(
    published: list[Any],
) -> tuple[list[str], list[str], dict[str, tuple[str, str, dict[str, Any] | None, str]]]:
    refs: list[str] = []
    lignes: list[str] = []
    meta: dict[str, tuple[str, str, dict[str, Any] | None, str]] = {}
    rows: list[tuple[tuple[str, tuple[int, ...], str], str, str, dict[str, Any] | None, str]] = []
    for item in published:
        fact = item.validated if hasattr(item, "validated") else item
        if fact.nature not in NATURES_RAPPROCHEMENT:
            continue
        ref = fact_ref(fact.piece, fact.fact_id)
        date = dict(fact.date) if fact.date else None
        iso = str((date or {}).get("iso") or "") or None
        rows.append(
            (
                _tri_rapprochement_key(ref, fact.descripteur, iso),
                ref,
                fact.enonciateur,
                date,
                fact.descripteur,
            )
        )
    rows.sort(key=lambda row: row[0])
    for _key, ref, enonciateur, date, descripteur in rows:
        refs.append(ref)
        lignes.append(formater_ligne_rapprochement(ref, enonciateur, date, descripteur))
        meta[ref] = (ref, enonciateur, date, descripteur)
    return refs, lignes, meta


def parse_reponse_rapprochement(
    text: str, *, refs_connues: set[str], inconnues: list[str] | None = None
) -> list[list[str]]:
    """Groupes de la réponse. Sans `inconnues`, une référence inconnue invalide la
    réponse ; avec, elle est retirée du groupe et consignée — elle ne peut rien inventer,
    puisqu'un groupe ne porte que des références de faits déjà publiés."""
    payload = _objet_json(text, "groupes")
    groupes = payload.get("groupes")
    if not isinstance(groupes, list):
        raise ValueError("groupes doit être une liste")
    parsed: list[list[str]] = []
    for group in groupes:
        if not isinstance(group, list):
            raise ValueError("groupe non liste")
        refs: list[str] = []
        for item in group:
            ref = str(item)
            if ref not in refs_connues:
                if inconnues is None:
                    raise ValueError(f"référence inconnue : {ref}")
                inconnues.append(ref)
            elif ref not in refs:
                refs.append(ref)
        if len(refs) >= 2:
            parsed.append(refs)
    return parsed


def fusionner_groupes(groupes: list[list[str]]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in groupes:
        if len(group) < 2:
            continue
        head = group[0]
        for ref in group[1:]:
            union(head, ref)

    buckets: dict[str, set[str]] = {}
    for group in groupes:
        for ref in group:
            buckets.setdefault(find(ref), set()).add(ref)
    return [sorted(items) for items in buckets.values() if len(items) >= 2]


def ecrire_groupes(
    dossier: str,
    groupes: list[list[str]],
    *,
    constructeur_root: Path | None = None,
) -> Path:
    path = groupes_path(dossier, constructeur_root=constructeur_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dossier": dossier, "groupes": groupes}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def lot_path(
    dossier: str, lot: list[str], *, constructeur_root: Path | None = None
) -> Path:
    """Un fichier par lot, nommé d'après son contenu : un lot déjà rendu n'est pas rappelé."""
    digest = hashlib.sha256("\n".join(lot).encode("utf-8")).hexdigest()[:16]
    return (
        constructeur_dossier_root(dossier, constructeur_root=constructeur_root)
        / "rapprochement"
        / f"lot-{digest}.json"
    )


def rapprocher_dossier(
    *,
    dossier: str,
    published: list[Any],
    chat_factory_builder: Callable[[], ChatFactory],
    constructeur_root: Path | None = None,
) -> list[list[str]]:
    """Rapproche lot par lot, avec reprise au lot. Un lot refusé deux fois est marqué en
    échec et le passage continue ; une panne du fournisseur ou le plafond l'arrêtent."""
    refs, lignes, _meta = preparer_entree_rapprochement(published)
    if len(refs) < 2:
        ecrire_groupes(dossier, [], constructeur_root=constructeur_root)
        return []
    ligne_de = dict(zip(refs, lignes, strict=True))
    lots = decouper_lots_rapprochement(refs, lignes)
    groupes: list[list[str]] = []
    for lot in lots:
        path = lot_path(dossier, lot, constructeur_root=constructeur_root)
        if path.is_file():
            conserve = json.loads(path.read_text(encoding="utf-8"))
            if conserve.get("statut") == "fait":
                groupes.extend(conserve.get("groupes") or [])
                continue
        invite = "\n".join(ligne_de[ref] for ref in lot) + "\n\n" + RAPPROCHEMENT_PROMPT
        parsed: list[list[str]] | None = None
        inconnues: list[str] = []
        brut = ""
        for reprise, nature in ((False, "rapprochement"), (True, "reprise")):
            chat = chat_factory_builder()(nature)
            brut, _ = _appeler_modele(chat, invite, reprise=reprise)
            try:
                inconnues = []
                parsed = parse_reponse_rapprochement(
                    brut, refs_connues=set(lot), inconnues=inconnues
                )
                break
            except ValueError:
                parsed = None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dossier": dossier,
                    "statut": "fait" if parsed is not None else "echec",
                    "references": lot,
                    "groupes": parsed or [],
                    "references_inconnues": inconnues,
                    "brut": brut,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if parsed:
            groupes.extend(parsed)
    merged = fusionner_groupes(groupes)
    ecrire_groupes(dossier, merged, constructeur_root=constructeur_root)
    return merged


def bilan_rapprochement(
    dossier: str, *, constructeur_root: Path | None = None
) -> dict[str, int]:
    directory = (
        constructeur_dossier_root(dossier, constructeur_root=constructeur_root)
        / "rapprochement"
    )
    lots = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("lot-*.json"))]
    return {
        "lots": len(lots),
        "lots_faits": sum(1 for lot in lots if lot.get("statut") == "fait"),
        "lots_en_echec": sum(1 for lot in lots if lot.get("statut") != "fait"),
        "groupes_bruts": sum(len(lot.get("groupes") or []) for lot in lots),
        "references_inconnues": sum(len(lot.get("references_inconnues") or []) for lot in lots),
    }


def build_chat_factory_builder(
    compteur: CompteurAppels,
    *,
    dossier: str,
    run_fn=None,
) -> Callable[[str, int], ChatFactory]:
    """Retourne une fabrique (pièce, morceau) → chat(nature)."""

    inner = run_fn or _default_run_fn_guarded

    def builder(piece: str, morceau: int) -> ChatFactory:
        def factory(nature: str) -> ChatFn:
            wrapped = compteur.envelopper_run_fn(
                inner,
                dossier=dossier,
                piece=piece,
                morceau=morceau,
                nature=nature,
            )
            return make_cursor_chat_fn(run_fn=wrapped)

        return factory

    return builder


def build_rapprochement_chat_factory(
    compteur: CompteurAppels,
    *,
    dossier: str,
    run_fn=None,
) -> Callable[[], ChatFactory]:
    inner = run_fn or _default_run_fn_guarded

    def builder() -> ChatFactory:
        def factory(nature: str) -> ChatFn:
            wrapped = compteur.envelopper_run_fn(
                inner,
                dossier=dossier,
                piece="",
                morceau=None,
                nature=nature,
            )
            return make_cursor_chat_fn(run_fn=wrapped)

        return factory

    return builder


def _default_run_fn_guarded(*args, **kwargs):
    import os

    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("appel cursor-agent interdit sous pytest")
    from .cursor_chat import _default_run_fn

    return _default_run_fn(*args, **kwargs)
