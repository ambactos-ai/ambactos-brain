# SPDX-License-Identifier: AGPL-3.0-only
"""Recherche filtrée par reconstitution d'entités — sélection sans modèle."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from banc_realite.compat_moteur import _HYBRID_RRF_K

from .lecteur_faits_classes import cosine, empreinte_texte
from .reconstitution_entites import (
    ID_PATTERN,
    _occurrences_cles_masquees,
    _toutes_cles_entites,
)

_JOURNAL = logging.getLogger(__name__)

K_GRANDEURS = 6
K_UNITES_REPLI = 6
N_OBJETS_FILTRE = 12
N_DESACCORDS_EN_TETE = 3
BM25_K1 = 1.5
BM25_B = 0.75
SORTES_OBJET = ("fait", "regle", "lignes")
COMPARAISONS_PRIORITAIRES = frozenset({"desaccord", "non_ordonnable"})

DATE_JOUR = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
DATE_MOIS = re.compile(r"\b(20\d{2})-(\d{2})\b(?!\-\d{2}\b)")
ANNEE_SEULE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
NOMBRE = re.compile(r"(?<![\w.])(?:\d{1,3}(?:[ ,]\d{3})*\.\d+|\d+(?:\.\d+)?)(?![\w.])")
MOT = re.compile(r"[A-Za-zÀ-ÿ]{3,}")

MOIS_NOMS: dict[str, str] = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
    "janvier": "01",
    "fevrier": "02",
    "février": "02",
    "mars": "03",
    "avril": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07",
    "aout": "08",
    "août": "08",
    "septembre": "09",
    "octobre": "10",
    "novembre": "11",
    "decembre": "12",
    "décembre": "12",
}
MOIS_NOM_ANNEE = re.compile(
    r"\b(" + "|".join(sorted(MOIS_NOMS, key=len, reverse=True)) + r")\s+(20\d{2})\b",
    re.IGNORECASE,
)

MOTS_VIDES = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "its",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "did",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "that",
        "with",
        "have",
        "this",
        "will",
        "your",
        "from",
        "they",
        "been",
        "than",
        "what",
        "when",
        "where",
        "which",
        "while",
        "about",
        "after",
        "before",
        "between",
        "during",
        "into",
        "through",
        "under",
        "over",
        "such",
        "each",
        "other",
        "some",
        "more",
        "most",
        "many",
        "much",
        "any",
        "also",
        "just",
        "only",
        "very",
        "well",
        "back",
        "even",
        "still",
        "les",
        "des",
        "une",
        "dans",
        "pour",
        "par",
        "sur",
        "avec",
        "sans",
        "plus",
        "tout",
        "tous",
        "toute",
        "cette",
        "ces",
        "son",
        "ses",
        "leur",
        "leurs",
        "qui",
        "que",
        "quoi",
        "dont",
        "est",
        "sont",
        "été",
        "etre",
        "être",
        "avoir",
        "faire",
        "peut",
        "comme",
        "mais",
        "ou",
        "où",
        "si",
        "ne",
        "pas",
        "nous",
        "vous",
        "ils",
        "elles",
        "mon",
        "ton",
        "mes",
        "tes",
        "were",
        "does",
    }
)


def chemin_reconstitution(dossier: str) -> Path:
    return BANC_VAR / "dossiers" / dossier / "reconstitution"


@dataclass(frozen=True, slots=True)
class ReconnaissanceQuestion:
    cles: tuple[str, ...]
    identifiants: tuple[str, ...]
    dates: tuple[str, ...]
    mois: tuple[str, ...]
    annees: tuple[str, ...]
    nombres: tuple[str, ...]
    mots: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "cles": list(self.cles),
            "identifiants": list(self.identifiants),
            "dates": list(self.dates),
            "mois": list(self.mois),
            "annees": list(self.annees),
            "nombres": list(self.nombres),
            "mots": list(self.mots),
        }

    @property
    def a_temporel(self) -> bool:
        return bool(self.dates or self.mois or self.annees)


@dataclass(frozen=True, slots=True)
class UniteCandidate:
    id: str
    motif: str


@dataclass(frozen=True, slots=True)
class LigneDesignee:
    piece: str
    lignes: tuple[int, ...]
    lettre: str
    score: int
    source: str


@dataclass(frozen=True, slots=True)
class FaitClasse:
    fait: Mapping[str, Any]
    score: int
    rang: int
    unite_id: str


@dataclass(frozen=True, slots=True)
class LigneObjet:
    piece: str
    numero: int
    lettre: str


@dataclass(frozen=True, slots=True)
class Objet:
    id: str
    unite_id: str
    sorte: str
    comparaison: str
    lignes: tuple[LigneObjet, ...]
    texte_recherche: str
    payload: Mapping[str, Any] | None = None

    @property
    def text(self) -> str:
        return self.texte_recherche

    @property
    def empreinte(self) -> str:
        return empreinte_texte(self.texte_recherche)


@dataclass(frozen=True, slots=True)
class ObjetClasse:
    objet: Objet
    score: float
    rang: int


@dataclass(frozen=True, slots=True)
class Selection:
    reconnaissance: ReconnaissanceQuestion
    unites_candidates: tuple[UniteCandidate, ...]
    faits_classes: tuple[FaitClasse, ...]
    lignes_designees: tuple[LigneDesignee, ...]
    pieces_designees: tuple[str, ...]
    objets_classes: tuple[ObjetClasse, ...] = ()
    unites_absentes: int = 0


@dataclass
class IndexReconstitution:
    repertoire: Path
    registre: dict[str, Any]
    toutes_cles: list[str]
    unites: list[dict[str, Any]]
    unites_par_id: dict[str, dict[str, Any]]
    entites_par_nom: dict[str, dict[str, Any]]
    identifiant_vers_unites: dict[str, frozenset[str]]
    texte_unite: dict[str, str]
    absentes: tuple[str, ...] = ()
    _absentes_set: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)
    _entites_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def entite(self, unite_id: str) -> dict[str, Any]:
        if unite_id not in self._entites_cache:
            path = self.repertoire / "entites" / f"{unite_id}.json"
            if not path.is_file():
                raise FileNotFoundError(f"reconstitution entité absente : {path}")
            self._entites_cache[unite_id] = json.loads(path.read_text(encoding="utf-8"))
        return self._entites_cache[unite_id]


def _dedupe_ordre(items: list[str]) -> tuple[str, ...]:
    vus: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in vus:
            continue
        vus.add(item)
        out.append(item)
    return tuple(out)


def lire_question(texte: str, index: IndexReconstitution) -> ReconnaissanceQuestion:
    cles = _dedupe_ordre([cle for _, _, cle in _occurrences_cles_masquees(texte, index.toutes_cles)])
    identifiants = _dedupe_ordre(ID_PATTERN.findall(texte))
    dates = _dedupe_ordre(
        f"{a}-{m}-{j}" for a, m, j in DATE_JOUR.findall(texte)
    )
    mois = _dedupe_ordre(f"{a}-{m}" for a, m in DATE_MOIS.findall(texte))
    for nom, annee in MOIS_NOM_ANNEE.findall(texte):
        mois = _dedupe_ordre([*mois, f"{annee}-{MOIS_NOMS[nom.lower()]}"])
    annees = _dedupe_ordre(ANNEE_SEULE.findall(texte))
    nombres = _dedupe_ordre(NOMBRE.findall(texte))
    mots_bruts = [m.lower() for m in MOT.findall(texte)]
    mots = _dedupe_ordre(m for m in mots_bruts if m not in MOTS_VIDES)
    return ReconnaissanceQuestion(
        cles=cles,
        identifiants=identifiants,
        dates=dates,
        mois=mois,
        annees=annees,
        nombres=nombres,
        mots=mots,
    )


def _texte_paquet(paquet: Mapping[str, Any]) -> str:
    morceaux = [str(paquet.get("titre") or "")]
    for contenu in (paquet.get("sections") or {}).values():
        for lignes in contenu.values():
            morceaux.extend(str(texte) for texte in lignes.values())
    return "\n".join(m for m in morceaux if m)


def _index_identifiants_paquet(unite_id: str, paquet: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    trouves: dict[str, set[str]] = {}
    for contenu in (paquet.get("sections") or {}).values():
        for lignes in contenu.values():
            for texte in lignes.values():
                for ident in ID_PATTERN.findall(str(texte)):
                    trouves.setdefault(ident, set()).add(unite_id)
    return {ident: frozenset(unites) for ident, unites in trouves.items()}


def charger_index(repertoire: Path | str) -> IndexReconstitution:
    root = Path(repertoire).expanduser()
    registre_path = root / "registre.json"
    if not registre_path.is_file():
        raise FileNotFoundError(
            f"reconstitution absente pour {root.name} : {registre_path} — lancer la reconstitution"
        )
    registre = json.loads(registre_path.read_text(encoding="utf-8"))
    toutes_cles = _toutes_cles_entites(registre.get("entites") or [])
    unites = list(registre.get("unites") or [])
    unites_par_id = {str(u["id"]): u for u in unites if u.get("id")}
    entites_par_nom = {
        str(entite["nom"]): entite
        for entite in registre.get("entites") or []
        if entite.get("nom")
    }
    identifiant_vers_unites: dict[str, set[str]] = {}
    texte_unite: dict[str, str] = {}
    paquets_dir = root / "paquets"
    for unite in unites:
        uid = str(unite["id"])
        morceaux = [str(unite.get("titre") or "")]
        paquet_path = paquets_dir / f"{uid}.json"
        paquet: dict[str, Any] = {}
        if paquet_path.is_file():
            paquet = json.loads(paquet_path.read_text(encoding="utf-8"))
            morceaux.append(_texte_paquet(paquet))
            for ident, uids in _index_identifiants_paquet(uid, paquet).items():
                identifiant_vers_unites.setdefault(ident, set()).update(uids)
        for ref in unite.get("lignes") or []:
            if not isinstance(ref, Mapping):
                continue
            piece = ref.get("piece")
            nums = ref.get("lignes") or []
            if piece and nums and paquet:
                section = (paquet.get("sections") or {}).get("emergent") or {}
                lignes_piece = section.get(piece) or {}
                for num in nums:
                    texte = lignes_piece.get(str(int(num)))
                    if texte:
                        morceaux.append(str(texte))
        texte_unite[uid] = "\n".join(m for m in morceaux if m)
    entites_dir = root / "entites"
    absentes = tuple(
        sorted(
            uid
            for uid in unites_par_id
            if not (entites_dir / f"{uid}.json").is_file()
        )
    )
    return IndexReconstitution(
        repertoire=root,
        registre=registre,
        toutes_cles=toutes_cles,
        unites=unites,
        unites_par_id=unites_par_id,
        entites_par_nom=entites_par_nom,
        identifiant_vers_unites={
            ident: frozenset(uids) for ident, uids in identifiant_vers_unites.items()
        },
        texte_unite=texte_unite,
        absentes=absentes,
        _absentes_set=frozenset(absentes),
    )


def _unite_reconstituee(index: IndexReconstitution, uid: str) -> bool:
    return bool(uid) and uid not in index._absentes_set


def _periode_recouvre(unite: Mapping[str, Any], reconnaissance: ReconnaissanceQuestion) -> bool:
    periode = list(unite.get("periode") or [])
    if not periode:
        return True
    if not reconnaissance.a_temporel:
        return True
    mois_contraints = set(reconnaissance.mois)
    for jour in reconnaissance.dates:
        mois_contraints.add(jour[:7])
    if mois_contraints:
        return any(m in periode for m in mois_contraints)
    annees_seules = set(reconnaissance.annees)
    for mois in reconnaissance.mois:
        annees_seules.discard(mois[:4])
    for jour in reconnaissance.dates:
        annees_seules.discard(jour[:4])
    if annees_seules:
        return any(
            any(str(p).startswith(annee) for p in periode) for annee in annees_seules
        )
    return True


def _entites_depuis_cles(
    cles: tuple[str, ...], index: IndexReconstitution
) -> set[str]:
    cles_vues = {c.lower() for c in cles}
    entites: set[str] = set()
    for entite in index.registre.get("entites") or []:
        nom = str(entite.get("nom") or "")
        for cle in entite.get("cles") or []:
            if isinstance(cle, str) and cle.lower() in cles_vues:
                entites.add(nom)
                break
    return entites


def _tokeniser_bm25(reconnaissance: ReconnaissanceQuestion) -> list[str]:
    tokens: list[str] = []
    tokens.extend(reconnaissance.mots)
    tokens.extend(reconnaissance.mois)
    tokens.extend(reconnaissance.dates)
    tokens.extend(reconnaissance.annees)
    tokens.extend(reconnaissance.nombres)
    return [t.lower() for t in tokens if t]


def _tokeniser_texte(texte: str) -> list[str]:
    tokens = [m.lower() for m in MOT.findall(texte)]
    tokens.extend(f"{a}-{m}" for a, m in DATE_MOIS.findall(texte))
    tokens.extend(f"{a}-{m}-{j}" for a, m, j in DATE_JOUR.findall(texte))
    tokens.extend(ANNEE_SEULE.findall(texte))
    tokens.extend(NOMBRE.findall(texte))
    return tokens


def _bm25(
    query_tokens: list[str],
    documents: dict[str, list[str]],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> dict[str, float]:
    if not query_tokens or not documents:
        return {}
    index = IndexBm25Preparation.depuis_documents(documents)
    return index.interroger(query_tokens, k1=k1, b=b)


def _variantes_nombre(nombre: str) -> tuple[str, ...]:
    normalise = nombre.replace(" ", "").replace(",", "")
    compact = re.sub(r"\.0+$", "", normalise) if "." in normalise else normalise
    return tuple({nombre, normalise, compact} - {""})


@dataclass(frozen=True, slots=True)
class IndexBm25Preparation:
    """Statistiques BM25 précalculées ; ``interroger`` reproduit ``_bm25`` à 1e-9."""

    documents: dict[str, tuple[str, ...]]
    n_docs: int
    avgdl: float
    df: Counter[str]
    tf_par_doc: dict[str, Counter[str]]
    longueur_par_doc: dict[str, int]
    postings: dict[str, frozenset[str]]

    @classmethod
    def depuis_documents(cls, documents: Mapping[str, Sequence[str]]) -> IndexBm25Preparation:
        docs = {uid: tuple(doc) for uid, doc in documents.items()}
        n_docs = len(docs)
        longueurs = [len(doc) for doc in docs.values()]
        avgdl = sum(longueurs) / n_docs if longueurs else 0.0
        df: Counter[str] = Counter()
        tf_par_doc: dict[str, Counter[str]] = {}
        longueur_par_doc: dict[str, int] = {}
        postings_sets: dict[str, set[str]] = {}
        for uid, doc in docs.items():
            dl = len(doc) or 1
            longueur_par_doc[uid] = dl
            tf = Counter(doc)
            tf_par_doc[uid] = tf
            df.update(set(doc))
            for term in tf:
                postings_sets.setdefault(term, set()).add(uid)
        return cls(
            documents=docs,
            n_docs=n_docs,
            avgdl=avgdl,
            df=df,
            tf_par_doc=tf_par_doc,
            longueur_par_doc=longueur_par_doc,
            postings={k: frozenset(v) for k, v in postings_sets.items()},
        )

    def interroger(
        self,
        query_tokens: list[str],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> dict[str, float]:
        if not query_tokens or not self.documents:
            return {}
        scores: dict[str, float] = {}
        query = set(query_tokens)
        candidats: set[str] = set()
        for term in query:
            candidats.update(self.postings.get(term, ()))
        for uid in candidats:
            dl = self.longueur_par_doc[uid]
            tf = self.tf_par_doc[uid]
            total = 0.0
            for term in query:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = math.log(
                    (self.n_docs - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0
                )
                denom = freq + k1 * (1.0 - b + b * dl / self.avgdl)
                total += idf * (freq * (k1 + 1.0)) / denom
            if total > 0:
                scores[uid] = total
        return scores


@dataclass(frozen=True, slots=True)
class PreparationObjets:
    """Caches par tenant pour scores lettre, BM25 objets et candidats d'unités."""

    objet_ids: tuple[str, ...]
    lettre_par_id: dict[str, str]
    index_cles: dict[str, frozenset[str]]
    index_mots: dict[str, frozenset[str]]
    index_identifiants: dict[str, frozenset[str]]
    bm25_objets: IndexBm25Preparation
    unites_par_entite: dict[str, tuple[str, ...]]
    unites_reconstituees: frozenset[str]
    bm25_grandeurs: IndexBm25Preparation
    bm25_unites_repli: IndexBm25Preparation


def preparer_objets(index: IndexReconstitution) -> PreparationObjets:
    objets = objets_de_recherche(index)
    objet_ids: list[str] = []
    lettre_par_id: dict[str, str] = {}
    index_cles: dict[str, set[str]] = {}
    index_mots: dict[str, set[str]] = {}
    index_identifiants: dict[str, set[str]] = {}
    documents_bm25: dict[str, list[str]] = {}

    cles_connues = list(index.toutes_cles)
    for objet in objets:
        oid = objet.id
        objet_ids.append(oid)
        lettre = _lettre_objet(objet)
        lettre_par_id[oid] = lettre
        if not lettre.strip():
            documents_bm25[oid] = _tokeniser_texte(objet.texte_recherche)
            continue
        lettre_l = lettre.lower()
        documents_bm25[oid] = _tokeniser_texte(objet.texte_recherche)
        for cle in cles_connues:
            cle_l = cle.lower()
            if cle_l not in lettre_l:
                continue
            if _contient_cle(lettre, cle):
                index_cles.setdefault(cle_l, set()).add(oid)
        for mot in MOT.findall(lettre_l):
            index_mots.setdefault(mot, set()).add(oid)
        for ident in ID_PATTERN.findall(lettre):
            index_identifiants.setdefault(ident, set()).add(oid)

    unites_par_entite: dict[str, list[str]] = {}
    unites_reconst: set[str] = set()
    docs_grandeur: dict[str, list[str]] = {}
    docs_repli: dict[str, list[str]] = {}
    for unite in index.unites:
        uid = str(unite.get("id") or "")
        if not uid or not _unite_reconstituee(index, uid):
            continue
        unites_reconst.add(uid)
        entite = str(unite.get("entite") or "")
        if entite:
            unites_par_entite.setdefault(entite, []).append(uid)
        texte = index.texte_unite.get(uid, "")
        if not texte.strip():
            continue
        tokens = _tokeniser_texte(texte)
        if unite.get("type") == "grandeur":
            docs_grandeur[uid] = tokens
        docs_repli[uid] = tokens

    return PreparationObjets(
        objet_ids=tuple(objet_ids),
        lettre_par_id=lettre_par_id,
        index_cles={k: frozenset(v) for k, v in index_cles.items()},
        index_mots={k: frozenset(v) for k, v in index_mots.items()},
        index_identifiants={k: frozenset(v) for k, v in index_identifiants.items()},
        bm25_objets=IndexBm25Preparation.depuis_documents(documents_bm25),
        unites_par_entite={k: tuple(v) for k, v in unites_par_entite.items()},
        unites_reconstituees=frozenset(unites_reconst),
        bm25_grandeurs=IndexBm25Preparation.depuis_documents(docs_grandeur),
        bm25_unites_repli=IndexBm25Preparation.depuis_documents(docs_repli),
    )


def scores_lettre_prepares(
    reconnaissance: ReconnaissanceQuestion,
    preparation: PreparationObjets,
) -> dict[str, int]:
    scores: dict[str, int] = {oid: 0 for oid in preparation.objet_ids}
    for cle in reconnaissance.cles:
        for oid in preparation.index_cles.get(cle.lower(), ()):
            scores[oid] += 3
    for ident in reconnaissance.identifiants:
        for oid in preparation.index_identifiants.get(ident, ()):
            scores[oid] += 3
    for date in reconnaissance.dates:
        for oid in preparation.objet_ids:
            if date in preparation.lettre_par_id[oid]:
                scores[oid] += 3
    for mois in reconnaissance.mois:
        for oid in preparation.objet_ids:
            if mois in preparation.lettre_par_id[oid]:
                scores[oid] += 3
    for annee in reconnaissance.annees:
        for oid in preparation.objet_ids:
            if annee in preparation.lettre_par_id[oid]:
                scores[oid] += 3
    if reconnaissance.nombres:
        for oid in preparation.objet_ids:
            lettre = preparation.lettre_par_id[oid]
            if not any(ch.isdigit() for ch in lettre):
                continue
            for nombre in reconnaissance.nombres:
                if _contient_nombre(lettre, nombre):
                    scores[oid] += 3
    for mot in reconnaissance.mots:
        for oid in preparation.index_mots.get(mot, ()):
            scores[oid] += 1
    return scores


def _unites_par_cle(
    reconnaissance: ReconnaissanceQuestion,
    index: IndexReconstitution,
    *,
    preparation: PreparationObjets | None = None,
) -> list[UniteCandidate]:
    entites = _entites_depuis_cles(reconnaissance.cles, index)
    if not entites:
        return []
    out: list[UniteCandidate] = []
    vus: set[str] = set()
    if preparation is not None:
        for entite in entites:
            for uid in preparation.unites_par_entite.get(entite, ()):
                unite = index.unites_par_id.get(uid) or {}
                if unite.get("type") == "grandeur":
                    continue
                if not _periode_recouvre(unite, reconnaissance):
                    continue
                if uid in vus:
                    continue
                vus.add(uid)
                out.append(UniteCandidate(id=uid, motif="cle"))
        return out
    for unite in index.unites:
        uid = str(unite.get("id") or "")
        if not uid or unite.get("type") == "grandeur":
            continue
        if not _unite_reconstituee(index, uid):
            continue
        if str(unite.get("entite") or "") not in entites:
            continue
        if not _periode_recouvre(unite, reconnaissance):
            continue
        if uid in vus:
            continue
        vus.add(uid)
        out.append(UniteCandidate(id=uid, motif="cle"))
    return out


def _unites_par_identifiant(
    reconnaissance: ReconnaissanceQuestion, index: IndexReconstitution
) -> list[UniteCandidate]:
    out: list[UniteCandidate] = []
    vus: set[str] = set()
    for ident in reconnaissance.identifiants:
        for uid in index.identifiant_vers_unites.get(ident, frozenset()):
            if uid in vus or not _unite_reconstituee(index, uid):
                continue
            vus.add(uid)
            out.append(UniteCandidate(id=uid, motif="identifiant"))
    return out


def _meilleures_unites_bm25(
    index: IndexReconstitution,
    reconnaissance: ReconnaissanceQuestion,
    *,
    type_unite: str | None,
    k: int,
    preparation: PreparationObjets | None = None,
) -> list[UniteCandidate]:
    query = _tokeniser_bm25(reconnaissance)
    if preparation is not None:
        bm25 = (
            preparation.bm25_grandeurs
            if type_unite == "grandeur"
            else preparation.bm25_unites_repli
        )
        scores = bm25.interroger(query)
    else:
        documents: dict[str, list[str]] = {}
        for unite in index.unites:
            if type_unite is not None and unite.get("type") != type_unite:
                continue
            uid = str(unite.get("id") or "")
            if not _unite_reconstituee(index, uid):
                continue
            texte = index.texte_unite.get(uid, "")
            if not texte.strip():
                continue
            documents[uid] = _tokeniser_texte(texte)
        scores = _bm25(query, documents)
    ordered = sorted(scores, key=lambda uid: (-scores[uid], uid))[:k]
    motif = "grandeur" if type_unite == "grandeur" else "repli"
    return [UniteCandidate(id=uid, motif=motif) for uid in ordered]


def _unites_candidates(
    reconnaissance: ReconnaissanceQuestion,
    index: IndexReconstitution,
    *,
    preparation: PreparationObjets | None = None,
) -> tuple[UniteCandidate, ...]:
    par_cle = _unites_par_cle(reconnaissance, index, preparation=preparation)
    par_ident = _unites_par_identifiant(reconnaissance, index)
    grandeur = _meilleures_unites_bm25(
        index,
        reconnaissance,
        type_unite="grandeur",
        k=K_GRANDEURS,
        preparation=preparation,
    )
    repli: list[UniteCandidate] = []
    if not par_cle and not par_ident:
        repli = _meilleures_unites_bm25(
            index,
            reconnaissance,
            type_unite=None,
            k=K_UNITES_REPLI,
            preparation=preparation,
        )
    vus: set[str] = set()
    out: list[UniteCandidate] = []
    for groupe in (par_cle, par_ident, grandeur, repli):
        for candidate in groupe:
            if candidate.id in vus:
                continue
            vus.add(candidate.id)
            out.append(candidate)
    return tuple(out)


def _contient_identifiant(lettre: str, ident: str) -> bool:
    return ident in lettre


def _contient_cle(lettre: str, cle: str) -> bool:
    lettre_l = lettre.lower()
    cle_l = cle.lower()
    pos = lettre_l.find(cle_l)
    if pos == -1:
        return False
    fin = pos + len(cle_l)
    avant_ok = pos == 0 or not lettre_l[pos - 1].isalnum()
    apres_ok = fin >= len(lettre_l) or not lettre_l[fin].isalnum()
    return avant_ok and apres_ok


def _contient_nombre(lettre: str, nombre: str) -> bool:
    normalise = nombre.replace(" ", "").replace(",", "")
    compact = re.sub(r"\.0+$", "", normalise) if "." in normalise else normalise
    variantes = {nombre, normalise, compact}
    return any(v in lettre for v in variantes if v)


def _score_lettre(reconnaissance: ReconnaissanceQuestion, lettre: str) -> int:
    if not lettre.strip():
        return 0
    score = 0
    for cle in reconnaissance.cles:
        if _contient_cle(lettre, cle):
            score += 3
    for ident in reconnaissance.identifiants:
        if _contient_identifiant(lettre, ident):
            score += 3
    for date in reconnaissance.dates:
        if date in lettre:
            score += 3
    for mois in reconnaissance.mois:
        if mois in lettre:
            score += 3
    for annee in reconnaissance.annees:
        if annee in lettre:
            score += 3
    for nombre in reconnaissance.nombres:
        if _contient_nombre(lettre, nombre):
            score += 3
    lettre_l = lettre.lower()
    for mot in reconnaissance.mots:
        if re.search(rf"(?<![A-Za-zÀ-ÿ]){re.escape(mot)}(?![A-Za-zÀ-ÿ])", lettre_l):
            score += 1
    return score


def _lignes_objet(
    obj: Mapping[str, Any],
    *,
    source: str,
    reconnaissance: ReconnaissanceQuestion,
) -> LigneDesignee | None:
    piece = str(obj.get("piece") or "")
    try:
        lignes = tuple(sorted({int(n) for n in obj.get("lignes") or []}))
    except (TypeError, ValueError):
        lignes = ()
    lettre = str(obj.get("lettre") or "")
    score = _score_lettre(reconnaissance, lettre)
    if not piece or not lignes or not lettre:
        return None
    return LigneDesignee(
        piece=piece,
        lignes=lignes,
        lettre=lettre,
        score=score,
        source=source,
    )


def _classer_faits_unite(
    payload: Mapping[str, Any],
    unite_id: str,
    reconnaissance: ReconnaissanceQuestion,
) -> tuple[list[FaitClasse], list[LigneDesignee]]:
    faits_classes: list[FaitClasse] = []
    lignes: list[LigneDesignee] = []
    for fait in payload.get("faits") or []:
        versions = list(fait.get("versions") or [])
        scores: list[int] = []
        lignes_fait: list[LigneDesignee] = []
        for version in versions:
            ligne = _lignes_objet(version, source="version", reconnaissance=reconnaissance)
            if ligne is None:
                continue
            lignes_fait.append(ligne)
            scores.append(ligne.score)
        score_fait = max(scores) if scores else 0
        comparaison = str(fait.get("comparaison") or "")
        if score_fait > 0 and comparaison in {"desaccord", "non_ordonnable"}:
            score_fait += 2
        if score_fait <= 0:
            continue
        rang = len(faits_classes) + 1
        faits_classes.append(
            FaitClasse(fait=fait, score=score_fait, rang=rang, unite_id=unite_id)
        )
        lignes.extend(lignes_fait)
    for regle in payload.get("regles") or []:
        ligne = _lignes_objet(regle, source="regle", reconnaissance=reconnaissance)
        if ligne is not None and ligne.score > 0:
            lignes.append(ligne)
    for identite in payload.get("identites") or []:
        ligne = _lignes_objet(identite, source="identite", reconnaissance=reconnaissance)
        if ligne is not None and ligne.score > 0:
            lignes.append(ligne)
    return faits_classes, lignes


def selectionner(question_texte: str, index: IndexReconstitution) -> Selection:
    reconnaissance = lire_question(question_texte, index)
    candidates = _unites_candidates(reconnaissance, index)
    faits: list[FaitClasse] = []
    lignes: list[LigneDesignee] = []
    for candidate in candidates:
        payload = index.entite(candidate.id)
        faits_u, lignes_u = _classer_faits_unite(payload, candidate.id, reconnaissance)
        faits.extend(faits_u)
        lignes.extend(lignes_u)
    faits.sort(key=lambda item: (-item.score, item.rang, item.unite_id))
    for idx, fait in enumerate(faits, start=1):
        faits[idx - 1] = FaitClasse(
            fait=fait.fait, score=fait.score, rang=idx, unite_id=fait.unite_id
        )
    pieces: list[str] = []
    vus_pieces: set[str] = set()
    for ligne in lignes:
        if ligne.score <= 0:
            continue
        if ligne.piece not in vus_pieces:
            vus_pieces.add(ligne.piece)
            pieces.append(ligne.piece)
    return Selection(
        reconnaissance=reconnaissance,
        unites_candidates=candidates,
        faits_classes=tuple(faits),
        lignes_designees=tuple(lignes),
        pieces_designees=tuple(pieces),
        unites_absentes=len(index.absentes),
    )


def _ints(valeurs: Any) -> tuple[int, ...]:
    out: list[int] = []
    for item in valeurs or []:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def _cles_entite_unite(unite: Mapping[str, Any], index: IndexReconstitution) -> tuple[str, ...]:
    nom = str(unite.get("entite") or "")
    entite = index.entites_par_nom.get(nom) or {}
    cles = [str(cle) for cle in (entite.get("cles") or []) if cle]
    if not cles:
        cles = [str(cle) for cle in (unite.get("cles") or []) if cle]
    return tuple(cles)


def _charger_paquet(index: IndexReconstitution, unite_id: str) -> dict[str, Any]:
    cache = getattr(index, "_paquets_cache", None)
    if cache is None:
        cache = {}
        index._paquets_cache = cache
    if unite_id not in cache:
        path = index.repertoire / "paquets" / f"{unite_id}.json"
        cache[unite_id] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return cache[unite_id]


def _lignes_paquet_entite_document(paquet: Mapping[str, Any]) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    sections = paquet.get("sections") or {}
    for nom in ("entite", "document"):
        for piece, lignes in (sections.get(nom) or {}).items():
            dest = out.setdefault(str(piece), {})
            if not isinstance(lignes, Mapping):
                continue
            for num_s, lettre in lignes.items():
                try:
                    num = int(num_s)
                except (TypeError, ValueError):
                    continue
                dest.setdefault(num, str(lettre or ""))
    return out


def _ajouter_citation(cites: dict[str, set[int]], piece: str, numeros: Any) -> None:
    if not piece:
        return
    nums = _ints(numeros)
    if not nums:
        return
    cites.setdefault(piece, set()).update(nums)


def _lignes_citees_payload(payload: Mapping[str, Any]) -> dict[str, set[int]]:
    cites: dict[str, set[int]] = {}
    for fait in payload.get("faits") or []:
        for version in fait.get("versions") or []:
            _ajouter_citation(cites, str(version.get("piece") or ""), version.get("lignes"))
    for regle in payload.get("regles") or []:
        _ajouter_citation(cites, str(regle.get("piece") or ""), regle.get("lignes"))
    for identite in payload.get("identites") or []:
        _ajouter_citation(cites, str(identite.get("piece") or ""), identite.get("lignes"))
    return cites


def _lignes_depuis_refs(refs: Sequence[Mapping[str, Any]]) -> tuple[LigneObjet, ...]:
    vus: set[tuple[str, int]] = set()
    out: list[LigneObjet] = []
    for ref in refs:
        piece = str(ref.get("piece") or "")
        lettre = str(ref.get("lettre") or "")
        for num in _ints(ref.get("lignes")):
            cle = (piece, num)
            if not piece or cle in vus:
                continue
            vus.add(cle)
            out.append(LigneObjet(piece=piece, numero=num, lettre=lettre))
    out.sort(key=lambda item: (item.piece, item.numero))
    return tuple(out)


def _blocs_consecutifs(numeros: Sequence[int]) -> list[list[int]]:
    if not numeros:
        return []
    ordered = sorted(set(int(n) for n in numeros))
    blocs: list[list[int]] = []
    courant = [ordered[0]]
    for num in ordered[1:]:
        if num == courant[-1] + 1:
            courant.append(num)
        else:
            blocs.append(courant)
            courant = [num]
    blocs.append(courant)
    return blocs


def _texte_recherche_objet(
    *,
    titre: str,
    cles_entite: Sequence[str],
    tete: str,
    lignes: Sequence[LigneObjet],
) -> str:
    parts: list[str] = []
    if titre:
        parts.append(titre)
    for cle in cles_entite:
        if cle:
            parts.append(str(cle))
    if tete:
        parts.append(tete)
    for ligne in lignes:
        if ligne.piece:
            parts.append(ligne.piece)
        if ligne.lettre:
            parts.append(ligne.lettre)
    return "\n".join(parts)


def _objet(
    *,
    ident: str,
    unite_id: str,
    sorte: str,
    comparaison: str,
    lignes: Sequence[LigneObjet],
    titre: str,
    cles_entite: Sequence[str],
    tete: str,
    payload: Mapping[str, Any] | None = None,
) -> Objet:
    texte = _texte_recherche_objet(
        titre=titre, cles_entite=cles_entite, tete=tete, lignes=lignes
    )
    return Objet(
        id=ident,
        unite_id=unite_id,
        sorte=sorte,
        comparaison=comparaison,
        lignes=tuple(lignes),
        texte_recherche=texte,
        payload=payload,
    )


def objets_de_recherche(index: IndexReconstitution) -> list[Objet]:
    """Un objet par fait, par règle, et par bloc de lignes de paquet non citées."""
    cached = getattr(index, "_objets_cache", None)
    if cached is not None:
        return cached
    objets: list[Objet] = []
    for unite in index.unites:
        uid = str(unite.get("id") or "")
        if not uid or not _unite_reconstituee(index, uid):
            continue
        titre = str(unite.get("titre") or "")
        cles = _cles_entite_unite(unite, index)
        payload = index.entite(uid)
        for idx, fait in enumerate(payload.get("faits") or [], start=1):
            lignes = _lignes_depuis_refs(list(fait.get("versions") or []))
            objets.append(
                _objet(
                    ident=f"{uid}#f{idx}",
                    unite_id=uid,
                    sorte="fait",
                    comparaison=str(fait.get("comparaison") or ""),
                    lignes=lignes,
                    titre=titre,
                    cles_entite=cles,
                    tete=str(fait.get("quoi") or ""),
                    payload=fait if isinstance(fait, Mapping) else None,
                )
            )
        for idx, regle in enumerate(payload.get("regles") or [], start=1):
            lignes = _lignes_depuis_refs([regle] if isinstance(regle, Mapping) else [])
            objets.append(
                _objet(
                    ident=f"{uid}#r{idx}",
                    unite_id=uid,
                    sorte="regle",
                    comparaison="",
                    lignes=lignes,
                    titre=titre,
                    cles_entite=cles,
                    tete=str(regle.get("enonce") or "") if isinstance(regle, Mapping) else "",
                    payload=regle if isinstance(regle, Mapping) else None,
                )
            )
        cites = _lignes_citees_payload(payload)
        paquet_lignes = _lignes_paquet_entite_document(_charger_paquet(index, uid))
        for piece, par_num in paquet_lignes.items():
            restants = [
                num for num in par_num if num not in cites.get(piece, set())
            ]
            for bloc in _blocs_consecutifs(restants):
                lignes = tuple(
                    LigneObjet(piece=piece, numero=num, lettre=par_num.get(num, ""))
                    for num in bloc
                )
                premiere = bloc[0]
                objets.append(
                    _objet(
                        ident=f"{uid}#l{piece}:{premiere}",
                        unite_id=uid,
                        sorte="lignes",
                        comparaison="",
                        lignes=lignes,
                        titre=titre,
                        cles_entite=cles,
                        tete="",
                    )
                )
    index._objets_cache = objets
    return objets


def _lettre_objet(objet: Objet) -> str:
    return "\n".join(ligne.lettre for ligne in objet.lignes if ligne.lettre)


def _rangs_par_score(scores: Mapping[str, float], identifiants: Sequence[str]) -> dict[str, int]:
    """Rang 1-based ; à score égal, même rang (les nuls partagent le dernier)."""
    valeurs = {ident: float(scores.get(ident, 0.0)) for ident in identifiants}
    return {
        ident: 1 + sum(1 for autre in identifiants if valeurs[autre] > valeurs[ident])
        for ident in identifiants
    }


def _fusion_rrf(*rangs: Mapping[str, int], k: int = _HYBRID_RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for table in rangs:
        for ident, rang in table.items():
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k + int(rang))
    return scores


def _priorite_desaccord(objet: Objet) -> int:
    if objet.sorte == "fait" and objet.comparaison in COMPARAISONS_PRIORITAIRES:
        return 0
    return 1


def _unites_bonus(
    reconnaissance: ReconnaissanceQuestion,
    index: IndexReconstitution,
    *,
    preparation: PreparationObjets | None = None,
) -> set[str]:
    bonus: set[str] = set()
    for candidate in (
        *_unites_par_cle(reconnaissance, index, preparation=preparation),
        *_unites_par_identifiant(reconnaissance, index),
    ):
        unite = index.unites_par_id.get(candidate.id) or {}
        if _periode_recouvre(unite, reconnaissance):
            bonus.add(candidate.id)
    return bonus


def _lignes_designees_objet(objet: Objet, score: int) -> list[LigneDesignee]:
    par_piece: dict[str, list[LigneObjet]] = {}
    ordre: list[str] = []
    for ligne in objet.lignes:
        if ligne.piece not in par_piece:
            ordre.append(ligne.piece)
            par_piece[ligne.piece] = []
        par_piece[ligne.piece].append(ligne)
    out: list[LigneDesignee] = []
    for piece in ordre:
        items = par_piece[piece]
        nums = tuple(sorted({item.numero for item in items}))
        lettre = "\n".join(item.lettre for item in items if item.lettre)
        if not piece or not nums:
            continue
        out.append(
            LigneDesignee(
                piece=piece,
                lignes=nums,
                lettre=lettre,
                score=score,
                source=objet.sorte,
            )
        )
    return out


def _pieces_des_objets(objets: Sequence[Objet]) -> tuple[str, ...]:
    pieces: list[str] = []
    vus: set[str] = set()
    for objet in objets:
        for ligne in objet.lignes:
            if not ligne.piece or ligne.piece in vus:
                continue
            vus.add(ligne.piece)
            pieces.append(ligne.piece)
    return tuple(pieces)


def matrice_depuis_cache(
    vecteurs_objets: Mapping[str, Sequence[float]],
    empreintes: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Matrice float32 L2-normalisée (lignes = empreintes) ; produit scalaire = cosinus.

    Une empreinte absente du cache est ignorée (avertissement journalisé).
    """
    if not empreintes:
        return np.zeros((0, 0), dtype=np.float32), []
    lignes: list[np.ndarray] = []
    ordre: list[str] = []
    dim = 0
    for empreinte in empreintes:
        brut = vecteurs_objets.get(empreinte)
        if brut is None:
            _JOURNAL.warning(
                "empreinte absente du cache d'embeddings, ignorée : %s",
                empreinte,
            )
            continue
        vecteur = np.asarray(brut, dtype=np.float32)
        dim = int(vecteur.shape[0]) if vecteur.size else dim
        norme = float(np.linalg.norm(vecteur))
        if norme > 0.0:
            vecteur = vecteur / norme
        lignes.append(vecteur)
        ordre.append(empreinte)
    if not lignes:
        return np.zeros((0, dim), dtype=np.float32), ordre
    return np.stack(lignes, axis=0), ordre


def selectionner_par_le_sens(
    question: str,
    index: IndexReconstitution,
    *,
    vecteur_question: Sequence[float],
    vecteurs_objets: Mapping[str, Sequence[float]],
    matrice_objets: tuple[np.ndarray, list[str]] | None = None,
    preparation_objets: PreparationObjets | None = None,
) -> Selection:
    """Note tous les objets du dossier par fusion RRF sens / mots / clés. Pas de vivier."""
    reconnaissance = lire_question(question, index)
    objets = objets_de_recherche(index)
    if not objets:
        return Selection(
            reconnaissance=reconnaissance,
            unites_candidates=_unites_candidates(
                reconnaissance, index, preparation=preparation_objets
            ),
            faits_classes=(),
            lignes_designees=(),
            pieces_designees=(),
            objets_classes=(),
            unites_absentes=len(index.absentes),
        )
    for objet in objets:
        if objet.empreinte not in vecteurs_objets:
            raise RuntimeError(
                f"embedding d'objet absent ({objet.id} {objet.empreinte[:12]}…)"
            )
    ids = [objet.id for objet in objets]
    par_id = {objet.id: objet for objet in objets}

    def _rangs(scores: Mapping[str, float]) -> dict[str, int]:
        """Même sémantique que _rangs_par_score, sans coût quadratique."""
        valeurs = {ident: float(scores.get(ident, 0.0)) for ident in ids}
        ordered = sorted(ids, key=lambda ident: (-valeurs[ident], ident))
        rangs: dict[str, int] = {}
        index = 0
        while index < len(ordered):
            score = valeurs[ordered[index]]
            fin = index + 1
            while fin < len(ordered) and valeurs[ordered[fin]] == score:
                fin += 1
            rang = 1 + index
            for pos in range(index, fin):
                rangs[ordered[pos]] = rang
            index = fin
        return rangs

    scores_sens: dict[str, float] = {}
    if matrice_objets is not None:
        matrice, empreintes_ordre = matrice_objets
        indice_par_empreinte = {emp: i for i, emp in enumerate(empreintes_ordre)}
        vecteur_q = np.asarray(vecteur_question, dtype=np.float32)
        norme_q = float(np.linalg.norm(vecteur_q))
        if norme_q > 0.0:
            vecteur_q = vecteur_q / norme_q
        for objet in objets:
            ligne = int(indice_par_empreinte[objet.empreinte])
            scores_sens[objet.id] = float(np.dot(matrice[ligne], vecteur_q))
    else:
        for objet in objets:
            scores_sens[objet.id] = cosine(vecteur_question, vecteurs_objets[objet.empreinte])
    rangs_sens = _rangs(scores_sens)

    if preparation_objets is not None:
        scores_mots = preparation_objets.bm25_objets.interroger(
            _tokeniser_bm25(reconnaissance)
        )
    else:
        documents = getattr(index, "_documents_bm25_cache", None)
        if documents is None:
            documents = {
                objet.id: _tokeniser_texte(objet.texte_recherche) for objet in objets
            }
            index._documents_bm25_cache = documents
        scores_mots = _bm25(_tokeniser_bm25(reconnaissance), documents)
    rangs_mots = _rangs(scores_mots)

    unites_bonus = _unites_bonus(
        reconnaissance, index, preparation=preparation_objets
    )
    scores_cles: dict[str, float] = {}
    if preparation_objets is not None:
        scores_lettre = scores_lettre_prepares(reconnaissance, preparation_objets)
        for objet in objets:
            score = float(scores_lettre.get(objet.id, 0))
            if objet.unite_id in unites_bonus:
                score += 3.0
            scores_cles[objet.id] = score
    else:
        for objet in objets:
            score = float(_score_lettre(reconnaissance, _lettre_objet(objet)))
            if objet.unite_id in unites_bonus:
                score += 3.0
            scores_cles[objet.id] = score
    rangs_cles = _rangs(scores_cles)

    scores_rrf = _fusion_rrf(rangs_sens, rangs_mots, rangs_cles, k=_HYBRID_RRF_K)
    ordered = sorted(
        objets,
        key=lambda objet: (
            -float(scores_rrf.get(objet.id, 0.0)),
            _priorite_desaccord(objet),
            objet.id,
        ),
    )
    objets_classes: list[ObjetClasse] = []
    faits: list[FaitClasse] = []
    lignes: list[LigneDesignee] = []
    for rang, objet in enumerate(ordered, start=1):
        score = float(scores_rrf.get(objet.id, 0.0))
        objets_classes.append(ObjetClasse(objet=objet, score=score, rang=rang))
        if objet.sorte == "fait":
            faits.append(
                FaitClasse(
                    fait=objet.payload or {},
                    score=int(scores_cles.get(objet.id, 0.0)),
                    rang=rang,
                    unite_id=objet.unite_id,
                )
            )
        lignes.extend(_lignes_designees_objet(objet, int(scores_cles.get(objet.id, 0.0))))
    pieces = _pieces_des_objets(ordered[:N_OBJETS_FILTRE])
    return Selection(
        reconnaissance=reconnaissance,
        unites_candidates=_unites_candidates(
            reconnaissance, index, preparation=preparation_objets
        ),
        faits_classes=tuple(faits),
        lignes_designees=tuple(lignes),
        pieces_designees=pieces,
        objets_classes=tuple(objets_classes),
        unites_absentes=len(index.absentes),
    )


def desaccords_en_tete(
    selection: Selection,
    *,
    n: int = N_DESACCORDS_EN_TETE,
) -> tuple[ObjetClasse, ...]:
    """Les n premiers faits ``desaccord`` ou ``non_ordonnable`` dans l'ordre du classement."""
    out: list[ObjetClasse] = []
    for note in selection.objets_classes:
        objet = note.objet
        if objet.sorte != "fait":
            continue
        if objet.comparaison not in COMPARAISONS_PRIORITAIRES:
            continue
        out.append(note)
        if len(out) >= n:
            break
    return tuple(out)
