# SPDX-License-Identifier: AGPL-3.0-only
"""Banc lecture Drive — registre, aiguillage, lecture de passage."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    from brain.lecture import BlockKind, Reading, read as lecture_read
except ImportError:
    BlockKind = None  # type: ignore[misc,assignment]
    Reading = None  # type: ignore[misc,assignment]
    lecture_read = None  # type: ignore[misc,assignment]
try:
    from brain.lecture.canonical import canonical_text_of, looks_scanned, split_canonical_lines
except ImportError:
    canonical_text_of = None  # type: ignore[misc,assignment]
    looks_scanned = None  # type: ignore[misc,assignment]
    split_canonical_lines = None  # type: ignore[misc,assignment]

MIME_DOSSIER = "application/vnd.google-apps.folder"
MIME_SHEET = "application/vnd.google-apps.spreadsheet"
MIME_DOC = "application/vnd.google-apps.document"
MIME_PDF = "application/pdf"
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MIME_CSV = "text/csv"
MIME_HTML = "text/html"
MIME_PNG = "image/png"
MIME_JPEG = "image/jpeg"
MIME_MESSAGE_RFC822 = "message/rfc822"

EXPORT_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
EXPORT_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

TRAITEMENTS_VALIDES = frozenset(
    {
        "reconstitution",
        "reconstitution_entite",
        "lecture_simple",
        "ocr_lecture_simple",
        "registre_seul",
    }
)

GROUPES_TRAITEMENT: dict[str, str] = {
    "reconstitution": "a-reconstituer",
    "reconstitution_entite": "a-reconstituer",
    "lecture_simple": "lecture-simple",
    "ocr_lecture_simple": "lecture-simple",
}

GROUPES_VERS_TOUS: tuple[str, ...] = ("a-reconstituer", "lecture-simple")

OcrImageFn = Callable[[bytes, str], str]


class TypeNonSupporte(ValueError):
    """Type de fichier non pris en charge par la lecture de passage."""


@dataclass
class Lecture:
    lignes: list[str]
    reperes: list[dict[str, Any]]
    ocr: bool
    pages: int | None
    avertissements: list[str]


class ClientDrive(Protocol):
    def lister(self, drive_id: str) -> list[dict[str, Any]]: ...

    def telecharger(self, file_id: str) -> bytes: ...

    def exporter(self, file_id: str, mime: str) -> bytes: ...


class ClientDriveHttp:
    """Client HTTP Drive v3 (urllib, sans journaliser le jeton)."""

    _BASE = "https://www.googleapis.com/drive/v3"

    def __init__(self, jeton: str) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise RuntimeError("appel Drive HTTP interdit sous pytest")
        if not jeton:
            raise ValueError("jeton Drive vide")
        self._jeton = jeton

    def lister(self, drive_id: str) -> list[dict[str, Any]]:
        champs = "nextPageToken,files(id,name,mimeType,size,parents,modifiedTime,md5Checksum,webViewLink)"
        resultat: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, str] = {
                "corpora": "drive",
                "driveId": drive_id,
                "includeItemsFromAllDrives": "true",
                "supportsAllDrives": "true",
                "pageSize": "100",
                "fields": champs,
                "q": "trashed=false",
            }
            if page_token:
                params["pageToken"] = page_token
            payload = self._get_json(f"{self._BASE}/files", params)
            resultat.extend(payload.get("files") or [])
            page_token = payload.get("nextPageToken")
            if not page_token:
                break
        return resultat

    def telecharger(self, file_id: str) -> bytes:
        params = {"alt": "media", "supportsAllDrives": "true"}
        return self._get_bytes(f"{self._BASE}/files/{file_id}", params)

    def exporter(self, file_id: str, mime: str) -> bytes:
        params = {"mimeType": mime, "supportsAllDrives": "true"}
        return self._get_bytes(f"{self._BASE}/files/{file_id}/export", params)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._jeton}"}

    def _get_json(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{url}?{query}", headers=self._headers())
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Drive API erreur {exc.code}") from exc

    def _get_bytes(self, url: str, params: dict[str, str]) -> bytes:
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{url}?{query}", headers=self._headers())
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Drive API erreur {exc.code}") from exc


def construire_registre(fichiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    par_id = {item["id"]: item for item in fichiers}
    dossiers = {fid for fid, item in par_id.items() if item.get("mimeType") == MIME_DOSSIER}

    def chemin_de(fid: str) -> str:
        parents = par_id[fid].get("parents") or []
        parent = parents[0] if parents else None
        if not parent or parent not in par_id:
            return ""
        if parent in dossiers:
            parent_path = chemin_de(parent)
            parent_name = par_id[parent]["name"]
            return f"{parent_path}/{parent_name}" if parent_path else parent_name
        return chemin_de(parent)

    entrees: list[dict[str, Any]] = []
    for item in fichiers:
        if item.get("mimeType") == MIME_DOSSIER:
            continue
        chemin = chemin_de(item["id"])
        entrees.append(
            {
                "id": item["id"],
                "nom": item.get("name") or "",
                "chemin": chemin,
                "mime": item.get("mimeType") or "",
                "taille": int(item.get("size") or 0),
                "modifie": item.get("modifiedTime") or "",
                "lien": item.get("webViewLink") or "",
                "md5": item.get("md5Checksum") or "",
            }
        )
    entrees.sort(key=lambda e: (e["chemin"].lower(), e["nom"].lower()))

    md5_vus: dict[str, str] = {}
    for entree in entrees:
        md5 = entree.get("md5") or ""
        if md5 and md5 in md5_vus:
            entree["doublon_de"] = md5_vus[md5]
        elif md5:
            md5_vus[md5] = entree["id"]
    return entrees


def charger_regles(chemin: Path) -> list[dict[str, Any]]:
    payload = json.loads(chemin.read_text(encoding="utf-8"))
    regles = payload.get("regles") or []
    for regle in regles:
        traitement = regle.get("traitement")
        if traitement not in TRAITEMENTS_VALIDES:
            raise ValueError(f"traitement inconnu: {traitement!r}")
    return regles


def aiguiller(registre: list[dict[str, Any]], regles: list[dict[str, Any]]) -> None:
    for entree in registre:
        if entree.get("doublon_de"):
            entree["famille"] = entree.get("famille", "")
            entree["traitement"] = "registre_seul"
            entree["raison"] = f"doublon de {entree['doublon_de']}"
            entree["regle"] = 0
            continue
        for rang, regle in enumerate(regles, start=1):
            if _regle_correspond(entree, regle):
                entree["famille"] = regle.get("famille") or ""
                entree["traitement"] = regle["traitement"]
                entree["raison"] = regle.get("raison") or ""
                entree["regle"] = rang
                break
        else:
            entree.setdefault("famille", "")
            entree["traitement"] = "a_trancher"
            entree["raison"] = ""
            entree["regle"] = 0


def fusionner_registre(
    ancien: list[dict[str, Any]],
    nouveau: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    par_id = {entree["id"]: entree for entree in ancien}
    ids_nouveaux = {entree["id"] for entree in nouveau}
    disparues = [entree for entree in ancien if entree["id"] not in ids_nouveaux]
    fusionne: list[dict[str, Any]] = []
    for entree in nouveau:
        copie = dict(entree)
        precedente = par_id.get(entree["id"])
        if precedente and precedente.get("modifie") == entree.get("modifie"):
            if "piece" in precedente:
                copie["piece"] = precedente["piece"]
            if "groupe" in precedente:
                copie["groupe"] = precedente["groupe"]
            if "lecture" in precedente:
                copie["lecture"] = precedente["lecture"]
        fusionne.append(copie)
    return fusionne, disparues


def _groupe_traitement(traitement: str) -> str | None:
    return GROUPES_TRAITEMENT.get(traitement)


def _base_et_piece(entree: dict[str, Any], rang: int) -> tuple[str, str]:
    """Nom de pièce stable : réutilise le registre, sinon rang + haché Drive id."""
    piece = entree.get("piece")
    if isinstance(piece, str) and piece.endswith(".md"):
        return piece[:-3], piece
    file_id = str(entree.get("id") or "")
    id_tag = hashlib.sha256(file_id.encode("utf-8")).hexdigest()[:6]
    base = f"{rang:03d}-{id_tag}-{_slug(entree.get('nom') or 'fichier')}"
    return base, f"{base}.md"


def _empreinte_fichier(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _destination_pieces_tous_a_jour(dest: Path, src: Path) -> bool:
    if not dest.is_file():
        return False
    try:
        if dest.samefile(src):
            return True
    except OSError:
        pass
    return _empreinte_fichier(dest) == _empreinte_fichier(src)


def _poser_lien_ou_copie_pieces_tous(src: Path, dest: Path) -> None:
    import shutil

    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def actualiser_pieces_tous(cache: Path) -> None:
    """Synchronise ``pieces/tous`` sur l'union des groupes (lien dur ou copie identique)."""
    tous = cache / "pieces" / "tous"
    tous.mkdir(parents=True, exist_ok=True)
    attendus: dict[str, Path] = {}
    for groupe in GROUPES_VERS_TOUS:
        src_dir = cache / "pieces" / groupe
        if not src_dir.is_dir():
            continue
        for src in sorted(src_dir.glob("*.md")):
            attendus[src.name] = src
    for dest in sorted(tous.glob("*.md")):
        if dest.name not in attendus:
            dest.unlink()
    for nom, src in sorted(attendus.items()):
        dest = tous / nom
        if _destination_pieces_tous_a_jour(dest, src):
            continue
        if dest.exists():
            dest.unlink()
        _poser_lien_ou_copie_pieces_tous(src, dest)


def _retirer_piece_cache(sortie_dir: Path, entree: dict[str, Any]) -> None:
    groupe = entree.get("groupe")
    piece = entree.get("piece")
    if not groupe or not piece:
        return
    cache = sortie_dir / "cache-lecture"
    piece_path = cache / "pieces" / groupe / piece
    if piece_path.is_file():
        piece_path.unlink()
    base = piece[: -len(".md")] if piece.endswith(".md") else piece
    repere_path = cache / "reperes" / groupe / f"{base}.json"
    if repere_path.is_file():
        repere_path.unlink()


def _regle_correspond(entree: dict[str, Any], regle: dict[str, Any]) -> bool:
    for cle, champ in (("chemin", "chemin"), ("nom", "nom"), ("mime", "mime")):
        motif = regle.get(cle)
        if not motif:
            continue
        if not re.search(motif, entree.get(champ) or "", re.IGNORECASE):
            return False
    return True


def lire_fichier(
    entree: dict[str, Any],
    client: ClientDrive,
    *,
    ocr_image: OcrImageFn | None = None,
) -> Lecture:
    mime = entree.get("mime") or ""
    nom = entree.get("nom") or ""
    file_id = entree["id"]

    if mime == MIME_PDF or nom.lower().endswith(".pdf"):
        data = client.telecharger(file_id)
        return _lire_pdf_bytes(data)

    if mime == MIME_SHEET:
        data = client.exporter(file_id, EXPORT_XLSX)
        return _lire_xlsx_bytes(data)

    if mime == MIME_XLSX or nom.lower().endswith(".xlsx"):
        data = client.telecharger(file_id)
        return _lire_xlsx_bytes(data)

    if mime == MIME_CSV or nom.lower().endswith(".csv"):
        data = client.telecharger(file_id)
        return _lire_csv_bytes(data)

    if mime == MIME_DOC:
        data = client.exporter(file_id, EXPORT_DOCX)
        return _lire_docx_bytes(data)

    if mime == MIME_DOCX or nom.lower().endswith(".docx"):
        data = client.telecharger(file_id)
        return _lire_docx_bytes(data)

    if mime == MIME_HTML or nom.lower().endswith((".html", ".htm")):
        data = client.telecharger(file_id)
        return _lire_html_bytes(data)

    if mime in (MIME_PNG, MIME_JPEG) or nom.lower().endswith((".png", ".jpg", ".jpeg")):
        data = client.telecharger(file_id)
        return _lire_image_bytes(data, nom, ocr_image=ocr_image)

    raise TypeNonSupporte(f"type non supporté: {mime or nom}")


def lire_dossier(
    registre: list[dict[str, Any]],
    client: ClientDrive,
    sortie_dir: Path,
    *,
    refaire: bool = False,
    seulement: set[str] | None = None,
    ocr_image: OcrImageFn | None = None,
) -> None:
    cache = sortie_dir / "cache-lecture"
    registre_path = sortie_dir / "registre.json"

    for rang, entree in enumerate(registre, start=1):
        if (entree.get("mime") or "") == MIME_MESSAGE_RFC822:
            continue
        traitement = entree.get("traitement") or ""
        if traitement in ("registre_seul", "a_trancher"):
            continue
        groupe = _groupe_traitement(traitement)
        if not groupe:
            continue
        if seulement and traitement not in seulement:
            continue
        if not refaire and _deja_lu(entree, cache):
            continue

        pieces_dir = cache / "pieces" / groupe
        reperes_dir = cache / "reperes" / groupe
        pieces_dir.mkdir(parents=True, exist_ok=True)
        reperes_dir.mkdir(parents=True, exist_ok=True)

        base, piece_nom = _base_et_piece(entree, rang)
        piece_path = pieces_dir / piece_nom
        repere_path = reperes_dir / f"{base}.json"

        try:
            lecture = lire_fichier(entree, client, ocr_image=ocr_image)
            texte = "\n".join(lecture.lignes)
            if texte and not texte.endswith("\n"):
                texte += "\n"
            piece_path.write_text(texte, encoding="utf-8")
            repere_path.write_text(
                json.dumps(lecture.reperes, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            entree["piece"] = piece_nom
            entree["groupe"] = groupe
            empreinte_manifeste = entree.get("manifeste_sha256")
            entree["lecture"] = {
                "lignes": len(lecture.lignes),
                "caracteres": len(texte.rstrip("\n")),
                "pages": lecture.pages,
                "ocr": lecture.ocr,
                "avertissements": lecture.avertissements,
                "empreinte": empreinte_manifeste
                or hashlib.sha256(texte.encode("utf-8")).hexdigest(),
                "modifie": entree.get("modifie") or "",
            }
        except Exception as exc:
            entree["lecture"] = {"erreur": f"{type(exc).__name__}: {exc}"}
            entree.pop("piece", None)
        ecrire_json_atomique(registre_path, registre)

    actualiser_pieces_tous(cache)


def purger(sortie_dir: Path) -> None:
    cache = sortie_dir / "cache-lecture"
    if cache.is_dir():
        import shutil

        shutil.rmtree(cache)
    registre_path = sortie_dir / "registre.json"
    if registre_path.is_file():
        registre = json.loads(registre_path.read_text(encoding="utf-8"))
        for entree in registre:
            entree.pop("piece", None)
            entree.pop("groupe", None)
            entree.pop("lecture", None)
        ecrire_json_atomique(registre_path, registre)


def compte_traitements(registre: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    par_traitement: dict[str, dict[str, int]] = {}
    par_famille: dict[str, dict[str, int]] = {}
    for entree in registre:
        traitement = entree.get("traitement") or "?"
        famille = entree.get("famille") or "?"
        taille = int(entree.get("taille") or 0)
        bloc_t = par_traitement.setdefault(traitement, {"fichiers": 0, "volume": 0})
        bloc_t["fichiers"] += 1
        bloc_t["volume"] += taille
        bloc_f = par_famille.setdefault(famille, {"fichiers": 0, "volume": 0})
        bloc_f["fichiers"] += 1
        bloc_f["volume"] += taille
    return {"traitement": par_traitement, "famille": par_famille}


def compte_lecture(registre: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    par_famille: dict[str, dict[str, int]] = {}
    for entree in registre:
        famille = entree.get("famille") or "?"
        bloc = par_famille.setdefault(
            famille,
            {
                "fichiers_lus": 0,
                "lignes": 0,
                "caracteres": 0,
                "ocr": 0,
                "avertissements": 0,
                "erreurs": 0,
            },
        )
        lecture = entree.get("lecture")
        if not isinstance(lecture, dict):
            continue
        if "erreur" in lecture:
            bloc["erreurs"] += 1
            continue
        bloc["fichiers_lus"] += 1
        bloc["lignes"] += int(lecture.get("lignes") or 0)
        bloc["caracteres"] += int(lecture.get("caracteres") or 0)
        if lecture.get("ocr"):
            bloc["ocr"] += 1
        bloc["avertissements"] += len(lecture.get("avertissements") or [])
    return par_famille


def compte_lecture_par_groupe(
    registre: list[dict[str, Any]],
    sortie_dir: Path,
) -> dict[str, dict[str, Any]]:
    par_groupe: dict[str, dict[str, Any]] = {}
    for entree in registre:
        groupe = entree.get("groupe")
        if not groupe:
            continue
        lecture = entree.get("lecture")
        if not isinstance(lecture, dict) or "erreur" in lecture:
            continue
        bloc = par_groupe.setdefault(
            groupe,
            {"pieces": 0, "lignes": 0, "caracteres": 0},
        )
        bloc["pieces"] += 1
        bloc["lignes"] += int(lecture.get("lignes") or 0)
        bloc["caracteres"] += int(lecture.get("caracteres") or 0)
    cache = sortie_dir / "cache-lecture" / "pieces"
    for groupe, bloc in par_groupe.items():
        bloc["pieces_dir"] = str(cache / groupe)
    return par_groupe


def _deja_lu(entree: dict[str, Any], cache: Path) -> bool:
    piece = entree.get("piece")
    groupe = entree.get("groupe") or _groupe_traitement(entree.get("traitement") or "")
    lecture = entree.get("lecture")
    if not piece or not groupe or not isinstance(lecture, dict) or "erreur" in lecture:
        return False
    if lecture.get("modifie") != entree.get("modifie"):
        return False
    manifeste_sha = entree.get("manifeste_sha256")
    if manifeste_sha and lecture.get("empreinte") != manifeste_sha:
        return False
    return (cache / "pieces" / groupe / piece).is_file()


def ecrire_json_atomique(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _slug(nom: str) -> str:
    base = Path(nom).stem
    normalise = unicodedata.normalize("NFKD", base)
    ascii_only = "".join(ch for ch in normalise if not unicodedata.combining(ch))
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return slug or "fichier"


def _echapper_cellule(texte: str) -> str:
    return texte.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _lignes_tableau_markdown(lignes_brutes: list[list[str]]) -> list[str]:
    lignes, _ = _tableau_markdown_avec_reperes(
        [(index, ligne) for index, ligne in enumerate(lignes_brutes, start=1)]
    )
    return lignes


def _tableau_markdown_avec_reperes(
    rangees_cellules: list[tuple[int, list[str]]],
    *,
    feuille: str | None = None,
    tableau: int | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    non_vides = [
        (numero, cellules)
        for numero, cellules in rangees_cellules
        if any(cellule.strip() for cellule in cellules)
    ]
    if not non_vides:
        return [], []

    entete_numero, entete = non_vides[0]
    cols = len(entete)
    sep = "| " + " | ".join("---" for _ in range(cols)) + " |"
    lignes = [
        "| " + " | ".join(_echapper_cellule(cellule) for cellule in entete) + " |",
        sep,
    ]
    reperes: list[dict[str, Any]] = []

    def repere(rangee: int | None, separateur: bool = False) -> dict[str, Any]:
        rep: dict[str, Any] = {}
        if feuille is not None:
            rep["feuille"] = feuille
        if tableau is not None:
            rep["tableau"] = tableau
        rep["rangee"] = rangee
        if separateur and feuille is not None:
            rep["separateur"] = True
        return rep

    reperes.append(repere(entete_numero))
    reperes.append(repere(None, separateur=feuille is not None))

    for numero, cellules in non_vides[1:]:
        padded = list(cellules) + [""] * (cols - len(cellules))
        lignes.append(
            "| " + " | ".join(_echapper_cellule(cellule) for cellule in padded[:cols]) + " |"
        )
        reperes.append(repere(numero))

    return lignes, reperes


def _avertissements_lecture(
    texte: str,
    pages: int | None,
    warnings: tuple[str, ...] | list[str] = (),
) -> list[str]:
    avertissements = list(warnings)
    corps = texte.strip()
    if not corps:
        avertissements.append("texte vide")
    if pages and pages > 1 and len(corps) < 200:
        avertissements.append("moins de 200 caractères pour un document de plus d'une page")
    return avertissements


def _lire_pdf_bytes(data: bytes) -> Lecture:
    ocr = _pdf_utilisera_ocr(data)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        reading = lecture_read(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    canon = canonical_text_of(reading.blocks)
    lignes_canon = split_canonical_lines(canon)
    lignes = [line.content for line in lignes_canon]
    reperes = [{"page": line.page, "ligne": line.number} for line in lignes_canon]
    pages = max((line.page for line in lignes_canon), default=1) if lignes_canon else 1
    avertissements = _avertissements_lecture(canon, pages, reading.warnings)
    return Lecture(lignes=lignes, reperes=reperes, ocr=ocr, pages=pages, avertissements=avertissements)


def _pdf_utilisera_ocr(data: bytes) -> bool:
    try:
        from brain.lecture.canonical import pdftotext
    except ImportError:
        pdftotext = None  # type: ignore[misc,assignment]

    try:
        canon = pdftotext(data)
    except RuntimeError:
        return False
    n_pages = _pdf_page_count(data)
    return looks_scanned(canon, n_pages=n_pages) and _pdf_has_images(data)


def _pdf_page_count(data: bytes) -> int:
    try:
        from io import BytesIO

        from pypdf import PdfReader

        return max(len(PdfReader(BytesIO(data)).pages), 1)
    except Exception:
        return 1


def _pdf_has_images(data: bytes) -> bool:
    try:
        from io import BytesIO

        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        for page in reader.pages:
            resources = page.get("/Resources")
            if resources is None:
                continue
            xobjects = resources.get("/XObject")
            if not xobjects:
                continue
            for name in xobjects:
                try:
                    xobj = xobjects[name]
                except Exception:
                    continue
                subtype = xobj.get("/Subtype") if xobj is not None else None
                if str(subtype) in ("/Image", "Image"):
                    return True
        return False
    except Exception:
        return True


def _lire_xlsx_bytes(data: bytes) -> Lecture:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        reading = lecture_read(tmp_path)
        return _reading_vers_tableaux_markdown(reading)
    finally:
        tmp_path.unlink(missing_ok=True)


def _reading_vers_tableaux_markdown(reading: Reading) -> Lecture:
    lignes: list[str] = []
    reperes: list[dict[str, Any]] = []
    feuille_courante = ""
    grille: dict[int, dict[int, str]] = {}

    def vider_feuille() -> None:
        nonlocal grille, feuille_courante
        if not feuille_courante:
            return
        if lignes and lignes[-1] != "":
            lignes.append("")
        lignes.append(f"## {feuille_courante}")
        reperes.append({"feuille": feuille_courante, "rangee": 0})
        rows = sorted(grille)
        rangees_cellules: list[tuple[int, list[str]]] = []
        for row_i in rows:
            cols = grille[row_i]
            max_col = max(cols)
            rangees_cellules.append(
                (row_i, [cols.get(c, "") for c in range(1, max_col + 1)])
            )
        table_lignes, table_reperes = _tableau_markdown_avec_reperes(
            rangees_cellules,
            feuille=feuille_courante,
        )
        lignes.extend(table_lignes)
        reperes.extend(table_reperes)
        grille = {}

    for block in reading.blocks:
        if block.kind == BlockKind.HEADING:
            titre = block.text.strip()
            if titre and titre != feuille_courante:
                vider_feuille()
                feuille_courante = titre
        elif block.kind == BlockKind.CELL and block.sheet and block.row and block.column:
            if block.sheet != feuille_courante:
                vider_feuille()
                feuille_courante = block.sheet
            grille.setdefault(block.row, {})[block.column] = block.text

    vider_feuille()
    texte = "\n".join(lignes)
    avertissements = _avertissements_lecture(texte, 1, reading.warnings)
    return Lecture(lignes=lignes, reperes=reperes, ocr=False, pages=1, avertissements=avertissements)


def _lire_csv_bytes(data: bytes) -> Lecture:
    texte = data.decode("utf-8-sig", errors="replace")
    delim = ";" if texte.count(";") > texte.count(",") else ","
    reader = csv.reader(io.StringIO(texte), delimiter=delim)
    rangees_cellules = [
        (index, [cellule for cellule in ligne]) for index, ligne in enumerate(reader, start=1)
    ]
    lignes_table, reperes = _tableau_markdown_avec_reperes(rangees_cellules)
    texte_out = "\n".join(lignes_table)
    return Lecture(
        lignes=lignes_table,
        reperes=reperes,
        ocr=False,
        pages=1,
        avertissements=_avertissements_lecture(texte_out, 1),
    )


def _lire_docx_bytes(data: bytes) -> Lecture:
    from docx import Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        doc = Document(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    lignes: list[str] = []
    reperes: list[dict[str, Any]] = []
    paragraphe = 0
    tableau = 0

    for child in doc.element.body:
        if isinstance(child, CT_P):
            paragraphe += 1
            texte = Paragraph(child, doc).text.strip()
            if texte:
                lignes.append(texte)
                reperes.append({"paragraphe": paragraphe})
        elif isinstance(child, CT_Tbl):
            tableau += 1
            table = Table(child, doc)
            rangees_cellules = [
                (index, [cellule.text.strip() for cellule in row.cells])
                for index, row in enumerate(table.rows, start=1)
            ]
            table_lignes, table_reperes = _tableau_markdown_avec_reperes(
                rangees_cellules,
                tableau=tableau,
            )
            lignes.extend(table_lignes)
            reperes.extend(table_reperes)

    texte = "\n".join(lignes)
    return Lecture(
        lignes=lignes,
        reperes=reperes,
        ocr=False,
        pages=1,
        avertissements=_avertissements_lecture(texte, 1),
    )


def _lire_html_bytes(data: bytes) -> Lecture:
    class Extracteur(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.blocs: list[str] = []
            self._ignore = False
            self._table_rows: list[list[str]] = []
            self._current_row: list[str] = []
            self._cell_parts: list[str] = []
            self._in_table = False
            self._in_cell = False

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            tag = tag.lower()
            if tag in ("script", "style"):
                self._ignore = True
            elif tag == "table":
                self._in_table = True
                self._table_rows = []
            elif tag == "tr" and self._in_table:
                self._current_row = []
            elif tag in ("td", "th") and self._in_table:
                self._in_cell = True
                self._cell_parts = []

        def handle_endtag(self, tag: str) -> None:
            tag = tag.lower()
            if tag in ("script", "style"):
                self._ignore = False
            elif tag in ("td", "th") and self._in_cell:
                self._current_row.append("".join(self._cell_parts).strip())
                self._in_cell = False
            elif tag == "tr" and self._in_table:
                if self._current_row:
                    self._table_rows.append(self._current_row)
            elif tag == "table" and self._in_table:
                for table_ligne in _lignes_tableau_markdown(self._table_rows):
                    self.blocs.append(table_ligne)
                self._in_table = False

        def handle_data(self, data: str) -> None:
            if self._ignore:
                return
            if self._in_cell:
                self._cell_parts.append(data)
                return
            if self._in_table:
                return
            texte = data.strip()
            if texte:
                self.blocs.append(texte)

    parser = Extracteur()
    parser.feed(data.decode("utf-8", errors="replace"))
    lignes = parser.blocs
    reperes = [{"bloc": index} for index in range(1, len(lignes) + 1)]
    texte = "\n".join(lignes)
    return Lecture(
        lignes=lignes,
        reperes=reperes,
        ocr=False,
        pages=1,
        avertissements=_avertissements_lecture(texte, 1),
    )


def _lire_image_bytes(
    data: bytes,
    nom: str,
    *,
    ocr_image: OcrImageFn | None = None,
) -> Lecture:
    if ocr_image is not None:
        texte = ocr_image(data, nom)
    else:
        suffix = Path(nom).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
            proc = subprocess.run(
                ["tesseract", str(tmp_path), "stdout", "-l", "fra+eng", "--oem", "1", "--psm", "3"],
                capture_output=True,
                check=False,
                env=env,
            )
            texte = proc.stdout.decode("utf-8", errors="replace")
        finally:
            tmp_path.unlink(missing_ok=True)

    lignes = [ligne for ligne in texte.splitlines() if ligne.strip()]
    if not lignes and texte.strip():
        lignes = [texte.strip()]
    reperes = [{"ligne": index} for index in range(1, len(lignes) + 1)]
    return Lecture(
        lignes=lignes,
        reperes=reperes,
        ocr=True,
        pages=1,
        avertissements=_avertissements_lecture("\n".join(lignes), 1),
    )
