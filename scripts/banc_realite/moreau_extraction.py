# SPDX-License-Identifier: AGPL-3.0-only
"""Extraction et contrôle de fidélité du dossier juridique MOREAU (banc comparatif)."""
from __future__ import annotations

import argparse
import email
import hashlib
import json
import re
import subprocess
import unicodedata
from datetime import datetime
from email import policy
from pathlib import Path
from typing import Any

import yaml

BANC_ROOT = Path.home() / ".agents" / "var" / "banc-realite"
DOSSIER_ROOT = BANC_ROOT / "dossiers" / "moreau"
SOURCE_DIR = DOSSIER_ROOT / "source"
DOCS_DIR = DOSSIER_ROOT / "documents"
GOLDEN_PATH = SOURCE_DIR / "GOLDEN-faits.yaml"
MANIFEST_PATH = SOURCE_DIR / "MANIFEST.sha256"
EXTRACTION_MD = DOSSIER_ROOT / "EXTRACTION.md"
EXTRACTION_JSON = DOSSIER_ROOT / "extraction.json"
INVENTAIRE_MD = DOSSIER_ROOT / "INVENTAIRE.md"
MANIFEST_REPORT = DOSSIER_ROOT / "MANIFEST-CONTROLE.md"

_NBSP_RE = re.compile(r"[\u00a0\u202f]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SIREN_RE = re.compile(r"749\s*382\s*615")
_PORTE_RE = re.compile(r"la porte est large", re.IGNORECASE)
_LUMO_RE = re.compile(r"\bLUMO\b")
_NUMBER_RE = re.compile(r"\b\d[\d\s.,/]*\d\b|\b\d+\b")
_MONTHS_FR = {
    1: "janvier",
    2: "février",
    3: "mars",
    4: "avril",
    5: "mai",
    6: "juin",
    7: "juillet",
    8: "août",
    9: "septembre",
    10: "octobre",
    11: "novembre",
    12: "décembre",
}
_DATE_HEAD_RE = re.compile(
    r"(?:\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b"
    r"|(?:\d{4}-\d{2}-\d{2})"
    r"|(?:\d{2}/\d{2}/\d{4}))",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SLASH_DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_NUMBER_SPAN_RE = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[\s\u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?(?!\d)"
)


def normalize_extraction_text(text: str) -> str:
    """Normalise espaces insécables, césures de fin de ligne et lignes vides multiples."""
    if not text:
        return ""
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = _NBSP_RE.sub(" ", value)
    value = _HYPHEN_BREAK_RE.sub(r"\1\2", value)
    value = _BLANK_LINES_RE.sub("\n\n", value)
    return value.strip() + "\n"


def _normalize_for_match(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = _NBSP_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _parse_numeric_value(raw: str) -> float | None:
    token = _NBSP_RE.sub(" ", raw.strip())
    if not token:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        return None
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", token):
        return None
    if "/" in token:
        return None
    if any(month in token.lower() for month in _MONTHS_FR.values()):
        return None
    cleaned = re.sub(r"[^\d.,\s\u00a0\u202f]", "", token)
    cleaned = _NBSP_RE.sub("", cleaned).replace(" ", "")
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        left, right = cleaned.split(",", 1)
        cleaned = f"{left}.{right}" if len(right) <= 2 else cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_numeric_token(token: str) -> bool:
    return _parse_numeric_value(token) is not None


def _numeric_values_in_text(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_SPAN_RE.finditer(text):
        parsed = _parse_numeric_value(match.group(0))
        if parsed is not None:
            values.append(parsed)
    return values


def _numeric_match(token: str, text: str) -> bool:
    target = _parse_numeric_value(token)
    if target is None:
        return False
    target_int = int(target) if target.is_integer() else None
    for value in _numeric_values_in_text(text):
        if target_int is not None and value.is_integer() and int(value) == target_int:
            return True
        if abs(value - target) < 0.001:
            return True
    return False


def _text_match(token: str, text: str) -> bool:
    needle = _normalize_for_match(token)
    if not needle:
        return False
    if needle in _normalize_for_match(text):
        return True
    compact_needle = re.sub(r"\s+", "", needle)
    compact_haystack = re.sub(r"\s+", "", _normalize_for_match(text))
    return compact_needle in compact_haystack


def token_in_text(token: str, text: str) -> bool:
    """Recherche un jeton dans un texte (API publique pour moreau_or)."""
    if _is_numeric_token(token):
        return _numeric_match(token, text)
    return _text_match(token, text)


def extract_pdf_layout(path: Path) -> str:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"pdftotext a échoué pour {path.name}: {stderr}")
    return normalize_extraction_text(proc.stdout.decode("utf-8", "replace"))


def extract_eml_text(path: Path) -> str:
    msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    headers: list[str] = []
    for key in ("Date", "From", "To", "Subject"):
        value = msg.get(key)
        if value:
            headers.append(f"{key}: {value}")
    plains: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype.startswith("multipart/"):
                continue
            disp = str(part.get("Content-Disposition") or "")
            if ctype == "text/plain" and "attachment" not in disp.lower():
                payload = part.get_content()
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    payload = payload.decode(charset, "replace")
                plains.append(str(payload).strip())
    else:
        payload = msg.get_content()
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", "replace")
        plains.append(str(payload).strip())
    out = "\n".join(headers)
    if out:
        out += "\n\n"
    out += "\n\n".join(plains)
    return normalize_extraction_text(out)


def piece_to_markdown_name(piece: str) -> str:
    return Path(piece).with_suffix(".md").name


def _sha256_prefix(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def verify_manifest(manifest_path: Path = MANIFEST_PATH, source_dir: Path = SOURCE_DIR) -> dict[str, Any]:
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    results: list[dict[str, Any]] = []
    ok = True
    covered: set[str] = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            ok = False
            results.append(
                {"path": line, "status": "malformed", "expected": None, "actual": None}
            )
            continue
        expected_prefix, rel_path = parts
        covered.add(rel_path)
        file_path = source_dir / rel_path
        if not file_path.is_file():
            ok = False
            results.append(
                {"path": rel_path, "status": "absent", "expected": expected_prefix, "actual": None}
            )
            continue
        digest = _sha256_prefix(file_path)
        status = "ok" if digest == expected_prefix else "mismatch"
        if status != "ok":
            ok = False
        results.append(
            {"path": rel_path, "status": status, "expected": expected_prefix, "actual": digest}
        )
    pieces_dir = source_dir / "pieces"
    if pieces_dir.is_dir():
        for piece in sorted(pieces_dir.iterdir()):
            if not piece.is_file():
                continue
            if piece.suffix.lower() not in {".pdf", ".eml"}:
                continue
            rel_path = str(piece.relative_to(source_dir))
            if rel_path in covered:
                continue
            ok = False
            results.append(
                {
                    "path": rel_path,
                    "status": "non-couverte",
                    "expected": None,
                    "actual": _sha256_prefix(piece),
                }
            )
    if not results:
        ok = False
    return {"ok": ok, "count": len(results), "results": results}


def write_manifest_report(report: dict[str, Any], out_path: Path = MANIFEST_REPORT) -> None:
    lines = [
        "# Contrôle MANIFEST.sha256 — dossier MOREAU",
        "",
        f"Statut global : **{'OK' if report['ok'] else 'ÉCART'}** ({report['count']} fichiers)",
        "",
        "| Fichier | Attendu | Obtenu | Statut |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["results"]:
        lines.append(
            f"| `{row['path']}` | `{row['expected']}` | `{row['actual'] or '—'}` | {row['status']} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_all(source_dir: Path = SOURCE_DIR, docs_dir: Path = DOCS_DIR) -> list[Path]:
    pieces_dir = source_dir / "pieces"
    docs_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        piece.with_suffix(".md").name
        for piece in sorted(pieces_dir.iterdir())
        if piece.is_file() and piece.suffix.lower() in {".pdf", ".eml"}
    }
    for stale in docs_dir.glob("*.md"):
        if stale.name not in expected:
            stale.unlink()
    written: list[Path] = []
    for piece in sorted(pieces_dir.iterdir()):
        if not piece.is_file():
            continue
        out = docs_dir / piece.with_suffix(".md").name
        if piece.suffix.lower() == ".eml":
            text = extract_eml_text(piece)
        elif piece.suffix.lower() == ".pdf":
            text = extract_pdf_layout(piece)
        else:
            continue
        out.write_text(text, encoding="utf-8")
        written.append(out)
    return written


def _date_variants(iso_date: str) -> list[str]:
    try:
        dt = datetime.strptime(iso_date, "%Y-%m-%d")
    except ValueError:
        return [iso_date]
    month = _MONTHS_FR[dt.month]
    french = f"{dt.day}er {month} {dt.year}" if dt.day == 1 else f"{dt.day} {month} {dt.year}"
    return [
        iso_date,
        f"{dt.day:02d}/{dt.month:02d}/{dt.year}",
        french,
    ]


def _amount_variants(amount: int | float) -> list[str]:
    whole = int(amount)
    tokens = [str(whole)]
    if float(amount) != whole:
        tokens.append(f"{amount:.2f}".replace(".", ","))
    spaced = f"{whole:,}".replace(",", " ")
    if spaced not in tokens:
        tokens.append(spaced)
    if whole >= 1000:
        tokens.append(f"{whole // 1000} {whole % 1000:03d}".replace(" 0", " ").strip())
    return tokens


def derive_discriminant_tokens(fact: dict[str, Any], meta: dict[str, Any] | None = None) -> list[str]:
    tokens: list[str] = []
    if fact.get("date"):
        tokens.extend(_date_variants(str(fact["date"])))
    if fact.get("montant_eur") is not None:
        tokens.extend(_amount_variants(fact["montant_eur"]))
    if fact.get("quantite") is not None:
        tokens.append(str(fact["quantite"]))
    resume = str(fact.get("resume") or "")
    if _SIREN_RE.search(resume):
        tokens.append("749 382 615")
    if _PORTE_RE.search(resume):
        tokens.append("la porte est large")
    if _LUMO_RE.search(resume):
        tokens.append("LUMO")
    if meta and fact.get("id") == "F15" and meta.get("rg"):
        tokens.append(str(meta["rg"]))
    for match in _NUMBER_RE.findall(resume):
        cleaned = re.sub(r"\s+", " ", match.strip())
        digits = re.sub(r"\D", "", cleaned)
        if len(digits) >= 3 and cleaned and cleaned not in tokens:
            tokens.append(cleaned)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(token)
    return deduped


def load_corpus(docs_dir: Path = DOCS_DIR) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(docs_dir.glob("*.md"))}


def _is_date_token(token: str) -> bool:
    if _ISO_DATE_RE.match(token) or _SLASH_DATE_RE.match(token):
        return True
    lowered = token.lower()
    return any(month in lowered for month in _MONTHS_FR.values())


def _fact_verified(token_results: list[dict[str, Any]]) -> bool:
    if not token_results:
        return False
    if not any(row["status"] == "verifie" for row in token_results):
        return False
    date_ok = any(
        row["status"] == "verifie" for row in token_results if _is_date_token(row["token"])
    )
    for row in token_results:
        status = row["status"]
        if status == "verifie":
            continue
        if status == "hors-piece":
            if _is_date_token(row["token"]) and date_ok:
                continue
            return False
        if status == "absent":
            if _is_date_token(row["token"]) and date_ok:
                continue
            return False
        return False
    return True


def search_token(
    token: str,
    corpus: dict[str, str],
    announced: list[str],
) -> dict[str, Any]:
    announced_md = [piece_to_markdown_name(p) for p in announced]
    found_announced = [name for name in announced_md if token_in_text(token, corpus.get(name, ""))]
    if found_announced:
        return {"status": "verifie", "pieces": found_announced}
    found_elsewhere = [
        name
        for name, text in corpus.items()
        if name not in announced_md and token_in_text(token, text)
    ]
    if found_elsewhere:
        return {"status": "hors-piece", "pieces": found_elsewhere}
    return {"status": "absent", "pieces": []}


def verify_facts(
    golden_path: Path = GOLDEN_PATH,
    docs_dir: Path = DOCS_DIR,
) -> dict[str, Any]:
    meta = yaml.safe_load(golden_path.read_text(encoding="utf-8"))
    corpus = load_corpus(docs_dir)
    rows: list[dict[str, Any]] = []
    facts_ok = 0
    for fact in meta.get("facts", []):
        tokens = derive_discriminant_tokens(fact, meta)
        token_results: list[dict[str, Any]] = []
        for token in tokens:
            result = search_token(token, corpus, fact.get("pieces", []))
            token_results.append({"token": token, **result})
        fact_ok = _fact_verified(token_results)
        if fact_ok:
            facts_ok += 1
        rows.append(
            {
                "id": fact.get("id"),
                "resume": fact.get("resume"),
                "pieces": fact.get("pieces", []),
                "tokens": token_results,
                "ok": fact_ok and bool(tokens),
            }
        )
    total = len(meta.get("facts", []))
    return {
        "affaire": meta.get("affaire"),
        "rg": meta.get("rg"),
        "facts_total": total,
        "facts_verified": facts_ok,
        "facts": rows,
    }


def write_extraction_report(report: dict[str, Any]) -> None:
    lines = [
        "# Contrôle d'extraction — dossier MOREAU",
        "",
        f"Faits vérifiés : **{report['facts_verified']}/{report['facts_total']}**",
        "",
        "| Fait | Jeton | Statut | Pièce(s) |",
        "| --- | --- | --- | --- |",
    ]
    for fact in report["facts"]:
        for token_row in fact["tokens"]:
            pieces = ", ".join(f"`{p}`" for p in token_row["pieces"]) or "—"
            lines.append(
                f"| {fact['id']} | `{token_row['token']}` | {token_row['status']} | {pieces} |"
            )
    EXTRACTION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    EXTRACTION_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def detect_document_date(text: str) -> str | None:
    head = text[:800]
    match = _DATE_HEAD_RE.search(head)
    return match.group(0) if match else None


def build_inventory(docs_dir: Path = DOCS_DIR, out_path: Path = INVENTAIRE_MD) -> dict[str, Any]:
    corpus = load_corpus(docs_dir)
    lines = [
        "# Inventaire — dossier MOREAU c/ ATELIER RIVIÈRE",
        "",
        "Comparaison GreenLoop : 28 pièces, 235 804 caractères.",
        "",
        "| Pièce | Type | Caractères | Lignes | Date (en-tête) | Extrait (200 car.) |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    total_chars = 0
    for name in sorted(corpus):
        text = corpus[name]
        chars = len(text)
        total_chars += chars
        line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        doc_date = detect_document_date(text) or "—"
        excerpt = text[:200].replace("\n", " ").replace("|", "\\|")
        piece_type = "email" if "echange-email" in name else "pdf"
        lines.append(
            f"| `{name}` | {piece_type} | {chars} | {line_count} | {doc_date} | {excerpt} |"
        )
    lines.extend(
        [
            "",
            f"**Total : {len(corpus)} pièces, {total_chars} caractères.**",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"pieces": len(corpus), "caracteres": total_chars}


def cmd_extraire(_: argparse.Namespace) -> int:
    manifest = verify_manifest()
    write_manifest_report(manifest)
    if not manifest["ok"]:
        print("MANIFEST.sha256 : écarts détectés")
        return 1
    written = extract_all()
    print(f"Extraction : {len(written)} fichiers -> {DOCS_DIR}")
    inv = build_inventory()
    print(f"Inventaire : {inv['pieces']} pièces, {inv['caracteres']} caractères")
    return 0


def cmd_verifier(_: argparse.Namespace) -> int:
    report = verify_facts()
    write_extraction_report(report)
    verified = report["facts_verified"]
    total = report["facts_total"]
    if verified != total:
        failing = [str(fact["id"]) for fact in report["facts"] if not fact["ok"]]
        deficit = total - verified
        print(
            f"Vérification : {verified}/{total} faits OK — "
            f"{deficit} fait(s) en défaut : {', '.join(failing)} -> {EXTRACTION_MD}"
        )
        return 1
    print(f"Vérification : {verified}/{total} faits OK -> {EXTRACTION_MD}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extraction et contrôle MOREAU (banc comparatif)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extraire", help="Extraire les pièces source vers documents/").set_defaults(
        func=cmd_extraire
    )
    sub.add_parser("verifier", help="Vérifier les jetons discriminants des faits").set_defaults(
        func=cmd_verifier
    )
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
