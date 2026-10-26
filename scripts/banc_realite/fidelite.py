# SPDX-License-Identifier: AGPL-3.0-only
"""Measure fidelity (étage 1) against GreenLoop gold — never raises."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

CORPUS = (BANC_VAR / "dossiers" / "greenloop" / "documents")

from .infra import connect_existing
from .memory import memory_elements_by_piece, read_active_claims
from .normalize import claim_text, normalize_text, piece_basename

BANC_ROOT = (BANC_VAR)
GOLD = (BANC_VAR / "dossiers" / "greenloop" / "gold")
OUT = BANC_ROOT / "sorties" / "fidelite-C.json"
RESIDU = BANC_ROOT / "sorties" / "residu-a-adjudger.json"
HORS_OR = BANC_ROOT / "sorties" / "hors-or-echantillon.json"
SEED = 20260920
VAULT_REF = "greenloop"
TABULAR_SKILL = "tabulaire-deterministe"
F1_KEY_GLOBAL = "global"
F1_KEY_PAR_VOIE = "par_voie"
F1_KEY_PROSE = "prose"
F1_KEY_TABULAIRE = "tabulaire"
F1_KEY_MEMOIRE = "memoire"
F1_KEY_CLAIMS_TOTAL = "claims_total"
F1_KEY_SEUIL_ATTEINT = "seuil_atteint"
DATE_QUALIFIER_KEYS = frozenset(
    {
        "since",
        "period",
        "periode",
        "periode_debut",
        "periode_fin",
        "valid_from",
        "date",
    }
)
ISO_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
PAGE_MARKER_RE = re.compile(r"^---\s*page\s+(\d+)\s*---\s*$", re.IGNORECASE)
def load_gold(dossier=None) -> dict[str, Any]:
    from .or_dossier import load_gold as _load_gold_dossier

    return _load_gold_dossier(dossier)


def assertion_piece_tabular(source: str, dossier=None) -> bool:
    from .dossiers import Dossier, dossier_courant

    current = dossier if isinstance(dossier, Dossier) else dossier_courant(dossier)
    return piece_basename(source) in current.pieces_tabulaires


def claim_is_tabular(element: dict[str, Any]) -> bool:
    return element.get("extracted_by_skill") == TABULAR_SKILL


def _assertion_needle(assertion: dict[str, Any]) -> str:
    return normalize_text(assertion.get("quote") or "") or normalize_text(assertion.get("content") or "")


def _assertion_has_tabular_quote(assertion: dict[str, Any], dossier=None) -> bool:
    raw = str(assertion.get("quote") or assertion.get("content") or "")
    if "|" in raw:
        return True
    source = str(assertion.get("source") or "")
    return raw.count(",") >= 3 and assertion_piece_tabular(source, dossier=dossier)


def _needle_row_tokens(assertion: dict[str, Any], dossier=None) -> set[str]:
    raw = str(assertion.get("quote") or assertion.get("content") or "")
    if "|" in raw:
        return {normalize_text(part) for part in raw.split("|") if normalize_text(part)}
    if _assertion_has_tabular_quote(assertion, dossier=dossier) and raw.count(",") >= 3:
        return {normalize_text(part) for part in raw.split(",") if normalize_text(part)}
    return {token for token in _assertion_needle(assertion).split() if token}


def _entity_or_attribute_in_assertion(element: dict[str, Any], needle: str) -> bool:
    quals = element.get("qualifiers") or {}
    attribute = normalize_text(str(quals.get("attribute_id") or "").replace("_", " "))
    entity = normalize_text(str(element.get("entity_id") or "").replace("_", " "))
    if attribute and attribute in needle:
        return True
    if entity and entity in needle:
        return True
    return False


def _cell_value_in_assertion(cell_text: str, needle: str) -> bool:
    hay = normalize_text(cell_text)
    if not hay:
        return False
    tokens = {token.rstrip(",.;:") for token in needle.split() if token}
    if hay in tokens:
        return True
    return hay in needle


def _prose_cell_matches(element: dict[str, Any], assertion: dict[str, Any]) -> bool:
    needle = _assertion_needle(assertion)
    if not needle:
        return False
    if not _entity_or_attribute_in_assertion(element, needle):
        return False
    return _cell_value_in_assertion(str(element.get("text") or ""), needle)


def _month_hints_from_assertion(assertion: dict[str, Any]) -> set[str]:
    hints: set[str] = set()
    for field in ("content", "quote"):
        text = str(assertion.get(field) or "")
        for match in re.finditer(r"(20\d{2})-(\d{2})", text):
            year, month = match.group(1), match.group(2)
            hints.add(month)
            hints.add(f"{year}_{month}")
    return hints


def _tabular_scope_rank(element: dict[str, Any]) -> int:
    scope = (element.get("qualifiers") or {}).get("tabular_scope")
    if scope == "cell":
        return 0
    if scope == "row":
        return 1
    return 2


def _assertion_match_rank(
    element: dict[str, Any],
    needle: str,
    kind: str,
    *,
    assertion: dict[str, Any],
    row_tokens: set[str],
    month_hints: set[str],
    dossier=None,
) -> tuple[int, ...]:
    text = normalize_text(element.get("text") or "")
    entity = normalize_text(str(element.get("entity_id") or "").replace("_", " "))
    entity_in_needle = 0 if entity and entity in needle else 1
    attribute_id = str((element.get("qualifiers") or {}).get("attribute_id") or "")
    month_rank = 0 if month_hints and attribute_id in month_hints else 1
    if kind == "cellule":
        if _assertion_has_tabular_quote(assertion, dossier=dossier):
            value_rank = 0 if text and text in row_tokens else 1
        else:
            value_rank = 0 if _prose_cell_matches(element, assertion) else 1
    elif text == needle:
        value_rank = 0
    elif text and needle in text:
        value_rank = 1
    else:
        value_rank = 2
    text_len = len(text) if text else 9999
    return (_tabular_scope_rank(element), entity_in_needle, month_rank, value_rank, text_len)


def _collect_assertion_candidates(
    assertion: dict[str, Any],
    elements: list[dict[str, Any]],
    *,
    dossier=None,
) -> tuple[list[tuple[dict[str, Any], str]], str]:
    needle = _assertion_needle(assertion)
    if not needle:
        return [], ""
    row_tokens = _needle_row_tokens(assertion, dossier=dossier)
    candidates: list[tuple[dict[str, Any], str]] = []
    seen: set[Any] = set()

    def add(element: dict[str, Any], kind: str) -> None:
        key = element.get("claim_id") or id(element)
        if key in seen:
            return
        seen.add(key)
        candidates.append((element, kind))

    for element in elements:
        hay = normalize_text(element.get("text") or "")
        if hay and needle in hay:
            add(element, "texte")
    for element in elements:
        span = element.get("source_span") or {}
        raw = span.get("raw_line") or ""
        if raw and needle in normalize_text(raw):
            add(element, "provenance")
    for element in elements:
        if (element.get("qualifiers") or {}).get("tabular_scope") != "cell":
            continue
        if _assertion_has_tabular_quote(assertion, dossier=dossier):
            hay = normalize_text(element.get("text") or "")
            if hay and hay in row_tokens:
                add(element, "cellule")
        elif _prose_cell_matches(element, assertion):
            add(element, "cellule")
    return candidates, needle


def match_assertion(
    assertion: dict[str, Any],
    elements: list[dict[str, Any]],
    *,
    dossier=None,
) -> tuple[dict[str, Any] | None, str | None]:
    candidates, needle = _collect_assertion_candidates(assertion, elements, dossier=dossier)
    if not candidates:
        return None, None
    row_tokens = _needle_row_tokens(assertion, dossier=dossier)
    month_hints = _month_hints_from_assertion(assertion)
    return min(
        candidates,
        key=lambda item: _assertion_match_rank(
            item[0],
            needle,
            item[1],
            assertion=assertion,
            row_tokens=row_tokens,
            month_hints=month_hints,
            dossier=dossier,
        ),
    )


def page_for_line(piece_name: str, line_start: int | None) -> int | None:
    if line_start is None:
        return None
    piece_path = CORPUS / piece_name
    if not piece_path.is_file():
        return None
    page = 1
    for idx, raw in enumerate(piece_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = PAGE_MARKER_RE.match(raw.strip())
        if match and idx < line_start:
            page = int(match.group(1))
    return page


def lexical_distance(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(ISO_DATE_RE.match(value.strip()))


def extract_ordering_date(claim: dict[str, Any]) -> str | None:
    value = claim.get("value")
    if _is_iso_date(value):
        return str(value).strip()
    qualifiers = claim.get("qualifiers") or {}
    for key in DATE_QUALIFIER_KEYS:
        candidate = qualifiers.get(key)
        if _is_iso_date(candidate):
            return str(candidate).strip()
    for candidate in qualifiers.values():
        if _is_iso_date(candidate):
            return str(candidate).strip()
    return None


def claim_has_ordering_date(claim: dict[str, Any]) -> bool:
    return extract_ordering_date(claim) is not None


def evaluate_trajectory_order(
    trajectory: dict[str, Any],
    gold: dict[str, Any],
    by_piece: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    versions = trajectory.get("versions", [])
    details = []
    present_count = 0
    dated_count = 0
    fact_dates: list[str] = []
    gold_dates: list[str] = []

    for version in versions:
        aid = version.get("assertion")
        assertion = gold["assertion_index"].get(aid)
        if not assertion:
            details.append({"assertion": aid, "presente": False, "datee": False})
            continue
        piece = piece_basename(assertion["source"])
        hit, _ = match_assertion(assertion, by_piece.get(piece, []))
        presente = hit is not None
        datee = presente and claim_has_ordering_date(hit)
        if presente:
            present_count += 1
        if datee:
            dated_count += 1
            fact_dates.append(extract_ordering_date(hit) or "")
        gold_start = version.get("validity_start")
        if gold_start:
            gold_dates.append(str(gold_start))
        details.append({"assertion": aid, "presente": presente, "datee": datee})

    motif: str | None = None
    if present_count < len(versions):
        motif = "versions manquantes"
    elif dated_count < present_count:
        motif = "non datées"
    elif fact_dates != gold_dates:
        motif = "ordre incorrect"

    return {
        "id": trajectory.get("id"),
        "versions": details,
        "versions_presentes": present_count,
        "versions_datees": dated_count,
        "ordre_restituable": motif is None,
        "motif": motif,
    }


def archive_if_exists(path: Path) -> None:
    if path.is_file():
        archived = path.with_name(f"{path.stem}.avant-correction-tabulaire{path.suffix}")
        archived.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def analyze_natural_key_collisions(connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path, "
        "extracted_by_skill, lifecycle_status, created_at "
        "FROM brain.claims WHERE vault_ref = %s ORDER BY created_at, claim_id",
        (VAULT_REF,),
    ).fetchall()
    try:
        from brain.adapters.supabase_claims import AtomicClaim, compute_natural_key
    except ImportError:
        AtomicClaim = None  # type: ignore[misc,assignment]
        compute_natural_key = None  # type: ignore[misc,assignment]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        claim = AtomicClaim(
            entity_id=row["entity_id"],
            predicate=row["predicate"],
            value=row["value"],
            qualifiers=row.get("qualifiers") or {},
            modality="asserted",
        )
        key = compute_natural_key(claim)
        grouped[key].append(dict(row))
    collisions = []
    for key, items in grouped.items():
        active = [item for item in items if item.get("lifecycle_status") == "active"]
        if len(active) <= 1 and len(items) <= 1:
            continue
        if len(items) < 2:
            continue
        collisions.append(
            {
                "natural_key": key,
                "occurrences": len(items),
                "actifs": len(active),
                "items": [
                    {
                        "piece": piece_basename(item.get("source_file_path") or ""),
                        "voie": "tabulaire"
                        if item.get("extracted_by_skill") == TABULAR_SKILL
                        else "modele",
                        "entity_id": item.get("entity_id"),
                        "qualifiers": item.get("qualifiers") or {},
                        "value": item.get("value"),
                        "lifecycle_status": item.get("lifecycle_status"),
                        "claim_id": str(item.get("claim_id")),
                    }
                    for item in items
                ],
            }
        )
    return collisions[:10]


def _fidelite_paths(output_dir: Path | None = None) -> tuple[Path, Path, Path]:
    base = output_dir or (BANC_ROOT / "sorties")
    return (
        base / "fidelite-C.json",
        base / "residu-a-adjudger.json",
        base / "hors-or-echantillon.json",
    )


def measure(
    output_dir: Path | None = None,
    dossier: str | None = None,
) -> dict[str, Any]:
    out_path, residu_path, hors_or_path = _fidelite_paths(output_dir)
    result: dict[str, Any] = {"status": "ok", "seed": SEED}
    try:
        body = _measure_body(out_path, residu_path, hors_or_path, dossier=dossier)
        result.update(body)
    except Exception as exc:  # noqa: BLE001 — outil de mesure sans exception
        result = {"status": "error", "message": str(exc)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.with_suffix(".erreur.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
    return result


def _measure_body(
    out_path: Path,
    residu_path: Path,
    hors_or_path: Path,
    *,
    dossier: str | None = None,
) -> dict[str, Any]:
    from .dossiers import dossier_courant

    current = dossier_courant(dossier) if dossier else dossier_courant("greenloop")
    corpus_dir = current.corpus_dir
    vault_ref = current.vault_ref
    result: dict[str, Any] = {"status": "ok", "seed": SEED, "dossier": current.nom}
    gold = load_gold(current)
    connection = connect_existing()
    claims = read_active_claims(connection, vault_ref=vault_ref)
    by_piece = memory_elements_by_piece(claims)

    all_assertions = gold["assertions"]
    matched = []
    unmatched = []
    by_piece_scores: dict[str, list[bool]] = defaultdict(list)
    by_perspective_scores: dict[str, list[bool]] = defaultdict(list)
    by_voie_scores: dict[str, list[bool]] = defaultdict(list)
    croisement = {
        "pieces_tabulaires": {
            "appariee_tabulaire": 0,
            "appariee_modele": 0,
            "non_appariee": 0,
        },
        "pieces_prose": {
            "appariee_tabulaire": 0,
            "appariee_modele": 0,
            "non_appariee": 0,
        },
    }
    f1_apparie_par_texte = 0
    f1_apparie_par_provenance_seule = 0

    for assertion in all_assertions:
        piece = piece_basename(assertion["source"])
        elements = by_piece.get(piece, [])
        hit, match_kind = match_assertion(assertion, elements, dossier=current)
        ok = hit is not None
        by_piece_scores[piece].append(ok)
        by_perspective_scores[assertion.get("perspective", "unknown")].append(ok)
        expect_tabular_piece = assertion_piece_tabular(assertion["source"], dossier=current)
        by_voie_scores["tabulaire" if expect_tabular_piece else "prose"].append(ok)
        bucket = (
            croisement["pieces_tabulaires"]
            if expect_tabular_piece
            else croisement["pieces_prose"]
        )
        if ok:
            if match_kind == "texte":
                f1_apparie_par_texte += 1
            elif match_kind == "provenance":
                f1_apparie_par_provenance_seule += 1
            matched.append(
                {
                    "assertion_id": assertion["id"],
                    "claim_id": str(hit["claim_id"]),
                    "piece_tabulaire": expect_tabular_piece,
                    "voie_reelle": "tabulaire" if claim_is_tabular(hit) else "modele",
                    "appariement": match_kind,
                }
            )
            if claim_is_tabular(hit):
                bucket["appariee_tabulaire"] += 1
            else:
                bucket["appariee_modele"] += 1
        else:
            unmatched.append(assertion)
            bucket["non_appariee"] += 1

    denom = len(all_assertions)
    f1_global = len(matched) / denom if denom else 0.0
    tab_assertions = [
        a for a in all_assertions if assertion_piece_tabular(a["source"], dossier=current)
    ]
    prose_assertions = [
        a for a in all_assertions if not assertion_piece_tabular(a["source"], dossier=current)
    ]
    tab_matched = sum(1 for m in matched if m["piece_tabulaire"])
    prose_matched = sum(1 for m in matched if not m["piece_tabulaire"])
    result["F1"] = {
        "global": round(f1_global, 4),
        "denominateur": denom,
        "appariees": len(matched),
        "appariees_par_texte": f1_apparie_par_texte,
        "appariees_par_provenance_seule": f1_apparie_par_provenance_seule,
        "F1_tabulaire": round(tab_matched / len(tab_assertions), 4) if tab_assertions else 0.0,
        "croisement": croisement,
        "par_piece": {
            piece: round(sum(vals) / len(vals), 4) if vals else 0.0
            for piece, vals in sorted(by_piece_scores.items())
        },
        "par_perspective": {
            persp: round(sum(vals) / len(vals), 4) if vals else 0.0
            for persp, vals in sorted(by_perspective_scores.items())
        },
        "par_voie": {
            voie: round(sum(vals) / len(vals), 4) if vals else 0.0
            for voie, vals in sorted(by_voie_scores.items())
        },
        "normalisation": (
            "NFKC, minuscules, déaccentuation NFD, guillemets/traits unifiés, "
            "retrait markdown (*_`#>|), ponctuation hors alphanum réduite à espace, "
            "espaces compressés ; appariement par sous-chaîne de quote normalisée "
            "ou à défaut content normalisé dans le texte d'un claim de la même pièce, "
            "sinon dans source_span.raw_line"
        ),
    }

    rng = random.Random(SEED)
    sample = unmatched[:] if len(unmatched) <= 40 else rng.sample(unmatched, 40)
    residu = []
    for assertion in sample:
        piece = piece_basename(assertion["source"])
        elements = by_piece.get(piece, [])
        ranked = sorted(
            elements,
            key=lambda e: lexical_distance(
                assertion.get("quote") or assertion.get("content") or "",
                e.get("text") or "",
            ),
            reverse=True,
        )[:3]
        residu.append(
            {
                "assertion": assertion,
                "proches": [
                    {
                        "claim_id": str(e["claim_id"]),
                        "predicate": e.get("predicate"),
                        "text": (e.get("text") or "")[:500],
                        "score_lexical": round(
                            lexical_distance(
                                assertion.get("quote") or assertion.get("content") or "",
                                e.get("text") or "",
                            ),
                            4,
                        ),
                    }
                    for e in ranked
                ],
            }
        )
    residu_path.parent.mkdir(parents=True, exist_ok=True)
    residu_path.write_text(
        json.dumps({"seed": SEED, "items": residu}, ensure_ascii=False, indent=2) + "\n"
    )

    def f2_for_matches(match_subset: list[dict[str, Any]]) -> dict[str, Any]:
        mesurable = 0
        piece_ok = 0
        line_ok = 0
        page_ok = 0
        sans_ligne = 0
        for match in match_subset:
            assertion = gold["assertion_index"][match["assertion_id"]]
            element = next(c for c in claims if str(c["claim_id"]) == match["claim_id"])
            span = element.get("source_span") or {}
            line_start = span.get("line_start")
            if line_start is None:
                sans_ligne += 1
            mesurable += 1
            if piece_basename(assertion["source"]) == element.get("piece"):
                piece_ok += 1
            if match["voie_reelle"] == "tabulaire":
                quote = assertion.get("quote") or ""
                line_from_quote = None
                piece_path = corpus_dir / piece_basename(assertion["source"])
                if piece_path.is_file() and quote:
                    for idx, raw in enumerate(piece_path.read_text(encoding="utf-8").splitlines(), start=1):
                        if normalize_text(quote) in normalize_text(raw):
                            line_from_quote = idx
                            break
                expected_line = line_from_quote or line_start
                if expected_line is not None and int(line_start or -1) == int(expected_line):
                    line_ok += 1
            expected_page = assertion.get("page")
            if expected_page is not None:
                actual_page = page_for_line(element.get("piece") or "", line_start)
                if actual_page is not None and int(actual_page) == int(expected_page):
                    page_ok += 1
        line_den = sum(1 for m in match_subset if m["voie_reelle"] == "tabulaire")
        page_den = sum(
            1
            for m in match_subset
            if gold["assertion_index"][m["assertion_id"]].get("page") is not None
        )
        return {
            "mesurable": mesurable,
            "sans_ligne": sans_ligne,
            "piece_correcte": round(piece_ok / mesurable, 4) if mesurable else 0.0,
            "ligne_correcte": round(line_ok / line_den, 4) if line_den else None,
            "page_correcte": round(page_ok / page_den, 4) if page_den else None,
        }

    tab_matches = [m for m in matched if m["voie_reelle"] == "tabulaire"]
    model_matches = [m for m in matched if m["voie_reelle"] == "modele"]
    result["F2"] = {
        "global": f2_for_matches(matched),
        "tabulaire": f2_for_matches(tab_matches),
        "modele": f2_for_matches(model_matches),
    }

    dispute_classes = Counter()
    dispute_details = []
    for dispute in gold["disputes"]:
        sides = dispute.get("sides", [])
        matched_elements = []
        for side in sides:
            aid = side.get("assertion")
            assertion = gold["assertion_index"].get(aid)
            if not assertion:
                continue
            piece = piece_basename(assertion["source"])
            hit, _ = match_assertion(assertion, by_piece.get(piece, []))
            if hit:
                matched_elements.append(str(hit["claim_id"]))
        unique = len(set(matched_elements))
        total = len(sides)
        if unique == 0:
            klass = "aucun"
        elif unique == 1 and len(matched_elements) > 1:
            klass = "cotes_fusionnes"
        elif unique == total:
            klass = "tous_les_cotes"
        elif unique == 1:
            klass = "un_seul_cote"
        else:
            klass = "partiel"
        dispute_classes[klass] += 1
        dispute_details.append(
            {"id": dispute["id"], "classe": klass, "cotes_apparies": unique, "cotes_total": total}
        )
    result["F3"] = {"total": len(gold["disputes"]), "classes": dict(dispute_classes), "details": dispute_details}

    separations = []
    for situation in gold["situations"]:
        for other in situation.get("must_stay_separate_from", []) or []:
            separations.append((situation["id"], other))
    rev_cases = gold.get("reversibility_cases", [])
    f4 = []
    for left, right in separations:
        left_assertions = next(s for s in gold["situations"] if s["id"] == left)["assertions"]
        right_assertions = next(s for s in gold["situations"] if s["id"] == right)["assertions"]
        left_hits = {
            str(match_assertion(a, by_piece.get(piece_basename(a["source"]), []))[0]["claim_id"])
            for a in left_assertions
            if match_assertion(a, by_piece.get(piece_basename(a["source"]), []))[0]
        }
        right_hits = {
            str(match_assertion(a, by_piece.get(piece_basename(a["source"]), []))[0]["claim_id"])
            for a in right_assertions
            if match_assertion(a, by_piece.get(piece_basename(a["source"]), []))[0]
        }
        f4.append(
            {
                "left": left,
                "right": right,
                "distinct": bool(left_hits and right_hits and not (left_hits & right_hits)),
                "mesurable": bool(left_hits and right_hits),
            }
        )
    for case in rev_cases:
        ids = case.get("situations", [])
        if len(ids) != 2:
            continue
        left, right = ids
        left_assertions = next(s for s in gold["situations"] if s["id"] == left)["assertions"]
        right_assertions = next(s for s in gold["situations"] if s["id"] == right)["assertions"]
        left_hits = {
            str(match_assertion(a, by_piece.get(piece_basename(a["source"]), []))[0]["claim_id"])
            for a in left_assertions
            if match_assertion(a, by_piece.get(piece_basename(a["source"]), []))[0]
        }
        right_hits = {
            str(match_assertion(a, by_piece.get(piece_basename(a["source"]), []))[0]["claim_id"])
            for a in right_assertions
            if match_assertion(a, by_piece.get(piece_basename(a["source"]), []))[0]
        }
        f4.append(
            {
                "reversibility": case.get("id"),
                "left": left,
                "right": right,
                "distinct": bool(left_hits and right_hits and not (left_hits & right_hits)),
                "mesurable": bool(left_hits and right_hits),
            }
        )
    non_meas = [x for x in f4 if not x["mesurable"]]
    result["F4"] = {
        "paires": len(f4),
        "distinctes_mesurables": sum(1 for x in f4 if x["mesurable"] and x["distinct"]),
        "non_mesurable": len(non_meas),
        "motif": (
            "notes plates sans lien explicite entre situations : séparation structurelle non représentée"
            if non_meas
            else None
        ),
        "details": f4,
    }

    f6 = []
    for trajectory in gold["trajectories"]:
        evaluated = evaluate_trajectory_order(trajectory, gold, by_piece)
        f6.append(evaluated)
    result["F6"] = {
        "trajectoires": len(f6),
        "completes": sum(1 for t in f6 if t["ordre_restituable"]),
        "details": f6,
    }

    matched_assertion_ids = {m["assertion_id"] for m in matched}

    def claim_matches_any_assertion(claim: dict[str, Any]) -> bool:
        # Même règle que match_assertion : un fait tient une assertion par son texte
        # OU par sa provenance (source_span.raw_line). Sans le second test, presque
        # tous les faits tabulaires sortaient hors-or depuis que la ligne brute a
        # quitté la valeur.
        piece = claim.get("piece") or ""
        text = normalize_text(claim.get("text") or "")
        raw = normalize_text(((claim.get("source_span") or {}).get("raw_line")) or "")
        if not text and not raw:
            return False
        for aid in matched_assertion_ids:
            assertion = gold["assertion_index"][aid]
            if piece_basename(assertion["source"]) != piece:
                continue
            needle = _assertion_needle(assertion)
            if needle and (needle in text or (raw and needle in raw)):
                return True
        return False

    hors_or_tabulaire = [
        c
        for c in claims
        if c.get("predicate") == "note.body" and claim_is_tabular(c) and not claim_matches_any_assertion(c)
    ]
    hors_or_modele = [
        c
        for c in claims
        if c.get("predicate") == "note.body" and not claim_is_tabular(c) and not claim_matches_any_assertion(c)
    ]
    rng = random.Random(SEED)
    sample_hors = (hors_or_tabulaire + hors_or_modele) if len(hors_or_tabulaire + hors_or_modele) <= 40 else rng.sample(hors_or_tabulaire + hors_or_modele, 40)
    archive_if_exists(hors_or_path)
    hors_or_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "total": len(hors_or_tabulaire) + len(hors_or_modele),
                "par_voie": {
                    "tabulaire": len(hors_or_tabulaire),
                    "modele": len(hors_or_modele),
                },
                "echantillon": [
                    {
                        "claim_id": str(c["claim_id"]),
                        "piece": c.get("piece"),
                        "voie": "tabulaire" if claim_is_tabular(c) else "modele",
                        "text": (c.get("text") or "")[:400],
                    }
                    for c in sample_hors
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    result["hors_or"] = {
        "total": len(hors_or_tabulaire) + len(hors_or_modele),
        "tabulaire": len(hors_or_tabulaire),
        "modele": len(hors_or_modele),
        "echantillon": len(sample_hors),
    }

    predicates = Counter(c.get("predicate") for c in claims)
    bodies = [len(c.get("text") or "") for c in claims if c.get("predicate") == "note.body"]
    bodies.sort()
    median = bodies[len(bodies) // 2] if bodies else 0
    entities = Counter(c.get("entity_id") for c in claims)
    result["memoire"] = {
        "claims_total": len(claims),
        "predicates": dict(predicates),
        "note_body_median_chars": median,
        "entites_distinctes": len(entities),
        "top_entites": entities.most_common(15),
    }
    result["collisions_natural_key"] = analyze_natural_key_collisions(connection)
    result["seuil_atteint"] = f1_global >= 0.80
    archive_if_exists(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result

if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
