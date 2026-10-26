# SPDX-License-Identifier: AGPL-3.0-only
"""Build human relecture data and page for banc-realite bras C."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import html
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from banc_realite.compat_moteur import embed_with_retry

from .fidelite import CORPUS, claim_is_tabular, lexical_distance
from .infra import connect_existing
from .memory import memory_elements_by_piece, read_active_claims
from .normalize import claim_text, normalize_text, piece_basename

BANC_ROOT = (BANC_VAR)
RESIDU_IN = BANC_ROOT / "sorties" / "residu-a-adjudger.json"
HORS_OR_IN = BANC_ROOT / "sorties" / "hors-or-echantillon.json"
DATA_OUT = BANC_ROOT / "sorties" / "relecture-donnees.json"
HTML_OUT = BANC_ROOT / "relecture.html"
SEED = 20260920
VAULT_REF = "greenloop"


def parse_vector(raw: str) -> list[float]:
    return [float(x) for x in raw.strip("[]").split(",") if x.strip()]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def piece_path(piece_name: str, *, corpus: Path = CORPUS) -> Path:
    return corpus / piece_name


def read_piece_lines(piece_name: str, *, corpus: Path = CORPUS) -> list[str] | None:
    path = piece_path(piece_name, corpus=corpus)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").splitlines()


def find_quote_line(piece_name: str, quote: str, *, corpus: Path = CORPUS) -> int | None:
    lines = read_piece_lines(piece_name, corpus=corpus)
    if lines is None or not quote:
        return None
    needle = normalize_text(quote)
    if not needle:
        return None
    for idx, raw in enumerate(lines, start=1):
        if needle in normalize_text(raw):
            return idx
    return None


def format_numbered_excerpt(
    lines: list[str],
    center_line: int,
    *,
    before: int = 2,
    after: int = 2,
) -> str:
    start = max(1, center_line - before)
    end = min(len(lines), center_line + after)
    out: list[str] = []
    for num in range(start, end + 1):
        prefix = "→ " if num == center_line else "  "
        out.append(f"{prefix}{num:4d}  {lines[num - 1]}")
    return "\n".join(out)


def excerpt_around_quote(
    piece_name: str,
    quote: str,
    *,
    corpus: Path = CORPUS,
) -> dict[str, Any]:
    lines = read_piece_lines(piece_name, corpus=corpus)
    if lines is None:
        return {
            "status": "piece_introuvable",
            "texte": f"Pièce « {piece_name} » introuvable dans le corpus.",
        }
    line = find_quote_line(piece_name, quote, corpus=corpus)
    if line is None:
        return {
            "status": "citation_introuvable",
            "texte": "Citation introuvable dans la pièce (recherche par normalisation).",
        }
    return {
        "status": "ok",
        "texte": format_numbered_excerpt(lines, line),
        "ligne_centrale": line,
    }


def excerpt_around_span(
    piece_name: str,
    source_span: dict[str, Any] | None,
    *,
    corpus: Path = CORPUS,
) -> dict[str, Any]:
    span = source_span or {}
    line_start = span.get("line_start")
    if line_start is None:
        return {
            "status": "sans_ligne",
            "texte": "Aucune ligne de source enregistrée pour ce fait.",
        }
    lines = read_piece_lines(piece_name, corpus=corpus)
    if lines is None:
        return {
            "status": "piece_introuvable",
            "texte": f"Pièce « {piece_name} » introuvable dans le corpus.",
        }
    center = int(line_start)
    if center < 1 or center > len(lines):
        return {
            "status": "ligne_hors_fichier",
            "texte": f"Ligne {center} hors limites du fichier ({len(lines)} lignes).",
        }
    return {
        "status": "ok",
        "texte": format_numbered_excerpt(lines, center),
        "ligne_centrale": center,
    }


def span_lines_label(source_span: dict[str, Any] | None) -> str | None:
    span = source_span or {}
    start = span.get("line_start")
    end = span.get("line_end", start)
    if start is None:
        return None
    if end is not None and int(end) != int(start):
        return f"lignes {start}–{end}"
    return f"ligne {start}"


def claim_voie(claim: dict[str, Any]) -> str:
    return "tabulaire" if claim_is_tabular(claim) else "modele"


def humanize_id(identifier: Any) -> str:
    if identifier is None:
        return ""
    return str(identifier).replace("_", " ").strip()


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def autres_qualifiers(qualifiers: dict[str, Any] | None) -> dict[str, Any]:
    return {
        key: value
        for key, value in (qualifiers or {}).items()
        if key not in ("attribute_id", "tabular_row")
    }


def format_triplet(claim: dict[str, Any]) -> dict[str, Any]:
    qualifiers = claim.get("qualifiers") or {}
    value = format_value(claim.get("value"))
    if claim_is_tabular(claim):
        return {
            "triplet": value,
            "entite": None,
            "attribut": None,
            "valeur": value,
            "sans_attribut": False,
        }
    entity = humanize_id(claim.get("entity_id"))
    attribute = humanize_id(qualifiers.get("attribute_id"))
    if attribute:
        triplet = f"{entity} · {attribute} : {value}"
        sans_attribut = False
    elif entity:
        triplet = f"{entity} : {value}"
        sans_attribut = True
    else:
        triplet = value
        sans_attribut = True
    return {
        "triplet": triplet,
        "entite": entity or None,
        "attribut": attribute or None,
        "valeur": value,
        "sans_attribut": sans_attribut,
    }


def fact_payload(claim: dict[str, Any]) -> dict[str, Any]:
    triplet_info = format_triplet(claim)
    span = claim.get("source_span") or {}
    return {
        "claim_id": str(claim["claim_id"]),
        "entity_id": claim.get("entity_id"),
        "predicate": claim.get("predicate"),
        "qualifiers": claim.get("qualifiers") or {},
        "autres_qualifiers": autres_qualifiers(claim.get("qualifiers")),
        "voie": claim_voie(claim),
        "texte": claim.get("text") or claim_text(claim.get("value")),
        "triplet": triplet_info["triplet"],
        "entite": triplet_info["entite"],
        "attribut": triplet_info["attribut"],
        "valeur": triplet_info["valeur"],
        "sans_attribut": triplet_info["sans_attribut"],
        "lignes_source": span_lines_label(span),
    }


def neighbor_payload(claim: dict[str, Any]) -> dict[str, Any]:
    return fact_payload(claim)


def lexical_neighbors(
    *,
    piece: str,
    query: str,
    by_piece: dict[str, list[dict[str, Any]]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    ranked = sorted(
        by_piece.get(piece, []),
        key=lambda claim: lexical_distance(query, format_triplet(claim)["triplet"]),
        reverse=True,
    )[:limit]
    return [neighbor_payload(claim) for claim in ranked]


def load_embeddings_by_piece(connection) -> dict[str, list[dict[str, Any]]]:
    rows = connection.execute(
        "SELECT claim_id, entity_id, predicate, value, qualifiers, source_file_path, "
        "source_span, extracted_by_skill, value_embedding_mistral::text AS embedding "
        "FROM brain.claims "
        "WHERE lifecycle_status = 'active' AND vault_ref = %s "
        "AND predicate = 'note.body' AND value_embedding_mistral IS NOT NULL",
        (VAULT_REF,),
    ).fetchall()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        item["text"] = claim_text(item.get("value"))
        item["piece"] = piece_basename(item.get("source_file_path") or "")
        item["vector"] = parse_vector(item.pop("embedding"))
        grouped.setdefault(item["piece"], []).append(item)
    return grouped


def vector_neighbors(
    *,
    piece: str,
    content: str,
    exclude_ids: set[str],
    embeddings_by_piece: dict[str, list[dict[str, Any]]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    candidates = [
        c for c in embeddings_by_piece.get(piece, []) if str(c["claim_id"]) not in exclude_ids
    ]
    if not candidates or not content.strip():
        return []
    query = embed_with_retry(content)
    ranked = sorted(
        candidates,
        key=lambda c: cosine_similarity(query, c["vector"]),
        reverse=True,
    )[:limit]
    return [neighbor_payload(c) for c in ranked]


def merge_neighbors(
    lexical: list[dict[str, Any]],
    vectorial: list[dict[str, Any]],
    *,
    claims_index: dict[str, dict[str, Any]],
    max_total: int = 8,
) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    vector_added = 0
    vector_ids = {str(v.get("claim_id")) for v in vectorial}
    for item in lexical + vectorial:
        cid = str(item.get("claim_id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        claim = claims_index.get(cid)
        if claim:
            merged.append(neighbor_payload(claim))
        else:
            merged.append(
                {
                    "claim_id": cid,
                    "entity_id": item.get("entity_id"),
                    "predicate": item.get("predicate"),
                    "voie": item.get("voie") or "modele",
                    "texte": item.get("text") or item.get("texte") or "",
                    "lignes_source": item.get("lignes_source"),
                }
            )
        if cid in vector_ids and cid not in {str(x.get("claim_id")) for x in lexical}:
            vector_added += 1
        if len(merged) >= max_total:
            break
    return merged[:max_total], vector_added


def build_data(
    *,
    corpus: Path = CORPUS,
    residu_in: Path = RESIDU_IN,
    hors_or_in: Path = HORS_OR_IN,
) -> dict[str, Any]:
    residu_blob = json.loads(residu_in.read_text(encoding="utf-8"))
    hors_blob = json.loads(hors_or_in.read_text(encoding="utf-8"))
    connection = connect_existing()
    claims = read_active_claims(connection, vault_ref=VAULT_REF)
    claims_index = {str(c["claim_id"]): c for c in claims}
    by_piece = memory_elements_by_piece(claims)
    embeddings_by_piece = load_embeddings_by_piece(connection)

    citations_introuvables = 0
    voisins_vectoriels_ajoutes = 0
    faits_avec_triplet = 0
    faits_sans_attribut = 0
    residu_out: list[dict[str, Any]] = []

    def count_fact(payload: dict[str, Any]) -> None:
        nonlocal faits_avec_triplet, faits_sans_attribut
        if payload.get("triplet"):
            faits_avec_triplet += 1
        if payload.get("sans_attribut"):
            faits_sans_attribut += 1

    for item in residu_blob["items"]:
        assertion = item["assertion"]
        piece = piece_basename(assertion["source"])
        excerpt = excerpt_around_quote(piece, assertion.get("quote") or "", corpus=corpus)
        if excerpt["status"] == "citation_introuvable":
            citations_introuvables += 1

        query = assertion.get("quote") or assertion.get("content") or ""
        lexical = lexical_neighbors(piece=piece, query=query, by_piece=by_piece)
        lexical_ids = {str(n["claim_id"]) for n in lexical}
        vectorial = vector_neighbors(
            piece=piece,
            content=assertion.get("content") or "",
            exclude_ids=lexical_ids,
            embeddings_by_piece=embeddings_by_piece,
        )
        neighbors, added = merge_neighbors(
            lexical,
            vectorial,
            claims_index=claims_index,
        )
        voisins_vectoriels_ajoutes += added
        for neighbor in neighbors:
            count_fact(neighbor)

        residu_out.append(
            {
                "id": assertion["id"],
                "piece": piece,
                "page": assertion.get("page"),
                "perspective": assertion.get("perspective"),
                "content": assertion.get("content") or "",
                "quote": assertion.get("quote") or "",
                "extrait_source": excerpt["texte"],
                "extrait_status": excerpt["status"],
                "voisins": neighbors,
            }
        )

    hors_or_out: list[dict[str, Any]] = []
    for sample in hors_blob["echantillon"]:
        cid = str(sample["claim_id"])
        claim = claims_index[cid]
        piece = sample.get("piece") or claim.get("piece") or ""
        excerpt = excerpt_around_span(piece, claim.get("source_span"), corpus=corpus)
        payload = fact_payload(claim)
        payload["piece"] = piece
        payload["extrait_source"] = excerpt["texte"]
        payload["extrait_status"] = excerpt["status"]
        count_fact(payload)
        hors_or_out.append(payload)

    return {
        "seed": SEED,
        "date": date.today().isoformat(),
        "residu": residu_out,
        "hors_or": hors_or_out,
        "stats": {
            "citations_introuvables": citations_introuvables,
            "voisins_vectoriels_ajoutes": voisins_vectoriels_ajoutes,
            "faits_avec_triplet": faits_avec_triplet,
            "faits_sans_attribut": faits_sans_attribut,
        },
    }


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _render_triplet_html(fact: dict[str, Any]) -> str:
    voie = _esc(fact.get("voie"))
    if fact.get("voie") == "tabulaire" or (not fact.get("entite") and not fact.get("attribut")):
        return (
            f'<div class="triplet"><span class="triplet-val">{_esc(fact.get("valeur") or fact.get("triplet"))}</span>'
            f'<span class="triplet-voie"> · {voie}</span></div>'
        )
    ctx_parts = [_esc(fact.get("entite"))]
    if fact.get("attribut"):
        ctx_parts.append(_esc(fact.get("attribut")))
    ctx = " · ".join(part for part in ctx_parts if part)
    return (
        f'<div class="triplet"><span class="triplet-ctx">{ctx}</span>'
        f'<span class="triplet-sep"> : </span>'
        f'<span class="triplet-val">{_esc(fact.get("valeur"))}</span>'
        f'<span class="triplet-voie"> · {voie}</span></div>'
    )


def render_html(data: dict[str, Any]) -> str:
    intro = (
        "Sur 2 403 assertions de l'or, le test automatique ne reconnaît qu'une seule comme tenue "
        "en mémoire, et 809 faits produits par le modèle ne correspondent à rien dans l'or. "
        "Deux lectures possibles : le modèle reformule (le test ne le voit pas) ou il perd "
        "l'information. Cette page sert à trancher, assertion par assertion et fait par fait, "
        "sans ouvrir de fichier."
    )
    mode = (
        "Parcourez les 40 cartes du résidu (partie A) puis les 40 faits hors-or (partie B). "
        "Choisissez un verdict par carte ; pour « Partielle », précisez ce qui manque. "
        "Votre progression est sauvegardée dans le navigateur. "
        "Copiez le bloc « Mes verdicts » en bas pour le coller dans l'atelier. "
        "Un fait du modèle est un triplet — entité, attribut, valeur. "
        "Les noms d'entité et d'attribut sont souvent en français, même quand la pièce est en anglais : "
        "ce n'est pas une erreur à sanctionner, juge le sens."
    )
    css = """\
  :root {
    --bg: #f8f8f6;
    --surface: #ffffff;
    --text: #1a1a1a;
    --muted: #4a4a4a;
    --border: #c8c8c4;
    --accent: #1e3a5f;
    --decision-bg: #eef4fb;
    --decision-border: #2c5282;
  }
  * { box-sizing: border-box; }
  html { font-size: 16px; scroll-behavior: smooth; }
  body {
    margin: 0;
    padding: 5rem 1.25rem 4rem;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.55;
    color: var(--text);
    background: var(--bg);
  }
  .page { max-width: 72ch; margin: 0 auto; }
  header.doc-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid var(--text);
  }
  header.doc-header h1 {
    margin: 0 0 1rem;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.25;
  }
  .meta { font-size: 0.875rem; color: var(--muted); margin-bottom: 1rem; }
  .counter {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 100;
    padding: 0.65rem 1.25rem;
    background: var(--surface);
    border-bottom: 2px solid var(--decision-border);
    font-weight: 700;
    text-align: center;
  }
  section > h2 {
    margin: 2rem 0 1rem;
    padding-top: 1.5rem;
    font-size: 1.3125rem;
    font-weight: 700;
    border-top: 1px solid var(--border);
  }
  section:first-of-type > h2 { border-top: none; padding-top: 0; }
  .card {
    margin: 1.25rem 0;
    padding: 1.15rem 1.25rem;
    background: var(--surface);
    border: 1px solid var(--border);
  }
  .card h3 { margin: 0 0 0.75rem; font-size: 1rem; font-weight: 700; color: var(--accent); }
  .field { margin: 0.75rem 0; }
  .field-label {
    display: block;
    margin-bottom: 0.25rem;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }
  .field-body { margin: 0; white-space: pre-wrap; word-break: break-word; }
  .scroll-box {
    max-height: 14rem;
    overflow: auto;
    padding: 0.65rem 0.75rem;
    background: #efefec;
    border: 1px solid var(--border);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.8125rem;
    line-height: 1.45;
    white-space: pre;
  }
  .neighbor {
    margin: 0.65rem 0;
    padding: 0.65rem 0.75rem;
    border-left: 3px solid var(--decision-border);
    background: var(--decision-bg);
  }
  .neighbor-meta { font-size: 0.8125rem; color: var(--muted); margin-bottom: 0.35rem; }
  .triplet { margin: 0; white-space: pre-wrap; word-break: break-word; }
  .triplet-ctx {
    font-variant: small-caps;
    color: var(--muted);
    letter-spacing: 0.02em;
  }
  .triplet-sep { color: var(--muted); }
  .triplet-val { font-weight: 600; }
  .triplet-voie { font-size: 0.8125rem; color: var(--muted); }
  .verdicts { margin-top: 1rem; border: none; padding: 0; }
  .verdicts label { display: block; margin: 0.35rem 0; cursor: pointer; }
  .partial-note { margin-top: 0.5rem; }
  .partial-note input {
    width: 100%;
    max-width: 32rem;
    padding: 0.35rem 0.5rem;
    border: 1px solid var(--border);
    font: inherit;
  }
  .partial-note[hidden] { display: none; }
  #verdicts-block {
    margin-top: 3rem;
    padding: 1.5rem;
    background: var(--surface);
    border: 2px solid var(--decision-border);
  }
  #verdicts-block h2 { margin: 0 0 1rem; font-size: 1.25rem; color: var(--accent); }
  #verdicts-out {
    width: 100%;
    min-height: 6rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.8125rem;
    padding: 0.65rem;
    border: 1px solid var(--border);
    background: #efefec;
  }
  #copy-btn {
    margin-top: 0.75rem;
    padding: 0.45rem 0.9rem;
    font: inherit;
    font-weight: 600;
    background: var(--decision-bg);
    border: 2px solid var(--decision-border);
    cursor: pointer;
  }
  .help { font-size: 0.875rem; color: var(--muted); margin-top: 0.75rem; }
"""

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="fr">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Relecture du résidu et du hors-or — bras C</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        '<div class="counter" id="counter" aria-live="polite">0 / 80</div>',
        '<div class="page">',
        "<header class=\"doc-header\">",
        "<h1>Relecture du résidu et du hors-or — bras C</h1>",
        f'<p class="meta">Date : {_esc(data.get("date"))} · Durée estimée : une demi-heure · 80 éléments</p>',
        f"<p>{_esc(intro)}</p>",
        f"<p>{_esc(mode)}</p>",
        "</header>",
    ]

    residu_verdicts = [
        (1, "Tenue, autoportante"),
        (2, "Tenue, mais fragment"),
        (3, "Partielle"),
        (4, "Absente"),
    ]
    hors_verdicts = [
        (1, "Vrai et utile"),
        (2, "Reformulation"),
        (3, "Vrai mais fragment"),
        (4, "Faux ou déformé"),
        (5, "Bruit"),
    ]

    parts.append('<section id="partie-a"><h2>Partie A — Le résidu (R01 à R40)</h2>')
    for idx, item in enumerate(data["residu"], start=1):
        card_id = f"r{idx:02d}"
        parts.append(f'<article class="card" id="{card_id}">')
        parts.append(f"<h3>R{idx:02d} · {_esc(item['piece'])} · page {_esc(item.get('page'))}</h3>")
        parts.append('<div class="field"><span class="field-label">Ce que dit l\'or</span>')
        parts.append(f'<p class="field-body">{_esc(item["content"])}</p></div>')
        parts.append('<div class="field"><span class="field-label">Citation</span>')
        parts.append(f'<p class="field-body">{_esc(item["quote"])}</p></div>')
        parts.append('<div class="field"><span class="field-label">Extrait de la source</span>')
        parts.append(f'<div class="scroll-box">{_esc(item["extrait_source"])}</div></div>')
        parts.append('<div class="field"><span class="field-label">Ce que la mémoire a de plus proche</span>')
        for neighbor in item["voisins"]:
            meta = neighbor.get("lignes_source") or ""
            parts.append('<div class="neighbor">')
            if meta:
                parts.append(f'<div class="neighbor-meta">{_esc(meta)}</div>')
            parts.append(_render_triplet_html(neighbor))
            parts.append("</div>")
        if not item["voisins"]:
            parts.append('<p class="field-body">Aucun voisin.</p>')
        parts.append('</div><fieldset class="verdicts"><legend class="field-label">Verdict</legend>')
        for value, label in residu_verdicts:
            input_id = f"{card_id}-v{value}"
            parts.append(
                f'<label for="{input_id}"><input type="radio" name="{card_id}" '
                f'id="{input_id}" value="{value}"> {_esc(label)}</label>'
            )
        parts.append(
            f'<div class="partial-note" id="{card_id}-partial" hidden>'
            f'<label for="{card_id}-note">Ce qui manque</label>'
            f'<input type="text" id="{card_id}-note" maxlength="120"></div>'
        )
        parts.append("</fieldset></article>")
    parts.append("</section>")

    parts.append('<section id="partie-b"><h2>Partie B — Hors-or (H01 à H40)</h2>')
    for idx, item in enumerate(data["hors_or"], start=1):
        card_id = f"h{idx:02d}"
        parts.append(f'<article class="card" id="{card_id}">')
        parts.append(f"<h3>H{idx:02d} · {_esc(item['piece'])} · {_esc(item.get('entity_id'))}</h3>")
        parts.append('<div class="field"><span class="field-label">Fait</span>')
        parts.append(_render_triplet_html(item))
        parts.append("</div>")
        parts.append('<div class="field"><span class="field-label">Extrait de la source</span>')
        parts.append(f'<div class="scroll-box">{_esc(item["extrait_source"])}</div></div>')
        parts.append('<fieldset class="verdicts"><legend class="field-label">Verdict</legend>')
        for value, label in hors_verdicts:
            input_id = f"{card_id}-v{value}"
            parts.append(
                f'<label for="{input_id}"><input type="radio" name="{card_id}" '
                f'id="{input_id}" value="{value}"> {_esc(label)}</label>'
            )
        parts.append("</fieldset></article>")
    parts.append("</section>")

    parts.extend(
        [
            '<section id="verdicts-block">',
            "<h2>Mes verdicts</h2>",
            '<textarea id="verdicts-out" readonly aria-label="Mes verdicts"></textarea>',
            '<button type="button" id="copy-btn">Copier</button>',
            '<p class="help">Colle ce bloc dans la conversation de l\'atelier, '
            "ou donne-le à Claude dans le terminal.</p>",
            "</section>",
            "</div>",
            "<script>",
            f"const SEED = {SEED};",
            "const STORAGE_KEY = 'banc-relecture-' + SEED;",
            "const R_IDS = Array.from({length: 40}, (_, i) => 'r' + String(i + 1).padStart(2, '0'));",
            "const H_IDS = Array.from({length: 40}, (_, i) => 'h' + String(i + 1).padStart(2, '0'));",
            "function loadState() {",
            "  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }",
            "  catch { return {}; }",
            "}",
            "function saveState(state) { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }",
            "function verdictCount(state) {",
            "  return R_IDS.concat(H_IDS).filter(id => state[id] && state[id].v).length;",
            "}",
            "function formatLine(prefix, ids, state) {",
            "  const parts = [];",
            "  ids.forEach((id, i) => {",
            "    const entry = state[id];",
            "    if (!entry || !entry.v) return;",
            "    const num = String(i + 1).padStart(2, '0');",
            "    let token = num + '=' + entry.v;",
            "    if (entry.note) token += '(' + entry.note + ')';",
            "    parts.push(token);",
            "  });",
            "  return prefix + ': ' + parts.join(' ');",
            "}",
            "function refreshOutput() {",
            "  const state = loadState();",
            "  document.getElementById('counter').textContent = verdictCount(state) + ' / 80';",
            "  const lines = [];",
            "  const rLine = formatLine('R', R_IDS, state);",
            "  const hLine = formatLine('H', H_IDS, state);",
            "  if (rLine.length > 2) lines.push(rLine);",
            "  if (hLine.length > 2) lines.push(hLine);",
            "  document.getElementById('verdicts-out').value = lines.join('\\n');",
            "}",
            "function bindCard(id, partial) {",
            "  const radios = document.querySelectorAll('input[name=\"' + id + '\"]');",
            "  const noteEl = partial ? document.getElementById(id + '-note') : null;",
            "  const partialBox = partial ? document.getElementById(id + '-partial') : null;",
            "  function syncFromState() {",
            "    const state = loadState();",
            "    const entry = state[id] || {};",
            "    radios.forEach(r => { r.checked = entry.v === r.value; });",
            "    if (noteEl) noteEl.value = entry.note || '';",
            "    if (partialBox) partialBox.hidden = entry.v !== '3';",
            "  }",
            "  radios.forEach(radio => {",
            "    radio.addEventListener('change', () => {",
            "      const state = loadState();",
            "      state[id] = state[id] || {};",
            "      state[id].v = radio.value;",
            "      if (radio.value !== '3') state[id].note = '';",
            "      saveState(state);",
            "      if (partialBox) partialBox.hidden = radio.value !== '3';",
            "      if (radio.value !== '3' && noteEl) noteEl.value = '';",
            "      refreshOutput();",
            "    });",
            "  });",
            "  if (noteEl) {",
            "    noteEl.addEventListener('input', () => {",
            "      const state = loadState();",
            "      state[id] = state[id] || { v: '3' };",
            "      state[id].note = noteEl.value.trim();",
            "      saveState(state);",
            "      refreshOutput();",
            "    });",
            "  }",
            "  syncFromState();",
            "}",
            "R_IDS.forEach(id => bindCard(id, true));",
            "H_IDS.forEach(id => bindCard(id, false));",
            "document.getElementById('copy-btn').addEventListener('click', () => {",
            "  const text = document.getElementById('verdicts-out').value;",
            "  navigator.clipboard.writeText(text);",
            "});",
            "refreshOutput();",
            "</script>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(parts) + "\n"


def write_outputs(
    *,
    corpus: Path = CORPUS,
    residu_in: Path = RESIDU_IN,
    hors_or_in: Path = HORS_OR_IN,
    data_out: Path = DATA_OUT,
    html_out: Path | None = HTML_OUT,
) -> dict[str, Any]:
    data_out.parent.mkdir(parents=True, exist_ok=True)
    data = build_data(corpus=corpus, residu_in=residu_in, hors_or_in=hors_or_in)
    data_out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if html_out is not None:
        html_out.write_text(render_html(data), encoding="utf-8")
    return data


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Construire les données de relecture du bras C.")
    parser.add_argument(
        "--sorties-dir",
        type=Path,
        default=None,
        help="Dossier contenant residu-a-adjudger.json et hors-or-echantillon.json.",
    )
    parser.add_argument(
        "--sans-html",
        action="store_true",
        help="Ne pas régénérer la page HTML de relecture.",
    )
    args = parser.parse_args()
    sorties = args.sorties_dir or (BANC_ROOT / "sorties")
    residu_in = sorties / "residu-a-adjudger.json"
    hors_or_in = sorties / "hors-or-echantillon.json"
    data_out = sorties / "relecture-donnees.json"
    html_out = None if args.sans_html else HTML_OUT
    data = write_outputs(
        residu_in=residu_in,
        hors_or_in=hors_or_in,
        data_out=data_out,
        html_out=html_out,
    )
    stats = data["stats"]
    payload: dict[str, Any] = {
        "residu": len(data["residu"]),
        "hors_or": len(data["hors_or"]),
        **stats,
        "data": str(data_out),
    }
    if html_out is not None:
        payload["html"] = str(html_out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
