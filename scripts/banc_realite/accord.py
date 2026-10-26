# SPDX-License-Identifier: AGPL-3.0-only
"""Accord entre lecteurs A et B, page des désaccords."""
from __future__ import annotations
from banc_realite.dossiers import BANC_VAR

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

BANC_ROOT = (BANC_VAR)
SORTIES_DIR = BANC_ROOT / "sorties" / "ingestion-2"
ACCORD_PATH = SORTIES_DIR / "relecture-accord.json"
CLE_PATH = SORTIES_DIR / "desaccords-cle.json"
PAGE_PATH = BANC_ROOT / "desaccords.html"
DONNEES_PATH = SORTIES_DIR / "relecture-donnees.json"

SEED = 20260920
NOTE_MARGES = (
    "Les marges sont très déséquilibrées (34 / 40 dans une classe côté A) : "
    "le kappa peut être bas avec un accord brut élevé ; lire la matrice"
)

VERDICT_LABELS_RELECTURE = {
    "residu": {
        "1": "Tenue, autoportante",
        "2": "Tenue, mais fragment",
        "3": "Partielle",
        "4": "Absente",
        "illisible": "Illisible",
    },
    "hors_or": {
        "V": "Vrai",
        "3": "Vrai mais fragment",
        "4": "Faux ou déformé",
        "5": "Bruit",
        "illisible": "Illisible",
    },
}
VERDICT_LABELS_ASSERTION = {
    "1": "Tient",
    "3": "Tient en partie",
    "4": "Ne tient pas",
    "illisible": "Illisible",
}


def cohens_kappa(counts: dict[str, dict[str, int]], labels: list[str]) -> float:
    total = sum(counts[a][b] for a in labels for b in labels)
    if total == 0:
        return 0.0
    observed = sum(counts[label][label] for label in labels) / total
    row_marginals = {label: sum(counts[label][b] for b in labels) / total for label in labels}
    col_marginals = {label: sum(counts[a][label] for a in labels) / total for label in labels}
    expected = sum(row_marginals[label] * col_marginals[label] for label in labels)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def confusion_matrix(
    pairs: list[tuple[str, str]],
    labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for left, right in pairs:
        matrix[left][right] += 1
    return matrix


def brut_accord(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for left, right in pairs if left == right) / len(pairs)


def binary_group(partie: str, verdict: str) -> str:
    if partie == "residu":
        return "tenue" if verdict in {"1", "2"} else "non_tenue"
    if partie == "hors_or":
        return "vrai" if verdict in {"V", "3"} else "pas_vrai"
    return "tient" if verdict == "1" else "ne_tient_pas"


def extract_verdict_map(payload: dict[str, Any], *, jeu: str) -> dict[str, str]:
    if jeu == "relecture":
        out: dict[str, str] = {}
        for section in ("residu", "hors_or"):
            for ident, verdict in (payload.get(section) or {}).items():
                value = verdict[0] if isinstance(verdict, list) else verdict
                out[str(ident)] = str(value)
        return out
    if jeu == "assertions":
        return {str(key): str(value) for key, value in (payload.get("par_assertion") or {}).items()}
    raise ValueError(f"jeu inconnu: {jeu}")


def justification_map(payload: dict[str, Any], *, jeu: str) -> dict[str, str]:
    if jeu == "relecture":
        out: dict[str, str] = {}
        for section in ("residu", "hors_or"):
            for ident, verdict in (payload.get(section) or {}).items():
                if isinstance(verdict, list) and len(verdict) > 1:
                    out[str(ident)] = str(verdict[1] or "")
        return out
    out = {}
    for row in payload.get("paires") or []:
        out[str(row.get("id") or f"A-{row.get('assertion')}")] = str(row.get("note") or "")
    return out


def expected_relecture_ids() -> list[str]:
    return [f"R{idx:02d}" for idx in range(1, 41)] + [f"H{idx:02d}" for idx in range(1, 41)]


def is_complete_b(payload: dict[str, Any], *, jeu: str) -> bool:
    if payload.get("complet") is False:
        return False
    if jeu == "relecture":
        ids = set(expected_relecture_ids())
        found = set()
        for section in ("residu", "hors_or"):
            found.update((payload.get(section) or {}).keys())
        return ids.issubset(found)
    par = payload.get("par_assertion") or {}
    return bool(par)


def analyze_relecture_partie(
    *,
    partie: str,
    ids: list[str],
    a_map: dict[str, str],
    b_map: dict[str, str],
    just_a: dict[str, str],
    just_b: dict[str, str],
) -> dict[str, Any]:
    present = [ident for ident in ids if ident in a_map and ident in b_map]
    pairs = [(a_map[ident], b_map[ident]) for ident in present]
    labels = sorted({value for pair in pairs for value in pair})
    matrix = confusion_matrix(pairs, labels)
    binary_pairs = [
        (binary_group(partie, a_map[ident]), binary_group(partie, b_map[ident])) for ident in present
    ]
    binary_labels = sorted({value for pair in binary_pairs for value in pair})
    binary_matrix = confusion_matrix(binary_pairs, binary_labels)
    desaccords = []
    desaccords_binaires = []
    for ident in present:
        if a_map[ident] != b_map[ident]:
            desaccords.append(
                {
                    "id": ident,
                    "partie": partie,
                    "lecteur_a": a_map[ident],
                    "lecteur_b": b_map[ident],
                    "justification_a": just_a.get(ident, ""),
                    "justification_b": just_b.get(ident, ""),
                }
            )
        ba = binary_group(partie, a_map[ident])
        bb = binary_group(partie, b_map[ident])
        if ba != bb:
            desaccords_binaires.append(
                {
                    "id": ident,
                    "partie": partie,
                    "binaire_a": ba,
                    "binaire_b": bb,
                }
            )
    return {
        "n": len(present),
        "accord_brut": round(brut_accord(pairs), 4),
        "kappa": round(cohens_kappa(matrix, labels), 4),
        "matrice": matrix,
        "binaire": {
            "accord_brut": round(brut_accord(binary_pairs), 4),
            "kappa": round(cohens_kappa(binary_matrix, binary_labels), 4),
            "matrice": binary_matrix,
        },
        "desaccords": desaccords,
        "desaccords_binaires": desaccords_binaires,
    }


def analyze_game(
    *,
    jeu: str,
    lecteur_a: dict[str, Any],
    lecteur_b: dict[str, Any],
    partie_lookup: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    if not is_complete_b(lecteur_b, jeu=jeu):
        return None
    a_map = extract_verdict_map(lecteur_a, jeu=jeu)
    b_map = extract_verdict_map(lecteur_b, jeu=jeu)
    just_a = justification_map(lecteur_a, jeu=jeu)
    just_b = justification_map(lecteur_b, jeu=jeu)
    if jeu == "relecture":
        lookup = partie_lookup or partie_lookup_for_relecture()
        residu_ids = [ident for ident, partie in lookup.items() if partie == "residu"]
        hors_or_ids = [ident for ident, partie in lookup.items() if partie == "hors_or"]
        residu = analyze_relecture_partie(
            partie="residu",
            ids=residu_ids,
            a_map=a_map,
            b_map=b_map,
            just_a=just_a,
            just_b=just_b,
        )
        hors_or = analyze_relecture_partie(
            partie="hors_or",
            ids=hors_or_ids,
            a_map=a_map,
            b_map=b_map,
            just_a=just_a,
            just_b=just_b,
        )
        desaccords = residu["desaccords"] + hors_or["desaccords"]
        desaccords_binaires = residu["desaccords_binaires"] + hors_or["desaccords_binaires"]
        return {
            "disponible": True,
            "n": residu["n"] + hors_or["n"],
            "residu": residu,
            "hors_or": hors_or,
            "desaccords": desaccords,
            "desaccords_binaires": desaccords_binaires,
        }
    ids = sorted(set(a_map) & set(b_map))
    pairs = [(a_map[ident], b_map[ident]) for ident in ids]
    labels = sorted({value for pair in pairs for value in pair})
    matrix = confusion_matrix(pairs, labels)
    binary_pairs = []
    for ident in ids:
        partie = (partie_lookup or {}).get(ident, "assertions")
        binary_pairs.append((binary_group(partie, a_map[ident]), binary_group(partie, b_map[ident])))
    binary_labels = sorted({value for pair in binary_pairs for value in pair})
    binary_matrix = confusion_matrix(binary_pairs, binary_labels)
    desaccords = []
    desaccords_binaires = []
    for ident in ids:
        partie = (partie_lookup or {}).get(ident, "assertions")
        if a_map[ident] != b_map[ident]:
            desaccords.append(
                {
                    "id": ident,
                    "partie": partie,
                    "lecteur_a": a_map[ident],
                    "lecteur_b": b_map[ident],
                    "justification_a": just_a.get(ident, ""),
                    "justification_b": just_b.get(ident, ""),
                }
            )
        ba = binary_group(partie, a_map[ident])
        bb = binary_group(partie, b_map[ident])
        if ba != bb:
            desaccords_binaires.append(
                {
                    "id": ident,
                    "partie": partie,
                    "binaire_a": ba,
                    "binaire_b": bb,
                }
            )
    return {
        "disponible": True,
        "n": len(ids),
        "accord_brut": round(brut_accord(pairs), 4),
        "kappa": round(cohens_kappa(matrix, labels), 4),
        "matrice": matrix,
        "binaire": {
            "accord_brut": round(brut_accord(binary_pairs), 4),
            "kappa": round(cohens_kappa(binary_matrix, binary_labels), 4),
            "matrice": binary_matrix,
        },
        "desaccords": desaccords,
        "desaccords_binaires": desaccords_binaires,
    }


def partie_lookup_for_relecture() -> dict[str, str]:
    lookup: dict[str, str] = {}
    lookup.update({f"R{idx:02d}": "residu" for idx in range(1, 41)})
    lookup.update({f"H{idx:02d}": "hors_or" for idx in range(1, 41)})
    return lookup


def compute_accord(
    *,
    lecteur_a_relecture_path: Path,
    lecteur_b_relecture_path: Path,
    lecteur_a_assertions_path: Path,
    lecteur_b_assertions_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "seed": SEED,
        "note_marges": NOTE_MARGES,
        "relecture": {"disponible": False},
        "assertions": {"disponible": False},
    }
    if lecteur_a_relecture_path.is_file() and lecteur_b_relecture_path.is_file():
        relecture = analyze_game(
            jeu="relecture",
            lecteur_a=json.loads(lecteur_a_relecture_path.read_text(encoding="utf-8")),
            lecteur_b=json.loads(lecteur_b_relecture_path.read_text(encoding="utf-8")),
            partie_lookup=partie_lookup_for_relecture(),
        )
        if relecture is not None:
            payload["relecture"] = relecture
    if lecteur_a_assertions_path.is_file() and lecteur_b_assertions_path.is_file():
        assertions = analyze_game(
            jeu="assertions",
            lecteur_a=json.loads(lecteur_a_assertions_path.read_text(encoding="utf-8")),
            lecteur_b=json.loads(lecteur_b_assertions_path.read_text(encoding="utf-8")),
        )
        if assertions is not None:
            payload["assertions"] = assertions
    return payload


def shuffle_lecture_order(disagreements: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(SEED)
    cle: dict[str, int] = {}
    cards: list[dict[str, Any]] = []
    for idx, row in enumerate(disagreements, start=1):
        order = [1, 2]
        rng.shuffle(order)
        lecteur_1_is_a = order[0] == 1
        cle[row["id"]] = 1 if lecteur_1_is_a else 2
        cards.append(
            {
                "card_id": f"d{idx:02d}",
                "source_id": row["id"],
                "partie": row["partie"],
                "lecture_1_verdict": row["lecteur_a"] if lecteur_1_is_a else row["lecteur_b"],
                "lecture_1_note": row["justification_a"] if lecteur_1_is_a else row["justification_b"],
                "lecture_2_verdict": row["lecteur_b"] if lecteur_1_is_a else row["lecteur_a"],
                "lecture_2_note": row["justification_b"] if lecteur_1_is_a else row["justification_a"],
            }
        )
    return cards, cle


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def render_disagreements_page(
    *,
    accord: dict[str, Any],
    donnees: dict[str, Any],
    cards: list[dict[str, Any]],
) -> str:
    relecture = accord.get("relecture") or {}
    desaccords = relecture.get("desaccords") or []
    n_accord = (relecture.get("n") or 0) - len(desaccords)
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
    padding: 2rem 1.25rem 4rem;
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
  header.doc-header h1 { margin: 0 0 1rem; font-size: 1.75rem; font-weight: 700; }
  .meta { font-size: 0.875rem; color: var(--muted); margin-bottom: 1rem; }
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
  .lecture {
    margin: 0.65rem 0;
    padding: 0.65rem 0.75rem;
    border-left: 3px solid var(--decision-border);
    background: var(--decision-bg);
  }
  .verdicts label { display: block; margin: 0.35rem 0; cursor: pointer; }
  #reponses-block {
    margin-top: 3rem;
    padding: 1.5rem;
    background: var(--surface);
    border: 2px solid var(--decision-border);
  }
  #reponses-out {
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
  .neighbor {
    margin: 0.5rem 0;
    padding: 0.5rem 0.65rem;
    border-left: 3px solid var(--decision-border);
    background: var(--decision-bg);
  }
"""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="fr">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Désaccords entre lecteurs — bras C</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        '<div class="page">',
        '<header class="doc-header">',
        "<h1>Désaccords entre lecteurs</h1>",
        '<p class="meta">Graine ' + _esc(SEED) + "</p>",
        "<p>Deux lecteurs ont jugé séparément les mêmes 80 éléments.</p>",
        f"<p>Ils sont d'accord sur {_esc(n_accord)}.</p>",
        f"<p>Voici les {_esc(len(desaccords))} où ils ne le sont pas.</p>",
        "<p>Pour chacun, lis l'extrait et dis laquelle des deux lectures tu retiens : "
        "tu n'as pas besoin de connaître le dossier.</p>",
        f"<p>Résidu — accord brut : {_esc(relecture.get('residu', {}).get('accord_brut'))} · "
        f"kappa : {_esc(relecture.get('residu', {}).get('kappa'))} · "
        f"kappa binaire : {_esc(relecture.get('residu', {}).get('binaire', {}).get('kappa'))}</p>",
        f"<p>Hors-or — accord brut : {_esc(relecture.get('hors_or', {}).get('accord_brut'))} · "
        f"kappa : {_esc(relecture.get('hors_or', {}).get('kappa'))} · "
        f"kappa binaire : {_esc(relecture.get('hors_or', {}).get('binaire', {}).get('kappa'))}</p>",
        "</header>",
    ]
    if not cards:
        parts.append("<p>Aucun désaccord : les deux lecteurs sont d'accord sur tous les éléments.</p>")
    lookup_residu = {f"R{idx:02d}": item for idx, item in enumerate(donnees["residu"], start=1)}
    lookup_hors = {f"H{idx:02d}": item for idx, item in enumerate(donnees["hors_or"], start=1)}
    for card in cards:
        ident = card["source_id"]
        partie = card["partie"]
        parts.append(f'<article class="card" id="{_esc(card["card_id"])}">')
        parts.append(f"<h3>{_esc(ident)}</h3>")
        if partie == "residu":
            item = lookup_residu[ident]
            parts.append('<div class="field"><span class="field-label">Ce que dit l\'or</span>')
            parts.append(f'<p class="field-body">{_esc(item["content"])}</p></div>')
            parts.append('<div class="field"><span class="field-label">Citation</span>')
            parts.append(f'<p class="field-body">{_esc(item["quote"])}</p></div>')
            parts.append('<div class="field"><span class="field-label">Extrait de la source</span>')
            parts.append(f'<div class="scroll-box">{_esc(item["extrait_source"])}</div></div>')
            parts.append('<div class="field"><span class="field-label">Ce que la mémoire a de plus proche</span>')
            for neighbor in item.get("voisins") or []:
                triplet = neighbor.get("triplet") or neighbor.get("texte") or ""
                parts.append(f'<div class="neighbor">{_esc(triplet)}</div>')
            parts.append("</div>")
            label_map = VERDICT_LABELS_RELECTURE["residu"]
        else:
            item = lookup_hors[ident]
            triplet = item.get("triplet") or item.get("texte") or ""
            parts.append('<div class="field"><span class="field-label">Fait</span>')
            parts.append(f'<p class="field-body">{_esc(triplet)}</p></div>')
            parts.append('<div class="field"><span class="field-label">Extrait de la source</span>')
            parts.append(f'<div class="scroll-box">{_esc(item["extrait_source"])}</div></div>')
            label_map = VERDICT_LABELS_RELECTURE["hors_or"]
        parts.append('<div class="lecture"><span class="field-label">Lecture 1</span>')
        parts.append(
            f'<p class="field-body">{_esc(label_map.get(card["lecture_1_verdict"], card["lecture_1_verdict"]))}'
            f" — {_esc(card['lecture_1_note'])}</p></div>"
        )
        parts.append('<div class="lecture"><span class="field-label">Lecture 2</span>')
        parts.append(
            f'<p class="field-body">{_esc(label_map.get(card["lecture_2_verdict"], card["lecture_2_verdict"]))}'
            f" — {_esc(card['lecture_2_note'])}</p></div>"
        )
        parts.append('<fieldset class="verdicts"><legend class="field-label">Tu retiens</legend>')
        for value, label in (("1", "Lecture 1"), ("2", "Lecture 2"), ("n", "Ni l'une ni l'autre"), ("?", "Je ne peux pas trancher")):
            input_id = f"{card['card_id']}-v{value}"
            parts.append(
                f'<label for="{input_id}"><input type="radio" name="{card["card_id"]}" '
                f'id="{input_id}" value="{value}"> {_esc(label)}</label>'
            )
        parts.append(
            f'<div class="field"><label for="{card["card_id"]}-note">Note facultative</label>'
            f'<input type="text" id="{card["card_id"]}-note" maxlength="120"></div>'
        )
        parts.append("</fieldset></article>")
    parts.extend(
        [
            '<section id="reponses-block">',
            "<h2>Mes réponses</h2>",
            '<textarea id="reponses-out" readonly aria-label="Mes réponses"></textarea>',
            '<button type="button" id="copy-btn">Copier</button>',
            "</section>",
            "</div>",
            "<script>",
            f"const SEED = {SEED};",
            "const STORAGE_KEY = 'banc-desaccords-' + SEED;",
            f"const CARD_IDS = {json.dumps([card['card_id'] for card in cards])};",
            "function loadState() {",
            "  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }",
            "  catch { return {}; }",
            "}",
            "function saveState(state) { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }",
            "function refreshOutput() {",
            "  const state = loadState();",
            "  const tokens = CARD_IDS.map((id, i) => {",
            "    const entry = state[id];",
            "    if (!entry || !entry.v) return null;",
            "    const num = String(i + 1).padStart(2, '0');",
            "    let token = num + '=' + entry.v;",
            "    if (entry.note) token += '(' + entry.note + ')';",
            "    return token;",
            "  }).filter(Boolean);",
            "  document.getElementById('reponses-out').value = 'D: ' + tokens.join(' ');",
            "}",
            "CARD_IDS.forEach(id => {",
            "  const radios = document.querySelectorAll('input[name=\"' + id + '\"]');",
            "  const noteEl = document.getElementById(id + '-note');",
            "  function syncFromState() {",
            "    const state = loadState();",
            "    const entry = state[id] || {};",
            "    radios.forEach(r => { r.checked = entry.v === r.value; });",
            "    if (noteEl) noteEl.value = entry.note || '';",
            "  }",
            "  radios.forEach(radio => {",
            "    radio.addEventListener('change', () => {",
            "      const state = loadState();",
            "      state[id] = state[id] || {};",
            "      state[id].v = radio.value;",
            "      saveState(state);",
            "      refreshOutput();",
            "    });",
            "  });",
            "  if (noteEl) {",
            "    noteEl.addEventListener('input', () => {",
            "      const state = loadState();",
            "      state[id] = state[id] || {};",
            "      state[id].note = noteEl.value.trim();",
            "      saveState(state);",
            "      refreshOutput();",
            "    });",
            "  }",
            "  syncFromState();",
            "});",
            "document.getElementById('copy-btn').addEventListener('click', () => {",
            "  navigator.clipboard.writeText(document.getElementById('reponses-out').value);",
            "});",
            "refreshOutput();",
            "</script>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(parts) + "\n"


def write_accord_outputs(
    *,
    accord_path: Path = ACCORD_PATH,
    cle_path: Path = CLE_PATH,
    page_path: Path = PAGE_PATH,
    donnees_path: Path = DONNEES_PATH,
    lecteur_a_relecture_path: Path = SORTIES_DIR / "relecture-verdicts-claude.json",
    lecteur_b_relecture_path: Path = SORTIES_DIR / "relecture-verdicts-lecteur-b.json",
    lecteur_a_assertions_path: Path = SORTIES_DIR / "etage2-paires-verdicts-claude.json",
    lecteur_b_assertions_path: Path = SORTIES_DIR / "paires-verdicts-lecteur-b.json",
    write_page: bool = False,
) -> dict[str, Any]:
    accord = compute_accord(
        lecteur_a_relecture_path=lecteur_a_relecture_path,
        lecteur_b_relecture_path=lecteur_b_relecture_path,
        lecteur_a_assertions_path=lecteur_a_assertions_path,
        lecteur_b_assertions_path=lecteur_b_assertions_path,
    )
    accord_path.parent.mkdir(parents=True, exist_ok=True)
    accord_path.write_text(json.dumps(accord, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result: dict[str, Any] = {"accord": str(accord_path)}
    if write_page:
        relecture = accord.get("relecture") or {}
        desaccords = relecture.get("desaccords") or []
        cards, cle = shuffle_lecture_order(desaccords)
        cle_path.write_text(json.dumps({"seed": SEED, "lecture_1": cle}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        donnees = json.loads(donnees_path.read_text(encoding="utf-8"))
        page_path.write_text(
            render_disagreements_page(accord=accord, donnees=donnees, cards=cards),
            encoding="utf-8",
        )
        result["page"] = str(page_path)
        result["cle"] = str(cle_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculer l'accord entre lecteurs A et B.")
    parser.add_argument("--page", action="store_true", help="Générer la page HTML des désaccords.")
    args = parser.parse_args()
    result = write_accord_outputs(write_page=args.page)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
