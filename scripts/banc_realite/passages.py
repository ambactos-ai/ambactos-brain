# SPDX-License-Identifier: AGPL-3.0-only
"""Découpage déterministe des pièces du corpus en passages."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import CORPUS

MAX_CHARS = 1200
OVERLAP_CHARS = 150  # défaut historique GreenLoop ; MOREAU calcule depuis l'or
MARGE_RECUVREMENT_MIN = 10
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def calculer_recouvrement_depuis_or(
    assertions: list[dict[str, Any]],
    *,
    max_chars: int | None = MAX_CHARS,
) -> dict[str, Any]:
    """Dimensionne le recouvrement pour que la plus longue citation tienne dans un passage."""
    longest_len = 0
    longest_id = ""
    for assertion in assertions:
        quote = str(assertion.get("quote") or assertion.get("content") or "")
        if len(quote) > longest_len:
            longest_len = len(quote)
            longest_id = str(assertion.get("id") or "")
    marge = max(MARGE_RECUVREMENT_MIN, longest_len // 20)
    # Une citation de longueur L à cheval sur deux passages exige overlap >= L - 1 ;
    # la marge compense le rognage au saut de ligne dans _overlap_tail.
    brut = 0 if longest_len <= 1 else longest_len - 1 + marge
    if max_chars is None:
        recouvrement = brut
        calcul = (
            f"{longest_len} - 1 + {marge} = {recouvrement} "
            f"— citation la plus longue {longest_id!r} ({longest_len} car.)"
        )
    else:
        recouvrement = 0 if longest_len <= 1 else min(max_chars - 1, brut)
        calcul = (
            f"max(0, min({max_chars - 1}, {longest_len} - 1 + {marge})) "
            f"— citation la plus longue {longest_id!r} ({longest_len} car.)"
        )
    passage_ref = max_chars if max_chars is not None else MAX_CHARS
    passage_trop_petit = recouvrement > passage_ref // 2
    return {
        "recouvrement_chars": recouvrement,
        "citation_plus_longue_chars": longest_len,
        "citation_plus_longue_id": longest_id,
        "marge_chars": marge,
        "max_chars_passage": passage_ref,
        "passage_trop_petit": passage_trop_petit,
        "calcul": calcul,
    }


@dataclass(frozen=True)
class Passage:
    piece: str
    content: str
    line_start: int
    line_end: int
    section_path: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "piece": self.piece,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "section_path": self.section_path,
        }


def stats_duplication(
    passages: list[Passage],
    *,
    corpus_dir: Path,
    max_chars: int = MAX_CHARS,
) -> dict[str, Any]:
    """Mesure les caractères dupliqués par le recouvrement entre passages."""
    corpus_chars = 0
    pieces = {p.piece for p in passages}
    for piece in pieces:
        path = corpus_dir / piece
        if path.is_file():
            corpus_chars += len(path.read_text(encoding="utf-8"))
    passage_chars = sum(len(p.content) for p in passages)
    dupliques = max(0, passage_chars - corpus_chars)
    pct = round(100.0 * dupliques / passage_chars, 2) if passage_chars else 0.0
    passages_hors_gabarit = sum(1 for p in passages if len(p.content) > max_chars)
    return {
        "caracteres_corpus": corpus_chars,
        "caracteres_passages": passage_chars,
        "caracteres_dupliques": dupliques,
        "duplication_pct": pct,
        "passages_hors_gabarit": passages_hors_gabarit,
    }


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and "|" in stripped[1:]


def _heading_level(line: str) -> int | None:
    match = HEADING_RE.match(line.strip())
    if not match:
        return None
    return len(match.group(1))


def _heading_title(line: str) -> str:
    match = HEADING_RE.match(line.strip())
    return match.group(2).strip() if match else ""


def _iter_blocks(lines: list[str]) -> Iterator[tuple[str, list[tuple[int, str]], str]]:
    """Yield (section_path, [(line_no, text), ...], joined_text) content blocks."""
    section_titles: list[str] = []
    buffer: list[tuple[int, str]] = []
    buffer_section = ""

    def flush() -> Iterator[tuple[str, list[tuple[int, str]], str]]:
        nonlocal buffer, buffer_section
        if buffer:
            text = "\n".join(line for _, line in buffer)
            yield buffer_section, list(buffer), text
            buffer = []

    idx = 0
    while idx < len(lines):
        line_no = idx + 1
        line = lines[idx]
        level = _heading_level(line)
        if level is not None:
            yield from flush()
            title = _heading_title(line)
            if level <= len(section_titles):
                section_titles = section_titles[: level - 1]
            section_titles.append(title)
            buffer_section = " / ".join(section_titles)
            buffer = [(line_no, line)]
            idx += 1
            continue

        if _is_table_row(line):
            yield from flush()
            table_rows: list[tuple[int, str]] = [(line_no, line)]
            idx += 1
            while idx < len(lines) and _is_table_row(lines[idx]):
                table_rows.append((idx + 1, lines[idx]))
                idx += 1
            text = "\n".join(row for _, row in table_rows)
            yield buffer_section, table_rows, text
            continue

        if not line.strip():
            yield from flush()
            idx += 1
            continue

        if not buffer:
            buffer_section = " / ".join(section_titles)
            buffer = [(line_no, line)]
        else:
            buffer.append((line_no, line))
        idx += 1

    yield from flush()


def _split_table_block(
    section_path: str,
    rows: list[tuple[int, str]],
    *,
    max_chars: int = MAX_CHARS,
) -> list[tuple[str, list[tuple[int, str]], str]]:
    """Split a table block between rows, never within a row."""
    if not rows:
        return []
    chunks: list[tuple[str, list[tuple[int, str]], str]] = []
    current: list[tuple[int, str]] = []
    current_len = 0
    for row in rows:
        row_text = row[1]
        row_len = len(row_text) + (1 if current else 0)
        if current and current_len + row_len > max_chars:
            text = "\n".join(line for _, line in current)
            chunks.append((section_path, current, text))
            current = [row]
            current_len = len(row_text)
        else:
            current.append(row)
            current_len += row_len
    if current:
        text = "\n".join(line for _, line in current)
        chunks.append((section_path, current, text))
    return chunks


def _overlap_tail(text: str, overlap_chars: int) -> str:
    if not text or overlap_chars <= 0:
        return ""
    overlap = text[-overlap_chars:] if len(text) > overlap_chars else text
    if overlap and not overlap.startswith("\n"):
        cut = overlap.find("\n")
        if cut >= 0:
            overlap = overlap[cut + 1 :]
    return overlap


def _line_start_from_rows(rows: list[tuple[int, str]], first_line: str) -> int:
    for line_no, text in reversed(rows):
        if text.strip() == first_line.strip():
            return line_no
    return rows[0][0] if rows else 1


def _line_start_for_overlap(prev: Passage, overlap: str, *, rows: list[tuple[int, str]] | None = None) -> int:
    if not overlap:
        return prev.line_start
    first_line = overlap.splitlines()[0]
    if rows:
        return _line_start_from_rows(rows, first_line)
    lines = prev.content.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == first_line.strip():
            return prev.line_start + i
    return prev.line_start


def _rows_from_overlap_tail(
    rows: list[tuple[int, str]],
    overlap_chars: int,
) -> list[tuple[int, str]]:
    if not rows or overlap_chars <= 0:
        return []
    text = "\n".join(line for _, line in rows)
    overlap = _overlap_tail(text, overlap_chars)
    if not overlap:
        return []
    first_line = overlap.splitlines()[0]
    for idx, (_, line) in enumerate(rows):
        if line.strip() == first_line.strip():
            return rows[idx:]
    return rows


def _shrink_overlap_for_unit(
    overlap_rows: list[tuple[int, str]],
    unit_text: str,
    *,
    max_chars: int,
) -> list[tuple[int, str]]:
    """Raccourcit le recouvrement (frontière de ligne) pour que l'unité entière tienne."""
    if len(unit_text) > max_chars:
        return []
    rows = list(overlap_rows)
    while rows:
        overlap_text = "\n".join(line for _, line in rows)
        combined = f"{overlap_text}\n{unit_text}" if overlap_text else unit_text
        if len(combined) <= max_chars:
            return rows
        rows = rows[1:]
    return []


def lignes_non_couvertes(
    corpus_dir: Path,
    passages: list[Passage],
) -> list[tuple[str, int]]:
    """Lignes non vides d'une pièce absentes de tout passage (texte verbatim)."""
    by_piece: dict[str, list[Passage]] = {}
    for passage in passages:
        by_piece.setdefault(passage.piece, []).append(passage)

    def ligne_dans_passages(line: str, piece_passages: list[Passage]) -> bool:
        for passage in piece_passages:
            if line in passage.content.splitlines():
                return True
        return False

    missing: list[tuple[str, int]] = []
    if not corpus_dir.is_dir():
        return missing
    for path in sorted(corpus_dir.iterdir()):
        if not path.is_file():
            continue
        piece = path.name
        piece_passages = by_piece.get(piece, [])
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            if not ligne_dans_passages(line, piece_passages):
                missing.append((piece, line_no))
    return missing


def _pack_units(
    piece: str,
    units: list[tuple[str, list[tuple[int, str]], str]],
    *,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Passage]:
    passages: list[Passage] = []
    current_section = ""
    current_rows: list[tuple[int, str]] = []
    current_text = ""
    last_emitted_rows: list[tuple[int, str]] = []

    def emit() -> None:
        nonlocal current_rows, current_text, current_section, last_emitted_rows
        if not current_text:
            return
        line_start = current_rows[0][0]
        line_end = current_rows[-1][0]
        passages.append(
            Passage(
                piece=piece,
                content=current_text,
                line_start=line_start,
                line_end=line_end,
                section_path=current_section,
            )
        )
        last_emitted_rows = list(current_rows)
        current_rows = []
        current_text = ""

    def append_unit(section_path: str, rows: list[tuple[int, str]], text: str) -> None:
        nonlocal current_section, current_rows, current_text
        if not text:
            return
        if current_text and len(current_text) + 1 + len(text) > max_chars:
            emit()
            if passages and overlap_chars > 0:
                overlap_rows = _rows_from_overlap_tail(
                    last_emitted_rows, overlap_chars
                )
                overlap_rows = _shrink_overlap_for_unit(
                    overlap_rows, text, max_chars=max_chars
                )
                if overlap_rows:
                    current_rows = list(overlap_rows)
                    current_text = "\n".join(line for _, line in overlap_rows)
                else:
                    current_text = ""
                    current_rows = []
            current_section = section_path
        if not current_text:
            current_section = section_path
            current_rows = list(rows)
            current_text = text
        else:
            current_rows.extend(rows)
            current_text = f"{current_text}\n{text}"

    for section_path, rows, text in units:
        if all(_is_table_row(line) for _, line in rows) and len(text) > max_chars:
            for chunk_section, chunk_rows, chunk_text in _split_table_block(
                section_path, rows, max_chars=max_chars
            ):
                append_unit(chunk_section, chunk_rows, chunk_text)
        elif len(text) > max_chars:
            for line_no, line in rows:
                append_unit(section_path, [(line_no, line)], line)
        else:
            append_unit(section_path, rows, text)

    emit()
    return passages


def _chunk_csv_piece(
    piece: str,
    lines: list[str],
    *,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Passage]:
    rows = [(idx + 1, line) for idx, line in enumerate(lines) if line.strip()]
    if not rows:
        return []
    units: list[tuple[str, list[tuple[int, str]], str]] = []
    for section_path, chunk_rows, text in _split_table_block("", rows, max_chars=max_chars):
        units.append((section_path, chunk_rows, text))
    return _pack_units(piece, units, max_chars=max_chars, overlap_chars=overlap_chars)


def chunk_piece(
    piece: str,
    corpus: Path = CORPUS,
    *,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Passage]:
    path = corpus / piece
    lines = path.read_text(encoding="utf-8").splitlines()
    if piece.endswith(".csv"):
        return _chunk_csv_piece(
            piece, lines, max_chars=max_chars, overlap_chars=overlap_chars
        )
    units: list[tuple[str, list[tuple[int, str]], str]] = []
    for section_path, rows, text in _iter_blocks(lines):
        if text.strip():
            units.append((section_path, rows, text))
    return _pack_units(
        piece, units, max_chars=max_chars, overlap_chars=overlap_chars
    )


def chunk_corpus(
    corpus: Path = CORPUS,
    *,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Passage]:
    pieces = sorted(p.name for p in corpus.iterdir() if p.is_file())
    out: list[Passage] = []
    for piece in pieces:
        out.extend(
            chunk_piece(
                piece,
                corpus=corpus,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )
    return out


def compter_pieces_un_seul_passage(
    passages: list[Passage],
    corpus_dir: Path,
) -> int:
    """Pièces dont le texte source tient en un seul passage (zéro recouvrement interne)."""
    by_piece: dict[str, int] = {}
    for passage in passages:
        by_piece[passage.piece] = by_piece.get(passage.piece, 0) + 1
    count = 0
    for piece, n_passages in by_piece.items():
        if n_passages != 1:
            continue
        path = corpus_dir / piece
        if path.is_file():
            count += 1
    return count
