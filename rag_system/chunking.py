from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import DocumentChunk

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WORD_RE = re.compile(r"\S+")


@dataclass
class _MarkdownSection:
    heading_path: list[str]
    start_line: int
    end_line: int
    text: str


def _clean_heading(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:240]


def _iter_sections(markdown_text: str) -> Iterable[_MarkdownSection]:
    lines = markdown_text.splitlines()
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_start = 1

    def flush(end_line: int) -> _MarkdownSection | None:
        if not current_lines or not any(line.strip() for line in current_lines):
            return None
        has_body = any(line.strip() and not HEADING_RE.match(line) for line in current_lines)
        if not has_body:
            return None
        return _MarkdownSection(
            heading_path=list(heading_stack),
            start_line=current_start,
            end_line=end_line,
            text="\n".join(current_lines).strip(),
        )

    for line_no, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if match:
            section = flush(line_no - 1)
            prefix_lines = []
            prefix_start = current_start
            if section is not None:
                yield section
            elif current_lines:
                prefix_lines = [previous_line for previous_line in current_lines if previous_line.strip()]

            level = len(match.group(1))
            title = _clean_heading(match.group(2))
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_lines = [*prefix_lines, line]
            current_start = prefix_start if prefix_lines else line_no
        else:
            if not current_lines:
                current_start = line_no
            current_lines.append(line)

    section = flush(len(lines))
    if section is not None:
        yield section


def _word_windows(words: list[str], size: int, overlap: int) -> Iterable[tuple[int, int, list[str]]]:
    if size <= 0:
        raise ValueError("chunk_words debe ser mayor que 0")
    if overlap < 0:
        raise ValueError("chunk_overlap_words no puede ser negativo")
    if overlap >= size:
        raise ValueError("chunk_overlap_words debe ser menor que chunk_words")

    step = size - overlap
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        yield start, end, words[start:end]
        if end == len(words):
            break
        start += step


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_markdown_by_words(
    markdown_path: str | Path,
    chunk_words: int = 220,
    chunk_overlap_words: int = 40,
) -> list[DocumentChunk]:
    """Segmenta Markdown en chunks estaticos por palabras, respetando secciones Markdown.

    La division por numero de palabras se aplica dentro de cada seccion para evitar mezclar,
    por ejemplo, dos alarmas distintas del manual en un mismo fragmento.
    """

    path = Path(markdown_path).expanduser().resolve()
    markdown_text = path.read_text(encoding="utf-8")
    source = path.name
    chunks: list[DocumentChunk] = []

    for section in _iter_sections(markdown_text):
        words = WORD_RE.findall(section.text)
        if not words:
            continue

        section_name = section.heading_path[-1] if section.heading_path else "Document start"
        for _, _, window_words in _word_windows(words, chunk_words, chunk_overlap_words):
            chunk_text = " ".join(window_words).strip()
            if not chunk_text:
                continue

            chunk_index = len(chunks) + 1
            chunk_id = f"C{chunk_index:06d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=source,
                    source_path=str(path),
                    chunk_index=chunk_index,
                    text=chunk_text,
                    section=section_name,
                    section_path=list(section.heading_path),
                    start_line=section.start_line,
                    end_line=section.end_line,
                    word_count=len(window_words),
                    content_sha256=_sha256(chunk_text),
                )
            )

    return chunks


def write_chunks_jsonl(chunks: Iterable[DocumentChunk], output_path: str | Path) -> None:
    path = Path(output_path).expanduser().resolve()
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_weaviate_properties(), ensure_ascii=True) + "\n")


# ── Strategy 2: alarm-based chunking ─────────────────────────────────────────
#
# Splits the document at alarm-number heading boundaries, then recursively
# subdivides oversized chunks at alarm field boundaries (Explanation / Reaction
# / Remedy / Programm continuation), injecting the alarm header as a prefix in
# every continuation sub-chunk.  Non-alarm content falls back to blank-line
# splitting.  Parameters are character-based (not word-based).

_S2_ALARM_START_RE = re.compile(
    r"^(#{1,3}\s+\d{4,6}[\s%][^\n]*|\*\*\d{4,6}\*\*[^\n]*)",
    re.MULTILINE,
)
_S2_FIELD_RE = re.compile(
    r"(?=\n\*\*(?:Parameters|Explanation|Reaction|Remedy|Programm continuation):\*\*)",
    re.MULTILINE,
)


def _s2_split_by_alarm(text: str) -> list[str]:
    pattern = re.compile(
        r"(?=^#{1,3}\s+\d{4,6}[\s%]|^\*\*\d{4,6}\*\*)",
        re.MULTILINE,
    )
    return [c.strip() for c in pattern.split(text) if c.strip()]


def _s2_extract_alarm_header(chunk: str) -> str:
    m = _S2_ALARM_START_RE.search(chunk)
    return m.group(0).strip() if m else ""


def _s2_merge_to_size(pieces: list[str], max_size: int) -> list[str]:
    merged: list[str] = []
    buf = ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= max_size:
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged


def _s2_recursive_split(text: str, max_size: int, seps: list[str]) -> list[str]:
    if len(text) <= max_size or not seps:
        return [text]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    if len(parts) > 1:
        parts = [parts[0]] + [sep + p for p in parts[1:]]
    result: list[str] = []
    for part in parts:
        if len(part) <= max_size:
            result.append(part)
        else:
            result.extend(_s2_recursive_split(part, max_size, rest))
    return result


def _s2_subdivide_alarm_chunk(chunk: str, max_size: int, overlap: int) -> list[str]:
    header = _s2_extract_alarm_header(chunk)
    merged = _s2_merge_to_size(_S2_FIELD_RE.split(chunk), max_size)
    final_pieces: list[str] = []
    for piece in merged:
        if len(piece) <= max_size:
            final_pieces.append(piece)
        else:
            sub = _s2_recursive_split(piece, max_size, ["\n\n", "\n", " "])
            final_pieces.extend(_s2_merge_to_size(sub, max_size))
    if len(final_pieces) <= 1:
        return final_pieces or [chunk]
    result = [final_pieces[0]]
    n = len(final_pieces)
    for i in range(1, n):
        tail = final_pieces[i - 1][-overlap:].strip() if overlap > 0 else ""
        parts: list[str] = []
        if header:
            parts.append(f"{header} [cont. {i}/{n-1}]")
        if tail:
            parts.append(f"[...] {tail}")
        if parts:
            parts.append("")
        parts.append(final_pieces[i].strip())
        result.append("\n".join(parts))
    return result


def _s2_subdivide_plain_chunk(chunk: str, max_size: int) -> list[str]:
    pieces = _s2_recursive_split(chunk, max_size, ["\n\n", "\n", " "])
    return _s2_merge_to_size(pieces, max_size) or [chunk]


def chunk_strategy_2(
    markdown_path: str | Path,
    max_chars: int = 2000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """Alarm-based chunking with recursive field-boundary subdivision.

    Splits the document at alarm-number heading boundaries, subdivides chunks
    exceeding `max_chars` at semantic field boundaries (Explanation / Reaction
    / Remedy / Programm continuation), and injects the alarm header as a prefix
    into every continuation sub-chunk so that retrieval can always identify
    which alarm the text belongs to.

    Returns DocumentChunk objects with the same schema as chunk_markdown_by_words,
    so they are drop-in compatible with build_index / write_chunks_jsonl.
    """
    path = Path(markdown_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    source = path.name
    source_path_str = str(path)
    doc_lines = text.splitlines()
    total_lines = len(doc_lines)

    alarm_raw_chunks = _s2_split_by_alarm(text)

    # For each raw alarm chunk, find its start line (1-indexed) by matching
    # its first non-empty line against the original document, scanning forward.
    chunk_start_lines: list[int] = []
    search_from = 0
    for raw_chunk in alarm_raw_chunks:
        first_line = next((ln for ln in raw_chunk.splitlines() if ln.strip()), "")
        for i in range(search_from, total_lines):
            if doc_lines[i].strip() == first_line.strip():
                chunk_start_lines.append(i + 1)
                search_from = i + 1
                break
        else:
            chunk_start_lines.append(search_from + 1)

    chunks: list[DocumentChunk] = []

    for raw_idx, raw_chunk in enumerate(alarm_raw_chunks):
        start_line = chunk_start_lines[raw_idx]
        end_line = (
            chunk_start_lines[raw_idx + 1] - 1
            if raw_idx + 1 < len(chunk_start_lines)
            else total_lines
        )

        if len(raw_chunk) <= max_chars:
            sub_chunks = [raw_chunk]
        elif _s2_extract_alarm_header(raw_chunk):
            sub_chunks = _s2_subdivide_alarm_chunk(raw_chunk, max_chars, overlap)
        else:
            sub_chunks = _s2_subdivide_plain_chunk(raw_chunk, max_chars)

        for sub_text in sub_chunks:
            sub_text = sub_text.strip()
            if not sub_text:
                continue

            header = _s2_extract_alarm_header(sub_text)
            if header:
                clean = re.sub(r"^#{1,3}\s*", "", header).replace("**", "").strip()
                section = clean
                section_path = [clean]
            else:
                section = "Document preamble"
                section_path = []

            chunk_index = len(chunks) + 1
            chunks.append(
                DocumentChunk(
                    chunk_id=f"C{chunk_index:06d}",
                    source=source,
                    source_path=source_path_str,
                    chunk_index=chunk_index,
                    text=sub_text,
                    section=section,
                    section_path=section_path,
                    start_line=start_line,
                    end_line=end_line,
                    word_count=len(WORD_RE.findall(sub_text)),
                    content_sha256=_sha256(sub_text),
                )
            )

    return chunks
