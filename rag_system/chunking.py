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
