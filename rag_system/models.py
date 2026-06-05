from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    source_path: str
    chunk_index: int
    text: str
    section: str
    section_path: list[str]
    start_line: int
    end_line: int
    word_count: int
    content_sha256: str

    def to_weaviate_properties(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "content": self.text,
            "section": self.section,
            "section_path": self.section_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "word_count": self.word_count,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_weaviate_properties(cls, properties: dict[str, Any]) -> "DocumentChunk":
        section_path = properties.get("section_path") or []
        if isinstance(section_path, str):
            section_path = [section_path]

        return cls(
            chunk_id=str(properties["chunk_id"]),
            source=str(properties.get("source", "")),
            source_path=str(properties.get("source_path", "")),
            chunk_index=int(properties.get("chunk_index", 0)),
            text=str(properties.get("content", "")),
            section=str(properties.get("section", "")),
            section_path=list(section_path),
            start_line=int(properties.get("start_line", 0)),
            end_line=int(properties.get("end_line", 0)),
            word_count=int(properties.get("word_count", 0)),
            content_sha256=str(properties.get("content_sha256", "")),
        )


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    rank: int
    mode: str
    score: float | None = None
    bm25_score: float | None = None
    vector_distance: float | None = None
    rrf_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id
