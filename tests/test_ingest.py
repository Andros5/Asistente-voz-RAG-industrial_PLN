import json
from pathlib import Path

from rag_system.ingest import load_chunks_jsonl
from rag_system.models import DocumentChunk
from rag_system.weaviate_store import WeaviateRAGStore


def test_load_chunks_jsonl_roundtrip(tmp_path: Path):
    chunk = DocumentChunk(
        chunk_id="C000001",
        source="manual.md",
        source_path="/any/local/path/manual.md",
        chunk_index=1,
        text="Example chunk content",
        section="Example section",
        section_path=["Example section"],
        start_line=1,
        end_line=3,
        word_count=3,
        content_sha256="abc123",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps(chunk.to_weaviate_properties(), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    loaded = load_chunks_jsonl(chunks_path)

    assert loaded == [chunk]


def test_weaviate_uuid_does_not_depend_on_local_source_path():
    first = DocumentChunk(
        chunk_id="C000001",
        source="manual.md",
        source_path="/machine/a/manual.md",
        chunk_index=1,
        text="Same content",
        section="Same section",
        section_path=["Same section"],
        start_line=1,
        end_line=3,
        word_count=2,
        content_sha256="samehash",
    )
    second = DocumentChunk(
        chunk_id="C000001",
        source="manual.md",
        source_path="/machine/b/manual.md",
        chunk_index=1,
        text="Same content",
        section="Same section",
        section_path=["Same section"],
        start_line=1,
        end_line=3,
        word_count=2,
        content_sha256="samehash",
    )

    assert WeaviateRAGStore._uuid_for_chunk(first) == WeaviateRAGStore._uuid_for_chunk(second)

