from pathlib import Path

import pytest

from rag_system.chunking import chunk_markdown_by_words


def test_word_chunking_respects_markdown_sections(tmp_path: Path):
    markdown = tmp_path / "manual.md"
    markdown.write_text(
        "\n".join(
            [
                "# Alarm 10000 First section",
                "alpha beta gamma delta epsilon zeta eta theta",
                "# Alarm 20000 Second section",
                "iota kappa lambda mu nu xi omicron pi",
            ]
        ),
        encoding="utf-8",
    )

    chunks = chunk_markdown_by_words(markdown, chunk_words=4, chunk_overlap_words=1)
    sections = {chunk.section for chunk in chunks}

    assert "Alarm 10000 First section" in sections
    assert "Alarm 20000 Second section" in sections
    assert all("Alarm 20000" not in chunk.text for chunk in chunks if chunk.section == "Alarm 10000 First section")


def test_word_chunking_rejects_invalid_overlap(tmp_path: Path):
    markdown = tmp_path / "manual.md"
    markdown.write_text("# Alarm\none two three", encoding="utf-8")

    with pytest.raises(ValueError, match="menor que chunk_words"):
        chunk_markdown_by_words(markdown, chunk_words=4, chunk_overlap_words=4)

