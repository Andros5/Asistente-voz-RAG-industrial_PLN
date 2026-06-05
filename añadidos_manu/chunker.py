"""
chunker.py  —  Final chunking pipeline for the SINUMERIK 808D Diagnostics Manual.

Pipeline:
  1. Split by alarm entry (alarm-based, Strategy 1 from the analysis).
  2. For chunks > MAX_CHARS, subdivide recursively.
       - If the chunk is an alarm entry: split at alarm field boundaries
         (Explanation / Reaction / Remedy / Programm continuation) and
         inject the alarm header into every continuation sub-chunk so that
         retrieval can always identify which alarm the text belongs to.
       - If the chunk has no alarm structure (preface, chapter intros, etc.):
         split at blank lines recursively.
  3. Print percentage of alarm chunks that exceeded MAX_CHARS.
  4. Show a frequency histogram of the final chunk lengths.
  5. Save all chunks to <output_file> using "<CHUNK>\\n" as delimiter.
"""

import re
import statistics
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────

DOC_PATH    = "../data/manuals/808D_ADV_diagnostics_man_0718_en-US.md"
OUT_PATH    = "../data/manuals/808D_ADV_diagnostics_man_0718_en-US_chunked.md"
PLOT_PATH   = "img/chunk_final_analysis.png"

MAX_CHARS   = 2000   # chunks above this will be subdivided
OVERLAP     = 150    # character overlap between sub-chunks of a split alarm
CHUNK_SEP   = "\n<CHUNK>\n"

# ── regex patterns ────────────────────────────────────────────────────────────

# Alarm entry start: "# 8065 description", "## 25000 ...", "**400000** **...**"
ALARM_START_RE = re.compile(
    r'^(#{1,3}\s+\d{4,6}[\s%][^\n]*|\*\*\d{4,6}\*\*[^\n]*)',
    re.MULTILINE
)

# Alarm field names as they appear in the document
FIELD_RE = re.compile(
    r'(?=\n\*\*(?:Parameters|Explanation|Reaction|Remedy|Programm continuation):\*\*)',
    re.MULTILINE
)


# ── Step 1: alarm-based split ─────────────────────────────────────────────────

def split_by_alarm(text: str) -> list[str]:
    """
    Split the full document at the start of each alarm entry.
    The alarm heading is kept as part of its chunk (lookahead split).
    """
    pattern = re.compile(
        r'(?=^#{1,3}\s+\d{4,6}[\s%]|^\*\*\d{4,6}\*\*)',
        re.MULTILINE
    )
    raw = pattern.split(text)
    return [c.strip() for c in raw if c.strip()]


# ── Step 2a: subdivide alarm chunks ──────────────────────────────────────────

def extract_alarm_header(chunk: str) -> str:
    """Return the first line that identifies the alarm (number + name)."""
    m = ALARM_START_RE.search(chunk)
    return m.group(0).strip() if m else ""


def _merge_to_size(pieces: list[str], max_size: int) -> list[str]:
    """Greedily merge consecutive pieces until they would exceed max_size."""
    merged, buf = [], ""
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


def _recursive_split(text: str, max_size: int, seps: list[str]) -> list[str]:
    """
    Recursive character splitter: cascades through `seps` until every piece
    fits within max_size.
    """
    if len(text) <= max_size or not seps:
        return [text]

    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    # Re-attach separator so context is preserved
    parts = [parts[0]] + [sep + p for p in parts[1:]] if len(parts) > 1 else parts

    result = []
    for part in parts:
        if len(part) <= max_size:
            result.append(part)
        else:
            result.extend(_recursive_split(part, max_size, rest))
    return result


def subdivide_alarm_chunk(chunk: str, max_size: int, overlap: int) -> list[str]:
    """
    Subdivide a large alarm chunk while preserving semantic context.

    Strategy:
      1. Try splitting at alarm field boundaries (Explanation, Reaction, …).
      2. If any field block is still too large, fall back to blank-line / line splits.
      3. Inject the alarm header as a prefix in every continuation sub-chunk.
      4. Add character-level overlap between consecutive sub-chunks.
    """
    header = extract_alarm_header(chunk)

    # -- split at field boundaries
    field_parts = FIELD_RE.split(chunk)
    merged = _merge_to_size(field_parts, max_size)

    # -- if any merged piece is still too large, recurse on it
    final_pieces = []
    for piece in merged:
        if len(piece) <= max_size:
            final_pieces.append(piece)
        else:
            sub = _recursive_split(piece, max_size, ["\n\n", "\n", " "])
            final_pieces.extend(_merge_to_size(sub, max_size))

    if len(final_pieces) <= 1:
        return final_pieces or [chunk]

    # -- inject header + overlap into continuation sub-chunks
    result = [final_pieces[0]]
    n = len(final_pieces)
    for i in range(1, n):
        prev   = final_pieces[i - 1]
        curr   = final_pieces[i]

        # Overlap: last `overlap` chars of the previous sub-chunk
        tail = prev[-overlap:].strip() if overlap > 0 else ""

        # Build continuation prefix
        parts = []
        if header:
            parts.append(f"{header} [cont. {i}/{n-1}]")
        if tail:
            parts.append(f"[...] {tail}")
        if parts:
            parts.append("")   # blank line before content
        parts.append(curr.strip())

        result.append("\n".join(parts))

    return result


# ── Step 2b: subdivide non-alarm chunks ───────────────────────────────────────

def subdivide_plain_chunk(chunk: str, max_size: int) -> list[str]:
    """
    Generic recursive subdivision for chunks without alarm structure
    (preface, chapter introductions, etc.).
    """
    pieces = _recursive_split(chunk, max_size, ["\n\n", "\n", " "])
    return _merge_to_size(pieces, max_size) or [chunk]


# ── main pipeline ─────────────────────────────────────────────────────────────

def chunk_document(text: str, max_size: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    # Step 1
    alarm_chunks = split_by_alarm(text)

    large_count = sum(1 for c in alarm_chunks if len(c) > max_size)
    pct = large_count / len(alarm_chunks) * 100
    print(f"Step 1 – alarm-based split")
    print(f"  Total alarm chunks : {len(alarm_chunks):,}")
    print(f"  Chunks > {max_size} chars : {large_count:,}  ({pct:.1f}%)")

    # Step 2
    final: list[str] = []
    for ch in alarm_chunks:
        if len(ch) <= max_size:
            final.append(ch)
        elif extract_alarm_header(ch):
            final.extend(subdivide_alarm_chunk(ch, max_size, overlap))
        else:
            final.extend(subdivide_plain_chunk(ch, max_size))

    after_large = sum(1 for c in final if len(c) > max_size)
    print(f"\nStep 2 – recursive subdivision")
    print(f"  Total final chunks   : {len(final):,}")
    print(f"  Still > {max_size} chars : {after_large:,}  (these are single unsplittable lines)")
    return final


# ── statistics + plot ─────────────────────────────────────────────────────────

def print_stats(lengths: list[int]) -> None:
    print(f"\n{'='*55}")
    print(f"  Final chunk statistics")
    print(f"{'='*55}")
    print(f"  Count  : {len(lengths):,}")
    print(f"  Mean   : {statistics.mean(lengths):,.0f} chars")
    print(f"  Median : {statistics.median(lengths):,.0f} chars")
    print(f"  Min    : {min(lengths):,} chars")
    print(f"  Max    : {max(lengths):,} chars")
    print(f"  P5     : {np.percentile(lengths, 5):,.0f} chars")
    print(f"  P95    : {np.percentile(lengths, 95):,.0f} chars")


def plot_lengths(lengths: list[int], output_path: str = PLOT_PATH) -> None:
    clip    = np.percentile(lengths, 99)
    clipped = [l for l in lengths if l <= clip]
    hidden  = len(lengths) - len(clipped)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.hist(clipped, bins=70, color="#4C72B0", edgecolor="white", linewidth=0.4)

    mean_v   = statistics.mean(lengths)
    median_v = statistics.median(lengths)
    p5_v     = np.percentile(lengths, 5)
    p95_v    = np.percentile(lengths, 95)

    ax.axvline(mean_v,   color="black",     ls="--", lw=1.5, label=f"Mean     {mean_v:,.0f}")
    ax.axvline(median_v, color="gray",      ls=":",  lw=1.5, label=f"Median   {median_v:,.0f}")
    ax.axvline(p5_v,     color="navy",      ls="-.", lw=1.2, label=f"P5       {p5_v:,.0f}")
    ax.axvline(p95_v,    color="firebrick", ls="-.", lw=1.2, label=f"P95      {p95_v:,.0f}")
    ax.axvline(MAX_CHARS, color="orange",   ls="-",  lw=1.5, label=f"Max threshold  {MAX_CHARS}")

    note = f"(x-axis clipped at P99={clip:,.0f};  {hidden} outliers not shown)"
    ax.set_xlabel(f"Chunk length (characters)   {note}")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"SINUMERIK 808D — Final chunking: alarm-based + recursive subdivision\n"
        f"n = {len(lengths):,} chunks  |  MAX_CHARS = {MAX_CHARS}  |  OVERLAP = {OVERLAP}",
        fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.text(0.98, 0.78,
            f"Max = {max(lengths):,}\nMin = {min(lengths):,}",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved -> {output_path}")
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Loading {DOC_PATH} ...")
    text = Path(DOC_PATH).read_text(encoding="utf-8")
    print(f"Document: {len(text):,} chars  ({len(text.splitlines()):,} lines)\n")

    chunks  = chunk_document(text)
    lengths = [len(c) for c in chunks]
    print_stats(lengths)

    # Save
    output = CHUNK_SEP.join(chunks)
    Path(OUT_PATH).write_text(output, encoding="utf-8")
    size_mb = Path(OUT_PATH).stat().st_size / 1_048_576
    print(f"\nSaved -> {OUT_PATH}  ({len(chunks):,} chunks, {size_mb:.1f} MB)")

    plot_lengths(lengths)


if __name__ == "__main__":
    main()
