"""
Chunking strategies for the SINUMERIK 808D ADVANCED Diagnostics Manual.

The document is a ~500-page diagnostics manual converted from PDF to Markdown.
It contains alarm entries, each with structured fields:
  Parameters / Explanation / Reaction / Remedy / Programm continuation

Four strategies are compared:
  1. Alarm-based      – one chunk per alarm entry (primary semantic unit)
  2. Header-based     – split on Markdown headings (# / ##)
  3. Separator-based  – split on horizontal rules (--- / ***)
  4. Recursive        – langchain-style recursive character splitting
"""

import re
import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────

DOC_PATH    = "../data/manuals/808D_ADV_diagnostics_man_0718_en-US.md"
PLOT_PATH   = "img/chunk_final_analysis.png"


# ── helpers ─────────────────────────────────────────────────────────────────

def load_document(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def analyze(chunks: list[str], name: str) -> dict:
    """Print stats and return a dict with lengths list."""
    lengths = [len(c) for c in chunks]
    p5  = np.percentile(lengths, 5)
    p95 = np.percentile(lengths, 95)
    print(f"\n{'-'*60}")
    print(f"  Strategy : {name}")
    print(f"  Chunks   : {len(chunks):,}")
    print(f"  Mean     : {statistics.mean(lengths):,.0f} chars")
    print(f"  Median   : {statistics.median(lengths):,.0f} chars")
    print(f"  Min      : {min(lengths):,} chars")
    print(f"  Max      : {max(lengths):,} chars")
    print(f"  P5       : {p5:,.0f} chars")
    print(f"  P95      : {p95:,.0f} chars")
    return {"name": name, "lengths": lengths}


# ── strategy 1: alarm-based ──────────────────────────────────────────────────

def strategy_alarm_based(text: str) -> list[str]:
    """
    Each NC/drive alarm is its own chunk.

    Alarm entries appear in three formats:
      # 8065 Some description          <- h1 heading + alarm number
      ## 25000 Some description        <- h2 heading + alarm number
      **400000** **Some description**  <- bold number (PLC alarms section)

    We split at every point where one of these patterns starts.
    """
    # Lookahead so the delimiter itself is kept as part of the chunk
    pattern = re.compile(
        r'(?=^#{1,3}\s+\d{4,6}[\s%]|^\*\*\d{4,6}\*\*)',
        re.MULTILINE
    )
    raw = pattern.split(text)
    return [c.strip() for c in raw if c.strip()]


# ── strategy 2: header-based ─────────────────────────────────────────────────

def strategy_header_based(text: str, max_level: int = 2) -> list[str]:
    """
    Split on Markdown headings up to `max_level`.

    max_level=1 → split only on #  (chapters + individual alarms titled with #)
    max_level=2 → split on # and ## (more granular, catches ## alarm entries)

    Large chunks that exceed MAX_CHUNK_CHARS are further split on blank lines
    to avoid enormous chapter-introduction blobs.
    """
    MAX_CHUNK_CHARS = 4000

    if max_level == 1:
        pattern = re.compile(r'(?=^#\s)', re.MULTILINE)
    else:
        pattern = re.compile(r'(?=^#{1,2}\s)', re.MULTILINE)

    raw = pattern.split(text)
    chunks = []
    for c in raw:
        c = c.strip()
        if not c:
            continue
        if len(c) <= MAX_CHUNK_CHARS:
            chunks.append(c)
        else:
            # Split oversized chunks on blank lines
            sub = [s.strip() for s in re.split(r'\n{2,}', c) if s.strip()]
            # Re-merge tiny sub-chunks (< 200 chars) with their neighbour
            merged, buf = [], ""
            for s in sub:
                if len(buf) + len(s) < MAX_CHUNK_CHARS:
                    buf = (buf + "\n\n" + s).strip()
                else:
                    if buf:
                        merged.append(buf)
                    buf = s
            if buf:
                merged.append(buf)
            chunks.extend(merged)
    return chunks


# ── strategy 3: separator-based ──────────────────────────────────────────────

def strategy_separator_based(text: str) -> list[str]:
    """
    Split on horizontal rules (--- and ***) which the PDF parser inserted
    between alarm entries in some sections.

    This complements strategy 1: some alarm blocks are separated by --- rather
    than by a heading, so this catches them.
    """
    pattern = re.compile(r'\n(?:---|\*{3})\n', re.MULTILINE)
    raw = pattern.split(text)
    return [c.strip() for c in raw if c.strip()]


# ── strategy 4: recursive character splitter ─────────────────────────────────

def strategy_recursive(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 150,
) -> list[str]:
    """
    Langchain-style recursive character splitter.

    Tries to split on separators in priority order, recursing on pieces that
    are still too large.  Overlap is added by repeating the tail of the
    previous chunk at the start of the next.
    """
    separators = [
        "\n# ",       # h1 heading
        "\n## ",      # h2 heading
        "\n---\n",    # horizontal rule
        "\n***\n",    # horizontal rule
        "\n\n",       # blank line / paragraph
        "\n",         # newline
        " ",          # space
        "",           # character fallback
    ]

    def _split(txt: str, seps: list[str]) -> list[str]:
        if not seps:
            # Final fallback: hard split by character count
            return [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size - overlap)]

        sep = seps[0]
        rest = seps[1:]

        if sep == "":
            parts = list(txt)
        else:
            parts = txt.split(sep)
            # Re-attach the separator to the beginning of each subsequent part
            parts = [parts[0]] + [sep + p for p in parts[1:]]

        chunks = []
        for part in parts:
            if len(part) <= chunk_size:
                chunks.append(part)
            else:
                chunks.extend(_split(part, rest))
        return chunks

    raw_chunks = _split(text, separators)

    # Merge tiny chunks with the next one, then add overlap
    merged: list[str] = []
    buf = ""
    for piece in raw_chunks:
        piece = piece.strip()
        if not piece:
            continue
        if len(buf) + len(piece) + 1 <= chunk_size:
            buf = (buf + "\n" + piece).strip() if buf else piece
        else:
            if buf:
                merged.append(buf)
            buf = piece
    if buf:
        merged.append(buf)

    # Add overlap: prepend tail of previous chunk to current chunk
    final: list[str] = []
    for i, chunk in enumerate(merged):
        if i == 0 or overlap == 0:
            final.append(chunk)
        else:
            tail = merged[i - 1][-overlap:]
            final.append((tail + "\n" + chunk).strip())

    return final


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_all(results: list[dict], output_path: str = "img/chunk_analysis.png") -> None:
    """
    One subplot per strategy: histogram of chunk lengths.
    X-axis is clipped at P99 to avoid a few outliers compressing the view.
    """
    n = len(results)
    fig = plt.figure(figsize=(16, 4 * n))
    gs  = gridspec.GridSpec(n, 1, hspace=0.55)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, res in enumerate(results):
        ax  = fig.add_subplot(gs[i])
        lengths = res["lengths"]
        clip = np.percentile(lengths, 99)
        clipped = [l for l in lengths if l <= clip]

        ax.hist(clipped, bins=60, color=colors[i % len(colors)], edgecolor="white", linewidth=0.4)
        ax.axvline(statistics.mean(lengths),   color="black",  ls="--", lw=1.2, label=f"Mean  {statistics.mean(lengths):,.0f}")
        ax.axvline(statistics.median(lengths), color="gray",   ls=":",  lw=1.2, label=f"Median {statistics.median(lengths):,.0f}")
        ax.axvline(np.percentile(lengths, 5),  color="navy",   ls="-.", lw=1.0, label=f"P5    {np.percentile(lengths,5):,.0f}")
        ax.axvline(np.percentile(lengths, 95), color="firebrick", ls="-.", lw=1.0, label=f"P95   {np.percentile(lengths,95):,.0f}")

        ax.set_title(f"Strategy: {res['name']}  (n={len(lengths):,} chunks)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Chunk length (characters)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9, loc="upper right")
        ax.text(0.98, 0.80,
                f"Max = {max(lengths):,}\nMin = {min(lengths):,}",
                transform=ax.transAxes, ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        note = f"(x-axis clipped at P99={clip:,.0f}; {len(lengths)-len(clipped)} outliers hidden)"
        ax.set_xlabel(f"Chunk length (characters)  {note}")

    fig.suptitle("SINUMERIK 808D Diagnostics Manual — Chunking Strategy Comparison",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {output_path}")
    plt.show()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    text = load_document(DOC_PATH)
    print(f"Document size: {len(text):,} characters  ({len(text.splitlines()):,} lines)")

    strategies = [
        ("1 - Alarm-based",        strategy_alarm_based(text)),
        ("2 - Header-based (h1+h2)", strategy_header_based(text, max_level=2)),
        ("3 - Separator-based",    strategy_separator_based(text)),
        ("4 - Recursive (1500 ch)", strategy_recursive(text, chunk_size=1500, overlap=150)),
    ]

    results = []
    for name, chunks in strategies:
        results.append(analyze(chunks, name))

    print("\n")
    plot_all(results, output_path=PLOT_PATH)

    # Optional: dump sample chunks to inspect quality
    print("\n" + "="*60)
    print("SAMPLE CHUNKS (first 2 from each strategy)")
    print("="*60)
    for name, chunks in strategies:
        print(f"\n=== {name} ===")
        for j, ch in enumerate(chunks[:2]):
            preview = ch[:400].replace("\n", " | ")
            preview = preview.encode("ascii", "replace").decode("ascii")
            print(f"  [{j}] ({len(ch)} chars): {preview}...")


if __name__ == "__main__":
    main()
