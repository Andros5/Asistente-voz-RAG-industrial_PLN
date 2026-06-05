#!/usr/bin/env python3
"""
RAG Evaluation Dataset Generator — SINUMERIK 808D ADVANCED Diagnostics Manual.

Generates 100 (question, chunk_index) pairs for evaluating a hybrid RAG system.
Progress is saved after every successful pair so a crash does not lose work.

Setup (free — uses Google Gemini API, no billing required):
    1. Go to https://aistudio.google.com/apikey and create a free API key.
    2. Create a file named ".env" in the same folder as this script with:
           GOOGLE_API_KEY=AIza...
    3. python generate_rag_eval.py

Free tier limits (Gemini 2.0 Flash): 15 requests/min, 1500 requests/day.
The script adds a 4-second delay between calls to stay within the rate limit.
"""

import json
import random
import re
import os
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
# from rank_bm25 import BM25Okapi

load_dotenv()  # loads .env from the current directory

# ── Config ───────────────────────────────────────────────────────────────────
CHUNK_FILE    = "../data/manuals/808D_ADV_diagnostics_man_0718_en-US_chunked.md"
OUTPUT_FILE   = "../data/eval/rag_eval_dataset.json"
NUM_PAIRS     = 100
MAX_RETRIES   = 3     # question-generation attempts per chunk before skipping
TOP_K_PASS    = 3     # target chunk must appear in BM25 top-K to count as valid
SLEEP_BETWEEN = 4.0   # seconds between Gemini calls (15 RPM free limit → 4 s/call)
GEMINI_MODEL  = "gemini-2.0-flash"
RANDOM_SEED   = 42

# ── Prompt ───────────────────────────────────────────────────────────────────
PROMPT = """\
You are building an evaluation dataset for a RAG system on the SINUMERIK 808D \
ADVANCED Diagnostics Manual. An industrial operator speaks a question to an \
assistant in Spanish; the system normalises it to English and retrieves manual chunks.

Given the manual chunk below, write ONE question in English that:
1. An operator would naturally ask when troubleshooting this specific alarm or situation.
2. Requires ONLY the information in this chunk to be answered.
3. Does NOT quote alarm numbers (e.g. 10750, 380000) — ask about symptoms or actions.
4. Uses clear, direct technical language (no filler words, no greetings).
5. Is specific enough that a keyword search would surface this chunk first.

Chunk:
{chunk}

Reply with ONLY the question — no preamble, no quotes, no punctuation after the question mark.\
"""

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_chunks(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    parts = raw.split("\n<CHUNK>\n")
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def generate_question(chunk: str, model: genai.GenerativeModel) -> str:
    # Truncate to avoid huge prompts (median chunk ~565 chars, rarely needed)
    resp = model.generate_content(PROMPT.format(chunk=chunk[:2500]))
    return resp.text.strip().strip('"').strip("'")


def save(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

"""
def bm25_rank(bm25: BM25Okapi, query_tokens: list[str], target_idx: int, n: int) -> int:
    scores = bm25.get_scores(query_tokens)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return order.index(target_idx) + 1  # 1-based
"""


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: GOOGLE_API_KEY not found.\n"
            "Get a free key at https://aistudio.google.com/apikey\n"
            "Then add this line to a .env file in the same folder:\n"
            "  GOOGLE_API_KEY=AIza..."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    random.seed(RANDOM_SEED)

    print("Loading chunks …")
    chunks = load_chunks(CHUNK_FILE)
    n = len(chunks)
    print(f"  {n} chunks loaded  ({sum(len(c) for c in chunks)/1024:.0f} KB total)")

    # print("Building BM25 index …")
    # bm25 = BM25Okapi([tokenize(c) for c in chunks])
    # print("  Index ready")

    # ── Resume support ────────────────────────────────────────────────────────
    results: list[dict] = []
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            results = json.load(f)
        print(f"  Resuming from {len(results)} existing pairs")

    used_indices = {r["chunk_index"] for r in results}

    # Candidate pool: valid chunks not yet used, shuffled deterministically
    candidates = [
        i for i in range(n)
        if i not in used_indices
    ]
    random.shuffle(candidates)
    cand_ptr = 0

    # ── Generation loop ───────────────────────────────────────────────────────
    while len(results) < NUM_PAIRS:
        if cand_ptr >= len(candidates):
            print("\nRan out of candidate chunks — stopping early.")
            break

        idx = candidates[cand_ptr]
        cand_ptr += 1
        chunk = chunks[idx]
        pair_num = len(results) + 1

        print(f"\n[{pair_num:3d}/{NUM_PAIRS}] chunk {idx:4d}  ({len(chunk)} chars)")

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                question = generate_question(chunk, model)
                print(f"  Q: {question}")

                save(results, OUTPUT_FILE)
                success = True
                break

                """
                rank = bm25_rank(bm25, tokenize(question), idx, n)
                if rank <= TOP_K_PASS:
                    print(f"  BM25 rank: {rank} ✓ PASS")
                    results.append({
                        "id":          pair_num,
                        "question":    question,
                        "chunk_index": idx,
                        "bm25_rank":   rank,
                    })
                    save(results, OUTPUT_FILE)
                    success = True
                    break
                else:
                    print(f"  BM25 rank: {rank} ✗ FAIL  (attempt {attempt}/{MAX_RETRIES})")

                time.sleep(SLEEP_BETWEEN)
                """

            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                    wait = 30 * attempt
                    print(f"  Rate limit — waiting {wait}s …")
                    time.sleep(wait)
                else:
                    print(f"  Error: {exc}")
                    time.sleep(3)

        if not success:
            print(f"  → Skipping chunk {idx} after {MAX_RETRIES} attempts")
        else:
            time.sleep(SLEEP_BETWEEN)  # pace calls to stay within 15 RPM

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done. {len(results)} pairs saved to '{OUTPUT_FILE}'")
    """
    if results:
        rank1 = sum(1 for r in results if r["bm25_rank"] == 1)
        rank2 = sum(1 for r in results if r["bm25_rank"] == 2)
        rank3 = sum(1 for r in results if r["bm25_rank"] == 3)
        print(f"BM25 rank-1: {rank1}/{len(results)} ({rank1/len(results)*100:.0f}%)")
        print(f"BM25 rank-2: {rank2}")
        print(f"BM25 rank-3: {rank3}")
    """


if __name__ == "__main__":
    main()
