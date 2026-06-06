#!/usr/bin/env python3
"""
RAG Evaluation Dataset Generator — SINUMERIK 808D ADVANCED Diagnostics Manual.

Generates (query, relevant_chunk_ids, notes, difficulty) tuples at three levels:
  easy   (50 pairs): high lexical overlap between question and chunk
  medium (30 pairs): moderate overlap, some paraphrasing
  hard   (20 pairs): low overlap, semantic understanding required

Progress is appended after each successful pair so a crash does not lose work.

Setup (free — uses Groq API):
    1. Create a free account at console.groq.com and generate an API key.
    2. Add to .env:  GROQ_API_KEY=gsk_...
    3. python -m scripts.generate_rag_eval
"""

import json
import random
import os
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_FILE    = "data/processed/chunks_strategy2_2500c_150o.jsonl"
OUTPUT_FILE   = "data/eval/eval_queries.jsonl"
SLEEP_BETWEEN = 5.0   # seconds between calls; Groq free tier allows ~30 RPM
GROQ_MODEL    = "llama-3.3-70b-versatile"
RANDOM_SEED   = 42
MAX_RETRIES   = 3

DIFFICULTY_TARGETS = {"easy": 50, "medium": 30, "hard": 20}

# ── Prompts ───────────────────────────────────────────────────────────────────
_DIFFICULTY_INSTRUCTIONS = {
    "easy": (
        "EASY — use many of the same technical keywords and phrases present in the chunk. "
        "High lexical overlap is expected: the question should read almost like a direct "
        "reference to the chunk text."
    ),
    "medium": (
        "MEDIUM — paraphrase the main topic using related but different wording. "
        "Avoid copying exact multi-word phrases; the technical subject must still match clearly."
    ),
    "hard": (
        "HARD — describe the operator's observable symptom or high-level goal without using the "
        "specific technical identifiers from the chunk (alarm numbers, parameter names, section "
        "headings). Semantic understanding is required to link this question to the chunk."
    ),
}

_PROMPT_TEMPLATE = """\
You are building an evaluation dataset for a RAG system on the SINUMERIK 808D ADVANCED Diagnostics Manual.
An industrial operator speaks a question in Spanish; the system normalises it to English and retrieves manual chunks.

Given the manual chunk below, generate:
1. ONE question in English that an operator would naturally ask when troubleshooting, \
following the difficulty instruction.
2. A short note (3–8 words) summarising what the chunk is about \
(e.g. "Alarm 26120 remedy", "Axis speed limit configuration").

The question must:
- Be answerable ONLY from the information in this chunk.
- Be specific enough that this chunk would be the top retrieval result.
- Use direct technical language (no filler words, no greetings).

Difficulty instruction: {difficulty_instruction}

Chunk:
{chunk}

Reply ONLY with a valid JSON object — no markdown fences, no extra text:
{{"question": "<question here>", "note": "<note here>"}}"""


def _make_prompt(difficulty: str, chunk_content: str) -> str:
    return _PROMPT_TEMPLATE.format(
        difficulty_instruction=_DIFFICULTY_INSTRUCTIONS[difficulty],
        chunk=chunk_content[:2500],
    )


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_existing_results(path: str) -> list[dict]:
    results = []
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def append_result(result: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ── LLM call ──────────────────────────────────────────────────────────────────
def generate_pair(chunk_content: str, difficulty: str, client: Groq) -> tuple[str, str]:
    """Call Groq and return (question, note) for the given chunk and difficulty."""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": _make_prompt(difficulty, chunk_content)}],
        temperature=0.7,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences the model sometimes wraps around JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())
    return str(data["question"]).strip(), str(data["note"]).strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: GROQ_API_KEY not found.\n"
            "Create a free key at console.groq.com\n"
            "Then add to .env:  GROQ_API_KEY=gsk_..."
        )

    client = Groq(api_key=api_key)

    print("Loading chunks …")
    chunks = load_chunks(CHUNK_FILE)
    print(f"  {len(chunks)} chunks loaded  ({sum(len(c['content']) for c in chunks) / 1024:.0f} KB total)")

    # ── Resume support ────────────────────────────────────────────────────────
    existing    = load_existing_results(OUTPUT_FILE)
    used_ids    = {r["relevant_chunk_ids"][0] for r in existing}
    counts_done = {d: sum(1 for r in existing if r["difficulty"] == d) for d in DIFFICULTY_TARGETS}
    if existing:
        print(f"  Resuming: {len(existing)} pairs already saved "
              f"(easy={counts_done['easy']}, medium={counts_done['medium']}, hard={counts_done['hard']})")

    # ── Build remaining difficulty sequence (interleaved) ─────────────────────
    random.seed(RANDOM_SEED)
    remaining_difficulties: list[str] = []
    for d in ("easy", "medium", "hard"):
        need = DIFFICULTY_TARGETS[d] - counts_done[d]
        if need > 0:
            remaining_difficulties.extend([d] * need)
    random.shuffle(remaining_difficulties)

    # ── Candidate chunks (unused, shuffled) ───────────────────────────────────
    candidates = [c for c in chunks if c["chunk_id"] not in used_ids]
    random.shuffle(candidates)
    cand_ptr = 0

    total_target = sum(DIFFICULTY_TARGETS.values())

    # ── Generation loop ───────────────────────────────────────────────────────
    for pair_num, difficulty in enumerate(remaining_difficulties, start=len(existing) + 1):
        if cand_ptr >= len(candidates):
            print("\nRan out of candidate chunks — stopping early.")
            break

        chunk = candidates[cand_ptr]
        cand_ptr += 1
        print(f"\n[{pair_num:3d}/{total_target}] {chunk['chunk_id']}  "
              f"({len(chunk['content'])} chars)  [{difficulty}]")

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                question, note = generate_pair(chunk["content"], difficulty, client)
                print(f"  Q: {question}")
                print(f"  N: {note}")
                append_result(
                    {
                        "query":              question,
                        "relevant_chunk_ids": [chunk["chunk_id"]],
                        "notes":              note,
                        "difficulty":         difficulty,
                    },
                    OUTPUT_FILE,
                )
                success = True
                break
            except json.JSONDecodeError as exc:
                print(f"  JSON parse error (attempt {attempt}/{MAX_RETRIES}): {exc}")
                time.sleep(3)
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
            print(f"  → Skipping {chunk['chunk_id']} after {MAX_RETRIES} attempts")
        else:
            time.sleep(SLEEP_BETWEEN)

    # ── Summary ───────────────────────────────────────────────────────────────
    final        = load_existing_results(OUTPUT_FILE)
    final_counts = {d: sum(1 for r in final if r["difficulty"] == d) for d in DIFFICULTY_TARGETS}
    print(f"\n{'='*60}")
    print(f"Done. {len(final)} pairs saved to '{OUTPUT_FILE}'")
    print(f"  easy={final_counts['easy']}  medium={final_counts['medium']}  hard={final_counts['hard']}")


if __name__ == "__main__":
    main()
