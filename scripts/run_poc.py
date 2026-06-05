from __future__ import annotations

import argparse
import os
import sys

from llm_system import DEFAULT_LLM_MODEL_ID
from poc_system import run_poc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main() -> None:
    _load_dotenv_if_available()
    parser = argparse.ArgumentParser(description="Ejecuta la PoC completa T2a -> T2b -> T3 -> T4a -> T4b.")
    parser.add_argument("--query", required=True, help="Consulta original en espanol.")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default=os.getenv("RAG_MODE", "hybrid"))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAG_TOP_K", "5")))
    parser.add_argument("--model-id", default=os.getenv("LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LLM_MAX_NEW_TOKENS", "512")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0.0")))
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default=os.getenv("LLM_DTYPE", "auto"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--stream-final", action="store_true", help="Emite la respuesta final con streaming.")
    args = parser.parse_args()

    run_poc(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        stream_final=args.stream_final,
        print_steps=True,
    )


if __name__ == "__main__":
    main()
