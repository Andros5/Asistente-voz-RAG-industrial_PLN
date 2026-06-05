from __future__ import annotations

import argparse
import os

from llm_system import DEFAULT_LLM_MODEL_ID

DEFAULT_EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def snapshot(model_id: str, cache_dir: str | None) -> None:
    from huggingface_hub import snapshot_download

    print(f"[HF] Descargando o verificando cache de: {model_id}", flush=True)
    path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    print(f"[HF] Disponible en cache: {path}", flush=True)


def main() -> None:
    _load_dotenv_if_available()
    parser = argparse.ArgumentParser(description="Pre-descarga modelos de la PoC en la cache local de Hugging Face.")
    parser.add_argument("--target", choices=["llm", "rag", "all"], default="all")
    parser.add_argument("--llm-model-id", default=os.getenv("LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID))
    parser.add_argument("--embedding-model-id", default=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL_ID))
    parser.add_argument("--cache-dir", default=os.getenv("HF_HOME") or None)
    args = parser.parse_args()

    if args.target in {"llm", "all"}:
        snapshot(args.llm_model_id, args.cache_dir)
    if args.target in {"rag", "all"}:
        snapshot(args.embedding_model_id, args.cache_dir)


if __name__ == "__main__":
    main()
