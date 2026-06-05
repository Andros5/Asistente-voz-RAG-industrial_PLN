from __future__ import annotations

import os

from llm_system import DEFAULT_LLM_MODEL_ID
from poc_system import PoCRunner


def main() -> None:
    runner = PoCRunner(
        mode=os.getenv("RAG_MODE", "hybrid"),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        model_id=os.getenv("LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID),
        dtype=os.getenv("LLM_DTYPE", "auto"),
    )

    print("(Escribe 'exit' o 'quit' para salir)")
    while True:
        consulta_original_es = input("\n> ")
        if consulta_original_es.lower() in ["exit", "quit"]:
            break

        runner.run(
            consulta_original_es,
            stream_final=True,
            print_steps=True,
        )
        print("\n" + "-" * 40)


if __name__ == "__main__":
    main()
