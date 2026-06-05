from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path


async def extract_markdown(pdf_path: Path, output_path: Path, force: bool) -> None:
    try:
        from llama_cloud import AsyncLlamaCloud
    except ImportError as exc:
        raise RuntimeError("Instala las dependencias de parseo con: pip install -r requirements-parse.txt") from exc

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise RuntimeError("Falta LLAMA_CLOUD_API_KEY en el entorno o en .env")

    if output_path.exists() and not force:
        print(f"[Aviso] El archivo '{output_path}' ya existe. Usa --force para regenerarlo.")
        return
    if not pdf_path.exists():
        raise FileNotFoundError(f"No se encuentra el PDF: {pdf_path}")

    os.environ["LLAMA_CLOUD_API_KEY"] = api_key
    client = AsyncLlamaCloud()

    print(f"[LlamaCloud] Subiendo documento: {pdf_path}", flush=True)
    uploaded = await client.files.create(file=str(pdf_path), purpose="parse")

    print(f"[LlamaCloud] Procesando estructura visual (file_id={uploaded.id})...", flush=True)
    result = await client.parsing.parse(
        file_id=uploaded.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],
    )

    if not getattr(result, "markdown", None) or not result.markdown.pages:
        raise RuntimeError("LlamaCloud no devolvio paginas Markdown")

    full_text = "\n\n".join(page.markdown for page in result.markdown.pages)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_text, encoding="utf-8")
    print(f"[Exito] Markdown generado: {output_path} ({len(result.markdown.pages)} paginas)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parsea un PDF tecnico con LlamaCloud y genera Markdown.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output or args.pdf.with_suffix(".md")
    asyncio.run(extract_markdown(args.pdf.expanduser().resolve(), output.expanduser().resolve(), args.force))


if __name__ == "__main__":
    main()
