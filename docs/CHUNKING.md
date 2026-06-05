# Chunking

## Estrategia actual

La PoC usa chunking estatico por numero de palabras, respetando secciones Markdown.

Implementacion principal:

- `rag_system/chunking.py`
- Funcion publica: `chunk_markdown_by_words`
- Configuracion: `RAG_CHUNK_WORDS` y `RAG_CHUNK_OVERLAP_WORDS`

Valores base:

- `chunk_words = 220`
- `chunk_overlap_words = 40`

El flujo interno es:

1. Leer el Markdown.
2. Detectar encabezados `#`, `##`, `###`, etc.
3. Crear secciones mediante `_iter_sections`.
4. Aplicar ventanas de palabras dentro de cada seccion mediante `_word_windows`.
5. Devolver objetos `DocumentChunk`.

El objetivo de respetar secciones es reducir mezclas entre alarmas distintas.

## Donde ampliar

Para incorporar nuevas estrategias, el punto natural es `rag_system/chunking.py`.

Estrategias posibles:

- Chunking por seccion completa.
- Chunking por tokens del modelo de embeddings.
- Chunking por alarma detectada.
- Chunking semantico por similitud entre parrafos.
- Chunking jerarquico: seccion completa para BM25 y subchunks para vectorial.

Despues, `rag_system/ingest.py` deberia elegir la estrategia segun configuracion, por ejemplo `RAG_CHUNKING_STRATEGY`.

## Archivos relacionados

- `rag_system/models.py`: estructura `DocumentChunk`.
- `rag_system/ingest.py`: llama al chunking antes de generar embeddings.
- `scripts/build_index.py`: expone parametros de chunking por CLI.
- `scripts/export_chunks.py`: exporta chunks a JSONL para inspeccion y evaluacion.
- `tests/test_chunking.py`: tests unitarios de la estrategia actual.

