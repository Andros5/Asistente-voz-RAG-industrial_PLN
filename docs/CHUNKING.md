# Chunking

## Estrategia actual

La PoC usa `chunk_strategy_2`: chunking por alarmas, con subdivision por campos tecnicos y limite de caracteres.

Implementacion principal:

- `rag_system/chunking.py`
- Funcion publica: `chunk_strategy_2`
- Configuracion: `RAG_CHUNK_MAX_CHARS` y `RAG_CHUNK_OVERLAP_CHARS`

Valores base:

- `chunk_max_chars = 2500`
- `chunk_overlap_chars = 150`

El flujo interno es:

1. Leer el Markdown.
2. Detectar limites de alarma mediante encabezados con numero de alarma.
3. Separar cada alarma como unidad documental.
4. Si una alarma supera el limite, subdividir por campos como `Parameters`, `Explanation`, `Reaction`, `Remedy` y `Programm continuation`.
5. Inyectar el encabezado de la alarma en subchunks de continuacion.
6. Devolver objetos `DocumentChunk`.

El objetivo es evitar mezclar alarmas distintas y conservar el numero de alarma en cada subchunk recuperable.

## Donde ampliar

Para incorporar nuevas estrategias, el punto natural es `rag_system/chunking.py`.

Estrategias posibles:

- Chunking por seccion completa.
- Chunking por tokens del modelo de embeddings.
- Chunking por alarma detectada.
- Chunking semantico por similitud entre parrafos.
- Chunking jerarquico: seccion completa para BM25 y subchunks para vectorial.

Actualmente `scripts.export_chunks.py` genera esta estrategia y `scripts.build_index.py` indexa por defecto desde el JSONL resultante. Si se anade otra estrategia, debe alinearse tambien con `data/eval/eval_queries.jsonl`.

## Exportacion e indexacion

Hay dos pasos separados:

```text
Markdown -> scripts.export_chunks -> data/processed/chunks_strategy2_2500c_150o.jsonl
JSONL    -> scripts.build_index   -> Weaviate
```

`scripts.export_chunks` crea el JSONL para inspeccion, evaluacion y generacion de queries. No inserta en Weaviate.

`scripts.build_index` lee el JSONL versionado, calcula embeddings BGE e inserta chunks y vectores en Weaviate. Por defecto no regenera chunks desde Markdown.

Para desarrollo existe `scripts.build_index --from-markdown`, que regenera chunks desde el Markdown antes de indexar. No es el flujo recomendado para desplegar la PoC porque puede producir divergencias si no se actualiza tambien el JSONL.

Para que la evaluacion sea valida, `data/processed/chunks_strategy2_2500c_150o.jsonl`, `data/eval/eval_queries.jsonl` y la coleccion de Weaviate deben estar alineados en `chunk_id` y contenido.

## Archivos relacionados

- `rag_system/models.py`: estructura `DocumentChunk`.
- `rag_system/ingest.py`: carga chunks desde JSONL o Markdown y genera embeddings.
- `scripts/build_index.py`: indexa desde JSONL por defecto.
- `scripts/export_chunks.py`: exporta chunks a JSONL para inspeccion y evaluacion.
- `tests/test_chunking.py`: tests unitarios de la estrategia actual.
