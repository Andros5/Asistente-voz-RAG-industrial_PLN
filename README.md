# Asistente de voz RAG industrial PLN

Repositorio de la PoC T2 -> T3 -> T4 para un asistente tecnico de mantenimiento industrial sobre el manual Siemens SINUMERIK 808D ADVANCED.

El flujo completo es:

```text
consulta oral ES -> T2a normalizacion ES -> T2b traduccion EN -> T3 RAG -> T4a respuesta EN -> T4b respuesta ES
```

## Contenido del repositorio

```text
data/
  manuals/       Markdown fuente parseado del manual
  processed/     Chunks exportados con la configuracion base
  eval/          Dataset JSONL pequeno para evaluar recuperacion
docs/
  Proyecto_Longitudinal_PLN.pdf
  CHUNKING.md
  EVALUATION.md
rag_system/      Chunking, embeddings BGE, Weaviate, retrieval y metricas
llm_system/      Carga local de Ministral 3
poc_system/      Orquestacion T2a -> T2b -> T3 -> T4a -> T4b
scripts/         CLI para setup, indexacion, consulta, evaluacion y descarga de modelos
integration/     Bucle interactivo de la PoC
tests/           Tests ligeros de chunking, metricas y datos de evaluacion
```

## Requisitos

- Python 3.11 o 3.12 recomendado.
- Docker para Weaviate.
- Git instalado, porque `transformers` se instala desde GitHub para tener soporte actualizado de Ministral 3.
- Cache local de Hugging Face con `mistralai/Ministral-3-8B-Instruct-2512`, o conexion para descargarlo.

Los modelos no se versionan en Git. Se almacenan en la cache local de Hugging Face del equipo.

## Instalacion rapida

Windows:

```powershell
.\scripts\setup_windows.ps1
```

Linux/macOS:

```bash
bash scripts/setup_linux.sh
```

Instalacion manual:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En Linux/macOS cambia `.\.venv\Scripts\python.exe` por `./.venv/bin/python`.

## Configuracion

Copia `.env.example` a `.env` si quieres cambiar rutas o parametros:

```powershell
Copy-Item .env.example .env
```

Valores principales:

- `RAG_MARKDOWN_PATH=./data/manuals/808D_ADV_diagnostics_man_0718_en-US.md`
- `RAG_CHUNK_WORDS=220`
- `RAG_CHUNK_OVERLAP_WORDS=40`
- `RAG_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`
- `LLM_MODEL_ID=mistralai/Ministral-3-8B-Instruct-2512`

## Arrancar Weaviate

```bash
docker compose up -d
```

## Construir el indice

```bash
python -m scripts.build_index --markdown ./data/manuals/808D_ADV_diagnostics_man_0718_en-US.md
```

La indexacion genera chunks, calcula embeddings BGE y guarda los objetos en Weaviate. No hay que repetirla antes de cada prueba: solo si cambia el Markdown, el chunking, el modelo de embeddings o se borra el volumen de Weaviate.

## Probar T3

```bash
python -m scripts.query_rag "What procedure should be followed to resolve alarm 26120 associated with the axis?" --mode hybrid --top-k 5
```

## Ejecutar la PoC completa

```bash
python -m scripts.run_poc --query "Error 26120 en el eje, que hago ahora?" --mode hybrid --top-k 5 --stream-final
```

Si el LLM ya esta descargado:

```bash
python -m scripts.run_poc --query "Error 26120 en el eje, que hago ahora?" --mode hybrid --top-k 5 --stream-final --local-files-only
```

El comando imprime las salidas intermedias de T2a, T2b, T3, T4a y T4b.

## Evaluacion de recuperacion

El repositorio incluye un dataset pequeno para validar el cableado de evaluacion:

```bash
python -m scripts.evaluate_retrieval --dataset ./data/eval/eval_queries.sample.jsonl --modes bm25,hybrid --top-k 5
```

Metricas implementadas: `Recall@k`, `Hit@k`, `MRR`, `MAP` y latencia media. La evaluacion automatica cubre T3; la calidad completa T2 -> T4 se revisa funcionalmente.

## Exportar chunks

```bash
python -m scripts.export_chunks --markdown ./data/manuals/808D_ADV_diagnostics_man_0718_en-US.md --output ./data/processed/chunks_default_220w_40o.jsonl
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Los tests no cargan el LLM ni Weaviate. Comprueban chunking, metricas y consistencia del dataset de evaluacion.

## Documentacion

- `docs/Proyecto_Longitudinal_PLN.pdf`: memoria base del trabajo.
- `docs/CHUNKING.md`: funcionamiento actual y puntos de extension del chunking.
- `docs/EVALUATION.md`: formato del dataset y uso de metricas.

