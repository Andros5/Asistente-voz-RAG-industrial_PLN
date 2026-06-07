# PoC completa T2 -> T3 -> T4

Implementacion de la prueba de concepto descrita en la memoria final del proyecto incluida en `docs/Proyecto_Longitudinal_PLN.pdf`.
La entrada principal es `scripts.run_poc` y ejecuta el ciclo completo:

```text
consulta ES -> T2a normalizacion -> T2b traduccion EN -> T3 RAG -> T4a respuesta EN -> T4b respuesta ES
```

## Arquitectura

- `T0`: parseo opcional del PDF a Markdown con LlamaCloud.
- `T2/T4`: LLM local `mistralai/Ministral-3-8B-Instruct-2512`.
- `T3`: Weaviate local con chunks Markdown, BM25, embeddings `BAAI/bge-base-en-v1.5` y fusion RRF.

Weaviate se ejecuta en Docker con volumen persistente. Los modelos no se meten en Docker: se guardan en la cache local de Hugging Face del sistema anfitrion.

La PoC usa un unico entorno Python. BGE se carga directamente con `AutoTokenizer` y `AutoModel`, asi que T2, T3 y T4 pueden convivir en el mismo entorno.

## Estructura

```text
rag_system/
  chunking.py        # Chunking por palabras y Strategy 2 por alarmas/campos
  embeddings.py      # Embeddings BGE con AutoTokenizer/AutoModel
  weaviate_store.py  # Coleccion, insercion y busquedas BM25/vector
  retriever.py       # BM25, vector e hibrido RRF
  ingest.py          # Construccion del indice
  pipeline.py        # API de recuperacion T3
  poc_prompts.py     # Prompts T2/T4 de la PoC
  metrics.py         # Recall@k, Hit@k, MRR, MAP
llm_system/
  local_llm.py       # Carga local de Ministral 3 y query_llm
poc_system/
  orchestrator.py    # Orquestacion T2a -> T2b -> T3 -> T4a -> T4b
scripts/
  setup_windows.ps1
  setup_linux.sh
  build_index.py
  query_rag.py
  run_poc.py
  evaluate_retrieval.py
  evaluate_poc_latency.py
  download_models.py
  generate_rag_eval.py
  export_chunks.py
integration/
  poc_loop_t2_t3_t4.py
docs/
  CHUNKING.md        # Analisis de estrategias y chunking activo
  EVALUATION.md      # Dataset, metricas y reportes
  assets/chunking/   # Figuras del analisis de chunking
```

## Requisitos

- Docker.
- Python con `pip`. Se recomienda Python 3.11 o 3.12 para evitar problemas de compatibilidad con PyTorch.
- Git instalado, porque `transformers` se instala desde GitHub para tener soporte actualizado de Ministral 3.
- Acceso a la cache de Hugging Face donde este descargado `mistralai/Ministral-3-8B-Instruct-2512`, o conexion para descargarlo.

El `requirements.txt` instala `transformers` desde la rama principal de Hugging Face porque Ministral 3 lo requiere, junto con `mistral-common>=1.8.6`.

## Instalacion

Windows:

```powershell
.\scripts\setup_windows.ps1
```

Linux/macOS:

```bash
bash scripts/setup_linux.sh
```

Con conda tambien puedes usar tu entorno y ejecutar directamente:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cache de modelos

Los modelos se descargan una vez y quedan persistidos en la cache local de Hugging Face. Para controlar la ubicacion:

Windows:

```powershell
$env:HF_HOME="C:\hf-cache-pln"
```

Linux/macOS:

```bash
export HF_HOME="$HOME/hf-cache-pln"
```

Pre-descarga BGE y Ministral:

```bash
python -m scripts.download_models --target all
```

Si el LLM ya esta descargado, `--local-files-only` evita que `transformers` intente volver a bajarlo al ejecutar la PoC.

## Arrancar Weaviate

```bash
docker compose up -d
```

## Indexar en Weaviate desde el JSONL

```bash
python -m scripts.build_index --chunks ./data/processed/chunks_strategy2_2500c_150o.jsonl
```

La indexacion lee los chunks versionados en JSONL, calcula embeddings BGE y guarda todo en Weaviate. Hay que repetirla si cambia el JSONL, el modelo de embeddings o se borra el volumen de Weaviate.

Para desarrollo tambien se puede regenerar desde Markdown con `--from-markdown`, pero el despliegue reproducible de la PoC debe usar `--chunks`.

## Probar solo T3

```bash
python -m scripts.query_rag "What is the remedy for alarm 26120?" --mode hybrid --top-k 5
```

La salida queda lista para T4a:

```text
[REF:1] internal_id=C000985; source=...; section=26120 ...
texto del chunk...
```

## Ejecutar la PoC completa

```bash
python -m scripts.run_poc --query "Error 26120 en el eje, que hago ahora?" --mode hybrid --top-k 5 --stream-final
```

Si Ministral 3 ya esta descargado:

```bash
python -m scripts.run_poc --query "Error 26120 en el eje, que hago ahora?" --mode hybrid --top-k 5 --stream-final --local-files-only
```

El comando imprime cada subtarea:

```text
[T2a] consulta normalizada en espanol
[T2b] consulta traducida al ingles
[T3] evidencia recuperada con [REF:n]
[T4a] respuesta tecnica en ingles
[T4b] respuesta final en espanol
```

Para imprimir tiempos por fase en una consulta:

```bash
python -m scripts.run_poc --query "Error 26120 en el eje, que hago ahora?" --mode hybrid --top-k 5 --local-files-only --show-timings
```

Para guardar un JSON auditable de la consulta:

```bash
python -m scripts.run_poc --query "Error 26120 en el eje, que hago ahora?" --mode hybrid --top-k 5 --local-files-only --timings-output ./data/eval/reports/poc_latency_single.json
```

## Bucle interactivo

```bash
python -m integration.poc_loop_t2_t3_t4
```

Este bucle carga el LLM una vez y permite lanzar varias consultas seguidas.

## Parseo opcional con LlamaCloud

Si ya tienes `data/manuals/808D_ADV_diagnostics_man_0718_en-US.md`, no hace falta parsear el PDF.

```bash
python -m pip install -r requirements-parse.txt
export LLAMA_CLOUD_API_KEY="llx-..."
python -m scripts.parse_with_llamacloud --pdf ./RAG-docs/808D_ADV_diagnostics_man_0718_en-US.pdf --output ./data/manuals/808D_ADV_diagnostics_man_0718_en-US.md
```

## Evaluacion de T3

```bash
python -m scripts.build_index --chunks ./data/processed/chunks_strategy2_2500c_150o.jsonl
python -m scripts.evaluate_retrieval --dataset ./data/eval/eval_queries.jsonl --modes bm25,hybrid --top-k 5
```

El script reporta por consola `Recall@k`, `Hit@k`, `MRR`, `MAP`, latencia media y numero de fallos. Tambien guarda por defecto un reporte JSON auditable en `data/eval/reports/`.

El paso de `export_chunks` solo es necesario si se cambia la estrategia de chunking. Para reproducir la entrega, la indexacion debe hacerse desde el JSONL versionado `data/processed/chunks_strategy2_2500c_150o.jsonl`.

Para fijar una ruta concreta:

```bash
python -m scripts.evaluate_retrieval --dataset ./data/eval/eval_queries.jsonl --modes bm25,vector,hybrid --top-k 5 --audit-output ./data/eval/reports/last_eval.json
```

El reporte contiene cada pregunta con dificultad, notas, chunks relevantes, ranking recuperado por modo, scores/ranks, fallos, exitos parciales, posicion del primer relevante, metricas individuales y resumen por dificultad.

La ficha del dataset esta en `data/eval/DATASET_CARD.md`.

## Auditoria de latencia T2 -> T4

La evaluacion de T3 mide recuperacion. Para medir el ciclo completo con LLM:

```bash
python -m scripts.evaluate_poc_latency --query "Error 26120 en el eje, que hago ahora?" --mode hybrid --top-k 5 --local-files-only
```

Para varias consultas, usa un TXT o JSONL con `query_es`:

```bash
python -m scripts.evaluate_poc_latency --queries-file ./data/eval/poc_latency_queries.jsonl --mode hybrid --top-k 5 --local-files-only --output ./data/eval/reports/poc_latency_last.json
```

El reporte separa `T2a`, `T2b`, `T3`, `T4a`, `T4b` y `total_s`. Dentro de T3 separa el tiempo de embedding de query (`t3_embedding_s`) y la recuperacion/fusion posterior (`t3_retrieval_s`, `t3_bm25_s`, `t3_vector_s`, `t3_fusion_s`). La carga inicial del LLM se mide aparte como `initial_load_s` o `load_s`.

## Generacion de queries de evaluacion

```bash
export GROQ_API_KEY="gsk-..."
python -m scripts.generate_rag_eval
```

El generador lee `data/processed/chunks_strategy2_2500c_150o.jsonl` y escribe `data/eval/eval_queries.jsonl`. No lee Weaviate. Si el archivo de salida ya existe, continua desde los ejemplos existentes; para regenerar desde cero hay que borrarlo o renombrarlo antes.
