# Evaluacion

## Alcance actual

La evaluacion automatica implementada cubre T3, es decir, recuperacion documental.

No evalua automaticamente:

- Normalizacion T2a.
- Traduccion T2b.
- Generacion T4a.
- Adaptacion final T4b.

## Dataset

Formato JSONL:

```json
{"query": "What procedure should be followed to resolve alarm 26120 associated with the axis?", "relevant_chunk_ids": ["C000985"], "notes": "Alarm 26120 remedy.", "difficulty": "medium"}
```

Dataset incluido:

- `data/eval/eval_queries.jsonl`
- `data/eval/DATASET_CARD.md`

Este dataset es pequeno y sirve como comprobacion reproducible del cableado. Para resultados defendibles en memoria, conviene ampliarlo con mas consultas y varios chunks relevantes cuando proceda.

Los `relevant_chunk_ids` deben pertenecer al archivo de chunks exportado con la misma estrategia de chunking. Si se cambia el chunking, los IDs pueden cambiar y el dataset debe regenerarse o realinearse.

El campo `notes` es una nota breve para auditoria humana. No participa en el calculo de metricas.

## Comando

```bash
python -m scripts.evaluate_retrieval --dataset ./data/eval/eval_queries.jsonl --modes bm25,vector,hybrid --top-k 5
```

Por defecto, ademas del resumen por consola, el script guarda un reporte JSON de auditoria en:

```text
data/eval/reports/retrieval_eval_<timestamp>.json
```

Tambien puede fijarse una ruta explicita:

```bash
python -m scripts.evaluate_retrieval --dataset ./data/eval/eval_queries.jsonl --modes bm25,vector,hybrid --top-k 5 --audit-output ./data/eval/reports/last_eval.json
```

Para ejecutar solo la salida historica de consola:

```bash
python -m scripts.evaluate_retrieval --dataset ./data/eval/eval_queries.jsonl --modes bm25,vector,hybrid --top-k 5 --no-audit
```

## Metricas

Implementadas en `rag_system/metrics.py`:

- `Recall@k`
- `Hit@k`
- `MRR`
- `MAP`
- Latencia media/p95 por consulta de recuperacion T3

## Reporte de auditoria

El JSON de auditoria contiene:

- Metadatos de ejecucion: fecha, dataset, archivo de chunks, coleccion de Weaviate, `top_k`, modos evaluados y parametros relevantes de retrieval.
- `summaries`: metricas agregadas por modo, fallos, exitos, exitos parciales, latencia media/p95 y resumen por dificultad (`easy`, `medium`, `hard`).
- `examples`: detalle de cada pregunta con `query`, `difficulty`, `notes`, `relevant_chunk_ids`, metadatos de los chunks relevantes y resultados por modo.
- Para cada modo de cada pregunta: `success`, `fully_recalled`, `first_relevant_rank`, `missing_relevant_chunk_ids`, ranking recuperado, scores/ranks BM25-vector-RRF, metricas individuales y latencia.

Una pregunta se considera fallada en un modo cuando ningun `relevant_chunk_id` aparece en el top-k recuperado (`Hit@k = 0`). Si aparece al menos uno pero no todos los relevantes, queda marcada como exito parcial.

## Chunks de referencia

Los chunks base estan en:

- `data/processed/chunks_strategy2_2500c_150o.jsonl`

Si se cambia el chunking, hay que regenerar ese archivo o crear uno nuevo con nombre explicito, y actualizar los `relevant_chunk_ids` del dataset.

## Generacion de queries

El script `scripts/generate_rag_eval.py` permite generar el dataset de evaluacion con Groq.

Inputs actuales del script:

- `CHUNK_FILE = "data/processed/chunks_strategy2_2500c_150o.jsonl"`
- `OUTPUT_FILE = "data/eval/eval_queries.jsonl"`
- `GROQ_MODEL = "llama-3.3-70b-versatile"`
- `DIFFICULTY_TARGETS = {"easy": 50, "medium": 30, "hard": 20}`
- Variable de entorno obligatoria: `GROQ_API_KEY`

Ejecucion:

```bash
python -m scripts.generate_rag_eval
```

El script no lee Weaviate. Lee el JSONL de chunks, selecciona chunks sin repetir y pide a Groq una pregunta, una nota y una dificultad. Cada ejemplo queda vinculado al `chunk_id` del chunk de origen.

Si `data/eval/eval_queries.jsonl` ya existe, el script continua desde ahi. Para regenerar desde cero, hay que borrar o renombrar ese archivo antes de ejecutar el script.

## Latencia del ciclo completo T2 -> T4

La evaluacion de recuperacion anterior mide solo T3. Para medir tiempos del flujo completo se usa:

```bash
python -m scripts.evaluate_poc_latency --query "Error 26120 en el eje, que hago ahora?" --mode hybrid --top-k 5 --local-files-only
```

Tambien puede recibir un TXT o JSONL con consultas en espanol:

```bash
python -m scripts.evaluate_poc_latency --queries-file ./data/eval/poc_latency_queries.jsonl --mode hybrid --top-k 5 --local-files-only --output ./data/eval/reports/poc_latency_last.json
```

El reporte incluye:

- `initial_load_s`: tiempo de carga inicial del LLM, medido aparte.
- `t2a_s`: normalizacion en espanol.
- `t2b_s`: traduccion al ingles.
- `t3_s`: recuperacion RAG completa.
- `t3_embedding_s`: embedding de la query.
- `t3_retrieval_s`: busqueda/fusion una vez disponible el embedding.
- `t3_bm25_s`, `t3_vector_s`, `t3_fusion_s`: desglose interno de T3.
- `t4a_s`: generacion tecnica en ingles.
- `t4b_s`: traduccion/adaptacion final al espanol.
- `total_s`: tiempo total de consulta, sin incluir la carga inicial del LLM.

Para una unica consulta tambien puede usarse `scripts.run_poc`:

```bash
python -m scripts.run_poc --query "Error 26120 en el eje, que hago ahora?" --mode hybrid --top-k 5 --local-files-only --show-timings
```

