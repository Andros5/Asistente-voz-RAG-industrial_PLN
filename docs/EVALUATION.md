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

Este dataset es pequeno y sirve como comprobacion reproducible del cableado. Para resultados defendibles en memoria, conviene ampliarlo con mas consultas y varios chunks relevantes cuando proceda.

Los `relevant_chunk_ids` deben pertenecer al archivo de chunks exportado con la misma estrategia de chunking. Si se cambia el chunking, los IDs pueden cambiar y el dataset debe regenerarse o realinearse.

## Comando

```bash
python -m scripts.evaluate_retrieval --dataset ./data/eval/eval_queries.jsonl --modes bm25,vector,hybrid --top-k 5
```

## Metricas

Implementadas en `rag_system/metrics.py`:

- `Recall@k`
- `Hit@k`
- `MRR`
- `MAP`
- Latencia media por consulta

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

