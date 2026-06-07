# Dataset Card - Evaluacion RAG T3

## Nombre

`eval_queries.jsonl`

## Proposito

Dataset pequeno y reproducible para evaluar la recuperacion documental de T3 en la PoC del asistente industrial. El objetivo es comprobar si el sistema recupera los chunks correctos del manual Siemens SINUMERIK 808D ADVANCED para consultas tecnicas en ingles.

Este dataset no evalua automaticamente T2a, T2b, T4a ni T4b. Para esas fases se usa ejecucion funcional o auditoria manual del ciclo completo.

## Fuente documental

Los ejemplos estan alineados con los chunks versionados en:

```text
data/processed/chunks_strategy2_2500c_150o.jsonl
```

Ese JSONL es la fuente canonica para construir Weaviate en la PoC. Si cambia el chunking, el tamano de chunk, el solapamiento o los IDs, este dataset debe revisarse o regenerarse.

## Ficheros

- `data/eval/eval_queries.jsonl`: dataset completo usado en la evaluacion.
- `data/eval/eval_queries.sample.jsonl`: muestra pequena para inspeccion rapida.
- `data/eval/reports/last_eval.json`: ultimo reporte auditable de evaluacion de recuperacion.

## Esquema

Cada linea es un objeto JSON independiente:

```json
{
  "query": "What procedure should be followed to resolve alarm 26120 associated with the axis?",
  "relevant_chunk_ids": ["C000985"],
  "notes": "Alarm 26120 remedy.",
  "difficulty": "medium"
}
```

Campos:

- `query`: consulta tecnica en ingles usada para evaluar T3.
- `relevant_chunk_ids`: lista de chunks que contienen la evidencia esperada.
- `notes`: nota breve para auditoria humana. No se usa para calcular metricas.
- `difficulty`: dificultad esperada del ejemplo (`easy`, `medium` o `hard`).

## Generacion

El dataset se genera con:

```bash
python -m scripts.generate_rag_eval
```

El script lee `data/processed/chunks_strategy2_2500c_150o.jsonl`, selecciona chunks sin repetir y usa Groq para proponer una consulta, una nota y una dificultad. El modelo configurado actualmente en el script es:

```text
llama-3.3-70b-versatile
```

Variable de entorno necesaria:

```text
GROQ_API_KEY
```

El script no lee Weaviate. Por eso la coherencia se garantiza usando el mismo JSONL de chunks tanto para generar el dataset como para indexar la base de datos.

## Distribucion actual

Objetivo de generacion:

- `easy`: 50 consultas.
- `medium`: 30 consultas.
- `hard`: 20 consultas.

Total esperado:

- 100 consultas.

## Uso en evaluacion

Comando recomendado:

```bash
python -m scripts.evaluate_retrieval --dataset ./data/eval/eval_queries.jsonl --modes bm25,vector,hybrid --top-k 5 --audit-output ./data/eval/reports/last_eval.json
```

Metricas calculadas:

- `Recall@k`
- `Hit@k`
- `MRR`
- `MAP`
- Latencia media y p95 por modo de recuperacion
- Resumen por dificultad
- Detalle por pregunta para auditoria de fallos

## Limitaciones

- Las consultas estan en ingles porque T3 recibe la salida de T2b.
- La evaluacion automatica cubre solo recuperacion, no calidad final de respuesta.
- La mayoria de ejemplos tiene un unico chunk relevante; por eso `Recall@5` y `Hit@5` tienden a coincidir.
- Las preguntas fueron generadas automaticamente y deben revisarse cuando se use el dataset como evidencia fuerte en la memoria.
- Si se cambia el archivo de chunks activo, los `relevant_chunk_ids` pueden dejar de ser validos.

## Criterio de validez

Antes de usar el dataset para resultados finales:

1. Indexar Weaviate desde `data/processed/chunks_strategy2_2500c_150o.jsonl`.
2. Confirmar que todos los `relevant_chunk_ids` existen en ese JSONL.
3. Ejecutar la evaluacion y guardar el reporte en `data/eval/reports/last_eval.json`.
4. Revisar manualmente los ejemplos fallados en el reporte de auditoria.
