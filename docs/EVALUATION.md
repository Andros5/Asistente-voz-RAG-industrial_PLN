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
{"query": "What procedure should be followed to resolve alarm 26120 associated with the axis?", "relevant_chunk_ids": ["C001282"], "notes": "Alarm 26120 remedy.", "difficulty": "medium"}
```

Dataset incluido:

- `data/eval/eval_queries.sample.jsonl`

Este dataset es pequeno y sirve como comprobacion reproducible del cableado. Para resultados defendibles en memoria, conviene ampliarlo con mas consultas y varios chunks relevantes cuando proceda.

## Comando

```bash
python -m scripts.evaluate_retrieval --dataset ./data/eval/eval_queries.sample.jsonl --modes bm25,vector,hybrid --top-k 5
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

- `data/processed/chunks_default_220w_40o.jsonl`

Si se cambia el chunking, hay que regenerar ese archivo o crear uno nuevo con nombre explicito, y actualizar los `relevant_chunk_ids` del dataset.

