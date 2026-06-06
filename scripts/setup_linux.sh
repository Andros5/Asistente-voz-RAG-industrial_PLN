#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo "[Setup] Entorno listo."
echo "Arranca Weaviate con: docker compose up -d"
echo "Indexa Weaviate con: ./.venv/bin/python -m scripts.build_index --chunks ./data/processed/chunks_strategy2_2500c_150o.jsonl"
echo "Ejecuta la PoC con: ./.venv/bin/python -m scripts.run_poc --query \"Error 26120 en el eje, que hago ahora?\" --mode hybrid --top-k 5 --stream-final"
