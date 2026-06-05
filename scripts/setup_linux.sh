#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo "[Setup] Entorno listo."
echo "Arranca Weaviate con: docker compose up -d"
echo "Indexa el manual con: ./.venv/bin/python -m scripts.build_index --markdown ./data/manuals/808D_ADV_diagnostics_man_0718_en-US.md"
echo "Ejecuta la PoC con: ./.venv/bin/python -m scripts.run_poc --query \"Error 26120 en el eje, que hago ahora?\" --mode hybrid --top-k 5 --stream-final"
