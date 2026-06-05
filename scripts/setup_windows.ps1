param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

Write-Host "[Setup] Creando entorno unico de la PoC (.venv)"
& $Python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "[Setup] Entorno listo."
Write-Host "Arranca Weaviate con: docker compose up -d"
Write-Host "Indexa el manual con: .\.venv\Scripts\python.exe -m scripts.build_index --markdown .\data\manuals\808D_ADV_diagnostics_man_0718_en-US.md"
Write-Host "Ejecuta la PoC con: .\.venv\Scripts\python.exe -m scripts.run_poc --query `"Error 26120 en el eje, que hago ahora?`" --mode hybrid --top-k 5 --stream-final"
