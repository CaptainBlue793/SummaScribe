param([int]$Port = 8502)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & conda run --no-capture-output -n starGPU python -m streamlit run Summascribe.py --server.port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
