$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Streamlit = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"
if (-not (Test-Path -LiteralPath $Streamlit)) {
    throw "The app is not set up yet. Run setup.ps1 first."
}
Set-Location -LiteralPath $ProjectRoot
& $Streamlit run (Join-Path $ProjectRoot "app.py") --server.headless false
