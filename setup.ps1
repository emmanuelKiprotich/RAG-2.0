$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    $PythonCommand = $PythonLauncher.Source
    $PythonPrefix = @("-3")
} else {
    $PythonLauncher = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonLauncher) {
        throw "Python 3.10 or newer is required. Install Python, then run setup.ps1 again."
    }
    $PythonCommand = $PythonLauncher.Source
    $PythonPrefix = @()
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $PythonCommand @PythonPrefix -m venv (Join-Path $ProjectRoot ".venv")
}
if ($LASTEXITCODE -ne 0) { throw "Could not create the app's Python environment." }
& $VenvPython -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.10 or newer is required to run this app." }

Write-Host "Installing Personal RAG Assistant requirements..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install the app requirements." }

$SecretsPath = Join-Path $ProjectRoot ".streamlit\secrets.toml"
if (-not (Test-Path -LiteralPath $SecretsPath)) {
    $SecretsDirectory = Split-Path -Parent $SecretsPath
    New-Item -ItemType Directory -Force -Path $SecretsDirectory | Out-Null
    $ApiUrl = Read-Host "KerocheAI API URL [http://127.0.0.1:8787/v1]"
    if ([string]::IsNullOrWhiteSpace($ApiUrl)) { $ApiUrl = "http://127.0.0.1:8787/v1" }
    $SecureKey = Read-Host "KerocheAI developer API key (leave blank to enter it in the app later)" -AsSecureString
    $ApiKey = [System.Net.NetworkCredential]::new("", $SecureKey).Password
    $QuotedUrl = ConvertTo-Json -InputObject $ApiUrl -Compress
    $SecretsLines = @("KEROCHEAI_BASE_URL = $QuotedUrl")
    if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
        $QuotedKey = ConvertTo-Json -InputObject $ApiKey -Compress
        $SecretsLines += "KEROCHEAI_API_KEY = $QuotedKey"
    }
    [System.IO.File]::WriteAllLines($SecretsPath, [string[]]$SecretsLines, [System.Text.UTF8Encoding]::new($false))
    $ApiKey = $null
    $SecureKey.Dispose()
}

$DesktopPath = [Environment]::GetFolderPath("Desktop")
if (-not [string]::IsNullOrWhiteSpace($DesktopPath) -and (Test-Path -LiteralPath $DesktopPath)) {
    $ShortcutPath = Join-Path $DesktopPath "Personal RAG Assistant.lnk"
    $RunScript = Join-Path $ProjectRoot "run.ps1"
    $IconFile = Join-Path $ProjectRoot "assets\personal-rag-icon.ico"
    $Shell = New-Object -ComObject WScript.Shell
    $Shortcut = $Shell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = (Get-Command powershell.exe).Source
    $Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScript`""
    $Shortcut.WorkingDirectory = $ProjectRoot
    $Shortcut.IconLocation = "$IconFile,0"
    $Shortcut.Description = "Open your personal study assistant"
    $Shortcut.Save()
    Write-Host "Desktop shortcut created: $ShortcutPath"
}

Write-Host "Setup complete. Starting the assistant..."
& (Join-Path $ProjectRoot "run.ps1")
