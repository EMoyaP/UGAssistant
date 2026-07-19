$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Launcher = Join-Path $ProjectRoot "scripts\windows_launcher.py"

if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Start-Process -FilePath $PythonExe -ArgumentList "`"$Launcher`"" -WorkingDirectory $ProjectRoot -WindowStyle Hidden
