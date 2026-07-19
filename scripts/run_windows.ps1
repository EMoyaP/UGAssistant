$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LauncherExe = Join-Path $ProjectRoot "UGAssistant.exe"

if (Test-Path $LauncherExe) {
    Start-Process -FilePath $LauncherExe -WorkingDirectory $ProjectRoot
    exit 0
}

Write-Warning "No se encontro UGAssistant.exe. Se iniciara el lanzador de desarrollo."
& (Join-Path $PSScriptRoot "open_launcher_windows.ps1")
