param(
    [switch]$InstallBuildTool
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LauncherSource = Join-Path $ProjectRoot "scripts\windows_launcher.py"
$BuildDirectory = Join-Path $ProjectRoot "build\pyinstaller"

if (-not (Test-Path $PythonExe)) {
    throw "No se encontro .venv\Scripts\python.exe. Ejecuta scripts\setup_windows.ps1 primero."
}

& $PythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    if (-not $InstallBuildTool) {
        throw "PyInstaller no esta instalado. Ejecuta: .\scripts\build_windows_launcher.ps1 -InstallBuildTool"
    }
    & $PythonExe -m pip install "PyInstaller>=6.0,<7.0"
}

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --noconsole `
    --name "UGAssistant" `
    --distpath $ProjectRoot `
    --workpath $BuildDirectory `
    --specpath $BuildDirectory `
    $LauncherSource

Write-Host "Ejecutable creado: $ProjectRoot\UGAssistant.exe"
