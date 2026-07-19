param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

Write-Host "UGAssistant Windows setup"
Write-Host "Los modelos no se descargan automaticamente."

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ no esta disponible en PATH."
}

if (-not (Test-Path $VenvPath)) {
    python -m venv $VenvPath
}

& $PythonExe -m pip install --upgrade pip

if (-not $SkipInstall) {
    & $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}

Write-Host "Modelos de vision (descarga explicita):"
Write-Host "  & $PythonExe scripts\download_models.py --model face_detection"
Write-Host "  & $PythonExe scripts\download_models.py --model palm_detection"
Write-Host "  & $PythonExe scripts\download_models.py --model hand_pose"
Write-Host "Voz local Piper (instalacion y descarga explicitas):"
Write-Host "  & $PythonExe scripts\install_piper.py"
Write-Host "  & $PythonExe scripts\download_models.py --model tts"
Write-Host "  & $PythonExe scripts\download_models.py --model tts_config"
Write-Host "Reconocimiento local whisper.cpp (instalacion y descarga explicitas):"
Write-Host "  & $PythonExe scripts\install_whisper.py"
Write-Host "  & $PythonExe scripts\download_models.py --model stt"
Write-Host "LLM local Ollama (instalacion y descarga explicitas):"
Write-Host "  winget install --id Ollama.Ollama --exact"
Write-Host "  & $PythonExe scripts\download_models.py --model llm"
Write-Host "Comprueba audio: & $PythonExe scripts\check_audio.py"
Write-Host "Abre el frontal: .\scripts\run_windows.ps1"
