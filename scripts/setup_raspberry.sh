#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

echo "UGAssistant Raspberry Pi setup"
echo "Los modelos no se descargan automaticamente."

python3 - <<'PY'
import platform
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ es obligatorio.")
if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
    print("Aviso: este script esta pensado para Raspberry Pi OS 64-bit.")
PY

if [ ! -d "${VENV_PATH}" ]; then
    python3 -m venv "${VENV_PATH}"
fi

MISSING_SYSTEM_PACKAGES=()
for package in libportaudio2 alsa-utils build-essential cmake; do
    if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q "install ok installed"; then
        MISSING_SYSTEM_PACKAGES+=("${package}")
    fi
done

if [ ${#MISSING_SYSTEM_PACKAGES[@]} -gt 0 ]; then
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        apt-get install -y "${MISSING_SYSTEM_PACKAGES[@]}"
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y "${MISSING_SYSTEM_PACKAGES[@]}"
    else
        echo "Faltan paquetes del sistema: ${MISSING_SYSTEM_PACKAGES[*]}" >&2
        echo "Ejecuta el setup con un usuario que pueda usar sudo." >&2
        exit 1
    fi
fi

"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install \
  --only-binary=opencv-python-headless,numpy \
  -r "${PROJECT_ROOT}/requirements.txt"

echo "Modelos de vision (descarga explicita):"
echo "  ${VENV_PATH}/bin/python scripts/download_models.py --model face_detection"
echo "  ${VENV_PATH}/bin/python scripts/download_models.py --model palm_detection"
echo "  ${VENV_PATH}/bin/python scripts/download_models.py --model hand_pose"
echo "Voz local Piper (instalacion y descarga explicitas):"
echo "  ${VENV_PATH}/bin/python scripts/install_piper.py"
echo "  ${VENV_PATH}/bin/python scripts/download_models.py --model tts"
echo "  ${VENV_PATH}/bin/python scripts/download_models.py --model tts_config"
echo "Reconocimiento local whisper.cpp (instalacion y descarga explicitas):"
echo "  ${VENV_PATH}/bin/python scripts/install_whisper.py"
echo "  ${VENV_PATH}/bin/python scripts/download_models.py --model stt"
echo "LLM local Ollama para Linux ARM64 (instalacion y descarga explicitas):"
echo "  curl -fsSL https://ollama.com/install.sh | sh"
echo "  ${VENV_PATH}/bin/python scripts/download_models.py --model llm"
echo "Comprueba audio: ${VENV_PATH}/bin/python scripts/check_audio.py"
echo "Entorno listo. Ejecuta: ${VENV_PATH}/bin/python -m ugassistant"
echo "Inicio automatico en Raspberry: sudo ./scripts/install_raspberry_autostart.sh"
