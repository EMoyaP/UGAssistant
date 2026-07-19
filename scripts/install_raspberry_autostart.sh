#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="ugassistant.service"
TARGET_USER="${SUDO_USER:-${USER}}"
SERVICE_TARGET="/etc/systemd/system/${SERVICE_NAME}"
AUTOSTART_DIR="/home/${TARGET_USER}/.config/autostart"

if [ "$(id -u)" -ne 0 ]; then
    echo "Ejecuta: sudo ./scripts/install_raspberry_autostart.sh" >&2
    exit 1
fi

if [ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    echo "No existe ${PROJECT_ROOT}/.venv/bin/python. Ejecuta primero scripts/setup_raspberry.sh." >&2
    exit 1
fi

sed \
    -e "s|@PROJECT_ROOT@|${PROJECT_ROOT}|g" \
    -e "s|@USER@|${TARGET_USER}|g" \
    "${PROJECT_ROOT}/deploy/ugassistant.service.template" > "${SERVICE_TARGET}"

install -d -o "${TARGET_USER}" -g "${TARGET_USER}" "${AUTOSTART_DIR}"
install -o "${TARGET_USER}" -g "${TARGET_USER}" -m 0644 \
    "${PROJECT_ROOT}/deploy/ugassistant-kiosk.desktop.template" \
    "${AUTOSTART_DIR}/ugassistant-kiosk.desktop"

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

echo "UGAssistant se iniciara con el sistema y Chromium se abrira en modo quiosco."
