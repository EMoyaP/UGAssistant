# AGENTS.md — UGAssistant

## Objetivo

Construir un asistente multimodal completamente local y gratuito que se desarrolle primero en Windows y se ejecute posteriormente en una Raspberry Pi 5 sin reescribir el núcleo de la aplicación.

El asistente debe:

* Conversar en español mediante voz.
* Usar modelos locales.
* Mostrar un avatar animado.
* Detectar presencia mediante webcam.
* Seguir el rostro de la persona con la mirada del avatar.
* Reaccionar visualmente mientras escucha, piensa y habla.
* Funcionar sin servicios de IA en la nube.
* No enviar audio, vídeo, texto ni telemetría a terceros.

## Hardware final objetivo

* Raspberry Pi 5.
* MicroSD A2/U3/V30 de 128 GB.
* Pantalla táctil GeeekPi de 7 pulgadas y 1024 × 600.
* Altavoces integrados en la pantalla.
* Micrófono USB Nicoone.
* Webcam Logitech C270.
* Raspberry Pi OS de 64 bits.

## Modelos bloqueados

Windows y Raspberry Pi deben utilizar exactamente los mismos modelos y archivos:

* LLM: Ollama `qwen3:1.7b`.
* STT: whisper.cpp `ggml-base.bin`, modelo multilingüe.
* TTS: Piper `es_ES-davefx-medium.onnx`.
* Visión facial: YuNet `face_detection_yunet_2023mar.onnx`.

No sustituir modelos, cuantizaciones o voces sin actualizar previamente `config/models.lock.yaml`.

Registrar en `models.lock.yaml`:

* Nombre lógico.
* Nombre de archivo.
* Versión o etiqueta.
* SHA-256.
* URL oficial.
* Tamaño.
* Parámetros de ejecución.

## Arquitectura

Utilizar una arquitectura modular:

* Backend: Python.
* API local: FastAPI.
* Interfaz: HTML, CSS y JavaScript.
* Comunicación de estados: WebSocket.
* LLM: API local de Ollama.
* STT: proceso local de whisper.cpp.
* TTS: Piper local.
* Visión: OpenCV.
* Configuración: YAML.
* Persistencia: SQLite únicamente cuando sea necesaria.

La interfaz debe funcionar:

* En Windows, como aplicación web local o PWA.
* En Raspberry Pi, mediante Chromium en modo quiosco.
* A una resolución base de 1024 × 600.

El núcleo de dominio no debe importar directamente APIs específicas de Windows o Raspberry Pi. Usar adaptadores de plataforma.

## Estados del asistente

Implementar una máquina de estados explícita:

* `SLEEPING`
* `IDLE`
* `PERSON_DETECTED`
* `LISTENING`
* `TRANSCRIBING`
* `THINKING`
* `SPEAKING`
* `INTERRUPTED`
* `ERROR`

La interfaz y el avatar deben reaccionar a estos estados.

## Restricciones de rendimiento

Diseñar siempre para la Raspberry Pi 5:

* Cámara de análisis: máximo 640 × 480.
* Detección facial: objetivo de 5 a 10 FPS.
* Interfaz gráfica: objetivo de 30 FPS.
* Una única inferencia pesada simultánea.
* Contexto conversacional limitado.
* Respuestas de voz breves.
* No analizar cada fotograma con el LLM.
* No cargar modelos visuales generativos.
* No depender de GPU en Windows.

El perfil Windows debe permitir ejecutar el mismo modo limitado de Raspberry.

## Privacidad

* No guardar vídeo por defecto.
* No guardar audio por defecto.
* Descartar los fotogramas después de analizarlos.
* Mostrar indicadores visibles de cámara y micrófono.
* Proporcionar controles para desactivar cámara y micrófono.
* No implementar reconocimiento de identidad facial.
* No realizar inferencias sobre emociones o atributos personales.

## Calidad

* Añadir type hints.
* Usar logs estructurados.
* Separar dominio, infraestructura e interfaz.
* Evitar archivos excesivamente grandes.
* Añadir pruebas unitarias para la lógica independiente del hardware.
* Simular cámara, micrófono, LLM, STT y TTS en las pruebas.
* Gestionar errores de dispositivos desconectados.
* No ocultar excepciones sin registrarlas.
* Mantener instrucciones reproducibles para Windows y Raspberry Pi.

## Flujo de trabajo

Antes de implementar una fase:

1. Revisar la arquitectura existente.
2. Identificar riesgos de compatibilidad Windows/ARM64.
3. Dividir el trabajo en cambios verificables.
4. Implementar únicamente el alcance solicitado.
5. Ejecutar las pruebas.
6. Documentar comandos y resultados.
7. Informar de limitaciones reales.

No descargar modelos pesados automáticamente durante las pruebas.

## Comandos esperados

El repositorio debe proporcionar progresivamente:

* `scripts/setup_windows.ps1`
* `scripts/setup_raspberry.sh`
* `scripts/download_models.py`
* `scripts/check_hardware.py`
* `scripts/run_windows.ps1`
* `scripts/run_raspberry.sh`

Las instalaciones deben ser repetibles y no depender de pasos manuales no documentados.
