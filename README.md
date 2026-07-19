# UGAssistant

UGAssistant es un asistente multimodal local para Windows y Raspberry Pi 5.
Procesa voz, imagen y texto en el propio equipo: no requiere suscripciones ni
envia contenido de la conversacion a servicios de IA en la nube.

> Estado: desarrollo activo. Windows es el entorno de desarrollo y Raspberry
> Pi 5 el objetivo de despliegue, con el mismo nucleo y modelos bloqueados.

## Estado actual

La aplicacion incluye:

- Backend Python con FastAPI y WebSockets.
- Interfaz HTML, CSS y JavaScript para 1024 x 600.
- Maquina de estados independiente y probada.
- Avatar con expresiones de reposo, presencia, escucha, pensamiento, habla, error, aburrimiento y sueno.
- Mirada controlada por el raton y por la posicion del rostro.
- Captura de camara OpenCV a un maximo de 640 x 480 y 8 FPS.
- Deteccion facial local con el YuNet bloqueado.
- Deteccion local de hasta dos manos con 21 puntos por mano.
- Recuento de 0 a 5 dedos por mano y total estable de 0 a 10.
- Deteccion, seleccion y monitorizacion local de microfonos mediante PortAudio.
- Deteccion de actividad de sonido con nivel RMS, histeresis y estado `LISTENING`.
- Reconocimiento local con whisper.cpp y el modelo multilingue bloqueado
  `ggml-base.bin`, limitado por la aplicacion a castellano y frances.
- Flujo manual de prueba: escucha, cierre por silencio, transcripcion, seleccion
  automatica de idioma/voz y repeticion de la frase reconocida.
- Turno por palabra de activacion local: `hola` inicia el turno en espanol y
  `salut` lo inicia en frances; despues escucha la pregunta y la procesa localmente.
- LLM local mediante Ollama y el modelo bloqueado `qwen3:1.7b`, con historial
  limitado, una unica inferencia pesada simultanea y eleccion verbal entre
  respuesta corta o completa antes de cada consulta.
- Sintesis de texto local con Piper y las voces bloqueadas
  `es_ES-davefx-medium` y `fr_FR-tom-medium`.
- Reproduccion por el altavoz seleccionado, volumen y velocidad configurables,
  sincronizacion labial por RMS y fragmentacion de respuestas extensas con una
  pausa breve configurable entre partes.
- Paneo estereo suave segun la posicion horizontal del rostro detectado.
- Gestos geometricos: puno, palma, senalar, pulgar arriba/abajo y victoria.
- El avatar sonrie con pulgar arriba y muestra tristeza con lagrimas con pulgar abajo.
- Selector de camaras instaladas y opcion `Ninguna`.
- Enumeracion por backend OpenCV para conservar la correspondencia nombre-indice.
- Vista previa MJPEG local que no guarda fotogramas.
- Adaptadores simulados para pruebas sin hardware.

## Experiencia local

La pantalla principal esta pensada para 1024 x 600: muestra la palabra de
activacion, hora y fecha junto al avatar. Durante una interaccion aparece un
panel de lectura con la pregunta y la respuesta. Tras cada respuesta, el
asistente permite pedir una aclaracion manteniendo el contexto local de la
sesion; puede terminarse por silencio, una respuesta negativa o pulgar abajo.

El modal de configuracion concentra dispositivos, volumen, velocidad de voz,
voces, palabras de activacion en castellano y frances, diagnosticos de camara y
audio, y el cierre ordenado del sistema.

Los runtimes de Piper y whisper.cpp, las voces TTS en castellano y frances y
`ggml-base.bin` ya estan instalados para desarrollo en Windows. Ollama y
`qwen3:1.7b` se instalan deliberadamente mediante un paso explicito.

El proyecto toma como referencia la experiencia de BillAI Bass, sustituyendo
el pez y los componentes mecanicos por el avatar de pantalla. La arquitectura
no usa Strands, Bedrock ni otros servicios remotos: vision, voz y el futuro LLM
se ejecutan localmente y sin suscripciones.

## Windows

Abre PowerShell en el proyecto. Si la ejecucion de scripts esta bloqueada, habilitala solo para la ventana actual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Prepara o actualiza el entorno:

```powershell
.\scripts\setup_windows.ps1
```

Descarga o verifica de forma explicita los modelos de vision bloqueados:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --model face_detection
.\.venv\Scripts\python.exe scripts\download_models.py --model palm_detection
.\.venv\Scripts\python.exe scripts\download_models.py --model hand_pose
```

Instala el runtime bloqueado de Piper y descarga explicitamente las voces
bloqueadas. DaveFX sigue siendo la voz predeterminada establecida en AGENTS.md;
Tom es la voz francesa masculina solicitada y no sustituye a DaveFX:

```powershell
.\.venv\Scripts\python.exe scripts\install_piper.py
.\.venv\Scripts\python.exe scripts\download_models.py --model tts
.\.venv\Scripts\python.exe scripts\download_models.py --model tts_config
.\.venv\Scripts\python.exe scripts\download_models.py --model tts_fr
.\.venv\Scripts\python.exe scripts\download_models.py --model tts_fr_config
```

Los scripts verifican plataforma, tamano y SHA-256. No sustituyen la voz ni
descargan el LLM automaticamente.

Instala Ollama y descarga explicitamente el unico LLM bloqueado:

```powershell
winget install --id Ollama.Ollama --exact
.\.venv\Scripts\python.exe scripts\download_models.py --model llm
```

Ollama descarga el modelo una vez desde su registro y lo sirve despues por la
API local `127.0.0.1:11434`; UGAssistant no usa una API de IA externa durante
el funcionamiento.

Instala el runtime bloqueado de whisper.cpp y descarga explicitamente el
modelo STT multilingue bloqueado:

```powershell
.\.venv\Scripts\python.exe scripts\install_whisper.py
.\.venv\Scripts\python.exe scripts\download_models.py --model stt
.\.venv\Scripts\python.exe scripts\check_stt.py --language es
.\.venv\Scripts\python.exe scripts\check_stt.py --language fr
```

Comprueba que la camara configurada abre un fotograma:

```powershell
.\.venv\Scripts\python.exe scripts\check_camera.py
```

Enumera microfonos y altavoces sin abrirlos:

```powershell
.\.venv\Scripts\python.exe scripts\check_audio.py
```

Genera el ejecutable local de Windows (solo necesita hacerse tras cambios en el
lanzador):

```powershell
.\scripts\build_windows_launcher.ps1 -InstallBuildTool
```

Inicia UGAssistant desde el ejecutable `UGAssistant.exe`, que arranca el
backend local y abre el asistente en el navegador:

```powershell
.\UGAssistant.exe
```

Tambien puedes usar `.\scripts\run_windows.ps1`, que usa el ejecutable cuando
existe. Dentro de Configuracion, el boton `Cerrar sistema` apaga el backend de
forma ordenada; el ejecutable espera ese cierre y termina tambien. Si ya hay
un servidor local iniciado manualmente, el ejecutable abre el navegador sin
tomar control de ese proceso.

La vista de desarrollo de vision esta disponible en
`http://127.0.0.1:8000/debug/camera`. Reutiliza la captura y la inferencia del
servicio principal; no abre la webcam ni ejecuta inferencias una segunda vez.

La vista de desarrollo de audio esta disponible en
`http://127.0.0.1:8000/debug/audio`. Muestra el nivel RMS, el umbral, el
historial reciente, los dispositivos seleccionados, el paneo y el estado del
asistente. Tambien permite elegir la voz instalada, escribir una frase y
reproducirla con Piper. `Reconocer voz` abre el microfono seleccionado, espera
una frase, termina tras 1,5 segundos de silencio, transcribe en castellano o
frances y reproduce el texto con DaveFX o Tom. El boton permite cancelar el
proceso mientras esta activo. Reutiliza los servicios principales y no abre un
segundo flujo de audio.

Con microfono y altavoz activos, el modo normal queda esperando `hola` o
`salut`. Tras reconocerlos, saluda en espanol o frances, espera una segunda
frase y pregunta si se desea una respuesta `corta` o `completa` antes de enviar
la consulta a `qwen3:1.7b` por la API local de Ollama. La respuesta se
reproduce con DaveFX o Tom segun el idioma transcrito. Si la
primera frase no contiene una de esas palabras, se descarta. Cada captura se
cierra tras dos segundos de silencio y vuelve a reposo si no hay activacion o
pregunta. El breve bufer necesario para no cortar la activacion permanece solo
en memoria y se borra despues de cada intento.

Si la deteccion global de Whisper devuelve otro idioma, UGAssistant compara
secuencialmente las transcripciones forzadas a castellano y frances. En frases
intrinsecamente ambiguas, como una lista de numeros que se convierte solo en
digitos, el idioma de la voz seleccionada sirve como desempate. No se ejecutan
dos inferencias pesadas al mismo tiempo.

La pantalla principal agrupa la configuracion de camara, microfono, altavoces,
idioma y voz en el boton de engranaje. Es posible elegir entre espanol/DaveFX y
frances/Tom. Cada dispositivo y la voz conservan su seleccion, incluso al
cerrar el modal y reiniciar el servidor.

Las preferencias locales se guardan en `data/preferences.yaml` mediante una
escritura atomica. Incluyen camara, microfono, altavoz, estados activo/inactivo,
volumen, velocidad, voz e idioma. Los dispositivos se restauran por nombre y backend, no
solo por el indice variable de Windows o ALSA. Si el dispositivo guardado no
esta conectado, UGAssistant selecciona ninguno en vez de cambiar a otro. El
archivo es local, no contiene audio o video y no se envia fuera del equipo.

## Spotify opcional

Spotify es la unica integracion externa opcional. La voz, vision, LLM y TTS
siguen ejecutandose en local. Para activarlo, crea una aplicacion personal en
el panel de Spotify for Developers, copia su `Client ID` en Configuracion y
registra exactamente esta URI de retorno:

```text
http://127.0.0.1:8000/api/spotify/callback
```

Pulsa `Conectar` para abrir la autorizacion oficial de Spotify. Tras esta
actualizacion, desconecta y conecta Spotify una vez para conceder los permisos
del reproductor local. UGAssistant usa
OAuth 2.0 con PKCE: no lee ni guarda cookies del navegador. El token renovable
se conserva solo en `data/spotify.tokens.json`, cifrado con DPAPI en Windows y
con permisos locales restrictivos en Raspberry Pi; queda fuera del control de
versiones. `Desconectar` borra ese token local.

Con Spotify conectado y un dispositivo Spotify activo, se puede decir `hola,
pon musica` para que pregunte que se desea escuchar, o `hola, reproduce <tema>`
para buscarlo directamente. Tambien se entiende `hola, reproduce el ultimo
disco de Shakira`: UGAssistant consulta los albumes oficiales del artista y
reproduce el mas reciente disponible en Spotify. La frase adicional
`ordenado por popularidad` se ignora en ese caso porque un album se reproduce
en el orden fijado por su autor. `hola, deten la reproduccion` y el gesto de
cremallera pausan Spotify. El control remoto de reproduccion requiere Spotify
Premium; el lateral muestra la portada original enlazada a Spotify, la pista y
ofrece pausa, reanudar, anterior y siguiente cuando Spotify permite esos
controles. Tambien se admiten `hola, pausar`, `hola, reanudar`, `hola,
siguiente`, `hola, anterior`, `hola, subir volumen` y `hola, bajar volumen`.

La salida de altavoz configurada en UGAssistant se aplica al audio de Piper.
Spotify reproduce desde su cliente o dispositivo Spotify Connect activo, por lo
que el destino de su musica se elige en Spotify, Chromium o en el sistema
operativo. En Raspberry Pi el navegador de quiosco usa la salida de audio
predeterminada de ALSA.

UGAssistant crea un reproductor Spotify Connect propio dentro de Chromium con
el Web Playback SDK oficial. Esto evita abrir una pestaña o aplicacion de
Spotify aparte en navegadores compatibles. Requiere Spotify Premium y, por la
politica de reproduccion automatica del navegador, puede requerir pulsar una vez
`Activar reproductor local` en Configuracion al iniciar la sesion.

## Dispositivos de audio

La aplicacion enumera entradas y salidas con `sounddevice` y PortAudio. En
Windows prioriza WASAPI y en Raspberry Pi OS prioriza ALSA, evitando que el
mismo dispositivo aparezca repetido por varios backends. Cada dispositivo
informa de su indice real, canales, frecuencia predeterminada, backend y si es
el endpoint predeterminado.

Los selectores `Micro` y `Altavoz` permiten conservar una seleccion durante la
sesion o elegir `Sin microfono`/`Sin altavoz`. El boton `Activar micro` abre una
entrada mono, calcula el nivel RMS por bloques de 50 ms y descarta cada bloque
inmediatamente. Cuando el nivel supera el umbral estable, la maquina cambia a
`LISTENING`; tras el periodo de silencio vuelve a `PERSON_DETECTED` o `IDLE`.

La API equivalente es:

- `GET /api/audio/devices`
- `GET /api/audio`
- `POST /api/audio/select/input/{indice}`
- `POST /api/audio/select/output/{indice}`
- `POST /api/audio/enable`
- `POST /api/audio/disable`
- `POST /api/audio/output/enable`
- `POST /api/audio/output/disable`
- `POST /api/audio/output/volume/{porcentaje}`
- `WS /ws/audio`
- `GET /api/tts`
- `POST /api/tts/select/{voice_id}`
- `POST /api/tts/speed/{porcentaje}`
- `POST /api/tts/language/{idioma}`
- `POST /api/tts/speak` con JSON `{"text": "Hola"}`
- `WS /ws/tts`
- `GET /api/stt`
- `POST /api/stt/recognize`
- `POST /api/stt/cancel`
- `WS /ws/stt`
- `GET /api/llm`
- `POST /api/llm/ask` con JSON `{"text": "Hola", "language": "es"}`
- `WS /ws/llm`
- `GET /api/assistant`
- `WS /ws/assistant`
- `GET /api/preferences`

Usa `-1` como indice para seleccionar ninguno. La monitorizacion ordinaria
descarta cada bloque de entrada inmediatamente. El reconocimiento mantiene la
frase solo en memoria y whisper.cpp elimina su WAV temporal al terminar. Los
umbrales, volumen y paneo se ajustan en `audio`; los tiempos de espera y silencio
del reconocimiento se ajustan en `stt`, dentro de `config/app.yaml`.

Al comenzar cada frase, el servicio toma la ultima coordenada horizontal del
rostro. El centro se reproduce equilibrado; hacia la izquierda o derecha reduce
suavemente el canal opuesto sin silenciarlo. El valor se congela durante esa
frase para evitar saltos por fluctuaciones de deteccion. Sin rostro o con la
camara apagada, la salida queda centrada.

Windows entrega el PCM original de Piper al mezclador WASAPI. En Linux/ALSA,
si la salida necesita 48 kHz, UGAssistant utiliza remuestreo Lanczos por bloques
con NumPy, ya incluido por OpenCV, y no requiere otra dependencia.

## Uso de la camara

La camara empieza apagada. El menu superior enumera los dispositivos disponibles:

- Seleccionar una camara la abre y activa la deteccion.
- `Ninguna` cierra el dispositivo y descarta el ultimo fotograma.
- El boton `Camara` permite pausar o reanudar el dispositivo seleccionado.
- Si hay un rostro, el estado cambia a `PERSON_DETECTED` y las pupilas siguen su centro.
- Las manos detectadas muestran sus 21 puntos y el gesto reconocido en la vista de desarrollo.
- Los fotogramas solo se mantienen en memoria y no se escriben en disco.

## Manos y gestos

La deteccion usa los modelos MP-PalmDet y MP-HandPose de OpenCV Zoo mediante
OpenCV DNN sobre CPU. No depende del paquete `mediapipe`, lo que mantiene la
misma ruta de ejecucion en Windows x64 y Linux ARM64.

La cara y las manos comparten el mismo fotograma de la webcam. YuNet se ejecuta
a la cadencia de captura y las manos cada tres fotogramas, aproximadamente 2.7
veces por segundo con la configuracion actual de 8 FPS. Las redes se ejecutan
secuencialmente y se limita la salida a dos manos.

Los gestos se clasifican a partir de la geometria de los 21 puntos, sin un
modelo adicional. Se reconocen `CLOSED_FIST`, `OPEN_PALM`, `POINTING`,
`THUMB_UP`, `THUMB_DOWN` y `VICTORY`; las poses ambiguas se informan como
`UNKNOWN`.

La vista de desarrollo muestra el recuento de cada mano y un total de 0 a 10.
El total se publica tras dos inferencias de mano consecutivas con el mismo
resultado para evitar parpadeos: `0` representa un puno cerrado y `--` indica
que no hay una mano detectada. El dato se expone tambien como `finger_count`
en la API y el WebSocket de camara; no altera la expresion del avatar.

YuNet aporta ademas cinco puntos normalizados: ambos ojos, punta de la nariz y
las dos comisuras de la boca. Combinados con los puntos de la mano permiten
detectar `BOTH_HANDS_OVER_EYES`, `POINTING_AT_NOSE`, `POINTING_AT_MOUTH`,
`HAND_OVER_MOUTH`,
`OPEN_PALM_NEAR_FACE`, `VICTORY_NEAR_FACE`, `THUMB_UP_NEAR_FACE` y
`THUMB_DOWN_NEAR_FACE`. Son reglas espaciales deliberadas; no se infieren
emociones ni identidad.

`POINTING_AT_MOUTH` muestra la cremallera en el avatar y detiene una lectura
en curso de forma local.

La frecuencia puede reducirse para Raspberry Pi en `config/app.yaml`:

```yaml
hands:
  enabled: true
  inference_interval_frames: 3
  finger_count_stable_samples: 2
  max_hands: 2
```

El indice inicial se configura en `config/app.yaml`:

```yaml
camera:
  device_index: 0
  enabled_by_default: false
  preview_fps: 8
```

Para probar otro indice desde la terminal:

```powershell
.\.venv\Scripts\python.exe scripts\check_camera.py --device-index 1
```

## Comportamiento del avatar

Al mover el raton, las pupilas siguen el cursor y la cara muestra curiosidad. Cuando no hay interaccion ni un rostro presente, el avatar pasa por este ciclo aproximado:

1. Tras 35 segundos muestra aburrimiento.
2. Tras 47 segundos entra en `SLEEPING` y muestra `zZz`.
3. Cerca del minuto despierta y vuelve a `IDLE`.

Cualquier movimiento, clic, tecla o rostro detectado reinicia el ciclo.
Durante `TRANSCRIBING`, el avatar conserva el estado real en la interfaz y usa
la misma expresion y animacion visual que durante `THINKING`.

## Raspberry Pi 5

La dependencia esta fijada a `opencv-python-headless==4.13.0.92`. PyPI proporciona una rueda `manylinux2014_aarch64`, por lo que el script impide compilar OpenCV desde fuentes accidentalmente.

En Raspberry Pi OS de 64 bits:

```bash
cd ~/UGAssistant
chmod +x scripts/setup_raspberry.sh scripts/run_raspberry.sh
./scripts/setup_raspberry.sh
.venv/bin/python scripts/download_models.py --model face_detection
.venv/bin/python scripts/download_models.py --model palm_detection
.venv/bin/python scripts/download_models.py --model hand_pose
.venv/bin/python scripts/install_piper.py
.venv/bin/python scripts/download_models.py --model tts
.venv/bin/python scripts/download_models.py --model tts_config
.venv/bin/python scripts/download_models.py --model tts_fr
.venv/bin/python scripts/download_models.py --model tts_fr_config
.venv/bin/python scripts/install_whisper.py
.venv/bin/python scripts/download_models.py --model stt
.venv/bin/python scripts/download_models.py --model llm
.venv/bin/python scripts/check_stt.py --language es
.venv/bin/python scripts/check_stt.py --language fr
.venv/bin/python scripts/check_camera.py
.venv/bin/python scripts/check_audio.py
./scripts/run_raspberry.sh
```

Para el despliegue habitual en Raspberry, activa el inicio automatico una vez
terminada la configuracion. Crea un servicio `systemd` para el backend y un
autostart de Chromium en modo quiosco:

```bash
chmod +x scripts/install_raspberry_autostart.sh
sudo ./scripts/install_raspberry_autostart.sh
```

El servicio usa `Restart=on-failure`: se inicia de nuevo en cada arranque del
sistema, pero el boton `Cerrar sistema` de Configuracion lo detiene de forma
intencionada y no se reinicia hasta el siguiente arranque o un `systemctl start
ugassistant`.

Instala Ollama ARM64 de forma explicita antes de descargar el LLM:

```bash
curl -fsSL https://ollama.com/install.sh | sh
.venv/bin/python scripts/download_models.py --model llm
```

El setup instala `libportaudio2` y `alsa-utils` si faltan. PortAudio utiliza
ALSA para enumerar el microfono USB y las salidas disponibles; no abre ningun
flujo durante la comprobacion.

La Logitech C270 se utiliza mediante V4L2 y aparecera como `/dev/videoN`. El usuario que ejecuta Chromium y UGAssistant debe tener acceso al grupo `video`. Si el indice cambia, modifica `camera.device_index` en `config/app.yaml` o usa `--device-index` en la comprobacion.

OpenCV 4 se mantiene deliberadamente: el archivo bloqueado `face_detection_yunet_2023mar.onnx` tiene dimensiones fijas y la documentacion oficial advierte de diferencias con el motor ONNX de OpenCV 5. No se sustituye por el modelo YuNet de 2026.

## Diagnostico

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\check_hardware.py
```

La comprobacion informa sobre plataforma, Python, Ollama, dispositivos de
camara y audio, OpenCV, whisper.cpp, Ollama, el modelo `qwen3:1.7b` instalado y
espacio libre.

## Pruebas

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Las pruebas unitarias usan adaptadores simulados y no necesitan abrir la camara ni acceder a Internet.

## Modelos bloqueados

`config/models.lock.yaml` conserva exactamente:

- LLM: Ollama `qwen3:1.7b`.
- STT: whisper.cpp `ggml-base.bin`.
- TTS espanol: Piper `es_ES-davefx-medium.onnx`.
- TTS frances: Piper `fr_FR-tom-medium.onnx`.
- Vision: YuNet `face_detection_yunet_2023mar.onnx`.
- Palma: MP-PalmDet `palm_detection_mediapipe_2023feb.onnx`.
- Pose de mano: MP-HandPose `handpose_estimation_mediapipe_2023feb.onnx`.

Los tres modelos de vision, las voces TTS DaveFX y Tom y el modelo STT estan
descargados y verificados. El modelo LLM se instala explicitamente con
`scripts/download_models.py --model llm`; su contenido y digest se gestionan
localmente por Ollama bajo la etiqueta bloqueada `qwen3:1.7b`.
`config/runtimes.lock.yaml` fija Piper `2023.11.14-2` y whisper.cpp `v1.8.1`
para Windows AMD64 y Linux ARM64; no se sustituye ningun modelo.

## Privacidad

- Camara y microfono empiezan desactivados.
- No se guarda audio ni video.
- El WAV temporal generado por Piper se elimina inmediatamente despues de
  cargarlo en memoria para la reproduccion.
- El audio de reconocimiento se mantiene en memoria; el WAV temporal necesario
  para whisper.cpp se elimina al terminar cada transcripcion.
- No hay reconocimiento de identidad ni analisis de emociones.
- No se envia telemetria ni contenido a servicios externos.
- La previsualizacion y la deteccion se sirven unicamente desde `127.0.0.1`.
