const connection = document.querySelector(".connection");
const connectionText = document.querySelector("#connectionText");
const assistantState = document.querySelector("#assistantState");
const inputSelect = document.querySelector("#inputSelect");
const outputSelect = document.querySelector("#outputSelect");
const monitorToggle = document.querySelector("#monitorToggle");
const levelStage = document.querySelector(".level-stage");
const levelValue = document.querySelector("#levelValue");
const levelFill = document.querySelector("#levelFill");
const thresholdMarker = document.querySelector("#thresholdMarker");
const detectionText = document.querySelector("#detectionText");
const levelHistory = document.querySelector("#levelHistory");
const inputName = document.querySelector("#inputName");
const outputName = document.querySelector("#outputName");
const hostApi = document.querySelector("#hostApi");
const channelCount = document.querySelector("#channelCount");
const sampleRate = document.querySelector("#sampleRate");
const monitorStatus = document.querySelector("#monitorStatus");
const rmsValue = document.querySelector("#rmsValue");
const thresholdValue = document.querySelector("#thresholdValue");
const soundStatus = document.querySelector("#soundStatus");
const inputCount = document.querySelector("#inputCount");
const outputCount = document.querySelector("#outputCount");
const outputVolume = document.querySelector("#outputVolume");
const outputBalance = document.querySelector("#outputBalance");
const voiceSelect = document.querySelector("#voiceSelect");
const speechText = document.querySelector("#speechText");
const speakButton = document.querySelector("#speakButton");
const recognizeButton = document.querySelector("#recognizeButton");
const speechMessage = document.querySelector("#speechMessage");
const ttsStatus = document.querySelector("#ttsStatus");
const sttStatus = document.querySelector("#sttStatus");
const ttsModel = document.querySelector("#ttsModel");
const ttsLanguage = document.querySelector("#ttsLanguage");
const sttModel = document.querySelector("#sttModel");
const detectedLanguage = document.querySelector("#detectedLanguage");
const llmStatus = document.querySelector("#llmStatus");
const llmModel = document.querySelector("#llmModel");
const voiceTurn = document.querySelector("#voiceTurn");

const historyBars = Array.from({ length: 40 }, () => {
  const bar = document.createElement("span");
  levelHistory.append(bar);
  return bar;
});

let historyIndex = 0;
let inventorySignature = "";
let voiceInventorySignature = "";
let latestAudioStatus = null;
let latestTTSStatus = null;
let latestRecognitionStatus = null;
let latestLLMStatus = null;

function languageLabel(language) {
  const labels = {
    es_ES: "Español (España)",
    fr_FR: "Français (France)",
    es: "Español",
    fr: "Français",
  };
  return labels[language] || language;
}

function refreshSpeechControls() {
  const voices = Array.isArray(latestTTSStatus?.voices)
    ? latestTTSStatus.voices
    : [];
  const hasAvailableVoice = voices.some((voice) => voice.available);
  const outputReady = Boolean(
    latestAudioStatus?.output_enabled
    && latestAudioStatus?.selected_output_index !== null,
  );
  const busy = Boolean(latestTTSStatus?.busy);
  const recognitionBusy = Boolean(latestRecognitionStatus?.busy);
  const inputReady = latestAudioStatus?.selected_input_index !== null;
  const recording = Boolean(latestAudioStatus?.recording);
  const monitoring = Boolean(latestAudioStatus?.monitoring);
  voiceSelect.disabled = busy || recognitionBusy || !hasAvailableVoice;
  speechText.disabled = busy || recognitionBusy || !hasAvailableVoice;
  speakButton.disabled = busy
    || recognitionBusy
    || !hasAvailableVoice
    || !outputReady
    || !speechText.value.trim();
  monitorToggle.disabled = busy || recognitionBusy || !inputReady;
  inputSelect.disabled = busy || recognitionBusy || monitoring || recording;
  outputSelect.disabled = busy || recognitionBusy || Boolean(latestAudioStatus?.output_playing);
  recognizeButton.disabled = recognitionBusy
    ? false
    : busy
      || !latestRecognitionStatus?.available
      || !inputReady
      || !outputReady;
  recognizeButton.dataset.active = String(recognitionBusy);
  recognizeButton.textContent = recognitionBusy ? "Cancelar" : "Reconocer voz";
}

function populateSelect(select, devices, selectedIndex, noneLabel) {
  const noneOption = document.createElement("option");
  noneOption.value = "-1";
  noneOption.textContent = noneLabel;
  select.replaceChildren(noneOption);
  devices.forEach((device) => {
    const option = document.createElement("option");
    option.value = String(device.device_index);
    option.textContent = `${device.name}${device.is_default ? " (pred.)" : ""}`;
    select.append(option);
  });
  select.value = selectedIndex === null ? "-1" : String(selectedIndex);
  select.disabled = false;
}

function selectedDevice(devices, selectedIndex) {
  return devices.find((device) => device.device_index === selectedIndex) || null;
}

function applyAudioStatus(payload) {
  latestAudioStatus = payload;
  const inputs = Array.isArray(payload.inputs) ? payload.inputs : [];
  const outputs = Array.isArray(payload.outputs) ? payload.outputs : [];
  const signature = JSON.stringify([
    inputs.map((device) => [device.device_index, device.name, device.is_default]),
    outputs.map((device) => [device.device_index, device.name, device.is_default]),
  ]);
  if (signature !== inventorySignature) {
    inventorySignature = signature;
    populateSelect(inputSelect, inputs, payload.selected_input_index, "Sin microfono");
    populateSelect(outputSelect, outputs, payload.selected_output_index, "Sin altavoz");
  } else {
    inputSelect.value = payload.selected_input_index === null
      ? "-1"
      : String(payload.selected_input_index);
    outputSelect.value = payload.selected_output_index === null
      ? "-1"
      : String(payload.selected_output_index);
  }

  const input = selectedDevice(inputs, payload.selected_input_index);
  const output = selectedDevice(outputs, payload.selected_output_index);
  const monitoring = Boolean(payload.monitoring);
  const recording = Boolean(payload.recording);
  const soundDetected = Boolean(payload.sound_detected);
  const level = Number(payload.input_level) || 0;
  const threshold = Math.max(Number(payload.activation_threshold) || 0.015, 0.001);
  const displayMaximum = Math.max(threshold * 3, 0.05);
  const scaledLevel = Math.min(level / displayMaximum, 1);

  levelStage.dataset.monitoring = String(monitoring || recording);
  levelStage.dataset.sound = String(soundDetected);
  levelValue.textContent = level.toFixed(4);
  levelFill.style.width = `${scaledLevel * 100}%`;
  thresholdMarker.style.left = `${Math.min(threshold / displayMaximum, 1) * 100}%`;
  detectionText.textContent = recording
    ? soundDetected
      ? "GRABANDO VOZ"
      : "ESPERANDO VOZ"
    : !monitoring
    ? "MICRO APAGADO"
    : soundDetected
      ? "SONIDO DETECTADO"
      : "ESCUCHANDO";

  monitorToggle.setAttribute("aria-pressed", String(monitoring));
  monitorToggle.textContent = monitoring ? "Desactivar" : "Activar micro";
  monitorToggle.disabled = payload.selected_input_index === null;
  inputSelect.disabled = monitoring;
  outputSelect.disabled = Boolean(payload.output_playing);

  inputName.textContent = input?.name || "Ninguna";
  inputName.title = input?.name || "";
  outputName.textContent = output?.name || "Ninguna";
  outputName.title = output?.name || "";
  hostApi.textContent = input?.host_api || "--";
  channelCount.textContent = input ? String(input.channels) : "--";
  sampleRate.textContent = input?.default_sample_rate
    ? `${Math.round(input.default_sample_rate)} Hz`
    : "--";
  monitorStatus.textContent = recording
    ? "Reconociendo"
    : monitoring ? "Activo" : "Inactivo";
  rmsValue.textContent = level.toFixed(4);
  thresholdValue.textContent = threshold.toFixed(4);
  soundStatus.textContent = soundDetected ? "Si" : "No";
  inputCount.textContent = String(inputs.length);
  outputCount.textContent = String(outputs.length);
  outputVolume.textContent = `${Math.round((Number(payload.output_volume) || 0) * 100)}%`;
  const balance = Number(payload.output_balance) || 0;
  outputBalance.textContent = Math.abs(balance) < 0.01
    ? "Centro"
    : `${Math.round(Math.abs(balance) * 100)}% ${balance < 0 ? "Izq." : "Der."}`;

  const bar = historyBars[historyIndex];
  bar.style.height = `${Math.max(2, scaledLevel * 90)}px`;
  bar.classList.toggle("active", soundDetected);
  historyIndex = (historyIndex + 1) % historyBars.length;
  refreshSpeechControls();
}

function applyTTSStatus(payload) {
  latestTTSStatus = payload;
  const voices = Array.isArray(payload.voices) ? payload.voices : [];
  const signature = JSON.stringify(
    voices.map((voice) => [
      voice.voice_id,
      voice.display_name,
      voice.language,
      voice.available,
    ]),
  );
  if (signature !== voiceInventorySignature) {
    voiceInventorySignature = signature;
    voiceSelect.replaceChildren();
    voices.forEach((voice) => {
      const option = document.createElement("option");
      option.value = voice.voice_id;
      option.textContent = `${voice.display_name} · ${languageLabel(voice.language)}`;
      option.disabled = !voice.available;
      voiceSelect.append(option);
    });
  }
  voiceSelect.value = payload.selected_voice_id || "";
  const selectedVoice = voices.find(
    (voice) => voice.voice_id === payload.selected_voice_id,
  );
  ttsModel.textContent = selectedVoice?.display_name || "--";
  ttsModel.title = selectedVoice?.voice_id || "";
  ttsLanguage.textContent = payload.selected_language
    ? languageLabel(payload.selected_language)
    : "--";

  const phaseLabels = {
    synthesizing: "Generando voz local",
    playing: "Reproduciendo",
    ready: payload.available ? "Piper listo" : "Piper no disponible",
    error: "Error de sintesis",
    not_scanned: "Comprobando Piper",
  };
  ttsStatus.textContent = phaseLabels[payload.phase] || payload.detail || "--";
  if (payload.phase === "playing") {
    speechMessage.dataset.error = "false";
    speechMessage.textContent = "Reproduciendo por el dispositivo seleccionado";
  } else if (payload.phase === "ready" && speechMessage.textContent) {
    speechMessage.dataset.error = "false";
    speechMessage.textContent = "Reproduccion terminada";
  }
  refreshSpeechControls();
}

function applyRecognitionStatus(payload) {
  latestRecognitionStatus = payload;
  const phaseLabels = {
    ready: "Whisper listo",
    unavailable: "Whisper no disponible",
    listening: "Escuchando voz",
    transcribing: "Transcribiendo",
    recognized: "Texto reconocido",
    completed: "Proceso completado",
    cancelled: "Reconocimiento cancelado",
    timeout: "No se detecto voz",
    unsupported_language: "Idioma no compatible",
    error: "Error de reconocimiento",
    not_scanned: "Comprobando Whisper",
  };
  sttStatus.textContent = phaseLabels[payload.phase] || payload.detail || "--";
  sttModel.textContent = payload.available ? "Whisper base" : "No disponible";
  detectedLanguage.textContent = payload.language
    ? languageLabel(payload.language)
    : "--";
  if (payload.transcript) {
    speechText.value = payload.transcript;
  }
  if (payload.phase === "completed") {
    speechMessage.dataset.error = "false";
    speechMessage.textContent = "Texto reconocido y reproducido";
  } else if (["timeout", "unsupported_language", "error"].includes(payload.phase)) {
    speechMessage.dataset.error = "true";
    speechMessage.textContent = phaseLabels[payload.phase];
  } else if (payload.phase === "cancelled") {
    speechMessage.dataset.error = "false";
    speechMessage.textContent = "Reconocimiento cancelado";
  }
  refreshSpeechControls();
}

function applyLLMStatus(payload) {
  latestLLMStatus = payload;
  llmModel.textContent = payload.model_available ? "gemma3:4b" : "No disponible";
  const phaseLabels = {
    ready: "Ollama listo",
    unavailable: "Ollama o modelo no disponible",
    thinking: "Generando respuesta local",
    completed: "Respuesta lista",
    error: "Error de Ollama",
    not_scanned: "Comprobando Ollama",
  };
  llmStatus.textContent = phaseLabels[payload.phase] || payload.detail || "--";
}

function applyVoiceAssistantStatus(payload) {
  const phaseLabels = {
    waiting_for_wake_word: "Esperando hola o salut",
    detecting_wake_word: "Reconociendo activacion",
    greeting: "Saludando",
    listening_for_question: "Escuchando pregunta",
    thinking: "Consultando modelo",
    speaking: "Reproduciendo respuesta",
    error: "Error",
  };
  voiceTurn.textContent = phaseLabels[payload.phase] || payload.detail || "--";
}

function applyState(payload) {
  assistantState.textContent = payload.state || "--";
}

async function updateMonitor(enabled) {
  monitorToggle.disabled = true;
  try {
    const action = enabled ? "enable" : "disable";
    const response = await fetch(`/api/audio/${action}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Audio monitor failed: ${response.status}`);
    }
    applyAudioStatus(payload);
  } catch (error) {
    detectionText.textContent = "ERROR DE AUDIO";
    console.error(error);
  } finally {
    monitorToggle.disabled = inputSelect.value === "-1";
  }
}

async function selectDevice(kind, deviceIndex) {
  const select = kind === "input" ? inputSelect : outputSelect;
  select.disabled = true;
  try {
    const response = await fetch(
      `/api/audio/select/${kind}/${deviceIndex}`,
      { method: "POST" },
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Audio selection failed: ${response.status}`);
    }
    applyAudioStatus(payload);
  } catch (error) {
    detectionText.textContent = "ERROR DE AUDIO";
    console.error(error);
  } finally {
    select.disabled = false;
  }
}

async function selectVoice(voiceId) {
  voiceSelect.disabled = true;
  try {
    const response = await fetch(
      `/api/tts/select/${encodeURIComponent(voiceId)}`,
      { method: "POST" },
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Voice selection failed: ${response.status}`);
    }
    applyTTSStatus(payload);
  } catch (error) {
    speechMessage.dataset.error = "true";
    speechMessage.textContent = error.message;
    console.error(error);
  } finally {
    refreshSpeechControls();
  }
}

async function speakText() {
  speakButton.disabled = true;
  speechMessage.dataset.error = "false";
  speechMessage.textContent = "Preparando voz local";
  try {
    const response = await fetch("/api/tts/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: speechText.value }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Speech synthesis failed: ${response.status}`);
    }
    applyTTSStatus(payload);
  } catch (error) {
    speechMessage.dataset.error = "true";
    speechMessage.textContent = error.message;
    console.error(error);
  } finally {
    refreshSpeechControls();
  }
}

async function toggleRecognition() {
  const cancelling = Boolean(latestRecognitionStatus?.busy);
  recognizeButton.disabled = true;
  speechMessage.dataset.error = "false";
  speechMessage.textContent = cancelling
    ? "Cancelando reconocimiento"
    : "Habla en español o francés";
  try {
    const response = await fetch(
      cancelling ? "/api/stt/cancel" : "/api/stt/recognize",
      { method: "POST" },
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Voice recognition failed: ${response.status}`);
    }
    applyRecognitionStatus(payload);
  } catch (error) {
    speechMessage.dataset.error = "true";
    speechMessage.textContent = error.message;
    console.error(error);
  } finally {
    refreshSpeechControls();
  }
}

async function loadInitialData() {
  const [audioResponse, stateResponse, ttsResponse, sttResponse, llmResponse, assistantResponse] = await Promise.all([
    fetch("/api/audio/devices"),
    fetch("/api/state"),
    fetch("/api/tts"),
    fetch("/api/stt"),
    fetch("/api/llm"),
    fetch("/api/assistant"),
  ]);
  if (!audioResponse.ok || !stateResponse.ok || !ttsResponse.ok || !sttResponse.ok || !llmResponse.ok || !assistantResponse.ok) {
    throw new Error("Initial audio data unavailable");
  }
  applyAudioStatus(await audioResponse.json());
  applyState(await stateResponse.json());
  applyTTSStatus(await ttsResponse.json());
  applyRecognitionStatus(await sttResponse.json());
  applyLLMStatus(await llmResponse.json());
  applyVoiceAssistantStatus(await assistantResponse.json());
}

function connectAudioSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/audio`);
  socket.addEventListener("open", () => {
    connection.dataset.connected = "true";
    connectionText.textContent = "Datos en directo";
  });
  socket.addEventListener("message", (event) => applyAudioStatus(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    connection.dataset.connected = "false";
    connectionText.textContent = "Reconectando";
    window.setTimeout(connectAudioSocket, 1200);
  });
  socket.addEventListener("error", () => {
    connection.dataset.connected = "false";
  });
}

function connectStateSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/state`);
  socket.addEventListener("message", (event) => applyState(JSON.parse(event.data)));
  socket.addEventListener("close", () => window.setTimeout(connectStateSocket, 1500));
}

function connectTTSSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/tts`);
  socket.addEventListener("message", (event) => applyTTSStatus(JSON.parse(event.data)));
  socket.addEventListener("close", () => window.setTimeout(connectTTSSocket, 1500));
}

function connectSTTSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/stt`);
  socket.addEventListener("message", (event) => {
    applyRecognitionStatus(JSON.parse(event.data));
  });
  socket.addEventListener("close", () => window.setTimeout(connectSTTSocket, 1500));
}

function connectLLMSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/llm`);
  socket.addEventListener("message", (event) => applyLLMStatus(JSON.parse(event.data)));
  socket.addEventListener("close", () => window.setTimeout(connectLLMSocket, 1500));
}

function connectVoiceAssistantSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/assistant`);
  socket.addEventListener("message", (event) => applyVoiceAssistantStatus(JSON.parse(event.data)));
  socket.addEventListener("close", () => window.setTimeout(connectVoiceAssistantSocket, 1500));
}

monitorToggle.addEventListener("click", () => {
  updateMonitor(monitorToggle.getAttribute("aria-pressed") !== "true");
});
inputSelect.addEventListener("change", () => selectDevice("input", inputSelect.value));
outputSelect.addEventListener("change", () => selectDevice("output", outputSelect.value));
voiceSelect.addEventListener("change", () => selectVoice(voiceSelect.value));
speechText.addEventListener("input", refreshSpeechControls);
speakButton.addEventListener("click", speakText);
recognizeButton.addEventListener("click", toggleRecognition);

loadInitialData().catch((error) => {
  connectionText.textContent = "Sin datos";
  console.error(error);
});
connectAudioSocket();
connectStateSocket();
connectTTSSocket();
connectSTTSocket();
connectLLMSocket();
connectVoiceAssistantSocket();
