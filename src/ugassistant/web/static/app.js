const shell = document.querySelector(".app-shell");
const stage = document.querySelector(".stage");
const avatar = document.querySelector(".avatar");
const stateLabel = document.querySelector("#stateLabel");
const assistantPrompt = document.querySelector("#assistantPrompt");
const updatedAt = document.querySelector("#updatedAt");
const currentTime = document.querySelector("#currentTime");
const currentDate = document.querySelector("#currentDate");
const connectionDot = document.querySelector("#connectionDot");
const connectionText = document.querySelector("#connectionText");
const cameraToggle = document.querySelector("#cameraToggle");
const cameraSelect = document.querySelector("#cameraSelect");
const micSelect = document.querySelector("#micSelect");
const speakerSelect = document.querySelector("#speakerSelect");
const micToggle = document.querySelector("#micToggle");
const speakerToggle = document.querySelector("#speakerToggle");
const speakerVolume = document.querySelector("#speakerVolume");
const speakerVolumeValue = document.querySelector("#speakerVolumeValue");
const speechRate = document.querySelector("#speechRate");
const speechRateValue = document.querySelector("#speechRateValue");
const ttsLanguageSelect = document.querySelector("#ttsLanguageSelect");
const ttsVoiceSelect = document.querySelector("#ttsVoiceSelect");
const micLevelFill = document.querySelector("#micLevelFill");
const settingsButton = document.querySelector("#settingsButton");
const settingsDialog = document.querySelector("#settingsDialog");
const shutdownButton = document.querySelector("#shutdownButton");
const cameraSettingStatus = document.querySelector("#cameraSettingStatus");
const microphoneSettingStatus = document.querySelector("#microphoneSettingStatus");
const speakerSettingStatus = document.querySelector("#speakerSettingStatus");
const cameraStatus = document.querySelector("#cameraStatus");
const audioStatus = document.querySelector("#audioStatus");
const cameraPreview = document.querySelector("#cameraPreview");
const cameraFeed = document.querySelector("#cameraFeed");
const presenceText = document.querySelector("#presenceText");
const stateButtons = Array.from(document.querySelectorAll("[data-next-state]"));
const wakeSpanishInput = document.querySelector("#wakeSpanishInput");
const wakeFrenchInput = document.querySelector("#wakeFrenchInput");
const saveProfileButton = document.querySelector("#saveProfileButton");
const conversationTurns = document.querySelector("#conversationTurns");
const timerStack = document.querySelector("#timerStack");
const spotifyPanel = document.querySelector("#spotifyPanel");
const spotifyTrackTitle = document.querySelector("#spotifyTrackTitle");
const spotifyTrackArtist = document.querySelector("#spotifyTrackArtist");
const spotifyCoverLink = document.querySelector("#spotifyCoverLink");
const spotifyCover = document.querySelector("#spotifyCover");
const spotifyPlaybackState = document.querySelector("#spotifyPlaybackState");
const spotifyDeviceName = document.querySelector("#spotifyDeviceName");
const spotifyVolume = document.querySelector("#spotifyVolume");
const spotifyPreviousButton = document.querySelector("#spotifyPreviousButton");
const spotifyToggleButton = document.querySelector("#spotifyToggleButton");
const spotifyNextButton = document.querySelector("#spotifyNextButton");
const spotifyStopButton = document.querySelector("#spotifyStopButton");
const spotifyClientIdInput = document.querySelector("#spotifyClientIdInput");
const spotifySettingStatus = document.querySelector("#spotifySettingStatus");
const spotifySaveButton = document.querySelector("#spotifySaveButton");
const spotifyConnectButton = document.querySelector("#spotifyConnectButton");
const spotifyDisconnectButton = document.querySelector("#spotifyDisconnectButton");
const spotifyActivatePlayerButton = document.querySelector("#spotifyActivatePlayerButton");
const spotifyLocalPlayerStatus = document.querySelector("#spotifyLocalPlayerStatus");

const IDLE_BORED_AFTER_MS = 35000;
const IDLE_SLEEP_AFTER_MS = 47000;
const IDLE_WAKE_AFTER_MS = 59000;
const GESTURE_MOOD_HOLD_MS = 1800;

let latestCameraStatus = null;
let pointerActive = false;
let pointerTimer = null;
let pointerFrameRequested = false;
let latestPointer = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
let lastActivityAt = Date.now();
let idlePhase = "active";
let automaticStateRequest = false;
let gestureMoodTimer = null;
let audioInventorySignature = "";
let ttsInventorySignature = "";
let lipSyncFrame = null;
let assistantProfile = { spanish_wake_word: "hola", french_wake_word: "salut" };
let renderedTurnKey = "";
let spotifyStatus = { configured: false, connected: false, playback: null };
let spotifyPollTimer = null;
let spotifyWebPlayer = null;
let spotifyWebPlayerLoading = false;
let spotifyWebPlayerActivated = false;
let spotifyWebPlayerDeviceId = "";
let activeTimers = [];

function stopLipSync() {
  if (lipSyncFrame !== null) {
    window.cancelAnimationFrame(lipSyncFrame);
    lipSyncFrame = null;
  }
  shell.dataset.lipSync = "false";
  shell.style.setProperty("--speech-mouth-open", "0");
  shell.style.setProperty("--speech-mouth-width", "52px");
  shell.style.setProperty("--speech-mouth-height", "8px");
}

function startLipSync(levels, intervalMs) {
  stopLipSync();
  if (!Array.isArray(levels) || !levels.length || !Number.isFinite(intervalMs)) {
    return;
  }
  shell.dataset.lipSync = "true";
  const startedAt = window.performance.now();
  const updateMouth = (now) => {
    const cueIndex = Math.floor((now - startedAt) / intervalMs);
    if (cueIndex >= levels.length) {
      shell.style.setProperty("--speech-mouth-open", "0");
      shell.style.setProperty("--speech-mouth-width", "52px");
      shell.style.setProperty("--speech-mouth-height", "8px");
      lipSyncFrame = null;
      return;
    }
    const level = Math.min(Math.max(Number(levels[cueIndex]) || 0, 0), 1);
    shell.style.setProperty("--speech-mouth-open", level.toFixed(3));
    shell.style.setProperty("--speech-mouth-width", `${52 + (14 * level)}px`);
    shell.style.setProperty("--speech-mouth-height", `${8 + (36 * level)}px`);
    lipSyncFrame = window.requestAnimationFrame(updateMouth);
  };
  lipSyncFrame = window.requestAnimationFrame(updateMouth);
}

function languageLabel(language) {
  const labels = {
    es_ES: "Español (España)",
    fr_FR: "Français (France)",
  };
  return labels[language] || language;
}

function clearBoredMood() {
  if (shell.dataset.mood === "bored") {
    shell.dataset.mood = "";
  }
}

function clearGestureMood() {
  window.clearTimeout(gestureMoodTimer);
  gestureMoodTimer = null;
  if (
    ["happy", "sad", "playful", "surprised", "zipped", "eyes-covered"]
      .includes(shell.dataset.mood)
  ) {
    shell.dataset.mood = "";
  }
}

function setGestureMood(mood) {
  window.clearTimeout(gestureMoodTimer);
  shell.dataset.mood = mood;
  gestureMoodTimer = window.setTimeout(() => {
    if (shell.dataset.mood === mood) {
      shell.dataset.mood = "";
    }
    gestureMoodTimer = null;
  }, GESTURE_MOOD_HOLD_MS);
}

function applyDetectedMood(hands, combinedGestures) {
  const handGestures = new Set(
    (Array.isArray(hands) ? hands : []).map((hand) => hand.gesture),
  );
  const combined = new Set(
    (Array.isArray(combinedGestures) ? combinedGestures : [])
      .map((detection) => detection.gesture),
  );
  if (combined.has("BOTH_HANDS_OVER_EYES")) {
    setGestureMood("eyes-covered");
  } else if (handGestures.has("THUMB_DOWN") || combined.has("THUMB_DOWN_NEAR_FACE")) {
    setGestureMood("sad");
  } else if (combined.has("POINTING_AT_MOUTH")) {
    setGestureMood("zipped");
  } else if (combined.has("HAND_OVER_MOUTH")) {
    setGestureMood("surprised");
  } else if (combined.has("POINTING_AT_NOSE")) {
    setGestureMood("playful");
  } else if (
    handGestures.has("THUMB_UP")
    || combined.has("THUMB_UP_NEAR_FACE")
    || combined.has("VICTORY_NEAR_FACE")
    || combined.has("OPEN_PALM_NEAR_FACE")
  ) {
    setGestureMood("happy");
  }
}

function applyState(payload) {
  const state = payload.state || "ERROR";
  shell.dataset.state = state;
  stateLabel.textContent = state;
  const prompts = {
    LISTENING: "Te escucho",
    TRANSCRIBING: "Estoy entendiendo tu pregunta",
    THINKING: "Estoy pensando",
    SPEAKING: "Te respondo",
    ERROR: "Necesito revisar un dispositivo",
  };
  assistantPrompt.textContent = prompts[state]
    || `Di "${assistantProfile.spanish_wake_word}" o "${assistantProfile.french_wake_word}" para hablar con UGAssistant`;
  updatedAt.textContent = payload.updated_at
    ? new Date(payload.updated_at).toLocaleTimeString("es-ES")
    : "";
  stateButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.nextState === state);
  });
  if (state !== "IDLE") {
    clearBoredMood();
  }
  if (state !== "SLEEPING") {
    automaticStateRequest = false;
  }
}

function updateLocalClock() {
  const now = new Date();
  currentTime.textContent = now.toLocaleTimeString("es-ES", {
    hour: "2-digit",
    minute: "2-digit",
  });
  currentDate.textContent = now.toLocaleDateString("es-ES", {
    weekday: "long",
    day: "numeric",
    month: "long",
  });
}

async function loadAssistantProfile() {
  const response = await fetch("/api/assistant/profile");
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Profile request failed");
  assistantProfile = payload;
  wakeSpanishInput.value = payload.spanish_wake_word;
  wakeFrenchInput.value = payload.french_wake_word;
  applyState({ state: shell.dataset.state || "IDLE" });
}

async function saveAssistantProfile() {
  saveProfileButton.disabled = true;
  try {
    const response = await fetch("/api/assistant/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        spanish_wake_word: wakeSpanishInput.value,
        french_wake_word: wakeFrenchInput.value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Profile update failed");
    assistantProfile = payload;
    applyState({ state: shell.dataset.state || "IDLE" });
  } catch (error) {
    console.error(error);
  } finally {
    saveProfileButton.disabled = false;
  }
}

function applySpotifyStatus(payload) {
  spotifyStatus = payload;
  const playback = payload.playback || null;
  const isPlaying = Boolean(playback?.is_playing);
  // The player and music expression are reserved for active playback.
  shell.dataset.spotify = isPlaying ? "true" : "false";
  shell.dataset.spotifyPlaying = isPlaying ? "true" : "false";
  spotifyTrackTitle.textContent = playback?.title || "Sin reproduccion";
  spotifyTrackArtist.textContent = playback?.artists || "Di que musica quieres escuchar";
  const hasCover = Boolean(playback?.album_art_url);
  spotifyCoverLink.hidden = !hasCover;
  if (hasCover) {
    spotifyCover.src = playback.album_art_url;
    spotifyCover.alt = `Portada de ${playback.album || playback.title}`;
    spotifyCoverLink.href = playback.spotify_url || "https://open.spotify.com";
  } else {
    spotifyCover.removeAttribute("src");
  }
  spotifyPlaybackState.textContent = isPlaying ? "Reproduciendo" : "Listo para reproducir";
  spotifyDeviceName.textContent = playback?.device_name || "Spotify conectado";
  spotifyVolume.textContent = playback?.supports_volume && Number.isInteger(playback.volume_percent)
    ? `Volumen de Spotify: ${playback.volume_percent}%`
    : "";
  spotifyToggleButton.innerHTML = isPlaying ? "&#10074;&#10074;" : "&#9654;";
  spotifyToggleButton.setAttribute("aria-label", isPlaying ? "Pausar" : "Reanudar");
  spotifyToggleButton.title = isPlaying ? "Pausar" : "Reanudar";
  const controlsEnabled = Boolean(payload.connected && playback);
  [spotifyPreviousButton, spotifyToggleButton, spotifyNextButton, spotifyStopButton]
    .forEach((button) => { button.disabled = !controlsEnabled; });
  spotifySettingStatus.textContent = payload.connected
    ? "Conectado"
    : payload.configured ? "Pendiente de conexion" : "Sin configurar";
  spotifyConnectButton.disabled = !payload.configured || payload.connected;
  spotifyDisconnectButton.disabled = !payload.connected;
  spotifyActivatePlayerButton.disabled = !payload.connected;
  if (payload.connected) {
    ensureSpotifyWebPlayer();
  } else {
    disconnectSpotifyWebPlayer();
  }
  syncSpotifyPolling(isPlaying);
}

function setSpotifyLocalPlayerStatus(message) {
  spotifyLocalPlayerStatus.textContent = message;
}

function ensureSpotifyWebPlayer() {
  if (!spotifyStatus.connected || spotifyWebPlayer || spotifyWebPlayerLoading) return;
  spotifyWebPlayerLoading = true;
  setSpotifyLocalPlayerStatus("Preparando reproductor local...");
  const startPlayer = () => {
    if (!window.Spotify || spotifyWebPlayer) {
      spotifyWebPlayerLoading = false;
      return;
    }
    spotifyWebPlayer = new window.Spotify.Player({
      name: "UGAssistant",
      getOAuthToken: async (callback) => {
        try {
          const response = await fetch("/api/spotify/web-player/token");
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.detail || "Spotify token failed");
          callback(payload.access_token);
        } catch (error) {
          callback("");
          setSpotifyLocalPlayerStatus("No se pudo renovar el acceso de Spotify.");
          console.error("spotify_web_player_token_failed", error);
        }
      },
      volume: 1,
      enableMediaSession: true,
    });
    spotifyWebPlayer.addListener("ready", ({ device_id: deviceId }) => {
      spotifyWebPlayerDeviceId = deviceId;
      noteSpotifyWebPlayerAvailable();
    });
    spotifyWebPlayer.addListener("not_ready", () => {
      spotifyWebPlayerDeviceId = "";
      setSpotifyLocalPlayerStatus("El reproductor local no esta disponible.");
    });
    spotifyWebPlayer.addListener("player_state_changed", () => {
      loadSpotifyStatus().catch((error) => console.error("spotify_web_player_state_failed", error));
    });
    spotifyWebPlayer.addListener("initialization_error", ({ message }) => {
      setSpotifyLocalPlayerStatus("Este navegador no admite el reproductor local.");
      console.error("spotify_web_player_initialization_failed", message);
    });
    spotifyWebPlayer.addListener("authentication_error", ({ message }) => {
      setSpotifyLocalPlayerStatus("Desconecta y conecta Spotify para actualizar los permisos.");
      console.error("spotify_web_player_authentication_failed", message);
    });
    spotifyWebPlayer.addListener("account_error", ({ message }) => {
      setSpotifyLocalPlayerStatus("El reproductor local requiere Spotify Premium.");
      console.error("spotify_web_player_account_failed", message);
    });
    spotifyWebPlayer.connect().then((connected) => {
      spotifyWebPlayerLoading = false;
      if (!connected) setSpotifyLocalPlayerStatus("No se pudo iniciar el reproductor local.");
    }).catch((error) => {
      spotifyWebPlayerLoading = false;
      setSpotifyLocalPlayerStatus("No se pudo iniciar el reproductor local.");
      console.error("spotify_web_player_connect_failed", error);
    });
  };
  if (window.Spotify) {
    startPlayer();
    return;
  }
  window.onSpotifyWebPlaybackSDKReady = startPlayer;
  const script = document.createElement("script");
  script.src = "https://sdk.scdn.co/spotify-player.js";
  script.async = true;
  script.onerror = () => {
    spotifyWebPlayerLoading = false;
    setSpotifyLocalPlayerStatus("No se pudo cargar el reproductor local de Spotify.");
  };
  document.head.appendChild(script);
}

async function noteSpotifyWebPlayerAvailable() {
  try {
    const response = await fetch("/api/spotify/web-player/pending", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Spotify local player pending failed");
    setSpotifyLocalPlayerStatus("Pulsa Activar reproductor local para permitir el audio.");
  } catch (error) {
    setSpotifyLocalPlayerStatus("No se pudo preparar el reproductor local.");
    console.error("spotify_web_player_pending_failed", error);
  }
}

async function registerSpotifyWebPlayer(deviceId) {
  try {
    const response = await fetch("/api/spotify/web-player/device", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: deviceId }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Spotify device registration failed");
    setSpotifyLocalPlayerStatus("Reproductor local listo.");
    applySpotifyStatus(payload);
    return true;
  } catch (error) {
    setSpotifyLocalPlayerStatus("No se pudo registrar el reproductor local.");
    console.error("spotify_web_player_registration_failed", error);
    return false;
  }
}

function activateSpotifyWebPlayer() {
  if (!spotifyWebPlayer) {
    ensureSpotifyWebPlayer();
    return;
  }
  if (!spotifyWebPlayerDeviceId) {
    setSpotifyLocalPlayerStatus("El reproductor local aun se esta preparando.");
    return;
  }
  spotifyWebPlayer.activateElement().then(async () => {
    if (await registerSpotifyWebPlayer(spotifyWebPlayerDeviceId)) {
      spotifyWebPlayerActivated = true;
      setSpotifyLocalPlayerStatus("Reproductor local activado.");
    }
  }).catch((error) => {
    setSpotifyLocalPlayerStatus("Pulsa de nuevo para permitir el audio de Spotify.");
    console.error("spotify_web_player_activation_failed", error);
  });
}

function disconnectSpotifyWebPlayer() {
  if (spotifyWebPlayer) spotifyWebPlayer.disconnect();
  spotifyWebPlayer = null;
  spotifyWebPlayerLoading = false;
  spotifyWebPlayerActivated = false;
  spotifyWebPlayerDeviceId = "";
  setSpotifyLocalPlayerStatus("El reproductor local se prepara al conectar Spotify.");
}

function syncSpotifyPolling(isPlaying) {
  if (!isPlaying) {
    if (spotifyPollTimer !== null) {
      window.clearInterval(spotifyPollTimer);
      spotifyPollTimer = null;
    }
    return;
  }
  if (spotifyPollTimer !== null) return;
  spotifyPollTimer = window.setInterval(() => {
    loadSpotifyStatus().catch((error) => console.error("spotify_status_poll_failed", error));
  }, 2500);
}

async function loadSpotifyStatus() {
  const response = await fetch("/api/spotify");
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Spotify status failed");
  applySpotifyStatus(payload);
}

async function loadSpotifyConfiguration() {
  const response = await fetch("/api/spotify/config");
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Spotify configuration request failed");
  spotifyClientIdInput.value = payload.client_id || "";
}

async function saveSpotifyConfiguration() {
  spotifySaveButton.disabled = true;
  try {
    const response = await fetch("/api/spotify/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: spotifyClientIdInput.value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Spotify configuration failed");
    applySpotifyStatus(payload);
  } catch (error) {
    spotifySettingStatus.textContent = "Error de configuracion";
    console.error(error);
  } finally {
    spotifySaveButton.disabled = false;
  }
}

async function connectSpotify() {
  try {
    const response = await fetch("/api/spotify/connect", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Spotify connection failed");
    window.open(payload.authorization_url, "_blank", "noopener");
  } catch (error) {
    spotifySettingStatus.textContent = "Guarda el Client ID primero";
    console.error(error);
  }
}

async function controlSpotify(action) {
  try {
    const response = await fetch(`/api/spotify/control/${action}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Spotify control failed");
    applySpotifyStatus(payload);
  } catch (error) {
    spotifySettingStatus.textContent = "Error de reproduccion";
    console.error(error);
  }
}

async function disconnectSpotify() {
  try {
    const response = await fetch("/api/spotify/disconnect", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Spotify disconnect failed");
    disconnectSpotifyWebPlayer();
    applySpotifyStatus(payload);
  } catch (error) {
    spotifySettingStatus.textContent = "Error al desconectar";
    console.error(error);
  }
}

function clearConversationPanel() {
  shell.dataset.session = "false";
  renderedTurnKey = "";
  conversationTurns.replaceChildren();
}

function formatTimerRemaining(remainingSeconds) {
  const seconds = Math.max(0, Math.ceil(Number(remainingSeconds) || 0));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return [hours, minutes, remainder]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

function renderTimers(timers = activeTimers) {
  activeTimers = Array.isArray(timers) ? timers : [];
  const now = Date.now();
  const visibleTimers = activeTimers
    .map((timer) => ({
      ...timer,
      remaining: Math.max(0, Math.ceil((Number(timer.ends_at_epoch_ms) - now) / 1000)),
    }))
    .filter((timer) => timer.remaining > 0)
    .sort((left, right) => (
      Number(left.ends_at_epoch_ms) - Number(right.ends_at_epoch_ms)
      || Number(left.label) - Number(right.label)
    ));
  timerStack.hidden = visibleTimers.length === 0;
  shell.dataset.timers = visibleTimers.length ? "true" : "false";
  timerStack.replaceChildren();
  visibleTimers.forEach((timer) => {
    const item = document.createElement("div");
    item.className = "timer-chip";
    const label = document.createElement("span");
    label.textContent = `Temporizador ${timer.label}`;
    const value = document.createElement("strong");
    value.textContent = formatTimerRemaining(timer.remaining);
    item.append(label, value);
    timerStack.append(item);
  });
}

function applyAssistantStatus(payload) {
  renderTimers(payload.timers);
  const sessionFinished = payload.phase === "waiting_for_wake_word"
    && !payload.busy
    && ["completed", "ended_by_gesture", "interrupted", "cancelled"].includes(payload.detail);

  if (sessionFinished) {
    clearConversationPanel();
    return;
  }

  if (!payload.question || !payload.answer) return;
  const turnKey = `${payload.question}\u0000${payload.answer}`;
  if (turnKey === renderedTurnKey) return;
  if (!renderedTurnKey) conversationTurns.replaceChildren();
  renderedTurnKey = turnKey;
  shell.dataset.session = "true";
  for (const [label, text, className] of [
    ["TU PREGUNTA", payload.question, "turn-user"],
    ["UGASSISTANT", payload.answer, "turn-assistant"],
  ]) {
    const turn = document.createElement("article");
    turn.className = `turn ${className}`;
    const heading = document.createElement("span");
    heading.className = "turn-label";
    heading.textContent = label;
    const body = document.createElement("p");
    body.textContent = text;
    turn.append(heading, body);
    conversationTurns.append(turn);
  }
}

function setConnected(isConnected) {
  connectionDot.classList.toggle("connected", isConnected);
  connectionText.textContent = isConnected ? "WebSocket activo" : "Sin conexion";
}

function setGaze(normalizedX, normalizedY) {
  const x = Math.max(-1, Math.min(1, normalizedX));
  const y = Math.max(-1, Math.min(1, normalizedY));
  avatar.style.setProperty("--gaze-x", `${(x * 10).toFixed(2)}px`);
  avatar.style.setProperty("--gaze-y", `${(y * 8).toFixed(2)}px`);
}

function clearGaze() {
  avatar.style.removeProperty("--gaze-x");
  avatar.style.removeProperty("--gaze-y");
}

function restoreCameraGaze() {
  if (
    latestCameraStatus?.person_detected
    && Number.isFinite(latestCameraStatus.face_center_x)
    && Number.isFinite(latestCameraStatus.face_center_y)
  ) {
    setGaze(
      (latestCameraStatus.face_center_x - 0.5) * 2,
      (latestCameraStatus.face_center_y - 0.5) * 2,
    );
  } else {
    clearGaze();
  }
}

function registerActivity() {
  lastActivityAt = Date.now();
  idlePhase = "active";
  clearBoredMood();
  if (shell.dataset.state === "SLEEPING" && !automaticStateRequest) {
    automaticStateRequest = true;
    sendState("IDLE").finally(() => {
      automaticStateRequest = false;
    });
  }
}

function processPointer() {
  pointerFrameRequested = false;
  registerActivity();
  pointerActive = true;
  shell.dataset.curious = "true";
  setGaze(
    (latestPointer.x / Math.max(window.innerWidth, 1) - 0.5) * 2,
    (latestPointer.y / Math.max(window.innerHeight, 1) - 0.5) * 2,
  );
  window.clearTimeout(pointerTimer);
  pointerTimer = window.setTimeout(() => {
    pointerActive = false;
    shell.dataset.curious = "false";
    restoreCameraGaze();
  }, 900);
}

function handlePointerMove(event) {
  latestPointer = { x: event.clientX, y: event.clientY };
  if (!pointerFrameRequested) {
    pointerFrameRequested = true;
    window.requestAnimationFrame(processPointer);
  }
}

async function loadState() {
  const response = await fetch("/api/state");
  if (!response.ok) {
    throw new Error(`State request failed: ${response.status}`);
  }
  applyState(await response.json());
}

async function sendState(state) {
  const response = await fetch(`/api/state/${state}?force=true`, { method: "POST" });
  if (!response.ok) {
    applyState({ state: "ERROR", updated_at: new Date().toISOString() });
    throw new Error(`State update failed: ${response.status}`);
  }
  const payload = await response.json();
  applyState(payload);
  return payload;
}

async function shutdownSystem() {
  const confirmed = window.confirm("Cerrar UGAssistant ahora?");
  if (!confirmed) {
    return;
  }
  shutdownButton.disabled = true;
  connectionText.textContent = "Cerrando sistema";
  try {
    const response = await fetch("/api/system/shutdown", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Shutdown failed: ${response.status}`);
    }
    settingsDialog.close();
  } catch (error) {
    connectionText.textContent = "No se pudo cerrar";
    shutdownButton.disabled = false;
    console.error(error);
  }
}

function connectStateSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/state`);

  socket.addEventListener("open", () => setConnected(true));
  socket.addEventListener("message", (event) => applyState(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    setConnected(false);
    window.setTimeout(connectStateSocket, 1500);
  });
  socket.addEventListener("error", () => setConnected(false));
}

function applyCameraStatus(payload) {
  latestCameraStatus = payload;
  const enabled = Boolean(payload.enabled);
  cameraToggle.checked = enabled;
  const selectedDevice = payload.selected_device_index;
  const selectedValue = selectedDevice === null || selectedDevice === undefined
    ? "-1"
    : String(selectedDevice);
  if (Array.from(cameraSelect.options).some((option) => option.value === selectedValue)) {
    cameraSelect.value = selectedValue;
  }
  cameraToggle.disabled = selectedValue === "-1";
  cameraPreview.hidden = true;

  if (enabled) {
    cameraStatus.textContent = payload.person_detected ? "Rostro detectado" : "Buscando rostro";
    cameraSettingStatus.textContent = payload.person_detected ? "Activa - rostro detectado" : "Activa";
    presenceText.textContent = payload.person_detected ? "Rostro detectado" : "Buscando rostro";
  } else {
    clearGestureMood();
    if (payload.detail === "no_camera_selected") {
      cameraStatus.textContent = "Ninguna";
      cameraSettingStatus.textContent = "Sin dispositivo";
    } else {
      cameraStatus.textContent = payload.detail && payload.detail !== "camera_disabled"
        ? "Error"
        : "Inactiva";
      cameraSettingStatus.textContent = cameraStatus.textContent;
    }
    presenceText.textContent = "Camara inactiva";
  }

  if (payload.person_detected || (payload.hands && payload.hands.length)) {
    registerActivity();
  }
  if (
    Array.isArray(payload.combined_gestures)
    && payload.combined_gestures.some(
      (gesture) => gesture.gesture === "POINTING_AT_MOUTH",
    )
  ) {
    clearConversationPanel();
  }
  applyDetectedMood(payload.hands, payload.combined_gestures);
  if (!pointerActive) {
    restoreCameraGaze();
  }
}

async function loadCameraStatus() {
  const response = await fetch("/api/camera");
  if (!response.ok) {
    throw new Error(`Camera request failed: ${response.status}`);
  }
  applyCameraStatus(await response.json());
}

async function loadCameraDevices() {
  const response = await fetch("/api/camera/devices");
  if (!response.ok) {
    throw new Error(`Camera devices request failed: ${response.status}`);
  }
  const payload = await response.json();
  const noneOption = document.createElement("option");
  noneOption.value = "-1";
  noneOption.textContent = "Ninguna";
  cameraSelect.replaceChildren(noneOption);
  payload.devices.forEach((device) => {
    const option = document.createElement("option");
    option.value = String(device.device_index);
    option.textContent = device.name;
    cameraSelect.append(option);
  });
  cameraSelect.value = payload.selected_device_index === null
    ? "-1"
    : String(payload.selected_device_index);
  cameraToggle.disabled = cameraSelect.value === "-1";
}

async function setCameraEnabled(enabled) {
  cameraToggle.disabled = true;
  try {
    const action = enabled ? "enable" : "disable";
    const response = await fetch(`/api/camera/${action}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Camera update failed: ${response.status}`);
    }
    applyCameraStatus(payload);
  } catch (error) {
    cameraStatus.textContent = "Error";
    cameraSettingStatus.textContent = "Error";
    cameraToggle.checked = false;
    cameraPreview.hidden = true;
    cameraFeed.removeAttribute("src");
    console.error(error);
  } finally {
    cameraToggle.disabled = cameraSelect.value === "-1";
  }
}

async function selectCamera(deviceIndex) {
  cameraSelect.disabled = true;
  cameraToggle.disabled = true;
  try {
    const response = await fetch(
      `/api/camera/select/${deviceIndex}?enable=${cameraToggle.checked}`,
      { method: "POST" },
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Camera selection failed: ${response.status}`);
    }
    applyCameraStatus(payload);
  } catch (error) {
    cameraStatus.textContent = "Error";
    console.error(error);
    await loadCameraDevices();
  } finally {
    cameraSelect.disabled = false;
    cameraToggle.disabled = cameraSelect.value === "-1";
  }
}

function connectCameraSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/camera`);
  socket.addEventListener("message", (event) => applyCameraStatus(JSON.parse(event.data)));
  socket.addEventListener("close", () => window.setTimeout(connectCameraSocket, 1500));
}

function populateAudioSelect(select, devices, selectedIndex, noneLabel) {
  const noneOption = document.createElement("option");
  noneOption.value = "-1";
  noneOption.textContent = noneLabel;
  select.replaceChildren(noneOption);
  devices.forEach((device) => {
    const option = document.createElement("option");
    option.value = String(device.device_index);
    option.textContent = `${device.name}${device.is_default ? " (pred.)" : ""}`;
    const sampleRate = Number.isFinite(device.default_sample_rate)
      ? `${Math.round(device.default_sample_rate)} Hz`
      : "frecuencia desconocida";
    option.title = `${device.host_api || "PortAudio"} - ${device.channels} canales - ${sampleRate}`;
    select.append(option);
  });
  select.value = selectedIndex === null ? "-1" : String(selectedIndex);
  select.disabled = false;
  select.title = select.selectedOptions[0]?.textContent || noneLabel;
}

function applyAudioStatus(payload) {
  const inputs = Array.isArray(payload.inputs) ? payload.inputs : [];
  const outputs = Array.isArray(payload.outputs) ? payload.outputs : [];
  const signature = JSON.stringify([
    inputs.map((device) => [device.device_index, device.name, device.is_default]),
    outputs.map((device) => [device.device_index, device.name, device.is_default]),
  ]);
  if (signature !== audioInventorySignature) {
    audioInventorySignature = signature;
    populateAudioSelect(
      micSelect,
      inputs,
      payload.selected_input_index,
      "Sin microfono",
    );
    populateAudioSelect(
      speakerSelect,
      outputs,
      payload.selected_output_index,
      "Sin altavoz",
    );
  } else {
    micSelect.value = payload.selected_input_index === null
      ? "-1"
      : String(payload.selected_input_index);
    speakerSelect.value = payload.selected_output_index === null
      ? "-1"
      : String(payload.selected_output_index);
  }

  const monitoring = Boolean(payload.monitoring);
  const outputEnabled = Boolean(payload.output_enabled);
  const outputPlaying = Boolean(payload.output_playing);
  const soundDetected = Boolean(payload.sound_detected);
  const inputLevel = Number(payload.input_level) || 0;
  const threshold = Math.max(Number(payload.activation_threshold) || 0.015, 0.001);
  micToggle.checked = monitoring;
  micToggle.disabled = payload.selected_input_index === null;
  speakerToggle.checked = outputEnabled;
  speakerToggle.disabled = payload.selected_output_index === null || outputPlaying;
  speakerVolume.value = String(Math.round((Number(payload.output_volume) || 0) * 100));
  speakerVolumeValue.value = `${speakerVolume.value}%`;
  speakerVolume.disabled = payload.selected_output_index === null || outputPlaying;
  speakerSelect.disabled = outputPlaying;
  microphoneSettingStatus.textContent = soundDetected
    ? "Sonido detectado"
    : monitoring ? "Activo" : payload.selected_input_index === null ? "Sin dispositivo" : "Inactivo";
  speakerSettingStatus.textContent = payload.selected_output_index === null
    ? "Sin dispositivo"
    : outputPlaying ? "Reproduciendo" : outputEnabled ? "Activos" : "Inactivos";
  micLevelFill.style.width = `${Math.min(inputLevel / (threshold * 2), 1) * 100}%`;
  if (!inputs.length && !outputs.length) {
    audioStatus.textContent = "No disponible";
  } else if (soundDetected) {
    audioStatus.textContent = "Sonido detectado";
  } else if (monitoring) {
    audioStatus.textContent = "Escuchando";
  } else {
    audioStatus.textContent = `${inputs.length} ent. / ${outputs.length} sal.`;
  }
}

function applyTTSStatus(payload) {
  const voices = Array.isArray(payload.voices) ? payload.voices : [];
  const languages = Array.isArray(payload.languages) ? payload.languages : [];
  const signature = JSON.stringify([
    languages,
    voices.map((voice) => [voice.voice_id, voice.language, voice.available]),
  ]);
  if (signature !== ttsInventorySignature) {
    ttsInventorySignature = signature;
    ttsLanguageSelect.replaceChildren();
    languages.forEach((language) => {
      const option = document.createElement("option");
      option.value = language;
      option.textContent = languageLabel(language);
      option.disabled = !voices.some(
        (voice) => voice.language === language && voice.available,
      );
      ttsLanguageSelect.append(option);
    });
    ttsVoiceSelect.replaceChildren();
    voices.forEach((voice) => {
      const option = document.createElement("option");
      option.value = voice.voice_id;
      option.textContent = `${voice.display_name} · ${languageLabel(voice.language)}`;
      option.disabled = !voice.available;
      ttsVoiceSelect.append(option);
    });
  }
  ttsLanguageSelect.value = payload.selected_language || "";
  ttsVoiceSelect.value = payload.selected_voice_id || "";
  ttsLanguageSelect.disabled = Boolean(payload.busy) || !languages.length;
  ttsVoiceSelect.disabled = Boolean(payload.busy)
    || !voices.some((voice) => voice.available);
  const ratePercent = Math.round((Number(payload.speech_rate) || 0.85) * 100);
  speechRate.value = String(ratePercent);
  speechRateValue.value = `${ratePercent}%`;
  speechRate.disabled = Boolean(payload.busy)
    || !voices.some((voice) => voice.available);
  if (payload.phase === "playing") {
    startLipSync(payload.mouth_levels, Number(payload.mouth_cue_interval_ms));
  } else if (!payload.busy) {
    stopLipSync();
  }
}

async function loadAudioDevices() {
  const response = await fetch("/api/audio/devices");
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `Audio devices request failed: ${response.status}`);
  }
  applyAudioStatus(payload);
}

async function loadTTSStatus() {
  const response = await fetch("/api/tts");
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `TTS status failed: ${response.status}`);
  }
  applyTTSStatus(payload);
}

async function selectAudioDevice(kind, deviceIndex) {
  const select = kind === "input" ? micSelect : speakerSelect;
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
    audioStatus.textContent = "Error";
    console.error(error);
    await loadAudioDevices();
  } finally {
    select.disabled = false;
  }
}

async function setAudioMonitoring(enabled) {
  micToggle.disabled = true;
  try {
    const action = enabled ? "enable" : "disable";
    const response = await fetch(`/api/audio/${action}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Audio monitor failed: ${response.status}`);
    }
    applyAudioStatus(payload);
  } catch (error) {
    audioStatus.textContent = "Error";
    console.error(error);
  } finally {
    micToggle.disabled = micSelect.value === "-1";
  }
}

async function setAudioOutputEnabled(enabled) {
  speakerToggle.disabled = true;
  try {
    const action = enabled ? "enable" : "disable";
    const response = await fetch(`/api/audio/output/${action}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Audio output update failed: ${response.status}`);
    }
    applyAudioStatus(payload);
  } catch (error) {
    audioStatus.textContent = "Error";
    speakerSettingStatus.textContent = "Error";
    console.error(error);
    await loadAudioDevices();
  } finally {
    speakerToggle.disabled = speakerSelect.value === "-1";
  }
}

async function setAudioOutputVolume(volumePercent) {
  speakerVolume.disabled = true;
  try {
    const response = await fetch(
      `/api/audio/output/volume/${volumePercent}`,
      { method: "POST" },
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Output volume failed: ${response.status}`);
    }
    applyAudioStatus(payload);
  } catch (error) {
    audioStatus.textContent = "Error";
    console.error(error);
    await loadAudioDevices();
  }
}

async function setTTSLanguage(language) {
  ttsLanguageSelect.disabled = true;
  try {
    const response = await fetch(
      `/api/tts/language/${encodeURIComponent(language)}`,
      { method: "POST" },
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `TTS language failed: ${response.status}`);
    }
    applyTTSStatus(payload);
  } catch (error) {
    speakerSettingStatus.textContent = "Error de voz";
    console.error(error);
    await loadTTSStatus();
  }
}

async function setTTSVoice(voiceId) {
  ttsVoiceSelect.disabled = true;
  try {
    const response = await fetch(
      `/api/tts/select/${encodeURIComponent(voiceId)}`,
      { method: "POST" },
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `TTS voice failed: ${response.status}`);
    }
    applyTTSStatus(payload);
  } catch (error) {
    speakerSettingStatus.textContent = "Error de voz";
    console.error(error);
    await loadTTSStatus();
  }
}

async function setTTSSpeechRate(ratePercent) {
  speechRate.disabled = true;
  try {
    const response = await fetch(`/api/tts/speed/${ratePercent}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `TTS speed failed: ${response.status}`);
    }
    applyTTSStatus(payload);
  } catch (error) {
    speakerSettingStatus.textContent = "Error de velocidad";
    console.error(error);
    await loadTTSStatus();
  }
}

function connectAudioSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/audio`);
  socket.addEventListener("message", (event) => applyAudioStatus(JSON.parse(event.data)));
  socket.addEventListener("close", () => window.setTimeout(connectAudioSocket, 1500));
}

function connectTTSSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/tts`);
  socket.addEventListener("message", (event) => applyTTSStatus(JSON.parse(event.data)));
  socket.addEventListener("close", () => window.setTimeout(connectTTSSocket, 1500));
}

function connectAssistantSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/assistant`);
  socket.addEventListener("message", (event) => applyAssistantStatus(JSON.parse(event.data)));
  socket.addEventListener("close", () => window.setTimeout(connectAssistantSocket, 1500));
}

function connectSpotifySocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/spotify`);
  socket.addEventListener("message", (event) => applySpotifyStatus(JSON.parse(event.data)));
  socket.addEventListener("close", () => window.setTimeout(connectSpotifySocket, 1500));
}

function updateIdleCycle() {
  const state = shell.dataset.state;
  if (state !== "IDLE" && state !== "SLEEPING") {
    lastActivityAt = Date.now();
    idlePhase = "active";
    clearBoredMood();
    return;
  }

  const elapsed = Date.now() - lastActivityAt;
  if (elapsed >= IDLE_WAKE_AFTER_MS && idlePhase !== "wake") {
    idlePhase = "wake";
    shell.dataset.mood = "";
    automaticStateRequest = true;
    sendState("IDLE").finally(() => {
      automaticStateRequest = false;
      lastActivityAt = Date.now();
      idlePhase = "active";
    });
  } else if (elapsed >= IDLE_SLEEP_AFTER_MS && idlePhase !== "sleep") {
    idlePhase = "sleep";
    shell.dataset.mood = "";
    automaticStateRequest = true;
    sendState("SLEEPING").finally(() => {
      automaticStateRequest = false;
    });
  } else if (elapsed >= IDLE_BORED_AFTER_MS && idlePhase === "active") {
    idlePhase = "bored";
    shell.dataset.mood = "bored";
  }
}

stateButtons.forEach((button) => {
  button.addEventListener("click", () => {
    registerActivity();
    sendState(button.dataset.nextState).catch(() => {});
  });
});

cameraToggle.addEventListener("change", () => {
  registerActivity();
  setCameraEnabled(cameraToggle.checked);
});

cameraSelect.addEventListener("change", () => {
  registerActivity();
  selectCamera(cameraSelect.value);
});

micSelect.addEventListener("change", () => {
  registerActivity();
  selectAudioDevice("input", micSelect.value);
});

speakerSelect.addEventListener("change", () => {
  registerActivity();
  selectAudioDevice("output", speakerSelect.value);
});

micToggle.addEventListener("change", () => {
  registerActivity();
  setAudioMonitoring(micToggle.checked);
});

speakerToggle.addEventListener("change", () => {
  registerActivity();
  setAudioOutputEnabled(speakerToggle.checked);
});

speakerVolume.addEventListener("input", () => {
  speakerVolumeValue.value = `${speakerVolume.value}%`;
});

speakerVolume.addEventListener("change", () => {
  registerActivity();
  setAudioOutputVolume(speakerVolume.value);
});

speechRate.addEventListener("input", () => {
  speechRateValue.value = `${speechRate.value}%`;
});

speechRate.addEventListener("change", () => {
  registerActivity();
  setTTSSpeechRate(speechRate.value);
});

ttsLanguageSelect.addEventListener("change", () => {
  registerActivity();
  setTTSLanguage(ttsLanguageSelect.value);
});

ttsVoiceSelect.addEventListener("change", () => {
  registerActivity();
  setTTSVoice(ttsVoiceSelect.value);
});

settingsButton.addEventListener("click", () => {
  registerActivity();
  settingsDialog.showModal();
});

settingsDialog.addEventListener("click", (event) => {
  if (event.target === settingsDialog) {
    settingsDialog.close();
  }
});

shutdownButton.addEventListener("click", shutdownSystem);
saveProfileButton.addEventListener("click", saveAssistantProfile);
spotifySaveButton.addEventListener("click", saveSpotifyConfiguration);
spotifyConnectButton.addEventListener("click", connectSpotify);
spotifyDisconnectButton.addEventListener("click", disconnectSpotify);
spotifyActivatePlayerButton.addEventListener("click", activateSpotifyWebPlayer);
spotifyPreviousButton.addEventListener("click", () => controlSpotify("previous"));
spotifyToggleButton.addEventListener("click", () => {
  controlSpotify(spotifyStatus.playback?.is_playing ? "pause" : "resume");
});
spotifyNextButton.addEventListener("click", () => controlSpotify("next"));
spotifyStopButton.addEventListener("click", () => controlSpotify("pause"));

window.addEventListener("pointermove", handlePointerMove, { passive: true });
window.addEventListener("pointerdown", () => {
  registerActivity();
  if (!spotifyWebPlayerActivated) activateSpotifyWebPlayer();
}, { passive: true });
window.addEventListener("keydown", registerActivity);
stage.addEventListener("touchstart", registerActivity, { passive: true });

loadState().catch(() => applyState({ state: "ERROR" }));
loadCameraStatus().catch(() => {
  cameraStatus.textContent = "No disponible";
});
loadCameraDevices().catch(() => {
  cameraSelect.disabled = true;
});
loadAudioDevices().catch((error) => {
  micSelect.disabled = true;
  speakerSelect.disabled = true;
  audioStatus.textContent = "No disponible";
  console.error(error);
});
loadTTSStatus().catch((error) => {
  ttsLanguageSelect.disabled = true;
  console.error(error);
});
loadAssistantProfile().catch((error) => console.error(error));
loadSpotifyConfiguration().catch((error) => console.error(error));
loadSpotifyStatus().catch((error) => console.error(error));
updateLocalClock();
window.setInterval(updateLocalClock, 1000);
window.setInterval(() => renderTimers(), 1000);
connectStateSocket();
connectCameraSocket();
connectAudioSocket();
connectTTSSocket();
connectAssistantSocket();
connectSpotifySocket();
window.setInterval(updateIdleCycle, 1000);
