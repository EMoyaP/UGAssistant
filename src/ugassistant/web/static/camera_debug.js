const cameraFeed = document.querySelector("#cameraFeed");
const emptyState = document.querySelector("#emptyState");
const monitorLabel = document.querySelector("#monitorLabel");
const connection = document.querySelector(".connection");
const connectionText = document.querySelector("#connectionText");
const assistantState = document.querySelector("#assistantState");
const deviceName = document.querySelector("#deviceName");
const captureStatus = document.querySelector("#captureStatus");
const resolution = document.querySelector("#resolution");
const sequence = document.querySelector("#sequence");
const faceDetected = document.querySelector("#faceDetected");
const faceX = document.querySelector("#faceX");
const faceY = document.querySelector("#faceY");
const faceLandmarks = document.querySelector("#faceLandmarks");
const handCount = document.querySelector("#handCount");
const fingerCount = document.querySelector("#fingerCount");
const perHandFingerCount = document.querySelector("#perHandFingerCount");
const handLandmarks = document.querySelector("#handLandmarks");
const handGesture = document.querySelector("#handGesture");
const handedness = document.querySelector("#handedness");
const combinedGesture = document.querySelector("#combinedGesture");

let devices = new Map();
let reconnectDelay = 1000;

function formatCoordinate(value) {
  return Number.isFinite(value) ? Number(value).toFixed(3) : "--";
}

function applyState(payload) {
  assistantState.textContent = payload.state || "--";
}

function applyCameraStatus(payload) {
  const selectedIndex = payload.selected_device_index;
  const enabled = Boolean(payload.enabled);
  const hasFace = Boolean(payload.person_detected);
  const hands = Array.isArray(payload.hands) ? payload.hands : [];
  const facePoints = payload.face_landmarks
    ? Object.values(payload.face_landmarks)
    : [];
  const combined = Array.isArray(payload.combined_gestures)
    ? payload.combined_gestures
    : [];

  deviceName.textContent = selectedIndex === null
    ? "Ninguno"
    : devices.get(Number(selectedIndex)) || `Camara ${selectedIndex}`;
  captureStatus.textContent = enabled ? "Activa" : "Inactiva";
  resolution.textContent = payload.width && payload.height
    ? `${payload.width} x ${payload.height}`
    : "--";
  sequence.textContent = String(payload.sequence || 0);
  faceDetected.textContent = hasFace ? "Si" : "No";
  faceX.textContent = formatCoordinate(payload.face_center_x);
  faceY.textContent = formatCoordinate(payload.face_center_y);
  faceLandmarks.textContent = String(facePoints.length);
  handCount.textContent = String(hands.length);
  fingerCount.textContent = Number.isInteger(payload.finger_count)
    ? String(payload.finger_count)
    : "--";
  perHandFingerCount.textContent = hands.length
    ? hands.map((hand) => String(hand.finger_count)).join(" + ")
    : "--";
  handLandmarks.textContent = String(
    hands.reduce((total, hand) => total + (hand.landmarks?.length || 0), 0),
  );
  handGesture.textContent = hands.length
    ? hands.map((hand) => hand.gesture).join(" / ")
    : "--";
  handedness.textContent = hands.length
    ? hands.map((hand) => hand.handedness).join(" / ")
    : "--";
  combinedGesture.textContent = combined.length
    ? combined.map((detection) => detection.gesture).join(" / ")
    : "--";

  if (!enabled) {
    monitorLabel.textContent = "CAMARA INACTIVA";
  } else if (combined.length) {
    monitorLabel.textContent = combined[0].gesture;
  } else if (hands.length) {
    monitorLabel.textContent = hasFace
      ? `ROSTRO + ${hands[0].gesture}`
      : hands[0].gesture;
  } else {
    monitorLabel.textContent = hasFace ? "ROSTRO DETECTADO" : "ANALIZANDO";
  }

  cameraFeed.hidden = !enabled;
  emptyState.hidden = enabled;
  if (enabled && !cameraFeed.getAttribute("src")) {
    cameraFeed.src = `/api/camera/stream?t=${Date.now()}`;
  } else if (!enabled) {
    cameraFeed.removeAttribute("src");
  }
}

async function loadInitialData() {
  const [devicesResponse, cameraResponse, stateResponse] = await Promise.all([
    fetch("/api/camera/devices"),
    fetch("/api/camera"),
    fetch("/api/state"),
  ]);
  if (!devicesResponse.ok || !cameraResponse.ok || !stateResponse.ok) {
    throw new Error("Initial vision data unavailable");
  }

  const devicesPayload = await devicesResponse.json();
  devices = new Map(
    devicesPayload.devices.map((device) => [Number(device.device_index), device.name]),
  );
  applyCameraStatus(await cameraResponse.json());
  applyState(await stateResponse.json());
}

function connectSocket(path, onMessage) {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}${path}`);

  socket.addEventListener("open", () => {
    connection.dataset.connected = "true";
    connectionText.textContent = "Datos en directo";
    reconnectDelay = 1000;
  });
  socket.addEventListener("message", (event) => onMessage(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    connection.dataset.connected = "false";
    connectionText.textContent = "Reconectando";
    window.setTimeout(() => connectSocket(path, onMessage), reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 1.5, 5000);
  });
  socket.addEventListener("error", () => {
    connection.dataset.connected = "false";
  });
}

loadInitialData().catch((error) => {
  connectionText.textContent = "Sin datos";
  console.error(error);
});
connectSocket("/ws/camera", applyCameraStatus);
connectSocket("/ws/state", applyState);
