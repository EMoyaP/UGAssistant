const statusText = document.querySelector("#mobileStatus");
const avatar = document.querySelector("#mobileAvatar");
const talkButton = document.querySelector("#talkButton");
const transcript = document.querySelector("#transcript");
const answer = document.querySelector("#answer");
const microphoneLabel = document.querySelector("#microphoneLabel");
const speakerLabel = document.querySelector("#speakerLabel");

const query = new URLSearchParams(window.location.search);
let stored = null;
try { stored = JSON.parse(localStorage.getItem("ugassistant-mobile") || "null"); } catch { localStorage.removeItem("ugassistant-mobile"); }
const credentials = stored || { access_id: query.get("access"), token: query.get("token"), device_id: crypto.randomUUID(), device_label: navigator.userAgent.includes("Android") ? "Android" : "Navegador" };
let recorder = null;
let audioContext = null;
let samples = [];

function setState(state, label) { avatar.className = `mobile-avatar state-${state}`; statusText.textContent = label; }
function headers() { return { "Content-Type": "audio/wav", "X-UG-Access": credentials.access_id, "X-UG-Token": credentials.token, "X-UG-Device": credentials.device_id, "X-UG-Device-Label": credentials.device_label }; }

function wavFromSamples(source, sampleRate) {
  const targetRate = 16000;
  const length = Math.floor(source.length * targetRate / sampleRate);
  const buffer = new ArrayBuffer(44 + length * 2); const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + length * 2, true); text(8, "WAVEfmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, targetRate, true); view.setUint32(28, targetRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, "data"); view.setUint32(40, length * 2, true);
  for (let index = 0; index < length; index += 1) { const position = index * sampleRate / targetRate; const before = source[Math.floor(position)] || 0; const after = source[Math.ceil(position)] || before; const value = before + (after - before) * (position - Math.floor(position)); view.setInt16(44 + index * 2, Math.max(-1, Math.min(1, value)) * 32767, true); }
  return new Blob([buffer], { type: "audio/wav" });
}

async function connect() {
  if (!window.isSecureContext || !credentials.access_id || !credentials.token) { setState("error", "Conexión segura requerida"); return; }
  const response = await fetch("/api/session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(credentials) });
  if (!response.ok) { setState("error", "Acceso denegado"); return; }
  localStorage.setItem("ugassistant-mobile", JSON.stringify(credentials)); history.replaceState({}, "", "/"); setState("idle", "Listo"); talkButton.disabled = false;
}

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const devices = await navigator.mediaDevices.enumerateDevices(); const input = devices.find((device) => device.kind === "audioinput" && device.label); microphoneLabel.textContent = input?.label || "Micrófono";
  const output = devices.find((device) => device.kind === "audiooutput" && device.label);
  microphoneLabel.textContent = input?.label || "Microfono del telefono";
  speakerLabel.textContent = output?.label || "Salida del telefono";
  audioContext = new AudioContext(); const source = audioContext.createMediaStreamSource(stream); recorder = audioContext.createScriptProcessor(4096, 1, 1); samples = [];
  recorder.onaudioprocess = (event) => samples.push(new Float32Array(event.inputBuffer.getChannelData(0))); source.connect(recorder); recorder.connect(audioContext.destination); recorder.stream = stream; setState("listening", "Escuchando"); talkButton.textContent = "Terminar";
}

async function stopRecording() {
  recorder.disconnect(); recorder.stream.getTracks().forEach((track) => track.stop()); const source = new Float32Array(samples.reduce((total, part) => total + part.length, 0)); let offset = 0; samples.forEach((part) => { source.set(part, offset); offset += part.length; }); const wav = wavFromSamples(source, audioContext.sampleRate); await audioContext.close(); recorder = null; setState("thinking", "Pensando"); talkButton.disabled = true;
  const response = await fetch("/api/ask", { method: "POST", headers: headers(), body: wav }); const payload = await response.json(); if (!response.ok) { setState("error", "No disponible"); talkButton.disabled = false; return; }
  transcript.textContent = payload.transcript; answer.textContent = payload.answer; const bytes = Uint8Array.from(atob(payload.audio_wav_base64), (character) => character.charCodeAt(0)); const audio = new Audio(URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }))); setState("speaking", "Respondiendo"); await audio.play(); await new Promise((resolve) => { audio.onended = resolve; }); setState("idle", "Listo"); talkButton.disabled = false;
}

talkButton.addEventListener("click", async () => { try { if (recorder) { talkButton.textContent = "Hablar"; await stopRecording(); } else { await startRecording(); } } catch (error) { console.error(error); setState("error", "Permiso necesario"); talkButton.disabled = false; } });
connect().catch((error) => { console.error(error); setState("error", "No disponible"); });
