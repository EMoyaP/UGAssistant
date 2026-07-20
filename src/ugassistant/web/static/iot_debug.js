const urlInput = document.querySelector("#urlInput");
const tokenInput = document.querySelector("#tokenInput");
const saveButton = document.querySelector("#saveButton");
const refreshButton = document.querySelector("#refreshButton");
const connectionStatus = document.querySelector("#connectionStatus");
const entityCount = document.querySelector("#entityCount");
const entityList = document.querySelector("#entityList");

function actionLabel(action) {
  return {
    turn_on: "Encender",
    turn_off: "Apagar",
    toggle: "Alternar",
    open: "Abrir",
    close: "Cerrar",
    stop: "Detener",
    lock: "Cerrar",
    unlock: "Abrir",
    activate: "Activar",
  }[action] || action;
}

function applyStatus(payload) {
  urlInput.value = payload.home_assistant_url || urlInput.value;
  const entities = Array.isArray(payload.entities) ? payload.entities : [];
  connectionStatus.textContent = payload.connected
    ? "Conectado en la red local"
    : payload.configured ? "No se pudo conectar" : "Sin configurar";
  connectionStatus.dataset.connected = String(Boolean(payload.connected));
  entityCount.textContent = `${entities.length} ${entities.length === 1 ? "elemento" : "elementos"}`;
  entityList.replaceChildren();
  if (!entities.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = payload.configured
      ? "No hay elementos controlables disponibles."
      : "Configura una instancia local de Home Assistant para descubrir los elementos.";
    entityList.append(empty);
    return;
  }
  entities.forEach((entity) => {
    const card = document.createElement("article");
    card.className = "entity";
    const title = document.createElement("h3");
    title.textContent = entity.name;
    const metadata = document.createElement("p");
    metadata.textContent = `${entity.domain} · ${entity.state}${entity.available ? "" : " · No disponible"}`;
    const actions = document.createElement("div");
    actions.className = "entity-actions";
    entity.actions.forEach((action) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = actionLabel(action);
      button.disabled = !entity.available;
      button.addEventListener("click", () => controlEntity(entity.entity_id, action, button));
      actions.append(button);
    });
    card.append(title, metadata, actions);
    entityList.append(card);
  });
}

async function loadConfig() {
  const response = await fetch("/api/iot/config");
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "IoT configuration failed");
  urlInput.value = payload.home_assistant_url || "";
}

async function refreshIoT() {
  refreshButton.disabled = true;
  try {
    const response = await fetch("/api/iot");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "IoT refresh failed");
    applyStatus(payload);
  } catch (error) {
    connectionStatus.textContent = "Error al consultar Home Assistant";
    connectionStatus.dataset.connected = "false";
    console.error(error);
  } finally {
    refreshButton.disabled = false;
  }
}

async function saveConfig() {
  saveButton.disabled = true;
  try {
    const response = await fetch("/api/iot/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ home_assistant_url: urlInput.value, token: tokenInput.value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "IoT configuration failed");
    tokenInput.value = "";
    applyStatus(payload);
  } catch (error) {
    connectionStatus.textContent = "Revisa la URL y el token";
    connectionStatus.dataset.connected = "false";
    console.error(error);
  } finally {
    saveButton.disabled = false;
  }
}

async function controlEntity(entityId, action, button) {
  if (["unlock", "open"].includes(action) && !window.confirm(`Confirmar ${actionLabel(action).toLowerCase()} este elemento?`)) return;
  button.disabled = true;
  try {
    const response = await fetch(`/api/iot/entities/${encodeURIComponent(entityId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "IoT action failed");
    applyStatus(payload);
  } catch (error) {
    connectionStatus.textContent = "La acción no se pudo completar";
    console.error(error);
  } finally {
    button.disabled = false;
  }
}

saveButton.addEventListener("click", saveConfig);
refreshButton.addEventListener("click", refreshIoT);
loadConfig().then(refreshIoT).catch((error) => console.error(error));
