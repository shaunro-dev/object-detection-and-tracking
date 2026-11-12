/**
 * AI-Based Robot Remote Detection and Retrieval System
 * Unified Dashboard Controller — Powered by Flask + YOLO Backend
 *
 * All detection data comes from Python via Socket.IO.
 * The camera feed is a live MJPEG stream from Python/OpenCV.
 * No simulation, no COCO-SSD browser model needed.
 */

// ─────────────────────────────────────────────
//  STATE
// ─────────────────────────────────────────────
const state = {
  connected: false,
  detected:  false,
  label:     '--',
  confidence: 0,
  position:  '--',
  direction: '---',
  speed:     0,
  fps:       0,
  mode:      'ALL',
  target:    'all',
  objectsCount: 0,
  status:    'Initializing...',
  customModels: [],
  logs: []
};

// ─────────────────────────────────────────────
//  DOM ELEMENTS
// ─────────────────────────────────────────────
const el = {
  systemTime:       document.getElementById('system-time'),
  connectionPulse:  document.getElementById('connection-pulse'),
  connectionStatus: document.getElementById('connection-status'),
  robotStatusBadge: document.getElementById('robot-status-badge'),
  objectDetected:   document.getElementById('object-detected-text'),
  objectName:       document.getElementById('object-name-text'),
  objectPosition:   document.getElementById('object-position-text'),
  objectDistance:   document.getElementById('object-distance-text'),  // now shows confidence
  directionText:    document.getElementById('direction-text'),
  speedText:        document.getElementById('speed-text'),
  fpsText:          document.getElementById('fps-text'),
  currentAction:    document.getElementById('current-action-text'),
  actionBox:        document.getElementById('action-box-container'),
  hudObjectTag:     document.getElementById('hud-object-tag'),
  hudFpsVal:        document.getElementById('hud-fps-val'),
  terminalBody:     document.getElementById('terminal-body-content'),
  targetSelect:     document.getElementById('target-object-select'),
  customTargetInput: document.getElementById('custom-target-input'),
  btnApplyTarget:   document.getElementById('btn-apply-target'),
  btnExport:        document.getElementById('btn-export'),
  btnClearLogs:     document.getElementById('btn-clear-logs'),
  objCountVal:      document.getElementById('obj-count-val'),
  modeVal:          document.getElementById('mode-val'),
  modelPills:       document.getElementById('model-pills-container'),
  cameraStream:     document.getElementById('camera-stream'),
  cameraSourceLabel: document.getElementById('camera-source-label'),
  padUp:            document.getElementById('pad-up'),
  padDown:          document.getElementById('pad-down'),
  padLeft:          document.getElementById('pad-left'),
  padRight:         document.getElementById('pad-right'),
  padStop:          document.getElementById('pad-stop'),
};

// ─────────────────────────────────────────────
//  CLOCK
// ─────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  el.systemTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
setInterval(updateClock, 1000);
updateClock();

// ─────────────────────────────────────────────
//  LOGGING UTILITY
// ─────────────────────────────────────────────
function addLog(type, message) {
  const timestamp = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const logItem = { timestamp, type, message };
  state.logs.push(logItem);

  const row = document.createElement('div');
  row.className = 'log-entry';

  const tSpan = document.createElement('span');
  tSpan.className = 'log-time';
  tSpan.textContent = `[${timestamp}]`;

  const tagSpan = document.createElement('span');
  tagSpan.className = `log-tag ${type.toLowerCase()}`;
  tagSpan.textContent = `[${type}]`;

  const msgSpan = document.createElement('span');
  msgSpan.className = 'log-msg';
  msgSpan.textContent = message;

  row.appendChild(tSpan);
  row.appendChild(tagSpan);
  row.appendChild(msgSpan);
  el.terminalBody.appendChild(row);
  el.terminalBody.scrollTop = el.terminalBody.scrollHeight;

  // Cap at 500 log entries
  while (el.terminalBody.children.length > 500) {
    el.terminalBody.removeChild(el.terminalBody.firstChild);
  }
}

// ─────────────────────────────────────────────
//  EXPORT LOGS
// ─────────────────────────────────────────────
function exportLogs() {
  if (state.logs.length === 0) {
    addLog('WARNING', 'No logs available to export.');
    return;
  }
  let txt = `====================================================================\n`;
  txt += `  AI ROBOT DETECTION SYSTEM — EVENT LOGS\n`;
  txt += `  Generated: ${new Date().toLocaleString()}\n`;
  txt += `====================================================================\n\n`;
  state.logs.forEach(l => {
    txt += `[${l.timestamp}] [${l.type.toUpperCase()}] ${l.message}\n`;
  });
  const blob = new Blob([txt], { type: 'text/plain;charset=utf-8' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `robot_detection_logs_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  addLog('INFO', 'Event logs exported successfully.');
}

// ─────────────────────────────────────────────
//  UI UPDATE — called on every detection event
// ─────────────────────────────────────────────
let lastLabel    = null;
let lastDetected = null;

function updateUI(data) {
  // Connection indicator handled separately

  // Mode badge
  const modeMap = { ALL: 'searching', CUSTOM: 'picking', FOCUS: 'moving' };
  const badgeClass = modeMap[data.mode] || 'idle';
  el.robotStatusBadge.className = `status-badge status-${badgeClass}`;
  el.robotStatusBadge.textContent = data.mode === 'ALL' ? 'All Objects'
    : data.mode === 'CUSTOM' ? `Custom: ${data.label}`
    : `Focus: ${data.target}`;

  // Detection status
  el.objectDetected.innerHTML = data.detected
    ? `<span style="color:var(--accent-green);font-weight:700;"><i class="fa-solid fa-check-circle"></i> YES</span>`
    : `<span style="color:var(--text-muted);"><i class="fa-solid fa-circle-xmark"></i> NO</span>`;

  el.objectName.textContent     = data.detected ? data.label        : '--';
  el.objectPosition.textContent = data.detected ? data.position     : '--';
  el.objectDistance.textContent = data.detected ? `${data.confidence}%` : '--%';
  el.directionText.textContent  = data.detected ? data.direction    : '---';
  el.speedText.textContent      = data.detected ? `${data.speed} px/s` : '-- px/s';

  // FPS
  const fpsColor = data.fps > 22 ? 'var(--accent-green)' : data.fps > 13 ? 'var(--accent-amber)' : 'var(--accent-red)';
  el.fpsText.style.color    = fpsColor;
  el.fpsText.textContent    = `${data.fps} fps`;
  el.hudFpsVal.textContent  = `FPS: ${data.fps}`;

  // Action text
  el.currentAction.textContent = data.status;
  if (data.detected) {
    el.actionBox.classList.add('action-active');
  } else {
    el.actionBox.classList.remove('action-active');
  }

  // HUD overlay
  el.hudObjectTag.textContent = data.detected
    ? `TARGET: ${data.label.toUpperCase()} [${data.confidence}%]`
    : 'TARGET: SEARCHING...';

  // Counters
  el.objCountVal.textContent = data.objects_count;
  el.modeVal.textContent     = data.mode;

  // Custom model pills
  if (data.custom_models && data.custom_models.join(',') !== state.customModels.join(',')) {
    state.customModels = data.custom_models;
    renderModelPills(data.custom_models);
    populateCustomModelOptions(data.custom_models);
  }

  // Log on detection change
  if (data.detected !== lastDetected || data.label !== lastLabel) {
    lastDetected = data.detected;
    lastLabel    = data.label;
    if (data.detected) {
      addLog('DETECT', `Object detected: '${data.label}' @ ${data.confidence}% confidence — Position: ${data.position}`);
    }
  }
}

// ─────────────────────────────────────────────
//  MODEL PILLS — show loaded custom models
// ─────────────────────────────────────────────
function renderModelPills(models) {
  el.modelPills.innerHTML = '';
  if (models.length === 0) {
    el.modelPills.innerHTML = '<span style="font-size:0.75rem;color:var(--text-muted);">No custom models loaded</span>';
    return;
  }
  models.forEach(name => {
    const pill = document.createElement('span');
    pill.style.cssText = `
      background: rgba(0,255,255,0.08);
      border: 1px solid rgba(0,255,255,0.3);
      color: var(--accent-cyan);
      font-size: 0.72rem;
      font-weight: 600;
      padding: 0.2rem 0.6rem;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      cursor: pointer;
    `;
    pill.title = `Click to focus on ${name}`;
    pill.textContent = `✦ ${name}`;
    pill.addEventListener('click', () => {
      el.targetSelect.value = name.toLowerCase();
      applyTarget();
    });
    el.modelPills.appendChild(pill);
  });
}

function populateCustomModelOptions(models) {
  const group = document.getElementById('custom-models-group');
  group.innerHTML = '';
  models.forEach(name => {
    const opt = document.createElement('option');
    opt.value       = name.toLowerCase();
    opt.textContent = `🎯 ${name.charAt(0).toUpperCase() + name.slice(1)} (Custom Model)`;
    group.appendChild(opt);
  });
}

// ─────────────────────────────────────────────
//  APPLY TARGET
// ─────────────────────────────────────────────
function applyTarget() {
  let target = el.customTargetInput.value.trim().toLowerCase();
  if (!target) {
    target = el.targetSelect.value;
  }
  socket.emit('set_target', { target });
  addLog('INFO', `Target updated → '${target}'`);
}

// ─────────────────────────────────────────────
//  SOCKET.IO CONNECTION
// ─────────────────────────────────────────────
const socket = io();

socket.on('connect', () => {
  state.connected = true;
  el.connectionPulse.classList.remove('disconnected');
  el.connectionStatus.textContent = 'YOLO SERVER LIVE';
  el.cameraSourceLabel.textContent = 'YOLO LIVE';
  addLog('INFO', 'Connected to YOLO detection server.');
  addLog('INFO', 'Live camera feed streaming from Python/OpenCV.');
});

socket.on('disconnect', () => {
  state.connected = false;
  el.connectionPulse.classList.add('disconnected');
  el.connectionStatus.textContent = 'DISCONNECTED';
  addLog('WARNING', 'Lost connection to detection server.');
});

socket.on('detection_update', (data) => {
  updateUI(data);
});

socket.on('target_changed', (data) => {
  addLog('ACTION', `Detection mode changed → ${data.mode}: '${data.target}'`);
  el.currentAction.textContent = `Switched to ${data.mode} mode — target: ${data.target}`;
});

socket.on('error', (data) => {
  addLog('ERROR', data.msg || 'Server error.');
});

// ─────────────────────────────────────────────
//  CAMERA FEED ERROR HANDLING
// ─────────────────────────────────────────────
el.cameraStream.addEventListener('error', () => {
  addLog('ERROR', 'Camera stream error. Retrying in 3 seconds...');
  setTimeout(() => {
    el.cameraStream.src = '/video_feed?' + Date.now();
  }, 3000);
});

// ─────────────────────────────────────────────
//  MANUAL OVERRIDE CONTROL PAD
// ─────────────────────────────────────────────
function sendManualCommand(cmd) {
  if (!state.connected) {
    addLog('WARNING', `Offline. Cannot send manual command: ${cmd}`);
    return;
  }
  socket.emit('manual_command', { command: cmd });
  addLog('ACTION', `Manual command sent: ${cmd}`);
}

// Click events for DOM buttons
if (el.padUp) el.padUp.addEventListener('click', () => sendManualCommand('MOVE_FORWARD'));
if (el.padDown) el.padDown.addEventListener('click', () => sendManualCommand('MOVE_BACKWARD'));
if (el.padLeft) el.padLeft.addEventListener('click', () => sendManualCommand('STEER_LEFT'));
if (el.padRight) el.padRight.addEventListener('click', () => sendManualCommand('STEER_RIGHT'));
if (el.padStop) el.padStop.addEventListener('click', () => sendManualCommand('EMERGENCY_STOP'));

// Keyboard controls
document.addEventListener('keydown', (e) => {
  // If the user is typing in the input field, don't trigger movement controls
  if (document.activeElement === el.customTargetInput) {
    return;
  }

  let buttonEl = null;
  let command = null;

  switch (e.key.toLowerCase()) {
    case 'arrowup':
    case 'w':
      buttonEl = el.padUp;
      command = 'MOVE_FORWARD';
      break;
    case 'arrowdown':
    case 's':
      buttonEl = el.padDown;
      command = 'MOVE_BACKWARD';
      break;
    case 'arrowleft':
    case 'a':
      buttonEl = el.padLeft;
      command = 'STEER_LEFT';
      break;
    case 'arrowright':
    case 'd':
      buttonEl = el.padRight;
      command = 'STEER_RIGHT';
      break;
    case ' ':
    case 'escape':
    case 'x':
      buttonEl = el.padStop;
      command = 'EMERGENCY_STOP';
      e.preventDefault(); // prevent scrolling with spacebar
      break;
  }

  if (command && buttonEl) {
    buttonEl.classList.add('active-highlight');
    sendManualCommand(command);
    setTimeout(() => buttonEl.classList.remove('active-highlight'), 150);
  }
});

// ─────────────────────────────────────────────
//  EVENT LISTENERS
// ─────────────────────────────────────────────
el.btnApplyTarget.addEventListener('click', applyTarget);

el.targetSelect.addEventListener('change', () => {
  if (el.targetSelect.value !== "") {
    el.customTargetInput.value = "";
    const placeholderOpt = document.getElementById('opt-custom-placeholder');
    if (placeholderOpt) placeholderOpt.style.display = 'none';
  }
  applyTarget();
});

el.customTargetInput.addEventListener('input', () => {
  const val = el.customTargetInput.value.trim();
  const placeholderOpt = document.getElementById('opt-custom-placeholder');
  if (placeholderOpt) {
    if (val) {
      placeholderOpt.textContent = `✏️ Custom: "${val}"`;
      placeholderOpt.style.display = 'block';
      el.targetSelect.value = "";
    } else {
      placeholderOpt.style.display = 'none';
      el.targetSelect.value = "all";
    }
  }
});

el.customTargetInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    applyTarget();
  }
});

el.btnExport.addEventListener('click', exportLogs);

el.btnClearLogs.addEventListener('click', () => {
  el.terminalBody.innerHTML = '';
  state.logs = [];
  addLog('INFO', 'Logs cleared.');
});

// Socket Events
socket.on('log_event', (data) => {
  addLog(data.type, data.message);
});

// ─────────────────────────────────────────────
//  INIT
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  addLog('INFO', 'Dashboard initialized. Connecting to YOLO detection server...');
  addLog('INFO', 'Camera feed will appear once Python server starts.');

  // Fetch available models from server on load
  fetch('/api/models')
    .then(r => r.json())
    .then(data => {
      if (data.custom_models && data.custom_models.length > 0) {
        renderModelPills(data.custom_models);
        populateCustomModelOptions(data.custom_models);
        addLog('DETECT', `Custom models available: ${data.custom_models.join(', ')}`);
      }
    })
    .catch(() => {
      addLog('WARNING', 'Could not fetch model list (server may still be loading).');
    });
});
