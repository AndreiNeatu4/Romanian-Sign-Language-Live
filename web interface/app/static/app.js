/* ── State ─────────────────────────────────────────────────────────────── */
let ws              = null;
let stream          = null;
let sendLoop        = null;
let pendingResponse = false;
let mirrored        = true;   // on by default — feels natural (selfie view)
let cameraRunning   = false;
let showHands       = true;
let showFace        = true;

// Capture resolution (what gets sent to the server).
// 320x240 cuts MediaPipe work by 4x vs 640x480 with no visible accuracy loss
// for hands that are close to the camera.
const CAP_W = 320;
const CAP_H = 240;

// FPS / RTT tracking
let fpsCounter   = 0;
let fpsTimestamp = performance.now();
let lastSentAt   = 0;

// ── Landmark interpolation (Path A: smooth 60fps overlay) ──
// Server responses arrive at inference rate (~7-10fps); we glide the drawn
// landmarks toward the latest response every animation frame so the overlay
// looks smooth instead of jumping. null target = nothing received yet.
let tgtHands = null, curHands = null;
let tgtFace  = null, curFace  = null;
let lastConnections = null;
const LERP = 0.4;   // 0 = frozen, 1 = snap instantly. Higher = more responsive.

/* ── DOM refs ──────────────────────────────────────────────────────────── */
const video       = document.getElementById('video');
const overlay     = document.getElementById('overlay');
const capture     = document.getElementById('capture');
const ctx         = overlay.getContext('2d');
const capCtx      = capture.getContext('2d');
const bufBar      = document.getElementById('buffer-bar');
const bufText     = document.getElementById('buffer-text');
const predLetter  = document.getElementById('pred-letter');
const predConf    = document.getElementById('pred-confidence');
const predBox     = document.getElementById('prediction-box');
const predLetterMobile = document.getElementById('pred-letter-mobile');
const predConfMobile   = document.getElementById('pred-confidence-mobile');
const predBoxMobile    = document.getElementById('prediction-box-mobile');
const statusBadge = document.getElementById('status-badge');
const statFps     = document.getElementById('stat-fps');
const statRtt     = document.getElementById('stat-rtt');
const statHands   = document.getElementById('stat-hands');
const statMode    = document.getElementById('stat-mode');
const refSubtitle = document.getElementById('ref-subtitle');
const camSubtitle = document.getElementById('cam-subtitle');
const leftSidebar = document.getElementById('left-sidebar');
const rightSidebar = document.getElementById('right-sidebar');
const mobileOverlay = document.getElementById('mobile-overlay');
const btnLeftMenu = document.getElementById('btn-left-menu');
const btnRightMenu = document.getElementById('btn-right-menu');
const btnToggleAlphabet = document.getElementById('btn-toggle-alphabet');
const btnToggleSemne = document.getElementById('btn-toggle-semne');
const alphabetSection = document.getElementById('alphabet-section');
const semneSection = document.getElementById('semne-section');
const semneList = document.getElementById('semne-list');

capture.width  = CAP_W;
capture.height = CAP_H;

/* ── Mobile menus ──────────────────────────────────────────────────────── */
let leftMenuOpen = false;
let rightMenuOpen = false;

function isMobileLayout() {
  return window.matchMedia && window.matchMedia('(max-width: 900px)').matches;
}

function syncSidebarLayout() {
  if (!leftSidebar || !rightSidebar) return;

  if (isMobileLayout()) {
    leftSidebar.classList.add('is-mobile', 'left');
    rightSidebar.classList.add('is-mobile', 'right');
    // Ensure overlay is hidden unless a menu is open
    if (!leftMenuOpen && !rightMenuOpen && mobileOverlay) mobileOverlay.classList.add('hidden');
  } else {
    leftSidebar.classList.remove('is-mobile', 'left', 'open');
    rightSidebar.classList.remove('is-mobile', 'right', 'open');
    leftMenuOpen = false;
    rightMenuOpen = false;
    if (mobileOverlay) mobileOverlay.classList.add('hidden');
  }
}

function setMenus({ left, right }) {
  leftMenuOpen = !!left;
  rightMenuOpen = !!right;

  if (!isMobileLayout()) return;

  if (leftSidebar) leftSidebar.classList.toggle('open', leftMenuOpen);
  if (rightSidebar) rightSidebar.classList.toggle('open', rightMenuOpen);
  if (mobileOverlay) mobileOverlay.classList.toggle('hidden', !(leftMenuOpen || rightMenuOpen));
}

/* ── WebSocket ─────────────────────────────────────────────────────────── */
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    setStatus('online');
    console.log('WebSocket connected');
  };

  ws.onmessage = (evt) => {
    const rtt = performance.now() - lastSentAt;
    statRtt.textContent = `${Math.round(rtt)} ms`;
    pendingResponse = false;

    let data;
    try { data = JSON.parse(evt.data); } catch { return; }

    if (data.error) { console.warn('Server error:', data.error); return; }

    updateBuffer(data.buffer_fill, data.buffer_max);
    // Hand the landmarks to the rAF render loop instead of drawing here — this
    // is what decouples overlay smoothness from the server's response rate.
    tgtHands = data.hands || [];
    tgtFace  = data.face  || [];
    if (data.hand_connections) lastConnections = data.hand_connections;
    updatePrediction(data.prediction, data.confidence, data.top_predictions, data.frozen);
    updateStats(data.hands ? data.hands.length : 0);
    updateFPS();
  };

  ws.onerror = () => setStatus('offline');
  ws.onclose = () => {
    setStatus('offline');
    // Auto-reconnect after 3 s
    setTimeout(connectWS, 3000);
  };
}

/* ── Camera ────────────────────────────────────────────────────────────── */
async function startCamera() {
  if (cameraRunning) { stopCamera(); return; }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    // Size overlay to match rendered video
    video.addEventListener('loadedmetadata', resizeOverlay, { once: true });
    window.addEventListener('resize', resizeOverlay);
    resizeOverlay();

    cameraRunning = true;
    document.getElementById('btn-start').textContent = 'Stop Camera';
    document.getElementById('btn-start').classList.add('btn-danger');
    document.getElementById('btn-start').classList.remove('btn-primary');
    if (camSubtitle) camSubtitle.textContent = 'Detecting…';
    statMode.textContent = 'live';

    // Start send loop — fire every 33 ms so a new frame goes out
    // immediately after a response arrives rather than waiting up to 80 ms.
    // The pendingResponse guard prevents flooding regardless of this interval.
    sendLoop = setInterval(captureAndSend, 33); // ~30 fps cap
  } catch (err) {
    alert(`Camera error: ${err.message}`);
  }
}

function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (sendLoop) { clearInterval(sendLoop); sendLoop = null; }
  video.srcObject = null;
  cameraRunning = false;
  document.getElementById('btn-start').textContent = 'Start Camera';
  document.getElementById('btn-start').classList.add('btn-primary');
  document.getElementById('btn-start').classList.remove('btn-danger');
  if (camSubtitle) camSubtitle.textContent = 'Camera stopped';
  statMode.textContent = 'idle';
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width  = rect.width  || video.videoWidth  || CAP_W;
  overlay.height = rect.height || video.videoHeight || CAP_H;
}

/* ── Frame capture & send ──────────────────────────────────────────────── */
function captureAndSend() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (pendingResponse) return;                    // back-pressure
  if (!video.videoWidth) return;                  // camera not ready

  // Capture exactly the visible portion of the video (matching object-fit: cover),
  // so landmark coordinates from the server align with what is displayed.
  const vW = video.videoWidth;
  const vH = video.videoHeight;
  const cW = video.clientWidth;
  const cH = video.clientHeight;
  const scale = Math.max(cW / vW, cH / vH);
  const visW = cW / scale;   // visible region width  in video pixels
  const visH = cH / scale;   // visible region height in video pixels
  const sx   = (vW - visW) / 2;
  const sy   = (vH - visH) / 2;

  // Always draw the raw (non-mirrored) video frame.
  // drawImage reads raw pixel data — CSS transforms on the video element have no effect here.
  // The overlay's CSS scaleX(-1) automatically mirrors the drawn landmarks to match the display.
  capCtx.drawImage(video, sx, sy, visW, visH, 0, 0, CAP_W, CAP_H);

  const dataURL = capture.toDataURL('image/jpeg', 0.6);
  ws.send(dataURL);
  pendingResponse = true;
  lastSentAt = performance.now();
}

/* ── Overlay toggles ───────────────────────────────────────────────────── */
function toggleHands() {
  showHands = !showHands;
  const btn = document.getElementById('btn-hands');
  btn.classList.toggle('toggle-on', showHands);
  btn.style.opacity = showHands ? '' : '0.4';
  if (!showHands) ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function toggleFace() {
  showFace = !showFace;
  const btn = document.getElementById('btn-face');
  btn.classList.toggle('toggle-on', showFace);
  btn.style.opacity = showFace ? '' : '0.4';
}

/* ── Mirror ────────────────────────────────────────────────────────────── */
function toggleMirror() {
  mirrored = !mirrored;
  // Only flip the video element — landmark coords are flipped in drawLandmarks()
  video.style.transform = mirrored ? 'scaleX(-1)' : '';
  document.getElementById('btn-mirror').classList.toggle('mirror-on', mirrored);
}

/* ── Reset buffer ──────────────────────────────────────────────────────── */
async function resetBuffer() {
  try { await fetch('/reset', { method: 'POST' }); } catch {}
  predLetter.textContent = '—';
  predConf.textContent   = 'waiting for buffer…';
  predBox.classList.remove('has-pred');
  if (predLetterMobile) predLetterMobile.textContent = '—';
  if (predConfMobile) predConfMobile.textContent = 'waiting for buffer…';
  if (predBoxMobile) predBoxMobile.classList.remove('has-pred');
  updateBuffer(0, 30);
}

/* ── UI update helpers ─────────────────────────────────────────────────── */
function setStatus(state) {
  statusBadge.className = `badge badge-${state}`;
  statusBadge.textContent = state.charAt(0).toUpperCase() + state.slice(1);
}

function updateBuffer(fill, max) {
  const pct = max ? (fill / max) * 100 : 0;
  bufBar.style.width = `${pct}%`;
  bufText.textContent = `${fill} / ${max}`;
}

function updatePrediction(pred, conf, _top, frozen) {
  if (!pred) return;
  const suffix = frozen ? ' (held — show a hand to resume)' : ' confidence';
  const confText = `${(conf * 100).toFixed(1)} %${suffix}`;
  predLetter.textContent = pred;
  predConf.textContent   = confText;
  predBox.classList.add('has-pred');
  predBox.classList.toggle('is-frozen', !!frozen);
  if (predLetterMobile) predLetterMobile.textContent = pred;
  if (predConfMobile) predConfMobile.textContent = confText;
  if (predBoxMobile) {
    predBoxMobile.classList.add('has-pred');
    predBoxMobile.classList.toggle('is-frozen', !!frozen);
  }
}

function updateStats(handCount) {
  statHands.textContent = handCount;
}

function updateFPS() {
  fpsCounter++;
  const now = performance.now();
  const elapsed = (now - fpsTimestamp) / 1000;
  if (elapsed >= 1) {
    statFps.textContent = Math.round(fpsCounter / elapsed);
    fpsCounter   = 0;
    fpsTimestamp = now;
  }
}

/* ── Landmark drawing ──────────────────────────────────────────────────── */
// Colours
const HAND_COLORS   = ['#4f8ef7', '#7c5af7'];
const FACE_COLOR    = '#f7c34f';
const CONN_COLOR    = ['rgba(79,142,247,.6)', 'rgba(124,90,247,.6)'];
const LANDMARK_R    = 4;

function drawLandmarks(hands, face, connections) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!overlay.width) return;

  const W = overlay.width;
  const H = overlay.height;

  // When mirrored, flip x to match the CSS-flipped video element.
  // Server coords are always in non-mirrored space.
  const px = (x) => (mirrored ? 1 - x : x) * W;
  const py = (y) => y * H;

  // ── Hands ──
  if (showHands && hands) {
    hands.forEach((pts, hi) => {
      // Connections
      if (connections) {
        ctx.strokeStyle = CONN_COLOR[hi] || CONN_COLOR[0];
        ctx.lineWidth   = 2;
        connections.forEach(([a, b]) => {
          if (!pts[a] || !pts[b]) return;
          ctx.beginPath();
          ctx.moveTo(px(pts[a][0]), py(pts[a][1]));
          ctx.lineTo(px(pts[b][0]), py(pts[b][1]));
          ctx.stroke();
        });
      }

      // Dots
      pts.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(px(x), py(y), LANDMARK_R, 0, Math.PI * 2);
        ctx.fillStyle   = HAND_COLORS[hi] || HAND_COLORS[0];
        ctx.strokeStyle = '#fff';
        ctx.lineWidth   = 1.5;
        ctx.fill();
        ctx.stroke();
      });
    });
  }

  // ── Face ──
  if (showFace && face && face.length) {
    face.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(px(x), py(y), 3, 0, Math.PI * 2);
      ctx.fillStyle = FACE_COLOR;
      ctx.fill();
    });
  }
}

/* ── Smooth overlay render loop (Path A) ───────────────────────────────── */
// Interpolate each landmark toward its latest server position every animation
// frame. When the number of hands/points changes (hand enters/leaves frame)
// we snap instead of interpolating, so points don't slide in from stale spots.
function _lerpPts(cur, tgt) {
  const out = new Array(tgt.length);
  for (let i = 0; i < tgt.length; i++) {
    const c = cur[i] || tgt[i];
    out[i] = [c[0] + (tgt[i][0] - c[0]) * LERP, c[1] + (tgt[i][1] - c[1]) * LERP];
  }
  return out;
}

function _lerpHands(cur, tgt) {
  if (!tgt) return null;
  if (!cur || cur.length !== tgt.length) return tgt.map(h => h.map(p => [p[0], p[1]]));
  return tgt.map((hand, hi) =>
    (cur[hi] && cur[hi].length === hand.length) ? _lerpPts(cur[hi], hand) : hand
  );
}

function _lerpFace(cur, tgt) {
  if (!tgt) return null;
  if (!cur || cur.length !== tgt.length) return tgt.map(p => [p[0], p[1]]);
  return _lerpPts(cur, tgt);
}

function renderLoop() {
  curHands = _lerpHands(curHands, tgtHands);
  curFace  = _lerpFace(curFace, tgtFace);
  drawLandmarks(curHands, curFace, lastConnections);
  requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);

/* ── Keyboard shortcuts ────────────────────────────────────────────────── */
document.addEventListener('keydown', (e) => {
  switch (e.key.toLowerCase()) {
    case 'r': resetBuffer();   break;
    case 'm': toggleMirror();  break;
    case 's': startCamera();   break;
    case 'h': toggleHands();   break;
    case 'f': toggleFace();    break;
  }
});

/* ── Reference video ───────────────────────────────────────────────────── */
const refVideo       = document.getElementById('ref-video');
const refPlaceholder = document.getElementById('ref-placeholder');
const letterList     = document.getElementById('letter-list');

function selectLetter(item, url) {
  letterList.querySelectorAll('.letter-item').forEach(el => el.classList.remove('active'));
  if (semneList) semneList.querySelectorAll('.letter-item').forEach(el => el.classList.remove('active'));
  item.classList.add('active');
  if (refSubtitle) refSubtitle.textContent = `Letter: ${item.textContent}`;
  if (isMobileLayout()) setMenus({ left: false, right: false });
  if (url) {
    refPlaceholder.classList.add('hidden');
    refVideo.classList.remove('hidden');
    refVideo.src = url;
    refVideo.load();
    refVideo.play().catch(() => {});
  } else {
    refVideo.pause();
    refVideo.removeAttribute('src');
    refVideo.load();
    refVideo.classList.add('hidden');
    refPlaceholder.classList.remove('hidden');
    refPlaceholder.textContent = 'No reference video for this item';
  }
}

async function fetchVideoList() {
  try {
    const res  = await fetch('/videos');
    const data = await res.json();
    if (!data.videos || !data.videos.length) return;

    data.videos.forEach(({ letter, url }) => {
      const item = document.createElement('div');
      item.className   = 'letter-item';
      item.textContent = letter;
      item.addEventListener('click', () => selectLetter(item, url));
      letterList.appendChild(item);
    });
  } catch (e) {
    console.warn('Could not load video list:', e);
  }
}

/* ── Boot ──────────────────────────────────────────────────────────────── */
// Apply the default mirror state immediately on load (only the video element)
video.style.transform = 'scaleX(-1)';
document.getElementById('btn-mirror').classList.add('mirror-on');

// Sidebar category toggles (Alphabet / Semne)
function setCollapsed(btn, section, collapsed) {
  if (!btn || !section) return;
  section.classList.toggle('hidden', collapsed);
  btn.setAttribute('aria-expanded', String(!collapsed));
  const chev = btn.querySelector('.chev');
  if (chev) {
    chev.classList.toggle('chev-right', collapsed);
    chev.classList.toggle('chev-down', !collapsed);
  }
}

let alphabetCollapsed = false;
let semneCollapsed = true;

if (btnToggleAlphabet) {
  btnToggleAlphabet.addEventListener('click', () => {
    alphabetCollapsed = !alphabetCollapsed;
    setCollapsed(btnToggleAlphabet, alphabetSection, alphabetCollapsed);
  });
}
if (btnToggleSemne) {
  btnToggleSemne.addEventListener('click', () => {
    semneCollapsed = !semneCollapsed;
    setCollapsed(btnToggleSemne, semneSection, semneCollapsed);
  });
}
setCollapsed(btnToggleAlphabet, alphabetSection, alphabetCollapsed);
setCollapsed(btnToggleSemne, semneSection, semneCollapsed);

// Seed Semne items (matches Figma sidebar)
if (semneList) {
  const semne = ['Hello', 'Thank you', 'Please', 'Yes', 'No', 'Help'];
  semne.forEach((name) => {
    const item = document.createElement('div');
    item.className = 'letter-item';
    item.textContent = name;
    item.addEventListener('click', () => selectLetter(item, null));
    semneList.appendChild(item);
  });
}

// Ensure sidebars behave on mobile/desktop
syncSidebarLayout();
window.addEventListener('resize', syncSidebarLayout);

if (btnLeftMenu) {
  btnLeftMenu.addEventListener('click', () => {
    setMenus({ left: !leftMenuOpen, right: false });
  });
}
if (btnRightMenu) {
  btnRightMenu.addEventListener('click', () => {
    setMenus({ left: false, right: !rightMenuOpen });
  });
}
if (mobileOverlay) {
  mobileOverlay.addEventListener('click', () => setMenus({ left: false, right: false }));
}

fetchVideoList();
connectWS();
