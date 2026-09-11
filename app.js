/* ══════════════════════════════════════════════════════════
   ROLLCALL — on-device face attendance
   Storage: IndexedDB (classes, people, face descriptors, sessions)
   Engine : face-api.js (TinyFaceDetector + 68 landmarks + ResNet embeddings)
   ══════════════════════════════════════════════════════════ */
'use strict';

/* ─────────── CONFIG ─────────── */
const CFG = {
  MODELS: 'models',
  // squared-euclidean distance on 128-d embeddings; 0.45-0.55 is the usable band
  MATCH_DIST: 0.50,
  MIN_DETECT_SCORE: 0.55,
  INPUT_SIZE: 416,
  MAX_SAMPLES: 12,
  RECHECK_MS: 700,
  COOLDOWN_MS: 8000,
  CONSEC_HITS: 2,
  UNKNOWN_DIST: 0.62,
  THUMB: 220,
};

/* ─────────── HELPERS ─────────── */
const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
const norm = s => String(s || '').trim().replace(/\s+/g, ' ');
const key  = s => norm(s).toLowerCase();
const initials = n => norm(n).split(' ').filter(Boolean).slice(0, 2).map(w => w[0]).join('').toUpperCase() || '?';
const fmtTime = ts => new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
const fmtDate = ts => new Date(ts).toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' });

/* ─────────── STORAGE (IndexedDB) ─────────── */
const DB = (() => {
  const NAME = 'rollcall', VER = 1;
  let db = null;
  function open() {
    return new Promise((res, rej) => {
      const rq = indexedDB.open(NAME, VER);
      rq.onupgradeneeded = e => {
        const d = e.target.result;
        if (!d.objectStoreNames.contains('classes')) d.createObjectStore('classes', { keyPath: 'id' });
        if (!d.objectStoreNames.contains('people')) {
          const st = d.createObjectStore('people', { keyPath: 'id' });
          st.createIndex('classId', 'classId', { unique: false });
        }
        if (!d.objectStoreNames.contains('sessions')) {
          const st = d.createObjectStore('sessions', { keyPath: 'id' });
          st.createIndex('classId', 'classId', { unique: false });
        }
      };
      rq.onsuccess = () => { db = rq.result; res(db); };
      rq.onerror = () => rej(rq.error);
    });
  }
  const tx = (s, m = 'readonly') => db.transaction(s, m).objectStore(s);
  const wrap = rq => new Promise((res, rej) => { rq.onsuccess = () => res(rq.result); rq.onerror = () => rej(rq.error); });
  return {
    open,
    all: s => wrap(tx(s).getAll()),
    get: (s, id) => wrap(tx(s).get(id)),
    put: (s, v) => wrap(tx(s, 'readwrite').put(v)),
    del: (s, id) => wrap(tx(s, 'readwrite').delete(id)),
    byIndex: (s, i, v) => wrap(tx(s).index(i).getAll(v)),
  };
})();

/* ─────────── STATE ─────────── */
const S = {
  classes: [], people: [], cls: null,
  route: { name: 'classes' },
  engineReady: false, matcher: null,
  session: null, stream: null, loopId: null,
  paused: false, kioskTab: 'log', mirror: true,
};

/* ─────────── TOAST ─────────── */
function toast(msg, kind = '', ms = 3200) {
  const host = $('#kiosk').hidden ? $('#toastsMain') : $('#toasts');
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.innerHTML = msg;
  host.appendChild(el);
  setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 320); }, ms);
}

/* ─────────── MODAL ─────────── */
function modal(html, opts = {}) {
  const m = $('#modal'), card = $('#modalCard');
  card.className = 'modal-card' + (opts.wide ? ' wide' : '');
  card.innerHTML = html;
  m.hidden = false;
  if (opts.onMount) opts.onMount(card);
  const f = card.querySelector('[autofocus],input,select,textarea');
  if (f) setTimeout(() => f.focus(), 60);
  return card;
}
function closeModal() { $('#modal').hidden = true; $('#modalCard').innerHTML = ''; }

function confirmBox(title, body, danger = true) {
  return new Promise(res => {
    modal(`<div class="modal-head"><div><h3>${esc(title)}</h3><p>${body}</p></div></div>
      <div class="modal-foot">
        <button class="btn btn-ghost" data-x>Cancel</button>
        <button class="btn ${danger ? 'btn-danger' : 'btn-primary'}" data-ok>Confirm</button>
      </div>`, {
      onMount(c) {
        c.querySelector('[data-x]').onclick = () => { closeModal(); res(false); };
        c.querySelector('[data-ok]').onclick = () => { closeModal(); res(true); };
      }
    });
  });
}

/* ══════════════════════════════════════════════════════════
   FACE ENGINE
   ══════════════════════════════════════════════════════════ */
const Engine = {
  opts: null,
  backend: null,
  /* The bundled tfjs ranks a wasm backend first, but we ship no .wasm binary
     (it would be another ~10MB and webgl is faster here anyway). Pick an
     available backend explicitly — tfjs will NOT fall back on its own. */
  async pickBackend(onStep) {
    const tf = faceapi.tf;
    // drop the wasm backend: we ship no .wasm binary, and leaving it registered
    // makes tfjs fetch a missing file and log a 404 on every load
    try { tf.removeBackend('wasm'); } catch (_) {}
    const available = Object.keys(tf.engine().registryFactory || {});
    for (const b of ['webgl', 'cpu']) {
      if (!available.includes(b)) continue;
      try {
        onStep?.(`starting ${b} backend…`, 12);
        await tf.setBackend(b);
        await tf.ready();
        if (tf.getBackend() === b) { this.backend = b; return b; }
      } catch (e) { console.warn(`backend ${b} failed`, e); }
    }
    throw new Error('no usable compute backend (needs WebGL or CPU)');
  },
  async load(onStep) {
    if (S.engineReady) return;
    await this.pickBackend(onStep);
    onStep?.('loading face detector…', 20);
    await faceapi.nets.tinyFaceDetector.loadFromUri(CFG.MODELS);
    onStep?.('loading landmark model…', 48);
    await faceapi.nets.faceLandmark68Net.loadFromUri(CFG.MODELS);
    onStep?.('loading recognition net…', 74);
    await faceapi.nets.faceRecognitionNet.loadFromUri(CFG.MODELS);
    this.opts = new faceapi.TinyFaceDetectorOptions({
      inputSize: CFG.INPUT_SIZE, scoreThreshold: CFG.MIN_DETECT_SCORE
    });
    onStep?.('warming up…', 92);
    // warm-up pass on a blank canvas so the first real frame isn't slow
    const c = document.createElement('canvas'); c.width = c.height = CFG.INPUT_SIZE;
    const cx = c.getContext('2d'); cx.fillStyle = '#222'; cx.fillRect(0, 0, c.width, c.height);
    try { await faceapi.detectAllFaces(c, this.opts); } catch (_) {}
    S.engineReady = true;
    onStep?.('ready', 100);
  },
  /* all faces + descriptors from any media element */
  async detectAll(el) {
    return faceapi.detectAllFaces(el, this.opts).withFaceLandmarks().withFaceDescriptors();
  },
  /* single best face — used for enrolment from a still photo */
  async detectOne(el) {
    return faceapi.detectSingleFace(el, this.opts).withFaceLandmarks().withFaceDescriptor();
  },
};

/* squared euclidean distance */
function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

/* Build a flat matcher list from the current class roster.
   Every sample is its own entry -> nearest-sample matching, which
   handles glasses/hair/lighting variation better than averaging. */
function buildMatcher() {
  const entries = [];
  for (const p of S.people) {
    for (const d of (p.samples || [])) {
      entries.push({ personId: p.id, name: p.name, desc: Float32Array.from(d) });
    }
  }
  S.matcher = entries;
  return entries;
}

/* returns {personId,name,d} | {unknown:true,d} */
function matchDescriptor(desc) {
  if (!S.matcher || !S.matcher.length) return { unknown: true, d: 99 };
  let best = null, bd = Infinity;
  for (const e of S.matcher) {
    const d = dist(desc, e.desc);
    if (d < bd) { bd = d; best = e; }
  }
  if (bd <= CFG.MATCH_DIST) return { personId: best.personId, name: best.name, d: bd };
  return { unknown: true, d: bd, nearest: best ? best.name : null };
}

/* ══════════════════════════════════════════════════════════
   IMAGE UTILITIES
   ══════════════════════════════════════════════════════════ */
function fileToImage(file) {
  return new Promise((res, rej) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); res(img); };
    img.onerror = () => { URL.revokeObjectURL(url); rej(new Error('unreadable image')); };
    img.src = url;
  });
}

/* crop a detection box (with padding) to a square jpeg dataURL */
function cropThumb(src, box, size = CFG.THUMB) {
  const pad = box.width * 0.32;
  let x = box.x - pad, y = box.y - pad * 1.15;
  let w = box.width + pad * 2, h = box.height + pad * 2.1;
  const side = Math.max(w, h);
  x -= (side - w) / 2; y -= (side - h) / 2;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const cx = c.getContext('2d');
  cx.fillStyle = '#15170f'; cx.fillRect(0, 0, size, size);
  cx.drawImage(src, x, y, side, side, 0, 0, size, size);
  return c.toDataURL('image/jpeg', 0.82);
}

/* snapshot a video frame into a canvas at native resolution */
function frameToCanvas(video) {
  const c = document.createElement('canvas');
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  return c;
}
/* ══════════════════════════════════════════════════════════
   DATA OPS
   ══════════════════════════════════════════════════════════ */
async function loadClasses() {
  S.classes = (await DB.all('classes')).sort((a, b) => b.updated - a.updated);
  // attach counts
  const people = await DB.all('people');
  for (const c of S.classes) {
    const mine = people.filter(p => p.classId === c.id);
    c._people = mine.length;
    c._samples = mine.reduce((n, p) => n + (p.samples?.length || 0), 0);
  }
}

async function openClass(id) {
  S.cls = await DB.get('classes', id);
  if (!S.cls) { go('classes'); return; }
  S.people = (await DB.byIndex('people', 'classId', id))
    .sort((a, b) => a.name.localeCompare(b.name));
  buildMatcher();
}

async function createClass(name, code) {
  const c = { id: uid(), name: norm(name), code: norm(code), created: Date.now(), updated: Date.now() };
  await DB.put('classes', c);
  return c;
}

async function touchClass() {
  if (!S.cls) return;
  S.cls.updated = Date.now();
  await DB.put('classes', S.cls);
}

async function deleteClass(id) {
  const people = await DB.byIndex('people', 'classId', id);
  for (const p of people) await DB.del('people', p.id);
  const sess = await DB.byIndex('sessions', 'classId', id);
  for (const s of sess) await DB.del('sessions', s.id);
  await DB.del('classes', id);
}

/* find person in current class by name (case/space-insensitive) */
function findPersonByName(name) {
  const k = key(name);
  return S.people.find(p => key(p.name) === k) || null;
}

/**
 * Core enrolment primitive.
 * If a person with this name already exists in the class -> append the
 * descriptor as an extra sample (exactly the merge behaviour requested).
 * Otherwise create the person.
 * Returns {person, merged:boolean, skipped:boolean}
 */
async function enrolFace(name, descriptor, thumb) {
  const clean = norm(name);
  if (!clean) throw new Error('name required');
  let person = findPersonByName(clean);
  let merged = false;

  if (person) {
    merged = true;
    // guard: near-identical sample adds nothing but slows matching
    const tooClose = (person.samples || []).some(s => dist(descriptor, Float32Array.from(s)) < 0.055);
    if (tooClose) return { person, merged, skipped: true };
    person.samples = person.samples || [];
    person.samples.push(Array.from(descriptor));
    if (person.samples.length > CFG.MAX_SAMPLES) person.samples.shift();
    if (thumb && !person.thumb) person.thumb = thumb;
    person.updated = Date.now();
  } else {
    person = {
      id: uid(), classId: S.cls.id, name: clean,
      samples: [Array.from(descriptor)],
      thumb: thumb || null,
      created: Date.now(), updated: Date.now(),
    };
    S.people.push(person);
    S.people.sort((a, b) => a.name.localeCompare(b.name));
  }
  await DB.put('people', person);
  await touchClass();
  buildMatcher();
  return { person, merged, skipped: false };
}

async function deletePerson(id) {
  await DB.del('people', id);
  S.people = S.people.filter(p => p.id !== id);
  buildMatcher();
  await touchClass();
}

async function renamePerson(id, newName) {
  const p = S.people.find(x => x.id === id);
  if (!p) return;
  const clash = S.people.find(x => x.id !== id && key(x.name) === key(newName));
  if (clash) {
    // merge into the existing person
    clash.samples = [...(clash.samples || []), ...(p.samples || [])].slice(-CFG.MAX_SAMPLES);
    clash.thumb = clash.thumb || p.thumb;
    await DB.put('people', clash);
    await deletePerson(id);
    toast(`Merged into <b>${esc(clash.name)}</b> — ${clash.samples.length} samples`);
    return;
  }
  p.name = norm(newName); p.updated = Date.now();
  await DB.put('people', p);
  S.people.sort((a, b) => a.name.localeCompare(b.name));
  buildMatcher();
}
/* ══════════════════════════════════════════════════════════
   ROUTER + SCREENS
   ══════════════════════════════════════════════════════════ */
async function go(name, arg) {
  S.route = { name, arg };
  if (name === 'classes') { await loadClasses(); renderClasses(); }
  else if (name === 'class') { await openClass(arg); renderClass(); }
  else if (name === 'history') { await openClass(arg); await renderHistory(); }
  renderCrumbs();
  updateStats();
  $('#view').scrollTop = 0;
}

function renderCrumbs() {
  const c = $('#crumbs');
  if (S.route.name === 'classes') { c.innerHTML = `<b>all classes</b>`; return; }
  const nm = S.cls ? esc(S.cls.name) : '';
  if (S.route.name === 'history') c.innerHTML = `<span>classes</span> / <span>${nm}</span> / <b>history</b>`;
  else c.innerHTML = `<span>classes</span> / <b>${nm}</b>`;
}

function updateStats() {
  const t = $('#statsTxt');
  if (S.route.name === 'classes') {
    const ppl = S.classes.reduce((n, c) => n + (c._people || 0), 0);
    t.textContent = `${S.classes.length} class${S.classes.length === 1 ? '' : 'es'} · ${ppl} student${ppl === 1 ? '' : 's'}`;
  } else if (S.cls) {
    const smp = S.people.reduce((n, p) => n + (p.samples?.length || 0), 0);
    t.textContent = `${S.people.length} student${S.people.length === 1 ? '' : 's'} · ${smp} face sample${smp === 1 ? '' : 's'}`;
  }
}

/* ── SCREEN: class list ── */
function renderClasses() {
  const v = $('#view');
  const cards = S.classes.map(c => `
    <article class="card" data-open="${c.id}">
      <button class="kebab" data-del-class="${c.id}" title="Delete class">⋯</button>
      <div class="card-top">
        <div>
          ${c.code ? `<div class="card-code">${esc(c.code)}</div>` : ''}
          <h3 class="card-name">${esc(c.name)}</h3>
        </div>
      </div>
      <div class="card-stats">
        <div class="stat"><b>${c._people}</b><span>students</span></div>
        <div class="stat"><b>${c._samples}</b><span>samples</span></div>
      </div>
    </article>`).join('');

  v.innerHTML = `
    <div class="wrap">
      <div class="page-head">
        <div>
          <div class="eyebrow">roll call · on-device</div>
          <h1 class="h1">Your classes</h1>
          <p class="lede">Enrol each student's face once, then park the camera at the door
          and let it mark attendance as they walk in. Everything — photos, face data, logs —
          stays in this browser on this device.</p>
        </div>
      </div>
      ${S.classes.length ? `<div class="grid-cards">${cards}
        <button class="card card-add" id="addClass"><span class="plus">+</span>New class</button>
      </div>` : `
      <div class="empty">
        <div class="empty-mark">R / C</div>
        <h3>No classes yet</h3>
        <p>Create a class, add your students' photos, and you're ready to take attendance.</p>
        <button class="btn btn-primary btn-lg" id="addClass">+ Create your first class</button>
      </div>`}
    </div>`;

  $('#addClass').onclick = classDialog;
  $$('[data-open]', v).forEach(el => el.onclick = e => {
    if (e.target.closest('[data-del-class]')) return;
    go('class', el.dataset.open);
  });
  $$('[data-del-class]', v).forEach(b => b.onclick = async e => {
    e.stopPropagation();
    const c = S.classes.find(x => x.id === b.dataset.delClass);
    if (await confirmBox('Delete class?',
      `<b>${esc(c.name)}</b> and its ${c._people} enrolled student${c._people === 1 ? '' : 's'} will be permanently removed from this device. This cannot be undone.`)) {
      await deleteClass(c.id);
      toast('Class deleted');
      go('classes');
    }
  });
}

function classDialog() {
  modal(`
    <div class="modal-head"><div><h3>New class</h3><p>Give it a name you'll recognise at the door.</p></div></div>
    <label class="field"><span>Class name</span>
      <input class="input" id="cName" placeholder="e.g. Intro to Statistics — Tue lab" autofocus></label>
    <label class="field"><span>Course code <span class="dim">(optional)</span></span>
      <input class="input" id="cCode" placeholder="e.g. STAT 101"></label>
    <div class="modal-foot">
      <button class="btn btn-ghost" data-x>Cancel</button>
      <button class="btn btn-primary" data-ok>Create class</button>
    </div>`, {
    onMount(c) {
      const save = async () => {
        const n = $('#cName').value;
        if (!norm(n)) { toast('Please enter a class name', 'warn'); return; }
        const cls = await createClass(n, $('#cCode').value);
        closeModal();
        toast(`Class <b>${esc(cls.name)}</b> created`);
        go('class', cls.id);
      };
      c.querySelector('[data-x]').onclick = closeModal;
      c.querySelector('[data-ok]').onclick = save;
      $('#cName').onkeydown = e => { if (e.key === 'Enter') save(); };
      $('#cCode').onkeydown = e => { if (e.key === 'Enter') save(); };
    }
  });
}

/* ── SCREEN: single class / roster ── */
function renderClass() {
  const v = $('#view');
  const c = S.cls;
  v.innerHTML = `
    <div class="wrap">
      <div class="page-head">
        <div style="flex:1;min-width:240px">
          <div class="eyebrow">${c.code ? esc(c.code) + ' · ' : ''}class roster</div>
          <h1 class="h1">${esc(c.name)}</h1>
        </div>
        <div class="btn-row">
          <button class="btn btn-ghost" id="btnHistory">History</button>
          <button class="btn btn-ghost" id="btnAddPhotos">+ Add photos</button>
          <button class="btn btn-primary btn-lg" id="btnStart">▶ Take attendance</button>
        </div>
      </div>

      <div id="rosterHost"></div>
      <div class="rule"></div>
      <div class="btn-row">
        <button class="btn btn-sm btn-ghost" id="btnExport">Export backup</button>
        <button class="btn btn-sm btn-ghost" id="btnImport">Import backup</button>
        <button class="btn btn-sm btn-ghost" id="btnWebcamEnrol">Enrol from webcam</button>
      </div>
      <p class="hint" style="max-width:70ch">A backup is a single JSON file holding this class's
      names and face data. Keep one — if this browser's site data is cleared, the roster goes with it.</p>
    </div>`;

  renderRoster();
  $('#btnStart').onclick = () => startKiosk();
  $('#btnAddPhotos').onclick = uploadDialog;
  $('#btnHistory').onclick = () => go('history', c.id);
  $('#btnExport').onclick = exportClass;
  $('#btnImport').onclick = importClass;
  $('#btnWebcamEnrol').onclick = webcamEnrolDialog;
}

function renderRoster(filter = '') {
  const host = $('#rosterHost');
  const f = key(filter);
  const list = f ? S.people.filter(p => key(p.name).includes(f)) : S.people;

  const bar = `
    <div class="toolbar">
      <input class="input" id="rSearch" placeholder="Search students…" value="${esc(filter)}">
      <span class="badge ${S.people.length ? 'badge-ok' : 'badge-warn'}">${S.people.length} enrolled</span>
    </div>`;

  if (!S.people.length) {
    host.innerHTML = bar + `
      <div class="empty">
        <div class="empty-mark">□ □ □</div>
        <h3>No students enrolled</h3>
        <p>Upload photos of your students with their names. One clear, front-facing photo each
        is enough to start — the system adds more samples automatically as it sees them.</p>
        <button class="btn btn-primary btn-lg" id="emptyUpload">+ Upload student photos</button>
      </div>`;
    $('#emptyUpload').onclick = uploadDialog;
    wireSearch();
    return;
  }

  host.innerHTML = bar + `<div class="roster">` + list.map(p => {
    const n = p.samples?.length || 0;
    const pips = Array.from({ length: 5 }, (_, i) => `<i class="${i < Math.min(n, 5) ? 'on' : ''}"></i>`).join('');
    return `
      <div class="person" data-person="${p.id}">
        <button class="person-x" data-del="${p.id}" title="Remove student">✕</button>
        <div class="person-thumb">${p.thumb
          ? `<img src="${p.thumb}" alt="">`
          : `<div class="noimg">${esc(initials(p.name))}</div>`}</div>
        <div class="person-body">
          <div class="person-name">${esc(p.name)}</div>
          <div class="person-meta">${n} sample${n === 1 ? '' : 's'}</div>
        </div>
        <div class="samples-strip">${pips}</div>
      </div>`;
  }).join('') + `</div>`;

  wireSearch();
  $$('[data-del]', host).forEach(b => b.onclick = async e => {
    e.stopPropagation();
    const p = S.people.find(x => x.id === b.dataset.del);
    if (await confirmBox('Remove student?', `<b>${esc(p.name)}</b> and their face data will be deleted from this class.`)) {
      await deletePerson(p.id);
      toast('Student removed');
      renderRoster($('#rSearch')?.value || '');
      updateStats();
    }
  });
  $$('[data-person]', host).forEach(el => el.onclick = e => {
    if (e.target.closest('[data-del]')) return;
    personDialog(el.dataset.person);
  });

  function wireSearch() {
    const s = $('#rSearch');
    if (s) s.oninput = () => renderRoster(s.value);
  }
}

function personDialog(id) {
  const p = S.people.find(x => x.id === id);
  if (!p) return;
  modal(`
    <div class="modal-head">
      <div><h3>${esc(p.name)}</h3>
      <p>${p.samples?.length || 0} face sample${(p.samples?.length || 0) === 1 ? '' : 's'} ·
      added ${fmtDate(p.created)}</p></div>
    </div>
    ${p.thumb ? `<img src="${p.thumb}" alt="" style="width:132px;height:132px;object-fit:cover;border-radius:4px;border:1px solid var(--line-2);margin-bottom:18px">` : ''}
    <label class="field"><span>Name</span><input class="input" id="pName" value="${esc(p.name)}"></label>
    <p class="hint">Renaming to a name that already exists in this class merges the two —
    their face samples are combined into one student.</p>
    <div class="modal-foot">
      <button class="btn btn-ghost" data-x>Cancel</button>
      <button class="btn btn-primary" data-ok>Save</button>
    </div>`, {
    onMount(c) {
      c.querySelector('[data-x]').onclick = closeModal;
      c.querySelector('[data-ok]').onclick = async () => {
        const nn = $('#pName').value;
        if (!norm(nn)) { toast('Name cannot be empty', 'warn'); return; }
        if (norm(nn) !== p.name) await renamePerson(p.id, nn);
        closeModal();
        renderRoster();
        updateStats();
      };
    }
  });
}

/* ── SCREEN: session history ── */
async function renderHistory() {
  const sessions = (await DB.byIndex('sessions', 'classId', S.cls.id)).sort((a, b) => b.started - a.started);
  const v = $('#view');
  v.innerHTML = `
    <div class="wrap">
      <div class="page-head">
        <div>
          <div class="eyebrow">${esc(S.cls.name)}</div>
          <h1 class="h1">Attendance history</h1>
        </div>
        <div class="btn-row"><button class="btn btn-ghost" id="back">← Roster</button></div>
      </div>
      ${!sessions.length ? `
        <div class="empty"><div class="empty-mark">— —</div><h3>No sessions yet</h3>
        <p>Run attendance once and the record will show up here, ready to export as CSV.</p></div>`
      : sessions.map(s => `
        <article class="card" style="cursor:default;margin-bottom:14px">
          <div class="card-top">
            <div>
              <div class="card-code">${fmtDate(s.started)} · ${fmtTime(s.started)}</div>
              <h3 class="card-name">${s.marks.length} present<span class="dim" style="font-weight:400"> / ${s.rosterSize} enrolled</span></h3>
            </div>
            <div class="btn-row">
              <button class="btn btn-sm btn-ghost" data-csv="${s.id}">CSV</button>
              <button class="btn btn-sm btn-danger" data-dels="${s.id}">Delete</button>
            </div>
          </div>
          <div class="card-stats">
            <div class="stat"><b>${s.marks.length}</b><span>present</span></div>
            <div class="stat"><b>${Math.max(0, s.rosterSize - s.marks.length)}</b><span>absent</span></div>
            <div class="stat"><b>${s.unknownCount || 0}</b><span>unknown</span></div>
          </div>
        </article>`).join('')}
    </div>`;
  $('#back').onclick = () => go('class', S.cls.id);
  $$('[data-csv]', v).forEach(b => b.onclick = () => {
    const s = sessions.find(x => x.id === b.dataset.csv);
    downloadCSV(s);
  });
  $$('[data-dels]', v).forEach(b => b.onclick = async () => {
    if (await confirmBox('Delete session record?', 'This attendance record will be removed.')) {
      await DB.del('sessions', b.dataset.dels);
      renderHistory();
    }
  });
}

function downloadCSV(s) {
  const present = new Map(s.marks.map(m => [m.personId, m]));
  const rows = [['Name', 'Status', 'Time', 'Confidence']];
  for (const r of s.roster) {
    const m = present.get(r.id);
    rows.push([r.name, m ? 'Present' : 'Absent', m ? new Date(m.at).toLocaleString() : '', m ? m.score.toFixed(3) : '']);
  }
  const csv = rows.map(r => r.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(',')).join('\r\n');
  const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `attendance_${(S.cls.code || S.cls.name).replace(/[^\w-]+/g, '_')}_${new Date(s.started).toISOString().slice(0, 10)}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  toast('CSV downloaded');
}
/* ══════════════════════════════════════════════════════════
   ENROLMENT — bulk photo upload
   ══════════════════════════════════════════════════════════ */
function guessNameFromFile(fname) {
  return norm(fname.replace(/\.[a-z0-9]+$/i, '')       // strip extension
    .replace(/[_\-.]+/g, ' ')                            // separators -> space
    .replace(/\b(img|image|photo|pic|dsc|screenshot)\b/gi, '')
    .replace(/\d{3,}/g, '')                              // long digit runs
    .replace(/\s+/g, ' '))
    .split(' ').filter(Boolean)
    .map(w => w[0].toUpperCase() + w.slice(1))
    .join(' ');
}

function uploadDialog() {
  modal(`
    <div class="modal-head">
      <div><h3>Add student photos</h3>
      <p>Pick one or more photos. Filenames are used as a first guess at the name —
      check each one before saving. A name that already exists gets an extra face sample
      instead of a duplicate entry.</p></div>
    </div>
    <div class="drop" id="drop">
      <div class="drop-mark">⬚</div>
      <b>Choose photos or drag them here</b>
      <p>JPG / PNG / WebP · one face per photo works best</p>
    </div>
    <input type="file" id="fileIn" accept="image/*" multiple hidden>
    <div id="qHost" style="margin-top:18px"></div>
    <div class="modal-foot">
      <button class="btn btn-ghost" data-x>Close</button>
      <button class="btn btn-primary" id="qSave" disabled>Save to class</button>
    </div>`, {
    wide: true,
    onMount(card) {
      const drop = $('#drop'), input = $('#fileIn'), host = $('#qHost'), saveBtn = $('#qSave');
      let queue = [];

      card.querySelector('[data-x]').onclick = closeModal;
      drop.onclick = () => input.click();
      drop.ondragover = e => { e.preventDefault(); drop.classList.add('hot'); };
      drop.ondragleave = () => drop.classList.remove('hot');
      drop.ondrop = e => {
        e.preventDefault(); drop.classList.remove('hot');
        handle([...e.dataTransfer.files].filter(f => f.type.startsWith('image/')));
      };
      input.onchange = () => { handle([...input.files]); input.value = ''; };

      async function handle(files) {
        if (!files.length) return;
        for (const f of files) {
          const row = { id: uid(), file: f, name: guessNameFromFile(f.name), state: 'wait', thumb: null, desc: null };
          queue.push(row);
        }
        paint();
        for (const row of queue.filter(r => r.state === 'wait')) {
          row.state = 'scan'; paint();
          try {
            const img = await fileToImage(row.file);
            const det = await Engine.detectOne(img);
            if (!det) { row.state = 'noface'; }
            else {
              row.desc = det.descriptor;
              row.thumb = cropThumb(img, det.detection.box);
              row.state = 'ok';
            }
          } catch (err) { row.state = 'bad'; }
          paint();
        }
      }

      function paint() {
        if (!queue.length) { host.innerHTML = ''; saveBtn.disabled = true; return; }
        host.innerHTML = `<div class="q-list">` + queue.map(r => {
          const tag = {
            wait:  '<span class="q-tag warn">queued</span>',
            scan:  '<span class="q-tag warn">scanning…</span>',
            ok:    '<span class="q-tag ok">face found</span>',
            noface:'<span class="q-tag bad">no face</span>',
            bad:   '<span class="q-tag bad">unreadable</span>',
            saved: '<span class="q-tag ok">saved</span>',
          }[r.state];
          const exists = r.state === 'ok' && findPersonByName(r.name);
          return `
          <div class="q-row ${r.state === 'ok' ? 'ok' : (r.state === 'noface' || r.state === 'bad') ? 'bad' : ''}">
            <div class="q-thumb">${r.thumb ? `<img src="${r.thumb}" alt="">` : ''}</div>
            <div class="q-main">
              <div class="q-file">${esc(r.file.name)}</div>
              <input class="input" data-name="${r.id}" value="${esc(r.name)}" placeholder="Student name"
                ${r.state === 'saved' ? 'disabled' : ''}>
              ${exists ? `<div class="hint" style="margin-top:5px;color:var(--cyan)">↳ adds a sample to existing <b>${esc(exists.name)}</b></div>` : ''}
            </div>
            <div style="display:flex;flex-direction:column;gap:6px;align-items:flex-end">
              ${tag}
              <button class="btn btn-sm btn-ghost" data-drop="${r.id}">✕</button>
            </div>
          </div>`;
        }).join('') + `</div>`;

        $$('[data-name]', host).forEach(i => i.oninput = () => {
          const r = queue.find(x => x.id === i.dataset.name);
          if (r) { r.name = i.value; }
        });
        $$('[data-name]', host).forEach(i => i.onblur = () => paint());
        $$('[data-drop]', host).forEach(b => b.onclick = () => {
          queue = queue.filter(x => x.id !== b.dataset.drop); paint();
        });

        const ready = queue.filter(r => r.state === 'ok' && norm(r.name));
        saveBtn.disabled = !ready.length;
        saveBtn.textContent = ready.length ? `Save ${ready.length} to class` : 'Save to class';
      }

      saveBtn.onclick = async () => {
        const ready = queue.filter(r => r.state === 'ok' && norm(r.name));
        let created = 0, merged = 0, skipped = 0;
        for (const r of ready) {
          const res = await enrolFace(r.name, r.desc, r.thumb);
          if (res.skipped) skipped++; else if (res.merged) merged++; else created++;
          r.state = 'saved';
        }
        paint();
        renderRoster();
        updateStats();
        const bits = [];
        if (created) bits.push(`${created} added`);
        if (merged) bits.push(`${merged} extra sample${merged === 1 ? '' : 's'}`);
        if (skipped) bits.push(`${skipped} near-duplicate skipped`);
        toast(bits.join(' · ') || 'Nothing to save');
        closeModal();
      };
    }
  });
}

/* ── enrol straight from webcam (before class, at your desk) ── */
function webcamEnrolDialog() {
  modal(`
    <div class="modal-head"><div><h3>Enrol from webcam</h3>
    <p>Type the name, frame the face, then capture. Capture two or three times at slightly
    different angles for noticeably better recognition at the door.</p></div></div>
    <label class="field"><span>Student name</span>
      <input class="input" id="wName" placeholder="Full name" autofocus></label>
    <div class="camwrap" style="aspect-ratio:4/3;max-height:44vh;margin-bottom:14px">
      <video id="wVideo" playsinline autoplay muted style="width:100%;height:100%;object-fit:cover;transform:scaleX(-1)"></video>
    </div>
    <div id="wStatus" class="hint"></div>
    <div class="modal-foot">
      <button class="btn btn-ghost" data-x>Done</button>
      <button class="btn btn-primary" id="wShot">◉ Capture face</button>
    </div>`, {
    wide: true,
    onMount(card) {
      const vid = $('#wVideo'); let stream = null; let count = 0;
      (async () => {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 } }, audio: false });
          vid.srcObject = stream;
        } catch (e) { $('#wStatus').innerHTML = `<span style="color:var(--red)">Camera unavailable: ${esc(e.message)}</span>`; }
      })();
      const stop = () => { stream?.getTracks().forEach(t => t.stop()); };
      card.querySelector('[data-x]').onclick = () => { stop(); closeModal(); renderRoster(); updateStats(); };
      $('#wShot').onclick = async () => {
        const name = $('#wName').value;
        if (!norm(name)) { toast('Enter a name first', 'warn'); return; }
        if (!vid.videoWidth) { toast('Camera not ready', 'warn'); return; }
        const cv = frameToCanvas(vid);
        const det = await Engine.detectOne(cv);
        if (!det) { $('#wStatus').innerHTML = `<span style="color:var(--amber)">No face detected — move closer or improve the lighting.</span>`; return; }
        const thumb = cropThumb(cv, det.detection.box);
        const res = await enrolFace(name, det.descriptor, thumb);
        count++;
        $('#wStatus').innerHTML = `<span style="color:var(--lime)">Saved sample ${count} for <b>${esc(res.person.name)}</b>
          — ${res.person.samples.length} total.${res.skipped ? ' (too similar to an existing sample, not stored)' : ''}</span>`;
      };
    }
  });
}

/* ══════════════════════════════════════════════════════════
   BACKUP
   ══════════════════════════════════════════════════════════ */
async function exportClass() {
  const payload = {
    format: 'rollcall.class.v1',
    exported: Date.now(),
    class: { name: S.cls.name, code: S.cls.code },
    people: S.people.map(p => ({ name: p.name, samples: p.samples, thumb: p.thumb, created: p.created })),
  };
  const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `rollcall_${(S.cls.code || S.cls.name).replace(/[^\w-]+/g, '_')}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  toast('Backup downloaded');
}

function importClass() {
  const inp = document.createElement('input');
  inp.type = 'file'; inp.accept = 'application/json,.json';
  inp.onchange = async () => {
    const f = inp.files[0]; if (!f) return;
    try {
      const data = JSON.parse(await f.text());
      if (data.format !== 'rollcall.class.v1' || !Array.isArray(data.people)) throw new Error('not a RollCall backup');
      let created = 0, merged = 0;
      for (const p of data.people) {
        if (!p.name || !Array.isArray(p.samples) || !p.samples.length) continue;
        const existing = findPersonByName(p.name);
        if (existing) {
          existing.samples = [...(existing.samples || []), ...p.samples].slice(-CFG.MAX_SAMPLES);
          existing.thumb = existing.thumb || p.thumb || null;
          await DB.put('people', existing);
          merged++;
        } else {
          const person = {
            id: uid(), classId: S.cls.id, name: norm(p.name),
            samples: p.samples, thumb: p.thumb || null,
            created: p.created || Date.now(), updated: Date.now(),
          };
          await DB.put('people', person);
          S.people.push(person);
          created++;
        }
      }
      S.people.sort((a, b) => a.name.localeCompare(b.name));
      buildMatcher();
      await touchClass();
      renderRoster(); updateStats();
      toast(`Imported — ${created} new, ${merged} merged`);
    } catch (e) {
      toast(`Import failed: ${esc(e.message)}`, 'err', 4600);
    }
  };
  inp.click();
}
/* ══════════════════════════════════════════════════════════
   KIOSK — continuous auto-scan
   ══════════════════════════════════════════════════════════ */
const Beep = (() => {
  let ctx = null;
  return (freq = 880, ms = 110, vol = 0.06) => {
    try {
      ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = 'sine'; o.frequency.value = freq;
      g.gain.value = vol;
      o.connect(g); g.connect(ctx.destination);
      o.start();
      g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + ms / 1000);
      o.stop(ctx.currentTime + ms / 1000);
    } catch (_) {}
  };
})();

async function startKiosk() {
  if (!S.people.length) {
    toast('Enrol at least one student first', 'warn', 4200);
    return;
  }
  S.session = {
    id: uid(), classId: S.cls.id, started: Date.now(),
    marks: [], unknowns: [],
    rosterSize: S.people.length,
    roster: S.people.map(p => ({ id: p.id, name: p.name })),
    cooldown: new Map(),     // personId | 'unknown' -> ts
    streak: { id: null, n: 0 },
  };
  S.paused = false;
  S.kioskTab = 'log';
  $('#kiosk').hidden = false;
  $('#shell').hidden = true;
  $('#ksClass').textContent = S.cls.name;
  $('#ksMeta').textContent = `${fmtDate(Date.now())} · started ${fmtTime(Date.now())}`;
  $('#ksTotal').textContent = '/' + S.people.length;
  paintSide();
  hud('STARTING', 'requesting camera…');

  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (e) {
    $('#camMsg').hidden = false;
    $('#camMsg').innerHTML = `<div>
      <b style="color:var(--red);font-family:var(--f-disp);font-size:17px">Camera blocked</b><br><br>
      ${esc(e.message)}<br><br>
      Allow camera access for this page, then press <b>Take attendance</b> again.<br>
      <span class="dim">Note: browsers only permit camera access on https:// or localhost.</span>
    </div>`;
    hud('NO CAMERA', 'permission denied');
    return;
  }
  const v = $('#video');
  v.srcObject = S.stream;
  v.classList.toggle('no-mirror', !S.mirror);
  await v.play().catch(() => {});
  hud('READY', 'scanning for faces…');
  loop();
}

function hud(name, sub, cls = '') {
  const n = $('#hudName'), s = $('#hudSub');
  n.className = 'hud-name ' + cls;
  n.textContent = name;
  s.textContent = sub || '';
}

function stopKiosk() {
  clearTimeout(S.loopId);
  S.loopId = null;
  S.stream?.getTracks().forEach(t => t.stop());
  S.stream = null;
  $('#video').srcObject = null;
  $('#kiosk').hidden = true;
  $('#shell').hidden = false;
  $('#camMsg').hidden = true;
}

/* main recognition loop */
async function loop() {
  const v = $('#video');
  if (!S.stream) return;

  if (!S.paused && v.videoWidth) {
    try {
      const dets = await Engine.detectAll(v);
      drawOverlay(dets, v);
      if (dets.length) handleDetections(dets, v);
      else {
        if (S.session.streak.n) S.session.streak = { id: null, n: 0 };
        if ($('#hudName').textContent !== 'READY' && !$('#hudName').classList.contains('hit'))
          hud('READY', 'scanning for faces…');
      }
    } catch (e) {
      console.warn('detect error', e);
    }
  }
  S.loopId = setTimeout(loop, S.paused ? 400 : CFG.RECHECK_MS);
}

/* draw boxes on the overlay canvas, mirrored to match the video */
function drawOverlay(dets, video) {
  const cv = $('#overlay');
  const w = video.clientWidth, h = video.clientHeight;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const cx = cv.getContext('2d');
  cx.clearRect(0, 0, w, h);
  if (!dets.length) return;

  // video uses object-fit:cover -> replicate that transform
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.max(w / vw, h / vh);
  const ox = (w - vw * scale) / 2, oy = (h - vh * scale) / 2;

  for (const d of dets) {
    const m = matchDescriptor(d.descriptor);
    const b = d.detection.box;
    let x = b.x * scale + ox, y = b.y * scale + oy;
    const bw = b.width * scale, bh = b.height * scale;
    if (S.mirror) x = w - x - bw;

    const known = !m.unknown;
    const col = known ? '#c3f53c' : '#ffb020';
    cx.strokeStyle = col; cx.lineWidth = 2.5;
    // corner brackets rather than a full box — reads better over faces
    const c = Math.min(bw, bh) * 0.24;
    cx.beginPath();
    cx.moveTo(x, y + c); cx.lineTo(x, y); cx.lineTo(x + c, y);
    cx.moveTo(x + bw - c, y); cx.lineTo(x + bw, y); cx.lineTo(x + bw, y + c);
    cx.moveTo(x + bw, y + bh - c); cx.lineTo(x + bw, y + bh); cx.lineTo(x + bw - c, y + bh);
    cx.moveTo(x + c, y + bh); cx.lineTo(x, y + bh); cx.lineTo(x, y + bh - c);
    cx.stroke();

    const label = known ? m.name : 'UNKNOWN';
    cx.font = '600 15px "IBM Plex Sans",sans-serif';
    const tw = cx.measureText(label).width;
    // keep the label inside the canvas — faces near an edge would clip it
    const lw = tw + 16, lh = 24;
    let lx = Math.min(Math.max(0, x), Math.max(0, w - lw));
    let ly = y - lh - 2;
    // flip below the face if it would run off the top, or collide with the
    // close button that sits in the top-left corner
    if (ly < 0 || (ly < 62 && lx < 78)) ly = Math.min(y + bh + 2, h - lh);
    cx.fillStyle = 'rgba(8,9,7,.82)';
    cx.fillRect(lx, ly, lw, lh);
    cx.fillStyle = col;
    cx.fillText(label, lx + 8, ly + 17);
  }
}

/* decide what to do with the faces in this frame */
function handleDetections(dets, video) {
  const now = Date.now();
  // evaluate the largest face first — that's whoever is at the door
  dets.sort((a, b) => b.detection.box.area - a.detection.box.area);
  const d = dets[0];
  const m = matchDescriptor(d.descriptor);

  if (!m.unknown) {
    const cd = S.session.cooldown.get(m.personId) || 0;
    if (now - cd < CFG.COOLDOWN_MS) {
      const already = S.session.marks.find(x => x.personId === m.personId);
      hud(m.name, already ? 'already marked present' : 'confirming…', 'dup');
      return;
    }
    // require N consecutive agreeing frames before committing
    if (S.session.streak.id === m.personId) S.session.streak.n++;
    else S.session.streak = { id: m.personId, n: 1 };

    if (S.session.streak.n < CFG.CONSEC_HITS) {
      hud(m.name, 'hold still…', 'dup');
      return;
    }
    S.session.streak = { id: null, n: 0 };
    markPresent(m, d, video);
    return;
  }

  // ── unknown face ──
  S.session.streak = { id: null, n: 0 };
  const cd = S.session.cooldown.get('unknown') || 0;
  hud('CANNOT DETECT', 'face not in this class — tap to add', 'miss');
  if (now - cd < CFG.COOLDOWN_MS) return;
  S.session.cooldown.set('unknown', now);

  const cv = frameToCanvas(video);
  const thumb = cropThumb(cv, d.detection.box);
  const rec = { id: uid(), at: now, thumb, desc: Array.from(d.descriptor), dist: m.d, nearest: m.nearest };
  S.session.unknowns.unshift(rec);
  S.session.unknownCount = S.session.unknowns.length;
  Beep(300, 150, .05);
  paintSide();
  if (S.kioskTab !== 'unknown') {
    toast(`Unrecognised face captured — see <b>Unknown</b> to add them`, 'warn', 3600);
  }
}

async function markPresent(m, det, video) {
  const now = Date.now();
  S.session.cooldown.set(m.personId, now);
  if (S.session.marks.find(x => x.personId === m.personId)) {
    hud(m.name, 'already marked present', 'dup');
    return;
  }
  const person = S.people.find(p => p.id === m.personId);
  S.session.marks.unshift({
    personId: m.personId, name: m.name, at: now, score: m.d,
    thumb: person?.thumb || null, isNew: true,
  });
  setTimeout(() => {
    const mk = S.session?.marks.find(x => x.personId === m.personId);
    if (mk) { mk.isNew = false; }
  }, 2600);

  hud(m.name, `marked present · ${fmtTime(now)}`, 'hit');
  Beep(1040, 110, .06);
  paintSide();

  // opportunistic learning: a confident live match becomes an extra sample,
  // which makes the model steadily better in this room's lighting
  if (person && m.d < 0.34 && (person.samples?.length || 0) < CFG.MAX_SAMPLES) {
    const tooClose = person.samples.some(s => dist(det.descriptor, Float32Array.from(s)) < 0.16);
    if (!tooClose) {
      person.samples.push(Array.from(det.descriptor));
      person.updated = now;
      await DB.put('people', person);
      buildMatcher();
    }
  }
}

/* ── kiosk side panel ── */
function paintSide() {
  const s = S.session;
  $('#ksPresent').textContent = s.marks.length;
  $('#ksUnkCount').textContent = s.unknowns.length;
  const markedIds = new Set(s.marks.map(m => m.personId));
  const absent = S.people.filter(p => !markedIds.has(p.id));
  $('#ksAbsCount').textContent = absent.length;

  const body = $('#ksBody');
  if (S.kioskTab === 'log') {
    body.innerHTML = s.marks.length ? s.marks.map(m => `
      <div class="log-row ${m.isNew ? 'is-new' : ''}">
        <div class="log-thumb">${m.thumb ? `<img src="${m.thumb}" alt="">` : `<div class="noimg">${esc(initials(m.name))}</div>`}</div>
        <div class="log-main">
          <div class="log-name">${esc(m.name)}</div>
          <div class="log-time">${fmtTime(m.at)} · <span class="log-score">d=${m.score.toFixed(2)}</span></div>
        </div>
      </div>`).join('')
      : `<div class="side-empty">no one marked yet<br>point the camera at the doorway</div>`;
  }
  else if (S.kioskTab === 'unknown') {
    body.innerHTML = s.unknowns.length ? s.unknowns.map(u => `
      <div class="unk-row">
        <div class="log-thumb"><img src="${u.thumb}" alt=""></div>
        <div class="log-main">
          <div class="log-name" style="color:var(--amber)">Not in class</div>
          <div class="log-time">${fmtTime(u.at)}${u.nearest ? ` · closest: ${esc(u.nearest)}` : ''}</div>
        </div>
        <button class="btn btn-sm btn-primary" data-add="${u.id}">Add</button>
        <button class="btn btn-sm btn-ghost" data-skip="${u.id}">✕</button>
      </div>`).join('')
      : `<div class="side-empty">no unrecognised faces<br><span class="dim">anyone not enrolled shows up here</span></div>`;

    $$('[data-add]', body).forEach(b => b.onclick = () => addUnknownDialog(b.dataset.add));
    $$('[data-skip]', body).forEach(b => b.onclick = () => {
      S.session.unknowns = S.session.unknowns.filter(x => x.id !== b.dataset.skip);
      paintSide();
    });
  }
  else {
    body.innerHTML = absent.length ? absent.map(p => `
      <div class="log-row" style="opacity:.72">
        <div class="log-thumb">${p.thumb ? `<img src="${p.thumb}" alt="">` : `<div class="noimg">${esc(initials(p.name))}</div>`}</div>
        <div class="log-main">
          <div class="log-name">${esc(p.name)}</div>
          <div class="log-time">not yet seen</div>
        </div>
        <button class="btn btn-sm btn-ghost" data-manual="${p.id}">Mark</button>
      </div>`).join('')
      : `<div class="side-empty">everyone is present 🎉</div>`.replace(' 🎉', '');

    $$('[data-manual]', body).forEach(b => b.onclick = () => {
      const p = S.people.find(x => x.id === b.dataset.manual);
      S.session.marks.unshift({ personId: p.id, name: p.name, at: Date.now(), score: 0, thumb: p.thumb, isNew: true, manual: true });
      S.session.cooldown.set(p.id, Date.now());
      toast(`<b>${esc(p.name)}</b> marked manually`);
      paintSide();
    });
  }
}

/* enrol an unknown capture without leaving the kiosk */
function addUnknownDialog(unkId) {
  const u = S.session.unknowns.find(x => x.id === unkId);
  if (!u) return;
  const wasPaused = S.paused;
  S.paused = true;

  const names = S.people.map(p => `<option value="${esc(p.name)}"></option>`).join('');
  modal(`
    <div class="modal-head"><div><h3>Add to class</h3>
    <p>Type the name. If it matches a student already in this class, this face is stored
    as an extra sample for them instead of creating a duplicate.</p></div></div>
    <img src="${u.thumb}" alt="" style="width:148px;height:148px;object-fit:cover;border-radius:4px;border:1px solid var(--line-2);margin-bottom:18px">
    <label class="field"><span>Student name</span>
      <input class="input" id="uName" list="uList" placeholder="Full name" autofocus>
      <datalist id="uList">${names}</datalist></label>
    <label class="field" style="display:flex;align-items:center;gap:10px;cursor:pointer">
      <input type="checkbox" id="uMark" checked style="width:20px;height:20px;accent-color:var(--lime)">
      <span style="margin:0;text-transform:none;letter-spacing:0;font-family:var(--f-body);font-size:14px;color:var(--ink-2)">Also mark present right now</span>
    </label>
    <div class="modal-foot">
      <button class="btn btn-ghost" data-x>Cancel</button>
      <button class="btn btn-primary" data-ok>Save</button>
    </div>`, {
    onMount(c) {
      const done = () => { S.paused = wasPaused; closeModal(); };
      c.querySelector('[data-x]').onclick = done;
      const save = async () => {
        const nm = $('#uName').value;
        if (!norm(nm)) { toast('Enter a name', 'warn'); return; }
        const res = await enrolFace(nm, Float32Array.from(u.desc), u.thumb);
        if ($('#uMark').checked && !S.session.marks.find(x => x.personId === res.person.id)) {
          S.session.marks.unshift({
            personId: res.person.id, name: res.person.name, at: Date.now(),
            score: 0, thumb: res.person.thumb, isNew: true,
          });
          S.session.cooldown.set(res.person.id, Date.now());
        }
        S.session.unknowns = S.session.unknowns.filter(x => x.id !== u.id);
        S.session.rosterSize = S.people.length;
        S.session.roster = S.people.map(p => ({ id: p.id, name: p.name }));
        $('#ksTotal').textContent = '/' + S.people.length;
        toast(res.merged
          ? `Extra sample added to <b>${esc(res.person.name)}</b> (${res.person.samples.length} total)`
          : `<b>${esc(res.person.name)}</b> added to the class`);
        done();
        paintSide();
      };
      c.querySelector('[data-ok]').onclick = save;
      $('#uName').onkeydown = e => { if (e.key === 'Enter') save(); };
    }
  });
}

/* ── end session -> report ── */
async function endSession() {
  const s = S.session;
  s.ended = Date.now();
  s.unknownCount = s.unknowns.length;
  const record = {
    id: s.id, classId: s.classId, started: s.started, ended: s.ended,
    rosterSize: S.people.length,
    roster: S.people.map(p => ({ id: p.id, name: p.name })),
    marks: s.marks.map(m => ({ personId: m.personId, name: m.name, at: m.at, score: m.score, manual: !!m.manual })),
    unknownCount: s.unknowns.length,
  };
  await DB.put('sessions', record);

  const markedIds = new Set(record.marks.map(m => m.personId));
  const absent = S.people.filter(p => !markedIds.has(p.id));
  stopKiosk();

  modal(`
    <div class="modal-head"><div><h3>Attendance report</h3>
    <p>${esc(S.cls.name)} · ${fmtDate(record.started)} · ${fmtTime(record.started)}–${fmtTime(record.ended)}</p></div></div>
    <div style="display:flex;gap:26px;margin-bottom:20px;padding-bottom:18px;border-bottom:1px solid var(--line)">
      <div class="stat"><b style="color:var(--lime)">${record.marks.length}</b><span>present</span></div>
      <div class="stat"><b>${absent.length}</b><span>absent</span></div>
      <div class="stat"><b>${record.rosterSize}</b><span>enrolled</span></div>
    </div>
    <div class="scroll-y">
      <table class="tbl">
        <thead><tr><th>Student</th><th>Status</th><th>Time</th></tr></thead>
        <tbody>
          ${record.marks.map(m => `<tr><td>${esc(m.name)}</td>
            <td><span class="tag-present">● present${m.manual ? ' (manual)' : ''}</span></td>
            <td class="mono dim">${fmtTime(m.at)}</td></tr>`).join('')}
          ${absent.map(p => `<tr><td>${esc(p.name)}</td>
            <td><span class="tag-absent">○ absent</span></td><td></td></tr>`).join('')}
        </tbody>
      </table>
    </div>
    <div class="modal-foot">
      <button class="btn btn-ghost" data-x>Close</button>
      <button class="btn btn-primary" data-csv>Download CSV</button>
    </div>`, {
    wide: true,
    onMount(c) {
      c.querySelector('[data-x]').onclick = () => { closeModal(); go('class', S.cls.id); };
      c.querySelector('[data-csv]').onclick = () => downloadCSV(record);
    }
  });
  S.session = null;
}
/* ══════════════════════════════════════════════════════════
   SETTINGS
   ══════════════════════════════════════════════════════════ */
function settingsDialog() {
  const pct = Math.round((0.75 - CFG.MATCH_DIST) / 0.4 * 100);
  modal(`
    <div class="modal-head"><div><h3>Settings</h3>
    <p>Tune how strict recognition is. Changes apply immediately and are not saved between visits.</p></div></div>

    <label class="field"><span>Match strictness — currently ${CFG.MATCH_DIST.toFixed(2)}</span>
      <input class="input" type="range" id="sThresh" min="0.38" max="0.62" step="0.01" value="${CFG.MATCH_DIST}"
        style="padding:0;min-height:auto;accent-color:var(--lime)">
    </label>
    <p class="hint" style="margin-top:-6px">
      <b>Lower</b> = stricter: fewer wrong names, more "cannot detect".<br>
      <b>Higher</b> = looser: catches more students, small risk of confusing similar faces.<br>
      Move it lower if two students get mixed up; higher if enrolled students aren't recognised.
    </p>

    <div class="rule"></div>
    <label class="field" style="display:flex;align-items:center;gap:10px;cursor:pointer">
      <input type="checkbox" id="sMirror" ${S.mirror ? 'checked' : ''} style="width:20px;height:20px;accent-color:var(--lime)">
      <span style="margin:0;text-transform:none;letter-spacing:0;font-family:var(--f-body);font-size:14px;color:var(--ink-2)">Mirror the camera preview (natural for a front-facing camera)</span>
    </label>

    <div class="rule"></div>
    <div style="font-family:var(--f-mono);font-size:11.5px;color:var(--ink-3);line-height:1.85">
      engine · tiny-face-detector + 68 landmarks + resnet-34 embeddings<br>
      compute · ${esc(Engine.backend || '—')}<br>
      matching · nearest-sample over 128-d descriptors<br>
      data · IndexedDB on this device, nothing is uploaded<br>
      samples kept per student · max ${CFG.MAX_SAMPLES}
    </div>
    <div class="modal-foot"><button class="btn btn-primary" data-x>Done</button></div>`, {
    onMount(c) {
      c.querySelector('[data-x]').onclick = closeModal;
      const r = $('#sThresh');
      r.oninput = () => {
        CFG.MATCH_DIST = parseFloat(r.value);
        c.querySelector('.field span').textContent = `Match strictness — currently ${CFG.MATCH_DIST.toFixed(2)}`;
      };
      $('#sMirror').onchange = e => {
        S.mirror = e.target.checked;
        $('#video').classList.toggle('no-mirror', !S.mirror);
      };
    }
  });
}

/* ══════════════════════════════════════════════════════════
   WIRING + BOOT
   ══════════════════════════════════════════════════════════ */
$('#btnSettings').onclick = settingsDialog;
$$('[data-nav]').forEach(b => b.onclick = () => go(b.dataset.nav));

$('#btnPause').onclick = () => {
  S.paused = !S.paused;
  $('#btnPause').textContent = S.paused ? 'Resume' : 'Pause';
  $('#scanline').classList.toggle('off', S.paused);
  hud(S.paused ? 'PAUSED' : 'READY', S.paused ? 'scanning stopped' : 'scanning for faces…');
};
$('#btnEndSession').onclick = endSession;
$('#btnKioskClose').onclick = async () => {
  if (S.session?.marks.length) {
    if (!await confirmBox('Leave without saving?',
      `${S.session.marks.length} student${S.session.marks.length === 1 ? '' : 's'} marked present will be discarded. Use <b>End &amp; report</b> to save instead.`))
      return;
  }
  stopKiosk();
  go('class', S.cls.id);
};
$$('[data-kstab]').forEach(b => b.onclick = () => {
  S.kioskTab = b.dataset.kstab;
  $$('[data-kstab]').forEach(x => x.classList.toggle('is-on', x === b));
  paintSide();
});

window.addEventListener('beforeunload', e => {
  if (S.session?.marks.length) { e.preventDefault(); e.returnValue = ''; }
});
document.addEventListener('visibilitychange', () => {
  // browsers throttle timers in background tabs; surface it rather than silently missing people
  if (document.hidden && S.stream && !S.paused) {
    S.paused = true;
    $('#btnPause').textContent = 'Resume';
    hud('PAUSED', 'tab was hidden — press Resume');
  }
});

(async function boot() {
  const bar = $('#bootBar'), msg = $('#bootMsg');
  const step = (t, p) => { msg.textContent = t; bar.style.width = p + '%'; };
  try {
    step('opening local database…', 8);
    await DB.open();
    if (typeof faceapi === 'undefined') throw new Error('face-api library failed to load');
    await Engine.load(step);

    if (!navigator.mediaDevices?.getUserMedia) {
      $('#storeBadge').textContent = 'no camera api';
      $('#storeBadge').className = 'badge badge-warn';
    } else {
      $('#storeBadge').textContent = 'local · ready';
      $('#storeBadge').className = 'badge badge-ok';
    }
    if (!window.isSecureContext) {
      $('#storeBadge').textContent = 'insecure ctx';
      $('#storeBadge').className = 'badge badge-warn';
      $('#storeBadge').title = 'Camera needs https:// or localhost';
    }

    await new Promise(r => setTimeout(r, 260));
    $('#boot').classList.add('gone');
    setTimeout(() => { $('#boot').hidden = true; }, 520);
    $('#shell').hidden = false;
    $('#engineTxt').textContent = `engine ready · ${Engine.backend}`;
    if (Engine.backend === 'cpu') {
      toast('Running on CPU — recognition will be slower. A device with WebGL is much faster.', 'warn', 6000);
    }
    await go('classes');
  } catch (e) {
    console.error(e);
    msg.className = 'boot-msg err';
    msg.innerHTML = `startup failed — ${esc(e.message)}<br><span style="color:var(--ink-3)">try reloading the page</span>`;
    $('#engineDot').className = 'dot off';
  }
})();
